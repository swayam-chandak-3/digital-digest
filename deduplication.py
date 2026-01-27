"""
Deduplication: vector similarity (FAISS) + topic/content similarity.
Uses evaluations table (KEEP items) to build embeddings from title + topic + why_it_matters,
clusters by similarity, keeps a single canonical entry per unique topic/idea.
"""

import sqlite3
import numpy as np
from datetime import datetime, timezone


def _ensure_dedup_tables(conn):
    """Create dedup_clusters and dedup_item_cluster if they do not exist."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dedup_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_item_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (canonical_item_id) REFERENCES items(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dedup_item_cluster (
            item_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            is_canonical INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (item_id),
            FOREIGN KEY (item_id) REFERENCES items(id),
            FOREIGN KEY (cluster_id) REFERENCES dedup_clusters(id)
        )
    """)
    conn.commit()


def get_keep_items_for_dedup(db_path="mydb.db", persona="GENAI_NEWS"):
    """
    Get items that were evaluated and marked KEEP (candidates for deduplication).
    Returns list of dicts: id, title, content, summary, topic, why_it_matters, relevance_score.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                i.id,
                i.title,
                i.content,
                i.summary,
                e.topic,
                e.why_it_matters,
                e.relevance_score
            FROM items i
            INNER JOIN evaluations e ON i.id = e.item_id AND e.persona = ?
            WHERE e.decision = 'KEEP'
            ORDER BY e.relevance_score DESC, i.id
        """, (persona,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _text_for_embedding(item):
    """Build a single string from item for embedding (topic + content similarity)."""
    title = (item.get("title") or "").strip()
    topic = (item.get("topic") or "").strip()
    why = (item.get("why_it_matters") or "").strip()
    summary = (item.get("summary") or "").strip()
    content = (item.get("content") or "").strip()
    # Prefer semantic fields (topic, why_it_matters) and title; add summary/content head for extra signal
    parts = [title, topic, why]
    if summary:
        parts.append(summary[:1500])
    elif content:
        parts.append(content[:1500])
    return " ".join(p for p in parts if p).strip() or title or "unknown"


def run_deduplication(
    db_path="mydb.db",
    persona="GENAI_NEWS",
    similarity_threshold=0.85,
    embedding_model="all-MiniLM-L6-v2",
    verbose=True,
):
    """
    Run deduplication pipeline:

    1. Load KEEP items from evaluations (joined with items).
    2. Embed each item (title + topic + why_it_matters + summary/content) with sentence-transformers.
    3. Build FAISS index (cosine similarity via normalized vectors + IndexFlatIP).
    4. Cluster items above similarity_threshold (union-find).
    5. Pick one canonical item per cluster (highest relevance_score).
    6. Persist: dedup_clusters, dedup_item_cluster; set items.status = 'DEDUPED' for all involved.

    Returns (n_candidates, n_clusters, n_deduped).
    """
    try:
        from sentence_transformers import SentenceTransformer  # package: sentence-transformers
        import faiss
    except ImportError as e:
        raise ImportError(
            "Deduplication requires: pip install faiss-cpu sentence-transformers"
        ) from e

    if verbose:
        print("=" * 60)
        print("DEDUPLICATION (FAISS + topic/content similarity)")
        print("=" * 60)

    conn = sqlite3.connect(db_path)
    _ensure_dedup_tables(conn)
    conn.close()

    items = get_keep_items_for_dedup(db_path, persona=persona)
    if not items:
        if verbose:
            print("[OK] No KEEP items from evaluations; nothing to deduplicate.")
        return 0, 0, 0

    n_candidates = len(items)
    if verbose:
        print(f"\nStep 1: Loaded {n_candidates} KEEP items from evaluations.")

    # Build texts and embed
    texts = [_text_for_embedding(it) for it in items]
    if verbose:
        print(f"Step 2: Embedding with '{embedding_model}'...")
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # For each vector, search k nearest (including self); use threshold to form edges
    k = min(n_candidates, max(2, n_candidates))
    scores, indices = index.search(embeddings.astype(np.float32), k)

    # Union-find to build clusters
    parent = list(range(n_candidates))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n_candidates):
        for j_idx, idx in enumerate(indices[i]):
            if idx < 0 or idx == i:
                continue
            # scores from IndexFlatIP on normalized vectors = cosine similarity
            if scores[i][j_idx] >= similarity_threshold:
                union(i, idx)

    # Cluster id -> list of item indices
    clusters = {}
    for i in range(n_candidates):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Pick canonical per cluster (max relevance_score); wrap cluster indices with item_id
    item_ids = [it["id"] for it in items]
    relevance = [it["relevance_score"] or 0.0 for it in items]

    cluster_to_canonical_idx = {}
    for root, indices_in_cluster in clusters.items():
        best_idx = max(indices_in_cluster, key=lambda i: (relevance[i], -item_ids[i]))
        cluster_to_canonical_idx[root] = best_idx

    n_clusters = len(clusters)
    n_deduped = n_candidates - n_clusters  # number of items that are duplicates (not canonical)

    if verbose:
        print(f"Step 3: Formed {n_clusters} clusters (threshold={similarity_threshold}).")
        print(f"Step 4: Persisting clusters and updating item status to DEDUPED...")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM dedup_item_cluster")
        cur.execute("DELETE FROM dedup_clusters")
        cluster_id_by_root = {}

        for root, indices_in_cluster in clusters.items():
            canonical_idx = cluster_to_canonical_idx[root]
            canonical_item_id = item_ids[canonical_idx]
            cur.execute(
                "INSERT INTO dedup_clusters (canonical_item_id) VALUES (?)",
                (canonical_item_id,),
            )
            cluster_id = cur.lastrowid
            cluster_id_by_root[root] = cluster_id
            for idx in indices_in_cluster:
                item_id = item_ids[idx]
                is_canonical = 1 if idx == canonical_idx else 0
                cur.execute(
                    "INSERT INTO dedup_item_cluster (item_id, cluster_id, is_canonical) VALUES (?, ?, ?)",
                    (item_id, cluster_id, is_canonical),
                )

        # Mark all involved items as DEDUPED
        cur.execute(
            "UPDATE items SET status = 'DEDUPED' WHERE id IN (SELECT item_id FROM dedup_item_cluster)"
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()

    if verbose:
        print(f"\n[OK] Deduplication complete.")
        print(f"     Candidates: {n_candidates} | Unique topics (clusters): {n_clusters} | Duplicates merged: {n_deduped}")

    return n_candidates, n_clusters, n_deduped


def get_canonical_items(db_path="mydb.db"):
    """
    Return item_ids that are the single canonical entry per cluster (for digest).
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT canonical_item_id FROM dedup_clusters ORDER BY id
        """)
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate KEEP items from evaluations using FAISS + topic similarity"
    )
    parser.add_argument("--db", type=str, default="mydb.db", help="Database path")
    parser.add_argument("--persona", type=str, default="GENAI_NEWS", help="Evaluation persona")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold to merge (default: 0.85)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument("--quiet", action="store_true", help="Less output")

    args = parser.parse_args()
    run_deduplication(
        db_path=args.db,
        persona=args.persona,
        similarity_threshold=args.threshold,
        embedding_model=args.model,
        verbose=not args.quiet,
    )
