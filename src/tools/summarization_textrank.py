"""
Summarization using TextRank: uses existing items.summary (from TextRank) plus
evaluations (topic, why_it_matters, target_audience). Formats output as news-article-style
digest entries (headline, lead, why it matters, audience) for daily digest. No LLM.
"""

import os
from pathlib import Path
import re
import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()
DB_PATH=Path(os.getenv("DB_PATH"))
SUMMARY_SOURCE = "TEXTRANK"


def _ensure_item_summaries_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS item_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL,
            technical_summary TEXT NOT NULL,
            why_it_matters TEXT,
            target_audience TEXT,
            source TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (item_id) REFERENCES items(id),
            UNIQUE(item_id, source)
        )
    """)
    # Optional digest columns (news-article style): headline, topic, url
    for col, ctype in [("title", "TEXT"), ("topic", "TEXT"), ("url", "TEXT")]:
        try:
            cur.execute(f"ALTER TABLE item_summaries ADD COLUMN {col} {ctype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


def get_items_with_textrank_and_evaluation(db_path, persona="GENAI_NEWS", use_canonical_only=False):
    """
    Get items that have non-empty items.summary (TextRank) and an evaluation (KEEP).
    Returns list of dicts: id, title, summary, url, topic, why_it_matters, target_audience.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        canonical_ids = None
        if use_canonical_only:
            cur.execute("SELECT canonical_item_id FROM dedup_clusters")
            canonical_ids = [r[0] for r in cur.fetchall()]
        if canonical_ids:
            placeholders = ",".join("?" * len(canonical_ids))
            cur.execute("""
                SELECT i.id, i.title, i.summary, i.url, e.topic, e.why_it_matters, e.target_audience
                FROM items i
                INNER JOIN evaluations e ON i.id = e.item_id AND e.persona = ?
                WHERE e.decision = 'KEEP'
                  AND TRIM(COALESCE(i.summary, '')) != ''
                  AND i.id IN (""" + placeholders + """
                )
                ORDER BY i.id
            """, [persona] + list(canonical_ids))
        else:
            cur.execute("""
                SELECT i.id, i.title, i.summary, i.url, e.topic, e.why_it_matters, e.target_audience
                FROM items i
                INNER JOIN evaluations e ON i.id = e.item_id AND e.persona = ?
                WHERE e.decision = 'KEEP'
                  AND TRIM(COALESCE(i.summary, '')) != ''
                ORDER BY i.id
            """, (persona,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _format_news_lead(text, max_sentences=5):
    """
    Format raw TextRank summary as a news-article-style lead paragraph:
    - Split into sentences (by . ! ?)
    - Take up to max_sentences, ensure each ends with punctuation
    - Normalize spacing and capitalize first letter
    - Single clean paragraph suitable for daily digest
    """
    if not (text or "").strip():
        return ""
    s = re.sub(r"\s+", " ", text.strip())
    # Split on sentence boundaries but keep the delimiter
    parts = re.split(r"(?<=[.!?])\s+", s)
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not re.search(r"[.!?]$", p):
            p = p + "."
        sentences.append(p)
        if len(sentences) >= max_sentences:
            break
    if not sentences:
        return s[:800] if len(s) > 800 else s
    lead = " ".join(sentences)
    if lead and lead[0].islower():
        lead = lead[0].upper() + lead[1:]
    return lead


def _normalize_audience(audience):
    a = (audience or "").strip().lower()
    if "architect" in a:
        return "software architect"
    if "manager" in a:
        return "manager"
    return "developer"


def save_item_summary(
    conn,
    item_id,
    technical_summary,
    why_it_matters,
    target_audience,
    source=SUMMARY_SOURCE,
    title=None,
    topic=None,
    url=None,
):
    cur = conn.cursor()
    # Support with or without digest columns (title, topic, url)
    try:
        cur.execute("""
            INSERT OR REPLACE INTO item_summaries
            (item_id, technical_summary, why_it_matters, target_audience, source, created_at, title, topic, url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item_id,
            technical_summary,
            why_it_matters or "",
            target_audience or "developer",
            source,
            datetime.now(timezone.utc).isoformat(),
            title or "",
            topic or "",
            url or "",
        ))
    except sqlite3.OperationalError:
        cur.execute("""
            INSERT OR REPLACE INTO item_summaries (item_id, technical_summary, why_it_matters, target_audience, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (item_id, technical_summary, why_it_matters or "", target_audience or "developer", source, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    return True


def run_summarization_textrank_pipeline(
    db_path,
    persona="GENAI_NEWS",
    use_canonical_only=False,
    max_summary_sentences=5,
    verbose=True,
):
    """
    For each KEEP item that has items.summary (TextRank) and evaluation:
    use summary as technical_summary (trimmed to ~3-5 sentences), and
    why_it_matters + target_audience from evaluations. Save to item_summaries (source=TEXTRANK).
    """
    print("=" * 60)
    print("SUMMARIZATION (TextRank summary + evaluation)")
    print("=" * 60)

    conn = sqlite3.connect(db_path)
    _ensure_item_summaries_table(conn)
    conn.close()

    items = get_items_with_textrank_and_evaluation(db_path, persona=persona, use_canonical_only=use_canonical_only)
    if not items:
        if verbose:
            print("[OK] No items with TextRank summary and KEEP evaluation; run evaluation_textrank.py first.")
        return 0

    if verbose:
        print(f"\nStep 1: Loaded {len(items)} items with items.summary (TextRank) + KEEP evaluation.")
        print("Step 2: Formatting as news-article-style digest (headline, lead, why it matters, audience)...")

    conn = sqlite3.connect(db_path)
    saved = 0
    for i, item in enumerate(items, 1):
        if verbose:
            print(f"  [{i}/{len(items)}] {item['title'][:55]}...")
        # News-style lead paragraph (3–5 sentences, clean punctuation)
        technical_summary = _format_news_lead(item.get("summary") or "", max_sentences=max_summary_sentences)
        why_it_matters = (item.get("why_it_matters") or "").strip()
        target_audience = _normalize_audience(item.get("target_audience"))
        title = (item.get("title") or "").strip()
        topic = (item.get("topic") or "").strip()
        url = (item.get("url") or "").strip()
        save_item_summary(
            conn,
            item["id"],
            technical_summary,
            why_it_matters,
            target_audience,
            source=SUMMARY_SOURCE,
            title=title,
            topic=topic,
            url=url,
        )
        saved += 1
        if verbose:
            print(f"       -> {target_audience} | {topic or '—'} | {technical_summary[:60]}...")

    conn.close()
    if verbose:
        print(f"\n[OK] Summarization complete. Saved: {saved} (source=TEXTRANK, news-article style)")
    return saved


def get_digest_entries(db_path):
    """
    Load digest entries directly from items + evaluations.
    Returns list of dicts with keys:
    headline, lead, why_it_matters, target_audience, topic, url,
    item_id, likes, comments, digest_type
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                i.id AS item_id,
                i.title AS headline,
                i.summary AS lead,
                e.why_it_matters,
                e.target_audience,
                e.topic,
                i.url,
                COALESCE(i.likes, 0) AS likes,
                COALESCE(i.comments, 0) AS comments,
                i.digest_type AS digest_type
            FROM items i
            JOIN evaluations e
              ON e.item_id = i.id
            WHERE e.decision = 'KEEP'
            ORDER BY
                i.published_at DESC,
                i.ingestion_time DESC
            """
        )

        rows = cur.fetchall()
        return [
            {
                "item_id": r["item_id"],
                "headline": r["headline"] or "(No title)",
                "lead": r["lead"] or "",
                "why_it_matters": r["why_it_matters"] or "",
                "target_audience": r["target_audience"] or "developer",
                "topic": r["topic"] or "",
                "url": r["url"] or "",
                "likes": r["likes"],
                "comments": r["comments"],
                "digest_type": r["digest_type"] or "GENAI",
            }
            for r in rows
        ]

    finally:
        conn.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Summarize using TextRank summary + evaluation (why it matters, audience)")
    p.add_argument("--db", default=DB_PATH, help="Database path")
    p.add_argument("--canonical-only", action="store_true", help="Only canonical (dedup) items")
    p.add_argument("--max-sentences", type=int, default=5, help="Max sentences in technical summary (default 5)")
    p.add_argument("--deliver", action="store_true", help="Run delivery (email/Telegram/file) after summarization")
    p.add_argument("--persona", default="GENAI_NEWS", help="Persona for delivery config (with --deliver)")
    p.add_argument("--output-dir", default="output", help="Delivery fallback output dir (with --deliver)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run_summarization_textrank_pipeline(
        db_path=args.db,
        use_canonical_only=args.canonical_only,
        max_summary_sentences=args.max_sentences,
        verbose=not args.quiet,
    )
    if args.deliver:
        from delivery import run_delivery
        run_delivery(
            persona=args.persona,
            db_path=args.db,
            source=SUMMARY_SOURCE,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
