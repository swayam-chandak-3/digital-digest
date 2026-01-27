"""
Summarization via LLM: 3-5 line technical summaries, "why it matters", audience targeting.
Reads content from items table; uses Ollama to generate technical_summary, why_it_matters, target_audience.
Writes to item_summaries table with source='LLM'.
"""

import json
import sqlite3
import time
import requests
from datetime import datetime, timezone

SUMMARY_SOURCE = "LLM"


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
    conn.commit()


def get_items_for_summarization(db_path="mydb.db", persona="GENAI_NEWS", use_canonical_only=False):
    """
    Get items that need summarization: evaluated with decision=KEEP.
    If use_canonical_only and dedup has run, only returns canonical items from dedup_clusters.
    Returns list of dicts: id, title, content, summary, url.
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
                SELECT i.id, i.title, i.content, i.summary, i.url
                FROM items i
                INNER JOIN evaluations e ON i.id = e.item_id AND e.persona = ?
                WHERE e.decision = 'KEEP' AND i.id IN (""" + placeholders + """
                )
                ORDER BY i.id
            """, [persona] + list(canonical_ids))
        else:
            cur.execute("""
                SELECT i.id, i.title, i.content, i.summary, i.url
                FROM items i
                INNER JOIN evaluations e ON i.id = e.item_id AND e.persona = ?
                WHERE e.decision = 'KEEP'
                ORDER BY i.id
            """, (persona,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def summarize_with_llm(
    title,
    content,
    url="",
    ollama_base_url="http://localhost:11434",
    model="gemma3:12b",
    timeout=120,
    max_retries=2,
):
    """
    Call Ollama to produce:
    - technical_summary: 3-5 line technical summary
    - why_it_matters: short explanation
    - target_audience: developer | software architect | manager
    Returns dict with those keys, or None on failure.
    """
    text = (content or "").strip()[:4000] or (title or "")
    prompt = f"""You are a technical summarizer for a developer digest.
Given the following article, provide a concise technical summary and context.

Title: {title}
URL: {url}
Content:
{text}

Respond with ONLY a JSON object (no markdown, no extra text) with these exact keys:
- technical_summary: string, 3 to 5 short sentences summarizing the key technical points.
- why_it_matters: string, 1-2 sentences on why this matters for practitioners.
- target_audience: one of "developer", "software architect", or "manager"
"""

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            break
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep((attempt + 1) * 5)
            else:
                print(f"LLM summarization request failed: {e}")
                return None

    try:
        result = resp.json()
        raw = (result.get("response") or "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        obj = json.loads(raw)
        return {
            "technical_summary": (obj.get("technical_summary") or "").strip() or title[:500],
            "why_it_matters": (obj.get("why_it_matters") or "").strip(),
            "target_audience": (obj.get("target_audience") or "developer").strip().lower(),
        }
    except (json.JSONDecodeError, Exception) as e:
        print(f"LLM summary parse error: {e}")
        return None


def save_item_summary(conn, item_id, technical_summary, why_it_matters, target_audience, source=SUMMARY_SOURCE):
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO item_summaries (item_id, technical_summary, why_it_matters, target_audience, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (item_id, technical_summary, why_it_matters or "", target_audience or "developer", source, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    return True


def run_summarization_llm_pipeline(
    db_path="mydb.db",
    ollama_base_url="http://localhost:11434",
    model="gemma3:12b",
    persona="GENAI_NEWS",
    use_canonical_only=False,
    timeout=120,
    verbose=True,
):
    """
    For each KEEP item (from items + evaluations), generate LLM summary and save to item_summaries (source=LLM).
    """
    print("=" * 60)
    print("SUMMARIZATION (LLM)")
    print("=" * 60)

    conn = sqlite3.connect(db_path)
    _ensure_item_summaries_table(conn)
    conn.close()

    items = get_items_for_summarization(db_path, persona=persona, use_canonical_only=use_canonical_only)
    if not items:
        if verbose:
            print("[OK] No KEEP items to summarize.")
        return 0

    if verbose:
        print(f"\nStep 1: Loaded {len(items)} items (decision=KEEP) from items + evaluations.")
        print(f"Step 2: Generating 3-5 line technical summaries with {model}...")

    conn = sqlite3.connect(db_path)
    saved = 0
    errors = 0
    for i, item in enumerate(items, 1):
        if verbose:
            print(f"  [{i}/{len(items)}] {item['title'][:55]}...")
        content = item.get("content") or item.get("summary") or item.get("title") or ""
        out = summarize_with_llm(
            title=item.get("title") or "",
            content=content,
            url=item.get("url") or "",
            ollama_base_url=ollama_base_url,
            model=model,
            timeout=timeout,
        )
        if out:
            save_item_summary(
                conn,
                item["id"],
                out["technical_summary"],
                out["why_it_matters"],
                out["target_audience"],
                source=SUMMARY_SOURCE,
            )
            saved += 1
            if verbose:
                print(f"       -> {out['target_audience']} | {out['technical_summary'][:60]}...")
        else:
            errors += 1

    conn.close()
    if verbose:
        print(f"\n[OK] Summarization complete. Saved: {saved} | Errors: {errors}")
    return saved


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Summarize items with LLM (3-5 line technical summary + why it matters + audience)")
    p.add_argument("--db", default="mydb.db", help="Database path")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("--model", default="gemma3:12b", help="Ollama model")
    p.add_argument("--canonical-only", action="store_true", help="Only summarize canonical (dedup) items")
    p.add_argument("--timeout", type=int, default=120, help="Request timeout")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run_summarization_llm_pipeline(
        db_path=args.db,
        ollama_base_url=args.ollama_url,
        model=args.model,
        use_canonical_only=args.canonical_only,
        timeout=args.timeout,
        verbose=not args.quiet,
    )
