"""TF-IDF summary + LLM evaluation: build extractive summary, store in items.summary, evaluate with same schema as evaluation.py."""

import re
import sqlite3

from sklearn.feature_extraction.text import TfidfVectorizer

from evaluation import (
    get_items_for_evaluation,
    evaluate_with_gemma3,
    save_evaluation,
)


def _split_sentences(text):
    """Split text into sentences (regex-based, no NLTK)."""
    if not text or not text.strip():
        return []
    # Split on . ! ? followed by space or end, keep separators for consistency
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def summarize_by_tfidf(text, top_n=5):
    """
    Build extractive summary: split into sentences, rank by TF-IDF score, pick top N.
    Treats each sentence as a document; scores by sum of TF-IDF weights in that sentence.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return ''
    if len(sentences) <= top_n:
        return ' '.join(sentences)
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        # Rows = sentences, cols = terms
        X = vectorizer.fit_transform(sentences)
        # Score each sentence by sum of its TF-IDF vector (importance in doc)
        scores = X.sum(axis=1).A1
        if scores.size == 0:
            return ' '.join(sentences[:top_n])
        top_indices = scores.argsort()[-top_n:][::-1]
        # Preserve original order of sentences for readability
        top_indices_sorted = sorted(top_indices)
        return ' '.join(sentences[i] for i in top_indices_sorted)
    except Exception:
        return ' '.join(sentences[:top_n])


def update_item_summary(item_id, summary, db_path='mydb.db'):
    """Write TF-IDF summary to items.summary for the given item."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("UPDATE items SET summary = ? WHERE id = ?", (summary or '', item_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error updating summary for item {item_id}: {e}")
        return False
    finally:
        conn.close()


def _process_one_item(item, ollama_base_url, model, timeout, db_path, top_n_sentences):
    """
    For one item: build TF-IDF summary, update items.summary, evaluate using summary, return (item_id, evaluation, error).
    """
    try:
        content = item.get('content') or ''
        summary = summarize_by_tfidf(content, top_n=top_n_sentences)
        if not summary.strip():
            summary = (item.get('title') or '')[:500]
        update_item_summary(item['id'], summary, db_path=db_path)
        # Evaluate using summary (same prompt/output as evaluation.py)
        evaluation = evaluate_with_gemma3(
            title=item['title'],
            content=summary,  # pass summary instead of full content
            url=item['url'] or '',
            ollama_base_url=ollama_base_url,
            model=model,
            timeout=timeout,
        )
        return (item['id'], evaluation, None)
    except Exception as e:
        return (item['id'], None, e)


def run_evaluation_tfidf_pipeline(
    db_path='mydb.db',
    ollama_base_url='http://localhost:11434',
    model='gemma3:12b',
    hours=24,
    verbose=True,
    timeout=180,
    top_n_sentences=5,
):
    """
    Run TF-IDF summary + evaluation pipeline (one item at a time):

    1. Fetch items that need evaluation (same criteria as evaluation.py).
    2. For each item: split into sentences, rank by TF-IDF, take top N → summary.
    3. Save summary to items.summary.
    4. Evaluate with Gemma using summary (same output: relevance_score, topic, why_it_matters, target_audience, decision).
    5. Save to evaluations table.
    """
    print("=" * 60)
    print("EVALUATION PIPELINE (TF-IDF summary)")
    print("=" * 60)

    print(f"\nStep 1: Fetching items for evaluation (published in last {hours} hours)...")
    items = get_items_for_evaluation(db_path, hours)

    if not items:
        print("[OK] No items found that need evaluation.")
        return 0

    print(f"[OK] Found {len(items)} items (summarize with top {top_n_sentences} sentences, then evaluate with {model})")

    print("\nStep 2: Building TF-IDF summaries and evaluating (one at a time)...")
    evaluated_count = 0
    error_count = 0

    for i, item in enumerate(items, 1):
        if verbose:
            print(f"[{i}/{len(items)}] {item['title'][:60]}...")
        try:
            item_id, evaluation, err = _process_one_item(
                item, ollama_base_url, model, timeout, db_path, top_n_sentences
            )
            if err:
                error_count += 1
                if verbose:
                    print(f"    [ERROR] {err}")
                continue
            if save_evaluation(item_id, evaluation, persona='GENAI_NEWS', db_path=db_path, evaluation_type='TFIDF'):
                evaluated_count += 1
                if verbose:
                    decision = evaluation['decision']
                    score = evaluation['relevance_score']
                    print(f"    [{decision}] Relevance: {score:.2f} | Topic: {evaluation['topic']}")
            else:
                error_count += 1
                if verbose:
                    print(f"    [ERROR] Failed to save evaluation for item {item_id}")
        except Exception as e:
            error_count += 1
            if verbose:
                print(f"    [ERROR] {e}")

    print(f"\n[OK] Evaluation complete!")
    print(f"[OK] Evaluated: {evaluated_count} items (summaries saved to items.summary)")
    if error_count > 0:
        print(f"[WARNING] Errors: {error_count} items")

    return evaluated_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluation with TF-IDF summary: summarize articles, then evaluate with Gemma (same output as evaluation.py)'
    )
    parser.add_argument('--db', type=str, default='mydb.db', help='Database file path')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--model', type=str, default='gemma3:12b', help='Ollama model name')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back (default: 24). Use 0 for no time limit.')
    parser.add_argument('--timeout', type=int, default=180, help='Request timeout in seconds')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top TF-IDF sentences for summary (default: 5)')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output')

    args = parser.parse_args()

    run_evaluation_tfidf_pipeline(
        db_path=args.db,
        ollama_base_url=args.ollama_url,
        model=args.model,
        hours=args.hours,
        verbose=not args.quiet,
        timeout=args.timeout,
        top_n_sentences=args.top_n,
    )
