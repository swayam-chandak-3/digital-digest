"""TextRank summary + LLM evaluation: graph-based summarization (sumy), store in items.summary, evaluate with same schema as evaluation.py."""

import sqlite3

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from evaluation import (
    get_items_for_evaluation,
    evaluate_with_gemma3,
    save_evaluation,
)


def summarize_by_textrank(text, top_n=5, language='english'):
    """
    Build extractive summary using TextRank (graph-based sentence ranking).
    Uses sumy's TextRankSummarizer.
    """
    if not text or not text.strip():
        return ''
    text = text.strip()
    try:
        tokenizer = Tokenizer(language)
        parser = PlaintextParser.from_string(text, tokenizer)
        summarizer = TextRankSummarizer()
        summary_sentences = summarizer(parser.document, top_n)
        if not summary_sentences:
            return text[:2000] if len(text) > 2000 else text
        return ' '.join(str(s) for s in summary_sentences)
    except Exception:
        # Fallback: first N sentences by simple split
        parts = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        return ' '.join(parts[:top_n]) if parts else text[:2000]


def update_item_summary(item_id, summary, db_path='mydb.db'):
    """Write TextRank summary to items.summary for the given item."""
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
    For one item: build TextRank summary, update items.summary, evaluate using summary, return (item_id, evaluation, error).
    """
    try:
        content = item.get('content') or ''
        summary = summarize_by_textrank(content, top_n=top_n_sentences)
        if not summary.strip():
            summary = (item.get('title') or '')[:500]
        update_item_summary(item['id'], summary, db_path=db_path)
        evaluation = evaluate_with_gemma3(
            title=item['title'],
            content=summary,
            url=item['url'] or '',
            ollama_base_url=ollama_base_url,
            model=model,
            timeout=timeout,
        )
        return (item['id'], evaluation, None)
    except Exception as e:
        return (item['id'], None, e)


def run_evaluation_textrank_pipeline(
    db_path='mydb.db',
    ollama_base_url='http://localhost:11434',
    model='gemma3:12b',
    hours=24,
    verbose=True,
    timeout=180,
    top_n_sentences=5,
):
    """
    Run TextRank summary + evaluation pipeline (one item at a time):

    1. Fetch items that need evaluation (same criteria as evaluation.py).
    2. For each item: TextRank (graph-based) summary → top N sentences.
    3. Save summary to items.summary.
    4. Evaluate with Gemma using summary (same output: relevance_score, topic, why_it_matters, target_audience, decision).
    5. Save to evaluations table.
    """
    print("=" * 60)
    print("EVALUATION PIPELINE (TextRank summary)")
    print("=" * 60)

    print(f"\nStep 1: Fetching items for evaluation (published in last {hours} hours)...")
    items = get_items_for_evaluation(db_path, hours)

    if not items:
        print("[OK] No items found that need evaluation.")
        return 0

    print(f"[OK] Found {len(items)} items (summarize with TextRank top {top_n_sentences} sentences, then evaluate with {model})")

    print("\nStep 2: Building TextRank summaries and evaluating (one at a time)...")
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
            # Save to evaluations table (mydb.db by default) with evaluation_type='TEXTRANK'
            if save_evaluation(item_id, evaluation, persona='GENAI_NEWS', db_path=db_path, evaluation_type='TEXTRANK'):
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
        description='Evaluation with TextRank summary: graph-based summarization (sumy), then evaluate with Gemma'
    )
    parser.add_argument('--db', type=str, default='mydb.db', help='Database file path')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--model', type=str, default='gemma3:12b', help='Ollama model name')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back (default: 24). Use 0 for no time limit.')
    parser.add_argument('--timeout', type=int, default=180, help='Request timeout in seconds')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top TextRank sentences for summary (default: 5)')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output')

    args = parser.parse_args()

    run_evaluation_textrank_pipeline(
        db_path=args.db,
        ollama_base_url=args.ollama_url,
        model=args.model,
        hours=args.hours,
        verbose=not args.quiet,
        timeout=args.timeout,
        top_n_sentences=args.top_n,
    )
