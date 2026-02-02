"""TextRank summary + LLM evaluation for PRODUCT news: graph-based summarization (sumy), store in items.summary, evaluate with product-focused schema."""

import sqlite3
import os
from pathlib import Path
import requests
import time
import json
import re
from dotenv import load_dotenv

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from src.tools.db_utils import save_evaluation

load_dotenv()

DB_PATH = Path(os.getenv('DB_PATH', 'evalution.db'))

PRODUCT_NEWS_MIN_REUSABILITY = float(os.getenv('PRODUCT_IDEAS_MIN_REUSABILITY', '0.5'))

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


def update_item_summary(item_id, summary, db_path=DB_PATH):
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
    For one item: build TextRank summary, update items.summary, evaluate using summary for PRODUCT news, return (item_id, evaluation, error).
    """
    try:
        content = item.get('content') or ''
        summary = summarize_by_textrank(content, top_n=top_n_sentences)
        if not summary.strip():
            summary = (item.get('title') or '')[:500]
        update_item_summary(item['id'], summary, db_path=db_path)
        evaluation = evaluate_product_with_llm(
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


def run_evaluation_textrank_product_pipeline(
    db_path=str(DB_PATH),
    ollama_base_url='http://localhost:11434',
    model='gemma3:12b',
    verbose=True,
    timeout=180,
    top_n_sentences=5,
):
    """
    Run TextRank summary + product evaluation pipeline (one item at a time):

    1. Fetch items that need evaluation from today's ingestion (digest_type='PRODUCT').
    2. For each item: TextRank (graph-based) summary → top N sentences.
    3. Save summary to items.summary.
    4. Evaluate with LLM using summary for PRODUCT news (output: idea_type, problem_statement, solution_summary, maturity_level, reusability_score, decision).
    5. Save to evaluations table.
    """
    print("=" * 60)
    print("PRODUCT EVALUATION PIPELINE (TextRank summary)")
    print("=" * 60)

    print(f"\nStep 1: Fetching PRODUCT items from today for evaluation...")
    items = get_items_for_evaluation(db_path)

    if not items:
        print("[OK] No items found that need evaluation.")
        return 0

    print(f"[OK] Found {len(items)} items (summarize with TextRank top {top_n_sentences} sentences, then evaluate with {model})")

    print("\nStep 2: Building TextRank summaries and evaluating for PRODUCT news (one at a time)...")
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
            
            score = float(evaluation.get('reusability_score', 0.0))
            evaluation['decision'] = 'KEEP' if score >= PRODUCT_NEWS_MIN_REUSABILITY else 'DROP'
            if verbose and evaluation['decision'] == 'DROP':
                print(f"    [DROP] Reusability {score:.2f} < {PRODUCT_NEWS_MIN_REUSABILITY}")
            
            # Save to evaluations table with evaluation_type='TEXTRANK_PRODUCT'
            if save_evaluation(item_id, evaluation, persona='PRODUCT_IDEAS', db_path=db_path, evaluation_type='TEXTRANK_PRODUCT'):
                evaluated_count += 1
                if verbose:
                    decision = evaluation['decision']
                    score = evaluation['reusability_score']
                    print(f"    [{decision}] Reusability: {score:.2f} | Idea Type: {evaluation['idea_type']}")
            else:
                error_count += 1
                if verbose:
                    print(f"    [ERROR] Failed to save evaluation for item {item_id}")
        except Exception as e:
            error_count += 1
            if verbose:
                print(f"    [ERROR] {e}")

    print(f"\n[OK] Product evaluation complete!")
    print(f"[OK] Evaluated: {evaluated_count} items (summaries saved to items.summary)")
    if error_count > 0:
        print(f"[WARNING] Errors: {error_count} items")

    return evaluated_count


def get_items_for_evaluation(db_path):
    """
    Get items that need evaluation.

    Criteria:
    - Status is 'INGESTED' or 'PREFILTERED'
    - Not already evaluated for PRODUCT_IDEAS persona
    - digest_type is 'PRODUCT' (from raw_metadata JSON)
    - ingestion_time is today's date (regardless of current time)
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        query = """
        SELECT i.id, i.title, i.content, i.url, i.published_at, i.source, i.status
        FROM items i
        LEFT JOIN evaluations e
          ON i.id = e.item_id
         AND e.persona = 'PRODUCT_IDEAS'
        WHERE i.status IN ('INGESTED', 'PREFILTERED')
          AND e.id IS NULL
          AND i.digest_type = 'PRODUCT'
          AND DATE(i.ingestion_time) = DATE('now')
        ORDER BY
          i.ingestion_time DESC,
          i.id DESC
        """

        cur.execute(query)
        rows = cur.fetchall()

        items = []
        for row in rows:
            items.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'url': row[3],
                'published_at': row[4],
                'source': row[5],
                'status': row[6],
            })

        return items
    finally:
        conn.close()


def evaluate_product_with_llm(title, content, url, ollama_base_url='http://localhost:11434', model='gemma3:12b', timeout=180, max_retries=2):
    """Evaluate an article for PRODUCT news using LLM via Ollama API.
    
    Args:
        title: Article title
        content: Article content (summary)
        url: Article URL
        ollama_base_url: Ollama API base URL
        model: Model name to use
        timeout: Request timeout in seconds (default: 180 for large models)
        max_retries: Maximum number of retry attempts (default: 2)
    
    Returns a dict with: idea_type, problem_statement, solution_summary, maturity_level, target_audience, topic, why_it_matters, reusability_score, decision
    """
    # Prepare the prompt for PRODUCT_IDEAS evaluation
    prompt = f"""You are evaluating Indie Hackers product stories for a product-ideas digest.
Evaluate the post and return JSON with:
idea_type, problem_statement, solution_summary, maturity_level, target_audience (developer|architect|manager),
topic, why_it_matters, reusability_score (0-1), decision (KEEP|DROP).

Scoring guidance:
- Higher scores when the post includes concrete product signals: MVP/launch details, tech stack,
  pricing, traction, user outcomes, or distribution experiments.
- Lower scores for generic founder stories without actionable product details.
Decision rule: KEEP if reusability_score >= {PRODUCT_NEWS_MIN_REUSABILITY}, else DROP.

Title: {title}
URL: {url}
Content: {content[:2000] if content else 'No content available'}

Provide a JSON response with the following fields:
- idea_type: brief category (e.g., "SaaS Tool", "Mobile App", "AI Product", "Marketplace")
- problem_statement: what problem does this solve (1-2 sentences)
- solution_summary: how does the product solve it (1-2 sentences)
- maturity_level: one of "concept", "mvp", "beta", "launched", "scaling"
- target_audience: primary audience persona (one of "developer", "architect", or "manager")
- topic: category or domain (e.g., "Creator Tools", "AI/ML", "E-commerce", "DevTools")
- why_it_matters: why this is relevant for builders (2-3 sentences)
- reusability_score: float between 0.0 and 1.0
- decision: "KEEP" or "DROP"

Respond ONLY with valid JSON, no additional text."""

    # Retry logic for handling timeouts
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # Call Ollama API
            response = requests.post(
                f"{ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for deterministic output
                    }
                },
                timeout=timeout
            )
            response.raise_for_status()
            break  # Success, exit retry loop
            
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries:
                wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s
                print(f"Timeout on attempt {attempt + 1}/{max_retries + 1}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries:
                wait_time = (attempt + 1) * 5
                print(f"Request error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    try:
        result = response.json()
        response_text = result.get('response', '').strip()
        
        # Try to extract JSON from the response
        # Sometimes LLMs wrap JSON in markdown code blocks or add extra text
        # Try to find JSON object boundaries
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            response_text = response_text[json_start:json_end+1]
        
        # Try parsing JSON
        try:
            evaluation = json.loads(response_text)
        except json.JSONDecodeError:
            # If parsing fails, try to extract with regex (fallback)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group(0))
            else:
                raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")
        
        # Validate and normalize the response
        target_audience = evaluation.get('target_audience', 'developer').lower()
        if target_audience not in ('developer', 'architect', 'manager'):
            target_audience = 'developer'  # Default fallback
        
        return {
            'idea_type': evaluation.get('idea_type', 'Unknown'),
            'problem_statement': evaluation.get('problem_statement', ''),
            'solution_summary': evaluation.get('solution_summary', ''),
            'maturity_level': evaluation.get('maturity_level', 'unknown'),
            'target_audience': target_audience,
            'topic': evaluation.get('topic', ''),
            'why_it_matters': evaluation.get('why_it_matters', ''),
            'reusability_score': float(evaluation.get('reusability_score', 0.0)),
            'decision': evaluation.get('decision', 'DROP').upper(),
            'llm_model': model
        }
        
    except Exception as e:
        print(f"Error evaluating product with LLM: {e}")
        # Return default values on error
        return {
            'idea_type': 'Error',
            'problem_statement': f'Evaluation failed: {str(e)}',
            'solution_summary': '',
            'maturity_level': 'unknown',
            'target_audience': 'developer',
            'topic': '',
            'why_it_matters': '',
            'reusability_score': 0.0,
            'decision': 'DROP',
            'llm_model': model
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Product Evaluation with TextRank summary: graph-based summarization (sumy), then evaluate for product insights'
    )
    parser.add_argument('--db', type=str, default=DB_PATH, help='Database file path')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--model', type=str, default='gemma3:12b', help='Ollama model name')
    parser.add_argument('--timeout', type=int, default=180, help='Request timeout in seconds')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top TextRank sentences for summary (default: 5)')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output')

    args = parser.parse_args()

    run_evaluation_textrank_product_pipeline(
        db_path=args.db,
        ollama_base_url=args.ollama_url,
        model=args.model,
        verbose=not args.quiet,
        timeout=args.timeout,
        top_n_sentences=args.top_n,
    )