"""Evaluation module for assessing article relevance using Gemma3 via Ollama."""

import json
import re
import sqlite3
import time
import requests
from datetime import datetime, timedelta, timezone


def evaluate_with_gemma3(title, content, url, ollama_base_url='http://localhost:11434', model='gemma3', timeout=180, max_retries=2):
    """Evaluate an article using Gemma3 via Ollama API.
    
    Args:
        title: Article title
        content: Article content
        url: Article URL
        ollama_base_url: Ollama API base URL
        model: Model name to use
        timeout: Request timeout in seconds (default: 180 for large models)
        max_retries: Maximum number of retry attempts (default: 2)
    
    Returns a dict with: relevance_score, decision, topic, why_it_matters, target_audience
    """
    # Prepare the prompt for GENAI_NEWS evaluation
    prompt = f"""You are a technical news evaluator for AI engineers. 
        Evaluate the following Hacker News article for relevance to AI/LLM/Programming news.

Title: {title}
URL: {url}
Content: {content[:2000] if content else 'No content available'}

Provide a JSON response with the following fields:
- relevance_score: float between 0.0 and 1.0 (how relevant is this to AI/LLM/Programming?)
- decision: "KEEP" or "DROP" (should this be included in the digest?)
- topic: brief topic/category (e.g., "LLM Research", "AI Tools", "Programming Languages")
- why_it_matters: 1-2 sentence explanation of why this matters
- target_audience: one of "developer", "software architect", or "manager"

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
        return {
            'relevance_score': float(evaluation.get('relevance_score', 0.0)),
            'decision': evaluation.get('decision', 'DROP').upper(),
            'topic': evaluation.get('topic', 'Unknown'),
            'why_it_matters': evaluation.get('why_it_matters', ''),
            'target_audience': evaluation.get('target_audience', 'developer'),
            'llm_model': model
        }
        
    except Exception as e:
        print(f"Error evaluating with Gemma3: {e}")
        # Return default values on error
        return {
            'relevance_score': 0.0,
            'decision': 'DROP',
            'topic': 'Error',
            'why_it_matters': f'Evaluation failed: {str(e)}',
            'target_audience': 'developer',
            'llm_model': model
        }


def get_items_for_evaluation(db_path='mydb.db', hours=24):
    """Get items that need evaluation.
    
    Criteria:
    - Published within last 24 hours (or specified hours)
    - Status is 'INGESTED' or 'PREFILTERED' (passed keyword/time filters)
    - Not already evaluated (no entry in evaluations table for GENAI_NEWS persona)
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        
        # Calculate cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Query items that meet criteria
        query = """
        SELECT i.id, i.title, i.content, i.url, i.published_at, i.source, i.status
        FROM items i
        LEFT JOIN evaluations e ON i.id = e.item_id AND e.persona = 'GENAI_NEWS'
        WHERE (i.published_at IS NULL OR datetime(i.published_at) >= datetime(?))
          AND i.status IN ('INGESTED', 'PREFILTERED')
          AND e.id IS NULL
        ORDER BY i.published_at DESC NULLS LAST, i.ingestion_time DESC
        """
        
        cur.execute(query, (cutoff_str,))
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
                'status': row[6]
            })
        
        return items
    finally:
        conn.close()


def save_evaluation(item_id, evaluation_result, persona='GENAI_NEWS', db_path='mydb.db'):
    """Save evaluation result to the evaluations table."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        
        cur.execute("""
            INSERT OR REPLACE INTO evaluations (
                item_id, persona, decision, relevance_score,
                topic, why_it_matters, target_audience, llm_model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item_id,
            persona,
            evaluation_result['decision'],
            evaluation_result['relevance_score'],
            evaluation_result['topic'],
            evaluation_result['why_it_matters'],
            evaluation_result['target_audience'],
            evaluation_result['llm_model']
        ))
        
        # Update item status to EVALUATED
        cur.execute("UPDATE items SET status = 'EVALUATED' WHERE id = ?", (item_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error saving evaluation for item {item_id}: {e}")
        return False
    finally:
        conn.close()


def run_evaluation_pipeline(db_path='mydb.db', ollama_base_url='http://localhost:11434', 
                           model='gemma3:12b', hours=24, verbose=True, timeout=180):
    """Run the evaluation pipeline on items that need evaluation.
    
    Filters items by:
    - Published in last 24 hours (or specified hours)
    - Status is INGESTED or PREFILTERED
    - Not already evaluated
    
    Evaluates each item with Gemma3 and saves results to evaluations table.
    """
    print("=" * 60)
    print("EVALUATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Get items that need evaluation
    print(f"\nStep 1: Fetching items for evaluation (published in last {hours} hours)...")
    items = get_items_for_evaluation(db_path, hours)
    
    if not items:
        print("[OK] No items found that need evaluation.")
        return 0
    
    print(f"[OK] Found {len(items)} items to evaluate")
    
    # Step 2: Evaluate each item
    print(f"\nStep 2: Evaluating items with Gemma3 ({model})...")
    evaluated_count = 0
    error_count = 0
    
    for i, item in enumerate(items, 1):
        if verbose:
            print(f"[{i}/{len(items)}] Evaluating: {item['title'][:60]}...")
        
        try:
            # Evaluate with Gemma3
            evaluation = evaluate_with_gemma3(
                title=item['title'],
                content=item['content'] or '',
                url=item['url'] or '',
                ollama_base_url=ollama_base_url,
                model=model,
                timeout=timeout
            )
            
            # Save evaluation
            if save_evaluation(item['id'], evaluation, persona='GENAI_NEWS', db_path=db_path):
                evaluated_count += 1
                if verbose:
                    decision = evaluation['decision']
                    score = evaluation['relevance_score']
                    print(f"    [{decision}] Relevance: {score:.2f} | Topic: {evaluation['topic']}")
            else:
                error_count += 1
                if verbose:
                    print(f"    [ERROR] Failed to save evaluation")
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
            
        except Exception as e:
            error_count += 1
            if verbose:
                print(f"    [ERROR] {str(e)}")
    
    print(f"\n[OK] Evaluation complete!")
    print(f"[OK] Evaluated: {evaluated_count} items")
    if error_count > 0:
        print(f"[WARNING] Errors: {error_count} items")
    
    return evaluated_count


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluation Pipeline: Evaluate articles using Gemma3 via Ollama')
    parser.add_argument('--db', type=str, default='mydb.db', help='Database file path')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--model', type=str, default='gemma3:12b', help='Ollama model name (default: gemma3:12b)')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back for recent items (default: 24)')
    parser.add_argument('--timeout', type=int, default=180, help='Request timeout in seconds (default: 180)')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output')
    
    args = parser.parse_args()
    
    run_evaluation_pipeline(
        db_path=args.db,
        ollama_base_url=args.ollama_url,
        model=args.model,
        hours=args.hours,
        verbose=not args.quiet,
        timeout=args.timeout
    )
