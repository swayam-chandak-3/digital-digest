"""Evaluation module for assessing article relevance using LLM via Ollama."""

import json
import re
import sqlite3
import time
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone


GENAI_NEWS_MIN_RELEVANCE = float(os.getenv('GENAI_NEWS_MIN_RELEVANCE', '0.6'))

# Predefined topics for GENAI_NEWS classification
GENAI_TOPICS = [
    "llm_research",       # LLM Research and Papers
    "ai_tools",           # AI Tools and Applications
    "ml_infrastructure",  # ML Infrastructure and MLOps
    "prompt_engineering", # Prompt Engineering and Techniques
    "ai_ethics",          # AI Ethics, Safety, and Alignment
]


def save_item_topics(item_id, evaluation_result, db_path='mydb.db'):
    """Save LLM-assigned topics to the item_topics table.
    
    Stores topics as boolean (presence = assigned).
    
    Args:
        item_id: ID of the item in the items table
        evaluation_result: Dictionary from evaluate_with_llm() containing 'topics'
        db_path: Path to SQLite database
        
    Returns:
        Number of topics assigned to the item
    """
    topics = evaluation_result.get('topics', [])
    
    if not topics:
        return 0
    
    try:
        # Import here to avoid circular imports
        from src.services.topic_service import assign_topics_to_item
        return assign_topics_to_item(item_id, topics, None, db_path)
    except ImportError:
        # Fallback: direct database insert if service not available
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            assigned_count = 0
            for topic_name in topics:
                cur.execute("SELECT id FROM topics WHERE name = ?", (topic_name,))
                row = cur.fetchone()
                if not row:
                    continue
                topic_id = row[0]
                cur.execute(
                    """INSERT OR IGNORE INTO item_topics (item_id, topic_id, assigned_at)
                       VALUES (?, ?, CURRENT_TIMESTAMP)""",
                    (item_id, topic_id)
                )
                if cur.rowcount > 0:
                    assigned_count += 1
            conn.commit()
            return assigned_count
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Failed to save topics for item {item_id}: {e}")
            return 0
        finally:
            conn.close()


def evaluate_with_llm(title, content, url, ollama_base_url='http://localhost:11434', model='llama3.1', timeout=180, max_retries=2):
    """Evaluate an article using LLM via Ollama API.
    
    Args:
        title: Article title
        content: Article content
        url: Article URL
        ollama_base_url: Ollama API base URL
        model: Model name to use
        timeout: Request timeout in seconds (default: 180 for large models)
        max_retries: Maximum number of retry attempts (default: 2)
    
    Returns a dict with: relevance_score, decision, topic, topics, topic_confidences, why_it_matters, target_audience
    """
    # Build topic list for the prompt
    topics_str = ", ".join(GENAI_TOPICS)
    
    # Prepare the prompt for GENAI_NEWS evaluation with topic classification
    prompt = f"""You are a technical news evaluator for AI engineers. 
Evaluate the following Hacker News article for relevance to AI/LLM/Programming news.

Title: {title}
URL: {url}
Content: {content[:2000] if content else 'No content available'}

AVAILABLE TOPICS FOR CLASSIFICATION:
{topics_str}

Topic descriptions:
- llm_research: LLM Research, papers, transformer models, fine-tuning, training
- ai_tools: AI-powered tools, chatbots, assistants, copilots, agents
- ml_infrastructure: MLOps, deployment, inference, GPU, Kubernetes, serving
- prompt_engineering: Prompt design, chain-of-thought, few-shot, RAG, retrieval
- ai_ethics: AI safety, alignment, bias, ethics, responsible AI, hallucination

Provide a JSON response with the following fields:
- relevance_score: float between 0.0 and 1.0 (how relevant is this to AI/LLM/Programming?)
- decision: "KEEP" or "DROP" (should this be included in the digest?)
- topic: brief topic/category description (e.g., "LLM Research", "AI Tools", "Programming Languages")
- topics: list of 1-3 matching topic names from the AVAILABLE TOPICS above (use exact names)
- topic_confidences: object mapping each topic name to a confidence score (0.0-1.0)
- why_it_matters: 1-2 sentence explanation of why this matters
- target_audience: one of "developer", "software architect", or "manager"

Example topics and topic_confidences format:
"topics": ["llm_research", "ai_tools"],
"topic_confidences": {{"llm_research": 0.9, "ai_tools": 0.6}}

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
        relevance_score = float(evaluation.get('relevance_score', 0.0))
        decision = 'KEEP' if relevance_score >= GENAI_NEWS_MIN_RELEVANCE else 'DROP'
        
        # Extract and validate topics
        raw_topics = evaluation.get('topics', [])
        # Filter to only valid predefined topics
        valid_topics = [t for t in raw_topics if t in GENAI_TOPICS]
        if not valid_topics and evaluation.get('topic'):
            # Fallback: try to match the free-form topic to predefined topics
            topic_lower = evaluation.get('topic', '').lower()
            for t in GENAI_TOPICS:
                if t.replace('_', ' ') in topic_lower or t.replace('_', '') in topic_lower.replace(' ', ''):
                    valid_topics.append(t)
                    break
        
        # Extract and validate topic confidences
        raw_confidences = evaluation.get('topic_confidences', {})
        topic_confidences = {k: float(v) for k, v in raw_confidences.items() if k in GENAI_TOPICS}
        # Ensure all valid_topics have a confidence score
        for t in valid_topics:
            if t not in topic_confidences:
                topic_confidences[t] = 1.0  # Default confidence
        
        return {
            'relevance_score': relevance_score,
            'decision': decision,
            'topic': evaluation.get('topic', 'Unknown'),
            'topics': valid_topics,
            'topic_confidences': topic_confidences,
            'why_it_matters': evaluation.get('why_it_matters', ''),
            'target_audience': evaluation.get('target_audience', 'developer'),
            'llm_model': model
        }
        
    except Exception as e:
        print(f"Error evaluating: {e}")
        # Return default values on error
        return {
            'relevance_score': 0.0,
            'decision': 'DROP',
            'topic': 'Error',
            'topics': [],
            'topic_confidences': {},
            'why_it_matters': f'Evaluation failed: {str(e)}',
            'target_audience': 'developer',
            'llm_model': model
        }


def get_items_for_evaluation(db_path='mydb.db', hours=24):
    """Get items that need evaluation.
    
    Criteria:
    - Published within last `hours` (default 24). Use hours=0 or None to skip time filter (all unevaluated).
    - Status is 'INGESTED' or 'PREFILTERED'
    - Not already evaluated (no entry in evaluations table for GENAI_NEWS persona)
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        use_time_filter = hours is not None and hours > 0
        if use_time_filter:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')

        # published_at can be: NULL, ISO 8601 text (e.g. ...T...Z or ...+00:00), or Unix timestamp (integer).
        time_cond = """
            (
            i.published_at IS NULL
            OR (
                typeof(i.published_at) IN ('integer', 'real')
                AND datetime(cast(i.published_at AS INTEGER), 'unixepoch') >= datetime(?)
            )
            OR (
                typeof(i.published_at) = 'text'
                AND datetime(
                    replace(replace(replace(trim(i.published_at), 'T', ' '), 'Z', ''), '+00:00', '')
                ) >= datetime(?)
            )
            )
        """ if use_time_filter else "1"
        query = f"""
        SELECT i.id, i.title, i.content, i.url, i.published_at, i.source, i.status
        FROM items i
        LEFT JOIN evaluations e ON i.id = e.item_id AND e.persona = 'GENAI_NEWS'
        WHERE {time_cond}
          AND i.status IN ('INGESTED', 'PREFILTERED')
          AND e.id IS NULL
        ORDER BY i.published_at DESC NULLS LAST, i.ingestion_time DESC
        """
        cur.execute(query, (cutoff_str, cutoff_str) if use_time_filter else ())
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


def save_evaluation(item_id, evaluation_result, persona='GENAI_NEWS', db_path='mydb.db', evaluation_type='FULL'):
    """Save evaluation result to the evaluations table.
    evaluation_type: 'FULL' (full content), 'TFIDF', 'TEXTRANK', or other.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        
        cur.execute("""
            INSERT OR REPLACE INTO evaluations (
                item_id, persona, decision, relevance_score,
                topic, why_it_matters, target_audience, llm_model, evaluation_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item_id,
            persona,
            evaluation_result['decision'],
            evaluation_result['relevance_score'],
            evaluation_result['topic'],
            evaluation_result['why_it_matters'],
            evaluation_result['target_audience'],
            evaluation_result['llm_model'],
            evaluation_type,
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


def _evaluate_one_item(item, ollama_base_url, model, timeout, db_path):
    """Evaluate a single item (for use with ThreadPoolExecutor). Returns (item_id, evaluation, error)."""
    try:
        evaluation = evaluate_with_llm(
            title=item['title'],
            content=item['content'] or '',
            url=item['url'] or '',
            ollama_base_url=ollama_base_url,
            model=model,
            timeout=timeout
        )
        return (item['id'], evaluation, None)
    except Exception as e:
        return (item['id'], None, e)


def run_evaluation_pipeline(db_path='mydb.db', ollama_base_url='http://localhost:11434', 
                           model='llama3.1', hours=24, verbose=True, timeout=180, max_workers=3):
    """Run the evaluation pipeline on items that need evaluation.
    
    Filters items by:
    - Published in last 24 hours (or specified hours)
    - Status is INGESTED or PREFILTERED
    - Not already evaluated
    
    Evaluates up to max_workers items concurrently with LLM and saves results.
    """
    print("=" * 60)
    print("EVALUATION PIPELINE")
    print("=" * 60)
    
    print(f"\nStep 1: Fetching items for evaluation (published in last {hours} hours)...")
    items = get_items_for_evaluation(db_path, hours)
    
    if not items:
        print("[OK] No items found that need evaluation.")
        return 0
    
    print(f"[OK] Found {len(items)} items to evaluate (up to {max_workers} concurrent)")
    
    print(f"\nStep 2: Evaluating items...")
    evaluated_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(
                _evaluate_one_item,
                item,
                ollama_base_url,
                model,
                timeout,
                db_path,
            ): item
            for item in items
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                item_id, evaluation, err = future.result()
                if err:
                    error_count += 1
                    if verbose:
                        print(f"    [ERROR] {item['title'][:50]}...: {err}")
                    continue
                if save_evaluation(item_id, evaluation, persona='GENAI_NEWS', db_path=db_path, evaluation_type='FULL'):
                    evaluated_count += 1
                    # Save topic assignments for KEEP items
                    if evaluation.get('decision') == 'KEEP' and evaluation.get('topics'):
                        topics_saved = save_item_topics(item_id, evaluation, db_path=db_path)
                        if verbose and topics_saved > 0:
                            topic_names = evaluation.get('topics', [])
                            print(f"        Topics: {', '.join(topic_names)}")
                    if verbose:
                        decision = evaluation['decision']
                        score = evaluation['relevance_score']
                        print(f"    [{decision}] Relevance: {score:.2f} | Topic: {evaluation['topic']} | {item['title'][:45]}...")
                else:
                    error_count += 1
                    if verbose:
                        print(f"    [ERROR] Failed to save evaluation for item {item_id}")
            except Exception as e:
                error_count += 1
                if verbose:
                    print(f"    [ERROR] {item['title'][:50]}...: {e}")
    
    print(f"\n[OK] Evaluation complete!")
    print(f"[OK] Evaluated: {evaluated_count} items")
    if error_count > 0:
        print(f"[WARNING] Errors: {error_count} items")
    
    return evaluated_count


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluation Pipeline: Evaluate articles using LLM via Ollama')
    parser.add_argument('--db', type=str, default='mydb.db', help='Database file path')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--model', type=str, default='llama3.1', help='Ollama model name (default: llama3.1)')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back (default: 24). Use 0 for no time limit (all unevaluated items).')
    parser.add_argument('--timeout', type=int, default=180, help='Request timeout in seconds (default: 180)')
    parser.add_argument('--workers', type=int, default=3, help='Number of concurrent evaluations (default: 3)')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output')
    
    args = parser.parse_args()
    
    run_evaluation_pipeline(
        db_path=args.db,
        ollama_base_url=args.ollama_url,
        model=args.model,
        hours=args.hours,
        verbose=not args.quiet,
        timeout=args.timeout,
        max_workers=args.workers
    )
