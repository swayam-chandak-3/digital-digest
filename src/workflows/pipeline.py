# Fix for Python 3.9+ compatibility with older BeautifulSoup versions
import sys
import collections
import os
import requests
import time
import re
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from readability import Document
from datetime import datetime, timezone, timedelta
from ..tools.helper import _evaluate_one_item, get_items_for_evaluation, save_evaluation
from ..tools.db_utils import save_items_to_db, categorize_article, TOPIC_KEYWORDS
from .reddit_scraper import (
    run_reddit_pipeline,
    collect_reddit_posts,
    reddit_posts_to_items,
    REDDIT_TIME_WINDOW_HOURS,
    REDDIT_LIMIT_PER_SUBREDDIT,
    REDDIT_LISTING_PAGE_LIMIT,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DB_PATH = Path(os.getenv('DB_PATH', 'mydb.db'))

if sys.version_info >= (3, 9):
    try:
        from collections.abc import Callable
        if not hasattr(collections, 'Callable'):
            collections.Callable = Callable
    except ImportError:
        from typing import Callable
        if not hasattr(collections, 'Callable'):
            collections.Callable = Callable

def _extract_metadata(soup):
    """Extract author and publish datetime metadata from a BeautifulSoup document."""
    author = None
    publish_datetime = None

    # Common author meta tags
    author_meta = (
        soup.find('meta', attrs={'name': 'author'})
        or soup.find('meta', attrs={'property': 'article:author'})
        or soup.find('meta', attrs={'name': 'twitter:creator'})
    )
    if author_meta and author_meta.get('content'):
        author = author_meta['content'].strip()

    # Fallback: look for elements with common author classes
    if not author:
        author_elem = soup.find(class_=re.compile('author', re.I))
        if author_elem:
            author_text = author_elem.get_text(strip=True)
            if author_text:
                author = author_text

    # Common publish datetime meta tags
    publish_meta_candidates = [
        {'property': 'article:published_time'},
        {'name': 'article:published_time'},
        {'property': 'og:updated_time'},
        {'property': 'article:modified_time'},
        {'itemprop': 'datePublished'},
        {'name': 'date'},
        {'name': 'DC.date.issued'},
    ]
    for attrs in publish_meta_candidates:
        meta = soup.find('meta', attrs=attrs)
        if meta and meta.get('content'):
            publish_datetime = meta['content'].strip()
            break

    # Fallback: look for <time> element with datetime attribute
    if not publish_datetime:
        time_elem = soup.find('time')
        if time_elem and time_elem.get('datetime'):
            publish_datetime = time_elem['datetime'].strip()

    return author, publish_datetime


def extract_article_content(url, timeout=10):
    """Extract article content and metadata from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Build a soup of the full page once for metadata extraction
        full_soup = BeautifulSoup(response.content, 'html.parser')
        author, publish_datetime = _extract_metadata(full_soup)

        # Try readability first for main content
        try:
            doc = Document(response.content)
            title = doc.title()
            content_html = doc.summary()
            
            soup = BeautifulSoup(content_html, 'html.parser')
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            if text and len(text) > 100:
                return {
                    'title': title,
                    'text': text,
                    'author': author,
                    'publish_datetime': publish_datetime,
                    'success': True,
                }
        except Exception:
            pass
        
        # Fallback to BeautifulSoup
        soup = full_soup
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "advertisement", "ads"]):
            element.decompose()
        
        article = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile('article|content|post', re.I))
        
        if article:
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else "No Title"
            text = article.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return {
                'title': title,
                'text': text,
                'author': author,
                'publish_datetime': publish_datetime,
                'success': True,
            }
        
        return {
            'success': False,
            'error': 'Could not extract content',
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Pre-filter defaults (align with AI-Powered Intelligence Digest System)
DEFAULT_HOURS_WINDOW = 24
DEFAULT_MIN_POINTS = 1
DEFAULT_MIN_COMMENTS = 0


def _parse_time_ago_to_hours(time_ago_str):
    """Parse HN 'time ago' string (e.g. '2 hours ago', '1 day ago') to hours ago.
    Returns None if unparseable (e.g. 'just now').
    """
    if not time_ago_str or not isinstance(time_ago_str, str):
        return None
    s = time_ago_str.strip().lower()
    m = re.match(r'(\d+)\s*(minute|hour|day)s?\s*ago', s)
    if not m:
        if 'minute' in s or 'hour' in s or 'day' in s:
            m = re.search(r'(\d+)\s*(minute|hour|day)', s)
        if not m:
            return 0  # "just now" or similar -> within window
    num, unit = int(m.group(1)), m.group(2)
    if unit.startswith('minute'):
        return num / 60.0
    if unit.startswith('hour'):
        return float(num)
    if unit.startswith('day'):
        return num * 24.0
    return None


def scrape_hackernews_pages(num_pages=1, verbose=False):
    """Scrape Hacker News and extract URL, title, points, comments, and time_ago per item."""
    if not os.path.exists('HackerNews'):
        os.makedirs('HackerNews')
    
    all_entries = []
    num_pages = min(num_pages, 20)
    
    for page_no in range(1, num_pages + 1):
        if verbose:
            print(f'Fetching Hacker News Page {page_no}...')
        
        try:
            res = requests.get('https://news.ycombinator.com/?p=' + str(page_no))
            soup = BeautifulSoup(res.content, 'html.parser')
            # Rows: tr.athing = title row; next tr = subtext (points, time, comments)
            athing_rows = soup.find_all('tr', class_='athing')
            
            for tr in athing_rows:
                row_id = tr.get('id')
                title_span = tr.find('span', class_='titleline')
                if not title_span:
                    continue
                a = title_span.find('a', href=True)
                if not a:
                    continue
                url = a['href']
                if not url.startswith('http'):
                    url = 'https://news.ycombinator.com/' + url
                title = a.get_text(strip=True)
                # Skip "More" link and HN-internal (discussion) links; only keep external article URLs
                if url.startswith('?p=') or ('news.ycombinator.com' in url and 'item' in url):
                    continue
                # Find subline row (next sibling tr contains points, time, comments)
                subline_tr = tr.find_next_sibling('tr')
                points = 0
                num_comments = 0
                time_ago_str = ''
                if subline_tr:
                    subline = subline_tr.find('span', class_='subline') or subline_tr
                    line_text = subline.get_text() if subline else ''
                    # e.g. "51 points by user 2 hours ago | 40 comments"
                    pm = re.search(r'(\d+)\s*points?', line_text, re.I)
                    if pm:
                        points = int(pm.group(1))
                    cm = re.search(r'(\d+)\s*comments?', line_text, re.I)
                    if cm:
                        num_comments = int(cm.group(1))
                    # "X minutes/hours/days ago"
                    tam = re.search(r'(\d+\s*(?:minute|hour|day)s?\s*ago)', line_text, re.I)
                    if tam:
                        time_ago_str = tam.group(1)
                
                all_entries.append({
                    'url': url,
                    'title': title,
                    'points': points,
                    'num_comments': num_comments,
                    'time_ago_str': time_ago_str,
                })
            
            if verbose:
                print(f'  Found {len(athing_rows)} items on page {page_no}')
        
        except Exception as e:
            print(f'Error fetching page {page_no}: {e}')
    
    return all_entries


def apply_prefilter(entries, hours_window=24, min_points=0, min_comments=0, verbose=False):
    """Apply time window and engagement thresholds. Returns list of entries that pass."""
    filtered = []
    for e in entries:
        # Time window: last N hours
        hours_ago = _parse_time_ago_to_hours(e.get('time_ago_str'))
        if hours_ago is not None and hours_ago > hours_window:
            if verbose:
                print(f"  [PRE-SKIP] Time: {e.get('time_ago_str')} (> {hours_window}h): {e.get('title', '')[:50]}...")
            continue
        if e.get('points', 0) < min_points:
            if verbose:
                print(f"  [PRE-SKIP] Points {e.get('points')} < {min_points}: {e.get('title', '')[:50]}...")
            continue
        if e.get('num_comments', 0) < min_comments:
            if verbose:
                print(f"  [PRE-SKIP] Comments {e.get('num_comments')} < {min_comments}: {e.get('title', '')[:50]}...")
            continue
        filtered.append(e)
    return filtered

def scrape_and_categorize_articles(entries, delay=1, verbose=False):
    """Scrape full article content and apply keyword-based category filter.
    entries: list of dicts with url, title, points, num_comments, time_ago_str from HN listing.
    """
    target_categories = ['LLM/AI']  # Only AI/LLM articles (keyword relevance)
    filtered_articles = []
    
    print(f"\nScraping {len(entries)} articles (keyword + category filter)...")
    print("=" * 60)
    
    for i, entry in enumerate(entries, 1):
        url = entry.get('url', '')
        if verbose:
            print(f"[{i}/{len(entries)}] Processing: {url[:60]}...")
        
        result = extract_article_content(url)
        
        if not result.get('success'):
            if verbose:
                print(f"    Failed: {result.get('error', 'Unknown error')}")
            continue
        
        title = result.get('title', 'No Title')
        text = result.get('text', '')
        author = result.get('author')
        publish_datetime = hn_time_ago_to_utc(entry.get('time_ago_str'))

        
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        categories = categorize_article(text, title, domain, news_type=['LLM/AI', 'Programming/Software'])
        matches_target = any(cat in categories for cat in target_categories)
        
        if matches_target:
            primary_category = categories[0] if categories else 'Uncategorized'
            scraping_timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            points = entry.get('points', 0)
            num_comments = entry.get('num_comments', 0)
            # engagement_score: points + weight * comments for DB/sorting
            engagement_score = float(points + 2 * num_comments)
            article_data = {
                'link': url,
                'author': author,
                'title': title,
                'source': 'hackernews',
                'content': text,
                'publish_datetime': publish_datetime,
                'scraping_timestamp': scraping_timestamp,
                'domain': domain,
                'categories': categories,
                'primary_category': primary_category,
                'points': points,
                'num_comments': num_comments,
                'engagement_score': engagement_score,
            }
            filtered_articles.append(article_data)
            if verbose:
                print(f"    [MATCH] {primary_category}: {title[:50]}... (pts={points}, comments={num_comments})")
        else:
            if verbose:
                print(f"    [SKIP] Categories: {categories if categories else 'Uncategorized'}")
        
        if i < len(entries):
            time.sleep(delay)
    
    return filtered_articles


def hn_time_ago_to_utc(time_ago_str):
    """Convert HN 'X minutes/hours/days ago' to UTC ISO timestamp."""
    hours_ago = _parse_time_ago_to_hours(time_ago_str)
    if hours_ago is None:
        return None
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return dt.isoformat().replace('+00:00', 'Z')


def save_filtered_articles(articles, output_file='HackerNews/llm_programming_articles.json'):
    """Save filtered articles to a JSON file.

    The JSON will be a list of objects with at least:
      - link
      - author
      - title
      - source
      - content
      - publish_datetime
      - scraping_timestamp
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    # Only keep the primary fields requested plus categories info if present
    cleaned_articles = []
    for article in articles:
        cleaned = {
            'link': article.get('link'),
            'author': article.get('author'),
            'title': article.get('title'),
            'source': article.get('source'),
            'content': article.get('content'),
            'publish_datetime': article.get('publish_datetime'),
            'scraping_timestamp': article.get('scraping_timestamp'),
        }
        if 'categories' in article:
            cleaned['categories'] = article['categories']
        if 'primary_category' in article:
            cleaned['primary_category'] = article['primary_category']
        if 'points' in article:
            cleaned['points'] = article['points']
        if 'num_comments' in article:
            cleaned['num_comments'] = article['num_comments']
        if 'engagement_score' in article:
            cleaned['engagement_score'] = article['engagement_score']
        cleaned_articles.append(cleaned)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'generated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'total_articles': len(cleaned_articles),
                'articles': cleaned_articles,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_file


def run_pipeline(num_pages=3, delay=1, verbose=True, output_file='HackerNews/llm_programming_articles.json',
                 hours_window=DEFAULT_HOURS_WINDOW, min_points=DEFAULT_MIN_POINTS, min_comments=DEFAULT_MIN_COMMENTS):
    """Run the complete pipeline with pre-filtering (time window + engagement)."""
    print("=" * 60)
    print("HACKER NEWS PIPELINE")
    print("=" * 60)
    print("\nStep 1: Scraping Hacker News (with engagement metrics)...")
    
    entries = scrape_hackernews_pages(num_pages, verbose)
    print(f"\n[OK] Found {len(entries)} items from Hacker News")
    
    if not entries:
        print("No items found. Exiting.")
        return
    
    # Pre-filter: time window (last N hours) + engagement thresholds
    print(f"\nStep 2: Pre-filtering (last {hours_window}h, min_points={min_points}, min_comments={min_comments})...")
    entries = apply_prefilter(entries, hours_window=hours_window, min_points=min_points, min_comments=min_comments, verbose=verbose)
    print(f"[OK] {len(entries)} items pass pre-filter")
    
    if not entries:
        print("No items pass pre-filter. Exiting.")
        return
    
    # Step 3: Scrape full content and apply keyword/category filter (LLM/AI only)
    print("\nStep 3: Scraping articles and keyword/category filtering...")
    filtered_articles = scrape_and_categorize_articles(entries, delay, verbose)
    
    print(f"\n[OK] Found {len(filtered_articles)} articles matching LLM/AI category")
    
    if not filtered_articles:
        print("No matching articles found.")
        return
    
    # Step 4: Save filtered articles
    print("\nStep 4: Saving filtered articles...")
    output_path = save_filtered_articles(filtered_articles, output_file)

    # Step 5: Persist into the items table
    print("\nStep 5: Writing articles to database (items table)...")
    try:
        inserted = save_items_to_db(filtered_articles, db_path=DB_PATH, digest_type='GENAI')
        print(f"[OK] Inserted {inserted} rows into items table")
    except Exception as e:
        print(f"[ERROR] Failed to insert into items table: {e}")
    
    print("\n[OK] Pipeline complete!")
    print(f"[OK] Saved {len(filtered_articles)} articles to: {output_path}")
    
    # Summary by category
    category_counts = {}
    for article in filtered_articles:
        cat = article['primary_category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nArticles by Category:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} articles")
    
    return output_path


def run_evaluation_pipeline(db_path=DB_PATH, ollama_base_url='http://localhost:11434', 
                           model='gemma3:12b', hours=24, verbose=True, timeout=180, max_workers=3):
    """Run the evaluation pipeline on items that need evaluation.
    
    Filters items by:
    - Published in last 24 hours (or specified hours)
    - Status is INGESTED or PREFILTERED
    - Not already evaluated
    
    Evaluates up to max_workers items concurrently with Gemma3 and saves results.
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
    
    print(f"\nStep 2: Evaluating items with Gemma3 ({model})...")
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
    
    parser = argparse.ArgumentParser(description='Hacker News Pipeline: Scrape, categorize, filter, and evaluate articles')
    parser.add_argument('--pages', type=int, default=3, help='Number of Hacker News pages to scrape (max 20, default: 3)')
    parser.add_argument('--delay', type=float, default=1, help='Delay between article requests (seconds)')
    parser.add_argument('--output', type=str, default='HackerNews/llm_programming_articles.json',
                       help='Output file path')
    parser.add_argument('--hours-window', type=int, default=DEFAULT_HOURS_WINDOW,
                       help='Pre-filter: only items from last N hours (default: 24)')
    parser.add_argument('--min-points', type=int, default=DEFAULT_MIN_POINTS,
                       help='Pre-filter: minimum HN points (default: 1)')
    parser.add_argument('--min-comments', type=int, default=DEFAULT_MIN_COMMENTS,
                       help='Pre-filter: minimum comment count (default: 0)')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation pipeline after ingestion')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation, skip ingestion')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--model', type=str, default='gemma3:12b', help='Ollama model name')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back for recent items')
    parser.add_argument('--no-reddit', action='store_true', help='Skip Reddit JSON scraper (by default Reddit runs)')
    parser.add_argument('--no-hackernews', action='store_true', help='Skip Hacker News scraper (by default HN runs)')
    parser.add_argument('--reddit-output-dir', type=str, default='output', help='Output directory for Reddit JSON files')
    parser.add_argument('--reddit-hours-window', type=int, default=REDDIT_TIME_WINDOW_HOURS,
                       help='Reddit: only items from last N hours (default: 24)')
    parser.add_argument('--reddit-limit', type=int, default=REDDIT_LIMIT_PER_SUBREDDIT,
                       help='Reddit: limit posts per subreddit (default: 20)')
    parser.add_argument('--reddit-listing-limit', type=int, default=REDDIT_LISTING_PAGE_LIMIT,
                       help='Reddit: listing page limit (default: 100)')
    parser.add_argument('--reddit-no-db', action='store_true', help='Skip saving Reddit posts to database')
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Only run evaluation
        run_evaluation_pipeline(
            db_path=DB_PATH,
            ollama_base_url=args.ollama_url,
            model=args.model,
            hours=args.hours,
            verbose=not args.quiet
        )
    else:
        # By default, run both Reddit and Hacker News pipelines
        if not args.no_reddit:
            run_reddit_pipeline(
                output_dir=args.reddit_output_dir,
                hours_window=args.reddit_hours_window,
                limit_per_subreddit=args.reddit_limit,
                listing_page_limit=args.reddit_listing_limit,
                verbose=not args.quiet,
                persist_to_db=not args.reddit_no_db,
                db_path=DB_PATH,
            )

        if not args.no_hackernews:
            # Run Hacker News ingestion pipeline
            run_pipeline(
                num_pages=min(args.pages, 20),
                delay=args.delay,
                verbose=not args.quiet,
                output_file=args.output,
                hours_window=args.hours_window,
                min_points=args.min_points,
                min_comments=args.min_comments,
            )
        
        # Optionally run evaluation after ingestion
        if args.evaluate:
            print("\n" + "=" * 60)
            run_evaluation_pipeline(
                db_path=DB_PATH,
                ollama_base_url=args.ollama_url,
                model=args.model,
                hours=args.hours,
                verbose=not args.quiet
            )
