# Fix for Python 3.9+ compatibility with older BeautifulSoup versions
import sys
import collections
if sys.version_info >= (3, 9):
    try:
        from collections.abc import Callable
        if not hasattr(collections, 'Callable'):
            collections.Callable = Callable
    except ImportError:
        from typing import Callable
        if not hasattr(collections, 'Callable'):
            collections.Callable = Callable

import os
import requests
import time
import re
import json
from urllib.parse import urlparse
from bs4 import BeautifulSoup, SoupStrainer
from readability import Document
from datetime import datetime

# Topic keywords for categorization
TOPIC_KEYWORDS = {
    'LLM/AI': [
        'llm', 'large language model', 'gpt', 'claude', 'ai model', 'language model',
        'transformer', 'neural network', 'machine learning', 'deep learning',
        'artificial intelligence', 'chatbot', 'generative ai', 'openai', 'anthropic',
        'prompt engineering', 'fine-tuning', 'hallucination', 'constitution', 'alignment'
    ],
    'Programming/Software': [
        'programming', 'code', 'developer', 'software', 'algorithm', 'api', 'framework',
        'library', 'github', 'repository', 'open source', 'programming language',
        'javascript', 'python', 'java', 'c++', 'rust', 'go', 'function', 'variable'
    ]
}

def categorize_article(text, title='', domain=''):
    """Categorize an article based on its content."""
    if not text:
        return []
    
    full_text = f"{title} {text}".lower()
    category_scores = {}
    
    for category, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = len(re.findall(pattern, full_text))
            score += matches
        
        if score > 0:
            category_scores[category] = score
    
    # Return categories with significant scores
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_categories:
        return []
    
    primary_score = sorted_categories[0][1]
    threshold = primary_score * 0.3
    categories = [cat for cat, score in sorted_categories if score >= threshold]
    
    return categories

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

def scrape_hackernews_pages(num_pages=1, verbose=False):
    """Scrape Hacker News and extract URLs."""
    if not os.path.exists('HackerNews'):
        os.makedirs('HackerNews')
    
    all_urls = []
    num_pages = min(num_pages, 20)
    
    for page_no in range(1, num_pages + 1):
        if verbose:
            print(f'Fetching Hacker News Page {page_no}...')
        
        try:
            res = requests.get('https://news.ycombinator.com/?p=' + str(page_no))
            only_td = SoupStrainer('td')
            soup = BeautifulSoup(res.content, 'html.parser', parse_only=only_td)
            
            tdtitle = soup.find_all('td', attrs={'class': 'title'})
            tdrank = soup.find_all('td', attrs={'class': 'title', 'align': 'right'})
            tdtitleonly = [t for t in tdtitle if t not in tdrank]
            
            page_urls = []
            for tdt in tdtitleonly:
                titleline = tdt.find('span', attrs={'class': 'titleline'})
                if titleline:
                    titl = titleline.find('a')
                    if titl and 'href' in titl.attrs:
                        url = titl['href']
                        if not url.startswith('http'):
                            url = 'https://news.ycombinator.com/' + url
                        page_urls.append(url)
            
            all_urls.extend(page_urls)
            if verbose:
                print(f'  Found {len(page_urls)} URLs on page {page_no}')
        
        except Exception as e:
            print(f'Error fetching page {page_no}: {e}')
    
    return all_urls

def scrape_and_categorize_articles(urls, delay=1, verbose=False):
    """Scrape articles from URLs and categorize them."""
    target_categories = ['LLM/AI', 'Programming/Software']
    filtered_articles = []
    
    print(f"\nScraping {len(urls)} articles...")
    print("=" * 60)
    
    for i, url in enumerate(urls, 1):
        if verbose:
            print(f"[{i}/{len(urls)}] Processing: {url[:60]}...")
        
        # Extract article content
        result = extract_article_content(url)
        
        if not result.get('success'):
            if verbose:
                print(f"    Failed: {result.get('error', 'Unknown error')}")
            continue
        
        title = result.get('title', 'No Title')
        text = result.get('text', '')
        author = result.get('author')
        publish_datetime = result.get('publish_datetime')
        
        # Extract domain
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        
        # Categorize
        categories = categorize_article(text, title, domain)
        
        # Check if article matches target categories
        matches_target = any(cat in categories for cat in target_categories)
        
        if matches_target:
            primary_category = categories[0] if categories else 'Uncategorized'
            scraping_timestamp = datetime.utcnow().isoformat() + 'Z'
            article_data = {
                # Core JSON output fields
                'link': url,
                'author': author,
                'title': title,
                'source': domain,
                'content': text,
                'publish_datetime': publish_datetime,
                'scraping_timestamp': scraping_timestamp,
                # Additional internal fields
                'categories': categories,
                'primary_category': primary_category,
            }
            filtered_articles.append(article_data)
            
            if verbose:
                print(f"    [MATCH] {primary_category}: {title[:50]}...")
        else:
            if verbose:
                print(f"    [SKIP] Categories: {categories if categories else 'Uncategorized'}")
        
        # Delay between requests
        if i < len(urls):
            time.sleep(delay)
    
    return filtered_articles

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
        # Optional: keep categories if you want them for downstream use
        if 'categories' in article:
            cleaned['categories'] = article['categories']
        if 'primary_category' in article:
            cleaned['primary_category'] = article['primary_category']
        cleaned_articles.append(cleaned)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'generated_at': datetime.utcnow().isoformat() + 'Z',
                'total_articles': len(cleaned_articles),
                'articles': cleaned_articles,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_file

def run_pipeline(num_pages=1, delay=1, verbose=True, output_file='HackerNews/llm_programming_articles.json'):
    """Run the complete pipeline."""
    print("=" * 60)
    print("HACKER NEWS PIPELINE")
    print("=" * 60)
    print("\nStep 1: Scraping Hacker News...")
    
    # Step 1: Scrape Hacker News
    urls = scrape_hackernews_pages(num_pages, verbose)
    print(f"\n[OK] Found {len(urls)} URLs from Hacker News")
    
    if not urls:
        print("No URLs found. Exiting.")
        return
    
    # Step 2: Scrape and categorize articles
    print("\nStep 2: Scraping articles and categorizing...")
    filtered_articles = scrape_and_categorize_articles(urls, delay, verbose)
    
    print(f"\n[OK] Found {len(filtered_articles)} articles matching LLM/AI or Programming/Software")
    
    if not filtered_articles:
        print("No matching articles found.")
        return
    
    # Step 3: Save filtered articles
    print("\nStep 3: Saving filtered articles...")
    output_path = save_filtered_articles(filtered_articles, output_file)
    
    print(f"\n[OK] Pipeline complete!")
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

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hacker News Pipeline: Scrape, categorize, and filter LLM/AI & Programming articles')
    parser.add_argument('--pages', type=int, default=1, help='Number of Hacker News pages to scrape (max 20)')
    parser.add_argument('--delay', type=float, default=1, help='Delay between article requests (seconds)')
    parser.add_argument('--output', type=str, default='HackerNews/llm_programming_articles.json',
                       help='Output file path')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output')
    
    args = parser.parse_args()
    
    run_pipeline(
        num_pages=min(args.pages, 20),
        delay=args.delay,
        verbose=not args.quiet,
        output_file=args.output
    )
