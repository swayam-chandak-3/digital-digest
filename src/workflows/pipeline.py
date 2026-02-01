# Fix for Python 3.9+ compatibility with older BeautifulSoup versions
import sys
import collections
import os
import requests
import time
import re
import json
import httpx
from dataclasses import dataclass, field
from html import unescape
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from readability import Document
from datetime import datetime, timezone, timedelta
from ..tools.helper import _evaluate_one_item, get_items_for_evaluation, save_evaluation
from ..tools.db_utils import save_items_to_db, categorize_article
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DB_PATH = Path(os.getenv('DB_PATH', 'evalution.db'))

# Reddit defaults
REDDIT_TIME_WINDOW_HOURS = int(os.getenv('REDDIT_TIME_WINDOW_HOURS', '24'))
REDDIT_LIMIT_PER_SUBREDDIT = int(os.getenv('REDDIT_LIMIT_PER_SUBREDDIT', '20'))
REDDIT_LISTING_PAGE_LIMIT = int(os.getenv('REDDIT_LISTING_PAGE_LIMIT', '100'))
REDDIT_SUBREDDITS = [
    s.strip() for s in os.getenv('REDDIT_SUBREDDITS', 'LocalLLaMA,MachineLearning,artificial').split(',')
    if s.strip()
]

TECH_URL = "https://www.indiehackers.com/tech"
AI_TAG_URL = "https://www.indiehackers.com/tags/artificial-intelligence"

ENTRY_RE = re.compile(
    r'<a[^>]+href="(?P<href>/post/[^"]+)"[^>]*class="[^"]*portal-entry[^"]*"[^>]*>(?P<body>.*?)</a>',
    re.DOTALL,
)
DATE_RE = re.compile(r'<span class="portal-entry__date">([^<]+)</span>')
TITLE_RE = re.compile(r'<span class="portal-entry__title">\s*(.*?)\s*</span>', re.DOTALL)
SUMMARY_RE = re.compile(r'<span class="portal-entry__summary">\s*(.*?)\s*</span>', re.DOTALL)
BYLINE_RE = re.compile(r'<span class="portal-entry__byline">by\s+([^<]+)</span>')
COMMENTS_RE = re.compile(
    r'<div[^>]+class="[^"]*portal-entry__comments[^"]*"[^>]*>.*?<span>(\d+)</span>',
    re.DOTALL,
)
IMAGE_RE = re.compile(r'<img[^>]+class="portal-entry__image"[^>]+src="([^"]+)"')
OG_TITLE_RE = re.compile(r'<meta property="og:title" content="([^"]+)"')
OG_DESC_RE = re.compile(r'<meta property="og:description" content="([^"]+)"')
ARTICLE_RE = re.compile(r"<article[^>]*>(?P<body>.*?)</article>", re.DOTALL | re.IGNORECASE)
POST_CONTENT_RE = re.compile(
    r'<div[^>]+class="[^"]*post__content[^"]*"[^>]*>(?P<body>.*?)</div>',
    re.DOTALL,
)
TIME_RE = re.compile(r'<time[^>]+datetime="([^"]+)"')
JSON_COMMENTS_RE = re.compile(r'"comment[s]?Count"\s*:\s*(\d+)', re.IGNORECASE)
COMMENTS_TEXT_RE = re.compile(r'\b(\d+)\s+comments\b', re.IGNORECASE)

LIKES_COUNT_RE = re.compile(
    r'<div[^>]+class="[^"]*post-liker__count[^"]*"[^>]*>(\d+)</div>',
    re.IGNORECASE,
)
LIKES_COUNT_ALT_RE = re.compile(
    r'class="post-liker__count"[^>]*>(\d+)<',
    re.IGNORECASE,
)

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


def collect_reddit_posts(
    subreddits: List[str],
    hours_window: int = REDDIT_TIME_WINDOW_HOURS,
    limit_per_subreddit: int = REDDIT_LIMIT_PER_SUBREDDIT,
    listing_limit: int = REDDIT_LISTING_PAGE_LIMIT,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """Collect recent Reddit posts from a list of subreddits."""
    if not subreddits:
        return []
    headers = {
        "User-Agent": "digital-digest/1.0 (reddit scraper)"
    }
    cutoff_ts = datetime.now(timezone.utc).timestamp() - hours_window * 3600
    collected: List[Dict[str, object]] = []

    for subreddit in subreddits:
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={listing_limit}"
        if verbose:
            print(f"Fetching Reddit /r/{subreddit}...")
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            if verbose:
                print(f"  [ERROR] /r/{subreddit}: {e}")
            continue

        children = payload.get("data", {}).get("children", [])
        for child in children:
            data = child.get("data", {})
            if data.get("stickied"):
                continue
            created_utc = data.get("created_utc") or 0
            if created_utc and created_utc < cutoff_ts:
                continue
            collected.append({
                "subreddit": subreddit,
                "title": data.get("title") or "",
                "url": data.get("url") or data.get("permalink") or "",
                "permalink": f"https://www.reddit.com{data.get('permalink', '')}",
                "selftext": data.get("selftext") or "",
                "score": int(data.get("score") or 0),
                "num_comments": int(data.get("num_comments") or 0),
                "created_utc": created_utc,
                "author": data.get("author"),
            })
            if len([p for p in collected if p.get("subreddit") == subreddit]) >= limit_per_subreddit:
                break

    return collected


def reddit_posts_to_items(posts: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Convert Reddit posts to item dicts compatible with save_items_to_db."""
    items: List[Dict[str, object]] = []
    for post in posts:
        url = post.get("url") or post.get("permalink")
        title = post.get("title") or "Untitled"
        text = post.get("selftext") or ""
        created_utc = post.get("created_utc") or None
        publish_datetime = None
        if created_utc:
            publish_datetime = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat().replace("+00:00", "Z")

        categories = categorize_article(
            text,
            title,
            "reddit.com",
            news_type=['LLM/AI', 'Programming/Software']
        )

        items.append({
            "link": url,
            "author": post.get("author"),
            "title": title,
            "source": "reddit",
            "content": text,
            "publish_datetime": publish_datetime,
            "scraping_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "domain": "reddit.com",
            "categories": categories,
            "primary_category": categories[0] if categories else "Uncategorized",
            "points": post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
            "engagement_score": float((post.get("score") or 0) + 2 * (post.get("num_comments") or 0)),
            "raw": post,
        })
    return items


def run_reddit_pipeline(
    subreddits: List[str] = None,
    hours_window: int = REDDIT_TIME_WINDOW_HOURS,
    limit_per_subreddit: int = REDDIT_LIMIT_PER_SUBREDDIT,
    listing_page_limit: int = REDDIT_LISTING_PAGE_LIMIT,
    output_dir: str = "output_reddit",
    verbose: bool = True,
    persist_to_db: bool = True,
    db_path: Path = DB_PATH,
) -> str:
    """Run Reddit scraping, save JSON, and persist to DB."""
    print("=" * 60)
    print("REDDIT PIPELINE")
    print("=" * 60)

    subs = subreddits or REDDIT_SUBREDDITS
    print(f"\nStep 1: Scraping Reddit ({', '.join(subs)})...")
    posts = collect_reddit_posts(
        subs,
        hours_window=hours_window,
        limit_per_subreddit=limit_per_subreddit,
        listing_limit=listing_page_limit,
        verbose=verbose,
    )
    print(f"[OK] Collected {len(posts)} posts from Reddit")
    if not posts:
        return ""

    items = reddit_posts_to_items(posts)

    print("\nStep 2: Saving Reddit JSON...")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "reddit_latest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved outputs to: {json_path}")

    if persist_to_db:
        print("\nStep 3: Writing Reddit items to database...")
        inserted = save_items_to_db(items, db_path=str(db_path), digest_type="GENAI")
        print(f"[OK] Inserted {inserted} rows into items table")

    return json_path


@dataclass
class IngestedItem:
    source: str
    source_id: Optional[str]
    url: str
    title: str
    description: Optional[str]
    content: Optional[str]
    created_at: datetime
    fetched_at: datetime = field(default_factory=lambda: datetime.utcnow())
    engagement: Dict[str, object] = field(default_factory=dict)
    raw: Dict[str, object] = field(default_factory=dict)

    def as_json(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "engagement": self.engagement,
            "raw": self.raw,
        }


def _load_env(path: str = ".env") -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def _get_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _strip(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return " ".join(unescape(text).split())


def _strip_tags(html_text: str) -> str:
    return re.sub(r"<[^>]+>", " ", html_text)


def _parse_date(text: str) -> datetime:
    try:
        dt = datetime.strptime(text.strip(), "%b %d, %Y")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _parse_iso_datetime(text: Optional[str]) -> datetime:
    if not text:
        return datetime.now(timezone.utc)
    try:
        value = text.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def _fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    with httpx.Client(timeout=30, headers=headers, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    return resp.text


def _extract_post_content(html_text: str) -> Optional[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    root = (
        soup.select_one("div.tiptap.ProseMirror.firestore-post__content")
        or soup.select_one("div.tiptap.ProseMirror")
        or soup.select_one("div.firestore-post__main")
        or soup.select_one("div.post-page__content")
    )
    if not root:
        return None
    for selector in [
        "nav",
        "footer",
        "form",
        ".post-page__comments",
        ".embedded-comments",
        ".comment-tree",
        ".comment",
        ".comment-box",
        ".mailing-list-form",
        ".ih-newsletter-cta",
        ".ssi-table-of-contents",
        ".ssi-actions",
        ".ssi-actions-wrapper",
        ".share-button",
    ]:
        for node in root.select(selector):
            node.decompose()
    text = root.get_text(separator="\n", strip=True)
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_likes_and_comments_from_detail_page(html: str) -> Tuple[Optional[int], Optional[int]]:
    likes_count: Optional[int] = None
    comments_count: Optional[int] = None

    likes_match = LIKES_COUNT_RE.search(html)
    if not likes_match:
        likes_match = LIKES_COUNT_ALT_RE.search(html)

    if likes_match:
        try:
            likes_count = int(likes_match.group(1))
        except (ValueError, IndexError):
            likes_count = None

    comments_match = JSON_COMMENTS_RE.search(html)
    if comments_match:
        try:
            comments_count = int(comments_match.group(1))
        except (ValueError, IndexError):
            comments_count = None

    return likes_count, comments_count


def _parse_entry_detail(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int], Optional[int]]:
    try:
        html = _fetch_html(url)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None, None, None, None, None

    title = None
    description = None
    content = None

    title_match = OG_TITLE_RE.search(html)
    if title_match:
        title = _strip(title_match.group(1))

    desc_match = OG_DESC_RE.search(html)
    if desc_match:
        description = _strip(desc_match.group(1))

    content = _extract_post_content(html)
    if not content:
        body_match = ARTICLE_RE.search(html) or POST_CONTENT_RE.search(html)
        if body_match:
            content = _strip(_strip_tags(body_match.group("body")))

    likes_count, comments_count = _extract_likes_and_comments_from_detail_page(html)

    return title, description, content, likes_count, comments_count


def _extract_comments_from_html(html_text: str) -> Optional[int]:
    soup = BeautifulSoup(html_text, "html.parser")
    comments_div = soup.find("div", class_="portal-entry__comments")
    if not comments_div:
        return None
    span = comments_div.find("span")
    if not span:
        return None
    count_text = span.get_text(strip=True)
    if not count_text.isdigit():
        return None
    return int(count_text)


def _extract_comments_from_json(html_text: str, slug: str) -> Optional[int]:
    soup = BeautifulSoup(html_text, "html.parser")
    for link in soup.find_all("a"):
        text = link.get_text(strip=True)
        if "comments" not in text.lower():
            continue
        href = link.get("href") or ""
        if slug not in href:
            continue
        number_match = re.search(r"(\d+)\s*comments", text, re.IGNORECASE)
        if number_match:
            return int(number_match.group(1))
    return None


def _fetch_comments_from_json_page(full_url: str, slug: str) -> Optional[int]:
    try:
        html = _fetch_html(f"{full_url}.json")
    except Exception:
        return None

    soup_comments = _extract_comments_from_json(html, slug)
    if soup_comments is not None:
        return soup_comments

    pattern = re.compile(
        rf'\[(\d+)\s+comments\]\([^\)]*{re.escape(slug)}[^\)]*\)',
        re.IGNORECASE,
    )
    match = pattern.search(html)
    if not match:
        pattern = re.compile(
            rf'\[(\d+)\s+comments\]\([^\)]*{re.escape(full_url)}[^\)]*\)',
            re.IGNORECASE,
        )
        match = pattern.search(html)
    if not match:
        pattern = re.compile(
            rf'(\d+)\s+comments[\s\S]*?{re.escape(slug)}',
            re.IGNORECASE,
        )
        match = pattern.search(html)
    if not match:
        pattern = re.compile(
            rf'<a[^>]+href="[^"]*{re.escape(slug)}[^"]*"[^>]*>\s*(\d+)\s+comments\s*</a>',
            re.IGNORECASE,
        )
        match = pattern.search(html)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _extract_published_at(html_text: str) -> datetime:
    time_match = TIME_RE.search(html_text)
    if time_match:
        return _parse_iso_datetime(time_match.group(1))
    return datetime.now(timezone.utc)


def _parse_entries(
    page_html: str,
    source: str,
    hours_back: int,
    max_items: Optional[int] = None,
    ignore_time_window: bool = False,
) -> List[IngestedItem]:
    items: List[IngestedItem] = []
    cutoff_ts = 0.0
    if not ignore_time_window:
        cutoff_ts = datetime.now(timezone.utc).timestamp() - hours_back * 3600

    for match in ENTRY_RE.finditer(page_html):
        body = match.group("body")
        href = match.group("href")

        date_text = _strip(DATE_RE.search(body).group(1)) if DATE_RE.search(body) else None
        title_html = TITLE_RE.search(body).group(1) if TITLE_RE.search(body) else None
        summary_html = SUMMARY_RE.search(body).group(1) if SUMMARY_RE.search(body) else None
        byline = _strip(BYLINE_RE.search(body).group(1)) if BYLINE_RE.search(body) else None

        comments: Optional[int] = None
        comments_match = COMMENTS_RE.search(body)
        if comments_match:
            try:
                comments = int(comments_match.group(1))
            except ValueError:
                comments = None

        if comments is None:
            soup_comments = _extract_comments_from_html(body)
            if soup_comments is not None:
                comments = soup_comments

        if comments is None:
            text_match = COMMENTS_TEXT_RE.search(body)
            if text_match:
                try:
                    comments = int(text_match.group(1))
                except ValueError:
                    comments = None

        if comments is None:
            idx = page_html.find(href)
            if idx != -1:
                window = page_html[idx: idx + 2500]
                window_match = COMMENTS_RE.search(window) or COMMENTS_TEXT_RE.search(window)
                if window_match:
                    try:
                        comments = int(window_match.group(1))
                    except ValueError:
                        comments = None

        image = IMAGE_RE.search(body).group(1) if IMAGE_RE.search(body) else None

        title = _strip(_strip_tags(title_html or ""))
        summary = _strip(_strip_tags(summary_html or ""))
        created_at = _parse_date(date_text or "")
        if created_at.timestamp() < cutoff_ts:
            continue

        full_url = f"https://www.indiehackers.com{href}"
        slug = href.strip("/").split("/")[-1]

        detail_title, detail_desc, detail_content, detail_likes, detail_comments = _parse_entry_detail(full_url)

        if comments is None and detail_comments is not None:
            comments = detail_comments
        if comments is None:
            json_comments = _fetch_comments_from_json_page(full_url, slug)
            if json_comments is not None:
                comments = json_comments

        final_comments = comments if comments is not None else 0
        final_likes = detail_likes if detail_likes is not None else 0

        categories = categorize_article(
            detail_content or detail_desc or summary or "",
            detail_title or title,
            "indiehackers.com",
            news_type=['Product/Startup']
        )

        items.append(
            IngestedItem(
                source=source,
                source_id=href.strip("/").split("/")[-1],
                url=full_url,
                title=detail_title or title,
                description=detail_desc or summary,
                content=detail_content,
                created_at=created_at,
                engagement={
                    "comments": final_comments,
                    "likes": final_likes,
                    "score": final_comments + final_likes,
                },
                raw={
                    "byline": byline,
                    "comments": final_comments,
                    "likes": final_likes,
                    "image": image,
                    "summary": summary,
                    "detail_comments": detail_comments,
                    "detail_likes": detail_likes,
                    "categories": categories,
                },
            )
        )
        if max_items and len(items) >= max_items:
            break

    return items


def fetch_tech(hours_back: int, limit: Optional[int] = None) -> List[IngestedItem]:
    html = _fetch_html(TECH_URL)
    items = _parse_entries(html, "indiehackers_tech", hours_back, limit)
    if not items:
        items = _parse_entries(html, "indiehackers_tech", hours_back, limit, True)
    return items


def fetch_ai_tag(hours_back: int, limit: Optional[int] = None) -> List[IngestedItem]:
    html = _fetch_html(AI_TAG_URL)
    items = _parse_entries(html, "indiehackers_ai", hours_back, limit)
    if not items:
        items = _parse_entries(html, "indiehackers_ai", hours_back, limit, True)
    return items


def _prefilter(items: List[IngestedItem], keywords: List[str], min_engagement: int) -> List[IngestedItem]:
    if not items:
        return items
    if not keywords and min_engagement <= 0:
        return items
    lowered = [k.lower() for k in keywords]
    kept: List[IngestedItem] = []
    for item in items:
        engagement = 0
        try:
            engagement = int(item.engagement.get("score", 0))
        except Exception:
            engagement = 0
        if engagement < min_engagement:
            continue
        if lowered:
            blob = f"{item.title} {item.description or ''} {item.content or ''}".lower()
            if not any(k in blob for k in lowered):
                continue
        kept.append(item)
    return kept


def _write_outputs(outputs_dir: str, title: str, items: List[IngestedItem]) -> str:
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, "scrape_latest.json")
    md_path = os.path.join(outputs_dir, "scrape_latest.md")

    payload = [item.as_json() for item in items]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [f"# {title}", ""]
    for item in items:
        lines.append(f"## {item.title}")
        lines.append(f"- URL: {item.url}")
        if item.description:
            lines.append(f"- Description: {item.description}")
        comments = item.engagement.get("comments") if isinstance(item.engagement, dict) else None
        likes = item.engagement.get("likes") if isinstance(item.engagement, dict) else None
        if likes is not None:
            lines.append(f"- Likes: {likes}")
        if comments is not None:
            lines.append(f"- Comments: {comments}")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    return json_path


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


def run_indiehackers_pipeline(
    source: str = "tech",
    limit: int = 10,
    hours_back: int = 24,
    keywords: Optional[List[str]] = None,
    min_engagement: int = 0,
    outputs_dir: str = "output_products",
    db_path: str = str(DB_PATH),
    verbose: bool = True
) -> str:
    """Run the complete Indie Hackers scraping and ingestion pipeline."""

    print("=" * 60)
    print("INDIE HACKERS PIPELINE")
    print("=" * 60)

    print(f"\nStep 1: Scraping Indie Hackers ({source})...")

    items: List[IngestedItem] = []
    if source in {"tech", "both"}:
        tech_items = fetch_tech(hours_back, limit)
        items += tech_items
        if verbose:
            print(f"[OK] Found {len(tech_items)} items from tech section")

    if source in {"ai", "both"}:
        ai_items = fetch_ai_tag(hours_back, limit)
        items += ai_items
        if verbose:
            print(f"[OK] Found {len(ai_items)} items from AI tag")

    if limit:
        items = items[:limit]

    print(f"[OK] Total items scraped: {len(items)}")

    if not items:
        print("No items found. Exiting.")
        return ""

    if keywords or min_engagement > 0:
        print(f"\nStep 2: Pre-filtering (keywords: {keywords}, min_engagement: {min_engagement})...")
        original_count = len(items)
        items = _prefilter(items, keywords or [], min_engagement)
        filtered_count = len(items)
        print(f"[OK] {filtered_count}/{original_count} items pass pre-filter")

        if not items:
            print("No items pass pre-filter. Exiting.")
            return ""

    print(f"\nStep 3: Saving {len(items)} items to database...")
    inserted_count = 0
    try:
        inserted_count = save_items_to_db(items, db_path, digest_type='PRODUCT')
        print(f"[OK] Inserted {inserted_count} new items into database")
    except Exception as e:
        print(f"[ERROR] Failed to save to database: {e}")

    print(f"\nStep 4: Saving outputs...")
    output_path = _write_outputs(outputs_dir, "Indie Hackers Scrape", items)
    print(f"[OK] Saved outputs to: {output_path}")

    print(f"\n[OK] Pipeline complete!")
    print(f"[OK] Scraped: {len(items)} items")
    print(f"[OK] Inserted: {inserted_count} new items to database")

    source_counts = {}
    for item in items:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1

    print("\nItems by Source:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count} items")

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
    parser.add_argument('--pipeline', choices=['hackernews', 'indiehackers', 'both'], default='hackernews',
                       help='Which pipeline to run (default: hackernews)')
    parser.add_argument('--pages', type=int, default=3, help='Number of Hacker News pages to scrape (max 20, default: 3)')
    parser.add_argument('--delay', type=float, default=1, help='Delay between article requests (seconds)')
    parser.add_argument('--output', type=str, default='HackerNews/llm_programming_articles.json',
                       help='Hacker News output file path')
    parser.add_argument('--hours-window', type=int, default=DEFAULT_HOURS_WINDOW,
                       help='HN pre-filter: only items from last N hours (default: 24)')
    parser.add_argument('--min-points', type=int, default=DEFAULT_MIN_POINTS,
                       help='HN pre-filter: minimum HN points (default: 1)')
    parser.add_argument('--min-comments', type=int, default=DEFAULT_MIN_COMMENTS,
                       help='HN pre-filter: minimum comment count (default: 0)')
    parser.add_argument('--ih-source', choices=['tech', 'ai', 'both'], default='tech',
                       help='Indie Hackers source (default: tech)')
    parser.add_argument('--ih-limit', type=int, default=10, help='Indie Hackers item limit (default: 10)')
    parser.add_argument('--ih-hours-back', type=int, default=None, help='Indie Hackers hours back (default: env or 24)')
    parser.add_argument('--ih-output-dir', type=str, default='output_products',
                       help='Indie Hackers output directory')
    parser.add_argument('--ih-keywords', type=str, default=None,
                       help='Indie Hackers keyword prefilter (comma-separated)')
    parser.add_argument('--ih-min-engagement', type=int, default=None,
                       help='Indie Hackers minimum engagement (default: env or 0)')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation pipeline after ingestion')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation, skip ingestion')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--model', type=str, default='gemma3:12b', help='Ollama model name')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back for recent items')
    parser.add_argument('--no-reddit', action='store_true', help='Skip Reddit scraper')
    parser.add_argument('--reddit-output-dir', type=str, default='output_reddit', help='Output directory for Reddit JSON')
    parser.add_argument('--reddit-hours-window', type=int, default=REDDIT_TIME_WINDOW_HOURS,
                       help='Reddit: only items from last N hours')
    parser.add_argument('--reddit-limit', type=int, default=REDDIT_LIMIT_PER_SUBREDDIT,
                       help='Reddit: limit posts per subreddit')
    parser.add_argument('--reddit-listing-limit', type=int, default=REDDIT_LISTING_PAGE_LIMIT,
                       help='Reddit: listing page limit')
    parser.add_argument('--reddit-subreddits', type=str, default=",".join(REDDIT_SUBREDDITS),
                       help='Reddit: comma-separated subreddit list')
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
        env = _load_env()

        if not args.no_reddit:
            subreddits = [s.strip() for s in (args.reddit_subreddits or "").split(",") if s.strip()]
            run_reddit_pipeline(
                subreddits=subreddits,
                hours_window=args.reddit_hours_window,
                limit_per_subreddit=args.reddit_limit,
                listing_page_limit=args.reddit_listing_limit,
                output_dir=args.reddit_output_dir,
                verbose=not args.quiet,
                persist_to_db=not args.reddit_no_db,
                db_path=DB_PATH,
            )

        if args.pipeline in {'hackernews', 'both'}:
            run_pipeline(
                num_pages=min(args.pages, 20),
                delay=args.delay,
                verbose=not args.quiet,
                output_file=args.output,
                hours_window=args.hours_window,
                min_points=args.min_points,
                min_comments=args.min_comments,
            )

        if args.pipeline in {'indiehackers', 'both'}:
            hours_back = args.ih_hours_back if args.ih_hours_back is not None else _get_int(
                env.get("HOURS_BACK"), 24
            )
            keywords = _get_list(args.ih_keywords) if args.ih_keywords else _get_list(env.get("PREFILTER_KEYWORDS"))
            min_engagement = args.ih_min_engagement if args.ih_min_engagement is not None else _get_int(
                env.get("MIN_ENGAGEMENT"), 0
            )
            run_indiehackers_pipeline(
                source=args.ih_source,
                limit=args.ih_limit,
                hours_back=hours_back,
                keywords=keywords,
                min_engagement=min_engagement,
                outputs_dir=args.ih_output_dir,
                db_path=str(DB_PATH),
                verbose=not args.quiet
            )
        
        # Run evaluation after ingestion
        print("\n" + "=" * 60)
        run_evaluation_pipeline(
            db_path=DB_PATH,
            ollama_base_url=args.ollama_url,
            model=args.model,
            hours=args.hours,
            verbose=not args.quiet
        )
