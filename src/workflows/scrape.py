from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from typing import Dict, List, Optional
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from src.tools.db_utils import save_items_to_db, categorize_article

load_dotenv()

TECH_URL = "https://www.indiehackers.com/tech"
AI_TAG_URL = "https://www.indiehackers.com/tags/artificial-intelligence"

# Database path from environment
DB_PATH = Path(os.getenv('DB_PATH', 'mydb.db'))

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

# Regex for likes count from the detail page
LIKES_COUNT_RE = re.compile(r'<div[^>]+class="[^"]*post-liker__count[^"]*"[^>]*>(\d+)</div>', re.IGNORECASE)
LIKES_COUNT_ALT_RE = re.compile(r'class="post-liker__count"[^>]*>(\d+)<', re.IGNORECASE)


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


def _parse_iso_datetime(text: str | None) -> datetime:
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


def _extract_post_content(html_text: str) -> str | None:
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


def _extract_likes_and_comments_from_detail_page(html: str) -> tuple[int, int]:
    """
    Extract likes and comments count from the detail page HTML.
    Returns (likes_count, comments_count)
    """
    likes_count = 0
    comments_count = 0
    
    # Extract likes count using the specific class structure you showed
    likes_match = LIKES_COUNT_RE.search(html)
    if not likes_match:
        likes_match = LIKES_COUNT_ALT_RE.search(html)
    
    if likes_match:
        try:
            likes_count = int(likes_match.group(1))
        except (ValueError, IndexError):
            likes_count = 0
    
    # Extract comments count from JSON data in the page
    comments_match = JSON_COMMENTS_RE.search(html)
    if comments_match:
        try:
            comments_count = int(comments_match.group(1))
        except (ValueError, IndexError):
            comments_count = 0
    
    return likes_count, comments_count


def _parse_entry_detail(url: str) -> tuple[str | None, str | None, str | None, int, int]:
    """
    Parse entry detail page and return title, description, content, likes_count, comments_count
    """
    try:
        html = _fetch_html(url)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None, None, None, 0, 0
    
    title = None
    description = None
    content = None

    # Extract title
    title_match = OG_TITLE_RE.search(html)
    if title_match:
        title = _strip(title_match.group(1))

    # Extract description
    desc_match = OG_DESC_RE.search(html)
    if desc_match:
        description = _strip(desc_match.group(1))

    # Extract content
    content = _extract_post_content(html)
    if not content:
        body_match = ARTICLE_RE.search(html) or POST_CONTENT_RE.search(html)
        if body_match:
            content = _strip(_strip_tags(body_match.group("body")))

    # Extract likes and comments from detail page
    likes_count, comments_count = _extract_likes_and_comments_from_detail_page(html)

    return title, description, content, likes_count, comments_count


def _extract_published_at(html_text: str) -> datetime:
    time_match = TIME_RE.search(html_text)
    if time_match:
        return _parse_iso_datetime(time_match.group(1))
    return datetime.now(timezone.utc)


def _parse_entries(
    page_html: str,
    source: str,
    hours_back: int,
    max_items: int | None = None,
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
        
        # Try to extract comments from listing page
        comments_match = COMMENTS_RE.search(body)
        listing_comments = 0
        if comments_match:
            try:
                listing_comments = int(comments_match.group(1))
            except ValueError:
                listing_comments = 0
        
        # Alternative comment extraction
        if listing_comments == 0:
            text_match = COMMENTS_TEXT_RE.search(body)
            if text_match:
                try:
                    listing_comments = int(text_match.group(1))
                except ValueError:
                    listing_comments = 0
        
        image = IMAGE_RE.search(body).group(1) if IMAGE_RE.search(body) else None

        title = _strip(_strip_tags(title_html or ""))
        summary = _strip(_strip_tags(summary_html or ""))
        created_at = _parse_date(date_text or "")
        if created_at.timestamp() < cutoff_ts:
            continue

        full_url = f"https://www.indiehackers.com{href}"
        
        # Get detailed information from the post page including likes and comments
        detail_title, detail_desc, detail_content, detail_likes, detail_comments = _parse_entry_detail(full_url)
        
        # Use detail page data if available, otherwise fall back to listing page data
        final_comments = detail_comments if detail_comments > 0 else listing_comments
        final_likes = detail_likes  # Likes only come from detail page

        print(f"[DEBUG] {title[:50]}... - Likes: {final_likes}, Comments: {final_comments}")

        # Categorize article by Product/Startup keywords
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
                    "score": final_comments + final_likes,  # Combined engagement score
                },
                raw={
                    "byline": byline,
                    "comments": final_comments,
                    "likes": final_likes,
                    "image": image,
                    "summary": summary,
                    "listing_comments": listing_comments,
                    "detail_comments": detail_comments,
                    "detail_likes": detail_likes,
                    "categories": categories,
                },
            )
        )
        if max_items and len(items) >= max_items:
            break

    return items


def fetch_tech(hours_back: int, limit: int | None = None) -> List[IngestedItem]:
    html = _fetch_html(TECH_URL)
    items = _parse_entries(html, "indiehackers_tech", hours_back, limit)
    if not items:
        items = _parse_entries(html, "indiehackers_tech", hours_back, limit, True)
    return items


def fetch_ai_tag(hours_back: int, limit: int | None = None) -> List[IngestedItem]:
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
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(outputs_dir, f"scrape_{ts}.json")
    md_path = os.path.join(outputs_dir, f"scrape_{ts}.md")

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


def run_pipeline(
    source: str = "tech",
    limit: int = 10,
    hours_back: int = 24,
    keywords: List[str] = None,
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
    
    # Step 2: Apply pre-filtering
    if keywords or min_engagement > 0:
        print(f"\nStep 2: Pre-filtering (keywords: {keywords}, min_engagement: {min_engagement})...")
        original_count = len(items)
        items = _prefilter(items, keywords or [], min_engagement)
        filtered_count = len(items)
        print(f"[OK] {filtered_count}/{original_count} items pass pre-filter")
        
        if not items:
            print("No items pass pre-filter. Exiting.")
            return ""
    
    # Step 3: Save to database
    print(f"\nStep 3: Saving {len(items)} items to database...")
    try:
        inserted_count = save_items_to_db(items, db_path, digest_type='PRODUCT')
        print(f"[OK] Inserted {inserted_count} new items into database")
    except Exception as e:
        print(f"[ERROR] Failed to save to database: {e}")
    
    # Step 4: Save to output files
    print(f"\nStep 4: Saving outputs...")
    output_path = _write_outputs(outputs_dir, "Indie Hackers Scrape", items)
    print(f"[OK] Saved outputs to: {output_path}")
    
    # Summary
    print(f"\n[OK] Pipeline complete!")
    print(f"[OK] Scraped: {len(items)} items")
    print(f"[OK] Inserted: {inserted_count} new items to database")
    
    # Summary by source
    source_counts = {}
    
    for item in items:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
    
    print("\nItems by Source:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count} items")
    
    
    
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Indie Hackers data and save to database")
    parser.add_argument("--source", choices=["tech", "ai", "both"], default="tech")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--hours-back", type=int, default=None)
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database file path")
    parser.add_argument("--output-dir", type=str, default="output_products", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbose output")
    args = parser.parse_args()

    env = _load_env()
    hours_back = args.hours_back if args.hours_back is not None else _get_int(
        env.get("HOURS_BACK"), 24
    )
    keywords = _get_list(env.get("PREFILTER_KEYWORDS"))
    min_engagement = _get_int(env.get("MIN_ENGAGEMENT"), 0)

    run_pipeline(
        source=args.source,
        limit=args.limit,
        hours_back=hours_back,
        keywords=keywords,
        min_engagement=min_engagement,
        outputs_dir=args.output_dir,
        db_path=args.db,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()