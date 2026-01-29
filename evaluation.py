from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from typing import Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

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

PRODUCT_PROMPT = """You are a technical news evaluator for AI/product engineers.
Evaluate the following Indie Hackers article for PRODUCT-related news and return JSON with:
idea_type, problem_statement, solution_summary, maturity_level,
reusability_score (0-1), decision (accept|reject|review).

Scoring guidance:
- Higher scores when the article describes a specific product, MVP, launch, or growth experiment,
  with concrete signals (tech stack, pricing, traction, user outcomes).
- Lower scores for generic founder stories without actionable product signals.
Decision rule: accept if reusability_score >= {min_reusability}, else reject.

Title: {title}
Description: {description}
Content: {content}
"""

GENAI_PROMPT = """You are a technical news evaluator for AI engineers.
Analyze the following tech article and return JSON with:
relevance_score (0-1), topic, why_it_matters, target_audience, decision (accept|reject|review).
Decision rule: accept if relevance_score >= {min_relevance}, else reject.
Target audience must be one of: developer, architect, manager.

Title: {title}
Description: {description}
Content: {content}
"""


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


def _get_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


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


def _parse_entry_detail(url: str) -> tuple[str | None, str | None, str | None, int | None]:
    try:
        html = _fetch_html(url)
    except Exception:
        return None, None, None, None
    title = None
    description = None
    content = None
    comments = None

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

    comments_match = JSON_COMMENTS_RE.search(html)
    if comments_match:
        try:
            comments = int(comments_match.group(1))
        except ValueError:
            comments = None

    return title, description, content, comments


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
        comments_match = COMMENTS_RE.search(body)
        comments = comments_match.group(1) if comments_match else None
        image = IMAGE_RE.search(body).group(1) if IMAGE_RE.search(body) else None

        title = _strip(_strip_tags(title_html or ""))
        summary = _strip(_strip_tags(summary_html or ""))
        created_at = _parse_date(date_text or "")
        if created_at.timestamp() < cutoff_ts:
            continue

        full_url = f"https://www.indiehackers.com{href}"
        detail_title, detail_desc, detail_content, detail_comments = _parse_entry_detail(
            full_url
        )
        if comments is None and detail_comments is not None:
            comments = str(detail_comments)

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
                    "comments": int(comments) if comments else 0,
                    "score": int(comments) if comments else 0,
                },
                raw={
                    "byline": byline,
                    "comments": comments,
                    "image": image,
                    "summary": summary,
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


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            source_id INTEGER,
            source_url TEXT,
            title TEXT NOT NULL,
            description TEXT,
            summary TEXT,
            content TEXT,
            url TEXT,
            published_at DATETIME,
            engagement_score REAL,
            likes INTEGER DEFAULT 0,
            comments INTEGER DEFAULT 0,
            views INTEGER DEFAULT 0,
            raw_metadata JSON,
            ingestion_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'INGESTED',
            FOREIGN KEY (source_id) REFERENCES sources(id)
        );
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL,
            persona TEXT NOT NULL,
            decision TEXT CHECK(decision IN ('KEEP', 'DROP')),
            relevance_score REAL,
            topic TEXT,
            why_it_matters TEXT,
            target_audience TEXT,
            idea_type TEXT,
            problem_statement TEXT,
            solution_summary TEXT,
            maturity_level TEXT,
            reusability_score REAL,
            llm_model TEXT,
            evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            evaluation_type TEXT,
            FOREIGN KEY (item_id) REFERENCES items(id),
            UNIQUE(item_id, persona)
        );
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
        );
        """
    )


def _get_or_create_source(conn: sqlite3.Connection, source: str) -> int:
    conn.execute("INSERT OR IGNORE INTO sources (source) VALUES (?)", (source,))
    row = conn.execute("SELECT id FROM sources WHERE source = ?", (source,)).fetchone()
    return int(row[0])


def _insert_item(conn: sqlite3.Connection, item: IngestedItem, source_pk: int) -> int:
    row = conn.execute(
        "SELECT id FROM items WHERE url = ? AND source = ?",
        (item.url, item.source),
    ).fetchone()
    if row:
        return int(row[0])
    comments = 0
    try:
        comments = int(item.engagement.get("comments", 0))
    except Exception:
        comments = 0
    cur = conn.execute(
        """
        INSERT INTO items (
            source, source_id, source_url, title, description, summary,
            content, url, published_at, engagement_score, likes, comments, views,
            raw_metadata, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.source,
            source_pk,
            item.url,
            item.title,
            item.description,
            None,
            item.content,
            item.url,
            item.created_at.isoformat(),
            float(comments or 0),
            0,
            comments,
            0,
            json.dumps(item.raw),
            "INGESTED",
        ),
    )
    return int(cur.lastrowid)


def _write_outputs(outputs_dir: str, items: List[Dict[str, object]]) -> str:
    os.makedirs(outputs_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(outputs_dir, f"evaluation_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return json_path


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _extract_json(raw: str) -> dict | None:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _llm_chat(base_url: str, model: str, timeout: int, messages: List[Dict[str, str]], max_tokens: int) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": max_tokens}
    url = f"{base_url.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _evaluate_product(
    item: IngestedItem,
    base_url: str,
    model: str,
    timeout: int,
    min_reusability: float,
    max_chars: int,
    max_tokens: int,
) -> Dict[str, object]:
    prompt = PRODUCT_PROMPT.format(
        title=item.title,
        description=item.description or "",
        content=_truncate(item.content or "", max_chars),
        min_reusability=min_reusability,
    )
    raw = _llm_chat(
        base_url,
        model,
        timeout,
        [{"role": "system", "content": "Return ONLY valid JSON."}, {"role": "user", "content": prompt}],
        max_tokens,
    )
    payload = _extract_json(raw) or {}
    score = float(payload.get("reusability_score") or 0.0)
    decision = "KEEP" if score >= min_reusability else "DROP"
    return {
        "idea_type": payload.get("idea_type", "unknown"),
        "problem_statement": payload.get("problem_statement", ""),
        "solution_summary": payload.get("solution_summary", ""),
        "maturity_level": payload.get("maturity_level", "unknown"),
        "reusability_score": score,
        "decision": decision,
    }


def _evaluate_genai(
    item: IngestedItem,
    base_url: str,
    model: str,
    timeout: int,
    min_relevance: float,
    max_chars: int,
    max_tokens: int,
) -> Dict[str, object]:
    prompt = GENAI_PROMPT.format(
        title=item.title,
        description=item.description or "",
        content=_truncate(item.content or "", max_chars),
        min_relevance=min_relevance,
    )
    raw = _llm_chat(
        base_url,
        model,
        timeout,
        [{"role": "system", "content": "Return ONLY valid JSON."}, {"role": "user", "content": prompt}],
        max_tokens,
    )
    payload = _extract_json(raw) or {}
    score = float(payload.get("relevance_score") or 0.0)
    decision = "KEEP" if score >= min_relevance else "DROP"
    return {
        "relevance_score": score,
        "topic": payload.get("topic", "unknown"),
        "why_it_matters": payload.get("why_it_matters", ""),
        "target_audience": payload.get("target_audience", "developer"),
        "decision": decision,
    }


def _insert_evaluation(
    conn: sqlite3.Connection,
    item_id: int,
    persona: str,
    evaluation: Dict[str, object],
    model: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO evaluations (
            item_id, model, evaluated_at, persona, decision, relevance_score, topic,
            why_it_matters, target_audience, idea_type, problem_statement,
            solution_summary, maturity_level, reusability_score, llm_model,
            evaluation_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            model,
            datetime.utcnow().isoformat(),
            persona,
            evaluation.get("decision"),
            evaluation.get("relevance_score"),
            evaluation.get("topic"),
            evaluation.get("why_it_matters"),
            evaluation.get("target_audience"),
            evaluation.get("idea_type"),
            evaluation.get("problem_statement"),
            evaluation.get("solution_summary"),
            evaluation.get("maturity_level"),
            evaluation.get("reusability_score"),
            model,
            "LLM",
        ),
    )


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _tokenize(sentence: str) -> List[str]:
    words = re.findall(r"[A-Za-z0-9']+", sentence.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _sentence_vectors(sentences: List[str]) -> List[Dict[str, int]]:
    vectors: List[Dict[str, int]] = []
    for sent in sentences:
        counts: Dict[str, int] = {}
        for token in _tokenize(sent):
            counts[token] = counts.get(token, 0) + 1
        vectors.append(counts)
    return vectors


def _cosine_similarity(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for key, val in a.items():
        if key in b:
            dot += val * b[key]
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pagerank(similarity: List[List[float]], damping: float = 0.85) -> List[float]:
    n = len(similarity)
    if n == 0:
        return []
    scores = [1.0 / n] * n
    for _ in range(30):
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = 0.0
            for j in range(n):
                if i == j:
                    continue
                row_sum = sum(similarity[j])
                if row_sum == 0:
                    continue
                rank_sum += similarity[j][i] / row_sum * scores[j]
            new_scores[i] = (1 - damping) / n + damping * rank_sum
        scores = new_scores
    return scores


def textrank_summarize(text: str, max_sentences: int = 5) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    vectors = _sentence_vectors(sentences)
    n = len(sentences)
    similarity = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            similarity[i][j] = _cosine_similarity(vectors[i], vectors[j])
    scores = _pagerank(similarity)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top = sorted(ranked[: min(max_sentences, n)])
    return [sentences[i] for i in top]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Indie Hackers items with LLM")
    parser.add_argument("--source", choices=["tech", "ai", "both"], default="tech")
    parser.add_argument("--mode", choices=["product", "genai"], default="product")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--hours-back", type=int, default=None)
    args = parser.parse_args()

    env = _load_env()
    hours_back = args.hours_back if args.hours_back is not None else _get_int(
        env.get("HOURS_BACK"), 24
    )
    sqlite_path = env.get("SQLITE_PATH", "data/db.sqlite")
    outputs_dir = env.get("OUTPUTS_DIR", "outputs")
    base_url = env.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = env.get("OLLAMA_MODEL", "llama3:1")
    timeout = _get_int(env.get("OLLAMA_TIMEOUT_SEC"), 300)
    max_tokens = _get_int(env.get("OLLAMA_MAX_TOKENS"), 512)
    max_chars = _get_int(env.get("EVAL_MAX_CHARS"), 4000)
    textrank_sentences = _get_int(env.get("TEXTRANK_SENTENCES"), 5)
    min_relevance = _get_float(env.get("GENAI_NEWS_MIN_RELEVANCE"), 0.6)
    min_reusability = _get_float(env.get("PRODUCT_IDEAS_MIN_REUSABILITY"), 0.5)

    items: List[IngestedItem] = []
    if args.source in {"tech", "both"}:
        items += fetch_tech(hours_back, args.limit)
    if args.source in {"ai", "both"}:
        items += fetch_ai_tag(hours_back, args.limit)
    if args.limit:
        items = items[: args.limit]

    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    payload: List[Dict[str, object]] = []
    with sqlite3.connect(sqlite_path) as conn:
        _init_db(conn)
        for item in items:
            source_pk = _get_or_create_source(conn, item.source)
            item_id = _insert_item(conn, item, source_pk)
            if args.mode == "product":
                evaluation = _evaluate_product(
                    item, base_url, model, timeout, min_reusability, max_chars, max_tokens
                )
                persona = "PRODUCT_IDEAS"
            else:
                evaluation = _evaluate_genai(
                    item, base_url, model, timeout, min_relevance, max_chars, max_tokens
                )
                persona = "GENAI_NEWS"
            _insert_evaluation(conn, item_id, persona, evaluation, model)

            item_payload = item.as_json()
            summary_lines: List[str] = []
            if item.content:
                summary_lines = textrank_summarize(item.content, textrank_sentences)
                if summary_lines:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO item_summaries (
                            item_id, technical_summary, source
                        ) VALUES (?, ?, ?)
                        """,
                        (item_id, "\n".join(summary_lines), "textrank"),
                    )
            item_payload["evaluation"] = evaluation
            item_payload["textrank_summary"] = summary_lines
            payload.append(item_payload)

    output = _write_outputs(outputs_dir, payload)
    print(output)


if __name__ == "__main__":
    main()
