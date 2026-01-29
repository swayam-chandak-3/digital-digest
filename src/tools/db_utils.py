"""Database utilities for managing items and evaluations."""

import json
import re
import sqlite3
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from scrape import IngestedItem


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
    ],
    'Product/Startup': [
        # Launch & announcements
        'product', 'startup', 'launch', 'released', 'announcement', 'introducing',
        'shipping', 'rollout', 'beta', 'alpha', 'early access',
        # Product building
        'mvp', 'prototype', 'side project', 'build', 'building', 'maker',
        'indie hacker', 'bootstrapped', 'solopreneur',
        # SaaS & tools
        'saas', 'platform', 'tool', 'service', 'app', 'dashboard',
        'subscription', 'pricing', 'freemium',
        # Problem / solution framing
        'problem', 'pain point', 'solution', 'workflow', 'use case',
        'customer', 'user feedback',
        # Traction & growth signals
        'users', 'customers', 'revenue', 'growth', 'traction', 'churn',
        'conversion', 'retention', 'metrics',
    ]
}


def categorize_article(text: str, title: str = '', domain: str = '', news_type: List[str] = None) -> List[str]:
    """Categorize an article based on its content.
    
    Args:
        text: Article content text
        title: Article title
        domain: Domain name
        news_type: List of category names to filter by (e.g., ['LLM/AI', 'Programming/Software'])
                   If None, check all categories
    
    Returns:
        List of matching categories
    """
    if not text:
        return []
    
    full_text = f"{title} {text}".lower()
    category_scores = {}
    
    # Determine which categories to check
    categories_to_check = news_type if news_type else list(TOPIC_KEYWORDS.keys())
    
    for category in categories_to_check:
        if category not in TOPIC_KEYWORDS:
            continue
            
        keywords = TOPIC_KEYWORDS[category]
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

def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize database tables if they don't exist."""
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
        """
    )


def _get_or_create_source(conn: sqlite3.Connection, source: str) -> int:
    """Get or create a source and return its ID."""
    conn.execute("INSERT OR IGNORE INTO sources (source) VALUES (?)", (source,))
    row = conn.execute("SELECT id FROM sources WHERE source = ?", (source,)).fetchone()
    return int(row[0])


def save_items_to_db(items, db_path: str, digest_type: str = None) -> int:
    """Save items to the database and return the number of items inserted.
    
    Unified function that handles both IngestedItem objects and article dictionaries.
    Deduplicates by title to avoid duplicate content.
    Adds ingestion_time as CURRENT_TIMESTAMP.
    
    Args:
        items: List of IngestedItem objects or article dictionaries to save
        db_path: Path to the SQLite database file
        digest_type: Type of digest ('GENAI' for AI-generated news, 'PRODUCT' for product/startup news)
        
    Returns:
        Number of items successfully inserted into the database
    """
    if not items:
        return 0

    conn = sqlite3.connect(db_path)
    try:
        _init_db(conn)
        
        # Fetch existing titles once to avoid duplicates
        cur = conn.cursor()
        cur.execute("SELECT title FROM items")
        existing_titles = {row[0] for row in cur.fetchall() if row[0]}
        
        inserted_count = 0
        skipped_count = 0
        
        for item in items:
            # Determine if item is an IngestedItem object or dictionary
            is_dict = isinstance(item, dict)
            
            # Extract common fields
            title = item.get('title') if is_dict else item.title
            
            # Skip if title already exists
            if title in existing_titles:
                skipped_count += 1
                continue
            
            # Extract source
            source_name = item.get('source') if is_dict else item.source
            
            # Get or create source
            source_id = _get_or_create_source(conn, source_name)
            
            # Extract fields based on item type
            if is_dict:
                # Dictionary (article from Hacker News, etc.)
                link = item.get('link')
                description = None  # description not available in articles
                summary = item.get('summary')
                content = item.get('content')
                published_at = item.get('publish_datetime')
                engagement_score = item.get('engagement_score')
                likes = item.get('points', 0)
                comments = item.get('num_comments', 0)
                status = 'PREFILTERED'
                raw_metadata = item.copy()
            else:
                # IngestedItem object
                link = item.url
                description = item.description
                summary = None  # will be filled later by TextRank
                content = item.content
                published_at = item.created_at.isoformat()
                likes = 0
                comments = 0
                try:
                    comments = int(item.engagement.get("comments", 0))
                    likes = int(item.engagement.get("likes", 0))
                except Exception:
                    pass
                engagement_score = float(comments + likes)
                status = 'INGESTED'
                raw_metadata = item.raw.copy()
            
            # Add digest_type to metadata
            if digest_type:
                raw_metadata['digest_type'] = digest_type
            
            # Insert item into database
            cur.execute(
                """
                INSERT INTO items (
                    source,
                    source_id,
                    source_url,
                    title,
                    description,
                    summary,
                    content,
                    url,
                    published_at,
                    engagement_score,
                    likes,
                    comments,
                    raw_metadata,
                    status,
                    ingestion_time,
                    digest_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (
                    source_name,
                    source_id,
                    link,
                    title,
                    description,
                    summary,
                    content,
                    link,
                    published_at,
                    engagement_score,
                    likes,
                    comments,
                    json.dumps(raw_metadata),
                    status,
                    digest_type
                ),
            )
            
            inserted_count += 1
            existing_titles.add(title)  # prevent duplicates in same run
        
        conn.commit()
        
        if skipped_count > 0:
            print(f"[INFO] Skipped {skipped_count} duplicate items")
        
        return inserted_count
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to save items to database: {e}")
        return 0
    finally:
        conn.close()