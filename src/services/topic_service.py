"""
Topic Service: CRUD functions for topic-based preference system.

Manages topics, item-topic mappings, and user topic preferences.
"""
import os
import sqlite3
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv("DB_PATH")


def get_all_topics(db_path: str = None, category: str = None) -> List[Dict]:
    """
    Fetch all topics, optionally filtered by category.
    
    Args:
        db_path: Path to SQLite database (defaults to DB_PATH env var)
        category: Filter by category ('GENAI_NEWS' or 'PRODUCT_IDEAS'), or None for all
        
    Returns:
        List of topic dictionaries with id, name, display_name, category, description, keywords
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        if category:
            cur.execute(
                """SELECT id, name, display_name, category, description, keywords 
                   FROM topics 
                   WHERE category = ? 
                   ORDER BY id""",
                (category,)
            )
        else:
            cur.execute(
                """SELECT id, name, display_name, category, description, keywords 
                   FROM topics 
                   ORDER BY category, id"""
            )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_topic_by_name(name: str, db_path: str = None) -> Optional[Dict]:
    """
    Fetch a single topic by its internal name.
    
    Args:
        name: Internal topic name (e.g., 'llm_research')
        db_path: Path to SQLite database
        
    Returns:
        Topic dictionary or None if not found
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, name, display_name, category, description, keywords 
               FROM topics 
               WHERE name = ?""",
            (name,)
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_topic_names_by_category(category: str, db_path: str = None) -> List[str]:
    """
    Get list of topic names for a given category.
    
    Args:
        category: 'GENAI_NEWS' or 'PRODUCT_IDEAS'
        db_path: Path to SQLite database
        
    Returns:
        List of topic names (internal names)
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM topics WHERE category = ? ORDER BY id",
            (category,)
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def assign_topics_to_item(
    item_id: int,
    topic_names: List[str],
    confidence_scores: Dict[str, float] = None,
    db_path: str = None
) -> int:
    """
    Assign topics to an item (many-to-many relationship).
    Presence of a row = topic assigned (boolean true).
    
    Args:
        item_id: ID of the item in the items table
        topic_names: List of topic internal names (e.g., ['llm_research', 'ai_tools'])
        confidence_scores: Ignored (kept for backward compatibility)
        db_path: Path to SQLite database
        
    Returns:
        Number of topic assignments created
    """
    if not topic_names:
        return 0
    
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        assigned_count = 0
        
        for topic_name in topic_names:
            # Get topic_id from topic name
            cur.execute("SELECT id FROM topics WHERE name = ?", (topic_name,))
            row = cur.fetchone()
            if not row:
                print(f"[WARN] Topic '{topic_name}' not found, skipping")
                continue
            
            topic_id = row[0]
            
            # Insert item_topics mapping (presence = assigned/true)
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
        print(f"[ERROR] Failed to assign topics to item {item_id}: {e}")
        return 0
    finally:
        conn.close()


def get_item_topics(item_id: int, db_path: str = None) -> List[Dict]:
    """
    Get all topics assigned to an item.
    
    Args:
        item_id: ID of the item
        db_path: Path to SQLite database
        
    Returns:
        List of topic dictionaries with topic info
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT t.id, t.name, t.display_name, t.category, it.assigned_at
               FROM item_topics it
               JOIN topics t ON it.topic_id = t.id
               WHERE it.item_id = ?
               ORDER BY t.id""",
            (item_id,)
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_user_preferences(user_id: int, db_path: str = None) -> List[Dict]:
    """
    Get all topic preferences for a user.
    
    Args:
        user_id: ID of the user
        db_path: Path to SQLite database
        
    Returns:
        List of preference dictionaries with topic info and score
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT t.id as topic_id, t.name, t.display_name, t.category,
                      COALESCE(utp.score, 0) as score, utp.updated_at
               FROM topics t
               LEFT JOIN user_topic_preferences utp 
                   ON t.id = utp.topic_id AND utp.user_id = ?
               ORDER BY t.category, utp.score DESC, t.id""",
            (user_id,)
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_user_top_topics(
    user_id: int,
    category: str = None,
    limit: int = 5,
    db_path: str = None
) -> List[Dict]:
    """
    Get user's top preferred topics by score.
    
    Args:
        user_id: ID of the user
        category: Optional category filter ('GENAI_NEWS' or 'PRODUCT_IDEAS')
        limit: Maximum number of topics to return
        db_path: Path to SQLite database
        
    Returns:
        List of top topic preferences ordered by score descending
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        if category:
            cur.execute(
                """SELECT t.id as topic_id, t.name, t.display_name, t.category,
                          COALESCE(utp.score, 0) as score
                   FROM topics t
                   LEFT JOIN user_topic_preferences utp 
                       ON t.id = utp.topic_id AND utp.user_id = ?
                   WHERE t.category = ?
                   ORDER BY score DESC, t.id
                   LIMIT ?""",
                (user_id, category, limit)
            )
        else:
            cur.execute(
                """SELECT t.id as topic_id, t.name, t.display_name, t.category,
                          COALESCE(utp.score, 0) as score
                   FROM topics t
                   LEFT JOIN user_topic_preferences utp 
                       ON t.id = utp.topic_id AND utp.user_id = ?
                   ORDER BY score DESC, t.id
                   LIMIT ?""",
                (user_id, limit)
            )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def increment_user_preference(
    user_id: int,
    topic_id: int,
    delta: int = 1,
    db_path: str = None
) -> bool:
    """
    Increment (or decrement) a user's preference score for a topic.
    
    Args:
        user_id: ID of the user
        topic_id: ID of the topic
        delta: Amount to change score (positive for like, negative for dislike)
        db_path: Path to SQLite database
        
    Returns:
        True if successful, False otherwise
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Use UPSERT pattern: insert if not exists, update if exists
        cur.execute(
            """INSERT INTO user_topic_preferences (user_id, topic_id, score, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, topic_id) DO UPDATE SET
                   score = score + excluded.score,
                   updated_at = CURRENT_TIMESTAMP""",
            (user_id, topic_id, delta)
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to update preference for user {user_id}, topic {topic_id}: {e}")
        return False
    finally:
        conn.close()


def increment_user_preference_by_name(
    user_id: int,
    topic_name: str,
    delta: int = 1,
    db_path: str = None
) -> bool:
    """
    Increment user preference score using topic name instead of ID.
    
    Args:
        user_id: ID of the user
        topic_name: Internal name of the topic (e.g., 'llm_research')
        delta: Amount to change score
        db_path: Path to SQLite database
        
    Returns:
        True if successful, False otherwise
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Get topic_id
        cur.execute("SELECT id FROM topics WHERE name = ?", (topic_name,))
        row = cur.fetchone()
        if not row:
            print(f"[WARN] Topic '{topic_name}' not found")
            return False
        topic_id = row[0]
        conn.close()
        
        # Call the ID-based function
        return increment_user_preference(user_id, topic_id, delta, db_path)
    except Exception as e:
        conn.close()
        print(f"[ERROR] Failed to update preference: {e}")
        return False


def initialize_user_preferences(user_id: int, db_path: str = None) -> int:
    """
    Initialize topic preferences for a new user (all topics with score=0).
    
    Args:
        user_id: ID of the user
        db_path: Path to SQLite database
        
    Returns:
        Number of preference records created
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Get all topic IDs
        cur.execute("SELECT id FROM topics")
        topic_ids = [row[0] for row in cur.fetchall()]
        
        # Insert preference records with score=0 (ignore if already exists)
        created_count = 0
        for topic_id in topic_ids:
            cur.execute(
                """INSERT OR IGNORE INTO user_topic_preferences (user_id, topic_id, score, updated_at)
                   VALUES (?, ?, 0, CURRENT_TIMESTAMP)""",
                (user_id, topic_id)
            )
            if cur.rowcount > 0:
                created_count += 1
        
        conn.commit()
        return created_count
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to initialize preferences for user {user_id}: {e}")
        return 0
    finally:
        conn.close()


def update_preferences_from_item_interaction(
    user_id: int,
    item_id: int,
    delta: int = 1,
    db_path: str = None
) -> int:
    """
    Update user preferences based on interaction with an item.
    Increments scores for all topics associated with the item.
    
    Args:
        user_id: ID of the user
        item_id: ID of the item the user interacted with
        delta: Amount to change scores (+1 for like, -1 for dislike)
        db_path: Path to SQLite database
        
    Returns:
        Number of topic preferences updated
    """
    db_path = db_path or DB_PATH
    
    # Get topics for the item
    item_topics = get_item_topics(item_id, db_path)
    if not item_topics:
        return 0
    
    updated_count = 0
    for topic in item_topics:
        if increment_user_preference(user_id, topic['id'], delta, db_path):
            updated_count += 1
    
    return updated_count


# Utility function for LLM prompt generation
def get_topic_names_for_prompt(category: str = None, db_path: str = None) -> str:
    """
    Get formatted topic names for use in LLM prompts.
    
    Args:
        category: Optional category filter
        db_path: Path to SQLite database
        
    Returns:
        Comma-separated string of topic names
    """
    topics = get_all_topics(db_path, category)
    return ", ".join(t['name'] for t in topics)


if __name__ == "__main__":
    # Simple test/demo
    import argparse
    
    parser = argparse.ArgumentParser(description="Topic Service Demo")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Database path")
    parser.add_argument("--list", action="store_true", help="List all topics")
    parser.add_argument("--category", type=str, help="Filter by category")
    
    args = parser.parse_args()
    
    if args.list:
        topics = get_all_topics(args.db, args.category)
        print(f"\nTopics ({len(topics)} total):")
        print("-" * 50)
        for t in topics:
            print(f"  [{t['id']:2}] {t['name']:20} ({t['category']})")
        print()
