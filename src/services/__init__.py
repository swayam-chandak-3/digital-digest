"""Services package for digital-digest."""

from .topic_service import (
    get_all_topics,
    get_topic_by_name,
    get_topic_names_by_category,
    assign_topics_to_item,
    get_item_topics,
    get_user_preferences,
    get_user_top_topics,
    increment_user_preference,
    increment_user_preference_by_name,
    initialize_user_preferences,
    update_preferences_from_item_interaction,
    get_topic_names_for_prompt,
)

__all__ = [
    "get_all_topics",
    "get_topic_by_name",
    "get_topic_names_by_category",
    "assign_topics_to_item",
    "get_item_topics",
    "get_user_preferences",
    "get_user_top_topics",
    "increment_user_preference",
    "increment_user_preference_by_name",
    "initialize_user_preferences",
    "update_preferences_from_item_interaction",
    "get_topic_names_for_prompt",
]
