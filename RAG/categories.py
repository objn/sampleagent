"""Shared category validation for uploads and scoped retrieval."""

import re


def normalize_category(category: str) -> str:
    value = category.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", value):
        raise ValueError("Category must be 1-63 letters, digits, underscores, or hyphens")
    return value
