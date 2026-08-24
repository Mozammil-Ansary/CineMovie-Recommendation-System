"""Reusable preprocessing helpers for TMDB metadata."""

import json
from typing import Any


def parse_json_records(value: Any) -> list[dict]:
    """Safely parse a TMDB JSON-like string into a list of dictionaries."""
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def parse_json_names(value: Any) -> list[str]:
    """Extract the `name` field from a TMDB JSON-like list."""
    return [item["name"] for item in parse_json_records(value) if item.get("name")]
