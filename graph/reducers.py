from __future__ import annotations


def dedup_append(existing: list[dict], new: list[dict]) -> list[dict]:
    """Append events from new only if their event_key is not already in existing."""
    seen = {e["event_key"] for e in existing}
    return existing + [e for e in new if e["event_key"] not in seen]


def latest_by_timestamp(existing: dict[str, dict], new: dict[str, dict]) -> dict[str, dict]:
    """Per-key, keep the entry with the newer fetched_at string (ISO 8601 lexicographic)."""
    result = dict(existing)
    for k, v in new.items():
        if k not in result or v["fetched_at"] > result[k]["fetched_at"]:
            result[k] = v
    return result


def merge_by_key(existing: dict, new: dict) -> dict:
    """Merge dicts; new keys win on conflict."""
    return {**existing, **new}
