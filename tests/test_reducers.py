import pytest
from datetime import datetime, timezone
from hypothesis import given, settings, strategies as st
from graph.reducers import dedup_append, latest_by_timestamp, merge_by_key


def make_event(key: str) -> dict:
    return {"event_key": key, "provider": "test", "data": key}


def make_snapshot(key: str, ts: str) -> dict:
    return {key: {"fetched_at": ts, "value": key}}


# --- dedup_append ---

class TestDedupAppend:
    def test_appends_new_event(self):
        existing = [make_event("aaa")]
        new = [make_event("bbb")]
        result = dedup_append(existing, new)
        assert len(result) == 2

    def test_does_not_duplicate_existing_key(self):
        existing = [make_event("aaa")]
        new = [make_event("aaa")]
        result = dedup_append(existing, new)
        assert len(result) == 1

    def test_empty_existing(self):
        result = dedup_append([], [make_event("aaa")])
        assert len(result) == 1

    @given(
        keys=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20)
    )
    def test_idempotent(self, keys):
        events = [make_event(k) for k in set(keys)]
        once = dedup_append([], events)
        twice = dedup_append(once, events)
        assert len(once) == len(twice)


# --- latest_by_timestamp ---

class TestLatestByTimestamp:
    def test_newer_entry_wins(self):
        existing = {"k": {"fetched_at": "2026-04-26T10:00:00", "value": "old"}}
        new = {"k": {"fetched_at": "2026-04-26T11:00:00", "value": "new"}}
        result = latest_by_timestamp(existing, new)
        assert result["k"]["value"] == "new"

    def test_older_entry_does_not_overwrite(self):
        existing = {"k": {"fetched_at": "2026-04-26T11:00:00", "value": "new"}}
        new = {"k": {"fetched_at": "2026-04-26T10:00:00", "value": "old"}}
        result = latest_by_timestamp(existing, new)
        assert result["k"]["value"] == "new"

    def test_new_key_is_added(self):
        existing = {"a": {"fetched_at": "2026-04-26T10:00:00", "value": "a"}}
        new = {"b": {"fetched_at": "2026-04-26T10:00:00", "value": "b"}}
        result = latest_by_timestamp(existing, new)
        assert "a" in result and "b" in result

    @given(
        ts_a=st.integers(min_value=0, max_value=1_000_000),
        ts_b=st.integers(min_value=0, max_value=1_000_000),
    )
    def test_commutative(self, ts_a, ts_b):
        # Commutativity is only guaranteed when timestamps differ.
        # When ts_a == ts_b, existing wins by design (strict > comparison).
        from hypothesis import assume
        assume(ts_a != ts_b)
        snap_a = {"k": {"fetched_at": f"2026-01-01T{ts_a:06d}", "value": "a"}}
        snap_b = {"k": {"fetched_at": f"2026-01-01T{ts_b:06d}", "value": "b"}}
        r1 = latest_by_timestamp(snap_a, snap_b)
        r2 = latest_by_timestamp(snap_b, snap_a)
        assert r1["k"]["value"] == r2["k"]["value"]


# --- merge_by_key ---

class TestMergeByKey:
    def test_new_key_wins_on_conflict(self):
        result = merge_by_key({"a": [1]}, {"a": [2]})
        assert result["a"] == [2]

    def test_non_conflicting_keys_preserved(self):
        result = merge_by_key({"a": [1]}, {"b": [2]})
        assert "a" in result and "b" in result

    def test_empty_existing(self):
        result = merge_by_key({}, {"a": [1]})
        assert result == {"a": [1]}

    @given(
        keys=st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=10)
    )
    def test_associative(self, keys):
        d1 = {k: [1] for k in keys[:len(keys)//2]}
        d2 = {k: [2] for k in keys[len(keys)//2:]}
        d3 = {k: [3] for k in keys}
        r1 = merge_by_key(merge_by_key(d1, d2), d3)
        r2 = merge_by_key(d1, merge_by_key(d2, d3))
        assert r1 == r2
