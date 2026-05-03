from __future__ import annotations
import sqlite3
import pytest
from datetime import datetime
from unittest.mock import patch
from models.disruption import DisruptionEvent, DisruptionSeverity
from workers.queue import enqueue, dequeue_pending, mark_processed, _init_db

DB_PATH = ":memory:"


@pytest.fixture
def conn():
    c = sqlite3.connect(DB_PATH)
    _init_db(c)
    yield c
    c.close()


def _make_event(key: str = "key_abc") -> DisruptionEvent:
    return DisruptionEvent(
        event_key=key,
        provider="duffel",
        entity_id="seg_001",
        status_code="cancelled",
        severity=DisruptionSeverity.CRITICAL,
        detected_at=datetime(2026, 5, 10, 12, 0, 0),
        raw_payload={},
    )


def test_enqueue_then_dequeue(conn):
    with patch("workers.queue._get_conn", return_value=conn):
        enqueue(_make_event("key_1"))
        results = dequeue_pending()
    assert len(results) == 1
    assert results[0].event_key == "key_1"
    assert results[0].severity == DisruptionSeverity.CRITICAL


def test_enqueue_is_idempotent(conn):
    with patch("workers.queue._get_conn", return_value=conn):
        enqueue(_make_event("key_dup"))
        enqueue(_make_event("key_dup"))
        results = dequeue_pending()
    assert len(results) == 1


def test_mark_processed_hides_from_dequeue(conn):
    with patch("workers.queue._get_conn", return_value=conn):
        enqueue(_make_event("key_proc"))
        mark_processed("key_proc")
        results = dequeue_pending()
    assert results == []


def test_dequeue_returns_only_unprocessed(conn):
    with patch("workers.queue._get_conn", return_value=conn):
        enqueue(_make_event("key_a"))
        enqueue(_make_event("key_b"))
        mark_processed("key_a")
        results = dequeue_pending()
    assert len(results) == 1
    assert results[0].event_key == "key_b"
