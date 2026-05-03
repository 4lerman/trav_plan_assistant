from __future__ import annotations
import os
import sqlite3
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from models.disruption import NormalisedEvent, DisruptionSeverity
from workers.queue import _init_db


def _make_normalised(provider: str, status_code: str) -> NormalisedEvent:
    return NormalisedEvent(
        provider=provider,
        entity_id="ent_001",
        status_code=status_code,
        window="2026-05-10",
        raw_payload={},
    )


@pytest.fixture(autouse=True)
def in_memory_queue():
    """Redirect queue to in-memory SQLite for all worker tests."""
    conn = sqlite3.connect(":memory:")
    _init_db(conn)
    with patch("workers.queue._get_conn", return_value=conn):
        yield conn
    conn.close()


def test_critical_event_is_enqueued_and_wake_signal_written(tmp_path):
    wake_file = tmp_path / ".wake_signal"

    fake_itinerary = MagicMock()
    fake_itinerary.stops = []

    fake_profile = MagicMock()

    with patch("workers.live_data_worker.WAKE_SIGNAL_PATH", str(wake_file)), \
         patch("workers.live_data_worker._load_entity_ids_from_checkpoint", return_value={"duffel": ["seg_1"], "wheelmap": []}), \
         patch("workers.live_data_worker._load_profile_from_checkpoint", return_value=fake_profile), \
         patch("workers.providers.duffel.poll", return_value=[_make_normalised("duffel", "cancelled")]), \
         patch("workers.providers.wheelmap.poll", return_value=[]):

        from workers.live_data_worker import _run_once
        _run_once()

    from workers.queue import dequeue_pending
    events = dequeue_pending()
    assert len(events) == 1
    assert events[0].provider == "duffel"
    assert events[0].status_code == "cancelled"
    assert events[0].severity == DisruptionSeverity.CRITICAL
    assert wake_file.exists()


def test_none_severity_event_is_not_enqueued(tmp_path):
    wake_file = tmp_path / ".wake_signal"
    fake_profile = MagicMock()

    with patch("workers.live_data_worker.WAKE_SIGNAL_PATH", str(wake_file)), \
         patch("workers.live_data_worker._load_entity_ids_from_checkpoint", return_value={"duffel": [], "wheelmap": ["node_1"]}), \
         patch("workers.live_data_worker._load_profile_from_checkpoint", return_value=fake_profile), \
         patch("workers.providers.duffel.poll", return_value=[]), \
         patch("workers.providers.wheelmap.poll", return_value=[_make_normalised("wheelmap", "accessible")]):

        from workers.live_data_worker import _run_once
        _run_once()

    from workers.queue import dequeue_pending
    assert dequeue_pending() == []
    assert not wake_file.exists()


def test_provider_exception_does_not_crash_worker(tmp_path):
    wake_file = tmp_path / ".wake_signal"
    fake_profile = MagicMock()

    with patch("workers.live_data_worker.WAKE_SIGNAL_PATH", str(wake_file)), \
         patch("workers.live_data_worker._load_entity_ids_from_checkpoint", return_value={"duffel": ["seg_1"], "wheelmap": []}), \
         patch("workers.live_data_worker._load_profile_from_checkpoint", return_value=fake_profile), \
         patch("workers.providers.duffel.poll", side_effect=Exception("network down")), \
         patch("workers.providers.wheelmap.poll", return_value=[]):

        from workers.live_data_worker import _run_once
        _run_once()  # must not raise
