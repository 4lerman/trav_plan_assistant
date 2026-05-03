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
    conn = sqlite3.connect(":memory:")
    _init_db(conn)
    with patch("workers.queue._get_conn", return_value=conn):
        yield conn
    conn.close()

def test_critical_event_is_enqueued_and_wake_signal_written(tmp_path):
    wake_file = tmp_path / ".wake_signal"
    with patch("workers.live_data_worker.WAKE_SIGNAL_PATH", str(wake_file)), \
         patch("workers.live_data_worker._load_entity_ids_from_checkpoint", return_value={"aviationstack": ["seg_1"], "weather": [], "advisories": []}), \
         patch("workers.live_data_worker._load_profile_from_checkpoint", return_value=MagicMock()), \
         patch("workers.providers.aviationstack.poll", return_value=[_make_normalised("aviationstack", "cancelled")]), \
         patch("workers.providers.weather.poll", return_value=[]), \
         patch("workers.providers.advisories.poll", return_value=[]):
        from workers.live_data_worker import _run_once
        _run_once()
    from workers.queue import dequeue_pending
    events = dequeue_pending()
    assert len(events) == 1
    assert events[0].status_code == "cancelled"
    assert events[0].severity == DisruptionSeverity.CRITICAL

def test_none_severity_event_is_not_enqueued(tmp_path):
    wake_file = tmp_path / ".wake_signal"
    with patch("workers.live_data_worker.WAKE_SIGNAL_PATH", str(wake_file)), \
         patch("workers.live_data_worker._load_entity_ids_from_checkpoint", return_value={"aviationstack": [], "weather": ["London"], "advisories": []}), \
         patch("workers.live_data_worker._load_profile_from_checkpoint", return_value=MagicMock()), \
         patch("workers.providers.aviationstack.poll", return_value=[]), \
         patch("workers.providers.weather.poll", return_value=[_make_normalised("weather", "nice_weather")]), \
         patch("workers.providers.advisories.poll", return_value=[]):
        from workers.live_data_worker import _run_once
        _run_once()
    from workers.queue import dequeue_pending
    assert dequeue_pending() == []

def test_provider_exception_does_not_crash_worker(tmp_path):
    wake_file = tmp_path / ".wake_signal"
    with patch("workers.live_data_worker.WAKE_SIGNAL_PATH", str(wake_file)), \
         patch("workers.live_data_worker._load_entity_ids_from_checkpoint", return_value={"aviationstack": ["seg_1"], "weather": [], "advisories": []}), \
         patch("workers.live_data_worker._load_profile_from_checkpoint", return_value=MagicMock()), \
         patch("workers.providers.aviationstack.poll", side_effect=Exception("network down")), \
         patch("workers.providers.weather.poll", return_value=[]), \
         patch("workers.providers.advisories.poll", return_value=[]):
        from workers.live_data_worker import _run_once
        _run_once()
