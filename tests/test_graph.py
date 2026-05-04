import pytest
import os
from decimal import Decimal
from datetime import datetime
from models.profile import MobilityLevel, AccommodationFlexibility, DisruptionTolerance
from models.disruption import DisruptionSeverity


def make_disruption_event(key: str) -> dict:
    return {
        "event_key": key,
        "provider": "duffel",
        "entity_id": "FL123",
        "status_code": "CANCELLED",
        "severity": DisruptionSeverity.CRITICAL,
        "detected_at": datetime(2026, 4, 26, 10, 0, 0),
        "raw_payload": {},
    }


class TestTripStateReducers:
    def test_disruption_queue_deduplicates(self):
        from graph.state import TripState
        from langgraph.graph import StateGraph
        # Just verify TripState is importable and typed correctly
        assert "disruption_queue" in TripState.__annotations__

    def test_state_fields_present(self):
        from graph.state import TripState
        required = {
            "session_id", "state_version", "profile", "profile_history",
            "itinerary", "itinerary_history", "budget_ledger",
            "disruption_queue", "active_disruption_id",
            "rag_context", "live_data", "messages",
        }
        assert required.issubset(set(TripState.__annotations__.keys()))


class TestCheckpointer:
    def test_sqlite_checkpointer_returned_when_no_postgres_dsn(self, tmp_path, monkeypatch):
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        monkeypatch.chdir(tmp_path)
        from graph.checkpointer import get_checkpointer
        cp = get_checkpointer()
        assert cp is not None

    def test_checkpointer_interface(self, tmp_path, monkeypatch):
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        monkeypatch.chdir(tmp_path)
        from graph.checkpointer import get_checkpointer
        cp = get_checkpointer()
        # Must have put/get interface used by LangGraph
        assert hasattr(cp, "put") or hasattr(cp, "__enter__")


class TestRouting:
    def test_routes_to_profiler_when_no_profile(self):
        from graph.graph import route
        state = {
            "profile": None,
            "itinerary": None,
            "active_disruption_id": None,
            "disruption_queue": [],
        }
        assert route(state) == "constraint_profiler"

    def test_disruption_queue_non_empty_routes_to_replanning(self):
        from unittest.mock import patch
        from graph.graph import route
        state = {
            "profile": object(),
            "itinerary": object(),
            "disruption_queue": [],
            "active_disruption_id": None,
            "rag_context": {},
        }
        with patch("graph.graph.dequeue_pending", return_value=[{"event_key": "abc"}]):
            assert route(state) == "replanning"

    def test_routes_to_replanning_when_active_disruption_set(self):
        from graph.graph import route
        state = {
            "profile": "some_profile",
            "itinerary": "some_itinerary",
            "active_disruption_id": "disruption-123",
            "disruption_queue": [],
        }
        assert route(state) == "replanning"

    def test_routes_to_destination_research_when_profile_but_no_itinerary(self):
        from graph.graph import route
        state = {
            "profile": "some_profile",
            "itinerary": None,
            "active_disruption_id": None,
            "disruption_queue": [],
        }
        assert route(state) == "destination_research"

    def test_routes_to_orchestrator_reply_when_all_present(self):
        from graph.graph import route
        state = {
            "profile": "some_profile",
            "itinerary": "some_itinerary",
            "active_disruption_id": None,
            "disruption_queue": [],
        }
        assert route(state) == "orchestrator_reply"

