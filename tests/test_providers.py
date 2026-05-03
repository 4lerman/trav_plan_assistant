from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from models.profile import (
    ConstraintProfile, MobilityLevel, AccommodationFlexibility, DisruptionTolerance,
)
from decimal import Decimal


def _profile() -> ConstraintProfile:
    return ConstraintProfile(
        mobility_level=MobilityLevel.FULL,
        dietary_tags=["halal"],
        medical_needs=[],
        daily_budget=Decimal("100.00"),
        base_currency="EUR",
        accommodation_flexibility=AccommodationFlexibility.STRICT,
        disruption_tolerance=DisruptionTolerance.LOW,
        language="en",
    )


# --- Duffel tests ---

class TestDuffelPoll:
    def test_cancelled_flight_returns_normalised_event(self):
        from workers.providers.duffel import poll

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "seg_001", "status": "cancelled"}]
        }

        with patch("workers.providers.duffel.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"DUFFEL_API_KEY": "test_key"}):
                events = poll(["seg_001"], _profile())

        assert len(events) == 1
        assert events[0].provider == "duffel"
        assert events[0].entity_id == "seg_001"
        assert events[0].status_code == "cancelled"

    def test_http_error_returns_empty_list(self):
        from workers.providers.duffel import poll
        import httpx

        with patch("workers.providers.duffel.httpx.get", side_effect=httpx.HTTPError("timeout")):
            with patch.dict("os.environ", {"DUFFEL_API_KEY": "test_key"}):
                events = poll(["seg_001"], _profile())

        assert events == []

    def test_missing_api_key_returns_empty_list(self):
        from workers.providers.duffel import poll
        import os

        env = {k: v for k, v in os.environ.items() if k != "DUFFEL_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            events = poll(["seg_001"], _profile())

        assert events == []

    def test_large_delay_maps_to_delayed_major(self):
        from workers.providers.duffel import poll

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "seg_002", "status": "changed", "delay_minutes": 150}]
        }

        with patch("workers.providers.duffel.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"DUFFEL_API_KEY": "test_key"}):
                events = poll(["seg_002"], _profile())

        assert len(events) == 1
        assert events[0].status_code == "delayed_major"

    def test_small_delay_maps_to_delayed_minor(self):
        from workers.providers.duffel import poll

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "seg_003", "status": "changed", "delay_minutes": 45}]
        }

        with patch("workers.providers.duffel.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"DUFFEL_API_KEY": "test_key"}):
                events = poll(["seg_003"], _profile())

        assert len(events) == 1
        assert events[0].status_code == "delayed_minor"

# --- Wheelmap tests ---

class TestWheelmapPoll:
    def test_inaccessible_node_returns_event(self):
        from workers.providers.wheelmap import poll

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "node": {"id": 12345, "wheelchair": "no"}
        }

        with patch("workers.providers.wheelmap.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"WHEELMAP_API_KEY": "test_key"}):
                events = poll(["12345"], _profile())

        assert len(events) == 1
        assert events[0].provider == "wheelmap"
        assert events[0].entity_id == "12345"
        assert events[0].status_code == "inaccessible"

    def test_accessible_node_returns_event(self):
        from workers.providers.wheelmap import poll

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "node": {"id": 12346, "wheelchair": "yes"}
        }

        with patch("workers.providers.wheelmap.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"WHEELMAP_API_KEY": "test_key"}):
                events = poll(["12346"], _profile())

        assert len(events) == 1
        assert events[0].status_code == "accessible"

    def test_limited_access_maps_correctly(self):
        from workers.providers.wheelmap import poll

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "node": {"id": 12347, "wheelchair": "limited"}
        }

        with patch("workers.providers.wheelmap.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"WHEELMAP_API_KEY": "test_key"}):
                events = poll(["12347"], _profile())

        assert events[0].status_code == "limited_access"

    def test_http_error_returns_empty_list(self):
        from workers.providers.wheelmap import poll
        import httpx

        with patch("workers.providers.wheelmap.httpx.get", side_effect=httpx.HTTPError("timeout")):
            with patch.dict("os.environ", {"WHEELMAP_API_KEY": "test_key"}):
                events = poll(["12348"], _profile())

        assert events == []

    def test_missing_api_key_returns_empty_list(self):
        from workers.providers.wheelmap import poll
        import os

        env = {k: v for k, v in os.environ.items() if k != "WHEELMAP_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            events = poll(["12349"], _profile())

        assert events == []
