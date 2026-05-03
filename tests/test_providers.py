from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from models.profile import ConstraintProfile, MobilityLevel, AccommodationFlexibility, DisruptionTolerance
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

class TestAviationstackPoll:
    def test_cancelled_flight(self):
        from workers.providers.aviationstack import poll
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"flight_status": "cancelled"}]}
        with patch("workers.providers.aviationstack.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"AVIATIONSTACK_API_KEY": "test_key"}):
                events = poll(["AA123"], _profile())
        assert len(events) == 1
        assert events[0].status_code == "cancelled"

class TestWeatherPoll:
    def test_extreme_weather(self):
        from workers.providers.weather import poll
        mock_response = MagicMock()
        mock_response.json.return_value = {"weather": [{"id": 212}]}
        with patch("workers.providers.weather.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test"}):
                events = poll(["London"], _profile())
        assert len(events) == 1
        assert events[0].status_code == "extreme_weather"

class TestAdvisoriesPoll:
    def test_high_risk(self):
        from workers.providers.advisories import poll
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"safetyScores": {"overall": 80}}]}
        with patch("workers.providers.advisories.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"ADVISORY_API_KEY": "test"}):
                events = poll(["London"], _profile())
        assert len(events) == 1
        assert events[0].status_code == "high_risk"

class TestTransitPoll:
    def test_service_suspended(self):
        from workers.providers.transit import poll
        mock_response = MagicMock()
        mock_response.json.return_value = {"alerts": [{"effect": "strike"}]}
        with patch("workers.providers.transit.httpx.get", return_value=mock_response):
            with patch.dict("os.environ", {"TRANSIT_API_KEY": "test"}):
                events = poll(["route_1"], _profile())
        assert len(events) == 1
        assert events[0].status_code == "service_suspended"
