from __future__ import annotations
import pytest
from models.disruption import DisruptionSeverity, NormalisedEvent
from workers.disruption_rules import evaluate

def _event(provider: str, status_code: str) -> NormalisedEvent:
    return NormalisedEvent(
        provider=provider,
        entity_id="ent_001",
        status_code=status_code,
        window="2026-05-10",
        raw_payload={},
    )

def test_aviationstack_cancelled_is_critical():
    assert evaluate(_event("aviationstack", "cancelled")) == DisruptionSeverity.CRITICAL

def test_aviationstack_delayed_major_is_critical():
    assert evaluate(_event("aviationstack", "delayed_major")) == DisruptionSeverity.CRITICAL

def test_aviationstack_delayed_minor_is_warning():
    assert evaluate(_event("aviationstack", "delayed_minor")) == DisruptionSeverity.WARNING

def test_weather_extreme_is_critical():
    assert evaluate(_event("weather", "extreme_weather")) == DisruptionSeverity.CRITICAL

def test_weather_severe_is_warning():
    assert evaluate(_event("weather", "severe_weather")) == DisruptionSeverity.WARNING

def test_advisories_high_risk_is_critical():
    assert evaluate(_event("advisories", "high_risk")) == DisruptionSeverity.CRITICAL

def test_advisories_medium_risk_is_warning():
    assert evaluate(_event("advisories", "medium_risk")) == DisruptionSeverity.WARNING

def test_unknown_status_is_none():
    assert evaluate(_event("aviationstack", "unknown_status")) is None

def test_transit_suspended_is_critical():
    assert evaluate(_event("transit", "service_suspended")) == DisruptionSeverity.CRITICAL

def test_transit_modified_is_warning():
    assert evaluate(_event("transit", "service_modified")) == DisruptionSeverity.WARNING
