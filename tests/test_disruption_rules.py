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


def test_amadeus_cancelled_is_critical():
    assert evaluate(_event("amadeus", "cancelled")) == DisruptionSeverity.CRITICAL


def test_amadeus_delayed_major_is_critical():
    assert evaluate(_event("amadeus", "delayed_major")) == DisruptionSeverity.CRITICAL


def test_amadeus_delayed_minor_is_warning():
    assert evaluate(_event("amadeus", "delayed_minor")) == DisruptionSeverity.WARNING


def test_wheelmap_inaccessible_is_critical():
    assert evaluate(_event("wheelmap", "inaccessible")) == DisruptionSeverity.CRITICAL


def test_wheelmap_limited_access_is_warning():
    assert evaluate(_event("wheelmap", "limited_access")) == DisruptionSeverity.WARNING


def test_wheelmap_accessible_is_none():
    assert evaluate(_event("wheelmap", "accessible")) is None


def test_unknown_status_is_none():
    assert evaluate(_event("amadeus", "unknown_status")) is None
