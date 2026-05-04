import pytest
from decimal import Decimal
from datetime import datetime
from models.budget import LedgerEntry, FXRate, SunkCost, BudgetLedger


def test_ledger_entry_schema():
    entry = LedgerEntry(
        entry_id="e1",
        stop_id="stop_1",
        category="accommodation",
        amount=Decimal("120.00"),
        currency="EUR",
        is_sunk=False,
        timestamp=datetime(2026, 5, 5, 10, 0, 0),
    )
    assert entry.is_sunk is False
    assert entry.currency == "EUR"


def test_fx_rate_schema():
    rate = FXRate(
        source="EUR",
        target="USD",
        rate=Decimal("1.08"),
        card_markup_pct=1.75,
        fetched_at=datetime(2026, 5, 5, 10, 0, 0),
    )
    assert rate.is_stale is False


def test_budget_ledger_schema():
    ledger = BudgetLedger(
        ledger_id="l1",
        home_currency="EUR",
        daily_cap=Decimal("150.00"),
    )
    assert ledger.entries == []
    assert ledger.fx_rates == {}


from models.replanning import ReplanningContext, RelaxationStep, ReplanningResult
from models.disruption import DisruptionEvent, DisruptionSeverity


def test_replanning_context_schema():
    event = DisruptionEvent(
        event_key="abc123",
        provider="aviationstack",
        entity_id="FL123",
        status_code="CANCELLED",
        severity=DisruptionSeverity.CRITICAL,
        detected_at=datetime(2026, 5, 5, 8, 0, 0),
    )
    ctx = ReplanningContext(
        profile_version_id=1,
        itinerary_version_id=2,
        disruption_event=event,
        affected_stop_ids=["stop_1"],
        started_at=datetime(2026, 5, 5, 9, 0, 0),
    )
    assert ctx.profile_version_id == 1
    assert len(ctx.affected_stop_ids) == 1


def test_replanning_result_escalation_flag():
    result = ReplanningResult(
        result_id="r1",
        proposed_itinerary_version_id=None,
        utility_score=0.0,
        relaxation_steps=[],
        escalated=True,
        escalation_reason="No feasible plan found after full ladder",
        completed_at=datetime(2026, 5, 5, 9, 5, 0),
    )
    assert result.escalated is True
    assert result.proposed_itinerary_version_id is None


from models.itinerary import FallbackOption, Stop, StopType


def test_fallback_option_schema():
    fo = FallbackOption(
        venue_id="v_alt_1",
        name="Alt Hotel",
        stop_type=StopType.LODGING,
        rag_confidence=0.87,
        estimated_cost=Decimal("95.00"),
        currency="EUR",
        constraint_flags={"wheelchair": True, "halal": True},
        staged_at=datetime(2026, 5, 5, 10, 0, 0),
    )
    assert fo.rag_confidence == 0.87


def test_stop_has_fallback_options_field():
    stop = Stop(id="s1", type=StopType.LODGING, name="Hotel A")
    assert stop.fallback_options == []


