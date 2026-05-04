import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import patch, MagicMock
from models.budget import BudgetLedger, LedgerEntry, FXRate


@pytest.fixture
def ledger_model():
    return BudgetLedger(
        ledger_id="l1",
        home_currency="EUR",
        daily_cap=Decimal("150.00"),
    )


def test_remaining_budget_excludes_sunk_costs(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    ledger_model.entries = [
        LedgerEntry(
            entry_id="e1", stop_id="s1", category="accommodation",
            amount=Decimal("80.00"), currency="EUR", is_sunk=True,
            timestamp=datetime(2026, 5, 5, 10, 0, 0),
        ),
        LedgerEntry(
            entry_id="e2", stop_id="s2", category="food",
            amount=Decimal("20.00"), currency="EUR", is_sunk=False,
            timestamp=datetime(2026, 5, 5, 12, 0, 0),
        ),
    ]
    # sunk=80 excluded, non-sunk=20 counted → remaining = 150 - 20 = 130
    remaining = svc.remaining_budget("EUR")
    assert remaining == Decimal("130.00")


def test_score_candidate_no_overrun(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    # candidate costs 50 EUR, daily_cap=150, no existing spend → no overrun
    score = svc.score_candidate(Decimal("50.00"), "EUR")
    assert score == 1.0


def test_score_candidate_with_overrun(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    ledger_model.entries = [
        LedgerEntry(
            entry_id="e1", stop_id="s1", category="accommodation",
            amount=Decimal("130.00"), currency="EUR", is_sunk=False,
            timestamp=datetime(2026, 5, 5, 10, 0, 0),
        ),
    ]
    # remaining=20, candidate=50 → overrun=30, score = 1 - 30/150 = 0.8
    score = svc.score_candidate(Decimal("50.00"), "EUR")
    assert abs(score - 0.8) < 0.001


def test_fx_conversion_applies_markup(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    # Store a cached rate: 1 USD = 0.92 EUR mid-market, markup 1.75%
    ledger_model.fx_rates["USD:EUR"] = FXRate(
        source="USD", target="EUR",
        rate=Decimal("0.92"),
        card_markup_pct=1.75,
        fetched_at=datetime(2026, 5, 5, 10, 0, 0),
    )
    converted = svc.convert_to_home(Decimal("100.00"), "USD")
    # 100 * 0.92 * (1 - 0.0175) = 100 * 0.92 * 0.9825 = 90.39
    assert abs(converted - Decimal("90.39")) < Decimal("0.01")


def test_wise_rate_stale_flag_on_outage(ledger_model):
    from budget.ledger import BudgetLedgerService, FXUnavailableError
    svc = BudgetLedgerService(ledger_model)
    # No cached rate, Wise call fails → FXUnavailableError
    with patch("budget.ledger._fetch_wise_rate", side_effect=Exception("timeout")):
        with pytest.raises(FXUnavailableError):
            svc.get_fx_rate("USD", "EUR")


def test_wise_cached_rate_returned_when_stale(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    ledger_model.fx_rates["USD:EUR"] = FXRate(
        source="USD", target="EUR",
        rate=Decimal("0.92"),
        card_markup_pct=1.75,
        fetched_at=datetime(2026, 1, 1, 0, 0, 0),  # old
    )
    with patch("budget.ledger._fetch_wise_rate", side_effect=Exception("timeout")):
        rate = svc.get_fx_rate("USD", "EUR")
        assert rate.is_stale is True
        assert rate.rate == Decimal("0.92")
