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
