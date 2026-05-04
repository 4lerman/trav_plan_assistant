from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel


class LedgerEntry(BaseModel):
    entry_id: str
    stop_id: str
    category: Literal["accommodation", "transport", "food", "activity"]
    amount: Decimal
    currency: str  # ISO 4217
    is_sunk: bool
    timestamp: datetime


class FXRate(BaseModel):
    source: str
    target: str
    rate: Decimal
    card_markup_pct: float = 1.75
    fetched_at: datetime
    is_stale: bool = False


class SunkCost(BaseModel):
    stop_id: str
    amount: Decimal
    currency: str
    reason: str


class BudgetLedger(BaseModel):
    ledger_id: str
    home_currency: str
    daily_cap: Decimal
    entries: list[LedgerEntry] = []
    fx_rates: dict[str, FXRate] = {}  # key: "{src}:{tgt}"
