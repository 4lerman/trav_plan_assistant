from __future__ import annotations
import os
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional
import httpx
from models.budget import BudgetLedger, LedgerEntry, FXRate

_WISE_BASE = "https://api.wise.com"
_REDIS_TTL = 900  # 15 minutes


class FXUnavailableError(Exception):
    pass


def _fetch_wise_rate(source: str, target: str) -> Decimal:
    """Fetch mid-market rate from Wise API. Raises on any failure."""
    api_key = os.environ.get("WISE_API_KEY", "")
    resp = httpx.get(
        f"{_WISE_BASE}/v1/rates",
        params={"source": source, "target": target},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=5.0,
    )
    resp.raise_for_status()
    data = resp.json()
    # Wise returns a list; first item is the rate
    return Decimal(str(data[0]["rate"]))


class BudgetLedgerService:
    def __init__(self, ledger: BudgetLedger) -> None:
        self._ledger = ledger

    # ------------------------------------------------------------------
    # FX
    # ------------------------------------------------------------------

    def get_fx_rate(self, source: str, target: str) -> FXRate:
        """Return FXRate for source→target. Falls back to cached stale rate on outage."""
        if source == target:
            return FXRate(
                source=source, target=target,
                rate=Decimal("1.0"), card_markup_pct=0.0,
                fetched_at=datetime.now(timezone.utc),
            )

        cache_key = f"{source}:{target}"
        try:
            rate_value = _fetch_wise_rate(source, target)
            rate = FXRate(
                source=source, target=target,
                rate=rate_value, card_markup_pct=1.75,
                fetched_at=datetime.now(timezone.utc),
            )
            self._ledger.fx_rates[cache_key] = rate
            return rate
        except Exception:
            cached = self._ledger.fx_rates.get(cache_key)
            if cached is None:
                raise FXUnavailableError(
                    f"Wise API unavailable and no cached rate for {source}→{target}"
                )
            stale = cached.model_copy(update={"is_stale": True})
            self._ledger.fx_rates[cache_key] = stale
            return stale

    def convert_to_home(self, amount: Decimal, currency: str) -> Decimal:
        """Convert amount in `currency` to ledger's home_currency applying card markup."""
        home = self._ledger.home_currency
        if currency == home:
            return amount
        rate = self.get_fx_rate(currency, home)
        markup_factor = Decimal(str(1 - rate.card_markup_pct / 100))
        return (amount * rate.rate * markup_factor).quantize(Decimal("0.01"))

    # ------------------------------------------------------------------
    # Ledger
    # ------------------------------------------------------------------

    def remaining_budget(self, currency: str) -> Decimal:
        """Daily cap minus non-sunk spend, in requested currency."""
        non_sunk_total = sum(
            self.convert_to_home(e.amount, e.currency)
            for e in self._ledger.entries
            if not e.is_sunk
        )
        remaining_home = self._ledger.daily_cap - non_sunk_total
        if currency == self._ledger.home_currency:
            return remaining_home
        rate = self.get_fx_rate(self._ledger.home_currency, currency)
        markup_factor = Decimal(str(1 - rate.card_markup_pct / 100))
        return (remaining_home * rate.rate * markup_factor).quantize(Decimal("0.01"))

    def score_candidate(self, candidate_cost: Decimal, currency: str) -> float:
        """Return budget_score = 1 − max(0, overrun) / daily_cap, clamped to [0, 1]."""
        remaining = self.remaining_budget(self._ledger.home_currency)
        cost_home = self.convert_to_home(candidate_cost, currency)
        overrun = max(Decimal("0"), cost_home - remaining)
        if self._ledger.daily_cap == 0:
            return 0.0
        score = float(1 - overrun / self._ledger.daily_cap)
        return max(0.0, min(1.0, score))

    def record_entry(self, entry: LedgerEntry) -> None:
        """Append entry; idempotent on entry_id."""
        existing_ids = {e.entry_id for e in self._ledger.entries}
        if entry.entry_id not in existing_ids:
            self._ledger.entries.append(entry)
