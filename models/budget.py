from __future__ import annotations
from datetime import date
from decimal import Decimal
from pydantic import BaseModel


class Booking(BaseModel):
    id: str
    description: str
    amount: Decimal
    currency: str
    refundable: bool
    sunk: Decimal = Decimal("0.00")


class BudgetLedger(BaseModel):
    daily_cap: Decimal
    base_currency: str
    spent_by_day: dict[date, Decimal] = {}
    committed_by_day: dict[date, Decimal] = {}
    bookings: list[Booking] = []

    def remaining_for(self, day: date) -> Decimal:
        spent = self.spent_by_day.get(day, Decimal("0.00"))
        committed = self.committed_by_day.get(day, Decimal("0.00"))
        return self.daily_cap - spent - committed
