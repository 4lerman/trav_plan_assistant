from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class StopType(str, Enum):
    TRANSIT = "transit"
    LODGING = "lodging"
    ACTIVITY = "activity"
    MEAL = "meal"


class Stop(BaseModel):
    id: str
    type: StopType
    name: str
    constraint_flags: dict = {}
    budget_estimate: Decimal = Decimal("0.00")


class ItineraryVersion(BaseModel):
    version_id: int
    created_at: datetime
    stops: list[Stop] = []
    validation_report: dict = {}
    profile_version_id: Optional[int] = None
