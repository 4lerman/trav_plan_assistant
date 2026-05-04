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


class FallbackOption(BaseModel):
    venue_id: str
    name: str
    stop_type: StopType
    rag_confidence: float
    estimated_cost: Decimal
    currency: str
    constraint_flags: dict[str, bool] = {}  # constraint_name -> passes
    staged_at: datetime


class Stop(BaseModel):
    id: str
    type: StopType
    name: str
    constraint_flags: dict = {}
    budget_estimate: Decimal = Decimal("0.00")
    depends_on: list[str] = []
    fallback_alternatives: list["Stop"] = []
    fallback_options: list[FallbackOption] = []
    doc_id: Optional[str] = None
    confidence_score: float = 0.0

class ItineraryVersion(BaseModel):
    version_id: int
    created_at: datetime
    stops: list[Stop] = []
    validation_report: dict = {}
    profile_version_id: Optional[int] = None
    dag_edges: list[tuple[str, str]] = []
    days: int = 0
