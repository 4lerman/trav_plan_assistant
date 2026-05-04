from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from models.disruption import DisruptionEvent


class ReplanningContext(BaseModel):
    profile_version_id: int
    itinerary_version_id: int
    disruption_event: DisruptionEvent
    affected_stop_ids: list[str]
    started_at: datetime


class RelaxationStep(BaseModel):
    step_number: int  # 1–5
    description: str
    constraint_relaxed: Optional[str] = None
    utility_score_after: float


class ReplanningResult(BaseModel):
    result_id: str
    proposed_itinerary_version_id: Optional[int] = None
    utility_score: float
    relaxation_steps: list[RelaxationStep] = []
    escalated: bool = False
    escalation_reason: Optional[str] = None
    completed_at: datetime
