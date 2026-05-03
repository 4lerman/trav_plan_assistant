from __future__ import annotations
import hashlib
from datetime import datetime
from enum import Enum
from pydantic import BaseModel


class DisruptionSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    ADVISORY = "advisory"


class DisruptionEvent(BaseModel):
    event_key: str
    provider: str
    entity_id: str
    status_code: str
    severity: DisruptionSeverity
    detected_at: datetime
    raw_payload: dict = {}


def make_event_key(*, provider: str, entity_id: str, status_code: str, window: str) -> str:
    raw = f"{provider}:{entity_id}:{status_code}:{window}"
    return hashlib.sha256(raw.encode()).hexdigest()


class NormalisedEvent(BaseModel):
    provider: str       # "aviationstack" | "weather" | "transit"
    entity_id: str      # flight segment ID or Wheelmap node ID
    status_code: str    # provider-specific normalised status string
    window: str         # ISO date string — used for event_key dedup
    raw_payload: dict = {}
