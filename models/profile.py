from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_validator


class MobilityLevel(str, Enum):
    FULL = "full"        # step-free everywhere required
    PARTIAL = "partial"  # some steps tolerable
    NONE = "none"        # no mobility restriction


class AccommodationFlexibility(str, Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    FLEXIBLE = "flexible"


class DisruptionTolerance(str, Enum):
    LOW = "low"        # replan immediately
    MEDIUM = "medium"  # notify, replan within 2h
    HIGH = "high"      # notify only


class ConstraintProfile(BaseModel):
    mobility_level: MobilityLevel
    dietary_tags: list[str]
    medical_needs: list[str]
    daily_budget: Decimal
    base_currency: str          # ISO 4217
    accommodation_flexibility: AccommodationFlexibility
    disruption_tolerance: DisruptionTolerance
    language: str               # BCP 47
    offline_max_relaxation: int = 2

    @field_validator("daily_budget")
    @classmethod
    def budget_must_be_positive(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("daily_budget must be non-negative")
        return v


class ProfileVersion(BaseModel):
    version_id: int
    created_at: datetime
    profile: ConstraintProfile
    diff: Optional[dict] = None
    consent_recorded: bool = False


def make_profile_version(
    profile: ConstraintProfile,
    previous_version_id: Optional[int],
    diff: Optional[dict] = None,
    consent_recorded: bool = False,
) -> ProfileVersion:
    version_id = 1 if previous_version_id is None else previous_version_id + 1
    return ProfileVersion(
        version_id=version_id,
        created_at=datetime.utcnow(),
        profile=profile,
        diff=diff,
        consent_recorded=consent_recorded,
    )
