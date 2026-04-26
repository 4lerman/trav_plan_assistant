import pytest
from datetime import datetime
from decimal import Decimal
from models.profile import (
    ConstraintProfile, ProfileVersion,
    MobilityLevel, AccommodationFlexibility, DisruptionTolerance,
)


@pytest.fixture
def sample_profile() -> ConstraintProfile:
    return ConstraintProfile(
        mobility_level=MobilityLevel.FULL,
        dietary_tags=["halal"],
        medical_needs=[],
        daily_budget=Decimal("80.00"),
        base_currency="EUR",
        accommodation_flexibility=AccommodationFlexibility.STRICT,
        disruption_tolerance=DisruptionTolerance.LOW,
        language="en",
    )


@pytest.fixture
def sample_profile_version(sample_profile) -> ProfileVersion:
    return ProfileVersion(
        version_id=1,
        created_at=datetime(2026, 4, 26, 12, 0, 0),
        profile=sample_profile,
        consent_recorded=True,
    )
