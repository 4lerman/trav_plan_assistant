import pytest
from decimal import Decimal
from datetime import datetime, date
from pydantic import ValidationError
from models.profile import (
    ConstraintProfile, ProfileVersion,
    MobilityLevel, AccommodationFlexibility, DisruptionTolerance,
)
from models.disruption import DisruptionEvent, DisruptionSeverity
from models.budget import BudgetLedger, Booking


class TestConstraintProfile:
    def test_valid_profile_constructs(self):
        profile = ConstraintProfile(
            mobility_level=MobilityLevel.FULL,
            dietary_tags=["halal"],
            medical_needs=[],
            daily_budget=Decimal("80.00"),
            base_currency="EUR",
            accommodation_flexibility=AccommodationFlexibility.STRICT,
            disruption_tolerance=DisruptionTolerance.LOW,
            language="en",
        )
        assert profile.mobility_level == MobilityLevel.FULL
        assert profile.dietary_tags == ["halal"]

    def test_invalid_mobility_level_raises(self):
        with pytest.raises(ValidationError):
            ConstraintProfile(
                mobility_level="flying",
                dietary_tags=[],
                medical_needs=[],
                daily_budget=Decimal("80.00"),
                base_currency="EUR",
                accommodation_flexibility=AccommodationFlexibility.STRICT,
                disruption_tolerance=DisruptionTolerance.LOW,
                language="en",
            )

    def test_negative_budget_raises(self):
        with pytest.raises(ValidationError):
            ConstraintProfile(
                mobility_level=MobilityLevel.NONE,
                dietary_tags=[],
                medical_needs=[],
                daily_budget=Decimal("-1.00"),
                base_currency="EUR",
                accommodation_flexibility=AccommodationFlexibility.FLEXIBLE,
                disruption_tolerance=DisruptionTolerance.HIGH,
                language="en",
            )

    def test_offline_max_relaxation_defaults_to_2(self):
        profile = ConstraintProfile(
            mobility_level=MobilityLevel.NONE,
            dietary_tags=[],
            medical_needs=[],
            daily_budget=Decimal("100.00"),
            base_currency="USD",
            accommodation_flexibility=AccommodationFlexibility.FLEXIBLE,
            disruption_tolerance=DisruptionTolerance.HIGH,
            language="en",
        )
        assert profile.offline_max_relaxation == 2


class TestProfileVersion:
    def test_version_id_monotonicity_via_factory(self):
        from models.profile import make_profile_version
        p = ConstraintProfile(
            mobility_level=MobilityLevel.NONE,
            dietary_tags=[],
            medical_needs=[],
            daily_budget=Decimal("50.00"),
            base_currency="GBP",
            accommodation_flexibility=AccommodationFlexibility.FLEXIBLE,
            disruption_tolerance=DisruptionTolerance.HIGH,
            language="en",
        )
        v1 = make_profile_version(p, previous_version_id=None)
        v2 = make_profile_version(p, previous_version_id=v1.version_id)
        assert v2.version_id == v1.version_id + 1

    def test_consent_defaults_false(self):
        from models.profile import make_profile_version
        p = ConstraintProfile(
            mobility_level=MobilityLevel.NONE,
            dietary_tags=[],
            medical_needs=[],
            daily_budget=Decimal("50.00"),
            base_currency="GBP",
            accommodation_flexibility=AccommodationFlexibility.FLEXIBLE,
            disruption_tolerance=DisruptionTolerance.HIGH,
            language="en",
        )
        v = make_profile_version(p, previous_version_id=None)
        assert v.consent_recorded is False


class TestDisruptionEvent:
    def test_event_key_stable_across_identical_inputs(self):
        from models.disruption import make_event_key
        key1 = make_event_key(provider="duffel", entity_id="FL123", status_code="CANCELLED", window="2026-04-26T10:00")
        key2 = make_event_key(provider="duffel", entity_id="FL123", status_code="CANCELLED", window="2026-04-26T10:00")
        assert key1 == key2

    def test_event_key_differs_for_different_inputs(self):
        from models.disruption import make_event_key
        key1 = make_event_key(provider="duffel", entity_id="FL123", status_code="CANCELLED", window="2026-04-26T10:00")
        key2 = make_event_key(provider="duffel", entity_id="FL456", status_code="CANCELLED", window="2026-04-26T10:00")
        assert key1 != key2


class TestBudgetLedger:
    def test_remaining_for_day(self):
        ledger = BudgetLedger(
            daily_cap=Decimal("80.00"),
            base_currency="EUR",
            spent_by_day={date(2026, 5, 1): Decimal("30.00")},
            committed_by_day={date(2026, 5, 1): Decimal("20.00")},
            bookings=[],
        )
        assert ledger.remaining_for(date(2026, 5, 1)) == Decimal("30.00")

    def test_remaining_for_day_with_no_spend(self):
        ledger = BudgetLedger(
            daily_cap=Decimal("80.00"),
            base_currency="EUR",
            spent_by_day={},
            committed_by_day={},
            bookings=[],
        )
        assert ledger.remaining_for(date(2026, 5, 1)) == Decimal("80.00")
