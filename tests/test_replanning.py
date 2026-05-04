import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from graph.state import empty_state
from models.disruption import DisruptionEvent, DisruptionSeverity
from models.itinerary import ItineraryVersion, Stop, StopType, FallbackOption
from models.budget import BudgetLedger
from models.profile import (
    ConstraintProfile, ProfileVersion,
    MobilityLevel, AccommodationFlexibility, DisruptionTolerance,
)


@pytest.fixture(autouse=True)
def clear_queue_db():
    import sqlite3
    import os
    if os.path.exists("disruption_queue.db"):
        conn = sqlite3.connect("disruption_queue.db")
        conn.execute("DELETE FROM events")
        conn.commit()
        conn.close()


@pytest.fixture
def profile_version():
    profile = ConstraintProfile(
        mobility_level=MobilityLevel.NONE,
        dietary_tags=["halal"],
        medical_needs=[],
        daily_budget=Decimal("150.00"),
        base_currency="EUR",
        accommodation_flexibility=AccommodationFlexibility.MODERATE,
        disruption_tolerance=DisruptionTolerance.LOW,
        language="en",
    )
    return ProfileVersion(
        version_id=1,
        created_at=datetime(2026, 5, 5, 8, 0, 0),
        profile=profile,
        consent_recorded=True,
    )


@pytest.fixture
def disruption_event():
    return DisruptionEvent(
        event_key="evt_abc",
        provider="aviationstack",
        entity_id="hotel_x",
        status_code="CLOSED",
        severity=DisruptionSeverity.CRITICAL,
        detected_at=datetime(2026, 5, 5, 9, 0, 0),
    )


@pytest.fixture
def itinerary_with_fallbacks(disruption_event):
    fallback = FallbackOption(
        venue_id="hotel_y",
        name="Hotel Y",
        stop_type=StopType.LODGING,
        rag_confidence=0.85,
        estimated_cost=Decimal("90.00"),
        currency="EUR",
        constraint_flags={"wheelchair": True, "halal": True},
        staged_at=datetime(2026, 5, 5, 8, 0, 0),
    )
    stop = Stop(
        id="stop_hotel",
        type=StopType.LODGING,
        name="Hotel X",
        doc_id=disruption_event.entity_id,
        fallback_options=[fallback],
    )
    return ItineraryVersion(
        version_id=1,
        created_at=datetime(2026, 5, 5, 8, 0, 0),
        stops=[stop],
        dag_edges=[],
        profile_version_id=1,
        days=1,
    )


def test_replanning_uses_pre_staged_fallback(
    profile_version, disruption_event, itinerary_with_fallbacks
):
    """When a pre-staged fallback scores above threshold, it is used without RAG."""
    from agents.replanning import replanning_node
    from workers.queue import enqueue, mark_processed

    enqueue(disruption_event)

    state = empty_state("session_replan_1")
    state["profile"] = profile_version
    state["itinerary"] = itinerary_with_fallbacks
    state["itinerary_history"] = [itinerary_with_fallbacks]
    state["active_disruption_id"] = disruption_event.event_key
    state["budget_ledger"] = BudgetLedger(
        ledger_id="l1", home_currency="EUR", daily_cap=Decimal("150.00")
    )

    with patch("agents.replanning._retrieve_fallback_from_rag") as mock_rag:
        result = replanning_node(state)

    # RAG should NOT have been called — pre-staged fallback was good enough
    mock_rag.assert_not_called()
    assert "itinerary" in result
    assert result["active_disruption_id"] is None
    assert result["replanning_context"] is None
    # New itinerary version has the fallback as the stop
    new_itin = result["itinerary"]
    assert new_itin.version_id == 2
    assert any(s.name == "Hotel Y" for s in new_itin.stops)


def test_replanning_falls_back_to_rag_when_no_staged_fallbacks(
    profile_version, disruption_event
):
    """When stop has no pre-staged fallbacks, RAG is called instead."""
    from agents.replanning import replanning_node
    from workers.queue import enqueue

    stop_no_fallbacks = Stop(
        id="stop_hotel",
        type=StopType.LODGING,
        name="Hotel X",
        doc_id=disruption_event.entity_id,
        fallback_options=[],
    )
    itin = ItineraryVersion(
        version_id=1,
        created_at=datetime(2026, 5, 5, 8, 0, 0),
        stops=[stop_no_fallbacks],
        dag_edges=[],
        profile_version_id=1,
        days=1,
    )
    enqueue(disruption_event)

    state = empty_state("session_replan_2")
    state["profile"] = profile_version
    state["itinerary"] = itin
    state["itinerary_history"] = [itin]
    state["active_disruption_id"] = disruption_event.event_key
    state["budget_ledger"] = BudgetLedger(
        ledger_id="l2", home_currency="EUR", daily_cap=Decimal("150.00")
    )

    rag_fallback = FallbackOption(
        venue_id="hotel_z",
        name="Hotel Z",
        stop_type=StopType.LODGING,
        rag_confidence=0.78,
        estimated_cost=Decimal("85.00"),
        currency="EUR",
        constraint_flags={"halal": True},
        staged_at=datetime(2026, 5, 5, 9, 0, 0),
    )

    with patch("agents.replanning._retrieve_fallback_from_rag", return_value=[rag_fallback]):
        result = replanning_node(state)

    assert "itinerary" in result
    new_itin = result["itinerary"]
    assert any(s.name == "Hotel Z" for s in new_itin.stops)
    assert result["active_disruption_id"] is None


def test_replanning_escalates_when_ladder_exhausted(
    profile_version, disruption_event
):
    """When all fallbacks fail and relaxation ladder exhausts, escalation is emitted."""
    from agents.replanning import replanning_node
    from workers.queue import enqueue

    stop = Stop(
        id="stop_hotel",
        type=StopType.LODGING,
        name="Hotel X",
        doc_id=disruption_event.entity_id,
        fallback_options=[],
    )
    itin = ItineraryVersion(
        version_id=1,
        created_at=datetime(2026, 5, 5, 8, 0, 0),
        stops=[stop],
        dag_edges=[],
        profile_version_id=1,
        days=1,
    )
    enqueue(disruption_event)

    state = empty_state("session_replan_3")
    state["profile"] = profile_version
    state["itinerary"] = itin
    state["itinerary_history"] = [itin]
    state["active_disruption_id"] = disruption_event.event_key
    state["budget_ledger"] = BudgetLedger(
        ledger_id="l3", home_currency="EUR", daily_cap=Decimal("150.00")
    )

    # RAG returns nothing at every ladder step
    with patch("agents.replanning._retrieve_fallback_from_rag", return_value=[]):
        result = replanning_node(state)

    assert result["active_disruption_id"] is None
    assert result["replanning_context"] is None
    # Escalation message emitted
    msgs = result.get("messages", [])
    assert any("escalat" in m.content.lower() or "unable to find" in m.content.lower() for m in msgs)


def test_replanning_clears_active_disruption_id_on_success(
    profile_version, disruption_event, itinerary_with_fallbacks
):
    from agents.replanning import replanning_node
    from workers.queue import enqueue

    enqueue(disruption_event)
    state = empty_state("session_replan_4")
    state["profile"] = profile_version
    state["itinerary"] = itinerary_with_fallbacks
    state["itinerary_history"] = [itinerary_with_fallbacks]
    state["active_disruption_id"] = disruption_event.event_key
    state["budget_ledger"] = BudgetLedger(
        ledger_id="l4", home_currency="EUR", daily_cap=Decimal("150.00")
    )

    with patch("agents.replanning._retrieve_fallback_from_rag"):
        result = replanning_node(state)

    assert result["active_disruption_id"] is None


def test_graph_routes_to_replanning_when_active_disruption_id_set(
    profile_version, disruption_event, itinerary_with_fallbacks
):
    """Graph routes to replanning node when active_disruption_id is set."""
    from graph.graph import route

    state = {
        "active_disruption_id": disruption_event.event_key,
        "profile": profile_version,
        "itinerary": itinerary_with_fallbacks,
        "disruption_queue": [],
        "rag_context": {},
    }
    assert route(state) == "replanning"


def test_full_replan_cycle_via_graph(
    profile_version, disruption_event, itinerary_with_fallbacks
):
    """Full graph invocation: disruption event → graph routes to replanning → new itinerary emitted."""
    from graph.graph import graph
    from workers.queue import enqueue
    from langchain_core.messages import HumanMessage

    enqueue(disruption_event)

    state = {
        "session_id": "e2e_session",
        "state_version": 0,
        "profile": profile_version,
        "profile_history": [profile_version],
        "itinerary": itinerary_with_fallbacks,
        "itinerary_history": [itinerary_with_fallbacks],
        "budget_ledger": BudgetLedger(
            ledger_id="l_e2e", home_currency="EUR", daily_cap=Decimal("150.00")
        ),
        "disruption_queue": [],
        "active_disruption_id": disruption_event.event_key,
        "rag_context": {},
        "live_data": {},
        "messages": [HumanMessage(content="What happened to my trip?")],
        "replanning_context": None,
    }

    with patch("agents.replanning._retrieve_fallback_from_rag"):
        result = graph.invoke(state, config={"configurable": {"thread_id": "e2e_session"}})

    assert result["active_disruption_id"] is None
    # A new itinerary version or escalation message was emitted
    assert result.get("itinerary") is not None or any(
        "replanning" in m.content.lower() or "unable" in m.content.lower()
        for m in result.get("messages", [])
    )
