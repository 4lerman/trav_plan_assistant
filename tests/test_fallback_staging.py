import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import datetime
from langchain_core.messages import HumanMessage
from agents.itinerary_builder import itinerary_builder_node
from graph.state import empty_state
from models.itinerary import StopType


@pytest.fixture
def profile_version(sample_profile_version):
    return sample_profile_version


def test_fallback_options_staged_for_bookable_stop(profile_version):
    """Bookable stop (lodging) gets FallbackOption objects, not raw Stop objects."""
    state = empty_state("session_1")
    state["profile"] = profile_version
    state["messages"] = [HumanMessage(content="1 day trip")]
    state["rag_context"] = {
        "req_1": [
            {"doc_id": "doc_1", "category": "lodging", "name": "Hotel A",
             "description": "Nice hotel", "avg_cost_per_person": 90,
             "constraint_flags": {"wheelchair": True}},
            {"doc_id": "doc_2", "category": "lodging", "name": "Hotel B",
             "description": "Alt hotel", "avg_cost_per_person": 80,
             "constraint_flags": {"wheelchair": True}},
            {"doc_id": "doc_3", "category": "lodging", "name": "Hotel C",
             "description": "Budget hotel", "avg_cost_per_person": 70,
             "constraint_flags": {"wheelchair": True}},
        ]
    }

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [{"id": "s1", "type": "lodging", "name": "Hotel A", "doc_id": "doc_1", "depends_on": []}],
            "dag_edges": [],
            "days": 1
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]
        result = itinerary_builder_node(state)

    stop = result["itinerary"].stops[0]
    assert len(stop.fallback_options) == 2
    assert stop.fallback_options[0].venue_id == "doc_2"
    assert stop.fallback_options[0].stop_type == StopType.LODGING
    assert isinstance(stop.fallback_options[0].estimated_cost, Decimal)
    assert stop.fallback_options[0].staged_at is not None


def test_fallback_options_not_staged_for_activity(profile_version):
    """Activity stops (non-bookable) do NOT get fallback_options."""
    state = empty_state("session_1")
    state["profile"] = profile_version
    state["messages"] = [HumanMessage(content="1 day trip")]
    state["rag_context"] = {
        "req_1": [
            {"doc_id": "doc_1", "category": "activity", "name": "Museum",
             "description": "Art museum", "avg_cost_per_person": 15},
            {"doc_id": "doc_2", "category": "activity", "name": "Park",
             "description": "City park", "avg_cost_per_person": 0},
        ]
    }

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [{"id": "s1", "type": "activity", "name": "Museum", "doc_id": "doc_1", "depends_on": []}],
            "dag_edges": [],
            "days": 1
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]
        result = itinerary_builder_node(state)

    stop = result["itinerary"].stops[0]
    assert stop.fallback_options == []


def test_fallback_options_capped_at_three(profile_version):
    """At most 3 fallback options are staged per stop."""
    state = empty_state("session_1")
    state["profile"] = profile_version
    state["messages"] = [HumanMessage(content="1 day trip")]
    state["rag_context"] = {
        "req_1": [
            {"doc_id": f"doc_{i}", "category": "meal", "name": f"Restaurant {i}",
             "description": f"Desc {i}", "avg_cost_per_person": 20 + i,
             "constraint_flags": {"halal": True}}
            for i in range(6)
        ]
    }

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [{"id": "s1", "type": "meal", "name": "Restaurant 0", "doc_id": "doc_0", "depends_on": []}],
            "dag_edges": [],
            "days": 1
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]
        result = itinerary_builder_node(state)

    stop = result["itinerary"].stops[0]
    assert len(stop.fallback_options) <= 3
