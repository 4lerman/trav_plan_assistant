import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from agents.itinerary_builder import itinerary_builder_node
from graph.state import empty_state

def test_builder_emits_itinerary_version(sample_profile_version):
    state = empty_state("session_1")
    state["profile"] = sample_profile_version
    state["messages"] = [HumanMessage(content="3 day trip")]
    state["rag_context"] = {"req_1": [{"doc_id": "doc_1", "category": "meal", "name": "Rest"}]}
    
    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [{"id": "s1", "type": "meal", "name": "Rest", "doc_id": "doc_1"}],
            "dag_edges": [],
            "days": 3
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]
        
        result = itinerary_builder_node(state)
        
        assert "itinerary" in result
        assert result["itinerary"].days == 3
        assert result["itinerary"].profile_version_id == sample_profile_version.version_id
        assert result["rag_context"]["req_1"] == []

def test_builder_stages_fallbacks(sample_profile_version):
    state = empty_state("session_1")
    state["profile"] = sample_profile_version
    state["messages"] = [HumanMessage(content="1 day trip")]
    state["rag_context"] = {
        "req_1": [
            {"doc_id": "doc_1", "category": "meal", "name": "Rest"},
            {"doc_id": "doc_2", "category": "meal", "name": "Alt 1"},
            {"doc_id": "doc_3", "category": "meal", "name": "Alt 2"}
        ]
    }

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [{"id": "s1", "type": "meal", "name": "Rest", "doc_id": "doc_1"}],
            "dag_edges": [],
            "days": 1
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]

        result = itinerary_builder_node(state)

        stop = result["itinerary"].stops[0]
        assert len(stop.fallback_alternatives) == 2
        assert stop.fallback_alternatives[0].doc_id == "doc_2"


def test_builder_dag_edges_stored(sample_profile_version):
    state = empty_state("session_1")
    state["profile"] = sample_profile_version
    state["messages"] = [HumanMessage(content="2 day trip")]
    state["rag_context"] = {
        "req_1": [
            {"doc_id": "doc_1", "category": "hotel", "name": "Hotel"},
            {"doc_id": "doc_2", "category": "meal", "name": "Rest"},
        ]
    }

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [
                {"id": "s1", "type": "lodging", "name": "Hotel", "doc_id": "doc_1", "depends_on": []},
                {"id": "s2", "type": "meal", "name": "Rest", "doc_id": "doc_2", "depends_on": ["s1"]}
            ],
            "dag_edges": [["s1", "s2"]],
            "days": 2
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]

        result = itinerary_builder_node(state)

    assert ("s1", "s2") in result["itinerary"].dag_edges
    assert result["itinerary"].stops[1].depends_on == ["s1"]


def test_builder_skips_when_no_profile():
    state = empty_state("session_1")
    result = itinerary_builder_node(state)
    assert result == {}


def test_builder_returns_error_message_on_bad_llm_output(sample_profile_version):
    state = empty_state("session_1")
    state["profile"] = sample_profile_version
    state["messages"] = [HumanMessage(content="3 day trip")]
    state["rag_context"] = {"req_1": [{"doc_id": "doc_1", "category": "meal", "name": "Rest"}]}

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = "Sorry, I cannot help with that."  # no <itinerary> tags
        mock_create.return_value.content = [mock_reply]

        result = itinerary_builder_node(state)

    assert "itinerary" not in result
    assert isinstance(result["messages"][0], AIMessage)


def test_builder_version_increments_from_history(sample_profile_version):
    from datetime import datetime
    from models.itinerary import ItineraryVersion

    prev = ItineraryVersion(version_id=5, created_at=datetime.utcnow(), days=1)
    state = empty_state("session_1")
    state["profile"] = sample_profile_version
    state["messages"] = [HumanMessage(content="1 day trip")]
    state["rag_context"] = {"req_1": [{"doc_id": "doc_1", "category": "meal", "name": "Rest"}]}
    state["itinerary_history"] = [prev]

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [{"id": "s1", "type": "meal", "name": "Rest", "doc_id": "doc_1"}],
            "dag_edges": [],
            "days": 1
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]

        result = itinerary_builder_node(state)

    assert result["itinerary"].version_id == 6
