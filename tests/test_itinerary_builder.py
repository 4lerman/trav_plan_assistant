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
