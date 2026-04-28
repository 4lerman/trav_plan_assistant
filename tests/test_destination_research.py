import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from agents.destination_research import destination_research_node
from rag.retriever import NoFeasibleResultsError
from graph.state import empty_state

def test_research_writes_to_rag_context(sample_profile_version):
    state = empty_state("session_1")
    state["profile"] = sample_profile_version
    state["messages"] = [HumanMessage(content="find me a place")]
    
    with patch("agents.destination_research.retrieve") as mock_retrieve, \
         patch("agents.destination_research.rerank") as mock_rerank:
        
        mock_retrieve.return_value = [{"doc_id": "1", "name": "Place"}]
        mock_rerank.return_value = [{"doc_id": "1", "name": "Place", "rerank_score": 0.9}]
        
        result = destination_research_node(state)
        
        assert "rag_context" in result
        req_id = list(result["rag_context"].keys())[0]
        assert result["rag_context"][req_id][0]["doc_id"] == "1"
        assert result["rag_context"][req_id][0]["source"] == "corpus"
        assert isinstance(result["messages"][0], AIMessage)

def test_research_no_results_returns_proposal(sample_profile_version):
    state = empty_state("session_1")
    state["profile"] = sample_profile_version
    state["messages"] = [HumanMessage(content="find me a place")]
    
    with patch("agents.destination_research.retrieve") as mock_retrieve:
        mock_retrieve.side_effect = NoFeasibleResultsError(sample_profile_version.profile, "find me a place")
        
        result = destination_research_node(state)
        
        assert "rag_context" not in result
        assert "relax some constraints" in result["messages"][0].content.lower()

def test_research_skips_when_no_profile():
    state = empty_state("session_1")
    result = destination_research_node(state)
    assert result == {}
