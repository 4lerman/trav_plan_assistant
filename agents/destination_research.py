from anthropic import Anthropic
from langchain_core.messages import AIMessage
import uuid

from graph.state import TripState
from rag.retriever import retrieve, NoFeasibleResultsError
from rag.reranker import rerank

_client = Anthropic()
_MODEL = "claude-haiku-4-5-20251001"

def _format_no_results(e: NoFeasibleResultsError) -> str:
    return f"I couldn't find any destinations matching your constraints for '{e.query}'. Can we relax some constraints?"

def _format_research_summary(ranked: list[dict]) -> str:
    names = [d.get("name", "Unknown") for d in ranked[:3]]
    return f"I've found some great options, including {', '.join(names)}."

def run_destination_research(state: TripState) -> dict:
    if not state.get("profile"):
        # Should not happen if routing is correct, but fail gracefully
        return {}
        
    profile_ver = state["profile"]
    profile = profile_ver.profile
    
    # Extract destination query from last user message
    human_messages = [m for m in state["messages"] if m.type == "human"]
    if not human_messages:
        return {}
    last_msg = human_messages[-1].content
    
    request_id = str(uuid.uuid4())
    
    try:
        candidates = retrieve(last_msg, profile, request_id)
        ranked = rerank(last_msg, candidates, top_n=10)
    except NoFeasibleResultsError as e:
        return {"messages": [AIMessage(content=_format_no_results(e))]}
        
    for doc in ranked:
        # confidence = 0.5 * source_reliability + 0.3 * recency_factor + 0.2 * cross_source_agreement
        rel = doc.get("source_reliability", 0.8)
        doc["confidence_score"] = 0.5 * rel + 0.5 # basic stub
        doc["source"] = "corpus"
        # TODO: enrich from live_data if fetched_at within 7 days
        
    return {
        "rag_context": {request_id: ranked},
        "messages": [AIMessage(content=_format_research_summary(ranked))],
        "state_version": state.get("state_version", 0) + 1,
    }

def destination_research_node(state: TripState) -> dict:
    return run_destination_research(state)
