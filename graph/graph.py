from __future__ import annotations
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from graph.state import TripState
from graph.checkpointer import get_checkpointer


def route(state: dict) -> str:
    """Priority-ordered routing — deterministic Python, no LLM."""
    if state.get("active_disruption_id") or state.get("disruption_queue"):
        return "replanning"
    if state.get("profile") is None:
        return "constraint_profiler"
    if state.get("itinerary") is None and state.get("rag_context"):
        return "itinerary_builder"
    if state.get("itinerary") is None:
        return "destination_research"
    return "orchestrator_reply"


def _stub_node(name: str):
    def node(state: TripState) -> dict:
        return {"messages": [AIMessage(content=f"[{name} stub — not yet implemented]")]}
    node.__name__ = name
    return node


def build_graph():
    builder = StateGraph(TripState)

    # Real node — added in Task 6
    from agents.constraint_profiler import constraint_profiler_node
    builder.add_node("constraint_profiler", constraint_profiler_node)

    from agents.destination_research import destination_research_node
    from agents.itinerary_builder import itinerary_builder_node
    
    builder.add_node("destination_research", destination_research_node)
    builder.add_node("itinerary_builder", itinerary_builder_node)
    builder.add_node("replanning", _stub_node("replanning"))
    builder.add_node("orchestrator_reply", _stub_node("orchestrator_reply"))

    builder.set_conditional_entry_point(route)

    builder.add_edge("constraint_profiler", END)
    builder.add_edge("replanning", END)
    builder.add_edge("destination_research", "itinerary_builder")
    builder.add_edge("itinerary_builder", END)
    builder.add_edge("orchestrator_reply", END)

    return builder.compile(checkpointer=get_checkpointer())


graph = build_graph()
