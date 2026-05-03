from __future__ import annotations
from anthropic import Anthropic
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from graph.state import TripState
from graph.checkpointer import get_checkpointer

_anthropic = Anthropic()


def route(state: dict) -> str:
    """Priority-ordered routing — deterministic Python, no LLM."""
    if state.get("active_disruption_id") or state.get("disruption_queue"):
        return "replanning"
    if state.get("profile") is None:
        return "constraint_profiler"
    has_rag = any(v for v in state.get("rag_context", {}).values())
    if state.get("itinerary") is None and has_rag:
        return "itinerary_builder"
    if state.get("itinerary") is None:
        return "destination_research"
    return "orchestrator_reply"


def _stub_node(name: str):
    def node(state: TripState) -> dict:
        return {"messages": [AIMessage(content=f"[{name} stub — not yet implemented]")]}

    node.__name__ = name
    return node


def orchestrator_reply_node(state: TripState) -> dict:
    """Answer conversational questions using current trip state as context."""
    profile = state.get("profile")
    itinerary = state.get("itinerary")

    profile_summary = "No profile collected yet."
    if profile:
        p = profile.profile
        tags = ", ".join(p.dietary_tags) if p.dietary_tags else "none"
        profile_summary = (
            f"Mobility: {p.mobility_level}, dietary tags: {tags}, "
            f"budget: {p.daily_budget} {p.base_currency}/day, "
            f"language: {p.language}, "
            f"accommodation flexibility: {p.accommodation_flexibility}, "
            f"disruption tolerance: {p.disruption_tolerance}."
        )

    itinerary_summary = "No itinerary built yet."
    if itinerary:
        stop_names = [s.name for s in itinerary.stops[:5]]
        itinerary_summary = (
            f"{itinerary.days}-day itinerary, "
            f"{len(itinerary.stops)} stops. First stops: {', '.join(stop_names)}."
        )

    system = (
        "You are the Adaptive Travel Companion. Answer the user's question concisely "
        "using the trip context below. The data below is real and verified — do not "
        "invent uncertainty or disclaim capabilities beyond what the context shows.\n\n"
        f"PROFILE: {profile_summary}\n"
        f"ITINERARY: {itinerary_summary}"
    )

    messages = state.get("messages", [])
    conversation = []
    for m in messages:
        role = "user" if m.__class__.__name__ == "HumanMessage" else "assistant"
        conversation.append({"role": role, "content": m.content})

    # Ensure conversation starts with a user message
    if not conversation or conversation[0]["role"] != "user":
        conversation = [{"role": "user", "content": "Hello"}] + conversation

    response = _anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=conversation,
    )
    reply = response.content[0].text
    return {"messages": [AIMessage(content=reply)]}


def build_graph():
    builder = StateGraph(TripState)

    from agents.constraint_profiler import constraint_profiler_node
    from agents.destination_research import destination_research_node
    from agents.itinerary_builder import itinerary_builder_node

    builder.add_node("constraint_profiler", constraint_profiler_node)
    builder.add_node("destination_research", destination_research_node)
    builder.add_node("itinerary_builder", itinerary_builder_node)
    builder.add_node("replanning", _stub_node("replanning"))
    builder.add_node("orchestrator_reply", orchestrator_reply_node)

    builder.set_conditional_entry_point(route)

    builder.add_edge("constraint_profiler", END)
    builder.add_edge("replanning", END)
    builder.add_edge("destination_research", "itinerary_builder")
    builder.add_edge("itinerary_builder", END)
    builder.add_edge("orchestrator_reply", END)

    return builder.compile(checkpointer=get_checkpointer())


graph = build_graph()
