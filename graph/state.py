from __future__ import annotations
from typing import Annotated, Optional, TypedDict
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from graph.reducers import dedup_append, latest_by_timestamp, merge_by_key
from models.profile import ProfileVersion
from models.itinerary import ItineraryVersion
from models.budget import BudgetLedger
from models.replanning import ReplanningContext


class TripState(TypedDict, total=False):
    session_id: str
    state_version: int
    profile: Optional[ProfileVersion]
    profile_history: Annotated[list[ProfileVersion], add]
    itinerary: Optional[ItineraryVersion]
    itinerary_history: Annotated[list[ItineraryVersion], add]
    budget_ledger: Optional[BudgetLedger]
    disruption_queue: Annotated[list[dict], dedup_append]
    active_disruption_id: Optional[str]
    rag_context: Annotated[dict[str, list], merge_by_key]
    live_data: Annotated[dict[str, dict], latest_by_timestamp]
    messages: Annotated[list[BaseMessage], add_messages]
    replanning_context: Optional[ReplanningContext]


def empty_state(session_id: str) -> TripState:
    return TripState(
        session_id=session_id,
        state_version=0,
        profile=None,
        profile_history=[],
        itinerary=None,
        itinerary_history=[],
        budget_ledger=None,
        disruption_queue=[],
        active_disruption_id=None,
        rag_context={},
        live_data={},
        messages=[],
        replanning_context=None,
    )
