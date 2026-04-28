from __future__ import annotations
import os
import sqlite3
from typing import TYPE_CHECKING
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

if TYPE_CHECKING:
    # langgraph-checkpoint-postgres is an optional prod dependency.
    # This import is for type analysis only — never executed at runtime here.
    from langgraph.checkpoint.postgres import PostgresSaver  # noqa: F401

# All custom types stored in checkpoints must be listed here so LangGraph's
# msgpack deserializer can reconstruct them without warnings (or future errors
# when LANGGRAPH_STRICT_MSGPACK=true is set).
_ALLOWED_MSGPACK_MODULES = [
    ("models.profile", "MobilityLevel"),
    ("models.profile", "AccommodationFlexibility"),
    ("models.profile", "DisruptionTolerance"),
    ("models.profile", "ProfileVersion"),
    ("models.profile", "ConstraintProfile"),
    ("models.itinerary", "Stop"),
    ("models.itinerary", "StopType"),
    ("models.itinerary", "ItineraryVersion"),
    ("models.budget", "BudgetLedger"),
]

_serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


def get_checkpointer():
    """
    Returns SqliteSaver in dev (POSTGRES_DSN not set).
    Returns PostgresSaver in prod — swap happens here only.
    Install prod dep: uv add langgraph-checkpoint-postgres
    """
    postgres_dsn = os.getenv("POSTGRES_DSN")
    if postgres_dsn:
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import-not-found]
        return PostgresSaver.from_conn_string(postgres_dsn)
    conn = sqlite3.connect(".checkpoints.db", check_same_thread=False)
    return SqliteSaver(conn, serde=_serde)
