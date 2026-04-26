from __future__ import annotations
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer():
    """
    Returns SqliteSaver in dev (POSTGRES_DSN not set).
    Returns PostgresSaver in prod — swap happens here only.
    """
    postgres_dsn = os.getenv("POSTGRES_DSN")
    if postgres_dsn:
        from langgraph.checkpoint.postgres import PostgresSaver
        return PostgresSaver.from_conn_string(postgres_dsn)
    conn = sqlite3.connect(".checkpoints.db", check_same_thread=False)
    return SqliteSaver(conn)
