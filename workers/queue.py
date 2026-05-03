from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone

from models.disruption import DisruptionEvent

_DB_PATH = "disruption_queue.db"
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_key   TEXT PRIMARY KEY,
            payload     TEXT NOT NULL,
            enqueued_at TEXT NOT NULL,
            processed   INTEGER DEFAULT 0
        )
    """)
    conn.commit()


def enqueue(event: DisruptionEvent) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO events (event_key, payload, enqueued_at) VALUES (?, ?, ?)",
        (
            event.event_key,
            event.model_dump_json(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def dequeue_pending() -> list[DisruptionEvent]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT payload FROM events WHERE processed = 0"
    ).fetchall()
    return [DisruptionEvent.model_validate_json(row[0]) for row in rows]


def mark_processed(event_key: str) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE events SET processed = 1 WHERE event_key = ?",
        (event_key,),
    )
    conn.commit()
