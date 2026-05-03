from __future__ import annotations
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone

from models.disruption import DisruptionEvent, make_event_key
from models.itinerary import ItineraryVersion
from models.profile import ConstraintProfile, ProfileVersion
from workers import queue
from workers.disruption_rules import evaluate
from workers.providers import aviationstack, weather, transit

log = logging.getLogger(__name__)

WAKE_SIGNAL_PATH = ".wake_signal"
_CHECKPOINT_DB = ".checkpoints.db"


def _load_entity_ids_from_checkpoint() -> dict[str, list[str]]:
    """Read itinerary stop IDs from the LangGraph SQLite checkpoint (read-only)."""
    result: dict[str, list[str]] = {"aviationstack": [], "weather": [], "transit": []}
    try:
        conn = sqlite3.connect(f"file:{_CHECKPOINT_DB}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT v FROM writes ORDER BY task_id DESC LIMIT 50"
        ).fetchall()
        conn.close()
        for (v,) in rows:
            try:
                data = json.loads(v)
                itinerary_data = data.get("itinerary")
                if itinerary_data:
                    itinerary = ItineraryVersion.model_validate(itinerary_data)
                    for stop in itinerary.stops:
                        if stop.type.value == "transit" and stop.doc_id:
                            if stop.doc_id.startswith("flight_"):
                                result["aviationstack"].append(stop.doc_id)
                            else:
                                result["transit"].append(stop.doc_id)
                        elif stop.type.value == "destination":
                            result["weather"].append(stop.name)
                    break
            except Exception:
                continue
    except Exception as exc:
        log.warning("Could not read checkpoint for entity IDs: %s", exc)
    return result


def _load_profile_from_checkpoint() -> ConstraintProfile | None:
    """Read current ConstraintProfile from the LangGraph SQLite checkpoint (read-only)."""
    try:
        conn = sqlite3.connect(f"file:{_CHECKPOINT_DB}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT v FROM writes ORDER BY task_id DESC LIMIT 50"
        ).fetchall()
        conn.close()
        for (v,) in rows:
            try:
                data = json.loads(v)
                profile_data = data.get("profile")
                if profile_data:
                    return ProfileVersion.model_validate(profile_data).profile
            except Exception:
                continue
    except Exception as exc:
        log.warning("Could not read checkpoint for profile: %s", exc)
    return None


def _write_wake_signal() -> None:
    open(WAKE_SIGNAL_PATH, "w").close()


def _run_once() -> None:
    """Single poll cycle — separated from the sleep loop for testability."""
    entity_ids = _load_entity_ids_from_checkpoint()
    profile = _load_profile_from_checkpoint()

    providers = [
        (aviationstack.poll, entity_ids["aviationstack"]),
        (weather.poll, entity_ids["weather"]),
        (transit.poll, entity_ids["transit"]),
    ]

    for provider_poll, ids in providers:
        try:
            events = provider_poll(ids, profile)
        except Exception as exc:
            log.warning("Provider poll failed: %s", exc)
            continue

        for event in events:
            severity = evaluate(event)
            if severity is None:
                continue
            disruption = DisruptionEvent(
                event_key=make_event_key(
                    provider=event.provider,
                    entity_id=event.entity_id,
                    status_code=event.status_code,
                    window=event.window,
                ),
                provider=event.provider,
                entity_id=event.entity_id,
                status_code=event.status_code,
                severity=severity,
                detected_at=datetime.now(timezone.utc),
                raw_payload=event.raw_payload,
            )
            queue.enqueue(disruption)
            log.info("Enqueued disruption: %s / %s", event.provider, event.status_code)
            _write_wake_signal()


def run(poll_interval_seconds: int = 60) -> None:
    logging.basicConfig(level=logging.INFO)
    log.info("Live Data Worker started (poll interval: %ss)", poll_interval_seconds)
    while True:
        _run_once()
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    run()
