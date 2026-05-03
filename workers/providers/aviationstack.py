from __future__ import annotations
import logging
import os
from datetime import date

import httpx

from models.disruption import NormalisedEvent
from models.profile import ConstraintProfile

log = logging.getLogger(__name__)

_BASE_URL = "http://api.aviationstack.com/v1/flights"

def poll(entity_ids: list[str], profile: ConstraintProfile) -> list[NormalisedEvent]:
    api_key = os.getenv("AVIATIONSTACK_API_KEY")
    if not api_key:
        log.warning("AVIATIONSTACK_API_KEY not set — skipping Aviationstack poll")
        return []

    events: list[NormalisedEvent] = []
    window = date.today().isoformat()

    for segment_id in entity_ids:
        try:
            resp = httpx.get(
                _BASE_URL,
                params={
                    "access_key": api_key,
                    "flight_iata": segment_id
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                status_code = _map_status(item)
                if status_code:
                    events.append(NormalisedEvent(
                        provider="aviationstack",
                        entity_id=segment_id,
                        status_code=status_code,
                        window=window,
                        raw_payload=item,
                    ))
        except httpx.HTTPError as exc:
            log.warning("Aviationstack poll failed for %s: %s", segment_id, exc)

    return events

def _map_status(item: dict) -> str | None:
    flight_status = item.get("flight_status", "")
    if flight_status == "cancelled":
        return "cancelled"
    
    departure = item.get("departure", {})
    delay = departure.get("delay") or 0
    if delay >= 120:
        return "delayed_major"
    if delay > 0:
        return "delayed_minor"
    return None
