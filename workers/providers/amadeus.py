from __future__ import annotations
import logging
import os
from datetime import date

import httpx

from models.disruption import NormalisedEvent
from models.profile import ConstraintProfile

log = logging.getLogger(__name__)

_BASE_URL = "https://test.api.amadeus.com/v2"


def poll(entity_ids: list[str], profile: ConstraintProfile) -> list[NormalisedEvent]:
    api_key = os.getenv("AMADEUS_API_KEY")
    if not api_key:
        log.warning("AMADEUS_API_KEY not set — skipping Amadeus poll")
        return []

    events: list[NormalisedEvent] = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    window = date.today().isoformat()

    for segment_id in entity_ids:
        try:
            resp = httpx.get(
                f"{_BASE_URL}/schedule/flights",
                headers=headers,
                params={"carrierCode": segment_id[:2], "flightNumber": segment_id[2:]},
                timeout=10.0,
            )
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                status_code = _map_status(item)
                if status_code:
                    events.append(NormalisedEvent(
                        provider="amadeus",
                        entity_id=segment_id, # Use segment_id as the entity_id
                        status_code=status_code,
                        window=window,
                        raw_payload=item,
                    ))
        except httpx.HTTPError as exc:
            log.warning("Amadeus poll failed for %s: %s", segment_id, exc)

    return events


def _map_status(item: dict) -> str | None:
    status = item.get("status", "")
    if status == "cancelled":
        return "cancelled"
    delay = item.get("delay_minutes", 0) or 0
    if delay >= 120:
        return "delayed_major"
    if delay > 0:
        return "delayed_minor"
    return None
