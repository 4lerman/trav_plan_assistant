from __future__ import annotations
import logging
import os
from datetime import date

import httpx

from models.disruption import NormalisedEvent
from models.profile import ConstraintProfile

log = logging.getLogger(__name__)

_BASE_URL = "https://external.transitapp.com/v3/public/routes"

def poll(entity_ids: list[str], profile: ConstraintProfile) -> list[NormalisedEvent]:
    api_key = os.getenv("TRANSIT_API_KEY")
    if not api_key:
        log.warning("TRANSIT_API_KEY not set — skipping Transit poll")
        return []

    events: list[NormalisedEvent] = []
    headers = {
        "apiKey": api_key,
    }
    window = date.today().isoformat()

    for route_id in entity_ids:
        try:
            resp = httpx.get(
                f"{_BASE_URL}/{route_id}/alerts",
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            for item in resp.json().get("alerts", []):
                status_code = _map_status(item.get("effect", ""))
                if status_code:
                    events.append(NormalisedEvent(
                        provider="transit",
                        entity_id=route_id,
                        status_code=status_code,
                        window=window,
                        raw_payload=item,
                    ))
        except httpx.HTTPError as exc:
            log.warning("Transit poll failed for %s: %s", route_id, exc)

    return events

def _map_status(effect: str) -> str | None:
    effect = effect.lower()
    if effect in ["no_service", "strike", "significant_delays"]:
        return "service_suspended"
    if effect in ["reduced_service", "detour", "modified_service"]:
        return "service_modified"
    return None
