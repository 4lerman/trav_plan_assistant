from __future__ import annotations
import logging
import os
from datetime import date

import httpx

from models.disruption import NormalisedEvent
from models.profile import ConstraintProfile

log = logging.getLogger(__name__)

_BASE_URL = "https://test.api.amadeus.com/v1/safety/safety-rated-locations"

def poll(entity_ids: list[str], profile: ConstraintProfile) -> list[NormalisedEvent]:
    api_key = os.getenv("AMADEUS_API_KEY")
    if not api_key:
        log.warning("AMADEUS_API_KEY not set — skipping Advisories poll")
        return []

    events: list[NormalisedEvent] = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    window = date.today().isoformat()

    for location_id in entity_ids:
        try:
            resp = httpx.get(
                _BASE_URL,
                headers=headers,
                params={"keyword": location_id},
                timeout=10.0,
            )
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                safety_score = item.get("safetyScores", {}).get("overall", 0)
                status_code = _map_status(safety_score)
                if status_code:
                    events.append(NormalisedEvent(
                        provider="advisories",
                        entity_id=location_id,
                        status_code=status_code,
                        window=window,
                        raw_payload=item,
                    ))
        except httpx.HTTPError as exc:
            log.warning("Advisories poll failed for %s: %s", location_id, exc)

    return events

def _map_status(overall_score: int) -> str | None:
    if overall_score >= 75:
        return "high_risk"
    if overall_score >= 50:
        return "medium_risk"
    return None
