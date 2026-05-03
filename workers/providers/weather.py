from __future__ import annotations
import logging
import os
from datetime import date

import httpx

from models.disruption import NormalisedEvent
from models.profile import ConstraintProfile

log = logging.getLogger(__name__)

_BASE_URL = "https://api.openweathermap.org/data/2.5"

def poll(entity_ids: list[str], profile: ConstraintProfile) -> list[NormalisedEvent]:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        log.warning("OPENWEATHER_API_KEY not set — skipping Weather poll")
        return []

    events: list[NormalisedEvent] = []
    window = date.today().isoformat()

    for location in entity_ids:
        try:
            resp = httpx.get(
                f"{_BASE_URL}/weather",
                params={"q": location, "appid": api_key},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            
            weather_items = data.get("weather", [])
            for w in weather_items:
                status_code = _map_status(w.get("id", 0))
                if status_code:
                    events.append(NormalisedEvent(
                        provider="weather",
                        entity_id=location,
                        status_code=status_code,
                        window=window,
                        raw_payload=data,
                    ))
        except httpx.HTTPError as exc:
            log.warning("Weather poll failed for %s: %s", location, exc)

    return events

def _map_status(condition_id: int) -> str | None:
    if condition_id in [212, 221, 504, 602, 781]:
        return "extreme_weather"
    if 200 <= condition_id < 300 or 600 <= condition_id < 602:
        return "severe_weather"
    return None
