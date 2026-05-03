from __future__ import annotations
import logging
import os
from datetime import date

import httpx

from models.disruption import NormalisedEvent
from models.profile import ConstraintProfile

log = logging.getLogger(__name__)

_BASE_URL = "https://wheelmap.org"

_WHEELCHAIR_MAP = {
    "yes": "accessible",
    "limited": "limited_access",
    "no": "inaccessible",
}


def poll(entity_ids: list[str], profile: ConstraintProfile) -> list[NormalisedEvent]:
    api_key = os.getenv("WHEELMAP_API_KEY")
    if not api_key:
        log.warning("WHEELMAP_API_KEY not set — skipping Wheelmap poll")
        return []

    events: list[NormalisedEvent] = []
    window = date.today().isoformat()

    for node_id in entity_ids:
        try:
            resp = httpx.get(
                f"{_BASE_URL}/api/nodes/{node_id}",
                params={"api_key": api_key},
                timeout=10.0,
            )
            resp.raise_for_status()
            node = resp.json().get("node", {})
            wheelchair = node.get("wheelchair")
            status_code = _WHEELCHAIR_MAP.get(wheelchair)
            if status_code:
                events.append(NormalisedEvent(
                    provider="wheelmap",
                    entity_id=node_id,
                    status_code=status_code,
                    window=window,
                    raw_payload=node,
                ))
        except httpx.HTTPError as exc:
            log.warning("Wheelmap poll failed for node %s: %s", node_id, exc)

    return events
