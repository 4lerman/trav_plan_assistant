from __future__ import annotations
from models.disruption import DisruptionSeverity, NormalisedEvent

# Rule table — replace this function body with LLM classifier when upgrading.
# Public signature must stay identical: (NormalisedEvent) -> DisruptionSeverity | None
def evaluate(event: NormalisedEvent) -> DisruptionSeverity | None:
    if event.provider == "amadeus":
        if event.status_code == "cancelled":
            return DisruptionSeverity.CRITICAL
        if event.status_code == "delayed_major":
            return DisruptionSeverity.CRITICAL
        if event.status_code == "delayed_minor":
            return DisruptionSeverity.WARNING
    elif event.provider == "wheelmap":
        if event.status_code == "inaccessible":
            return DisruptionSeverity.CRITICAL
        if event.status_code == "limited_access":
            return DisruptionSeverity.WARNING
        if event.status_code == "accessible":
            return None
    return None
