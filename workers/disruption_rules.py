from __future__ import annotations
from models.disruption import DisruptionSeverity, NormalisedEvent

def evaluate(event: NormalisedEvent) -> DisruptionSeverity | None:
    if event.provider == "aviationstack":
        if event.status_code == "cancelled":
            return DisruptionSeverity.CRITICAL
        if event.status_code == "delayed_major":
            return DisruptionSeverity.CRITICAL
        if event.status_code == "delayed_minor":
            return DisruptionSeverity.WARNING
    elif event.provider == "weather":
        if event.status_code == "extreme_weather":
            return DisruptionSeverity.CRITICAL
        if event.status_code == "severe_weather":
            return DisruptionSeverity.WARNING
    elif event.provider == "advisories":
        if event.status_code == "high_risk":
            return DisruptionSeverity.CRITICAL
        if event.status_code == "medium_risk":
            return DisruptionSeverity.WARNING
    elif event.provider == "transit":
        if event.status_code == "service_suspended":
            return DisruptionSeverity.CRITICAL
        if event.status_code == "service_modified":
            return DisruptionSeverity.WARNING
    return None
