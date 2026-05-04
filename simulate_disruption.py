import uuid
import os
from datetime import datetime, timezone

from models.disruption import DisruptionEvent, DisruptionSeverity
from workers.queue import enqueue

def main():
    event = DisruptionEvent(
        event_key=f"evt_{uuid.uuid4().hex[:8]}",
        provider="aviationstack",
        entity_id="doc_001",
        status_code="CLOSED",
        severity=DisruptionSeverity.CRITICAL,
        detected_at=datetime.now(timezone.utc),
        raw_payload={"reason": "Emergency maintenance"}
    )
    
    enqueue(event)
    
    with open(".wake_signal", "w") as f:
        f.write("disruption")
        
    print(f"Simulated disruption {event.event_key} for entity '{event.entity_id}'")
    print("Run `uv run python cli.py` to trigger the replanning pipeline.")

if __name__ == "__main__":
    main()
