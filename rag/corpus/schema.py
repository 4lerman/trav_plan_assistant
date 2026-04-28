from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class AccessibilityInfo(BaseModel):
    wheelchair_accessible: bool
    step_free_routes: bool
    accessible_restrooms: bool
    notes: str = ""


class DestinationDoc(BaseModel):
    doc_id: str                        # unique, stable across re-ingests
    name: str
    city: str
    country: str
    category: str                      # "restaurant" | "hotel" | "attraction" | "transit"
    description: str                   # free text — embedded
    dietary_tags: list[str] = []       # controlled vocabulary (matches dietary_tags.yaml)
    accessibility: AccessibilityInfo
    avg_cost_per_person: float         # in USD
    currency: str = "USD"
    source: str                        # "corpus" | "wheelmap" | "booking" etc.
    source_reliability: float = 0.8   # 0.0–1.0
    last_verified: str                 # ISO 8601 date string
