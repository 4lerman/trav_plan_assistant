# Phase 1 Design: Foundation — Graph, State, Constraint Profiler

**Date:** 2026-04-26
**Scope:** W1–2 of the Adaptive Travel Companion roadmap
**Approach:** Core-first, bottom-up (models → reducers → graph → agent → CLI)

---

## 1. Project Setup

**Package manager:** `uv`

**`pyproject.toml` dependency groups:**
- Core: `langgraph`, `anthropic`, `pydantic>=2`, `langsmith`
- Dev: `pytest`, `pytest-asyncio`, `hypothesis`

**Checkpointer:** SQLite in dev (`POSTGRES_DSN` not set), Postgres in prod. The swap is isolated to `graph/checkpointer.py::get_checkpointer()` — nothing else changes.

**Environment:** `.env.example` documents all required variables. Only `ANTHROPIC_API_KEY` and `LANGSMITH_*` are needed for Phase 1.

---

## 2. Package Structure (Phase 1 only)

```
trav_planner_assistant/
├── pyproject.toml
├── .env.example
├── models/
│   ├── __init__.py
│   ├── profile.py        # ProfileVersion, ConstraintProfile, all enums
│   ├── itinerary.py      # ItineraryVersion + Stop (fully typed stub)
│   ├── disruption.py     # DisruptionEvent (fully typed stub)
│   └── budget.py         # BudgetLedger + remaining_for() (fully typed stub)
├── graph/
│   ├── __init__.py
│   ├── state.py          # TripState TypedDict with Annotated reducers
│   ├── reducers.py       # dedup_append, latest_by_timestamp, merge_by_key
│   ├── graph.py          # StateGraph, routing rule table, stub nodes
│   └── checkpointer.py   # get_checkpointer() — SQLite/Postgres switch
├── agents/
│   ├── __init__.py
│   └── constraint_profiler.py
├── rag/
│   └── vocabulary/
│       └── dietary_tags.yaml   # controlled vocabulary (initial set)
├── cli.py                # python -m trav_planner_assistant entry point
└── tests/
    ├── test_models.py
    ├── test_reducers.py
    ├── test_profiler.py   # integration, skipped without ANTHROPIC_API_KEY
    └── test_graph.py
```

**Stub models** (`ItineraryVersion`, `DisruptionEvent`, `BudgetLedger`) have their full Pydantic v2 schemas defined now so `TripState` is accurate from day one. They just have no agents driving them yet.

---

## 3. Shared Models (`models/`)

### `models/profile.py`

```python
class MobilityLevel(str, Enum):
    FULL = "full"        # step-free everywhere required
    PARTIAL = "partial"  # some steps tolerable
    NONE = "none"        # no mobility restriction

class AccommodationFlexibility(str, Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    FLEXIBLE = "flexible"

class DisruptionTolerance(str, Enum):
    LOW = "low"      # replan immediately
    MEDIUM = "medium"  # notify, replan within 2h
    HIGH = "high"    # notify only

class ConstraintProfile(BaseModel):
    mobility_level: MobilityLevel
    dietary_tags: list[str]            # validated against controlled vocab at runtime
    medical_needs: list[str]           # PII-sensitive; "what is needed", not clinical
    daily_budget: Decimal
    base_currency: str                 # ISO 4217
    accommodation_flexibility: AccommodationFlexibility
    disruption_tolerance: DisruptionTolerance
    language: str                      # BCP 47
    offline_max_relaxation: int = 2    # relaxation ladder steps allowed without user confirmation

class ProfileVersion(BaseModel):
    version_id: int                    # monotonic; managed by a factory function
    created_at: datetime
    profile: ConstraintProfile
    diff: dict | None = None           # what changed from previous version
    consent_recorded: bool = False     # must be True before medical_needs is populated
```

**Key decisions:**
- `dietary_tags` is `list[str]` (not an enum) because the controlled vocabulary is versioned YAML and will grow without code changes.
- `medical_needs` is `list[str]` to stay flexible while being explicitly flagged for PII redaction.
- `offline_max_relaxation` lives on the profile — the user sets their own offline tolerance during intake.

### `models/itinerary.py` (stub)

`ItineraryVersion`: `version_id`, `created_at`, `stops: list[Stop]`, `validation_report: dict`.
`Stop`: `id`, `type` (enum: TRANSIT/LODGING/ACTIVITY/MEAL), `constraint_flags: dict`, `budget_estimate: Decimal`.

### `models/disruption.py` (stub)

`DisruptionEvent`: `event_key: str` (dedup hash), `provider`, `entity_id`, `status_code`, `severity` (enum: CRITICAL/WARNING/ADVISORY), `detected_at: datetime`, `raw_payload: dict`.

### `models/budget.py` (stub)

`BudgetLedger`: `daily_cap: Decimal`, `base_currency: str`, `spent_by_day: dict[date, Decimal]`, `committed_by_day: dict[date, Decimal]`, `bookings: list[Booking]`.
`remaining_for(day: date) -> Decimal` implemented: `daily_cap - spent_by_day[day] - committed_by_day[day]`.

---

## 4. Graph State & Reducers (`graph/`)

### `graph/reducers.py`

Three reducers — each has a clear contract:

```python
def merge_by_key(existing: dict, new: dict) -> dict:
    """For rag_context: merge dicts, new keys win."""
    return {**existing, **new}

def latest_by_timestamp(existing: dict, new: dict) -> dict:
    """For live_data: per-key, keep whichever entry has newer fetched_at."""
    result = dict(existing)
    for k, v in new.items():
        if k not in result or v["fetched_at"] > result[k]["fetched_at"]:
            result[k] = v
    return result

def dedup_append(existing: list, new: list) -> list:
    """For disruption_queue: append only if event_key not already present."""
    seen = {e["event_key"] for e in existing}
    return existing + [e for e in new if e["event_key"] not in seen]
```

All three are verified via Hypothesis property tests for commutativity/idempotency — this is the correctness guarantee for the concurrency model.

### `graph/state.py`

```python
class TripState(TypedDict):
    session_id: str
    state_version: int
    profile: Optional[ProfileVersion]
    profile_history: Annotated[list[ProfileVersion], add]
    itinerary: Optional[ItineraryVersion]
    itinerary_history: Annotated[list[ItineraryVersion], add]
    budget_ledger: Optional[BudgetLedger]
    disruption_queue: Annotated[list[DisruptionEvent], dedup_append]
    active_disruption_id: Optional[str]
    rag_context: Annotated[dict[str, list], merge_by_key]
    live_data: Annotated[dict[str, dict], latest_by_timestamp]
    messages: Annotated[list[BaseMessage], add_messages]
```

### `graph/checkpointer.py`

```python
def get_checkpointer():
    if os.getenv("POSTGRES_DSN"):
        return PostgresSaver.from_conn_string(os.environ["POSTGRES_DSN"])
    return SqliteSaver.from_conn_string(".checkpoints.db")
```

Only place the dev/prod split lives.

### `graph/graph.py`

Priority-ordered routing function (deterministic Python, no LLM):

```python
def route(state: TripState) -> str:
    if state["active_disruption_id"] or state["disruption_queue"]:
        return "replanning"            # stub node in Phase 1
    if state["profile"] is None:
        return "constraint_profiler"
    if state["itinerary"] is None:
        return "destination_research"  # stub in Phase 1
    return "orchestrator_reply"
```

Stub nodes for `replanning`, `destination_research`, `orchestrator_reply` return state unchanged and emit a placeholder message.

---

## 5. Constraint Profiler Agent (`agents/constraint_profiler.py`)

**Model:** Claude Sonnet 4.6

**Trigger:** Orchestrator routes here when `profile is None` (rule 4) or user requests profile update (rule 3).

**Conversation order:**
1. Mobility requirements
2. Dietary restrictions / religious requirements
3. Medical needs (consent gate: explicit yes/no before collecting)
4. Daily budget + currency
5. Accommodation flexibility
6. Disruption tolerance + offline preference (`offline_max_relaxation`)

The profiler loops back to itself (conditional edge) until all required fields are collected and Pydantic validation passes.

**Prompt injection defence:** User input is fenced with `<user_input>...</user_input>` delimiters in the system prompt. Agent is instructed to treat anything inside as data, never instructions.

**`dietary_tags` mapping:**
- Sonnet extracts tags from free text
- Tags are matched against `rag/vocabulary/dietary_tags.yaml` (initial set: `halal`, `kosher`, `vegan`, `vegetarian`, `gluten_free`, `nut_free`; extended in Phase 3)
- Unrecognised tags are kept with `_unverified` suffix and logged — no silent discard, no crash

**Medical guardrail:**
- Sonnet collects *what is needed* (e.g. "insulin refrigeration"), not clinical details
- Any response that looks like clinical advice triggers an explicit redirect message
- `consent_recorded` is set to `True` only after explicit user consent

**On completion:**
- Constructs `ConstraintProfile` → `ProfileVersion` (monotonic `version_id`)
- Appends to `profile_history`
- Sets `profile` to new version
- Bumps `state_version`

---

## 6. CLI (`cli.py`)

```bash
python -m trav_planner_assistant
```

- Interactive stdin loop
- Fixed `session_id` so SQLite checkpoint persists across runs
- Prints assistant reply after each graph turn
- `Ctrl+C` exits cleanly

---

## 7. Testing Strategy

### `tests/test_reducers.py` — Hypothesis property tests
- `dedup_append` idempotency: appending same event twice = once
- `latest_by_timestamp` commutativity: write order doesn't change result
- `merge_by_key` associativity

### `tests/test_models.py` — Pydantic validation
- Valid `ConstraintProfile` constructs correctly
- Invalid `mobility_level` raises `ValidationError`
- Negative `daily_budget` rejected
- `ProfileVersion` factory produces monotonically increasing `version_id`
- `DisruptionEvent` `event_key` hash is stable across identical inputs

### `tests/test_graph.py` — routing + state
- Routes to `constraint_profiler` when `profile is None`
- Routes to `replanning` when `disruption_queue` non-empty
- `state_version` bumps on write
- SQLite checkpointer round-trips state correctly

### `tests/test_profiler.py` — integration (`@pytest.mark.integration`)
- Skipped unless `ANTHROPIC_API_KEY` is set
- Happy path: full intake produces valid `ProfileVersion`
- Partial input: profiler loops and asks follow-up
- Prompt injection in dietary free text: fencing holds
- Medical consent gate: profiler asks before collecting `medical_needs`
- Unrecognised dietary tag: kept with `_unverified` suffix

---

## 8. Out of Scope for Phase 1

- Postgres checkpointer (deferred to Phase 5)
- LangSmith PII redaction middleware (deferred to Phase 5)
- Destination Research, Itinerary Builder, Replanning, Live Data Worker
- Qdrant, Redis, MCP integrations
- Docker Compose dev stack
