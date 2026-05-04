# Phase 4 Design: Replanning Agent + BudgetLedger + Relaxation Ladder + Offline Fallback Staging

## Overview

Phase 4 implements the full replanning subsystem: a constraint-aware Replanning Agent that responds to disruption events, a sunk-cost-aware BudgetLedger with live Wise FX integration, offline fallback staging in the Itinerary Builder, and the complete relaxation ladder. Together these four components close the loop between Phase 3's disruption detection and a user-facing replanning proposal.

**Roadmap position:** W7–8

**Approach:** Sequential pipeline (Option A) — linear stages with clear input/output contracts, independently testable. BudgetLedger is a stateless service called by the Replanning Agent; fallback staging is a post-planning pass in the Itinerary Builder.

---

## 1. Architecture Overview

Four new components, all wired into the existing LangGraph graph:

| Component | File | Type |
|---|---|---|
| Replanning Agent | `agents/replanning.py` | LangGraph node |
| BudgetLedger | `budget/ledger.py` | Stateless service class |
| Offline fallback staging | `agents/itinerary_builder.py` (extension) | Post-planning pass |
| New Pydantic models | `models/budget.py`, `models/replanning.py`, `models/itinerary.py` (extension) | Pydantic v2 |

The Replanning Agent is triggered by the existing routing rules (priority 1 & 2 in CLAUDE.md): when `active_disruption_id` is set, the Orchestrator routes to `replanning`. No new routing logic is needed.

---

## 2. BudgetLedger + Wise FX Integration

### `budget/ledger.py` — `BudgetLedger` class

**Sunk-cost tracking:**
- Each `LedgerEntry` carries: `amount`, `currency`, `category` (`accommodation | transport | food | activity`), `is_sunk` (bool — true once payment is confirmed/non-refundable), `stop_id`, `timestamp`
- `remaining_budget(currency: str) -> Decimal` — subtracts only non-sunk spend from the daily cap; sunk costs are excluded from the replanning budget constraint
- `record_entry(entry: LedgerEntry) -> None` — appends to ledger; idempotent on `stop_id + category`

**FX model:**
- `FXRate` fetched via the existing `mcp/` client wrapper (rate limiting, circuit breakers, idempotency) — Wise API `GET /v1/rates?source={src}&target={tgt}`
- `card_markup_pct: float = 1.75` — applied on top of mid-market rate (standard Wise card markup)
- Rates cached in Redis with 15-minute TTL (`fx_rate:{src}:{tgt}`)
- On Wise API outage: serves last cached rate, sets `rate_is_stale: true` in score output; if no cached rate exists, raises `FXUnavailableError` and blocks the replanning cycle with an explicit user notification
- All amounts converted to user's `home_currency` (from `ConstraintProfile`) before scoring

**Budget scoring:**
- `score_candidate(candidate_cost: Decimal, currency: str) -> float`
- Returns `budget_score = 1 − max(0, overrun) / daily_cap` (clamped to [0, 1])
- Used by Replanning Agent as the `0.20` weight component of the utility function

### `models/budget.py`

```python
class LedgerEntry(BaseModel):
    entry_id: str
    stop_id: str
    category: Literal["accommodation", "transport", "food", "activity"]
    amount: Decimal
    currency: str  # ISO 4217
    is_sunk: bool
    timestamp: datetime

class FXRate(BaseModel):
    source: str
    target: str
    rate: Decimal
    card_markup_pct: float
    fetched_at: datetime
    is_stale: bool = False

class SunkCost(BaseModel):
    stop_id: str
    amount: Decimal
    currency: str
    reason: str  # e.g. "non-refundable hotel deposit"

class BudgetLedger(BaseModel):
    ledger_id: str
    home_currency: str
    daily_cap: Decimal
    entries: list[LedgerEntry] = []
    fx_rates: dict[str, FXRate] = {}  # key: "{src}:{tgt}"
```

`BudgetLedger` is stored as a new field in `TripState` and persisted via the checkpointer.

---

## 3. Offline Fallback Staging

### Extension to `agents/itinerary_builder.py`

After the primary itinerary DAG is composed and written to `TripState`, the builder runs a **post-planning fallback staging pass** before the node exits.

**Scope:** Only bookable stops — `stop_type in {accommodation, transport, restaurant}`. Free attractions are excluded.

**Per-stop process:**
1. Fire a constraint-pre-filtered RAG query identical to the one that found the primary stop
2. Exclude the primary result by `venue_id`
3. Take top 3 ranked results from the reranker
4. For each: verify it passes all hard constraints (wheelchair, halal/dietary, medical, budget)
5. Store passing results as `FallbackOption` in `stop.fallback_options`

**Constraint guarantee:** Every staged fallback passes the same hard constraint filter as the primary. If fewer than 3 pass, store what's available — count is best-effort, never padded.

**Timing:** The staging pass runs synchronously within the Itinerary Builder node after the primary plan is confirmed. It does not block the user-facing response (the primary itinerary is already in state before staging starts).

### `FallbackOption` model (added to `models/itinerary.py`)

```python
class FallbackOption(BaseModel):
    venue_id: str
    name: str
    stop_type: StopType
    rag_confidence: float
    estimated_cost: Decimal
    currency: str
    constraint_flags: dict[str, bool]  # constraint_name -> passes
    staged_at: datetime
```

`Stop.fallback_options: list[FallbackOption] = []` — new optional field on `Stop`.

---

## 4. Replanning Agent — Sequential Pipeline

### `agents/replanning.py`

Single LangGraph node. Runs when `active_disruption_id is not None`.

**Step 1 — Bind versions:**
- Reads `active_disruption_id`, fetches `DisruptionEvent` from disruption queue
- Records `ReplanningContext(profile_version_id, itinerary_version_id, disruption_event, affected_stop_ids)`
- Writes `replanning_context` to `TripState`
- If itinerary has been updated since the disruption was enqueued: replan against latest version, log version delta

**Step 2 — Identify affected subgraph:**
- Walks itinerary DAG: find all stops with `entity_id` matching the disruption
- Extend to downstream stops via `depends_on` edges
- Records `affected_stop_ids` in `ReplanningContext`
- Unaffected stops are never touched

**Step 3 — Score pre-staged fallbacks:**
- For each affected stop, iterate `stop.fallback_options` in `rag_confidence` order
- Score each via utility function:
  ```
  U(plan) = 0.55 · constraint_score
           + 0.20 · budget_score          # via BudgetLedger.score_candidate()
           + 0.15 · quality_score         # rag_confidence of fallback
           + 0.10 · disruption_blast_radius  # 1 − (affected_downstream_stops / total_stops), higher = less disruption
  ```
- `constraint_score` must be `1.0` for a plan to be emitted as a confirmed replan (not a proposal)
- Select highest-scoring feasible candidate

**Step 4 — RAG fallback:**
- Triggered if: no pre-staged fallbacks exist, OR all fallbacks score `U < 0.6`, OR itinerary predates Phase 4
- Fires live RAG query with constraint pre-filter, identical scoring
- This is the graceful degradation path — functionally equivalent to having fresh fallbacks

**Step 5 — Relaxation ladder:**
If no feasible plan after RAG fallback, walk ladder in order:

| Step | Action | Constraint touched |
|---|---|---|
| 1 | Widen search radius ≤30% | soft |
| 2 | Widen budget ≤15% | soft |
| 3 | Downgrade accommodation flexibility one step | soft |
| 4 | Loosen one secondary dietary tag | never primary halal/kosher/medical |
| 5 | Escalate to user | — |

Each step re-runs RAG + scoring. Each applied step is appended to `ReplanningResult.relaxation_steps`. A plan with `constraint_score < 1.0` is **always emitted as a proposal**, never silently applied.

**Step 6 — Emit:**
- Success path: append new `ItineraryVersion` to `itinerary_history`, clear `replanning_context`, clear `active_disruption_id`, mark disruption processed in queue
- Escalation path: write `ReplanningResult(escalated=True)` to state, clear `active_disruption_id` — Orchestrator surfaces to user via conversational reply node

---

## 5. Data Flow + State Changes

### New `TripState` fields

```python
budget_ledger: BudgetLedger | None  # persisted via checkpointer
replanning_context: ReplanningContext | None  # ephemeral, cleared post-replan
```

### `models/replanning.py`

```python
class ReplanningContext(BaseModel):
    profile_version_id: str
    itinerary_version_id: str
    disruption_event: DisruptionEvent
    affected_stop_ids: list[str]
    started_at: datetime

class RelaxationStep(BaseModel):
    step_number: int  # 1–5
    description: str
    constraint_relaxed: str | None
    utility_score_after: float

class ReplanningResult(BaseModel):
    result_id: str
    proposed_itinerary_version_id: str | None
    utility_score: float
    relaxation_steps: list[RelaxationStep]
    escalated: bool
    escalation_reason: str | None
    completed_at: datetime
```

### Write path

1. Orchestrator sets `active_disruption_id` → routes to Replanning Agent
2. Replanning Agent writes `replanning_context` at entry (Step 1)
3. On success: appends `ItineraryVersion` to `itinerary_history`, clears `replanning_context`, clears `active_disruption_id`, marks disruption processed
4. On escalation: writes `ReplanningResult(escalated=True)`, clears `active_disruption_id` — Orchestrator surfaces to user

All writes go through the existing checkpointer transactional API (SQLite dev / Postgres prod). No direct state mutation outside the graph node.

---

## 6. Tests

| File | Coverage |
|---|---|
| `tests/positive/test_replanning_with_fallbacks.py` | Happy path: disruption → fallback scored → new itinerary version emitted |
| `tests/positive/test_replanning_rag_fallback.py` | No pre-staged fallbacks → RAG query fires → plan emitted |
| `tests/negative/test_replanning_relaxation_ladder.py` | All fallbacks fail → ladder walks steps 1–4 → proposal emitted |
| `tests/edge/test_replanning_escalation.py` | Ladder exhausted → escalation to user |
| `tests/edge/test_replanning_no_fallbacks.py` | Pre-Phase-4 itinerary (no `fallback_options`) → RAG fallback path |
| `tests/positive/test_budget_ledger_fx.py` | Wise FX fetch, cache hit/miss, stale rate flag, `score_candidate()` |
| `tests/positive/test_budget_ledger_sunk_cost.py` | Sunk costs excluded from `remaining_budget()` |
| `tests/positive/test_fallback_staging.py` | Itinerary Builder stages 3 fallbacks per bookable stop, constraint-filtered |
| `tests/negative/test_fallback_staging_constraint_filter.py` | Fallbacks failing hard constraints are never staged |

---

## 7. Key Invariants

- **No silent relaxation** — any plan with `constraint_score < 1.0` is a proposal, never auto-applied
- **Sunk cost isolation** — sunk costs never count against replanning budget; `remaining_budget()` is the authoritative source
- **Version binding** — every replanning cycle records the `(profile_version_id, itinerary_version_id)` it started with
- **Constraint guarantee on fallbacks** — every staged `FallbackOption` passes the same hard constraints as the primary stop
- **FX staleness transparency** — stale FX rates are flagged in `ReplanningResult`; no silent degradation
- **Disruption dedup** — `active_disruption_id` lifecycle ensures one active replanning cycle at a time; new disruptions queue behind
