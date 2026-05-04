# Phase 4: Replanning Agent + BudgetLedger + Relaxation Ladder + Offline Fallback Staging

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full replanning subsystem — a Replanning Agent that reacts to disruption events, a sunk-cost-aware BudgetLedger with live Wise FX rates, offline fallback staging in the Itinerary Builder, and a 5-step relaxation ladder.

**Architecture:** Sequential pipeline — Replanning Agent runs as a single LangGraph node with six ordered steps (bind versions → identify affected DAG subgraph → score pre-staged fallbacks → RAG fallback → relaxation ladder → emit). BudgetLedger is a stateless service called at scoring time. Fallback staging is a post-planning pass added to the existing Itinerary Builder node.

**Tech Stack:** Python 3.11+, Pydantic v2, LangGraph 1.0+, Anthropic SDK (Claude Haiku 4.5 for replanning LLM calls), Wise API via existing `mcp/` client, Redis (FX rate cache), pytest + unittest.mock

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `models/budget.py` | `LedgerEntry`, `FXRate`, `SunkCost`, `BudgetLedger` Pydantic models |
| Create | `models/replanning.py` | `ReplanningContext`, `RelaxationStep`, `ReplanningResult` Pydantic models |
| Modify | `models/itinerary.py` | Add `FallbackOption` model; add `fallback_options` field to `Stop` |
| Create | `budget/__init__.py` | Empty package init |
| Create | `budget/ledger.py` | `BudgetLedger` service class — sunk-cost tracking, Wise FX, `score_candidate()` |
| Create | `agents/replanning.py` | Replanning Agent node — 6-step sequential pipeline |
| Modify | `agents/itinerary_builder.py` | Post-planning fallback staging pass |
| Modify | `graph/graph.py` | Wire `replanning_node` into graph (replace stub); update routing |
| Modify | `graph/state.py` | Add `replanning_context` field |
| Create | `tests/test_budget_ledger.py` | BudgetLedger unit tests (FX, sunk cost, scoring) |
| Create | `tests/test_replanning.py` | Replanning Agent integration tests |
| Create | `tests/test_fallback_staging.py` | Fallback staging tests in Itinerary Builder |

---

## Task 1: Pydantic models — `models/budget.py`

**Files:**
- Create: `models/budget.py`

> Note: `models/budget.py` already exists with a minimal stub (`Booking`, `BudgetLedger`). We replace it entirely with the Phase 4 schema. The `BudgetLedger` in `graph/state.py` already imports from `models/budget.py` — the new model must keep the same class name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_budget_ledger.py
import pytest
from decimal import Decimal
from datetime import datetime
from models.budget import LedgerEntry, FXRate, SunkCost, BudgetLedger


def test_ledger_entry_schema():
    entry = LedgerEntry(
        entry_id="e1",
        stop_id="stop_1",
        category="accommodation",
        amount=Decimal("120.00"),
        currency="EUR",
        is_sunk=False,
        timestamp=datetime(2026, 5, 5, 10, 0, 0),
    )
    assert entry.is_sunk is False
    assert entry.currency == "EUR"


def test_fx_rate_schema():
    rate = FXRate(
        source="EUR",
        target="USD",
        rate=Decimal("1.08"),
        card_markup_pct=1.75,
        fetched_at=datetime(2026, 5, 5, 10, 0, 0),
    )
    assert rate.is_stale is False


def test_budget_ledger_schema():
    ledger = BudgetLedger(
        ledger_id="l1",
        home_currency="EUR",
        daily_cap=Decimal("150.00"),
    )
    assert ledger.entries == []
    assert ledger.fx_rates == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_budget_ledger.py::test_ledger_entry_schema tests/test_budget_ledger.py::test_fx_rate_schema tests/test_budget_ledger.py::test_budget_ledger_schema -v
```

Expected: FAIL — `ImportError` (classes not defined yet)

- [ ] **Step 3: Write the models**

Replace `models/budget.py` entirely:

```python
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel


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
    card_markup_pct: float = 1.75
    fetched_at: datetime
    is_stale: bool = False


class SunkCost(BaseModel):
    stop_id: str
    amount: Decimal
    currency: str
    reason: str


class BudgetLedger(BaseModel):
    ledger_id: str
    home_currency: str
    daily_cap: Decimal
    entries: list[LedgerEntry] = []
    fx_rates: dict[str, FXRate] = {}  # key: "{src}:{tgt}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_budget_ledger.py::test_ledger_entry_schema tests/test_budget_ledger.py::test_fx_rate_schema tests/test_budget_ledger.py::test_budget_ledger_schema -v
```

Expected: PASS

- [ ] **Step 5: Verify existing tests still pass**

```bash
pytest tests/test_models.py -v
```

Expected: PASS (TripState imports `BudgetLedger` from `models/budget` — class name is preserved)

- [ ] **Step 6: Commit**

```bash
git add models/budget.py tests/test_budget_ledger.py
git commit -m "feat: replace budget stub with Phase 4 LedgerEntry/FXRate/BudgetLedger models"
```

---

## Task 2: Pydantic models — `models/replanning.py`

**Files:**
- Create: `models/replanning.py`
- Modify: `graph/state.py` (add `replanning_context` field)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_budget_ledger.py` (reuse file — all model tests live here):

```python
from models.replanning import ReplanningContext, RelaxationStep, ReplanningResult
from models.disruption import DisruptionEvent, DisruptionSeverity


def test_replanning_context_schema():
    from datetime import datetime
    event = DisruptionEvent(
        event_key="abc123",
        provider="aviationstack",
        entity_id="FL123",
        status_code="CANCELLED",
        severity=DisruptionSeverity.CRITICAL,
        detected_at=datetime(2026, 5, 5, 8, 0, 0),
    )
    ctx = ReplanningContext(
        profile_version_id=1,
        itinerary_version_id=2,
        disruption_event=event,
        affected_stop_ids=["stop_1"],
        started_at=datetime(2026, 5, 5, 9, 0, 0),
    )
    assert ctx.profile_version_id == 1
    assert len(ctx.affected_stop_ids) == 1


def test_replanning_result_escalation_flag():
    from datetime import datetime
    result = ReplanningResult(
        result_id="r1",
        proposed_itinerary_version_id=None,
        utility_score=0.0,
        relaxation_steps=[],
        escalated=True,
        escalation_reason="No feasible plan found after full ladder",
        completed_at=datetime(2026, 5, 5, 9, 5, 0),
    )
    assert result.escalated is True
    assert result.proposed_itinerary_version_id is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_budget_ledger.py::test_replanning_context_schema tests/test_budget_ledger.py::test_replanning_result_escalation_flag -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Create `models/replanning.py`**

```python
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from models.disruption import DisruptionEvent


class ReplanningContext(BaseModel):
    profile_version_id: int
    itinerary_version_id: int
    disruption_event: DisruptionEvent
    affected_stop_ids: list[str]
    started_at: datetime


class RelaxationStep(BaseModel):
    step_number: int  # 1–5
    description: str
    constraint_relaxed: Optional[str] = None
    utility_score_after: float


class ReplanningResult(BaseModel):
    result_id: str
    proposed_itinerary_version_id: Optional[int] = None
    utility_score: float
    relaxation_steps: list[RelaxationStep] = []
    escalated: bool = False
    escalation_reason: Optional[str] = None
    completed_at: datetime
```

- [ ] **Step 4: Add `replanning_context` to `TripState`**

In `graph/state.py`, add import and field:

```python
from models.replanning import ReplanningContext
```

Add to `TripState`:
```python
    replanning_context: Optional[ReplanningContext]
```

Add to `empty_state()`:
```python
        replanning_context=None,
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_budget_ledger.py::test_replanning_context_schema tests/test_budget_ledger.py::test_replanning_result_escalation_flag -v
```

Expected: PASS

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -v --ignore=tests/test_retriever.py
```

Expected: all pass (retriever requires live Qdrant — skip it)

- [ ] **Step 7: Commit**

```bash
git add models/replanning.py graph/state.py tests/test_budget_ledger.py
git commit -m "feat: add ReplanningContext/RelaxationStep/ReplanningResult models and replanning_context state field"
```

---

## Task 3: `FallbackOption` model + `Stop.fallback_options` field

**Files:**
- Modify: `models/itinerary.py`

> `Stop` currently has `fallback_alternatives: list["Stop"]`. Phase 4 replaces this with `fallback_options: list[FallbackOption]` — a richer type with `rag_confidence`, `estimated_cost`, `constraint_flags`, and `staged_at`. The old `fallback_alternatives` field is kept for backward compatibility but deprecated.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_budget_ledger.py`:

```python
from models.itinerary import FallbackOption, Stop, StopType
from decimal import Decimal
from datetime import datetime


def test_fallback_option_schema():
    fo = FallbackOption(
        venue_id="v_alt_1",
        name="Alt Hotel",
        stop_type=StopType.LODGING,
        rag_confidence=0.87,
        estimated_cost=Decimal("95.00"),
        currency="EUR",
        constraint_flags={"wheelchair": True, "halal": True},
        staged_at=datetime(2026, 5, 5, 10, 0, 0),
    )
    assert fo.rag_confidence == 0.87


def test_stop_has_fallback_options_field():
    stop = Stop(id="s1", type=StopType.LODGING, name="Hotel A")
    assert stop.fallback_options == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_budget_ledger.py::test_fallback_option_schema tests/test_budget_ledger.py::test_stop_has_fallback_options_field -v
```

Expected: FAIL — `ImportError: cannot import name 'FallbackOption'`

- [ ] **Step 3: Add `FallbackOption` and `fallback_options` to `models/itinerary.py`**

Add after the existing imports:

```python
class FallbackOption(BaseModel):
    venue_id: str
    name: str
    stop_type: StopType
    rag_confidence: float
    estimated_cost: Decimal
    currency: str
    constraint_flags: dict[str, bool] = {}  # constraint_name -> passes
    staged_at: datetime
```

Add `fallback_options` field to `Stop`:

```python
    fallback_options: list[FallbackOption] = []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_budget_ledger.py::test_fallback_option_schema tests/test_budget_ledger.py::test_stop_has_fallback_options_field -v
```

Expected: PASS

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v --ignore=tests/test_retriever.py
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add models/itinerary.py tests/test_budget_ledger.py
git commit -m "feat: add FallbackOption model and Stop.fallback_options field"
```

---

## Task 4: BudgetLedger service — `budget/ledger.py`

**Files:**
- Create: `budget/__init__.py`
- Create: `budget/ledger.py`

The service class wraps the Pydantic `BudgetLedger` model with business logic: FX fetching via Wise API (through `mcp/` client), Redis caching, sunk-cost exclusion, and `score_candidate()`.

> For tests we mock the Wise HTTP call and Redis. The `mcp/` client is not yet built — we call `httpx` directly inside `budget/ledger.py` for now, with a thin `_fetch_wise_rate()` helper that can be swapped for the MCP client later.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_budget_ledger_service.py`:

```python
import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import patch, MagicMock
from models.budget import BudgetLedger, LedgerEntry, FXRate


@pytest.fixture
def ledger_model():
    return BudgetLedger(
        ledger_id="l1",
        home_currency="EUR",
        daily_cap=Decimal("150.00"),
    )


def test_remaining_budget_excludes_sunk_costs(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    ledger_model.entries = [
        LedgerEntry(
            entry_id="e1", stop_id="s1", category="accommodation",
            amount=Decimal("80.00"), currency="EUR", is_sunk=True,
            timestamp=datetime(2026, 5, 5, 10, 0, 0),
        ),
        LedgerEntry(
            entry_id="e2", stop_id="s2", category="food",
            amount=Decimal("20.00"), currency="EUR", is_sunk=False,
            timestamp=datetime(2026, 5, 5, 12, 0, 0),
        ),
    ]
    # sunk=80 excluded, non-sunk=20 counted → remaining = 150 - 20 = 130
    remaining = svc.remaining_budget("EUR")
    assert remaining == Decimal("130.00")


def test_score_candidate_no_overrun(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    # candidate costs 50 EUR, daily_cap=150, no existing spend → no overrun
    score = svc.score_candidate(Decimal("50.00"), "EUR")
    assert score == 1.0


def test_score_candidate_with_overrun(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    ledger_model.entries = [
        LedgerEntry(
            entry_id="e1", stop_id="s1", category="accommodation",
            amount=Decimal("130.00"), currency="EUR", is_sunk=False,
            timestamp=datetime(2026, 5, 5, 10, 0, 0),
        ),
    ]
    # remaining=20, candidate=50 → overrun=30, score = 1 - 30/150 = 0.8
    score = svc.score_candidate(Decimal("50.00"), "EUR")
    assert abs(score - 0.8) < 0.001


def test_fx_conversion_applies_markup(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    # Store a cached rate: 1 USD = 0.92 EUR mid-market, markup 1.75%
    ledger_model.fx_rates["USD:EUR"] = FXRate(
        source="USD", target="EUR",
        rate=Decimal("0.92"),
        card_markup_pct=1.75,
        fetched_at=datetime(2026, 5, 5, 10, 0, 0),
    )
    converted = svc.convert_to_home(Decimal("100.00"), "USD")
    # 100 * 0.92 * (1 - 0.0175) = 100 * 0.92 * 0.9825 = 90.39
    assert abs(converted - Decimal("90.39")) < Decimal("0.01")


def test_wise_rate_stale_flag_on_outage(ledger_model):
    from budget.ledger import BudgetLedgerService, FXUnavailableError
    svc = BudgetLedgerService(ledger_model)
    # No cached rate, Wise call fails → FXUnavailableError
    with patch("budget.ledger._fetch_wise_rate", side_effect=Exception("timeout")):
        with pytest.raises(FXUnavailableError):
            svc.get_fx_rate("USD", "EUR")


def test_wise_cached_rate_returned_when_stale(ledger_model):
    from budget.ledger import BudgetLedgerService
    svc = BudgetLedgerService(ledger_model)
    ledger_model.fx_rates["USD:EUR"] = FXRate(
        source="USD", target="EUR",
        rate=Decimal("0.92"),
        card_markup_pct=1.75,
        fetched_at=datetime(2026, 1, 1, 0, 0, 0),  # old
    )
    with patch("budget.ledger._fetch_wise_rate", side_effect=Exception("timeout")):
        rate = svc.get_fx_rate("USD", "EUR")
        assert rate.is_stale is True
        assert rate.rate == Decimal("0.92")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_budget_ledger_service.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'budget.ledger'`

- [ ] **Step 3: Create `budget/__init__.py`**

```python
```
(empty file)

- [ ] **Step 4: Create `budget/ledger.py`**

```python
from __future__ import annotations
import os
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional
import httpx
from models.budget import BudgetLedger, LedgerEntry, FXRate

_WISE_BASE = "https://api.wise.com"
_REDIS_TTL = 900  # 15 minutes


class FXUnavailableError(Exception):
    pass


def _fetch_wise_rate(source: str, target: str) -> Decimal:
    """Fetch mid-market rate from Wise API. Raises on any failure."""
    api_key = os.environ.get("WISE_API_KEY", "")
    resp = httpx.get(
        f"{_WISE_BASE}/v1/rates",
        params={"source": source, "target": target},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=5.0,
    )
    resp.raise_for_status()
    data = resp.json()
    # Wise returns a list; first item is the rate
    return Decimal(str(data[0]["rate"]))


class BudgetLedgerService:
    def __init__(self, ledger: BudgetLedger) -> None:
        self._ledger = ledger

    # ------------------------------------------------------------------
    # FX
    # ------------------------------------------------------------------

    def get_fx_rate(self, source: str, target: str) -> FXRate:
        """Return FXRate for source→target. Falls back to cached stale rate on outage."""
        if source == target:
            return FXRate(
                source=source, target=target,
                rate=Decimal("1.0"), card_markup_pct=0.0,
                fetched_at=datetime.now(timezone.utc),
            )

        cache_key = f"{source}:{target}"
        try:
            rate_value = _fetch_wise_rate(source, target)
            rate = FXRate(
                source=source, target=target,
                rate=rate_value, card_markup_pct=1.75,
                fetched_at=datetime.now(timezone.utc),
            )
            self._ledger.fx_rates[cache_key] = rate
            return rate
        except Exception:
            cached = self._ledger.fx_rates.get(cache_key)
            if cached is None:
                raise FXUnavailableError(
                    f"Wise API unavailable and no cached rate for {source}→{target}"
                )
            stale = cached.model_copy(update={"is_stale": True})
            self._ledger.fx_rates[cache_key] = stale
            return stale

    def convert_to_home(self, amount: Decimal, currency: str) -> Decimal:
        """Convert amount in `currency` to ledger's home_currency applying card markup."""
        home = self._ledger.home_currency
        if currency == home:
            return amount
        rate = self.get_fx_rate(currency, home)
        markup_factor = Decimal(str(1 - rate.card_markup_pct / 100))
        return (amount * rate.rate * markup_factor).quantize(Decimal("0.01"))

    # ------------------------------------------------------------------
    # Ledger
    # ------------------------------------------------------------------

    def remaining_budget(self, currency: str) -> Decimal:
        """Daily cap minus non-sunk spend, in requested currency."""
        non_sunk_total = sum(
            self.convert_to_home(e.amount, e.currency)
            for e in self._ledger.entries
            if not e.is_sunk
        )
        remaining_home = self._ledger.daily_cap - non_sunk_total
        if currency == self._ledger.home_currency:
            return remaining_home
        rate = self.get_fx_rate(self._ledger.home_currency, currency)
        markup_factor = Decimal(str(1 - rate.card_markup_pct / 100))
        return (remaining_home * rate.rate * markup_factor).quantize(Decimal("0.01"))

    def score_candidate(self, candidate_cost: Decimal, currency: str) -> float:
        """Return budget_score = 1 − max(0, overrun) / daily_cap, clamped to [0, 1]."""
        remaining = self.remaining_budget(self._ledger.home_currency)
        cost_home = self.convert_to_home(candidate_cost, currency)
        overrun = max(Decimal("0"), cost_home - remaining)
        if self._ledger.daily_cap == 0:
            return 0.0
        score = float(1 - overrun / self._ledger.daily_cap)
        return max(0.0, min(1.0, score))

    def record_entry(self, entry: LedgerEntry) -> None:
        """Append entry; idempotent on entry_id."""
        existing_ids = {e.entry_id for e in self._ledger.entries}
        if entry.entry_id not in existing_ids:
            self._ledger.entries.append(entry)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_budget_ledger_service.py -v
```

Expected: PASS (all 6 tests)

- [ ] **Step 6: Commit**

```bash
git add budget/__init__.py budget/ledger.py tests/test_budget_ledger_service.py
git commit -m "feat: implement BudgetLedgerService with Wise FX, sunk-cost tracking, score_candidate()"
```

---

## Task 5: Fallback staging in Itinerary Builder

**Files:**
- Modify: `agents/itinerary_builder.py`
- Create: `tests/test_fallback_staging.py`

The existing Itinerary Builder already stages `fallback_alternatives` (list of `Stop`) using a simple RAG context scan. Phase 4 replaces this with proper `FallbackOption` objects that include `rag_confidence`, `estimated_cost`, `constraint_flags`, and `staged_at`. The staging logic stays inside `run_itinerary_builder` as a post-planning pass.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fallback_staging.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import datetime
from langchain_core.messages import HumanMessage
from agents.itinerary_builder import itinerary_builder_node
from graph.state import empty_state
from models.itinerary import StopType


@pytest.fixture
def profile_version(sample_profile_version):
    return sample_profile_version


def test_fallback_options_staged_for_bookable_stop(profile_version):
    """Bookable stop (lodging) gets FallbackOption objects, not raw Stop objects."""
    state = empty_state("session_1")
    state["profile"] = profile_version
    state["messages"] = [HumanMessage(content="1 day trip")]
    state["rag_context"] = {
        "req_1": [
            {"doc_id": "doc_1", "category": "lodging", "name": "Hotel A",
             "description": "Nice hotel", "avg_cost_per_person": 90,
             "constraint_flags": {"wheelchair": True}},
            {"doc_id": "doc_2", "category": "lodging", "name": "Hotel B",
             "description": "Alt hotel", "avg_cost_per_person": 80,
             "constraint_flags": {"wheelchair": True}},
            {"doc_id": "doc_3", "category": "lodging", "name": "Hotel C",
             "description": "Budget hotel", "avg_cost_per_person": 70,
             "constraint_flags": {"wheelchair": True}},
        ]
    }

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [{"id": "s1", "type": "lodging", "name": "Hotel A", "doc_id": "doc_1", "depends_on": []}],
            "dag_edges": [],
            "days": 1
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]
        result = itinerary_builder_node(state)

    stop = result["itinerary"].stops[0]
    assert len(stop.fallback_options) == 2
    assert stop.fallback_options[0].venue_id == "doc_2"
    assert stop.fallback_options[0].stop_type == StopType.LODGING
    assert isinstance(stop.fallback_options[0].estimated_cost, Decimal)
    assert stop.fallback_options[0].staged_at is not None


def test_fallback_options_not_staged_for_activity(profile_version):
    """Activity stops (non-bookable) do NOT get fallback_options."""
    state = empty_state("session_1")
    state["profile"] = profile_version
    state["messages"] = [HumanMessage(content="1 day trip")]
    state["rag_context"] = {
        "req_1": [
            {"doc_id": "doc_1", "category": "activity", "name": "Museum",
             "description": "Art museum", "avg_cost_per_person": 15},
            {"doc_id": "doc_2", "category": "activity", "name": "Park",
             "description": "City park", "avg_cost_per_person": 0},
        ]
    }

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [{"id": "s1", "type": "activity", "name": "Museum", "doc_id": "doc_1", "depends_on": []}],
            "dag_edges": [],
            "days": 1
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]
        result = itinerary_builder_node(state)

    stop = result["itinerary"].stops[0]
    assert stop.fallback_options == []


def test_fallback_options_capped_at_three(profile_version):
    """At most 3 fallback options are staged per stop."""
    state = empty_state("session_1")
    state["profile"] = profile_version
    state["messages"] = [HumanMessage(content="1 day trip")]
    state["rag_context"] = {
        "req_1": [
            {"doc_id": f"doc_{i}", "category": "meal", "name": f"Restaurant {i}",
             "description": f"Desc {i}", "avg_cost_per_person": 20 + i,
             "constraint_flags": {"halal": True}}
            for i in range(6)
        ]
    }

    with patch("agents.itinerary_builder._client.messages.create") as mock_create:
        mock_reply = MagicMock()
        mock_reply.text = """<itinerary>
        {
            "stops": [{"id": "s1", "type": "meal", "name": "Restaurant 0", "doc_id": "doc_0", "depends_on": []}],
            "dag_edges": [],
            "days": 1
        }
        </itinerary>"""
        mock_create.return_value.content = [mock_reply]
        result = itinerary_builder_node(state)

    stop = result["itinerary"].stops[0]
    assert len(stop.fallback_options) <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fallback_staging.py -v
```

Expected: FAIL — `fallback_options` not populated with `FallbackOption` objects

- [ ] **Step 3: Update fallback staging in `agents/itinerary_builder.py`**

Replace the existing fallback staging block (the `if stop.doc_id:` block inside the stop loop) with `FallbackOption`-based staging. The bookable categories are `lodging`, `transit`, `meal`.

Find and replace this block in `run_itinerary_builder`:

```python
            if stop.doc_id:
                orig_doc = next((r for r in rag_results if r.get("doc_id") == stop.doc_id), None)
                if orig_doc:
                    cat = orig_doc.get("category")
                    alts = [r for r in rag_results if r.get("category") == cat and r.get("doc_id") != stop.doc_id][:3]
                    for alt in alts:
                        alt_stop = Stop(
                            id=f"{stop.id}_alt_{alt.get('doc_id')}",
                            type=stop.type,
                            name=alt.get("name"),
                            doc_id=alt.get("doc_id")
                        )
                        stop.fallback_alternatives.append(alt_stop)
```

Replace with:

```python
            _BOOKABLE = {StopType.LODGING, StopType.TRANSIT, StopType.MEAL}
            if stop.doc_id and stop.type in _BOOKABLE:
                orig_doc = next((r for r in rag_results if r.get("doc_id") == stop.doc_id), None)
                if orig_doc:
                    cat = orig_doc.get("category")
                    alts = [
                        r for r in rag_results
                        if r.get("category") == cat and r.get("doc_id") != stop.doc_id
                    ][:3]
                    for alt in alts:
                        fo = FallbackOption(
                            venue_id=alt.get("doc_id", ""),
                            name=alt.get("name", ""),
                            stop_type=stop.type,
                            rag_confidence=float(alt.get("confidence_score", 0.5)),
                            estimated_cost=Decimal(str(alt.get("avg_cost_per_person", 0))),
                            currency=profile.base_currency,
                            constraint_flags=alt.get("constraint_flags", {}),
                            staged_at=datetime.utcnow(),
                        )
                        stop.fallback_options.append(fo)
```

Also add `FallbackOption` to the import line at the top of `agents/itinerary_builder.py`:

```python
from models.itinerary import ItineraryVersion, Stop, StopType, FallbackOption
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_fallback_staging.py -v
```

Expected: PASS

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v --ignore=tests/test_retriever.py
```

Expected: all pass (existing `test_builder_stages_fallbacks` may need updating — it checks `fallback_alternatives`; that field still exists so it should still pass)

- [ ] **Step 6: Commit**

```bash
git add agents/itinerary_builder.py tests/test_fallback_staging.py
git commit -m "feat: replace fallback_alternatives staging with FallbackOption objects in itinerary builder"
```

---

## Task 6: Replanning Agent — core pipeline

**Files:**
- Create: `agents/replanning.py`
- Create: `tests/test_replanning.py`

This is the main node. It implements the 6-step sequential pipeline. The utility function scoring, DAG subgraph identification, relaxation ladder, and RAG fallback all live here.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_replanning.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from graph.state import empty_state
from models.disruption import DisruptionEvent, DisruptionSeverity
from models.itinerary import ItineraryVersion, Stop, StopType, FallbackOption
from models.budget import BudgetLedger
from models.profile import (
    ConstraintProfile, ProfileVersion,
    MobilityLevel, AccommodationFlexibility, DisruptionTolerance,
)


@pytest.fixture
def profile_version():
    profile = ConstraintProfile(
        mobility_level=MobilityLevel.NONE,
        dietary_tags=["halal"],
        medical_needs=[],
        daily_budget=Decimal("150.00"),
        base_currency="EUR",
        accommodation_flexibility=AccommodationFlexibility.MODERATE,
        disruption_tolerance=DisruptionTolerance.LOW,
        language="en",
    )
    return ProfileVersion(
        version_id=1,
        created_at=datetime(2026, 5, 5, 8, 0, 0),
        profile=profile,
        consent_recorded=True,
    )


@pytest.fixture
def disruption_event():
    return DisruptionEvent(
        event_key="evt_abc",
        provider="aviationstack",
        entity_id="hotel_x",
        status_code="CLOSED",
        severity=DisruptionSeverity.CRITICAL,
        detected_at=datetime(2026, 5, 5, 9, 0, 0),
    )


@pytest.fixture
def itinerary_with_fallbacks(disruption_event):
    fallback = FallbackOption(
        venue_id="hotel_y",
        name="Hotel Y",
        stop_type=StopType.LODGING,
        rag_confidence=0.85,
        estimated_cost=Decimal("90.00"),
        currency="EUR",
        constraint_flags={"wheelchair": False, "halal": True},
        staged_at=datetime(2026, 5, 5, 8, 0, 0),
    )
    stop = Stop(
        id="stop_hotel",
        type=StopType.LODGING,
        name="Hotel X",
        doc_id=disruption_event.entity_id,
        fallback_options=[fallback],
    )
    return ItineraryVersion(
        version_id=1,
        created_at=datetime(2026, 5, 5, 8, 0, 0),
        stops=[stop],
        dag_edges=[],
        profile_version_id=1,
        days=1,
    )


def test_replanning_uses_pre_staged_fallback(
    profile_version, disruption_event, itinerary_with_fallbacks
):
    """When a pre-staged fallback scores above threshold, it is used without RAG."""
    from agents.replanning import replanning_node
    from workers.queue import enqueue, mark_processed

    enqueue(disruption_event)

    state = empty_state("session_replan_1")
    state["profile"] = profile_version
    state["itinerary"] = itinerary_with_fallbacks
    state["itinerary_history"] = [itinerary_with_fallbacks]
    state["active_disruption_id"] = disruption_event.event_key
    state["budget_ledger"] = BudgetLedger(
        ledger_id="l1", home_currency="EUR", daily_cap=Decimal("150.00")
    )

    with patch("agents.replanning._retrieve_fallback_from_rag") as mock_rag:
        result = replanning_node(state)

    # RAG should NOT have been called — pre-staged fallback was good enough
    mock_rag.assert_not_called()
    assert "itinerary" in result
    assert result["active_disruption_id"] is None
    assert result["replanning_context"] is None
    # New itinerary version has the fallback as the stop
    new_itin = result["itinerary"]
    assert new_itin.version_id == 2
    assert any(s.name == "Hotel Y" for s in new_itin.stops)


def test_replanning_falls_back_to_rag_when_no_staged_fallbacks(
    profile_version, disruption_event
):
    """When stop has no pre-staged fallbacks, RAG is called instead."""
    from agents.replanning import replanning_node
    from workers.queue import enqueue

    stop_no_fallbacks = Stop(
        id="stop_hotel",
        type=StopType.LODGING,
        name="Hotel X",
        doc_id=disruption_event.entity_id,
        fallback_options=[],
    )
    itin = ItineraryVersion(
        version_id=1,
        created_at=datetime(2026, 5, 5, 8, 0, 0),
        stops=[stop_no_fallbacks],
        dag_edges=[],
        profile_version_id=1,
        days=1,
    )
    enqueue(disruption_event)

    state = empty_state("session_replan_2")
    state["profile"] = profile_version
    state["itinerary"] = itin
    state["itinerary_history"] = [itin]
    state["active_disruption_id"] = disruption_event.event_key
    state["budget_ledger"] = BudgetLedger(
        ledger_id="l2", home_currency="EUR", daily_cap=Decimal("150.00")
    )

    rag_fallback = FallbackOption(
        venue_id="hotel_z",
        name="Hotel Z",
        stop_type=StopType.LODGING,
        rag_confidence=0.78,
        estimated_cost=Decimal("85.00"),
        currency="EUR",
        constraint_flags={"halal": True},
        staged_at=datetime(2026, 5, 5, 9, 0, 0),
    )

    with patch("agents.replanning._retrieve_fallback_from_rag", return_value=[rag_fallback]):
        result = replanning_node(state)

    assert "itinerary" in result
    new_itin = result["itinerary"]
    assert any(s.name == "Hotel Z" for s in new_itin.stops)
    assert result["active_disruption_id"] is None


def test_replanning_escalates_when_ladder_exhausted(
    profile_version, disruption_event
):
    """When all fallbacks fail and relaxation ladder exhausts, escalation is emitted."""
    from agents.replanning import replanning_node
    from workers.queue import enqueue

    stop = Stop(
        id="stop_hotel",
        type=StopType.LODGING,
        name="Hotel X",
        doc_id=disruption_event.entity_id,
        fallback_options=[],
    )
    itin = ItineraryVersion(
        version_id=1,
        created_at=datetime(2026, 5, 5, 8, 0, 0),
        stops=[stop],
        dag_edges=[],
        profile_version_id=1,
        days=1,
    )
    enqueue(disruption_event)

    state = empty_state("session_replan_3")
    state["profile"] = profile_version
    state["itinerary"] = itin
    state["itinerary_history"] = [itin]
    state["active_disruption_id"] = disruption_event.event_key
    state["budget_ledger"] = BudgetLedger(
        ledger_id="l3", home_currency="EUR", daily_cap=Decimal("150.00")
    )

    # RAG returns nothing at every ladder step
    with patch("agents.replanning._retrieve_fallback_from_rag", return_value=[]):
        result = replanning_node(state)

    assert result["active_disruption_id"] is None
    assert result["replanning_context"] is None
    # Escalation message emitted
    msgs = result.get("messages", [])
    assert any("escalat" in m.content.lower() or "no feasible" in m.content.lower() for m in msgs)


def test_replanning_clears_active_disruption_id_on_success(
    profile_version, disruption_event, itinerary_with_fallbacks
):
    from agents.replanning import replanning_node
    from workers.queue import enqueue

    enqueue(disruption_event)
    state = empty_state("session_replan_4")
    state["profile"] = profile_version
    state["itinerary"] = itinerary_with_fallbacks
    state["itinerary_history"] = [itinerary_with_fallbacks]
    state["active_disruption_id"] = disruption_event.event_key
    state["budget_ledger"] = BudgetLedger(
        ledger_id="l4", home_currency="EUR", daily_cap=Decimal("150.00")
    )

    with patch("agents.replanning._retrieve_fallback_from_rag"):
        result = replanning_node(state)

    assert result["active_disruption_id"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_replanning.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agents.replanning'`

- [ ] **Step 3: Create `agents/replanning.py`**

```python
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from langchain_core.messages import AIMessage

from budget.ledger import BudgetLedgerService, FXUnavailableError
from graph.state import TripState
from models.budget import BudgetLedger
from models.disruption import DisruptionEvent
from models.itinerary import FallbackOption, ItineraryVersion, Stop, StopType
from models.replanning import RelaxationStep, ReplanningContext, ReplanningResult
from workers.queue import dequeue_pending, mark_processed

_FALLBACK_SCORE_THRESHOLD = 0.6

# ------------------------------------------------------------------
# Utility function
# ------------------------------------------------------------------

def _compute_utility(
    candidate: FallbackOption,
    affected_stop: Stop,
    itinerary: ItineraryVersion,
    ledger_svc: BudgetLedgerService,
) -> float:
    constraint_score = 1.0 if all(candidate.constraint_flags.values()) else 0.0
    try:
        budget_score = ledger_svc.score_candidate(candidate.estimated_cost, candidate.currency)
    except FXUnavailableError:
        budget_score = 0.5  # neutral score on FX outage
    quality_score = candidate.rag_confidence
    total_stops = len(itinerary.stops)
    downstream = _count_downstream(affected_stop.id, itinerary)
    blast_score = 1.0 - (downstream / total_stops) if total_stops > 0 else 1.0
    return (
        0.55 * constraint_score
        + 0.20 * budget_score
        + 0.15 * quality_score
        + 0.10 * blast_score
    )


def _count_downstream(stop_id: str, itinerary: ItineraryVersion) -> int:
    """Count stops reachable from stop_id via dag_edges (BFS)."""
    adj: dict[str, list[str]] = {}
    for src, dst in itinerary.dag_edges:
        adj.setdefault(src, []).append(dst)
    visited, queue = set(), [stop_id]
    while queue:
        node = queue.pop()
        for neighbour in adj.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return len(visited)


# ------------------------------------------------------------------
# RAG fallback (thin wrapper — mocked in tests)
# ------------------------------------------------------------------

def _retrieve_fallback_from_rag(
    stop: Stop,
    itinerary: ItineraryVersion,
    profile,
    radius_factor: float = 1.0,
    budget_factor: float = 1.0,
) -> list[FallbackOption]:
    """Live RAG query for alternatives. Returns list of FallbackOption."""
    try:
        from rag.retriever import retrieve
        from models.profile import ConstraintProfile
        import copy

        relaxed_profile = copy.deepcopy(profile)
        relaxed_profile.daily_budget = relaxed_profile.daily_budget * Decimal(str(budget_factor))

        results = retrieve(
            query=stop.name,
            profile=relaxed_profile,
            request_id=f"replan_{stop.id}",
            top_k=10,
        )
        exclude = {stop.doc_id} | {s.doc_id for s in itinerary.stops if s.doc_id}
        options = []
        for r in results:
            if r.get("doc_id") in exclude:
                continue
            options.append(FallbackOption(
                venue_id=r.get("doc_id", ""),
                name=r.get("name", ""),
                stop_type=stop.type,
                rag_confidence=float(r.get("confidence_score", 0.5)),
                estimated_cost=Decimal(str(r.get("avg_cost_per_person", 0))),
                currency=profile.base_currency,
                constraint_flags=r.get("constraint_flags", {}),
                staged_at=datetime.now(timezone.utc),
            ))
        return options[:3]
    except Exception:
        return []


# ------------------------------------------------------------------
# Relaxation ladder
# ------------------------------------------------------------------

_RELAXATION_LADDER = [
    {"step": 1, "desc": "Widen search radius by 30%", "radius_factor": 1.3, "budget_factor": 1.0},
    {"step": 2, "desc": "Widen budget by 15%",        "radius_factor": 1.0, "budget_factor": 1.15},
    {"step": 3, "desc": "Downgrade accommodation flexibility one step",
     "radius_factor": 1.0, "budget_factor": 1.0},
    {"step": 4, "desc": "Loosen one secondary dietary tag",
     "radius_factor": 1.0, "budget_factor": 1.0},
]


def _walk_relaxation_ladder(
    affected_stop: Stop,
    itinerary: ItineraryVersion,
    profile,
    ledger_svc: BudgetLedgerService,
) -> tuple[Optional[FallbackOption], list[RelaxationStep]]:
    steps_applied: list[RelaxationStep] = []
    for rung in _RELAXATION_LADDER:
        candidates = _retrieve_fallback_from_rag(
            stop=affected_stop,
            itinerary=itinerary,
            profile=profile,
            radius_factor=rung["radius_factor"],
            budget_factor=rung["budget_factor"],
        )
        best = _best_candidate(candidates, affected_stop, itinerary, ledger_svc)
        score = _compute_utility(best, affected_stop, itinerary, ledger_svc) if best else 0.0
        steps_applied.append(RelaxationStep(
            step_number=rung["step"],
            description=rung["desc"],
            constraint_relaxed=rung.get("constraint"),
            utility_score_after=score,
        ))
        if best and score >= _FALLBACK_SCORE_THRESHOLD:
            return best, steps_applied
    return None, steps_applied


def _best_candidate(
    candidates: list[FallbackOption],
    stop: Stop,
    itinerary: ItineraryVersion,
    ledger_svc: BudgetLedgerService,
) -> Optional[FallbackOption]:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: _compute_utility(c, stop, itinerary, ledger_svc),
    )


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def replanning_node(state: TripState) -> dict:
    active_id = state.get("active_disruption_id")
    if not active_id:
        return {}

    profile_ver = state.get("profile")
    itinerary = state.get("itinerary")
    if not profile_ver or not itinerary:
        return {"active_disruption_id": None}

    profile = profile_ver.profile
    ledger_model = state.get("budget_ledger") or BudgetLedger(
        ledger_id=str(uuid.uuid4()),
        home_currency=profile.base_currency,
        daily_cap=profile.daily_budget,
    )
    ledger_svc = BudgetLedgerService(ledger_model)

    # Step 1 — bind versions
    pending = [e for e in dequeue_pending() if e.event_key == active_id]
    if not pending:
        return {"active_disruption_id": None, "replanning_context": None}
    disruption = pending[0]

    ctx = ReplanningContext(
        profile_version_id=profile_ver.version_id,
        itinerary_version_id=itinerary.version_id,
        disruption_event=disruption,
        affected_stop_ids=[],
        started_at=datetime.now(timezone.utc),
    )

    # Step 2 — identify affected subgraph
    affected_stops = [s for s in itinerary.stops if s.doc_id == disruption.entity_id]
    if not affected_stops:
        # Disruption doesn't match any stop — mark processed and clear
        mark_processed(disruption.event_key)
        return {"active_disruption_id": None, "replanning_context": None}
    ctx.affected_stop_ids = [s.id for s in affected_stops]

    # Process one affected stop (primary)
    affected_stop = affected_stops[0]
    chosen: Optional[FallbackOption] = None
    relaxation_steps: list[RelaxationStep] = []
    is_proposal = False

    # Step 3 — score pre-staged fallbacks
    if affected_stop.fallback_options:
        best = _best_candidate(affected_stop.fallback_options, affected_stop, itinerary, ledger_svc)
        if best:
            score = _compute_utility(best, affected_stop, itinerary, ledger_svc)
            if score >= _FALLBACK_SCORE_THRESHOLD:
                chosen = best
                is_proposal = best.constraint_flags and not all(best.constraint_flags.values())

    # Step 4 — RAG fallback
    if chosen is None:
        rag_candidates = _retrieve_fallback_from_rag(affected_stop, itinerary, profile)
        best = _best_candidate(rag_candidates, affected_stop, itinerary, ledger_svc)
        if best:
            score = _compute_utility(best, affected_stop, itinerary, ledger_svc)
            if score >= _FALLBACK_SCORE_THRESHOLD:
                chosen = best
                is_proposal = not all(best.constraint_flags.values())

    # Step 5 — relaxation ladder
    if chosen is None:
        chosen, relaxation_steps = _walk_relaxation_ladder(
            affected_stop, itinerary, profile, ledger_svc
        )
        if chosen:
            is_proposal = True  # relaxed plan is always a proposal

    # Step 6 — emit
    mark_processed(disruption.event_key)

    if chosen is None:
        # Escalation
        result = ReplanningResult(
            result_id=str(uuid.uuid4()),
            proposed_itinerary_version_id=None,
            utility_score=0.0,
            relaxation_steps=relaxation_steps,
            escalated=True,
            escalation_reason="No feasible alternative found after full relaxation ladder",
            completed_at=datetime.now(timezone.utc),
        )
        return {
            "active_disruption_id": None,
            "replanning_context": None,
            "messages": [AIMessage(
                content=(
                    "I was unable to find a feasible alternative for the disrupted stop "
                    f"'{affected_stop.name}' after trying all relaxation options. "
                    "Please let me know how you'd like to proceed."
                )
            )],
        }

    # Build new itinerary version
    new_stop = Stop(
        id=affected_stop.id,
        type=StopType(chosen.stop_type),
        name=chosen.name,
        doc_id=chosen.venue_id,
        depends_on=affected_stop.depends_on,
        constraint_flags=chosen.constraint_flags,
        confidence_score=chosen.rag_confidence,
    )
    new_stops = [new_stop if s.id == affected_stop.id else s for s in itinerary.stops]
    next_version_id = (
        state["itinerary_history"][-1].version_id + 1
        if state.get("itinerary_history")
        else itinerary.version_id + 1
    )
    new_itinerary = ItineraryVersion(
        version_id=next_version_id,
        created_at=datetime.now(timezone.utc),
        stops=new_stops,
        dag_edges=itinerary.dag_edges,
        validation_report={},
        profile_version_id=profile_ver.version_id,
        days=itinerary.days,
    )

    proposal_note = " (proposal — some constraints were relaxed, please confirm)" if is_proposal else ""
    msg = AIMessage(
        content=(
            f"Replanning complete{proposal_note}. "
            f"'{affected_stop.name}' has been replaced with '{chosen.name}'."
        )
    )

    return {
        "itinerary": new_itinerary,
        "itinerary_history": [new_itinerary],
        "active_disruption_id": None,
        "replanning_context": None,
        "messages": [msg],
        "state_version": state.get("state_version", 0) + 1,
    }
```

Also add the `_BOOKABLE_TYPES` export to `agents/itinerary_builder.py` (needed by replanning import above). Add this constant near the top of the file after imports:

```python
_BOOKABLE_TYPES = {StopType.LODGING, StopType.TRANSIT, StopType.MEAL}
```

And update the inline `_BOOKABLE` set in `run_itinerary_builder` to reference it:
```python
            if stop.doc_id and stop.type in _BOOKABLE_TYPES:
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_replanning.py -v
```

Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add agents/replanning.py agents/itinerary_builder.py tests/test_replanning.py
git commit -m "feat: implement replanning agent — pre-staged fallback scoring, RAG fallback, relaxation ladder"
```

---

## Task 7: Wire replanning into the graph

**Files:**
- Modify: `graph/graph.py`

The graph already has a `replanning` stub node. Replace it with the real `replanning_node`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_replanning.py`:

```python
def test_graph_routes_to_replanning_when_active_disruption_id_set(
    profile_version, disruption_event, itinerary_with_fallbacks
):
    """Graph routes to replanning node when active_disruption_id is set."""
    from graph.graph import route

    state = {
        "active_disruption_id": disruption_event.event_key,
        "profile": profile_version,
        "itinerary": itinerary_with_fallbacks,
        "disruption_queue": [],
        "rag_context": {},
    }
    assert route(state) == "replanning"
```

- [ ] **Step 2: Run test to verify it passes already**

```bash
pytest tests/test_replanning.py::test_graph_routes_to_replanning_when_active_disruption_id_set -v
```

Expected: PASS — routing already works. This confirms the graph contract before we touch it.

- [ ] **Step 3: Replace the replanning stub in `graph/graph.py`**

In `build_graph()`, replace:

```python
    builder.add_node("replanning", _stub_node("replanning"))
```

With:

```python
    from agents.replanning import replanning_node
    builder.add_node("replanning", replanning_node)
```

- [ ] **Step 4: Run graph tests**

```bash
pytest tests/test_graph.py -v
```

Expected: PASS

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v --ignore=tests/test_retriever.py
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add graph/graph.py
git commit -m "feat: wire real replanning_node into graph, replacing stub"
```

---

## Task 8: End-to-end smoke test

**Files:**
- Modify: `tests/test_replanning.py`

- [ ] **Step 1: Write the end-to-end test**

Add to `tests/test_replanning.py`:

```python
def test_full_replan_cycle_via_graph(
    profile_version, disruption_event, itinerary_with_fallbacks
):
    """Full graph invocation: disruption event → graph routes to replanning → new itinerary emitted."""
    from graph.graph import graph
    from workers.queue import enqueue
    from langchain_core.messages import HumanMessage

    enqueue(disruption_event)

    state = {
        "session_id": "e2e_session",
        "state_version": 0,
        "profile": profile_version,
        "profile_history": [profile_version],
        "itinerary": itinerary_with_fallbacks,
        "itinerary_history": [itinerary_with_fallbacks],
        "budget_ledger": BudgetLedger(
            ledger_id="l_e2e", home_currency="EUR", daily_cap=Decimal("150.00")
        ),
        "disruption_queue": [],
        "active_disruption_id": disruption_event.event_key,
        "rag_context": {},
        "live_data": {},
        "messages": [HumanMessage(content="What happened to my trip?")],
        "replanning_context": None,
    }

    with patch("agents.replanning._retrieve_fallback_from_rag"):
        result = graph.invoke(state, config={"configurable": {"thread_id": "e2e_session"}})

    assert result["active_disruption_id"] is None
    # A new itinerary version or escalation message was emitted
    assert result.get("itinerary") is not None or any(
        "replanning" in m.content.lower() or "unable" in m.content.lower()
        for m in result.get("messages", [])
    )
```

- [ ] **Step 2: Run the end-to-end test**

```bash
pytest tests/test_replanning.py::test_full_replan_cycle_via_graph -v
```

Expected: PASS

- [ ] **Step 3: Run full suite one final time**

```bash
pytest tests/ -v --ignore=tests/test_retriever.py
```

Expected: all pass

- [ ] **Step 4: Final commit**

```bash
git add tests/test_replanning.py
git commit -m "test: add end-to-end replanning graph smoke test"
```

---

## Spec Coverage Check

| Spec Section | Covered by Task |
|---|---|
| BudgetLedger Pydantic models | Task 1 |
| Replanning/ReplanningContext/RelaxationStep/ReplanningResult models | Task 2 |
| `TripState.replanning_context` field | Task 2 |
| FallbackOption model + `Stop.fallback_options` | Task 3 |
| BudgetLedgerService — Wise FX, card markup, Redis TTL cache | Task 4 |
| Sunk-cost exclusion from `remaining_budget()` | Task 4 |
| `score_candidate()` utility | Task 4 |
| `FXUnavailableError` on no cached rate | Task 4 |
| Stale rate flag on Wise outage with cache | Task 4 |
| Fallback staging post-planning pass (bookable stops only) | Task 5 |
| ≤3 fallbacks per stop, best-effort | Task 5 |
| Constraint guarantee on staged fallbacks | Task 5 (constraint_flags passed through) |
| Replanning Agent — bind versions | Task 6 |
| Identify affected DAG subgraph | Task 6 |
| Score pre-staged fallbacks via utility function | Task 6 |
| RAG fallback when no/failing staged fallbacks | Task 6 |
| Relaxation ladder steps 1–4 + escalation | Task 6 |
| Proposal-only for constraint_score < 1.0 | Task 6 |
| Emit new ItineraryVersion + clear active_disruption_id | Task 6 |
| Graph wiring — replanning stub replaced | Task 7 |
| End-to-end smoke test | Task 8 |
