# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Adaptive Travel Companion** — a multi-agent system for constraint-aware travel planning and real-time disruption recovery. Target users: wheelchair/mobility-limited travellers, dietary-restriction travellers (halal/kosher/allergy), medical travellers, and hard-budget travellers.

Architecture: Orchestrator-spoke LangGraph `StateGraph` + out-of-band Live Data Worker + constraint-pre-filtered RAG pipeline. All inter-agent communication passes through a single shared, versioned `TripState` — agents never call each other directly.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 1.0+ (StateGraph, conditional edges, Postgres checkpointer) |
| Reasoning LLM | Claude Sonnet 4.6 — constraint extraction, replanning, itinerary composition |
| Fast LLM | Claude Haiku 4.5 — parsing, intent classification, disruption normalisation |
| Vector store | Qdrant 1.10+ (3-node cluster, hybrid search) |
| Embeddings | BAAI/bge-m3 (multilingual dense + learned sparse in one pass) |
| Re-ranker | bge-reranker-v2-m3 cross-encoder |
| Checkpointer | Postgres (prod) / SQLite (dev) |
| Queue / rate limits | Redis 7+ |
| Validation | Pydantic v2 |
| Observability | LangSmith + OpenTelemetry + Prometheus + Grafana |
| Testing | pytest + LangSmith Datasets + Hypothesis + pytest-asyncio |
| Infra | Kubernetes + Helm + Terraform |
| Secrets | HashiCorp Vault / AWS Secrets Manager |
| Language | Python 3.11+ |

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/positive/test_wheelchair_paris.py

# Run tests by category
pytest tests/positive/
pytest tests/negative/
pytest tests/edge/
pytest tests/adversarial/
pytest tests/concurrency/

# Run with LangSmith tracing
LANGSMITH_TRACING=true pytest

# Local dev stack (Qdrant + Postgres + Redis)
docker compose -f deploy/docker-compose.yml up -d

# Ingest RAG corpus
python rag/ingest.py

# Start the Live Data Worker (separate process)
python workers/live_data_worker.py
```

## Repository Layout

```
adaptive-travel-companion/
├── agents/               # One file per agent (see Agent Roles below)
├── workers/              # Live Data Worker (out-of-band process) + webhooks + poll scheduler
├── graph/                # state.py (TripState TypedDict + reducers), graph.py, checkpointer.py
├── rag/                  # corpus/, ingest.py, retriever.py, reranker.py, vocabulary/ (versioned YAML)
├── mcp/                  # MCP server wrappers + client (rate limiting, circuit breakers, idempotency)
├── models/               # Pydantic v2 schemas (ProfileVersion, ItineraryVersion, etc.)
├── budget/               # ledger.py — sunk-cost-aware FX model
├── security/             # pii_redactor.py, encryption.py, retention.py
├── tests/                # positive/, negative/, edge/, adversarial/, concurrency/
├── observability/        # langsmith_config.py, metrics.py, slo.py
└── deploy/               # helm/, docker-compose.yml, terraform/
```

## Architecture: Shared State & Concurrency

`TripState` in `graph/state.py` is a `TypedDict` with `Annotated` reducers that make concurrent writes safe:

- `disruption_queue` — append-with-dedup via `event_key = hash(provider + entity_id + status_code + window)`
- `rag_context` — keyed by `request_id` so parallel agents don't clobber each other; consuming agent removes its own key
- `live_data` — latest-by-timestamp reducer; each entry carries `source`, `fetched_at`, `ttl_seconds`
- `profile_history` / `itinerary_history` — append-only audit trail
- `state_version` — optimistic concurrency for HTTP layer (409 on stale write)

The `ConstraintProfile` is **append-only versioned**. Every confirmed change produces a new `ProfileVersion` with a monotonic `version_id`. In-flight replans bind to the `profile_version_id` they started with.

## Architecture: Routing Logic

The Orchestrator is **deterministic Python** (not an LLM). It evaluates a priority-ordered rule table:

| Priority | Condition | Routes to |
|---|---|---|
| 1 | `active_disruption_id is None` and `disruption_queue` non-empty | Replanning (sets `active_disruption_id`) |
| 2 | `active_disruption_id` is set | Replanning (resume) |
| 3 | User requests profile update | Constraint Profiler |
| 4 | `profile is None` | Constraint Profiler |
| 5 | Profile present, `itinerary is None` | Destination Research → Itinerary Builder |
| 6 | User intent ambiguous | Haiku classifier → re-evaluate |
| 7 | Else | Conversational reply |

LLM routing is used **only** for genuinely ambiguous intent (rule 6). All other routing is code.

## Architecture: Agent Responsibilities

- **Orchestrator** (`agents/orchestrator.py`) — routing + `active_disruption_id` lifecycle. No domain work, no external tools.
- **Constraint Profiler** (`agents/constraint_profiler.py`) — structured conversational intake → validated `ProfileVersion`. Uses Sonnet; Pydantic v2 validation; controlled vocabulary for `dietary_tags`.
- **Destination Research** (`agents/destination_research.py`) — hybrid Qdrant search with constraint pre-filter as hard gate, live enrichment via MCP, corpus-vs-live reconciliation rule (live source wins if timestamp within 7 days).
- **Itinerary Builder** (`agents/itinerary_builder.py`) — composes day-by-day plan over a **dependency DAG** (not a list); emits `ItineraryVersion` + per-stop validation flags. Stages 3 offline-fallback alternatives per booked component at planning time.
- **Replanning Agent** (`agents/replanning.py`) — binds `(profile_version_id, itinerary_version_id)` at start; reconstructs only the DAG subgraph affected by the disruption; scores alternatives via utility function (below); walks relaxation ladder if no feasible plan.
- **Live Data Worker** (`workers/live_data_worker.py`) — separate process; webhook subscribe + poll fallback; deterministic disruption rules; normalises to `DisruptionEvent`; writes to `disruption_queue` via checkpointer transactional API; emits wake-up signal.

## Architecture: Replanning Utility Function

```
U(plan) = 0.55 · constraint_score   # 1.0 if all hard constraints met
        + 0.20 · budget_score        # 1 − max(0, overrun) / daily_cap
        + 0.15 · quality_score       # RAG confidence + reviews
        + 0.10 · disruption_blast_radius  # fewer downstream perturbations = better
```

A plan with `constraint_score < 1.0` is only emitted as a relaxation **proposal**, never silently applied. The relaxation ladder (in order): (1) widen radius ≤30%, (2) widen budget ≤15%, (3) downgrade accommodation flexibility one step, (4) loosen one secondary dietary tag (never primary halal/kosher/medical), (5) escalate to user. Each step is logged.

## Architecture: RAG Pipeline

Retrieval flow: constraint pre-filter (hard payload filter in Qdrant) → hybrid BM25 + dense search with RRF fusion → top-50 to bge-reranker-v2-m3 cross-encoder → ranked results.

Confidence score: `0.5 · source_reliability + 0.3 · recency_factor + 0.2 · cross_source_agreement`

Empty-result behaviour: climb the relaxation ladder and emit a structured "no exact match — proposing N relaxations" — never silent fallback.

`dietary_tags` use a **controlled vocabulary** (`rag/vocabulary/`). User free text is mapped by Haiku with a confidence threshold; unmapped values trigger human review, not silent discard.

## Required Environment Variables

```
ANTHROPIC_API_KEY
LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_TRACING
QDRANT_URL, QDRANT_API_KEY
POSTGRES_DSN                  # prod checkpointer
REDIS_URL
DUFFEL_API_KEY                # flight MCP
BOOKING_API_KEY / EXPEDIA_API_KEY
WHEELMAP_API_KEY
OPENWEATHER_API_KEY
WISE_API_KEY                  # FX with card-markup model
VAULT_ADDR, VAULT_TOKEN       # prod secrets
PII_REDACTION_CONFIG          # versioned redaction rules path
```

SQLite is used as the checkpointer in dev (no `POSTGRES_DSN` needed).

## Key Invariants

- **PII redaction** — the `security/pii_redactor.py` middleware must wrap every LangSmith export and structured log. `medical_needs` and free-text health descriptions are hashed with a per-tenant salt. CI fails if a LangSmith export contains medical PII.
- **No silent degradation** — Claude API outage: serve last checkpoint, queue replans in Redis with TTL, notify user explicitly. MCP outage: serve `live_data` within `ttl`, flag staleness.
- **Corpus-vs-live reconciliation** — when the same datum exists in both corpus and live MCP (e.g. Wheelmap), the live source wins if its timestamp is within 7 days; otherwise the corpus value is used and a re-verification job is enqueued. Conflicts produce a `confidence_penalty`.
- **Disruption deduplication** — `event_key` hash prevents double-processing when both webhook and poll detect the same event.
- **Medical guardrail** — queries crossing from information-surfacing to clinical decision-making (dosing, symptom interpretation) are intercepted and redirected; covered by a regression eval set in CI.
- **Prompt injection defence** — user free text is fenced inside agent prompts with a delimiter contract; injection-detection eval runs in CI.

## Development Phases (Roadmap)

| Phase | Focus |
|---|---|
| W1–2 | Graph + state + Constraint Profiler + Postgres checkpointer + LangSmith with redaction |
| W3–4 | Qdrant cluster (dev) + corpus ingest + bge-m3 + Destination Research + Itinerary Builder |
| W5–6 | Live Data Worker + provider MCPs + disruption rule engine |
| W7–8 | Replanning + BudgetLedger + relaxation ladder + offline fallback staging |
| W9–10 | PII redaction + encryption at rest + DSAR endpoints + Helm + Terraform |
| W11–12 | Full test suite + LangSmith eval datasets + Grafana SLO dashboards |
| W13+ | Streamlit demo (multi-device sync) + runbooks |
