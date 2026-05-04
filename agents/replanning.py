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
