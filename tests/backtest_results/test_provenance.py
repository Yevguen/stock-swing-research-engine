from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    ExecutionApplicationStatus,
    ExecutionProvenanceSource,
    HistoricalBacktestResultValidationError,
    compute_source_payload_fingerprint,
    compute_source_run_fingerprint,
    project_allocation_provenance,
    project_execution_provenance,
    project_ranking_provenance,
    project_rejections,
    project_signal_provenance,
    project_transition_audits,
    validate_historical_backtest_source_run,
)
from stock_swing_d1.backtester import (
    HistoricalBacktestOrchestrator,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionInput,
)
from stock_swing_d1.execution.entry import EntryExecutionStatus
from stock_swing_d1.portfolio import PortfolioCandidateAction
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from tests.backtest_results.conftest import (
    SOURCE_DECISION_INTERVAL,
    SOURCE_ENTRY_SESSION,
    SOURCE_SESSION,
    source_decision_time,
)


def _empty_run() -> HistoricalBacktestRunResult:
    state = PortfolioState(settled_cash=Decimal("1000"))
    fingerprint = hash_portfolio_state(state)
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        initial_state_fingerprint=fingerprint,
        final_state_fingerprint=fingerprint,
    )


def _corrupt_dataclass(value, **updates):
    corrupted = copy(value)
    for field_name, replacement in updates.items():
        object.__setattr__(corrupted, field_name, replacement)
    return corrupted


def test_empty_run_has_no_synthetic_projection_rows() -> None:
    run = _empty_run()

    assert project_transition_audits(run) == ()
    assert project_signal_provenance(run) == ()
    assert project_ranking_provenance(run) == ((), ())
    assert project_allocation_provenance(run) == ((), ())
    assert project_execution_provenance(run) == ()
    assert project_rejections(run) == ()


def test_transition_projection_uses_the_authoritative_state_chain(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    rows = project_transition_audits(run)

    assert len(rows) == 2
    assert tuple(row.session for row in rows) == tuple(
        session.session for session in run.session_results
    )
    assert rows[0].state_hash_before == run.initial_state_fingerprint
    assert rows[-1].state_hash_after == run.final_state_fingerprint
    assert rows[0].settled_cash_before == run.initial_state.settled_cash
    assert rows[1].open_position_count_after == 1
    assert rows[1].ledger_entry_count == 1
    assert rows[1].newly_applied_event_ids == (
        run.session_results[1].ordered_execution_events[0].execution_id,
    )


def test_zero_candidate_ranking_and_allocation_cycles_are_preserved(
    zero_allocation_run,
) -> None:
    ranking_cycles, ranking_candidates = project_ranking_provenance(
        zero_allocation_run
    )
    allocation_cycles, allocation_candidates = project_allocation_provenance(
        zero_allocation_run
    )

    assert len(ranking_cycles) == 1
    assert ranking_cycles[0].candidate_count == 0
    assert ranking_candidates == ()
    assert len(allocation_cycles) == 1
    assert allocation_cycles[0].candidate_count == 0
    assert allocation_cycles[0].admitted_count == 0
    assert allocation_cycles[0].rejected_count == 0
    assert allocation_candidates == ()


def test_signal_ranking_and_allocation_values_are_copied_exactly(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    source_session = run.session_results[0]

    signal_rows = project_signal_provenance(run)
    ranking_cycles, ranking_rows = project_ranking_provenance(run)
    allocation_cycles, allocation_rows = project_allocation_provenance(run)

    assert tuple(row.security_id for row in signal_rows) == tuple(
        decision.security_id for decision in source_session.signal_decisions
    )
    assert signal_rows[0].sma20 == source_session.signal_decisions[0].sma_20
    assert signal_rows[0].atr_fraction == (
        source_session.signal_decisions[0].atr_fraction
    )
    assert signal_rows[0].source_payload_fingerprint == (
        compute_source_payload_fingerprint(
            source_type="baseline_signal_decision",
            payload=source_session.signal_decisions[0],
        )
    )

    snapshot = source_session.ranking_snapshot
    assert ranking_cycles[0].snapshot_fingerprint == snapshot.snapshot_fingerprint
    assert tuple(row.rank for row in ranking_rows[:6]) == tuple(range(1, 7))
    assert ranking_rows[0].trend_separation_atr == (
        snapshot.ranked_candidates[0].trend_separation_atr
    )
    assert ranking_rows[0].candidate_input_fingerprint == (
        snapshot.ranked_candidates[0].input_fingerprint
    )

    allocation = source_session.allocation_decision
    assert allocation is not None
    assert allocation_cycles[0].allocation_policy_fingerprint == (
        allocation.allocation_policy_ref.policy_fingerprint
    )
    assert allocation_cycles[0].settled_cash_input == Decimal(
        str(allocation.starting_cash)
    )
    assert tuple(row.processing_rank for row in allocation_rows) == tuple(
        range(1, 7)
    )
    assert allocation_rows[0].action is PortfolioCandidateAction.ADMITTED
    assert allocation_rows[-1].action is not PortfolioCandidateAction.ADMITTED
    assert allocation_rows[0].fixed_shares == (
        allocation.candidate_decisions[0].sized_pending_entry.fixed_shares
    )


def test_allocation_ranking_corruption_fails_closed(source_run_bundle) -> None:
    run, _manifest = source_run_bundle
    first = run.session_results[0]
    allocation = first.allocation_decision
    assert allocation is not None
    bad_candidate = _corrupt_dataclass(
        allocation.candidate_decisions[0],
        ranking_input_fingerprint="e" * 64,
    )
    bad_allocation = _corrupt_dataclass(
        allocation,
        candidate_decisions=(
            bad_candidate,
            *allocation.candidate_decisions[1:],
        ),
    )
    malformed_session = first.model_copy(
        update={"allocation_decision": bad_allocation}
    )
    malformed = run.model_copy(
        update={
            "session_results": (malformed_session, *run.session_results[1:])
        }
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ARTIFACT_PROVENANCE_MISMATCH",
    ):
        project_allocation_provenance(malformed)


def test_generated_buy_is_linked_to_executed_phase9_decision(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    source_session = run.session_results[1]
    event = source_session.ordered_execution_events[0]
    decision = next(
        value
        for value in source_session.entry_execution_decisions
        if value.status is EntryExecutionStatus.EXECUTED
    )

    rows = project_execution_provenance(run)

    assert len(rows) == 1
    row = rows[0]
    assert row.provenance_source is ExecutionProvenanceSource.GENERATED_ENTRY
    assert row.application_status is ExecutionApplicationStatus.APPLIED
    assert row.entry_decision_fingerprint == compute_source_payload_fingerprint(
        source_type="entry_execution_decision", payload=decision
    )
    assert row.exit_decision_fingerprint is None
    assert row.source_payload_fingerprint == compute_source_payload_fingerprint(
        source_type="portfolio_execution_event", payload=event
    )


def test_generated_buy_mismatch_cannot_fall_back_to_external(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    first, second = run.session_results[:2]
    event = second.ordered_execution_events[0]
    changed_event = event.model_copy(
        update={"fill_price": event.fill_price + Decimal("1")}
    )
    transition = PortfolioTransitionEngine.transition(
        first.authoritative_state,
        second.session,
        (changed_event,),
    )
    malformed_session = second.model_copy(
        update={
            "ordered_execution_events": (changed_event,),
            "state_transition_result": transition,
            "authoritative_state": transition.resulting_state,
        }
    )
    malformed = run.model_copy(
        update={
            "session_results": (first, malformed_session),
            "final_state": transition.resulting_state,
            "final_state_fingerprint": hash_portfolio_state(
                transition.resulting_state
            ),
        }
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ENTRY_LINKAGE_MISMATCH",
    ):
        project_execution_provenance(malformed)


def test_generated_sell_is_linked_to_terminal_phase15b_decision(
    generated_exit_run,
) -> None:
    session = generated_exit_run.session_results[0]
    event = session.ordered_execution_events[0]
    decision = session.open_position_exit_decisions[0]

    rows = project_execution_provenance(generated_exit_run)

    assert len(rows) == 1
    assert rows[0].provenance_source is ExecutionProvenanceSource.GENERATED_EXIT
    assert rows[0].application_status is ExecutionApplicationStatus.APPLIED
    assert rows[0].entry_decision_fingerprint is None
    assert rows[0].exit_decision_fingerprint == (
        compute_source_payload_fingerprint(
            source_type="open_position_exit_decision", payload=decision
        )
    )
    assert rows[0].source_payload_fingerprint == (
        compute_source_payload_fingerprint(
            source_type="portfolio_execution_event", payload=event
        )
    )


def test_generated_sell_mismatch_cannot_fall_back_to_external(
    generated_exit_run,
) -> None:
    session = generated_exit_run.session_results[0]
    event = session.ordered_execution_events[0]
    changed_event = event.model_copy(
        update={"fill_price": event.fill_price + Decimal("1")}
    )
    transition = PortfolioTransitionEngine.transition(
        generated_exit_run.initial_state,
        session.session,
        (changed_event,),
    )
    malformed_session = session.model_copy(
        update={
            "ordered_execution_events": (changed_event,),
            "state_transition_result": transition,
            "authoritative_state": transition.resulting_state,
        }
    )
    malformed = generated_exit_run.model_copy(
        update={
            "session_results": (malformed_session,),
            "final_state": transition.resulting_state,
            "final_state_fingerprint": hash_portfolio_state(
                transition.resulting_state
            ),
        }
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="EXIT_LINKAGE_MISMATCH",
    ):
        project_execution_provenance(malformed)


def test_external_scheduled_and_replayed_events_use_phase13_authority(
    external_execution_run,
    replay_execution_run,
) -> None:
    applied = project_execution_provenance(external_execution_run)
    replayed = project_execution_provenance(replay_execution_run)

    assert applied[0].provenance_source is (
        ExecutionProvenanceSource.EXTERNAL_SCHEDULED
    )
    assert applied[0].application_status is ExecutionApplicationStatus.APPLIED
    assert applied[0].entry_decision_fingerprint is None
    assert applied[0].exit_decision_fingerprint is None
    assert replayed[0].provenance_source is (
        ExecutionProvenanceSource.EXTERNAL_SCHEDULED
    )
    assert replayed[0].application_status is ExecutionApplicationStatus.REPLAYED


def test_execution_projection_preserves_phase15a_event_order() -> None:
    events = (
        PortfolioExecutionEvent(
            execution_id="BUY-2",
            source_order_id="ORDER-2",
            session=SOURCE_SESSION,
            asset_id="NORGATE:912",
            side=ExecutionSide.BUY,
            quantity=1,
            fill_price=Decimal("100"),
            execution_cost=Decimal("1"),
        ),
        PortfolioExecutionEvent(
            execution_id="BUY-1",
            source_order_id="ORDER-1",
            session=SOURCE_SESSION,
            asset_id="NORGATE:911",
            side=ExecutionSide.BUY,
            quantity=1,
            fill_price=Decimal("100"),
            execution_cost=Decimal("1"),
        ),
    )
    run = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        (
            HistoricalBacktestSessionInput(
                session=SOURCE_SESSION,
                decision_time=source_decision_time(SOURCE_SESSION),
                scheduled_execution_events=events,
            ),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )

    rows = project_execution_provenance(run)

    assert tuple(row.event_order for row in rows) == (0, 1)
    assert tuple(row.execution_id for row in rows) == tuple(
        event.execution_id
        for event in run.session_results[0].ordered_execution_events
    )


@dataclass(frozen=True)
class _CachedPayload:
    public_value: int
    _cache: str


def test_source_payload_fingerprint_is_domain_separated_and_cache_agnostic() -> None:
    first = _CachedPayload(public_value=1, _cache="first")
    second = _CachedPayload(public_value=1, _cache="second")

    fingerprint = compute_source_payload_fingerprint(
        source_type="test_payload", payload=first
    )

    assert fingerprint == compute_source_payload_fingerprint(
        source_type="test_payload", payload=first
    )
    assert fingerprint == compute_source_payload_fingerprint(
        source_type="test_payload", payload=second
    )
    assert fingerprint != compute_source_payload_fingerprint(
        source_type="other_payload", payload=first
    )
    assert fingerprint != compute_source_payload_fingerprint(
        source_type="test_payload", payload=replace(first, public_value=2)
    )


def test_valid_run_projections_are_unchanged_by_phase15d2a_validation(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    projections = (
        project_transition_audits,
        project_signal_provenance,
        project_ranking_provenance,
        project_allocation_provenance,
        project_execution_provenance,
        project_rejections,
    )
    before = tuple(projection(run) for projection in projections)

    validate_historical_backtest_source_run(
        run_result=run, run_manifest=manifest
    )

    after = tuple(projection(run) for projection in projections)
    assert after == before


def test_all_projections_are_deterministic_and_read_only(source_run_bundle) -> None:
    run, _manifest = source_run_bundle
    projections = (
        project_transition_audits,
        project_signal_provenance,
        project_ranking_provenance,
        project_allocation_provenance,
        project_execution_provenance,
        project_rejections,
    )
    source_fingerprint = compute_source_run_fingerprint(run)

    for projection in projections:
        assert projection(run) == projection(run)

    assert compute_source_run_fingerprint(run) == source_fingerprint
