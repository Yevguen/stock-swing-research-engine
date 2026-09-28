from __future__ import annotations

from copy import copy
from dataclasses import fields, replace
from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    HistoricalBacktestResultValidationError,
    PolicyArtifactRef,
    build_run_manifest,
    validate_historical_backtest_source_run,
)
from stock_swing_d1.backtest_results import source_validation
from stock_swing_d1.backtester.models import HistoricalBacktestRunResult
from stock_swing_d1.backtester import (
    HistoricalBacktestEntryIntent,
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionResult,
    HistoricalBacktestSessionInput,
    HistoricalDecisionInterval,
)
from stock_swing_d1.execution.costs import ExecutionCostPolicyRef
from stock_swing_d1.portfolio.portfolio_errors import PortfolioStateError
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_invariants import (
    PortfolioInvariantChecker,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioState,
    PortfolioTransitionResult,
)
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from stock_swing_d1.ranking import rank_candidates
from stock_swing_d1.ranking.models import RankingPolicyRef
from tests.backtest_results.conftest import (
    SOURCE_DECISION_INTERVAL,
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


def _persisted_audit_only_failure_run() -> HistoricalBacktestRunResult:
    buy = PortfolioExecutionEvent(
        execution_id="AUDIT-BUY",
        source_order_id="AUDIT-BUY-ORDER",
        session=date(2026, 8, 20),
        asset_id="NORGATE:801",
        side=ExecutionSide.BUY,
        quantity=2,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    bought = PortfolioTransitionEngine.transition(
        PortfolioState(settled_cash=Decimal("1000")),
        buy.session,
        (buy,),
    ).resulting_state
    sell = PortfolioExecutionEvent(
        execution_id="AUDIT-SELL",
        source_order_id="AUDIT-SELL-ORDER",
        session=SOURCE_SESSION,
        asset_id="NORGATE:801",
        side=ExecutionSide.SELL,
        quantity=2,
        fill_price=Decimal("101"),
        execution_cost=Decimal("1"),
        settlement_id="AUDIT-SETTLEMENT",
        settlement_session=date(2026, 8, 22),
    )
    initial = PortfolioTransitionEngine.transition(
        bought, sell.session, (sell,)
    ).resulting_state
    audit_session = date(2026, 8, 23)
    final = PortfolioState(
        base_currency=initial.base_currency,
        as_of_session=audit_session,
        state_version=initial.state_version + 1,
        settled_cash=initial.settled_cash,
        open_positions=initial.open_positions,
        pending_settlements=initial.pending_settlements,
        applied_events=initial.applied_events,
    )
    transition = PortfolioTransitionResult(
        session=audit_session,
        state_hash_before=hash_portfolio_state(initial),
        state_hash_after=hash_portfolio_state(final),
        resulting_state=final,
    )
    session = HistoricalBacktestSessionResult(
        session=audit_session,
        decision_time=source_decision_time(audit_session),
        prior_state_fingerprint=hash_portfolio_state(initial),
        state_transition_result=transition,
        authoritative_state=final,
        ranking_snapshot=rank_candidates(
            ranking_session=audit_session,
            decision_time=source_decision_time(audit_session),
            candidates=(),
        ),
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=initial,
        final_state=final,
        session_results=(session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _rebuild_manifest(manifest, **updates):
    values = {
        field_name: getattr(manifest, field_name)
        for field_name in (
            "software_revision",
            "strategy_configuration_ref",
            "allocation_policy_ref",
            "ranking_policy_ref",
            "execution_cost_policy_ref",
            "valuation_policy_ref",
            "universe_artifact_ref",
            "market_data_artifact_ref",
            "corporate_action_artifact_ref",
            "earnings_artifact_ref",
            "earnings_provider_name",
        )
    }
    values.update(updates)
    return build_run_manifest(**values)


def _replace_session_allocation(run, **updates):
    first = run.session_results[0]
    allocation = copy(first.allocation_decision)
    assert allocation is not None
    for field_name, value in updates.items():
        object.__setattr__(allocation, field_name, value)
    changed = first.model_copy(update={"allocation_decision": allocation})
    return run.model_copy(
        update={"session_results": (changed, *run.session_results[1:])}
    )


def test_empty_and_genuine_nonempty_source_runs_validate(
    run_manifest, source_run_bundle
) -> None:
    validate_historical_backtest_source_run(
        run_result=_empty_run(), run_manifest=run_manifest
    )
    run, manifest = source_run_bundle
    validate_historical_backtest_source_run(
        run_result=run, run_manifest=manifest
    )


def test_source_session_outside_authoritative_decision_interval_fails_closed(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    excluded = run.model_copy(
        update={
            "decision_interval": HistoricalDecisionInterval(
                decision_start_date=date(2026, 8, 25),
                decision_end_date=date(2026, 8, 31),
            )
        }
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="SESSION_OUTSIDE_DECISION_INTERVAL",
    ):
        validate_historical_backtest_source_run(
            run_result=excluded,
            run_manifest=manifest,
        )


def test_source_validation_never_reconstructs_missing_interval_from_sessions(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    malformed = copy(run)
    object.__delattr__(malformed, "decision_interval")

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="NONCANONICAL_SOURCE_RUN",
    ):
        validate_historical_backtest_source_run(
            run_result=malformed,
            run_manifest=manifest,
        )


def test_allocation_inputs_match_authoritative_phase13_state(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    session = run.session_results[0]
    allocation = session.allocation_decision
    assert allocation is not None

    assert Decimal(str(allocation.starting_cash)) == (
        session.authoritative_state.settled_cash
    )
    assert allocation.starting_open_position_count == len(
        session.authoritative_state.open_positions
    )
    validate_historical_backtest_source_run(
        run_result=run, run_manifest=manifest
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"starting_cash": 10_001.0},
        {"starting_open_position_count": 1},
    ],
    ids=("cash", "open-position-count"),
)
def test_allocation_state_boundary_corruption_fails_closed(
    source_run_bundle, updates
) -> None:
    run, manifest = source_run_bundle
    malformed = _replace_session_allocation(run, **updates)

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ARTIFACT_PROVENANCE_MISMATCH",
    ) as captured:
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=manifest
        )

    assert captured.value.code == "ARTIFACT_PROVENANCE_MISMATCH"


def test_same_session_sell_proceeds_are_not_added_to_allocation_cash(
    same_session_sell_allocation_run, same_session_sell_run_manifest
) -> None:
    run = same_session_sell_allocation_run
    manifest = same_session_sell_run_manifest
    session = run.session_results[0]
    allocation = session.allocation_decision
    assert allocation is not None

    state = session.authoritative_state
    assert len(state.pending_settlements) == 1
    pending = state.pending_settlements[0]
    assert pending.amount > Decimal("0")

    starting_cash = Decimal(str(allocation.starting_cash))
    assert starting_cash == state.settled_cash
    assert starting_cash != state.settled_cash + pending.amount

    validate_historical_backtest_source_run(
        run_result=run, run_manifest=manifest
    )


def test_entry_intents_are_canonical_and_carried_exactly(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    first, second = run.session_results

    assert first.future_entry_intents
    assert second.scheduled_entry_intents == first.future_entry_intents
    validate_historical_backtest_source_run(
        run_result=run, run_manifest=manifest
    )


def test_post_construction_entry_intent_corruption_is_rejected(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    first, second = run.session_results
    corrupted = first.future_entry_intents[0].model_copy(
        update={"source_rank": 0}
    )
    malformed_first = first.model_copy(
        update={
            "future_entry_intents": (
                corrupted,
                *first.future_entry_intents[1:],
            )
        }
    )
    malformed = run.model_copy(
        update={"session_results": (malformed_first, second)}
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="NONCANONICAL_SOURCE_RUN|ENTRY_LINKAGE_MISMATCH",
    ):
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=manifest
        )


def test_cross_session_entry_intent_order_must_be_exact(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    first, second = run.session_results
    assert len(first.future_entry_intents) >= 2
    malformed_second = second.model_copy(
        update={
            "scheduled_entry_intents": tuple(
                reversed(second.scheduled_entry_intents)
            )
        }
    )
    malformed = run.model_copy(
        update={"session_results": (first, malformed_second)}
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ENTRY_LINKAGE_MISMATCH",
    ):
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=manifest
        )


def test_cross_session_entry_intent_substitution_is_rejected(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    first, second = run.session_results
    original = second.scheduled_entry_intents[0]
    mutated_decision = copy(original.candidate_decision)
    object.__setattr__(
        mutated_decision,
        "processing_rank",
        mutated_decision.processing_rank + 1,
    )
    substituted = original.model_copy(
        update={"candidate_decision": mutated_decision}
    )
    assert substituted != original
    assert substituted.security_id == original.security_id

    # `substituted` is individually canonical, not malformed: every
    # candidate_decision field except processing_rank is untouched, and it
    # still satisfies the model's own structural/identity invariant
    # (unchanged here since that invariant never inspects processing_rank).
    # It differs from `original` only in authoritative lineage -- the rank
    # this decision claims to have been processed at no longer matches what
    # the real ranking/allocation batch produced -- which only
    # validate_historical_backtest_source_run's cross-session linkage check
    # detects, not per-object structural validation.
    changed_fields = {
        field.name
        for field in fields(original.candidate_decision)
        if getattr(mutated_decision, field.name)
        != getattr(original.candidate_decision, field.name)
    }
    assert changed_fields == {"processing_rank"}
    assert (
        HistoricalBacktestEntryIntent.validate_chronology_and_identity(
            substituted
        )
        is substituted
    )

    malformed_second = second.model_copy(
        update={
            "scheduled_entry_intents": (
                substituted,
                *second.scheduled_entry_intents[1:],
            )
        }
    )
    assert len(malformed_second.scheduled_entry_intents) == len(
        second.scheduled_entry_intents
    )
    malformed = run.model_copy(
        update={"session_results": (first, malformed_second)}
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ENTRY_LINKAGE_MISMATCH",
    ) as captured:
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=manifest
        )

    assert captured.value.code == "ENTRY_LINKAGE_MISMATCH"


def test_first_source_session_may_start_with_scheduled_intents(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    first, second = run.session_results
    boundary_run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=first.authoritative_state,
        final_state=second.authoritative_state,
        session_results=(second,),
        initial_state_fingerprint=hash_portfolio_state(
            first.authoritative_state
        ),
        final_state_fingerprint=hash_portfolio_state(
            second.authoritative_state
        ),
    )

    assert boundary_run.session_results[0].scheduled_entry_intents
    validate_historical_backtest_source_run(
        run_result=boundary_run, run_manifest=manifest
    )


def test_empty_previous_future_intents_do_not_forbid_scheduled_inputs(
    source_run_bundle,
) -> None:
    source, manifest = source_run_bundle
    allocation_session, entry_session = source.session_results
    bridge_date = entry_session.session.fromordinal(
        entry_session.session.toordinal() - 1
    )
    bridge_transition = PortfolioTransitionEngine.transition(
        allocation_session.authoritative_state,
        bridge_date,
        (),
    )
    bridge = HistoricalBacktestSessionResult(
        session=bridge_date,
        decision_time=source_decision_time(bridge_date),
        prior_state_fingerprint=hash_portfolio_state(
            allocation_session.authoritative_state
        ),
        state_transition_result=bridge_transition,
        authoritative_state=bridge_transition.resulting_state,
        ranking_snapshot=rank_candidates(
            ranking_session=bridge_date,
            decision_time=source_decision_time(bridge_date),
            candidates=(),
        ),
    )
    entry_transition = PortfolioTransitionEngine.transition(
        bridge.authoritative_state,
        entry_session.session,
        entry_session.ordered_execution_events,
    )
    carried = entry_session.model_copy(
        update={
            "prior_state_fingerprint": hash_portfolio_state(
                bridge.authoritative_state
            ),
            "state_transition_result": entry_transition,
            "authoritative_state": entry_transition.resulting_state,
        }
    )
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=allocation_session.authoritative_state,
        final_state=entry_transition.resulting_state,
        session_results=(bridge, carried),
        initial_state_fingerprint=hash_portfolio_state(
            allocation_session.authoritative_state
        ),
        final_state_fingerprint=hash_portfolio_state(
            entry_transition.resulting_state
        ),
    )

    assert bridge.future_entry_intents == ()
    assert carried.scheduled_entry_intents
    validate_historical_backtest_source_run(
        run_result=run, run_manifest=manifest
    )


@pytest.mark.parametrize("field_name", ["symbol", "signal_time"])
def test_entry_decision_checks_nested_signal_identity(
    source_run_bundle, field_name: str
) -> None:
    run, manifest = source_run_bundle
    first, second = run.session_results
    original = second.entry_execution_decisions[0]
    replacement = (
        "BROKEN"
        if field_name == "symbol"
        else original.signal_time.replace(microsecond=1)
    )
    corrupted = replace(original, **{field_name: replacement})
    malformed_second = second.model_copy(
        update={
            "entry_execution_decisions": (
                corrupted,
                *second.entry_execution_decisions[1:],
            )
        }
    )
    malformed = run.model_copy(
        update={"session_results": (first, malformed_second)}
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ENTRY_LINKAGE_MISMATCH",
    ):
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=manifest
        )


def test_exact_source_run_type_is_required(run_manifest) -> None:
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="INVALID_SOURCE_RUN"
    ) as captured:
        validate_historical_backtest_source_run(
            run_result=object(), run_manifest=run_manifest
        )

    assert captured.value.code == "INVALID_SOURCE_RUN"


@pytest.mark.parametrize(
    ("field_name", "code"),
    [
        ("initial_state_fingerprint", "INITIAL_STATE_HASH_MISMATCH"),
        ("final_state_fingerprint", "FINAL_STATE_HASH_MISMATCH"),
    ],
)
def test_initial_and_final_hash_mismatches_fail_closed(
    run_manifest, field_name: str, code: str
) -> None:
    malformed = _empty_run().model_copy(update={field_name: "0" * 64})

    with pytest.raises(
        HistoricalBacktestResultValidationError, match=code
    ) as captured:
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=run_manifest
        )

    assert captured.value.code == code


def test_empty_run_cannot_change_final_state(run_manifest) -> None:
    run = _empty_run()
    changed = PortfolioState(settled_cash=Decimal("999"))
    malformed = run.model_copy(
        update={
            "final_state": changed,
            "final_state_fingerprint": hash_portfolio_state(changed),
        }
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="STATE_CHAIN_BREAK"
    ):
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=run_manifest
        )


def test_session_order_is_rejected_without_sorting(source_run_bundle) -> None:
    run, manifest = source_run_bundle
    malformed = run.model_copy(
        update={"session_results": tuple(reversed(run.session_results))}
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="STATE_CHAIN_BREAK"
    ):
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=manifest
        )


def test_adjacent_state_chain_break_is_detected(source_run_bundle) -> None:
    run, manifest = source_run_bundle
    second = run.session_results[1].model_copy(
        update={"prior_state_fingerprint": "0" * 64}
    )
    malformed = run.model_copy(
        update={"session_results": (run.session_results[0], second)}
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="NONCANONICAL_SOURCE_RUN|STATE_CHAIN_BREAK",
    ):
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=manifest
        )


def test_genuine_three_session_state_chain_validates(run_manifest) -> None:
    sessions = tuple(
        SOURCE_SESSION.fromordinal(SOURCE_SESSION.toordinal() + offset)
        for offset in range(3)
    )
    run = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        tuple(
            HistoricalBacktestSessionInput(
                session=session,
                decision_time=source_decision_time(session),
            )
            for session in sessions
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )

    validate_historical_backtest_source_run(
        run_result=run, run_manifest=run_manifest
    )

    middle = run.session_results[1].model_copy(
        update={"prior_state_fingerprint": "0" * 64}
    )
    malformed = run.model_copy(
        update={
            "session_results": (
                run.session_results[0],
                middle,
                run.session_results[2],
            )
        }
    )
    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="NONCANONICAL_SOURCE_RUN|STATE_CHAIN_BREAK",
    ):
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=run_manifest
        )


def test_phase13_persisted_accounting_bridge_uses_exact_current_arguments(
    source_run_bundle, monkeypatch
) -> None:
    calls = []
    original = PortfolioInvariantChecker.validate_persisted_accounting_history

    def recording(
        initial_state,
        final_state,
        ledger_entries,
        session_snapshots,
        dividend_ledger_entries=(),
        dividend_evidence=(),
        dividend_outcomes=(),
    ):
        calls.append(
            (
                initial_state,
                final_state,
                ledger_entries,
                session_snapshots,
                dividend_ledger_entries,
                dividend_evidence,
                dividend_outcomes,
            )
        )
        return original(
            initial_state=initial_state,
            final_state=final_state,
            ledger_entries=ledger_entries,
            session_snapshots=session_snapshots,
            dividend_ledger_entries=dividend_ledger_entries,
            dividend_evidence=dividend_evidence,
            dividend_outcomes=dividend_outcomes,
        )

    monkeypatch.setattr(
        PortfolioInvariantChecker,
        "validate_persisted_accounting_history",
        staticmethod(recording),
    )

    run, manifest = source_run_bundle
    validate_historical_backtest_source_run(
        run_result=run, run_manifest=manifest
    )

    assert len(calls) == 1
    (
        initial,
        final,
        ledger_entries,
        snapshots,
        dividend_ledger_entries,
        dividend_evidence,
        dividend_outcomes,
    ) = calls[0]
    assert initial == run.initial_state
    assert final == run.final_state
    assert ledger_entries == tuple(
        entry
        for session in run.session_results
        for entry in session.state_transition_result.ledger_entries
    )
    assert ledger_entries
    assert tuple(snapshot.session for snapshot in snapshots) == tuple(
        session.session for session in run.session_results
    )
    assert dividend_ledger_entries == tuple(
        entry
        for session in run.session_results
        for entry in session.state_transition_result.dividend_ledger_entries
    )
    assert dividend_outcomes == tuple(
        outcome
        for session in run.session_results
        for outcome in session.state_transition_result.dividend_outcomes
    )
    assert dividend_evidence == ()


def test_phase13_accounting_failure_is_wrapped_with_cause(
    run_manifest,
) -> None:
    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="PHASE13_ACCOUNTING_AUDIT_FAILED",
    ) as captured:
        validate_historical_backtest_source_run(
            run_result=_persisted_audit_only_failure_run(),
            run_manifest=run_manifest,
        )

    assert isinstance(captured.value.__cause__, PortfolioStateError)


def test_link_executions_defensive_assertion_is_wrapped_with_cause(
    source_run_bundle, monkeypatch
) -> None:
    # `_link_executions` defensively asserts that an EXECUTED Phase 9 entry
    # decision carries its execution cost quote; that AssertionError must
    # never escape the public API as a raw AssertionError.
    run, manifest = source_run_bundle
    injected_error = AssertionError(
        "simulated defensive assertion for ENTRY_LINKAGE_MISMATCH coverage"
    )

    def _raising_link_executions(session, *, state_before):
        raise injected_error

    monkeypatch.setattr(
        source_validation, "_link_executions", _raising_link_executions
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ENTRY_LINKAGE_MISMATCH",
    ) as captured:
        validate_historical_backtest_source_run(
            run_result=run, run_manifest=manifest
        )

    assert captured.value.code == "ENTRY_LINKAGE_MISMATCH"
    assert captured.value.__cause__ is injected_error


def test_ranking_policy_must_equal_manifest(source_run_bundle) -> None:
    run, manifest = source_run_bundle
    mismatch = _rebuild_manifest(
        manifest,
        ranking_policy_ref=RankingPolicyRef(
            policy_id="different-ranking",
            policy_version="0.1",
            policy_fingerprint="e" * 64,
        ),
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="RANKING_POLICY_MISMATCH",
    ):
        validate_historical_backtest_source_run(
            run_result=run, run_manifest=mismatch
        )


def test_allocation_policy_must_equal_manifest(source_run_bundle) -> None:
    run, manifest = source_run_bundle
    mismatch = _rebuild_manifest(
        manifest,
        allocation_policy_ref=PolicyArtifactRef(
            policy_id="different-allocation",
            policy_version="0.1",
            policy_fingerprint="d" * 64,
        ),
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ARTIFACT_PROVENANCE_MISMATCH",
    ):
        validate_historical_backtest_source_run(
            run_result=run, run_manifest=mismatch
        )


def test_execution_cost_policy_must_equal_manifest(source_run_bundle) -> None:
    run, manifest = source_run_bundle
    mismatch = _rebuild_manifest(
        manifest,
        execution_cost_policy_ref=ExecutionCostPolicyRef(
            policy_id=manifest.execution_cost_policy_ref.policy_id,
            policy_fingerprint="e" * 64,
        ),
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="EXECUTION_COST_POLICY_MISMATCH",
    ):
        validate_historical_backtest_source_run(
            run_result=run, run_manifest=mismatch
        )


def test_corrupted_ranking_snapshot_is_not_trusted(source_run_bundle) -> None:
    run, manifest = source_run_bundle
    first = run.session_results[0]
    snapshot = first.ranking_snapshot.model_copy(
        update={"candidate_count": first.ranking_snapshot.candidate_count - 1}
    )
    malformed_session = first.model_copy(update={"ranking_snapshot": snapshot})
    malformed = run.model_copy(
        update={
            "session_results": (malformed_session, run.session_results[1])
        }
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="NONCANONICAL_SOURCE_RUN",
    ):
        validate_historical_backtest_source_run(
            run_result=malformed, run_manifest=manifest
        )
