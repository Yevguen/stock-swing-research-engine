from __future__ import annotations

from dataclasses import replace

from stock_swing_d1.backtest_results import (
    HistoricalRejectionStage,
    compute_source_payload_fingerprint,
    project_rejections,
    project_signal_provenance,
)
from stock_swing_d1.backtester import HistoricalBacktestOrchestrator
from stock_swing_d1.execution.entry import EntryExecutionStatus
from stock_swing_d1.execution.open_position_exit import (
    EarningsExitStatus,
    OpenPositionExitEvaluator,
)
from stock_swing_d1.portfolio import PortfolioCandidateAction
from stock_swing_d1.strategy.baseline import BaselineSignalAction
from tests.backtester.test_phase15b_exit_integration import (
    earnings_decision,
    exit_evaluation,
    session_input,
    state_with_positions,
)
from tests.execution.open_position_exit.conftest import (
    ExplicitTradingCalendar,
    SESSIONS,
)
from tests.backtest_results.conftest import SOURCE_DECISION_INTERVAL


def test_allocation_and_entry_rejections_preserve_exact_upstream_reasons(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    rows = project_rejections(run)

    allocation_rows = tuple(
        row for row in rows if row.stage is HistoricalRejectionStage.ALLOCATION
    )
    entry_rows = tuple(
        row for row in rows if row.stage is HistoricalRejectionStage.ENTRY_EXECUTION
    )

    assert len(allocation_rows) == 1
    assert allocation_rows[0].security_id == "NORGATE:106"
    assert allocation_rows[0].reason == (
        PortfolioCandidateAction.REJECTED_MAX_SIMULTANEOUS_POSITIONS.value
    )
    assert allocation_rows[0].source_rank == 6

    source_decisions = {
        decision.security_id: decision
        for decision in run.session_results[1].entry_execution_decisions
        if decision.status is not EntryExecutionStatus.EXECUTED
    }
    assert len(entry_rows) == len(source_decisions) == 4
    assert {row.reason for row in entry_rows} == {
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION.value,
        EntryExecutionStatus.NO_EXECUTABLE_BAR.value,
    }
    for row in entry_rows:
        decision = source_decisions[row.security_id]
        assert row.reason == decision.status.value
        assert row.requested_quantity == decision.requested_shares
        assert row.source_artifact_fingerprint == (
            compute_source_payload_fingerprint(
                source_type="entry_execution_decision",
                payload=decision,
            )
        )


def test_cost_driven_cancellation_is_a_rejection_without_a_generated_buy(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    cancelled = next(
        decision
        for decision in run.session_results[1].entry_execution_decisions
        if decision.status
        is EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
    )

    row = next(
        rejection
        for rejection in project_rejections(run)
        if rejection.security_id == cancelled.security_id
        and rejection.stage is HistoricalRejectionStage.ENTRY_EXECUTION
    )

    assert row.reason == (
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION.value
    )
    assert all(
        event.asset_id != cancelled.security_id
        for event in run.session_results[1].ordered_execution_events
    )
    assert cancelled.executed_shares == 0


def test_rejections_use_session_stage_security_canonical_order(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    rows = project_rejections(run)
    stage_order = {
        HistoricalRejectionStage.ALLOCATION: 0,
        HistoricalRejectionStage.ENTRY_EXECUTION: 1,
    }

    assert tuple(
        (row.decision_session, stage_order[row.stage], row.security_id) for row in rows
    ) == tuple(
        sorted(
            (
                row.decision_session,
                stage_order[row.stage],
                row.security_id,
            )
            for row in rows
        )
    )


def test_no_signal_remains_signal_provenance_not_a_rejection(
    source_run_bundle,
    external_execution_run,
) -> None:
    source_run, _manifest = source_run_bundle
    source_signal = source_run.session_results[0].signal_decisions[-1]
    no_signal = replace(
        source_signal,
        security_id="NORGATE:999",
        symbol="S999",
        close_above_sma50=False,
        action=BaselineSignalAction.NO_SIGNAL,
    )
    session = external_execution_run.session_results[0].model_copy(
        update={"signal_decisions": (no_signal,)}
    )
    run = external_execution_run.model_copy(update={"session_results": (session,)})

    signals = project_signal_provenance(run)

    assert len(signals) == 1
    assert signals[0].action is BaselineSignalAction.NO_SIGNAL
    assert project_rejections(run) == ()


def test_deadline_missed_exit_audit_is_not_an_entry_rejection() -> None:
    asset_id = "NORGATE:1001"
    current_index = 5
    evaluation = exit_evaluation(
        asset_id,
        session_index=current_index,
        earnings_decision=earnings_decision(
            asset_id=asset_id,
            boundary_index=current_index,
            scheduled_index=current_index + 1,
        ),
        prior_boundary_earnings_decision=earnings_decision(
            asset_id=asset_id,
            boundary_index=current_index - 1,
            scheduled_index=None,
            active=False,
        ),
    )
    run = HistoricalBacktestOrchestrator(
        open_position_exit_evaluator=OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        )
    ).run(
        state_with_positions(asset_id),
        (
            session_input(
                session=SESSIONS[current_index],
                open_position_exit_evaluations=(evaluation,),
            ),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )

    decision = run.session_results[0].open_position_exit_decisions[0]
    assert decision.earnings_exit_status is EarningsExitStatus.DEADLINE_MISSED
    assert decision.exit_required is False
    assert project_rejections(run) == ()
