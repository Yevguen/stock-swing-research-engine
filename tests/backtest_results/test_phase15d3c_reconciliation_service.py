"""Phase 15D.3C: reconciliation and final in-memory audit-result assembly."""

from __future__ import annotations

import inspect
from datetime import date, datetime, time, timezone
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    ArtifactRef,
    HistoricalBacktestResultService,
    HistoricalBacktestResultValidationError,
    HistoricalBacktestRunManifest,
    HistoricalBacktestValuationMark,
    HistoricalBacktestValuationPolicy,
    HistoricalClosedTradeRecord,
    HistoricalExitReason,
    HistoricalOpenTradeRecord,
    HistoricalRejectionStage,
    PolicyArtifactRef,
    build_run_manifest,
    build_valuation_policy_ref,
    build_valuation_snapshot,
    compute_content_fingerprint,
    compute_result_fingerprint,
    compute_source_run_fingerprint,
    project_closed_trades,
    project_entries,
    project_execution_provenance,
    project_rejections,
)
from stock_swing_d1.backtester import (
    HistoricalBacktestOrchestrator,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionInput,
)
from stock_swing_d1.execution.costs.models import (
    BROKER_NEUTRAL_POLICY_ID,
    ExecutionCostPolicyRef,
)
from stock_swing_d1.execution.open_position_exit import OpenPositionExitEvaluator
from stock_swing_d1.portfolio import PORTFOLIO_ALLOCATION_POLICY_REF
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from stock_swing_d1.ranking.hashing import CANDIDATE_RANKING_POLICY_FINGERPRINT
from stock_swing_d1.ranking.models import (
    CANDIDATE_RANKING_POLICY_ID,
    CANDIDATE_RANKING_POLICY_VERSION,
    RankingPolicyRef,
)
from tests.backtester.test_phase15b_exit_integration import (
    ENTRY_SESSION,
    EXIT_SESSION,
    exit_evaluation,
    session_input,
    state_with_positions,
)
from tests.execution.open_position_exit.conftest import ExplicitTradingCalendar
from tests.backtest_results.conftest import SOURCE_DECISION_INTERVAL


D0 = date(2026, 11, 1)
D1 = date(2026, 11, 2)
D2 = date(2026, 11, 3)
D3 = date(2026, 11, 4)


def _dt(session: date) -> datetime:
    return datetime.combine(session, time(20), tzinfo=timezone.utc)


def _session(session: date, **overrides) -> HistoricalBacktestSessionInput:
    values = {"session": session, "decision_time": _dt(session)}
    values.update(overrides)
    return HistoricalBacktestSessionInput(**values)


def _buy(
    *,
    execution_id: str,
    session: date,
    asset_id: str,
    quantity: int = 1,
    fill_price: Decimal = Decimal("100"),
    execution_cost: Decimal = Decimal("1"),
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"{execution_id}-ORDER",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
    )


def _sell(
    *,
    execution_id: str,
    session: date,
    asset_id: str,
    quantity: int,
    fill_price: Decimal,
    execution_cost: Decimal,
    settlement_id: str,
    settlement_session: date,
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"{execution_id}-ORDER",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.SELL,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
        settlement_id=settlement_id,
        settlement_session=settlement_session,
    )


def _buy_only_state(
    *,
    asset_id: str,
    session: date,
    quantity: int = 2,
    fill_price: Decimal = Decimal("100"),
    execution_cost: Decimal = Decimal("1"),
    settled_cash_after: Decimal = Decimal("10000"),
) -> PortfolioState:
    cost = Decimal(quantity) * fill_price + execution_cost
    virgin = PortfolioState(settled_cash=settled_cash_after + cost)
    event = _buy(
        execution_id=f"BUY-{asset_id}",
        session=session,
        asset_id=asset_id,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
    )
    return PortfolioTransitionEngine.transition(
        virgin, session, (event,)
    ).resulting_state


def _mark(market_ref, *, security_id: str, session: date, close: Decimal):
    return HistoricalBacktestValuationMark(
        security_id=security_id,
        session=session,
        close=close,
        source_artifact_ref=market_ref,
    )


def _snap(session: date, marks):
    return build_valuation_snapshot(session=session, marks=marks)


def _zero_session_run(state: PortfolioState) -> HistoricalBacktestRunResult:
    fingerprint = hash_portfolio_state(state)
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        session_results=(),
        initial_state_fingerprint=fingerprint,
        final_state_fingerprint=fingerprint,
    )


def _in_run_closed_trade_run() -> HistoricalBacktestRunResult:
    entry_session = D1
    exit_session = D2
    buy = _buy(
        execution_id="RUN-BUY-1",
        session=entry_session,
        asset_id="NORGATE:8001",
        quantity=3,
        fill_price=Decimal("50"),
        execution_cost=Decimal("2"),
    )
    sell = _sell(
        execution_id="RUN-SELL-1",
        session=exit_session,
        asset_id="NORGATE:8001",
        quantity=3,
        fill_price=Decimal("55"),
        execution_cost=Decimal("2"),
        settlement_id="RUN-SETTLEMENT-1",
        settlement_session=D3,
    )
    return HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            _session(entry_session, scheduled_execution_events=(buy,)),
            _session(exit_session, scheduled_execution_events=(sell,)),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def _carried_in_closed_run() -> HistoricalBacktestRunResult:
    initial_state = _buy_only_state(asset_id="NORGATE:2001", session=D0)
    sell = _sell(
        execution_id="SELL-2001",
        session=D1,
        asset_id="NORGATE:2001",
        quantity=2,
        fill_price=Decimal("115"),
        execution_cost=Decimal("2"),
        settlement_id="SETT-2001",
        settlement_session=D3,
    )
    return HistoricalBacktestOrchestrator().run(
        initial_state,
        (_session(D1, scheduled_execution_events=(sell,)),),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def _carried_in_still_open_run() -> HistoricalBacktestRunResult:
    initial = state_with_positions("NORGATE:1001")
    orchestrator = HistoricalBacktestOrchestrator(
        open_position_exit_evaluator=OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        ),
    )
    return orchestrator.run(
        initial,
        (session_input(open_position_exit_evaluations=(exit_evaluation(),)),),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def _replay_buy_run() -> HistoricalBacktestRunResult:
    event = _buy(execution_id="REPLAY-BUY-1", session=D0, asset_id="NORGATE:9101")
    virgin = PortfolioState(settled_cash=Decimal("10000"))
    initial = PortfolioTransitionEngine.transition(
        virgin, D0, (event,)
    ).resulting_state
    transition = PortfolioTransitionEngine.transition(initial, D1, (event,))
    assert transition.ledger_entries == ()

    from stock_swing_d1.backtester.models import HistoricalBacktestSessionResult
    from stock_swing_d1.ranking import rank_candidates

    snapshot = rank_candidates(
        ranking_session=D1, decision_time=_dt(D1), candidates=()
    )
    session = HistoricalBacktestSessionResult(
        session=D1,
        decision_time=_dt(D1),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(event,),
        state_transition_result=transition,
        authoritative_state=transition.resulting_state,
        ranking_snapshot=snapshot,
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=initial,
        final_state=transition.resulting_state,
        session_results=(session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(transition.resulting_state),
    )


def _wrong_valuation_policy_manifest(
    *, artifact_ref_factory, ranking_policy_ref, execution_cost_policy_ref
) -> HistoricalBacktestRunManifest:
    from stock_swing_d1.backtest_results.models import (
        HISTORICAL_BACKTEST_VALUATION_POLICY_ID,
        HISTORICAL_BACKTEST_VALUATION_POLICY_SCHEMA_VERSION,
    )
    from stock_swing_d1.backtest_results.models import (
        HistoricalBacktestValuationPolicyRef,
    )

    wrong_ref = HistoricalBacktestValuationPolicyRef(
        policy_id=HISTORICAL_BACKTEST_VALUATION_POLICY_ID,
        schema_version=HISTORICAL_BACKTEST_VALUATION_POLICY_SCHEMA_VERSION,
        policy_fingerprint="f" * 64,
    )
    return build_run_manifest(
        software_revision="c" * 40,
        strategy_configuration_ref=artifact_ref_factory("strategy"),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id=PORTFOLIO_ALLOCATION_POLICY_REF.policy_id,
            policy_version=PORTFOLIO_ALLOCATION_POLICY_REF.policy_version,
            policy_fingerprint=PORTFOLIO_ALLOCATION_POLICY_REF.policy_fingerprint,
        ),
        ranking_policy_ref=ranking_policy_ref,
        execution_cost_policy_ref=execution_cost_policy_ref,
        valuation_policy_ref=wrong_ref,
        universe_artifact_ref=artifact_ref_factory("universe"),
        market_data_artifact_ref=artifact_ref_factory("market_data"),
    )


def _wrong_ranking_policy_manifest(
    *, artifact_ref_factory, valuation_policy_ref, execution_cost_policy_ref
) -> HistoricalBacktestRunManifest:
    return build_run_manifest(
        software_revision="c" * 40,
        strategy_configuration_ref=artifact_ref_factory("strategy"),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id=PORTFOLIO_ALLOCATION_POLICY_REF.policy_id,
            policy_version=PORTFOLIO_ALLOCATION_POLICY_REF.policy_version,
            policy_fingerprint=PORTFOLIO_ALLOCATION_POLICY_REF.policy_fingerprint,
        ),
        ranking_policy_ref=RankingPolicyRef(
            policy_id=CANDIDATE_RANKING_POLICY_ID,
            policy_version=CANDIDATE_RANKING_POLICY_VERSION,
            policy_fingerprint="e" * 64,
        ),
        execution_cost_policy_ref=execution_cost_policy_ref,
        valuation_policy_ref=valuation_policy_ref,
        universe_artifact_ref=artifact_ref_factory("universe"),
        market_data_artifact_ref=artifact_ref_factory("market_data"),
    )


def _manifest_with_wrong_allocation_policy(
    real_manifest: HistoricalBacktestRunManifest,
) -> HistoricalBacktestRunManifest:
    """Rebuild a real, otherwise-correct manifest with only the allocation
    policy fingerprint changed (same policy_id/version, wrong fingerprint,
    so it is structurally valid but disagrees with authoritative Phase 12
    provenance)."""

    return build_run_manifest(
        software_revision=real_manifest.software_revision,
        strategy_configuration_ref=real_manifest.strategy_configuration_ref,
        allocation_policy_ref=PolicyArtifactRef(
            policy_id=real_manifest.allocation_policy_ref.policy_id,
            policy_version=real_manifest.allocation_policy_ref.policy_version,
            policy_fingerprint="9" * 64,
        ),
        ranking_policy_ref=real_manifest.ranking_policy_ref,
        execution_cost_policy_ref=real_manifest.execution_cost_policy_ref,
        valuation_policy_ref=real_manifest.valuation_policy_ref,
        universe_artifact_ref=real_manifest.universe_artifact_ref,
        market_data_artifact_ref=real_manifest.market_data_artifact_ref,
    )


def _manifest_with_wrong_execution_cost_policy(
    real_manifest: HistoricalBacktestRunManifest,
) -> HistoricalBacktestRunManifest:
    """Rebuild a real, otherwise-correct manifest with only the
    execution-cost policy fingerprint changed (same policy_id, wrong
    fingerprint, so it is structurally valid but disagrees with
    authoritative execution-cost-quote provenance)."""

    return build_run_manifest(
        software_revision=real_manifest.software_revision,
        strategy_configuration_ref=real_manifest.strategy_configuration_ref,
        allocation_policy_ref=PolicyArtifactRef(
            policy_id=real_manifest.allocation_policy_ref.policy_id,
            policy_version=real_manifest.allocation_policy_ref.policy_version,
            policy_fingerprint=real_manifest.allocation_policy_ref.policy_fingerprint,
        ),
        ranking_policy_ref=real_manifest.ranking_policy_ref,
        execution_cost_policy_ref=ExecutionCostPolicyRef(
            policy_id=real_manifest.execution_cost_policy_ref.policy_id,
            policy_fingerprint="8" * 64,
        ),
        valuation_policy_ref=real_manifest.valuation_policy_ref,
        universe_artifact_ref=real_manifest.universe_artifact_ref,
        market_data_artifact_ref=real_manifest.market_data_artifact_ref,
    )


_ALL_AUDIT_FLAGS = (
    "source_run_canonical",
    "state_chain_valid",
    "ledger_reconstruction_valid",
    "execution_ledger_provenance_valid",
    "signal_provenance_valid",
    "ranking_provenance_valid",
    "allocation_provenance_valid",
    "execution_provenance_valid",
    "trade_linkage_valid",
    "valuation_coverage_valid",
    "equity_reconciliation_valid",
    "pnl_reconciliation_valid",
    "cost_reconciliation_valid",
    "policy_consistency_valid",
    "artifact_fingerprints_valid",
)


# ---------------------------------------------------------------------------
# Successful assembly: non-empty, zero-session cash-only, zero-session
# carried-in-open, carried-in-closed, carried-in-open (non-empty)
# ---------------------------------------------------------------------------


def test_successful_non_empty_full_audit_result_assembly(source_run_bundle) -> None:
    run, manifest = source_run_bundle
    entries = project_entries(run)
    assert len(entries) == 1
    entry = entries[0]
    snapshots = (
        _snap(run.session_results[0].session, ()),
        _snap(
            run.session_results[1].session,
            (
                _mark(
                    manifest.market_data_artifact_ref,
                    security_id=entry.security_id,
                    session=run.session_results[1].session,
                    close=Decimal("105"),
                ),
            ),
        ),
    )

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=manifest, valuation_snapshots=snapshots
    )

    assert result.audit_summary.audit_passed is True
    assert result.schema_version == "historical_backtest_audit_result.v0.2"
    assert result.decision_interval == run.decision_interval
    assert all(getattr(result.audit_summary, flag) for flag in _ALL_AUDIT_FLAGS)
    assert result.summary.entry_count == 1
    assert result.summary.open_trade_count == 1
    assert result.summary.closed_trade_count == 0
    assert len(result.trades) == 1
    assert result.result_fingerprint == compute_result_fingerprint(
        schema_version=result.schema_version,
        decision_interval=result.decision_interval,
        run_configuration_fingerprint=manifest.run_configuration_fingerprint,
        source_run_fingerprint=result.source_run_fingerprint,
        initial_state_fingerprint=result.initial_state_fingerprint,
        final_state_fingerprint=result.final_state_fingerprint,
        content_fingerprints=result.content_fingerprints,
    )


def test_zero_session_cash_only_result(run_manifest) -> None:
    state = PortfolioState(settled_cash=Decimal("777"))
    run = _zero_session_run(state)

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=()
    )

    assert result.audit_summary.audit_passed is True
    assert result.summary.processed_session_count == 0
    assert result.initial_equity == Decimal("777")
    assert result.final_equity == Decimal("777")
    assert result.period_pnl == Decimal("0")
    assert result.summary.gross_realized_pnl == Decimal("0")
    assert result.summary.total_execution_cost == Decimal("0")
    assert result.trades == ()
    assert result.session_pnl == ()
    assert result.equity_curve == ()


def test_zero_session_carried_in_open_position_result(run_manifest) -> None:
    state = _buy_only_state(asset_id="NORGATE:6001", session=D0)
    run = _zero_session_run(state)
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:6001",
        session=D0,
        close=Decimal("115"),
    )
    snapshots = (_snap(D0, (mark,)),)

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert result.audit_summary.audit_passed is True
    assert result.summary.processed_session_count == 0
    assert result.summary.open_trade_count == 1
    assert result.summary.closed_trade_count == 0
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert isinstance(trade, HistoricalOpenTradeRecord)
    assert trade.carried_in is True
    assert trade.final_mark_session == D0
    assert trade.final_market_value == Decimal("230")
    assert trade.unrealized_pnl == Decimal("29")
    assert result.initial_equity == state.settled_cash + Decimal("230")
    assert result.final_equity == result.initial_equity
    assert result.period_pnl == Decimal("0")
    assert result.summary.final_unrealized_pnl == Decimal("29")


def test_carried_in_trade_closed_during_run(run_manifest) -> None:
    run = _carried_in_closed_run()
    initial_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:2001",
        session=D0,
        close=Decimal("110"),
    )
    snapshots = (_snap(D0, (initial_mark,)), _snap(D1, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert result.audit_summary.audit_passed is True
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert isinstance(trade, HistoricalClosedTradeRecord)
    assert trade.carried_in is True
    assert trade.trade_id == "BUY-NORGATE:2001"
    assert trade.entry_execution_id == "BUY-NORGATE:2001"
    assert trade.realized_pnl == Decimal("27")
    assert result.summary.gross_realized_pnl == Decimal("27")
    assert result.entries == ()  # no synthetic entry row for a carried-in trade


def test_carried_in_trade_open_at_run_end_non_empty_run(run_manifest) -> None:
    run = _carried_in_still_open_run()
    initial_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:1001",
        session=ENTRY_SESSION,
        close=Decimal("105"),
    )
    exit_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:1001",
        session=EXIT_SESSION,
        close=Decimal("112"),
    )
    snapshots = (_snap(ENTRY_SESSION, (initial_mark,)), _snap(EXIT_SESSION, (exit_mark,)))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert result.audit_summary.audit_passed is True
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert isinstance(trade, HistoricalOpenTradeRecord)
    assert trade.carried_in is True
    assert trade.final_mark_session == EXIT_SESSION
    assert trade.unrealized_pnl == Decimal("23")
    assert result.entries == ()


# ---------------------------------------------------------------------------
# Trade merge order, uniqueness, applied-only linkage, replay exclusion
# ---------------------------------------------------------------------------


def test_exact_merged_trade_order_and_no_duplicate_trade_ids(run_manifest) -> None:
    # A carried-in position (entry D0) closed during the single processed
    # session, plus a brand-new position opened and left open in that same
    # session (entry D1). The merged order must place the earlier
    # entry_session first regardless of closed/open status.
    initial_state = _buy_only_state(asset_id="NORGATE:2001", session=D0)
    sell = _sell(
        execution_id="SELL-2001",
        session=D1,
        asset_id="NORGATE:2001",
        quantity=2,
        fill_price=Decimal("115"),
        execution_cost=Decimal("2"),
        settlement_id="SETT-2001",
        settlement_session=D3,
    )
    buy_new = _buy(
        execution_id="NEW-BUY-9001",
        session=D1,
        asset_id="NORGATE:9001",
        quantity=1,
        fill_price=Decimal("200"),
        execution_cost=Decimal("2"),
    )
    run = HistoricalBacktestOrchestrator().run(
        initial_state,
        (_session(D1, scheduled_execution_events=(sell, buy_new)),),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )
    initial_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:2001",
        session=D0,
        close=Decimal("110"),
    )
    final_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:9001",
        session=D1,
        close=Decimal("210"),
    )
    snapshots = (_snap(D0, (initial_mark,)), _snap(D1, (final_mark,)))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert len(result.trades) == 2
    assert tuple(trade.entry_session for trade in result.trades) == (D0, D1)
    assert isinstance(result.trades[0], HistoricalClosedTradeRecord)
    assert isinstance(result.trades[1], HistoricalOpenTradeRecord)
    trade_ids = tuple(trade.trade_id for trade in result.trades)
    assert len(set(trade_ids)) == len(trade_ids)
    assert result.trades == tuple(
        sorted(result.trades, key=lambda trade: (trade.entry_session, trade.trade_id))
    )


def test_every_applied_buy_sell_linked_exactly_once(run_manifest) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    provenance = project_execution_provenance(run)
    applied_buys = [
        row
        for row in provenance
        if row.application_status.value == "APPLIED" and row.side.value == "BUY"
    ]
    applied_sells = [
        row
        for row in provenance
        if row.application_status.value == "APPLIED" and row.side.value == "SELL"
    ]
    assert len(result.entries) == len(applied_buys) == 1
    assert len(result.exits) == len(applied_sells) == 1
    assert result.summary.entry_count == 1
    assert result.summary.exit_count == 1


def test_replayed_execution_contributes_no_trade_or_cost(run_manifest) -> None:
    # The BUY is applied once (baked directly into `initial_state`, outside
    # this run's own sessions) and then REPLAYED in the run's one session.
    # The replay must create no new entries/exits/cost row; the resulting
    # open position is a carried-in trade, not a newly-created one.
    run = _replay_buy_run()
    mark_d0 = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:9101",
        session=D0,
        close=Decimal("105"),
    )
    mark_d1 = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:9101",
        session=D1,
        close=Decimal("105"),
    )
    snapshots = (_snap(D0, (mark_d0,)), _snap(D1, (mark_d1,)))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert result.entries == ()
    assert result.exits == ()
    assert result.summary.entry_count == 0
    assert result.cost_summary.applied_buy_count == 0
    assert result.cost_summary.total_execution_cost == Decimal("0")
    assert len(result.trades) == 1
    assert isinstance(result.trades[0], HistoricalOpenTradeRecord)
    assert result.trades[0].carried_in is True


# ---------------------------------------------------------------------------
# Cost summary and reconciliation
# ---------------------------------------------------------------------------


def test_cost_summary_exact_and_reconciled_against_all_sources(
    run_manifest,
) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert result.cost_summary.buy_execution_cost_total == Decimal("2")
    assert result.cost_summary.sell_execution_cost_total == Decimal("2")
    assert result.cost_summary.total_execution_cost == Decimal("4")
    assert result.cost_summary.applied_buy_count == 1
    assert result.cost_summary.applied_sell_count == 1

    # against APPLIED execution provenance
    provenance = project_execution_provenance(run)
    applied_cost = sum(
        (
            row.execution_cost
            for row in provenance
            if row.application_status.value == "APPLIED"
        ),
        Decimal("0"),
    )
    assert result.cost_summary.total_execution_cost == applied_cost

    # against raw Phase 13 BUY_APPLIED/SELL_APPLIED ledger execution_cost
    raw_cost = Decimal("0")
    for session in run.session_results:
        for entry in session.state_transition_result.ledger_entries:
            if entry.event_type.value in ("BUY_APPLIED", "SELL_APPLIED"):
                raw_cost += entry.execution_cost
    assert result.cost_summary.total_execution_cost == raw_cost

    # against the final cumulative execution cost from Phase 15D.3B
    assert result.cost_summary.total_execution_cost == (
        result.equity_curve[-1].cumulative_execution_cost
    )


def test_corrupted_entry_cost_evidence_fails_closed(
    run_manifest, monkeypatch
) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    real_entries = project_entries(run)
    corrupted_entries = (
        real_entries[0].model_copy(update={"execution_cost": Decimal("999")}),
    )
    monkeypatch.setattr(
        "stock_swing_d1.backtest_results.service.project_entries",
        lambda _run_result: corrupted_entries,
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="COST_RECONCILIATION_MISMATCH"
    ):
        HistoricalBacktestResultService.build(
            run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
        )


# ---------------------------------------------------------------------------
# Exit-reason summary
# ---------------------------------------------------------------------------


def test_exit_reason_summary_complete_zero_filled_and_reconciled(
    run_manifest,
) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    rows = result.exit_reason_summary.rows
    assert tuple(row.reason for row in rows) == tuple(HistoricalExitReason)
    assert len(rows) == 7
    external_row = next(
        row for row in rows if row.reason is HistoricalExitReason.EXTERNAL_SCHEDULED
    )
    assert external_row.exit_count == 1
    assert external_row.realized_pnl == Decimal("11")
    zero_rows = [row for row in rows if row.reason != HistoricalExitReason.EXTERNAL_SCHEDULED]
    assert all(row.exit_count == 0 for row in zero_rows)
    assert all(row.realized_pnl == Decimal("0") for row in zero_rows)

    assert sum(row.exit_count for row in rows) == len(
        [t for t in result.trades if isinstance(t, HistoricalClosedTradeRecord)]
    )
    assert sum((row.realized_pnl for row in rows), Decimal("0")) == sum(
        (
            t.realized_pnl
            for t in result.trades
            if isinstance(t, HistoricalClosedTradeRecord)
        ),
        Decimal("0"),
    )


# ---------------------------------------------------------------------------
# Summary scalar exactness
# ---------------------------------------------------------------------------


def test_summary_counts_gross_realized_pnl_and_rejection_counts_exact(
    source_run_bundle,
) -> None:
    run, manifest = source_run_bundle
    entries = project_entries(run)
    entry = entries[0]
    snapshots = (
        _snap(run.session_results[0].session, ()),
        _snap(
            run.session_results[1].session,
            (
                _mark(
                    manifest.market_data_artifact_ref,
                    security_id=entry.security_id,
                    session=run.session_results[1].session,
                    close=Decimal("105"),
                ),
            ),
        ),
    )

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=manifest, valuation_snapshots=snapshots
    )

    rejections = project_rejections(run)
    expected_allocation_rejections = sum(
        1 for row in rejections if row.stage is HistoricalRejectionStage.ALLOCATION
    )
    expected_entry_rejections = sum(
        1
        for row in rejections
        if row.stage is HistoricalRejectionStage.ENTRY_EXECUTION
    )
    closed_trades = project_closed_trades(run)

    assert result.summary.allocation_rejection_count == expected_allocation_rejections
    assert result.summary.entry_execution_rejection_count == expected_entry_rejections
    assert result.summary.gross_realized_pnl == sum(
        (t.realized_pnl for t in closed_trades), Decimal("0")
    )
    assert result.summary.closed_trade_count == len(closed_trades)


def test_final_unrealized_pnl_and_period_pnl_exact(run_manifest) -> None:
    run = _carried_in_still_open_run()
    initial_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:1001",
        session=ENTRY_SESSION,
        close=Decimal("105"),
    )
    exit_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:1001",
        session=EXIT_SESSION,
        close=Decimal("112"),
    )
    snapshots = (_snap(ENTRY_SESSION, (initial_mark,)), _snap(EXIT_SESSION, (exit_mark,)))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert result.summary.final_unrealized_pnl == Decimal("23")
    assert result.summary.period_pnl == Decimal("14")
    assert result.period_pnl == result.summary.period_pnl


def test_top_level_scalars_equal_summary_scalars(run_manifest) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert result.initial_equity == result.summary.initial_equity
    assert result.final_equity == result.summary.final_equity
    assert result.period_pnl == result.summary.period_pnl


# ---------------------------------------------------------------------------
# Source validation gate and policy-lineage failures
# ---------------------------------------------------------------------------


def test_source_validation_with_real_manifest_occurs_before_final_assembly(
    run_manifest, monkeypatch
) -> None:
    run = _in_run_closed_trade_run()
    calls: list[str] = []

    def _spy(*, run_result, run_manifest):
        calls.append("validate_historical_backtest_source_run")
        raise HistoricalBacktestResultValidationError(
            "STATE_CHAIN_BREAK", "forced failure for ordering proof"
        )

    monkeypatch.setattr(
        "stock_swing_d1.backtest_results.service.validate_historical_backtest_source_run",
        _spy,
    )
    monkeypatch.setattr(
        "stock_swing_d1.backtest_results.service.project_entries",
        lambda _run_result: calls.append("project_entries") or (),
    )

    with pytest.raises(HistoricalBacktestResultValidationError, match="STATE_CHAIN_BREAK"):
        HistoricalBacktestResultService.build(
            run_result=run, run_manifest=run_manifest, valuation_snapshots=()
        )
    assert calls == ["validate_historical_backtest_source_run"]


def test_malformed_ranking_policy_lineage_fails_closed(
    artifact_ref_factory, valuation_policy_ref, execution_cost_policy_ref
) -> None:
    run = _in_run_closed_trade_run()
    bad_manifest = _wrong_ranking_policy_manifest(
        artifact_ref_factory=artifact_ref_factory,
        valuation_policy_ref=valuation_policy_ref,
        execution_cost_policy_ref=execution_cost_policy_ref,
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="RANKING_POLICY_MISMATCH"
    ):
        HistoricalBacktestResultService.build(
            run_result=run, run_manifest=bad_manifest, valuation_snapshots=()
        )


def test_wrong_but_valid_valuation_policy_fingerprint_fails_closed(
    artifact_ref_factory, ranking_policy_ref, execution_cost_policy_ref
) -> None:
    run = _in_run_closed_trade_run()
    bad_manifest = _wrong_valuation_policy_manifest(
        artifact_ref_factory=artifact_ref_factory,
        ranking_policy_ref=ranking_policy_ref,
        execution_cost_policy_ref=execution_cost_policy_ref,
    )
    assert bad_manifest.valuation_policy_ref.policy_id == (
        build_valuation_policy_ref(HistoricalBacktestValuationPolicy()).policy_id
    )
    assert bad_manifest.valuation_policy_ref.policy_fingerprint != (
        build_valuation_policy_ref(HistoricalBacktestValuationPolicy()).policy_fingerprint
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_ARTIFACT_MISMATCH"
    ):
        HistoricalBacktestResultService.build(
            run_result=run, run_manifest=bad_manifest, valuation_snapshots=()
        )


def test_allocation_policy_mismatch_fails_closed_through_the_service(
    source_run_bundle,
) -> None:
    # Uses a run with a REAL authoritative Phase 12 allocation cycle
    # (source_run_bundle), not a simple fixture with no allocation_decision
    # at all — _validate_allocation has nothing to compare in that case, and
    # such a test would pass vacuously rather than proving anything.
    run, real_manifest = source_run_bundle
    bad_manifest = _manifest_with_wrong_allocation_policy(real_manifest)
    assert bad_manifest.allocation_policy_ref != real_manifest.allocation_policy_ref

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ARTIFACT_PROVENANCE_MISMATCH",
    ):
        HistoricalBacktestResultService.build(
            run_result=run, run_manifest=bad_manifest, valuation_snapshots=()
        )


def test_execution_cost_policy_mismatch_fails_closed_through_the_service(
    source_run_bundle,
) -> None:
    # Uses the same real run: its one EXECUTED entry carries a genuine
    # authoritative execution-cost quote to compare against the manifest.
    run, real_manifest = source_run_bundle
    bad_manifest = _manifest_with_wrong_execution_cost_policy(real_manifest)
    assert (
        bad_manifest.execution_cost_policy_ref
        != real_manifest.execution_cost_policy_ref
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="EXECUTION_COST_POLICY_MISMATCH",
    ):
        HistoricalBacktestResultService.build(
            run_result=run, run_manifest=bad_manifest, valuation_snapshots=()
        )


def test_wrong_market_data_artifact_binding_on_a_mark_fails_closed_through_service(
    run_manifest, artifact_ref_factory
) -> None:
    run = _in_run_closed_trade_run()
    wrong_ref = artifact_ref_factory("market_data", digest="f" * 64)
    assert wrong_ref != run_manifest.market_data_artifact_ref

    bad_mark = _mark(
        wrong_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (bad_mark,)), _snap(D2, ()))

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_ARTIFACT_MISMATCH"
    ):
        HistoricalBacktestResultService.build(
            run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
        )


# ---------------------------------------------------------------------------
# Audit summary: proof-grounded flags, non-circular artifact_fingerprints_valid
# ---------------------------------------------------------------------------


def test_all_fifteen_audit_flags_true_and_audit_passed_matches(
    run_manifest,
) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    flags = {name: getattr(result.audit_summary, name) for name in _ALL_AUDIT_FLAGS}
    assert all(flags.values())
    assert result.audit_summary.audit_passed == all(flags.values())


def test_artifact_fingerprints_valid_is_not_circular_on_content_hashes() -> None:
    from stock_swing_d1.backtest_results import service as service_module

    source = inspect.getsource(service_module)
    flag_index = source.index("artifact_fingerprints_valid = True")
    content_index = source.index("content_fingerprints = HistoricalBacktestContentFingerprints(")
    assert flag_index < content_index, (
        "artifact_fingerprints_valid must be proven before the Phase 15D "
        "content fingerprints are ever built, not from them"
    )


# ---------------------------------------------------------------------------
# Content fingerprints and result fingerprint
# ---------------------------------------------------------------------------


def test_all_nineteen_content_fingerprints_match_direct_computation(
    run_manifest,
) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    fields = (
        ("session_transitions", result.session_transitions),
        ("signal_provenance", result.signal_provenance),
        ("ranking_cycles", result.ranking_cycles),
        ("ranking_candidates", result.ranking_candidates),
        ("allocation_cycles", result.allocation_cycles),
        ("allocation_candidates", result.allocation_candidates),
        ("execution_provenance", result.execution_provenance),
        ("entries", result.entries),
        ("exits", result.exits),
        ("trades", result.trades),
        ("rejections", result.rejections),
        ("cash_ledger", result.cash_ledger),
        ("settlement_ledger", result.settlement_ledger),
        ("session_pnl", result.session_pnl),
        ("equity_curve", result.equity_curve),
        ("cost_summary", result.cost_summary),
        ("exit_reason_summary", result.exit_reason_summary),
        ("summary", result.summary),
        ("audit_summary", result.audit_summary),
    )
    assert len(fields) == 19
    for name, value in fields:
        expected = compute_content_fingerprint(artifact_name=name, rows_or_value=value)
        assert getattr(result.content_fingerprints, name) == expected


def test_mutating_a_component_changes_its_content_fingerprint(run_manifest) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    original = compute_content_fingerprint(
        artifact_name="entries", rows_or_value=result.entries
    )
    mutated_entries = (
        result.entries[0].model_copy(update={"execution_cost": Decimal("999")}),
    )
    mutated = compute_content_fingerprint(
        artifact_name="entries", rows_or_value=mutated_entries
    )
    assert original != mutated
    assert result.content_fingerprints.entries == original


def test_result_fingerprint_equals_direct_computation(run_manifest) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    expected = compute_result_fingerprint(
        schema_version=result.schema_version,
        decision_interval=result.decision_interval,
        run_configuration_fingerprint=run_manifest.run_configuration_fingerprint,
        source_run_fingerprint=compute_source_run_fingerprint(run),
        initial_state_fingerprint=run.initial_state_fingerprint,
        final_state_fingerprint=run.final_state_fingerprint,
        content_fingerprints=result.content_fingerprints,
    )
    assert result.result_fingerprint == expected


def test_changing_a_content_fingerprint_changes_result_fingerprint(
    run_manifest,
) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    changed_content = result.content_fingerprints.model_copy(
        update={"entries": "f" * 64}
    )
    original_hash = compute_result_fingerprint(
        schema_version=result.schema_version,
        decision_interval=result.decision_interval,
        run_configuration_fingerprint=run_manifest.run_configuration_fingerprint,
        source_run_fingerprint=result.source_run_fingerprint,
        initial_state_fingerprint=result.initial_state_fingerprint,
        final_state_fingerprint=result.final_state_fingerprint,
        content_fingerprints=result.content_fingerprints,
    )
    changed_hash = compute_result_fingerprint(
        schema_version=result.schema_version,
        decision_interval=result.decision_interval,
        run_configuration_fingerprint=run_manifest.run_configuration_fingerprint,
        source_run_fingerprint=result.source_run_fingerprint,
        initial_state_fingerprint=result.initial_state_fingerprint,
        final_state_fingerprint=result.final_state_fingerprint,
        content_fingerprints=changed_content,
    )
    assert original_hash != changed_hash
    assert result.result_fingerprint == original_hash


# ---------------------------------------------------------------------------
# Determinism, read-only guarantees, and no forbidden operations
# ---------------------------------------------------------------------------


def test_repeated_build_calls_are_exactly_deterministic(run_manifest) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))

    first = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    second = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert first.result_fingerprint == second.result_fingerprint
    assert first == second


def test_source_run_and_valuation_snapshots_remain_unchanged(run_manifest) -> None:
    run = _in_run_closed_trade_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:8001",
        session=D1,
        close=Decimal("52"),
    )
    snapshots = (_snap(D1, (mark,)), _snap(D2, ()))
    run_fingerprint_before = compute_source_run_fingerprint(run)
    snapshots_before = snapshots

    HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert compute_source_run_fingerprint(run) == run_fingerprint_before
    assert snapshots == snapshots_before


def test_no_forbidden_upstream_owner_or_persistence_calls() -> None:
    from stock_swing_d1.backtest_results import reconciliation as reconciliation_module
    from stock_swing_d1.backtest_results import service as service_module

    source = inspect.getsource(service_module) + inspect.getsource(
        reconciliation_module
    )
    forbidden = (
        "PortfolioTransitionEngine.transition(",
        "rank_candidates(",
        "allocate_ranked_candidates(",
        "execute_pending_entry(",
        "evaluate_signal(",
        "OpenPositionExitEvaluator(",
        "BacktestBuyExecutionEventAdapter(",
        "BacktestSellExecutionEventAdapter(",
        "AdministrativeExitPricingService(",
        "HistoricalUsEquitySettlementResolver(",
        "SettlementSessionCalendar",
        "next_settlement_session",
        "norgatedata",
        "NorgateD1Adapter",
        "d1_pipeline",
        "open(",
        ".write(",
        "Path(",
        "pickle",
        "json.dump",
    )
    for token in forbidden:
        assert token not in source


def test_no_float_tolerance_or_rounding_constructs_in_new_modules() -> None:
    from stock_swing_d1.backtest_results import reconciliation as reconciliation_module
    from stock_swing_d1.backtest_results import service as service_module

    source = inspect.getsource(service_module) + inspect.getsource(
        reconciliation_module
    )
    forbidden = (
        "Decimal(float",
        "round(",
        ".quantize(",
        "math.isclose",
        "pytest.approx",
        "rel_tol",
        "abs_tol",
    )
    for token in forbidden:
        assert token not in source
