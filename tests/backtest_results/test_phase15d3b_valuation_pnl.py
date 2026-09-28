"""Phase 15D.3B: valuation, equity, and P&L projection tests."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    HistoricalBacktestResultValidationError,
    HistoricalBacktestValuationMark,
    HistoricalBacktestValuationSnapshot,
    PolicyArtifactRef,
    build_run_manifest,
    build_valuation_snapshot,
    project_equity_curve,
    project_open_trades,
    project_session_pnl,
)
from stock_swing_d1.backtest_results.valuation import _compute_initial_valuation
from stock_swing_d1.backtester import (
    HistoricalBacktestOrchestrator,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionInput,
)
from stock_swing_d1.portfolio import PORTFOLIO_ALLOCATION_POLICY_REF
from stock_swing_d1.ranking.models import (
    CANDIDATE_RANKING_POLICY_ID,
    CANDIDATE_RANKING_POLICY_VERSION,
    RankingPolicyRef,
)
from stock_swing_d1.execution.open_position_exit import OpenPositionExitEvaluator
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
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


PRE0 = date(2026, 8, 28)
PRE1 = date(2026, 8, 29)
D0 = date(2026, 9, 1)
D1 = date(2026, 9, 2)
D2 = date(2026, 9, 3)
D3 = date(2026, 9, 4)


def _dt(session: date) -> datetime:
    return datetime.combine(session, time(20), tzinfo=timezone.utc)


def _session_input(session: date, **overrides) -> HistoricalBacktestSessionInput:
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
    return PortfolioTransitionEngine.transition(virgin, session, (event,)).resulting_state


def _mark(market_ref, *, security_id: str, session: date, close: Decimal):
    return HistoricalBacktestValuationMark(
        security_id=security_id,
        session=session,
        close=close,
        source_artifact_ref=market_ref,
    )


def _snap(session: date, marks):
    return build_valuation_snapshot(session=session, marks=marks)


# ---------------------------------------------------------------------------
# 1, 14, 15: cash-only initial valuation and exact session equity
# ---------------------------------------------------------------------------


def test_cash_only_initial_and_session_equity_are_exact(run_manifest) -> None:
    virgin = PortfolioState(settled_cash=Decimal("5000"))
    run = HistoricalBacktestOrchestrator().run(
        virgin,
        (_session_input(D1),),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )
    snapshots = (_snap(D1, ()),)

    equity_rows = project_equity_curve(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert len(equity_rows) == 1
    row = equity_rows[0]
    assert row.settled_cash == Decimal("5000")
    assert row.pending_receivable_value == Decimal("0")
    assert row.open_position_market_value == Decimal("0")
    assert row.equity == Decimal("5000")
    assert row.unrealized_pnl == Decimal("0")
    assert row.realized_pnl_this_session == Decimal("0")
    assert row.period_pnl == Decimal("0")  # implies initial_equity == 5000


# ---------------------------------------------------------------------------
# 2, 18, 19: initial and in-run pending settlements at face; settlement is
# not a new realized P&L event
# ---------------------------------------------------------------------------


def test_initial_and_session_pending_settlements_valued_at_face(
    run_manifest,
) -> None:
    virgin = PortfolioState(settled_cash=Decimal("10201"))
    buy = _buy(
        execution_id="BUY-X", session=PRE0, asset_id="NORGATE:5001", quantity=2
    )
    state_a = PortfolioTransitionEngine.transition(virgin, PRE0, (buy,)).resulting_state
    sell = _sell(
        execution_id="SELL-X",
        session=PRE1,
        asset_id="NORGATE:5001",
        quantity=2,
        fill_price=Decimal("110"),
        execution_cost=Decimal("1"),
        settlement_id="SETT-X",
        settlement_session=D2,
    )
    initial_state = PortfolioTransitionEngine.transition(
        state_a, PRE1, (sell,)
    ).resulting_state

    run = HistoricalBacktestOrchestrator().run(
        initial_state,
        (_session_input(D1),),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )
    equity_rows = project_equity_curve(
        run_result=run,
        run_manifest=run_manifest,
        valuation_snapshots=(_snap(D1, ()),),
    )

    assert len(equity_rows) == 1
    row = equity_rows[0]
    assert row.pending_receivable_value == Decimal("219")
    assert row.settled_cash == initial_state.settled_cash
    assert row.equity == initial_state.settled_cash + Decimal("219")
    assert row.period_pnl == Decimal("0")
    assert row.realized_pnl_this_session == Decimal("0")


def test_settlement_cash_movement_creates_no_new_realized_pnl_or_equity_change(
    run_manifest,
) -> None:
    virgin = PortfolioState(settled_cash=Decimal("10201"))
    buy = _buy(
        execution_id="BUY-Y", session=PRE0, asset_id="NORGATE:5002", quantity=2
    )
    state_a = PortfolioTransitionEngine.transition(virgin, PRE0, (buy,)).resulting_state
    sell = _sell(
        execution_id="SELL-Y",
        session=PRE1,
        asset_id="NORGATE:5002",
        quantity=2,
        fill_price=Decimal("110"),
        execution_cost=Decimal("1"),
        settlement_id="SETT-Y",
        settlement_session=D1,
    )
    initial_state = PortfolioTransitionEngine.transition(
        state_a, PRE1, (sell,)
    ).resulting_state

    run = HistoricalBacktestOrchestrator().run(
        initial_state,
        (_session_input(D1),),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )
    equity_rows = project_equity_curve(
        run_result=run,
        run_manifest=run_manifest,
        valuation_snapshots=(_snap(D1, ()),),
    )

    assert len(equity_rows) == 1
    row = equity_rows[0]
    assert row.pending_receivable_value == Decimal("0")
    assert row.settled_cash == initial_state.settled_cash + Decimal("219")
    assert row.equity == initial_state.settled_cash + Decimal("219")
    assert row.realized_pnl_this_session == Decimal("0")
    assert row.period_pnl == Decimal("0")


# ---------------------------------------------------------------------------
# 3, 4, 17, 22 (carried-in, closed during run): initial snapshot required;
# a future mark cannot substitute; realized P&L on SELL session
# ---------------------------------------------------------------------------


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
        (_session_input(D1, scheduled_execution_events=(sell,)),),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def test_carried_in_position_requires_initial_snapshot(run_manifest) -> None:
    run = _carried_in_closed_run()
    initial_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:2001",
        session=D0,
        close=Decimal("110"),
    )
    snapshots = (_snap(D0, (initial_mark,)), _snap(D1, ()))

    equity_rows = project_equity_curve(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    session_rows = project_session_pnl(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert len(equity_rows) == 1
    row = equity_rows[0]
    assert row.settled_cash == Decimal("10000")
    assert row.pending_receivable_value == Decimal("228")
    assert row.open_position_market_value == Decimal("0")
    assert row.equity == Decimal("10228")
    assert row.realized_pnl_this_session == Decimal("27")
    assert row.cumulative_realized_pnl == Decimal("27")
    assert row.unrealized_pnl == Decimal("0")
    assert row.period_pnl == Decimal("8")
    assert session_rows[0].period_pnl == row.period_pnl
    assert session_rows[0].realized_pnl_this_session == Decimal("27")

    # reconciliation: period_pnl == cumulative_realized + unrealized - initial_unrealized
    initial_valuation = _compute_initial_valuation(
        initial_state=run.initial_state,
        initial_snapshot=snapshots[0],
    )
    assert initial_valuation.initial_unrealized_pnl == Decimal("19")
    assert row.period_pnl == (
        row.cumulative_realized_pnl
        + row.unrealized_pnl
        - initial_valuation.initial_unrealized_pnl
    )


def test_missing_initial_snapshot_for_carried_in_position_rejected(
    run_manifest,
) -> None:
    run = _carried_in_closed_run()
    snapshots = (_snap(D1, ()),)  # omits the required D0 initial snapshot

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
        )


def test_first_future_session_mark_cannot_substitute_for_initial_mark(
    run_manifest,
) -> None:
    run = _carried_in_closed_run()
    # Attempt to smuggle the initial position's mark into the *processed*
    # session snapshot instead of a dedicated initial (D0) snapshot.
    smuggled_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:2001",
        session=D1,
        close=Decimal("110"),
    )
    snapshots = (_snap(D1, (smuggled_mark,)),)

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
        )


# ---------------------------------------------------------------------------
# 5, 6, 7, 8: missing / extra / duplicate / non-ascending snapshots rejected
# ---------------------------------------------------------------------------


def _two_session_cash_run() -> HistoricalBacktestRunResult:
    return HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        (_session_input(D1), _session_input(D2)),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def test_missing_processed_session_snapshot_rejected(run_manifest) -> None:
    run = _two_session_cash_run()
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(_snap(D1, ()),),
        )


def test_extra_snapshot_rejected(run_manifest) -> None:
    run = _two_session_cash_run()
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(_snap(D1, ()), _snap(D2, ()), _snap(D3, ())),
        )


def test_duplicate_snapshot_session_rejected(run_manifest) -> None:
    run = _two_session_cash_run()
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(_snap(D1, ()), _snap(D1, ())),
        )


def test_non_ascending_snapshots_rejected(run_manifest) -> None:
    run = _two_session_cash_run()
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(_snap(D2, ()), _snap(D1, ())),
        )


# ---------------------------------------------------------------------------
# 9, 10, 11, 12, 13, 16, 20, 21, 23, 25: exact mark coverage, cost_basis use,
# execution-cost reporting, period_pnl identity, in-run open trade, no
# forced liquidation
# ---------------------------------------------------------------------------


def _single_open_position_run() -> HistoricalBacktestRunResult:
    buy = _buy(
        execution_id="BUY-3001",
        session=D1,
        asset_id="NORGATE:3001",
        quantity=1,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    return HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("10000")),
        (_session_input(D1, scheduled_execution_events=(buy,)),),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def test_exact_mark_coverage_is_accepted(run_manifest) -> None:
    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    equity_rows = project_equity_curve(
        run_result=run,
        run_manifest=run_manifest,
        valuation_snapshots=(_snap(D1, (mark,)),),
    )
    assert len(equity_rows) == 1


def test_missing_security_mark_rejected(run_manifest) -> None:
    run = _single_open_position_run()
    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="VALUATION_MARK_COVERAGE_MISMATCH",
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(_snap(D1, ()),),
        )


def test_extra_security_mark_rejected(run_manifest) -> None:
    run = _single_open_position_run()
    marks = (
        _mark(
            run_manifest.market_data_artifact_ref,
            security_id="NORGATE:3001",
            session=D1,
            close=Decimal("120"),
        ),
        _mark(
            run_manifest.market_data_artifact_ref,
            security_id="NORGATE:9999",
            session=D1,
            close=Decimal("50"),
        ),
    )
    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="VALUATION_MARK_COVERAGE_MISMATCH",
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(_snap(D1, marks),),
        )


def test_empty_open_positions_require_empty_marks(run_manifest) -> None:
    run = _two_session_cash_run()
    stray_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:1",
        session=D1,
        close=Decimal("10"),
    )
    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="VALUATION_MARK_COVERAGE_MISMATCH",
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(_snap(D1, (stray_mark,)), _snap(D2, ())),
        )


def test_mark_source_artifact_ref_must_equal_manifest(
    run_manifest, artifact_ref_factory
) -> None:
    run = _single_open_position_run()
    wrong_ref = artifact_ref_factory("market_data", digest="f" * 64)
    assert wrong_ref != run_manifest.market_data_artifact_ref
    mark = _mark(
        wrong_ref, security_id="NORGATE:3001", session=D1, close=Decimal("120")
    )
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_ARTIFACT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(_snap(D1, (mark,)),),
        )


def test_unrealized_pnl_uses_phase13_cost_basis_and_reports_costs_once(
    run_manifest,
) -> None:
    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    snapshots = (_snap(D1, (mark,)),)

    equity_rows = project_equity_curve(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    session_rows = project_session_pnl(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    row = equity_rows[0]
    assert row.settled_cash == Decimal("9899")  # 10000 - (1*100+1)
    assert row.open_position_market_value == Decimal("120")
    assert row.unrealized_pnl == Decimal("19")  # 120 - (1*100+1)
    assert row.equity == Decimal("10019")
    assert row.period_pnl == Decimal("19")  # equity - initial_equity(10000)
    assert row.execution_cost_this_session == Decimal("1")
    assert row.cumulative_execution_cost == Decimal("1")
    assert session_rows[0].execution_cost_this_session == Decimal("1")
    assert session_rows[0].period_pnl == Decimal("19")


def test_final_in_run_open_position_yields_exact_open_trade_without_liquidation(
    run_manifest,
) -> None:
    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    snapshots = (_snap(D1, (mark,)),)

    open_trades = project_open_trades(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert len(open_trades) == 1
    trade = open_trades[0]
    assert trade.trade_id == "BUY-3001"
    assert trade.entry_execution_id == "BUY-3001"
    assert trade.carried_in is False
    assert trade.quantity == 1
    assert trade.entry_fill_price == Decimal("100")
    assert trade.entry_execution_cost == Decimal("1")
    assert trade.entry_cost_basis == Decimal("101")
    assert trade.final_mark_session == D1
    assert trade.final_mark_price == Decimal("120")
    assert trade.final_market_value == Decimal("120")
    assert trade.unrealized_pnl == Decimal("19")
    # no forced liquidation: the position is still open in final_state
    assert run.final_state.open_positions != ()
    assert run.final_state.open_positions[0].asset_id == "NORGATE:3001"


# ---------------------------------------------------------------------------
# 24: carried-in final open position -> exact HistoricalOpenTradeRecord
# ---------------------------------------------------------------------------


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


def test_carried_in_final_open_position_yields_exact_open_trade(
    run_manifest,
) -> None:
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
    snapshots = (
        _snap(ENTRY_SESSION, (initial_mark,)),
        _snap(EXIT_SESSION, (exit_mark,)),
    )

    equity_rows = project_equity_curve(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    open_trades = project_open_trades(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert len(equity_rows) == 1
    row = equity_rows[0]
    assert row.settled_cash == Decimal("10000")
    assert row.open_position_market_value == Decimal("224")
    assert row.unrealized_pnl == Decimal("23")
    assert row.equity == Decimal("10224")
    assert row.realized_pnl_this_session == Decimal("0")
    assert row.period_pnl == Decimal("14")

    assert len(open_trades) == 1
    trade = open_trades[0]
    assert trade.trade_id == "ENTRY-NORGATE:1001"
    assert trade.carried_in is True
    assert trade.entry_fill_price == Decimal("100")
    assert trade.entry_cost_basis == Decimal("201")
    assert trade.final_mark_session == EXIT_SESSION
    assert trade.final_mark_price == Decimal("112")
    assert trade.final_market_value == Decimal("224")
    assert trade.unrealized_pnl == Decimal("23")
    assert run.final_state.open_positions[0].asset_id == "NORGATE:1001"


# ---------------------------------------------------------------------------
# 22: carried-in baseline reconciliation across scenario families
# ---------------------------------------------------------------------------


def test_period_pnl_reconciliation_holds_across_scenario_families(
    run_manifest,
) -> None:
    virgin_run = _two_session_cash_run()
    virgin_snapshots = (_snap(D1, ()), _snap(D2, ()))

    single_open_run = _single_open_position_run()
    single_open_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    single_open_snapshots = (_snap(D1, (single_open_mark,)),)

    carried_in_closed_run = _carried_in_closed_run()
    carried_in_closed_initial_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:2001",
        session=D0,
        close=Decimal("110"),
    )
    carried_in_closed_snapshots = (
        _snap(D0, (carried_in_closed_initial_mark,)),
        _snap(D1, ()),
    )

    carried_in_open_run = _carried_in_still_open_run()
    carried_in_open_initial_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:1001",
        session=ENTRY_SESSION,
        close=Decimal("105"),
    )
    carried_in_open_exit_mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:1001",
        session=EXIT_SESSION,
        close=Decimal("112"),
    )
    carried_in_open_snapshots = (
        _snap(ENTRY_SESSION, (carried_in_open_initial_mark,)),
        _snap(EXIT_SESSION, (carried_in_open_exit_mark,)),
    )

    for run, snapshots in (
        (virgin_run, virgin_snapshots),
        (single_open_run, single_open_snapshots),
        (carried_in_closed_run, carried_in_closed_snapshots),
        (carried_in_open_run, carried_in_open_snapshots),
    ):
        rows = project_equity_curve(
            run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
        )
        initial_snapshot = next(
            (
                snapshot
                for snapshot in snapshots
                if snapshot.session == run.initial_state.as_of_session
            ),
            None,
        )
        initial_valuation = _compute_initial_valuation(
            initial_state=run.initial_state, initial_snapshot=initial_snapshot
        )
        for row in rows:
            assert row.period_pnl == (
                row.cumulative_realized_pnl
                + row.unrealized_pnl
                - initial_valuation.initial_unrealized_pnl
            )


# ---------------------------------------------------------------------------
# 26, 27: zero-session runs
# ---------------------------------------------------------------------------


def test_zero_session_cash_only_run(run_manifest) -> None:
    state = PortfolioState(settled_cash=Decimal("777"))
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        session_results=(),
        initial_state_fingerprint=hash_portfolio_state(state),
        final_state_fingerprint=hash_portfolio_state(state),
    )

    assert project_session_pnl(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=()
    ) == ()
    assert project_equity_curve(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=()
    ) == ()
    assert project_open_trades(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=()
    ) == ()


def test_zero_session_carried_in_open_position_run(run_manifest) -> None:
    state = _buy_only_state(asset_id="NORGATE:6001", session=D0)
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        session_results=(),
        initial_state_fingerprint=hash_portfolio_state(state),
        final_state_fingerprint=hash_portfolio_state(state),
    )
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:6001",
        session=D0,
        close=Decimal("115"),
    )
    snapshots = (_snap(D0, (mark,)),)

    assert project_session_pnl(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    ) == ()
    assert project_equity_curve(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    ) == ()

    open_trades = project_open_trades(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    assert len(open_trades) == 1
    trade = open_trades[0]
    assert trade.carried_in is True
    assert trade.final_mark_session == D0
    assert trade.final_mark_price == Decimal("115")
    assert trade.final_market_value == Decimal("230")
    assert trade.unrealized_pnl == Decimal("29")


def test_zero_session_run_requires_no_snapshot_when_cash_only(
    run_manifest,
) -> None:
    state = PortfolioState(settled_cash=Decimal("1"))
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        session_results=(),
        initial_state_fingerprint=hash_portfolio_state(state),
        final_state_fingerprint=hash_portfolio_state(state),
    )
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(_snap(D0, ()),),
        )


# ---------------------------------------------------------------------------
# 28: determinism
# ---------------------------------------------------------------------------


def test_repeated_calls_are_deterministic(run_manifest) -> None:
    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    snapshots = (_snap(D1, (mark,)),)

    for projector in (project_session_pnl, project_equity_curve, project_open_trades):
        first = projector(
            run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
        )
        second = projector(
            run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
        )
        assert first == second


# ---------------------------------------------------------------------------
# 29, 30: no provider/upstream owner calls; no float/tolerance/rounding
# ---------------------------------------------------------------------------


def test_no_provider_or_upstream_owner_calls_in_valuation_module() -> None:
    import inspect

    from stock_swing_d1.backtest_results import valuation

    source = inspect.getsource(valuation)
    forbidden = (
        "PortfolioTransitionEngine.transition(",
        "rank_candidates(",
        "allocate_ranked_candidates(",
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
    )
    for token in forbidden:
        assert token not in source


def test_no_float_tolerance_or_rounding_constructs_in_valuation_module() -> None:
    import inspect

    from stock_swing_d1.backtest_results import valuation

    source = inspect.getsource(valuation)
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


# ---------------------------------------------------------------------------
# Input-boundary hardening: exact valuation-snapshot container contract
# ---------------------------------------------------------------------------


def test_exact_tuple_of_valuation_snapshots_is_accepted(run_manifest) -> None:
    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    snapshots = (_snap(D1, (mark,)),)
    assert type(snapshots) is tuple

    rows = project_equity_curve(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    assert len(rows) == 1


def test_list_of_valuation_snapshots_is_rejected(run_manifest) -> None:
    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    snapshots_as_list = [_snap(D1, (mark,))]

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=snapshots_as_list,
        )


def test_generator_of_valuation_snapshots_is_rejected(run_manifest) -> None:
    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    snapshots_generator = (_snap(D1, (mark,)) for _ in range(1))

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=snapshots_generator,
        )


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda snap: {snap}, id="set"),
        pytest.param(lambda snap: frozenset({snap}), id="frozenset"),
        pytest.param(lambda snap: {snap.session: snap}, id="mapping"),
    ],
)
def test_other_iterable_containers_of_valuation_snapshots_are_rejected(
    run_manifest, wrap
) -> None:
    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    malformed_container = wrap(_snap(D1, (mark,)))
    assert type(malformed_container) is not tuple

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=malformed_container,
        )


def test_non_exact_snapshot_member_rejected_before_economic_projection(
    run_manifest,
) -> None:
    class _SnapshotLookalike(HistoricalBacktestValuationSnapshot):
        """Same shape as the frozen contract but not the exact type."""

    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    genuine = _snap(D1, (mark,))
    lookalike = _SnapshotLookalike.model_validate(genuine.model_dump(mode="python"))
    assert lookalike.model_dump(mode="python") == genuine.model_dump(mode="python")
    assert type(lookalike) is not HistoricalBacktestValuationSnapshot  # not exact type

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=(lookalike,),
        )


def test_non_snapshot_member_rejected(run_manifest) -> None:
    run = _single_open_position_run()

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="VALUATION_SNAPSHOT_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=run_manifest,
            valuation_snapshots=({"session": D1, "marks": ()},),
        )


# ---------------------------------------------------------------------------
# Input-boundary hardening: run_manifest is proven through the existing
# validated-source boundary before market_data_artifact_ref is trusted
# ---------------------------------------------------------------------------


def _manifest_with_wrong_ranking_policy(
    *, artifact_ref_factory, valuation_policy_ref, execution_cost_policy_ref
):
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
            policy_fingerprint="f" * 64,
        ),
        execution_cost_policy_ref=execution_cost_policy_ref,
        valuation_policy_ref=valuation_policy_ref,
        universe_artifact_ref=artifact_ref_factory("universe"),
        market_data_artifact_ref=artifact_ref_factory("market_data"),
    )


def test_manifest_ranking_policy_mismatch_fails_through_source_validation(
    artifact_ref_factory, valuation_policy_ref, execution_cost_policy_ref
) -> None:
    run = _single_open_position_run()
    mismatched_manifest = _manifest_with_wrong_ranking_policy(
        artifact_ref_factory=artifact_ref_factory,
        valuation_policy_ref=valuation_policy_ref,
        execution_cost_policy_ref=execution_cost_policy_ref,
    )
    mark = _mark(
        mismatched_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="RANKING_POLICY_MISMATCH"
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=mismatched_manifest,
            valuation_snapshots=(_snap(D1, (mark,)),),
        )
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="RANKING_POLICY_MISMATCH"
    ):
        project_session_pnl(
            run_result=run,
            run_manifest=mismatched_manifest,
            valuation_snapshots=(_snap(D1, (mark,)),),
        )
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="RANKING_POLICY_MISMATCH"
    ):
        project_open_trades(
            run_result=run,
            run_manifest=mismatched_manifest,
            valuation_snapshots=(_snap(D1, (mark,)),),
        )


def test_valid_manifest_behavior_remains_unchanged(run_manifest) -> None:
    run = _single_open_position_run()
    mark = _mark(
        run_manifest.market_data_artifact_ref,
        security_id="NORGATE:3001",
        session=D1,
        close=Decimal("120"),
    )
    snapshots = (_snap(D1, (mark,)),)

    equity_rows = project_equity_curve(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    session_rows = project_session_pnl(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    open_trades = project_open_trades(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )

    assert len(equity_rows) == 1
    assert equity_rows[0].equity == Decimal("10019")
    assert len(session_rows) == 1
    assert session_rows[0].period_pnl == Decimal("19")
    assert len(open_trades) == 1
    assert open_trades[0].trade_id == "BUY-3001"
    assert open_trades[0].unrealized_pnl == Decimal("19")
