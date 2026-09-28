"""Slice 12 acceptance tests: Phase 15D portfolio-level ordinary-dividend
P&L integration and reconciliation.

Covers the frozen Ordinary Dividend Amendment v0.7's OD-20.6/OD-21.9/
OD-21.11/OD-21.12/OD-21.13 portfolio-scope contract:

- `ordinary_dividend_income_this_session` / `cumulative_ordinary_dividend_
  income` on `HistoricalBacktestSessionPnl` / `HistoricalBacktestEquityRow`,
  sourced exclusively from the projected cash ledger (never closed-trade
  totals), including dividends attributed to still-open and
  `PARTIAL_PRE_RUN_UNKNOWN` trades (acceptance cases K.84-90);
- the exact identity
  `period_pnl == cumulative_realized_pnl + cumulative_ordinary_dividend_
  income + unrealized_pnl - initial_unrealized_pnl` (K.86-87, K.96-97);
- realized/unrealized P&L, cost basis, and proceeds remain untouched
  (K.91-93);
- `HistoricalBacktestSummary.ordinary_dividend_income_total` (K.94-95);
- legacy (non-dividend-aware) runs never fabricate a dividend-aware zero.

Phase 16B consumption remains explicitly deferred to a later slice; this
file adds no winner/loser, profit-factor, or other Phase 16B assertion.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from stock_swing_d1.backtest_results import (
    HistoricalBacktestResultValidationError,
    HistoricalBacktestValuationMark,
    build_valuation_snapshot,
    project_closed_trades,
    project_equity_curve,
    project_session_pnl,
)
from stock_swing_d1.backtest_results.hashing import (
    SOURCE_RUN_HASH_DOMAIN,
    compute_content_fingerprint,
)
from stock_swing_d1.backtest_results.models import (
    DividendAttributionCompleteness,
    HistoricalBacktestEquityRow,
    HistoricalBacktestSessionPnl,
    HistoricalBacktestSummary,
)
from stock_swing_d1.backtest_results.persistence_schema import (
    BUNDLE_SCHEMA_VERSION,
    decode_model_row,
    encode_model_row,
)
from stock_swing_d1.backtest_results.valuation import (
    _compute_initial_valuation,
    _dividend_income_by_session,
)
from stock_swing_d1.backtester.models import (
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionResult,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import add_exact_decimal
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    ExecutionSide,
    PortfolioEventKind,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PendingSettlement,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine

from tests.backtest_results.conftest import (
    SOURCE_DECISION_INTERVAL,
    source_decision_time,
)
from tests.backtest_results.test_phase15d_dividend_cash_ledger_projection import (
    _build_buy_on_x_no_op_run,
    _build_replayed_run,
    _build_sell_and_reentry_run,
    _build_sell_on_x_run,
)
from tests.backtest_results.test_phase15d_dividend_source_validation import (
    ASSET_A,
    ASSET_B,
    T_SESSION,
    X_SESSION,
    _ranking,
    make_evidence,
    make_run_evidence,
    make_session_evidence,
)
from tests.backtest_results.test_phase15d_dividend_trade_grouping import (
    X2_SESSION,
    _build_carried_in_closed_trade_run,
    _build_legacy_carried_in_closed_trade_run,
    _build_multi_session_dividend_run,
    _build_real_applied_dividend_on_still_open_trade_run,
    _build_two_closed_trades_with_dividends_run,
    _build_zero_dividend_trade_run,
    _high_precision_row,
    _make_custom_evidence,
)

X3_SESSION = date(2026, 8, 21)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _mark(run_manifest, *, security_id: str, session: date, close: Decimal):
    return HistoricalBacktestValuationMark(
        security_id=security_id,
        session=session,
        close=close,
        source_artifact_ref=run_manifest.market_data_artifact_ref,
    )


def _snap(session: date, marks=()):
    return build_valuation_snapshot(session=session, marks=marks)


def _build_open_position_dividend_run(
    *,
    quantity: int = 10,
    entry_fill_price: Decimal = Decimal("100"),
    entry_cost: Decimal = Decimal("1"),
    d_capitalspecial: Decimal = Decimal("1"),
    unadjusted_close_t: Decimal = Decimal("1"),
    close_capital_t: Decimal = Decimal("1"),
    trade_id: str = "BUY-OPEN",
) -> HistoricalBacktestRunResult:
    """T: BUY. X: one dividend applied; the position is never sold."""

    initial = PortfolioState(settled_cash=Decimal("100000000"))
    buy = PortfolioExecutionEvent(
        execution_id=trade_id, source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=quantity,
        fill_price=entry_fill_price, execution_cost=entry_cost,
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    opened = t_result.resulting_state

    evidence = _make_custom_evidence(
        event_id=f"D:{trade_id}",
        d_capitalspecial=d_capitalspecial,
        unadjusted_close_t=unadjusted_close_t,
        close_capital_t=close_capital_t,
    )
    x_result = PortfolioTransitionEngine.transition(
        opened, X_SESSION, (), (evidence,)
    )
    final = x_result.resulting_state

    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy,),
        state_transition_result=t_result, authoritative_state=opened,
        ranking_snapshot=_ranking(T_SESSION),
    )
    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(opened),
        state_transition_result=x_result, authoritative_state=final,
        ranking_snapshot=_ranking(X_SESSION),
    )
    session_evidence = (
        make_session_evidence(session=T_SESSION),
        make_session_evidence(session=X_SESSION, distribution_events=(evidence,)),
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(t_session, x_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _build_carried_in_open_with_dividend_run() -> HistoricalBacktestRunResult:
    """Carried-in ASSET_A (nonzero initial unrealized P&L) stays open at
    run end and receives one in-run dividend on X while still held."""

    initial = PortfolioState(
        settled_cash=Decimal("100000"),
        open_positions=(
            OpenPosition(
                asset_id=ASSET_A, quantity=10, entry_session=date(2026, 8, 1),
                entry_price=Decimal("90"), entry_execution_id="CARRIED-OPEN",
                entry_execution_cost=Decimal("1"), cost_basis=Decimal("901"),
            ),
        ),
        applied_events=(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id="CARRIED-OPEN", payload_sha256="a" * 64,
            ),
        ),
        as_of_session=date(2026, 8, 1), state_version=1,
    )
    evidence = make_evidence(event_id="D:CARRIEDOPEN", security_id=ASSET_A)
    x_result = PortfolioTransitionEngine.transition(
        initial, X_SESSION, (), (evidence,)
    )
    final = x_result.resulting_state
    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        state_transition_result=x_result, authoritative_state=final,
        ranking_snapshot=_ranking(X_SESSION),
    )
    session_evidence = (
        make_session_evidence(session=X_SESSION, distribution_events=(evidence,)),
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(x_session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _build_all_four_nonzero_run() -> HistoricalBacktestRunResult:
    """Carried-in ASSET_A (nonzero initial unrealized) stays open with an
    in-run dividend on X; ASSET_B is bought at T and sold at X for a
    realized gain. Realized, dividend, unrealized, and initial-unrealized
    are all simultaneously nonzero."""

    initial = PortfolioState(
        settled_cash=Decimal("100000"),
        open_positions=(
            OpenPosition(
                asset_id=ASSET_A, quantity=10, entry_session=date(2026, 8, 1),
                entry_price=Decimal("90"), entry_execution_id="CARRIED-ALL",
                entry_execution_cost=Decimal("1"), cost_basis=Decimal("901"),
            ),
        ),
        applied_events=(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id="CARRIED-ALL", payload_sha256="a" * 64,
            ),
        ),
        as_of_session=date(2026, 8, 1), state_version=1,
    )
    buy_b = PortfolioExecutionEvent(
        execution_id="BUY-B-ALL", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_B, side=ExecutionSide.BUY, quantity=4,
        fill_price=Decimal("50"), execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy_b,))
    opened = t_result.resulting_state

    evidence = make_evidence(event_id="D:ALLFOUR", security_id=ASSET_A)
    sell_b = PortfolioExecutionEvent(
        execution_id="SELL-B-ALL", source_order_id="O-2", session=X_SESSION,
        asset_id=ASSET_B, side=ExecutionSide.SELL, quantity=4,
        fill_price=Decimal("60"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-B-ALL", settlement_session=X2_SESSION,
    )
    x_result = PortfolioTransitionEngine.transition(
        opened, X_SESSION, (sell_b,), (evidence,)
    )
    final = x_result.resulting_state

    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy_b,),
        state_transition_result=t_result, authoritative_state=opened,
        ranking_snapshot=_ranking(T_SESSION),
    )
    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(opened),
        ordered_execution_events=(sell_b,),
        state_transition_result=x_result, authoritative_state=final,
        ranking_snapshot=_ranking(X_SESSION),
    )
    session_evidence = (
        make_session_evidence(session=T_SESSION),
        make_session_evidence(session=X_SESSION, distribution_events=(evidence,)),
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(t_session, x_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _build_terminal_t_only_run() -> HistoricalBacktestRunResult:
    """A dividend-aware run that ends at T with zero dividends ever
    applied (no X is processed): the correct terminal-cutoff shape."""

    initial = PortfolioState(settled_cash=Decimal("100000"))
    buy = PortfolioExecutionEvent(
        execution_id="BUY-TERM", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    final = t_result.resulting_state
    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy,),
        state_transition_result=t_result, authoritative_state=final,
        ranking_snapshot=_ranking(T_SESSION),
    )
    session_evidence = (make_session_evidence(session=T_SESSION),)
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(t_session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _assert_identity_holds(run_result, session_rows, equity_rows, initial_snapshot):
    # Exact arithmetic throughout: the dividend term can require more
    # significant digits than the ambient Decimal context allows, and
    # naive `+`/`-` would silently round it (OD-6.7) -- this must not be
    # a false negative purely from the test's own comparison arithmetic.
    initial_valuation = _compute_initial_valuation(
        initial_state=run_result.initial_state, initial_snapshot=initial_snapshot
    )
    for row in (*session_rows, *equity_rows):
        dividend_aware = run_result.dividend_run_evidence is not None
        dividend_term = (
            row.cumulative_ordinary_dividend_income if dividend_aware else Decimal("0")
        )
        combined = add_exact_decimal(row.cumulative_realized_pnl, dividend_term)
        combined = add_exact_decimal(combined, row.unrealized_pnl)
        sign, digits, exponent = initial_valuation.initial_unrealized_pnl.as_tuple()
        negated_initial_unrealized = Decimal((0 if sign else 1, digits, exponent))
        expected = add_exact_decimal(combined, negated_initial_unrealized)
        assert row.period_pnl == expected


# ---------------------------------------------------------------------------
# Basic portfolio identity (required tests 1-10; acceptance K.84-87)
# ---------------------------------------------------------------------------


def test_dividend_aware_run_no_dividends_has_zero_cumulative(run_manifest) -> None:
    run = _build_zero_dividend_trade_run(dividend_aware=True)
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    for row in (*session_rows, *equity_rows):
        assert row.cumulative_ordinary_dividend_income == Decimal("0")
        assert row.ordinary_dividend_income_this_session == Decimal("0")
    _assert_identity_holds(run, session_rows, equity_rows, None)


def test_one_applied_dividend_included_exactly_once(run_manifest) -> None:
    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = next(r for r in equity_rows if r.session == X_SESSION)
    assert x_row.ordinary_dividend_income_this_session == Decimal("15")
    assert x_row.cumulative_ordinary_dividend_income == Decimal("15")
    _assert_identity_holds(run, session_rows, equity_rows, None)


def test_two_dividends_across_sessions_cumulative_increments(run_manifest) -> None:
    run = _build_multi_session_dividend_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("108")),)),
        _snap(X2_SESSION, ()),
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    by_session = {row.session: row for row in session_rows}
    assert by_session[X_SESSION].ordinary_dividend_income_this_session == Decimal("15")
    assert by_session[X_SESSION].cumulative_ordinary_dividend_income == Decimal("15")
    assert by_session[X2_SESSION].ordinary_dividend_income_this_session == Decimal("15")
    assert by_session[X2_SESSION].cumulative_ordinary_dividend_income == Decimal("30")
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    _assert_identity_holds(run, session_rows, equity_rows, None)


def test_two_dividends_same_session_exact_sum(run_manifest) -> None:
    run = _build_two_closed_trades_with_dividends_run()
    snapshots = (
        _snap(T_SESSION, (
            _mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("101")),
            _mark(run_manifest, security_id=ASSET_B, session=T_SESSION, close=Decimal("51")),
        )),
        _snap(X_SESSION, ()),
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = next(r for r in session_rows if r.session == X_SESSION)
    dividend_rows = tuple(project_closed_trades(run))
    expected = sum((t.ordinary_dividend_income for t in dividend_rows), Decimal("0"))
    assert x_row.ordinary_dividend_income_this_session == expected
    assert expected > 0
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    _assert_identity_holds(run, session_rows, equity_rows, None)


def test_zero_realized_dividend_zero_unrealized(run_manifest) -> None:
    run = _build_open_position_dividend_run(
        entry_fill_price=Decimal("100"), d_capitalspecial=Decimal("2"),
        unadjusted_close_t=Decimal("1"), close_capital_t=Decimal("1"),
    )
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("100")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("100")),)),
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = next(r for r in equity_rows if r.session == X_SESSION)
    assert x_row.cumulative_realized_pnl == Decimal("0")
    assert x_row.unrealized_pnl == Decimal("-1")  # entry cost of 1 only
    assert x_row.cumulative_ordinary_dividend_income == Decimal("20")
    _assert_identity_holds(run, session_rows, equity_rows, None)


def test_positive_realized_plus_dividend(run_manifest) -> None:
    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = next(r for r in session_rows if r.session == X_SESSION)
    assert x_row.cumulative_realized_pnl == Decimal("198")
    assert x_row.cumulative_ordinary_dividend_income == Decimal("15")
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    _assert_identity_holds(run, session_rows, equity_rows, None)


def test_negative_realized_plus_dividend(run_manifest) -> None:
    initial = PortfolioState(settled_cash=Decimal("1000000"))
    buy = PortfolioExecutionEvent(
        execution_id="BUY-NEGR", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    opened = t_result.resulting_state
    sell = PortfolioExecutionEvent(
        execution_id="SELL-NEGR", source_order_id="O-2", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=10,
        fill_price=Decimal("95"), execution_cost=Decimal("1"),
        settlement_id="SETT-NEGR", settlement_session=X2_SESSION,
    )
    evidence = _make_custom_evidence(
        event_id="D:NEGR", d_capitalspecial=Decimal("1"),
        unadjusted_close_t=Decimal("1"), close_capital_t=Decimal("1"),
    )
    x_result = PortfolioTransitionEngine.transition(opened, X_SESSION, (sell,), (evidence,))
    final = x_result.resulting_state
    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy,), state_transition_result=t_result,
        authoritative_state=opened, ranking_snapshot=_ranking(T_SESSION),
    )
    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(opened),
        ordered_execution_events=(sell,), state_transition_result=x_result,
        authoritative_state=final, ranking_snapshot=_ranking(X_SESSION),
    )
    session_evidence = (
        make_session_evidence(session=T_SESSION),
        make_session_evidence(session=X_SESSION, distribution_events=(evidence,)),
    )
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(t_session, x_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("100")),)),
        _snap(X_SESSION, ()),
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = next(r for r in session_rows if r.session == X_SESSION)
    assert x_row.cumulative_realized_pnl == Decimal("-52")
    assert x_row.cumulative_ordinary_dividend_income == Decimal("10")
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    _assert_identity_holds(run, session_rows, equity_rows, None)


def test_nonzero_unrealized_plus_dividend(run_manifest) -> None:
    run = _build_open_position_dividend_run(
        entry_fill_price=Decimal("100"), d_capitalspecial=Decimal("1"),
    )
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("100")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("120")),)),
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = next(r for r in equity_rows if r.session == X_SESSION)
    assert x_row.unrealized_pnl != Decimal("0")
    assert x_row.cumulative_ordinary_dividend_income == Decimal("10")
    _assert_identity_holds(run, session_rows, equity_rows, None)


def test_nonzero_initial_unrealized_plus_dividend(run_manifest) -> None:
    run = _build_carried_in_open_with_dividend_run()
    snapshots = (
        _snap(date(2026, 8, 1), (_mark(run_manifest, security_id=ASSET_A, session=date(2026, 8, 1), close=Decimal("95")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("98")),)),
    )
    initial_valuation = _compute_initial_valuation(
        initial_state=run.initial_state, initial_snapshot=snapshots[0]
    )
    assert initial_valuation.initial_unrealized_pnl != Decimal("0")
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    assert equity_rows[0].cumulative_ordinary_dividend_income == Decimal("15")
    _assert_identity_holds(run, session_rows, equity_rows, snapshots[0])


def test_all_four_components_nonzero(run_manifest) -> None:
    run = _build_all_four_nonzero_run()
    snapshots = (
        _snap(date(2026, 8, 1), (_mark(run_manifest, security_id=ASSET_A, session=date(2026, 8, 1), close=Decimal("95")),)),
        _snap(T_SESSION, (
            _mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("96")),
            _mark(run_manifest, security_id=ASSET_B, session=T_SESSION, close=Decimal("51")),
        )),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("98")),)),
    )
    initial_valuation = _compute_initial_valuation(
        initial_state=run.initial_state, initial_snapshot=snapshots[0]
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = next(r for r in equity_rows if r.session == X_SESSION)
    assert x_row.cumulative_realized_pnl != Decimal("0")
    assert x_row.cumulative_ordinary_dividend_income != Decimal("0")
    assert x_row.unrealized_pnl != Decimal("0")
    assert initial_valuation.initial_unrealized_pnl != Decimal("0")
    assert x_row.unrealized_pnl != initial_valuation.initial_unrealized_pnl
    _assert_identity_holds(run, session_rows, equity_rows, snapshots[0])


# ---------------------------------------------------------------------------
# Exact Decimal determinism (required tests 11-20)
# ---------------------------------------------------------------------------


def test_high_precision_dividend_reconciles_at_precisions_28_60_200(run_manifest) -> None:
    run = _build_open_position_dividend_run(
        quantity=9_999_999, entry_fill_price=Decimal("1"), entry_cost=Decimal("0"),
        d_capitalspecial=Decimal("1"), unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("3"), trade_id="BUY-HP",
    )
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("1")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("1")),)),
    )
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
            equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
        results.append((session_rows, equity_rows))
        _assert_identity_holds(run, session_rows, equity_rows, None)

    first_session, first_equity = results[0]
    for session_rows, equity_rows in results[1:]:
        assert session_rows == first_session
        assert equity_rows == first_equity

    x_row = next(r for r in first_equity if r.session == X_SESSION)
    assert x_row.cumulative_ordinary_dividend_income.adjusted() >= 0
    # Genuinely needs more than 28 significant digits.
    assert len(x_row.cumulative_ordinary_dividend_income.as_tuple().digits) > 28


def test_high_precision_naive_accumulation_would_have_rounded() -> None:
    row, _q_t, _d_h = _high_precision_row(trade_id="PNL-NAIVE")
    with localcontext() as ctx:
        ctx.prec = 28
        naive = Decimal("0") + row.settled_cash_delta
    assert naive != row.settled_cash_delta


def test_dividend_by_session_helper_is_exact_across_precisions() -> None:
    run = _build_open_position_dividend_run(
        quantity=9_999_999, entry_fill_price=Decimal("1"), entry_cost=Decimal("0"),
        d_capitalspecial=Decimal("1"), unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("3"), trade_id="BUY-HELPER",
    )
    with localcontext() as ctx:
        ctx.prec = 28
        totals_28 = _dividend_income_by_session(run)
    with localcontext() as ctx:
        ctx.prec = 200
        totals_200 = _dividend_income_by_session(run)
    assert totals_28 == totals_200
    assert type(totals_28[X_SESSION]) is Decimal


def test_no_float_or_rounding_constructs_touch_dividend_helpers() -> None:
    import inspect

    from stock_swing_d1.backtest_results import valuation

    source = inspect.getsource(valuation)
    for forbidden in ("Decimal(float", "round(", ".quantize(", "math.isclose", "pytest.approx"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Narrow audit follow-up: every additive/subtractive component feeding the
# published OD-21.11 identity must itself be ambient-Decimal-context
# independent, not only the dividend term / the identity's final outer
# composition. Each fixture below is deliberately constructed so that the
# *multiplication* `quantity * mark.close` is exact under every tested
# precision (operands never exceed 28 significant digits on their own) --
# isolating the addition/subtraction accumulation defect from the
# separate, explicitly out-of-scope multiplication-precision question
# (Slice-12-audit Section 4's STOP condition).
# ---------------------------------------------------------------------------


def test_cumulative_realized_pnl_exact_across_precisions(run_manifest) -> None:
    # Two closed trades whose individually-exact realized_pnl values
    # (10^20 exactly, and 10^-19 exactly) combine to a sum needing far
    # more than 28 significant digits once aligned -- naive `+=`
    # accumulation rounds this; add_exact_decimal must not.
    buy1 = PortfolioExecutionEvent(
        execution_id="BUY-R1", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=1,
        fill_price=Decimal("1"), execution_cost=Decimal("0"),
    )
    buy2 = PortfolioExecutionEvent(
        execution_id="BUY-R2", source_order_id="O-2", session=T_SESSION,
        asset_id=ASSET_B, side=ExecutionSide.BUY, quantity=1,
        fill_price=Decimal("1"), execution_cost=Decimal("0"),
    )
    initial = PortfolioState(settled_cash=Decimal("1000"))
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy1, buy2))
    opened = t_result.resulting_state
    sell1 = PortfolioExecutionEvent(
        execution_id="SELL-R1", source_order_id="O-3", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=1,
        fill_price=Decimal("100000000000000000001"), execution_cost=Decimal("0"),
        settlement_id="SETT-R1", settlement_session=X2_SESSION,
    )
    sell2 = PortfolioExecutionEvent(
        execution_id="SELL-R2", source_order_id="O-4", session=X_SESSION,
        asset_id=ASSET_B, side=ExecutionSide.SELL, quantity=1,
        fill_price=Decimal("1.0000000000000000001"), execution_cost=Decimal("0"),
        settlement_id="SETT-R2", settlement_session=X2_SESSION,
    )
    x_result = PortfolioTransitionEngine.transition(opened, X_SESSION, (sell1, sell2))
    final = x_result.resulting_state
    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy1, buy2), state_transition_result=t_result,
        authoritative_state=opened, ranking_snapshot=_ranking(T_SESSION),
    )
    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(opened),
        ordered_execution_events=(sell1, sell2), state_transition_result=x_result,
        authoritative_state=final, ranking_snapshot=_ranking(X_SESSION),
    )
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=initial, final_state=final,
        session_results=(t_session, x_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )
    snapshots = (
        _snap(T_SESSION, (
            _mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("1")),
            _mark(run_manifest, security_id=ASSET_B, session=T_SESSION, close=Decimal("1")),
        )),
        _snap(X_SESSION, ()),
    )
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
        results.append(rows[-1].cumulative_realized_pnl)
    assert results[0] == results[1] == results[2]
    assert len(results[0].as_tuple().digits) > 28


def test_unrealized_pnl_exact_across_precisions_multi_position(run_manifest) -> None:
    # Two open positions whose values (10^20 and 10^-19, each exact under
    # any precision since neither exceeds 28 significant digits) combine
    # to an unrealized-P&L sum needing far more than 28 digits once
    # aligned.
    buy1 = PortfolioExecutionEvent(
        execution_id="BUY-U1", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=1,
        fill_price=Decimal("1"), execution_cost=Decimal("0"),
    )
    buy2 = PortfolioExecutionEvent(
        execution_id="BUY-U2", source_order_id="O-2", session=T_SESSION,
        asset_id=ASSET_B, side=ExecutionSide.BUY, quantity=1,
        fill_price=Decimal("1"), execution_cost=Decimal("0"),
    )
    initial = PortfolioState(settled_cash=Decimal("1000"))
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy1, buy2))
    final = t_result.resulting_state
    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy1, buy2), state_transition_result=t_result,
        authoritative_state=final, ranking_snapshot=_ranking(T_SESSION),
    )
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=initial, final_state=final,
        session_results=(t_session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )
    snapshots = (
        _snap(T_SESSION, (
            _mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("100000000000000000000")),
            _mark(run_manifest, security_id=ASSET_B, session=T_SESSION, close=Decimal("0.0000000000000000001")),
        )),
    )
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
        results.append(rows[-1].unrealized_pnl)
    assert results[0] == results[1] == results[2]
    assert len(results[0].as_tuple().digits) > 28


def test_initial_unrealized_pnl_exact_across_precisions(run_manifest) -> None:
    initial_state = PortfolioState(
        settled_cash=Decimal("100000"),
        open_positions=(
            OpenPosition(
                asset_id=ASSET_A, quantity=1, entry_session=date(2026, 8, 1),
                entry_price=Decimal("1"), entry_execution_id="CARRIED-1",
                entry_execution_cost=Decimal("0"), cost_basis=Decimal("1"),
            ),
            OpenPosition(
                asset_id=ASSET_B, quantity=1, entry_session=date(2026, 8, 1),
                entry_price=Decimal("1"), entry_execution_id="CARRIED-2",
                entry_execution_cost=Decimal("0"), cost_basis=Decimal("1"),
            ),
        ),
        applied_events=(
            AppliedEventFingerprint(event_kind=PortfolioEventKind.EXECUTION, event_id="CARRIED-1", payload_sha256="a" * 64),
            AppliedEventFingerprint(event_kind=PortfolioEventKind.EXECUTION, event_id="CARRIED-2", payload_sha256="b" * 64),
        ),
        as_of_session=date(2026, 8, 1), state_version=2,
    )
    initial_snapshot = _snap(date(2026, 8, 1), (
        _mark(run_manifest, security_id=ASSET_A, session=date(2026, 8, 1), close=Decimal("100000000000000000000")),
        _mark(run_manifest, security_id=ASSET_B, session=date(2026, 8, 1), close=Decimal("0.0000000000000000001")),
    ))
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            valuation = _compute_initial_valuation(
                initial_state=initial_state, initial_snapshot=initial_snapshot
            )
        results.append(valuation.initial_unrealized_pnl)
    assert results[0] == results[1] == results[2]
    assert len(results[0].as_tuple().digits) > 28


def test_pending_value_exact_across_precisions() -> None:
    initial_state = PortfolioState(
        settled_cash=Decimal("1000"),
        pending_settlements=(
            PendingSettlement(
                settlement_id="S1", source_execution_id="E1", asset_id=ASSET_A,
                amount=Decimal("100000000000000000000"),
                trade_session=date(2026, 8, 1), settlement_session=date(2026, 8, 5),
            ),
            PendingSettlement(
                settlement_id="S2", source_execution_id="E2", asset_id=ASSET_B,
                amount=Decimal("0.0000000000000000001"),
                trade_session=date(2026, 8, 1), settlement_session=date(2026, 8, 5),
            ),
        ),
    )
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            valuation = _compute_initial_valuation(
                initial_state=initial_state, initial_snapshot=None
            )
        results.append(valuation.initial_equity)
    assert results[0] == results[1] == results[2]
    assert len(results[0].as_tuple().digits) > 28


def test_correct_data_does_not_fail_reconciliation_under_low_precision(run_manifest) -> None:
    # Regression for the confirmed pre-fix defect: legitimate, correctly
    # supplied data (two ordinary open positions with high-precision
    # marks) must not spuriously raise PNL_RECONCILIATION_MISMATCH merely
    # because the ambient Decimal context is the low pytest default.
    buy1 = PortfolioExecutionEvent(
        execution_id="BUY-OK1", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=1,
        fill_price=Decimal("100"), execution_cost=Decimal("0"),
    )
    buy2 = PortfolioExecutionEvent(
        execution_id="BUY-OK2", source_order_id="O-2", session=T_SESSION,
        asset_id=ASSET_B, side=ExecutionSide.BUY, quantity=1,
        fill_price=Decimal("50"), execution_cost=Decimal("0"),
    )
    initial = PortfolioState(settled_cash=Decimal("1000"))
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy1, buy2))
    final = t_result.resulting_state
    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy1, buy2), state_transition_result=t_result,
        authoritative_state=final, ranking_snapshot=_ranking(T_SESSION),
    )
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=initial, final_state=final,
        session_results=(t_session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )
    snapshots = (
        _snap(T_SESSION, (
            _mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("100.12345678901234567890123456789")),
            _mark(run_manifest, security_id=ASSET_B, session=T_SESSION, close=Decimal("50.98765432109876543210987654321")),
        )),
    )
    # Under the default (prec=28) ambient context this must not raise.
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    assert len(equity_rows) == 1


def test_deliberately_wrong_period_pnl_still_fails_closed_at_high_precision(
    run_manifest, monkeypatch
) -> None:
    from stock_swing_d1.backtest_results import valuation as valuation_module

    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )

    def _wrong_totals(_run_result):
        return {X_SESSION: Decimal("999")}

    monkeypatch.setattr(valuation_module, "_dividend_income_by_session", _wrong_totals)
    with localcontext() as ctx:
        ctx.prec = 60
        with pytest.raises(HistoricalBacktestResultValidationError, match="PNL_RECONCILIATION_MISMATCH"):
            project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)


# ---------------------------------------------------------------------------
# Second narrow audit follow-up: `Decimal(quantity) * mark.close` is itself
# an ambient-context-sensitive Decimal multiplication once `mark.close`
# carries enough significant digits -- `HistoricalBacktestValuationMark.
# close` enforces only `type is Decimal`, finite, and `> 0` (no scale/
# precision bound), so this is not a contrived input. Fixed via
# `exact_decimal_times_int` in both `_compute_initial_valuation` and
# `_compute_session_valuation_rows`.
# ---------------------------------------------------------------------------


def test_exact_decimal_times_int_matches_ordinary_multiplication_normally() -> None:
    from stock_swing_d1.backtest_results.valuation import exact_decimal_times_int

    assert exact_decimal_times_int(Decimal("100.50"), 3) == Decimal("100.50") * 3
    assert exact_decimal_times_int(Decimal("1"), 1) == Decimal("1")


def test_exact_decimal_times_int_zero_quantity_and_zero_value() -> None:
    from stock_swing_d1.backtest_results.valuation import exact_decimal_times_int

    assert exact_decimal_times_int(Decimal("100.50"), 0) == Decimal("0")
    assert exact_decimal_times_int(Decimal("0"), 5) == Decimal("0")


def test_exact_decimal_times_int_preserves_trailing_zero_scale() -> None:
    from stock_swing_d1.backtest_results.valuation import exact_decimal_times_int

    result = exact_decimal_times_int(Decimal("2.50"), 4)
    assert result == Decimal("10.00")
    assert result.as_tuple().exponent == Decimal("2.50").as_tuple().exponent


def test_exact_decimal_times_int_large_quantity_exact() -> None:
    from stock_swing_d1.backtest_results.valuation import exact_decimal_times_int

    assert exact_decimal_times_int(Decimal("0.1"), 9_999_999) == Decimal("0.1") * 9_999_999


def test_exact_decimal_times_int_does_not_mutate_ambient_context() -> None:
    from stock_swing_d1.backtest_results.valuation import exact_decimal_times_int

    with localcontext() as ctx:
        ctx.prec = 5
        exact_decimal_times_int(Decimal("123.456789012345678901234567890"), 7)
        assert ctx.prec == 5


def test_exact_decimal_times_int_high_precision_identical_across_ambient_context() -> None:
    from stock_swing_d1.backtest_results.valuation import exact_decimal_times_int

    close = Decimal("33.333333333333333333333333333333333")  # 35 sig figs
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            results.append(exact_decimal_times_int(close, 3))
    assert results[0] == results[1] == results[2]
    # Cross-checked against an explicit high-precision ordinary multiply
    # (never against ambient-context `*`, which would itself round).
    with localcontext() as ctx:
        ctx.prec = 200
        expected = close * 3
    assert results[0] == expected
    assert len(results[0].as_tuple().digits) > 28


def test_exact_decimal_times_int_no_float_conversion() -> None:
    import inspect

    from stock_swing_d1.backtest_results import valuation

    source = inspect.getsource(valuation.exact_decimal_times_int)
    assert "float(" not in source
    assert "Decimal(float" not in source


def test_high_significant_digit_mark_is_accepted_by_live_model(run_manifest) -> None:
    # Proves the "real stock prices have few decimals" claim is NOT an
    # enforced bound: the live HistoricalBacktestValuationMark model
    # accepts a 35-significant-digit close with no rejection.
    mark = _mark(
        run_manifest, security_id=ASSET_A, session=T_SESSION,
        close=Decimal("33.333333333333333333333333333333333"),
    )
    assert len(mark.close.as_tuple().digits) == 35


def test_market_value_unrealized_equity_period_pnl_exact_across_precisions(
    run_manifest,
) -> None:
    # Single open position, quantity=3, with a 35-significant-digit mark:
    # isolates the multiplication defect from any accumulation concern
    # (already separately audited/fixed). Pre-fix this diverged even at
    # the level of the integer part (100 vs 99.999...) between precision
    # 28 and 60/200.
    buy = PortfolioExecutionEvent(
        execution_id="BUY-MUL", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=3,
        fill_price=Decimal("1"), execution_cost=Decimal("0"),
    )
    initial = PortfolioState(settled_cash=Decimal("1000"))
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    final = t_result.resulting_state
    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy,), state_transition_result=t_result,
        authoritative_state=final, ranking_snapshot=_ranking(T_SESSION),
    )
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=initial, final_state=final,
        session_results=(t_session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )
    snapshots = (
        _snap(T_SESSION, (
            _mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("33.333333333333333333333333333333333")),
        )),
    )
    market_values, unrealized, equities, periods = [], [], [], []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
        market_values.append(rows[-1].open_position_market_value)
        unrealized.append(rows[-1].unrealized_pnl)
        equities.append(rows[-1].equity)
        periods.append(rows[-1].period_pnl)

    assert market_values[0] == market_values[1] == market_values[2]
    assert unrealized[0] == unrealized[1] == unrealized[2]
    assert equities[0] == equities[1] == equities[2]
    assert periods[0] == periods[1] == periods[2]
    assert len(market_values[0].as_tuple().digits) > 28


def test_initial_valuation_market_value_exact_across_precisions(run_manifest) -> None:
    # Same isolation, but for _compute_initial_valuation's own copy of
    # the multiplication (a carried-in position, valued only at run
    # start).
    initial_state = PortfolioState(
        settled_cash=Decimal("100000"),
        open_positions=(
            OpenPosition(
                asset_id=ASSET_A, quantity=3, entry_session=date(2026, 8, 1),
                entry_price=Decimal("1"), entry_execution_id="CARRIED-MUL",
                entry_execution_cost=Decimal("0"), cost_basis=Decimal("3"),
            ),
        ),
        applied_events=(
            AppliedEventFingerprint(event_kind=PortfolioEventKind.EXECUTION, event_id="CARRIED-MUL", payload_sha256="a" * 64),
        ),
        as_of_session=date(2026, 8, 1), state_version=1,
    )
    initial_snapshot = _snap(date(2026, 8, 1), (
        _mark(run_manifest, security_id=ASSET_A, session=date(2026, 8, 1), close=Decimal("33.333333333333333333333333333333333")),
    ))
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            valuation = _compute_initial_valuation(
                initial_state=initial_state, initial_snapshot=initial_snapshot
            )
        results.append(valuation.initial_unrealized_pnl)
    assert results[0] == results[1] == results[2]
    assert len(results[0].as_tuple().digits) > 28


def test_ordinary_price_fixtures_remain_unchanged_by_multiplication_fix(run_manifest) -> None:
    # Regression: normal, non-adversarial prices (the vast majority of
    # existing Slice-12 fixtures) must produce byte-identical results to
    # before this fix, since exact_decimal_times_int matches ordinary
    # Decimal multiplication whenever no rounding would occur.
    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    assert equity_rows[-1].cumulative_ordinary_dividend_income == Decimal("15")
    assert session_rows[-1].period_pnl == equity_rows[-1].period_pnl


# ---------------------------------------------------------------------------
# Open / partial trades (required tests 21-26; acceptance K.88-90)
# ---------------------------------------------------------------------------


def test_open_trade_dividend_contributes_with_no_closed_trade_record(run_manifest) -> None:
    run = _build_open_position_dividend_run(d_capitalspecial=Decimal("3"))
    assert project_closed_trades(run) == ()  # zero closed trades
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("100")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("100")),)),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = next(r for r in equity_rows if r.session == X_SESSION)
    # A closed-trade-only aggregate would be zero (no closed trades exist);
    # the authoritative portfolio-scope figure is not.
    assert x_row.cumulative_ordinary_dividend_income == Decimal("30")
    assert x_row.cumulative_ordinary_dividend_income != Decimal("0")


def test_carried_in_partial_dividend_contributes_while_trade_total_pnl_none(
    run_manifest,
) -> None:
    run = _build_carried_in_closed_trade_run(with_dividend=True)
    closed = project_closed_trades(run)
    assert len(closed) == 1
    assert closed[0].dividend_attribution_completeness == (
        DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
    )
    assert closed[0].trade_total_pnl is None
    snapshots = (
        _snap(date(2026, 8, 1), (_mark(run_manifest, security_id=ASSET_A, session=date(2026, 8, 1), close=Decimal("90")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("95")),)),
        _snap(X2_SESSION, ()),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    final_row = equity_rows[-1]
    assert final_row.cumulative_ordinary_dividend_income == closed[0].ordinary_dividend_income
    assert final_row.cumulative_ordinary_dividend_income > 0
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    _assert_identity_holds(run, session_rows, equity_rows, snapshots[0])


def test_real_applied_dividend_on_still_open_trade_reaches_portfolio_level(
    run_manifest,
) -> None:
    run = _build_real_applied_dividend_on_still_open_trade_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("55")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("62")),)),
        _snap(date(2026, 8, 20), (_mark(run_manifest, security_id=ASSET_A, session=date(2026, 8, 20), close=Decimal("63")),)),
        _snap(X3_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X3_SESSION, close=Decimal("64")),)),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    final_row = equity_rows[-1]
    assert final_row.cumulative_ordinary_dividend_income > 0
    closed = project_closed_trades(run)
    assert len(closed) == 1 and closed[0].ordinary_dividend_income == Decimal("0")
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    _assert_identity_holds(run, session_rows, equity_rows, None)


# ---------------------------------------------------------------------------
# Same-X / status (required tests 27-32; acceptance M.101-103)
# ---------------------------------------------------------------------------


def test_same_x_sell_reentry_dividend_counted_exactly_once(run_manifest) -> None:
    run = _build_sell_and_reentry_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("122")),)),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = equity_rows[-1]
    assert x_row.ordinary_dividend_income_this_session == Decimal("15")
    assert x_row.cumulative_ordinary_dividend_income == Decimal("15")
    closed = project_closed_trades(run)
    assert len(closed) == 1
    # No double count: closed trade's own income equals the full portfolio
    # figure exactly (not counted again through realized P&L).
    assert closed[0].ordinary_dividend_income == x_row.cumulative_ordinary_dividend_income
    assert closed[0].realized_pnl == closed[0].net_exit_proceeds - closed[0].entry_cost_basis
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    _assert_identity_holds(run, session_rows, equity_rows, None)


def test_replayed_dividend_adds_zero() -> None:
    run = _build_replayed_run()
    totals = _dividend_income_by_session(run)
    assert totals.get(X_SESSION, Decimal("0")) == Decimal("0")


def test_no_op_dividend_adds_zero() -> None:
    run = _build_buy_on_x_no_op_run()
    totals = _dividend_income_by_session(run)
    assert totals.get(X_SESSION, Decimal("0")) == Decimal("0")


def test_terminal_t_excludes_future_x_dividend(run_manifest) -> None:
    run = _build_terminal_t_only_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("100")),)),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    assert len(equity_rows) == 1
    assert equity_rows[0].ordinary_dividend_income_this_session == Decimal("0")
    assert equity_rows[0].cumulative_ordinary_dividend_income == Decimal("0")
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    _assert_identity_holds(run, session_rows, equity_rows, None)


# ---------------------------------------------------------------------------
# Legacy coverage (required tests 33-36)
# ---------------------------------------------------------------------------


def test_legacy_run_never_publishes_dividend_fields(run_manifest) -> None:
    run = _build_legacy_carried_in_closed_trade_run()
    assert run.dividend_run_evidence is None
    snapshots = (
        _snap(date(2026, 8, 1), (_mark(run_manifest, security_id=ASSET_A, session=date(2026, 8, 1), close=Decimal("90")),)),
        _snap(X_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=X_SESSION, close=Decimal("92")),)),
        _snap(X2_SESSION, ()),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    for row in (*equity_rows, *session_rows):
        assert row.ordinary_dividend_income_this_session is None
        assert row.cumulative_ordinary_dividend_income is None
        assert row.schema_version.endswith("v0.1")
    _assert_identity_holds(run, session_rows, equity_rows, snapshots[0])


def test_dividend_aware_zero_coverage_publishes_exact_zero(run_manifest) -> None:
    run = _build_zero_dividend_trade_run(dividend_aware=True)
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("100")),)),
        _snap(X_SESSION, ()),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    for row in equity_rows:
        assert row.cumulative_ordinary_dividend_income == Decimal("0")
        assert row.schema_version.endswith("v0.2")


# ---------------------------------------------------------------------------
# Model / validation (required tests 37-44)
# ---------------------------------------------------------------------------


def test_dividend_aware_row_missing_cumulative_field_fails() -> None:
    with pytest.raises(ValidationError):
        HistoricalBacktestSessionPnl(
            schema_version="historical_backtest_session_pnl.v0.2",
            session=T_SESSION, realized_pnl_this_session=Decimal("0"),
            cumulative_realized_pnl=Decimal("0"), unrealized_pnl=Decimal("0"),
            execution_cost_this_session=Decimal("0"), cumulative_execution_cost=Decimal("0"),
            period_pnl=Decimal("0"), ordinary_dividend_income_this_session=Decimal("5"),
            cumulative_ordinary_dividend_income=None,
        )


def test_legacy_row_with_dividend_field_fails() -> None:
    with pytest.raises(ValidationError):
        HistoricalBacktestEquityRow(
            schema_version="historical_backtest_equity_row.v0.1",
            session=T_SESSION, state_hash="a" * 64, settled_cash=Decimal("100"),
            pending_receivable_value=Decimal("0"), open_position_market_value=Decimal("0"),
            equity=Decimal("100"), realized_pnl_this_session=Decimal("0"),
            cumulative_realized_pnl=Decimal("0"), unrealized_pnl=Decimal("0"),
            execution_cost_this_session=Decimal("0"), cumulative_execution_cost=Decimal("0"),
            period_pnl=Decimal("0"), ordinary_dividend_income_this_session=Decimal("0"),
            cumulative_ordinary_dividend_income=Decimal("0"),
        )


def test_cumulative_less_than_this_session_fails() -> None:
    with pytest.raises(ValidationError):
        HistoricalBacktestSessionPnl(
            schema_version="historical_backtest_session_pnl.v0.2",
            session=T_SESSION, realized_pnl_this_session=Decimal("0"),
            cumulative_realized_pnl=Decimal("0"), unrealized_pnl=Decimal("0"),
            execution_cost_this_session=Decimal("0"), cumulative_execution_cost=Decimal("0"),
            period_pnl=Decimal("0"), ordinary_dividend_income_this_session=Decimal("10"),
            cumulative_ordinary_dividend_income=Decimal("5"),
        )


def test_correct_dividend_identity_passes() -> None:
    row = HistoricalBacktestSessionPnl(
        schema_version="historical_backtest_session_pnl.v0.2",
        session=T_SESSION, realized_pnl_this_session=Decimal("0"),
        cumulative_realized_pnl=Decimal("0"), unrealized_pnl=Decimal("0"),
        execution_cost_this_session=Decimal("0"), cumulative_execution_cost=Decimal("0"),
        period_pnl=Decimal("10"), ordinary_dividend_income_this_session=Decimal("10"),
        cumulative_ordinary_dividend_income=Decimal("10"),
    )
    assert row.cumulative_ordinary_dividend_income == Decimal("10")


def test_wrong_period_pnl_fails_reconciliation(run_manifest, monkeypatch) -> None:
    from stock_swing_d1.backtest_results import valuation as valuation_module

    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )

    def _wrong_totals(_run_result):
        return {X_SESSION: Decimal("999")}

    monkeypatch.setattr(valuation_module, "_dividend_income_by_session", _wrong_totals)
    with pytest.raises(HistoricalBacktestResultValidationError, match="PNL_RECONCILIATION_MISMATCH"):
        project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)


def test_double_counted_dividend_in_realized_pnl_is_detected(run_manifest, monkeypatch) -> None:
    from stock_swing_d1.backtest_results import valuation as valuation_module

    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    real_project_closed_trades = valuation_module.project_closed_trades

    def _inflated_realized(run_result):
        trades = real_project_closed_trades(run_result)
        return tuple(
            trade.model_copy(
                update={
                    "realized_pnl": add_exact_decimal(
                        trade.realized_pnl, trade.ordinary_dividend_income
                    )
                }
            )
            for trade in trades
        )

    monkeypatch.setattr(valuation_module, "project_closed_trades", _inflated_realized)
    with pytest.raises(HistoricalBacktestResultValidationError, match="PNL_RECONCILIATION_MISMATCH"):
        project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)


def test_initial_unrealized_pnl_semantics_unchanged(run_manifest) -> None:
    run = _build_carried_in_open_with_dividend_run()
    snapshot = _snap(date(2026, 8, 1), (_mark(run_manifest, security_id=ASSET_A, session=date(2026, 8, 1), close=Decimal("95")),))
    initial_valuation = _compute_initial_valuation(
        initial_state=run.initial_state, initial_snapshot=snapshot
    )
    assert initial_valuation.initial_unrealized_pnl == Decimal("10") * Decimal("95") - Decimal("901")


# ---------------------------------------------------------------------------
# Persistence / versioning (required tests 45-53)
# ---------------------------------------------------------------------------


def test_session_pnl_schema_versions_are_v0_1_and_v0_2() -> None:
    import typing

    annotation = HistoricalBacktestSessionPnl.model_fields["schema_version"].annotation
    assert set(typing.get_args(annotation)) == {
        "historical_backtest_session_pnl.v0.1",
        "historical_backtest_session_pnl.v0.2",
    }


def test_equity_row_schema_versions_are_v0_1_and_v0_2() -> None:
    import typing

    annotation = HistoricalBacktestEquityRow.model_fields["schema_version"].annotation
    assert set(typing.get_args(annotation)) == {
        "historical_backtest_equity_row.v0.1",
        "historical_backtest_equity_row.v0.2",
    }


def test_bundle_schema_version_is_v0_6() -> None:
    assert BUNDLE_SCHEMA_VERSION == "historical_backtest_result_bundle.v0.6"


def test_session_dividend_fields_round_trip() -> None:
    row = HistoricalBacktestSessionPnl(
        schema_version="historical_backtest_session_pnl.v0.2",
        session=T_SESSION, realized_pnl_this_session=Decimal("5"),
        cumulative_realized_pnl=Decimal("5"), unrealized_pnl=Decimal("0"),
        execution_cost_this_session=Decimal("0"), cumulative_execution_cost=Decimal("0"),
        period_pnl=Decimal("15"), ordinary_dividend_income_this_session=Decimal("10"),
        cumulative_ordinary_dividend_income=Decimal("10"),
    )
    encoded = encode_model_row(row, 0)
    decoded = decode_model_row(HistoricalBacktestSessionPnl, {k: v for k, v in encoded.items() if k != "ordinal"})
    assert decoded == row
    assert decoded.ordinary_dividend_income_this_session == Decimal("10")
    assert decoded.cumulative_ordinary_dividend_income == Decimal("10")


def test_legacy_session_pnl_none_dividend_fields_round_trip() -> None:
    row = HistoricalBacktestSessionPnl(
        schema_version="historical_backtest_session_pnl.v0.1",
        session=T_SESSION, realized_pnl_this_session=Decimal("5"),
        cumulative_realized_pnl=Decimal("5"), unrealized_pnl=Decimal("0"),
        execution_cost_this_session=Decimal("0"), cumulative_execution_cost=Decimal("0"),
        period_pnl=Decimal("5"),
    )
    encoded = encode_model_row(row, 0)
    decoded = decode_model_row(HistoricalBacktestSessionPnl, {k: v for k, v in encoded.items() if k != "ordinal"})
    assert decoded == row
    assert decoded.ordinary_dividend_income_this_session is None
    assert decoded.cumulative_ordinary_dividend_income is None


def test_summary_dividend_total_round_trips_via_service_and_persistence(
    run_manifest, tmp_path
) -> None:
    from stock_swing_d1.backtest_results import HistoricalBacktestResultService
    from stock_swing_d1.backtest_results.persistence import (
        HistoricalBacktestResultPersistence,
    )

    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    assert result.summary.ordinary_dividend_income_total == Decimal("15")

    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(result=result, destination=destination)
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert read_back.summary.ordinary_dividend_income_total == Decimal("15")
    assert read_back == result


def test_zero_session_dividend_aware_summary_is_exact_zero(run_manifest) -> None:
    from stock_swing_d1.backtest_results import HistoricalBacktestResultService

    state = PortfolioState(settled_cash=Decimal("777"))
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=()),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=state, final_state=state, session_results=(),
        initial_state_fingerprint=hash_portfolio_state(state),
        final_state_fingerprint=hash_portfolio_state(state),
    )
    assert project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=()) == ()
    assert project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=()) == ()

    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=()
    )
    assert result.summary.ordinary_dividend_income_total == Decimal("0")
    assert result.summary.processed_session_count == 0


def test_legacy_summary_dividend_total_is_none_end_to_end(run_manifest, tmp_path) -> None:
    from stock_swing_d1.backtest_results import HistoricalBacktestResultService
    from stock_swing_d1.backtest_results.persistence import (
        HistoricalBacktestResultPersistence,
    )
    from tests.backtest_results.test_phase15d4_persistence import _closed_trade_run

    run = _closed_trade_run()
    assert run.dividend_run_evidence is None
    from tests.backtest_results.test_phase15d4_persistence import E1, E2

    mark = HistoricalBacktestValuationMark(
        security_id="NORGATE:9001", session=E1, close=Decimal("52"),
        source_artifact_ref=run_manifest.market_data_artifact_ref,
    )
    snapshots = (
        build_valuation_snapshot(session=E1, marks=(mark,)),
        build_valuation_snapshot(session=E2, marks=()),
    )
    result = HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )
    assert result.summary.ordinary_dividend_income_total is None

    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(result=result, destination=destination)
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert read_back.summary.ordinary_dividend_income_total is None


def test_old_bundle_version_rejected_after_v0_6_bump(tmp_path, run_manifest, content_fingerprints) -> None:
    from stock_swing_d1.backtest_results.errors import HistoricalBacktestPersistenceError
    from stock_swing_d1.backtest_results.persistence import (
        HistoricalBacktestResultPersistence,
    )
    from stock_swing_d1.backtest_results import persistence_schema as schema
    import json

    manifest = schema.build_bundle_manifest(
        result_schema_version="historical_backtest_audit_result.v0.2",
        run_configuration_fingerprint="a" * 64, source_run_fingerprint="b" * 64,
        initial_state_fingerprint="c" * 64, final_state_fingerprint="d" * 64,
        result_fingerprint="e" * 64, content_fingerprints=content_fingerprints,
        row_counts={}, payload_sha256={},
    )
    raw = json.loads(schema.encode_bundle_manifest(manifest).decode("utf-8"))
    raw["persistence_schema_version"] = "historical_backtest_result_bundle.v0.5"
    tampered_bytes = json.dumps(raw).encode("utf-8")
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        schema.decode_bundle_manifest(tampered_bytes)
    assert excinfo.value.code == "UNSUPPORTED_SCHEMA_VERSION"


# ---------------------------------------------------------------------------
# Hashing (required tests 54-60)
# ---------------------------------------------------------------------------


def test_content_fingerprint_deterministic_for_dividend_pnl(run_manifest) -> None:
    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    first = compute_content_fingerprint(artifact_name="equity_curve", rows_or_value=equity_rows)
    second = compute_content_fingerprint(artifact_name="equity_curve", rows_or_value=equity_rows)
    assert first == second


def test_content_fingerprint_changes_when_only_session_dividend_income_changes(run_manifest) -> None:
    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    tampered = tuple(
        row.model_copy(update={"ordinary_dividend_income_this_session": Decimal("999")})
        for row in equity_rows
    )
    original_fp = compute_content_fingerprint(artifact_name="equity_curve", rows_or_value=equity_rows)
    tampered_fp = compute_content_fingerprint(artifact_name="equity_curve", rows_or_value=tampered)
    assert original_fp != tampered_fp


def test_content_fingerprint_changes_when_only_cumulative_dividend_income_changes(run_manifest) -> None:
    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    session_rows = project_session_pnl(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    tampered = tuple(
        row.model_copy(update={"cumulative_ordinary_dividend_income": Decimal("999")})
        for row in session_rows
    )
    original_fp = compute_content_fingerprint(artifact_name="session_pnl", rows_or_value=session_rows)
    tampered_fp = compute_content_fingerprint(artifact_name="session_pnl", rows_or_value=tampered)
    assert original_fp != tampered_fp


def test_content_fingerprint_changes_when_only_period_pnl_changes(run_manifest) -> None:
    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    tampered = tuple(
        row.model_copy(update={"period_pnl": row.period_pnl + Decimal("1")})
        for row in equity_rows
    )
    original_fp = compute_content_fingerprint(artifact_name="equity_curve", rows_or_value=equity_rows)
    tampered_fp = compute_content_fingerprint(artifact_name="equity_curve", rows_or_value=tampered)
    assert original_fp != tampered_fp


def test_attribution_fingerprint_unrelated_to_portfolio_pnl_fields() -> None:
    import inspect

    from stock_swing_d1.backtest_results.hashing import (
        compute_dividend_attribution_fingerprint,
    )

    source = inspect.getsource(compute_dividend_attribution_fingerprint)
    for forbidden in (
        "ordinary_dividend_income_this_session",
        "cumulative_ordinary_dividend_income",
        "period_pnl",
    ):
        assert forbidden not in source


def test_source_run_hash_domain_is_the_frozen_5cb_v0_3() -> None:
    # Frozen Task 5C-B baseline: v0.3 (nested session results gained
    # entry_session_protective_decisions); Task 5C-C adds no further bump.
    assert SOURCE_RUN_HASH_DOMAIN == "historical_backtest_source_run.v0.3"


# ---------------------------------------------------------------------------
# Regression (required tests 61-72)
# ---------------------------------------------------------------------------


def test_realized_and_unrealized_pnl_unchanged_by_dividend_integration(run_manifest) -> None:
    run = _build_sell_on_x_run()
    snapshots = (
        _snap(T_SESSION, (_mark(run_manifest, security_id=ASSET_A, session=T_SESSION, close=Decimal("105")),)),
        _snap(X_SESSION, ()),
    )
    equity_rows = project_equity_curve(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    x_row = equity_rows[-1]
    closed = project_closed_trades(run)[0]
    assert x_row.cumulative_realized_pnl == closed.realized_pnl
    assert x_row.settled_cash == run.final_state.settled_cash
    assert closed.entry_cost_basis == (
        Decimal(closed.quantity) * closed.entry_fill_price + closed.entry_execution_cost
    )
    assert closed.net_exit_proceeds == closed.gross_exit_proceeds - closed.exit_execution_cost


def test_trade_total_pnl_unaffected_by_portfolio_integration(run_manifest) -> None:
    run = _build_sell_on_x_run()
    closed = project_closed_trades(run)[0]
    assert closed.trade_total_pnl == closed.realized_pnl + closed.ordinary_dividend_income
