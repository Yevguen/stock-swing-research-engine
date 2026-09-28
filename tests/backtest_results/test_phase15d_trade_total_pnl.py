"""Slice 11 acceptance tests: Phase 15D closed-trade `trade_total_pnl`.

Covers the frozen Ordinary Dividend Amendment v0.7's OD-16.4/OD-16.5a/
OD-21.7 authoritative total-trade-P&L contract:

- `trade_total_pnl = realized_pnl + ordinary_dividend_income` (exact,
  ambient-Decimal-context-independent) for a `COMPLETE_TRADE_LIFETIME`
  closed trade (acceptance case I.75);
- `trade_total_pnl is None` for a `PARTIAL_PRE_RUN_UNKNOWN` (carried-in)
  closed trade, even though its exact in-run dividend income is known
  (acceptance case J.79);
- a legacy (`v0.1`) or grouping-only (`v0.2`) closed trade never
  publishes `trade_total_pnl` (no inference, no backfill).

Portfolio-scope/session/equity dividend P&L and Phase 16B consumption
remain explicitly deferred to a later slice; this file adds no such
field or assertion.
"""

from __future__ import annotations

import json
import typing
from datetime import date
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from stock_swing_d1.backtest_results import (
    DividendAttributionCompleteness,
    HistoricalBacktestPersistenceError,
    HistoricalClosedTradeRecord,
    compute_content_fingerprint,
    compute_dividend_attribution_fingerprint,
    project_closed_trades,
)
from stock_swing_d1.backtest_results.hashing import SOURCE_RUN_HASH_DOMAIN
from stock_swing_d1.backtest_results.persistence_schema import (
    BUNDLE_SCHEMA_VERSION,
    build_bundle_manifest,
    decode_bundle_manifest,
    decode_trade_row,
    encode_bundle_manifest,
    encode_trade_row,
)
from stock_swing_d1.backtester.models import (
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionResult,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import add_exact_decimal
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
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
    T_SESSION,
    X_SESSION,
    _ranking,
    make_run_evidence,
    make_session_evidence,
)
from tests.backtest_results.test_phase15d_dividend_trade_grouping import (
    X2_SESSION,
    _build_carried_in_closed_trade_run,
    _build_legacy_carried_in_closed_trade_run,
    _build_multi_session_dividend_run,
    _build_real_applied_dividend_on_still_open_trade_run,
    _build_zero_dividend_trade_run,
    _closed_trade_for,
    _high_precision_row,
    _make_custom_evidence,
)


def _build_custom_pnl_dividend_run(
    *,
    quantity: int = 10,
    entry_fill_price: Decimal,
    exit_fill_price: Decimal,
    entry_cost: Decimal = Decimal("1"),
    exit_cost: Decimal = Decimal("1"),
    dividend_per_share: Decimal,
    trade_id: str = "BUY-PNL",
) -> HistoricalBacktestRunResult:
    """One BUY-then-SELL-on-X trade with a caller-controlled exact
    realized P&L and a caller-controlled exact per-share ordinary
    dividend. `_make_custom_evidence`'s Gate3 normalization inputs are
    chosen as `dividend_per_share / 1 / 1` so `D_H == dividend_per_share`
    exactly, letting each test construct a specific total-P&L scenario
    without hand-deriving Gate3 arithmetic.
    """

    initial = PortfolioState(settled_cash=Decimal("1000000"))
    buy = PortfolioExecutionEvent(
        execution_id=trade_id, source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=quantity,
        fill_price=entry_fill_price, execution_cost=entry_cost,
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    opened = t_result.resulting_state

    sell = PortfolioExecutionEvent(
        execution_id=f"{trade_id}-EXIT", source_order_id="O-2", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=quantity,
        fill_price=exit_fill_price, execution_cost=exit_cost,
        settlement_id=f"{trade_id}-SETTLE", settlement_session=X2_SESSION,
    )
    evidence = _make_custom_evidence(
        event_id=f"D:{trade_id}",
        d_capitalspecial=dividend_per_share,
        unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("1"),
    )
    x_result = PortfolioTransitionEngine.transition(
        opened, X_SESSION, (sell,), (evidence,)
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
        ordered_execution_events=(sell,),
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


def _minimal_complete_trade(
    *,
    realized_pnl: Decimal = Decimal("0"),
    ordinary_dividend_income: Decimal = Decimal("10"),
    ordinary_dividend_event_count: int = 1,
    dividend_attribution_completeness: DividendAttributionCompleteness = (
        DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
    ),
    trade_total_pnl: Decimal | None = Decimal("10"),
    carried_in: bool = False,
    schema_version: str = "historical_closed_trade.v0.3",
    **more_overrides: object,
) -> HistoricalClosedTradeRecord:
    """A single-share trade whose `realized_pnl` is caller-controlled via
    `exit_fill_price` (quantity=1, zero execution costs, entry_cost_basis
    fixed at 1000), so callers can freely request positive, zero, or
    negative realized P&L without violating the existing entry/exit
    reporting-identity validators.
    """

    entry_cost_basis = Decimal("1000")
    net_exit_proceeds = entry_cost_basis + realized_pnl
    values: dict[str, object] = dict(
        trade_id="X", security_id=ASSET_A, quantity=1, carried_in=carried_in,
        entry_execution_id="X", entry_session=T_SESSION,
        entry_fill_price=entry_cost_basis, entry_execution_cost=Decimal("0"),
        entry_cost_basis=entry_cost_basis, exit_execution_id="Y",
        exit_session=X_SESSION, exit_fill_price=net_exit_proceeds,
        exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
        gross_exit_proceeds=net_exit_proceeds, net_exit_proceeds=net_exit_proceeds,
        realized_pnl=realized_pnl,
        ordinary_dividend_income=ordinary_dividend_income,
        ordinary_dividend_event_count=ordinary_dividend_event_count,
        dividend_attribution_completeness=dividend_attribution_completeness,
        ordinary_dividend_attribution_fingerprint="f" * 64,
        schema_version=schema_version,
        trade_total_pnl=trade_total_pnl,
    )
    values.update(more_overrides)
    return HistoricalClosedTradeRecord(**values)


# ---------------------------------------------------------------------------
# Core COMPLETE-trade identity (acceptance case I.75; required tests 1-9)
# ---------------------------------------------------------------------------


def test_complete_zero_dividend_trade_total_equals_realized() -> None:
    run = _build_zero_dividend_trade_run(dividend_aware=True)
    trade = _closed_trade_for(run, "BUY-ZERO")
    assert trade.ordinary_dividend_income == Decimal("0")
    assert trade.trade_total_pnl == trade.realized_pnl


def test_complete_one_dividend_trade_total_is_exact_sum() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    assert trade.ordinary_dividend_income > 0
    assert trade.trade_total_pnl == trade.realized_pnl + trade.ordinary_dividend_income


def test_complete_multiple_dividends_trade_total_is_exact_sum_once() -> None:
    run = _build_multi_session_dividend_run()
    trade = _closed_trade_for(run, "BUY-1")
    assert trade.ordinary_dividend_event_count == 2
    assert trade.trade_total_pnl == trade.realized_pnl + trade.ordinary_dividend_income


def test_positive_realized_plus_dividend() -> None:
    run = _build_custom_pnl_dividend_run(
        entry_fill_price=Decimal("100"), exit_fill_price=Decimal("120"),
        dividend_per_share=Decimal("1"), trade_id="BUY-POS",
    )
    trade = _closed_trade_for(run, "BUY-POS")
    assert trade.realized_pnl == Decimal("198")
    assert trade.ordinary_dividend_income == Decimal("10")
    assert trade.trade_total_pnl == Decimal("208")


def test_negative_realized_plus_dividend_remains_negative() -> None:
    run = _build_custom_pnl_dividend_run(
        entry_fill_price=Decimal("100"), exit_fill_price=Decimal("95"),
        dividend_per_share=Decimal("1"), trade_id="BUY-NEG1",
    )
    trade = _closed_trade_for(run, "BUY-NEG1")
    assert trade.realized_pnl == Decimal("-52")
    assert trade.trade_total_pnl == Decimal("-42")
    assert trade.trade_total_pnl < 0


def test_negative_realized_plus_dividend_becomes_zero() -> None:
    run = _build_custom_pnl_dividend_run(
        entry_fill_price=Decimal("100"), exit_fill_price=Decimal("95"),
        dividend_per_share=Decimal("5.2"), trade_id="BUY-NEG2",
    )
    trade = _closed_trade_for(run, "BUY-NEG2")
    assert trade.realized_pnl == Decimal("-52")
    assert trade.trade_total_pnl == Decimal("0")


def test_negative_realized_plus_dividend_becomes_positive() -> None:
    run = _build_custom_pnl_dividend_run(
        entry_fill_price=Decimal("100"), exit_fill_price=Decimal("95"),
        dividend_per_share=Decimal("10"), trade_id="BUY-NEG3",
    )
    trade = _closed_trade_for(run, "BUY-NEG3")
    assert trade.realized_pnl == Decimal("-52")
    assert trade.trade_total_pnl == Decimal("48")
    assert trade.trade_total_pnl > 0


def test_zero_realized_plus_positive_dividend() -> None:
    run = _build_custom_pnl_dividend_run(
        entry_fill_price=Decimal("100"), exit_fill_price=Decimal("100.2"),
        dividend_per_share=Decimal("3"), trade_id="BUY-ZEROR",
    )
    trade = _closed_trade_for(run, "BUY-ZEROR")
    assert trade.realized_pnl == Decimal("0")
    assert trade.trade_total_pnl == Decimal("30")


def test_no_dividend_double_count() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    assert trade.realized_pnl == trade.net_exit_proceeds - trade.entry_cost_basis
    assert trade.trade_total_pnl != trade.realized_pnl
    assert trade.trade_total_pnl - trade.ordinary_dividend_income == trade.realized_pnl


# ---------------------------------------------------------------------------
# PARTIAL / legacy semantics (acceptance case J.79; required tests 10-14)
# ---------------------------------------------------------------------------


def test_carried_in_partial_with_dividend_has_no_total() -> None:
    run = _build_carried_in_closed_trade_run(with_dividend=True)
    trade = _closed_trade_for(run, "CARRIED-BUY-1")
    assert trade.dividend_attribution_completeness == (
        DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
    )
    assert trade.ordinary_dividend_income > 0
    assert trade.trade_total_pnl is None


def test_carried_in_partial_with_zero_in_run_dividend_still_has_no_total() -> None:
    run = _build_carried_in_closed_trade_run(with_dividend=False)
    trade = _closed_trade_for(run, "CARRIED-BUY-1")
    assert trade.ordinary_dividend_income == Decimal("0")
    assert trade.trade_total_pnl is None


def test_legacy_closed_trade_has_no_total() -> None:
    run = _build_legacy_carried_in_closed_trade_run()
    trade = _closed_trade_for(run, "CARRIED-BUY-1")
    assert trade.schema_version == "historical_closed_trade.v0.1"
    assert trade.trade_total_pnl is None


def test_legacy_zero_row_run_does_not_fabricate_realized_plus_zero() -> None:
    run = _build_zero_dividend_trade_run(dividend_aware=False)
    trade = _closed_trade_for(run, "BUY-ZERO")
    assert trade.schema_version == "historical_closed_trade.v0.1"
    # Absence of dividend evidence is not proof of zero dividends
    # (OD-7.1): this must not equal realized_pnl + 0 as a fabricated
    # dividend-aware total, it must be None outright.
    assert trade.trade_total_pnl is None


def test_v0_2_grouping_only_trade_is_not_silently_backfilled() -> None:
    trade = HistoricalClosedTradeRecord(
        trade_id="X", security_id=ASSET_A, quantity=1, carried_in=False,
        entry_execution_id="X", entry_session=T_SESSION,
        entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
        entry_cost_basis=Decimal("1"), exit_execution_id="Y",
        exit_session=X_SESSION, exit_fill_price=Decimal("1"),
        exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
        gross_exit_proceeds=Decimal("1"), net_exit_proceeds=Decimal("1"),
        realized_pnl=Decimal("0"), ordinary_dividend_income=Decimal("0"),
        ordinary_dividend_event_count=0,
        dividend_attribution_completeness=(
            DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
        ),
        ordinary_dividend_attribution_fingerprint="f" * 64,
        schema_version="historical_closed_trade.v0.2",
    )
    assert trade.trade_total_pnl is None


# ---------------------------------------------------------------------------
# Decimal determinism (required tests 15-22)
# ---------------------------------------------------------------------------


def test_high_precision_total_identical_at_precisions_28_60_200() -> None:
    row, _q_t, _d_h = _high_precision_row(trade_id="TP-HIGH")
    realized_pnl = Decimal("123.45")
    expected_total = add_exact_decimal(realized_pnl, row.settled_cash_delta)
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            trade = _minimal_complete_trade(
                trade_id="TP-HIGH", entry_execution_id="TP-HIGH",
                realized_pnl=realized_pnl,
                ordinary_dividend_income=row.settled_cash_delta,
                trade_total_pnl=expected_total,
            )
        assert trade.trade_total_pnl == expected_total
        assert type(trade.trade_total_pnl) is Decimal


def test_naive_addition_would_have_rounded_the_total() -> None:
    row, _q_t, _d_h = _high_precision_row(trade_id="TP-NAIVE")
    realized_pnl = Decimal("123.45")
    exact_expected = add_exact_decimal(realized_pnl, row.settled_cash_delta)
    with localcontext() as ctx:
        ctx.prec = 28
        naive = realized_pnl + row.settled_cash_delta
    assert naive != exact_expected


def test_validator_rejects_naively_rounded_total_under_low_ambient_precision() -> None:
    row, _q_t, _d_h = _high_precision_row(trade_id="TP-REJECT")
    realized_pnl = Decimal("123.45")
    with localcontext() as ctx:
        ctx.prec = 28
        naive = realized_pnl + row.settled_cash_delta
    with pytest.raises(ValidationError):
        _minimal_complete_trade(
            trade_id="TP-REJECT", entry_execution_id="TP-REJECT",
            realized_pnl=realized_pnl,
            ordinary_dividend_income=row.settled_cash_delta,
            trade_total_pnl=naive,
        )


def test_trade_total_pnl_rejects_binary_float() -> None:
    with pytest.raises(ValidationError):
        _minimal_complete_trade(trade_total_pnl=10.0)


def test_trade_total_pnl_rejects_non_finite_decimal() -> None:
    with pytest.raises(ValidationError):
        _minimal_complete_trade(trade_total_pnl=Decimal("NaN"))


# ---------------------------------------------------------------------------
# Same-X re-entry / open-trade non-publication (required tests 23-26)
# ---------------------------------------------------------------------------


def test_same_x_sell_reentry_old_trade_gets_total_new_trade_does_not_exist() -> None:
    run = _build_sell_and_reentry_run()
    closed = project_closed_trades(run)
    assert len(closed) == 1
    old_trade = closed[0]
    assert old_trade.trade_id == "BUY-1"
    assert old_trade.trade_total_pnl == (
        old_trade.realized_pnl + old_trade.ordinary_dividend_income
    )
    assert all(trade.trade_id != "BUY-2" for trade in closed)


def test_real_applied_dividend_on_open_trade_publishes_no_closed_total() -> None:
    run = _build_real_applied_dividend_on_still_open_trade_run()
    closed = project_closed_trades(run)
    assert len(closed) == 1
    old_trade = closed[0]
    assert old_trade.trade_id == "BUY-OLD"
    # The real dividend went to still-open BUY-NEW, never stolen by the
    # earlier same-security closed trade, so BUY-OLD's own total is just
    # its own (zero-dividend) realized P&L.
    assert old_trade.ordinary_dividend_income == Decimal("0")
    assert old_trade.trade_total_pnl == old_trade.realized_pnl
    # BUY-NEW has no closed-trade record at all, so no trade_total_pnl is
    # ever published for it.
    assert all(trade.trade_id != "BUY-NEW" for trade in closed)


def test_buy_on_x_no_op_trade_never_closes_so_publishes_no_total() -> None:
    run = _build_buy_on_x_no_op_run()
    assert project_closed_trades(run) == ()


def test_replayed_run_has_zero_closed_trades_and_does_not_crash() -> None:
    run = _build_replayed_run()
    assert project_closed_trades(run) == ()


# ---------------------------------------------------------------------------
# Model validation (required tests 27-35)
# ---------------------------------------------------------------------------


def test_complete_with_none_total_fails() -> None:
    with pytest.raises(ValidationError):
        _minimal_complete_trade(trade_total_pnl=None)


def test_complete_with_wrong_total_fails() -> None:
    with pytest.raises(ValidationError):
        _minimal_complete_trade(trade_total_pnl=Decimal("999"))


def test_complete_with_exact_total_passes() -> None:
    trade = _minimal_complete_trade(
        realized_pnl=Decimal("5"), ordinary_dividend_income=Decimal("10"),
        trade_total_pnl=Decimal("15"),
    )
    assert trade.trade_total_pnl == Decimal("15")


def test_partial_with_non_none_total_fails() -> None:
    with pytest.raises(ValidationError):
        _minimal_complete_trade(
            carried_in=True,
            dividend_attribution_completeness=(
                DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
            ),
            trade_total_pnl=Decimal("10"),
        )


def test_partial_with_none_total_passes() -> None:
    trade = _minimal_complete_trade(
        carried_in=True,
        dividend_attribution_completeness=(
            DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
        ),
        trade_total_pnl=None,
    )
    assert trade.trade_total_pnl is None


def test_legacy_v0_1_with_non_none_total_fails() -> None:
    with pytest.raises(ValidationError):
        HistoricalClosedTradeRecord(
            trade_id="X", security_id=ASSET_A, quantity=1, carried_in=False,
            entry_execution_id="X", entry_session=T_SESSION,
            entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
            entry_cost_basis=Decimal("1"), exit_execution_id="Y",
            exit_session=X_SESSION, exit_fill_price=Decimal("1"),
            exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
            gross_exit_proceeds=Decimal("1"), net_exit_proceeds=Decimal("1"),
            realized_pnl=Decimal("0"),
            schema_version="historical_closed_trade.v0.1",
            trade_total_pnl=Decimal("5"),
        )


def test_v0_2_with_non_none_total_fails() -> None:
    with pytest.raises(ValidationError):
        HistoricalClosedTradeRecord(
            trade_id="X", security_id=ASSET_A, quantity=1, carried_in=False,
            entry_execution_id="X", entry_session=T_SESSION,
            entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
            entry_cost_basis=Decimal("1"), exit_execution_id="Y",
            exit_session=X_SESSION, exit_fill_price=Decimal("1"),
            exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
            gross_exit_proceeds=Decimal("1"), net_exit_proceeds=Decimal("1"),
            realized_pnl=Decimal("0"), ordinary_dividend_income=Decimal("0"),
            ordinary_dividend_event_count=0,
            dividend_attribution_completeness=(
                DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
            ),
            ordinary_dividend_attribution_fingerprint="f" * 64,
            schema_version="historical_closed_trade.v0.2",
            trade_total_pnl=Decimal("0"),
        )


def test_v0_3_with_missing_grouping_fields_fails() -> None:
    with pytest.raises(ValidationError):
        HistoricalClosedTradeRecord(
            trade_id="X", security_id=ASSET_A, quantity=1, carried_in=False,
            entry_execution_id="X", entry_session=T_SESSION,
            entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
            entry_cost_basis=Decimal("1"), exit_execution_id="Y",
            exit_session=X_SESSION, exit_fill_price=Decimal("1"),
            exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
            gross_exit_proceeds=Decimal("1"), net_exit_proceeds=Decimal("1"),
            realized_pnl=Decimal("0"),
            schema_version="historical_closed_trade.v0.3",
            trade_total_pnl=Decimal("0"),
        )


def test_schema_version_rejects_unknown_future_version() -> None:
    with pytest.raises(ValidationError):
        _minimal_complete_trade(schema_version="historical_closed_trade.v0.4")


def test_schema_version_literal_set_is_v0_1_v0_2_v0_3() -> None:
    annotation = HistoricalClosedTradeRecord.model_fields["schema_version"].annotation
    assert set(typing.get_args(annotation)) == {
        "historical_closed_trade.v0.1",
        "historical_closed_trade.v0.2",
        "historical_closed_trade.v0.3",
    }


def test_trade_total_pnl_field_exists_and_is_optional() -> None:
    assert "trade_total_pnl" in HistoricalClosedTradeRecord.model_fields
    field = HistoricalClosedTradeRecord.model_fields["trade_total_pnl"]
    assert field.default is None


# ---------------------------------------------------------------------------
# Persistence / versioning (required tests 36-44)
# ---------------------------------------------------------------------------


def test_bundle_schema_version_is_v0_6() -> None:
    # Slice 12 bumped the bundle version again (v0.5 -> v0.6) for its own
    # portfolio-scope dividend P&L columns; see
    # test_phase15d_portfolio_dividend_pnl.py for that slice's own tests.
    assert BUNDLE_SCHEMA_VERSION == "historical_backtest_result_bundle.v0.6"


def test_positive_total_round_trips_exactly() -> None:
    trade = _minimal_complete_trade(
        realized_pnl=Decimal("5"), ordinary_dividend_income=Decimal("10"),
        trade_total_pnl=Decimal("15"),
    )
    decoded = decode_trade_row(encode_trade_row(trade, 0))
    assert decoded == trade
    assert decoded.trade_total_pnl == Decimal("15")


def test_zero_total_round_trips_exactly() -> None:
    trade = _minimal_complete_trade(
        realized_pnl=Decimal("-10"), ordinary_dividend_income=Decimal("10"),
        trade_total_pnl=Decimal("0"),
    )
    decoded = decode_trade_row(encode_trade_row(trade, 0))
    assert decoded.trade_total_pnl == Decimal("0")
    assert type(decoded.trade_total_pnl) is Decimal


def test_negative_total_round_trips_exactly() -> None:
    trade = _minimal_complete_trade(
        realized_pnl=Decimal("-100"), ordinary_dividend_income=Decimal("10"),
        trade_total_pnl=Decimal("-90"),
    )
    decoded = decode_trade_row(encode_trade_row(trade, 0))
    assert decoded.trade_total_pnl == Decimal("-90")


def test_partial_none_total_round_trips() -> None:
    trade = _minimal_complete_trade(
        carried_in=True,
        dividend_attribution_completeness=(
            DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
        ),
        trade_total_pnl=None,
    )
    decoded = decode_trade_row(encode_trade_row(trade, 0))
    assert decoded.trade_total_pnl is None


def test_legacy_v0_1_none_total_round_trips() -> None:
    trade = HistoricalClosedTradeRecord(
        trade_id="X", security_id=ASSET_A, quantity=1, carried_in=False,
        entry_execution_id="X", entry_session=T_SESSION,
        entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
        entry_cost_basis=Decimal("1"), exit_execution_id="Y",
        exit_session=X_SESSION, exit_fill_price=Decimal("1"),
        exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
        gross_exit_proceeds=Decimal("1"), net_exit_proceeds=Decimal("1"),
        realized_pnl=Decimal("0"), schema_version="historical_closed_trade.v0.1",
    )
    decoded = decode_trade_row(encode_trade_row(trade, 0))
    assert decoded == trade
    assert decoded.trade_total_pnl is None


def test_v0_2_grouping_only_round_trips_without_backfill() -> None:
    trade = HistoricalClosedTradeRecord(
        trade_id="X", security_id=ASSET_A, quantity=1, carried_in=False,
        entry_execution_id="X", entry_session=T_SESSION,
        entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
        entry_cost_basis=Decimal("1"), exit_execution_id="Y",
        exit_session=X_SESSION, exit_fill_price=Decimal("1"),
        exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
        gross_exit_proceeds=Decimal("1"), net_exit_proceeds=Decimal("1"),
        realized_pnl=Decimal("0"), ordinary_dividend_income=Decimal("0"),
        ordinary_dividend_event_count=0,
        dividend_attribution_completeness=(
            DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
        ),
        ordinary_dividend_attribution_fingerprint="f" * 64,
        schema_version="historical_closed_trade.v0.2",
    )
    decoded = decode_trade_row(encode_trade_row(trade, 0))
    assert decoded == trade
    assert decoded.trade_total_pnl is None


def test_old_bundle_schema_version_rejected_after_v0_5_bump(
    content_fingerprints,
) -> None:
    manifest = build_bundle_manifest(
        result_schema_version="historical_backtest_audit_result.v0.2",
        run_configuration_fingerprint="a" * 64,
        source_run_fingerprint="b" * 64,
        initial_state_fingerprint="c" * 64,
        final_state_fingerprint="d" * 64,
        result_fingerprint="e" * 64,
        content_fingerprints=content_fingerprints,
        row_counts={},
        payload_sha256={},
    )
    raw = json.loads(encode_bundle_manifest(manifest).decode("utf-8"))
    raw["persistence_schema_version"] = "historical_backtest_result_bundle.v0.4"
    tampered_bytes = json.dumps(raw).encode("utf-8")
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        decode_bundle_manifest(tampered_bytes)
    assert excinfo.value.code == "UNSUPPORTED_SCHEMA_VERSION"


# ---------------------------------------------------------------------------
# Fingerprints (required tests 45-49)
# ---------------------------------------------------------------------------


def test_content_fingerprint_deterministic_with_trade_total_pnl() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    first = compute_content_fingerprint(artifact_name="trades", rows_or_value=(trade,))
    second = compute_content_fingerprint(artifact_name="trades", rows_or_value=(trade,))
    assert first == second


def test_content_fingerprint_changes_when_only_trade_total_pnl_changes() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    changed = trade.model_copy(
        update={"trade_total_pnl": trade.trade_total_pnl + Decimal("1")}
    )
    original_fp = compute_content_fingerprint(
        artifact_name="trades", rows_or_value=(trade,)
    )
    changed_fp = compute_content_fingerprint(
        artifact_name="trades", rows_or_value=(changed,)
    )
    assert original_fp != changed_fp


def test_dividend_attribution_fingerprint_function_never_takes_total_pnl() -> None:
    import inspect

    source = inspect.getsource(compute_dividend_attribution_fingerprint)
    assert "trade_total_pnl" not in source


def test_source_run_hash_domain_is_the_frozen_5cb_v0_3() -> None:
    # Frozen Task 5C-B baseline: v0.3 (nested session results gained
    # entry_session_protective_decisions); Task 5C-C adds no further bump.
    assert SOURCE_RUN_HASH_DOMAIN == "historical_backtest_source_run.v0.3"


# ---------------------------------------------------------------------------
# Regression (required tests 50-62)
# ---------------------------------------------------------------------------


def test_slice10_fields_unchanged_by_adding_trade_total_pnl() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    assert trade.realized_pnl == trade.net_exit_proceeds - trade.entry_cost_basis
    assert trade.entry_cost_basis == (
        Decimal(trade.quantity) * trade.entry_fill_price + trade.entry_execution_cost
    )
    assert trade.net_exit_proceeds == (
        trade.gross_exit_proceeds - trade.exit_execution_cost
    )
    assert trade.ordinary_dividend_event_count == 1
    assert trade.dividend_attribution_completeness == (
        DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
    )
    assert trade.ordinary_dividend_attribution_fingerprint is not None


# Slice 11's `test_no_portfolio_scope_dividend_pnl_fields_introduced`
# boundary marker is superseded by Slice 12, which adds exactly those
# fields; see test_phase15d_portfolio_dividend_pnl.py.
