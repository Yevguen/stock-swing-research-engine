"""Cross-layer exact-arithmetic hardening: three known residuals identified
during the final Slice-12 audit.

1. Phase 15D open-trade projection (`project_open_trades` / project's own
   `HistoricalOpenTradeRecord.validate_trade_identity`): `quantity *
   final_mark_price` and the adjacent `unrealized_pnl` subtraction.
2. Phase 13 `OpenPosition.validate_cost_basis` and its authoritative
   construction paths (`PortfolioTransitionEngine.transition`'s BUY
   processing and `PortfolioInvariantChecker._apply_buy_ledger`'s ledger
   replay): `quantity * entry_price + entry_execution_cost`.
3. Phase 15D closed-trade gross exit proceeds (`project_closed_trades` /
   `HistoricalClosedTradeRecord.validate_trade_identity`): `quantity *
   exit_fill_price`, `net_exit_proceeds`, `realized_pnl`.

The shared exact Decimal x integer / exact subtraction primitives now live
in `portfolio_dividend_events.py` (Phase 13's existing exact-Decimal
utility module, already the accepted home for `add_exact_decimal`), so
Phase 15D imports from Phase 13 -- never the reverse.

This is not a new economic-design slice: no field, schema version, bundle
version, hash domain, or economic definition changes here. Only the
arithmetic implementation of already-frozen identities is hardened.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from stock_swing_d1.backtest_results import (
    build_run_manifest,
    build_valuation_snapshot,
    project_closed_trades,
    project_open_trades,
    HistoricalBacktestValuationMark,
    HistoricalClosedTradeRecord,
    HistoricalOpenTradeRecord,
    PolicyArtifactRef,
    ArtifactRef,
)
from stock_swing_d1.backtest_results.models import (
    HistoricalBacktestValuationPolicy,
    build_valuation_policy_ref,
)
from stock_swing_d1.backtester.models import (
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionResult,
    HistoricalDecisionInterval,
)
from stock_swing_d1.execution.costs.models import (
    BROKER_NEUTRAL_POLICY_ID,
    ExecutionCostPolicyRef,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    add_exact_decimal,
    exact_decimal_times_int,
    subtract_exact_decimal,
)
from stock_swing_d1.portfolio.portfolio_errors import PortfolioStateError
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from stock_swing_d1.ranking import rank_candidates
from stock_swing_d1.ranking.hashing import CANDIDATE_RANKING_POLICY_FINGERPRINT
from stock_swing_d1.ranking.models import (
    CANDIDATE_RANKING_POLICY_ID,
    CANDIDATE_RANKING_POLICY_VERSION,
    RankingPolicyRef,
)

DECISION_INTERVAL = HistoricalDecisionInterval(
    decision_start_date=date(2020, 1, 1), decision_end_date=date(2030, 12, 31)
)
HIGH_PRECISION = Decimal("33.333333333333333333333333333333333")  # 35 sig figs


def _dt(session: date) -> datetime:
    return datetime.combine(session, time(20), tzinfo=timezone.utc)


def _artifact_ref(kind: str, digest: str = "a" * 64) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=kind, schema_version="artifact.v0.1", content_sha256=digest
    )


def _ranking(session: date):
    return rank_candidates(
        ranking_session=session, decision_time=_dt(session), candidates=()
    )


@pytest.fixture
def run_manifest():
    return build_run_manifest(
        software_revision="c" * 40,
        strategy_configuration_ref=_artifact_ref("strategy"),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id="allocation_v0.1", policy_version="0.1",
            policy_fingerprint="d" * 64,
        ),
        ranking_policy_ref=RankingPolicyRef(
            policy_id=CANDIDATE_RANKING_POLICY_ID,
            policy_version=CANDIDATE_RANKING_POLICY_VERSION,
            policy_fingerprint=CANDIDATE_RANKING_POLICY_FINGERPRINT,
        ),
        execution_cost_policy_ref=ExecutionCostPolicyRef(
            policy_id=BROKER_NEUTRAL_POLICY_ID, policy_fingerprint="b" * 64
        ),
        valuation_policy_ref=build_valuation_policy_ref(
            HistoricalBacktestValuationPolicy()
        ),
        universe_artifact_ref=_artifact_ref("universe"),
        market_data_artifact_ref=_artifact_ref("market_data"),
    )


def _mark(run_manifest, *, security_id: str, session: date, close: Decimal):
    return HistoricalBacktestValuationMark(
        security_id=security_id, session=session, close=close,
        source_artifact_ref=run_manifest.market_data_artifact_ref,
    )


# ---------------------------------------------------------------------------
# Shared exact Decimal x integer / exact subtraction helper (Section 12)
# ---------------------------------------------------------------------------


def test_helper_positive_decimal_times_positive_int_matches_ordinary_multiply() -> None:
    assert exact_decimal_times_int(Decimal("100.50"), 3) == Decimal("100.50") * 3


def test_helper_zero_decimal() -> None:
    assert exact_decimal_times_int(Decimal("0"), 7) == Decimal("0")


def test_helper_zero_integer() -> None:
    assert exact_decimal_times_int(Decimal("123.456"), 0) == Decimal("0")


def test_helper_negative_decimal_permitted() -> None:
    assert exact_decimal_times_int(Decimal("-5.5"), 2) == Decimal("-11.0")


def test_helper_large_integer_multiplier_exact() -> None:
    with localcontext() as ctx:
        ctx.prec = 200
        expected = Decimal("0.1") * 9_999_999
    assert exact_decimal_times_int(Decimal("0.1"), 9_999_999) == expected


def test_helper_preserves_trailing_zero_exponent() -> None:
    result = exact_decimal_times_int(Decimal("2.50"), 4)
    assert result == Decimal("10.00")
    assert result.as_tuple().exponent == Decimal("2.50").as_tuple().exponent


def test_helper_high_significant_digit_decimal() -> None:
    result = exact_decimal_times_int(HIGH_PRECISION, 3)
    assert len(result.as_tuple().digits) > 28


def test_helper_identical_across_precision_28_60_200() -> None:
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            results.append(exact_decimal_times_int(HIGH_PRECISION, 3))
    assert results[0] == results[1] == results[2]


def test_helper_naive_multiplication_demonstrably_differs() -> None:
    with localcontext() as ctx:
        ctx.prec = 28
        naive = HIGH_PRECISION * 3
    exact = exact_decimal_times_int(HIGH_PRECISION, 3)
    assert naive != exact


def test_helper_rejects_bool_multiplier() -> None:
    with pytest.raises(ValueError):
        exact_decimal_times_int(Decimal("1"), True)


def test_helper_rejects_float_value() -> None:
    with pytest.raises(ValueError):
        exact_decimal_times_int(1.0, 3)


def test_helper_rejects_nan_and_infinity() -> None:
    with pytest.raises(ValueError):
        exact_decimal_times_int(Decimal("NaN"), 3)
    with pytest.raises(ValueError):
        exact_decimal_times_int(Decimal("Infinity"), 3)


def test_helper_does_not_mutate_ambient_context() -> None:
    with localcontext() as ctx:
        ctx.prec = 5
        exact_decimal_times_int(HIGH_PRECISION, 3)
        assert ctx.prec == 5


def test_subtract_exact_decimal_matches_ordinary_subtraction_normally() -> None:
    assert subtract_exact_decimal(Decimal("10"), Decimal("3")) == Decimal("7")


def test_subtract_exact_decimal_high_precision_identical_across_contexts() -> None:
    base = exact_decimal_times_int(HIGH_PRECISION, 3)
    delta = Decimal("1")
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            results.append(subtract_exact_decimal(base, delta))
    assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------------
# Residual A -- Phase 15D open-trade projection (Section 13)
# ---------------------------------------------------------------------------


def _single_open_position_run(*, quantity: int = 3) -> HistoricalBacktestRunResult:
    buy = PortfolioExecutionEvent(
        execution_id="BUY-OPEN", source_order_id="O-1", session=date(2026, 8, 18),
        asset_id="NORGATE:501", side=ExecutionSide.BUY, quantity=quantity,
        fill_price=Decimal("1"), execution_cost=Decimal("0"),
    )
    initial = PortfolioState(settled_cash=Decimal("1000"))
    t_result = PortfolioTransitionEngine.transition(initial, buy.session, (buy,))
    final = t_result.resulting_state
    t_session = HistoricalBacktestSessionResult(
        session=buy.session, decision_time=_dt(buy.session),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy,), state_transition_result=t_result,
        authoritative_state=final, ranking_snapshot=_ranking(buy.session),
    )
    return HistoricalBacktestRunResult(
        decision_interval=DECISION_INTERVAL,
        initial_state=initial, final_state=final,
        session_results=(t_session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def test_open_trade_high_precision_mark_accepted_by_live_model(run_manifest) -> None:
    mark = _mark(run_manifest, security_id="NORGATE:501", session=date(2026, 8, 18), close=HIGH_PRECISION)
    assert len(mark.close.as_tuple().digits) == 35


def test_open_trade_final_market_value_identical_across_precisions(run_manifest) -> None:
    run = _single_open_position_run(quantity=3)
    snapshots = (
        build_valuation_snapshot(
            session=date(2026, 8, 18),
            marks=(_mark(run_manifest, security_id="NORGATE:501", session=date(2026, 8, 18), close=HIGH_PRECISION),),
        ),
    )
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            trades = project_open_trades(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
        results.append(trades[0].final_market_value)
    assert results[0] == results[1] == results[2]
    assert len(results[0].as_tuple().digits) > 28


def test_open_trade_unrealized_pnl_identical_across_precisions(run_manifest) -> None:
    run = _single_open_position_run(quantity=3)
    snapshots = (
        build_valuation_snapshot(
            session=date(2026, 8, 18),
            marks=(_mark(run_manifest, security_id="NORGATE:501", session=date(2026, 8, 18), close=HIGH_PRECISION),),
        ),
    )
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            trades = project_open_trades(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
        results.append(trades[0].unrealized_pnl)
    assert results[0] == results[1] == results[2]


def test_open_trade_ordinary_price_fixture_unchanged(run_manifest) -> None:
    run = _single_open_position_run(quantity=2)
    snapshots = (
        build_valuation_snapshot(
            session=date(2026, 8, 18),
            marks=(_mark(run_manifest, security_id="NORGATE:501", session=date(2026, 8, 18), close=Decimal("1.5")),),
        ),
    )
    trades = project_open_trades(run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots)
    assert trades[0].final_market_value == Decimal("3")
    assert trades[0].unrealized_pnl == Decimal("1")


def test_open_trade_no_dividend_attribution_fields_exist() -> None:
    assert "ordinary_dividend_income" not in HistoricalOpenTradeRecord.model_fields
    assert "trade_total_pnl" not in HistoricalOpenTradeRecord.model_fields


def test_open_trade_model_validator_accepts_exact_record_at_all_precisions() -> None:
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            market_value = exact_decimal_times_int(HIGH_PRECISION, 3)
            unrealized = subtract_exact_decimal(market_value, Decimal("3"))
            trade = HistoricalOpenTradeRecord(
                trade_id="X", security_id="NORGATE:501", quantity=3, carried_in=False,
                entry_execution_id="X", entry_session=date(2026, 8, 18),
                entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
                entry_cost_basis=Decimal("3"), final_mark_session=date(2026, 8, 18),
                final_mark_price=HIGH_PRECISION, final_market_value=market_value,
                unrealized_pnl=unrealized,
            )
        assert trade.final_market_value == market_value


def test_open_trade_model_validator_rejects_wrong_market_value() -> None:
    with pytest.raises(ValidationError):
        HistoricalOpenTradeRecord(
            trade_id="X", security_id="NORGATE:501", quantity=3, carried_in=False,
            entry_execution_id="X", entry_session=date(2026, 8, 18),
            entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
            entry_cost_basis=Decimal("3"), final_mark_session=date(2026, 8, 18),
            final_mark_price=HIGH_PRECISION, final_market_value=Decimal("999"),
            unrealized_pnl=Decimal("996"),
        )


# ---------------------------------------------------------------------------
# Residual B -- Phase 13 OpenPosition cost basis (Section 14)
# ---------------------------------------------------------------------------


def test_cost_basis_exact_high_precision_entry_price_times_quantity() -> None:
    expected = add_exact_decimal(
        exact_decimal_times_int(HIGH_PRECISION, 3), Decimal("0")
    )
    position = OpenPosition(
        asset_id="NORGATE:502", quantity=3, entry_session=date(2026, 8, 18),
        entry_price=HIGH_PRECISION, entry_execution_id="E-1",
        entry_execution_cost=Decimal("0"), cost_basis=expected,
    )
    assert position.cost_basis == expected
    assert len(position.cost_basis.as_tuple().digits) > 28


def test_cost_basis_exact_execution_cost_addition() -> None:
    expected = add_exact_decimal(
        exact_decimal_times_int(HIGH_PRECISION, 3), Decimal("1.5")
    )
    position = OpenPosition(
        asset_id="NORGATE:502", quantity=3, entry_session=date(2026, 8, 18),
        entry_price=HIGH_PRECISION, entry_execution_id="E-1",
        entry_execution_cost=Decimal("1.5"), cost_basis=expected,
    )
    assert position.cost_basis == expected


def test_open_position_valid_exact_cost_basis_passes_at_all_precisions() -> None:
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            expected = add_exact_decimal(
                exact_decimal_times_int(HIGH_PRECISION, 3), Decimal("0")
            )
            position = OpenPosition(
                asset_id="NORGATE:502", quantity=3, entry_session=date(2026, 8, 18),
                entry_price=HIGH_PRECISION, entry_execution_id="E-1",
                entry_execution_cost=Decimal("0"), cost_basis=expected,
            )
        assert position.cost_basis == expected


def test_open_position_wrong_cost_basis_fails_at_all_precisions() -> None:
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            with pytest.raises(ValidationError):
                OpenPosition(
                    asset_id="NORGATE:502", quantity=3, entry_session=date(2026, 8, 18),
                    entry_price=HIGH_PRECISION, entry_execution_id="E-1",
                    entry_execution_cost=Decimal("0"), cost_basis=Decimal("999"),
                )


def test_ordinary_open_position_fixture_unchanged() -> None:
    position = OpenPosition(
        asset_id="NORGATE:1", quantity=2, entry_session=date(2026, 8, 18),
        entry_price=Decimal("10"), entry_execution_id="BUY-1",
        entry_execution_cost=Decimal("1"), cost_basis=Decimal("21"),
    )
    assert position.cost_basis == Decimal("21")


def test_buy_transition_with_high_precision_fill_price_succeeds_at_all_precisions() -> None:
    # End-to-end: the authoritative construction path (PortfolioTransitionEngine)
    # plus its site-3 replay validation (PortfolioInvariantChecker) must agree
    # on cost_basis/settled_cash_delta regardless of ambient precision.
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            buy = PortfolioExecutionEvent(
                execution_id="BUY-HP", source_order_id="O-1", session=date(2026, 8, 18),
                asset_id="NORGATE:503", side=ExecutionSide.BUY, quantity=3,
                fill_price=HIGH_PRECISION, execution_cost=Decimal("0"),
            )
            initial = PortfolioState(settled_cash=Decimal("1000"))
            result = PortfolioTransitionEngine.transition(initial, buy.session, (buy,))
            position = result.resulting_state.open_positions[0]
        expected = exact_decimal_times_int(HIGH_PRECISION, 3)
        assert position.cost_basis == expected
        assert len(position.cost_basis.as_tuple().digits) > 28


def test_portfolio_state_hash_deterministic_across_precisions_for_equivalent_state() -> None:
    hashes = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            buy = PortfolioExecutionEvent(
                execution_id="BUY-HASH", source_order_id="O-1", session=date(2026, 8, 18),
                asset_id="NORGATE:504", side=ExecutionSide.BUY, quantity=3,
                fill_price=HIGH_PRECISION, execution_cost=Decimal("0"),
            )
            initial = PortfolioState(settled_cash=Decimal("1000"))
            result = PortfolioTransitionEngine.transition(initial, buy.session, (buy,))
        hashes.append(hash_portfolio_state(result.resulting_state))
    assert hashes[0] == hashes[1] == hashes[2]


def test_no_change_to_event_ordering_or_settlement_semantics() -> None:
    # Regression: an ordinary BUY-then-SELL round trip is unaffected.
    buy = PortfolioExecutionEvent(
        execution_id="BUY-RT", source_order_id="O-1", session=date(2026, 8, 18),
        asset_id="NORGATE:505", side=ExecutionSide.BUY, quantity=2,
        fill_price=Decimal("10"), execution_cost=Decimal("1"),
    )
    initial = PortfolioState(settled_cash=Decimal("1000"))
    opened = PortfolioTransitionEngine.transition(initial, buy.session, (buy,)).resulting_state
    sell = PortfolioExecutionEvent(
        execution_id="SELL-RT", source_order_id="O-2", session=date(2026, 8, 19),
        asset_id="NORGATE:505", side=ExecutionSide.SELL, quantity=2,
        fill_price=Decimal("12"), execution_cost=Decimal("1"),
        settlement_id="SETT-RT", settlement_session=date(2026, 8, 21),
    )
    final = PortfolioTransitionEngine.transition(opened, sell.session, (sell,)).resulting_state
    assert final.open_positions == ()
    assert len(final.pending_settlements) == 1
    assert final.pending_settlements[0].amount == Decimal("23")  # 2*12-1


# ---------------------------------------------------------------------------
# Residual C -- Phase 15D closed-trade gross exit proceeds (Section 15)
# ---------------------------------------------------------------------------


def _closed_trade_run(*, quantity: int = 3) -> HistoricalBacktestRunResult:
    buy = PortfolioExecutionEvent(
        execution_id="BUY-CT", source_order_id="O-1", session=date(2026, 8, 18),
        asset_id="NORGATE:601", side=ExecutionSide.BUY, quantity=quantity,
        fill_price=Decimal("1"), execution_cost=Decimal("0"),
    )
    initial = PortfolioState(settled_cash=Decimal("1000"))
    t_result = PortfolioTransitionEngine.transition(initial, buy.session, (buy,))
    opened = t_result.resulting_state

    sell = PortfolioExecutionEvent(
        execution_id="SELL-CT", source_order_id="O-2", session=date(2026, 8, 19),
        asset_id="NORGATE:601", side=ExecutionSide.SELL, quantity=quantity,
        fill_price=HIGH_PRECISION, execution_cost=Decimal("0"),
        settlement_id="SETT-CT", settlement_session=date(2026, 8, 21),
    )
    x_result = PortfolioTransitionEngine.transition(opened, sell.session, (sell,))
    final = x_result.resulting_state

    t_session = HistoricalBacktestSessionResult(
        session=buy.session, decision_time=_dt(buy.session),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy,), state_transition_result=t_result,
        authoritative_state=opened, ranking_snapshot=_ranking(buy.session),
    )
    x_session = HistoricalBacktestSessionResult(
        session=sell.session, decision_time=_dt(sell.session),
        prior_state_fingerprint=hash_portfolio_state(opened),
        ordered_execution_events=(sell,), state_transition_result=x_result,
        authoritative_state=final, ranking_snapshot=_ranking(sell.session),
    )
    return HistoricalBacktestRunResult(
        decision_interval=DECISION_INTERVAL,
        initial_state=initial, final_state=final,
        session_results=(t_session, x_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def test_closed_trade_exact_high_precision_exit_price_times_quantity() -> None:
    run = _closed_trade_run(quantity=3)
    trade = project_closed_trades(run)[0]
    expected = exact_decimal_times_int(HIGH_PRECISION, 3)
    assert trade.gross_exit_proceeds == expected
    assert len(trade.gross_exit_proceeds.as_tuple().digits) > 28


def test_closed_trade_gross_exit_proceeds_identical_across_precisions() -> None:
    # Construction and projection happen under the *same* ambient context
    # per iteration (matching real single-process usage, where the
    # context never changes mid-run): this isolates Residual C's own
    # derivation.py arithmetic from the separately out-of-scope Phase-13
    # SELL-side settlement-amount naive computation (see final report).
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            run = _closed_trade_run(quantity=3)
            trades = project_closed_trades(run)
        results.append(trades[0].gross_exit_proceeds)
    assert results[0] == results[1] == results[2]


def test_closed_trade_net_exit_proceeds_identical_across_precisions() -> None:
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            run = _closed_trade_run(quantity=3)
            trades = project_closed_trades(run)
        results.append(trades[0].net_exit_proceeds)
    assert results[0] == results[1] == results[2]


def test_closed_trade_realized_pnl_identical_across_precisions() -> None:
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            run = _closed_trade_run(quantity=3)
            trades = project_closed_trades(run)
        results.append(trades[0].realized_pnl)
    assert results[0] == results[1] == results[2]
    assert len(results[0].as_tuple().digits) > 28


def test_closed_trade_model_validator_accepts_exact_record_at_all_precisions() -> None:
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            gross = exact_decimal_times_int(HIGH_PRECISION, 3)
            net = subtract_exact_decimal(gross, Decimal("0"))
            realized = subtract_exact_decimal(net, Decimal("3"))
            trade = HistoricalClosedTradeRecord(
                trade_id="X", security_id="NORGATE:601", quantity=3, carried_in=False,
                entry_execution_id="X", entry_session=date(2026, 8, 18),
                entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
                entry_cost_basis=Decimal("3"), exit_execution_id="Y",
                exit_session=date(2026, 8, 19), exit_fill_price=HIGH_PRECISION,
                exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
                gross_exit_proceeds=gross, net_exit_proceeds=net, realized_pnl=realized,
                schema_version="historical_closed_trade.v0.1",
            )
        assert trade.gross_exit_proceeds == gross
        assert trade.realized_pnl == realized


def test_closed_trade_model_validator_rejects_wrong_gross_proceeds() -> None:
    with pytest.raises(ValidationError, match="gross_exit_proceeds"):
        HistoricalClosedTradeRecord(
            trade_id="X", security_id="NORGATE:601", quantity=3, carried_in=False,
            entry_execution_id="X", entry_session=date(2026, 8, 18),
            entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
            entry_cost_basis=Decimal("3"), exit_execution_id="Y",
            exit_session=date(2026, 8, 19), exit_fill_price=HIGH_PRECISION,
            exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
            gross_exit_proceeds=Decimal("999"), net_exit_proceeds=Decimal("999"),
            realized_pnl=Decimal("996"),
            schema_version="historical_closed_trade.v0.1",
        )


def test_closed_trade_model_validator_rejects_wrong_net_proceeds() -> None:
    gross = exact_decimal_times_int(HIGH_PRECISION, 3)
    with pytest.raises(ValidationError, match="net_exit_proceeds"):
        HistoricalClosedTradeRecord(
            trade_id="X", security_id="NORGATE:601", quantity=3, carried_in=False,
            entry_execution_id="X", entry_session=date(2026, 8, 18),
            entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
            entry_cost_basis=Decimal("3"), exit_execution_id="Y",
            exit_session=date(2026, 8, 19), exit_fill_price=HIGH_PRECISION,
            exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
            gross_exit_proceeds=gross, net_exit_proceeds=Decimal("999"),
            realized_pnl=Decimal("996"),
            schema_version="historical_closed_trade.v0.1",
        )


def test_closed_trade_model_validator_rejects_wrong_realized_pnl() -> None:
    gross = exact_decimal_times_int(HIGH_PRECISION, 3)
    with pytest.raises(ValidationError, match="realized_pnl"):
        HistoricalClosedTradeRecord(
            trade_id="X", security_id="NORGATE:601", quantity=3, carried_in=False,
            entry_execution_id="X", entry_session=date(2026, 8, 18),
            entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
            entry_cost_basis=Decimal("3"), exit_execution_id="Y",
            exit_session=date(2026, 8, 19), exit_fill_price=HIGH_PRECISION,
            exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
            gross_exit_proceeds=gross, net_exit_proceeds=gross,
            realized_pnl=Decimal("999"),
            schema_version="historical_closed_trade.v0.1",
        )


def test_complete_trade_total_pnl_remains_exact_realized_plus_dividend() -> None:
    # Slice-11 non-interference: trade_total_pnl formula/exactness must
    # still hold for a COMPLETE trade using the now-hardened realized_pnl.
    gross = exact_decimal_times_int(HIGH_PRECISION, 3)
    net = subtract_exact_decimal(gross, Decimal("0"))
    realized = subtract_exact_decimal(net, Decimal("3"))
    dividend_income = Decimal("5")
    from stock_swing_d1.backtest_results import DividendAttributionCompleteness

    trade = HistoricalClosedTradeRecord(
        trade_id="X", security_id="NORGATE:601", quantity=3, carried_in=False,
        entry_execution_id="X", entry_session=date(2026, 8, 18),
        entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
        entry_cost_basis=Decimal("3"), exit_execution_id="Y",
        exit_session=date(2026, 8, 19), exit_fill_price=HIGH_PRECISION,
        exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
        gross_exit_proceeds=gross, net_exit_proceeds=net, realized_pnl=realized,
        ordinary_dividend_income=dividend_income, ordinary_dividend_event_count=1,
        dividend_attribution_completeness=DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME,
        ordinary_dividend_attribution_fingerprint="f" * 64,
        schema_version="historical_closed_trade.v0.3",
        trade_total_pnl=add_exact_decimal(realized, dividend_income),
    )
    assert trade.trade_total_pnl == add_exact_decimal(realized, dividend_income)


def test_partial_trade_total_pnl_still_none() -> None:
    from stock_swing_d1.backtest_results import DividendAttributionCompleteness

    trade = HistoricalClosedTradeRecord(
        trade_id="X", security_id="NORGATE:601", quantity=1, carried_in=True,
        entry_execution_id="X", entry_session=date(2026, 8, 18),
        entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
        entry_cost_basis=Decimal("1"), exit_execution_id="Y",
        exit_session=date(2026, 8, 19), exit_fill_price=Decimal("1"),
        exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
        gross_exit_proceeds=Decimal("1"), net_exit_proceeds=Decimal("1"),
        realized_pnl=Decimal("0"), ordinary_dividend_income=Decimal("5"),
        ordinary_dividend_event_count=1,
        dividend_attribution_completeness=DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN,
        ordinary_dividend_attribution_fingerprint="f" * 64,
        schema_version="historical_closed_trade.v0.3",
    )
    assert trade.trade_total_pnl is None


def test_ordinary_closed_trade_fixture_unchanged(run_manifest) -> None:
    buy = PortfolioExecutionEvent(
        execution_id="BUY-ORD", source_order_id="O-1", session=date(2026, 8, 18),
        asset_id="NORGATE:602", side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    initial = PortfolioState(settled_cash=Decimal("100000"))
    t_result = PortfolioTransitionEngine.transition(initial, buy.session, (buy,))
    opened = t_result.resulting_state
    sell = PortfolioExecutionEvent(
        execution_id="SELL-ORD", source_order_id="O-2", session=date(2026, 8, 19),
        asset_id="NORGATE:602", side=ExecutionSide.SELL, quantity=10,
        fill_price=Decimal("110"), execution_cost=Decimal("1"),
        settlement_id="SETT-ORD", settlement_session=date(2026, 8, 21),
    )
    x_result = PortfolioTransitionEngine.transition(opened, sell.session, (sell,))
    final = x_result.resulting_state
    t_session = HistoricalBacktestSessionResult(
        session=buy.session, decision_time=_dt(buy.session),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy,), state_transition_result=t_result,
        authoritative_state=opened, ranking_snapshot=_ranking(buy.session),
    )
    x_session = HistoricalBacktestSessionResult(
        session=sell.session, decision_time=_dt(sell.session),
        prior_state_fingerprint=hash_portfolio_state(opened),
        ordered_execution_events=(sell,), state_transition_result=x_result,
        authoritative_state=final, ranking_snapshot=_ranking(sell.session),
    )
    run = HistoricalBacktestRunResult(
        decision_interval=DECISION_INTERVAL,
        initial_state=initial, final_state=final,
        session_results=(t_session, x_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )
    trade = project_closed_trades(run)[0]
    assert trade.gross_exit_proceeds == Decimal("1100")
    assert trade.net_exit_proceeds == Decimal("1099")
    assert trade.realized_pnl == Decimal("98")
