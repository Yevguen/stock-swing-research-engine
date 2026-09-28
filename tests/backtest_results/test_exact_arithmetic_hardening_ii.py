"""Exact-Arithmetic Hardening II: three residuals identified by an
internal post-audit review of the accepted working tree.

1. Phase 13 SELL proceeds (`PortfolioTransitionEngine.transition`'s SELL
   branch, and its replay counterpart
   `PortfolioInvariantChecker._apply_sell_ledger`): `quantity * fill_price -
   execution_cost`, feeding the authoritative `PendingSettlement.amount`
   and ledger `pending_cash_delta`.
2. Phase 13 settlement maturation (`PortfolioTransitionEngine.transition`'s
   due-settlement loop): `settled_cash += settlement.amount`.
3. Phase 15D raw entry/exit projection -- `project_entries` /
   `HistoricalBacktestEntryRecord.validate_cost_basis` (reconciliation
   against the exact authoritative Phase 13 `OpenPosition.cost_basis`) and
   `project_exits` / `HistoricalBacktestExitRecord.validate_proceeds`.
   These are distinct call sites from the already-hardened
   `project_closed_trades` / `HistoricalClosedTradeRecord` path covered by
   the first hardening slice (`test_exact_arithmetic_hardening.py`); that
   projection never called `project_entries`/`project_exits`, so its
   already-exact `gross_exit_proceeds`/`net_exit_proceeds` fields left
   these two sites unaffected.

Same shared exact-Decimal primitives as the first slice
(`portfolio_dividend_events.add_exact_decimal` / `exact_decimal_times_int` /
`subtract_exact_decimal`); Phase 15D imports from Phase 13, never the
reverse. This is not a new economic-design slice: no field, schema version,
bundle version, hash domain, or economic definition changes here. Only the
arithmetic implementation of already-frozen identities is hardened.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from stock_swing_d1.backtest_results import (
    ExecutionProvenanceSource,
    HistoricalBacktestEntryRecord,
    HistoricalBacktestExitRecord,
    HistoricalExitReason,
    compute_source_payload_fingerprint,
    project_entries,
    project_exits,
)
from stock_swing_d1.backtester.models import (
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionResult,
    HistoricalDecisionInterval,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    add_exact_decimal,
    exact_decimal_times_int,
    subtract_exact_decimal,
)
from stock_swing_d1.portfolio.portfolio_errors import PortfolioStateError
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    ExecutionSide,
    PortfolioEventKind,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import (
    PendingSettlement,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from stock_swing_d1.ranking import rank_candidates

DECISION_INTERVAL = HistoricalDecisionInterval(
    decision_start_date=date(2020, 1, 1), decision_end_date=date(2030, 12, 31)
)
HIGH_PRECISION = Decimal("33.333333333333333333333333333333333")  # 35 sig figs


def _dt(session: date) -> datetime:
    return datetime.combine(session, time(20), tzinfo=timezone.utc)


def _ranking(session: date):
    return rank_candidates(
        ranking_session=session, decision_time=_dt(session), candidates=()
    )


def _buy(
    execution_id: str,
    *,
    asset_id: str,
    quantity: int = 3,
    fill_price: Decimal = Decimal("1"),
    session: date = date(2026, 8, 18),
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id, source_order_id=f"O-{execution_id}",
        session=session, asset_id=asset_id, side=ExecutionSide.BUY,
        quantity=quantity, fill_price=fill_price, execution_cost=Decimal("0"),
    )


def _sell(
    execution_id: str,
    *,
    asset_id: str,
    quantity: int = 3,
    fill_price: Decimal,
    execution_cost: Decimal = Decimal("0"),
    session: date = date(2026, 8, 19),
    settlement_id: str = "SETT-1",
    settlement_session: date = date(2026, 8, 21),
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id, source_order_id=f"O-{execution_id}",
        session=session, asset_id=asset_id, side=ExecutionSide.SELL,
        quantity=quantity, fill_price=fill_price, execution_cost=execution_cost,
        settlement_id=settlement_id, settlement_session=settlement_session,
    )


def _opened(*, asset_id: str, quantity: int = 3, cash: Decimal = Decimal("1000000")):
    buy = _buy(f"BUY-{asset_id}", asset_id=asset_id, quantity=quantity)
    initial = PortfolioState(settled_cash=cash)
    opened = PortfolioTransitionEngine.transition(
        initial, buy.session, (buy,)
    ).resulting_state
    return initial, opened


# ---------------------------------------------------------------------------
# Residual A -- Phase 13 SELL proceeds
# ---------------------------------------------------------------------------


def test_sell_high_significant_digit_fill_price_accepted() -> None:
    _, opened = _opened(asset_id="NORGATE:8010")
    sell = _sell("SELL-A1", asset_id="NORGATE:8010", fill_price=HIGH_PRECISION)
    result = PortfolioTransitionEngine.transition(opened, sell.session, (sell,))
    amount = result.resulting_state.pending_settlements[0].amount
    assert len(amount.as_tuple().digits) > 28


def test_sell_zero_cost_net_proceeds_equal_exact_gross_at_all_precisions() -> None:
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            _, opened = _opened(asset_id=f"NORGATE:802{prec}")
            sell = _sell(
                "SELL-A2", asset_id=f"NORGATE:802{prec}", fill_price=HIGH_PRECISION
            )
            result = PortfolioTransitionEngine.transition(opened, sell.session, (sell,))
        results.append(result.resulting_state.pending_settlements[0].amount)
    assert results[0] == results[1] == results[2]
    assert results[0] == exact_decimal_times_int(HIGH_PRECISION, 3)


def test_sell_net_proceeds_identical_across_precisions_with_nonzero_cost() -> None:
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            _, opened = _opened(asset_id=f"NORGATE:803{prec}")
            sell = _sell(
                "SELL-A3", asset_id=f"NORGATE:803{prec}",
                fill_price=HIGH_PRECISION, execution_cost=Decimal("2.50"),
            )
            result = PortfolioTransitionEngine.transition(opened, sell.session, (sell,))
        results.append(result.resulting_state.pending_settlements[0].amount)
    assert results[0] == results[1] == results[2]
    expected = subtract_exact_decimal(
        exact_decimal_times_int(HIGH_PRECISION, 3), Decimal("2.50")
    )
    assert results[0] == expected


def test_pending_settlement_amount_and_ledger_pending_cash_delta_agree() -> None:
    _, opened = _opened(asset_id="NORGATE:8040")
    sell = _sell(
        "SELL-A4", asset_id="NORGATE:8040",
        fill_price=HIGH_PRECISION, execution_cost=Decimal("1"),
    )
    result = PortfolioTransitionEngine.transition(opened, sell.session, (sell,))
    amount = result.resulting_state.pending_settlements[0].amount
    ledger_entry = result.ledger_entries[0]
    assert ledger_entry.pending_cash_delta == amount
    assert ledger_entry.settled_cash_delta == Decimal("0")


def test_sell_ordinary_price_fixture_unchanged() -> None:
    _, opened = _opened(asset_id="NORGATE:8050", quantity=2)
    sell = _sell(
        "SELL-A5", asset_id="NORGATE:8050", quantity=2,
        fill_price=Decimal("12"), execution_cost=Decimal("1"),
    )
    result = PortfolioTransitionEngine.transition(opened, sell.session, (sell,))
    assert result.resulting_state.pending_settlements[0].amount == Decimal("23")


def test_sell_settlement_id_and_session_unaffected_by_high_precision_price() -> None:
    _, opened = _opened(asset_id="NORGATE:8060")
    sell = _sell(
        "SELL-A6", asset_id="NORGATE:8060", fill_price=HIGH_PRECISION,
        settlement_id="SETT-A6", settlement_session=date(2026, 8, 24),
    )
    result = PortfolioTransitionEngine.transition(opened, sell.session, (sell,))
    settlement = result.resulting_state.pending_settlements[0]
    assert settlement.settlement_id == "SETT-A6"
    assert settlement.settlement_session == date(2026, 8, 24)
    assert settlement.source_execution_id == "SELL-A6"


def test_sell_replay_with_high_precision_price_does_not_duplicate_settlement() -> None:
    _, opened = _opened(asset_id="NORGATE:8070")
    sell = _sell("SELL-A7", asset_id="NORGATE:8070", fill_price=HIGH_PRECISION)
    sold = PortfolioTransitionEngine.transition(
        opened, sell.session, (sell,)
    ).resulting_state

    replay_result = PortfolioTransitionEngine.transition(
        sold, date(2026, 8, 20), (sell,)
    )
    assert replay_result.newly_applied_events == ()
    assert replay_result.resulting_state.pending_settlements == sold.pending_settlements
    assert replay_result.resulting_state.settled_cash == sold.settled_cash


def test_sell_with_high_precision_price_and_no_dividend_evidence_is_unaffected() -> None:
    _, opened = _opened(asset_id="NORGATE:8080")
    sell = _sell("SELL-A8", asset_id="NORGATE:8080", fill_price=HIGH_PRECISION)
    result = PortfolioTransitionEngine.transition(
        opened, sell.session, (sell,), dividend_evidence=()
    )
    assert result.dividend_outcomes == ()
    assert result.dividend_ledger_entries == ()


def test_sell_ledger_replay_validation_agrees_with_live_transition_at_high_precision() -> None:
    # PortfolioInvariantChecker.validate_transition (called internally by
    # `transition`) replays every ledger entry via `_apply_sell_ledger`.
    # This exercises that replay path end-to-end: if it disagreed with the
    # live SELL computation, `transition` itself would raise.
    _, opened = _opened(asset_id="NORGATE:8090")
    sell = _sell(
        "SELL-A9", asset_id="NORGATE:8090",
        fill_price=HIGH_PRECISION, execution_cost=Decimal("3"),
    )
    result = PortfolioTransitionEngine.transition(opened, sell.session, (sell,))
    assert result.resulting_state.pending_settlements[0].amount == subtract_exact_decimal(
        exact_decimal_times_int(HIGH_PRECISION, 3), Decimal("3")
    )


# ---------------------------------------------------------------------------
# Residual B -- Phase 13 settlement maturation
# ---------------------------------------------------------------------------


def _sold(*, asset_id: str, quantity: int = 3, fill_price: Decimal,
          settlement_session: date = date(2026, 8, 21)):
    _, opened = _opened(asset_id=asset_id, quantity=quantity)
    sell = _sell(
        f"SELL-{asset_id}", asset_id=asset_id, quantity=quantity,
        fill_price=fill_price, settlement_session=settlement_session,
    )
    return PortfolioTransitionEngine.transition(
        opened, sell.session, (sell,)
    ).resulting_state


def test_single_high_precision_settlement_matures_identically_across_precisions() -> None:
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            sold = _sold(asset_id=f"NORGATE:810{prec}", fill_price=HIGH_PRECISION)
            pending = sold.pending_settlements[0]
            matured = PortfolioTransitionEngine.transition(
                sold, pending.settlement_session, ()
            ).resulting_state
        results.append(matured.settled_cash)
    assert results[0] == results[1] == results[2]


def test_multiple_same_session_settlements_accumulate_identically_across_precisions() -> None:
    settlement_session = date(2026, 8, 21)
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            asset_1 = f"NORGATE:811{prec}"
            asset_2 = f"NORGATE:812{prec}"
            buy_1 = _buy("BUY-1", asset_id=asset_1)
            buy_2 = _buy("BUY-2", asset_id=asset_2)
            initial = PortfolioState(settled_cash=Decimal("1000000"))
            opened = PortfolioTransitionEngine.transition(
                initial, buy_1.session, (buy_1, buy_2)
            ).resulting_state

            sell_1 = _sell(
                "SELL-1", asset_id=asset_1, fill_price=HIGH_PRECISION,
                settlement_id="SETT-1", settlement_session=settlement_session,
            )
            sell_2 = _sell(
                "SELL-2", asset_id=asset_2, fill_price=HIGH_PRECISION,
                settlement_id="SETT-2", settlement_session=settlement_session,
            )
            sold_both = PortfolioTransitionEngine.transition(
                opened, sell_1.session, (sell_1, sell_2)
            ).resulting_state
            assert len(sold_both.pending_settlements) == 2

            matured = PortfolioTransitionEngine.transition(
                sold_both, settlement_session, ()
            ).resulting_state
        results.append(matured.settled_cash)
    assert results[0] == results[1] == results[2]
    assert matured.pending_settlements == ()
    expected_per_settlement = exact_decimal_times_int(HIGH_PRECISION, 3)
    expected_total = add_exact_decimal(
        sold_both.settled_cash,
        add_exact_decimal(expected_per_settlement, expected_per_settlement),
    )
    assert matured.settled_cash == expected_total


def test_naive_decimal_accumulation_differs_from_exact_on_adversarial_fixture() -> None:
    exact_amount = exact_decimal_times_int(HIGH_PRECISION, 3)
    base = Decimal("1000")
    with localcontext() as ctx:
        ctx.prec = 28
        naive = base + exact_amount
    exact = add_exact_decimal(base, exact_amount)
    assert naive != exact


def test_matured_settlement_is_removed_exactly_once() -> None:
    sold = _sold(asset_id="NORGATE:8130", fill_price=HIGH_PRECISION)
    pending = sold.pending_settlements[0]
    matured = PortfolioTransitionEngine.transition(
        sold, pending.settlement_session, ()
    ).resulting_state
    assert matured.pending_settlements == ()


def test_settled_cash_increases_by_exactly_the_settlement_amount_once() -> None:
    sold = _sold(asset_id="NORGATE:8140", fill_price=HIGH_PRECISION)
    pending = sold.pending_settlements[0]
    matured = PortfolioTransitionEngine.transition(
        sold, pending.settlement_session, ()
    ).resulting_state
    assert matured.settled_cash == add_exact_decimal(sold.settled_cash, pending.amount)


def test_replay_does_not_mature_the_same_high_precision_settlement_twice() -> None:
    sold = _sold(asset_id="NORGATE:8150", fill_price=HIGH_PRECISION)
    pending = sold.pending_settlements[0]
    matured = PortfolioTransitionEngine.transition(
        sold, pending.settlement_session, ()
    ).resulting_state
    malformed = matured.model_copy(update={"pending_settlements": (pending,)})

    with pytest.raises(PortfolioStateError, match="already-applied"):
        PortfolioTransitionEngine.transition(malformed, date(2026, 8, 24), ())

    assert matured.settled_cash == add_exact_decimal(sold.settled_cash, pending.amount)


def test_settlement_maturation_ordinary_fixture_unchanged() -> None:
    sold = _sold(asset_id="NORGATE:8160", fill_price=Decimal("20"))
    pending = sold.pending_settlements[0]
    matured = PortfolioTransitionEngine.transition(
        sold, pending.settlement_session, ()
    ).resulting_state
    assert matured.settled_cash == sold.settled_cash + Decimal("60")


def test_settlement_ordering_and_availability_timing_unchanged() -> None:
    sold = _sold(asset_id="NORGATE:8170", fill_price=HIGH_PRECISION)
    pending = sold.pending_settlements[0]
    before_due = PortfolioTransitionEngine.transition(
        sold, date(2026, 8, 20), ()
    )
    assert before_due.resulting_state.settled_cash == sold.settled_cash
    assert before_due.resulting_state.pending_settlements == sold.pending_settlements
    assert before_due.ledger_entries == ()

    on_due = PortfolioTransitionEngine.transition(sold, pending.settlement_session, ())
    assert on_due.resulting_state.pending_settlements == ()
    assert on_due.ledger_entries[0].settled_cash_delta == pending.amount


# ---------------------------------------------------------------------------
# Residual C -- Phase 15D raw entry/exit projection
# ---------------------------------------------------------------------------


def _entry_run(*, fill_price: Decimal, quantity: int = 3) -> HistoricalBacktestRunResult:
    buy = _buy("BUY-C", asset_id="NORGATE:820", quantity=quantity, fill_price=fill_price)
    initial = PortfolioState(settled_cash=Decimal("100000000"))
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


def _exit_run(*, exit_fill_price: Decimal, quantity: int = 3) -> HistoricalBacktestRunResult:
    buy = _buy("BUY-D", asset_id="NORGATE:830", quantity=quantity, fill_price=Decimal("1"))
    initial = PortfolioState(settled_cash=Decimal("100000000"))
    t_result = PortfolioTransitionEngine.transition(initial, buy.session, (buy,))
    opened = t_result.resulting_state
    sell = _sell(
        "SELL-D", asset_id="NORGATE:830", quantity=quantity,
        fill_price=exit_fill_price, settlement_id="SETT-D",
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


def test_project_entries_high_significant_digit_fill_price_accepted() -> None:
    run = _entry_run(fill_price=HIGH_PRECISION)
    entries = project_entries(run)
    assert len(entries[0].cost_basis.as_tuple().digits) > 28


def test_project_entries_cost_basis_identical_across_precisions() -> None:
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            run = _entry_run(fill_price=HIGH_PRECISION)
            entries = project_entries(run)
        results.append(entries[0].cost_basis)
    assert results[0] == results[1] == results[2]
    assert results[0] == exact_decimal_times_int(HIGH_PRECISION, 3)


def test_project_exits_high_significant_digit_fill_price_accepted() -> None:
    run = _exit_run(exit_fill_price=HIGH_PRECISION)
    exits = project_exits(run)
    assert len(exits[0].gross_proceeds.as_tuple().digits) > 28


def test_project_exits_gross_proceeds_identical_across_precisions() -> None:
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            run = _exit_run(exit_fill_price=HIGH_PRECISION)
            exits = project_exits(run)
        results.append(exits[0].gross_proceeds)
    assert results[0] == results[1] == results[2]
    assert results[0] == exact_decimal_times_int(HIGH_PRECISION, 3)


def test_project_exits_net_proceeds_identical_across_precisions() -> None:
    results = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            run = _exit_run(exit_fill_price=HIGH_PRECISION)
            exits = project_exits(run)
        results.append(exits[0].net_proceeds)
    assert results[0] == results[1] == results[2]


def test_raw_entry_and_exit_records_carry_no_pnl_field() -> None:
    # Residual C classification (Section 13, item 21/25): project_entries
    # and project_exits are the raw Phase 15D.3A entry/exit tables. Neither
    # record type carries a directly linked realized/unrealized P&L field
    # -- that is computed only downstream by the already-hardened
    # project_closed_trades/project_open_trades -- so there is no separate
    # P&L identity to harden here.
    assert "realized_pnl" not in HistoricalBacktestExitRecord.model_fields
    assert "unrealized_pnl" not in HistoricalBacktestExitRecord.model_fields
    assert "cost_basis" not in HistoricalBacktestExitRecord.model_fields  # exit-side has none
    assert "ordinary_dividend_income" not in HistoricalBacktestEntryRecord.model_fields
    assert "ordinary_dividend_income" not in HistoricalBacktestExitRecord.model_fields


def test_entry_model_validator_accepts_exact_record_at_all_precisions() -> None:
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            cost_basis = add_exact_decimal(
                exact_decimal_times_int(HIGH_PRECISION, 3), Decimal("0")
            )
            entry = HistoricalBacktestEntryRecord(
                entry_execution_id="E", source_order_id="O", security_id="NORGATE:1",
                quantity=3, entry_session=date(2026, 8, 18), fill_price=HIGH_PRECISION,
                execution_cost=Decimal("0"), cost_basis=cost_basis,
                provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
            )
        assert entry.cost_basis == cost_basis


def test_entry_model_validator_rejects_wrong_cost_basis_at_all_precisions() -> None:
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            with pytest.raises(ValidationError, match="cost_basis"):
                HistoricalBacktestEntryRecord(
                    entry_execution_id="E", source_order_id="O", security_id="NORGATE:1",
                    quantity=3, entry_session=date(2026, 8, 18), fill_price=HIGH_PRECISION,
                    execution_cost=Decimal("0"), cost_basis=Decimal("999"),
                    provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
                )


def test_exit_model_validator_accepts_exact_record_at_all_precisions() -> None:
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            gross = exact_decimal_times_int(HIGH_PRECISION, 3)
            net = subtract_exact_decimal(gross, Decimal("0"))
            exit_record = HistoricalBacktestExitRecord(
                exit_execution_id="X", source_order_id="O", entry_execution_id="E",
                security_id="NORGATE:1", quantity=3, exit_session=date(2026, 8, 19),
                fill_price=HIGH_PRECISION, execution_cost=Decimal("0"),
                gross_proceeds=gross, net_proceeds=net,
                settlement_id="S", settlement_session=date(2026, 8, 21),
                provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
                exit_reason=HistoricalExitReason.EXTERNAL_SCHEDULED,
            )
        assert exit_record.gross_proceeds == gross
        assert exit_record.net_proceeds == net


def test_exit_model_validator_rejects_wrong_gross_proceeds_at_all_precisions() -> None:
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            with pytest.raises(ValidationError, match="gross_proceeds"):
                HistoricalBacktestExitRecord(
                    exit_execution_id="X", source_order_id="O", entry_execution_id="E",
                    security_id="NORGATE:1", quantity=3, exit_session=date(2026, 8, 19),
                    fill_price=HIGH_PRECISION, execution_cost=Decimal("0"),
                    gross_proceeds=Decimal("999"), net_proceeds=Decimal("999"),
                    settlement_id="S", settlement_session=date(2026, 8, 21),
                    provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
                    exit_reason=HistoricalExitReason.EXTERNAL_SCHEDULED,
                )


def test_exit_model_validator_rejects_wrong_net_proceeds_at_all_precisions() -> None:
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            gross = exact_decimal_times_int(HIGH_PRECISION, 3)
            with pytest.raises(ValidationError, match="net_proceeds"):
                HistoricalBacktestExitRecord(
                    exit_execution_id="X", source_order_id="O", entry_execution_id="E",
                    security_id="NORGATE:1", quantity=3, exit_session=date(2026, 8, 19),
                    fill_price=HIGH_PRECISION, execution_cost=Decimal("0"),
                    gross_proceeds=gross, net_proceeds=Decimal("999"),
                    settlement_id="S", settlement_session=date(2026, 8, 21),
                    provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
                    exit_reason=HistoricalExitReason.EXTERNAL_SCHEDULED,
                )


def test_project_entries_ordinary_fixture_unchanged() -> None:
    run = _entry_run(fill_price=Decimal("100"), quantity=10)
    entries = project_entries(run)
    assert entries[0].cost_basis == Decimal("1000")


def test_project_exits_ordinary_fixture_unchanged() -> None:
    run = _exit_run(exit_fill_price=Decimal("110"), quantity=10)
    exits = project_exits(run)
    assert exits[0].gross_proceeds == Decimal("1100")
    assert exits[0].net_proceeds == Decimal("1100")


def test_source_payload_fingerprint_of_exit_record_deterministic_across_precision() -> None:
    fingerprints = []
    for prec in (28, 60, 200):
        with localcontext() as ctx:
            ctx.prec = prec
            run = _exit_run(exit_fill_price=HIGH_PRECISION)
            exit_record = project_exits(run)[0]
            fingerprints.append(
                compute_source_payload_fingerprint(
                    source_type="test_exit_record", payload=exit_record
                )
            )
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_project_entries_no_dividend_attribution_semantics_change() -> None:
    run = _entry_run(fill_price=HIGH_PRECISION)
    entries = project_entries(run)
    assert "ordinary_dividend_income" not in type(entries[0]).model_fields


# ---------------------------------------------------------------------------
# Cross-layer consistency (Section 14)
# ---------------------------------------------------------------------------


def test_sell_net_proceeds_flows_unchanged_into_settlement_and_maturation() -> None:
    sold = _sold(asset_id="NORGATE:840", fill_price=HIGH_PRECISION)
    pending = sold.pending_settlements[0]
    matured = PortfolioTransitionEngine.transition(
        sold, pending.settlement_session, ()
    ).resulting_state
    assert matured.settled_cash == add_exact_decimal(sold.settled_cash, pending.amount)
    assert pending.amount == exact_decimal_times_int(HIGH_PRECISION, 3)


def test_project_exits_agrees_with_authoritative_pending_settlement_amount() -> None:
    # Proves Phase 15D's project_exits derives the same authoritative
    # execution economics as Phase 13's own PendingSettlement.amount,
    # without creating a second, independently-diverging accounting source.
    run = _exit_run(exit_fill_price=HIGH_PRECISION, quantity=3)
    exits = project_exits(run)
    sell_session_result = run.session_results[-1]
    pending = sell_session_result.state_transition_result.ledger_entries[0]
    assert exits[0].net_proceeds == pending.pending_cash_delta
    assert exits[0].gross_proceeds == exact_decimal_times_int(HIGH_PRECISION, 3)
