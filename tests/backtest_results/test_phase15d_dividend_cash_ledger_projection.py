"""Slice 9 acceptance tests: Phase 15D authoritative dividend
`HistoricalCashLedgerRow` projection.

Covers the frozen Ordinary Dividend Amendment v0.7's strict 1:1 projection
of authoritative Phase 13 `DIVIDEND_APPLIED` ledger rows
(`portfolio_dividend_events.DividendLedgerEntry`) into the Phase 15D
`HistoricalCashLedgerRow` cash ledger (OD-16.2, OD-20.4, OD-20.5,
OD-21.1/21.2/21.3). This slice is projection-only: no trade grouping,
`ordinary_dividend_income`, `trade_total_pnl`, carried-in completeness, or
portfolio-scope P&L identity is implemented or tested here.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    HistoricalBacktestResultValidationError,
    HistoricalCashLedgerRow,
    compute_source_payload_fingerprint,
    project_cash_ledger,
)
from stock_swing_d1.backtest_results.hashing import compute_content_fingerprint
from stock_swing_d1.backtester.models import (
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionResult,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    DividendApplicationStatus,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine

from tests.backtest_results.conftest import (
    SOURCE_DECISION_INTERVAL,
    source_decision_time,
)
from tests.backtest_results.test_phase15d_dividend_source_validation import (
    ASSET_A,
    ASSET_B,
    T_SESSION,
    X_SESSION,
    _ranking,
    build_applied_and_noop_run,
    make_evidence,
    make_run_evidence,
    make_session_evidence,
)
from tests.portfolio.test_phase13_dividend_application import (
    _seed_prior_dividend_application,
    opened_state,
)


def _dividend_rows(
    run: HistoricalBacktestRunResult,
) -> tuple[HistoricalCashLedgerRow, ...]:
    return tuple(
        row
        for row in project_cash_ledger(run)
        if row.ledger_event_type is PortfolioLedgerEventType.DIVIDEND_APPLIED
    )


def _build_sell_on_x_run() -> HistoricalBacktestRunResult:
    """T: BUY 10 @ A. X: full-exit SELL of A plus a dividend on A.

    OD-9.4: a sell on X retains entitlement, so the dividend must still
    attribute to the trade that held Q_T (BUY-1).
    """

    initial = PortfolioState(settled_cash=Decimal("100000"))
    buy = PortfolioExecutionEvent(
        execution_id="BUY-1", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    opened = t_result.resulting_state

    sell = PortfolioExecutionEvent(
        execution_id="SELL-1", source_order_id="O-2", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=10,
        fill_price=Decimal("120"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-1", settlement_session=date(2026, 8, 20),
    )
    evidence = make_evidence(event_id="D:SELLONX", security_id=ASSET_A)
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


def _build_sell_and_reentry_run() -> HistoricalBacktestRunResult:
    """T: BUY 10 @ A. X: SELL 10 @ A (old trade) + re-entry BUY 5 @ A (new
    trade) plus a dividend on A, which must remain attributed to the OLD
    trade (BUY-1), never the new re-entry trade (BUY-2)."""

    initial = PortfolioState(settled_cash=Decimal("100000"))
    buy = PortfolioExecutionEvent(
        execution_id="BUY-1", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    opened = t_result.resulting_state

    sell = PortfolioExecutionEvent(
        execution_id="SELL-1", source_order_id="O-2", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=10,
        fill_price=Decimal("120"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-1", settlement_session=date(2026, 8, 20),
    )
    reentry = PortfolioExecutionEvent(
        execution_id="BUY-2", source_order_id="O-3", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=5,
        fill_price=Decimal("120"), execution_cost=Decimal("1"),
    )
    evidence = make_evidence(event_id="D:OLDTRADE", security_id=ASSET_A)
    x_result = PortfolioTransitionEngine.transition(
        opened, X_SESSION, (sell, reentry), (evidence,)
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
        ordered_execution_events=(sell, reentry),
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


def _build_buy_on_x_no_op_run() -> HistoricalBacktestRunResult:
    """A BUY made ON X (unheld before X) must not receive the dividend."""

    initial = PortfolioState(settled_cash=Decimal("100000"))
    buy_on_x = PortfolioExecutionEvent(
        execution_id="BUY-ON-X", source_order_id="O-1", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    evidence = make_evidence(event_id="D:BUYONX", security_id=ASSET_A)
    x_result = PortfolioTransitionEngine.transition(
        initial, X_SESSION, (buy_on_x,), (evidence,)
    )
    final = x_result.resulting_state
    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy_on_x,),
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


def _build_replayed_run() -> HistoricalBacktestRunResult:
    """A dividend already applied in a prior (unrepresented) run must
    project zero NEW dividend cash-ledger rows when its evidence is
    resupplied against a carried-in state that already recorded it."""

    state, buy = opened_state(
        cash=Decimal("100000"), session=T_SESSION, execution_id="BUY-1",
        asset_id=ASSET_A, quantity=10, fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    evidence = make_evidence(event_id="D:REPLAY", security_id=ASSET_A)
    seeded = _seed_prior_dividend_application(
        state, evidence, q_t=10, trade_id=buy.execution_id
    )
    x_result = PortfolioTransitionEngine.transition(
        seeded, X_SESSION, (), (evidence,)
    )
    final = x_result.resulting_state
    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(seeded),
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
        initial_state=seeded, final_state=final,
        session_results=(x_session,),
        initial_state_fingerprint=hash_portfolio_state(seeded),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _build_two_asset_dividend_run(
    *, reverse_evidence_order: bool = False
) -> HistoricalBacktestRunResult:
    """T: BUY A and B. X: dividends APPLIED on both A and B."""

    initial = PortfolioState(settled_cash=Decimal("100000"))
    buy_a = PortfolioExecutionEvent(
        execution_id="BUY-A", source_order_id="O-A", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    buy_b = PortfolioExecutionEvent(
        execution_id="BUY-B", source_order_id="O-B", session=T_SESSION,
        asset_id=ASSET_B, side=ExecutionSide.BUY, quantity=4,
        fill_price=Decimal("50"), execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(
        initial, T_SESSION, (buy_a, buy_b)
    )
    opened = t_result.resulting_state

    evidence_a = make_evidence(event_id="D:A", security_id=ASSET_A)
    evidence_b = make_evidence(event_id="D:B", security_id=ASSET_B)
    supplied = (
        (evidence_b, evidence_a) if reverse_evidence_order else (evidence_a, evidence_b)
    )
    x_result = PortfolioTransitionEngine.transition(
        opened, X_SESSION, (), supplied
    )
    final = x_result.resulting_state

    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy_a, buy_b),
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
        make_session_evidence(
            session=X_SESSION, distribution_events=supplied
        ),
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


# ---------------------------------------------------------------------------
# A. Core 1:1 projection
# ---------------------------------------------------------------------------


def test_one_applied_dividend_projects_to_exactly_one_row() -> None:
    run = build_applied_and_noop_run()
    dividend_rows = _dividend_rows(run)
    assert len(dividend_rows) == 1
    assert (
        dividend_rows[0].ledger_event_type
        is PortfolioLedgerEventType.DIVIDEND_APPLIED
    )


def test_two_dividend_rows_on_distinct_assets_project_to_two_rows() -> None:
    run = _build_two_asset_dividend_run()
    dividend_rows = _dividend_rows(run)
    assert len(dividend_rows) == 2
    assert {row.security_id for row in dividend_rows} == {ASSET_A, ASSET_B}


def test_no_op_dividend_contributes_zero_rows() -> None:
    run = build_applied_and_noop_run()
    dividend_rows = _dividend_rows(run)
    # Only ASSET_A (APPLIED); ASSET_B (NO_OP_NOT_ENTITLED) contributes none.
    assert {row.security_id for row in dividend_rows} == {ASSET_A}


def test_one_upstream_dividend_row_cannot_project_twice() -> None:
    run = build_applied_and_noop_run()
    rows = project_cash_ledger(run)
    source_event_ids = [row.source_event_id for row in rows]
    assert len(source_event_ids) == len(set(source_event_ids))


def test_unsupported_ledger_event_type_still_fails_closed() -> None:
    with pytest.raises(Exception):
        HistoricalCashLedgerRow(
            session=X_SESSION,
            sequence_in_session=0,
            ledger_event_type="NOT_A_REAL_EVENT_TYPE",
            source_event_id="x",
            settled_cash_delta=Decimal("1"),
            pending_cash_delta=Decimal("0"),
            settled_cash_after=Decimal("1"),
            state_hash_before="a" * 64,
            state_hash_after="b" * 64,
            source_payload_fingerprint="c" * 64,
        )


# ---------------------------------------------------------------------------
# B. Field / source binding
# ---------------------------------------------------------------------------


def test_dividend_row_preserves_session_security_and_attribution() -> None:
    run = _build_sell_on_x_run()
    entry = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    row = _dividend_rows(run)[0]

    assert row.session == X_SESSION == entry.session
    assert row.security_id == ASSET_A == entry.asset_id
    assert row.attribution_trade_id == "BUY-1" == entry.attribution_trade_id
    assert row.source_event_id == entry.application_id
    assert row.settled_cash_delta == entry.settled_cash_delta
    assert row.settled_cash_after == entry.settled_cash_after
    assert row.state_hash_before == entry.state_hash_before
    assert row.state_hash_after == entry.state_hash_after
    assert row.settlement_id is None
    assert row.settlement_session is None
    assert row.source_order_id is None
    assert row.pending_cash_delta == Decimal("0")


def test_dividend_row_settled_cash_delta_is_exact_q_t_times_d_h() -> None:
    run = _build_sell_on_x_run()
    entry = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    row = _dividend_rows(run)[0]

    assert row.settled_cash_delta == entry.q_t * entry.d_h
    assert row.settled_cash_delta == entry.gross_cash_amount
    assert row.settled_cash_delta > 0
    assert type(row.settled_cash_delta) is Decimal


def test_dividend_row_source_payload_fingerprint_uses_dedicated_source_type() -> (
    None
):
    run = _build_sell_on_x_run()
    entry = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    row = _dividend_rows(run)[0]

    assert row.source_payload_fingerprint == compute_source_payload_fingerprint(
        source_type="dividend_ledger_entry", payload=entry
    )
    assert row.source_payload_fingerprint != entry.source_payload_sha256


def test_dividend_row_fingerprint_changes_when_entry_content_changes() -> None:
    run = _build_sell_on_x_run()
    entry = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    changed_entry = entry.model_copy(
        update={"canonical_distribution_event_id": "D:DIFFERENT"}
    )
    original = compute_source_payload_fingerprint(
        source_type="dividend_ledger_entry", payload=entry
    )
    changed = compute_source_payload_fingerprint(
        source_type="dividend_ledger_entry", payload=changed_entry
    )
    assert original != changed


# ---------------------------------------------------------------------------
# C. Status / cardinality
# ---------------------------------------------------------------------------


def test_applied_status_yields_exactly_one_row() -> None:
    run = _build_sell_on_x_run()
    assert len(_dividend_rows(run)) == 1


def test_replayed_status_yields_zero_new_rows() -> None:
    run = _build_replayed_run()
    x_result = run.session_results[0].state_transition_result
    assert x_result.dividend_outcomes[0].status is DividendApplicationStatus.REPLAYED
    assert x_result.dividend_ledger_entries == ()
    assert _dividend_rows(run) == ()


def test_no_op_not_entitled_yields_zero_rows() -> None:
    run = _build_buy_on_x_no_op_run()
    x_result = run.session_results[0].state_transition_result
    assert (
        x_result.dividend_outcomes[0].status
        is DividendApplicationStatus.NO_OP_NOT_ENTITLED
    )
    assert _dividend_rows(run) == ()


def test_no_zero_cash_placeholder_row_for_replayed_or_no_op() -> None:
    for run in (_build_replayed_run(), _build_buy_on_x_no_op_run()):
        rows = project_cash_ledger(run)
        assert all(
            row.ledger_event_type is not PortfolioLedgerEventType.DIVIDEND_APPLIED
            for row in rows
        )


# ---------------------------------------------------------------------------
# D. Same-X chronology
# ---------------------------------------------------------------------------


def test_sell_on_x_dividend_attributes_to_old_trade_and_follows_sell() -> None:
    run = _build_sell_on_x_run()
    rows = project_cash_ledger(run)
    x_rows = [row for row in rows if row.session == X_SESSION]
    dividend_row = next(
        row
        for row in x_rows
        if row.ledger_event_type is PortfolioLedgerEventType.DIVIDEND_APPLIED
    )
    sell_row = next(
        row
        for row in x_rows
        if row.ledger_event_type is PortfolioLedgerEventType.SELL_APPLIED
    )
    assert dividend_row.attribution_trade_id == "BUY-1"
    assert dividend_row.sequence_in_session > sell_row.sequence_in_session


def test_sell_and_reentry_dividend_stays_with_old_trade_not_new() -> None:
    run = _build_sell_and_reentry_run()
    rows = project_cash_ledger(run)
    x_rows = [row for row in rows if row.session == X_SESSION]
    dividend_row = next(
        row
        for row in x_rows
        if row.ledger_event_type is PortfolioLedgerEventType.DIVIDEND_APPLIED
    )
    buy_row = next(
        row
        for row in x_rows
        if row.ledger_event_type is PortfolioLedgerEventType.BUY_APPLIED
    )
    assert dividend_row.attribution_trade_id == "BUY-1"
    assert dividend_row.attribution_trade_id != "BUY-2"
    assert dividend_row.sequence_in_session > buy_row.sequence_in_session
    new_position = run.final_state.open_positions[0]
    assert new_position.entry_execution_id == "BUY-2"


def test_buy_on_x_produces_no_dividend_row() -> None:
    run = _build_buy_on_x_no_op_run()
    assert _dividend_rows(run) == ()


# ---------------------------------------------------------------------------
# E. Ordering
# ---------------------------------------------------------------------------


def test_dividend_rows_follow_settlement_sell_buy_within_session() -> None:
    run = _build_sell_and_reentry_run()
    rows = project_cash_ledger(run)
    x_rows = [row for row in rows if row.session == X_SESSION]
    # Canonical OD-10 chronology: SETTLEMENT -> SELL -> BUY -> DIVIDEND.
    rank = {
        PortfolioLedgerEventType.SETTLEMENT_APPLIED: 0,
        PortfolioLedgerEventType.SELL_APPLIED: 1,
        PortfolioLedgerEventType.BUY_APPLIED: 2,
        PortfolioLedgerEventType.DIVIDEND_APPLIED: 3,
    }
    ranks = [rank[row.ledger_event_type] for row in x_rows]
    assert ranks == sorted(ranks)
    assert [row.sequence_in_session for row in x_rows] == list(range(len(x_rows)))


def test_two_dividend_rows_on_one_x_ordered_by_asset_then_event_id() -> None:
    run = _build_two_asset_dividend_run()
    dividend_rows = _dividend_rows(run)
    assert [row.security_id for row in dividend_rows] == [ASSET_A, ASSET_B]


def test_reversed_supplied_evidence_order_yields_same_canonical_projection() -> (
    None
):
    forward = _dividend_rows(_build_two_asset_dividend_run())
    reversed_input = _dividend_rows(
        _build_two_asset_dividend_run(reverse_evidence_order=True)
    )
    forward_keys = [
        (row.security_id, row.sequence_in_session, row.settled_cash_delta)
        for row in forward
    ]
    reversed_keys = [
        (row.security_id, row.sequence_in_session, row.settled_cash_delta)
        for row in reversed_input
    ]
    assert forward_keys == reversed_keys


# ---------------------------------------------------------------------------
# F. Source-validation prerequisite
# ---------------------------------------------------------------------------


def test_tampered_attribution_trade_id_fails_before_projection() -> None:
    run = build_applied_and_noop_run()
    x_session = next(s for s in run.session_results if s.session == X_SESSION)
    tampered_transition = x_session.state_transition_result.model_copy(
        update={
            "dividend_ledger_entries": tuple(
                entry.model_copy(update={"attribution_trade_id": "SOME-OTHER-TRADE"})
                if entry.asset_id == ASSET_A
                else entry
                for entry in x_session.state_transition_result.dividend_ledger_entries
            )
        }
    )
    tampered_x_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = run.model_copy(
        update={
            "session_results": tuple(
                tampered_x_session if s.session == X_SESSION else s
                for s in run.session_results
            )
        }
    )
    with pytest.raises(HistoricalBacktestResultValidationError):
        project_cash_ledger(tampered_run)


def test_row_without_supporting_evidence_cannot_project() -> None:
    run = build_applied_and_noop_run()
    truncated_evidence = run.dividend_run_evidence.model_copy(
        update={
            "session_evidence": tuple(
                item.model_copy(
                    update={"distribution_events": (), "distribution_coverage": item.distribution_coverage}
                )
                if item.session == X_SESSION
                else item
                for item in run.dividend_run_evidence.session_evidence
            )
        }
    )
    tampered_run = run.model_copy(
        update={"dividend_run_evidence": truncated_evidence}
    )
    with pytest.raises(HistoricalBacktestResultValidationError):
        project_cash_ledger(tampered_run)


# ---------------------------------------------------------------------------
# G. Schema / version / fingerprint
# ---------------------------------------------------------------------------


def test_cash_ledger_row_schema_version_is_v0_2() -> None:
    run = build_applied_and_noop_run()
    row = _dividend_rows(run)[0]
    assert row.schema_version == "historical_cash_ledger_row.v0.2"


def test_content_fingerprint_changes_when_dividend_row_is_added() -> None:
    run = build_applied_and_noop_run()
    all_rows = project_cash_ledger(run)
    execution_only = tuple(
        row
        for row in all_rows
        if row.ledger_event_type is not PortfolioLedgerEventType.DIVIDEND_APPLIED
    )
    with_dividend = compute_content_fingerprint(
        artifact_name="cash_ledger", rows_or_value=all_rows
    )
    without_dividend = compute_content_fingerprint(
        artifact_name="cash_ledger", rows_or_value=execution_only
    )
    assert with_dividend != without_dividend


def test_non_dividend_row_rejects_attribution_trade_id() -> None:
    with pytest.raises(Exception):
        HistoricalCashLedgerRow(
            session=X_SESSION,
            sequence_in_session=0,
            ledger_event_type=PortfolioLedgerEventType.BUY_APPLIED,
            source_event_id="exec-1",
            settled_cash_delta=Decimal("-100"),
            pending_cash_delta=Decimal("0"),
            settled_cash_after=Decimal("900"),
            state_hash_before="a" * 64,
            state_hash_after="b" * 64,
            source_payload_fingerprint="c" * 64,
            attribution_trade_id="should-not-be-allowed",
        )


def test_dividend_row_requires_attribution_trade_id() -> None:
    with pytest.raises(Exception):
        HistoricalCashLedgerRow(
            session=X_SESSION,
            sequence_in_session=0,
            ledger_event_type=PortfolioLedgerEventType.DIVIDEND_APPLIED,
            source_event_id="app-1",
            settled_cash_delta=Decimal("100"),
            pending_cash_delta=Decimal("0"),
            settled_cash_after=Decimal("1100"),
            state_hash_before="a" * 64,
            state_hash_after="b" * 64,
            source_payload_fingerprint="c" * 64,
        )


def test_dividend_row_rejects_settlement_identity() -> None:
    with pytest.raises(Exception):
        HistoricalCashLedgerRow(
            session=X_SESSION,
            sequence_in_session=0,
            ledger_event_type=PortfolioLedgerEventType.DIVIDEND_APPLIED,
            source_event_id="app-1",
            settled_cash_delta=Decimal("100"),
            pending_cash_delta=Decimal("0"),
            settled_cash_after=Decimal("1100"),
            settlement_id="s-1",
            settlement_session=X_SESSION,
            state_hash_before="a" * 64,
            state_hash_after="b" * 64,
            source_payload_fingerprint="c" * 64,
            attribution_trade_id="BUY-1",
        )


def test_dividend_row_rejects_nonzero_pending_cash_delta() -> None:
    with pytest.raises(Exception):
        HistoricalCashLedgerRow(
            session=X_SESSION,
            sequence_in_session=0,
            ledger_event_type=PortfolioLedgerEventType.DIVIDEND_APPLIED,
            source_event_id="app-1",
            settled_cash_delta=Decimal("100"),
            pending_cash_delta=Decimal("5"),
            settled_cash_after=Decimal("1100"),
            state_hash_before="a" * 64,
            state_hash_after="b" * 64,
            source_payload_fingerprint="c" * 64,
            attribution_trade_id="BUY-1",
        )


# ---------------------------------------------------------------------------
# H. Existing BUY/SELL/SETTLEMENT projection regression
# ---------------------------------------------------------------------------


def test_non_dividend_run_produces_no_dividend_rows(source_run_bundle) -> None:
    run, _manifest = source_run_bundle
    assert _dividend_rows(run) == ()


def test_non_dividend_run_execution_rows_unchanged(source_run_bundle) -> None:
    run, _manifest = source_run_bundle
    rows = project_cash_ledger(run)
    authoritative_entries = tuple(
        entry
        for session in run.session_results
        for entry in session.state_transition_result.ledger_entries
    )
    assert len(rows) == len(authoritative_entries)
    for row, entry in zip(rows, authoritative_entries, strict=True):
        assert row.sequence_in_session == entry.sequence_in_session
        assert row.attribution_trade_id is None
