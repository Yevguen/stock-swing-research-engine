"""Slice 10 acceptance tests: Phase 15D trade-level ordinary-dividend
grouping, `ordinary_dividend_income`, event count, attribution
fingerprint, and carried-in completeness.

Covers the frozen Ordinary Dividend Amendment v0.7's OD-16 trade-grouping
contract: `HistoricalClosedTradeRecord` gains `ordinary_dividend_income`,
`ordinary_dividend_event_count`, `dividend_attribution_completeness`, and
`ordinary_dividend_attribution_fingerprint`, grouped exclusively from the
Slice-9 projected `HistoricalCashLedgerRow` tuple (OD-16.2), never from
Phase-13 objects directly (OD-16.3). `trade_total_pnl` and
portfolio-scope dividend P&L remain explicitly deferred.
"""

from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal, localcontext

import pytest

from stock_swing_d1.backtest_results import (
    DividendAttributionCompleteness,
    HistoricalBacktestResultValidationError,
    HistoricalCashLedgerRow,
    HistoricalClosedTradeRecord,
    compute_dividend_attribution_fingerprint,
    project_cash_ledger,
    project_closed_trades,
)
from stock_swing_d1.backtest_results import derivation as derivation_module
from stock_swing_d1.backtest_results.derivation import (
    _exact_dividend_income,
    _group_dividend_cash_ledger_rows_by_trade,
)
from stock_swing_d1.backtest_results.persistence_schema import (
    decode_trade_row,
    encode_trade_row,
)
from stock_swing_d1.backtester.models import (
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionResult,
)
from stock_swing_d1.data.ordinary_dividend_accounting import (
    CANONICAL_DIVIDEND_CURRENCY,
    CanonicalDividendAccountingEvidence,
    build_canonical_dividend_accounting_evidence,
    build_dividend_calendar_resolution_proof,
    build_ordinary_cash_classification_proof,
)
from stock_swing_d1.data.ordinary_dividend_normalization import (
    Gate3DividendNormalizationInputs,
    normalize_gate3_dividend,
)
from stock_swing_d1.data.ordinary_dividend_run_evidence import (
    build_dividend_aware_run_evidence,
)
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    ExecutionSide,
    PortfolioEventKind,
    PortfolioExecutionEvent,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
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
    CALENDAR_POLICY_ID,
    CALENDAR_POLICY_VERSION,
    CALENDAR_SOURCE_ID,
    CLASSIFICATION_CONTRACT_ID,
    SNAPSHOT_FINGERPRINT,
    T_SESSION,
    X_SESSION,
    _ranking,
    make_evidence,
    make_run_evidence,
    make_session_evidence,
)

X2_SESSION = date(2026, 8, 20)


def _make_custom_evidence(
    *,
    event_id: str,
    security_id: str = ASSET_A,
    entitlement_session: date = T_SESSION,
    ex_session: date = X_SESSION,
    d_capitalspecial: Decimal = Decimal("2"),
    unadjusted_close_t: Decimal = Decimal("3"),
    close_capital_t: Decimal = Decimal("4"),
) -> CanonicalDividendAccountingEvidence:
    inputs = Gate3DividendNormalizationInputs(
        entitlement_session=entitlement_session,
        d_capitalspecial=d_capitalspecial,
        unadjusted_close_t=unadjusted_close_t,
        close_capital_t=close_capital_t,
    )
    result = normalize_gate3_dividend(inputs)
    classification = build_ordinary_cash_classification_proof(
        canonical_distribution_event_id=event_id,
        classification_contract_id=CLASSIFICATION_CONTRACT_ID,
        upstream_source_evidence_fingerprint="a" * 64,
    )
    calendar = build_dividend_calendar_resolution_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        entitlement_session=entitlement_session,
        ex_session=ex_session,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint="b" * 64,
    )
    return build_canonical_dividend_accounting_evidence(
        canonical_distribution_event_id=event_id,
        canonical_security_id=security_id,
        normalization_inputs=inputs,
        normalization_result=result,
        classification_proof=classification,
        calendar_resolution_proof=calendar,
        currency=CANONICAL_DIVIDEND_CURRENCY,
        canonical_distribution_snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
    )


def _closed_trade_for(
    run: HistoricalBacktestRunResult, trade_id: str
) -> HistoricalClosedTradeRecord:
    return next(
        trade for trade in project_closed_trades(run) if trade.trade_id == trade_id
    )


def _build_two_closed_trades_with_dividends_run() -> HistoricalBacktestRunResult:
    """T: BUY A and B. X: dividends on both, then SELL both (full exit)."""

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

    evidence_a = _make_custom_evidence(event_id="D:A", security_id=ASSET_A)
    evidence_b = _make_custom_evidence(
        event_id="D:B", security_id=ASSET_B,
        d_capitalspecial=Decimal("3"), unadjusted_close_t=Decimal("5"),
        close_capital_t=Decimal("6"),
    )
    sell_a = PortfolioExecutionEvent(
        execution_id="SELL-A", source_order_id="O-C", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=10,
        fill_price=Decimal("110"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-A", settlement_session=X2_SESSION,
    )
    sell_b = PortfolioExecutionEvent(
        execution_id="SELL-B", source_order_id="O-D", session=X_SESSION,
        asset_id=ASSET_B, side=ExecutionSide.SELL, quantity=4,
        fill_price=Decimal("55"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-B", settlement_session=X2_SESSION,
    )
    x_result = PortfolioTransitionEngine.transition(
        opened, X_SESSION, (sell_a, sell_b), (evidence_a, evidence_b)
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
        ordered_execution_events=(sell_a, sell_b),
        state_transition_result=x_result, authoritative_state=final,
        ranking_snapshot=_ranking(X_SESSION),
    )
    session_evidence = (
        make_session_evidence(session=T_SESSION),
        make_session_evidence(
            session=X_SESSION, distribution_events=(evidence_a, evidence_b)
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


def _build_multi_session_dividend_run() -> HistoricalBacktestRunResult:
    """T: BUY 10@A. X1: dividend #1 (still held). X2: SELL 10@A + dividend
    #2 (same-X2 SELL retains entitlement, OD-9.4). One closed trade
    receives both dividends, in chronological order."""

    initial = PortfolioState(settled_cash=Decimal("100000"))
    buy = PortfolioExecutionEvent(
        execution_id="BUY-1", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    opened = t_result.resulting_state

    evidence_1 = _make_custom_evidence(
        event_id="D:MULTI1", entitlement_session=T_SESSION, ex_session=X_SESSION
    )
    x1_result = PortfolioTransitionEngine.transition(
        opened, X_SESSION, (), (evidence_1,)
    )
    after_x1 = x1_result.resulting_state

    sell = PortfolioExecutionEvent(
        execution_id="SELL-1", source_order_id="O-2", session=X2_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=10,
        fill_price=Decimal("120"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-1", settlement_session=date(2026, 8, 21),
    )
    evidence_2 = _make_custom_evidence(
        event_id="D:MULTI2", entitlement_session=X_SESSION, ex_session=X2_SESSION
    )
    x2_result = PortfolioTransitionEngine.transition(
        after_x1, X2_SESSION, (sell,), (evidence_2,)
    )
    final = x2_result.resulting_state

    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy,),
        state_transition_result=t_result, authoritative_state=opened,
        ranking_snapshot=_ranking(T_SESSION),
    )
    x1_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(opened),
        state_transition_result=x1_result, authoritative_state=after_x1,
        ranking_snapshot=_ranking(X_SESSION),
    )
    x2_session = HistoricalBacktestSessionResult(
        session=X2_SESSION, decision_time=source_decision_time(X2_SESSION),
        prior_state_fingerprint=hash_portfolio_state(after_x1),
        ordered_execution_events=(sell,),
        state_transition_result=x2_result, authoritative_state=final,
        ranking_snapshot=_ranking(X2_SESSION),
    )
    session_evidence = (
        make_session_evidence(session=T_SESSION),
        make_session_evidence(session=X_SESSION, distribution_events=(evidence_1,)),
        make_session_evidence(session=X2_SESSION, distribution_events=(evidence_2,)),
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(t_session, x1_session, x2_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _build_carried_in_closed_trade_run(
    *, with_dividend: bool
) -> HistoricalBacktestRunResult:
    """A carried-in position (opened before this run) closes via SELL
    within the run, optionally receiving one in-run dividend on X."""

    initial = PortfolioState(
        settled_cash=Decimal("100000"),
        open_positions=(
            OpenPosition(
                asset_id=ASSET_A, quantity=10, entry_session=date(2026, 8, 1),
                entry_price=Decimal("90"), entry_execution_id="CARRIED-BUY-1",
                entry_execution_cost=Decimal("1"), cost_basis=Decimal("901"),
            ),
        ),
        applied_events=(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id="CARRIED-BUY-1",
                payload_sha256="a" * 64,
            ),
        ),
        as_of_session=date(2026, 8, 1), state_version=1,
    )

    if with_dividend:
        evidence = make_evidence(event_id="D:CARRIEDIN", security_id=ASSET_A)
        x_result = PortfolioTransitionEngine.transition(
            initial, X_SESSION, (), (evidence,)
        )
        session_evidence_x = make_session_evidence(
            session=X_SESSION, distribution_events=(evidence,)
        )
    else:
        x_result = PortfolioTransitionEngine.transition(initial, X_SESSION, ())
        session_evidence_x = make_session_evidence(session=X_SESSION)
    after_x = x_result.resulting_state

    sell = PortfolioExecutionEvent(
        execution_id="SELL-CARRIED", source_order_id="O-1", session=X2_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=10,
        fill_price=Decimal("120"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-CARRIED", settlement_session=date(2026, 8, 21),
    )
    x2_result = PortfolioTransitionEngine.transition(after_x, X2_SESSION, (sell,))
    final = x2_result.resulting_state

    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        state_transition_result=x_result, authoritative_state=after_x,
        ranking_snapshot=_ranking(X_SESSION),
    )
    x2_session = HistoricalBacktestSessionResult(
        session=X2_SESSION, decision_time=source_decision_time(X2_SESSION),
        prior_state_fingerprint=hash_portfolio_state(after_x),
        ordered_execution_events=(sell,),
        state_transition_result=x2_result, authoritative_state=final,
        ranking_snapshot=_ranking(X2_SESSION),
    )
    session_evidence = (session_evidence_x, make_session_evidence(session=X2_SESSION))
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(x_session, x2_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _build_legacy_carried_in_closed_trade_run() -> HistoricalBacktestRunResult:
    """A carried-in position closes via SELL within a legacy (non-
    dividend-aware) run: no dividend evidence, no coverage, no contiguity
    proof at all -- `dividend_run_evidence is None`."""

    initial = PortfolioState(
        settled_cash=Decimal("100000"),
        open_positions=(
            OpenPosition(
                asset_id=ASSET_A, quantity=10, entry_session=date(2026, 8, 1),
                entry_price=Decimal("90"), entry_execution_id="CARRIED-BUY-1",
                entry_execution_cost=Decimal("1"), cost_basis=Decimal("901"),
            ),
        ),
        applied_events=(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id="CARRIED-BUY-1",
                payload_sha256="a" * 64,
            ),
        ),
        as_of_session=date(2026, 8, 1), state_version=1,
    )
    x_result = PortfolioTransitionEngine.transition(initial, X_SESSION, ())
    after_x = x_result.resulting_state

    sell = PortfolioExecutionEvent(
        execution_id="SELL-CARRIED", source_order_id="O-1", session=X2_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=10,
        fill_price=Decimal("120"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-CARRIED", settlement_session=date(2026, 8, 21),
    )
    x2_result = PortfolioTransitionEngine.transition(after_x, X2_SESSION, (sell,))
    final = x2_result.resulting_state

    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        state_transition_result=x_result, authoritative_state=after_x,
        ranking_snapshot=_ranking(X_SESSION),
    )
    x2_session = HistoricalBacktestSessionResult(
        session=X2_SESSION, decision_time=source_decision_time(X2_SESSION),
        prior_state_fingerprint=hash_portfolio_state(after_x),
        ordered_execution_events=(sell,),
        state_transition_result=x2_result, authoritative_state=final,
        ranking_snapshot=_ranking(X2_SESSION),
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=initial, final_state=final,
        session_results=(x_session, x2_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _high_precision_row(
    *, trade_id: str = "T-1"
) -> tuple[HistoricalCashLedgerRow, int, Decimal]:
    """One dividend cash-ledger row whose amount needs far more than 28
    significant digits: D_H = 1/3 exactly at scale 38 (38 repeating
    threes), times a 7-digit Q_T. Returns (row, q_t, d_h)."""

    evidence = _make_custom_evidence(
        event_id="D:PRECISION",
        d_capitalspecial=Decimal("1"),
        unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("3"),
    )
    initial = PortfolioState(
        settled_cash=Decimal("0"),
        open_positions=(
            OpenPosition(
                asset_id=ASSET_A, quantity=9_999_999, entry_session=T_SESSION,
                entry_price=Decimal("1"), entry_execution_id=trade_id,
                entry_execution_cost=Decimal("0"), cost_basis=Decimal("9999999"),
            ),
        ),
        applied_events=(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id=trade_id,
                payload_sha256="a" * 64,
            ),
        ),
        as_of_session=T_SESSION, state_version=1,
    )
    x_result = PortfolioTransitionEngine.transition(
        initial, X_SESSION, (), (evidence,)
    )
    entry = x_result.dividend_ledger_entries[0]
    return HistoricalCashLedgerRow(
        session=entry.session,
        sequence_in_session=0,
        ledger_event_type=entry.event_type,
        source_event_id=entry.application_id,
        settled_cash_delta=entry.settled_cash_delta,
        pending_cash_delta=Decimal("0"),
        settled_cash_after=entry.settled_cash_after,
        state_hash_before=entry.state_hash_before,
        state_hash_after=entry.state_hash_after,
        source_payload_fingerprint="d" * 64,
        attribution_trade_id=trade_id,
    ), entry.q_t, entry.d_h


# ---------------------------------------------------------------------------
# Normative source
# ---------------------------------------------------------------------------


def test_grouping_helper_never_references_phase13_dividend_objects() -> None:
    source = inspect.getsource(_group_dividend_cash_ledger_rows_by_trade)
    for forbidden in (
        "DividendLedgerEntry",
        "CanonicalDividendAccountingEvidence",
        "DividendApplicationOutcome",
        ".q_t",
        ".d_h",
        "norgatedata",
        "calendar",
    ):
        assert forbidden not in source


def test_grouping_module_never_recomputes_gate3_or_application_identity() -> None:
    source = inspect.getsource(derivation_module)
    assert "normalize_gate3_dividend" not in source
    assert "compute_dividend_application_id" not in source
    assert "compute_gross_dividend_cash" not in source


def test_only_dividend_applied_rows_contribute_to_grouping() -> None:
    run = _build_sell_on_x_run()
    cash_ledger = project_cash_ledger(run)
    non_dividend = tuple(
        row
        for row in cash_ledger
        if row.ledger_event_type is not PortfolioLedgerEventType.DIVIDEND_APPLIED
    )
    assert non_dividend  # BUY/SELL/SETTLEMENT rows exist in this fixture
    trade = _closed_trade_for(run, "BUY-1")
    assert trade.ordinary_dividend_event_count == 1


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_one_dividend_row_gives_exact_income_for_one_closed_trade() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    dividend_row = next(
        row
        for row in project_cash_ledger(run)
        if row.ledger_event_type is PortfolioLedgerEventType.DIVIDEND_APPLIED
    )
    assert trade.ordinary_dividend_income == dividend_row.settled_cash_delta
    assert trade.ordinary_dividend_event_count == 1
    assert trade.dividend_attribution_completeness == (
        DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
    )


def test_two_dividend_rows_across_sessions_sum_exactly() -> None:
    run = _build_multi_session_dividend_run()
    trade = _closed_trade_for(run, "BUY-1")
    dividend_rows = tuple(
        row
        for row in project_cash_ledger(run)
        if row.ledger_event_type is PortfolioLedgerEventType.DIVIDEND_APPLIED
    )
    assert len(dividend_rows) == 2
    expected = dividend_rows[0].settled_cash_delta + dividend_rows[1].settled_cash_delta
    assert trade.ordinary_dividend_income == expected
    assert trade.ordinary_dividend_event_count == 2


def test_two_trades_each_receive_only_their_own_dividend() -> None:
    run = _build_two_closed_trades_with_dividends_run()
    trade_a = _closed_trade_for(run, "BUY-A")
    trade_b = _closed_trade_for(run, "BUY-B")
    assert trade_a.ordinary_dividend_event_count == 1
    assert trade_b.ordinary_dividend_event_count == 1
    assert trade_a.ordinary_dividend_income != trade_b.ordinary_dividend_income
    assert trade_a.ordinary_dividend_attribution_fingerprint != (
        trade_b.ordinary_dividend_attribution_fingerprint
    )


def test_zero_dividend_complete_trade_has_exact_zero() -> None:
    run = _build_buy_on_x_no_op_run()
    # BUY-ON-X never closes (stays open); confirm the projection path
    # itself is dividend-free and crash-free with zero closed trades.
    assert project_closed_trades(run) == ()


def test_replayed_run_has_zero_closed_trades_and_does_not_crash() -> None:
    run = _build_replayed_run()
    assert project_closed_trades(run) == ()


# ---------------------------------------------------------------------------
# Decimal determinism
# ---------------------------------------------------------------------------


def test_high_precision_income_survives_ambient_precision_28_and_60() -> None:
    # `row.settled_cash_delta` is already Phase 13's own exact,
    # ambient-context-independent Q_T*D_H (built from integer coefficient
    # arithmetic in `compute_gross_dividend_cash`, never recomputed here).
    row, _q_t, _d_h = _high_precision_row()
    exact_expected = row.settled_cash_delta
    with localcontext() as ctx:
        ctx.prec = 28
        income_28 = _exact_dividend_income((row,))
    with localcontext() as ctx:
        ctx.prec = 60
        income_60 = _exact_dividend_income((row,))
    assert income_28 == exact_expected
    assert income_60 == exact_expected
    assert income_28 == income_60
    assert type(income_28) is Decimal


def test_high_precision_income_naive_accumulation_would_have_rounded() -> None:
    """Sanity check that this fixture actually exercises the >28-sig-fig
    risk `_exact_dividend_income` is built to avoid."""

    row, _q_t, _d_h = _high_precision_row()
    with localcontext() as ctx:
        ctx.prec = 28
        naive = Decimal("0") + row.settled_cash_delta
    assert naive != row.settled_cash_delta  # ordinary '+' under prec=28 rounded
    assert _exact_dividend_income((row,)) == row.settled_cash_delta  # ours did not


def test_high_precision_fingerprint_identical_across_precision_contexts() -> None:
    row, _q_t, _d_h = _high_precision_row(trade_id="T-2")
    grouped_rows = ((row.source_event_id, row.source_payload_fingerprint),)
    with localcontext() as ctx:
        ctx.prec = 28
        fingerprint_28 = compute_dividend_attribution_fingerprint(
            trade_id="T-2",
            completeness=DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME.value,
            dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
            canonical_distribution_snapshot_fingerprint=None,
            grouped_rows=grouped_rows,
        )
    with localcontext() as ctx:
        ctx.prec = 60
        fingerprint_60 = compute_dividend_attribution_fingerprint(
            trade_id="T-2",
            completeness=DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME.value,
            dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
            canonical_distribution_snapshot_fingerprint=None,
            grouped_rows=grouped_rows,
        )
    assert fingerprint_28 == fingerprint_60


def test_high_precision_trade_persistence_round_trip_is_exact() -> None:
    row, _q_t, _d_h = _high_precision_row(trade_id="T-3")
    trade = HistoricalClosedTradeRecord(
        trade_id="T-3", security_id=ASSET_A, quantity=9_999_999,
        carried_in=False, entry_execution_id="T-3", entry_session=T_SESSION,
        entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
        entry_cost_basis=Decimal("9999999"), exit_execution_id="EXIT-3",
        exit_session=X_SESSION, exit_fill_price=Decimal("1"),
        exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
        gross_exit_proceeds=Decimal("9999999"), net_exit_proceeds=Decimal("9999999"),
        realized_pnl=Decimal("0"),
        ordinary_dividend_income=row.settled_cash_delta,
        ordinary_dividend_event_count=1,
        dividend_attribution_completeness=(
            DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
        ),
        ordinary_dividend_attribution_fingerprint="e" * 64,
        # Slice 11 (OD-16.5a): COMPLETE_TRADE_LIFETIME requires
        # trade_total_pnl == realized_pnl + ordinary_dividend_income;
        # realized_pnl is zero here, so it equals the dividend income.
        trade_total_pnl=row.settled_cash_delta,
    )
    encoded = encode_trade_row(trade, 0)
    decoded = decode_trade_row(encoded)
    assert decoded == trade
    assert decoded.ordinary_dividend_income == row.settled_cash_delta
    assert type(decoded.ordinary_dividend_income) is Decimal


# ---------------------------------------------------------------------------
# Same-X re-entry
# ---------------------------------------------------------------------------


def test_same_x_sell_reentry_groups_dividend_to_old_trade_only() -> None:
    run = _build_sell_and_reentry_run()
    closed = project_closed_trades(run)
    assert len(closed) == 1
    old_trade = closed[0]
    assert old_trade.trade_id == "BUY-1"
    assert old_trade.trade_id == old_trade.entry_execution_id
    assert old_trade.ordinary_dividend_event_count == 1
    assert old_trade.ordinary_dividend_income > 0
    # BUY-2 (re-entry) is still open at run end, never a closed trade, and
    # therefore never receives any of BUY-1's dividend income.
    open_ids = {p.entry_execution_id for p in run.final_state.open_positions}
    assert open_ids == {"BUY-2"}


def test_same_x_sell_on_x_alone_groups_to_old_trade() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    assert trade.trade_id == "BUY-1" == trade.entry_execution_id
    assert trade.ordinary_dividend_event_count == 1


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def test_non_carried_in_closed_trade_is_complete_lifetime() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    assert trade.carried_in is False
    assert trade.dividend_attribution_completeness == (
        DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
    )


def _build_zero_dividend_trade_run(
    *, dividend_aware: bool, snapshot_fingerprint: str = SNAPSHOT_FINGERPRINT
) -> HistoricalBacktestRunResult:
    """One non-carried-in closed trade with zero dividend ledger rows.

    `dividend_aware=True` builds a run with affirmative empty per-session
    distribution coverage (OD-7.1's "known no applicable distribution"),
    the legitimate source for a `COMPLETE_TRADE_LIFETIME` zero.
    `dividend_aware=False` omits all dividend evidence entirely (a legacy
    run), which must NOT be able to reach the same conclusion merely from
    row absence.
    """

    initial = PortfolioState(settled_cash=Decimal("100000"))
    buy = PortfolioExecutionEvent(
        execution_id="BUY-ZERO", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    opened = t_result.resulting_state
    sell = PortfolioExecutionEvent(
        execution_id="SELL-ZERO", source_order_id="O-2", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=10,
        fill_price=Decimal("110"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-ZERO", settlement_session=X2_SESSION,
    )
    x_result = PortfolioTransitionEngine.transition(opened, X_SESSION, (sell,))
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
    if not dividend_aware:
        return HistoricalBacktestRunResult(
            decision_interval=SOURCE_DECISION_INTERVAL,
            initial_state=initial, final_state=final,
            session_results=(t_session, x_session),
            initial_state_fingerprint=hash_portfolio_state(initial),
            final_state_fingerprint=hash_portfolio_state(final),
        )
    session_evidence = (
        make_session_evidence(
            session=T_SESSION, snapshot_fingerprint=snapshot_fingerprint
        ),
        make_session_evidence(
            session=X_SESSION, snapshot_fingerprint=snapshot_fingerprint
        ),
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(
            session_evidence=session_evidence,
            snapshot_fingerprint=snapshot_fingerprint,
        ),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(t_session, x_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def test_complete_zero_dividend_trade_is_zero_and_complete() -> None:
    # Dividend-aware run, affirmative empty coverage on every session:
    # OD-7.1's "known no applicable distribution" legitimately proves the
    # complete-lifetime zero.
    dividend_aware_run = _build_zero_dividend_trade_run(dividend_aware=True)
    trade = _closed_trade_for(dividend_aware_run, "BUY-ZERO")
    # Slice 11 bumped the live dividend-aware projector's emitted version
    # to v0.3 (grouping + authoritative trade_total_pnl); see
    # test_phase15d_trade_total_pnl.py for the trade_total_pnl-specific
    # assertions.
    assert trade.schema_version == "historical_closed_trade.v0.3"
    assert trade.ordinary_dividend_income == Decimal("0")
    assert trade.ordinary_dividend_event_count == 0
    assert trade.dividend_attribution_completeness == (
        DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
    )
    assert trade.ordinary_dividend_attribution_fingerprint is not None
    # Deterministic even for an empty group.
    trade_again = _closed_trade_for(dividend_aware_run, "BUY-ZERO")
    assert trade.ordinary_dividend_attribution_fingerprint == (
        trade_again.ordinary_dividend_attribution_fingerprint
    )


def test_legacy_run_does_not_masquerade_as_dividend_aware_completeness() -> None:
    # The exact same trade shape, but with no dividend evidence at all
    # (dividend_run_evidence is None): no OD-15.7/OD-15.8 contiguity was
    # ever validated for this run, so it must not publish
    # COMPLETE_TRADE_LIFETIME + zero merely because no dividend rows
    # exist -- absence of evidence is not evidence of absence (OD-7.1).
    legacy_run = _build_zero_dividend_trade_run(dividend_aware=False)
    assert legacy_run.dividend_run_evidence is None
    trade = _closed_trade_for(legacy_run, "BUY-ZERO")
    assert trade.schema_version == "historical_closed_trade.v0.1"
    assert trade.ordinary_dividend_income is None
    assert trade.ordinary_dividend_event_count is None
    assert trade.dividend_attribution_completeness is None
    assert trade.ordinary_dividend_attribution_fingerprint is None
    # Realized economics are unaffected by dividend-awareness.
    assert trade.realized_pnl == trade.net_exit_proceeds - trade.entry_cost_basis


def test_dividend_aware_complete_zero_fingerprint_binds_distribution_snapshot() -> None:
    # OD-16.7 binds the canonical distribution snapshot into the
    # attribution fingerprint: changing only the snapshot must change the
    # published fingerprint, proving it is not a null/ignored input.
    run_a = _build_zero_dividend_trade_run(
        dividend_aware=True, snapshot_fingerprint="c" * 64
    )
    run_b = _build_zero_dividend_trade_run(
        dividend_aware=True, snapshot_fingerprint="d" * 64
    )
    trade_a = _closed_trade_for(run_a, "BUY-ZERO")
    trade_b = _closed_trade_for(run_b, "BUY-ZERO")
    assert trade_a.ordinary_dividend_attribution_fingerprint is not None
    assert trade_b.ordinary_dividend_attribution_fingerprint is not None
    assert trade_a.ordinary_dividend_attribution_fingerprint != (
        trade_b.ordinary_dividend_attribution_fingerprint
    )


def test_dividend_aware_run_without_snapshot_cannot_publish_closed_trades() -> None:
    # A dividend-aware run that (pathologically) supplies no session
    # evidence/snapshot at all must not be able to publish a
    # COMPLETE_TRADE_LIFETIME attribution unbound to a real distribution
    # snapshot (OD-16.7): fail closed instead of defaulting to None.
    run = _build_zero_dividend_trade_run(dividend_aware=True)
    empty_evidence = build_dividend_aware_run_evidence(
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=None,
        canonical_distribution_snapshot_fingerprint=None,
        session_evidence=(),
    )
    tampered_run = run.model_copy(update={"dividend_run_evidence": empty_evidence})
    with pytest.raises(HistoricalBacktestResultValidationError):
        project_closed_trades(tampered_run)


def test_carried_in_closed_trade_is_partial_pre_run_unknown() -> None:
    run = _build_carried_in_closed_trade_run(with_dividend=True)
    trade = _closed_trade_for(run, "CARRIED-BUY-1")
    assert trade.carried_in is True
    assert trade.dividend_attribution_completeness == (
        DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
    )
    assert trade.ordinary_dividend_event_count == 1
    assert trade.ordinary_dividend_income > 0


def test_carried_in_closed_trade_zero_in_run_dividends_stays_partial() -> None:
    run = _build_carried_in_closed_trade_run(with_dividend=False)
    trade = _closed_trade_for(run, "CARRIED-BUY-1")
    assert trade.carried_in is True
    assert trade.dividend_attribution_completeness == (
        DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
    )
    assert trade.ordinary_dividend_income == Decimal("0")
    assert trade.ordinary_dividend_event_count == 0


def test_legacy_carried_in_trade_does_not_masquerade_as_known_in_run_zero() -> None:
    # A legacy (non-dividend-aware) carried-in trade must not be labeled
    # PARTIAL_PRE_RUN_UNKNOWN with an authoritative in-run zero merely
    # because it has no dividend evidence at all -- the run never
    # affirmatively covered any session, so zero rows is not proof of
    # zero in-run dividends either (OD-7.1).
    run = _build_legacy_carried_in_closed_trade_run()
    assert run.dividend_run_evidence is None
    trade = _closed_trade_for(run, "CARRIED-BUY-1")
    assert trade.carried_in is True
    assert trade.schema_version == "historical_closed_trade.v0.1"
    assert trade.ordinary_dividend_income is None
    assert trade.ordinary_dividend_event_count is None
    assert trade.dividend_attribution_completeness is None
    assert trade.ordinary_dividend_attribution_fingerprint is None


def test_model_rejects_complete_status_for_carried_in_trade() -> None:
    with pytest.raises(Exception):
        HistoricalClosedTradeRecord(
            trade_id="X", security_id=ASSET_A, quantity=1, carried_in=True,
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
        )


def test_model_rejects_partial_status_for_non_carried_in_trade() -> None:
    with pytest.raises(Exception):
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
                DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
            ),
            ordinary_dividend_attribution_fingerprint="f" * 64,
        )


def test_model_rejects_income_count_mismatch() -> None:
    with pytest.raises(Exception):
        HistoricalClosedTradeRecord(
            trade_id="X", security_id=ASSET_A, quantity=1, carried_in=False,
            entry_execution_id="X", entry_session=T_SESSION,
            entry_fill_price=Decimal("1"), entry_execution_cost=Decimal("0"),
            entry_cost_basis=Decimal("1"), exit_execution_id="Y",
            exit_session=X_SESSION, exit_fill_price=Decimal("1"),
            exit_execution_cost=Decimal("0"), exit_reason="EXTERNAL_SCHEDULED",
            gross_exit_proceeds=Decimal("1"), net_exit_proceeds=Decimal("1"),
            realized_pnl=Decimal("0"), ordinary_dividend_income=Decimal("5"),
            ordinary_dividend_event_count=0,
            dividend_attribution_completeness=(
                DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
            ),
            ordinary_dividend_attribution_fingerprint="f" * 64,
        )


# ---------------------------------------------------------------------------
# Open / orphan attribution
# ---------------------------------------------------------------------------


def test_open_trade_dividend_not_assigned_to_any_closed_trade() -> None:
    run = _build_buy_on_x_no_op_run()
    assert project_closed_trades(run) == ()


X3_SESSION = date(2026, 8, 21)


def _build_real_applied_dividend_on_still_open_trade_run() -> HistoricalBacktestRunResult:
    """A prior round-trip closes one A trade; a later BUY of A holds
    through a REAL `DIVIDEND_APPLIED` and is never sold by run end.

    Unlike `_build_buy_on_x_no_op_run` (NO_OP, unheld) and
    `_build_replayed_run` (REPLAYED, zero new rows), this produces one
    genuine new dividend cash-ledger row attributed to a trade that is
    still open at run end -- OD-16.8's case. It also gives an earlier
    closed trade of the SAME security a chance to wrongly steal it.
    """

    initial = PortfolioState(settled_cash=Decimal("100000"))
    buy_old = PortfolioExecutionEvent(
        execution_id="BUY-OLD", source_order_id="O-1", session=T_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=5,
        fill_price=Decimal("50"), execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy_old,))
    after_t = t_result.resulting_state

    sell_old = PortfolioExecutionEvent(
        execution_id="SELL-OLD", source_order_id="O-2", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.SELL, quantity=5,
        fill_price=Decimal("55"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-OLD", settlement_session=X2_SESSION,
    )
    buy_new = PortfolioExecutionEvent(
        execution_id="BUY-NEW", source_order_id="O-3", session=X_SESSION,
        asset_id=ASSET_A, side=ExecutionSide.BUY, quantity=10,
        fill_price=Decimal("60"), execution_cost=Decimal("1"),
    )
    x_result = PortfolioTransitionEngine.transition(
        after_t, X_SESSION, (sell_old, buy_new)
    )
    after_x = x_result.resulting_state

    # X2_SESSION: BUY-NEW held through close with no executions -- this is
    # the real dividend's entitlement session T2.
    x2_result = PortfolioTransitionEngine.transition(after_x, X2_SESSION, ())
    after_x2 = x2_result.resulting_state

    evidence = _make_custom_evidence(
        event_id="D:OPEN-NEW",
        security_id=ASSET_A,
        entitlement_session=X2_SESSION,
        ex_session=X3_SESSION,
    )
    x3_result = PortfolioTransitionEngine.transition(
        after_x2, X3_SESSION, (), (evidence,)
    )
    final = x3_result.resulting_state

    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION, decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy_old,),
        state_transition_result=t_result, authoritative_state=after_t,
        ranking_snapshot=_ranking(T_SESSION),
    )
    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION, decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(after_t),
        ordered_execution_events=(sell_old, buy_new),
        state_transition_result=x_result, authoritative_state=after_x,
        ranking_snapshot=_ranking(X_SESSION),
    )
    x2_session = HistoricalBacktestSessionResult(
        session=X2_SESSION, decision_time=source_decision_time(X2_SESSION),
        prior_state_fingerprint=hash_portfolio_state(after_x),
        state_transition_result=x2_result, authoritative_state=after_x2,
        ranking_snapshot=_ranking(X2_SESSION),
    )
    x3_session = HistoricalBacktestSessionResult(
        session=X3_SESSION, decision_time=source_decision_time(X3_SESSION),
        prior_state_fingerprint=hash_portfolio_state(after_x2),
        state_transition_result=x3_result, authoritative_state=final,
        ranking_snapshot=_ranking(X3_SESSION),
    )
    session_evidence = (
        make_session_evidence(session=T_SESSION),
        make_session_evidence(session=X_SESSION),
        make_session_evidence(session=X2_SESSION),
        make_session_evidence(session=X3_SESSION, distribution_events=(evidence,)),
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(t_session, x_session, x2_session, x3_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def test_real_applied_dividend_on_open_trade_is_authoritative_and_not_stolen() -> None:
    run = _build_real_applied_dividend_on_still_open_trade_run()

    # Phase 13 really produced exactly one DIVIDEND_APPLIED outcome on X3.
    x3 = next(s for s in run.session_results if s.session == X3_SESSION)
    applied_entries = tuple(
        entry
        for entry in x3.state_transition_result.dividend_ledger_entries
        if entry.event_type == PortfolioLedgerEventType.DIVIDEND_APPLIED
    )
    assert len(applied_entries) == 1
    assert applied_entries[0].attribution_trade_id == "BUY-NEW"

    # Slice 9: exactly one projected DIVIDEND_APPLIED row, attributed to
    # the still-open trade.
    dividend_rows = tuple(
        row
        for row in project_cash_ledger(run)
        if row.ledger_event_type is PortfolioLedgerEventType.DIVIDEND_APPLIED
    )
    assert len(dividend_rows) == 1
    assert dividend_rows[0].attribution_trade_id == "BUY-NEW"

    # BUY-NEW is still open at run end; BUY-OLD is the run's one closed
    # trade (a genuine same-security prior trade that could try to steal
    # the dividend).
    open_ids = {p.entry_execution_id for p in run.final_state.open_positions}
    assert open_ids == {"BUY-NEW"}

    closed = project_closed_trades(run)
    assert len(closed) == 1
    old_trade = closed[0]
    assert old_trade.trade_id == "BUY-OLD"
    assert old_trade.security_id == ASSET_A  # same security as BUY-NEW
    # The real dividend was not stolen by the earlier same-security closed
    # trade: it received none of it.
    assert old_trade.ordinary_dividend_event_count == 0
    assert old_trade.ordinary_dividend_income == Decimal("0")
    # No closed-trade record exists for BUY-NEW at all -- it is still
    # open, so its authoritative dividend evidence lives only in the
    # projected cash ledger, never in a closed-trade record.
    assert all(trade.trade_id != "BUY-NEW" for trade in closed)


def test_orphan_attribution_fails_closed_at_grouping_helper() -> None:
    orphan_row = HistoricalCashLedgerRow(
        session=X_SESSION,
        sequence_in_session=0,
        ledger_event_type=PortfolioLedgerEventType.DIVIDEND_APPLIED,
        source_event_id="app-orphan",
        settled_cash_delta=Decimal("100"),
        pending_cash_delta=Decimal("0"),
        settled_cash_after=Decimal("100"),
        state_hash_before="a" * 64,
        state_hash_after="b" * 64,
        source_payload_fingerprint="c" * 64,
        attribution_trade_id="GHOST-TRADE",
    )
    with pytest.raises(HistoricalBacktestResultValidationError):
        _group_dividend_cash_ledger_rows_by_trade(
            (orphan_row,),
            closed_trade_ids=frozenset(),
            open_trade_ids=frozenset(),
        )


def test_tampered_attribution_cannot_enter_through_public_derivation() -> None:
    run = _build_sell_on_x_run()
    x_session = next(s for s in run.session_results if s.session == X_SESSION)
    tampered_transition = x_session.state_transition_result.model_copy(
        update={
            "dividend_ledger_entries": tuple(
                entry.model_copy(update={"attribution_trade_id": "GHOST-TRADE"})
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
        project_closed_trades(tampered_run)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic() -> None:
    run = _build_sell_on_x_run()
    first = _closed_trade_for(run, "BUY-1").ordinary_dividend_attribution_fingerprint
    second = _closed_trade_for(run, "BUY-1").ordinary_dividend_attribution_fingerprint
    assert first == second


def test_fingerprint_changes_when_grouped_row_payload_changes() -> None:
    base = compute_dividend_attribution_fingerprint(
        trade_id="T", completeness="COMPLETE_TRADE_LIFETIME",
        dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
        canonical_distribution_snapshot_fingerprint=None,
        grouped_rows=(("app-1", "f" * 64),),
    )
    changed = compute_dividend_attribution_fingerprint(
        trade_id="T", completeness="COMPLETE_TRADE_LIFETIME",
        dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
        canonical_distribution_snapshot_fingerprint=None,
        grouped_rows=(("app-1", "e" * 64),),  # different payload hash
    )
    assert base != changed


def test_fingerprint_changes_when_application_id_changes() -> None:
    base = compute_dividend_attribution_fingerprint(
        trade_id="T", completeness="COMPLETE_TRADE_LIFETIME",
        dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
        canonical_distribution_snapshot_fingerprint=None,
        grouped_rows=(("app-1", "f" * 64),),
    )
    changed = compute_dividend_attribution_fingerprint(
        trade_id="T", completeness="COMPLETE_TRADE_LIFETIME",
        dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
        canonical_distribution_snapshot_fingerprint=None,
        grouped_rows=(("app-2", "f" * 64),),
    )
    assert base != changed


def test_fingerprint_changes_when_completeness_changes() -> None:
    base = compute_dividend_attribution_fingerprint(
        trade_id="T", completeness="COMPLETE_TRADE_LIFETIME",
        dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
        canonical_distribution_snapshot_fingerprint=None,
        grouped_rows=(),
    )
    changed = compute_dividend_attribution_fingerprint(
        trade_id="T", completeness="PARTIAL_PRE_RUN_UNKNOWN",
        dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
        canonical_distribution_snapshot_fingerprint=None,
        grouped_rows=(),
    )
    assert base != changed


def test_empty_group_has_deterministic_fingerprint() -> None:
    first = compute_dividend_attribution_fingerprint(
        trade_id="T", completeness="COMPLETE_TRADE_LIFETIME",
        dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
        canonical_distribution_snapshot_fingerprint=None,
        grouped_rows=(),
    )
    second = compute_dividend_attribution_fingerprint(
        trade_id="T", completeness="COMPLETE_TRADE_LIFETIME",
        dividend_policy_semantic_identity="ORDINARY_DIVIDEND_ACCOUNTING_V0_1",
        canonical_distribution_snapshot_fingerprint=None,
        grouped_rows=(),
    )
    assert first == second
    assert len(first) == 64


def test_different_trade_ids_change_grouping_not_just_fingerprint() -> None:
    run = _build_two_closed_trades_with_dividends_run()
    trade_a = _closed_trade_for(run, "BUY-A")
    trade_b = _closed_trade_for(run, "BUY-B")
    assert trade_a.trade_id != trade_b.trade_id
    assert trade_a.ordinary_dividend_attribution_fingerprint != (
        trade_b.ordinary_dividend_attribution_fingerprint
    )


# ---------------------------------------------------------------------------
# Version / persistence / regression
# ---------------------------------------------------------------------------


def test_closed_trade_schema_version_is_v0_3() -> None:
    # Slice 11 bumped the live dividend-aware projector's emitted version
    # from v0.2 to v0.3 (grouping + authoritative trade_total_pnl); v0.2
    # itself remains a representable (but no-longer-emitted) schema, see
    # test_phase15d_trade_total_pnl.py.
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    assert trade.schema_version == "historical_closed_trade.v0.3"


def test_trades_round_trip_preserves_grouping_fields() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    encoded = encode_trade_row(trade, 0)
    decoded = decode_trade_row(encoded)
    assert decoded == trade
    assert decoded.ordinary_dividend_income == trade.ordinary_dividend_income
    assert decoded.ordinary_dividend_event_count == trade.ordinary_dividend_event_count
    assert decoded.dividend_attribution_completeness == (
        trade.dividend_attribution_completeness
    )
    assert decoded.ordinary_dividend_attribution_fingerprint == (
        trade.ordinary_dividend_attribution_fingerprint
    )


def test_content_fingerprint_changes_when_trade_dividend_field_changes() -> None:
    from stock_swing_d1.backtest_results.hashing import compute_content_fingerprint

    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    zero_income_trade = trade.model_copy(
        update={
            "ordinary_dividend_income": Decimal("0"),
            "ordinary_dividend_event_count": 0,
        }
    )
    original = compute_content_fingerprint(
        artifact_name="trades", rows_or_value=(trade,)
    )
    changed = compute_content_fingerprint(
        artifact_name="trades", rows_or_value=(zero_income_trade,)
    )
    assert original != changed


def test_source_run_hash_domain_unchanged() -> None:
    from stock_swing_d1.backtest_results.hashing import SOURCE_RUN_HASH_DOMAIN

    # Frozen Task 5C-B baseline: v0.3 (nested session results gained
    # entry_session_protective_decisions); Task 5C-C adds no further bump.
    assert SOURCE_RUN_HASH_DOMAIN == "historical_backtest_source_run.v0.3"


def test_realized_pnl_cost_basis_and_proceeds_unchanged_by_grouping() -> None:
    run = _build_sell_on_x_run()
    trade = _closed_trade_for(run, "BUY-1")
    assert trade.realized_pnl == trade.net_exit_proceeds - trade.entry_cost_basis
    assert trade.entry_cost_basis == (
        Decimal(trade.quantity) * trade.entry_fill_price + trade.entry_execution_cost
    )
    assert trade.net_exit_proceeds == trade.gross_exit_proceeds - trade.exit_execution_cost


# Slice 10's `test_no_trade_total_pnl_field_exists` boundary marker is
# superseded by Slice 11, which adds the field; see
# test_phase15d_trade_total_pnl.py's
# `test_trade_total_pnl_field_exists_and_is_optional` for its successor.
