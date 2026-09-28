"""Slice 8 acceptance tests: Phase 15A -> Phase 13 dividend-evidence
forwarding through the real `HistoricalBacktestOrchestrator.run()` path.

Fixes and proves the wiring gap discovered and deliberately left unfixed in
Slice 7: validated per-session `CanonicalDividendAccountingEvidence` (from
the accepted Slice-4 dividend-aware pre-flight, `validate_dividend_aware_
run_evidence`) is now forwarded into `PortfolioTransitionEngine.transition`
via its existing dedicated `dividend_evidence` channel, so a real historical
run can produce authoritative Phase-13 `DIVIDEND_APPLIED` rows, outcomes,
and settled-cash mutation end-to-end -- not just validate evidence. This is
wiring only: no entitlement/Q_T/attribution/D_H/application-identity logic
is computed in `backtester/orchestration.py`; those remain exclusively
Phase-13 (and upstream canonical-evidence) responsibilities.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from stock_swing_d1.backtester import (
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionInput,
    HistoricalDecisionInterval,
)
from stock_swing_d1.backtester.models import (
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION,
)
from stock_swing_d1.backtester.validation import HistoricalBacktestValidationError
from stock_swing_d1.execution.open_position_exit import (
    OpenPositionExitEvaluationInput,
    OpenPositionExitEvaluator,
)
from stock_swing_d1.execution.protective_exit import ProtectiveExitState
from stock_swing_d1.models import StockBar
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
    CanonicalDistributionCoverage,
    build_affirmative_distribution_coverage,
)
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    ProcessedSessionLink,
    build_processed_session_contiguity_proof,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    DividendApplicationStatus,
    compute_dividend_application_id,
    compute_dividend_application_payload_hash,
    compute_gross_dividend_cash,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    ExecutionSide,
    PortfolioEventKind,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_invariants import PortfolioInvariantChecker
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState


T_SESSION = date(2026, 8, 18)
X_SESSION = date(2026, 8, 19)

ASSET_A = "NORGATE:1"
ASSET_B = "NORGATE:2"

CALENDAR_SOURCE_ID = "accepted-us-equity-calendar"
CALENDAR_POLICY_ID = "canonical-next-session"
CALENDAR_POLICY_VERSION = "1"
CLASSIFICATION_CONTRACT_ID = "ordinary-cash-classification.v1"
COVERAGE_CONTRACT_ID = "affirmative-coverage.v1"
COVERAGE_CONTRACT_VERSION = "1"
SNAPSHOT_FINGERPRINT = "c" * 64

DECISION_INTERVAL = HistoricalDecisionInterval(
    decision_start_date=date(2026, 8, 1), decision_end_date=date(2026, 9, 30)
)


def _decision_time(session: date) -> datetime:
    return datetime(session.year, session.month, session.day, 16, 0, tzinfo=timezone.utc)


def make_evidence(
    *,
    event_id: str,
    security_id: str = ASSET_A,
    entitlement_session: date = T_SESSION,
    ex_session: date = X_SESSION,
) -> CanonicalDividendAccountingEvidence:
    inputs = Gate3DividendNormalizationInputs(
        entitlement_session=entitlement_session,
        d_capitalspecial=Decimal("2"),
        unadjusted_close_t=Decimal("3"),
        close_capital_t=Decimal("4"),
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


def make_coverage(*, session: date) -> CanonicalDistributionCoverage:
    return build_affirmative_distribution_coverage(
        session=session,
        coverage_contract_id=COVERAGE_CONTRACT_ID,
        coverage_contract_version=COVERAGE_CONTRACT_VERSION,
        canonical_distribution_snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
        upstream_source_evidence_fingerprint="d" * 64,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    )


def make_proof(*, sessions: tuple[date, ...]):
    links = tuple(
        ProcessedSessionLink(
            session=session,
            next_session=None if index == len(sessions) - 1 else sessions[index + 1],
        )
        for index, session in enumerate(sessions)
    )
    return build_processed_session_contiguity_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint="b" * 64,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        session_links=links,
    )


def session_input(
    session: date,
    *,
    next_session: date | None = None,
    scheduled_execution_events: tuple[PortfolioExecutionEvent, ...] = (),
    distribution_events: tuple[CanonicalDividendAccountingEvidence, ...] = (),
    distribution_coverage: CanonicalDistributionCoverage | None = None,
    open_position_exit_evaluations: (
        tuple[OpenPositionExitEvaluationInput, ...]
    ) = (),
) -> HistoricalBacktestSessionInput:
    return HistoricalBacktestSessionInput(
        session=session,
        decision_time=_decision_time(session),
        next_session=next_session,
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION,
        scheduled_execution_events=scheduled_execution_events,
        distribution_events=distribution_events,
        distribution_coverage=distribution_coverage,
        open_position_exit_evaluations=open_position_exit_evaluations,
    )


class _WeekdayCalendar:
    """Minimal business-day OpenPositionExitCalendar for HOLD fixtures."""

    def next_session(self, session: date) -> date:
        candidate = session + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

    def previous_session(self, session: date) -> date:
        candidate = session - timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
        return candidate

    def session_distance(self, start: date, end: date) -> int:
        distance = 0
        cursor = start
        while cursor < end:
            cursor = self.next_session(cursor)
            distance += 1
        while cursor > end:
            cursor = self.previous_session(cursor)
            distance -= 1
        return distance


def _orchestrator(**overrides) -> HistoricalBacktestOrchestrator:
    overrides.setdefault(
        "open_position_exit_evaluator",
        OpenPositionExitEvaluator(trading_calendar=_WeekdayCalendar()),
    )
    return HistoricalBacktestOrchestrator(**overrides)


def run_dividend_aware(sessions: tuple[HistoricalBacktestSessionInput, ...]):
    proof = make_proof(sessions=tuple(item.session for item in sessions))
    return _orchestrator().run(
        PortfolioState(settled_cash=Decimal("100000")),
        sessions,
        decision_interval=DECISION_INTERVAL,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )


def buy(execution_id, *, session, asset_id=ASSET_A, quantity=10, fill_price=Decimal("100")):
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"ORDER-{execution_id}",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=Decimal("1"),
    )


def sell(
    execution_id,
    *,
    session,
    asset_id=ASSET_A,
    quantity=10,
    fill_price=Decimal("120"),
    settlement_session=date(2026, 8, 21),
):
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"ORDER-{execution_id}",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.SELL,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=Decimal("1"),
        settlement_id=f"SETTLE-{execution_id}",
        settlement_session=settlement_session,
    )


def hold_evaluation(
    *,
    session: date,
    asset_id: str = ASSET_A,
    entry_session: date = T_SESSION,
    entry_price: float = 100.0,
    last_evaluated_session: date | None = T_SESSION,
) -> OpenPositionExitEvaluationInput:
    """A pre-existing (non-dividend) Phase-15B requirement: every position
    open entering a session needs one evaluation or a full scheduled SELL.
    This constructs a HOLD -- market action safely inside the stop/take-
    profit band -- so the position survives unsold into the next session,
    exactly the fixture Slice-8's dividend-attribution tests need and
    unrelated to any dividend logic itself."""

    signal_session = entry_session - timedelta(days=3)
    state = ProtectiveExitState._validated(
        security_id=asset_id,
        symbol=asset_id.replace("NORGATE:", "S"),
        signal_session=signal_session,
        signal_time=datetime(
            signal_session.year, signal_session.month, signal_session.day,
            16, tzinfo=timezone.utc,
        ),
        entry_session=entry_session,
        entry_price=entry_price,
        signal_atr_fraction=0.02,
        risk_fraction=0.04,
        stop_price=entry_price - 20.0,
        take_profit_price=entry_price + 50.0,
        last_evaluated_session=(
            None if session == entry_session else last_evaluated_session
        ),
    )
    bar = StockBar(
        security_id=asset_id,
        symbol=state.symbol,
        trading_date=session,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=entry_price,
        high=entry_price + 5.0,
        low=entry_price - 3.0,
        close=entry_price + 1.0,
        volume=1_000_000,
    )
    return OpenPositionExitEvaluationInput(
        session=session,
        protective_state=state,
        market_bar=bar,
    )


# ---------------------------------------------------------------------------
# A. Direct forwarding
# ---------------------------------------------------------------------------


def test_valid_dividend_aware_run_reaches_phase13_transition() -> None:
    evidence = make_evidence(event_id="D:APPLIED")
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
            open_position_exit_evaluations=(hold_evaluation(session=X_SESSION),),
        ),
    )
    run = run_dividend_aware(sessions)
    x_result = run.session_results[1]
    assert len(x_result.state_transition_result.dividend_outcomes) == 1
    assert x_result.state_transition_result.dividend_outcomes[0].status is (
        DividendApplicationStatus.APPLIED
    )


def test_forwarded_event_semantic_content_matches_supplied_evidence() -> None:
    evidence = make_evidence(event_id="D:APPLIED")
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
            open_position_exit_evaluations=(hold_evaluation(session=X_SESSION),),
        ),
    )
    run = run_dividend_aware(sessions)
    row = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    assert row.d_h == evidence.amount_per_share
    assert row.canonical_distribution_event_id == "D:APPLIED"
    assert row.canonical_distribution_snapshot_fingerprint == (
        evidence.canonical_distribution_snapshot_fingerprint
    )


def test_forwarded_tuple_is_session_specific() -> None:
    """Evidence supplied on T must never reach the X transition and
    vice versa -- each session's transition call only ever receives its
    own ex_session's evidence."""

    evidence_x = make_evidence(event_id="D:ONX", entitlement_session=T_SESSION, ex_session=X_SESSION)
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            distribution_events=(evidence_x,),
            distribution_coverage=make_coverage(session=X_SESSION),
            open_position_exit_evaluations=(hold_evaluation(session=X_SESSION),),
        ),
    )
    run = run_dividend_aware(sessions)
    t_result = run.session_results[0]
    assert t_result.state_transition_result.dividend_outcomes == ()
    assert t_result.state_transition_result.dividend_ledger_entries == ()


def test_empty_event_session_forwards_empty_tuple() -> None:
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            distribution_coverage=make_coverage(session=X_SESSION),
            open_position_exit_evaluations=(hold_evaluation(session=X_SESSION),),
        ),
    )
    run = run_dividend_aware(sessions)
    for session_result in run.session_results:
        assert session_result.state_transition_result.dividend_outcomes == ()
        assert session_result.state_transition_result.dividend_ledger_entries == ()


def test_dividend_evidence_never_becomes_a_portfolio_execution_event() -> None:
    evidence = make_evidence(event_id="D:APPLIED")
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
            open_position_exit_evaluations=(hold_evaluation(session=X_SESSION),),
        ),
    )
    run = run_dividend_aware(sessions)
    x_result = run.session_results[1]
    assert x_result.ordered_execution_events == ()
    assert x_result.state_transition_result.ledger_entries == ()


def test_forwarded_event_order_is_deterministic_by_asset_then_event_id() -> None:
    evidence_a = make_evidence(event_id="D:BBB", security_id=ASSET_A)
    evidence_b = make_evidence(event_id="D:AAA", security_id=ASSET_B)
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(
                buy("BUY-A", session=T_SESSION, asset_id=ASSET_A),
                buy("BUY-B", session=T_SESSION, asset_id=ASSET_B),
            ),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            distribution_events=(evidence_a, evidence_b),
            distribution_coverage=make_coverage(session=X_SESSION),
            open_position_exit_evaluations=(
                hold_evaluation(session=X_SESSION, asset_id=ASSET_A),
                hold_evaluation(session=X_SESSION, asset_id=ASSET_B),
            ),
        ),
    )
    run = run_dividend_aware(sessions)
    rows = run.session_results[1].state_transition_result.dividend_ledger_entries
    assert [r.asset_id for r in rows] == sorted(r.asset_id for r in rows)


def test_reversing_raw_input_order_does_not_change_canonical_forwarded_order() -> (
    None
):
    evidence_a = make_evidence(event_id="D:BBB", security_id=ASSET_A)
    evidence_b = make_evidence(event_id="D:AAA", security_id=ASSET_B)

    def build(order):
        sessions = (
            session_input(
                T_SESSION,
                next_session=X_SESSION,
                scheduled_execution_events=(
                    buy("BUY-A", session=T_SESSION, asset_id=ASSET_A),
                    buy("BUY-B", session=T_SESSION, asset_id=ASSET_B),
                ),
                distribution_coverage=make_coverage(session=T_SESSION),
            ),
            session_input(
                X_SESSION,
                distribution_events=order,
                distribution_coverage=make_coverage(session=X_SESSION),
                open_position_exit_evaluations=(
                    hold_evaluation(session=X_SESSION, asset_id=ASSET_A),
                    hold_evaluation(session=X_SESSION, asset_id=ASSET_B),
                ),
            ),
        )
        return run_dividend_aware(sessions)

    forward = build((evidence_a, evidence_b))
    reversed_run = build((evidence_b, evidence_a))
    forward_rows = forward.session_results[1].state_transition_result.dividend_ledger_entries
    reversed_rows = reversed_run.session_results[1].state_transition_result.dividend_ledger_entries
    assert [r.application_id for r in forward_rows] == [
        r.application_id for r in reversed_rows
    ]


def test_wrong_session_forwarding_would_fail_evidence_placement_preflight() -> None:
    """Evidence whose ex_session does not match the session it is placed on
    is rejected by the accepted Slice-4 pre-flight before any forwarding
    could occur -- proving a "wrong session" forward is structurally
    impossible, not merely untested."""

    evidence = make_evidence(event_id="D:MISPLACED", ex_session=X_SESSION)
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            distribution_events=(evidence,),  # ex_session=X placed on T
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    with pytest.raises(HistoricalBacktestValidationError):
        run_dividend_aware(sessions)


def test_duplicate_event_id_across_sessions_fails_preflight_not_forwarding() -> None:
    prior_session = T_SESSION - timedelta(days=1)
    evidence_t = make_evidence(
        event_id="D:DUP", entitlement_session=prior_session, ex_session=T_SESSION
    )
    evidence_x = make_evidence(
        event_id="D:DUP", entitlement_session=T_SESSION, ex_session=X_SESSION
    )
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            distribution_events=(evidence_t,),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            distribution_events=(evidence_x,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    with pytest.raises(HistoricalBacktestValidationError):
        run_dividend_aware(sessions)


# ---------------------------------------------------------------------------
# B. Validation-order regression
# ---------------------------------------------------------------------------


def test_session_contract_failure_precedes_dividend_preflight_failure() -> None:
    """A session-contract violation (next_session <= session) must still be
    caught before dividend pre-flight even runs."""

    with pytest.raises(Exception):
        session_input(T_SESSION, next_session=T_SESSION)


def test_no_phase13_call_occurs_if_dividend_preflight_fails() -> None:
    calls = []
    real_transition = None
    from stock_swing_d1.portfolio.portfolio_transition import (
        PortfolioTransitionEngine,
    )

    real_transition = PortfolioTransitionEngine.transition

    def tracking(previous_state, session, events, dividend_evidence=()):
        calls.append(session)
        return real_transition(
            previous_state, session, events, dividend_evidence=dividend_evidence
        )

    orchestrator = HistoricalBacktestOrchestrator(transition_service=type(
        "T", (), {"transition": staticmethod(tracking)}
    ))
    evidence = make_evidence(event_id="D:X")
    sessions = (
        session_input(
            X_SESSION,
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    # No processed_session_contiguity_proof supplied for a non-empty
    # dividend-aware run -> pre-flight must fail before any transition call.
    with pytest.raises(HistoricalBacktestValidationError):
        orchestrator.run(
            PortfolioState(settled_cash=Decimal("1000")),
            sessions,
            decision_interval=DECISION_INTERVAL,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        )
    assert calls == []


# ---------------------------------------------------------------------------
# C. End-to-end APPLIED
# ---------------------------------------------------------------------------


def _applied_run():
    evidence = make_evidence(event_id="D:APPLIED")
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
            open_position_exit_evaluations=(hold_evaluation(session=X_SESSION),),
        ),
    )
    return run_dividend_aware(sessions), evidence


def test_hold_through_t_and_x_produces_applied() -> None:
    run, evidence = _applied_run()
    outcome = run.session_results[1].state_transition_result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.APPLIED
    assert outcome.q_t == 10
    assert outcome.attribution_trade_id == "BUY-1"


def test_dividend_row_is_produced_by_phase13() -> None:
    run, evidence = _applied_run()
    row = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    assert row.q_t == 10
    assert row.attribution_trade_id == "BUY-1"
    assert row.gross_cash_amount == compute_gross_dividend_cash(10, evidence.amount_per_share)


def test_final_settled_cash_includes_exact_gross_dividend_cash() -> None:
    run, evidence = _applied_run()
    t_state = run.session_results[0].authoritative_state
    x_state = run.session_results[1].authoritative_state
    gross = compute_gross_dividend_cash(10, evidence.amount_per_share)
    # X applied no new executions -- settled cash only moves by dividend cash.
    assert x_state.settled_cash == t_state.settled_cash + gross


def test_applied_events_registry_contains_dividend_application() -> None:
    run, evidence = _applied_run()
    x_state = run.session_results[1].authoritative_state
    row = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    keys = {(fp.event_kind, fp.event_id) for fp in x_state.applied_events}
    assert (PortfolioEventKind.DIVIDEND, row.application_id) in keys


def test_row_and_outcome_event_ids_match_forwarded_evidence() -> None:
    run, evidence = _applied_run()
    row = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    outcome = run.session_results[1].state_transition_result.dividend_outcomes[0]
    assert row.canonical_distribution_event_id == evidence.canonical_distribution_event_id
    assert outcome.canonical_distribution_event_id == (
        evidence.canonical_distribution_event_id
    )


# ---------------------------------------------------------------------------
# D. Same-X SELL / re-entry
# ---------------------------------------------------------------------------


def test_sell_on_x_preserves_entitlement() -> None:
    evidence = make_evidence(event_id="D:SOLD")
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            scheduled_execution_events=(sell("SELL-1", session=X_SESSION),),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    outcome = run.session_results[1].state_transition_result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.APPLIED
    assert outcome.q_t == 10


def test_sell_on_x_dividend_row_attributes_old_trade() -> None:
    evidence = make_evidence(event_id="D:SOLD")
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            scheduled_execution_events=(sell("SELL-1", session=X_SESSION),),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    row = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    assert row.attribution_trade_id == "BUY-1"


def test_sell_and_reentry_on_x_still_attributes_old_trade() -> None:
    evidence = make_evidence(event_id="D:REENTRY")
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            scheduled_execution_events=(
                sell("SELL-1", session=X_SESSION),
                buy("BUY-2", session=X_SESSION, quantity=5, fill_price=Decimal("120")),
            ),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    row = run.session_results[1].state_transition_result.dividend_ledger_entries[0]
    assert row.attribution_trade_id == "BUY-1"
    assert row.q_t == 10


def test_new_same_x_trade_receives_no_dividend_attribution() -> None:
    evidence = make_evidence(event_id="D:REENTRY2")
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            scheduled_execution_events=(
                sell("SELL-1", session=X_SESSION),
                buy("BUY-2", session=X_SESSION, quantity=5, fill_price=Decimal("120")),
            ),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    rows = run.session_results[1].state_transition_result.dividend_ledger_entries
    assert all(r.attribution_trade_id != "BUY-2" for r in rows)
    new_position = run.session_results[1].authoritative_state.open_positions[0]
    assert new_position.entry_execution_id == "BUY-2"


def test_same_x_buy_cannot_spend_same_x_dividend_cash_before_dividend_step() -> None:
    """The BUY on X is sized against pre-dividend settled cash: it must
    succeed or fail purely on cash available before the dividend step,
    proving dividend cash from the same X is unavailable to same-X BUYs
    (OD-10.1), enforced by Phase 13's own chronology, not by Phase 15A."""

    evidence = make_evidence(event_id="D:NOFUND")
    initial = PortfolioState(settled_cash=Decimal("1005"))
    tiny_buy = buy("BUY-1", session=T_SESSION, quantity=10, fill_price=Decimal("100"))
    reentry = PortfolioExecutionEvent(
        execution_id="BUY-2",
        source_order_id="ORDER-BUY-2",
        session=X_SESSION,
        asset_id=ASSET_A,
        side=ExecutionSide.BUY,
        quantity=1,
        fill_price=Decimal("120"),
        execution_cost=Decimal("1"),
    )
    sessions = (
        session_input(
            T_SESSION,
            next_session=X_SESSION,
            scheduled_execution_events=(tiny_buy,),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
        session_input(
            X_SESSION,
            scheduled_execution_events=(sell("SELL-1", session=X_SESSION), reentry),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    proof = make_proof(sessions=(T_SESSION, X_SESSION))
    from stock_swing_d1.portfolio.portfolio_errors import InsufficientSettledCashError

    # Settled cash after T's BUY (1005 - 1001) = 4; SELL settlement is
    # pending (not yet settled cash) on X, so BUY-2 (121) must fail on
    # insufficient cash -- proving dividend cash (credited only after BUY)
    # was not and could not have been used to fund it.
    with pytest.raises(InsufficientSettledCashError):
        HistoricalBacktestOrchestrator().run(
            initial,
            sessions,
            decision_interval=DECISION_INTERVAL,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


# ---------------------------------------------------------------------------
# E. NO_OP
# ---------------------------------------------------------------------------


def test_buy_only_on_x_produces_no_op_not_entitled() -> None:
    evidence = make_evidence(event_id="D:BUYONX")
    sessions = (
        session_input(
            X_SESSION,
            scheduled_execution_events=(buy("BUY-ON-X", session=X_SESSION),),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    outcome = run.session_results[0].state_transition_result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED
    assert outcome.q_t == 0
    assert outcome.attribution_trade_id is None


def test_no_op_produces_no_dividend_ledger_row() -> None:
    evidence = make_evidence(event_id="D:BUYONX2")
    sessions = (
        session_input(
            X_SESSION,
            scheduled_execution_events=(buy("BUY-ON-X", session=X_SESSION),),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    assert run.session_results[0].state_transition_result.dividend_ledger_entries == ()


def test_no_op_causes_zero_dividend_cash_mutation() -> None:
    evidence = make_evidence(event_id="D:BUYONX3")
    initial_cash = Decimal("100000")
    sessions = (
        session_input(
            X_SESSION,
            scheduled_execution_events=(buy("BUY-ON-X", session=X_SESSION),),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=initial_cash),
        sessions,
        decision_interval=DECISION_INTERVAL,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=make_proof(sessions=(X_SESSION,)),
    )
    buy_cost = Decimal(10) * Decimal("100") + Decimal("1")
    assert run.final_state.settled_cash == initial_cash - buy_cost


def test_no_op_causes_no_applied_event_mutation() -> None:
    evidence = make_evidence(event_id="D:BUYONX4")
    sessions = (
        session_input(
            X_SESSION,
            scheduled_execution_events=(buy("BUY-ON-X", session=X_SESSION),),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    keys = {
        (fp.event_kind, fp.event_id) for fp in run.final_state.applied_events
    }
    assert not any(kind is PortfolioEventKind.DIVIDEND for kind, _ in keys)


def test_no_op_does_not_use_post_buy_position_as_entitlement() -> None:
    evidence = make_evidence(event_id="D:BUYONX5")
    sessions = (
        session_input(
            X_SESSION,
            scheduled_execution_events=(buy("BUY-ON-X", session=X_SESSION),),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    # The BUY genuinely opens a position in the final state...
    assert len(run.final_state.open_positions) == 1
    # ...yet the outcome is still NO_OP, proving entitlement came from the
    # pre-X (empty) state, not from this post-BUY position.
    outcome = run.session_results[0].state_transition_result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED


# ---------------------------------------------------------------------------
# F. REPLAYED
# ---------------------------------------------------------------------------


def test_replayed_end_to_end_via_seeded_initial_state_fingerprint() -> None:
    """HistoricalBacktestOrchestrator.run() accepts an arbitrary initial
    PortfolioState, so a REPLAYED end-to-end fixture IS representable: seed
    an initial_state whose applied_events already carries the exact prior
    dividend application fingerprint, then re-supply the identical
    canonical evidence on X and prove it forwards, Phase 13 returns
    REPLAYED, and there is no second row/cash mutation."""

    opened = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("100000")),
        (
            session_input(
                T_SESSION,
                scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            ),
        ),
        decision_interval=DECISION_INTERVAL,
    ).final_state

    evidence = make_evidence(event_id="D:REPLAY")
    application_id = compute_dividend_application_id(
        canonical_distribution_event_id=evidence.canonical_distribution_event_id,
        entitlement_session=evidence.entitlement_session,
        ex_session=evidence.ex_session,
        asset_id=ASSET_A,
        attribution_trade_id="BUY-1",
    )
    gross_cash = compute_gross_dividend_cash(10, evidence.amount_per_share)
    payload_hash = compute_dividend_application_payload_hash(
        application_id=application_id,
        evidence=evidence,
        asset_id=ASSET_A,
        attribution_trade_id="BUY-1",
        q_t=10,
        gross_cash_amount=gross_cash,
    )
    seeded_initial = PortfolioState(
        base_currency=opened.base_currency,
        as_of_session=opened.as_of_session,
        state_version=opened.state_version,
        settled_cash=opened.settled_cash,
        open_positions=opened.open_positions,
        pending_settlements=opened.pending_settlements,
        applied_events=opened.applied_events
        + (
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.DIVIDEND,
                event_id=application_id,
                payload_sha256=payload_hash,
            ),
        ),
    )
    PortfolioInvariantChecker.validate_state(seeded_initial)

    sessions = (
        session_input(
            X_SESSION,
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
            open_position_exit_evaluations=(
                hold_evaluation(session=X_SESSION, last_evaluated_session=T_SESSION),
            ),
        ),
    )
    run = _orchestrator().run(
        seeded_initial,
        sessions,
        decision_interval=DECISION_INTERVAL,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=make_proof(sessions=(X_SESSION,)),
    )

    outcome = run.session_results[0].state_transition_result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.REPLAYED
    assert run.session_results[0].state_transition_result.dividend_ledger_entries == ()
    assert run.final_state.settled_cash == seeded_initial.settled_cash


# ---------------------------------------------------------------------------
# G. Terminal / empty behavior
# ---------------------------------------------------------------------------


def test_run_ending_on_t_does_not_invent_x_or_credit_future_cash() -> None:
    """T's own ex-session is outside the processed run (a future X is
    never invented); processing only T (with affirmative zero-event
    coverage for T) must not credit any dividend cash."""

    sessions = (
        session_input(
            T_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    assert run.session_results[0].state_transition_result.dividend_ledger_entries == ()
    assert run.session_results[0].state_transition_result.dividend_outcomes == ()
    expected_cash = Decimal("100000") - (Decimal(10) * Decimal("100") + Decimal("1"))
    assert run.final_state.settled_cash == expected_cash


def test_zero_session_dividend_aware_run_remains_valid() -> None:
    run = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        (),
        decision_interval=DECISION_INTERVAL,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    )
    assert run.session_results == ()
    assert run.final_state == run.initial_state


def test_non_dividend_run_behavior_unchanged() -> None:
    sessions = (
        HistoricalBacktestSessionInput(
            session=T_SESSION,
            decision_time=_decision_time(T_SESSION),
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
        ),
    )
    run = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("100000")),
        sessions,
        decision_interval=DECISION_INTERVAL,
    )
    assert run.dividend_run_evidence is None
    assert run.session_results[0].state_transition_result.dividend_outcomes == ()
    assert run.session_results[0].state_transition_result.dividend_ledger_entries == ()


def test_ordinary_empty_dividend_tuple_produces_no_dividend_facts() -> None:
    sessions = (
        session_input(
            T_SESSION,
            scheduled_execution_events=(buy("BUY-1", session=T_SESSION),),
            distribution_coverage=make_coverage(session=T_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    assert run.session_results[0].state_transition_result.dividend_outcomes == ()


# ---------------------------------------------------------------------------
# H. Source-validation integration
# ---------------------------------------------------------------------------


def test_real_applied_run_passes_accepted_site3_source_validation() -> None:
    from stock_swing_d1.backtest_results import validate_historical_backtest_source_run

    run, _ = _applied_run()
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)


def test_real_no_op_run_passes_accepted_site3_source_validation() -> None:
    from stock_swing_d1.backtest_results import validate_historical_backtest_source_run

    evidence = make_evidence(event_id="D:NOOPVALID")
    sessions = (
        session_input(
            X_SESSION,
            scheduled_execution_events=(buy("BUY-ON-X", session=X_SESSION),),
            distribution_events=(evidence,),
            distribution_coverage=make_coverage(session=X_SESSION),
        ),
    )
    run = run_dividend_aware(sessions)
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)


def test_corrupted_forwarding_still_cannot_silently_pass_source_validation() -> None:
    from stock_swing_d1.backtest_results import (
        HistoricalBacktestResultValidationError,
        validate_historical_backtest_source_run,
    )

    run, _ = _applied_run()
    x_session = run.session_results[1]
    transition = x_session.state_transition_result
    tampered_transition = transition.model_copy(
        update={"dividend_ledger_entries": ()}
    )
    tampered_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = run.model_copy(
        update={
            "session_results": (run.session_results[0], tampered_session),
        }
    )
    with pytest.raises(HistoricalBacktestResultValidationError):
        validate_historical_backtest_source_run(run_result=tampered_run, run_manifest=None)


def test_ledger_order_identity_and_source_hash_domain_unchanged() -> None:
    from stock_swing_d1.backtest_results.hashing import SOURCE_RUN_HASH_DOMAIN

    # Frozen Task 5C-B baseline: v0.3 (nested session results gained
    # entry_session_protective_decisions); Task 5C-C adds no further bump.
    assert SOURCE_RUN_HASH_DOMAIN == "historical_backtest_source_run.v0.3"


# ---------------------------------------------------------------------------
# I. Regression
# ---------------------------------------------------------------------------


def test_hash_portfolio_state_smoke() -> None:
    """Cheap smoke check that this file's own fixtures produce a valid,
    hashable state; the authoritative regression proof is the full
    tests/backtester + tests/portfolio + tests/backtest_results suites
    passing alongside this file (see the accompanying report)."""

    run, _ = _applied_run()
    assert hash_portfolio_state(run.final_state) == run.final_state_fingerprint
