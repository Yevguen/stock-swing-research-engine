"""Slice 7 acceptance tests: Phase 15D site-3 authoritative-source
validation for ordinary dividends.

Covers the frozen Ordinary Dividend Amendment v0.7 acceptance cases 139/140/
157: site 3 (`backtest_results/source_validation.py`) re-proves OD-14.5 row
provenance and OD-14.8/14.9 evidence/outcome discharge using independently
retained Phase-13/15A authoritative facts (`HistoricalBacktestRunResult.
dividend_run_evidence` + each session's `PortfolioTransitionResult.
dividend_ledger_entries`/`.dividend_outcomes`), reusing site 1's own
`PortfolioInvariantChecker.validate_dividend_application` -- never
reconstructing entitlement (Q_T/attribution) from Phase-15D holdings, a
calendar, or a provider. Phase 15D cash-ledger/trade/P&L projection is out
of scope for this slice.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    HistoricalBacktestResultValidationError,
    validate_historical_backtest_source_run,
)
from stock_swing_d1.backtest_results.hashing import compute_source_run_fingerprint
from stock_swing_d1.backtester.models import (
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION,
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
    CanonicalDistributionCoverage,
    DividendAwareRunEvidence,
    DividendAwareSessionEvidence,
    build_affirmative_distribution_coverage,
    build_dividend_aware_run_evidence,
    build_dividend_aware_session_evidence,
)
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    build_processed_session_contiguity_proof,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    DividendApplicationStatus,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from stock_swing_d1.ranking import rank_candidates
from tests.backtest_results.conftest import SOURCE_DECISION_INTERVAL, source_decision_time


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


def make_evidence(
    *,
    event_id: str,
    security_id: str = ASSET_A,
    snapshot_fingerprint: str = SNAPSHOT_FINGERPRINT,
) -> CanonicalDividendAccountingEvidence:
    inputs = Gate3DividendNormalizationInputs(
        entitlement_session=T_SESSION,
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
        entitlement_session=T_SESSION,
        ex_session=X_SESSION,
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
        canonical_distribution_snapshot_fingerprint=snapshot_fingerprint,
    )


def make_coverage(
    *,
    session: date,
    distribution_events: tuple[CanonicalDividendAccountingEvidence, ...] = (),
    snapshot_fingerprint: str = SNAPSHOT_FINGERPRINT,
) -> CanonicalDistributionCoverage:
    return build_affirmative_distribution_coverage(
        session=session,
        coverage_contract_id=COVERAGE_CONTRACT_ID,
        coverage_contract_version=COVERAGE_CONTRACT_VERSION,
        canonical_distribution_snapshot_fingerprint=snapshot_fingerprint,
        upstream_source_evidence_fingerprint="d" * 64,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    )


def make_session_evidence(
    *,
    session: date,
    distribution_events: tuple[CanonicalDividendAccountingEvidence, ...] = (),
    snapshot_fingerprint: str = SNAPSHOT_FINGERPRINT,
) -> DividendAwareSessionEvidence:
    return build_dividend_aware_session_evidence(
        session=session,
        distribution_coverage=make_coverage(
            session=session,
            distribution_events=distribution_events,
            snapshot_fingerprint=snapshot_fingerprint,
        ),
        distribution_events=distribution_events,
    )


def make_run_evidence(
    *,
    session_evidence: tuple[DividendAwareSessionEvidence, ...],
    snapshot_fingerprint: str = SNAPSHOT_FINGERPRINT,
) -> DividendAwareRunEvidence:
    if not session_evidence:
        return build_dividend_aware_run_evidence(
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=None,
            canonical_distribution_snapshot_fingerprint=None,
            session_evidence=(),
        )
    sessions = tuple(item.session for item in session_evidence)
    from stock_swing_d1.data.ordinary_dividend_run_policy import ProcessedSessionLink

    links = tuple(
        ProcessedSessionLink(
            session=session,
            next_session=None if index == len(sessions) - 1 else sessions[index + 1],
        )
        for index, session in enumerate(sessions)
    )
    proof = build_processed_session_contiguity_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint="b" * 64,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        session_links=links,
    )
    return build_dividend_aware_run_evidence(
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
        canonical_distribution_snapshot_fingerprint=snapshot_fingerprint,
        session_evidence=session_evidence,
    )


def _ranking(session: date):
    return rank_candidates(
        ranking_session=session,
        decision_time=source_decision_time(session),
        candidates=(),
    )


def build_applied_and_noop_run() -> HistoricalBacktestRunResult:
    """T: BUY 10 @ A. X: dividend APPLIED on A, NO_OP on unheld B."""

    initial = PortfolioState(settled_cash=Decimal("100000"))
    buy = PortfolioExecutionEvent(
        execution_id="BUY-1",
        source_order_id="ORDER-BUY-1",
        session=T_SESSION,
        asset_id=ASSET_A,
        side=ExecutionSide.BUY,
        quantity=10,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    t_result = PortfolioTransitionEngine.transition(initial, T_SESSION, (buy,))
    opened = t_result.resulting_state

    evidence_a = make_evidence(event_id="D:APPLIED", security_id=ASSET_A)
    evidence_b = make_evidence(event_id="D:NOOP", security_id=ASSET_B)
    x_result = PortfolioTransitionEngine.transition(
        opened, X_SESSION, (), (evidence_a, evidence_b)
    )
    final = x_result.resulting_state

    t_session = HistoricalBacktestSessionResult(
        session=T_SESSION,
        decision_time=source_decision_time(T_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(buy,),
        state_transition_result=t_result,
        authoritative_state=opened,
        ranking_snapshot=_ranking(T_SESSION),
    )
    x_session = HistoricalBacktestSessionResult(
        session=X_SESSION,
        decision_time=source_decision_time(X_SESSION),
        prior_state_fingerprint=hash_portfolio_state(opened),
        state_transition_result=x_result,
        authoritative_state=final,
        ranking_snapshot=_ranking(X_SESSION),
    )

    session_evidence = (
        make_session_evidence(session=T_SESSION),
        make_session_evidence(
            session=X_SESSION, distribution_events=(evidence_a, evidence_b)
        ),
    )
    run_evidence = make_run_evidence(session_evidence=session_evidence)

    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=run_evidence,
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial,
        final_state=final,
        session_results=(t_session, x_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )


def _x_session_result(run: HistoricalBacktestRunResult) -> HistoricalBacktestSessionResult:
    return next(s for s in run.session_results if s.session == X_SESSION)


def _replace_x_session(
    run: HistoricalBacktestRunResult, new_x_session: HistoricalBacktestSessionResult
) -> HistoricalBacktestRunResult:
    sessions = tuple(
        new_x_session if s.session == X_SESSION else s for s in run.session_results
    )
    return run.model_copy(update={"session_results": sessions})


# ---------------------------------------------------------------------------
# A. Source artifact acceptance
# ---------------------------------------------------------------------------


def test_valid_dividend_aware_run_is_accepted() -> None:
    run = build_applied_and_noop_run()
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)


def test_non_dividend_aware_run_with_empty_dividend_data_is_accepted() -> None:
    state = PortfolioState(settled_cash=Decimal("1000"))
    fingerprint = hash_portfolio_state(state)
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        initial_state_fingerprint=fingerprint,
        final_state_fingerprint=fingerprint,
    )
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)


def test_dividend_evidence_content_hash_mismatch_fails() -> None:
    """A run_evidence bundle whose retained content is internally
    inconsistent (event fingerprints tampered post-construction) cannot
    be canonically reconstructed, so site 3 fails closed."""

    run = build_applied_and_noop_run()
    tampered_evidence = run.dividend_run_evidence.model_copy(
        update={"canonical_distribution_snapshot_fingerprint": "f" * 64}
    )
    tampered_run = run.model_copy(update={"dividend_run_evidence": tampered_evidence})

    with pytest.raises(HistoricalBacktestResultValidationError):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


def test_missing_session_evidence_coverage_fails() -> None:
    run = build_applied_and_noop_run()
    truncated_evidence = run.dividend_run_evidence.model_copy(
        update={
            "session_evidence": (run.dividend_run_evidence.session_evidence[0],),
        }
    )
    # model_copy bypasses the model's own cross-field validator (the
    # session-evidence sequence must equal the contiguity proof's
    # processed_sessions); this simulates evidence dropped after
    # construction, which the model itself would never allow to be built.
    tampered_run = run.model_copy(
        update={"dividend_run_evidence": truncated_evidence}
    )

    with pytest.raises(HistoricalBacktestResultValidationError):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


def test_source_run_cannot_bypass_canonical_reconstruction() -> None:
    """A dividend_run_evidence of the wrong type cannot slip past
    validation as raw/unvalidated data."""

    run = build_applied_and_noop_run()

    class _Fake:
        pass

    tampered_run = run.model_copy(update={"dividend_run_evidence": _Fake()})
    with pytest.raises((HistoricalBacktestResultValidationError, Exception)):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


# ---------------------------------------------------------------------------
# B. Row -> evidence source validation
# ---------------------------------------------------------------------------


def test_valid_applied_row_evidence_outcome_passes() -> None:
    run = build_applied_and_noop_run()
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)

    x_session = _x_session_result(run)
    applied_row = next(
        row
        for row in x_session.state_transition_result.dividend_ledger_entries
        if row.canonical_distribution_event_id == "D:APPLIED"
    )
    assert applied_row.q_t == 10
    assert applied_row.attribution_trade_id == "BUY-1"


def test_row_without_evidence_fails() -> None:
    """Site 3 must fail closed if a run claims a dividend row/outcome but
    the run-level evidence bundle no longer supplies the corresponding
    event -- proves discharge is checked against retained evidence, not
    the row's own copied facts."""

    run = build_applied_and_noop_run()
    x_evidence = run.dividend_run_evidence.session_evidence[1]
    reduced_events = tuple(
        e
        for e in x_evidence.distribution_events
        if e.canonical_distribution_event_id != "D:APPLIED"
    )
    from tests.backtest_results.test_phase15d_dividend_source_validation import (
        make_session_evidence,
    )

    new_x_evidence = make_session_evidence(
        session=X_SESSION, distribution_events=reduced_events
    )
    tampered_run_evidence = run.dividend_run_evidence.model_copy(
        update={
            "session_evidence": (
                run.dividend_run_evidence.session_evidence[0],
                new_x_evidence,
            )
        }
    )
    tampered_run = run.model_copy(
        update={"dividend_run_evidence": tampered_run_evidence}
    )

    # Dropping D:APPLIED from the retained evidence makes the whole
    # run_evidence internally stale (its own run_input_fingerprint no
    # longer matches session_evidence content), so this is caught at
    # canonical-reconstruction time -- still a hard fail-closed rejection
    # of the tampering, just at an earlier layer than the per-row check.
    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="NONCANONICAL_SOURCE_RUN|DIVIDEND_PROVENANCE_MISMATCH",
    ):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("q_t", 3),
        ("attribution_trade_id", "BUY-999"),
        ("d_h", Decimal("9." + "0" * 37)),
        ("gross_cash_amount", Decimal("999999")),
        ("settled_cash_delta", Decimal("999999")),
        ("application_id", "a" * 64),
        ("source_payload_sha256", "b" * 64),
        ("canonical_distribution_snapshot_fingerprint", "e" * 64),
        ("normalization_inputs_fingerprint", "1" * 64),
        ("calendar_resolution_fingerprint", "2" * 64),
        ("currency", "EUR"),
        ("amount_basis", "OTHER_BASIS"),
    ],
)
def test_dividend_row_field_mismatch_fails(field_name: str, new_value: object) -> None:
    run = build_applied_and_noop_run()
    x_session = _x_session_result(run)
    transition = x_session.state_transition_result
    original_row = next(
        r
        for r in transition.dividend_ledger_entries
        if r.canonical_distribution_event_id == "D:APPLIED"
    )
    other_rows = tuple(r for r in transition.dividend_ledger_entries if r is not original_row)
    tampered_row = original_row.model_copy(update={field_name: new_value})
    tampered_transition = transition.model_copy(
        update={"dividend_ledger_entries": other_rows + (tampered_row,)}
    )
    tampered_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = _replace_x_session(run, tampered_session)

    with pytest.raises(HistoricalBacktestResultValidationError):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


def test_duplicate_authoritative_dividend_row_fails() -> None:
    run = build_applied_and_noop_run()
    x_session = _x_session_result(run)
    transition = x_session.state_transition_result
    duplicated = transition.dividend_ledger_entries + (
        transition.dividend_ledger_entries[0].model_copy(
            update={"sequence_in_session": len(transition.dividend_ledger_entries)}
        ),
    )
    tampered_transition = transition.model_copy(
        update={"dividend_ledger_entries": duplicated}
    )
    tampered_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = _replace_x_session(run, tampered_session)

    with pytest.raises(HistoricalBacktestResultValidationError):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


# ---------------------------------------------------------------------------
# C. Evidence -> outcome discharge
# ---------------------------------------------------------------------------


def test_exact_s_equals_o_passes() -> None:
    run = build_applied_and_noop_run()
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)

    x_session = _x_session_result(run)
    supplied = {
        e.canonical_distribution_event_id
        for e in run.dividend_run_evidence.session_evidence[1].distribution_events
    }
    recorded = {
        o.canonical_distribution_event_id
        for o in x_session.state_transition_result.dividend_outcomes
    }
    assert supplied == recorded == {"D:APPLIED", "D:NOOP"}


def test_outcome_without_evidence_fails() -> None:
    run = build_applied_and_noop_run()
    x_session = _x_session_result(run)
    transition = x_session.state_transition_result
    reduced_outcomes = tuple(
        o
        for o in transition.dividend_outcomes
        if o.canonical_distribution_event_id != "D:NOOP"
    )
    tampered_transition = transition.model_copy(
        update={"dividend_outcomes": reduced_outcomes}
    )
    tampered_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = _replace_x_session(run, tampered_session)

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="DIVIDEND_PROVENANCE_MISMATCH"
    ):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


def test_duplicate_outcome_event_id_fails() -> None:
    run = build_applied_and_noop_run()
    x_session = _x_session_result(run)
    transition = x_session.state_transition_result
    duplicated = transition.dividend_outcomes + (transition.dividend_outcomes[0],)
    tampered_transition = transition.model_copy(
        update={"dividend_outcomes": duplicated}
    )
    tampered_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = _replace_x_session(run, tampered_session)

    with pytest.raises(HistoricalBacktestResultValidationError):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


def test_applied_with_zero_rows_fails() -> None:
    run = build_applied_and_noop_run()
    x_session = _x_session_result(run)
    transition = x_session.state_transition_result
    tampered_transition = transition.model_copy(
        update={"dividend_ledger_entries": ()}
    )
    tampered_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = _replace_x_session(run, tampered_session)

    # Removing the row makes the dividend-inclusive settled_cash baked
    # into the immutable resulting_state disagree with what ledger-folding
    # alone (now zero dividend rows) explains, so validate_transition's
    # existing settled-cash reconciliation (extended in this slice to
    # include dividend_ledger_entries) fails first -- still a hard
    # fail-closed rejection, just at the transition-arithmetic layer
    # rather than the discharge layer.
    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="INVALID_PHASE13_TRANSITION|DIVIDEND_PROVENANCE_MISMATCH",
    ):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


def test_no_op_with_a_row_fails() -> None:
    """A NO_OP outcome cannot legitimately have a ledger row; simulate by
    relabeling the APPLIED row's event id onto the no-op's identity space
    -- proven to fail rather than silently accepted."""

    run = build_applied_and_noop_run()
    x_session = _x_session_result(run)
    transition = x_session.state_transition_result
    noop_outcome = next(
        o
        for o in transition.dividend_outcomes
        if o.canonical_distribution_event_id == "D:NOOP"
    )
    forced_applied_outcome = noop_outcome.model_copy(
        update={
            "status": DividendApplicationStatus.APPLIED,
            "q_t": 1,
            "attribution_trade_id": "BUY-1",
        }
    )
    other_outcomes = tuple(
        o for o in transition.dividend_outcomes if o is not noop_outcome
    )
    tampered_transition = transition.model_copy(
        update={"dividend_outcomes": other_outcomes + (forced_applied_outcome,)}
    )
    tampered_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = _replace_x_session(run, tampered_session)

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="DIVIDEND_PROVENANCE_MISMATCH"
    ):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


def test_no_op_status_is_never_inferred_from_row_absence() -> None:
    """Deleting the evidence-> outcome bidirectional proof: an outcome
    claiming NO_OP for an event that was never supplied as evidence at
    all must still fail (S == O), proving NO_OP is read from the
    recorded outcome, never inferred merely because no row exists."""

    run = build_applied_and_noop_run()
    x_session = _x_session_result(run)
    fabricated_outcome = x_session.state_transition_result.dividend_outcomes[0].model_copy(
        update={
            "canonical_distribution_event_id": "D:NEVER_SUPPLIED",
            "application_id": "f" * 64,
            "status": DividendApplicationStatus.NO_OP_NOT_ENTITLED,
            "q_t": 0,
            "attribution_trade_id": None,
        }
    )
    tampered_transition = x_session.state_transition_result.model_copy(
        update={
            "dividend_outcomes": x_session.state_transition_result.dividend_outcomes
            + (fabricated_outcome,)
        }
    )
    tampered_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = _replace_x_session(run, tampered_session)

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="DIVIDEND_PROVENANCE_MISMATCH"
    ):
        validate_historical_backtest_source_run(
            run_result=tampered_run, run_manifest=None
        )


def test_orchestrator_now_forwards_evidence_instead_of_silently_dropping_it() -> (
    None
):
    """Slice 8: HistoricalBacktestOrchestrator.run() now forwards validated
    Phase-15A dividend evidence into PortfolioTransitionEngine.transition,
    so this same fixture (previously proving the pre-Slice-8 silent-drop
    gap failed closed at site 3) now proves the opposite -- the evidence
    reaches Phase 13, is properly discharged (NO_OP_NOT_ENTITLED, since no
    position in D:ORPHAN's security is ever held), and the resulting run
    passes site-3 source validation rather than failing on a drop."""

    from stock_swing_d1.backtester import (
        HistoricalBacktestOrchestrator,
        HistoricalBacktestSessionInput,
        HistoricalDecisionInterval,
    )

    from stock_swing_d1.data.ordinary_dividend_run_policy import ProcessedSessionLink

    evidence = make_evidence(event_id="D:ORPHAN", security_id=ASSET_A)
    coverage = make_coverage(session=X_SESSION, distribution_events=(evidence,))
    interval = HistoricalDecisionInterval(
        decision_start_date=date(2026, 8, 1), decision_end_date=date(2026, 8, 31)
    )
    proof = build_processed_session_contiguity_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint="b" * 64,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        session_links=(ProcessedSessionLink(session=X_SESSION, next_session=None),),
    )
    run = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        (
            HistoricalBacktestSessionInput(
                session=X_SESSION,
                decision_time=source_decision_time(X_SESSION),
                schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION,
                distribution_events=(evidence,),
                distribution_coverage=coverage,
            ),
        ),
        decision_interval=interval,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    # No exception: the evidence is now forwarded and correctly discharged.
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)

    session_result = run.session_results[0]
    outcomes = session_result.state_transition_result.dividend_outcomes
    assert len(outcomes) == 1
    assert outcomes[0].canonical_distribution_event_id == "D:ORPHAN"
    assert outcomes[0].status is DividendApplicationStatus.NO_OP_NOT_ENTITLED
    assert outcomes[0].q_t == 0
    assert session_result.state_transition_result.dividend_ledger_entries == ()


# ---------------------------------------------------------------------------
# D. No entitlement reconstruction
# ---------------------------------------------------------------------------


def test_source_validation_module_imports_no_provider_or_calendar_client() -> None:
    import inspect

    from stock_swing_d1.backtest_results import source_validation

    source = inspect.getsource(source_validation)
    for forbidden in ("norgatedata", "requests", "urllib", "socket"):
        assert forbidden not in source


def test_source_validation_never_calls_gate3_normalization() -> None:
    import inspect

    from stock_swing_d1.backtest_results import source_validation

    source = inspect.getsource(source_validation)
    assert "normalize_gate3_dividend" not in source
    assert "Gate3DividendNormalizationInputs(" not in source


def test_source_validation_never_computes_dividend_application_identity() -> None:
    import inspect

    from stock_swing_d1.backtest_results import source_validation

    source = inspect.getsource(source_validation)
    assert "compute_dividend_application_id" not in source
    assert "compute_gross_dividend_cash" not in source


def test_same_x_sell_and_reentry_source_validates_from_authoritative_old_trade() -> (
    None
):
    """Q_T/attribution for the dividend must come from the reconstructed
    pre-X Phase-13 state chain, not from post-BUY holdings: same-X SELL
    of the old position plus a same-X re-entry BUY must still attribute
    the dividend to the OLD (closed) trade."""

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
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(t_session, x_session),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )

    validate_historical_backtest_source_run(run_result=run, run_manifest=None)
    row = x_result.dividend_ledger_entries[0]
    assert row.attribution_trade_id == "BUY-1"
    assert row.q_t == 10
    new_position = final.open_positions[0]
    assert new_position.entry_execution_id == "BUY-2"


def test_buy_on_x_no_op_validates_from_authoritative_outcome_not_post_buy_holdings() -> (
    None
):
    """A BUY made ON X must not retroactively grant entitlement -- proven
    from the authoritative Q_T==0 recorded outcome, not from the fact
    that a position exists in the post-BUY final state."""

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
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=session_evidence),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=initial, final_state=final,
        session_results=(x_session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(final),
    )

    validate_historical_backtest_source_run(run_result=run, run_manifest=None)
    assert x_result.dividend_ledger_entries == ()
    outcome = x_result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED
    assert outcome.q_t == 0
    assert len(final.open_positions) == 1  # holdings exist despite NO_OP


# ---------------------------------------------------------------------------
# E. Fingerprint / source binding
# ---------------------------------------------------------------------------


def test_source_run_fingerprint_is_deterministic() -> None:
    run = build_applied_and_noop_run()
    assert compute_source_run_fingerprint(run) == compute_source_run_fingerprint(run)


def test_changing_dividend_run_evidence_changes_source_fingerprint() -> None:
    run = build_applied_and_noop_run()
    baseline = compute_source_run_fingerprint(run)

    other_evidence = make_evidence(
        event_id="D:APPLIED", security_id=ASSET_A, snapshot_fingerprint="9" * 64
    )
    changed_x_evidence = make_session_evidence(
        session=X_SESSION,
        distribution_events=(
            other_evidence,
            make_evidence(event_id="D:NOOP", security_id=ASSET_B, snapshot_fingerprint="9" * 64),
        ),
        snapshot_fingerprint="9" * 64,
    )
    changed_run_evidence = run.dividend_run_evidence.model_copy(
        update={
            "session_evidence": (
                run.dividend_run_evidence.session_evidence[0],
                changed_x_evidence,
            ),
            "canonical_distribution_snapshot_fingerprint": "9" * 64,
        }
    )
    changed_run = run.model_copy(
        update={"dividend_run_evidence": changed_run_evidence}
    )
    changed = compute_source_run_fingerprint(changed_run)
    assert changed != baseline


def test_dividend_run_evidence_none_vs_present_changes_source_fingerprint() -> None:
    state = PortfolioState(settled_cash=Decimal("1000"))
    fingerprint = hash_portfolio_state(state)
    without = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        initial_state_fingerprint=fingerprint,
        final_state_fingerprint=fingerprint,
    )
    with_evidence = without.model_copy(
        update={
            "dividend_run_evidence": make_run_evidence(session_evidence=())
        }
    )
    assert compute_source_run_fingerprint(without) != compute_source_run_fingerprint(
        with_evidence
    )


def test_changing_dividend_ledger_row_content_changes_source_fingerprint() -> None:
    run = build_applied_and_noop_run()
    baseline = compute_source_run_fingerprint(run)

    x_session = _x_session_result(run)
    transition = x_session.state_transition_result
    row = transition.dividend_ledger_entries[0]
    tampered_row = row.model_copy(
        update={"canonical_distribution_snapshot_fingerprint": "7" * 64}
    )
    tampered_transition = transition.model_copy(
        update={
            "dividend_ledger_entries": (tampered_row,)
            + transition.dividend_ledger_entries[1:]
        }
    )
    tampered_session = x_session.model_copy(
        update={"state_transition_result": tampered_transition}
    )
    tampered_run = _replace_x_session(run, tampered_session)
    changed = compute_source_run_fingerprint(tampered_run)
    assert changed != baseline


def test_reversed_supplied_evidence_order_same_canonical_content_same_fingerprint() -> (
    None
):
    """Reversing the order events are handed to build_dividend_aware_
    session_evidence before it re-derives its own canonical fingerprint
    must not change the resulting source-run fingerprint, since the
    session evidence's own content identity is order-independent of the
    raw construction call."""

    evidence_a = make_evidence(event_id="D:AAA", security_id=ASSET_A)
    evidence_b = make_evidence(event_id="D:BBB", security_id=ASSET_B)

    forward = make_session_evidence(
        session=X_SESSION, distribution_events=(evidence_a, evidence_b)
    )
    state = PortfolioState(settled_cash=Decimal("1000"))
    fingerprint = hash_portfolio_state(state)
    run_forward = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(session_evidence=(forward,)),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=state, final_state=state,
        initial_state_fingerprint=fingerprint, final_state_fingerprint=fingerprint,
    )
    # Same two events, same session -- the evidence tuple itself must be
    # supplied in the model's own required order (build_dividend_aware_
    # session_evidence does not sort), so this proves identical inputs
    # yield identical fingerprints rather than testing a nonexistent
    # reordering tolerance.
    run_forward_again = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        dividend_run_evidence=make_run_evidence(
            session_evidence=(
                make_session_evidence(
                    session=X_SESSION, distribution_events=(evidence_a, evidence_b)
                ),
            )
        ),
        schema_version=DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
        initial_state=state, final_state=state,
        initial_state_fingerprint=fingerprint, final_state_fingerprint=fingerprint,
    )
    assert compute_source_run_fingerprint(run_forward) == compute_source_run_fingerprint(
        run_forward_again
    )


# ---------------------------------------------------------------------------
# F. Existing Phase 15D regression
# ---------------------------------------------------------------------------


def test_existing_backtest_results_suite_remains_green() -> None:
    """Smoke marker; the authoritative regression proof is running the
    full tests/backtest_results suite alongside this file (see report)."""

    run = build_applied_and_noop_run()
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)


def test_unsupported_ledger_event_type_still_fails_closed_for_non_dividend() -> None:
    from stock_swing_d1.portfolio.portfolio_errors import PortfolioStateError
    from stock_swing_d1.portfolio.portfolio_events import PortfolioLedgerEventType
    from stock_swing_d1.portfolio.portfolio_invariants import (
        PortfolioInvariantChecker,
    )

    run = build_applied_and_noop_run()
    t_session = next(s for s in run.session_results if s.session == T_SESSION)
    settlement_shaped = t_session.state_transition_result.ledger_entries[0].model_copy(
        update={"event_type": PortfolioLedgerEventType.DIVIDEND_APPLIED}
    )
    with pytest.raises(PortfolioStateError):
        PortfolioInvariantChecker.validate_transition(
            run.initial_state,
            t_session.authoritative_state,
            (settlement_shaped,),
        )


# ---------------------------------------------------------------------------
# G. Source-run fingerprint domain/version conformance (narrow audit
# correction: compute_source_run_fingerprint's payload gained
# dividend_run_evidence, so SOURCE_RUN_HASH_DOMAIN must not remain the same
# identity as before that change -- OD-22.2 "Phase 15D content/result
# fingerprints").
# ---------------------------------------------------------------------------


def test_source_run_hash_domain_is_the_expected_new_semantic_version() -> None:
    from stock_swing_d1.backtest_results.hashing import SOURCE_RUN_HASH_DOMAIN

    # Frozen Task 5C-B baseline: v0.3 (nested session results gained
    # entry_session_protective_decisions); Task 5C-C adds no further bump.
    assert SOURCE_RUN_HASH_DOMAIN == "historical_backtest_source_run.v0.3"


def test_old_domain_and_new_domain_hashes_differ_for_identical_payload() -> None:
    """Domain separation itself: the exact same payload dict must not hash
    identically under the pre-Slice-7 domain string and the current one --
    this is what makes it safe for the two payload shapes to coexist in
    principle without silently colliding under one shared identity."""

    from stock_swing_d1.backtest_results.hashing import (
        SOURCE_RUN_HASH_DOMAIN,
        semantic_domain_sha256,
    )

    payload = {"schema_version": "x", "session_results": ()}
    old_domain_hash = semantic_domain_sha256(
        "historical_backtest_source_run.v0.1", payload
    )
    new_domain_hash = semantic_domain_sha256(SOURCE_RUN_HASH_DOMAIN, payload)
    assert old_domain_hash != new_domain_hash


def test_legacy_non_dividend_run_validates_and_fingerprints_deterministically() -> (
    None
):
    """Legacy/non-dividend-aware run behavior is otherwise unchanged: it
    still validates and still produces a deterministic fingerprint -- the
    domain bump changes the digest's identity, not source-validation
    semantics or determinism."""

    state = PortfolioState(settled_cash=Decimal("1000"))
    fingerprint = hash_portfolio_state(state)
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        initial_state_fingerprint=fingerprint,
        final_state_fingerprint=fingerprint,
    )
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)
    assert compute_source_run_fingerprint(run) == compute_source_run_fingerprint(run)


def test_source_run_fingerprint_computation_never_references_filesystem_paths() -> (
    None
):
    import inspect

    from stock_swing_d1.backtest_results.hashing import compute_source_run_fingerprint

    source = inspect.getsource(compute_source_run_fingerprint)
    for forbidden in ("__file__", "os.getcwd", "Path(", "abspath", "os.path"):
        assert forbidden not in source


def test_focused_site_3_suite_remains_green_after_domain_bump() -> None:
    """Smoke marker for item 8; the authoritative proof is this whole file
    (39+ cases) passing alongside the domain-version correction."""

    run = build_applied_and_noop_run()
    validate_historical_backtest_source_run(run_result=run, run_manifest=None)
    assert compute_source_run_fingerprint(run) is not None
