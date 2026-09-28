# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
"""Slice 4 acceptance tests: Phase 15A dividend-aware evidence integration.

Covers the Phase 15A portions of the frozen Ordinary Dividend Amendment
v0.7 (`phase-13-15a-15d-ordinary-dividend-amendment-v0.7-FROZEN.md`):
run-level mode activation (Sec. 3.3/3.4), per-session coverage (OD-7),
processed-session contiguity attachment (OD-7.4/OD-7.5/OD-15.7), event
placement and T/X adjacency (OD-5, OD-15.5, OD-15.5a), and run/snapshot
identity (OD-20.1). This slice validates and carries evidence only; it
does not implement Phase 13 entitlement, cash, or ledger economics.
"""

from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtester import (
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionInput,
    HistoricalDecisionInterval,
)
from stock_swing_d1.backtester.models import (
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION,
    HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
)
from stock_swing_d1.backtester.validation import (
    HistoricalBacktestValidationError,
    validate_dividend_aware_run_evidence,
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
    build_affirmative_distribution_coverage,
)
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    OrdinaryDividendAccountingPolicyRef,
    ProcessedSessionContiguityProof,
    ProcessedSessionLink,
    build_ordinary_dividend_accounting_policy_ref,
    build_processed_session_contiguity_proof,
)
from stock_swing_d1.models import CorporateActionEvent
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState


WEDNESDAY = date(2026, 8, 19)
THURSDAY = date(2026, 8, 20)
FRIDAY = date(2026, 8, 21)
MONDAY = date(2026, 8, 24)

SNAPSHOT_FINGERPRINT = "c" * 64
COVERAGE_SOURCE_FINGERPRINT = "d" * 64
CALENDAR_SOURCE_FINGERPRINT = "b" * 64
CLASSIFICATION_SOURCE_FINGERPRINT = "a" * 64
CALENDAR_SOURCE_ID = "accepted-us-equity-calendar"
CALENDAR_POLICY_ID = "canonical-next-session"
CALENDAR_POLICY_VERSION = "1"
COVERAGE_CONTRACT_ID = "affirmative-coverage.v1"
COVERAGE_CONTRACT_VERSION = "1"
CLASSIFICATION_CONTRACT_ID = "ordinary-cash-classification.v1"
SECURITY_ID = "NORGATE:1900000003"


def make_event(
    *,
    event_id: str,
    entitlement_session: date,
    ex_session: date,
    security_id: str = SECURITY_ID,
    snapshot_fingerprint: str = SNAPSHOT_FINGERPRINT,
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
        upstream_source_evidence_fingerprint=CLASSIFICATION_SOURCE_FINGERPRINT,
    )
    calendar = build_dividend_calendar_resolution_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        entitlement_session=entitlement_session,
        ex_session=ex_session,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint=CALENDAR_SOURCE_FINGERPRINT,
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
    snapshot_fingerprint: str = SNAPSHOT_FINGERPRINT,
    policy_ref: OrdinaryDividendAccountingPolicyRef = (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
    ),
) -> CanonicalDistributionCoverage:
    return build_affirmative_distribution_coverage(
        session=session,
        coverage_contract_id=COVERAGE_CONTRACT_ID,
        coverage_contract_version=COVERAGE_CONTRACT_VERSION,
        canonical_distribution_snapshot_fingerprint=snapshot_fingerprint,
        upstream_source_evidence_fingerprint=COVERAGE_SOURCE_FINGERPRINT,
        dividend_accounting_policy_ref=policy_ref,
    )


def make_links(*sessions: date) -> tuple[ProcessedSessionLink, ...]:
    return tuple(
        ProcessedSessionLink(
            session=session,
            next_session=(
                None if index == len(sessions) - 1 else sessions[index + 1]
            ),
        )
        for index, session in enumerate(sessions)
    )


def make_proof(
    *,
    sessions: tuple[date, ...],
    policy_ref: OrdinaryDividendAccountingPolicyRef = (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
    ),
) -> ProcessedSessionContiguityProof:
    return build_processed_session_contiguity_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint=CALENDAR_SOURCE_FINGERPRINT,
        dividend_accounting_policy_ref=policy_ref,
        session_links=make_links(*sessions),
    )


def session_input(
    session: date,
    *,
    next_session: date | None = None,
    distribution_events: tuple[CanonicalDividendAccountingEvidence, ...] = (),
    distribution_coverage: CanonicalDistributionCoverage | None = None,
    pre_open_corporate_actions: tuple[CorporateActionEvent, ...] = (),
    dividend_aware_schema: bool = True,
) -> HistoricalBacktestSessionInput:
    kwargs: dict[str, object] = {
        "session": session,
        "decision_time": _decision_time(session),
        "next_session": next_session,
        "pre_open_corporate_actions": pre_open_corporate_actions,
    }
    if dividend_aware_schema:
        kwargs["schema_version"] = (
            DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION
        )
        kwargs["distribution_events"] = distribution_events
        kwargs["distribution_coverage"] = distribution_coverage
    return HistoricalBacktestSessionInput(**kwargs)


def _decision_time(session: date):
    from datetime import datetime, timezone

    return datetime(
        session.year, session.month, session.day, 16, 0, tzinfo=timezone.utc
    )


DECISION_INTERVAL = HistoricalDecisionInterval(
    decision_start_date=date(2026, 8, 1),
    decision_end_date=date(2026, 9, 30),
)


def run_with(
    sessions: tuple[HistoricalBacktestSessionInput, ...],
    *,
    dividend_accounting_policy_ref=None,
    processed_session_contiguity_proof=None,
):
    return HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        sessions,
        decision_interval=DECISION_INTERVAL,
        dividend_accounting_policy_ref=dividend_accounting_policy_ref,
        processed_session_contiguity_proof=processed_session_contiguity_proof,
    )


# ---------------------------------------------------------------------------
# A. Mode activation
# ---------------------------------------------------------------------------


def test_legacy_run_without_dividend_params_is_unaffected() -> None:
    result = run_with((session_input(FRIDAY, dividend_aware_schema=False),))

    assert result.dividend_run_evidence is None
    assert result.schema_version == HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION


def test_non_dividend_aware_run_rejects_distribution_events() -> None:
    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=THURSDAY
    )
    sessions = (session_input(THURSDAY, distribution_events=(event,)),)

    with pytest.raises(
        HistoricalBacktestValidationError, match="ORDINARY_DIVIDEND_MODE_MISMATCH"
    ):
        run_with(sessions)


def test_non_dividend_aware_run_rejects_coverage() -> None:
    coverage = make_coverage(session=THURSDAY)
    sessions = (session_input(THURSDAY, distribution_coverage=coverage),)

    with pytest.raises(
        HistoricalBacktestValidationError, match="ORDINARY_DIVIDEND_MODE_MISMATCH"
    ):
        run_with(sessions)


def test_non_dividend_aware_run_rejects_contiguity_proof() -> None:
    proof = make_proof(sessions=(THURSDAY,))
    sessions = (session_input(THURSDAY, dividend_aware_schema=False),)

    with pytest.raises(
        HistoricalBacktestValidationError, match="ORDINARY_DIVIDEND_MODE_MISMATCH"
    ):
        run_with(sessions, processed_session_contiguity_proof=proof)


def test_legacy_no_dividend_run_remains_valid_with_multiple_sessions() -> None:
    result = run_with(
        (
            session_input(FRIDAY, next_session=MONDAY, dividend_aware_schema=False),
            session_input(MONDAY, dividend_aware_schema=False),
        )
    )

    assert result.dividend_run_evidence is None


def test_dividend_aware_nonempty_run_requires_contiguity_proof() -> None:
    coverage = make_coverage(session=THURSDAY)
    sessions = (session_input(THURSDAY, distribution_coverage=coverage),)

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_PROCESSED_SPAN_NOT_CONTIGUOUS",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        )


def test_dividend_events_do_not_implicitly_activate_mode() -> None:
    """Supplying events with no policy ref fails MODE_MISMATCH, not silently."""

    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=THURSDAY
    )
    coverage = make_coverage(session=THURSDAY)
    sessions = (
        session_input(
            THURSDAY,
            distribution_events=(event,),
            distribution_coverage=coverage,
        ),
    )

    with pytest.raises(
        HistoricalBacktestValidationError, match="ORDINARY_DIVIDEND_MODE_MISMATCH"
    ):
        run_with(sessions)


def test_session_contract_failure_precedes_dividend_preflight_failure() -> None:
    """Frozen validation order is chronology -> containment -> existing
    session contract -> dividend-aware pre-flight -> orchestration. When a
    session violates both the existing session contract (an unsupported
    pre-open corporate action) and dividend-aware evidence (missing
    coverage), the session-contract failure must win because it is
    validated first.
    """

    corporate_action = CorporateActionEvent(
        security_id="NORGATE:1",
        symbol="S1",
        event_date=THURSDAY,
        event_type="split",
        source_asset_id=1,
        date_semantics="effective_date",
        old_shares=1.0,
        new_shares=2.0,
        terms_verified=True,
        source_provider="Norgate Data",
    )
    # distribution_coverage is omitted, so this session is also invalid
    # dividend-aware evidence (would fail ORDINARY_DIVIDEND_COVERAGE_MISSING
    # if dividend pre-flight ran first).
    sessions = (
        session_input(
            THURSDAY,
            pre_open_corporate_actions=(corporate_action,),
        ),
    )
    proof = make_proof(sessions=(THURSDAY,))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="UNSUPPORTED_CORPORATE_ACTION_TRANSITION",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_valid_session_contract_proceeds_to_dividend_preflight_failure() -> None:
    """A structurally valid session contract (no corporate-action violation)
    must proceed past session-contract validation and then fail on the
    dividend-aware pre-flight (missing coverage)."""

    sessions = (session_input(THURSDAY),)
    proof = make_proof(sessions=(THURSDAY,))

    with pytest.raises(
        HistoricalBacktestValidationError, match="ORDINARY_DIVIDEND_COVERAGE_MISSING"
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


# ---------------------------------------------------------------------------
# B. Zero-session compatibility
# ---------------------------------------------------------------------------


def test_zero_session_run_does_not_require_dividend_evidence() -> None:
    result = run_with(())

    assert result.dividend_run_evidence is None
    assert result.schema_version == HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION


def test_zero_session_dividend_aware_run_permits_explicit_policy() -> None:
    result = run_with(
        (), dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
    )

    assert result.dividend_run_evidence is not None
    assert result.dividend_run_evidence.session_evidence == ()
    assert result.dividend_run_evidence.processed_session_contiguity_proof is None
    assert result.schema_version == (
        DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION
    )


def test_zero_session_run_rejects_fake_contiguity_proof() -> None:
    proof = make_proof(sessions=(THURSDAY,))

    with pytest.raises(
        HistoricalBacktestValidationError, match="ORDINARY_DIVIDEND_MODE_MISMATCH"
    ):
        run_with(
            (),
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


# ---------------------------------------------------------------------------
# C. Contiguity attachment
# ---------------------------------------------------------------------------


def test_valid_two_session_contiguity_proof_matches_run() -> None:
    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    proof = make_proof(sessions=(THURSDAY, FRIDAY))
    sessions = (
        session_input(
            THURSDAY, next_session=FRIDAY, distribution_coverage=coverage_thu
        ),
        session_input(FRIDAY, distribution_coverage=coverage_fri),
    )

    result = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert result.dividend_run_evidence.processed_session_contiguity_proof == proof
    assert len(result.dividend_run_evidence.session_evidence) == 2


def test_missing_nonfinal_next_session_fails_contiguity() -> None:
    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    proof = make_proof(sessions=(THURSDAY, FRIDAY))
    sessions = (
        session_input(THURSDAY, distribution_coverage=coverage_thu),
        session_input(FRIDAY, distribution_coverage=coverage_fri),
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_PROCESSED_SPAN_NOT_CONTIGUOUS",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_nonfinal_next_session_differing_from_proof_fails() -> None:
    """This divergence cannot survive the full run() chronology gate (which
    already requires next_session to equal the actual following session), so
    it is exercised directly against the dividend validation function to
    prove its own defense-in-depth check against the bound proof.
    """

    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    proof = make_proof(sessions=(THURSDAY, FRIDAY))
    sessions = (
        session_input(
            THURSDAY, next_session=MONDAY, distribution_coverage=coverage_thu
        ),
        session_input(FRIDAY, distribution_coverage=coverage_fri),
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_PROCESSED_SPAN_NOT_CONTIGUOUS",
    ):
        validate_dividend_aware_run_evidence(
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
            sessions=sessions,
        )


def test_proof_sequence_differing_from_actual_sessions_fails() -> None:
    coverage_thu = make_coverage(session=THURSDAY)
    proof = make_proof(sessions=(THURSDAY, FRIDAY))
    sessions = (session_input(THURSDAY, distribution_coverage=coverage_thu),)

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_PROCESSED_SPAN_NOT_CONTIGUOUS",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_proof_and_run_policy_ref_are_bound_and_checked() -> None:
    """v0.1 has exactly one frozen policy identity, so this documents that
    the run/proof policy equality check is exercised (same singleton) rather
    than skipped; a genuinely differing policy is not constructible under
    the current frozen Literal-typed policy schema.
    """

    coverage_thu = make_coverage(session=THURSDAY)
    proof = make_proof(
        sessions=(THURSDAY,), policy_ref=build_ordinary_dividend_accounting_policy_ref()
    )
    sessions = (session_input(THURSDAY, distribution_coverage=coverage_thu),)

    result = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert result.dividend_run_evidence.dividend_accounting_policy_ref == (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
    )


def test_final_session_has_no_next_session_presence_obligation() -> None:
    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    proof = make_proof(sessions=(THURSDAY, FRIDAY))
    sessions = (
        session_input(
            THURSDAY, next_session=FRIDAY, distribution_coverage=coverage_thu
        ),
        session_input(FRIDAY, next_session=None, distribution_coverage=coverage_fri),
    )

    result = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert result.dividend_run_evidence is not None


def test_generic_fallback_does_not_repair_missing_dividend_aware_next_session() -> (
    None
):
    """The legacy following-session fallback used elsewhere must not mask this."""

    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    proof = make_proof(sessions=(THURSDAY, FRIDAY))
    sessions = (
        # next_session omitted: legacy orchestration would silently infer
        # FRIDAY as the "following" session; dividend-aware mode must not.
        session_input(THURSDAY, distribution_coverage=coverage_thu),
        session_input(FRIDAY, distribution_coverage=coverage_fri),
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_PROCESSED_SPAN_NOT_CONTIGUOUS",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


# ---------------------------------------------------------------------------
# D. Coverage
# ---------------------------------------------------------------------------


def test_affirmative_coverage_with_one_event_passes() -> None:
    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=THURSDAY
    )
    coverage = make_coverage(session=THURSDAY)
    sessions = (
        session_input(
            THURSDAY, distribution_events=(event,), distribution_coverage=coverage
        ),
    )
    proof = make_proof(sessions=(THURSDAY,))

    result = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert len(result.dividend_run_evidence.session_evidence[0].distribution_events) == 1


def test_affirmative_coverage_with_empty_events_passes() -> None:
    coverage = make_coverage(session=THURSDAY)
    sessions = (session_input(THURSDAY, distribution_coverage=coverage),)
    proof = make_proof(sessions=(THURSDAY,))

    result = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert result.dividend_run_evidence.session_evidence[0].distribution_events == ()


def test_missing_coverage_fails() -> None:
    sessions = (session_input(THURSDAY),)
    proof = make_proof(sessions=(THURSDAY,))

    with pytest.raises(
        HistoricalBacktestValidationError, match="ORDINARY_DIVIDEND_COVERAGE_MISSING"
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_coverage_session_mismatch_fails() -> None:
    coverage_wrong_session = make_coverage(session=FRIDAY)
    sessions = (
        session_input(THURSDAY, distribution_coverage=coverage_wrong_session),
    )
    proof = make_proof(sessions=(THURSDAY,))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_SESSION_EVIDENCE_INVALID",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_different_snapshots_across_sessions_fail() -> None:
    coverage_thu = make_coverage(session=THURSDAY, snapshot_fingerprint="c" * 64)
    coverage_fri = make_coverage(session=FRIDAY, snapshot_fingerprint="e" * 64)
    sessions = (
        session_input(
            THURSDAY, next_session=FRIDAY, distribution_coverage=coverage_thu
        ),
        session_input(FRIDAY, distribution_coverage=coverage_fri),
    )
    proof = make_proof(sessions=(THURSDAY, FRIDAY))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_RUN_EVIDENCE_INVALID",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_event_snapshot_differing_from_coverage_snapshot_fails() -> None:
    event_wrong_snapshot = make_event(
        event_id="D:1",
        entitlement_session=WEDNESDAY,
        ex_session=THURSDAY,
        snapshot_fingerprint="f" * 64,
    )
    coverage = make_coverage(session=THURSDAY, snapshot_fingerprint=SNAPSHOT_FINGERPRINT)
    sessions = (
        session_input(
            THURSDAY,
            distribution_events=(event_wrong_snapshot,),
            distribution_coverage=coverage,
        ),
    )
    proof = make_proof(sessions=(THURSDAY,))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_SESSION_EVIDENCE_INVALID",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


# ---------------------------------------------------------------------------
# E. Event placement and duplicate identity
# ---------------------------------------------------------------------------


def test_event_placed_on_wrong_session_fails() -> None:
    """T=WEDNESDAY, X=THURSDAY, but the event is attached to FRIDAY's input."""

    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=THURSDAY
    )
    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    sessions = (
        session_input(
            THURSDAY, next_session=FRIDAY, distribution_coverage=coverage_thu
        ),
        session_input(
            FRIDAY,
            distribution_events=(event,),
            distribution_coverage=coverage_fri,
        ),
    )
    proof = make_proof(sessions=(THURSDAY, FRIDAY))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_SESSION_EVIDENCE_INVALID",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_duplicate_event_id_within_session_fails() -> None:
    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=THURSDAY
    )
    coverage = make_coverage(session=THURSDAY)
    sessions = (
        session_input(
            THURSDAY,
            distribution_events=(event, event),
            distribution_coverage=coverage,
        ),
    )
    proof = make_proof(sessions=(THURSDAY,))

    # The run-wide uniqueness check (which also covers within-session
    # duplicates) fires first; the diagnosis is still fail-closed.
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_DUPLICATE_EVENT",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_duplicate_event_id_across_sessions_fails() -> None:
    event_thu = make_event(
        event_id="D:1",
        entitlement_session=WEDNESDAY,
        ex_session=THURSDAY,
        security_id="NORGATE:1",
    )
    event_fri = make_event(
        event_id="D:1",
        entitlement_session=THURSDAY,
        ex_session=FRIDAY,
        security_id="NORGATE:2",
    )
    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    sessions = (
        session_input(
            THURSDAY,
            next_session=FRIDAY,
            distribution_events=(event_thu,),
            distribution_coverage=coverage_thu,
        ),
        session_input(
            FRIDAY,
            distribution_events=(event_fri,),
            distribution_coverage=coverage_fri,
        ),
    )
    proof = make_proof(sessions=(THURSDAY, FRIDAY))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_DUPLICATE_EVENT",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


# ---------------------------------------------------------------------------
# F. T -> X adjacency (OD-15.5a)
# ---------------------------------------------------------------------------


def test_processed_t_immediately_before_x_passes() -> None:
    event = make_event(
        event_id="D:1", entitlement_session=THURSDAY, ex_session=FRIDAY
    )
    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    sessions = (
        session_input(
            THURSDAY, next_session=FRIDAY, distribution_coverage=coverage_thu
        ),
        session_input(
            FRIDAY,
            distribution_events=(event,),
            distribution_coverage=coverage_fri,
        ),
    )
    proof = make_proof(sessions=(THURSDAY, FRIDAY))

    result = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert result.dividend_run_evidence is not None


def test_intervening_processed_session_between_t_and_x_fails() -> None:
    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=FRIDAY
    )
    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    sessions = (
        session_input(
            THURSDAY, next_session=FRIDAY, distribution_coverage=coverage_thu
        ),
        session_input(
            FRIDAY,
            distribution_events=(event,),
            distribution_coverage=coverage_fri,
        ),
    )
    proof = make_proof(sessions=(THURSDAY, FRIDAY))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_SESSION_NOT_ADJACENT",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_multiple_intervening_processed_sessions_fail() -> None:
    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=MONDAY
    )
    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    coverage_mon = make_coverage(session=MONDAY)
    sessions = (
        session_input(
            THURSDAY, next_session=FRIDAY, distribution_coverage=coverage_thu
        ),
        session_input(
            FRIDAY, next_session=MONDAY, distribution_coverage=coverage_fri
        ),
        session_input(
            MONDAY,
            distribution_events=(event,),
            distribution_coverage=coverage_mon,
        ),
    )
    proof = make_proof(sessions=(THURSDAY, FRIDAY, MONDAY))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_SESSION_NOT_ADJACENT",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_prerun_t_with_first_processed_x_may_pass() -> None:
    """T predates the run entirely; X is the first processed session."""

    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=THURSDAY
    )
    coverage = make_coverage(session=THURSDAY)
    sessions = (
        session_input(
            THURSDAY, distribution_events=(event,), distribution_coverage=coverage
        ),
    )
    proof = make_proof(sessions=(THURSDAY,))

    result = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert len(result.dividend_run_evidence.session_evidence[0].distribution_events) == 1


def test_prerun_t_with_later_processed_x_and_intervening_session_fails() -> None:
    """T predates the run; X is a LATER processed session with a gap between."""

    far_past_t = date(2026, 8, 1)
    event = make_event(
        event_id="D:1", entitlement_session=far_past_t, ex_session=FRIDAY
    )
    coverage_thu = make_coverage(session=THURSDAY)
    coverage_fri = make_coverage(session=FRIDAY)
    sessions = (
        session_input(
            THURSDAY, next_session=FRIDAY, distribution_coverage=coverage_thu
        ),
        session_input(
            FRIDAY,
            distribution_events=(event,),
            distribution_coverage=coverage_fri,
        ),
    )
    proof = make_proof(sessions=(THURSDAY, FRIDAY))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_SESSION_NOT_ADJACENT",
    ):
        run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )


def test_adjacency_validation_does_not_query_a_calendar() -> None:
    source = inspect.getsource(validate_dividend_aware_run_evidence)

    for forbidden in ("socket", "requests", "norgatedata", "calendar."):
        assert forbidden not in source.lower()


# ---------------------------------------------------------------------------
# G. Run/snapshot identity
# ---------------------------------------------------------------------------


def test_run_input_fingerprint_is_deterministic_across_identical_runs() -> None:
    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=THURSDAY
    )
    coverage = make_coverage(session=THURSDAY)
    sessions = (
        session_input(
            THURSDAY, distribution_events=(event,), distribution_coverage=coverage
        ),
    )
    proof = make_proof(sessions=(THURSDAY,))

    first = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )
    second = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert (
        first.dividend_run_evidence.run_input_fingerprint
        == second.dividend_run_evidence.run_input_fingerprint
    )


def test_changing_snapshot_changes_run_identity() -> None:
    def build(snapshot: str) -> DividendAwareRunEvidence:
        event = make_event(
            event_id="D:1",
            entitlement_session=WEDNESDAY,
            ex_session=THURSDAY,
            snapshot_fingerprint=snapshot,
        )
        coverage = make_coverage(session=THURSDAY, snapshot_fingerprint=snapshot)
        sessions = (
            session_input(
                THURSDAY,
                distribution_events=(event,),
                distribution_coverage=coverage,
            ),
        )
        proof = make_proof(sessions=(THURSDAY,))
        result = run_with(
            sessions,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
        )
        return result.dividend_run_evidence

    first = build(SNAPSHOT_FINGERPRINT)
    second = build("e" * 64)

    assert first.run_input_fingerprint != second.run_input_fingerprint


def test_changing_contiguity_fingerprint_changes_run_identity() -> None:
    coverage = make_coverage(session=THURSDAY)
    event = make_event(
        event_id="D:1", entitlement_session=WEDNESDAY, ex_session=THURSDAY
    )
    sessions = (
        session_input(
            THURSDAY, distribution_events=(event,), distribution_coverage=coverage
        ),
    )
    proof_a = build_processed_session_contiguity_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint=CALENDAR_SOURCE_FINGERPRINT,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        session_links=make_links(THURSDAY),
    )
    proof_b = build_processed_session_contiguity_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint="9" * 64,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        session_links=make_links(THURSDAY),
    )

    first = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof_a,
    )
    second = run_with(
        sessions,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof_b,
    )

    assert (
        first.dividend_run_evidence.run_input_fingerprint
        != second.dividend_run_evidence.run_input_fingerprint
    )


# ---------------------------------------------------------------------------
# H. Decision Interval non-interference
# ---------------------------------------------------------------------------


def test_out_of_interval_session_still_fails_with_existing_decision_interval_error() -> (
    None
):
    narrow_interval = HistoricalDecisionInterval(
        decision_start_date=THURSDAY, decision_end_date=THURSDAY
    )
    coverage = make_coverage(session=FRIDAY)

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="SESSION_OUTSIDE_DECISION_INTERVAL",
    ):
        HistoricalBacktestOrchestrator().run(
            PortfolioState(settled_cash=Decimal("1000")),
            (session_input(FRIDAY, distribution_coverage=coverage),),
            decision_interval=narrow_interval,
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=make_proof(sessions=(FRIDAY,)),
        )


def test_prerun_t_evidence_not_rejected_solely_for_predating_interval() -> None:
    interval = HistoricalDecisionInterval(
        decision_start_date=THURSDAY, decision_end_date=FRIDAY
    )
    far_past_t = date(2026, 1, 1)
    event = make_event(
        event_id="D:1", entitlement_session=far_past_t, ex_session=THURSDAY
    )
    coverage = make_coverage(session=THURSDAY)
    sessions = (
        session_input(
            THURSDAY, distribution_events=(event,), distribution_coverage=coverage
        ),
    )
    proof = make_proof(sessions=(THURSDAY,))

    result = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        sessions,
        decision_interval=interval,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert result.dividend_run_evidence is not None


def test_no_out_of_interval_terminal_session_is_invented() -> None:
    interval = HistoricalDecisionInterval(
        decision_start_date=THURSDAY, decision_end_date=THURSDAY
    )
    coverage = make_coverage(session=THURSDAY)
    sessions = (session_input(THURSDAY, distribution_coverage=coverage),)
    proof = make_proof(sessions=(THURSDAY,))

    result = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        sessions,
        decision_interval=interval,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
    )

    assert tuple(item.session for item in result.session_results) == (THURSDAY,)


# ---------------------------------------------------------------------------
# I. Phase 15A architectural prohibitions
# ---------------------------------------------------------------------------


def test_dividend_validation_does_not_inspect_portfolio_state() -> None:
    source = inspect.getsource(validate_dividend_aware_run_evidence)

    for forbidden in (
        "portfoliostate",
        "open_positions",
        "settled_cash",
        "quantity",
    ):
        assert forbidden not in source.lower()


def test_dividend_validation_does_not_compute_q_t_or_normalize_d_h() -> None:
    source = inspect.getsource(validate_dividend_aware_run_evidence)

    for forbidden in ("q_t", "d_h", "normalize", "gross_cash"):
        assert forbidden not in source.lower()
