"""Acceptance tests for ordinary-dividend Slice 4 run-evidence primitives.

Direct unit coverage for `stock_swing_d1.data.ordinary_dividend_run_evidence`
(distribution coverage, per-session evidence, and run-level evidence
identity), independent of the Phase 15A orchestration wiring exercised in
`tests/backtester/test_phase15a_dividend_evidence.py`.
"""

from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

import stock_swing_d1.data.ordinary_dividend_run_evidence as run_evidence_module
from stock_swing_d1.data.ordinary_dividend_accounting import (
    CANONICAL_DIVIDEND_CURRENCY,
    build_canonical_dividend_accounting_evidence,
    build_dividend_calendar_resolution_proof,
    build_ordinary_cash_classification_proof,
)
from stock_swing_d1.data.ordinary_dividend_normalization import (
    Gate3DividendNormalizationInputs,
    normalize_gate3_dividend,
)
from stock_swing_d1.data.ordinary_dividend_run_evidence import (
    DISTRIBUTION_COVERAGE_SCOPE,
    CanonicalDistributionCoverage,
    DistributionCoverageStatus,
    DividendAwareRunEvidence,
    DividendAwareSessionEvidence,
    build_affirmative_distribution_coverage,
    build_dividend_aware_run_evidence,
    build_dividend_aware_session_evidence,
    compute_dividend_accounting_evidence_content_fingerprint,
    compute_distribution_coverage_fingerprint,
)
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    ProcessedSessionLink,
    build_processed_session_contiguity_proof,
)


THURSDAY = date(2026, 8, 20)
FRIDAY = date(2026, 8, 21)
SNAPSHOT = "c" * 64
SOURCE_FINGERPRINT = "d" * 64
CALENDAR_FINGERPRINT = "b" * 64
CLASSIFICATION_FINGERPRINT = "a" * 64
CALENDAR_SOURCE_ID = "accepted-us-equity-calendar"
CALENDAR_POLICY_ID = "canonical-next-session"
CALENDAR_POLICY_VERSION = "1"
COVERAGE_CONTRACT_ID = "affirmative-coverage.v1"
COVERAGE_CONTRACT_VERSION = "1"


def make_coverage(
    *, session: date = THURSDAY, snapshot: str = SNAPSHOT
) -> CanonicalDistributionCoverage:
    return build_affirmative_distribution_coverage(
        session=session,
        coverage_contract_id=COVERAGE_CONTRACT_ID,
        coverage_contract_version=COVERAGE_CONTRACT_VERSION,
        canonical_distribution_snapshot_fingerprint=snapshot,
        upstream_source_evidence_fingerprint=SOURCE_FINGERPRINT,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    )


def make_event(
    *,
    event_id: str = "D:1",
    entitlement_session: date = THURSDAY,
    ex_session: date = FRIDAY,
    security_id: str = "NORGATE:1",
    snapshot: str = SNAPSHOT,
):
    inputs = Gate3DividendNormalizationInputs(
        entitlement_session=entitlement_session,
        d_capitalspecial=Decimal("2"),
        unadjusted_close_t=Decimal("3"),
        close_capital_t=Decimal("4"),
    )
    result = normalize_gate3_dividend(inputs)
    classification = build_ordinary_cash_classification_proof(
        canonical_distribution_event_id=event_id,
        classification_contract_id="ordinary-cash-classification.v1",
        upstream_source_evidence_fingerprint=CLASSIFICATION_FINGERPRINT,
    )
    calendar = build_dividend_calendar_resolution_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        entitlement_session=entitlement_session,
        ex_session=ex_session,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint=CALENDAR_FINGERPRINT,
    )
    return build_canonical_dividend_accounting_evidence(
        canonical_distribution_event_id=event_id,
        canonical_security_id=security_id,
        normalization_inputs=inputs,
        normalization_result=result,
        classification_proof=classification,
        calendar_resolution_proof=calendar,
        currency=CANONICAL_DIVIDEND_CURRENCY,
        canonical_distribution_snapshot_fingerprint=snapshot,
    )


# ---------------------------------------------------------------------------
# CanonicalDistributionCoverage
# ---------------------------------------------------------------------------


def test_affirmative_coverage_builds_with_matching_fingerprint() -> None:
    coverage = make_coverage()

    assert coverage.coverage_status is DistributionCoverageStatus.AFFIRMATIVE_COMPLETE
    assert coverage.coverage_scope == DISTRIBUTION_COVERAGE_SCOPE
    assert coverage.coverage_fingerprint == compute_distribution_coverage_fingerprint(
        session=THURSDAY,
        coverage_contract_id=COVERAGE_CONTRACT_ID,
        coverage_contract_version=COVERAGE_CONTRACT_VERSION,
        coverage_status=DistributionCoverageStatus.AFFIRMATIVE_COMPLETE,
        canonical_distribution_snapshot_fingerprint=SNAPSHOT,
        upstream_source_evidence_fingerprint=SOURCE_FINGERPRINT,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    )


def test_coverage_fingerprint_mismatch_fails_closed() -> None:
    coverage = make_coverage()
    values = {
        name: getattr(coverage, name)
        for name in CanonicalDistributionCoverage.model_fields
    }
    values["coverage_fingerprint"] = "0" * 64

    with pytest.raises(ValidationError, match="coverage_fingerprint"):
        CanonicalDistributionCoverage(**values)


def test_coverage_is_frozen() -> None:
    coverage = make_coverage()

    with pytest.raises(ValidationError, match="frozen"):
        coverage.coverage_status = DistributionCoverageStatus.INCOMPLETE


def test_coverage_for_different_sessions_has_different_fingerprints() -> None:
    thu = make_coverage(session=THURSDAY)
    fri = make_coverage(session=FRIDAY)

    assert thu.coverage_fingerprint != fri.coverage_fingerprint


def test_incomplete_coverage_status_is_representable_but_distinct() -> None:
    affirmative = make_coverage()
    fingerprint = compute_distribution_coverage_fingerprint(
        session=THURSDAY,
        coverage_contract_id=COVERAGE_CONTRACT_ID,
        coverage_contract_version=COVERAGE_CONTRACT_VERSION,
        coverage_status=DistributionCoverageStatus.INCOMPLETE,
        canonical_distribution_snapshot_fingerprint=SNAPSHOT,
        upstream_source_evidence_fingerprint=SOURCE_FINGERPRINT,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    )
    values = {
        name: getattr(affirmative, name)
        for name in CanonicalDistributionCoverage.model_fields
    }
    values["coverage_status"] = DistributionCoverageStatus.INCOMPLETE
    values["coverage_fingerprint"] = fingerprint
    incomplete = CanonicalDistributionCoverage(**values)

    assert incomplete.coverage_status is DistributionCoverageStatus.INCOMPLETE
    assert incomplete.coverage_fingerprint != affirmative.coverage_fingerprint


# ---------------------------------------------------------------------------
# Dividend accounting evidence content fingerprint
# ---------------------------------------------------------------------------


def test_evidence_content_fingerprint_is_deterministic() -> None:
    event = make_event()

    first = compute_dividend_accounting_evidence_content_fingerprint(event)
    second = compute_dividend_accounting_evidence_content_fingerprint(event)

    assert first == second


def test_evidence_content_fingerprint_changes_with_event_id() -> None:
    first = compute_dividend_accounting_evidence_content_fingerprint(
        make_event(event_id="D:1")
    )
    second = compute_dividend_accounting_evidence_content_fingerprint(
        make_event(event_id="D:2")
    )

    assert first != second


def test_evidence_content_fingerprint_rejects_non_evidence_type() -> None:
    with pytest.raises(TypeError):
        compute_dividend_accounting_evidence_content_fingerprint(object())


# ---------------------------------------------------------------------------
# DividendAwareSessionEvidence
# ---------------------------------------------------------------------------


def test_session_evidence_builds_for_matching_session_and_snapshot() -> None:
    coverage = make_coverage(session=FRIDAY)
    event = make_event(entitlement_session=THURSDAY, ex_session=FRIDAY)

    evidence = build_dividend_aware_session_evidence(
        session=FRIDAY,
        distribution_coverage=coverage,
        distribution_events=(event,),
    )

    assert evidence.session == FRIDAY
    assert len(evidence.distribution_events) == 1
    assert len(evidence.event_evidence_fingerprints) == 1


def test_session_evidence_rejects_non_affirmative_coverage() -> None:
    fingerprint = compute_distribution_coverage_fingerprint(
        session=FRIDAY,
        coverage_contract_id=COVERAGE_CONTRACT_ID,
        coverage_contract_version=COVERAGE_CONTRACT_VERSION,
        coverage_status=DistributionCoverageStatus.INCOMPLETE,
        canonical_distribution_snapshot_fingerprint=SNAPSHOT,
        upstream_source_evidence_fingerprint=SOURCE_FINGERPRINT,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    )
    incomplete = CanonicalDistributionCoverage(
        session=FRIDAY,
        coverage_contract_id=COVERAGE_CONTRACT_ID,
        coverage_contract_version=COVERAGE_CONTRACT_VERSION,
        coverage_status=DistributionCoverageStatus.INCOMPLETE,
        canonical_distribution_snapshot_fingerprint=SNAPSHOT,
        upstream_source_evidence_fingerprint=SOURCE_FINGERPRINT,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        coverage_fingerprint=fingerprint,
    )

    with pytest.raises(ValidationError, match="affirmative coverage"):
        build_dividend_aware_session_evidence(
            session=FRIDAY,
            distribution_coverage=incomplete,
            distribution_events=(),
        )


def test_session_evidence_rejects_coverage_session_mismatch() -> None:
    coverage = make_coverage(session=THURSDAY)

    with pytest.raises(ValidationError, match="coverage session"):
        build_dividend_aware_session_evidence(
            session=FRIDAY,
            distribution_coverage=coverage,
            distribution_events=(),
        )


def test_session_evidence_rejects_event_ex_session_mismatch() -> None:
    coverage = make_coverage(session=FRIDAY)
    # Valid T < X pair (Thursday's entitlement, an earlier ex-session), but
    # attached to Friday's session evidence: ex_session != evidence.session.
    event = make_event(
        entitlement_session=date(2026, 8, 19), ex_session=THURSDAY
    )

    with pytest.raises(ValidationError, match="ex_session must equal"):
        build_dividend_aware_session_evidence(
            session=FRIDAY,
            distribution_coverage=coverage,
            distribution_events=(event,),
        )


def test_session_evidence_rejects_snapshot_disagreement() -> None:
    coverage = make_coverage(session=FRIDAY, snapshot=SNAPSHOT)
    event = make_event(
        entitlement_session=THURSDAY, ex_session=FRIDAY, snapshot="e" * 64
    )

    with pytest.raises(ValidationError, match="snapshots must agree"):
        build_dividend_aware_session_evidence(
            session=FRIDAY,
            distribution_coverage=coverage,
            distribution_events=(event,),
        )


def test_session_evidence_rejects_duplicate_event_ids() -> None:
    coverage = make_coverage(session=FRIDAY)
    event = make_event(
        event_id="D:1", entitlement_session=THURSDAY, ex_session=FRIDAY
    )

    with pytest.raises(ValidationError, match="unique"):
        build_dividend_aware_session_evidence(
            session=FRIDAY,
            distribution_coverage=coverage,
            distribution_events=(event, event),
        )


def test_session_evidence_is_frozen() -> None:
    evidence = build_dividend_aware_session_evidence(
        session=FRIDAY, distribution_coverage=make_coverage(session=FRIDAY),
        distribution_events=(),
    )

    with pytest.raises(ValidationError, match="frozen"):
        evidence.session = THURSDAY


# ---------------------------------------------------------------------------
# DividendAwareRunEvidence
# ---------------------------------------------------------------------------


def make_proof(*sessions: date):
    links = tuple(
        ProcessedSessionLink(
            session=session,
            next_session=(
                None if index == len(sessions) - 1 else sessions[index + 1]
            ),
        )
        for index, session in enumerate(sessions)
    )
    return build_processed_session_contiguity_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint=CALENDAR_FINGERPRINT,
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        session_links=links,
    )


def test_zero_session_run_evidence_forbids_proof_and_snapshot() -> None:
    with pytest.raises(ValidationError, match="must not carry a proof"):
        build_dividend_aware_run_evidence(
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=make_proof(THURSDAY),
            canonical_distribution_snapshot_fingerprint=None,
            session_evidence=(),
        )


def test_zero_session_run_evidence_builds_cleanly() -> None:
    evidence = build_dividend_aware_run_evidence(
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=None,
        canonical_distribution_snapshot_fingerprint=None,
        session_evidence=(),
    )

    assert evidence.session_evidence == ()
    assert evidence.processed_session_contiguity_proof is None


def test_nonempty_run_evidence_requires_proof_and_snapshot() -> None:
    session_evidence = (
        build_dividend_aware_session_evidence(
            session=FRIDAY,
            distribution_coverage=make_coverage(session=FRIDAY),
            distribution_events=(),
        ),
    )

    with pytest.raises(ValidationError, match="requires proof and snapshot"):
        build_dividend_aware_run_evidence(
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=None,
            canonical_distribution_snapshot_fingerprint=None,
            session_evidence=session_evidence,
        )


def test_run_evidence_requires_proof_sequence_to_match_session_evidence() -> None:
    session_evidence = (
        build_dividend_aware_session_evidence(
            session=FRIDAY,
            distribution_coverage=make_coverage(session=FRIDAY),
            distribution_events=(),
        ),
    )
    proof = make_proof(THURSDAY, FRIDAY)

    with pytest.raises(ValidationError, match="proof sequence must equal"):
        build_dividend_aware_run_evidence(
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
            canonical_distribution_snapshot_fingerprint=SNAPSHOT,
            session_evidence=session_evidence,
        )


def test_run_evidence_requires_one_snapshot_across_sessions() -> None:
    session_evidence = (
        build_dividend_aware_session_evidence(
            session=THURSDAY,
            distribution_coverage=make_coverage(session=THURSDAY, snapshot=SNAPSHOT),
            distribution_events=(),
        ),
        build_dividend_aware_session_evidence(
            session=FRIDAY,
            distribution_coverage=make_coverage(session=FRIDAY, snapshot="e" * 64),
            distribution_events=(),
        ),
    )
    proof = make_proof(THURSDAY, FRIDAY)

    with pytest.raises(ValidationError, match="one snapshot"):
        build_dividend_aware_run_evidence(
            dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
            processed_session_contiguity_proof=proof,
            canonical_distribution_snapshot_fingerprint=SNAPSHOT,
            session_evidence=session_evidence,
        )


def test_run_evidence_fingerprint_is_reproducible() -> None:
    session_evidence = (
        build_dividend_aware_session_evidence(
            session=FRIDAY,
            distribution_coverage=make_coverage(session=FRIDAY),
            distribution_events=(),
        ),
    )
    proof = make_proof(FRIDAY)

    first = build_dividend_aware_run_evidence(
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
        canonical_distribution_snapshot_fingerprint=SNAPSHOT,
        session_evidence=session_evidence,
    )
    second = build_dividend_aware_run_evidence(
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=proof,
        canonical_distribution_snapshot_fingerprint=SNAPSHOT,
        session_evidence=session_evidence,
    )

    assert first.run_input_fingerprint == second.run_input_fingerprint
    assert first == second


def test_run_evidence_is_frozen() -> None:
    evidence = build_dividend_aware_run_evidence(
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=None,
        canonical_distribution_snapshot_fingerprint=None,
        session_evidence=(),
    )

    with pytest.raises(ValidationError, match="frozen"):
        evidence.session_evidence = ()


def test_run_evidence_fingerprint_mismatch_fails_closed() -> None:
    evidence = build_dividend_aware_run_evidence(
        dividend_accounting_policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
        processed_session_contiguity_proof=None,
        canonical_distribution_snapshot_fingerprint=None,
        session_evidence=(),
    )
    values = {
        name: getattr(evidence, name)
        for name in DividendAwareRunEvidence.model_fields
    }
    values["run_input_fingerprint"] = "0" * 64

    with pytest.raises(ValidationError, match="run_input_fingerprint"):
        DividendAwareRunEvidence(**values)


# ---------------------------------------------------------------------------
# Architecture / no provider dependency
# ---------------------------------------------------------------------------


def test_run_evidence_module_has_no_provider_or_calendar_dependency() -> None:
    source = inspect.getsource(run_evidence_module)

    for forbidden in ("socket", "requests", "norgatedata", "urllib"):
        assert forbidden not in source.lower()
