"""Provider-neutral dividend coverage and Phase 15A run-evidence identity."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    model_validator,
)

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CanonicalDividendAccountingEvidence,
)
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    OrdinaryDividendAccountingPolicyRef,
    ProcessedSessionContiguityProof,
)


DISTRIBUTION_COVERAGE_SCHEMA_VERSION: Final[str] = (
    "canonical_distribution_coverage.v0.1"
)
DISTRIBUTION_COVERAGE_SCOPE: Final[str] = (
    "ALL_CANONICAL_DISTRIBUTIONS_FOR_SESSION"
)
DISTRIBUTION_COVERAGE_HASH_DOMAIN: Final[str] = (
    "ordinary_dividend_distribution_coverage.v0.1"
)

DIVIDEND_AWARE_SESSION_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "dividend_aware_session_evidence.v0.1"
)
DIVIDEND_AWARE_SESSION_EVIDENCE_HASH_DOMAIN: Final[str] = (
    "ordinary_dividend_phase15a_session_evidence.v0.1"
)
DIVIDEND_ACCOUNTING_EVIDENCE_CONTENT_HASH_DOMAIN: Final[str] = (
    "ordinary_dividend_accounting_evidence_content.v0.1"
)

DIVIDEND_AWARE_RUN_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "dividend_aware_run_evidence.v0.1"
)
DIVIDEND_AWARE_RUN_INPUT_HASH_DOMAIN: Final[str] = (
    "ordinary_dividend_phase15a_run_input.v0.1"
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class DistributionCoverageStatus(StrEnum):
    """Whether upstream evidence completely covers the declared scope."""

    AFFIRMATIVE_COMPLETE = "AFFIRMATIVE_COMPLETE"
    INCOMPLETE = "INCOMPLETE"


def _require_session_date(value: object) -> object:
    if type(value) is not date:
        raise ValueError("must be a Python datetime.date")
    return value


def _require_canonical_text(value: object) -> object:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("must be canonical non-empty text")
    return value


def _require_sha256(value: object) -> object:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("must be a lowercase hexadecimal SHA-256 fingerprint")
    return value


def _require_coverage_status(value: object) -> object:
    if type(value) is not DistributionCoverageStatus:
        raise ValueError("must be an exact DistributionCoverageStatus value")
    return value


def _require_tuple(value: object) -> object:
    if type(value) is not tuple:
        raise ValueError("must be an immutable tuple")
    return value


class _ImmutableDividendRunEvidenceModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


_SessionDate = Annotated[date, BeforeValidator(_require_session_date)]
_CanonicalText = Annotated[str, BeforeValidator(_require_canonical_text)]
_Sha256 = Annotated[str, BeforeValidator(_require_sha256)]
_CoverageStatus = Annotated[
    DistributionCoverageStatus,
    BeforeValidator(_require_coverage_status),
]


def _domain_sha256(domain: str, payload: object) -> str:
    _require_canonical_text(domain)
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_policy_ref(
    value: object,
) -> OrdinaryDividendAccountingPolicyRef:
    if type(value) is not OrdinaryDividendAccountingPolicyRef:
        raise ValueError(
            "must be exactly OrdinaryDividendAccountingPolicyRef"
        )
    return OrdinaryDividendAccountingPolicyRef.model_validate(
        value.model_dump(mode="python")
    )


_DividendAccountingPolicyRef = Annotated[
    OrdinaryDividendAccountingPolicyRef,
    BeforeValidator(_canonical_policy_ref),
]


def _policy_ref_payload(
    policy_ref: OrdinaryDividendAccountingPolicyRef,
) -> dict[str, object]:
    return {
        "schema_version": policy_ref.schema_version,
        "policy_id": policy_ref.policy_id,
        "policy_version": policy_ref.policy_version,
        "policy_semantic_identity": policy_ref.policy_semantic_identity,
        "policy_fingerprint": policy_ref.policy_fingerprint,
    }


def _distribution_coverage_payload(
    *,
    schema_version: str,
    session: date,
    coverage_contract_id: str,
    coverage_contract_version: str,
    coverage_status: DistributionCoverageStatus,
    coverage_scope: str,
    canonical_distribution_snapshot_fingerprint: str,
    upstream_source_evidence_fingerprint: str,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "session": session.isoformat(),
        "coverage_contract_id": coverage_contract_id,
        "coverage_contract_version": coverage_contract_version,
        "coverage_status": coverage_status.value,
        "coverage_scope": coverage_scope,
        "canonical_distribution_snapshot_fingerprint": (
            canonical_distribution_snapshot_fingerprint
        ),
        "upstream_source_evidence_fingerprint": (
            upstream_source_evidence_fingerprint
        ),
        "dividend_accounting_policy_ref": _policy_ref_payload(
            dividend_accounting_policy_ref
        ),
    }


def compute_distribution_coverage_fingerprint(
    *,
    session: date,
    coverage_contract_id: str,
    coverage_contract_version: str,
    coverage_status: DistributionCoverageStatus,
    canonical_distribution_snapshot_fingerprint: str,
    upstream_source_evidence_fingerprint: str,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef,
    coverage_scope: str = DISTRIBUTION_COVERAGE_SCOPE,
    schema_version: str = DISTRIBUTION_COVERAGE_SCHEMA_VERSION,
) -> str:
    """Hash one upstream distribution-coverage claim."""

    _require_session_date(session)
    for value in (coverage_contract_id, coverage_contract_version):
        _require_canonical_text(value)
    _require_coverage_status(coverage_status)
    _require_sha256(canonical_distribution_snapshot_fingerprint)
    _require_sha256(upstream_source_evidence_fingerprint)
    policy_ref = _canonical_policy_ref(dividend_accounting_policy_ref)
    if coverage_scope != DISTRIBUTION_COVERAGE_SCOPE:
        raise ValueError("coverage_scope must cover all canonical distributions")
    if schema_version != DISTRIBUTION_COVERAGE_SCHEMA_VERSION:
        raise ValueError("unsupported distribution coverage schema version")
    return _domain_sha256(
        DISTRIBUTION_COVERAGE_HASH_DOMAIN,
        _distribution_coverage_payload(
            schema_version=schema_version,
            session=session,
            coverage_contract_id=coverage_contract_id,
            coverage_contract_version=coverage_contract_version,
            coverage_status=coverage_status,
            coverage_scope=coverage_scope,
            canonical_distribution_snapshot_fingerprint=(
                canonical_distribution_snapshot_fingerprint
            ),
            upstream_source_evidence_fingerprint=(
                upstream_source_evidence_fingerprint
            ),
            dividend_accounting_policy_ref=policy_ref,
        ),
    )


class CanonicalDistributionCoverage(_ImmutableDividendRunEvidenceModel):
    """Source-bound coverage assertion for one processed session."""

    schema_version: Literal[
        "canonical_distribution_coverage.v0.1"
    ] = DISTRIBUTION_COVERAGE_SCHEMA_VERSION
    session: _SessionDate
    coverage_contract_id: _CanonicalText
    coverage_contract_version: _CanonicalText
    coverage_status: _CoverageStatus
    coverage_scope: Literal[
        "ALL_CANONICAL_DISTRIBUTIONS_FOR_SESSION"
    ] = DISTRIBUTION_COVERAGE_SCOPE
    canonical_distribution_snapshot_fingerprint: _Sha256
    upstream_source_evidence_fingerprint: _Sha256
    dividend_accounting_policy_ref: _DividendAccountingPolicyRef
    coverage_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_coverage_fingerprint(self) -> Self:
        expected = compute_distribution_coverage_fingerprint(
            schema_version=self.schema_version,
            session=self.session,
            coverage_contract_id=self.coverage_contract_id,
            coverage_contract_version=self.coverage_contract_version,
            coverage_status=self.coverage_status,
            coverage_scope=self.coverage_scope,
            canonical_distribution_snapshot_fingerprint=(
                self.canonical_distribution_snapshot_fingerprint
            ),
            upstream_source_evidence_fingerprint=(
                self.upstream_source_evidence_fingerprint
            ),
            dividend_accounting_policy_ref=(
                self.dividend_accounting_policy_ref
            ),
        )
        if self.coverage_fingerprint != expected:
            raise ValueError(
                "coverage_fingerprint does not match coverage content"
            )
        return self


def build_affirmative_distribution_coverage(
    *,
    session: date,
    coverage_contract_id: str,
    coverage_contract_version: str,
    canonical_distribution_snapshot_fingerprint: str,
    upstream_source_evidence_fingerprint: str,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef,
) -> CanonicalDistributionCoverage:
    """Build affirmative complete coverage without consulting a provider."""

    status = DistributionCoverageStatus.AFFIRMATIVE_COMPLETE
    fingerprint = compute_distribution_coverage_fingerprint(
        session=session,
        coverage_contract_id=coverage_contract_id,
        coverage_contract_version=coverage_contract_version,
        coverage_status=status,
        canonical_distribution_snapshot_fingerprint=(
            canonical_distribution_snapshot_fingerprint
        ),
        upstream_source_evidence_fingerprint=(
            upstream_source_evidence_fingerprint
        ),
        dividend_accounting_policy_ref=dividend_accounting_policy_ref,
    )
    return CanonicalDistributionCoverage(
        session=session,
        coverage_contract_id=coverage_contract_id,
        coverage_contract_version=coverage_contract_version,
        coverage_status=status,
        canonical_distribution_snapshot_fingerprint=(
            canonical_distribution_snapshot_fingerprint
        ),
        upstream_source_evidence_fingerprint=(
            upstream_source_evidence_fingerprint
        ),
        dividend_accounting_policy_ref=dividend_accounting_policy_ref,
        coverage_fingerprint=fingerprint,
    )


def compute_dividend_accounting_evidence_content_fingerprint(
    evidence: CanonicalDividendAccountingEvidence,
) -> str:
    """Bind canonical event semantics without recomputing normalization or X."""

    if type(evidence) is not CanonicalDividendAccountingEvidence:
        raise TypeError("evidence must be CanonicalDividendAccountingEvidence")
    rebuilt = CanonicalDividendAccountingEvidence.model_validate(
        evidence.model_dump(mode="python")
    )
    return _domain_sha256(
        DIVIDEND_ACCOUNTING_EVIDENCE_CONTENT_HASH_DOMAIN,
        {
            "evidence_schema_version": rebuilt.evidence_schema_version,
            "canonical_distribution_event_id": (
                rebuilt.canonical_distribution_event_id
            ),
            "canonical_security_id": rebuilt.canonical_security_id,
            "entitlement_session": rebuilt.entitlement_session.isoformat(),
            "ex_session": rebuilt.ex_session.isoformat(),
            "distribution_type": rebuilt.distribution_type,
            "amount_per_share": format(rebuilt.amount_per_share, "f"),
            "amount_basis": rebuilt.amount_basis,
            "normalization_method_id": rebuilt.normalization_method_id,
            "normalization_scale": rebuilt.normalization_scale,
            "normalization_rounding_mode": (
                rebuilt.normalization_rounding_mode
            ),
            "normalization_arithmetic_mode": (
                rebuilt.normalization_arithmetic_mode
            ),
            "normalization_inputs_fingerprint": (
                rebuilt.normalization_inputs_fingerprint
            ),
            "classification_proof_fingerprint": (
                rebuilt.classification_proof.classification_proof_fingerprint
            ),
            "calendar_resolution_fingerprint": (
                rebuilt.calendar_resolution_fingerprint
            ),
            "currency": rebuilt.currency,
            "canonical_distribution_snapshot_fingerprint": (
                rebuilt.canonical_distribution_snapshot_fingerprint
            ),
        },
    )


def _canonical_distribution_coverage(
    value: object,
) -> CanonicalDistributionCoverage:
    if type(value) is not CanonicalDistributionCoverage:
        raise ValueError("must be exactly CanonicalDistributionCoverage")
    values = {
        name: getattr(value, name)
        for name in CanonicalDistributionCoverage.model_fields
    }
    return CanonicalDistributionCoverage.model_validate(values)


def _require_distribution_events(value: object) -> object:
    _require_tuple(value)
    assert type(value) is tuple
    if any(
        type(event) is not CanonicalDividendAccountingEvidence
        for event in value
    ):
        raise ValueError(
            "must contain exact CanonicalDividendAccountingEvidence values"
        )
    return tuple(
        CanonicalDividendAccountingEvidence.model_validate(
            event.model_dump(mode="python")
        )
        for event in value
    )


_DistributionCoverage = Annotated[
    CanonicalDistributionCoverage,
    BeforeValidator(_canonical_distribution_coverage),
]
_DistributionEvents = Annotated[
    tuple[CanonicalDividendAccountingEvidence, ...],
    BeforeValidator(_require_distribution_events),
]
_FingerprintTuple = Annotated[
    tuple[_Sha256, ...],
    BeforeValidator(_require_tuple),
]


def _session_evidence_payload(
    *,
    schema_version: str,
    session: date,
    distribution_coverage_fingerprint: str,
    distribution_events: tuple[CanonicalDividendAccountingEvidence, ...],
    event_evidence_fingerprints: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "session": session.isoformat(),
        "distribution_coverage_fingerprint": (
            distribution_coverage_fingerprint
        ),
        "distribution_events": [
            {
                "canonical_distribution_event_id": (
                    event.canonical_distribution_event_id
                ),
                "canonical_security_id": event.canonical_security_id,
                "evidence_content_fingerprint": fingerprint,
            }
            for event, fingerprint in zip(
                distribution_events,
                event_evidence_fingerprints,
                strict=True,
            )
        ],
    }


def compute_dividend_aware_session_evidence_fingerprint(
    *,
    session: date,
    distribution_coverage: CanonicalDistributionCoverage,
    distribution_events: tuple[CanonicalDividendAccountingEvidence, ...],
    schema_version: str = DIVIDEND_AWARE_SESSION_EVIDENCE_SCHEMA_VERSION,
) -> str:
    """Bind one validated session's coverage and ordered event evidence."""

    _require_session_date(session)
    coverage = _canonical_distribution_coverage(distribution_coverage)
    events = _require_distribution_events(distribution_events)
    assert type(events) is tuple
    if schema_version != DIVIDEND_AWARE_SESSION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported dividend-aware session evidence schema")
    fingerprints = tuple(
        compute_dividend_accounting_evidence_content_fingerprint(event)
        for event in events
    )
    return _domain_sha256(
        DIVIDEND_AWARE_SESSION_EVIDENCE_HASH_DOMAIN,
        _session_evidence_payload(
            schema_version=schema_version,
            session=session,
            distribution_coverage_fingerprint=coverage.coverage_fingerprint,
            distribution_events=events,
            event_evidence_fingerprints=fingerprints,
        ),
    )


class DividendAwareSessionEvidence(_ImmutableDividendRunEvidenceModel):
    """Retained Phase 15A evidence identity for one processed session."""

    schema_version: Literal[
        "dividend_aware_session_evidence.v0.1"
    ] = DIVIDEND_AWARE_SESSION_EVIDENCE_SCHEMA_VERSION
    session: _SessionDate
    distribution_coverage: _DistributionCoverage
    distribution_events: _DistributionEvents
    event_evidence_fingerprints: _FingerprintTuple
    session_evidence_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_session_evidence(self) -> Self:
        coverage = self.distribution_coverage
        if (
            coverage.coverage_status
            is not DistributionCoverageStatus.AFFIRMATIVE_COMPLETE
        ):
            raise ValueError("session evidence requires affirmative coverage")
        if coverage.session != self.session:
            raise ValueError("coverage session must equal evidence session")
        snapshot = coverage.canonical_distribution_snapshot_fingerprint
        event_ids: list[str] = []
        for event in self.distribution_events:
            if event.ex_session != self.session:
                raise ValueError("event ex_session must equal evidence session")
            if event.canonical_distribution_snapshot_fingerprint != snapshot:
                raise ValueError("event and coverage snapshots must agree")
            event_ids.append(event.canonical_distribution_event_id)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("session distribution event IDs must be unique")
        expected_event_fingerprints = tuple(
            compute_dividend_accounting_evidence_content_fingerprint(event)
            for event in self.distribution_events
        )
        if self.event_evidence_fingerprints != expected_event_fingerprints:
            raise ValueError("event evidence fingerprints do not match events")
        expected = compute_dividend_aware_session_evidence_fingerprint(
            schema_version=self.schema_version,
            session=self.session,
            distribution_coverage=coverage,
            distribution_events=self.distribution_events,
        )
        if self.session_evidence_fingerprint != expected:
            raise ValueError(
                "session_evidence_fingerprint does not match evidence content"
            )
        return self


def build_dividend_aware_session_evidence(
    *,
    session: date,
    distribution_coverage: CanonicalDistributionCoverage,
    distribution_events: tuple[CanonicalDividendAccountingEvidence, ...],
) -> DividendAwareSessionEvidence:
    """Retain one already-validated Phase 15A session evidence bundle."""

    events = _require_distribution_events(distribution_events)
    assert type(events) is tuple
    event_fingerprints = tuple(
        compute_dividend_accounting_evidence_content_fingerprint(event)
        for event in events
    )
    fingerprint = compute_dividend_aware_session_evidence_fingerprint(
        session=session,
        distribution_coverage=distribution_coverage,
        distribution_events=events,
    )
    return DividendAwareSessionEvidence(
        session=session,
        distribution_coverage=distribution_coverage,
        distribution_events=events,
        event_evidence_fingerprints=event_fingerprints,
        session_evidence_fingerprint=fingerprint,
    )


def _canonical_contiguity_proof(
    value: object,
) -> ProcessedSessionContiguityProof:
    if type(value) is not ProcessedSessionContiguityProof:
        raise ValueError("must be exactly ProcessedSessionContiguityProof")
    values = {
        name: getattr(value, name)
        for name in ProcessedSessionContiguityProof.model_fields
    }
    return ProcessedSessionContiguityProof.model_validate(values)


def _require_optional_contiguity_proof(value: object) -> object:
    if value is None:
        return None
    return _canonical_contiguity_proof(value)


def _require_session_evidence_tuple(value: object) -> object:
    _require_tuple(value)
    assert type(value) is tuple
    if any(type(item) is not DividendAwareSessionEvidence for item in value):
        raise ValueError(
            "must contain exact DividendAwareSessionEvidence values"
        )
    return tuple(
        DividendAwareSessionEvidence.model_validate(
            {
                name: getattr(item, name)
                for name in DividendAwareSessionEvidence.model_fields
            }
        )
        for item in value
    )


_OptionalContiguityProof = Annotated[
    ProcessedSessionContiguityProof | None,
    BeforeValidator(_require_optional_contiguity_proof),
]
_SessionEvidenceTuple = Annotated[
    tuple[DividendAwareSessionEvidence, ...],
    BeforeValidator(_require_session_evidence_tuple),
]


def _run_evidence_payload(
    *,
    schema_version: str,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef,
    processed_session_contiguity_proof: (
        ProcessedSessionContiguityProof | None
    ),
    canonical_distribution_snapshot_fingerprint: str | None,
    session_evidence: tuple[DividendAwareSessionEvidence, ...],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "dividend_aware_mode": (
            dividend_accounting_policy_ref.policy_semantic_identity
        ),
        "dividend_accounting_policy_ref": _policy_ref_payload(
            dividend_accounting_policy_ref
        ),
        "processed_session_contiguity_fingerprint": (
            None
            if processed_session_contiguity_proof is None
            else processed_session_contiguity_proof.processed_session_contiguity_fingerprint
        ),
        "canonical_distribution_snapshot_fingerprint": (
            canonical_distribution_snapshot_fingerprint
        ),
        "session_evidence": [
            {
                "session": item.session.isoformat(),
                "coverage_fingerprint": (
                    item.distribution_coverage.coverage_fingerprint
                ),
                "event_evidence_fingerprints": list(
                    item.event_evidence_fingerprints
                ),
                "session_evidence_fingerprint": (
                    item.session_evidence_fingerprint
                ),
            }
            for item in session_evidence
        ],
    }


def compute_dividend_aware_run_input_fingerprint(
    *,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef,
    processed_session_contiguity_proof: (
        ProcessedSessionContiguityProof | None
    ),
    canonical_distribution_snapshot_fingerprint: str | None,
    session_evidence: tuple[DividendAwareSessionEvidence, ...],
    schema_version: str = DIVIDEND_AWARE_RUN_EVIDENCE_SCHEMA_VERSION,
) -> str:
    """Bind all Phase 15A dividend-aware inputs in chronological order."""

    policy_ref = _canonical_policy_ref(dividend_accounting_policy_ref)
    proof = _require_optional_contiguity_proof(
        processed_session_contiguity_proof
    )
    items = _require_session_evidence_tuple(session_evidence)
    assert type(items) is tuple
    if canonical_distribution_snapshot_fingerprint is not None:
        _require_sha256(canonical_distribution_snapshot_fingerprint)
    if schema_version != DIVIDEND_AWARE_RUN_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported dividend-aware run evidence schema")
    return _domain_sha256(
        DIVIDEND_AWARE_RUN_INPUT_HASH_DOMAIN,
        _run_evidence_payload(
            schema_version=schema_version,
            dividend_accounting_policy_ref=policy_ref,
            processed_session_contiguity_proof=proof,
            canonical_distribution_snapshot_fingerprint=(
                canonical_distribution_snapshot_fingerprint
            ),
            session_evidence=items,
        ),
    )


class DividendAwareRunEvidence(_ImmutableDividendRunEvidenceModel):
    """Immutable Phase 15A evidence and identity for one explicit run mode."""

    schema_version: Literal[
        "dividend_aware_run_evidence.v0.1"
    ] = DIVIDEND_AWARE_RUN_EVIDENCE_SCHEMA_VERSION
    dividend_accounting_policy_ref: _DividendAccountingPolicyRef
    processed_session_contiguity_proof: _OptionalContiguityProof = None
    canonical_distribution_snapshot_fingerprint: _Sha256 | None = None
    session_evidence: _SessionEvidenceTuple = ()
    run_input_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_run_evidence(self) -> Self:
        if not self.session_evidence:
            if self.processed_session_contiguity_proof is not None:
                raise ValueError("zero-session evidence must not carry a proof")
            if self.canonical_distribution_snapshot_fingerprint is not None:
                raise ValueError("zero-session evidence must not invent a snapshot")
        else:
            proof = self.processed_session_contiguity_proof
            snapshot = self.canonical_distribution_snapshot_fingerprint
            if proof is None or snapshot is None:
                raise ValueError(
                    "non-empty dividend-aware evidence requires proof and snapshot"
                )
            if proof.dividend_accounting_policy_ref != (
                self.dividend_accounting_policy_ref
            ):
                raise ValueError("proof policy must equal run policy")
            sessions = tuple(item.session for item in self.session_evidence)
            if sessions != proof.processed_sessions:
                raise ValueError("proof sequence must equal session evidence sequence")
            if any(
                item.distribution_coverage.dividend_accounting_policy_ref
                != self.dividend_accounting_policy_ref
                for item in self.session_evidence
            ):
                raise ValueError("coverage policy must equal run policy")
            if any(
                item.distribution_coverage.canonical_distribution_snapshot_fingerprint
                != snapshot
                for item in self.session_evidence
            ):
                raise ValueError("all session evidence must use one snapshot")
        expected = compute_dividend_aware_run_input_fingerprint(
            schema_version=self.schema_version,
            dividend_accounting_policy_ref=(
                self.dividend_accounting_policy_ref
            ),
            processed_session_contiguity_proof=(
                self.processed_session_contiguity_proof
            ),
            canonical_distribution_snapshot_fingerprint=(
                self.canonical_distribution_snapshot_fingerprint
            ),
            session_evidence=self.session_evidence,
        )
        if self.run_input_fingerprint != expected:
            raise ValueError("run_input_fingerprint does not match run evidence")
        return self


def build_dividend_aware_run_evidence(
    *,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef,
    processed_session_contiguity_proof: (
        ProcessedSessionContiguityProof | None
    ),
    canonical_distribution_snapshot_fingerprint: str | None,
    session_evidence: tuple[DividendAwareSessionEvidence, ...],
) -> DividendAwareRunEvidence:
    """Build one self-validating Phase 15A run-evidence identity."""

    fingerprint = compute_dividend_aware_run_input_fingerprint(
        dividend_accounting_policy_ref=dividend_accounting_policy_ref,
        processed_session_contiguity_proof=(
            processed_session_contiguity_proof
        ),
        canonical_distribution_snapshot_fingerprint=(
            canonical_distribution_snapshot_fingerprint
        ),
        session_evidence=session_evidence,
    )
    return DividendAwareRunEvidence(
        dividend_accounting_policy_ref=dividend_accounting_policy_ref,
        processed_session_contiguity_proof=(
            processed_session_contiguity_proof
        ),
        canonical_distribution_snapshot_fingerprint=(
            canonical_distribution_snapshot_fingerprint
        ),
        session_evidence=session_evidence,
        run_input_fingerprint=fingerprint,
    )


__all__ = [
    "DISTRIBUTION_COVERAGE_HASH_DOMAIN",
    "DISTRIBUTION_COVERAGE_SCHEMA_VERSION",
    "DISTRIBUTION_COVERAGE_SCOPE",
    "DIVIDEND_ACCOUNTING_EVIDENCE_CONTENT_HASH_DOMAIN",
    "DIVIDEND_AWARE_RUN_EVIDENCE_SCHEMA_VERSION",
    "DIVIDEND_AWARE_RUN_INPUT_HASH_DOMAIN",
    "DIVIDEND_AWARE_SESSION_EVIDENCE_HASH_DOMAIN",
    "DIVIDEND_AWARE_SESSION_EVIDENCE_SCHEMA_VERSION",
    "CanonicalDistributionCoverage",
    "DistributionCoverageStatus",
    "DividendAwareRunEvidence",
    "DividendAwareSessionEvidence",
    "build_affirmative_distribution_coverage",
    "build_dividend_aware_run_evidence",
    "build_dividend_aware_session_evidence",
    "compute_distribution_coverage_fingerprint",
    "compute_dividend_accounting_evidence_content_fingerprint",
    "compute_dividend_aware_run_input_fingerprint",
    "compute_dividend_aware_session_evidence_fingerprint",
]
