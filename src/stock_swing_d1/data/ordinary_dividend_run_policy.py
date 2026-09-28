"""Run-level ordinary-dividend policy and session-contiguity evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    model_validator,
)

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CANONICAL_NEXT_TRADING_SESSION,
)


ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF_SCHEMA_VERSION: Final[str] = (
    "ordinary_dividend_accounting_policy_ref.v0.1"
)
ORDINARY_DIVIDEND_ACCOUNTING_POLICY_ID: Final[str] = (
    "ordinary_dividend_accounting_v0.1"
)
ORDINARY_DIVIDEND_ACCOUNTING_POLICY_VERSION: Final[str] = "0.1"
ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY: Final[str] = (
    "ORDINARY_DIVIDEND_ACCOUNTING_V0_1"
)
ORDINARY_DIVIDEND_ACCOUNTING_POLICY_HASH_DOMAIN: Final[str] = (
    "ordinary_dividend_accounting_policy.v0.1"
)

PROCESSED_SESSION_CONTIGUITY_PROOF_SCHEMA_VERSION: Final[str] = (
    "processed_session_contiguity_proof.v0.1"
)
PROCESSED_SESSION_CONTIGUITY_HASH_DOMAIN: Final[str] = (
    "ordinary_dividend_processed_session_contiguity.v0.1"
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _domain_sha256(domain: str, payload: object) -> str:
    _require_canonical_text(domain)
    canonical_bytes = json.dumps(
        {"domain": domain, "payload": payload},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def ordinary_dividend_accounting_policy_manifest() -> dict[str, object]:
    """Return a new mapping of the frozen run-level policy semantics."""

    return {
        "schema_version": (
            ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF_SCHEMA_VERSION
        ),
        "policy_id": ORDINARY_DIVIDEND_ACCOUNTING_POLICY_ID,
        "policy_version": ORDINARY_DIVIDEND_ACCOUNTING_POLICY_VERSION,
        "policy_semantic_identity": (
            ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY
        ),
        "activation_scope": "run_level",
        "activation_mode": "explicit_policy_discriminator",
        "event_presence_inference": "forbidden",
        "per_session_selection": "forbidden",
    }


def compute_ordinary_dividend_accounting_policy_fingerprint() -> str:
    """Compute the semantic identity of the sole production dividend policy."""

    return _domain_sha256(
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_HASH_DOMAIN,
        ordinary_dividend_accounting_policy_manifest(),
    )


class OrdinaryDividendAccountingPolicyRef(_ImmutableDividendRunEvidenceModel):
    """Immutable run-level discriminator for ordinary-dividend accounting."""

    schema_version: Literal[
        "ordinary_dividend_accounting_policy_ref.v0.1"
    ] = ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF_SCHEMA_VERSION
    policy_id: Literal[
        "ordinary_dividend_accounting_v0.1"
    ] = ORDINARY_DIVIDEND_ACCOUNTING_POLICY_ID
    policy_version: Literal["0.1"] = (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_VERSION
    )
    policy_semantic_identity: Literal[
        "ORDINARY_DIVIDEND_ACCOUNTING_V0_1"
    ] = ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY
    policy_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_policy_fingerprint(self) -> Self:
        expected = compute_ordinary_dividend_accounting_policy_fingerprint()
        if self.policy_fingerprint != expected:
            raise ValueError("policy_fingerprint does not match policy semantics")
        return self


def build_ordinary_dividend_accounting_policy_ref(
) -> OrdinaryDividendAccountingPolicyRef:
    """Build the sole supported, frozen v0.1 run-policy reference."""

    return OrdinaryDividendAccountingPolicyRef(
        policy_fingerprint=(
            compute_ordinary_dividend_accounting_policy_fingerprint()
        )
    )


ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF: Final[
    OrdinaryDividendAccountingPolicyRef
] = build_ordinary_dividend_accounting_policy_ref()


class ProcessedSessionLink(_ImmutableDividendRunEvidenceModel):
    """One ordered processed session and its explicit in-run successor."""

    session: _SessionDate
    next_session: _SessionDate | None


def _canonical_policy_ref(
    value: object,
) -> OrdinaryDividendAccountingPolicyRef:
    if type(value) is not OrdinaryDividendAccountingPolicyRef:
        raise ValueError(
            "dividend_accounting_policy_ref must be exactly "
            "OrdinaryDividendAccountingPolicyRef"
        )
    return OrdinaryDividendAccountingPolicyRef.model_validate(
        value.model_dump(mode="python")
    )


def _require_session_links(value: object) -> object:
    if type(value) is not tuple:
        raise ValueError("session_links must be an immutable tuple")
    if any(type(link) is not ProcessedSessionLink for link in value):
        raise ValueError(
            "session_links must contain exact ProcessedSessionLink values"
        )
    return tuple(
        ProcessedSessionLink.model_validate(link.model_dump(mode="python"))
        for link in value
    )


_DividendAccountingPolicyRef = Annotated[
    OrdinaryDividendAccountingPolicyRef,
    BeforeValidator(_canonical_policy_ref),
]
_SessionLinks = Annotated[
    tuple[ProcessedSessionLink, ...],
    BeforeValidator(_require_session_links),
]


def _validate_session_links(
    session_links: tuple[ProcessedSessionLink, ...],
) -> None:
    if not session_links:
        raise ValueError("processed-session sequence must be non-empty")

    sessions = tuple(link.session for link in session_links)
    if len(set(sessions)) != len(sessions):
        raise ValueError("processed-session dates must be unique")
    if any(left >= right for left, right in zip(sessions, sessions[1:])):
        raise ValueError("processed-session dates must be strictly increasing")

    for index, link in enumerate(session_links[:-1]):
        following_session = session_links[index + 1].session
        if link.next_session is None:
            raise ValueError(
                "every non-final processed session must have next_session"
            )
        if link.next_session != following_session:
            raise ValueError(
                "non-final next_session must equal the immediately following "
                "supplied processed session"
            )
    if session_links[-1].next_session is not None:
        raise ValueError("the final processed session must have next_session=None")


def _processed_session_contiguity_payload(
    *,
    schema_version: str,
    calendar_source_id: str,
    calendar_policy_id: str,
    calendar_policy_version: str,
    resolution_semantics: str,
    upstream_source_evidence_fingerprint: str,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef,
    session_links: tuple[ProcessedSessionLink, ...],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "calendar_source_id": calendar_source_id,
        "calendar_policy_id": calendar_policy_id,
        "calendar_policy_version": calendar_policy_version,
        "resolution_semantics": resolution_semantics,
        "upstream_source_evidence_fingerprint": (
            upstream_source_evidence_fingerprint
        ),
        "dividend_accounting_policy_ref": {
            "schema_version": dividend_accounting_policy_ref.schema_version,
            "policy_id": dividend_accounting_policy_ref.policy_id,
            "policy_version": dividend_accounting_policy_ref.policy_version,
            "policy_semantic_identity": (
                dividend_accounting_policy_ref.policy_semantic_identity
            ),
            "policy_fingerprint": (
                dividend_accounting_policy_ref.policy_fingerprint
            ),
        },
        "processed_sessions": [
            link.session.isoformat() for link in session_links
        ],
        "session_links": [
            {
                "session": link.session.isoformat(),
                "next_session": (
                    None
                    if link.next_session is None
                    else link.next_session.isoformat()
                ),
            }
            for link in session_links
        ],
    }


def compute_processed_session_contiguity_fingerprint(
    *,
    calendar_source_id: str,
    calendar_policy_id: str,
    calendar_policy_version: str,
    upstream_source_evidence_fingerprint: str,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef,
    session_links: tuple[ProcessedSessionLink, ...],
    resolution_semantics: str = CANONICAL_NEXT_TRADING_SESSION,
    schema_version: str = PROCESSED_SESSION_CONTIGUITY_PROOF_SCHEMA_VERSION,
) -> str:
    """Hash a supplied upstream continuity claim without resolving a calendar."""

    for value in (
        calendar_source_id,
        calendar_policy_id,
        calendar_policy_version,
    ):
        _require_canonical_text(value)
    _require_sha256(upstream_source_evidence_fingerprint)
    if resolution_semantics != CANONICAL_NEXT_TRADING_SESSION:
        raise ValueError(
            "resolution_semantics must be CANONICAL_NEXT_TRADING_SESSION"
        )
    if schema_version != PROCESSED_SESSION_CONTIGUITY_PROOF_SCHEMA_VERSION:
        raise ValueError("unsupported processed-session proof schema version")

    canonical_policy_ref = _canonical_policy_ref(
        dividend_accounting_policy_ref
    )
    canonical_session_links = _require_session_links(session_links)
    assert type(canonical_session_links) is tuple
    _validate_session_links(canonical_session_links)

    return _domain_sha256(
        PROCESSED_SESSION_CONTIGUITY_HASH_DOMAIN,
        _processed_session_contiguity_payload(
            schema_version=schema_version,
            calendar_source_id=calendar_source_id,
            calendar_policy_id=calendar_policy_id,
            calendar_policy_version=calendar_policy_version,
            resolution_semantics=resolution_semantics,
            upstream_source_evidence_fingerprint=(
                upstream_source_evidence_fingerprint
            ),
            dividend_accounting_policy_ref=canonical_policy_ref,
            session_links=canonical_session_links,
        ),
    )


class ProcessedSessionContiguityProof(
    _ImmutableDividendRunEvidenceModel
):
    """Source-bound proof claim for one non-empty ordered processed span."""

    schema_version: Literal[
        "processed_session_contiguity_proof.v0.1"
    ] = PROCESSED_SESSION_CONTIGUITY_PROOF_SCHEMA_VERSION
    calendar_source_id: _CanonicalText
    calendar_policy_id: _CanonicalText
    calendar_policy_version: _CanonicalText
    resolution_semantics: Literal[
        "CANONICAL_NEXT_TRADING_SESSION"
    ] = CANONICAL_NEXT_TRADING_SESSION
    upstream_source_evidence_fingerprint: _Sha256
    dividend_accounting_policy_ref: _DividendAccountingPolicyRef
    session_links: _SessionLinks
    processed_session_contiguity_fingerprint: _Sha256

    @property
    def processed_sessions(self) -> tuple[date, ...]:
        """Return the exact supplied processed-session order."""

        return tuple(link.session for link in self.session_links)

    @model_validator(mode="after")
    def validate_structure_and_fingerprint(self) -> Self:
        _validate_session_links(self.session_links)
        expected = compute_processed_session_contiguity_fingerprint(
            schema_version=self.schema_version,
            calendar_source_id=self.calendar_source_id,
            calendar_policy_id=self.calendar_policy_id,
            calendar_policy_version=self.calendar_policy_version,
            resolution_semantics=self.resolution_semantics,
            upstream_source_evidence_fingerprint=(
                self.upstream_source_evidence_fingerprint
            ),
            dividend_accounting_policy_ref=(
                self.dividend_accounting_policy_ref
            ),
            session_links=self.session_links,
        )
        if self.processed_session_contiguity_fingerprint != expected:
            raise ValueError(
                "processed_session_contiguity_fingerprint does not match "
                "proof content"
            )
        return self


def build_processed_session_contiguity_proof(
    *,
    calendar_source_id: str,
    calendar_policy_id: str,
    calendar_policy_version: str,
    upstream_source_evidence_fingerprint: str,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef,
    session_links: tuple[ProcessedSessionLink, ...],
) -> ProcessedSessionContiguityProof:
    """Build a source-bound claim; never query or reconstruct a calendar."""

    fingerprint = compute_processed_session_contiguity_fingerprint(
        calendar_source_id=calendar_source_id,
        calendar_policy_id=calendar_policy_id,
        calendar_policy_version=calendar_policy_version,
        upstream_source_evidence_fingerprint=(
            upstream_source_evidence_fingerprint
        ),
        dividend_accounting_policy_ref=dividend_accounting_policy_ref,
        session_links=session_links,
    )
    return ProcessedSessionContiguityProof(
        calendar_source_id=calendar_source_id,
        calendar_policy_id=calendar_policy_id,
        calendar_policy_version=calendar_policy_version,
        upstream_source_evidence_fingerprint=(
            upstream_source_evidence_fingerprint
        ),
        dividend_accounting_policy_ref=dividend_accounting_policy_ref,
        session_links=session_links,
        processed_session_contiguity_fingerprint=fingerprint,
    )


__all__ = [
    "ORDINARY_DIVIDEND_ACCOUNTING_POLICY_HASH_DOMAIN",
    "ORDINARY_DIVIDEND_ACCOUNTING_POLICY_ID",
    "ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF",
    "ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF_SCHEMA_VERSION",
    "ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY",
    "ORDINARY_DIVIDEND_ACCOUNTING_POLICY_VERSION",
    "PROCESSED_SESSION_CONTIGUITY_HASH_DOMAIN",
    "PROCESSED_SESSION_CONTIGUITY_PROOF_SCHEMA_VERSION",
    "OrdinaryDividendAccountingPolicyRef",
    "ProcessedSessionContiguityProof",
    "ProcessedSessionLink",
    "build_ordinary_dividend_accounting_policy_ref",
    "build_processed_session_contiguity_proof",
    "compute_ordinary_dividend_accounting_policy_fingerprint",
    "compute_processed_session_contiguity_fingerprint",
    "ordinary_dividend_accounting_policy_manifest",
]
