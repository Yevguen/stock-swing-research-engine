"""Provider-neutral ordinary-dividend accounting-evidence primitives."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from decimal import Decimal
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from stock_swing_d1.data.ordinary_dividend_normalization import (
    HISTORICAL_SHARE_AMOUNT_BASIS,
    NORMALIZATION_ARITHMETIC_MODE,
    NORMALIZATION_METHOD_ID,
    NORMALIZATION_ROUNDING_MODE,
    NORMALIZATION_SCALE,
    Gate3DividendNormalizationInputs,
    Gate3DividendNormalizationResult,
    compute_normalization_inputs_fingerprint,
)


ORDINARY_CASH_DISTRIBUTION_TYPE: Final[str] = "ordinary_cash"
CANONICAL_NEXT_TRADING_SESSION: Final[str] = (
    "CANONICAL_NEXT_TRADING_SESSION"
)
CANONICAL_DIVIDEND_CURRENCY: Final[str] = "USD"

ORDINARY_CASH_CLASSIFICATION_PROOF_SCHEMA_VERSION: Final[str] = (
    "ordinary_cash_classification_proof.v0.1"
)
DIVIDEND_CALENDAR_RESOLUTION_PROOF_SCHEMA_VERSION: Final[str] = (
    "dividend_calendar_resolution_proof.v0.1"
)
CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "canonical_dividend_accounting_evidence.v0.1"
)

CLASSIFICATION_PROOF_HASH_DOMAIN: Final[str] = (
    "ordinary_dividend_classification_proof.v0.1"
)
CALENDAR_RESOLUTION_PROOF_HASH_DOMAIN: Final[str] = (
    "ordinary_dividend_calendar_resolution_proof.v0.1"
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


def _require_positive_scale_38_decimal(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError(
            "must be an exact Decimal; binary floats, strings, ints, and bool "
            "are forbidden"
        )
    if not value.is_finite() or value <= 0:
        raise ValueError("must be a positive finite Decimal")
    if value.as_tuple().exponent != -NORMALIZATION_SCALE:
        raise ValueError("must be an exact fixed-scale-38 Decimal")
    return value


class _ImmutableDividendEvidenceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


_SessionDate = Annotated[date, BeforeValidator(_require_session_date)]
_CanonicalText = Annotated[str, BeforeValidator(_require_canonical_text)]
_CanonicalSecurityId = Annotated[
    str,
    Field(pattern=r"^NORGATE:[1-9][0-9]*$"),
]
_Sha256 = Annotated[str, BeforeValidator(_require_sha256)]
_PositiveScale38Decimal = Annotated[
    Decimal,
    BeforeValidator(_require_positive_scale_38_decimal),
]


def _domain_sha256(domain: str, payload: dict[str, object]) -> str:
    canonical_bytes = json.dumps(
        {"domain": domain, "payload": payload},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def _classification_proof_payload(
    *,
    schema_version: str,
    canonical_distribution_event_id: str,
    classification_contract_id: str,
    classification_result: str,
    upstream_source_evidence_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "canonical_distribution_event_id": canonical_distribution_event_id,
        "classification_contract_id": classification_contract_id,
        "classification_result": classification_result,
        "upstream_source_evidence_fingerprint": (
            upstream_source_evidence_fingerprint
        ),
    }


def compute_classification_proof_fingerprint(
    *,
    canonical_distribution_event_id: str,
    classification_contract_id: str,
    upstream_source_evidence_fingerprint: str,
    classification_result: str = ORDINARY_CASH_DISTRIBUTION_TYPE,
    schema_version: str = ORDINARY_CASH_CLASSIFICATION_PROOF_SCHEMA_VERSION,
) -> str:
    """Hash every semantic field in one ordinary-cash classification proof."""

    for value in (
        canonical_distribution_event_id,
        classification_contract_id,
    ):
        _require_canonical_text(value)
    _require_sha256(upstream_source_evidence_fingerprint)
    if classification_result != ORDINARY_CASH_DISTRIBUTION_TYPE:
        raise ValueError("classification_result must be ordinary_cash")
    if schema_version != ORDINARY_CASH_CLASSIFICATION_PROOF_SCHEMA_VERSION:
        raise ValueError("unsupported classification-proof schema version")
    return _domain_sha256(
        CLASSIFICATION_PROOF_HASH_DOMAIN,
        _classification_proof_payload(
            schema_version=schema_version,
            canonical_distribution_event_id=canonical_distribution_event_id,
            classification_contract_id=classification_contract_id,
            classification_result=classification_result,
            upstream_source_evidence_fingerprint=(
                upstream_source_evidence_fingerprint
            ),
        ),
    )


class OrdinaryCashClassificationProof(_ImmutableDividendEvidenceModel):
    """Affirmative, source-bound proof that one distribution is ordinary cash."""

    schema_version: Literal["ordinary_cash_classification_proof.v0.1"] = (
        ORDINARY_CASH_CLASSIFICATION_PROOF_SCHEMA_VERSION
    )
    canonical_distribution_event_id: _CanonicalText
    classification_contract_id: _CanonicalText
    classification_result: Literal["ordinary_cash"] = (
        ORDINARY_CASH_DISTRIBUTION_TYPE
    )
    upstream_source_evidence_fingerprint: _Sha256
    classification_proof_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_fingerprint(self) -> Self:
        expected = compute_classification_proof_fingerprint(
            schema_version=self.schema_version,
            canonical_distribution_event_id=(
                self.canonical_distribution_event_id
            ),
            classification_contract_id=self.classification_contract_id,
            classification_result=self.classification_result,
            upstream_source_evidence_fingerprint=(
                self.upstream_source_evidence_fingerprint
            ),
        )
        if self.classification_proof_fingerprint != expected:
            raise ValueError(
                "classification_proof_fingerprint does not match proof content"
            )
        return self


def build_ordinary_cash_classification_proof(
    *,
    canonical_distribution_event_id: str,
    classification_contract_id: str,
    upstream_source_evidence_fingerprint: str,
) -> OrdinaryCashClassificationProof:
    """Construct one internally consistent ordinary-cash proof."""

    fingerprint = compute_classification_proof_fingerprint(
        canonical_distribution_event_id=canonical_distribution_event_id,
        classification_contract_id=classification_contract_id,
        upstream_source_evidence_fingerprint=(
            upstream_source_evidence_fingerprint
        ),
    )
    return OrdinaryCashClassificationProof(
        canonical_distribution_event_id=canonical_distribution_event_id,
        classification_contract_id=classification_contract_id,
        upstream_source_evidence_fingerprint=(
            upstream_source_evidence_fingerprint
        ),
        classification_proof_fingerprint=fingerprint,
    )


def _calendar_resolution_payload(
    *,
    schema_version: str,
    calendar_source_id: str,
    entitlement_session: date,
    ex_session: date,
    calendar_policy_id: str,
    calendar_policy_version: str,
    resolution_semantics: str,
    upstream_source_evidence_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "calendar_source_id": calendar_source_id,
        "entitlement_session": entitlement_session.isoformat(),
        "ex_session": ex_session.isoformat(),
        "calendar_policy_id": calendar_policy_id,
        "calendar_policy_version": calendar_policy_version,
        "resolution_semantics": resolution_semantics,
        "upstream_source_evidence_fingerprint": (
            upstream_source_evidence_fingerprint
        ),
    }


def compute_calendar_resolution_fingerprint(
    *,
    calendar_source_id: str,
    entitlement_session: date,
    ex_session: date,
    calendar_policy_id: str,
    calendar_policy_version: str,
    upstream_source_evidence_fingerprint: str,
    resolution_semantics: str = CANONICAL_NEXT_TRADING_SESSION,
    schema_version: str = DIVIDEND_CALENDAR_RESOLUTION_PROOF_SCHEMA_VERSION,
) -> str:
    """Hash one upstream-resolved T-to-X claim without resolving a calendar."""

    for value in (
        calendar_source_id,
        calendar_policy_id,
        calendar_policy_version,
    ):
        _require_canonical_text(value)
    _require_session_date(entitlement_session)
    _require_session_date(ex_session)
    if entitlement_session >= ex_session:
        raise ValueError("entitlement_session must be before ex_session")
    _require_sha256(upstream_source_evidence_fingerprint)
    if resolution_semantics != CANONICAL_NEXT_TRADING_SESSION:
        raise ValueError(
            "resolution_semantics must be CANONICAL_NEXT_TRADING_SESSION"
        )
    if schema_version != DIVIDEND_CALENDAR_RESOLUTION_PROOF_SCHEMA_VERSION:
        raise ValueError("unsupported calendar-resolution-proof schema version")
    return _domain_sha256(
        CALENDAR_RESOLUTION_PROOF_HASH_DOMAIN,
        _calendar_resolution_payload(
            schema_version=schema_version,
            calendar_source_id=calendar_source_id,
            entitlement_session=entitlement_session,
            ex_session=ex_session,
            calendar_policy_id=calendar_policy_id,
            calendar_policy_version=calendar_policy_version,
            resolution_semantics=resolution_semantics,
            upstream_source_evidence_fingerprint=(
                upstream_source_evidence_fingerprint
            ),
        ),
    )


class DividendCalendarResolutionProof(_ImmutableDividendEvidenceModel):
    """Source-bound proof of an upstream-resolved canonical next session X."""

    schema_version: Literal["dividend_calendar_resolution_proof.v0.1"] = (
        DIVIDEND_CALENDAR_RESOLUTION_PROOF_SCHEMA_VERSION
    )
    calendar_source_id: _CanonicalText
    entitlement_session: _SessionDate
    ex_session: _SessionDate
    calendar_policy_id: _CanonicalText
    calendar_policy_version: _CanonicalText
    resolution_semantics: Literal["CANONICAL_NEXT_TRADING_SESSION"] = (
        CANONICAL_NEXT_TRADING_SESSION
    )
    upstream_source_evidence_fingerprint: _Sha256
    calendar_resolution_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_chronology_and_fingerprint(self) -> Self:
        expected = compute_calendar_resolution_fingerprint(
            schema_version=self.schema_version,
            calendar_source_id=self.calendar_source_id,
            entitlement_session=self.entitlement_session,
            ex_session=self.ex_session,
            calendar_policy_id=self.calendar_policy_id,
            calendar_policy_version=self.calendar_policy_version,
            resolution_semantics=self.resolution_semantics,
            upstream_source_evidence_fingerprint=(
                self.upstream_source_evidence_fingerprint
            ),
        )
        if self.calendar_resolution_fingerprint != expected:
            raise ValueError(
                "calendar_resolution_fingerprint does not match proof content"
            )
        return self


def build_dividend_calendar_resolution_proof(
    *,
    calendar_source_id: str,
    entitlement_session: date,
    ex_session: date,
    calendar_policy_id: str,
    calendar_policy_version: str,
    upstream_source_evidence_fingerprint: str,
) -> DividendCalendarResolutionProof:
    """Bind an upstream T-to-X resolution; perform no calendar calculation."""

    fingerprint = compute_calendar_resolution_fingerprint(
        calendar_source_id=calendar_source_id,
        entitlement_session=entitlement_session,
        ex_session=ex_session,
        calendar_policy_id=calendar_policy_id,
        calendar_policy_version=calendar_policy_version,
        upstream_source_evidence_fingerprint=(
            upstream_source_evidence_fingerprint
        ),
    )
    return DividendCalendarResolutionProof(
        calendar_source_id=calendar_source_id,
        entitlement_session=entitlement_session,
        ex_session=ex_session,
        calendar_policy_id=calendar_policy_id,
        calendar_policy_version=calendar_policy_version,
        upstream_source_evidence_fingerprint=(
            upstream_source_evidence_fingerprint
        ),
        calendar_resolution_fingerprint=fingerprint,
    )


class CanonicalDividendAccountingEvidence(_ImmutableDividendEvidenceModel):
    """Canonical immutable evidence for one normalized ordinary dividend."""

    evidence_schema_version: Literal[
        "canonical_dividend_accounting_evidence.v0.1"
    ] = CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION
    canonical_distribution_event_id: _CanonicalText
    canonical_security_id: _CanonicalSecurityId
    entitlement_session: _SessionDate
    ex_session: _SessionDate
    distribution_type: Literal["ordinary_cash"] = (
        ORDINARY_CASH_DISTRIBUTION_TYPE
    )
    amount_per_share: _PositiveScale38Decimal
    amount_basis: Literal["HISTORICAL_SHARE_BASIS_D_H"] = (
        HISTORICAL_SHARE_AMOUNT_BASIS
    )
    normalization_method_id: Literal["GATE3_D_H_EXACT_RATIONAL_V0_1"] = (
        NORMALIZATION_METHOD_ID
    )
    normalization_scale: Literal[38] = NORMALIZATION_SCALE
    normalization_rounding_mode: Literal["ROUND_HALF_EVEN"] = (
        NORMALIZATION_ROUNDING_MODE
    )
    normalization_arithmetic_mode: Literal["EXACT_RATIONAL"] = (
        NORMALIZATION_ARITHMETIC_MODE
    )
    normalization_inputs_fingerprint: _Sha256
    normalization_inputs: Gate3DividendNormalizationInputs
    normalization_result: Gate3DividendNormalizationResult
    classification_proof: OrdinaryCashClassificationProof
    calendar_resolution_proof: DividendCalendarResolutionProof
    calendar_resolution_fingerprint: _Sha256
    currency: Literal["USD"] = CANONICAL_DIVIDEND_CURRENCY
    canonical_distribution_snapshot_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_accounting_evidence_integrity(self) -> Self:
        if (
            self.classification_proof.canonical_distribution_event_id
            != self.canonical_distribution_event_id
        ):
            raise ValueError(
                "classification proof must identify the accounting event"
            )
        if (
            self.classification_proof.classification_result
            != ORDINARY_CASH_DISTRIBUTION_TYPE
        ):
            raise ValueError("classification proof must resolve ordinary_cash")

        if (
            self.normalization_inputs.entitlement_session
            != self.entitlement_session
        ):
            raise ValueError(
                "normalization input entitlement session must equal accounting T"
            )
        if (
            self.calendar_resolution_proof.entitlement_session
            != self.entitlement_session
        ):
            raise ValueError("calendar proof entitlement session must equal T")
        if self.calendar_resolution_proof.ex_session != self.ex_session:
            raise ValueError("calendar proof resolved session must equal X")
        if self.entitlement_session >= self.ex_session:
            raise ValueError("entitlement_session must be before ex_session")

        expected_inputs_fingerprint = compute_normalization_inputs_fingerprint(
            self.normalization_inputs
        )
        if (
            self.normalization_result.normalization_inputs_fingerprint
            != expected_inputs_fingerprint
        ):
            raise ValueError(
                "normalization result fingerprint must bind normalization inputs"
            )
        if self.normalization_inputs_fingerprint != expected_inputs_fingerprint:
            raise ValueError(
                "normalization_inputs_fingerprint must match normalization inputs"
            )

        result = self.normalization_result
        if self.amount_per_share != result.d_h:
            raise ValueError("amount_per_share must equal normalized D_H")
        for field_name in (
            "amount_basis",
            "normalization_method_id",
            "normalization_scale",
            "normalization_rounding_mode",
            "normalization_arithmetic_mode",
            "normalization_inputs_fingerprint",
        ):
            if getattr(self, field_name) != getattr(result, field_name):
                raise ValueError(
                    f"{field_name} must match the normalization result"
                )

        if (
            self.calendar_resolution_fingerprint
            != self.calendar_resolution_proof.calendar_resolution_fingerprint
        ):
            raise ValueError(
                "calendar_resolution_fingerprint must match calendar proof"
            )
        return self


def build_canonical_dividend_accounting_evidence(
    *,
    canonical_distribution_event_id: str,
    canonical_security_id: str,
    normalization_inputs: Gate3DividendNormalizationInputs,
    normalization_result: Gate3DividendNormalizationResult,
    classification_proof: OrdinaryCashClassificationProof,
    calendar_resolution_proof: DividendCalendarResolutionProof,
    currency: str,
    canonical_distribution_snapshot_fingerprint: str,
    evidence_schema_version: str = (
        CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION
    ),
) -> CanonicalDividendAccountingEvidence:
    """Build evidence exclusively from an existing Gate-3 result and proofs."""

    if type(normalization_inputs) is not Gate3DividendNormalizationInputs:
        raise TypeError(
            "normalization_inputs must be Gate3DividendNormalizationInputs"
        )
    if type(normalization_result) is not Gate3DividendNormalizationResult:
        raise TypeError(
            "normalization_result must be Gate3DividendNormalizationResult"
        )
    if type(classification_proof) is not OrdinaryCashClassificationProof:
        raise TypeError(
            "classification_proof must be OrdinaryCashClassificationProof"
        )
    if type(calendar_resolution_proof) is not DividendCalendarResolutionProof:
        raise TypeError(
            "calendar_resolution_proof must be "
            "DividendCalendarResolutionProof"
        )

    return CanonicalDividendAccountingEvidence(
        evidence_schema_version=evidence_schema_version,
        canonical_distribution_event_id=canonical_distribution_event_id,
        canonical_security_id=canonical_security_id,
        entitlement_session=normalization_inputs.entitlement_session,
        ex_session=calendar_resolution_proof.ex_session,
        amount_per_share=normalization_result.d_h,
        amount_basis=normalization_result.amount_basis,
        normalization_method_id=normalization_result.normalization_method_id,
        normalization_scale=normalization_result.normalization_scale,
        normalization_rounding_mode=(
            normalization_result.normalization_rounding_mode
        ),
        normalization_arithmetic_mode=(
            normalization_result.normalization_arithmetic_mode
        ),
        normalization_inputs_fingerprint=(
            normalization_result.normalization_inputs_fingerprint
        ),
        normalization_inputs=normalization_inputs,
        normalization_result=normalization_result,
        classification_proof=classification_proof,
        calendar_resolution_proof=calendar_resolution_proof,
        calendar_resolution_fingerprint=(
            calendar_resolution_proof.calendar_resolution_fingerprint
        ),
        currency=currency,
        canonical_distribution_snapshot_fingerprint=(
            canonical_distribution_snapshot_fingerprint
        ),
    )


__all__ = [
    "CALENDAR_RESOLUTION_PROOF_HASH_DOMAIN",
    "CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION",
    "CANONICAL_DIVIDEND_CURRENCY",
    "CANONICAL_NEXT_TRADING_SESSION",
    "CLASSIFICATION_PROOF_HASH_DOMAIN",
    "DIVIDEND_CALENDAR_RESOLUTION_PROOF_SCHEMA_VERSION",
    "ORDINARY_CASH_CLASSIFICATION_PROOF_SCHEMA_VERSION",
    "ORDINARY_CASH_DISTRIBUTION_TYPE",
    "CanonicalDividendAccountingEvidence",
    "DividendCalendarResolutionProof",
    "OrdinaryCashClassificationProof",
    "build_canonical_dividend_accounting_evidence",
    "build_dividend_calendar_resolution_proof",
    "build_ordinary_cash_classification_proof",
    "compute_calendar_resolution_fingerprint",
    "compute_classification_proof_fingerprint",
]
