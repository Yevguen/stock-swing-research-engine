"""Immutable Phase 13 ordinary-dividend application, ledger, and outcome models.

Implements the Phase 13 in-memory portion of the frozen Ordinary Dividend
Amendment v0.7 (OD-9, OD-10, OD-11, OD-13, OD-14, OD-20.4): deterministic
dividend-application identity/payload hashing, exact context-independent
gross-cash arithmetic, the ``DIVIDEND_APPLIED`` ledger row, and the
APPLIED / REPLAYED / NO_OP_NOT_ENTITLED outcome model.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, model_validator

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CanonicalDividendAccountingEvidence,
)
from stock_swing_d1.data.ordinary_dividend_normalization import NORMALIZATION_SCALE
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY,
)
from stock_swing_d1.portfolio.portfolio_events import (
    PortfolioLedgerEventType,
    _CanonicalAssetId,
    _ImmutablePortfolioModel,
    _NonEmptyText,
    _NonNegativeDecimal,
    _NonNegativeInt,
    _PositiveDecimal,
    _PositiveQuantity,
    _SessionDate,
    _Sha256,
)
from stock_swing_d1.portfolio.portfolio_hashing import canonical_payload_sha256


DIVIDEND_LEDGER_ENTRY_SCHEMA_VERSION = "dividend_ledger_entry.v0.1"
DIVIDEND_APPLICATION_IDENTITY_SCHEMA_VERSION = (
    "phase13_dividend_application_identity.v0.1"
)
DIVIDEND_APPLICATION_PAYLOAD_SCHEMA_VERSION = (
    "phase13_dividend_application_payload.v0.1"
)
DIVIDEND_APPLICATION_ACCOUNTING_SCOPE = "PHASE13_STRATEGY_PORTFOLIO"


def _require_positive_scale38_decimal(value: object) -> object:
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


_PositiveScale38Decimal = Annotated[
    Decimal,
    BeforeValidator(_require_positive_scale38_decimal),
]


class DividendApplicationStatus(StrEnum):
    """The complete frozen OD-14.8 dividend-application outcome set."""

    APPLIED = "APPLIED"
    REPLAYED = "REPLAYED"
    NO_OP_NOT_ENTITLED = "NO_OP_NOT_ENTITLED"


def _digits_of(coefficient: int) -> tuple[int, ...]:
    """Return decimal digits without consulting the ambient Decimal context."""

    if coefficient < 0:
        raise ValueError("coefficient must be non-negative")
    if coefficient == 0:
        return (0,)
    reversed_digits: list[int] = []
    while coefficient:
        coefficient, digit = divmod(coefficient, 10)
        reversed_digits.append(digit)
    return tuple(reversed(reversed_digits))


def _coefficient_as_int(digits: tuple[int, ...]) -> int:
    """Convert an ``as_tuple()`` digit tuple to an int without ambient context."""

    value = 0
    for digit in digits:
        value = value * 10 + digit
    return value


def add_exact_decimal(base: Decimal, delta: Decimal) -> Decimal:
    """Add ``delta`` to ``base`` exactly, independent of ambient Decimal context.

    Python's ``+`` operator rounds its result to ``decimal.getcontext().prec``
    significant digits, which silently discards information whenever the
    exact sum needs more digits than the ambient context allows (OD-6.7's
    "no free ambient-context precision parameter" applies to dividend
    settled-cash mutation, not only to ``Q_T * D_H``). This decomposes both
    operands via ``as_tuple()``, aligns their integer coefficients to the
    smaller exact exponent, sums them as unbounded Python integers, and
    reassembles the result with the exact tuple constructor, which performs
    no rounding. Used for the one Phase 13 dividend cash-application site
    (OD-12.1) and its invariant replay so both can never round.
    """

    if type(base) is not Decimal or not base.is_finite():
        raise ValueError("base must be a finite Decimal")
    if type(delta) is not Decimal or not delta.is_finite():
        raise ValueError("delta must be a finite Decimal")

    base_sign, base_digits, base_exponent = base.as_tuple()
    delta_sign, delta_digits, delta_exponent = delta.as_tuple()
    exponent = min(base_exponent, delta_exponent)

    base_value = _coefficient_as_int(base_digits) * 10 ** (base_exponent - exponent)
    if base_sign:
        base_value = -base_value
    delta_value = _coefficient_as_int(delta_digits) * 10 ** (delta_exponent - exponent)
    if delta_sign:
        delta_value = -delta_value

    total = base_value + delta_value
    sign = 1 if total < 0 else 0
    return Decimal((sign, _digits_of(abs(total)), exponent))


def _negate_exact(value: Decimal) -> Decimal:
    """Negate a Decimal by reconstructing its exact tuple representation.

    Unlike Python's unary ``-`` (which passes through ``_fix()`` and can
    round to the ambient context precision), constructing directly from
    ``as_tuple()`` performs no arithmetic and therefore cannot round.
    """

    sign, digits, exponent = value.as_tuple()
    return Decimal((0 if sign else 1, digits, exponent))


def subtract_exact_decimal(base: Decimal, delta: Decimal) -> Decimal:
    """Subtract ``delta`` from ``base`` exactly, independent of ambient
    Decimal context (OD-6.7). The general-purpose subtraction counterpart
    to ``add_exact_decimal``, implemented by negating ``delta``'s exact
    tuple representation and reusing the same exact addition.
    """

    if type(base) is not Decimal or not base.is_finite():
        raise ValueError("base must be a finite Decimal")
    if type(delta) is not Decimal or not delta.is_finite():
        raise ValueError("delta must be a finite Decimal")
    return add_exact_decimal(base, _negate_exact(delta))


def exact_decimal_times_int(value: Decimal, multiplier: int) -> Decimal:
    """Multiply a finite Decimal by an exact integer, independent of
    ambient Decimal context (OD-6.7).

    Quantities throughout Phase 13/15D are genuine Python ``int`` values
    with no enforced upper bound, and prices/marks/cost bases are finite
    Decimals with no enforced significant-digit bound. Python's Decimal
    ``*`` passes its result through ``_fix()`` and rounds to the ambient
    context's precision once the exact product needs more significant
    digits than that precision allows -- this is reproducible even for
    quantity/price combinations that look economically unremarkable, not
    only pathological ones. This instead decomposes ``value`` via
    ``as_tuple()``, scales its integer coefficient by ``multiplier``
    using unbounded Python integer arithmetic (which never rounds), and
    reconstructs the exact Decimal result directly from its parts --
    never invoking ``Decimal.__mul__``. ``bool`` is rejected as a
    multiplier (``type(True) is bool``, not ``int``), and binary floats
    are rejected as ``value`` (``type(1.0) is float``, not ``Decimal``).
    """

    if type(value) is not Decimal or not value.is_finite():
        raise ValueError("value must be a finite Decimal")
    if type(multiplier) is not int:
        raise ValueError("multiplier must be a genuine int")

    value_sign, digits, exponent = value.as_tuple()
    coefficient = _coefficient_as_int(digits)
    signed_coefficient = -coefficient if value_sign else coefficient
    product = signed_coefficient * multiplier
    if product == 0:
        return Decimal((0, (0,), exponent))
    result_sign = 1 if product < 0 else 0
    return Decimal((result_sign, _digits_of(abs(product)), exponent))


def compute_gross_dividend_cash(q_t: int, d_h: Decimal) -> Decimal:
    """Compute ``Q_T * D_H`` exactly, independent of ambient Decimal context.

    ``d_h`` must already be the frozen exact fixed-scale-38 normalized
    Decimal (OD-6). The product is built directly from integer coefficients
    so no ambient ``decimal.getcontext().prec`` participates and no
    intermediate or cent rounding occurs (OD-11).
    """

    if type(q_t) is not int or q_t < 0:
        raise ValueError("q_t must be a non-negative int")
    if type(d_h) is not Decimal or not d_h.is_finite() or d_h <= 0:
        raise ValueError("d_h must be a positive finite Decimal")
    sign, digits, exponent = d_h.as_tuple()
    if sign != 0:
        raise ValueError("d_h must be positive")
    if exponent != -NORMALIZATION_SCALE:
        raise ValueError("d_h must be an exact fixed-scale-38 Decimal")

    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    product = coefficient * q_t
    return Decimal((0, _digits_of(product), -NORMALIZATION_SCALE))


class _DividendApplicationIdentityPayload(_ImmutablePortfolioModel):
    """OD-13.4 application-identity payload: hashed to build ``application_id``."""

    schema_version: Literal[
        "phase13_dividend_application_identity.v0.1"
    ] = DIVIDEND_APPLICATION_IDENTITY_SCHEMA_VERSION
    accounting_scope: Literal[
        "PHASE13_STRATEGY_PORTFOLIO"
    ] = DIVIDEND_APPLICATION_ACCOUNTING_SCOPE
    canonical_distribution_event_id: _NonEmptyText
    entitlement_session: _SessionDate
    ex_session: _SessionDate
    asset_id: _CanonicalAssetId
    attribution_trade_id: _NonEmptyText | None


def compute_dividend_application_id(
    *,
    canonical_distribution_event_id: str,
    entitlement_session: date,
    ex_session: date,
    asset_id: str,
    attribution_trade_id: str | None,
) -> str:
    """Compute the OD-13.4 deterministic dividend-application identity."""

    payload = _DividendApplicationIdentityPayload(
        canonical_distribution_event_id=canonical_distribution_event_id,
        entitlement_session=entitlement_session,
        ex_session=ex_session,
        asset_id=asset_id,
        attribution_trade_id=attribution_trade_id,
    )
    return canonical_payload_sha256(payload)


class _DividendApplicationPayload(_ImmutablePortfolioModel):
    """OD-13.5 application-payload fingerprint content."""

    schema_version: Literal[
        "phase13_dividend_application_payload.v0.1"
    ] = DIVIDEND_APPLICATION_PAYLOAD_SCHEMA_VERSION
    application_id: _NonEmptyText
    dividend_policy_semantic_identity: _NonEmptyText
    canonical_distribution_snapshot_fingerprint: _Sha256
    canonical_distribution_event_id: _NonEmptyText
    asset_id: _CanonicalAssetId
    entitlement_session: _SessionDate
    ex_session: _SessionDate
    attribution_trade_id: _NonEmptyText | None
    q_t: _NonNegativeInt
    d_h: _PositiveScale38Decimal
    amount_basis: _NonEmptyText
    normalization_method_id: _NonEmptyText
    normalization_scale: _NonNegativeInt
    normalization_rounding_mode: _NonEmptyText
    normalization_arithmetic_mode: _NonEmptyText
    normalization_inputs_fingerprint: _Sha256
    calendar_resolution_fingerprint: _Sha256
    gross_cash_amount: _NonNegativeDecimal
    currency: _NonEmptyText


def compute_dividend_application_payload_hash(
    *,
    application_id: str,
    evidence: CanonicalDividendAccountingEvidence,
    asset_id: str,
    attribution_trade_id: str | None,
    q_t: int,
    gross_cash_amount: Decimal,
) -> str:
    """Compute the OD-13.5 deterministic dividend-application payload hash."""

    payload = _DividendApplicationPayload(
        application_id=application_id,
        dividend_policy_semantic_identity=(
            ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY
        ),
        canonical_distribution_snapshot_fingerprint=(
            evidence.canonical_distribution_snapshot_fingerprint
        ),
        canonical_distribution_event_id=evidence.canonical_distribution_event_id,
        asset_id=asset_id,
        entitlement_session=evidence.entitlement_session,
        ex_session=evidence.ex_session,
        attribution_trade_id=attribution_trade_id,
        q_t=q_t,
        d_h=evidence.amount_per_share,
        amount_basis=evidence.amount_basis,
        normalization_method_id=evidence.normalization_method_id,
        normalization_scale=evidence.normalization_scale,
        normalization_rounding_mode=evidence.normalization_rounding_mode,
        normalization_arithmetic_mode=evidence.normalization_arithmetic_mode,
        normalization_inputs_fingerprint=(
            evidence.normalization_inputs_fingerprint
        ),
        calendar_resolution_fingerprint=evidence.calendar_resolution_fingerprint,
        gross_cash_amount=gross_cash_amount,
        currency=evidence.currency,
    )
    return canonical_payload_sha256(payload)


class DividendApplicationOutcome(_ImmutablePortfolioModel):
    """OD-14.8 recorded discharge outcome for one supplied dividend event."""

    canonical_distribution_event_id: _NonEmptyText
    application_id: _NonEmptyText
    payload_sha256: _Sha256
    asset_id: _CanonicalAssetId
    entitlement_session: _SessionDate
    ex_session: _SessionDate
    q_t: _NonNegativeInt
    attribution_trade_id: _NonEmptyText | None
    status: DividendApplicationStatus

    @model_validator(mode="after")
    def validate_status_consistency(self) -> Self:
        if self.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED:
            if self.q_t != 0 or self.attribution_trade_id is not None:
                raise ValueError(
                    "NO_OP_NOT_ENTITLED requires zero quantity and no "
                    "trade attribution"
                )
        else:
            if self.q_t <= 0 or self.attribution_trade_id is None:
                raise ValueError(
                    "APPLIED/REPLAYED requires positive quantity and "
                    "trade attribution"
                )
        return self


class DividendLedgerEntry(_ImmutablePortfolioModel):
    """One immutable ``DIVIDEND_APPLIED`` Phase 13 ledger row (OD-14.1/14.3)."""

    schema_version: Literal[
        "dividend_ledger_entry.v0.1"
    ] = DIVIDEND_LEDGER_ENTRY_SCHEMA_VERSION
    session: _SessionDate
    sequence_in_session: _NonNegativeInt
    event_type: Literal[
        PortfolioLedgerEventType.DIVIDEND_APPLIED
    ] = PortfolioLedgerEventType.DIVIDEND_APPLIED
    application_id: _NonEmptyText
    canonical_distribution_event_id: _NonEmptyText
    canonical_distribution_snapshot_fingerprint: _Sha256
    asset_id: _CanonicalAssetId
    entitlement_session: _SessionDate
    attribution_trade_id: _NonEmptyText
    q_t: _PositiveQuantity
    d_h: _PositiveScale38Decimal
    amount_basis: _NonEmptyText
    normalization_method_id: _NonEmptyText
    normalization_scale: _NonNegativeInt
    normalization_rounding_mode: _NonEmptyText
    normalization_arithmetic_mode: _NonEmptyText
    normalization_inputs_fingerprint: _Sha256
    calendar_resolution_fingerprint: _Sha256
    currency: _NonEmptyText
    gross_cash_amount: _PositiveDecimal
    settled_cash_delta: _PositiveDecimal
    settled_cash_after: _NonNegativeDecimal
    source_payload_sha256: _Sha256
    state_hash_before: _Sha256
    state_hash_after: _Sha256

    @model_validator(mode="after")
    def validate_cash_identity(self) -> Self:
        if self.entitlement_session >= self.session:
            raise ValueError("entitlement_session must precede the ex-session")
        expected_gross = compute_gross_dividend_cash(self.q_t, self.d_h)
        if self.gross_cash_amount != expected_gross:
            raise ValueError("gross_cash_amount must equal q_t * d_h exactly")
        if self.settled_cash_delta != self.gross_cash_amount:
            raise ValueError(
                "settled_cash_delta must equal gross_cash_amount exactly"
            )
        return self


__all__ = [
    "DIVIDEND_APPLICATION_ACCOUNTING_SCOPE",
    "DIVIDEND_APPLICATION_IDENTITY_SCHEMA_VERSION",
    "DIVIDEND_APPLICATION_PAYLOAD_SCHEMA_VERSION",
    "DIVIDEND_LEDGER_ENTRY_SCHEMA_VERSION",
    "DividendApplicationOutcome",
    "DividendApplicationStatus",
    "DividendLedgerEntry",
    "add_exact_decimal",
    "compute_dividend_application_id",
    "compute_dividend_application_payload_hash",
    "compute_gross_dividend_cash",
    "exact_decimal_times_int",
    "subtract_exact_decimal",
]
