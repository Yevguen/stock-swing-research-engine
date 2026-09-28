"""Exact-rational Gate-3 ordinary-dividend normalization primitives."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from typing import Annotated, Final, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


NORMALIZATION_METHOD_ID: Final[str] = "GATE3_D_H_EXACT_RATIONAL_V0_1"
NORMALIZATION_SCALE: Final[int] = 38
NORMALIZATION_ROUNDING_MODE: Final[str] = ROUND_HALF_EVEN
NORMALIZATION_ARITHMETIC_MODE: Final[str] = "EXACT_RATIONAL"
HISTORICAL_SHARE_AMOUNT_BASIS: Final[str] = "HISTORICAL_SHARE_BASIS_D_H"
NORMALIZATION_INPUTS_HASH_DOMAIN: Final[str] = (
    "ordinary_dividend_normalization_inputs.v0.1"
)


def _require_session_date(value: object) -> object:
    if type(value) is not date:
        raise ValueError("must be a Python datetime.date")
    return value


def _require_positive_finite_decimal(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError(
            "must be an exact Decimal; binary floats, strings, ints, and bool "
            "are forbidden"
        )
    if not value.is_finite():
        raise ValueError("must be a finite Decimal")
    if value <= 0:
        raise ValueError("must be a positive Decimal")
    return value


def _require_positive_scale_38_decimal(value: object) -> object:
    value = _require_positive_finite_decimal(value)
    assert type(value) is Decimal
    if value.as_tuple().exponent != -NORMALIZATION_SCALE:
        raise ValueError("must be an exact fixed-scale-38 Decimal")
    return value


class _ImmutableNormalizationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


_SessionDate = Annotated[date, BeforeValidator(_require_session_date)]
_PositiveFiniteDecimal = Annotated[
    Decimal,
    BeforeValidator(_require_positive_finite_decimal),
]
_PositiveScale38Decimal = Annotated[
    Decimal,
    BeforeValidator(_require_positive_scale_38_decimal),
]
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Gate3DividendNormalizationInputs(_ImmutableNormalizationModel):
    """Strict finite-decimal inputs belonging to entitlement session ``T``."""

    entitlement_session: _SessionDate
    d_capitalspecial: _PositiveFiniteDecimal
    unadjusted_close_t: _PositiveFiniteDecimal
    close_capital_t: _PositiveFiniteDecimal


class Gate3DividendNormalizationResult(_ImmutableNormalizationModel):
    """Frozen Gate-3 output and the provenance needed by later evidence."""

    d_h: _PositiveScale38Decimal
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


def _decimal_to_fraction(value: Decimal) -> Fraction:
    """Convert one validated finite Decimal to its exact base-10 rational."""

    sign, digits, exponent = value.as_tuple()
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    if sign:
        coefficient = -coefficient
    if exponent >= 0:
        return Fraction(coefficient * (10**exponent), 1)
    return Fraction(coefficient, 10 ** (-exponent))


def _round_positive_fraction_half_even(value: Fraction, scale: int) -> int:
    """Return the integer coefficient after one exact half-even rounding."""

    if value <= 0:
        raise ValueError("normalization rational must be positive")
    scaled_numerator = value.numerator * (10**scale)
    quotient, remainder = divmod(scaled_numerator, value.denominator)
    doubled_remainder = remainder * 2
    if doubled_remainder > value.denominator or (
        doubled_remainder == value.denominator and quotient % 2 == 1
    ):
        quotient += 1
    return quotient


def _decimal_digits(coefficient: int) -> tuple[int, ...]:
    """Return decimal digits without a context-sensitive Decimal operation."""

    if coefficient < 0:
        raise ValueError("coefficient must be non-negative")
    if coefficient == 0:
        return (0,)
    reversed_digits: list[int] = []
    while coefficient:
        coefficient, digit = divmod(coefficient, 10)
        reversed_digits.append(digit)
    return tuple(reversed(reversed_digits))


def _normalization_inputs_payload(
    inputs: Gate3DividendNormalizationInputs,
) -> dict[str, object]:
    return {
        "entitlement_session": inputs.entitlement_session.isoformat(),
        "d_capitalspecial": _canonical_decimal(inputs.d_capitalspecial),
        "unadjusted_close_t": _canonical_decimal(inputs.unadjusted_close_t),
        "close_capital_t": _canonical_decimal(inputs.close_capital_t),
        "normalization_method_id": NORMALIZATION_METHOD_ID,
        "normalization_scale": NORMALIZATION_SCALE,
        "normalization_rounding_mode": NORMALIZATION_ROUNDING_MODE,
        "normalization_arithmetic_mode": NORMALIZATION_ARITHMETIC_MODE,
    }


def _canonical_decimal(value: Decimal) -> str:
    """Use the repository's canonical finite-Decimal text convention."""

    if not value.is_finite():
        raise ValueError("canonical normalization Decimal values must be finite")
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def compute_normalization_inputs_fingerprint(
    inputs: Gate3DividendNormalizationInputs,
) -> str:
    """Hash exact inputs and every frozen normalization-policy semantic."""

    if type(inputs) is not Gate3DividendNormalizationInputs:
        raise TypeError("inputs must be Gate3DividendNormalizationInputs")
    canonical_envelope = {
        "domain": NORMALIZATION_INPUTS_HASH_DOMAIN,
        "payload": _normalization_inputs_payload(inputs),
    }
    canonical_bytes = json.dumps(
        canonical_envelope,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def normalize_gate3_dividend(
    inputs: Gate3DividendNormalizationInputs,
) -> Gate3DividendNormalizationResult:
    """Compute ``D_H = D_CAPITALSPECIAL * UnadjustedClose_T / Close_CAPITAL_T``.

    All arithmetic before the one scale-38 half-even rounding is exact integer
    and rational arithmetic. Decimal construction from the rounded integer
    coefficient is exact and does not consult the ambient Decimal context.
    """

    if type(inputs) is not Gate3DividendNormalizationInputs:
        raise TypeError("inputs must be Gate3DividendNormalizationInputs")

    exact_result = (
        _decimal_to_fraction(inputs.d_capitalspecial)
        * _decimal_to_fraction(inputs.unadjusted_close_t)
        / _decimal_to_fraction(inputs.close_capital_t)
    )
    rounded_coefficient = _round_positive_fraction_half_even(
        exact_result,
        NORMALIZATION_SCALE,
    )
    if rounded_coefficient == 0:
        raise ValueError("normalized D_H rounds to zero at frozen scale 38")

    d_h = Decimal(
        (
            0,
            _decimal_digits(rounded_coefficient),
            -NORMALIZATION_SCALE,
        )
    )
    return Gate3DividendNormalizationResult(
        d_h=d_h,
        normalization_inputs_fingerprint=(
            compute_normalization_inputs_fingerprint(inputs)
        ),
    )


__all__ = [
    "Gate3DividendNormalizationInputs",
    "Gate3DividendNormalizationResult",
    "HISTORICAL_SHARE_AMOUNT_BASIS",
    "NORMALIZATION_ARITHMETIC_MODE",
    "NORMALIZATION_INPUTS_HASH_DOMAIN",
    "NORMALIZATION_METHOD_ID",
    "NORMALIZATION_ROUNDING_MODE",
    "NORMALIZATION_SCALE",
    "compute_normalization_inputs_fingerprint",
    "normalize_gate3_dividend",
]
