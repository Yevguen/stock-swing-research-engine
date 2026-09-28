"""Shared strict field vocabulary for the Phase 16C experiment layer.

Every Phase 16C model validates identity and shape exactly, in the fail-closed
style already used by Phase 13/15D/16B: an exact ``type is`` check rather than
a coercion, so a bool never passes as an int, a str never passes as a Decimal,
and a binary floating-point value never enters a monetary or price path.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def require_canonical_text(value: object) -> object:
    """Require non-empty text carrying no leading or trailing whitespace."""

    if type(value) is not str or not value or value != value.strip():
        raise ValueError("must be canonical non-empty text")
    return value


def require_optional_canonical_text(value: object) -> object:
    if value is None:
        return value
    return require_canonical_text(value)


def require_session_date(value: object) -> object:
    if type(value) is not date:
        raise ValueError("must be an exact Python datetime.date")
    return value


def require_finite_decimal(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError(
            "must be a Decimal; binary floats, strings, and bool are forbidden"
        )
    if not value.is_finite():
        raise ValueError("must be a finite Decimal; NaN and infinity are forbidden")
    return value


def require_optional_finite_decimal(value: object) -> object:
    if value is None:
        return value
    return require_finite_decimal(value)


def require_positive_decimal(value: object) -> object:
    require_finite_decimal(value)
    if value <= Decimal("0"):
        raise ValueError("must be a strictly positive Decimal")
    return value


def require_optional_positive_decimal(value: object) -> object:
    if value is None:
        return value
    return require_positive_decimal(value)


def require_non_negative_decimal(value: object) -> object:
    require_finite_decimal(value)
    if value < Decimal("0"):
        raise ValueError("must be a non-negative Decimal")
    return value


def require_optional_non_negative_decimal(value: object) -> object:
    if value is None:
        return value
    return require_non_negative_decimal(value)


def require_non_negative_count(value: object) -> object:
    if type(value) is not int:
        raise ValueError("must be an exact int; bool and float are forbidden")
    if value < 0:
        raise ValueError("must be a non-negative count")
    return value


def require_optional_non_negative_count(value: object) -> object:
    if value is None:
        return value
    return require_non_negative_count(value)


def require_positive_count(value: object) -> object:
    require_non_negative_count(value)
    if value <= 0:
        raise ValueError("must be a positive count")
    return value


def require_optional_positive_count(value: object) -> object:
    if value is None:
        return value
    return require_positive_count(value)


def require_tuple(value: object) -> object:
    if type(value) is not tuple:
        raise ValueError("must be an immutable tuple")
    return value


def require_exact_bool(value: object) -> object:
    if type(value) is not bool:
        raise ValueError("must be an exact bool")
    return value


def require_sha256(value: object) -> object:
    require_canonical_text(value)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("must be a lowercase SHA-256 digest")
    return value


def require_optional_sha256(value: object) -> object:
    if value is None:
        return value
    return require_sha256(value)


def require_exact_type(expected_type: type):
    """Build a validator demanding one exact type, never a subclass."""

    def validate(value: object) -> object:
        if type(value) is not expected_type:
            raise ValueError(f"must be an exact {expected_type.__name__}")
        return value

    return validate


def require_optional_exact_type(expected_type: type):
    inner = require_exact_type(expected_type)

    def validate(value: object) -> object:
        if value is None:
            return value
        return inner(value)

    return validate


def require_exact_enum(enum_type: type[StrEnum]):
    def validate(value: object) -> object:
        if type(value) is not enum_type:
            raise ValueError(f"must be an exact {enum_type.__name__} value")
        return value

    return validate


def require_optional_exact_enum(enum_type: type[StrEnum]):
    inner = require_exact_enum(enum_type)

    def validate(value: object) -> object:
        if value is None:
            return value
        return inner(value)

    return validate


class ImmutableBaselineExperimentModel(BaseModel):
    """The one frozen, extra-forbidding base of every Phase 16C model."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_default=True,
    )


CanonicalText = Annotated[str, BeforeValidator(require_canonical_text)]
OptionalCanonicalText = Annotated[
    str | None, BeforeValidator(require_optional_canonical_text)
]
SessionDate = Annotated[date, BeforeValidator(require_session_date)]
OptionalSessionDate = Annotated[
    date | None, BeforeValidator(require_optional_exact_type(date))
]
FiniteDecimal = Annotated[Decimal, BeforeValidator(require_finite_decimal)]
OptionalFiniteDecimal = Annotated[
    Decimal | None, BeforeValidator(require_optional_finite_decimal)
]
PositiveDecimal = Annotated[Decimal, BeforeValidator(require_positive_decimal)]
OptionalPositiveDecimal = Annotated[
    Decimal | None, BeforeValidator(require_optional_positive_decimal)
]
OptionalNonNegativeDecimal = Annotated[
    Decimal | None, BeforeValidator(require_optional_non_negative_decimal)
]
NonNegativeCount = Annotated[int, BeforeValidator(require_non_negative_count)]
OptionalNonNegativeCount = Annotated[
    int | None, BeforeValidator(require_optional_non_negative_count)
]
OptionalPositiveCount = Annotated[
    int | None, BeforeValidator(require_optional_positive_count)
]
ExactBool = Annotated[bool, BeforeValidator(require_exact_bool)]
Sha256 = Annotated[
    str,
    BeforeValidator(require_canonical_text),
    Field(pattern=r"^[0-9a-f]{64}$", strict=True),
]
OptionalSha256 = Annotated[
    str | None, BeforeValidator(require_optional_sha256)
]
GitCommitSha = Annotated[
    str,
    BeforeValidator(require_canonical_text),
    Field(pattern=r"^[0-9a-f]{40}$", strict=True),
]
CanonicalSecurityId = Annotated[
    str,
    BeforeValidator(require_canonical_text),
    Field(pattern=r"^NORGATE:[1-9][0-9]*$", strict=True),
]


__all__ = [
    "CanonicalSecurityId",
    "CanonicalText",
    "ExactBool",
    "FiniteDecimal",
    "GitCommitSha",
    "ImmutableBaselineExperimentModel",
    "NonNegativeCount",
    "OptionalCanonicalText",
    "OptionalFiniteDecimal",
    "OptionalNonNegativeCount",
    "OptionalNonNegativeDecimal",
    "OptionalPositiveCount",
    "OptionalPositiveDecimal",
    "OptionalSessionDate",
    "OptionalSha256",
    "PositiveDecimal",
    "SessionDate",
    "Sha256",
    "require_canonical_text",
    "require_exact_bool",
    "require_exact_enum",
    "require_exact_type",
    "require_finite_decimal",
    "require_non_negative_count",
    "require_non_negative_decimal",
    "require_optional_canonical_text",
    "require_optional_exact_enum",
    "require_optional_exact_type",
    "require_optional_finite_decimal",
    "require_optional_non_negative_count",
    "require_optional_non_negative_decimal",
    "require_optional_positive_count",
    "require_optional_positive_decimal",
    "require_optional_sha256",
    "require_positive_count",
    "require_positive_decimal",
    "require_session_date",
    "require_sha256",
    "require_tuple",
]
