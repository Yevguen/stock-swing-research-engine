"""Single frozen Phase-16 performance-measurement policy foundation."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from stock_swing_d1.research_metrics.hashing import (
    _compute_performance_measurement_policy_fingerprint,
    compute_performance_measurement_policy_fingerprint,
)


PERFORMANCE_MEASUREMENT_POLICY_SCHEMA_VERSION = (
    "performance_measurement_policy.v0.3"
)
PERFORMANCE_MEASUREMENT_POLICY_ID = (
    "explicit_research_performance_measurement_v0.3"
)
PERFORMANCE_MEASUREMENT_POLICY_VERSION = "0.3"

# Phase 16B.2 v0.2.1 Clause 44: the three arithmetic semantics the v0.3
# successor adds to the nine v0.2 methodology semantics. They are frozen
# constants, never runtime configuration, and `arithmetic.py` binds the
# canonical Decimal context to exactly these values so the v0.3 policy
# fingerprint certifies the arithmetic that produced a metric.
STATISTICAL_WORKING_PRECISION = 50
STATISTICAL_ROUNDING_MODE = "ROUND_HALF_EVEN"
CANONICAL_METRIC_QUANTUM = Decimal("1E-18")


class AnnualizationPeriodConvention(StrEnum):
    """How an explicit observation count represents one year."""

    FIXED_OBSERVATIONS_PER_YEAR = "fixed_observations_per_year"


class CagrDayCountConvention(StrEnum):
    """Explicit calendar-day divisor used to measure elapsed years.

    This remains a methodology vocabulary. Members other than the frozen
    canonical one stay available for historical artifacts, noncanonical
    research, and future separately authorized experiment contracts; they are
    never interchangeable with it, because the convention is fingerprint-bearing.
    """

    ACTUAL_365 = "actual_365"
    ACTUAL_365_25 = "actual_365_25"
    ACTUAL_365_2425 = "actual_365_2425"


class UndefinedMetricBehavior(StrEnum):
    """Required handling when a later metric has no defined value."""

    RETURN_NONE = "return_none"
    RAISE_VALIDATION_ERROR = "raise_validation_error"


class UndefinedMetricReason(StrEnum):
    """The complete frozen Phase-16 undefined-metric reason vocabulary.

    Phase 16B.2 v0.2.1 Clause 41 places this vocabulary here, alongside the
    other methodology vocabularies, rather than in ``models``: the model layer
    keeps its frozen no-new-enum scope guard, and metric models merely
    reference this enum. The family is closed (Clause 40) -- a mathematically
    undefined metric is never replaced with zero, NaN, or infinity, and a
    calculation/determinism failure is never disguised as one of these.

    The Phase 16B Amendment v0.1 adds the two reasons its winner/loser averages
    need. Neither is expressible by an existing member: a population that holds
    completed trades but no winner is a different fact from one that holds no
    completed trade at all, and collapsing the two would misreport why the
    average is absent. Closing the family means a reason is never invented at
    call time, not that a newly measured metric may go without an honest one.
    """

    NO_COMPLETED_TRADES = "NO_COMPLETED_TRADES"
    INSUFFICIENT_RETURN_OBSERVATIONS = "INSUFFICIENT_RETURN_OBSERVATIONS"
    ZERO_RETURN_VARIANCE = "ZERO_RETURN_VARIANCE"
    NO_DOWNSIDE_DEVIATION = "NO_DOWNSIDE_DEVIATION"
    ZERO_GROSS_LOSS = "ZERO_GROSS_LOSS"
    NO_WINNING_TRADES = "NO_WINNING_TRADES"
    NO_LOSING_TRADES = "NO_LOSING_TRADES"


def _require_exact_int(value: object) -> object:
    if type(value) is not int:
        raise ValueError("must be an exact int; bool and float are forbidden")
    return value


def _require_canonical_zero_decimal(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError("must be a Decimal; floats, strings, and bool are forbidden")
    if not value.is_finite():
        raise ValueError("must be a finite Decimal")
    if value != Decimal("0"):
        raise ValueError(
            "frozen Phase-16 semantics require exactly zero; "
            "a nonzero value is rejected, never normalized"
        )
    return value


def _require_canonical_metric_quantum(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError("must be a Decimal; floats, strings, and bool are forbidden")
    if not value.is_finite():
        raise ValueError("must be a finite Decimal")
    if value != CANONICAL_METRIC_QUANTUM or value.as_tuple().exponent != -18:
        raise ValueError(
            "frozen Phase-16B.2 semantics require exactly Decimal('1E-18'); "
            "a different value or scale is rejected, never normalized"
        )
    return value


def _require_exact_enum(enum_type: type[StrEnum]):
    def validate(value: object) -> object:
        if type(value) is not enum_type:
            raise ValueError(f"must be an exact {enum_type.__name__} value")
        return value

    return validate


_CanonicalZeroDecimal = Annotated[
    Decimal, BeforeValidator(_require_canonical_zero_decimal)
]
_CanonicalPeriodsPerYear = Annotated[
    Literal[252], BeforeValidator(_require_exact_int)
]
_CanonicalAnnualizationPeriodConvention = Annotated[
    Literal[AnnualizationPeriodConvention.FIXED_OBSERVATIONS_PER_YEAR],
    BeforeValidator(_require_exact_enum(AnnualizationPeriodConvention)),
]
_CanonicalCagrDayCountConvention = Annotated[
    Literal[CagrDayCountConvention.ACTUAL_365_25],
    BeforeValidator(_require_exact_enum(CagrDayCountConvention)),
]
_CanonicalUndefinedMetricBehavior = Annotated[
    Literal[UndefinedMetricBehavior.RETURN_NONE],
    BeforeValidator(_require_exact_enum(UndefinedMetricBehavior)),
]
_CanonicalStatisticalWorkingPrecision = Annotated[
    Literal[50], BeforeValidator(_require_exact_int)
]
_CanonicalStatisticalRoundingMode = Literal["ROUND_HALF_EVEN"]
_CanonicalMetricQuantum = Annotated[
    Decimal, BeforeValidator(_require_canonical_metric_quantum)
]
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]


class PerformanceMeasurementPolicy(BaseModel):
    """The one frozen production Phase-16 measurement methodology.

    Every semantic field is fixed by the frozen contract rather than chosen by a
    caller, so the model itself rejects a noncanonical value even when the
    builder is bypassed. Every semantically valid instance therefore shares one
    constant policy fingerprint; that fingerprint still certifies exactly which
    policy version governed a result and still detects tampering.

    Phase 16B.2 v0.2.1 Clauses 42-45 advance the canonical production identity
    to v0.3. The v0.2 policy is not edited in place: its nine-key payload under
    the ``performance_measurement_policy.v0.2`` domain remains exactly what it
    always was for already-published v0.2 artifacts, while v0.3 adds the three
    arithmetic semantics and its own hash domain, so a v0.2 policy can never
    masquerade as v0.3.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_default=True,
    )

    schema_version: Literal[
        "performance_measurement_policy.v0.3"
    ] = PERFORMANCE_MEASUREMENT_POLICY_SCHEMA_VERSION
    policy_id: Literal[
        "explicit_research_performance_measurement_v0.3"
    ] = PERFORMANCE_MEASUREMENT_POLICY_ID
    policy_version: Literal["0.3"] = PERFORMANCE_MEASUREMENT_POLICY_VERSION
    annualization_period_convention: _CanonicalAnnualizationPeriodConvention
    annualization_periods_per_year: _CanonicalPeriodsPerYear
    annual_risk_free_rate: _CanonicalZeroDecimal
    sortino_minimum_acceptable_return: _CanonicalZeroDecimal
    cagr_day_count_convention: _CanonicalCagrDayCountConvention
    undefined_metric_behavior: _CanonicalUndefinedMetricBehavior
    statistical_working_precision: _CanonicalStatisticalWorkingPrecision
    statistical_rounding_mode: _CanonicalStatisticalRoundingMode
    canonical_metric_quantum: _CanonicalMetricQuantum
    policy_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_fingerprint(self) -> Self:
        expected = compute_performance_measurement_policy_fingerprint(self)
        if self.policy_fingerprint != expected:
            raise ValueError("policy_fingerprint does not match policy content")
        return self


def build_performance_measurement_policy() -> PerformanceMeasurementPolicy:
    """Build the single canonical frozen Phase-16 measurement policy."""

    values = {
        "schema_version": PERFORMANCE_MEASUREMENT_POLICY_SCHEMA_VERSION,
        "policy_id": PERFORMANCE_MEASUREMENT_POLICY_ID,
        "policy_version": PERFORMANCE_MEASUREMENT_POLICY_VERSION,
        "annualization_period_convention": (
            AnnualizationPeriodConvention.FIXED_OBSERVATIONS_PER_YEAR
        ),
        "annualization_periods_per_year": 252,
        "annual_risk_free_rate": Decimal("0"),
        "sortino_minimum_acceptable_return": Decimal("0"),
        "cagr_day_count_convention": CagrDayCountConvention.ACTUAL_365_25,
        "undefined_metric_behavior": UndefinedMetricBehavior.RETURN_NONE,
        "statistical_working_precision": STATISTICAL_WORKING_PRECISION,
        "statistical_rounding_mode": STATISTICAL_ROUNDING_MODE,
        "canonical_metric_quantum": CANONICAL_METRIC_QUANTUM,
    }
    fingerprint = _compute_performance_measurement_policy_fingerprint(**values)
    return PerformanceMeasurementPolicy(
        **values, policy_fingerprint=fingerprint
    )


__all__ = [
    "CANONICAL_METRIC_QUANTUM",
    "PERFORMANCE_MEASUREMENT_POLICY_ID",
    "PERFORMANCE_MEASUREMENT_POLICY_SCHEMA_VERSION",
    "PERFORMANCE_MEASUREMENT_POLICY_VERSION",
    "STATISTICAL_ROUNDING_MODE",
    "STATISTICAL_WORKING_PRECISION",
    "AnnualizationPeriodConvention",
    "CagrDayCountConvention",
    "PerformanceMeasurementPolicy",
    "UndefinedMetricBehavior",
    "UndefinedMetricReason",
    "build_performance_measurement_policy",
]
