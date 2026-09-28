"""Canonical Phase 16B.2 statistical Decimal arithmetic infrastructure.

This module owns numerical infrastructure only. It never reconstructs
economics: it constructs the one explicit canonical ``decimal.Context``, wraps
the standard-library Decimal operations that run under it, quantizes a finished
metric at its canonical boundary, and re-exposes the frozen upstream
exact-arithmetic primitives that authoritative monetary operations must use.

Two rules shape every helper here (Phase 16B.2 v0.2.1 Clauses 32/33/38):

* Nothing reads ``decimal.getcontext()`` and nothing calls ``setcontext()``.
  Every operation is invoked as a method of an explicitly supplied canonical
  ``Context``, so the caller's ambient precision, rounding, traps, exponent
  limits and flags neither influence a canonical result nor change because one
  was computed.
* No binary float participates. ``float(...)``, ``math``, NumPy, pandas and
  binary-float exponentiation are all absent from the authoritative path;
  ``FloatOperation`` is additionally trapped so a stray float can only raise.

Determinism comes from one pinned 50-digit context plus a single final
quantization to ``Decimal("1E-18")`` under ``ROUND_HALF_EVEN`` -- never from an
agreement-across-precisions ladder, which is a heuristic rather than a proof
(Clause 36). ``Decimal.ln``, ``Decimal.exp`` and ``Decimal.sqrt`` are documented
as correctly rounded to the context precision, so a correctly-rounded result is
mathematically unique and independent of the Decimal implementation in use.
"""

from __future__ import annotations

from decimal import (
    Clamped,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    FloatOperation,
    Inexact,
    InvalidOperation,
    Overflow,
    Rounded,
    Subnormal,
    Underflow,
)

from stock_swing_d1.backtest_results import (
    add_exact_decimal,
    exact_decimal_times_int,
    subtract_exact_decimal,
    sum_exact_decimal,
)
from stock_swing_d1.research_metrics.errors import (
    ResearchMetricsCalculationError,
)
from stock_swing_d1.research_metrics.policy import (
    CANONICAL_METRIC_QUANTUM,
    STATISTICAL_ROUNDING_MODE,
    STATISTICAL_WORKING_PRECISION,
)


CANONICAL_STATISTICAL_EMIN = -999999
CANONICAL_STATISTICAL_EMAX = 999999
CANONICAL_STATISTICAL_CAPITALS = 1
CANONICAL_STATISTICAL_CLAMP = 0

# Trapped: a division by zero, an invalid nonlinear domain, an overflow, or a
# binary-float operand raises instead of silently publishing Infinity or NaN
# (Clause 40, AR-10/AR-11). Untrapped: ordinary inexactness/rounding of a
# 50-digit statistical result is expected, not an error.
CANONICAL_STATISTICAL_TRAPS: dict[type[ArithmeticError], bool] = {
    InvalidOperation: True,
    DivisionByZero: True,
    Overflow: True,
    FloatOperation: True,
    Inexact: False,
    Rounded: False,
    Underflow: False,
    Subnormal: False,
    Clamped: False,
}


def canonical_statistical_context() -> Context:
    """Build the one explicit canonical statistical Decimal context.

    A fresh context is returned per call and is never installed as the ambient
    context, so canonical computation neither inherits nor mutates caller
    numerical semantics. Flags start cleared.
    """

    context = Context(
        prec=STATISTICAL_WORKING_PRECISION,
        rounding=STATISTICAL_ROUNDING_MODE,
        Emin=CANONICAL_STATISTICAL_EMIN,
        Emax=CANONICAL_STATISTICAL_EMAX,
        capitals=CANONICAL_STATISTICAL_CAPITALS,
        clamp=CANONICAL_STATISTICAL_CLAMP,
        traps=dict(CANONICAL_STATISTICAL_TRAPS),
    )
    context.clear_flags()
    return context


def _canonical(code: str, operation, *operands: Decimal) -> Decimal:
    """Run one canonical context operation, failing closed on a trapped signal.

    A trapped Decimal signal means canonical arithmetic could not produce the
    requested value. That is a typed calculation failure (Clause 39.3), never
    an undefined-metric value: callers guard every mathematically undefined
    denominator or domain before reaching this helper.
    """

    for operand in operands:
        if type(operand) is not Decimal or not operand.is_finite():
            raise ResearchMetricsCalculationError(
                code, "canonical statistical arithmetic requires finite Decimals"
            )
    try:
        result = operation(*operands)
    except DecimalException as error:
        raise ResearchMetricsCalculationError(
            code, f"canonical statistical arithmetic failed: {error}"
        ) from error
    if type(result) is not Decimal or not result.is_finite():
        raise ResearchMetricsCalculationError(
            code, "canonical statistical arithmetic produced a nonfinite result"
        )
    return result


def add(left: Decimal, right: Decimal, *, context: Context) -> Decimal:
    """Add under the canonical context."""

    return _canonical("CANONICAL_ADD", context.add, left, right)


def subtract(left: Decimal, right: Decimal, *, context: Context) -> Decimal:
    """Subtract under the canonical context."""

    return _canonical("CANONICAL_SUBTRACT", context.subtract, left, right)


def multiply(left: Decimal, right: Decimal, *, context: Context) -> Decimal:
    """Multiply under the canonical context."""

    return _canonical("CANONICAL_MULTIPLY", context.multiply, left, right)


def divide(
    numerator: Decimal, denominator: Decimal, *, context: Context
) -> Decimal:
    """Divide under the canonical context; a zero divisor fails closed."""

    return _canonical("CANONICAL_DIVIDE", context.divide, numerator, denominator)


def square_root(value: Decimal, *, context: Context) -> Decimal:
    """Take the standard-library Decimal square root under the canonical context."""

    return _canonical("CANONICAL_SQUARE_ROOT", context.sqrt, value)


def natural_logarithm(value: Decimal, *, context: Context) -> Decimal:
    """Take the standard-library Decimal natural logarithm under the canonical context."""

    return _canonical("CANONICAL_NATURAL_LOGARITHM", context.ln, value)


def exponential(value: Decimal, *, context: Context) -> Decimal:
    """Take the standard-library Decimal exponential under the canonical context."""

    return _canonical("CANONICAL_EXPONENTIAL", context.exp, value)


def quantize_canonical_metric(value: Decimal, *, context: Context) -> Decimal:
    """Quantize one finished metric to the canonical result quantum.

    Applied once, at the final canonical metric boundary (Clause 30). It is
    never applied to an intermediate calculation, and never to authoritative
    source money, whose exact scale belongs to the upstream accounting layer.
    """

    return _canonical(
        "CANONICAL_QUANTIZE",
        lambda operand: operand.quantize(
            CANONICAL_METRIC_QUANTUM,
            rounding=STATISTICAL_ROUNDING_MODE,
            context=context,
        ),
        value,
    )


__all__ = [
    "CANONICAL_STATISTICAL_CAPITALS",
    "CANONICAL_STATISTICAL_CLAMP",
    "CANONICAL_STATISTICAL_EMAX",
    "CANONICAL_STATISTICAL_EMIN",
    "CANONICAL_STATISTICAL_TRAPS",
    "add",
    "add_exact_decimal",
    "canonical_statistical_context",
    "divide",
    "exact_decimal_times_int",
    "exponential",
    "multiply",
    "natural_logarithm",
    "quantize_canonical_metric",
    "square_root",
    "subtract",
    "subtract_exact_decimal",
    "sum_exact_decimal",
]
