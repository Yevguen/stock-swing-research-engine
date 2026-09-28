"""Phase 16B.2 canonical statistical-arithmetic acceptance cases (AR-01..AR-13)."""

from __future__ import annotations

import decimal
from decimal import (
    Clamped,
    Context,
    Decimal,
    DivisionByZero,
    FloatOperation,
    Inexact,
    InvalidOperation,
    Overflow,
    ROUND_HALF_EVEN,
    Rounded,
    Subnormal,
    Underflow,
)

import pytest

from stock_swing_d1.research_metrics import (
    CANONICAL_METRIC_QUANTUM,
    CANONICAL_STATISTICAL_EMAX,
    CANONICAL_STATISTICAL_EMIN,
    CANONICAL_STATISTICAL_TRAPS,
    STATISTICAL_ROUNDING_MODE,
    STATISTICAL_WORKING_PRECISION,
    ResearchMetricsCalculationError,
    ResearchMetricValue,
    calculate_research_metrics,
    canonical_statistical_context,
    quantize_canonical_metric,
)
from stock_swing_d1.research_metrics import arithmetic


def _ambient_snapshot() -> dict:
    context = decimal.getcontext()
    return {
        "prec": context.prec,
        "rounding": context.rounding,
        "Emin": context.Emin,
        "Emax": context.Emax,
        "capitals": context.capitals,
        "clamp": context.clamp,
        "traps": dict(context.traps),
        "flags": dict(context.flags),
    }


def test_the_canonical_context_pins_every_numerical_semantic():
    """AR-01/AR-02/AR-03 plus the complete Clause 32 context configuration."""

    context = canonical_statistical_context()

    assert context.prec == 50 == STATISTICAL_WORKING_PRECISION
    assert context.rounding == ROUND_HALF_EVEN == STATISTICAL_ROUNDING_MODE
    assert context.Emin == -999999 == CANONICAL_STATISTICAL_EMIN
    assert context.Emax == 999999 == CANONICAL_STATISTICAL_EMAX
    assert context.capitals == 1
    assert context.clamp == 0
    assert CANONICAL_METRIC_QUANTUM == Decimal("1E-18")
    assert CANONICAL_METRIC_QUANTUM.as_tuple().exponent == -18


def test_the_canonical_context_traps_exactly_the_frozen_signals():
    context = canonical_statistical_context()

    for signal in (InvalidOperation, DivisionByZero, Overflow, FloatOperation):
        assert context.traps[signal] is True
        assert CANONICAL_STATISTICAL_TRAPS[signal] is True
    for signal in (Inexact, Rounded, Underflow, Subnormal, Clamped):
        assert context.traps[signal] is False
        assert CANONICAL_STATISTICAL_TRAPS[signal] is False


def test_the_canonical_context_starts_with_cleared_flags():
    context = canonical_statistical_context()

    assert not any(context.flags.values())


def test_a_fresh_context_is_returned_per_call_and_never_installed():
    first = canonical_statistical_context()
    second = canonical_statistical_context()

    assert first is not second
    assert first is not decimal.getcontext()
    arithmetic.divide(Decimal("1"), Decimal("3"), context=first)
    assert not any(second.flags.values())


def test_quantization_uses_the_canonical_quantum_and_half_even_rounding():
    """AR-02/AR-03: exact half-way values round to even, not away from zero."""

    context = canonical_statistical_context()

    assert quantize_canonical_metric(
        Decimal("0.0000000000000000005"), context=context
    ) == Decimal("0E-18")
    assert quantize_canonical_metric(
        Decimal("0.0000000000000000015"), context=context
    ) == Decimal("0.000000000000000002")
    assert quantize_canonical_metric(
        Decimal("0.0000000000000000025"), context=context
    ) == Decimal("0.000000000000000002")
    quantized = quantize_canonical_metric(Decimal("0.1"), context=context)
    assert quantized.as_tuple().exponent == -18


def test_division_by_zero_fails_closed_instead_of_publishing_infinity():
    """AR-10."""

    context = canonical_statistical_context()

    with pytest.raises(ResearchMetricsCalculationError) as captured:
        arithmetic.divide(Decimal("1"), Decimal("0"), context=context)
    assert captured.value.code == "CANONICAL_DIVIDE"


def test_an_invalid_nonlinear_domain_fails_closed_instead_of_publishing_nan():
    """AR-11."""

    context = canonical_statistical_context()

    for operation, operand in (
        (arithmetic.square_root, Decimal("-1")),
        (arithmetic.natural_logarithm, Decimal("-1")),
    ):
        with pytest.raises(ResearchMetricsCalculationError):
            operation(operand, context=context)


def test_the_natural_logarithm_of_zero_fails_closed():
    context = canonical_statistical_context()

    with pytest.raises(ResearchMetricsCalculationError):
        arithmetic.natural_logarithm(Decimal("0"), context=context)


@pytest.mark.parametrize(
    "operation",
    (
        arithmetic.add,
        arithmetic.subtract,
        arithmetic.multiply,
        arithmetic.divide,
    ),
)
def test_a_binary_float_can_never_reach_a_canonical_operation(operation):
    """AR-04: no binary float determines a canonical metric value."""

    context = canonical_statistical_context()

    with pytest.raises(ResearchMetricsCalculationError):
        operation(0.1, Decimal("1"), context=context)
    with pytest.raises(ResearchMetricsCalculationError):
        operation(Decimal("1"), 0.1, context=context)


def test_a_nonfinite_operand_can_never_reach_a_canonical_operation():
    context = canonical_statistical_context()

    for operand in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
        with pytest.raises(ResearchMetricsCalculationError):
            arithmetic.add(operand, Decimal("1"), context=context)


_AMBIENT_VARIATIONS = (
    Context(prec=28),
    Context(prec=60),
    Context(prec=200),
    Context(prec=28, rounding=decimal.ROUND_UP),
    Context(prec=9, rounding=decimal.ROUND_FLOOR, Emin=-99, Emax=99),
    # Every trap inverted relative to the canonical configuration: an
    # untrapped DivisionByZero in the caller's context would otherwise let a
    # canonical result become Infinity instead of failing closed.
    Context(
        prec=28,
        traps={
            InvalidOperation: False,
            DivisionByZero: False,
            Overflow: False,
            FloatOperation: False,
            Inexact: True,
            Rounded: True,
            Underflow: True,
            Subnormal: True,
            Clamped: True,
        },
    ),
    Context(prec=400, Emin=-1000, Emax=1000, capitals=0, clamp=1),
)


@pytest.mark.parametrize("ambient", _AMBIENT_VARIATIONS)
def test_canonical_output_is_identical_under_every_ambient_context(
    ambient, canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """AR-05/AR-06/AR-07: precision, rounding, traps and exponent limits."""

    baseline = calculate_research_metrics(
        source_result=canonical_audit_result,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )

    original = decimal.getcontext()
    decimal.setcontext(ambient)
    try:
        under_variation = calculate_research_metrics(
            source_result=canonical_audit_result,
            policy=performance_policy,
            benchmark_series=canonical_benchmark_series,
        )
    finally:
        decimal.setcontext(original)

    assert under_variation == baseline
    assert under_variation.result_fingerprint == baseline.result_fingerprint


def test_the_callers_ambient_context_and_flags_are_unchanged(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """AR-08/AR-09."""

    decimal.getcontext().clear_flags()
    before = _ambient_snapshot()

    calculate_research_metrics(
        source_result=canonical_audit_result,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )

    assert _ambient_snapshot() == before
    assert not any(decimal.getcontext().flags.values())


def test_no_canonical_metric_is_ever_nan_or_infinite(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """AR-11/UF-02/UF-03/UF-04."""

    result = calculate_research_metrics(
        source_result=canonical_audit_result,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )

    for value in _every_decimal(result.model_dump(mode="python")):
        assert value.is_finite()


def test_every_derived_metric_is_quantized_only_at_the_final_boundary(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """AR-12: published derived metrics carry exactly the canonical scale.

    Authoritative monetary facts keep their own upstream scale, which is how
    the guard proves the quantum was applied at the metric boundary rather
    than smeared over the source evidence.
    """

    result = calculate_research_metrics(
        source_result=canonical_audit_result,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )

    groups = (
        result.strategy_metrics,
        result.trade_metrics,
        result.benchmark_metrics,
        result.relative_metrics,
    )
    derived = 0
    for group in groups:
        for name in type(group).model_fields:
            member = getattr(group, name)
            if type(member) is not ResearchMetricValue:
                continue
            if member.value is None:
                continue
            derived += 1
            assert member.value.as_tuple().exponent == -18
    assert derived > 0

    assert result.strategy_metrics.ending_equity == Decimal("110000")
    assert result.strategy_metrics.ending_equity.as_tuple().exponent != -18
    assert result.trade_metrics.gross_profit.as_tuple().exponent != -18


def _every_decimal(value):
    if isinstance(value, Decimal):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _every_decimal(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _every_decimal(item)
