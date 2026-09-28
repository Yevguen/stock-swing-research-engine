from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from stock_swing_d1.research_metrics import (
    BENCHMARK_PERFORMANCE_METRICS_SCHEMA_VERSION,
    RELATIVE_PERFORMANCE_METRICS_SCHEMA_VERSION,
    RESEARCH_METRIC_VALUE_SCHEMA_VERSION,
    RESEARCH_METRICS_RESULT_SCHEMA_VERSION,
    STRATEGY_PERFORMANCE_METRICS_SCHEMA_VERSION,
    TRADE_PERFORMANCE_METRICS_SCHEMA_VERSION,
    BenchmarkPerformanceMetrics,
    RelativePerformanceMetrics,
    ResearchMetricValue,
    ResearchMetricsProvenance,
    StrategyPerformanceMetrics,
    TradePerformanceMetrics,
    UndefinedMetricReason,
    build_research_metrics_result,
    defined_metric,
    undefined_metric,
)


def _strategy_metrics(**overrides) -> StrategyPerformanceMetrics:
    values = {
        "ending_equity": Decimal("110000"),
        "net_pnl": Decimal("10000"),
        "total_return": defined_metric(Decimal("0.100000000000000000")),
        "cagr": defined_metric(Decimal("0.100000000000000000")),
        "annualized_volatility": defined_metric(Decimal("0.2")),
        "maximum_drawdown": defined_metric(Decimal("0.05")),
        "sharpe_ratio": defined_metric(Decimal("1.5")),
        "sortino_ratio": defined_metric(Decimal("2.5")),
    }
    values.update(overrides)
    return StrategyPerformanceMetrics(**values)


def _trade_metrics(**overrides) -> TradePerformanceMetrics:
    values = {
        "completed_trade_count": 2,
        "win_count": 1,
        "loss_count": 1,
        "breakeven_count": 0,
        "gross_profit": Decimal("205"),
        "gross_loss": Decimal("100"),
        "win_rate": defined_metric(Decimal("0.5")),
        "loss_rate": defined_metric(Decimal("0.5")),
        "breakeven_rate": defined_metric(Decimal("0")),
        "profit_factor": defined_metric(Decimal("2.05")),
        "expectancy": defined_metric(Decimal("52.5")),
        "average_winner_usd": defined_metric(Decimal("205")),
        "average_loser_usd": defined_metric(Decimal("-100")),
        "worst_trade_usd": defined_metric(Decimal("-100")),
        "average_holding_sessions": defined_metric(Decimal("2")),
    }
    values.update(overrides)
    return TradePerformanceMetrics(**values)


def _benchmark_metrics() -> BenchmarkPerformanceMetrics:
    return BenchmarkPerformanceMetrics(
        ending_equity=Decimal("104000"),
        total_return=defined_metric(Decimal("0.04")),
        cagr=defined_metric(Decimal("0.04")),
        annualized_volatility=defined_metric(Decimal("0.1")),
        maximum_drawdown=defined_metric(Decimal("0.01")),
        sharpe_ratio=defined_metric(Decimal("0.9")),
        sortino_ratio=defined_metric(Decimal("1.1")),
    )


def _relative_metrics() -> RelativePerformanceMetrics:
    return RelativePerformanceMetrics(
        strategy_minus_benchmark_total_return=defined_metric(Decimal("0.06")),
        strategy_minus_benchmark_cagr=defined_metric(Decimal("0.06")),
        ending_wealth_ratio=defined_metric(Decimal("1.057692307692307692")),
    )


def test_metric_schema_identities_advance_only_where_the_contract_changed():
    """The trade/result payloads and metric-value reason contract advanced."""

    assert STRATEGY_PERFORMANCE_METRICS_SCHEMA_VERSION == (
        "strategy_performance_metrics.v0.2"
    )
    assert TRADE_PERFORMANCE_METRICS_SCHEMA_VERSION == (
        "trade_performance_metrics.v0.3"
    )
    assert BENCHMARK_PERFORMANCE_METRICS_SCHEMA_VERSION == (
        "benchmark_performance_metrics.v0.2"
    )
    assert RELATIVE_PERFORMANCE_METRICS_SCHEMA_VERSION == (
        "relative_performance_metrics.v0.2"
    )
    assert RESEARCH_METRICS_RESULT_SCHEMA_VERSION == (
        "research_metrics_result.v0.3"
    )
    assert RESEARCH_METRIC_VALUE_SCHEMA_VERSION == (
        "research_metric_value.v0.2"
    )
    assert _strategy_metrics().schema_version == (
        "strategy_performance_metrics.v0.2"
    )
    assert _trade_metrics().schema_version == "trade_performance_metrics.v0.3"
    assert _benchmark_metrics().schema_version == (
        "benchmark_performance_metrics.v0.2"
    )
    assert _relative_metrics().schema_version == (
        "relative_performance_metrics.v0.2"
    )


@pytest.mark.parametrize(
    "reason",
    (
        UndefinedMetricReason.NO_WINNING_TRADES,
        UndefinedMetricReason.NO_LOSING_TRADES,
    ),
)
def test_metric_value_v0_2_accepts_the_amendment_reason_codes(reason):
    metric = ResearchMetricValue(undefined_reason=reason)
    assert metric.schema_version == "research_metric_value.v0.2"
    assert metric.undefined_reason is reason


@pytest.mark.parametrize(
    "reason",
    (
        UndefinedMetricReason.NO_WINNING_TRADES,
        UndefinedMetricReason.NO_LOSING_TRADES,
    ),
)
def test_v0_1_identity_cannot_describe_an_amendment_reason(reason):
    with pytest.raises(ValidationError):
        ResearchMetricValue(
            schema_version="research_metric_value.v0.1",
            undefined_reason=reason,
        )


def test_metric_groups_carry_the_frozen_phase16a_inventory():
    assert tuple(StrategyPerformanceMetrics.model_fields) == (
        "schema_version",
        "ending_equity",
        "net_pnl",
        "total_return",
        "cagr",
        "annualized_volatility",
        "maximum_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
    )
    assert tuple(TradePerformanceMetrics.model_fields) == (
        "schema_version",
        "completed_trade_count",
        "win_count",
        "loss_count",
        "breakeven_count",
        "gross_profit",
        "gross_loss",
        "win_rate",
        "loss_rate",
        "breakeven_rate",
        "profit_factor",
        "expectancy",
        "average_winner_usd",
        "average_loser_usd",
        "worst_trade_usd",
        "average_holding_sessions",
    )
    assert tuple(BenchmarkPerformanceMetrics.model_fields) == (
        "schema_version",
        "ending_equity",
        "total_return",
        "cagr",
        "annualized_volatility",
        "maximum_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
    )
    assert tuple(RelativePerformanceMetrics.model_fields) == (
        "schema_version",
        "strategy_minus_benchmark_total_return",
        "strategy_minus_benchmark_cagr",
        "ending_wealth_ratio",
    )


def test_metric_groups_are_immutable_and_reject_unknown_metrics():
    for group in (
        _strategy_metrics(),
        _trade_metrics(),
        _benchmark_metrics(),
        _relative_metrics(),
    ):
        with pytest.raises(ValidationError):
            group.schema_version = "changed"
        with pytest.raises(ValidationError):
            type(group)(
                **group.model_dump(mode="python"), unfrozen_metric=Decimal("1")
            )


def test_a_metric_is_either_defined_or_carries_exactly_one_reason():
    defined = defined_metric(Decimal("0.25"))
    assert defined.value == Decimal("0.25")
    assert defined.undefined_reason is None

    absent = undefined_metric(UndefinedMetricReason.NO_COMPLETED_TRADES)
    assert absent.value is None
    assert absent.undefined_reason is UndefinedMetricReason.NO_COMPLETED_TRADES

    with pytest.raises(ValidationError):
        ResearchMetricValue()
    with pytest.raises(ValidationError):
        ResearchMetricValue(
            value=Decimal("0"),
            undefined_reason=UndefinedMetricReason.ZERO_GROSS_LOSS,
        )


@pytest.mark.parametrize(
    "invalid",
    (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity"), 0.25, "0.25"),
)
def test_a_metric_value_is_never_nonfinite_or_nondecimal(invalid):
    with pytest.raises(ValidationError):
        ResearchMetricValue(value=invalid)


def test_an_undefined_reason_must_be_the_exact_canonical_enum():
    with pytest.raises(ValidationError):
        ResearchMetricValue(undefined_reason="NO_COMPLETED_TRADES")


def test_monetary_metric_members_reject_nonfinite_and_nondecimal_values():
    for invalid in (Decimal("NaN"), Decimal("Infinity"), 110000.0, "110000"):
        with pytest.raises(ValidationError):
            _strategy_metrics(ending_equity=invalid)
    with pytest.raises(ValidationError):
        _trade_metrics(gross_loss=Decimal("-1"))
    with pytest.raises(ValidationError):
        _trade_metrics(win_count=-1)
    with pytest.raises(ValidationError):
        _trade_metrics(win_count=True)


def test_research_result_binds_required_provenance_and_is_immutable(
    synthetic_audit_result, performance_policy, benchmark_series
):
    provenance = ResearchMetricsProvenance(
        source_audit_result_fingerprint=(
            synthetic_audit_result.result_fingerprint
        ),
        performance_measurement_policy_fingerprint=(
            performance_policy.policy_fingerprint
        ),
        benchmark_series_fingerprint=benchmark_series.content_fingerprint,
    )
    result = build_research_metrics_result(
        provenance=provenance,
        strategy_metrics=_strategy_metrics(),
        trade_metrics=_trade_metrics(),
        benchmark_metrics=_benchmark_metrics(),
        relative_metrics=_relative_metrics(),
    )
    assert result.schema_version == "research_metrics_result.v0.3"
    assert result.provenance.source_audit_result_fingerprint == (
        synthetic_audit_result.result_fingerprint
    )
    with pytest.raises(ValidationError):
        result.result_fingerprint = "f" * 64
    with pytest.raises(ValidationError):
        type(result)(**result.model_dump(mode="python"), unexpected=True)


def test_benchmark_and_relative_groups_require_benchmark_provenance(
    synthetic_audit_result, performance_policy
):
    provenance = ResearchMetricsProvenance(
        source_audit_result_fingerprint=(
            synthetic_audit_result.result_fingerprint
        ),
        performance_measurement_policy_fingerprint=(
            performance_policy.policy_fingerprint
        ),
    )
    with pytest.raises(ValidationError):
        build_research_metrics_result(
            provenance=provenance,
            strategy_metrics=_strategy_metrics(),
            trade_metrics=_trade_metrics(),
            benchmark_metrics=_benchmark_metrics(),
            relative_metrics=_relative_metrics(),
        )
