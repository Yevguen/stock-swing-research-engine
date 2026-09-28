"""Domain-separated deterministic SHA-256 helpers for Phase 16B.1."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from stock_swing_d1.research_metrics.canonical import semantic_json_bytes

if TYPE_CHECKING:
    from stock_swing_d1.research_metrics.models import (
        BenchmarkPerformanceSeries,
        ResearchMetricsResult,
    )
    from stock_swing_d1.research_metrics.policy import (
        PerformanceMeasurementPolicy,
    )


# Domain string == schema version, uniformly across this package (Phase 16B.1
# Clauses 14/23). Phase 16B.2 v0.2.1 Clauses 43/48 advanced the policy domain to
# v0.3 and the result domain to v0.2; the Phase 16B Amendment v0.1 advances the
# result domain again, to v0.3, because the canonical result payload gained the
# four amendment trade metrics. A digest computed under an older domain and one
# computed under a newer domain never share an identity, so a v0.2 result
# identity can never silently stand for an amended payload.
PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN = (
    "performance_measurement_policy.v0.3"
)
BENCHMARK_PERFORMANCE_SERIES_HASH_DOMAIN = (
    "benchmark_performance_series.v0.2"
)
RESEARCH_METRICS_RESULT_HASH_DOMAIN = "research_metrics_result.v0.3"


def semantic_sha256(value: object) -> str:
    """Return lowercase SHA-256 over canonical semantic JSON bytes."""

    return hashlib.sha256(semantic_json_bytes(value)).hexdigest()


def semantic_domain_sha256(domain: str, payload: object) -> str:
    """Hash semantic content within an explicit, versioned domain."""

    if type(domain) is not str or not domain or domain != domain.strip():
        raise ValueError("domain must be canonical non-empty text")
    return semantic_sha256({"domain": domain, "payload": payload})


def _compute_performance_measurement_policy_fingerprint(
    *,
    schema_version: object,
    policy_id: object,
    policy_version: object,
    annualization_period_convention: object,
    annualization_periods_per_year: object,
    annual_risk_free_rate: object,
    sortino_minimum_acceptable_return: object,
    cagr_day_count_convention: object,
    undefined_metric_behavior: object,
    statistical_working_precision: object,
    statistical_rounding_mode: object,
    canonical_metric_quantum: object,
) -> str:
    return semantic_domain_sha256(
        PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN,
        {
            "schema_version": schema_version,
            "policy_id": policy_id,
            "policy_version": policy_version,
            "annualization_period_convention": (
                annualization_period_convention
            ),
            "annualization_periods_per_year": annualization_periods_per_year,
            "annual_risk_free_rate": annual_risk_free_rate,
            "sortino_minimum_acceptable_return": (
                sortino_minimum_acceptable_return
            ),
            "cagr_day_count_convention": cagr_day_count_convention,
            "undefined_metric_behavior": undefined_metric_behavior,
            "statistical_working_precision": statistical_working_precision,
            "statistical_rounding_mode": statistical_rounding_mode,
            "canonical_metric_quantum": canonical_metric_quantum,
        },
    )


def compute_performance_measurement_policy_fingerprint(
    policy: PerformanceMeasurementPolicy,
) -> str:
    """Fingerprint all semantic fields of one exact policy instance."""

    from stock_swing_d1.research_metrics.policy import (
        PerformanceMeasurementPolicy,
    )

    if type(policy) is not PerformanceMeasurementPolicy:
        raise TypeError("policy must be a PerformanceMeasurementPolicy")
    return _compute_performance_measurement_policy_fingerprint(
        schema_version=policy.schema_version,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        annualization_period_convention=(
            policy.annualization_period_convention
        ),
        annualization_periods_per_year=(
            policy.annualization_periods_per_year
        ),
        annual_risk_free_rate=policy.annual_risk_free_rate,
        sortino_minimum_acceptable_return=(
            policy.sortino_minimum_acceptable_return
        ),
        cagr_day_count_convention=policy.cagr_day_count_convention,
        undefined_metric_behavior=policy.undefined_metric_behavior,
        statistical_working_precision=policy.statistical_working_precision,
        statistical_rounding_mode=policy.statistical_rounding_mode,
        canonical_metric_quantum=policy.canonical_metric_quantum,
    )


def _compute_benchmark_performance_series_fingerprint(
    *,
    schema_version: object,
    benchmark_id: object,
    source_artifact_ref: object,
    observations: object,
) -> str:
    return semantic_domain_sha256(
        BENCHMARK_PERFORMANCE_SERIES_HASH_DOMAIN,
        {
            "schema_version": schema_version,
            "benchmark_id": benchmark_id,
            "source_artifact_ref": source_artifact_ref,
            "observations": observations,
        },
    )


def compute_benchmark_performance_series_fingerprint(
    series: BenchmarkPerformanceSeries,
) -> str:
    """Fingerprint provider-neutral benchmark identity and ordered evidence."""

    from stock_swing_d1.research_metrics.models import (
        BenchmarkPerformanceSeries,
    )

    if type(series) is not BenchmarkPerformanceSeries:
        raise TypeError("series must be a BenchmarkPerformanceSeries")
    return _compute_benchmark_performance_series_fingerprint(
        schema_version=series.schema_version,
        benchmark_id=series.benchmark_id,
        source_artifact_ref=series.source_artifact_ref,
        observations=series.observations,
    )


def _compute_research_metrics_result_fingerprint(
    *,
    schema_version: object,
    provenance: object,
    strategy_metrics: object,
    trade_metrics: object,
    benchmark_metrics: object,
    relative_metrics: object,
) -> str:
    return semantic_domain_sha256(
        RESEARCH_METRICS_RESULT_HASH_DOMAIN,
        {
            "schema_version": schema_version,
            "provenance": provenance,
            "strategy_metrics": strategy_metrics,
            "trade_metrics": trade_metrics,
            "benchmark_metrics": benchmark_metrics,
            "relative_metrics": relative_metrics,
        },
    )


def compute_research_metrics_result_fingerprint(
    result: ResearchMetricsResult,
) -> str:
    """Fingerprint all Phase 16B.1 result foundation fields."""

    from stock_swing_d1.research_metrics.models import ResearchMetricsResult

    if type(result) is not ResearchMetricsResult:
        raise TypeError("result must be a ResearchMetricsResult")
    return _compute_research_metrics_result_fingerprint(
        schema_version=result.schema_version,
        provenance=result.provenance,
        strategy_metrics=result.strategy_metrics,
        trade_metrics=result.trade_metrics,
        benchmark_metrics=result.benchmark_metrics,
        relative_metrics=result.relative_metrics,
    )


__all__ = [
    "BENCHMARK_PERFORMANCE_SERIES_HASH_DOMAIN",
    "PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN",
    "RESEARCH_METRICS_RESULT_HASH_DOMAIN",
    "compute_benchmark_performance_series_fingerprint",
    "compute_performance_measurement_policy_fingerprint",
    "compute_research_metrics_result_fingerprint",
    "semantic_domain_sha256",
    "semantic_sha256",
]
