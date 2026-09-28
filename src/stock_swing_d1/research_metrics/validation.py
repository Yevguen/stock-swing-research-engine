"""Fail-closed Phase 15D-to-16B evidence validation boundary."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from stock_swing_d1.backtest_results import HistoricalBacktestAuditResult
from stock_swing_d1.research_metrics.errors import (
    ResearchMetricsValidationError,
)
from stock_swing_d1.research_metrics.hashing import (
    compute_benchmark_performance_series_fingerprint,
    compute_performance_measurement_policy_fingerprint,
)
from stock_swing_d1.research_metrics.models import BenchmarkPerformanceSeries
from stock_swing_d1.research_metrics.policy import PerformanceMeasurementPolicy


_REQUIRED_AUDIT_FLAGS = (
    "source_run_canonical",
    "state_chain_valid",
    "ledger_reconstruction_valid",
    "execution_ledger_provenance_valid",
    "signal_provenance_valid",
    "ranking_provenance_valid",
    "allocation_provenance_valid",
    "execution_provenance_valid",
    "trade_linkage_valid",
    "valuation_coverage_valid",
    "equity_reconciliation_valid",
    "pnl_reconciliation_valid",
    "cost_reconciliation_valid",
    "policy_consistency_valid",
    "artifact_fingerprints_valid",
    "audit_passed",
)


def _rebuild_model_value(value: object) -> object:
    if isinstance(value, BaseModel):
        values = {
            field_name: _rebuild_model_value(getattr(value, field_name))
            for field_name in type(value).model_fields
        }
        return type(value)(**values)
    if type(value) is tuple:
        return tuple(_rebuild_model_value(item) for item in value)
    return value


def _canonical_rebuild(model_type: type, value: object, *, code: str):
    try:
        rebuilt = _rebuild_model_value(value)
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise ResearchMetricsValidationError(
            code, "input fails canonical model validation"
        ) from error
    if type(rebuilt) is not model_type:
        raise ResearchMetricsValidationError(
            code, "input is not the required exact model type"
        )
    if rebuilt != value:
        raise ResearchMetricsValidationError(
            code, "input is not the canonical model representation"
        )
    return rebuilt


def validate_research_metrics_inputs(
    *,
    source_result: HistoricalBacktestAuditResult,
    policy: PerformanceMeasurementPolicy,
    benchmark_series: BenchmarkPerformanceSeries | None = None,
) -> None:
    """Validate authoritative evidence without repairing or deriving it."""

    if type(source_result) is not HistoricalBacktestAuditResult:
        raise ResearchMetricsValidationError(
            "INVALID_SOURCE_RESULT",
            "source_result must be an exact HistoricalBacktestAuditResult",
        )
    canonical_source = _canonical_rebuild(
        HistoricalBacktestAuditResult,
        source_result,
        code="INVALID_SOURCE_RESULT",
    )
    if any(
        getattr(canonical_source.audit_summary, field_name) is not True
        for field_name in _REQUIRED_AUDIT_FLAGS
    ):
        raise ResearchMetricsValidationError(
            "SOURCE_AUDIT_NOT_PASSED",
            "every required Phase 15D audit-validity flag must be true",
        )

    if type(policy) is not PerformanceMeasurementPolicy:
        raise ResearchMetricsValidationError(
            "INVALID_POLICY",
            "policy must be an exact PerformanceMeasurementPolicy",
        )
    canonical_policy = _canonical_rebuild(
        PerformanceMeasurementPolicy, policy, code="INVALID_POLICY"
    )
    if (
        canonical_policy.policy_fingerprint
        != compute_performance_measurement_policy_fingerprint(canonical_policy)
    ):
        raise ResearchMetricsValidationError(
            "POLICY_FINGERPRINT_MISMATCH",
            "policy fingerprint does not match policy content",
        )

    if benchmark_series is None:
        return
    if type(benchmark_series) is not BenchmarkPerformanceSeries:
        raise ResearchMetricsValidationError(
            "INVALID_BENCHMARK_SERIES",
            "benchmark_series must be an exact BenchmarkPerformanceSeries",
        )
    canonical_benchmark = _canonical_rebuild(
        BenchmarkPerformanceSeries,
        benchmark_series,
        code="INVALID_BENCHMARK_SERIES",
    )
    if (
        canonical_benchmark.content_fingerprint
        != compute_benchmark_performance_series_fingerprint(
            canonical_benchmark
        )
    ):
        raise ResearchMetricsValidationError(
            "BENCHMARK_FINGERPRINT_MISMATCH",
            "benchmark fingerprint does not match benchmark content",
        )


__all__ = ["validate_research_metrics_inputs"]
