"""Machine-readable report projection of a Phase 16B.3 artifact."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from stock_swing_d1.backtester.decision_interval import HistoricalDecisionInterval
from stock_swing_d1.research_metrics.models import ResearchMetricValue
from stock_swing_d1.research_metrics.persistence import (
    ResearchMetricsPersistenceArtifact,
    ResearchMetricsPolicyRef,
)


RESEARCH_METRICS_REPORT_PROJECTION_SCHEMA_VERSION = (
    "research_metrics_report_projection.v0.1"
)


def _require_exact_type(expected_type: type):
    def validate(value: object) -> object:
        if type(value) is not expected_type:
            raise ValueError(f"must be an exact {expected_type.__name__}")
        return value

    return validate


_DecisionInterval = Annotated[
    HistoricalDecisionInterval,
    BeforeValidator(_require_exact_type(HistoricalDecisionInterval)),
]
_Decimal = Annotated[Decimal, BeforeValidator(_require_exact_type(Decimal))]
_Count = Annotated[int, BeforeValidator(_require_exact_type(int))]
_Metric = Annotated[
    ResearchMetricValue,
    BeforeValidator(_require_exact_type(ResearchMetricValue)),
]
_PolicyRef = Annotated[
    ResearchMetricsPolicyRef,
    BeforeValidator(_require_exact_type(ResearchMetricsPolicyRef)),
]
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]


class _ImmutableReportModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_default=True,
    )


class ResearchMetricsReportContext(_ImmutableReportModel):
    decision_interval: _DecisionInterval
    starting_capital: _Decimal
    equity_observation_count: _Count
    periodic_return_observation_count: _Count
    policy_ref: _PolicyRef
    source_audit_result_fingerprint: _Sha256
    benchmark_series_fingerprint: _Sha256 | None
    research_metrics_result_fingerprint: _Sha256


class ResearchMetricsReportStrategy(_ImmutableReportModel):
    ending_equity: _Decimal
    total_return: _Metric
    cagr: _Metric
    annualized_volatility: _Metric
    sharpe_ratio: _Metric
    sortino_ratio: _Metric
    maximum_drawdown: _Metric


class ResearchMetricsReportBenchmark(_ImmutableReportModel):
    ending_equity: _Decimal
    total_return: _Metric
    cagr: _Metric
    annualized_volatility: _Metric
    sharpe_ratio: _Metric
    sortino_ratio: _Metric
    maximum_drawdown: _Metric


class ResearchMetricsReportComparative(_ImmutableReportModel):
    strategy_minus_benchmark_total_return: _Metric
    strategy_minus_benchmark_cagr: _Metric
    ending_wealth_ratio: _Metric


class ResearchMetricsReportProjection(_ImmutableReportModel):
    schema_version: Literal[
        "research_metrics_report_projection.v0.1"
    ] = RESEARCH_METRICS_REPORT_PROJECTION_SCHEMA_VERSION
    context: ResearchMetricsReportContext
    strategy: ResearchMetricsReportStrategy
    benchmark: ResearchMetricsReportBenchmark | None
    comparative: ResearchMetricsReportComparative | None


def project_research_metrics_report(
    artifact: ResearchMetricsPersistenceArtifact,
) -> ResearchMetricsReportProjection:
    """Copy authoritative fields into the stable report-facing shape."""

    if type(artifact) is not ResearchMetricsPersistenceArtifact:
        raise TypeError("artifact must be a ResearchMetricsPersistenceArtifact")
    artifact = ResearchMetricsPersistenceArtifact(
        **{
            name: getattr(artifact, name)
            for name in ResearchMetricsPersistenceArtifact.model_fields
        }
    )
    result = artifact.research_metrics_result
    strategy = result.strategy_metrics
    benchmark = result.benchmark_metrics
    comparative = result.relative_metrics
    return ResearchMetricsReportProjection(
        context=ResearchMetricsReportContext(
            decision_interval=artifact.decision_interval,
            starting_capital=artifact.starting_capital,
            equity_observation_count=artifact.equity_observation_count,
            periodic_return_observation_count=(
                artifact.periodic_return_observation_count
            ),
            policy_ref=artifact.policy_ref,
            source_audit_result_fingerprint=(
                result.provenance.source_audit_result_fingerprint
            ),
            benchmark_series_fingerprint=(
                result.provenance.benchmark_series_fingerprint
            ),
            research_metrics_result_fingerprint=result.result_fingerprint,
        ),
        strategy=ResearchMetricsReportStrategy(
            ending_equity=strategy.ending_equity,
            total_return=strategy.total_return,
            cagr=strategy.cagr,
            annualized_volatility=strategy.annualized_volatility,
            sharpe_ratio=strategy.sharpe_ratio,
            sortino_ratio=strategy.sortino_ratio,
            maximum_drawdown=strategy.maximum_drawdown,
        ),
        benchmark=(
            None
            if benchmark is None
            else ResearchMetricsReportBenchmark(
                ending_equity=benchmark.ending_equity,
                total_return=benchmark.total_return,
                cagr=benchmark.cagr,
                annualized_volatility=benchmark.annualized_volatility,
                sharpe_ratio=benchmark.sharpe_ratio,
                sortino_ratio=benchmark.sortino_ratio,
                maximum_drawdown=benchmark.maximum_drawdown,
            )
        ),
        comparative=(
            None
            if comparative is None
            else ResearchMetricsReportComparative(
                strategy_minus_benchmark_total_return=(
                    comparative.strategy_minus_benchmark_total_return
                ),
                strategy_minus_benchmark_cagr=(
                    comparative.strategy_minus_benchmark_cagr
                ),
                ending_wealth_ratio=comparative.ending_wealth_ratio,
            )
        ),
    )


__all__ = [
    "RESEARCH_METRICS_REPORT_PROJECTION_SCHEMA_VERSION",
    "ResearchMetricsReportBenchmark",
    "ResearchMetricsReportComparative",
    "ResearchMetricsReportContext",
    "ResearchMetricsReportProjection",
    "ResearchMetricsReportStrategy",
    "project_research_metrics_report",
]
