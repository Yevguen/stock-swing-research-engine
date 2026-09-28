"""Immutable Phase 16B.1 evidence and result-model foundations."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from stock_swing_d1.research_metrics.hashing import (
    _compute_benchmark_performance_series_fingerprint,
    _compute_research_metrics_result_fingerprint,
    compute_benchmark_performance_series_fingerprint,
    compute_research_metrics_result_fingerprint,
)
from stock_swing_d1.research_metrics.policy import UndefinedMetricReason


BENCHMARK_PERFORMANCE_SERIES_SCHEMA_VERSION = (
    "benchmark_performance_series.v0.2"
)
RESEARCH_METRIC_VALUE_SCHEMA_VERSION = "research_metric_value.v0.2"
STRATEGY_PERFORMANCE_METRICS_SCHEMA_VERSION = (
    "strategy_performance_metrics.v0.2"
)
TRADE_PERFORMANCE_METRICS_SCHEMA_VERSION = "trade_performance_metrics.v0.3"
BENCHMARK_PERFORMANCE_METRICS_SCHEMA_VERSION = (
    "benchmark_performance_metrics.v0.2"
)
RELATIVE_PERFORMANCE_METRICS_SCHEMA_VERSION = (
    "relative_performance_metrics.v0.2"
)
RESEARCH_METRICS_RESULT_SCHEMA_VERSION = "research_metrics_result.v0.3"


def _require_canonical_text(value: object) -> object:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("must be canonical non-empty text")
    return value


def _require_optional_canonical_text(value: object) -> object:
    if value is None:
        return value
    return _require_canonical_text(value)


def _require_session_date(value: object) -> object:
    if type(value) is not date:
        raise ValueError("must be a Python datetime.date")
    return value


def _require_positive_decimal(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError("must be a Decimal; floats, strings, and bool are forbidden")
    if not value.is_finite() or value <= Decimal("0"):
        raise ValueError("must be a finite positive Decimal")
    return value


def _require_finite_decimal(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError("must be a Decimal; floats, strings, and bool are forbidden")
    if not value.is_finite():
        raise ValueError("must be a finite Decimal; NaN and infinity are forbidden")
    return value


def _require_optional_finite_decimal(value: object) -> object:
    if value is None:
        return value
    return _require_finite_decimal(value)


def _require_non_negative_decimal(value: object) -> object:
    _require_finite_decimal(value)
    if value < Decimal("0"):
        raise ValueError("must be a non-negative Decimal")
    return value


def _require_exact_count(value: object) -> object:
    if type(value) is not int:
        raise ValueError("must be an exact int; bool and float are forbidden")
    if value < 0:
        raise ValueError("must be a non-negative count")
    return value


def _require_optional_undefined_reason(value: object) -> object:
    if value is None or type(value) is UndefinedMetricReason:
        return value
    raise ValueError("must be an exact UndefinedMetricReason value")


def _require_tuple(value: object) -> object:
    if type(value) is not tuple:
        raise ValueError("must be an immutable tuple")
    return value


class _ImmutableResearchMetricsModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_default=True,
    )


_CanonicalText = Annotated[str, BeforeValidator(_require_canonical_text)]
_OptionalCanonicalText = Annotated[
    str | None, BeforeValidator(_require_optional_canonical_text)
]
_SessionDate = Annotated[date, BeforeValidator(_require_session_date)]
_PositiveDecimal = Annotated[
    Decimal, BeforeValidator(_require_positive_decimal)
]
_FiniteDecimal = Annotated[Decimal, BeforeValidator(_require_finite_decimal)]
_OptionalFiniteDecimal = Annotated[
    Decimal | None, BeforeValidator(_require_optional_finite_decimal)
]
_NonNegativeDecimal = Annotated[
    Decimal, BeforeValidator(_require_non_negative_decimal)
]
_NonNegativeCount = Annotated[int, BeforeValidator(_require_exact_count)]
_OptionalUndefinedMetricReason = Annotated[
    UndefinedMetricReason | None,
    BeforeValidator(_require_optional_undefined_reason),
]
_Sha256 = Annotated[
    str,
    BeforeValidator(_require_canonical_text),
    Field(pattern=r"^[0-9a-f]{64}$", strict=True),
]


class BenchmarkSourceArtifactRef(_ImmutableResearchMetricsModel):
    """Provider-neutral semantic identity of one source artifact."""

    artifact_id: _CanonicalText
    schema_version: _CanonicalText
    content_sha256: _Sha256
    build_id: _OptionalCanonicalText = None


class BenchmarkPerformanceObservation(_ImmutableResearchMetricsModel):
    """One completed benchmark session of authoritative primary evidence.

    ``value`` is the authoritative primary upstream ``benchmark_equity`` for
    that completed session, denominated in the experiment base currency (USD
    for the current baseline). Its economic construction — residual cash,
    whole-share holdings, distributions, and applicable corporate actions — has
    already occurred upstream and is never rederived here from raw prices.
    """

    session: _SessionDate
    value: _PositiveDecimal


class BenchmarkPerformanceSeries(_ImmutableResearchMetricsModel):
    """Ordered evidence of the primary authoritative benchmark-equity path.

    The model is contractually scoped to the primary authoritative investable
    benchmark and carries no value-kind discriminator: its scope alone
    establishes the sole legal semantic. It stays structurally generic through
    ``benchmark_id``; requiring the authoritative baseline benchmark identity
    belongs to the applicable experiment/validation policy layer, not here.
    """

    schema_version: Literal[
        "benchmark_performance_series.v0.2"
    ] = BENCHMARK_PERFORMANCE_SERIES_SCHEMA_VERSION
    benchmark_id: _CanonicalText
    source_artifact_ref: BenchmarkSourceArtifactRef
    observations: Annotated[
        tuple[BenchmarkPerformanceObservation, ...],
        BeforeValidator(_require_tuple),
    ]
    content_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_series(self) -> Self:
        sessions = tuple(row.session for row in self.observations)
        if sessions != tuple(sorted(sessions)):
            raise ValueError("benchmark observations must be ordered by session")
        if len(sessions) != len(set(sessions)):
            raise ValueError("benchmark observation sessions must be unique")
        expected = compute_benchmark_performance_series_fingerprint(self)
        if self.content_fingerprint != expected:
            raise ValueError(
                "content_fingerprint does not match benchmark series content"
            )
        return self


def build_benchmark_performance_series(
    *,
    benchmark_id: str,
    source_artifact_ref: BenchmarkSourceArtifactRef,
    observations: tuple[BenchmarkPerformanceObservation, ...],
) -> BenchmarkPerformanceSeries:
    """Build without sorting or otherwise repairing caller-supplied evidence."""

    values = {
        "schema_version": BENCHMARK_PERFORMANCE_SERIES_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "source_artifact_ref": source_artifact_ref,
        "observations": observations,
    }
    fingerprint = _compute_benchmark_performance_series_fingerprint(**values)
    return BenchmarkPerformanceSeries(
        **values, content_fingerprint=fingerprint
    )


class ResearchMetricValue(_ImmutableResearchMetricsModel):
    """One canonical derived metric: a value, or exactly one undefined reason.

    Phase 16B.2 v0.2.1 Clauses 39.2/40: a mathematically undefined metric
    carries ``value = None`` plus a canonical reason and is never encoded as
    zero, NaN, infinity, or an omitted field. A defined metric carries a
    finite Decimal and no reason. The two are mutually exclusive, so an
    undefined metric can never smuggle a placeholder number through.
    """

    schema_version: Literal[
        "research_metric_value.v0.2"
    ] = RESEARCH_METRIC_VALUE_SCHEMA_VERSION
    value: _OptionalFiniteDecimal = None
    undefined_reason: _OptionalUndefinedMetricReason = None

    @model_validator(mode="after")
    def validate_definedness(self) -> Self:
        if (self.value is None) == (self.undefined_reason is None):
            raise ValueError(
                "a metric is either defined with a finite value and no reason, "
                "or undefined with exactly one canonical reason and no value"
            )
        return self


def defined_metric(value: Decimal) -> ResearchMetricValue:
    """Build one defined canonical metric value."""

    return ResearchMetricValue(value=value)


def undefined_metric(reason: UndefinedMetricReason) -> ResearchMetricValue:
    """Build one explicitly undefined canonical metric value."""

    return ResearchMetricValue(undefined_reason=reason)


class StrategyPerformanceMetrics(_ImmutableResearchMetricsModel):
    """The frozen Phase 16A strategy equity-path metric inventory.

    ``ending_equity`` and ``net_pnl`` are authoritative exact monetary facts
    measured from the audited run, not derived statistics, so they carry no
    canonical statistical quantum. Every other member is a derived statistic
    quantized at its final metric boundary.
    """

    schema_version: Literal[
        "strategy_performance_metrics.v0.2"
    ] = STRATEGY_PERFORMANCE_METRICS_SCHEMA_VERSION
    ending_equity: _FiniteDecimal
    net_pnl: _FiniteDecimal
    total_return: ResearchMetricValue
    cagr: ResearchMetricValue
    annualized_volatility: ResearchMetricValue
    maximum_drawdown: ResearchMetricValue
    sharpe_ratio: ResearchMetricValue
    sortino_ratio: ResearchMetricValue


class TradePerformanceMetrics(_ImmutableResearchMetricsModel):
    """The frozen Phase 16A completed-trade metric inventory.

    Every profit/loss-dependent member is measured from the authoritative
    dividend-inclusive total P&L of a completed trade episode (Phase 16B.2
    v0.2 Clause 16), never from trading-only realized P&L, and never by
    re-deriving trade economics here.

    The Phase 16B Amendment v0.1 adds four measurement/report-completeness
    members and advances this inventory to v0.3. All four are USD amounts or
    session counts, never percentage returns:

    * ``average_winner_usd`` -- mean total P&L over winning episodes only;
    * ``average_loser_usd`` -- mean total P&L over losing episodes only, kept
      signed negative rather than restated as an absolute loss magnitude;
    * ``worst_trade_usd`` -- the minimum *signed* total P&L over every
      completed episode, so an all-winning population still has a defined
      worst trade (its smallest winner) rather than an undefined one;
    * ``average_holding_sessions`` -- mean holding length counted in regular
      trading sessions, never in calendar days or wall-clock duration.
    """

    schema_version: Literal[
        "trade_performance_metrics.v0.3"
    ] = TRADE_PERFORMANCE_METRICS_SCHEMA_VERSION
    completed_trade_count: _NonNegativeCount
    win_count: _NonNegativeCount
    loss_count: _NonNegativeCount
    breakeven_count: _NonNegativeCount
    gross_profit: _NonNegativeDecimal
    gross_loss: _NonNegativeDecimal
    win_rate: ResearchMetricValue
    loss_rate: ResearchMetricValue
    breakeven_rate: ResearchMetricValue
    profit_factor: ResearchMetricValue
    expectancy: ResearchMetricValue
    average_winner_usd: ResearchMetricValue
    average_loser_usd: ResearchMetricValue
    worst_trade_usd: ResearchMetricValue
    average_holding_sessions: ResearchMetricValue

    @model_validator(mode="after")
    def validate_amendment_metric_domains(self) -> Self:
        """Reject a defined amendment metric that contradicts its definition.

        These are declarative domain checks on an already-computed value, not
        a second calculation: a winner average that is not strictly positive,
        a loser average that lost its negative sign, or a holding average
        shorter than the one session every completed episode necessarily
        occupies, each means the value was produced under semantics this model
        does not describe, so it is refused rather than published.
        """

        winner = self.average_winner_usd.value
        if winner is not None and winner <= Decimal("0"):
            raise ValueError(
                "a defined average winning trade must be strictly positive"
            )
        loser = self.average_loser_usd.value
        if loser is not None and loser >= Decimal("0"):
            raise ValueError(
                "a defined average losing trade must stay signed negative and "
                "is never restated as an absolute loss magnitude"
            )
        held = self.average_holding_sessions.value
        if held is not None and held < Decimal("1"):
            raise ValueError(
                "a defined average holding period must be at least the one "
                "regular trading session every completed episode occupies"
            )
        return self


class BenchmarkPerformanceMetrics(_ImmutableResearchMetricsModel):
    """The frozen Phase 16A benchmark equity-path metric inventory.

    Measured from the supplied authoritative benchmark-equity evidence under
    exactly the strategy methodology (Phase 16B.2 v0.2 Clause 20).
    """

    schema_version: Literal[
        "benchmark_performance_metrics.v0.2"
    ] = BENCHMARK_PERFORMANCE_METRICS_SCHEMA_VERSION
    ending_equity: _FiniteDecimal
    total_return: ResearchMetricValue
    cagr: ResearchMetricValue
    annualized_volatility: ResearchMetricValue
    maximum_drawdown: ResearchMetricValue
    sharpe_ratio: ResearchMetricValue
    sortino_ratio: ResearchMetricValue


class RelativePerformanceMetrics(_ImmutableResearchMetricsModel):
    """The three frozen Phase 16A strategy-versus-benchmark comparisons."""

    schema_version: Literal[
        "relative_performance_metrics.v0.2"
    ] = RELATIVE_PERFORMANCE_METRICS_SCHEMA_VERSION
    strategy_minus_benchmark_total_return: ResearchMetricValue
    strategy_minus_benchmark_cagr: ResearchMetricValue
    ending_wealth_ratio: ResearchMetricValue


class ResearchMetricsProvenance(_ImmutableResearchMetricsModel):
    """Exact upstream and policy identities bound into a metrics result."""

    schema_version: Literal[
        "research_metrics_provenance.v0.1"
    ] = "research_metrics_provenance.v0.1"
    source_audit_result_fingerprint: _Sha256
    performance_measurement_policy_fingerprint: _Sha256
    benchmark_series_fingerprint: _Sha256 | None = None


class ResearchMetricsResult(_ImmutableResearchMetricsModel):
    """One immutable, fingerprint-bound canonical research-metrics result."""

    schema_version: Literal[
        "research_metrics_result.v0.3"
    ] = RESEARCH_METRICS_RESULT_SCHEMA_VERSION
    provenance: ResearchMetricsProvenance
    strategy_metrics: StrategyPerformanceMetrics
    trade_metrics: TradePerformanceMetrics
    benchmark_metrics: BenchmarkPerformanceMetrics | None = None
    relative_metrics: RelativePerformanceMetrics | None = None
    result_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        benchmark_supplied = (
            self.provenance.benchmark_series_fingerprint is not None
        )
        if benchmark_supplied != (self.benchmark_metrics is not None):
            raise ValueError(
                "benchmark metrics presence must match benchmark provenance"
            )
        if benchmark_supplied != (self.relative_metrics is not None):
            raise ValueError(
                "relative metrics presence must match benchmark provenance"
            )
        expected = compute_research_metrics_result_fingerprint(self)
        if self.result_fingerprint != expected:
            raise ValueError("result_fingerprint does not match result content")
        return self


def build_research_metrics_result(
    *,
    provenance: ResearchMetricsProvenance,
    strategy_metrics: StrategyPerformanceMetrics,
    trade_metrics: TradePerformanceMetrics,
    benchmark_metrics: BenchmarkPerformanceMetrics | None = None,
    relative_metrics: RelativePerformanceMetrics | None = None,
) -> ResearchMetricsResult:
    """Build one canonical research-metrics result and bind its fingerprint."""

    values = {
        "schema_version": RESEARCH_METRICS_RESULT_SCHEMA_VERSION,
        "provenance": provenance,
        "strategy_metrics": strategy_metrics,
        "trade_metrics": trade_metrics,
        "benchmark_metrics": benchmark_metrics,
        "relative_metrics": relative_metrics,
    }
    fingerprint = _compute_research_metrics_result_fingerprint(**values)
    return ResearchMetricsResult(**values, result_fingerprint=fingerprint)


__all__ = [
    "BENCHMARK_PERFORMANCE_METRICS_SCHEMA_VERSION",
    "BENCHMARK_PERFORMANCE_SERIES_SCHEMA_VERSION",
    "RELATIVE_PERFORMANCE_METRICS_SCHEMA_VERSION",
    "RESEARCH_METRICS_RESULT_SCHEMA_VERSION",
    "RESEARCH_METRIC_VALUE_SCHEMA_VERSION",
    "STRATEGY_PERFORMANCE_METRICS_SCHEMA_VERSION",
    "TRADE_PERFORMANCE_METRICS_SCHEMA_VERSION",
    "BenchmarkPerformanceMetrics",
    "BenchmarkPerformanceObservation",
    "BenchmarkPerformanceSeries",
    "BenchmarkSourceArtifactRef",
    "RelativePerformanceMetrics",
    "ResearchMetricValue",
    "ResearchMetricsProvenance",
    "ResearchMetricsResult",
    "StrategyPerformanceMetrics",
    "TradePerformanceMetrics",
    "build_benchmark_performance_series",
    "build_research_metrics_result",
    "defined_metric",
    "undefined_metric",
]
