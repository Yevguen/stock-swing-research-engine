"""Deterministic Phase 16B.2 research-metric calculation.

Phase 16B measures authoritative economics. It does not reconstruct them.

Every economic fact consumed here -- experiment starting wealth, terminal
wealth, the session-equity path, the authoritative processed-session sequence,
benchmark equity, dividend-inclusive completed-trade P&L, and the entry/exit
sessions that place a completed episode inside that processed-session sequence
-- arrives already final from the accepted Phase 15D audited result or the
supplied authoritative benchmark evidence. This module reads those facts and
computes statistics from them. It never re-derives, repairs, re-prices, joins
away, back-fills or interpolates one of them.

Three outcomes are kept strictly apart (Clause 39):

* invalid canonical input -- no result is published at all;
* a mathematically undefined metric -- ``value = None`` plus one canonical
  reason, leaving independently defined metrics valid;
* a calculation/determinism failure -- a typed error, never disguised as an
  undefined metric.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import NamedTuple

from stock_swing_d1.backtest_results import (
    HistoricalBacktestAuditResult,
    HistoricalClosedTradeRecord,
)
from stock_swing_d1.research_metrics.arithmetic import (
    add,
    canonical_statistical_context,
    divide,
    exponential,
    multiply,
    natural_logarithm,
    quantize_canonical_metric,
    square_root,
    subtract,
    subtract_exact_decimal,
    sum_exact_decimal,
)
from stock_swing_d1.research_metrics.errors import (
    ResearchMetricsCalculationError,
    ResearchMetricsValidationError,
)
from stock_swing_d1.research_metrics.models import (
    BenchmarkPerformanceMetrics,
    BenchmarkPerformanceSeries,
    RelativePerformanceMetrics,
    ResearchMetricValue,
    ResearchMetricsProvenance,
    ResearchMetricsResult,
    StrategyPerformanceMetrics,
    TradePerformanceMetrics,
    build_research_metrics_result,
    defined_metric,
    undefined_metric,
)
from stock_swing_d1.research_metrics.policy import (
    PerformanceMeasurementPolicy,
    UndefinedMetricReason,
)
from stock_swing_d1.research_metrics.validation import (
    validate_research_metrics_inputs,
)


_ZERO = Decimal("0")
_ONE = Decimal("1")

# Gate 2 / Phase 16A §7: the frozen baseline experiment begins with exactly USD
# 100,000 of settled cash. Numeric equality governs, so Decimal("100000") and
# Decimal("100000.00") are the same anchor; Decimal scale is not semantic here.
_CANONICAL_EXPERIMENT_WEALTH_ANCHOR = Decimal("100000")

# 365.25 == 1461/4 exactly, so the frozen ACTUAL_365_25 convention is applied
# through exact integers rather than a repeating Decimal literal.
_DAY_COUNT_NUMERATOR = 1461
_DAY_COUNT_DENOMINATOR = 4

# Phase 16A §8 requires two return observations for a sample standard
# deviation; null-first semantics therefore require three genuine sessions.
_MINIMUM_DISPERSION_OBSERVATIONS = 2
_MINIMUM_LOCATION_OBSERVATIONS = 1

# For E_T == 0 with E_0 > 0 and D > 0 the compound growth rate is exactly -1
# (total loss). It is a defined value, so ln(0) is never evaluated.
_TERMINAL_LOSS_GROWTH_RATE = Decimal("-1")


class _CanonicalEndpoints(NamedTuple):
    """The validated Phase 16B.2 measurement boundary of one experiment."""

    wealth_anchor: Decimal
    terminal_wealth: Decimal
    sessions: tuple[date, ...]
    wealth_path: tuple[Decimal, ...]
    elapsed_days: int


class _CompletedTradeEvidence(NamedTuple):
    """One completed episode's authoritative P&L and holding length.

    Both members are read from the accepted audited result: the P&L from the
    episode's own authoritative dividend-inclusive total, and the holding
    length by locating the episode's entry and exit sessions inside the
    authoritative ordered processed-session chronology. Neither is derived
    from prices, fills, portfolio state or a calendar.
    """

    total_pnl: Decimal
    holding_sessions: int


class _PathStatistics(NamedTuple):
    """The canonical wealth-path statistics shared by strategy and benchmark."""

    total_return: Decimal
    growth_rate: Decimal
    annualized_volatility: ResearchMetricValue
    maximum_drawdown: ResearchMetricValue
    sharpe_ratio: ResearchMetricValue
    sortino_ratio: ResearchMetricValue


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ResearchMetricsValidationError(code, message)


def _validate_canonical_endpoints(
    source_result: HistoricalBacktestAuditResult,
) -> _CanonicalEndpoints:
    """Validate the authoritative measurement boundary without rebuilding it.

    Both wealth anchors and the whole session sequence are read from the
    accepted audited result. Phase 16B never recomputes them, never appends a
    synthetic observation for a non-trading calendar boundary, and never
    reconstructs the processed-session sequence from an exchange calendar --
    the audited result's own ordered session-transition evidence is the
    authority (Clause 7).
    """

    anchor = source_result.initial_equity
    _require(
        anchor > _ZERO,
        "NONPOSITIVE_EXPERIMENT_WEALTH_ANCHOR",
        "experiment starting wealth must be strictly positive",
    )
    _require(
        anchor == _CANONICAL_EXPERIMENT_WEALTH_ANCHOR,
        "NONCANONICAL_EXPERIMENT_WEALTH_ANCHOR",
        "the frozen baseline requires exactly USD 100,000 starting wealth",
    )

    interval = source_result.decision_interval
    elapsed_days = (
        interval.decision_end_date - interval.decision_start_date
    ).days
    _require(
        elapsed_days > 0,
        "NONPOSITIVE_DECISION_DURATION",
        "canonical measurement requires a positive elapsed calendar duration",
    )

    rows = source_result.equity_curve
    _require(
        len(rows) > 0,
        "EMPTY_STRATEGY_EQUITY_EVIDENCE",
        "a zero-session run is not a valid canonical performance experiment",
    )
    _require(
        len(rows) == source_result.summary.processed_session_count,
        "INCOMPLETE_STRATEGY_SESSION_COVERAGE",
        "equity observations must cover every authoritative processed session",
    )

    observed = tuple(row.session for row in rows)
    authoritative = tuple(
        transition.session for transition in source_result.session_transitions
    )
    _require(
        observed == authoritative,
        "STRATEGY_SESSION_SEQUENCE_MISMATCH",
        "equity sessions must equal the authoritative processed-session "
        "sequence exactly and in order",
    )
    _require(
        observed[-1] == interval.decision_end_date,
        "STRATEGY_TERMINAL_SESSION_MISMATCH",
        "the final genuine strategy session must be the decision end boundary",
    )

    return _CanonicalEndpoints(
        wealth_anchor=anchor,
        terminal_wealth=source_result.final_equity,
        sessions=observed,
        wealth_path=tuple(row.equity for row in rows),
        elapsed_days=elapsed_days,
    )


def _validate_benchmark_congruence(
    benchmark_series: BenchmarkPerformanceSeries,
    endpoints: _CanonicalEndpoints,
) -> tuple[Decimal, ...]:
    """Require exact ordered session congruence, never a join or a fill."""

    observed = tuple(row.session for row in benchmark_series.observations)
    _require(
        observed == endpoints.sessions,
        "BENCHMARK_SESSION_INCONGRUENCE",
        "benchmark sessions must equal the strategy session sequence exactly "
        "and in order; no join, fill, interpolation or nearest-date match",
    )
    return tuple(row.value for row in benchmark_series.observations)


def _periodic_returns(
    wealth_path: tuple[Decimal, ...], *, context
) -> tuple[Decimal, ...]:
    """Build the canonical null-first periodic-return vector.

    A periodic return exists only between two genuine consecutive authoritative
    session observations, so N genuine sessions yield exactly N-1 returns
    (Clauses 13-16). The experiment wealth anchor is a boundary wealth level,
    not a prior session observation: it never enters this recursion, and no
    zero is inserted for the first observation.
    """

    returns: list[Decimal] = []
    previous = wealth_path[0]
    for current in wealth_path[1:]:
        if previous == _ZERO:
            raise ResearchMetricsCalculationError(
                "UNDEFINED_PERIODIC_RETURN",
                "a periodic return from zero prior wealth has no canonical value",
            )
        returns.append(
            subtract(divide(current, previous, context=context), _ONE, context=context)
        )
        previous = current
    return tuple(returns)


def _mean(values: tuple[Decimal, ...], *, context) -> Decimal:
    total = _ZERO
    for value in values:
        total = add(total, value, context=context)
    return divide(total, Decimal(len(values)), context=context)


def _sample_standard_deviation(
    values: tuple[Decimal, ...], location: Decimal, *, context
) -> Decimal:
    """Sample standard deviation with the frozen ddof = 1 divisor."""

    total = _ZERO
    for value in values:
        deviation = subtract(value, location, context=context)
        total = add(
            total, multiply(deviation, deviation, context=context), context=context
        )
    return square_root(
        divide(total, Decimal(len(values) - 1), context=context), context=context
    )


def _downside_deviation(
    values: tuple[Decimal, ...], minimum_acceptable_return: Decimal, *, context
) -> Decimal:
    """Root second lower partial moment about the MAR, over all N observations.

    The complete-observation denominator is intentional (Phase 16A Amendment 1):
    this measures a target downside deviation, not a sample standard deviation
    of the negative observations, so there is deliberately no N-1 correction and
    observations whose shortfall is zero still count in the denominator.
    """

    total = _ZERO
    for value in values:
        shortfall = subtract(value, minimum_acceptable_return, context=context)
        if shortfall >= _ZERO:
            continue
        total = add(
            total, multiply(shortfall, shortfall, context=context), context=context
        )
    return square_root(
        divide(total, Decimal(len(values)), context=context), context=context
    )


def _total_return(
    anchor: Decimal, terminal: Decimal, *, context
) -> Decimal:
    """Total return against the experiment wealth anchor, not the first session."""

    return quantize_canonical_metric(
        subtract(divide(terminal, anchor, context=context), _ONE, context=context),
        context=context,
    )


def _compound_growth_rate(
    anchor: Decimal, terminal: Decimal, elapsed_days: int, *, context
) -> Decimal:
    """Compute the canonical calendar-time compound growth rate.

    This is the sole normative computational algorithm (Clause 35). For a
    positive terminal wealth it evaluates

        exp( ln(E_T / E_0) * 1461 / (4 * D) ) - 1

    with Decimal ``ln``/``exp`` under the canonical context; no fractional-power
    pathway is used, because an alternative exponentiation route could change
    the canonical last digits. ``D`` is exact elapsed calendar days with no
    inclusive ``+1``: closed-interval membership describes which sessions belong
    to the experiment, not how long it lasted.
    """

    if terminal == _ZERO:
        return _TERMINAL_LOSS_GROWTH_RATE
    ratio = divide(terminal, anchor, context=context)
    if ratio <= _ZERO:
        raise ResearchMetricsCalculationError(
            "UNDEFINED_GROWTH_RATE_DOMAIN",
            "a nonpositive wealth ratio has no canonical growth rate",
        )
    scaled = divide(
        multiply(
            natural_logarithm(ratio, context=context),
            Decimal(_DAY_COUNT_NUMERATOR),
            context=context,
        ),
        Decimal(_DAY_COUNT_DENOMINATOR * elapsed_days),
        context=context,
    )
    return quantize_canonical_metric(
        subtract(exponential(scaled, context=context), _ONE, context=context),
        context=context,
    )


def _maximum_drawdown(
    anchor: Decimal, wealth_path: tuple[Decimal, ...], *, context
) -> ResearchMetricValue:
    """Maximum peak-to-trough decline as a nonnegative magnitude.

    Unlike the periodic-return recursion, the drawdown wealth path deliberately
    begins at the experiment wealth anchor (Clause 23), so a loss on the very
    first genuine session can create a drawdown from starting wealth.
    """

    peak = anchor
    largest = _ZERO
    for wealth in (anchor,) + wealth_path:
        if wealth > peak:
            peak = wealth
        if peak <= _ZERO:
            raise ResearchMetricsCalculationError(
                "UNDEFINED_DRAWDOWN_DENOMINATOR",
                "a nonpositive running peak has no canonical drawdown",
            )
        decline = divide(
            subtract_exact_decimal(peak, wealth), peak, context=context
        )
        if decline > largest:
            largest = decline
    return defined_metric(quantize_canonical_metric(largest, context=context))


def _path_statistics(
    *,
    anchor: Decimal,
    terminal: Decimal,
    wealth_path: tuple[Decimal, ...],
    returns: tuple[Decimal, ...],
    elapsed_days: int,
    policy: PerformanceMeasurementPolicy,
    context,
) -> _PathStatistics:
    """Compute one wealth path's canonical statistics.

    Strategy and benchmark share this one implementation, which is how the
    frozen methodological symmetry of Clause 20 is guaranteed rather than
    merely asserted.
    """

    annualizer = square_root(
        Decimal(policy.annualization_periods_per_year), context=context
    )
    observation_count = len(returns)

    if observation_count < _MINIMUM_LOCATION_OBSERVATIONS:
        location = None
    else:
        location = _mean(returns, context=context)

    if observation_count < _MINIMUM_DISPERSION_OBSERVATIONS:
        insufficient = undefined_metric(
            UndefinedMetricReason.INSUFFICIENT_RETURN_OBSERVATIONS
        )
        annualized_volatility = insufficient
        sharpe_ratio = insufficient
    else:
        dispersion = _sample_standard_deviation(
            returns, location, context=context
        )
        annualized_volatility = defined_metric(
            quantize_canonical_metric(
                multiply(dispersion, annualizer, context=context), context=context
            )
        )
        if dispersion == _ZERO:
            sharpe_ratio = undefined_metric(
                UndefinedMetricReason.ZERO_RETURN_VARIANCE
            )
        else:
            excess = subtract(
                location, policy.annual_risk_free_rate, context=context
            )
            sharpe_ratio = defined_metric(
                quantize_canonical_metric(
                    multiply(
                        divide(excess, dispersion, context=context),
                        annualizer,
                        context=context,
                    ),
                    context=context,
                )
            )

    if observation_count < _MINIMUM_LOCATION_OBSERVATIONS:
        sortino_ratio = undefined_metric(
            UndefinedMetricReason.INSUFFICIENT_RETURN_OBSERVATIONS
        )
    else:
        downside = _downside_deviation(
            returns, policy.sortino_minimum_acceptable_return, context=context
        )
        if downside == _ZERO:
            sortino_ratio = undefined_metric(
                UndefinedMetricReason.NO_DOWNSIDE_DEVIATION
            )
        else:
            excess = subtract(
                location, policy.sortino_minimum_acceptable_return, context=context
            )
            sortino_ratio = defined_metric(
                quantize_canonical_metric(
                    multiply(
                        divide(excess, downside, context=context),
                        annualizer,
                        context=context,
                    ),
                    context=context,
                )
            )

    return _PathStatistics(
        total_return=_total_return(anchor, terminal, context=context),
        growth_rate=_compound_growth_rate(
            anchor, terminal, elapsed_days, context=context
        ),
        annualized_volatility=annualized_volatility,
        maximum_drawdown=_maximum_drawdown(anchor, wealth_path, context=context),
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
    )


def _completed_trade_evidence(
    source_result: HistoricalBacktestAuditResult,
    sessions: tuple[date, ...],
) -> tuple[_CompletedTradeEvidence, ...]:
    """Read authoritative P&L and holding length for every completed episode.

    A trade still open at the decision end boundary is excluded from
    completed-trade statistics even though its marked value remains part of
    ending equity. A closed episode that does not publish an authoritative
    dividend-inclusive total -- a legacy non-dividend-aware record, or one
    whose pre-run dividend component is unknown -- is invalid canonical input
    for these metrics: Phase 16B may neither infer the missing attribution nor
    fall back to trading-only realized P&L (Clauses 8.4/18 of the parent
    contract).

    Holding length is counted in regular trading sessions. Phase 15D publishes
    an episode's entry and exit sessions but no explicit holding-session count,
    so the count is taken from the one authoritative ordered trading-session
    chronology that already governs the episode -- the validated processed-
    session sequence in ``sessions`` -- as the inclusive number of sessions
    from entry through exit. This is exactly the frozen upstream convention
    (``execution/open_position_exit``: the entry session is holding session 1),
    reused rather than reinvented. No calendar subtraction is performed and no
    exchange calendar is consulted, so a weekend or a market holiday between
    two sessions cannot inflate the count. An episode whose entry or exit
    session lies outside that authoritative sequence has no canonical holding
    length at all, and fails closed rather than receiving a guessed one.
    """

    ordinal = {session: position for position, session in enumerate(sessions)}
    evidence: list[_CompletedTradeEvidence] = []
    for record in source_result.trades:
        if type(record) is not HistoricalClosedTradeRecord:
            continue
        total = record.trade_total_pnl
        _require(
            total is not None,
            "MISSING_AUTHORITATIVE_TRADE_TOTAL_PNL",
            "a completed trade episode without authoritative "
            "dividend-inclusive total P&L cannot be measured",
        )
        opened = ordinal.get(record.entry_session)
        closed = ordinal.get(record.exit_session)
        _require(
            opened is not None and closed is not None,
            "TRADE_OUTSIDE_AUTHORITATIVE_SESSION_CHRONOLOGY",
            "a completed trade whose entry or exit session is absent from the "
            "authoritative processed-session sequence has no canonical holding "
            "length, and no calendar stand-in may be supplied for it",
        )
        _require(
            closed >= opened,
            "INVERTED_TRADE_SESSION_CHRONOLOGY",
            "a completed trade cannot exit before the session it entered on",
        )
        evidence.append(
            _CompletedTradeEvidence(
                total_pnl=total, holding_sessions=closed - opened + 1
            )
        )
    return tuple(evidence)


def _trade_metrics(
    source_result: HistoricalBacktestAuditResult,
    *,
    sessions: tuple[date, ...],
    context,
) -> TradePerformanceMetrics:
    """Classify and aggregate completed trades from authoritative evidence."""

    evidence = _completed_trade_evidence(source_result, sessions)
    totals = tuple(row.total_pnl for row in evidence)
    completed = len(totals)
    wins = tuple(total for total in totals if total > _ZERO)
    losses = tuple(total for total in totals if total < _ZERO)
    breakeven_count = completed - len(wins) - len(losses)

    gross_profit = sum_exact_decimal(wins)
    # The signed total of losing episodes. `gross_loss` below reports the same
    # population as a nonnegative magnitude for the frozen v0.2 metrics, but
    # `average_loser_usd` must stay signed negative, so the signed total is
    # kept rather than recovered by negating the magnitude a second time.
    net_loss_total = sum_exact_decimal(losses)
    gross_loss = subtract_exact_decimal(_ZERO, net_loss_total)

    if completed == 0:
        absent = undefined_metric(UndefinedMetricReason.NO_COMPLETED_TRADES)
        # With no completed episode at all, NO_COMPLETED_TRADES is the honest
        # root cause for the winner/loser averages too: there is no trade
        # population to have failed to produce a winner or a loser.
        return TradePerformanceMetrics(
            completed_trade_count=0,
            win_count=0,
            loss_count=0,
            breakeven_count=0,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            win_rate=absent,
            loss_rate=absent,
            breakeven_rate=absent,
            profit_factor=absent,
            expectancy=absent,
            average_winner_usd=absent,
            average_loser_usd=absent,
            worst_trade_usd=absent,
            average_holding_sessions=absent,
        )

    population = Decimal(completed)

    def rate(count: int) -> ResearchMetricValue:
        return defined_metric(
            quantize_canonical_metric(
                divide(Decimal(count), population, context=context), context=context
            )
        )

    def canonical(value: Decimal) -> ResearchMetricValue:
        return defined_metric(
            quantize_canonical_metric(value, context=context)
        )

    if gross_loss == _ZERO:
        profit_factor = undefined_metric(UndefinedMetricReason.ZERO_GROSS_LOSS)
    else:
        profit_factor = defined_metric(
            quantize_canonical_metric(
                divide(gross_profit, gross_loss, context=context), context=context
            )
        )

    if len(wins) == 0:
        average_winner = undefined_metric(
            UndefinedMetricReason.NO_WINNING_TRADES
        )
    else:
        average_winner = canonical(
            divide(gross_profit, Decimal(len(wins)), context=context)
        )

    if len(losses) == 0:
        average_loser = undefined_metric(UndefinedMetricReason.NO_LOSING_TRADES)
    else:
        average_loser = canonical(
            divide(net_loss_total, Decimal(len(losses)), context=context)
        )

    # The minimum *signed* total P&L over the whole population, breakeven and
    # winning episodes included -- not the largest absolute loss. Selection is
    # by strict comparison rather than by any reducer whose tie-breaking would
    # depend on input order, and the canonical quantum then fixes the scale, so
    # two numerically equal worst episodes cannot yield different results.
    worst = totals[0]
    for total in totals[1:]:
        if total < worst:
            worst = total

    holding_total = 0
    for row in evidence:
        holding_total += row.holding_sessions

    return TradePerformanceMetrics(
        completed_trade_count=completed,
        win_count=len(wins),
        loss_count=len(losses),
        breakeven_count=breakeven_count,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        win_rate=rate(len(wins)),
        loss_rate=rate(len(losses)),
        breakeven_rate=rate(breakeven_count),
        profit_factor=profit_factor,
        expectancy=defined_metric(
            quantize_canonical_metric(
                divide(sum_exact_decimal(totals), population, context=context),
                context=context,
            )
        ),
        average_winner_usd=average_winner,
        average_loser_usd=average_loser,
        worst_trade_usd=canonical(worst),
        average_holding_sessions=canonical(
            divide(Decimal(holding_total), population, context=context)
        ),
    )


def _relative_metrics(
    *,
    strategy: _PathStatistics,
    benchmark: _PathStatistics,
    strategy_terminal: Decimal,
    benchmark_terminal: Decimal,
    context,
) -> RelativePerformanceMetrics:
    """The three frozen strategy-versus-benchmark comparisons."""

    return RelativePerformanceMetrics(
        strategy_minus_benchmark_total_return=defined_metric(
            quantize_canonical_metric(
                subtract(
                    strategy.total_return, benchmark.total_return, context=context
                ),
                context=context,
            )
        ),
        strategy_minus_benchmark_cagr=defined_metric(
            quantize_canonical_metric(
                subtract(
                    strategy.growth_rate, benchmark.growth_rate, context=context
                ),
                context=context,
            )
        ),
        ending_wealth_ratio=defined_metric(
            quantize_canonical_metric(
                divide(strategy_terminal, benchmark_terminal, context=context),
                context=context,
            )
        ),
    )


def calculate_research_metrics(
    *,
    source_result: HistoricalBacktestAuditResult,
    policy: PerformanceMeasurementPolicy,
    benchmark_series: BenchmarkPerformanceSeries | None = None,
) -> ResearchMetricsResult:
    """Measure one audited experiment under the frozen canonical methodology.

    Returns an immutable, fingerprint-bound result. Raises
    ``ResearchMetricsValidationError`` for invalid canonical input -- in which
    case no result is published at all -- and
    ``ResearchMetricsCalculationError`` when canonical arithmetic cannot
    produce an otherwise defined metric.
    """

    validate_research_metrics_inputs(
        source_result=source_result,
        policy=policy,
        benchmark_series=benchmark_series,
    )

    context = canonical_statistical_context()
    endpoints = _validate_canonical_endpoints(source_result)

    strategy = _path_statistics(
        anchor=endpoints.wealth_anchor,
        terminal=endpoints.terminal_wealth,
        wealth_path=endpoints.wealth_path,
        returns=_periodic_returns(endpoints.wealth_path, context=context),
        elapsed_days=endpoints.elapsed_days,
        policy=policy,
        context=context,
    )
    strategy_metrics = StrategyPerformanceMetrics(
        ending_equity=endpoints.terminal_wealth,
        net_pnl=subtract_exact_decimal(
            endpoints.terminal_wealth, endpoints.wealth_anchor
        ),
        total_return=defined_metric(strategy.total_return),
        cagr=defined_metric(strategy.growth_rate),
        annualized_volatility=strategy.annualized_volatility,
        maximum_drawdown=strategy.maximum_drawdown,
        sharpe_ratio=strategy.sharpe_ratio,
        sortino_ratio=strategy.sortino_ratio,
    )

    benchmark_metrics = None
    relative_metrics = None
    if benchmark_series is not None:
        benchmark_path = _validate_benchmark_congruence(
            benchmark_series, endpoints
        )
        # The benchmark shares the strategy's authoritative starting-wealth
        # anchor; no anchor field is added to the benchmark evidence model.
        benchmark = _path_statistics(
            anchor=endpoints.wealth_anchor,
            terminal=benchmark_path[-1],
            wealth_path=benchmark_path,
            returns=_periodic_returns(benchmark_path, context=context),
            elapsed_days=endpoints.elapsed_days,
            policy=policy,
            context=context,
        )
        benchmark_metrics = BenchmarkPerformanceMetrics(
            ending_equity=benchmark_path[-1],
            total_return=defined_metric(benchmark.total_return),
            cagr=defined_metric(benchmark.growth_rate),
            annualized_volatility=benchmark.annualized_volatility,
            maximum_drawdown=benchmark.maximum_drawdown,
            sharpe_ratio=benchmark.sharpe_ratio,
            sortino_ratio=benchmark.sortino_ratio,
        )
        relative_metrics = _relative_metrics(
            strategy=strategy,
            benchmark=benchmark,
            strategy_terminal=endpoints.terminal_wealth,
            benchmark_terminal=benchmark_path[-1],
            context=context,
        )

    return build_research_metrics_result(
        provenance=ResearchMetricsProvenance(
            source_audit_result_fingerprint=source_result.result_fingerprint,
            performance_measurement_policy_fingerprint=(
                policy.policy_fingerprint
            ),
            benchmark_series_fingerprint=(
                None
                if benchmark_series is None
                else benchmark_series.content_fingerprint
            ),
        ),
        strategy_metrics=strategy_metrics,
        trade_metrics=_trade_metrics(
            source_result, sessions=endpoints.sessions, context=context
        ),
        benchmark_metrics=benchmark_metrics,
        relative_metrics=relative_metrics,
    )


__all__ = ["calculate_research_metrics"]
