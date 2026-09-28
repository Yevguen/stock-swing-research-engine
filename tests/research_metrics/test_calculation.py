"""Phase 16B.2 calculation acceptance cases.

Covers the frozen endpoint (EC), periodic-return (RET), total-return and
drawdown (TR/DD), CAGR, and undefined/failure (UF) case families.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.research_metrics import (
    ResearchMetricsCalculationError,
    ResearchMetricsValidationError,
    UndefinedMetricReason,
    calculate_research_metrics,
    canonical_statistical_context,
)
from stock_swing_d1.research_metrics import calculation

from conftest import (
    CANONICAL_BENCHMARK_VALUES,
    CANONICAL_SESSIONS,
    benchmark_evidence,
    build_synthetic_audit_result,
    canonical_experiment,
    make_closed_trade,
    make_open_trade,
)


ANCHOR = Decimal("100000")


def _measure(result, policy, benchmark=None):
    return calculate_research_metrics(
        source_result=result, policy=policy, benchmark_series=benchmark
    )


def _codes(captured) -> str:
    return captured.value.code


# ---------------------------------------------------------------------------
# EC -- endpoint and session-coverage cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "anchor", (Decimal("100000"), Decimal("100000.00"), Decimal("1.0E+5"))
)
def test_the_canonical_starting_capital_is_accepted_at_any_scale(
    anchor, performance_policy
):
    """EC-01/EC-02: numeric equality governs; Decimal scale is not semantic."""

    experiment = canonical_experiment(initial_equity=anchor)

    metrics = _measure(experiment, performance_policy)

    assert metrics.strategy_metrics.total_return.value == Decimal(
        "0.100000000000000000"
    )


@pytest.mark.parametrize(
    "anchor", (Decimal("50000"), Decimal("100001"), Decimal("99999.99"))
)
def test_a_noncanonical_starting_capital_fails_closed(anchor, performance_policy):
    """EC-03."""

    experiment = canonical_experiment(initial_equity=anchor)

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(experiment, performance_policy)
    assert _codes(captured) == "NONCANONICAL_EXPERIMENT_WEALTH_ANCHOR"


def test_zero_starting_capital_fails_closed(performance_policy):
    """EC-04: rejected as invalid input before any ratio is attempted."""

    experiment = canonical_experiment(initial_equity=Decimal("0"))

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(experiment, performance_policy)
    assert _codes(captured) == "NONPOSITIVE_EXPERIMENT_WEALTH_ANCHOR"


def test_a_january_first_start_boundary_with_a_later_first_session_is_accepted(
    canonical_audit_result, performance_policy
):
    """EC-05/EC-06: a calendar boundary need not be a trading session."""

    from stock_swing_d1.research_metrics import semantic_json_bytes

    assert canonical_audit_result.decision_interval.decision_start_date == (
        date(2025, 1, 1)
    )
    assert canonical_audit_result.equity_curve[0].session == date(2025, 1, 2)
    before = semantic_json_bytes(canonical_audit_result)

    metrics = _measure(canonical_audit_result, performance_policy)

    # The audited result is left exactly as it arrived: no prepended anchor,
    # no persisted fake session, no change to Phase 15D.
    assert semantic_json_bytes(canonical_audit_result) == before

    # No synthetic January-1 observation is created: the genuine session count
    # is unchanged, and it still yields exactly N-1 periodic returns.
    context = canonical_statistical_context()
    endpoints = calculation._validate_canonical_endpoints(
        canonical_audit_result
    )
    assert endpoints.sessions == CANONICAL_SESSIONS
    assert len(endpoints.wealth_path) == 3
    assert len(calculation._periodic_returns(
        endpoints.wealth_path, context=context
    )) == 2
    assert metrics.strategy_metrics.ending_equity == Decimal("110000")


def test_an_empty_equity_curve_is_invalid_canonical_input(performance_policy):
    """EC-07: not zero performance, not an undefined metric -- no result."""

    experiment = build_synthetic_audit_result(initial_equity=ANCHOR)

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(experiment, performance_policy)
    assert _codes(captured) == "EMPTY_STRATEGY_EQUITY_EVIDENCE"


def test_equity_row_count_must_equal_the_authoritative_session_count(
    performance_policy,
):
    """EC-08."""

    experiment = canonical_experiment(
        transition_sessions=(
            CANONICAL_SESSIONS[0],
            CANONICAL_SESSIONS[1],
            date(2025, 12, 30),
            CANONICAL_SESSIONS[2],
        )
    )
    assert experiment.summary.processed_session_count == 4
    assert len(experiment.equity_curve) == 3

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(experiment, performance_policy)
    assert _codes(captured) == "INCOMPLETE_STRATEGY_SESSION_COVERAGE"


def test_a_missing_interior_strategy_session_fails_closed(performance_policy):
    """EC-09: a gap would silently compound a return across it."""

    experiment = build_synthetic_audit_result(
        initial_equity=ANCHOR,
        equity_observations=(
            (CANONICAL_SESSIONS[0], Decimal("101000")),
            (CANONICAL_SESSIONS[2], Decimal("110000")),
        ),
        transition_sessions=CANONICAL_SESSIONS[:1] + CANONICAL_SESSIONS[2:],
    )
    # Coverage counts agree; only the authoritative sequence disagrees.
    interior = build_synthetic_audit_result(
        initial_equity=ANCHOR,
        equity_observations=(
            (CANONICAL_SESSIONS[0], Decimal("101000")),
            (CANONICAL_SESSIONS[2], Decimal("110000")),
        ),
        transition_sessions=CANONICAL_SESSIONS,
    )
    assert _measure(experiment, performance_policy) is not None

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(interior, performance_policy)
    assert _codes(captured) == "INCOMPLETE_STRATEGY_SESSION_COVERAGE"


def test_strategy_sessions_out_of_correspondence_fail_closed(performance_policy):
    """EC-10: equal counts, different sequence, still rejected."""

    experiment = canonical_experiment(
        transition_sessions=(
            CANONICAL_SESSIONS[0],
            date(2025, 1, 6),
            CANONICAL_SESSIONS[2],
        )
    )

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(experiment, performance_policy)
    assert _codes(captured) == "STRATEGY_SESSION_SEQUENCE_MISMATCH"


def test_a_reordered_session_sequence_fails_closed():
    """EC-10: the ordered comparison, exercised directly on the two tuples."""

    ordered = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
    reordered = (date(2025, 1, 3), date(2025, 1, 2), date(2025, 1, 6))

    assert ordered != reordered
    assert sorted(ordered) == sorted(reordered)


def test_a_final_strategy_session_before_the_decision_end_fails_closed(
    performance_policy,
):
    """EC-11."""

    sessions = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 12, 30))
    experiment = build_synthetic_audit_result(
        initial_equity=ANCHOR,
        equity_observations=tuple(
            zip(sessions, (Decimal("101000"), Decimal("99000"), Decimal("110000")))
        ),
    )

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(experiment, performance_policy)
    assert _codes(captured) == "STRATEGY_TERMINAL_SESSION_MISMATCH"


def test_a_final_strategy_session_at_the_decision_end_is_accepted(
    canonical_audit_result, performance_policy
):
    """EC-12."""

    assert canonical_audit_result.equity_curve[-1].session == (
        canonical_audit_result.decision_interval.decision_end_date
    )

    metrics = _measure(canonical_audit_result, performance_policy)

    assert metrics.strategy_metrics.ending_equity == (
        canonical_audit_result.final_equity
    )


def test_exact_strategy_and_benchmark_session_congruence_is_accepted(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """EC-13."""

    metrics = _measure(
        canonical_audit_result, performance_policy, canonical_benchmark_series
    )

    assert metrics.benchmark_metrics is not None
    assert metrics.relative_metrics is not None
    assert metrics.benchmark_metrics.ending_equity == Decimal("104000")


@pytest.mark.parametrize(
    ("sessions", "values"),
    (
        pytest.param(
            CANONICAL_SESSIONS[:2],
            CANONICAL_BENCHMARK_VALUES[:2],
            id="strict-subset",
        ),
        pytest.param(
            CANONICAL_SESSIONS + (date(2026, 1, 2),),
            CANONICAL_BENCHMARK_VALUES + (Decimal("105000"),),
            id="strict-superset",
        ),
    ),
)
def test_a_benchmark_subset_or_superset_fails_closed(
    sessions,
    values,
    canonical_audit_result,
    benchmark_source_ref,
    performance_policy,
):
    """EC-14/EC-15: no inner join, no outer join."""

    series = benchmark_evidence(benchmark_source_ref, sessions, values)

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(canonical_audit_result, performance_policy, series)
    assert _codes(captured) == "BENCHMARK_SESSION_INCONGRUENCE"


def test_a_benchmark_covering_different_sessions_fails_closed(
    canonical_audit_result, benchmark_source_ref, performance_policy
):
    """EC-16: same count and same order, different sessions -- still rejected."""

    series = benchmark_evidence(
        benchmark_source_ref,
        (date(2025, 1, 2), date(2025, 1, 6), date(2025, 12, 31)),
        CANONICAL_BENCHMARK_VALUES,
    )

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(canonical_audit_result, performance_policy, series)
    assert _codes(captured) == "BENCHMARK_SESSION_INCONGRUENCE"


# ---------------------------------------------------------------------------
# RET -- null-first periodic-return cases
# ---------------------------------------------------------------------------


def test_n_genuine_sessions_produce_exactly_n_minus_one_returns():
    """RET-01/RET-03/RET-05/RET-06: null-first, with no synthetic first return."""

    context = canonical_statistical_context()
    path = (Decimal("101000"), Decimal("99000"), Decimal("110000"))

    returns = calculation._periodic_returns(path, context=context)

    assert len(returns) == len(path) - 1 == 2
    # The first genuine observation has no periodic return at all: neither a
    # zero placeholder nor an E1/E0 anchored return appears in the vector.
    anchored_first = calculation.divide(path[0], ANCHOR, context=context)
    assert Decimal("0") not in returns
    assert calculation.subtract(
        anchored_first, Decimal("1"), context=context
    ) not in returns
    assert returns[0] == calculation.subtract(
        calculation.divide(path[1], path[0], context=context),
        Decimal("1"),
        context=context,
    )


def test_strategy_and_benchmark_return_arity_is_symmetric(
    canonical_audit_result, canonical_benchmark_series
):
    """RET-02/RET-04/RET-07."""

    context = canonical_statistical_context()
    endpoints = calculation._validate_canonical_endpoints(
        canonical_audit_result
    )
    benchmark_path = calculation._validate_benchmark_congruence(
        canonical_benchmark_series, endpoints
    )

    strategy_returns = calculation._periodic_returns(
        endpoints.wealth_path, context=context
    )
    benchmark_returns = calculation._periodic_returns(
        benchmark_path, context=context
    )

    assert len(strategy_returns) == len(benchmark_returns) == 2
    assert len(benchmark_path) == 3
    assert Decimal("0") not in benchmark_returns


def test_two_genuine_sessions_produce_exactly_one_periodic_return():
    """RET-08."""

    context = canonical_statistical_context()

    returns = calculation._periodic_returns(
        (Decimal("101000"), Decimal("99000")), context=context
    )

    assert len(returns) == 1


def test_one_periodic_return_leaves_independent_metrics_valid(
    performance_policy,
):
    """RET-09: two-return statistics are undefined; the rest still publish."""

    experiment = build_synthetic_audit_result(
        initial_equity=ANCHOR,
        equity_observations=(
            (date(2025, 1, 2), Decimal("101000")),
            (date(2025, 12, 31), Decimal("99000")),
        ),
    )

    metrics = _measure(experiment, performance_policy).strategy_metrics

    for undefined in (metrics.annualized_volatility, metrics.sharpe_ratio):
        assert undefined.value is None
        assert undefined.undefined_reason is (
            UndefinedMetricReason.INSUFFICIENT_RETURN_OBSERVATIONS
        )
    assert metrics.sortino_ratio.value is not None
    assert metrics.total_return.value == Decimal("-0.010000000000000000")
    assert metrics.maximum_drawdown.value is not None


def test_a_single_genuine_session_leaves_every_return_statistic_undefined(
    performance_policy,
):
    """Zero periodic returns: even the one-observation statistics are undefined."""

    experiment = build_synthetic_audit_result(
        initial_equity=ANCHOR,
        equity_observations=((date(2025, 12, 31), Decimal("110000")),),
    )

    metrics = _measure(experiment, performance_policy).strategy_metrics

    for undefined in (
        metrics.annualized_volatility,
        metrics.sharpe_ratio,
        metrics.sortino_ratio,
    ):
        assert undefined.value is None
        assert undefined.undefined_reason is (
            UndefinedMetricReason.INSUFFICIENT_RETURN_OBSERVATIONS
        )
    assert metrics.total_return.value == Decimal("0.100000000000000000")


def test_a_flat_return_sequence_has_no_sharpe_ratio_rather_than_zero(
    performance_policy,
):
    """UF-01 with ZERO_RETURN_VARIANCE."""

    experiment = build_synthetic_audit_result(
        initial_equity=ANCHOR,
        equity_observations=(
            (date(2025, 1, 2), Decimal("100000")),
            (date(2025, 1, 3), Decimal("100000")),
            (date(2025, 12, 31), Decimal("100000")),
        ),
    )

    metrics = _measure(experiment, performance_policy).strategy_metrics

    assert metrics.annualized_volatility.value == Decimal("0E-18")
    assert metrics.sharpe_ratio.value is None
    assert metrics.sharpe_ratio.undefined_reason is (
        UndefinedMetricReason.ZERO_RETURN_VARIANCE
    )
    assert metrics.sortino_ratio.value is None
    assert metrics.sortino_ratio.undefined_reason is (
        UndefinedMetricReason.NO_DOWNSIDE_DEVIATION
    )


def test_a_monotonically_rising_path_has_no_downside_deviation(
    performance_policy,
):
    """UF-01 with NO_DOWNSIDE_DEVIATION, and a zero maximum drawdown."""

    experiment = build_synthetic_audit_result(
        initial_equity=ANCHOR,
        equity_observations=(
            (date(2025, 1, 2), Decimal("101000")),
            (date(2025, 1, 3), Decimal("103000")),
            (date(2025, 12, 31), Decimal("110000")),
        ),
    )

    metrics = _measure(experiment, performance_policy).strategy_metrics

    assert metrics.sortino_ratio.value is None
    assert metrics.sortino_ratio.undefined_reason is (
        UndefinedMetricReason.NO_DOWNSIDE_DEVIATION
    )
    assert metrics.maximum_drawdown.value == Decimal("0E-18")


# ---------------------------------------------------------------------------
# TR / DD -- wealth-anchor cases
# ---------------------------------------------------------------------------


def test_total_return_uses_the_wealth_anchor_not_the_first_session(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """TR-01/TR-02."""

    metrics = _measure(
        canonical_audit_result, performance_policy, canonical_benchmark_series
    )

    # 110000 / 100000 - 1, not 110000 / 101000 - 1.
    assert metrics.strategy_metrics.total_return.value == Decimal(
        "0.100000000000000000"
    )
    # 104000 / 100000 - 1, where B0 == E0 and not the first benchmark value.
    assert metrics.benchmark_metrics.total_return.value == Decimal(
        "0.040000000000000000"
    )
    assert metrics.strategy_metrics.net_pnl == Decimal("10000")


def test_the_drawdown_path_begins_at_the_wealth_anchor(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """DD-01/DD-02: the benchmark anchor is B0 == E0, never its first value."""

    metrics = _measure(
        canonical_audit_result, performance_policy, canonical_benchmark_series
    )

    # The benchmark drawdown is measured over the anchored wealth path
    # 100000, 100500, 100250, 104000: (100500 - 100250) / 100500.
    assert metrics.benchmark_metrics.maximum_drawdown.value == Decimal(
        "0.002487562189054726"
    )
    assert metrics.strategy_metrics.maximum_drawdown.value == Decimal(
        "0.019801980198019802"
    )


def test_a_first_session_loss_creates_drawdown_from_starting_wealth(
    performance_policy,
):
    """DD-03: the anchor is in the wealth path even though it makes no return."""

    experiment = build_synthetic_audit_result(
        initial_equity=ANCHOR,
        equity_observations=(
            (date(2025, 1, 2), Decimal("90000")),
            (date(2025, 1, 3), Decimal("95000")),
            (date(2025, 12, 31), Decimal("99000")),
        ),
    )

    metrics = _measure(experiment, performance_policy).strategy_metrics

    # (100000 - 90000) / 100000 == 0.1, which is only reachable if E0 is in
    # the path: the largest peak-to-trough decline among genuine sessions
    # alone would be zero, since the path only rises after the first session.
    assert metrics.maximum_drawdown.value == Decimal("0.100000000000000000")


# ---------------------------------------------------------------------------
# CAGR cases
# ---------------------------------------------------------------------------


def test_elapsed_days_are_exact_calendar_days_with_no_inclusive_increment(
    canonical_audit_result,
):
    """CAGR-01/CAGR-02."""

    endpoints = calculation._validate_canonical_endpoints(
        canonical_audit_result
    )
    interval = canonical_audit_result.decision_interval

    expected = (
        interval.decision_end_date - interval.decision_start_date
    ).days
    assert endpoints.elapsed_days == expected == 364
    assert endpoints.elapsed_days != expected + 1
    # Nor is it a session, observation, return or trade count.
    assert endpoints.elapsed_days != len(endpoints.sessions)


def test_a_zero_length_decision_interval_is_rejected(performance_policy):
    """CAGR-04: structurally legal upstream, inadmissible for measurement."""

    session = date(2025, 6, 30)
    experiment = build_synthetic_audit_result(
        decision_start=session,
        decision_end=session,
        initial_equity=ANCHOR,
        equity_observations=((session, Decimal("110000")),),
    )

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(experiment, performance_policy)
    assert _codes(captured) == "NONPOSITIVE_DECISION_DURATION"


def test_the_growth_rate_follows_the_normative_decimal_ln_exp_algorithm(
    canonical_audit_result, performance_policy
):
    """CAGR-03/CAGR-05/CAGR-06: 365.25 == 1461/4, computed with ln and exp."""

    context = canonical_statistical_context()
    elapsed_days = 364
    ratio = calculation.divide(
        Decimal("110000"), ANCHOR, context=context
    )
    expected = calculation.quantize_canonical_metric(
        calculation.subtract(
            calculation.exponential(
                calculation.divide(
                    calculation.multiply(
                        calculation.natural_logarithm(ratio, context=context),
                        Decimal(1461),
                        context=context,
                    ),
                    Decimal(4 * elapsed_days),
                    context=context,
                ),
                context=context,
            ),
            Decimal("1"),
            context=context,
        ),
        context=context,
    )

    metrics = _measure(canonical_audit_result, performance_policy)

    assert metrics.strategy_metrics.cagr.value == expected
    assert calculation._DAY_COUNT_NUMERATOR == 1461
    assert calculation._DAY_COUNT_DENOMINATOR == 4


def test_a_zero_terminal_wealth_gives_exactly_negative_one(performance_policy):
    """CAGR-07/CAGR-08: a defined total loss, and ln(0) is never evaluated."""

    context = canonical_statistical_context()
    with pytest.raises(ResearchMetricsCalculationError):
        calculation.natural_logarithm(Decimal("0"), context=context)

    assert calculation._compound_growth_rate(
        ANCHOR, Decimal("0"), 364, context=context
    ) == Decimal("-1")

    experiment = build_synthetic_audit_result(
        initial_equity=ANCHOR,
        equity_observations=(
            (date(2025, 1, 2), Decimal("50000")),
            (date(2025, 1, 3), Decimal("25000")),
            (date(2025, 12, 31), Decimal("0")),
        ),
    )

    metrics = _measure(experiment, performance_policy).strategy_metrics

    assert metrics.cagr.value == Decimal("-1")
    assert metrics.total_return.value == Decimal("-1.000000000000000000")
    assert metrics.maximum_drawdown.value == Decimal("1.000000000000000000")


def test_strategy_and_benchmark_growth_rates_use_one_implementation(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """CAGR-09: symmetry is guaranteed by construction, not by convention."""

    context = canonical_statistical_context()
    metrics = _measure(
        canonical_audit_result, performance_policy, canonical_benchmark_series
    )

    assert metrics.benchmark_metrics.cagr.value == (
        calculation._compound_growth_rate(
            ANCHOR, Decimal("104000"), 364, context=context
        )
    )
    assert metrics.strategy_metrics.cagr.value == (
        calculation._compound_growth_rate(
            ANCHOR, Decimal("110000"), 364, context=context
        )
    )
    assert metrics.relative_metrics.strategy_minus_benchmark_cagr.value == (
        calculation.quantize_canonical_metric(
            calculation.subtract(
                metrics.strategy_metrics.cagr.value,
                metrics.benchmark_metrics.cagr.value,
                context=context,
            ),
            context=context,
        )
    )


def test_the_ending_wealth_ratio_compares_terminal_wealth(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    metrics = _measure(
        canonical_audit_result, performance_policy, canonical_benchmark_series
    )

    context = canonical_statistical_context()
    assert metrics.relative_metrics.ending_wealth_ratio.value == (
        calculation.quantize_canonical_metric(
            calculation.divide(
                Decimal("110000"), Decimal("104000"), context=context
            ),
            context=context,
        )
    )
    assert metrics.relative_metrics.ending_wealth_ratio.value > Decimal("1")


# ---------------------------------------------------------------------------
# Completed-trade cases
# ---------------------------------------------------------------------------


def _trade(trade_id, entry, exit_price, **overrides):
    return make_closed_trade(
        trade_id=trade_id,
        entry_session=date(2025, 1, 2),
        exit_session=date(2025, 1, 3),
        entry_fill_price=entry,
        exit_fill_price=exit_price,
        **overrides,
    )


def test_completed_trades_are_classified_from_dividend_inclusive_total_pnl(
    performance_policy,
):
    """A dividend can turn a trading loss into a winning completed trade."""

    experiment = canonical_experiment(
        trades=(
            _trade("T1", Decimal("10"), Decimal("12")),
            _trade(
                "T2",
                Decimal("10"),
                Decimal("9"),
                ordinary_dividend_income=Decimal("150"),
            ),
            _trade(
                "T3",
                Decimal("10"),
                Decimal("9"),
                ordinary_dividend_income=Decimal("100"),
            ),
            _trade("T4", Decimal("10"), Decimal("8")),
        )
    )

    metrics = _measure(experiment, performance_policy).trade_metrics

    assert metrics.completed_trade_count == 4
    # T1 (+200) and T2 (-100 + 150 = +50) win; T3 (-100 + 100) is exactly
    # breakeven; T4 (-200) loses.
    assert metrics.win_count == 2
    assert metrics.breakeven_count == 1
    assert metrics.loss_count == 1
    assert metrics.gross_profit == Decimal("250")
    assert metrics.gross_loss == Decimal("200")
    assert metrics.profit_factor.value == Decimal("1.250000000000000000")
    assert metrics.expectancy.value == Decimal("12.500000000000000000")
    assert metrics.win_rate.value == Decimal("0.500000000000000000")
    assert metrics.breakeven_rate.value == Decimal("0.250000000000000000")


def test_a_position_open_at_the_decision_end_is_excluded(performance_policy):
    experiment = canonical_experiment(
        trades=(
            _trade("T1", Decimal("10"), Decimal("12")),
            make_open_trade(
                trade_id="T2",
                entry_session=date(2025, 1, 2),
                final_mark_session=date(2025, 12, 31),
            ),
        )
    )

    metrics = _measure(experiment, performance_policy).trade_metrics

    assert metrics.completed_trade_count == 1
    assert metrics.win_count == 1


def test_no_completed_trades_leaves_the_rates_undefined_not_zero(
    canonical_audit_result, performance_policy
):
    """UF-01 with NO_COMPLETED_TRADES."""

    metrics = _measure(canonical_audit_result, performance_policy).trade_metrics

    assert metrics.completed_trade_count == 0
    for undefined in (
        metrics.win_rate,
        metrics.loss_rate,
        metrics.breakeven_rate,
        metrics.profit_factor,
        metrics.expectancy,
    ):
        assert undefined.value is None
        assert undefined.undefined_reason is (
            UndefinedMetricReason.NO_COMPLETED_TRADES
        )
    assert metrics.gross_profit == Decimal("0")
    assert metrics.gross_loss == Decimal("0")


def test_a_strategy_with_no_losses_has_no_profit_factor_rather_than_zero(
    performance_policy,
):
    """UF-01 with ZERO_GROSS_LOSS."""

    experiment = canonical_experiment(
        trades=(_trade("T1", Decimal("10"), Decimal("12")),)
    )

    metrics = _measure(experiment, performance_policy).trade_metrics

    assert metrics.gross_loss == Decimal("0")
    assert metrics.profit_factor.value is None
    assert metrics.profit_factor.undefined_reason is (
        UndefinedMetricReason.ZERO_GROSS_LOSS
    )
    assert metrics.expectancy.value == Decimal("200.000000000000000000")


@pytest.mark.parametrize(
    "unattributed",
    (
        pytest.param({"dividend_aware": False}, id="legacy-record"),
        pytest.param({"carried_in": True}, id="pre-run-attribution-unknown"),
    ),
)
def test_a_trade_without_authoritative_total_pnl_fails_closed(
    unattributed, performance_policy
):
    """Neither inferred, nor replaced by trading-only realized P&L."""

    experiment = canonical_experiment(
        trades=(_trade("T1", Decimal("10"), Decimal("12"), **unattributed),)
    )

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(experiment, performance_policy)
    assert _codes(captured) == "MISSING_AUTHORITATIVE_TRADE_TOTAL_PNL"


# ---------------------------------------------------------------------------
# UF -- failure taxonomy and determinism
# ---------------------------------------------------------------------------


def test_invalid_canonical_input_publishes_no_result_at_all(performance_policy):
    """UF-05: invalidity is not degraded into an undefined metric."""

    experiment = build_synthetic_audit_result(initial_equity=ANCHOR)

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _measure(experiment, performance_policy)
    assert isinstance(captured.value, ResearchMetricsValidationError)
    assert not isinstance(captured.value, ResearchMetricsCalculationError)


def test_a_calculation_failure_is_never_reported_as_an_undefined_metric():
    """UF-06: the two categories are distinct types, not one field convention."""

    context = canonical_statistical_context()

    with pytest.raises(ResearchMetricsCalculationError) as captured:
        calculation._periodic_returns(
            (Decimal("0"), Decimal("100")), context=context
        )
    assert captured.value.code == "UNDEFINED_PERIODIC_RETURN"
    assert not isinstance(captured.value, ResearchMetricsValidationError)
    assert not isinstance(captured.value, ValueError)


def test_repeated_measurement_of_identical_input_is_byte_identical(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """VER-08."""

    from stock_swing_d1.research_metrics import semantic_json_bytes

    first = _measure(
        canonical_audit_result, performance_policy, canonical_benchmark_series
    )
    second = _measure(
        canonical_audit_result, performance_policy, canonical_benchmark_series
    )

    assert semantic_json_bytes(first) == semantic_json_bytes(second)
    assert first.result_fingerprint == second.result_fingerprint


def test_a_result_binds_its_source_policy_and_benchmark_provenance(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    result = _measure(
        canonical_audit_result, performance_policy, canonical_benchmark_series
    )

    assert result.provenance.source_audit_result_fingerprint == (
        canonical_audit_result.result_fingerprint
    )
    assert result.provenance.performance_measurement_policy_fingerprint == (
        performance_policy.policy_fingerprint
    )
    assert result.provenance.benchmark_series_fingerprint == (
        canonical_benchmark_series.content_fingerprint
    )

    without_benchmark = _measure(canonical_audit_result, performance_policy)
    assert without_benchmark.provenance.benchmark_series_fingerprint is None
    assert without_benchmark.benchmark_metrics is None
    assert without_benchmark.relative_metrics is None
