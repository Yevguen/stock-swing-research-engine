"""Phase 16B Amendment v0.1: four additional authoritative trade metrics.

The amendment changes measurement/report completeness only. Every assertion
here therefore measures the four new metrics from already-audited completed
Phase 15D trades, and pins the pre-existing Phase 16B values against change.
"""

from __future__ import annotations

import ast
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.research_metrics import (
    TradePerformanceMetrics,
    UndefinedMetricReason,
    calculate_research_metrics,
    defined_metric,
    undefined_metric,
)
from stock_swing_d1.research_metrics.errors import (
    ResearchMetricsValidationError,
)

from tests.research_metrics.conftest import (
    canonical_experiment,
    make_closed_trade,
    make_open_trade,
)


PRODUCTION = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "stock_swing_d1"
    / "research_metrics"
)

AMENDMENT_METRICS = (
    "average_winner_usd",
    "average_loser_usd",
    "worst_trade_usd",
    "average_holding_sessions",
)

# The frozen three-session canonical baseline. The last two sessions are
# 2025-01-03 and 2025-12-31: adjacent in the authoritative chronology, 362
# calendar days apart. Any calendar arithmetic is therefore glaringly visible.
CANONICAL_ENTRY = date(2025, 1, 2)
CANONICAL_EXIT = date(2025, 1, 3)


def _measure(result, policy):
    return calculate_research_metrics(
        source_result=result, policy=policy, benchmark_series=None
    )


def _trade(
    trade_id,
    entry_price,
    exit_price,
    *,
    entry_session=CANONICAL_ENTRY,
    exit_session=CANONICAL_EXIT,
    **overrides,
):
    return make_closed_trade(
        trade_id=trade_id,
        entry_session=entry_session,
        exit_session=exit_session,
        entry_fill_price=entry_price,
        exit_fill_price=exit_price,
        **overrides,
    )


def _trades(experiment_trades, policy, **experiment_overrides):
    experiment = canonical_experiment(
        trades=experiment_trades, **experiment_overrides
    )
    return _measure(experiment, policy).trade_metrics


# ---------------------------------------------------------------------------
# A -- average_winner_usd
# ---------------------------------------------------------------------------


def test_average_winner_is_the_exact_mean_of_winning_total_pnl(
    performance_policy,
):
    """+200 and +300 average to exactly +250, as a USD amount."""

    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("12")),
            _trade("T2", Decimal("10"), Decimal("13")),
        ),
        performance_policy,
    )

    assert metrics.average_winner_usd.value == Decimal("250")
    assert type(metrics.average_winner_usd.value) is Decimal
    assert metrics.average_winner_usd.undefined_reason is None


def test_average_winner_excludes_losers_and_breakevens(performance_policy):
    """Only the winning population enters the numerator and the divisor."""

    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("12")),
            _trade("T2", Decimal("10"), Decimal("13")),
            _trade("T3", Decimal("10"), Decimal("10")),
            _trade("T4", Decimal("10"), Decimal("8")),
        ),
        performance_policy,
    )

    assert metrics.completed_trade_count == 4
    assert metrics.win_count == 2
    # (200 + 300) / 2, not /3 (breakeven) and not /4 (whole population).
    assert metrics.average_winner_usd.value == Decimal(
        "250.000000000000000000"
    )


def test_average_winner_is_undefined_without_a_winner(performance_policy):
    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("8")),
            _trade("T2", Decimal("10"), Decimal("10")),
        ),
        performance_policy,
    )

    assert metrics.completed_trade_count == 2
    assert metrics.win_count == 0
    assert metrics.average_winner_usd.value is None
    assert metrics.average_winner_usd.undefined_reason is (
        UndefinedMetricReason.NO_WINNING_TRADES
    )


def test_a_defined_average_winner_is_always_strictly_positive(
    performance_policy,
):
    """A dividend-rescued win is still a win, and still positive."""

    metrics = _trades(
        (
            _trade(
                "T1",
                Decimal("10"),
                Decimal("9"),
                ordinary_dividend_income=Decimal("150"),
            ),
        ),
        performance_policy,
    )

    # -100 trading + 150 dividend = +50 authoritative total.
    assert metrics.average_winner_usd.value == Decimal("50")
    assert metrics.average_winner_usd.value > Decimal("0")


def test_average_winner_is_exact_where_the_mean_does_not_terminate(
    performance_policy,
):
    """(200 + 300 + 300) / 3 uses the canonical quantum, not a float."""

    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("12")),
            _trade("T2", Decimal("10"), Decimal("13")),
            _trade("T3", Decimal("10"), Decimal("13.05")),
        ),
        performance_policy,
    )

    # (200 + 300 + 305) / 3 == 268.33...
    assert metrics.average_winner_usd.value == Decimal(
        "268.333333333333333333"
    )
    assert metrics.average_winner_usd.value.as_tuple().exponent == -18


# ---------------------------------------------------------------------------
# B -- average_loser_usd
# ---------------------------------------------------------------------------


def test_average_loser_is_the_exact_signed_mean_of_losing_total_pnl(
    performance_policy,
):
    """-200 and -400 average to exactly -300; the sign is preserved."""

    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("8")),
            _trade("T2", Decimal("10"), Decimal("6")),
        ),
        performance_policy,
    )

    assert metrics.average_loser_usd.value == Decimal("-300")
    assert metrics.average_loser_usd.value < Decimal("0")
    assert type(metrics.average_loser_usd.value) is Decimal


def test_average_loser_is_never_an_absolute_loss_magnitude(
    performance_policy,
):
    """The signed negative result is not restated as gross_loss / loss_count."""

    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("8")),
            _trade("T2", Decimal("10"), Decimal("6")),
        ),
        performance_policy,
    )

    magnitude = metrics.gross_loss / Decimal(metrics.loss_count)
    assert magnitude == Decimal("300")
    assert metrics.average_loser_usd.value != magnitude
    assert metrics.average_loser_usd.value == -magnitude


def test_average_loser_excludes_winners_and_breakevens(performance_policy):
    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("12")),
            _trade("T2", Decimal("10"), Decimal("10")),
            _trade("T3", Decimal("10"), Decimal("8")),
            _trade("T4", Decimal("10"), Decimal("6")),
        ),
        performance_policy,
    )

    assert metrics.loss_count == 2
    assert metrics.average_loser_usd.value == Decimal(
        "-300.000000000000000000"
    )


def test_average_loser_is_undefined_without_a_loser(performance_policy):
    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("12")),
            _trade("T2", Decimal("10"), Decimal("10")),
        ),
        performance_policy,
    )

    assert metrics.loss_count == 0
    assert metrics.average_loser_usd.value is None
    assert metrics.average_loser_usd.undefined_reason is (
        UndefinedMetricReason.NO_LOSING_TRADES
    )


def test_average_loser_is_exact_where_the_mean_does_not_terminate(
    performance_policy,
):
    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("8")),
            _trade("T2", Decimal("10"), Decimal("6")),
            _trade("T3", Decimal("10"), Decimal("6.05")),
        ),
        performance_policy,
    )

    # (-200 + -400 + -395) / 3 == -331.66...
    assert metrics.average_loser_usd.value == Decimal(
        "-331.666666666666666667"
    )


# ---------------------------------------------------------------------------
# C -- worst_trade_usd
# ---------------------------------------------------------------------------


def test_worst_trade_is_the_minimum_signed_pnl_of_a_mixed_population(
    performance_policy,
):
    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("12")),
            _trade("T2", Decimal("10"), Decimal("10")),
            _trade("T3", Decimal("10"), Decimal("8")),
            _trade("T4", Decimal("10"), Decimal("6")),
        ),
        performance_policy,
    )

    assert metrics.worst_trade_usd.value == Decimal("-400")
    assert type(metrics.worst_trade_usd.value) is Decimal


def test_an_all_winning_population_still_has_a_defined_worst_trade(
    performance_policy,
):
    """The smallest winner -- not undefined, and not a maximum absolute loss."""

    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("12")),
            _trade("T2", Decimal("10"), Decimal("13")),
        ),
        performance_policy,
    )

    assert metrics.loss_count == 0
    assert metrics.worst_trade_usd.value == Decimal("200")
    assert metrics.worst_trade_usd.value > Decimal("0")
    assert metrics.worst_trade_usd.undefined_reason is None


def test_an_all_breakeven_population_has_a_zero_worst_trade(
    performance_policy,
):
    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("10")),
            _trade("T2", Decimal("11"), Decimal("11")),
        ),
        performance_policy,
    )

    assert metrics.breakeven_count == 2
    assert metrics.worst_trade_usd.value == Decimal("0")
    assert metrics.average_winner_usd.value is None
    assert metrics.average_loser_usd.value is None


def test_worst_trade_includes_a_breakeven_when_it_is_the_minimum(
    performance_policy,
):
    """Breakeven is part of the population, so it can be the worst trade."""

    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("12")),
            _trade("T2", Decimal("10"), Decimal("10")),
        ),
        performance_policy,
    )

    assert metrics.worst_trade_usd.value == Decimal("0E-18")


def test_worst_trade_is_undefined_without_completed_trades(
    canonical_audit_result, performance_policy
):
    metrics = _measure(canonical_audit_result, performance_policy).trade_metrics

    assert metrics.completed_trade_count == 0
    assert metrics.worst_trade_usd.value is None
    assert metrics.worst_trade_usd.undefined_reason is (
        UndefinedMetricReason.NO_COMPLETED_TRADES
    )


# ---------------------------------------------------------------------------
# D -- average_holding_sessions
# ---------------------------------------------------------------------------


def test_a_single_trade_holds_for_its_inclusive_session_count(
    performance_policy,
):
    """Entry session through exit session inclusive: sessions 1 and 2."""

    metrics = _trades(
        (_trade("T1", Decimal("10"), Decimal("12")),), performance_policy
    )

    assert metrics.average_holding_sessions.value == Decimal("2")
    assert type(metrics.average_holding_sessions.value) is Decimal


def test_a_same_session_entry_and_exit_holds_for_one_session(
    performance_policy,
):
    metrics = _trades(
        (
            _trade(
                "T1",
                Decimal("10"),
                Decimal("12"),
                exit_session=CANONICAL_ENTRY,
            ),
        ),
        performance_policy,
    )

    assert metrics.average_holding_sessions.value == Decimal("1")


def test_multiple_trades_average_their_holding_session_counts(
    performance_policy,
):
    """One 1-session hold and one 2-session hold average to exactly 1.5."""

    metrics = _trades(
        (
            _trade(
                "T1",
                Decimal("10"),
                Decimal("12"),
                exit_session=CANONICAL_ENTRY,
            ),
            _trade("T2", Decimal("10"), Decimal("13")),
        ),
        performance_policy,
    )

    assert metrics.completed_trade_count == 2
    assert metrics.average_holding_sessions.value == Decimal(
        "1.500000000000000000"
    )


def test_a_non_integral_holding_average_is_exact(performance_policy):
    """(1 + 2 + 2) / 3 == 1.66..., carried at the canonical quantum."""

    metrics = _trades(
        (
            _trade(
                "T1",
                Decimal("10"),
                Decimal("12"),
                exit_session=CANONICAL_ENTRY,
            ),
            _trade("T2", Decimal("10"), Decimal("13")),
            _trade("T3", Decimal("10"), Decimal("8")),
        ),
        performance_policy,
    )

    assert metrics.average_holding_sessions.value == Decimal(
        "1.666666666666666667"
    )
    assert metrics.average_holding_sessions.value.as_tuple().exponent == -18


def test_a_weekend_between_two_sessions_does_not_inflate_the_count(
    performance_policy,
):
    """Friday to Monday is two trading sessions, not three or four days."""

    friday = date(2025, 1, 3)
    monday = date(2025, 1, 6)
    assert friday.weekday() == 4 and monday.weekday() == 0
    assert (monday - friday).days == 3

    metrics = _trades(
        (
            _trade(
                "T1",
                Decimal("10"),
                Decimal("12"),
                entry_session=friday,
                exit_session=monday,
            ),
        ),
        performance_policy,
        equity_observations=(
            (friday, Decimal("101000")),
            (monday, Decimal("99000")),
            (date(2025, 12, 31), Decimal("110000")),
        ),
    )

    assert metrics.average_holding_sessions.value == Decimal("2")


def test_a_market_holiday_between_two_sessions_does_not_inflate_the_count(
    performance_policy,
):
    """The authoritative chronology simply omits the closed session.

    2025-01-01 is a Wednesday and a U.S. market holiday, so the audited run
    processed 2024-12-31 and then 2025-01-02 with no weekend involved. The
    holding count is two sessions; every calendar reading is larger.
    """

    before = date(2024, 12, 31)
    after = date(2025, 1, 2)
    assert before.weekday() == 1 and after.weekday() == 3
    assert (after - before).days == 2

    metrics = _trades(
        (
            _trade(
                "T1",
                Decimal("10"),
                Decimal("12"),
                entry_session=before,
                exit_session=after,
            ),
        ),
        performance_policy,
        decision_start=date(2024, 12, 30),
        equity_observations=(
            (before, Decimal("101000")),
            (after, Decimal("99000")),
            (date(2025, 12, 31), Decimal("110000")),
        ),
    )

    assert metrics.average_holding_sessions.value == Decimal("2")


def test_calendar_day_subtraction_is_not_used(performance_policy):
    """Two adjacent authoritative sessions 362 calendar days apart.

    ``exit_date - entry_date`` would report 362 (or 363 inclusive). The
    authoritative chronology reports two sessions, because it is a session
    sequence and not a calendar.
    """

    entry = date(2025, 1, 3)
    exit_session = date(2025, 12, 31)
    assert (exit_session - entry).days == 362

    metrics = _trades(
        (
            _trade(
                "T1",
                Decimal("10"),
                Decimal("12"),
                entry_session=entry,
                exit_session=exit_session,
            ),
        ),
        performance_policy,
    )

    held = metrics.average_holding_sessions.value
    assert held == Decimal("2")
    assert held != Decimal("362")
    assert held != Decimal("363")


def test_average_holding_sessions_is_undefined_without_completed_trades(
    canonical_audit_result, performance_policy
):
    metrics = _measure(canonical_audit_result, performance_policy).trade_metrics

    assert metrics.average_holding_sessions.value is None
    assert metrics.average_holding_sessions.undefined_reason is (
        UndefinedMetricReason.NO_COMPLETED_TRADES
    )


def test_a_position_open_at_the_decision_end_has_no_holding_contribution(
    performance_policy,
):
    metrics = _trades(
        (
            _trade("T1", Decimal("10"), Decimal("12")),
            make_open_trade(
                trade_id="T2",
                entry_session=CANONICAL_ENTRY,
                final_mark_session=date(2025, 12, 31),
            ),
        ),
        performance_policy,
    )

    assert metrics.completed_trade_count == 1
    assert metrics.average_holding_sessions.value == Decimal("2")


def test_a_trade_outside_the_authoritative_chronology_fails_closed(
    performance_policy,
):
    """No canonical holding length exists, so none is guessed."""

    with pytest.raises(ResearchMetricsValidationError) as captured:
        _trades(
            (
                _trade(
                    "T1",
                    Decimal("10"),
                    Decimal("12"),
                    entry_session=date(2024, 6, 3),
                ),
            ),
            performance_policy,
        )
    assert captured.value.code == (
        "TRADE_OUTSIDE_AUTHORITATIVE_SESSION_CHRONOLOGY"
    )


def test_an_inverted_trade_chronology_fails_closed(performance_policy):
    with pytest.raises(ResearchMetricsValidationError) as captured:
        _trades(
            (
                _trade(
                    "T1",
                    Decimal("10"),
                    Decimal("12"),
                    entry_session=CANONICAL_EXIT,
                    exit_session=CANONICAL_ENTRY,
                ),
            ),
            performance_policy,
        )
    assert captured.value.code == "INVERTED_TRADE_SESSION_CHRONOLOGY"


# ---------------------------------------------------------------------------
# E -- general invariants
# ---------------------------------------------------------------------------


def _mixed_population():
    return (
        _trade("T1", Decimal("10"), Decimal("12")),
        _trade("T2", Decimal("10"), Decimal("10")),
        _trade("T3", Decimal("10"), Decimal("8")),
        _trade(
            "T4", Decimal("10"), Decimal("13"), exit_session=CANONICAL_ENTRY
        ),
    )


def test_input_ordering_does_not_change_any_aggregate(performance_policy):
    """Trade order is non-semantic for every aggregate the amendment adds."""

    forward = _mixed_population()
    reversed_order = tuple(reversed(forward))

    first = _trades(forward, performance_policy)
    second = _trades(reversed_order, performance_policy)

    assert first == second
    for name in AMENDMENT_METRICS:
        assert getattr(first, name) == getattr(second, name)


def test_equal_worst_trades_of_different_scale_are_order_independent(
    performance_policy,
):
    """-400 and -400.00 are numerically equal; the canonical scale settles it."""

    small = _trade("T1", Decimal("10"), Decimal("6"))
    same_value_other_scale = _trade("T2", Decimal("10.00"), Decimal("6.00"))

    forward = _trades((small, same_value_other_scale), performance_policy)
    backward = _trades((same_value_other_scale, small), performance_policy)

    assert forward.worst_trade_usd == backward.worst_trade_usd
    assert forward.worst_trade_usd.value.as_tuple() == (
        backward.worst_trade_usd.value.as_tuple()
    )


def test_no_float_enters_the_amendment_metrics(performance_policy):
    metrics = _trades(_mixed_population(), performance_policy)

    for name in AMENDMENT_METRICS:
        value = getattr(metrics, name).value
        assert value is None or type(value) is Decimal
        assert type(value) is not float


def test_the_amendment_metrics_are_frozen_and_immutable(performance_policy):
    metrics = _trades(_mixed_population(), performance_policy)

    for name in AMENDMENT_METRICS:
        with pytest.raises(Exception):
            setattr(metrics, name, defined_metric(Decimal("1")))
        assert type(getattr(metrics, name)).model_config["frozen"] is True


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("average_winner_usd", Decimal("-1")),
        ("average_winner_usd", Decimal("0")),
        ("average_loser_usd", Decimal("1")),
        ("average_loser_usd", Decimal("0")),
        ("average_holding_sessions", Decimal("0.5")),
    ),
)
def test_a_defined_value_contradicting_its_definition_is_rejected(
    trade_metrics, field_name, bad_value
):
    """The model refuses a sign or domain the amendment does not allow."""

    values = {
        name: getattr(trade_metrics, name)
        for name in type(trade_metrics).model_fields
    }
    values[field_name] = defined_metric(bad_value)
    with pytest.raises(Exception):
        TradePerformanceMetrics(**values)


def test_an_undefined_amendment_metric_is_still_accepted(trade_metrics):
    values = {
        name: getattr(trade_metrics, name)
        for name in type(trade_metrics).model_fields
    }
    for name in AMENDMENT_METRICS:
        values[name] = undefined_metric(
            UndefinedMetricReason.NO_COMPLETED_TRADES
        )
    rebuilt = TradePerformanceMetrics(**values)
    for name in AMENDMENT_METRICS:
        assert getattr(rebuilt, name).value is None


def test_calculation_is_deterministic_across_repeated_measurement(
    performance_policy,
):
    experiment = canonical_experiment(trades=_mixed_population())

    first = _measure(experiment, performance_policy)
    second = _measure(experiment, performance_policy)

    assert first.trade_metrics == second.trade_metrics
    assert first.result_fingerprint == second.result_fingerprint


# ---------------------------------------------------------------------------
# Non-interference with the pre-existing authoritative values
# ---------------------------------------------------------------------------


def test_every_preexisting_trade_value_is_unchanged(performance_policy):
    """The exact expectations of the frozen v0.2 trade suite, value for value."""

    metrics = _trades(
        (
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
        ),
        performance_policy,
    )

    assert metrics.completed_trade_count == 4
    assert metrics.win_count == 2
    assert metrics.loss_count == 1
    assert metrics.breakeven_count == 1
    assert metrics.gross_profit == Decimal("250")
    assert metrics.gross_loss == Decimal("200")
    assert metrics.win_rate.value == Decimal("0.500000000000000000")
    assert metrics.loss_rate.value == Decimal("0.250000000000000000")
    assert metrics.breakeven_rate.value == Decimal("0.250000000000000000")
    assert metrics.profit_factor.value == Decimal("1.250000000000000000")
    assert metrics.expectancy.value == Decimal("12.500000000000000000")

    # ... and the amendment values measured from the same population.
    assert metrics.average_winner_usd.value == Decimal(
        "125.000000000000000000"
    )
    assert metrics.average_loser_usd.value == Decimal(
        "-200.000000000000000000"
    )
    assert metrics.worst_trade_usd.value == Decimal("-200.000000000000000000")
    assert metrics.average_holding_sessions.value == Decimal(
        "2.000000000000000000"
    )


def test_no_equity_or_benchmark_metric_depends_on_the_amendment(
    performance_policy, canonical_benchmark_series
):
    """Adding completed trades leaves every equity-path metric untouched."""

    without = calculate_research_metrics(
        source_result=canonical_experiment(),
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )
    with_trades = calculate_research_metrics(
        source_result=canonical_experiment(trades=_mixed_population()),
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )

    assert with_trades.strategy_metrics == without.strategy_metrics
    assert with_trades.benchmark_metrics == without.benchmark_metrics
    assert with_trades.relative_metrics == without.relative_metrics


# ---------------------------------------------------------------------------
# Ownership boundary
# ---------------------------------------------------------------------------


def _called_names(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            yield function.id
        elif isinstance(function, ast.Attribute):
            yield function.attr


def test_only_the_calculation_module_computes_the_amendment_metrics():
    """Persistence and reporting may carry the four values; never derive them.

    Both modules necessarily *name* the fields -- persistence decodes them --
    so naming is not the test, and neither is ordinary structural arithmetic
    such as the artifact's own ``N-1`` observation invariant. Reaching the
    Phase 16B.2 calculation boundary, or invoking the canonical statistical
    machinery that would be needed to recompute a metric, is the test.
    """

    forbidden_calls = frozenset(
        {
            "calculate_research_metrics",
            "canonical_statistical_context",
            "quantize_canonical_metric",
            "divide",
            "multiply",
            "square_root",
            "natural_logarithm",
            "exponential",
            "sum_exact_decimal",
            "subtract_exact_decimal",
            "add_exact_decimal",
            "exact_decimal_times_int",
            "mean",
            "min",
            "max",
        }
    )
    forbidden_modules = (
        "stock_swing_d1.research_metrics.calculation",
        "stock_swing_d1.research_metrics.arithmetic",
    )
    offenders = []
    for name in ("persistence.py", "reporting.py"):
        tree = ast.parse((PRODUCTION / name).read_text(encoding="utf-8"))
        for called in _called_names(tree):
            if called in forbidden_calls:
                offenders.append((name, "call", called))
        for node in ast.walk(tree):
            modules = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or "",)
            for module in modules:
                if module in forbidden_modules:
                    offenders.append((name, "import", module))
    assert offenders == []


def test_persistence_and_reporting_perform_no_decimal_arithmetic():
    """No Decimal operand is ever an operand of an arithmetic operator there.

    The structural int arithmetic those modules legitimately perform (the
    artifact's ``N-1`` invariant, set differences in error reporting) never
    touches a metric value, so the guard is on Decimal-valued expressions
    rather than on operators as such: no metric is added, subtracted, scaled
    or divided anywhere in the persistence or report boundary.
    """

    offenders = []
    for name in ("persistence.py", "reporting.py"):
        source = (PRODUCTION / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp):
                continue
            for operand in (node.left, node.right):
                if isinstance(operand, ast.Call) and isinstance(
                    operand.func, ast.Name
                ):
                    if operand.func.id == "Decimal":
                        offenders.append((name, node.lineno))
                if isinstance(operand, ast.Name) and operand.id in (
                    "value",
                    "decoded",
                    "metric",
                ):
                    offenders.append((name, node.lineno))
    assert offenders == []


def test_the_amendment_metrics_are_declared_only_by_the_model_layer():
    """One owner declares the inventory; the calculation module fills it in."""

    declaring = []
    for path in sorted(PRODUCTION.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name
            ):
                if node.target.id in AMENDMENT_METRICS:
                    declaring.append(path.name)
    assert set(declaring) == {"models.py"}
