"""Phase 16B.2 exact-arithmetic dependency cases (EXACT-01..EXACT-05).

The frozen Exact-Arithmetic Hardening primitives have exactly one
implementation, upstream. Phase 16B consumes them through the approved
``stock_swing_d1.backtest_results`` boundary, never by importing their
portfolio-owner module and never by writing a second copy of the algorithms.
"""

from __future__ import annotations

import ast
import decimal
import inspect
from decimal import Context, Decimal
from pathlib import Path

import pytest

import stock_swing_d1.backtest_results as backtest_results
from stock_swing_d1.portfolio import portfolio_dividend_events
from stock_swing_d1.backtest_results import valuation as valuation_module
from stock_swing_d1.research_metrics import arithmetic


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "src" / "stock_swing_d1" / "research_metrics"

PUBLIC_EXACT_HELPERS = (
    "add_exact_decimal",
    "subtract_exact_decimal",
    "exact_decimal_times_int",
    "sum_exact_decimal",
)


def _production_paths() -> tuple[Path, ...]:
    return tuple(sorted(PRODUCTION.glob("*.py")))


def test_the_exact_helpers_are_public_on_the_approved_boundary():
    """EXACT-01."""

    for name in PUBLIC_EXACT_HELPERS:
        assert name in backtest_results.__all__
        assert callable(getattr(backtest_results, name))


def test_phase16b_consumes_them_only_through_that_boundary():
    """EXACT-01/EXACT-02: no direct import of the portfolio implementation."""

    importers = []
    offenders = []
    for path in _production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            names = {alias.name for alias in node.names}
            if names & set(PUBLIC_EXACT_HELPERS):
                importers.append((path.name, module))
            if module.startswith("stock_swing_d1.portfolio"):
                offenders.append((path.name, module))

    assert offenders == []
    assert importers, "Phase 16B must actually consume the exact primitives"
    assert {module for _, module in importers} <= {
        "stock_swing_d1.backtest_results",
        "stock_swing_d1.research_metrics.arithmetic",
    }
    assert ("arithmetic.py", "stock_swing_d1.backtest_results") in importers


def test_the_public_exposure_performs_no_new_arithmetic():
    """EXACT-04: re-export and thin alias only -- same function objects."""

    assert (
        backtest_results.add_exact_decimal
        is portfolio_dividend_events.add_exact_decimal
    )
    assert (
        backtest_results.subtract_exact_decimal
        is portfolio_dividend_events.subtract_exact_decimal
    )
    assert (
        backtest_results.exact_decimal_times_int
        is portfolio_dividend_events.exact_decimal_times_int
    )
    assert backtest_results.sum_exact_decimal is valuation_module._exact_sum

    # And Phase 16B re-exposes exactly those same objects, adding nothing.
    for name in PUBLIC_EXACT_HELPERS:
        assert getattr(arithmetic, name) is getattr(backtest_results, name)


def test_phase16b_contains_no_second_exact_arithmetic_implementation():
    """EXACT-03: the tuple-reconstruction algorithms are not reimplemented."""

    source = "\n".join(
        path.read_text(encoding="utf-8") for path in _production_paths()
    )
    # The frozen algorithms work by decomposing operands and rebuilding a
    # Decimal from its exact ``(sign, digits, exponent)`` parts. None of that
    # machinery may appear here. (Reading ``as_tuple().exponent`` to *validate*
    # the canonical quantum's scale is an inspection, not an arithmetic
    # reimplementation, so the markers below are the constructor and the digit
    # decomposition rather than the accessor.)
    for token in (
        "Decimal((",
        "DecimalTuple",
        "_coefficient_as_int",
        "_digits_of",
        "divmod(",
    ):
        assert token not in source


def test_the_exact_helper_behaviour_is_unchanged_by_the_re_export():
    """EXACT-05: still exact, still independent of the ambient context."""

    tiny = Decimal("1E-38")
    big = Decimal("100000")
    original = decimal.getcontext()
    decimal.setcontext(Context(prec=6))
    try:
        assert backtest_results.add_exact_decimal(big, tiny) == Decimal(
            "100000.00000000000000000000000000000000000001"
        )
        assert backtest_results.subtract_exact_decimal(big, tiny) == Decimal(
            "99999.99999999999999999999999999999999999999"
        )
        assert backtest_results.exact_decimal_times_int(
            Decimal("1.0000000000000000000000000000000000001"), 3
        ) == Decimal("3.0000000000000000000000000000000000003")
        assert backtest_results.sum_exact_decimal(
            (big, tiny, tiny)
        ) == Decimal("100000.00000000000000000000000000000000000002")
        # Ordinary context-rounded `+` would silently lose those digits.
        assert big + tiny == Decimal("100000")
    finally:
        decimal.setcontext(original)


def test_phase16b_uses_the_exact_primitives_where_money_is_involved():
    """A monetary difference or aggregation never rides on context rounding."""

    from stock_swing_d1.research_metrics import calculation

    source = inspect.getsource(calculation)
    assert "subtract_exact_decimal(" in source
    assert "sum_exact_decimal(" in source

    # net_pnl and the drawdown numerator are monetary differences.
    tree = ast.parse(source)
    exact_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } & set(PUBLIC_EXACT_HELPERS)
    assert exact_calls == {"subtract_exact_decimal", "sum_exact_decimal"}


def test_a_scale_38_monetary_fact_survives_measurement_exactly(
    performance_policy,
):
    """The exact primitives keep dividend-scale cash out of a rounding trap."""

    from conftest import build_synthetic_audit_result
    from datetime import date

    from stock_swing_d1.research_metrics import calculate_research_metrics

    terminal = Decimal("100000.00000000000000000000000000000000000001")
    experiment = build_synthetic_audit_result(
        initial_equity=Decimal("100000"),
        equity_observations=(
            (date(2025, 1, 2), Decimal("100000")),
            (date(2025, 1, 3), Decimal("100000")),
            (date(2025, 12, 31), terminal),
        ),
    )

    original = decimal.getcontext()
    decimal.setcontext(Context(prec=6))
    try:
        metrics = calculate_research_metrics(
            source_result=experiment, policy=performance_policy
        ).strategy_metrics
    finally:
        decimal.setcontext(original)

    assert metrics.ending_equity == terminal
    assert metrics.net_pnl == Decimal("1E-38")


@pytest.mark.parametrize("name", PUBLIC_EXACT_HELPERS)
def test_the_public_names_are_stable_and_documented_as_re_exports(name):
    module = inspect.getmodule(getattr(backtest_results, name))
    assert module is not None
    assert module.__name__ in {
        "stock_swing_d1.portfolio.portfolio_dividend_events",
        "stock_swing_d1.backtest_results.valuation",
    }
