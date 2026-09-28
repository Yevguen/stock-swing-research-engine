"""The Phase 16B.2 v0.2.1 Clause 27/28 public exact-arithmetic exposure.

``backtest_results/__init__.py`` may expose the already-frozen
Exact-Arithmetic Hardening primitives so a downstream research layer can
consume them across an approved package boundary. That exposure is a pure
re-export: it introduces no arithmetic, no new owner, and no change to any
Phase 15D semantic.
"""

from __future__ import annotations

import ast
import decimal
from decimal import Context, Decimal
from pathlib import Path

import pytest

import stock_swing_d1.backtest_results as backtest_results
from stock_swing_d1.backtest_results import valuation as valuation_module
from stock_swing_d1.portfolio import portfolio_dividend_events


PACKAGE_INIT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "stock_swing_d1"
    / "backtest_results"
    / "__init__.py"
)

EXPORTED = {
    "add_exact_decimal": portfolio_dividend_events.add_exact_decimal,
    "subtract_exact_decimal": portfolio_dividend_events.subtract_exact_decimal,
    "exact_decimal_times_int": portfolio_dividend_events.exact_decimal_times_int,
    "sum_exact_decimal": valuation_module._exact_sum,
}


@pytest.mark.parametrize(("name", "owner"), sorted(EXPORTED.items()))
def test_each_public_name_is_the_existing_implementation_object(name, owner):
    assert getattr(backtest_results, name) is owner
    assert name in backtest_results.__all__


def test_the_package_init_defines_no_arithmetic_of_its_own():
    """A re-export module contains imports and a name list, nothing more."""

    tree = ast.parse(PACKAGE_INIT.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        assert not isinstance(node, ast.ClassDef)
        assert not isinstance(node, ast.Lambda)
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            assert not isinstance(
                node.op,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.FloorDiv,
                    ast.Mod,
                    ast.Pow,
                ),
            )


def test_the_re_export_does_not_alter_exact_helper_behaviour():
    """Same results, still independent of the caller's ambient context."""

    tiny = Decimal("1E-38")
    big = Decimal("100000")
    original = decimal.getcontext()
    decimal.setcontext(Context(prec=6))
    try:
        assert backtest_results.add_exact_decimal(
            big, tiny
        ) == portfolio_dividend_events.add_exact_decimal(big, tiny)
        assert backtest_results.add_exact_decimal(big, tiny) == Decimal(
            "100000.00000000000000000000000000000000000001"
        )
        assert backtest_results.subtract_exact_decimal(big, tiny) == Decimal(
            "99999.99999999999999999999999999999999999999"
        )
        assert backtest_results.exact_decimal_times_int(tiny, 3) == Decimal(
            "3E-38"
        )
        assert backtest_results.sum_exact_decimal((big, tiny)) == Decimal(
            "100000.00000000000000000000000000000000000001"
        )
    finally:
        decimal.setcontext(original)


def test_the_private_accumulator_name_remains_available_upstream():
    """The alias is additive: Phase 15D's own call sites are untouched."""

    assert valuation_module._exact_sum is not None
    assert valuation_module._subtract_exact is (
        portfolio_dividend_events.subtract_exact_decimal
    )
