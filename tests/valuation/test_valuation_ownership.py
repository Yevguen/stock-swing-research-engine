"""Task 5C-A ownership boundaries: one formula, one owner, no leakage."""

from __future__ import annotations

import ast
from pathlib import Path

import stock_swing_d1.backtest_results as backtest_results_package
from stock_swing_d1.backtest_results import models as backtest_results_models
from stock_swing_d1.valuation import (
    PortfolioValuationMark,
    PortfolioValuationPolicy,
)


SRC = Path(__file__).resolve().parents[2] / "src" / "stock_swing_d1"
VALUATION = SRC / "valuation"
BACKTESTER = SRC / "backtester"
AUDIT_PACKAGE = "backtest_results"


def _imported_packages(root: Path) -> set[str]:
    packages: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                packages.add(node.module)
            elif isinstance(node, ast.Import):
                packages.update(alias.name for alias in node.names)
    return packages


def test_valuation_imports_no_downstream_package():
    forbidden = ("backtester", AUDIT_PACKAGE, "baseline_experiment", "data")
    for module in _imported_packages(VALUATION):
        if not module.startswith("stock_swing_d1"):
            continue
        tail = module.removeprefix("stock_swing_d1.").split(".")[0]
        assert tail not in forbidden, module


def test_phase15a_never_imports_the_audit_package():
    for module in _imported_packages(BACKTESTER):
        assert AUDIT_PACKAGE not in module


def test_phase15a_never_references_the_audit_valuation_snapshot():
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in BACKTESTER.rglob("*.py")
    )
    assert "HistoricalBacktestValuationSnapshot" not in source


def test_valuation_owns_no_fingerprint_or_hash_domain():
    """Identifier-level check, so prose about Phase 15D cannot mask a leak."""

    identifiers: set[str] = set()
    for path in VALUATION.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr.lower())
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                identifiers.add(node.name.lower())
            elif isinstance(node, ast.Import):
                identifiers.update(
                    alias.name.split(".")[0].lower() for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                identifiers.update(
                    alias.name.lower() for alias in node.names
                )

    for token in ("sha256", "fingerprint", "hash_domain", "snapshot", "hashlib"):
        assert not any(token in identifier for identifier in identifiers), token


def test_the_audit_package_still_owns_snapshot_and_policy_identity():
    for name in (
        "HistoricalBacktestValuationSnapshot",
        "build_valuation_snapshot",
        "HistoricalBacktestValuationPolicyRef",
        "build_valuation_policy_ref",
    ):
        assert hasattr(backtest_results_models, name)
    from stock_swing_d1.backtest_results import hashing

    assert (
        hashing.VALUATION_POLICY_HASH_DOMAIN
        == "historical_backtest_valuation_policy.v0.1"
    )
    assert (
        hashing.VALUATION_SNAPSHOT_HASH_DOMAIN
        == "historical_backtest_valuation_snapshot.v0.1"
    )


def test_relocated_policy_and_mark_are_the_same_class_objects():
    assert (
        backtest_results_models.HistoricalBacktestValuationPolicy
        is PortfolioValuationPolicy
    )
    assert (
        backtest_results_models.HistoricalBacktestValuationMark
        is PortfolioValuationMark
    )
    assert (
        backtest_results_package.HistoricalBacktestValuationMark
        is PortfolioValuationMark
    )


def test_exactly_one_implementation_of_the_equity_formula():
    """Only the shared owner multiplies a mark by a position quantity."""

    offenders = []
    for path in SRC.rglob("*.py"):
        if VALUATION in path.parents:
            continue
        source = path.read_text(encoding="utf-8")
        if "exact_decimal_times_int(mark.close" in source and (
            "pending_receivable_value" in source
            or "portfolio_equity" in source
        ):
            # Phase 15D still owns unrealized P&L, which needs cost basis;
            # what it must not do is recompute equity itself.
            if "settled_cash, pending_value" in source:
                offenders.append(path.name)
    assert offenders == []


def test_no_protective_exit_formula_is_copied_into_phase15a():
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in BACKTESTER.rglob("*.py")
    )
    for token in (
        "take_profit_price =",
        "stop_price =",
        "risk_fraction =",
        "2.0 * initial_risk",
        "atr_fraction *",
    ):
        assert token not in source
