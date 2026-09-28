"""Phase 16C production-scope guards.

Phase 16C is an orchestration/experiment-preservation layer. These guards keep
it from becoming a second backtester, metrics calculator, portfolio engine,
benchmark engine, earnings engine or corporate-action engine, and keep every
prohibited nondeterminism source structurally out of it.

Identifier checks are AST-based rather than textual so that a docstring
*stating* a prohibition cannot trip them, while an actual call, import,
parameter or field carrying a prohibited name always does.
"""

from __future__ import annotations

import ast
from pathlib import Path

PRODUCTION = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "stock_swing_d1"
    / "baseline_experiment"
)

SCOPED_MODULES = (
    "__init__.py",
    "canonical.py",
    "earnings_diagnostic.py",
    "errors.py",
    "execution.py",
    "field_types.py",
    "gap_diagnostic.py",
    "gap_policy.py",
    "hashing.py",
    "manifest.py",
    "preflight.py",
    "result.py",
)


def production_paths():
    return tuple(sorted(PRODUCTION.glob("*.py")))


def production_text():
    return "\n".join(
        path.read_text(encoding="utf-8") for path in production_paths()
    )


def identifiers(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.keyword) and node.arg is not None:
            yield node.arg
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""
            for alias in node.names:
                yield alias.name


def called_names(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            yield function.id
        elif isinstance(function, ast.Attribute):
            yield function.attr


def test_phase16c_has_exactly_the_scoped_production_files():
    assert {path.name for path in production_paths()} == set(SCOPED_MODULES)


def test_no_clock_random_or_generated_identifier_dependence():
    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in called_names(tree):
            if name in {
                "now",
                "utcnow",
                "today",
                "time",
                "time_ns",
                "monotonic",
                "uuid1",
                "uuid4",
                "getrandbits",
                "randint",
                "choice",
                "shuffle",
                "hash",
            }:
                offenders.append((path.name, "call", name))
        for name in identifiers(tree):
            lowered = name.lower()
            for token in ("uuid", "random", "generated_at", "created_at"):
                if token in lowered:
                    offenders.append((path.name, "name", name))
    assert offenders == []


def test_no_binary_float_construction_of_decimal_values():
    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in called_names(tree):
            if name in {"float", "getcontext", "setcontext", "localcontext"}:
                offenders.append((path.name, name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) is float:
                offenders.append((path.name, f"float literal line {node.lineno}"))
    assert offenders == []
    assert "Decimal(float" not in production_text()


def test_no_series_repair_or_search_vocabulary_exists_in_code():
    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in identifiers(tree):
            lowered = name.lower()
            for token in (
                "forward_fill",
                "ffill",
                "backfill",
                "bfill",
                "interpolat",
                "grid_search",
                "parameter_search",
                "hyperparameter",
                "optimiz",
                "parameter_sweep",
            ):
                if token in lowered:
                    offenders.append((path.name, name))
    assert offenders == []
    # The single "tuned" identifier in the package is the Gate-D guard that
    # requires a tuned-parameter count of exactly zero -- never a tuning knob.
    tuned = {
        name
        for path in production_paths()
        for name in identifiers(ast.parse(path.read_text(encoding="utf-8")))
        if "tuned" in name.lower()
    }
    assert tuned == {"tuned_parameter_count"}


def test_phase16c_never_imports_an_upstream_execution_owner():
    """Identity types may be bound; owning services may never be imported."""

    permitted_upstream_symbols = {
        "ArtifactRef",
        "PolicyArtifactRef",
        "RankingPolicyRef",
        "ExecutionCostPolicyRef",
        "HistoricalBacktestValuationPolicyRef",
        "HistoricalBacktestAuditResult",
        "HistoricalClosedTradeRecord",
        "HistoricalDecisionInterval",
        "EarningsIntegrationAction",
        "BROKER_NEUTRAL_POLICY_ID",
        "CANDIDATE_RANKING_POLICY_ID",
        "CANDIDATE_RANKING_POLICY_VERSION",
        "HISTORICAL_BACKTEST_VALUATION_POLICY_ID",
        "PORTFOLIO_ALLOCATION_POLICY_REF",
    }
    forbidden_service_symbols = {
        "HistoricalBacktestOrchestrator",
        "PortfolioBacktestOrchestrator",
        "PortfolioTransitionEngine",
        "PortfolioAllocationService",
        "CandidateRankingService",
        "BacktestExecutionCostService",
        "BaselineSignalEvaluator",
        "EntryExecutionService",
        "OpenPositionExitEvaluator",
        "PublishedEarningsPITQuery",
        "HistoricalBacktestResultService",
        "validate_historical_backtest_source_run",
        "project_equity_curve",
        "project_closed_trades",
        "run_d1_pipeline",
        "run_corporate_action_pipeline",
        "run_dividend_event_pipeline",
        "calculate_indicators",
    }
    imported_symbols = set()
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_symbols.update(
                    alias.name for alias in node.names
                )
    assert not imported_symbols & forbidden_service_symbols
    assert permitted_upstream_symbols <= imported_symbols


def test_phase16c_defines_no_second_backtest_or_benchmark_engine():
    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            lowered = node.name.lower()
            for token in (
                "orchestrat",
                "engine",
                "simulate",
                "backtest_session",
                "build_benchmark",
                "price_execution",
                "apply_corporate_action",
                "reconstruct",
            ):
                if token in lowered:
                    offenders.append((path.name, node.name))
    assert offenders == []


def test_every_phase16c_hash_domain_is_versioned_and_distinct():
    from stock_swing_d1.baseline_experiment import hashing

    domains = [
        value
        for name, value in vars(hashing).items()
        if name.endswith("_HASH_DOMAIN")
    ]
    assert len(domains) == len(set(domains))
    assert set(domains) == {
        "material_adverse_overnight_gap_policy.v0.1",
        "material_adverse_overnight_gap_diagnostic.v0.1",
        "earnings_filter_exclusion_diagnostic.v0.1",
        "canonical_baseline_experiment_manifest.v0.2",
        "canonical_baseline_experiment_result.v0.2",
    }


def test_every_phase16c_model_is_frozen_and_forbids_unknown_fields():
    import importlib

    from pydantic import BaseModel

    from stock_swing_d1.baseline_experiment.field_types import (
        ImmutableBaselineExperimentModel,
    )

    checked = 0
    for name in SCOPED_MODULES:
        if name == "__init__.py":
            continue
        module = importlib.import_module(
            f"stock_swing_d1.baseline_experiment.{name[:-3]}"
        )
        for member in vars(module).values():
            if (
                isinstance(member, type)
                and issubclass(member, BaseModel)
                and member.__module__ == module.__name__
                and member is not ImmutableBaselineExperimentModel
            ):
                assert member.model_config["frozen"] is True
                assert member.model_config["extra"] == "forbid"
                checked += 1
    assert checked > 20
