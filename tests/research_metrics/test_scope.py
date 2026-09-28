"""Phase 16B production-scope guards.

Phase 16B.2 v0.2.1 partitions -- never weakens -- the Phase 16B.1 guards.

* The seven foundation modules keep the statistical-arithmetic AST prohibition
  *without exception* (Clauses 50/55). Gaining policy fields, schema fields,
  enum vocabulary, version identities and validation declarations does not
  authorize a single arithmetic operation there.
* ``arithmetic.py`` and ``calculation.py`` are the only modules that may
  compute (Clauses 51-53, 56). The authorized statistical concepts are lifted
  for them and for them alone.
* Every economic-reconstruction and numerical-implementation prohibition stays
  active across all nine modules, the two calculation modules included
  (Clauses 57/58).
* The endpoint-token guards are narrowed to their actual ownership purpose
  (Clause 59): authorized Phase 16B.2 code may *read* authoritative upstream
  evidence, and may still never own, name, or recompute it.
"""

from __future__ import annotations

import ast
from enum import Enum
from pathlib import Path

from stock_swing_d1.research_metrics import models as models_module
from stock_swing_d1.research_metrics import (
    build_benchmark_performance_series,
)


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "src" / "stock_swing_d1" / "research_metrics"

FOUNDATION_MODULES = (
    "__init__.py",
    "canonical.py",
    "errors.py",
    "hashing.py",
    "models.py",
    "policy.py",
    "validation.py",
)
CALCULATION_MODULES = ("arithmetic.py", "calculation.py")
PERSISTENCE_MODULES = ("persistence.py", "reporting.py")

_ARITHMETIC_OPERATORS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)

# Names that perform a numerical operation rather than name a schema field.
# A foundation module may declare a field called ``annualized_volatility``;
# it may not call one of these.
_COMPUTATIONAL_CALL_NAMES = frozenset(
    {
        "sqrt",
        "square_root",
        "ln",
        "log",
        "log10",
        "logb",
        "exp",
        "exponential",
        "natural_logarithm",
        "pow",
        "power",
        "mean",
        "stdev",
        "fmean",
        "median",
        "quantize",
        "quantize_canonical_metric",
        "add",
        "subtract",
        "multiply",
        "divide",
        "divide_int",
        "remainder",
        "fma",
        "sum",
        "max",
        "min",
        "add_exact_decimal",
        "subtract_exact_decimal",
        "exact_decimal_times_int",
        "sum_exact_decimal",
    }
)

# Authoritative upstream evidence Phase 16B.2 is explicitly permitted to read
# (Clause 59) but never to own as a field of its own or to recompute.
_AUTHORITATIVE_EVIDENCE_NAMES = frozenset(
    {
        "initial_equity",
        "final_equity",
        "decision_interval",
        "decision_start_date",
        "decision_end_date",
        "equity_curve",
        "session_transitions",
        "processed_session_count",
        "trade_total_pnl",
        "realized_pnl",
        "ordinary_dividend_income",
    }
)


def _production_paths() -> tuple[Path, ...]:
    return tuple(sorted(PRODUCTION.glob("*.py")))


def _paths(names: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(PRODUCTION / name for name in names)


def _text(paths: tuple[Path, ...]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _production_text() -> str:
    return _text(_production_paths())


def _called_names(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            yield function.id
        elif isinstance(function, ast.Attribute):
            yield function.attr


def test_phase16b_has_exactly_the_scoped_production_files():
    """Phase 16B.3 adds only persistence and report-adaptation modules."""

    assert {path.name for path in _production_paths()} == {
        *FOUNDATION_MODULES,
        *CALCULATION_MODULES,
        *PERSISTENCE_MODULES,
    }
    assert (
        len(FOUNDATION_MODULES)
        + len(CALCULATION_MODULES)
        + len(PERSISTENCE_MODULES)
        == 11
    )


def test_production_has_no_provider_or_upstream_owner_imports():
    forbidden_modules = (
        "norgatedata",
        "norgate",
        "norgate_data",
        "stock_swing_d1.data.norgate_",
        "stock_swing_d1.portfolio",
        "stock_swing_d1.backtester",
        "stock_swing_d1.strategy",
        "stock_swing_d1.ranking",
        "stock_swing_d1.risk.position_sizing",
        "stock_swing_d1.execution",
    )
    offenders = []
    for path in _production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or "",)
            for module in modules:
                if (
                    path.name in PERSISTENCE_MODULES
                    and module
                    == "stock_swing_d1.backtester.decision_interval"
                ):
                    continue
                if module != "stock_swing_d1.backtest_results" and any(
                    module == token or module.startswith(token)
                    for token in forbidden_modules
                ):
                    offenders.append((path.name, module))
    assert offenders == []


def test_production_does_not_call_upstream_owners_or_pipelines():
    source = _production_text()
    for token in (
        "PortfolioTransitionEngine",
        "HistoricalBacktestOrchestrator",
        "PortfolioBacktestOrchestrator",
        "rank_candidates(",
        "allocate_ranked_candidates(",
        "size_pending_entry(",
        "execute_pending_entry(",
        "evaluate_signal(",
        "OpenPositionExitEvaluator(",
        "d1_pipeline",
        "project_equity_curve",
        "project_closed_trades",
        "validate_historical_backtest_source_run",
        "HistoricalBacktestResultService",
    ):
        assert token not in source


def test_models_have_no_clock_random_uuid_cache_or_path_dependence():
    source = _production_text().lower()
    for token in (
        "datetime.now",
        "date.today",
        "time.time",
        "uuid",
        "random",
        "generated_at",
        "output_path",
        "filesystem_path",
        "privateattr",
        "cached_property",
        "hash(",
        "decimal(float",
    ):
        assert token not in source


def test_the_seven_foundation_modules_contain_no_statistical_arithmetic():
    """SCOPE-03: the Phase 16B.1 AST prohibition survives without exception."""

    offenders = []
    for path in _paths(FOUNDATION_MODULES):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(
                node.op, _ARITHMETIC_OPERATORS
            ):
                offenders.append((path.name, "operator", node.lineno))
        for name in _called_names(tree):
            if name in _COMPUTATIONAL_CALL_NAMES:
                offenders.append((path.name, "call", name))
    assert offenders == []


def test_statistical_computation_is_confined_to_the_calculation_boundary():
    """SCOPE-02/SCOPE-06: only the two authorized modules may compute."""

    foundation = _text(_paths(FOUNDATION_MODULES)).lower()
    for token in (
        "sqrt",
        "ln(",
        "log(",
        "exp(",
        "mean(",
        "stdev",
        "getcontext",
        "localcontext",
        "decimal.context",
    ):
        assert token not in foundation

    calculation = _text(_paths(CALCULATION_MODULES)).lower()
    for authorized in (
        "sqrt",
        "ln(",
        "exp(",
        "mean(",
        "volatility",
        "sharpe",
        "sortino",
        "drawdown",
    ):
        assert authorized in calculation


def test_no_economic_reconstruction_in_phase16b1_or_phase16b2():
    """SCOPE-04/SCOPE-05/SCOPE-07 remain intact in their frozen modules."""

    source = _text(_paths((*FOUNDATION_MODULES, *CALCULATION_MODULES))).lower()
    for token in (
        "reinvest",
        "adjusted_close",
        "unadjusted",
        "expense_ratio",
        "drip",
        "quantity",
        "market_price",
        "corporate_action",
        "initial_capital",
        "starting_capital",
        "benchmark_initial",
        "initial_wealth",
    ):
        assert token not in source


def test_the_calculation_boundary_uses_no_binary_float_or_ambient_context():
    """AR-04, CAGR-06 and Clause 58, enforced structurally rather than by prose.

    The checks are AST-based so that a comment *describing* the prohibition
    cannot trip them and, more importantly, so that an actual call cannot hide
    behind an alias or a line break.
    """

    forbidden_calls = frozenset(
        {"float", "setcontext", "getcontext", "localcontext", "pow", "power"}
    )
    forbidden_imports = ("math", "numpy", "pandas")
    offenders = []
    for path in _production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _called_names(tree):
            if name in forbidden_calls:
                offenders.append((path.name, "call", name))
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(
                node.op, ast.Pow
            ):
                offenders.append((path.name, "exponentiation", node.lineno))
            modules = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or "",)
            for module in modules:
                if any(
                    module == token or module.startswith(f"{token}.")
                    for token in forbidden_imports
                ):
                    offenders.append((path.name, "import", module))
    assert offenders == []
    assert "Decimal(float" not in _production_text()


def test_no_multi_precision_convergence_ladder_exists():
    """AR-13: determinism comes from one pinned context, not from agreement.

    The canonical working precision is one frozen integer constant bound once
    into one context, so there is no sequence of precisions to iterate, and no
    agreement-across-precisions stopping rule can exist.
    """

    from stock_swing_d1.research_metrics import arithmetic as arithmetic_module

    assert type(arithmetic_module.STATISTICAL_WORKING_PRECISION) is int
    assert arithmetic_module.STATISTICAL_WORKING_PRECISION == 50

    precision_bindings = []
    for path in _production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "prec":
                bound = node.value
                precision_bindings.append(
                    (
                        path.name,
                        bound.id if isinstance(bound, ast.Name) else "literal",
                    )
                )
            if isinstance(node, ast.Attribute) and node.attr == "prec":
                precision_bindings.append((path.name, "attribute"))
    assert precision_bindings == [
        ("arithmetic.py", "STATISTICAL_WORKING_PRECISION")
    ]

    contexts = tuple(
        arithmetic_module.canonical_statistical_context() for _ in range(3)
    )
    assert {context.prec for context in contexts} == {50}


def test_production_has_no_value_kind_or_replacement_discriminator():
    source = _production_text()
    for token in (
        "BenchmarkValueKind",
        "value_kind",
        "PRICE_INDEX",
        "price_index",
        "TOTAL_RETURN_INDEX",
        "total_return_index",
        "benchmark_type",
        "equity_kind",
        "return_kind",
        "performance_kind",
    ):
        assert token not in source

    enums = [
        name
        for name, member in vars(models_module).items()
        if isinstance(member, type)
        and issubclass(member, Enum)
        and member.__module__ == models_module.__name__
    ]
    assert enums == []


def test_the_foundation_layer_never_touches_authoritative_upstream_evidence():
    """The Phase 16B.1 field-ownership guard, kept where it belongs.

    A foundation module may neither name nor read one of these: it declares
    contracts, it does not consume audited economics. Only the two authorized
    calculation modules may read them, and only as attribute reads.
    """

    offenders = []
    for path in _paths(FOUNDATION_MODULES):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            touched = None
            if isinstance(node, ast.Name):
                touched = node.id
            elif isinstance(node, ast.Attribute):
                touched = node.attr
            elif isinstance(node, ast.arg):
                touched = node.arg
            elif isinstance(node, ast.keyword):
                touched = node.arg
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                touched = node.name
            elif isinstance(node, ast.Constant) and type(node.value) is str:
                touched = node.value
            if touched in _AUTHORITATIVE_EVIDENCE_NAMES:
                offenders.append((path.name, touched, node.lineno))
    assert offenders == []


def test_the_calculation_layer_only_reads_authoritative_upstream_evidence():
    """SCOPE-08: reading authoritative evidence is not economic reconstruction.

    Phase 16B.2 must consume ``initial_equity``, ``final_equity``,
    ``decision_interval``, ``equity_curve``, the authoritative session-transition
    evidence and ``trade_total_pnl``. The narrowed guard therefore permits those
    names to appear only as attribute reads on an upstream object -- never as a
    Phase 16B binding, parameter, keyword or field of its own, which is what
    taking ownership of an upstream economic fact would look like.
    """

    offenders = []
    for path in _paths(CALCULATION_MODULES):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            owned = None
            if isinstance(node, ast.Name):
                owned = node.id
            elif isinstance(node, ast.arg):
                owned = node.arg
            elif isinstance(node, ast.keyword):
                owned = node.arg
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                owned = node.name
            elif isinstance(node, ast.Attribute) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                owned = node.attr
            if owned in _AUTHORITATIVE_EVIDENCE_NAMES:
                offenders.append((path.name, owned, node.lineno))
    assert offenders == []


def test_production_adds_no_benchmark_initial_capital_or_interval_field():
    """B0 is never duplicated into the frozen benchmark evidence model."""

    from stock_swing_d1.research_metrics import (
        BenchmarkPerformanceObservation,
        BenchmarkPerformanceSeries,
    )

    for model in (BenchmarkPerformanceSeries, BenchmarkPerformanceObservation):
        for field_name in model.model_fields:
            assert field_name not in {
                "initial_capital",
                "starting_capital",
                "benchmark_initial_equity",
                "initial_equity",
                "initial_wealth",
                "b0",
                "decision_interval",
            }

    # RET/Clause 17: null-first semantics never add an observation-level
    # return field to a frozen upstream observation model.
    assert not {"daily_return", "periodic_return", "return"} & set(
        BenchmarkPerformanceObservation.model_fields
    )


def test_production_embeds_no_benchmark_identity_denylist_or_substitution_check(
    benchmark_source_ref, benchmark_series
):
    """Baseline benchmark identity is enforced outside Phase 16B.

    Phase 16B has no structural property distinguishing the optional
    reinvested contextual reference from the authorized primary benchmark, so
    it embeds no denylisted identity string and performs no identity-based
    primary-versus-alternative check. Requiring the authoritative baseline
    identity belongs to the applicable experiment/validation policy layer.
    """

    source = _production_text().lower()
    for token in (
        "spy",
        "denylist",
        "blocklist",
        "blacklist",
        "allowlist",
        "forbidden_benchmark",
        "secondary",
        "substitut",
    ):
        assert token not in source

    generic = build_benchmark_performance_series(
        benchmark_id="SOME_OTHER_BENCHMARK_IDENTITY_V9_9",
        source_artifact_ref=benchmark_source_ref,
        observations=benchmark_series.observations,
    )
    assert generic.benchmark_id == "SOME_OTHER_BENCHMARK_IDENTITY_V9_9"


def test_all_phase16b_fixtures_are_explicitly_synthetic(
    synthetic_audit_result, canonical_audit_result, benchmark_series
):
    for result in (synthetic_audit_result, canonical_audit_result):
        manifest = result.run_manifest
        assert manifest.strategy_configuration_ref.build_id == (
            "synthetic-fixture"
        )
        assert manifest.universe_artifact_ref.build_id == "synthetic-fixture"
        assert manifest.market_data_artifact_ref.build_id == "synthetic-fixture"
    assert benchmark_series.source_artifact_ref.build_id == "synthetic-fixture"
