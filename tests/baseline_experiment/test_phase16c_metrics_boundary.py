"""Phase 16C consumes Phase 16B; it never recomputes or repairs it."""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.research_metrics.reporting import (
    project_research_metrics_report,
)

from stock_swing_d1.baseline_experiment import (
    CANONICAL_BASELINE_EXPERIMENT_INTEGRITY_REPORT_SCHEMA_VERSION,
    CANONICAL_BASELINE_EXPERIMENT_REPORT_SCHEMA_VERSION,
    CANONICAL_BASELINE_EXPERIMENT_RESULT_HASH_DOMAIN,
    CANONICAL_BASELINE_EXPERIMENT_RESULT_SCHEMA_VERSION,
    UNAVAILABLE_AUTHORITATIVE_REPORT_METRICS,
    build_canonical_baseline_experiment_result,
    build_earnings_filter_exclusion_diagnostic,
    build_material_adverse_overnight_gap_diagnostic,
    build_material_adverse_overnight_gap_policy,
    project_canonical_baseline_experiment_report,
    verify_canonical_baseline_experiment_integrity,
)

from tests.baseline_experiment.conftest import (
    build_synthetic_audit_result,
    build_synthetic_manifest,
    build_synthetic_metrics_artifact,
)


PRODUCTION = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "stock_swing_d1"
    / "baseline_experiment"
)


def production_paths():
    return tuple(sorted(PRODUCTION.glob("*.py")))


def owned_names(tree: ast.AST):
    """Every identifier Phase 16C code actually uses, excluding prose."""

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


@pytest.fixture
def preserved_result():
    source_result = build_synthetic_audit_result()
    return build_canonical_baseline_experiment_result(
        manifest=build_synthetic_manifest(),
        source_audit_result_fingerprint=source_result.result_fingerprint,
        research_metrics_artifact=build_synthetic_metrics_artifact(
            source_result
        ),
        gap_diagnostic=build_material_adverse_overnight_gap_diagnostic(
            source_result=source_result,
            policy=build_material_adverse_overnight_gap_policy(),
            trade_gap_evidence=(),
        ),
        earnings_exclusion_diagnostic=(
            build_earnings_filter_exclusion_diagnostic(
                source_result=source_result
            )
        ),
    )


def test_phase16c_never_imports_or_calls_the_phase16b_calculation_boundary():
    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in owned_names(tree):
            if name in {
                "calculate_research_metrics",
                "stock_swing_d1.research_metrics.calculation",
            } or name.endswith(".calculation"):
                offenders.append((path.name, name))
    assert offenders == []


def test_phase16c_reimplements_no_phase16b_statistic():
    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in owned_names(tree):
            lowered = name.lower()
            for concept in (
                "sharpe",
                "sortino",
                "drawdown",
                "volatility",
                "cagr",
                "profit_factor",
                "expectancy",
                "win_rate",
                "total_return",
            ):
                if concept in lowered:
                    offenders.append((path.name, name))
    assert offenders == []


def test_the_phase16b_result_is_consumed_directly(preserved_result):
    artifact = preserved_result.research_metrics_artifact
    report = project_canonical_baseline_experiment_report(preserved_result)
    authoritative = artifact.research_metrics_result

    assert report.trades == authoritative.trade_metrics
    assert report.performance == project_research_metrics_report(artifact)
    assert (
        report.performance.strategy.total_return
        == authoritative.strategy_metrics.total_return
    )
    assert (
        report.performance.strategy.cagr == authoritative.strategy_metrics.cagr
    )
    assert (
        report.performance.strategy.maximum_drawdown
        == authoritative.strategy_metrics.maximum_drawdown
    )
    assert (
        report.performance.strategy.sharpe_ratio
        == authoritative.strategy_metrics.sharpe_ratio
    )
    assert (
        report.performance.strategy.sortino_ratio
        == authoritative.strategy_metrics.sortino_ratio
    )
    assert (
        report.performance.benchmark.total_return
        == authoritative.benchmark_metrics.total_return
    )
    assert (
        report.performance.comparative.ending_wealth_ratio
        == authoritative.relative_metrics.ending_wealth_ratio
    )
    assert (
        report.identity.research_metrics_result_fingerprint
        == authoritative.result_fingerprint
    )


def test_phase16c_versions_only_changed_payload_contracts(preserved_result):
    report = project_canonical_baseline_experiment_report(preserved_result)
    integrity = verify_canonical_baseline_experiment_integrity(preserved_result)

    assert CANONICAL_BASELINE_EXPERIMENT_RESULT_SCHEMA_VERSION == (
        "canonical_baseline_experiment_result.v0.2"
    )
    assert CANONICAL_BASELINE_EXPERIMENT_RESULT_HASH_DOMAIN == (
        CANONICAL_BASELINE_EXPERIMENT_RESULT_SCHEMA_VERSION
    )
    assert preserved_result.schema_version == (
        CANONICAL_BASELINE_EXPERIMENT_RESULT_SCHEMA_VERSION
    )
    assert CANONICAL_BASELINE_EXPERIMENT_REPORT_SCHEMA_VERSION == (
        "canonical_baseline_experiment_report.v0.2"
    )
    assert report.schema_version == (
        CANONICAL_BASELINE_EXPERIMENT_REPORT_SCHEMA_VERSION
    )
    # The integrity-check inventory and payload did not change; its opaque
    # result fingerprint remains independently domain-bound by the result.
    assert CANONICAL_BASELINE_EXPERIMENT_INTEGRITY_REPORT_SCHEMA_VERSION == (
        "canonical_baseline_experiment_integrity_report.v0.1"
    )
    assert integrity.schema_version == (
        CANONICAL_BASELINE_EXPERIMENT_INTEGRITY_REPORT_SCHEMA_VERSION
    )


def test_the_phase16b3_artifact_and_fingerprint_are_preserved(
    preserved_result,
):
    artifact = preserved_result.research_metrics_artifact
    report = project_canonical_baseline_experiment_report(preserved_result)
    assert (
        report.identity.research_metrics_artifact_fingerprint
        == artifact.artifact_fingerprint
    )
    integrity = verify_canonical_baseline_experiment_integrity(preserved_result)
    assert integrity.integrity_verified is True
    names = {check.check_id for check in integrity.checks}
    assert "metrics_artifact_round_trips_to_identical_bytes" in names
    assert "metrics_result_identity_preserved" in names


def test_no_frozen_report_item_lacks_an_authoritative_field(preserved_result):
    """The four formerly unavailable items are now authoritative Phase 16B fields.

    Phase 16C names what Phase 16B does not publish and never invents it. Since
    the Phase 16B Amendment v0.1 there is nothing left to name, so the
    declaration is empty and the report says so rather than continuing to claim
    an unavailability that no longer exists.
    """

    assert UNAVAILABLE_AUTHORITATIVE_REPORT_METRICS == ()

    report = project_canonical_baseline_experiment_report(preserved_result)
    assert report.unavailable_authoritative_metrics == ()
    for item in (
        "average_winner_usd",
        "average_loser_usd",
        "worst_trade_usd",
        "average_holding_sessions",
    ):
        assert item in type(report.trades).model_fields


def test_the_four_amendment_metrics_are_consumed_verbatim(preserved_result):
    """Phase 16C copies all four authoritative values; it derives none of them."""

    authoritative = (
        preserved_result.research_metrics_artifact.research_metrics_result
    ).trade_metrics
    report = project_canonical_baseline_experiment_report(preserved_result)

    for name in (
        "average_winner_usd",
        "average_loser_usd",
        "worst_trade_usd",
        "average_holding_sessions",
    ):
        projected = getattr(report.trades, name)
        expected = getattr(authoritative, name)
        assert projected is expected
        assert projected.value == expected.value
        assert projected.value.as_tuple() == expected.value.as_tuple()

    # The signed-loss and minimum-signed semantics survive the boundary.
    assert report.trades.average_loser_usd.value < Decimal("0")
    assert report.trades.average_winner_usd.value > Decimal("0")


def test_phase16c_does_not_reimplement_the_four_amendment_metrics():
    """No Phase 16C identifier names or recomputes one of the four metrics."""

    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in owned_names(tree):
            lowered = name.lower()
            for concept in (
                "average_winner",
                "average_loser",
                "worst_trade",
                "average_holding",
                "holding_session",
                "net_realized",
            ):
                if concept in lowered:
                    offenders.append((path.name, name))
    assert offenders == []


def test_no_percentage_trade_return_semantics_are_introduced():
    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in owned_names(tree):
            lowered = name.lower()
            if "trade_return" in lowered or "pct_return" in lowered:
                offenders.append((path.name, name))
    assert offenders == []


def test_no_equity_series_is_repaired_or_reconstructed():
    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in owned_names(tree):
            lowered = name.lower()
            for concept in (
                "forward_fill",
                "ffill",
                "backfill",
                "bfill",
                "interpolat",
                "fillna",
                "reindex",
                "resample",
                "equity_curve",
                "session_transitions",
            ):
                if concept in lowered:
                    offenders.append((path.name, name))
    assert offenders == []


def test_phase16c_imports_no_dataframe_or_provider_library():
    offenders = []
    for path in production_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or "",)
            for module in modules:
                if any(
                    module == token or module.startswith(f"{token}.")
                    for token in ("pandas", "numpy", "norgatedata", "pyarrow")
                ):
                    offenders.append((path.name, module))
    assert offenders == []


def test_tampering_with_a_preserved_identity_is_detected(preserved_result):
    with pytest.raises(ValueError):
        type(preserved_result)(
            **{
                **{
                    name: getattr(preserved_result, name)
                    for name in type(preserved_result).model_fields
                },
                "result_fingerprint": "0" * 64,
            }
        )


def test_a_metrics_artifact_from_another_run_is_rejected():
    source_result = build_synthetic_audit_result()
    other_result = build_synthetic_audit_result(
        final_equity=source_result.final_equity + 1
    )
    with pytest.raises(ValueError):
        build_canonical_baseline_experiment_result(
            manifest=build_synthetic_manifest(),
            source_audit_result_fingerprint=other_result.result_fingerprint,
            research_metrics_artifact=build_synthetic_metrics_artifact(
                source_result
            ),
            gap_diagnostic=build_material_adverse_overnight_gap_diagnostic(
                source_result=other_result,
                policy=build_material_adverse_overnight_gap_policy(),
                trade_gap_evidence=(),
            ),
            earnings_exclusion_diagnostic=(
                build_earnings_filter_exclusion_diagnostic(
                    source_result=other_result
                )
            ),
        )


def test_integrity_verification_is_deterministic(preserved_result):
    first = verify_canonical_baseline_experiment_integrity(preserved_result)
    second = verify_canonical_baseline_experiment_integrity(preserved_result)
    assert first == second
