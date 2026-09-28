"""Phase 16C canonical experiment manifest determinism and fail-closed rules."""

from __future__ import annotations

import ast
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.backtester.decision_interval import (
    HistoricalDecisionInterval,
)
from stock_swing_d1.research_metrics.hashing import (
    RESEARCH_METRICS_RESULT_HASH_DOMAIN,
)
from stock_swing_d1.research_metrics.policy import (
    build_performance_measurement_policy,
)

from stock_swing_d1.baseline_experiment import (
    CANONICAL_BASELINE_EXPERIMENT_MANIFEST_HASH_DOMAIN,
    CANONICAL_BASELINE_EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    BaselineExperimentValidationError,
    CanonicalBaselineExperimentManifest,
    build_material_adverse_overnight_gap_policy,
    compute_manifest_fingerprint,
)

from tests.baseline_experiment.conftest import (
    DECISION_END,
    DECISION_START,
    SYNTHETIC_GIT_COMMIT,
    build_synthetic_manifest,
    synthetic_artifact_ref,
)


PRODUCTION = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "stock_swing_d1"
    / "baseline_experiment"
)


def test_manifest_construction_is_deterministic():
    first = build_synthetic_manifest()
    second = build_synthetic_manifest()
    assert first == second
    assert first.manifest_fingerprint == second.manifest_fingerprint


def test_manifest_fingerprint_covers_every_semantic_field_but_itself():
    manifest = build_synthetic_manifest()
    assert manifest.manifest_fingerprint == compute_manifest_fingerprint(
        manifest
    )
    fields = set(type(manifest).model_fields)
    assert "manifest_fingerprint" in fields
    # Nothing volatile is declared, so nothing beyond the fingerprint itself
    # is excluded from the digest.
    assert not fields & {
        "generated_at",
        "created_at",
        "run_id",
        "output_path",
        "attempt",
    }


def test_a_changed_semantic_field_changes_the_fingerprint():
    manifest = build_synthetic_manifest()
    other = build_synthetic_manifest(
        decision_interval=HistoricalDecisionInterval(
            decision_start_date=DECISION_START,
            decision_end_date=date(2025, 12, 30),
        )
    )
    assert other.manifest_fingerprint != manifest.manifest_fingerprint


def test_a_tampered_fingerprint_is_rejected():
    manifest = build_synthetic_manifest()
    with pytest.raises(ValueError):
        CanonicalBaselineExperimentManifest(
            **{
                **{
                    name: getattr(manifest, name)
                    for name in CanonicalBaselineExperimentManifest.model_fields
                },
                "manifest_fingerprint": "0" * 64,
            }
        )


def test_git_commit_identity_is_preserved_exactly():
    manifest = build_synthetic_manifest()
    assert manifest.git_commit_sha == SYNTHETIC_GIT_COMMIT
    assert manifest.git_identity_kind == "git_commit"
    rebound = build_synthetic_manifest(git_commit_sha="b" * 40)
    assert rebound.git_commit_sha == "b" * 40
    assert rebound.manifest_fingerprint != manifest.manifest_fingerprint


def test_short_or_uppercase_git_identity_is_rejected():
    for invalid in ("f7aca3a", ("A" * 40), "z" * 40, "a" * 39):
        with pytest.raises((ValueError, BaselineExperimentValidationError)):
            build_synthetic_manifest(git_commit_sha=invalid)


def test_the_exact_decision_interval_is_preserved():
    manifest = build_synthetic_manifest()
    assert manifest.decision_interval.decision_start_date == DECISION_START
    assert manifest.decision_interval.decision_end_date == DECISION_END


def test_exactly_usd_one_hundred_thousand_capital_is_preserved():
    manifest = build_synthetic_manifest()
    assert manifest.starting_capital == Decimal("100000")
    assert manifest.base_currency == "USD"
    for invalid in (Decimal("99999.99"), Decimal("100001"), Decimal("0")):
        with pytest.raises(ValueError):
            build_synthetic_manifest(starting_capital=invalid)


def test_policy_and_artifact_fingerprints_are_preserved():
    manifest = build_synthetic_manifest()
    policy = build_performance_measurement_policy()
    gap_policy = build_material_adverse_overnight_gap_policy()
    assert (
        manifest.performance_measurement_policy_ref.policy_fingerprint
        == policy.policy_fingerprint
    )
    assert (
        manifest.gap_policy_ref.policy_fingerprint
        == gap_policy.policy_fingerprint
    )
    assert (
        manifest.expected_research_metrics_result_hash_domain
        == RESEARCH_METRICS_RESULT_HASH_DOMAIN
    )
    assert manifest.gap_policy_ref.materiality_threshold == "-0.010000"
    assert manifest.gap_policy_ref.trade_counting_rule == "one_per_trade"
    assert (
        manifest.earnings_exclusion_diagnostic_version
        == "earnings_filter_exclusions.v0.1"
    )
    for ref in (
        manifest.strategy_configuration_ref,
        manifest.universe_artifact_ref,
        manifest.market_data_artifact_ref,
        manifest.corporate_action_artifact_ref,
        manifest.dividend_evidence_ref,
        manifest.earnings_pit_artifact_ref,
    ):
        assert len(ref.content_sha256) == 64


def test_every_missing_required_evidence_item_fails_closed():
    required = (
        "git_commit_sha",
        "decision_interval",
        "starting_capital",
        "strategy_configuration_ref",
        "universe_artifact_ref",
        "market_data_artifact_ref",
        "corporate_action_artifact_ref",
        "dividend_evidence_ref",
        "earnings_pit_artifact_ref",
        "earnings_provider_name",
        "allocation_policy_ref",
        "ranking_policy_ref",
        "execution_cost_policy_ref",
        "valuation_policy_ref",
        "run_configuration_fingerprint",
        "benchmark_id",
        "benchmark_source_artifact_ref",
        "benchmark_series_fingerprint",
        "performance_measurement_policy_ref",
        "gap_policy_ref",
    )
    for name in required:
        with pytest.raises(BaselineExperimentValidationError) as error:
            build_synthetic_manifest(**{name: None})
        assert error.value.code == "INCOMPLETE_MANIFEST_EVIDENCE"
        assert name in str(error.value)


def test_the_manifest_hash_domain_is_versioned_and_separate():
    assert CANONICAL_BASELINE_EXPERIMENT_MANIFEST_SCHEMA_VERSION == (
        "canonical_baseline_experiment_manifest.v0.2"
    )
    assert CANONICAL_BASELINE_EXPERIMENT_MANIFEST_HASH_DOMAIN == (
        "canonical_baseline_experiment_manifest.v0.2"
    )
    assert build_synthetic_manifest().schema_version == (
        CANONICAL_BASELINE_EXPERIMENT_MANIFEST_SCHEMA_VERSION
    )
    assert (
        CANONICAL_BASELINE_EXPERIMENT_MANIFEST_HASH_DOMAIN
        != RESEARCH_METRICS_RESULT_HASH_DOMAIN
    )


def test_manifest_construction_has_no_clock_or_nondeterminism_dependence():
    source = (PRODUCTION / "manifest.py").read_text(encoding="utf-8").lower()
    for token in (
        "datetime.now",
        "datetime.utcnow",
        "date.today",
        "time.time",
        "uuid",
        "random",
        "generated_at",
        "output_path",
        "hash(",
    ):
        assert token not in source
    tree = ast.parse((PRODUCTION / "manifest.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not imported & {"time", "os", "socket", "getpass", "platform"}


def test_a_degenerate_interval_is_rejected():
    with pytest.raises(ValueError):
        build_synthetic_manifest(
            decision_interval=HistoricalDecisionInterval(
                decision_start_date=DECISION_START,
                decision_end_date=DECISION_START,
            )
        )


def test_unknown_manifest_fields_are_forbidden():
    manifest = build_synthetic_manifest()
    with pytest.raises(ValueError):
        CanonicalBaselineExperimentManifest(
            **{
                name: getattr(manifest, name)
                for name in CanonicalBaselineExperimentManifest.model_fields
            },
            experiment_note=synthetic_artifact_ref("note", "1"),
        )
