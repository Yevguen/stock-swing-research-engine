"""Phase 16C pre-run gate audit: one verdict, independently enumerated blockers."""

from __future__ import annotations

import ast
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.execution.costs.models import (
    BROKER_NEUTRAL_POLICY_ID,
    ExecutionCostPolicyRef,
)
from stock_swing_d1.ranking.models import RankingPolicyRef
from stock_swing_d1.research_metrics.persistence import (
    ResearchMetricsPolicyRef,
)

from stock_swing_d1.baseline_experiment import (
    GATE_IDS,
    GATE_NAMES,
    BenchmarkEvidence,
    CanonicalRunPreflightInput,
    CanonicalRunPreflightVerdict,
    CorporateActionEvidence,
    DecisionIntervalEvidence,
    DividendEvidence,
    EarningsPitEvidence,
    ExecutionSemanticsEvidence,
    MarketDataEvidence,
    MetricsPolicyEvidence,
    PreflightGateStatus,
    RepositoryIdentityEvidence,
    StartingCapitalEvidence,
    StrategyConfigurationEvidence,
    UniverseEvidence,
    evaluate_canonical_run_preflight,
)

from tests.baseline_experiment.conftest import (
    DECISION_END,
    DECISION_START,
    build_ready_preflight_input,
    build_synthetic_manifest,
)


PRODUCTION = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "stock_swing_d1"
    / "baseline_experiment"
)

# Which gate each preflight input member is responsible for.
GATE_BY_INPUT_FIELD = {
    "repository_identity": "A",
    "decision_interval_evidence": "B",
    "starting_capital_evidence": "C",
    "strategy_configuration_evidence": "D",
    "universe_evidence": "E",
    "market_data_evidence": "F",
    "corporate_action_evidence": "G",
    "dividend_evidence": "H",
    "earnings_pit_evidence": "I",
    "execution_semantics_evidence": "J",
    "benchmark_evidence": "K",
    "metrics_policy_evidence": "L",
    "persistence_readiness_probe": "M",
    "manifest": "N",
}


def gate(report, gate_id):
    return next(row for row in report.gates if row.gate_id == gate_id)


def test_a_fully_evidenced_preflight_is_ready(ready_preflight_input):
    report = evaluate_canonical_run_preflight(ready_preflight_input)
    assert report.verdict is (
        CanonicalRunPreflightVerdict.READY_FOR_CANONICAL_RUN
    )
    assert str(report.verdict) == "READY_FOR_CANONICAL_RUN"
    assert report.blockers == ()
    assert all(
        row.status is PreflightGateStatus.PASS for row in report.gates
    )
    assert tuple(row.gate_id for row in report.gates) == GATE_IDS
    assert len(GATE_IDS) == 14
    assert set(GATE_NAMES) == set("ABCDEFGHIJKLMN")


def test_an_empty_preflight_blocks_every_gate():
    report = evaluate_canonical_run_preflight(CanonicalRunPreflightInput())
    assert report.verdict is CanonicalRunPreflightVerdict.BLOCKED
    assert str(report.verdict) == "BLOCKED"
    assert all(
        row.status is PreflightGateStatus.BLOCKED for row in report.gates
    )
    assert len(report.blockers) >= len(GATE_IDS)
    assert all(
        blocker.startswith("BLOCKED — ") for blocker in report.blockers
    )


@pytest.mark.parametrize(
    ("field_name", "gate_id"), sorted(GATE_BY_INPUT_FIELD.items())
)
def test_every_required_gate_can_independently_block(field_name, gate_id):
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(**{field_name: None})
    )
    assert report.verdict is CanonicalRunPreflightVerdict.BLOCKED
    assert gate(report, gate_id).status is PreflightGateStatus.BLOCKED
    assert gate(report, gate_id).blockers


def test_multiple_blockers_are_all_reported():
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            earnings_pit_evidence=None,
            benchmark_evidence=None,
            corporate_action_evidence=None,
        )
    )
    joined = "\n".join(report.blockers)
    assert "canonical earnings PIT artifact unavailable" in joined
    assert "SPY benchmark evidence incomplete" in joined
    assert "canonical corporate-action evidence is unavailable" in joined
    assert len(report.blockers) >= 3


def test_an_unclean_working_tree_blocks_gate_a(ready_preflight_input):
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            repository_identity=RepositoryIdentityEvidence(
                reviewed_git_commit_sha="a" * 40,
                working_tree_clean=False,
                experiment_implementation_reviewed=True,
                required_tests_green=False,
            )
        )
    )
    outcome = gate(report, "A")
    assert outcome.status is PreflightGateStatus.BLOCKED
    assert len(outcome.blockers) == 2


def test_a_derived_or_post_hoc_interval_blocks_gate_b():
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            decision_interval_evidence=DecisionIntervalEvidence(
                decision_interval=build_synthetic_manifest().decision_interval,
                interval_source="frozen_authoritative_contract",
                derived_from_observed_sessions=True,
                selected_after_observing_results=True,
            )
        )
    )
    outcome = gate(report, "B")
    assert outcome.status is PreflightGateStatus.BLOCKED
    assert len(outcome.blockers) == 2


def test_a_non_frozen_capital_anchor_blocks_gate_c():
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            starting_capital_evidence=StartingCapitalEvidence(
                base_currency="USD", starting_capital=Decimal("250000")
            )
        )
    )
    assert gate(report, "C").status is PreflightGateStatus.BLOCKED


def test_tuned_strategy_parameters_block_gate_d(ready_preflight_input):
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            strategy_configuration_evidence=StrategyConfigurationEvidence(
                artifact_ref=(
                    ready_preflight_input.manifest.strategy_configuration_ref
                ),
                configuration_frozen=False,
                tuned_parameter_count=3,
            )
        )
    )
    outcome = gate(report, "D")
    assert outcome.status is PreflightGateStatus.BLOCKED
    assert len(outcome.blockers) == 2


def test_short_evidence_coverage_blocks_its_gate(ready_preflight_input):
    manifest = ready_preflight_input.manifest
    late_start = date(2025, 6, 1)
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            manifest=manifest,
            universe_evidence=UniverseEvidence(
                artifact_ref=manifest.universe_artifact_ref,
                point_in_time_membership=True,
                coverage_start=late_start,
                coverage_end=DECISION_END,
                validation_passed=True,
            ),
            market_data_evidence=MarketDataEvidence(
                artifact_ref=manifest.market_data_artifact_ref,
                timeframe="D1",
                session_type="regular",
                price_basis="unadjusted",
                currency="USD",
                coverage_start=DECISION_START,
                coverage_end=date(2025, 11, 30),
                dataset_validation_passed=True,
            ),
        )
    )
    assert gate(report, "E").status is PreflightGateStatus.BLOCKED
    assert gate(report, "F").status is PreflightGateStatus.BLOCKED
    joined = "\n".join(report.blockers)
    assert "universe evidence does not cover" in joined
    assert "market data does not cover" in joined


def test_an_unresolved_corporate_action_seam_blocks_gate_g(
    ready_preflight_input,
):
    manifest = ready_preflight_input.manifest
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            manifest=manifest,
            corporate_action_evidence=CorporateActionEvidence(
                artifact_ref=manifest.corporate_action_artifact_ref,
                coverage_start=DECISION_START,
                coverage_end=DECISION_END,
                unresolved_action_count=1,
                parity_validation_passed=True,
            ),
        )
    )
    assert gate(report, "G").status is PreflightGateStatus.BLOCKED


def test_unvalidated_dividend_evidence_blocks_gate_h(ready_preflight_input):
    manifest = ready_preflight_input.manifest
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            manifest=manifest,
            dividend_evidence=DividendEvidence(
                artifact_ref=manifest.dividend_evidence_ref,
                coverage_start=DECISION_START,
                coverage_end=DECISION_END,
                accounting_contract_version="synthetic.v0.1",
                validation_passed=False,
            ),
        )
    )
    assert gate(report, "H").status is PreflightGateStatus.BLOCKED


def test_a_look_ahead_earnings_substitution_blocks_gate_i(
    ready_preflight_input,
):
    manifest = ready_preflight_input.manifest
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            manifest=manifest,
            earnings_pit_evidence=EarningsPitEvidence(
                artifact_ref=manifest.earnings_pit_artifact_ref,
                provider_name=manifest.earnings_provider_name,
                coverage_start=DECISION_START,
                coverage_end=DECISION_END,
                point_in_time_reconstruction_validated=False,
                look_ahead_substitution_used=True,
            ),
        )
    )
    outcome = gate(report, "I")
    assert outcome.status is PreflightGateStatus.BLOCKED
    assert len(outcome.blockers) == 2


def test_a_foreign_allocation_or_ranking_policy_blocks_gate_j(
    ready_preflight_input,
):
    manifest = ready_preflight_input.manifest
    foreign_allocation = type(manifest.allocation_policy_ref)(
        policy_id=manifest.allocation_policy_ref.policy_id,
        policy_version=manifest.allocation_policy_ref.policy_version,
        policy_fingerprint="1" * 64,
    )
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            manifest=manifest,
            execution_semantics_evidence=ExecutionSemanticsEvidence(
                execution_cost_policy_ref=ExecutionCostPolicyRef(
                    policy_id=BROKER_NEUTRAL_POLICY_ID,
                    policy_fingerprint="5" * 64,
                ),
                settlement_model="historical_us_equity_standard_cycle",
                allocation_policy_ref=foreign_allocation,
                ranking_policy_ref=RankingPolicyRef(
                    policy_id="some_other_ranking_policy",
                    policy_version="9.9",
                    policy_fingerprint="4" * 64,
                ),
                valuation_policy_ref=manifest.valuation_policy_ref,
            ),
        )
    )
    outcome = gate(report, "J")
    assert outcome.status is PreflightGateStatus.BLOCKED
    assert len(outcome.blockers) == 2


def test_the_execution_cost_policy_identity_itself_fails_closed_upstream():
    """A non-baseline cost policy cannot even be constructed to be audited."""

    with pytest.raises(Exception):
        ExecutionCostPolicyRef(
            policy_id="some_other_execution_cost_policy",
            policy_fingerprint="5" * 64,
        )


def test_a_substituted_benchmark_identity_blocks_gate_k(ready_preflight_input):
    manifest = ready_preflight_input.manifest
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            manifest=manifest,
            benchmark_evidence=BenchmarkEvidence(
                benchmark_id="SPY_TOTAL_RETURN_INDEX_V9_9",
                source_artifact_ref=manifest.benchmark_source_artifact_ref,
                series_fingerprint=manifest.benchmark_series_fingerprint,
                coverage_start=DECISION_START,
                coverage_end=DECISION_END,
                price_basis="unadjusted",
                distribution_reinvestment="prohibited",
                source_acceptance_passed=False,
            ),
        )
    )
    outcome = gate(report, "K")
    assert outcome.status is PreflightGateStatus.BLOCKED
    assert len(outcome.blockers) == 2
    assert gate(report, "N").status is PreflightGateStatus.BLOCKED


def test_a_foreign_metrics_policy_blocks_gate_l():
    report = evaluate_canonical_run_preflight(
        build_ready_preflight_input(
            metrics_policy_evidence=MetricsPolicyEvidence(
                policy_ref=ResearchMetricsPolicyRef(
                    policy_fingerprint="7" * 64
                )
            )
        )
    )
    assert gate(report, "L").status is PreflightGateStatus.BLOCKED
    assert gate(report, "N").status is PreflightGateStatus.BLOCKED


def test_gate_m_actually_round_trips_the_persistence_boundary(
    ready_preflight_input,
):
    report = evaluate_canonical_run_preflight(ready_preflight_input)
    outcome = gate(report, "M")
    assert outcome.status is PreflightGateStatus.PASS
    probe = ready_preflight_input.persistence_readiness_probe
    assert probe.rehearsal_artifact.artifact_fingerprint in outcome.evidence


def test_a_manifest_that_disagrees_with_gate_evidence_blocks_gate_n():
    base = build_ready_preflight_input()
    swapped = CanonicalRunPreflightInput(
        **{
            **{
                name: getattr(base, name)
                for name in CanonicalRunPreflightInput.model_fields
            },
            "manifest": build_synthetic_manifest(git_commit_sha="c" * 40),
        }
    )
    report = evaluate_canonical_run_preflight(swapped)
    outcome = gate(report, "N")
    assert outcome.status is PreflightGateStatus.BLOCKED
    assert any("reviewed Git commit" in blocker for blocker in outcome.blockers)


def test_the_preflight_is_deterministic_and_repeatable(ready_preflight_input):
    first = evaluate_canonical_run_preflight(ready_preflight_input)
    second = evaluate_canonical_run_preflight(ready_preflight_input)
    assert first == second


def test_the_preflight_never_invokes_the_canonical_executor():
    source = (PRODUCTION / "preflight.py").read_text(encoding="utf-8")
    for token in (
        "execute_canonical_baseline_experiment",
        "CanonicalRunAuthorization",
        "baseline_experiment.execution",
        "HistoricalBacktestOrchestrator",
        "PortfolioTransitionEngine",
    ):
        assert token not in source
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert "stock_swing_d1.baseline_experiment.execution" not in imported


def test_the_preflight_performs_no_fallback_source_substitution():
    source = (PRODUCTION / "preflight.py").read_text(encoding="utf-8").lower()
    for token in (
        "fallback",
        "default_source",
        "alternative_source",
        "backup_source",
        "load_",
        "read_parquet",
        "open(",
    ):
        assert token not in source
    # A missing member stays missing: the audit never fills one in.
    blocked = evaluate_canonical_run_preflight(
        build_ready_preflight_input(market_data_evidence=None)
    )
    assert gate(blocked, "F").status is PreflightGateStatus.BLOCKED


def test_a_report_cannot_claim_readiness_while_naming_a_blocker(
    ready_preflight_input,
):
    report = evaluate_canonical_run_preflight(ready_preflight_input)
    with pytest.raises(ValueError):
        type(report)(
            verdict=CanonicalRunPreflightVerdict.READY_FOR_CANONICAL_RUN,
            audited_manifest_fingerprint=report.audited_manifest_fingerprint,
            gates=report.gates,
            blockers=("BLOCKED — invented blocker",),
        )
