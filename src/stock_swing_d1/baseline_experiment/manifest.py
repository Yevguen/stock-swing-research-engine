"""The immutable Phase 16C canonical baseline experiment manifest v0.2.

The manifest is the experiment's identity document. It binds, in one
fingerprinted object, the exact code revision, the exact frozen decision
interval, the exact starting capital, every authoritative input artifact
identity, every frozen policy identity, the benchmark evidence identity, the
Phase 16B measurement-policy identity, the Phase 16B result identity domain and
the frozen gap/earnings diagnostic contract versions.

Every field is semantic. The manifest declares no timestamp, no run counter, no
generated identifier and no filesystem path, so there is nothing volatile to
exclude from its fingerprint: the excluded set is exactly
``manifest_fingerprint`` itself. Two manifests describing the same experiment
are byte-identical and share one fingerprint.

Git identity is represented as ``git_commit_sha`` -- the full 40-hex SHA of the
*final reviewed commit immediately preceding the run* -- together with the
explicit ``git_identity_kind = "git_commit"`` discriminator, matching the
Phase 15D ``software_revision`` / ``software_revision_kind`` convention. Working
tree cleanliness is a pre-run gate fact, not manifest content: a manifest
describes what the experiment was, and the preflight decides whether it may run.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import model_validator

from stock_swing_d1.backtest_results.models import (
    ArtifactRef,
    HistoricalBacktestValuationPolicyRef,
    PolicyArtifactRef,
)
from stock_swing_d1.backtester.decision_interval import (
    HistoricalDecisionInterval,
)
from stock_swing_d1.execution.costs.models import ExecutionCostPolicyRef
from stock_swing_d1.ranking.models import RankingPolicyRef
from stock_swing_d1.research_metrics.hashing import (
    RESEARCH_METRICS_RESULT_HASH_DOMAIN,
)
from stock_swing_d1.research_metrics.models import BenchmarkSourceArtifactRef
from stock_swing_d1.research_metrics.persistence import (
    ResearchMetricsPolicyRef,
)

from stock_swing_d1.baseline_experiment.earnings_diagnostic import (
    EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_VERSION,
)
from stock_swing_d1.baseline_experiment.errors import (
    BaselineExperimentValidationError,
)
from stock_swing_d1.baseline_experiment.field_types import (
    CanonicalText,
    GitCommitSha,
    ImmutableBaselineExperimentModel,
    Sha256,
)
from stock_swing_d1.baseline_experiment.gap_policy import (
    MaterialAdverseOvernightGapPolicyRef,
)
from stock_swing_d1.baseline_experiment.hashing import (
    _compute_manifest_fingerprint,
    compute_manifest_fingerprint,
)


CANONICAL_BASELINE_EXPERIMENT_MANIFEST_SCHEMA_VERSION = (
    "canonical_baseline_experiment_manifest.v0.2"
)
CANONICAL_BASELINE_EXPERIMENT_ID = (
    "phase16c_first_canonical_baseline_experiment.v0.1"
)
CANONICAL_BASELINE_EXPERIMENT_CONTRACT_VERSION = "0.1"
CANONICAL_BASELINE_BASE_CURRENCY = "USD"
CANONICAL_BASELINE_STARTING_CAPITAL = Decimal("100000")

_REQUIRED_MANIFEST_ARGUMENTS = (
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


def _require_frozen_starting_capital(value: object) -> Decimal:
    if type(value) is not Decimal:
        raise ValueError("starting_capital must be an exact Decimal")
    if not value.is_finite() or value != CANONICAL_BASELINE_STARTING_CAPITAL:
        raise ValueError(
            "starting_capital must equal the frozen USD 100,000 anchor"
        )
    return value


class CanonicalBaselineExperimentManifest(ImmutableBaselineExperimentModel):
    """One deterministic, immutable identity document for the baseline run."""

    schema_version: Literal[
        "canonical_baseline_experiment_manifest.v0.2"
    ] = CANONICAL_BASELINE_EXPERIMENT_MANIFEST_SCHEMA_VERSION
    experiment_id: Literal[
        "phase16c_first_canonical_baseline_experiment.v0.1"
    ] = CANONICAL_BASELINE_EXPERIMENT_ID
    contract_version: Literal["0.1"] = (
        CANONICAL_BASELINE_EXPERIMENT_CONTRACT_VERSION
    )

    git_commit_sha: GitCommitSha
    git_identity_kind: Literal["git_commit"] = "git_commit"

    decision_interval: HistoricalDecisionInterval
    base_currency: Literal["USD"] = CANONICAL_BASELINE_BASE_CURRENCY
    starting_capital: Decimal

    strategy_configuration_ref: ArtifactRef
    universe_artifact_ref: ArtifactRef
    market_data_artifact_ref: ArtifactRef
    corporate_action_artifact_ref: ArtifactRef
    dividend_evidence_ref: ArtifactRef
    earnings_pit_artifact_ref: ArtifactRef
    earnings_provider_name: CanonicalText

    allocation_policy_ref: PolicyArtifactRef
    ranking_policy_ref: RankingPolicyRef
    execution_cost_policy_ref: ExecutionCostPolicyRef
    valuation_policy_ref: HistoricalBacktestValuationPolicyRef
    run_configuration_fingerprint: Sha256

    benchmark_id: CanonicalText
    benchmark_source_artifact_ref: BenchmarkSourceArtifactRef
    benchmark_series_fingerprint: Sha256

    performance_measurement_policy_ref: ResearchMetricsPolicyRef
    expected_research_metrics_result_hash_domain: Literal[
        "research_metrics_result.v0.3"
    ] = RESEARCH_METRICS_RESULT_HASH_DOMAIN

    gap_policy_ref: MaterialAdverseOvernightGapPolicyRef
    earnings_exclusion_diagnostic_version: Literal[
        "earnings_filter_exclusions.v0.1"
    ] = EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_VERSION

    manifest_fingerprint: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        _require_frozen_starting_capital(self.starting_capital)
        if (
            self.decision_interval.decision_start_date
            >= self.decision_interval.decision_end_date
        ):
            raise ValueError(
                "the canonical experiment requires a positive decision interval"
            )
        if self.manifest_fingerprint != compute_manifest_fingerprint(self):
            raise ValueError(
                "manifest_fingerprint does not match manifest content"
            )
        return self


def build_canonical_baseline_experiment_manifest(
    *,
    git_commit_sha,
    decision_interval,
    starting_capital,
    strategy_configuration_ref,
    universe_artifact_ref,
    market_data_artifact_ref,
    corporate_action_artifact_ref,
    dividend_evidence_ref,
    earnings_pit_artifact_ref,
    earnings_provider_name,
    allocation_policy_ref,
    ranking_policy_ref,
    execution_cost_policy_ref,
    valuation_policy_ref,
    run_configuration_fingerprint,
    benchmark_id,
    benchmark_source_artifact_ref,
    benchmark_series_fingerprint,
    performance_measurement_policy_ref,
    gap_policy_ref,
) -> CanonicalBaselineExperimentManifest:
    """Build one manifest, failing closed on any unestablished evidence.

    Every required identity must be supplied explicitly. A ``None`` argument
    means the corresponding canonical evidence has not been established, and
    the manifest is refused rather than completed with a placeholder.
    """

    supplied = {
        "git_commit_sha": git_commit_sha,
        "decision_interval": decision_interval,
        "starting_capital": starting_capital,
        "strategy_configuration_ref": strategy_configuration_ref,
        "universe_artifact_ref": universe_artifact_ref,
        "market_data_artifact_ref": market_data_artifact_ref,
        "corporate_action_artifact_ref": corporate_action_artifact_ref,
        "dividend_evidence_ref": dividend_evidence_ref,
        "earnings_pit_artifact_ref": earnings_pit_artifact_ref,
        "earnings_provider_name": earnings_provider_name,
        "allocation_policy_ref": allocation_policy_ref,
        "ranking_policy_ref": ranking_policy_ref,
        "execution_cost_policy_ref": execution_cost_policy_ref,
        "valuation_policy_ref": valuation_policy_ref,
        "run_configuration_fingerprint": run_configuration_fingerprint,
        "benchmark_id": benchmark_id,
        "benchmark_source_artifact_ref": benchmark_source_artifact_ref,
        "benchmark_series_fingerprint": benchmark_series_fingerprint,
        "performance_measurement_policy_ref": (
            performance_measurement_policy_ref
        ),
        "gap_policy_ref": gap_policy_ref,
    }
    missing = sorted(
        name
        for name in _REQUIRED_MANIFEST_ARGUMENTS
        if supplied[name] is None
    )
    if missing:
        raise BaselineExperimentValidationError(
            "INCOMPLETE_MANIFEST_EVIDENCE",
            f"required canonical experiment evidence is unavailable: {missing}",
        )
    values = {
        "schema_version": (
            CANONICAL_BASELINE_EXPERIMENT_MANIFEST_SCHEMA_VERSION
        ),
        "experiment_id": CANONICAL_BASELINE_EXPERIMENT_ID,
        "contract_version": CANONICAL_BASELINE_EXPERIMENT_CONTRACT_VERSION,
        "git_identity_kind": "git_commit",
        "base_currency": CANONICAL_BASELINE_BASE_CURRENCY,
        "expected_research_metrics_result_hash_domain": (
            RESEARCH_METRICS_RESULT_HASH_DOMAIN
        ),
        "earnings_exclusion_diagnostic_version": (
            EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_VERSION
        ),
        **supplied,
    }
    return CanonicalBaselineExperimentManifest(
        **values,
        manifest_fingerprint=_compute_manifest_fingerprint(**values),
    )


__all__ = [
    "CANONICAL_BASELINE_BASE_CURRENCY",
    "CANONICAL_BASELINE_EXPERIMENT_CONTRACT_VERSION",
    "CANONICAL_BASELINE_EXPERIMENT_ID",
    "CANONICAL_BASELINE_EXPERIMENT_MANIFEST_SCHEMA_VERSION",
    "CANONICAL_BASELINE_STARTING_CAPITAL",
    "CanonicalBaselineExperimentManifest",
    "build_canonical_baseline_experiment_manifest",
]
