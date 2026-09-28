"""The deterministic Phase 16C pre-run gate audit.

This module answers exactly one question, in exactly one of two words:

    READY_FOR_CANONICAL_RUN   or   BLOCKED

It is a pure function of explicitly supplied evidence. It reads no clock, no
environment, no filesystem and no provider; it never substitutes a convenience
source for missing canonical evidence; and it never downgrades a blocker to a
warning. Evidence that has not been established is supplied as ``None``, which
is a blocker -- never an assumption.

The audit also never executes the experiment. It does not import the canonical
executor, does not run the backtester, does not price execution, and does not
measure performance. The only computation it performs is the Gate M
serialization rehearsal, which round-trips a caller-supplied representative
Phase 16B.3 artifact to prove that the persistence boundary can losslessly
preserve the future canonical result.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, model_validator

from stock_swing_d1.backtest_results.models import (
    ArtifactRef,
    HISTORICAL_BACKTEST_VALUATION_POLICY_ID,
    HistoricalBacktestValuationPolicyRef,
    PolicyArtifactRef,
)
from stock_swing_d1.backtester.decision_interval import (
    HistoricalDecisionInterval,
)
from stock_swing_d1.execution.costs.models import (
    BROKER_NEUTRAL_POLICY_ID,
    ExecutionCostPolicyRef,
)
from stock_swing_d1.portfolio.allocation_policy import (
    PORTFOLIO_ALLOCATION_POLICY_REF,
)
from stock_swing_d1.ranking.models import (
    CANDIDATE_RANKING_POLICY_ID,
    CANDIDATE_RANKING_POLICY_VERSION,
    RankingPolicyRef,
)
from stock_swing_d1.research_metrics.models import BenchmarkSourceArtifactRef
from stock_swing_d1.research_metrics.persistence import (
    ResearchMetricsPersistenceArtifact,
    ResearchMetricsPolicyRef,
    deserialize_research_metrics_artifact,
    serialize_research_metrics_artifact,
)
from stock_swing_d1.research_metrics.policy import (
    build_performance_measurement_policy,
)
from stock_swing_d1.research_metrics.reporting import (
    project_research_metrics_report,
)

from stock_swing_d1.baseline_experiment.field_types import (
    CanonicalText,
    ExactBool,
    GitCommitSha,
    ImmutableBaselineExperimentModel,
    NonNegativeCount,
    OptionalSha256,
    SessionDate,
    Sha256,
    require_exact_enum,
    require_tuple,
)
from stock_swing_d1.baseline_experiment.gap_policy import (
    build_material_adverse_overnight_gap_policy,
)
from stock_swing_d1.baseline_experiment.manifest import (
    CANONICAL_BASELINE_STARTING_CAPITAL,
    CanonicalBaselineExperimentManifest,
)


CANONICAL_RUN_PREFLIGHT_INPUT_SCHEMA_VERSION = (
    "canonical_run_preflight_input.v0.1"
)
CANONICAL_RUN_PREFLIGHT_REPORT_SCHEMA_VERSION = (
    "canonical_run_preflight_report.v0.1"
)
CANONICAL_BASELINE_BENCHMARK_ID = "SPY_BUY_AND_HOLD_V0_1"


class CanonicalRunPreflightVerdict(StrEnum):
    """The only two admissible pre-run verdicts."""

    READY_FOR_CANONICAL_RUN = "READY_FOR_CANONICAL_RUN"
    BLOCKED = "BLOCKED"


class PreflightGateStatus(StrEnum):
    """The only two admissible per-gate outcomes."""

    PASS = "PASS"
    BLOCKED = "BLOCKED"


GATE_NAMES: dict[str, str] = {
    "A": "repository_and_code_identity",
    "B": "historical_decision_interval",
    "C": "starting_capital",
    "D": "strategy_configuration",
    "E": "historical_universe",
    "F": "historical_market_data",
    "G": "corporate_actions",
    "H": "ordinary_dividends",
    "I": "earnings_point_in_time",
    "J": "execution_and_cost_semantics",
    "K": "spy_benchmark",
    "L": "research_metrics_policy",
    "M": "persistence_readiness",
    "N": "manifest_completeness",
}
GATE_IDS = tuple(GATE_NAMES)


_Verdict = Annotated[
    CanonicalRunPreflightVerdict,
    BeforeValidator(require_exact_enum(CanonicalRunPreflightVerdict)),
]
_GateStatus = Annotated[
    PreflightGateStatus, BeforeValidator(require_exact_enum(PreflightGateStatus))
]


class RepositoryIdentityEvidence(ImmutableBaselineExperimentModel):
    """Gate A evidence: which reviewed code the run would bind itself to."""

    schema_version: Literal[
        "repository_identity_evidence.v0.1"
    ] = "repository_identity_evidence.v0.1"
    reviewed_git_commit_sha: GitCommitSha
    working_tree_clean: ExactBool
    experiment_implementation_reviewed: ExactBool
    required_tests_green: ExactBool


class DecisionIntervalEvidence(ImmutableBaselineExperimentModel):
    """Gate B evidence: the frozen authoritative interval, never a derived one."""

    schema_version: Literal[
        "decision_interval_evidence.v0.1"
    ] = "decision_interval_evidence.v0.1"
    decision_interval: HistoricalDecisionInterval
    interval_source: Literal["frozen_authoritative_contract"]
    derived_from_observed_sessions: ExactBool
    selected_after_observing_results: ExactBool


class StartingCapitalEvidence(ImmutableBaselineExperimentModel):
    """Gate C evidence: the frozen Phase 16A Gate-2 capital anchor."""

    schema_version: Literal[
        "starting_capital_evidence.v0.1"
    ] = "starting_capital_evidence.v0.1"
    base_currency: Literal["USD"]
    starting_capital: Decimal


class StrategyConfigurationEvidence(ImmutableBaselineExperimentModel):
    """Gate D evidence: the exact frozen baseline strategy configuration."""

    schema_version: Literal[
        "strategy_configuration_evidence.v0.1"
    ] = "strategy_configuration_evidence.v0.1"
    artifact_ref: ArtifactRef
    configuration_frozen: ExactBool
    tuned_parameter_count: NonNegativeCount


class UniverseEvidence(ImmutableBaselineExperimentModel):
    """Gate E evidence: the canonical point-in-time universe artifact."""

    schema_version: Literal[
        "universe_evidence.v0.1"
    ] = "universe_evidence.v0.1"
    artifact_ref: ArtifactRef
    point_in_time_membership: ExactBool
    coverage_start: SessionDate
    coverage_end: SessionDate
    validation_passed: ExactBool


class MarketDataEvidence(ImmutableBaselineExperimentModel):
    """Gate F evidence: the canonical D1 price/volume artifact."""

    schema_version: Literal[
        "market_data_evidence.v0.1"
    ] = "market_data_evidence.v0.1"
    artifact_ref: ArtifactRef
    timeframe: Literal["D1"]
    session_type: Literal["regular"]
    price_basis: Literal["unadjusted"]
    currency: Literal["USD"]
    coverage_start: SessionDate
    coverage_end: SessionDate
    dataset_validation_passed: ExactBool


class CorporateActionEvidence(ImmutableBaselineExperimentModel):
    """Gate G evidence: canonical split / reverse-split and related events."""

    schema_version: Literal[
        "corporate_action_evidence.v0.1"
    ] = "corporate_action_evidence.v0.1"
    artifact_ref: ArtifactRef
    coverage_start: SessionDate
    coverage_end: SessionDate
    unresolved_action_count: NonNegativeCount
    parity_validation_passed: ExactBool


class DividendEvidence(ImmutableBaselineExperimentModel):
    """Gate H evidence: canonical ordinary-dividend accounting evidence."""

    schema_version: Literal[
        "dividend_evidence.v0.1"
    ] = "dividend_evidence.v0.1"
    artifact_ref: ArtifactRef
    coverage_start: SessionDate
    coverage_end: SessionDate
    accounting_contract_version: CanonicalText
    validation_passed: ExactBool


class EarningsPitEvidence(ImmutableBaselineExperimentModel):
    """Gate I evidence: canonical point-in-time earnings evidence."""

    schema_version: Literal[
        "earnings_pit_evidence.v0.1"
    ] = "earnings_pit_evidence.v0.1"
    artifact_ref: ArtifactRef
    provider_name: CanonicalText
    coverage_start: SessionDate
    coverage_end: SessionDate
    point_in_time_reconstruction_validated: ExactBool
    look_ahead_substitution_used: ExactBool


class ExecutionSemanticsEvidence(ImmutableBaselineExperimentModel):
    """Gate J evidence: already-frozen execution, cost and allocation policies."""

    schema_version: Literal[
        "execution_semantics_evidence.v0.1"
    ] = "execution_semantics_evidence.v0.1"
    execution_cost_policy_ref: ExecutionCostPolicyRef
    settlement_model: CanonicalText
    allocation_policy_ref: PolicyArtifactRef
    ranking_policy_ref: RankingPolicyRef
    valuation_policy_ref: HistoricalBacktestValuationPolicyRef


class BenchmarkEvidence(ImmutableBaselineExperimentModel):
    """Gate K evidence: the frozen Gate-5 investable SPY benchmark."""

    schema_version: Literal[
        "benchmark_evidence.v0.1"
    ] = "benchmark_evidence.v0.1"
    benchmark_id: CanonicalText
    source_artifact_ref: BenchmarkSourceArtifactRef
    series_fingerprint: Sha256
    coverage_start: SessionDate
    coverage_end: SessionDate
    price_basis: Literal["unadjusted"]
    distribution_reinvestment: Literal["prohibited"]
    source_acceptance_passed: ExactBool


class MetricsPolicyEvidence(ImmutableBaselineExperimentModel):
    """Gate L evidence: the authoritative Phase 16B measurement policy."""

    schema_version: Literal[
        "metrics_policy_evidence.v0.1"
    ] = "metrics_policy_evidence.v0.1"
    policy_ref: ResearchMetricsPolicyRef


class PersistenceReadinessProbe(ImmutableBaselineExperimentModel):
    """Gate M evidence: a representative Phase 16B.3 artifact to round-trip."""

    schema_version: Literal[
        "persistence_readiness_probe.v0.1"
    ] = "persistence_readiness_probe.v0.1"
    rehearsal_artifact: ResearchMetricsPersistenceArtifact


class CanonicalRunPreflightInput(ImmutableBaselineExperimentModel):
    """Every piece of evidence the pre-run audit is permitted to consider.

    A ``None`` member means the corresponding canonical evidence has not been
    established. It is never replaced with a default, a convenience source, or
    an inference from another member.
    """

    schema_version: Literal[
        "canonical_run_preflight_input.v0.1"
    ] = CANONICAL_RUN_PREFLIGHT_INPUT_SCHEMA_VERSION
    repository_identity: RepositoryIdentityEvidence | None = None
    decision_interval_evidence: DecisionIntervalEvidence | None = None
    starting_capital_evidence: StartingCapitalEvidence | None = None
    strategy_configuration_evidence: StrategyConfigurationEvidence | None = None
    universe_evidence: UniverseEvidence | None = None
    market_data_evidence: MarketDataEvidence | None = None
    corporate_action_evidence: CorporateActionEvidence | None = None
    dividend_evidence: DividendEvidence | None = None
    earnings_pit_evidence: EarningsPitEvidence | None = None
    execution_semantics_evidence: ExecutionSemanticsEvidence | None = None
    benchmark_evidence: BenchmarkEvidence | None = None
    metrics_policy_evidence: MetricsPolicyEvidence | None = None
    persistence_readiness_probe: PersistenceReadinessProbe | None = None
    manifest: CanonicalBaselineExperimentManifest | None = None


class PreflightGateOutcome(ImmutableBaselineExperimentModel):
    """One gate's deterministic verdict plus its independent blockers."""

    schema_version: Literal[
        "preflight_gate_outcome.v0.1"
    ] = "preflight_gate_outcome.v0.1"
    gate_id: CanonicalText
    gate_name: CanonicalText
    status: _GateStatus
    evidence: CanonicalText
    blockers: Annotated[
        tuple[CanonicalText, ...], BeforeValidator(require_tuple)
    ] = ()

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if (self.status is PreflightGateStatus.BLOCKED) != bool(self.blockers):
            raise ValueError(
                "a blocked gate names at least one blocker and a passing gate "
                "names none"
            )
        return self


class CanonicalRunPreflightReport(ImmutableBaselineExperimentModel):
    """The complete deterministic pre-run audit of gates A through N."""

    schema_version: Literal[
        "canonical_run_preflight_report.v0.1"
    ] = CANONICAL_RUN_PREFLIGHT_REPORT_SCHEMA_VERSION
    verdict: _Verdict
    audited_manifest_fingerprint: OptionalSha256 = None
    gates: Annotated[
        tuple[PreflightGateOutcome, ...], BeforeValidator(require_tuple)
    ]
    blockers: Annotated[
        tuple[CanonicalText, ...], BeforeValidator(require_tuple)
    ] = ()

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if tuple(gate.gate_id for gate in self.gates) != GATE_IDS:
            raise ValueError(
                "the report must contain gates A through N exactly once, in order"
            )
        blocked = tuple(
            f"BLOCKED — {blocker}"
            for gate in self.gates
            for blocker in gate.blockers
        )
        if self.blockers != blocked:
            raise ValueError(
                "report blockers must enumerate every gate blocker, in gate order"
            )
        ready = not blocked
        if ready != (
            self.verdict is CanonicalRunPreflightVerdict.READY_FOR_CANONICAL_RUN
        ):
            raise ValueError(
                "the verdict is READY_FOR_CANONICAL_RUN exactly when no gate is "
                "blocked"
            )
        if ready and self.audited_manifest_fingerprint is None:
            raise ValueError(
                "a ready report must name the exact manifest it audited"
            )
        return self


def _covers(
    interval: HistoricalDecisionInterval | None,
    start: date,
    end: date,
    *,
    subject: str,
    blockers: list[str],
) -> None:
    """Require an evidence window to cover the whole authoritative interval."""

    if interval is None:
        blockers.append(
            f"{subject} coverage cannot be proven because the frozen "
            "historical decision interval is unavailable"
        )
        return
    if start > interval.decision_start_date or end < interval.decision_end_date:
        blockers.append(
            f"{subject} does not cover the authoritative decision interval"
        )


def _gate_a(source: CanonicalRunPreflightInput) -> tuple[str, list[str]]:
    evidence = source.repository_identity
    blockers: list[str] = []
    if evidence is None:
        blockers.append("final reviewed Git commit not established")
        return ("no repository identity evidence supplied", blockers)
    if not evidence.working_tree_clean:
        blockers.append(
            "working tree is not clean; the run would bind unreviewed code state"
        )
    if not evidence.experiment_implementation_reviewed:
        blockers.append(
            "the implementation required for the canonical run is not reviewed"
        )
    if not evidence.required_tests_green:
        blockers.append("required tests are not green")
    return (
        f"reviewed commit {evidence.reviewed_git_commit_sha}",
        blockers,
    )


def _gate_b(source: CanonicalRunPreflightInput) -> tuple[str, list[str]]:
    evidence = source.decision_interval_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append(
            "the frozen authoritative historical decision interval is "
            "unavailable"
        )
        return ("no decision interval evidence supplied", blockers)
    if evidence.derived_from_observed_sessions:
        blockers.append(
            "the decision interval was derived from observed sessions rather "
            "than the frozen contract"
        )
    if evidence.selected_after_observing_results:
        blockers.append(
            "the decision interval was selected after observing results"
        )
    interval = evidence.decision_interval
    return (
        f"frozen interval {interval.decision_start_date.isoformat()} to "
        f"{interval.decision_end_date.isoformat()}",
        blockers,
    )


def _gate_c(source: CanonicalRunPreflightInput) -> tuple[str, list[str]]:
    evidence = source.starting_capital_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append("frozen starting-capital evidence is unavailable")
        return ("no starting capital evidence supplied", blockers)
    if evidence.starting_capital != CANONICAL_BASELINE_STARTING_CAPITAL:
        blockers.append(
            "starting capital does not equal the frozen USD 100,000 anchor"
        )
    return (
        f"{evidence.base_currency} {evidence.starting_capital}",
        blockers,
    )


def _gate_d(source: CanonicalRunPreflightInput) -> tuple[str, list[str]]:
    evidence = source.strategy_configuration_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append(
            "the frozen baseline strategy configuration artifact is unavailable"
        )
        return ("no strategy configuration evidence supplied", blockers)
    if not evidence.configuration_frozen:
        blockers.append("the baseline strategy configuration is not frozen")
    if evidence.tuned_parameter_count != 0:
        blockers.append(
            "the strategy configuration carries tuned parameters, which the "
            "baseline experiment prohibits"
        )
    return (
        f"strategy artifact {evidence.artifact_ref.content_sha256}",
        blockers,
    )


def _gate_e(
    source: CanonicalRunPreflightInput,
    interval: HistoricalDecisionInterval | None,
) -> tuple[str, list[str]]:
    evidence = source.universe_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append(
            "the canonical point-in-time universe artifact is unavailable"
        )
        return ("no universe evidence supplied", blockers)
    if not evidence.point_in_time_membership:
        blockers.append("the universe artifact is not point-in-time")
    if not evidence.validation_passed:
        blockers.append("the universe artifact failed its own validation")
    _covers(
        interval,
        evidence.coverage_start,
        evidence.coverage_end,
        subject="universe evidence",
        blockers=blockers,
    )
    return (
        f"universe artifact {evidence.artifact_ref.content_sha256}",
        blockers,
    )


def _gate_f(
    source: CanonicalRunPreflightInput,
    interval: HistoricalDecisionInterval | None,
) -> tuple[str, list[str]]:
    evidence = source.market_data_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append("the canonical D1 market-data artifact is unavailable")
        return ("no market data evidence supplied", blockers)
    if not evidence.dataset_validation_passed:
        blockers.append("the market-data artifact failed dataset validation")
    _covers(
        interval,
        evidence.coverage_start,
        evidence.coverage_end,
        subject="market data",
        blockers=blockers,
    )
    return (
        f"{evidence.timeframe} {evidence.price_basis} market data "
        f"{evidence.artifact_ref.content_sha256}",
        blockers,
    )


def _gate_g(
    source: CanonicalRunPreflightInput,
    interval: HistoricalDecisionInterval | None,
) -> tuple[str, list[str]]:
    evidence = source.corporate_action_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append("canonical corporate-action evidence is unavailable")
        return ("no corporate action evidence supplied", blockers)
    if not evidence.parity_validation_passed:
        blockers.append("corporate-action parity validation did not pass")
    if evidence.unresolved_action_count != 0:
        blockers.append(
            "at least one required corporate-action seam is unresolved"
        )
    _covers(
        interval,
        evidence.coverage_start,
        evidence.coverage_end,
        subject="corporate action evidence",
        blockers=blockers,
    )
    return (
        f"corporate action artifact {evidence.artifact_ref.content_sha256}",
        blockers,
    )


def _gate_h(
    source: CanonicalRunPreflightInput,
    interval: HistoricalDecisionInterval | None,
) -> tuple[str, list[str]]:
    evidence = source.dividend_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append("canonical ordinary-dividend evidence is unavailable")
        return ("no dividend evidence supplied", blockers)
    if not evidence.validation_passed:
        blockers.append("the dividend evidence failed its own validation")
    _covers(
        interval,
        evidence.coverage_start,
        evidence.coverage_end,
        subject="dividend evidence",
        blockers=blockers,
    )
    return (
        f"dividend artifact {evidence.artifact_ref.content_sha256} under "
        f"{evidence.accounting_contract_version}",
        blockers,
    )


def _gate_i(
    source: CanonicalRunPreflightInput,
    interval: HistoricalDecisionInterval | None,
) -> tuple[str, list[str]]:
    evidence = source.earnings_pit_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append("canonical earnings PIT artifact unavailable")
        return ("no earnings PIT evidence supplied", blockers)
    if not evidence.point_in_time_reconstruction_validated:
        blockers.append(
            "point-in-time earnings reconstruction is not validated"
        )
    if evidence.look_ahead_substitution_used:
        blockers.append("the earnings evidence relies on a look-ahead substitution")
    _covers(
        interval,
        evidence.coverage_start,
        evidence.coverage_end,
        subject="earnings PIT evidence",
        blockers=blockers,
    )
    return (
        f"earnings artifact {evidence.artifact_ref.content_sha256} from "
        f"{evidence.provider_name}",
        blockers,
    )


def _gate_j(source: CanonicalRunPreflightInput) -> tuple[str, list[str]]:
    evidence = source.execution_semantics_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append(
            "frozen execution, cost, settlement and allocation policy "
            "references are unavailable"
        )
        return ("no execution semantics evidence supplied", blockers)
    if evidence.execution_cost_policy_ref.policy_id != BROKER_NEUTRAL_POLICY_ID:
        blockers.append(
            "the execution-cost policy is not the frozen baseline policy"
        )
    if (
        evidence.allocation_policy_ref.policy_fingerprint
        != PORTFOLIO_ALLOCATION_POLICY_REF.policy_fingerprint
    ):
        blockers.append(
            "the allocation policy fingerprint is not the frozen Phase 12 "
            "policy fingerprint"
        )
    if (
        evidence.ranking_policy_ref.policy_id != CANDIDATE_RANKING_POLICY_ID
        or evidence.ranking_policy_ref.policy_version
        != CANDIDATE_RANKING_POLICY_VERSION
    ):
        blockers.append("the ranking policy is not the frozen Phase 14 policy")
    if (
        evidence.valuation_policy_ref.policy_id
        != HISTORICAL_BACKTEST_VALUATION_POLICY_ID
    ):
        blockers.append(
            "the valuation policy is not the frozen Phase 15D policy"
        )
    return (
        f"cost policy {evidence.execution_cost_policy_ref.policy_fingerprint}, "
        f"settlement {evidence.settlement_model}",
        blockers,
    )


def _gate_k(
    source: CanonicalRunPreflightInput,
    interval: HistoricalDecisionInterval | None,
) -> tuple[str, list[str]]:
    evidence = source.benchmark_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append("SPY benchmark evidence incomplete")
        return ("no benchmark evidence supplied", blockers)
    if evidence.benchmark_id != CANONICAL_BASELINE_BENCHMARK_ID:
        blockers.append(
            "the benchmark identity is not the frozen investable SPY benchmark"
        )
    if not evidence.source_acceptance_passed:
        blockers.append(
            "the SPY benchmark source has not passed its acceptance gate"
        )
    _covers(
        interval,
        evidence.coverage_start,
        evidence.coverage_end,
        subject="SPY benchmark evidence",
        blockers=blockers,
    )
    return (
        f"{evidence.benchmark_id} series {evidence.series_fingerprint}",
        blockers,
    )


def _gate_l(source: CanonicalRunPreflightInput) -> tuple[str, list[str]]:
    evidence = source.metrics_policy_evidence
    blockers: list[str] = []
    if evidence is None:
        blockers.append(
            "the authoritative Phase 16B metrics policy identity is unavailable"
        )
        return ("no metrics policy evidence supplied", blockers)
    canonical = build_performance_measurement_policy()
    if evidence.policy_ref.policy_fingerprint != canonical.policy_fingerprint:
        blockers.append(
            "the supplied metrics policy fingerprint is not the authoritative "
            "Phase 16B policy fingerprint"
        )
    return (
        f"metrics policy {evidence.policy_ref.policy_fingerprint}",
        blockers,
    )


def _gate_m(source: CanonicalRunPreflightInput) -> tuple[str, list[str]]:
    probe = source.persistence_readiness_probe
    blockers: list[str] = []
    if probe is None:
        blockers.append(
            "Phase 16B.3 persistence readiness has not been demonstrated"
        )
        return ("no persistence readiness probe supplied", blockers)
    artifact = probe.rehearsal_artifact
    try:
        encoded = serialize_research_metrics_artifact(artifact)
        reloaded = deserialize_research_metrics_artifact(encoded)
        reencoded = serialize_research_metrics_artifact(reloaded)
        projection = project_research_metrics_report(reloaded)
    except (TypeError, ValueError) as error:
        blockers.append(
            f"the Phase 16B.3 persistence boundary is not ready: {error}"
        )
        return ("persistence rehearsal failed", blockers)
    if encoded != reencoded:
        blockers.append(
            "the Phase 16B.3 artifact does not round-trip to identical bytes"
        )
    if reloaded.artifact_fingerprint != artifact.artifact_fingerprint:
        blockers.append(
            "the Phase 16B.3 artifact fingerprint is not preserved by a "
            "round-trip"
        )
    if (
        projection.context.research_metrics_result_fingerprint
        != artifact.research_metrics_result.result_fingerprint
    ):
        blockers.append(
            "the report-facing projection does not preserve the authoritative "
            "Phase 16B result identity"
        )
    return (
        f"rehearsed artifact {artifact.artifact_fingerprint}",
        blockers,
    )


def _gate_n(source: CanonicalRunPreflightInput) -> tuple[str, list[str]]:
    manifest = source.manifest
    blockers: list[str] = []
    if manifest is None:
        blockers.append(
            "the canonical experiment manifest cannot be populated from "
            "established evidence"
        )
        return ("no manifest supplied", blockers)

    repository = source.repository_identity
    if (
        repository is not None
        and manifest.git_commit_sha != repository.reviewed_git_commit_sha
    ):
        blockers.append(
            "the manifest does not bind the final reviewed Git commit"
        )
    interval_evidence = source.decision_interval_evidence
    if (
        interval_evidence is not None
        and manifest.decision_interval != interval_evidence.decision_interval
    ):
        blockers.append(
            "the manifest interval is not the frozen authoritative interval"
        )
    capital = source.starting_capital_evidence
    if capital is not None and (
        manifest.starting_capital != capital.starting_capital
        or manifest.base_currency != capital.base_currency
    ):
        blockers.append(
            "the manifest starting capital is not the frozen Gate-2 capital"
        )
    benchmark = source.benchmark_evidence
    if benchmark is not None and (
        manifest.benchmark_id != benchmark.benchmark_id
        or manifest.benchmark_series_fingerprint != benchmark.series_fingerprint
        or manifest.benchmark_source_artifact_ref
        != benchmark.source_artifact_ref
    ):
        blockers.append(
            "the manifest benchmark identity disagrees with the Gate-K evidence"
        )
    metrics_policy = source.metrics_policy_evidence
    if (
        metrics_policy is not None
        and manifest.performance_measurement_policy_ref
        != metrics_policy.policy_ref
    ):
        blockers.append(
            "the manifest metrics-policy reference disagrees with the Gate-L "
            "evidence"
        )
    canonical_gap_policy = build_material_adverse_overnight_gap_policy()
    if (
        manifest.gap_policy_ref.policy_fingerprint
        != canonical_gap_policy.policy_fingerprint
    ):
        blockers.append(
            "the manifest does not bind the frozen v0.1 gap-loss-frequency "
            "contract"
        )
    for evidence, manifest_ref, subject in (
        (source.strategy_configuration_evidence, manifest.strategy_configuration_ref, "strategy configuration"),
        (source.universe_evidence, manifest.universe_artifact_ref, "universe"),
        (source.market_data_evidence, manifest.market_data_artifact_ref, "market data"),
        (source.corporate_action_evidence, manifest.corporate_action_artifact_ref, "corporate action"),
        (source.dividend_evidence, manifest.dividend_evidence_ref, "dividend"),
        (source.earnings_pit_evidence, manifest.earnings_pit_artifact_ref, "earnings PIT"),
    ):
        if evidence is not None and evidence.artifact_ref != manifest_ref:
            blockers.append(
                f"the manifest {subject} artifact reference disagrees with its "
                "gate evidence"
            )
    return (f"manifest {manifest.manifest_fingerprint}", blockers)


def evaluate_canonical_run_preflight(
    preflight_input: CanonicalRunPreflightInput,
) -> CanonicalRunPreflightReport:
    """Audit gates A through N and return exactly one deterministic verdict."""

    if type(preflight_input) is not CanonicalRunPreflightInput:
        raise TypeError(
            "preflight_input must be a CanonicalRunPreflightInput"
        )
    interval = (
        None
        if preflight_input.decision_interval_evidence is None
        else preflight_input.decision_interval_evidence.decision_interval
    )
    results = {
        "A": _gate_a(preflight_input),
        "B": _gate_b(preflight_input),
        "C": _gate_c(preflight_input),
        "D": _gate_d(preflight_input),
        "E": _gate_e(preflight_input, interval),
        "F": _gate_f(preflight_input, interval),
        "G": _gate_g(preflight_input, interval),
        "H": _gate_h(preflight_input, interval),
        "I": _gate_i(preflight_input, interval),
        "J": _gate_j(preflight_input),
        "K": _gate_k(preflight_input, interval),
        "L": _gate_l(preflight_input),
        "M": _gate_m(preflight_input),
        "N": _gate_n(preflight_input),
    }
    gates = tuple(
        PreflightGateOutcome(
            gate_id=gate_id,
            gate_name=GATE_NAMES[gate_id],
            status=(
                PreflightGateStatus.BLOCKED
                if results[gate_id][1]
                else PreflightGateStatus.PASS
            ),
            evidence=results[gate_id][0],
            blockers=tuple(results[gate_id][1]),
        )
        for gate_id in GATE_IDS
    )
    blockers = tuple(
        f"BLOCKED — {blocker}" for gate in gates for blocker in gate.blockers
    )
    return CanonicalRunPreflightReport(
        verdict=(
            CanonicalRunPreflightVerdict.BLOCKED
            if blockers
            else CanonicalRunPreflightVerdict.READY_FOR_CANONICAL_RUN
        ),
        audited_manifest_fingerprint=(
            None
            if preflight_input.manifest is None
            else preflight_input.manifest.manifest_fingerprint
        ),
        gates=gates,
        blockers=blockers,
    )


__all__ = [
    "CANONICAL_BASELINE_BENCHMARK_ID",
    "CANONICAL_RUN_PREFLIGHT_INPUT_SCHEMA_VERSION",
    "CANONICAL_RUN_PREFLIGHT_REPORT_SCHEMA_VERSION",
    "GATE_IDS",
    "GATE_NAMES",
    "BenchmarkEvidence",
    "CanonicalRunPreflightInput",
    "CanonicalRunPreflightReport",
    "CanonicalRunPreflightVerdict",
    "CorporateActionEvidence",
    "DecisionIntervalEvidence",
    "DividendEvidence",
    "EarningsPitEvidence",
    "ExecutionSemanticsEvidence",
    "MarketDataEvidence",
    "MetricsPolicyEvidence",
    "PersistenceReadinessProbe",
    "PreflightGateOutcome",
    "PreflightGateStatus",
    "RepositoryIdentityEvidence",
    "StartingCapitalEvidence",
    "StrategyConfigurationEvidence",
    "UniverseEvidence",
]
