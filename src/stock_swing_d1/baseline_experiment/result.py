"""Preservation, integrity verification and report projection for Phase 16C.

This module is a preservation boundary. It composes already-authoritative
objects into one immutable, fingerprint-bound canonical experiment record, and
it projects a report-facing view by *copying* authoritative values.

It computes no performance metric. Every number the report shows for total
return, CAGR, drawdown, Sharpe, Sortino, volatility, trade counts, win rate,
gross profit/loss, profit factor, expectancy, the average winner and loser,
the worst trade and the average holding time in sessions is read verbatim from
the authoritative Phase 16B result inside the Phase 16B.3 artifact. Nothing
here re-derives, repairs, re-prices, re-pairs, back-fills or interpolates any
of them, and no equity path is reconstructed at all.

Report items that the frozen report inventory names but that no authoritative
Phase 16B field publishes are neither invented nor silently dropped: they are
enumerated explicitly, with their reason, in
``UNAVAILABLE_AUTHORITATIVE_REPORT_METRICS`` and surfaced on the report. That
declaration is currently empty -- Phase 16B now publishes every item the
frozen inventory names.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, model_validator

from stock_swing_d1.backtester.decision_interval import (
    HistoricalDecisionInterval,
)
from stock_swing_d1.research_metrics.models import TradePerformanceMetrics
from stock_swing_d1.research_metrics.persistence import (
    ResearchMetricsPersistenceArtifact,
    ResearchMetricsPolicyRef,
    deserialize_research_metrics_artifact,
    serialize_research_metrics_artifact,
)
from stock_swing_d1.research_metrics.reporting import (
    ResearchMetricsReportProjection,
    project_research_metrics_report,
)

from stock_swing_d1.baseline_experiment.earnings_diagnostic import (
    EarningsFilterExclusionDiagnostic,
)
from stock_swing_d1.baseline_experiment.errors import (
    BaselineExperimentValidationError,
)
from stock_swing_d1.baseline_experiment.field_types import (
    CanonicalText,
    ExactBool,
    FiniteDecimal,
    GitCommitSha,
    ImmutableBaselineExperimentModel,
    Sha256,
    require_tuple,
)
from stock_swing_d1.baseline_experiment.gap_diagnostic import (
    MaterialAdverseOvernightGapDiagnostic,
)
from stock_swing_d1.baseline_experiment.hashing import (
    _compute_experiment_result_fingerprint,
    compute_experiment_result_fingerprint,
)
from stock_swing_d1.baseline_experiment.manifest import (
    CanonicalBaselineExperimentManifest,
)


CANONICAL_BASELINE_EXPERIMENT_RESULT_SCHEMA_VERSION = (
    "canonical_baseline_experiment_result.v0.2"
)
CANONICAL_BASELINE_EXPERIMENT_REPORT_SCHEMA_VERSION = (
    "canonical_baseline_experiment_report.v0.2"
)
CANONICAL_BASELINE_EXPERIMENT_INTEGRITY_REPORT_SCHEMA_VERSION = (
    "canonical_baseline_experiment_integrity_report.v0.1"
)

# Frozen report-inventory items with no authoritative Phase 16B field. Phase
# 16C is not a metrics calculator, so it names them rather than deriving them.
#
# Empty since the Phase 16B Amendment v0.1: the four items previously named
# here -- the average winner, the average loser, the worst trade and the
# average holding time in sessions -- are now measured and published by Phase
# 16B.2 on `TradePerformanceMetrics`, and the report copies them verbatim with
# the rest of that authoritative model. The declaration is kept rather than
# deleted because it is the standing structural channel for any future report
# item Phase 16B does not publish; what changed is that there is currently
# nothing to declare, and asserting otherwise would misreport the boundary.
UNAVAILABLE_AUTHORITATIVE_REPORT_METRICS: tuple[str, ...] = ()


class CanonicalBaselineExperimentResult(ImmutableBaselineExperimentModel):
    """One immutable, fingerprint-bound canonical baseline experiment record."""

    schema_version: Literal[
        "canonical_baseline_experiment_result.v0.2"
    ] = CANONICAL_BASELINE_EXPERIMENT_RESULT_SCHEMA_VERSION
    manifest: CanonicalBaselineExperimentManifest
    source_audit_result_fingerprint: Sha256
    research_metrics_artifact: ResearchMetricsPersistenceArtifact
    gap_diagnostic: MaterialAdverseOvernightGapDiagnostic
    earnings_exclusion_diagnostic: EarningsFilterExclusionDiagnostic
    result_fingerprint: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        artifact = self.research_metrics_artifact
        provenance = artifact.research_metrics_result.provenance
        if artifact.decision_interval != self.manifest.decision_interval:
            raise ValueError(
                "the preserved metrics artifact measures a different decision "
                "interval than the manifest declares"
            )
        if artifact.starting_capital != self.manifest.starting_capital:
            raise ValueError(
                "the preserved metrics artifact uses different starting capital "
                "than the manifest declares"
            )
        if artifact.policy_ref != self.manifest.performance_measurement_policy_ref:
            raise ValueError(
                "the preserved metrics artifact was produced under a different "
                "measurement policy than the manifest declares"
            )
        if (
            provenance.source_audit_result_fingerprint
            != self.source_audit_result_fingerprint
        ):
            raise ValueError(
                "the preserved metrics artifact measures a different audited run"
            )
        if (
            provenance.benchmark_series_fingerprint
            != self.manifest.benchmark_series_fingerprint
        ):
            raise ValueError(
                "the preserved metrics artifact does not bind the manifest "
                "benchmark evidence"
            )
        for diagnostic, subject in (
            (self.gap_diagnostic, "gap"),
            (self.earnings_exclusion_diagnostic, "earnings exclusion"),
        ):
            if (
                diagnostic.source_audit_result_fingerprint
                != self.source_audit_result_fingerprint
            ):
                raise ValueError(
                    f"the {subject} diagnostic describes a different audited run"
                )
        if (
            self.gap_diagnostic.policy_ref.policy_fingerprint
            != self.manifest.gap_policy_ref.policy_fingerprint
        ):
            raise ValueError(
                "the gap diagnostic was produced under a different gap policy "
                "than the manifest declares"
            )
        if (
            self.earnings_exclusion_diagnostic.diagnostic_version
            != self.manifest.earnings_exclusion_diagnostic_version
        ):
            raise ValueError(
                "the earnings exclusion diagnostic version disagrees with the "
                "manifest"
            )
        if self.result_fingerprint != compute_experiment_result_fingerprint(
            self
        ):
            raise ValueError(
                "result_fingerprint does not match experiment result content"
            )
        return self


def build_canonical_baseline_experiment_result(
    *,
    manifest: CanonicalBaselineExperimentManifest,
    source_audit_result_fingerprint: str,
    research_metrics_artifact: ResearchMetricsPersistenceArtifact,
    gap_diagnostic: MaterialAdverseOvernightGapDiagnostic,
    earnings_exclusion_diagnostic: EarningsFilterExclusionDiagnostic,
) -> CanonicalBaselineExperimentResult:
    """Preserve one canonical experiment without recomputing any of its parts."""

    expected_types = (
        (manifest, CanonicalBaselineExperimentManifest, "manifest"),
        (
            research_metrics_artifact,
            ResearchMetricsPersistenceArtifact,
            "research_metrics_artifact",
        ),
        (
            gap_diagnostic,
            MaterialAdverseOvernightGapDiagnostic,
            "gap_diagnostic",
        ),
        (
            earnings_exclusion_diagnostic,
            EarningsFilterExclusionDiagnostic,
            "earnings_exclusion_diagnostic",
        ),
    )
    for value, expected, name in expected_types:
        if type(value) is not expected:
            raise TypeError(f"{name} must be an exact {expected.__name__}")
    values = {
        "schema_version": (
            CANONICAL_BASELINE_EXPERIMENT_RESULT_SCHEMA_VERSION
        ),
        "manifest": manifest,
        "source_audit_result_fingerprint": source_audit_result_fingerprint,
        "research_metrics_artifact": research_metrics_artifact,
        "gap_diagnostic": gap_diagnostic,
        "earnings_exclusion_diagnostic": earnings_exclusion_diagnostic,
    }
    return CanonicalBaselineExperimentResult(
        **values,
        result_fingerprint=_compute_experiment_result_fingerprint(**values),
    )


class CanonicalBaselineExperimentIdentity(ImmutableBaselineExperimentModel):
    """The report-facing identity of one preserved canonical experiment."""

    experiment_id: CanonicalText
    contract_version: CanonicalText
    git_commit_sha: GitCommitSha
    decision_interval: HistoricalDecisionInterval
    base_currency: CanonicalText
    starting_capital: FiniteDecimal
    benchmark_id: CanonicalText
    benchmark_series_fingerprint: Sha256
    performance_measurement_policy_ref: ResearchMetricsPolicyRef
    gap_metric_version: CanonicalText
    earnings_exclusion_diagnostic_version: CanonicalText
    run_configuration_fingerprint: Sha256
    manifest_fingerprint: Sha256
    research_metrics_artifact_fingerprint: Sha256
    research_metrics_result_fingerprint: Sha256
    source_audit_result_fingerprint: Sha256
    experiment_result_fingerprint: Sha256


class CanonicalBaselineExperimentReport(ImmutableBaselineExperimentModel):
    """The frozen baseline report-facing projection of one canonical run."""

    schema_version: Literal[
        "canonical_baseline_experiment_report.v0.2"
    ] = CANONICAL_BASELINE_EXPERIMENT_REPORT_SCHEMA_VERSION
    identity: CanonicalBaselineExperimentIdentity
    performance: ResearchMetricsReportProjection
    trades: TradePerformanceMetrics
    gap_diagnostic: MaterialAdverseOvernightGapDiagnostic
    earnings_exclusion_diagnostic: EarningsFilterExclusionDiagnostic
    unavailable_authoritative_metrics: Annotated[
        tuple[CanonicalText, ...], BeforeValidator(require_tuple)
    ] = UNAVAILABLE_AUTHORITATIVE_REPORT_METRICS


def project_canonical_baseline_experiment_report(
    result: CanonicalBaselineExperimentResult,
) -> CanonicalBaselineExperimentReport:
    """Copy authoritative values into the stable report-facing shape."""

    if type(result) is not CanonicalBaselineExperimentResult:
        raise TypeError("result must be a CanonicalBaselineExperimentResult")
    manifest = result.manifest
    artifact = result.research_metrics_artifact
    return CanonicalBaselineExperimentReport(
        identity=CanonicalBaselineExperimentIdentity(
            experiment_id=manifest.experiment_id,
            contract_version=manifest.contract_version,
            git_commit_sha=manifest.git_commit_sha,
            decision_interval=manifest.decision_interval,
            base_currency=manifest.base_currency,
            starting_capital=manifest.starting_capital,
            benchmark_id=manifest.benchmark_id,
            benchmark_series_fingerprint=manifest.benchmark_series_fingerprint,
            performance_measurement_policy_ref=(
                manifest.performance_measurement_policy_ref
            ),
            gap_metric_version=manifest.gap_policy_ref.gap_metric_version,
            earnings_exclusion_diagnostic_version=(
                manifest.earnings_exclusion_diagnostic_version
            ),
            run_configuration_fingerprint=(
                manifest.run_configuration_fingerprint
            ),
            manifest_fingerprint=manifest.manifest_fingerprint,
            research_metrics_artifact_fingerprint=(
                artifact.artifact_fingerprint
            ),
            research_metrics_result_fingerprint=(
                artifact.research_metrics_result.result_fingerprint
            ),
            source_audit_result_fingerprint=(
                result.source_audit_result_fingerprint
            ),
            experiment_result_fingerprint=result.result_fingerprint,
        ),
        performance=project_research_metrics_report(artifact),
        trades=artifact.research_metrics_result.trade_metrics,
        gap_diagnostic=result.gap_diagnostic,
        earnings_exclusion_diagnostic=result.earnings_exclusion_diagnostic,
    )


class IntegrityCheckOutcome(ImmutableBaselineExperimentModel):
    """One named post-run integrity check and whether it held."""

    check_id: CanonicalText
    passed: ExactBool
    detail: CanonicalText


class CanonicalBaselineExperimentIntegrityReport(
    ImmutableBaselineExperimentModel
):
    """The complete deterministic post-run integrity verification."""

    schema_version: Literal[
        "canonical_baseline_experiment_integrity_report.v0.1"
    ] = CANONICAL_BASELINE_EXPERIMENT_INTEGRITY_REPORT_SCHEMA_VERSION
    experiment_result_fingerprint: Sha256
    checks: Annotated[
        tuple[IntegrityCheckOutcome, ...], BeforeValidator(require_tuple)
    ]
    integrity_verified: ExactBool

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.integrity_verified != all(
            check.passed for check in self.checks
        ):
            raise ValueError(
                "integrity is verified exactly when every check passed"
            )
        return self


_INTEGRITY_CHECK_IDS = (
    "manifest_fingerprint_recomputed",
    "experiment_result_fingerprint_recomputed",
    "gap_diagnostic_fingerprint_recomputed",
    "earnings_diagnostic_fingerprint_recomputed",
    "metrics_artifact_round_trips_to_identical_bytes",
    "metrics_result_identity_preserved",
    "audited_run_identity_linked",
)


def verify_canonical_baseline_experiment_integrity(
    result: CanonicalBaselineExperimentResult,
) -> CanonicalBaselineExperimentIntegrityReport:
    """Re-derive every preserved identity and report each check independently."""

    if type(result) is not CanonicalBaselineExperimentResult:
        raise TypeError("result must be a CanonicalBaselineExperimentResult")
    from stock_swing_d1.baseline_experiment.hashing import (
        compute_earnings_exclusion_diagnostic_fingerprint,
        compute_gap_diagnostic_fingerprint,
        compute_manifest_fingerprint,
    )

    artifact = result.research_metrics_artifact
    outcomes: list[IntegrityCheckOutcome] = []

    def record(check_id: str, passed: bool, detail: str) -> None:
        outcomes.append(
            IntegrityCheckOutcome(
                check_id=check_id, passed=passed, detail=detail
            )
        )

    recomputed_manifest = compute_manifest_fingerprint(result.manifest)
    record(
        "manifest_fingerprint_recomputed",
        recomputed_manifest == result.manifest.manifest_fingerprint,
        recomputed_manifest,
    )
    recomputed_result = compute_experiment_result_fingerprint(result)
    record(
        "experiment_result_fingerprint_recomputed",
        recomputed_result == result.result_fingerprint,
        recomputed_result,
    )
    recomputed_gap = compute_gap_diagnostic_fingerprint(result.gap_diagnostic)
    record(
        "gap_diagnostic_fingerprint_recomputed",
        recomputed_gap == result.gap_diagnostic.diagnostic_fingerprint,
        recomputed_gap,
    )
    recomputed_earnings = compute_earnings_exclusion_diagnostic_fingerprint(
        result.earnings_exclusion_diagnostic
    )
    record(
        "earnings_diagnostic_fingerprint_recomputed",
        recomputed_earnings
        == result.earnings_exclusion_diagnostic.diagnostic_fingerprint,
        recomputed_earnings,
    )
    try:
        encoded = serialize_research_metrics_artifact(artifact)
        reloaded = deserialize_research_metrics_artifact(encoded)
        round_trips = (
            serialize_research_metrics_artifact(reloaded) == encoded
            and reloaded.artifact_fingerprint == artifact.artifact_fingerprint
        )
        detail = reloaded.artifact_fingerprint
    except (TypeError, ValueError) as error:
        round_trips = False
        detail = f"round trip failed: {error}"
    record(
        "metrics_artifact_round_trips_to_identical_bytes", round_trips, detail
    )
    metrics_result = artifact.research_metrics_result
    record(
        "metrics_result_identity_preserved",
        metrics_result.provenance.performance_measurement_policy_fingerprint
        == result.manifest.performance_measurement_policy_ref.policy_fingerprint,
        metrics_result.result_fingerprint,
    )
    record(
        "audited_run_identity_linked",
        metrics_result.provenance.source_audit_result_fingerprint
        == result.source_audit_result_fingerprint
        == result.gap_diagnostic.source_audit_result_fingerprint
        == result.earnings_exclusion_diagnostic.source_audit_result_fingerprint,
        result.source_audit_result_fingerprint,
    )
    checks = tuple(outcomes)
    if tuple(check.check_id for check in checks) != _INTEGRITY_CHECK_IDS:
        raise BaselineExperimentValidationError(
            "INCOMPLETE_INTEGRITY_VERIFICATION",
            "post-run verification must run every frozen integrity check",
        )
    return CanonicalBaselineExperimentIntegrityReport(
        experiment_result_fingerprint=result.result_fingerprint,
        checks=checks,
        integrity_verified=all(check.passed for check in checks),
    )


__all__ = [
    "CANONICAL_BASELINE_EXPERIMENT_INTEGRITY_REPORT_SCHEMA_VERSION",
    "CANONICAL_BASELINE_EXPERIMENT_REPORT_SCHEMA_VERSION",
    "CANONICAL_BASELINE_EXPERIMENT_RESULT_SCHEMA_VERSION",
    "UNAVAILABLE_AUTHORITATIVE_REPORT_METRICS",
    "CanonicalBaselineExperimentIdentity",
    "CanonicalBaselineExperimentIntegrityReport",
    "CanonicalBaselineExperimentReport",
    "CanonicalBaselineExperimentResult",
    "IntegrityCheckOutcome",
    "build_canonical_baseline_experiment_result",
    "project_canonical_baseline_experiment_report",
    "verify_canonical_baseline_experiment_integrity",
]
