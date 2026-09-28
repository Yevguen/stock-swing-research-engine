"""The one deliberate entry point of the exactly-once canonical baseline run.

Operational separation is the whole point of this module. Nothing in the test
suite, the preflight audit, or any other Phase 16C module calls the function
below by accident: it refuses to do anything at all unless

* an explicit :class:`CanonicalRunAuthorization` carrying the exact frozen
  authorization token is supplied, with the exactly-once acknowledgement set;
* a pre-run audit report is supplied whose verdict is literally
  ``READY_FOR_CANONICAL_RUN``, meaning every gate A-N passed;
* that audit was performed against exactly the manifest being preserved; and
* the authorizing commit is the commit the manifest binds.

The function itself runs no backtest, prices no execution and measures no
performance. It is a preservation step: it composes already-authoritative
artifacts into the immutable canonical record and verifies its integrity before
returning it. Producing those authoritative artifacts from canonical historical
data is what actually consumes the exactly-once experiment, and that happens in
the upstream owners, under this gate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from stock_swing_d1.research_metrics.persistence import (
    ResearchMetricsPersistenceArtifact,
)

from stock_swing_d1.baseline_experiment.earnings_diagnostic import (
    EarningsFilterExclusionDiagnostic,
)
from stock_swing_d1.baseline_experiment.errors import (
    CanonicalRunAuthorizationError,
)
from stock_swing_d1.baseline_experiment.field_types import (
    ExactBool,
    GitCommitSha,
    ImmutableBaselineExperimentModel,
    Sha256,
)
from stock_swing_d1.baseline_experiment.gap_diagnostic import (
    MaterialAdverseOvernightGapDiagnostic,
)
from stock_swing_d1.baseline_experiment.manifest import (
    CanonicalBaselineExperimentManifest,
)
from stock_swing_d1.baseline_experiment.preflight import (
    CanonicalRunPreflightReport,
    CanonicalRunPreflightVerdict,
)
from stock_swing_d1.baseline_experiment.result import (
    CanonicalBaselineExperimentResult,
    build_canonical_baseline_experiment_result,
    verify_canonical_baseline_experiment_integrity,
)


CANONICAL_RUN_AUTHORIZATION_TOKEN = (
    "PHASE16C_CANONICAL_BASELINE_RUN_AUTHORIZED"
)
CANONICAL_RUN_AUTHORIZATION_SCHEMA_VERSION = (
    "canonical_run_authorization.v0.1"
)


class CanonicalBaselineExperimentArtifacts(ImmutableBaselineExperimentModel):
    """Authoritative artifacts produced by the guarded upstream operation.

    Phase 16C deliberately does not know how these artifacts are assembled.
    The producer remains owned by the Phase 13/15A/15D/16B economic owners and
    is called only after this module has accepted the explicit authorization.
    """

    source_audit_result_fingerprint: Sha256
    research_metrics_artifact: ResearchMetricsPersistenceArtifact
    gap_diagnostic: MaterialAdverseOvernightGapDiagnostic
    earnings_exclusion_diagnostic: EarningsFilterExclusionDiagnostic


class CanonicalRunAuthorization(ImmutableBaselineExperimentModel):
    """An explicit, non-default authorization to consume the one baseline run."""

    schema_version: Literal[
        "canonical_run_authorization.v0.1"
    ] = CANONICAL_RUN_AUTHORIZATION_SCHEMA_VERSION
    token: Literal["PHASE16C_CANONICAL_BASELINE_RUN_AUTHORIZED"]
    acknowledged_exactly_once: ExactBool
    authorizing_git_commit_sha: GitCommitSha


def execute_canonical_baseline_experiment(
    *,
    authorization: CanonicalRunAuthorization,
    preflight_report: CanonicalRunPreflightReport,
    manifest: CanonicalBaselineExperimentManifest,
    canonical_artifact_producer: Callable[
        [], CanonicalBaselineExperimentArtifacts
    ],
) -> CanonicalBaselineExperimentResult:
    """Preserve the canonical baseline experiment, or refuse to proceed."""

    if type(authorization) is not CanonicalRunAuthorization:
        raise CanonicalRunAuthorizationError(
            "MISSING_CANONICAL_RUN_AUTHORIZATION",
            "the exactly-once canonical baseline run requires an explicit "
            "CanonicalRunAuthorization",
        )
    if authorization.token != CANONICAL_RUN_AUTHORIZATION_TOKEN:
        raise CanonicalRunAuthorizationError(
            "INVALID_CANONICAL_RUN_AUTHORIZATION_TOKEN",
            "the supplied authorization token is not the frozen token",
        )
    if not authorization.acknowledged_exactly_once:
        raise CanonicalRunAuthorizationError(
            "EXACTLY_ONCE_NOT_ACKNOWLEDGED",
            "the caller has not acknowledged that this consumes the one "
            "canonical baseline experiment",
        )
    if type(preflight_report) is not CanonicalRunPreflightReport:
        raise CanonicalRunAuthorizationError(
            "MISSING_PRERUN_AUDIT",
            "the canonical run requires a completed pre-run gate audit",
        )
    if (
        preflight_report.verdict
        is not CanonicalRunPreflightVerdict.READY_FOR_CANONICAL_RUN
    ):
        raise CanonicalRunAuthorizationError(
            "PRERUN_AUDIT_BLOCKED",
            "the pre-run audit is BLOCKED: "
            + "; ".join(preflight_report.blockers),
        )
    if type(manifest) is not CanonicalBaselineExperimentManifest:
        raise CanonicalRunAuthorizationError(
            "MISSING_EXPERIMENT_MANIFEST",
            "the canonical run requires its canonical experiment manifest",
        )
    if (
        preflight_report.audited_manifest_fingerprint
        != manifest.manifest_fingerprint
    ):
        raise CanonicalRunAuthorizationError(
            "PRERUN_AUDIT_MANIFEST_MISMATCH",
            "the pre-run audit did not audit this exact experiment manifest",
        )
    if authorization.authorizing_git_commit_sha != manifest.git_commit_sha:
        raise CanonicalRunAuthorizationError(
            "AUTHORIZED_COMMIT_MISMATCH",
            "the authorization names a different commit than the manifest binds",
        )

    if not callable(canonical_artifact_producer):
        raise CanonicalRunAuthorizationError(
            "MISSING_CANONICAL_ARTIFACT_PRODUCER",
            "the canonical run requires a deferred canonical artifact producer",
        )

    # This is the true authorization boundary.  The injected operation may
    # consume the one canonical run; never evaluate it before every gate above
    # has passed, and never retry it here.
    artifacts = canonical_artifact_producer()
    if type(artifacts) is not CanonicalBaselineExperimentArtifacts:
        raise CanonicalRunAuthorizationError(
            "INVALID_CANONICAL_ARTIFACT_PRODUCER_RESULT",
            "the canonical artifact producer returned an invalid artifact bundle",
        )
    result = build_canonical_baseline_experiment_result(
        manifest=manifest,
        source_audit_result_fingerprint=artifacts.source_audit_result_fingerprint,
        research_metrics_artifact=artifacts.research_metrics_artifact,
        gap_diagnostic=artifacts.gap_diagnostic,
        earnings_exclusion_diagnostic=artifacts.earnings_exclusion_diagnostic,
    )
    integrity = verify_canonical_baseline_experiment_integrity(result)
    if not integrity.integrity_verified:
        failed = tuple(
            check.check_id for check in integrity.checks if not check.passed
        )
        raise CanonicalRunAuthorizationError(
            "POSTRUN_INTEGRITY_VERIFICATION_FAILED",
            f"the preserved canonical result failed integrity checks: {failed}",
        )
    return result


__all__ = [
    "CANONICAL_RUN_AUTHORIZATION_SCHEMA_VERSION",
    "CANONICAL_RUN_AUTHORIZATION_TOKEN",
    "CanonicalBaselineExperimentArtifacts",
    "CanonicalRunAuthorization",
    "execute_canonical_baseline_experiment",
]
