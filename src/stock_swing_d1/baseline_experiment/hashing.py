"""Domain-separated deterministic SHA-256 helpers for Phase 16C.

Every domain string equals the schema version of the object it identifies, the
uniform convention already used by Phase 15D and Phase 16B. A digest computed
under one domain therefore never collides with a digest computed under another,
even when the payloads happen to coincide.

No timestamp, process identity, filesystem path, insertion order or Python
``hash()`` participates in any digest here: Phase 16C fingerprints are pure
functions of frozen semantic content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stock_swing_d1.backtest_results.hashing import (
    semantic_domain_sha256,
    semantic_sha256,
)

if TYPE_CHECKING:
    from stock_swing_d1.baseline_experiment.earnings_diagnostic import (
        EarningsFilterExclusionDiagnostic,
    )
    from stock_swing_d1.baseline_experiment.gap_diagnostic import (
        MaterialAdverseOvernightGapDiagnostic,
    )
    from stock_swing_d1.baseline_experiment.gap_policy import (
        MaterialAdverseOvernightGapPolicy,
    )
    from stock_swing_d1.baseline_experiment.manifest import (
        CanonicalBaselineExperimentManifest,
    )
    from stock_swing_d1.baseline_experiment.result import (
        CanonicalBaselineExperimentResult,
    )


MATERIAL_ADVERSE_OVERNIGHT_GAP_POLICY_HASH_DOMAIN = (
    "material_adverse_overnight_gap_policy.v0.1"
)
MATERIAL_ADVERSE_OVERNIGHT_GAP_DIAGNOSTIC_HASH_DOMAIN = (
    "material_adverse_overnight_gap_diagnostic.v0.1"
)
EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_HASH_DOMAIN = (
    "earnings_filter_exclusion_diagnostic.v0.1"
)
CANONICAL_BASELINE_EXPERIMENT_MANIFEST_HASH_DOMAIN = (
    "canonical_baseline_experiment_manifest.v0.2"
)
CANONICAL_BASELINE_EXPERIMENT_RESULT_HASH_DOMAIN = (
    "canonical_baseline_experiment_result.v0.2"
)


def _compute_gap_policy_fingerprint(**values: object) -> str:
    return semantic_domain_sha256(
        MATERIAL_ADVERSE_OVERNIGHT_GAP_POLICY_HASH_DOMAIN, dict(values)
    )


def compute_gap_policy_fingerprint(
    policy: MaterialAdverseOvernightGapPolicy,
) -> str:
    """Fingerprint every semantic field of one frozen gap-metric policy."""

    from stock_swing_d1.baseline_experiment.gap_policy import (
        MaterialAdverseOvernightGapPolicy,
    )

    if type(policy) is not MaterialAdverseOvernightGapPolicy:
        raise TypeError("policy must be a MaterialAdverseOvernightGapPolicy")
    return _compute_gap_policy_fingerprint(
        **{
            name: getattr(policy, name)
            for name in MaterialAdverseOvernightGapPolicy.model_fields
            if name != "policy_fingerprint"
        }
    )


def _compute_gap_diagnostic_fingerprint(**values: object) -> str:
    return semantic_domain_sha256(
        MATERIAL_ADVERSE_OVERNIGHT_GAP_DIAGNOSTIC_HASH_DOMAIN, dict(values)
    )


def compute_gap_diagnostic_fingerprint(
    diagnostic: MaterialAdverseOvernightGapDiagnostic,
) -> str:
    """Fingerprint one canonical gap-loss-frequency diagnostic."""

    from stock_swing_d1.baseline_experiment.gap_diagnostic import (
        MaterialAdverseOvernightGapDiagnostic,
    )

    if type(diagnostic) is not MaterialAdverseOvernightGapDiagnostic:
        raise TypeError(
            "diagnostic must be a MaterialAdverseOvernightGapDiagnostic"
        )
    return _compute_gap_diagnostic_fingerprint(
        **{
            name: getattr(diagnostic, name)
            for name in MaterialAdverseOvernightGapDiagnostic.model_fields
            if name != "diagnostic_fingerprint"
        }
    )


def _compute_earnings_exclusion_diagnostic_fingerprint(**values: object) -> str:
    return semantic_domain_sha256(
        EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_HASH_DOMAIN, dict(values)
    )


def compute_earnings_exclusion_diagnostic_fingerprint(
    diagnostic: EarningsFilterExclusionDiagnostic,
) -> str:
    """Fingerprint one canonical earnings-filter-exclusion diagnostic."""

    from stock_swing_d1.baseline_experiment.earnings_diagnostic import (
        EarningsFilterExclusionDiagnostic,
    )

    if type(diagnostic) is not EarningsFilterExclusionDiagnostic:
        raise TypeError(
            "diagnostic must be an EarningsFilterExclusionDiagnostic"
        )
    return _compute_earnings_exclusion_diagnostic_fingerprint(
        **{
            name: getattr(diagnostic, name)
            for name in EarningsFilterExclusionDiagnostic.model_fields
            if name != "diagnostic_fingerprint"
        }
    )


def _compute_manifest_fingerprint(**values: object) -> str:
    return semantic_domain_sha256(
        CANONICAL_BASELINE_EXPERIMENT_MANIFEST_HASH_DOMAIN, dict(values)
    )


def compute_manifest_fingerprint(
    manifest: CanonicalBaselineExperimentManifest,
) -> str:
    """Fingerprint every semantic field of one canonical experiment manifest.

    The manifest declares no audit-only field, so the excluded set is exactly
    ``manifest_fingerprint`` itself. There is nothing volatile to exclude.
    """

    from stock_swing_d1.baseline_experiment.manifest import (
        CanonicalBaselineExperimentManifest,
    )

    if type(manifest) is not CanonicalBaselineExperimentManifest:
        raise TypeError(
            "manifest must be a CanonicalBaselineExperimentManifest"
        )
    return _compute_manifest_fingerprint(
        **{
            name: getattr(manifest, name)
            for name in CanonicalBaselineExperimentManifest.model_fields
            if name != "manifest_fingerprint"
        }
    )


def _compute_experiment_result_fingerprint(**values: object) -> str:
    return semantic_domain_sha256(
        CANONICAL_BASELINE_EXPERIMENT_RESULT_HASH_DOMAIN, dict(values)
    )


def compute_experiment_result_fingerprint(
    result: CanonicalBaselineExperimentResult,
) -> str:
    """Fingerprint one preserved canonical baseline experiment result."""

    from stock_swing_d1.baseline_experiment.result import (
        CanonicalBaselineExperimentResult,
    )

    if type(result) is not CanonicalBaselineExperimentResult:
        raise TypeError("result must be a CanonicalBaselineExperimentResult")
    return _compute_experiment_result_fingerprint(
        **{
            name: getattr(result, name)
            for name in CanonicalBaselineExperimentResult.model_fields
            if name != "result_fingerprint"
        }
    )


__all__ = [
    "CANONICAL_BASELINE_EXPERIMENT_MANIFEST_HASH_DOMAIN",
    "CANONICAL_BASELINE_EXPERIMENT_RESULT_HASH_DOMAIN",
    "EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_HASH_DOMAIN",
    "MATERIAL_ADVERSE_OVERNIGHT_GAP_DIAGNOSTIC_HASH_DOMAIN",
    "MATERIAL_ADVERSE_OVERNIGHT_GAP_POLICY_HASH_DOMAIN",
    "compute_earnings_exclusion_diagnostic_fingerprint",
    "compute_experiment_result_fingerprint",
    "compute_gap_diagnostic_fingerprint",
    "compute_gap_policy_fingerprint",
    "compute_manifest_fingerprint",
    "semantic_domain_sha256",
    "semantic_sha256",
]
