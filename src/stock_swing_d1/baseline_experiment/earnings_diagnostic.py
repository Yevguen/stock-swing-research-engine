"""Frozen Phase 16C earnings-filter exclusion diagnostic v0.1.

The diagnostic counts *otherwise-evaluated candidate entries whose canonical
point-in-time earnings-risk decision prohibited entry*.

It is read directly from the authoritative Phase 15D signal-provenance rows,
each of which preserves the earnings-risk action that the canonical PIT policy
actually returned at that historical decision time. Phase 16C therefore never
reconstructs earnings state, never consults an earnings query, and never
revisits a decision with knowledge that arrived later: a revision published
after a decision session cannot reach this counter at all, because the counter
reads only the immutable rows the run itself recorded.

Two exclusions follow from the definition and are enforced structurally:

* a candidate the earnings policy allowed but that failed some unrelated filter
  is not counted, because only ``ENTRY_BLOCKED`` increments the numerator; and
* a row the earnings entry-eligibility policy never evaluated -- a pending-entry
  revalidation, an open-position management action -- enters neither the
  numerator nor the denominator.

The denominator is preserved as context only. The frozen contract did not
freeze an exclusion *rate*, so none is published here.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from stock_swing_d1.backtest_results import HistoricalBacktestAuditResult
from stock_swing_d1.earnings.integration.models import (
    EarningsIntegrationAction,
)

from stock_swing_d1.baseline_experiment.errors import (
    BaselineExperimentDiagnosticError,
)
from stock_swing_d1.baseline_experiment.field_types import (
    ImmutableBaselineExperimentModel,
    NonNegativeCount,
    Sha256,
)
from stock_swing_d1.baseline_experiment.hashing import (
    _compute_earnings_exclusion_diagnostic_fingerprint,
    compute_earnings_exclusion_diagnostic_fingerprint,
)


EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_SCHEMA_VERSION = (
    "earnings_filter_exclusion_diagnostic.v0.1"
)
EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_VERSION = "earnings_filter_exclusions.v0.1"

# The two actions through which the canonical PIT earnings-risk policy answers
# the entry-eligibility question for one evaluated candidate entry. Every other
# member of the vocabulary answers a different question (pending-entry
# revalidation, open-position management) and is therefore outside this
# diagnostic entirely.
_ENTRY_ELIGIBILITY_ACTIONS = frozenset(
    {
        EarningsIntegrationAction.ENTRY_ALLOWED,
        EarningsIntegrationAction.ENTRY_BLOCKED,
    }
)


class EarningsFilterExclusionDiagnostic(ImmutableBaselineExperimentModel):
    """One immutable, fingerprint-bound earnings-filter exclusion count."""

    schema_version: Literal[
        "earnings_filter_exclusion_diagnostic.v0.1"
    ] = EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_SCHEMA_VERSION
    diagnostic_version: Literal[
        "earnings_filter_exclusions.v0.1"
    ] = EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_VERSION
    source_audit_result_fingerprint: Sha256
    earnings_filter_exclusions: NonNegativeCount
    earnings_evaluated_entry_candidate_count: NonNegativeCount
    diagnostic_fingerprint: Sha256

    @model_validator(mode="after")
    def validate_diagnostic(self) -> Self:
        if (
            self.earnings_filter_exclusions
            > self.earnings_evaluated_entry_candidate_count
        ):
            raise ValueError(
                "exclusions cannot exceed the candidates the earnings policy "
                "actually evaluated"
            )
        expected = compute_earnings_exclusion_diagnostic_fingerprint(self)
        if self.diagnostic_fingerprint != expected:
            raise ValueError(
                "diagnostic_fingerprint does not match diagnostic content"
            )
        return self


def build_earnings_filter_exclusion_diagnostic(
    *, source_result: HistoricalBacktestAuditResult
) -> EarningsFilterExclusionDiagnostic:
    """Count exclusions from authoritative point-in-time earnings decisions."""

    if type(source_result) is not HistoricalBacktestAuditResult:
        raise TypeError("source_result must be a HistoricalBacktestAuditResult")

    evaluated = 0
    exclusions = 0
    for row in source_result.signal_provenance:
        action = row.earnings_action
        if action not in _ENTRY_ELIGIBILITY_ACTIONS:
            continue
        allowed = action is EarningsIntegrationAction.ENTRY_ALLOWED
        if row.earnings_entry_allowed is not allowed:
            raise BaselineExperimentDiagnosticError(
                "INCONSISTENT_EARNINGS_PROVENANCE",
                "an authoritative signal row disagrees with its own earnings "
                "entry-eligibility action",
            )
        evaluated += 1
        if not allowed:
            exclusions += 1

    values = {
        "schema_version": EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_version": EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_VERSION,
        "source_audit_result_fingerprint": source_result.result_fingerprint,
        "earnings_filter_exclusions": exclusions,
        "earnings_evaluated_entry_candidate_count": evaluated,
    }
    return EarningsFilterExclusionDiagnostic(
        **values,
        diagnostic_fingerprint=(
            _compute_earnings_exclusion_diagnostic_fingerprint(**values)
        ),
    )


__all__ = [
    "EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_SCHEMA_VERSION",
    "EARNINGS_FILTER_EXCLUSION_DIAGNOSTIC_VERSION",
    "EarningsFilterExclusionDiagnostic",
    "build_earnings_filter_exclusion_diagnostic",
]
