"""Frozen Phase 16C material-adverse-overnight-gap exposure diagnostic v0.1.

This is a Phase 16C-owned experiment diagnostic, not a Phase 16B metric: it
measures overnight gap *exposure* of completed canonical trades, and Phase 16B
neither owns nor computes anything equivalent. It is therefore computed here
rather than consumed, but it never re-derives a Phase 16B statistic and never
touches portfolio accounting.

Two evidence sources meet here and neither is reconstructed:

* the authoritative completed-trade episodes of an accepted Phase 15D audited
  result supply the trade population and each trade's entry/exit sessions; and
* caller-supplied canonical unadjusted boundary evidence supplies the two
  tradable prices plus the frozen corporate-action and ordinary-dividend
  normalization terms for each overnight boundary.

All arithmetic runs under the frozen Phase 16B.2 canonical statistical Decimal
context. No binary floating-point value participates anywhere on this path:
the boundary evidence model rejects one outright, and the canonical context
additionally traps ``FloatOperation``.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, model_validator

from stock_swing_d1.backtest_results import (
    HistoricalBacktestAuditResult,
    HistoricalClosedTradeRecord,
)
from stock_swing_d1.research_metrics.arithmetic import (
    canonical_statistical_context,
    divide,
    multiply,
    quantize_canonical_metric,
    subtract,
)

from stock_swing_d1.baseline_experiment.errors import (
    BaselineExperimentDiagnosticError,
)
from stock_swing_d1.baseline_experiment.field_types import (
    CanonicalSecurityId,
    CanonicalText,
    ExactBool,
    ImmutableBaselineExperimentModel,
    NonNegativeCount,
    OptionalCanonicalText,
    OptionalFiniteDecimal,
    OptionalNonNegativeCount,
    OptionalNonNegativeDecimal,
    OptionalPositiveCount,
    OptionalPositiveDecimal,
    SessionDate,
    Sha256,
    require_exact_enum,
    require_optional_exact_enum,
    require_tuple,
)
from stock_swing_d1.baseline_experiment.gap_policy import (
    GAP_MATERIALITY_THRESHOLD,
    GAP_MATERIALITY_THRESHOLD_TEXT,
    MaterialAdverseOvernightGapPolicy,
    MaterialAdverseOvernightGapPolicyRef,
    build_material_adverse_overnight_gap_policy_ref,
)
from stock_swing_d1.baseline_experiment.hashing import (
    _compute_gap_diagnostic_fingerprint,
    compute_gap_diagnostic_fingerprint,
)


OVERNIGHT_BOUNDARY_EVIDENCE_SCHEMA_VERSION = (
    "overnight_boundary_evidence.v0.1"
)
COMPLETED_TRADE_GAP_EVIDENCE_SCHEMA_VERSION = (
    "completed_trade_gap_evidence.v0.1"
)
COMPLETED_TRADE_GAP_OBSERVATION_SCHEMA_VERSION = (
    "completed_trade_gap_observation.v0.1"
)
MATERIAL_ADVERSE_OVERNIGHT_GAP_DIAGNOSTIC_SCHEMA_VERSION = (
    "material_adverse_overnight_gap_diagnostic.v0.1"
)

_ZERO = Decimal("0")
_ONE = Decimal("1")


class OvernightBoundaryEvidenceStatus(StrEnum):
    """Whether one overnight boundary carries every required canonical fact."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class GapDiagnosticUndefinedReason(StrEnum):
    """The complete frozen v0.1 undefined-diagnostic reason family.

    The family is closed. An undefined gap-loss frequency is never published as
    zero, and a structurally invalid evidence set is never disguised as one of
    these: that raises instead.
    """

    NO_OVERNIGHT_COMPLETED_TRADES = "NO_OVERNIGHT_COMPLETED_TRADES"
    INCOMPLETE_GAP_EVIDENCE = "INCOMPLETE_GAP_EVIDENCE"


_BoundaryStatus = Annotated[
    OvernightBoundaryEvidenceStatus,
    BeforeValidator(require_exact_enum(OvernightBoundaryEvidenceStatus)),
]
_OptionalUndefinedReason = Annotated[
    GapDiagnosticUndefinedReason | None,
    BeforeValidator(require_optional_exact_enum(GapDiagnosticUndefinedReason)),
]


class OvernightBoundaryEvidence(ImmutableBaselineExperimentModel):
    """Canonical unadjusted evidence for one overnight boundary.

    ``previous_regular_session_close`` and ``current_regular_session_open`` are
    canonical *unadjusted* tradable regular-session prices. An adjusted-close
    series may never be substituted for either: the frozen contract normalizes
    mechanical effects explicitly, through the two terms below, rather than
    implicitly through a vendor-adjusted price.

    ``share_basis_new_shares`` / ``share_basis_old_shares`` express the frozen
    canonical share-basis change ``R = new shares / old shares`` becoming
    effective at the current session, if any. ``R > 1`` for a split, ``R < 1``
    for a reverse split. The prior close is re-expressed onto the
    current-session share basis by dividing by ``R``.

    ``ordinary_dividend_amount_per_share`` is the canonical cash distribution
    per entitled share, on the current-session share basis, for an ordinary
    dividend becoming economically effective across this boundary under the
    already-frozen ordinary-dividend semantics. It reduces the reference close.

    ``INCOMPLETE`` means at least one required fact could not be established.
    Such a boundary carries a ``missing_evidence_code`` and, when it is
    eligible, forces the whole diagnostic to be explicitly undefined rather
    than guessed at or silently omitted.
    """

    schema_version: Literal[
        "overnight_boundary_evidence.v0.1"
    ] = OVERNIGHT_BOUNDARY_EVIDENCE_SCHEMA_VERSION
    security_id: CanonicalSecurityId
    previous_session: SessionDate
    current_session: SessionDate
    evidence_status: _BoundaryStatus
    previous_regular_session_close: OptionalPositiveDecimal = None
    current_regular_session_open: OptionalPositiveDecimal = None
    share_basis_new_shares: OptionalPositiveCount = None
    share_basis_old_shares: OptionalPositiveCount = None
    ordinary_dividend_amount_per_share: OptionalNonNegativeDecimal = None
    missing_evidence_code: OptionalCanonicalText = None

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        if self.current_session <= self.previous_session:
            raise ValueError(
                "current_session must be strictly after previous_session"
            )
        share_basis_present = (
            self.share_basis_new_shares is not None,
            self.share_basis_old_shares is not None,
        )
        if any(share_basis_present) and not all(share_basis_present):
            raise ValueError(
                "share_basis_new_shares and share_basis_old_shares must both "
                "be present or both absent"
            )
        if self.evidence_status is OvernightBoundaryEvidenceStatus.COMPLETE:
            if (
                self.previous_regular_session_close is None
                or self.current_regular_session_open is None
            ):
                raise ValueError(
                    "a COMPLETE boundary requires both canonical unadjusted "
                    "regular-session prices"
                )
            if self.missing_evidence_code is not None:
                raise ValueError(
                    "a COMPLETE boundary must not name missing evidence"
                )
            return self
        if self.missing_evidence_code is None:
            raise ValueError(
                "an INCOMPLETE boundary must name the missing canonical evidence"
            )
        return self


class CompletedTradeGapEvidence(ImmutableBaselineExperimentModel):
    """Every observed overnight boundary spanning one completed trade.

    The assembler supplies the boundaries it observed across the trade's own
    session span. Which of them are *eligible* is decided here, from the
    authoritative entry and exit sessions of the audited trade episode, so a
    pre-entry or post-exit boundary can never be counted even when supplied.
    """

    schema_version: Literal[
        "completed_trade_gap_evidence.v0.1"
    ] = COMPLETED_TRADE_GAP_EVIDENCE_SCHEMA_VERSION
    trade_id: CanonicalText
    security_id: CanonicalSecurityId
    boundaries: Annotated[
        tuple[OvernightBoundaryEvidence, ...], BeforeValidator(require_tuple)
    ] = ()

    @model_validator(mode="after")
    def validate_boundaries(self) -> Self:
        keys = tuple(
            (row.previous_session, row.current_session)
            for row in self.boundaries
        )
        if keys != tuple(sorted(keys)):
            raise ValueError("boundaries must be ordered by session")
        if len(set(keys)) != len(keys):
            raise ValueError("boundaries must be unique by session pair")
        if any(row.security_id != self.security_id for row in self.boundaries):
            raise ValueError(
                "every boundary must belong to the trade's security"
            )
        return self


class CompletedTradeGapObservation(ImmutableBaselineExperimentModel):
    """The per-trade audit trail behind one gap-loss-frequency numerator."""

    schema_version: Literal[
        "completed_trade_gap_observation.v0.1"
    ] = COMPLETED_TRADE_GAP_OBSERVATION_SCHEMA_VERSION
    trade_id: CanonicalText
    security_id: CanonicalSecurityId
    eligible_boundary_count: NonNegativeCount
    ineligible_boundary_count: NonNegativeCount
    qualifying_gap_event_count: NonNegativeCount
    most_adverse_gap_return: OptionalFiniteDecimal = None
    counted_in_numerator: ExactBool

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.qualifying_gap_event_count > self.eligible_boundary_count:
            raise ValueError(
                "qualifying gap events cannot exceed eligible boundaries"
            )
        if self.counted_in_numerator != (self.qualifying_gap_event_count > 0):
            raise ValueError(
                "a trade is counted once exactly when it contains at least one "
                "qualifying material adverse overnight gap"
            )
        if (self.most_adverse_gap_return is None) != (
            self.eligible_boundary_count == 0
        ):
            raise ValueError(
                "a most adverse gap return exists exactly when the trade has "
                "at least one eligible overnight boundary"
            )
        return self


class MaterialAdverseOvernightGapDiagnostic(ImmutableBaselineExperimentModel):
    """One immutable, fingerprint-bound canonical gap-loss-frequency result."""

    schema_version: Literal[
        "material_adverse_overnight_gap_diagnostic.v0.1"
    ] = MATERIAL_ADVERSE_OVERNIGHT_GAP_DIAGNOSTIC_SCHEMA_VERSION
    policy_ref: MaterialAdverseOvernightGapPolicyRef
    source_audit_result_fingerprint: Sha256
    completed_trade_count: NonNegativeCount
    overnight_completed_trade_count: OptionalNonNegativeCount = None
    gap_loss_trade_count: OptionalNonNegativeCount = None
    qualifying_gap_event_count: OptionalNonNegativeCount = None
    value: OptionalFiniteDecimal = None
    undefined_reason: _OptionalUndefinedReason = None
    trade_observations: Annotated[
        tuple[CompletedTradeGapObservation, ...], BeforeValidator(require_tuple)
    ] = ()
    diagnostic_fingerprint: Sha256

    @model_validator(mode="after")
    def validate_diagnostic(self) -> Self:
        if (self.value is None) == (self.undefined_reason is None):
            raise ValueError(
                "the diagnostic is either defined with a finite value and no "
                "reason, or undefined with exactly one frozen reason and no value"
            )
        counts = (
            self.overnight_completed_trade_count,
            self.gap_loss_trade_count,
            self.qualifying_gap_event_count,
        )
        if (
            self.undefined_reason
            is GapDiagnosticUndefinedReason.INCOMPLETE_GAP_EVIDENCE
        ):
            if any(count is not None for count in counts):
                raise ValueError(
                    "incomplete gap evidence establishes no numerator, "
                    "denominator or event count"
                )
            if self.trade_observations:
                raise ValueError(
                    "incomplete gap evidence publishes no per-trade observations"
                )
            return self._validate_fingerprint()
        if any(count is None for count in counts):
            raise ValueError(
                "a defined or zero-denominator diagnostic requires every count"
            )
        assert self.overnight_completed_trade_count is not None
        assert self.gap_loss_trade_count is not None
        if self.gap_loss_trade_count > self.overnight_completed_trade_count:
            raise ValueError(
                "gap_loss_trade_count cannot exceed "
                "overnight_completed_trade_count"
            )
        if self.overnight_completed_trade_count > self.completed_trade_count:
            raise ValueError(
                "overnight_completed_trade_count cannot exceed "
                "completed_trade_count"
            )
        if len(self.trade_observations) != self.completed_trade_count:
            raise ValueError(
                "one observation must be published per completed trade"
            )
        if (
            self.undefined_reason
            is GapDiagnosticUndefinedReason.NO_OVERNIGHT_COMPLETED_TRADES
        ) != (self.overnight_completed_trade_count == 0):
            raise ValueError(
                "a zero overnight denominator is undefined, never zero, and a "
                "positive denominator is never undefined for that reason"
            )
        return self._validate_fingerprint()

    def _validate_fingerprint(self) -> Self:
        if self.diagnostic_fingerprint != compute_gap_diagnostic_fingerprint(
            self
        ):
            raise ValueError(
                "diagnostic_fingerprint does not match diagnostic content"
            )
        return self


def _fail(code: str, message: str) -> None:
    raise BaselineExperimentDiagnosticError(code, message)


def _economic_reference_close(
    boundary: OvernightBoundaryEvidence, *, context
) -> Decimal:
    """Re-express the prior close onto the current-session economic basis.

    A mechanical share-basis change or an ordinary ex-dividend adjustment is
    removed here so it can never be misclassified as an uncompensated adverse
    market gap.
    """

    reference = boundary.previous_regular_session_close
    assert reference is not None
    if boundary.share_basis_new_shares is not None:
        assert boundary.share_basis_old_shares is not None
        reference = divide(
            multiply(
                reference,
                Decimal(boundary.share_basis_old_shares),
                context=context,
            ),
            Decimal(boundary.share_basis_new_shares),
            context=context,
        )
    if boundary.ordinary_dividend_amount_per_share is not None:
        reference = subtract(
            reference,
            boundary.ordinary_dividend_amount_per_share,
            context=context,
        )
    if reference <= _ZERO:
        _fail(
            "NONPOSITIVE_ECONOMIC_REFERENCE_CLOSE",
            "a normalized reference close must remain strictly positive",
        )
    return reference


def _gap_return(boundary: OvernightBoundaryEvidence, *, context) -> Decimal:
    """Compute one canonical overnight gap return under exact Decimal rules."""

    reference = _economic_reference_close(boundary, context=context)
    current_open = boundary.current_regular_session_open
    assert current_open is not None
    return quantize_canonical_metric(
        subtract(
            divide(current_open, reference, context=context),
            _ONE,
            context=context,
        ),
        context=context,
    )


def _is_eligible(
    boundary: OvernightBoundaryEvidence,
    *,
    entry_session,
    exit_session,
) -> bool:
    """Apply the frozen eligibility rule to one supplied boundary.

    A boundary is eligible only when the position was already open at the
    previous regular-session close and remained economically exposed through
    the current regular-session open. Entry executes at the open of
    ``entry_session``, so the position is open at that session's close; the
    exit occurs during ``exit_session``, so the opening gap of the exit session
    is still an exposed boundary while the following one is not.
    """

    return (
        boundary.previous_session >= entry_session
        and boundary.current_session <= exit_session
    )


def build_material_adverse_overnight_gap_diagnostic(
    *,
    source_result: HistoricalBacktestAuditResult,
    policy: MaterialAdverseOvernightGapPolicy,
    trade_gap_evidence: tuple[CompletedTradeGapEvidence, ...],
) -> MaterialAdverseOvernightGapDiagnostic:
    """Measure frozen v0.1 gap-loss frequency over one audited experiment."""

    if type(source_result) is not HistoricalBacktestAuditResult:
        raise TypeError("source_result must be a HistoricalBacktestAuditResult")
    if type(policy) is not MaterialAdverseOvernightGapPolicy:
        raise TypeError("policy must be a MaterialAdverseOvernightGapPolicy")
    if type(trade_gap_evidence) is not tuple or any(
        type(item) is not CompletedTradeGapEvidence
        for item in trade_gap_evidence
    ):
        raise TypeError(
            "trade_gap_evidence must be a tuple of CompletedTradeGapEvidence"
        )
    if (
        policy.materiality_threshold != GAP_MATERIALITY_THRESHOLD_TEXT
        or Decimal(policy.materiality_threshold) != GAP_MATERIALITY_THRESHOLD
    ):
        _fail(
            "NONCANONICAL_MATERIALITY_THRESHOLD",
            "the frozen v0.1 threshold is the only admissible threshold",
        )

    completed = tuple(
        record
        for record in source_result.trades
        if type(record) is HistoricalClosedTradeRecord
    )
    by_trade_id: dict[str, CompletedTradeGapEvidence] = {}
    for evidence in trade_gap_evidence:
        if evidence.trade_id in by_trade_id:
            _fail(
                "DUPLICATE_TRADE_GAP_EVIDENCE",
                f"trade {evidence.trade_id} has more than one evidence record",
            )
        by_trade_id[evidence.trade_id] = evidence
    known_trade_ids = {record.trade_id for record in completed}
    unknown = sorted(set(by_trade_id) - known_trade_ids)
    if unknown:
        _fail(
            "UNKNOWN_TRADE_GAP_EVIDENCE",
            f"evidence supplied for non-completed trades: {unknown}",
        )

    context = canonical_statistical_context()
    observations: list[CompletedTradeGapObservation] = []
    incomplete = False
    for record in completed:
        evidence = by_trade_id.get(record.trade_id)
        if evidence is None:
            incomplete = True
            continue
        if evidence.security_id != record.security_id:
            _fail(
                "TRADE_GAP_EVIDENCE_SECURITY_MISMATCH",
                f"trade {record.trade_id} evidence names another security",
            )
        eligible = tuple(
            boundary
            for boundary in evidence.boundaries
            if _is_eligible(
                boundary,
                entry_session=record.entry_session,
                exit_session=record.exit_session,
            )
        )
        if any(
            boundary.evidence_status
            is OvernightBoundaryEvidenceStatus.INCOMPLETE
            for boundary in eligible
        ):
            incomplete = True
            continue
        returns = tuple(
            _gap_return(boundary, context=context) for boundary in eligible
        )
        qualifying = tuple(
            value for value in returns if value <= GAP_MATERIALITY_THRESHOLD
        )
        observations.append(
            CompletedTradeGapObservation(
                trade_id=record.trade_id,
                security_id=record.security_id,
                eligible_boundary_count=len(eligible),
                ineligible_boundary_count=(
                    len(evidence.boundaries) - len(eligible)
                ),
                qualifying_gap_event_count=len(qualifying),
                most_adverse_gap_return=(min(returns) if returns else None),
                counted_in_numerator=bool(qualifying),
            )
        )

    policy_ref = build_material_adverse_overnight_gap_policy_ref(policy)
    if incomplete:
        return _build_diagnostic(
            policy_ref=policy_ref,
            source_audit_result_fingerprint=source_result.result_fingerprint,
            completed_trade_count=len(completed),
            overnight_completed_trade_count=None,
            gap_loss_trade_count=None,
            qualifying_gap_event_count=None,
            value=None,
            undefined_reason=(
                GapDiagnosticUndefinedReason.INCOMPLETE_GAP_EVIDENCE
            ),
            trade_observations=(),
        )

    frozen_observations = tuple(observations)
    denominator = sum(
        1
        for observation in frozen_observations
        if observation.eligible_boundary_count > 0
    )
    numerator = sum(
        1
        for observation in frozen_observations
        if observation.counted_in_numerator
    )
    events = sum(
        observation.qualifying_gap_event_count
        for observation in frozen_observations
    )
    if denominator == 0:
        return _build_diagnostic(
            policy_ref=policy_ref,
            source_audit_result_fingerprint=source_result.result_fingerprint,
            completed_trade_count=len(completed),
            overnight_completed_trade_count=0,
            gap_loss_trade_count=0,
            qualifying_gap_event_count=0,
            value=None,
            undefined_reason=(
                GapDiagnosticUndefinedReason.NO_OVERNIGHT_COMPLETED_TRADES
            ),
            trade_observations=frozen_observations,
        )
    frequency = quantize_canonical_metric(
        divide(Decimal(numerator), Decimal(denominator), context=context),
        context=context,
    )
    return _build_diagnostic(
        policy_ref=policy_ref,
        source_audit_result_fingerprint=source_result.result_fingerprint,
        completed_trade_count=len(completed),
        overnight_completed_trade_count=denominator,
        gap_loss_trade_count=numerator,
        qualifying_gap_event_count=events,
        value=frequency,
        undefined_reason=None,
        trade_observations=frozen_observations,
    )


def _build_diagnostic(**values: object) -> MaterialAdverseOvernightGapDiagnostic:
    payload = {
        "schema_version": (
            MATERIAL_ADVERSE_OVERNIGHT_GAP_DIAGNOSTIC_SCHEMA_VERSION
        ),
        **values,
    }
    return MaterialAdverseOvernightGapDiagnostic(
        **payload,
        diagnostic_fingerprint=_compute_gap_diagnostic_fingerprint(**payload),
    )


__all__ = [
    "COMPLETED_TRADE_GAP_EVIDENCE_SCHEMA_VERSION",
    "COMPLETED_TRADE_GAP_OBSERVATION_SCHEMA_VERSION",
    "MATERIAL_ADVERSE_OVERNIGHT_GAP_DIAGNOSTIC_SCHEMA_VERSION",
    "OVERNIGHT_BOUNDARY_EVIDENCE_SCHEMA_VERSION",
    "CompletedTradeGapEvidence",
    "CompletedTradeGapObservation",
    "GapDiagnosticUndefinedReason",
    "MaterialAdverseOvernightGapDiagnostic",
    "OvernightBoundaryEvidence",
    "OvernightBoundaryEvidenceStatus",
    "build_material_adverse_overnight_gap_diagnostic",
]
