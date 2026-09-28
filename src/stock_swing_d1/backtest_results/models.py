"""Immutable foundational contracts for Phase 15D historical audit results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    model_validator,
)

from stock_swing_d1.earnings.integration.models import EarningsIntegrationAction
from stock_swing_d1.execution.costs.models import ExecutionCostPolicyRef
from stock_swing_d1.execution.entry.models import EntryExecutionStatus
from stock_swing_d1.portfolio.models import PortfolioCandidateAction
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    add_exact_decimal,
    exact_decimal_times_int,
    subtract_exact_decimal,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.ranking.models import RankingPolicyRef
from stock_swing_d1.strategy.baseline.models import BaselineSignalAction
from stock_swing_d1.valuation import (
    PORTFOLIO_VALUATION_POLICY_ID,
    PORTFOLIO_VALUATION_POLICY_SCHEMA_VERSION,
    PortfolioValuationMark,
    PortfolioValuationPolicy,
)
from stock_swing_d1.backtester.decision_interval import (
    HistoricalDecisionInterval,
)

from stock_swing_d1.backtest_results.hashing import (
    RUN_CONFIGURATION_HASH_DOMAIN,
    VALUATION_POLICY_HASH_DOMAIN,
    VALUATION_SNAPSHOT_HASH_DOMAIN,
    compute_result_fingerprint,
    semantic_domain_sha256,
)


# Task 5C-A relocated the valuation *policy* and *mark* models upstream into
# ``valuation`` so runtime and audit share one frozen statement of valuation
# semantics.  The names below are re-exported aliases bound to the SAME class
# objects and the SAME identity strings, never re-definitions: relocation must
# not change a single field, digest or public import path.
HISTORICAL_BACKTEST_VALUATION_POLICY_SCHEMA_VERSION = (
    PORTFOLIO_VALUATION_POLICY_SCHEMA_VERSION
)
HISTORICAL_BACKTEST_VALUATION_POLICY_ID = PORTFOLIO_VALUATION_POLICY_ID
HISTORICAL_BACKTEST_RUN_MANIFEST_SCHEMA_VERSION = (
    "historical_backtest_run_manifest.v0.1"
)
HISTORICAL_BACKTEST_AUDIT_RESULT_SCHEMA_VERSION = (
    "historical_backtest_audit_result.v0.2"
)


def _require_canonical_text(value: object) -> object:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("must be canonical non-empty text")
    return value


def _require_optional_canonical_text(value: object) -> object:
    if value is None:
        return value
    return _require_canonical_text(value)


def _require_session_date(value: object) -> object:
    if type(value) is not date:
        raise ValueError("must be a Python datetime.date")
    return value


def _require_aware_datetime(value: object) -> object:
    try:
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError("must be an explicitly timezone-aware datetime")
    except (OverflowError, TypeError) as error:
        raise ValueError(
            "must be an explicitly timezone-aware datetime"
        ) from error
    return value


def _require_decimal(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError("must be a Decimal; binary floats, strings, and bool are forbidden")
    if not value.is_finite():
        raise ValueError("must be a finite Decimal")
    return value


_subtract_exact = subtract_exact_decimal


def _require_finite_float(value: object) -> object:
    if type(value) is not float:
        raise ValueError("must be a binary float")
    if not isfinite(value):
        raise ValueError("must be finite")
    return value


def _require_tuple(value: object) -> object:
    if type(value) is not tuple:
        raise ValueError("must be an immutable tuple")
    return value


def _require_ranking_policy_ref(value: object) -> object:
    if type(value) is not RankingPolicyRef:
        raise ValueError("must be an authoritative RankingPolicyRef")
    rebuilt = RankingPolicyRef.model_validate(value.model_dump(mode="python"))
    _require_canonical_text(rebuilt.policy_id)
    _require_canonical_text(rebuilt.policy_version)
    return rebuilt


def _require_execution_cost_policy_ref(value: object) -> object:
    if type(value) is not ExecutionCostPolicyRef:
        raise ValueError("must be an authoritative ExecutionCostPolicyRef")
    return ExecutionCostPolicyRef(
        **{field.name: getattr(value, field.name) for field in fields(value)}
    )


def _require_portfolio_state(value: object) -> object:
    if type(value) is not PortfolioState:
        raise ValueError("must be an authoritative PortfolioState")
    return PortfolioState.model_validate(value.model_dump(mode="python"))


def _require_decision_interval(value: object) -> object:
    if type(value) is not HistoricalDecisionInterval:
        raise ValueError("must be exactly HistoricalDecisionInterval")
    return HistoricalDecisionInterval.model_validate(
        {
            field_name: getattr(value, field_name)
            for field_name in HistoricalDecisionInterval.model_fields
        }
    )


class _ImmutableBacktestResultModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_default=True,
    )


_CanonicalText = Annotated[str, BeforeValidator(_require_canonical_text)]
_OptionalCanonicalText = Annotated[
    str | None, BeforeValidator(_require_optional_canonical_text)
]
_CanonicalSecurityId = Annotated[
    str,
    BeforeValidator(_require_canonical_text),
    Field(pattern=r"^NORGATE:[1-9][0-9]*$"),
]
_Sha256 = Annotated[
    str,
    BeforeValidator(_require_canonical_text),
    Field(pattern=r"^[0-9a-f]{64}$"),
]
_GitCommitSha = Annotated[
    str,
    BeforeValidator(_require_canonical_text),
    Field(pattern=r"^[0-9a-f]{40}$"),
]
_SessionDate = Annotated[date, BeforeValidator(_require_session_date)]
_AwareDateTime = Annotated[
    datetime, BeforeValidator(_require_aware_datetime)
]
_StrictDecimal = Annotated[Decimal, BeforeValidator(_require_decimal)]
_PositiveDecimal = Annotated[
    Decimal,
    BeforeValidator(_require_decimal),
    Field(gt=Decimal("0")),
]
_NonNegativeDecimal = Annotated[
    Decimal,
    BeforeValidator(_require_decimal),
    Field(ge=Decimal("0")),
]
_StrictInt = Annotated[int, Field(strict=True)]
_PositiveQuantity = Annotated[int, Field(gt=0, strict=True)]
_NonNegativeCount = Annotated[int, Field(ge=0, strict=True)]
_PositiveRank = Annotated[int, Field(gt=0, strict=True)]
_FiniteFloat = Annotated[float, BeforeValidator(_require_finite_float)]
_PositiveFiniteFloat = Annotated[
    float,
    BeforeValidator(_require_finite_float),
    Field(gt=0.0),
]
_RankingPolicyRef = Annotated[
    RankingPolicyRef, BeforeValidator(_require_ranking_policy_ref)
]
_ExecutionCostPolicyRef = Annotated[
    ExecutionCostPolicyRef,
    BeforeValidator(_require_execution_cost_policy_ref),
]
_PortfolioState = Annotated[
    PortfolioState, BeforeValidator(_require_portfolio_state)
]
_HistoricalDecisionInterval = Annotated[
    HistoricalDecisionInterval,
    BeforeValidator(_require_decision_interval),
]


class PolicyArtifactRef(_ImmutableBacktestResultModel):
    """Generic reference used only when no authoritative policy ref exists."""

    policy_id: _CanonicalText
    policy_version: _CanonicalText
    policy_fingerprint: _Sha256


HistoricalBacktestValuationPolicy = PortfolioValuationPolicy


class HistoricalBacktestValuationPolicyRef(_ImmutableBacktestResultModel):
    policy_id: Literal[
        "completed_session_unadjusted_close_mark_v0.1"
    ] = HISTORICAL_BACKTEST_VALUATION_POLICY_ID
    schema_version: Literal[
        "historical_backtest_valuation_policy.v0.1"
    ] = HISTORICAL_BACKTEST_VALUATION_POLICY_SCHEMA_VERSION
    policy_fingerprint: _Sha256


def build_valuation_policy_ref(
    policy: HistoricalBacktestValuationPolicy,
) -> HistoricalBacktestValuationPolicyRef:
    if type(policy) is not HistoricalBacktestValuationPolicy:
        raise TypeError("policy must be a HistoricalBacktestValuationPolicy")
    rebuilt = HistoricalBacktestValuationPolicy.model_validate(
        policy.model_dump(mode="python")
    )
    return HistoricalBacktestValuationPolicyRef(
        policy_id=rebuilt.policy_id,
        schema_version=rebuilt.schema_version,
        policy_fingerprint=semantic_domain_sha256(
            VALUATION_POLICY_HASH_DOMAIN, rebuilt
        ),
    )


HistoricalBacktestValuationMark = PortfolioValuationMark


class HistoricalBacktestValuationSnapshot(_ImmutableBacktestResultModel):
    session: _SessionDate
    marks: Annotated[
        tuple[HistoricalBacktestValuationMark, ...],
        BeforeValidator(_require_tuple),
    ]
    snapshot_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        security_ids = tuple(mark.security_id for mark in self.marks)
        if any(mark.session != self.session for mark in self.marks):
            raise ValueError("every valuation mark must belong to snapshot session")
        if security_ids != tuple(sorted(security_ids)):
            raise ValueError("valuation marks must be ordered by security_id ascending")
        if len(set(security_ids)) != len(security_ids):
            raise ValueError("valuation mark security_ids must be unique")
        expected = semantic_domain_sha256(
            VALUATION_SNAPSHOT_HASH_DOMAIN,
            {"session": self.session, "marks": self.marks},
        )
        if self.snapshot_fingerprint != expected:
            raise ValueError("snapshot_fingerprint does not match valuation content")
        return self


def build_valuation_snapshot(
    *,
    session: date,
    marks: Iterable[HistoricalBacktestValuationMark],
) -> HistoricalBacktestValuationSnapshot:
    if type(session) is not date:
        raise ValueError("session must be a genuine date")
    if isinstance(marks, (str, bytes, dict, set, frozenset)):
        raise TypeError("marks must be a finite iterable of valuation marks")
    supplied = tuple(marks)
    if any(type(mark) is not HistoricalBacktestValuationMark for mark in supplied):
        raise TypeError("marks must contain HistoricalBacktestValuationMark values")
    ordered = tuple(sorted(supplied, key=lambda mark: mark.security_id))
    fingerprint = semantic_domain_sha256(
        VALUATION_SNAPSHOT_HASH_DOMAIN,
        {"session": session, "marks": ordered},
    )
    return HistoricalBacktestValuationSnapshot(
        session=session,
        marks=ordered,
        snapshot_fingerprint=fingerprint,
    )


class HistoricalBacktestRunManifest(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_run_manifest.v0.1"
    ] = HISTORICAL_BACKTEST_RUN_MANIFEST_SCHEMA_VERSION
    software_revision: _GitCommitSha
    software_revision_kind: Literal["git_commit"] = "git_commit"
    strategy_configuration_ref: ArtifactRef
    allocation_policy_ref: PolicyArtifactRef
    ranking_policy_ref: _RankingPolicyRef
    execution_cost_policy_ref: _ExecutionCostPolicyRef
    valuation_policy_ref: HistoricalBacktestValuationPolicyRef
    universe_artifact_ref: ArtifactRef
    market_data_artifact_ref: ArtifactRef
    corporate_action_artifact_ref: ArtifactRef | None = None
    earnings_artifact_ref: ArtifactRef | None = None
    earnings_provider_name: _OptionalCanonicalText = None
    run_configuration_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if (self.earnings_artifact_ref is None) != (
            self.earnings_provider_name is None
        ):
            raise ValueError(
                "earnings_artifact_ref and earnings_provider_name must both be present or absent"
            )
        expected = compute_run_configuration_fingerprint(
            schema_version=self.schema_version,
            software_revision=self.software_revision,
            software_revision_kind=self.software_revision_kind,
            strategy_configuration_ref=self.strategy_configuration_ref,
            allocation_policy_ref=self.allocation_policy_ref,
            ranking_policy_ref=self.ranking_policy_ref,
            execution_cost_policy_ref=self.execution_cost_policy_ref,
            valuation_policy_ref=self.valuation_policy_ref,
            universe_artifact_ref=self.universe_artifact_ref,
            market_data_artifact_ref=self.market_data_artifact_ref,
            corporate_action_artifact_ref=self.corporate_action_artifact_ref,
            earnings_artifact_ref=self.earnings_artifact_ref,
            earnings_provider_name=self.earnings_provider_name,
        )
        if self.run_configuration_fingerprint != expected:
            raise ValueError(
                "run_configuration_fingerprint does not match manifest content"
            )
        return self


def compute_run_configuration_fingerprint(
    *,
    schema_version: str,
    software_revision: str,
    software_revision_kind: str,
    strategy_configuration_ref: ArtifactRef,
    allocation_policy_ref: PolicyArtifactRef,
    ranking_policy_ref: RankingPolicyRef,
    execution_cost_policy_ref: ExecutionCostPolicyRef,
    valuation_policy_ref: HistoricalBacktestValuationPolicyRef,
    universe_artifact_ref: ArtifactRef,
    market_data_artifact_ref: ArtifactRef,
    corporate_action_artifact_ref: ArtifactRef | None,
    earnings_artifact_ref: ArtifactRef | None,
    earnings_provider_name: str | None,
) -> str:
    return semantic_domain_sha256(
        RUN_CONFIGURATION_HASH_DOMAIN,
        {
            "schema_version": schema_version,
            "software_revision": software_revision,
            "software_revision_kind": software_revision_kind,
            "strategy_configuration_ref": strategy_configuration_ref,
            "allocation_policy_ref": allocation_policy_ref,
            "ranking_policy_ref": ranking_policy_ref,
            "execution_cost_policy_ref": execution_cost_policy_ref,
            "valuation_policy_ref": valuation_policy_ref,
            "universe_artifact_ref": universe_artifact_ref,
            "market_data_artifact_ref": market_data_artifact_ref,
            "corporate_action_artifact_ref": corporate_action_artifact_ref,
            "earnings_artifact_ref": earnings_artifact_ref,
            "earnings_provider_name": earnings_provider_name,
        },
    )


def build_run_manifest(
    *,
    software_revision: str,
    strategy_configuration_ref: ArtifactRef,
    allocation_policy_ref: PolicyArtifactRef,
    ranking_policy_ref: RankingPolicyRef,
    execution_cost_policy_ref: ExecutionCostPolicyRef,
    valuation_policy_ref: HistoricalBacktestValuationPolicyRef,
    universe_artifact_ref: ArtifactRef,
    market_data_artifact_ref: ArtifactRef,
    corporate_action_artifact_ref: ArtifactRef | None = None,
    earnings_artifact_ref: ArtifactRef | None = None,
    earnings_provider_name: str | None = None,
) -> HistoricalBacktestRunManifest:
    values = {
        "schema_version": HISTORICAL_BACKTEST_RUN_MANIFEST_SCHEMA_VERSION,
        "software_revision": software_revision,
        "software_revision_kind": "git_commit",
        "strategy_configuration_ref": strategy_configuration_ref,
        "allocation_policy_ref": allocation_policy_ref,
        "ranking_policy_ref": ranking_policy_ref,
        "execution_cost_policy_ref": execution_cost_policy_ref,
        "valuation_policy_ref": valuation_policy_ref,
        "universe_artifact_ref": universe_artifact_ref,
        "market_data_artifact_ref": market_data_artifact_ref,
        "corporate_action_artifact_ref": corporate_action_artifact_ref,
        "earnings_artifact_ref": earnings_artifact_ref,
        "earnings_provider_name": earnings_provider_name,
    }
    return HistoricalBacktestRunManifest(
        **values,
        run_configuration_fingerprint=compute_run_configuration_fingerprint(
            **values
        ),
    )


class ExecutionApplicationStatus(StrEnum):
    APPLIED = "APPLIED"
    REPLAYED = "REPLAYED"


class ExecutionProvenanceSource(StrEnum):
    GENERATED_ENTRY = "GENERATED_ENTRY"
    GENERATED_EXIT = "GENERATED_EXIT"
    EXTERNAL_SCHEDULED = "EXTERNAL_SCHEDULED"


class HistoricalTradeStatus(StrEnum):
    CLOSED = "CLOSED"
    OPEN_AT_RUN_END = "OPEN_AT_RUN_END"


class DividendAttributionCompleteness(StrEnum):
    """The complete frozen OD-16.5 dividend-attribution completeness set."""

    COMPLETE_TRADE_LIFETIME = "COMPLETE_TRADE_LIFETIME"
    PARTIAL_PRE_RUN_UNKNOWN = "PARTIAL_PRE_RUN_UNKNOWN"


class HistoricalRejectionStage(StrEnum):
    ALLOCATION = "ALLOCATION"
    ENTRY_EXECUTION = "ENTRY_EXECUTION"


class HistoricalSettlementStatus(StrEnum):
    PENDING_AT_RUN_END = "PENDING_AT_RUN_END"
    SETTLED_DURING_RUN = "SETTLED_DURING_RUN"
    CARRIED_IN_AND_SETTLED = "CARRIED_IN_AND_SETTLED"


class HistoricalExitReason(StrEnum):
    GAP_THROUGH_STOP = "GAP_THROUGH_STOP"
    GAP_THROUGH_TARGET = "GAP_THROUGH_TARGET"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    EARNINGS_FORCED_EXIT = "EARNINGS_FORCED_EXIT"
    MAX_HOLDING = "MAX_HOLDING"
    EXTERNAL_SCHEDULED = "EXTERNAL_SCHEDULED"


class HistoricalBacktestTransitionAudit(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_transition_audit.v0.1"
    ] = "historical_backtest_transition_audit.v0.1"
    session: _SessionDate
    decision_time: _AwareDateTime
    state_hash_before: _Sha256
    state_hash_after: _Sha256
    state_version_before: _NonNegativeCount
    state_version_after: _NonNegativeCount
    settled_cash_before: _NonNegativeDecimal
    settled_cash_after: _NonNegativeDecimal
    open_position_count_before: _NonNegativeCount
    open_position_count_after: _NonNegativeCount
    pending_settlement_count_before: _NonNegativeCount
    pending_settlement_count_after: _NonNegativeCount
    newly_applied_event_ids: Annotated[
        tuple[_CanonicalText, ...], BeforeValidator(_require_tuple)
    ] = ()
    replayed_event_ids: Annotated[
        tuple[_CanonicalText, ...], BeforeValidator(_require_tuple)
    ] = ()
    ledger_entry_count: _NonNegativeCount


class HistoricalBacktestExecutionProvenance(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_execution_provenance.v0.1"
    ] = "historical_backtest_execution_provenance.v0.1"
    session: _SessionDate
    event_order: _NonNegativeCount
    execution_id: _CanonicalText
    source_order_id: _CanonicalText
    security_id: _CanonicalSecurityId
    side: ExecutionSide
    quantity: _PositiveQuantity
    fill_price: _PositiveDecimal
    execution_cost: _NonNegativeDecimal
    settlement_id: _OptionalCanonicalText = None
    settlement_session: _SessionDate | None = None
    application_status: ExecutionApplicationStatus
    provenance_source: ExecutionProvenanceSource
    source_payload_fingerprint: _Sha256
    entry_decision_fingerprint: _Sha256 | None = None
    exit_decision_fingerprint: _Sha256 | None = None

    @model_validator(mode="after")
    def validate_settlement_pair(self) -> Self:
        if (self.settlement_id is None) != (self.settlement_session is None):
            raise ValueError(
                "settlement_id and settlement_session must both be present or absent"
            )
        return self


class HistoricalBacktestEntryRecord(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_entry_record.v0.1"
    ] = "historical_backtest_entry_record.v0.1"
    entry_execution_id: _CanonicalText
    source_order_id: _CanonicalText
    security_id: _CanonicalSecurityId
    quantity: _PositiveQuantity
    entry_session: _SessionDate
    fill_price: _PositiveDecimal
    execution_cost: _NonNegativeDecimal
    cost_basis: _PositiveDecimal
    provenance_source: ExecutionProvenanceSource
    signal_session: _SessionDate | None = None
    signal_time: _AwareDateTime | None = None
    allocation_session: _SessionDate | None = None
    source_rank: _PositiveRank | None = None
    ranking_snapshot_fingerprint: _Sha256 | None = None
    ranking_input_fingerprint: _Sha256 | None = None
    entry_execution_status: EntryExecutionStatus | None = None
    execution_cost_policy_fingerprint: _Sha256 | None = None

    @model_validator(mode="after")
    def validate_cost_basis(self) -> Self:
        # Exact-arithmetic hardening (Residual C): mirrors project_entries's
        # exact reconciliation so this validator can never diverge from the
        # derivation site at any ambient Decimal precision.
        expected = add_exact_decimal(
            exact_decimal_times_int(self.fill_price, self.quantity),
            self.execution_cost,
        )
        if self.cost_basis != expected:
            raise ValueError(
                "cost_basis must equal quantity * fill_price + execution_cost"
            )
        generated_fields = (
            self.signal_session,
            self.signal_time,
            self.allocation_session,
            self.source_rank,
            self.ranking_snapshot_fingerprint,
            self.ranking_input_fingerprint,
        )
        if self.provenance_source is ExecutionProvenanceSource.GENERATED_ENTRY:
            if any(value is None for value in generated_fields):
                raise ValueError(
                    "generated entries require complete signal, allocation, and ranking provenance"
                )
            if self.entry_execution_status is not EntryExecutionStatus.EXECUTED:
                raise ValueError(
                    "generated entries require entry_execution_status=EXECUTED"
                )
            if self.execution_cost_policy_fingerprint is None:
                raise ValueError(
                    "generated entries require execution-cost policy provenance"
                )
        elif self.provenance_source is ExecutionProvenanceSource.EXTERNAL_SCHEDULED:
            external_only_absence = generated_fields + (
                self.entry_execution_status,
                self.execution_cost_policy_fingerprint,
            )
            if any(value is not None for value in external_only_absence):
                raise ValueError(
                    "external scheduled entries must not claim generated-entry provenance"
                )
        elif self.provenance_source is ExecutionProvenanceSource.GENERATED_EXIT:
            raise ValueError("an entry record cannot use GENERATED_EXIT provenance")
        return self


class HistoricalBacktestExitRecord(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_exit_record.v0.1"
    ] = "historical_backtest_exit_record.v0.1"
    exit_execution_id: _CanonicalText
    source_order_id: _CanonicalText
    entry_execution_id: _CanonicalText
    security_id: _CanonicalSecurityId
    quantity: _PositiveQuantity
    exit_session: _SessionDate
    fill_price: _PositiveDecimal
    execution_cost: _NonNegativeDecimal
    gross_proceeds: _PositiveDecimal
    net_proceeds: _StrictDecimal
    settlement_id: _CanonicalText
    settlement_session: _SessionDate
    provenance_source: ExecutionProvenanceSource
    exit_reason: HistoricalExitReason
    reference_exit_price: _PositiveDecimal | None = None

    @model_validator(mode="after")
    def validate_proceeds(self) -> Self:
        # Exact-arithmetic hardening (Residual C): mirrors project_exits's
        # exact computation so this validator can never diverge from the
        # derivation site at any ambient Decimal precision.
        expected_gross = exact_decimal_times_int(self.fill_price, self.quantity)
        if self.gross_proceeds != expected_gross:
            raise ValueError("gross_proceeds must equal quantity * fill_price")
        if self.net_proceeds != subtract_exact_decimal(
            self.gross_proceeds, self.execution_cost
        ):
            raise ValueError(
                "net_proceeds must equal gross_proceeds - execution_cost"
            )
        if self.settlement_session <= self.exit_session:
            raise ValueError("settlement_session must be after exit_session")
        if self.provenance_source is ExecutionProvenanceSource.GENERATED_EXIT:
            if self.reference_exit_price is None:
                raise ValueError(
                    "generated exits require reference_exit_price provenance"
                )
            if self.exit_reason is HistoricalExitReason.EXTERNAL_SCHEDULED:
                raise ValueError(
                    "generated exits require a Phase 15B exit reason"
                )
        elif self.provenance_source is ExecutionProvenanceSource.EXTERNAL_SCHEDULED:
            if self.exit_reason is not HistoricalExitReason.EXTERNAL_SCHEDULED:
                raise ValueError(
                    "external scheduled exits require EXTERNAL_SCHEDULED reason"
                )
            if self.reference_exit_price is not None:
                raise ValueError(
                    "external scheduled exits must not claim reference-price provenance"
                )
        elif self.provenance_source is ExecutionProvenanceSource.GENERATED_ENTRY:
            raise ValueError("an exit record cannot use GENERATED_ENTRY provenance")
        return self


class HistoricalClosedTradeRecord(_ImmutableBacktestResultModel):
    # v0.1 (legacy / non-dividend-aware source run) carries none of the
    # OD-16 dividend-attribution fields and no `trade_total_pnl`: a run
    # with `dividend_run_evidence is None` never underwent OD-15.7/OD-15.8
    # processed-session-contiguity validation (backtester/validation.py
    # skips it entirely for `not dividend_aware`), so OD-16.5's second
    # COMPLETE_TRADE_LIFETIME precondition can never be satisfied for it,
    # and absence of dividend ledger rows is not affirmative coverage
    # proving zero applicable dividends (OD-7.1). v0.2 (Slice 10,
    # OD-16.4/OD-22.2) is for a dividend-aware source run and gains the
    # four authoritative dividend-attribution fields grouped from the
    # Slice-9 projected cash ledger, but never carries `trade_total_pnl`
    # (OD-22.1: no backfill onto an already-published v0.2 record; rerun
    # under v0.3 to publish it). v0.3 (Slice 11, OD-16.4/OD-16.5a/OD-21.7)
    # adds the frozen authoritative `trade_total_pnl`: exactly
    # `realized_pnl + ordinary_dividend_income` for
    # `COMPLETE_TRADE_LIFETIME`, and `None` for `PARTIAL_PRE_RUN_UNKNOWN`
    # (the pre-run dividend component is unknown, so no lifetime total may
    # be published).
    schema_version: Literal[
        "historical_closed_trade.v0.1",
        "historical_closed_trade.v0.2",
        "historical_closed_trade.v0.3",
    ] = "historical_closed_trade.v0.3"
    status: Literal[HistoricalTradeStatus.CLOSED] = HistoricalTradeStatus.CLOSED
    trade_id: _CanonicalText
    security_id: _CanonicalSecurityId
    quantity: _PositiveQuantity
    carried_in: StrictBool
    entry_execution_id: _CanonicalText
    entry_session: _SessionDate
    entry_fill_price: _PositiveDecimal
    entry_execution_cost: _NonNegativeDecimal
    entry_cost_basis: _PositiveDecimal
    exit_execution_id: _CanonicalText
    exit_session: _SessionDate
    exit_fill_price: _PositiveDecimal
    exit_execution_cost: _NonNegativeDecimal
    exit_reason: HistoricalExitReason
    gross_exit_proceeds: _PositiveDecimal
    net_exit_proceeds: _StrictDecimal
    realized_pnl: _StrictDecimal
    ordinary_dividend_income: _NonNegativeDecimal | None = None
    ordinary_dividend_event_count: _NonNegativeCount | None = None
    dividend_attribution_completeness: DividendAttributionCompleteness | None = None
    ordinary_dividend_attribution_fingerprint: _Sha256 | None = None
    trade_total_pnl: _StrictDecimal | None = None

    @model_validator(mode="after")
    def validate_trade_identity(self) -> Self:
        # Exact-arithmetic hardening (Residual C): entry_fill_price/
        # exit_fill_price carry no enforced significant-digit bound, so
        # every multiplication/addition/subtraction in this identity
        # chain uses the shared exact helpers, not ambient-context
        # Decimal `*`/`+`/`-`.
        if self.trade_id != self.entry_execution_id:
            raise ValueError("trade_id must equal entry_execution_id")
        expected_entry = add_exact_decimal(
            exact_decimal_times_int(self.entry_fill_price, self.quantity),
            self.entry_execution_cost,
        )
        expected_gross = exact_decimal_times_int(self.exit_fill_price, self.quantity)
        if self.entry_cost_basis != expected_entry:
            raise ValueError("entry_cost_basis has an invalid reporting identity")
        if self.gross_exit_proceeds != expected_gross:
            raise ValueError("gross_exit_proceeds has an invalid reporting identity")
        if self.net_exit_proceeds != subtract_exact_decimal(
            expected_gross, self.exit_execution_cost
        ):
            raise ValueError("net_exit_proceeds has an invalid reporting identity")
        if self.realized_pnl != subtract_exact_decimal(
            self.net_exit_proceeds, self.entry_cost_basis
        ):
            raise ValueError(
                "realized_pnl must equal net_exit_proceeds - entry_cost_basis"
            )
        return self

    @model_validator(mode="after")
    def validate_dividend_schema_version(self) -> Self:
        dividend_fields_present = (
            self.ordinary_dividend_income is not None,
            self.ordinary_dividend_event_count is not None,
            self.dividend_attribution_completeness is not None,
            self.ordinary_dividend_attribution_fingerprint is not None,
        )
        if self.schema_version == "historical_closed_trade.v0.1":
            if any(dividend_fields_present) or self.trade_total_pnl is not None:
                raise ValueError(
                    "a legacy (v0.1) closed trade from a non-dividend-aware "
                    "source run must not carry OD-16 dividend-attribution "
                    "fields or trade_total_pnl"
                )
            return self
        if not all(dividend_fields_present):
            raise ValueError(
                "a dividend-aware closed trade requires all four OD-16 "
                "dividend-attribution fields"
            )
        if (
            self.schema_version == "historical_closed_trade.v0.2"
            and self.trade_total_pnl is not None
        ):
            raise ValueError(
                "a v0.2 closed trade must not carry trade_total_pnl "
                "(OD-22.1: no backfill onto an already-published record; "
                "rerun under v0.3 to publish an authoritative total P&L)"
            )
        return self

    @model_validator(mode="after")
    def validate_dividend_attribution(self) -> Self:
        if self.dividend_attribution_completeness is None:
            # Legacy (v0.1) trade: no OD-16 dividend economics are
            # published, so there is nothing to cross-check here.
            return self
        # OD-16.6: for every closed trade, a positive-only sum of
        # attributed rows (Slice-9 guarantees each DIVIDEND_APPLIED row's
        # settled_cash_delta > 0) is zero iff no rows were attributed.
        if (self.ordinary_dividend_event_count == 0) != (
            self.ordinary_dividend_income == Decimal("0")
        ):
            raise ValueError(
                "ordinary_dividend_event_count == 0 iff "
                "ordinary_dividend_income == 0"
            )
        if (
            self.dividend_attribution_completeness
            is DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
            and self.carried_in
        ):
            raise ValueError(
                "a carried_in trade cannot be COMPLETE_TRADE_LIFETIME"
            )
        if (
            self.dividend_attribution_completeness
            is DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
            and not self.carried_in
        ):
            raise ValueError(
                "a non-carried_in trade cannot be PARTIAL_PRE_RUN_UNKNOWN"
            )
        return self

    @model_validator(mode="after")
    def validate_trade_total_pnl(self) -> Self:
        # OD-16.5a/OD-21.7: only a v0.3 record ever publishes an
        # authoritative `trade_total_pnl`; v0.1/v0.2 already forced it to
        # None above. The COMPLETE identity check must itself be exact and
        # ambient-Decimal-context-independent (OD-6.7's "no free ambient-
        # context precision parameter" applies here too), so this reuses
        # `add_exact_decimal` rather than Python's `+`.
        if self.schema_version != "historical_closed_trade.v0.3":
            return self
        if (
            self.dividend_attribution_completeness
            is DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
        ):
            if self.trade_total_pnl is None:
                raise ValueError(
                    "a COMPLETE_TRADE_LIFETIME v0.3 closed trade requires "
                    "trade_total_pnl"
                )
            assert self.ordinary_dividend_income is not None
            expected = add_exact_decimal(
                self.realized_pnl, self.ordinary_dividend_income
            )
            if self.trade_total_pnl != expected:
                raise ValueError(
                    "trade_total_pnl must equal realized_pnl + "
                    "ordinary_dividend_income exactly"
                )
        elif self.trade_total_pnl is not None:
            raise ValueError(
                "a PARTIAL_PRE_RUN_UNKNOWN closed trade must not publish "
                "trade_total_pnl; the pre-run dividend component is unknown"
            )
        return self


class HistoricalOpenTradeRecord(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_open_trade.v0.1"
    ] = "historical_open_trade.v0.1"
    status: Literal[
        HistoricalTradeStatus.OPEN_AT_RUN_END
    ] = HistoricalTradeStatus.OPEN_AT_RUN_END
    trade_id: _CanonicalText
    security_id: _CanonicalSecurityId
    quantity: _PositiveQuantity
    carried_in: StrictBool
    entry_execution_id: _CanonicalText
    entry_session: _SessionDate
    entry_fill_price: _PositiveDecimal
    entry_execution_cost: _NonNegativeDecimal
    entry_cost_basis: _PositiveDecimal
    final_mark_session: _SessionDate
    final_mark_price: _PositiveDecimal
    final_market_value: _NonNegativeDecimal
    unrealized_pnl: _StrictDecimal

    @model_validator(mode="after")
    def validate_trade_identity(self) -> Self:
        # Exact-arithmetic hardening (Residual A): entry_fill_price/
        # final_mark_price carry no enforced significant-digit bound, so
        # every multiplication/addition/subtraction in this identity
        # chain uses the shared exact helpers, not ambient-context
        # Decimal `*`/`+`/`-`.
        if self.trade_id != self.entry_execution_id:
            raise ValueError("trade_id must equal entry_execution_id")
        expected_entry = add_exact_decimal(
            exact_decimal_times_int(self.entry_fill_price, self.quantity),
            self.entry_execution_cost,
        )
        expected_value = exact_decimal_times_int(self.final_mark_price, self.quantity)
        if self.entry_cost_basis != expected_entry:
            raise ValueError("entry_cost_basis has an invalid reporting identity")
        if self.final_market_value != expected_value:
            raise ValueError(
                "final_market_value must equal quantity * final_mark_price"
            )
        if self.unrealized_pnl != subtract_exact_decimal(
            self.final_market_value, self.entry_cost_basis
        ):
            raise ValueError(
                "unrealized_pnl must equal final_market_value - entry_cost_basis"
            )
        return self


HistoricalTradeRecord: TypeAlias = Annotated[
    HistoricalClosedTradeRecord | HistoricalOpenTradeRecord,
    Field(discriminator="status"),
]


class HistoricalBacktestRejectionRecord(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_rejection.v0.1"
    ] = "historical_backtest_rejection.v0.1"
    decision_session: _SessionDate
    security_id: _CanonicalSecurityId
    stage: HistoricalRejectionStage
    reason: _CanonicalText
    requested_quantity: _PositiveQuantity | None = None
    reserved_cash: _NonNegativeDecimal | None = None
    released_cash: _NonNegativeDecimal | None = None
    source_rank: _PositiveRank | None = None
    source_artifact_fingerprint: _Sha256 | None = None


class HistoricalSignalProvenance(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_signal_provenance.v0.1"
    ] = "historical_signal_provenance.v0.1"
    session: _SessionDate
    decision_time: _AwareDateTime
    security_id: _CanonicalSecurityId
    action: BaselineSignalAction
    planned_entry_session: _SessionDate
    adjusted_close: _PositiveFiniteFloat
    sma20: _FiniteFloat | None = None
    sma50: _FiniteFloat | None = None
    rsi14: _FiniteFloat | None = None
    atr14: _FiniteFloat | None = None
    atr_fraction: _FiniteFloat | None = None
    universe_eligible: StrictBool
    close_above_sma50: StrictBool
    sma20_above_sma50: StrictBool
    rsi_above_50: StrictBool
    atr_above_minimum: StrictBool
    earnings_entry_allowed: StrictBool
    earnings_action: EarningsIntegrationAction
    source_payload_fingerprint: _Sha256


class HistoricalRankingCycle(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_ranking_cycle.v0.1"
    ] = "historical_ranking_cycle.v0.1"
    ranking_session: _SessionDate
    decision_time: _AwareDateTime
    policy_id: _CanonicalText
    policy_version: _CanonicalText
    policy_fingerprint: _Sha256
    candidate_count: _NonNegativeCount
    input_set_fingerprint: _Sha256
    snapshot_fingerprint: _Sha256


class HistoricalRankingCandidateProvenance(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_ranking_candidate.v0.1"
    ] = "historical_ranking_candidate.v0.1"
    ranking_session: _SessionDate
    decision_time: _AwareDateTime
    security_id: _CanonicalSecurityId
    rank: _PositiveRank
    trend_separation_atr: _FiniteFloat
    rsi14: _FiniteFloat
    sma20: _FiniteFloat
    sma50: _FiniteFloat
    atr14: _FiniteFloat
    candidate_input_fingerprint: _Sha256
    ranking_snapshot_fingerprint: _Sha256


class HistoricalAllocationCycle(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_allocation_cycle.v0.1"
    ] = "historical_allocation_cycle.v0.1"
    allocation_session: _SessionDate
    decision_time: _AwareDateTime
    ranking_snapshot_fingerprint: _Sha256
    allocation_policy_fingerprint: _Sha256
    candidate_count: _NonNegativeCount
    admitted_count: _NonNegativeCount
    rejected_count: _NonNegativeCount
    settled_cash_input: _NonNegativeDecimal
    portfolio_equity_input: _NonNegativeDecimal
    open_position_count_input: _NonNegativeCount

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.candidate_count != self.admitted_count + self.rejected_count:
            raise ValueError(
                "candidate_count must equal admitted_count + rejected_count"
            )
        return self


class HistoricalAllocationCandidateProvenance(
    _ImmutableBacktestResultModel
):
    schema_version: Literal[
        "historical_allocation_candidate.v0.1"
    ] = "historical_allocation_candidate.v0.1"
    allocation_session: _SessionDate
    security_id: _CanonicalSecurityId
    source_rank: _PositiveRank
    processing_rank: _PositiveRank
    ranking_snapshot_fingerprint: _Sha256
    ranking_input_fingerprint: _Sha256
    action: PortfolioCandidateAction
    candidate_cash_limit: _NonNegativeDecimal | None = None
    reserved_cash: _NonNegativeDecimal
    fixed_shares: _PositiveQuantity | None = None
    used_slots_before: _NonNegativeCount
    used_slots_after: _NonNegativeCount
    unreserved_cash_before: _NonNegativeDecimal
    unreserved_cash_after: _NonNegativeDecimal


class HistoricalCashLedgerRow(_ImmutableBacktestResultModel):
    # Bumped v0.1 -> v0.2 (OD-20.5/OD-22.2): gained `attribution_trade_id`,
    # the authoritative Phase 13 dividend-attribution key preserved 1:1 for
    # `DIVIDEND_APPLIED` rows so a later slice can group them by closed
    # trade (OD-16.2) without re-deriving entitlement. A row persisted
    # under v0.1 never carried this fact, so the two versions must not
    # share one schema identity.
    schema_version: Literal[
        "historical_cash_ledger_row.v0.2"
    ] = "historical_cash_ledger_row.v0.2"
    session: _SessionDate
    sequence_in_session: _NonNegativeCount
    ledger_event_type: PortfolioLedgerEventType
    source_event_id: _CanonicalText
    source_order_id: _OptionalCanonicalText = None
    security_id: _CanonicalSecurityId | None = None
    settled_cash_delta: _StrictDecimal
    pending_cash_delta: _StrictDecimal
    settled_cash_after: _NonNegativeDecimal
    settlement_id: _OptionalCanonicalText = None
    settlement_session: _SessionDate | None = None
    state_hash_before: _Sha256
    state_hash_after: _Sha256
    source_payload_fingerprint: _Sha256
    attribution_trade_id: _OptionalCanonicalText = None

    @model_validator(mode="after")
    def validate_settlement_pair(self) -> Self:
        if (self.settlement_id is None) != (self.settlement_session is None):
            raise ValueError(
                "settlement_id and settlement_session must both be present or absent"
            )
        return self

    @model_validator(mode="after")
    def validate_dividend_attribution(self) -> Self:
        if self.ledger_event_type is PortfolioLedgerEventType.DIVIDEND_APPLIED:
            if self.attribution_trade_id is None:
                raise ValueError(
                    "a DIVIDEND_APPLIED row requires attribution_trade_id"
                )
            if self.settlement_id is not None or self.settlement_session is not None:
                raise ValueError(
                    "a DIVIDEND_APPLIED row must not carry settlement identity"
                )
            if self.pending_cash_delta != Decimal("0"):
                raise ValueError(
                    "a DIVIDEND_APPLIED row must not affect pending cash"
                )
            if self.settled_cash_delta <= Decimal("0"):
                raise ValueError(
                    "a DIVIDEND_APPLIED row must carry a positive settled_cash_delta"
                )
        elif self.attribution_trade_id is not None:
            raise ValueError(
                "attribution_trade_id is defined only for DIVIDEND_APPLIED rows"
            )
        return self


class HistoricalSettlementRecord(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_settlement_record.v0.1"
    ] = "historical_settlement_record.v0.1"
    settlement_id: _CanonicalText
    source_sell_execution_id: _CanonicalText
    security_id: _CanonicalSecurityId
    trade_session: _SessionDate
    settlement_session: _SessionDate
    amount: _PositiveDecimal
    status: HistoricalSettlementStatus

    @model_validator(mode="after")
    def validate_chronology(self) -> Self:
        if self.settlement_session <= self.trade_session:
            raise ValueError("settlement_session must be after trade_session")
        return self


def _validate_pnl_dividend_schema_version(
    *,
    schema_version: str,
    legacy_version: str,
    income_this_session: Decimal | None,
    cumulative_income: Decimal | None,
) -> None:
    """Shared OD-20.6 presence-iff-dividend-aware-version rule.

    `HistoricalBacktestSessionPnl` and `HistoricalBacktestEquityRow` both
    gain the same two portfolio-scope dividend fields under the same rule
    as the legacy/dividend-aware closed-trade split (Slice 10/11): a
    non-dividend-aware (legacy) row must carry neither field (a
    non-dividend-aware run never modeled dividend cash-flows at all, so a
    fabricated zero would misrepresent it as having done so and concluded
    zero -- OD-7.1's "absence of evidence is not evidence of absence"
    applies here just as it does at trade scope), while a dividend-aware
    row must carry both. When both are present, the cumulative figure can
    never be less than this session's own contribution, since it is a
    running sum of exclusively non-negative amounts (OD-21.12) including
    this session's own.
    """

    fields_present = (income_this_session is not None, cumulative_income is not None)
    if schema_version == legacy_version:
        if any(fields_present):
            raise ValueError(
                "a legacy (non-dividend-aware) row must not carry "
                "portfolio-scope ordinary-dividend fields"
            )
        return
    if not all(fields_present):
        raise ValueError(
            "a dividend-aware row requires both portfolio-scope "
            "ordinary-dividend fields"
        )
    assert income_this_session is not None and cumulative_income is not None
    if cumulative_income < income_this_session:
        raise ValueError(
            "cumulative_ordinary_dividend_income cannot be less than "
            "ordinary_dividend_income_this_session"
        )


class HistoricalBacktestSessionPnl(_ImmutableBacktestResultModel):
    # Bumped v0.1 -> v0.2 (Slice 12, OD-20.6/OD-22.2): gained the two
    # portfolio-scope ordinary-dividend fields. A legacy (non-dividend-
    # aware) row stays on v0.1 without them; see
    # _validate_pnl_dividend_schema_version.
    schema_version: Literal[
        "historical_backtest_session_pnl.v0.1",
        "historical_backtest_session_pnl.v0.2",
    ] = "historical_backtest_session_pnl.v0.1"
    session: _SessionDate
    realized_pnl_this_session: _StrictDecimal
    cumulative_realized_pnl: _StrictDecimal
    unrealized_pnl: _StrictDecimal
    execution_cost_this_session: _NonNegativeDecimal
    cumulative_execution_cost: _NonNegativeDecimal
    period_pnl: _StrictDecimal
    ordinary_dividend_income_this_session: _NonNegativeDecimal | None = None
    cumulative_ordinary_dividend_income: _NonNegativeDecimal | None = None

    @model_validator(mode="after")
    def validate_dividend_schema_version(self) -> Self:
        _validate_pnl_dividend_schema_version(
            schema_version=self.schema_version,
            legacy_version="historical_backtest_session_pnl.v0.1",
            income_this_session=self.ordinary_dividend_income_this_session,
            cumulative_income=self.cumulative_ordinary_dividend_income,
        )
        return self


class HistoricalBacktestEquityRow(_ImmutableBacktestResultModel):
    # Bumped v0.1 -> v0.2 (Slice 12, OD-20.6/OD-22.2): same two
    # portfolio-scope ordinary-dividend fields and the same legacy split
    # as HistoricalBacktestSessionPnl above.
    schema_version: Literal[
        "historical_backtest_equity_row.v0.1",
        "historical_backtest_equity_row.v0.2",
    ] = "historical_backtest_equity_row.v0.1"
    session: _SessionDate
    state_hash: _Sha256
    settled_cash: _NonNegativeDecimal
    pending_receivable_value: _NonNegativeDecimal
    open_position_market_value: _NonNegativeDecimal
    equity: _NonNegativeDecimal
    realized_pnl_this_session: _StrictDecimal
    cumulative_realized_pnl: _StrictDecimal
    unrealized_pnl: _StrictDecimal
    execution_cost_this_session: _NonNegativeDecimal
    cumulative_execution_cost: _NonNegativeDecimal
    period_pnl: _StrictDecimal
    ordinary_dividend_income_this_session: _NonNegativeDecimal | None = None
    cumulative_ordinary_dividend_income: _NonNegativeDecimal | None = None

    @model_validator(mode="after")
    def validate_equity(self) -> Self:
        # settled_cash can carry Gate3 scale-38 dividend-cash precision
        # (Slice 12, OD-6.7) once an in-run dividend has been applied, so
        # this identity check uses add_exact_decimal rather than naive
        # `+`, which could otherwise round under a low ambient Decimal
        # context and reject an exact, correct equity value.
        expected = add_exact_decimal(
            add_exact_decimal(self.settled_cash, self.pending_receivable_value),
            self.open_position_market_value,
        )
        if self.equity != expected:
            raise ValueError(
                "equity must equal cash + pending receivables + market value"
            )
        return self

    @model_validator(mode="after")
    def validate_dividend_schema_version(self) -> Self:
        _validate_pnl_dividend_schema_version(
            schema_version=self.schema_version,
            legacy_version="historical_backtest_equity_row.v0.1",
            income_this_session=self.ordinary_dividend_income_this_session,
            cumulative_income=self.cumulative_ordinary_dividend_income,
        )
        return self


class HistoricalBacktestCostSummary(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_cost_summary.v0.1"
    ] = "historical_backtest_cost_summary.v0.1"
    buy_execution_cost_total: _NonNegativeDecimal
    sell_execution_cost_total: _NonNegativeDecimal
    total_execution_cost: _NonNegativeDecimal
    applied_buy_count: _NonNegativeCount
    applied_sell_count: _NonNegativeCount

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if (
            self.total_execution_cost
            != self.buy_execution_cost_total + self.sell_execution_cost_total
        ):
            raise ValueError(
                "total_execution_cost must equal buy and sell cost totals"
            )
        return self


class HistoricalBacktestExitReasonRow(_ImmutableBacktestResultModel):
    reason: HistoricalExitReason
    exit_count: _NonNegativeCount
    realized_pnl: _StrictDecimal


_EXIT_REASON_ORDER = tuple(HistoricalExitReason)


class HistoricalBacktestExitReasonSummary(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_exit_reason_summary.v0.1"
    ] = "historical_backtest_exit_reason_summary.v0.1"
    rows: Annotated[
        tuple[HistoricalBacktestExitReasonRow, ...],
        BeforeValidator(_require_tuple),
    ]

    @model_validator(mode="after")
    def validate_complete_order(self) -> Self:
        if tuple(row.reason for row in self.rows) != _EXIT_REASON_ORDER:
            raise ValueError(
                "exit reason rows must contain every reason exactly once in frozen order"
            )
        return self


class HistoricalBacktestSummary(_ImmutableBacktestResultModel):
    processed_session_count: _NonNegativeCount
    entry_count: _NonNegativeCount
    exit_count: _NonNegativeCount
    closed_trade_count: _NonNegativeCount
    open_trade_count: _NonNegativeCount
    allocation_rejection_count: _NonNegativeCount
    entry_execution_rejection_count: _NonNegativeCount
    gross_realized_pnl: _StrictDecimal
    final_unrealized_pnl: _StrictDecimal
    initial_equity: _NonNegativeDecimal
    final_equity: _NonNegativeDecimal
    period_pnl: _StrictDecimal
    buy_execution_cost_total: _NonNegativeDecimal
    sell_execution_cost_total: _NonNegativeDecimal
    total_execution_cost: _NonNegativeDecimal
    # OD-20.7/OD-21.13 (Slice 12): None for a legacy (non-dividend-aware)
    # run -- it never modeled dividend cash-flows, so it cannot publish an
    # authoritative total, fabricated zero or otherwise. For a
    # dividend-aware run: zero for a zero-session run, otherwise the
    # final cumulative_ordinary_dividend_income.
    ordinary_dividend_income_total: _NonNegativeDecimal | None = None

    @model_validator(mode="after")
    def validate_local_identities(self) -> Self:
        if (
            self.total_execution_cost
            != self.buy_execution_cost_total + self.sell_execution_cost_total
        ):
            raise ValueError(
                "total_execution_cost must equal buy and sell cost totals"
            )
        if self.period_pnl != _subtract_exact(self.final_equity, self.initial_equity):
            raise ValueError("period_pnl must equal final_equity - initial_equity")
        if (
            self.ordinary_dividend_income_total is not None
            and self.processed_session_count == 0
            and self.ordinary_dividend_income_total != Decimal("0")
        ):
            raise ValueError(
                "ordinary_dividend_income_total must be exactly zero for a "
                "zero-session run"
            )
        return self


_AUDIT_FLAG_NAMES = (
    "source_run_canonical",
    "state_chain_valid",
    "ledger_reconstruction_valid",
    "execution_ledger_provenance_valid",
    "signal_provenance_valid",
    "ranking_provenance_valid",
    "allocation_provenance_valid",
    "execution_provenance_valid",
    "trade_linkage_valid",
    "valuation_coverage_valid",
    "equity_reconciliation_valid",
    "pnl_reconciliation_valid",
    "cost_reconciliation_valid",
    "policy_consistency_valid",
    "artifact_fingerprints_valid",
)


class HistoricalBacktestAuditSummary(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_audit_summary.v0.1"
    ] = "historical_backtest_audit_summary.v0.1"
    source_run_canonical: StrictBool
    state_chain_valid: StrictBool
    ledger_reconstruction_valid: StrictBool
    execution_ledger_provenance_valid: StrictBool
    signal_provenance_valid: StrictBool
    ranking_provenance_valid: StrictBool
    allocation_provenance_valid: StrictBool
    execution_provenance_valid: StrictBool
    trade_linkage_valid: StrictBool
    valuation_coverage_valid: StrictBool
    equity_reconciliation_valid: StrictBool
    pnl_reconciliation_valid: StrictBool
    cost_reconciliation_valid: StrictBool
    policy_consistency_valid: StrictBool
    artifact_fingerprints_valid: StrictBool
    audit_passed: StrictBool

    @model_validator(mode="after")
    def validate_passed(self) -> Self:
        if self.audit_passed and not all(
            getattr(self, field_name) for field_name in _AUDIT_FLAG_NAMES
        ):
            raise ValueError(
                "audit_passed may be true only when every audit flag is true"
            )
        return self


class HistoricalBacktestContentFingerprints(_ImmutableBacktestResultModel):
    session_transitions: _Sha256
    signal_provenance: _Sha256
    ranking_cycles: _Sha256
    ranking_candidates: _Sha256
    allocation_cycles: _Sha256
    allocation_candidates: _Sha256
    execution_provenance: _Sha256
    entries: _Sha256
    exits: _Sha256
    trades: _Sha256
    rejections: _Sha256
    cash_ledger: _Sha256
    settlement_ledger: _Sha256
    session_pnl: _Sha256
    equity_curve: _Sha256
    cost_summary: _Sha256
    exit_reason_summary: _Sha256
    summary: _Sha256
    audit_summary: _Sha256


class HistoricalBacktestAuditResult(_ImmutableBacktestResultModel):
    schema_version: Literal[
        "historical_backtest_audit_result.v0.2"
    ] = HISTORICAL_BACKTEST_AUDIT_RESULT_SCHEMA_VERSION
    run_manifest: HistoricalBacktestRunManifest
    source_run_fingerprint: _Sha256
    decision_interval: _HistoricalDecisionInterval
    initial_state: _PortfolioState
    final_state: _PortfolioState
    initial_state_fingerprint: _Sha256
    final_state_fingerprint: _Sha256
    initial_equity: _NonNegativeDecimal
    final_equity: _NonNegativeDecimal
    period_pnl: _StrictDecimal
    session_transitions: Annotated[
        tuple[HistoricalBacktestTransitionAudit, ...],
        BeforeValidator(_require_tuple),
    ] = ()
    signal_provenance: Annotated[
        tuple[HistoricalSignalProvenance, ...],
        BeforeValidator(_require_tuple),
    ] = ()
    ranking_cycles: Annotated[
        tuple[HistoricalRankingCycle, ...], BeforeValidator(_require_tuple)
    ] = ()
    ranking_candidates: Annotated[
        tuple[HistoricalRankingCandidateProvenance, ...],
        BeforeValidator(_require_tuple),
    ] = ()
    allocation_cycles: Annotated[
        tuple[HistoricalAllocationCycle, ...], BeforeValidator(_require_tuple)
    ] = ()
    allocation_candidates: Annotated[
        tuple[HistoricalAllocationCandidateProvenance, ...],
        BeforeValidator(_require_tuple),
    ] = ()
    execution_provenance: Annotated[
        tuple[HistoricalBacktestExecutionProvenance, ...],
        BeforeValidator(_require_tuple),
    ] = ()
    entries: Annotated[
        tuple[HistoricalBacktestEntryRecord, ...],
        BeforeValidator(_require_tuple),
    ] = ()
    exits: Annotated[
        tuple[HistoricalBacktestExitRecord, ...], BeforeValidator(_require_tuple)
    ] = ()
    trades: Annotated[
        tuple[HistoricalTradeRecord, ...], BeforeValidator(_require_tuple)
    ] = ()
    rejections: Annotated[
        tuple[HistoricalBacktestRejectionRecord, ...],
        BeforeValidator(_require_tuple),
    ] = ()
    cash_ledger: Annotated[
        tuple[HistoricalCashLedgerRow, ...], BeforeValidator(_require_tuple)
    ] = ()
    settlement_ledger: Annotated[
        tuple[HistoricalSettlementRecord, ...], BeforeValidator(_require_tuple)
    ] = ()
    session_pnl: Annotated[
        tuple[HistoricalBacktestSessionPnl, ...],
        BeforeValidator(_require_tuple),
    ] = ()
    equity_curve: Annotated[
        tuple[HistoricalBacktestEquityRow, ...],
        BeforeValidator(_require_tuple),
    ] = ()
    cost_summary: HistoricalBacktestCostSummary
    exit_reason_summary: HistoricalBacktestExitReasonSummary
    summary: HistoricalBacktestSummary
    audit_summary: HistoricalBacktestAuditSummary
    content_fingerprints: HistoricalBacktestContentFingerprints
    result_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_local_structure(self) -> Self:
        if not self.audit_summary.audit_passed:
            raise ValueError("a final audit result requires audit_passed=true")
        transition_sessions = tuple(row.session for row in self.session_transitions)
        if transition_sessions != tuple(sorted(transition_sessions)) or len(
            set(transition_sessions)
        ) != len(transition_sessions):
            raise ValueError(
                "session transitions must be unique and ordered by session"
            )
        equity_sessions = tuple(row.session for row in self.equity_curve)
        if equity_sessions != tuple(sorted(equity_sessions)) or len(
            set(equity_sessions)
        ) != len(equity_sessions):
            raise ValueError("equity rows must be unique and ordered by session")
        if self.summary.processed_session_count != len(self.session_transitions):
            raise ValueError(
                "processed_session_count must equal session transition count"
            )
        expected = compute_result_fingerprint(
            schema_version=self.schema_version,
            decision_interval=self.decision_interval,
            run_configuration_fingerprint=(
                self.run_manifest.run_configuration_fingerprint
            ),
            source_run_fingerprint=self.source_run_fingerprint,
            initial_state_fingerprint=self.initial_state_fingerprint,
            final_state_fingerprint=self.final_state_fingerprint,
            content_fingerprints=self.content_fingerprints,
        )
        if self.result_fingerprint != expected:
            raise ValueError("result_fingerprint does not match result components")
        return self


__all__ = [
    "ArtifactRef",
    "DividendAttributionCompleteness",
    "ExecutionApplicationStatus",
    "ExecutionProvenanceSource",
    "HistoricalAllocationCandidateProvenance",
    "HistoricalAllocationCycle",
    "HistoricalBacktestAuditResult",
    "HistoricalBacktestAuditSummary",
    "HistoricalBacktestContentFingerprints",
    "HistoricalBacktestCostSummary",
    "HistoricalBacktestEntryRecord",
    "HistoricalBacktestEquityRow",
    "HistoricalBacktestExecutionProvenance",
    "HistoricalBacktestExitReasonRow",
    "HistoricalBacktestExitReasonSummary",
    "HistoricalBacktestExitRecord",
    "HistoricalBacktestRejectionRecord",
    "HistoricalBacktestRunManifest",
    "HistoricalBacktestSessionPnl",
    "HistoricalBacktestSummary",
    "HistoricalBacktestTransitionAudit",
    "HistoricalBacktestValuationMark",
    "HistoricalBacktestValuationPolicy",
    "HistoricalBacktestValuationPolicyRef",
    "HistoricalBacktestValuationSnapshot",
    "HistoricalCashLedgerRow",
    "HistoricalClosedTradeRecord",
    "HistoricalExitReason",
    "HistoricalOpenTradeRecord",
    "HistoricalRankingCandidateProvenance",
    "HistoricalRankingCycle",
    "HistoricalRejectionStage",
    "HistoricalSettlementRecord",
    "HistoricalSettlementStatus",
    "HistoricalSignalProvenance",
    "HistoricalTradeRecord",
    "HistoricalTradeStatus",
    "PolicyArtifactRef",
    "build_run_manifest",
    "build_valuation_policy_ref",
    "build_valuation_snapshot",
    "compute_run_configuration_fingerprint",
]
