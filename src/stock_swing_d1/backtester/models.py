"""Immutable public contracts for Phase 15A historical chronology."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal, Protocol

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CanonicalDividendAccountingEvidence,
)
from stock_swing_d1.data.ordinary_dividend_run_evidence import (
    CanonicalDistributionCoverage,
    DividendAwareRunEvidence,
)
from stock_swing_d1.execution.entry import (
    EntryExecutionDecision,
    EntryExecutionStatus,
)
from stock_swing_d1.execution.open_position_exit import (
    ExitPrerequisiteStatus,
    OpenPositionExitDecision,
    OpenPositionExitEvaluationInput,
    OpenPositionExitReason,
)
from stock_swing_d1.indicators import TechnicalIndicatorRow
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    CorporateActionEvent,
    StockBar,
)
from stock_swing_d1.portfolio.models import (
    PortfolioCandidate,
    PortfolioCandidateDecision,
    PortfolioSnapshot,
    PortfolioReservationSettlement,
    RankedPortfolioAllocationDecision,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioEventKind,
    PortfolioExecutionEvent,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PortfolioState,
    PortfolioTransitionResult,
)
from stock_swing_d1.ranking.integration import RankedAllocationBatch
from stock_swing_d1.ranking.models import (
    CandidateRankingSnapshot,
    RankingCandidate,
)
from stock_swing_d1.strategy.baseline import BaselineSignalDecision

from stock_swing_d1.backtester.decision_interval import (
    HistoricalDecisionInterval,
)


_PROTECTIVE_EXIT_REASONS = frozenset(
    {
        OpenPositionExitReason.GAP_THROUGH_STOP,
        OpenPositionExitReason.GAP_THROUGH_TARGET,
        OpenPositionExitReason.STOP_LOSS,
        OpenPositionExitReason.TAKE_PROFIT,
    }
)


HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION = (
    "historical_backtest_session_input.v0.1"
)
DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION = (
    "historical_backtest_session_input.v0.2"
)
HISTORICAL_BACKTEST_ENTRY_INTENT_SCHEMA_VERSION = (
    "historical_backtest_entry_intent.v0.1"
)
HISTORICAL_BACKTEST_SESSION_PLAN_SCHEMA_VERSION = (
    "historical_backtest_session_plan.v0.1"
)
# Bumped v0.1 -> v0.2 (Task 5C-B): gained `entry_session_protective_decisions`,
# exactly one Phase 15B entry-session decision per Phase 9 EXECUTED entry,
# HOLD-only while same-session persistence was unsupported.
# Bumped v0.2 -> v0.3 (Task 5C-C): terminal (protective) elements are now
# admissible and must link to exactly one APPLIED BUY and one APPLIED
# ROUND_TRIP SELL of the same session.  A label binds one exact invariant
# set, so neither earlier label may be reused for this one.
HISTORICAL_BACKTEST_SESSION_RESULT_SCHEMA_VERSION = (
    "historical_backtest_session_result.v0.3"
)
HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION = (
    "historical_backtest_run_result.v0.2"
)
DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION = (
    "historical_backtest_run_result.v0.3"
)


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


class _ImmutableHistoricalBacktestModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )


_SessionDate = Annotated[date, BeforeValidator(_require_session_date)]
_AwareDateTime = Annotated[
    datetime,
    BeforeValidator(_require_aware_datetime),
]
_CanonicalSecurityId = Annotated[
    str,
    Field(pattern=r"^NORGATE:[1-9][0-9]*$"),
]
_PositiveRank = Annotated[int, Field(gt=0, strict=True)]
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _require_decision_interval(value: object) -> object:
    if type(value) is not HistoricalDecisionInterval:
        raise ValueError("must be exactly HistoricalDecisionInterval")
    return HistoricalDecisionInterval.model_validate(
        {
            field_name: getattr(value, field_name)
            for field_name in HistoricalDecisionInterval.model_fields
        }
    )


_HistoricalDecisionInterval = Annotated[
    HistoricalDecisionInterval,
    BeforeValidator(_require_decision_interval),
]


def _require_optional_distribution_events(value: object) -> object:
    if value is None:
        return None
    if type(value) is not tuple:
        raise ValueError("distribution_events must be an immutable tuple")
    if any(
        type(event) is not CanonicalDividendAccountingEvidence
        for event in value
    ):
        raise ValueError(
            "distribution_events must contain exact "
            "CanonicalDividendAccountingEvidence values"
        )
    return tuple(
        CanonicalDividendAccountingEvidence.model_validate(
            event.model_dump(mode="python")
        )
        for event in value
    )


def _require_optional_distribution_coverage(value: object) -> object:
    if value is None:
        return None
    if type(value) is not CanonicalDistributionCoverage:
        raise ValueError(
            "distribution_coverage must be exactly "
            "CanonicalDistributionCoverage"
        )
    return CanonicalDistributionCoverage.model_validate(
        {
            name: getattr(value, name)
            for name in CanonicalDistributionCoverage.model_fields
        }
    )


def _require_optional_dividend_run_evidence(value: object) -> object:
    if value is None:
        return None
    if type(value) is not DividendAwareRunEvidence:
        raise ValueError(
            "dividend_run_evidence must be exactly DividendAwareRunEvidence"
        )
    return DividendAwareRunEvidence.model_validate(
        {
            name: getattr(value, name)
            for name in DividendAwareRunEvidence.model_fields
        }
    )


_OptionalDistributionEvents = Annotated[
    tuple[CanonicalDividendAccountingEvidence, ...] | None,
    BeforeValidator(_require_optional_distribution_events),
]
_OptionalDistributionCoverage = Annotated[
    CanonicalDistributionCoverage | None,
    BeforeValidator(_require_optional_distribution_coverage),
]
_OptionalDividendRunEvidence = Annotated[
    DividendAwareRunEvidence | None,
    BeforeValidator(_require_optional_dividend_run_evidence),
]


class EntryExecutionEventAdapter(Protocol):
    """Authoritative transport seam from an executed Phase 9 BUY to Phase 13."""

    def build_buy_event(
        self,
        *,
        session: date,
        intent: HistoricalBacktestEntryIntent,
        entry_execution: EntryExecutionDecision,
    ) -> PortfolioExecutionEvent:
        ...


class OpenPositionExitEventAdapter(Protocol):
    """Authoritative transport seam from a Phase 15B exit to Phase 13.

    Implementations own every execution fact absent from the Phase 15B
    decision, including execution identifiers, administrative-exit fill,
    execution cost, and settlement metadata.  ``build_round_trip_sell_event``
    (Task 5C-C) adapts a terminal protective decision taken on the entry
    session itself against the Phase 15C transient exposure of that entry.
    """

    def build_sell_event(
        self,
        *,
        decision: OpenPositionExitDecision,
        open_position: OpenPosition,
    ) -> PortfolioExecutionEvent:
        ...

    def build_round_trip_sell_event(
        self,
        *,
        decision: OpenPositionExitDecision,
        exposure: object,
    ) -> PortfolioExecutionEvent:
        ...


class HistoricalBacktestEntryIntent(_ImmutableHistoricalBacktestModel):
    """A completed-T admission scheduled for one later market session.

    The nested Phase 12 objects remain authoritative for rank, reservation,
    sizing, and fixed quantity.  This wrapper adds chronology and audit
    linkage only; it is not an execution or open-position model.
    """

    schema_version: Literal[
        "historical_backtest_entry_intent.v0.1"
    ] = HISTORICAL_BACKTEST_ENTRY_INTENT_SCHEMA_VERSION
    allocation_session: _SessionDate
    planned_entry_session: _SessionDate
    security_id: _CanonicalSecurityId
    source_rank: _PositiveRank
    ranking_snapshot_fingerprint: _Sha256
    allocation_candidate: PortfolioCandidate
    candidate_decision: PortfolioCandidateDecision

    @model_validator(mode="after")
    def validate_chronology_and_identity(self) -> HistoricalBacktestEntryIntent:
        if self.planned_entry_session <= self.allocation_session:
            raise ValueError(
                "planned_entry_session must be after allocation_session"
            )
        candidate = self.allocation_candidate
        decision = self.candidate_decision
        if (
            candidate.security_id != self.security_id
            or decision.security_id != self.security_id
            or candidate.signal_session != self.allocation_session
            or candidate.pending_entry.planned_entry_session
            != self.planned_entry_session
            or decision.sized_pending_entry is None
            or decision.sized_pending_entry.planned_entry_session
            != self.planned_entry_session
        ):
            raise ValueError(
                "entry intent identity and nested Phase 12 artifacts disagree"
            )
        return self


class HistoricalBacktestSessionInput(_ImmutableHistoricalBacktestModel):
    """All immutable, historically available inputs for canonical session T.

    ``scheduled_execution_events`` are filled events already authorized by
    their frozen execution owners.  Phase 15A only validates, orders, and
    submits them to Phase 13.  ``ranking_candidates`` and
    ``allocation_candidates`` are the complete eligible completed-T batch;
    their strict one-to-one reconciliation remains Phase 14B-owned.
    """

    schema_version: Literal[
        "historical_backtest_session_input.v0.1",
        "historical_backtest_session_input.v0.2",
    ] = HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION
    session: _SessionDate
    decision_time: _AwareDateTime
    next_session: _SessionDate | None = None
    distribution_events: _OptionalDistributionEvents = None
    distribution_coverage: _OptionalDistributionCoverage = None
    pre_open_corporate_actions: tuple[CorporateActionEvent, ...] = ()
    scheduled_entry_intents: tuple[HistoricalBacktestEntryIntent, ...] = ()
    entry_execution_bars: tuple[StockBar, ...] = ()
    scheduled_execution_events: tuple[PortfolioExecutionEvent, ...] = ()
    open_position_exit_evaluations: tuple[
        OpenPositionExitEvaluationInput, ...
    ] = ()
    indicator_history_bars: tuple[CorporateActionAdjustedStockBar, ...] = ()
    completed_unadjusted_bars: tuple[StockBar, ...] = ()
    universe_eligible_security_ids: tuple[_CanonicalSecurityId, ...] = ()
    ranking_candidates: tuple[RankingCandidate, ...] = ()
    allocation_candidates: tuple[PortfolioCandidate, ...] = ()
    allocation_portfolio: PortfolioSnapshot | None = None

    @model_validator(mode="after")
    def validate_explicit_next_session(self) -> HistoricalBacktestSessionInput:
        if self.next_session is not None and self.next_session <= self.session:
            raise ValueError("next_session must be later than session")
        if self.schema_version == HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION and (
            self.distribution_events is not None
            or self.distribution_coverage is not None
        ):
            raise ValueError(
                "dividend evidence requires session input schema v0.2"
            )
        return self


class HistoricalBacktestSessionPlan(_ImmutableHistoricalBacktestModel):
    """The session-static half of one processed session (Task 5C).

    A plan carries only facts that are safely knowable before execution
    begins, so a whole run's plans can be pre-materialized and validated
    run-wide before the first Phase 13 transition.  Everything that depends
    on authoritative state -- carried entry intents, entry execution bars,
    open-position exit evaluations, ranking/allocation candidates and the
    allocation portfolio -- is deliberately absent: those are retrieved
    lazily from the state-aware provider or derived internally by their
    frozen owners.  ``scheduled_execution_events`` is absent too, so the
    provider path has no hidden external economic injection seam.
    """

    schema_version: Literal[
        "historical_backtest_session_plan.v0.1"
    ] = HISTORICAL_BACKTEST_SESSION_PLAN_SCHEMA_VERSION
    session: _SessionDate
    decision_time: _AwareDateTime
    next_session: _SessionDate | None = None
    distribution_events: _OptionalDistributionEvents = None
    distribution_coverage: _OptionalDistributionCoverage = None
    pre_open_corporate_actions: tuple[CorporateActionEvent, ...] = ()
    indicator_history_bars: tuple[CorporateActionAdjustedStockBar, ...] = ()
    completed_unadjusted_bars: tuple[StockBar, ...] = ()
    universe_eligible_security_ids: tuple[_CanonicalSecurityId, ...] = ()

    @model_validator(mode="after")
    def validate_explicit_next_session(self) -> HistoricalBacktestSessionPlan:
        if self.next_session is not None and self.next_session <= self.session:
            raise ValueError("next_session must be later than session")
        return self


class HistoricalBacktestSessionResult(_ImmutableHistoricalBacktestModel):
    """The immutable authoritative state and completed-T decision artifacts.

    ``entry_session_protective_decisions`` (Task 5C-B/5C-C) holds exactly one
    Phase 15B entry-session decision per Phase 9 ``EXECUTED`` entry of this
    session.  A HOLD element carries the exact Phase 10 state Phase 15A
    carries into T+1; a terminal element is a protective exit on the entry
    session itself and links to exactly one APPLIED BUY and one APPLIED
    ROUND_TRIP SELL of the same session.
    """

    schema_version: Literal[
        "historical_backtest_session_result.v0.3"
    ] = HISTORICAL_BACKTEST_SESSION_RESULT_SCHEMA_VERSION
    session: _SessionDate
    decision_time: _AwareDateTime
    prior_state_fingerprint: _Sha256
    scheduled_entry_intents: tuple[HistoricalBacktestEntryIntent, ...] = ()
    entry_execution_decisions: tuple[EntryExecutionDecision, ...] = ()
    reservation_settlements: tuple[PortfolioReservationSettlement, ...] = ()
    open_position_exit_decisions: tuple[OpenPositionExitDecision, ...] = ()
    entry_session_protective_decisions: tuple[
        OpenPositionExitDecision, ...
    ] = ()
    ordered_execution_events: tuple[PortfolioExecutionEvent, ...] = ()
    state_transition_result: PortfolioTransitionResult
    authoritative_state: PortfolioState
    indicator_rows: tuple[TechnicalIndicatorRow, ...] = ()
    signal_decisions: tuple[BaselineSignalDecision, ...] = ()
    ranking_snapshot: CandidateRankingSnapshot
    ranked_allocation_batch: RankedAllocationBatch | None = None
    allocation_decision: RankedPortfolioAllocationDecision | None = None
    future_entry_intents: tuple[HistoricalBacktestEntryIntent, ...] = ()

    @model_validator(mode="after")
    def validate_state_linkage(self) -> HistoricalBacktestSessionResult:
        transition = self.state_transition_result
        if (
            transition.session != self.session
            or transition.state_hash_before != self.prior_state_fingerprint
            or transition.resulting_state != self.authoritative_state
        ):
            raise ValueError(
                "session result and authoritative Phase 13 transition disagree"
            )
        if self.ranking_snapshot.ranking_session != self.session:
            raise ValueError("ranking snapshot must belong to result session")
        if self.ranking_snapshot.decision_time != self.decision_time:
            raise ValueError("ranking snapshot must preserve decision_time")
        exit_security_ids = tuple(
            decision.security_id
            for decision in self.open_position_exit_decisions
        )
        if exit_security_ids != tuple(sorted(exit_security_ids)):
            raise ValueError(
                "open-position exit decisions must use canonical security order"
            )
        if len(set(exit_security_ids)) != len(exit_security_ids):
            raise ValueError(
                "open-position exit decisions must be unique by security ID"
            )
        if any(
            decision.session != self.session
            for decision in self.open_position_exit_decisions
        ):
            raise ValueError(
                "open-position exit decisions must belong to result session"
            )
        if (self.ranked_allocation_batch is None) != (
            self.allocation_decision is None
        ):
            raise ValueError(
                "ranked batch and allocation decision must both be present or absent"
            )
        if self.allocation_decision is None and self.future_entry_intents:
            raise ValueError(
                "future entry intents require a completed allocation decision"
            )
        self._validate_entry_session_protective_decisions()
        return self

    def _validate_entry_session_protective_decisions(self) -> None:
        """Task 5C-B/5C-C invariants for the entry-session decision field."""

        decisions = self.entry_session_protective_decisions
        security_ids = tuple(decision.security_id for decision in decisions)
        if security_ids != tuple(sorted(security_ids)) or len(
            set(security_ids)
        ) != len(security_ids):
            raise ValueError(
                "entry-session protective decisions must be unique and in "
                "canonical security order"
            )
        executed_ids = {
            decision.security_id
            for decision in self.entry_execution_decisions
            if decision.status is EntryExecutionStatus.EXECUTED
        }
        if set(security_ids) != executed_ids:
            raise ValueError(
                "entry-session protective decisions must cover exactly the "
                "Phase 9 EXECUTED entries of this session"
            )
        if not decisions:
            return

        open_asset_ids = {
            position.asset_id
            for position in self.authoritative_state.open_positions
        }
        transition = self.state_transition_result
        newly_applied = {
            reference.event_id
            for reference in transition.newly_applied_events
            if reference.event_kind is PortfolioEventKind.EXECUTION
        }
        buy_events_by_asset: dict[str, list[PortfolioExecutionEvent]] = {}
        sell_events_by_asset: dict[str, list[PortfolioExecutionEvent]] = {}
        for event in self.ordered_execution_events:
            if event.execution_id not in newly_applied:
                continue  # REPLAYED presentations never satisfy linkage.
            target = (
                buy_events_by_asset
                if event.side is ExecutionSide.BUY
                else sell_events_by_asset
            )
            target.setdefault(event.asset_id, []).append(event)
        ledger_rows_by_asset: dict[
            tuple[str, PortfolioLedgerEventType], int
        ] = {}
        for entry in transition.ledger_entries:
            if entry.asset_id is not None and entry.event_type in (
                PortfolioLedgerEventType.BUY_APPLIED,
                PortfolioLedgerEventType.SELL_APPLIED,
            ):
                key = (entry.asset_id, entry.event_type)
                ledger_rows_by_asset[key] = ledger_rows_by_asset.get(key, 0) + 1

        for decision in decisions:
            asset_id = decision.security_id
            if (
                decision.session != self.session
                or decision.entry_session != self.session
                or decision.holding_session_number != 1
                or decision.exit_prerequisite_status
                is not ExitPrerequisiteStatus.READY
            ):
                raise ValueError(
                    "an entry-session protective decision must be a ready "
                    "holding-session-1 decision of this session"
                )
            protective = decision.protective_exit_decision
            if not decision.exit_required:
                resulting_state = protective.resulting_state
                if (
                    resulting_state is None
                    or resulting_state.last_evaluated_session != self.session
                    or asset_id not in open_asset_ids
                ):
                    raise ValueError(
                        "a HOLD entry-session decision must carry a Phase 10 "
                        "state evaluated on this session for a position open "
                        "in the authoritative state"
                    )
                continue

            # Terminal: a same-session round trip (Task 5C-C).
            buys = buy_events_by_asset.get(asset_id, ())
            sells = sell_events_by_asset.get(asset_id, ())
            if (
                decision.selected_reason not in _PROTECTIVE_EXIT_REASONS
                or decision.final_execution_price is None
                or asset_id in open_asset_ids
                or len(buys) != 1
                or len(sells) != 1
                or ledger_rows_by_asset.get(
                    (asset_id, PortfolioLedgerEventType.BUY_APPLIED), 0
                )
                != 1
                or ledger_rows_by_asset.get(
                    (asset_id, PortfolioLedgerEventType.SELL_APPLIED), 0
                )
                != 1
            ):
                raise ValueError(
                    "a terminal entry-session decision must be protective and "
                    "link to exactly one APPLIED BUY, one APPLIED ROUND_TRIP "
                    "SELL and their application ledger rows, with no "
                    "surviving position"
                )
            buy, sell = buys[0], sells[0]
            if (
                buy.session != self.session
                or sell.session != self.session
                or sell.quantity != buy.quantity
                or sell.execution_id == buy.execution_id
                or sell.fill_price != Decimal(str(decision.final_execution_price))
            ):
                raise ValueError(
                    "the round-trip SELL must close the full BUY quantity at "
                    "the exact Phase 10 final execution price on this session"
                )


class HistoricalBacktestRunResult(_ImmutableHistoricalBacktestModel):
    """One deterministic complete result for an explicit session sequence."""

    schema_version: Literal[
        "historical_backtest_run_result.v0.2",
        "historical_backtest_run_result.v0.3",
    ] = HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION
    decision_interval: _HistoricalDecisionInterval
    dividend_run_evidence: _OptionalDividendRunEvidence = None
    initial_state: PortfolioState
    final_state: PortfolioState
    session_results: tuple[HistoricalBacktestSessionResult, ...] = ()
    initial_state_fingerprint: _Sha256
    final_state_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_dividend_evidence_version(self) -> HistoricalBacktestRunResult:
        dividend_aware = self.dividend_run_evidence is not None
        if dividend_aware and self.schema_version != (
            DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION
        ):
            raise ValueError("dividend-aware result requires run schema v0.3")
        if not dividend_aware and self.schema_version != (
            HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION
        ):
            raise ValueError("legacy result requires run schema v0.2")
        return self


__all__ = [
    "DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION",
    "DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION",
    "HISTORICAL_BACKTEST_ENTRY_INTENT_SCHEMA_VERSION",
    "HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION",
    "HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION",
    "HISTORICAL_BACKTEST_SESSION_PLAN_SCHEMA_VERSION",
    "HISTORICAL_BACKTEST_SESSION_RESULT_SCHEMA_VERSION",
    "EntryExecutionEventAdapter",
    "HistoricalBacktestEntryIntent",
    "HistoricalDecisionInterval",
    "HistoricalBacktestRunResult",
    "HistoricalBacktestSessionInput",
    "HistoricalBacktestSessionPlan",
    "HistoricalBacktestSessionResult",
    "OpenPositionExitEventAdapter",
]
