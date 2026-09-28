"""Read-only validation of authoritative Phase 15A historical source runs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, datetime
from decimal import Decimal
from math import isclose, isfinite

from pydantic import ValidationError

from stock_swing_d1.backtest_results.errors import (
    HistoricalBacktestResultValidationError,
)
from stock_swing_d1.backtest_results.models import (
    ExecutionApplicationStatus,
    ExecutionProvenanceSource,
    HistoricalBacktestRunManifest,
)
from stock_swing_d1.backtester.models import (
    HistoricalBacktestEntryIntent,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionResult,
)
from stock_swing_d1.data.ordinary_dividend_accounting import (
    CanonicalDividendAccountingEvidence,
)
from stock_swing_d1.data.ordinary_dividend_run_evidence import (
    DividendAwareRunEvidence,
)
from stock_swing_d1.execution.entry.models import (
    EntryExecutionDecision,
    EntryExecutionStatus,
)
from stock_swing_d1.execution.open_position_exit.models import (
    ExitPrerequisiteStatus,
    OpenPositionExitDecision,
)
from stock_swing_d1.portfolio.allocation_policy import (
    PORTFOLIO_ALLOCATION_POLICY_REF,
    PortfolioAllocationPolicyRef,
)
from stock_swing_d1.portfolio.models import (
    PortfolioCandidateAction,
    PortfolioReservationSettlement,
    RankedPortfolioAllocationDecision,
    RankedPortfolioCandidateDecision,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioEventKind,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_invariants import (
    ROUND_TRIP_SELL_APPLICATION_RANK,
    PortfolioInvariantChecker,
    classify_session_applications,
)
from stock_swing_d1.backtester.orchestration import (
    execution_presentation_rank,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioSessionSnapshot,
    PortfolioState,
    PortfolioTransitionResult,
)
from stock_swing_d1.ranking.integration import (
    RankedAllocationValidationError,
    validate_ranked_allocation_batch,
)
from stock_swing_d1.ranking.models import CandidateRankingValidationError
from stock_swing_d1.ranking.service import validate_ranking_snapshot
from stock_swing_d1.earnings.integration import EarningsIntegrationAction
from stock_swing_d1.strategy.baseline.models import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


@dataclass(frozen=True, slots=True)
class _ExecutionLink:
    event: PortfolioExecutionEvent
    application_status: ExecutionApplicationStatus
    provenance_source: ExecutionProvenanceSource
    entry_decision: EntryExecutionDecision | None = None
    exit_decision: OpenPositionExitDecision | None = None
    # Task 5C-C: True only for an APPLIED SELL whose asset was absent from
    # the immutable session-start state and was bought by an APPLIED BUY of
    # the same session (a same-session round trip).  Never set for a
    # REPLAYED presentation or a prior-position SELL.
    round_trip: bool = False


@dataclass(frozen=True, slots=True)
class _ValidatedSourceRunContext:
    run_result: HistoricalBacktestRunResult
    run_manifest: HistoricalBacktestRunManifest | None
    states_before: tuple[PortfolioState, ...]
    execution_links: tuple[tuple[_ExecutionLink, ...], ...]


def _canonical_dataclass(value: object, expected_type: type[object]) -> object:
    if type(value) is not expected_type:
        raise TypeError(f"value must be exactly {expected_type.__name__}")
    rebuilt = expected_type(
        **{field.name: getattr(value, field.name) for field in fields(expected_type)}
    )
    if rebuilt != value:
        raise ValueError(f"{expected_type.__name__} is not canonical")
    return rebuilt


def _canonical_state(value: object) -> PortfolioState:
    if type(value) is not PortfolioState:
        raise TypeError("state must be exactly PortfolioState")
    rebuilt = PortfolioState.model_validate(value.model_dump(mode="python"))
    if rebuilt != value:
        raise ValueError("state is not canonical")
    PortfolioInvariantChecker.validate_state(rebuilt)
    return rebuilt


def _canonical_transition(value: object) -> PortfolioTransitionResult:
    if type(value) is not PortfolioTransitionResult:
        raise TypeError("transition must be exactly PortfolioTransitionResult")
    rebuilt = PortfolioTransitionResult.model_validate(
        value.model_dump(mode="python")
    )
    if rebuilt != value:
        raise ValueError("transition is not canonical")
    return rebuilt


def _canonical_dividend_run_evidence(value: object) -> DividendAwareRunEvidence:
    if type(value) is not DividendAwareRunEvidence:
        raise TypeError(
            "dividend_run_evidence must be exactly DividendAwareRunEvidence"
        )
    rebuilt = DividendAwareRunEvidence.model_validate(
        {
            field_name: getattr(value, field_name)
            for field_name in DividendAwareRunEvidence.model_fields
        }
    )
    if rebuilt != value:
        raise ValueError("dividend_run_evidence is not canonical")
    return rebuilt


def _dividend_evidence_by_session(
    dividend_run_evidence: DividendAwareRunEvidence | None,
    canonical_sessions: list[HistoricalBacktestSessionResult],
) -> dict[date, tuple[CanonicalDividendAccountingEvidence, ...]]:
    """Map each processed session to its retained Phase-15A-supplied
    dividend evidence, reusing OD-14.7's retained evidence as-is rather
    than re-deriving or inferring it.

    A non-dividend-aware run (``dividend_run_evidence is None``) supplies
    no evidence for any session; ``validate_dividend_application``'s own
    S == O check then fails closed if any session's authoritative
    ``dividend_outcomes``/``dividend_ledger_entries`` are non-empty anyway,
    without this helper needing to duplicate that proof.
    """

    if dividend_run_evidence is None:
        return {session.session: () for session in canonical_sessions}
    try:
        canonical = _canonical_dividend_run_evidence(dividend_run_evidence)
    except (TypeError, ValueError) as error:
        raise HistoricalBacktestResultValidationError(
            "NONCANONICAL_SOURCE_RUN",
            "dividend_run_evidence fails canonical reconstruction",
        ) from error
    by_session = {
        item.session: item.distribution_events for item in canonical.session_evidence
    }
    missing = tuple(
        session.session
        for session in canonical_sessions
        if session.session not in by_session
    )
    if missing:
        raise HistoricalBacktestResultValidationError(
            "DIVIDEND_PROVENANCE_MISMATCH",
            "every processed session in a dividend-aware run requires "
            "retained session dividend evidence coverage",
        )
    return {session.session: by_session[session.session] for session in canonical_sessions}


def _canonical_entry_intent(value: object) -> HistoricalBacktestEntryIntent:
    if type(value) is not HistoricalBacktestEntryIntent:
        raise HistoricalBacktestResultValidationError(
            "NONCANONICAL_SOURCE_RUN",
            "entry intents must be exactly HistoricalBacktestEntryIntent",
        )
    try:
        rebuilt = HistoricalBacktestEntryIntent.model_validate(
            {
                field_name: getattr(value, field_name)
                for field_name in HistoricalBacktestEntryIntent.model_fields
            }
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise HistoricalBacktestResultValidationError(
            "NONCANONICAL_SOURCE_RUN",
            "entry intent fails canonical reconstruction",
        ) from error
    if rebuilt != value:
        raise HistoricalBacktestResultValidationError(
            "NONCANONICAL_SOURCE_RUN",
            "entry intent must already be canonical",
        )
    return rebuilt


def _validate_entry_intent_models(
    session: HistoricalBacktestSessionResult,
) -> None:
    for intent in session.scheduled_entry_intents:
        _canonical_entry_intent(intent)
    for intent in session.future_entry_intents:
        _canonical_entry_intent(intent)


def _validate_signal_decisions(session: HistoricalBacktestSessionResult) -> None:
    security_ids: list[str] = []
    for value in session.signal_decisions:
        decision = _canonical_dataclass(value, BaselineSignalDecision)
        assert isinstance(decision, BaselineSignalDecision)
        if (
            type(decision.security_id) is not str
            or not decision.security_id.startswith("NORGATE:")
            or not decision.security_id.removeprefix("NORGATE:").isdigit()
            or decision.security_id.removeprefix("NORGATE:").startswith("0")
            or type(decision.symbol) is not str
            or not decision.symbol
            or decision.symbol != decision.symbol.strip()
            or type(decision.signal_session) is not date
            or not isinstance(decision.signal_time, datetime)
            or decision.signal_time.tzinfo is None
            or decision.signal_time.utcoffset() is None
            or type(decision.planned_entry_session) is not date
            or decision.planned_entry_session <= decision.signal_session
            or decision.signal_session != session.session
            or decision.signal_time > session.decision_time
            or type(decision.adjusted_close) is not float
            or not isfinite(decision.adjusted_close)
            or decision.adjusted_close <= 0.0
        ):
            raise ValueError("signal decision changed its explicit session boundary")
        if (
            type(decision.action) is not BaselineSignalAction
            or type(decision.earnings_action) is not EarningsIntegrationAction
            or any(
                type(value) is not bool
                for value in (
                    decision.universe_eligible,
                    decision.close_above_sma50,
                    decision.sma20_above_sma50,
                    decision.rsi_above_50,
                    decision.atr_above_minimum,
                    decision.earnings_entry_allowed,
                )
            )
        ):
            raise ValueError("signal decision contains noncanonical outcomes")
        for indicator in (
            decision.sma_20,
            decision.sma_50,
            decision.rsi_14,
            decision.atr_14,
            decision.atr_fraction,
        ):
            if indicator is not None and (
                type(indicator) is not float or not isfinite(indicator)
            ):
                raise ValueError("signal decision contains a noncanonical indicator")
        security_ids.append(decision.security_id)
    if tuple(security_ids) != tuple(sorted(security_ids)) or len(
        set(security_ids)
    ) != len(security_ids):
        raise ValueError("signal decisions must use canonical security order")


def _validate_entry_artifacts(session: HistoricalBacktestSessionResult) -> None:
    def linkage_error(message: str) -> HistoricalBacktestResultValidationError:
        return HistoricalBacktestResultValidationError(
            "ENTRY_LINKAGE_MISMATCH", message
        )

    canonical_intents = tuple(
        _canonical_entry_intent(intent)
        for intent in session.scheduled_entry_intents
    )
    intents_by_security = {
        intent.security_id: intent for intent in canonical_intents
    }
    if len(intents_by_security) != len(session.scheduled_entry_intents):
        raise linkage_error("scheduled entry intents must be unique by security")
    decision_ids: list[str] = []
    decisions_by_security: dict[str, EntryExecutionDecision] = {}
    for value in session.entry_execution_decisions:
        decision = _canonical_dataclass(value, EntryExecutionDecision)
        assert isinstance(decision, EntryExecutionDecision)
        intent = intents_by_security.get(decision.security_id)
        if intent is None or intent.candidate_decision.sized_pending_entry is None:
            raise linkage_error("entry decision lacks its scheduled source intent")
        sized = intent.candidate_decision.sized_pending_entry
        signal = intent.allocation_candidate.signal
        if (
            decision.planned_entry_session != session.session
            or decision.planned_entry_session != intent.planned_entry_session
            or decision.signal_session != intent.allocation_session
            or decision.signal_session != signal.signal_session
            or decision.signal_time != signal.signal_time
            or decision.symbol != signal.symbol
            or decision.symbol != intent.candidate_decision.symbol
            or decision.requested_shares != sized.fixed_shares
            or intent.source_rank
            != getattr(intent.candidate_decision, "source_rank", None)
            or intent.ranking_snapshot_fingerprint
            != getattr(
                intent.candidate_decision,
                "ranking_snapshot_fingerprint",
                None,
            )
            or decision.status is EntryExecutionStatus.PENDING_ENTRY
        ):
            raise linkage_error("entry decision changed source intent facts")
        decision_ids.append(decision.security_id)
        decisions_by_security[decision.security_id] = decision
    if tuple(decision_ids) != tuple(sorted(decision_ids)) or len(
        decisions_by_security
    ) != len(decision_ids):
        raise linkage_error("entry decisions must use canonical security order")
    if set(decisions_by_security) != set(intents_by_security):
        raise linkage_error("entry decisions must cover every scheduled intent")

    settlements = session.reservation_settlements
    if len(settlements) != len(session.entry_execution_decisions):
        raise linkage_error("reservation settlements must cover entry decisions")
    for settlement, decision in zip(
        settlements, session.entry_execution_decisions, strict=True
    ):
        if type(settlement) is not PortfolioReservationSettlement:
            raise TypeError("reservation settlement has an invalid type")
        intent = intents_by_security[decision.security_id]
        if (
            settlement.security_id != decision.security_id
            or settlement.symbol != decision.symbol
            or settlement.requested_shares != decision.requested_shares
            or settlement.executed_shares != decision.executed_shares
            or settlement.entry_execution_status is not decision.status
            or settlement.reserved_cash
            != intent.candidate_decision.reserved_cash
            or not isfinite(settlement.released_cash)
            or settlement.released_cash < 0.0
            or type(settlement.actual_cash_used) is not float
            or not isfinite(settlement.actual_cash_used)
            or settlement.actual_cash_used < 0.0
            or not isclose(
                settlement.reserved_cash,
                settlement.actual_cash_used + settlement.released_cash,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or type(settlement.slot_released) is not bool
            or type(settlement.slot_occupied_after_execution) is not bool
        ):
            raise linkage_error(
                "reservation settlement and entry decision disagree"
            )
        if decision.status is EntryExecutionStatus.EXECUTED:
            if (
                settlement.actual_cash_used != decision.actual_cash_required
                or settlement.slot_released is not False
                or settlement.slot_occupied_after_execution is not True
            ):
                raise linkage_error(
                    "executed entry reservation settlement is inconsistent"
                )
        elif (
            settlement.actual_cash_used != 0.0
            or not isclose(
                settlement.released_cash,
                settlement.reserved_cash,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or settlement.slot_released is not True
            or settlement.slot_occupied_after_execution is not False
        ):
            raise linkage_error(
                "non-executed entry reservation settlement is inconsistent"
            )


def _validate_exit_decisions(session: HistoricalBacktestSessionResult) -> None:
    for value in session.open_position_exit_decisions:
        decision = _canonical_dataclass(value, OpenPositionExitDecision)
        assert isinstance(decision, OpenPositionExitDecision)
        if decision.session != session.session:
            raise ValueError("exit decision changed its explicit session")
    for value in session.entry_session_protective_decisions:
        decision = _canonical_dataclass(value, OpenPositionExitDecision)
        assert isinstance(decision, OpenPositionExitDecision)
        if (
            decision.session != session.session
            or decision.entry_session != session.session
            or decision.holding_session_number != 1
        ):
            raise ValueError(
                "entry-session protective decision changed its explicit session"
            )


def _validate_allocation(
    session: HistoricalBacktestSessionResult,
    *,
    manifest: HistoricalBacktestRunManifest | None,
) -> None:
    batch = session.ranked_allocation_batch
    allocation = session.allocation_decision
    if batch is None and allocation is None:
        if session.future_entry_intents:
            raise ValueError("future entry intents require an allocation")
        return
    if batch is None or allocation is None:
        raise ValueError("ranked batch and allocation must both be present")
    try:
        validated_batch = validate_ranked_allocation_batch(batch)
    except RankedAllocationValidationError as error:
        raise HistoricalBacktestResultValidationError(
            "ARTIFACT_PROVENANCE_MISMATCH",
            "ranked allocation batch is not canonical",
        ) from error
    if type(allocation) is not RankedPortfolioAllocationDecision:
        raise TypeError("allocation must be RankedPortfolioAllocationDecision")

    policy_ref = allocation.allocation_policy_ref
    if type(policy_ref) is not PortfolioAllocationPolicyRef:
        raise TypeError("allocation policy ref has an invalid type")
    rebuilt_policy_ref = PortfolioAllocationPolicyRef(
        **{
            field.name: getattr(policy_ref, field.name)
            for field in fields(PortfolioAllocationPolicyRef)
        }
    )
    if (
        rebuilt_policy_ref != policy_ref
        or policy_ref != PORTFOLIO_ALLOCATION_POLICY_REF
    ):
        raise HistoricalBacktestResultValidationError(
            "ARTIFACT_PROVENANCE_MISMATCH",
            "allocation result has noncanonical Phase 12 policy provenance",
        )
    if manifest is not None:
        expected_policy = manifest.allocation_policy_ref
        if (
            policy_ref.policy_id != expected_policy.policy_id
            or policy_ref.policy_version != expected_policy.policy_version
            or policy_ref.policy_fingerprint
            != expected_policy.policy_fingerprint
        ):
            raise HistoricalBacktestResultValidationError(
                "ARTIFACT_PROVENANCE_MISMATCH",
                "allocation policy does not match the run manifest",
            )

    snapshot = session.ranking_snapshot
    if (
        validated_batch.ranking_session != session.session
        or validated_batch.decision_time != session.decision_time
        or allocation.allocation_session != session.session
        or allocation.decision_time != session.decision_time
        or allocation.ranking_snapshot_fingerprint
        != snapshot.snapshot_fingerprint
        or allocation.ranking_snapshot_fingerprint
        != validated_batch.ranking_snapshot_fingerprint
        or allocation.policy_fingerprint != snapshot.policy.policy_fingerprint
        or allocation.policy_fingerprint != validated_batch.policy_fingerprint
    ):
        raise HistoricalBacktestResultValidationError(
            "ARTIFACT_PROVENANCE_MISMATCH",
            "allocation cycle and ranking source disagree",
        )

    decisions = allocation.candidate_decisions
    aggregate_floats = (
        allocation.starting_portfolio_equity,
        allocation.starting_cash,
        allocation.total_reserved_cash,
        allocation.remaining_unreserved_cash,
    )
    aggregate_counts = (
        allocation.starting_open_position_count,
        allocation.admitted_count,
        allocation.ending_used_slots,
    )
    if any(
        type(value) not in {int, float}
        or not isfinite(value)
        or value < 0.0
        for value in aggregate_floats
    ) or any(
        type(value) is not int or value < 0 for value in aggregate_counts
    ):
        raise ValueError("allocation aggregate diagnostics are invalid")
    # Phase 15D compares the float Phase 12 actually consumed against the one
    # prescribed forward projection of authoritative Phase 13 settled cash.
    # The reverse direction (`Decimal(str(allocation.starting_cash))` against
    # the authoritative Decimal) is not a valid authority check: a legitimate
    # high-precision balance -- settled cash carrying Gate3 scale-38 ordinary
    # dividend cash -- can fail `Decimal(str(float(value))) == value` while
    # the forward projection is exactly right, which would reject a correct
    # dividend-aware run.
    if allocation.starting_cash != float(
        session.authoritative_state.settled_cash
    ):
        raise HistoricalBacktestResultValidationError(
            "ARTIFACT_PROVENANCE_MISMATCH",
            "allocation starting cash does not match authoritative Phase 13 settled cash",
        )
    if allocation.starting_open_position_count != len(
        session.authoritative_state.open_positions
    ):
        raise HistoricalBacktestResultValidationError(
            "ARTIFACT_PROVENANCE_MISMATCH",
            "allocation starting open-position count does not match authoritative Phase 13 state",
        )
    if len(decisions) != len(validated_batch.candidates):
        raise HistoricalBacktestResultValidationError(
            "ARTIFACT_PROVENANCE_MISMATCH",
            "allocation did not preserve the complete ranked batch",
        )
    expected_ranks = tuple(range(1, len(decisions) + 1))
    if tuple(decision.processing_rank for decision in decisions) != expected_ranks:
        raise HistoricalBacktestResultValidationError(
            "ARTIFACT_PROVENANCE_MISMATCH",
            "allocation processing ranks are not canonical",
        )

    admitted_count = 0
    reserved_total = 0.0
    for ranked, decision in zip(
        validated_batch.candidates, decisions, strict=True
    ):
        if type(decision) is not RankedPortfolioCandidateDecision:
            raise TypeError("allocation candidate decision has an invalid type")
        if (
            type(decision.source_rank) is not int
            or type(decision.processing_rank) is not int
            or type(decision.used_slots_before) is not int
            or type(decision.used_slots_after) is not int
            or decision.source_rank <= 0
            or decision.processing_rank <= 0
            or decision.used_slots_before < 0
            or decision.used_slots_after < 0
            or type(decision.action) is not PortfolioCandidateAction
            or decision.security_id != ranked.security_id
            or decision.source_rank != ranked.rank
            or decision.processing_rank != ranked.rank
            or decision.ranking_snapshot_fingerprint
            != ranked.ranking_snapshot_fingerprint
            or decision.ranking_input_fingerprint
            != ranked.ranking_input_fingerprint
        ):
            raise HistoricalBacktestResultValidationError(
                "ARTIFACT_PROVENANCE_MISMATCH",
                "allocation candidate and ranking lineage disagree",
            )
        ranked_source = snapshot.ranked_candidates[ranked.rank - 1]
        if (
            ranked_source.security_id != decision.security_id
            or ranked_source.input_fingerprint
            != decision.ranking_input_fingerprint
            or snapshot.snapshot_fingerprint
            != decision.ranking_snapshot_fingerprint
        ):
            raise HistoricalBacktestResultValidationError(
                "ARTIFACT_PROVENANCE_MISMATCH",
                "allocation candidate does not identify its ranking source",
            )
        numeric_values = (
            decision.unreserved_cash_before,
            decision.unreserved_cash_after,
            decision.reserved_cash,
        )
        if any(
            type(value) is not float or not isfinite(value) or value < 0.0
            for value in numeric_values
        ):
            raise ValueError("allocation decision contains invalid cash diagnostics")
        if decision.candidate_cash_limit is not None and (
            type(decision.candidate_cash_limit) is not float
            or not isfinite(decision.candidate_cash_limit)
            or decision.candidate_cash_limit < 0.0
        ):
            raise ValueError("candidate cash limit is invalid")
        if decision.action is PortfolioCandidateAction.ADMITTED:
            admitted_count += 1
            if (
                decision.sized_pending_entry is None
                or decision.used_slots_after != decision.used_slots_before + 1
                or decision.candidate_cash_limit != decision.reserved_cash
            ):
                raise ValueError("admitted allocation candidate is inconsistent")
        elif (
            decision.sized_pending_entry is not None
            or decision.reserved_cash != 0.0
            or decision.used_slots_after != decision.used_slots_before
        ):
            raise ValueError("rejected allocation candidate is inconsistent")
        reserved_total += decision.reserved_cash

    if (
        allocation.admitted_count != admitted_count
        or not isclose(
            allocation.total_reserved_cash,
            reserved_total,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or allocation.ending_used_slots
        != allocation.starting_open_position_count + admitted_count
        or not isclose(
            allocation.remaining_unreserved_cash,
            allocation.starting_cash - allocation.total_reserved_cash,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("allocation aggregate diagnostics are inconsistent")
    if decisions:
        if (
            decisions[0].used_slots_before
            != allocation.starting_open_position_count
            or decisions[0].unreserved_cash_before != allocation.starting_cash
            or decisions[-1].used_slots_after != allocation.ending_used_slots
            or decisions[-1].unreserved_cash_after
            != allocation.remaining_unreserved_cash
        ):
            raise ValueError("allocation candidate chain is inconsistent")
        for previous, current in zip(decisions, decisions[1:], strict=False):
            if (
                previous.used_slots_after != current.used_slots_before
                or previous.unreserved_cash_after
                != current.unreserved_cash_before
            ):
                raise ValueError("allocation candidate chain is discontinuous")

    admitted_sources = tuple(
        (ranked, decision)
        for ranked, decision in zip(
            validated_batch.candidates, decisions, strict=True
        )
        if decision.action is PortfolioCandidateAction.ADMITTED
    )
    if len(session.future_entry_intents) != len(admitted_sources):
        raise ValueError("future entry intents do not cover admitted candidates")
    for raw_intent, (ranked, decision) in zip(
        session.future_entry_intents, admitted_sources, strict=True
    ):
        intent = _canonical_entry_intent(raw_intent)
        sized = decision.sized_pending_entry
        assert sized is not None
        if (
            intent.allocation_session != session.session
            or intent.planned_entry_session != sized.planned_entry_session
            or intent.security_id != decision.security_id
            or intent.security_id != ranked.security_id
            or intent.source_rank != decision.source_rank
            or intent.ranking_snapshot_fingerprint
            != decision.ranking_snapshot_fingerprint
            or intent.allocation_candidate != ranked.allocation_candidate
            or intent.candidate_decision != decision
        ):
            raise ValueError("future entry intent changed allocation provenance")


def _validate_cross_session_entry_intent_carry(
    sessions: tuple[HistoricalBacktestSessionResult, ...],
) -> None:
    for previous, current in zip(sessions, sessions[1:], strict=False):
        if (
            previous.future_entry_intents
            and current.scheduled_entry_intents
            != previous.future_entry_intents
        ):
            raise HistoricalBacktestResultValidationError(
                "ENTRY_LINKAGE_MISMATCH",
                "next source session does not preserve the exact Phase 15A entry-intent carry",
            )


def _application_statuses(
    session: HistoricalBacktestSessionResult,
) -> dict[str, ExecutionApplicationStatus]:
    transition = session.state_transition_result
    newly = tuple(
        reference.event_id
        for reference in transition.newly_applied_events
        if reference.event_kind is PortfolioEventKind.EXECUTION
    )
    replayed = tuple(
        reference.event_id
        for reference in transition.replayed_events
        if reference.event_kind is PortfolioEventKind.EXECUTION
    )
    if len(set(newly)) != len(newly) or len(set(replayed)) != len(replayed):
        raise HistoricalBacktestResultValidationError(
            "EXECUTION_PROVENANCE_MISMATCH",
            "Phase 13 execution references contain duplicates",
        )
    if set(newly) & set(replayed):
        raise HistoricalBacktestResultValidationError(
            "EXECUTION_PROVENANCE_MISMATCH",
            "one execution cannot be both newly applied and replayed",
        )
    statuses = {
        execution_id: ExecutionApplicationStatus.APPLIED
        for execution_id in newly
    }
    statuses.update(
        {
            execution_id: ExecutionApplicationStatus.REPLAYED
            for execution_id in replayed
        }
    )
    source_ids = tuple(event.execution_id for event in session.ordered_execution_events)
    if set(statuses) != set(source_ids) or len(source_ids) != len(set(source_ids)):
        raise HistoricalBacktestResultValidationError(
            "EXECUTION_PROVENANCE_MISMATCH",
            "ordered executions and Phase 13 application references disagree",
        )
    return statuses


def _link_executions(
    session: HistoricalBacktestSessionResult,
    *,
    state_before: PortfolioState,
) -> tuple[_ExecutionLink, ...]:
    events = tuple(
        PortfolioExecutionEvent.model_validate(event.model_dump(mode="python"))
        for event in session.ordered_execution_events
    )
    if events != session.ordered_execution_events:
        raise HistoricalBacktestResultValidationError(
            "EXECUTION_PROVENANCE_MISMATCH",
            "ordered execution event is not canonical",
        )

    # Task 5C-C: APPLIED versus REPLAYED is established from the authoritative
    # transition evidence FIRST; only then are APPLIED SELLs classified by the
    # shared P/B rule (P = immutable session-start positions, B = the
    # session's APPLIED BUY assets).  REPLAYED presentations keep their
    # legacy presentation rank and never enter B, cardinality, or linkage.
    statuses = _application_statuses(session)
    replayed_ids = frozenset(
        execution_id
        for execution_id, status in statuses.items()
        if status is ExecutionApplicationStatus.REPLAYED
    )
    session_start_asset_ids = frozenset(
        position.asset_id for position in state_before.open_positions
    )
    applied_events = tuple(
        event for event in events if event.execution_id not in replayed_ids
    )
    classification = classify_session_applications(
        session_start_asset_ids=session_start_asset_ids,
        buy_asset_ids=tuple(
            event.asset_id
            for event in applied_events
            if event.side is ExecutionSide.BUY
        ),
        sell_asset_ids=tuple(
            event.asset_id
            for event in applied_events
            if event.side is ExecutionSide.SELL
        ),
    )
    if not classification.is_valid:
        raise HistoricalBacktestResultValidationError(
            "EXECUTION_PROVENANCE_MISMATCH",
            "applied executions violate the Phase 13 application "
            "classification or per-asset cardinality",
        )
    round_trip_asset_ids = frozenset(
        asset_id
        for asset_id, rank in classification.sell_rank_by_asset.items()
        if rank == ROUND_TRIP_SELL_APPLICATION_RANK
    )
    expected_order = tuple(
        sorted(
            events,
            key=lambda event: (
                execution_presentation_rank(
                    event,
                    session_start_asset_ids=session_start_asset_ids,
                    replayed_execution_ids=replayed_ids,
                ),
                event.asset_id,
                event.execution_id,
            ),
        )
    )
    if events != expected_order:
        raise HistoricalBacktestResultValidationError(
            "EXECUTION_PROVENANCE_MISMATCH",
            "ordered executions do not preserve Phase 15A order",
        )

    used: set[int] = set()
    entry_by_index: dict[int, EntryExecutionDecision] = {}
    for decision in session.entry_execution_decisions:
        if decision.status is not EntryExecutionStatus.EXECUTED:
            continue
        quote = decision.execution_cost_quote
        assert quote is not None
        candidates = tuple(
            index
            for index, event in enumerate(events)
            if index not in used
            and event.side is ExecutionSide.BUY
            and event.session == session.session
            and event.asset_id == decision.security_id
            and event.quantity == decision.executed_shares
            and event.fill_price == Decimal(str(decision.execution_price))
            and event.execution_cost == quote.execution_cost
        )
        if len(candidates) != 1:
            raise HistoricalBacktestResultValidationError(
                "ENTRY_LINKAGE_MISMATCH",
                "executed Phase 9 decision does not identify exactly one BUY",
            )
        index = candidates[0]
        used.add(index)
        entry_by_index[index] = decision

    positions = {position.asset_id: position for position in state_before.open_positions}
    exit_by_index: dict[int, OpenPositionExitDecision] = {}
    for decision in session.open_position_exit_decisions:
        sell_indexes = tuple(
            index
            for index, event in enumerate(events)
            if event.side is ExecutionSide.SELL
            and event.asset_id == decision.security_id
        )
        terminal = (
            decision.exit_required is True
            and decision.exit_prerequisite_status is ExitPrerequisiteStatus.READY
        )
        if not terminal:
            if sell_indexes:
                raise HistoricalBacktestResultValidationError(
                    "EXIT_LINKAGE_MISMATCH",
                    "nonterminal Phase 15B decision contradicts a SELL event",
                )
            continue
        position = positions.get(decision.security_id)
        if position is None:
            raise HistoricalBacktestResultValidationError(
                "EXIT_LINKAGE_MISMATCH",
                "terminal Phase 15B decision lacks its source open position",
            )
        candidates = tuple(
            index
            for index in sell_indexes
            if index not in used
            and events[index].session == session.session
            and events[index].quantity == position.quantity
            and (
                decision.final_execution_price is None
                or events[index].fill_price
                == Decimal(str(decision.final_execution_price))
            )
        )
        if len(candidates) != 1:
            raise HistoricalBacktestResultValidationError(
                "EXIT_LINKAGE_MISMATCH",
                "terminal Phase 15B decision does not identify exactly one SELL",
            )
        index = candidates[0]
        used.add(index)
        exit_by_index[index] = decision

    # Task 5C-C: a terminal entry-session decision links to exactly one
    # APPLIED BUY and exactly one APPLIED ROUND_TRIP SELL of this session,
    # by (session, asset_id) under the frozen cardinality; a HOLD links to
    # nothing.  REPLAYED presentations can never satisfy this linkage.
    round_trip_by_index: dict[int, OpenPositionExitDecision] = {}
    for decision in session.entry_session_protective_decisions:
        asset_id = decision.security_id
        if not decision.exit_required:
            if asset_id in round_trip_asset_ids:
                raise HistoricalBacktestResultValidationError(
                    "EXIT_LINKAGE_MISMATCH",
                    "a HOLD entry-session decision contradicts a round-trip SELL",
                )
            continue
        if asset_id not in round_trip_asset_ids or asset_id in positions:
            raise HistoricalBacktestResultValidationError(
                "EXIT_LINKAGE_MISMATCH",
                "terminal entry-session decision lacks its APPLIED "
                "same-session round-trip SELL",
            )
        buy_indexes = tuple(
            index
            for index in entry_by_index
            if events[index].asset_id == asset_id
        )
        if len(buy_indexes) != 1:
            raise HistoricalBacktestResultValidationError(
                "EXIT_LINKAGE_MISMATCH",
                "terminal entry-session decision lacks its APPLIED BUY",
            )
        buy_event = events[buy_indexes[0]]
        final_price = decision.final_execution_price
        candidates = tuple(
            index
            for index, event in enumerate(events)
            if index not in used
            and event.side is ExecutionSide.SELL
            and event.execution_id not in replayed_ids
            and event.asset_id == asset_id
            and event.session == session.session
            and event.quantity == buy_event.quantity
            and final_price is not None
            and event.fill_price == Decimal(str(final_price))
        )
        if len(candidates) != 1:
            raise HistoricalBacktestResultValidationError(
                "EXIT_LINKAGE_MISMATCH",
                "terminal entry-session decision does not identify exactly one "
                "APPLIED round-trip SELL",
            )
        index = candidates[0]
        used.add(index)
        round_trip_by_index[index] = decision
    unlinked_round_trips = round_trip_asset_ids - {
        events[index].asset_id for index in round_trip_by_index
    }
    if unlinked_round_trips:
        raise HistoricalBacktestResultValidationError(
            "EXIT_LINKAGE_MISMATCH",
            "an APPLIED round-trip SELL lacks its terminal entry-session decision",
        )

    links: list[_ExecutionLink] = []
    for index, event in enumerate(events):
        entry = entry_by_index.get(index)
        exit_decision = exit_by_index.get(index)
        round_trip_decision = round_trip_by_index.get(index)
        source = ExecutionProvenanceSource.EXTERNAL_SCHEDULED
        if entry is not None:
            source = ExecutionProvenanceSource.GENERATED_ENTRY
        elif exit_decision is not None or round_trip_decision is not None:
            source = ExecutionProvenanceSource.GENERATED_EXIT
        links.append(
            _ExecutionLink(
                event=event,
                application_status=statuses[event.execution_id],
                provenance_source=source,
                entry_decision=entry,
                exit_decision=(
                    round_trip_decision
                    if round_trip_decision is not None
                    else exit_decision
                ),
                round_trip=round_trip_decision is not None,
            )
        )
    return tuple(links)


def _validate_cost_policy(
    run: HistoricalBacktestRunResult,
    manifest: HistoricalBacktestRunManifest,
) -> None:
    for session in run.session_results:
        for decision in session.entry_execution_decisions:
            quotes = tuple(
                quote
                for quote in (
                    decision.candidate_execution_cost_quote,
                    decision.execution_cost_quote,
                )
                if quote is not None
            )
            for quote in quotes:
                if quote.policy_ref != manifest.execution_cost_policy_ref:
                    raise HistoricalBacktestResultValidationError(
                        "EXECUTION_COST_POLICY_MISMATCH",
                        "entry cost policy does not match the run manifest",
                    )


def _build_validated_source_context(
    *,
    run_result: HistoricalBacktestRunResult,
    run_manifest: HistoricalBacktestRunManifest | None,
) -> _ValidatedSourceRunContext:
    if type(run_result) is not HistoricalBacktestRunResult:
        raise HistoricalBacktestResultValidationError(
            "INVALID_SOURCE_RUN",
            "run_result must be exactly HistoricalBacktestRunResult",
        )
    try:
        run = HistoricalBacktestRunResult.model_validate(
            {
                field_name: getattr(run_result, field_name)
                for field_name in HistoricalBacktestRunResult.model_fields
            }
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise HistoricalBacktestResultValidationError(
            "NONCANONICAL_SOURCE_RUN",
            "source run fails canonical reconstruction",
        ) from error
    if run != run_result:
        raise HistoricalBacktestResultValidationError(
            "NONCANONICAL_SOURCE_RUN",
            "source run must already be canonical",
        )

    try:
        initial_state = _canonical_state(run.initial_state)
    except Exception as error:
        raise HistoricalBacktestResultValidationError(
            "NONCANONICAL_SOURCE_RUN", "initial state is not canonical"
        ) from error
    if hash_portfolio_state(initial_state) != run.initial_state_fingerprint:
        raise HistoricalBacktestResultValidationError(
            "INITIAL_STATE_HASH_MISMATCH",
            "initial state fingerprint does not match initial state",
        )
    try:
        final_state = _canonical_state(run.final_state)
    except Exception as error:
        raise HistoricalBacktestResultValidationError(
            "NONCANONICAL_SOURCE_RUN", "final state is not canonical"
        ) from error
    if hash_portfolio_state(final_state) != run.final_state_fingerprint:
        raise HistoricalBacktestResultValidationError(
            "FINAL_STATE_HASH_MISMATCH",
            "final state fingerprint does not match final state",
        )

    sessions = run.session_results
    session_dates = tuple(session.session for session in sessions)
    if any(
        current <= previous
        for previous, current in zip(session_dates, session_dates[1:], strict=False)
    ):
        raise HistoricalBacktestResultValidationError(
            "STATE_CHAIN_BREAK", "source sessions are not strictly chronological"
        )
    interval = run.decision_interval
    if any(
        not (
            interval.decision_start_date
            <= session_date
            <= interval.decision_end_date
        )
        for session_date in session_dates
    ):
        raise HistoricalBacktestResultValidationError(
            "SESSION_OUTSIDE_DECISION_INTERVAL",
            "every authoritative source session must lie within the closed "
            "historical decision interval",
        )
    canonical_sessions: list[HistoricalBacktestSessionResult] = []
    for session in sessions:
        try:
            if type(session) is not HistoricalBacktestSessionResult:
                raise TypeError("invalid session result type")
            rebuilt = HistoricalBacktestSessionResult.model_validate(
                {
                    field_name: getattr(session, field_name)
                    for field_name in HistoricalBacktestSessionResult.model_fields
                }
            )
            if rebuilt != session:
                raise ValueError("session result is not canonical")
            canonical_sessions.append(rebuilt)
            _validate_signal_decisions(rebuilt)
            _validate_entry_intent_models(rebuilt)
            _validate_entry_artifacts(rebuilt)
            _validate_exit_decisions(rebuilt)
        except HistoricalBacktestResultValidationError:
            raise
        except Exception as error:
            raise HistoricalBacktestResultValidationError(
                "NONCANONICAL_SOURCE_RUN",
                "source session fails canonical reconstruction",
            ) from error

    states_before: list[PortfolioState] = []
    previous_state = initial_state
    if not canonical_sessions:
        if (
            initial_state != final_state
            or run.initial_state_fingerprint != run.final_state_fingerprint
        ):
            raise HistoricalBacktestResultValidationError(
                "STATE_CHAIN_BREAK",
                "empty source run must preserve its initial state",
            )
    for session in canonical_sessions:
        states_before.append(previous_state)
        transition = session.state_transition_result
        before_hash = hash_portfolio_state(previous_state)
        after_hash = hash_portfolio_state(session.authoritative_state)
        if (
            session.prior_state_fingerprint != before_hash
            or transition.state_hash_before != before_hash
            or transition.session != session.session
            or transition.resulting_state != session.authoritative_state
            or transition.state_hash_after != after_hash
            or session.authoritative_state.as_of_session != session.session
        ):
            raise HistoricalBacktestResultValidationError(
                "STATE_CHAIN_BREAK",
                "source session does not continue the authoritative state chain",
            )
        previous_state = session.authoritative_state
    if canonical_sessions and (
        previous_state != final_state
        or hash_portfolio_state(previous_state) != run.final_state_fingerprint
    ):
        raise HistoricalBacktestResultValidationError(
            "STATE_CHAIN_BREAK", "last source session does not equal final state"
        )

    for state_before, session in zip(
        states_before, canonical_sessions, strict=True
    ):
        try:
            transition = _canonical_transition(session.state_transition_result)
            PortfolioInvariantChecker.validate_transition(
                state_before,
                transition.resulting_state,
                transition.ledger_entries,
                dividend_ledger_entries=transition.dividend_ledger_entries,
            )
            PortfolioInvariantChecker.validate_execution_ledger_provenance(
                session.ordered_execution_events,
                transition.ledger_entries,
            )
        except Exception as error:
            raise HistoricalBacktestResultValidationError(
                "INVALID_PHASE13_TRANSITION",
                "source transition fails authoritative Phase 13 validation",
            ) from error

    # Site 3 (OD-14.7): re-prove OD-14.5 row->evidence provenance and
    # OD-14.8/14.9 evidence<->outcome discharge for every processed
    # session, reusing site 1's own validate_dividend_application against
    # the independently reconstructed pre-session state chain
    # (states_before) rather than deriving Q_T/attribution from Phase 15D
    # holdings, a calendar, or a provider.
    dividend_evidence_by_session = _dividend_evidence_by_session(
        run.dividend_run_evidence, canonical_sessions
    )
    for state_before, session in zip(
        states_before, canonical_sessions, strict=True
    ):
        transition = _canonical_transition(session.state_transition_result)
        try:
            PortfolioInvariantChecker.validate_dividend_application(
                session=session.session,
                previous_state=state_before,
                dividend_evidence=dividend_evidence_by_session[session.session],
                dividend_ledger_entries=transition.dividend_ledger_entries,
                dividend_outcomes=transition.dividend_outcomes,
            )
        except Exception as error:
            raise HistoricalBacktestResultValidationError(
                "DIVIDEND_PROVENANCE_MISMATCH",
                "source dividend application fails authoritative Phase 13 "
                "provenance/discharge validation",
            ) from error

    snapshots = tuple(
        PortfolioSessionSnapshot(
            session=session.session,
            state_version=session.authoritative_state.state_version,
            settled_cash=session.authoritative_state.settled_cash,
            open_position_count=len(session.authoritative_state.open_positions),
            pending_settlement_count=len(
                session.authoritative_state.pending_settlements
            ),
            state_hash=hash_portfolio_state(session.authoritative_state),
        )
        for session in canonical_sessions
    )
    ledger_entries = tuple(
        entry
        for session in canonical_sessions
        for entry in session.state_transition_result.ledger_entries
    )
    dividend_ledger_entries = tuple(
        entry
        for session in canonical_sessions
        for entry in session.state_transition_result.dividend_ledger_entries
    )
    dividend_outcomes = tuple(
        outcome
        for session in canonical_sessions
        for outcome in session.state_transition_result.dividend_outcomes
    )
    dividend_evidence = tuple(
        event
        for session in canonical_sessions
        for event in dividend_evidence_by_session[session.session]
    )
    try:
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            initial_state=initial_state,
            final_state=final_state,
            ledger_entries=ledger_entries,
            session_snapshots=snapshots,
            dividend_ledger_entries=dividend_ledger_entries,
            dividend_evidence=dividend_evidence,
            dividend_outcomes=dividend_outcomes,
        )
    except Exception as error:
        raise HistoricalBacktestResultValidationError(
            "PHASE13_ACCOUNTING_AUDIT_FAILED",
            "complete persisted accounting history failed validation",
        ) from error

    manifest: HistoricalBacktestRunManifest | None = None
    if run_manifest is not None:
        if type(run_manifest) is not HistoricalBacktestRunManifest:
            raise HistoricalBacktestResultValidationError(
                "INVALID_SOURCE_RUN",
                "run_manifest must be exactly HistoricalBacktestRunManifest",
            )
        try:
            manifest = HistoricalBacktestRunManifest.model_validate(
                {
                    field_name: getattr(run_manifest, field_name)
                    for field_name in HistoricalBacktestRunManifest.model_fields
                }
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise HistoricalBacktestResultValidationError(
                "NONCANONICAL_SOURCE_RUN", "run manifest is not canonical"
            ) from error
        if manifest != run_manifest:
            raise HistoricalBacktestResultValidationError(
                "NONCANONICAL_SOURCE_RUN", "run manifest must already be canonical"
            )

    validated_snapshots = []
    for session in canonical_sessions:
        try:
            snapshot = validate_ranking_snapshot(session.ranking_snapshot)
        except CandidateRankingValidationError as error:
            raise HistoricalBacktestResultValidationError(
                "NONCANONICAL_SOURCE_RUN",
                "ranking snapshot fails authoritative Phase 14 validation",
            ) from error
        validated_snapshots.append(snapshot)

    if manifest is not None and any(
        snapshot.policy != manifest.ranking_policy_ref
        for snapshot in validated_snapshots
    ):
        raise HistoricalBacktestResultValidationError(
            "RANKING_POLICY_MISMATCH",
            "ranking policy does not match the run manifest",
        )

    for session in canonical_sessions:
        try:
            _validate_allocation(session, manifest=manifest)
        except HistoricalBacktestResultValidationError:
            raise
        except Exception as error:
            raise HistoricalBacktestResultValidationError(
                "ARTIFACT_PROVENANCE_MISMATCH",
                "allocation provenance is not canonical",
            ) from error

    _validate_cross_session_entry_intent_carry(tuple(canonical_sessions))

    if manifest is not None:
        _validate_cost_policy(run, manifest)
        # Current signal/entry/exit decisions contain earnings decisions but do
        # not bind provider name, published build_id, or output_sha256. There is
        # therefore no independent earnings-artifact identity to compare.

    execution_links_list: list[tuple[_ExecutionLink, ...]] = []
    for session, state_before in zip(canonical_sessions, states_before, strict=True):
        try:
            execution_links_list.append(
                _link_executions(session, state_before=state_before)
            )
        except AssertionError as error:
            # `_link_executions` defensively asserts that an EXECUTED Phase 9
            # entry decision carries its execution cost quote before matching
            # it to a BUY event; convert that into the same stable code it
            # itself raises one line later for the identical entry-linkage
            # concern, rather than letting a raw AssertionError escape.
            raise HistoricalBacktestResultValidationError(
                "ENTRY_LINKAGE_MISMATCH",
                "executed Phase 9 decision lacks its required execution cost quote",
            ) from error
    execution_links = tuple(execution_links_list)
    return _ValidatedSourceRunContext(
        run_result=run,
        run_manifest=manifest,
        states_before=tuple(states_before),
        execution_links=execution_links,
    )


def validate_historical_backtest_source_run(
    *,
    run_result: HistoricalBacktestRunResult,
    run_manifest: HistoricalBacktestRunManifest,
) -> None:
    """Validate one source run and manifest for deterministic projection."""

    _build_validated_source_context(
        run_result=run_result,
        run_manifest=run_manifest,
    )


__all__ = ["validate_historical_backtest_source_run"]
