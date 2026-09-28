"""Deterministic Phase 15A single-session and run chronology."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import date
from decimal import Decimal

from pydantic import ValidationError

from stock_swing_d1.execution.entry import (
    EntryExecutionDecision,
    EntryExecutionStatus,
)
from stock_swing_d1.execution.costs import (
    ExecutionCostPolicyRef,
    ExecutionCostValidationError,
)
from stock_swing_d1.execution.open_position_exit import (
    ExitPrerequisiteStatus,
    OpenPositionExitDecision,
    OpenPositionExitEvaluationInput,
    OpenPositionExitReason,
)
from stock_swing_d1.execution.protective_exit import (
    ProtectiveExitState,
    ProtectiveExitValidationError,
)
from stock_swing_d1.indicators import (
    TechnicalIndicatorRow,
    calculate_indicators,
)
from stock_swing_d1.portfolio import (
    PortfolioAllocationService,
    PortfolioCandidate,
    PortfolioCandidateAction,
    PortfolioReservationSettlement,
    RankedPortfolioAllocationDecision,
)
from stock_swing_d1.portfolio.models import (
    OpenPosition as AllocationOpenPosition,
    PortfolioAllocationValidationError,
    PortfolioSnapshot,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioEventKind,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_invariants import (
    PortfolioInvariantChecker,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioState,
    PortfolioTransitionResult,
)
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from stock_swing_d1.ranking import (
    CandidateRankingSnapshot,
    CandidateRankingService,
    RankingCandidate,
    RankedAllocationAdapter,
    RankedAllocationBatch,
    validate_ranking_snapshot,
)
from stock_swing_d1.ranking.hashing import (
    compute_candidate_input_fingerprint,
)
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CanonicalDividendAccountingEvidence,
)
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    OrdinaryDividendAccountingPolicyRef,
    ProcessedSessionContiguityProof,
)

from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.valuation import (
    PortfolioValuationError,
    PortfolioValuationMark,
    PortfolioValuationPolicy,
    PortfolioValuationResult,
    value_portfolio_at_allocation_boundary,
)

from stock_swing_d1.backtester.models import (
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
    DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION,
    HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
    HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION,
    HistoricalBacktestEntryIntent,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionInput,
    HistoricalBacktestSessionPlan,
    HistoricalBacktestSessionResult,
    HistoricalDecisionInterval,
)
from stock_swing_d1.backtester.session_input_provider import (
    HistoricalBacktestAllocationMarkContext,
    HistoricalBacktestSessionInputProvider,
    build_session_context,
)
from stock_swing_d1.backtester.execution_adapters import (
    BacktestBuyExecutionEventAdapter,
    BacktestExecutionAdapterValidationError,
    BacktestSellExecutionEventAdapter,
    TransientEntryExposure,
)
from stock_swing_d1.backtester.validation import (
    HistoricalBacktestValidationError,
    ensure_admitted_decision,
    reconcile_open_position_exit_coverage,
    reconcile_protective_custody,
    validate_allocation_boundary_marks,
    validate_allocation_portfolio_against_state,
    validate_decision_interval_containment,
    validate_dividend_aware_run_evidence,
    validate_initial_protective_states,
    validate_scheduled_entry_intents,
    validate_session_contract,
    validate_session_sequence,
    validate_session_state_inputs,
)


_PROTECTIVE_EXIT_REASONS = frozenset(
    {
        OpenPositionExitReason.GAP_THROUGH_STOP,
        OpenPositionExitReason.GAP_THROUGH_TARGET,
        OpenPositionExitReason.STOP_LOSS,
        OpenPositionExitReason.TAKE_PROFIT,
    }
)


def _same_corporate_action_audit(
    actual: tuple[object, ...],
    supplied: tuple[object, ...],
) -> bool:
    """Compare exact action payloads without imposing caller tuple order."""

    unmatched = list(actual)
    for expected in supplied:
        try:
            index = unmatched.index(expected)
        except ValueError:
            return False
        del unmatched[index]
    return not unmatched


def _require_operation(
    dependency: object,
    operation: str,
    *,
    code: str,
) -> object:
    if not callable(getattr(dependency, operation, None)):
        raise HistoricalBacktestValidationError(
            code,
            f"dependency must provide callable {operation}",
        )
    return dependency


class _TechnicalIndicatorService:
    """Thin facade over the frozen Phase 7 function for dependency recording."""

    @staticmethod
    def calculate_indicators(*, bars: tuple[object, ...]) -> object:
        return calculate_indicators(bars)


# Task 5C-A: Phase 15A never re-derives the equity formula; it calls the one
# shared valuation owner with the frozen policy and projects the authoritative
# Decimal result forward to float exactly once.
_FROZEN_VALUATION_POLICY = PortfolioValuationPolicy()


def _call_provider_operation(
    provider: object,
    operation: str,
    **kwargs: object,
) -> object:
    """Invoke one narrow provider operation exactly once, fail-closed.

    A provider is an external fact source, never an economic owner: a missing
    operation or any exception it raises fails the run immediately.  There is
    no retry, so a failed session can never be partially published.
    """

    bound = getattr(provider, operation, None)
    if not callable(bound):
        raise HistoricalBacktestValidationError(
            "INVALID_SESSION_INPUT_PROVIDER",
            f"the session input provider must expose callable {operation}",
        )
    try:
        return bound(**kwargs)
    except HistoricalBacktestValidationError:
        raise
    except Exception as error:  # noqa: BLE001 - fail closed on any provider fault
        raise HistoricalBacktestValidationError(
            "SESSION_INPUT_PROVIDER_FAILED",
            f"the session input provider raised during {operation}",
        ) from error


@dataclass(frozen=True, slots=True)
class _AllocationBoundaryMarkResolver:
    """The second provider boundary: canonical marks for authoritative S[T].

    Invoked exactly once per allocating session that holds at least one open
    position, never before S[T] exists, never prefetched and never retried.
    A session whose authoritative state holds no position uses the canonical
    empty mark tuple without calling the provider at all.
    """

    provider: object
    market_data_artifact_ref: ArtifactRef

    def marks_for(
        self, *, session: date, state: PortfolioState
    ) -> tuple[PortfolioValuationMark, ...]:
        security_ids = tuple(
            sorted(position.asset_id for position in state.open_positions)
        )
        if not security_ids:
            return ()
        context = HistoricalBacktestAllocationMarkContext(
            session=session,
            security_ids=security_ids,
            expected_market_data_artifact_ref=self.market_data_artifact_ref,
        )
        marks_result = _call_provider_operation(
            self.provider, "allocation_boundary_marks", context=context
        )
        return validate_allocation_boundary_marks(
            marks_result=marks_result,
            session=session,
            security_ids=security_ids,
            market_data_artifact_ref=self.market_data_artifact_ref,
        )


def _static_session_input(
    plan: object,
) -> HistoricalBacktestSessionInput:
    """Project one session-static plan into its session-input half.

    Every state-dependent field stays empty here: entry intents, entry bars,
    exit evaluations, ranking/allocation candidates, the allocation portfolio
    and ``scheduled_execution_events`` are all absent, so the provider path
    has no external economic injection seam and the run-level pre-flight can
    run in full before the first transition.
    """

    if type(plan) is not HistoricalBacktestSessionPlan:
        raise HistoricalBacktestValidationError(
            "INVALID_SESSION_PLAN",
            "session plans must be HistoricalBacktestSessionPlan values",
        )
    dividend_aware = (
        plan.distribution_events is not None
        or plan.distribution_coverage is not None
    )
    try:
        return HistoricalBacktestSessionInput(
            schema_version=(
                DIVIDEND_AWARE_HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION
                if dividend_aware
                else HISTORICAL_BACKTEST_SESSION_INPUT_SCHEMA_VERSION
            ),
            session=plan.session,
            decision_time=plan.decision_time,
            next_session=plan.next_session,
            distribution_events=plan.distribution_events,
            distribution_coverage=plan.distribution_coverage,
            pre_open_corporate_actions=plan.pre_open_corporate_actions,
            indicator_history_bars=plan.indicator_history_bars,
            completed_unadjusted_bars=plan.completed_unadjusted_bars,
            universe_eligible_security_ids=(
                plan.universe_eligible_security_ids
            ),
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise HistoricalBacktestValidationError(
            "INVALID_SESSION_PLAN",
            "session plan does not project a valid static session input",
        ) from error


def _value_allocation_boundary(
    *,
    state: PortfolioState,
    session: date,
    marks: tuple[PortfolioValuationMark, ...],
) -> PortfolioValuationResult:
    try:
        return value_portfolio_at_allocation_boundary(
            state=state,
            session=session,
            marks=marks,
            policy=_FROZEN_VALUATION_POLICY,
        )
    except PortfolioValuationError as error:
        raise HistoricalBacktestValidationError(
            "INVALID_ALLOCATION_BOUNDARY_VALUATION",
            "the shared allocation-boundary valuation rejected session T",
        ) from error


def _project_allocation_portfolio(
    *,
    session: date,
    decision_time: object,
    state: PortfolioState,
    valuation: PortfolioValuationResult,
) -> PortfolioSnapshot:
    """Project authoritative Decimal economics into the Phase 12 snapshot.

    This is the single prescribed ``Decimal -> float`` projection: pending
    settlements are inside ``portfolio_equity`` and outside ``cash_available``,
    and both floats originate from ``float(authoritative Decimal)``.  A
    non-finite or non-positive projection fails closed inside the frozen
    ``PortfolioSnapshot`` contract; nothing is clamped or rounded.
    """

    try:
        return PortfolioSnapshot(
            allocation_session=session,
            decision_time=decision_time,
            portfolio_equity=float(valuation.portfolio_equity),
            cash_available=float(state.settled_cash),
            open_positions=tuple(
                AllocationOpenPosition(
                    security_id=position.asset_id,
                    # Phase 13 tracks no ticker: `security_id` is the canonical
                    # identity and a symbol is display metadata only, so the
                    # canonical ID is carried through rather than a ticker
                    # being invented.  The authoritative identity check
                    # compares security_id, shares and entry_session only.
                    symbol=position.asset_id,
                    shares=position.quantity,
                    entry_session=position.entry_session,
                )
                for position in sorted(
                    state.open_positions, key=lambda item: item.asset_id
                )
            ),
        )
    except (PortfolioAllocationValidationError, TypeError, ValueError) as error:
        raise HistoricalBacktestValidationError(
            "INVALID_PROJECTED_ALLOCATION_PORTFOLIO",
            "the projected Phase 12 snapshot is not a valid allocation "
            "boundary for authoritative session T",
        ) from error


def execution_presentation_rank(
    event: PortfolioExecutionEvent,
    *,
    session_start_asset_ids: frozenset[str],
    replayed_execution_ids: frozenset[str],
) -> int:
    """Task 5C-C combined execution-presentation rank.

    rank 0: REPLAYED SELL presentations and APPLIED PRIOR SELLs;
    rank 1: REPLAYED BUY presentations and APPLIED BUYs;
    rank 2: APPLIED ROUND_TRIP SELLs (asset absent from session-start P).

    This is a presentation order over the evidence Phase 15A submits and
    preserves; it is never the economic application classification, which
    Phase 13 derives itself from P and the batch's applied BUY set.
    """

    if event.side is ExecutionSide.BUY:
        return 1
    if (
        event.execution_id in replayed_execution_ids
        or event.asset_id in session_start_asset_ids
    ):
        return 0
    return 2


def _replayed_execution_ids(
    prior_state: PortfolioState,
    events: tuple[PortfolioExecutionEvent, ...],
) -> frozenset[str]:
    """Execution IDs already applied in the prior state (historical replays)."""

    known = {
        fingerprint.event_id
        for fingerprint in prior_state.applied_events
        if fingerprint.event_kind is PortfolioEventKind.EXECUTION
    }
    return frozenset(
        event.execution_id for event in events if event.execution_id in known
    )


def _ordered_execution_events(
    events: tuple[PortfolioExecutionEvent, ...],
    *,
    prior_state: PortfolioState,
) -> tuple[PortfolioExecutionEvent, ...]:
    session_start_asset_ids = frozenset(
        position.asset_id for position in prior_state.open_positions
    )
    replayed = _replayed_execution_ids(prior_state, events)
    return tuple(
        sorted(
            events,
            key=lambda event: (
                execution_presentation_rank(
                    event,
                    session_start_asset_ids=session_start_asset_ids,
                    replayed_execution_ids=replayed,
                ),
                event.asset_id,
                event.execution_id,
            ),
        )
    )


def _validate_entry_execution_decision(
    *,
    decision: object,
    intent: HistoricalBacktestEntryIntent,
    session: date,
) -> EntryExecutionDecision:
    if not isinstance(decision, EntryExecutionDecision):
        raise HistoricalBacktestValidationError(
            "INVALID_ENTRY_EXECUTION_RESULT",
            "the Phase 9 owner must return EntryExecutionDecision",
        )
    try:
        decision = EntryExecutionDecision(
            **{
                field.name: getattr(decision, field.name)
                for field in fields(EntryExecutionDecision)
            }
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise HistoricalBacktestValidationError(
            "INVALID_ENTRY_EXECUTION_RESULT",
            "Phase 9 returned a noncanonical entry execution decision",
        ) from error
    sized = intent.candidate_decision.sized_pending_entry
    if sized is None:
        raise HistoricalBacktestValidationError(
            "INVALID_SCHEDULED_ENTRY",
            "scheduled entry lacks its authoritative fixed quantity",
        )
    if (
        decision.security_id != intent.security_id
        or decision.planned_entry_session != session
        or decision.planned_entry_session != intent.planned_entry_session
        or decision.requested_shares != sized.fixed_shares
        or decision.signal_session != intent.allocation_session
    ):
        raise HistoricalBacktestValidationError(
            "INVALID_ENTRY_EXECUTION_RESULT",
            "Phase 9 result changed entry identity, session, or fixed quantity",
        )
    if decision.status is EntryExecutionStatus.EXECUTED and (
        decision.executed_shares != sized.fixed_shares
        or decision.execution_price is None
        or decision.execution_cost_quote is None
    ):
        raise HistoricalBacktestValidationError(
            "INVALID_ENTRY_EXECUTION_RESULT",
            "an executed Phase 9 result must preserve fixed quantity, price, and cost",
        )
    return decision


def _validate_transition_result(
    *,
    result: object,
    prior_state: PortfolioState,
    session: date,
) -> PortfolioTransitionResult:
    if not isinstance(result, PortfolioTransitionResult):
        raise HistoricalBacktestValidationError(
            "INVALID_PHASE13_TRANSITION_RESULT",
            "Phase 13 must return PortfolioTransitionResult",
        )
    before = hash_portfolio_state(prior_state)
    if (
        result.session != session
        or result.state_hash_before != before
        or result.resulting_state.as_of_session != session
        or result.state_hash_after != hash_portfolio_state(result.resulting_state)
    ):
        raise HistoricalBacktestValidationError(
            "INVALID_PHASE13_TRANSITION_RESULT",
            "Phase 13 result does not preserve the requested state/session linkage",
        )
    PortfolioInvariantChecker.validate_state(result.resulting_state)
    return result


def _future_entry_intents(
    *,
    session_input: HistoricalBacktestSessionInput,
    ranked_batch: RankedAllocationBatch,
    allocation: RankedPortfolioAllocationDecision,
) -> tuple[HistoricalBacktestEntryIntent, ...]:
    if len(ranked_batch.candidates) != len(allocation.candidate_decisions):
        raise HistoricalBacktestValidationError(
            "INVALID_RANKED_ALLOCATION_RESULT",
            "Phase 12 result must preserve the complete ranked candidate batch",
        )

    intents: list[HistoricalBacktestEntryIntent] = []
    for ranked, decision in zip(
        ranked_batch.candidates,
        allocation.candidate_decisions,
        strict=True,
    ):
        if (
            ranked.security_id != decision.security_id
            or decision.processing_rank != ranked.rank
            or getattr(decision, "source_rank", None) != ranked.rank
        ):
            raise HistoricalBacktestValidationError(
                "INVALID_RANKED_ALLOCATION_RESULT",
                "Phase 12 result changed rank order or candidate identity",
            )
        if decision.action is not PortfolioCandidateAction.ADMITTED:
            continue
        ensure_admitted_decision(decision.action)
        sized = decision.sized_pending_entry
        if sized is None:
            raise HistoricalBacktestValidationError(
                "INVALID_FUTURE_ENTRY_INTENT",
                "admitted Phase 12 decision lacks its fixed quantity",
            )
        if (
            session_input.next_session is not None
            and sized.planned_entry_session != session_input.next_session
        ):
            raise HistoricalBacktestValidationError(
                "PLANNED_ENTRY_SESSION_MISMATCH",
                "admitted entry is not scheduled for the canonical next session",
            )
        intents.append(
            HistoricalBacktestEntryIntent(
                allocation_session=session_input.session,
                planned_entry_session=sized.planned_entry_session,
                security_id=ranked.security_id,
                source_rank=ranked.rank,
                ranking_snapshot_fingerprint=(
                    ranked_batch.ranking_snapshot_fingerprint
                ),
                allocation_candidate=ranked.allocation_candidate,
                candidate_decision=decision,
            )
        )
    return tuple(intents)


@dataclass(frozen=True, slots=True, init=False, repr=False)
class HistoricalBacktestOrchestrator:
    """Coordinate frozen owners around one authoritative Phase 13 call per T."""

    _transition_service: object
    _ranking_service: object
    _ranked_allocation_adapter: object
    _allocation_service: object
    _indicator_service: object
    _baseline_signal_evaluator: object | None
    _entry_execution_service: object | None
    _entry_event_adapter: object | None
    _open_position_exit_evaluator: object | None
    _open_position_exit_event_adapter: object | None
    _protective_exit_state_factory: object | None

    def __init__(
        self,
        *,
        transition_service: object | None = None,
        ranking_service: object | None = None,
        ranked_allocation_adapter: object | None = None,
        allocation_service: object | None = None,
        indicator_service: object | None = None,
        baseline_signal_evaluator: object | None = None,
        entry_execution_service: object | None = None,
        entry_event_adapter: object | None = None,
        open_position_exit_evaluator: object | None = None,
        open_position_exit_event_adapter: object | None = None,
        protective_exit_state_factory: object | None = None,
    ) -> None:
        transition_dependency = (
            PortfolioTransitionEngine
            if transition_service is None
            else transition_service
        )
        ranking_dependency = (
            CandidateRankingService
            if ranking_service is None
            else ranking_service
        )
        adapter_dependency = (
            RankedAllocationAdapter
            if ranked_allocation_adapter is None
            else ranked_allocation_adapter
        )
        allocation_dependency = (
            PortfolioAllocationService()
            if allocation_service is None
            else allocation_service
        )
        indicator_dependency = (
            _TechnicalIndicatorService
            if indicator_service is None
            else indicator_service
        )
        object.__setattr__(
            self,
            "_transition_service",
            _require_operation(
                transition_dependency,
                "transition",
                code="INVALID_TRANSITION_SERVICE",
            ),
        )
        object.__setattr__(
            self,
            "_ranking_service",
            _require_operation(
                ranking_dependency,
                "rank_candidates",
                code="INVALID_RANKING_SERVICE",
            ),
        )
        object.__setattr__(
            self,
            "_ranked_allocation_adapter",
            _require_operation(
                adapter_dependency,
                "build_batch",
                code="INVALID_RANKED_ALLOCATION_ADAPTER",
            ),
        )
        object.__setattr__(
            self,
            "_allocation_service",
            _require_operation(
                allocation_dependency,
                "allocate_ranked_candidates",
                code="INVALID_ALLOCATION_SERVICE",
            ),
        )
        object.__setattr__(
            self,
            "_indicator_service",
            _require_operation(
                indicator_dependency,
                "calculate_indicators",
                code="INVALID_INDICATOR_SERVICE",
            ),
        )
        if baseline_signal_evaluator is not None:
            baseline_signal_evaluator = _require_operation(
                baseline_signal_evaluator,
                "evaluate_signal",
                code="INVALID_BASELINE_SIGNAL_EVALUATOR",
            )
        object.__setattr__(
            self,
            "_baseline_signal_evaluator",
            baseline_signal_evaluator,
        )
        if entry_execution_service is not None:
            entry_execution_service = _require_operation(
                entry_execution_service,
                "execute_pending_entry",
                code="INVALID_ENTRY_EXECUTION_SERVICE",
            )
        if entry_event_adapter is not None:
            entry_event_adapter = _require_operation(
                entry_event_adapter,
                "build_buy_event",
                code="INVALID_ENTRY_EVENT_ADAPTER",
            )
        if open_position_exit_evaluator is not None:
            open_position_exit_evaluator = _require_operation(
                open_position_exit_evaluator,
                "evaluate",
                code="INVALID_OPEN_POSITION_EXIT_EVALUATOR",
            )
        if open_position_exit_event_adapter is not None:
            open_position_exit_event_adapter = _require_operation(
                open_position_exit_event_adapter,
                "build_sell_event",
                code="INVALID_OPEN_POSITION_EXIT_EVENT_ADAPTER",
            )
        if isinstance(
            entry_event_adapter, BacktestBuyExecutionEventAdapter
        ) and isinstance(
            open_position_exit_event_adapter,
            BacktestSellExecutionEventAdapter,
        ):
            try:
                buy_policy_ref = ExecutionCostPolicyRef(
                    **{
                        field.name: getattr(
                            entry_event_adapter.policy_ref, field.name
                        )
                        for field in fields(ExecutionCostPolicyRef)
                    }
                )
                sell_policy_ref = ExecutionCostPolicyRef(
                    **{
                        field.name: getattr(
                            open_position_exit_event_adapter.policy_ref,
                            field.name,
                        )
                        for field in fields(ExecutionCostPolicyRef)
                    }
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise HistoricalBacktestValidationError(
                    "EXECUTION_COST_POLICY_MISMATCH",
                    "production BUY and SELL adapters must use the same immutable "
                    "Phase 15C execution-cost policy",
                ) from error
            if buy_policy_ref != sell_policy_ref:
                raise HistoricalBacktestValidationError(
                    "EXECUTION_COST_POLICY_MISMATCH",
                    "production BUY and SELL adapters must use the same immutable "
                    "Phase 15C execution-cost policy",
                )
        object.__setattr__(
            self, "_entry_execution_service", entry_execution_service
        )
        object.__setattr__(self, "_entry_event_adapter", entry_event_adapter)
        object.__setattr__(
            self,
            "_open_position_exit_evaluator",
            open_position_exit_evaluator,
        )
        object.__setattr__(
            self,
            "_open_position_exit_event_adapter",
            open_position_exit_event_adapter,
        )
        # Task 5C: Phase 15A becomes the chronological custodian of Phase 10
        # protective state, never its economic owner, so creation is delegated
        # to the frozen Phase 10 capability injected here.
        if protective_exit_state_factory is not None:
            protective_exit_state_factory = _require_operation(
                protective_exit_state_factory,
                "create_state",
                code="INVALID_PROTECTIVE_EXIT_STATE_FACTORY",
            )
        object.__setattr__(
            self,
            "_protective_exit_state_factory",
            protective_exit_state_factory,
        )

    def _evaluate_open_position_exits(
        self,
        *,
        prior_state: PortfolioState,
        session_input: HistoricalBacktestSessionInput,
    ) -> tuple[
        tuple[OpenPositionExitDecision, ...],
        tuple[PortfolioExecutionEvent, ...],
    ]:
        covered_evaluations = reconcile_open_position_exit_coverage(
            prior_state=prior_state,
            session_input=session_input,
        )
        if not covered_evaluations:
            return (), ()
        if self._open_position_exit_evaluator is None:
            raise HistoricalBacktestValidationError(
                "MISSING_OPEN_POSITION_EXIT_EVALUATOR",
                "prior open positions require the configured Phase 15B owner",
            )

        decisions: list[OpenPositionExitDecision] = []
        sell_events: list[PortfolioExecutionEvent] = []
        for position, evaluation in covered_evaluations:
            decision_raw = self._open_position_exit_evaluator.evaluate(evaluation)
            if not isinstance(decision_raw, OpenPositionExitDecision):
                raise HistoricalBacktestValidationError(
                    "INVALID_OPEN_POSITION_EXIT_RESULT",
                    "Phase 15B must return OpenPositionExitDecision",
                )
            try:
                decision = OpenPositionExitDecision(
                    **{
                        field.name: getattr(decision_raw, field.name)
                        for field in fields(OpenPositionExitDecision)
                    }
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise HistoricalBacktestValidationError(
                    "INVALID_OPEN_POSITION_EXIT_RESULT",
                    "Phase 15B returned a noncanonical exit decision",
                ) from error
            protective = decision.protective_exit_decision
            if (
                decision.security_id != position.asset_id
                or decision.security_id
                != evaluation.protective_state.security_id
                or decision.session != session_input.session
                or decision.session != evaluation.session
                or decision.entry_session != position.entry_session
                or decision.entry_session
                != evaluation.protective_state.entry_session
                or decision.market_bar != evaluation.market_bar
                or decision.earnings_decision != evaluation.earnings_decision
                or decision.prior_boundary_earnings_decision
                != evaluation.prior_boundary_earnings_decision
                or protective.security_id
                != evaluation.protective_state.security_id
                or protective.session != evaluation.session
                or protective.entry_session
                != evaluation.protective_state.entry_session
                or protective.entry_price
                != evaluation.protective_state.entry_price
                or not _same_corporate_action_audit(
                    protective.corporate_actions,
                    evaluation.corporate_actions,
                )
            ):
                raise HistoricalBacktestValidationError(
                    "INVALID_OPEN_POSITION_EXIT_RESULT",
                    "Phase 15B result does not belong to the supplied evaluation",
                )
            decisions.append(decision)
            if not decision.exit_required:
                continue
            if decision.exit_prerequisite_status is not ExitPrerequisiteStatus.READY:
                raise HistoricalBacktestValidationError(
                    "UNSUPPORTED_OPEN_POSITION_EXIT_PREREQUISITE",
                    "the Phase 15B terminal exit is not persistence-ready",
                )
            if self._open_position_exit_event_adapter is None:
                raise HistoricalBacktestValidationError(
                    "MISSING_OPEN_POSITION_EXIT_EVENT_ADAPTER",
                    "a terminal Phase 15B decision needs authoritative "
                    "execution, cost, IDs, and settlement metadata",
                )
            try:
                event_raw = (
                    self._open_position_exit_event_adapter.build_sell_event(
                        decision=decision,
                        open_position=position,
                    )
                )
            except (
                BacktestExecutionAdapterValidationError,
                ExecutionCostValidationError,
            ) as error:
                raise HistoricalBacktestValidationError(
                    "OPEN_POSITION_EXIT_EVENT_ADAPTER_FAILED",
                    "the production SELL event adapter rejected the exit",
                ) from error
            if not isinstance(event_raw, PortfolioExecutionEvent):
                raise HistoricalBacktestValidationError(
                    "INVALID_OPEN_POSITION_EXIT_EVENT_ADAPTER_RESULT",
                    "exit event adapter must return one PortfolioExecutionEvent",
                )
            try:
                event = PortfolioExecutionEvent.model_validate(
                    event_raw.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise HistoricalBacktestValidationError(
                    "INVALID_OPEN_POSITION_EXIT_EVENT_ADAPTER_RESULT",
                    "adapted SELL must be a complete canonical Phase 13 event",
                ) from error
            if event != event_raw or (
                event.side is not ExecutionSide.SELL
                or event.session != session_input.session
                or event.asset_id != position.asset_id
                or event.quantity != position.quantity
            ):
                raise HistoricalBacktestValidationError(
                    "INVALID_OPEN_POSITION_EXIT_EVENT_ADAPTER_RESULT",
                    "adapted SELL changed side, session, identity, or full quantity",
                )
            if decision.selected_reason in _PROTECTIVE_EXIT_REASONS:
                final_price = decision.final_execution_price
                if final_price is None or event.fill_price != Decimal(
                    str(final_price)
                ):
                    raise HistoricalBacktestValidationError(
                        "PROTECTIVE_EXIT_PRICE_MISMATCH",
                        "adapted protective SELL must preserve Phase 10 final "
                        "execution price",
                    )
            sell_events.append(event)
        return tuple(decisions), tuple(sell_events)

    def _prepare_completed_candidate_batch(
        self,
        *,
        session_input: HistoricalBacktestSessionInput,
        effective_next_session: date | None,
    ) -> tuple[
        tuple[TechnicalIndicatorRow, ...],
        tuple[BaselineSignalDecision, ...],
        tuple[RankingCandidate, ...],
        tuple[PortfolioCandidate, ...],
    ]:
        raw_mode = bool(
            session_input.indicator_history_bars
            or session_input.completed_unadjusted_bars
            or session_input.universe_eligible_security_ids
        )
        if not raw_mode:
            return (
                (),
                (),
                session_input.ranking_candidates,
                session_input.allocation_candidates,
            )
        if self._baseline_signal_evaluator is None:
            raise HistoricalBacktestValidationError(
                "MISSING_BASELINE_SIGNAL_EVALUATOR",
                "raw completed-T inputs require the configured Phase 8 owner",
            )
        create_pending_entry = getattr(
            self._entry_execution_service, "create_pending_entry", None
        )
        if not callable(create_pending_entry):
            raise HistoricalBacktestValidationError(
                "MISSING_PENDING_ENTRY_SERVICE",
                "valid Phase 8 signals require the configured Phase 9 owner",
            )

        indicator_rows_raw = self._indicator_service.calculate_indicators(
            bars=session_input.indicator_history_bars
        )
        try:
            all_indicator_rows = tuple(indicator_rows_raw)
        except TypeError as error:
            raise HistoricalBacktestValidationError(
                "INVALID_INDICATOR_RESULT",
                "Phase 7 must return a finite indicator-row collection",
            ) from error
        if any(
            not isinstance(row, TechnicalIndicatorRow)
            for row in all_indicator_rows
        ):
            raise HistoricalBacktestValidationError(
                "INVALID_INDICATOR_RESULT",
                "Phase 7 returned a non-TechnicalIndicatorRow value",
            )
        current_rows = tuple(
            sorted(
                (
                    row
                    for row in all_indicator_rows
                    if row.trading_date == session_input.session
                ),
                key=lambda row: row.security_id,
            )
        )
        rows_by_security_id = {row.security_id: row for row in current_rows}
        current_adjusted_bars = tuple(
            sorted(
                (
                    bar
                    for bar in session_input.indicator_history_bars
                    if bar.trading_date == session_input.session
                ),
                key=lambda bar: bar.security_id,
            )
        )
        if set(rows_by_security_id) != {
            bar.security_id for bar in current_adjusted_bars
        }:
            raise HistoricalBacktestValidationError(
                "INCOMPLETE_INDICATOR_RESULT",
                "Phase 7 must return one completed-T row per adjusted bar",
            )

        eligible_ids = set(session_input.universe_eligible_security_ids)
        unadjusted_by_security_id = {
            bar.security_id: bar
            for bar in session_input.completed_unadjusted_bars
        }
        signals: list[BaselineSignalDecision] = []
        ranking_candidates: list[RankingCandidate] = []
        allocation_candidates: list[PortfolioCandidate] = []
        for bar in current_adjusted_bars:
            signal_raw = self._baseline_signal_evaluator.evaluate_signal(
                bar=bar,
                indicators=rows_by_security_id[bar.security_id],
                universe_eligible=bar.security_id in eligible_ids,
            )
            if not isinstance(signal_raw, BaselineSignalDecision):
                raise HistoricalBacktestValidationError(
                    "INVALID_BASELINE_SIGNAL_RESULT",
                    "Phase 8 must return BaselineSignalDecision",
                )
            signal = signal_raw
            if (
                signal.signal_session != session_input.session
                or signal.signal_time > session_input.decision_time
                or signal.planned_entry_session != effective_next_session
                or signal.security_id != bar.security_id
            ):
                raise HistoricalBacktestValidationError(
                    "INVALID_BASELINE_SIGNAL_RESULT",
                    "Phase 8 changed the T boundary, identity, or next session",
                )
            signals.append(signal)
            if signal.action is not BaselineSignalAction.VALID_LONG_SIGNAL:
                continue
            unadjusted_bar = unadjusted_by_security_id.get(signal.security_id)
            if unadjusted_bar is None:
                raise HistoricalBacktestValidationError(
                    "MISSING_COMPLETED_UNADJUSTED_BAR",
                    "a valid signal requires its completed-T sizing bar",
                )
            pending_entry = create_pending_entry(signal=signal)
            allocation_candidate = PortfolioCandidate(
                signal=signal,
                pending_entry=pending_entry,
                signal_bar=unadjusted_bar,
            )
            if any(
                value is None
                for value in (
                    signal.sma_20,
                    signal.sma_50,
                    signal.atr_14,
                    signal.rsi_14,
                )
            ):
                raise HistoricalBacktestValidationError(
                    "INVALID_VALID_SIGNAL_INDICATORS",
                    "valid Phase 8 signals require complete ranking indicators",
                )
            ranking_values = {
                "security_id": signal.security_id,
                "ranking_session": session_input.session,
                "decision_time": session_input.decision_time,
                "signal_session": signal.signal_session,
                "signal_time": signal.signal_time,
                "sma20": signal.sma_20,
                "sma50": signal.sma_50,
                "atr14": signal.atr_14,
                "rsi14": signal.rsi_14,
            }
            ranking_candidate = RankingCandidate(
                **ranking_values,
                input_fingerprint=compute_candidate_input_fingerprint(
                    **ranking_values
                ),
            )
            ranking_candidates.append(ranking_candidate)
            allocation_candidates.append(allocation_candidate)
        return (
            current_rows,
            tuple(signals),
            tuple(ranking_candidates),
            tuple(allocation_candidates),
        )

    def _execute_scheduled_entries(
        self,
        *,
        session_input: HistoricalBacktestSessionInput,
        intents: tuple[HistoricalBacktestEntryIntent, ...],
    ) -> tuple[
        tuple[EntryExecutionDecision, ...],
        tuple[PortfolioReservationSettlement, ...],
        tuple[PortfolioExecutionEvent, ...],
    ]:
        bars_by_security_id = {
            bar.security_id: bar for bar in session_input.entry_execution_bars
        }
        validate_scheduled_entry_intents(
            session=session_input.session,
            intents=intents,
            execution_bar_security_ids=set(bars_by_security_id),
        )
        if not intents:
            return (), (), ()
        if self._entry_execution_service is None:
            raise HistoricalBacktestValidationError(
                "MISSING_ENTRY_EXECUTION_SERVICE",
                "scheduled entries require the configured frozen Phase 9 owner",
            )
        settle_reservation = getattr(
            self._allocation_service, "settle_reservation", None
        )
        if not callable(settle_reservation):
            raise HistoricalBacktestValidationError(
                "MISSING_RESERVATION_SETTLEMENT_SERVICE",
                "scheduled entries require the frozen Phase 12 settlement operation",
            )

        decisions: list[EntryExecutionDecision] = []
        settlements: list[PortfolioReservationSettlement] = []
        buy_events: list[PortfolioExecutionEvent] = []
        for intent in sorted(intents, key=lambda item: item.security_id):
            candidate = intent.allocation_candidate
            sized = intent.candidate_decision.sized_pending_entry
            assert sized is not None
            decision_raw = self._entry_execution_service.execute_pending_entry(
                pending_entry=candidate.pending_entry,
                sized_pending_entry=sized,
                execution_bar=bars_by_security_id.get(intent.security_id),
            )
            decision = _validate_entry_execution_decision(
                decision=decision_raw,
                intent=intent,
                session=session_input.session,
            )
            decisions.append(decision)
            settlement = settle_reservation(
                candidate_decision=intent.candidate_decision,
                entry_execution=decision,
            )
            if not isinstance(settlement, PortfolioReservationSettlement):
                raise HistoricalBacktestValidationError(
                    "INVALID_RESERVATION_SETTLEMENT_RESULT",
                    "Phase 12 must return PortfolioReservationSettlement",
                )
            settlements.append(settlement)

            if decision.status is not EntryExecutionStatus.EXECUTED:
                continue
            if self._entry_event_adapter is None:
                raise HistoricalBacktestValidationError(
                    "MISSING_ENTRY_EVENT_ADAPTER",
                    "an executed Phase 9 entry needs an authoritative "
                    "Phase 13 event adapter",
                )
            try:
                event_raw = self._entry_event_adapter.build_buy_event(
                    session=session_input.session,
                    intent=intent,
                    entry_execution=decision,
                )
            except (
                BacktestExecutionAdapterValidationError,
                ExecutionCostValidationError,
            ) as error:
                raise HistoricalBacktestValidationError(
                    "ENTRY_EVENT_ADAPTER_FAILED",
                    "the production BUY event adapter rejected the entry",
                ) from error
            if not isinstance(event_raw, PortfolioExecutionEvent):
                raise HistoricalBacktestValidationError(
                    "INVALID_ENTRY_EVENT_ADAPTER_RESULT",
                    "entry event adapter must return PortfolioExecutionEvent",
                )
            try:
                event = PortfolioExecutionEvent.model_validate(
                    event_raw.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise HistoricalBacktestValidationError(
                    "INVALID_ENTRY_EVENT_ADAPTER_RESULT",
                    "adapted BUY must be a complete canonical Phase 13 event",
                ) from error
            if (
                event != event_raw
                or event.side is not ExecutionSide.BUY
                or event.session != session_input.session
                or event.asset_id != intent.security_id
                or event.quantity != decision.executed_shares
                or event.fill_price != Decimal(str(decision.execution_price))
                or event.execution_cost
                != decision.execution_cost_quote.execution_cost
                or event.settlement_id is not None
                or event.settlement_session is not None
            ):
                raise HistoricalBacktestValidationError(
                    "INVALID_ENTRY_EVENT_ADAPTER_RESULT",
                    "adapted BUY changed provenance, fixed quantity, or settlement",
                )
            buy_events.append(event)
        return tuple(decisions), tuple(settlements), tuple(buy_events)

    def _evaluate_entry_session_protection(
        self,
        *,
        session_input: HistoricalBacktestSessionInput,
        intents: tuple[HistoricalBacktestEntryIntent, ...],
        entry_execution_decisions: tuple[EntryExecutionDecision, ...],
        entry_execution_events: tuple[PortfolioExecutionEvent, ...],
    ) -> tuple[
        tuple[OpenPositionExitDecision, ...],
        tuple[PortfolioExecutionEvent, ...],
    ]:
        """Task 5C-B/5C-C: evaluate every entry executed on T against T's bar.

        For each Phase 9 ``EXECUTED`` entry, exactly once and in canonical
        security order: Phase 10 creates the protective state from the actual
        executed entry, Phase 15B evaluates it on the entry session itself,
        and the decision is retained as audit evidence.  A HOLD carries the
        exact Phase 10 ``resulting_state`` (evaluated on T) into custody.  A
        terminal protective decision is adapted by Phase 15C into an ordinary
        round-trip SELL that Phase 13 applies after the BUY inside the same
        single transition.  Phase 15A performs no protective, pricing, cost,
        settlement or accounting arithmetic here.
        """

        session = session_input.session
        executed = sorted(
            (
                decision
                for decision in entry_execution_decisions
                if decision.status is EntryExecutionStatus.EXECUTED
            ),
            key=lambda decision: decision.security_id,
        )
        if not executed:
            return (), ()
        factory = self._protective_exit_state_factory
        if factory is None:
            raise HistoricalBacktestValidationError(
                "MISSING_PROTECTIVE_EXIT_STATE_FACTORY",
                "an executed entry requires the configured Phase 10 "
                "protective-state creation capability",
            )
        evaluator = self._open_position_exit_evaluator
        if evaluator is None:
            raise HistoricalBacktestValidationError(
                "MISSING_OPEN_POSITION_EXIT_EVALUATOR",
                "an executed entry requires the configured Phase 15B owner "
                "for its entry-session protective evaluation",
            )
        intents_by_security = {intent.security_id: intent for intent in intents}
        bars_by_security = {
            bar.security_id: bar for bar in session_input.entry_execution_bars
        }
        buy_events_by_security = {
            event.asset_id: event for event in entry_execution_events
        }

        decisions: list[OpenPositionExitDecision] = []
        round_trip_sells: list[PortfolioExecutionEvent] = []
        for entry_decision in executed:
            security_id = entry_decision.security_id
            intent = intents_by_security.get(security_id)
            execution_bar = bars_by_security.get(security_id)
            buy_event = buy_events_by_security.get(security_id)
            if intent is None or execution_bar is None or buy_event is None:
                raise HistoricalBacktestValidationError(
                    "PROTECTIVE_STATE_CREATION_FAILED",
                    "an executed entry lacks its carried intent, execution "
                    "bar, or authoritative BUY event",
                )
            try:
                new_state = factory.create_state(
                    signal=intent.allocation_candidate.signal,
                    entry_execution=entry_decision,
                )
            except ProtectiveExitValidationError as error:
                raise HistoricalBacktestValidationError(
                    "PROTECTIVE_STATE_CREATION_FAILED",
                    "the frozen Phase 10 owner rejected this executed entry",
                ) from error
            if (
                not isinstance(new_state, ProtectiveExitState)
                or new_state.security_id != security_id
                or new_state.entry_session != session
                or new_state.last_evaluated_session is not None
            ):
                raise HistoricalBacktestValidationError(
                    "PROTECTIVE_STATE_CREATION_FAILED",
                    "Phase 10 must return an unevaluated ProtectiveExitState "
                    "for exactly this entry and session",
                )
            try:
                evaluation = OpenPositionExitEvaluationInput(
                    session=session,
                    protective_state=new_state,
                    market_bar=execution_bar,
                )
            except ValueError as error:
                raise HistoricalBacktestValidationError(
                    "INVALID_ENTRY_SESSION_PROTECTIVE_EVALUATION",
                    "the executed entry and its session-T bar do not compose "
                    "a valid Phase 15B evaluation",
                ) from error
            decision_raw = evaluator.evaluate(evaluation)
            if not isinstance(decision_raw, OpenPositionExitDecision):
                raise HistoricalBacktestValidationError(
                    "INVALID_ENTRY_SESSION_PROTECTIVE_RESULT",
                    "Phase 15B must return OpenPositionExitDecision",
                )
            try:
                decision = OpenPositionExitDecision(
                    **{
                        field.name: getattr(decision_raw, field.name)
                        for field in fields(OpenPositionExitDecision)
                    }
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise HistoricalBacktestValidationError(
                    "INVALID_ENTRY_SESSION_PROTECTIVE_RESULT",
                    "Phase 15B returned a noncanonical entry-session decision",
                ) from error
            protective = decision.protective_exit_decision
            if (
                decision.security_id != security_id
                or decision.session != session
                or decision.entry_session != session
                or decision.holding_session_number != 1
                or decision.market_bar != execution_bar
                or decision.exit_prerequisite_status
                is not ExitPrerequisiteStatus.READY
                or protective.security_id != security_id
                or protective.session != session
                or protective.entry_session != session
                or protective.entry_price != new_state.entry_price
                or Decimal(str(protective.entry_price)) != buy_event.fill_price
            ):
                raise HistoricalBacktestValidationError(
                    "INVALID_ENTRY_SESSION_PROTECTIVE_RESULT",
                    "Phase 15B result does not belong to the executed entry",
                )
            decisions.append(decision)
            if not decision.exit_required:
                resulting_state = protective.resulting_state
                if (
                    resulting_state is None
                    or resulting_state.last_evaluated_session != session
                ):
                    raise HistoricalBacktestValidationError(
                        "INVALID_ENTRY_SESSION_PROTECTIVE_RESULT",
                        "a HOLD on the entry session must advance the Phase 10 "
                        "state to that session",
                    )
                continue

            # Terminal on the entry session: Task 5C-C same-session round trip.
            adapter = self._open_position_exit_event_adapter
            if adapter is None:
                raise HistoricalBacktestValidationError(
                    "MISSING_OPEN_POSITION_EXIT_EVENT_ADAPTER",
                    "a terminal entry-session decision needs authoritative "
                    "execution, cost, IDs, and settlement metadata",
                )
            build = getattr(adapter, "build_round_trip_sell_event", None)
            if not callable(build):
                raise HistoricalBacktestValidationError(
                    "MISSING_ROUND_TRIP_SELL_CAPABILITY",
                    "the configured Phase 15C SELL adapter cannot adapt a "
                    "same-session round-trip protective exit",
                )
            try:
                exposure = TransientEntryExposure.from_buy_event(buy_event)
                event_raw = build(decision=decision, exposure=exposure)
            except (
                BacktestExecutionAdapterValidationError,
                ExecutionCostValidationError,
            ) as error:
                raise HistoricalBacktestValidationError(
                    "ROUND_TRIP_SELL_EVENT_ADAPTER_FAILED",
                    "the production SELL event adapter rejected the "
                    "same-session round-trip exit",
                ) from error
            if not isinstance(event_raw, PortfolioExecutionEvent):
                raise HistoricalBacktestValidationError(
                    "INVALID_ROUND_TRIP_SELL_EVENT_ADAPTER_RESULT",
                    "round-trip adapter must return one PortfolioExecutionEvent",
                )
            try:
                event = PortfolioExecutionEvent.model_validate(
                    event_raw.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise HistoricalBacktestValidationError(
                    "INVALID_ROUND_TRIP_SELL_EVENT_ADAPTER_RESULT",
                    "adapted round-trip SELL must be a canonical Phase 13 event",
                ) from error
            final_price = decision.final_execution_price
            if (
                event != event_raw
                or event.side is not ExecutionSide.SELL
                or event.session != session
                or event.asset_id != security_id
                or event.quantity != buy_event.quantity
                or event.execution_id == buy_event.execution_id
                or event.settlement_id is None
                or event.settlement_session is None
                or final_price is None
                or event.fill_price != Decimal(str(final_price))
            ):
                raise HistoricalBacktestValidationError(
                    "INVALID_ROUND_TRIP_SELL_EVENT_ADAPTER_RESULT",
                    "adapted round-trip SELL changed side, session, identity, "
                    "full quantity, settlement, or the Phase 10 final price",
                )
            round_trip_sells.append(event)
        return tuple(decisions), tuple(round_trip_sells)

    def _process_session(
        self,
        *,
        prior_state: PortfolioState,
        session_input: HistoricalBacktestSessionInput,
        effective_next_session: date | None = None,
        scheduled_entry_intents: (
            tuple[HistoricalBacktestEntryIntent, ...] | None
        ) = None,
        dividend_evidence: (
            tuple[CanonicalDividendAccountingEvidence, ...]
        ) = (),
        allocation_marks_resolver: (
            _AllocationBoundaryMarkResolver | None
        ) = None,
    ) -> HistoricalBacktestSessionResult:
        """Process T atomically, then create completed-T decision artifacts.

        ``allocation_marks_resolver`` is the state-aware provider seam: when
        present, the allocation portfolio is projected here from authoritative
        S[T] plus canonical marks instead of being supplied with the session
        input.  It is deliberately the only Task 5C addition to this frozen
        session body, so the legacy and provider paths share one chronology.
        """

        PortfolioInvariantChecker.validate_state(prior_state)
        resolved_next_session = (
            session_input.next_session
            if session_input.next_session is not None
            else effective_next_session
        )
        session_input = validate_session_contract(
            session_input,
            effective_next_session=resolved_next_session,
        )
        if (
            prior_state.as_of_session is not None
            and session_input.session <= prior_state.as_of_session
        ):
            raise HistoricalBacktestValidationError(
                "OUT_OF_ORDER_SESSION",
                "session T must be later than prior authoritative state",
            )

        prior_fingerprint = hash_portfolio_state(prior_state)
        (
            open_position_exit_decisions,
            open_position_exit_events,
        ) = self._evaluate_open_position_exits(
            prior_state=prior_state,
            session_input=session_input,
        )
        active_entry_intents = (
            session_input.scheduled_entry_intents
            if scheduled_entry_intents is None
            else scheduled_entry_intents
        )
        (
            entry_execution_decisions,
            reservation_settlements,
            entry_execution_events,
        ) = self._execute_scheduled_entries(
            session_input=session_input,
            intents=active_entry_intents,
        )
        # Task 5C-B/5C-C: the entry-session protective step lives here, in
        # the one shared session body, on every supported path.
        (
            entry_session_protective_decisions,
            round_trip_sell_events,
        ) = self._evaluate_entry_session_protection(
            session_input=session_input,
            intents=active_entry_intents,
            entry_execution_decisions=entry_execution_decisions,
            entry_execution_events=entry_execution_events,
        )
        ordered_events = _ordered_execution_events(
            session_input.scheduled_execution_events
            + open_position_exit_events
            + entry_execution_events
            + round_trip_sell_events,
            prior_state=prior_state,
        )
        sell_asset_ids = tuple(
            event.asset_id
            for event in ordered_events
            if event.side is ExecutionSide.SELL
        )
        if len(set(sell_asset_ids)) != len(sell_asset_ids):
            raise HistoricalBacktestValidationError(
                "DUPLICATE_TERMINAL_SELL",
                "one security can have at most one terminal SELL in session T",
            )
        ordered_execution_ids = tuple(
            event.execution_id for event in ordered_events
        )
        if len(set(ordered_execution_ids)) != len(ordered_execution_ids):
            raise HistoricalBacktestValidationError(
                "DUPLICATE_SCHEDULED_EXECUTION",
                "combined scheduled execution IDs must be unique within T",
            )

        # This is the only authoritative portfolio-state operation in T.
        transition_raw = self._transition_service.transition(
            prior_state,
            session_input.session,
            ordered_events,
            dividend_evidence=dividend_evidence,
        )
        transition = _validate_transition_result(
            result=transition_raw,
            prior_state=prior_state,
            session=session_input.session,
        )
        authoritative_state = transition.resulting_state

        # Completed-T research begins only after S[T] exists.
        (
            indicator_rows,
            signal_decisions,
            ranking_candidates,
            allocation_candidates,
        ) = self._prepare_completed_candidate_batch(
            session_input=session_input,
            effective_next_session=resolved_next_session,
        )
        # Phase 14 ranking must succeed before the allocation boundary is
        # touched: no mark-provider call, no valuation and no Phase 12 call
        # can happen if ranking fails.
        ranking_snapshot = self._ranking_service.rank_candidates(
            ranking_session=session_input.session,
            decision_time=session_input.decision_time,
            candidates=ranking_candidates,
        )
        if not isinstance(ranking_snapshot, CandidateRankingSnapshot):
            raise HistoricalBacktestValidationError(
                "INVALID_RANKING_RESULT",
                "Phase 14 must return CandidateRankingSnapshot",
            )
        ranking_snapshot = validate_ranking_snapshot(ranking_snapshot)
        if (
            ranking_snapshot.ranking_session != session_input.session
            or ranking_snapshot.decision_time != session_input.decision_time
        ):
            raise HistoricalBacktestValidationError(
                "INVALID_RANKING_RESULT",
                "Phase 14 result changed the completed-T decision boundary",
            )

        # The allocation boundary: S[T] exists, completed-T preparation has
        # decided whether Phase 12 runs at all and Phase 14 has succeeded, so
        # the second provider boundary and the shared valuation owner may be
        # consulted now -- and only now -- to project the Phase 12 snapshot.
        allocation_portfolio = session_input.allocation_portfolio
        authoritative_equity: Decimal | None = None
        if allocation_marks_resolver is not None and allocation_candidates:
            marks = allocation_marks_resolver.marks_for(
                session=session_input.session, state=authoritative_state
            )
            valuation = _value_allocation_boundary(
                state=authoritative_state,
                session=session_input.session,
                marks=marks,
            )
            authoritative_equity = valuation.portfolio_equity
            allocation_portfolio = _project_allocation_portfolio(
                session=session_input.session,
                decision_time=session_input.decision_time,
                state=authoritative_state,
                valuation=valuation,
            )

        prepared_input = session_input.model_copy(
            update={
                "indicator_history_bars": (),
                "completed_unadjusted_bars": (),
                "universe_eligible_security_ids": (),
                "ranking_candidates": ranking_candidates,
                "allocation_candidates": allocation_candidates,
                "allocation_portfolio": allocation_portfolio,
            }
        )
        validate_session_contract(
            prepared_input,
            effective_next_session=resolved_next_session,
        )

        ranked_batch: RankedAllocationBatch | None = None
        allocation: RankedPortfolioAllocationDecision | None = None
        future_intents: tuple[HistoricalBacktestEntryIntent, ...] = ()
        if prepared_input.allocation_portfolio is not None:
            validate_allocation_portfolio_against_state(
                session_input=prepared_input,
                authoritative_state=authoritative_state,
                authoritative_portfolio_equity=authoritative_equity,
            )
            ranked_batch = self._ranked_allocation_adapter.build_batch(
                ranking_snapshot=ranking_snapshot,
                allocation_candidates=allocation_candidates,
            )
            allocation_raw = self._allocation_service.allocate_ranked_candidates(
                portfolio=prepared_input.allocation_portfolio,
                ranked_batch=ranked_batch,
            )
            if not isinstance(
                allocation_raw, RankedPortfolioAllocationDecision
            ):
                raise HistoricalBacktestValidationError(
                    "INVALID_RANKED_ALLOCATION_RESULT",
                    "ranked Phase 12 path must return its ranked decision type",
                )
            allocation = allocation_raw
            future_intents = _future_entry_intents(
                session_input=prepared_input,
                ranked_batch=ranked_batch,
                allocation=allocation,
            )

        return HistoricalBacktestSessionResult(
            session=session_input.session,
            decision_time=session_input.decision_time,
            prior_state_fingerprint=prior_fingerprint,
            scheduled_entry_intents=active_entry_intents,
            entry_execution_decisions=entry_execution_decisions,
            reservation_settlements=reservation_settlements,
            open_position_exit_decisions=open_position_exit_decisions,
            entry_session_protective_decisions=(
                entry_session_protective_decisions
            ),
            ordered_execution_events=ordered_events,
            state_transition_result=transition,
            authoritative_state=authoritative_state,
            indicator_rows=indicator_rows,
            signal_decisions=signal_decisions,
            ranking_snapshot=ranking_snapshot,
            ranked_allocation_batch=ranked_batch,
            allocation_decision=allocation,
            future_entry_intents=future_intents,
        )

    def process_session(
        self,
        *,
        prior_state: PortfolioState,
        session_input: HistoricalBacktestSessionInput,
    ) -> HistoricalBacktestSessionResult:
        """Process one standalone T using only its explicit immutable input."""

        return self._process_session(
            prior_state=prior_state,
            session_input=session_input,
            effective_next_session=session_input.next_session,
            scheduled_entry_intents=session_input.scheduled_entry_intents,
        )

    def run(
        self,
        initial_state: PortfolioState,
        sessions: Sequence[HistoricalBacktestSessionInput],
        *,
        decision_interval: HistoricalDecisionInterval,
        dividend_accounting_policy_ref: (
            OrdinaryDividendAccountingPolicyRef | None
        ) = None,
        processed_session_contiguity_proof: (
            ProcessedSessionContiguityProof | None
        ) = None,
    ) -> HistoricalBacktestRunResult:
        """Thread one explicit authoritative state through the supplied sessions.

        The legacy fully pre-materialized entry point is unchanged in
        signature and observable semantics.  Task 5C removed its private copy
        of the run loop: it now validates the same run-level contract and
        delegates to the same authoritative loop the provider path uses.
        """

        session_inputs = validate_session_sequence(
            initial_state=initial_state,
            sessions=sessions,
        )
        (
            decision_interval,
            dividend_run_evidence,
            dividend_evidence_by_session,
        ) = self._validate_run_level_contract(
            session_inputs=session_inputs,
            decision_interval=decision_interval,
            dividend_accounting_policy_ref=dividend_accounting_policy_ref,
            processed_session_contiguity_proof=(
                processed_session_contiguity_proof
            ),
        )
        return self._run_sessions(
            initial_state=initial_state,
            session_inputs=session_inputs,
            decision_interval=decision_interval,
            dividend_run_evidence=dividend_run_evidence,
            dividend_evidence_by_session=dividend_evidence_by_session,
        )

    def run_with_session_input_provider(
        self,
        initial_state: PortfolioState,
        session_plans: Sequence[HistoricalBacktestSessionPlan],
        *,
        provider: HistoricalBacktestSessionInputProvider,
        decision_interval: HistoricalDecisionInterval,
        market_data_artifact_ref: ArtifactRef,
        initial_protective_states: tuple[ProtectiveExitState, ...] = (),
        dividend_accounting_policy_ref: (
            OrdinaryDividendAccountingPolicyRef | None
        ) = None,
        processed_session_contiguity_proof: (
            ProcessedSessionContiguityProof | None
        ) = None,
    ) -> HistoricalBacktestRunResult:
        """Run a state-aware session sequence from session-static plans.

        Two-level by construction: the run-level plan is pre-materialized and
        validated in full -- chronology, decision-interval containment, static
        session contract, dividend-accounting mode, processed-session
        contiguity, run-wide event uniqueness, entitlement/ex adjacency,
        per-session coverage, canonical snapshot binding and
        ``DividendAwareRunEvidence`` -- before the first Phase 13 transition;
        only genuinely state-dependent facts are retrieved lazily, per
        session, from the provider.

        ``market_data_artifact_ref`` is the run's authoritative market-data
        provenance.  It is supplied by the caller independently of the
        provider, is immutable for the run, and every returned valuation mark
        is checked against it: a provider may never self-authorize the
        artifact its marks claim to come from.
        """

        # Provider mode is explicit: a missing provider is a configuration
        # error, never a silent fall-back to the legacy path.  It fails before
        # any session processing, provider call, transition, valuation, Phase
        # 12 call or custody publication.
        if provider is None:
            raise HistoricalBacktestValidationError(
                "INVALID_SESSION_INPUT_PROVIDER",
                "run_with_session_input_provider requires an explicit "
                "session input provider; None is not a legacy fall-back",
            )
        if type(market_data_artifact_ref) is not ArtifactRef:
            raise HistoricalBacktestValidationError(
                "VALUATION_MARK_ARTIFACT_MISMATCH",
                "the run-level market-data artifact ref must be an ArtifactRef",
            )
        if isinstance(session_plans, (str, bytes)) or not isinstance(
            session_plans, Sequence
        ):
            raise HistoricalBacktestValidationError(
                "INVALID_SESSION_PLAN_SEQUENCE",
                "session plans must be a finite sequence",
            )
        session_inputs = validate_session_sequence(
            initial_state=initial_state,
            sessions=tuple(
                _static_session_input(plan) for plan in session_plans
            ),
        )
        (
            decision_interval,
            dividend_run_evidence,
            dividend_evidence_by_session,
        ) = self._validate_run_level_contract(
            session_inputs=session_inputs,
            decision_interval=decision_interval,
            dividend_accounting_policy_ref=dividend_accounting_policy_ref,
            processed_session_contiguity_proof=(
                processed_session_contiguity_proof
            ),
        )
        custody = validate_initial_protective_states(
            initial_state=initial_state,
            protective_states=initial_protective_states,
        )
        return self._run_sessions(
            initial_state=initial_state,
            session_inputs=session_inputs,
            decision_interval=decision_interval,
            dividend_run_evidence=dividend_run_evidence,
            dividend_evidence_by_session=dividend_evidence_by_session,
            provider=provider,
            market_data_artifact_ref=market_data_artifact_ref,
            initial_custody=custody,
        )

    def _validate_run_level_contract(
        self,
        *,
        session_inputs: tuple[HistoricalBacktestSessionInput, ...],
        decision_interval: HistoricalDecisionInterval,
        dividend_accounting_policy_ref: (
            OrdinaryDividendAccountingPolicyRef | None
        ),
        processed_session_contiguity_proof: (
            ProcessedSessionContiguityProof | None
        ),
    ) -> tuple[
        HistoricalDecisionInterval,
        object,
        dict[date, tuple[CanonicalDividendAccountingEvidence, ...]],
    ]:
        """Complete fail-fast run-level validation before any Phase 13 call.

        Frozen order: chronology -> containment -> session contract ->
        dividend-aware evidence/contiguity pre-flight -> orchestration.  Both
        entry points share it, so the provider path can never reach a
        transition with weaker run-level proof than the legacy path.
        """

        decision_interval = validate_decision_interval_containment(
            decision_interval=decision_interval,
            sessions=session_inputs,
        )
        for index, item in enumerate(session_inputs):
            following = (
                session_inputs[index + 1].session
                if index + 1 < len(session_inputs)
                else None
            )
            validate_session_contract(
                item,
                effective_next_session=item.next_session or following,
            )
        dividend_run_evidence = validate_dividend_aware_run_evidence(
            dividend_accounting_policy_ref=dividend_accounting_policy_ref,
            processed_session_contiguity_proof=(
                processed_session_contiguity_proof
            ),
            sessions=session_inputs,
        )
        # The validated per-session evidence already exists on the accepted
        # Slice-4 pre-flight result; build a deterministic session lookup
        # from it rather than re-deriving/re-normalizing anything. A
        # non-dividend-aware run (dividend_run_evidence is None) forwards an
        # empty tuple for every session, unchanged from before this slice.
        dividend_evidence_by_session: dict[
            date, tuple[CanonicalDividendAccountingEvidence, ...]
        ] = (
            {
                item.session: item.distribution_events
                for item in dividend_run_evidence.session_evidence
            }
            if dividend_run_evidence is not None
            else {}
        )
        return (
            decision_interval,
            dividend_run_evidence,
            dividend_evidence_by_session,
        )

    def _run_sessions(
        self,
        *,
        initial_state: PortfolioState,
        session_inputs: tuple[HistoricalBacktestSessionInput, ...],
        decision_interval: HistoricalDecisionInterval,
        dividend_run_evidence: object,
        dividend_evidence_by_session: dict[
            date, tuple[CanonicalDividendAccountingEvidence, ...]
        ],
        provider: object | None = None,
        market_data_artifact_ref: ArtifactRef | None = None,
        initial_custody: dict[str, ProtectiveExitState] | None = None,
    ) -> HistoricalBacktestRunResult:
        """The one authoritative run loop shared by both entry points.

        In provider mode each session's complete input is composed from the
        static plan plus lazily retrieved facts, carried entry intents and
        carried Phase 10 protective state; the session body itself is
        identical to the legacy path.  Custody is published only after that
        session body has returned successfully -- i.e. after its single Phase
        13 transition -- and its session result is appended only after custody
        reconciliation succeeds, so a failed session leaves neither a
        partially advanced protective chain nor a published result.
        """

        marks_resolver = (
            None
            if provider is None
            else _AllocationBoundaryMarkResolver(
                provider=provider,
                market_data_artifact_ref=market_data_artifact_ref,
            )
        )
        custody: dict[str, ProtectiveExitState] = dict(initial_custody or {})

        initial_fingerprint = hash_portfolio_state(initial_state)
        current_state = initial_state
        results: list[HistoricalBacktestSessionResult] = []
        carried_entry_intents: tuple[HistoricalBacktestEntryIntent, ...] = ()
        for index, item in enumerate(session_inputs):
            following = (
                session_inputs[index + 1].session
                if index + 1 < len(session_inputs)
                else None
            )
            if (
                carried_entry_intents
                and item.scheduled_entry_intents
                and item.scheduled_entry_intents != carried_entry_intents
            ):
                raise HistoricalBacktestValidationError(
                    "SCHEDULED_ENTRY_LINKAGE_MISMATCH",
                    "session T entries disagree with completed T-1 allocation",
                )
            active_entry_intents = (
                item.scheduled_entry_intents
                if item.scheduled_entry_intents
                else carried_entry_intents
            )
            effective_next_session = item.next_session or following
            session_input = item
            if provider is not None:
                session_input = self._compose_provider_session_input(
                    provider=provider,
                    static_input=item,
                    prior_state=current_state,
                    carried_entry_intents=active_entry_intents,
                    custody=custody,
                    effective_next_session=effective_next_session,
                )
            result = self._process_session(
                prior_state=current_state,
                session_input=session_input,
                effective_next_session=effective_next_session,
                scheduled_entry_intents=active_entry_intents,
                dividend_evidence=dividend_evidence_by_session.get(
                    item.session, ()
                ),
                allocation_marks_resolver=marks_resolver,
            )
            if provider is not None:
                custody = self._commit_protective_custody(
                    result=result, custody=custody
                )
            results.append(result)
            current_state = result.authoritative_state
            carried_entry_intents = result.future_entry_intents

        return HistoricalBacktestRunResult(
            schema_version=(
                DIVIDEND_AWARE_HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION
                if dividend_run_evidence is not None
                else HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION
            ),
            decision_interval=decision_interval,
            dividend_run_evidence=dividend_run_evidence,
            initial_state=initial_state,
            final_state=current_state,
            session_results=tuple(results),
            initial_state_fingerprint=initial_fingerprint,
            final_state_fingerprint=hash_portfolio_state(current_state),
        )

    def _compose_provider_session_input(
        self,
        *,
        provider: object,
        static_input: HistoricalBacktestSessionInput,
        prior_state: PortfolioState,
        carried_entry_intents: tuple[HistoricalBacktestEntryIntent, ...],
        custody: dict[str, ProtectiveExitState],
        effective_next_session: date | None,
    ) -> HistoricalBacktestSessionInput:
        """First provider boundary: retrieve session-T external facts once.

        Invoked after authoritative prior-state invariant and session-order
        validation and before any Phase 15B evaluation or Phase 9 execution,
        exactly once per processed session, with no prefetch and no retry.
        Protective state is supplied by Phase 15A custody, never by the
        provider.
        """

        PortfolioInvariantChecker.validate_state(prior_state)
        if (
            prior_state.as_of_session is not None
            and static_input.session <= prior_state.as_of_session
        ):
            raise HistoricalBacktestValidationError(
                "OUT_OF_ORDER_SESSION",
                "session T must be later than prior authoritative state",
            )

        context = build_session_context(
            session=static_input.session,
            decision_time=static_input.decision_time,
            next_session=effective_next_session,
            open_positions=prior_state.open_positions,
            scheduled_entry_security_ids=tuple(
                intent.security_id for intent in carried_entry_intents
            ),
        )
        state_inputs = validate_session_state_inputs(
            context=context,
            state_inputs=_call_provider_operation(
                provider, "session_state_inputs", context=context
            ),
        )

        evaluations: list[OpenPositionExitEvaluationInput] = []
        for facts in state_inputs.open_position_facts:
            protective_state = custody.get(facts.security_id)
            if protective_state is None:
                raise HistoricalBacktestValidationError(
                    "MISSING_CARRIED_PROTECTIVE_STATE",
                    "every open position entering T requires its carried "
                    "Phase 10 protective state",
                )
            try:
                evaluations.append(
                    OpenPositionExitEvaluationInput(
                        session=static_input.session,
                        protective_state=protective_state,
                        market_bar=facts.market_bar,
                        corporate_actions=facts.corporate_actions,
                        earnings_decision=facts.earnings_decision,
                        prior_boundary_earnings_decision=(
                            facts.prior_boundary_earnings_decision
                        ),
                    )
                )
            except ValueError as error:
                raise HistoricalBacktestValidationError(
                    "INVALID_OPEN_POSITION_EXIT_EVALUATION",
                    "provider facts and carried protective state do not "
                    "compose a valid Phase 15B evaluation",
                ) from error

        return static_input.model_copy(
            update={
                "scheduled_entry_intents": carried_entry_intents,
                "entry_execution_bars": state_inputs.entry_execution_bars,
                "open_position_exit_evaluations": tuple(evaluations),
            }
        )

    def _commit_protective_custody(
        self,
        *,
        result: HistoricalBacktestSessionResult,
        custody: dict[str, ProtectiveExitState],
    ) -> dict[str, ProtectiveExitState]:
        """Advance and retire carried Phase 10 state, then publish it.

        Phase 15A performs no protective arithmetic here and never creates
        state a second time: an entry executed on T was already created by
        Phase 10 and evaluated by Phase 15B inside the session body (Task
        5C-B), so its custody candidate is the exact ``resulting_state``
        object of that entry-session HOLD; a surviving prior position's
        candidate is the exact object Phase 10 returned inside its Phase 15B
        decision; retirement (a terminal exit, including a same-session round
        trip) is simply absence from authoritative S[T].  The whole mapping
        is replaced atomically, only after the session's single Phase 13
        transition has succeeded.
        """

        del custody  # the prior mapping is superseded wholesale by S[T]
        advanced: dict[str, ProtectiveExitState] = {}
        for decision in (
            result.open_position_exit_decisions
            + result.entry_session_protective_decisions
        ):
            resulting_state = decision.protective_exit_decision.resulting_state
            if resulting_state is not None:
                advanced[decision.security_id] = resulting_state

        candidate: dict[str, ProtectiveExitState] = {}
        for position in result.authoritative_state.open_positions:
            security_id = position.asset_id
            if security_id in advanced:
                candidate[security_id] = advanced[security_id]
            else:
                raise HistoricalBacktestValidationError(
                    "IMPOSSIBLE_PROTECTIVE_STATE_TRANSITION",
                    "a position surviving session T has no Phase 10 state to "
                    "carry forward",
                )
        return reconcile_protective_custody(
            authoritative_state=result.authoritative_state,
            candidate_custody=candidate,
        )


__all__ = ["HistoricalBacktestOrchestrator", "execution_presentation_rank"]
