"""Fail-closed Phase 15A chronology and ownership validation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from pydantic import ValidationError

from stock_swing_d1.data.ordinary_dividend_run_evidence import (
    DividendAwareRunEvidence,
    DividendAwareSessionEvidence,
    build_dividend_aware_run_evidence,
    build_dividend_aware_session_evidence,
)
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    OrdinaryDividendAccountingPolicyRef,
    ProcessedSessionContiguityProof,
)
from stock_swing_d1.execution.open_position_exit import (
    OpenPositionExitEvaluationInput,
)
from stock_swing_d1.execution.protective_exit import ProtectiveExitState
from stock_swing_d1.models import CorporateActionEvent
from stock_swing_d1.portfolio.models import PortfolioCandidateAction
from stock_swing_d1.portfolio.portfolio_events import ExecutionSide
from stock_swing_d1.portfolio.portfolio_errors import OutOfOrderSessionError
from stock_swing_d1.portfolio.portfolio_invariants import (
    PortfolioInvariantChecker,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PortfolioState,
)
from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.ranking.hashing import candidate_fingerprint
from stock_swing_d1.valuation import PortfolioValuationMark

from stock_swing_d1.backtester.models import (
    HistoricalDecisionInterval,
    HistoricalBacktestSessionInput,
)
from stock_swing_d1.backtester.session_input_provider import (
    HistoricalBacktestAllocationBoundaryMarks,
    HistoricalBacktestSessionContext,
    HistoricalBacktestSessionStateInputs,
)


class HistoricalBacktestValidationError(ValueError):
    """A Phase 15A chronology or ownership contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _revalidate_session_input(
    value: object,
) -> HistoricalBacktestSessionInput:
    if not isinstance(value, HistoricalBacktestSessionInput):
        raise HistoricalBacktestValidationError(
            "INVALID_SESSION_INPUT",
            "sessions must contain HistoricalBacktestSessionInput values",
        )
    values = {
        name: getattr(value, name)
        for name in HistoricalBacktestSessionInput.model_fields
    }
    try:
        rebuilt = HistoricalBacktestSessionInput.model_validate(values)
    except (ValidationError, TypeError, ValueError) as error:
        raise HistoricalBacktestValidationError(
            "INVALID_SESSION_INPUT",
            "session input fails structural validation",
        ) from error
    if rebuilt != value:
        raise HistoricalBacktestValidationError(
            "NONCANONICAL_SESSION_INPUT",
            "session input must already be canonical",
        )
    return rebuilt


def validate_session_sequence(
    *,
    initial_state: PortfolioState,
    sessions: Sequence[HistoricalBacktestSessionInput],
) -> tuple[HistoricalBacktestSessionInput, ...]:
    """Validate the complete caller sequence without sorting or mutation."""

    PortfolioInvariantChecker.validate_state(initial_state)
    if isinstance(sessions, (str, bytes)) or not isinstance(sessions, Sequence):
        raise HistoricalBacktestValidationError(
            "INVALID_SESSION_SEQUENCE",
            "sessions must be a finite sequence",
        )
    supplied = tuple(sessions)
    validated = tuple(_revalidate_session_input(item) for item in supplied)

    previous = initial_state.as_of_session
    for index, item in enumerate(validated):
        if previous is not None and item.session <= previous:
            raise OutOfOrderSessionError(
                "Phase 15A sessions must be supplied in strict chronological order"
            )
        if index + 1 < len(validated):
            following = validated[index + 1].session
            if item.next_session is not None and item.next_session != following:
                raise HistoricalBacktestValidationError(
                    "NEXT_SESSION_MISMATCH",
                    "next_session must equal the next supplied canonical session",
                )
        previous = item.session
    return validated


def validate_decision_interval_containment(
    *,
    decision_interval: HistoricalDecisionInterval,
    sessions: Sequence[HistoricalBacktestSessionInput],
) -> HistoricalDecisionInterval:
    """Validate every processed session against one explicit closed interval."""

    if type(decision_interval) is not HistoricalDecisionInterval:
        raise HistoricalBacktestValidationError(
            "INVALID_DECISION_INTERVAL",
            "decision_interval must be exactly HistoricalDecisionInterval",
        )
    try:
        interval = HistoricalDecisionInterval.model_validate(
            {
                field_name: getattr(decision_interval, field_name)
                for field_name in HistoricalDecisionInterval.model_fields
            }
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise HistoricalBacktestValidationError(
            "INVALID_DECISION_INTERVAL",
            "decision_interval fails canonical reconstruction",
        ) from error
    for item in sessions:
        if not (
            interval.decision_start_date
            <= item.session
            <= interval.decision_end_date
        ):
            raise HistoricalBacktestValidationError(
                "SESSION_OUTSIDE_DECISION_INTERVAL",
                "every processed session must lie within the closed "
                "historical decision interval",
            )
    return interval


def validate_dividend_aware_run_evidence(
    *,
    dividend_accounting_policy_ref: OrdinaryDividendAccountingPolicyRef | None,
    processed_session_contiguity_proof: ProcessedSessionContiguityProof | None,
    sessions: Sequence[HistoricalBacktestSessionInput],
) -> DividendAwareRunEvidence | None:
    """Validate and assemble one explicit Phase 15A dividend-aware run identity.

    Dividend-aware mode is an explicit run-level declaration
    (``dividend_accounting_policy_ref`` is supplied), never inferred from the
    presence of per-session dividend evidence. A non-dividend-aware run must
    not carry any dividend accounting evidence, coverage, or contiguity proof.
    """

    dividend_aware = dividend_accounting_policy_ref is not None

    if not dividend_aware:
        if processed_session_contiguity_proof is not None:
            raise HistoricalBacktestValidationError(
                "ORDINARY_DIVIDEND_MODE_MISMATCH",
                "a processed-session contiguity proof requires an explicit "
                "dividend accounting policy ref",
            )
        for item in sessions:
            if item.distribution_coverage is not None or item.distribution_events:
                raise HistoricalBacktestValidationError(
                    "ORDINARY_DIVIDEND_MODE_MISMATCH",
                    "a non-dividend-aware run must not carry dividend "
                    "accounting evidence or coverage",
                )
        return None

    if not sessions:
        if processed_session_contiguity_proof is not None:
            raise HistoricalBacktestValidationError(
                "ORDINARY_DIVIDEND_MODE_MISMATCH",
                "a zero-session run must not carry a contiguity proof",
            )
        try:
            return build_dividend_aware_run_evidence(
                dividend_accounting_policy_ref=dividend_accounting_policy_ref,
                processed_session_contiguity_proof=None,
                canonical_distribution_snapshot_fingerprint=None,
                session_evidence=(),
            )
        except (ValidationError, ValueError) as error:
            raise HistoricalBacktestValidationError(
                "ORDINARY_DIVIDEND_RUN_EVIDENCE_INVALID",
                "zero-session dividend-aware run evidence is malformed",
            ) from error

    if processed_session_contiguity_proof is None:
        raise HistoricalBacktestValidationError(
            "ORDINARY_DIVIDEND_PROCESSED_SPAN_NOT_CONTIGUOUS",
            "a non-empty dividend-aware run requires an explicit processed-"
            "session contiguity proof",
        )

    proof = processed_session_contiguity_proof
    actual_sessions = tuple(item.session for item in sessions)
    if proof.processed_sessions != actual_sessions:
        raise HistoricalBacktestValidationError(
            "ORDINARY_DIVIDEND_PROCESSED_SPAN_NOT_CONTIGUOUS",
            "the contiguity proof's processed-session sequence must equal "
            "the actual processed sessions",
        )
    for index, item in enumerate(sessions):
        if index + 1 >= len(sessions):
            continue
        proof_next_session = proof.session_links[index].next_session
        if item.next_session is None:
            raise HistoricalBacktestValidationError(
                "ORDINARY_DIVIDEND_PROCESSED_SPAN_NOT_CONTIGUOUS",
                "every non-final processed session requires an explicit "
                "next_session in a dividend-aware run",
            )
        if item.next_session != proof_next_session:
            raise HistoricalBacktestValidationError(
                "ORDINARY_DIVIDEND_PROCESSED_SPAN_NOT_CONTIGUOUS",
                "next_session must equal the run's bound contiguity proof",
            )

    all_event_ids: list[str] = []
    for item in sessions:
        for event in item.distribution_events or ():
            all_event_ids.append(event.canonical_distribution_event_id)
    if len(set(all_event_ids)) != len(all_event_ids):
        raise HistoricalBacktestValidationError(
            "ORDINARY_DIVIDEND_DUPLICATE_EVENT",
            "canonical distribution event IDs must be unique across the run",
        )

    for item in sessions:
        for event in item.distribution_events or ():
            if any(
                event.entitlement_session < session_date < event.ex_session
                for session_date in actual_sessions
            ):
                raise HistoricalBacktestValidationError(
                    "ORDINARY_DIVIDEND_SESSION_NOT_ADJACENT",
                    "no processed session may lie strictly between the "
                    "entitlement session and the ex-session",
                )

    session_evidence: list[DividendAwareSessionEvidence] = []
    snapshot: str | None = None
    for item in sessions:
        if item.distribution_coverage is None:
            raise HistoricalBacktestValidationError(
                "ORDINARY_DIVIDEND_COVERAGE_MISSING",
                "every processed session in a dividend-aware run requires "
                "affirmative distribution coverage",
            )
        if snapshot is None:
            snapshot = (
                item.distribution_coverage
                .canonical_distribution_snapshot_fingerprint
            )
        try:
            session_evidence.append(
                build_dividend_aware_session_evidence(
                    session=item.session,
                    distribution_coverage=item.distribution_coverage,
                    distribution_events=item.distribution_events or (),
                )
            )
        except (ValidationError, ValueError) as error:
            raise HistoricalBacktestValidationError(
                "ORDINARY_DIVIDEND_SESSION_EVIDENCE_INVALID",
                "session dividend evidence fails structural validation",
            ) from error

    try:
        return build_dividend_aware_run_evidence(
            dividend_accounting_policy_ref=dividend_accounting_policy_ref,
            processed_session_contiguity_proof=proof,
            canonical_distribution_snapshot_fingerprint=snapshot,
            session_evidence=tuple(session_evidence),
        )
    except (ValidationError, ValueError) as error:
        raise HistoricalBacktestValidationError(
            "ORDINARY_DIVIDEND_RUN_EVIDENCE_INVALID",
            "dividend-aware run evidence fails structural validation",
        ) from error


def validate_session_contract(
    session_input: HistoricalBacktestSessionInput,
    *,
    effective_next_session: date | None,
) -> HistoricalBacktestSessionInput:
    """Validate one T boundary before invoking any economic dependency."""

    item = _revalidate_session_input(session_input)
    if item.pre_open_corporate_actions:
        raise HistoricalBacktestValidationError(
            "UNSUPPORTED_CORPORATE_ACTION_TRANSITION",
            "Phase 13 v0.1 has no authoritative corporate-action state event",
        )

    execution_ids: list[str] = []
    for event in item.scheduled_execution_events:
        if event.session != item.session:
            raise HistoricalBacktestValidationError(
                "SCHEDULED_ACTION_SESSION_MISMATCH",
                "every scheduled execution must belong to session T",
            )
        execution_ids.append(event.execution_id)
    if len(set(execution_ids)) != len(execution_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_SCHEDULED_EXECUTION",
            "scheduled execution IDs must be unique within T",
        )

    exit_evaluation_ids: list[str] = []
    for evaluation in item.open_position_exit_evaluations:
        if not isinstance(evaluation, OpenPositionExitEvaluationInput):
            raise HistoricalBacktestValidationError(
                "INVALID_OPEN_POSITION_EXIT_EVALUATION",
                "exit evaluations must use the frozen Phase 15B input",
            )
        if evaluation.session != item.session:
            raise HistoricalBacktestValidationError(
                "OPEN_POSITION_EXIT_SESSION_MISMATCH",
                "every Phase 15B evaluation must belong to session T",
            )
        if any(
            isinstance(event, CorporateActionEvent)
            for event in evaluation.corporate_actions
        ):
            raise HistoricalBacktestValidationError(
                "UNSUPPORTED_CORPORATE_ACTION_TRANSITION",
                "Phase 13 v0.1 has no authoritative portfolio quantity or "
                "cost-basis corporate-action transition",
            )
        exit_evaluation_ids.append(evaluation.protective_state.security_id)
    if len(set(exit_evaluation_ids)) != len(exit_evaluation_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_OPEN_POSITION_EXIT_EVALUATION",
            "Phase 15B evaluations must be unique by security ID within T",
        )

    scheduled_entry_ids: list[str] = []
    for intent in item.scheduled_entry_intents:
        if intent.planned_entry_session != item.session:
            raise HistoricalBacktestValidationError(
                "SCHEDULED_ENTRY_SESSION_MISMATCH",
                "every entry intent must target session T",
            )
        scheduled_entry_ids.append(intent.security_id)
    if len(set(scheduled_entry_ids)) != len(scheduled_entry_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_SCHEDULED_ENTRY",
            "scheduled entry security IDs must be unique within T",
        )

    entry_bar_ids: list[str] = []
    for bar in item.entry_execution_bars:
        if bar.trading_date != item.session:
            raise HistoricalBacktestValidationError(
                "ENTRY_BAR_SESSION_MISMATCH",
                "every entry execution bar must belong to session T",
            )
        entry_bar_ids.append(bar.security_id)
    if len(set(entry_bar_ids)) != len(entry_bar_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_ENTRY_EXECUTION_BAR",
            "entry execution bars must be unique by canonical security ID",
        )

    history_keys: list[tuple[str, date]] = []
    current_adjusted_ids: list[str] = []
    for bar in item.indicator_history_bars:
        if bar.trading_date > item.session:
            raise HistoricalBacktestValidationError(
                "FUTURE_DATA_BOUNDARY_VIOLATION",
                "indicator history cannot contain a bar after session T",
            )
        history_keys.append((bar.security_id, bar.trading_date))
        if bar.trading_date == item.session:
            current_adjusted_ids.append(bar.security_id)
    if len(set(history_keys)) != len(history_keys):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_INDICATOR_BAR",
            "indicator history keys must be unique",
        )
    if len(set(current_adjusted_ids)) != len(current_adjusted_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_COMPLETED_ADJUSTED_BAR",
            "completed-T adjusted bars must be unique by security ID",
        )

    completed_unadjusted_ids: list[str] = []
    for bar in item.completed_unadjusted_bars:
        if bar.trading_date != item.session:
            raise HistoricalBacktestValidationError(
                "COMPLETED_BAR_SESSION_MISMATCH",
                "completed unadjusted bars must belong to session T",
            )
        completed_unadjusted_ids.append(bar.security_id)
    if len(set(completed_unadjusted_ids)) != len(completed_unadjusted_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_COMPLETED_UNADJUSTED_BAR",
            "completed unadjusted bars must be unique by security ID",
        )

    eligible_ids = item.universe_eligible_security_ids
    if len(set(eligible_ids)) != len(eligible_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_UNIVERSE_SECURITY_ID",
            "PIT universe security IDs must be unique",
        )
    raw_candidate_mode = bool(
        item.indicator_history_bars
        or item.completed_unadjusted_bars
        or eligible_ids
    )
    if raw_candidate_mode and (
        item.ranking_candidates or item.allocation_candidates
    ):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_CANDIDATE_PREPARATION_PATH",
            "supply raw completed-T inputs or prepared candidates, not both",
        )
    if raw_candidate_mode:
        missing_eligible_bars = set(eligible_ids) - set(current_adjusted_ids)
        if missing_eligible_bars:
            raise HistoricalBacktestValidationError(
                "MISSING_COMPLETED_ADJUSTED_BAR",
                "every PIT-eligible security requires its completed-T adjusted bar",
            )

    ranking_ids: list[str] = []
    for candidate in item.ranking_candidates:
        if (
            candidate.ranking_session != item.session
            or candidate.signal_session != item.session
        ):
            raise HistoricalBacktestValidationError(
                "CANDIDATE_SESSION_MISMATCH",
                "every ranking candidate must be a completed-T candidate",
            )
        if (
            candidate.decision_time != item.decision_time
            or candidate.signal_time > item.decision_time
        ):
            raise HistoricalBacktestValidationError(
                "FUTURE_DATA_BOUNDARY_VIOLATION",
                "ranking inputs must be known by the completed-T decision time",
            )
        if candidate.input_fingerprint != candidate_fingerprint(candidate):
            raise HistoricalBacktestValidationError(
                "CANDIDATE_FINGERPRINT_MISMATCH",
                "ranking candidate fingerprint is not canonical",
            )
        ranking_ids.append(candidate.security_id)
    if len(set(ranking_ids)) != len(ranking_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_CANDIDATE_SECURITY_ID",
            "completed-T ranking candidate IDs must be unique",
        )

    allocation_ids: list[str] = []
    for candidate in item.allocation_candidates:
        if candidate.signal_session != item.session:
            raise HistoricalBacktestValidationError(
                "CANDIDATE_SESSION_MISMATCH",
                "every allocation candidate must be a completed-T candidate",
            )
        if candidate.signal.signal_time > item.decision_time:
            raise HistoricalBacktestValidationError(
                "FUTURE_DATA_BOUNDARY_VIOLATION",
                "allocation inputs must be known by completed-T decision time",
            )
        planned = candidate.pending_entry.planned_entry_session
        if effective_next_session is None:
            raise HistoricalBacktestValidationError(
                "MISSING_NEXT_SESSION_IDENTITY",
                "a future entry requires an explicit canonical next session",
            )
        if planned != effective_next_session:
            raise HistoricalBacktestValidationError(
                "PLANNED_ENTRY_SESSION_MISMATCH",
                "future entry must target the supplied next market session",
            )
        allocation_ids.append(candidate.security_id)
    if len(set(allocation_ids)) != len(allocation_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_ALLOCATION_SECURITY_ID",
            "completed-T allocation candidate IDs must be unique",
        )

    if set(ranking_ids) != set(allocation_ids):
        raise HistoricalBacktestValidationError(
            "RANKING_ALLOCATION_RECONCILIATION_MISMATCH",
            "ranking and allocation candidates must be the same complete set",
        )
    allocations_by_id = {
        candidate.security_id: candidate
        for candidate in item.allocation_candidates
    }
    for ranking_candidate in item.ranking_candidates:
        allocation_candidate = allocations_by_id[ranking_candidate.security_id]
        signal = allocation_candidate.signal
        if (
            ranking_candidate.signal_time != signal.signal_time
            or ranking_candidate.sma20 != signal.sma_20
            or ranking_candidate.sma50 != signal.sma_50
            or ranking_candidate.atr14 != signal.atr_14
            or ranking_candidate.rsi14 != signal.rsi_14
        ):
            raise HistoricalBacktestValidationError(
                "RANKING_ALLOCATION_CANDIDATE_MISMATCH",
                "Phase 14 and Phase 12 candidates must preserve one Phase 8 decision",
            )
    if item.allocation_candidates and item.allocation_portfolio is None:
        raise HistoricalBacktestValidationError(
            "MISSING_ALLOCATION_PORTFOLIO",
            "a non-empty completed-T batch requires a Phase 12 portfolio snapshot",
        )
    return item


def reconcile_open_position_exit_coverage(
    *,
    prior_state: PortfolioState,
    session_input: HistoricalBacktestSessionInput,
) -> tuple[tuple[OpenPosition, OpenPositionExitEvaluationInput], ...]:
    """Prove one terminal treatment for every position open entering T."""

    positions_by_id = {
        position.asset_id: position for position in prior_state.open_positions
    }
    scheduled_sells_by_id: dict[str, object] = {}
    for event in session_input.scheduled_execution_events:
        if event.side is not ExecutionSide.SELL:
            continue
        position = positions_by_id.get(event.asset_id)
        if position is None:
            raise HistoricalBacktestValidationError(
                "SCHEDULED_SELL_WITHOUT_OPEN_POSITION",
                "a scheduled SELL must target a position open entering T",
            )
        if event.asset_id in scheduled_sells_by_id:
            raise HistoricalBacktestValidationError(
                "DUPLICATE_OPEN_POSITION_EXIT_TREATMENT",
                "a prior position cannot have multiple scheduled terminal SELLs",
            )
        if event.quantity != position.quantity:
            raise HistoricalBacktestValidationError(
                "PARTIAL_SCHEDULED_SELL_UNSUPPORTED",
                "a scheduled SELL for a prior position must close its full quantity",
            )
        scheduled_sells_by_id[event.asset_id] = event

    evaluations_by_id: dict[str, OpenPositionExitEvaluationInput] = {}
    for evaluation in session_input.open_position_exit_evaluations:
        security_id = evaluation.protective_state.security_id
        if security_id in evaluations_by_id:
            raise HistoricalBacktestValidationError(
                "DUPLICATE_OPEN_POSITION_EXIT_EVALUATION",
                "a prior position cannot have multiple Phase 15B evaluations",
            )
        position = positions_by_id.get(security_id)
        if position is None:
            raise HistoricalBacktestValidationError(
                "EXTRA_OPEN_POSITION_EXIT_EVALUATION",
                "a Phase 15B evaluation must target a position open entering T",
            )
        if security_id in scheduled_sells_by_id:
            raise HistoricalBacktestValidationError(
                "DUPLICATE_OPEN_POSITION_EXIT_TREATMENT",
                "a position cannot have both a scheduled SELL and Phase 15B evaluation",
            )
        if evaluation.protective_state.entry_session != position.entry_session:
            raise HistoricalBacktestValidationError(
                "OPEN_POSITION_EXIT_IDENTITY_MISMATCH",
                "Phase 15B and Phase 13 entry sessions must agree",
            )
        if (
            Decimal(str(evaluation.protective_state.entry_price))
            != position.entry_price
        ):
            raise HistoricalBacktestValidationError(
                "OPEN_POSITION_EXIT_IDENTITY_MISMATCH",
                "Phase 15B and Phase 13 entry prices must agree",
            )
        evaluations_by_id[security_id] = evaluation

    uncovered = (
        set(positions_by_id)
        - set(scheduled_sells_by_id)
        - set(evaluations_by_id)
    )
    if uncovered:
        raise HistoricalBacktestValidationError(
            "MISSING_OPEN_POSITION_EXIT_EVALUATION",
            "every prior open position needs one Phase 15B evaluation or "
            "full scheduled SELL",
        )

    return tuple(
        (positions_by_id[security_id], evaluations_by_id[security_id])
        for security_id in sorted(evaluations_by_id)
    )


def validate_allocation_portfolio_against_state(
    *,
    session_input: HistoricalBacktestSessionInput,
    authoritative_state: PortfolioState,
    authoritative_portfolio_equity: Decimal | None = None,
) -> None:
    """Prove Phase 12 sees Phase 13 cash/slot semantics, not future proceeds.

    Task 5C-A froze the projection direction: an authoritative ``Decimal`` is
    projected forward exactly once via ``float(...)``, and the float Phase 12
    consumed is compared against that same prescribed projection by exact
    equality.  The reverse check ``Decimal(str(portfolio.cash_available)) ==
    authoritative_state.settled_cash`` must never be used as authority:
    legitimate high-precision balances produced by the frozen dividend
    accounting -- ``settled_cash`` carrying Gate3 scale-38 dividend cash --
    can fail ``Decimal(str(float(value))) == value`` even though the forward
    projection is exactly right.  No tolerance, no epsilon, no quantization
    and no cent rounding participate in either direction.

    ``authoritative_portfolio_equity`` is the equity the shared valuation
    owner computed for exactly this state and session.  It is supplied
    wherever authoritative valuation evidence exists (the state-aware
    provider path); the legacy pre-materialized path supplies an allocation
    portfolio without marks, so no authoritative equity exists there and only
    the cash projection is provable.
    """

    portfolio = session_input.allocation_portfolio
    if portfolio is None:
        return
    if (
        portfolio.allocation_session != session_input.session
        or portfolio.decision_time != session_input.decision_time
    ):
        raise HistoricalBacktestValidationError(
            "ALLOCATION_BOUNDARY_MISMATCH",
            "Phase 12 portfolio identity must equal completed session T",
        )

    if portfolio.cash_available != float(authoritative_state.settled_cash):
        raise HistoricalBacktestValidationError(
            "NONAUTHORITATIVE_ALLOCATION_CASH",
            "Phase 12 cash must equal the prescribed float projection of "
            "authoritative Phase 13 settled cash",
        )
    if authoritative_portfolio_equity is not None:
        if type(authoritative_portfolio_equity) is not Decimal:
            raise HistoricalBacktestValidationError(
                "NONAUTHORITATIVE_ALLOCATION_EQUITY",
                "authoritative portfolio equity must be an exact Decimal",
            )
        if portfolio.portfolio_equity != float(authoritative_portfolio_equity):
            raise HistoricalBacktestValidationError(
                "NONAUTHORITATIVE_ALLOCATION_EQUITY",
                "Phase 12 equity must equal the prescribed float projection "
                "of authoritative allocation-boundary portfolio equity",
            )

    supplied_positions = {
        position.security_id: (position.shares, position.entry_session)
        for position in portfolio.open_positions
    }
    authoritative_positions = {
        position.asset_id: (position.quantity, position.entry_session)
        for position in authoritative_state.open_positions
    }
    if supplied_positions != authoritative_positions:
        raise HistoricalBacktestValidationError(
            "NONAUTHORITATIVE_ALLOCATION_POSITIONS",
            "Phase 12 open positions must equal authoritative Phase 13 positions",
        )


def ensure_admitted_decision(action: object) -> None:
    if action is not PortfolioCandidateAction.ADMITTED:
        raise HistoricalBacktestValidationError(
            "INVALID_FUTURE_ENTRY_INTENT",
            "only an admitted Phase 12 decision can schedule an entry intent",
        )


def validate_scheduled_entry_intents(
    *,
    session: date,
    intents: tuple[object, ...],
    execution_bar_security_ids: set[str],
) -> None:
    """Validate dynamically carried entry intents against the T input seam."""

    from stock_swing_d1.backtester.models import HistoricalBacktestEntryIntent

    security_ids: list[str] = []
    for intent in intents:
        if not isinstance(intent, HistoricalBacktestEntryIntent):
            raise HistoricalBacktestValidationError(
                "INVALID_SCHEDULED_ENTRY",
                "scheduled entries must use HistoricalBacktestEntryIntent",
            )
        if intent.planned_entry_session != session:
            raise HistoricalBacktestValidationError(
                "SCHEDULED_ENTRY_SESSION_MISMATCH",
                "carried entry intent must target session T",
            )
        security_ids.append(intent.security_id)
    if len(set(security_ids)) != len(security_ids):
        raise HistoricalBacktestValidationError(
            "DUPLICATE_SCHEDULED_ENTRY",
            "carried entry intent security IDs must be unique",
        )
    extra_bars = execution_bar_security_ids - set(security_ids)
    if extra_bars:
        raise HistoricalBacktestValidationError(
            "EXTRA_ENTRY_EXECUTION_BAR",
            "execution bars cannot be supplied for unscheduled entries",
        )


def validate_session_state_inputs(
    *,
    context: HistoricalBacktestSessionContext,
    state_inputs: object,
) -> HistoricalBacktestSessionStateInputs:
    """Prove one provider response covers exactly the requested identifiers.

    The provider supplies external facts only.  Coverage is exact in both
    directions: one session-T entry execution bar per carried entry intent and
    one fact record per authoritative open position, with nothing missing,
    nothing extra and nothing duplicated.
    """

    if type(state_inputs) is not HistoricalBacktestSessionStateInputs:
        raise HistoricalBacktestValidationError(
            "INVALID_SESSION_STATE_INPUTS",
            "the provider must return exactly "
            "HistoricalBacktestSessionStateInputs",
        )
    try:
        rebuilt = HistoricalBacktestSessionStateInputs.model_validate(
            {
                name: getattr(state_inputs, name)
                for name in HistoricalBacktestSessionStateInputs.model_fields
            }
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise HistoricalBacktestValidationError(
            "INVALID_SESSION_STATE_INPUTS",
            "provider session state inputs fail structural validation",
        ) from error
    if rebuilt != state_inputs:
        raise HistoricalBacktestValidationError(
            "INVALID_SESSION_STATE_INPUTS",
            "provider session state inputs must already be canonical",
        )
    if rebuilt.session != context.session:
        raise HistoricalBacktestValidationError(
            "SESSION_STATE_INPUTS_SESSION_MISMATCH",
            "provider session state inputs must belong to session T",
        )

    supplied_bars = {bar.security_id for bar in rebuilt.entry_execution_bars}
    expected_bars = set(context.scheduled_entry_security_ids)
    if supplied_bars - expected_bars:
        raise HistoricalBacktestValidationError(
            "EXTRA_ENTRY_EXECUTION_BAR",
            "execution bars cannot be supplied for unscheduled entries",
        )
    if expected_bars - supplied_bars:
        raise HistoricalBacktestValidationError(
            "MISSING_ENTRY_EXECUTION_BAR",
            "every carried entry intent requires its session-T execution bar",
        )

    supplied_facts = {
        facts.security_id for facts in rebuilt.open_position_facts
    }
    expected_facts = {ref.security_id for ref in context.open_position_refs}
    if supplied_facts - expected_facts:
        raise HistoricalBacktestValidationError(
            "EXTRA_OPEN_POSITION_FACTS",
            "provider facts cannot cover a security with no open position",
        )
    if expected_facts - supplied_facts:
        raise HistoricalBacktestValidationError(
            "MISSING_OPEN_POSITION_FACTS",
            "every authoritative open position requires its provider facts",
        )
    return rebuilt


def validate_allocation_boundary_marks(
    *,
    marks_result: object,
    session: date,
    security_ids: tuple[str, ...],
    market_data_artifact_ref: ArtifactRef,
) -> tuple[PortfolioValuationMark, ...]:
    """Prove the provider's marks are exact, complete and run-authoritative.

    The provider must not self-authorize provenance: a mark set whose refs
    are mutually identical but differ from the authoritative run-level
    market-data artifact is rejected here, before the shared valuation owner
    is invoked, before any ``PortfolioSnapshot`` exists and before Phase 12
    is reached.
    """

    if type(marks_result) is not HistoricalBacktestAllocationBoundaryMarks:
        raise HistoricalBacktestValidationError(
            "INVALID_ALLOCATION_BOUNDARY_MARKS",
            "the provider must return exactly "
            "HistoricalBacktestAllocationBoundaryMarks",
        )
    if any(
        type(mark) is not PortfolioValuationMark for mark in marks_result.marks
    ):
        raise HistoricalBacktestValidationError(
            "INVALID_ALLOCATION_BOUNDARY_MARKS",
            "provider valuation marks must be exact PortfolioValuationMark values",
        )
    try:
        # Rebuild from a plain dump so every nested mark -- and each mark's
        # nested ArtifactRef -- is fully revalidated: a `model_copy(update=)`
        # forgery keeps the right Python class but bypasses validation, and
        # must fail here, before valuation, Phase 12 and custody publication.
        rebuilt = HistoricalBacktestAllocationBoundaryMarks.model_validate(
            marks_result.model_dump(mode="python")
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise HistoricalBacktestValidationError(
            "INVALID_ALLOCATION_BOUNDARY_MARKS",
            "provider valuation marks fail structural validation",
        ) from error
    if rebuilt != marks_result:
        raise HistoricalBacktestValidationError(
            "INVALID_ALLOCATION_BOUNDARY_MARKS",
            "provider valuation marks must already be canonical",
        )
    if rebuilt.session != session:
        raise HistoricalBacktestValidationError(
            "ALLOCATION_MARK_SESSION_MISMATCH",
            "provider valuation marks must belong to completed session T",
        )
    for mark in rebuilt.marks:
        if mark.session != session:
            raise HistoricalBacktestValidationError(
                "ALLOCATION_MARK_SESSION_MISMATCH",
                "every valuation mark must belong to completed session T",
            )

    supplied = {mark.security_id for mark in rebuilt.marks}
    expected = set(security_ids)
    if supplied - expected:
        raise HistoricalBacktestValidationError(
            "EXTRA_VALUATION_MARK",
            "a valuation mark was supplied for a security with no "
            "authoritative open position",
        )
    if expected - supplied:
        raise HistoricalBacktestValidationError(
            "MISSING_VALUATION_MARK",
            "every authoritative open position requires its valuation mark",
        )

    if type(market_data_artifact_ref) is not ArtifactRef:
        raise HistoricalBacktestValidationError(
            "VALUATION_MARK_ARTIFACT_MISMATCH",
            "the run-level market-data artifact ref must be an ArtifactRef",
        )
    source_refs = {mark.source_artifact_ref for mark in rebuilt.marks}
    if len(source_refs) > 1:
        raise HistoricalBacktestValidationError(
            "VALUATION_MARK_ARTIFACT_MISMATCH",
            "every valuation mark must share one market-data artifact ref",
        )
    if any(ref != market_data_artifact_ref for ref in source_refs):
        raise HistoricalBacktestValidationError(
            "VALUATION_MARK_ARTIFACT_MISMATCH",
            "valuation marks must carry the authoritative run-level "
            "market-data artifact ref",
        )
    return rebuilt.marks


def _require_protective_state(value: object) -> ProtectiveExitState:
    if not isinstance(value, ProtectiveExitState):
        raise HistoricalBacktestValidationError(
            "INVALID_PROTECTIVE_STATE",
            "carried protective state must be a Phase 10 ProtectiveExitState",
        )
    return value


def validate_initial_protective_states(
    *,
    initial_state: PortfolioState,
    protective_states: tuple[object, ...],
) -> dict[str, ProtectiveExitState]:
    """Seed custody with exactly one Phase 10 state per initial position.

    Phase 15A never synthesizes protective state: a carried-in position
    without its explicit Phase 10 seed fails the run closed.
    """

    if type(protective_states) is not tuple:
        raise HistoricalBacktestValidationError(
            "INVALID_PROTECTIVE_STATE",
            "initial protective states must be an immutable tuple",
        )
    seeded: dict[str, ProtectiveExitState] = {}
    for value in protective_states:
        state = _require_protective_state(value)
        if state.security_id in seeded:
            raise HistoricalBacktestValidationError(
                "DUPLICATE_INITIAL_PROTECTIVE_STATE",
                "initial protective states must be unique by security ID",
            )
        seeded[state.security_id] = state

    positions = {
        position.asset_id: position for position in initial_state.open_positions
    }
    extra = set(seeded) - set(positions)
    if extra:
        raise HistoricalBacktestValidationError(
            "EXTRA_INITIAL_PROTECTIVE_STATE",
            "an initial protective state has no carried-in open position",
        )
    missing = set(positions) - set(seeded)
    if missing:
        raise HistoricalBacktestValidationError(
            "MISSING_INITIAL_PROTECTIVE_STATE",
            "every carried-in open position requires its explicit Phase 10 "
            "protective state",
        )
    for security_id, position in positions.items():
        if seeded[security_id].entry_session != position.entry_session:
            raise HistoricalBacktestValidationError(
                "PROTECTIVE_STATE_IDENTITY_MISMATCH",
                "protective state and authoritative position entry sessions "
                "must agree",
            )
    return seeded


def reconcile_protective_custody(
    *,
    authoritative_state: PortfolioState,
    candidate_custody: dict[str, ProtectiveExitState],
) -> dict[str, ProtectiveExitState]:
    """Prove candidate custody equals the surviving authoritative positions."""

    positions = {
        position.asset_id: position
        for position in authoritative_state.open_positions
    }
    if set(candidate_custody) != set(positions):
        raise HistoricalBacktestValidationError(
            "PROTECTIVE_CUSTODY_COVERAGE_MISMATCH",
            "carried protective state must cover exactly the authoritative "
            "post-transition open positions",
        )
    for security_id, position in positions.items():
        state = _require_protective_state(candidate_custody[security_id])
        if (
            state.security_id != security_id
            or state.entry_session != position.entry_session
        ):
            raise HistoricalBacktestValidationError(
                "PROTECTIVE_STATE_IDENTITY_MISMATCH",
                "carried protective state identity must equal its "
                "authoritative open position",
            )
    return dict(candidate_custody)


__all__ = ["HistoricalBacktestValidationError"]
