"""Pure Phase 15B coordination over Phase 10 and PIT earnings outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.earnings.models import EarningsRiskDecision, EarningsStateAsOf
from stock_swing_d1.earnings.risk import evaluate_open_position_earnings_risk
from stock_swing_d1.execution.open_position_exit.models import (
    EarningsExitStatus,
    ExitBoundary,
    ExitPrerequisiteStatus,
    IntrabarAmbiguityStatus,
    MAX_HOLDING_SESSIONS,
    OpenPositionExitCalendar,
    OpenPositionExitDecision,
    OpenPositionExitEvaluationInput,
    OpenPositionExitReason,
    OpenPositionExitValidationError,
)
from stock_swing_d1.execution.protective_exit import (
    ProtectiveExitAction,
    ProtectiveExitDecision,
    ProtectiveExitService,
    ProtectiveExitValidationError,
)


_PROTECTIVE_ACTIONS = frozenset(
    {
        ProtectiveExitAction.GAP_STOP_EXIT,
        ProtectiveExitAction.GAP_TAKE_PROFIT_EXIT,
        ProtectiveExitAction.STOP_LOSS_EXIT,
        ProtectiveExitAction.TAKE_PROFIT_EXIT,
    }
)
_SUPPORTED_EARNINGS_ACTIONS = frozenset(
    {
        EarningsIntegrationAction.HOLD_POSITION,
        EarningsIntegrationAction.EXIT_REQUIRED_THIS_SESSION,
        EarningsIntegrationAction.MISSED_EXIT_DEADLINE,
        EarningsIntegrationAction.UNAVOIDABLE_EARNINGS_EXPOSURE,
    }
)


def _is_aware(value: object) -> bool:
    try:
        return (
            isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        )
    except Exception:
        return False


def _calendar_session_distance(
    calendar: OpenPositionExitCalendar, start: date, end: date
) -> int:
    try:
        distance = calendar.session_distance(start, end)
    except Exception as error:
        raise OpenPositionExitValidationError(
            "HOLDING_SESSION_CALENDAR_FAILURE",
            "the calendar could not calculate the inclusive holding age",
        ) from error
    if type(distance) is not int or distance < 0:
        raise OpenPositionExitValidationError(
            "INVALID_HOLDING_SESSION_DISTANCE",
            "session_distance must return a non-negative whole-session distance",
        )
    return distance


def _expected_protective_session(
    evaluation: OpenPositionExitEvaluationInput,
    calendar: OpenPositionExitCalendar,
) -> date:
    state = evaluation.protective_state
    if state.last_evaluated_session is None:
        expected = state.entry_session
    else:
        try:
            expected = calendar.next_session(state.last_evaluated_session)
        except Exception as error:
            raise OpenPositionExitValidationError(
                "PROTECTIVE_SESSION_CALENDAR_FAILURE",
                "the calendar could not resolve the next protective-exit session",
            ) from error
    if type(expected) is not date or expected != evaluation.session:
        raise OpenPositionExitValidationError(
            "EXIT_SESSION_MISMATCH",
            "session must be the next Phase 10 evaluation session",
        )
    return expected


def _validate_earnings_decision(
    *,
    decision: EarningsIntegrationDecision,
    evaluation: OpenPositionExitEvaluationInput,
    calendar: OpenPositionExitCalendar,
    boundary_session: date,
    provenance_name: str,
) -> date | None:
    if not isinstance(decision, EarningsIntegrationDecision) or not isinstance(
        decision.action, EarningsIntegrationAction
    ):
        raise OpenPositionExitValidationError(
            "INVALID_EARNINGS_DECISION",
            f"{provenance_name} must be authoritative earnings-layer output",
        )
    if decision.action not in _SUPPORTED_EARNINGS_ACTIONS:
        raise OpenPositionExitValidationError(
            "INVALID_OPEN_POSITION_EARNINGS_ACTION",
            f"{provenance_name} must be an open-position PIT earnings decision",
        )
    state = decision.earnings_state
    risk = decision.risk_decision
    if not isinstance(state, EarningsStateAsOf) or not isinstance(
        risk, EarningsRiskDecision
    ):
        raise OpenPositionExitValidationError(
            "INCOMPLETE_EARNINGS_DECISION",
            f"{provenance_name} must retain its PIT state and risk decision",
        )
    if state.canonical_asset_id != evaluation.protective_state.security_id:
        raise OpenPositionExitValidationError(
            "EARNINGS_IDENTITY_MISMATCH",
            "earnings state canonical asset does not match the open position",
        )
    try:
        boundary_time = calendar.decision_time(boundary_session)
    except Exception as error:
        raise OpenPositionExitValidationError(
            "EARNINGS_BOUNDARY_CALENDAR_FAILURE",
            "the calendar could not resolve the D1 administrative boundary",
        ) from error
    if not _is_aware(boundary_time) or state.as_of != boundary_time:
        raise OpenPositionExitValidationError(
            "INVALID_EARNINGS_DECISION_BOUNDARY",
            f"{provenance_name} must be reconstructed at its completed-D1 boundary",
        )
    try:
        authoritative_risk = evaluate_open_position_earnings_risk(
            state,
            boundary_session,
            calendar,
            position_entry_session=evaluation.protective_state.entry_session,
        )
    except Exception as error:
        raise OpenPositionExitValidationError(
            "INVALID_EARNINGS_TIMING_CLASSIFICATION",
            "earnings timing must have an authoritative D1 deadline",
        ) from error
    if risk != authoritative_risk:
        raise OpenPositionExitValidationError(
            "EARNINGS_RISK_DECISION_MISMATCH",
            "the integration risk decision must match the PIT state and position",
        )

    current = boundary_session
    action = decision.action
    authoritative_deadline = authoritative_risk.last_safe_exit_session
    if authoritative_deadline is None or current < authoritative_deadline:
        expected_action = EarningsIntegrationAction.HOLD_POSITION
    elif current == authoritative_deadline:
        expected_action = EarningsIntegrationAction.EXIT_REQUIRED_THIS_SESSION
    elif authoritative_risk.unavoidable_earnings_exposure:
        expected_action = EarningsIntegrationAction.UNAVOIDABLE_EARNINGS_EXPOSURE
    else:
        expected_action = EarningsIntegrationAction.MISSED_EXIT_DEADLINE
    if action is not expected_action:
        raise OpenPositionExitValidationError(
            "EARNINGS_INTEGRATION_ACTION_MISMATCH",
            "the integration action must match the authoritative PIT risk decision",
        )
    return authoritative_deadline


def _previous_session(
    evaluation: OpenPositionExitEvaluationInput,
    calendar: OpenPositionExitCalendar,
) -> date:
    try:
        previous = calendar.previous_session(evaluation.session)
    except Exception as error:
        raise OpenPositionExitValidationError(
            "PRIOR_EARNINGS_BOUNDARY_CALENDAR_FAILURE",
            "the calendar could not resolve the prior completed-D1 boundary",
        ) from error
    if type(previous) is not date or previous >= evaluation.session:
        raise OpenPositionExitValidationError(
            "INVALID_PRIOR_EARNINGS_BOUNDARY",
            "previous_session must return an earlier canonical session",
        )
    return previous


def _earnings_authorization(
    *,
    evaluation: OpenPositionExitEvaluationInput,
    calendar: OpenPositionExitCalendar,
) -> tuple[EarningsExitStatus, date | None, bool]:
    """Separate current-T observation from causally prior T-close authority."""

    current_decision = evaluation.earnings_decision
    prior_decision = evaluation.prior_boundary_earnings_decision
    current_deadline = (
        _validate_earnings_decision(
            decision=current_decision,
            evaluation=evaluation,
            calendar=calendar,
            boundary_session=evaluation.session,
            provenance_name="earnings_decision",
        )
        if current_decision is not None
        else None
    )
    prior_deadline = None
    if prior_decision is not None:
        prior_deadline = _validate_earnings_decision(
            decision=prior_decision,
            evaluation=evaluation,
            calendar=calendar,
            boundary_session=_previous_session(evaluation, calendar),
            provenance_name="prior_boundary_earnings_decision",
        )

    # Only the explicit PIT view from the prior completed session can authorize
    # execution at T close.  The current view cannot create or cancel that
    # already-established instruction retroactively.
    if prior_deadline is not None and prior_deadline <= evaluation.session:
        status = (
            EarningsExitStatus.EXIT_REQUIRED_THIS_SESSION
            if prior_deadline == evaluation.session
            else EarningsExitStatus.DEADLINE_MISSED
        )
        return status, prior_deadline, True

    if current_decision is not None:
        if current_deadline is not None and current_deadline <= evaluation.session:
            return EarningsExitStatus.DEADLINE_MISSED, current_deadline, False
        if current_deadline is not None:
            return EarningsExitStatus.NOT_DUE, current_deadline, False
        return EarningsExitStatus.NOT_APPLICABLE, None, False
    if prior_deadline is not None:
        return EarningsExitStatus.NOT_DUE, prior_deadline, False
    return EarningsExitStatus.NOT_APPLICABLE, None, False


def _protective_classification(
    decision: ProtectiveExitDecision,
) -> tuple[
    tuple[OpenPositionExitReason, ...],
    OpenPositionExitReason | None,
    ExitBoundary | None,
    IntrabarAmbiguityStatus,
]:
    action = decision.action
    if action is ProtectiveExitAction.HOLD:
        return (), None, None, IntrabarAmbiguityStatus.NOT_AMBIGUOUS
    if action is ProtectiveExitAction.GAP_STOP_EXIT:
        reason = (
            OpenPositionExitReason.GAP_THROUGH_STOP
            if decision.open < decision.effective_stop
            else OpenPositionExitReason.STOP_LOSS
        )
        return (
            (reason,),
            reason,
            ExitBoundary.OPEN,
            IntrabarAmbiguityStatus.NOT_AMBIGUOUS,
        )
    if action is ProtectiveExitAction.GAP_TAKE_PROFIT_EXIT:
        reason = (
            OpenPositionExitReason.GAP_THROUGH_TARGET
            if decision.open > decision.effective_take_profit
            else OpenPositionExitReason.TAKE_PROFIT
        )
        return (
            (reason,),
            reason,
            ExitBoundary.OPEN,
            IntrabarAmbiguityStatus.NOT_AMBIGUOUS,
        )
    if action is ProtectiveExitAction.STOP_LOSS_EXIT:
        ambiguous = decision.ambiguous_both_hit is True
        reasons = (
            (
                OpenPositionExitReason.STOP_LOSS,
                OpenPositionExitReason.TAKE_PROFIT,
            )
            if ambiguous
            else (OpenPositionExitReason.STOP_LOSS,)
        )
        return (
            reasons,
            OpenPositionExitReason.STOP_LOSS,
            ExitBoundary.INTRADAY,
            (
                IntrabarAmbiguityStatus.STOP_AND_TARGET_TOUCHED
                if ambiguous
                else IntrabarAmbiguityStatus.NOT_AMBIGUOUS
            ),
        )
    if action is ProtectiveExitAction.TAKE_PROFIT_EXIT:
        return (
            (OpenPositionExitReason.TAKE_PROFIT,),
            OpenPositionExitReason.TAKE_PROFIT,
            ExitBoundary.INTRADAY,
            IntrabarAmbiguityStatus.NOT_AMBIGUOUS,
        )
    raise OpenPositionExitValidationError(
        "UNRESOLVED_CORPORATE_ACTION_PREREQUISITE",
        "Phase 10 could not establish transformed protective levels for session T",
    )


def _prerequisite_status(
    evaluation: OpenPositionExitEvaluationInput, *, exit_required: bool
) -> ExitPrerequisiteStatus:
    """Downstream persistence readiness of a terminal decision.

    Task 5C-C removed the only clause this function ever had -- the
    same-session entry-to-exit persistence limitation -- once Phase 13 gained
    round-trip persistence.  Every other Phase 15B prerequisite (unresolved
    corporate actions, missing market data, chronology, holding limits,
    earnings authorization) is an error path elsewhere in this module and is
    unchanged.
    """

    del evaluation, exit_required
    return ExitPrerequisiteStatus.READY


@dataclass(frozen=True, slots=True, init=False, repr=False)
class OpenPositionExitEvaluator:
    """Stateless deterministic coordinator for one long position and D1 bar."""

    _trading_calendar: OpenPositionExitCalendar
    _protective_exit_service: ProtectiveExitService

    def __init__(self, *, trading_calendar: OpenPositionExitCalendar) -> None:
        object.__setattr__(self, "_trading_calendar", trading_calendar)
        object.__setattr__(
            self,
            "_protective_exit_service",
            ProtectiveExitService(trading_calendar=trading_calendar),
        )

    def evaluate(
        self, evaluation: OpenPositionExitEvaluationInput
    ) -> OpenPositionExitDecision:
        """Evaluate OPEN, then INTRADAY, then CLOSE without mutating inputs."""

        if not isinstance(evaluation, OpenPositionExitEvaluationInput):
            raise OpenPositionExitValidationError(
                "INVALID_EVALUATION_INPUT",
                "evaluation must be an OpenPositionExitEvaluationInput",
            )
        _expected_protective_session(evaluation, self._trading_calendar)
        holding_session_number = (
            _calendar_session_distance(
                self._trading_calendar,
                evaluation.protective_state.entry_session,
                evaluation.session,
            )
            + 1
        )
        if holding_session_number > MAX_HOLDING_SESSIONS:
            raise OpenPositionExitValidationError(
                "MAX_HOLDING_DEADLINE_MISSED",
                "an exposed position cannot be deliberately carried beyond session 10",
            )
        if evaluation.market_bar is None:
            raise OpenPositionExitValidationError(
                "MISSING_MARKET_DATA",
                "an exposed position requires an authoritative executable D1 bar",
            )

        earnings_status, earnings_deadline, earnings_due = (
            _earnings_authorization(
                evaluation=evaluation,
                calendar=self._trading_calendar,
            )
        )
        try:
            protective = self._protective_exit_service.evaluate_session(
                state=evaluation.protective_state,
                bar=evaluation.market_bar,
                corporate_actions=evaluation.corporate_actions,
            )
        except ProtectiveExitValidationError as error:
            raise OpenPositionExitValidationError(
                "PROTECTIVE_EXIT_EVALUATION_FAILED",
                f"Phase 10 rejected the supplied session: {error.code}",
            ) from error

        (
            protective_reasons,
            protective_selected,
            protective_boundary,
            ambiguity_status,
        ) = _protective_classification(protective)

        if protective.action in _PROTECTIVE_ACTIONS:
            if earnings_due and earnings_status is not EarningsExitStatus.DEADLINE_MISSED:
                earnings_status = EarningsExitStatus.AVOIDED_BY_EARLIER_EXIT
            return OpenPositionExitDecision(
                security_id=evaluation.protective_state.security_id,
                session=evaluation.session,
                entry_session=evaluation.protective_state.entry_session,
                holding_session_number=holding_session_number,
                market_bar=evaluation.market_bar,
                stop_price=protective.effective_stop,
                take_profit_price=protective.effective_take_profit,
                triggered_reasons=protective_reasons,
                selected_reason=protective_selected,
                exit_boundary=protective_boundary,
                reference_exit_price=protective.reference_exit_price,
                final_execution_price=protective.execution_exit_price,
                exit_required=True,
                intrabar_ambiguity_status=ambiguity_status,
                earnings_exit_status=earnings_status,
                earnings_deadline_session=earnings_deadline,
                exit_prerequisite_status=_prerequisite_status(
                    evaluation, exit_required=True
                ),
                protective_exit_decision=protective,
                earnings_decision=evaluation.earnings_decision,
                prior_boundary_earnings_decision=(
                    evaluation.prior_boundary_earnings_decision
                ),
            )

        close_reasons: list[OpenPositionExitReason] = []
        if earnings_due:
            close_reasons.append(OpenPositionExitReason.EARNINGS_FORCED_EXIT)
        if holding_session_number == MAX_HOLDING_SESSIONS:
            close_reasons.append(OpenPositionExitReason.MAX_HOLDING)

        if close_reasons:
            selected = close_reasons[0]
            return OpenPositionExitDecision(
                security_id=evaluation.protective_state.security_id,
                session=evaluation.session,
                entry_session=evaluation.protective_state.entry_session,
                holding_session_number=holding_session_number,
                market_bar=evaluation.market_bar,
                stop_price=protective.effective_stop,
                take_profit_price=protective.effective_take_profit,
                triggered_reasons=tuple(close_reasons),
                selected_reason=selected,
                exit_boundary=ExitBoundary.CLOSE,
                reference_exit_price=evaluation.market_bar.close,
                final_execution_price=None,
                exit_required=True,
                intrabar_ambiguity_status=ambiguity_status,
                earnings_exit_status=earnings_status,
                earnings_deadline_session=earnings_deadline,
                exit_prerequisite_status=_prerequisite_status(
                    evaluation, exit_required=True
                ),
                protective_exit_decision=protective,
                earnings_decision=evaluation.earnings_decision,
                prior_boundary_earnings_decision=(
                    evaluation.prior_boundary_earnings_decision
                ),
            )

        return OpenPositionExitDecision(
            security_id=evaluation.protective_state.security_id,
            session=evaluation.session,
            entry_session=evaluation.protective_state.entry_session,
            holding_session_number=holding_session_number,
            market_bar=evaluation.market_bar,
            stop_price=protective.effective_stop,
            take_profit_price=protective.effective_take_profit,
            triggered_reasons=(),
            selected_reason=None,
            exit_boundary=None,
            reference_exit_price=None,
            final_execution_price=None,
            exit_required=False,
            intrabar_ambiguity_status=ambiguity_status,
            earnings_exit_status=earnings_status,
            earnings_deadline_session=earnings_deadline,
            exit_prerequisite_status=ExitPrerequisiteStatus.READY,
            protective_exit_decision=protective,
            earnings_decision=evaluation.earnings_decision,
            prior_boundary_earnings_decision=(
                evaluation.prior_boundary_earnings_decision
            ),
        )


__all__ = ["OpenPositionExitEvaluator"]
