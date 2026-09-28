"""Immutable public models for Phase 15B open-position exit decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite
from typing import Protocol

from stock_swing_d1.earnings.integration import EarningsIntegrationDecision
from stock_swing_d1.execution.protective_exit import (
    ProtectiveExitAction,
    ProtectiveExitDecision,
    ProtectiveExitState,
)
from stock_swing_d1.execution.protective_exit.models import ProtectiveCorporateAction
from stock_swing_d1.models import StockBar


OPEN_POSITION_EXIT_INPUT_SCHEMA_VERSION = "open_position_exit_input.v0.1"
# Bumped v0.1 -> v0.2 (Task 5C-C): the v0.1 invariant "a same-session
# entry-to-exit terminal decision must carry
# SAME_SESSION_BUY_SELL_PERSISTENCE_UNSUPPORTED" was retired together with
# that enum member once Phase 13 gained same-session round-trip persistence.
# Every other v0.1 field, validation and error path is unchanged.
OPEN_POSITION_EXIT_DECISION_SCHEMA_VERSION = "open_position_exit_decision.v0.2"
MAX_HOLDING_SESSIONS = 10


class OpenPositionExitValidationError(ValueError):
    """A public Phase 15B contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class OpenPositionExitCalendar(Protocol):
    """Authoritative calendar operations required by Phase 15B."""

    def previous_session(self, session: date) -> date:
        ...

    def next_session(self, session: date) -> date:
        ...

    def session_distance(self, start: date, end: date) -> int:
        ...

    def decision_time(self, session: date) -> datetime:
        ...


class ExitBoundary(StrEnum):
    """Executable D1 boundaries in their canonical chronological order."""

    OPEN = "OPEN"
    INTRADAY = "INTRADAY"
    CLOSE = "CLOSE"


class OpenPositionExitReason(StrEnum):
    """Complete Phase 15B v0.1 terminal reason set."""

    GAP_THROUGH_STOP = "GAP_THROUGH_STOP"
    GAP_THROUGH_TARGET = "GAP_THROUGH_TARGET"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    EARNINGS_FORCED_EXIT = "EARNINGS_FORCED_EXIT"
    MAX_HOLDING = "MAX_HOLDING"


class IntrabarAmbiguityStatus(StrEnum):
    """Whether one D1 bar leaves stop/target touch order unknowable."""

    NOT_AMBIGUOUS = "NOT_AMBIGUOUS"
    STOP_AND_TARGET_TOUCHED = "STOP_AND_TARGET_TOUCHED"


class EarningsExitStatus(StrEnum):
    """Auditable state of the PIT earnings-avoidance obligation."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_DUE = "NOT_DUE"
    EXIT_REQUIRED_THIS_SESSION = "EXIT_REQUIRED_THIS_SESSION"
    AVOIDED_BY_EARLIER_EXIT = "AVOIDED_BY_EARLIER_EXIT"
    DEADLINE_MISSED = "DEADLINE_MISSED"


class ExitPrerequisiteStatus(StrEnum):
    """Downstream persistence readiness of an economic exit decision.

    Task 5C-C retired ``SAME_SESSION_BUY_SELL_PERSISTENCE_UNSUPPORTED``: a
    same-session entry-to-exit terminal decision is persistence-ready once
    Phase 13 applies the round-trip SELL after its BUY.  The vocabulary and
    field are retained so a future prerequisite can be added under a new
    decision version.
    """

    READY = "READY"


def _is_positive_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
        and value > 0.0
    )


@dataclass(frozen=True, slots=True)
class OpenPositionExitEvaluationInput:
    """One explicit immutable input for an exposed long position on session T.

    ``protective_state`` remains Phase 10-owned.  Phase 15B passes it and the
    authoritative unadjusted D1 bar to Phase 10 instead of reconstructing any
    stop, target, gap, corporate-action, or slippage economics.  The current
    earnings decision is completed-T audit evidence; only the separately
    supplied prior-boundary decision can authorize an administrative T close.
    """

    session: date
    protective_state: ProtectiveExitState
    market_bar: StockBar | None
    corporate_actions: tuple[ProtectiveCorporateAction, ...] = ()
    earnings_decision: EarningsIntegrationDecision | None = None
    prior_boundary_earnings_decision: EarningsIntegrationDecision | None = None
    schema_version: str = OPEN_POSITION_EXIT_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPEN_POSITION_EXIT_INPUT_SCHEMA_VERSION:
            raise OpenPositionExitValidationError(
                "UNSUPPORTED_INPUT_SCHEMA",
                "schema_version must identify the frozen Phase 15B v0.1 input",
            )
        if type(self.session) is not date:
            raise OpenPositionExitValidationError(
                "INVALID_EXIT_SESSION", "session must be a genuine date"
            )
        if not isinstance(self.protective_state, ProtectiveExitState):
            raise OpenPositionExitValidationError(
                "INVALID_PROTECTIVE_STATE",
                "protective_state must be a Phase 10 ProtectiveExitState",
            )
        if self.market_bar is not None:
            if not isinstance(self.market_bar, StockBar):
                raise OpenPositionExitValidationError(
                    "INVALID_MARKET_BAR",
                    "market_bar must be a canonical unadjusted StockBar or None",
                )
            if (
                self.market_bar.security_id
                != self.protective_state.security_id
                or self.market_bar.trading_date != self.session
            ):
                raise OpenPositionExitValidationError(
                    "MARKET_BAR_IDENTITY_MISMATCH",
                    "market_bar must match the position security and session",
                )
        if type(self.corporate_actions) is not tuple:
            raise OpenPositionExitValidationError(
                "INVALID_CORPORATE_ACTIONS",
                "corporate_actions must be an immutable tuple",
            )
        if self.earnings_decision is not None and not isinstance(
            self.earnings_decision, EarningsIntegrationDecision
        ):
            raise OpenPositionExitValidationError(
                "INVALID_EARNINGS_DECISION",
                "earnings_decision must be an authoritative integration decision",
            )
        if self.prior_boundary_earnings_decision is not None and not isinstance(
            self.prior_boundary_earnings_decision, EarningsIntegrationDecision
        ):
            raise OpenPositionExitValidationError(
                "INVALID_PRIOR_BOUNDARY_EARNINGS_DECISION",
                "prior_boundary_earnings_decision must be authoritative earnings output",
            )


@dataclass(frozen=True, slots=True)
class OpenPositionExitDecision:
    """One terminal-or-hold Phase 15B decision with complete D1 audit data.

    ``exit_required`` means Phase 15B selected a terminal executable boundary
    within the evaluated session.  A future or newly observed missed earnings
    obligation can therefore coexist with ``exit_required=False``.
    """

    security_id: str
    session: date
    entry_session: date
    holding_session_number: int

    market_bar: StockBar
    stop_price: float
    take_profit_price: float

    triggered_reasons: tuple[OpenPositionExitReason, ...]
    selected_reason: OpenPositionExitReason | None
    exit_boundary: ExitBoundary | None
    reference_exit_price: float | None
    final_execution_price: float | None
    exit_required: bool

    intrabar_ambiguity_status: IntrabarAmbiguityStatus
    earnings_exit_status: EarningsExitStatus
    earnings_deadline_session: date | None
    exit_prerequisite_status: ExitPrerequisiteStatus

    protective_exit_decision: ProtectiveExitDecision
    earnings_decision: EarningsIntegrationDecision | None
    prior_boundary_earnings_decision: EarningsIntegrationDecision | None
    schema_version: str = OPEN_POSITION_EXIT_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPEN_POSITION_EXIT_DECISION_SCHEMA_VERSION:
            raise OpenPositionExitValidationError(
                "UNSUPPORTED_DECISION_SCHEMA",
                "schema_version must identify the frozen Phase 15B v0.2 decision",
            )
        if (
            not isinstance(self.security_id, str)
            or not self.security_id
            or self.security_id != self.security_id.strip()
            or type(self.session) is not date
            or type(self.entry_session) is not date
            or type(self.holding_session_number) is not int
            or self.holding_session_number < 1
            or self.holding_session_number > MAX_HOLDING_SESSIONS
            or self.entry_session > self.session
        ):
            raise OpenPositionExitValidationError(
                "INVALID_DECISION_IDENTITY",
                "decision identity and holding-session number must be canonical",
            )
        if type(self.exit_required) is not bool:
            raise OpenPositionExitValidationError(
                "INVALID_EXIT_REQUIRED_FLAG",
                "exit_required must be an explicit boolean",
            )
        protective = self.protective_exit_decision
        if not isinstance(protective, ProtectiveExitDecision):
            raise OpenPositionExitValidationError(
                "INVALID_NESTED_PROTECTIVE_DECISION",
                "protective_exit_decision must be authoritative Phase 10 output",
            )
        if (
            not isinstance(self.market_bar, StockBar)
            or self.market_bar.security_id != self.security_id
            or self.market_bar.trading_date != self.session
            or not _is_positive_finite_number(self.stop_price)
            or not _is_positive_finite_number(self.take_profit_price)
            or self.stop_price >= self.take_profit_price
        ):
            raise OpenPositionExitValidationError(
                "INVALID_DECISION_MARKET_AUDIT",
                "bar identity and effective protective levels must be valid",
            )
        if (
            protective.security_id != self.security_id
            or protective.session != self.session
            or protective.entry_session != self.entry_session
            or protective.open != self.market_bar.open
            or protective.high != self.market_bar.high
            or protective.low != self.market_bar.low
            or self.stop_price != protective.effective_stop
            or self.take_profit_price != protective.effective_take_profit
        ):
            raise OpenPositionExitValidationError(
                "PROTECTIVE_DECISION_AUDIT_MISMATCH",
                "Phase 15B identity, bar, and levels must match Phase 10 output",
            )
        if (
            type(self.triggered_reasons) is not tuple
            or len(set(self.triggered_reasons)) != len(self.triggered_reasons)
            or any(
                not isinstance(reason, OpenPositionExitReason)
                for reason in self.triggered_reasons
            )
        ):
            raise OpenPositionExitValidationError(
                "INVALID_TRIGGERED_REASONS",
                "triggered_reasons must be an immutable unique reason tuple",
            )
        if self.selected_reason is not None and not isinstance(
            self.selected_reason, OpenPositionExitReason
        ):
            raise OpenPositionExitValidationError(
                "INVALID_SELECTED_REASON",
                "selected_reason must use the frozen Phase 15B reason enum",
            )
        if self.exit_boundary is not None and not isinstance(
            self.exit_boundary, ExitBoundary
        ):
            raise OpenPositionExitValidationError(
                "INVALID_EXIT_BOUNDARY",
                "exit_boundary must use the frozen D1 boundary enum",
            )
        if not isinstance(
            self.intrabar_ambiguity_status, IntrabarAmbiguityStatus
        ):
            raise OpenPositionExitValidationError(
                "INVALID_AMBIGUITY_STATUS",
                "intrabar_ambiguity_status must use the frozen enum",
            )
        if not isinstance(self.earnings_exit_status, EarningsExitStatus):
            raise OpenPositionExitValidationError(
                "INVALID_EARNINGS_EXIT_STATUS",
                "earnings_exit_status must use the frozen enum",
            )
        if not isinstance(self.exit_prerequisite_status, ExitPrerequisiteStatus):
            raise OpenPositionExitValidationError(
                "INVALID_EXIT_PREREQUISITE",
                "exit_prerequisite_status must use the frozen enum",
            )
        for field_name, value in (
            ("earnings_decision", self.earnings_decision),
            (
                "prior_boundary_earnings_decision",
                self.prior_boundary_earnings_decision,
            ),
        ):
            if value is not None and not isinstance(
                value, EarningsIntegrationDecision
            ):
                raise OpenPositionExitValidationError(
                    "INVALID_EARNINGS_AUDIT_PROVENANCE",
                    f"{field_name} must be authoritative earnings-layer output",
                )
        terminal_fields_present = (
            self.selected_reason is not None
            and self.exit_boundary is not None
            and self.reference_exit_price is not None
        )
        if self.exit_required != terminal_fields_present:
            raise OpenPositionExitValidationError(
                "INVALID_TERMINAL_DECISION",
                "exit_required must agree with selected reason, boundary, and reference",
            )
        if self.exit_required:
            if (
                self.selected_reason not in self.triggered_reasons
                or not _is_positive_finite_number(self.reference_exit_price)
            ):
                raise OpenPositionExitValidationError(
                    "INVALID_TERMINAL_DECISION",
                    "the selected reason and positive reference price must be auditable",
                )
        elif (
            self.triggered_reasons
            or self.selected_reason is not None
            or self.exit_boundary is not None
            or self.reference_exit_price is not None
            or self.final_execution_price is not None
        ):
            raise OpenPositionExitValidationError(
                "INVALID_HOLD_DECISION",
                "a hold decision cannot contain terminal exit fields",
            )
        if self.final_execution_price is not None and not _is_positive_finite_number(
            self.final_execution_price
        ):
            raise OpenPositionExitValidationError(
                "INVALID_FINAL_EXECUTION_PRICE",
                "final_execution_price must be positive and finite when present",
            )

        expected_ambiguity = (
            IntrabarAmbiguityStatus.STOP_AND_TARGET_TOUCHED
            if protective.ambiguous_both_hit is True
            else IntrabarAmbiguityStatus.NOT_AMBIGUOUS
        )
        if self.intrabar_ambiguity_status is not expected_ambiguity:
            raise OpenPositionExitValidationError(
                "INVALID_AMBIGUITY_DECISION",
                "Phase 15B ambiguity status must preserve Phase 10 audit output",
            )

        allowed_boundaries = {
            OpenPositionExitReason.GAP_THROUGH_STOP: (ExitBoundary.OPEN,),
            OpenPositionExitReason.GAP_THROUGH_TARGET: (ExitBoundary.OPEN,),
            OpenPositionExitReason.STOP_LOSS: (
                ExitBoundary.OPEN,
                ExitBoundary.INTRADAY,
            ),
            OpenPositionExitReason.TAKE_PROFIT: (
                ExitBoundary.OPEN,
                ExitBoundary.INTRADAY,
            ),
            OpenPositionExitReason.EARNINGS_FORCED_EXIT: (ExitBoundary.CLOSE,),
            OpenPositionExitReason.MAX_HOLDING: (ExitBoundary.CLOSE,),
        }
        if self.exit_required and self.exit_boundary not in allowed_boundaries[
            self.selected_reason
        ]:
            raise OpenPositionExitValidationError(
                "REASON_BOUNDARY_MISMATCH",
                "the selected reason is incompatible with the executable boundary",
            )

        if self.exit_required:
            if self.intrabar_ambiguity_status is (
                IntrabarAmbiguityStatus.STOP_AND_TARGET_TOUCHED
            ):
                canonical_reasons = (
                    OpenPositionExitReason.STOP_LOSS,
                    OpenPositionExitReason.TAKE_PROFIT,
                )
                canonical_selected = OpenPositionExitReason.STOP_LOSS
                canonical_boundary = ExitBoundary.INTRADAY
            elif self.selected_reason is OpenPositionExitReason.EARNINGS_FORCED_EXIT:
                canonical_reasons = (
                    (
                        OpenPositionExitReason.EARNINGS_FORCED_EXIT,
                        OpenPositionExitReason.MAX_HOLDING,
                    )
                    if OpenPositionExitReason.MAX_HOLDING in self.triggered_reasons
                    else (OpenPositionExitReason.EARNINGS_FORCED_EXIT,)
                )
                canonical_selected = OpenPositionExitReason.EARNINGS_FORCED_EXIT
                canonical_boundary = ExitBoundary.CLOSE
            else:
                canonical_reasons = (self.selected_reason,)
                canonical_selected = self.selected_reason
                canonical_boundary = self.exit_boundary
            if (
                self.triggered_reasons != canonical_reasons
                or self.selected_reason is not canonical_selected
                or self.exit_boundary is not canonical_boundary
            ):
                raise OpenPositionExitValidationError(
                    "NONCANONICAL_REASON_COMBINATION",
                    "triggered reasons, primary reason, and boundary must be canonical",
                )

        if (
            OpenPositionExitReason.MAX_HOLDING in self.triggered_reasons
            and self.holding_session_number != MAX_HOLDING_SESSIONS
        ):
            raise OpenPositionExitValidationError(
                "INVALID_MAX_HOLDING_TRIGGER",
                "MAX_HOLDING may trigger only on holding session 10",
            )

        protective_reasons = {
            OpenPositionExitReason.GAP_THROUGH_STOP,
            OpenPositionExitReason.GAP_THROUGH_TARGET,
            OpenPositionExitReason.STOP_LOSS,
            OpenPositionExitReason.TAKE_PROFIT,
        }
        if self.selected_reason in protective_reasons:
            valid_phase10_action = {
                (OpenPositionExitReason.GAP_THROUGH_STOP, ExitBoundary.OPEN): (
                    protective.action is ProtectiveExitAction.GAP_STOP_EXIT
                    and protective.open < protective.effective_stop
                ),
                (OpenPositionExitReason.GAP_THROUGH_TARGET, ExitBoundary.OPEN): (
                    protective.action is ProtectiveExitAction.GAP_TAKE_PROFIT_EXIT
                    and protective.open > protective.effective_take_profit
                ),
                (OpenPositionExitReason.STOP_LOSS, ExitBoundary.OPEN): (
                    protective.action is ProtectiveExitAction.GAP_STOP_EXIT
                    and protective.open == protective.effective_stop
                ),
                (OpenPositionExitReason.TAKE_PROFIT, ExitBoundary.OPEN): (
                    protective.action is ProtectiveExitAction.GAP_TAKE_PROFIT_EXIT
                    and protective.open == protective.effective_take_profit
                ),
                (OpenPositionExitReason.STOP_LOSS, ExitBoundary.INTRADAY): (
                    protective.action is ProtectiveExitAction.STOP_LOSS_EXIT
                ),
                (OpenPositionExitReason.TAKE_PROFIT, ExitBoundary.INTRADAY): (
                    protective.action is ProtectiveExitAction.TAKE_PROFIT_EXIT
                ),
            }.get((self.selected_reason, self.exit_boundary), False)
            if not valid_phase10_action:
                raise OpenPositionExitValidationError(
                    "PROTECTIVE_REASON_PROVENANCE_MISMATCH",
                    "the protective reason must be supported by the Phase 10 action",
                )
            if (
                self.reference_exit_price != protective.reference_exit_price
                or self.final_execution_price != protective.execution_exit_price
            ):
                raise OpenPositionExitValidationError(
                    "PROTECTIVE_EXECUTION_PROVENANCE_MISMATCH",
                    "protective prices must equal the authoritative Phase 10 result",
                )
        elif self.exit_required:
            if protective.action is not ProtectiveExitAction.HOLD:
                raise OpenPositionExitValidationError(
                    "ADMINISTRATIVE_EXIT_BEFORE_PROTECTIVE_HOLD",
                    "a close exit requires survival through Phase 10",
                )
            if (
                self.reference_exit_price != self.market_bar.close
                or self.final_execution_price is not None
            ):
                raise OpenPositionExitValidationError(
                    "INVALID_ADMINISTRATIVE_EXECUTION_PRICE",
                    "administrative exits use T close and no Phase 15B final price",
                )
        elif protective.action is not ProtectiveExitAction.HOLD:
            raise OpenPositionExitValidationError(
                "INVALID_HOLD_PROTECTIVE_PROVENANCE",
                "a non-terminal Phase 15B decision requires a Phase 10 hold",
            )

        if self.earnings_deadline_session is not None and type(
            self.earnings_deadline_session
        ) is not date:
            raise OpenPositionExitValidationError(
                "INVALID_EARNINGS_DEADLINE",
                "earnings_deadline_session must be a genuine date when present",
            )
        if self.earnings_exit_status is EarningsExitStatus.NOT_APPLICABLE:
            if self.earnings_deadline_session is not None:
                raise OpenPositionExitValidationError(
                    "INVALID_EARNINGS_STATUS_DEADLINE",
                    "NOT_APPLICABLE cannot retain an earnings deadline",
                )
        elif self.earnings_deadline_session is None:
            raise OpenPositionExitValidationError(
                "INVALID_EARNINGS_STATUS_DEADLINE",
                "an applicable earnings status requires its historical deadline",
            )
        if (
            self.earnings_exit_status is not EarningsExitStatus.NOT_APPLICABLE
            and self.earnings_decision is None
            and self.prior_boundary_earnings_decision is None
        ):
            raise OpenPositionExitValidationError(
                "MISSING_EARNINGS_AUDIT_PROVENANCE",
                "an applicable earnings status requires current or prior PIT evidence",
            )
        if self.selected_reason is OpenPositionExitReason.EARNINGS_FORCED_EXIT:
            if (
                self.earnings_exit_status
                not in {
                    EarningsExitStatus.EXIT_REQUIRED_THIS_SESSION,
                    EarningsExitStatus.DEADLINE_MISSED,
                }
                or self.prior_boundary_earnings_decision is None
            ):
                raise OpenPositionExitValidationError(
                    "UNAUTHORIZED_EARNINGS_EXIT",
                    "an earnings exit requires explicit prior-boundary provenance",
                )
        elif self.earnings_exit_status is EarningsExitStatus.EXIT_REQUIRED_THIS_SESSION:
            raise OpenPositionExitValidationError(
                "UNSELECTED_REQUIRED_EARNINGS_EXIT",
                "an authorized on-time earnings close must select the earnings reason",
            )
        if self.earnings_exit_status is EarningsExitStatus.AVOIDED_BY_EARLIER_EXIT:
            if (
                self.selected_reason not in protective_reasons
                or self.prior_boundary_earnings_decision is None
            ):
                raise OpenPositionExitValidationError(
                    "INVALID_EARNINGS_AVOIDANCE_STATUS",
                    "planned earnings avoidance requires prior authorization and a protective exit",
                )
        # Task 5C-C: the same-session persistence limitation is retired; every
        # decision -- including a same-session entry-to-exit terminal one -- is
        # persistence-ready.  No other prerequisite condition exists.
        if self.exit_prerequisite_status is not ExitPrerequisiteStatus.READY:
            raise OpenPositionExitValidationError(
                "INVALID_EXIT_PREREQUISITE",
                "every Phase 15B v0.2 decision is persistence-ready",
            )


__all__ = [
    "EarningsExitStatus",
    "ExitBoundary",
    "ExitPrerequisiteStatus",
    "IntrabarAmbiguityStatus",
    "MAX_HOLDING_SESSIONS",
    "OPEN_POSITION_EXIT_DECISION_SCHEMA_VERSION",
    "OPEN_POSITION_EXIT_INPUT_SCHEMA_VERSION",
    "OpenPositionExitCalendar",
    "OpenPositionExitDecision",
    "OpenPositionExitEvaluationInput",
    "OpenPositionExitReason",
    "OpenPositionExitValidationError",
]
