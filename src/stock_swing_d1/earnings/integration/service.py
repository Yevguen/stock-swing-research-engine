"""Published earnings query to strategy-risk orchestration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

from stock_swing_d1.earnings.integration.models import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
    EarningsRiskTradingCalendar,
)
from stock_swing_d1.earnings.models import EarningsValidationError
from stock_swing_d1.earnings.query import PublishedEarningsPITQuery
from stock_swing_d1.earnings.risk import (
    evaluate_entry_blackout,
    evaluate_open_position_earnings_risk,
    revalidate_pending_entry as revalidate_pending_entry_risk,
)


_MAX_HOLDING_SESSIONS: Final[int] = 10


def _require_aware_timestamp(value: datetime, field_name: str) -> None:
    try:
        is_aware = (
            isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        )
    except Exception:
        is_aware = False
    if not is_aware:
        raise EarningsValidationError(
            "INVALID_INTEGRATION_TIMESTAMP",
            f"{field_name} must be an explicitly timezone-aware datetime",
        )


def _require_session(value: date, field_name: str) -> None:
    if type(value) is not date:
        raise EarningsValidationError(
            "INVALID_INTEGRATION_SESSION",
            f"{field_name} must be a genuine date",
        )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class PublishedEarningsRiskOverlay:
    """Immutable orchestration view over one verified published PIT query."""

    _query: PublishedEarningsPITQuery
    _provider_name: str
    _trading_calendar: EarningsRiskTradingCalendar

    def __init__(
        self,
        query: PublishedEarningsPITQuery,
        *,
        provider_name: str,
        trading_calendar: EarningsRiskTradingCalendar,
    ) -> None:
        if (
            not isinstance(provider_name, str)
            or not provider_name
            or provider_name != provider_name.strip()
        ):
            raise EarningsValidationError(
                "INVALID_QUERY_PROVIDER",
                "provider_name must be an explicit non-empty provider value",
            )
        object.__setattr__(self, "_query", query)
        object.__setattr__(self, "_provider_name", provider_name)
        object.__setattr__(self, "_trading_calendar", trading_calendar)

    def __repr__(self) -> str:
        return (
            "PublishedEarningsRiskOverlay("
            f"build_id={self.build_id!r}, output_sha256={self.output_sha256!r}, "
            f"provider_name={self._provider_name!r})"
        )

    @property
    def build_id(self) -> str:
        """Build identity delegated from the exact bound query snapshot."""

        return self._query.build_id

    @property
    def output_sha256(self) -> str:
        """Artifact identity delegated from the exact bound query snapshot."""

        return self._query.output_sha256

    @property
    def max_holding_sessions(self) -> int:
        """The frozen Phase 6C.8B holding-horizon value."""

        return _MAX_HOLDING_SESSIONS

    def evaluate_entry_candidate(
        self,
        *,
        canonical_asset_id: str,
        signal_time: datetime,
        signal_session: date,
        planned_entry_session: date,
    ) -> EarningsIntegrationDecision:
        """Evaluate a completed-session signal for ordinary T+1 execution."""

        _require_aware_timestamp(signal_time, "signal_time")
        _require_session(signal_session, "signal_session")
        _require_session(planned_entry_session, "planned_entry_session")
        next_session = self._trading_calendar.next_session(signal_session)
        if planned_entry_session != next_session:
            raise EarningsValidationError(
                "INVALID_ENTRY_SESSION",
                "planned_entry_session must equal the next trading session",
            )
        if signal_time < self._trading_calendar.decision_time(signal_session):
            raise EarningsValidationError(
                "INCOMPLETE_SIGNAL_SESSION",
                "signal_time must represent a completed signal-session context",
            )

        state = self._query.query(
            canonical_asset_id=canonical_asset_id,
            provider_name=self._provider_name,
            as_of=signal_time,
            decision_session=planned_entry_session,
        )
        risk = evaluate_entry_blackout(
            state,
            planned_entry_session,
            self._trading_calendar,
            max_holding_sessions=_MAX_HOLDING_SESSIONS,
        )
        action = (
            EarningsIntegrationAction.ENTRY_BLOCKED
            if risk.entry_blackout
            else EarningsIntegrationAction.ENTRY_ALLOWED
        )
        return EarningsIntegrationDecision(action, state, risk, risk.risk_reason)

    def revalidate_pending_entry(
        self,
        *,
        canonical_asset_id: str,
        signal_time: datetime,
        execution_time: datetime,
        planned_entry_session: date,
    ) -> EarningsIntegrationDecision:
        """Freshly revalidate a pending T+1 entry immediately before execution."""

        _require_aware_timestamp(signal_time, "signal_time")
        _require_aware_timestamp(execution_time, "execution_time")
        _require_session(planned_entry_session, "planned_entry_session")
        if execution_time < signal_time:
            raise EarningsValidationError(
                "INVALID_REVALIDATION_TIME",
                "execution_time cannot precede signal_time",
            )

        state = self._query.query(
            canonical_asset_id=canonical_asset_id,
            provider_name=self._provider_name,
            as_of=execution_time,
            decision_session=planned_entry_session,
        )
        risk = revalidate_pending_entry_risk(
            signal_time,
            planned_entry_session,
            state,
            self._trading_calendar,
        )
        action = (
            EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
            if risk.pending_entry_invalidated
            else EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
        )
        return EarningsIntegrationDecision(action, state, risk, risk.risk_reason)

    def evaluate_open_position(
        self,
        *,
        canonical_asset_id: str,
        as_of: datetime,
        current_session: date,
        position_entry_session: date | None,
        position_is_open: bool = True,
        close_already_scheduled: bool = False,
    ) -> EarningsIntegrationDecision:
        """Reevaluate the current PIT earnings deadline for one position."""

        if not position_is_open:
            return EarningsIntegrationDecision(
                EarningsIntegrationAction.POSITION_ALREADY_CLOSED,
                None,
                None,
                "POSITION_ALREADY_CLOSED",
            )

        _require_aware_timestamp(as_of, "as_of")
        _require_session(current_session, "current_session")
        if position_entry_session is not None:
            _require_session(position_entry_session, "position_entry_session")

        state = self._query.query(
            canonical_asset_id=canonical_asset_id,
            provider_name=self._provider_name,
            as_of=as_of,
            decision_session=current_session,
        )
        risk = evaluate_open_position_earnings_risk(
            state,
            current_session,
            self._trading_calendar,
            position_is_open=True,
            position_entry_session=position_entry_session,
        )
        deadline = risk.last_safe_exit_session
        if deadline is None or current_session < deadline:
            action = EarningsIntegrationAction.HOLD_POSITION
        elif close_already_scheduled:
            action = EarningsIntegrationAction.NO_ADDITIONAL_EXIT
        elif current_session == deadline:
            action = EarningsIntegrationAction.EXIT_REQUIRED_THIS_SESSION
        elif risk.unavoidable_earnings_exposure:
            action = EarningsIntegrationAction.UNAVOIDABLE_EARNINGS_EXPOSURE
        else:
            action = EarningsIntegrationAction.MISSED_EXIT_DEADLINE
        return EarningsIntegrationDecision(action, state, risk, risk.risk_reason)
