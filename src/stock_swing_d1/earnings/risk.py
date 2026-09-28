"""Strategy-facing earnings risk and post-event session rules."""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

from stock_swing_d1.earnings.models import (
    EarningsRiskDecision,
    EarningsScheduleStatus,
    EarningsStateAsOf,
    LifecycleState,
    TimingClass,
)


class _TradingCalendar(Protocol):
    def previous_session(self, session: date) -> date:
        ...

    def next_session(self, session: date) -> date:
        ...

    def session_distance(self, start: date, end: date) -> int:
        ...

    def decision_time(self, session: date) -> datetime:
        ...


def _is_active(state: EarningsStateAsOf) -> bool:
    return (
        state.earnings_schedule_known
        and state.lifecycle_state in {LifecycleState.ESTIMATED, LifecycleState.CONFIRMED}
        and state.scheduled_date is not None
    )


def evaluate_entry_blackout(
    state: EarningsStateAsOf,
    prospective_entry_session: date,
    trading_calendar: _TradingCalendar,
    max_holding_sessions: int = 10,
) -> EarningsRiskDecision:
    """Block entries for active events in S0 through S(max_holding_sessions-1)."""

    if state.schedule_status is EarningsScheduleStatus.UNKNOWN:
        return EarningsRiskDecision(
            entry_blackout=True,
            risk_reason="EARNINGS_SCHEDULE_UNKNOWN",
        )
    if (
        state.schedule_status
        is EarningsScheduleStatus.NO_EVENT_WITHIN_PROVIDER_HORIZON
    ):
        return EarningsRiskDecision(
            entry_blackout=False,
            risk_reason="NO_EARNINGS_WITHIN_PROVIDER_HORIZON",
        )
    if not _is_active(state):
        return EarningsRiskDecision(risk_reason="NO_ACTIVE_EARNINGS_SCHEDULE")
    assert state.scheduled_date is not None
    try:
        distance = trading_calendar.session_distance(
            prospective_entry_session, state.scheduled_date
        )
    except (KeyError, ValueError):
        return EarningsRiskDecision(
            entry_blackout=True,
            risk_reason="EARNINGS_CALENDAR_EVALUATION_FAILED",
        )
    blocked = 0 <= distance < max_holding_sessions
    return EarningsRiskDecision(
        entry_blackout=blocked,
        risk_reason="EARNINGS_WITHIN_HOLDING_HORIZON" if blocked else None,
    )


def last_safe_exit_session(
    state: EarningsStateAsOf,
    trading_calendar: _TradingCalendar,
) -> date | None:
    """Return the final session allowed by the frozen timing rules."""

    if not _is_active(state):
        return None
    assert state.scheduled_date is not None
    if state.timing_class is TimingClass.AMC:
        return state.scheduled_date
    if state.timing_class in {
        TimingClass.BMO,
        TimingClass.DURING_MARKET,
        TimingClass.UNKNOWN,
    }:
        return trading_calendar.previous_session(state.scheduled_date)
    raise ValueError("EXACT_TIME requires market-hours classification before risk evaluation")


def evaluate_open_position_earnings_risk(
    state: EarningsStateAsOf,
    current_session: date,
    trading_calendar: _TradingCalendar,
    *,
    position_is_open: bool = True,
    position_entry_session: date | None = None,
) -> EarningsRiskDecision:
    """Evaluate an existing position without inventing a historical exit."""

    if not position_is_open:
        return EarningsRiskDecision(risk_reason="POSITION_ALREADY_CLOSED")
    holding_age = (
        trading_calendar.session_distance(position_entry_session, current_session)
        if position_entry_session is not None
        else None
    )
    deadline = last_safe_exit_session(state, trading_calendar)
    if deadline is None:
        return EarningsRiskDecision(
            risk_reason="NO_ACTIVE_EARNINGS_SCHEDULE",
            holding_age_sessions=holding_age,
        )
    if current_session > deadline:
        late_knowledge = (
            state.knowledge_effective_at is not None
            and state.knowledge_effective_at > trading_calendar.decision_time(deadline)
        )
        if late_knowledge:
            return EarningsRiskDecision(
                last_safe_exit_session=deadline,
                unavoidable_earnings_exposure=True,
                risk_reason="EARNINGS_KNOWN_AFTER_LAST_SAFE_EXIT",
                holding_age_sessions=holding_age,
            )
        return EarningsRiskDecision(
            last_safe_exit_session=deadline,
            unavoidable_earnings_exposure=False,
            risk_reason="LAST_SAFE_EXIT_MISSED",
            holding_age_sessions=holding_age,
        )
    return EarningsRiskDecision(
        mandatory_exit=True,
        last_safe_exit_session=deadline,
        risk_reason="MANDATORY_EARNINGS_EXIT",
        holding_age_sessions=holding_age,
    )


def revalidate_pending_entry(
    signal_time: datetime,
    planned_entry_session: date,
    earnings_state: EarningsStateAsOf,
    trading_calendar: _TradingCalendar,
) -> EarningsRiskDecision:
    """Cancel a pending entry when the current PIT state now blocks execution."""

    if signal_time.tzinfo is None or signal_time.utcoffset() is None:
        raise ValueError("signal_time must be timezone-aware")
    blackout = evaluate_entry_blackout(
        earnings_state, planned_entry_session, trading_calendar
    )
    effective_at = earnings_state.knowledge_effective_at
    intervened_before_execution = (
        effective_at is not None
        and signal_time < effective_at <= trading_calendar.decision_time(planned_entry_session)
    )
    fail_closed = blackout.risk_reason in {
        "EARNINGS_SCHEDULE_UNKNOWN",
        "EARNINGS_CALENDAR_EVALUATION_FAILED",
    }
    invalidated = blackout.entry_blackout and (
        intervened_before_execution or fail_closed
    )
    return EarningsRiskDecision(
        entry_blackout=blackout.entry_blackout,
        pending_entry_invalidated=invalidated,
        risk_reason="PENDING_ENTRY_INVALIDATED" if invalidated else blackout.risk_reason,
    )


def first_clean_post_event_session(
    event_session: date,
    timing_class: TimingClass,
    trading_calendar: _TradingCalendar,
) -> date:
    """Return the first D1 candle uncontaminated by the event timing."""

    if timing_class is TimingClass.BMO:
        return event_session
    if timing_class in {
        TimingClass.AMC,
        TimingClass.DURING_MARKET,
        TimingClass.UNKNOWN,
    }:
        return trading_calendar.next_session(event_session)
    raise ValueError("EXACT_TIME requires market-hours classification")


def earliest_post_event_entry_session(
    event_session: date,
    timing_class: TimingClass,
    trading_calendar: _TradingCalendar,
) -> date:
    """Enter only after one complete clean post-event candle exists."""

    clean_session = first_clean_post_event_session(
        event_session, timing_class, trading_calendar
    )
    return trading_calendar.next_session(clean_session)
