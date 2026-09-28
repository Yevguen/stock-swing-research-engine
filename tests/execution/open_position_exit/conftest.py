"""Deterministic Phase 15B fixtures over the real Phase 10 contract."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.earnings.models import (
    EarningsScheduleStatus,
    EarningsStateAsOf,
    LifecycleState,
    TimingClass,
)
from stock_swing_d1.earnings.risk import evaluate_open_position_earnings_risk
from stock_swing_d1.execution.open_position_exit import (
    OpenPositionExitEvaluationInput,
    OpenPositionExitEvaluator,
)
from stock_swing_d1.execution.protective_exit import ProtectiveExitState
from stock_swing_d1.models import StockBar


NEW_YORK = ZoneInfo("America/New_York")
SESSIONS = (
    date(2026, 8, 3),
    date(2026, 8, 4),
    date(2026, 8, 5),
    date(2026, 8, 6),
    date(2026, 8, 7),
    date(2026, 8, 10),
    date(2026, 8, 11),
    date(2026, 8, 12),
    date(2026, 8, 13),
    date(2026, 8, 14),
    date(2026, 8, 17),
)


class ExplicitTradingCalendar:
    """Tiny explicit canonical-session calendar used only by focused tests."""

    def __init__(self, sessions: tuple[date, ...] = SESSIONS) -> None:
        self.sessions = sessions

    def previous_session(self, session: date) -> date:
        index = self.sessions.index(session)
        if index == 0:
            raise ValueError("no previous session")
        return self.sessions[index - 1]

    def next_session(self, session: date) -> date:
        index = self.sessions.index(session)
        if index + 1 >= len(self.sessions):
            raise ValueError("no next session")
        return self.sessions[index + 1]

    def session_distance(self, start: date, end: date) -> int:
        return self.sessions.index(end) - self.sessions.index(start)

    def decision_time(self, session: date) -> datetime:
        if session not in self.sessions:
            raise ValueError("not a canonical session")
        return datetime.combine(session, time(16), tzinfo=NEW_YORK)


@pytest.fixture
def calendar() -> ExplicitTradingCalendar:
    return ExplicitTradingCalendar()


@pytest.fixture
def evaluator(calendar) -> OpenPositionExitEvaluator:
    return OpenPositionExitEvaluator(trading_calendar=calendar)


@pytest.fixture
def make_state(calendar):
    def factory(
        *,
        session_index: int = 1,
        entry_session: date = SESSIONS[0],
        stop_price: float = 96.0,
        take_profit_price: float = 108.0,
    ) -> ProtectiveExitState:
        session = calendar.sessions[session_index]
        last_evaluated = (
            None
            if session == entry_session
            else calendar.previous_session(session)
        )
        return ProtectiveExitState._validated(
            security_id="NORGATE:1001",
            symbol="ACME",
            signal_session=date(2026, 7, 31),
            signal_time=datetime(2026, 7, 31, 16, tzinfo=NEW_YORK),
            entry_session=entry_session,
            entry_price=100.0,
            signal_atr_fraction=0.02,
            risk_fraction=0.04,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            last_evaluated_session=last_evaluated,
        )

    return factory


@pytest.fixture
def make_bar():
    def factory(
        *,
        session: date = SESSIONS[1],
        open: float = 100.0,
        high: float = 105.0,
        low: float = 97.0,
        close: float = 101.0,
    ) -> StockBar:
        return StockBar(
            security_id="NORGATE:1001",
            symbol="ACME",
            trading_date=session,
            timeframe="D1",
            session_type="regular",
            currency="USD",
            price_basis="unadjusted",
            open=open,
            high=high,
            low=low,
            close=close,
            volume=1_000_000,
        )

    return factory


@pytest.fixture
def make_earnings_decision(calendar):
    def factory(
        *,
        boundary_session: date,
        scheduled_date: date | None,
        timing_class: TimingClass = TimingClass.UNKNOWN,
        knowledge_effective_at: datetime | None = None,
        entry_session: date = SESSIONS[0],
        earnings_schedule_known: bool = True,
        lifecycle_state: LifecycleState = LifecycleState.CONFIRMED,
        schedule_status: EarningsScheduleStatus = EarningsScheduleStatus.KNOWN_EVENT,
    ) -> EarningsIntegrationDecision:
        state = EarningsStateAsOf(
            as_of=calendar.decision_time(boundary_session),
            canonical_asset_id="NORGATE:1001",
            event_instance_id="earnings-1",
            earnings_schedule_known=earnings_schedule_known,
            lifecycle_state=lifecycle_state,
            scheduled_date=scheduled_date,
            timing_class=timing_class,
            scheduled_at=None,
            knowledge_effective_at=(
                knowledge_effective_at
                if knowledge_effective_at is not None
                else calendar.decision_time(SESSIONS[0])
            ),
            schedule_status=schedule_status,
        )
        risk = evaluate_open_position_earnings_risk(
            state,
            boundary_session,
            calendar,
            position_entry_session=entry_session,
        )
        deadline = risk.last_safe_exit_session
        if deadline is None or boundary_session < deadline:
            action = EarningsIntegrationAction.HOLD_POSITION
        elif boundary_session == deadline:
            action = EarningsIntegrationAction.EXIT_REQUIRED_THIS_SESSION
        elif risk.unavoidable_earnings_exposure:
            action = EarningsIntegrationAction.UNAVOIDABLE_EARNINGS_EXPOSURE
        else:
            action = EarningsIntegrationAction.MISSED_EXIT_DEADLINE
        return EarningsIntegrationDecision(action, state, risk, risk.risk_reason)

    return factory


@pytest.fixture
def make_input(make_state, make_bar):
    def factory(
        *,
        session_index: int = 1,
        state: ProtectiveExitState | None = None,
        bar: StockBar | None | object = ...,
        earnings_decision: EarningsIntegrationDecision | None = None,
        prior_boundary_earnings_decision: EarningsIntegrationDecision | None = None,
    ) -> OpenPositionExitEvaluationInput:
        session = SESSIONS[session_index]
        actual_state = state or make_state(session_index=session_index)
        actual_bar = make_bar(session=session) if bar is ... else bar
        return OpenPositionExitEvaluationInput(
            session=session,
            protective_state=actual_state,
            market_bar=actual_bar,
            earnings_decision=earnings_decision,
            prior_boundary_earnings_decision=prior_boundary_earnings_decision,
        )

    return factory
