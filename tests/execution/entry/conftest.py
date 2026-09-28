"""Deterministic Phase 9 entry-execution fixtures."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.execution.entry import EntryExecutionService
from stock_swing_d1.execution.costs import (
    BacktestExecutionCostService,
    load_backtest_execution_cost_policy,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


SIGNAL_SESSION = date(2026, 8, 14)
ENTRY_SESSION = date(2026, 8, 17)
FOLLOWING_SESSION = date(2026, 8, 18)
NEW_YORK = ZoneInfo("America/New_York")
SIGNAL_TIME = datetime(2026, 8, 14, 16, tzinfo=NEW_YORK)
ENTRY_OPEN_TIME = datetime(2026, 8, 17, 9, 30, tzinfo=NEW_YORK)
COST_POLICY_PATH = Path(__file__).resolve().parents[3] / "config" / "costs.yaml"


class EntryCalendar:
    def __init__(
        self,
        sessions: tuple[date, ...] = (
            SIGNAL_SESSION,
            ENTRY_SESSION,
            FOLLOWING_SESSION,
        ),
    ) -> None:
        self.sessions = sessions
        self.next_session_calls: list[date] = []
        self.open_time_calls: list[date] = []

    def next_session(self, session: date) -> date:
        self.next_session_calls.append(session)
        return self.sessions[self.sessions.index(session) + 1]

    def regular_session_open_time(self, session: date) -> datetime:
        self.open_time_calls.append(session)
        if session not in self.sessions:
            raise ValueError("not an execution test session")
        return datetime.combine(session, time(9, 30), tzinfo=NEW_YORK)


class RecordingEarningsOverlay:
    def __init__(
        self,
        action: EarningsIntegrationAction = (
            EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
        ),
        *,
        reason: str | None = None,
    ) -> None:
        self.action = action
        self.reason = reason
        self.calls: list[dict[str, object]] = []

    def revalidate_pending_entry(
        self,
        *,
        canonical_asset_id: str,
        signal_time: datetime,
        execution_time: datetime,
        planned_entry_session: date,
    ) -> EarningsIntegrationDecision:
        self.calls.append(
            {
                "canonical_asset_id": canonical_asset_id,
                "signal_time": signal_time,
                "execution_time": execution_time,
                "planned_entry_session": planned_entry_session,
            }
        )
        return EarningsIntegrationDecision(
            action=self.action,
            earnings_state=None,
            risk_decision=None,
            reason=self.reason,
        )


@pytest.fixture
def entry_calendar() -> EntryCalendar:
    return EntryCalendar()


@pytest.fixture
def execution_cost_service() -> BacktestExecutionCostService:
    return BacktestExecutionCostService(
        policy=load_backtest_execution_cost_policy(COST_POLICY_PATH)
    )


@pytest.fixture
def make_signal():
    def factory(
        *,
        security_id: str = "NORGATE:1001",
        symbol: str = "PH9",
        signal_session: date = SIGNAL_SESSION,
        signal_time: datetime = SIGNAL_TIME,
        planned_entry_session: date = ENTRY_SESSION,
        action: BaselineSignalAction = BaselineSignalAction.VALID_LONG_SIGNAL,
        adjusted_close: float = 100.0,
    ) -> BaselineSignalDecision:
        entry_allowed = action is BaselineSignalAction.VALID_LONG_SIGNAL
        earnings_action = (
            EarningsIntegrationAction.ENTRY_ALLOWED
            if entry_allowed
            else EarningsIntegrationAction.ENTRY_BLOCKED
        )
        earnings_decision = EarningsIntegrationDecision(
            earnings_action, None, None
        )
        return BaselineSignalDecision(
            security_id=security_id,
            symbol=symbol,
            signal_session=signal_session,
            signal_time=signal_time,
            planned_entry_session=planned_entry_session,
            adjusted_close=adjusted_close,
            sma_20=102.0,
            sma_50=100.0,
            rsi_14=55.0,
            atr_14=2.0,
            atr_fraction=0.02,
            universe_eligible=True,
            close_above_sma50=True,
            sma20_above_sma50=True,
            rsi_above_50=True,
            atr_above_minimum=True,
            earnings_entry_allowed=entry_allowed,
            earnings_action=earnings_action,
            earnings_reason=None,
            earnings_decision=earnings_decision,
            action=action,
        )

    return factory


@pytest.fixture
def make_execution_bar():
    def factory(
        *,
        security_id: str = "NORGATE:1001",
        symbol: str = "PH9",
        trading_date: date = ENTRY_SESSION,
        opening_price: float = 100.0,
        high: float | None = None,
        low: float | None = None,
        close: float | None = None,
    ) -> StockBar:
        resolved_high = high if high is not None else opening_price + 2.0
        resolved_low = low if low is not None else opening_price * 0.5
        resolved_close = close if close is not None else opening_price + 1.0
        return StockBar(
            security_id=security_id,
            symbol=symbol,
            trading_date=trading_date,
            timeframe="D1",
            session_type="regular",
            currency="USD",
            price_basis="unadjusted",
            open=opening_price,
            high=resolved_high,
            low=resolved_low,
            close=resolved_close,
            volume=1_000_000,
        )

    return factory


@pytest.fixture
def service_factory(entry_calendar, execution_cost_service):
    def factory(
        action: EarningsIntegrationAction = (
            EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
        ),
        *,
        reason: str | None = None,
        overlay=None,
        calendar=None,
        cost_service=None,
    ):
        active_overlay = overlay or RecordingEarningsOverlay(
            action, reason=reason
        )
        active_calendar = calendar or entry_calendar
        active_cost_service = (
            execution_cost_service if cost_service is None else cost_service
        )
        service = EntryExecutionService(
            earnings_overlay=active_overlay,
            trading_calendar=active_calendar,
            execution_cost_service=active_cost_service,
        )
        return service, active_overlay, active_calendar

    return factory
