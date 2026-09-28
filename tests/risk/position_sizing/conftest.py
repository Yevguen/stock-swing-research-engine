"""Synthetic provider-neutral Phase 9/11 remediation fixtures."""

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
from stock_swing_d1.execution.protective_exit import ProtectiveExitService
from stock_swing_d1.models import StockBar
from stock_swing_d1.risk.position_sizing import (
    PortfolioSizingSnapshot,
    PositionSizingService,
)
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


SIGNAL_SESSION = date(2026, 8, 14)
ENTRY_SESSION = date(2026, 8, 17)
NEXT_SESSION = date(2026, 8, 18)
NEW_YORK = ZoneInfo("America/New_York")
SIGNAL_TIME = datetime.combine(SIGNAL_SESSION, time(16), tzinfo=NEW_YORK)
ENTRY_TIME = datetime.combine(ENTRY_SESSION, time(9, 30), tzinfo=NEW_YORK)
COST_POLICY_PATH = Path(__file__).resolve().parents[3] / "config" / "costs.yaml"


class TradingCalendar:
    def next_session(self, session: date) -> date:
        return {
            SIGNAL_SESSION: ENTRY_SESSION,
            ENTRY_SESSION: NEXT_SESSION,
        }[session]

    def regular_session_open_time(self, session: date) -> datetime:
        return datetime.combine(session, time(9, 30), tzinfo=NEW_YORK)


class EarningsOverlay:
    def __init__(
        self,
        action: EarningsIntegrationAction = (
            EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
        ),
    ) -> None:
        self.action = action

    def revalidate_pending_entry(self, **kwargs) -> EarningsIntegrationDecision:
        return EarningsIntegrationDecision(
            action=self.action,
            earnings_state=None,
            risk_decision=None,
        )


@pytest.fixture
def service() -> PositionSizingService:
    return PositionSizingService()


@pytest.fixture
def entry_service_factory():
    execution_cost_service = BacktestExecutionCostService(
        policy=load_backtest_execution_cost_policy(COST_POLICY_PATH)
    )

    def factory(
        action: EarningsIntegrationAction = (
            EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
        ),
    ) -> EntryExecutionService:
        return EntryExecutionService(
            earnings_overlay=EarningsOverlay(action),
            trading_calendar=TradingCalendar(),
            execution_cost_service=execution_cost_service,
        )

    return factory


@pytest.fixture
def protective_service() -> ProtectiveExitService:
    return ProtectiveExitService(trading_calendar=TradingCalendar())


@pytest.fixture
def make_signal():
    def factory(
        *,
        security_id: str = "SECURITY:1001",
        symbol: str = "PH11",
        signal_session: date = SIGNAL_SESSION,
        signal_time: datetime = SIGNAL_TIME,
        planned_entry_session: date = ENTRY_SESSION,
        adjusted_close: float = 75.0,
        atr_fraction: float | None = 0.02,
        action: BaselineSignalAction = BaselineSignalAction.VALID_LONG_SIGNAL,
    ) -> BaselineSignalDecision:
        allowed = action is BaselineSignalAction.VALID_LONG_SIGNAL
        earnings_action = (
            EarningsIntegrationAction.ENTRY_ALLOWED
            if allowed
            else EarningsIntegrationAction.ENTRY_BLOCKED
        )
        earnings_decision = EarningsIntegrationDecision(
            action=earnings_action,
            earnings_state=None,
            risk_decision=None,
        )
        return BaselineSignalDecision(
            security_id=security_id,
            symbol=symbol,
            signal_session=signal_session,
            signal_time=signal_time,
            planned_entry_session=planned_entry_session,
            adjusted_close=adjusted_close,
            sma_20=76.0,
            sma_50=74.0,
            rsi_14=55.0,
            atr_14=(
                None
                if atr_fraction is None
                else adjusted_close * atr_fraction
            ),
            atr_fraction=atr_fraction,
            universe_eligible=True,
            close_above_sma50=True,
            sma20_above_sma50=True,
            rsi_above_50=True,
            atr_above_minimum=True,
            earnings_entry_allowed=allowed,
            earnings_action=earnings_action,
            earnings_reason=None,
            earnings_decision=earnings_decision,
            action=action,
        )

    return factory


@pytest.fixture
def make_signal_bar():
    def factory(
        *,
        security_id: str = "SECURITY:1001",
        symbol: str = "PH11",
        trading_date: date = SIGNAL_SESSION,
        close: float = 100.0,
    ) -> StockBar:
        return StockBar(
            security_id=security_id,
            symbol=symbol,
            trading_date=trading_date,
            timeframe="D1",
            session_type="regular",
            currency="USD",
            price_basis="unadjusted",
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1_000_000,
        )

    return factory


@pytest.fixture
def make_execution_bar():
    def factory(
        *,
        security_id: str = "SECURITY:1001",
        symbol: str = "PH11",
        opening_price: float = 100.0,
    ) -> StockBar:
        return StockBar(
            security_id=security_id,
            symbol=symbol,
            trading_date=ENTRY_SESSION,
            timeframe="D1",
            session_type="regular",
            currency="USD",
            price_basis="unadjusted",
            open=opening_price,
            high=opening_price,
            low=opening_price,
            close=opening_price,
            volume=1_000_000,
        )

    return factory


@pytest.fixture
def make_completed_t(make_signal, make_signal_bar, entry_service_factory):
    def factory(
        *,
        adjusted_close: float = 75.0,
        unadjusted_close: float = 100.0,
        atr_fraction: float | None = 0.02,
        signal_overrides: dict[str, object] | None = None,
        bar_overrides: dict[str, object] | None = None,
    ):
        signal_values = dict(signal_overrides or {})
        signal = make_signal(
            adjusted_close=adjusted_close,
            atr_fraction=atr_fraction,
            **signal_values,
        )
        pending = entry_service_factory().create_pending_entry(signal=signal)
        bar_values = dict(bar_overrides or {})
        signal_bar = make_signal_bar(close=unadjusted_close, **bar_values)
        return signal, pending, signal_bar

    return factory


@pytest.fixture
def make_sizing(service, make_completed_t):
    def factory(
        *,
        equity: float = 10_000,
        cash: float = 10_000,
        adjusted_close: float = 75.0,
        unadjusted_close: float = 100.0,
        atr_fraction: float = 0.02,
    ):
        signal, pending, signal_bar = make_completed_t(
            adjusted_close=adjusted_close,
            unadjusted_close=unadjusted_close,
            atr_fraction=atr_fraction,
        )
        portfolio = PortfolioSizingSnapshot(
            portfolio_equity=equity,
            cash_available=cash,
        )
        sizing = service.size_pending_entry(
            signal=signal,
            pending_entry=pending,
            signal_bar=signal_bar,
            portfolio=portfolio,
        )
        return sizing, signal, pending, signal_bar, portfolio

    return factory


@pytest.fixture
def execute_sizing(entry_service_factory, make_execution_bar):
    def factory(
        sizing,
        pending,
        *,
        opening_price: float = 100.0,
        earnings_action: EarningsIntegrationAction = (
            EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
        ),
    ):
        assert sizing.sized_pending_entry is not None
        return entry_service_factory(earnings_action).execute_pending_entry(
            pending_entry=pending,
            sized_pending_entry=sizing.sized_pending_entry,
            execution_bar=make_execution_bar(opening_price=opening_price),
        )

    return factory
