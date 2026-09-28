"""Deterministic Phase 10 protective-exit fixtures."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.execution.entry import (
    ENTRY_SLIPPAGE_BPS,
    EntryExecutionDecision,
    EntryExecutionStatus,
)
from stock_swing_d1.execution.costs import (
    BacktestExecutionCostService,
    ExecutionCostSide,
    load_backtest_execution_cost_policy,
)
from stock_swing_d1.execution.protective_exit import ProtectiveExitService
from stock_swing_d1.models import CorporateActionEvent, DividendEvent, StockBar
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


SIGNAL_SESSION = date(2026, 8, 14)
ENTRY_SESSION = date(2026, 8, 17)
NEXT_SESSION = date(2026, 8, 18)
THIRD_SESSION = date(2026, 8, 19)
NEW_YORK = ZoneInfo("America/New_York")
SIGNAL_TIME = datetime.combine(SIGNAL_SESSION, time(16), tzinfo=NEW_YORK)
ENTRY_TIME = datetime.combine(ENTRY_SESSION, time(9, 30), tzinfo=NEW_YORK)


class ExitCalendar:
    def __init__(
        self,
        sessions: tuple[date, ...] = (
            SIGNAL_SESSION,
            ENTRY_SESSION,
            NEXT_SESSION,
            THIRD_SESSION,
        ),
    ) -> None:
        self.sessions = sessions
        self.calls: list[date] = []

    def next_session(self, session: date) -> date:
        self.calls.append(session)
        return self.sessions[self.sessions.index(session) + 1]


@pytest.fixture
def exit_calendar() -> ExitCalendar:
    return ExitCalendar()


@pytest.fixture
def service(exit_calendar) -> ProtectiveExitService:
    return ProtectiveExitService(trading_calendar=exit_calendar)


@pytest.fixture
def make_signal():
    def factory(
        *,
        security_id: str = "NORGATE:1001",
        symbol: str = "PH10",
        signal_session: date = SIGNAL_SESSION,
        signal_time: datetime = SIGNAL_TIME,
        planned_entry_session: date = ENTRY_SESSION,
        atr_fraction: float | None = 0.02,
        action: BaselineSignalAction = BaselineSignalAction.VALID_LONG_SIGNAL,
    ) -> BaselineSignalDecision:
        entry_allowed = action is BaselineSignalAction.VALID_LONG_SIGNAL
        earnings_action = (
            EarningsIntegrationAction.ENTRY_ALLOWED
            if entry_allowed
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
            adjusted_close=80.0,
            sma_20=79.0,
            sma_50=78.0,
            rsi_14=55.0,
            atr_14=(
                None if atr_fraction is None else 80.0 * atr_fraction
            ),
            atr_fraction=atr_fraction,
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
def make_entry():
    cost_service = BacktestExecutionCostService(
        policy=load_backtest_execution_cost_policy(
            Path(__file__).resolve().parents[3] / "config" / "costs.yaml"
        )
    )

    def factory(
        *,
        security_id: str = "NORGATE:1001",
        symbol: str = "PH10",
        signal_session: date = SIGNAL_SESSION,
        signal_time: datetime = SIGNAL_TIME,
        planned_entry_session: date = ENTRY_SESSION,
        execution_time: datetime = ENTRY_TIME,
        execution_price: float | None = 100.0,
        status: EntryExecutionStatus = EntryExecutionStatus.EXECUTED,
    ) -> EntryExecutionDecision:
        is_executed = status is EntryExecutionStatus.EXECUTED
        valid_requested_price = (
            not isinstance(execution_price, bool)
            and isinstance(execution_price, (int, float))
            and isfinite(execution_price)
            and execution_price > 0.0
        )
        construction_price = (
            float(execution_price) if valid_requested_price else 100.0
        )
        has_candidate = is_executed or status is (
            EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
        )
        reference_open = (
            None
            if not has_candidate
            else construction_price / 1.0005
        )
        slippage_amount = (
            None
            if reference_open is None
            else reference_open * (ENTRY_SLIPPAGE_BPS / 10_000.0)
        )
        earnings_decision = EarningsIntegrationDecision(
            action=EarningsIntegrationAction.PENDING_ENTRY_ALLOWED,
            earnings_state=None,
            risk_decision=None,
        )
        cost_quote = (
            cost_service.quote(
                side=ExecutionCostSide.BUY,
                quantity=12,
                fill_price=Decimal(str(construction_price)),
            )
            if has_candidate
            else None
        )
        cash_required = (
            None
            if cost_quote is None
            else float(cost_quote.notional + cost_quote.execution_cost)
        )
        decision = EntryExecutionDecision(
            security_id=security_id,
            symbol=symbol,
            signal_session=signal_session,
            signal_time=signal_time,
            planned_entry_session=planned_entry_session,
            execution_time=execution_time,
            earnings_revalidation_action=(
                EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
            ),
            earnings_reason=None,
            earnings_decision=earnings_decision,
            cash_available=10_000.0,
            requested_shares=12,
            executed_shares=12 if is_executed else 0,
            reference_open=reference_open,
            slippage_bps=ENTRY_SLIPPAGE_BPS,
            slippage_amount=slippage_amount,
            candidate_execution_price=(
                construction_price if has_candidate else None
            ),
            candidate_execution_cost_quote=cost_quote,
            candidate_cash_required=cash_required,
            execution_price=construction_price if is_executed else None,
            execution_cost_quote=cost_quote if is_executed else None,
            actual_cash_required=cash_required if is_executed else None,
            status=status,
        )
        if is_executed and not valid_requested_price:
            object.__setattr__(decision, "execution_price", execution_price)
        return decision

    return factory


@pytest.fixture
def make_bar():
    def factory(
        *,
        security_id: str = "NORGATE:1001",
        symbol: str = "PH10",
        trading_date: date = ENTRY_SESSION,
        open: float = 100.0,
        high: float = 105.0,
        low: float = 97.0,
        close: float = 101.0,
    ) -> StockBar:
        return StockBar(
            security_id=security_id,
            symbol=symbol,
            trading_date=trading_date,
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
def make_capital_event():
    def factory(
        *,
        security_id: str = "NORGATE:1001",
        symbol: str = "PH10",
        event_date: date = ENTRY_SESSION,
        date_semantics: str = "entitlement_close",
        event_type: str = "split",
        terms_verified: bool = True,
        new_shares: float | None = 2.0,
        old_shares: float | None = 1.0,
        source_asset_id: int = 1001,
    ) -> CorporateActionEvent:
        return CorporateActionEvent(
            security_id=security_id,
            symbol=symbol,
            event_date=event_date,
            date_semantics=date_semantics,
            event_type=event_type,
            terms_verified=terms_verified,
            new_shares=new_shares,
            old_shares=old_shares,
            source_provider="Norgate Data",
            source_asset_id=source_asset_id,
        )

    return factory


@pytest.fixture
def make_dividend():
    def factory(
        *,
        security_id: str = "NORGATE:1001",
        symbol: str = "PH10",
        entitlement_date: date = ENTRY_SESSION,
        source_asset_id: int = 1001,
    ) -> DividendEvent:
        return DividendEvent(
            security_id=security_id,
            symbol=symbol,
            entitlement_date=entitlement_date,
            date_semantics="entitlement_close",
            dividend_type="ordinary_cash",
            amount_per_share=1.25,
            currency="USD",
            source_provider="Norgate Data",
            source_asset_id=source_asset_id,
            source_adjustment_mode="CAPITALSPECIAL",
        )

    return factory


@pytest.fixture
def initial_state(service, make_signal, make_entry):
    return service.create_state(
        signal=make_signal(), entry_execution=make_entry()
    )


@pytest.fixture
def held_state(service, initial_state, make_bar):
    decision = service.evaluate_session(state=initial_state, bar=make_bar())
    assert decision.resulting_state is not None
    return decision.resulting_state
