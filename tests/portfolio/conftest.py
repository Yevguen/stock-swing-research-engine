"""Synthetic provider-neutral Phase 12 fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.execution.entry import (
    EntryExecutionDecision,
    EntryExecutionStatus,
    create_pending_entry,
)
from stock_swing_d1.execution.costs import (
    BacktestExecutionCostService,
    ExecutionCostSide,
    load_backtest_execution_cost_policy,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.portfolio import (
    OpenPosition,
    PortfolioAllocationService,
    PortfolioCandidate,
    PortfolioSnapshot,
)
from stock_swing_d1.risk.position_sizing import PositionSizingService
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


SIGNAL_SESSION = date(2026, 8, 14)
ENTRY_SESSION = date(2026, 8, 17)
SIGNAL_TIME = datetime.combine(SIGNAL_SESSION, time(20), tzinfo=timezone.utc)
DECISION_TIME = datetime.combine(
    SIGNAL_SESSION, time(20, 5), tzinfo=timezone.utc
)
ENTRY_TIME = datetime.combine(ENTRY_SESSION, time(13, 30), tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class NextSessionCalendar:
    signal_session: date
    entry_session: date

    def next_session(self, session: date) -> date:
        if session != self.signal_session:
            raise ValueError("unknown session")
        return self.entry_session


class RecordingSizingService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._service = PositionSizingService()

    def size_pending_entry(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._service.size_pending_entry(**kwargs)


@pytest.fixture
def make_open_position():
    def factory(
        security_id: str = "OPEN:1",
        *,
        symbol: str | None = None,
        shares: int = 10,
        entry_session: date = date(2026, 8, 10),
    ) -> OpenPosition:
        return OpenPosition(
            security_id=security_id,
            symbol=symbol or security_id.replace(":", ""),
            shares=shares,
            entry_session=entry_session,
        )

    return factory


@pytest.fixture
def make_portfolio():
    def factory(
        *,
        equity: float = 10_000,
        cash: float = 10_000,
        open_positions=(),
        allocation_session: date = SIGNAL_SESSION,
        decision_time: datetime | None = None,
    ) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            allocation_session=allocation_session,
            decision_time=decision_time or datetime.combine(
                allocation_session, time(20, 5), tzinfo=timezone.utc
            ),
            portfolio_equity=equity,
            cash_available=cash,
            open_positions=open_positions,
        )

    return factory


@pytest.fixture
def make_candidate():
    def factory(
        security_id: str = "NORGATE:100",
        *,
        symbol: str | None = None,
        signal_session: date = SIGNAL_SESSION,
        planned_entry_session: date | None = None,
        close: float = 100.0,
        atr_fraction: float = 0.02,
    ) -> PortfolioCandidate:
        entry_session = planned_entry_session or (
            signal_session + timedelta(days=3)
        )
        signal_time = datetime.combine(
            signal_session, time(20), tzinfo=timezone.utc
        )
        candidate_symbol = symbol or security_id.replace("NORGATE:", "S")
        earnings_decision = EarningsIntegrationDecision(
            action=EarningsIntegrationAction.ENTRY_ALLOWED,
            earnings_state=None,
            risk_decision=None,
        )
        signal = BaselineSignalDecision(
            security_id=security_id,
            symbol=candidate_symbol,
            signal_session=signal_session,
            signal_time=signal_time,
            planned_entry_session=entry_session,
            adjusted_close=close,
            sma_20=close + 1.0,
            sma_50=close - 1.0,
            rsi_14=55.0,
            atr_14=close * atr_fraction,
            atr_fraction=atr_fraction,
            universe_eligible=True,
            close_above_sma50=True,
            sma20_above_sma50=True,
            rsi_above_50=True,
            atr_above_minimum=True,
            earnings_entry_allowed=True,
            earnings_action=EarningsIntegrationAction.ENTRY_ALLOWED,
            earnings_reason=None,
            earnings_decision=earnings_decision,
            action=BaselineSignalAction.VALID_LONG_SIGNAL,
        )
        pending_entry = create_pending_entry(
            signal,
            trading_calendar=NextSessionCalendar(
                signal_session=signal_session,
                entry_session=entry_session,
            ),
        )
        signal_bar = StockBar(
            security_id=security_id,
            symbol=candidate_symbol,
            trading_date=signal_session,
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
        return PortfolioCandidate(
            signal=signal,
            pending_entry=pending_entry,
            signal_bar=signal_bar,
        )

    return factory


@pytest.fixture
def recording_service():
    sizing_service = RecordingSizingService()
    return PortfolioAllocationService(
        position_sizing_service=sizing_service
    ), sizing_service


@pytest.fixture
def admitted_decision(make_portfolio, make_candidate):
    allocation = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(),
        candidates=(make_candidate(),),
    )
    return allocation.candidate_decisions[0]


@pytest.fixture
def make_entry_execution():
    cost_service = BacktestExecutionCostService(
        policy=load_backtest_execution_cost_policy(
            Path(__file__).resolve().parents[2] / "config" / "costs.yaml"
        )
    )

    def factory(
        admitted,
        *,
        status: EntryExecutionStatus = EntryExecutionStatus.EXECUTED,
        actual_cash_required: float | None = 1_950.0,
    ) -> EntryExecutionDecision:
        sized = admitted.sized_pending_entry
        assert sized is not None
        executed = status is EntryExecutionStatus.EXECUTED
        cancelled = status is (
            EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
        )
        cost_quote = None
        execution_price = None
        candidate_price = None
        candidate_cash = None
        if executed:
            assert actual_cash_required is not None
            target = Decimal(str(actual_cash_required))
            quantity = Decimal(sized.fixed_shares)
            minimum_adjusted = max(
                quantity * Decimal("0.005"), Decimal("1.00")
            )
            notional = (target - minimum_adjusted) / Decimal("1.0001")
            if notional * Decimal("0.01") < minimum_adjusted:
                notional = target / Decimal("1.0101")
            execution_price = float(notional / quantity)
            candidate_price = execution_price
        elif cancelled:
            candidate_price = sized.cash_available / sized.fixed_shares
        if candidate_price is not None:
            cost_quote = cost_service.quote(
                side=ExecutionCostSide.BUY,
                quantity=sized.fixed_shares,
                fill_price=Decimal(str(candidate_price)),
            )
            candidate_cash = float(
                cost_quote.notional + cost_quote.execution_cost
            )
        actual_cash = candidate_cash if executed else None
        return EntryExecutionDecision(
            security_id=sized.security_id,
            symbol=sized.symbol,
            signal_session=sized.signal_session,
            signal_time=sized.signal_time,
            planned_entry_session=sized.planned_entry_session,
            execution_time=ENTRY_TIME,
            earnings_revalidation_action=(
                EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
            ),
            earnings_reason=None,
            earnings_decision=EarningsIntegrationDecision(
                action=EarningsIntegrationAction.PENDING_ENTRY_ALLOWED,
                earnings_state=None,
                risk_decision=None,
            ),
            cash_available=sized.cash_available,
            requested_shares=sized.fixed_shares,
            executed_shares=sized.fixed_shares if executed else 0,
            reference_open=candidate_price,
            slippage_bps=5.0,
            slippage_amount=0.0 if executed else None,
            candidate_execution_price=candidate_price,
            candidate_execution_cost_quote=cost_quote,
            candidate_cash_required=candidate_cash,
            execution_price=execution_price,
            execution_cost_quote=cost_quote if executed else None,
            actual_cash_required=actual_cash,
            status=status,
        )

    return factory
