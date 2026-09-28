"""Provider-neutral Phase 15A fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path

import pytest

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.execution.entry import create_pending_entry
from stock_swing_d1.execution.costs import (
    BacktestExecutionCostService,
    load_backtest_execution_cost_policy,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.portfolio import PortfolioCandidate, PortfolioSnapshot
from stock_swing_d1.ranking import (
    RankingCandidate,
    compute_candidate_input_fingerprint,
)
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


FRIDAY = date(2026, 8, 21)
MONDAY = date(2026, 8, 24)
TUESDAY = date(2026, 8, 25)
COST_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "costs.yaml"


@pytest.fixture
def execution_cost_service() -> BacktestExecutionCostService:
    return BacktestExecutionCostService(
        policy=load_backtest_execution_cost_policy(COST_POLICY_PATH)
    )


def decision_time(session: date) -> datetime:
    return datetime.combine(session, time(20), tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class ExplicitCalendar:
    current: date
    following: date

    def next_session(self, session: date) -> date:
        if session != self.current:
            raise ValueError("unknown session")
        return self.following


@pytest.fixture
def make_candidate_pair():
    def factory(
        security_id: str,
        *,
        session: date = FRIDAY,
        following: date = MONDAY,
        close: float = 100.0,
        sma20: float = 110.0,
        sma50: float = 100.0,
        atr14: float = 5.0,
        rsi14: float = 60.0,
    ) -> tuple[RankingCandidate, PortfolioCandidate]:
        signal_at = decision_time(session)
        symbol = security_id.replace("NORGATE:", "S")
        earnings = EarningsIntegrationDecision(
            action=EarningsIntegrationAction.ENTRY_ALLOWED,
            earnings_state=None,
            risk_decision=None,
        )
        signal = BaselineSignalDecision(
            security_id=security_id,
            symbol=symbol,
            signal_session=session,
            signal_time=signal_at,
            planned_entry_session=following,
            adjusted_close=close,
            sma_20=sma20,
            sma_50=sma50,
            rsi_14=rsi14,
            atr_14=atr14,
            atr_fraction=atr14 / close,
            universe_eligible=True,
            close_above_sma50=True,
            sma20_above_sma50=True,
            rsi_above_50=True,
            atr_above_minimum=True,
            earnings_entry_allowed=True,
            earnings_action=EarningsIntegrationAction.ENTRY_ALLOWED,
            earnings_reason=None,
            earnings_decision=earnings,
            action=BaselineSignalAction.VALID_LONG_SIGNAL,
        )
        pending = create_pending_entry(
            signal,
            trading_calendar=ExplicitCalendar(session, following),
        )
        bar = StockBar(
            security_id=security_id,
            symbol=symbol,
            trading_date=session,
            timeframe="D1",
            session_type="regular",
            currency="USD",
            price_basis="unadjusted",
            open=close,
            high=close + 1.0,
            low=close - 1.0,
            close=close,
            volume=1_000_000,
        )
        allocation_candidate = PortfolioCandidate(
            signal=signal,
            pending_entry=pending,
            signal_bar=bar,
        )
        fingerprint = compute_candidate_input_fingerprint(
            security_id=security_id,
            ranking_session=session,
            decision_time=signal_at,
            signal_session=session,
            signal_time=signal_at,
            sma20=sma20,
            sma50=sma50,
            atr14=atr14,
            rsi14=rsi14,
        )
        ranking_candidate = RankingCandidate(
            security_id=security_id,
            ranking_session=session,
            decision_time=signal_at,
            signal_session=session,
            signal_time=signal_at,
            sma20=sma20,
            sma50=sma50,
            atr14=atr14,
            rsi14=rsi14,
            input_fingerprint=fingerprint,
        )
        return ranking_candidate, allocation_candidate

    return factory


@pytest.fixture
def make_portfolio_snapshot():
    def factory(
        *,
        session: date = FRIDAY,
        cash: float = 10_000.0,
        equity: float = 10_000.0,
        open_positions=(),
    ) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            allocation_session=session,
            decision_time=decision_time(session),
            portfolio_equity=equity,
            cash_available=cash,
            open_positions=open_positions,
        )

    return factory
