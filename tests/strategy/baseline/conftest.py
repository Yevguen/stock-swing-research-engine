"""Deterministic Phase 8 baseline strategy fixtures."""

from __future__ import annotations

from datetime import date, datetime, time, timezone

import pytest

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.indicators import TechnicalIndicatorRow
from stock_swing_d1.models import CorporateActionAdjustedStockBar
from stock_swing_d1.strategy.baseline import BaselineSignalEvaluator


CANONICAL_SIGNAL_TIME = datetime(2026, 8, 14, 20, tzinfo=timezone.utc)


class BaselineTradingCalendar:
    sessions = (
        date(2026, 8, 14),
        date(2026, 8, 17),
        date(2026, 8, 18),
    )

    def next_session(self, session: date) -> date:
        index = self.sessions.index(session)
        return self.sessions[index + 1]

    def signal_decision_time(self, session: date) -> datetime:
        if session not in self.sessions:
            raise ValueError("not a baseline test session")
        return datetime.combine(session, time(20), tzinfo=timezone.utc)


class RecordingEarningsOverlay:
    def __init__(
        self,
        action: EarningsIntegrationAction = EarningsIntegrationAction.ENTRY_ALLOWED,
        *,
        reason: str | None = None,
    ) -> None:
        self.action = action
        self.reason = reason
        self.calls: list[dict[str, object]] = []

    def evaluate_entry_candidate(
        self,
        *,
        canonical_asset_id: str,
        signal_time: datetime,
        signal_session: date,
        planned_entry_session: date,
    ) -> EarningsIntegrationDecision:
        self.calls.append(
            {
                "canonical_asset_id": canonical_asset_id,
                "signal_time": signal_time,
                "signal_session": signal_session,
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
def baseline_calendar() -> BaselineTradingCalendar:
    return BaselineTradingCalendar()


@pytest.fixture
def signal_time() -> datetime:
    """Expected internally derived canonical signal time for assertions."""

    return CANONICAL_SIGNAL_TIME


@pytest.fixture
def make_bar():
    def factory(
        *,
        security_id: str = "NORGATE:1001",
        symbol: str = "BASE",
        trading_date: date = date(2026, 8, 14),
        close: float = 105.0,
    ) -> CorporateActionAdjustedStockBar:
        return CorporateActionAdjustedStockBar(
            security_id=security_id,
            symbol=symbol,
            trading_date=trading_date,
            timeframe="D1",
            session_type="regular",
            currency="USD",
            price_basis="capital_special_adjusted",
            open=close,
            high=close + 2.0,
            low=max(0.01, close - 2.0),
            close=close,
            volume=1_000_000.0,
        )

    return factory


@pytest.fixture
def make_indicators():
    def factory(
        *,
        security_id: str = "NORGATE:1001",
        symbol: str = "BASE",
        trading_date: date = date(2026, 8, 14),
        sma_20: float | None = 102.0,
        sma_50: float | None = 100.0,
        rsi_14: float | None = 55.0,
        atr_14: float | None = 2.0,
    ) -> TechnicalIndicatorRow:
        return TechnicalIndicatorRow(
            security_id=security_id,
            symbol=symbol,
            trading_date=trading_date,
            price_basis="capital_special_adjusted",
            sma_20=sma_20,
            sma_50=sma_50,
            sma_100=999_999.0,
            sma_200=0.0,
            rsi_14=rsi_14,
            atr_14=atr_14,
            avg_volume_20=0.0,
            relative_volume_20=999_999.0,
        )

    return factory


@pytest.fixture
def evaluator_factory(baseline_calendar):
    def factory(
        action: EarningsIntegrationAction = EarningsIntegrationAction.ENTRY_ALLOWED,
        *,
        reason: str | None = None,
    ) -> tuple[BaselineSignalEvaluator, RecordingEarningsOverlay]:
        overlay = RecordingEarningsOverlay(action, reason=reason)
        evaluator = BaselineSignalEvaluator(
            earnings_overlay=overlay,
            trading_calendar=baseline_calendar,
        )
        return evaluator, overlay

    return factory
