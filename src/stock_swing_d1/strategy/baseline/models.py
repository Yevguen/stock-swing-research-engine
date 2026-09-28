"""Immutable public results for Phase 8 Baseline Strategy v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)


class BaselineStrategyValidationError(ValueError):
    """A public baseline-strategy input violated the domain contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class BaselineSignalCalendar(Protocol):
    """Calendar boundary for completed-D1 signal timing and T+1 planning.

    ``signal_decision_time`` is the canonical historical instant at which the
    completed D1 bar for ``session`` is available to the baseline strategy.
    """

    def next_session(self, session: date) -> date:
        ...

    def signal_decision_time(self, session: date) -> datetime:
        ...


class BaselineSignalAction(StrEnum):
    """The only two outcomes produced by the long-only baseline."""

    VALID_LONG_SIGNAL = "VALID_LONG_SIGNAL"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass(frozen=True, slots=True)
class BaselineSignalDecision:
    """One immutable, diagnostic completed-session signal decision."""

    security_id: str
    symbol: str
    signal_session: date
    signal_time: datetime
    planned_entry_session: date

    adjusted_close: float
    sma_20: float | None
    sma_50: float | None
    rsi_14: float | None
    atr_14: float | None
    atr_fraction: float | None

    universe_eligible: bool

    close_above_sma50: bool
    sma20_above_sma50: bool
    rsi_above_50: bool
    atr_above_minimum: bool

    earnings_entry_allowed: bool
    earnings_action: EarningsIntegrationAction
    earnings_reason: str | None
    earnings_decision: EarningsIntegrationDecision

    action: BaselineSignalAction

    @property
    def valid_long_signal(self) -> bool:
        """Derive the convenience Boolean from the authoritative action."""

        return self.action is BaselineSignalAction.VALID_LONG_SIGNAL


__all__ = [
    "BaselineSignalCalendar",
    "BaselineSignalAction",
    "BaselineSignalDecision",
    "BaselineStrategyValidationError",
]
