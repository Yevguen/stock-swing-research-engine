"""Provider-neutral types for earnings-risk orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol

from stock_swing_d1.earnings.models import (
    EarningsRiskDecision,
    EarningsStateAsOf,
)


class EarningsRiskTradingCalendar(Protocol):
    """Calendar operations required by the frozen earnings risk layer."""

    def previous_session(self, session: date) -> date:
        ...

    def next_session(self, session: date) -> date:
        ...

    def session_distance(self, start: date, end: date) -> int:
        ...

    def decision_time(self, session: date) -> datetime:
        ...


class EarningsIntegrationAction(StrEnum):
    """Earnings-only eligibility and position-management consequences."""

    ENTRY_ALLOWED = "ENTRY_ALLOWED"
    ENTRY_BLOCKED = "ENTRY_BLOCKED"

    PENDING_ENTRY_ALLOWED = "PENDING_ENTRY_ALLOWED"
    PENDING_ENTRY_INVALIDATED = "PENDING_ENTRY_INVALIDATED"

    HOLD_POSITION = "HOLD_POSITION"
    EXIT_REQUIRED_THIS_SESSION = "EXIT_REQUIRED_THIS_SESSION"

    MISSED_EXIT_DEADLINE = "MISSED_EXIT_DEADLINE"
    UNAVOIDABLE_EARNINGS_EXPOSURE = "UNAVOIDABLE_EARNINGS_EXPOSURE"

    POSITION_ALREADY_CLOSED = "POSITION_ALREADY_CLOSED"
    NO_ADDITIONAL_EXIT = "NO_ADDITIONAL_EXIT"


@dataclass(frozen=True, slots=True)
class EarningsIntegrationDecision:
    """One immutable earnings-risk integration result."""

    action: EarningsIntegrationAction
    earnings_state: EarningsStateAsOf | None
    risk_decision: EarningsRiskDecision | None
    reason: str | None = None
