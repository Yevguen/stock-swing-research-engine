"""Immutable public models for Phase 10 protective exits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol

from stock_swing_d1.models import CorporateActionEvent, DividendEvent


class ProtectiveExitValidationError(ValueError):
    """A public protective-exit contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class ProtectiveExitCalendar(Protocol):
    """Calendar operation required for sequential D1 exit evaluation."""

    def next_session(self, session: date) -> date:
        ...


class ProtectiveExitAction(StrEnum):
    """The complete Phase 10 v0.1 protective-exit action set."""

    HOLD = "HOLD"
    STOP_LOSS_EXIT = "STOP_LOSS_EXIT"
    GAP_STOP_EXIT = "GAP_STOP_EXIT"
    TAKE_PROFIT_EXIT = "TAKE_PROFIT_EXIT"
    GAP_TAKE_PROFIT_EXIT = "GAP_TAKE_PROFIT_EXIT"
    UNRESOLVED_CAPITAL_EVENT = "UNRESOLVED_CAPITAL_EVENT"


@dataclass(frozen=True, slots=True, init=False)
class ProtectiveExitState:
    """One open position's fixed protective levels and evaluation cursor.

    Instances are created through :meth:`ProtectiveExitService.create_state`.
    The private constructor is also used when a HOLD advances the immutable
    state or verified share-ratio events rescale its price domain.
    """

    security_id: str
    symbol: str

    signal_session: date
    signal_time: datetime

    entry_session: date
    entry_price: float

    signal_atr_fraction: float
    risk_fraction: float

    stop_price: float
    take_profit_price: float

    last_evaluated_session: date | None

    @classmethod
    def _validated(
        cls,
        *,
        security_id: str,
        symbol: str,
        signal_session: date,
        signal_time: datetime,
        entry_session: date,
        entry_price: float,
        signal_atr_fraction: float,
        risk_fraction: float,
        stop_price: float,
        take_profit_price: float,
        last_evaluated_session: date | None,
    ) -> ProtectiveExitState:
        instance = object.__new__(cls)
        for field_name, value in (
            ("security_id", security_id),
            ("symbol", symbol),
            ("signal_session", signal_session),
            ("signal_time", signal_time),
            ("entry_session", entry_session),
            ("entry_price", entry_price),
            ("signal_atr_fraction", signal_atr_fraction),
            ("risk_fraction", risk_fraction),
            ("stop_price", stop_price),
            ("take_profit_price", take_profit_price),
            ("last_evaluated_session", last_evaluated_session),
        ):
            object.__setattr__(instance, field_name, value)
        return instance


ProtectiveCorporateAction = CorporateActionEvent | DividendEvent


@dataclass(frozen=True, slots=True)
class ProtectiveExitDecision:
    """One immutable and auditable protective-exit session decision."""

    security_id: str
    symbol: str
    session: date

    entry_session: date
    entry_price: float

    effective_stop: float
    effective_take_profit: float

    open: float
    high: float
    low: float

    stop_hit: bool | None
    take_profit_hit: bool | None
    ambiguous_both_hit: bool | None

    reference_exit_price: float | None
    exit_slippage_bps: float
    slippage_amount: float | None
    execution_exit_price: float | None

    action: ProtectiveExitAction
    resulting_state: ProtectiveExitState | None

    corporate_actions: tuple[ProtectiveCorporateAction, ...]
    applied_price_rescaling_factor: float
    unresolved_capital_events: tuple[CorporateActionEvent, ...]


__all__ = [
    "ProtectiveExitAction",
    "ProtectiveExitCalendar",
    "ProtectiveExitDecision",
    "ProtectiveExitState",
    "ProtectiveExitValidationError",
]
