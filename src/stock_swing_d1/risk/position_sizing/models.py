"""Immutable public models for Phase 11 position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite

from stock_swing_d1.execution.entry import SizedPendingEntry


class PositionSizingValidationError(ValueError):
    """A public position-sizing contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _is_finite_number(value: object, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
        and (not positive or value > 0.0)
    )


class PositionSizingAction(StrEnum):
    """The complete Phase 11 v0.1 sizing action set."""

    SIZED = "SIZED"
    SKIPPED_INSUFFICIENT_CASH = "SKIPPED_INSUFFICIENT_CASH"
    SKIPPED_RISK_TOO_SMALL = "SKIPPED_RISK_TOO_SMALL"


class PositionSizingConstraint(StrEnum):
    """The whole-share constraint that binds a sized candidate."""

    RISK = "RISK"
    CASH = "CASH"
    BOTH_EQUAL = "BOTH_EQUAL"


@dataclass(frozen=True, slots=True)
class PortfolioSizingSnapshot:
    """One immutable long-only portfolio snapshot used for sizing."""

    portfolio_equity: float
    cash_available: float

    def __post_init__(self) -> None:
        if not _is_finite_number(self.portfolio_equity, positive=True):
            raise PositionSizingValidationError(
                "INVALID_PORTFOLIO_EQUITY",
                "portfolio_equity must be finite and greater than zero",
            )
        if not _is_finite_number(self.cash_available):
            raise PositionSizingValidationError(
                "INVALID_CASH_AVAILABLE",
                "cash_available must be finite and nonnegative",
            )
        if self.cash_available < 0.0:
            raise PositionSizingValidationError(
                "INVALID_CASH_AVAILABLE",
                "cash_available must be finite and nonnegative",
            )
        if self.cash_available > self.portfolio_equity:
            raise PositionSizingValidationError(
                "CASH_EXCEEDS_PORTFOLIO_EQUITY",
                "cash_available must not exceed portfolio_equity",
            )

        object.__setattr__(self, "portfolio_equity", float(self.portfolio_equity))
        object.__setattr__(self, "cash_available", float(self.cash_available))


@dataclass(frozen=True, slots=True, init=False)
class PositionSizingDecision:
    """One immutable completed-T, single-candidate sizing decision."""

    security_id: str
    symbol: str

    signal_session: date
    signal_time: datetime
    planned_entry_session: date

    portfolio_equity: float
    cash_available: float

    target_risk_fraction: float
    sizing_target_risk_amount: float

    signal_atr_fraction: float
    sizing_risk_fraction: float

    sizing_reference_price: float
    sizing_reference_stop_price: float
    sizing_reference_stop_exit_price: float
    sizing_loss_per_share: float

    risk_sized_shares: int
    cash_sized_shares: int
    final_shares: int

    sizing_reference_cash_required: float
    sizing_reference_cash_after_entry: float

    sizing_planned_risk_amount: float
    sizing_planned_risk_fraction: float
    unused_sizing_risk_budget: float

    binding_constraint: PositionSizingConstraint | None
    action: PositionSizingAction

    sized_pending_entry: SizedPendingEntry | None

    @classmethod
    def _validated(cls, **values: object) -> PositionSizingDecision:
        instance = object.__new__(cls)
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class ExecutedInitialRisk:
    """Immutable actual initial risk after genuine entry execution."""

    security_id: str
    symbol: str

    signal_session: date
    entry_session: date

    shares: int
    sizing_portfolio_equity: float

    sizing_target_risk_amount: float
    sizing_planned_risk_amount: float
    sizing_planned_risk_fraction: float

    actual_entry_execution_price: float
    actual_stop_price: float
    executed_normal_stop_exit_price: float
    executed_initial_loss_per_share: float

    executed_initial_risk_amount: float
    executed_initial_risk_fraction: float

    @classmethod
    def _validated(cls, **values: object) -> ExecutedInitialRisk:
        instance = object.__new__(cls)
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class TradeRiskRealization:
    """One immutable ex-post risk record for a genuinely executed trade."""

    security_id: str
    symbol: str

    signal_session: date
    entry_session: date

    shares: int
    sizing_portfolio_equity: float

    actual_entry_execution_price: float
    actual_exit_execution_price: float
    realized_loss_per_share: float

    sizing_planned_risk_amount: float
    sizing_planned_risk_fraction: float

    realized_risk_amount: float
    realized_risk_fraction: float

    risk_overrun_amount: float
    risk_overrun_ratio: float

    @classmethod
    def _validated(cls, **values: object) -> TradeRiskRealization:
        instance = object.__new__(cls)
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance


__all__ = [
    "ExecutedInitialRisk",
    "PortfolioSizingSnapshot",
    "PositionSizingAction",
    "PositionSizingConstraint",
    "PositionSizingDecision",
    "PositionSizingValidationError",
    "TradeRiskRealization",
]
