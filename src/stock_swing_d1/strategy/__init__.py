"""Deterministic, provider-neutral strategy signal evaluation."""

from stock_swing_d1.strategy.baseline import (
    BaselineSignalCalendar,
    BaselineSignalAction,
    BaselineSignalDecision,
    BaselineSignalEvaluator,
    BaselineStrategyValidationError,
)

__all__ = [
    "BaselineSignalCalendar",
    "BaselineSignalAction",
    "BaselineSignalDecision",
    "BaselineSignalEvaluator",
    "BaselineStrategyValidationError",
]
