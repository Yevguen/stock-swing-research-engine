"""Static scope guards for the intentionally narrow Phase 8 package."""

from __future__ import annotations

from dataclasses import fields
import inspect

from stock_swing_d1.strategy import (
    BaselineSignalCalendar,
    BaselineSignalAction,
    BaselineSignalDecision,
    BaselineSignalEvaluator,
    BaselineStrategyValidationError,
)
from stock_swing_d1.strategy.baseline import models, service


def _baseline_source() -> str:
    return (inspect.getsource(models) + inspect.getsource(service)).lower()


def test_public_strategy_api_is_exported() -> None:
    assert BaselineSignalCalendar.__name__ == "BaselineSignalCalendar"
    assert hasattr(BaselineSignalCalendar, "signal_decision_time")
    assert hasattr(BaselineSignalCalendar, "next_session")
    assert BaselineSignalEvaluator.__name__ == "BaselineSignalEvaluator"
    assert set(BaselineSignalAction) == {
        BaselineSignalAction.VALID_LONG_SIGNAL,
        BaselineSignalAction.NO_SIGNAL,
    }
    assert issubclass(BaselineStrategyValidationError, ValueError)


def test_evaluator_derives_signal_time_internally_from_phase8_calendar() -> None:
    parameters = inspect.signature(
        BaselineSignalEvaluator.evaluate_signal
    ).parameters
    service_source = inspect.getsource(service)

    assert "signal_time" not in parameters
    assert "EarningsRiskTradingCalendar" not in service_source
    assert ".decision_time(" not in service_source
    assert ".signal_decision_time(" in service_source


def test_decision_contains_diagnostics_but_no_execution_or_portfolio_fields() -> None:
    field_names = {field.name for field in fields(BaselineSignalDecision)}

    assert {
        "security_id",
        "symbol",
        "signal_session",
        "signal_time",
        "planned_entry_session",
        "adjusted_close",
        "sma_20",
        "sma_50",
        "rsi_14",
        "atr_14",
        "atr_fraction",
        "universe_eligible",
        "close_above_sma50",
        "sma20_above_sma50",
        "rsi_above_50",
        "atr_above_minimum",
        "earnings_entry_allowed",
        "earnings_action",
        "earnings_reason",
        "action",
    } <= field_names
    assert {
        "entry_price",
        "exit_price",
        "position_size",
        "expected_return",
        "confidence_score",
        "fill_price",
        "commission",
        "slippage",
    }.isdisjoint(field_names)


def test_baseline_source_has_only_frozen_indicator_conditions() -> None:
    source = _baseline_source()

    assert "sma_20" in source
    assert "sma_50" in source
    assert "rsi_14" in source
    assert "atr_14" in source
    for forbidden in (
        "sma_10",
        "sma_100",
        "sma_200",
        "avg_volume_20",
        "relative_volume_20",
        "adx",
        "macd",
        "rate_of_change",
        "52_week",
    ):
        assert forbidden not in source


def test_baseline_does_not_recompute_universe_liquidity() -> None:
    source = _baseline_source()

    for forbidden in (
        "adtv",
        "average_dollar_volume",
        "raw_volume",
        "liquidity_threshold",
        "close * volume",
    ):
        assert forbidden not in source


def test_baseline_has_no_out_of_scope_methodology() -> None:
    source = _baseline_source()

    for forbidden in (
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "position_size",
        "portfolio_rank",
        "gap_adjust",
        "fill_price",
        "commission",
        "slippage",
        "grid_search",
        "bayesian",
        "machine_learning",
        "neural_network",
        "forecast_probability",
        "wall_street_horizon",
        "factset",
        "s&p_global",
        "norgatedata",
    ):
        assert forbidden not in source
