"""Shared fixtures for the pure execution-cost domain."""

from pathlib import Path

import pytest

from stock_swing_d1.execution.costs import (
    BacktestExecutionCostPolicy,
    load_backtest_execution_cost_policy,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
COST_POLICY_PATH = REPOSITORY_ROOT / "config" / "costs.yaml"


@pytest.fixture
def baseline_policy() -> BacktestExecutionCostPolicy:
    return load_backtest_execution_cost_policy(COST_POLICY_PATH)
