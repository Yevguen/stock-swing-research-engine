"""Frozen policy loading, validation, and parity tests."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from stock_swing_d1.execution.costs import (
    BACKTEST_EXECUTION_COST_POLICY_SCHEMA_VERSION,
    BacktestExecutionCostPolicy,
    CommissionCostModel,
    ExecutionCostValidationError,
    SettlementPolicyModel,
    SpreadCostModel,
    load_backtest_execution_cost_policy,
    validate_execution_cost_policy_parity,
)
from stock_swing_d1.execution.entry import ENTRY_SLIPPAGE_BPS
from stock_swing_d1.execution.protective_exit import EXIT_SLIPPAGE_BPS

from .conftest import COST_POLICY_PATH


def test_canonical_baseline_loads_with_exact_frozen_values(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    policy = baseline_policy
    assert policy.schema_version == BACKTEST_EXECUTION_COST_POLICY_SCHEMA_VERSION
    assert policy.policy_id == (
        "broker_neutral_us_large_cap_execution_cost_v0.1"
    )
    assert policy.currency == "USD"
    assert policy.slippage.entry_bps == Decimal("5")
    assert policy.slippage.protective_exit_bps == Decimal("5")
    assert policy.slippage.administrative_exit_bps == Decimal("5")
    assert policy.spread.model is SpreadCostModel.FIXED_CASH_EQUIVALENT_BPS_PER_SIDE
    assert policy.spread.bps_per_side == Decimal("1")
    assert (
        policy.commission.model
        is CommissionCostModel.PER_SHARE_WITH_MINIMUM_AND_NOTIONAL_CAP
    )
    assert policy.commission.per_share_usd == Decimal("0.005")
    assert policy.commission.minimum_per_order_usd == Decimal("1.00")
    assert policy.commission.maximum_fraction_of_notional == Decimal("0.01")
    assert (
        policy.settlement.model
        is SettlementPolicyModel.HISTORICAL_US_EQUITY_STANDARD_CYCLE
    )


def test_policy_and_nested_models_are_immutable(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    with pytest.raises(FrozenInstanceError):
        baseline_policy.currency = "EUR"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        baseline_policy.commission.per_share_usd = Decimal("1")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        (
            'schema_version: "backtest_execution_cost_policy.v0.1"',
            'schema_version: "backtest_execution_cost_policy.v9"',
            "UNSUPPORTED_POLICY_SCHEMA",
        ),
        (
            'policy_id: "broker_neutral_us_large_cap_execution_cost_v0.1"',
            'policy_id: "unknown"',
            "UNSUPPORTED_POLICY_ID",
        ),
        ('currency: "USD"', 'currency: "EUR"', "UNSUPPORTED_CURRENCY"),
        (
            "  administrative_exit_bps: 5",
            "  administrative_exit_bps: 6",
            "FROZEN_POLICY_MISMATCH",
        ),
    ],
)
def test_policy_loader_fails_closed_on_unsupported_or_altered_baseline(
    tmp_path, old: str, new: str, code: str
) -> None:
    text = COST_POLICY_PATH.read_text(encoding="utf-8").replace(old, new)
    path = tmp_path / "costs.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ExecutionCostValidationError) as raised:
        load_backtest_execution_cost_policy(path)
    assert raised.value.code == code


def test_policy_loader_rejects_unknown_extra_fields(tmp_path) -> None:
    text = COST_POLICY_PATH.read_text(encoding="utf-8") + "extra: true\n"
    path = tmp_path / "costs.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ExecutionCostValidationError) as raised:
        load_backtest_execution_cost_policy(path)
    assert raised.value.code == "INVALID_POLICY_CONFIG"


def test_policy_loader_rejects_binary_float_monetary_config(tmp_path) -> None:
    text = COST_POLICY_PATH.read_text(encoding="utf-8").replace(
        '  per_share_usd: "0.005"', "  per_share_usd: 0.005"
    )
    path = tmp_path / "costs.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ExecutionCostValidationError) as raised:
        load_backtest_execution_cost_policy(path)
    assert raised.value.code == "INVALID_DECIMAL_CONFIG"


def test_same_baseline_id_cannot_be_reconstructed_with_changed_economics(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    altered_spread = replace(
        baseline_policy.spread, bps_per_side=Decimal("2")
    )
    with pytest.raises(ExecutionCostValidationError) as raised:
        replace(baseline_policy, spread=altered_spread)
    assert raised.value.code == "FROZEN_POLICY_MISMATCH"


def test_actual_phase9_and_phase10_slippage_constants_match_policy(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    assert validate_execution_cost_policy_parity(
        baseline_policy,
        entry_slippage_bps=ENTRY_SLIPPAGE_BPS,
        protective_exit_slippage_bps=EXIT_SLIPPAGE_BPS,
    ) is None


def test_slippage_parity_mismatch_fails_closed(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    with pytest.raises(ExecutionCostValidationError) as raised:
        validate_execution_cost_policy_parity(
            baseline_policy,
            entry_slippage_bps=6.0,
            protective_exit_slippage_bps=EXIT_SLIPPAGE_BPS,
        )
    assert raised.value.code == "ENTRY_SLIPPAGE_PARITY_MISMATCH"
