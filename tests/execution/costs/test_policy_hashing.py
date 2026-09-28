"""Semantic policy fingerprint tests."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from stock_swing_d1.execution.costs import (
    AdministrativeExitPricingService,
    BacktestExecutionCostPolicy,
    BacktestExecutionCostService,
    ExecutionCostPolicyRef,
    compute_execution_cost_policy_fingerprint,
    load_backtest_execution_cost_policy,
)

from .conftest import COST_POLICY_PATH


def test_policy_fingerprint_and_reference_are_deterministic(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    first = compute_execution_cost_policy_fingerprint(baseline_policy)
    second = compute_execution_cost_policy_fingerprint(baseline_policy)
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")

    cost_service = BacktestExecutionCostService(policy=baseline_policy)
    pricing_service = AdministrativeExitPricingService(policy=baseline_policy)
    assert cost_service.policy_ref == pricing_service.policy_ref
    assert cost_service.policy_ref.policy_fingerprint == first
    with pytest.raises(FrozenInstanceError):
        cost_service.policy_ref.policy_id = "changed"  # type: ignore[misc]


def test_yaml_format_and_key_order_do_not_change_fingerprint(tmp_path) -> None:
    reordered = '''
settlement:
  model: historical_us_equity_standard_cycle
commission:
  maximum_fraction_of_notional: "0.010"
  minimum_per_order_usd: "1.0"
  per_share_usd: "0.0050"
  model: per_share_with_minimum_and_notional_cap
spread:
  bps_per_side: 1
  model: fixed_cash_equivalent_bps_per_side
slippage:
  administrative_exit_bps: 5
  protective_exit_bps: 5
  entry_bps: 5
currency: USD
policy_id: broker_neutral_us_large_cap_execution_cost_v0.1
schema_version: backtest_execution_cost_policy.v0.1
'''
    path = tmp_path / "reordered.yaml"
    path.write_text(reordered, encoding="utf-8")
    original = load_backtest_execution_cost_policy(COST_POLICY_PATH)
    reformatted = load_backtest_execution_cost_policy(path)
    assert reformatted == original
    assert compute_execution_cost_policy_fingerprint(reformatted) == (
        compute_execution_cost_policy_fingerprint(original)
    )


def test_policy_reference_reconstruction_validates_its_own_fields() -> None:
    with pytest.raises(ValueError):
        ExecutionCostPolicyRef(
            policy_id="broker_neutral_us_large_cap_execution_cost_v0.1",
            policy_fingerprint="not-a-sha256",
        )


@pytest.mark.parametrize(
    ("component", "field_name", "altered"),
    [
        ("slippage", "entry_bps", Decimal("6")),
        ("slippage", "protective_exit_bps", Decimal("6")),
        ("slippage", "administrative_exit_bps", Decimal("6")),
        ("spread", "bps_per_side", Decimal("2")),
        ("commission", "per_share_usd", Decimal("0.006")),
        ("commission", "minimum_per_order_usd", Decimal("1.01")),
        (
            "commission",
            "maximum_fraction_of_notional",
            Decimal("0.02"),
        ),
    ],
)
def test_changing_each_economic_field_changes_the_semantic_fingerprint(
    baseline_policy: BacktestExecutionCostPolicy,
    component: str,
    field_name: str,
    altered: Decimal,
) -> None:
    # Low-level mutation is used only to probe the hasher. Public construction
    # of every one of these altered baseline policies fails closed.
    changed = deepcopy(baseline_policy)
    object.__setattr__(getattr(changed, component), field_name, altered)
    assert compute_execution_cost_policy_fingerprint(changed) != (
        compute_execution_cost_policy_fingerprint(baseline_policy)
    )
