"""Bounded property tests for exact Phase 15C formulas."""

from decimal import Decimal

from hypothesis import given, strategies as st

from stock_swing_d1.execution.costs import (
    AdministrativeExitPricingService,
    BacktestExecutionCostService,
    ExecutionCostSide,
    load_backtest_execution_cost_policy,
)

from .conftest import COST_POLICY_PATH


_prices = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("100000"),
    places=6,
    allow_nan=False,
    allow_infinity=False,
)
_POLICY = load_backtest_execution_cost_policy(COST_POLICY_PATH)


@given(quantity=st.integers(min_value=1, max_value=1_000_000), price=_prices)
def test_cost_formula_properties(
    quantity: int,
    price: Decimal,
) -> None:
    service = BacktestExecutionCostService(policy=_POLICY)
    quote = service.quote(
        side=ExecutionCostSide.BUY, quantity=quantity, fill_price=price
    )
    notional = Decimal(quantity) * price
    expected_spread = notional / Decimal("10000")
    expected_commission = min(
        max(Decimal(quantity) * Decimal("0.005"), Decimal("1.00")),
        notional * Decimal("0.01"),
    )
    assert quote.notional == notional
    assert quote.spread_cost == expected_spread
    assert quote.commission == expected_commission
    assert quote.execution_cost == expected_spread + expected_commission
    assert quote.commission <= notional * Decimal("0.01")
    assert quote == service.quote(
        side=ExecutionCostSide.BUY, quantity=quantity, fill_price=price
    )


@given(reference=_prices)
def test_administrative_price_properties(
    reference: Decimal,
) -> None:
    quote = AdministrativeExitPricingService(policy=_POLICY).price(
        reference_exit_price=reference
    )
    assert quote.final_fill_price == reference * Decimal("0.9995")
    assert quote.final_fill_price < reference
