"""Exact execution-cost arithmetic and validation tests."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from stock_swing_d1.execution.costs import (
    BacktestExecutionCostPolicy,
    BacktestExecutionCostService,
    ExecutionCostSide,
    ExecutionCostValidationError,
)


@pytest.mark.parametrize(
    (
        "quantity",
        "price",
        "notional",
        "spread",
        "raw",
        "minimum_adjusted",
        "cap",
        "commission",
        "total",
    ),
    [
        (
            100,
            "100",
            "10000",
            "1.00",
            "0.500",
            "1.00",
            "100.00",
            "1.00",
            "2.00",
        ),
        (
            1000,
            "100",
            "100000",
            "10.00",
            "5.000",
            "5.000",
            "1000.00",
            "5.000",
            "15.000",
        ),
        (
            1,
            "50",
            "50",
            "0.005",
            "0.005",
            "1.00",
            "0.50",
            "0.50",
            "0.505",
        ),
    ],
)
def test_exact_frozen_cost_examples(
    baseline_policy: BacktestExecutionCostPolicy,
    quantity: int,
    price: str,
    notional: str,
    spread: str,
    raw: str,
    minimum_adjusted: str,
    cap: str,
    commission: str,
    total: str,
) -> None:
    quote = BacktestExecutionCostService(policy=baseline_policy).quote(
        side=ExecutionCostSide.BUY,
        quantity=quantity,
        fill_price=Decimal(price),
    )
    assert quote.notional == Decimal(notional)
    assert quote.spread_cost == Decimal(spread)
    assert quote.raw_commission == Decimal(raw)
    assert quote.minimum_adjusted_commission == Decimal(minimum_adjusted)
    assert quote.commission_cap == Decimal(cap)
    assert quote.commission == Decimal(commission)
    assert quote.execution_cost == Decimal(total)


def test_buy_and_sell_costs_are_symmetric_except_for_audit_side(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    service = BacktestExecutionCostService(policy=baseline_policy)
    buy = service.quote(
        side=ExecutionCostSide.BUY,
        quantity=37,
        fill_price=Decimal("123.4567"),
    )
    sell = service.quote(
        side=ExecutionCostSide.SELL,
        quantity=37,
        fill_price=Decimal("123.4567"),
    )
    assert buy.execution_cost == sell.execution_cost
    assert replace(buy, side=ExecutionCostSide.SELL) == sell


@pytest.mark.parametrize("quantity", [True, False, 0, -1, Decimal("1"), 1.0])
def test_invalid_quantities_fail_closed(
    baseline_policy: BacktestExecutionCostPolicy, quantity: object
) -> None:
    with pytest.raises(ExecutionCostValidationError) as raised:
        BacktestExecutionCostService(policy=baseline_policy).quote(
            side=ExecutionCostSide.BUY,
            quantity=quantity,  # type: ignore[arg-type]
            fill_price=Decimal("10"),
        )
    assert raised.value.code == "INVALID_QUANTITY"


@pytest.mark.parametrize(
    "price", [0.0, 1.0, True, Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity")]
)
def test_invalid_fill_prices_fail_closed(
    baseline_policy: BacktestExecutionCostPolicy, price: object
) -> None:
    with pytest.raises(ExecutionCostValidationError):
        BacktestExecutionCostService(policy=baseline_policy).quote(
            side=ExecutionCostSide.BUY,
            quantity=1,
            fill_price=price,  # type: ignore[arg-type]
        )


def test_invalid_side_fails_closed(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    with pytest.raises(ExecutionCostValidationError) as raised:
        BacktestExecutionCostService(policy=baseline_policy).quote(
            side="BUY",  # type: ignore[arg-type]
            quantity=1,
            fill_price=Decimal("10"),
        )
    assert raised.value.code == "UNSUPPORTED_SIDE"


def test_quote_is_immutable_and_reconstruction_revalidates_arithmetic(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    quote = BacktestExecutionCostService(policy=baseline_policy).quote(
        side=ExecutionCostSide.BUY,
        quantity=100,
        fill_price=Decimal("100"),
    )
    with pytest.raises(FrozenInstanceError):
        quote.execution_cost = Decimal("0")  # type: ignore[misc]
    with pytest.raises(ExecutionCostValidationError) as raised:
        replace(quote, execution_cost=Decimal("999"))
    assert raised.value.code == "INVALID_COST_QUOTE_ARITHMETIC"
