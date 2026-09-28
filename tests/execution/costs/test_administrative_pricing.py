"""Administrative-exit price quote tests."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from stock_swing_d1.execution.costs import (
    AdministrativeExitPricingService,
    BacktestExecutionCostPolicy,
    ExecutionCostValidationError,
)


def test_administrative_exit_is_exactly_five_bps_adverse(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    quote = AdministrativeExitPricingService(policy=baseline_policy).price(
        reference_exit_price=Decimal("100")
    )
    assert quote.slippage_bps == Decimal("5")
    assert quote.slippage_amount == Decimal("0.05")
    assert quote.final_fill_price == Decimal("99.95")


def test_administrative_exit_does_not_round_to_cents(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    reference = Decimal("123.456789")
    quote = AdministrativeExitPricingService(policy=baseline_policy).price(
        reference_exit_price=reference
    )
    assert quote.slippage_amount == reference * Decimal("5") / Decimal("10000")
    assert quote.final_fill_price == reference * Decimal("0.9995")
    assert quote.final_fill_price.as_tuple().exponent < -2


@pytest.mark.parametrize(
    "value",
    [0.0, 100.0, True, Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity")],
)
def test_invalid_administrative_reference_prices_fail_closed(
    baseline_policy: BacktestExecutionCostPolicy, value: object
) -> None:
    with pytest.raises(ExecutionCostValidationError):
        AdministrativeExitPricingService(policy=baseline_policy).price(
            reference_exit_price=value  # type: ignore[arg-type]
        )


def test_administrative_quote_is_immutable_and_self_validating(
    baseline_policy: BacktestExecutionCostPolicy,
) -> None:
    quote = AdministrativeExitPricingService(policy=baseline_policy).price(
        reference_exit_price=Decimal("100")
    )
    with pytest.raises(FrozenInstanceError):
        quote.final_fill_price = Decimal("100")  # type: ignore[misc]
    with pytest.raises(ExecutionCostValidationError) as raised:
        replace(quote, slippage_amount=Decimal("0.01"))
    assert raised.value.code == "INVALID_ADMINISTRATIVE_PRICE_ARITHMETIC"
