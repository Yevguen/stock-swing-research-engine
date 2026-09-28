"""Pure adverse pricing for already-approved administrative long exits."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from stock_swing_d1.execution.costs.models import (
    BASIS_POINTS_PER_UNIT,
    AdministrativeExitPriceQuote,
    BacktestExecutionCostPolicy,
    ExecutionCostPolicyRef,
    ExecutionCostValidationError,
)
from stock_swing_d1.execution.costs.policy import build_execution_cost_policy_ref


@dataclass(frozen=True, slots=True)
class AdministrativeExitPricingService:
    """Price an approved long administrative exit at 5 bps adverse."""

    policy: BacktestExecutionCostPolicy
    policy_ref: ExecutionCostPolicyRef = field(init=False)

    def __post_init__(self) -> None:
        if type(self.policy) is not BacktestExecutionCostPolicy:
            raise ExecutionCostValidationError(
                "INVALID_POLICY", "policy must be a BacktestExecutionCostPolicy"
            )
        object.__setattr__(
            self, "policy_ref", build_execution_cost_policy_ref(self.policy)
        )

    def price(
        self, *, reference_exit_price: Decimal
    ) -> AdministrativeExitPriceQuote:
        """Apply exact policy slippage, with no cent quantization."""

        if type(reference_exit_price) is not Decimal:
            raise ExecutionCostValidationError(
                "INVALID_REFERENCE_EXIT_PRICE",
                "reference_exit_price must be a Decimal",
            )
        if not reference_exit_price.is_finite() or reference_exit_price <= 0:
            raise ExecutionCostValidationError(
                "INVALID_REFERENCE_EXIT_PRICE",
                "reference_exit_price must be finite and positive",
            )
        slippage_bps = self.policy.slippage.administrative_exit_bps
        slippage_amount = (
            reference_exit_price * slippage_bps / BASIS_POINTS_PER_UNIT
        )
        final_fill_price = reference_exit_price - slippage_amount
        return AdministrativeExitPriceQuote(
            policy_ref=self.policy_ref,
            reference_exit_price=reference_exit_price,
            slippage_bps=slippage_bps,
            slippage_amount=slippage_amount,
            final_fill_price=final_fill_price,
        )


__all__ = ["AdministrativeExitPricingService"]
