"""Pure deterministic execution-cost service for Phase 15C."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from stock_swing_d1.execution.costs.models import (
    BASIS_POINTS_PER_UNIT,
    BacktestExecutionCostPolicy,
    ExecutionCostPolicyRef,
    ExecutionCostQuote,
    ExecutionCostSide,
    ExecutionCostValidationError,
)
from stock_swing_d1.execution.costs.policy import build_execution_cost_policy_ref


@dataclass(frozen=True, slots=True)
class BacktestExecutionCostService:
    """Stateless calculator bound to one immutable local policy snapshot."""

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

    def quote(
        self,
        *,
        side: ExecutionCostSide,
        quantity: int,
        fill_price: Decimal,
    ) -> ExecutionCostQuote:
        """Calculate exact spread plus capped commission without rounding."""

        if type(side) is not ExecutionCostSide:
            raise ExecutionCostValidationError(
                "UNSUPPORTED_SIDE", "side must be ExecutionCostSide.BUY or SELL"
            )
        if type(quantity) is not int or quantity <= 0:
            raise ExecutionCostValidationError(
                "INVALID_QUANTITY", "quantity must be a positive genuine integer"
            )
        if type(fill_price) is not Decimal:
            raise ExecutionCostValidationError(
                "INVALID_FILL_PRICE",
                "fill_price must be a Decimal (binary float and bool are forbidden)",
            )
        if not fill_price.is_finite() or fill_price <= 0:
            raise ExecutionCostValidationError(
                "INVALID_FILL_PRICE", "fill_price must be finite and positive"
            )

        decimal_quantity = Decimal(quantity)
        notional = decimal_quantity * fill_price
        spread = self.policy.spread
        commission_policy = self.policy.commission
        spread_cost = notional * spread.bps_per_side / BASIS_POINTS_PER_UNIT
        raw_commission = decimal_quantity * commission_policy.per_share_usd
        minimum_adjusted_commission = max(
            raw_commission, commission_policy.minimum_per_order_usd
        )
        commission_cap = (
            notional * commission_policy.maximum_fraction_of_notional
        )
        commission = min(minimum_adjusted_commission, commission_cap)
        execution_cost = spread_cost + commission
        return ExecutionCostQuote(
            policy_ref=self.policy_ref,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            notional=notional,
            spread_bps_per_side=spread.bps_per_side,
            spread_cost=spread_cost,
            commission_per_share=commission_policy.per_share_usd,
            commission_minimum=commission_policy.minimum_per_order_usd,
            commission_cap_fraction=(
                commission_policy.maximum_fraction_of_notional
            ),
            raw_commission=raw_commission,
            minimum_adjusted_commission=minimum_adjusted_commission,
            commission_cap=commission_cap,
            commission=commission,
            execution_cost=execution_cost,
        )


__all__ = ["BacktestExecutionCostService"]
