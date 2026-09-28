"""Immutable public models for Phase 15C execution costs and settlement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Protocol


BACKTEST_EXECUTION_COST_POLICY_SCHEMA_VERSION = (
    "backtest_execution_cost_policy.v0.1"
)
EXECUTION_COST_POLICY_REF_SCHEMA_VERSION = "execution_cost_policy_ref.v0.1"
EXECUTION_COST_QUOTE_SCHEMA_VERSION = "execution_cost_quote.v0.1"
ADMINISTRATIVE_EXIT_PRICE_QUOTE_SCHEMA_VERSION = (
    "administrative_exit_price_quote.v0.1"
)
SETTLEMENT_RESOLUTION_SCHEMA_VERSION = "settlement_resolution.v0.1"

BROKER_NEUTRAL_POLICY_ID = (
    "broker_neutral_us_large_cap_execution_cost_v0.1"
)
BASELINE_CURRENCY = "USD"

BASELINE_ENTRY_SLIPPAGE_BPS = Decimal("5")
BASELINE_PROTECTIVE_EXIT_SLIPPAGE_BPS = Decimal("5")
BASELINE_ADMINISTRATIVE_EXIT_SLIPPAGE_BPS = Decimal("5")
BASELINE_SPREAD_BPS_PER_SIDE = Decimal("1")
BASELINE_COMMISSION_PER_SHARE_USD = Decimal("0.005")
BASELINE_COMMISSION_MINIMUM_PER_ORDER_USD = Decimal("1.00")
BASELINE_COMMISSION_MAXIMUM_FRACTION_OF_NOTIONAL = Decimal("0.01")
BASIS_POINTS_PER_UNIT = Decimal("10000")


class ExecutionCostValidationError(ValueError):
    """A public Phase 15C execution-cost contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class SpreadCostModel(StrEnum):
    """Supported spread-cost models in the frozen Phase 15C policy."""

    FIXED_CASH_EQUIVALENT_BPS_PER_SIDE = (
        "fixed_cash_equivalent_bps_per_side"
    )


class CommissionCostModel(StrEnum):
    """Supported commission models in the frozen Phase 15C policy."""

    PER_SHARE_WITH_MINIMUM_AND_NOTIONAL_CAP = (
        "per_share_with_minimum_and_notional_cap"
    )


class SettlementPolicyModel(StrEnum):
    """Supported settlement-policy declarations."""

    HISTORICAL_US_EQUITY_STANDARD_CYCLE = (
        "historical_us_equity_standard_cycle"
    )


class ExecutionCostSide(StrEnum):
    """Execution sides understood by the pure cost calculator."""

    BUY = "BUY"
    SELL = "SELL"


class HistoricalSettlementRegime(StrEnum):
    """Historical standard U.S. equity settlement regimes."""

    T_PLUS_3 = "T_PLUS_3"
    T_PLUS_2 = "T_PLUS_2"
    T_PLUS_1 = "T_PLUS_1"

    @property
    def lag_sessions(self) -> int:
        return {
            HistoricalSettlementRegime.T_PLUS_3: 3,
            HistoricalSettlementRegime.T_PLUS_2: 2,
            HistoricalSettlementRegime.T_PLUS_1: 1,
        }[self]


class SettlementSessionCalendar(Protocol):
    """Certified settlement-session operation required by the resolver."""

    def next_settlement_session(self, session: date) -> date:
        ...


def _require_decimal(
    value: object,
    *,
    field_name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    if type(value) is not Decimal:
        raise ExecutionCostValidationError(
            "INVALID_DECIMAL",
            f"{field_name} must be a Decimal (binary float and bool are forbidden)",
        )
    if not value.is_finite():
        raise ExecutionCostValidationError(
            "NON_FINITE_DECIMAL", f"{field_name} must be finite"
        )
    if positive and value <= 0:
        raise ExecutionCostValidationError(
            "NON_POSITIVE_DECIMAL", f"{field_name} must be positive"
        )
    if nonnegative and value < 0:
        raise ExecutionCostValidationError(
            "NEGATIVE_DECIMAL", f"{field_name} must be non-negative"
        )
    return value


def _require_exact_type(
    value: object,
    expected_type: type[object],
    *,
    field_name: str,
    code: str,
) -> None:
    if type(value) is not expected_type:
        raise ExecutionCostValidationError(
            code, f"{field_name} must be {expected_type.__name__}"
        )


@dataclass(frozen=True, slots=True)
class SlippagePolicy:
    """Declared adverse slippage assumptions in basis points."""

    entry_bps: Decimal
    protective_exit_bps: Decimal
    administrative_exit_bps: Decimal

    def __post_init__(self) -> None:
        for field_name in (
            "entry_bps",
            "protective_exit_bps",
            "administrative_exit_bps",
        ):
            _require_decimal(
                getattr(self, field_name),
                field_name=field_name,
                nonnegative=True,
            )


@dataclass(frozen=True, slots=True)
class SpreadCostPolicy:
    """One explicit cash-equivalent spread-cost declaration."""

    model: SpreadCostModel
    bps_per_side: Decimal

    def __post_init__(self) -> None:
        _require_exact_type(
            self.model,
            SpreadCostModel,
            field_name="model",
            code="UNSUPPORTED_SPREAD_MODEL",
        )
        if self.model is not SpreadCostModel.FIXED_CASH_EQUIVALENT_BPS_PER_SIDE:
            raise ExecutionCostValidationError(
                "UNSUPPORTED_SPREAD_MODEL",
                "spread model is not supported by Phase 15C v0.1",
            )
        _require_decimal(
            self.bps_per_side,
            field_name="bps_per_side",
            nonnegative=True,
        )


@dataclass(frozen=True, slots=True)
class CommissionCostPolicy:
    """Per-share commission with a minimum and a final notional cap."""

    model: CommissionCostModel
    per_share_usd: Decimal
    minimum_per_order_usd: Decimal
    maximum_fraction_of_notional: Decimal

    def __post_init__(self) -> None:
        _require_exact_type(
            self.model,
            CommissionCostModel,
            field_name="model",
            code="UNSUPPORTED_COMMISSION_MODEL",
        )
        if (
            self.model
            is not CommissionCostModel.PER_SHARE_WITH_MINIMUM_AND_NOTIONAL_CAP
        ):
            raise ExecutionCostValidationError(
                "UNSUPPORTED_COMMISSION_MODEL",
                "commission model is not supported by Phase 15C v0.1",
            )
        for field_name in (
            "per_share_usd",
            "minimum_per_order_usd",
            "maximum_fraction_of_notional",
        ):
            _require_decimal(
                getattr(self, field_name),
                field_name=field_name,
                positive=True,
            )


@dataclass(frozen=True, slots=True)
class SettlementPolicyRef:
    """Reference to the historical settlement model selected by a policy."""

    model: SettlementPolicyModel

    def __post_init__(self) -> None:
        _require_exact_type(
            self.model,
            SettlementPolicyModel,
            field_name="model",
            code="UNSUPPORTED_SETTLEMENT_MODEL",
        )
        if (
            self.model
            is not SettlementPolicyModel.HISTORICAL_US_EQUITY_STANDARD_CYCLE
        ):
            raise ExecutionCostValidationError(
                "UNSUPPORTED_SETTLEMENT_MODEL",
                "settlement model is not supported by Phase 15C v0.1",
            )


@dataclass(frozen=True, slots=True)
class BacktestExecutionCostPolicy:
    """Frozen broker-neutral Phase 15C v0.1 execution-cost policy."""

    schema_version: str
    policy_id: str
    currency: str
    slippage: SlippagePolicy
    spread: SpreadCostPolicy
    commission: CommissionCostPolicy
    settlement: SettlementPolicyRef

    def __post_init__(self) -> None:
        if self.schema_version != BACKTEST_EXECUTION_COST_POLICY_SCHEMA_VERSION:
            raise ExecutionCostValidationError(
                "UNSUPPORTED_POLICY_SCHEMA",
                "schema_version must identify the frozen Phase 15C v0.1 policy",
            )
        if self.policy_id != BROKER_NEUTRAL_POLICY_ID:
            raise ExecutionCostValidationError(
                "UNSUPPORTED_POLICY_ID",
                "policy_id is not supported by Phase 15C v0.1",
            )
        if self.currency != BASELINE_CURRENCY:
            raise ExecutionCostValidationError(
                "UNSUPPORTED_CURRENCY", "currency must be USD"
            )
        for field_name, expected_type in (
            ("slippage", SlippagePolicy),
            ("spread", SpreadCostPolicy),
            ("commission", CommissionCostPolicy),
            ("settlement", SettlementPolicyRef),
        ):
            _require_exact_type(
                getattr(self, field_name),
                expected_type,
                field_name=field_name,
                code="INVALID_POLICY_COMPONENT",
            )
        expected_values = (
            ("entry slippage", self.slippage.entry_bps, BASELINE_ENTRY_SLIPPAGE_BPS),
            (
                "protective-exit slippage",
                self.slippage.protective_exit_bps,
                BASELINE_PROTECTIVE_EXIT_SLIPPAGE_BPS,
            ),
            (
                "administrative-exit slippage",
                self.slippage.administrative_exit_bps,
                BASELINE_ADMINISTRATIVE_EXIT_SLIPPAGE_BPS,
            ),
            (
                "spread cost",
                self.spread.bps_per_side,
                BASELINE_SPREAD_BPS_PER_SIDE,
            ),
            (
                "per-share commission",
                self.commission.per_share_usd,
                BASELINE_COMMISSION_PER_SHARE_USD,
            ),
            (
                "minimum commission",
                self.commission.minimum_per_order_usd,
                BASELINE_COMMISSION_MINIMUM_PER_ORDER_USD,
            ),
            (
                "commission notional cap",
                self.commission.maximum_fraction_of_notional,
                BASELINE_COMMISSION_MAXIMUM_FRACTION_OF_NOTIONAL,
            ),
        )
        for name, actual, expected in expected_values:
            if actual != expected:
                raise ExecutionCostValidationError(
                    "FROZEN_POLICY_MISMATCH",
                    f"{name} does not match the frozen baseline policy",
                )


def _validate_policy_ref(value: object) -> None:
    _require_exact_type(
        value,
        ExecutionCostPolicyRef,
        field_name="policy_ref",
        code="INVALID_POLICY_REFERENCE",
    )
    if value.schema_version != EXECUTION_COST_POLICY_REF_SCHEMA_VERSION:
        raise ExecutionCostValidationError(
            "UNSUPPORTED_POLICY_REF_SCHEMA",
            "policy reference schema is not supported",
        )
    if value.policy_id != BROKER_NEUTRAL_POLICY_ID:
        raise ExecutionCostValidationError(
            "UNSUPPORTED_POLICY_ID", "policy reference has an unsupported policy_id"
        )
    if (
        type(value.policy_fingerprint) is not str
        or len(value.policy_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in value.policy_fingerprint)
    ):
        raise ExecutionCostValidationError(
            "INVALID_POLICY_FINGERPRINT",
            "policy_fingerprint must be a lowercase SHA-256 digest",
        )


@dataclass(frozen=True, slots=True)
class ExecutionCostPolicyRef:
    """Immutable identity and semantic fingerprint of one cost policy."""

    policy_id: str
    policy_fingerprint: str
    schema_version: str = EXECUTION_COST_POLICY_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_policy_ref(self)


@dataclass(frozen=True, slots=True)
class ExecutionCostQuote:
    """Auditable exact-Decimal spread and commission calculation."""

    policy_ref: ExecutionCostPolicyRef
    side: ExecutionCostSide
    quantity: int
    fill_price: Decimal
    notional: Decimal
    spread_bps_per_side: Decimal
    spread_cost: Decimal
    commission_per_share: Decimal
    commission_minimum: Decimal
    commission_cap_fraction: Decimal
    raw_commission: Decimal
    minimum_adjusted_commission: Decimal
    commission_cap: Decimal
    commission: Decimal
    execution_cost: Decimal
    schema_version: str = EXECUTION_COST_QUOTE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_COST_QUOTE_SCHEMA_VERSION:
            raise ExecutionCostValidationError(
                "UNSUPPORTED_COST_QUOTE_SCHEMA",
                "schema_version must identify an execution-cost quote v0.1",
            )
        _validate_policy_ref(self.policy_ref)
        _require_exact_type(
            self.side,
            ExecutionCostSide,
            field_name="side",
            code="UNSUPPORTED_SIDE",
        )
        if type(self.quantity) is not int or self.quantity <= 0:
            raise ExecutionCostValidationError(
                "INVALID_QUANTITY", "quantity must be a positive genuine integer"
            )
        for field_name in (
            "fill_price",
            "notional",
            "spread_bps_per_side",
            "spread_cost",
            "commission_per_share",
            "commission_minimum",
            "commission_cap_fraction",
            "raw_commission",
            "minimum_adjusted_commission",
            "commission_cap",
            "commission",
            "execution_cost",
        ):
            _require_decimal(
                getattr(self, field_name),
                field_name=field_name,
                positive=field_name in {"fill_price", "notional"},
                nonnegative=field_name not in {"fill_price", "notional"},
            )
        if (
            self.spread_bps_per_side != BASELINE_SPREAD_BPS_PER_SIDE
            or self.commission_per_share != BASELINE_COMMISSION_PER_SHARE_USD
            or self.commission_minimum
            != BASELINE_COMMISSION_MINIMUM_PER_ORDER_USD
            or self.commission_cap_fraction
            != BASELINE_COMMISSION_MAXIMUM_FRACTION_OF_NOTIONAL
        ):
            raise ExecutionCostValidationError(
                "COST_QUOTE_POLICY_MISMATCH",
                "quote audit rates do not match the frozen policy",
            )
        quantity = Decimal(self.quantity)
        expected_notional = quantity * self.fill_price
        expected_spread = (
            expected_notional * self.spread_bps_per_side / BASIS_POINTS_PER_UNIT
        )
        expected_raw = quantity * self.commission_per_share
        expected_minimum_adjusted = max(expected_raw, self.commission_minimum)
        expected_cap = expected_notional * self.commission_cap_fraction
        expected_commission = min(expected_minimum_adjusted, expected_cap)
        expected_execution_cost = expected_spread + expected_commission
        if (
            self.notional != expected_notional
            or self.spread_cost != expected_spread
            or self.raw_commission != expected_raw
            or self.minimum_adjusted_commission != expected_minimum_adjusted
            or self.commission_cap != expected_cap
            or self.commission != expected_commission
            or self.execution_cost != expected_execution_cost
        ):
            raise ExecutionCostValidationError(
                "INVALID_COST_QUOTE_ARITHMETIC",
                "execution-cost quote fields do not match the frozen formulas",
            )
        if self.execution_cost >= self.notional:
            raise ExecutionCostValidationError(
                "INVALID_COST_QUOTE_ECONOMICS",
                "execution_cost must be less than notional",
            )


@dataclass(frozen=True, slots=True)
class AdministrativeExitPriceQuote:
    """Exact adverse fill-price calculation for an approved long exit."""

    policy_ref: ExecutionCostPolicyRef
    reference_exit_price: Decimal
    slippage_bps: Decimal
    slippage_amount: Decimal
    final_fill_price: Decimal
    schema_version: str = ADMINISTRATIVE_EXIT_PRICE_QUOTE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADMINISTRATIVE_EXIT_PRICE_QUOTE_SCHEMA_VERSION:
            raise ExecutionCostValidationError(
                "UNSUPPORTED_ADMINISTRATIVE_PRICE_SCHEMA",
                "schema_version must identify an administrative price quote v0.1",
            )
        _validate_policy_ref(self.policy_ref)
        for field_name in (
            "reference_exit_price",
            "slippage_bps",
            "slippage_amount",
            "final_fill_price",
        ):
            _require_decimal(
                getattr(self, field_name),
                field_name=field_name,
                positive=True,
            )
        if self.slippage_bps != BASELINE_ADMINISTRATIVE_EXIT_SLIPPAGE_BPS:
            raise ExecutionCostValidationError(
                "ADMINISTRATIVE_PRICE_POLICY_MISMATCH",
                "slippage_bps does not match the frozen policy",
            )
        expected_slippage = (
            self.reference_exit_price * self.slippage_bps / BASIS_POINTS_PER_UNIT
        )
        expected_fill = self.reference_exit_price - expected_slippage
        if (
            self.slippage_amount != expected_slippage
            or self.final_fill_price != expected_fill
            or self.final_fill_price >= self.reference_exit_price
        ):
            raise ExecutionCostValidationError(
                "INVALID_ADMINISTRATIVE_PRICE_ARITHMETIC",
                "administrative price fields do not match the frozen formula",
            )


@dataclass(frozen=True, slots=True)
class SettlementResolution:
    """One historical regime selection and certified session resolution."""

    trade_session: date
    regime: HistoricalSettlementRegime
    lag_sessions: int
    settlement_session: date
    schema_version: str = SETTLEMENT_RESOLUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SETTLEMENT_RESOLUTION_SCHEMA_VERSION:
            raise ExecutionCostValidationError(
                "UNSUPPORTED_SETTLEMENT_RESOLUTION_SCHEMA",
                "schema_version must identify a settlement resolution v0.1",
            )
        if type(self.trade_session) is not date or type(self.settlement_session) is not date:
            raise ExecutionCostValidationError(
                "INVALID_SETTLEMENT_DATE",
                "trade and settlement sessions must be genuine dates",
            )
        _require_exact_type(
            self.regime,
            HistoricalSettlementRegime,
            field_name="regime",
            code="INVALID_SETTLEMENT_REGIME",
        )
        if type(self.lag_sessions) is not int or self.lag_sessions != self.regime.lag_sessions:
            raise ExecutionCostValidationError(
                "INVALID_SETTLEMENT_LAG",
                "lag_sessions must equal the historical regime lag",
            )
        if self.settlement_session <= self.trade_session:
            raise ExecutionCostValidationError(
                "INVALID_SETTLEMENT_CHRONOLOGY",
                "settlement_session must be after trade_session",
            )


__all__ = [
    "ADMINISTRATIVE_EXIT_PRICE_QUOTE_SCHEMA_VERSION",
    "BACKTEST_EXECUTION_COST_POLICY_SCHEMA_VERSION",
    "BASIS_POINTS_PER_UNIT",
    "BROKER_NEUTRAL_POLICY_ID",
    "AdministrativeExitPriceQuote",
    "BacktestExecutionCostPolicy",
    "CommissionCostModel",
    "CommissionCostPolicy",
    "ExecutionCostPolicyRef",
    "ExecutionCostQuote",
    "ExecutionCostSide",
    "ExecutionCostValidationError",
    "HistoricalSettlementRegime",
    "SettlementPolicyModel",
    "SettlementPolicyRef",
    "SettlementResolution",
    "SettlementSessionCalendar",
    "SlippagePolicy",
    "SpreadCostModel",
    "SpreadCostPolicy",
]
