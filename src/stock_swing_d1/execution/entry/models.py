"""Immutable public models for Phase 9 entry execution."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import Protocol

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.execution.costs import (
    ExecutionCostPolicyRef,
    ExecutionCostQuote,
    ExecutionCostSide,
)


class EntryExecutionValidationError(ValueError):
    """A public entry-execution contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class EntryExecutionCalendar(Protocol):
    """Calendar operations required by the Phase 9 execution boundary."""

    def next_session(self, session: date) -> date:
        ...

    def regular_session_open_time(self, session: date) -> datetime:
        ...


class EntryExecutionCostQuoteService(Protocol):
    """Authoritative execution-cost operations consumed by Phase 9."""

    @property
    def policy_ref(self) -> ExecutionCostPolicyRef:
        ...

    def quote(
        self,
        *,
        side: ExecutionCostSide,
        quantity: int,
        fill_price: Decimal,
    ) -> ExecutionCostQuote:
        ...


class EntryExecutionStatus(StrEnum):
    """The complete Phase 9 v0.1 pending-entry lifecycle."""

    PENDING_ENTRY = "PENDING_ENTRY"
    EXECUTED = "EXECUTED"
    INVALIDATED_BY_EARNINGS = "INVALIDATED_BY_EARNINGS"
    NO_EXECUTABLE_BAR = "NO_EXECUTABLE_BAR"
    INVALID_OPEN_PRICE = "INVALID_OPEN_PRICE"
    CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION = (
        "CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION"
    )


def _canonical_execution_cost_quote(
    value: object,
    *,
    field_name: str,
) -> ExecutionCostQuote:
    if type(value) is not ExecutionCostQuote:
        raise EntryExecutionValidationError(
            "INVALID_EXECUTION_COST_QUOTE",
            f"{field_name} must be an ExecutionCostQuote",
        )
    try:
        return ExecutionCostQuote(
            **{
                field.name: getattr(value, field.name)
                for field in fields(ExecutionCostQuote)
            }
        )
    except Exception as error:
        raise EntryExecutionValidationError(
            "INVALID_EXECUTION_COST_QUOTE",
            f"{field_name} failed canonical Phase 15C reconstruction",
        ) from error


@dataclass(frozen=True, slots=True, init=False)
class PendingEntry:
    """A valid Phase 8 signal awaiting its one T+1 opportunity.

    Instances are created through ``create_pending_entry`` so the original
    Phase 8 decision and next-session calendar contract are both validated.
    """

    security_id: str
    symbol: str
    signal_session: date
    signal_time: datetime
    planned_entry_session: date

    @property
    def status(self) -> EntryExecutionStatus:
        """Return the non-terminal lifecycle state represented by this model."""

        return EntryExecutionStatus.PENDING_ENTRY

    @classmethod
    def _validated(
        cls,
        *,
        security_id: str,
        symbol: str,
        signal_session: date,
        signal_time: datetime,
        planned_entry_session: date,
    ) -> PendingEntry:
        instance = object.__new__(cls)
        object.__setattr__(instance, "security_id", security_id)
        object.__setattr__(instance, "symbol", symbol)
        object.__setattr__(instance, "signal_session", signal_session)
        object.__setattr__(instance, "signal_time", signal_time)
        object.__setattr__(
            instance, "planned_entry_session", planned_entry_session
        )
        return instance


@dataclass(frozen=True, slots=True, init=False)
class SizedPendingEntry:
    """One pending entry with a quantity fixed before its T+1 open."""

    security_id: str
    symbol: str
    signal_session: date
    signal_time: datetime
    planned_entry_session: date

    fixed_shares: int
    cash_available: float

    @classmethod
    def _validated(
        cls,
        *,
        security_id: str,
        symbol: str,
        signal_session: date,
        signal_time: datetime,
        planned_entry_session: date,
        fixed_shares: int,
        cash_available: float,
    ) -> SizedPendingEntry:
        instance = object.__new__(cls)
        for field_name, value in (
            ("security_id", security_id),
            ("symbol", symbol),
            ("signal_session", signal_session),
            ("signal_time", signal_time),
            ("planned_entry_session", planned_entry_session),
            ("fixed_shares", fixed_shares),
            ("cash_available", cash_available),
        ):
            object.__setattr__(instance, field_name, value)
        return instance


@dataclass(frozen=True, slots=True)
class EntryExecutionDecision:
    """One immutable terminal T+1 entry-execution result."""

    security_id: str
    symbol: str

    signal_session: date
    signal_time: datetime

    planned_entry_session: date
    execution_time: datetime

    earnings_revalidation_action: EarningsIntegrationAction
    earnings_reason: str | None
    earnings_decision: EarningsIntegrationDecision

    cash_available: float
    requested_shares: int
    executed_shares: int

    reference_open: float | None

    slippage_bps: float
    slippage_amount: float | None

    candidate_execution_price: float | None
    candidate_execution_cost_quote: ExecutionCostQuote | None
    candidate_cash_required: float | None

    execution_price: float | None
    execution_cost_quote: ExecutionCostQuote | None
    actual_cash_required: float | None

    status: EntryExecutionStatus

    def __post_init__(self) -> None:
        """Validate only the new Phase 15C quote/cash audit relationships."""

        early_statuses = {
            EntryExecutionStatus.PENDING_ENTRY,
            EntryExecutionStatus.INVALIDATED_BY_EARNINGS,
            EntryExecutionStatus.NO_EXECUTABLE_BAR,
            EntryExecutionStatus.INVALID_OPEN_PRICE,
        }
        if self.status in early_statuses:
            if (
                self.candidate_execution_cost_quote is not None
                or self.execution_cost_quote is not None
            ):
                raise EntryExecutionValidationError(
                    "INVALID_EXECUTION_COST_QUOTE",
                    "an early terminal decision must not contain a cost quote",
                )
            return

        if self.candidate_execution_cost_quote is None:
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_QUOTE",
                "a priced entry decision requires a candidate cost quote",
            )
        candidate_quote = _canonical_execution_cost_quote(
            self.candidate_execution_cost_quote,
            field_name="candidate_execution_cost_quote",
        )
        if (
            type(self.requested_shares) is not int
            or self.candidate_execution_price is None
            or isinstance(self.candidate_execution_price, bool)
            or not isinstance(self.candidate_execution_price, (int, float))
            or not isfinite(self.candidate_execution_price)
            or candidate_quote.side is not ExecutionCostSide.BUY
            or candidate_quote.quantity != self.requested_shares
            or candidate_quote.fill_price
            != Decimal(str(self.candidate_execution_price))
        ):
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_QUOTE",
                "candidate quote does not belong to this entry candidate",
            )
        expected_cash = float(
            candidate_quote.notional + candidate_quote.execution_cost
        )
        if (
            not isfinite(expected_cash)
            or self.candidate_cash_required != expected_cash
        ):
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_QUOTE",
                "candidate cash diagnostic must equal quote notional plus cost",
            )

        if (
            self.status
            is EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
        ):
            if self.execution_cost_quote is not None:
                raise EntryExecutionValidationError(
                    "INVALID_EXECUTION_COST_QUOTE",
                    "a cancelled entry must not contain an execution cost quote",
                )
            return

        if self.status is not EntryExecutionStatus.EXECUTED:
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_QUOTE",
                "cost quotes are unsupported for this entry status",
            )
        if self.execution_cost_quote is None:
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_QUOTE",
                "an executed entry requires an execution cost quote",
            )
        execution_quote = _canonical_execution_cost_quote(
            self.execution_cost_quote,
            field_name="execution_cost_quote",
        )
        if (
            execution_quote != candidate_quote
            or self.execution_price != self.candidate_execution_price
            or self.executed_shares != self.requested_shares
            or self.actual_cash_required != expected_cash
        ):
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_QUOTE",
                "executed quote, price, quantity, and cash must match the candidate",
            )


__all__ = [
    "EntryExecutionCalendar",
    "EntryExecutionCostQuoteService",
    "EntryExecutionDecision",
    "EntryExecutionStatus",
    "EntryExecutionValidationError",
    "PendingEntry",
    "SizedPendingEntry",
]
