"""Immutable Phase 13 portfolio event and ledger models."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


PORTFOLIO_EXECUTION_EVENT_SCHEMA_VERSION = "portfolio_execution_event.v0.1"
PORTFOLIO_LEDGER_ENTRY_SCHEMA_VERSION = "portfolio_ledger_entry.v0.1"


def _require_session_date(value: object) -> object:
    if type(value) is not date:
        raise ValueError("must be a Python datetime.date")
    return value


def _reject_binary_float(value: object) -> object:
    if isinstance(value, (float, bool)):
        raise ValueError("must use Decimal-compatible input, not binary float or bool")
    return value


class _ImmutablePortfolioModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


_SessionDate = Annotated[date, BeforeValidator(_require_session_date)]
_NonEmptyText = Annotated[str, Field(min_length=1)]
_CanonicalAssetId = Annotated[
    str,
    Field(pattern=r"^NORGATE:[1-9][0-9]*$"),
]
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_PositiveQuantity = Annotated[int, Field(gt=0, strict=True)]
_StrictInt = Annotated[int, Field(strict=True)]
_NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
_FiniteDecimal = Annotated[
    Decimal,
    BeforeValidator(_reject_binary_float),
    Field(allow_inf_nan=False),
]
_PositiveDecimal = Annotated[
    Decimal,
    BeforeValidator(_reject_binary_float),
    Field(gt=Decimal("0"), allow_inf_nan=False),
]
_NonNegativeDecimal = Annotated[
    Decimal,
    BeforeValidator(_reject_binary_float),
    Field(ge=Decimal("0"), allow_inf_nan=False),
]


class ExecutionSide(StrEnum):
    """The complete Phase 13 v0.1 execution-side set."""

    BUY = "BUY"
    SELL = "SELL"


class PortfolioEventKind(StrEnum):
    """Kinds that form the namespace of exactly-once event fingerprints."""

    EXECUTION = "EXECUTION"
    SETTLEMENT = "SETTLEMENT"
    DIVIDEND = "DIVIDEND"


class PortfolioEventReference(_ImmutablePortfolioModel):
    """Unambiguous event identity across Phase 13 event-kind namespaces."""

    event_kind: PortfolioEventKind
    event_id: _NonEmptyText


class PortfolioLedgerEventType(StrEnum):
    """The complete Phase 13 v0.1 portfolio-ledger event set."""

    BUY_APPLIED = "BUY_APPLIED"
    SELL_APPLIED = "SELL_APPLIED"
    SETTLEMENT_APPLIED = "SETTLEMENT_APPLIED"
    DIVIDEND_APPLIED = "DIVIDEND_APPLIED"


class PortfolioExecutionEvent(_ImmutablePortfolioModel):
    """One filled buy or full-exit sell supplied to the future engine."""

    schema_version: Literal["portfolio_execution_event.v0.1"] = (
        PORTFOLIO_EXECUTION_EVENT_SCHEMA_VERSION
    )
    execution_id: _NonEmptyText
    source_order_id: _NonEmptyText
    session: _SessionDate
    asset_id: _CanonicalAssetId
    side: ExecutionSide
    quantity: _PositiveQuantity
    fill_price: _PositiveDecimal
    execution_cost: _NonNegativeDecimal
    settlement_id: _NonEmptyText | None = None
    settlement_session: _SessionDate | None = None

    @model_validator(mode="after")
    def validate_side_specific_settlement_fields(self) -> Self:
        if self.side is ExecutionSide.BUY:
            if self.settlement_id is not None or self.settlement_session is not None:
                raise ValueError(
                    "BUY executions must not contain settlement_id or "
                    "settlement_session"
                )
            return self

        if self.settlement_id is None:
            raise ValueError("SELL executions require settlement_id")
        if self.settlement_session is None:
            raise ValueError("SELL executions require settlement_session")
        if self.settlement_session <= self.session:
            raise ValueError("settlement_session must be after execution session")
        return self


class AppliedEventFingerprint(_ImmutablePortfolioModel):
    """Canonical exactly-once identity and payload hash for an applied event."""

    event_kind: PortfolioEventKind
    event_id: _NonEmptyText
    payload_sha256: _Sha256


class PortfolioLedgerEntry(_ImmutablePortfolioModel):
    """One immutable ledger row emitted by a future transition."""

    schema_version: Literal["portfolio_ledger_entry.v0.1"] = (
        PORTFOLIO_LEDGER_ENTRY_SCHEMA_VERSION
    )
    session: _SessionDate
    sequence_in_session: _NonNegativeInt
    event_type: PortfolioLedgerEventType
    source_event_id: _NonEmptyText
    source_order_id: _NonEmptyText | None = None
    asset_id: _CanonicalAssetId | None = None
    quantity_delta: _StrictInt | None = None
    fill_price: _PositiveDecimal | None = None
    execution_cost: _NonNegativeDecimal | None = None
    settlement_id: _NonEmptyText | None = None
    settlement_session: _SessionDate | None = None
    settled_cash_delta: _FiniteDecimal
    pending_cash_delta: _FiniteDecimal
    settled_cash_after: _NonNegativeDecimal
    position_quantity_after: _NonNegativeInt | None = None
    source_payload_sha256: _Sha256
    state_hash_before: _Sha256
    state_hash_after: _Sha256


__all__ = [
    "AppliedEventFingerprint",
    "ExecutionSide",
    "PORTFOLIO_EXECUTION_EVENT_SCHEMA_VERSION",
    "PORTFOLIO_LEDGER_ENTRY_SCHEMA_VERSION",
    "PortfolioEventKind",
    "PortfolioEventReference",
    "PortfolioExecutionEvent",
    "PortfolioLedgerEntry",
    "PortfolioLedgerEventType",
]
