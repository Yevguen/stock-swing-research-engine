"""Immutable Phase 13 portfolio state and orchestration support models."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import model_validator

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CanonicalDividendAccountingEvidence,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    DividendApplicationOutcome,
    DividendLedgerEntry,
    add_exact_decimal,
    exact_decimal_times_int,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    PortfolioEventReference,
    PortfolioExecutionEvent,
    PortfolioLedgerEntry,
    _CanonicalAssetId,
    _ImmutablePortfolioModel,
    _NonEmptyText,
    _NonNegativeDecimal,
    _NonNegativeInt,
    _PositiveDecimal,
    _PositiveQuantity,
    _SessionDate,
    _Sha256,
)


PORTFOLIO_STATE_SCHEMA_VERSION = "portfolio_state.v0.1"
PORTFOLIO_TRANSITION_RESULT_SCHEMA_VERSION = "portfolio_transition_result.v0.1"
PORTFOLIO_SESSION_SNAPSHOT_SCHEMA_VERSION = "portfolio_session_snapshot.v0.1"
PORTFOLIO_BACKTEST_RESULT_SCHEMA_VERSION = "portfolio_backtest_result.v0.1"


class OpenPosition(_ImmutablePortfolioModel):
    """One fully open Phase 13 v0.1 long position."""

    asset_id: _CanonicalAssetId
    quantity: _PositiveQuantity
    entry_session: _SessionDate
    entry_price: _PositiveDecimal
    entry_execution_id: _NonEmptyText
    entry_execution_cost: _NonNegativeDecimal
    cost_basis: _PositiveDecimal

    @model_validator(mode="after")
    def validate_cost_basis(self) -> Self:
        # Exact-arithmetic hardening (Residual B): entry_price carries no
        # enforced significant-digit bound, so this multiplication and
        # addition use the shared exact helpers, not ambient-context
        # Decimal `*`/`+`. The economic meaning of cost_basis is
        # unchanged: quantity * entry_price + entry_execution_cost.
        expected = add_exact_decimal(
            exact_decimal_times_int(self.entry_price, self.quantity),
            self.entry_execution_cost,
        )
        if self.cost_basis != expected:
            raise ValueError(
                "cost_basis must equal quantity * entry_price + "
                "entry_execution_cost"
            )
        return self


class PendingSettlement(_ImmutablePortfolioModel):
    """Unsettled cash proceeds from one completed sell execution."""

    settlement_id: _NonEmptyText
    source_execution_id: _NonEmptyText
    asset_id: _CanonicalAssetId
    amount: _PositiveDecimal
    trade_session: _SessionDate
    settlement_session: _SessionDate

    @model_validator(mode="after")
    def validate_settlement_chronology(self) -> Self:
        if self.settlement_session <= self.trade_session:
            raise ValueError("settlement_session must be after trade_session")
        return self


class PortfolioState(_ImmutablePortfolioModel):
    """Canonical immutable accounting state at one completed session."""

    schema_version: Literal["portfolio_state.v0.1"] = PORTFOLIO_STATE_SCHEMA_VERSION
    base_currency: Literal["USD"] = "USD"
    as_of_session: _SessionDate | None = None
    state_version: _NonNegativeInt = 0
    settled_cash: _NonNegativeDecimal = Decimal("0")
    open_positions: tuple[OpenPosition, ...] = ()
    pending_settlements: tuple[PendingSettlement, ...] = ()
    applied_events: tuple[AppliedEventFingerprint, ...] = ()

    @model_validator(mode="after")
    def validate_uniqueness_and_canonical_order(self) -> Self:
        position_ids = tuple(position.asset_id for position in self.open_positions)
        if len(set(position_ids)) != len(position_ids):
            raise ValueError("open position asset_ids must be unique")

        settlement_ids = tuple(
            settlement.settlement_id for settlement in self.pending_settlements
        )
        if len(set(settlement_ids)) != len(settlement_ids):
            raise ValueError("pending settlement_ids must be unique")

        applied_event_keys = tuple(
            (fingerprint.event_kind, fingerprint.event_id)
            for fingerprint in self.applied_events
        )
        if len(set(applied_event_keys)) != len(applied_event_keys):
            raise ValueError("applied event kind and ID pairs must be unique")

        object.__setattr__(
            self,
            "open_positions",
            tuple(sorted(self.open_positions, key=lambda item: item.asset_id)),
        )
        object.__setattr__(
            self,
            "pending_settlements",
            tuple(
                sorted(
                    self.pending_settlements,
                    key=lambda item: (
                        item.settlement_session,
                        item.settlement_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "applied_events",
            tuple(
                sorted(
                    self.applied_events,
                    key=lambda item: (item.event_kind.value, item.event_id),
                )
            ),
        )
        return self


class PortfolioTransitionResult(_ImmutablePortfolioModel):
    """The complete immutable output contract of a future transition."""

    schema_version: Literal["portfolio_transition_result.v0.1"] = (
        PORTFOLIO_TRANSITION_RESULT_SCHEMA_VERSION
    )
    session: _SessionDate
    state_hash_before: _Sha256
    state_hash_after: _Sha256
    resulting_state: PortfolioState
    ledger_entries: tuple[PortfolioLedgerEntry, ...] = ()
    newly_applied_events: tuple[PortfolioEventReference, ...] = ()
    replayed_events: tuple[PortfolioEventReference, ...] = ()
    dividend_ledger_entries: tuple[DividendLedgerEntry, ...] = ()
    dividend_outcomes: tuple[DividendApplicationOutcome, ...] = ()


class PortfolioSessionInput(_ImmutablePortfolioModel):
    """Execution events and dividend evidence presented for one session."""

    session: _SessionDate
    execution_events: tuple[PortfolioExecutionEvent, ...] = ()
    dividend_evidence: tuple[CanonicalDividendAccountingEvidence, ...] = ()


class PortfolioSessionSnapshot(_ImmutablePortfolioModel):
    """Compact end-of-session portfolio accounting snapshot."""

    schema_version: Literal["portfolio_session_snapshot.v0.1"] = (
        PORTFOLIO_SESSION_SNAPSHOT_SCHEMA_VERSION
    )
    session: _SessionDate
    state_version: _NonNegativeInt
    settled_cash: _NonNegativeDecimal
    open_position_count: _NonNegativeInt
    pending_settlement_count: _NonNegativeInt
    state_hash: _Sha256


class PortfolioBacktestResult(_ImmutablePortfolioModel):
    """Immutable aggregate output contract for a future backtest run."""

    schema_version: Literal["portfolio_backtest_result.v0.1"] = (
        PORTFOLIO_BACKTEST_RESULT_SCHEMA_VERSION
    )
    initial_state: PortfolioState
    final_state: PortfolioState
    session_results: tuple[PortfolioTransitionResult, ...] = ()
    ledger_entries: tuple[PortfolioLedgerEntry, ...] = ()
    dividend_ledger_entries: tuple[DividendLedgerEntry, ...] = ()
    dividend_outcomes: tuple[DividendApplicationOutcome, ...] = ()
    snapshots: tuple[PortfolioSessionSnapshot, ...] = ()
    initial_state_hash: _Sha256
    final_state_hash: _Sha256


__all__ = [
    "OpenPosition",
    "PORTFOLIO_BACKTEST_RESULT_SCHEMA_VERSION",
    "PORTFOLIO_SESSION_SNAPSHOT_SCHEMA_VERSION",
    "PORTFOLIO_STATE_SCHEMA_VERSION",
    "PORTFOLIO_TRANSITION_RESULT_SCHEMA_VERSION",
    "PendingSettlement",
    "PortfolioBacktestResult",
    "PortfolioSessionInput",
    "PortfolioSessionSnapshot",
    "PortfolioState",
    "PortfolioTransitionResult",
]
