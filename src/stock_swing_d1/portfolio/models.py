"""Immutable public models for Phase 12 portfolio allocation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite

from stock_swing_d1.execution.entry import (
    EntryExecutionStatus,
    PendingEntry,
    SizedPendingEntry,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.portfolio.allocation_policy import (
    PortfolioAllocationPolicyRef,
)
from stock_swing_d1.risk.position_sizing import PositionSizingDecision
from stock_swing_d1.strategy.baseline import BaselineSignalDecision


MAX_SIMULTANEOUS_POSITIONS = 5
MAX_SINGLE_POSITION_ALLOCATION = 0.20


class PortfolioAllocationValidationError(ValueError):
    """A public portfolio-allocation contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _is_aware(value: object) -> bool:
    try:
        return (
            isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        )
    except Exception:
        return False


def _is_canonical_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _is_finite_number(value: object, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
        and (not positive or value > 0.0)
    )


@dataclass(frozen=True, slots=True)
class OpenPosition:
    """The identity and slot occupancy of one genuinely open position."""

    security_id: str
    symbol: str
    shares: int
    entry_session: date

    def __post_init__(self) -> None:
        if not _is_canonical_text(self.security_id):
            raise PortfolioAllocationValidationError(
                "INVALID_OPEN_POSITION",
                "security_id must be canonical non-empty text",
            )
        if not _is_canonical_text(self.symbol):
            raise PortfolioAllocationValidationError(
                "INVALID_OPEN_POSITION",
                "symbol must be canonical non-empty text",
            )
        if type(self.shares) is not int or self.shares <= 0:
            raise PortfolioAllocationValidationError(
                "INVALID_OPEN_POSITION",
                "shares must be a positive whole integer",
            )
        if type(self.entry_session) is not date:
            raise PortfolioAllocationValidationError(
                "INVALID_OPEN_POSITION",
                "entry_session must be a genuine date",
            )


def _validate_open_position(
    position: object, *, allocation_session: date
) -> OpenPosition:
    if not isinstance(position, OpenPosition):
        raise PortfolioAllocationValidationError(
            "INVALID_OPEN_POSITION",
            "open_positions must contain only OpenPosition values",
        )
    if (
        not _is_canonical_text(position.security_id)
        or not _is_canonical_text(position.symbol)
        or type(position.shares) is not int
        or position.shares <= 0
        or type(position.entry_session) is not date
    ):
        raise PortfolioAllocationValidationError(
            "INVALID_OPEN_POSITION",
            "open position identity, shares, and timing must be canonical",
        )
    if position.entry_session > allocation_session:
        raise PortfolioAllocationValidationError(
            "FUTURE_OPEN_POSITION",
            "an open position cannot begin after the allocation session",
        )
    return position


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """One completed-session portfolio snapshot used for an allocation cycle."""

    allocation_session: date
    decision_time: datetime
    portfolio_equity: float
    cash_available: float
    open_positions: tuple[OpenPosition, ...] = ()

    def __post_init__(self) -> None:
        if type(self.allocation_session) is not date:
            raise PortfolioAllocationValidationError(
                "INVALID_PORTFOLIO_SNAPSHOT",
                "allocation_session must be a genuine date",
            )
        if (
            not _is_aware(self.decision_time)
            or self.decision_time.date() != self.allocation_session
        ):
            raise PortfolioAllocationValidationError(
                "INVALID_PORTFOLIO_SNAPSHOT",
                "decision_time must be timezone-aware and belong to allocation_session",
            )
        if not _is_finite_number(self.portfolio_equity, positive=True):
            raise PortfolioAllocationValidationError(
                "INVALID_PORTFOLIO_EQUITY",
                "portfolio_equity must be finite and greater than zero",
            )
        if (
            not _is_finite_number(self.cash_available)
            or self.cash_available < 0.0
        ):
            raise PortfolioAllocationValidationError(
                "INVALID_PORTFOLIO_CASH",
                "cash_available must be finite and nonnegative",
            )
        if self.cash_available > self.portfolio_equity:
            raise PortfolioAllocationValidationError(
                "INVALID_PORTFOLIO_CASH",
                "cash_available must not exceed portfolio_equity",
            )

        try:
            open_positions = tuple(self.open_positions)
        except TypeError as error:
            raise PortfolioAllocationValidationError(
                "INVALID_PORTFOLIO_SNAPSHOT",
                "open_positions must be an immutable-compatible collection",
            ) from error
        for position in open_positions:
            _validate_open_position(
                position, allocation_session=self.allocation_session
            )
        security_ids = tuple(
            position.security_id for position in open_positions
        )
        if len(set(security_ids)) != len(security_ids):
            raise PortfolioAllocationValidationError(
                "DUPLICATE_OPEN_SECURITY_ID",
                "open position security_ids must be unique",
            )
        if len(open_positions) > MAX_SIMULTANEOUS_POSITIONS:
            raise PortfolioAllocationValidationError(
                "TOO_MANY_OPEN_POSITIONS",
                "open position count exceeds the five-slot limit",
            )

        object.__setattr__(self, "portfolio_equity", float(self.portfolio_equity))
        object.__setattr__(self, "cash_available", float(self.cash_available))
        object.__setattr__(self, "open_positions", open_positions)


def _validate_portfolio_snapshot(portfolio: object) -> PortfolioSnapshot:
    if not isinstance(portfolio, PortfolioSnapshot):
        raise PortfolioAllocationValidationError(
            "INVALID_PORTFOLIO_SNAPSHOT",
            "portfolio must be a PortfolioSnapshot",
        )
    if type(portfolio.allocation_session) is not date:
        raise PortfolioAllocationValidationError(
            "INVALID_PORTFOLIO_SNAPSHOT",
            "allocation_session must be a genuine date",
        )
    if (
        not _is_aware(portfolio.decision_time)
        or portfolio.decision_time.date() != portfolio.allocation_session
    ):
        raise PortfolioAllocationValidationError(
            "INVALID_PORTFOLIO_SNAPSHOT",
            "decision_time must be timezone-aware and belong to allocation_session",
        )
    if not _is_finite_number(portfolio.portfolio_equity, positive=True):
        raise PortfolioAllocationValidationError(
            "INVALID_PORTFOLIO_EQUITY",
            "portfolio_equity must be finite and greater than zero",
        )
    if (
        not _is_finite_number(portfolio.cash_available)
        or portfolio.cash_available < 0.0
        or portfolio.cash_available > portfolio.portfolio_equity
    ):
        raise PortfolioAllocationValidationError(
            "INVALID_PORTFOLIO_CASH",
            "cash_available must be finite, nonnegative, and no greater than equity",
        )
    try:
        open_positions = tuple(portfolio.open_positions)
    except TypeError as error:
        raise PortfolioAllocationValidationError(
            "INVALID_PORTFOLIO_SNAPSHOT",
            "open_positions must be an immutable-compatible collection",
        ) from error
    for position in open_positions:
        _validate_open_position(
            position, allocation_session=portfolio.allocation_session
        )
    security_ids = tuple(position.security_id for position in open_positions)
    if len(set(security_ids)) != len(security_ids):
        raise PortfolioAllocationValidationError(
            "DUPLICATE_OPEN_SECURITY_ID",
            "open position security_ids must be unique",
        )
    if len(open_positions) > MAX_SIMULTANEOUS_POSITIONS:
        raise PortfolioAllocationValidationError(
            "TOO_MANY_OPEN_POSITIONS",
            "open position count exceeds the five-slot limit",
        )
    return portfolio


def _validate_candidate_components(
    *,
    signal: object,
    pending_entry: object,
    signal_bar: object,
) -> None:
    if not isinstance(signal, BaselineSignalDecision):
        raise PortfolioAllocationValidationError(
            "INVALID_CANDIDATE", "signal must be a BaselineSignalDecision"
        )
    if not isinstance(pending_entry, PendingEntry):
        raise PortfolioAllocationValidationError(
            "INVALID_CANDIDATE", "pending_entry must be a PendingEntry"
        )
    if not isinstance(signal_bar, StockBar):
        raise PortfolioAllocationValidationError(
            "INVALID_CANDIDATE", "signal_bar must be a StockBar"
        )
    if (
        not _is_canonical_text(signal.security_id)
        or not _is_canonical_text(signal.symbol)
        or type(signal.signal_session) is not date
        or not _is_aware(signal.signal_time)
        or signal.signal_time.date() != signal.signal_session
        or type(signal.planned_entry_session) is not date
        or signal.planned_entry_session <= signal.signal_session
    ):
        raise PortfolioAllocationValidationError(
            "INVALID_CANDIDATE",
            "signal identity and timing must be canonical",
        )

    pending_mismatches = tuple(
        field_name
        for field_name in (
            "security_id",
            "symbol",
            "signal_session",
            "signal_time",
            "planned_entry_session",
        )
        if getattr(signal, field_name) != getattr(pending_entry, field_name)
    )
    bar_mismatches = tuple(
        field_name
        for field_name, expected in (
            ("security_id", signal.security_id),
            ("symbol", signal.symbol),
            ("trading_date", signal.signal_session),
            ("timeframe", "D1"),
            ("session_type", "regular"),
            ("price_basis", "unadjusted"),
        )
        if getattr(signal_bar, field_name) != expected
    )
    if pending_mismatches or bar_mismatches:
        mismatches = pending_mismatches + bar_mismatches
        raise PortfolioAllocationValidationError(
            "CANDIDATE_IDENTITY_MISMATCH",
            "candidate inputs disagree on " + ", ".join(mismatches),
        )


@dataclass(frozen=True, slots=True)
class PortfolioCandidate:
    """The authoritative Phase 8/9 inputs required by Phase 11 sizing."""

    signal: BaselineSignalDecision
    pending_entry: PendingEntry
    signal_bar: StockBar

    def __post_init__(self) -> None:
        _validate_candidate_components(
            signal=self.signal,
            pending_entry=self.pending_entry,
            signal_bar=self.signal_bar,
        )

    @property
    def security_id(self) -> str:
        return self.signal.security_id

    @property
    def symbol(self) -> str:
        return self.signal.symbol

    @property
    def signal_session(self) -> date:
        return self.signal.signal_session


def _validate_portfolio_candidate(candidate: object) -> PortfolioCandidate:
    if not isinstance(candidate, PortfolioCandidate):
        raise PortfolioAllocationValidationError(
            "INVALID_CANDIDATE_BATCH",
            "candidates must contain only PortfolioCandidate values",
        )
    _validate_candidate_components(
        signal=candidate.signal,
        pending_entry=candidate.pending_entry,
        signal_bar=candidate.signal_bar,
    )
    return candidate


class PortfolioCandidateAction(StrEnum):
    """The complete Phase 12 v0.1 candidate action set."""

    ADMITTED = "ADMITTED"
    REJECTED_DUPLICATE_SECURITY = "REJECTED_DUPLICATE_SECURITY"
    REJECTED_MAX_SIMULTANEOUS_POSITIONS = (
        "REJECTED_MAX_SIMULTANEOUS_POSITIONS"
    )
    NOT_ADMITTED_BY_POSITION_SIZING = "NOT_ADMITTED_BY_POSITION_SIZING"


@dataclass(frozen=True, slots=True, init=False)
class PortfolioCandidateDecision:
    """One immutable candidate admission and reservation decision."""

    security_id: str
    symbol: str
    processing_rank: int
    used_slots_before: int
    used_slots_after: int
    unreserved_cash_before: float
    unreserved_cash_after: float
    single_position_cap: float | None
    candidate_cash_limit: float | None
    action: PortfolioCandidateAction
    position_sizing_decision: PositionSizingDecision | None
    reserved_cash: float
    sized_pending_entry: SizedPendingEntry | None

    @classmethod
    def _validated(cls, **values: object) -> PortfolioCandidateDecision:
        instance = object.__new__(cls)
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class RankedPortfolioCandidateDecision(PortfolioCandidateDecision):
    """A Phase 12 candidate decision with immutable Phase 14 provenance."""

    source_rank: int
    ranking_snapshot_fingerprint: str
    ranking_input_fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class PortfolioAllocationDecision:
    """One immutable completed-T portfolio allocation-cycle result."""

    allocation_session: date
    decision_time: datetime
    starting_portfolio_equity: float
    starting_cash: float
    starting_open_position_count: int
    candidate_decisions: tuple[PortfolioCandidateDecision, ...]
    admitted_count: int
    total_reserved_cash: float
    remaining_unreserved_cash: float
    ending_used_slots: int
    allocation_policy_ref: PortfolioAllocationPolicyRef

    @classmethod
    def _validated(cls, **values: object) -> PortfolioAllocationDecision:
        instance = object.__new__(cls)
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class RankedPortfolioAllocationDecision(PortfolioAllocationDecision):
    """A Phase 12 allocation result tied to one authoritative ranking."""

    candidate_decisions: tuple[RankedPortfolioCandidateDecision, ...]
    ranking_snapshot_fingerprint: str
    policy_fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class PortfolioReservationSettlement:
    """One immutable terminal T+1 reservation settlement diagnostic."""

    security_id: str
    symbol: str
    reserved_cash: float
    requested_shares: int
    executed_shares: int
    entry_execution_status: EntryExecutionStatus
    actual_cash_used: float
    released_cash: float
    slot_released: bool
    slot_occupied_after_execution: bool

    @classmethod
    def _validated(cls, **values: object) -> PortfolioReservationSettlement:
        instance = object.__new__(cls)
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance


__all__ = [
    "MAX_SIMULTANEOUS_POSITIONS",
    "MAX_SINGLE_POSITION_ALLOCATION",
    "OpenPosition",
    "PortfolioAllocationDecision",
    "PortfolioAllocationValidationError",
    "PortfolioCandidate",
    "PortfolioCandidateAction",
    "PortfolioCandidateDecision",
    "PortfolioReservationSettlement",
    "PortfolioSnapshot",
    "RankedPortfolioAllocationDecision",
    "RankedPortfolioCandidateDecision",
]
