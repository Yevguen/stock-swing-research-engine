"""Phase 9 historical T+1 entry execution."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, datetime
from decimal import Decimal
from math import isfinite
from typing import Final

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
    PublishedEarningsRiskOverlay,
)
from stock_swing_d1.execution.entry.models import (
    EntryExecutionCalendar,
    EntryExecutionCostQuoteService,
    EntryExecutionDecision,
    EntryExecutionStatus,
    EntryExecutionValidationError,
    PendingEntry,
    SizedPendingEntry,
    _canonical_execution_cost_quote,
)
from stock_swing_d1.execution.costs import (
    ExecutionCostPolicyRef,
    ExecutionCostSide,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


ENTRY_SLIPPAGE_BPS: Final[float] = 5.0
_BASIS_POINTS_PER_UNIT: Final[float] = 10_000.0
_ENTRY_SLIPPAGE_FRACTION: Final[float] = (
    ENTRY_SLIPPAGE_BPS / _BASIS_POINTS_PER_UNIT
)
_EXECUTION_TIMEFRAME: Final[str] = "D1"
_EXECUTION_SESSION_TYPE: Final[str] = "regular"
_EXECUTION_PRICE_BASIS: Final[str] = "unadjusted"


def _is_aware(value: object) -> bool:
    try:
        return (
            isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        )
    except Exception:
        return False


def _is_nonempty_canonical_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
    )


def _resolve_next_session(
    *,
    trading_calendar: EntryExecutionCalendar,
    signal_session: date,
) -> date:
    try:
        next_session = trading_calendar.next_session(signal_session)
    except Exception as error:
        raise EntryExecutionValidationError(
            "INVALID_EXECUTION_SESSION",
            "the execution calendar could not resolve the next session",
        ) from error
    if type(next_session) is not date or next_session <= signal_session:
        raise EntryExecutionValidationError(
            "INVALID_EXECUTION_SESSION",
            "the execution calendar must return a later genuine date",
        )
    return next_session


def create_pending_entry(
    signal: BaselineSignalDecision,
    *,
    trading_calendar: EntryExecutionCalendar,
) -> PendingEntry:
    """Create the sole Phase 9 pending state from a valid Phase 8 signal."""

    if not isinstance(signal, BaselineSignalDecision):
        raise EntryExecutionValidationError(
            "INVALID_PENDING_ENTRY",
            "signal must be a BaselineSignalDecision",
        )
    if (
        signal.action is not BaselineSignalAction.VALID_LONG_SIGNAL
        or signal.valid_long_signal is not True
    ):
        raise EntryExecutionValidationError(
            "INVALID_PENDING_ENTRY",
            "only VALID_LONG_SIGNAL can create a pending entry",
        )
    if not _is_nonempty_canonical_text(signal.security_id):
        raise EntryExecutionValidationError(
            "INVALID_PENDING_ENTRY", "security_id must be canonical non-empty text"
        )
    if not _is_nonempty_canonical_text(signal.symbol):
        raise EntryExecutionValidationError(
            "INVALID_PENDING_ENTRY", "symbol must be canonical non-empty text"
        )
    if type(signal.signal_session) is not date:
        raise EntryExecutionValidationError(
            "INVALID_PENDING_ENTRY", "signal_session must be a genuine date"
        )
    if not _is_aware(signal.signal_time):
        raise EntryExecutionValidationError(
            "INVALID_PENDING_ENTRY", "signal_time must be timezone-aware"
        )
    if (
        type(signal.planned_entry_session) is not date
        or signal.planned_entry_session <= signal.signal_session
    ):
        raise EntryExecutionValidationError(
            "INVALID_PENDING_ENTRY",
            "planned_entry_session must be later than signal_session",
        )

    next_session = _resolve_next_session(
        trading_calendar=trading_calendar,
        signal_session=signal.signal_session,
    )
    if signal.planned_entry_session != next_session:
        raise EntryExecutionValidationError(
            "INVALID_EXECUTION_SESSION",
            "planned_entry_session must equal the next trading session",
        )

    return PendingEntry._validated(
        security_id=signal.security_id,
        symbol=signal.symbol,
        signal_session=signal.signal_session,
        signal_time=signal.signal_time,
        planned_entry_session=signal.planned_entry_session,
    )


def _validate_pending_entry(
    *,
    pending_entry: PendingEntry,
    trading_calendar: EntryExecutionCalendar,
) -> None:
    if not isinstance(pending_entry, PendingEntry):
        raise EntryExecutionValidationError(
            "INVALID_PENDING_ENTRY", "pending_entry must be a PendingEntry"
        )
    if (
        not _is_nonempty_canonical_text(pending_entry.security_id)
        or not _is_nonempty_canonical_text(pending_entry.symbol)
        or type(pending_entry.signal_session) is not date
        or not _is_aware(pending_entry.signal_time)
        or type(pending_entry.planned_entry_session) is not date
        or pending_entry.planned_entry_session <= pending_entry.signal_session
    ):
        raise EntryExecutionValidationError(
            "INVALID_PENDING_ENTRY", "pending_entry has an invalid identity or time"
        )

    next_session = _resolve_next_session(
        trading_calendar=trading_calendar,
        signal_session=pending_entry.signal_session,
    )
    if pending_entry.planned_entry_session != next_session:
        raise EntryExecutionValidationError(
            "INVALID_EXECUTION_SESSION",
            "pending entry is not bound to the next trading session",
        )


def create_sized_pending_entry(
    *,
    pending_entry: PendingEntry,
    fixed_shares: int,
    cash_available: float,
) -> SizedPendingEntry:
    """Bind a pre-calculated whole-share quantity to one pending entry."""

    if not isinstance(pending_entry, PendingEntry):
        raise EntryExecutionValidationError(
            "INVALID_SIZED_PENDING_ENTRY",
            "pending_entry must be a PendingEntry",
        )
    if (
        not _is_nonempty_canonical_text(pending_entry.security_id)
        or not _is_nonempty_canonical_text(pending_entry.symbol)
        or type(pending_entry.signal_session) is not date
        or not _is_aware(pending_entry.signal_time)
        or type(pending_entry.planned_entry_session) is not date
        or pending_entry.planned_entry_session <= pending_entry.signal_session
    ):
        raise EntryExecutionValidationError(
            "INVALID_SIZED_PENDING_ENTRY",
            "pending entry identity and timing must be canonical",
        )
    if type(fixed_shares) is not int or fixed_shares <= 0:
        raise EntryExecutionValidationError(
            "INVALID_FIXED_SHARES",
            "fixed_shares must be a positive whole-share integer",
        )
    if (
        isinstance(cash_available, bool)
        or not isinstance(cash_available, (int, float))
        or not isfinite(cash_available)
        or cash_available < 0.0
    ):
        raise EntryExecutionValidationError(
            "INVALID_EXECUTION_CASH",
            "cash_available must be finite and nonnegative",
        )

    return SizedPendingEntry._validated(
        security_id=pending_entry.security_id,
        symbol=pending_entry.symbol,
        signal_session=pending_entry.signal_session,
        signal_time=pending_entry.signal_time,
        planned_entry_session=pending_entry.planned_entry_session,
        fixed_shares=fixed_shares,
        cash_available=float(cash_available),
    )


def _validate_sized_pending_entry(
    *, pending_entry: PendingEntry, sized_pending_entry: SizedPendingEntry
) -> None:
    if not isinstance(sized_pending_entry, SizedPendingEntry):
        raise EntryExecutionValidationError(
            "INVALID_SIZED_PENDING_ENTRY",
            "sized_pending_entry must be a SizedPendingEntry",
        )
    if (
        type(sized_pending_entry.fixed_shares) is not int
        or sized_pending_entry.fixed_shares <= 0
        or isinstance(sized_pending_entry.cash_available, bool)
        or not isinstance(sized_pending_entry.cash_available, (int, float))
        or not isfinite(sized_pending_entry.cash_available)
        or sized_pending_entry.cash_available < 0.0
    ):
        raise EntryExecutionValidationError(
            "INVALID_SIZED_PENDING_ENTRY",
            "fixed shares and execution cash must be valid",
        )

    mismatches = tuple(
        field_name
        for field_name in (
            "security_id",
            "symbol",
            "signal_session",
            "signal_time",
            "planned_entry_session",
        )
        if getattr(sized_pending_entry, field_name)
        != getattr(pending_entry, field_name)
    )
    if mismatches:
        raise EntryExecutionValidationError(
            "SIZED_PENDING_ENTRY_IDENTITY_MISMATCH",
            "sized and ordinary pending entries disagree on "
            + ", ".join(mismatches),
        )


def _resolve_execution_time(
    *,
    pending_entry: PendingEntry,
    trading_calendar: EntryExecutionCalendar,
) -> datetime:
    try:
        execution_time = trading_calendar.regular_session_open_time(
            pending_entry.planned_entry_session
        )
    except Exception as error:
        raise EntryExecutionValidationError(
            "INVALID_EXECUTION_TIME",
            "the execution calendar could not resolve the regular-session open",
        ) from error
    if (
        not _is_aware(execution_time)
        or execution_time.date() != pending_entry.planned_entry_session
        or execution_time <= pending_entry.signal_time
    ):
        raise EntryExecutionValidationError(
            "INVALID_EXECUTION_TIME",
            "regular-session open must be aware, later, and bound to the entry session",
        )
    return execution_time


def _terminal_decision(
    *,
    pending_entry: PendingEntry,
    sized_pending_entry: SizedPendingEntry,
    execution_time: datetime,
    earnings_decision: EarningsIntegrationDecision,
    status: EntryExecutionStatus,
) -> EntryExecutionDecision:
    return EntryExecutionDecision(
        security_id=pending_entry.security_id,
        symbol=pending_entry.symbol,
        signal_session=pending_entry.signal_session,
        signal_time=pending_entry.signal_time,
        planned_entry_session=pending_entry.planned_entry_session,
        execution_time=execution_time,
        earnings_revalidation_action=earnings_decision.action,
        earnings_reason=earnings_decision.reason,
        earnings_decision=earnings_decision,
        cash_available=sized_pending_entry.cash_available,
        requested_shares=sized_pending_entry.fixed_shares,
        executed_shares=0,
        reference_open=None,
        slippage_bps=ENTRY_SLIPPAGE_BPS,
        slippage_amount=None,
        candidate_execution_price=None,
        candidate_execution_cost_quote=None,
        candidate_cash_required=None,
        execution_price=None,
        execution_cost_quote=None,
        actual_cash_required=None,
        status=status,
    )


def _validate_execution_bar_identity(
    *, pending_entry: PendingEntry, execution_bar: StockBar
) -> None:
    mismatches = tuple(
        field_name
        for field_name, expected in (
            ("security_id", pending_entry.security_id),
            ("trading_date", pending_entry.planned_entry_session),
            ("timeframe", _EXECUTION_TIMEFRAME),
            ("session_type", _EXECUTION_SESSION_TYPE),
            ("price_basis", _EXECUTION_PRICE_BASIS),
        )
        if getattr(execution_bar, field_name) != expected
    )
    if mismatches:
        raise EntryExecutionValidationError(
            "EXECUTION_BAR_IDENTITY_MISMATCH",
            "execution bar disagrees on " + ", ".join(mismatches),
        )


def _canonical_cost_policy_ref(
    value: object,
    *,
    code: str,
) -> ExecutionCostPolicyRef:
    if type(value) is not ExecutionCostPolicyRef:
        raise EntryExecutionValidationError(
            code, "execution cost service must expose ExecutionCostPolicyRef"
        )
    try:
        return ExecutionCostPolicyRef(
            **{
                field.name: getattr(value, field.name)
                for field in fields(ExecutionCostPolicyRef)
            }
        )
    except Exception as error:
        raise EntryExecutionValidationError(
            code, "execution cost policy_ref failed canonical reconstruction"
        ) from error


@dataclass(frozen=True, slots=True, init=False, repr=False)
class EntryExecutionService:
    """Execute one valid signal at its T+1 regular-session opening instant."""

    _earnings_overlay: PublishedEarningsRiskOverlay
    _trading_calendar: EntryExecutionCalendar
    _execution_cost_service: EntryExecutionCostQuoteService
    _execution_cost_policy_ref: ExecutionCostPolicyRef

    def __init__(
        self,
        *,
        earnings_overlay: PublishedEarningsRiskOverlay,
        trading_calendar: EntryExecutionCalendar,
        execution_cost_service: EntryExecutionCostQuoteService,
    ) -> None:
        if not callable(getattr(execution_cost_service, "quote", None)):
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_SERVICE",
                "execution_cost_service must provide callable quote",
            )
        try:
            policy_ref_raw = execution_cost_service.policy_ref
        except Exception as error:
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_SERVICE",
                "execution_cost_service must expose a valid policy_ref",
            ) from error
        policy_ref = _canonical_cost_policy_ref(
            policy_ref_raw, code="INVALID_EXECUTION_COST_SERVICE"
        )
        object.__setattr__(self, "_earnings_overlay", earnings_overlay)
        object.__setattr__(self, "_trading_calendar", trading_calendar)
        object.__setattr__(self, "_execution_cost_service", execution_cost_service)
        object.__setattr__(self, "_execution_cost_policy_ref", policy_ref)

    def _validate_current_cost_policy(self) -> None:
        try:
            current_raw = self._execution_cost_service.policy_ref
        except Exception as error:
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_SERVICE",
                "execution cost service policy_ref is no longer readable",
            ) from error
        current = _canonical_cost_policy_ref(
            current_raw, code="INVALID_EXECUTION_COST_SERVICE"
        )
        if current != self._execution_cost_policy_ref:
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_SERVICE",
                "execution cost service policy_ref changed after construction",
            )

    def create_pending_entry(
        self, *, signal: BaselineSignalDecision
    ) -> PendingEntry:
        """Accept one valid Phase 8 signal for its next-session opportunity."""

        return create_pending_entry(
            signal, trading_calendar=self._trading_calendar
        )

    def execute_pending_entry(
        self,
        *,
        pending_entry: PendingEntry,
        sized_pending_entry: SizedPendingEntry,
        execution_bar: StockBar | None,
    ) -> EntryExecutionDecision:
        """Return the terminal result of the pending entry's sole opportunity."""

        _validate_pending_entry(
            pending_entry=pending_entry,
            trading_calendar=self._trading_calendar,
        )
        _validate_sized_pending_entry(
            pending_entry=pending_entry,
            sized_pending_entry=sized_pending_entry,
        )
        execution_time = _resolve_execution_time(
            pending_entry=pending_entry,
            trading_calendar=self._trading_calendar,
        )

        earnings_decision = self._earnings_overlay.revalidate_pending_entry(
            canonical_asset_id=pending_entry.security_id,
            signal_time=pending_entry.signal_time,
            execution_time=execution_time,
            planned_entry_session=pending_entry.planned_entry_session,
        )
        if not isinstance(earnings_decision, EarningsIntegrationDecision):
            raise EntryExecutionValidationError(
                "INVALID_EARNINGS_REVALIDATION_ACTION",
                "earnings overlay must return EarningsIntegrationDecision",
            )

        if (
            earnings_decision.action
            is EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
        ):
            return _terminal_decision(
                pending_entry=pending_entry,
                sized_pending_entry=sized_pending_entry,
                execution_time=execution_time,
                earnings_decision=earnings_decision,
                status=EntryExecutionStatus.INVALIDATED_BY_EARNINGS,
            )
        if (
            earnings_decision.action
            is not EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
        ):
            raise EntryExecutionValidationError(
                "INVALID_EARNINGS_REVALIDATION_ACTION",
                "only pending-entry revalidation actions are valid",
            )

        if execution_bar is None:
            return _terminal_decision(
                pending_entry=pending_entry,
                sized_pending_entry=sized_pending_entry,
                execution_time=execution_time,
                earnings_decision=earnings_decision,
                status=EntryExecutionStatus.NO_EXECUTABLE_BAR,
            )
        if not isinstance(execution_bar, StockBar):
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_BAR",
                "execution_bar must be a validated unadjusted StockBar",
            )
        _validate_execution_bar_identity(
            pending_entry=pending_entry,
            execution_bar=execution_bar,
        )

        reference_open = execution_bar.open
        if (
            isinstance(reference_open, bool)
            or not isinstance(reference_open, (int, float))
            or not isfinite(reference_open)
            or reference_open <= 0.0
        ):
            return _terminal_decision(
                pending_entry=pending_entry,
                sized_pending_entry=sized_pending_entry,
                execution_time=execution_time,
                earnings_decision=earnings_decision,
                status=EntryExecutionStatus.INVALID_OPEN_PRICE,
            )

        slippage_amount = reference_open * _ENTRY_SLIPPAGE_FRACTION
        candidate_execution_price = reference_open + slippage_amount
        if not isfinite(slippage_amount) or not isfinite(
            candidate_execution_price
        ):
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_BAR",
                "the fixed entry calculation must produce finite values",
            )
        if candidate_execution_price <= reference_open:
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_BAR",
                "the fixed long-entry calculation must be adverse",
            )

        candidate_fill_decimal = Decimal(str(candidate_execution_price))
        self._validate_current_cost_policy()
        try:
            quote_raw = self._execution_cost_service.quote(
                side=ExecutionCostSide.BUY,
                quantity=sized_pending_entry.fixed_shares,
                fill_price=candidate_fill_decimal,
            )
        except Exception as error:
            raise EntryExecutionValidationError(
                "EXECUTION_COST_QUOTE_FAILED",
                "execution cost service could not quote the entry candidate",
            ) from error
        self._validate_current_cost_policy()
        quote = _canonical_execution_cost_quote(
            quote_raw, field_name="execution_cost_service result"
        )
        if (
            quote.policy_ref != self._execution_cost_policy_ref
            or quote.side is not ExecutionCostSide.BUY
            or quote.quantity != sized_pending_entry.fixed_shares
            or quote.fill_price != candidate_fill_decimal
        ):
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_COST_QUOTE",
                "execution cost quote provenance does not match this entry candidate",
            )
        candidate_cash_required_exact = quote.notional + quote.execution_cost
        cash_available_decimal = Decimal(
            str(sized_pending_entry.cash_available)
        )
        candidate_cash_required = float(candidate_cash_required_exact)
        if not isfinite(candidate_cash_required):
            raise EntryExecutionValidationError(
                "INVALID_EXECUTION_CALCULATION",
                "candidate cash requirement must be finite",
            )

        if candidate_cash_required_exact > cash_available_decimal:
            return EntryExecutionDecision(
                security_id=pending_entry.security_id,
                symbol=pending_entry.symbol,
                signal_session=pending_entry.signal_session,
                signal_time=pending_entry.signal_time,
                planned_entry_session=pending_entry.planned_entry_session,
                execution_time=execution_time,
                earnings_revalidation_action=earnings_decision.action,
                earnings_reason=earnings_decision.reason,
                earnings_decision=earnings_decision,
                cash_available=sized_pending_entry.cash_available,
                requested_shares=sized_pending_entry.fixed_shares,
                executed_shares=0,
                reference_open=reference_open,
                slippage_bps=ENTRY_SLIPPAGE_BPS,
                slippage_amount=slippage_amount,
                candidate_execution_price=candidate_execution_price,
                candidate_execution_cost_quote=quote,
                candidate_cash_required=candidate_cash_required,
                execution_price=None,
                execution_cost_quote=None,
                actual_cash_required=None,
                status=(
                    EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
                ),
            )

        return EntryExecutionDecision(
            security_id=pending_entry.security_id,
            symbol=pending_entry.symbol,
            signal_session=pending_entry.signal_session,
            signal_time=pending_entry.signal_time,
            planned_entry_session=pending_entry.planned_entry_session,
            execution_time=execution_time,
            earnings_revalidation_action=earnings_decision.action,
            earnings_reason=earnings_decision.reason,
            earnings_decision=earnings_decision,
            cash_available=sized_pending_entry.cash_available,
            requested_shares=sized_pending_entry.fixed_shares,
            executed_shares=sized_pending_entry.fixed_shares,
            reference_open=reference_open,
            slippage_bps=ENTRY_SLIPPAGE_BPS,
            slippage_amount=slippage_amount,
            candidate_execution_price=candidate_execution_price,
            candidate_execution_cost_quote=quote,
            candidate_cash_required=candidate_cash_required,
            execution_price=candidate_execution_price,
            execution_cost_quote=quote,
            actual_cash_required=candidate_cash_required,
            status=EntryExecutionStatus.EXECUTED,
        )


__all__ = [
    "ENTRY_SLIPPAGE_BPS",
    "EntryExecutionService",
    "create_pending_entry",
    "create_sized_pending_entry",
]
