"""Deterministic Phase 11 completed-T position sizing and risk diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import floor, isfinite
from typing import Final

from stock_swing_d1.execution.entry import (
    EntryExecutionDecision,
    EntryExecutionStatus,
    PendingEntry,
    SizedPendingEntry,
    create_sized_pending_entry,
)
from stock_swing_d1.execution.protective_exit import (
    EXIT_SLIPPAGE_BPS,
    ProtectiveExitState,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.risk.position_sizing.models import (
    ExecutedInitialRisk,
    PortfolioSizingSnapshot,
    PositionSizingAction,
    PositionSizingConstraint,
    PositionSizingDecision,
    PositionSizingValidationError,
    TradeRiskRealization,
)
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


TARGET_RISK_FRACTION: Final[float] = 0.005
_REQUIRED_EXIT_SLIPPAGE_BPS: Final[float] = 5.0
_BASIS_POINTS_PER_UNIT: Final[float] = 10_000.0
_SIGNAL_TIMEFRAME: Final[str] = "D1"
_SIGNAL_SESSION_TYPE: Final[str] = "regular"
_SIGNAL_PRICE_BASIS: Final[str] = "unadjusted"

if EXIT_SLIPPAGE_BPS != _REQUIRED_EXIT_SLIPPAGE_BPS:
    raise RuntimeError("Phase 11 requires the frozen Phase 10 5-bps exit slippage")

_NORMAL_STOP_EXIT_FACTOR: Final[float] = (
    1.0 - (EXIT_SLIPPAGE_BPS / _BASIS_POINTS_PER_UNIT)
)


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
    return isinstance(value, str) and bool(value) and value == value.strip()


def _is_finite_number(value: object, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
        and (not positive or value > 0.0)
    )


def _validate_signal(signal: BaselineSignalDecision) -> tuple[float, float]:
    if not isinstance(signal, BaselineSignalDecision):
        raise PositionSizingValidationError(
            "INVALID_SIGNAL",
            "signal must be a BaselineSignalDecision",
        )
    if (
        signal.action is not BaselineSignalAction.VALID_LONG_SIGNAL
        or signal.valid_long_signal is not True
    ):
        raise PositionSizingValidationError(
            "INVALID_SIGNAL",
            "only VALID_LONG_SIGNAL can be sized",
        )
    if (
        not _is_nonempty_canonical_text(signal.security_id)
        or not _is_nonempty_canonical_text(signal.symbol)
        or type(signal.signal_session) is not date
        or not _is_aware(signal.signal_time)
        or signal.signal_time.date() != signal.signal_session
        or type(signal.planned_entry_session) is not date
        or signal.planned_entry_session <= signal.signal_session
    ):
        raise PositionSizingValidationError(
            "INVALID_SIGNAL",
            "signal identity and timing must be canonical",
        )
    if not _is_finite_number(signal.atr_fraction, positive=True):
        raise PositionSizingValidationError(
            "INVALID_SIGNAL_ATR_FRACTION",
            "signal atr_fraction must be finite and greater than zero",
        )

    signal_atr_fraction = float(signal.atr_fraction)
    sizing_risk_fraction = 2.0 * signal_atr_fraction
    if not isfinite(sizing_risk_fraction) or not 0.0 < sizing_risk_fraction < 1.0:
        raise PositionSizingValidationError(
            "INVALID_SIZING_RISK_FRACTION",
            "2 * signal atr_fraction must be strictly between zero and one",
        )
    return signal_atr_fraction, sizing_risk_fraction


def _validate_pending_identity(
    *, signal: BaselineSignalDecision, pending_entry: PendingEntry
) -> None:
    if not isinstance(pending_entry, PendingEntry):
        raise PositionSizingValidationError(
            "INVALID_PENDING_ENTRY",
            "pending_entry must be a PendingEntry",
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
        if getattr(signal, field_name) != getattr(pending_entry, field_name)
    )
    if mismatches:
        raise PositionSizingValidationError(
            "SIGNAL_PENDING_ENTRY_IDENTITY_MISMATCH",
            "signal and pending entry disagree on " + ", ".join(mismatches),
        )


def _validate_signal_bar(
    *, signal: BaselineSignalDecision, signal_bar: StockBar
) -> float:
    if not isinstance(signal_bar, StockBar):
        raise PositionSizingValidationError(
            "INVALID_SIGNAL_BAR",
            "signal_bar must be a validated unadjusted StockBar",
        )
    mismatches = tuple(
        field_name
        for field_name, expected in (
            ("security_id", signal.security_id),
            ("symbol", signal.symbol),
            ("trading_date", signal.signal_session),
            ("timeframe", _SIGNAL_TIMEFRAME),
            ("session_type", _SIGNAL_SESSION_TYPE),
            ("price_basis", _SIGNAL_PRICE_BASIS),
        )
        if getattr(signal_bar, field_name) != expected
    )
    if mismatches:
        raise PositionSizingValidationError(
            "SIGNAL_BAR_IDENTITY_MISMATCH",
            "signal bar disagrees on " + ", ".join(mismatches),
        )
    if not _is_finite_number(signal_bar.close, positive=True):
        raise PositionSizingValidationError(
            "INVALID_SIZING_REFERENCE_PRICE",
            "signal bar close must be finite and greater than zero",
        )
    return float(signal_bar.close)


def _whole_shares(*, numerator: float, denominator: float) -> int:
    ratio = numerator / denominator
    if not isfinite(ratio) or ratio < 0.0:
        raise PositionSizingValidationError(
            "INVALID_SIZING_CALCULATION",
            "whole-share sizing ratio must be finite and nonnegative",
        )
    return floor(ratio)


def _size_pending_entry(
    *,
    signal: BaselineSignalDecision,
    pending_entry: PendingEntry,
    signal_bar: StockBar,
    portfolio: PortfolioSizingSnapshot,
) -> PositionSizingDecision:
    if not isinstance(portfolio, PortfolioSizingSnapshot):
        raise PositionSizingValidationError(
            "INVALID_PORTFOLIO_SNAPSHOT",
            "portfolio must be a PortfolioSizingSnapshot",
        )

    signal_atr_fraction, sizing_risk_fraction = _validate_signal(signal)
    _validate_pending_identity(signal=signal, pending_entry=pending_entry)
    sizing_reference_price = _validate_signal_bar(
        signal=signal, signal_bar=signal_bar
    )

    portfolio_equity = portfolio.portfolio_equity
    cash_available = portfolio.cash_available
    sizing_target_risk_amount = portfolio_equity * TARGET_RISK_FRACTION
    sizing_reference_stop_price = sizing_reference_price * (
        1.0 - sizing_risk_fraction
    )
    sizing_reference_stop_exit_price = (
        sizing_reference_stop_price * _NORMAL_STOP_EXIT_FACTOR
    )
    sizing_loss_per_share = (
        sizing_reference_price - sizing_reference_stop_exit_price
    )
    if not _is_finite_number(sizing_target_risk_amount, positive=True):
        raise PositionSizingValidationError(
            "INVALID_SIZING_TARGET_RISK",
            "sizing target risk amount must be finite and greater than zero",
        )
    if (
        not _is_finite_number(sizing_reference_stop_price, positive=True)
        or sizing_reference_stop_price >= sizing_reference_price
        or not _is_finite_number(
            sizing_reference_stop_exit_price, positive=True
        )
    ):
        raise PositionSizingValidationError(
            "INVALID_SIZING_REFERENCE_STOP",
            "reference stop prices must be finite, positive, and adverse",
        )
    if not _is_finite_number(sizing_loss_per_share, positive=True):
        raise PositionSizingValidationError(
            "INVALID_SIZING_LOSS_PER_SHARE",
            "sizing loss per share must be finite and greater than zero",
        )

    risk_sized_shares = _whole_shares(
        numerator=sizing_target_risk_amount,
        denominator=sizing_loss_per_share,
    )
    cash_sized_shares = _whole_shares(
        numerator=cash_available,
        denominator=sizing_reference_price,
    )

    if cash_sized_shares == 0:
        action = PositionSizingAction.SKIPPED_INSUFFICIENT_CASH
    elif risk_sized_shares == 0:
        action = PositionSizingAction.SKIPPED_RISK_TOO_SMALL
    else:
        action = PositionSizingAction.SIZED

    if action is PositionSizingAction.SIZED:
        final_shares = min(risk_sized_shares, cash_sized_shares)
        sizing_reference_cash_required = (
            final_shares * sizing_reference_price
        )
        sizing_reference_cash_after_entry = (
            cash_available - sizing_reference_cash_required
        )
        sizing_planned_risk_amount = final_shares * sizing_loss_per_share
        sizing_planned_risk_fraction = (
            sizing_planned_risk_amount / portfolio_equity
        )
        if risk_sized_shares < cash_sized_shares:
            binding_constraint = PositionSizingConstraint.RISK
        elif cash_sized_shares < risk_sized_shares:
            binding_constraint = PositionSizingConstraint.CASH
        else:
            binding_constraint = PositionSizingConstraint.BOTH_EQUAL
        sized_pending_entry: SizedPendingEntry | None = create_sized_pending_entry(
            pending_entry=pending_entry,
            fixed_shares=final_shares,
            cash_available=cash_available,
        )
    else:
        final_shares = 0
        sizing_reference_cash_required = 0.0
        sizing_reference_cash_after_entry = cash_available
        sizing_planned_risk_amount = 0.0
        sizing_planned_risk_fraction = 0.0
        binding_constraint = None
        sized_pending_entry = None

    unused_sizing_risk_budget = (
        sizing_target_risk_amount - sizing_planned_risk_amount
    )
    if (
        sizing_reference_cash_required > cash_available
        or sizing_reference_cash_after_entry < 0.0
        or sizing_planned_risk_amount > sizing_target_risk_amount
        or sizing_planned_risk_fraction > TARGET_RISK_FRACTION
        or unused_sizing_risk_budget < 0.0
        or not all(
            isfinite(value)
            for value in (
                sizing_reference_cash_required,
                sizing_reference_cash_after_entry,
                sizing_planned_risk_amount,
                sizing_planned_risk_fraction,
                unused_sizing_risk_budget,
            )
        )
    ):
        raise PositionSizingValidationError(
            "INVALID_SIZING_CALCULATION",
            "sizing result violated a cash, risk, or finiteness invariant",
        )

    return PositionSizingDecision._validated(
        security_id=signal.security_id,
        symbol=signal.symbol,
        signal_session=signal.signal_session,
        signal_time=signal.signal_time,
        planned_entry_session=signal.planned_entry_session,
        portfolio_equity=portfolio_equity,
        cash_available=cash_available,
        target_risk_fraction=TARGET_RISK_FRACTION,
        sizing_target_risk_amount=sizing_target_risk_amount,
        signal_atr_fraction=signal_atr_fraction,
        sizing_risk_fraction=sizing_risk_fraction,
        sizing_reference_price=sizing_reference_price,
        sizing_reference_stop_price=sizing_reference_stop_price,
        sizing_reference_stop_exit_price=sizing_reference_stop_exit_price,
        sizing_loss_per_share=sizing_loss_per_share,
        risk_sized_shares=risk_sized_shares,
        cash_sized_shares=cash_sized_shares,
        final_shares=final_shares,
        sizing_reference_cash_required=sizing_reference_cash_required,
        sizing_reference_cash_after_entry=sizing_reference_cash_after_entry,
        sizing_planned_risk_amount=sizing_planned_risk_amount,
        sizing_planned_risk_fraction=sizing_planned_risk_fraction,
        unused_sizing_risk_budget=unused_sizing_risk_budget,
        binding_constraint=binding_constraint,
        action=action,
        sized_pending_entry=sized_pending_entry,
    )


def _validate_sized_decision(sizing: PositionSizingDecision) -> None:
    if not isinstance(sizing, PositionSizingDecision):
        raise PositionSizingValidationError(
            "INVALID_SIZING_DECISION",
            "sizing must be a PositionSizingDecision",
        )
    if sizing.action is not PositionSizingAction.SIZED:
        raise PositionSizingValidationError(
            "POSITION_NOT_SIZED",
            "post-execution risk is defined only for a SIZED decision",
        )
    if (
        type(sizing.final_shares) is not int
        or sizing.final_shares <= 0
        or not _is_finite_number(sizing.portfolio_equity, positive=True)
        or not _is_finite_number(sizing.sizing_target_risk_amount, positive=True)
        or not _is_finite_number(sizing.sizing_planned_risk_amount, positive=True)
        or not _is_finite_number(sizing.sizing_planned_risk_fraction)
        or sizing.sizing_planned_risk_fraction < 0.0
        or not isinstance(sizing.sized_pending_entry, SizedPendingEntry)
        or sizing.sized_pending_entry.fixed_shares != sizing.final_shares
        or sizing.sized_pending_entry.cash_available != sizing.cash_available
        or sizing.sized_pending_entry.security_id != sizing.security_id
        or sizing.sized_pending_entry.symbol != sizing.symbol
        or sizing.sized_pending_entry.signal_session != sizing.signal_session
        or sizing.sized_pending_entry.signal_time != sizing.signal_time
        or sizing.sized_pending_entry.planned_entry_session
        != sizing.planned_entry_session
    ):
        raise PositionSizingValidationError(
            "INVALID_SIZING_DECISION",
            "sizing decision has invalid post-execution risk inputs",
        )


def _validate_executed_entry(
    *, sizing: PositionSizingDecision, entry_execution: EntryExecutionDecision
) -> float:
    _validate_sized_decision(sizing)
    if not isinstance(entry_execution, EntryExecutionDecision):
        raise PositionSizingValidationError(
            "INVALID_ENTRY_EXECUTION",
            "entry_execution must be an EntryExecutionDecision",
        )
    if entry_execution.status is not EntryExecutionStatus.EXECUTED:
        raise PositionSizingValidationError(
            "INVALID_ENTRY_EXECUTION",
            "post-execution risk requires an EXECUTED entry decision",
        )
    if not _is_finite_number(entry_execution.execution_price, positive=True):
        raise PositionSizingValidationError(
            "INVALID_ENTRY_EXECUTION",
            "actual entry execution price must be finite and positive",
        )

    mismatches = tuple(
        field_name
        for field_name, sizing_value, entry_value in (
            ("security_id", sizing.security_id, entry_execution.security_id),
            ("signal_session", sizing.signal_session, entry_execution.signal_session),
            ("signal_time", sizing.signal_time, entry_execution.signal_time),
            (
                "planned_entry_session",
                sizing.planned_entry_session,
                entry_execution.planned_entry_session,
            ),
            ("requested_shares", sizing.final_shares, entry_execution.requested_shares),
            ("executed_shares", sizing.final_shares, entry_execution.executed_shares),
            ("cash_available", sizing.cash_available, entry_execution.cash_available),
        )
        if sizing_value != entry_value
    )
    if mismatches:
        raise PositionSizingValidationError(
            "SIZING_EXECUTION_IDENTITY_MISMATCH",
            "sizing and entry execution disagree on " + ", ".join(mismatches),
        )

    actual_entry_price = float(entry_execution.execution_price)
    if (
        entry_execution.candidate_execution_price != actual_entry_price
        or not _is_finite_number(
            entry_execution.candidate_cash_required, positive=True
        )
        or entry_execution.actual_cash_required
        != entry_execution.candidate_cash_required
        or entry_execution.actual_cash_required > sizing.cash_available
    ):
        raise PositionSizingValidationError(
            "INVALID_ENTRY_EXECUTION",
            "executed entry cash and price diagnostics are inconsistent",
        )
    return actual_entry_price


def calculate_executed_initial_risk(
    *,
    sizing: PositionSizingDecision,
    entry_execution: EntryExecutionDecision,
    protective_state: ProtectiveExitState,
) -> ExecutedInitialRisk:
    """Calculate actual initial risk without changing completed-T sizing."""

    actual_entry_price = _validate_executed_entry(
        sizing=sizing, entry_execution=entry_execution
    )
    if not isinstance(protective_state, ProtectiveExitState):
        raise PositionSizingValidationError(
            "INVALID_PROTECTIVE_STATE",
            "protective_state must be a ProtectiveExitState",
        )
    if protective_state.last_evaluated_session is not None:
        raise PositionSizingValidationError(
            "PROTECTIVE_STATE_NOT_INITIAL",
            "executed initial risk requires an unevaluated protective state",
        )

    mismatches = tuple(
        field_name
        for field_name, expected, actual in (
            ("security_id", sizing.security_id, protective_state.security_id),
            ("signal_session", sizing.signal_session, protective_state.signal_session),
            ("signal_time", sizing.signal_time, protective_state.signal_time),
            (
                "entry_session",
                sizing.planned_entry_session,
                protective_state.entry_session,
            ),
            ("entry_price", actual_entry_price, protective_state.entry_price),
            (
                "signal_atr_fraction",
                sizing.signal_atr_fraction,
                protective_state.signal_atr_fraction,
            ),
            (
                "risk_fraction",
                sizing.sizing_risk_fraction,
                protective_state.risk_fraction,
            ),
        )
        if expected != actual
    )
    if mismatches:
        raise PositionSizingValidationError(
            "EXECUTED_RISK_IDENTITY_MISMATCH",
            "sizing, execution, and protective state disagree on "
            + ", ".join(mismatches),
        )

    expected_stop = actual_entry_price * (1.0 - sizing.sizing_risk_fraction)
    expected_initial_loss = actual_entry_price - expected_stop
    expected_take_profit = actual_entry_price + 2.0 * expected_initial_loss
    if (
        not _is_finite_number(protective_state.stop_price, positive=True)
        or protective_state.stop_price != expected_stop
        or not _is_finite_number(protective_state.take_profit_price, positive=True)
        or protective_state.take_profit_price != expected_take_profit
    ):
        raise PositionSizingValidationError(
            "INVALID_PROTECTIVE_STATE",
            "initial protective state risk geometry is inconsistent",
        )

    executed_normal_stop_exit_price = (
        protective_state.stop_price * _NORMAL_STOP_EXIT_FACTOR
    )
    executed_initial_loss_per_share = (
        actual_entry_price - executed_normal_stop_exit_price
    )
    executed_initial_risk_amount = (
        sizing.final_shares * executed_initial_loss_per_share
    )
    executed_initial_risk_fraction = (
        executed_initial_risk_amount / sizing.portfolio_equity
    )
    if not all(
        _is_finite_number(value, positive=True)
        for value in (
            executed_normal_stop_exit_price,
            executed_initial_loss_per_share,
            executed_initial_risk_amount,
            executed_initial_risk_fraction,
        )
    ):
        raise PositionSizingValidationError(
            "INVALID_EXECUTED_INITIAL_RISK",
            "executed initial risk results must be finite and positive",
        )

    return ExecutedInitialRisk._validated(
        security_id=sizing.security_id,
        symbol=entry_execution.symbol,
        signal_session=sizing.signal_session,
        entry_session=sizing.planned_entry_session,
        shares=sizing.final_shares,
        sizing_portfolio_equity=sizing.portfolio_equity,
        sizing_target_risk_amount=sizing.sizing_target_risk_amount,
        sizing_planned_risk_amount=sizing.sizing_planned_risk_amount,
        sizing_planned_risk_fraction=sizing.sizing_planned_risk_fraction,
        actual_entry_execution_price=actual_entry_price,
        actual_stop_price=float(protective_state.stop_price),
        executed_normal_stop_exit_price=executed_normal_stop_exit_price,
        executed_initial_loss_per_share=executed_initial_loss_per_share,
        executed_initial_risk_amount=executed_initial_risk_amount,
        executed_initial_risk_fraction=executed_initial_risk_fraction,
    )


def realize_risk(
    *,
    sizing: PositionSizingDecision,
    entry_execution: EntryExecutionDecision,
    actual_exit_execution_price: float,
) -> TradeRiskRealization:
    """Calculate ex-post nonnegative risk without changing prior records."""

    actual_entry_price = _validate_executed_entry(
        sizing=sizing, entry_execution=entry_execution
    )
    if not _is_finite_number(actual_exit_execution_price, positive=True):
        raise PositionSizingValidationError(
            "INVALID_ACTUAL_EXIT_PRICE",
            "actual_exit_execution_price must be finite and greater than zero",
        )

    actual_exit_price = float(actual_exit_execution_price)
    realized_loss_per_share = max(0.0, actual_entry_price - actual_exit_price)
    realized_risk_amount = sizing.final_shares * realized_loss_per_share
    realized_risk_fraction = realized_risk_amount / sizing.portfolio_equity
    risk_overrun_amount = max(
        0.0, realized_risk_amount - sizing.sizing_planned_risk_amount
    )
    risk_overrun_ratio = (
        realized_risk_amount / sizing.sizing_planned_risk_amount
    )
    if not all(
        isfinite(value)
        for value in (
            realized_loss_per_share,
            realized_risk_amount,
            realized_risk_fraction,
            risk_overrun_amount,
            risk_overrun_ratio,
        )
    ):
        raise PositionSizingValidationError(
            "INVALID_REALIZED_RISK_CALCULATION",
            "realized-risk results must be finite",
        )

    return TradeRiskRealization._validated(
        security_id=sizing.security_id,
        symbol=entry_execution.symbol,
        signal_session=sizing.signal_session,
        entry_session=sizing.planned_entry_session,
        shares=sizing.final_shares,
        sizing_portfolio_equity=sizing.portfolio_equity,
        actual_entry_execution_price=actual_entry_price,
        actual_exit_execution_price=actual_exit_price,
        realized_loss_per_share=realized_loss_per_share,
        sizing_planned_risk_amount=sizing.sizing_planned_risk_amount,
        sizing_planned_risk_fraction=sizing.sizing_planned_risk_fraction,
        realized_risk_amount=realized_risk_amount,
        realized_risk_fraction=realized_risk_fraction,
        risk_overrun_amount=risk_overrun_amount,
        risk_overrun_ratio=risk_overrun_ratio,
    )


@dataclass(frozen=True, slots=True)
class PositionSizingService:
    """Size one pending candidate from completed-T information."""

    def size_pending_entry(
        self,
        *,
        signal: BaselineSignalDecision,
        pending_entry: PendingEntry,
        signal_bar: StockBar,
        portfolio: PortfolioSizingSnapshot,
    ) -> PositionSizingDecision:
        """Return one immutable pre-execution whole-share decision."""

        return _size_pending_entry(
            signal=signal,
            pending_entry=pending_entry,
            signal_bar=signal_bar,
            portfolio=portfolio,
        )

    def calculate_executed_initial_risk(
        self,
        *,
        sizing: PositionSizingDecision,
        entry_execution: EntryExecutionDecision,
        protective_state: ProtectiveExitState,
    ) -> ExecutedInitialRisk:
        """Return separate actual initial risk after genuine execution."""

        return calculate_executed_initial_risk(
            sizing=sizing,
            entry_execution=entry_execution,
            protective_state=protective_state,
        )

    def realize_risk(
        self,
        *,
        sizing: PositionSizingDecision,
        entry_execution: EntryExecutionDecision,
        actual_exit_execution_price: float,
    ) -> TradeRiskRealization:
        """Return a separate immutable ex-post risk record."""

        return realize_risk(
            sizing=sizing,
            entry_execution=entry_execution,
            actual_exit_execution_price=actual_exit_execution_price,
        )


__all__ = [
    "TARGET_RISK_FRACTION",
    "PositionSizingService",
    "calculate_executed_initial_risk",
    "realize_risk",
]
