"""Phase 10 gap-aware protective exit methodology."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite, prod
from typing import Final

from stock_swing_d1.execution.entry import (
    EntryExecutionDecision,
    EntryExecutionStatus,
)
from stock_swing_d1.execution.protective_exit.models import (
    ProtectiveCorporateAction,
    ProtectiveExitAction,
    ProtectiveExitCalendar,
    ProtectiveExitDecision,
    ProtectiveExitState,
    ProtectiveExitValidationError,
)
from stock_swing_d1.models import CorporateActionEvent, DividendEvent, StockBar
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


EXIT_SLIPPAGE_BPS: Final[float] = 5.0
_BASIS_POINTS_PER_UNIT: Final[float] = 10_000.0
_EXIT_SLIPPAGE_FRACTION: Final[float] = (
    EXIT_SLIPPAGE_BPS / _BASIS_POINTS_PER_UNIT
)
_EXIT_EXECUTION_FACTOR: Final[float] = 1.0 - _EXIT_SLIPPAGE_FRACTION
_EXECUTION_TIMEFRAME: Final[str] = "D1"
_EXECUTION_SESSION_TYPE: Final[str] = "regular"
_EXECUTION_PRICE_BASIS: Final[str] = "unadjusted"
_SHARE_RATIO_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"split", "reverse_split"}
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


def _validate_creation_inputs(
    *,
    signal: BaselineSignalDecision,
    entry_execution: EntryExecutionDecision,
) -> None:
    if not isinstance(signal, BaselineSignalDecision):
        raise ProtectiveExitValidationError(
            "INVALID_SIGNAL", "signal must be a BaselineSignalDecision"
        )
    if (
        signal.action is not BaselineSignalAction.VALID_LONG_SIGNAL
        or signal.valid_long_signal is not True
    ):
        raise ProtectiveExitValidationError(
            "INVALID_SIGNAL",
            "only VALID_LONG_SIGNAL can create protective state",
        )
    if not isinstance(entry_execution, EntryExecutionDecision):
        raise ProtectiveExitValidationError(
            "INVALID_ENTRY_EXECUTION",
            "entry_execution must be an EntryExecutionDecision",
        )
    if entry_execution.status is not EntryExecutionStatus.EXECUTED:
        raise ProtectiveExitValidationError(
            "INVALID_ENTRY_EXECUTION",
            "only EXECUTED entry decisions can create protective state",
        )

    if (
        not _is_nonempty_canonical_text(signal.security_id)
        or not _is_nonempty_canonical_text(entry_execution.security_id)
        or not _is_nonempty_canonical_text(signal.symbol)
        or not _is_nonempty_canonical_text(entry_execution.symbol)
        or type(signal.signal_session) is not date
        or type(signal.planned_entry_session) is not date
        or not _is_aware(signal.signal_time)
        or type(entry_execution.signal_session) is not date
        or type(entry_execution.planned_entry_session) is not date
        or not _is_aware(entry_execution.signal_time)
        or not _is_aware(entry_execution.execution_time)
    ):
        raise ProtectiveExitValidationError(
            "INVALID_UPSTREAM_IDENTITY",
            "upstream identity and time fields must be canonical",
        )
    if (
        signal.planned_entry_session <= signal.signal_session
        or entry_execution.execution_time.date()
        != entry_execution.planned_entry_session
        or entry_execution.execution_time <= signal.signal_time
    ):
        raise ProtectiveExitValidationError(
            "INVALID_UPSTREAM_TIMELINE",
            "entry timing must be later and bound to the planned entry session",
        )

    mismatches = tuple(
        field_name
        for field_name in (
            "security_id",
            "signal_session",
            "signal_time",
            "planned_entry_session",
        )
        if getattr(signal, field_name) != getattr(entry_execution, field_name)
    )
    if mismatches:
        raise ProtectiveExitValidationError(
            "UPSTREAM_IDENTITY_MISMATCH",
            "signal and entry execution disagree on " + ", ".join(mismatches),
        )


def _validate_state(state: ProtectiveExitState) -> None:
    if not isinstance(state, ProtectiveExitState):
        raise ProtectiveExitValidationError(
            "INVALID_PROTECTIVE_STATE",
            "state must be a ProtectiveExitState",
        )
    if (
        not _is_nonempty_canonical_text(state.security_id)
        or not _is_nonempty_canonical_text(state.symbol)
        or type(state.signal_session) is not date
        or not _is_aware(state.signal_time)
        or type(state.entry_session) is not date
        or state.entry_session <= state.signal_session
        or state.signal_time.date() != state.signal_session
    ):
        raise ProtectiveExitValidationError(
            "INVALID_PROTECTIVE_STATE",
            "state identity and timing must be canonical",
        )
    if (
        not _is_finite_number(state.entry_price, positive=True)
        or not _is_finite_number(state.signal_atr_fraction, positive=True)
        or not _is_finite_number(state.risk_fraction, positive=True)
        or state.risk_fraction >= 1.0
        or state.risk_fraction != 2.0 * state.signal_atr_fraction
        or not _is_finite_number(state.stop_price, positive=True)
        or not _is_finite_number(state.take_profit_price, positive=True)
        or state.stop_price >= state.take_profit_price
    ):
        raise ProtectiveExitValidationError(
            "INVALID_PROTECTIVE_STATE",
            "state prices and fixed risk terms must be finite and ordered",
        )
    if state.last_evaluated_session is not None and (
        type(state.last_evaluated_session) is not date
        or state.last_evaluated_session < state.entry_session
    ):
        raise ProtectiveExitValidationError(
            "INVALID_PROTECTIVE_STATE",
            "last_evaluated_session must not precede entry_session",
        )


def _next_session(
    *, trading_calendar: ProtectiveExitCalendar, session: date, code: str
) -> date:
    try:
        next_session = trading_calendar.next_session(session)
    except Exception as error:
        raise ProtectiveExitValidationError(
            code, "the protective-exit calendar could not resolve the next session"
        ) from error
    if type(next_session) is not date or next_session <= session:
        raise ProtectiveExitValidationError(
            code, "the protective-exit calendar must return a later genuine date"
        )
    return next_session


def _expected_session(
    *, state: ProtectiveExitState, trading_calendar: ProtectiveExitCalendar
) -> date:
    if state.last_evaluated_session is None:
        return state.entry_session
    return _next_session(
        trading_calendar=trading_calendar,
        session=state.last_evaluated_session,
        code="INVALID_EXPECTED_EXIT_SESSION",
    )


def _validate_bar(
    *, state: ProtectiveExitState, bar: StockBar | None, expected_session: date
) -> StockBar:
    if bar is None:
        raise ProtectiveExitValidationError(
            "MISSING_EXIT_BAR",
            f"the expected D1 exit bar for {expected_session.isoformat()} is missing",
        )
    if not isinstance(bar, StockBar):
        raise ProtectiveExitValidationError(
            "INVALID_EXIT_BAR",
            "bar must be a validated unadjusted StockBar",
        )
    mismatches = tuple(
        field_name
        for field_name, expected in (
            ("security_id", state.security_id),
            ("trading_date", expected_session),
            ("timeframe", _EXECUTION_TIMEFRAME),
            ("session_type", _EXECUTION_SESSION_TYPE),
            ("price_basis", _EXECUTION_PRICE_BASIS),
        )
        if getattr(bar, field_name) != expected
    )
    if mismatches:
        raise ProtectiveExitValidationError(
            "EXIT_BAR_IDENTITY_MISMATCH",
            "exit bar disagrees on " + ", ".join(mismatches),
        )
    return bar


def _corporate_action_sort_key(
    event: ProtectiveCorporateAction,
) -> tuple[object, ...]:
    if isinstance(event, CorporateActionEvent):
        return (
            0,
            event.event_date,
            event.date_semantics,
            event.event_type,
            event.security_id,
            event.source_asset_id,
            event.old_shares or 0.0,
            event.new_shares or 0.0,
        )
    return (
        1,
        event.entitlement_date,
        event.date_semantics,
        event.dividend_type,
        event.security_id,
        event.source_asset_id,
        event.amount_per_share,
        0.0,
    )


def _action_effective_session(
    *,
    event: ProtectiveCorporateAction,
    evaluated_session: date,
    trading_calendar: ProtectiveExitCalendar,
) -> date:
    if isinstance(event, CorporateActionEvent):
        event_date = event.event_date
        date_semantics = event.date_semantics
    else:
        event_date = event.entitlement_date
        date_semantics = event.date_semantics

    if date_semantics == "entitlement_close":
        return _next_session(
            trading_calendar=trading_calendar,
            session=event_date,
            code="INVALID_CORPORATE_ACTION_SESSION",
        )
    if date_semantics in {"ex_date", "effective_date"}:
        return event_date
    if date_semantics == "unknown" and event_date == evaluated_session:
        # The caller has placed the canonical event on this session, but the
        # unknown semantics prevent any continuity inference. Its unverified
        # terms will therefore stop price evaluation below.
        return event_date
    raise ProtectiveExitValidationError(
        "INVALID_CORPORATE_ACTION_SESSION",
        "a corporate action with unknown date semantics cannot be placed",
    )


def _validated_corporate_actions(
    *,
    state: ProtectiveExitState,
    session: date,
    corporate_actions: Sequence[ProtectiveCorporateAction],
    trading_calendar: ProtectiveExitCalendar,
) -> tuple[ProtectiveCorporateAction, ...]:
    if isinstance(corporate_actions, (str, bytes)) or not isinstance(
        corporate_actions, Sequence
    ):
        raise ProtectiveExitValidationError(
            "INVALID_CORPORATE_ACTIONS",
            "corporate_actions must be a sequence of canonical events",
        )

    validated: list[ProtectiveCorporateAction] = []
    for event in corporate_actions:
        if not isinstance(event, (CorporateActionEvent, DividendEvent)):
            raise ProtectiveExitValidationError(
                "INVALID_CORPORATE_ACTION",
                "each item must be a canonical corporate-action event",
            )
        if event.security_id != state.security_id:
            raise ProtectiveExitValidationError(
                "CORPORATE_ACTION_IDENTITY_MISMATCH",
                "corporate action security_id does not match the position",
            )
        effective_session = _action_effective_session(
            event=event,
            evaluated_session=session,
            trading_calendar=trading_calendar,
        )
        if effective_session != session:
            raise ProtectiveExitValidationError(
                "CORPORATE_ACTION_SESSION_MISMATCH",
                "corporate action is not effective for the evaluated session",
            )
        validated.append(event)
    return tuple(sorted(validated, key=_corporate_action_sort_key))


def _effective_levels(
    *,
    state: ProtectiveExitState,
    session: date,
    corporate_actions: tuple[ProtectiveCorporateAction, ...],
) -> tuple[float, float, float, tuple[CorporateActionEvent, ...]]:
    capital_events = tuple(
        event
        for event in corporate_actions
        if isinstance(event, CorporateActionEvent)
    )
    supported_ratio_events = tuple(
        event
        for event in capital_events
        if event.event_type in _SHARE_RATIO_EVENT_TYPES
        and event.terms_verified is True
        and _is_finite_number(event.old_shares, positive=True)
        and _is_finite_number(event.new_shares, positive=True)
    )
    unresolved = tuple(
        event
        for event in capital_events
        if event not in supported_ratio_events
    )
    if unresolved:
        return state.stop_price, state.take_profit_price, 1.0, unresolved

    # The Phase 9 fill and new Phase 10 levels already use the entry session's
    # post-action unadjusted price scale. Supported events are classified above
    # but are deliberately not applied twice on this first session.
    if session == state.entry_session:
        return state.stop_price, state.take_profit_price, 1.0, ()

    factors = sorted(
        event.old_shares / event.new_shares
        for event in supported_ratio_events
    )
    cumulative_factor = prod(factors, start=1.0)
    effective_stop = state.stop_price * cumulative_factor
    effective_take_profit = state.take_profit_price * cumulative_factor
    if (
        not _is_finite_number(cumulative_factor, positive=True)
        or not _is_finite_number(effective_stop, positive=True)
        or not _is_finite_number(effective_take_profit, positive=True)
        or effective_stop >= effective_take_profit
    ):
        raise ProtectiveExitValidationError(
            "INVALID_CORPORATE_ACTION_RESCALING",
            "verified share-ratio rescaling must produce finite ordered levels",
        )

    return (
        effective_stop,
        effective_take_profit,
        cumulative_factor,
        (),
    )


def _advanced_state(
    *,
    state: ProtectiveExitState,
    session: date,
    effective_stop: float,
    effective_take_profit: float,
) -> ProtectiveExitState:
    return ProtectiveExitState._validated(
        security_id=state.security_id,
        symbol=state.symbol,
        signal_session=state.signal_session,
        signal_time=state.signal_time,
        entry_session=state.entry_session,
        entry_price=state.entry_price,
        signal_atr_fraction=state.signal_atr_fraction,
        risk_fraction=state.risk_fraction,
        stop_price=effective_stop,
        take_profit_price=effective_take_profit,
        last_evaluated_session=session,
    )


def _exit_prices(reference_exit_price: float) -> tuple[float, float]:
    slippage_amount = reference_exit_price * _EXIT_SLIPPAGE_FRACTION
    execution_exit_price = reference_exit_price * _EXIT_EXECUTION_FACTOR
    if (
        not _is_finite_number(slippage_amount, positive=True)
        or not _is_finite_number(execution_exit_price, positive=True)
        or execution_exit_price >= reference_exit_price
    ):
        raise ProtectiveExitValidationError(
            "INVALID_EXIT_CALCULATION",
            "the fixed long-exit calculation must be finite and adverse",
        )
    return slippage_amount, execution_exit_price


def _decision(
    *,
    state: ProtectiveExitState,
    bar: StockBar,
    effective_stop: float,
    effective_take_profit: float,
    stop_hit: bool | None,
    take_profit_hit: bool | None,
    ambiguous_both_hit: bool | None,
    reference_exit_price: float | None,
    action: ProtectiveExitAction,
    resulting_state: ProtectiveExitState | None,
    corporate_actions: tuple[ProtectiveCorporateAction, ...],
    applied_price_rescaling_factor: float,
    unresolved_capital_events: tuple[CorporateActionEvent, ...] = (),
) -> ProtectiveExitDecision:
    if reference_exit_price is None:
        slippage_amount = None
        execution_exit_price = None
    else:
        slippage_amount, execution_exit_price = _exit_prices(
            reference_exit_price
        )
    return ProtectiveExitDecision(
        security_id=state.security_id,
        symbol=state.symbol,
        session=bar.trading_date,
        entry_session=state.entry_session,
        entry_price=state.entry_price,
        effective_stop=effective_stop,
        effective_take_profit=effective_take_profit,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        stop_hit=stop_hit,
        take_profit_hit=take_profit_hit,
        ambiguous_both_hit=ambiguous_both_hit,
        reference_exit_price=reference_exit_price,
        exit_slippage_bps=EXIT_SLIPPAGE_BPS,
        slippage_amount=slippage_amount,
        execution_exit_price=execution_exit_price,
        action=action,
        resulting_state=resulting_state,
        corporate_actions=corporate_actions,
        applied_price_rescaling_factor=applied_price_rescaling_factor,
        unresolved_capital_events=unresolved_capital_events,
    )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class ProtectiveExitService:
    """Create and evaluate deterministic long-position protective exits."""

    _trading_calendar: ProtectiveExitCalendar

    def __init__(self, *, trading_calendar: ProtectiveExitCalendar) -> None:
        object.__setattr__(self, "_trading_calendar", trading_calendar)

    def create_state(
        self,
        *,
        signal: BaselineSignalDecision,
        entry_execution: EntryExecutionDecision,
    ) -> ProtectiveExitState:
        """Create fixed initial levels from the authoritative Phase 8/9 data."""

        _validate_creation_inputs(
            signal=signal, entry_execution=entry_execution
        )
        signal_atr_fraction = signal.atr_fraction
        entry_price = entry_execution.execution_price
        if not _is_finite_number(signal_atr_fraction, positive=True):
            raise ProtectiveExitValidationError(
                "INVALID_SIGNAL_ATR_FRACTION",
                "signal-time atr_fraction must be finite and greater than zero",
            )
        risk_fraction = 2.0 * signal_atr_fraction
        if not isfinite(risk_fraction) or not 0.0 < risk_fraction < 1.0:
            raise ProtectiveExitValidationError(
                "INVALID_RISK_FRACTION",
                "2 * signal-time atr_fraction must be strictly between zero and one",
            )
        if not _is_finite_number(entry_price, positive=True):
            raise ProtectiveExitValidationError(
                "INVALID_ENTRY_PRICE",
                "the executed entry price must be finite and greater than zero",
            )

        stop_price = entry_price * (1.0 - risk_fraction)
        initial_risk = entry_price - stop_price
        take_profit_price = entry_price + 2.0 * initial_risk
        if (
            not _is_finite_number(stop_price, positive=True)
            or not _is_finite_number(initial_risk, positive=True)
            or not _is_finite_number(take_profit_price, positive=True)
            or not stop_price < entry_price < take_profit_price
        ):
            raise ProtectiveExitValidationError(
                "INVALID_PROTECTIVE_LEVELS",
                "derived stop, entry, risk, and take-profit must be finite and ordered",
            )

        return ProtectiveExitState._validated(
            security_id=entry_execution.security_id,
            symbol=entry_execution.symbol,
            signal_session=signal.signal_session,
            signal_time=signal.signal_time,
            entry_session=entry_execution.planned_entry_session,
            entry_price=float(entry_price),
            signal_atr_fraction=float(signal_atr_fraction),
            risk_fraction=risk_fraction,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            last_evaluated_session=None,
        )

    def evaluate_session(
        self,
        *,
        state: ProtectiveExitState,
        bar: StockBar | None,
        corporate_actions: Sequence[ProtectiveCorporateAction] = (),
    ) -> ProtectiveExitDecision:
        """Evaluate exactly the expected session without inventing OHLC path."""

        _validate_state(state)
        expected_session = _expected_session(
            state=state, trading_calendar=self._trading_calendar
        )
        validated_bar = _validate_bar(
            state=state, bar=bar, expected_session=expected_session
        )
        validated_actions = _validated_corporate_actions(
            state=state,
            session=expected_session,
            corporate_actions=corporate_actions,
            trading_calendar=self._trading_calendar,
        )
        (
            effective_stop,
            effective_take_profit,
            rescaling_factor,
            unresolved_events,
        ) = _effective_levels(
            state=state,
            session=expected_session,
            corporate_actions=validated_actions,
        )

        if unresolved_events:
            return _decision(
                state=state,
                bar=validated_bar,
                effective_stop=effective_stop,
                effective_take_profit=effective_take_profit,
                stop_hit=None,
                take_profit_hit=None,
                ambiguous_both_hit=None,
                reference_exit_price=None,
                action=ProtectiveExitAction.UNRESOLVED_CAPITAL_EVENT,
                resulting_state=None,
                corporate_actions=validated_actions,
                applied_price_rescaling_factor=rescaling_factor,
                unresolved_capital_events=unresolved_events,
            )

        if expected_session != state.entry_session:
            if validated_bar.open <= effective_stop:
                return _decision(
                    state=state,
                    bar=validated_bar,
                    effective_stop=effective_stop,
                    effective_take_profit=effective_take_profit,
                    stop_hit=True,
                    take_profit_hit=False,
                    ambiguous_both_hit=False,
                    reference_exit_price=validated_bar.open,
                    action=ProtectiveExitAction.GAP_STOP_EXIT,
                    resulting_state=None,
                    corporate_actions=validated_actions,
                    applied_price_rescaling_factor=rescaling_factor,
                )
            if validated_bar.open >= effective_take_profit:
                return _decision(
                    state=state,
                    bar=validated_bar,
                    effective_stop=effective_stop,
                    effective_take_profit=effective_take_profit,
                    stop_hit=False,
                    take_profit_hit=True,
                    ambiguous_both_hit=False,
                    reference_exit_price=validated_bar.open,
                    action=ProtectiveExitAction.GAP_TAKE_PROFIT_EXIT,
                    resulting_state=None,
                    corporate_actions=validated_actions,
                    applied_price_rescaling_factor=rescaling_factor,
                )

        stop_hit = validated_bar.low <= effective_stop
        take_profit_hit = validated_bar.high >= effective_take_profit
        ambiguous_both_hit = stop_hit and take_profit_hit
        if stop_hit:
            return _decision(
                state=state,
                bar=validated_bar,
                effective_stop=effective_stop,
                effective_take_profit=effective_take_profit,
                stop_hit=stop_hit,
                take_profit_hit=take_profit_hit,
                ambiguous_both_hit=ambiguous_both_hit,
                reference_exit_price=effective_stop,
                action=ProtectiveExitAction.STOP_LOSS_EXIT,
                resulting_state=None,
                corporate_actions=validated_actions,
                applied_price_rescaling_factor=rescaling_factor,
            )
        if take_profit_hit:
            return _decision(
                state=state,
                bar=validated_bar,
                effective_stop=effective_stop,
                effective_take_profit=effective_take_profit,
                stop_hit=False,
                take_profit_hit=True,
                ambiguous_both_hit=False,
                reference_exit_price=effective_take_profit,
                action=ProtectiveExitAction.TAKE_PROFIT_EXIT,
                resulting_state=None,
                corporate_actions=validated_actions,
                applied_price_rescaling_factor=rescaling_factor,
            )

        resulting_state = _advanced_state(
            state=state,
            session=expected_session,
            effective_stop=effective_stop,
            effective_take_profit=effective_take_profit,
        )
        return _decision(
            state=state,
            bar=validated_bar,
            effective_stop=effective_stop,
            effective_take_profit=effective_take_profit,
            stop_hit=False,
            take_profit_hit=False,
            ambiguous_both_hit=False,
            reference_exit_price=None,
            action=ProtectiveExitAction.HOLD,
            resulting_state=resulting_state,
            corporate_actions=validated_actions,
            applied_price_rescaling_factor=rescaling_factor,
        )


__all__ = ["EXIT_SLIPPAGE_BPS", "ProtectiveExitService"]
