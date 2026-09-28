"""State-creation acceptance tests for Phase 10."""

from dataclasses import FrozenInstanceError
from datetime import timedelta
from math import inf, nan

import pytest

from stock_swing_d1.execution.entry import EntryExecutionStatus
from stock_swing_d1.execution.protective_exit import (
    ProtectiveExitState,
    ProtectiveExitValidationError,
)
from stock_swing_d1.strategy.baseline import BaselineSignalAction


def test_initial_state_uses_signal_atr_and_actual_execution_price(
    service, make_signal, make_entry
) -> None:
    signal = make_signal(atr_fraction=0.02)
    entry = make_entry(execution_price=100.0)

    state = service.create_state(signal=signal, entry_execution=entry)

    assert state.security_id == entry.security_id
    assert state.symbol == entry.symbol
    assert state.signal_session == signal.signal_session
    assert state.signal_time == signal.signal_time
    assert state.entry_session == entry.planned_entry_session
    assert state.entry_price == 100.0
    assert state.signal_atr_fraction == 0.02
    assert state.risk_fraction == 0.04
    assert state.stop_price == 96.0
    assert state.entry_price - state.stop_price == 4.0
    assert state.take_profit_price == 108.0
    assert state.last_evaluated_session is None
    assert state.entry_price != entry.reference_open
    assert state.entry_price != signal.adjusted_close


def test_state_is_immutable_and_direct_construction_is_closed(initial_state) -> None:
    with pytest.raises(FrozenInstanceError):
        initial_state.stop_price = 1.0
    with pytest.raises(TypeError):
        ProtectiveExitState(
            security_id="NORGATE:1001",
            symbol="PH10",
            signal_session=initial_state.signal_session,
            signal_time=initial_state.signal_time,
            entry_session=initial_state.entry_session,
            entry_price=100.0,
            signal_atr_fraction=0.02,
            risk_fraction=0.04,
            stop_price=1.0,
            take_profit_price=2.0,
            last_evaluated_session=None,
        )


@pytest.mark.parametrize("atr_fraction", [None, 0.0, -0.01, nan, inf, -inf])
def test_invalid_signal_atr_fraction_fails_closed(
    service, make_signal, make_entry, atr_fraction
) -> None:
    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.create_state(
            signal=make_signal(atr_fraction=atr_fraction),
            entry_execution=make_entry(),
        )

    assert raised.value.code == "INVALID_SIGNAL_ATR_FRACTION"


@pytest.mark.parametrize("atr_fraction", [0.5, 0.75, 1.0])
def test_risk_fraction_at_or_above_one_fails_closed(
    service, make_signal, make_entry, atr_fraction
) -> None:
    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.create_state(
            signal=make_signal(atr_fraction=atr_fraction),
            entry_execution=make_entry(),
        )

    assert raised.value.code == "INVALID_RISK_FRACTION"


@pytest.mark.parametrize("entry_price", [None, 0.0, -1.0, nan, inf, -inf])
def test_invalid_execution_price_fails_closed(
    service, make_signal, make_entry, entry_price
) -> None:
    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.create_state(
            signal=make_signal(),
            entry_execution=make_entry(execution_price=entry_price),
        )

    assert raised.value.code == "INVALID_ENTRY_PRICE"


def test_only_valid_long_signal_can_create_state(
    service, make_signal, make_entry
) -> None:
    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.create_state(
            signal=make_signal(action=BaselineSignalAction.NO_SIGNAL),
            entry_execution=make_entry(),
        )

    assert raised.value.code == "INVALID_SIGNAL"


@pytest.mark.parametrize(
    "status",
    [
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION,
        EntryExecutionStatus.INVALIDATED_BY_EARNINGS,
        EntryExecutionStatus.NO_EXECUTABLE_BAR,
        EntryExecutionStatus.INVALID_OPEN_PRICE,
    ],
)
def test_only_executed_entry_can_create_state(
    service, make_signal, make_entry, status
) -> None:
    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.create_state(
            signal=make_signal(),
            entry_execution=make_entry(
                status=status,
                execution_price=None,
            ),
        )

    assert raised.value.code == "INVALID_ENTRY_EXECUTION"


@pytest.mark.parametrize(
    "entry_override",
    [
        {"security_id": "NORGATE:1002"},
        {"signal_session": None},
        {"signal_time": None},
        {"planned_entry_session": None},
    ],
)
def test_upstream_identity_and_timing_must_match(
    service, make_signal, make_entry, entry_override
) -> None:
    signal = make_signal()
    overrides = dict(entry_override)
    if overrides.get("signal_session") is None and "signal_session" in overrides:
        overrides["signal_session"] = signal.signal_session + timedelta(days=1)
    if overrides.get("signal_time") is None and "signal_time" in overrides:
        overrides["signal_time"] = signal.signal_time + timedelta(minutes=1)
    if (
        overrides.get("planned_entry_session") is None
        and "planned_entry_session" in overrides
    ):
        overrides["planned_entry_session"] = signal.planned_entry_session + timedelta(
            days=1
        )
        overrides["execution_time"] = make_entry().execution_time + timedelta(days=1)

    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.create_state(
            signal=signal, entry_execution=make_entry(**overrides)
        )

    assert raised.value.code == "UPSTREAM_IDENTITY_MISMATCH"


def test_ticker_change_does_not_break_stable_identity(
    service, make_signal, make_entry
) -> None:
    state = service.create_state(
        signal=make_signal(symbol="OLD"),
        entry_execution=make_entry(symbol="NEW"),
    )

    assert state.security_id == "NORGATE:1001"
    assert state.symbol == "NEW"
