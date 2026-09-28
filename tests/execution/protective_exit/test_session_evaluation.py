"""Session ordering and ordinary touch tests for Phase 10."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, time

import pytest

from stock_swing_d1.execution.protective_exit import (
    ProtectiveExitAction,
    ProtectiveExitService,
    ProtectiveExitValidationError,
)
from stock_swing_d1.models import CorporateActionAdjustedStockBar
from tests.execution.protective_exit.conftest import (
    NEW_YORK,
    NEXT_SESSION,
    THIRD_SESSION,
    ExitCalendar,
)


def test_entry_session_stop_is_not_a_gap(initial_state, service, make_bar) -> None:
    decision = service.evaluate_session(
        state=initial_state,
        bar=make_bar(open=100.0, high=105.0, low=95.0, close=101.0),
    )

    assert decision.action is ProtectiveExitAction.STOP_LOSS_EXIT
    assert decision.reference_exit_price == 96.0


def test_entry_session_take_profit(initial_state, service, make_bar) -> None:
    decision = service.evaluate_session(
        state=initial_state,
        bar=make_bar(open=100.0, high=109.0, low=97.0, close=107.0),
    )

    assert decision.action is ProtectiveExitAction.TAKE_PROFIT_EXIT
    assert decision.reference_exit_price == 108.0


def test_entry_session_both_touched_is_stop_first(
    initial_state, service, make_bar
) -> None:
    decision = service.evaluate_session(
        state=initial_state,
        bar=make_bar(open=100.0, high=110.0, low=95.0, close=109.0),
    )

    assert decision.action is ProtectiveExitAction.STOP_LOSS_EXIT
    assert decision.ambiguous_both_hit is True
    assert decision.reference_exit_price == 96.0


def test_entry_open_is_not_a_preexisting_gap_trigger(
    initial_state, service, make_bar
) -> None:
    assert initial_state.entry_price != 90.0

    decision = service.evaluate_session(
        state=initial_state,
        bar=make_bar(open=90.0, high=105.0, low=89.0, close=100.0),
    )

    assert decision.action is ProtectiveExitAction.STOP_LOSS_EXIT
    assert decision.reference_exit_price == initial_state.stop_price


def test_hold_advances_new_immutable_state(initial_state, service, make_bar) -> None:
    decision = service.evaluate_session(state=initial_state, bar=make_bar())

    assert decision.action is ProtectiveExitAction.HOLD
    assert decision.reference_exit_price is None
    assert decision.slippage_amount is None
    assert decision.execution_exit_price is None
    assert decision.resulting_state is not None
    assert decision.resulting_state is not initial_state
    assert (
        decision.resulting_state.last_evaluated_session
        == initial_state.entry_session
    )
    assert initial_state.last_evaluated_session is None
    with pytest.raises(FrozenInstanceError):
        decision.action = ProtectiveExitAction.STOP_LOSS_EXIT
    with pytest.raises(FrozenInstanceError):
        decision.resulting_state.stop_price = 1.0


def test_later_normal_stop(held_state, service, make_bar) -> None:
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=100.0,
            high=105.0,
            low=95.0,
            close=101.0,
        ),
    )

    assert decision.action is ProtectiveExitAction.STOP_LOSS_EXIT
    assert decision.reference_exit_price == 96.0


def test_later_normal_take_profit(held_state, service, make_bar) -> None:
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=100.0,
            high=109.0,
            low=97.0,
            close=107.0,
        ),
    )

    assert decision.action is ProtectiveExitAction.TAKE_PROFIT_EXIT
    assert decision.reference_exit_price == 108.0


@pytest.mark.parametrize("close", [95.0, 109.0])
def test_close_cannot_resolve_same_bar_path(
    held_state, service, make_bar, close
) -> None:
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=100.0,
            high=110.0,
            low=95.0,
            close=close,
        ),
    )

    assert decision.action is ProtectiveExitAction.STOP_LOSS_EXIT
    assert decision.ambiguous_both_hit is True
    assert decision.reference_exit_price == 96.0


def test_exact_threshold_touches_are_inclusive(held_state, service, make_bar) -> None:
    stop = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=100.0,
            high=105.0,
            low=96.0,
            close=101.0,
        ),
    )
    target = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=100.0,
            high=108.0,
            low=97.0,
            close=107.0,
        ),
    )

    assert stop.action is ProtectiveExitAction.STOP_LOSS_EXIT
    assert target.action is ProtectiveExitAction.TAKE_PROFIT_EXIT


def test_missing_expected_bar_fails_closed(held_state, service) -> None:
    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.evaluate_session(state=held_state, bar=None)

    assert raised.value.code == "MISSING_EXIT_BAR"


def test_missing_entry_session_bar_also_fails_closed(initial_state, service) -> None:
    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.evaluate_session(state=initial_state, bar=None)

    assert raised.value.code == "MISSING_EXIT_BAR"


def test_skipped_session_fails_closed(held_state, service, make_bar) -> None:
    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.evaluate_session(
            state=held_state, bar=make_bar(trading_date=THIRD_SESSION)
        )

    assert raised.value.code == "EXIT_BAR_IDENTITY_MISMATCH"


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("security_id", "NORGATE:1002"),
        ("trading_date", THIRD_SESSION),
        ("timeframe", "H1"),
        ("session_type", "extended"),
        ("price_basis", "capital_special_adjusted"),
    ],
)
def test_wrong_bar_identity_fails_closed(
    held_state, service, make_bar, field_name, wrong_value
) -> None:
    bar = make_bar(trading_date=NEXT_SESSION).model_copy(
        update={field_name: wrong_value}
    )

    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.evaluate_session(state=held_state, bar=bar)

    assert raised.value.code == "EXIT_BAR_IDENTITY_MISMATCH"


def test_adjusted_bar_cannot_be_an_exit_fill(held_state, service) -> None:
    adjusted = CorporateActionAdjustedStockBar(
        security_id=held_state.security_id,
        symbol=held_state.symbol,
        trading_date=NEXT_SESSION,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="capital_special_adjusted",
        open=100.0,
        high=105.0,
        low=97.0,
        close=101.0,
        volume=1_000_000.0,
    )

    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.evaluate_session(state=held_state, bar=adjusted)

    assert raised.value.code == "INVALID_EXIT_BAR"


def test_bar_ticker_change_is_allowed_by_stable_identity(
    held_state, service, make_bar
) -> None:
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(symbol="RENAMED", trading_date=NEXT_SESSION),
    )

    assert decision.action is ProtectiveExitAction.HOLD


def test_repeated_evaluation_is_deterministic_and_does_not_mutate_inputs(
    held_state, service, make_bar
) -> None:
    bar = make_bar(trading_date=NEXT_SESSION)

    first = service.evaluate_session(state=held_state, bar=bar)
    second = service.evaluate_session(state=held_state, bar=bar)

    assert first == second
    assert held_state.last_evaluated_session != NEXT_SESSION
    assert bar.close == 101.0


def test_calendar_supports_friday_to_tuesday_holiday_boundary(
    make_signal, make_entry, make_bar
) -> None:
    friday = date(2026, 9, 4)
    tuesday = date(2026, 9, 8)
    wednesday = date(2026, 9, 9)
    calendar = ExitCalendar((friday, tuesday, wednesday))
    holiday_service = ProtectiveExitService(trading_calendar=calendar)
    signal_time = datetime.combine(friday, time(16), tzinfo=NEW_YORK)
    execution_time = datetime.combine(tuesday, time(9, 30), tzinfo=NEW_YORK)
    state = holiday_service.create_state(
        signal=make_signal(
            signal_session=friday,
            signal_time=signal_time,
            planned_entry_session=tuesday,
        ),
        entry_execution=make_entry(
            signal_session=friday,
            signal_time=signal_time,
            planned_entry_session=tuesday,
            execution_time=execution_time,
        ),
    )
    entry_decision = holiday_service.evaluate_session(
        state=state, bar=make_bar(trading_date=tuesday)
    )
    assert entry_decision.resulting_state is not None

    next_decision = holiday_service.evaluate_session(
        state=entry_decision.resulting_state,
        bar=make_bar(trading_date=wednesday),
    )

    assert next_decision.action is ProtectiveExitAction.HOLD
    assert calendar.calls == [tuesday]
