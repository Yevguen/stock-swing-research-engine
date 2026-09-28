"""Gap ordering and slippage tests for Phase 10."""

import pytest

from stock_swing_d1.execution.protective_exit import ProtectiveExitAction
from tests.execution.protective_exit.conftest import NEXT_SESSION


@pytest.mark.parametrize("opening_price", [91.0, 96.0])
def test_later_open_at_or_below_stop_is_gap_stop(
    held_state, service, make_bar, opening_price
) -> None:
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=opening_price,
            high=100.0,
            low=90.0,
            close=99.0,
        ),
    )

    assert decision.action is ProtectiveExitAction.GAP_STOP_EXIT
    assert decision.reference_exit_price == opening_price
    assert decision.execution_exit_price == opening_price * 0.9995


@pytest.mark.parametrize("opening_price", [108.0, 112.0])
def test_later_open_at_or_above_target_is_gap_take_profit(
    held_state, service, make_bar, opening_price
) -> None:
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=opening_price,
            high=opening_price + 2.0,
            low=100.0,
            close=opening_price + 1.0,
        ),
    )

    assert decision.action is ProtectiveExitAction.GAP_TAKE_PROFIT_EXIT
    assert decision.reference_exit_price == opening_price
    assert decision.execution_exit_price == opening_price * 0.9995


def test_gap_outcome_ignores_later_hlc(held_state, service, make_bar) -> None:
    first = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=91.0,
            high=92.0,
            low=90.0,
            close=91.0,
        ),
    )
    second = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=91.0,
            high=120.0,
            low=50.0,
            close=110.0,
        ),
    )

    assert first.action is second.action is ProtectiveExitAction.GAP_STOP_EXIT
    assert first.reference_exit_price == second.reference_exit_price == 91.0
    assert first.execution_exit_price == second.execution_exit_price
    assert first.execution_exit_price == 91.0 * 0.9995


@pytest.mark.parametrize(
    ("reference", "expected_slippage", "expected_execution"),
    [(100.0, 0.05, 99.95), (250.0, 0.125, 249.875)],
)
def test_exact_five_bps_adverse_exit_slippage(
    service,
    make_signal,
    make_entry,
    make_bar,
    reference,
    expected_slippage,
    expected_execution,
) -> None:
    state = service.create_state(
        signal=make_signal(atr_fraction=0.02),
        entry_execution=make_entry(execution_price=reference / 1.08),
    )
    decision = service.evaluate_session(
        state=state,
        bar=make_bar(
            open=reference,
            high=reference + 1.0,
            low=reference - 1.0,
            close=reference,
        ),
    )

    assert decision.action is ProtectiveExitAction.TAKE_PROFIT_EXIT
    assert decision.reference_exit_price == reference
    assert decision.slippage_amount == expected_slippage
    assert decision.execution_exit_price == expected_execution


def test_exit_calculation_has_no_intermediate_rounding(
    service, make_signal, make_entry, make_bar
) -> None:
    reference = 123.456789
    state = service.create_state(
        signal=make_signal(atr_fraction=0.02),
        entry_execution=make_entry(execution_price=reference / 1.08),
    )
    decision = service.evaluate_session(
        state=state,
        bar=make_bar(
            open=reference,
            high=reference + 1.0,
            low=reference - 1.0,
            close=reference,
        ),
    )

    assert decision.reference_exit_price == reference
    assert decision.slippage_amount == reference * 0.0005
    assert decision.execution_exit_price == reference * 0.9995
    assert decision.execution_exit_price != round(
        decision.execution_exit_price, 2
    )
