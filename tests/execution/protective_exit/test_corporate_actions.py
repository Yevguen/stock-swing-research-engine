"""Corporate-action continuity tests for Phase 10."""

import pytest

from stock_swing_d1.execution.protective_exit import (
    ProtectiveExitAction,
    ProtectiveExitValidationError,
)
from tests.execution.protective_exit.conftest import (
    ENTRY_SESSION,
    NEXT_SESSION,
    SIGNAL_SESSION,
)


def test_verified_two_for_one_split_rescales_before_ohlc(
    held_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event(new_shares=2.0, old_shares=1.0)
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=50.0,
            high=53.0,
            low=49.0,
            close=52.0,
        ),
        corporate_actions=(event,),
    )

    assert decision.action is ProtectiveExitAction.HOLD
    assert decision.effective_stop == 48.0
    assert decision.effective_take_profit == 54.0
    assert decision.applied_price_rescaling_factor == 0.5
    assert decision.resulting_state is not None
    assert decision.resulting_state.stop_price == 48.0
    assert decision.resulting_state.take_profit_price == 54.0


def test_verified_reverse_split_rescales_upward(
    held_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event(
        event_type="reverse_split", new_shares=1.0, old_shares=5.0
    )
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=500.0,
            high=530.0,
            low=490.0,
            close=510.0,
        ),
        corporate_actions=(event,),
    )

    assert decision.action is ProtectiveExitAction.HOLD
    assert decision.effective_stop == 480.0
    assert decision.effective_take_profit == 540.0
    assert decision.applied_price_rescaling_factor == 5.0


def test_unresolved_capital_event_stops_raw_price_evaluation(
    held_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event(
        event_type="unknown_capital_event",
        terms_verified=False,
        new_shares=None,
        old_shares=None,
    )
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.0,
        ),
        corporate_actions=(event,),
    )

    assert decision.action is ProtectiveExitAction.UNRESOLVED_CAPITAL_EVENT
    assert decision.stop_hit is None
    assert decision.take_profit_hit is None
    assert decision.ambiguous_both_hit is None
    assert decision.reference_exit_price is None
    assert decision.slippage_amount is None
    assert decision.execution_exit_price is None
    assert decision.resulting_state is None
    assert decision.unresolved_capital_events == (event,)


def test_verified_unsupported_event_cannot_fall_through_to_gap_stop(
    held_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event(
        event_type="special_distribution",
        terms_verified=True,
        new_shares=None,
        old_shares=None,
    )
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.0,
        ),
        corporate_actions=(event,),
    )

    assert decision.action is ProtectiveExitAction.UNRESOLVED_CAPITAL_EVENT
    assert decision.stop_hit is None
    assert decision.take_profit_hit is None
    assert decision.ambiguous_both_hit is None
    assert decision.reference_exit_price is None
    assert decision.slippage_amount is None
    assert decision.execution_exit_price is None
    assert decision.resulting_state is None
    assert decision.unresolved_capital_events == (event,)


@pytest.mark.parametrize("event_type", ["split", "reverse_split"])
def test_unverified_share_ratio_event_fails_closed_without_rescaling(
    held_state, service, make_bar, make_capital_event, event_type
) -> None:
    event = make_capital_event(
        event_type=event_type,
        terms_verified=False,
        new_shares=None,
        old_shares=None,
    )
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.0,
        ),
        corporate_actions=(event,),
    )

    assert decision.action is ProtectiveExitAction.UNRESOLVED_CAPITAL_EVENT
    assert decision.applied_price_rescaling_factor == 1.0
    assert decision.effective_stop == held_state.stop_price
    assert decision.effective_take_profit == held_state.take_profit_price
    assert decision.reference_exit_price is None


def test_verified_split_without_usable_ratio_fails_closed(
    held_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event().model_copy(update={"old_shares": None})
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.0,
        ),
        corporate_actions=(event,),
    )

    assert decision.action is ProtectiveExitAction.UNRESOLVED_CAPITAL_EVENT
    assert decision.applied_price_rescaling_factor == 1.0
    assert decision.reference_exit_price is None


def test_unknown_date_semantics_on_current_unresolved_event_fails_closed(
    held_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event(
        event_date=NEXT_SESSION,
        date_semantics="unknown",
        event_type="unknown_capital_event",
        terms_verified=False,
        new_shares=None,
        old_shares=None,
    )
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.0,
        ),
        corporate_actions=(event,),
    )

    assert decision.action is ProtectiveExitAction.UNRESOLVED_CAPITAL_EVENT
    assert decision.reference_exit_price is None


def test_ordinary_cash_dividend_does_not_rescale(
    held_state, service, make_bar, make_dividend
) -> None:
    dividend = make_dividend()
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=100.0,
            high=105.0,
            low=97.0,
            close=101.0,
        ),
        corporate_actions=(dividend,),
    )

    assert decision.action is ProtectiveExitAction.HOLD
    assert decision.effective_stop == held_state.stop_price
    assert decision.effective_take_profit == held_state.take_profit_price
    assert decision.applied_price_rescaling_factor == 1.0


def test_entry_session_dividend_preserves_levels_and_normal_ohlc_logic(
    initial_state, service, make_bar, make_dividend
) -> None:
    dividend = make_dividend(entitlement_date=SIGNAL_SESSION)
    decision = service.evaluate_session(
        state=initial_state,
        bar=make_bar(open=100.0, high=105.0, low=95.0, close=101.0),
        corporate_actions=(dividend,),
    )

    assert decision.action is ProtectiveExitAction.STOP_LOSS_EXIT
    assert decision.effective_stop == initial_state.stop_price
    assert decision.effective_take_profit == initial_state.take_profit_price
    assert decision.applied_price_rescaling_factor == 1.0
    assert decision.reference_exit_price == initial_state.stop_price


def test_entry_session_unsupported_event_stops_low_high_evaluation(
    initial_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event(
        event_date=SIGNAL_SESSION,
        event_type="other",
        terms_verified=True,
    )
    decision = service.evaluate_session(
        state=initial_state,
        bar=make_bar(open=100.0, high=110.0, low=95.0, close=109.0),
        corporate_actions=(event,),
    )

    assert decision.action is ProtectiveExitAction.UNRESOLVED_CAPITAL_EVENT
    assert decision.stop_hit is None
    assert decision.take_profit_hit is None
    assert decision.ambiguous_both_hit is None
    assert decision.reference_exit_price is None
    assert decision.execution_exit_price is None
    assert decision.unresolved_capital_events == (event,)


def test_entry_session_split_is_not_double_applied(
    initial_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event(
        event_date=SIGNAL_SESSION,
        new_shares=2.0,
        old_shares=1.0,
    )
    decision = service.evaluate_session(
        state=initial_state,
        bar=make_bar(open=100.0, high=110.0, low=95.0, close=109.0),
        corporate_actions=(event,),
    )

    assert decision.action is ProtectiveExitAction.STOP_LOSS_EXIT
    assert decision.effective_stop == 96.0
    assert decision.effective_take_profit == 108.0
    assert decision.applied_price_rescaling_factor == 1.0
    assert decision.reference_exit_price == 96.0


def test_multiple_ratio_events_are_cumulative_and_order_independent(
    held_state, service, make_bar, make_capital_event
) -> None:
    split = make_capital_event(
        event_type="split", new_shares=2.0, old_shares=1.0
    )
    reverse = make_capital_event(
        event_type="reverse_split", new_shares=1.0, old_shares=4.0
    )
    bar = make_bar(
        trading_date=NEXT_SESSION,
        open=200.0,
        high=210.0,
        low=195.0,
        close=205.0,
    )

    first = service.evaluate_session(
        state=held_state, bar=bar, corporate_actions=(split, reverse)
    )
    second = service.evaluate_session(
        state=held_state, bar=bar, corporate_actions=(reverse, split)
    )

    assert first == second
    assert first.applied_price_rescaling_factor == 2.0
    assert first.effective_stop == 192.0
    assert first.effective_take_profit == 216.0


@pytest.mark.parametrize("include_dividend", [False, True])
def test_unsupported_event_dominates_mixed_session_actions(
    held_state,
    service,
    make_bar,
    make_capital_event,
    make_dividend,
    include_dividend,
) -> None:
    supported_split = make_capital_event()
    unsupported = make_capital_event(
        event_type="spinoff",
        terms_verified=True,
        new_shares=None,
        old_shares=None,
    )
    companion = make_dividend() if include_dividend else supported_split
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(
            trading_date=NEXT_SESSION,
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.0,
        ),
        corporate_actions=(companion, unsupported),
    )

    assert decision.action is ProtectiveExitAction.UNRESOLVED_CAPITAL_EVENT
    assert decision.applied_price_rescaling_factor == 1.0
    assert decision.effective_stop == held_state.stop_price
    assert decision.effective_take_profit == held_state.take_profit_price
    assert decision.stop_hit is None
    assert decision.reference_exit_price is None
    assert decision.unresolved_capital_events == (unsupported,)


def test_corporate_action_security_mismatch_fails_closed(
    held_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event(
        security_id="NORGATE:1002", source_asset_id=1002
    )

    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.evaluate_session(
            state=held_state,
            bar=make_bar(trading_date=NEXT_SESSION),
            corporate_actions=(event,),
        )

    assert raised.value.code == "CORPORATE_ACTION_IDENTITY_MISMATCH"


def test_corporate_action_must_be_effective_for_current_session(
    held_state, service, make_bar, make_capital_event
) -> None:
    wrong_event = make_capital_event(
        event_date=ENTRY_SESSION,
        date_semantics="effective_date",
    )

    with pytest.raises(ProtectiveExitValidationError) as raised:
        service.evaluate_session(
            state=held_state,
            bar=make_bar(trading_date=NEXT_SESSION),
            corporate_actions=(wrong_event,),
        )

    assert raised.value.code == "CORPORATE_ACTION_SESSION_MISMATCH"


def test_event_symbol_change_does_not_break_stable_identity(
    held_state, service, make_bar, make_capital_event
) -> None:
    event = make_capital_event(symbol="RENAMED")
    decision = service.evaluate_session(
        state=held_state,
        bar=make_bar(trading_date=NEXT_SESSION, symbol="OTHER"),
        corporate_actions=(event,),
    )

    assert decision.effective_stop == 48.0
