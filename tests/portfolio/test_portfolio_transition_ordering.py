"""Canonical ordering, chronology, and empty-session tests for Phase 13B.2."""

from datetime import date
from itertools import permutations

import pytest

from stock_swing_d1.portfolio.portfolio_errors import (
    OutOfOrderSessionError,
    OverdueSettlementError,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import (
    buy_event,
    initial_state,
    sell_event,
)


def test_input_permutations_produce_identical_state_ledger_and_hashes() -> None:
    entry_session = date(2026, 8, 18)
    previous = PortfolioTransitionEngine.transition(
        initial_state(),
        entry_session,
        (
            buy_event("BUY-1", session=entry_session, asset_id="NORGATE:1"),
            buy_event("BUY-2", session=entry_session, asset_id="NORGATE:2"),
        ),
    ).resulting_state
    session = date(2026, 8, 19)
    events = (
        sell_event(
            "SELL-2",
            session=session,
            asset_id="NORGATE:2",
            settlement_id="SETTLE-2",
        ),
        buy_event("BUY-4", session=session, asset_id="NORGATE:4"),
        sell_event(
            "SELL-1",
            session=session,
            asset_id="NORGATE:1",
            settlement_id="SETTLE-1",
        ),
        buy_event("BUY-3", session=session, asset_id="NORGATE:3"),
    )

    results = [
        PortfolioTransitionEngine.transition(previous, session, order)
        for order in permutations(events)
    ]

    assert all(result == results[0] for result in results[1:])
    assert [item.source_event_id for item in results[0].ledger_entries] == [
        "SELL-1",
        "SELL-2",
        "BUY-3",
        "BUY-4",
    ]
    assert [item.sequence_in_session for item in results[0].ledger_entries] == [
        0,
        1,
        2,
        3,
    ]


def test_first_and_noncontiguous_later_sessions_are_accepted() -> None:
    first = PortfolioTransitionEngine.transition(
        initial_state(), date(2026, 8, 10), ()
    )
    later = PortfolioTransitionEngine.transition(
        first.resulting_state, date(2026, 8, 20), ()
    )

    assert first.resulting_state.state_version == 1
    assert later.resulting_state.state_version == 2
    assert later.resulting_state.as_of_session == date(2026, 8, 20)


@pytest.mark.parametrize("session", [date(2026, 8, 20), date(2026, 8, 19)])
def test_same_or_earlier_transition_session_is_rejected(session: date) -> None:
    previous = PortfolioTransitionEngine.transition(
        initial_state(), date(2026, 8, 20), ()
    ).resulting_state

    with pytest.raises(OutOfOrderSessionError):
        PortfolioTransitionEngine.transition(previous, session, ())


def test_overdue_settlement_rejects_transition() -> None:
    entry_session = date(2026, 8, 18)
    entered = PortfolioTransitionEngine.transition(
        initial_state(), entry_session, (buy_event(session=entry_session),)
    ).resulting_state
    sold = PortfolioTransitionEngine.transition(
        entered, date(2026, 8, 19), (sell_event(),)
    ).resulting_state

    with pytest.raises(OverdueSettlementError):
        PortfolioTransitionEngine.transition(sold, date(2026, 8, 22), ())


def test_empty_later_session_advances_date_and_version_once() -> None:
    first = PortfolioTransitionEngine.transition(
        initial_state(), date(2026, 8, 18), ()
    ).resulting_state
    second = PortfolioTransitionEngine.transition(
        first, date(2026, 8, 19), ()
    ).resulting_state

    assert first.as_of_session == date(2026, 8, 18)
    assert first.state_version == 1
    assert second.as_of_session == date(2026, 8, 19)
    assert second.state_version == 2
    assert second.settled_cash == first.settled_cash
    assert second.open_positions == first.open_positions
    assert second.pending_settlements == first.pending_settlements
    assert second.applied_events == first.applied_events
