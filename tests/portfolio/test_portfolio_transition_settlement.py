"""Pending-settlement application tests for Phase 13B.2."""

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.portfolio.portfolio_errors import PortfolioStateError
from stock_swing_d1.portfolio.portfolio_events import (
    PortfolioEventKind,
    PortfolioEventReference,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import buy_event, sell_event, state_with_position


def _sold_state():
    previous = state_with_position()
    return PortfolioTransitionEngine.transition(
        previous, date(2026, 8, 19), (sell_event(),)
    ).resulting_state


def test_settlement_remains_pending_before_its_exact_session() -> None:
    sold = _sold_state()

    result = PortfolioTransitionEngine.transition(sold, date(2026, 8, 20), ())

    assert result.resulting_state.settled_cash == sold.settled_cash
    assert result.resulting_state.pending_settlements == sold.pending_settlements
    assert result.ledger_entries == ()


def test_due_settlement_credits_cash_removes_pending_and_records_fingerprint() -> None:
    sold = _sold_state()
    pending = sold.pending_settlements[0]

    result = PortfolioTransitionEngine.transition(
        sold, pending.settlement_session, ()
    )

    assert result.resulting_state.settled_cash == sold.settled_cash + pending.amount
    assert result.resulting_state.pending_settlements == ()
    fingerprint = result.resulting_state.applied_events[-1]
    assert fingerprint.event_kind is PortfolioEventKind.SETTLEMENT
    assert fingerprint.event_id == pending.settlement_id
    ledger = result.ledger_entries[0]
    assert ledger.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED
    assert ledger.settled_cash_delta == pending.amount
    assert ledger.pending_cash_delta == -pending.amount


def test_already_applied_settlement_cannot_be_credited_again() -> None:
    sold = _sold_state()
    pending = sold.pending_settlements[0]
    settled = PortfolioTransitionEngine.transition(
        sold, pending.settlement_session, ()
    ).resulting_state
    malformed = settled.model_copy(update={"pending_settlements": (pending,)})

    with pytest.raises(PortfolioStateError, match="already-applied"):
        PortfolioTransitionEngine.transition(
            malformed, date(2026, 8, 22), ()
        )

    assert settled.settled_cash == sold.settled_cash + pending.amount


def test_friday_sell_settles_on_supplied_monday_without_calendar_inference() -> None:
    entered = state_with_position(session=date(2026, 8, 20))
    friday = date(2026, 8, 21)
    monday = date(2026, 8, 24)
    event = sell_event(
        session=friday,
        settlement_session=monday,
    )
    sold = PortfolioTransitionEngine.transition(
        entered, friday, (event,)
    ).resulting_state

    result = PortfolioTransitionEngine.transition(sold, monday, ())

    assert result.resulting_state.as_of_session == monday
    assert result.resulting_state.pending_settlements == ()
    assert result.resulting_state.settled_cash == Decimal("1018")


def test_result_references_disambiguate_same_execution_and_settlement_id() -> None:
    entered = state_with_position()
    sold = PortfolioTransitionEngine.transition(
        entered,
        date(2026, 8, 19),
        (sell_event(settlement_id="X"),),
    ).resulting_state
    due_session = sold.pending_settlements[0].settlement_session
    same_id_execution = buy_event(
        "X",
        session=due_session,
        asset_id="NORGATE:2",
    )

    result = PortfolioTransitionEngine.transition(
        sold,
        due_session,
        (same_id_execution,),
    )

    assert result.newly_applied_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.SETTLEMENT,
            event_id="X",
        ),
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id="X",
        ),
    )
