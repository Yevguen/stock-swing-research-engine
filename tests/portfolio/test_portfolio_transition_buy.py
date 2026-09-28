"""BUY accounting tests for Phase 13B.2 transitions."""

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.portfolio.portfolio_errors import (
    InsufficientSettledCashError,
    PositionAlreadyOpenError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    PortfolioEventKind,
    PortfolioEventReference,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import (
    buy_event,
    initial_state,
    state_with_position,
)


def test_normal_buy_debits_settled_cash_and_opens_one_position() -> None:
    previous = initial_state(Decimal("100"))
    event = buy_event(quantity=2, fill_price=Decimal("10"), execution_cost=Decimal("1"))

    result = PortfolioTransitionEngine.transition(previous, event.session, (event,))

    state = result.resulting_state
    assert previous.settled_cash == Decimal("100")
    assert state.settled_cash == Decimal("79")
    assert state.state_version == 1
    assert state.as_of_session == event.session
    assert len(state.open_positions) == 1
    assert state.open_positions[0].cost_basis == Decimal("21")
    assert result.newly_applied_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id=event.execution_id,
        ),
    )
    ledger = result.ledger_entries[0]
    assert ledger.event_type is PortfolioLedgerEventType.BUY_APPLIED
    assert ledger.quantity_delta == 2
    assert ledger.settled_cash_delta == Decimal("-21")
    assert ledger.pending_cash_delta == 0


def test_exact_cash_buy_is_allowed() -> None:
    event = buy_event(
        quantity=3, fill_price=Decimal("10"), execution_cost=Decimal("2")
    )
    result = PortfolioTransitionEngine.transition(
        initial_state(Decimal("32")), event.session, (event,)
    )

    assert result.resulting_state.settled_cash == 0


def test_insufficient_cash_buy_is_rejected_without_mutation() -> None:
    previous = initial_state(Decimal("20"))
    event = buy_event()

    with pytest.raises(InsufficientSettledCashError):
        PortfolioTransitionEngine.transition(previous, event.session, (event,))

    assert previous == initial_state(Decimal("20"))


def test_buy_for_an_open_asset_is_rejected() -> None:
    previous = state_with_position()
    event = buy_event(
        "BUY-2", session=date(2026, 8, 19), asset_id="NORGATE:1"
    )

    with pytest.raises(PositionAlreadyOpenError):
        PortfolioTransitionEngine.transition(previous, event.session, (event,))


def test_multiple_independent_buys_apply_in_canonical_order() -> None:
    session = date(2026, 8, 18)
    events = (
        buy_event("BUY-20", session=session, asset_id="NORGATE:20"),
        buy_event("BUY-10", session=session, asset_id="NORGATE:10"),
    )

    result = PortfolioTransitionEngine.transition(initial_state(), session, events)

    assert [item.asset_id for item in result.resulting_state.open_positions] == [
        "NORGATE:10",
        "NORGATE:20",
    ]
    assert [item.source_event_id for item in result.ledger_entries] == [
        "BUY-10",
        "BUY-20",
    ]


def test_collectively_unaffordable_buys_fail_the_whole_session() -> None:
    previous = initial_state(Decimal("30"))
    events = (
        buy_event("BUY-1", asset_id="NORGATE:1"),
        buy_event("BUY-2", asset_id="NORGATE:2"),
    )

    with pytest.raises(InsufficientSettledCashError):
        PortfolioTransitionEngine.transition(
            previous, date(2026, 8, 18), events
        )

    assert previous == initial_state(Decimal("30"))
