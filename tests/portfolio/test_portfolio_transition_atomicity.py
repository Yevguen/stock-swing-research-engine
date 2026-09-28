"""Whole-session atomicity tests for Phase 13B.2."""

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.portfolio.portfolio_errors import InsufficientSettledCashError
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import (
    buy_event,
    initial_state,
    sell_event,
    state_with_position,
)


def test_same_day_sale_proceeds_cannot_fund_buy_and_sell_is_not_committed() -> None:
    previous = state_with_position(cash=Decimal("121"))
    session = date(2026, 8, 19)
    sale = sell_event(
        session=session,
        fill_price=Decimal("500"),
        execution_cost=Decimal("0"),
    )
    purchase = buy_event(
        "BUY-2",
        session=session,
        asset_id="NORGATE:2",
        quantity=1,
        fill_price=Decimal("500"),
        execution_cost=Decimal("0"),
    )

    with pytest.raises(InsufficientSettledCashError):
        PortfolioTransitionEngine.transition(previous, session, (sale, purchase))

    assert previous.settled_cash == Decimal("100")
    assert len(previous.open_positions) == 1
    assert previous.pending_settlements == ()


def test_valid_valid_invalid_valid_batch_fails_wholly() -> None:
    previous = initial_state(Decimal("50"))
    session = date(2026, 8, 18)
    events = (
        buy_event(
            "BUY-1", session=session, asset_id="NORGATE:1", quantity=1
        ),
        buy_event(
            "BUY-2", session=session, asset_id="NORGATE:2", quantity=1
        ),
        buy_event(
            "BUY-3",
            session=session,
            asset_id="NORGATE:3",
            quantity=1,
            fill_price=Decimal("100"),
        ),
        buy_event(
            "BUY-4",
            session=session,
            asset_id="NORGATE:4",
            quantity=1,
            fill_price=Decimal("1"),
            execution_cost=Decimal("0"),
        ),
    )

    with pytest.raises(InsufficientSettledCashError):
        PortfolioTransitionEngine.transition(previous, session, events)

    assert previous == initial_state(Decimal("50"))


def test_due_settlement_is_not_committed_when_later_buy_fails() -> None:
    entered = state_with_position()
    sold = PortfolioTransitionEngine.transition(
        entered, date(2026, 8, 19), (sell_event(),)
    ).resulting_state
    due_session = sold.pending_settlements[0].settlement_session
    invalid_buy = buy_event(
        "BUY-2",
        session=due_session,
        asset_id="NORGATE:2",
        quantity=1,
        fill_price=Decimal("5000"),
        execution_cost=Decimal("0"),
    )

    with pytest.raises(InsufficientSettledCashError):
        PortfolioTransitionEngine.transition(sold, due_session, (invalid_buy,))

    assert len(sold.pending_settlements) == 1
    assert sold.settled_cash == entered.settled_cash
