"""Full-exit SELL accounting tests for Phase 13B.2 transitions."""

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.portfolio.portfolio_errors import (
    InvalidExitQuantityError,
    PositionNotFoundError,
    SettlementIntegrityError,
)
from stock_swing_d1.portfolio.portfolio_events import PortfolioLedgerEventType
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import (
    buy_event,
    initial_state,
    sell_event,
    state_with_position,
)


def test_valid_full_exit_creates_pending_proceeds_without_settled_cash() -> None:
    previous = state_with_position()
    event = sell_event()

    result = PortfolioTransitionEngine.transition(previous, event.session, (event,))

    state = result.resulting_state
    assert state.settled_cash == previous.settled_cash
    assert state.open_positions == ()
    assert len(state.pending_settlements) == 1
    pending = state.pending_settlements[0]
    assert pending.amount == Decimal("39")
    assert pending.source_execution_id == event.execution_id
    assert pending.trade_session == event.session
    ledger = result.ledger_entries[0]
    assert ledger.event_type is PortfolioLedgerEventType.SELL_APPLIED
    assert ledger.quantity_delta == -2
    assert ledger.settled_cash_delta == 0
    assert ledger.pending_cash_delta == Decimal("39")


def test_sell_without_open_position_is_rejected() -> None:
    event = sell_event()

    with pytest.raises(PositionNotFoundError):
        PortfolioTransitionEngine.transition(
            initial_state(), event.session, (event,)
        )


@pytest.mark.parametrize("quantity", [1, 3])
def test_partial_exit_and_oversell_are_rejected(quantity: int) -> None:
    previous = state_with_position()
    event = sell_event(quantity=quantity)

    with pytest.raises(InvalidExitQuantityError):
        PortfolioTransitionEngine.transition(previous, event.session, (event,))


@pytest.mark.parametrize("execution_cost", [Decimal("40"), Decimal("41")])
def test_nonpositive_net_proceeds_are_rejected(
    execution_cost: Decimal,
) -> None:
    previous = state_with_position()
    event = sell_event(execution_cost=execution_cost)

    with pytest.raises(SettlementIntegrityError, match="positive"):
        PortfolioTransitionEngine.transition(previous, event.session, (event,))


def test_duplicate_settlement_identity_rejects_entire_sell_batch() -> None:
    entry_session = date(2026, 8, 18)
    previous = PortfolioTransitionEngine.transition(
        initial_state(),
        entry_session,
        (
            buy_event("BUY-1", session=entry_session, asset_id="NORGATE:1"),
            buy_event("BUY-2", session=entry_session, asset_id="NORGATE:2"),
        ),
    ).resulting_state
    sell_session = date(2026, 8, 19)
    events = (
        sell_event(
            "SELL-1",
            session=sell_session,
            asset_id="NORGATE:1",
            settlement_id="SETTLE-SAME",
        ),
        sell_event(
            "SELL-2",
            session=sell_session,
            asset_id="NORGATE:2",
            settlement_id="SETTLE-SAME",
        ),
    )

    with pytest.raises(SettlementIntegrityError, match="not unique"):
        PortfolioTransitionEngine.transition(previous, sell_session, events)

    assert len(previous.open_positions) == 2
    assert previous.pending_settlements == ()
