"""Shared synthetic helpers for focused Phase 13B.2 tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine


def buy_event(
    execution_id: str = "BUY-1",
    *,
    session: date = date(2026, 8, 18),
    asset_id: str = "NORGATE:1",
    quantity: int = 2,
    fill_price: Decimal = Decimal("10"),
    execution_cost: Decimal = Decimal("1"),
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"ORDER-{execution_id}",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
    )


def sell_event(
    execution_id: str = "SELL-1",
    *,
    session: date = date(2026, 8, 19),
    asset_id: str = "NORGATE:1",
    quantity: int = 2,
    fill_price: Decimal = Decimal("20"),
    execution_cost: Decimal = Decimal("1"),
    settlement_id: str = "SETTLE-1",
    settlement_session: date = date(2026, 8, 21),
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"ORDER-{execution_id}",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.SELL,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
        settlement_id=settlement_id,
        settlement_session=settlement_session,
    )


def initial_state(cash: Decimal = Decimal("1000")) -> PortfolioState:
    return PortfolioState(settled_cash=cash)


def state_with_position(
    *,
    cash: Decimal = Decimal("1000"),
    session: date = date(2026, 8, 18),
    event: PortfolioExecutionEvent | None = None,
) -> PortfolioState:
    result = PortfolioTransitionEngine.transition(
        initial_state(cash),
        session,
        (event or buy_event(session=session),),
    )
    return result.resulting_state
