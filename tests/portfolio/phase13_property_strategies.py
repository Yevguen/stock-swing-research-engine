"""Bounded Hypothesis strategies for Phase 13 portfolio properties."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from hypothesis import strategies as st

from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioSessionInput,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine


CENT = Decimal("0.01")
BASE_SESSION = date(2026, 1, 5)

positive_money = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("500"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)
nonnegative_cost = st.decimals(
    min_value=Decimal("0"),
    max_value=Decimal("25"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)
positive_cash = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("20000"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)
quantities = st.integers(min_value=1, max_value=20)
asset_numbers = st.integers(min_value=1, max_value=500)
session_offsets = st.integers(min_value=0, max_value=120)


@dataclass(frozen=True, slots=True)
class BuyCase:
    previous_state: PortfolioState
    execution: PortfolioExecutionEvent
    required_cash: Decimal


@dataclass(frozen=True, slots=True)
class SellCase:
    previous_state: PortfolioState
    execution: PortfolioExecutionEvent
    net_proceeds: Decimal


@dataclass(frozen=True, slots=True)
class HistoryCase:
    initial_state: PortfolioState
    sessions: tuple[PortfolioSessionInput, ...]


@st.composite
def funded_buy_cases(draw) -> BuyCase:
    quantity = draw(quantities)
    fill_price = draw(positive_money)
    execution_cost = draw(nonnegative_cost)
    buffer = draw(
        st.decimals(
            min_value=Decimal("0"),
            max_value=Decimal("1000"),
            places=2,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    asset_number = draw(asset_numbers)
    session = BASE_SESSION + timedelta(days=draw(session_offsets))
    required_cash = Decimal(quantity) * fill_price + execution_cost
    event = PortfolioExecutionEvent(
        execution_id=f"BUY-{asset_number}-{session.toordinal()}",
        source_order_id=f"ORDER-{asset_number}-{session.toordinal()}",
        session=session,
        asset_id=f"NORGATE:{asset_number}",
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
    )
    return BuyCase(
        previous_state=PortfolioState(settled_cash=required_cash + buffer),
        execution=event,
        required_cash=required_cash,
    )


@st.composite
def insufficient_buy_cases(draw) -> BuyCase:
    quantity = draw(quantities)
    fill_price = draw(positive_money)
    execution_cost = draw(nonnegative_cost)
    asset_number = draw(asset_numbers)
    session = BASE_SESSION + timedelta(days=draw(session_offsets))
    required_cash = Decimal(quantity) * fill_price + execution_cost
    shortfall = draw(
        st.decimals(
            min_value=CENT,
            max_value=required_cash,
            places=2,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    event = PortfolioExecutionEvent(
        execution_id=f"BUY-{asset_number}-{session.toordinal()}",
        source_order_id=f"ORDER-{asset_number}-{session.toordinal()}",
        session=session,
        asset_id=f"NORGATE:{asset_number}",
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
    )
    return BuyCase(
        previous_state=PortfolioState(settled_cash=required_cash - shortfall),
        execution=event,
        required_cash=required_cash,
    )


@st.composite
def full_sell_cases(draw) -> SellCase:
    buy_case = draw(funded_buy_cases())
    entered = PortfolioTransitionEngine.transition(
        buy_case.previous_state,
        buy_case.execution.session,
        (buy_case.execution,),
    ).resulting_state
    sell_session = buy_case.execution.session + timedelta(days=1)
    fill_price = draw(positive_money)
    gross_proceeds = Decimal(buy_case.execution.quantity) * fill_price
    execution_cost = draw(
        st.decimals(
            min_value=Decimal("0"),
            max_value=gross_proceeds - CENT,
            places=2,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    settlement_session = sell_session + timedelta(
        days=draw(st.integers(min_value=1, max_value=5))
    )
    event = PortfolioExecutionEvent(
        execution_id=f"SELL-{buy_case.execution.execution_id}",
        source_order_id=f"ORDER-SELL-{buy_case.execution.execution_id}",
        session=sell_session,
        asset_id=buy_case.execution.asset_id,
        side=ExecutionSide.SELL,
        quantity=buy_case.execution.quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
        settlement_id=f"SETTLE-{buy_case.execution.execution_id}",
        settlement_session=settlement_session,
    )
    return SellCase(
        previous_state=entered,
        execution=event,
        net_proceeds=gross_proceeds - execution_cost,
    )


@st.composite
def valid_history_cases(draw) -> HistoryCase:
    buy_case = draw(funded_buy_cases())
    buy = buy_case.execution
    replay_session = buy.session + timedelta(days=1)
    empty_session = replay_session + timedelta(days=1)
    sell_session = empty_session + timedelta(days=1)
    settlement_session = sell_session + timedelta(days=1)
    sell_price = draw(positive_money)
    gross_proceeds = Decimal(buy.quantity) * sell_price
    sell_cost = draw(
        st.decimals(
            min_value=Decimal("0"),
            max_value=gross_proceeds - CENT,
            places=2,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    sell = PortfolioExecutionEvent(
        execution_id=f"SELL-{buy.execution_id}",
        source_order_id=f"ORDER-SELL-{buy.execution_id}",
        session=sell_session,
        asset_id=buy.asset_id,
        side=ExecutionSide.SELL,
        quantity=buy.quantity,
        fill_price=sell_price,
        execution_cost=sell_cost,
        settlement_id=f"SETTLE-{buy.execution_id}",
        settlement_session=settlement_session,
    )
    return HistoryCase(
        initial_state=buy_case.previous_state,
        sessions=(
            PortfolioSessionInput(session=buy.session, execution_events=(buy,)),
            PortfolioSessionInput(
                session=replay_session,
                execution_events=(buy,),
            ),
            PortfolioSessionInput(session=empty_session),
            PortfolioSessionInput(
                session=sell_session,
                execution_events=(sell,),
            ),
            PortfolioSessionInput(session=settlement_session),
        ),
    )


__all__ = [
    "BASE_SESSION",
    "BuyCase",
    "CENT",
    "HistoryCase",
    "SellCase",
    "asset_numbers",
    "full_sell_cases",
    "funded_buy_cases",
    "insufficient_buy_cases",
    "nonnegative_cost",
    "positive_cash",
    "positive_money",
    "quantities",
    "session_offsets",
    "valid_history_cases",
]
