"""Real Phase 9 to Phase 12 compatibility under cost-inclusive cash."""

from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.execution.costs import (
    BacktestExecutionCostService,
    load_backtest_execution_cost_policy,
)
from stock_swing_d1.execution.entry import (
    EntryExecutionService,
    EntryExecutionStatus,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.portfolio import (
    PortfolioAllocationService,
    PortfolioCandidateAction,
)


class EntryCalendar:
    def __init__(self, signal_session: date, entry_session: date) -> None:
        self.signal_session = signal_session
        self.entry_session = entry_session

    def next_session(self, session: date) -> date:
        if session != self.signal_session:
            raise ValueError("unexpected session")
        return self.entry_session

    def regular_session_open_time(self, session: date) -> datetime:
        if session != self.entry_session:
            raise ValueError("unexpected entry session")
        return datetime.combine(session, time(13, 30), tzinfo=timezone.utc)


class AllowingOverlay:
    @staticmethod
    def revalidate_pending_entry(**kwargs) -> EarningsIntegrationDecision:
        return EarningsIntegrationDecision(
            action=EarningsIntegrationAction.PENDING_ENTRY_ALLOWED,
            earnings_state=None,
            risk_decision=None,
        )


def _cost_service() -> BacktestExecutionCostService:
    return BacktestExecutionCostService(
        policy=load_backtest_execution_cost_policy(
            Path(__file__).resolve().parents[2] / "config" / "costs.yaml"
        )
    )


def _execute(admitted, candidate, *, opening_price: float):
    sized = admitted.sized_pending_entry
    assert sized is not None
    calendar = EntryCalendar(
        candidate.signal.signal_session,
        candidate.signal.planned_entry_session,
    )
    service = EntryExecutionService(
        earnings_overlay=AllowingOverlay(),
        trading_calendar=calendar,
        execution_cost_service=_cost_service(),
    )
    bar = StockBar(
        security_id=candidate.security_id,
        symbol=candidate.symbol,
        trading_date=candidate.signal.planned_entry_session,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=opening_price,
        high=opening_price + 1.0,
        low=opening_price - 1.0,
        close=opening_price,
        volume=1_000_000,
    )
    return service.execute_pending_entry(
        pending_entry=candidate.pending_entry,
        sized_pending_entry=sized,
        execution_bar=bar,
    )


def test_real_affordable_phase9_execution_settles_cost_inclusive_cash(
    make_portfolio, make_candidate
) -> None:
    candidate = make_candidate()
    admitted = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(), candidates=(candidate,)
    ).candidate_decisions[0]
    execution = _execute(admitted, candidate, opening_price=100.0)

    assert execution.status is EntryExecutionStatus.EXECUTED
    assert execution.execution_cost_quote is not None
    price_notional = Decimal(execution.executed_shares) * Decimal(
        str(execution.execution_price)
    )
    exact_total = (
        execution.execution_cost_quote.notional
        + execution.execution_cost_quote.execution_cost
    )
    assert exact_total - price_notional == (
        execution.execution_cost_quote.execution_cost
    )
    assert execution.actual_cash_required == float(exact_total)
    assert execution.actual_cash_required > float(price_notional)

    settlement = PortfolioAllocationService().settle_reservation(
        candidate_decision=admitted,
        entry_execution=execution,
    )
    assert settlement.actual_cash_used == execution.actual_cash_required
    assert settlement.released_cash == pytest.approx(
        settlement.reserved_cash - settlement.actual_cash_used
    )
    assert settlement.slot_released is False
    assert settlement.slot_occupied_after_execution is True


def test_real_cost_cancellation_releases_slot_without_backfill(
    make_portfolio,
    make_open_position,
    make_candidate,
) -> None:
    first_candidate = make_candidate("NORGATE:100")
    second_candidate = make_candidate("NORGATE:200")
    allocation = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(
            open_positions=tuple(
                make_open_position(f"OPEN:{number}") for number in range(4)
            )
        ),
        candidates=(first_candidate, second_candidate),
    )
    admitted, rejected = allocation.candidate_decisions
    assert admitted.action is PortfolioCandidateAction.ADMITTED
    assert rejected.action is (
        PortfolioCandidateAction.REJECTED_MAX_SIMULTANEOUS_POSITIONS
    )
    rejected_before = rejected
    sized = admitted.sized_pending_entry
    assert sized is not None
    desired_price_notional = Decimal(str(admitted.reserved_cash)) - Decimal(
        "0.50"
    )
    candidate_fill = desired_price_notional / Decimal(sized.fixed_shares)
    reference_open = float(candidate_fill / Decimal("1.0005"))

    execution = _execute(
        admitted, first_candidate, opening_price=reference_open
    )
    assert execution.candidate_execution_cost_quote is not None
    assert execution.status is (
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
    )
    assert execution.requested_shares == sized.fixed_shares
    assert execution.executed_shares == 0
    assert execution.candidate_execution_cost_quote.notional <= Decimal(
        str(admitted.reserved_cash)
    )
    assert execution.candidate_cash_required > admitted.reserved_cash

    settlement = PortfolioAllocationService().settle_reservation(
        candidate_decision=admitted,
        entry_execution=execution,
    )
    assert settlement.actual_cash_used == 0
    assert settlement.released_cash == admitted.reserved_cash
    assert settlement.slot_released is True
    assert settlement.slot_occupied_after_execution is False
    assert allocation.candidate_decisions[1] == rejected_before
