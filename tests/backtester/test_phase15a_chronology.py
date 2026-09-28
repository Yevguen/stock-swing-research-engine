"""Phase 15A chronology, ownership, and anti-look-ahead acceptance tests."""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

import stock_swing_d1.backtester as backtester_package
import stock_swing_d1.backtester.decision_interval as decision_interval_module
from stock_swing_d1.backtester import (
    HistoricalDecisionInterval,
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionInput,
    HistoricalBacktestValidationError,
)
from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
    PublishedEarningsRiskOverlay,
)
from stock_swing_d1.earnings.models import (
    LifecycleState,
    TimingClass,
    TransitionType,
)
from stock_swing_d1.earnings.pit import reconstruct_earnings_state
from stock_swing_d1.execution.entry import (
    EntryExecutionService,
    create_pending_entry,
)
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    CorporateActionEvent,
    StockBar,
)
from stock_swing_d1.portfolio import (
    PORTFOLIO_ALLOCATION_POLICY_REF,
    PortfolioAllocationService,
)
from stock_swing_d1.portfolio.portfolio_errors import (
    InsufficientSettledCashError,
    OutOfOrderSessionError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from stock_swing_d1.ranking import RankedAllocationAdapter
from stock_swing_d1.strategy.baseline import BaselineSignalEvaluator
from tests.backtester.conftest import FRIDAY, MONDAY, TUESDAY, decision_time
from tests.earnings._builders import make_revision


DECISION_INTERVAL = HistoricalDecisionInterval(
    decision_start_date=date(2026, 8, 1),
    decision_end_date=date(2026, 9, 30),
)


def empty_input(
    session: date,
    *,
    next_session: date | None = None,
    **overrides,
) -> HistoricalBacktestSessionInput:
    values = {
        "session": session,
        "decision_time": decision_time(session),
        "next_session": next_session,
    }
    values.update(overrides)
    return HistoricalBacktestSessionInput(**values)


def buy_event(
    session: date,
    *,
    asset_id: str = "NORGATE:1",
    execution_id: str = "BUY-1",
    quantity: int = 1,
    fill_price: Decimal = Decimal("100"),
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"ORDER-{execution_id}",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=Decimal("0"),
    )


def sell_event(
    session: date,
    *,
    asset_id: str = "NORGATE:1",
    execution_id: str = "SELL-1",
    quantity: int = 1,
    fill_price: Decimal = Decimal("110"),
    settlement_session: date = TUESDAY,
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"ORDER-{execution_id}",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.SELL,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=Decimal("0"),
        settlement_id=f"SETTLE-{execution_id}",
        settlement_session=settlement_session,
    )


def state_with_position(
    *,
    cash_after_entry: Decimal,
    asset_id: str = "NORGATE:1",
) -> PortfolioState:
    entry_session = date(2026, 8, 20)
    initial = PortfolioState(
        settled_cash=cash_after_entry + Decimal("100")
    )
    return PortfolioTransitionEngine.transition(
        initial,
        entry_session,
        (
            buy_event(
                entry_session,
                asset_id=asset_id,
                execution_id="OLD-BUY",
                fill_price=Decimal("100"),
            ),
        ),
    ).resulting_state


class RecordingTransition:
    def __init__(self) -> None:
        self.calls = []

    def transition(self, previous_state, session, events, dividend_evidence=()):
        self.calls.append((previous_state, session, tuple(events)))
        return PortfolioTransitionEngine.transition(
            previous_state, session, events, dividend_evidence=dividend_evidence
        )


class EntryCalendar:
    def next_session(self, session: date) -> date:
        candidate = session + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

    def previous_session(self, session: date) -> date:
        candidate = session - timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
        return candidate

    def session_distance(self, start: date, end: date) -> int:
        if start == end:
            return 0
        if start > end:
            return -self.session_distance(end, start)
        distance = 0
        cursor = start
        while cursor < end:
            cursor = self.next_session(cursor)
            distance += 1
        if cursor != end:
            raise ValueError("end is not a trading session")
        return distance

    def decision_time(self, session: date) -> datetime:
        return decision_time(session)

    def regular_session_open_time(self, session: date) -> datetime:
        return datetime(
            session.year,
            session.month,
            session.day,
            13,
            30,
            tzinfo=timezone.utc,
        )

    def signal_decision_time(self, session: date) -> datetime:
        return decision_time(session)


class EntryEarningsOverlay:
    def __init__(self, action=EarningsIntegrationAction.PENDING_ENTRY_ALLOWED):
        self.action = action
        self.calls = []

    def revalidate_pending_entry(self, **kwargs):
        self.calls.append(kwargs)
        return EarningsIntegrationDecision(
            action=self.action,
            earnings_state=None,
            risk_decision=None,
        )

    def evaluate_entry_candidate(self, **kwargs):
        return EarningsIntegrationDecision(
            action=EarningsIntegrationAction.ENTRY_ALLOWED,
            earnings_state=None,
            risk_decision=None,
        )


class RevisionPITQuery:
    """Recording PIT boundary backed by the real revision replay function."""

    build_id = "PHASE15A-REVISION-PIT"
    output_sha256 = "a" * 64

    def __init__(self, revisions) -> None:
        self.revisions = tuple(revisions)
        self.calls = []

    def query(
        self,
        *,
        canonical_asset_id,
        as_of,
        decision_session,
        provider_name=None,
    ):
        self.calls.append(
            {
                "canonical_asset_id": canonical_asset_id,
                "as_of": as_of,
                "decision_session": decision_session,
                "provider_name": provider_name,
            }
        )
        return reconstruct_earnings_state(self.revisions, as_of)


class ReexpressedExecutionTimeService:
    """Delegate to Phase 9, then retain an equivalent zoned representation."""

    def __init__(self, delegate, target_timezone) -> None:
        self.delegate = delegate
        self.target_timezone = target_timezone

    def create_pending_entry(self, *, signal):
        return self.delegate.create_pending_entry(signal=signal)

    def execute_pending_entry(self, **kwargs):
        decision = self.delegate.execute_pending_entry(**kwargs)
        return replace(
            decision,
            execution_time=decision.execution_time.astimezone(
                self.target_timezone
            ),
        )


def entry_protective_collaborators() -> dict[str, object]:
    """Frozen Phase 10/15B owners required by every executed entry (5C-B).

    The entry-session protective evaluation lives in the one shared session
    body, so the legacy ``run``/``process_session`` paths need the same
    collaborators as provider mode whenever a Phase 9 entry executes.
    """

    from stock_swing_d1.execution.open_position_exit import (
        OpenPositionExitEvaluator,
    )
    from stock_swing_d1.execution.protective_exit import ProtectiveExitService

    calendar = EntryCalendar()
    return {
        "protective_exit_state_factory": ProtectiveExitService(
            trading_calendar=calendar
        ),
        "open_position_exit_evaluator": OpenPositionExitEvaluator(
            trading_calendar=calendar
        ),
    }


class EntryEventAdapter:
    def build_buy_event(self, *, session, intent, entry_execution):
        return PortfolioExecutionEvent(
            execution_id=f"ENTRY-{session.isoformat()}-{intent.security_id}",
            source_order_id=(
                f"ALLOC-{intent.allocation_session.isoformat()}-{intent.security_id}"
            ),
            session=session,
            asset_id=intent.security_id,
            side=ExecutionSide.BUY,
            quantity=entry_execution.executed_shares,
            fill_price=Decimal(str(entry_execution.execution_price)),
            execution_cost=entry_execution.execution_cost_quote.execution_cost,
        )


def entry_bar(security_id: str, *, opening_price: float = 100.0) -> StockBar:
    return StockBar(
        security_id=security_id,
        symbol=security_id.replace("NORGATE:", "S"),
        trading_date=MONDAY,
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


def adjusted_history(security_id: str) -> tuple[CorporateActionAdjustedStockBar, ...]:
    sessions = []
    cursor = FRIDAY
    while len(sessions) < 200:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor -= timedelta(days=1)
    sessions.reverse()
    bars = []
    for index, session in enumerate(sessions):
        close = 100.0 + index
        bars.append(
            CorporateActionAdjustedStockBar(
                security_id=security_id,
                symbol=security_id.replace("NORGATE:", "S"),
                trading_date=session,
                timeframe="D1",
                session_type="regular",
                currency="USD",
                price_basis="capital_special_adjusted",
                open=close,
                high=close + 2.0,
                low=close - 2.0,
                close=close,
                volume=1_000_000.0,
            )
        )
    return tuple(bars)


def test_public_contract_is_narrow_and_models_are_frozen() -> None:
    assert {
        "HistoricalBacktestEntryIntent",
        "HistoricalDecisionInterval",
        "HistoricalBacktestOrchestrator",
        "HistoricalBacktestRunResult",
        "HistoricalBacktestSessionInput",
        "HistoricalBacktestSessionResult",
        "HistoricalBacktestValidationError",
    } <= set(backtester_package.__all__)
    item = empty_input(FRIDAY)

    with pytest.raises(ValidationError, match="frozen"):
        item.session = MONDAY
    with pytest.raises(ValidationError, match="Extra inputs"):
        HistoricalBacktestSessionInput(
            session=FRIDAY,
            decision_time=decision_time(FRIDAY),
            generated_at=datetime.now(timezone.utc),
        )


@pytest.mark.parametrize(
    "sessions",
    [
        (empty_input(FRIDAY), empty_input(FRIDAY)),
        (empty_input(MONDAY), empty_input(FRIDAY)),
    ],
)
def test_duplicate_or_backward_sessions_fail_without_sorting(sessions) -> None:
    recording = RecordingTransition()
    orchestrator = HistoricalBacktestOrchestrator(
        transition_service=recording
    )

    with pytest.raises(OutOfOrderSessionError, match="strict chronological"):
        orchestrator.run(
            PortfolioState(settled_cash=Decimal("1000")),
            sessions,
            decision_interval=DECISION_INTERVAL,
        )

    assert recording.calls == []


def test_weekend_gap_creates_no_artificial_sessions_and_is_deterministic() -> None:
    sessions = (
        empty_input(FRIDAY, next_session=MONDAY),
        empty_input(MONDAY),
    )
    initial = PortfolioState(settled_cash=Decimal("1000"))
    orchestrator = HistoricalBacktestOrchestrator()

    first = orchestrator.run(
        initial, sessions, decision_interval=DECISION_INTERVAL
    )
    second = orchestrator.run(
        initial, sessions, decision_interval=DECISION_INTERVAL
    )

    assert first == second
    assert tuple(item.session for item in first.session_results) == (
        FRIDAY,
        MONDAY,
    )
    assert tuple(
        item.authoritative_state.state_version for item in first.session_results
    ) == (1, 2)
    assert first.session_results[0].ranking_snapshot.candidate_count == 0


def test_phase13_is_called_once_per_session_and_state_threads_forward() -> None:
    recording = RecordingTransition()
    initial = PortfolioState(settled_cash=Decimal("1000"))
    result = HistoricalBacktestOrchestrator(
        transition_service=recording
    ).run(
        initial,
        (
            empty_input(FRIDAY, next_session=MONDAY),
            empty_input(MONDAY),
        ),
        decision_interval=DECISION_INTERVAL,
    )

    assert len(recording.calls) == 2
    assert recording.calls[0][0] is initial
    assert recording.calls[1][0] == result.session_results[0].authoritative_state
    assert recording.calls[1][0].as_of_session == FRIDAY


def test_same_boundary_exits_precede_entries_with_security_id_tiebreak() -> None:
    initial = state_with_position(
        cash_after_entry=Decimal("1000"),
        asset_id="NORGATE:2",
    )
    events = (
        buy_event(FRIDAY, asset_id="NORGATE:3", execution_id="BUY-3"),
        sell_event(FRIDAY, asset_id="NORGATE:2"),
        buy_event(FRIDAY, asset_id="NORGATE:1", execution_id="BUY-1"),
    )

    result = HistoricalBacktestOrchestrator().run(
        initial,
        (empty_input(FRIDAY, scheduled_execution_events=events),),
        decision_interval=DECISION_INTERVAL,
    )

    assert tuple(
        (event.side, event.asset_id)
        for event in result.session_results[0].ordered_execution_events
    ) == (
        (ExecutionSide.SELL, "NORGATE:2"),
        (ExecutionSide.BUY, "NORGATE:1"),
        (ExecutionSide.BUY, "NORGATE:3"),
    )


def test_failed_phase13_transition_publishes_no_result_or_state_mutation() -> None:
    initial = PortfolioState(settled_cash=Decimal("10"))
    before = initial.model_dump(mode="python")

    with pytest.raises(InsufficientSettledCashError):
        HistoricalBacktestOrchestrator().run(
            initial,
            (
                empty_input(
                    FRIDAY,
                    scheduled_execution_events=(
                        buy_event(FRIDAY, fill_price=Decimal("100")),
                    ),
                ),
            ),
            decision_interval=DECISION_INTERVAL,
        )

    assert initial.model_dump(mode="python") == before
    assert initial.as_of_session is None


def test_complete_ranked_batch_reaches_distinct_phase12_path_unchanged(
    make_candidate_pair,
    make_portfolio_snapshot,
) -> None:
    a = make_candidate_pair(
        "NORGATE:100", sma20=102.0, sma50=100.0, atr14=2.0
    )
    b = make_candidate_pair(
        "NORGATE:200", sma20=106.0, sma50=100.0, atr14=2.0
    )
    c = make_candidate_pair(
        "NORGATE:300", sma20=104.0, sma50=100.0, atr14=2.0
    )

    class RecordingRankedAllocation:
        def __init__(self) -> None:
            self.calls = []
            self.real = PortfolioAllocationService()

        def allocate_ranked_candidates(self, *, portfolio, ranked_batch):
            self.calls.append(ranked_batch)
            return self.real.allocate_ranked_candidates(
                portfolio=portfolio, ranked_batch=ranked_batch
            )

    allocation = RecordingRankedAllocation()
    item = empty_input(
        FRIDAY,
        next_session=MONDAY,
        ranking_candidates=(a[0], b[0], c[0]),
        allocation_candidates=(a[1], b[1], c[1]),
        allocation_portfolio=make_portfolio_snapshot(),
    )
    result = HistoricalBacktestOrchestrator(
        allocation_service=allocation
    ).run(
        PortfolioState(settled_cash=Decimal("10000")),
        (item,),
        decision_interval=DECISION_INTERVAL,
    )

    assert len(allocation.calls) == 1
    assert [
        candidate.security_id for candidate in allocation.calls[0].candidates
    ] == ["NORGATE:200", "NORGATE:300", "NORGATE:100"]
    session_result = result.session_results[0]
    assert session_result.allocation_decision is not None
    assert session_result.allocation_decision.allocation_policy_ref == (
        PORTFOLIO_ALLOCATION_POLICY_REF
    )
    assert [
        decision.security_id
        for decision in session_result.allocation_decision.candidate_decisions
    ] == ["NORGATE:200", "NORGATE:300", "NORGATE:100"]
    assert session_result.authoritative_state.open_positions == ()
    assert all(
        intent.planned_entry_session == MONDAY
        for intent in session_result.future_entry_intents
    )


def test_completed_t_raw_pipeline_uses_frozen_phase7_and_phase8_owners(
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    security_id = "NORGATE:100"
    history = adjusted_history(security_id)
    current = history[-1]
    sizing_bar = StockBar(
        security_id=security_id,
        symbol=current.symbol,
        trading_date=FRIDAY,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=current.close,
        high=current.close + 1.0,
        low=current.close - 1.0,
        close=current.close,
        volume=1_000_000,
    )
    overlay = EntryEarningsOverlay()
    calendar = EntryCalendar()
    entry_service = EntryExecutionService(
        earnings_overlay=overlay,
        trading_calendar=calendar,
        execution_cost_service=execution_cost_service,
    )
    result = HistoricalBacktestOrchestrator(
        baseline_signal_evaluator=BaselineSignalEvaluator(
            earnings_overlay=overlay,
            trading_calendar=calendar,
        ),
        entry_execution_service=entry_service,
    ).run(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            empty_input(
                FRIDAY,
                next_session=MONDAY,
                indicator_history_bars=history,
                completed_unadjusted_bars=(sizing_bar,),
                universe_eligible_security_ids=(security_id,),
                allocation_portfolio=make_portfolio_snapshot(),
            ),
        ),
        decision_interval=DECISION_INTERVAL,
    )
    session = result.session_results[0]

    assert len(session.indicator_rows) == 1
    assert session.indicator_rows[0].trading_date == FRIDAY
    assert len(session.signal_decisions) == 1
    assert session.signal_decisions[0].valid_long_signal is True
    assert session.ranking_snapshot.candidate_count == 1
    assert session.ranked_allocation_batch.candidate_count == 1
    assert len(session.future_entry_intents) == 1
    assert session.authoritative_state.open_positions == ()


def test_completed_t_allocation_becomes_open_only_through_future_execution(
    make_candidate_pair,
    make_portfolio_snapshot,
) -> None:
    ranking, candidate = make_candidate_pair("NORGATE:100")
    initial = PortfolioState(settled_cash=Decimal("10000"))
    friday = HistoricalBacktestOrchestrator().process_session(
        prior_state=initial,
        session_input=empty_input(
            FRIDAY,
            next_session=MONDAY,
            ranking_candidates=(ranking,),
            allocation_candidates=(candidate,),
            allocation_portfolio=make_portfolio_snapshot(),
        ),
    )
    intent = friday.future_entry_intents[0]
    fixed_quantity = intent.candidate_decision.sized_pending_entry.fixed_shares

    monday = HistoricalBacktestOrchestrator().process_session(
        prior_state=friday.authoritative_state,
        session_input=empty_input(
            MONDAY,
            scheduled_execution_events=(
                buy_event(
                    MONDAY,
                    asset_id=intent.security_id,
                    quantity=fixed_quantity,
                    fill_price=Decimal("100"),
                ),
            ),
        ),
    )

    assert friday.authoritative_state.open_positions == ()
    assert friday.authoritative_state.as_of_session == FRIDAY
    assert monday.authoritative_state.open_positions[0].asset_id == intent.security_id
    assert monday.authoritative_state.open_positions[0].quantity == fixed_quantity


def test_run_carries_intent_and_invokes_phase9_before_one_phase13_call_per_t(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    ranking, candidate = make_candidate_pair("NORGATE:100")
    recording = RecordingTransition()
    overlay = EntryEarningsOverlay()
    entry_service = EntryExecutionService(
        earnings_overlay=overlay,
        trading_calendar=EntryCalendar(),
        execution_cost_service=execution_cost_service,
    )
    result = HistoricalBacktestOrchestrator(
        transition_service=recording,
        entry_execution_service=entry_service,
        entry_event_adapter=EntryEventAdapter(),
        **entry_protective_collaborators(),
    ).run(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            empty_input(
                FRIDAY,
                next_session=MONDAY,
                ranking_candidates=(ranking,),
                allocation_candidates=(candidate,),
                allocation_portfolio=make_portfolio_snapshot(),
            ),
            empty_input(
                MONDAY,
                next_session=TUESDAY,
                entry_execution_bars=(entry_bar("NORGATE:100"),),
            ),
        ),
        decision_interval=DECISION_INTERVAL,
    )

    monday = result.session_results[1]
    assert len(recording.calls) == 2
    assert len(overlay.calls) == 1
    assert len(monday.scheduled_entry_intents) == 1
    assert monday.entry_execution_decisions[0].status.value == "EXECUTED"
    # Task 5C-B: the executed entry was evaluated on its own session (HOLD).
    (entry_decision,) = monday.entry_session_protective_decisions
    assert entry_decision.exit_required is False
    assert entry_decision.holding_session_number == 1
    assert (
        entry_decision.protective_exit_decision.resulting_state
        .last_evaluated_session
        == MONDAY
    )
    assert monday.reservation_settlements[0].slot_occupied_after_execution is True
    assert monday.ordered_execution_events[0].quantity == (
        monday.scheduled_entry_intents[0]
        .candidate_decision.sized_pending_entry.fixed_shares
    )
    assert result.final_state.open_positions[0].asset_id == "NORGATE:100"


def test_scheduled_entry_accepts_cross_date_timezone_representation(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    """Phase 9/15A accept a zoned execution time whose calendar date differs
    from the planned entry session; since Task 5C-B every executed entry
    then meets the frozen Phase 10 owner on its own session, and Phase 10's
    frozen ``INVALID_UPSTREAM_TIMELINE`` rule (execution_time.date() must
    equal planned_entry_session) fails the run closed there -- no Phase 15A
    re-expression of a Phase 9 decision is permitted."""

    ranking, candidate = make_candidate_pair("NORGATE:100")
    overlay = EntryEarningsOverlay()
    calendar = EntryCalendar()
    boundary_timezone = timezone(timedelta(hours=14))
    entry_service = ReexpressedExecutionTimeService(
        EntryExecutionService(
            earnings_overlay=overlay,
            trading_calendar=calendar,
            execution_cost_service=execution_cost_service,
        ),
        boundary_timezone,
    )

    friday = HistoricalBacktestOrchestrator(
        entry_execution_service=entry_service,
        entry_event_adapter=EntryEventAdapter(),
        **entry_protective_collaborators(),
    ).process_session(
        prior_state=PortfolioState(settled_cash=Decimal("10000")),
        session_input=empty_input(
            FRIDAY,
            next_session=MONDAY,
            ranking_candidates=(ranking,),
            allocation_candidates=(candidate,),
            allocation_portfolio=make_portfolio_snapshot(),
        ),
    )
    intent = friday.future_entry_intents[0]
    execution = entry_service.execute_pending_entry(
        pending_entry=intent.allocation_candidate.pending_entry,
        sized_pending_entry=intent.candidate_decision.sized_pending_entry,
        execution_bar=entry_bar("NORGATE:100"),
    )
    # Phase 9 itself executed and retained the equivalent zoned instant.
    assert execution.status.value == "EXECUTED"
    assert execution.planned_entry_session == MONDAY
    assert execution.execution_time.date() == TUESDAY
    assert execution.execution_time.tzinfo is boundary_timezone
    assert execution.execution_time == datetime(
        2026, 8, 25, 3, 30, tzinfo=boundary_timezone
    )
    assert execution.execution_time.astimezone(timezone.utc) == datetime(
        2026, 8, 24, 13, 30, tzinfo=timezone.utc
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="PROTECTIVE_STATE_CREATION_FAILED",
    ) as error:
        HistoricalBacktestOrchestrator(
            entry_execution_service=entry_service,
            entry_event_adapter=EntryEventAdapter(),
            **entry_protective_collaborators(),
        ).run(
            PortfolioState(settled_cash=Decimal("10000")),
            (
                empty_input(
                    FRIDAY,
                    next_session=MONDAY,
                    ranking_candidates=(ranking,),
                    allocation_candidates=(candidate,),
                    allocation_portfolio=make_portfolio_snapshot(),
                ),
                empty_input(
                    MONDAY,
                    next_session=TUESDAY,
                    entry_execution_bars=(entry_bar("NORGATE:100"),),
                ),
            ),
            decision_interval=DECISION_INTERVAL,
        )
    assert error.value.__cause__.code == "INVALID_UPSTREAM_TIMELINE"


def test_terminal_future_entry_failure_never_reranks_or_backfills_prior_cycle(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    ranking, candidate = make_candidate_pair("NORGATE:100")
    overlay = EntryEarningsOverlay(
        EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
    )
    result = HistoricalBacktestOrchestrator(
        entry_execution_service=EntryExecutionService(
            earnings_overlay=overlay,
            trading_calendar=EntryCalendar(),
            execution_cost_service=execution_cost_service,
        ),
        entry_event_adapter=EntryEventAdapter(),
    ).run(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            empty_input(
                FRIDAY,
                next_session=MONDAY,
                ranking_candidates=(ranking,),
                allocation_candidates=(candidate,),
                allocation_portfolio=make_portfolio_snapshot(),
            ),
            empty_input(MONDAY, next_session=TUESDAY),
        ),
        decision_interval=DECISION_INTERVAL,
    )
    friday, monday = result.session_results

    assert friday.ranking_snapshot.candidate_count == 1
    assert friday.allocation_decision.candidate_decisions[0].security_id == (
        "NORGATE:100"
    )
    assert monday.entry_execution_decisions[0].status.value == (
        "INVALIDATED_BY_EARNINGS"
    )
    assert monday.reservation_settlements[0].slot_released is True
    assert monday.ordered_execution_events == ()
    assert monday.ranking_snapshot.candidate_count == 0
    assert result.final_state.open_positions == ()


def test_future_execution_values_cannot_change_earlier_ranking_or_allocation(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    ranking, candidate = make_candidate_pair("NORGATE:100")
    friday_input = empty_input(
        FRIDAY,
        next_session=MONDAY,
        ranking_candidates=(ranking,),
        allocation_candidates=(candidate,),
        allocation_portfolio=make_portfolio_snapshot(),
    )
    initial = PortfolioState(settled_cash=Decimal("10000"))

    def run_with_open(opening_price: float):
        overlay = EntryEarningsOverlay()
        return HistoricalBacktestOrchestrator(
            entry_execution_service=EntryExecutionService(
                earnings_overlay=overlay,
                trading_calendar=EntryCalendar(),
                execution_cost_service=execution_cost_service,
            ),
            entry_event_adapter=EntryEventAdapter(),
            **entry_protective_collaborators(),
        ).run(
            initial,
            (
                friday_input,
                empty_input(
                    MONDAY,
                    next_session=TUESDAY,
                    entry_execution_bars=(
                        entry_bar(
                            "NORGATE:100",
                            opening_price=opening_price,
                        ),
                    ),
                ),
            ),
            decision_interval=DECISION_INTERVAL,
        )

    ordinary_open = run_with_open(100.0)
    adverse_open = run_with_open(120.0)
    ordinary_friday = ordinary_open.session_results[0]
    adverse_friday = adverse_open.session_results[0]

    assert ordinary_friday.ranking_snapshot == adverse_friday.ranking_snapshot
    assert ordinary_friday.ranked_allocation_batch == (
        adverse_friday.ranked_allocation_batch
    )
    assert ordinary_friday.allocation_decision == (
        adverse_friday.allocation_decision
    )
    assert ordinary_friday.future_entry_intents == (
        adverse_friday.future_entry_intents
    )
    assert (
        ordinary_open.session_results[1].entry_execution_decisions
        != adverse_open.session_results[1].entry_execution_decisions
    )


def test_later_earnings_revision_cannot_change_prior_session_artifacts(
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    security_id = "NORGATE:100"
    known_at = datetime(2026, 8, 20, 19, tzinfo=timezone.utc)
    initial_revision = make_revision(
        event_revision_id="PHASE15A-INITIAL",
        event_instance_id="PHASE15A-EARNINGS",
        canonical_asset_id=security_id,
        provider_record_id="PHASE15A-INITIAL-RECORD",
        knowledge_date=known_at.date(),
        knowledge_available_at=known_at,
        strategy_effective_at=known_at,
        provider_sequence=1,
        transition_type=TransitionType.CONFIRMED,
        lifecycle_state=LifecycleState.CONFIRMED,
        scheduled_date=date(2026, 9, 30),
        timing_class=TimingClass.BMO,
        ingested_at=known_at + timedelta(minutes=5),
    )
    revised_at = datetime(2026, 8, 24, 13, tzinfo=timezone.utc)
    later_revision = replace(
        initial_revision,
        event_revision_id="PHASE15A-LATER",
        provider_record_id="PHASE15A-LATER-RECORD",
        knowledge_date=revised_at.date(),
        knowledge_available_at=revised_at,
        strategy_effective_at=revised_at,
        provider_sequence=2,
        transition_type=TransitionType.ADVANCED,
        scheduled_date=TUESDAY,
        ingested_at=revised_at + timedelta(minutes=5),
    )
    history = adjusted_history(security_id)
    current = history[-1]
    sizing_bar = StockBar(
        security_id=security_id,
        symbol=current.symbol,
        trading_date=FRIDAY,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=current.close,
        high=current.close + 1.0,
        low=current.close - 1.0,
        close=current.close,
        volume=1_000_000,
    )
    calendar = EntryCalendar()

    def run_with_revisions(*revisions):
        query = RevisionPITQuery(revisions)
        overlay = PublishedEarningsRiskOverlay(
            query,
            provider_name="SYNTHETIC",
            trading_calendar=calendar,
        )
        entry_service = EntryExecutionService(
            earnings_overlay=overlay,
            trading_calendar=calendar,
            execution_cost_service=execution_cost_service,
        )
        result = HistoricalBacktestOrchestrator(
            baseline_signal_evaluator=BaselineSignalEvaluator(
                earnings_overlay=overlay,
                trading_calendar=calendar,
            ),
            entry_execution_service=entry_service,
            entry_event_adapter=EntryEventAdapter(),
            **entry_protective_collaborators(),
        ).run(
            PortfolioState(settled_cash=Decimal("10000")),
            (
                empty_input(
                    FRIDAY,
                    next_session=MONDAY,
                    indicator_history_bars=history,
                    completed_unadjusted_bars=(sizing_bar,),
                    universe_eligible_security_ids=(security_id,),
                    allocation_portfolio=make_portfolio_snapshot(),
                ),
                empty_input(
                    MONDAY,
                    next_session=TUESDAY,
                    entry_execution_bars=(entry_bar(security_id),),
                ),
            ),
            decision_interval=DECISION_INTERVAL,
        )
        return result, query

    original, original_query = run_with_revisions(initial_revision)
    revised, revised_query = run_with_revisions(
        initial_revision,
        later_revision,
    )
    original_friday = original.session_results[0]
    revised_friday = revised.session_results[0]

    assert original_query.calls[0]["as_of"] == decision_time(FRIDAY)
    assert revised_query.calls[0]["as_of"] == decision_time(FRIDAY)
    assert revised_at > revised_query.calls[0]["as_of"]
    assert revised_query.calls[1]["as_of"] == datetime(
        2026, 8, 24, 13, 30, tzinfo=timezone.utc
    )
    assert revised_at <= revised_query.calls[1]["as_of"]
    assert original_friday.signal_decisions == revised_friday.signal_decisions
    assert original_friday.ranking_snapshot == revised_friday.ranking_snapshot
    assert original_friday.ranked_allocation_batch == (
        revised_friday.ranked_allocation_batch
    )
    assert original_friday.allocation_decision == (
        revised_friday.allocation_decision
    )
    assert original_friday.future_entry_intents == (
        revised_friday.future_entry_intents
    )
    original_status = original.session_results[1].entry_execution_decisions[0].status
    revised_status = revised.session_results[1].entry_execution_decisions[0].status
    assert original_status.value == "EXECUTED"
    assert revised_status.value == "INVALIDATED_BY_EARNINGS"


def test_poisoned_future_indicator_bar_is_rejected_before_phase13() -> None:
    history = adjusted_history("NORGATE:100")
    future_bar = history[-1].model_copy(update={"trading_date": MONDAY})
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="FUTURE_DATA_BOUNDARY_VIOLATION",
    ):
        HistoricalBacktestOrchestrator(
            transition_service=recording
        ).run(
            PortfolioState(settled_cash=Decimal("10000")),
            (
                empty_input(
                    FRIDAY,
                    next_session=MONDAY,
                    indicator_history_bars=history + (future_bar,),
                ),
            ),
            decision_interval=DECISION_INTERVAL,
        )

    assert recording.calls == []


def test_ranking_allocation_missing_or_extra_fails_before_phase13(
    make_candidate_pair,
    make_portfolio_snapshot,
) -> None:
    ranking, candidate = make_candidate_pair("NORGATE:100")
    other_ranking, _ = make_candidate_pair("NORGATE:200")
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="RANKING_ALLOCATION_RECONCILIATION_MISMATCH",
    ):
        HistoricalBacktestOrchestrator(
            transition_service=recording
        ).run(
            PortfolioState(settled_cash=Decimal("10000")),
            (
                empty_input(
                    FRIDAY,
                    next_session=MONDAY,
                    ranking_candidates=(ranking, other_ranking),
                    allocation_candidates=(candidate,),
                    allocation_portfolio=make_portfolio_snapshot(),
                ),
            ),
            decision_interval=DECISION_INTERVAL,
        )

    assert recording.calls == []


def test_sale_frees_slot_but_unsettled_proceeds_are_not_allocation_cash(
    make_candidate_pair,
    make_portfolio_snapshot,
) -> None:
    initial = state_with_position(cash_after_entry=Decimal("100"))
    ranking, candidate = make_candidate_pair("NORGATE:2")
    valid_portfolio = make_portfolio_snapshot(cash=100.0, equity=1100.0)
    result = HistoricalBacktestOrchestrator().run(
        initial,
        (
            empty_input(
                FRIDAY,
                next_session=MONDAY,
                scheduled_execution_events=(sell_event(FRIDAY),),
                ranking_candidates=(ranking,),
                allocation_candidates=(candidate,),
                allocation_portfolio=valid_portfolio,
            ),
        ),
        decision_interval=DECISION_INTERVAL,
    )
    state = result.final_state

    assert state.open_positions == ()
    assert state.settled_cash == Decimal("100")
    assert state.pending_settlements[0].amount == Decimal("110")

    leaked_cash = make_portfolio_snapshot(cash=210.0, equity=1100.0)
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="NONAUTHORITATIVE_ALLOCATION_CASH",
    ):
        HistoricalBacktestOrchestrator().run(
            initial,
            (
                empty_input(
                    FRIDAY,
                    next_session=MONDAY,
                    scheduled_execution_events=(sell_event(FRIDAY),),
                    ranking_candidates=(ranking,),
                    allocation_candidates=(candidate,),
                    allocation_portfolio=leaked_cash,
                ),
            ),
            decision_interval=DECISION_INTERVAL,
        )


def test_stock_split_seam_fails_closed_before_transition() -> None:
    event = CorporateActionEvent(
        security_id="NORGATE:1",
        symbol="S1",
        event_date=FRIDAY,
        event_type="split",
        source_asset_id=1,
        date_semantics="effective_date",
        old_shares=1.0,
        new_shares=2.0,
        terms_verified=True,
        source_provider="Norgate Data",
    )
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="UNSUPPORTED_CORPORATE_ACTION_TRANSITION",
    ):
        HistoricalBacktestOrchestrator(
            transition_service=recording
        ).run(
            PortfolioState(settled_cash=Decimal("1000")),
            (
                empty_input(
                    FRIDAY,
                    pre_open_corporate_actions=(event,),
                ),
            ),
            decision_interval=DECISION_INTERVAL,
        )

    assert recording.calls == []


def test_next_session_identity_is_explicit_and_never_calendar_arithmetic(
    make_candidate_pair,
    make_portfolio_snapshot,
) -> None:
    ranking, candidate = make_candidate_pair("NORGATE:1")

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="NEXT_SESSION_MISMATCH",
    ):
        HistoricalBacktestOrchestrator().run(
            PortfolioState(settled_cash=Decimal("10000")),
            (
                empty_input(
                    FRIDAY,
                    next_session=TUESDAY,
                    ranking_candidates=(ranking,),
                    allocation_candidates=(candidate,),
                    allocation_portfolio=make_portfolio_snapshot(),
                ),
                empty_input(MONDAY),
            ),
            decision_interval=DECISION_INTERVAL,
        )


def test_final_open_positions_are_returned_without_liquidation() -> None:
    initial = PortfolioState(settled_cash=Decimal("1000"))
    result = HistoricalBacktestOrchestrator().run(
        initial,
        (
            empty_input(
                FRIDAY,
                scheduled_execution_events=(buy_event(FRIDAY),),
            ),
        ),
        decision_interval=DECISION_INTERVAL,
    )

    assert len(result.final_state.open_positions) == 1
    assert result.final_state.open_positions[0].asset_id == "NORGATE:1"


def test_result_collections_and_nested_artifacts_are_immutable() -> None:
    result = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        (empty_input(FRIDAY),),
        decision_interval=DECISION_INTERVAL,
    )
    session = result.session_results[0]

    assert isinstance(result.session_results, tuple)
    assert isinstance(session.ordered_execution_events, tuple)
    assert isinstance(session.future_entry_intents, tuple)
    with pytest.raises(ValidationError, match="frozen"):
        result.final_state = PortfolioState()
    with pytest.raises(ValidationError, match="frozen"):
        session.session = MONDAY
    with pytest.raises(TypeError):
        result.session_results[0] = session


def test_historical_decision_interval_is_immutable_and_compares_by_value() -> None:
    first = HistoricalDecisionInterval(
        decision_start_date=date(2024, 2, 29),
        decision_end_date=date(2024, 3, 9),
    )
    second = HistoricalDecisionInterval(
        decision_start_date=date(2024, 2, 29),
        decision_end_date=date(2024, 3, 9),
    )

    assert first == second
    assert first is not second
    with pytest.raises(ValidationError, match="frozen"):
        first.decision_end_date = date(2024, 3, 10)


@pytest.mark.parametrize(
    "endpoint",
    [datetime(2024, 1, 1), "2024-01-01", None, 1],
)
@pytest.mark.parametrize(
    "field_name", ["decision_start_date", "decision_end_date"]
)
def test_historical_decision_interval_rejects_non_exact_dates(
    endpoint, field_name
) -> None:
    values = {
        "decision_start_date": date(2024, 1, 1),
        "decision_end_date": date(2024, 1, 2),
    }
    values[field_name] = endpoint

    with pytest.raises(ValidationError, match="exact Python datetime.date"):
        HistoricalDecisionInterval(**values)


def test_historical_decision_interval_accepts_ordered_equal_and_nontrading_dates() -> None:
    ordered = HistoricalDecisionInterval(
        decision_start_date=date(2007, 1, 1),
        decision_end_date=date(2007, 1, 7),
    )
    equal = HistoricalDecisionInterval(
        decision_start_date=date(2007, 1, 6),
        decision_end_date=date(2007, 1, 6),
    )

    assert ordered.decision_start_date == date(2007, 1, 1)
    assert ordered.decision_end_date == date(2007, 1, 7)
    assert equal.decision_start_date == equal.decision_end_date


def test_historical_decision_interval_rejects_reversed_endpoints() -> None:
    with pytest.raises(ValidationError, match="on or before"):
        HistoricalDecisionInterval(
            decision_start_date=date(2024, 1, 2),
            decision_end_date=date(2024, 1, 1),
        )


def test_run_requires_keyword_only_nonoptional_decision_interval() -> None:
    parameter = inspect.signature(
        HistoricalBacktestOrchestrator.run
    ).parameters["decision_interval"]
    orchestrator = HistoricalBacktestOrchestrator()
    state = PortfolioState(settled_cash=Decimal("1000"))

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert parameter.annotation == "HistoricalDecisionInterval"
    with pytest.raises(TypeError, match="decision_interval"):
        orchestrator.run(state, ())
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="INVALID_DECISION_INTERVAL",
    ):
        orchestrator.run(state, (), decision_interval=None)


def test_processed_sessions_may_be_strictly_inside_decision_interval() -> None:
    interval = HistoricalDecisionInterval(
        decision_start_date=date(2026, 8, 1),
        decision_end_date=date(2026, 8, 31),
    )

    result = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        (empty_input(FRIDAY, next_session=MONDAY), empty_input(MONDAY)),
        decision_interval=interval,
    )

    assert result.decision_interval == interval
    assert result.schema_version == "historical_backtest_run_result.v0.2"
    assert result.session_results[0].session > interval.decision_start_date
    assert result.session_results[-1].session < interval.decision_end_date


@pytest.mark.parametrize(
    ("interval", "offending_session"),
    [
        (
            HistoricalDecisionInterval(
                decision_start_date=MONDAY,
                decision_end_date=TUESDAY,
            ),
            FRIDAY,
        ),
        (
            HistoricalDecisionInterval(
                decision_start_date=FRIDAY,
                decision_end_date=FRIDAY,
            ),
            MONDAY,
        ),
    ],
)
def test_out_of_interval_session_fails_before_any_orchestration(
    interval, offending_session
) -> None:
    recording = RecordingTransition()

    class RecordingOrchestrator(HistoricalBacktestOrchestrator):
        processed: list[date] = []

        def _process_session(self, **kwargs):
            self.processed.append(kwargs["session_input"].session)
            return super()._process_session(**kwargs)

    orchestrator = RecordingOrchestrator(transition_service=recording)
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="SESSION_OUTSIDE_DECISION_INTERVAL",
    ):
        orchestrator.run(
            PortfolioState(settled_cash=Decimal("1000")),
            (empty_input(offending_session),),
            decision_interval=interval,
        )

    assert orchestrator.processed == []
    assert recording.calls == []


def test_existing_chronology_failure_precedes_interval_failure() -> None:
    recording = RecordingTransition()
    interval = HistoricalDecisionInterval(
        decision_start_date=MONDAY,
        decision_end_date=TUESDAY,
    )

    with pytest.raises(OutOfOrderSessionError, match="strict chronological"):
        HistoricalBacktestOrchestrator(
            transition_service=recording
        ).run(
            PortfolioState(settled_cash=Decimal("1000")),
            (empty_input(FRIDAY), empty_input(FRIDAY)),
            decision_interval=interval,
        )

    assert recording.calls == []


def test_prestart_lookback_is_allowed_on_first_and_later_processed_sessions() -> None:
    history = adjusted_history("NORGATE:100")
    monday_bar = history[-1].model_copy(update={"trading_date": MONDAY})
    sessions = (
        empty_input(
            FRIDAY,
            next_session=MONDAY,
            indicator_history_bars=history,
        ),
        empty_input(
            MONDAY,
            next_session=TUESDAY,
            indicator_history_bars=history + (monday_bar,),
        ),
    )
    narrow = HistoricalDecisionInterval(
        decision_start_date=FRIDAY,
        decision_end_date=MONDAY,
    )
    wide = HistoricalDecisionInterval(
        decision_start_date=date(2007, 1, 1),
        decision_end_date=date(2026, 12, 31),
    )

    def orchestrator() -> HistoricalBacktestOrchestrator:
        class PendingEntryOwner:
            @staticmethod
            def create_pending_entry(*, signal):
                return create_pending_entry(signal=signal)

            @staticmethod
            def execute_pending_entry(**_kwargs):
                raise AssertionError("no entry execution is scheduled")

        return HistoricalBacktestOrchestrator(
            baseline_signal_evaluator=BaselineSignalEvaluator(
                earnings_overlay=EntryEarningsOverlay(),
                trading_calendar=EntryCalendar(),
            ),
            entry_execution_service=PendingEntryOwner(),
        )

    narrow_result = orchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        sessions,
        decision_interval=narrow,
    )
    wide_result = orchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        sessions,
        decision_interval=wide,
    )

    assert all(
        any(
            bar.trading_date < narrow.decision_start_date
            for bar in item.indicator_history_bars
        )
        for item in sessions
    )
    assert narrow_result.final_state == wide_result.final_state
    assert narrow_result.initial_state == wide_result.initial_state
    assert narrow_result.session_results == wide_result.session_results
    assert narrow_result.initial_state_fingerprint == (
        wide_result.initial_state_fingerprint
    )
    assert narrow_result.final_state_fingerprint == (
        wide_result.final_state_fingerprint
    )
    assert tuple(row.indicator_rows for row in narrow_result.session_results) == tuple(
        row.indicator_rows for row in wide_result.session_results
    )


def test_empty_run_preserves_existing_economics_and_required_interval() -> None:
    state = PortfolioState(settled_cash=Decimal("777"))
    result = HistoricalBacktestOrchestrator().run(
        state,
        (),
        decision_interval=DECISION_INTERVAL,
    )

    assert result.decision_interval == DECISION_INTERVAL
    assert result.initial_state == result.final_state == state
    assert result.initial_state_fingerprint == result.final_state_fingerprint
    assert result.session_results == ()


@pytest.mark.parametrize("session", [FRIDAY, MONDAY])
def test_closed_interval_accepts_each_endpoint(session: date) -> None:
    result = HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        (empty_input(session),),
        decision_interval=HistoricalDecisionInterval(
            decision_start_date=FRIDAY,
            decision_end_date=MONDAY,
        ),
    )

    assert result.session_results[0].session == session


def test_decision_interval_value_module_has_no_downstream_or_provider_dependencies() -> None:
    source = inspect.getsource(decision_interval_module)

    assert "provider" not in source.lower()
    assert "norgate" not in source.lower()
    assert "portfolio" not in source.lower()
    assert "research_metrics" not in source.lower()


def test_phase16b_has_no_session_derived_interval_fallback() -> None:
    """Phase 16B consumes or persists the interval; it never derives one.

    Narrowed under Phase 16B.2 v0.2.1 Clause 59, which explicitly authorizes
    the Phase 16B.2 calculation layer to *read* ``decision_interval`` and its
    boundaries -- CAGR's elapsed calendar duration is defined in terms of them.
    The ownership intent is unchanged and is now enforced more precisely than
    the old blanket token ban could: the boundaries may appear only as
    attribute reads on an upstream object, and only in the two authorized
    calculation modules. Phase 16B.3 is separately allowed to name these two
    fields solely to encode/decode the supplied ``HistoricalDecisionInterval``;
    its own scope tests prohibit session/equity evidence and metric calculation.
    Thus no Phase 16B layer can infer an interval from sessions, trades, bars,
    or observations.
    """

    root = Path(__file__).resolve().parents[2]
    production = root / "src" / "stock_swing_d1" / "research_metrics"
    boundaries = {"decision_start_date", "decision_end_date"}
    authorized = {"arithmetic.py", "calculation.py"}
    persistence_adapters = {"persistence.py", "reporting.py"}

    offenders = []
    for path in sorted(production.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if path.name in persistence_adapters:
                continue
            named = None
            if isinstance(node, ast.Attribute):
                # An attribute *read* inside an authorized calculation module
                # is consumption of upstream authority, which Clause 59 allows.
                reads_upstream_authority = path.name in authorized and isinstance(
                    node.ctx, ast.Load
                )
                named = None if reads_upstream_authority else node.attr
            elif isinstance(node, ast.Name):
                named = node.id
            elif isinstance(node, ast.arg):
                named = node.arg
            elif isinstance(node, ast.keyword):
                named = node.arg
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                named = node.name
            elif isinstance(node, ast.Constant) and type(node.value) is str:
                named = node.value
            if named in boundaries:
                offenders.append((path.name, named, node.lineno))

    assert offenders == []

    # And the interval is never reconstructed from a calendar or an observation.
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(production.glob("*.py"))
    ).lower()
    for forbidden in (
        "timedelta",
        "exchange_calendar",
        "trading_calendar",
        "session_calendar",
        "first_session",
        "last_session",
    ):
        assert forbidden not in source


def test_phase15a_scope_contains_no_strategy_or_sizing_economics() -> None:
    source = "\n".join(
        inspect.getsource(member)
        for member in (
            HistoricalBacktestOrchestrator,
            RankedAllocationAdapter,
        )
    ).lower()

    for forbidden in (
        "risk_sized_shares",
        "cash_sized_shares",
        "loss_per_share",
        "rank multiplier",
        "slippage_bps =",
        "commission =",
        "stop_price =",
        "take_profit_price =",
    ):
        assert forbidden not in source
