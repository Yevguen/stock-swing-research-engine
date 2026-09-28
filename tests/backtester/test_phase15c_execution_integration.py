"""Phase 15C production adapters integrated through Phase 15A and Phase 13."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtester import (
    BacktestBuyExecutionEventAdapter,
    BacktestSellExecutionEventAdapter,
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionInput,
    HistoricalBacktestValidationError,
)
from stock_swing_d1.execution.costs import (
    AdministrativeExitPricingService,
    ExecutionCostPolicyRef,
    ExecutionIdentifierService,
    HistoricalUsEquitySettlementResolver,
)
from stock_swing_d1.execution.entry import (
    EntryExecutionService,
    EntryExecutionStatus,
)
from stock_swing_d1.execution.open_position_exit import (
    OpenPositionExitEvaluator,
    OpenPositionExitReason,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.portfolio.portfolio_events import ExecutionSide
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from tests.backtester.conftest import decision_time
from tests.backtester.test_phase15a_chronology import (
    EntryCalendar,
    EntryEarningsOverlay,
)
from tests.backtester.test_phase15b_exit_integration import (
    earnings_decision,
    exit_evaluation,
)
from tests.backtester.test_phase15c_execution_adapters import (
    ExplicitSettlementCalendar,
)
from tests.execution.open_position_exit.conftest import (
    ExplicitTradingCalendar,
    SESSIONS,
)


ALLOCATION_SESSION = date(2026, 7, 31)
ENTRY_SESSION = SESSIONS[0]
EXIT_SESSION = SESSIONS[1]


class RecordingTransition:
    def __init__(self) -> None:
        self.calls: list[tuple[object, date, tuple[object, ...]]] = []

    def transition(self, previous_state, session, events, dividend_evidence=()):
        self.calls.append((previous_state, session, tuple(events)))
        return PortfolioTransitionEngine.transition(
            previous_state, session, events, dividend_evidence=dividend_evidence
        )


class RecordingExitEvaluator:
    def __init__(self) -> None:
        self.delegate = OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        )
        self.calls = []

    def evaluate(self, evaluation):
        self.calls.append(evaluation)
        return self.delegate.evaluate(evaluation)


class DelegatingBuyAdapter:
    def __init__(self, delegate, updates) -> None:
        self.delegate = delegate
        self.updates = updates
        self.calls = 0

    def build_buy_event(self, **kwargs):
        self.calls += 1
        event = self.delegate.build_buy_event(**kwargs)
        return event.model_copy(update=self.updates(event))


class CorruptingEntryService:
    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def execute_pending_entry(self, **kwargs):
        decision = self.delegate.execute_pending_entry(**kwargs)
        object.__setattr__(decision, "execution_cost_quote", None)
        return decision


class SyntheticBuyAdapter:
    def build_buy_event(self, **kwargs):
        raise AssertionError("construction-only synthetic adapter")


class SyntheticSellAdapter:
    def build_sell_event(self, **kwargs):
        raise AssertionError("construction-only synthetic adapter")


def _bar(
    *,
    security_id: str,
    session: date,
    opening_price: float = 100.0,
) -> StockBar:
    return StockBar(
        security_id=security_id,
        symbol=security_id.replace("NORGATE:", "S"),
        trading_date=session,
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


def _entry_service(execution_cost_service):
    return EntryExecutionService(
        earnings_overlay=EntryEarningsOverlay(),
        trading_calendar=EntryCalendar(),
        execution_cost_service=execution_cost_service,
    )


def _protective_factory():
    """Frozen Phase 10 creation capability every executed entry requires
    (Task 5C-B: the entry-session evaluation runs in the shared body)."""

    from stock_swing_d1.execution.protective_exit import ProtectiveExitService

    return ProtectiveExitService(trading_calendar=ExplicitTradingCalendar())


def _buy_adapter(execution_cost_service):
    return BacktestBuyExecutionEventAdapter(
        execution_cost_service=execution_cost_service,
        identifier_service=ExecutionIdentifierService(),
    )


def _sell_adapter(
    execution_cost_service,
    *,
    settlement_sessions=SESSIONS,
):
    return BacktestSellExecutionEventAdapter(
        execution_cost_service=execution_cost_service,
        administrative_pricing_service=AdministrativeExitPricingService(
            policy=execution_cost_service.policy
        ),
        settlement_resolver=HistoricalUsEquitySettlementResolver(
            settlement_calendar=ExplicitSettlementCalendar(
                tuple(settlement_sessions)
            )
        ),
        identifier_service=ExecutionIdentifierService(),
    )


def _intent(
    *,
    asset_id,
    allocation_session,
    entry_session,
    make_candidate_pair,
    make_portfolio_snapshot,
    cash=10_000.0,
):
    ranking, candidate = make_candidate_pair(
        asset_id,
        session=allocation_session,
        following=entry_session,
    )
    allocation = HistoricalBacktestOrchestrator().process_session(
        prior_state=PortfolioState(settled_cash=Decimal(str(cash))),
        session_input=HistoricalBacktestSessionInput(
            session=allocation_session,
            decision_time=decision_time(allocation_session),
            next_session=entry_session,
            ranking_candidates=(ranking,),
            allocation_candidates=(candidate,),
            allocation_portfolio=make_portfolio_snapshot(
                session=allocation_session,
                cash=cash,
                equity=cash,
            ),
        ),
    )
    return allocation.future_entry_intents[0]


def _state_from_production_buy(
    *,
    asset_id,
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
    initial_cash=Decimal("10000"),
):
    intent = _intent(
        asset_id=asset_id,
        allocation_session=ALLOCATION_SESSION,
        entry_session=ENTRY_SESSION,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        cash=float(initial_cash),
    )
    sized = intent.candidate_decision.sized_pending_entry
    assert sized is not None
    decision = _entry_service(execution_cost_service).execute_pending_entry(
        pending_entry=intent.allocation_candidate.pending_entry,
        sized_pending_entry=sized,
        execution_bar=_bar(security_id=asset_id, session=ENTRY_SESSION),
    )
    event = _buy_adapter(execution_cost_service).build_buy_event(
        session=ENTRY_SESSION,
        intent=intent,
        entry_execution=decision,
    )
    state = PortfolioTransitionEngine.transition(
        PortfolioState(settled_cash=initial_cash),
        ENTRY_SESSION,
        (event,),
    ).resulting_state
    return state, event, decision


def test_production_adapter_policy_composition_accepts_same_policy(
    execution_cost_service,
) -> None:
    buy_adapter = _buy_adapter(execution_cost_service)
    sell_adapter = _sell_adapter(execution_cost_service)

    orchestrator = HistoricalBacktestOrchestrator(
        entry_event_adapter=buy_adapter,
        open_position_exit_event_adapter=sell_adapter,
    )

    assert orchestrator is not None


def test_production_adapter_policy_composition_rejects_different_policy(
    execution_cost_service,
) -> None:
    buy_adapter = _buy_adapter(execution_cost_service)
    sell_adapter = _sell_adapter(execution_cost_service)
    object.__setattr__(
        sell_adapter,
        "policy_ref",
        ExecutionCostPolicyRef(
            policy_id=sell_adapter.policy_ref.policy_id,
            policy_fingerprint="e" * 64,
        ),
    )
    recording = RecordingTransition()

    with pytest.raises(HistoricalBacktestValidationError) as captured:
        HistoricalBacktestOrchestrator(
            transition_service=recording,
            entry_event_adapter=buy_adapter,
            open_position_exit_event_adapter=sell_adapter,
        )

    assert captured.value.code == "EXECUTION_COST_POLICY_MISMATCH"
    assert recording.calls == []


def test_production_adapter_policy_composition_rejects_malformed_policy_ref(
    execution_cost_service,
) -> None:
    buy_adapter = _buy_adapter(execution_cost_service)
    sell_adapter = _sell_adapter(execution_cost_service)
    object.__setattr__(sell_adapter, "policy_ref", None)
    recording = RecordingTransition()

    with pytest.raises(HistoricalBacktestValidationError) as captured:
        HistoricalBacktestOrchestrator(
            transition_service=recording,
            entry_event_adapter=buy_adapter,
            open_position_exit_event_adapter=sell_adapter,
        )

    assert captured.value.code == "EXECUTION_COST_POLICY_MISMATCH"
    assert captured.value.__cause__ is not None
    assert recording.calls == []


def test_production_adapter_policy_composition_ignores_synthetic_adapters() -> None:
    orchestrator = HistoricalBacktestOrchestrator(
        entry_event_adapter=SyntheticBuyAdapter(),
        open_position_exit_event_adapter=SyntheticSellAdapter(),
    )

    assert orchestrator is not None


def test_real_buy_reaches_phase13_with_exact_cost_cash_and_cost_basis(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    asset_id = "NORGATE:2001"
    intent = _intent(
        asset_id=asset_id,
        allocation_session=ALLOCATION_SESSION,
        entry_session=ENTRY_SESSION,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
    )
    initial = PortfolioState(settled_cash=Decimal("10000"))
    recording = RecordingTransition()
    result = HistoricalBacktestOrchestrator(
        transition_service=recording,
        entry_execution_service=_entry_service(execution_cost_service),
        entry_event_adapter=_buy_adapter(execution_cost_service),
        open_position_exit_evaluator=RecordingExitEvaluator(),
        protective_exit_state_factory=_protective_factory(),
    ).process_session(
        prior_state=initial,
        session_input=HistoricalBacktestSessionInput(
            session=ENTRY_SESSION,
            decision_time=decision_time(ENTRY_SESSION),
            scheduled_entry_intents=(intent,),
            entry_execution_bars=(
                _bar(security_id=asset_id, session=ENTRY_SESSION),
            ),
        ),
    )

    decision = result.entry_execution_decisions[0]
    event = result.ordered_execution_events[0]
    position = result.authoritative_state.open_positions[0]
    exact_total = Decimal(event.quantity) * event.fill_price + event.execution_cost
    assert len(recording.calls) == 1
    assert event.fill_price == Decimal(str(decision.execution_price))
    assert event.execution_cost == decision.execution_cost_quote.execution_cost
    assert exact_total == (
        decision.execution_cost_quote.notional
        + decision.execution_cost_quote.execution_cost
    )
    assert result.reservation_settlements[0].actual_cash_used == float(exact_total)
    assert position.entry_price == event.fill_price
    assert position.entry_execution_cost == event.execution_cost
    assert position.entry_execution_id == (
        f"ENTRY:{ENTRY_SESSION.isoformat()}:{asset_id}"
    )
    assert position.cost_basis == exact_total
    assert result.authoritative_state.settled_cash == initial.settled_cash - exact_total


def test_cost_driven_phase9_cancellation_releases_reservation_and_creates_no_buy(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    asset_id = "NORGATE:2002"
    intent = _intent(
        asset_id=asset_id,
        allocation_session=ALLOCATION_SESSION,
        entry_session=ENTRY_SESSION,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
    )
    sized = intent.candidate_decision.sized_pending_entry
    assert sized is not None
    reserved = Decimal(str(intent.candidate_decision.reserved_cash))
    candidate_fill = (reserved - Decimal("0.50")) / Decimal(
        sized.fixed_shares
    )
    reference_open = float(candidate_fill / Decimal("1.0005"))
    initial = PortfolioState(settled_cash=Decimal("10000"))
    recording = RecordingTransition()
    result = HistoricalBacktestOrchestrator(
        transition_service=recording,
        entry_execution_service=_entry_service(execution_cost_service),
        entry_event_adapter=_buy_adapter(execution_cost_service),
    ).process_session(
        prior_state=initial,
        session_input=HistoricalBacktestSessionInput(
            session=ENTRY_SESSION,
            decision_time=decision_time(ENTRY_SESSION),
            scheduled_entry_intents=(intent,),
            entry_execution_bars=(
                _bar(
                    security_id=asset_id,
                    session=ENTRY_SESSION,
                    opening_price=reference_open,
                ),
            ),
        ),
    )

    decision = result.entry_execution_decisions[0]
    settlement = result.reservation_settlements[0]
    assert decision.status is (
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
    )
    assert decision.candidate_execution_cost_quote.notional <= reserved
    assert decision.candidate_cash_required > float(reserved)
    assert decision.requested_shares == sized.fixed_shares
    assert result.ordered_execution_events == ()
    assert settlement.actual_cash_used == 0
    assert settlement.released_cash == float(reserved)
    assert result.authoritative_state.open_positions == ()
    assert result.authoritative_state.settled_cash == initial.settled_cash
    assert len(recording.calls) == 1


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
@pytest.mark.parametrize(
    "updates",
    (
        lambda event: {"fill_price": event.fill_price + Decimal("1")},
        lambda event: {"execution_cost": event.execution_cost + Decimal("1")},
        lambda event: {"side": ExecutionSide.SELL},
        lambda event: {"session": SESSIONS[2]},
        lambda event: {"asset_id": "NORGATE:9999"},
        lambda event: {"quantity": event.quantity + 1},
        lambda event: {
            "settlement_id": "INVALID-BUY-SETTLEMENT",
            "settlement_session": SESSIONS[2],
        },
        lambda event: {"fill_price": float(event.fill_price)},
    ),
)
def test_phase15a_rejects_each_malformed_buy_before_phase13(
    updates,
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    intent = _intent(
        asset_id="NORGATE:2003",
        allocation_session=ALLOCATION_SESSION,
        entry_session=ENTRY_SESSION,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
    )
    recording = RecordingTransition()
    bad_adapter = DelegatingBuyAdapter(
        _buy_adapter(execution_cost_service),
        updates,
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="INVALID_ENTRY_EVENT_ADAPTER_RESULT",
    ):
        HistoricalBacktestOrchestrator(
            transition_service=recording,
            entry_execution_service=_entry_service(execution_cost_service),
            entry_event_adapter=bad_adapter,
        ).process_session(
            prior_state=PortfolioState(settled_cash=Decimal("10000")),
            session_input=HistoricalBacktestSessionInput(
                session=ENTRY_SESSION,
                decision_time=decision_time(ENTRY_SESSION),
                scheduled_entry_intents=(intent,),
                entry_execution_bars=(
                    _bar(security_id="NORGATE:2003", session=ENTRY_SESSION),
                ),
            ),
        )
    assert recording.calls == []


def test_phase15a_reconstructs_phase9_result_before_buy_adapter_and_phase13(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    intent = _intent(
        asset_id="NORGATE:2004",
        allocation_session=ALLOCATION_SESSION,
        entry_session=ENTRY_SESSION,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
    )
    recording = RecordingTransition()
    adapter = DelegatingBuyAdapter(
        _buy_adapter(execution_cost_service), lambda event: {}
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="INVALID_ENTRY_EXECUTION_RESULT",
    ):
        HistoricalBacktestOrchestrator(
            transition_service=recording,
            entry_execution_service=CorruptingEntryService(
                _entry_service(execution_cost_service)
            ),
            entry_event_adapter=adapter,
        ).process_session(
            prior_state=PortfolioState(settled_cash=Decimal("10000")),
            session_input=HistoricalBacktestSessionInput(
                session=ENTRY_SESSION,
                decision_time=decision_time(ENTRY_SESSION),
                scheduled_entry_intents=(intent,),
                entry_execution_bars=(
                    _bar(security_id="NORGATE:2004", session=ENTRY_SESSION),
                ),
            ),
        )
    assert adapter.calls == 0
    assert recording.calls == []


def test_phase15a_wraps_production_buy_adapter_failure_with_cause(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    asset_id = "NORGATE:2005"
    intent = _intent(
        asset_id=asset_id,
        allocation_session=ALLOCATION_SESSION,
        entry_session=ENTRY_SESSION,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
    )
    adapter = _buy_adapter(execution_cost_service)
    object.__setattr__(
        execution_cost_service,
        "policy_ref",
        ExecutionCostPolicyRef(
            policy_id=adapter.policy_ref.policy_id,
            policy_fingerprint="d" * 64,
        ),
    )
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ENTRY_EVENT_ADAPTER_FAILED",
    ) as captured:
        HistoricalBacktestOrchestrator(
            transition_service=recording,
            entry_execution_service=_entry_service(execution_cost_service),
            entry_event_adapter=adapter,
        ).process_session(
            prior_state=PortfolioState(settled_cash=Decimal("10000")),
            session_input=HistoricalBacktestSessionInput(
                session=ENTRY_SESSION,
                decision_time=decision_time(ENTRY_SESSION),
                scheduled_entry_intents=(intent,),
                entry_execution_bars=(
                    _bar(security_id=asset_id, session=ENTRY_SESSION),
                ),
            ),
        )
    assert captured.value.__cause__ is not None
    assert recording.calls == []


@pytest.mark.parametrize(
    "reason", ("STOP", "GAP_STOP", "EARNINGS", "MAX_HOLDING")
)
def test_real_sell_creates_exact_pending_net_proceeds(
    reason,
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    asset_id = "NORGATE:3001"
    prior, _, _ = _state_from_production_buy(
        asset_id=asset_id,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        execution_cost_service=execution_cost_service,
    )
    position = prior.open_positions[0]
    if reason == "STOP":
        current_index = 1
        evaluation = exit_evaluation(
            asset_id,
            session_index=current_index,
            entry_price=float(position.entry_price),
            low=95.0,
        )
        expected_reason = OpenPositionExitReason.STOP_LOSS
    elif reason == "GAP_STOP":
        current_index = 1
        evaluation = exit_evaluation(
            asset_id,
            session_index=current_index,
            entry_price=float(position.entry_price),
            open=95.0,
            high=100.0,
            low=94.0,
            close=97.0,
        )
        expected_reason = OpenPositionExitReason.GAP_THROUGH_STOP
    elif reason == "EARNINGS":
        current_index = 5
        evaluation = exit_evaluation(
            asset_id,
            session_index=current_index,
            entry_price=float(position.entry_price),
            close=104.0,
            prior_boundary_earnings_decision=earnings_decision(
                asset_id=asset_id,
                boundary_index=current_index - 1,
                scheduled_index=current_index + 1,
            ),
        )
        expected_reason = OpenPositionExitReason.EARNINGS_FORCED_EXIT
    else:
        current_index = 9
        evaluation = exit_evaluation(
            asset_id,
            session_index=current_index,
            entry_price=float(position.entry_price),
            close=103.0,
        )
        expected_reason = OpenPositionExitReason.MAX_HOLDING
    session = SESSIONS[current_index]
    recording = RecordingTransition()
    result = HistoricalBacktestOrchestrator(
        transition_service=recording,
        open_position_exit_evaluator=OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        ),
        open_position_exit_event_adapter=_sell_adapter(execution_cost_service),
    ).process_session(
        prior_state=prior,
        session_input=HistoricalBacktestSessionInput(
            session=session,
            decision_time=decision_time(session),
            open_position_exit_evaluations=(evaluation,),
        ),
    )

    decision = result.open_position_exit_decisions[0]
    event = result.ordered_execution_events[0]
    pending = result.authoritative_state.pending_settlements[0]
    expected_pending = Decimal(event.quantity) * event.fill_price - event.execution_cost
    assert decision.selected_reason is expected_reason
    assert result.authoritative_state.open_positions == ()
    assert result.authoritative_state.settled_cash == prior.settled_cash
    assert pending.amount == expected_pending
    assert pending.settlement_id == event.settlement_id
    assert pending.settlement_session == event.settlement_session
    assert len(recording.calls) == 1


def test_earnings_and_max_holding_same_close_create_one_sell_and_one_cost(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    asset_id = "NORGATE:3003"
    prior, _, _ = _state_from_production_buy(
        asset_id=asset_id,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        execution_cost_service=execution_cost_service,
    )
    position = prior.open_positions[0]
    current_index = 9
    evaluation = exit_evaluation(
        asset_id,
        session_index=current_index,
        entry_price=float(position.entry_price),
        close=104.0,
        prior_boundary_earnings_decision=earnings_decision(
            asset_id=asset_id,
            boundary_index=current_index - 1,
            scheduled_index=current_index + 1,
        ),
    )
    session = SESSIONS[current_index]
    result = HistoricalBacktestOrchestrator(
        open_position_exit_evaluator=OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        ),
        open_position_exit_event_adapter=_sell_adapter(execution_cost_service),
    ).process_session(
        prior_state=prior,
        session_input=HistoricalBacktestSessionInput(
            session=session,
            decision_time=decision_time(session),
            open_position_exit_evaluations=(evaluation,),
        ),
    )

    decision = result.open_position_exit_decisions[0]
    assert decision.triggered_reasons == (
        OpenPositionExitReason.EARNINGS_FORCED_EXIT,
        OpenPositionExitReason.MAX_HOLDING,
    )
    assert decision.selected_reason is OpenPositionExitReason.EARNINGS_FORCED_EXIT
    assert len(result.ordered_execution_events) == 1
    assert len(result.authoritative_state.pending_settlements) == 1


def test_sell_proceeds_remain_pending_until_phase13_settlement_session(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    asset_id = "NORGATE:3002"
    prior, _, _ = _state_from_production_buy(
        asset_id=asset_id,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        execution_cost_service=execution_cost_service,
    )
    position = prior.open_positions[0]
    settlement_session = SESSIONS[3]
    trade_result = HistoricalBacktestOrchestrator(
        open_position_exit_evaluator=OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        ),
        open_position_exit_event_adapter=_sell_adapter(
            execution_cost_service,
            settlement_sessions=(EXIT_SESSION, settlement_session),
        ),
    ).process_session(
        prior_state=prior,
        session_input=HistoricalBacktestSessionInput(
            session=EXIT_SESSION,
            decision_time=decision_time(EXIT_SESSION),
            open_position_exit_evaluations=(
                exit_evaluation(
                    asset_id,
                    entry_price=float(position.entry_price),
                    low=95.0,
                ),
            ),
        ),
    )
    pending_amount = trade_result.authoritative_state.pending_settlements[0].amount
    trade_cash = trade_result.authoritative_state.settled_cash

    intermediate = HistoricalBacktestOrchestrator().process_session(
        prior_state=trade_result.authoritative_state,
        session_input=HistoricalBacktestSessionInput(
            session=SESSIONS[2],
            decision_time=decision_time(SESSIONS[2]),
        ),
    )
    settled = HistoricalBacktestOrchestrator().process_session(
        prior_state=intermediate.authoritative_state,
        session_input=HistoricalBacktestSessionInput(
            session=settlement_session,
            decision_time=decision_time(settlement_session),
        ),
    )

    assert intermediate.authoritative_state.settled_cash == trade_cash
    assert len(intermediate.authoritative_state.pending_settlements) == 1
    assert settled.authoritative_state.pending_settlements == ()
    assert settled.authoritative_state.settled_cash == trade_cash + pending_amount


def test_same_t_production_sell_precedes_buy_without_funding_it(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    old_asset = "NORGATE:4001"
    new_asset = "NORGATE:4002"
    prior, _, _ = _state_from_production_buy(
        asset_id=old_asset,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        execution_cost_service=execution_cost_service,
    )
    position = prior.open_positions[0]
    intent = _intent(
        asset_id=new_asset,
        allocation_session=ENTRY_SESSION,
        entry_session=EXIT_SESSION,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        cash=float(prior.settled_cash),
    )
    recording = RecordingTransition()
    exit_evaluator = RecordingExitEvaluator()
    result = HistoricalBacktestOrchestrator(
        transition_service=recording,
        entry_execution_service=_entry_service(execution_cost_service),
        entry_event_adapter=_buy_adapter(execution_cost_service),
        open_position_exit_evaluator=exit_evaluator,
        open_position_exit_event_adapter=_sell_adapter(execution_cost_service),
        protective_exit_state_factory=_protective_factory(),
    ).process_session(
        prior_state=prior,
        session_input=HistoricalBacktestSessionInput(
            session=EXIT_SESSION,
            decision_time=decision_time(EXIT_SESSION),
            scheduled_entry_intents=(intent,),
            entry_execution_bars=(
                _bar(security_id=new_asset, session=EXIT_SESSION),
            ),
            open_position_exit_evaluations=(
                exit_evaluation(
                    old_asset,
                    entry_price=float(position.entry_price),
                    low=95.0,
                ),
            ),
        ),
    )

    sell, buy = result.ordered_execution_events
    buy_cash = Decimal(buy.quantity) * buy.fill_price + buy.execution_cost
    assert tuple(event.side for event in recording.calls[0][2]) == (
        ExecutionSide.SELL,
        ExecutionSide.BUY,
    )
    assert result.authoritative_state.settled_cash == prior.settled_cash - buy_cash
    assert len(result.authoritative_state.pending_settlements) == 1
    assert tuple(
        item.asset_id for item in result.authoritative_state.open_positions
    ) == (new_asset,)
    assert len(recording.calls) == 1
    # Prior-position exit first, then (Task 5C-B) the new entry's own
    # entry-session evaluation -- both through the one Phase 15B owner.
    assert len(exit_evaluator.calls) == 2
    assert exit_evaluator.calls[0].protective_state.security_id == old_asset
    assert exit_evaluator.calls[1].protective_state.security_id == new_asset
    assert exit_evaluator.calls[1].protective_state.entry_session == EXIT_SESSION
    assert result.entry_session_protective_decisions[0].exit_required is False


def test_generated_event_failure_is_atomic_across_sell_and_buy(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    old_asset = "NORGATE:5001"
    new_asset = "NORGATE:5002"
    prior, _, _ = _state_from_production_buy(
        asset_id=old_asset,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        execution_cost_service=execution_cost_service,
    )
    position = prior.open_positions[0]
    intent = _intent(
        asset_id=new_asset,
        allocation_session=ENTRY_SESSION,
        entry_session=EXIT_SESSION,
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        cash=float(prior.settled_cash),
    )
    recording = RecordingTransition()
    bad_buy_adapter = DelegatingBuyAdapter(
        _buy_adapter(execution_cost_service),
        lambda event: {"execution_cost": event.execution_cost + Decimal("1")},
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="INVALID_ENTRY_EVENT_ADAPTER_RESULT",
    ):
        HistoricalBacktestOrchestrator(
            transition_service=recording,
            entry_execution_service=_entry_service(execution_cost_service),
            entry_event_adapter=bad_buy_adapter,
            open_position_exit_evaluator=OpenPositionExitEvaluator(
                trading_calendar=ExplicitTradingCalendar()
            ),
            open_position_exit_event_adapter=_sell_adapter(
                execution_cost_service
            ),
        ).process_session(
            prior_state=prior,
            session_input=HistoricalBacktestSessionInput(
                session=EXIT_SESSION,
                decision_time=decision_time(EXIT_SESSION),
                scheduled_entry_intents=(intent,),
                entry_execution_bars=(
                    _bar(security_id=new_asset, session=EXIT_SESSION),
                ),
                open_position_exit_evaluations=(
                    exit_evaluation(
                        old_asset,
                        entry_price=float(position.entry_price),
                        low=95.0,
                    ),
                ),
            ),
        )
    assert recording.calls == []
    assert prior.open_positions == (position,)
    assert prior.pending_settlements == ()
