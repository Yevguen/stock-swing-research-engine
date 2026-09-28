"""Phase 15B decisions integrated into the single Phase 15A transition."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from stock_swing_d1.backtester import (
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionInput,
    HistoricalBacktestValidationError,
)
from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.earnings.models import (
    EarningsScheduleStatus,
    EarningsStateAsOf,
    LifecycleState,
    TimingClass,
)
from stock_swing_d1.earnings.risk import evaluate_open_position_earnings_risk
from stock_swing_d1.execution.entry import EntryExecutionService
from stock_swing_d1.execution.open_position_exit import (
    EarningsExitStatus,
    ExitPrerequisiteStatus,
    IntrabarAmbiguityStatus,
    OpenPositionExitEvaluationInput,
    OpenPositionExitEvaluator,
    OpenPositionExitReason,
)
from stock_swing_d1.execution.protective_exit import (
    ProtectiveExitService,
    ProtectiveExitState,
)
from stock_swing_d1.models import CorporateActionEvent, DividendEvent, StockBar
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.backtester.conftest import decision_time
from tests.backtester.test_phase15a_chronology import (
    EntryCalendar,
    EntryEarningsOverlay,
    EntryEventAdapter,
)
from tests.execution.open_position_exit.conftest import (
    ExplicitTradingCalendar,
    NEW_YORK,
    SESSIONS,
)


ENTRY_SESSION = SESSIONS[0]
EXIT_SESSION = SESSIONS[1]
SETTLEMENT_SESSION = SESSIONS[2]


def session_input(
    session: date = EXIT_SESSION,
    **overrides,
) -> HistoricalBacktestSessionInput:
    values = {
        "session": session,
        "decision_time": decision_time(session),
    }
    values.update(overrides)
    return HistoricalBacktestSessionInput(**values)


def buy_event(
    *,
    session: date,
    asset_id: str,
    quantity: int = 2,
    execution_id: str | None = None,
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id or f"ENTRY-{asset_id}",
        source_order_id=f"ENTRY-ORDER-{asset_id}",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )


def scheduled_sell(
    *,
    asset_id: str = "NORGATE:1001",
    quantity: int = 2,
    execution_id: str | None = None,
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id or f"SCHEDULED-EXIT-{asset_id}",
        source_order_id=f"SCHEDULED-ORDER-{asset_id}",
        session=EXIT_SESSION,
        asset_id=asset_id,
        side=ExecutionSide.SELL,
        quantity=quantity,
        fill_price=Decimal("101"),
        execution_cost=Decimal("2"),
        settlement_id=f"SCHEDULED-SETTLEMENT-{asset_id}",
        settlement_session=SETTLEMENT_SESSION,
    )


def state_with_positions(
    *asset_ids: str,
    quantity: int = 2,
    settled_cash: Decimal = Decimal("10000"),
    entry_session: date = ENTRY_SESSION,
) -> PortfolioState:
    entry_cost = Decimal(quantity) * Decimal("100") + Decimal("1")
    initial = PortfolioState(
        settled_cash=settled_cash + entry_cost * len(asset_ids)
    )
    events = tuple(
        buy_event(
            session=entry_session,
            asset_id=asset_id,
            quantity=quantity,
        )
        for asset_id in asset_ids
    )
    return PortfolioTransitionEngine.transition(
        initial,
        entry_session,
        events,
    ).resulting_state


def exit_evaluation(
    asset_id: str = "NORGATE:1001",
    *,
    session_index: int = 1,
    entry_session: date = ENTRY_SESSION,
    entry_price: float = 100.0,
    open: float = 100.0,
    high: float = 105.0,
    low: float = 97.0,
    close: float = 101.0,
    earnings_decision: EarningsIntegrationDecision | None = None,
    prior_boundary_earnings_decision: EarningsIntegrationDecision | None = None,
    corporate_actions: tuple[object, ...] = (),
) -> OpenPositionExitEvaluationInput:
    session = SESSIONS[session_index]
    state = ProtectiveExitState._validated(
        security_id=asset_id,
        symbol=asset_id.replace("NORGATE:", "S"),
        signal_session=date(2026, 7, 31),
        signal_time=datetime(2026, 7, 31, 16, tzinfo=NEW_YORK),
        entry_session=entry_session,
        entry_price=entry_price,
        signal_atr_fraction=0.02,
        risk_fraction=0.04,
        stop_price=96.0,
        take_profit_price=108.0,
        last_evaluated_session=(
            None
            if session == entry_session
            else ExplicitTradingCalendar().previous_session(session)
        ),
    )
    bar = StockBar(
        security_id=asset_id,
        symbol=state.symbol,
        trading_date=session,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=open,
        high=high,
        low=low,
        close=close,
        volume=1_000_000,
    )
    return OpenPositionExitEvaluationInput(
        session=session,
        protective_state=state,
        market_bar=bar,
        corporate_actions=corporate_actions,
        earnings_decision=earnings_decision,
        prior_boundary_earnings_decision=prior_boundary_earnings_decision,
    )


class RecordingTransition:
    def __init__(self) -> None:
        self.calls: list[tuple[object, date, tuple[object, ...]]] = []

    def transition(self, previous_state, session, events, dividend_evidence=()):
        self.calls.append((previous_state, session, tuple(events)))
        return PortfolioTransitionEngine.transition(
            previous_state, session, events, dividend_evidence=dividend_evidence
        )


class RecordingExitEvaluator:
    def __init__(self, delegate=None) -> None:
        self.delegate = delegate or OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        )
        self.calls: list[OpenPositionExitEvaluationInput] = []

    def evaluate(self, evaluation):
        self.calls.append(evaluation)
        return self.delegate.evaluate(evaluation)


class ExitEventAdapter:
    """Test-only authoritative execution facts; not production economics."""

    def __init__(
        self,
        *,
        settlement_session: date = SETTLEMENT_SESSION,
        administrative_fill: Decimal | None = None,
        side: ExecutionSide = ExecutionSide.SELL,
        session: date | None = None,
        asset_id: str | None = None,
        quantity_delta: int = 0,
        price_delta: Decimal = Decimal("0"),
        execution_id: str | None = None,
    ) -> None:
        self.settlement_session = settlement_session
        self.administrative_fill = administrative_fill
        self.side = side
        self.session = session
        self.asset_id = asset_id
        self.quantity_delta = quantity_delta
        self.price_delta = price_delta
        self.execution_id = execution_id
        self.calls: list[tuple[object, object]] = []

    def build_sell_event(self, *, decision, open_position):
        self.calls.append((decision, open_position))
        base_fill = (
            Decimal(str(decision.final_execution_price))
            if decision.final_execution_price is not None
            else self.administrative_fill
        )
        if base_fill is None:
            raise AssertionError("test adapter needs an explicit administrative fill")
        event_session = self.session or decision.session
        event_asset_id = self.asset_id or decision.security_id
        if self.side is ExecutionSide.BUY:
            return PortfolioExecutionEvent(
                execution_id=self.execution_id or f"EXIT-{event_asset_id}",
                source_order_id=f"EXIT-ORDER-{event_asset_id}",
                session=event_session,
                asset_id=event_asset_id,
                side=self.side,
                quantity=open_position.quantity + self.quantity_delta,
                fill_price=base_fill + self.price_delta,
                execution_cost=Decimal("2.75"),
            )
        return PortfolioExecutionEvent(
            execution_id=self.execution_id or f"EXIT-{event_asset_id}",
            source_order_id=f"EXIT-ORDER-{event_asset_id}",
            session=event_session,
            asset_id=event_asset_id,
            side=self.side,
            quantity=open_position.quantity + self.quantity_delta,
            fill_price=base_fill + self.price_delta,
            execution_cost=Decimal("2.75"),
            settlement_id=f"EXIT-SETTLEMENT-{event_asset_id}",
            settlement_session=self.settlement_session,
        )


def orchestrator(
    *,
    transition=None,
    evaluator=None,
    adapter=None,
    **dependencies,
) -> HistoricalBacktestOrchestrator:
    return HistoricalBacktestOrchestrator(
        transition_service=transition,
        open_position_exit_evaluator=evaluator,
        open_position_exit_event_adapter=adapter,
        **dependencies,
    )


def process(
    prior_state: PortfolioState,
    evaluations: tuple[OpenPositionExitEvaluationInput, ...] = (),
    *,
    transition=None,
    evaluator=None,
    adapter=None,
    **input_overrides,
):
    return orchestrator(
        transition=transition,
        evaluator=evaluator,
        adapter=adapter,
    ).process_session(
        prior_state=prior_state,
        session_input=session_input(
            open_position_exit_evaluations=evaluations,
            **input_overrides,
        ),
    )


def earnings_decision(
    *,
    asset_id: str,
    boundary_index: int,
    scheduled_index: int | None,
    timing_class: TimingClass = TimingClass.BMO,
    active: bool = True,
) -> EarningsIntegrationDecision:
    calendar = ExplicitTradingCalendar()
    boundary = SESSIONS[boundary_index]
    state = EarningsStateAsOf(
        as_of=calendar.decision_time(boundary),
        canonical_asset_id=asset_id,
        event_instance_id=f"earnings-{asset_id}",
        earnings_schedule_known=True,
        lifecycle_state=(
            LifecycleState.CONFIRMED if active else LifecycleState.CANCELLED
        ),
        scheduled_date=(SESSIONS[scheduled_index] if active else None),
        timing_class=timing_class,
        scheduled_at=None,
        knowledge_effective_at=calendar.decision_time(ENTRY_SESSION),
        schedule_status=(
            EarningsScheduleStatus.KNOWN_EVENT
            if active
            else EarningsScheduleStatus.NO_EVENT_WITHIN_PROVIDER_HORIZON
        ),
    )
    risk = evaluate_open_position_earnings_risk(
        state,
        boundary,
        calendar,
        position_entry_session=ENTRY_SESSION,
    )
    deadline = risk.last_safe_exit_session
    if deadline is None or boundary < deadline:
        action = EarningsIntegrationAction.HOLD_POSITION
    elif boundary == deadline:
        action = EarningsIntegrationAction.EXIT_REQUIRED_THIS_SESSION
    elif risk.unavoidable_earnings_exposure:
        action = EarningsIntegrationAction.UNAVOIDABLE_EARNINGS_EXPOSURE
    else:
        action = EarningsIntegrationAction.MISSED_EXIT_DEADLINE
    return EarningsIntegrationDecision(action, state, risk, risk.risk_reason)


def test_no_prior_positions_needs_no_phase15b_dependency() -> None:
    result = process(PortfolioState(settled_cash=Decimal("1000")))

    assert result.open_position_exit_decisions == ()
    assert result.ordered_execution_events == ()


def test_hold_is_audited_without_a_sell_or_direct_state_mutation() -> None:
    prior = state_with_positions("NORGATE:1001")
    evaluator = RecordingExitEvaluator()
    result = process(prior, (exit_evaluation(),), evaluator=evaluator)

    assert result.open_position_exit_decisions[0].exit_required is False
    assert result.ordered_execution_events == ()
    assert result.authoritative_state.open_positions == prior.open_positions
    assert result.authoritative_state is not prior


@pytest.mark.parametrize(
    ("bar_values", "reason"),
    [
        ({"low": 95.0, "close": 97.0}, OpenPositionExitReason.STOP_LOSS),
        ({"high": 109.0, "close": 107.0}, OpenPositionExitReason.TAKE_PROFIT),
    ],
)
def test_ordinary_protective_exit_produces_one_sell(bar_values, reason) -> None:
    prior = state_with_positions("NORGATE:1001")
    evaluation = exit_evaluation(**bar_values)
    result = process(
        prior,
        (evaluation,),
        evaluator=RecordingExitEvaluator(),
        adapter=ExitEventAdapter(),
    )

    assert result.open_position_exit_decisions[0].selected_reason is reason
    assert len(result.ordered_execution_events) == 1
    assert result.ordered_execution_events[0].side is ExecutionSide.SELL
    assert result.authoritative_state.open_positions == ()


@pytest.mark.parametrize(
    ("bar_values", "reason"),
    [
        (
            {"open": 91.0, "high": 100.0, "low": 90.0, "close": 99.0},
            OpenPositionExitReason.GAP_THROUGH_STOP,
        ),
        (
            {"open": 112.0, "high": 114.0, "low": 100.0, "close": 111.0},
            OpenPositionExitReason.GAP_THROUGH_TARGET,
        ),
    ],
)
def test_gap_exit_price_reaches_event_unchanged(bar_values, reason) -> None:
    prior = state_with_positions("NORGATE:1001")
    adapter = ExitEventAdapter()
    result = process(
        prior,
        (exit_evaluation(**bar_values),),
        evaluator=RecordingExitEvaluator(),
        adapter=adapter,
    )
    decision = result.open_position_exit_decisions[0]

    assert decision.selected_reason is reason
    assert result.ordered_execution_events[0].fill_price == Decimal(
        str(decision.final_execution_price)
    )
    assert adapter.calls[0][0] is decision


def test_ambiguous_bar_generates_only_the_canonical_stop_sell() -> None:
    prior = state_with_positions("NORGATE:1001")
    result = process(
        prior,
        (exit_evaluation(high=110.0, low=95.0, close=109.0),),
        evaluator=RecordingExitEvaluator(),
        adapter=ExitEventAdapter(),
    )
    decision = result.open_position_exit_decisions[0]

    assert decision.triggered_reasons == (
        OpenPositionExitReason.STOP_LOSS,
        OpenPositionExitReason.TAKE_PROFIT,
    )
    assert decision.intrabar_ambiguity_status is (
        IntrabarAmbiguityStatus.STOP_AND_TARGET_TOUCHED
    )
    assert len(result.ordered_execution_events) == 1


def test_mixed_positions_are_canonical_and_input_permutation_independent() -> None:
    ids = ("NORGATE:1001", "NORGATE:1002", "NORGATE:1003")
    prior = state_with_positions(*ids)
    evaluations = (
        exit_evaluation(ids[2], high=109.0, close=107.0),
        exit_evaluation(ids[0]),
        exit_evaluation(ids[1], low=95.0, close=97.0),
    )
    first = process(
        prior,
        evaluations,
        evaluator=RecordingExitEvaluator(),
        adapter=ExitEventAdapter(),
    )
    second = process(
        prior,
        tuple(reversed(evaluations)),
        evaluator=RecordingExitEvaluator(),
        adapter=ExitEventAdapter(),
    )

    assert tuple(
        item.security_id for item in first.open_position_exit_decisions
    ) == ids
    assert tuple(event.asset_id for event in first.ordered_execution_events) == ids[1:]
    assert first.open_position_exit_decisions == second.open_position_exit_decisions
    assert first.ordered_execution_events == second.ordered_execution_events
    assert first.authoritative_state == second.authoritative_state
    assert tuple(
        item.asset_id for item in first.authoritative_state.open_positions
    ) == (ids[0],)


@pytest.mark.parametrize(
    ("evaluations", "code"),
    [
        ((), "MISSING_OPEN_POSITION_EXIT_EVALUATION"),
        (
            (exit_evaluation("NORGATE:9999"),),
            "EXTRA_OPEN_POSITION_EXIT_EVALUATION",
        ),
        (
            (exit_evaluation(), exit_evaluation()),
            "DUPLICATE_OPEN_POSITION_EXIT_EVALUATION",
        ),
    ],
)
def test_incomplete_or_duplicate_coverage_fails_before_phase13(
    evaluations, code
) -> None:
    recording = RecordingTransition()

    with pytest.raises(HistoricalBacktestValidationError, match=code):
        process(
            state_with_positions("NORGATE:1001"),
            evaluations,
            transition=recording,
            evaluator=RecordingExitEvaluator(),
        )

    assert recording.calls == []


def test_scheduled_sell_and_phase15b_evaluation_conflict_before_phase13() -> None:
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="DUPLICATE_OPEN_POSITION_EXIT_TREATMENT",
    ):
        process(
            state_with_positions("NORGATE:1001"),
            (exit_evaluation(),),
            transition=recording,
            evaluator=RecordingExitEvaluator(),
            scheduled_execution_events=(scheduled_sell(),),
        )

    assert recording.calls == []


@pytest.mark.parametrize(
    "evaluation",
    [
        exit_evaluation(entry_session=date(2026, 7, 31)),
        exit_evaluation(entry_price=99.0),
    ],
)
def test_phase15b_and_phase13_position_identity_must_match(evaluation) -> None:
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="OPEN_POSITION_EXIT_IDENTITY_MISMATCH",
    ):
        process(
            state_with_positions("NORGATE:1001"),
            (evaluation,),
            transition=recording,
            evaluator=RecordingExitEvaluator(),
        )

    assert recording.calls == []


def test_phase15b_evaluation_session_must_equal_t() -> None:
    recording = RecordingTransition()
    wrong_session = exit_evaluation(session_index=2)

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="OPEN_POSITION_EXIT_SESSION_MISMATCH",
    ):
        process(
            state_with_positions("NORGATE:1001"),
            (wrong_session,),
            transition=recording,
            evaluator=RecordingExitEvaluator(),
        )

    assert recording.calls == []


@pytest.mark.parametrize(
    ("event_type", "new_shares", "old_shares"),
    [
        ("split", 2.0, 1.0),
        ("reverse_split", 1.0, 2.0),
    ],
)
def test_nested_capital_action_cannot_bypass_portfolio_prerequisite(
    event_type,
    new_shares,
    old_shares,
) -> None:
    prior = state_with_positions("NORGATE:1001")
    before = prior.model_dump(mode="python")
    split = CorporateActionEvent(
        security_id="NORGATE:1001",
        symbol="S1001",
        event_date=EXIT_SESSION,
        date_semantics="effective_date",
        event_type=event_type,
        terms_verified=True,
        new_shares=new_shares,
        old_shares=old_shares,
        source_provider="Norgate Data",
        source_asset_id=1001,
    )
    evaluator = RecordingExitEvaluator()
    recording = RecordingTransition()

    with pytest.raises(HistoricalBacktestValidationError) as raised:
        process(
            prior,
            (exit_evaluation(corporate_actions=(split,)),),
            transition=recording,
            evaluator=evaluator,
        )

    assert raised.value.code == "UNSUPPORTED_CORPORATE_ACTION_TRANSITION"
    assert evaluator.calls == []
    assert recording.calls == []
    assert prior.model_dump(mode="python") == before


def test_nested_noncapital_dividend_audit_is_not_overblocked() -> None:
    dividend = DividendEvent(
        security_id="NORGATE:1001",
        symbol="S1001",
        entitlement_date=ENTRY_SESSION,
        date_semantics="entitlement_close",
        dividend_type="ordinary_cash",
        amount_per_share=0.25,
        currency="USD",
        source_provider="Norgate Data",
        source_asset_id=1001,
        source_adjustment_mode="CAPITALSPECIAL",
    )
    evaluator = RecordingExitEvaluator()
    result = process(
        state_with_positions("NORGATE:1001"),
        (exit_evaluation(corporate_actions=(dividend,)),),
        evaluator=evaluator,
    )

    assert len(evaluator.calls) == 1
    decision = result.open_position_exit_decisions[0]
    assert decision.protective_exit_decision.corporate_actions == (dividend,)
    assert result.ordered_execution_events == ()


def test_partial_and_duplicate_scheduled_sells_fail_before_phase13() -> None:
    prior = state_with_positions("NORGATE:1001")
    for events, code in (
        ((scheduled_sell(quantity=1),), "PARTIAL_SCHEDULED_SELL_UNSUPPORTED"),
        (
            (
                scheduled_sell(execution_id="SELL-A"),
                scheduled_sell(execution_id="SELL-B"),
            ),
            "DUPLICATE_OPEN_POSITION_EXIT_TREATMENT",
        ),
    ):
        recording = RecordingTransition()
        with pytest.raises(HistoricalBacktestValidationError, match=code):
            process(
                prior,
                transition=recording,
                scheduled_execution_events=events,
            )
        assert recording.calls == []


def test_full_scheduled_sell_is_complete_coverage_without_phase15b() -> None:
    result = process(
        state_with_positions("NORGATE:1001"),
        scheduled_execution_events=(scheduled_sell(),),
    )

    assert result.open_position_exit_decisions == ()
    assert result.authoritative_state.open_positions == ()


def test_missing_evaluator_fails_closed_when_evaluation_is_required() -> None:
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="MISSING_OPEN_POSITION_EXIT_EVALUATOR",
    ):
        process(
            state_with_positions("NORGATE:1001"),
            (exit_evaluation(),),
            transition=recording,
        )

    assert recording.calls == []


def test_same_session_terminal_decision_is_persistence_ready_after_5cc() -> None:
    """Task 5C-C retired the Phase 15B same-session persistence limitation:
    a terminal decision on the entry session itself is READY and reaches
    the Phase 15C adapter instead of failing a prerequisite guard."""

    adapter = ExitEventAdapter()
    evaluator = RecordingExitEvaluator()
    integration = orchestrator(evaluator=evaluator, adapter=adapter)
    prior = state_with_positions(
        "NORGATE:1001",
        entry_session=EXIT_SESSION,
    )
    evaluation = exit_evaluation(
        entry_session=EXIT_SESSION,
        low=95.0,
    )
    decisions, events = integration._evaluate_open_position_exits(
        prior_state=prior,
        session_input=session_input(
            open_position_exit_evaluations=(evaluation,),
        ),
    )

    assert len(evaluator.calls) == 1
    assert len(adapter.calls) == 1
    (decision,) = decisions
    assert decision.exit_required is True
    assert decision.entry_session == decision.session == EXIT_SESSION
    assert decision.holding_session_number == 1
    assert decision.exit_prerequisite_status is ExitPrerequisiteStatus.READY
    assert len(events) == 1 and events[0].side is ExecutionSide.SELL


@pytest.mark.parametrize(
    "adapter",
    [
        ExitEventAdapter(side=ExecutionSide.BUY),
        ExitEventAdapter(session=ENTRY_SESSION),
        ExitEventAdapter(asset_id="NORGATE:9999"),
        ExitEventAdapter(quantity_delta=1),
    ],
)
def test_malformed_adapter_identity_fails_before_phase13(adapter) -> None:
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="INVALID_OPEN_POSITION_EXIT_EVENT_ADAPTER_RESULT",
    ):
        process(
            state_with_positions("NORGATE:1001"),
            (exit_evaluation(low=95.0),),
            transition=recording,
            evaluator=RecordingExitEvaluator(),
            adapter=adapter,
        )

    assert recording.calls == []


def test_adapter_must_return_one_portfolio_execution_event() -> None:
    class MultipleEventAdapter:
        def build_sell_event(self, *, decision, open_position):
            return ()

    recording = RecordingTransition()
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="INVALID_OPEN_POSITION_EXIT_EVENT_ADAPTER_RESULT",
    ):
        process(
            state_with_positions("NORGATE:1001"),
            (exit_evaluation(low=95.0),),
            transition=recording,
            evaluator=RecordingExitEvaluator(),
            adapter=MultipleEventAdapter(),
        )

    assert recording.calls == []


def test_postconstruction_corrupted_phase15b_result_is_rejected() -> None:
    real = OpenPositionExitEvaluator(
        trading_calendar=ExplicitTradingCalendar()
    )

    class CorruptingEvaluator:
        def evaluate(self, evaluation):
            decision = real.evaluate(evaluation)
            object.__setattr__(decision, "holding_session_number", 11)
            return decision

    adapter = ExitEventAdapter()
    recording = RecordingTransition()

    with pytest.raises(HistoricalBacktestValidationError) as raised:
        process(
            state_with_positions("NORGATE:1001"),
            (exit_evaluation(low=95.0),),
            transition=recording,
            evaluator=CorruptingEvaluator(),
            adapter=adapter,
        )

    assert raised.value.code == "INVALID_OPEN_POSITION_EXIT_RESULT"
    assert adapter.calls == []
    assert recording.calls == []


def test_valid_phase15b_result_must_belong_to_exact_evaluation() -> None:
    real = OpenPositionExitEvaluator(
        trading_calendar=ExplicitTradingCalendar()
    )
    supplied = exit_evaluation(low=95.0, close=97.0)
    different_payload = exit_evaluation(high=109.0, close=107.0)

    class WrongPayloadEvaluator:
        def evaluate(self, evaluation):
            assert evaluation is supplied
            return real.evaluate(different_payload)

    adapter = ExitEventAdapter()
    recording = RecordingTransition()

    with pytest.raises(HistoricalBacktestValidationError) as raised:
        process(
            state_with_positions("NORGATE:1001"),
            (supplied,),
            transition=recording,
            evaluator=WrongPayloadEvaluator(),
            adapter=adapter,
        )

    assert raised.value.code == "INVALID_OPEN_POSITION_EXIT_RESULT"
    assert adapter.calls == []
    assert recording.calls == []


def test_protective_adapter_cannot_change_phase10_final_price() -> None:
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="PROTECTIVE_EXIT_PRICE_MISMATCH",
    ):
        process(
            state_with_positions("NORGATE:1001"),
            (exit_evaluation(low=95.0),),
            transition=recording,
            evaluator=RecordingExitEvaluator(),
            adapter=ExitEventAdapter(price_delta=Decimal("0.01")),
        )

    assert recording.calls == []


@pytest.mark.parametrize("session_index", [1, 9])
def test_terminal_exit_without_adapter_fails_closed(session_index) -> None:
    recording = RecordingTransition()
    evaluation = (
        exit_evaluation(low=95.0)
        if session_index == 1
        else exit_evaluation(session_index=9)
    )
    prior = state_with_positions("NORGATE:1001")

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="MISSING_OPEN_POSITION_EXIT_EVENT_ADAPTER",
    ):
        orchestrator(
            transition=recording,
            evaluator=RecordingExitEvaluator(),
        ).process_session(
            prior_state=prior,
            session_input=session_input(
                session=SESSIONS[session_index],
                open_position_exit_evaluations=(evaluation,),
            ),
        )

    assert recording.calls == []


def test_prior_authorized_earnings_exit_needs_adapter_and_can_be_persisted() -> None:
    asset_id = "NORGATE:1001"
    current_index = 5
    evaluation = exit_evaluation(
        asset_id,
        session_index=current_index,
        close=104.0,
        prior_boundary_earnings_decision=earnings_decision(
            asset_id=asset_id,
            boundary_index=current_index - 1,
            scheduled_index=current_index + 1,
        ),
    )
    prior = state_with_positions(asset_id)
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="MISSING_OPEN_POSITION_EXIT_EVENT_ADAPTER",
    ):
        orchestrator(
            transition=recording,
            evaluator=RecordingExitEvaluator(),
        ).process_session(
            prior_state=prior,
            session_input=session_input(
                session=SESSIONS[current_index],
                open_position_exit_evaluations=(evaluation,),
            ),
        )
    assert recording.calls == []

    result = orchestrator(
        evaluator=RecordingExitEvaluator(),
        adapter=ExitEventAdapter(
            settlement_session=SESSIONS[current_index + 1],
            administrative_fill=Decimal("103.40"),
        ),
    ).process_session(
        prior_state=prior,
        session_input=session_input(
            session=SESSIONS[current_index],
            open_position_exit_evaluations=(evaluation,),
        ),
    )

    assert result.open_position_exit_decisions[0].selected_reason is (
        OpenPositionExitReason.EARNINGS_FORCED_EXIT
    )
    assert result.open_position_exit_decisions[0].final_execution_price is None
    assert result.ordered_execution_events[0].fill_price == Decimal("103.40")


def test_max_holding_exit_uses_only_adapter_supplied_execution_facts() -> None:
    current_index = 9
    result = orchestrator(
        evaluator=RecordingExitEvaluator(),
        adapter=ExitEventAdapter(
            settlement_session=SESSIONS[current_index + 1],
            administrative_fill=Decimal("102.25"),
        ),
    ).process_session(
        prior_state=state_with_positions("NORGATE:1001"),
        session_input=session_input(
            session=SESSIONS[current_index],
            open_position_exit_evaluations=(
                exit_evaluation(session_index=current_index, close=103.0),
            ),
        ),
    )

    decision = result.open_position_exit_decisions[0]
    assert decision.selected_reason is OpenPositionExitReason.MAX_HOLDING
    assert decision.reference_exit_price == 103.0
    assert decision.final_execution_price is None
    assert result.ordered_execution_events[0].fill_price == Decimal("102.25")
    assert result.ordered_execution_events[0].execution_cost == Decimal("2.75")


def test_late_earnings_deadline_missed_hold_does_not_sell() -> None:
    asset_id = "NORGATE:1001"
    current_index = 5
    evaluation = exit_evaluation(
        asset_id,
        session_index=current_index,
        earnings_decision=earnings_decision(
            asset_id=asset_id,
            boundary_index=current_index,
            scheduled_index=current_index + 1,
        ),
        prior_boundary_earnings_decision=earnings_decision(
            asset_id=asset_id,
            boundary_index=current_index - 1,
            scheduled_index=None,
            active=False,
        ),
    )
    recording = RecordingTransition()
    result = orchestrator(
        transition=recording,
        evaluator=RecordingExitEvaluator(),
    ).process_session(
        prior_state=state_with_positions(asset_id),
        session_input=session_input(
            session=SESSIONS[current_index],
            open_position_exit_evaluations=(evaluation,),
        ),
    )

    decision = result.open_position_exit_decisions[0]
    assert decision.earnings_exit_status is EarningsExitStatus.DEADLINE_MISSED
    assert decision.exit_required is False
    assert result.ordered_execution_events == ()
    assert len(result.authoritative_state.open_positions) == 1
    assert len(recording.calls) == 1


def test_generated_sale_frees_slot_but_proceeds_stay_unsettled(
    make_candidate_pair,
    make_portfolio_snapshot,
) -> None:
    asset_id = "NORGATE:1001"
    prior = state_with_positions(asset_id, settled_cash=Decimal("100"))
    ranking, candidate = make_candidate_pair(
        "NORGATE:2001",
        session=EXIT_SESSION,
        following=SETTLEMENT_SESSION,
    )
    allocation_portfolio = make_portfolio_snapshot(
        session=EXIT_SESSION,
        cash=100.0,
        equity=1000.0,
        open_positions=(),
    )
    result = process(
        prior,
        (exit_evaluation(low=95.0),),
        evaluator=RecordingExitEvaluator(),
        adapter=ExitEventAdapter(),
        next_session=SETTLEMENT_SESSION,
        ranking_candidates=(ranking,),
        allocation_candidates=(candidate,),
        allocation_portfolio=allocation_portfolio,
    )

    assert result.authoritative_state.open_positions == ()
    assert result.authoritative_state.settled_cash == Decimal("100")
    assert len(result.authoritative_state.pending_settlements) == 1
    assert result.allocation_decision is not None


def test_generated_exit_does_not_change_ranking_or_allocation(
    make_candidate_pair,
    make_portfolio_snapshot,
) -> None:
    ranking, candidate = make_candidate_pair(
        "NORGATE:2001",
        session=EXIT_SESSION,
        following=SETTLEMENT_SESSION,
    )
    allocation_portfolio = make_portfolio_snapshot(
        session=EXIT_SESSION,
        cash=100.0,
        equity=1000.0,
        open_positions=(),
    )
    shared = {
        "next_session": SETTLEMENT_SESSION,
        "ranking_candidates": (ranking,),
        "allocation_candidates": (candidate,),
        "allocation_portfolio": allocation_portfolio,
    }
    exited = process(
        state_with_positions("NORGATE:1001", settled_cash=Decimal("100")),
        (exit_evaluation(low=95.0),),
        evaluator=RecordingExitEvaluator(),
        adapter=ExitEventAdapter(),
        **shared,
    )
    control = process(PortfolioState(settled_cash=Decimal("100")), **shared)

    assert exited.ranking_snapshot == control.ranking_snapshot
    assert exited.ranked_allocation_batch == control.ranked_allocation_batch
    assert exited.allocation_decision == control.allocation_decision
    assert exited.future_entry_intents == control.future_entry_intents


def test_new_t_buy_is_not_fed_back_into_phase15b() -> None:
    evaluator = RecordingExitEvaluator()
    result = process(
        PortfolioState(settled_cash=Decimal("1000")),
        evaluator=evaluator,
        scheduled_execution_events=(
            buy_event(session=EXIT_SESSION, asset_id="NORGATE:2001"),
        ),
    )

    assert evaluator.calls == []
    assert result.open_position_exit_decisions == ()
    assert result.authoritative_state.open_positions[0].asset_id == "NORGATE:2001"


def test_generated_old_position_sell_and_carried_buy_share_one_transition(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    new_asset = "NORGATE:2001"
    ranking, candidate = make_candidate_pair(
        new_asset,
        session=ENTRY_SESSION,
        following=EXIT_SESSION,
    )
    intent_source = HistoricalBacktestOrchestrator().process_session(
        prior_state=PortfolioState(settled_cash=Decimal("10000")),
        session_input=session_input(
            session=ENTRY_SESSION,
            next_session=EXIT_SESSION,
            ranking_candidates=(ranking,),
            allocation_candidates=(candidate,),
            allocation_portfolio=make_portfolio_snapshot(
                session=ENTRY_SESSION,
                cash=10000.0,
                equity=10000.0,
            ),
        ),
    )
    intent = intent_source.future_entry_intents[0]
    bar = StockBar(
        security_id=new_asset,
        symbol="S2001",
        trading_date=EXIT_SESSION,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1_000_000,
    )
    recording = RecordingTransition()
    overlay = EntryEarningsOverlay()
    result = orchestrator(
        transition=recording,
        evaluator=RecordingExitEvaluator(),
        adapter=ExitEventAdapter(),
        entry_execution_service=EntryExecutionService(
            earnings_overlay=overlay,
            trading_calendar=EntryCalendar(),
            execution_cost_service=execution_cost_service,
        ),
        entry_event_adapter=EntryEventAdapter(),
        protective_exit_state_factory=ProtectiveExitService(
            trading_calendar=ExplicitTradingCalendar()
        ),
    ).process_session(
        prior_state=state_with_positions("NORGATE:1001"),
        session_input=session_input(
            scheduled_entry_intents=(intent,),
            entry_execution_bars=(bar,),
            open_position_exit_evaluations=(exit_evaluation(low=95.0),),
        ),
    )

    assert len(recording.calls) == 1
    assert tuple(event.side for event in recording.calls[0][2]) == (
        ExecutionSide.SELL,
        ExecutionSide.BUY,
    )
    assert tuple(
        position.asset_id for position in result.authoritative_state.open_positions
    ) == (new_asset,)
    assert tuple(
        decision.security_id for decision in result.open_position_exit_decisions
    ) == ("NORGATE:1001",)
    # Task 5C-B: the carried entry was evaluated on its own session (HOLD).
    assert tuple(
        decision.security_id
        for decision in result.entry_session_protective_decisions
    ) == (new_asset,)
    assert result.entry_session_protective_decisions[0].exit_required is False


def test_one_failing_exit_among_many_keeps_transition_atomic() -> None:
    prior = state_with_positions("NORGATE:1001", "NORGATE:1002")
    recording = RecordingTransition()
    adapter = ExitEventAdapter(price_delta=Decimal("0.01"))

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="PROTECTIVE_EXIT_PRICE_MISMATCH",
    ):
        process(
            prior,
            (
                exit_evaluation("NORGATE:1001"),
                exit_evaluation("NORGATE:1002", low=95.0),
            ),
            transition=recording,
            evaluator=RecordingExitEvaluator(),
            adapter=adapter,
        )

    assert recording.calls == []
    assert tuple(position.asset_id for position in prior.open_positions) == (
        "NORGATE:1001",
        "NORGATE:1002",
    )


def test_session_result_retains_complete_immutable_phase15b_decision() -> None:
    evaluation = exit_evaluation(low=95.0)
    evaluator = RecordingExitEvaluator()
    result = process(
        state_with_positions("NORGATE:1001"),
        (evaluation,),
        evaluator=evaluator,
        adapter=ExitEventAdapter(),
    )
    authoritative = evaluator.delegate.evaluate(evaluation)

    assert result.open_position_exit_decisions == (authoritative,)
    with pytest.raises((AttributeError, TypeError)):
        result.open_position_exit_decisions[0].exit_required = False
