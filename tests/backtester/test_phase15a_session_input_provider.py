"""Task 5C: the state-aware session-input provider path end to end.

One authoritative loop, two narrow provider boundaries, Phase 15A custody of
Phase 10 protective state, and the frozen Task 5C-A valuation path from
authoritative S[T] to the Phase 12 snapshot.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from stock_swing_d1.backtester import (
    BacktestBuyExecutionEventAdapter,
    BacktestSellExecutionEventAdapter,
    HistoricalBacktestAllocationBoundaryMarks,
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionInput,
    HistoricalBacktestSessionPlan,
    HistoricalBacktestSessionStateInputs,
    HistoricalBacktestValidationError,
    HistoricalDecisionInterval,
)
from stock_swing_d1.backtester.session_input_provider import (
    HistoricalBacktestOpenPositionFacts,
)
from stock_swing_d1.execution.costs import (
    AdministrativeExitPricingService,
    BacktestExecutionCostService,
    ExecutionIdentifierService,
    HistoricalUsEquitySettlementResolver,
    load_backtest_execution_cost_policy,
)
from stock_swing_d1.execution.entry import (
    EntryExecutionService,
    EntryExecutionStatus,
)
from stock_swing_d1.execution.open_position_exit import (
    OpenPositionExitEvaluator,
)
from stock_swing_d1.execution.protective_exit import (
    ProtectiveExitService,
    ProtectiveExitState,
)
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    StockBar,
)
from stock_swing_d1.portfolio import PortfolioAllocationService
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    PortfolioEventKind,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PendingSettlement,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.strategy.baseline import BaselineSignalEvaluator
from stock_swing_d1.valuation import PortfolioValuationMark

from tests.backtester.conftest import COST_POLICY_PATH, decision_time
from tests.backtester.test_phase15a_chronology import (
    EntryCalendar,
    EntryEarningsOverlay,
)
from tests.backtester.test_phase15c_execution_adapters import (
    ExplicitSettlementCalendar,
)
from tests.execution.open_position_exit.conftest import (
    NEW_YORK,
    SESSIONS,
    ExplicitTradingCalendar,
)


ASSET = "NORGATE:1001"
CARRIED_ENTRY_ID = f"ENTRY:{SESSIONS[0].isoformat()}:{ASSET}"
S1, S2, S3 = SESSIONS[0], SESSIONS[1], SESSIONS[2]
INTERVAL = HistoricalDecisionInterval(
    decision_start_date=S1,
    decision_end_date=SESSIONS[-1],
)
MARKET_DATA_REF = ArtifactRef(
    artifact_type="canonical_market_data",
    schema_version="stock_bars_v0_1",
    content_sha256="b" * 64,
)
OTHER_MARKET_DATA_REF = ArtifactRef(
    artifact_type="canonical_market_data",
    schema_version="stock_bars_v0_1",
    content_sha256="c" * 64,
)


# --------------------------------------------------------------------------
# deterministic synthetic canonical facts (never canonical provider data)
# --------------------------------------------------------------------------


def adjusted_history(
    security_id: str, *, end_session: date, count: int = 200
) -> tuple[CorporateActionAdjustedStockBar, ...]:
    sessions: list[date] = []
    cursor = end_session
    while len(sessions) < count:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor -= timedelta(days=1)
    sessions.reverse()
    return tuple(
        CorporateActionAdjustedStockBar(
            security_id=security_id,
            symbol=security_id.replace("NORGATE:", "S"),
            trading_date=session,
            timeframe="D1",
            session_type="regular",
            currency="USD",
            price_basis="capital_special_adjusted",
            open=100.0 + index,
            high=102.0 + index,
            low=98.0 + index,
            close=100.0 + index,
            volume=1_000_000.0,
        )
        for index, session in enumerate(sessions)
    )


def unadjusted_bar(
    security_id: str,
    *,
    session: date,
    close: float,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> StockBar:
    opening = close if open_price is None else open_price
    return StockBar(
        security_id=security_id,
        symbol=security_id.replace("NORGATE:", "S"),
        trading_date=session,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=opening,
        high=close + 2.0 if high is None else high,
        low=close - 2.0 if low is None else low,
        close=close,
        volume=1_000_000,
    )


def candidate_plan(
    session: date,
    *,
    next_session: date | None,
    security_id: str = ASSET,
) -> HistoricalBacktestSessionPlan:
    history = adjusted_history(security_id, end_session=session)
    return HistoricalBacktestSessionPlan(
        session=session,
        decision_time=decision_time(session),
        next_session=next_session,
        indicator_history_bars=history,
        completed_unadjusted_bars=(
            unadjusted_bar(
                security_id, session=session, close=history[-1].close
            ),
        ),
        universe_eligible_security_ids=(security_id,),
    )


def quiet_plan(
    session: date, *, next_session: date | None
) -> HistoricalBacktestSessionPlan:
    return HistoricalBacktestSessionPlan(
        session=session,
        decision_time=decision_time(session),
        next_session=next_session,
    )


# --------------------------------------------------------------------------
# the provider under test: external facts only
# --------------------------------------------------------------------------


class RecordingProvider:
    """Supplies only external facts, and records exactly when it is asked."""

    def __init__(
        self,
        *,
        entry_prices: dict[date, float] | None = None,
        position_bars: dict[date, StockBar] | None = None,
        marks: dict[date, dict[str, Decimal]] | None = None,
        artifact_ref: ArtifactRef = MARKET_DATA_REF,
        mark_artifact_ref: ArtifactRef | None = None,
        state_inputs_error: Exception | None = None,
    ) -> None:
        self.entry_prices = entry_prices or {}
        self.position_bars = position_bars or {}
        self.marks = marks or {}
        self.artifact_ref = artifact_ref
        self.mark_artifact_ref = mark_artifact_ref or artifact_ref
        self.state_inputs_error = state_inputs_error
        self.calls: list[tuple[str, date]] = []
        self.state_contexts = []
        self.mark_contexts = []

    def session_state_inputs(self, *, context):
        self.calls.append(("session_state_inputs", context.session))
        self.state_contexts.append(context)
        if self.state_inputs_error is not None:
            raise self.state_inputs_error
        return HistoricalBacktestSessionStateInputs(
            session=context.session,
            entry_execution_bars=tuple(
                unadjusted_bar(
                    security_id,
                    session=context.session,
                    close=self.entry_prices.get(context.session, 100.0),
                )
                for security_id in context.scheduled_entry_security_ids
            ),
            open_position_facts=tuple(
                HistoricalBacktestOpenPositionFacts(
                    security_id=ref.security_id,
                    market_bar=self.position_bars.get(
                        context.session,
                        unadjusted_bar(
                            ref.security_id,
                            session=context.session,
                            close=101.0,
                        ),
                    ),
                )
                for ref in context.open_position_refs
            ),
        )

    def allocation_boundary_marks(self, *, context):
        self.calls.append(("allocation_boundary_marks", context.session))
        self.mark_contexts.append(context)
        closes = self.marks.get(context.session, {})
        return HistoricalBacktestAllocationBoundaryMarks(
            session=context.session,
            marks=tuple(
                PortfolioValuationMark(
                    security_id=security_id,
                    session=context.session,
                    close=closes.get(security_id, Decimal("101")),
                    source_artifact_ref=self.mark_artifact_ref,
                )
                for security_id in context.security_ids
            ),
        )


class CountingProtectiveFactory:
    def __init__(self) -> None:
        self.delegate = ProtectiveExitService(
            trading_calendar=ExplicitTradingCalendar()
        )
        self.calls = []

    def create_state(self, *, signal, entry_execution):
        self.calls.append((signal.security_id, entry_execution.security_id))
        return self.delegate.create_state(
            signal=signal, entry_execution=entry_execution
        )


class RecordingAllocationService:
    """Wraps the frozen Phase 12 owner to record whether it was reached."""

    def __init__(self) -> None:
        self.delegate = PortfolioAllocationService()
        self.calls = []

    def allocate_ranked_candidates(self, **kwargs):
        self.calls.append(kwargs)
        return self.delegate.allocate_ranked_candidates(**kwargs)

    def settle_reservation(self, **kwargs):
        return self.delegate.settle_reservation(**kwargs)


def build_orchestrator(
    *,
    protective_factory=None,
    transition_service=None,
    allocation_service=None,
    exit_evaluator=None,
) -> HistoricalBacktestOrchestrator:
    cost_service = BacktestExecutionCostService(
        policy=load_backtest_execution_cost_policy(COST_POLICY_PATH)
    )
    kwargs = {
        "baseline_signal_evaluator": BaselineSignalEvaluator(
            earnings_overlay=EntryEarningsOverlay(),
            trading_calendar=EntryCalendar(),
        ),
        "entry_execution_service": EntryExecutionService(
            earnings_overlay=EntryEarningsOverlay(),
            trading_calendar=EntryCalendar(),
            execution_cost_service=cost_service,
        ),
        "entry_event_adapter": BacktestBuyExecutionEventAdapter(
            execution_cost_service=cost_service,
            identifier_service=ExecutionIdentifierService(),
        ),
        "open_position_exit_evaluator": (
            OpenPositionExitEvaluator(
                trading_calendar=ExplicitTradingCalendar()
            )
            if exit_evaluator is None
            else exit_evaluator
        ),
        "open_position_exit_event_adapter": BacktestSellExecutionEventAdapter(
            execution_cost_service=cost_service,
            administrative_pricing_service=AdministrativeExitPricingService(
                policy=cost_service.policy
            ),
            settlement_resolver=HistoricalUsEquitySettlementResolver(
                settlement_calendar=ExplicitSettlementCalendar(SESSIONS)
            ),
            identifier_service=ExecutionIdentifierService(),
        ),
        "protective_exit_state_factory": (
            CountingProtectiveFactory()
            if protective_factory is None
            else protective_factory
        ),
    }
    if transition_service is not None:
        kwargs["transition_service"] = transition_service
    if allocation_service is not None:
        kwargs["allocation_service"] = allocation_service
    return HistoricalBacktestOrchestrator(**kwargs)


def carried_position_state(
    *,
    settled_cash: Decimal = Decimal("10000"),
    quantity: int = 10,
    entry_price: Decimal = Decimal("100"),
    pending: tuple[PendingSettlement, ...] = (),
) -> PortfolioState:
    return PortfolioState(
        settled_cash=settled_cash,
        open_positions=(
            OpenPosition(
                asset_id=ASSET,
                quantity=quantity,
                entry_session=SESSIONS[0],
                entry_price=entry_price,
                entry_execution_id=CARRIED_ENTRY_ID,
                entry_execution_cost=Decimal("0"),
                cost_basis=entry_price * quantity,
            ),
        ),
        pending_settlements=pending,
        applied_events=(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id=CARRIED_ENTRY_ID,
                payload_sha256="a" * 64,
            ),
        )
        + tuple(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id=settlement.source_execution_id,
                payload_sha256="d" * 64,
            )
            for settlement in pending
        ),
        as_of_session=SESSIONS[0],
        state_version=1,
    )


def carried_protective_state(
    *, last_evaluated_session: date = SESSIONS[0]
) -> ProtectiveExitState:
    return ProtectiveExitState._validated(
        security_id=ASSET,
        symbol="S1001",
        signal_session=date(2026, 7, 31),
        signal_time=decision_time(date(2026, 7, 31)).astimezone(NEW_YORK),
        entry_session=SESSIONS[0],
        entry_price=100.0,
        signal_atr_fraction=0.02,
        risk_fraction=0.04,
        stop_price=96.0,
        take_profit_price=108.0,
        last_evaluated_session=last_evaluated_session,
    )


# --------------------------------------------------------------------------
# A. stateful entry, provider operation 1, custody creation
# --------------------------------------------------------------------------


def test_carried_intent_drives_the_next_session_entry_and_custody():
    provider = RecordingProvider()
    factory = CountingProtectiveFactory()
    orchestrator = build_orchestrator(protective_factory=factory)

    result = orchestrator.run_with_session_input_provider(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            candidate_plan(S1, next_session=S2),
            quiet_plan(S2, next_session=S3),
        ),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
    )

    first, second = result.session_results
    assert first.allocation_decision is not None
    assert first.future_entry_intents[0].security_id == ASSET
    assert second.scheduled_entry_intents[0].security_id == ASSET
    assert second.entry_execution_decisions[0].status is (
        EntryExecutionStatus.EXECUTED
    )
    assert result.final_state.open_positions[0].asset_id == ASSET

    # T+1's context exposed exactly the scheduled security and nothing else.
    entry_context = provider.state_contexts[1]
    assert entry_context.scheduled_entry_security_ids == (ASSET,)
    assert entry_context.open_position_refs == ()

    # Phase 10 created the protective state exactly once, for that entry.
    assert factory.calls == [(ASSET, ASSET)]


def test_session_state_inputs_is_invoked_exactly_once_per_session():
    provider = RecordingProvider()

    build_orchestrator().run_with_session_input_provider(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            candidate_plan(S1, next_session=S2),
            quiet_plan(S2, next_session=S3),
        ),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
    )

    state_calls = [call for call in provider.calls if call[0] == "session_state_inputs"]
    assert state_calls == [
        ("session_state_inputs", S1),
        ("session_state_inputs", S2),
    ]


def test_provider_never_prefetches_or_sees_a_future_session():
    provider = RecordingProvider()

    build_orchestrator().run_with_session_input_provider(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            candidate_plan(S1, next_session=S2),
            quiet_plan(S2, next_session=S3),
        ),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
    )

    # Chronology: every call for T precedes every call for T+1.
    sessions_in_call_order = [session for _name, session in provider.calls]
    assert sessions_in_call_order == sorted(sessions_in_call_order)
    assert S3 not in sessions_in_call_order


def test_the_context_exposes_no_authoritative_economics():
    provider = RecordingProvider()

    build_orchestrator().run_with_session_input_provider(
        carried_position_state(),
        (candidate_plan(S2, next_session=S3),),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
        initial_protective_states=(carried_protective_state(),),
    )

    context = provider.state_contexts[0]
    fields = set(type(context).model_fields)
    assert fields == {
        "schema_version",
        "session",
        "decision_time",
        "next_session",
        "open_position_refs",
        "scheduled_entry_security_ids",
    }
    ref_fields = set(type(context.open_position_refs[0]).model_fields)
    assert ref_fields == {"security_id", "entry_session"}

    mark_context = provider.mark_contexts[0]
    assert set(type(mark_context).model_fields) == {
        "schema_version",
        "session",
        "security_ids",
        "expected_market_data_artifact_ref",
    }


# --------------------------------------------------------------------------
# C/D. protective HOLD chain and retirement
# --------------------------------------------------------------------------


class RecordingExitEvaluator:
    """Captures exactly which protective-state object Phase 15B received."""

    def __init__(self) -> None:
        self.delegate = OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        )
        self.supplied_states = []

    def evaluate(self, evaluation):
        self.supplied_states.append(evaluation.protective_state)
        return self.delegate.evaluate(evaluation)


def test_hold_advances_the_exact_phase10_object_into_the_next_session():
    provider = RecordingProvider()
    evaluator = RecordingExitEvaluator()
    orchestrator = build_orchestrator(exit_evaluator=evaluator)

    result = orchestrator.run_with_session_input_provider(
        carried_position_state(),
        (
            quiet_plan(S2, next_session=S3),
            quiet_plan(S3, next_session=SESSIONS[3]),
        ),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
        initial_protective_states=(carried_protective_state(),),
    )

    first, second = result.session_results
    advanced = first.open_position_exit_decisions[
        0
    ].protective_exit_decision.resulting_state
    assert advanced is not None
    assert advanced.last_evaluated_session == S2

    # Custody forwarded the exact Phase 10 object, not a reconstruction.
    assert len(evaluator.supplied_states) == 2
    assert evaluator.supplied_states[1] is advanced
    assert second.open_position_exit_decisions[0].session == S3


def test_a_surviving_position_without_carried_state_fails_closed():
    """Custody, not the provider, owns protective state; absence is fatal."""

    from stock_swing_d1.backtester.validation import (
        reconcile_protective_custody,
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="PROTECTIVE_CUSTODY_COVERAGE_MISMATCH",
    ):
        reconcile_protective_custody(
            authoritative_state=carried_position_state(),
            candidate_custody={},
        )


def test_terminal_exit_retires_custody_after_the_transition():
    stopped_bar = unadjusted_bar(
        ASSET, session=S2, close=90.0, open_price=95.0, high=96.0, low=90.0
    )
    provider = RecordingProvider(position_bars={S2: stopped_bar})

    result = build_orchestrator().run_with_session_input_provider(
        carried_position_state(),
        (
            quiet_plan(S2, next_session=S3),
            quiet_plan(S3, next_session=SESSIONS[3]),
        ),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
        initial_protective_states=(carried_protective_state(),),
    )

    first, second = result.session_results
    assert first.open_position_exit_decisions[0].exit_required is True
    assert first.authoritative_state.open_positions == ()
    # The retired security is never asked about again.
    assert provider.state_contexts[1].open_position_refs == ()
    assert second.open_position_exit_decisions == ()


# --------------------------------------------------------------------------
# E. initial protective-state seeding
# --------------------------------------------------------------------------


def test_initial_seed_must_cover_the_carried_in_positions_exactly():
    provider = RecordingProvider()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="MISSING_INITIAL_PROTECTIVE_STATE",
    ):
        build_orchestrator().run_with_session_input_provider(
            carried_position_state(),
            (quiet_plan(S2, next_session=S3),),
            provider=provider,
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
        )
    assert provider.calls == []


def test_extra_initial_seed_fails_closed():
    other = ProtectiveExitState._validated(
        security_id="NORGATE:2002",
        symbol="S2002",
        signal_session=date(2026, 7, 31),
        signal_time=decision_time(date(2026, 7, 31)).astimezone(NEW_YORK),
        entry_session=SESSIONS[0],
        entry_price=100.0,
        signal_atr_fraction=0.02,
        risk_fraction=0.04,
        stop_price=96.0,
        take_profit_price=108.0,
        last_evaluated_session=SESSIONS[0],
    )
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="EXTRA_INITIAL_PROTECTIVE_STATE",
    ):
        build_orchestrator().run_with_session_input_provider(
            carried_position_state(),
            (quiet_plan(S2, next_session=S3),),
            provider=RecordingProvider(),
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(carried_protective_state(), other),
        )


def test_duplicate_initial_seed_fails_closed():
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="DUPLICATE_INITIAL_PROTECTIVE_STATE",
    ):
        build_orchestrator().run_with_session_input_provider(
            carried_position_state(),
            (quiet_plan(S2, next_session=S3),),
            provider=RecordingProvider(),
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(
                carried_protective_state(),
                carried_protective_state(),
            ),
        )


def test_mismatched_initial_seed_identity_fails_closed():
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="PROTECTIVE_STATE_IDENTITY_MISMATCH",
    ):
        build_orchestrator().run_with_session_input_provider(
            carried_position_state(),
            (quiet_plan(S2, next_session=S3),),
            provider=RecordingProvider(),
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(
                ProtectiveExitState._validated(
                    security_id=ASSET,
                    symbol="S1001",
                    signal_session=date(2026, 7, 31),
                    signal_time=decision_time(
                        date(2026, 7, 31)
                    ).astimezone(NEW_YORK),
                    entry_session=SESSIONS[1],
                    entry_price=100.0,
                    signal_atr_fraction=0.02,
                    risk_fraction=0.04,
                    stop_price=96.0,
                    take_profit_price=108.0,
                    last_evaluated_session=SESSIONS[1],
                ),
            ),
        )


def test_a_cash_only_initial_state_rejects_a_non_empty_seed():
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="EXTRA_INITIAL_PROTECTIVE_STATE",
    ):
        build_orchestrator().run_with_session_input_provider(
            PortfolioState(settled_cash=Decimal("10000")),
            (quiet_plan(S2, next_session=S3),),
            provider=RecordingProvider(),
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(carried_protective_state(),),
        )


# --------------------------------------------------------------------------
# F. custody atomicity
# --------------------------------------------------------------------------


class FailingTransition:
    def __init__(self, *, fail_on: date) -> None:
        self.fail_on = fail_on

    def transition(self, previous_state, session, events, dividend_evidence=()):
        if session == self.fail_on:
            raise RuntimeError("deliberate Phase 13 failure")
        return PortfolioTransitionEngine.transition(
            previous_state, session, events, dividend_evidence=dividend_evidence
        )


def test_provider_failure_publishes_nothing():
    provider = RecordingProvider(
        state_inputs_error=RuntimeError("provider is unavailable")
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="SESSION_INPUT_PROVIDER_FAILED",
    ):
        build_orchestrator().run_with_session_input_provider(
            carried_position_state(),
            (quiet_plan(S2, next_session=S3),),
            provider=provider,
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(carried_protective_state(),),
        )

    assert len(provider.calls) == 1


def test_transition_failure_publishes_no_session_and_no_custody():
    provider = RecordingProvider()

    with pytest.raises(RuntimeError, match="deliberate Phase 13 failure"):
        build_orchestrator(
            transition_service=FailingTransition(fail_on=S3)
        ).run_with_session_input_provider(
            carried_position_state(),
            (
                quiet_plan(S2, next_session=S3),
                quiet_plan(S3, next_session=SESSIONS[3]),
            ),
            provider=provider,
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(carried_protective_state(),),
        )


def test_protective_creation_failure_publishes_nothing():
    class FailingFactory:
        def create_state(self, *, signal, entry_execution):
            from stock_swing_d1.execution.protective_exit import (
                ProtectiveExitValidationError,
            )

            raise ProtectiveExitValidationError(
                "INVALID_SIGNAL", "deliberate Phase 10 failure"
            )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="PROTECTIVE_STATE_CREATION_FAILED",
    ):
        build_orchestrator(
            protective_factory=FailingFactory()
        ).run_with_session_input_provider(
            PortfolioState(settled_cash=Decimal("10000")),
            (
                candidate_plan(S1, next_session=S2),
                quiet_plan(S2, next_session=S3),
            ),
            provider=RecordingProvider(),
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
        )


# --------------------------------------------------------------------------
# I. allocation-mark invocation arithmetic
# --------------------------------------------------------------------------


def test_candidates_with_open_positions_make_exactly_one_mark_call():
    provider = RecordingProvider()

    build_orchestrator().run_with_session_input_provider(
        carried_position_state(),
        (candidate_plan(S2, next_session=S3),),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
        initial_protective_states=(carried_protective_state(),),
    )

    mark_calls = [
        call for call in provider.calls if call[0] == "allocation_boundary_marks"
    ]
    assert mark_calls == [("allocation_boundary_marks", S2)]
    assert provider.mark_contexts[0].security_ids == (ASSET,)


def test_candidates_without_open_positions_make_no_mark_call():
    provider = RecordingProvider()

    result = build_orchestrator().run_with_session_input_provider(
        PortfolioState(settled_cash=Decimal("10000")),
        (candidate_plan(S1, next_session=S2),),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
    )

    assert not any(
        call[0] == "allocation_boundary_marks" for call in provider.calls
    )
    assert result.session_results[0].allocation_decision is not None


def test_no_candidates_means_no_mark_call_and_no_allocation():
    provider = RecordingProvider()

    result = build_orchestrator().run_with_session_input_provider(
        carried_position_state(),
        (quiet_plan(S2, next_session=S3),),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
        initial_protective_states=(carried_protective_state(),),
    )

    assert not any(
        call[0] == "allocation_boundary_marks" for call in provider.calls
    )
    assert result.session_results[0].allocation_decision is None


# --------------------------------------------------------------------------
# run-level market-data provenance authority
# --------------------------------------------------------------------------


def test_internally_consistent_but_wrong_artifact_ref_is_rejected():
    provider = RecordingProvider(mark_artifact_ref=OTHER_MARKET_DATA_REF)
    allocation_service = RecordingAllocationService()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="VALUATION_MARK_ARTIFACT_MISMATCH",
    ):
        build_orchestrator(
            allocation_service=allocation_service
        ).run_with_session_input_provider(
            carried_position_state(),
            (candidate_plan(S2, next_session=S3),),
            provider=provider,
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(carried_protective_state(),),
        )

    # The returned marks were internally consistent: only run-level
    # authority failed, and it failed before Phase 12 was reached.
    returned = _last_marks(provider)
    assert len({mark.source_artifact_ref for mark in returned}) == 1
    assert returned[0].source_artifact_ref != MARKET_DATA_REF
    assert allocation_service.calls == []


def _last_marks(provider: RecordingProvider):
    context = provider.mark_contexts[-1]
    return provider.allocation_boundary_marks(context=context).marks


def test_a_non_artifact_ref_run_argument_fails_before_any_provider_call():
    provider = RecordingProvider()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="VALUATION_MARK_ARTIFACT_MISMATCH",
    ):
        build_orchestrator().run_with_session_input_provider(
            PortfolioState(settled_cash=Decimal("10000")),
            (candidate_plan(S1, next_session=S2),),
            provider=provider,
            decision_interval=INTERVAL,
            market_data_artifact_ref="not-an-artifact-ref",
        )

    assert provider.calls == []


def test_missing_mark_coverage_fails_closed():
    class ShortProvider(RecordingProvider):
        def allocation_boundary_marks(self, *, context):
            self.calls.append(("allocation_boundary_marks", context.session))
            self.mark_contexts.append(context)
            return HistoricalBacktestAllocationBoundaryMarks(
                session=context.session, marks=()
            )

    with pytest.raises(
        HistoricalBacktestValidationError, match="MISSING_VALUATION_MARK"
    ):
        build_orchestrator().run_with_session_input_provider(
            carried_position_state(),
            (candidate_plan(S2, next_session=S3),),
            provider=ShortProvider(),
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(carried_protective_state(),),
        )


# --------------------------------------------------------------------------
# valuation, projection and pending settlements
# --------------------------------------------------------------------------


def test_pending_settlements_are_in_equity_but_not_in_cash_available():
    settlement = PendingSettlement(
        settlement_id="S-1",
        source_execution_id="E-1",
        asset_id="NORGATE:2002",
        amount=Decimal("2500"),
        trade_session=SESSIONS[0],
        settlement_session=SESSIONS[3],
    )
    provider = RecordingProvider(marks={S2: {ASSET: Decimal("110")}})

    result = build_orchestrator().run_with_session_input_provider(
        carried_position_state(pending=(settlement,)),
        (candidate_plan(S2, next_session=S3),),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
        initial_protective_states=(carried_protective_state(),),
    )

    allocation = result.session_results[0].allocation_decision
    state = result.session_results[0].authoritative_state
    assert allocation.starting_cash == float(state.settled_cash)
    assert allocation.starting_portfolio_equity == float(
        state.settled_cash + Decimal("2500") + Decimal("110") * 10
    )
    assert allocation.starting_cash < allocation.starting_portfolio_equity


def test_the_provider_never_returns_equity():
    fields = set(HistoricalBacktestAllocationBoundaryMarks.model_fields)

    assert fields == {"schema_version", "session", "marks"}
    assert not any("equity" in field for field in fields)
    assert not any(
        "value" in field
        for field in PortfolioValuationMark.model_fields
        if field != "price_basis"
    )


# --------------------------------------------------------------------------
# legacy compatibility and equivalence
# --------------------------------------------------------------------------


def test_legacy_and_provider_paths_agree_on_a_premateralizable_run():
    provider = RecordingProvider()
    plan = candidate_plan(S1, next_session=S2)

    provider_result = build_orchestrator().run_with_session_input_provider(
        PortfolioState(settled_cash=Decimal("10000")),
        (plan,),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
    )

    legacy_input = HistoricalBacktestSessionInput(
        session=plan.session,
        decision_time=plan.decision_time,
        next_session=plan.next_session,
        indicator_history_bars=plan.indicator_history_bars,
        completed_unadjusted_bars=plan.completed_unadjusted_bars,
        universe_eligible_security_ids=plan.universe_eligible_security_ids,
        allocation_portfolio=provider_result.session_results[
            0
        ].allocation_decision
        and _snapshot_like(provider_result),
    )
    legacy_result = build_orchestrator().run(
        PortfolioState(settled_cash=Decimal("10000")),
        (legacy_input,),
        decision_interval=INTERVAL,
    )

    assert legacy_result.initial_state_fingerprint == (
        provider_result.initial_state_fingerprint
    )
    assert legacy_result.final_state_fingerprint == (
        provider_result.final_state_fingerprint
    )
    assert legacy_result.final_state == provider_result.final_state
    assert legacy_result.session_results == provider_result.session_results


def _snapshot_like(provider_result):
    from stock_swing_d1.portfolio import PortfolioSnapshot

    session = provider_result.session_results[0]
    allocation = session.allocation_decision
    return PortfolioSnapshot(
        allocation_session=session.session,
        decision_time=session.decision_time,
        portfolio_equity=allocation.starting_portfolio_equity,
        cash_available=allocation.starting_cash,
    )


def test_process_session_remains_non_canonical():
    orchestrator = build_orchestrator()
    result = orchestrator.process_session(
        prior_state=PortfolioState(settled_cash=Decimal("10000")),
        session_input=HistoricalBacktestSessionInput(
            session=S1,
            decision_time=decision_time(S1),
            next_session=S2,
        ),
    )

    assert not hasattr(result, "dividend_run_evidence")
    assert type(result).__name__ == "HistoricalBacktestSessionResult"


def test_the_session_plan_carries_no_state_dependent_field():
    fields = set(HistoricalBacktestSessionPlan.model_fields)

    assert fields == {
        "schema_version",
        "session",
        "decision_time",
        "next_session",
        "distribution_events",
        "distribution_coverage",
        "pre_open_corporate_actions",
        "indicator_history_bars",
        "completed_unadjusted_bars",
        "universe_eligible_security_ids",
    }
    for forbidden in (
        "scheduled_entry_intents",
        "entry_execution_bars",
        "open_position_exit_evaluations",
        "ranking_candidates",
        "allocation_candidates",
        "allocation_portfolio",
        "scheduled_execution_events",
    ):
        assert forbidden not in fields


# --------------------------------------------------------------------------
# G. dividend-aware provider run
# --------------------------------------------------------------------------


def test_dividend_evidence_is_validated_before_the_first_transition():
    from tests.backtester.test_phase15a_dividend_evidence import (
        make_coverage,
        make_proof,
    )
    from stock_swing_d1.data.ordinary_dividend_run_policy import (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    )

    class RecordingTransition:
        def __init__(self) -> None:
            self.calls = []

        def transition(
            self, previous_state, session, events, dividend_evidence=()
        ):
            self.calls.append((session, tuple(dividend_evidence)))
            return PortfolioTransitionEngine.transition(
                previous_state,
                session,
                events,
                dividend_evidence=dividend_evidence,
            )

    recording = RecordingTransition()
    plans = (
        HistoricalBacktestSessionPlan(
            session=S2,
            decision_time=decision_time(S2),
            next_session=S3,
            distribution_coverage=make_coverage(session=S2),
        ),
        HistoricalBacktestSessionPlan(
            session=S3,
            decision_time=decision_time(S3),
            next_session=SESSIONS[3],
            distribution_coverage=make_coverage(session=S3),
        ),
    )

    result = build_orchestrator(
        transition_service=recording
    ).run_with_session_input_provider(
        carried_position_state(),
        plans,
        provider=RecordingProvider(),
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
        initial_protective_states=(carried_protective_state(),),
        dividend_accounting_policy_ref=(
            ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
        ),
        processed_session_contiguity_proof=make_proof(sessions=(S2, S3)),
    )

    assert result.schema_version == "historical_backtest_run_result.v0.3"
    assert result.dividend_run_evidence is not None
    assert tuple(
        item.session for item in result.dividend_run_evidence.session_evidence
    ) == (S2, S3)
    # Phase 13 received the validated per-session evidence, unchanged.
    assert [session for session, _evidence in recording.calls] == [S2, S3]
    assert all(evidence == () for _session, evidence in recording.calls)


def test_a_missing_dividend_coverage_fails_before_any_provider_call():
    from tests.backtester.test_phase15a_dividend_evidence import make_proof
    from stock_swing_d1.data.ordinary_dividend_run_policy import (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    )

    provider = RecordingProvider()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="ORDINARY_DIVIDEND_COVERAGE_MISSING",
    ):
        build_orchestrator().run_with_session_input_provider(
            carried_position_state(),
            (quiet_plan(S2, next_session=S3),),
            provider=provider,
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(carried_protective_state(),),
            dividend_accounting_policy_ref=(
                ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
            ),
            processed_session_contiguity_proof=make_proof(sessions=(S2,)),
        )

    assert provider.calls == []


def test_the_provider_cannot_supply_dividend_evidence():
    """Dividend facts are plan-owned; the provider contract has no field."""

    assert not any(
        "distribution" in field
        for field in HistoricalBacktestSessionStateInputs.model_fields
    )
    assert not any(
        "dividend" in field
        for field in HistoricalBacktestSessionStateInputs.model_fields
    )


# --------------------------------------------------------------------------
# corporate-action non-scope and legacy signature preservation
# --------------------------------------------------------------------------


def test_pre_open_corporate_actions_still_fail_closed_in_provider_mode():
    from stock_swing_d1.models import CorporateActionEvent

    event = CorporateActionEvent(
        security_id=ASSET,
        symbol="S1001",
        event_date=S2,
        event_type="split",
        source_asset_id=1001,
        date_semantics="effective_date",
        old_shares=1.0,
        new_shares=2.0,
        terms_verified=True,
        source_provider="Norgate Data",
    )
    plan = HistoricalBacktestSessionPlan(
        session=S2,
        decision_time=decision_time(S2),
        next_session=S3,
        pre_open_corporate_actions=(event,),
    )
    provider = RecordingProvider()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="UNSUPPORTED_CORPORATE_ACTION_TRANSITION",
    ):
        build_orchestrator().run_with_session_input_provider(
            carried_position_state(),
            (plan,),
            provider=provider,
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(carried_protective_state(),),
        )

    assert provider.calls == []


def test_the_legacy_run_signature_is_unchanged():
    import inspect

    signature = inspect.signature(HistoricalBacktestOrchestrator.run)

    assert list(signature.parameters) == [
        "self",
        "initial_state",
        "sessions",
        "decision_interval",
        "dividend_accounting_policy_ref",
        "processed_session_contiguity_proof",
    ]


def test_there_is_one_authoritative_run_loop():
    """Both entry points delegate to the same private session sequencer."""

    import inspect

    legacy = inspect.getsource(HistoricalBacktestOrchestrator.run)
    provider_path = inspect.getsource(
        HistoricalBacktestOrchestrator.run_with_session_input_provider
    )
    loop_body = inspect.getsource(
        HistoricalBacktestOrchestrator._run_sessions
    )

    assert "self._run_sessions(" in legacy
    assert "self._run_sessions(" in provider_path
    assert "for index, item in enumerate(" not in legacy
    assert "for index, item in enumerate(" not in provider_path
    assert "self._process_session(" in loop_body
    assert legacy.count("self._process_session(") == 0
    assert provider_path.count("self._process_session(") == 0


# --------------------------------------------------------------------------
# Task 5C-B: entry-session protective evaluation closes the T -> T+1 seam
# --------------------------------------------------------------------------


def test_entry_session_hold_is_evaluated_on_t_and_carried_exactly_into_t_plus_1():
    """Task 5C-B: an entry executed on T is created by Phase 10 exactly once,
    evaluated by Phase 15B on T exactly once, and the exact HOLD
    ``resulting_state`` (``last_evaluated_session == T``) is the custody
    object Phase 15B receives at T+1 -- so the run no longer fails closed
    at T+1 with EXIT_SESSION_MISMATCH."""

    provider = RecordingProvider()
    factory = CountingProtectiveFactory()
    evaluator = RecordingExitEvaluator()

    result = build_orchestrator(
        protective_factory=factory, exit_evaluator=evaluator
    ).run_with_session_input_provider(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            candidate_plan(S1, next_session=S2),
            quiet_plan(S2, next_session=S3),
            quiet_plan(S3, next_session=SESSIONS[3]),
        ),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
    )

    _first, entry_session, following = result.session_results
    # create_state exactly once, for the one executed entry.
    assert factory.calls == [(ASSET, ASSET)]
    (entry_decision,) = entry_session.entry_session_protective_decisions
    assert entry_decision.session == entry_decision.entry_session == S2
    assert entry_decision.holding_session_number == 1
    assert entry_decision.exit_required is False
    hold_state = entry_decision.protective_exit_decision.resulting_state
    assert hold_state is not None
    assert hold_state.last_evaluated_session == S2
    # Entry-session evaluation exactly once on T, then the T+1 evaluation
    # received THAT EXACT object (not a reconstruction, not a fresh state).
    assert len(evaluator.supplied_states) == 2
    assert evaluator.supplied_states[0].last_evaluated_session is None
    assert evaluator.supplied_states[0].entry_session == S2
    assert evaluator.supplied_states[1] is hold_state
    assert following.open_position_exit_decisions[0].session == S3
    assert entry_session.schema_version == (
        "historical_backtest_session_result.v0.3"
    )
