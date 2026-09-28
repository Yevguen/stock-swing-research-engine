"""Task 5C-B / 5C-C end to end through the one shared Phase 15A session body.

An entry executed on T is created by Phase 10 and evaluated by Phase 15B on T
itself; a HOLD carries the exact Phase 10 object into T+1, a terminal
protective decision becomes an ordinary round-trip SELL that Phase 13 applies
after the BUY inside the same single transition.  Also the Task 5C defect
corrections that live at the Phase 15A boundary: forged nested marks, the
ranking-before-marks chronology, and ``provider=None``.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    project_closed_trades,
    project_entries,
    project_exits,
)
from stock_swing_d1.backtester import (
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionInput,
    HistoricalBacktestSessionResult,
    HistoricalBacktestValidationError,
)
from stock_swing_d1.backtester import orchestration as orchestration_module
from stock_swing_d1.execution.open_position_exit import (
    ExitBoundary,
    ExitPrerequisiteStatus,
    IntrabarAmbiguityStatus,
    OpenPositionExitReason,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.portfolio import PortfolioSnapshot
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    add_exact_decimal,
    subtract_exact_decimal,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from stock_swing_d1.valuation import PortfolioValuationMark

from tests.backtester.test_phase15a_session_input_provider import (
    ASSET,
    INTERVAL,
    MARKET_DATA_REF,
    S1,
    S2,
    S3,
    CountingProtectiveFactory,
    RecordingAllocationService,
    RecordingExitEvaluator,
    RecordingProvider,
    build_orchestrator,
    candidate_plan,
    carried_position_state,
    carried_protective_state,
    quiet_plan,
    unadjusted_bar,
)
from tests.execution.open_position_exit.conftest import SESSIONS


S4 = SESSIONS[3]


class EntryBarProvider(RecordingProvider):
    """RecordingProvider whose session-T entry execution bar is explicit."""

    def __init__(self, entry_bar: StockBar, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entry_bar = entry_bar

    def session_state_inputs(self, *, context):
        inputs = super().session_state_inputs(context=context)
        if context.session == self.entry_bar.trading_date:
            return inputs.model_copy(update={"entry_execution_bars": (self.entry_bar,)})
        return inputs


class RecordingTransition:
    def __init__(self) -> None:
        self.calls = []

    def transition(self, previous_state, session, events, dividend_evidence=()):
        self.calls.append((session, tuple(events)))
        return PortfolioTransitionEngine.transition(
            previous_state, session, events, dividend_evidence=dividend_evidence
        )


def stop_bar() -> StockBar:
    # Entry ~100.05 -> stop ~97.37: the low pierces it, the high does not
    # reach the target.
    return unadjusted_bar(
        ASSET, session=S2, close=90.0, open_price=100.0, high=101.0, low=80.0
    )


def target_bar() -> StockBar:
    return unadjusted_bar(
        ASSET, session=S2, close=110.0, open_price=100.0, high=120.0, low=99.0
    )


def both_bar() -> StockBar:
    return unadjusted_bar(
        ASSET, session=S2, close=100.0, open_price=100.0, high=120.0, low=80.0
    )


def run_round_trip(
    entry_bar: StockBar,
    *,
    plans=None,
    orchestrator=None,
    provider=None,
):
    provider = provider or EntryBarProvider(entry_bar)
    orchestrator = orchestrator or build_orchestrator()
    result = orchestrator.run_with_session_input_provider(
        PortfolioState(settled_cash=Decimal("10000")),
        plans
        or (
            candidate_plan(S1, next_session=S2),
            quiet_plan(S2, next_session=S3),
            quiet_plan(S3, next_session=S4),
        ),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
    )
    return result, provider


# ---------------------------------------------------------------------------
# Same-session round trips (Task 5C-C, tests 11-17, 42)
# ---------------------------------------------------------------------------


def _assert_round_trip(result, recording: RecordingTransition, reason):
    entry_session = result.session_results[1]
    assert entry_session.schema_version == (
        "historical_backtest_session_result.v0.3"
    )
    (decision,) = entry_session.entry_session_protective_decisions
    assert decision.exit_required is True
    assert decision.selected_reason is reason
    assert decision.exit_boundary is ExitBoundary.INTRADAY
    assert decision.session == decision.entry_session == S2
    assert decision.holding_session_number == 1
    assert decision.exit_prerequisite_status is ExitPrerequisiteStatus.READY

    buy, sell = entry_session.ordered_execution_events
    assert (buy.side, sell.side) == (ExecutionSide.BUY, ExecutionSide.SELL)
    assert buy.asset_id == sell.asset_id == ASSET
    assert buy.session == sell.session == S2
    assert sell.quantity == buy.quantity
    assert sell.execution_id != buy.execution_id
    assert sell.execution_id == (
        f"EXIT:{buy.execution_id}:{S2.isoformat()}:{ASSET}"
    )
    # The SELL fill is exactly the Phase 10 final execution price: Phase 10
    # already applied the protective slippage, Phase 15C applied none again.
    assert sell.fill_price == Decimal(str(decision.final_execution_price))
    assert sell.settlement_session is not None and sell.settlement_session > S2

    # Exactly one Phase 13 transition on T carried both executions.
    s2_calls = [call for call in recording.calls if call[0] == S2]
    assert len(s2_calls) == 1
    assert {event.execution_id for event in s2_calls[0][1]} == {
        buy.execution_id,
        sell.execution_id,
    }

    buy_row, sell_row = entry_session.state_transition_result.ledger_entries
    assert buy_row.event_type is PortfolioLedgerEventType.BUY_APPLIED
    assert sell_row.event_type is PortfolioLedgerEventType.SELL_APPLIED

    state = entry_session.authoritative_state
    buy_cash = add_exact_decimal(
        Decimal(buy.quantity) * buy.fill_price, buy.execution_cost
    )
    net_proceeds = subtract_exact_decimal(
        Decimal(sell.quantity) * sell.fill_price, sell.execution_cost
    )
    assert state.open_positions == ()
    assert state.settled_cash == Decimal("10000") - buy_cash
    (pending,) = state.pending_settlements
    assert pending.amount == net_proceeds
    assert pending.source_execution_id == sell.execution_id
    assert buy_row.settled_cash_delta == -buy_cash
    assert sell_row.pending_cash_delta == net_proceeds
    return decision, buy, sell


def test_same_session_stop_round_trip_is_persisted_in_one_transition():
    recording = RecordingTransition()
    result, provider = run_round_trip(
        stop_bar(), orchestrator=build_orchestrator(transition_service=recording)
    )

    decision, _buy, _sell = _assert_round_trip(
        result, recording, OpenPositionExitReason.STOP_LOSS
    )
    assert decision.intrabar_ambiguity_status is (
        IntrabarAmbiguityStatus.NOT_AMBIGUOUS
    )
    # The round-tripped security is absent from S[T] and from custody: the
    # provider is never asked about it at T+1 and nothing is evaluated there.
    assert provider.state_contexts[2].open_position_refs == ()
    assert result.session_results[2].open_position_exit_decisions == ()
    assert result.final_state.open_positions == ()


def test_same_session_target_round_trip_is_persisted_in_one_transition():
    recording = RecordingTransition()
    result, _provider = run_round_trip(
        target_bar(),
        orchestrator=build_orchestrator(transition_service=recording),
    )

    _assert_round_trip(result, recording, OpenPositionExitReason.TAKE_PROFIT)


def test_stop_and_target_touched_on_entry_session_keeps_stop_first_ambiguity():
    recording = RecordingTransition()
    result, _provider = run_round_trip(
        both_bar(), orchestrator=build_orchestrator(transition_service=recording)
    )

    decision, _buy, sell = _assert_round_trip(
        result, recording, OpenPositionExitReason.STOP_LOSS
    )
    assert decision.intrabar_ambiguity_status is (
        IntrabarAmbiguityStatus.STOP_AND_TARGET_TOUCHED
    )
    assert decision.triggered_reasons == (
        OpenPositionExitReason.STOP_LOSS,
        OpenPositionExitReason.TAKE_PROFIT,
    )
    assert sell.fill_price == Decimal(str(decision.final_execution_price))


def test_same_session_sell_proceeds_are_unavailable_to_the_t_allocation():
    """Test 18: the pending round-trip proceeds are inside equity and
    outside cash_available at T's own allocation boundary; the closed slot
    is free again (no cooldown rule is invented)."""

    result, provider = run_round_trip(
        stop_bar(),
        plans=(
            candidate_plan(S1, next_session=S2),
            candidate_plan(S2, next_session=S3),
        ),
    )

    entry_session = result.session_results[1]
    state = entry_session.authoritative_state
    allocation = entry_session.allocation_decision
    assert allocation is not None
    assert allocation.starting_cash == float(state.settled_cash)
    (pending,) = state.pending_settlements
    assert allocation.starting_portfolio_equity == float(
        add_exact_decimal(state.settled_cash, pending.amount)
    )
    assert allocation.starting_cash < allocation.starting_portfolio_equity
    assert allocation.starting_open_position_count == 0
    # No mark call: S[T] holds no position after the round trip.
    assert not any(
        call[0] == "allocation_boundary_marks" for call in provider.calls
    )
    # The same security may be admitted again for T+1.
    assert [intent.security_id for intent in entry_session.future_entry_intents] == [
        ASSET
    ]


# ---------------------------------------------------------------------------
# Phase 15D linkage / projection of the round trip (tests 19, 20, 32-36)
# ---------------------------------------------------------------------------


def test_phase15d_projects_exactly_one_same_session_closed_trade():
    result, _provider = run_round_trip(stop_bar())
    entry_session = result.session_results[1]
    buy, sell = entry_session.ordered_execution_events
    buy_row = entry_session.state_transition_result.ledger_entries[0]

    (entry,) = project_entries(result)
    (exit_record,) = project_exits(result)
    (trade,) = project_closed_trades(result)

    assert entry.entry_execution_id == buy.execution_id
    assert exit_record.entry_execution_id == buy.execution_id
    assert exit_record.exit_execution_id == sell.execution_id
    assert trade.trade_id == buy.execution_id
    assert trade.entry_session == trade.exit_session == S2
    assert trade.carried_in is False
    # Authoritative cost basis: exact sign reversal of BUY_APPLIED.
    ledger_basis = subtract_exact_decimal(Decimal("0"), buy_row.settled_cash_delta)
    assert trade.entry_cost_basis == ledger_basis
    assert entry.cost_basis == ledger_basis
    # qty * fill + cost only reconciles to it.
    assert trade.entry_cost_basis == add_exact_decimal(
        Decimal(trade.quantity) * trade.entry_fill_price, trade.entry_execution_cost
    )
    # Realized P&L includes both executions' costs exactly once.
    assert trade.realized_pnl == subtract_exact_decimal(
        trade.net_exit_proceeds, trade.entry_cost_basis
    )
    assert trade.net_exit_proceeds == subtract_exact_decimal(
        Decimal(sell.quantity) * sell.fill_price, sell.execution_cost
    )
    assert trade.entry_execution_cost == buy.execution_cost
    assert trade.exit_execution_cost == sell.execution_cost
    assert trade.exit_reason.value == "STOP_LOSS"


def test_full_phase15d_audit_assembles_over_a_same_session_round_trip():
    """The complete audit -- source validation, ledger replay, valuation and
    the OD-21.11 P&L identity -- passes over a round trip: equity moves by
    exactly the realized P&L on T, and exactly one closed trade exists."""

    from stock_swing_d1.backtest_results import (
        HistoricalBacktestResultService,
        PolicyArtifactRef,
        build_run_manifest,
        build_valuation_policy_ref,
        build_valuation_snapshot,
    )
    from stock_swing_d1.execution.costs import (
        BacktestExecutionCostService,
        load_backtest_execution_cost_policy,
    )
    from stock_swing_d1.portfolio import PORTFOLIO_ALLOCATION_POLICY_REF
    from stock_swing_d1.provenance import ArtifactRef
    from stock_swing_d1.ranking.hashing import CANDIDATE_RANKING_POLICY_FINGERPRINT
    from stock_swing_d1.ranking.models import (
        CANDIDATE_RANKING_POLICY_ID,
        CANDIDATE_RANKING_POLICY_VERSION,
        RankingPolicyRef,
    )
    from stock_swing_d1.valuation import PortfolioValuationPolicy
    from tests.backtester.conftest import COST_POLICY_PATH

    result, _provider = run_round_trip(
        stop_bar(),
        plans=(candidate_plan(S1, next_session=S2), quiet_plan(S2, next_session=S3)),
    )
    cost_service = BacktestExecutionCostService(
        policy=load_backtest_execution_cost_policy(COST_POLICY_PATH)
    )
    allocation_ref = PORTFOLIO_ALLOCATION_POLICY_REF

    def ref(kind: str) -> ArtifactRef:
        return ArtifactRef(
            artifact_type=kind, schema_version="v0.1", content_sha256="e" * 64
        )

    manifest = build_run_manifest(
        software_revision="c" * 40,
        strategy_configuration_ref=ref("strategy"),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id=allocation_ref.policy_id,
            policy_version=allocation_ref.policy_version,
            policy_fingerprint=allocation_ref.policy_fingerprint,
        ),
        ranking_policy_ref=RankingPolicyRef(
            policy_id=CANDIDATE_RANKING_POLICY_ID,
            policy_version=CANDIDATE_RANKING_POLICY_VERSION,
            policy_fingerprint=CANDIDATE_RANKING_POLICY_FINGERPRINT,
        ),
        execution_cost_policy_ref=cost_service.policy_ref,
        valuation_policy_ref=build_valuation_policy_ref(PortfolioValuationPolicy()),
        universe_artifact_ref=ref("universe"),
        market_data_artifact_ref=MARKET_DATA_REF,
    )
    snapshots = tuple(
        build_valuation_snapshot(session=session.session, marks=())
        for session in result.session_results
    )

    audit = HistoricalBacktestResultService.build(
        run_result=result, run_manifest=manifest, valuation_snapshots=snapshots
    )

    assert audit.audit_summary.audit_passed is True
    assert audit.summary.closed_trade_count == 1
    assert audit.summary.open_trade_count == 0
    (trade,) = audit.trades
    assert trade.entry_session == trade.exit_session == S2
    t_row = audit.session_pnl[1]
    assert t_row.realized_pnl_this_session == trade.realized_pnl
    assert t_row.unrealized_pnl == 0
    assert t_row.period_pnl == trade.realized_pnl
    # Both executions' costs are counted exactly once on T.
    assert t_row.execution_cost_this_session == add_exact_decimal(
        trade.entry_execution_cost, trade.exit_execution_cost
    )
    (equity_before, equity_after) = audit.equity_curve
    assert equity_after.equity == add_exact_decimal(
        equity_before.equity, trade.realized_pnl
    )


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_tampered_buy_applied_evidence_fails_the_phase15d_audit():
    from stock_swing_d1.backtest_results.errors import (
        HistoricalBacktestResultValidationError,
    )

    result, _provider = run_round_trip(stop_bar())
    entry_session = result.session_results[1]
    transition = entry_session.state_transition_result
    buy_row, sell_row = transition.ledger_entries
    tampered_row = buy_row.model_copy(
        update={"settled_cash_delta": buy_row.settled_cash_delta - Decimal("1")}
    )
    tampered_session = entry_session.model_copy(
        update={
            "state_transition_result": transition.model_copy(
                update={"ledger_entries": (tampered_row, sell_row)}
            )
        }
    )
    tampered_run = result.model_copy(
        update={
            "session_results": (
                result.session_results[0],
                tampered_session,
                *result.session_results[2:],
            )
        }
    )

    with pytest.raises(HistoricalBacktestResultValidationError):
        project_closed_trades(tampered_run)


# ---------------------------------------------------------------------------
# Task 5C-B: same entry-session economics on every supported path (test 9)
# ---------------------------------------------------------------------------


def test_legacy_run_and_process_session_apply_the_same_entry_session_economics():
    provider_result, _provider = run_round_trip(
        stop_bar(),
        plans=(candidate_plan(S1, next_session=S2), quiet_plan(S2, next_session=S3)),
    )
    first, second = provider_result.session_results
    s1_plan = candidate_plan(S1, next_session=S2)
    allocation = first.allocation_decision
    legacy_first = HistoricalBacktestSessionInput(
        session=S1,
        decision_time=s1_plan.decision_time,
        next_session=S2,
        indicator_history_bars=s1_plan.indicator_history_bars,
        completed_unadjusted_bars=s1_plan.completed_unadjusted_bars,
        universe_eligible_security_ids=s1_plan.universe_eligible_security_ids,
        allocation_portfolio=PortfolioSnapshot(
            allocation_session=S1,
            decision_time=first.decision_time,
            portfolio_equity=allocation.starting_portfolio_equity,
            cash_available=allocation.starting_cash,
        ),
    )
    legacy_second = HistoricalBacktestSessionInput(
        session=S2,
        decision_time=second.decision_time,
        next_session=S3,
        entry_execution_bars=(stop_bar(),),
    )

    legacy_result = build_orchestrator().run(
        PortfolioState(settled_cash=Decimal("10000")),
        (legacy_first, legacy_second),
        decision_interval=INTERVAL,
    )
    standalone = build_orchestrator().process_session(
        prior_state=first.authoritative_state,
        session_input=legacy_second.model_copy(
            update={"scheduled_entry_intents": first.future_entry_intents}
        ),
    )

    for other in (legacy_result.session_results[1], standalone):
        assert other.entry_session_protective_decisions == (
            second.entry_session_protective_decisions
        )
        assert other.ordered_execution_events == second.ordered_execution_events
        assert other.authoritative_state == second.authoritative_state
    assert legacy_result.final_state_fingerprint == (
        provider_result.final_state_fingerprint
    )


def test_missing_factory_or_evaluator_fails_closed_with_an_executed_entry():
    """Test 10: the frozen typed errors, on every path, before any transition
    of the entry session and with nothing published."""

    base = build_orchestrator()
    kwargs = {
        "baseline_signal_evaluator": base._baseline_signal_evaluator,
        "entry_execution_service": base._entry_execution_service,
        "entry_event_adapter": base._entry_event_adapter,
        "open_position_exit_event_adapter": base._open_position_exit_event_adapter,
    }
    plans = (candidate_plan(S1, next_session=S2), quiet_plan(S2, next_session=S3))
    for missing, code in (
        ("protective_exit_state_factory", "MISSING_PROTECTIVE_EXIT_STATE_FACTORY"),
        ("open_position_exit_evaluator", "MISSING_OPEN_POSITION_EXIT_EVALUATOR"),
    ):
        recording = RecordingTransition()
        configured = dict(kwargs, transition_service=recording)
        if missing == "protective_exit_state_factory":
            configured["open_position_exit_evaluator"] = (
                base._open_position_exit_evaluator
            )
        else:
            configured["protective_exit_state_factory"] = (
                base._protective_exit_state_factory
            )
        orchestrator = HistoricalBacktestOrchestrator(**configured)
        provider = EntryBarProvider(stop_bar())

        with pytest.raises(HistoricalBacktestValidationError, match=code):
            orchestrator.run_with_session_input_provider(
                PortfolioState(settled_cash=Decimal("10000")),
                plans,
                provider=provider,
                decision_interval=INTERVAL,
                market_data_artifact_ref=MARKET_DATA_REF,
            )
        # S1 transitioned (no executed entry there); S2 never did.
        assert [session for session, _events in recording.calls] == [S1]


# ---------------------------------------------------------------------------
# Session-result invariants (Task 5C-B/5C-C, test 42)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_terminal_entry_session_decision_requires_its_applied_round_trip_sell():
    result, _provider = run_round_trip(stop_bar())
    entry_session = result.session_results[1]
    buy, sell = entry_session.ordered_execution_events
    values = {
        name: getattr(entry_session, name)
        for name in HistoricalBacktestSessionResult.model_fields
    }

    # Dropping the SELL presentation breaks the terminal linkage.
    with pytest.raises(ValueError):
        HistoricalBacktestSessionResult.model_validate(
            dict(values, ordered_execution_events=(buy,))
        )
    # Dropping the decision while the round trip exists breaks coverage.
    with pytest.raises(ValueError):
        HistoricalBacktestSessionResult.model_validate(
            dict(values, entry_session_protective_decisions=())
        )
    # A SELL at a price other than the Phase 10 final price is rejected.
    with pytest.raises(ValueError):
        HistoricalBacktestSessionResult.model_validate(
            dict(
                values,
                ordered_execution_events=(
                    buy,
                    sell.model_copy(
                        update={"fill_price": sell.fill_price + Decimal("1")}
                    ),
                ),
            )
        )
    assert HistoricalBacktestSessionResult.model_validate(values) == entry_session


# ---------------------------------------------------------------------------
# Task 5C defects at the Phase 15A boundary (tests 1-4, 7)
# ---------------------------------------------------------------------------


class ForgedMarkProvider(RecordingProvider):
    """Returns a mark forged via model_copy: right class, invalid basis."""

    def allocation_boundary_marks(self, *, context):
        marks = super().allocation_boundary_marks(context=context)
        forged = tuple(
            mark.model_copy(update={"price_basis": "capital_special_adjusted"})
            for mark in marks.marks
        )
        return marks.model_copy(update={"marks": forged})


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_forged_nested_mark_fails_before_valuation_phase12_and_custody(
    monkeypatch,
):
    valuation_calls = []
    original = orchestration_module.value_portfolio_at_allocation_boundary

    def recording_valuation(**kwargs):
        valuation_calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(
        orchestration_module,
        "value_portfolio_at_allocation_boundary",
        recording_valuation,
    )
    allocation_service = RecordingAllocationService()
    provider = ForgedMarkProvider()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="INVALID_ALLOCATION_BOUNDARY_MARKS",
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

    returned = provider.allocation_boundary_marks(context=provider.mark_contexts[-1])
    assert type(returned.marks[0]) is PortfolioValuationMark
    assert returned.marks[0].price_basis == "capital_special_adjusted"
    assert valuation_calls == []
    assert allocation_service.calls == []


def test_phase14_failure_prevents_mark_valuation_and_phase12_calls(monkeypatch):
    class FailingRanking:
        def rank_candidates(self, **kwargs):
            raise RuntimeError("deliberate Phase 14 failure")

    valuation_calls = []
    monkeypatch.setattr(
        orchestration_module,
        "value_portfolio_at_allocation_boundary",
        lambda **kwargs: valuation_calls.append(kwargs),
    )
    allocation_service = RecordingAllocationService()
    provider = RecordingProvider()
    orchestrator = HistoricalBacktestOrchestrator(
        **{
            name: getattr(build_orchestrator(), f"_{name}")
            for name in (
                "baseline_signal_evaluator",
                "entry_execution_service",
                "entry_event_adapter",
                "open_position_exit_evaluator",
                "open_position_exit_event_adapter",
                "protective_exit_state_factory",
            )
        },
        ranking_service=FailingRanking(),
        allocation_service=allocation_service,
    )

    with pytest.raises(RuntimeError, match="deliberate Phase 14 failure"):
        orchestrator.run_with_session_input_provider(
            carried_position_state(),
            (candidate_plan(S2, next_session=S3),),
            provider=provider,
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
            initial_protective_states=(carried_protective_state(),),
        )

    assert not any(
        call[0] == "allocation_boundary_marks" for call in provider.calls
    )
    assert valuation_calls == []
    assert allocation_service.calls == []


def test_marks_are_retrieved_only_after_phase14_ranking_succeeds(monkeypatch):
    order = []

    class RecordingRanking:
        def __init__(self) -> None:
            from stock_swing_d1.ranking import CandidateRankingService

            self.delegate = CandidateRankingService

        def rank_candidates(self, **kwargs):
            order.append("rank")
            return self.delegate.rank_candidates(**kwargs)

    class OrderedProvider(RecordingProvider):
        def allocation_boundary_marks(self, *, context):
            order.append("marks")
            return super().allocation_boundary_marks(context=context)

    original = orchestration_module.value_portfolio_at_allocation_boundary

    def recording_valuation(**kwargs):
        order.append("valuation")
        return original(**kwargs)

    monkeypatch.setattr(
        orchestration_module,
        "value_portfolio_at_allocation_boundary",
        recording_valuation,
    )

    class OrderedAllocation(RecordingAllocationService):
        def allocate_ranked_candidates(self, **kwargs):
            order.append("phase12")
            return super().allocate_ranked_candidates(**kwargs)

    orchestrator = HistoricalBacktestOrchestrator(
        **{
            name: getattr(build_orchestrator(), f"_{name}")
            for name in (
                "baseline_signal_evaluator",
                "entry_execution_service",
                "entry_event_adapter",
                "open_position_exit_evaluator",
                "open_position_exit_event_adapter",
                "protective_exit_state_factory",
            )
        },
        ranking_service=RecordingRanking(),
        allocation_service=OrderedAllocation(),
    )
    orchestrator.run_with_session_input_provider(
        carried_position_state(),
        (candidate_plan(S2, next_session=S3),),
        provider=OrderedProvider(),
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
        initial_protective_states=(carried_protective_state(),),
    )

    assert order == ["rank", "marks", "valuation", "phase12"]


def test_provider_none_is_rejected_immediately():
    recording = RecordingTransition()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="INVALID_SESSION_INPUT_PROVIDER",
    ):
        build_orchestrator(
            transition_service=recording
        ).run_with_session_input_provider(
            PortfolioState(settled_cash=Decimal("10000")),
            (candidate_plan(S1, next_session=S2),),
            provider=None,
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
        )

    assert recording.calls == []
