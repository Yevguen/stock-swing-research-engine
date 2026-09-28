from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.backtest_results import (
    ArtifactRef,
    HistoricalBacktestAuditSummary,
    HistoricalBacktestContentFingerprints,
    HistoricalBacktestCostSummary,
    HistoricalBacktestExitReasonRow,
    HistoricalBacktestExitReasonSummary,
    HistoricalBacktestSummary,
    HistoricalBacktestValuationPolicy,
    PolicyArtifactRef,
    build_run_manifest,
    build_valuation_policy_ref,
)
from stock_swing_d1.backtester import (
    BacktestBuyExecutionEventAdapter,
    HistoricalDecisionInterval,
    HistoricalBacktestOrchestrator,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionInput,
    HistoricalBacktestSessionResult,
)
from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.execution.costs.models import (
    BROKER_NEUTRAL_POLICY_ID,
    ExecutionCostPolicyRef,
)
from stock_swing_d1.execution.costs import (
    BacktestExecutionCostService,
    ExecutionIdentifierService,
    load_backtest_execution_cost_policy,
)
from stock_swing_d1.execution.entry import (
    EntryExecutionService,
    create_pending_entry,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.portfolio import (
    PORTFOLIO_ALLOCATION_POLICY_REF,
    PortfolioCandidate,
    PortfolioSnapshot,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from stock_swing_d1.ranking.hashing import (
    CANDIDATE_RANKING_POLICY_FINGERPRINT,
)
from stock_swing_d1.ranking.models import (
    CANDIDATE_RANKING_POLICY_ID,
    CANDIDATE_RANKING_POLICY_VERSION,
    RankingPolicyRef,
)
from stock_swing_d1.ranking import (
    RankingCandidate,
    compute_candidate_input_fingerprint,
    rank_candidates,
)
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalDecision,
)


SOURCE_SESSION = date(2026, 8, 21)
SOURCE_ENTRY_SESSION = date(2026, 8, 24)
SOURCE_DECISION_INTERVAL = HistoricalDecisionInterval(
    decision_start_date=date(2020, 1, 1),
    decision_end_date=date(2030, 12, 31),
)
SOURCE_COST_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "costs.yaml"
)


def source_decision_time(session: date) -> datetime:
    return datetime.combine(session, time(20), tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class SourceCalendar:
    def next_session(self, session: date) -> date:
        if session == SOURCE_SESSION:
            return SOURCE_ENTRY_SESSION
        return session.fromordinal(session.toordinal() + 1)

    def previous_session(self, session: date) -> date:
        if session == SOURCE_ENTRY_SESSION:
            return SOURCE_SESSION
        return session.fromordinal(session.toordinal() - 1)

    def session_distance(self, start: date, end: date) -> int:
        distance = 0
        cursor = start
        while cursor < end:
            cursor = self.next_session(cursor)
            distance += 1
        if cursor != end:
            raise ValueError("end is not a canonical session")
        return distance

    def decision_time(self, session: date) -> datetime:
        return source_decision_time(session)

    def regular_session_open_time(self, session: date) -> datetime:
        return datetime.combine(session, time(13, 30), tzinfo=timezone.utc)


def source_protective_collaborators() -> dict[str, object]:
    """The frozen Phase 10/15B owners every executed entry now requires.

    Task 5C-B: an entry executed on T receives its Phase 10 state and its
    Phase 15B entry-session evaluation inside the shared session body, on
    every path, so a source run with executed entries must configure both.
    """

    from stock_swing_d1.execution.open_position_exit import (
        OpenPositionExitEvaluator,
    )
    from stock_swing_d1.execution.protective_exit import ProtectiveExitService

    calendar = SourceCalendar()
    return {
        "protective_exit_state_factory": ProtectiveExitService(
            trading_calendar=calendar
        ),
        "open_position_exit_evaluator": OpenPositionExitEvaluator(
            trading_calendar=calendar
        ),
    }


class SourceEarningsOverlay:
    def revalidate_pending_entry(self, **_kwargs):
        return EarningsIntegrationDecision(
            action=EarningsIntegrationAction.PENDING_ENTRY_ALLOWED,
            earnings_state=None,
            risk_decision=None,
        )


def _source_candidate(
    security_id: str,
) -> tuple[BaselineSignalDecision, RankingCandidate, PortfolioCandidate]:
    symbol = security_id.replace("NORGATE:", "S")
    signal_time = source_decision_time(SOURCE_SESSION)
    earnings = EarningsIntegrationDecision(
        action=EarningsIntegrationAction.ENTRY_ALLOWED,
        earnings_state=None,
        risk_decision=None,
    )
    signal = BaselineSignalDecision(
        security_id=security_id,
        symbol=symbol,
        signal_session=SOURCE_SESSION,
        signal_time=signal_time,
        planned_entry_session=SOURCE_ENTRY_SESSION,
        adjusted_close=100.0,
        sma_20=110.0,
        sma_50=100.0,
        rsi_14=60.0,
        atr_14=5.0,
        atr_fraction=0.05,
        universe_eligible=True,
        close_above_sma50=True,
        sma20_above_sma50=True,
        rsi_above_50=True,
        atr_above_minimum=True,
        earnings_entry_allowed=True,
        earnings_action=EarningsIntegrationAction.ENTRY_ALLOWED,
        earnings_reason=None,
        earnings_decision=earnings,
        action=BaselineSignalAction.VALID_LONG_SIGNAL,
    )
    pending = create_pending_entry(signal, trading_calendar=SourceCalendar())
    bar = StockBar(
        security_id=security_id,
        symbol=symbol,
        trading_date=SOURCE_SESSION,
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
    allocation_candidate = PortfolioCandidate(
        signal=signal,
        pending_entry=pending,
        signal_bar=bar,
    )
    ranking_values = {
        "security_id": security_id,
        "ranking_session": SOURCE_SESSION,
        "decision_time": signal_time,
        "signal_session": SOURCE_SESSION,
        "signal_time": signal_time,
        "sma20": 110.0,
        "sma50": 100.0,
        "atr14": 5.0,
        "rsi14": 60.0,
    }
    ranking_candidate = RankingCandidate(
        **ranking_values,
        input_fingerprint=compute_candidate_input_fingerprint(
            **ranking_values
        ),
    )
    return signal, ranking_candidate, allocation_candidate


def _entry_bar(security_id: str, *, opening_price: float) -> StockBar:
    return StockBar(
        security_id=security_id,
        symbol=security_id.replace("NORGATE:", "S"),
        trading_date=SOURCE_ENTRY_SESSION,
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


@pytest.fixture
def artifact_ref_factory() -> Callable[..., ArtifactRef]:
    def build(
        artifact_type: str = "market_data",
        *,
        schema_version: str = "artifact.v0.1",
        digest: str = "a" * 64,
        build_id: str | None = None,
    ) -> ArtifactRef:
        return ArtifactRef(
            artifact_type=artifact_type,
            schema_version=schema_version,
            content_sha256=digest,
            build_id=build_id,
        )

    return build


@pytest.fixture
def valuation_policy_ref():
    return build_valuation_policy_ref(HistoricalBacktestValuationPolicy())


@pytest.fixture
def ranking_policy_ref() -> RankingPolicyRef:
    return RankingPolicyRef(
        policy_id=CANDIDATE_RANKING_POLICY_ID,
        policy_version=CANDIDATE_RANKING_POLICY_VERSION,
        policy_fingerprint=CANDIDATE_RANKING_POLICY_FINGERPRINT,
    )


@pytest.fixture
def execution_cost_policy_ref() -> ExecutionCostPolicyRef:
    return ExecutionCostPolicyRef(
        policy_id=BROKER_NEUTRAL_POLICY_ID,
        policy_fingerprint="b" * 64,
    )


@pytest.fixture
def run_manifest(
    artifact_ref_factory,
    valuation_policy_ref,
    ranking_policy_ref,
    execution_cost_policy_ref,
):
    return build_run_manifest(
        software_revision="c" * 40,
        strategy_configuration_ref=artifact_ref_factory("strategy"),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id="allocation_v0.1",
            policy_version="0.1",
            policy_fingerprint="d" * 64,
        ),
        ranking_policy_ref=ranking_policy_ref,
        execution_cost_policy_ref=execution_cost_policy_ref,
        valuation_policy_ref=valuation_policy_ref,
        universe_artifact_ref=artifact_ref_factory("universe"),
        market_data_artifact_ref=artifact_ref_factory("market_data"),
    )


@pytest.fixture
def source_run_bundle(
    artifact_ref_factory,
    valuation_policy_ref,
    ranking_policy_ref,
):
    source_values = tuple(
        _source_candidate(f"NORGATE:{number}")
        for number in range(101, 107)
    )
    signals = tuple(value[0] for value in source_values)
    ranking_candidates = tuple(value[1] for value in source_values)
    allocation_candidates = tuple(value[2] for value in source_values)
    cost_service = BacktestExecutionCostService(
        policy=load_backtest_execution_cost_policy(SOURCE_COST_POLICY_PATH)
    )
    entry_service = EntryExecutionService(
        earnings_overlay=SourceEarningsOverlay(),
        trading_calendar=SourceCalendar(),
        execution_cost_service=cost_service,
    )
    orchestrator = HistoricalBacktestOrchestrator(
        entry_execution_service=entry_service,
        entry_event_adapter=BacktestBuyExecutionEventAdapter(
            execution_cost_service=cost_service,
            identifier_service=ExecutionIdentifierService(),
        ),
        **source_protective_collaborators(),
    )
    run = orchestrator.run(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            HistoricalBacktestSessionInput(
                session=SOURCE_SESSION,
                decision_time=source_decision_time(SOURCE_SESSION),
                next_session=SOURCE_ENTRY_SESSION,
                ranking_candidates=ranking_candidates,
                allocation_candidates=allocation_candidates,
                allocation_portfolio=PortfolioSnapshot(
                    allocation_session=SOURCE_SESSION,
                    decision_time=source_decision_time(SOURCE_SESSION),
                    portfolio_equity=10_000.0,
                    cash_available=10_000.0,
                ),
            ),
            HistoricalBacktestSessionInput(
                session=SOURCE_ENTRY_SESSION,
                decision_time=source_decision_time(SOURCE_ENTRY_SESSION),
                entry_execution_bars=(
                    _entry_bar("NORGATE:101", opening_price=100.0),
                    _entry_bar("NORGATE:102", opening_price=600.0),
                ),
            ),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )
    first_session = run.session_results[0].model_copy(
        update={"signal_decisions": signals}
    )
    run = run.model_copy(
        update={
            "session_results": (first_session, *run.session_results[1:]),
        }
    )
    allocation_ref = PORTFOLIO_ALLOCATION_POLICY_REF
    manifest = build_run_manifest(
        software_revision="c" * 40,
        strategy_configuration_ref=artifact_ref_factory("strategy"),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id=allocation_ref.policy_id,
            policy_version=allocation_ref.policy_version,
            policy_fingerprint=allocation_ref.policy_fingerprint,
        ),
        ranking_policy_ref=ranking_policy_ref,
        execution_cost_policy_ref=cost_service.policy_ref,
        valuation_policy_ref=valuation_policy_ref,
        universe_artifact_ref=artifact_ref_factory("universe"),
        market_data_artifact_ref=artifact_ref_factory("market_data"),
    )
    return run, manifest


@pytest.fixture
def zero_allocation_run():
    return HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        (
            HistoricalBacktestSessionInput(
                session=SOURCE_SESSION,
                decision_time=source_decision_time(SOURCE_SESSION),
                next_session=SOURCE_ENTRY_SESSION,
                ranking_candidates=(),
                allocation_candidates=(),
                allocation_portfolio=PortfolioSnapshot(
                    allocation_session=SOURCE_SESSION,
                    decision_time=source_decision_time(SOURCE_SESSION),
                    portfolio_equity=1_000.0,
                    cash_available=1_000.0,
                ),
            ),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


@pytest.fixture
def same_session_sell_allocation_run():
    prior_session = SOURCE_SESSION.fromordinal(SOURCE_SESSION.toordinal() - 1)
    buy = PortfolioExecutionEvent(
        execution_id="BOUNDARY-BUY",
        source_order_id="BOUNDARY-BUY-ORDER",
        session=prior_session,
        asset_id="NORGATE:901",
        side=ExecutionSide.BUY,
        quantity=2,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    initial = PortfolioTransitionEngine.transition(
        PortfolioState(settled_cash=Decimal("1000")),
        prior_session,
        (buy,),
    ).resulting_state
    sell = PortfolioExecutionEvent(
        execution_id="BOUNDARY-SELL",
        source_order_id="BOUNDARY-SELL-ORDER",
        session=SOURCE_SESSION,
        asset_id="NORGATE:901",
        side=ExecutionSide.SELL,
        quantity=2,
        fill_price=Decimal("110"),
        execution_cost=Decimal("1"),
        settlement_id="BOUNDARY-SETTLEMENT",
        settlement_session=SOURCE_ENTRY_SESSION,
    )
    _signal, ranking_candidate, allocation_candidate = _source_candidate(
        "NORGATE:201"
    )
    return HistoricalBacktestOrchestrator().run(
        initial,
        (
            HistoricalBacktestSessionInput(
                session=SOURCE_SESSION,
                decision_time=source_decision_time(SOURCE_SESSION),
                next_session=SOURCE_ENTRY_SESSION,
                scheduled_execution_events=(sell,),
                ranking_candidates=(ranking_candidate,),
                allocation_candidates=(allocation_candidate,),
                allocation_portfolio=PortfolioSnapshot(
                    allocation_session=SOURCE_SESSION,
                    decision_time=source_decision_time(SOURCE_SESSION),
                    portfolio_equity=1_000.0,
                    cash_available=float(initial.settled_cash),
                ),
            ),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


@pytest.fixture
def same_session_sell_run_manifest(
    artifact_ref_factory,
    valuation_policy_ref,
    ranking_policy_ref,
    execution_cost_policy_ref,
):
    allocation_ref = PORTFOLIO_ALLOCATION_POLICY_REF
    return build_run_manifest(
        software_revision="c" * 40,
        strategy_configuration_ref=artifact_ref_factory("strategy"),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id=allocation_ref.policy_id,
            policy_version=allocation_ref.policy_version,
            policy_fingerprint=allocation_ref.policy_fingerprint,
        ),
        ranking_policy_ref=ranking_policy_ref,
        execution_cost_policy_ref=execution_cost_policy_ref,
        valuation_policy_ref=valuation_policy_ref,
        universe_artifact_ref=artifact_ref_factory("universe"),
        market_data_artifact_ref=artifact_ref_factory("market_data"),
    )


@pytest.fixture
def generated_exit_run():
    from stock_swing_d1.execution.open_position_exit import (
        OpenPositionExitEvaluator,
    )
    from tests.backtester.test_phase15b_exit_integration import (
        ExitEventAdapter,
        exit_evaluation,
        session_input,
        state_with_positions,
    )
    from tests.execution.open_position_exit.conftest import (
        ExplicitTradingCalendar,
    )

    initial = state_with_positions("NORGATE:1001")
    orchestrator = HistoricalBacktestOrchestrator(
        open_position_exit_evaluator=OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        ),
        open_position_exit_event_adapter=ExitEventAdapter(),
    )
    return orchestrator.run(
        initial,
        (
            session_input(
                open_position_exit_evaluations=(
                    exit_evaluation(low=95.0),
                )
            ),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


@pytest.fixture
def external_execution_run():
    event = PortfolioExecutionEvent(
        execution_id="EXTERNAL-BUY-1",
        source_order_id="EXTERNAL-ORDER-1",
        session=SOURCE_SESSION,
        asset_id="NORGATE:901",
        side=ExecutionSide.BUY,
        quantity=2,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    return HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("1000")),
        (
            HistoricalBacktestSessionInput(
                session=SOURCE_SESSION,
                decision_time=source_decision_time(SOURCE_SESSION),
                scheduled_execution_events=(event,),
            ),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


@pytest.fixture
def replay_execution_run():
    event = PortfolioExecutionEvent(
        execution_id="REPLAY-BUY-1",
        source_order_id="REPLAY-ORDER-1",
        session=SOURCE_SESSION,
        asset_id="NORGATE:902",
        side=ExecutionSide.BUY,
        quantity=2,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    virgin = PortfolioState(settled_cash=Decimal("1000"))
    initial = PortfolioTransitionEngine.transition(
        virgin, SOURCE_SESSION, (event,)
    ).resulting_state
    transition = PortfolioTransitionEngine.transition(
        initial, SOURCE_ENTRY_SESSION, (event,)
    )
    snapshot = rank_candidates(
        ranking_session=SOURCE_ENTRY_SESSION,
        decision_time=source_decision_time(SOURCE_ENTRY_SESSION),
        candidates=(),
    )
    session = HistoricalBacktestSessionResult(
        session=SOURCE_ENTRY_SESSION,
        decision_time=source_decision_time(SOURCE_ENTRY_SESSION),
        prior_state_fingerprint=hash_portfolio_state(initial),
        ordered_execution_events=(event,),
        state_transition_result=transition,
        authoritative_state=transition.resulting_state,
        ranking_snapshot=snapshot,
    )
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=initial,
        final_state=transition.resulting_state,
        session_results=(session,),
        initial_state_fingerprint=hash_portfolio_state(initial),
        final_state_fingerprint=hash_portfolio_state(
            transition.resulting_state
        ),
    )


@pytest.fixture
def zero_cost_summary() -> HistoricalBacktestCostSummary:
    return HistoricalBacktestCostSummary(
        buy_execution_cost_total=Decimal("0"),
        sell_execution_cost_total=Decimal("0"),
        total_execution_cost=Decimal("0"),
        applied_buy_count=0,
        applied_sell_count=0,
    )


@pytest.fixture
def zero_exit_reason_summary() -> HistoricalBacktestExitReasonSummary:
    from stock_swing_d1.backtest_results import HistoricalExitReason

    return HistoricalBacktestExitReasonSummary(
        rows=tuple(
            HistoricalBacktestExitReasonRow(
                reason=reason,
                exit_count=0,
                realized_pnl=Decimal("0"),
            )
            for reason in HistoricalExitReason
        )
    )


@pytest.fixture
def zero_summary() -> HistoricalBacktestSummary:
    return HistoricalBacktestSummary(
        processed_session_count=0,
        entry_count=0,
        exit_count=0,
        closed_trade_count=0,
        open_trade_count=0,
        allocation_rejection_count=0,
        entry_execution_rejection_count=0,
        gross_realized_pnl=Decimal("0"),
        final_unrealized_pnl=Decimal("0"),
        initial_equity=Decimal("1000"),
        final_equity=Decimal("1000"),
        period_pnl=Decimal("0"),
        buy_execution_cost_total=Decimal("0"),
        sell_execution_cost_total=Decimal("0"),
        total_execution_cost=Decimal("0"),
    )


@pytest.fixture
def passing_audit_summary() -> HistoricalBacktestAuditSummary:
    return HistoricalBacktestAuditSummary(
        source_run_canonical=True,
        state_chain_valid=True,
        ledger_reconstruction_valid=True,
        execution_ledger_provenance_valid=True,
        signal_provenance_valid=True,
        ranking_provenance_valid=True,
        allocation_provenance_valid=True,
        execution_provenance_valid=True,
        trade_linkage_valid=True,
        valuation_coverage_valid=True,
        equity_reconciliation_valid=True,
        pnl_reconciliation_valid=True,
        cost_reconciliation_valid=True,
        policy_consistency_valid=True,
        artifact_fingerprints_valid=True,
        audit_passed=True,
    )


@pytest.fixture
def content_fingerprints() -> HistoricalBacktestContentFingerprints:
    return HistoricalBacktestContentFingerprints(
        **{
            field_name: f"{index % 16:x}" * 64
            for index, field_name in enumerate(
                HistoricalBacktestContentFingerprints.model_fields,
                start=1,
            )
        }
    )
