from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    ArtifactRef,
    DividendAttributionCompleteness,
    HistoricalBacktestAuditResult,
    HistoricalBacktestAuditSummary,
    HistoricalBacktestContentFingerprints,
    HistoricalBacktestCostSummary,
    HistoricalBacktestEquityRow,
    HistoricalBacktestExitReasonRow,
    HistoricalBacktestExitReasonSummary,
    HistoricalBacktestSummary,
    HistoricalBacktestTransitionAudit,
    HistoricalClosedTradeRecord,
    HistoricalExitReason,
    HistoricalOpenTradeRecord,
    PolicyArtifactRef,
    build_run_manifest,
    build_valuation_policy_ref,
    compute_content_fingerprint,
    compute_result_fingerprint,
    compute_source_run_fingerprint,
    HistoricalBacktestValuationPolicy,
)
from stock_swing_d1.backtester import (
    HistoricalBacktestRunResult,
    HistoricalDecisionInterval,
)
from stock_swing_d1.execution.costs.models import (
    BROKER_NEUTRAL_POLICY_ID,
    ExecutionCostPolicyRef,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.ranking.models import RankingPolicyRef
from stock_swing_d1.research_metrics import (
    BenchmarkPerformanceMetrics,
    BenchmarkPerformanceObservation,
    BenchmarkSourceArtifactRef,
    RelativePerformanceMetrics,
    StrategyPerformanceMetrics,
    TradePerformanceMetrics,
    build_benchmark_performance_series,
    build_performance_measurement_policy,
    defined_metric,
)


@pytest.fixture
def performance_policy():
    return build_performance_measurement_policy()


@pytest.fixture
def strategy_metrics():
    return StrategyPerformanceMetrics(
        ending_equity=Decimal("110000"),
        net_pnl=Decimal("10000"),
        total_return=defined_metric(Decimal("0.100000000000000000")),
        cagr=defined_metric(Decimal("0.100360090511656501")),
        annualized_volatility=defined_metric(Decimal("1.469495805366663448")),
        maximum_drawdown=defined_metric(Decimal("0.019801980198019802")),
        sharpe_ratio=defined_metric(Decimal("7.829182263081608445")),
        sortino_ratio=defined_metric(Decimal("51.759593850372855834")),
    )


@pytest.fixture
def trade_metrics():
    return TradePerformanceMetrics(
        completed_trade_count=2,
        win_count=1,
        loss_count=1,
        breakeven_count=0,
        gross_profit=Decimal("205"),
        gross_loss=Decimal("100"),
        win_rate=defined_metric(Decimal("0.500000000000000000")),
        loss_rate=defined_metric(Decimal("0.500000000000000000")),
        breakeven_rate=defined_metric(Decimal("0E-18")),
        profit_factor=defined_metric(Decimal("2.050000000000000000")),
        expectancy=defined_metric(Decimal("52.500000000000000000")),
        average_winner_usd=defined_metric(Decimal("205.000000000000000000")),
        average_loser_usd=defined_metric(Decimal("-100.000000000000000000")),
        worst_trade_usd=defined_metric(Decimal("-100.000000000000000000")),
        average_holding_sessions=defined_metric(
            Decimal("2.000000000000000000")
        ),
    )


@pytest.fixture
def benchmark_metrics():
    return BenchmarkPerformanceMetrics(
        ending_equity=Decimal("104000"),
        total_return=defined_metric(Decimal("0.040000000000000000")),
        cagr=defined_metric(Decimal("0.040140083409008749")),
        annualized_volatility=defined_metric(Decimal("0.447809555483366529")),
        maximum_drawdown=defined_metric(Decimal("0.002487562189054726")),
        sharpe_ratio=defined_metric(Decimal("2.5")),
        sortino_ratio=defined_metric(Decimal("3.5")),
    )


@pytest.fixture
def relative_metrics():
    return RelativePerformanceMetrics(
        strategy_minus_benchmark_total_return=defined_metric(
            Decimal("0.060000000000000000")
        ),
        strategy_minus_benchmark_cagr=defined_metric(
            Decimal("0.060220007102647752")
        ),
        ending_wealth_ratio=defined_metric(Decimal("1.057692307692307692")),
    )


@pytest.fixture
def benchmark_source_ref():
    return BenchmarkSourceArtifactRef(
        artifact_id="synthetic-benchmark-evidence",
        schema_version="synthetic-index.v0.1",
        content_sha256="a" * 64,
        build_id="synthetic-fixture",
    )


@pytest.fixture
def benchmark_series(benchmark_source_ref):
    """Synthetic evidence of one primary authoritative benchmark-equity path."""

    return build_benchmark_performance_series(
        benchmark_id="SYNTHETIC_PRIMARY_BENCHMARK_V0_1",
        source_artifact_ref=benchmark_source_ref,
        observations=(
            BenchmarkPerformanceObservation(
                session=date(2025, 1, 2), value=Decimal("100000")
            ),
            BenchmarkPerformanceObservation(
                session=date(2025, 1, 3), value=Decimal("101250.75")
            ),
        ),
    )


_SYNTHETIC_DECISION_TIME = datetime(2025, 1, 2, 21, 0, tzinfo=timezone.utc)
_SYNTHETIC_SECURITY_ID = "NORGATE:1"


def _synthetic_run_manifest():
    return build_run_manifest(
        software_revision="1" * 40,
        strategy_configuration_ref=ArtifactRef(
            artifact_type="synthetic_strategy",
            schema_version="synthetic.v0.1",
            content_sha256="2" * 64,
            build_id="synthetic-fixture",
        ),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id="synthetic-allocation-policy",
            policy_version="0.1",
            policy_fingerprint="3" * 64,
        ),
        ranking_policy_ref=RankingPolicyRef(
            policy_id="synthetic-ranking-policy",
            policy_version="0.1",
            policy_fingerprint="4" * 64,
        ),
        execution_cost_policy_ref=ExecutionCostPolicyRef(
            policy_id=BROKER_NEUTRAL_POLICY_ID,
            policy_fingerprint="5" * 64,
        ),
        valuation_policy_ref=build_valuation_policy_ref(
            HistoricalBacktestValuationPolicy()
        ),
        universe_artifact_ref=ArtifactRef(
            artifact_type="synthetic_universe",
            schema_version="synthetic.v0.1",
            content_sha256="6" * 64,
            build_id="synthetic-fixture",
        ),
        market_data_artifact_ref=ArtifactRef(
            artifact_type="synthetic_market_data",
            schema_version="synthetic.v0.1",
            content_sha256="7" * 64,
            build_id="synthetic-fixture",
        ),
    )


def make_closed_trade(
    *,
    trade_id: str,
    entry_session: date,
    exit_session: date,
    entry_fill_price: Decimal,
    exit_fill_price: Decimal,
    quantity: int = 100,
    ordinary_dividend_income: Decimal = Decimal("0"),
    dividend_aware: bool = True,
    carried_in: bool = False,
) -> HistoricalClosedTradeRecord:
    """One authoritative completed trade episode with the frozen identities.

    ``dividend_aware=False`` produces the legacy v0.1 record, which publishes
    no authoritative ``trade_total_pnl``; ``carried_in=True`` produces the v0.3
    PARTIAL_PRE_RUN_UNKNOWN record, which likewise cannot publish one because
    its pre-run dividend component is unknown upstream.
    """

    entry_cost_basis = entry_fill_price * quantity
    gross_exit_proceeds = exit_fill_price * quantity
    realized_pnl = gross_exit_proceeds - entry_cost_basis
    shared = {
        "trade_id": trade_id,
        "security_id": _SYNTHETIC_SECURITY_ID,
        "quantity": quantity,
        "carried_in": carried_in,
        "entry_execution_id": trade_id,
        "entry_session": entry_session,
        "entry_fill_price": entry_fill_price,
        "entry_execution_cost": Decimal("0"),
        "entry_cost_basis": entry_cost_basis,
        "exit_execution_id": trade_id + "-exit",
        "exit_session": exit_session,
        "exit_fill_price": exit_fill_price,
        "exit_execution_cost": Decimal("0"),
        "exit_reason": HistoricalExitReason.TAKE_PROFIT,
        "gross_exit_proceeds": gross_exit_proceeds,
        "net_exit_proceeds": gross_exit_proceeds,
        "realized_pnl": realized_pnl,
    }
    if not dividend_aware:
        return HistoricalClosedTradeRecord(
            schema_version="historical_closed_trade.v0.1", **shared
        )
    completeness = (
        DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
        if carried_in
        else DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
    )
    return HistoricalClosedTradeRecord(
        schema_version="historical_closed_trade.v0.3",
        ordinary_dividend_income=ordinary_dividend_income,
        ordinary_dividend_event_count=(
            0 if ordinary_dividend_income == Decimal("0") else 1
        ),
        dividend_attribution_completeness=completeness,
        ordinary_dividend_attribution_fingerprint="8" * 64,
        trade_total_pnl=(
            None if carried_in else realized_pnl + ordinary_dividend_income
        ),
        **shared,
    )


def make_open_trade(
    *,
    trade_id: str,
    entry_session: date,
    final_mark_session: date,
    entry_fill_price: Decimal = Decimal("10"),
    final_mark_price: Decimal = Decimal("12"),
    quantity: int = 100,
) -> HistoricalOpenTradeRecord:
    """One position still open at the run end, excluded from trade statistics."""

    return HistoricalOpenTradeRecord(
        trade_id=trade_id,
        security_id=_SYNTHETIC_SECURITY_ID,
        quantity=quantity,
        carried_in=False,
        entry_execution_id=trade_id,
        entry_session=entry_session,
        entry_fill_price=entry_fill_price,
        entry_execution_cost=Decimal("0"),
        entry_cost_basis=entry_fill_price * quantity,
        final_mark_session=final_mark_session,
        final_mark_price=final_mark_price,
        final_market_value=final_mark_price * quantity,
        unrealized_pnl=(final_mark_price - entry_fill_price) * quantity,
    )


def build_synthetic_audit_result(
    *,
    decision_start: date = date(2025, 1, 1),
    decision_end: date = date(2025, 12, 31),
    equity_observations: tuple[tuple[date, Decimal], ...] = (),
    transition_sessions: tuple[date, ...] | None = None,
    initial_equity: Decimal = Decimal("1000"),
    final_equity: Decimal | None = None,
    trades: tuple = (),
) -> HistoricalBacktestAuditResult:
    """Build one explicitly synthetic, self-consistent Phase 15D audit result.

    ``equity_observations`` and ``transition_sessions`` are supplied
    independently on purpose: the Phase 15D model itself never binds equity-row
    count or ordering to the authoritative processed-session sequence, so a
    fixture can express exactly the coverage defects Phase 16B.2 must reject.
    """

    state = PortfolioState(settled_cash=Decimal("1000"))
    state_fingerprint = hash_portfolio_state(state)
    decision_interval = HistoricalDecisionInterval(
        decision_start_date=decision_start,
        decision_end_date=decision_end,
    )
    source_run = HistoricalBacktestRunResult(
        decision_interval=decision_interval,
        initial_state=state,
        final_state=state,
        session_results=(),
        initial_state_fingerprint=state_fingerprint,
        final_state_fingerprint=state_fingerprint,
    )
    if final_equity is None:
        final_equity = (
            equity_observations[-1][1] if equity_observations else initial_equity
        )
    if transition_sessions is None:
        transition_sessions = tuple(
            session for session, _ in equity_observations
        )

    equity_curve = tuple(
        HistoricalBacktestEquityRow(
            session=session,
            state_hash=state_fingerprint,
            settled_cash=equity,
            pending_receivable_value=Decimal("0"),
            open_position_market_value=Decimal("0"),
            equity=equity,
            realized_pnl_this_session=Decimal("0"),
            cumulative_realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            execution_cost_this_session=Decimal("0"),
            cumulative_execution_cost=Decimal("0"),
            period_pnl=equity - initial_equity,
        )
        for session, equity in equity_observations
    )
    session_transitions = tuple(
        HistoricalBacktestTransitionAudit(
            session=session,
            decision_time=_SYNTHETIC_DECISION_TIME,
            state_hash_before=state_fingerprint,
            state_hash_after=state_fingerprint,
            state_version_before=0,
            state_version_after=1,
            settled_cash_before=Decimal("1000"),
            settled_cash_after=Decimal("1000"),
            open_position_count_before=0,
            open_position_count_after=0,
            pending_settlement_count_before=0,
            pending_settlement_count_after=0,
            ledger_entry_count=0,
        )
        for session in transition_sessions
    )

    cost_summary = HistoricalBacktestCostSummary(
        buy_execution_cost_total=Decimal("0"),
        sell_execution_cost_total=Decimal("0"),
        total_execution_cost=Decimal("0"),
        applied_buy_count=0,
        applied_sell_count=0,
    )
    exit_reason_summary = HistoricalBacktestExitReasonSummary(
        rows=tuple(
            HistoricalBacktestExitReasonRow(
                reason=reason,
                exit_count=0,
                realized_pnl=Decimal("0"),
            )
            for reason in HistoricalExitReason
        )
    )
    closed_trade_count = sum(
        1 for trade in trades if type(trade) is HistoricalClosedTradeRecord
    )
    summary = HistoricalBacktestSummary(
        processed_session_count=len(session_transitions),
        entry_count=len(trades),
        exit_count=closed_trade_count,
        closed_trade_count=closed_trade_count,
        open_trade_count=len(trades) - closed_trade_count,
        allocation_rejection_count=0,
        entry_execution_rejection_count=0,
        gross_realized_pnl=Decimal("0"),
        final_unrealized_pnl=Decimal("0"),
        initial_equity=initial_equity,
        final_equity=final_equity,
        period_pnl=final_equity - initial_equity,
        buy_execution_cost_total=Decimal("0"),
        sell_execution_cost_total=Decimal("0"),
        total_execution_cost=Decimal("0"),
    )
    audit_summary = HistoricalBacktestAuditSummary(
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

    populated = {
        "session_transitions": session_transitions,
        "equity_curve": equity_curve,
        "trades": trades,
    }
    empty_artifacts = (
        "signal_provenance",
        "ranking_cycles",
        "ranking_candidates",
        "allocation_cycles",
        "allocation_candidates",
        "execution_provenance",
        "entries",
        "exits",
        "rejections",
        "cash_ledger",
        "settlement_ledger",
        "session_pnl",
    )
    content_values = {
        name: compute_content_fingerprint(
            artifact_name=name, rows_or_value=()
        )
        for name in empty_artifacts
    }
    content_values.update(
        {
            name: compute_content_fingerprint(
                artifact_name=name, rows_or_value=rows
            )
            for name, rows in populated.items()
        }
    )
    content_values.update(
        {
            "cost_summary": compute_content_fingerprint(
                artifact_name="cost_summary", rows_or_value=cost_summary
            ),
            "exit_reason_summary": compute_content_fingerprint(
                artifact_name="exit_reason_summary",
                rows_or_value=exit_reason_summary,
            ),
            "summary": compute_content_fingerprint(
                artifact_name="summary", rows_or_value=summary
            ),
            "audit_summary": compute_content_fingerprint(
                artifact_name="audit_summary", rows_or_value=audit_summary
            ),
        }
    )
    content_fingerprints = HistoricalBacktestContentFingerprints(
        **content_values
    )
    source_run_fingerprint = compute_source_run_fingerprint(source_run)
    manifest = _synthetic_run_manifest()
    return HistoricalBacktestAuditResult(
        run_manifest=manifest,
        source_run_fingerprint=source_run_fingerprint,
        decision_interval=decision_interval,
        initial_state=state,
        final_state=state,
        initial_state_fingerprint=state_fingerprint,
        final_state_fingerprint=state_fingerprint,
        initial_equity=initial_equity,
        final_equity=final_equity,
        period_pnl=final_equity - initial_equity,
        session_transitions=session_transitions,
        trades=trades,
        equity_curve=equity_curve,
        cost_summary=cost_summary,
        exit_reason_summary=exit_reason_summary,
        summary=summary,
        audit_summary=audit_summary,
        content_fingerprints=content_fingerprints,
        result_fingerprint=compute_result_fingerprint(
            schema_version="historical_backtest_audit_result.v0.2",
            decision_interval=decision_interval,
            run_configuration_fingerprint=(
                manifest.run_configuration_fingerprint
            ),
            source_run_fingerprint=source_run_fingerprint,
            initial_state_fingerprint=state_fingerprint,
            final_state_fingerprint=state_fingerprint,
            content_fingerprints=content_fingerprints,
        ),
    )


@pytest.fixture
def synthetic_audit_result():
    """The original calculation-free Phase 16B.1 zero-session fixture."""

    return build_synthetic_audit_result()


CANONICAL_SESSIONS = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 12, 31))
CANONICAL_EQUITIES = (
    Decimal("101000"),
    Decimal("99000"),
    Decimal("110000"),
)
CANONICAL_BENCHMARK_VALUES = (
    Decimal("100500"),
    Decimal("100250"),
    Decimal("104000"),
)


def canonical_experiment(**overrides) -> HistoricalBacktestAuditResult:
    """The valid three-session canonical baseline, with optional overrides."""

    values = {
        "equity_observations": tuple(
            zip(CANONICAL_SESSIONS, CANONICAL_EQUITIES, strict=True)
        ),
        "initial_equity": Decimal("100000"),
    }
    values.update(overrides)
    return build_synthetic_audit_result(**values)


@pytest.fixture
def canonical_audit_result():
    """A valid three-session canonical baseline experiment.

    Starting wealth is exactly the frozen USD 100,000, the final genuine
    session is the decision end boundary, and the equity sessions equal the
    authoritative processed-session sequence exactly.
    """

    return canonical_experiment()


def benchmark_evidence(source_ref, sessions, values):
    """Build one synthetic authoritative benchmark-equity path."""

    return build_benchmark_performance_series(
        benchmark_id="SYNTHETIC_PRIMARY_BENCHMARK_V0_1",
        source_artifact_ref=source_ref,
        observations=tuple(
            BenchmarkPerformanceObservation(session=session, value=value)
            for session, value in zip(sessions, values, strict=True)
        ),
    )


@pytest.fixture
def canonical_benchmark_series(benchmark_source_ref):
    """Benchmark evidence exactly congruent with the canonical experiment."""

    return benchmark_evidence(
        benchmark_source_ref, CANONICAL_SESSIONS, CANONICAL_BENCHMARK_VALUES
    )
