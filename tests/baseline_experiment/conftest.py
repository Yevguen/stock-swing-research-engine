"""Explicitly synthetic Phase 16C fixtures.

Every artifact built here is synthetic. None of it requires -- or stands in for
-- the canonical historical dataset, and every artifact reference carries the
``synthetic-fixture`` build id so a fixture can never be mistaken for canonical
experiment evidence.
"""

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
    HistoricalBacktestExitReasonRow,
    HistoricalBacktestExitReasonSummary,
    HistoricalBacktestSummary,
    HistoricalBacktestValuationPolicy,
    HistoricalClosedTradeRecord,
    HistoricalExitReason,
    HistoricalSignalProvenance,
    PolicyArtifactRef,
    build_run_manifest,
    build_valuation_policy_ref,
    compute_content_fingerprint,
    compute_result_fingerprint,
    compute_source_run_fingerprint,
)
from stock_swing_d1.backtester import (
    HistoricalBacktestRunResult,
    HistoricalDecisionInterval,
)
from stock_swing_d1.earnings.integration.models import (
    EarningsIntegrationAction,
)
from stock_swing_d1.execution.costs.models import (
    BROKER_NEUTRAL_POLICY_ID,
    ExecutionCostPolicyRef,
)
from stock_swing_d1.portfolio.allocation_policy import (
    PORTFOLIO_ALLOCATION_POLICY_REF,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.ranking.models import (
    CANDIDATE_RANKING_POLICY_ID,
    CANDIDATE_RANKING_POLICY_VERSION,
    RankingPolicyRef,
)
from stock_swing_d1.research_metrics import (
    BenchmarkPerformanceMetrics,
    BenchmarkSourceArtifactRef,
    RelativePerformanceMetrics,
    ResearchMetricsProvenance,
    StrategyPerformanceMetrics,
    TradePerformanceMetrics,
    build_performance_measurement_policy,
    build_research_metrics_persistence_artifact,
    build_research_metrics_result,
    defined_metric,
)
from stock_swing_d1.research_metrics.persistence import (
    ResearchMetricsPolicyRef,
)
from stock_swing_d1.strategy.baseline.models import BaselineSignalAction

from stock_swing_d1.baseline_experiment import (
    BenchmarkEvidence,
    CanonicalRunPreflightInput,
    CorporateActionEvidence,
    DecisionIntervalEvidence,
    DividendEvidence,
    EarningsPitEvidence,
    ExecutionSemanticsEvidence,
    MarketDataEvidence,
    MetricsPolicyEvidence,
    PersistenceReadinessProbe,
    RepositoryIdentityEvidence,
    StartingCapitalEvidence,
    StrategyConfigurationEvidence,
    UniverseEvidence,
    build_canonical_baseline_experiment_manifest,
    build_material_adverse_overnight_gap_policy,
    build_material_adverse_overnight_gap_policy_ref,
)


SYNTHETIC_SECURITY_ID = "NORGATE:1"
SYNTHETIC_OTHER_SECURITY_ID = "NORGATE:2"
SYNTHETIC_DECISION_TIME = datetime(2025, 1, 2, 21, 0, tzinfo=timezone.utc)
SYNTHETIC_GIT_COMMIT = "a" * 40
SYNTHETIC_BENCHMARK_SERIES_FINGERPRINT = "b" * 64
DECISION_START = date(2025, 1, 1)
DECISION_END = date(2025, 12, 31)


def synthetic_artifact_ref(artifact_type: str, digest_seed: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=artifact_type,
        schema_version="synthetic.v0.1",
        content_sha256=(digest_seed * 64)[:64],
        build_id="synthetic-fixture",
    )


def synthetic_run_manifest():
    return build_run_manifest(
        software_revision=SYNTHETIC_GIT_COMMIT,
        strategy_configuration_ref=synthetic_artifact_ref(
            "synthetic_strategy", "2"
        ),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id=PORTFOLIO_ALLOCATION_POLICY_REF.policy_id,
            policy_version=PORTFOLIO_ALLOCATION_POLICY_REF.policy_version,
            policy_fingerprint=(
                PORTFOLIO_ALLOCATION_POLICY_REF.policy_fingerprint
            ),
        ),
        ranking_policy_ref=RankingPolicyRef(
            policy_id=CANDIDATE_RANKING_POLICY_ID,
            policy_version=CANDIDATE_RANKING_POLICY_VERSION,
            policy_fingerprint="4" * 64,
        ),
        execution_cost_policy_ref=ExecutionCostPolicyRef(
            policy_id=BROKER_NEUTRAL_POLICY_ID,
            policy_fingerprint="5" * 64,
        ),
        valuation_policy_ref=build_valuation_policy_ref(
            HistoricalBacktestValuationPolicy()
        ),
        universe_artifact_ref=synthetic_artifact_ref(
            "synthetic_universe", "6"
        ),
        market_data_artifact_ref=synthetic_artifact_ref(
            "synthetic_market_data", "7"
        ),
    )


def make_closed_trade(
    *,
    trade_id: str,
    entry_session: date,
    exit_session: date,
    security_id: str = SYNTHETIC_SECURITY_ID,
    entry_fill_price: Decimal = Decimal("100"),
    exit_fill_price: Decimal = Decimal("110"),
    quantity: int = 100,
) -> HistoricalClosedTradeRecord:
    entry_cost_basis = entry_fill_price * quantity
    gross_exit_proceeds = exit_fill_price * quantity
    realized_pnl = gross_exit_proceeds - entry_cost_basis
    return HistoricalClosedTradeRecord(
        schema_version="historical_closed_trade.v0.3",
        trade_id=trade_id,
        security_id=security_id,
        quantity=quantity,
        carried_in=False,
        entry_execution_id=trade_id,
        entry_session=entry_session,
        entry_fill_price=entry_fill_price,
        entry_execution_cost=Decimal("0"),
        entry_cost_basis=entry_cost_basis,
        exit_execution_id=f"{trade_id}-exit",
        exit_session=exit_session,
        exit_fill_price=exit_fill_price,
        exit_execution_cost=Decimal("0"),
        exit_reason=HistoricalExitReason.TAKE_PROFIT,
        gross_exit_proceeds=gross_exit_proceeds,
        net_exit_proceeds=gross_exit_proceeds,
        realized_pnl=realized_pnl,
        ordinary_dividend_income=Decimal("0"),
        ordinary_dividend_event_count=0,
        dividend_attribution_completeness=(
            DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
        ),
        ordinary_dividend_attribution_fingerprint="8" * 64,
        trade_total_pnl=realized_pnl,
    )


def make_signal_row(
    *,
    session: date,
    earnings_action: EarningsIntegrationAction,
    security_id: str = SYNTHETIC_SECURITY_ID,
    action: BaselineSignalAction = BaselineSignalAction.NO_SIGNAL,
    universe_eligible: bool = True,
    close_above_sma50: bool = True,
    sma20_above_sma50: bool = True,
    rsi_above_50: bool = True,
    atr_above_minimum: bool = True,
    earnings_entry_allowed: bool | None = None,
    digest_seed: str = "9",
) -> HistoricalSignalProvenance:
    """One authoritative point-in-time signal/earnings decision row."""

    if earnings_entry_allowed is None:
        earnings_entry_allowed = (
            earnings_action is EarningsIntegrationAction.ENTRY_ALLOWED
        )
    return HistoricalSignalProvenance(
        session=session,
        decision_time=SYNTHETIC_DECISION_TIME,
        security_id=security_id,
        action=action,
        planned_entry_session=date.fromordinal(session.toordinal() + 1),
        adjusted_close=100.0,
        universe_eligible=universe_eligible,
        close_above_sma50=close_above_sma50,
        sma20_above_sma50=sma20_above_sma50,
        rsi_above_50=rsi_above_50,
        atr_above_minimum=atr_above_minimum,
        earnings_entry_allowed=earnings_entry_allowed,
        earnings_action=earnings_action,
        source_payload_fingerprint=(digest_seed * 64)[:64],
    )


def build_synthetic_audit_result(
    *,
    trades: tuple = (),
    signal_provenance: tuple = (),
    decision_start: date = DECISION_START,
    decision_end: date = DECISION_END,
    initial_equity: Decimal = Decimal("100000"),
    final_equity: Decimal = Decimal("110000"),
) -> HistoricalBacktestAuditResult:
    """Build one explicitly synthetic, self-consistent Phase 15D audit result."""

    state = PortfolioState(settled_cash=Decimal("100000"))
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
                reason=reason, exit_count=0, realized_pnl=Decimal("0")
            )
            for reason in HistoricalExitReason
        )
    )
    closed_trade_count = sum(
        1 for trade in trades if type(trade) is HistoricalClosedTradeRecord
    )
    summary = HistoricalBacktestSummary(
        processed_session_count=0,
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
        "trades": trades,
        "signal_provenance": signal_provenance,
    }
    empty_artifacts = (
        "session_transitions",
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
        "equity_curve",
    )
    content_values = {
        name: compute_content_fingerprint(artifact_name=name, rows_or_value=())
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
    manifest = synthetic_run_manifest()
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
        trades=trades,
        signal_provenance=signal_provenance,
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


def build_synthetic_metrics_artifact(source_result):
    """Copy hand-written synthetic metrics into a Phase 16B.3 artifact.

    The values are fixtures, not calculations: this file never invokes the
    Phase 16B calculation boundary, which is exactly the boundary Phase 16C is
    forbidden to reach for.
    """

    policy = build_performance_measurement_policy()
    result = build_research_metrics_result(
        provenance=ResearchMetricsProvenance(
            source_audit_result_fingerprint=source_result.result_fingerprint,
            performance_measurement_policy_fingerprint=(
                policy.policy_fingerprint
            ),
            benchmark_series_fingerprint=(
                SYNTHETIC_BENCHMARK_SERIES_FINGERPRINT
            ),
        ),
        strategy_metrics=StrategyPerformanceMetrics(
            ending_equity=Decimal("110000"),
            net_pnl=Decimal("10000"),
            total_return=defined_metric(Decimal("0.100000000000000000")),
            cagr=defined_metric(Decimal("0.100360090511656501")),
            annualized_volatility=defined_metric(
                Decimal("1.469495805366663448")
            ),
            maximum_drawdown=defined_metric(
                Decimal("0.019801980198019802")
            ),
            sharpe_ratio=defined_metric(Decimal("7.829182263081608445")),
            sortino_ratio=defined_metric(Decimal("51.759593850372855834")),
        ),
        trade_metrics=TradePerformanceMetrics(
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
            average_winner_usd=defined_metric(
                Decimal("205.000000000000000000")
            ),
            average_loser_usd=defined_metric(
                Decimal("-100.000000000000000000")
            ),
            worst_trade_usd=defined_metric(Decimal("-100.000000000000000000")),
            average_holding_sessions=defined_metric(
                Decimal("2.000000000000000000")
            ),
        ),
        benchmark_metrics=BenchmarkPerformanceMetrics(
            ending_equity=Decimal("104000"),
            total_return=defined_metric(Decimal("0.040000000000000000")),
            cagr=defined_metric(Decimal("0.040140083409008749")),
            annualized_volatility=defined_metric(
                Decimal("0.447809555483366529")
            ),
            maximum_drawdown=defined_metric(
                Decimal("0.002487562189054726")
            ),
            sharpe_ratio=defined_metric(Decimal("2.5")),
            sortino_ratio=defined_metric(Decimal("3.5")),
        ),
        relative_metrics=RelativePerformanceMetrics(
            strategy_minus_benchmark_total_return=defined_metric(
                Decimal("0.060000000000000000")
            ),
            strategy_minus_benchmark_cagr=defined_metric(
                Decimal("0.060220007102647752")
            ),
            ending_wealth_ratio=defined_metric(
                Decimal("1.057692307692307692")
            ),
        ),
    )
    return build_research_metrics_persistence_artifact(
        result=result,
        decision_interval=source_result.decision_interval,
        starting_capital=Decimal("100000"),
        equity_observation_count=3,
        periodic_return_observation_count=2,
        policy=policy,
    )


def synthetic_benchmark_source_ref() -> BenchmarkSourceArtifactRef:
    return BenchmarkSourceArtifactRef(
        artifact_id="synthetic-spy-benchmark-evidence",
        schema_version="synthetic-benchmark.v0.1",
        content_sha256="c" * 64,
        build_id="synthetic-fixture",
    )


def build_synthetic_manifest(**overrides):
    """Build one synthetic canonical experiment manifest."""

    policy = build_performance_measurement_policy()
    values = {
        "git_commit_sha": SYNTHETIC_GIT_COMMIT,
        "decision_interval": HistoricalDecisionInterval(
            decision_start_date=DECISION_START,
            decision_end_date=DECISION_END,
        ),
        "starting_capital": Decimal("100000"),
        "strategy_configuration_ref": synthetic_artifact_ref(
            "synthetic_strategy", "2"
        ),
        "universe_artifact_ref": synthetic_artifact_ref(
            "synthetic_universe", "6"
        ),
        "market_data_artifact_ref": synthetic_artifact_ref(
            "synthetic_market_data", "7"
        ),
        "corporate_action_artifact_ref": synthetic_artifact_ref(
            "synthetic_corporate_actions", "d"
        ),
        "dividend_evidence_ref": synthetic_artifact_ref(
            "synthetic_dividends", "e"
        ),
        "earnings_pit_artifact_ref": synthetic_artifact_ref(
            "synthetic_earnings_pit", "f"
        ),
        "earnings_provider_name": "SYNTHETIC_EARNINGS_PROVIDER",
        "allocation_policy_ref": PolicyArtifactRef(
            policy_id=PORTFOLIO_ALLOCATION_POLICY_REF.policy_id,
            policy_version=PORTFOLIO_ALLOCATION_POLICY_REF.policy_version,
            policy_fingerprint=(
                PORTFOLIO_ALLOCATION_POLICY_REF.policy_fingerprint
            ),
        ),
        "ranking_policy_ref": RankingPolicyRef(
            policy_id=CANDIDATE_RANKING_POLICY_ID,
            policy_version=CANDIDATE_RANKING_POLICY_VERSION,
            policy_fingerprint="4" * 64,
        ),
        "execution_cost_policy_ref": ExecutionCostPolicyRef(
            policy_id=BROKER_NEUTRAL_POLICY_ID, policy_fingerprint="5" * 64
        ),
        "valuation_policy_ref": build_valuation_policy_ref(
            HistoricalBacktestValuationPolicy()
        ),
        "run_configuration_fingerprint": (
            synthetic_run_manifest().run_configuration_fingerprint
        ),
        "benchmark_id": "SPY_BUY_AND_HOLD_V0_1",
        "benchmark_source_artifact_ref": synthetic_benchmark_source_ref(),
        "benchmark_series_fingerprint": (
            SYNTHETIC_BENCHMARK_SERIES_FINGERPRINT
        ),
        "performance_measurement_policy_ref": ResearchMetricsPolicyRef(
            policy_fingerprint=policy.policy_fingerprint
        ),
        "gap_policy_ref": build_material_adverse_overnight_gap_policy_ref(
            build_material_adverse_overnight_gap_policy()
        ),
    }
    values.update(overrides)
    return build_canonical_baseline_experiment_manifest(**values)


def build_ready_preflight_input(**overrides) -> CanonicalRunPreflightInput:
    """Assemble a synthetic input for which every gate A-N passes."""

    # The reference manifest supplies the identities every other gate's
    # evidence must agree with. An explicit ``manifest=None`` override still
    # removes the manifest from the audited input afterwards.
    manifest = overrides.get("manifest") or build_synthetic_manifest()
    source_result = build_synthetic_audit_result()
    values = {
        "repository_identity": RepositoryIdentityEvidence(
            reviewed_git_commit_sha=manifest.git_commit_sha,
            working_tree_clean=True,
            experiment_implementation_reviewed=True,
            required_tests_green=True,
        ),
        "decision_interval_evidence": DecisionIntervalEvidence(
            decision_interval=manifest.decision_interval,
            interval_source="frozen_authoritative_contract",
            derived_from_observed_sessions=False,
            selected_after_observing_results=False,
        ),
        "starting_capital_evidence": StartingCapitalEvidence(
            base_currency="USD", starting_capital=Decimal("100000")
        ),
        "strategy_configuration_evidence": StrategyConfigurationEvidence(
            artifact_ref=manifest.strategy_configuration_ref,
            configuration_frozen=True,
            tuned_parameter_count=0,
        ),
        "universe_evidence": UniverseEvidence(
            artifact_ref=manifest.universe_artifact_ref,
            point_in_time_membership=True,
            coverage_start=DECISION_START,
            coverage_end=DECISION_END,
            validation_passed=True,
        ),
        "market_data_evidence": MarketDataEvidence(
            artifact_ref=manifest.market_data_artifact_ref,
            timeframe="D1",
            session_type="regular",
            price_basis="unadjusted",
            currency="USD",
            coverage_start=DECISION_START,
            coverage_end=DECISION_END,
            dataset_validation_passed=True,
        ),
        "corporate_action_evidence": CorporateActionEvidence(
            artifact_ref=manifest.corporate_action_artifact_ref,
            coverage_start=DECISION_START,
            coverage_end=DECISION_END,
            unresolved_action_count=0,
            parity_validation_passed=True,
        ),
        "dividend_evidence": DividendEvidence(
            artifact_ref=manifest.dividend_evidence_ref,
            coverage_start=DECISION_START,
            coverage_end=DECISION_END,
            accounting_contract_version="synthetic_dividend_contract.v0.1",
            validation_passed=True,
        ),
        "earnings_pit_evidence": EarningsPitEvidence(
            artifact_ref=manifest.earnings_pit_artifact_ref,
            provider_name=manifest.earnings_provider_name,
            coverage_start=DECISION_START,
            coverage_end=DECISION_END,
            point_in_time_reconstruction_validated=True,
            look_ahead_substitution_used=False,
        ),
        "execution_semantics_evidence": ExecutionSemanticsEvidence(
            execution_cost_policy_ref=manifest.execution_cost_policy_ref,
            settlement_model="historical_us_equity_standard_cycle",
            allocation_policy_ref=manifest.allocation_policy_ref,
            ranking_policy_ref=manifest.ranking_policy_ref,
            valuation_policy_ref=manifest.valuation_policy_ref,
        ),
        "benchmark_evidence": BenchmarkEvidence(
            benchmark_id=manifest.benchmark_id,
            source_artifact_ref=manifest.benchmark_source_artifact_ref,
            series_fingerprint=manifest.benchmark_series_fingerprint,
            coverage_start=DECISION_START,
            coverage_end=DECISION_END,
            price_basis="unadjusted",
            distribution_reinvestment="prohibited",
            source_acceptance_passed=True,
        ),
        "metrics_policy_evidence": MetricsPolicyEvidence(
            policy_ref=manifest.performance_measurement_policy_ref
        ),
        "persistence_readiness_probe": PersistenceReadinessProbe(
            rehearsal_artifact=build_synthetic_metrics_artifact(source_result)
        ),
        "manifest": manifest,
    }
    values.update(overrides)
    return CanonicalRunPreflightInput(**values)


@pytest.fixture
def gap_policy():
    return build_material_adverse_overnight_gap_policy()


@pytest.fixture
def synthetic_manifest():
    return build_synthetic_manifest()


@pytest.fixture
def ready_preflight_input():
    return build_ready_preflight_input()
