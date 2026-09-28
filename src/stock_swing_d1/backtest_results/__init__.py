"""Phase 15D historical backtest result and audit contracts."""

# Phase 16B.2 v0.2.1 Clauses 27/28/60: the frozen Exact-Arithmetic Hardening
# primitives are exposed here, unchanged, through the one package boundary
# Phase 16B is already permitted to depend on. This is a pure re-export --
# no new arithmetic, no new owner, no Phase-15D semantic change. Phase 16B
# must consume these names rather than importing their portfolio-owner
# implementation module directly. `sum_exact_decimal` is the stable public
# alias for the already-existing exact accumulator used by Phase 15D
# valuation; the alias performs no arithmetic of its own.
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    add_exact_decimal,
    exact_decimal_times_int,
    subtract_exact_decimal,
)
from stock_swing_d1.backtest_results.valuation import (
    _exact_sum as sum_exact_decimal,
)
from stock_swing_d1.backtest_results.canonical import (
    canonicalize_semantic_value,
    semantic_json_bytes,
)
from stock_swing_d1.backtest_results.errors import (
    HistoricalBacktestPersistenceError,
    HistoricalBacktestResultValidationError,
)
from stock_swing_d1.backtest_results.derivation import (
    project_allocation_provenance,
    project_cash_ledger,
    project_closed_trades,
    project_entries,
    project_execution_provenance,
    project_exits,
    project_ranking_provenance,
    project_rejections,
    project_settlement_ledger,
    project_signal_provenance,
    project_transition_audits,
)
from stock_swing_d1.backtest_results.hashing import (
    compute_content_fingerprint,
    compute_dividend_attribution_fingerprint,
    compute_result_fingerprint,
    compute_source_payload_fingerprint,
    compute_source_run_fingerprint,
    semantic_domain_sha256,
    semantic_sha256,
)
from stock_swing_d1.backtest_results.persistence import (
    HistoricalBacktestResultPersistence,
)
from stock_swing_d1.backtest_results.service import (
    HistoricalBacktestResultService,
)
from stock_swing_d1.backtest_results.source_validation import (
    validate_historical_backtest_source_run,
)
from stock_swing_d1.backtest_results.valuation import (
    project_equity_curve,
    project_open_trades,
    project_session_pnl,
)
from stock_swing_d1.backtest_results.models import (
    ArtifactRef,
    DividendAttributionCompleteness,
    ExecutionApplicationStatus,
    ExecutionProvenanceSource,
    HistoricalAllocationCandidateProvenance,
    HistoricalAllocationCycle,
    HistoricalBacktestAuditResult,
    HistoricalBacktestAuditSummary,
    HistoricalBacktestContentFingerprints,
    HistoricalBacktestCostSummary,
    HistoricalBacktestEntryRecord,
    HistoricalBacktestEquityRow,
    HistoricalBacktestExecutionProvenance,
    HistoricalBacktestExitReasonRow,
    HistoricalBacktestExitReasonSummary,
    HistoricalBacktestExitRecord,
    HistoricalBacktestRejectionRecord,
    HistoricalBacktestRunManifest,
    HistoricalBacktestSessionPnl,
    HistoricalBacktestSummary,
    HistoricalBacktestTransitionAudit,
    HistoricalBacktestValuationMark,
    HistoricalBacktestValuationPolicy,
    HistoricalBacktestValuationPolicyRef,
    HistoricalBacktestValuationSnapshot,
    HistoricalCashLedgerRow,
    HistoricalClosedTradeRecord,
    HistoricalExitReason,
    HistoricalOpenTradeRecord,
    HistoricalRankingCandidateProvenance,
    HistoricalRankingCycle,
    HistoricalRejectionStage,
    HistoricalSettlementRecord,
    HistoricalSettlementStatus,
    HistoricalSignalProvenance,
    HistoricalTradeRecord,
    HistoricalTradeStatus,
    PolicyArtifactRef,
    build_run_manifest,
    build_valuation_policy_ref,
    build_valuation_snapshot,
    compute_run_configuration_fingerprint,
)


__all__ = [
    "ArtifactRef",
    "DividendAttributionCompleteness",
    "ExecutionApplicationStatus",
    "ExecutionProvenanceSource",
    "HistoricalAllocationCandidateProvenance",
    "HistoricalAllocationCycle",
    "HistoricalBacktestAuditResult",
    "HistoricalBacktestAuditSummary",
    "HistoricalBacktestContentFingerprints",
    "HistoricalBacktestCostSummary",
    "HistoricalBacktestEntryRecord",
    "HistoricalBacktestEquityRow",
    "HistoricalBacktestExecutionProvenance",
    "HistoricalBacktestExitReasonRow",
    "HistoricalBacktestExitReasonSummary",
    "HistoricalBacktestExitRecord",
    "HistoricalBacktestPersistenceError",
    "HistoricalBacktestRejectionRecord",
    "HistoricalBacktestResultPersistence",
    "HistoricalBacktestResultService",
    "HistoricalBacktestResultValidationError",
    "HistoricalBacktestRunManifest",
    "HistoricalBacktestSessionPnl",
    "HistoricalBacktestSummary",
    "HistoricalBacktestTransitionAudit",
    "HistoricalBacktestValuationMark",
    "HistoricalBacktestValuationPolicy",
    "HistoricalBacktestValuationPolicyRef",
    "HistoricalBacktestValuationSnapshot",
    "HistoricalCashLedgerRow",
    "HistoricalClosedTradeRecord",
    "HistoricalExitReason",
    "HistoricalOpenTradeRecord",
    "HistoricalRankingCandidateProvenance",
    "HistoricalRankingCycle",
    "HistoricalRejectionStage",
    "HistoricalSettlementRecord",
    "HistoricalSettlementStatus",
    "HistoricalSignalProvenance",
    "HistoricalTradeRecord",
    "HistoricalTradeStatus",
    "PolicyArtifactRef",
    "add_exact_decimal",
    "build_run_manifest",
    "build_valuation_policy_ref",
    "build_valuation_snapshot",
    "canonicalize_semantic_value",
    "compute_content_fingerprint",
    "compute_dividend_attribution_fingerprint",
    "compute_result_fingerprint",
    "compute_run_configuration_fingerprint",
    "compute_source_payload_fingerprint",
    "compute_source_run_fingerprint",
    "exact_decimal_times_int",
    "project_allocation_provenance",
    "project_cash_ledger",
    "project_closed_trades",
    "project_entries",
    "project_equity_curve",
    "project_execution_provenance",
    "project_exits",
    "project_open_trades",
    "project_ranking_provenance",
    "project_rejections",
    "project_session_pnl",
    "project_settlement_ledger",
    "project_signal_provenance",
    "project_transition_audits",
    "semantic_domain_sha256",
    "semantic_json_bytes",
    "semantic_sha256",
    "subtract_exact_decimal",
    "sum_exact_decimal",
    "validate_historical_backtest_source_run",
]
