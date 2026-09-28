"""Phase 15D.3C: thin, stateless orchestration of the final audit result.

`HistoricalBacktestResultService.build` is the only public Phase 15D.3C
orchestration API. It calls only Phase 15D validation/projection/
reconciliation functions — it never queries market data, calls a
provider/data-pipeline API, reruns signals/ranking/sizing/allocation/entry-
exit decisions, reprices an execution, recomputes an execution-cost-policy
formula, calls a settlement resolver/calendar, invokes
`PortfolioTransitionEngine`, mutates Phase 13 state, force-liquidates a
position, or persists anything.
"""

from __future__ import annotations

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
from stock_swing_d1.backtest_results.errors import (
    HistoricalBacktestResultValidationError,
)
from stock_swing_d1.backtest_results.hashing import (
    compute_content_fingerprint,
    compute_result_fingerprint,
    compute_source_run_fingerprint,
)
from stock_swing_d1.backtest_results.models import (
    HISTORICAL_BACKTEST_AUDIT_RESULT_SCHEMA_VERSION,
    HistoricalBacktestAuditResult,
    HistoricalBacktestAuditSummary,
    HistoricalBacktestContentFingerprints,
    HistoricalBacktestRunManifest,
    HistoricalBacktestSummary,
    HistoricalBacktestValuationSnapshot,
    HistoricalRejectionStage,
)
from stock_swing_d1.backtest_results.reconciliation import (
    _assemble_trades,
    _build_exit_reason_summary,
    _compute_equity_and_pnl_scalars,
    _count_rejections_by_stage,
    _prove_and_build_cost_summary,
    _prove_trade_linkage,
    _prove_valuation_policy_consistency,
)
from stock_swing_d1.backtest_results.source_validation import (
    validate_historical_backtest_source_run,
)
from stock_swing_d1.backtest_results.valuation import (
    project_equity_curve,
    project_open_trades,
    project_session_pnl,
)
from stock_swing_d1.backtester.models import HistoricalBacktestRunResult


class HistoricalBacktestResultService:
    """Stateless, read-only assembly of one final `HistoricalBacktestAuditResult`."""

    @staticmethod
    def build(
        *,
        run_result: HistoricalBacktestRunResult,
        run_manifest: HistoricalBacktestRunManifest,
        valuation_snapshots: tuple[HistoricalBacktestValuationSnapshot, ...],
    ) -> HistoricalBacktestAuditResult:
        """Assemble the complete in-memory audit result, or fail closed.

        Every substantive `HistoricalBacktestAuditSummary` flag corresponds
        to a proof that actually executed immediately before the flag is
        recorded `True`; none is set from mere reachability. `audit_passed`
        is the exact conjunction of all fifteen flags.
        """

        # Explicit source-audit gate: proves source_run_canonical,
        # state_chain_valid, ledger_reconstruction_valid,
        # execution_ledger_provenance_valid, and signal_provenance_valid.
        # Called with the REAL manifest, this same gate also proves three of
        # the four conjuncts behind policy_consistency_valid (see below):
        #   1. Phase 14 ranking policy == run_manifest.ranking_policy_ref
        #      (raises RANKING_POLICY_MISMATCH on disagreement);
        #   2. Phase 12 allocation policy == run_manifest.allocation_policy_ref
        #      (raises ARTIFACT_PROVENANCE_MISMATCH on disagreement);
        #   3. authoritative execution-cost policy evidence ==
        #      run_manifest.execution_cost_policy_ref (raises
        #      EXECUTION_COST_POLICY_MISMATCH on disagreement).
        # Any failure raises here, before ranking_provenance_valid /
        # allocation_provenance_valid / execution_provenance_valid /
        # policy_consistency_valid are ever assigned True.
        validate_historical_backtest_source_run(
            run_result=run_result, run_manifest=run_manifest
        )
        source_run_canonical = True
        state_chain_valid = True
        ledger_reconstruction_valid = True
        execution_ledger_provenance_valid = True
        signal_provenance_valid = True
        ranking_provenance_valid = True
        allocation_provenance_valid = True
        execution_provenance_valid = True

        # The 4th conjunct of policy_consistency_valid: the frozen Phase 15D
        # valuation policy is not covered by the gate above or by 15D.3B, so
        # it is proven here explicitly. Raises VALUATION_ARTIFACT_MISMATCH
        # if run_manifest.valuation_policy_ref disagrees with
        # build_valuation_policy_ref(HistoricalBacktestValuationPolicy()).
        _prove_valuation_policy_consistency(run_manifest)

        # Explicit Phase 15D.3B valuation-evidence boundary
        # (`_validate_valuation_snapshots`), exercised for the exact
        # caller-supplied valuation_snapshots. This proves, together:
        #   - valuation_coverage_valid: the snapshot tuple is exactly the
        #     frozen canonical container, covers exactly the required
        #     sessions, covers exactly each session's open-position security
        #     set, and every snapshot's own fingerprint is exact;
        #   - artifact_fingerprints_valid (see the note below for its exact,
        #     narrower scope): every valuation mark's source_artifact_ref
        #     equals run_manifest.market_data_artifact_ref exactly (raises
        #     VALUATION_ARTIFACT_MISMATCH otherwise).
        # Any failure raises inside these three calls, before
        # valuation_coverage_valid / artifact_fingerprints_valid /
        # policy_consistency_valid are ever assigned True.
        session_pnl = project_session_pnl(
            run_result=run_result,
            run_manifest=run_manifest,
            valuation_snapshots=valuation_snapshots,
        )
        equity_curve = project_equity_curve(
            run_result=run_result,
            run_manifest=run_manifest,
            valuation_snapshots=valuation_snapshots,
        )
        open_trades = project_open_trades(
            run_result=run_result,
            run_manifest=run_manifest,
            valuation_snapshots=valuation_snapshots,
        )
        valuation_coverage_valid = True

        # artifact_fingerprints_valid — exact frozen meaning:
        #
        # 1. Every manifest ArtifactRef that is present has already passed
        #    its own canonical structural validation (a plain consequence of
        #    run_manifest being a genuine HistoricalBacktestRunManifest
        #    instance, already proven above).
        # 2. Every artifact identity relationship for which Phase 15D holds
        #    independent immutable cross-object evidence is proven exactly.
        # 3. In the current source contract, market_data_artifact_ref is the
        #    only such relationship: every valuation mark carries its own
        #    source_artifact_ref, and the calls above prove it equals
        #    run_manifest.market_data_artifact_ref exactly.
        # 4. strategy_configuration_ref, universe_artifact_ref,
        #    corporate_action_artifact_ref, and earnings_artifact_ref
        #    currently have NO independent source-run artifact-fingerprint
        #    binding available to Phase 15D — nothing in the immutable
        #    run_result carries a comparable identity for them — so Phase
        #    15D does not fabricate a comparison for any of the four.
        # 5. In particular, current source earnings decisions do not expose
        #    provider name, published build_id, or output SHA-256 as an
        #    independent binding (see source_validation.py's own note on
        #    this), so earnings_artifact_ref is never falsely claimed to
        #    have been independently verified.
        # 6. Phase 15D.3C never re-hashes the underlying provider/artifact
        #    files themselves — only the already-supplied ArtifactRef
        #    identity fields are compared.
        # 7. This flag is entirely independent of the 19
        #    HistoricalBacktestContentFingerprints built later in this
        #    method: it is assigned from the proofs above, long before
        #    `content_fingerprints` is ever constructed.
        artifact_fingerprints_valid = True

        # policy_consistency_valid is the exact conjunction of all four
        # policy-vs-manifest proofs above: ranking (1), allocation (2), and
        # execution-cost (3) via validate_historical_backtest_source_run
        # earlier in this method, plus valuation (4) via
        # _prove_valuation_policy_consistency. A failure in any one of the
        # four raises before this line is ever reached, so this is a record
        # of a completed four-part proof, not a valuation-only check.
        policy_consistency_valid = True

        transition_audits = project_transition_audits(run_result)
        signal_provenance = project_signal_provenance(run_result)
        ranking_cycles, ranking_candidates = project_ranking_provenance(run_result)
        allocation_cycles, allocation_candidates = project_allocation_provenance(
            run_result
        )
        execution_provenance = project_execution_provenance(run_result)
        rejections = project_rejections(run_result)
        entries = project_entries(run_result)
        exits = project_exits(run_result)
        closed_trades = project_closed_trades(run_result)
        cash_ledger = project_cash_ledger(run_result)
        settlement_ledger = project_settlement_ledger(run_result)

        _prove_trade_linkage(
            run_result=run_result,
            entries=entries,
            exits=exits,
            closed_trades=closed_trades,
            open_trades=open_trades,
        )
        trades = _assemble_trades(closed_trades, open_trades)
        trade_linkage_valid = True

        scalars = _compute_equity_and_pnl_scalars(
            run_result=run_result,
            run_manifest=run_manifest,
            valuation_snapshots=valuation_snapshots,
            equity_curve=equity_curve,
            session_pnl=session_pnl,
            closed_trades=closed_trades,
        )
        equity_reconciliation_valid = True
        pnl_reconciliation_valid = True

        cost_summary = _prove_and_build_cost_summary(
            run_result=run_result,
            entries=entries,
            exits=exits,
            execution_provenance=execution_provenance,
            final_cumulative_execution_cost=(
                scalars.final_cumulative_execution_cost
            ),
        )
        cost_reconciliation_valid = True

        exit_reason_summary = _build_exit_reason_summary(closed_trades)

        allocation_rejection_count = _count_rejections_by_stage(
            rejections, HistoricalRejectionStage.ALLOCATION
        )
        entry_execution_rejection_count = _count_rejections_by_stage(
            rejections, HistoricalRejectionStage.ENTRY_EXECUTION
        )

        summary = HistoricalBacktestSummary(
            processed_session_count=len(run_result.session_results),
            entry_count=len(entries),
            exit_count=len(exits),
            closed_trade_count=len(closed_trades),
            open_trade_count=len(open_trades),
            allocation_rejection_count=allocation_rejection_count,
            entry_execution_rejection_count=entry_execution_rejection_count,
            gross_realized_pnl=scalars.gross_realized_pnl,
            final_unrealized_pnl=scalars.final_unrealized_pnl,
            initial_equity=scalars.initial_equity,
            final_equity=scalars.final_equity,
            period_pnl=scalars.period_pnl,
            buy_execution_cost_total=cost_summary.buy_execution_cost_total,
            sell_execution_cost_total=cost_summary.sell_execution_cost_total,
            total_execution_cost=cost_summary.total_execution_cost,
            ordinary_dividend_income_total=scalars.ordinary_dividend_income_total,
        )

        # Explicit top-level-vs-summary scalar reconciliation (never relying
        # on the model's own validator to prove this).
        if (
            scalars.initial_equity != summary.initial_equity
            or scalars.final_equity != summary.final_equity
            or scalars.period_pnl != summary.period_pnl
            or scalars.ordinary_dividend_income_total
            != summary.ordinary_dividend_income_total
        ):
            raise HistoricalBacktestResultValidationError(
                "SUMMARY_RECONCILIATION_MISMATCH",
                "top-level equity/P&L scalars disagree with the summary",
            )

        audit_flags = {
            "source_run_canonical": source_run_canonical,
            "state_chain_valid": state_chain_valid,
            "ledger_reconstruction_valid": ledger_reconstruction_valid,
            "execution_ledger_provenance_valid": (
                execution_ledger_provenance_valid
            ),
            "signal_provenance_valid": signal_provenance_valid,
            "ranking_provenance_valid": ranking_provenance_valid,
            "allocation_provenance_valid": allocation_provenance_valid,
            "execution_provenance_valid": execution_provenance_valid,
            "trade_linkage_valid": trade_linkage_valid,
            "valuation_coverage_valid": valuation_coverage_valid,
            "equity_reconciliation_valid": equity_reconciliation_valid,
            "pnl_reconciliation_valid": pnl_reconciliation_valid,
            "cost_reconciliation_valid": cost_reconciliation_valid,
            "policy_consistency_valid": policy_consistency_valid,
            "artifact_fingerprints_valid": artifact_fingerprints_valid,
        }
        audit_passed = all(audit_flags.values())
        if not audit_passed:
            raise HistoricalBacktestResultValidationError(
                "SUMMARY_RECONCILIATION_MISMATCH",
                "not every Phase 15D.3C reconciliation proof succeeded",
            )
        audit_summary = HistoricalBacktestAuditSummary(
            **audit_flags, audit_passed=audit_passed
        )

        content_fingerprints = HistoricalBacktestContentFingerprints(
            session_transitions=compute_content_fingerprint(
                artifact_name="session_transitions",
                rows_or_value=transition_audits,
            ),
            signal_provenance=compute_content_fingerprint(
                artifact_name="signal_provenance", rows_or_value=signal_provenance
            ),
            ranking_cycles=compute_content_fingerprint(
                artifact_name="ranking_cycles", rows_or_value=ranking_cycles
            ),
            ranking_candidates=compute_content_fingerprint(
                artifact_name="ranking_candidates",
                rows_or_value=ranking_candidates,
            ),
            allocation_cycles=compute_content_fingerprint(
                artifact_name="allocation_cycles", rows_or_value=allocation_cycles
            ),
            allocation_candidates=compute_content_fingerprint(
                artifact_name="allocation_candidates",
                rows_or_value=allocation_candidates,
            ),
            execution_provenance=compute_content_fingerprint(
                artifact_name="execution_provenance",
                rows_or_value=execution_provenance,
            ),
            entries=compute_content_fingerprint(
                artifact_name="entries", rows_or_value=entries
            ),
            exits=compute_content_fingerprint(
                artifact_name="exits", rows_or_value=exits
            ),
            trades=compute_content_fingerprint(
                artifact_name="trades", rows_or_value=trades
            ),
            rejections=compute_content_fingerprint(
                artifact_name="rejections", rows_or_value=rejections
            ),
            cash_ledger=compute_content_fingerprint(
                artifact_name="cash_ledger", rows_or_value=cash_ledger
            ),
            settlement_ledger=compute_content_fingerprint(
                artifact_name="settlement_ledger", rows_or_value=settlement_ledger
            ),
            session_pnl=compute_content_fingerprint(
                artifact_name="session_pnl", rows_or_value=session_pnl
            ),
            equity_curve=compute_content_fingerprint(
                artifact_name="equity_curve", rows_or_value=equity_curve
            ),
            cost_summary=compute_content_fingerprint(
                artifact_name="cost_summary", rows_or_value=cost_summary
            ),
            exit_reason_summary=compute_content_fingerprint(
                artifact_name="exit_reason_summary",
                rows_or_value=exit_reason_summary,
            ),
            summary=compute_content_fingerprint(
                artifact_name="summary", rows_or_value=summary
            ),
            audit_summary=compute_content_fingerprint(
                artifact_name="audit_summary", rows_or_value=audit_summary
            ),
        )

        source_run_fingerprint = compute_source_run_fingerprint(run_result)
        result_fingerprint = compute_result_fingerprint(
            schema_version=HISTORICAL_BACKTEST_AUDIT_RESULT_SCHEMA_VERSION,
            decision_interval=run_result.decision_interval,
            run_configuration_fingerprint=(
                run_manifest.run_configuration_fingerprint
            ),
            source_run_fingerprint=source_run_fingerprint,
            initial_state_fingerprint=run_result.initial_state_fingerprint,
            final_state_fingerprint=run_result.final_state_fingerprint,
            content_fingerprints=content_fingerprints,
        )

        return HistoricalBacktestAuditResult(
            run_manifest=run_manifest,
            source_run_fingerprint=source_run_fingerprint,
            decision_interval=run_result.decision_interval,
            initial_state=run_result.initial_state,
            final_state=run_result.final_state,
            initial_state_fingerprint=run_result.initial_state_fingerprint,
            final_state_fingerprint=run_result.final_state_fingerprint,
            initial_equity=scalars.initial_equity,
            final_equity=scalars.final_equity,
            period_pnl=scalars.period_pnl,
            session_transitions=transition_audits,
            signal_provenance=signal_provenance,
            ranking_cycles=ranking_cycles,
            ranking_candidates=ranking_candidates,
            allocation_cycles=allocation_cycles,
            allocation_candidates=allocation_candidates,
            execution_provenance=execution_provenance,
            entries=entries,
            exits=exits,
            trades=trades,
            rejections=rejections,
            cash_ledger=cash_ledger,
            settlement_ledger=settlement_ledger,
            session_pnl=session_pnl,
            equity_curve=equity_curve,
            cost_summary=cost_summary,
            exit_reason_summary=exit_reason_summary,
            summary=summary,
            audit_summary=audit_summary,
            content_fingerprints=content_fingerprints,
            result_fingerprint=result_fingerprint,
        )


__all__ = ["HistoricalBacktestResultService"]
