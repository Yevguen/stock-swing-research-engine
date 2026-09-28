"""Phase 15D.3C: private cross-component reconciliation and summary builders.

Every function here consumes already-approved Phase 15D.3A/15D.3B outputs
and the raw, already-validated `run_result`/`run_manifest`; none re-runs an
upstream economic owner, reprices an execution, recomputes a cost-policy
formula, or touches Phase 13 state. These proofs exist to catch an
*assembly*-level defect in `service.py` (e.g. a mismatched valuation snapshot
between calls) — they are not a second source of upstream truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stock_swing_d1.backtest_results.errors import (
    HistoricalBacktestResultValidationError,
)
from stock_swing_d1.backtest_results.models import (
    ExecutionApplicationStatus,
    HistoricalBacktestCostSummary,
    HistoricalBacktestEntryRecord,
    HistoricalBacktestEquityRow,
    HistoricalBacktestExecutionProvenance,
    HistoricalBacktestExitRecord,
    HistoricalBacktestExitReasonRow,
    HistoricalBacktestExitReasonSummary,
    HistoricalBacktestRejectionRecord,
    HistoricalBacktestRunManifest,
    HistoricalBacktestSessionPnl,
    HistoricalBacktestValuationPolicy,
    HistoricalBacktestValuationSnapshot,
    HistoricalClosedTradeRecord,
    HistoricalExitReason,
    HistoricalOpenTradeRecord,
    HistoricalRejectionStage,
    HistoricalTradeRecord,
    build_valuation_policy_ref,
)
from stock_swing_d1.backtest_results.valuation import (
    _compute_initial_valuation,
    _exact_sum,
    _subtract_exact,
    _validate_valuation_snapshots,
)
from stock_swing_d1.backtester.models import HistoricalBacktestRunResult
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioLedgerEventType,
)


_ZERO = Decimal("0")


def _prove_valuation_policy_consistency(
    run_manifest: HistoricalBacktestRunManifest,
) -> None:
    """Prove the manifest's valuation policy is exactly the frozen v0.1 policy.

    Phase 15D.3B's valuation-snapshot validation proves mark coverage and
    the market-data artifact reference, but never checks
    `run_manifest.valuation_policy_ref` itself. This closes that gap without
    touching the frozen valuation model — it only recomputes the canonical
    ref from the frozen default policy and compares by equality.
    """

    expected = build_valuation_policy_ref(HistoricalBacktestValuationPolicy())
    if run_manifest.valuation_policy_ref != expected:
        raise HistoricalBacktestResultValidationError(
            "VALUATION_ARTIFACT_MISMATCH",
            "run_manifest.valuation_policy_ref does not match the frozen "
            "Phase 15D valuation policy",
        )


def _prove_trade_linkage(
    *,
    run_result: HistoricalBacktestRunResult,
    entries: tuple[HistoricalBacktestEntryRecord, ...],
    exits: tuple[HistoricalBacktestExitRecord, ...],
    closed_trades: tuple[HistoricalClosedTradeRecord, ...],
    open_trades: tuple[HistoricalOpenTradeRecord, ...],
) -> None:
    """Prove the assembled entries/exits/trades collections are consistent.

    Each individual projection is already correct in isolation (15D.3A); this
    proves the *assembly* did not introduce a new inconsistency (e.g. one
    projection run against a different `run_result`).
    """

    entry_ids = tuple(entry.entry_execution_id for entry in entries)
    if len(set(entry_ids)) != len(entry_ids):
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "entries must be unique by entry_execution_id",
        )
    exit_ids = tuple(exit_.exit_execution_id for exit_ in exits)
    if len(set(exit_ids)) != len(exit_ids):
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "exits must be unique by exit_execution_id",
        )

    # Every exit maps to exactly one closed trade; the trade's own linkage
    # to its entry must agree with the exit's linkage to that same entry.
    exits_by_id = {exit_.exit_execution_id: exit_ for exit_ in exits}
    closed_trade_exit_ids = tuple(trade.exit_execution_id for trade in closed_trades)
    if len(set(closed_trade_exit_ids)) != len(closed_trade_exit_ids):
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "closed trades must be unique by exit_execution_id",
        )
    if set(closed_trade_exit_ids) != set(exit_ids):
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "every exit must map to exactly one closed trade, and vice versa",
        )
    for trade in closed_trades:
        exit_row = exits_by_id[trade.exit_execution_id]
        if exit_row.entry_execution_id != trade.entry_execution_id:
            raise HistoricalBacktestResultValidationError(
                "TRADE_RECONCILIATION_MISMATCH",
                "a closed trade's entry_execution_id disagrees with its "
                "linked exit's entry_execution_id",
            )

    # Every final Phase 13 open position maps to exactly one open trade.
    final_open_asset_ids = {
        position.asset_id for position in run_result.final_state.open_positions
    }
    open_trade_asset_ids = tuple(trade.security_id for trade in open_trades)
    if len(set(open_trade_asset_ids)) != len(open_trade_asset_ids):
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "open trades must be unique by security_id",
        )
    if set(open_trade_asset_ids) != final_open_asset_ids:
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "every final open position must map to exactly one open trade, "
            "and vice versa",
        )

    # No trade_id duplicate across closed/open.
    closed_trade_ids = {trade.trade_id for trade in closed_trades}
    open_trade_ids = {trade.trade_id for trade in open_trades}
    if len(closed_trade_ids) != len(closed_trades):
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "closed trades must be unique by trade_id",
        )
    if len(open_trade_ids) != len(open_trades):
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "open trades must be unique by trade_id",
        )
    if closed_trade_ids & open_trade_ids:
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "a trade_id cannot appear in both closed and open trades",
        )

    # Every non-carried-in entry belongs to exactly one non-carried-in trade;
    # a carried-in trade must never require a synthetic entries row (15D.3A
    # Rule 8), so it must never appear in `entries` at all.
    non_carried_in_trade_entry_ids = {
        trade.entry_execution_id
        for trade in (*closed_trades, *open_trades)
        if not trade.carried_in
    }
    if set(entry_ids) != non_carried_in_trade_entry_ids:
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "every entry must belong to exactly one non-carried-in trade, "
            "and every non-carried-in trade must have its entries row",
        )
    carried_in_trade_entry_ids = {
        trade.entry_execution_id
        for trade in (*closed_trades, *open_trades)
        if trade.carried_in
    }
    if carried_in_trade_entry_ids & set(entry_ids):
        raise HistoricalBacktestResultValidationError(
            "TRADE_RECONCILIATION_MISMATCH",
            "a carried-in trade must not have a synthetic entries row",
        )

    if not run_result.session_results and (closed_trades or open_trades):
        # An open trade can legitimately exist with zero processed sessions
        # (a carried-in position that was never touched); a closed trade
        # cannot, since closing always requires an APPLIED SELL inside a
        # processed session.
        if closed_trades:
            raise HistoricalBacktestResultValidationError(
                "TRADE_RECONCILIATION_MISMATCH",
                "a zero-session run cannot contain a closed trade",
            )


def _assemble_trades(
    closed_trades: tuple[HistoricalClosedTradeRecord, ...],
    open_trades: tuple[HistoricalOpenTradeRecord, ...],
) -> tuple[HistoricalTradeRecord, ...]:
    """Merge closed and open trades in the frozen (entry_session, trade_id) order.

    The frozen `HistoricalBacktestAuditResult` model imposes no ordering
    constraint on `trades`; this ordering is a Phase 15D.3C decision, applied
    only to this new merged collection — no upstream projection is resorted.
    """

    return tuple(
        sorted(
            (*closed_trades, *open_trades),
            key=lambda trade: (trade.entry_session, trade.trade_id),
        )
    )


def _prove_and_build_cost_summary(
    *,
    run_result: HistoricalBacktestRunResult,
    entries: tuple[HistoricalBacktestEntryRecord, ...],
    exits: tuple[HistoricalBacktestExitRecord, ...],
    execution_provenance: tuple[HistoricalBacktestExecutionProvenance, ...],
    final_cumulative_execution_cost: Decimal | None,
) -> HistoricalBacktestCostSummary:
    """Build the cost summary and reconcile it against every independent source.

    Sources: (A) APPLIED-only execution provenance, separated by side;
    (B) entries/exits counts and totals; (C) the raw Phase 13
    `PortfolioLedgerEntry` rows (never the public `cash_ledger` projection,
    which carries no `execution_cost` field); (D) the final cumulative
    execution cost already reported by Phase 15D.3B, when a processed
    session exists. No cost-policy formula is recomputed.
    """

    buy_total = sum((entry.execution_cost for entry in entries), _ZERO)
    sell_total = sum((exit_.execution_cost for exit_ in exits), _ZERO)
    total = buy_total + sell_total

    applied_buy_provenance = tuple(
        row
        for row in execution_provenance
        if row.application_status is ExecutionApplicationStatus.APPLIED
        and row.side is ExecutionSide.BUY
    )
    applied_sell_provenance = tuple(
        row
        for row in execution_provenance
        if row.application_status is ExecutionApplicationStatus.APPLIED
        and row.side is ExecutionSide.SELL
    )
    if len(applied_buy_provenance) != len(entries):
        raise HistoricalBacktestResultValidationError(
            "COST_RECONCILIATION_MISMATCH",
            "entry count disagrees with APPLIED BUY execution provenance",
        )
    if len(applied_sell_provenance) != len(exits):
        raise HistoricalBacktestResultValidationError(
            "COST_RECONCILIATION_MISMATCH",
            "exit count disagrees with APPLIED SELL execution provenance",
        )
    if sum((row.execution_cost for row in applied_buy_provenance), _ZERO) != buy_total:
        raise HistoricalBacktestResultValidationError(
            "COST_RECONCILIATION_MISMATCH",
            "buy execution-cost total disagrees with APPLIED execution "
            "provenance",
        )
    if (
        sum((row.execution_cost for row in applied_sell_provenance), _ZERO)
        != sell_total
    ):
        raise HistoricalBacktestResultValidationError(
            "COST_RECONCILIATION_MISMATCH",
            "sell execution-cost total disagrees with APPLIED execution "
            "provenance",
        )

    raw_buy_cost = _ZERO
    raw_sell_cost = _ZERO
    for session in run_result.session_results:
        for entry in session.state_transition_result.ledger_entries:
            if (
                entry.event_type is PortfolioLedgerEventType.BUY_APPLIED
                and entry.execution_cost is not None
            ):
                raw_buy_cost += entry.execution_cost
            elif (
                entry.event_type is PortfolioLedgerEventType.SELL_APPLIED
                and entry.execution_cost is not None
            ):
                raw_sell_cost += entry.execution_cost
    if raw_buy_cost != buy_total or raw_sell_cost != sell_total:
        raise HistoricalBacktestResultValidationError(
            "COST_RECONCILIATION_MISMATCH",
            "execution-cost totals disagree with the raw Phase 13 "
            "BUY_APPLIED/SELL_APPLIED ledger rows",
        )

    if run_result.session_results:
        if final_cumulative_execution_cost != total:
            raise HistoricalBacktestResultValidationError(
                "COST_RECONCILIATION_MISMATCH",
                "total_execution_cost disagrees with the final cumulative "
                "execution cost reported by Phase 15D.3B",
            )
    elif total != _ZERO:
        raise HistoricalBacktestResultValidationError(
            "COST_RECONCILIATION_MISMATCH",
            "a zero-session run must report zero execution cost",
        )

    return HistoricalBacktestCostSummary(
        buy_execution_cost_total=buy_total,
        sell_execution_cost_total=sell_total,
        total_execution_cost=total,
        applied_buy_count=len(entries),
        applied_sell_count=len(exits),
    )


def _build_exit_reason_summary(
    closed_trades: tuple[HistoricalClosedTradeRecord, ...],
) -> HistoricalBacktestExitReasonSummary:
    """Build the exit-reason summary from CLOSED trades only, zero-filled."""

    grouped: dict[HistoricalExitReason, list[HistoricalClosedTradeRecord]] = {
        reason: [] for reason in HistoricalExitReason
    }
    for trade in closed_trades:
        grouped[trade.exit_reason].append(trade)

    rows = tuple(
        HistoricalBacktestExitReasonRow(
            reason=reason,
            exit_count=len(grouped[reason]),
            realized_pnl=sum(
                (trade.realized_pnl for trade in grouped[reason]), _ZERO
            ),
        )
        for reason in HistoricalExitReason
    )
    summary = HistoricalBacktestExitReasonSummary(rows=rows)

    if sum(row.exit_count for row in rows) != len(closed_trades):
        raise HistoricalBacktestResultValidationError(
            "SUMMARY_RECONCILIATION_MISMATCH",
            "exit-reason counts do not sum to the closed-trade count",
        )
    total_realized = sum((row.realized_pnl for row in rows), _ZERO)
    expected_total = sum((trade.realized_pnl for trade in closed_trades), _ZERO)
    if total_realized != expected_total:
        raise HistoricalBacktestResultValidationError(
            "SUMMARY_RECONCILIATION_MISMATCH",
            "exit-reason realized P&L does not sum to the closed-trade total",
        )
    return summary


def _count_rejections_by_stage(
    rejections: tuple[HistoricalBacktestRejectionRecord, ...],
    stage: HistoricalRejectionStage,
) -> int:
    return sum(1 for row in rejections if row.stage is stage)


@dataclass(frozen=True, slots=True)
class _EquityPnlScalars:
    initial_equity: Decimal
    final_equity: Decimal
    final_unrealized_pnl: Decimal
    period_pnl: Decimal
    gross_realized_pnl: Decimal
    ordinary_dividend_income_total: Decimal | None
    final_cumulative_execution_cost: Decimal | None


def _compute_equity_and_pnl_scalars(
    *,
    run_result: HistoricalBacktestRunResult,
    run_manifest: HistoricalBacktestRunManifest,
    valuation_snapshots: tuple[HistoricalBacktestValuationSnapshot, ...],
    equity_curve: tuple[HistoricalBacktestEquityRow, ...],
    session_pnl: tuple[HistoricalBacktestSessionPnl, ...],
    closed_trades: tuple[HistoricalClosedTradeRecord, ...],
) -> _EquityPnlScalars:
    """Compute the top-level equity/P&L scalars and prove their identities.

    For a non-empty run, `initial_equity` is recomputed via the exact
    Phase 15D.3B initial-valuation helper (never re-derived independently)
    and cross-checked against the equity curve's own internal identity;
    `final_equity`/`final_unrealized_pnl`/`period_pnl`/`gross_realized_pnl`
    are read directly from the final equity-curve row and cross-checked
    against the corresponding final session-P&L row. For a zero-session run,
    no session/equity row is synthesized: the same initial-valuation helper
    supplies both the initial *and* final scalars, since Phase 13 already
    proves `initial_state == final_state` in that case.
    """

    valuation_context = _validate_valuation_snapshots(
        run_result=run_result,
        run_manifest=run_manifest,
        valuation_snapshots=valuation_snapshots,
    )
    initial_valuation = _compute_initial_valuation(
        initial_state=run_result.initial_state,
        initial_snapshot=valuation_context.initial_snapshot,
    )
    initial_equity = initial_valuation.initial_equity
    dividend_aware = run_result.dividend_run_evidence is not None

    if not run_result.session_results:
        return _EquityPnlScalars(
            initial_equity=initial_equity,
            final_equity=initial_equity,
            final_unrealized_pnl=initial_valuation.initial_unrealized_pnl,
            period_pnl=_ZERO,
            gross_realized_pnl=_ZERO,
            ordinary_dividend_income_total=(_ZERO if dividend_aware else None),
            final_cumulative_execution_cost=None,
        )

    if not equity_curve or not session_pnl:
        raise HistoricalBacktestResultValidationError(
            "EQUITY_RECONCILIATION_MISMATCH",
            "a non-empty run requires equity-curve and session-P&L rows",
        )
    final_row = equity_curve[-1]
    final_pnl_row = session_pnl[-1]
    if (
        final_row.session != final_pnl_row.session
        or final_row.realized_pnl_this_session
        != final_pnl_row.realized_pnl_this_session
        or final_row.cumulative_realized_pnl != final_pnl_row.cumulative_realized_pnl
        or final_row.unrealized_pnl != final_pnl_row.unrealized_pnl
        or final_row.execution_cost_this_session
        != final_pnl_row.execution_cost_this_session
        or final_row.cumulative_execution_cost
        != final_pnl_row.cumulative_execution_cost
        or final_row.period_pnl != final_pnl_row.period_pnl
        or final_row.ordinary_dividend_income_this_session
        != final_pnl_row.ordinary_dividend_income_this_session
        or final_row.cumulative_ordinary_dividend_income
        != final_pnl_row.cumulative_ordinary_dividend_income
    ):
        raise HistoricalBacktestResultValidationError(
            "EQUITY_RECONCILIATION_MISMATCH",
            "the final equity-curve row disagrees with the final "
            "session-P&L row",
        )

    final_equity = final_row.equity
    period_pnl = _subtract_exact(final_equity, initial_equity)
    if period_pnl != final_row.period_pnl:
        raise HistoricalBacktestResultValidationError(
            "EQUITY_RECONCILIATION_MISMATCH",
            "final_equity minus initial_equity disagrees with the equity "
            "curve's own period_pnl",
        )

    # Exact accumulation (never Python's `sum`), matching how
    # `final_row.cumulative_realized_pnl` was itself accumulated in
    # valuation.py -- both sides of this equality must use identical
    # arithmetic technique, or a spurious mismatch could arise purely from
    # ambient-context rounding differences between the two computations.
    gross_realized_pnl = _exact_sum(trade.realized_pnl for trade in closed_trades)
    if gross_realized_pnl != final_row.cumulative_realized_pnl:
        raise HistoricalBacktestResultValidationError(
            "PNL_RECONCILIATION_MISMATCH",
            "gross_realized_pnl disagrees with the final cumulative "
            "realized P&L",
        )

    return _EquityPnlScalars(
        initial_equity=initial_equity,
        final_equity=final_equity,
        final_unrealized_pnl=final_row.unrealized_pnl,
        period_pnl=period_pnl,
        gross_realized_pnl=gross_realized_pnl,
        ordinary_dividend_income_total=(
            final_row.cumulative_ordinary_dividend_income
        ),
        final_cumulative_execution_cost=final_row.cumulative_execution_cost,
    )


__all__: tuple[str, ...] = ()
