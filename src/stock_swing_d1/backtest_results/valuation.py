"""Phase 15D.3B: valuation, equity, and P&L projection from caller evidence.

Valuation snapshots are immutable caller-supplied evidence bound to the run
manifest's market-data artifact. This module never queries a market-data
provider, reprices an execution, reruns any upstream owner, invokes a Phase
13 transition, or force-liquidates a final position; it only projects
Phase 13/15A/15D.3A authoritative facts against caller-supplied marks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from stock_swing_d1.backtest_results.derivation import (
    _project_open_trade_links,
    project_cash_ledger,
    project_closed_trades,
)
from stock_swing_d1.backtest_results.errors import (
    HistoricalBacktestResultValidationError,
)
from stock_swing_d1.backtest_results.models import (
    HistoricalBacktestEquityRow,
    HistoricalBacktestRunManifest,
    HistoricalBacktestSessionPnl,
    HistoricalBacktestValuationSnapshot,
    HistoricalOpenTradeRecord,
)
from stock_swing_d1.backtest_results.source_validation import (
    _build_validated_source_context,
    _ValidatedSourceRunContext,
)
from stock_swing_d1.backtester.models import HistoricalBacktestRunResult
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    add_exact_decimal,
    exact_decimal_times_int,
    subtract_exact_decimal,
)
from stock_swing_d1.portfolio.portfolio_events import PortfolioLedgerEventType
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.valuation import (
    PortfolioValuationError,
    PortfolioValuationPolicy,
    PortfolioValuationResult,
    value_portfolio_at_allocation_boundary,
)


_ZERO = Decimal("0")
# Task 5C-A: settled cash, pending receivable value, open-position market
# value and equity come from the one shared upstream valuation primitive.
# Phase 15D supplies its own independently validated evidence and keeps every
# audit responsibility it already had (snapshot ownership, artifact binding,
# fingerprints, unrealized/realized P&L, dividend attribution, reconciliation);
# it simply no longer carries a second copy of the equity formula.
_FROZEN_VALUATION_POLICY = PortfolioValuationPolicy()
_VALUATION_ERROR_CODES = {
    "VALUATION_MARK_COVERAGE_MISMATCH": "VALUATION_MARK_COVERAGE_MISMATCH",
    "VALUATION_ARTIFACT_MISMATCH": "VALUATION_ARTIFACT_MISMATCH",
    "VALUATION_SESSION_MISMATCH": "VALUATION_SNAPSHOT_MISMATCH",
    "VALUATION_INPUT_INVALID": "VALUATION_ACCOUNTING_MISMATCH",
    "VALUATION_INVARIANT_VIOLATION": "VALUATION_ACCOUNTING_MISMATCH",
}
_APPLIED_LEDGER_EVENT_TYPES = (
    PortfolioLedgerEventType.BUY_APPLIED,
    PortfolioLedgerEventType.SELL_APPLIED,
)

# `exact_decimal_times_int`/`subtract_exact_decimal` now live in the shared
# Phase 13 exact-Decimal utility module (`portfolio_dividend_events.py`),
# alongside `add_exact_decimal`, so Phase 15D and Phase 13 use the exact
# same primitives (never the reverse dependency direction). `_subtract_exact`
# is kept as a local alias purely to avoid renaming every call site below.
_subtract_exact = subtract_exact_decimal


def _exact_sum(values: object) -> Decimal:
    """Sum exactly via repeated `add_exact_decimal`, never Python's `sum`/`+`.

    Equivalent in intent to the built-in `sum(values, Decimal("0"))`, but a
    running total accumulated via ordinary `+` is re-rounded to the ambient
    Decimal context's precision at every intermediate step (OD-6.7's "no
    free ambient-context precision parameter" is not unique to dividend
    cash: any multi-term Decimal accumulation is exposed to it once a
    single term, or the running total itself, needs more significant
    digits than the context allows).

    Task 5C-A moved this package's own pending/market accumulations into the
    shared valuation primitive, but this accumulator stays: it is the frozen
    Phase 16B.2 public export ``backtest_results.sum_exact_decimal``, whose
    identity downstream research code and its pinned test depend on.
    """

    total = _ZERO
    for value in values:
        total = add_exact_decimal(total, value)
    return total


def _shared_valuation(
    *,
    state: PortfolioState,
    session: date | None,
    marks: tuple[object, ...],
) -> PortfolioValuationResult:
    """Value one authoritative state through the single shared primitive.

    The shared owner's narrow typed errors are translated back into this
    package's existing public error vocabulary, so the Phase 15D error
    surface is unchanged by the Task 5C-A extraction.
    """

    try:
        return value_portfolio_at_allocation_boundary(
            state=state,
            session=session,
            marks=marks,
            policy=_FROZEN_VALUATION_POLICY,
        )
    except PortfolioValuationError as error:
        raise HistoricalBacktestResultValidationError(
            _VALUATION_ERROR_CODES.get(
                error.code, "VALUATION_ACCOUNTING_MISMATCH"
            ),
            "the shared allocation-boundary valuation rejected this evidence",
        ) from error


def _dividend_income_by_session(
    run_result: HistoricalBacktestRunResult,
) -> dict[date, Decimal]:
    """OD-20.6/OD-21.12: exact per-session sum of authoritative strategy
    ``DIVIDEND_APPLIED`` cash-ledger rows.

    This is the sole normative portfolio-scope dividend source (OD-21.9,
    Section 5 of the Slice 12 governance): never derived from
    ``HistoricalClosedTradeRecord.ordinary_dividend_income`` or
    ``trade_total_pnl``, since those omit rows attributed to a trade
    still open at run end or to a ``PARTIAL_PRE_RUN_UNKNOWN`` trade. For
    a non-dividend-aware run this is structurally always empty (Phase 13
    cannot emit ``DIVIDEND_APPLIED`` rows outside dividend-aware mode),
    not merely coincidentally zero.
    """

    totals: dict[date, Decimal] = {}
    for row in project_cash_ledger(run_result):
        if row.ledger_event_type is not PortfolioLedgerEventType.DIVIDEND_APPLIED:
            continue
        totals[row.session] = add_exact_decimal(
            totals.get(row.session, _ZERO), row.settled_cash_delta
        )
    return totals


def _require_exact_valuation_snapshot_tuple(
    valuation_snapshots: object,
) -> tuple[HistoricalBacktestValuationSnapshot, ...]:
    """Prove the outer container is exactly the frozen evidence contract.

    The Phase 15D.3B public contract is exactly
    ``tuple[HistoricalBacktestValuationSnapshot, ...]``. This never coerces
    a list, generator, set/frozenset, mapping, or any other iterable via
    ``tuple(...)``; any non-exact container or member is rejected outright,
    before any manifest or source-run validation runs.
    """

    if type(valuation_snapshots) is not tuple:
        raise HistoricalBacktestResultValidationError(
            "VALUATION_SNAPSHOT_MISMATCH",
            "valuation_snapshots must be exactly a tuple of "
            "HistoricalBacktestValuationSnapshot, not a list, generator, "
            "set, mapping, or other iterable",
        )
    for item in valuation_snapshots:
        if type(item) is not HistoricalBacktestValuationSnapshot:
            raise HistoricalBacktestResultValidationError(
                "VALUATION_SNAPSHOT_MISMATCH",
                "valuation_snapshots must contain only exact "
                "HistoricalBacktestValuationSnapshot members",
            )
    return valuation_snapshots


def _canonical_valuation_snapshot(
    value: HistoricalBacktestValuationSnapshot,
) -> HistoricalBacktestValuationSnapshot:
    rebuilt = HistoricalBacktestValuationSnapshot.model_validate(
        value.model_dump(mode="python")
    )
    if rebuilt != value:
        raise HistoricalBacktestResultValidationError(
            "VALUATION_SNAPSHOT_MISMATCH",
            "a valuation snapshot must already be canonical",
        )
    return rebuilt


@dataclass(frozen=True, slots=True)
class _ValuationContext:
    """Validated Phase 13/15A context plus its exact evidenced marks."""

    source: _ValidatedSourceRunContext
    snapshot_by_session: dict[date, HistoricalBacktestValuationSnapshot]
    initial_snapshot: HistoricalBacktestValuationSnapshot | None


def _validate_valuation_snapshots(
    *,
    run_result: HistoricalBacktestRunResult,
    run_manifest: HistoricalBacktestRunManifest,
    valuation_snapshots: tuple[HistoricalBacktestValuationSnapshot, ...],
) -> _ValuationContext:
    """Prove caller-supplied marks exactly cover the required session set.

    The outer container contract is validated first and exactly, with no
    coercion of any other iterable, before the manifest or source run is
    touched. Required sessions are exactly every processed session plus,
    only when `initial_state` has open positions, `initial_state.as_of_session`.
    No missing snapshot, no extra snapshot, no non-ascending or duplicate
    session is tolerated; malformed input is rejected, never resorted.
    """

    exact_snapshots = _require_exact_valuation_snapshot_tuple(valuation_snapshots)

    if type(run_manifest) is not HistoricalBacktestRunManifest:
        raise HistoricalBacktestResultValidationError(
            "VALUATION_ARTIFACT_MISMATCH",
            "run_manifest must be exactly HistoricalBacktestRunManifest",
        )
    source = _build_validated_source_context(
        run_result=run_result, run_manifest=run_manifest
    )
    canonical_snapshots = tuple(
        _canonical_valuation_snapshot(item) for item in exact_snapshots
    )

    sessions = tuple(item.session for item in canonical_snapshots)
    if len(set(sessions)) != len(sessions):
        raise HistoricalBacktestResultValidationError(
            "VALUATION_SNAPSHOT_MISMATCH",
            "valuation snapshot sessions must be unique",
        )
    if sessions != tuple(sorted(sessions)):
        raise HistoricalBacktestResultValidationError(
            "VALUATION_SNAPSHOT_MISMATCH",
            "valuation snapshots must be strictly session-ascending",
        )

    initial_state = source.run_result.initial_state
    session_results = source.run_result.session_results
    processed_sessions = tuple(session.session for session in session_results)
    requires_initial_snapshot = bool(initial_state.open_positions)
    if requires_initial_snapshot and initial_state.as_of_session is None:
        raise HistoricalBacktestResultValidationError(
            "VALUATION_SNAPSHOT_MISMATCH",
            "an initial state with open positions must carry a completed as_of_session",
        )

    required_sessions = set(processed_sessions)
    if requires_initial_snapshot:
        required_sessions.add(initial_state.as_of_session)
    supplied_sessions = set(sessions)
    if supplied_sessions != required_sessions:
        raise HistoricalBacktestResultValidationError(
            "VALUATION_SNAPSHOT_MISMATCH",
            "valuation snapshots do not exactly cover the required session set",
        )

    snapshot_by_session = {
        item.session: item for item in canonical_snapshots
    }
    initial_snapshot = (
        snapshot_by_session[initial_state.as_of_session]
        if requires_initial_snapshot
        else None
    )

    market_data_ref = run_manifest.market_data_artifact_ref

    def _require_exact_mark_coverage(
        snapshot: HistoricalBacktestValuationSnapshot,
        expected_security_ids: set[str],
    ) -> None:
        actual_security_ids = {mark.security_id for mark in snapshot.marks}
        if actual_security_ids != expected_security_ids:
            raise HistoricalBacktestResultValidationError(
                "VALUATION_MARK_COVERAGE_MISMATCH",
                "valuation mark security coverage does not equal the exact "
                "open-position set for this session",
            )
        for mark in snapshot.marks:
            if mark.source_artifact_ref != market_data_ref:
                raise HistoricalBacktestResultValidationError(
                    "VALUATION_ARTIFACT_MISMATCH",
                    "valuation mark source_artifact_ref does not match the "
                    "run manifest market-data artifact",
                )

    if initial_snapshot is not None:
        _require_exact_mark_coverage(
            initial_snapshot,
            {position.asset_id for position in initial_state.open_positions},
        )
    for session in session_results:
        snapshot = snapshot_by_session[session.session]
        _require_exact_mark_coverage(
            snapshot,
            {
                position.asset_id
                for position in session.authoritative_state.open_positions
            },
        )

    return _ValuationContext(
        source=source,
        snapshot_by_session=snapshot_by_session,
        initial_snapshot=initial_snapshot,
    )


@dataclass(frozen=True, slots=True)
class _InitialValuation:
    initial_equity: Decimal
    initial_unrealized_pnl: Decimal


def _mark_for(
    marks_by_security: dict[str, object], security_id: str
) -> object:
    mark = marks_by_security.get(security_id)
    if mark is None:
        raise HistoricalBacktestResultValidationError(
            "VALUATION_ACCOUNTING_MISMATCH",
            "an open position lacks its exact evidenced valuation mark",
        )
    return mark


def _compute_initial_valuation(
    *,
    initial_state: PortfolioState,
    initial_snapshot: HistoricalBacktestValuationSnapshot | None,
) -> _InitialValuation:
    """Value the run's starting point without recomputing cost basis.

    A cash-only initial state has zero market value and zero unrealized
    P&L by definition; pending settlements are valued at exact face amount.
    Every multi-term accumulation here (`pending_value` across settlements,
    `market_value`/`unrealized_pnl` across positions, and the final
    summation into `initial_equity`) uses exact helpers rather than
    Python's `sum`/`+`: `settled_cash` can carry Gate3 scale-38
    dividend-cash precision once a carried-in position has received a
    prior dividend, and OD-6.7's "no free ambient-context precision
    parameter" applies to any Decimal accumulation, not only dividend
    cash -- a running total accumulated via ordinary `+`/`sum` is
    re-rounded to the ambient context's precision at every step.
    """

    if initial_state.open_positions and initial_snapshot is None:
        raise HistoricalBacktestResultValidationError(
            "VALUATION_SNAPSHOT_MISMATCH",
            "an initial state with open positions requires its exact snapshot",
        )
    marks = () if initial_snapshot is None else initial_snapshot.marks
    valuation = _shared_valuation(
        state=initial_state,
        session=initial_state.as_of_session,
        marks=marks,
    )
    if not initial_state.open_positions:
        return _InitialValuation(
            initial_equity=valuation.portfolio_equity,
            initial_unrealized_pnl=_ZERO,
        )

    # Unrealized P&L stays Phase 15D-owned: it needs authoritative Phase 13
    # cost basis, which is deliberately absent from the shared market-value
    # primitive.
    marks_by_security = {mark.security_id: mark for mark in marks}
    unrealized_pnl = _ZERO
    for position in initial_state.open_positions:
        mark = _mark_for(marks_by_security, position.asset_id)
        unrealized_pnl = add_exact_decimal(
            unrealized_pnl,
            _subtract_exact(
                exact_decimal_times_int(mark.close, position.quantity),
                position.cost_basis,
            ),
        )
    return _InitialValuation(
        initial_equity=valuation.portfolio_equity,
        initial_unrealized_pnl=unrealized_pnl,
    )


@dataclass(frozen=True, slots=True)
class _SessionValuationRow:
    session: date
    state_hash: str
    settled_cash: Decimal
    pending_receivable_value: Decimal
    open_position_market_value: Decimal
    equity: Decimal
    realized_pnl_this_session: Decimal
    cumulative_realized_pnl: Decimal
    unrealized_pnl: Decimal
    execution_cost_this_session: Decimal
    cumulative_execution_cost: Decimal
    period_pnl: Decimal
    ordinary_dividend_income_this_session: Decimal | None
    cumulative_ordinary_dividend_income: Decimal | None


def _compute_session_valuation_rows(
    valuation_context: _ValuationContext,
) -> tuple[_SessionValuationRow, ...]:
    """Compute one exact row per processed session, proving the P&L identity.

    Realized P&L is attributed to the APPLIED SELL's own session (via the
    already-approved Phase 15D.3A closed-trade projection), never to its
    settlement session. Execution costs are reported only; they are never
    subtracted a second time from realized/unrealized P&L or equity, since
    they are already embedded in Phase 13 cost basis and net proceeds.

    OD-21.11 (Slice 12): ordinary-dividend cash is a third, separate
    source of period P&L alongside realized and unrealized P&L. `equity`
    already includes it (Phase 13 credits dividend cash directly into
    `state.settled_cash` at OD-12.1's step D), so the bottom-up identity
    must add the same exact per-session cash-ledger sum
    (`_dividend_income_by_session`, never a closed-trade aggregate) for
    the two sides to reconcile. Both sides of the reconciliation are
    computed with the same exact, ambient-Decimal-context-independent
    arithmetic (`add_exact_decimal`/`_subtract_exact`), since the dividend
    term can require more significant digits than the default context
    allows (OD-6.7).
    """

    run_result = valuation_context.source.run_result
    initial_valuation = _compute_initial_valuation(
        initial_state=run_result.initial_state,
        initial_snapshot=valuation_context.initial_snapshot,
    )
    dividend_aware = run_result.dividend_run_evidence is not None
    dividend_by_session = _dividend_income_by_session(run_result)

    realized_by_session: dict[date, Decimal] = {}
    for trade in project_closed_trades(run_result):
        realized_by_session[trade.exit_session] = add_exact_decimal(
            realized_by_session.get(trade.exit_session, _ZERO),
            trade.realized_pnl,
        )

    rows: list[_SessionValuationRow] = []
    cumulative_realized = _ZERO
    cumulative_cost = _ZERO
    cumulative_dividend = _ZERO
    for session in run_result.session_results:
        snapshot = valuation_context.snapshot_by_session[session.session]
        marks_by_security = {
            mark.security_id: mark for mark in snapshot.marks
        }
        state = session.authoritative_state

        # state.settled_cash can carry Gate3 scale-38 dividend-cash
        # precision (Slice 12, OD-6.7) once an in-run dividend has been
        # applied; the shared primitive accumulates every term with the same
        # exact helpers rather than naive `+`/`*`, so nothing is rounded away.
        valuation = _shared_valuation(
            state=state, session=session.session, marks=snapshot.marks
        )
        pending_value = valuation.pending_receivable_value
        market_value = valuation.open_position_market_value
        equity = valuation.portfolio_equity

        # Unrealized P&L remains Phase 15D-owned (it needs cost basis).
        unrealized_pnl = _ZERO
        for position in state.open_positions:
            mark = _mark_for(marks_by_security, position.asset_id)
            unrealized_pnl = add_exact_decimal(
                unrealized_pnl,
                _subtract_exact(
                    exact_decimal_times_int(mark.close, position.quantity),
                    position.cost_basis,
                ),
            )

        realized_this_session = realized_by_session.get(session.session, _ZERO)
        cumulative_realized = add_exact_decimal(
            cumulative_realized, realized_this_session
        )

        dividend_this_session = dividend_by_session.get(session.session, _ZERO)
        cumulative_dividend = add_exact_decimal(
            cumulative_dividend, dividend_this_session
        )

        # Execution cost is reported only and is not a term of the OD-21.11
        # identity (period_pnl never subtracts it), so it is intentionally
        # left outside this narrow exact-arithmetic audit's scope.
        execution_cost_this_session = sum(
            (
                entry.execution_cost
                for entry in session.state_transition_result.ledger_entries
                if entry.event_type in _APPLIED_LEDGER_EVENT_TYPES
                and entry.execution_cost is not None
            ),
            _ZERO,
        )
        cumulative_cost += execution_cost_this_session

        period_pnl = _subtract_exact(equity, initial_valuation.initial_equity)
        reconciled_period_pnl = _subtract_exact(
            add_exact_decimal(
                add_exact_decimal(cumulative_realized, cumulative_dividend),
                unrealized_pnl,
            ),
            initial_valuation.initial_unrealized_pnl,
        )
        if period_pnl != reconciled_period_pnl:
            raise HistoricalBacktestResultValidationError(
                "PNL_RECONCILIATION_MISMATCH",
                "period_pnl does not reconcile to realized plus ordinary "
                "dividend plus unrealized P&L",
            )

        rows.append(
            _SessionValuationRow(
                session=session.session,
                state_hash=session.state_transition_result.state_hash_after,
                settled_cash=state.settled_cash,
                pending_receivable_value=pending_value,
                open_position_market_value=market_value,
                equity=equity,
                realized_pnl_this_session=realized_this_session,
                cumulative_realized_pnl=cumulative_realized,
                unrealized_pnl=unrealized_pnl,
                execution_cost_this_session=execution_cost_this_session,
                cumulative_execution_cost=cumulative_cost,
                period_pnl=period_pnl,
                ordinary_dividend_income_this_session=(
                    dividend_this_session if dividend_aware else None
                ),
                cumulative_ordinary_dividend_income=(
                    cumulative_dividend if dividend_aware else None
                ),
            )
        )
    return tuple(rows)


def project_session_pnl(
    *,
    run_result: HistoricalBacktestRunResult,
    run_manifest: HistoricalBacktestRunManifest,
    valuation_snapshots: tuple[HistoricalBacktestValuationSnapshot, ...],
) -> tuple[HistoricalBacktestSessionPnl, ...]:
    """Project one exact session P&L row per processed session, in order."""

    valuation_context = _validate_valuation_snapshots(
        run_result=run_result,
        run_manifest=run_manifest,
        valuation_snapshots=valuation_snapshots,
    )
    rows = _compute_session_valuation_rows(valuation_context)
    dividend_aware = (
        valuation_context.source.run_result.dividend_run_evidence is not None
    )
    schema_version = (
        "historical_backtest_session_pnl.v0.2"
        if dividend_aware
        else "historical_backtest_session_pnl.v0.1"
    )
    return tuple(
        HistoricalBacktestSessionPnl(
            schema_version=schema_version,
            session=row.session,
            realized_pnl_this_session=row.realized_pnl_this_session,
            cumulative_realized_pnl=row.cumulative_realized_pnl,
            unrealized_pnl=row.unrealized_pnl,
            execution_cost_this_session=row.execution_cost_this_session,
            cumulative_execution_cost=row.cumulative_execution_cost,
            period_pnl=row.period_pnl,
            ordinary_dividend_income_this_session=(
                row.ordinary_dividend_income_this_session
            ),
            cumulative_ordinary_dividend_income=(
                row.cumulative_ordinary_dividend_income
            ),
        )
        for row in rows
    )


def project_equity_curve(
    *,
    run_result: HistoricalBacktestRunResult,
    run_manifest: HistoricalBacktestRunManifest,
    valuation_snapshots: tuple[HistoricalBacktestValuationSnapshot, ...],
) -> tuple[HistoricalBacktestEquityRow, ...]:
    """Project one exact equity-curve row per processed session, in order."""

    valuation_context = _validate_valuation_snapshots(
        run_result=run_result,
        run_manifest=run_manifest,
        valuation_snapshots=valuation_snapshots,
    )
    rows = _compute_session_valuation_rows(valuation_context)
    dividend_aware = (
        valuation_context.source.run_result.dividend_run_evidence is not None
    )
    schema_version = (
        "historical_backtest_equity_row.v0.2"
        if dividend_aware
        else "historical_backtest_equity_row.v0.1"
    )
    return tuple(
        HistoricalBacktestEquityRow(
            schema_version=schema_version,
            session=row.session,
            state_hash=row.state_hash,
            settled_cash=row.settled_cash,
            pending_receivable_value=row.pending_receivable_value,
            open_position_market_value=row.open_position_market_value,
            equity=row.equity,
            realized_pnl_this_session=row.realized_pnl_this_session,
            cumulative_realized_pnl=row.cumulative_realized_pnl,
            unrealized_pnl=row.unrealized_pnl,
            execution_cost_this_session=row.execution_cost_this_session,
            cumulative_execution_cost=row.cumulative_execution_cost,
            period_pnl=row.period_pnl,
            ordinary_dividend_income_this_session=(
                row.ordinary_dividend_income_this_session
            ),
            cumulative_ordinary_dividend_income=(
                row.cumulative_ordinary_dividend_income
            ),
        )
        for row in rows
    )


def project_open_trades(
    *,
    run_result: HistoricalBacktestRunResult,
    run_manifest: HistoricalBacktestRunManifest,
    valuation_snapshots: tuple[HistoricalBacktestValuationSnapshot, ...],
) -> tuple[HistoricalOpenTradeRecord, ...]:
    """Value every `final_state.open_positions` item without liquidating it.

    Entry-side facts come from the private Phase 15D.3A open-trade handoff
    (`_project_open_trade_links`), which itself reads authoritative Phase 13
    `OpenPosition` values; only the final mark is new evidence here.
    """

    valuation_context = _validate_valuation_snapshots(
        run_result=run_result,
        run_manifest=run_manifest,
        valuation_snapshots=valuation_snapshots,
    )
    links = _project_open_trade_links(run_result)
    if not links:
        return ()

    source = valuation_context.source
    session_results = source.run_result.session_results
    if session_results:
        final_mark_session = session_results[-1].session
        final_snapshot = valuation_context.snapshot_by_session.get(
            final_mark_session
        )
    else:
        final_mark_session = source.run_result.initial_state.as_of_session
        final_snapshot = valuation_context.initial_snapshot
    if final_mark_session is None or final_snapshot is None:
        raise HistoricalBacktestResultValidationError(
            "VALUATION_SNAPSHOT_MISMATCH",
            "final open positions require an exact final valuation snapshot",
        )

    marks_by_security = {
        mark.security_id: mark for mark in final_snapshot.marks
    }
    rows = []
    for link in sorted(
        links, key=lambda item: (item.entry_session, item.entry_execution_id)
    ):
        mark = _mark_for(marks_by_security, link.security_id)
        # Residual A (exact-arithmetic hardening): mark.close carries no
        # enforced significant-digit bound, so quantity * mark.close and
        # the adjacent unrealized_pnl subtraction must use the shared
        # exact helpers, not ambient-context Decimal `*`/`-`.
        final_market_value = exact_decimal_times_int(mark.close, link.quantity)
        rows.append(
            HistoricalOpenTradeRecord(
                trade_id=link.entry_execution_id,
                security_id=link.security_id,
                quantity=link.quantity,
                carried_in=link.carried_in,
                entry_execution_id=link.entry_execution_id,
                entry_session=link.entry_session,
                entry_fill_price=link.entry_fill_price,
                entry_execution_cost=link.entry_execution_cost,
                entry_cost_basis=link.entry_cost_basis,
                final_mark_session=final_mark_session,
                final_mark_price=mark.close,
                final_market_value=final_market_value,
                unrealized_pnl=subtract_exact_decimal(
                    final_market_value, link.entry_cost_basis
                ),
            )
        )
    return tuple(rows)


__all__ = [
    "project_equity_curve",
    "project_open_trades",
    "project_session_pnl",
]
