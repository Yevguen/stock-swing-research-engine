"""Read-only provenance projections from validated Phase 15A source runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from stock_swing_d1.backtest_results.errors import (
    HistoricalBacktestResultValidationError,
)
from stock_swing_d1.backtest_results.hashing import (
    compute_dividend_attribution_fingerprint,
    compute_source_payload_fingerprint,
)
from stock_swing_d1.backtest_results.models import (
    DividendAttributionCompleteness,
    ExecutionApplicationStatus,
    ExecutionProvenanceSource,
    HistoricalAllocationCandidateProvenance,
    HistoricalAllocationCycle,
    HistoricalBacktestEntryRecord,
    HistoricalBacktestExecutionProvenance,
    HistoricalBacktestExitRecord,
    HistoricalBacktestRejectionRecord,
    HistoricalBacktestTransitionAudit,
    HistoricalCashLedgerRow,
    HistoricalClosedTradeRecord,
    HistoricalExitReason,
    HistoricalRankingCandidateProvenance,
    HistoricalRankingCycle,
    HistoricalRejectionStage,
    HistoricalSettlementRecord,
    HistoricalSettlementStatus,
    HistoricalSignalProvenance,
)
from stock_swing_d1.backtest_results.source_validation import (
    _build_validated_source_context,
)
from stock_swing_d1.backtester.models import HistoricalBacktestRunResult
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY,
)
from stock_swing_d1.execution.entry.models import EntryExecutionStatus
from stock_swing_d1.portfolio.models import (
    PortfolioCandidateAction,
    RankedPortfolioCandidateDecision,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    add_exact_decimal,
    exact_decimal_times_int,
    subtract_exact_decimal,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioLedgerEventType,
)


@dataclass(frozen=True, slots=True)
class _RoundTripEntryFacts:
    """Task 5C-C: authoritative entry facts of a same-session round trip.

    The unique APPLIED BUY event supplies identity, session, quantity, fill
    price and execution cost; the authoritative ``entry_cost_basis`` comes
    from the matching ``BUY_APPLIED.settled_cash_delta`` by exact sign
    reversal.  ``quantity * fill + cost`` is asserted against it and never
    originates it.  This narrowly replaces the surviving-``OpenPosition``
    authority only for a validated round trip; every other case is
    unchanged.
    """

    entry_execution_id: str
    entry_session: date
    quantity: int
    entry_fill_price: Decimal
    entry_execution_cost: Decimal
    entry_cost_basis: Decimal


def _round_trip_entry_facts(
    *,
    session_result: object,
    buy_event: object,
) -> _RoundTripEntryFacts:
    """Derive one round trip's entry facts from its APPLIED BUY + ledger row."""

    matching_rows = tuple(
        entry
        for entry in session_result.state_transition_result.ledger_entries
        if entry.event_type is PortfolioLedgerEventType.BUY_APPLIED
        and entry.source_event_id == buy_event.execution_id
        and entry.asset_id == buy_event.asset_id
        and entry.session == buy_event.session
    )
    if len(matching_rows) != 1:
        raise HistoricalBacktestResultValidationError(
            "ENTRY_LINKAGE_MISMATCH",
            "a same-session round-trip BUY lacks exactly one matching "
            "BUY_APPLIED ledger row",
        )
    row = matching_rows[0]
    if (
        row.quantity_delta != buy_event.quantity
        or row.fill_price != buy_event.fill_price
        or row.execution_cost != buy_event.execution_cost
        or row.position_quantity_after != buy_event.quantity
    ):
        raise HistoricalBacktestResultValidationError(
            "ENTRY_LINKAGE_MISMATCH",
            "the round-trip BUY_APPLIED ledger row disagrees with its BUY event",
        )
    # Authoritative cost basis: exact sign reversal of the Phase 13 ledger's
    # settled-cash effect (the same exact helper Phase 13 used to write it).
    entry_cost_basis = subtract_exact_decimal(
        Decimal("0"), row.settled_cash_delta
    )
    reconciled = add_exact_decimal(
        exact_decimal_times_int(buy_event.fill_price, buy_event.quantity),
        buy_event.execution_cost,
    )
    if entry_cost_basis != reconciled or entry_cost_basis <= 0:
        raise HistoricalBacktestResultValidationError(
            "ENTRY_COST_BASIS_MISMATCH",
            "ledger-derived round-trip cost basis does not reconcile with "
            "quantity, fill price, and execution cost",
        )
    return _RoundTripEntryFacts(
        entry_execution_id=buy_event.execution_id,
        entry_session=buy_event.session,
        quantity=buy_event.quantity,
        entry_fill_price=buy_event.fill_price,
        entry_execution_cost=buy_event.execution_cost,
        entry_cost_basis=entry_cost_basis,
    )


def _round_trip_buy_event(session_result: object, links: tuple, asset_id: str):
    """The unique APPLIED BUY event of a same-session round trip asset."""

    candidates = tuple(
        link.event
        for link in links
        if link.application_status is ExecutionApplicationStatus.APPLIED
        and link.event.side is ExecutionSide.BUY
        and link.event.asset_id == asset_id
        and link.event.session == session_result.session
    )
    if len(candidates) != 1:
        raise HistoricalBacktestResultValidationError(
            "ENTRY_LINKAGE_MISMATCH",
            "a same-session round trip lacks exactly one APPLIED BUY",
        )
    return candidates[0]


@dataclass(frozen=True, slots=True)
class _HistoricalOpenTradeLink:
    """Internal Phase 15D.3A linkage for one `final_state` open position.

    Carries every entry-side fact a future ``HistoricalOpenTradeRecord``
    needs except its Phase 15D.3B valuation mark. This is not part of the
    frozen ``HistoricalBacktestAuditResult`` schema; it exists only so
    Phase 15D.3A can hand off exact, unrepriced linkage to Phase 15D.3B
    instead of inventing a mark or force-liquidating the position. It is a
    private internal handoff type, not part of the public backtest_results
    API, and must never be exported from `__init__.py`.
    """

    security_id: str
    quantity: int
    carried_in: bool
    entry_execution_id: str
    entry_session: date
    entry_fill_price: Decimal
    entry_execution_cost: Decimal
    entry_cost_basis: Decimal


def project_transition_audits(
    run_result: HistoricalBacktestRunResult,
) -> tuple[HistoricalBacktestTransitionAudit, ...]:
    """Project one exact Phase 13 transition audit per source session."""

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    rows: list[HistoricalBacktestTransitionAudit] = []
    for state_before, session in zip(
        context.states_before,
        context.run_result.session_results,
        strict=True,
    ):
        state_after = session.authoritative_state
        transition = session.state_transition_result
        rows.append(
            HistoricalBacktestTransitionAudit(
                session=session.session,
                decision_time=session.decision_time,
                state_hash_before=transition.state_hash_before,
                state_hash_after=transition.state_hash_after,
                state_version_before=state_before.state_version,
                state_version_after=state_after.state_version,
                settled_cash_before=state_before.settled_cash,
                settled_cash_after=state_after.settled_cash,
                open_position_count_before=len(state_before.open_positions),
                open_position_count_after=len(state_after.open_positions),
                pending_settlement_count_before=len(
                    state_before.pending_settlements
                ),
                pending_settlement_count_after=len(
                    state_after.pending_settlements
                ),
                newly_applied_event_ids=tuple(
                    reference.event_id
                    for reference in transition.newly_applied_events
                ),
                replayed_event_ids=tuple(
                    reference.event_id
                    for reference in transition.replayed_events
                ),
                ledger_entry_count=len(transition.ledger_entries),
            )
        )
    return tuple(rows)


def project_signal_provenance(
    run_result: HistoricalBacktestRunResult,
) -> tuple[HistoricalSignalProvenance, ...]:
    """Copy authoritative Phase 8 signal decisions without recomputation."""

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    return tuple(
        HistoricalSignalProvenance(
            session=session.session,
            decision_time=session.decision_time,
            security_id=decision.security_id,
            action=decision.action,
            planned_entry_session=decision.planned_entry_session,
            adjusted_close=decision.adjusted_close,
            sma20=decision.sma_20,
            sma50=decision.sma_50,
            rsi14=decision.rsi_14,
            atr14=decision.atr_14,
            atr_fraction=decision.atr_fraction,
            universe_eligible=decision.universe_eligible,
            close_above_sma50=decision.close_above_sma50,
            sma20_above_sma50=decision.sma20_above_sma50,
            rsi_above_50=decision.rsi_above_50,
            atr_above_minimum=decision.atr_above_minimum,
            earnings_entry_allowed=decision.earnings_entry_allowed,
            earnings_action=decision.earnings_action,
            source_payload_fingerprint=compute_source_payload_fingerprint(
                source_type="baseline_signal_decision",
                payload=decision,
            ),
        )
        for session in context.run_result.session_results
        for decision in session.signal_decisions
    )


def project_ranking_provenance(
    run_result: HistoricalBacktestRunResult,
) -> tuple[
    tuple[HistoricalRankingCycle, ...],
    tuple[HistoricalRankingCandidateProvenance, ...],
]:
    """Copy validated Phase 14 cycles and candidates in authoritative rank order."""

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    cycles: list[HistoricalRankingCycle] = []
    candidates: list[HistoricalRankingCandidateProvenance] = []
    for session in context.run_result.session_results:
        snapshot = session.ranking_snapshot
        cycles.append(
            HistoricalRankingCycle(
                ranking_session=snapshot.ranking_session,
                decision_time=snapshot.decision_time,
                policy_id=snapshot.policy.policy_id,
                policy_version=snapshot.policy.policy_version,
                policy_fingerprint=snapshot.policy.policy_fingerprint,
                candidate_count=snapshot.candidate_count,
                input_set_fingerprint=snapshot.input_set_fingerprint,
                snapshot_fingerprint=snapshot.snapshot_fingerprint,
            )
        )
        candidates.extend(
            HistoricalRankingCandidateProvenance(
                ranking_session=candidate.ranking_session,
                decision_time=candidate.decision_time,
                security_id=candidate.security_id,
                rank=candidate.rank,
                trend_separation_atr=candidate.trend_separation_atr,
                rsi14=candidate.rsi14,
                sma20=candidate.sma20,
                sma50=candidate.sma50,
                atr14=candidate.atr14,
                candidate_input_fingerprint=candidate.input_fingerprint,
                ranking_snapshot_fingerprint=snapshot.snapshot_fingerprint,
            )
            for candidate in snapshot.ranked_candidates
        )
    return tuple(cycles), tuple(candidates)


def project_allocation_provenance(
    run_result: HistoricalBacktestRunResult,
) -> tuple[
    tuple[HistoricalAllocationCycle, ...],
    tuple[HistoricalAllocationCandidateProvenance, ...],
]:
    """Copy authoritative Phase 12 allocation cycles and candidate decisions."""

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    cycles: list[HistoricalAllocationCycle] = []
    candidates: list[HistoricalAllocationCandidateProvenance] = []
    for session in context.run_result.session_results:
        allocation = session.allocation_decision
        if allocation is None:
            continue
        decisions = allocation.candidate_decisions
        cycles.append(
            HistoricalAllocationCycle(
                allocation_session=allocation.allocation_session,
                decision_time=allocation.decision_time,
                ranking_snapshot_fingerprint=(
                    allocation.ranking_snapshot_fingerprint
                ),
                allocation_policy_fingerprint=(
                    allocation.allocation_policy_ref.policy_fingerprint
                ),
                candidate_count=len(decisions),
                admitted_count=allocation.admitted_count,
                rejected_count=len(decisions) - allocation.admitted_count,
                settled_cash_input=Decimal(str(allocation.starting_cash)),
                portfolio_equity_input=Decimal(
                    str(allocation.starting_portfolio_equity)
                ),
                open_position_count_input=(
                    allocation.starting_open_position_count
                ),
            )
        )
        candidates.extend(
            HistoricalAllocationCandidateProvenance(
                allocation_session=allocation.allocation_session,
                security_id=decision.security_id,
                source_rank=decision.source_rank,
                processing_rank=decision.processing_rank,
                ranking_snapshot_fingerprint=(
                    decision.ranking_snapshot_fingerprint
                ),
                ranking_input_fingerprint=(
                    decision.ranking_input_fingerprint
                ),
                action=decision.action,
                candidate_cash_limit=(
                    None
                    if decision.candidate_cash_limit is None
                    else Decimal(str(decision.candidate_cash_limit))
                ),
                reserved_cash=Decimal(str(decision.reserved_cash)),
                fixed_shares=(
                    None
                    if decision.sized_pending_entry is None
                    else decision.sized_pending_entry.fixed_shares
                ),
                used_slots_before=decision.used_slots_before,
                used_slots_after=decision.used_slots_after,
                unreserved_cash_before=Decimal(
                    str(decision.unreserved_cash_before)
                ),
                unreserved_cash_after=Decimal(
                    str(decision.unreserved_cash_after)
                ),
            )
            for decision in decisions
        )
    return tuple(cycles), tuple(candidates)


def project_execution_provenance(
    run_result: HistoricalBacktestRunResult,
) -> tuple[HistoricalBacktestExecutionProvenance, ...]:
    """Classify each ordered Phase 15A execution exactly once."""

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    rows: list[HistoricalBacktestExecutionProvenance] = []
    for session, links in zip(
        context.run_result.session_results,
        context.execution_links,
        strict=True,
    ):
        for event_order, link in enumerate(links):
            rows.append(
                HistoricalBacktestExecutionProvenance(
                    session=session.session,
                    event_order=event_order,
                    execution_id=link.event.execution_id,
                    source_order_id=link.event.source_order_id,
                    security_id=link.event.asset_id,
                    side=link.event.side,
                    quantity=link.event.quantity,
                    fill_price=link.event.fill_price,
                    execution_cost=link.event.execution_cost,
                    settlement_id=link.event.settlement_id,
                    settlement_session=link.event.settlement_session,
                    application_status=link.application_status,
                    provenance_source=link.provenance_source,
                    source_payload_fingerprint=(
                        compute_source_payload_fingerprint(
                            source_type="portfolio_execution_event",
                            payload=link.event,
                        )
                    ),
                    entry_decision_fingerprint=(
                        None
                        if link.entry_decision is None
                        else compute_source_payload_fingerprint(
                            source_type="entry_execution_decision",
                            payload=link.entry_decision,
                        )
                    ),
                    exit_decision_fingerprint=(
                        None
                        if link.exit_decision is None
                        else compute_source_payload_fingerprint(
                            source_type="open_position_exit_decision",
                            payload=link.exit_decision,
                        )
                    ),
                )
            )
    return tuple(rows)


def project_rejections(
    run_result: HistoricalBacktestRunResult,
) -> tuple[HistoricalBacktestRejectionRecord, ...]:
    """Project exact Phase 12 and Phase 9 rejection outcomes."""

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    rows: list[HistoricalBacktestRejectionRecord] = []
    for session in context.run_result.session_results:
        allocation = session.allocation_decision
        if allocation is not None:
            rows.extend(
                HistoricalBacktestRejectionRecord(
                    decision_session=allocation.allocation_session,
                    security_id=decision.security_id,
                    stage=HistoricalRejectionStage.ALLOCATION,
                    reason=decision.action.value,
                    reserved_cash=Decimal(str(decision.reserved_cash)),
                    source_rank=decision.source_rank,
                    source_artifact_fingerprint=(
                        compute_source_payload_fingerprint(
                            source_type="portfolio_candidate_decision",
                            payload=decision,
                        )
                    ),
                )
                for decision in allocation.candidate_decisions
                if decision.action is not PortfolioCandidateAction.ADMITTED
            )

        intents = {
            intent.security_id: intent
            for intent in session.scheduled_entry_intents
        }
        settlements = {
            settlement.security_id: settlement
            for settlement in session.reservation_settlements
        }
        for decision in session.entry_execution_decisions:
            if decision.status is EntryExecutionStatus.EXECUTED:
                continue
            intent = intents[decision.security_id]
            settlement = settlements[decision.security_id]
            rows.append(
                HistoricalBacktestRejectionRecord(
                    decision_session=decision.planned_entry_session,
                    security_id=decision.security_id,
                    stage=HistoricalRejectionStage.ENTRY_EXECUTION,
                    reason=decision.status.value,
                    requested_quantity=decision.requested_shares,
                    reserved_cash=Decimal(str(settlement.reserved_cash)),
                    released_cash=Decimal(str(settlement.released_cash)),
                    source_rank=intent.source_rank,
                    source_artifact_fingerprint=(
                        compute_source_payload_fingerprint(
                            source_type="entry_execution_decision",
                            payload=decision,
                        )
                    ),
                )
            )
    stage_order = {
        HistoricalRejectionStage.ALLOCATION: 0,
        HistoricalRejectionStage.ENTRY_EXECUTION: 1,
    }
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.decision_session,
                stage_order[row.stage],
                row.security_id,
                row.reason,
            ),
        )
    )


def project_entries(
    run_result: HistoricalBacktestRunResult,
) -> tuple[HistoricalBacktestEntryRecord, ...]:
    """Project exactly one entry record per Phase 15A APPLIED BUY.

    REPLAYED executions and non-BUY executions create no rows (Phase 15D.3A
    Rules 2-3). No upstream owner is re-run; every fact is read from the
    already-validated Phase 13 execution event or its Phase 9/14 decision.
    `cost_basis` is never independently originated: it is read from the
    exact authoritative Phase 13 `OpenPosition` that BUY created, with the
    `quantity * fill_price + execution_cost` identity proven only as a
    reconciliation assertion against that authoritative value.
    """

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    rows: list[HistoricalBacktestEntryRecord] = []
    for session, links in zip(
        context.run_result.session_results,
        context.execution_links,
        strict=True,
    ):
        intents_by_security = {
            intent.security_id: intent
            for intent in session.scheduled_entry_intents
        }
        positions_after = {
            position.asset_id: position
            for position in session.authoritative_state.open_positions
        }
        round_trip_asset_ids = {
            link.event.asset_id for link in links if link.round_trip
        }
        for link in links:
            if (
                link.application_status is not ExecutionApplicationStatus.APPLIED
                or link.event.side is not ExecutionSide.BUY
            ):
                continue
            event = link.event
            position = positions_after.get(event.asset_id)
            if position is None:
                # Task 5C-C: an applied BUY may be absent from the final
                # positions only when a validated APPLIED ROUND_TRIP SELL
                # closed it in this same session; its authoritative cost
                # basis then comes from the BUY_APPLIED ledger row.
                if event.asset_id not in round_trip_asset_ids:
                    raise HistoricalBacktestResultValidationError(
                        "ENTRY_LINKAGE_MISMATCH",
                        "an applied BUY lacks its resulting Phase 13 open position",
                    )
                cost_basis = _round_trip_entry_facts(
                    session_result=session, buy_event=event
                ).entry_cost_basis
            else:
                if (
                    position.entry_execution_id != event.execution_id
                    or position.asset_id != event.asset_id
                    or position.quantity != event.quantity
                    or position.entry_session != event.session
                    or position.entry_price != event.fill_price
                    or position.entry_execution_cost != event.execution_cost
                ):
                    raise HistoricalBacktestResultValidationError(
                        "ENTRY_LINKAGE_MISMATCH",
                        "an applied BUY does not match its Phase 13 open position",
                    )
                # Exact-arithmetic hardening (Residual C): event.fill_price
                # carries no enforced significant-digit bound, so this
                # reconciliation multiplication/addition -- checked against
                # the exact authoritative Phase 13 cost_basis -- uses the
                # shared exact helpers, not ambient-context Decimal `*`/`+`.
                reconciled_cost_basis = add_exact_decimal(
                    exact_decimal_times_int(event.fill_price, event.quantity),
                    event.execution_cost,
                )
                if position.cost_basis != reconciled_cost_basis:
                    raise HistoricalBacktestResultValidationError(
                        "ENTRY_COST_BASIS_MISMATCH",
                        "Phase 13 position cost_basis does not reconcile with "
                        "quantity, fill price, and execution cost",
                    )
                cost_basis = position.cost_basis
            if link.provenance_source is ExecutionProvenanceSource.GENERATED_ENTRY:
                decision = link.entry_decision
                if decision is None or decision.execution_cost_quote is None:
                    raise HistoricalBacktestResultValidationError(
                        "ENTRY_LINKAGE_MISMATCH",
                        "an applied generated BUY lacks its Phase 9 execution decision",
                    )
                intent = intents_by_security.get(event.asset_id)
                if intent is None:
                    raise HistoricalBacktestResultValidationError(
                        "ENTRY_LINKAGE_MISMATCH",
                        "an applied generated BUY lacks its scheduled entry intent",
                    )
                candidate_decision = intent.candidate_decision
                if not isinstance(
                    candidate_decision, RankedPortfolioCandidateDecision
                ):
                    raise HistoricalBacktestResultValidationError(
                        "ENTRY_LINKAGE_MISMATCH",
                        "a generated entry intent lacks ranked Phase 14 provenance",
                    )
                rows.append(
                    HistoricalBacktestEntryRecord(
                        entry_execution_id=event.execution_id,
                        source_order_id=event.source_order_id,
                        security_id=event.asset_id,
                        quantity=event.quantity,
                        entry_session=event.session,
                        fill_price=event.fill_price,
                        execution_cost=event.execution_cost,
                        cost_basis=cost_basis,
                        provenance_source=link.provenance_source,
                        signal_session=decision.signal_session,
                        signal_time=decision.signal_time,
                        allocation_session=intent.allocation_session,
                        source_rank=intent.source_rank,
                        ranking_snapshot_fingerprint=(
                            intent.ranking_snapshot_fingerprint
                        ),
                        ranking_input_fingerprint=(
                            candidate_decision.ranking_input_fingerprint
                        ),
                        entry_execution_status=decision.status,
                        execution_cost_policy_fingerprint=(
                            decision.execution_cost_quote.policy_ref.policy_fingerprint
                        ),
                    )
                )
            elif (
                link.provenance_source
                is ExecutionProvenanceSource.EXTERNAL_SCHEDULED
            ):
                rows.append(
                    HistoricalBacktestEntryRecord(
                        entry_execution_id=event.execution_id,
                        source_order_id=event.source_order_id,
                        security_id=event.asset_id,
                        quantity=event.quantity,
                        entry_session=event.session,
                        fill_price=event.fill_price,
                        execution_cost=event.execution_cost,
                        cost_basis=cost_basis,
                        provenance_source=link.provenance_source,
                    )
                )
            else:
                raise HistoricalBacktestResultValidationError(
                    "ENTRY_LINKAGE_MISMATCH",
                    "an applied BUY cannot carry exit provenance",
                )
    return tuple(rows)


def project_exits(
    run_result: HistoricalBacktestRunResult,
) -> tuple[HistoricalBacktestExitRecord, ...]:
    """Project exactly one exit record per Phase 15A APPLIED SELL.

    Each exit is linked to the exact `OpenPosition` present immediately
    before that SELL (Phase 15D.3A Rule 4). REPLAYED and non-SELL
    executions create no rows.
    """

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    rows: list[HistoricalBacktestExitRecord] = []
    for session, links, state_before in zip(
        context.run_result.session_results,
        context.execution_links,
        context.states_before,
        strict=True,
    ):
        positions_before = {
            position.asset_id: position for position in state_before.open_positions
        }
        for link in links:
            if (
                link.application_status is not ExecutionApplicationStatus.APPLIED
                or link.event.side is not ExecutionSide.SELL
            ):
                continue
            event = link.event
            if link.round_trip:
                # Task 5C-C: the closed position never existed in the prior
                # state; its entry identity is the same session's APPLIED BUY.
                entry_execution_id = _round_trip_buy_event(
                    session, links, event.asset_id
                ).execution_id
            else:
                position = positions_before.get(event.asset_id)
                if position is None:
                    raise HistoricalBacktestResultValidationError(
                        "EXIT_LINKAGE_MISMATCH",
                        "an applied SELL lacks its prior open position",
                    )
                entry_execution_id = position.entry_execution_id
            # Exact-arithmetic hardening (Residual C): event.fill_price
            # carries no enforced significant-digit bound, so this
            # multiplication/subtraction uses the shared exact helpers, not
            # ambient-context Decimal `*`/`-`.
            gross_proceeds = exact_decimal_times_int(event.fill_price, event.quantity)
            net_proceeds = subtract_exact_decimal(gross_proceeds, event.execution_cost)
            if link.provenance_source is ExecutionProvenanceSource.GENERATED_EXIT:
                decision = link.exit_decision
                if decision is None or decision.selected_reason is None:
                    raise HistoricalBacktestResultValidationError(
                        "EXIT_LINKAGE_MISMATCH",
                        "an applied generated SELL lacks its terminal Phase 15B reason",
                    )
                exit_reason = HistoricalExitReason(decision.selected_reason.value)
                reference_exit_price = Decimal(str(decision.reference_exit_price))
            elif (
                link.provenance_source
                is ExecutionProvenanceSource.EXTERNAL_SCHEDULED
            ):
                exit_reason = HistoricalExitReason.EXTERNAL_SCHEDULED
                reference_exit_price = None
            else:
                raise HistoricalBacktestResultValidationError(
                    "EXIT_LINKAGE_MISMATCH",
                    "an applied SELL cannot carry entry provenance",
                )
            if event.settlement_id is None or event.settlement_session is None:
                raise HistoricalBacktestResultValidationError(
                    "EXIT_LINKAGE_MISMATCH",
                    "an applied SELL lacks its settlement identity",
                )
            rows.append(
                HistoricalBacktestExitRecord(
                    exit_execution_id=event.execution_id,
                    source_order_id=event.source_order_id,
                    entry_execution_id=entry_execution_id,
                    security_id=event.asset_id,
                    quantity=event.quantity,
                    exit_session=event.session,
                    fill_price=event.fill_price,
                    execution_cost=event.execution_cost,
                    gross_proceeds=gross_proceeds,
                    net_proceeds=net_proceeds,
                    settlement_id=event.settlement_id,
                    settlement_session=event.settlement_session,
                    provenance_source=link.provenance_source,
                    exit_reason=exit_reason,
                    reference_exit_price=reference_exit_price,
                )
            )
    return tuple(rows)


def _group_dividend_cash_ledger_rows_by_trade(
    cash_ledger: tuple[HistoricalCashLedgerRow, ...],
    *,
    closed_trade_ids: frozenset[str],
    open_trade_ids: frozenset[str],
) -> dict[str, tuple[HistoricalCashLedgerRow, ...]]:
    """Group Slice-9 projected `DIVIDEND_APPLIED` rows by `attribution_trade_id`.

    OD-16.2: the projected `HistoricalCashLedgerRow` tuple is the sole
    normative grouping source -- this reads only that projected row shape,
    never any raw upstream Phase 13 accounting/evidence/outcome object,
    never infers entitlement quantity or per-share amount, and never falls
    back to security/date coincidence. A row whose `attribution_trade_id`
    matches neither a closed trade produced by this same run nor a
    security still open in `final_state` fails closed (OD-16.3): no
    frozen fallback exists for that case in this amendment. Preserves
    `cash_ledger`'s own already-canonical (session, sequence_in_session)
    order within each group, so a trade receiving dividends across
    multiple ex-sessions sees them in chronological order.
    """

    groups: dict[str, list[HistoricalCashLedgerRow]] = {}
    for row in cash_ledger:
        if row.ledger_event_type is not PortfolioLedgerEventType.DIVIDEND_APPLIED:
            continue
        trade_id = row.attribution_trade_id
        if trade_id is None or (
            trade_id not in closed_trade_ids and trade_id not in open_trade_ids
        ):
            raise HistoricalBacktestResultValidationError(
                "DIVIDEND_ATTRIBUTION_ORPHANED",
                "a projected DIVIDEND_APPLIED row's attribution_trade_id "
                "matches neither a closed nor a still-open authoritative trade",
            )
        groups.setdefault(trade_id, []).append(row)
    return {trade_id: tuple(rows) for trade_id, rows in groups.items()}


def _exact_dividend_income(rows: tuple[HistoricalCashLedgerRow, ...]) -> Decimal:
    """Sum `settled_cash_delta` exactly, independent of ambient Decimal context.

    OD-6.7's "no free ambient-context precision parameter" applies to this
    accumulation too: scale-38 dividend amounts can require more
    significant digits than the default Decimal context precision (28)
    allows, so ordinary `+` accumulation could silently round. Reuses the
    accepted Phase 13 `add_exact_decimal` (integer coefficient/exponent
    arithmetic, no ambient context) rather than duplicating it.
    """

    total = Decimal("0")
    for row in rows:
        total = add_exact_decimal(total, row.settled_cash_delta)
    return total


def project_closed_trades(
    run_result: HistoricalBacktestRunResult,
) -> tuple[HistoricalClosedTradeRecord, ...]:
    """Project exactly one closed trade per Phase 15A APPLIED SELL.

    `carried_in` is true exactly when the originating `entry_execution_id`
    already belongs to an `OpenPosition` in `run_result.initial_state`
    (Phase 15D.3A Rules 7-9); such a trade has no entries-table row, so its
    entry-side facts are read directly from the authoritative prior
    `OpenPosition` rather than a synthetic entry record.

    Slice 10 (OD-16): for a dividend-aware source run
    (`run_result.dividend_run_evidence is not None`), each trade also
    carries its authoritative ordinary-dividend attribution, grouped from
    the Slice-9 projected cash ledger (`project_cash_ledger`) and never
    recomputed from Phase 13 economics. `dividend_attribution_completeness`
    is `PARTIAL_PRE_RUN_UNKNOWN` for a carried-in trade and
    `COMPLETE_TRADE_LIFETIME` otherwise -- OD-16.5's second condition (the
    run passed OD-15.7/OD-15.8 processed-session-contiguity validation) is
    then a run-level precondition enforced upstream by Phase 15A/13 before
    this `run_result` could exist at all, not something Phase 15D
    re-derives from a calendar.

    A non-dividend-aware (legacy) source run never underwent OD-15.7/
    OD-15.8 validation at all -- `backtester/validation.py` skips
    processed-session-contiguity checking entirely when the run is not
    dividend-aware -- so OD-16.5's precondition can never be satisfied,
    and the absence of dividend ledger rows is not affirmative coverage
    proving zero applicable dividends (OD-7.1: "absence of evidence is
    not evidence of absence"). Such a trade is therefore published on the
    legacy `historical_closed_trade.v0.1` path with none of the OD-16
    dividend-attribution fields and no `trade_total_pnl`, rather than a
    fabricated `COMPLETE_TRADE_LIFETIME` + zero.

    Slice 11 (OD-16.4/OD-16.5a/OD-21.7): a dividend-aware trade is
    published on the `historical_closed_trade.v0.3` path, which also
    carries the authoritative `trade_total_pnl` -- exactly
    `realized_pnl + ordinary_dividend_income` (via the same exact,
    ambient-context-independent `add_exact_decimal` Slice 9 already uses
    for dividend cash) for `COMPLETE_TRADE_LIFETIME`, and `None` for
    `PARTIAL_PRE_RUN_UNKNOWN` -- the pre-run dividend component is
    unknown, so no lifetime total may be published for a carried-in trade.
    """

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    initial_entry_ids = {
        position.entry_execution_id
        for position in context.run_result.initial_state.open_positions
    }
    drafts: list[dict[str, object]] = []
    for session, links, state_before in zip(
        context.run_result.session_results,
        context.execution_links,
        context.states_before,
        strict=True,
    ):
        positions_before = {
            position.asset_id: position for position in state_before.open_positions
        }
        for link in links:
            if (
                link.application_status is not ExecutionApplicationStatus.APPLIED
                or link.event.side is not ExecutionSide.SELL
            ):
                continue
            event = link.event
            if link.round_trip:
                # Task 5C-C: entry facts come from this session's APPLIED
                # BUY event; the authoritative cost basis from its
                # BUY_APPLIED ledger row (exact sign reversal).
                facts = _round_trip_entry_facts(
                    session_result=session,
                    buy_event=_round_trip_buy_event(
                        session, links, event.asset_id
                    ),
                )
                entry_execution_id = facts.entry_execution_id
                entry_session = facts.entry_session
                entry_fill_price = facts.entry_fill_price
                entry_execution_cost = facts.entry_execution_cost
                entry_cost_basis = facts.entry_cost_basis
            else:
                position = positions_before.get(event.asset_id)
                if position is None:
                    raise HistoricalBacktestResultValidationError(
                        "EXIT_LINKAGE_MISMATCH",
                        "a closed trade lacks its prior open position",
                    )
                entry_execution_id = position.entry_execution_id
                entry_session = position.entry_session
                entry_fill_price = position.entry_price
                entry_execution_cost = position.entry_execution_cost
                entry_cost_basis = position.cost_basis
            if link.provenance_source is ExecutionProvenanceSource.GENERATED_EXIT:
                decision = link.exit_decision
                if decision is None or decision.selected_reason is None:
                    raise HistoricalBacktestResultValidationError(
                        "EXIT_LINKAGE_MISMATCH",
                        "an applied generated SELL lacks its terminal Phase 15B reason",
                    )
                exit_reason = HistoricalExitReason(decision.selected_reason.value)
            elif (
                link.provenance_source
                is ExecutionProvenanceSource.EXTERNAL_SCHEDULED
            ):
                exit_reason = HistoricalExitReason.EXTERNAL_SCHEDULED
            else:
                raise HistoricalBacktestResultValidationError(
                    "EXIT_LINKAGE_MISMATCH",
                    "an applied SELL cannot carry entry provenance",
                )
            # Exact-arithmetic hardening (Residual C): event.fill_price
            # carries no enforced significant-digit bound, so this
            # multiplication and its two adjacent subtractions use the
            # shared exact helpers, not ambient-context Decimal `*`/`-`.
            gross_exit_proceeds = exact_decimal_times_int(
                event.fill_price, event.quantity
            )
            net_exit_proceeds = subtract_exact_decimal(
                gross_exit_proceeds, event.execution_cost
            )
            realized_pnl = subtract_exact_decimal(
                net_exit_proceeds, entry_cost_basis
            )
            drafts.append(
                {
                    "trade_id": entry_execution_id,
                    "security_id": event.asset_id,
                    "quantity": event.quantity,
                    "carried_in": entry_execution_id in initial_entry_ids,
                    "entry_execution_id": entry_execution_id,
                    "entry_session": entry_session,
                    "entry_fill_price": entry_fill_price,
                    "entry_execution_cost": entry_execution_cost,
                    "entry_cost_basis": entry_cost_basis,
                    "exit_execution_id": event.execution_id,
                    "exit_session": event.session,
                    "exit_fill_price": event.fill_price,
                    "exit_execution_cost": event.execution_cost,
                    "exit_reason": exit_reason,
                    "gross_exit_proceeds": gross_exit_proceeds,
                    "net_exit_proceeds": net_exit_proceeds,
                    "realized_pnl": realized_pnl,
                }
            )

    closed_trade_ids = frozenset(draft["trade_id"] for draft in drafts)
    open_trade_ids = frozenset(
        position.entry_execution_id
        for position in context.run_result.final_state.open_positions
    )
    dividend_groups = _group_dividend_cash_ledger_rows_by_trade(
        project_cash_ledger(run_result),
        closed_trade_ids=closed_trade_ids,
        open_trade_ids=open_trade_ids,
    )
    dividend_aware = context.run_result.dividend_run_evidence is not None
    distribution_snapshot_fingerprint = (
        context.run_result.dividend_run_evidence.canonical_distribution_snapshot_fingerprint
        if dividend_aware
        else None
    )
    if dividend_aware and drafts and distribution_snapshot_fingerprint is None:
        # Structurally unreachable today (OD-15.1/validation.py require
        # non-empty session evidence, hence a snapshot, whenever a
        # dividend-aware run has processed sessions to draw trades from),
        # but OD-16.7 forbids ever publishing a COMPLETE dividend-aware
        # attribution fingerprint unbound to a real distribution snapshot,
        # so this is asserted fail-closed rather than left implicit.
        raise HistoricalBacktestResultValidationError(
            "ORDINARY_DIVIDEND_ATTRIBUTION_SNAPSHOT_MISSING",
            "a dividend-aware run publishing closed trades must carry a "
            "canonical distribution snapshot fingerprint",
        )

    rows: list[HistoricalClosedTradeRecord] = []
    for draft in drafts:
        trade_id = draft["trade_id"]
        assert isinstance(trade_id, str)
        if not dividend_aware:
            # Legacy (non-dividend-aware) source run: OD-15.7/OD-15.8
            # contiguity was never validated for it (validation.py skips
            # that check entirely when not dividend-aware), so OD-16.5's
            # COMPLETE_TRADE_LIFETIME precondition can never be proven,
            # and zero dividend ledger rows is not affirmative coverage of
            # zero applicable dividends (OD-7.1). Publish the legacy
            # schema with no OD-16 dividend-attribution claim at all.
            rows.append(
                HistoricalClosedTradeRecord(
                    **draft,
                    schema_version="historical_closed_trade.v0.1",
                )
            )
            continue
        grouped_rows = dividend_groups.get(trade_id, ())
        completeness = (
            DividendAttributionCompleteness.PARTIAL_PRE_RUN_UNKNOWN
            if draft["carried_in"]
            else DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
        )
        fingerprint = compute_dividend_attribution_fingerprint(
            trade_id=trade_id,
            completeness=completeness.value,
            dividend_policy_semantic_identity=(
                ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY
            ),
            canonical_distribution_snapshot_fingerprint=(
                distribution_snapshot_fingerprint
            ),
            grouped_rows=tuple(
                (row.source_event_id, row.source_payload_fingerprint)
                for row in grouped_rows
            ),
        )
        income = _exact_dividend_income(grouped_rows)
        realized_pnl = draft["realized_pnl"]
        assert isinstance(realized_pnl, Decimal)
        trade_total_pnl = (
            add_exact_decimal(realized_pnl, income)
            if completeness
            is DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
            else None
        )
        rows.append(
            HistoricalClosedTradeRecord(
                **draft,
                schema_version="historical_closed_trade.v0.3",
                ordinary_dividend_income=income,
                ordinary_dividend_event_count=len(grouped_rows),
                dividend_attribution_completeness=completeness,
                ordinary_dividend_attribution_fingerprint=fingerprint,
                trade_total_pnl=trade_total_pnl,
            )
        )
    return tuple(rows)


def _project_open_trade_links(
    run_result: HistoricalBacktestRunResult,
) -> tuple[_HistoricalOpenTradeLink, ...]:
    """Link every `final_state.open_positions` item without valuing or closing it.

    Phase 15D.3A defers final-mark construction of `HistoricalOpenTradeRecord`
    to Phase 15D.3B (Rule 10); this hands off exact entry-side facts instead
    of inventing a mark or force-liquidating the position.
    """

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    initial_entry_ids = {
        position.entry_execution_id
        for position in context.run_result.initial_state.open_positions
    }
    return tuple(
        _HistoricalOpenTradeLink(
            security_id=position.asset_id,
            quantity=position.quantity,
            carried_in=position.entry_execution_id in initial_entry_ids,
            entry_execution_id=position.entry_execution_id,
            entry_session=position.entry_session,
            entry_fill_price=position.entry_price,
            entry_execution_cost=position.entry_execution_cost,
            entry_cost_basis=position.cost_basis,
        )
        for position in context.run_result.final_state.open_positions
    )


def project_cash_ledger(
    run_result: HistoricalBacktestRunResult,
) -> tuple[HistoricalCashLedgerRow, ...]:
    """Project a strict 1:1 row per authoritative Phase 13 ledger entry.

    Preserves exact (session, sequence_in_session) order (Phase 15D.3A
    Rule 11); synthesizes no opening-balance row. Ordinary-dividend
    `DIVIDEND_APPLIED` rows (Slice 9, OD-16.2/OD-20.4/OD-20.5) are
    projected 1:1 from `session.state_transition_result.
    dividend_ledger_entries` -- which `_build_validated_source_context`
    has already gated through the site-3 OD-14.5/14.8/14.9 provenance and
    discharge proof before this function ever sees it -- and are placed
    after that session's execution/settlement rows to preserve the frozen
    settlement->SELL->BUY->dividend chronology (OD-10/OD-12.1). Only an
    `APPLIED` outcome ever produces a `DividendLedgerEntry`; `REPLAYED`
    and `NO_OP_NOT_ENTITLED` produce none, so this loop naturally emits
    zero rows for those statuses without special-casing them.
    """

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    rows: list[HistoricalCashLedgerRow] = []
    for session in context.run_result.session_results:
        execution_entries = session.state_transition_result.ledger_entries
        for entry in execution_entries:
            rows.append(
                HistoricalCashLedgerRow(
                    session=entry.session,
                    sequence_in_session=entry.sequence_in_session,
                    ledger_event_type=entry.event_type,
                    source_event_id=entry.source_event_id,
                    source_order_id=entry.source_order_id,
                    security_id=entry.asset_id,
                    settled_cash_delta=entry.settled_cash_delta,
                    pending_cash_delta=entry.pending_cash_delta,
                    settled_cash_after=entry.settled_cash_after,
                    settlement_id=entry.settlement_id,
                    settlement_session=entry.settlement_session,
                    state_hash_before=entry.state_hash_before,
                    state_hash_after=entry.state_hash_after,
                    source_payload_fingerprint=(
                        compute_source_payload_fingerprint(
                            source_type="portfolio_ledger_entry",
                            payload=entry,
                        )
                    ),
                )
            )

        dividend_entries = session.state_transition_result.dividend_ledger_entries
        # Site 3 (via _build_validated_source_context ->
        # PortfolioInvariantChecker.validate_transition) already proved
        # dividend_ledger_entries is consecutive-from-zero and ascending
        # by (asset_id, canonical_distribution_event_id) (OD-20.4); this
        # projector trusts and preserves that upstream order rather than
        # re-deriving or re-sorting it, and continues the execution/
        # settlement sequence_in_session numbering (OD-12's D step comes
        # after SETTLEMENT/SELL/BUY) instead of colliding with Phase 13's
        # own separate per-type dividend counter.
        for local_index, dividend_entry in enumerate(dividend_entries):
            if dividend_entry.sequence_in_session != local_index:
                raise HistoricalBacktestResultValidationError(
                    "DIVIDEND_LEDGER_PROJECTION_ORDER_MISMATCH",
                    "authoritative dividend ledger entries are not "
                    "consecutive from zero",
                )
            rows.append(
                HistoricalCashLedgerRow(
                    session=dividend_entry.session,
                    sequence_in_session=len(execution_entries) + local_index,
                    ledger_event_type=dividend_entry.event_type,
                    source_event_id=dividend_entry.application_id,
                    source_order_id=None,
                    security_id=dividend_entry.asset_id,
                    settled_cash_delta=dividend_entry.settled_cash_delta,
                    pending_cash_delta=Decimal("0"),
                    settled_cash_after=dividend_entry.settled_cash_after,
                    settlement_id=None,
                    settlement_session=None,
                    state_hash_before=dividend_entry.state_hash_before,
                    state_hash_after=dividend_entry.state_hash_after,
                    source_payload_fingerprint=(
                        compute_source_payload_fingerprint(
                            source_type="dividend_ledger_entry",
                            payload=dividend_entry,
                        )
                    ),
                    attribution_trade_id=dividend_entry.attribution_trade_id,
                )
            )
    return tuple(rows)


def project_settlement_ledger(
    run_result: HistoricalBacktestRunResult,
) -> tuple[HistoricalSettlementRecord, ...]:
    """Reconstruct the exact settlement lifecycle from Phase 13 state and ledger.

    The settlement universe comes only from `initial_state.pending_settlements`
    and settlements created by APPLIED SELLs; resolution comes only from
    authoritative SETTLEMENT_APPLIED ledger rows and
    `final_state.pending_settlements` (Phase 15D.3A Rule 13). No settlement
    date is ever recalculated.
    """

    context = _build_validated_source_context(
        run_result=run_result, run_manifest=None
    )
    initial_state = context.run_result.initial_state
    final_state = context.run_result.final_state

    registry = {
        settlement.settlement_id: settlement
        for settlement in initial_state.pending_settlements
    }
    carried_in_ids = set(registry)
    originated_in_run_ids: set[str] = set()
    resolved_in_run_ids: set[str] = set()

    for session in context.run_result.session_results:
        post_state_by_id = {
            settlement.settlement_id: settlement
            for settlement in session.authoritative_state.pending_settlements
        }
        for entry in session.state_transition_result.ledger_entries:
            if entry.event_type is PortfolioLedgerEventType.SELL_APPLIED:
                settlement_id = entry.settlement_id
                if settlement_id is None:
                    raise HistoricalBacktestResultValidationError(
                        "SETTLEMENT_LINKAGE_MISMATCH",
                        "an applied SELL ledger row lacks its settlement identity",
                    )
                settlement = post_state_by_id.get(settlement_id)
                if settlement is None:
                    raise HistoricalBacktestResultValidationError(
                        "SETTLEMENT_LINKAGE_MISMATCH",
                        "an applied SELL did not create its pending settlement",
                    )
                existing = registry.get(settlement_id)
                if existing is not None and existing != settlement:
                    raise HistoricalBacktestResultValidationError(
                        "SETTLEMENT_LINKAGE_MISMATCH",
                        "settlement identity is not unique across the run",
                    )
                registry[settlement_id] = settlement
                originated_in_run_ids.add(settlement_id)
            elif entry.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED:
                settlement_id = entry.source_event_id
                if settlement_id not in registry:
                    raise HistoricalBacktestResultValidationError(
                        "SETTLEMENT_LINKAGE_MISMATCH",
                        "a settlement was resolved without a known pending settlement",
                    )
                if settlement_id in resolved_in_run_ids:
                    raise HistoricalBacktestResultValidationError(
                        "SETTLEMENT_LINKAGE_MISMATCH",
                        "a settlement identity was resolved more than once",
                    )
                resolved_in_run_ids.add(settlement_id)

    final_pending_ids = {
        settlement.settlement_id for settlement in final_state.pending_settlements
    }

    rows: list[HistoricalSettlementRecord] = []
    for settlement_id, settlement in registry.items():
        if settlement_id in final_pending_ids:
            status = HistoricalSettlementStatus.PENDING_AT_RUN_END
        elif settlement_id not in resolved_in_run_ids:
            raise HistoricalBacktestResultValidationError(
                "SETTLEMENT_LINKAGE_MISMATCH",
                "a settlement is neither resolved nor pending at run end",
            )
        elif settlement_id in carried_in_ids:
            status = HistoricalSettlementStatus.CARRIED_IN_AND_SETTLED
        elif settlement_id in originated_in_run_ids:
            status = HistoricalSettlementStatus.SETTLED_DURING_RUN
        else:
            raise HistoricalBacktestResultValidationError(
                "SETTLEMENT_LINKAGE_MISMATCH",
                "a resolved settlement has no known origin",
            )
        rows.append(
            HistoricalSettlementRecord(
                settlement_id=settlement.settlement_id,
                source_sell_execution_id=settlement.source_execution_id,
                security_id=settlement.asset_id,
                trade_session=settlement.trade_session,
                settlement_session=settlement.settlement_session,
                amount=settlement.amount,
                status=status,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.trade_session,
                row.settlement_session,
                row.settlement_id,
            ),
        )
    )


__all__ = [
    "project_allocation_provenance",
    "project_cash_ledger",
    "project_closed_trades",
    "project_entries",
    "project_execution_provenance",
    "project_exits",
    "project_ranking_provenance",
    "project_rejections",
    "project_settlement_ledger",
    "project_signal_provenance",
    "project_transition_audits",
]
