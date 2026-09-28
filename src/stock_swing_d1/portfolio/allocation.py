"""Deterministic Phase 12 portfolio admission and reservation settlement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import TYPE_CHECKING

from stock_swing_d1.execution.entry import (
    EntryExecutionDecision,
    EntryExecutionStatus,
    SizedPendingEntry,
)
from stock_swing_d1.portfolio.allocation_policy import (
    PORTFOLIO_ALLOCATION_POLICY_REF,
)
from stock_swing_d1.portfolio.models import (
    MAX_SIMULTANEOUS_POSITIONS,
    MAX_SINGLE_POSITION_ALLOCATION,
    PortfolioAllocationDecision,
    PortfolioAllocationValidationError,
    PortfolioCandidate,
    PortfolioCandidateAction,
    PortfolioCandidateDecision,
    PortfolioReservationSettlement,
    PortfolioSnapshot,
    RankedPortfolioAllocationDecision,
    RankedPortfolioCandidateDecision,
    _is_finite_number,
    _validate_portfolio_candidate,
    _validate_portfolio_snapshot,
)
from stock_swing_d1.risk.position_sizing import (
    PortfolioSizingSnapshot,
    PositionSizingAction,
    PositionSizingDecision,
    PositionSizingService,
    PositionSizingValidationError,
)

if TYPE_CHECKING:
    from stock_swing_d1.ranking.integration import RankedAllocationBatch


_TERMINAL_NON_EXECUTION_STATUSES = frozenset(
    {
        EntryExecutionStatus.INVALIDATED_BY_EARNINGS,
        EntryExecutionStatus.NO_EXECUTABLE_BAR,
        EntryExecutionStatus.INVALID_OPEN_PRICE,
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION,
    }
)


def _candidate_identity_mismatches(
    *, candidate: PortfolioCandidate, sizing: PositionSizingDecision
) -> tuple[str, ...]:
    return tuple(
        field_name
        for field_name, expected in (
            ("security_id", candidate.signal.security_id),
            ("symbol", candidate.signal.symbol),
            ("signal_session", candidate.signal.signal_session),
            ("signal_time", candidate.signal.signal_time),
            (
                "planned_entry_session",
                candidate.signal.planned_entry_session,
            ),
        )
        if getattr(sizing, field_name) != expected
    )


def _validate_position_sizing_result(
    *,
    candidate: PortfolioCandidate,
    sizing: object,
    portfolio_equity: float,
    candidate_cash_limit: float,
) -> PositionSizingDecision:
    if not isinstance(sizing, PositionSizingDecision):
        raise PortfolioAllocationValidationError(
            "INVALID_POSITION_SIZING_RESULT",
            "Phase 11 must return a PositionSizingDecision",
        )
    identity_mismatches = _candidate_identity_mismatches(
        candidate=candidate, sizing=sizing
    )
    if identity_mismatches:
        raise PortfolioAllocationValidationError(
            "INVALID_POSITION_SIZING_RESULT",
            "Phase 11 result disagrees on " + ", ".join(identity_mismatches),
        )
    if (
        sizing.portfolio_equity != portfolio_equity
        or sizing.cash_available != candidate_cash_limit
    ):
        raise PortfolioAllocationValidationError(
            "INVALID_POSITION_SIZING_RESULT",
            "Phase 11 result does not preserve the supplied portfolio snapshot",
        )

    if sizing.action is PositionSizingAction.SIZED:
        sized_pending_entry = sizing.sized_pending_entry
        if (
            type(sizing.final_shares) is not int
            or sizing.final_shares <= 0
            or not isinstance(sized_pending_entry, SizedPendingEntry)
            or sized_pending_entry.fixed_shares != sizing.final_shares
            or sized_pending_entry.cash_available != candidate_cash_limit
            or not _is_finite_number(
                sizing.sizing_reference_cash_required, positive=True
            )
            or sizing.sizing_reference_cash_required > candidate_cash_limit
        ):
            raise PortfolioAllocationValidationError(
                "INVALID_POSITION_SIZING_RESULT",
                "a SIZED result must contain one valid authoritative fixed quantity",
            )
        sized_mismatches = tuple(
            field_name
            for field_name in (
                "security_id",
                "symbol",
                "signal_session",
                "signal_time",
                "planned_entry_session",
            )
            if getattr(sized_pending_entry, field_name)
            != getattr(sizing, field_name)
        )
        if sized_mismatches:
            raise PortfolioAllocationValidationError(
                "INVALID_POSITION_SIZING_RESULT",
                "sized pending entry disagrees on " + ", ".join(sized_mismatches),
            )
    elif sizing.action in {
        PositionSizingAction.SKIPPED_INSUFFICIENT_CASH,
        PositionSizingAction.SKIPPED_RISK_TOO_SMALL,
    }:
        if (
            type(sizing.final_shares) is not int
            or sizing.final_shares != 0
            or sizing.sized_pending_entry is not None
        ):
            raise PortfolioAllocationValidationError(
                "INVALID_POSITION_SIZING_RESULT",
                "a sizing skip must contain zero shares and no pending order",
            )
    else:
        raise PortfolioAllocationValidationError(
            "INVALID_POSITION_SIZING_RESULT",
            "Phase 11 returned an unsupported sizing action",
        )
    return sizing


def _candidate_decision(
    *,
    candidate: PortfolioCandidate,
    processing_rank: int,
    used_slots_before: int,
    used_slots_after: int,
    unreserved_cash_before: float,
    unreserved_cash_after: float,
    single_position_cap: float | None,
    candidate_cash_limit: float | None,
    action: PortfolioCandidateAction,
    position_sizing_decision: PositionSizingDecision | None,
    reserved_cash: float,
    sized_pending_entry: SizedPendingEntry | None,
    source_rank: int | None = None,
    ranking_snapshot_fingerprint: str | None = None,
    ranking_input_fingerprint: str | None = None,
) -> PortfolioCandidateDecision:
    values = dict(
        security_id=candidate.security_id,
        symbol=candidate.symbol,
        processing_rank=processing_rank,
        used_slots_before=used_slots_before,
        used_slots_after=used_slots_after,
        unreserved_cash_before=unreserved_cash_before,
        unreserved_cash_after=unreserved_cash_after,
        single_position_cap=single_position_cap,
        candidate_cash_limit=candidate_cash_limit,
        action=action,
        position_sizing_decision=position_sizing_decision,
        reserved_cash=reserved_cash,
        sized_pending_entry=sized_pending_entry,
    )
    provenance = (
        source_rank,
        ranking_snapshot_fingerprint,
        ranking_input_fingerprint,
    )
    if all(value is None for value in provenance):
        return PortfolioCandidateDecision._validated(**values)
    if any(value is None for value in provenance):
        raise PortfolioAllocationValidationError(
            "INVALID_RANKED_PROVENANCE",
            "ranked candidate provenance must be supplied as one complete set",
        )
    return RankedPortfolioCandidateDecision._validated(
        **values,
        source_rank=source_rank,
        ranking_snapshot_fingerprint=ranking_snapshot_fingerprint,
        ranking_input_fingerprint=ranking_input_fingerprint,
    )


def _validate_admitted_decision(
    candidate_decision: object,
) -> tuple[PortfolioCandidateDecision, PositionSizingDecision, SizedPendingEntry]:
    if not isinstance(candidate_decision, PortfolioCandidateDecision):
        raise PortfolioAllocationValidationError(
            "INVALID_RESERVATION_SETTLEMENT",
            "candidate_decision must be a PortfolioCandidateDecision",
        )
    if candidate_decision.action is not PortfolioCandidateAction.ADMITTED:
        raise PortfolioAllocationValidationError(
            "INVALID_RESERVATION_SETTLEMENT",
            "only an ADMITTED candidate reservation can be settled",
        )
    sizing = candidate_decision.position_sizing_decision
    sized_pending_entry = candidate_decision.sized_pending_entry
    if (
        not isinstance(sizing, PositionSizingDecision)
        or sizing.action is not PositionSizingAction.SIZED
        or not isinstance(sized_pending_entry, SizedPendingEntry)
        or sized_pending_entry != sizing.sized_pending_entry
        or type(sizing.final_shares) is not int
        or sizing.final_shares <= 0
        or sized_pending_entry.fixed_shares != sizing.final_shares
        or not _is_finite_number(candidate_decision.reserved_cash, positive=True)
        or candidate_decision.candidate_cash_limit
        != candidate_decision.reserved_cash
        or sizing.cash_available != candidate_decision.reserved_cash
        or sized_pending_entry.cash_available != candidate_decision.reserved_cash
        or candidate_decision.security_id != sizing.security_id
        or candidate_decision.symbol != sizing.symbol
    ):
        raise PortfolioAllocationValidationError(
            "INVALID_RESERVATION_SETTLEMENT",
            "admitted candidate reservation and Phase 11 result are inconsistent",
        )
    return candidate_decision, sizing, sized_pending_entry


def _validate_candidate_batch_for_portfolio(
    *,
    portfolio: PortfolioSnapshot,
    candidates: Sequence[PortfolioCandidate],
) -> tuple[PortfolioCandidate, ...]:
    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates, Sequence
    ):
        raise PortfolioAllocationValidationError(
            "INVALID_CANDIDATE_BATCH",
            "candidates must be a finite sequence",
        )
    candidate_batch = tuple(candidates)
    for candidate in candidate_batch:
        _validate_portfolio_candidate(candidate)
        if candidate.signal_session != portfolio.allocation_session:
            raise PortfolioAllocationValidationError(
                "CANDIDATE_SESSION_MISMATCH",
                "every candidate must belong to allocation_session",
            )
        if candidate.signal.signal_time > portfolio.decision_time:
            raise PortfolioAllocationValidationError(
                "CANDIDATE_SIGNAL_AFTER_PORTFOLIO_DECISION",
                "candidate signal_time cannot be later than portfolio decision_time",
            )

    candidate_security_ids = tuple(
        candidate.security_id for candidate in candidate_batch
    )
    if len(set(candidate_security_ids)) != len(candidate_security_ids):
        raise PortfolioAllocationValidationError(
            "DUPLICATE_CANDIDATE_SECURITY_ID",
            "candidate security_ids must be unique before allocation",
        )
    return candidate_batch


@dataclass(frozen=True, slots=True, init=False, repr=False)
class PortfolioAllocationService:
    """Allocate completed-T cash and slots, then diagnose T+1 settlement."""

    _position_sizing_service: object

    def __init__(
        self, *, position_sizing_service: PositionSizingService | None = None
    ) -> None:
        dependency = (
            PositionSizingService()
            if position_sizing_service is None
            else position_sizing_service
        )
        if not callable(getattr(dependency, "size_pending_entry", None)):
            raise PortfolioAllocationValidationError(
                "INVALID_POSITION_SIZING_SERVICE",
                "position_sizing_service must provide size_pending_entry",
            )
        object.__setattr__(self, "_position_sizing_service", dependency)

    def allocate_candidates(
        self,
        *,
        portfolio: PortfolioSnapshot,
        candidates: Sequence[PortfolioCandidate],
    ) -> PortfolioAllocationDecision:
        """Return the deterministic completed-T allocation decision."""

        portfolio = _validate_portfolio_snapshot(portfolio)
        candidate_batch = _validate_candidate_batch_for_portfolio(
            portfolio=portfolio,
            candidates=candidates,
        )
        ordered_candidates = tuple(
            sorted(candidate_batch, key=lambda candidate: candidate.security_id)
        )
        return self._allocate_ordered_candidates(
            portfolio=portfolio,
            ordered_candidates=ordered_candidates,
            ranked_provenance=None,
            ranking_snapshot_fingerprint=None,
            policy_fingerprint=None,
        )

    def allocate_ranked_candidates(
        self,
        *,
        portfolio: PortfolioSnapshot,
        ranked_batch: RankedAllocationBatch,
    ) -> RankedPortfolioAllocationDecision:
        """Allocate a fully preflighted batch in authoritative rank order."""

        from stock_swing_d1.ranking.integration import (
            RankedAllocationValidationError,
            validate_ranked_allocation_batch,
        )

        portfolio = _validate_portfolio_snapshot(portfolio)
        try:
            ranked_batch = validate_ranked_allocation_batch(ranked_batch)
        except RankedAllocationValidationError as error:
            raise PortfolioAllocationValidationError(
                "INVALID_RANKED_ALLOCATION_BATCH",
                f"ranked allocation preflight failed with {error.code}",
            ) from error
        if ranked_batch.ranking_session != portfolio.allocation_session:
            raise PortfolioAllocationValidationError(
                "RANKED_BATCH_SESSION_MISMATCH",
                "ranking_session must equal allocation_session",
            )
        if ranked_batch.decision_time != portfolio.decision_time:
            raise PortfolioAllocationValidationError(
                "RANKED_BATCH_DECISION_TIME_MISMATCH",
                "ranked batch and portfolio decision instants must agree",
            )

        ordered_candidates = tuple(
            candidate.allocation_candidate
            for candidate in ranked_batch.candidates
        )
        _validate_candidate_batch_for_portfolio(
            portfolio=portfolio,
            candidates=ordered_candidates,
        )
        ranked_provenance = {
            candidate.security_id: (
                candidate.rank,
                candidate.ranking_snapshot_fingerprint,
                candidate.ranking_input_fingerprint,
            )
            for candidate in ranked_batch.candidates
        }
        return self._allocate_ordered_candidates(
            portfolio=portfolio,
            ordered_candidates=ordered_candidates,
            ranked_provenance=ranked_provenance,
            ranking_snapshot_fingerprint=(
                ranked_batch.ranking_snapshot_fingerprint
            ),
            policy_fingerprint=ranked_batch.policy_fingerprint,
        )

    def _allocate_ordered_candidates(
        self,
        *,
        portfolio: PortfolioSnapshot,
        ordered_candidates: tuple[PortfolioCandidate, ...],
        ranked_provenance: dict[str, tuple[int, str, str]] | None,
        ranking_snapshot_fingerprint: str | None,
        policy_fingerprint: str | None,
    ) -> PortfolioAllocationDecision:
        """Apply frozen Phase 12 economics to an already ordered sequence.

        Ranking determines candidate processing priority only.  The frozen
        sequential cash state determines each Phase 11 cash limit before
        sizing, so priority can affect Q indirectly; Phase 12 never applies a
        rank multiplier or mutates the fixed Phase 11 quantity afterward.
        """

        if ranked_provenance is None:
            if (
                ranking_snapshot_fingerprint is not None
                or policy_fingerprint is not None
            ):
                raise PortfolioAllocationValidationError(
                    "INVALID_RANKED_PROVENANCE",
                    "legacy allocation cannot carry ranking provenance",
                )
        else:
            if (
                ranking_snapshot_fingerprint is None
                or policy_fingerprint is None
                or set(ranked_provenance)
                != {candidate.security_id for candidate in ordered_candidates}
            ):
                raise PortfolioAllocationValidationError(
                    "INVALID_RANKED_PROVENANCE",
                    "ranked allocation provenance must cover the complete batch",
                )
            expected_source_ranks = tuple(
                ranked_provenance[candidate.security_id][0]
                for candidate in ordered_candidates
            )
            if expected_source_ranks != tuple(
                range(1, len(ordered_candidates) + 1)
            ):
                raise PortfolioAllocationValidationError(
                    "INVALID_RANKED_PROVENANCE",
                    "ordered candidates must already be in source-rank order",
                )

        open_security_ids = {
            position.security_id for position in portfolio.open_positions
        }
        reserved_security_ids: set[str] = set()
        used_slots = len(portfolio.open_positions)
        currently_unreserved_cash = portfolio.cash_available
        decisions: list[PortfolioCandidateDecision] = []

        for processing_rank, candidate in enumerate(ordered_candidates, start=1):
            used_slots_before = used_slots
            unreserved_cash_before = currently_unreserved_cash
            provenance_kwargs: dict[str, object] = {}
            if ranked_provenance is not None:
                (
                    source_rank,
                    candidate_snapshot_fingerprint,
                    ranking_input_fingerprint,
                ) = ranked_provenance[candidate.security_id]
                provenance_kwargs = {
                    "source_rank": source_rank,
                    "ranking_snapshot_fingerprint": (
                        candidate_snapshot_fingerprint
                    ),
                    "ranking_input_fingerprint": ranking_input_fingerprint,
                }

            if (
                candidate.security_id in open_security_ids
                or candidate.security_id in reserved_security_ids
            ):
                decisions.append(
                    _candidate_decision(
                        candidate=candidate,
                        processing_rank=processing_rank,
                        used_slots_before=used_slots_before,
                        used_slots_after=used_slots_before,
                        unreserved_cash_before=unreserved_cash_before,
                        unreserved_cash_after=unreserved_cash_before,
                        single_position_cap=None,
                        candidate_cash_limit=None,
                        action=PortfolioCandidateAction.REJECTED_DUPLICATE_SECURITY,
                        position_sizing_decision=None,
                        reserved_cash=0.0,
                        sized_pending_entry=None,
                        **provenance_kwargs,
                    )
                )
                continue

            if used_slots >= MAX_SIMULTANEOUS_POSITIONS:
                decisions.append(
                    _candidate_decision(
                        candidate=candidate,
                        processing_rank=processing_rank,
                        used_slots_before=used_slots_before,
                        used_slots_after=used_slots_before,
                        unreserved_cash_before=unreserved_cash_before,
                        unreserved_cash_after=unreserved_cash_before,
                        single_position_cap=None,
                        candidate_cash_limit=None,
                        action=(
                            PortfolioCandidateAction.
                            REJECTED_MAX_SIMULTANEOUS_POSITIONS
                        ),
                        position_sizing_decision=None,
                        reserved_cash=0.0,
                        sized_pending_entry=None,
                        **provenance_kwargs,
                    )
                )
                continue

            single_position_cap = (
                portfolio.portfolio_equity
                * MAX_SINGLE_POSITION_ALLOCATION
            )
            candidate_cash_limit = min(
                currently_unreserved_cash, single_position_cap
            )
            if (
                candidate_cash_limit < 0.0
                or candidate_cash_limit > currently_unreserved_cash
                or candidate_cash_limit > single_position_cap
                or not isfinite(candidate_cash_limit)
            ):
                raise PortfolioAllocationValidationError(
                    "INVALID_CANDIDATE_CASH_LIMIT",
                    "candidate capital access violated a cash invariant",
                )

            sizing_portfolio = PortfolioSizingSnapshot(
                portfolio_equity=portfolio.portfolio_equity,
                cash_available=candidate_cash_limit,
            )
            try:
                raw_sizing = self._position_sizing_service.size_pending_entry(
                    signal=candidate.signal,
                    pending_entry=candidate.pending_entry,
                    signal_bar=candidate.signal_bar,
                    portfolio=sizing_portfolio,
                )
            except PositionSizingValidationError as error:
                raise PortfolioAllocationValidationError(
                    "INVALID_POSITION_SIZING_INPUT",
                    f"Phase 11 rejected candidate input with {error.code}",
                ) from error
            except Exception as error:
                raise PortfolioAllocationValidationError(
                    "POSITION_SIZING_FAILURE",
                    "Phase 11 sizing failed before producing a valid decision",
                ) from error
            sizing = _validate_position_sizing_result(
                candidate=candidate,
                sizing=raw_sizing,
                portfolio_equity=portfolio.portfolio_equity,
                candidate_cash_limit=candidate_cash_limit,
            )

            if sizing.action is not PositionSizingAction.SIZED:
                decisions.append(
                    _candidate_decision(
                        candidate=candidate,
                        processing_rank=processing_rank,
                        used_slots_before=used_slots_before,
                        used_slots_after=used_slots_before,
                        unreserved_cash_before=unreserved_cash_before,
                        unreserved_cash_after=unreserved_cash_before,
                        single_position_cap=single_position_cap,
                        candidate_cash_limit=candidate_cash_limit,
                        action=(
                            PortfolioCandidateAction.
                            NOT_ADMITTED_BY_POSITION_SIZING
                        ),
                        position_sizing_decision=sizing,
                        reserved_cash=0.0,
                        sized_pending_entry=None,
                        **provenance_kwargs,
                    )
                )
                continue

            sized_pending_entry = sizing.sized_pending_entry
            assert isinstance(sized_pending_entry, SizedPendingEntry)
            reserved_cash = candidate_cash_limit
            currently_unreserved_cash = (
                currently_unreserved_cash - reserved_cash
            )
            used_slots += 1
            reserved_security_ids.add(candidate.security_id)
            if (
                currently_unreserved_cash < 0.0
                or used_slots > MAX_SIMULTANEOUS_POSITIONS
            ):
                raise PortfolioAllocationValidationError(
                    "INVALID_ALLOCATION_STATE",
                    "allocation violated a cash or slot invariant",
                )
            decisions.append(
                _candidate_decision(
                    candidate=candidate,
                    processing_rank=processing_rank,
                    used_slots_before=used_slots_before,
                    used_slots_after=used_slots,
                    unreserved_cash_before=unreserved_cash_before,
                    unreserved_cash_after=currently_unreserved_cash,
                    single_position_cap=single_position_cap,
                    candidate_cash_limit=candidate_cash_limit,
                    action=PortfolioCandidateAction.ADMITTED,
                    position_sizing_decision=sizing,
                    reserved_cash=reserved_cash,
                    sized_pending_entry=sized_pending_entry,
                    **provenance_kwargs,
                )
            )

        admitted_count = sum(
            decision.action is PortfolioCandidateAction.ADMITTED
            for decision in decisions
        )
        total_reserved_cash = sum(
            decision.reserved_cash for decision in decisions
        )
        ending_used_slots = len(portfolio.open_positions) + admitted_count
        if (
            ending_used_slots != used_slots
            or ending_used_slots > MAX_SIMULTANEOUS_POSITIONS
            or total_reserved_cash > portfolio.cash_available
            or currently_unreserved_cash < 0.0
            or not isclose(
                portfolio.cash_available,
                total_reserved_cash + currently_unreserved_cash,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise PortfolioAllocationValidationError(
                "INVALID_ALLOCATION_STATE",
                "allocation result violated its accounting invariants",
            )

        result_values = dict(
            allocation_session=portfolio.allocation_session,
            decision_time=portfolio.decision_time,
            starting_portfolio_equity=portfolio.portfolio_equity,
            starting_cash=portfolio.cash_available,
            starting_open_position_count=len(portfolio.open_positions),
            candidate_decisions=tuple(decisions),
            admitted_count=admitted_count,
            total_reserved_cash=total_reserved_cash,
            remaining_unreserved_cash=currently_unreserved_cash,
            ending_used_slots=ending_used_slots,
            allocation_policy_ref=PORTFOLIO_ALLOCATION_POLICY_REF,
        )
        if ranked_provenance is None:
            return PortfolioAllocationDecision._validated(**result_values)
        assert ranking_snapshot_fingerprint is not None
        assert policy_fingerprint is not None
        return RankedPortfolioAllocationDecision._validated(
            **result_values,
            ranking_snapshot_fingerprint=ranking_snapshot_fingerprint,
            policy_fingerprint=policy_fingerprint,
        )

    def settle_reservation(
        self,
        *,
        candidate_decision: PortfolioCandidateDecision,
        entry_execution: EntryExecutionDecision,
    ) -> PortfolioReservationSettlement:
        """Return terminal T+1 cash and slot settlement without reallocating."""

        candidate_decision, sizing, sized_pending_entry = (
            _validate_admitted_decision(candidate_decision)
        )
        if not isinstance(entry_execution, EntryExecutionDecision):
            raise PortfolioAllocationValidationError(
                "INVALID_RESERVATION_SETTLEMENT",
                "entry_execution must be an EntryExecutionDecision",
            )

        mismatches = tuple(
            field_name
            for field_name, expected in (
                ("security_id", sized_pending_entry.security_id),
                ("symbol", sized_pending_entry.symbol),
                ("signal_session", sized_pending_entry.signal_session),
                ("signal_time", sized_pending_entry.signal_time),
                (
                    "planned_entry_session",
                    sized_pending_entry.planned_entry_session,
                ),
                ("requested_shares", sizing.final_shares),
                ("cash_available", candidate_decision.reserved_cash),
            )
            if getattr(entry_execution, field_name) != expected
        )
        if mismatches:
            raise PortfolioAllocationValidationError(
                "INVALID_RESERVATION_SETTLEMENT",
                "entry execution disagrees on " + ", ".join(mismatches),
            )
        if (
            type(entry_execution.requested_shares) is not int
            or entry_execution.requested_shares <= 0
        ):
            raise PortfolioAllocationValidationError(
                "INVALID_RESERVATION_SETTLEMENT",
                "requested_shares must preserve the positive whole fixed quantity",
            )

        if entry_execution.status is EntryExecutionStatus.PENDING_ENTRY:
            raise PortfolioAllocationValidationError(
                "INVALID_RESERVATION_SETTLEMENT",
                "PENDING_ENTRY is not a terminal settlement status",
            )
        if entry_execution.status is EntryExecutionStatus.EXECUTED:
            if (
                type(entry_execution.executed_shares) is not int
                or entry_execution.executed_shares != sizing.final_shares
                or not _is_finite_number(
                    entry_execution.actual_cash_required, positive=True
                )
                or entry_execution.actual_cash_required
                > candidate_decision.reserved_cash
            ):
                raise PortfolioAllocationValidationError(
                    "INVALID_RESERVATION_SETTLEMENT",
                    "executed quantity and cash must fit the admitted reservation",
                )
            actual_cash_used = float(entry_execution.actual_cash_required)
            released_cash = (
                candidate_decision.reserved_cash - actual_cash_used
            )
            slot_released = False
            slot_occupied_after_execution = True
        elif entry_execution.status in _TERMINAL_NON_EXECUTION_STATUSES:
            if (
                type(entry_execution.executed_shares) is not int
                or entry_execution.executed_shares != 0
                or entry_execution.actual_cash_required is not None
            ):
                raise PortfolioAllocationValidationError(
                    "INVALID_RESERVATION_SETTLEMENT",
                    "a non-execution must use zero shares and zero actual cash",
                )
            actual_cash_used = 0.0
            released_cash = candidate_decision.reserved_cash
            slot_released = True
            slot_occupied_after_execution = False
        else:
            raise PortfolioAllocationValidationError(
                "INVALID_RESERVATION_SETTLEMENT",
                "entry execution status is not a supported terminal outcome",
            )

        if (
            actual_cash_used < 0.0
            or released_cash < 0.0
            or not isclose(
                candidate_decision.reserved_cash,
                actual_cash_used + released_cash,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise PortfolioAllocationValidationError(
                "INVALID_RESERVATION_SETTLEMENT",
                "settlement cash does not reconcile to the reservation",
            )
        return PortfolioReservationSettlement._validated(
            security_id=candidate_decision.security_id,
            symbol=candidate_decision.symbol,
            reserved_cash=candidate_decision.reserved_cash,
            requested_shares=entry_execution.requested_shares,
            executed_shares=entry_execution.executed_shares,
            entry_execution_status=entry_execution.status,
            actual_cash_used=actual_cash_used,
            released_cash=released_cash,
            slot_released=slot_released,
            slot_occupied_after_execution=slot_occupied_after_execution,
        )


__all__ = ["PortfolioAllocationService"]
