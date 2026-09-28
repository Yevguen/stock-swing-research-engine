"""Phase 14B rank-preserving Phase 11-to-Phase 12 integration tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import pytest
from pydantic import ValidationError

from stock_swing_d1.portfolio import (
    PORTFOLIO_ALLOCATION_POLICY_REF,
    PortfolioAllocationService,
    PortfolioAllocationValidationError,
    PortfolioCandidate,
    PortfolioCandidateAction,
    RankedPortfolioAllocationDecision,
    RankedPortfolioCandidateDecision,
)
from stock_swing_d1.ranking import (
    RANKED_ALLOCATION_BATCH_SCHEMA_VERSION,
    RankedAllocationBatch,
    RankedAllocationCandidate,
    RankedAllocationValidationError,
    RankingCandidate,
    build_ranked_allocation_batch,
    compute_candidate_input_fingerprint,
    rank_candidates,
    validate_ranked_allocation_batch,
)
from stock_swing_d1.ranking import integration as ranking_integration
from stock_swing_d1.portfolio import allocation as portfolio_allocation
from stock_swing_d1.risk.position_sizing import PositionSizingConstraint
from tests.portfolio.conftest import DECISION_TIME, SIGNAL_SESSION


def _ranking_candidate(candidate: PortfolioCandidate) -> RankingCandidate:
    signal = candidate.signal
    assert signal.sma_20 is not None
    assert signal.sma_50 is not None
    assert signal.atr_14 is not None
    assert signal.rsi_14 is not None
    values = {
        "security_id": candidate.security_id,
        "ranking_session": signal.signal_session,
        "decision_time": DECISION_TIME,
        "signal_session": signal.signal_session,
        "signal_time": signal.signal_time,
        "sma20": signal.sma_20,
        "sma50": signal.sma_50,
        "atr14": signal.atr_14,
        "rsi14": signal.rsi_14,
    }
    return RankingCandidate(
        **values,
        input_fingerprint=compute_candidate_input_fingerprint(**values),
    )


def _ranking_snapshot(*candidates: PortfolioCandidate):
    return rank_candidates(
        ranking_session=SIGNAL_SESSION,
        decision_time=DECISION_TIME,
        candidates=tuple(_ranking_candidate(candidate) for candidate in candidates),
    )


@pytest.fixture
def conflicting_rank_case(make_candidate):
    # Canonical identity order is A, B, C; economic order is deliberately B, C, A.
    candidate_a = make_candidate("NORGATE:100", close=200.0)
    candidate_b = make_candidate("NORGATE:200", close=50.0)
    candidate_c = make_candidate("NORGATE:300", close=100.0)
    snapshot = _ranking_snapshot(candidate_a, candidate_b, candidate_c)
    assert [item.security_id for item in snapshot.ranked_candidates] == [
        "NORGATE:200",
        "NORGATE:300",
        "NORGATE:100",
    ]
    return candidate_a, candidate_b, candidate_c, snapshot


def _batch(conflicting_rank_case, *, input_order=(0, 2, 1)):
    candidate_a, candidate_b, candidate_c, snapshot = conflicting_rank_case
    candidates = (candidate_a, candidate_b, candidate_c)
    return build_ranked_allocation_batch(
        ranking_snapshot=snapshot,
        allocation_candidates=tuple(candidates[index] for index in input_order),
    )


def _replace_candidates(batch, candidates):
    return batch.model_copy(update={"candidates": tuple(candidates)})


def test_adapter_reconciles_exactly_by_security_and_preserves_snapshot_rank(
    conflicting_rank_case,
) -> None:
    candidate_a, candidate_b, candidate_c, snapshot = conflicting_rank_case
    first = build_ranked_allocation_batch(
        ranking_snapshot=snapshot,
        allocation_candidates=(candidate_a, candidate_c, candidate_b),
    )
    second = build_ranked_allocation_batch(
        ranking_snapshot=snapshot,
        allocation_candidates=(candidate_c, candidate_b, candidate_a),
    )

    assert first == second
    assert [item.security_id for item in first.candidates] == [
        "NORGATE:200",
        "NORGATE:300",
        "NORGATE:100",
    ]
    assert [item.rank for item in first.candidates] == [1, 2, 3]
    assert [item.allocation_candidate for item in first.candidates] == [
        candidate_b,
        candidate_c,
        candidate_a,
    ]
    assert [item.rank for item in first.candidates] == [
        item.rank for item in snapshot.ranked_candidates
    ]
    assert all(
        item.ranking_snapshot_fingerprint == snapshot.snapshot_fingerprint
        for item in first.candidates
    )
    assert [item.ranking_input_fingerprint for item in first.candidates] == [
        item.input_fingerprint for item in snapshot.ranked_candidates
    ]


def test_adapter_does_not_use_positional_zip_or_originate_rank() -> None:
    source = inspect.getsource(ranking_integration)

    assert "zip(" not in source
    assert "enumerate(" not in inspect.getsource(
        ranking_integration.build_ranked_allocation_batch
    )


def test_missing_allocation_security_fails_complete_reconciliation(
    conflicting_rank_case,
) -> None:
    candidate_a, candidate_b, _, snapshot = conflicting_rank_case

    with pytest.raises(
        RankedAllocationValidationError, match="MISSING_ALLOCATION_SECURITY"
    ):
        build_ranked_allocation_batch(
            ranking_snapshot=snapshot,
            allocation_candidates=(candidate_a, candidate_b),
        )


def test_extra_allocation_security_fails_complete_reconciliation(
    conflicting_rank_case, make_candidate
) -> None:
    candidate_a, candidate_b, candidate_c, snapshot = conflicting_rank_case

    with pytest.raises(
        RankedAllocationValidationError, match="EXTRA_ALLOCATION_SECURITY"
    ):
        build_ranked_allocation_batch(
            ranking_snapshot=snapshot,
            allocation_candidates=(
                candidate_a,
                candidate_b,
                candidate_c,
                make_candidate("NORGATE:400"),
            ),
        )


def test_duplicate_allocation_security_fails_complete_reconciliation(
    conflicting_rank_case,
) -> None:
    candidate_a, candidate_b, candidate_c, snapshot = conflicting_rank_case

    with pytest.raises(
        RankedAllocationValidationError, match="DUPLICATE_ALLOCATION_SECURITY_ID"
    ):
        build_ranked_allocation_batch(
            ranking_snapshot=snapshot,
            allocation_candidates=(
                candidate_a,
                candidate_b,
                candidate_c,
                candidate_a,
            ),
        )


def test_reconciliation_security_identity_mismatch_fails(
    conflicting_rank_case, make_candidate
) -> None:
    candidate_a, candidate_b, _, snapshot = conflicting_rank_case

    with pytest.raises(
        RankedAllocationValidationError, match="SECURITY_IDENTITY_MISMATCH"
    ):
        build_ranked_allocation_batch(
            ranking_snapshot=snapshot,
            allocation_candidates=(
                candidate_a,
                candidate_b,
                make_candidate("NORGATE:999"),
            ),
        )


def test_invalid_phase12_candidate_identity_fails_adapter_preflight(
    conflicting_rank_case,
) -> None:
    candidate_a, candidate_b, candidate_c, snapshot = conflicting_rank_case
    object.__setattr__(
        candidate_c,
        "signal_bar",
        candidate_c.signal_bar.model_copy(update={"security_id": "NORGATE:999"}),
    )

    with pytest.raises(
        RankedAllocationValidationError, match="INVALID_ALLOCATION_CANDIDATE"
    ):
        build_ranked_allocation_batch(
            ranking_snapshot=snapshot,
            allocation_candidates=(candidate_a, candidate_b, candidate_c),
        )


def test_ranked_models_are_immutable_and_forbid_extra_fields(
    conflicting_rank_case,
    make_portfolio,
) -> None:
    batch = _batch(conflicting_rank_case)
    decision = PortfolioAllocationService().allocate_ranked_candidates(
        portfolio=make_portfolio(), ranked_batch=batch
    )

    with pytest.raises(ValidationError, match="frozen"):
        batch.candidate_count = 0
    with pytest.raises(ValidationError, match="frozen"):
        batch.candidates[0].rank = 2
    with pytest.raises(TypeError):
        batch.candidates[0] = batch.candidates[0]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RankedAllocationCandidate(
            **batch.candidates[0].model_dump(mode="python"),
            ranking_weight=1.0,
        )
    assert isinstance(decision.candidate_decisions, tuple)
    with pytest.raises(FrozenInstanceError):
        decision.policy_fingerprint = "f" * 64
    with pytest.raises(FrozenInstanceError):
        decision.candidate_decisions[0].source_rank = 2


def test_ranked_batch_candidate_count_validation(conflicting_rank_case) -> None:
    batch = _batch(conflicting_rank_case).model_copy(
        update={"candidate_count": 2}
    )

    with pytest.raises(
        RankedAllocationValidationError, match="RANKED_CANDIDATE_COUNT_MISMATCH"
    ):
        validate_ranked_allocation_batch(batch)


def test_ranked_batch_duplicate_rank_rejection(conflicting_rank_case) -> None:
    batch = _batch(conflicting_rank_case)
    first, second, third = batch.candidates
    malformed = _replace_candidates(
        batch, (first, second.model_copy(update={"rank": 1}), third)
    )

    with pytest.raises(RankedAllocationValidationError, match="DUPLICATE_RANK"):
        validate_ranked_allocation_batch(malformed)


def test_ranked_batch_rank_gap_rejection(conflicting_rank_case) -> None:
    batch = _batch(conflicting_rank_case)
    first, second, third = batch.candidates
    malformed = _replace_candidates(
        batch, (first, second.model_copy(update={"rank": 4}), third)
    )

    with pytest.raises(
        RankedAllocationValidationError, match="INVALID_RANK_DOMAIN"
    ):
        validate_ranked_allocation_batch(malformed)


def test_ranked_batch_tuple_rank_order_rejection(conflicting_rank_case) -> None:
    batch = _batch(conflicting_rank_case)
    first, second, third = batch.candidates
    malformed = _replace_candidates(batch, (second, first, third))

    with pytest.raises(RankedAllocationValidationError, match="RANK_ORDER_MISMATCH"):
        validate_ranked_allocation_batch(malformed)


def test_ranked_batch_snapshot_fingerprint_mismatch_rejection(
    conflicting_rank_case,
) -> None:
    batch = _batch(conflicting_rank_case)
    candidates = list(batch.candidates)
    candidates[1] = candidates[1].model_copy(
        update={"ranking_snapshot_fingerprint": "f" * 64}
    )
    malformed = _replace_candidates(batch, candidates)

    with pytest.raises(
        RankedAllocationValidationError,
        match="RANKING_SNAPSHOT_FINGERPRINT_MISMATCH",
    ):
        validate_ranked_allocation_batch(malformed)


def test_ranked_phase12_preflights_complete_batch_before_sizing(
    conflicting_rank_case, make_portfolio, recording_service
) -> None:
    service, sizing_service = recording_service
    malformed = _batch(conflicting_rank_case).model_copy(
        update={"candidate_count": 2}
    )

    with pytest.raises(
        PortfolioAllocationValidationError,
        match="INVALID_RANKED_ALLOCATION_BATCH",
    ):
        service.allocate_ranked_candidates(
            portfolio=make_portfolio(), ranked_batch=malformed
        )
    assert sizing_service.calls == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("ranking_snapshot_fingerprint", "not-a-sha256"),
        ("ranking_input_fingerprint", None),
    ],
)
def test_ranked_phase12_rejects_invalid_or_missing_provenance_before_sizing(
    conflicting_rank_case,
    make_portfolio,
    recording_service,
    field_name: str,
    value: object,
) -> None:
    service, sizing_service = recording_service
    batch = _batch(conflicting_rank_case)
    candidates = list(batch.candidates)
    candidates[0] = candidates[0].model_copy(update={field_name: value})
    malformed = _replace_candidates(batch, candidates)

    with pytest.raises(
        PortfolioAllocationValidationError,
        match="INVALID_RANKED_ALLOCATION_BATCH",
    ):
        service.allocate_ranked_candidates(
            portfolio=make_portfolio(), ranked_batch=malformed
        )
    assert sizing_service.calls == []


def test_ranked_phase12_rejects_wrapper_identity_mismatch_before_sizing(
    conflicting_rank_case, make_portfolio, recording_service
) -> None:
    service, sizing_service = recording_service
    batch = _batch(conflicting_rank_case)
    candidates = list(batch.candidates)
    candidates[0] = candidates[0].model_copy(
        update={"security_id": "NORGATE:999"}
    )
    malformed = _replace_candidates(batch, candidates)

    with pytest.raises(
        PortfolioAllocationValidationError,
        match="INVALID_RANKED_ALLOCATION_BATCH",
    ):
        service.allocate_ranked_candidates(
            portfolio=make_portfolio(), ranked_batch=malformed
        )
    assert sizing_service.calls == []


def test_empty_ranking_produces_valid_empty_ranked_allocation(
    make_portfolio,
) -> None:
    snapshot = rank_candidates(
        ranking_session=SIGNAL_SESSION,
        decision_time=DECISION_TIME,
        candidates=(),
    )
    batch = build_ranked_allocation_batch(
        ranking_snapshot=snapshot,
        allocation_candidates=(),
    )
    decision = PortfolioAllocationService().allocate_ranked_candidates(
        portfolio=make_portfolio(), ranked_batch=batch
    )

    assert batch.schema_version == RANKED_ALLOCATION_BATCH_SCHEMA_VERSION
    assert batch.candidate_count == 0
    assert batch.candidates == ()
    assert isinstance(decision, RankedPortfolioAllocationDecision)
    assert decision.candidate_decisions == ()
    assert decision.ranking_snapshot_fingerprint == snapshot.snapshot_fingerprint
    assert decision.policy_fingerprint == snapshot.policy.policy_fingerprint
    assert decision.allocation_policy_ref == PORTFOLIO_ALLOCATION_POLICY_REF


def test_mandatory_rank_conflict_processes_b_c_a_not_a_b_c(
    conflicting_rank_case, make_portfolio
) -> None:
    batch = _batch(conflicting_rank_case)

    decision = PortfolioAllocationService().allocate_ranked_candidates(
        portfolio=make_portfolio(), ranked_batch=batch
    )

    assert [item.security_id for item in decision.candidate_decisions] == [
        "NORGATE:200",
        "NORGATE:300",
        "NORGATE:100",
    ]
    assert [item.processing_rank for item in decision.candidate_decisions] == [
        1,
        2,
        3,
    ]
    assert [item.source_rank for item in decision.candidate_decisions] == [1, 2, 3]


def test_legacy_unranked_path_retains_security_id_order(
    conflicting_rank_case, make_portfolio
) -> None:
    candidate_a, candidate_b, candidate_c, _ = conflicting_rank_case

    legacy = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(),
        candidates=(candidate_c, candidate_b, candidate_a),
    )

    assert [item.security_id for item in legacy.candidate_decisions] == [
        "NORGATE:100",
        "NORGATE:200",
        "NORGATE:300",
    ]
    assert all(
        type(item).__name__ == "PortfolioCandidateDecision"
        for item in legacy.candidate_decisions
    )


def test_equivalent_sizing_conditions_produce_equal_q_across_paths(
    conflicting_rank_case, make_portfolio
) -> None:
    candidate_a, candidate_b, candidate_c, _ = conflicting_rank_case
    batch = _batch(conflicting_rank_case)
    service = PortfolioAllocationService()
    ranked = service.allocate_ranked_candidates(
        portfolio=make_portfolio(), ranked_batch=batch
    )
    legacy = service.allocate_candidates(
        portfolio=make_portfolio(),
        candidates=(candidate_a, candidate_b, candidate_c),
    )
    legacy_by_security = {
        item.security_id: item
        for item in legacy.candidate_decisions
    }

    for item in ranked.candidate_decisions:
        legacy_item = legacy_by_security[item.security_id]
        sizing = item.position_sizing_decision
        legacy_sizing = legacy_item.position_sizing_decision
        sized = item.sized_pending_entry
        assert sizing is not None and legacy_sizing is not None
        assert sized is not None
        # Equality is conditional on this fixture preserving the Phase 11
        # equity and cash inputs for each security across both orderings.
        assert item.candidate_cash_limit == legacy_item.candidate_cash_limit
        assert sizing.portfolio_equity == legacy_sizing.portfolio_equity
        assert sizing.cash_available == legacy_sizing.cash_available
        assert sizing.final_shares == sized.fixed_shares
        assert sizing.final_shares == legacy_sizing.final_shares
        assert sized is sizing.sized_pending_entry


def test_same_phase11_inputs_produce_same_q_regardless_of_source_rank(
    make_candidate,
    make_portfolio,
    make_open_position,
    recording_service,
) -> None:
    blocker = make_candidate("NORGATE:100", close=50.0)
    target = make_candidate("NORGATE:200", close=100.0)
    single_snapshot = _ranking_snapshot(target)
    blocked_first_snapshot = _ranking_snapshot(blocker, target)
    assert [
        item.security_id for item in blocked_first_snapshot.ranked_candidates
    ] == [blocker.security_id, target.security_id]

    single_batch = build_ranked_allocation_batch(
        ranking_snapshot=single_snapshot,
        allocation_candidates=(target,),
    )
    blocked_first_batch = build_ranked_allocation_batch(
        ranking_snapshot=blocked_first_snapshot,
        allocation_candidates=(target, blocker),
    )
    portfolio = make_portfolio(
        open_positions=(make_open_position(blocker.security_id),)
    )
    service, sizing_service = recording_service

    target_at_rank_one = service.allocate_ranked_candidates(
        portfolio=portfolio,
        ranked_batch=single_batch,
    ).candidate_decisions[0]
    target_at_rank_two = service.allocate_ranked_candidates(
        portfolio=portfolio,
        ranked_batch=blocked_first_batch,
    ).candidate_decisions[1]

    assert target_at_rank_one.source_rank == 1
    assert target_at_rank_two.source_rank == 2
    assert len(sizing_service.calls) == 2
    first_call, second_call = sizing_service.calls
    assert first_call["signal"] is second_call["signal"] is target.signal
    assert (
        first_call["pending_entry"]
        is second_call["pending_entry"]
        is target.pending_entry
    )
    assert (
        first_call["signal_bar"]
        is second_call["signal_bar"]
        is target.signal_bar
    )
    assert first_call["portfolio"] == second_call["portfolio"]

    first_sizing = target_at_rank_one.position_sizing_decision
    second_sizing = target_at_rank_two.position_sizing_decision
    assert first_sizing is not None and second_sizing is not None
    assert first_sizing.final_shares == second_sizing.final_shares
    assert target_at_rank_one.sized_pending_entry is not None
    assert target_at_rank_two.sized_pending_entry is not None
    assert (
        first_sizing.final_shares
        == target_at_rank_one.sized_pending_entry.fixed_shares
    )
    assert (
        second_sizing.final_shares
        == target_at_rank_two.sized_pending_entry.fixed_shares
    )


def test_rank_priority_affects_q_only_through_sequential_cash_state(
    make_candidate,
    make_portfolio,
    recording_service,
    make_entry_execution,
) -> None:
    candidates = tuple(
        make_candidate(
            f"NORGATE:{security_number}",
            close=100.0,
            atr_fraction=0.005,
        )
        for security_number in (100, 200, 300)
    )
    snapshot = _ranking_snapshot(*candidates)
    batch = build_ranked_allocation_batch(
        ranking_snapshot=snapshot,
        allocation_candidates=tuple(reversed(candidates)),
    )
    service, sizing_service = recording_service

    allocation = service.allocate_ranked_candidates(
        portfolio=make_portfolio(equity=100_000.0, cash=30_000.0),
        ranked_batch=batch,
    )
    first, second, cash_exhausted = allocation.candidate_decisions

    assert [item.security_id for item in allocation.candidate_decisions] == [
        "NORGATE:100",
        "NORGATE:200",
        "NORGATE:300",
    ]
    assert [item.source_rank for item in allocation.candidate_decisions] == [
        1,
        2,
        3,
    ]
    assert [
        item.unreserved_cash_before
        for item in allocation.candidate_decisions
    ] == [30_000.0, 10_000.0, 0.0]
    assert [
        item.candidate_cash_limit for item in allocation.candidate_decisions
    ] == [20_000.0, 10_000.0, 0.0]
    assert [
        call["portfolio"].cash_available for call in sizing_service.calls
    ] == [20_000.0, 10_000.0, 0.0]

    sizings = tuple(
        item.position_sizing_decision
        for item in allocation.candidate_decisions
    )
    assert all(sizing is not None for sizing in sizings)
    assert [sizing.risk_sized_shares for sizing in sizings] == [476, 476, 476]
    assert [sizing.cash_sized_shares for sizing in sizings] == [200, 100, 0]
    assert [sizing.final_shares for sizing in sizings] == [200, 100, 0]
    assert [sizing.binding_constraint for sizing in sizings] == [
        PositionSizingConstraint.CASH,
        PositionSizingConstraint.CASH,
        None,
    ]
    for item in (first, second):
        sizing = item.position_sizing_decision
        assert sizing is not None
        assert item.sized_pending_entry is sizing.sized_pending_entry
        assert item.sized_pending_entry is not None
        assert sizing.final_shares == item.sized_pending_entry.fixed_shares

    assert cash_exhausted.action is (
        PortfolioCandidateAction.NOT_ADMITTED_BY_POSITION_SIZING
    )
    assert cash_exhausted.sized_pending_entry is None
    second_before_release = second
    cash_exhausted_before_release = cash_exhausted

    from stock_swing_d1.execution.entry import EntryExecutionStatus

    settlement = service.settle_reservation(
        candidate_decision=first,
        entry_execution=make_entry_execution(
            first,
            status=(
                EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
            ),
        ),
    )

    assert settlement.released_cash == 20_000.0
    assert settlement.slot_released is True
    assert len(sizing_service.calls) == 3
    assert allocation.candidate_decisions[1] is second_before_release
    assert allocation.candidate_decisions[2] is cash_exhausted_before_release
    assert second.position_sizing_decision.final_shares == 100
    assert cash_exhausted.position_sizing_decision.final_shares == 0


def test_ranked_results_preserve_candidate_and_batch_provenance(
    conflicting_rank_case, make_portfolio
) -> None:
    batch = _batch(conflicting_rank_case)
    decision = PortfolioAllocationService().allocate_ranked_candidates(
        portfolio=make_portfolio(), ranked_batch=batch
    )

    assert isinstance(decision, RankedPortfolioAllocationDecision)
    assert decision.ranking_snapshot_fingerprint == (
        batch.ranking_snapshot_fingerprint
    )
    assert decision.policy_fingerprint == batch.policy_fingerprint
    assert decision.allocation_policy_ref == PORTFOLIO_ALLOCATION_POLICY_REF
    for source, result in zip(batch.candidates, decision.candidate_decisions):
        assert isinstance(result, RankedPortfolioCandidateDecision)
        assert result.security_id == source.security_id
        assert result.source_rank == source.rank
        assert result.ranking_snapshot_fingerprint == (
            source.ranking_snapshot_fingerprint
        )
        assert result.ranking_input_fingerprint == (
            source.ranking_input_fingerprint
        )


def test_allocation_rejection_never_rewrites_source_rank(
    conflicting_rank_case, make_portfolio, make_open_position
) -> None:
    batch = _batch(conflicting_rank_case)
    portfolio = make_portfolio(
        open_positions=(
            make_open_position("NORGATE:200"),
            make_open_position("OPEN:1"),
            make_open_position("OPEN:2"),
            make_open_position("OPEN:3"),
        )
    )

    decision = PortfolioAllocationService().allocate_ranked_candidates(
        portfolio=portfolio, ranked_batch=batch
    )
    first, second, third = decision.candidate_decisions

    assert first.action is PortfolioCandidateAction.REJECTED_DUPLICATE_SECURITY
    assert second.action is PortfolioCandidateAction.ADMITTED
    assert third.action is (
        PortfolioCandidateAction.REJECTED_MAX_SIMULTANEOUS_POSITIONS
    )
    assert [first.source_rank, second.source_rank, third.source_rank] == [1, 2, 3]


def test_ranked_settlement_release_does_not_backfill_or_reallocate(
    conflicting_rank_case,
    make_portfolio,
    make_open_position,
    make_entry_execution,
) -> None:
    batch = _batch(conflicting_rank_case)
    portfolio = make_portfolio(
        open_positions=(
            make_open_position("NORGATE:200"),
            make_open_position("OPEN:1"),
            make_open_position("OPEN:2"),
            make_open_position("OPEN:3"),
        )
    )
    service = PortfolioAllocationService()
    allocation = service.allocate_ranked_candidates(
        portfolio=portfolio, ranked_batch=batch
    )
    admitted = allocation.candidate_decisions[1]
    rejected = allocation.candidate_decisions[2]
    rejected_before = rejected

    from stock_swing_d1.execution.entry import EntryExecutionStatus

    settlement = service.settle_reservation(
        candidate_decision=admitted,
        entry_execution=make_entry_execution(
            admitted,
            status=(
                EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
            ),
        ),
    )

    assert settlement.slot_released is True
    assert allocation.candidate_decisions[2] == rejected_before
    assert rejected.action is (
        PortfolioCandidateAction.REJECTED_MAX_SIMULTANEOUS_POSITIONS
    )
    assert rejected.source_rank == 3


def test_ranked_phase12_does_not_recompute_or_resort_ranking() -> None:
    ranked_entry_source = inspect.getsource(
        PortfolioAllocationService.allocate_ranked_candidates
    ).lower()
    kernel_source = inspect.getsource(
        PortfolioAllocationService._allocate_ordered_candidates
    ).lower()

    for forbidden in (
        "sma20",
        "sma_20",
        "sma50",
        "sma_50",
        "atr14",
        "atr_14",
        "rsi14",
        "rsi_14",
    ):
        assert forbidden not in ranked_entry_source
        assert forbidden not in kernel_source
    assert "sorted(" not in ranked_entry_source
    assert "sorted(" not in kernel_source
    assert "sorted(" in inspect.getsource(
        PortfolioAllocationService.allocate_candidates
    )


def test_ranked_batch_rejects_unsupported_schema(conflicting_rank_case) -> None:
    malformed = _batch(conflicting_rank_case).model_copy(
        update={"schema_version": "ranked_allocation_batch_v9"}
    )

    with pytest.raises(
        RankedAllocationValidationError,
        match="UNSUPPORTED_RANKED_ALLOCATION_SCHEMA",
    ):
        validate_ranked_allocation_batch(malformed)
