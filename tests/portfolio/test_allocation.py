"""Acceptance tests for deterministic Phase 12 portfolio allocation."""

from dataclasses import FrozenInstanceError

import pytest

from stock_swing_d1.portfolio import (
    MAX_SIMULTANEOUS_POSITIONS,
    MAX_SINGLE_POSITION_ALLOCATION,
    PortfolioAllocationService,
    PortfolioCandidateAction,
)
from stock_swing_d1.risk.position_sizing import PositionSizingAction


def test_frozen_constants_and_exact_action_set() -> None:
    assert MAX_SIMULTANEOUS_POSITIONS == 5
    assert MAX_SINGLE_POSITION_ALLOCATION == 0.20
    assert tuple(PortfolioCandidateAction) == (
        PortfolioCandidateAction.ADMITTED,
        PortfolioCandidateAction.REJECTED_DUPLICATE_SECURITY,
        PortfolioCandidateAction.REJECTED_MAX_SIMULTANEOUS_POSITIONS,
        PortfolioCandidateAction.NOT_ADMITTED_BY_POSITION_SIZING,
    )


def test_empty_portfolio_admits_five_and_never_over_reserves(
    make_portfolio, make_candidate
) -> None:
    candidates = tuple(
        make_candidate(f"NORGATE:{number}")
        for number in (500, 100, 400, 200, 300)
    )

    decision = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(), candidates=candidates
    )

    assert [item.security_id for item in decision.candidate_decisions] == [
        "NORGATE:100",
        "NORGATE:200",
        "NORGATE:300",
        "NORGATE:400",
        "NORGATE:500",
    ]
    assert [item.processing_rank for item in decision.candidate_decisions] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert all(
        item.single_position_cap == 2_000
        for item in decision.candidate_decisions
    )
    assert decision.admitted_count == 5
    assert decision.ending_used_slots == 5
    assert decision.total_reserved_cash == 10_000
    assert decision.remaining_unreserved_cash == 0


def test_input_order_does_not_affect_full_allocation_result(
    make_portfolio, make_candidate
) -> None:
    candidates = tuple(
        make_candidate(f"NORGATE:{number}") for number in (300, 100, 200)
    )
    service = PortfolioAllocationService()

    first = service.allocate_candidates(
        portfolio=make_portfolio(equity=10_000, cash=7_000),
        candidates=candidates,
    )
    second = service.allocate_candidates(
        portfolio=make_portfolio(equity=10_000, cash=7_000),
        candidates=tuple(reversed(candidates)),
    )

    assert first == second


def test_duplicate_open_security_precedes_full_portfolio_and_skips_sizing(
    make_portfolio, make_open_position, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service
    open_positions = tuple(
        make_open_position(f"NORGATE:{number}")
        for number in (100, 101, 102, 103, 104)
    )

    decision = service.allocate_candidates(
        portfolio=make_portfolio(open_positions=open_positions),
        candidates=(
            make_candidate("NORGATE:200"),
            make_candidate("NORGATE:100"),
        ),
    )

    duplicate, unique = decision.candidate_decisions
    assert duplicate.security_id == "NORGATE:100"
    assert duplicate.action is (
        PortfolioCandidateAction.REJECTED_DUPLICATE_SECURITY
    )
    assert unique.action is (
        PortfolioCandidateAction.REJECTED_MAX_SIMULTANEOUS_POSITIONS
    )
    assert all(item.reserved_cash == 0 for item in decision.candidate_decisions)
    assert all(
        item.used_slots_before == item.used_slots_after == 5
        for item in decision.candidate_decisions
    )
    assert sizing_service.calls == []


def test_four_open_positions_first_admission_fills_last_slot(
    make_portfolio, make_open_position, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service
    opens = tuple(
        make_open_position(f"OPEN:{number}") for number in range(4)
    )

    decision = service.allocate_candidates(
        portfolio=make_portfolio(open_positions=opens),
        candidates=(
            make_candidate("NORGATE:200"),
            make_candidate("NORGATE:100"),
        ),
    )

    admitted, rejected = decision.candidate_decisions
    assert admitted.action is PortfolioCandidateAction.ADMITTED
    assert admitted.used_slots_before == 4
    assert admitted.used_slots_after == 5
    assert rejected.action is (
        PortfolioCandidateAction.REJECTED_MAX_SIMULTANEOUS_POSITIONS
    )
    assert rejected.used_slots_before == rejected.used_slots_after == 5
    assert len(sizing_service.calls) == 1


def test_sizing_skip_preserves_slot_and_cash_then_next_candidate_can_admit(
    make_portfolio, make_open_position, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service
    opens = tuple(
        make_open_position(f"OPEN:{number}") for number in range(4)
    )

    decision = service.allocate_candidates(
        portfolio=make_portfolio(equity=100, cash=100, open_positions=opens),
        candidates=(
            make_candidate("NORGATE:100", close=15.0),
            make_candidate("NORGATE:200", close=5.0),
        ),
    )

    skipped, admitted = decision.candidate_decisions
    assert skipped.position_sizing_decision is not None
    assert skipped.position_sizing_decision.action is (
        PositionSizingAction.SKIPPED_RISK_TOO_SMALL
    )
    assert skipped.action is (
        PortfolioCandidateAction.NOT_ADMITTED_BY_POSITION_SIZING
    )
    assert skipped.reserved_cash == 0
    assert skipped.used_slots_before == skipped.used_slots_after == 4
    assert skipped.unreserved_cash_before == skipped.unreserved_cash_after == 100
    assert admitted.action is PortfolioCandidateAction.ADMITTED
    assert admitted.used_slots_after == 5
    assert len(sizing_service.calls) == 2


def test_twenty_percent_limit_is_the_phase11_cash_snapshot(
    make_portfolio, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service

    decision = service.allocate_candidates(
        portfolio=make_portfolio(equity=10_000, cash=8_000),
        candidates=(make_candidate(),),
    )

    candidate = decision.candidate_decisions[0]
    sizing_portfolio = sizing_service.calls[0]["portfolio"]
    assert candidate.single_position_cap == 2_000
    assert candidate.candidate_cash_limit == 2_000
    assert sizing_portfolio.portfolio_equity == 10_000
    assert sizing_portfolio.cash_available == 2_000


def test_cash_below_twenty_percent_cap_is_passed_exactly(
    make_portfolio, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service

    decision = service.allocate_candidates(
        portfolio=make_portfolio(equity=10_000, cash=1_400),
        candidates=(make_candidate(),),
    )

    candidate = decision.candidate_decisions[0]
    sizing_portfolio = sizing_service.calls[0]["portfolio"]
    assert candidate.single_position_cap == 2_000
    assert candidate.candidate_cash_limit == 1_400
    assert sizing_portfolio.cash_available == 1_400


def test_phase11_owns_q_and_phase12_reserves_full_cash_limit(
    make_portfolio, make_candidate
) -> None:
    decision = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(), candidates=(make_candidate(),)
    )

    candidate = decision.candidate_decisions[0]
    sizing = candidate.position_sizing_decision
    sized = candidate.sized_pending_entry
    assert sizing is not None and sized is not None
    assert sizing.final_shares == sized.fixed_shares == 12
    assert sizing.sizing_reference_cash_required == 1_200
    assert candidate.candidate_cash_limit == 2_000
    assert candidate.reserved_cash == 2_000
    assert candidate.reserved_cash != sizing.sizing_reference_cash_required
    assert (
        candidate.candidate_cash_limit
        == sizing.cash_available
        == sized.cash_available
        == candidate.reserved_cash
    )


def test_sequential_cash_never_double_counts_reservations(
    make_portfolio, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service
    candidates = tuple(
        make_candidate(f"NORGATE:{number}")
        for number in (100, 200, 300, 400)
    )

    decision = service.allocate_candidates(
        portfolio=make_portfolio(equity=10_000, cash=7_000),
        candidates=candidates,
    )

    assert [
        item.candidate_cash_limit for item in decision.candidate_decisions
    ] == [2_000, 2_000, 2_000, 1_000]
    assert [item.reserved_cash for item in decision.candidate_decisions] == [
        2_000,
        2_000,
        2_000,
        1_000,
    ]
    assert [
        call["portfolio"].cash_available for call in sizing_service.calls
    ] == [2_000, 2_000, 2_000, 1_000]
    assert decision.total_reserved_cash == 7_000
    assert decision.remaining_unreserved_cash == 0
    assert (
        decision.starting_cash
        == decision.total_reserved_cash + decision.remaining_unreserved_cash
    )


def test_zero_remaining_cash_reaches_phase11_and_skips_without_reservation(
    make_portfolio, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service

    decision = service.allocate_candidates(
        portfolio=make_portfolio(equity=10_000, cash=2_000),
        candidates=(
            make_candidate("NORGATE:100"),
            make_candidate("NORGATE:200"),
        ),
    )

    admitted, skipped = decision.candidate_decisions
    assert admitted.action is PortfolioCandidateAction.ADMITTED
    assert admitted.reserved_cash == 2_000
    assert skipped.candidate_cash_limit == 0
    assert skipped.position_sizing_decision is not None
    assert skipped.position_sizing_decision.action is (
        PositionSizingAction.SKIPPED_INSUFFICIENT_CASH
    )
    assert skipped.action is (
        PortfolioCandidateAction.NOT_ADMITTED_BY_POSITION_SIZING
    )
    assert skipped.reserved_cash == 0
    assert skipped.used_slots_before == skipped.used_slots_after == 1
    assert skipped.unreserved_cash_before == skipped.unreserved_cash_after == 0
    assert sizing_service.calls[1]["portfolio"].cash_available == 0


def test_allocation_accounting_and_immutability(
    make_portfolio, make_candidate
) -> None:
    portfolio = make_portfolio(equity=10_000, cash=7_000)
    candidates = [
        make_candidate("NORGATE:200"),
        make_candidate("NORGATE:100"),
    ]
    portfolio_before = portfolio
    candidates_before = tuple(candidates)

    decision = PortfolioAllocationService().allocate_candidates(
        portfolio=portfolio, candidates=candidates
    )

    assert portfolio == portfolio_before
    assert tuple(candidates) == candidates_before
    assert decision.admitted_count == sum(
        item.action is PortfolioCandidateAction.ADMITTED
        for item in decision.candidate_decisions
    )
    assert decision.ending_used_slots == (
        decision.starting_open_position_count + decision.admitted_count
    )
    assert decision.total_reserved_cash == sum(
        item.reserved_cash for item in decision.candidate_decisions
    )
    assert decision.remaining_unreserved_cash == (
        decision.starting_cash - decision.total_reserved_cash
    )
    with pytest.raises(FrozenInstanceError):
        decision.admitted_count = 0
    with pytest.raises(FrozenInstanceError):
        decision.candidate_decisions[0].reserved_cash = 0
