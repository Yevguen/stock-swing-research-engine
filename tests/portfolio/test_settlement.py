"""Terminal T+1 reservation-settlement tests for Phase 12."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from stock_swing_d1.execution.entry import EntryExecutionStatus
from stock_swing_d1.portfolio import (
    PortfolioAllocationService,
    PortfolioAllocationValidationError,
    PortfolioCandidateAction,
)


def test_executed_entry_releases_unused_cash_and_keeps_slot(
    admitted_decision, make_entry_execution
) -> None:
    execution = make_entry_execution(
        admitted_decision, actual_cash_required=1_950
    )

    settlement = PortfolioAllocationService().settle_reservation(
        candidate_decision=admitted_decision,
        entry_execution=execution,
    )

    assert settlement.reserved_cash == 2_000
    assert settlement.actual_cash_used == 1_950
    assert settlement.released_cash == 50
    assert settlement.slot_released is False
    assert settlement.slot_occupied_after_execution is True
    assert settlement.requested_shares == settlement.executed_shares == 12
    assert (
        settlement.reserved_cash
        == settlement.actual_cash_used + settlement.released_cash
    )


def test_exact_reservation_use_releases_zero_cash(
    admitted_decision, make_entry_execution
) -> None:
    execution = make_entry_execution(
        admitted_decision, actual_cash_required=2_000
    )

    settlement = PortfolioAllocationService().settle_reservation(
        candidate_decision=admitted_decision,
        entry_execution=execution,
    )

    assert settlement.actual_cash_used == 2_000
    assert settlement.released_cash == 0
    assert settlement.slot_released is False
    assert settlement.slot_occupied_after_execution is True


@pytest.mark.parametrize(
    "status",
    [
        EntryExecutionStatus.INVALIDATED_BY_EARNINGS,
        EntryExecutionStatus.NO_EXECUTABLE_BAR,
        EntryExecutionStatus.INVALID_OPEN_PRICE,
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION,
    ],
)
def test_terminal_non_execution_releases_full_cash_and_slot(
    admitted_decision, make_entry_execution, status
) -> None:
    execution = make_entry_execution(admitted_decision, status=status)

    settlement = PortfolioAllocationService().settle_reservation(
        candidate_decision=admitted_decision,
        entry_execution=execution,
    )

    assert settlement.entry_execution_status is status
    assert settlement.executed_shares == 0
    assert settlement.actual_cash_used == 0
    assert settlement.released_cash == settlement.reserved_cash == 2_000
    assert settlement.slot_released is True
    assert settlement.slot_occupied_after_execution is False


def test_execution_cannot_exceed_reservation(
    admitted_decision, make_entry_execution
) -> None:
    execution = make_entry_execution(
        admitted_decision, actual_cash_required=2_001
    )

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        PortfolioAllocationService().settle_reservation(
            candidate_decision=admitted_decision,
            entry_execution=execution,
        )

    assert raised.value.code == "INVALID_RESERVATION_SETTLEMENT"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("security_id", "OTHER"),
        ("symbol", "OTHER"),
        ("signal_session", None),
        ("signal_time", None),
        ("planned_entry_session", None),
        ("requested_shares", 99),
        ("cash_available", 1_999),
    ],
)
def test_settlement_identity_cash_and_q_mismatch_fail_closed(
    admitted_decision, make_entry_execution, field_name, replacement
) -> None:
    malformed = deepcopy(make_entry_execution(admitted_decision))
    object.__setattr__(malformed, field_name, replacement)

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        PortfolioAllocationService().settle_reservation(
            candidate_decision=admitted_decision,
            entry_execution=malformed,
        )

    assert raised.value.code == "INVALID_RESERVATION_SETTLEMENT"


def test_executed_share_mismatch_fails_closed(
    admitted_decision, make_entry_execution
) -> None:
    execution = deepcopy(make_entry_execution(admitted_decision))
    object.__setattr__(execution, "executed_shares", 11)

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        PortfolioAllocationService().settle_reservation(
            candidate_decision=admitted_decision,
            entry_execution=execution,
        )

    assert raised.value.code == "INVALID_RESERVATION_SETTLEMENT"


def test_non_execution_with_cash_or_shares_fails_closed(
    admitted_decision, make_entry_execution
) -> None:
    base = make_entry_execution(
        admitted_decision, status=EntryExecutionStatus.NO_EXECUTABLE_BAR
    )
    malformed_shares = replace(base, executed_shares=1)
    malformed_cash = replace(base, actual_cash_required=1.0)

    for execution in (malformed_shares, malformed_cash):
        with pytest.raises(PortfolioAllocationValidationError) as raised:
            PortfolioAllocationService().settle_reservation(
                candidate_decision=admitted_decision,
                entry_execution=execution,
            )
        assert raised.value.code == "INVALID_RESERVATION_SETTLEMENT"


def test_pending_execution_cannot_settle(
    admitted_decision, make_entry_execution
) -> None:
    execution = make_entry_execution(
        admitted_decision, status=EntryExecutionStatus.PENDING_ENTRY
    )

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        PortfolioAllocationService().settle_reservation(
            candidate_decision=admitted_decision,
            entry_execution=execution,
        )

    assert raised.value.code == "INVALID_RESERVATION_SETTLEMENT"


def test_only_admitted_candidates_can_settle(
    make_portfolio, make_open_position, make_candidate, make_entry_execution
) -> None:
    duplicate = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(
            open_positions=(make_open_position("NORGATE:100"),)
        ),
        candidates=(make_candidate("NORGATE:100"),),
    ).candidate_decisions[0]
    full = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(
            open_positions=tuple(
                make_open_position(f"OPEN:{number}") for number in range(5)
            )
        ),
        candidates=(make_candidate(),),
    ).candidate_decisions[0]
    sizing_skip = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(equity=100, cash=100),
        candidates=(make_candidate(close=15),),
    ).candidate_decisions[0]
    assert [item.action for item in (duplicate, full, sizing_skip)] == [
        PortfolioCandidateAction.REJECTED_DUPLICATE_SECURITY,
        PortfolioCandidateAction.REJECTED_MAX_SIMULTANEOUS_POSITIONS,
        PortfolioCandidateAction.NOT_ADMITTED_BY_POSITION_SIZING,
    ]

    for candidate_decision in (duplicate, full, sizing_skip):
        with pytest.raises(PortfolioAllocationValidationError) as raised:
            PortfolioAllocationService().settle_reservation(
                candidate_decision=candidate_decision,
                entry_execution=object(),
            )
        assert raised.value.code == "INVALID_RESERVATION_SETTLEMENT"


def test_release_does_not_backfill_rejected_candidate(
    make_portfolio,
    make_open_position,
    make_candidate,
    make_entry_execution,
) -> None:
    allocation = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(
            open_positions=tuple(
                make_open_position(f"OPEN:{number}") for number in range(4)
            )
        ),
        candidates=(
            make_candidate("NORGATE:100"),
            make_candidate("NORGATE:200"),
        ),
    )
    admitted, rejected = allocation.candidate_decisions
    rejected_before = rejected
    execution = make_entry_execution(
        admitted,
        status=EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION,
    )

    settlement = PortfolioAllocationService().settle_reservation(
        candidate_decision=admitted, entry_execution=execution
    )

    assert settlement.slot_released is True
    assert settlement.released_cash == admitted.reserved_cash
    assert allocation.candidate_decisions[1] == rejected_before
    assert rejected.action is (
        PortfolioCandidateAction.REJECTED_MAX_SIMULTANEOUS_POSITIONS
    )


def test_unused_execution_cash_does_not_resize_another_admission(
    make_portfolio, make_candidate, make_entry_execution
) -> None:
    allocation = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(),
        candidates=(
            make_candidate("NORGATE:100"),
            make_candidate("NORGATE:200"),
        ),
    )
    first, second = allocation.candidate_decisions
    second_before = second

    settlement = PortfolioAllocationService().settle_reservation(
        candidate_decision=first,
        entry_execution=make_entry_execution(
            first, actual_cash_required=1_000
        ),
    )

    assert settlement.released_cash == 1_000
    assert allocation.candidate_decisions[1] == second_before
    assert second.candidate_cash_limit == 2_000
    assert second.reserved_cash == 2_000
    assert second.sized_pending_entry is not None
    assert second.sized_pending_entry.fixed_shares == 12


def test_different_future_outcomes_do_not_mutate_t_allocation(
    make_portfolio, make_candidate, make_entry_execution
) -> None:
    allocation = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(), candidates=(make_candidate(),)
    )
    admitted = allocation.candidate_decisions[0]
    before = allocation
    service = PortfolioAllocationService()

    executed = service.settle_reservation(
        candidate_decision=admitted,
        entry_execution=make_entry_execution(
            admitted, actual_cash_required=1_000
        ),
    )
    cancelled = service.settle_reservation(
        candidate_decision=admitted,
        entry_execution=make_entry_execution(
            admitted,
            status=(
                EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
            ),
        ),
    )

    assert executed != cancelled
    assert allocation == before
    assert allocation.candidate_decisions[0].sized_pending_entry.fixed_shares == 12


def test_settlement_is_deterministic_and_immutable(
    admitted_decision, make_entry_execution
) -> None:
    service = PortfolioAllocationService()
    execution = make_entry_execution(admitted_decision)

    first = service.settle_reservation(
        candidate_decision=admitted_decision, entry_execution=execution
    )
    second = service.settle_reservation(
        candidate_decision=admitted_decision, entry_execution=execution
    )

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.released_cash = 0
