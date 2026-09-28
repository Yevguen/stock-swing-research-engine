"""Hard gates for Phase 9 execution of a pre-fixed quantity."""

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from stock_swing_d1.earnings.integration import EarningsIntegrationAction
from stock_swing_d1.execution.entry import (
    EntryExecutionStatus,
    EntryExecutionValidationError,
    create_sized_pending_entry,
)


def _pending_and_order(
    service,
    make_signal,
    *,
    fixed_shares: int = 12,
    cash_available: float = 10_000.0,
):
    pending = service.create_pending_entry(signal=make_signal())
    order = create_sized_pending_entry(
        pending_entry=pending,
        fixed_shares=fixed_shares,
        cash_available=cash_available,
    )
    return pending, order


@pytest.mark.parametrize("opening_price", [50.0, 100.0, 150.0])
def test_t_plus_one_open_never_changes_fixed_quantity_when_affordable(
    service_factory, make_signal, make_execution_bar, opening_price
) -> None:
    service, _, _ = service_factory()
    pending, order = _pending_and_order(service, make_signal)

    decision = service.execute_pending_entry(
        pending_entry=pending,
        sized_pending_entry=order,
        execution_bar=make_execution_bar(opening_price=opening_price),
    )

    assert decision.status is EntryExecutionStatus.EXECUTED
    assert decision.requested_shares == 12
    assert decision.executed_shares == 12
    assert decision.candidate_execution_price == opening_price * 1.0005
    assert decision.execution_price == decision.candidate_execution_price
    assert decision.candidate_execution_cost_quote is not None
    assert decision.execution_cost_quote == (
        decision.candidate_execution_cost_quote
    )
    assert decision.actual_cash_required == float(
        decision.execution_cost_quote.notional
        + decision.execution_cost_quote.execution_cost
    )
    assert decision.actual_cash_required > 12 * decision.execution_price


def test_unaffordable_gap_up_cancels_entire_entry_without_partial_resize(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory()
    pending, order = _pending_and_order(
        service,
        make_signal,
        fixed_shares=12,
        cash_available=1_200.0,
    )

    decision = service.execute_pending_entry(
        pending_entry=pending,
        sized_pending_entry=order,
        execution_bar=make_execution_bar(opening_price=110.0),
    )

    assert decision.status is (
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
    )
    assert decision.requested_shares == 12
    assert decision.executed_shares == 0
    assert decision.reference_open == 110.0
    assert decision.candidate_execution_price == 110.055
    assert decision.candidate_execution_cost_quote is not None
    assert decision.candidate_cash_required == float(
        decision.candidate_execution_cost_quote.notional
        + decision.candidate_execution_cost_quote.execution_cost
    )
    assert decision.candidate_cash_required > decision.cash_available
    assert decision.execution_price is None
    assert decision.actual_cash_required is None
    assert int(decision.cash_available // decision.candidate_execution_price) > 0


def test_exact_cash_boundary_executes_all_fixed_shares(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory()
    opening_price = 100.0
    probe_service, _, _ = service_factory()
    probe_pending, probe_order = _pending_and_order(
        probe_service,
        make_signal,
        fixed_shares=12,
        cash_available=10_000.0,
    )
    probe = probe_service.execute_pending_entry(
        pending_entry=probe_pending,
        sized_pending_entry=probe_order,
        execution_bar=make_execution_bar(opening_price=opening_price),
    )
    assert probe.execution_cost_quote is not None
    exact_cash_decimal = (
        probe.execution_cost_quote.notional
        + probe.execution_cost_quote.execution_cost
    )
    exact_cash = float(exact_cash_decimal)
    pending, order = _pending_and_order(
        service,
        make_signal,
        fixed_shares=12,
        cash_available=exact_cash,
    )

    decision = service.execute_pending_entry(
        pending_entry=pending,
        sized_pending_entry=order,
        execution_bar=make_execution_bar(opening_price=opening_price),
    )

    assert decision.status is EntryExecutionStatus.EXECUTED
    assert decision.executed_shares == 12
    assert decision.actual_cash_required == decision.cash_available
    assert Decimal(str(decision.actual_cash_required)) == exact_cash_decimal


def test_earnings_invalidation_after_sizing_executes_zero_shares(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory(
        EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
    )
    pending, order = _pending_and_order(service, make_signal)

    decision = service.execute_pending_entry(
        pending_entry=pending,
        sized_pending_entry=order,
        execution_bar=make_execution_bar(),
    )

    assert decision.status is EntryExecutionStatus.INVALIDATED_BY_EARNINGS
    assert decision.requested_shares == 12
    assert decision.executed_shares == 0
    assert decision.candidate_execution_price is None
    assert decision.execution_price is None


def test_no_bar_and_invalid_open_preserve_requested_but_execute_zero(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory()
    pending, order = _pending_and_order(service, make_signal)

    no_bar = service.execute_pending_entry(
        pending_entry=pending,
        sized_pending_entry=order,
        execution_bar=None,
    )
    invalid_bar = make_execution_bar().model_copy(update={"open": 0.0})
    invalid_open = service.execute_pending_entry(
        pending_entry=pending,
        sized_pending_entry=order,
        execution_bar=invalid_bar,
    )

    assert no_bar.status is EntryExecutionStatus.NO_EXECUTABLE_BAR
    assert invalid_open.status is EntryExecutionStatus.INVALID_OPEN_PRICE
    for decision in (no_bar, invalid_open):
        assert decision.requested_shares == 12
        assert decision.executed_shares == 0
        assert decision.execution_price is None
        assert decision.actual_cash_required is None


def test_sized_pending_entry_is_positive_whole_share_and_immutable(
    service_factory, make_signal
) -> None:
    service, _, _ = service_factory()
    pending, order = _pending_and_order(service, make_signal)

    assert order.fixed_shares == 12
    assert type(order.fixed_shares) is int
    with pytest.raises(FrozenInstanceError):
        order.fixed_shares = 10
    with pytest.raises(EntryExecutionValidationError) as raised:
        create_sized_pending_entry(
            pending_entry=pending,
            fixed_shares=12.5,
            cash_available=10_000,
        )
    assert raised.value.code == "INVALID_FIXED_SHARES"


def test_sized_pending_identity_mismatch_fails_before_earnings(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, overlay, _ = service_factory()
    pending, order = _pending_and_order(service, make_signal)
    object.__setattr__(order, "security_id", "OTHER")

    with pytest.raises(EntryExecutionValidationError) as raised:
        service.execute_pending_entry(
            pending_entry=pending,
            sized_pending_entry=order,
            execution_bar=make_execution_bar(),
        )

    assert raised.value.code == "SIZED_PENDING_ENTRY_IDENTITY_MISMATCH"
    assert overlay.calls == []
