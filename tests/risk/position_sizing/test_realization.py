"""Acceptance tests for separate executed-initial and realized risk."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from math import inf, nan

import pytest

from stock_swing_d1.execution.entry import EntryExecutionStatus
from stock_swing_d1.execution.protective_exit import (
    ProtectiveExitValidationError,
)
from stock_swing_d1.risk.position_sizing import (
    ExecutedInitialRisk,
    PositionSizingValidationError,
    TradeRiskRealization,
    calculate_executed_initial_risk,
    realize_risk,
)


def _executed_trade(
    make_sizing,
    execute_sizing,
    protective_service,
    *,
    opening_price=100.0,
    cash=10_000,
):
    sizing, signal, pending, _, _ = make_sizing(cash=cash)
    execution = execute_sizing(
        sizing, pending, opening_price=opening_price
    )
    assert execution.status is EntryExecutionStatus.EXECUTED
    protective_state = protective_service.create_state(
        signal=signal,
        entry_execution=execution,
    )
    return sizing, signal, pending, execution, protective_state


def test_executed_initial_risk_uses_actual_phase9_fill_and_phase10_stop(
    make_sizing, execute_sizing, protective_service
) -> None:
    sizing, _, _, execution, protective_state = _executed_trade(
        make_sizing, execute_sizing, protective_service
    )

    executed_risk = calculate_executed_initial_risk(
        sizing=sizing,
        entry_execution=execution,
        protective_state=protective_state,
    )

    assert executed_risk.shares == sizing.final_shares == 12
    assert executed_risk.actual_entry_execution_price == 100.05
    assert execution.execution_cost_quote is not None
    assert execution.execution_cost_quote.execution_cost > Decimal("0")
    assert execution.actual_cash_required > (
        execution.executed_shares * execution.execution_price
    )
    assert executed_risk.actual_stop_price == protective_state.stop_price
    assert executed_risk.executed_normal_stop_exit_price == pytest.approx(
        protective_state.stop_price * 0.9995
    )
    assert executed_risk.executed_initial_loss_per_share == pytest.approx(
        100.05 - protective_state.stop_price * 0.9995
    )
    assert executed_risk.executed_initial_risk_amount == pytest.approx(
        12 * executed_risk.executed_initial_loss_per_share
    )
    assert executed_risk.executed_initial_risk_fraction == pytest.approx(
        executed_risk.executed_initial_risk_amount / 10_000
    )
    assert sizing.sizing_reference_price == 100.0
    assert sizing.sizing_planned_risk_amount == pytest.approx(48.576)


def test_higher_affordable_open_can_make_executed_initial_risk_exceed_target(
    make_sizing, execute_sizing, protective_service
) -> None:
    sizing, _, _, execution, state = _executed_trade(
        make_sizing,
        execute_sizing,
        protective_service,
        opening_price=110.0,
    )

    executed_risk = calculate_executed_initial_risk(
        sizing=sizing,
        entry_execution=execution,
        protective_state=state,
    )

    assert sizing.sizing_target_risk_amount == 50.0
    assert executed_risk.executed_initial_risk_amount > 50.0
    assert sizing.final_shares == execution.executed_shares == 12


def test_lower_open_makes_executed_initial_risk_below_target_without_more_shares(
    make_sizing, execute_sizing, protective_service
) -> None:
    sizing, _, _, execution, state = _executed_trade(
        make_sizing,
        execute_sizing,
        protective_service,
        opening_price=90.0,
    )

    executed_risk = calculate_executed_initial_risk(
        sizing=sizing,
        entry_execution=execution,
        protective_state=state,
    )

    assert executed_risk.executed_initial_risk_amount < 50.0
    assert sizing.final_shares == execution.executed_shares == 12


def test_realized_loss_uses_genuine_actual_entry_execution(
    make_sizing, execute_sizing, protective_service
) -> None:
    sizing, _, _, execution, _ = _executed_trade(
        make_sizing, execute_sizing, protective_service
    )

    realization = realize_risk(
        sizing=sizing,
        entry_execution=execution,
        actual_exit_execution_price=90.0,
    )

    assert realization.actual_entry_execution_price == 100.05
    assert realization.realized_loss_per_share == pytest.approx(10.05)
    assert realization.realized_risk_amount == pytest.approx(120.6)
    assert realization.realized_risk_fraction == pytest.approx(120.6 / 10_000)
    assert realization.risk_overrun_amount == pytest.approx(120.6 - 48.576)
    assert realization.risk_overrun_ratio == pytest.approx(120.6 / 48.576)


@pytest.mark.parametrize("exit_price", [100.05, 108.0, 1_000.0])
def test_break_even_and_profitable_trades_have_zero_realized_risk(
    make_sizing,
    execute_sizing,
    protective_service,
    exit_price,
) -> None:
    sizing, _, _, execution, _ = _executed_trade(
        make_sizing, execute_sizing, protective_service
    )

    realization = realize_risk(
        sizing=sizing,
        entry_execution=execution,
        actual_exit_execution_price=exit_price,
    )

    assert realization.realized_loss_per_share == 0.0
    assert realization.realized_risk_amount == 0.0
    assert realization.realized_risk_fraction == 0.0
    assert realization.risk_overrun_amount == 0.0
    assert realization.risk_overrun_ratio == 0.0


def test_four_risk_concepts_are_distinct_and_prior_records_remain_immutable(
    make_sizing, execute_sizing, protective_service
) -> None:
    sizing, _, _, execution, state = _executed_trade(
        make_sizing,
        execute_sizing,
        protective_service,
        opening_price=110.0,
    )
    sizing_before = sizing
    executed_risk = calculate_executed_initial_risk(
        sizing=sizing,
        entry_execution=execution,
        protective_state=state,
    )
    realization = realize_risk(
        sizing=sizing,
        entry_execution=execution,
        actual_exit_execution_price=90.0,
    )

    metrics = {
        sizing.sizing_target_risk_amount,
        sizing.sizing_planned_risk_amount,
        executed_risk.executed_initial_risk_amount,
        realization.realized_risk_amount,
    }
    assert len(metrics) == 4
    assert sizing is sizing_before
    assert sizing.final_shares == 12


def test_realized_fraction_always_uses_original_sizing_equity(
    make_sizing, execute_sizing, protective_service
) -> None:
    sizing, _, _, execution, _ = _executed_trade(
        make_sizing, execute_sizing, protective_service
    )
    unrelated_later_equity = 1_000

    realization = realize_risk(
        sizing=sizing,
        entry_execution=execution,
        actual_exit_execution_price=90.0,
    )

    assert unrelated_later_equity != sizing.portfolio_equity
    assert realization.sizing_portfolio_equity == 10_000
    assert realization.realized_risk_fraction == (
        realization.realized_risk_amount / 10_000
    )


def test_cancelled_entry_cannot_create_phase10_state_or_post_execution_risk(
    make_sizing,
    execute_sizing,
    protective_service,
) -> None:
    sizing, signal, pending, _, _ = make_sizing(cash=1_200)
    cancelled = execute_sizing(sizing, pending, opening_price=110.0)
    assert cancelled.status is (
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
    )

    with pytest.raises(ProtectiveExitValidationError) as phase10_error:
        protective_service.create_state(
            signal=signal,
            entry_execution=cancelled,
        )
    assert phase10_error.value.code == "INVALID_ENTRY_EXECUTION"

    with pytest.raises(PositionSizingValidationError) as initial_error:
        calculate_executed_initial_risk(
            sizing=sizing,
            entry_execution=cancelled,
            protective_state=object(),
        )
    assert initial_error.value.code == "INVALID_ENTRY_EXECUTION"

    with pytest.raises(PositionSizingValidationError) as realized_error:
        realize_risk(
            sizing=sizing,
            entry_execution=cancelled,
            actual_exit_execution_price=90.0,
        )
    assert realized_error.value.code == "INVALID_ENTRY_EXECUTION"


def test_execution_quantity_or_identity_mismatch_fails_closed(
    make_sizing, execute_sizing, protective_service
) -> None:
    sizing, _, _, execution, _ = _executed_trade(
        make_sizing, execute_sizing, protective_service
    )

    invalid_executions = []
    for field_name, value in (
        ("security_id", "OTHER"),
        ("requested_shares", 11),
        ("executed_shares", 11),
    ):
        corrupted = deepcopy(execution)
        object.__setattr__(corrupted, field_name, value)
        invalid_executions.append(corrupted)

    for invalid_execution in invalid_executions:
        with pytest.raises(PositionSizingValidationError) as raised:
            realize_risk(
                sizing=sizing,
                entry_execution=invalid_execution,
                actual_exit_execution_price=90.0,
            )
        assert raised.value.code == "SIZING_EXECUTION_IDENTITY_MISMATCH"


@pytest.mark.parametrize("exit_price", [0, -1, nan, inf, -inf, True])
def test_invalid_actual_exit_price_fails_closed(
    make_sizing,
    execute_sizing,
    protective_service,
    exit_price,
) -> None:
    sizing, _, _, execution, _ = _executed_trade(
        make_sizing, execute_sizing, protective_service
    )

    with pytest.raises(PositionSizingValidationError) as raised:
        realize_risk(
            sizing=sizing,
            entry_execution=execution,
            actual_exit_execution_price=exit_price,
        )

    assert raised.value.code == "INVALID_ACTUAL_EXIT_PRICE"


def test_post_execution_records_are_immutable_deterministic_and_nonmutating(
    make_sizing, execute_sizing, protective_service
) -> None:
    sizing, _, _, execution, state = _executed_trade(
        make_sizing, execute_sizing, protective_service
    )
    before = (sizing, execution, state)

    first_initial = calculate_executed_initial_risk(
        sizing=sizing,
        entry_execution=execution,
        protective_state=state,
    )
    second_initial = calculate_executed_initial_risk(
        sizing=sizing,
        entry_execution=execution,
        protective_state=state,
    )
    first_realized = realize_risk(
        sizing=sizing,
        entry_execution=execution,
        actual_exit_execution_price=90.0,
    )
    second_realized = realize_risk(
        sizing=sizing,
        entry_execution=execution,
        actual_exit_execution_price=90.0,
    )

    assert first_initial == second_initial
    assert first_realized == second_realized
    assert (sizing, execution, state) == before
    with pytest.raises(FrozenInstanceError):
        first_initial.executed_initial_risk_amount = 0.0
    with pytest.raises(FrozenInstanceError):
        first_realized.realized_risk_amount = 0.0
    with pytest.raises(TypeError):
        ExecutedInitialRisk(security_id="SECURITY:1001")
    with pytest.raises(TypeError):
        TradeRiskRealization(security_id="SECURITY:1001")
