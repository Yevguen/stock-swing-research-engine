"""Direct-construction corruption tests for Phase 15B decisions."""

from dataclasses import replace
from datetime import timedelta

import pytest

from stock_swing_d1.earnings.models import TimingClass
from stock_swing_d1.execution.open_position_exit import (
    ExitBoundary,
    ExitPrerequisiteStatus,
    OpenPositionExitReason,
    OpenPositionExitValidationError,
)
from tests.execution.open_position_exit.conftest import SESSIONS


def _reject(decision, **changes) -> None:
    with pytest.raises(OpenPositionExitValidationError):
        replace(decision, **changes)


@pytest.fixture
def stop_decision(evaluator, make_input, make_bar):
    return evaluator.evaluate(make_input(bar=make_bar(low=95.0, close=97.0)))


@pytest.fixture
def gap_stop_decision(evaluator, make_input, make_bar):
    return evaluator.evaluate(
        make_input(bar=make_bar(open=91.0, high=100.0, low=90.0, close=99.0))
    )


@pytest.fixture
def gap_target_decision(evaluator, make_input, make_bar):
    return evaluator.evaluate(
        make_input(bar=make_bar(open=112.0, high=114.0, low=100.0, close=111.0))
    )


@pytest.fixture
def max_holding_decision(evaluator, make_input, make_bar):
    return evaluator.evaluate(
        make_input(
            session_index=9,
            bar=make_bar(session=SESSIONS[9], close=104.0),
        )
    )


@pytest.fixture
def ambiguous_decision(evaluator, make_input, make_bar):
    return evaluator.evaluate(
        make_input(bar=make_bar(high=110.0, low=95.0, close=109.0))
    )


@pytest.fixture
def earnings_max_decision(
    evaluator, make_input, make_bar, make_earnings_decision
):
    current_index = 9
    prior = make_earnings_decision(
        boundary_session=SESSIONS[8],
        scheduled_date=SESSIONS[9],
        timing_class=TimingClass.AMC,
    )
    current = make_earnings_decision(
        boundary_session=SESSIONS[9],
        scheduled_date=SESSIONS[9],
        timing_class=TimingClass.AMC,
    )
    return evaluator.evaluate(
        make_input(
            session_index=current_index,
            bar=make_bar(session=SESSIONS[current_index], close=104.0),
            earnings_decision=current,
            prior_boundary_earnings_decision=prior,
        )
    )


@pytest.mark.parametrize("holding_session_number", [0, 11])
def test_holding_session_number_outside_frozen_range_is_rejected(
    stop_decision, holding_session_number
) -> None:
    _reject(stop_decision, holding_session_number=holding_session_number)


def test_exit_required_must_be_an_explicit_boolean(stop_decision) -> None:
    _reject(stop_decision, exit_required=1)


def test_entry_session_after_evaluation_session_is_rejected(stop_decision) -> None:
    _reject(stop_decision, entry_session=stop_decision.session + timedelta(days=1))


def test_gap_stop_cannot_claim_intraday_boundary(gap_stop_decision) -> None:
    _reject(gap_stop_decision, exit_boundary=ExitBoundary.INTRADAY)


def test_gap_target_cannot_claim_close_boundary(gap_target_decision) -> None:
    _reject(gap_target_decision, exit_boundary=ExitBoundary.CLOSE)


def test_earnings_exit_cannot_claim_open_boundary(earnings_max_decision) -> None:
    _reject(earnings_max_decision, exit_boundary=ExitBoundary.OPEN)


def test_max_holding_cannot_claim_intraday_boundary(max_holding_decision) -> None:
    _reject(max_holding_decision, exit_boundary=ExitBoundary.INTRADAY)


def test_max_holding_cannot_trigger_on_holding_session_9(
    max_holding_decision,
) -> None:
    _reject(max_holding_decision, holding_session_number=9)


def test_administrative_exit_cannot_supply_final_execution_price(
    max_holding_decision,
) -> None:
    _reject(max_holding_decision, final_execution_price=103.0)


def test_administrative_reference_must_equal_market_close(
    max_holding_decision,
) -> None:
    _reject(max_holding_decision, reference_exit_price=103.0)


def test_protective_final_price_must_equal_phase10_result(stop_decision) -> None:
    _reject(
        stop_decision,
        final_execution_price=stop_decision.final_execution_price + 1.0,
    )


@pytest.mark.parametrize("field_name", ["stop_price", "take_profit_price"])
def test_protective_level_audit_must_equal_phase10(
    stop_decision, field_name
) -> None:
    _reject(stop_decision, **{field_name: getattr(stop_decision, field_name) + 1.0})


def test_ambiguous_reason_tuple_cannot_drop_target(ambiguous_decision) -> None:
    _reject(
        ambiguous_decision,
        triggered_reasons=(OpenPositionExitReason.STOP_LOSS,),
    )


def test_earnings_max_reason_tuple_cannot_be_reversed(
    earnings_max_decision,
) -> None:
    _reject(
        earnings_max_decision,
        triggered_reasons=(
            OpenPositionExitReason.MAX_HOLDING,
            OpenPositionExitReason.EARNINGS_FORCED_EXIT,
        ),
    )


@pytest.mark.parametrize(
    "nested_change",
    [
        {"security_id": "NORGATE:9999"},
        {"session": SESSIONS[2]},
    ],
)
def test_nested_phase10_identity_mismatch_is_rejected(
    stop_decision, nested_change
) -> None:
    corrupted = replace(stop_decision.protective_exit_decision, **nested_change)
    _reject(stop_decision, protective_exit_decision=corrupted)


def test_non_ready_prerequisite_status_is_rejected(stop_decision) -> None:
    """Task 5C-C: READY is the only constructible prerequisite status; a
    foreign/forged status value is rejected by the frozen v0.2 validator."""

    _reject(stop_decision, exit_prerequisite_status="NOT_READY")
    _reject(stop_decision, schema_version="open_position_exit_decision.v0.1")
