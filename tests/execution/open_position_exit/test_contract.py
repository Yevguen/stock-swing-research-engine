"""Purity, prerequisite, and audit separation tests for Phase 15B."""

from dataclasses import FrozenInstanceError

import pytest

from stock_swing_d1.execution.open_position_exit import (
    ExitPrerequisiteStatus,
    OpenPositionExitEvaluationInput,
    OpenPositionExitReason,
    OpenPositionExitValidationError,
)
from tests.execution.open_position_exit.conftest import SESSIONS


def test_missing_t_bar_fails_closed(evaluator, make_input) -> None:
    with pytest.raises(OpenPositionExitValidationError) as raised:
        evaluator.evaluate(make_input(bar=None))

    assert raised.value.code == "MISSING_MARKET_DATA"


def test_repeated_identical_evaluation_is_equal(evaluator, make_input) -> None:
    evaluation = make_input()

    assert evaluator.evaluate(evaluation) == evaluator.evaluate(evaluation)


def test_input_objects_remain_unchanged(evaluator, make_input) -> None:
    evaluation = make_input()
    state_before = evaluation.protective_state
    bar_before = evaluation.market_bar

    evaluator.evaluate(evaluation)

    assert evaluation.protective_state is state_before
    assert evaluation.protective_state.last_evaluated_session == SESSIONS[0]
    assert evaluation.market_bar is bar_before
    assert evaluation.market_bar.close == 101.0
    with pytest.raises(FrozenInstanceError):
        evaluation.session = SESSIONS[2]


def test_exactly_one_terminal_decision_is_selected(
    evaluator, make_input, make_bar
) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(high=110.0, low=95.0, close=109.0))
    )

    assert len(decision.triggered_reasons) == 2
    assert decision.selected_reason is OpenPositionExitReason.STOP_LOSS
    assert decision.exit_required is True


def test_reference_price_is_distinct_from_phase10_final_execution_price(
    evaluator, make_input, make_bar
) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(low=95.0, close=97.0))
    )

    assert decision.reference_exit_price == 96.0
    assert decision.final_execution_price == 96.0 * 0.9995
    assert decision.final_execution_price != decision.reference_exit_price
    assert (
        decision.final_execution_price
        == decision.protective_exit_decision.execution_exit_price
    )


def test_close_exit_has_no_phase15b_invented_final_execution_price(
    evaluator, make_input, make_bar
) -> None:
    decision = evaluator.evaluate(
        make_input(
            session_index=9,
            bar=make_bar(session=SESSIONS[9], close=104.0),
        )
    )

    assert decision.reference_exit_price == 104.0
    assert decision.final_execution_price is None


def test_same_session_entry_to_exit_terminal_decision_is_persistence_ready(
    evaluator, make_state, make_bar
) -> None:
    """Task 5C-C retired the same-session persistence limitation: a terminal
    decision on the entry session itself is READY, with holding session 1,
    and the decision schema is the frozen v0.2."""

    state = make_state(session_index=0)
    evaluation = OpenPositionExitEvaluationInput(
        session=SESSIONS[0],
        protective_state=state,
        market_bar=make_bar(
            session=SESSIONS[0],
            open=100.0,
            high=105.0,
            low=95.0,
            close=101.0,
        ),
    )

    decision = evaluator.evaluate(evaluation)

    assert decision.selected_reason is OpenPositionExitReason.STOP_LOSS
    assert decision.entry_session == decision.session == SESSIONS[0]
    assert decision.holding_session_number == 1
    assert decision.exit_prerequisite_status is ExitPrerequisiteStatus.READY
    assert decision.schema_version == "open_position_exit_decision.v0.2"


def test_the_same_session_limitation_member_is_retired() -> None:
    assert [member.value for member in ExitPrerequisiteStatus] == ["READY"]
    assert not hasattr(
        ExitPrerequisiteStatus, "SAME_SESSION_BUY_SELL_PERSISTENCE_UNSUPPORTED"
    )


def test_unprocessed_capital_event_fails_closed_at_phase10_prerequisite(
    evaluator, make_input, make_bar, make_state
) -> None:
    from stock_swing_d1.models import CorporateActionEvent

    event = CorporateActionEvent(
        security_id="NORGATE:1001",
        symbol="ACME",
        source_asset_id=1001,
        event_date=SESSIONS[1],
        date_semantics="effective_date",
        event_type="split",
        new_shares=None,
        old_shares=None,
        terms_verified=False,
        source_provider="Norgate Data",
    )
    base = make_input(bar=make_bar())
    evaluation = OpenPositionExitEvaluationInput(
        session=base.session,
        protective_state=make_state(),
        market_bar=base.market_bar,
        corporate_actions=(event,),
    )

    with pytest.raises(OpenPositionExitValidationError) as raised:
        evaluator.evaluate(evaluation)

    assert raised.value.code == "UNRESOLVED_CORPORATE_ACTION_PREREQUISITE"


def test_unexpected_input_fields_are_rejected(make_state, make_bar) -> None:
    with pytest.raises(TypeError):
        OpenPositionExitEvaluationInput(
            session=SESSIONS[1],
            protective_state=make_state(),
            market_bar=make_bar(),
            random_uuid="not-allowed",
        )
