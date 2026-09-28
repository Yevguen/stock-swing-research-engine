"""Property-style invariants for the frozen Phase 15B boundary rules."""

from hypothesis import HealthCheck, given, settings, strategies as st

from stock_swing_d1.execution.open_position_exit import (
    ExitBoundary,
    IntrabarAmbiguityStatus,
    OpenPositionExitReason,
)
from tests.execution.open_position_exit.conftest import SESSIONS


PROPERTY_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)


@PROPERTY_SETTINGS
@given(opening=st.floats(min_value=50.0, max_value=95.99, allow_nan=False))
def test_property_open_below_stop_always_selects_open_boundary(
    evaluator, make_input, make_bar, opening
) -> None:
    decision = evaluator.evaluate(
        make_input(
            bar=make_bar(
                open=opening,
                high=max(100.0, opening),
                low=min(49.0, opening),
                close=opening,
            )
        )
    )

    assert decision.exit_boundary is ExitBoundary.OPEN
    assert decision.selected_reason is OpenPositionExitReason.GAP_THROUGH_STOP


@PROPERTY_SETTINGS
@given(opening=st.floats(min_value=108.01, max_value=150.0, allow_nan=False))
def test_property_open_above_target_always_selects_open_boundary(
    evaluator, make_input, make_bar, opening
) -> None:
    decision = evaluator.evaluate(
        make_input(
            bar=make_bar(
                open=opening,
                high=opening + 1.0,
                low=100.0,
                close=opening,
            )
        )
    )

    assert decision.exit_boundary is ExitBoundary.OPEN
    assert decision.selected_reason is OpenPositionExitReason.GAP_THROUGH_TARGET


@PROPERTY_SETTINGS
@given(
    opening=st.floats(min_value=96.01, max_value=107.99, allow_nan=False),
    low=st.floats(min_value=50.0, max_value=96.0, allow_nan=False),
    high=st.floats(min_value=108.0, max_value=150.0, allow_nan=False),
)
def test_property_non_gap_both_touch_is_ambiguous_stop(
    evaluator, make_input, make_bar, opening, low, high
) -> None:
    decision = evaluator.evaluate(
        make_input(
            bar=make_bar(
                open=opening,
                high=high,
                low=low,
                close=opening,
            )
        )
    )

    assert decision.intrabar_ambiguity_status is (
        IntrabarAmbiguityStatus.STOP_AND_TARGET_TOUCHED
    )
    assert decision.selected_reason is OpenPositionExitReason.STOP_LOSS


@PROPERTY_SETTINGS
@given(session_index=st.integers(min_value=0, max_value=8))
def test_property_before_session_10_max_holding_cannot_trigger(
    evaluator, make_input, make_bar, session_index
) -> None:
    decision = evaluator.evaluate(
        make_input(
            session_index=session_index,
            bar=make_bar(session=SESSIONS[session_index]),
        )
    )

    assert decision.holding_session_number < 10
    assert OpenPositionExitReason.MAX_HOLDING not in decision.triggered_reasons


@PROPERTY_SETTINGS
@given(close=st.floats(min_value=97.0, max_value=107.0, allow_nan=False))
def test_property_session_10_without_earlier_trigger_exits_at_close(
    evaluator, make_input, make_bar, close
) -> None:
    decision = evaluator.evaluate(
        make_input(
            session_index=9,
            bar=make_bar(
                session=SESSIONS[9], high=max(105.0, close), close=close
            ),
        )
    )

    assert decision.holding_session_number == 10
    assert decision.selected_reason is OpenPositionExitReason.MAX_HOLDING
    assert decision.exit_boundary is ExitBoundary.CLOSE
