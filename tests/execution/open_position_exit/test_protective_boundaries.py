"""OPEN and INTRADAY Phase 15B precedence over the real Phase 10 engine."""

import pytest

from stock_swing_d1.execution.open_position_exit import (
    ExitBoundary,
    IntrabarAmbiguityStatus,
    OpenPositionExitReason,
)


def test_no_exit(evaluator, make_input) -> None:
    decision = evaluator.evaluate(make_input())

    assert decision.exit_required is False
    assert decision.triggered_reasons == ()
    assert decision.selected_reason is None
    assert decision.exit_boundary is None


def test_ordinary_stop(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(low=95.0, close=97.0))
    )

    assert decision.selected_reason is OpenPositionExitReason.STOP_LOSS
    assert decision.exit_boundary is ExitBoundary.INTRADAY
    assert decision.reference_exit_price == 96.0


def test_ordinary_target(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(high=109.0, close=107.0))
    )

    assert decision.selected_reason is OpenPositionExitReason.TAKE_PROFIT
    assert decision.exit_boundary is ExitBoundary.INTRADAY
    assert decision.reference_exit_price == 108.0


def test_open_below_stop_uses_open_reference(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(open=91.0, high=100.0, low=90.0, close=99.0))
    )

    assert decision.selected_reason is OpenPositionExitReason.GAP_THROUGH_STOP
    assert decision.exit_boundary is ExitBoundary.OPEN
    assert decision.reference_exit_price == 91.0


def test_open_exactly_at_stop_is_open_stop(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(open=96.0, high=100.0, low=95.0, close=99.0))
    )

    assert decision.selected_reason is OpenPositionExitReason.STOP_LOSS
    assert decision.exit_boundary is ExitBoundary.OPEN
    assert decision.reference_exit_price == 96.0


def test_open_above_target_uses_open_reference(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(open=112.0, high=114.0, low=100.0, close=111.0))
    )

    assert decision.selected_reason is OpenPositionExitReason.GAP_THROUGH_TARGET
    assert decision.exit_boundary is ExitBoundary.OPEN
    assert decision.reference_exit_price == 112.0


def test_open_exactly_at_target_is_open_target(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(open=108.0, high=110.0, low=100.0, close=109.0))
    )

    assert decision.selected_reason is OpenPositionExitReason.TAKE_PROFIT
    assert decision.exit_boundary is ExitBoundary.OPEN
    assert decision.reference_exit_price == 108.0


def test_low_exactly_at_stop_is_inclusive(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(make_input(bar=make_bar(low=96.0)))

    assert decision.selected_reason is OpenPositionExitReason.STOP_LOSS


def test_high_exactly_at_target_is_inclusive(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(high=108.0, close=107.0))
    )

    assert decision.selected_reason is OpenPositionExitReason.TAKE_PROFIT


@pytest.mark.parametrize("close", [95.0, 109.0])
def test_non_gap_stop_and_target_is_ambiguous_stop(
    evaluator, make_input, make_bar, close
) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(open=100.0, high=110.0, low=95.0, close=close))
    )

    assert decision.triggered_reasons == (
        OpenPositionExitReason.STOP_LOSS,
        OpenPositionExitReason.TAKE_PROFIT,
    )
    assert decision.selected_reason is OpenPositionExitReason.STOP_LOSS
    assert decision.intrabar_ambiguity_status is (
        IntrabarAmbiguityStatus.STOP_AND_TARGET_TOUCHED
    )


def test_gap_stop_resolves_before_later_target(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(open=91.0, high=120.0, low=90.0, close=110.0))
    )

    assert decision.triggered_reasons == (
        OpenPositionExitReason.GAP_THROUGH_STOP,
    )
    assert decision.selected_reason is OpenPositionExitReason.GAP_THROUGH_STOP
    assert decision.intrabar_ambiguity_status is IntrabarAmbiguityStatus.NOT_AMBIGUOUS


def test_gap_target_resolves_before_later_stop(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(bar=make_bar(open=112.0, high=115.0, low=90.0, close=100.0))
    )

    assert decision.triggered_reasons == (
        OpenPositionExitReason.GAP_THROUGH_TARGET,
    )
    assert decision.selected_reason is OpenPositionExitReason.GAP_THROUGH_TARGET
    assert decision.intrabar_ambiguity_status is IntrabarAmbiguityStatus.NOT_AMBIGUOUS
