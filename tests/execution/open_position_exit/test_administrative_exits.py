"""Maximum-holding and PIT earnings close-boundary tests."""

from datetime import date

import pytest

from stock_swing_d1.earnings.models import (
    EarningsScheduleStatus,
    LifecycleState,
    TimingClass,
)
from stock_swing_d1.execution.open_position_exit import (
    EarningsExitStatus,
    ExitBoundary,
    OpenPositionExitEvaluationInput,
    OpenPositionExitEvaluator,
    OpenPositionExitReason,
)
from tests.execution.open_position_exit.conftest import (
    ExplicitTradingCalendar,
    SESSIONS,
)


def _known_by_prior_boundary(
    make_earnings_decision,
    *,
    current_index: int,
    scheduled_date: date,
    timing_class: TimingClass,
):
    prior = make_earnings_decision(
        boundary_session=SESSIONS[current_index - 1],
        scheduled_date=scheduled_date,
        timing_class=timing_class,
    )
    current = make_earnings_decision(
        boundary_session=SESSIONS[current_index],
        scheduled_date=scheduled_date,
        timing_class=timing_class,
    )
    return prior, current


def _no_active_earnings(make_earnings_decision, *, boundary_session: date):
    return make_earnings_decision(
        boundary_session=boundary_session,
        scheduled_date=None,
        earnings_schedule_known=True,
        lifecycle_state=LifecycleState.CANCELLED,
        schedule_status=EarningsScheduleStatus.NO_EVENT_WITHIN_PROVIDER_HORIZON,
    )


def test_max_hold_session_9_does_not_force_exit(evaluator, make_input) -> None:
    decision = evaluator.evaluate(make_input(session_index=8))

    assert decision.holding_session_number == 9
    assert decision.exit_required is False
    assert OpenPositionExitReason.MAX_HOLDING not in decision.triggered_reasons


def test_max_hold_session_10_exits_at_close(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(session_index=9, bar=make_bar(session=SESSIONS[9], close=103.0))
    )

    assert decision.holding_session_number == 10
    assert decision.selected_reason is OpenPositionExitReason.MAX_HOLDING
    assert decision.exit_boundary is ExitBoundary.CLOSE
    assert decision.reference_exit_price == 103.0


def test_weekend_does_not_increment_holding_count(evaluator, make_input) -> None:
    monday_after_weekend = evaluator.evaluate(make_input(session_index=5))

    assert (SESSIONS[5] - SESSIONS[4]).days == 3
    assert monday_after_weekend.holding_session_number == 6


def test_exchange_holiday_does_not_increment_holding_count(make_state, make_bar) -> None:
    friday = date(2026, 9, 4)
    tuesday = date(2026, 9, 8)
    calendar = ExplicitTradingCalendar((friday, tuesday))
    evaluator = OpenPositionExitEvaluator(trading_calendar=calendar)
    base_state = make_state()
    state = base_state._validated(
        security_id=base_state.security_id,
        symbol=base_state.symbol,
        signal_session=date(2026, 9, 3),
        signal_time=base_state.signal_time.replace(year=2026, month=9, day=3),
        entry_session=friday,
        entry_price=base_state.entry_price,
        signal_atr_fraction=base_state.signal_atr_fraction,
        risk_fraction=base_state.risk_fraction,
        stop_price=base_state.stop_price,
        take_profit_price=base_state.take_profit_price,
        last_evaluated_session=friday,
    )
    bar = make_bar(session=tuesday)

    decision = evaluator.evaluate(
        OpenPositionExitEvaluationInput(
            session=tuesday,
            protective_state=state,
            market_bar=bar,
        )
    )

    assert decision.holding_session_number == 2


def test_stop_precedes_max_hold_close(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(
            session_index=9,
            bar=make_bar(session=SESSIONS[9], low=95.0, close=97.0),
        )
    )

    assert decision.selected_reason is OpenPositionExitReason.STOP_LOSS
    assert decision.exit_boundary is ExitBoundary.INTRADAY
    assert OpenPositionExitReason.MAX_HOLDING not in decision.triggered_reasons


def test_target_precedes_max_hold_close(evaluator, make_input, make_bar) -> None:
    decision = evaluator.evaluate(
        make_input(
            session_index=9,
            bar=make_bar(session=SESSIONS[9], high=109.0, close=107.0),
        )
    )

    assert decision.selected_reason is OpenPositionExitReason.TAKE_PROFIT
    assert decision.exit_boundary is ExitBoundary.INTRADAY
    assert OpenPositionExitReason.MAX_HOLDING not in decision.triggered_reasons


def test_earnings_close_without_earlier_trigger(
    evaluator, make_input, make_bar, make_earnings_decision
) -> None:
    current = SESSIONS[5]
    prior, earnings = _known_by_prior_boundary(
        make_earnings_decision,
        current_index=5,
        scheduled_date=SESSIONS[6],
        timing_class=TimingClass.BMO,
    )
    decision = evaluator.evaluate(
        make_input(
            session_index=5,
            bar=make_bar(session=current, close=102.0),
            earnings_decision=earnings,
            prior_boundary_earnings_decision=prior,
        )
    )

    assert decision.selected_reason is OpenPositionExitReason.EARNINGS_FORCED_EXIT
    assert decision.exit_boundary is ExitBoundary.CLOSE
    assert decision.reference_exit_price == 102.0
    assert decision.earnings_exit_status is EarningsExitStatus.EXIT_REQUIRED_THIS_SESSION


@pytest.mark.parametrize(
    ("bar_values", "expected_reason"),
    [
        ({"low": 95.0, "close": 97.0}, OpenPositionExitReason.STOP_LOSS),
        ({"high": 109.0, "close": 107.0}, OpenPositionExitReason.TAKE_PROFIT),
    ],
)
def test_protective_exit_precedes_earnings_close(
    evaluator,
    make_input,
    make_bar,
    make_earnings_decision,
    bar_values,
    expected_reason,
) -> None:
    current = SESSIONS[5]
    prior, earnings = _known_by_prior_boundary(
        make_earnings_decision,
        current_index=5,
        scheduled_date=SESSIONS[6],
        timing_class=TimingClass.BMO,
    )
    decision = evaluator.evaluate(
        make_input(
            session_index=5,
            bar=make_bar(session=current, **bar_values),
            earnings_decision=earnings,
            prior_boundary_earnings_decision=prior,
        )
    )

    assert decision.selected_reason is expected_reason
    assert OpenPositionExitReason.EARNINGS_FORCED_EXIT not in decision.triggered_reasons
    assert decision.earnings_exit_status is EarningsExitStatus.AVOIDED_BY_EARLIER_EXIT


def test_earnings_beats_max_holding_at_same_close(
    evaluator, make_input, make_bar, make_earnings_decision
) -> None:
    current = SESSIONS[9]
    prior, earnings = _known_by_prior_boundary(
        make_earnings_decision,
        current_index=9,
        scheduled_date=current,
        timing_class=TimingClass.AMC,
    )
    decision = evaluator.evaluate(
        make_input(
            session_index=9,
            bar=make_bar(session=current, close=104.0),
            earnings_decision=earnings,
            prior_boundary_earnings_decision=prior,
        )
    )

    assert decision.triggered_reasons == (
        OpenPositionExitReason.EARNINGS_FORCED_EXIT,
        OpenPositionExitReason.MAX_HOLDING,
    )
    assert decision.selected_reason is OpenPositionExitReason.EARNINGS_FORCED_EXIT
    assert decision.reference_exit_price == 104.0


@pytest.mark.parametrize(
    ("timing", "event_index", "current_index", "deadline_index"),
    [
        (TimingClass.BMO, 6, 5, 5),
        (TimingClass.AMC, 5, 5, 5),
        (TimingClass.DURING_MARKET, 6, 5, 5),
        (TimingClass.UNKNOWN, 6, 5, 5),
    ],
)
def test_earnings_timing_uses_authoritative_deadline_semantics(
    evaluator,
    make_input,
    make_bar,
    make_earnings_decision,
    timing,
    event_index,
    current_index,
    deadline_index,
) -> None:
    current = SESSIONS[current_index]
    prior, earnings = _known_by_prior_boundary(
        make_earnings_decision,
        current_index=current_index,
        scheduled_date=SESSIONS[event_index],
        timing_class=timing,
    )
    decision = evaluator.evaluate(
        make_input(
            session_index=current_index,
            bar=make_bar(session=current),
            earnings_decision=earnings,
            prior_boundary_earnings_decision=prior,
        )
    )

    assert decision.earnings_deadline_session == SESSIONS[deadline_index]
    assert decision.selected_reason is OpenPositionExitReason.EARNINGS_FORCED_EXIT


def test_later_revision_cannot_backdate_exit(
    evaluator, make_input, make_bar, make_earnings_decision, calendar
) -> None:
    current = SESSIONS[6]
    historical_deadline = SESSIONS[5]
    prior = _no_active_earnings(
        make_earnings_decision, boundary_session=SESSIONS[5]
    )
    late = make_earnings_decision(
        boundary_session=current,
        scheduled_date=current,
        timing_class=TimingClass.BMO,
        knowledge_effective_at=calendar.decision_time(current),
    )
    decision = evaluator.evaluate(
        make_input(
            session_index=6,
            bar=make_bar(session=current, close=103.0),
            earnings_decision=late,
            prior_boundary_earnings_decision=prior,
        )
    )

    assert decision.earnings_deadline_session == historical_deadline
    assert decision.session == current
    assert decision.session != historical_deadline
    assert decision.earnings_exit_status is EarningsExitStatus.DEADLINE_MISSED
    assert decision.exit_required is False
    assert decision.selected_reason is None
    assert decision.exit_boundary is None
    assert decision.reference_exit_price is None


def test_late_discovery_records_missed_deadline_and_exits_next_d1_boundary(
    evaluator, make_input, make_bar, make_earnings_decision, calendar
) -> None:
    discovery_session = SESSIONS[6]
    current = SESSIONS[7]
    prior = make_earnings_decision(
        boundary_session=discovery_session,
        scheduled_date=discovery_session,
        timing_class=TimingClass.BMO,
        knowledge_effective_at=calendar.decision_time(discovery_session),
    )
    late = make_earnings_decision(
        boundary_session=current,
        scheduled_date=discovery_session,
        timing_class=TimingClass.BMO,
        knowledge_effective_at=calendar.decision_time(discovery_session),
    )
    decision = evaluator.evaluate(
        make_input(
            session_index=7,
            bar=make_bar(session=current, close=105.0),
            earnings_decision=late,
            prior_boundary_earnings_decision=prior,
        )
    )

    assert decision.earnings_exit_status is EarningsExitStatus.DEADLINE_MISSED
    assert decision.selected_reason is OpenPositionExitReason.EARNINGS_FORCED_EXIT
    assert decision.exit_boundary is ExitBoundary.CLOSE
    assert decision.reference_exit_price == 105.0
