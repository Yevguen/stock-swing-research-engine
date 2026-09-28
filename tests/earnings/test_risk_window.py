from datetime import date, datetime
from zoneinfo import ZoneInfo

from stock_swing_d1.earnings import (
    EarningsScheduleStatus,
    LifecycleState,
    TimingClass,
    evaluate_entry_blackout,
    evaluate_open_position_earnings_risk,
    last_safe_exit_session,
    revalidate_pending_entry,
)


def test_risk_c01_event_inside_ten_session_horizon_blocks_entry(
    make_state, synthetic_calendar
) -> None:
    state = make_state(scheduled_date=synthetic_calendar.session_date(9))

    decision = evaluate_entry_blackout(
        state, synthetic_calendar.session_date(0), synthetic_calendar
    )

    assert decision.entry_blackout is True


def test_risk_c02_event_at_s10_does_not_block_entry(
    make_state, synthetic_calendar
) -> None:
    state = make_state(scheduled_date=synthetic_calendar.session_date(10))

    decision = evaluate_entry_blackout(
        state, synthetic_calendar.session_date(0), synthetic_calendar
    )

    assert decision.entry_blackout is False


def test_risk_c03_estimated_event_blocks_entry(make_state, synthetic_calendar) -> None:
    state = make_state(
        lifecycle_state=LifecycleState.ESTIMATED,
        scheduled_date=synthetic_calendar.session_date(9),
    )

    decision = evaluate_entry_blackout(
        state, synthetic_calendar.session_date(0), synthetic_calendar
    )

    assert decision.entry_blackout is True


def test_risk_c04_bmo_last_safe_exit_is_previous_session(
    make_state, synthetic_calendar
) -> None:
    event_session = synthetic_calendar.session_date(7)
    state = make_state(scheduled_date=event_session, timing_class=TimingClass.BMO)

    result = last_safe_exit_session(state, synthetic_calendar)

    assert result == synthetic_calendar.session_date(6)


def test_risk_c05_amc_last_safe_exit_is_event_session(
    make_state, synthetic_calendar
) -> None:
    event_session = synthetic_calendar.session_date(7)
    state = make_state(scheduled_date=event_session, timing_class=TimingClass.AMC)

    result = last_safe_exit_session(state, synthetic_calendar)

    assert result == event_session


def test_risk_c06_unknown_timing_uses_previous_session(
    make_state, synthetic_calendar
) -> None:
    event_session = synthetic_calendar.session_date(7)
    state = make_state(scheduled_date=event_session, timing_class=TimingClass.UNKNOWN)

    result = last_safe_exit_session(state, synthetic_calendar)

    assert result == synthetic_calendar.session_date(6)


def test_risk_c07_during_market_uses_previous_session(
    make_state, synthetic_calendar
) -> None:
    event_session = synthetic_calendar.session_date(7)
    state = make_state(
        scheduled_date=event_session, timing_class=TimingClass.DURING_MARKET
    )

    result = last_safe_exit_session(state, synthetic_calendar)

    assert result == synthetic_calendar.session_date(6)


def test_risk_c08_confirmation_without_schedule_change_keeps_same_risk_deadline(
    make_state, synthetic_calendar
) -> None:
    event_session = synthetic_calendar.session_date(7)
    estimated = make_state(
        lifecycle_state=LifecycleState.ESTIMATED,
        scheduled_date=event_session,
        timing_class=TimingClass.BMO,
    )
    confirmed = make_state(
        lifecycle_state=LifecycleState.CONFIRMED,
        scheduled_date=event_session,
        timing_class=TimingClass.BMO,
    )

    assert last_safe_exit_session(estimated, synthetic_calendar) == last_safe_exit_session(
        confirmed, synthetic_calendar
    )


def test_risk_c09_postponement_recomputes_pending_forced_exit(
    make_state, synthetic_calendar
) -> None:
    original = make_state(
        scheduled_date=synthetic_calendar.session_date(5),
        timing_class=TimingClass.BMO,
    )
    postponed = make_state(
        scheduled_date=synthetic_calendar.session_date(8),
        timing_class=TimingClass.BMO,
    )
    current_session = synthetic_calendar.session_date(2)

    original_decision = evaluate_open_position_earnings_risk(
        original,
        current_session,
        synthetic_calendar,
        position_entry_session=synthetic_calendar.session_date(0),
    )
    postponed_decision = evaluate_open_position_earnings_risk(
        postponed,
        current_session,
        synthetic_calendar,
        position_entry_session=synthetic_calendar.session_date(0),
    )

    assert original_decision.last_safe_exit_session == synthetic_calendar.session_date(4)
    assert postponed_decision.last_safe_exit_session == synthetic_calendar.session_date(7)
    assert original_decision.mandatory_exit is True
    assert postponed_decision.mandatory_exit is True
    assert original_decision.holding_age_sessions == 2
    assert postponed_decision.holding_age_sessions == 2
    assert postponed_decision.pending_entry_invalidated is False


def test_risk_c10_postponement_after_exit_does_not_reenter(
    make_state, synthetic_calendar
) -> None:
    postponed = make_state(
        scheduled_date=synthetic_calendar.session_date(8),
        timing_class=TimingClass.BMO,
    )

    decision = evaluate_open_position_earnings_risk(
        postponed,
        synthetic_calendar.session_date(5),
        synthetic_calendar,
        position_is_open=False,
    )

    assert decision.mandatory_exit is False
    assert decision.pending_entry_invalidated is False
    assert decision.risk_reason == "POSITION_ALREADY_CLOSED"


def test_risk_c11_advancement_creates_earlier_exit_requirement(
    make_state, synthetic_calendar
) -> None:
    original = make_state(
        scheduled_date=synthetic_calendar.session_date(8),
        timing_class=TimingClass.BMO,
    )
    advanced = make_state(
        scheduled_date=synthetic_calendar.session_date(5),
        timing_class=TimingClass.BMO,
    )

    assert last_safe_exit_session(original, synthetic_calendar) == synthetic_calendar.session_date(7)
    assert last_safe_exit_session(advanced, synthetic_calendar) == synthetic_calendar.session_date(4)


def test_risk_c12_late_advancement_does_not_create_fictional_exit(
    make_state, synthetic_calendar
) -> None:
    event_session = synthetic_calendar.session_date(5)
    learned_at = synthetic_calendar.decision_time(event_session)
    advanced = make_state(
        as_of=learned_at,
        knowledge_effective_at=learned_at,
        scheduled_date=event_session,
        timing_class=TimingClass.BMO,
    )

    decision = evaluate_open_position_earnings_risk(
        advanced, event_session, synthetic_calendar
    )

    assert decision.last_safe_exit_session == synthetic_calendar.session_date(4)
    assert decision.unavoidable_earnings_exposure is True
    assert decision.mandatory_exit is False
    assert decision.risk_reason == "EARNINGS_KNOWN_AFTER_LAST_SAFE_EXIT"


def test_risk_c13_cancellation_removes_unexecuted_forced_exit(
    make_state, synthetic_calendar
) -> None:
    cancelled = make_state(
        earnings_schedule_known=False,
        lifecycle_state=LifecycleState.CANCELLED,
        scheduled_date=None,
    )

    decision = evaluate_open_position_earnings_risk(
        cancelled, synthetic_calendar.session_date(2), synthetic_calendar
    )

    assert decision.mandatory_exit is False
    assert decision.last_safe_exit_session is None


def test_risk_c14_pending_entry_is_revalidated_before_execution(
    make_state, synthetic_calendar
) -> None:
    ny = ZoneInfo("America/New_York")
    signal_time = datetime(2026, 10, 5, 16, 0, tzinfo=ny)
    intervening_knowledge = datetime(2026, 10, 6, 9, 0, tzinfo=ny)
    state = make_state(
        as_of=intervening_knowledge,
        knowledge_effective_at=intervening_knowledge,
        scheduled_date=synthetic_calendar.session_date(5),
        timing_class=TimingClass.AMC,
    )

    decision = revalidate_pending_entry(
        signal_time,
        synthetic_calendar.session_date(1),
        state,
        synthetic_calendar,
    )

    assert decision.entry_blackout is True
    assert decision.pending_entry_invalidated is True


def test_risk_c15_unknown_schedule_blocks_new_entry(
    make_state, synthetic_calendar
) -> None:
    unknown = make_state(
        earnings_schedule_known=False,
        lifecycle_state=LifecycleState.UNKNOWN,
        scheduled_date=None,
        schedule_status=EarningsScheduleStatus.UNKNOWN,
    )
    known_absence = make_state(
        earnings_schedule_known=False,
        lifecycle_state=LifecycleState.UNKNOWN,
        scheduled_date=None,
        schedule_status=EarningsScheduleStatus.NO_EVENT_WITHIN_PROVIDER_HORIZON,
    )

    unknown_decision = evaluate_entry_blackout(
        unknown, synthetic_calendar.session_date(0), synthetic_calendar
    )
    known_absence_decision = evaluate_entry_blackout(
        known_absence, synthetic_calendar.session_date(0), synthetic_calendar
    )

    assert unknown_decision.entry_blackout is True
    assert unknown_decision.risk_reason == "EARNINGS_SCHEDULE_UNKNOWN"
    assert known_absence_decision.entry_blackout is False


def test_risk_c16_calendar_evaluation_failure_blocks_entry(
    make_state, synthetic_calendar
) -> None:
    state = make_state(scheduled_date=date(2026, 11, 2))

    decision = evaluate_entry_blackout(
        state, synthetic_calendar.session_date(0), synthetic_calendar
    )

    assert decision.entry_blackout is True
    assert decision.risk_reason == "EARNINGS_CALENDAR_EVALUATION_FAILED"


def test_risk_c17_missed_known_exit_is_not_unavoidable_late_notice(
    make_state, synthetic_calendar
) -> None:
    event_session = synthetic_calendar.session_date(5)
    state = make_state(
        as_of=synthetic_calendar.decision_time(synthetic_calendar.session_date(0)),
        knowledge_effective_at=synthetic_calendar.decision_time(
            synthetic_calendar.session_date(0)
        ),
        scheduled_date=event_session,
        timing_class=TimingClass.BMO,
    )

    decision = evaluate_open_position_earnings_risk(
        state, event_session, synthetic_calendar
    )

    assert decision.last_safe_exit_session == synthetic_calendar.session_date(4)
    assert decision.unavoidable_earnings_exposure is False
    assert decision.risk_reason == "LAST_SAFE_EXIT_MISSED"


def test_risk_c18_known_no_event_within_provider_horizon_allows_entry(
    make_state, synthetic_calendar
) -> None:
    state = make_state(
        earnings_schedule_known=False,
        lifecycle_state=LifecycleState.UNKNOWN,
        scheduled_date=None,
        schedule_status=EarningsScheduleStatus.NO_EVENT_WITHIN_PROVIDER_HORIZON,
    )

    decision = evaluate_entry_blackout(
        state, synthetic_calendar.session_date(0), synthetic_calendar
    )

    assert decision.entry_blackout is False
    assert decision.risk_reason == "NO_EARNINGS_WITHIN_PROVIDER_HORIZON"
