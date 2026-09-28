from dataclasses import replace
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from stock_swing_d1.earnings import (
    EarningsValidationError,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
    TransitionType,
    derive_strategy_effective_at,
    reconstruct_earnings_state,
    validate_revision_history,
)


def _three_revision_history(make_revision, utc_dt):
    return (
        make_revision(
            event_revision_id="R1",
            knowledge_available_at=utc_dt(2026, 9, 1),
            strategy_effective_at=utc_dt(2026, 9, 1),
            scheduled_date=date(2026, 10, 27),
            timing_class=TimingClass.UNKNOWN,
        ),
        make_revision(
            event_revision_id="R2",
            provider_record_id="TEST-RECORD-002",
            knowledge_date=date(2026, 9, 15),
            knowledge_available_at=utc_dt(2026, 9, 15),
            strategy_effective_at=utc_dt(2026, 9, 15),
            transition_type=TransitionType.POSTPONED,
            scheduled_date=date(2026, 10, 29),
            timing_class=TimingClass.AMC,
        ),
        make_revision(
            event_revision_id="R3",
            provider_record_id="TEST-RECORD-003",
            knowledge_date=date(2026, 10, 3),
            knowledge_available_at=utc_dt(2026, 10, 3),
            strategy_effective_at=utc_dt(2026, 10, 3),
            transition_type=TransitionType.CONFIRMED,
            lifecycle_state=LifecycleState.CONFIRMED,
            scheduled_date=date(2026, 10, 29),
            timing_class=TimingClass.AMC,
        ),
    )


def test_pit_a01_future_revision_is_invisible(make_revision, utc_dt) -> None:
    state = reconstruct_earnings_state(
        _three_revision_history(make_revision, utc_dt), utc_dt(2026, 9, 10)
    )

    assert state.lifecycle_state is LifecycleState.ESTIMATED
    assert state.scheduled_date == date(2026, 10, 27)
    assert state.timing_class is TimingClass.UNKNOWN


def test_pit_a02_latest_qualifying_revision_wins(make_revision, utc_dt) -> None:
    state = reconstruct_earnings_state(
        _three_revision_history(make_revision, utc_dt), utc_dt(2026, 9, 20)
    )

    assert state.lifecycle_state is LifecycleState.ESTIMATED
    assert state.scheduled_date == date(2026, 10, 29)
    assert state.timing_class is TimingClass.AMC


def test_pit_a03_unknown_field_does_not_leak_from_future(make_revision, utc_dt) -> None:
    history = (
        make_revision(
            event_revision_id="R1",
            knowledge_available_at=utc_dt(2026, 9, 1),
            strategy_effective_at=utc_dt(2026, 9, 1),
            timing_class=TimingClass.UNKNOWN,
        ),
        make_revision(
            event_revision_id="R2",
            knowledge_available_at=utc_dt(2026, 9, 15),
            strategy_effective_at=utc_dt(2026, 9, 15),
            transition_type=TransitionType.TIMING_CHANGED,
            timing_class=TimingClass.BMO,
        ),
    )

    state = reconstruct_earnings_state(history, utc_dt(2026, 9, 10))

    assert state.timing_class is TimingClass.UNKNOWN


def test_pit_a04_no_qualifying_record_means_unknown_schedule(make_revision, utc_dt) -> None:
    revision = make_revision(
        knowledge_available_at=utc_dt(2026, 9, 15),
        strategy_effective_at=utc_dt(2026, 9, 15),
    )

    state = reconstruct_earnings_state((revision,), utc_dt(2026, 9, 10))

    assert state.earnings_schedule_known is False
    assert state.lifecycle_state is LifecycleState.UNKNOWN


def test_pit_a05_date_only_knowledge_is_not_available_at_midnight(
    make_revision, synthetic_calendar
) -> None:
    source_date = synthetic_calendar.session_date(0)
    revision = make_revision(
        knowledge_date=source_date,
        knowledge_available_at=None,
        knowledge_precision=KnowledgePrecision.DATE_ONLY,
        strategy_effective_at=None,
    )

    effective_at = derive_strategy_effective_at(revision, synthetic_calendar)
    revision = replace(revision, strategy_effective_at=effective_at)
    source_day_end = datetime.combine(
        source_date,
        time(23, 59),
        tzinfo=ZoneInfo("America/New_York"),
    )

    assert reconstruct_earnings_state((revision,), source_day_end).earnings_schedule_known is False
    assert reconstruct_earnings_state((revision,), effective_at).earnings_schedule_known is True
    assert effective_at == synthetic_calendar.decision_time(
        synthetic_calendar.session_date(1)
    )
    assert effective_at.timetz().replace(tzinfo=None) != time(0, 0)


def test_pit_a06_ingested_at_does_not_control_historical_knowledge(make_revision) -> None:
    known_at = datetime(2018, 7, 3, 15, 0, tzinfo=timezone.utc)
    revision = make_revision(
        knowledge_date=known_at.date(),
        knowledge_available_at=known_at,
        strategy_effective_at=known_at,
        scheduled_date=date(2018, 7, 26),
        ingested_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
    )

    state = reconstruct_earnings_state(
        (revision,), datetime(2018, 7, 4, 15, 0, tzinfo=timezone.utc)
    )

    assert state.earnings_schedule_known is True
    assert state.scheduled_date == date(2018, 7, 26)


def test_pit_a07_future_invalid_revision_does_not_break_earlier_state(
    make_revision,
) -> None:
    first_known_at = datetime(2019, 1, 10, 14, 0, tzinfo=timezone.utc)
    future_known_at = datetime(2020, 1, 10, 14, 0, tzinfo=timezone.utc)
    valid = make_revision(
        event_revision_id="R1",
        knowledge_date=first_known_at.date(),
        knowledge_available_at=first_known_at,
        strategy_effective_at=first_known_at,
        scheduled_date=date(2019, 2, 1),
    )
    future_invalid = make_revision(
        event_revision_id="R2",
        knowledge_date=future_known_at.date(),
        knowledge_available_at=future_known_at,
        strategy_effective_at=future_known_at,
        lifecycle_state=LifecycleState.INVALID,
        scheduled_date=None,
    )

    state = reconstruct_earnings_state(
        (valid, future_invalid),
        datetime(2019, 6, 1, 14, 0, tzinfo=timezone.utc),
    )

    assert state.event_revision_id == "R1"
    assert state.scheduled_date == date(2019, 2, 1)
    with pytest.raises(EarningsValidationError):
        validate_revision_history((valid, future_invalid))


def test_pit_a08_future_event_instance_id_is_hidden_before_first_known_revision(
    make_revision, utc_dt, canonical_asset_id
) -> None:
    first_revision = make_revision(
        event_instance_id="EARN-2026-Q3",
        knowledge_date=date(2026, 9, 15),
        knowledge_available_at=utc_dt(2026, 9, 15),
        strategy_effective_at=utc_dt(2026, 9, 15),
    )

    state = reconstruct_earnings_state((first_revision,), utc_dt(2026, 9, 10))

    assert state.canonical_asset_id == canonical_asset_id
    assert state.earnings_schedule_known is False
    assert state.lifecycle_state is LifecycleState.UNKNOWN
    assert state.event_instance_id is None
