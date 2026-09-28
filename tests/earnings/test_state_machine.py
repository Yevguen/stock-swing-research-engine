from datetime import date

import pytest

from stock_swing_d1.earnings import (
    EarningsValidationError,
    LifecycleState,
    TimingClass,
    TransitionType,
    apply_revision,
    reconstruct_earnings_state,
    validate_revision_history,
)


def test_state_b01_estimated_becomes_confirmed(make_revision, utc_dt) -> None:
    estimated = make_revision(event_revision_id="R1")
    confirmed = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 9, 5),
        strategy_effective_at=utc_dt(2026, 9, 5),
        transition_type=TransitionType.CONFIRMED,
        lifecycle_state=LifecycleState.CONFIRMED,
    )

    state = reconstruct_earnings_state((estimated, confirmed), utc_dt(2026, 9, 6))

    assert state.lifecycle_state is LifecycleState.CONFIRMED
    assert state.transition_type is TransitionType.CONFIRMED


def test_state_b02_postponement_is_transition_not_lifecycle_state(
    make_revision, utc_dt
) -> None:
    confirmed = make_revision(
        event_revision_id="R1",
        lifecycle_state=LifecycleState.CONFIRMED,
        scheduled_date=date(2026, 10, 29),
    )
    postponed = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 9, 5),
        strategy_effective_at=utc_dt(2026, 9, 5),
        transition_type=TransitionType.POSTPONED,
        lifecycle_state=LifecycleState.CONFIRMED,
        scheduled_date=date(2026, 10, 30),
    )

    state = reconstruct_earnings_state((confirmed, postponed), utc_dt(2026, 9, 6))

    assert state.lifecycle_state is LifecycleState.CONFIRMED
    assert state.transition_type is TransitionType.POSTPONED
    assert state.scheduled_date == date(2026, 10, 30)


def test_state_b03_advancement_updates_state(make_revision, utc_dt) -> None:
    original = make_revision(event_revision_id="R1", scheduled_date=date(2026, 10, 29))
    advanced = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 9, 5),
        strategy_effective_at=utc_dt(2026, 9, 5),
        transition_type=TransitionType.ADVANCED,
        scheduled_date=date(2026, 10, 27),
        timing_class=TimingClass.BMO,
    )

    state = reconstruct_earnings_state((original, advanced), utc_dt(2026, 9, 6))

    assert state.scheduled_date == date(2026, 10, 27)
    assert state.timing_class is TimingClass.BMO
    assert state.transition_type is TransitionType.ADVANCED


def test_state_b04_confirmed_can_revert_to_estimated(make_revision, utc_dt) -> None:
    confirmed = make_revision(
        event_revision_id="R1", lifecycle_state=LifecycleState.CONFIRMED
    )
    unconfirmed = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 9, 5),
        strategy_effective_at=utc_dt(2026, 9, 5),
        transition_type=TransitionType.UNCONFIRMED,
        lifecycle_state=LifecycleState.ESTIMATED,
    )

    state = reconstruct_earnings_state((confirmed, unconfirmed), utc_dt(2026, 9, 6))

    assert state.lifecycle_state is LifecycleState.ESTIMATED
    assert state.transition_type is TransitionType.UNCONFIRMED


def test_state_b05_cancellation_deactivates_upcoming_event(make_revision, utc_dt) -> None:
    estimated = make_revision(event_revision_id="R1")
    cancelled = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 9, 5),
        strategy_effective_at=utc_dt(2026, 9, 5),
        transition_type=TransitionType.CANCELLED,
        lifecycle_state=LifecycleState.CANCELLED,
        scheduled_date=None,
    )

    state = reconstruct_earnings_state((estimated, cancelled), utc_dt(2026, 9, 6))

    assert state.lifecycle_state is LifecycleState.CANCELLED
    assert state.earnings_schedule_known is False
    assert state.scheduled_date is None


def test_state_b06_cancelled_event_can_be_reinstated(make_revision, utc_dt) -> None:
    estimated = make_revision(event_revision_id="R1")
    cancelled = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 9, 5),
        strategy_effective_at=utc_dt(2026, 9, 5),
        transition_type=TransitionType.CANCELLED,
        lifecycle_state=LifecycleState.CANCELLED,
        scheduled_date=None,
    )
    reinstated = make_revision(
        event_revision_id="R3",
        knowledge_available_at=utc_dt(2026, 9, 8),
        strategy_effective_at=utc_dt(2026, 9, 8),
        transition_type=TransitionType.REINSTATED,
        lifecycle_state=LifecycleState.CONFIRMED,
        scheduled_date=date(2026, 10, 30),
    )

    state = reconstruct_earnings_state(
        (estimated, cancelled, reinstated), utc_dt(2026, 9, 9)
    )

    assert state.lifecycle_state is LifecycleState.CONFIRMED
    assert state.earnings_schedule_known is True
    assert state.transition_type is TransitionType.REINSTATED


def test_state_b07_occurrence_closes_event_instance(make_revision, utc_dt) -> None:
    confirmed = make_revision(
        event_revision_id="R1", lifecycle_state=LifecycleState.CONFIRMED
    )
    occurred = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 10, 30),
        strategy_effective_at=utc_dt(2026, 10, 30),
        transition_type=TransitionType.OCCURRED,
        lifecycle_state=LifecycleState.OCCURRED,
        scheduled_date=None,
        actual_event_date=date(2026, 10, 29),
    )

    state = reconstruct_earnings_state((confirmed, occurred), utc_dt(2026, 10, 31))

    assert state.lifecycle_state is LifecycleState.OCCURRED
    assert state.earnings_schedule_known is False
    assert state.actual_event_date == date(2026, 10, 29)


def test_state_b08_occurred_to_estimated_same_instance_fails(
    make_revision, utc_dt
) -> None:
    occurred = make_revision(
        event_revision_id="R1",
        transition_type=TransitionType.OCCURRED,
        lifecycle_state=LifecycleState.OCCURRED,
        scheduled_date=None,
        actual_event_date=date(2026, 10, 29),
    )
    reused_instance = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 11, 1),
        strategy_effective_at=utc_dt(2026, 11, 1),
        transition_type=TransitionType.CREATED,
        lifecycle_state=LifecycleState.ESTIMATED,
        scheduled_date=date(2027, 1, 28),
    )

    with pytest.raises(EarningsValidationError) as error:
        validate_revision_history((occurred, reused_instance))

    assert error.value.code == "INVALID_STATE_TRANSITION"


def test_state_b09_same_timestamp_requires_deterministic_order(
    make_revision, utc_dt
) -> None:
    same_time = utc_dt(2026, 9, 5)
    first = make_revision(
        event_revision_id="R17",
        knowledge_available_at=same_time,
        strategy_effective_at=same_time,
        provider_sequence=17,
        scheduled_date=date(2026, 10, 27),
    )
    second = make_revision(
        event_revision_id="R18",
        knowledge_available_at=same_time,
        strategy_effective_at=same_time,
        provider_sequence=18,
        transition_type=TransitionType.POSTPONED,
        scheduled_date=date(2026, 10, 29),
    )

    state = reconstruct_earnings_state((second, first), utc_dt(2026, 9, 6))
    assert state.event_revision_id == "R18"
    assert state.scheduled_date == date(2026, 10, 29)

    ambiguous_first = make_revision(
        event_revision_id="RA",
        knowledge_available_at=same_time,
        strategy_effective_at=same_time,
        provider_sequence=None,
        scheduled_date=date(2026, 10, 27),
    )
    ambiguous_second = make_revision(
        event_revision_id="RB",
        knowledge_available_at=same_time,
        strategy_effective_at=same_time,
        provider_sequence=None,
        scheduled_date=date(2026, 10, 29),
    )
    with pytest.raises(EarningsValidationError) as error:
        validate_revision_history((ambiguous_first, ambiguous_second))

    assert error.value.code == "AMBIGUOUS_REVISION_ORDER"


def test_state_b10_provider_correction_is_not_normal_schedule_revision(
    make_revision, utc_dt
) -> None:
    original = make_revision(event_revision_id="R1")
    correction = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 9, 5),
        strategy_effective_at=utc_dt(2026, 9, 5),
        transition_type=TransitionType.PROVIDER_CORRECTION,
        is_provider_correction=True,
        correction_of_revision_id="R1",
        scheduled_date=date(2026, 10, 28),
    )

    ordered = validate_revision_history((original, correction))
    state = apply_revision(None, ordered[-1])

    assert state.transition_type is TransitionType.PROVIDER_CORRECTION
    assert state.is_provider_correction is True


def test_state_b11_transition_and_lifecycle_state_must_be_compatible(
    make_revision,
) -> None:
    invalid_pairs = (
        (TransitionType.CONFIRMED, LifecycleState.ESTIMATED),
        (TransitionType.POSTPONED, LifecycleState.CANCELLED),
        (TransitionType.ADVANCED, LifecycleState.OCCURRED),
        (TransitionType.REINSTATED, LifecycleState.CANCELLED),
    )
    for transition, lifecycle in invalid_pairs:
        revision = make_revision(
            transition_type=transition,
            lifecycle_state=lifecycle,
        )
        with pytest.raises(EarningsValidationError) as error:
            validate_revision_history((revision,))
        assert error.value.code == "INVALID_TRANSITION_STATE_COMBINATION"

    valid_pairs = (
        (TransitionType.CONFIRMED, LifecycleState.CONFIRMED),
        (TransitionType.UNCONFIRMED, LifecycleState.ESTIMATED),
        (TransitionType.POSTPONED, LifecycleState.ESTIMATED),
        (TransitionType.REINSTATED, LifecycleState.CONFIRMED),
    )
    for transition, lifecycle in valid_pairs:
        revision = make_revision(
            transition_type=transition,
            lifecycle_state=lifecycle,
        )
        assert validate_revision_history((revision,)) == (revision,)


def test_state_b12_canonical_revision_is_complete_resulting_state_snapshot(
    make_revision, utc_dt
) -> None:
    original = make_revision(
        event_revision_id="R1",
        scheduled_date=date(2026, 10, 27),
        timing_class=TimingClass.AMC,
    )
    later_complete_snapshot = make_revision(
        event_revision_id="R2",
        knowledge_date=date(2026, 9, 15),
        knowledge_available_at=utc_dt(2026, 9, 15),
        strategy_effective_at=utc_dt(2026, 9, 15),
        transition_type=TransitionType.CONFIRMED,
        lifecycle_state=LifecycleState.CONFIRMED,
        scheduled_date=date(2026, 10, 29),
        timing_class=TimingClass.UNKNOWN,
    )

    state = reconstruct_earnings_state(
        (original, later_complete_snapshot), utc_dt(2026, 9, 20)
    )

    assert state.lifecycle_state is LifecycleState.CONFIRMED
    assert state.scheduled_date == date(2026, 10, 29)
    assert state.timing_class is TimingClass.UNKNOWN
