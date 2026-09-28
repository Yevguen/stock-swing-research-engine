from __future__ import annotations

from datetime import date, datetime, timezone

from stock_swing_d1.earnings import (
    EarningsScheduleRevision,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
    TransitionType,
)
from stock_swing_d1.earnings.normalization import normalize_earnings_delivery


def _later(raw_record, revision_key: str, day: int, **overrides):
    return raw_record(
        revision_key=revision_key,
        provider_record_id=f"SYNTHETIC-RECORD-{revision_key}",
        knowledge_date=date(2026, 9, day),
        knowledge_available_at=datetime(2026, 9, day, 14, tzinfo=timezone.utc),
        provider_sequence=day,
        **overrides,
    )


def test_norm_b01_adapter_outputs_only_canonical_schedule_revisions(
    delivery_adapter,
    identity_resolver,
    raw_record,
) -> None:
    batch = delivery_adapter.normalize_delivery(
        (raw_record(),), identity_resolver=identity_resolver
    )
    assert batch.revisions
    assert all(isinstance(item, EarningsScheduleRevision) for item in batch.revisions)


def test_norm_b02_every_emitted_revision_is_a_complete_state_snapshot(
    delivery_adapter,
    identity_resolver,
    raw_record,
    delta_record,
) -> None:
    revisions = delivery_adapter.normalize_delivery(
        (raw_record(timing_code="PRE"), delta_record(lifecycle_code="CONF")),
        identity_resolver=identity_resolver,
    ).revisions
    delta = revisions[-1]
    assert delta.lifecycle_state is LifecycleState.CONFIRMED
    assert delta.scheduled_date == date(2026, 10, 14)
    assert delta.timing_class is TimingClass.BMO


def test_norm_b03_provider_delta_is_resolved_before_canonical_output(
    delivery_adapter,
    identity_resolver,
    raw_record,
    delta_record,
) -> None:
    batch = delivery_adapter.normalize_delivery(
        (delta_record(scheduled_date=date(2026, 10, 21)), raw_record()),
        identity_resolver=identity_resolver,
    )
    assert not batch.failures
    assert batch.revisions[-1].scheduled_date == date(2026, 10, 21)
    assert batch.revisions[-1].lifecycle_state is LifecycleState.ESTIMATED


def test_norm_b04_explicit_none_and_unknown_are_not_implicitly_inherited(
    delivery_adapter,
    identity_resolver,
    raw_record,
    delta_record,
) -> None:
    baseline = raw_record(
        timing_code="CLOCK",
        scheduled_at=datetime(2026, 10, 14, 20, tzinfo=timezone.utc),
        event_timezone="America/New_York",
    )
    delta = delta_record(timing_code="UNK", scheduled_at=None, event_timezone=None)
    revision = delivery_adapter.normalize_delivery(
        (baseline, delta), identity_resolver=identity_resolver
    ).revisions[-1]
    assert revision.timing_class is TimingClass.UNKNOWN
    assert revision.scheduled_at is None
    assert revision.event_timezone is None


def test_norm_b05_schedule_revisions_preserve_event_instance_identity(
    delivery_adapter,
    identity_resolver,
    raw_record,
) -> None:
    records = (
        raw_record(),
        _later(raw_record, "R2", 2, lifecycle_code="CONF"),
        _later(raw_record, "R3", 3, scheduled_date=date(2026, 10, 21)),
        _later(raw_record, "R4", 4, timing_code="POST"),
        _later(raw_record, "R5", 5, lifecycle_code="CANC"),
        _later(raw_record, "R6", 6, lifecycle_code="CONF"),
    )
    revisions = delivery_adapter.normalize_delivery(
        records, identity_resolver=identity_resolver
    ).revisions
    assert len({revision.event_instance_id for revision in revisions}) == 1


def test_norm_b06_new_earnings_cycle_after_occurred_uses_new_event_instance(
    delivery_adapter,
    identity_resolver,
    raw_record,
) -> None:
    occurred = raw_record(
        lifecycle_code="DONE",
        transition_code="OCCUR",
        actual_event_date=date(2026, 10, 14),
    )
    next_cycle = _later(
        raw_record,
        "N1",
        3,
        cycle_id="2026-Q4",
        provider_event_id="SYNTHETIC-EVENT-Q4",
        scheduled_date=date(2027, 1, 20),
    )
    revisions = delivery_adapter.normalize_delivery(
        (occurred, next_cycle), identity_resolver=identity_resolver
    ).revisions
    assert len({revision.event_instance_id for revision in revisions}) == 2


def test_norm_b07_revision_identity_is_deterministic_across_reingestion(
    delivery_adapter,
    identity_resolver,
    raw_record,
) -> None:
    raw = raw_record()
    first = delivery_adapter.normalize_delivery(
        (raw,), identity_resolver=identity_resolver
    ).revisions[0]
    second = delivery_adapter.normalize_delivery(
        (raw,), identity_resolver=identity_resolver
    ).revisions[0]
    assert first.event_revision_id == second.event_revision_id


def test_norm_b08_revision_identity_does_not_depend_on_ingestion_or_row_position(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    records = (raw_record(), _later(raw_record, "R2", 2, lifecycle_code="CONF"))
    first = normalize_earnings_delivery(
        records,
        adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        trading_calendar=normalization_calendar,
        canonical_history=None,
        build_id="norm-b08-first",
        ingested_at=datetime(2026, 9, 8, 10, tzinfo=timezone.utc),
    )
    second = normalize_earnings_delivery(
        reversed(records),
        adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        trading_calendar=normalization_calendar,
        canonical_history=None,
        build_id="norm-b08-second",
        ingested_at=datetime(2026, 9, 9, 10, tzinfo=timezone.utc),
    )
    assert first.success and second.success
    assert {item.event_revision_id for item in first.revisions} == {
        item.event_revision_id for item in second.revisions
    }


def test_norm_b09_later_schedule_date_normalizes_to_postponed(
    delivery_adapter,
    identity_resolver,
    raw_record,
    delta_record,
) -> None:
    revision = delivery_adapter.normalize_delivery(
        (raw_record(), delta_record(scheduled_date=date(2026, 10, 21))),
        identity_resolver=identity_resolver,
    ).revisions[-1]
    assert revision.transition_type is TransitionType.POSTPONED


def test_norm_b10_earlier_schedule_date_normalizes_to_advanced(
    delivery_adapter,
    identity_resolver,
    raw_record,
    delta_record,
) -> None:
    revision = delivery_adapter.normalize_delivery(
        (
            raw_record(scheduled_date=date(2026, 10, 21)),
            delta_record(scheduled_date=date(2026, 10, 14)),
        ),
        identity_resolver=identity_resolver,
    ).revisions[-1]
    assert revision.transition_type is TransitionType.ADVANCED


def test_norm_b11_confirmation_and_unconfirmation_normalize_canonically(
    delivery_adapter,
    identity_resolver,
    raw_record,
    delta_record,
) -> None:
    confirmed = delta_record(lifecycle_code="CONF")
    unconfirmed = delta_record(
        revision_key="R3",
        provider_record_id="SYNTHETIC-RECORD-R3",
        knowledge_date=date(2026, 9, 3),
        knowledge_available_at=datetime(2026, 9, 3, 14, tzinfo=timezone.utc),
        provider_sequence=3,
        lifecycle_code="EST",
    )
    revisions = delivery_adapter.normalize_delivery(
        (unconfirmed, raw_record(), confirmed), identity_resolver=identity_resolver
    ).revisions
    assert revisions[1].transition_type is TransitionType.CONFIRMED
    assert revisions[2].transition_type is TransitionType.UNCONFIRMED


def test_norm_b12_timing_only_change_normalizes_to_timing_changed(
    delivery_adapter,
    identity_resolver,
    raw_record,
    delta_record,
) -> None:
    revision = delivery_adapter.normalize_delivery(
        (raw_record(timing_code="PRE"), delta_record(timing_code="POST")),
        identity_resolver=identity_resolver,
    ).revisions[-1]
    assert revision.transition_type is TransitionType.TIMING_CHANGED


def test_norm_b13_cancellation_and_reinstatement_normalize_canonically(
    delivery_adapter,
    identity_resolver,
    raw_record,
    delta_record,
) -> None:
    cancelled = delta_record(lifecycle_code="CANC")
    reinstated = delta_record(
        revision_key="R3",
        provider_record_id="SYNTHETIC-RECORD-R3",
        knowledge_date=date(2026, 9, 3),
        knowledge_available_at=datetime(2026, 9, 3, 14, tzinfo=timezone.utc),
        provider_sequence=3,
        lifecycle_code="CONF",
    )
    revisions = delivery_adapter.normalize_delivery(
        (reinstated, cancelled, raw_record()), identity_resolver=identity_resolver
    ).revisions
    assert revisions[1].transition_type is TransitionType.CANCELLED
    assert revisions[2].transition_type is TransitionType.REINSTATED


def test_norm_b14_provider_correction_is_distinct_and_targets_prior_revision(
    delivery_adapter,
    identity_resolver,
    raw_record,
    delta_record,
) -> None:
    revisions = delivery_adapter.normalize_delivery(
        (
            raw_record(),
            delta_record(
                scheduled_date=date(2026, 10, 15),
                correction_of_revision_key="R1",
            ),
        ),
        identity_resolver=identity_resolver,
    ).revisions
    correction = revisions[-1]
    assert correction.transition_type is TransitionType.PROVIDER_CORRECTION
    assert correction.is_provider_correction is True
    assert correction.correction_of_revision_id == revisions[0].event_revision_id


def test_norm_b15_isolated_delta_reconstructs_from_prior_same_provider_canonical_history(
    delivery_adapter,
    identity_resolver,
    delta_record,
    canonical_revision,
    history_reader,
) -> None:
    prior = canonical_revision(timing_class=TimingClass.BMO)
    batch = delivery_adapter.normalize_delivery(
        (delta_record(lifecycle_code="CONF"),),
        identity_resolver=identity_resolver,
        canonical_history=history_reader(prior),
    )
    assert not batch.failures
    assert batch.revisions[0].scheduled_date == prior.scheduled_date
    assert batch.revisions[0].timing_class is TimingClass.BMO


def test_norm_b16_delta_does_not_inherit_state_from_different_provider(
    delivery_adapter,
    identity_resolver,
    delta_record,
    canonical_revision,
    history_reader,
) -> None:
    other_provider = canonical_revision(provider_name="OTHER_PROVIDER")
    batch = delivery_adapter.normalize_delivery(
        (delta_record(lifecycle_code="CONF"),),
        identity_resolver=identity_resolver,
        canonical_history=history_reader(other_provider),
    )
    assert batch.revisions == ()
    assert batch.failures[0].error_code == "INCOMPLETE_DELTA_CONTEXT"


def test_norm_b17_delta_without_batch_or_canonical_antecedent_fails_closed(
    delivery_adapter,
    identity_resolver,
    delta_record,
) -> None:
    batch = delivery_adapter.normalize_delivery(
        (delta_record(scheduled_date=date(2026, 10, 21)),),
        identity_resolver=identity_resolver,
    )
    assert batch.revisions == ()
    assert batch.failures[0].error_code == "INCOMPLETE_DELTA_CONTEXT"


def test_norm_b18_future_or_nonprovably_prior_history_cannot_supply_delta_context(
    delivery_adapter,
    identity_resolver,
    delta_record,
    canonical_revision,
    history_reader,
) -> None:
    incoming = delta_record(scheduled_date=date(2026, 10, 21))
    future = canonical_revision(
        knowledge_date=date(2026, 9, 3),
        knowledge_available_at=datetime(2026, 9, 3, 14, tzinfo=timezone.utc),
        strategy_effective_at=datetime(2026, 9, 3, 14, tzinfo=timezone.utc),
    )
    future_batch = delivery_adapter.normalize_delivery(
        (incoming,),
        identity_resolver=identity_resolver,
        canonical_history=history_reader(future),
    )
    same_date = canonical_revision(
        knowledge_precision=KnowledgePrecision.DATE_ONLY,
        knowledge_date=date(2026, 9, 2),
        knowledge_available_at=None,
        strategy_effective_at=datetime(2026, 9, 3, 13, 30, tzinfo=timezone.utc),
    )
    date_only_incoming = delta_record(
        knowledge_precision_code="DATE",
        knowledge_available_at=None,
    )
    same_date_batch = delivery_adapter.normalize_delivery(
        (date_only_incoming,),
        identity_resolver=identity_resolver,
        canonical_history=history_reader(same_date),
    )
    earlier_same_date = canonical_revision(
        knowledge_date=date(2026, 9, 2),
        knowledge_available_at=datetime(2026, 9, 2, 13, tzinfo=timezone.utc),
        strategy_effective_at=datetime(2026, 9, 2, 13, tzinfo=timezone.utc),
    )
    earlier_timestamp_batch = delivery_adapter.normalize_delivery(
        (incoming,),
        identity_resolver=identity_resolver,
        canonical_history=history_reader(earlier_same_date),
    )
    equal_timestamp = canonical_revision(
        knowledge_date=date(2026, 9, 2),
        knowledge_available_at=datetime(2026, 9, 2, 14, tzinfo=timezone.utc),
        strategy_effective_at=datetime(2026, 9, 2, 14, tzinfo=timezone.utc),
    )
    equal_timestamp_batch = delivery_adapter.normalize_delivery(
        (incoming,),
        identity_resolver=identity_resolver,
        canonical_history=history_reader(equal_timestamp),
    )
    assert future_batch.revisions == ()
    assert same_date_batch.revisions == ()
    assert not earlier_timestamp_batch.failures
    assert len(earlier_timestamp_batch.revisions) == 1
    assert earlier_timestamp_batch.revisions[0].scheduled_date == date(2026, 10, 21)
    assert equal_timestamp_batch.revisions == ()
    assert {
        future_batch.failures[0].error_code,
        same_date_batch.failures[0].error_code,
        equal_timestamp_batch.failures[0].error_code,
    } == {"INCOMPLETE_DELTA_CONTEXT"}
