"""Authoritative Phase 6C.8A published earnings PIT acceptance cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone

import pyarrow.parquet as pq
import pytest

from stock_swing_d1.earnings import (
    EarningsScheduleRevision,
    EarningsScheduleStatus,
    EarningsValidationError,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
    TransitionType,
)
from stock_swing_d1.earnings.persistence import (
    EarningsPersistencePaths,
    publish_earnings_revisions,
    read_publication_manifest,
)
from stock_swing_d1.earnings.query import PublishedEarningsPITQuery


PROVIDER = "SYNTHETIC"
ASSET = "NORGATE:1001"


def _revision(
    *,
    revision_id: str = "REV-1",
    event_id: str = "EVENT-1",
    effective_at: datetime = datetime(2026, 9, 1, 14, tzinfo=timezone.utc),
    scheduled_date: date | None = date(2026, 10, 29),
    transition: TransitionType = TransitionType.CREATED,
    lifecycle: LifecycleState = LifecycleState.ESTIMATED,
    provider_sequence: int | None = 1,
    canonical_asset_id: str = ASSET,
    provider_name: str = PROVIDER,
) -> EarningsScheduleRevision:
    return EarningsScheduleRevision(
        event_revision_id=revision_id,
        event_instance_id=event_id,
        canonical_asset_id=canonical_asset_id,
        provider_name=provider_name,
        provider_entity_id=f"ENTITY-{canonical_asset_id}",
        provider_event_id=event_id,
        provider_record_id=revision_id,
        knowledge_date=effective_at.date(),
        knowledge_available_at=effective_at,
        knowledge_precision=KnowledgePrecision.TIMESTAMP,
        strategy_effective_at=effective_at,
        provider_sequence=provider_sequence,
        transition_type=transition,
        lifecycle_state=lifecycle,
        scheduled_date=scheduled_date,
        timing_class=TimingClass.UNKNOWN,
        ingested_at=effective_at + timedelta(minutes=5),
    )


def _paths(tmp_path, name: str = "earnings") -> EarningsPersistencePaths:
    return EarningsPersistencePaths(tmp_path / name)


def _publish(
    paths: EarningsPersistencePaths,
    revisions: list[EarningsScheduleRevision],
    *,
    build_id: str = "QUERY-BUILD",
) -> None:
    result = publish_earnings_revisions(
        revisions,
        paths=paths,
        build_id=build_id,
    )
    assert result.success is True


def _query(paths: EarningsPersistencePaths) -> PublishedEarningsPITQuery:
    return PublishedEarningsPITQuery.from_persistence_paths(paths)


def _at(day: int) -> datetime:
    return datetime(2026, 9, day, 14, tzinfo=timezone.utc)


def _run_query(
    query: PublishedEarningsPITQuery,
    *,
    as_of: datetime = _at(10),
    decision_session: date = date(2026, 9, 10),
    canonical_asset_id: str = ASSET,
    provider_name: str | None = PROVIDER,
):
    return query.query(
        canonical_asset_id=canonical_asset_id,
        provider_name=provider_name,
        as_of=as_of,
        decision_session=decision_session,
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_manifest(paths: EarningsPersistencePaths, **updates: object) -> None:
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    manifest.update(updates)
    paths.manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _assert_invalid_artifact(paths: EarningsPersistencePaths) -> None:
    with pytest.raises(EarningsValidationError) as error:
        _query(paths)
    assert error.value.code == "INVALID_PUBLISHED_ARTIFACT"


def _date_change_history() -> tuple[
    EarningsScheduleRevision, EarningsScheduleRevision
]:
    first = _revision(
        effective_at=_at(1),
        scheduled_date=date(2026, 10, 29),
    )
    changed = replace(
        first,
        event_revision_id="REV-2",
        provider_record_id="REV-2",
        knowledge_date=_at(15).date(),
        knowledge_available_at=_at(15),
        strategy_effective_at=_at(15),
        provider_sequence=2,
        transition_type=TransitionType.DATE_CHANGED,
        scheduled_date=date(2026, 11, 5),
        ingested_at=_at(15) + timedelta(minutes=5),
    )
    return first, changed


def _confirmation_history() -> tuple[
    EarningsScheduleRevision, EarningsScheduleRevision
]:
    estimated = _revision(effective_at=_at(1))
    confirmed = replace(
        estimated,
        event_revision_id="REV-2",
        provider_record_id="REV-2",
        knowledge_date=_at(5).date(),
        knowledge_available_at=_at(5),
        strategy_effective_at=_at(5),
        provider_sequence=2,
        transition_type=TransitionType.CONFIRMED,
        lifecycle_state=LifecycleState.CONFIRMED,
        ingested_at=_at(5) + timedelta(minutes=5),
    )
    return estimated, confirmed


def _cancellation_history() -> tuple[
    EarningsScheduleRevision, EarningsScheduleRevision
]:
    active = _revision(effective_at=_at(1))
    cancelled = replace(
        active,
        event_revision_id="REV-2",
        provider_record_id="REV-2",
        knowledge_date=_at(10).date(),
        knowledge_available_at=_at(10),
        strategy_effective_at=_at(10),
        provider_sequence=2,
        transition_type=TransitionType.CANCELLED,
        lifecycle_state=LifecycleState.CANCELLED,
        scheduled_date=None,
        ingested_at=_at(10) + timedelta(minutes=5),
    )
    return active, cancelled


# Suite A — Published Artifact Integrity and Snapshot Binding


def test_QRY_A01_valid_publication_identity(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()], build_id="BUILD-IDENTITY")

    query = _query(paths)
    manifest = read_publication_manifest(paths.manifest_path)

    assert query.build_id == manifest.build_id == "BUILD-IDENTITY"
    assert query.output_sha256 == manifest.output_sha256
    assert query.output_sha256 == _sha256(paths.published_path)


@pytest.mark.hard_gate
def test_QRY_A02_missing_publication_pair(tmp_path) -> None:
    for missing_member in ("both", "parquet", "manifest"):
        paths = _paths(tmp_path, f"missing-{missing_member}")
        _publish(paths, [_revision()])
        if missing_member in {"both", "parquet"}:
            paths.published_path.unlink()
        if missing_member in {"both", "manifest"}:
            paths.manifest_path.unlink()
        _assert_invalid_artifact(paths)


@pytest.mark.hard_gate
def test_QRY_A03_corrupt_parquet_or_sha_mismatch(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])
    corrupted = bytearray(paths.published_path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 0x01
    paths.published_path.write_bytes(corrupted)

    _assert_invalid_artifact(paths)


@pytest.mark.hard_gate
def test_QRY_A04_published_counts_mismatch(tmp_path) -> None:
    for field_name in (
        "published_record_count",
        "published_event_count",
        "published_asset_count",
    ):
        paths = _paths(tmp_path, field_name)
        _publish(paths, [_revision()])
        _rewrite_manifest(paths, **{field_name: 2})
        _assert_invalid_artifact(paths)


@pytest.mark.hard_gate
def test_QRY_A05_effective_bounds_mismatch(tmp_path) -> None:
    incorrect_bounds = {
        "minimum_strategy_effective_at": "2026-08-31T14:00:00.000000Z",
        "maximum_strategy_effective_at": "2026-09-02T14:00:00.000000Z",
    }
    for field_name, incorrect_value in incorrect_bounds.items():
        paths = _paths(tmp_path, field_name)
        _publish(paths, [_revision()])
        _rewrite_manifest(paths, **{field_name: incorrect_value})
        _assert_invalid_artifact(paths)


@pytest.mark.hard_gate
def test_QRY_A06_invalid_canonical_schema_artifact(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])
    table = pq.read_table(paths.published_path)
    pq.write_table(table.replace_schema_metadata({}), paths.published_path)
    _rewrite_manifest(paths, output_sha256=_sha256(paths.published_path))

    _assert_invalid_artifact(paths)


def test_QRY_A07_manifest_output_filename_mismatch(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])
    _rewrite_manifest(paths, output_file="different_snapshot.parquet")

    _assert_invalid_artifact(paths)


def test_QRY_A08_invalid_required_manifest_identity(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])
    _rewrite_manifest(paths, build_id="")

    _assert_invalid_artifact(paths)


@pytest.mark.hard_gate
def test_QRY_A09_cross_snapshot_parquet_manifest_mixing(tmp_path) -> None:
    first_paths = _paths(tmp_path, "first")
    second_paths = _paths(tmp_path, "second")
    _publish(first_paths, [_revision()], build_id="FIRST-SNAPSHOT")
    _publish(
        second_paths,
        [
            _revision(
                revision_id="OTHER-REVISION",
                event_id="OTHER-EVENT",
                scheduled_date=date(2026, 11, 20),
            )
        ],
        build_id="SECOND-SNAPSHOT",
    )

    with pytest.raises(EarningsValidationError) as error:
        PublishedEarningsPITQuery(
            first_paths.published_path,
            second_paths.manifest_path,
        )

    assert error.value.code == "INVALID_PUBLISHED_ARTIFACT"


@pytest.mark.hard_gate
def test_QRY_A10_immutable_snapshot_binding(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(
        paths,
        [_revision(scheduled_date=date(2026, 10, 29))],
        build_id="FIRST",
    )
    bound_query = _query(paths)
    first_sha = bound_query.output_sha256

    _publish(
        paths,
        [
            _revision(
                revision_id="REV-2",
                event_id="EVENT-2",
                effective_at=_at(2),
                scheduled_date=date(2026, 10, 10),
            )
        ],
        build_id="SECOND",
    )
    new_query = _query(paths)

    assert _run_query(bound_query).event_instance_id == "EVENT-1"
    assert bound_query.build_id == "FIRST"
    assert bound_query.output_sha256 == first_sha
    assert _run_query(new_query).event_instance_id == "EVENT-2"
    assert new_query.build_id == "SECOND"
    assert new_query.output_sha256 != first_sha
    with pytest.raises(FrozenInstanceError):
        bound_query._manifest = None  # type: ignore[assignment]


# Suite B — PIT Replay and Anti-Look-Ahead


@pytest.mark.hard_gate
def test_QRY_B01_before_first_effective_revision(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision(effective_at=_at(15))])

    state = _run_query(_query(paths), as_of=_at(15) - timedelta(microseconds=1))

    assert state.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert state.event_instance_id is None
    assert state.scheduled_date is None
    assert state.earnings_schedule_known is False


@pytest.mark.hard_gate
def test_QRY_B02_exactly_at_first_effective_revision(tmp_path) -> None:
    paths = _paths(tmp_path)
    first = _revision(effective_at=_at(1))
    _publish(paths, [first])

    state = _run_query(_query(paths), as_of=_at(1))

    assert state.event_instance_id == first.event_instance_id
    assert state.event_revision_id == first.event_revision_id
    assert state.scheduled_date == first.scheduled_date


@pytest.mark.hard_gate
def test_QRY_B03_immediately_before_later_revision(tmp_path) -> None:
    paths = _paths(tmp_path)
    first, changed = _date_change_history()
    _publish(paths, [first, changed])

    state = _run_query(
        _query(paths),
        as_of=changed.strategy_effective_at - timedelta(microseconds=1),
    )

    assert state.event_revision_id == first.event_revision_id
    assert state.scheduled_date == first.scheduled_date
    assert state.scheduled_date != changed.scheduled_date


@pytest.mark.hard_gate
def test_QRY_B04_exactly_at_later_revision(tmp_path) -> None:
    paths = _paths(tmp_path)
    first, changed = _date_change_history()
    _publish(paths, [first, changed])

    state = _run_query(_query(paths), as_of=changed.strategy_effective_at)

    assert state.event_revision_id == changed.event_revision_id
    assert state.scheduled_date == changed.scheduled_date


@pytest.mark.hard_gate
def test_QRY_B05_future_correction_cannot_rewrite_earlier_history(tmp_path) -> None:
    paths = _paths(tmp_path)
    first, changed = _date_change_history()
    _publish(paths, [first, changed])
    query = _query(paths)

    before = _run_query(query, as_of=_at(10))
    after = _run_query(query, as_of=_at(16))

    assert before.event_revision_id == first.event_revision_id
    assert before.scheduled_date == first.scheduled_date
    assert after.event_revision_id == changed.event_revision_id
    assert after.scheduled_date == changed.scheduled_date


def test_QRY_B06_confirmation_transition_pit_timing(tmp_path) -> None:
    paths = _paths(tmp_path)
    estimated, confirmed = _confirmation_history()
    _publish(paths, [estimated, confirmed])
    query = _query(paths)

    before = _run_query(query, as_of=_at(5) - timedelta(microseconds=1))
    at_confirmation = _run_query(query, as_of=_at(5))

    assert before.lifecycle_state is LifecycleState.ESTIMATED
    assert at_confirmation.lifecycle_state is LifecycleState.CONFIRMED


def test_QRY_B07_cancellation_transition_pit_timing(tmp_path) -> None:
    paths = _paths(tmp_path)
    active, cancelled = _cancellation_history()
    _publish(paths, [active, cancelled])
    query = _query(paths)

    before = _run_query(query, as_of=_at(10) - timedelta(microseconds=1))
    at_cancellation = _run_query(query, as_of=_at(10))

    assert before.event_instance_id == active.event_instance_id
    assert before.lifecycle_state is LifecycleState.ESTIMATED
    assert at_cancellation.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert at_cancellation.event_instance_id is None


@pytest.mark.hard_gate
def test_QRY_B08_independent_event_instance_replay(tmp_path) -> None:
    paths = _paths(tmp_path)
    active_first_event = _revision(
        event_id="EVENT-ACTIVE",
        effective_at=_at(1),
        scheduled_date=date(2026, 10, 20),
    )
    second_event = _revision(
        revision_id="REV-2",
        event_id="EVENT-LATER-CANCELLED",
        effective_at=_at(2),
        scheduled_date=date(2026, 10, 15),
    )
    second_event_cancelled = replace(
        second_event,
        event_revision_id="REV-3",
        provider_record_id="REV-3",
        knowledge_date=_at(5).date(),
        knowledge_available_at=_at(5),
        strategy_effective_at=_at(5),
        provider_sequence=2,
        transition_type=TransitionType.CANCELLED,
        lifecycle_state=LifecycleState.CANCELLED,
        scheduled_date=None,
        ingested_at=_at(5) + timedelta(minutes=5),
    )
    _publish(
        paths,
        [active_first_event, second_event, second_event_cancelled],
    )

    state = _run_query(_query(paths))

    assert state.event_instance_id == "EVENT-ACTIVE"
    assert state.scheduled_date == date(2026, 10, 20)


@pytest.mark.hard_gate
def test_QRY_B09_future_event_identity_non_leakage(tmp_path) -> None:
    paths = _paths(tmp_path)
    future_event_id = "FUTURE-SECRET-EVENT-ID"
    _publish(
        paths,
        [_revision(event_id=future_event_id, effective_at=_at(15))],
    )
    query = _query(paths)

    state = _run_query(query, as_of=_at(10))

    assert state.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert state.event_instance_id is None
    assert future_event_id not in repr(state)
    assert future_event_id not in repr(query)


def test_QRY_B10_replay_independent_of_publication_input_order(tmp_path) -> None:
    revisions = [
        _revision(event_id="EVENT-LATER", scheduled_date=date(2026, 11, 5)),
        _revision(
            revision_id="REV-2",
            event_id="EVENT-NEARER",
            scheduled_date=date(2026, 10, 15),
        ),
    ]
    forward_paths = _paths(tmp_path, "forward")
    reverse_paths = _paths(tmp_path, "reverse")
    _publish(forward_paths, revisions, build_id="ORDER-EQUIVALENT")
    _publish(reverse_paths, list(reversed(revisions)), build_id="ORDER-EQUIVALENT")

    forward = _run_query(_query(forward_paths))
    reverse = _run_query(_query(reverse_paths))

    assert forward == reverse
    assert forward.event_instance_id == "EVENT-NEARER"


# Suite C — Candidate Eligibility and Selection


def test_QRY_C01_estimated_schedule_qualifies(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision(lifecycle=LifecycleState.ESTIMATED)])

    state = _run_query(_query(paths))

    assert state.lifecycle_state is LifecycleState.ESTIMATED
    assert state.earnings_schedule_known is True
    assert state.schedule_status is EarningsScheduleStatus.KNOWN_EVENT


def test_QRY_C02_confirmed_schedule_qualifies(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision(lifecycle=LifecycleState.CONFIRMED)])

    state = _run_query(_query(paths))

    assert state.lifecycle_state is LifecycleState.CONFIRMED
    assert state.earnings_schedule_known is True
    assert state.schedule_status is EarningsScheduleStatus.KNOWN_EVENT


def test_QRY_C03_cancelled_event_excluded(tmp_path) -> None:
    paths = _paths(tmp_path)
    cancelled = _revision(
        scheduled_date=None,
        transition=TransitionType.CANCELLED,
        lifecycle=LifecycleState.CANCELLED,
    )
    _publish(paths, [cancelled])

    state = _run_query(_query(paths))

    assert state.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert state.event_instance_id is None


@pytest.mark.hard_gate
def test_QRY_C04_unknown_schedule_excluded(tmp_path) -> None:
    paths = _paths(tmp_path)
    unknown_schedule = _revision(
        scheduled_date=None,
        lifecycle=LifecycleState.UNKNOWN,
    )
    _publish(paths, [unknown_schedule])

    state = _run_query(_query(paths))

    assert state.earnings_schedule_known is False
    assert state.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert state.event_instance_id is None


def test_QRY_C05_non_active_lifecycle_excluded_despite_date(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision(lifecycle=LifecycleState.UNKNOWN)])

    state = _run_query(_query(paths))

    assert state.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert state.event_instance_id is None


@pytest.mark.hard_gate
def test_QRY_C06_past_event_excluded(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision(scheduled_date=date(2026, 9, 5))])

    state = _run_query(
        _query(paths),
        decision_session=date(2026, 9, 10),
    )

    assert state.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert state.event_instance_id is None


def test_QRY_C07_decision_session_equality_is_inclusive(tmp_path) -> None:
    paths = _paths(tmp_path)
    decision_session = date(2026, 9, 10)
    _publish(paths, [_revision(scheduled_date=decision_session)])

    state = _run_query(_query(paths), decision_session=decision_session)

    assert state.event_instance_id == "EVENT-1"
    assert state.scheduled_date == decision_session


@pytest.mark.hard_gate
def test_QRY_C08_nearest_qualifying_event_selected(tmp_path) -> None:
    paths = _paths(tmp_path)
    later = _revision(event_id="EVENT-LATER", scheduled_date=date(2026, 11, 5))
    nearer = _revision(
        revision_id="REV-2",
        event_id="EVENT-NEARER",
        scheduled_date=date(2026, 10, 15),
    )
    _publish(paths, [later, nearer])

    state = _run_query(_query(paths))

    assert state.event_instance_id == "EVENT-NEARER"
    assert state.scheduled_date == date(2026, 10, 15)


def test_QRY_C09_nearer_ineligible_does_not_displace_later_valid(tmp_path) -> None:
    paths = _paths(tmp_path)
    nearer_ineligible = _revision(
        event_id="EVENT-NEARER-INELIGIBLE",
        lifecycle=LifecycleState.UNKNOWN,
        scheduled_date=date(2026, 10, 15),
    )
    later_valid = _revision(
        revision_id="REV-2",
        event_id="EVENT-LATER-VALID",
        scheduled_date=date(2026, 10, 20),
    )
    _publish(paths, [nearer_ineligible, later_valid])

    state = _run_query(_query(paths))

    assert state.event_instance_id == "EVENT-LATER-VALID"
    assert state.scheduled_date == date(2026, 10, 20)


@pytest.mark.hard_gate
def test_QRY_C10_equal_nearest_dates_fail_closed_in_both_orders(tmp_path) -> None:
    first = _revision(event_id="EVENT-1", scheduled_date=date(2026, 10, 15))
    second = _revision(
        revision_id="REV-2",
        event_id="EVENT-2",
        scheduled_date=date(2026, 10, 15),
    )
    later = _revision(
        revision_id="REV-3",
        event_id="EVENT-3",
        scheduled_date=date(2026, 11, 5),
    )
    publication_input = [later, second, first]

    for order_name, revisions in (
        ("forward", publication_input),
        ("reverse", list(reversed(publication_input))),
    ):
        paths = _paths(tmp_path, order_name)
        _publish(paths, revisions)

        with pytest.raises(EarningsValidationError) as error:
            _run_query(_query(paths))

        assert error.value.code == "AMBIGUOUS_ACTIVE_EARNINGS_EVENTS"


# Suite D — Public Boundary, Conservative Semantics, Determinism


@pytest.mark.hard_gate
def test_QRY_D01_as_of_must_be_timezone_aware_datetime(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])
    query = _query(paths)

    for invalid_as_of in (
        datetime(2026, 9, 10, 14),
        date(2026, 9, 10),
        "2026-09-10T14:00:00Z",
        None,
    ):
        with pytest.raises(EarningsValidationError) as error:
            _run_query(query, as_of=invalid_as_of)
        assert error.value.code == "INVALID_QUERY_TIMESTAMP"


def test_QRY_D02_aware_non_utc_as_of_accepted(tmp_path) -> None:
    paths = _paths(tmp_path)
    effective_at = _at(10)
    _publish(paths, [_revision(effective_at=effective_at)])
    eastern_offset = timezone(timedelta(hours=-4))

    state = _run_query(
        _query(paths),
        as_of=datetime(2026, 9, 10, 10, tzinfo=eastern_offset),
    )

    assert state.event_instance_id == "EVENT-1"
    assert state.knowledge_effective_at == effective_at


def test_QRY_D03_decision_session_strictly_date_only(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])
    query = _query(paths)

    for invalid_session in (
        datetime(2026, 9, 10, tzinfo=timezone.utc),
        "2026-09-10",
        None,
        20260910,
    ):
        with pytest.raises(EarningsValidationError) as error:
            _run_query(query, decision_session=invalid_session)
        assert error.value.code == "INVALID_QUERY_SESSION"


def test_QRY_D04_strict_canonical_asset_id(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])
    query = _query(paths)

    for invalid_asset_id in (
        "",
        "NORGATE:0",
        "NORGATE:01",
        "norgate:1",
        "NORGATE:ABC",
        " NORGATE:1",
    ):
        with pytest.raises(EarningsValidationError) as error:
            _run_query(query, canonical_asset_id=invalid_asset_id)
        assert error.value.code == "INVALID_QUERY_ASSET_ID"


@pytest.mark.hard_gate
def test_QRY_D05_omitted_provider_uses_domain_error(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])

    with pytest.raises(EarningsValidationError) as error:
        _query(paths).query(
            canonical_asset_id=ASSET,
            as_of=_at(10),
            decision_session=date(2026, 9, 10),
        )

    assert error.value.code == "INVALID_QUERY_PROVIDER"


def test_QRY_D06_invalid_explicit_provider_values(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])
    query = _query(paths)

    for invalid_provider in (
        "",
        " ",
        " SYNTHETIC",
        "SYNTHETIC ",
        None,
        1,
    ):
        with pytest.raises(EarningsValidationError) as error:
            _run_query(query, provider_name=invalid_provider)
        assert error.value.code == "INVALID_QUERY_PROVIDER"


def test_QRY_D07_valid_absent_asset_returns_unknown(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])

    state = _run_query(_query(paths), canonical_asset_id="NORGATE:9999")

    assert state.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert state.event_instance_id is None


def test_QRY_D08_valid_unmatched_provider_returns_unknown(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision()])

    state = _run_query(_query(paths), provider_name="OTHER_PROVIDER")

    assert state.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert state.event_instance_id is None


@pytest.mark.hard_gate
def test_QRY_D09_absence_never_proves_provider_horizon(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(paths, [_revision(scheduled_date=date(2026, 10, 29))])

    state = _run_query(
        _query(paths),
        decision_session=date(2026, 12, 1),
    )

    assert state.schedule_status is EarningsScheduleStatus.UNKNOWN
    assert (
        state.schedule_status
        is not EarningsScheduleStatus.NO_EVENT_WITHIN_PROVIDER_HORIZON
    )
    assert state.event_instance_id is None


def test_QRY_D10_deterministic_repeated_query(tmp_path) -> None:
    paths = _paths(tmp_path)
    _publish(
        paths,
        [
            _revision(event_id="EVENT-LATER", scheduled_date=date(2026, 11, 5)),
            _revision(
                revision_id="REV-2",
                event_id="EVENT-NEARER",
                scheduled_date=date(2026, 10, 15),
            ),
        ],
        build_id="DETERMINISTIC-BUILD",
    )
    query = _query(paths)
    build_id = query.build_id
    output_sha256 = query.output_sha256

    results = tuple(_run_query(query) for _ in range(5))

    assert all(result == results[0] for result in results)
    assert query.build_id == build_id == "DETERMINISTIC-BUILD"
    assert query.output_sha256 == output_sha256
