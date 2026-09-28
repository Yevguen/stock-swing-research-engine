"""Suite B: strict canonical Parquet read/write acceptance."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stock_swing_d1.earnings import TimingClass
from stock_swing_d1.earnings.persistence import (
    EarningsPersistenceError,
    read_canonical_revisions,
    write_canonical_revisions,
)


def _replace_column(table: pa.Table, name: str, values: list[object]) -> pa.Table:
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def test_persist_b01_full_revision_round_trip_preserves_semantics(
    tmp_path, make_persist_revision
):
    """PERSIST_B01"""
    revision = make_persist_revision()
    path = tmp_path / "canonical.parquet"
    artifact = write_canonical_revisions(path, [revision])
    assert artifact.row_count == 1
    assert artifact.path == str(path)
    assert len(artifact.sha256) == 64
    assert read_canonical_revisions(path) == (revision,)


def test_persist_b02_nullable_fields_round_trip_as_none(
    tmp_path, make_persist_revision
):
    """PERSIST_B02"""
    revision = make_persist_revision(
        provider_event_id=None,
        provider_record_id=None,
        historical_symbol=None,
        historical_exchange=None,
        knowledge_date=None,
        provider_sequence=None,
        scheduled_at=None,
        event_timezone=None,
        actual_event_date=None,
        actual_event_at=None,
        correction_of_revision_id=None,
    )
    path = tmp_path / "nullable.parquet"
    write_canonical_revisions(path, [revision])
    loaded = read_canonical_revisions(path)[0]
    for name in (
        "provider_event_id",
        "provider_record_id",
        "historical_symbol",
        "historical_exchange",
        "knowledge_date",
        "provider_sequence",
        "scheduled_at",
        "event_timezone",
        "actual_event_date",
        "actual_event_at",
        "correction_of_revision_id",
    ):
        assert getattr(loaded, name) is None


def test_persist_b03_timezone_aware_values_round_trip_through_utc(
    tmp_path, make_persist_revision
):
    """PERSIST_B03"""
    eastern = timezone(timedelta(hours=-4))
    known_at = datetime(2026, 9, 1, 10, 0, 0, 123456, tzinfo=eastern)
    scheduled_at = datetime(2026, 10, 29, 16, 30, 0, 654321, tzinfo=eastern)
    revision = make_persist_revision(
        knowledge_available_at=known_at,
        strategy_effective_at=known_at,
        timing_class=TimingClass.EXACT_TIME,
        scheduled_at=scheduled_at,
        actual_event_date=date(2026, 10, 29),
        actual_event_at=scheduled_at,
        ingested_at=known_at + timedelta(minutes=5),
    )
    path = tmp_path / "timezone.parquet"
    write_canonical_revisions(path, [revision])
    loaded = read_canonical_revisions(path)[0]
    for name in (
        "knowledge_available_at",
        "strategy_effective_at",
        "scheduled_at",
        "actual_event_at",
        "ingested_at",
    ):
        value = getattr(loaded, name)
        assert value is not None
        assert value.tzinfo == timezone.utc
        assert value == getattr(revision, name)
    assert loaded.knowledge_available_at.microsecond == 123456


def test_persist_b04_writer_sorts_unsorted_input_canonically(
    tmp_path, make_persist_revision
):
    """PERSIST_B04"""
    last = make_persist_revision(
        canonical_asset_id="SYNTH-ASSET-Z",
        event_instance_id="EVENT-Z",
        event_revision_id="REV-Z",
    )
    first = make_persist_revision(
        canonical_asset_id="SYNTH-ASSET-A",
        event_instance_id="EVENT-A",
        event_revision_id="REV-A",
    )
    path = tmp_path / "sorted.parquet"
    write_canonical_revisions(path, [last, first])
    assert read_canonical_revisions(path) == (first, last)


def test_persist_b05_writer_rejects_duplicate_revision_identity(
    tmp_path, make_persist_revision
):
    """PERSIST_B05"""
    revision = make_persist_revision()
    duplicate = replace(
        revision, ingested_at=revision.ingested_at + timedelta(minutes=1)
    )
    with pytest.raises(EarningsPersistenceError) as caught:
        write_canonical_revisions(tmp_path / "duplicate.parquet", [revision, duplicate])
    assert caught.value.code == "DUPLICATE_REVISION_KEY"


def test_persist_b06_reader_rejects_missing_schema_column(
    tmp_path, make_persist_revision
):
    """PERSIST_B06"""
    path = tmp_path / "missing.parquet"
    write_canonical_revisions(path, [make_persist_revision()])
    table = pq.read_table(path).drop(["provider_event_id"])
    pq.write_table(table, path)
    with pytest.raises(EarningsPersistenceError) as caught:
        read_canonical_revisions(path)
    assert caught.value.code == "SCHEMA_MISMATCH"


def test_persist_b07_reader_rejects_extra_schema_column(
    tmp_path, make_persist_revision
):
    """PERSIST_B07"""
    path = tmp_path / "extra.parquet"
    write_canonical_revisions(path, [make_persist_revision()])
    table = pq.read_table(path).append_column("unexpected", pa.array(["x"]))
    pq.write_table(table, path)
    with pytest.raises(EarningsPersistenceError) as caught:
        read_canonical_revisions(path)
    assert caught.value.code == "SCHEMA_MISMATCH"


def test_persist_b08_reader_rejects_type_or_nullability_mismatch(
    tmp_path, make_persist_revision
):
    """PERSIST_B08"""
    path = tmp_path / "type.parquet"
    write_canonical_revisions(path, [make_persist_revision()])
    table = pq.read_table(path)
    index = table.schema.get_field_index("provider_sequence")
    table = table.set_column(index, "provider_sequence", pa.array([1], pa.int32()))
    pq.write_table(table, path)
    with pytest.raises(EarningsPersistenceError) as caught:
        read_canonical_revisions(path)
    assert caught.value.code == "SCHEMA_MISMATCH"


def test_persist_b09_reader_detects_canonical_hash_mismatch(
    tmp_path, make_persist_revision
):
    """PERSIST_B09"""
    path = tmp_path / "hash.parquet"
    write_canonical_revisions(path, [make_persist_revision()])
    table = _replace_column(
        pq.read_table(path), "scheduled_date", [date(2026, 11, 1)]
    )
    pq.write_table(table, path)
    with pytest.raises(EarningsPersistenceError) as caught:
        read_canonical_revisions(path)
    assert caught.value.code == "CANONICAL_HASH_MISMATCH"


def test_persist_b10_reader_rejects_noncanonical_enum_value(
    tmp_path, make_persist_revision
):
    """PERSIST_B10"""
    path = tmp_path / "enum.parquet"
    write_canonical_revisions(path, [make_persist_revision()])
    table = _replace_column(pq.read_table(path), "timing_class", ["NOT_CANONICAL"])
    pq.write_table(table, path)
    with pytest.raises(EarningsPersistenceError) as caught:
        read_canonical_revisions(path)
    assert caught.value.code == "SCHEMA_MISMATCH"


def test_persist_b11_reader_rejects_nondeterministic_row_order(
    tmp_path, make_persist_revision
):
    """PERSIST_B11"""
    first = make_persist_revision(
        canonical_asset_id="ASSET-A",
        event_instance_id="EVENT-A",
        event_revision_id="REV-A",
    )
    second = make_persist_revision(
        canonical_asset_id="ASSET-B",
        event_instance_id="EVENT-B",
        event_revision_id="REV-B",
    )
    path = tmp_path / "unsorted.parquet"
    write_canonical_revisions(path, [first, second])
    table = pq.read_table(path).take(pa.array([1, 0]))
    pq.write_table(table, path)
    with pytest.raises(EarningsPersistenceError) as caught:
        read_canonical_revisions(path, verify_sort_order=True)
    assert caught.value.code == "UNSORTED_CANONICAL_DATASET"
    assert read_canonical_revisions(path, verify_sort_order=False) == (second, first)
