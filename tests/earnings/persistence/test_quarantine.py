"""Suite E: strict quarantine and fail-closed acceptance."""

from __future__ import annotations

from dataclasses import fields
from datetime import date, datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from stock_swing_d1.earnings import LifecycleState, TransitionType
from stock_swing_d1.earnings.persistence import (
    EarningsPersistencePaths,
    EarningsQuarantineRecord,
    QuarantineStage,
    publish_earnings_revisions,
    read_quarantine_records,
    write_quarantine_records,
)
from stock_swing_d1.earnings.persistence.schema import quarantine_arrow_schema


def _paths(tmp_path) -> EarningsPersistencePaths:
    return EarningsPersistencePaths(tmp_path / "earnings")


def _quarantine_path(paths: EarningsPersistencePaths, build_id: str):
    return paths.quarantine_dir / f"earnings_quarantine_{build_id}.parquet"


def _history_failure_revision(make_persist_revision):
    known_at = datetime(2026, 9, 2, 14, tzinfo=timezone.utc)
    return make_persist_revision(
        event_revision_id="SYNTH-REV-002",
        knowledge_date=known_at.date(),
        knowledge_available_at=known_at,
        strategy_effective_at=known_at,
        provider_sequence=2,
        transition_type=TransitionType.POSTPONED,
        lifecycle_state=LifecycleState.ESTIMATED,
        scheduled_date=date(2026, 10, 20),
        ingested_at=known_at,
    )


def _diagnostic_record(**overrides: object) -> EarningsQuarantineRecord:
    values: dict[str, object] = {
        "quarantine_id": "Q-001",
        "build_id": "BUILD-Q",
        "quarantined_at": datetime(2026, 9, 1, 14, tzinfo=timezone.utc),
        "provider_name": "SYNTHETIC",
        "provider_entity_id": "SYNTH-ENTITY-001",
        "canonical_asset_id": "SYNTH-ASSET-001",
        "event_instance_id": "SYNTH-EVENT-001",
        "event_revision_id": "SYNTH-REV-001",
        "stage": QuarantineStage.RECORD_VALIDATION,
        "error_code": "SYNTHETIC_ERROR",
        "error_message": "synthetic diagnostic only",
        "source_record_locator": None,
        "source_record_hash": None,
        "canonical_record_hash": None,
        "details_json": '{"diagnostic":"synthetic-only"}',
    }
    values.update(overrides)
    return EarningsQuarantineRecord(**values)


def test_persist_e01_record_validation_failure_creates_record_validation_quarantine(
    tmp_path, make_persist_revision
):
    """PERSIST_E01"""
    paths = _paths(tmp_path)
    result = publish_earnings_revisions(
        [make_persist_revision(canonical_asset_id=None)],
        paths=paths,
        build_id="E01",
    )
    records = read_quarantine_records(_quarantine_path(paths, "E01"))
    assert result.success is False
    assert len(records) == 1
    assert records[0].stage is QuarantineStage.RECORD_VALIDATION
    assert records[0].error_code == "MISSING_CANONICAL_ASSET_ID"


def test_persist_e02_history_failure_creates_history_validation_quarantine(
    tmp_path, make_persist_revision
):
    """PERSIST_E02"""
    paths = _paths(tmp_path)
    assert publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="E02-A"
    ).success
    result = publish_earnings_revisions(
        [_history_failure_revision(make_persist_revision)],
        paths=paths,
        build_id="E02-B",
    )
    records = read_quarantine_records(_quarantine_path(paths, "E02-B"))
    assert result.success is False
    assert len(records) == 1
    assert records[0].stage is QuarantineStage.HISTORY_VALIDATION
    assert records[0].error_code == "INVALID_STATE_TRANSITION"


def test_persist_e03_revision_conflict_creates_idempotency_quarantine(
    tmp_path, make_persist_revision
):
    """PERSIST_E03"""
    paths = _paths(tmp_path)
    assert publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="E03-A"
    ).success
    result = publish_earnings_revisions(
        [make_persist_revision(scheduled_date=date(2026, 11, 6))],
        paths=paths,
        build_id="E03-B",
    )
    records = read_quarantine_records(_quarantine_path(paths, "E03-B"))
    assert result.success is False
    assert len(records) == 1
    assert records[0].stage is QuarantineStage.IDEMPOTENCY
    assert records[0].error_code == "IMMUTABLE_REVISION_CONFLICT"


def test_persist_e04_empty_input_creates_publication_level_quarantine(
    tmp_path,
):
    """PERSIST_E04"""
    paths = _paths(tmp_path)
    result = publish_earnings_revisions([], paths=paths, build_id="E04")
    records = read_quarantine_records(_quarantine_path(paths, "E04"))
    assert result.success is False
    assert len(records) == 1
    record = records[0]
    assert record.stage is QuarantineStage.PUBLICATION
    assert record.error_code == "EMPTY_INPUT_DATASET"
    assert record.provider_name is None
    assert record.canonical_asset_id is None
    assert record.event_instance_id is None
    assert record.event_revision_id is None


def test_persist_e05_quarantine_preserves_source_locator_and_hash(tmp_path):
    """PERSIST_E05"""
    record = _diagnostic_record(
        source_record_locator="synthetic-delivery://row/7",
        source_record_hash="a" * 64,
    )
    path = tmp_path / "quarantine.parquet"
    write_quarantine_records(path, [record])
    loaded = read_quarantine_records(path)[0]
    assert loaded.source_record_locator == "synthetic-delivery://row/7"
    assert loaded.source_record_hash == "a" * 64


def test_persist_e06_quarantine_does_not_embed_raw_provider_payload(tmp_path):
    """PERSIST_E06"""
    prohibited = {
        "raw_record",
        "raw_payload",
        "payload",
        "xml",
        "csv_row",
        "provider_payload",
        "source_payload",
    }
    dataclass_names = {field.name.lower() for field in fields(EarningsQuarantineRecord)}
    schema_names = {name.lower() for name in quarantine_arrow_schema().names}
    assert prohibited.isdisjoint(dataclass_names)
    assert prohibited.isdisjoint(schema_names)
    path = tmp_path / "safe-quarantine.parquet"
    write_quarantine_records(path, [_diagnostic_record()])
    serialized_names = {name.lower() for name in pq.read_table(path).column_names}
    assert prohibited.isdisjoint(serialized_names)


def test_persist_e07_quarantine_parquet_schema_is_exact(tmp_path):
    """PERSIST_E07"""
    expected = pa.schema(
        [
            pa.field("quarantine_id", pa.string(), nullable=False),
            pa.field("build_id", pa.string(), nullable=False),
            pa.field("quarantined_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("provider_name", pa.string()),
            pa.field("provider_entity_id", pa.string()),
            pa.field("canonical_asset_id", pa.string()),
            pa.field("event_instance_id", pa.string()),
            pa.field("event_revision_id", pa.string()),
            pa.field("stage", pa.string(), nullable=False),
            pa.field("error_code", pa.string(), nullable=False),
            pa.field("error_message", pa.string(), nullable=False),
            pa.field("source_record_locator", pa.string()),
            pa.field("source_record_hash", pa.string()),
            pa.field("canonical_record_hash", pa.string()),
            pa.field("details_json", pa.string()),
        ]
    )
    assert quarantine_arrow_schema().equals(expected, check_metadata=True)
    path = tmp_path / "exact-quarantine.parquet"
    write_quarantine_records(path, [_diagnostic_record()])
    assert pq.read_table(path).schema.equals(expected, check_metadata=True)


def test_persist_e08_any_quarantine_record_blocks_v01_publication(
    tmp_path, make_persist_revision
):
    """PERSIST_E08"""
    paths = _paths(tmp_path)
    valid = make_persist_revision(event_revision_id="VALID")
    invalid = make_persist_revision(
        event_revision_id="INVALID", canonical_asset_id=None
    )
    result = publish_earnings_revisions(
        [valid, invalid], paths=paths, build_id="E08"
    )
    assert result.quarantined_count >= 1
    assert result.success is False
    assert result.inserted_count == 0
    assert result.output_path is None
    assert result.output_sha256 is None
    assert not paths.published_path.exists()
