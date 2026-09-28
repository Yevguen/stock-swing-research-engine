"""Strict Parquet serialization for canonical revisions and quarantine rows."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from stock_swing_d1.earnings.models import (
    EarningsScheduleRevision,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
    TransitionType,
)
from stock_swing_d1.earnings.persistence.hashing import (
    canonical_record_hash,
    canonical_sort_key,
    event_history_key,
    revision_identity_key,
)
from stock_swing_d1.earnings.persistence.models import (
    CanonicalParquetArtifact,
    EarningsPersistenceError,
    EarningsQuarantineRecord,
    QuarantineStage,
)
from stock_swing_d1.earnings.persistence.schema import (
    CANONICAL_EARNINGS_SCHEMA_VERSION,
    canonical_earnings_arrow_schema,
    quarantine_arrow_schema,
)
from stock_swing_d1.earnings.validation import (
    validate_revision,
    validate_revision_history,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "persisted timestamps must be timezone-aware"
        )
    return value.astimezone(timezone.utc)


def _validate_persistence_fields(revision: EarningsScheduleRevision) -> None:
    if revision.schema_version != CANONICAL_EARNINGS_SCHEMA_VERSION:
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH",
            f"unsupported revision schema_version {revision.schema_version!r}",
        )
    if revision.strategy_effective_at is None:
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "strategy_effective_at is required for persistence"
        )
    if revision.ingested_at is None:
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "ingested_at is required for persistence"
        )
    optional_text = (
        revision.provider_event_id,
        revision.provider_record_id,
        revision.historical_symbol,
        revision.historical_exchange,
        revision.event_timezone,
        revision.correction_of_revision_id,
    )
    if any(value is not None and not isinstance(value, str) for value in optional_text):
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "optional text fields must contain strings or None"
        )
    for value in (
        revision.knowledge_date,
        revision.scheduled_date,
        revision.actual_event_date,
    ):
        if value is not None and (
            not isinstance(value, date) or isinstance(value, datetime)
        ):
            raise EarningsPersistenceError(
                "SCHEMA_MISMATCH", "date fields must contain dates or None"
            )
    for value in (
        revision.knowledge_available_at,
        revision.strategy_effective_at,
        revision.scheduled_at,
        revision.actual_event_at,
        revision.ingested_at,
    ):
        if value is not None and not isinstance(value, datetime):
            raise EarningsPersistenceError(
                "SCHEMA_MISMATCH", "timestamp fields must contain datetimes or None"
            )
    if revision.provider_sequence is not None and (
        not isinstance(revision.provider_sequence, int)
        or isinstance(revision.provider_sequence, bool)
    ):
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "provider_sequence must contain an integer or None"
        )
    if not isinstance(revision.is_provider_correction, bool):
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "is_provider_correction must be boolean"
        )


def _revision_to_row(revision: EarningsScheduleRevision) -> dict[str, Any]:
    return {
        "schema_version": revision.schema_version,
        "event_revision_id": revision.event_revision_id,
        "event_instance_id": revision.event_instance_id,
        "canonical_asset_id": revision.canonical_asset_id,
        "provider_name": revision.provider_name,
        "provider_entity_id": revision.provider_entity_id,
        "provider_event_id": revision.provider_event_id,
        "provider_record_id": revision.provider_record_id,
        "historical_symbol": revision.historical_symbol,
        "historical_exchange": revision.historical_exchange,
        "knowledge_date": revision.knowledge_date,
        "knowledge_available_at": _utc(revision.knowledge_available_at),
        "knowledge_precision": revision.knowledge_precision.value,
        "strategy_effective_at": _utc(revision.strategy_effective_at),
        "provider_sequence": revision.provider_sequence,
        "transition_type": revision.transition_type.value,
        "lifecycle_state": revision.lifecycle_state.value,
        "scheduled_date": revision.scheduled_date,
        "timing_class": revision.timing_class.value,
        "scheduled_at": _utc(revision.scheduled_at),
        "event_timezone": revision.event_timezone,
        "actual_event_date": revision.actual_event_date,
        "actual_event_at": _utc(revision.actual_event_at),
        "is_provider_correction": revision.is_provider_correction,
        "correction_of_revision_id": revision.correction_of_revision_id,
        "ingested_at": _utc(revision.ingested_at),
        "canonical_record_hash": canonical_record_hash(revision),
    }


def _revision_from_row(row: dict[str, Any]) -> EarningsScheduleRevision:
    try:
        return EarningsScheduleRevision(
            schema_version=row["schema_version"],
            event_revision_id=row["event_revision_id"],
            event_instance_id=row["event_instance_id"],
            canonical_asset_id=row["canonical_asset_id"],
            provider_name=row["provider_name"],
            provider_entity_id=row["provider_entity_id"],
            provider_event_id=row["provider_event_id"],
            provider_record_id=row["provider_record_id"],
            historical_symbol=row["historical_symbol"],
            historical_exchange=row["historical_exchange"],
            knowledge_date=row["knowledge_date"],
            knowledge_available_at=_utc(row["knowledge_available_at"]),
            knowledge_precision=KnowledgePrecision(row["knowledge_precision"]),
            strategy_effective_at=_utc(row["strategy_effective_at"]),
            provider_sequence=row["provider_sequence"],
            transition_type=TransitionType(row["transition_type"]),
            lifecycle_state=LifecycleState(row["lifecycle_state"]),
            scheduled_date=row["scheduled_date"],
            timing_class=TimingClass(row["timing_class"]),
            scheduled_at=_utc(row["scheduled_at"]),
            event_timezone=row["event_timezone"],
            actual_event_date=row["actual_event_date"],
            actual_event_at=_utc(row["actual_event_at"]),
            is_provider_correction=row["is_provider_correction"],
            correction_of_revision_id=row["correction_of_revision_id"],
            ingested_at=_utc(row["ingested_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", f"cannot reconstruct canonical revision: {exc}"
        ) from exc


def write_canonical_revisions(
    path: Path,
    revisions: Iterable[EarningsScheduleRevision],
    *,
    overwrite: bool = False,
) -> CanonicalParquetArtifact:
    """Write, strictly verify, and identify one canonical Parquet artifact."""

    materialized = tuple(revisions)
    if path.exists() and not overwrite:
        raise EarningsPersistenceError(
            "PARQUET_WRITE_FAILED", f"refusing to overwrite existing path: {path}"
        )
    seen: set[tuple[str, str]] = set()
    for revision in materialized:
        validate_revision(revision)
        _validate_persistence_fields(revision)
        identity = revision_identity_key(revision)
        if identity in seen:
            raise EarningsPersistenceError(
                "DUPLICATE_REVISION_KEY", f"duplicate revision identity: {identity!r}"
            )
        seen.add(identity)
    ordered = tuple(sorted(materialized, key=canonical_sort_key))
    rows = [_revision_to_row(revision) for revision in ordered]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=canonical_earnings_arrow_schema())
        pq.write_table(table, path)
    except EarningsPersistenceError:
        raise
    except Exception as exc:
        raise EarningsPersistenceError(
            "PARQUET_WRITE_FAILED", f"failed to write {path}: {exc}"
        ) from exc
    try:
        verified = read_canonical_revisions(path)
        if len(verified) != len(ordered):
            raise EarningsPersistenceError(
                "PARQUET_WRITE_FAILED", "written artifact row count did not verify"
            )
        return CanonicalParquetArtifact(
            path=str(path), row_count=len(verified), sha256=_file_sha256(path)
        )
    except EarningsPersistenceError:
        raise
    except Exception as exc:
        raise EarningsPersistenceError(
            "PARQUET_WRITE_FAILED", f"failed to verify {path}: {exc}"
        ) from exc


def _read_exact_table(path: Path, expected_schema: pa.Schema) -> pa.Table:
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise EarningsPersistenceError(
            "PARQUET_READ_FAILED", f"failed to read {path}: {exc}"
        ) from exc
    if not table.schema.equals(expected_schema, check_metadata=True):
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH",
            f"unexpected Arrow schema for {path}: {table.schema}",
        )
    null_violations = [
        field.name
        for field in expected_schema
        if not field.nullable and table.column(field.name).null_count
    ]
    if null_violations:
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH",
            f"non-null columns contain null values: {null_violations!r}",
        )
    return table


def read_canonical_revisions(
    path: Path,
    *,
    verify_hashes: bool = True,
    verify_sort_order: bool = True,
    validate_records: bool = True,
    validate_histories: bool = True,
) -> tuple[EarningsScheduleRevision, ...]:
    """Read canonical revisions with all integrity checks enabled by default."""

    table = _read_exact_table(path, canonical_earnings_arrow_schema())
    rows = table.to_pylist()
    revisions: list[EarningsScheduleRevision] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        revision = _revision_from_row(row)
        _validate_persistence_fields(revision)
        identity = revision_identity_key(revision)
        if identity in seen:
            raise EarningsPersistenceError(
                "DUPLICATE_REVISION_KEY", f"duplicate revision identity: {identity!r}"
            )
        seen.add(identity)
        if verify_hashes and row["canonical_record_hash"] != canonical_record_hash(
            revision
        ):
            raise EarningsPersistenceError(
                "CANONICAL_HASH_MISMATCH",
                f"canonical record hash mismatch for {identity!r}",
            )
        if validate_records:
            validate_revision(revision)
        revisions.append(revision)
    if verify_sort_order and revisions != sorted(revisions, key=canonical_sort_key):
        raise EarningsPersistenceError(
            "UNSORTED_CANONICAL_DATASET",
            "canonical rows are not in deterministic sort order",
        )
    if validate_histories:
        histories: dict[tuple[str, str], list[EarningsScheduleRevision]] = defaultdict(
            list
        )
        for revision in revisions:
            histories[event_history_key(revision)].append(revision)
        for history in histories.values():
            validate_revision_history(history)
    return tuple(revisions)


def _validate_quarantine_record(record: EarningsQuarantineRecord) -> None:
    if not isinstance(record, EarningsQuarantineRecord):
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "expected EarningsQuarantineRecord"
        )
    for field_name in (
        "quarantine_id",
        "build_id",
        "error_code",
        "error_message",
    ):
        value = getattr(record, field_name)
        if not isinstance(value, str) or not value.strip():
            raise EarningsPersistenceError(
                "SCHEMA_MISMATCH", f"{field_name} is required"
            )
    if not isinstance(record.stage, QuarantineStage):
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "stage must be a canonical QuarantineStage"
        )
    _utc(record.quarantined_at)
    optional_text = (
        record.provider_name,
        record.provider_entity_id,
        record.canonical_asset_id,
        record.event_instance_id,
        record.event_revision_id,
        record.source_record_locator,
        record.source_record_hash,
        record.canonical_record_hash,
        record.details_json,
    )
    if any(value is not None and not isinstance(value, str) for value in optional_text):
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "quarantine diagnostics must be strings or None"
        )
    if record.details_json is not None:
        try:
            details = json.loads(record.details_json)
        except json.JSONDecodeError as exc:
            raise EarningsPersistenceError(
                "SCHEMA_MISMATCH", "details_json must contain valid diagnostic JSON"
            ) from exc
        prohibited_keys = {
            "raw_record",
            "raw_payload",
            "payload",
            "xml",
            "csv_row",
            "provider_payload",
            "source_payload",
        }

        def contains_prohibited_key(value: Any) -> bool:
            if isinstance(value, dict):
                return any(
                    str(key).lower() in prohibited_keys
                    or contains_prohibited_key(child)
                    for key, child in value.items()
                )
            if isinstance(value, list):
                return any(contains_prohibited_key(child) for child in value)
            return False

        if contains_prohibited_key(details):
            raise EarningsPersistenceError(
                "SCHEMA_MISMATCH",
                "details_json cannot embed a raw provider payload",
            )


def write_quarantine_records(
    path: Path,
    records: Iterable[EarningsQuarantineRecord],
    *,
    overwrite: bool = False,
) -> CanonicalParquetArtifact:
    """Write diagnostic quarantine records using their separate strict schema."""

    materialized = tuple(records)
    if path.exists() and not overwrite:
        raise EarningsPersistenceError(
            "PARQUET_WRITE_FAILED", f"refusing to overwrite existing path: {path}"
        )
    for record in materialized:
        _validate_quarantine_record(record)
    rows = [
        {
            "quarantine_id": record.quarantine_id,
            "build_id": record.build_id,
            "quarantined_at": _utc(record.quarantined_at),
            "provider_name": record.provider_name,
            "provider_entity_id": record.provider_entity_id,
            "canonical_asset_id": record.canonical_asset_id,
            "event_instance_id": record.event_instance_id,
            "event_revision_id": record.event_revision_id,
            "stage": record.stage.value,
            "error_code": record.error_code,
            "error_message": record.error_message,
            "source_record_locator": record.source_record_locator,
            "source_record_hash": record.source_record_hash,
            "canonical_record_hash": record.canonical_record_hash,
            "details_json": record.details_json,
        }
        for record in materialized
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=quarantine_arrow_schema()), path)
        verified = read_quarantine_records(path)
        return CanonicalParquetArtifact(
            path=str(path), row_count=len(verified), sha256=_file_sha256(path)
        )
    except EarningsPersistenceError:
        raise
    except Exception as exc:
        raise EarningsPersistenceError(
            "PARQUET_WRITE_FAILED", f"failed to write {path}: {exc}"
        ) from exc


def read_quarantine_records(
    path: Path,
) -> tuple[EarningsQuarantineRecord, ...]:
    """Read strict diagnostic quarantine rows."""

    table = _read_exact_table(path, quarantine_arrow_schema())
    records: list[EarningsQuarantineRecord] = []
    for row in table.to_pylist():
        try:
            record = EarningsQuarantineRecord(
                quarantine_id=row["quarantine_id"],
                build_id=row["build_id"],
                quarantined_at=_utc(row["quarantined_at"]),  # type: ignore[arg-type]
                provider_name=row["provider_name"],
                provider_entity_id=row["provider_entity_id"],
                canonical_asset_id=row["canonical_asset_id"],
                event_instance_id=row["event_instance_id"],
                event_revision_id=row["event_revision_id"],
                stage=QuarantineStage(row["stage"]),
                error_code=row["error_code"],
                error_message=row["error_message"],
                source_record_locator=row["source_record_locator"],
                source_record_hash=row["source_record_hash"],
                canonical_record_hash=row["canonical_record_hash"],
                details_json=row["details_json"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EarningsPersistenceError(
                "SCHEMA_MISMATCH", f"cannot reconstruct quarantine record: {exc}"
            ) from exc
        _validate_quarantine_record(record)
        records.append(record)
    return tuple(records)
