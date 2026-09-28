"""Immutable clean storage and publication for DividendEvent v0.1."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from stock_swing_d1.models import DividendEvent
from stock_swing_d1.storage.stock_bar_parquet import (
    BatchExistsError,
    ParquetValidationError,
    PublishValidationError,
    validate_batch_id,
)
from stock_swing_d1.validation.dividend_event_dataset import (
    DIVIDEND_CANONICAL_KEY,
    DividendEventDatasetValidationError,
    validate_dividend_event_dataset,
)


DIVIDEND_EVENT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("entitlement_date", pa.date32(), nullable=False),
        pa.field("date_semantics", pa.string(), nullable=False),
        pa.field("dividend_type", pa.string(), nullable=False),
        pa.field("amount_per_share", pa.float64(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("source_provider", pa.string(), nullable=False),
        pa.field("source_asset_id", pa.int64(), nullable=False),
        pa.field("source_adjustment_mode", pa.string(), nullable=False),
    ]
)
DERIVATION_MANIFEST_FIELDS = (
    "dividend_batch_id",
    "phase5_source_batch_id",
    "phase5_source_raw_path",
    "phase5_source_adjusted_clean_path",
    "phase4_source_batch_id",
    "phase4_source_path",
    "derived_at_utc",
    "provider",
    "source_adjustment_mode",
    "source_raw_row_count",
    "source_dividend_nonzero_count",
)


@dataclass(frozen=True, slots=True)
class CleanDividendPaths:
    """Files in one immutable clean dividend derivation batch."""

    dividend_events: Path
    manifest: Path


def clean_dividend_batch_directory(data_root: Path, batch_id: str) -> Path:
    return (
        Path(data_root)
        / "dividends"
        / "clean"
        / f"batch={validate_batch_id(batch_id)}"
    )


def dividend_published_dataset_directory(data_root: Path) -> Path:
    return Path(data_root) / "parquet" / "dividend_events_v0_1"


def ensure_dividend_batch_path_available(data_root: Path, batch_id: str) -> None:
    path = clean_dividend_batch_directory(data_root, batch_id)
    if path.exists():
        raise BatchExistsError(f"batch directory already exists: {path}")


def _json_text(value: Mapping[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _event_key(event: DividendEvent) -> tuple[object, ...]:
    values = event.model_dump(mode="python")
    return tuple(values[field] for field in DIVIDEND_CANONICAL_KEY)


def _event_table(events: Sequence[DividendEvent]) -> pa.Table:
    if not events:
        return DIVIDEND_EVENT_ARROW_SCHEMA.empty_table()
    return pa.Table.from_pylist(
        [event.model_dump(mode="python") for event in events],
        schema=DIVIDEND_EVENT_ARROW_SCHEMA,
    )


def _verify_zstd(path: Path) -> None:
    metadata = pq.ParquetFile(path).metadata
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            compression = row_group.column(column_index).compression
            if compression.upper() != "ZSTD":
                raise ParquetValidationError(
                    f"Parquet column compression is {compression}, expected ZSTD"
                )


def verify_dividend_event_parquet(
    path: Path, *, expected_row_count: int | None = None
) -> list[DividendEvent]:
    """Read back the exact schema, count, order, keys, and every model row."""
    try:
        table = pq.ParquetFile(path).read()
    except Exception as error:
        raise ParquetValidationError(
            f"could not read dividend event Parquet {path}: {error}"
        ) from error
    if not table.schema.equals(DIVIDEND_EVENT_ARROW_SCHEMA, check_metadata=False):
        raise ParquetValidationError(
            f"dividend event Parquet schema mismatch: {table.schema}"
        )
    if expected_row_count is not None and table.num_rows != expected_row_count:
        raise ParquetValidationError(
            f"dividend event row count {table.num_rows} != {expected_row_count}"
        )
    if any(column.null_count for column in table.columns):
        raise ParquetValidationError("dividend event Parquet contains null values")
    _verify_zstd(path)
    records = table.to_pylist()
    persisted_order = [
        (record["security_id"], record["entitlement_date"]) for record in records
    ]
    if persisted_order != sorted(persisted_order):
        raise ParquetValidationError(
            "dividend event Parquet is not deterministically sorted"
        )
    try:
        return validate_dividend_event_dataset(records)
    except (TypeError, ValueError) as error:
        raise ParquetValidationError(
            f"dividend event read-back validation failed: {error}"
        ) from error


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    batch_id: str,
    source_raw_row_count: int,
    event_count: int,
) -> None:
    missing = [field for field in DERIVATION_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ParquetValidationError(
            "dividend manifest is missing fields: " + ", ".join(missing)
        )
    if manifest["dividend_batch_id"] != batch_id:
        raise ParquetValidationError("manifest dividend_batch_id does not match path")
    if manifest["provider"] != "Norgate Data":
        raise ParquetValidationError("manifest provider must be Norgate Data")
    if manifest["source_adjustment_mode"] != "CAPITALSPECIAL":
        raise ParquetValidationError(
            "manifest source_adjustment_mode must be CAPITALSPECIAL"
        )
    if manifest["source_raw_row_count"] != source_raw_row_count:
        raise ParquetValidationError("manifest source raw row count does not match")
    if manifest["source_dividend_nonzero_count"] != event_count:
        raise ParquetValidationError(
            "manifest source dividend count does not match canonical events"
        )


def write_dividend_clean_batch(
    rows: Sequence[DividendEvent | Mapping[str, object]],
    manifest: Mapping[str, object],
    *,
    data_root: Path,
    batch_id: str,
    source_raw_row_count: int,
) -> CleanDividendPaths:
    """Atomically write manifest and clean events, including a zero-row file."""
    normalized_batch_id = validate_batch_id(batch_id)
    events = validate_dividend_event_dataset(rows)
    _validate_manifest(
        manifest,
        batch_id=normalized_batch_id,
        source_raw_row_count=source_raw_row_count,
        event_count=len(events),
    )
    batch_directory = clean_dividend_batch_directory(
        data_root, normalized_batch_id
    )
    if batch_directory.exists():
        raise BatchExistsError(f"batch directory already exists: {batch_directory}")
    parent = batch_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".batch={normalized_batch_id}.tmp-", dir=parent)
    )
    try:
        parquet_path = temporary / "dividend_events.parquet"
        manifest_path = temporary / "manifest.json"
        pq.write_table(_event_table(events), parquet_path, compression="zstd")
        verify_dividend_event_parquet(
            parquet_path, expected_row_count=len(events)
        )
        manifest_path.write_text(
            _json_text(manifest), encoding="utf-8", newline=""
        )
        temporary.rename(batch_directory)
    except FileExistsError as error:
        raise BatchExistsError(
            f"batch directory already exists: {batch_directory}"
        ) from error
    except (DividendEventDatasetValidationError, ParquetValidationError):
        raise
    except Exception as error:
        raise ParquetValidationError(
            f"clean dividend batch write failed: {error}"
        ) from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return CleanDividendPaths(
        dividend_events=batch_directory / "dividend_events.parquet",
        manifest=batch_directory / "manifest.json",
    )


def write_dividend_validation_report(
    report: Mapping[str, object],
    *,
    data_root: Path,
    batch_id: str,
) -> Path:
    directory = clean_dividend_batch_directory(data_root, batch_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "validation_report.json"
    try:
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(_json_text(report))
    except FileExistsError as error:
        raise BatchExistsError(f"validation report already exists: {path}") from error
    return path


def _existing_keys_for_year(year_directory: Path) -> set[tuple[object, ...]]:
    keys: set[tuple[object, ...]] = set()
    for path in sorted(year_directory.glob("*.parquet")):
        try:
            events = verify_dividend_event_parquet(path)
        except ParquetValidationError as error:
            raise PublishValidationError(
                f"existing dividend published part is invalid: {path}: {error}"
            ) from error
        for event in events:
            key = _event_key(event)
            if key in keys:
                raise PublishValidationError(
                    f"existing dividend publication contains duplicate key: {key}",
                    duplicate_count=1,
                )
            keys.add(key)
    return keys


def publish_dividend_events(
    rows: Sequence[DividendEvent | Mapping[str, object]],
    *,
    data_root: Path,
    batch_id: str,
) -> tuple[Path, ...]:
    """Publish year-only no-clobber parts with coordinated rollback.

    A valid zero-event batch returns an empty tuple. That return represents a
    successfully validated publication step; no empty year partition is made.
    """
    normalized_batch_id = validate_batch_id(batch_id)
    events = validate_dividend_event_dataset(rows)
    if not events:
        return ()

    by_year: dict[int, list[DividendEvent]] = {}
    for event in events:
        by_year.setdefault(event.entitlement_date.year, []).append(event)
    published_root = dividend_published_dataset_directory(data_root)
    targets = {
        year: published_root
        / f"year={year}"
        / f"part-{normalized_batch_id}.parquet"
        for year in sorted(by_year)
    }
    new_keys = {_event_key(event) for event in events}
    for target in targets.values():
        if target.exists():
            raise PublishValidationError(
                f"published dividend batch part already exists: {target}"
            )
        overlaps = _existing_keys_for_year(target.parent) & new_keys
        if overlaps:
            sample = sorted(overlaps, key=repr)[0]
            raise PublishValidationError(
                f"new dividend batch overlaps an existing canonical key: {sample}",
                duplicate_count=len(overlaps),
            )

    staging_parent = published_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".dividend_events_v0_1-{normalized_batch_id}.tmp-",
            dir=staging_parent,
        )
    )
    staged: dict[int, Path] = {}
    moved: list[Path] = []
    created_year_directories: list[Path] = []
    try:
        for year, year_events in by_year.items():
            staged_path = staging_root / f"year={year}" / targets[year].name
            staged_path.parent.mkdir(parents=True, exist_ok=False)
            pq.write_table(
                _event_table(year_events), staged_path, compression="zstd"
            )
            verify_dividend_event_parquet(
                staged_path, expected_row_count=len(year_events)
            )
            staged[year] = staged_path

        for year, target in targets.items():
            if not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=False)
                created_year_directories.append(target.parent)
            os.link(staged[year], target)
            moved.append(target)
            staged[year].unlink()

        for year, target in targets.items():
            verify_dividend_event_parquet(
                target, expected_row_count=len(by_year[year])
            )
    except Exception as error:
        for path in moved:
            path.unlink(missing_ok=True)
        for directory in reversed(created_year_directories):
            try:
                directory.rmdir()
            except OSError:
                pass
        if isinstance(error, PublishValidationError):
            raise
        raise PublishValidationError(
            f"dividend publication failed: {error}"
        ) from error
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    return tuple(targets[year] for year in sorted(targets))


__all__ = [
    "DERIVATION_MANIFEST_FIELDS",
    "DIVIDEND_EVENT_ARROW_SCHEMA",
    "CleanDividendPaths",
    "clean_dividend_batch_directory",
    "dividend_published_dataset_directory",
    "ensure_dividend_batch_path_available",
    "publish_dividend_events",
    "verify_dividend_event_parquet",
    "write_dividend_clean_batch",
    "write_dividend_validation_report",
]
