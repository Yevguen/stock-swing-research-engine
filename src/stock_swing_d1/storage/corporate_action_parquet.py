"""Immutable Phase 5B raw, clean, and published Parquet storage."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from stock_swing_d1.data.norgate_adjusted_d1_adapter import ADJUSTED_RAW_COLUMNS
from stock_swing_d1.data.norgate_capital_event_adapter import EVENT_RAW_COLUMNS
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    CorporateActionEvent,
)
from stock_swing_d1.storage.stock_bar_parquet import (
    BatchExistsError,
    ParquetValidationError,
    PublishValidationError,
    validate_batch_id,
)


ADJUSTED_RAW_ARROW_SCHEMA = pa.schema(
    [
        pa.field("provider_asset_id", pa.int64(), nullable=False),
        pa.field("provider_symbol", pa.string(), nullable=False),
        pa.field("trading_date", pa.date32(), nullable=False),
        pa.field("Open", pa.float64(), nullable=False),
        pa.field("High", pa.float64(), nullable=False),
        pa.field("Low", pa.float64(), nullable=False),
        pa.field("Close", pa.float64(), nullable=False),
        pa.field("Volume", pa.float64(), nullable=False),
        pa.field("Turnover", pa.float64(), nullable=True),
        pa.field("Unadjusted Close", pa.float64(), nullable=False),
        pa.field("Dividend", pa.float64(), nullable=True),
    ]
)
EVENT_RAW_ARROW_SCHEMA = pa.schema(
    [
        pa.field("provider_asset_id", pa.int64(), nullable=False),
        pa.field("provider_symbol", pa.string(), nullable=False),
        pa.field("event_date", pa.date32(), nullable=False),
        pa.field("capital_event_flag", pa.int8(), nullable=False),
    ]
)
ADJUSTED_CLEAN_ARROW_SCHEMA = pa.schema(
    [
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("trading_date", pa.date32(), nullable=False),
        pa.field("timeframe", pa.string(), nullable=False),
        pa.field("session_type", pa.string(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("price_basis", pa.string(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
    ]
)
EVENT_CLEAN_ARROW_SCHEMA = pa.schema(
    [
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("event_date", pa.date32(), nullable=False),
        pa.field("date_semantics", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("terms_verified", pa.bool_(), nullable=False),
        pa.field("new_shares", pa.float64(), nullable=True),
        pa.field("old_shares", pa.float64(), nullable=True),
        pa.field("source_provider", pa.string(), nullable=False),
        pa.field("source_asset_id", pa.int64(), nullable=False),
    ]
)

ADJUSTED_CLEAN_FIELDS = tuple(ADJUSTED_CLEAN_ARROW_SCHEMA.names)
EVENT_CLEAN_FIELDS = tuple(EVENT_CLEAN_ARROW_SCHEMA.names)
ADJUSTED_CANONICAL_KEY = (
    "security_id",
    "trading_date",
    "timeframe",
    "session_type",
    "price_basis",
)
EVENT_CANONICAL_KEY = ("security_id", "event_date", "source_provider")
RAW_MANIFEST_FIELDS = (
    "provider",
    "package",
    "phase5_batch_id",
    "phase4_source_batch_id",
    "phase4_source_path",
    "extracted_at_utc",
    "requested_asset_ids",
    "requested_date_ranges",
    "adjustment_mode",
    "padding_mode",
    "interval",
    "adjusted_raw_row_count",
    "capital_event_flag_row_count",
    "norgatedata_python_package_version",
    "event_source_function",
    "event_date_filtering",
)


class CorporateActionDatasetValidationError(ValueError):
    """Canonical adjusted bars or events violated a dataset invariant."""


@dataclass(frozen=True, slots=True)
class CleanCorporateActionPaths:
    adjusted_stock_bars: Path
    corporate_action_events: Path


@dataclass(frozen=True, slots=True)
class CorporateActionPublicationPaths:
    adjusted_stock_bars: tuple[Path, ...]
    corporate_action_events: tuple[Path, ...]


def raw_corporate_action_batch_directory(data_root: Path, batch_id: str) -> Path:
    return (
        Path(data_root)
        / "corporate_actions"
        / "raw"
        / "norgate"
        / f"batch={validate_batch_id(batch_id)}"
    )


def clean_corporate_action_batch_directory(data_root: Path, batch_id: str) -> Path:
    return (
        Path(data_root)
        / "corporate_actions"
        / "clean"
        / f"batch={validate_batch_id(batch_id)}"
    )


def adjusted_published_dataset_directory(data_root: Path) -> Path:
    return Path(data_root) / "parquet" / "corporate_action_adjusted_stock_bars_v0_1"


def event_published_dataset_directory(data_root: Path) -> Path:
    return Path(data_root) / "parquet" / "corporate_action_events_v0_1"


def ensure_corporate_action_batch_paths_available(
    data_root: Path, batch_id: str
) -> None:
    for path in (
        raw_corporate_action_batch_directory(data_root, batch_id),
        clean_corporate_action_batch_directory(data_root, batch_id),
    ):
        if path.exists():
            raise BatchExistsError(f"batch directory already exists: {path}")


def _json_text(value: Mapping[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _read_table(path: Path) -> pa.Table:
    return pq.ParquetFile(path).read()


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


def _verify_table(
    path: Path, schema: pa.Schema, *, expected_row_count: int
) -> pa.Table:
    table = _read_table(path)
    if not table.schema.equals(schema, check_metadata=False):
        raise ParquetValidationError(
            f"Parquet schema mismatch for {path.name}: {table.schema}"
        )
    if table.num_rows != expected_row_count:
        raise ParquetValidationError(
            f"Parquet row count {table.num_rows} != {expected_row_count}"
        )
    for field in schema:
        if not field.nullable and table[field.name].null_count:
            raise ParquetValidationError(
                f"non-null Parquet field {field.name} contains null values"
            )
    _verify_zstd(path)
    return table


def verify_adjusted_raw_parquet(path: Path, *, expected_row_count: int) -> pa.Table:
    return _verify_table(
        path, ADJUSTED_RAW_ARROW_SCHEMA, expected_row_count=expected_row_count
    )


def verify_event_raw_parquet(path: Path, *, expected_row_count: int) -> pa.Table:
    table = _verify_table(
        path, EVENT_RAW_ARROW_SCHEMA, expected_row_count=expected_row_count
    )
    invalid_flags = [
        flag
        for flag in table["capital_event_flag"].to_pylist()
        if flag not in (0, 1)
    ]
    if invalid_flags:
        raise ParquetValidationError(
            "raw event Parquet capital_event_flag must be exactly 0 or 1"
        )
    return table


def _records(
    rows: Sequence[Any] | pd.DataFrame,
    *,
    fields: Sequence[str],
    model_type: type[Any],
) -> list[dict[str, object]]:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
        records = frame.to_dict(orient="records")
        actual_fields = list(frame.columns)
    else:
        records = []
        for row in rows:
            if isinstance(row, model_type):
                records.append(row.model_dump(mode="python"))
            elif isinstance(row, Mapping):
                records.append(dict(row))
            else:
                raise TypeError(f"rows must contain {model_type.__name__} or mappings")
        actual_fields = list(records[0]) if records else list(fields)
        expected_field_set = set(actual_fields)
        if any(set(record) != expected_field_set for record in records[1:]):
            raise CorporateActionDatasetValidationError(
                "canonical rows do not have a consistent field set"
            )
    missing = [field for field in fields if field not in actual_fields]
    extra = [field for field in actual_fields if field not in fields]
    if missing or extra:
        messages = []
        if missing:
            messages.append("missing canonical fields: " + ", ".join(missing))
        if extra:
            messages.append("unexpected canonical fields: " + ", ".join(extra))
        raise CorporateActionDatasetValidationError("; ".join(messages))
    return [{field: record[field] for field in fields} for record in records]


def validate_adjusted_bar_dataset(
    rows: Sequence[CorporateActionAdjustedStockBar | Mapping[str, object]]
    | pd.DataFrame,
) -> list[CorporateActionAdjustedStockBar]:
    """Validate exact fields, both uniqueness keys, models, and sorting."""
    records = _records(
        rows,
        fields=ADJUSTED_CLEAN_FIELDS,
        model_type=CorporateActionAdjustedStockBar,
    )
    bars: list[CorporateActionAdjustedStockBar] = []
    for record in records:
        try:
            bars.append(CorporateActionAdjustedStockBar(**record))
        except ValidationError as error:
            raise CorporateActionDatasetValidationError(
                f"row is not a valid CorporateActionAdjustedStockBar: {error}"
            ) from error
    bars.sort(key=lambda bar: (bar.security_id, bar.trading_date))
    canonical_keys: set[tuple[object, ...]] = set()
    security_date_keys: set[tuple[object, ...]] = set()
    for bar in bars:
        values = bar.model_dump(mode="python")
        canonical_key = tuple(values[field] for field in ADJUSTED_CANONICAL_KEY)
        security_date_key = (bar.security_id, bar.trading_date)
        if canonical_key in canonical_keys:
            raise CorporateActionDatasetValidationError(
                f"duplicate adjusted canonical key: {canonical_key}"
            )
        if security_date_key in security_date_keys:
            raise CorporateActionDatasetValidationError(
                f"duplicate adjusted security/date key: {security_date_key}"
            )
        canonical_keys.add(canonical_key)
        security_date_keys.add(security_date_key)
    return bars


def validate_corporate_action_event_dataset(
    rows: Sequence[CorporateActionEvent | Mapping[str, object]] | pd.DataFrame,
) -> list[CorporateActionEvent]:
    """Validate exact event fields, uniqueness, models, and sorting."""
    records = _records(
        rows, fields=EVENT_CLEAN_FIELDS, model_type=CorporateActionEvent
    )
    events: list[CorporateActionEvent] = []
    for record in records:
        try:
            events.append(CorporateActionEvent(**record))
        except ValidationError as error:
            raise CorporateActionDatasetValidationError(
                f"row is not a valid CorporateActionEvent: {error}"
            ) from error
    events.sort(key=lambda event: (event.security_id, event.event_date))
    keys: set[tuple[object, ...]] = set()
    for event in events:
        key = (event.security_id, event.event_date, event.source_provider)
        if key in keys:
            raise CorporateActionDatasetValidationError(
                f"duplicate corporate-action event key: {key}"
            )
        keys.add(key)
    return events


def _model_table(rows: Sequence[Any], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(
        [row.model_dump(mode="python") for row in rows], schema=schema
    )


def verify_adjusted_clean_parquet(
    path: Path, *, expected_row_count: int | None = None
) -> list[CorporateActionAdjustedStockBar]:
    table = _read_table(path)
    if not table.schema.equals(ADJUSTED_CLEAN_ARROW_SCHEMA, check_metadata=False):
        raise ParquetValidationError(
            f"adjusted clean Parquet schema mismatch: {table.schema}"
        )
    if expected_row_count is not None and table.num_rows != expected_row_count:
        raise ParquetValidationError(
            f"adjusted clean row count {table.num_rows} != {expected_row_count}"
        )
    if any(column.null_count for column in table.columns):
        raise ParquetValidationError("adjusted clean Parquet contains null values")
    _verify_zstd(path)
    try:
        return validate_adjusted_bar_dataset(table.to_pylist())
    except (TypeError, ValueError) as error:
        raise ParquetValidationError(
            f"adjusted clean read-back validation failed: {error}"
        ) from error


def verify_event_clean_parquet(
    path: Path, *, expected_row_count: int | None = None
) -> list[CorporateActionEvent]:
    table = _read_table(path)
    if not table.schema.equals(EVENT_CLEAN_ARROW_SCHEMA, check_metadata=False):
        raise ParquetValidationError(
            f"event clean Parquet schema mismatch: {table.schema}"
        )
    if expected_row_count is not None and table.num_rows != expected_row_count:
        raise ParquetValidationError(
            f"event clean row count {table.num_rows} != {expected_row_count}"
        )
    for field in EVENT_CLEAN_ARROW_SCHEMA:
        if not field.nullable and table[field.name].null_count:
            raise ParquetValidationError(
                f"event clean field {field.name} contains null values"
            )
    _verify_zstd(path)
    try:
        return validate_corporate_action_event_dataset(table.to_pylist())
    except (TypeError, ValueError) as error:
        raise ParquetValidationError(
            f"event clean read-back validation failed: {error}"
        ) from error


def _frame_table(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    return (
        schema.empty_table()
        if frame.empty
        else pa.Table.from_pandas(
            frame, schema=schema, preserve_index=False, safe=True
        )
    )


def write_corporate_action_raw_batch(
    adjusted_raw_frame: pd.DataFrame,
    event_raw_frame: pd.DataFrame,
    manifest: Mapping[str, object],
    *,
    data_root: Path,
    batch_id: str,
) -> Path:
    """Atomically write both immutable raw Phase 5B snapshots and manifest."""
    normalized_batch_id = validate_batch_id(batch_id)
    batch_directory = raw_corporate_action_batch_directory(
        data_root, normalized_batch_id
    )
    if batch_directory.exists():
        raise BatchExistsError(f"batch directory already exists: {batch_directory}")
    if list(adjusted_raw_frame.columns) != list(ADJUSTED_RAW_COLUMNS):
        raise ParquetValidationError(
            "adjusted raw frame must contain exactly the frozen columns in order"
        )
    if list(event_raw_frame.columns) != list(EVENT_RAW_COLUMNS):
        raise ParquetValidationError(
            "event raw frame must contain exactly the frozen columns in order"
        )
    missing_manifest = [field for field in RAW_MANIFEST_FIELDS if field not in manifest]
    if missing_manifest:
        raise ParquetValidationError(
            "raw manifest is missing fields: " + ", ".join(missing_manifest)
        )
    if manifest["phase5_batch_id"] != normalized_batch_id:
        raise ParquetValidationError("manifest phase5_batch_id does not match path")
    asset_ids = manifest["requested_asset_ids"]
    if not isinstance(asset_ids, list) or asset_ids != sorted(set(asset_ids)):
        raise ParquetValidationError(
            "requested_asset_ids must be a sorted, deduplicated list"
        )
    if manifest["adjusted_raw_row_count"] != len(adjusted_raw_frame.index):
        raise ParquetValidationError("manifest adjusted row count does not match")
    if manifest["capital_event_flag_row_count"] != len(event_raw_frame.index):
        raise ParquetValidationError("manifest event row count does not match")

    parent = batch_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".batch={normalized_batch_id}.tmp-", dir=parent)
    )
    try:
        adjusted_path = temporary / "adjusted_bars.parquet"
        event_path = temporary / "capital_event_flags.parquet"
        pq.write_table(
            _frame_table(adjusted_raw_frame, ADJUSTED_RAW_ARROW_SCHEMA),
            adjusted_path,
            compression="zstd",
        )
        pq.write_table(
            _frame_table(event_raw_frame, EVENT_RAW_ARROW_SCHEMA),
            event_path,
            compression="zstd",
        )
        verify_adjusted_raw_parquet(
            adjusted_path, expected_row_count=len(adjusted_raw_frame.index)
        )
        verify_event_raw_parquet(
            event_path, expected_row_count=len(event_raw_frame.index)
        )
        (temporary / "manifest.json").write_text(
            _json_text(manifest), encoding="utf-8", newline=""
        )
        temporary.rename(batch_directory)
    except FileExistsError as error:
        raise BatchExistsError(
            f"batch directory already exists: {batch_directory}"
        ) from error
    except (BatchExistsError, ParquetValidationError):
        raise
    except Exception as error:
        raise ParquetValidationError(f"raw Phase 5B write failed: {error}") from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return batch_directory


def write_corporate_action_clean_batch(
    adjusted_rows: Sequence[
        CorporateActionAdjustedStockBar | Mapping[str, object]
    ]
    | pd.DataFrame,
    event_rows: Sequence[CorporateActionEvent | Mapping[str, object]] | pd.DataFrame,
    *,
    data_root: Path,
    batch_id: str,
) -> CleanCorporateActionPaths:
    """Atomically write and read-back validate both clean datasets."""
    normalized_batch_id = validate_batch_id(batch_id)
    bars = validate_adjusted_bar_dataset(adjusted_rows)
    events = validate_corporate_action_event_dataset(event_rows)
    if not bars:
        raise CorporateActionDatasetValidationError(
            "adjusted clean dataset must not be empty"
        )
    batch_directory = clean_corporate_action_batch_directory(
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
        adjusted_path = temporary / "adjusted_stock_bars.parquet"
        event_path = temporary / "corporate_action_events.parquet"
        pq.write_table(
            _model_table(bars, ADJUSTED_CLEAN_ARROW_SCHEMA),
            adjusted_path,
            compression="zstd",
        )
        pq.write_table(
            _model_table(events, EVENT_CLEAN_ARROW_SCHEMA),
            event_path,
            compression="zstd",
        )
        verify_adjusted_clean_parquet(
            adjusted_path, expected_row_count=len(bars)
        )
        verify_event_clean_parquet(event_path, expected_row_count=len(events))
        temporary.rename(batch_directory)
    except FileExistsError as error:
        raise BatchExistsError(
            f"batch directory already exists: {batch_directory}"
        ) from error
    except (CorporateActionDatasetValidationError, ParquetValidationError):
        raise
    except Exception as error:
        raise ParquetValidationError(f"clean Phase 5B write failed: {error}") from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return CleanCorporateActionPaths(
        adjusted_stock_bars=batch_directory / "adjusted_stock_bars.parquet",
        corporate_action_events=batch_directory / "corporate_action_events.parquet",
    )


def write_corporate_action_validation_report(
    report: Mapping[str, object],
    *,
    data_root: Path,
    batch_id: str,
) -> Path:
    directory = clean_corporate_action_batch_directory(data_root, batch_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "validation_report.json"
    try:
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(_json_text(report))
    except FileExistsError as error:
        raise BatchExistsError(f"validation report already exists: {path}") from error
    return path


def _adjusted_key(bar: CorporateActionAdjustedStockBar) -> tuple[object, ...]:
    values = bar.model_dump(mode="python")
    return tuple(values[field] for field in ADJUSTED_CANONICAL_KEY)


def _event_key(event: CorporateActionEvent) -> tuple[object, ...]:
    return event.security_id, event.event_date, event.source_provider


def _existing_adjusted_keys(year_directory: Path) -> set[tuple[object, ...]]:
    keys: set[tuple[object, ...]] = set()
    for path in sorted(year_directory.glob("*.parquet")):
        try:
            rows = verify_adjusted_clean_parquet(path)
        except ParquetValidationError as error:
            raise PublishValidationError(
                f"existing adjusted published part is invalid: {path}: {error}"
            ) from error
        for row in rows:
            key = _adjusted_key(row)
            if key in keys:
                raise PublishValidationError(
                    f"existing adjusted publication contains duplicate key: {key}",
                    duplicate_count=1,
                )
            keys.add(key)
    return keys


def _existing_event_keys(year_directory: Path) -> set[tuple[object, ...]]:
    keys: set[tuple[object, ...]] = set()
    for path in sorted(year_directory.glob("*.parquet")):
        try:
            rows = verify_event_clean_parquet(path)
        except ParquetValidationError as error:
            raise PublishValidationError(
                f"existing event published part is invalid: {path}: {error}"
            ) from error
        for row in rows:
            key = _event_key(row)
            if key in keys:
                raise PublishValidationError(
                    f"existing event publication contains duplicate key: {key}",
                    duplicate_count=1,
                )
            keys.add(key)
    return keys


def _publish(
    *,
    data_root: Path,
    batch_id: str,
    adjusted_rows: Sequence[CorporateActionAdjustedStockBar] | None,
    event_rows: Sequence[CorporateActionEvent] | None,
) -> CorporateActionPublicationPaths:
    normalized_batch_id = validate_batch_id(batch_id)
    adjusted = (
        validate_adjusted_bar_dataset(adjusted_rows)
        if adjusted_rows is not None
        else []
    )
    events = (
        validate_corporate_action_event_dataset(event_rows)
        if event_rows is not None
        else []
    )
    if adjusted_rows is not None and not adjusted:
        raise PublishValidationError("cannot publish an empty adjusted-bar batch")

    adjusted_by_year: dict[int, list[CorporateActionAdjustedStockBar]] = {}
    for bar in adjusted:
        adjusted_by_year.setdefault(bar.trading_date.year, []).append(bar)
    events_by_year: dict[int, list[CorporateActionEvent]] = {}
    for event in events:
        events_by_year.setdefault(event.event_date.year, []).append(event)

    adjusted_root = adjusted_published_dataset_directory(data_root)
    event_root = event_published_dataset_directory(data_root)
    adjusted_targets = {
        year: adjusted_root
        / f"year={year}"
        / f"part-{normalized_batch_id}.parquet"
        for year in sorted(adjusted_by_year)
    }
    event_targets = {
        year: event_root
        / f"year={year}"
        / f"part-{normalized_batch_id}.parquet"
        for year in sorted(events_by_year)
    }

    new_adjusted_keys = {_adjusted_key(bar) for bar in adjusted}
    for target in adjusted_targets.values():
        if target.exists():
            raise PublishValidationError(f"published batch part already exists: {target}")
        overlaps = _existing_adjusted_keys(target.parent) & new_adjusted_keys
        if overlaps:
            sample = sorted(overlaps, key=repr)[0]
            raise PublishValidationError(
                f"new adjusted batch overlaps an existing canonical key: {sample}",
                duplicate_count=len(overlaps),
            )
    new_event_keys = {_event_key(event) for event in events}
    for target in event_targets.values():
        if target.exists():
            raise PublishValidationError(f"published batch part already exists: {target}")
        overlaps = _existing_event_keys(target.parent) & new_event_keys
        if overlaps:
            sample = sorted(overlaps, key=repr)[0]
            raise PublishValidationError(
                f"new event batch overlaps an existing canonical key: {sample}",
                duplicate_count=len(overlaps),
            )

    parquet_root = Path(data_root) / "parquet"
    parquet_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".corporate-actions-{normalized_batch_id}.tmp-",
            dir=parquet_root,
        )
    )
    staged: list[tuple[Path, Path, int, str]] = []
    moved: list[Path] = []
    created_directories: list[Path] = []
    try:
        for year, rows in adjusted_by_year.items():
            staged_path = staging_root / "adjusted" / f"year={year}" / adjusted_targets[year].name
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                _model_table(rows, ADJUSTED_CLEAN_ARROW_SCHEMA),
                staged_path,
                compression="zstd",
            )
            verify_adjusted_clean_parquet(staged_path, expected_row_count=len(rows))
            staged.append((staged_path, adjusted_targets[year], len(rows), "adjusted"))
        for year, rows in events_by_year.items():
            staged_path = staging_root / "events" / f"year={year}" / event_targets[year].name
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                _model_table(rows, EVENT_CLEAN_ARROW_SCHEMA),
                staged_path,
                compression="zstd",
            )
            verify_event_clean_parquet(staged_path, expected_row_count=len(rows))
            staged.append((staged_path, event_targets[year], len(rows), "event"))

        for staged_path, target, _, _ in staged:
            if not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=False)
                created_directories.append(target.parent)
            os.link(staged_path, target)
            moved.append(target)
            staged_path.unlink()

        for _, target, row_count, kind in staged:
            if kind == "adjusted":
                verify_adjusted_clean_parquet(target, expected_row_count=row_count)
            else:
                verify_event_clean_parquet(target, expected_row_count=row_count)
    except Exception as error:
        for path in moved:
            path.unlink(missing_ok=True)
        for directory in reversed(created_directories):
            try:
                directory.rmdir()
            except OSError:
                pass
        if isinstance(error, PublishValidationError):
            raise
        raise PublishValidationError(f"corporate-action publication failed: {error}") from error
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    return CorporateActionPublicationPaths(
        adjusted_stock_bars=tuple(
            adjusted_targets[year] for year in sorted(adjusted_targets)
        ),
        corporate_action_events=tuple(
            event_targets[year] for year in sorted(event_targets)
        ),
    )


def publish_adjusted_stock_bars(
    rows: Sequence[CorporateActionAdjustedStockBar],
    *,
    data_root: Path,
    batch_id: str,
) -> tuple[Path, ...]:
    return _publish(
        data_root=data_root,
        batch_id=batch_id,
        adjusted_rows=rows,
        event_rows=None,
    ).adjusted_stock_bars


def publish_corporate_action_events(
    rows: Sequence[CorporateActionEvent],
    *,
    data_root: Path,
    batch_id: str,
) -> tuple[Path, ...]:
    return _publish(
        data_root=data_root,
        batch_id=batch_id,
        adjusted_rows=None,
        event_rows=rows,
    ).corporate_action_events


def publish_corporate_action_datasets(
    adjusted_rows: Sequence[CorporateActionAdjustedStockBar],
    event_rows: Sequence[CorporateActionEvent],
    *,
    data_root: Path,
    batch_id: str,
) -> CorporateActionPublicationPaths:
    """Publish both datasets with one preflight, staging, and rollback boundary."""
    return _publish(
        data_root=data_root,
        batch_id=batch_id,
        adjusted_rows=adjusted_rows,
        event_rows=event_rows,
    )


__all__ = [
    "ADJUSTED_CLEAN_ARROW_SCHEMA",
    "ADJUSTED_RAW_ARROW_SCHEMA",
    "EVENT_CLEAN_ARROW_SCHEMA",
    "EVENT_RAW_ARROW_SCHEMA",
    "CleanCorporateActionPaths",
    "CorporateActionDatasetValidationError",
    "CorporateActionPublicationPaths",
    "adjusted_published_dataset_directory",
    "clean_corporate_action_batch_directory",
    "ensure_corporate_action_batch_paths_available",
    "event_published_dataset_directory",
    "publish_adjusted_stock_bars",
    "publish_corporate_action_datasets",
    "publish_corporate_action_events",
    "raw_corporate_action_batch_directory",
    "validate_adjusted_bar_dataset",
    "validate_corporate_action_event_dataset",
    "verify_adjusted_clean_parquet",
    "verify_adjusted_raw_parquet",
    "verify_event_clean_parquet",
    "verify_event_raw_parquet",
    "write_corporate_action_clean_batch",
    "write_corporate_action_raw_batch",
    "write_corporate_action_validation_report",
]
