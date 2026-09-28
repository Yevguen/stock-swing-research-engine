"""Immutable Parquet storage for Norgate-derived D1 market data.

Files below ``data/raw/norgate``, ``data/clean/d1``, and
``data/parquet/stock_bars_v0_1`` contain or derive from Norgate Data. They are
subject to the project's accepted data-deletion obligation if the relevant
subscription lapses.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stock_swing_d1.data.norgate_d1_adapter import RAW_COLUMNS
from stock_swing_d1.models import StockBar
from stock_swing_d1.validation.stock_bar_dataset import (
    CANONICAL_KEY,
    validate_stock_bar_dataset,
)


RAW_ARROW_SCHEMA = pa.schema(
    [
        pa.field("provider_asset_id", pa.int64(), nullable=False),
        pa.field("provider_symbol", pa.string(), nullable=False),
        pa.field("trading_date", pa.date32(), nullable=False),
        pa.field("Open", pa.float64()),
        pa.field("High", pa.float64()),
        pa.field("Low", pa.float64()),
        pa.field("Close", pa.float64()),
        pa.field("Volume", pa.float64()),
        pa.field("Turnover", pa.float64()),
        pa.field("Unadjusted Close", pa.float64()),
        pa.field("Dividend", pa.float64()),
    ]
)

CANONICAL_ARROW_SCHEMA = pa.schema(
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
        pa.field("volume", pa.int64(), nullable=False),
    ]
)

RAW_MANIFEST_FIELDS = (
    "provider",
    "package",
    "batch_id",
    "extracted_at_utc",
    "requested_asset_ids",
    "requested_start_date",
    "requested_end_date",
    "adjustment_mode",
    "padding_mode",
    "norgatedata_python_package_version",
    "raw_row_count",
)


class BatchExistsError(FileExistsError):
    """An immutable raw or clean batch path already exists."""


class ParquetValidationError(ValueError):
    """A Parquet write or read-back did not satisfy the frozen schema."""


class PublishValidationError(ValueError):
    """A batch cannot be safely published."""

    def __init__(self, reason: str, *, duplicate_count: int = 0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.duplicate_count = duplicate_count


def validate_batch_id(batch_id: object) -> str:
    """Reject path separators, traversal, and unsafe batch path components."""
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id must be a non-empty string")
    if not all(character.isalnum() or character in "-_" for character in batch_id):
        raise ValueError(
            "batch_id may contain only letters, numbers, hyphen, and underscore"
        )
    return batch_id


def raw_batch_directory(data_root: Path, batch_id: str) -> Path:
    return Path(data_root) / "raw" / "norgate" / "d1" / f"batch={validate_batch_id(batch_id)}"


def clean_batch_directory(data_root: Path, batch_id: str) -> Path:
    return Path(data_root) / "clean" / "d1" / f"batch={validate_batch_id(batch_id)}"


def published_dataset_directory(data_root: Path) -> Path:
    return Path(data_root) / "parquet" / "stock_bars_v0_1"


def ensure_batch_paths_available(data_root: Path, batch_id: str) -> None:
    """Fail before extraction if either immutable batch directory exists."""
    for path in (
        raw_batch_directory(data_root, batch_id),
        clean_batch_directory(data_root, batch_id),
    ):
        if path.exists():
            raise BatchExistsError(f"batch directory already exists: {path}")


def _json_text(value: Mapping[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _parquet_file_table(path: Path) -> pa.Table:
    # ParquetFile avoids interpreting ``year=...`` as an extra Hive column.
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


def verify_raw_parquet(path: Path, *, expected_row_count: int) -> pa.Table:
    """Read back and verify a raw snapshot."""
    table = _parquet_file_table(path)
    if not table.schema.equals(RAW_ARROW_SCHEMA, check_metadata=False):
        raise ParquetValidationError(
            f"raw Parquet schema mismatch: {table.schema}"
        )
    if table.num_rows != expected_row_count:
        raise ParquetValidationError(
            f"raw Parquet row count {table.num_rows} != {expected_row_count}"
        )
    _verify_zstd(path)
    return table


def _canonical_table(bars: Sequence[StockBar]) -> pa.Table:
    return pa.Table.from_pylist(
        [bar.model_dump(mode="python") for bar in bars],
        schema=CANONICAL_ARROW_SCHEMA,
    )


def verify_canonical_parquet(
    path: Path, *, expected_row_count: int | None = None
) -> list[StockBar]:
    """Read back schema, row count, nullability, compression, and uniqueness."""
    table = _parquet_file_table(path)
    if not table.schema.equals(CANONICAL_ARROW_SCHEMA, check_metadata=False):
        raise ParquetValidationError(
            f"canonical Parquet schema mismatch: {table.schema}"
        )
    if expected_row_count is not None and table.num_rows != expected_row_count:
        raise ParquetValidationError(
            f"canonical Parquet row count {table.num_rows} != {expected_row_count}"
        )
    if any(column.null_count for column in table.columns):
        raise ParquetValidationError("canonical Parquet contains null values")
    _verify_zstd(path)
    try:
        return validate_stock_bar_dataset(table.to_pylist())
    except (TypeError, ValueError) as error:
        raise ParquetValidationError(
            f"canonical Parquet dataset validation failed: {error}"
        ) from error


def write_raw_batch(
    raw_frame: pd.DataFrame,
    manifest: Mapping[str, object],
    *,
    data_root: Path,
    batch_id: str,
) -> Path:
    """Write one immutable raw ``bars.parquet`` and ``manifest.json`` batch."""
    normalized_batch_id = validate_batch_id(batch_id)
    batch_directory = raw_batch_directory(data_root, normalized_batch_id)
    if batch_directory.exists():
        raise BatchExistsError(f"batch directory already exists: {batch_directory}")

    if list(raw_frame.columns) != list(RAW_COLUMNS):
        raise ParquetValidationError(
            "raw frame must contain exactly the frozen raw columns in order"
        )
    missing_manifest = [
        field for field in RAW_MANIFEST_FIELDS if field not in manifest
    ]
    if missing_manifest:
        raise ParquetValidationError(
            f"raw manifest is missing fields: {', '.join(missing_manifest)}"
        )
    if manifest["batch_id"] != normalized_batch_id:
        raise ParquetValidationError("raw manifest batch_id does not match path")
    requested_asset_ids = manifest["requested_asset_ids"]
    if (
        not isinstance(requested_asset_ids, list)
        or requested_asset_ids != sorted(set(requested_asset_ids))
    ):
        raise ParquetValidationError(
            "requested_asset_ids must be a sorted, deduplicated list"
        )
    if manifest["raw_row_count"] != len(raw_frame.index):
        raise ParquetValidationError("raw manifest row count does not match frame")

    parent = batch_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".batch={normalized_batch_id}.tmp-", dir=parent)
    )
    try:
        table = (
            RAW_ARROW_SCHEMA.empty_table()
            if raw_frame.empty
            else pa.Table.from_pandas(
                raw_frame,
                schema=RAW_ARROW_SCHEMA,
                preserve_index=False,
                safe=True,
            )
        )
        parquet_path = temporary / "bars.parquet"
        pq.write_table(table, parquet_path, compression="zstd")
        verify_raw_parquet(parquet_path, expected_row_count=len(raw_frame.index))
        (temporary / "manifest.json").write_text(
            _json_text(manifest), encoding="utf-8", newline=""
        )
        temporary.rename(batch_directory)
    except FileExistsError as error:
        raise BatchExistsError(
            f"batch directory already exists: {batch_directory}"
        ) from error
    except ParquetValidationError:
        raise
    except Exception as error:
        raise ParquetValidationError(f"raw Parquet write failed: {error}") from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return batch_directory


def write_clean_batch(
    rows: Sequence[StockBar | Mapping[str, object]] | pd.DataFrame,
    *,
    data_root: Path,
    batch_id: str,
) -> Path:
    """Write and read-back verify one immutable canonical clean batch."""
    normalized_batch_id = validate_batch_id(batch_id)
    bars = validate_stock_bar_dataset(rows)
    batch_directory = clean_batch_directory(data_root, normalized_batch_id)
    if batch_directory.exists():
        raise BatchExistsError(f"batch directory already exists: {batch_directory}")

    parent = batch_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".batch={normalized_batch_id}.tmp-", dir=parent)
    )
    try:
        parquet_path = temporary / "stock_bars.parquet"
        pq.write_table(_canonical_table(bars), parquet_path, compression="zstd")
        verify_canonical_parquet(parquet_path, expected_row_count=len(bars))
        temporary.rename(batch_directory)
    except FileExistsError as error:
        raise BatchExistsError(
            f"batch directory already exists: {batch_directory}"
        ) from error
    except ParquetValidationError:
        raise
    except Exception as error:
        raise ParquetValidationError(
            f"clean Parquet write failed: {error}"
        ) from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return batch_directory / "stock_bars.parquet"


def write_validation_report(
    report: Mapping[str, object],
    *,
    data_root: Path,
    batch_id: str,
) -> Path:
    """Write a final report once, whether or not clean Parquet exists."""
    batch_directory = clean_batch_directory(data_root, batch_id)
    batch_directory.mkdir(parents=True, exist_ok=True)
    report_path = batch_directory / "validation_report.json"
    try:
        with report_path.open("x", encoding="utf-8", newline="") as file:
            file.write(_json_text(report))
    except FileExistsError as error:
        raise BatchExistsError(
            f"validation report already exists: {report_path}"
        ) from error
    return report_path


def _canonical_key(bar: StockBar) -> tuple[object, ...]:
    values = bar.model_dump(mode="python")
    return tuple(values[field] for field in CANONICAL_KEY)


def _existing_keys_for_year(year_directory: Path) -> set[tuple[object, ...]]:
    keys: set[tuple[object, ...]] = set()
    for path in sorted(year_directory.glob("*.parquet")):
        try:
            bars = verify_canonical_parquet(path)
        except ParquetValidationError as error:
            raise PublishValidationError(
                f"existing published part is invalid: {path}: {error}"
            ) from error
        for bar in bars:
            key = _canonical_key(bar)
            if key in keys:
                raise PublishValidationError(
                    f"existing published data contains duplicate key: {key}",
                    duplicate_count=1,
                )
            keys.add(key)
    return keys


def publish_stock_bars(
    rows: Sequence[StockBar | Mapping[str, object]] | pd.DataFrame,
    *,
    data_root: Path,
    batch_id: str,
) -> tuple[Path, ...]:
    """Safely publish canonical rows as year-only batch parts.

    All affected years are preflighted and staged before atomic file moves. If
    any move or final verification fails, newly moved parts are rolled back.
    Correction and replacement semantics are deliberately not implemented.
    """
    normalized_batch_id = validate_batch_id(batch_id)
    bars = validate_stock_bar_dataset(rows)
    if not bars:
        raise PublishValidationError("cannot publish an empty canonical batch")

    by_year: dict[int, list[StockBar]] = {}
    for bar in bars:
        by_year.setdefault(bar.trading_date.year, []).append(bar)

    published_root = published_dataset_directory(data_root)
    targets: dict[int, Path] = {
        year: published_root
        / f"year={year}"
        / f"part-{normalized_batch_id}.parquet"
        for year in sorted(by_year)
    }
    new_keys = {_canonical_key(bar) for bar in bars}
    for year, target in targets.items():
        if target.exists():
            raise PublishValidationError(
                f"published batch part already exists: {target}"
            )
        existing_keys = _existing_keys_for_year(target.parent)
        overlaps = existing_keys.intersection(new_keys)
        if overlaps:
            sample = sorted(overlaps, key=repr)[0]
            raise PublishValidationError(
                f"new batch overlaps an existing canonical key: {sample}",
                duplicate_count=len(overlaps),
            )

    staging_parent = published_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".stock_bars_v0_1-{normalized_batch_id}.tmp-",
            dir=staging_parent,
        )
    )
    staged: dict[int, Path] = {}
    moved: list[Path] = []
    created_year_directories: list[Path] = []
    try:
        for year, year_bars in by_year.items():
            staged_path = staging_root / f"year={year}" / targets[year].name
            staged_path.parent.mkdir(parents=True, exist_ok=False)
            pq.write_table(
                _canonical_table(year_bars), staged_path, compression="zstd"
            )
            verify_canonical_parquet(
                staged_path, expected_row_count=len(year_bars)
            )
            staged[year] = staged_path

        for year, target in targets.items():
            if not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=False)
                created_year_directories.append(target.parent)
            # A hard link is an atomic, no-clobber publish on the same
            # filesystem. The staging tree is deliberately created beside the
            # published dataset so this cannot cross filesystem boundaries.
            os.link(staged[year], target)
            moved.append(target)
            staged[year].unlink()

        for year, target in targets.items():
            verify_canonical_parquet(
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
        raise PublishValidationError(f"publication failed: {error}") from error
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    return tuple(targets[year] for year in sorted(targets))


__all__ = [
    "CANONICAL_ARROW_SCHEMA",
    "RAW_ARROW_SCHEMA",
    "BatchExistsError",
    "ParquetValidationError",
    "PublishValidationError",
    "clean_batch_directory",
    "ensure_batch_paths_available",
    "publish_stock_bars",
    "published_dataset_directory",
    "raw_batch_directory",
    "validate_batch_id",
    "verify_canonical_parquet",
    "verify_raw_parquet",
    "write_clean_batch",
    "write_raw_batch",
    "write_validation_report",
]
