"""Narrow orchestration for explicit Norgate AssetId D1 batches."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import pandas as pd
from pydantic import ValidationError

from stock_swing_d1.data.norgate_d1_adapter import (
    RAW_COLUMNS,
    ProviderValidationError,
    fetch_norgate_d1,
    prepare_norgate_raw_frame,
    provider_row_to_stock_bar,
    resolve_norgate_symbol,
    security_id_for_asset_id,
    validate_asset_id,
    validate_date_range,
    validate_norgate_raw_frame,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.storage.stock_bar_parquet import (
    ParquetValidationError,
    PublishValidationError,
    ensure_batch_paths_available,
    publish_stock_bars,
    validate_batch_id,
    write_clean_batch,
    write_raw_batch,
    write_validation_report,
)
from stock_swing_d1.validation.stock_bar_dataset import (
    DatasetValidationError,
    validate_stock_bar_dataset,
)


FetchD1 = Callable[[int, date, date], pd.DataFrame]
SymbolResolver = Callable[[int], str]


@dataclass(frozen=True, slots=True)
class D1PipelineResult:
    """Paths and final validation state for one attempted pipeline batch."""

    batch_id: str
    published: bool
    raw_batch_directory: Path | None
    clean_parquet_path: Path | None
    published_paths: tuple[Path, ...]
    validation_report_path: Path
    validation_report: dict[str, object]


def generate_batch_id(extracted_at_utc: datetime | None = None) -> str:
    """Generate the frozen second-resolution UTC batch identifier."""
    instant = extracted_at_utc or datetime.now(timezone.utc)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("extracted_at_utc must be timezone-aware")
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso_utc(instant: datetime) -> str:
    return (
        instant.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _package_version() -> str:
    try:
        return version("norgatedata")
    except PackageNotFoundError:
        # Fake-provider test runs remain portable even when the live adapter is
        # not installed in the test execution environment.
        return "unavailable"


def _normalize_asset_ids(asset_ids: Sequence[int]) -> list[int]:
    if isinstance(asset_ids, (str, bytes)):
        raise ProviderValidationError("asset_ids must be a sequence of AssetIds")
    normalized = [validate_asset_id(asset_id) for asset_id in asset_ids]
    if not normalized:
        raise ProviderValidationError("at least one explicit AssetId is required")
    return sorted(set(normalized))


def _provider_error(
    error: ProviderValidationError,
    *,
    fallback_asset_id: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "stage": "provider_validation",
        "reason": error.reason,
    }
    asset_id = error.provider_asset_id or fallback_asset_id
    if asset_id is not None:
        result["provider_asset_id"] = asset_id
        result["security_id"] = security_id_for_asset_id(asset_id)
    if error.trading_date is not None:
        result["trading_date"] = error.trading_date.isoformat()
    return result


def _stock_bar_error(
    error: ValidationError,
    *,
    asset_id: int,
    trading_date: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "provider_asset_id": asset_id,
        "security_id": security_id_for_asset_id(asset_id),
        "stage": "stock_bar_validation",
        "reason": str(error),
    }
    if type(trading_date) is date:
        result["trading_date"] = trading_date.isoformat()
    return result


def _report(
    *,
    batch_id: str,
    raw_row_count: int,
    bars: Sequence[StockBar],
    invalid_row_count: int,
    duplicate_count: int,
    published: bool,
    errors: Sequence[dict[str, object]],
) -> dict[str, object]:
    trading_dates = [bar.trading_date for bar in bars]
    return {
        "batch_id": batch_id,
        "raw_row_count": raw_row_count,
        "valid_row_count": len(bars),
        "invalid_row_count": invalid_row_count,
        "duplicate_count": duplicate_count,
        "number_of_securities": len({bar.security_id for bar in bars}),
        "minimum_trading_date": (
            min(trading_dates).isoformat() if trading_dates else None
        ),
        "maximum_trading_date": (
            max(trading_dates).isoformat() if trading_dates else None
        ),
        "published": published,
        "errors": list(errors),
    }


def _finish_failed(
    *,
    data_root: Path,
    batch_id: str,
    raw_batch_directory: Path | None,
    clean_parquet_path: Path | None,
    report: dict[str, object],
) -> D1PipelineResult:
    report_path = write_validation_report(
        report, data_root=data_root, batch_id=batch_id
    )
    return D1PipelineResult(
        batch_id=batch_id,
        published=False,
        raw_batch_directory=raw_batch_directory,
        clean_parquet_path=clean_parquet_path,
        published_paths=(),
        validation_report_path=report_path,
        validation_report=report,
    )


def run_d1_pipeline(
    asset_ids: Sequence[int],
    start_date: date,
    end_date: date,
    *,
    data_root: Path = Path("data"),
    batch_id: str | None = None,
    fetcher: FetchD1 | None = None,
    symbol_resolver: SymbolResolver | None = None,
    norgatedata_package_version: str | None = None,
) -> D1PipelineResult:
    """Extract, validate, store, and publish explicit Norgate AssetIds only.

    This function performs no universe discovery, selection, liquidity ranking,
    corporate-action adjustment, earnings join, indicator, or strategy logic.
    Provider exceptions from live fetching and symbol resolution are not
    swallowed. Contract validation failures are recorded and returned with
    ``published=False``.
    """
    normalized_ids = _normalize_asset_ids(asset_ids)
    normalized_start, normalized_end = validate_date_range(start_date, end_date)
    extracted_at = datetime.now(timezone.utc)
    normalized_batch_id = validate_batch_id(
        batch_id or generate_batch_id(extracted_at)
    )
    normalized_root = Path(data_root)
    ensure_batch_paths_available(normalized_root, normalized_batch_id)

    active_fetcher = fetcher or fetch_norgate_d1
    active_symbol_resolver = symbol_resolver or resolve_norgate_symbol
    prepared_frames: list[tuple[int, pd.DataFrame]] = []
    preparation_errors: list[dict[str, object]] = []
    fetched_row_count = 0

    for asset_id in normalized_ids:
        frame = active_fetcher(asset_id, normalized_start, normalized_end)
        if isinstance(frame, pd.DataFrame):
            fetched_row_count += len(frame.index)
        try:
            symbol = active_symbol_resolver(asset_id)
            raw_frame = prepare_norgate_raw_frame(asset_id, symbol, frame)
        except ProviderValidationError as error:
            preparation_errors.append(
                _provider_error(error, fallback_asset_id=asset_id)
            )
            continue
        prepared_frames.append((asset_id, raw_frame))

    if preparation_errors:
        report = _report(
            batch_id=normalized_batch_id,
            raw_row_count=fetched_row_count,
            bars=(),
            invalid_row_count=fetched_row_count,
            duplicate_count=0,
            published=False,
            errors=preparation_errors,
        )
        return _finish_failed(
            data_root=normalized_root,
            batch_id=normalized_batch_id,
            raw_batch_directory=None,
            clean_parquet_path=None,
            report=report,
        )

    if prepared_frames:
        raw_batch = pd.concat(
            [frame for _, frame in prepared_frames], ignore_index=True
        ).loc[:, list(RAW_COLUMNS)]
        raw_batch = raw_batch.sort_values(
            ["provider_asset_id", "trading_date"],
            kind="mergesort",
            ignore_index=True,
        )
    else:
        raw_batch = pd.DataFrame(columns=list(RAW_COLUMNS))

    manifest: dict[str, object] = {
        "provider": "Norgate Data",
        "package": "US Stocks Platinum",
        "batch_id": normalized_batch_id,
        "extracted_at_utc": _iso_utc(extracted_at),
        "requested_asset_ids": normalized_ids,
        "requested_start_date": normalized_start.isoformat(),
        "requested_end_date": normalized_end.isoformat(),
        "adjustment_mode": "NONE",
        "padding_mode": "NONE",
        "norgatedata_python_package_version": (
            norgatedata_package_version
            if norgatedata_package_version is not None
            else _package_version()
        ),
        "raw_row_count": len(raw_batch.index),
    }
    try:
        raw_directory = write_raw_batch(
            raw_batch,
            manifest,
            data_root=normalized_root,
            batch_id=normalized_batch_id,
        )
    except ParquetValidationError as error:
        report = _report(
            batch_id=normalized_batch_id,
            raw_row_count=len(raw_batch.index),
            bars=(),
            invalid_row_count=0,
            duplicate_count=0,
            published=False,
            errors=[{"stage": "parquet_write", "reason": str(error)}],
        )
        return _finish_failed(
            data_root=normalized_root,
            batch_id=normalized_batch_id,
            raw_batch_directory=None,
            clean_parquet_path=None,
            report=report,
        )

    bars: list[StockBar] = []
    validation_errors: list[dict[str, object]] = []
    duplicate_count = 0
    invalid_row_count = 0
    for asset_id, raw_frame in prepared_frames:
        try:
            validate_norgate_raw_frame(raw_frame)
        except ProviderValidationError as error:
            validation_errors.append(
                _provider_error(error, fallback_asset_id=asset_id)
            )
            duplicates = int(
                raw_frame.duplicated(subset=["trading_date"], keep="first").sum()
            )
            duplicate_count += duplicates
            invalid_row_count += len(raw_frame.index)
            continue

        for row in raw_frame.to_dict(orient="records"):
            try:
                bars.append(provider_row_to_stock_bar(row))
            except ProviderValidationError as error:
                invalid_row_count += 1
                validation_errors.append(
                    _provider_error(error, fallback_asset_id=asset_id)
                )
            except ValidationError as error:
                invalid_row_count += 1
                validation_errors.append(
                    _stock_bar_error(
                        error,
                        asset_id=asset_id,
                        trading_date=row.get("trading_date"),
                    )
                )

    if validation_errors:
        report = _report(
            batch_id=normalized_batch_id,
            raw_row_count=len(raw_batch.index),
            bars=bars,
            invalid_row_count=invalid_row_count,
            duplicate_count=duplicate_count,
            published=False,
            errors=validation_errors,
        )
        return _finish_failed(
            data_root=normalized_root,
            batch_id=normalized_batch_id,
            raw_batch_directory=raw_directory,
            clean_parquet_path=None,
            report=report,
        )

    try:
        bars = validate_stock_bar_dataset(bars)
    except DatasetValidationError as error:
        dataset_errors = [issue.as_error() for issue in error.issues]
        report = _report(
            batch_id=normalized_batch_id,
            raw_row_count=len(raw_batch.index),
            bars=bars,
            invalid_row_count=0,
            duplicate_count=error.duplicate_count,
            published=False,
            errors=dataset_errors,
        )
        return _finish_failed(
            data_root=normalized_root,
            batch_id=normalized_batch_id,
            raw_batch_directory=raw_directory,
            clean_parquet_path=None,
            report=report,
        )

    try:
        clean_path = write_clean_batch(
            bars,
            data_root=normalized_root,
            batch_id=normalized_batch_id,
        )
    except (DatasetValidationError, ParquetValidationError) as error:
        report = _report(
            batch_id=normalized_batch_id,
            raw_row_count=len(raw_batch.index),
            bars=bars,
            invalid_row_count=0,
            duplicate_count=(
                error.duplicate_count
                if isinstance(error, DatasetValidationError)
                else 0
            ),
            published=False,
            errors=[{"stage": "parquet_write", "reason": str(error)}],
        )
        return _finish_failed(
            data_root=normalized_root,
            batch_id=normalized_batch_id,
            raw_batch_directory=raw_directory,
            clean_parquet_path=None,
            report=report,
        )

    try:
        published_paths = publish_stock_bars(
            bars,
            data_root=normalized_root,
            batch_id=normalized_batch_id,
        )
    except PublishValidationError as error:
        report = _report(
            batch_id=normalized_batch_id,
            raw_row_count=len(raw_batch.index),
            bars=bars,
            invalid_row_count=0,
            duplicate_count=error.duplicate_count,
            published=False,
            errors=[{"stage": "publish_validation", "reason": str(error)}],
        )
        return _finish_failed(
            data_root=normalized_root,
            batch_id=normalized_batch_id,
            raw_batch_directory=raw_directory,
            clean_parquet_path=clean_path,
            report=report,
        )

    report = _report(
        batch_id=normalized_batch_id,
        raw_row_count=len(raw_batch.index),
        bars=bars,
        invalid_row_count=0,
        duplicate_count=0,
        published=True,
        errors=(),
    )
    report_path = write_validation_report(
        report, data_root=normalized_root, batch_id=normalized_batch_id
    )
    return D1PipelineResult(
        batch_id=normalized_batch_id,
        published=True,
        raw_batch_directory=raw_directory,
        clean_parquet_path=clean_path,
        published_paths=published_paths,
        validation_report_path=report_path,
        validation_report=report,
    )


__all__ = [
    "D1PipelineResult",
    "FetchD1",
    "SymbolResolver",
    "generate_batch_id",
    "run_d1_pipeline",
]
