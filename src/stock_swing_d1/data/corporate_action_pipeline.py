"""Narrow Phase 5B corporate-action provider/data orchestration."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from stock_swing_d1.data.d1_pipeline import generate_batch_id
from stock_swing_d1.data.norgate_adjusted_d1_adapter import (
    ADJUSTED_RAW_COLUMNS,
    adjusted_provider_row_to_bar,
    fetch_norgate_adjusted_d1,
    prepare_norgate_adjusted_raw_frame,
    validate_norgate_adjusted_raw_frame,
)
from stock_swing_d1.data.norgate_capital_event_adapter import (
    EVENT_RAW_COLUMNS,
    CapitalEventProviderValidationError,
    capital_event_row_to_model,
    fetch_norgate_capital_events,
    prepare_norgate_capital_event_raw_frame,
)
from stock_swing_d1.data.norgate_d1_adapter import (
    ProviderValidationError,
    resolve_norgate_symbol,
    security_id_for_asset_id,
    validate_asset_id,
)
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    CorporateActionEvent,
    StockBar,
)
from stock_swing_d1.storage.corporate_action_parquet import (
    CleanCorporateActionPaths,
    CorporateActionDatasetValidationError,
    CorporateActionPublicationPaths,
    ensure_corporate_action_batch_paths_available,
    publish_corporate_action_datasets,
    validate_adjusted_bar_dataset,
    validate_corporate_action_event_dataset,
    write_corporate_action_clean_batch,
    write_corporate_action_raw_batch,
    write_corporate_action_validation_report,
)
from stock_swing_d1.storage.stock_bar_parquet import (
    ParquetValidationError,
    PublishValidationError,
    validate_batch_id,
    verify_canonical_parquet,
)
from stock_swing_d1.validation.corporate_action_parity import (
    CorporateActionParityResult,
    EventPhase4CrosscheckError,
    check_corporate_action_parity,
    validate_events_against_phase4,
)


AdjustedFetcher = Callable[[int, date, date], pd.DataFrame]
EventFetcher = Callable[[int], pd.DataFrame]
SymbolResolver = Callable[[int], str]


@dataclass(frozen=True, slots=True)
class CorporateActionPipelineResult:
    """Paths and final state for one attempted Phase 5B batch."""

    phase5_batch_id: str
    phase4_source_batch_id: str
    published: bool
    raw_batch_directory: Path | None
    clean_adjusted_parquet_path: Path | None
    clean_event_parquet_path: Path | None
    adjusted_published_paths: tuple[Path, ...]
    event_published_paths: tuple[Path, ...]
    validation_report_path: Path
    validation_report: dict[str, object]

    @property
    def batch_id(self) -> str:
        """Compatibility-friendly shorthand for the Phase 5 batch ID."""
        return self.phase5_batch_id


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
        return "unavailable"


def _source_path(data_root: Path, source_batch_id: str) -> Path:
    return (
        Path(data_root)
        / "clean"
        / "d1"
        / f"batch={validate_batch_id(source_batch_id)}"
        / "stock_bars.parquet"
    )


def _error(
    stage: str,
    reason: str,
    *,
    asset_id: int | None = None,
    trading_date: date | None = None,
    event_date: date | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"stage": stage, "reason": reason}
    if asset_id is not None:
        result["provider_asset_id"] = asset_id
        result["security_id"] = security_id_for_asset_id(asset_id)
    if trading_date is not None:
        result["trading_date"] = trading_date.isoformat()
    if event_date is not None:
        result["event_date"] = event_date.isoformat()
    return result


def _report(
    *,
    phase5_batch_id: str,
    phase4_source_batch_id: str,
    phase4_bars: Sequence[StockBar],
    adjusted_raw_row_count: int,
    adjusted_bars: Sequence[CorporateActionAdjustedStockBar],
    adjusted_invalid_row_count: int,
    raw_event_flag_row_count: int,
    events: Sequence[CorporateActionEvent],
    event_invalid_count: int,
    parity: CorporateActionParityResult | None,
    adjusted_published: bool,
    events_validated: bool,
    published: bool,
    errors: Sequence[dict[str, object]],
) -> dict[str, object]:
    dates = [bar.trading_date for bar in phase4_bars]
    return {
        "phase5_batch_id": phase5_batch_id,
        "phase4_source_batch_id": phase4_source_batch_id,
        "phase4_row_count": len(phase4_bars),
        "adjusted_raw_row_count": adjusted_raw_row_count,
        "adjusted_valid_row_count": len(adjusted_bars),
        "adjusted_invalid_row_count": adjusted_invalid_row_count,
        "raw_event_flag_row_count": raw_event_flag_row_count,
        "capital_event_count": len(events),
        "event_invalid_count": event_invalid_count,
        "raw_adjusted_missing_key_count": (
            parity.raw_adjusted_missing_key_count if parity else 0
        ),
        "raw_adjusted_extra_key_count": (
            parity.raw_adjusted_extra_key_count if parity else 0
        ),
        "unadjusted_close_mismatch_count": (
            parity.unadjusted_close_mismatch_count if parity else 0
        ),
        "number_of_securities": len({bar.security_id for bar in phase4_bars}),
        "minimum_trading_date": min(dates).isoformat() if dates else None,
        "maximum_trading_date": max(dates).isoformat() if dates else None,
        "minimum_adjustment_factor": (
            parity.minimum_adjustment_factor if parity else None
        ),
        "maximum_adjustment_factor": (
            parity.maximum_adjustment_factor if parity else None
        ),
        "adjusted_published": adjusted_published,
        "events_validated": events_validated,
        "published": published,
        "errors": list(errors),
    }


def _finish(
    *,
    data_root: Path,
    phase5_batch_id: str,
    phase4_source_batch_id: str,
    raw_directory: Path | None,
    clean_paths: CleanCorporateActionPaths | None,
    publication_paths: CorporateActionPublicationPaths | None,
    report: dict[str, object],
) -> CorporateActionPipelineResult:
    report_path = write_corporate_action_validation_report(
        report, data_root=data_root, batch_id=phase5_batch_id
    )
    return CorporateActionPipelineResult(
        phase5_batch_id=phase5_batch_id,
        phase4_source_batch_id=phase4_source_batch_id,
        published=bool(report["published"]),
        raw_batch_directory=raw_directory,
        clean_adjusted_parquet_path=(
            clean_paths.adjusted_stock_bars if clean_paths else None
        ),
        clean_event_parquet_path=(
            clean_paths.corporate_action_events if clean_paths else None
        ),
        adjusted_published_paths=(
            publication_paths.adjusted_stock_bars if publication_paths else ()
        ),
        event_published_paths=(
            publication_paths.corporate_action_events if publication_paths else ()
        ),
        validation_report_path=report_path,
        validation_report=report,
    )


def _phase4_scope(
    bars: Sequence[StockBar],
) -> tuple[list[int], dict[int, tuple[date, date]]]:
    if not bars:
        raise ValueError("Phase 4 source must contain at least one StockBar")
    dates_by_asset: dict[int, list[date]] = {}
    for bar in bars:
        match = re.fullmatch(r"NORGATE:([1-9][0-9]*)", bar.security_id)
        if match is None:
            raise ValueError(
                f"malformed Phase 4 security_id: {bar.security_id!r}"
            )
        asset_id = validate_asset_id(int(match.group(1)))
        dates_by_asset.setdefault(asset_id, []).append(bar.trading_date)
    asset_ids = sorted(dates_by_asset)
    ranges = {
        asset_id: (min(dates_by_asset[asset_id]), max(dates_by_asset[asset_id]))
        for asset_id in asset_ids
    }
    return asset_ids, ranges


def _concat_frames(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True).loc[:, list(columns)].sort_values(
        [columns[0], columns[2]], kind="mergesort", ignore_index=True
    )


def run_corporate_action_pipeline(
    phase4_source_batch_id: str,
    *,
    data_root: Path = Path("data"),
    phase5_batch_id: str | None = None,
    adjusted_fetcher: AdjustedFetcher | None = None,
    event_fetcher: EventFetcher | None = None,
    symbol_resolver: SymbolResolver | None = None,
    norgatedata_package_version: str | None = None,
) -> CorporateActionPipelineResult:
    """Derive and publish Phase 5B data from one immutable Phase 4 clean batch."""
    normalized_source_id = validate_batch_id(phase4_source_batch_id)
    extracted_at = datetime.now(timezone.utc)
    normalized_phase5_id = validate_batch_id(
        phase5_batch_id or generate_batch_id(extracted_at)
    )
    normalized_root = Path(data_root)
    ensure_corporate_action_batch_paths_available(
        normalized_root, normalized_phase5_id
    )
    source_path = _source_path(normalized_root, normalized_source_id)

    phase4_bars: list[StockBar] = []
    try:
        if not source_path.is_file():
            raise FileNotFoundError(f"Phase 4 source does not exist: {source_path}")
        phase4_bars = verify_canonical_parquet(source_path)
        asset_ids, date_ranges = _phase4_scope(phase4_bars)
    except Exception as error:
        report = _report(
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            phase4_bars=phase4_bars,
            adjusted_raw_row_count=0,
            adjusted_bars=(),
            adjusted_invalid_row_count=0,
            raw_event_flag_row_count=0,
            events=(),
            event_invalid_count=0,
            parity=None,
            adjusted_published=False,
            events_validated=False,
            published=False,
            errors=[_error("phase4_source_validation", str(error))],
        )
        return _finish(
            data_root=normalized_root,
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            raw_directory=None,
            clean_paths=None,
            publication_paths=None,
            report=report,
        )

    active_adjusted_fetcher = adjusted_fetcher or fetch_norgate_adjusted_d1
    active_event_fetcher = event_fetcher or fetch_norgate_capital_events
    active_symbol_resolver = symbol_resolver or resolve_norgate_symbol
    adjusted_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    preparation_errors: list[dict[str, object]] = []
    adjusted_invalid_count = 0
    event_invalid_count = 0

    for asset_id in asset_ids:
        start_date, end_date = date_ranges[asset_id]
        adjusted_response = active_adjusted_fetcher(asset_id, start_date, end_date)
        event_response = active_event_fetcher(asset_id)
        symbol = active_symbol_resolver(asset_id)
        try:
            adjusted_frames.append(
                prepare_norgate_adjusted_raw_frame(
                    asset_id, symbol, adjusted_response
                )
            )
        except ProviderValidationError as error:
            adjusted_invalid_count += (
                1
                if error.trading_date is not None
                else (
                    len(adjusted_response.index)
                    if isinstance(adjusted_response, pd.DataFrame)
                    else 0
                )
            )
            preparation_errors.append(
                _error(
                    "adjusted_provider_validation",
                    error.reason,
                    asset_id=error.provider_asset_id or asset_id,
                    trading_date=error.trading_date,
                )
            )
        try:
            event_frames.append(
                prepare_norgate_capital_event_raw_frame(
                    asset_id,
                    symbol,
                    event_response,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        except CapitalEventProviderValidationError as error:
            event_invalid_count += 1 if error.event_date is not None else 0
            preparation_errors.append(
                _error(
                    "event_provider_validation",
                    error.reason,
                    asset_id=error.provider_asset_id or asset_id,
                    event_date=error.event_date,
                )
            )

    adjusted_raw = _concat_frames(adjusted_frames, ADJUSTED_RAW_COLUMNS)
    event_raw = _concat_frames(event_frames, EVENT_RAW_COLUMNS)
    if preparation_errors:
        report = _report(
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            phase4_bars=phase4_bars,
            adjusted_raw_row_count=len(adjusted_raw.index),
            adjusted_bars=(),
            adjusted_invalid_row_count=adjusted_invalid_count,
            raw_event_flag_row_count=len(event_raw.index),
            events=(),
            event_invalid_count=event_invalid_count,
            parity=None,
            adjusted_published=False,
            events_validated=False,
            published=False,
            errors=preparation_errors,
        )
        return _finish(
            data_root=normalized_root,
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            raw_directory=None,
            clean_paths=None,
            publication_paths=None,
            report=report,
        )

    manifest: dict[str, object] = {
        "provider": "Norgate Data",
        "package": "US Stocks Platinum",
        "phase5_batch_id": normalized_phase5_id,
        "phase4_source_batch_id": normalized_source_id,
        "phase4_source_path": str(source_path),
        "extracted_at_utc": _iso_utc(extracted_at),
        "requested_asset_ids": asset_ids,
        "requested_date_ranges": [
            {
                "provider_asset_id": asset_id,
                "start_date": date_ranges[asset_id][0].isoformat(),
                "end_date": date_ranges[asset_id][1].isoformat(),
            }
            for asset_id in asset_ids
        ],
        "adjustment_mode": "CAPITALSPECIAL",
        "padding_mode": "NONE",
        "interval": "D",
        "adjusted_raw_row_count": len(adjusted_raw.index),
        "capital_event_flag_row_count": len(event_raw.index),
        "norgatedata_python_package_version": (
            norgatedata_package_version
            if norgatedata_package_version is not None
            else _package_version()
        ),
        "event_source_function": "capital_event_timeseries",
        "event_date_filtering": "local_phase4_date_range",
    }
    try:
        raw_directory = write_corporate_action_raw_batch(
            adjusted_raw,
            event_raw,
            manifest,
            data_root=normalized_root,
            batch_id=normalized_phase5_id,
        )
    except ParquetValidationError as error:
        report = _report(
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            phase4_bars=phase4_bars,
            adjusted_raw_row_count=len(adjusted_raw.index),
            adjusted_bars=(),
            adjusted_invalid_row_count=0,
            raw_event_flag_row_count=len(event_raw.index),
            events=(),
            event_invalid_count=0,
            parity=None,
            adjusted_published=False,
            events_validated=False,
            published=False,
            errors=[_error("parquet_write", str(error))],
        )
        return _finish(
            data_root=normalized_root,
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            raw_directory=None,
            clean_paths=None,
            publication_paths=None,
            report=report,
        )

    adjusted_bars: list[CorporateActionAdjustedStockBar] = []
    events: list[CorporateActionEvent] = []
    validation_errors: list[dict[str, object]] = []
    adjusted_invalid_count = 0
    event_invalid_count = 0
    for asset_id, raw_frame in zip(asset_ids, adjusted_frames):
        try:
            validate_norgate_adjusted_raw_frame(raw_frame)
        except ProviderValidationError as error:
            adjusted_invalid_count += len(raw_frame.index)
            validation_errors.append(
                _error(
                    "adjusted_provider_validation",
                    error.reason,
                    asset_id=error.provider_asset_id or asset_id,
                    trading_date=error.trading_date,
                )
            )
            continue
        for row in raw_frame.to_dict(orient="records"):
            try:
                adjusted_bars.append(adjusted_provider_row_to_bar(row))
            except ProviderValidationError as error:
                adjusted_invalid_count += 1
                validation_errors.append(
                    _error(
                        "adjusted_provider_validation",
                        error.reason,
                        asset_id=error.provider_asset_id or asset_id,
                        trading_date=error.trading_date,
                    )
                )
            except ValidationError as error:
                adjusted_invalid_count += 1
                validation_errors.append(
                    _error(
                        "adjusted_model_validation",
                        str(error),
                        asset_id=asset_id,
                        trading_date=row.get("trading_date")
                        if type(row.get("trading_date")) is date
                        else None,
                    )
                )
    try:
        adjusted_bars = validate_adjusted_bar_dataset(adjusted_bars)
    except CorporateActionDatasetValidationError as error:
        validation_errors.append(_error("adjusted_model_validation", str(error)))

    parity = check_corporate_action_parity(
        phase4_bars, adjusted_bars, adjusted_raw
    )
    validation_errors.extend(parity.errors)

    for raw_frame in event_frames:
        for row in raw_frame.to_dict(orient="records"):
            if row["capital_event_flag"] != 1:
                continue
            try:
                events.append(capital_event_row_to_model(row))
            except (CapitalEventProviderValidationError, ValidationError) as error:
                event_invalid_count += 1
                asset_id = int(row["provider_asset_id"])
                event_date = row.get("event_date")
                validation_errors.append(
                    _error(
                        "event_model_validation",
                        str(error),
                        asset_id=asset_id,
                        event_date=event_date if type(event_date) is date else None,
                    )
                )
    try:
        events = validate_corporate_action_event_dataset(events)
    except CorporateActionDatasetValidationError as error:
        event_invalid_count += 1
        validation_errors.append(_error("event_model_validation", str(error)))
    try:
        validate_events_against_phase4(events, phase4_bars)
    except EventPhase4CrosscheckError as error:
        event_invalid_count += len(error.errors)
        validation_errors.extend(error.errors)

    events_validated = event_invalid_count == 0
    if validation_errors:
        report = _report(
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            phase4_bars=phase4_bars,
            adjusted_raw_row_count=len(adjusted_raw.index),
            adjusted_bars=adjusted_bars,
            adjusted_invalid_row_count=adjusted_invalid_count,
            raw_event_flag_row_count=len(event_raw.index),
            events=events,
            event_invalid_count=event_invalid_count,
            parity=parity,
            adjusted_published=False,
            events_validated=events_validated,
            published=False,
            errors=validation_errors,
        )
        return _finish(
            data_root=normalized_root,
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            raw_directory=raw_directory,
            clean_paths=None,
            publication_paths=None,
            report=report,
        )

    try:
        clean_paths = write_corporate_action_clean_batch(
            adjusted_bars,
            events,
            data_root=normalized_root,
            batch_id=normalized_phase5_id,
        )
    except (CorporateActionDatasetValidationError, ParquetValidationError) as error:
        report = _report(
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            phase4_bars=phase4_bars,
            adjusted_raw_row_count=len(adjusted_raw.index),
            adjusted_bars=adjusted_bars,
            adjusted_invalid_row_count=0,
            raw_event_flag_row_count=len(event_raw.index),
            events=events,
            event_invalid_count=0,
            parity=parity,
            adjusted_published=False,
            events_validated=True,
            published=False,
            errors=[_error("parquet_write", str(error))],
        )
        return _finish(
            data_root=normalized_root,
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            raw_directory=raw_directory,
            clean_paths=None,
            publication_paths=None,
            report=report,
        )

    try:
        publication_paths = publish_corporate_action_datasets(
            adjusted_bars,
            events,
            data_root=normalized_root,
            batch_id=normalized_phase5_id,
        )
    except PublishValidationError as error:
        report = _report(
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            phase4_bars=phase4_bars,
            adjusted_raw_row_count=len(adjusted_raw.index),
            adjusted_bars=adjusted_bars,
            adjusted_invalid_row_count=0,
            raw_event_flag_row_count=len(event_raw.index),
            events=events,
            event_invalid_count=0,
            parity=parity,
            adjusted_published=False,
            events_validated=True,
            published=False,
            errors=[_error("publish_validation", str(error))],
        )
        return _finish(
            data_root=normalized_root,
            phase5_batch_id=normalized_phase5_id,
            phase4_source_batch_id=normalized_source_id,
            raw_directory=raw_directory,
            clean_paths=clean_paths,
            publication_paths=None,
            report=report,
        )

    report = _report(
        phase5_batch_id=normalized_phase5_id,
        phase4_source_batch_id=normalized_source_id,
        phase4_bars=phase4_bars,
        adjusted_raw_row_count=len(adjusted_raw.index),
        adjusted_bars=adjusted_bars,
        adjusted_invalid_row_count=0,
        raw_event_flag_row_count=len(event_raw.index),
        events=events,
        event_invalid_count=0,
        parity=parity,
        adjusted_published=True,
        events_validated=True,
        published=True,
        errors=(),
    )
    return _finish(
        data_root=normalized_root,
        phase5_batch_id=normalized_phase5_id,
        phase4_source_batch_id=normalized_source_id,
        raw_directory=raw_directory,
        clean_paths=clean_paths,
        publication_paths=publication_paths,
        report=report,
    )


__all__ = [
    "AdjustedFetcher",
    "CorporateActionPipelineResult",
    "EventFetcher",
    "SymbolResolver",
    "run_corporate_action_pipeline",
]
