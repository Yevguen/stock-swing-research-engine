"""Phase 5C.2 DividendEvent derivation orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

from stock_swing_d1.data.d1_pipeline import generate_batch_id
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    DividendEvent,
    StockBar,
)
from stock_swing_d1.storage.corporate_action_parquet import (
    raw_corporate_action_batch_directory,
    clean_corporate_action_batch_directory,
    verify_adjusted_clean_parquet,
    verify_adjusted_raw_parquet,
)
from stock_swing_d1.storage.dividend_event_parquet import (
    CleanDividendPaths,
    ensure_dividend_batch_path_available,
    publish_dividend_events,
    write_dividend_clean_batch,
    write_dividend_validation_report,
)
from stock_swing_d1.storage.stock_bar_parquet import (
    ParquetValidationError,
    PublishValidationError,
    clean_batch_directory,
    validate_batch_id,
    verify_canonical_parquet,
)
from stock_swing_d1.validation.dividend_event_dataset import (
    DividendCrosscheckResult,
    DividendDerivationResult,
    DividendEventDatasetValidationError,
    check_dividend_event_crosschecks,
    check_dividend_source_rows,
    validate_dividend_event_dataset,
)


@dataclass(frozen=True, slots=True)
class DividendEventPipelineResult:
    """Paths, provenance, and final state for one Phase 5C.2 attempt."""

    dividend_batch_id: str
    phase5_source_batch_id: str
    phase4_source_batch_id: str | None
    published: bool
    clean_parquet_path: Path | None
    manifest_path: Path | None
    published_paths: tuple[Path, ...]
    validation_report_path: Path
    validation_report: dict[str, object]

    @property
    def batch_id(self) -> str:
        return self.dividend_batch_id


@dataclass(frozen=True, slots=True)
class _SourceData:
    raw_path: Path
    adjusted_path: Path
    phase4_batch_id: str
    phase4_path: Path
    raw_table: pa.Table
    adjusted_bars: tuple[CorporateActionAdjustedStockBar, ...]
    phase4_bars: tuple[StockBar, ...]


def _iso_utc(instant: datetime) -> str:
    return (
        instant.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _required_value(
    source: Mapping[str, object], field: str, label: str
) -> object:
    if field not in source:
        raise ValueError(f"{label} is missing {field}")
    return source[field]


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _load_and_validate_source(
    data_root: Path, phase5_batch_id: str
) -> _SourceData:
    raw_directory = raw_corporate_action_batch_directory(
        data_root, phase5_batch_id
    )
    clean_directory = clean_corporate_action_batch_directory(
        data_root, phase5_batch_id
    )
    raw_path = raw_directory / "adjusted_bars.parquet"
    manifest_path = raw_directory / "manifest.json"
    adjusted_path = clean_directory / "adjusted_stock_bars.parquet"
    report_path = clean_directory / "validation_report.json"
    required_paths = (raw_path, manifest_path, adjusted_path, report_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError("missing Phase 5B source artifacts: " + ", ".join(missing))

    manifest = _load_json_object(manifest_path, "Phase 5B manifest")
    report = _load_json_object(report_path, "Phase 5B validation report")
    if _required_value(report, "published", "Phase 5B report") is not True:
        raise ValueError("Phase 5B source report published must be true")
    if _required_value(report, "adjusted_published", "Phase 5B report") is not True:
        raise ValueError("Phase 5B source report adjusted_published must be true")
    if _required_value(report, "errors", "Phase 5B report") != []:
        raise ValueError("Phase 5B source report errors must be an empty list")

    expected_manifest = {
        "provider": "Norgate Data",
        "adjustment_mode": "CAPITALSPECIAL",
        "padding_mode": "NONE",
        "interval": "D",
    }
    for field, expected in expected_manifest.items():
        actual = _required_value(manifest, field, "Phase 5B manifest")
        if actual != expected:
            raise ValueError(
                f"Phase 5B manifest {field} must be {expected!r}, got {actual!r}"
            )
    if _required_value(manifest, "phase5_batch_id", "Phase 5B manifest") != phase5_batch_id:
        raise ValueError("Phase 5B manifest batch ID does not match source path")
    if _required_value(report, "phase5_batch_id", "Phase 5B report") != phase5_batch_id:
        raise ValueError("Phase 5B report batch ID does not match source path")

    raw_row_count = _nonnegative_int(
        _required_value(manifest, "adjusted_raw_row_count", "Phase 5B manifest"),
        "Phase 5B adjusted_raw_row_count",
    )
    report_raw_count = _nonnegative_int(
        _required_value(report, "adjusted_raw_row_count", "Phase 5B report"),
        "Phase 5B report adjusted_raw_row_count",
    )
    if report_raw_count != raw_row_count:
        raise ValueError("Phase 5B manifest/report adjusted row counts differ")

    phase4_batch_id = validate_batch_id(
        _required_value(manifest, "phase4_source_batch_id", "Phase 5B manifest")
    )
    if _required_value(report, "phase4_source_batch_id", "Phase 5B report") != phase4_batch_id:
        raise ValueError("Phase 5B manifest/report Phase 4 batch IDs differ")
    expected_phase4_path = (
        clean_batch_directory(data_root, phase4_batch_id) / "stock_bars.parquet"
    )
    phase4_path_value = _required_value(
        manifest, "phase4_source_path", "Phase 5B manifest"
    )
    if not isinstance(phase4_path_value, str) or not phase4_path_value:
        raise ValueError("Phase 5B manifest phase4_source_path must be a path string")
    phase4_path = Path(phase4_path_value)
    if ".." in phase4_path.parts:
        raise ValueError("Phase 5B manifest Phase 4 source path is unsafe")
    if not _same_path(phase4_path, expected_phase4_path):
        raise ValueError(
            "Phase 5B manifest Phase 4 source path does not match its source batch"
        )
    if not phase4_path.is_file():
        raise ValueError(f"referenced Phase 4 StockBar file does not exist: {phase4_path}")

    raw_table = verify_adjusted_raw_parquet(
        raw_path, expected_row_count=raw_row_count
    )
    adjusted_expected = _nonnegative_int(
        _required_value(report, "adjusted_valid_row_count", "Phase 5B report"),
        "Phase 5B report adjusted_valid_row_count",
    )
    adjusted_bars = verify_adjusted_clean_parquet(
        adjusted_path, expected_row_count=adjusted_expected
    )
    phase4_bars = verify_canonical_parquet(phase4_path)
    if "phase4_row_count" in report:
        phase4_expected = _nonnegative_int(
            report["phase4_row_count"], "Phase 5B report phase4_row_count"
        )
        if len(phase4_bars) != phase4_expected:
            raise ValueError("Phase 5B report Phase 4 row count does not match source")
    return _SourceData(
        raw_path=raw_path,
        adjusted_path=adjusted_path,
        phase4_batch_id=phase4_batch_id,
        phase4_path=phase4_path,
        raw_table=raw_table,
        adjusted_bars=tuple(adjusted_bars),
        phase4_bars=tuple(phase4_bars),
    )


def _error(stage: str, reason: str) -> dict[str, object]:
    return {"stage": stage, "reason": reason}


def _diagnostic_phase4_batch_id(
    data_root: Path, phase5_batch_id: str
) -> str | None:
    """Recover only a safe provenance ID for a failed-source report."""
    manifest_path = (
        raw_corporate_action_batch_directory(data_root, phase5_batch_id)
        / "manifest.json"
    )
    try:
        manifest = _load_json_object(manifest_path, "Phase 5B manifest")
        return validate_batch_id(manifest["phase4_source_batch_id"])
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _report(
    *,
    dividend_batch_id: str,
    phase5_source_batch_id: str,
    phase4_source_batch_id: str | None,
    source_raw_row_count: int,
    derivation: DividendDerivationResult | None,
    events: Sequence[DividendEvent],
    crosscheck: DividendCrosscheckResult | None,
    duplicate_count: int,
    dividend_published: bool,
    published: bool,
    errors: Sequence[dict[str, object]],
) -> dict[str, object]:
    dates = [event.entitlement_date for event in events]
    return {
        "dividend_batch_id": dividend_batch_id,
        "phase5_source_batch_id": phase5_source_batch_id,
        "phase4_source_batch_id": phase4_source_batch_id,
        "source_raw_row_count": source_raw_row_count,
        "raw_dividend_nonzero_count": (
            derivation.raw_dividend_nonzero_count if derivation else 0
        ),
        "dividend_event_count": len(events),
        "dividend_invalid_count": (
            derivation.dividend_invalid_count if derivation else 0
        ),
        "dividend_missing_adjusted_bar_count": (
            crosscheck.dividend_missing_adjusted_bar_count if crosscheck else 0
        ),
        "dividend_missing_phase4_bar_count": (
            crosscheck.dividend_missing_phase4_bar_count if crosscheck else 0
        ),
        "duplicate_dividend_key_count": duplicate_count,
        "number_of_securities_with_dividends": len(
            {event.security_id for event in events}
        ),
        "minimum_entitlement_date": min(dates).isoformat() if dates else None,
        "maximum_entitlement_date": max(dates).isoformat() if dates else None,
        "dividend_published": dividend_published,
        "published": published,
        "errors": list(errors),
    }


def _finish(
    *,
    data_root: Path,
    dividend_batch_id: str,
    phase5_source_batch_id: str,
    phase4_source_batch_id: str | None,
    clean_paths: CleanDividendPaths | None,
    published_paths: tuple[Path, ...],
    report: dict[str, object],
) -> DividendEventPipelineResult:
    report_path = write_dividend_validation_report(
        report, data_root=data_root, batch_id=dividend_batch_id
    )
    return DividendEventPipelineResult(
        dividend_batch_id=dividend_batch_id,
        phase5_source_batch_id=phase5_source_batch_id,
        phase4_source_batch_id=phase4_source_batch_id,
        published=bool(report["published"]),
        clean_parquet_path=clean_paths.dividend_events if clean_paths else None,
        manifest_path=clean_paths.manifest if clean_paths else None,
        published_paths=published_paths,
        validation_report_path=report_path,
        validation_report=report,
    )


def run_dividend_event_pipeline(
    phase5_source_batch_id: str,
    *,
    data_root: Path = Path("data"),
    dividend_batch_id: str | None = None,
) -> DividendEventPipelineResult:
    """Derive, validate, store, and publish only preserved raw dividends.

    This pipeline performs no provider call, price inference, shares-entitled
    calculation, cash-flow operation, payment-date logic, tax treatment, or
    reinvestment. Contract failures are reported with ``published=False``.
    """
    normalized_source_id = validate_batch_id(phase5_source_batch_id)
    derived_at = datetime.now(timezone.utc)
    normalized_dividend_id = validate_batch_id(
        dividend_batch_id or generate_batch_id(derived_at)
    )
    normalized_root = Path(data_root)
    ensure_dividend_batch_path_available(
        normalized_root, normalized_dividend_id
    )

    source: _SourceData | None = None
    derivation: DividendDerivationResult | None = None
    events: list[DividendEvent] = []
    crosscheck: DividendCrosscheckResult | None = None
    duplicate_count = 0
    try:
        source = _load_and_validate_source(normalized_root, normalized_source_id)
    except (OSError, TypeError, ValueError, ParquetValidationError) as error:
        diagnostic_phase4_id = _diagnostic_phase4_batch_id(
            normalized_root, normalized_source_id
        )
        report = _report(
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=diagnostic_phase4_id,
            source_raw_row_count=0,
            derivation=None,
            events=(),
            crosscheck=None,
            duplicate_count=0,
            dividend_published=False,
            published=False,
            errors=[_error("phase5_source_validation", str(error))],
        )
        return _finish(
            data_root=normalized_root,
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=diagnostic_phase4_id,
            clean_paths=None,
            published_paths=(),
            report=report,
        )

    derivation = check_dividend_source_rows(source.raw_table)
    events = list(derivation.events)
    if not derivation.valid:
        report = _report(
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            source_raw_row_count=source.raw_table.num_rows,
            derivation=derivation,
            events=events,
            crosscheck=None,
            duplicate_count=0,
            dividend_published=False,
            published=False,
            errors=derivation.errors,
        )
        return _finish(
            data_root=normalized_root,
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            clean_paths=None,
            published_paths=(),
            report=report,
        )

    try:
        events = validate_dividend_event_dataset(events)
    except DividendEventDatasetValidationError as error:
        duplicate_count = error.duplicate_count
        report = _report(
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            source_raw_row_count=source.raw_table.num_rows,
            derivation=derivation,
            events=events,
            crosscheck=None,
            duplicate_count=duplicate_count,
            dividend_published=False,
            published=False,
            errors=error.errors,
        )
        return _finish(
            data_root=normalized_root,
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            clean_paths=None,
            published_paths=(),
            report=report,
        )

    crosscheck = check_dividend_event_crosschecks(
        events, source.adjusted_bars, source.phase4_bars
    )
    if not crosscheck.valid:
        report = _report(
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            source_raw_row_count=source.raw_table.num_rows,
            derivation=derivation,
            events=events,
            crosscheck=crosscheck,
            duplicate_count=0,
            dividend_published=False,
            published=False,
            errors=crosscheck.errors,
        )
        return _finish(
            data_root=normalized_root,
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            clean_paths=None,
            published_paths=(),
            report=report,
        )

    manifest: dict[str, object] = {
        "dividend_batch_id": normalized_dividend_id,
        "phase5_source_batch_id": normalized_source_id,
        "phase5_source_raw_path": str(source.raw_path),
        "phase5_source_adjusted_clean_path": str(source.adjusted_path),
        "phase4_source_batch_id": source.phase4_batch_id,
        "phase4_source_path": str(source.phase4_path),
        "derived_at_utc": _iso_utc(derived_at),
        "provider": "Norgate Data",
        "source_adjustment_mode": "CAPITALSPECIAL",
        "source_raw_row_count": source.raw_table.num_rows,
        "source_dividend_nonzero_count": derivation.raw_dividend_nonzero_count,
    }
    clean_paths: CleanDividendPaths | None = None
    try:
        clean_paths = write_dividend_clean_batch(
            events,
            manifest,
            data_root=normalized_root,
            batch_id=normalized_dividend_id,
            source_raw_row_count=source.raw_table.num_rows,
        )
    except (DividendEventDatasetValidationError, ParquetValidationError) as error:
        report = _report(
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            source_raw_row_count=source.raw_table.num_rows,
            derivation=derivation,
            events=events,
            crosscheck=crosscheck,
            duplicate_count=getattr(error, "duplicate_count", 0),
            dividend_published=False,
            published=False,
            errors=[_error("parquet_write", str(error))],
        )
        return _finish(
            data_root=normalized_root,
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            clean_paths=None,
            published_paths=(),
            report=report,
        )

    try:
        published_paths = publish_dividend_events(
            events,
            data_root=normalized_root,
            batch_id=normalized_dividend_id,
        )
    except PublishValidationError as error:
        duplicate_count = error.duplicate_count
        report = _report(
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            source_raw_row_count=source.raw_table.num_rows,
            derivation=derivation,
            events=events,
            crosscheck=crosscheck,
            duplicate_count=duplicate_count,
            dividend_published=False,
            published=False,
            errors=[_error("publish_validation", str(error))],
        )
        return _finish(
            data_root=normalized_root,
            dividend_batch_id=normalized_dividend_id,
            phase5_source_batch_id=normalized_source_id,
            phase4_source_batch_id=source.phase4_batch_id,
            clean_paths=clean_paths,
            published_paths=(),
            report=report,
        )

    report = _report(
        dividend_batch_id=normalized_dividend_id,
        phase5_source_batch_id=normalized_source_id,
        phase4_source_batch_id=source.phase4_batch_id,
        source_raw_row_count=source.raw_table.num_rows,
        derivation=derivation,
        events=events,
        crosscheck=crosscheck,
        duplicate_count=0,
        dividend_published=True,
        published=True,
        errors=[],
    )
    return _finish(
        data_root=normalized_root,
        dividend_batch_id=normalized_dividend_id,
        phase5_source_batch_id=normalized_source_id,
        phase4_source_batch_id=source.phase4_batch_id,
        clean_paths=clean_paths,
        published_paths=published_paths,
        report=report,
    )


__all__ = ["DividendEventPipelineResult", "run_dividend_event_pipeline"]
