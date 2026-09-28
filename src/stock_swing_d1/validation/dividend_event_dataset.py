"""DividendEvent derivation and dataset cross-validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from numbers import Real
from typing import Any

import pandas as pd
import pyarrow as pa
from pydantic import ValidationError

from stock_swing_d1.data.norgate_d1_adapter import (
    security_id_for_asset_id,
    validate_asset_id,
    validate_provider_symbol,
)
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    DividendEvent,
    StockBar,
)


DIVIDEND_EVENT_FIELDS = (
    "security_id",
    "symbol",
    "entitlement_date",
    "date_semantics",
    "dividend_type",
    "amount_per_share",
    "currency",
    "source_provider",
    "source_asset_id",
    "source_adjustment_mode",
)
DIVIDEND_CANONICAL_KEY = (
    "security_id",
    "entitlement_date",
    "source_provider",
    "dividend_type",
)
DIVIDEND_RAW_REQUIRED_COLUMNS = (
    "provider_asset_id",
    "provider_symbol",
    "trading_date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Turnover",
    "Unadjusted Close",
    "Dividend",
)


@dataclass(frozen=True, slots=True)
class DividendDerivationResult:
    """Validated positive dividend rows and complete source diagnostics."""

    source_raw_row_count: int
    raw_dividend_nonzero_count: int
    dividend_invalid_count: int
    events: tuple[DividendEvent, ...]
    errors: tuple[dict[str, object], ...]

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class DividendCrosscheckResult:
    """Exact-key checks against both immutable source bar datasets."""

    dividend_missing_adjusted_bar_count: int
    dividend_missing_phase4_bar_count: int
    errors: tuple[dict[str, object], ...]

    @property
    def valid(self) -> bool:
        return not self.errors


class DividendEventDatasetValidationError(ValueError):
    """One or more canonical DividendEvent invariants failed."""

    def __init__(
        self,
        errors: Sequence[dict[str, object]],
        *,
        duplicate_count: int = 0,
    ) -> None:
        self.errors = tuple(errors)
        self.duplicate_count = duplicate_count
        super().__init__(
            "; ".join(str(error["reason"]) for error in self.errors)
            or "DividendEvent dataset validation failed"
        )


def _context(
    stage: str,
    reason: str,
    record: Mapping[str, object] | None = None,
) -> dict[str, object]:
    error: dict[str, object] = {"stage": stage, "reason": reason}
    if record is None:
        return error
    asset_id = record.get("provider_asset_id")
    if isinstance(asset_id, int) and not isinstance(asset_id, bool):
        error["provider_asset_id"] = asset_id
        if asset_id > 0:
            error["security_id"] = f"NORGATE:{asset_id}"
    trading_date = record.get("trading_date")
    if type(trading_date) is date:
        error["entitlement_date"] = trading_date.isoformat()
    return error


def _raw_records(
    rows: pa.Table | pd.DataFrame | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(rows, pa.Table):
        actual_columns = rows.column_names
        records = rows.to_pylist()
    elif isinstance(rows, pd.DataFrame):
        actual_columns = list(rows.columns)
        records = rows.to_dict(orient="records")
    else:
        records = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise DividendEventDatasetValidationError(
                    [_context("dividend_source_validation", "raw rows must be mappings")]
                )
            records.append(dict(row))
        actual_columns = list(records[0]) if records else list(DIVIDEND_RAW_REQUIRED_COLUMNS)
        expected = set(actual_columns)
        if any(set(record) != expected for record in records[1:]):
            raise DividendEventDatasetValidationError(
                [
                    _context(
                        "dividend_source_validation",
                        "raw dividend rows do not have a consistent field set",
                    )
                ]
            )

    missing = [
        column for column in DIVIDEND_RAW_REQUIRED_COLUMNS if column not in actual_columns
    ]
    if missing:
        raise DividendEventDatasetValidationError(
            [
                _context(
                    "dividend_source_validation",
                    "raw adjusted data is missing columns: " + ", ".join(missing),
                )
            ]
        )
    return records


def _positive_dividend(value: object) -> float | None:
    """Return a positive float, with Arrow null kept distinct from NaN."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Dividend must be numeric or an Arrow null")
    if not math.isfinite(value):
        raise ValueError("Dividend must be finite when non-null")
    if value < 0:
        raise ValueError("Dividend must not be negative")
    if value == 0:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Dividend must fit finite float64 storage")
    return numeric


def check_dividend_source_rows(
    rows: pa.Table | pd.DataFrame | Sequence[Mapping[str, object]],
) -> DividendDerivationResult:
    """Map only explicit positive raw Dividend values through DividendEvent.

    An Arrow validity-bit null becomes ``None`` and creates no event. A
    non-null NaN remains ``float('nan')`` and is rejected.
    """
    records = _raw_records(rows)
    events: list[DividendEvent] = []
    errors: list[dict[str, object]] = []
    nonzero_count = 0
    invalid_count = 0
    for record in records:
        try:
            amount = _positive_dividend(record["Dividend"])
        except (TypeError, ValueError, OverflowError) as error:
            invalid_count += 1
            errors.append(_context("dividend_source_validation", str(error), record))
            continue
        if amount is None:
            continue
        nonzero_count += 1
        try:
            asset_id = validate_asset_id(record["provider_asset_id"])
            symbol = validate_provider_symbol(
                record["provider_symbol"], asset_id=asset_id
            )
            raw_date = record["trading_date"]
            if type(raw_date) is not date:
                raise ValueError("trading_date must be a Python date")
            entitlement_date = date(raw_date.year, raw_date.month, raw_date.day)
            events.append(
                DividendEvent(
                    security_id=security_id_for_asset_id(asset_id),
                    symbol=symbol,
                    entitlement_date=entitlement_date,
                    date_semantics="entitlement_close",
                    dividend_type="ordinary_cash",
                    amount_per_share=amount,
                    currency="USD",
                    source_provider="Norgate Data",
                    source_asset_id=asset_id,
                    source_adjustment_mode="CAPITALSPECIAL",
                )
            )
        except (TypeError, ValueError, ValidationError) as error:
            invalid_count += 1
            errors.append(_context("dividend_model_validation", str(error), record))
    return DividendDerivationResult(
        source_raw_row_count=len(records),
        raw_dividend_nonzero_count=nonzero_count,
        dividend_invalid_count=invalid_count,
        events=tuple(events),
        errors=tuple(errors),
    )


def _canonical_records(
    rows: Sequence[DividendEvent | Mapping[str, object]] | pd.DataFrame,
) -> list[dict[str, object]]:
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict(orient="records")
        actual_fields = list(rows.columns)
    else:
        records = []
        for row in rows:
            if isinstance(row, DividendEvent):
                records.append(row.model_dump(mode="python"))
            elif isinstance(row, Mapping):
                records.append(dict(row))
            else:
                raise DividendEventDatasetValidationError(
                    [
                        _context(
                            "dividend_model_validation",
                            "canonical rows must be DividendEvent or mapping values",
                        )
                    ]
                )
        actual_fields = list(records[0]) if records else list(DIVIDEND_EVENT_FIELDS)
        expected = set(actual_fields)
        if any(set(record) != expected for record in records[1:]):
            raise DividendEventDatasetValidationError(
                [
                    _context(
                        "dividend_model_validation",
                        "canonical rows do not have a consistent field set",
                    )
                ]
            )
    missing = [field for field in DIVIDEND_EVENT_FIELDS if field not in actual_fields]
    extra = [field for field in actual_fields if field not in DIVIDEND_EVENT_FIELDS]
    if missing or extra:
        reasons: list[str] = []
        if missing:
            reasons.append("missing canonical fields: " + ", ".join(missing))
        if extra:
            reasons.append("unexpected canonical fields: " + ", ".join(extra))
        raise DividendEventDatasetValidationError(
            [_context("dividend_model_validation", "; ".join(reasons))]
        )
    return [{field: record[field] for field in DIVIDEND_EVENT_FIELDS} for record in records]


def validate_dividend_event_dataset(
    rows: Sequence[DividendEvent | Mapping[str, object]] | pd.DataFrame,
) -> list[DividendEvent]:
    """Reconstruct, uniquely validate, and deterministically sort events."""
    records = _canonical_records(rows)
    events: list[DividendEvent] = []
    for record in records:
        try:
            events.append(DividendEvent(**record))
        except ValidationError as error:
            raise DividendEventDatasetValidationError(
                [_context("dividend_model_validation", f"invalid DividendEvent: {error}")]
            ) from error
    events.sort(key=lambda event: (event.security_id, event.entitlement_date))
    keys: set[tuple[object, ...]] = set()
    duplicate_errors: list[dict[str, object]] = []
    for event in events:
        values = event.model_dump(mode="python")
        key = tuple(values[field] for field in DIVIDEND_CANONICAL_KEY)
        if key in keys:
            duplicate_errors.append(
                {
                    "provider_asset_id": event.source_asset_id,
                    "security_id": event.security_id,
                    "entitlement_date": event.entitlement_date.isoformat(),
                    "stage": "dividend_uniqueness",
                    "reason": f"duplicate canonical DividendEvent key: {key}",
                }
            )
        keys.add(key)
    if duplicate_errors:
        raise DividendEventDatasetValidationError(
            duplicate_errors, duplicate_count=len(duplicate_errors)
        )
    return events


def _bar_index(
    bars: Sequence[CorporateActionAdjustedStockBar] | Sequence[StockBar],
) -> dict[tuple[str, date], list[Any]]:
    result: dict[tuple[str, date], list[Any]] = {}
    for bar in bars:
        result.setdefault((bar.security_id, bar.trading_date), []).append(bar)
    return result


def check_dividend_event_crosschecks(
    events: Sequence[DividendEvent],
    adjusted_bars: Sequence[CorporateActionAdjustedStockBar],
    phase4_bars: Sequence[StockBar],
) -> DividendCrosscheckResult:
    """Require exactly one same-date, same-symbol bar in each source dataset."""
    adjusted = _bar_index(adjusted_bars)
    phase4 = _bar_index(phase4_bars)
    errors: list[dict[str, object]] = []
    missing_adjusted = 0
    missing_phase4 = 0
    for event in events:
        context = {
            "provider_asset_id": event.source_asset_id,
            "security_id": event.security_id,
            "entitlement_date": event.entitlement_date.isoformat(),
        }
        key = (event.security_id, event.entitlement_date)
        adjusted_matches = adjusted.get(key, [])
        if len(adjusted_matches) != 1:
            missing_adjusted += 1
            errors.append(
                {
                    **context,
                    "stage": "adjusted_bar_crosscheck",
                    "reason": "dividend must have exactly one same-date adjusted bar",
                }
            )
        elif adjusted_matches[0].symbol != event.symbol:
            errors.append(
                {
                    **context,
                    "stage": "adjusted_bar_crosscheck",
                    "reason": "dividend symbol does not match adjusted bar symbol",
                }
            )

        phase4_matches = phase4.get(key, [])
        if len(phase4_matches) != 1:
            missing_phase4 += 1
            errors.append(
                {
                    **context,
                    "stage": "phase4_bar_crosscheck",
                    "reason": "dividend must have exactly one same-date Phase 4 bar",
                }
            )
        elif phase4_matches[0].symbol != event.symbol:
            errors.append(
                {
                    **context,
                    "stage": "phase4_bar_crosscheck",
                    "reason": "dividend symbol does not match Phase 4 bar symbol",
                }
            )
    return DividendCrosscheckResult(
        dividend_missing_adjusted_bar_count=missing_adjusted,
        dividend_missing_phase4_bar_count=missing_phase4,
        errors=tuple(errors),
    )


__all__ = [
    "DIVIDEND_CANONICAL_KEY",
    "DIVIDEND_EVENT_FIELDS",
    "DIVIDEND_RAW_REQUIRED_COLUMNS",
    "DividendCrosscheckResult",
    "DividendDerivationResult",
    "DividendEventDatasetValidationError",
    "check_dividend_event_crosschecks",
    "check_dividend_source_rows",
    "validate_dividend_event_dataset",
]
