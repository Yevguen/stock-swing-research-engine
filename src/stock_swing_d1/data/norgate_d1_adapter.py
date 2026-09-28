"""Norgate Data adapter for raw, unadjusted daily price observations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, time
from importlib import import_module
from numbers import Integral
from typing import Any

import pandas as pd

from stock_swing_d1.models import StockBar


PROVIDER_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Turnover",
    "Unadjusted Close",
    "Dividend",
)
RAW_COLUMNS = (
    "provider_asset_id",
    "provider_symbol",
    "trading_date",
    *PROVIDER_COLUMNS,
)
MAX_INT64 = 2**63 - 1


class ProviderValidationError(ValueError):
    """A Norgate extraction failed provider-level validation."""

    def __init__(
        self,
        reason: str,
        *,
        provider_asset_id: int | None = None,
        trading_date: date | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.provider_asset_id = provider_asset_id
        self.trading_date = trading_date


def validate_asset_id(asset_id: object) -> int:
    """Return a positive Norgate AssetId that fits the raw int64 schema."""
    if isinstance(asset_id, bool) or not isinstance(asset_id, Integral):
        raise ProviderValidationError(
            "AssetId must be a positive integer and not bool"
        )
    normalized = int(asset_id)
    if normalized <= 0 or normalized > MAX_INT64:
        raise ProviderValidationError(
            "AssetId must be between 1 and the int64 maximum"
        )
    return normalized


def validate_date_range(start_date: object, end_date: object) -> tuple[date, date]:
    """Validate an explicit inclusive calendar-date range."""
    if type(start_date) is not date or type(end_date) is not date:
        raise ValueError("start_date and end_date must be Python date values")
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    return start_date, end_date


def security_id_for_asset_id(asset_id: object) -> str:
    """Map the authoritative Norgate AssetId to canonical project identity."""
    return f"NORGATE:{validate_asset_id(asset_id)}"


def _provider_module(provider: Any | None) -> Any:
    return provider if provider is not None else import_module("norgatedata")


def fetch_norgate_d1(
    asset_id: int,
    start_date: date,
    end_date: date,
    *,
    provider: Any | None = None,
) -> pd.DataFrame:
    """Fetch one explicit AssetId with unadjusted prices and no padding.

    Provider, network, and NDU exceptions intentionally propagate unchanged.
    The function does not persist or validate the returned observations.
    """
    normalized_asset_id = validate_asset_id(asset_id)
    normalized_start, normalized_end = validate_date_range(start_date, end_date)
    norgatedata = _provider_module(provider)
    return norgatedata.price_timeseries(
        normalized_asset_id,
        start_date=normalized_start,
        end_date=normalized_end,
        stock_price_adjustment_setting=(
            norgatedata.StockPriceAdjustmentType.NONE
        ),
        padding_setting=norgatedata.PaddingType.NONE,
        interval="D",
        timeseriesformat="pandas-dataframe",
    )


def resolve_norgate_symbol(
    asset_id: int, *, provider: Any | None = None
) -> str:
    """Resolve informational provider symbol metadata for an AssetId."""
    normalized_asset_id = validate_asset_id(asset_id)
    norgatedata = _provider_module(provider)
    symbol = norgatedata.symbol(normalized_asset_id)
    return validate_provider_symbol(symbol, asset_id=normalized_asset_id)


def validate_provider_symbol(symbol: object, *, asset_id: int) -> str:
    """Return a non-empty provider symbol without making it an identity key."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise ProviderValidationError(
            "provider symbol must resolve to a non-empty string",
            provider_asset_id=asset_id,
        )
    return symbol.strip()


def provider_date_to_date(value: object) -> date:
    """Convert a provider index value to an unambiguous calendar date."""
    if type(value) is date:
        return value

    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ProviderValidationError(
                "provider trading date is not an ISO calendar date"
            ) from error
        if parsed.isoformat() != value:
            raise ProviderValidationError(
                "provider trading date is not an unambiguous calendar date"
            )
        return parsed

    if isinstance(value, (bool, int, float)) or value is None:
        raise ProviderValidationError(
            "provider trading date is not an unambiguous calendar date"
        )

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProviderValidationError(
            "provider trading date is not an unambiguous calendar date"
        ) from error

    if pd.isna(timestamp):
        raise ProviderValidationError("provider trading date must not be null")
    if timestamp.tzinfo is not None or timestamp.time() != time.min:
        raise ProviderValidationError(
            "provider trading date must be timezone-naive and at midnight"
        )
    return timestamp.date()


def prepare_norgate_raw_frame(
    asset_id: int,
    provider_symbol: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Make provider dates and metadata explicit for an immutable raw snapshot.

    The authoritative identity is ``security_id`` derived from AssetId. Symbol
    remains informational provider metadata and may change over time.
    """
    normalized_asset_id = validate_asset_id(asset_id)
    normalized_symbol = validate_provider_symbol(
        provider_symbol, asset_id=normalized_asset_id
    )
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise ProviderValidationError(
            "provider result must be a pandas DataFrame",
            provider_asset_id=normalized_asset_id,
        )

    missing = [column for column in PROVIDER_COLUMNS if column not in frame.columns]
    if missing:
        raise ProviderValidationError(
            f"missing required provider columns: {', '.join(missing)}",
            provider_asset_id=normalized_asset_id,
        )
    duplicated_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ProviderValidationError(
            "duplicate provider columns: "
            + ", ".join(str(column) for column in duplicated_columns),
            provider_asset_id=normalized_asset_id,
        )

    trading_dates: list[date] = []
    for value in frame.index:
        try:
            trading_dates.append(provider_date_to_date(value))
        except ProviderValidationError as error:
            raise ProviderValidationError(
                error.reason,
                provider_asset_id=normalized_asset_id,
            ) from error

    raw = frame.loc[:, list(PROVIDER_COLUMNS)].copy()
    raw.insert(0, "trading_date", trading_dates)
    raw.insert(0, "provider_symbol", normalized_symbol)
    raw.insert(0, "provider_asset_id", normalized_asset_id)
    return raw.loc[:, list(RAW_COLUMNS)].sort_values(
        ["trading_date"], kind="mergesort", ignore_index=True
    )


def validate_norgate_raw_frame(raw_frame: pd.DataFrame) -> None:
    """Validate frame-level provider constraints before row mapping."""
    if raw_frame is None or not isinstance(raw_frame, pd.DataFrame):
        raise ProviderValidationError("raw provider data must be a DataFrame")
    missing = [column for column in RAW_COLUMNS if column not in raw_frame.columns]
    if missing:
        raise ProviderValidationError(
            f"missing required raw columns: {', '.join(missing)}"
        )
    if raw_frame.empty:
        asset_id = None
        if "provider_asset_id" in raw_frame and len(raw_frame.index):
            asset_id = int(raw_frame["provider_asset_id"].iloc[0])
        raise ProviderValidationError(
            "no observations returned", provider_asset_id=asset_id
        )

    asset_ids = raw_frame["provider_asset_id"].drop_duplicates().tolist()
    if len(asset_ids) != 1:
        raise ProviderValidationError(
            "one provider frame must contain exactly one AssetId"
        )
    asset_id = validate_asset_id(asset_ids[0])
    symbols = raw_frame["provider_symbol"].drop_duplicates().tolist()
    if len(symbols) != 1:
        raise ProviderValidationError(
            "one provider frame must contain exactly one provider symbol",
            provider_asset_id=asset_id,
        )
    validate_provider_symbol(symbols[0], asset_id=asset_id)

    duplicate_mask = raw_frame.duplicated(subset=["trading_date"], keep=False)
    if duplicate_mask.any():
        duplicate_date = raw_frame.loc[duplicate_mask, "trading_date"].iloc[0]
        raise ProviderValidationError(
            "duplicate provider trading date",
            provider_asset_id=asset_id,
            trading_date=duplicate_date,
        )

    expected_order = raw_frame.sort_values(
        ["trading_date"], kind="mergesort", ignore_index=True
    )
    if raw_frame["trading_date"].tolist() != expected_order["trading_date"].tolist():
        raise ProviderValidationError(
            "provider rows are not sorted by trading_date",
            provider_asset_id=asset_id,
        )


def _finite_price(
    value: object,
    field_name: str,
    *,
    asset_id: int,
    trading_date: date,
) -> float:
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise ProviderValidationError(
            f"{field_name} must be a finite numeric value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProviderValidationError(
            f"{field_name} must be a finite numeric value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        ) from error
    if not math.isfinite(normalized):
        raise ProviderValidationError(
            f"{field_name} must be a finite numeric value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    return normalized


def volume_to_int(
    value: object, *, asset_id: int, trading_date: date
) -> int:
    """Convert whole-share provider volume without rounding."""
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise ProviderValidationError(
            "Volume must be a finite, nonnegative whole-share value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProviderValidationError(
            "Volume must be a finite, nonnegative whole-share value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        ) from error
    if (
        not math.isfinite(numeric)
        or numeric < 0
        or not numeric.is_integer()
        or numeric > MAX_INT64
    ):
        raise ProviderValidationError(
            "Volume must be a finite, nonnegative whole-share int64 value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    return int(numeric)


def provider_row_to_stock_bar(row: Mapping[str, object]) -> StockBar:
    """Validate and map one raw Norgate row through the canonical StockBar."""
    try:
        asset_id = validate_asset_id(row["provider_asset_id"])
        provider_symbol = validate_provider_symbol(
            row["provider_symbol"], asset_id=asset_id
        )
        trading_date = provider_date_to_date(row["trading_date"])
    except KeyError as error:
        raise ProviderValidationError(
            f"missing required raw value: {error.args[0]}"
        ) from error

    try:
        prices = {
            field: _finite_price(
                row[field], field, asset_id=asset_id, trading_date=trading_date
            )
            for field in ("Open", "High", "Low", "Close", "Unadjusted Close")
        }
        raw_volume = row["Volume"]
    except KeyError as error:
        raise ProviderValidationError(
            f"missing required raw value: {error.args[0]}",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        ) from error
    if not math.isclose(
        prices["Close"],
        prices["Unadjusted Close"],
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        raise ProviderValidationError(
            "Close and Unadjusted Close differ beyond tolerance",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    volume = volume_to_int(
        raw_volume, asset_id=asset_id, trading_date=trading_date
    )

    # StockBar remains authoritative for positive prices and OHLC relationships.
    return StockBar(
        security_id=security_id_for_asset_id(asset_id),
        symbol=provider_symbol,
        trading_date=trading_date,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=prices["Open"],
        high=prices["High"],
        low=prices["Low"],
        close=prices["Unadjusted Close"],
        volume=volume,
    )


def map_norgate_raw_frame(raw_frame: pd.DataFrame) -> list[StockBar]:
    """Validate and map a deterministically ordered one-AssetId raw frame."""
    validate_norgate_raw_frame(raw_frame)
    bars: list[StockBar] = []
    for row in raw_frame.to_dict(orient="records"):
        bars.append(provider_row_to_stock_bar(row))
    return bars


__all__ = [
    "PROVIDER_COLUMNS",
    "RAW_COLUMNS",
    "ProviderValidationError",
    "fetch_norgate_d1",
    "map_norgate_raw_frame",
    "prepare_norgate_raw_frame",
    "provider_date_to_date",
    "provider_row_to_stock_bar",
    "resolve_norgate_symbol",
    "security_id_for_asset_id",
    "validate_asset_id",
    "validate_date_range",
    "validate_norgate_raw_frame",
    "validate_provider_symbol",
    "volume_to_int",
]
