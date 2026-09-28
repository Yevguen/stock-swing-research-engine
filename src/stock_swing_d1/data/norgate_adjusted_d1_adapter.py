"""Norgate CAPITALSPECIAL daily-price adapter for Phase 5B."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from importlib import import_module
from typing import Any

import pandas as pd

from stock_swing_d1.data.norgate_d1_adapter import (
    PROVIDER_COLUMNS,
    ProviderValidationError,
    provider_date_to_date,
    security_id_for_asset_id,
    validate_asset_id,
    validate_date_range,
    validate_provider_symbol,
)
from stock_swing_d1.models import CorporateActionAdjustedStockBar


ADJUSTED_RAW_COLUMNS = (
    "provider_asset_id",
    "provider_symbol",
    "trading_date",
    *PROVIDER_COLUMNS,
)


def _provider_module(provider: Any | None) -> Any:
    return provider if provider is not None else import_module("norgatedata")


def fetch_norgate_adjusted_d1(
    asset_id: int,
    start_date: date,
    end_date: date,
    *,
    provider: Any | None = None,
) -> pd.DataFrame:
    """Fetch one AssetId using the exact Phase 5B adjustment contract."""
    normalized_asset_id = validate_asset_id(asset_id)
    normalized_start, normalized_end = validate_date_range(start_date, end_date)
    norgatedata = _provider_module(provider)
    return norgatedata.price_timeseries(
        normalized_asset_id,
        start_date=normalized_start,
        end_date=normalized_end,
        interval="D",
        stock_price_adjustment_setting=(
            norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL
        ),
        padding_setting=norgatedata.PaddingType.NONE,
        timeseriesformat="pandas-dataframe",
    )


def _optional_float(
    value: object,
    field_name: str,
    *,
    asset_id: int,
    trading_date: date,
) -> float | None:
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    if pd.api.types.is_bool(value) or isinstance(value, (str, bytes)):
        raise ProviderValidationError(
            f"{field_name} must be numeric or null",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProviderValidationError(
            f"{field_name} must be numeric or null",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        ) from error
    if not math.isfinite(numeric):
        raise ProviderValidationError(
            f"{field_name} must be finite or null",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    return numeric


def prepare_norgate_adjusted_raw_frame(
    asset_id: int,
    provider_symbol: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize one provider response for immutable raw storage."""
    normalized_asset_id = validate_asset_id(asset_id)
    normalized_symbol = validate_provider_symbol(
        provider_symbol, asset_id=normalized_asset_id
    )
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise ProviderValidationError(
            "provider result must be a pandas DataFrame",
            provider_asset_id=normalized_asset_id,
        )
    duplicated_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ProviderValidationError(
            "duplicate provider columns: "
            + ", ".join(str(column) for column in duplicated_columns),
            provider_asset_id=normalized_asset_id,
        )
    missing = [column for column in PROVIDER_COLUMNS if column not in frame.columns]
    if missing:
        raise ProviderValidationError(
            f"missing required provider columns: {', '.join(missing)}",
            provider_asset_id=normalized_asset_id,
        )

    trading_dates: list[date] = []
    for value in frame.index:
        try:
            trading_dates.append(provider_date_to_date(value))
        except ProviderValidationError as error:
            raise ProviderValidationError(
                error.reason, provider_asset_id=normalized_asset_id
            ) from error

    raw = frame.loc[:, list(PROVIDER_COLUMNS)].copy()
    raw.insert(0, "trading_date", trading_dates)
    raw.insert(0, "provider_symbol", normalized_symbol)
    raw.insert(0, "provider_asset_id", normalized_asset_id)
    for column in ("Turnover", "Dividend"):
        raw[column] = [
            _optional_float(
                value,
                column,
                asset_id=normalized_asset_id,
                trading_date=trading_date,
            )
            for value, trading_date in zip(raw[column], raw["trading_date"])
        ]
    return raw.loc[:, list(ADJUSTED_RAW_COLUMNS)].sort_values(
        ["provider_asset_id", "trading_date"],
        kind="mergesort",
        ignore_index=True,
    )


def validate_norgate_adjusted_raw_frame(raw_frame: pd.DataFrame) -> None:
    """Validate frame structure and one-AssetId observation uniqueness."""
    if raw_frame is None or not isinstance(raw_frame, pd.DataFrame):
        raise ProviderValidationError("raw adjusted data must be a DataFrame")
    if list(raw_frame.columns) != list(ADJUSTED_RAW_COLUMNS):
        raise ProviderValidationError(
            "raw adjusted frame must contain exactly the frozen columns in order"
        )
    if raw_frame.empty:
        raise ProviderValidationError("no adjusted observations returned")

    asset_ids = raw_frame["provider_asset_id"].drop_duplicates().tolist()
    if len(asset_ids) != 1:
        raise ProviderValidationError(
            "one adjusted provider frame must contain exactly one AssetId"
        )
    asset_id = validate_asset_id(asset_ids[0])
    symbols = raw_frame["provider_symbol"].drop_duplicates().tolist()
    if len(symbols) != 1:
        raise ProviderValidationError(
            "one adjusted provider frame must contain exactly one provider symbol",
            provider_asset_id=asset_id,
        )
    validate_provider_symbol(symbols[0], asset_id=asset_id)

    duplicate_mask = raw_frame.duplicated(subset=["trading_date"], keep=False)
    if duplicate_mask.any():
        duplicate_date = raw_frame.loc[duplicate_mask, "trading_date"].iloc[0]
        raise ProviderValidationError(
            "duplicate adjusted provider trading date",
            provider_asset_id=asset_id,
            trading_date=(duplicate_date if type(duplicate_date) is date else None),
        )
    expected = raw_frame.sort_values(
        ["provider_asset_id", "trading_date"],
        kind="mergesort",
        ignore_index=True,
    )
    if raw_frame["trading_date"].tolist() != expected["trading_date"].tolist():
        raise ProviderValidationError(
            "adjusted provider rows are not sorted by trading_date",
            provider_asset_id=asset_id,
        )


def _positive_finite_price(
    value: object,
    field_name: str,
    *,
    asset_id: int,
    trading_date: date,
) -> float:
    if pd.api.types.is_bool(value) or isinstance(value, (str, bytes)) or value is None:
        raise ProviderValidationError(
            f"{field_name} must be a finite positive numeric value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProviderValidationError(
            f"{field_name} must be a finite positive numeric value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        ) from error
    if not math.isfinite(numeric) or numeric <= 0:
        raise ProviderValidationError(
            f"{field_name} must be a finite positive numeric value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    return numeric


def adjusted_volume_to_float(
    value: object, *, asset_id: int, trading_date: date
) -> float:
    """Convert adjusted volume to float without rounding fractional shares."""
    if pd.api.types.is_bool(value) or isinstance(value, (str, bytes)) or value is None:
        raise ProviderValidationError(
            "Volume must be a finite nonnegative numeric value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProviderValidationError(
            "Volume must be a finite nonnegative numeric value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        ) from error
    if not math.isfinite(numeric) or numeric < 0:
        raise ProviderValidationError(
            "Volume must be a finite nonnegative numeric value",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        )
    return numeric


def adjusted_provider_row_to_bar(
    row: Mapping[str, object],
) -> CorporateActionAdjustedStockBar:
    """Map one raw CAPITALSPECIAL row through the existing Pydantic model."""
    try:
        asset_id = validate_asset_id(row["provider_asset_id"])
        symbol = validate_provider_symbol(row["provider_symbol"], asset_id=asset_id)
        trading_date = provider_date_to_date(row["trading_date"])
    except KeyError as error:
        raise ProviderValidationError(
            f"missing required raw value: {error.args[0]}"
        ) from error

    prices: dict[str, float] = {}
    try:
        for field in ("Open", "High", "Low", "Close", "Unadjusted Close"):
            prices[field] = _positive_finite_price(
                row[field], field, asset_id=asset_id, trading_date=trading_date
            )
        volume = adjusted_volume_to_float(
            row["Volume"], asset_id=asset_id, trading_date=trading_date
        )
    except KeyError as error:
        raise ProviderValidationError(
            f"missing required raw value: {error.args[0]}",
            provider_asset_id=asset_id,
            trading_date=trading_date,
        ) from error

    return CorporateActionAdjustedStockBar(
        security_id=security_id_for_asset_id(asset_id),
        symbol=symbol,
        trading_date=trading_date,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="capital_special_adjusted",
        open=prices["Open"],
        high=prices["High"],
        low=prices["Low"],
        close=prices["Close"],
        volume=volume,
    )


def map_norgate_adjusted_raw_frame(
    raw_frame: pd.DataFrame,
) -> list[CorporateActionAdjustedStockBar]:
    """Validate and map one deterministically ordered adjusted frame."""
    validate_norgate_adjusted_raw_frame(raw_frame)
    return [
        adjusted_provider_row_to_bar(row)
        for row in raw_frame.to_dict(orient="records")
    ]


__all__ = [
    "ADJUSTED_RAW_COLUMNS",
    "adjusted_provider_row_to_bar",
    "adjusted_volume_to_float",
    "fetch_norgate_adjusted_d1",
    "map_norgate_adjusted_raw_frame",
    "prepare_norgate_adjusted_raw_frame",
    "validate_norgate_adjusted_raw_frame",
]
