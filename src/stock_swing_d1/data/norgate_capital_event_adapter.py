"""Norgate binary capital-event adapter for Phase 5B."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from importlib import import_module
from typing import Any

import pandas as pd

from stock_swing_d1.data.norgate_d1_adapter import (
    provider_date_to_date,
    security_id_for_asset_id,
    validate_asset_id,
    validate_date_range,
    validate_provider_symbol,
)
from stock_swing_d1.models import CorporateActionEvent


EVENT_RAW_COLUMNS = (
    "provider_asset_id",
    "provider_symbol",
    "event_date",
    "capital_event_flag",
)


class CapitalEventProviderValidationError(ValueError):
    """A capital-event provider response violated the Phase 5B contract."""

    def __init__(
        self,
        reason: str,
        *,
        provider_asset_id: int | None = None,
        event_date: date | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.provider_asset_id = provider_asset_id
        self.event_date = event_date


def _provider_module(provider: Any | None) -> Any:
    return provider if provider is not None else import_module("norgatedata")


def fetch_norgate_capital_events(
    asset_id: int, *, provider: Any | None = None
) -> pd.DataFrame:
    """Fetch the documented event series without invented range arguments."""
    normalized_asset_id = validate_asset_id(asset_id)
    norgatedata = _provider_module(provider)
    return norgatedata.capital_event_timeseries(
        normalized_asset_id,
        timeseriesformat="pandas-dataframe",
    )


def _capital_event_flag(
    value: object, *, asset_id: int, event_date: date
) -> int:
    if pd.api.types.is_bool(value) or isinstance(value, (str, bytes)) or value is None:
        raise CapitalEventProviderValidationError(
            "capital event flag must be exactly 0 or 1",
            provider_asset_id=asset_id,
            event_date=event_date,
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CapitalEventProviderValidationError(
            "capital event flag must be exactly 0 or 1",
            provider_asset_id=asset_id,
            event_date=event_date,
        ) from error
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric not in (0, 1):
        raise CapitalEventProviderValidationError(
            "capital event flag must be exactly 0 or 1",
            provider_asset_id=asset_id,
            event_date=event_date,
        )
    return int(numeric)


def prepare_norgate_capital_event_raw_frame(
    asset_id: int,
    provider_symbol: str,
    frame: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Resolve one indicator series, validate it, and filter dates locally."""
    normalized_asset_id = validate_asset_id(asset_id)
    normalized_start, normalized_end = validate_date_range(start_date, end_date)
    try:
        normalized_symbol = validate_provider_symbol(
            provider_symbol, asset_id=normalized_asset_id
        )
    except ValueError as error:
        raise CapitalEventProviderValidationError(
            str(error), provider_asset_id=normalized_asset_id
        ) from error
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise CapitalEventProviderValidationError(
            "capital-event result must be a pandas DataFrame",
            provider_asset_id=normalized_asset_id,
        )
    if frame.columns.duplicated().any():
        raise CapitalEventProviderValidationError(
            "capital-event response contains duplicate value columns",
            provider_asset_id=normalized_asset_id,
        )
    if len(frame.columns) != 1:
        reason = (
            "capital-event response has no usable event-value column"
            if len(frame.columns) == 0
            else "capital-event response has ambiguous event-value columns"
        )
        raise CapitalEventProviderValidationError(
            reason, provider_asset_id=normalized_asset_id
        )

    normalized_rows: list[dict[str, object]] = []
    value_column = frame.columns[0]
    for index_value, indicator in zip(frame.index, frame[value_column]):
        try:
            event_date = provider_date_to_date(index_value)
        except ValueError as error:
            raise CapitalEventProviderValidationError(
                "capital-event date index is malformed",
                provider_asset_id=normalized_asset_id,
            ) from error
        if not normalized_start <= event_date <= normalized_end:
            continue
        normalized_rows.append(
            {
                "provider_asset_id": normalized_asset_id,
                "provider_symbol": normalized_symbol,
                "event_date": event_date,
                "capital_event_flag": _capital_event_flag(
                    indicator,
                    asset_id=normalized_asset_id,
                    event_date=event_date,
                ),
            }
        )

    raw = pd.DataFrame.from_records(
        normalized_rows, columns=list(EVENT_RAW_COLUMNS)
    ).sort_values(
        ["provider_asset_id", "event_date"],
        kind="mergesort",
        ignore_index=True,
    )
    duplicate_mask = raw.duplicated(
        subset=["provider_asset_id", "event_date"], keep=False
    )
    if duplicate_mask.any():
        duplicate_date = raw.loc[duplicate_mask, "event_date"].iloc[0]
        raise CapitalEventProviderValidationError(
            "duplicate capital-event date",
            provider_asset_id=normalized_asset_id,
            event_date=duplicate_date,
        )
    return raw


def capital_event_row_to_model(row: Mapping[str, object]) -> CorporateActionEvent:
    """Map one validated flag==1 row through CorporateActionEvent."""
    try:
        asset_id = validate_asset_id(row["provider_asset_id"])
        symbol = validate_provider_symbol(row["provider_symbol"], asset_id=asset_id)
        event_date = provider_date_to_date(row["event_date"])
        flag = _capital_event_flag(
            row["capital_event_flag"], asset_id=asset_id, event_date=event_date
        )
    except KeyError as error:
        raise CapitalEventProviderValidationError(
            f"missing required raw event value: {error.args[0]}"
        ) from error
    except ValueError as error:
        if isinstance(error, CapitalEventProviderValidationError):
            raise
        raise CapitalEventProviderValidationError(
            str(error), provider_asset_id=asset_id if "asset_id" in locals() else None
        ) from error
    if flag != 1:
        raise CapitalEventProviderValidationError(
            "only capital_event_flag == 1 maps to CorporateActionEvent",
            provider_asset_id=asset_id,
            event_date=event_date,
        )
    return CorporateActionEvent(
        security_id=security_id_for_asset_id(asset_id),
        symbol=symbol,
        event_date=event_date,
        date_semantics="entitlement_close",
        event_type="unknown_capital_event",
        terms_verified=False,
        new_shares=None,
        old_shares=None,
        source_provider="Norgate Data",
        source_asset_id=asset_id,
    )


def map_norgate_capital_event_raw_frame(
    raw_frame: pd.DataFrame,
) -> list[CorporateActionEvent]:
    """Validate every flag, then map ones while zeroes emit nothing."""
    if raw_frame is None or not isinstance(raw_frame, pd.DataFrame):
        raise CapitalEventProviderValidationError(
            "raw capital-event data must be a DataFrame"
        )
    if list(raw_frame.columns) != list(EVENT_RAW_COLUMNS):
        raise CapitalEventProviderValidationError(
            "raw capital-event frame must contain exactly the frozen columns in order"
        )
    events: list[CorporateActionEvent] = []
    for row in raw_frame.to_dict(orient="records"):
        flag = _capital_event_flag(
            row["capital_event_flag"],
            asset_id=row["provider_asset_id"],
            event_date=row["event_date"],
        )
        if flag == 1:
            events.append(capital_event_row_to_model(row))
    return events


__all__ = [
    "EVENT_RAW_COLUMNS",
    "CapitalEventProviderValidationError",
    "capital_event_row_to_model",
    "fetch_norgate_capital_events",
    "map_norgate_capital_event_raw_frame",
    "prepare_norgate_capital_event_raw_frame",
]
