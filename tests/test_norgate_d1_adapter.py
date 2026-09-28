# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
from datetime import date, datetime, timezone
from math import inf, nan
import pandas as pd
import pytest
from pydantic import ValidationError

from stock_swing_d1.data.norgate_d1_adapter import (
    PROVIDER_COLUMNS,
    ProviderValidationError,
    fetch_norgate_d1,
    map_norgate_raw_frame,
    prepare_norgate_raw_frame,
    provider_date_to_date,
    provider_row_to_stock_bar,
    resolve_norgate_symbol,
    security_id_for_asset_id,
    validate_asset_id,
    validate_norgate_raw_frame,
)


def provider_frame(
    *,
    dates: tuple[str, ...] = ("2026-08-07",),
    **overrides: object,
) -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "Open": [100.0] * len(dates),
        "High": [102.0] * len(dates),
        "Low": [99.0] * len(dates),
        "Close": [101.0] * len(dates),
        "Volume": [12_345_678.0] * len(dates),
        "Turnover": [None] * len(dates),
        "Unadjusted Close": [101.0] * len(dates),
        "Dividend": [None] * len(dates),
    }
    for field, value in overrides.items():
        values[field] = value if isinstance(value, list) else [value] * len(dates)
    return pd.DataFrame(values, index=pd.to_datetime(list(dates)))


class FakeProvider:
    class StockPriceAdjustmentType:
        NONE = object()

    class PaddingType:
        NONE = object()

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def price_timeseries(self, *args: object, **kwargs: object) -> pd.DataFrame:
        self.calls.append((args, kwargs))
        return provider_frame()

    def symbol(self, asset_id: int) -> str:
        assert asset_id == 1_900_000_001
        return "syntha"


def test_fetch_uses_explicit_frozen_norgate_settings() -> None:
    provider = FakeProvider()
    start = date(2026, 1, 1)
    end = date(2026, 1, 31)

    result = fetch_norgate_d1(1_900_000_001, start, end, provider=provider)

    assert isinstance(result, pd.DataFrame)
    args, kwargs = provider.calls[0]
    assert args == (1_900_000_001,)
    assert kwargs == {
        "start_date": start,
        "end_date": end,
        "stock_price_adjustment_setting": provider.StockPriceAdjustmentType.NONE,
        "padding_setting": provider.PaddingType.NONE,
        "interval": "D",
        "timeseriesformat": "pandas-dataframe",
    }


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.0, "1900000001"])
def test_asset_id_validation_rejects_invalid_values(invalid: object) -> None:
    with pytest.raises(ProviderValidationError):
        validate_asset_id(invalid)


def test_security_id_mapping_uses_asset_id() -> None:
    assert security_id_for_asset_id(1_900_000_001) == "NORGATE:1900000001"


def test_provider_symbol_is_resolved_from_asset_id() -> None:
    assert resolve_norgate_symbol(1_900_000_001, provider=FakeProvider()) == "syntha"


@pytest.mark.parametrize("symbol", ["", "   ", None, 123])
def test_invalid_provider_symbol_is_rejected(symbol: object) -> None:
    frame = provider_frame()
    with pytest.raises(ProviderValidationError, match="non-empty"):
        prepare_norgate_raw_frame(1_900_000_001, symbol, frame)  # type: ignore[arg-type]


def test_timestamp_is_deliberately_converted_to_python_date() -> None:
    result = provider_date_to_date(pd.Timestamp("2026-08-07"))

    assert result == date(2026, 8, 7)
    assert type(result) is date


@pytest.mark.parametrize(
    "invalid",
    [
        pd.Timestamp("2026-08-07 12:00:00"),
        pd.Timestamp("2026-08-07", tz="UTC"),
        datetime(2026, 8, 7, 12),
        datetime(2026, 8, 7, tzinfo=timezone.utc),
        pd.NaT,
        1_786_060_800,
    ],
)
def test_ambiguous_or_non_calendar_provider_dates_are_rejected(
    invalid: object,
) -> None:
    with pytest.raises(ProviderValidationError):
        provider_date_to_date(invalid)


def test_all_required_provider_columns_are_required() -> None:
    frame = provider_frame().drop(columns="Dividend")

    with pytest.raises(ProviderValidationError, match="Dividend"):
        prepare_norgate_raw_frame(1_900_000_001, "SYNTHA", frame)


def test_provider_rows_are_sorted_deterministically() -> None:
    raw = prepare_norgate_raw_frame(
        1_900_000_001,
        "SYNTHA",
        provider_frame(dates=("2026-08-08", "2026-08-07")),
    )

    assert raw["trading_date"].tolist() == [date(2026, 8, 7), date(2026, 8, 8)]
    assert list(raw.columns) == [
        "provider_asset_id",
        "provider_symbol",
        "trading_date",
        *PROVIDER_COLUMNS,
    ]


def test_duplicate_provider_dates_are_rejected() -> None:
    raw = prepare_norgate_raw_frame(
        1_900_000_001,
        "SYNTHA",
        provider_frame(dates=("2026-08-07", "2026-08-07")),
    )

    with pytest.raises(ProviderValidationError, match="duplicate"):
        validate_norgate_raw_frame(raw)


def test_empty_provider_frame_is_a_reportable_no_observations_failure() -> None:
    raw = prepare_norgate_raw_frame(
        1_900_000_001, "SYNTHA", provider_frame(dates=())
    )

    with pytest.raises(ProviderValidationError, match="no observations"):
        map_norgate_raw_frame(raw)


def test_close_consistency_accepts_values_within_frozen_tolerance() -> None:
    raw = prepare_norgate_raw_frame(
        1_900_000_001,
        "SYNTHA",
        provider_frame(**{"Unadjusted Close": 101.0 + 1e-8}),
    )

    [bar] = map_norgate_raw_frame(raw)

    assert bar.close == pytest.approx(101.0 + 1e-8)


def test_close_consistency_rejects_mismatch_beyond_tolerance() -> None:
    raw = prepare_norgate_raw_frame(
        1_900_000_001,
        "SYNTHA",
        provider_frame(**{"Unadjusted Close": 100.5}),
    )

    with pytest.raises(ProviderValidationError, match="differ beyond tolerance"):
        map_norgate_raw_frame(raw)


@pytest.mark.parametrize("volume", [12_345_678, 12_345_678.0])
def test_integer_and_integer_valued_float_volume_are_converted_to_int(
    volume: object,
) -> None:
    raw = prepare_norgate_raw_frame(
        1_900_000_001, "SYNTHA", provider_frame(Volume=volume)
    )

    [bar] = map_norgate_raw_frame(raw)

    assert bar.volume == 12_345_678
    assert type(bar.volume) is int


@pytest.mark.parametrize("volume", [12_345_678.5, -1, nan, inf, -inf, True])
def test_invalid_volume_is_rejected_without_rounding(volume: object) -> None:
    raw = prepare_norgate_raw_frame(
        1_900_000_001, "SYNTHA", provider_frame(Volume=volume)
    )

    with pytest.raises(ProviderValidationError, match="Volume"):
        map_norgate_raw_frame(raw)


@pytest.mark.parametrize("field", ["Open", "High", "Low", "Close", "Unadjusted Close"])
@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_nonfinite_required_price_is_rejected(field: str, value: float) -> None:
    raw = prepare_norgate_raw_frame(
        1_900_000_001, "SYNTHA", provider_frame(**{field: value})
    )

    with pytest.raises(ProviderValidationError, match="finite"):
        map_norgate_raw_frame(raw)


def test_exact_stock_bar_mapping_uses_unadjusted_close() -> None:
    row = prepare_norgate_raw_frame(
        1_900_000_001,
        "syntha",
        provider_frame(Turnover=123.5, Dividend=0.25),
    ).to_dict(orient="records")[0]

    bar = provider_row_to_stock_bar(row)

    assert bar.model_dump() == {
        "security_id": "NORGATE:1900000001",
        "symbol": "SYNTHA",
        "trading_date": date(2026, 8, 7),
        "timeframe": "D1",
        "session_type": "regular",
        "currency": "USD",
        "price_basis": "unadjusted",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 12_345_678,
    }
    assert "Turnover" not in type(bar).model_fields
    assert "Dividend" not in type(bar).model_fields


@pytest.mark.parametrize(
    "missing_field",
    ["Open", "High", "Low", "Close", "Unadjusted Close", "Volume"],
)
def test_missing_required_provider_row_value_is_provider_validation_error(
    missing_field: str,
) -> None:
    row = prepare_norgate_raw_frame(
        1_900_000_001, "SYNTHA", provider_frame()
    ).to_dict(orient="records")[0]
    del row[missing_field]

    with pytest.raises(
        ProviderValidationError,
        match=f"missing required raw value: {missing_field}",
    ):
        provider_row_to_stock_bar(row)


def test_stock_bar_validation_remains_authoritative_for_ohlc_relationships() -> None:
    raw = prepare_norgate_raw_frame(
        1_900_000_001, "SYNTHA", provider_frame(Open=103.0)
    )

    with pytest.raises(ValidationError):
        map_norgate_raw_frame(raw)
