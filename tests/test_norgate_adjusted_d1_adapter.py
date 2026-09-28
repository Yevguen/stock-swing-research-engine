# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
from datetime import date
from math import inf, nan

import pandas as pd
import pytest
from pydantic import ValidationError

from stock_swing_d1.data.norgate_adjusted_d1_adapter import (
    ADJUSTED_RAW_COLUMNS,
    adjusted_provider_row_to_bar,
    fetch_norgate_adjusted_d1,
    map_norgate_adjusted_raw_frame,
    prepare_norgate_adjusted_raw_frame,
    validate_norgate_adjusted_raw_frame,
)
from stock_swing_d1.data.norgate_d1_adapter import ProviderValidationError


def adjusted_frame(
    dates: tuple[str, ...] = ("2026-08-07",), **overrides: object
) -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "Open": [50.0] * len(dates),
        "High": [51.0] * len(dates),
        "Low": [49.0] * len(dates),
        "Close": [50.5] * len(dates),
        "Volume": [2_000_000.25] * len(dates),
        "Turnover": [None] * len(dates),
        "Unadjusted Close": [101.0] * len(dates),
        "Dividend": [None] * len(dates),
    }
    for field, value in overrides.items():
        values[field] = value if isinstance(value, list) else [value] * len(dates)
    return pd.DataFrame(values, index=pd.to_datetime(list(dates)))


class FakeProvider:
    class StockPriceAdjustmentType:
        CAPITALSPECIAL = object()

    class PaddingType:
        NONE = object()

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def price_timeseries(self, *args: object, **kwargs: object) -> pd.DataFrame:
        self.calls.append((args, kwargs))
        return adjusted_frame()


def test_adjusted_fetch_uses_exact_capitalspecial_contract() -> None:
    provider = FakeProvider()
    start = date(2026, 1, 1)
    end = date(2026, 1, 31)

    result = fetch_norgate_adjusted_d1(1_900_000_001, start, end, provider=provider)

    assert isinstance(result, pd.DataFrame)
    args, kwargs = provider.calls[0]
    assert args == (1_900_000_001,)
    assert kwargs == {
        "start_date": start,
        "end_date": end,
        "interval": "D",
        "stock_price_adjustment_setting": (
            provider.StockPriceAdjustmentType.CAPITALSPECIAL
        ),
        "padding_setting": provider.PaddingType.NONE,
        "timeseriesformat": "pandas-dataframe",
    }


@pytest.mark.parametrize("asset_id", [True, 0, -1, 1.0, "1"])
def test_adjusted_fetch_rejects_invalid_asset_ids(asset_id: object) -> None:
    with pytest.raises(ProviderValidationError):
        fetch_norgate_adjusted_d1(  # type: ignore[arg-type]
            asset_id, date(2026, 1, 1), date(2026, 1, 2), provider=FakeProvider()
        )


def test_adjusted_fetch_rejects_invalid_date_range() -> None:
    with pytest.raises(ValueError, match="on or before"):
        fetch_norgate_adjusted_d1(
            1, date(2026, 1, 2), date(2026, 1, 1), provider=FakeProvider()
        )


def test_adjusted_raw_normalizes_dates_symbol_and_order() -> None:
    raw = prepare_norgate_adjusted_raw_frame(
        1_900_000_001,
        "syntha",
        adjusted_frame(("2026-08-08", "2026-08-07")),
    )

    assert list(raw.columns) == list(ADJUSTED_RAW_COLUMNS)
    assert raw["provider_symbol"].tolist() == ["syntha", "syntha"]
    assert raw["trading_date"].tolist() == [date(2026, 8, 7), date(2026, 8, 8)]
    assert all(type(value) is date for value in raw["trading_date"])


def test_adjusted_raw_requires_dataframe_and_all_provider_columns() -> None:
    with pytest.raises(ProviderValidationError, match="DataFrame"):
        prepare_norgate_adjusted_raw_frame(1, "AAA", None)  # type: ignore[arg-type]
    with pytest.raises(ProviderValidationError, match="Dividend"):
        prepare_norgate_adjusted_raw_frame(
            1, "AAA", adjusted_frame().drop(columns="Dividend")
        )


def test_adjusted_raw_rejects_empty_symbol() -> None:
    with pytest.raises(ProviderValidationError, match="non-empty"):
        prepare_norgate_adjusted_raw_frame(1, " ", adjusted_frame())


def test_adjusted_duplicate_provider_dates_are_rejected() -> None:
    raw = prepare_norgate_adjusted_raw_frame(
        1, "AAA", adjusted_frame(("2026-08-07", "2026-08-07"))
    )

    with pytest.raises(ProviderValidationError, match="duplicate"):
        validate_norgate_adjusted_raw_frame(raw)


def test_empty_adjusted_response_is_rejected_when_bars_are_expected() -> None:
    raw = prepare_norgate_adjusted_raw_frame(1, "AAA", adjusted_frame(()))

    with pytest.raises(ProviderValidationError, match="no adjusted observations"):
        map_norgate_adjusted_raw_frame(raw)


@pytest.mark.parametrize("field", ["Open", "High", "Low", "Close", "Unadjusted Close"])
@pytest.mark.parametrize("value", [0, -1, nan, inf, -inf])
def test_adjusted_required_prices_must_be_positive_and_finite(
    field: str, value: float
) -> None:
    raw = prepare_norgate_adjusted_raw_frame(
        1, "AAA", adjusted_frame(**{field: value})
    )

    with pytest.raises(ProviderValidationError, match="positive"):
        map_norgate_adjusted_raw_frame(raw)


def test_fractional_adjusted_volume_is_preserved_without_rounding() -> None:
    raw = prepare_norgate_adjusted_raw_frame(
        1, "AAA", adjusted_frame(Volume=1234.56789)
    )

    [bar] = map_norgate_adjusted_raw_frame(raw)

    assert bar.volume == 1234.56789
    assert type(bar.volume) is float


@pytest.mark.parametrize("value", [-1, nan, inf, -inf, True, "12"])
def test_invalid_adjusted_volume_is_rejected(value: object) -> None:
    raw = prepare_norgate_adjusted_raw_frame(
        1, "AAA", adjusted_frame(Volume=value)
    )

    with pytest.raises(ProviderValidationError, match="Volume"):
        map_norgate_adjusted_raw_frame(raw)


def test_adjusted_mapping_uses_close_and_excludes_raw_diagnostics() -> None:
    row = prepare_norgate_adjusted_raw_frame(
        1_900_000_001,
        "syntha",
        adjusted_frame(
            Close=50.5,
            **{"Unadjusted Close": 101.0, "Turnover": 123.0, "Dividend": 0.25},
        ),
    ).to_dict(orient="records")[0]

    bar = adjusted_provider_row_to_bar(row)

    assert bar.model_dump() == {
        "security_id": "NORGATE:1900000001",
        "symbol": "SYNTHA",
        "trading_date": date(2026, 8, 7),
        "timeframe": "D1",
        "session_type": "regular",
        "currency": "USD",
        "price_basis": "capital_special_adjusted",
        "open": 50.0,
        "high": 51.0,
        "low": 49.0,
        "close": 50.5,
        "volume": 2_000_000.25,
    }
    assert "Unadjusted Close" not in type(bar).model_fields
    assert "Turnover" not in type(bar).model_fields
    assert "Dividend" not in type(bar).model_fields


def test_existing_adjusted_model_remains_authoritative_for_ohlc() -> None:
    raw = prepare_norgate_adjusted_raw_frame(
        1, "AAA", adjusted_frame(Open=55.0)
    )

    with pytest.raises(ValidationError):
        map_norgate_adjusted_raw_frame(raw)


def test_optional_raw_values_must_be_numeric_or_null() -> None:
    with pytest.raises(ProviderValidationError, match="Turnover"):
        prepare_norgate_adjusted_raw_frame(
            1, "AAA", adjusted_frame(Turnover="not numeric")
        )
