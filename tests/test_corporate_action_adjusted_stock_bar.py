# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
from datetime import date, datetime, timezone
from math import inf, nan

import pandas as pd
import pytest
from pydantic import ValidationError

from stock_swing_d1.models import CorporateActionAdjustedStockBar, StockBar


def valid_bar_data() -> dict[str, object]:
    return {
        "security_id": "NORGATE:1900000001",
        "symbol": "SYNTHA",
        "trading_date": date(2026, 8, 7),
        "timeframe": "D1",
        "session_type": "regular",
        "currency": "USD",
        "price_basis": "capital_special_adjusted",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1_250_000.0,
    }


def test_valid_complete_adjusted_bar() -> None:
    bar = CorporateActionAdjustedStockBar(**valid_bar_data())

    assert bar.security_id == "NORGATE:1900000001"
    assert bar.symbol == "SYNTHA"
    assert bar.trading_date == date(2026, 8, 7)
    assert bar.timeframe == "D1"
    assert bar.session_type == "regular"
    assert bar.currency == "USD"
    assert bar.price_basis == "capital_special_adjusted"
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 102.0, 99.0, 101.0)
    assert bar.volume == 1_250_000.0


def test_model_contains_exactly_the_frozen_v01_fields() -> None:
    assert set(CorporateActionAdjustedStockBar.model_fields) == {
        "security_id",
        "symbol",
        "trading_date",
        "timeframe",
        "session_type",
        "currency",
        "price_basis",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }


def test_validated_adjusted_bar_is_immutable() -> None:
    bar = CorporateActionAdjustedStockBar(**valid_bar_data())

    with pytest.raises(ValidationError, match="frozen_instance"):
        bar.close = 100.5


@pytest.mark.parametrize(
    "field_name",
    [
        "security_id",
        "symbol",
        "trading_date",
        "timeframe",
        "session_type",
        "currency",
        "price_basis",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ],
)
def test_each_field_is_required(field_name: str) -> None:
    data = valid_bar_data()
    del data[field_name]

    with pytest.raises(ValidationError):
        CorporateActionAdjustedStockBar(**data)


@pytest.mark.parametrize("extra_field", ["adjustment_factor", "raw_close", "dividend"])
def test_extra_fields_are_rejected(extra_field: str) -> None:
    data = valid_bar_data()
    data[extra_field] = 1.0

    with pytest.raises(ValidationError, match="extra_forbidden"):
        CorporateActionAdjustedStockBar(**data)


@pytest.mark.parametrize("asset_id", [1, 1_900_000_001, 1_900_000_002])
def test_positive_norgate_security_id_is_accepted(asset_id: int) -> None:
    data = valid_bar_data()
    data["security_id"] = f"NORGATE:{asset_id}"

    bar = CorporateActionAdjustedStockBar(**data)

    assert bar.security_id == f"NORGATE:{asset_id}"


def test_security_id_is_trimmed_before_validation() -> None:
    data = valid_bar_data()
    data["security_id"] = "  NORGATE:1900000001  "

    bar = CorporateActionAdjustedStockBar(**data)

    assert bar.security_id == "NORGATE:1900000001"


@pytest.mark.parametrize(
    "invalid_value",
    ["", "   ", "SYNTHA", "NORGATE:0", "NORGATE:-1", "NORGATE:1.0", "norgate:1", "OTHER:1"],
)
def test_malformed_security_id_is_rejected(invalid_value: str) -> None:
    data = valid_bar_data()
    data["security_id"] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionAdjustedStockBar(**data)


@pytest.mark.parametrize("raw_symbol", ["  syntha  ", "  brk.b  "])
def test_symbol_normalization_matches_stock_bar(raw_symbol: str) -> None:
    data = valid_bar_data()
    data["symbol"] = raw_symbol
    adjusted_bar = CorporateActionAdjustedStockBar(**data)

    stock_bar = StockBar(
        security_id="audit-id",
        symbol=raw_symbol,
        trading_date=date(2026, 8, 7),
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=1,
    )

    assert adjusted_bar.symbol == stock_bar.symbol


def test_python_date_is_accepted_without_conversion() -> None:
    bar = CorporateActionAdjustedStockBar(**valid_bar_data())

    assert type(bar.trading_date) is date


@pytest.mark.parametrize(
    "invalid_value",
    [
        datetime(2026, 8, 7),
        datetime(2026, 8, 7, tzinfo=timezone.utc),
        pd.Timestamp("2026-08-07"),
        pd.Timestamp("2026-08-07", tz="UTC"),
        "2026-08-07",
    ],
)
def test_non_python_date_input_is_rejected(invalid_value: object) -> None:
    data = valid_bar_data()
    data["trading_date"] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionAdjustedStockBar(**data)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("timeframe", "H1"),
        ("session_type", "extended"),
        ("currency", "EUR"),
        ("price_basis", "unadjusted"),
        ("price_basis", "total_return"),
    ],
)
def test_invalid_frozen_value_is_rejected(field_name: str, invalid_value: str) -> None:
    data = valid_bar_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionAdjustedStockBar(**data)


def test_flat_positive_ohlc_is_accepted() -> None:
    data = valid_bar_data()
    data.update({"open": 100, "high": 100, "low": 100, "close": 100})

    bar = CorporateActionAdjustedStockBar(**data)

    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 100.0, 100.0, 100.0)


@pytest.mark.parametrize("field_name", ["open", "high", "low", "close"])
@pytest.mark.parametrize("invalid_value", [0, -1.0, nan, inf, -inf, True, False, "100.5"])
def test_invalid_ohlc_value_is_rejected(
    field_name: str, invalid_value: object
) -> None:
    data = valid_bar_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionAdjustedStockBar(**data)


@pytest.mark.parametrize(
    "overrides",
    [
        {"open": 103.0},
        {"open": 98.0},
        {"close": 103.0},
        {"close": 98.0},
        {"low": 103.0},
    ],
    ids=[
        "open-above-high",
        "open-below-low",
        "close-above-high",
        "close-below-low",
        "low-above-high",
    ],
)
def test_invalid_ohlc_relationship_is_rejected(overrides: dict[str, float]) -> None:
    data = valid_bar_data()
    data.update(overrides)

    with pytest.raises(ValidationError):
        CorporateActionAdjustedStockBar(**data)


@pytest.mark.parametrize("valid_volume", [0.0, 1_250_000.0, 666_666.6666667])
def test_nonnegative_fractional_adjusted_volume_is_accepted(
    valid_volume: float,
) -> None:
    data = valid_bar_data()
    data["volume"] = valid_volume

    bar = CorporateActionAdjustedStockBar(**data)

    assert bar.volume == valid_volume


@pytest.mark.parametrize("invalid_value", [-1, nan, inf, -inf, True, False, "100.5"])
def test_invalid_adjusted_volume_is_rejected(invalid_value: object) -> None:
    data = valid_bar_data()
    data["volume"] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionAdjustedStockBar(**data)
