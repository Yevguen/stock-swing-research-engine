from datetime import date, datetime
from math import inf, nan

import pytest
from pydantic import ValidationError

from stock_swing_d1.models import StockBar


def valid_bar_data() -> dict[str, object]:
    return {
        "security_id": "security-123",
        "symbol": "ABC",
        "trading_date": date(2026, 8, 7),
        "timeframe": "D1",
        "session_type": "regular",
        "currency": "USD",
        "price_basis": "unadjusted",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1_000_000,
    }


def test_normal_valid_d1_bar() -> None:
    bar = StockBar(**valid_bar_data())

    assert bar.security_id == "security-123"
    assert bar.symbol == "ABC"
    assert bar.trading_date == date(2026, 8, 7)
    assert bar.timeframe == "D1"
    assert bar.session_type == "regular"
    assert bar.currency == "USD"
    assert bar.price_basis == "unadjusted"
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 102.0, 99.0, 101.0)
    assert bar.volume == 1_000_000


def test_standard_iso_trading_date_string_parses_to_date() -> None:
    data = valid_bar_data()
    data["trading_date"] = "2026-08-07"

    bar = StockBar(**data)

    assert bar.trading_date == date(2026, 8, 7)
    assert isinstance(bar.trading_date, date)


@pytest.mark.parametrize(
    "invalid_value",
    ["2026-08-07T00:00:00", "2026-08-07 00:00:00"],
)
def test_datetime_formatted_trading_date_string_is_rejected(
    invalid_value: str,
) -> None:
    data = valid_bar_data()
    data["trading_date"] = invalid_value

    with pytest.raises(ValidationError):
        StockBar(**data)


def test_python_date_value_is_accepted() -> None:
    data = valid_bar_data()
    data["trading_date"] = date(2026, 8, 7)

    bar = StockBar(**data)

    assert bar.trading_date == date(2026, 8, 7)


def test_datetime_trading_date_is_rejected_even_at_midnight() -> None:
    data = valid_bar_data()
    data["trading_date"] = datetime(2026, 8, 7)

    with pytest.raises(ValidationError):
        StockBar(**data)


@pytest.mark.parametrize("invalid_value", [1_786_060_800, 1_786_060_800.0])
def test_numeric_timestamp_trading_date_is_rejected(invalid_value: object) -> None:
    data = valid_bar_data()
    data["trading_date"] = invalid_value

    with pytest.raises(ValidationError):
        StockBar(**data)


def test_text_is_trimmed_and_symbol_is_uppercased() -> None:
    data = valid_bar_data()
    data.update({"security_id": "  security-123  ", "symbol": "  abc  "})

    bar = StockBar(**data)

    assert bar.security_id == "security-123"
    assert bar.symbol == "ABC"


def test_symbol_with_punctuation_is_accepted_and_normalized() -> None:
    data = valid_bar_data()
    data["symbol"] = "  brk.b  "

    bar = StockBar(**data)

    assert bar.symbol == "BRK.B"


def test_completely_flat_ohlc_bar_is_accepted() -> None:
    data = valid_bar_data()
    data.update({"open": 100, "high": 100, "low": 100, "close": 100})

    bar = StockBar(**data)

    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 100.0, 100.0, 100.0)
    assert all(type(price) is float for price in (bar.open, bar.high, bar.low, bar.close))


def test_zero_volume_is_accepted() -> None:
    data = valid_bar_data()
    data["volume"] = 0

    bar = StockBar(**data)

    assert bar.volume == 0


def test_positive_integer_volume_is_accepted() -> None:
    data = valid_bar_data()
    data["volume"] = 42

    bar = StockBar(**data)

    assert bar.volume == 42


@pytest.mark.parametrize("field_name", ["security_id", "symbol"])
@pytest.mark.parametrize("invalid_value", ["", "   \t\n"])
def test_empty_or_whitespace_only_text_is_rejected(
    field_name: str, invalid_value: str
) -> None:
    data = valid_bar_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        StockBar(**data)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("timeframe", "H1"),
        ("session_type", "extended"),
        ("currency", "EUR"),
        ("price_basis", "adjusted"),
    ],
)
def test_invalid_fixed_framework_value_is_rejected(
    field_name: str, invalid_value: str
) -> None:
    data = valid_bar_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        StockBar(**data)


@pytest.mark.parametrize("field_name", ["open", "high", "low", "close"])
@pytest.mark.parametrize("invalid_value", [0, -1.0, nan, inf, -inf])
def test_nonpositive_or_nonfinite_price_is_rejected(
    field_name: str, invalid_value: float
) -> None:
    data = valid_bar_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        StockBar(**data)


@pytest.mark.parametrize("field_name", ["open", "high", "low", "close"])
@pytest.mark.parametrize("invalid_value", ["100.5", True, False])
def test_coercive_price_input_is_rejected(
    field_name: str, invalid_value: object
) -> None:
    data = valid_bar_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        StockBar(**data)


@pytest.mark.parametrize(
    "overrides",
    [
        {"open": 103.0},
        {"close": 103.0},
        {"open": 98.0},
        {"close": 98.0},
        {"low": 103.0},
    ],
    ids=[
        "open-above-high",
        "close-above-high",
        "open-below-low",
        "close-below-low",
        "low-above-high",
    ],
)
def test_invalid_ohlc_relationship_is_rejected(overrides: dict[str, float]) -> None:
    data = valid_bar_data()
    data.update(overrides)

    with pytest.raises(ValidationError):
        StockBar(**data)


def test_negative_volume_is_rejected() -> None:
    data = valid_bar_data()
    data["volume"] = -1

    with pytest.raises(ValidationError):
        StockBar(**data)


@pytest.mark.parametrize("invalid_value", [1.5, "100", True])
def test_non_integer_volume_is_rejected(invalid_value: object) -> None:
    data = valid_bar_data()
    data["volume"] = invalid_value

    with pytest.raises(ValidationError):
        StockBar(**data)


def test_trading_date_is_required() -> None:
    data = valid_bar_data()
    del data["trading_date"]

    with pytest.raises(ValidationError):
        StockBar(**data)


@pytest.mark.parametrize("field_name", ["open", "high", "low", "close", "volume"])
def test_required_ohlcv_field_cannot_be_omitted(field_name: str) -> None:
    data = valid_bar_data()
    del data[field_name]

    with pytest.raises(ValidationError):
        StockBar(**data)


@pytest.mark.parametrize("field_name", ["open", "high", "low", "close", "volume"])
def test_required_ohlcv_field_cannot_be_none(field_name: str) -> None:
    data = valid_bar_data()
    data[field_name] = None

    with pytest.raises(ValidationError):
        StockBar(**data)


def test_extra_field_is_rejected() -> None:
    data = valid_bar_data()
    data["adjusted_close"] = 100.5

    with pytest.raises(ValidationError):
        StockBar(**data)


def test_validated_bar_is_immutable() -> None:
    bar = StockBar(**valid_bar_data())

    with pytest.raises(ValidationError, match="frozen_instance"):
        bar.close = 100.5


def test_model_contains_exactly_the_frozen_v01_fields() -> None:
    assert set(StockBar.model_fields) == {
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
    assert {
        "adjusted_close",
        "market_cap",
        "sector",
        "RSI",
        "signal",
        "profit",
    }.isdisjoint(StockBar.model_fields)
