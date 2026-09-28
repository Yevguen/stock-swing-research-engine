# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
from datetime import date, datetime, timezone
from math import inf, nan

import pandas as pd
import pytest
from pydantic import ValidationError

from stock_swing_d1.models import DividendEvent, StockBar


EXPECTED_FIELDS = {
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
}


def valid_event_data() -> dict[str, object]:
    return {
        "security_id": "NORGATE:1900000003",
        "symbol": "SYNTHC",
        "entitlement_date": date(2026, 8, 7),
        "date_semantics": "entitlement_close",
        "dividend_type": "ordinary_cash",
        "amount_per_share": 0.25,
        "currency": "USD",
        "source_provider": "Norgate Data",
        "source_asset_id": 1_900_000_003,
        "source_adjustment_mode": "CAPITALSPECIAL",
    }


def test_valid_complete_dividend_event_has_exactly_the_v01_fields() -> None:
    event = DividendEvent(**valid_event_data())

    assert event.security_id == "NORGATE:1900000003"
    assert event.symbol == "SYNTHC"
    assert event.entitlement_date == date(2026, 8, 7)
    assert event.date_semantics == "entitlement_close"
    assert event.dividend_type == "ordinary_cash"
    assert event.amount_per_share == 0.25
    assert event.currency == "USD"
    assert event.source_provider == "Norgate Data"
    assert event.source_asset_id == 1_900_000_003
    assert event.source_adjustment_mode == "CAPITALSPECIAL"
    assert len(DividendEvent.model_fields) == 10
    assert set(DividendEvent.model_fields) == EXPECTED_FIELDS
    assert set(event.model_dump()) == EXPECTED_FIELDS


@pytest.mark.parametrize(
    "field_name",
    [
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
    ],
)
def test_each_field_is_required(field_name: str) -> None:
    data = valid_event_data()
    del data[field_name]

    with pytest.raises(ValidationError):
        DividendEvent(**data)


@pytest.mark.parametrize("asset_id", [1, 1_900_000_003])
def test_matching_positive_norgate_identity_is_accepted(asset_id: int) -> None:
    data = valid_event_data()
    data.update(
        {"security_id": f"NORGATE:{asset_id}", "source_asset_id": asset_id}
    )

    event = DividendEvent(**data)

    assert event.security_id == f"NORGATE:{asset_id}"
    assert event.source_asset_id == asset_id


def test_security_id_is_trimmed_before_validation() -> None:
    data = valid_event_data()
    data["security_id"] = "  NORGATE:1900000003  "

    event = DividendEvent(**data)

    assert event.security_id == "NORGATE:1900000003"


@pytest.mark.parametrize(
    "invalid_value",
    [
        "",
        " ",
        "SYNTHA",
        "NORGATE:0",
        "NORGATE:-1",
        "NORGATE:abc",
        "OTHER:1900000003",
        "NORGATE:",
        "NORGATE:1.0",
        "NORGATE:+1",
        "NORGATE:01",
        "norgate:1900000003",
        "NORGATE:1900000003:1",
    ],
)
def test_invalid_security_id_is_rejected(invalid_value: str) -> None:
    data = valid_event_data()
    data["security_id"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


@pytest.mark.parametrize("asset_id", [1, 1_900_000_003, 2_147_483_647])
def test_positive_source_asset_id_is_accepted(asset_id: int) -> None:
    data = valid_event_data()
    data.update(
        {"security_id": f"NORGATE:{asset_id}", "source_asset_id": asset_id}
    )

    event = DividendEvent(**data)

    assert event.source_asset_id == asset_id
    assert type(event.source_asset_id) is int


@pytest.mark.parametrize("invalid_value", [0, -1, True, False, 1.0, "1"])
def test_source_asset_id_must_be_a_positive_strict_integer(
    invalid_value: object,
) -> None:
    data = valid_event_data()
    data["source_asset_id"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


def test_security_id_source_asset_id_mismatch_is_rejected() -> None:
    data = valid_event_data()
    data["security_id"] = "NORGATE:1900000001"

    with pytest.raises(ValidationError, match="security_id must match"):
        DividendEvent(**data)


@pytest.mark.parametrize("raw_symbol", ["  synthc  ", "  brk.b  ", "bf-b"])
def test_symbol_normalization_matches_stock_bar(raw_symbol: str) -> None:
    data = valid_event_data()
    data["symbol"] = raw_symbol
    event = DividendEvent(**data)

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

    assert event.symbol == stock_bar.symbol


@pytest.mark.parametrize("invalid_value", ["", "   \t\n"])
def test_empty_or_whitespace_only_symbol_is_rejected(invalid_value: str) -> None:
    data = valid_event_data()
    data["symbol"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


def test_python_date_is_accepted_without_conversion() -> None:
    event = DividendEvent(**valid_event_data())

    assert type(event.entitlement_date) is date


@pytest.mark.parametrize(
    "invalid_value",
    [
        datetime(2026, 8, 7),
        datetime(2026, 8, 7, tzinfo=timezone.utc),
        pd.Timestamp("2026-08-07"),
        pd.Timestamp("2026-08-07", tz="UTC"),
        "2026-08-07",
        1_786_060_800,
        object(),
    ],
)
def test_non_python_date_input_is_rejected(invalid_value: object) -> None:
    data = valid_event_data()
    data["entitlement_date"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


@pytest.mark.parametrize(
    "invalid_value",
    ["ex_date", "effective_date", "payment_date", "unknown", "arbitrary"],
)
def test_only_entitlement_close_date_semantics_is_allowed(
    invalid_value: str,
) -> None:
    data = valid_event_data()
    data["date_semantics"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


@pytest.mark.parametrize(
    "invalid_value",
    [
        "special_dividend",
        "stock_dividend",
        "special_distribution",
        "capital_return",
        "spinoff",
        "earnings",
        "delisting",
        "arbitrary",
    ],
)
def test_only_ordinary_cash_dividend_type_is_allowed(invalid_value: str) -> None:
    data = valid_event_data()
    data["dividend_type"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


@pytest.mark.parametrize("valid_amount", [0.01, 0.25, 1.0, 3.75])
def test_positive_finite_amount_per_share_is_accepted(valid_amount: float) -> None:
    data = valid_event_data()
    data["amount_per_share"] = valid_amount

    event = DividendEvent(**data)

    assert event.amount_per_share == valid_amount
    assert type(event.amount_per_share) is float


def test_fractional_amount_precision_is_preserved_without_rounding() -> None:
    data = valid_event_data()
    data["amount_per_share"] = 0.123456789012345

    event = DividendEvent(**data)

    assert event.amount_per_share == 0.123456789012345


@pytest.mark.parametrize(
    "invalid_value", [0, 0.0, -0.01, -1, nan, inf, -inf, True, False, "0.25"]
)
def test_invalid_amount_per_share_is_rejected(invalid_value: object) -> None:
    data = valid_event_data()
    data["amount_per_share"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


@pytest.mark.parametrize("invalid_value", ["EUR", "GBP", "", "arbitrary"])
def test_only_usd_currency_is_allowed(invalid_value: str) -> None:
    data = valid_event_data()
    data["currency"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


@pytest.mark.parametrize(
    "invalid_value", ["Other Provider", "Norgate", "norgate data", ""]
)
def test_only_norgate_data_source_provider_is_allowed(invalid_value: str) -> None:
    data = valid_event_data()
    data["source_provider"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


@pytest.mark.parametrize(
    "invalid_value",
    [
        "NONE",
        "CAPITAL",
        "TOTALRETURN",
        "unadjusted",
        "capital_special_adjusted",
        "arbitrary",
    ],
)
def test_only_capitalspecial_adjustment_mode_is_allowed(
    invalid_value: str,
) -> None:
    data = valid_event_data()
    data["source_adjustment_mode"] = invalid_value

    with pytest.raises(ValidationError):
        DividendEvent(**data)


@pytest.mark.parametrize(
    "extra_field",
    [
        "payment_date",
        "shares_held",
        "gross_cash",
        "withholding_tax",
        "portfolio_id",
        "unexpected_field",
    ],
)
def test_deferred_and_arbitrary_extra_fields_are_rejected(
    extra_field: str,
) -> None:
    data = valid_event_data()
    data[extra_field] = "not allowed"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        DividendEvent(**data)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("amount_per_share", 0.5),
        ("symbol", "SYNTHA"),
        ("entitlement_date", date(2026, 8, 8)),
    ],
)
def test_validated_event_is_immutable(
    field_name: str, replacement: object
) -> None:
    event = DividendEvent(**valid_event_data())

    with pytest.raises(ValidationError, match="frozen_instance"):
        setattr(event, field_name, replacement)
