# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
from datetime import date, datetime, timezone
from math import inf, nan

import pandas as pd
import pytest
from pydantic import ValidationError

from stock_swing_d1.models import CorporateActionEvent, StockBar


def valid_unknown_event_data() -> dict[str, object]:
    return {
        "security_id": "NORGATE:1900000002",
        "symbol": "SYNTHA",
        "event_date": date(2026, 8, 7),
        "date_semantics": "entitlement_close",
        "event_type": "unknown_capital_event",
        "terms_verified": False,
        "new_shares": None,
        "old_shares": None,
        "source_provider": "Norgate Data",
        "source_asset_id": 1_900_000_002,
    }


def test_valid_unknown_capital_event() -> None:
    event = CorporateActionEvent(**valid_unknown_event_data())

    assert event.security_id == "NORGATE:1900000002"
    assert event.symbol == "SYNTHA"
    assert event.event_date == date(2026, 8, 7)
    assert event.date_semantics == "entitlement_close"
    assert event.event_type == "unknown_capital_event"
    assert event.terms_verified is False
    assert event.new_shares is None
    assert event.old_shares is None
    assert event.source_provider == "Norgate Data"
    assert event.source_asset_id == 1_900_000_002


def test_model_contains_exactly_the_frozen_v01_fields() -> None:
    assert set(CorporateActionEvent.model_fields) == {
        "security_id",
        "symbol",
        "event_date",
        "date_semantics",
        "event_type",
        "terms_verified",
        "new_shares",
        "old_shares",
        "source_provider",
        "source_asset_id",
    }


def test_validated_event_is_immutable() -> None:
    event = CorporateActionEvent(**valid_unknown_event_data())

    with pytest.raises(ValidationError, match="frozen_instance"):
        event.event_type = "other"


@pytest.mark.parametrize(
    "field_name",
    [
        "security_id",
        "symbol",
        "event_date",
        "date_semantics",
        "event_type",
        "terms_verified",
        "new_shares",
        "old_shares",
        "source_provider",
        "source_asset_id",
    ],
)
def test_each_field_is_required(field_name: str) -> None:
    data = valid_unknown_event_data()
    del data[field_name]

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


def test_extra_field_is_rejected() -> None:
    data = valid_unknown_event_data()
    data["cash_consideration"] = 10.0

    with pytest.raises(ValidationError, match="extra_forbidden"):
        CorporateActionEvent(**data)


@pytest.mark.parametrize("invalid_value", ["norgate data", "Norgate", "Other"])
def test_source_provider_must_be_norgate_data(invalid_value: str) -> None:
    data = valid_unknown_event_data()
    data["source_provider"] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


@pytest.mark.parametrize("invalid_value", [0, -1, True, False, 1_900_000_002.0, "1900000002"])
def test_source_asset_id_must_be_a_positive_strict_integer(
    invalid_value: object,
) -> None:
    data = valid_unknown_event_data()
    data["source_asset_id"] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


@pytest.mark.parametrize("asset_id", [1, 1_900_000_001, 1_900_000_002])
def test_matching_security_and_source_asset_id_are_accepted(asset_id: int) -> None:
    data = valid_unknown_event_data()
    data.update(
        {"security_id": f"NORGATE:{asset_id}", "source_asset_id": asset_id}
    )

    event = CorporateActionEvent(**data)

    assert event.security_id == f"NORGATE:{asset_id}"
    assert event.source_asset_id == asset_id


def test_security_id_source_asset_id_mismatch_is_rejected() -> None:
    data = valid_unknown_event_data()
    data["security_id"] = "NORGATE:1900000001"

    with pytest.raises(ValidationError, match="security_id must match"):
        CorporateActionEvent(**data)


@pytest.mark.parametrize(
    "invalid_value",
    ["", "   ", "SYNTHA", "NORGATE:0", "NORGATE:-1", "NORGATE:1.0", "OTHER:1900000002"],
)
def test_malformed_security_id_is_rejected(invalid_value: str) -> None:
    data = valid_unknown_event_data()
    data["security_id"] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


@pytest.mark.parametrize("raw_symbol", ["  syntha  ", "  brk.b  "])
def test_symbol_normalization_matches_stock_bar(raw_symbol: str) -> None:
    data = valid_unknown_event_data()
    data["symbol"] = raw_symbol
    event = CorporateActionEvent(**data)

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


def test_python_date_is_accepted_without_conversion() -> None:
    event = CorporateActionEvent(**valid_unknown_event_data())

    assert type(event.event_date) is date


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
    data = valid_unknown_event_data()
    data["event_date"] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


@pytest.mark.parametrize(
    "date_semantics",
    ["entitlement_close", "ex_date", "effective_date", "unknown"],
)
def test_frozen_date_semantics_are_accepted(date_semantics: str) -> None:
    data = valid_unknown_event_data()
    data["date_semantics"] = date_semantics

    event = CorporateActionEvent(**data)

    assert event.date_semantics == date_semantics


def test_arbitrary_date_semantics_is_rejected() -> None:
    data = valid_unknown_event_data()
    data["date_semantics"] = "announcement_date"

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


@pytest.mark.parametrize(
    "event_type",
    [
        "unknown_capital_event",
        "split",
        "reverse_split",
        "stock_dividend",
        "rights_issue",
        "special_distribution",
        "spinoff",
        "merger_or_reorganization",
        "other",
    ],
)
def test_each_frozen_event_type_is_accepted(event_type: str) -> None:
    data = valid_unknown_event_data()
    data["event_type"] = event_type

    event = CorporateActionEvent(**data)

    assert event.event_type == event_type


@pytest.mark.parametrize(
    "invalid_value",
    [
        "ordinary_dividend",
        "earnings",
        "delisting",
        "bankruptcy",
        "ticker_change",
        "arbitrary",
    ],
)
def test_non_capital_event_type_is_rejected(invalid_value: str) -> None:
    data = valid_unknown_event_data()
    data["event_type"] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


@pytest.mark.parametrize("invalid_value", [0, 1, "false", "true", None])
def test_terms_verified_must_be_a_genuine_boolean(invalid_value: object) -> None:
    data = valid_unknown_event_data()
    data["terms_verified"] = invalid_value

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


def test_unknown_event_with_verified_terms_is_rejected() -> None:
    data = valid_unknown_event_data()
    data["terms_verified"] = True

    with pytest.raises(ValidationError, match="cannot be verified"):
        CorporateActionEvent(**data)


def test_unknown_event_with_ratio_is_rejected() -> None:
    data = valid_unknown_event_data()
    data.update({"new_shares": 2.0, "old_shares": 1.0})

    with pytest.raises(ValidationError, match="cannot have share-ratio"):
        CorporateActionEvent(**data)


@pytest.mark.parametrize(
    ("new_shares", "old_shares"), [(2.0, None), (None, 1.0)]
)
def test_ratio_fields_must_be_all_or_nothing(
    new_shares: float | None, old_shares: float | None
) -> None:
    data = valid_unknown_event_data()
    data.update(
        {
            "event_type": "other",
            "new_shares": new_shares,
            "old_shares": old_shares,
        }
    )

    with pytest.raises(ValidationError, match="both be None or present"):
        CorporateActionEvent(**data)


@pytest.mark.parametrize("field_name", ["new_shares", "old_shares"])
@pytest.mark.parametrize("invalid_value", [0, -1.0, nan, inf, -inf, True, False, "2"])
def test_invalid_ratio_value_is_rejected(
    field_name: str, invalid_value: object
) -> None:
    data = valid_unknown_event_data()
    data.update(
        {
            "event_type": "other",
            "terms_verified": True,
            "new_shares": 2.0,
            "old_shares": 1.0,
            field_name: invalid_value,
        }
    )

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


@pytest.mark.parametrize(
    ("new_shares", "old_shares"), [(2.0, 1.0), (3.0, 2.0)]
)
def test_verified_forward_split_is_accepted(
    new_shares: float, old_shares: float
) -> None:
    data = valid_unknown_event_data()
    data.update(
        {
            "event_type": "split",
            "terms_verified": True,
            "new_shares": new_shares,
            "old_shares": old_shares,
        }
    )

    event = CorporateActionEvent(**data)

    assert (event.new_shares, event.old_shares) == (new_shares, old_shares)


@pytest.mark.parametrize(
    ("new_shares", "old_shares"), [(None, None), (1.0, 1.0), (1.0, 2.0)]
)
def test_invalid_verified_forward_split_is_rejected(
    new_shares: float | None, old_shares: float | None
) -> None:
    data = valid_unknown_event_data()
    data.update(
        {
            "event_type": "split",
            "terms_verified": True,
            "new_shares": new_shares,
            "old_shares": old_shares,
        }
    )

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


@pytest.mark.parametrize(
    ("new_shares", "old_shares"), [(1.0, 10.0), (2.0, 5.0)]
)
def test_verified_reverse_split_is_accepted(
    new_shares: float, old_shares: float
) -> None:
    data = valid_unknown_event_data()
    data.update(
        {
            "event_type": "reverse_split",
            "terms_verified": True,
            "new_shares": new_shares,
            "old_shares": old_shares,
        }
    )

    event = CorporateActionEvent(**data)

    assert (event.new_shares, event.old_shares) == (new_shares, old_shares)


@pytest.mark.parametrize(
    ("new_shares", "old_shares"), [(None, None), (1.0, 1.0), (2.0, 1.0)]
)
def test_invalid_verified_reverse_split_is_rejected(
    new_shares: float | None, old_shares: float | None
) -> None:
    data = valid_unknown_event_data()
    data.update(
        {
            "event_type": "reverse_split",
            "terms_verified": True,
            "new_shares": new_shares,
            "old_shares": old_shares,
        }
    )

    with pytest.raises(ValidationError):
        CorporateActionEvent(**data)


@pytest.mark.parametrize("event_type", ["split", "reverse_split"])
def test_unverified_split_classification_without_ratio_is_accepted(
    event_type: str,
) -> None:
    data = valid_unknown_event_data()
    data["event_type"] = event_type

    event = CorporateActionEvent(**data)

    assert event.terms_verified is False
    assert event.new_shares is None
    assert event.old_shares is None


@pytest.mark.parametrize("event_type", ["split", "reverse_split"])
def test_unverified_split_classification_with_ratio_is_rejected(
    event_type: str,
) -> None:
    data = valid_unknown_event_data()
    data.update(
        {"event_type": event_type, "new_shares": 2.0, "old_shares": 1.0}
    )

    with pytest.raises(ValidationError, match="must be verified"):
        CorporateActionEvent(**data)
