from datetime import date

import pytest
from pydantic import ValidationError

from stock_swing_d1.models import StockUniverseMember


def valid_member_data() -> dict[str, object]:
    return {
        "security_id": "security-123",
        "symbol": "ABC",
        "universe_kind": "historical_research",
        "universe_methodology_version": "0.1",
        "market": "United States",
        "security_type": "common_stock",
        "eligible_from": date(2020, 1, 2),
        "eligible_to": date(2020, 12, 31),
        "source_name": "membership archive",
        "source_reference": "dataset:members-2020",
    }


def test_valid_historical_research_record_with_closed_interval() -> None:
    member = StockUniverseMember(**valid_member_data())

    assert member.universe_kind == "historical_research"
    assert member.eligible_from == date(2020, 1, 2)
    assert member.eligible_to == date(2020, 12, 31)


def test_valid_development_record() -> None:
    data = valid_member_data()
    data["universe_kind"] = "development"

    member = StockUniverseMember(**data)

    assert member.universe_kind == "development"


def test_open_ended_interval_is_accepted() -> None:
    data = valid_member_data()
    data["eligible_to"] = None

    member = StockUniverseMember(**data)

    assert member.eligible_to is None


def test_eligible_to_is_required_even_though_none_is_allowed() -> None:
    data = valid_member_data()
    del data["eligible_to"]

    with pytest.raises(ValidationError):
        StockUniverseMember(**data)


def test_one_day_inclusive_interval_is_accepted() -> None:
    data = valid_member_data()
    data["eligible_to"] = data["eligible_from"]

    member = StockUniverseMember(**data)

    assert member.eligible_to == member.eligible_from


def test_text_is_trimmed_and_symbol_is_uppercased() -> None:
    data = valid_member_data()
    data.update(
        {
            "security_id": "  security-123  ",
            "symbol": "  brk.b  ",
            "source_name": "  membership archive  ",
            "source_reference": "  dataset:members-2020  ",
        }
    )

    member = StockUniverseMember(**data)

    assert member.security_id == "security-123"
    assert member.symbol == "BRK.B"
    assert member.source_name == "membership archive"
    assert member.source_reference == "dataset:members-2020"


def test_standard_date_strings_are_accepted() -> None:
    data = valid_member_data()
    data["eligible_from"] = "2021-03-15"
    data["eligible_to"] = "2021-04-30"

    member = StockUniverseMember(**data)

    assert member.eligible_from == date(2021, 3, 15)
    assert member.eligible_to == date(2021, 4, 30)


@pytest.mark.parametrize(
    "field_name", ["security_id", "symbol", "source_name", "source_reference"]
)
@pytest.mark.parametrize("invalid_value", ["", "   \t\n"])
def test_empty_or_whitespace_only_text_is_rejected(
    field_name: str, invalid_value: str
) -> None:
    data = valid_member_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        StockUniverseMember(**data)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("universe_kind", "production"),
        ("universe_methodology_version", "0.2"),
        ("market", "Canada"),
        ("security_type", "preferred_stock"),
    ],
)
def test_invalid_fixed_framework_value_is_rejected(
    field_name: str, invalid_value: str
) -> None:
    data = valid_member_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        StockUniverseMember(**data)


def test_interval_ending_before_it_starts_is_rejected() -> None:
    data = valid_member_data()
    data["eligible_to"] = date(2020, 1, 1)

    with pytest.raises(ValidationError, match="eligible_to must be on or after"):
        StockUniverseMember(**data)


def test_extra_field_is_rejected() -> None:
    data = valid_member_data()
    data["company_name"] = "Example Company"

    with pytest.raises(ValidationError):
        StockUniverseMember(**data)


def test_model_contains_only_frozen_v01_fields() -> None:
    assert set(StockUniverseMember.model_fields) == {
        "security_id",
        "symbol",
        "universe_kind",
        "universe_methodology_version",
        "market",
        "security_type",
        "eligible_from",
        "eligible_to",
        "source_name",
        "source_reference",
    }
    assert {
        "market_cap",
        "average_dollar_volume",
        "sector",
        "signal",
        "profit",
    }.isdisjoint(StockUniverseMember.model_fields)
