from datetime import date, datetime

import pytest
from pydantic import ValidationError

from stock_swing_d1.models import EarningsEvent


def valid_event_data() -> dict[str, object]:
    return {
        "event_id": "earnings-event-123",
        "security_id": "security-123",
        "symbol": "ABC",
        "earnings_date": date(2026, 7, 28),
        "announcement_timing": "BMO",
        "source_name": "historical earnings archive",
        "source_reference": "dataset:earnings-2026",
    }


@pytest.mark.parametrize(
    "announcement_timing",
    ["BMO", "AMC", "during_session", "unknown"],
)
def test_valid_announcement_timing(announcement_timing: str) -> None:
    data = valid_event_data()
    data["announcement_timing"] = announcement_timing

    event = EarningsEvent(**data)

    assert event.announcement_timing == announcement_timing


def test_python_date_value_is_accepted() -> None:
    data = valid_event_data()
    data["earnings_date"] = date(2026, 7, 28)

    event = EarningsEvent(**data)

    assert event.earnings_date == date(2026, 7, 28)
    assert type(event.earnings_date) is date


def test_pure_iso_date_string_is_accepted() -> None:
    data = valid_event_data()
    data["earnings_date"] = "2026-07-28"

    event = EarningsEvent(**data)

    assert event.earnings_date == date(2026, 7, 28)
    assert type(event.earnings_date) is date


@pytest.mark.parametrize("invalid_value", ["20260728", "2026-W31-2"])
def test_non_yyyy_mm_dd_iso_date_string_is_rejected(invalid_value: str) -> None:
    data = valid_event_data()
    data["earnings_date"] = invalid_value

    with pytest.raises(ValidationError):
        EarningsEvent(**data)


def test_required_text_is_trimmed_and_symbol_is_uppercased() -> None:
    data = valid_event_data()
    data.update(
        {
            "event_id": "  earnings-event-123  ",
            "security_id": "  security-123  ",
            "symbol": "  abc  ",
            "source_name": "  historical earnings archive  ",
            "source_reference": "  dataset:earnings-2026  ",
        }
    )

    event = EarningsEvent(**data)

    assert event.event_id == "earnings-event-123"
    assert event.security_id == "security-123"
    assert event.symbol == "ABC"
    assert event.source_name == "historical earnings archive"
    assert event.source_reference == "dataset:earnings-2026"


def test_symbol_with_punctuation_is_accepted_and_normalized() -> None:
    data = valid_event_data()
    data["symbol"] = "  brk.b  "

    event = EarningsEvent(**data)

    assert event.symbol == "BRK.B"


@pytest.mark.parametrize(
    "field_name",
    ["event_id", "security_id", "symbol", "source_name", "source_reference"],
)
@pytest.mark.parametrize("invalid_value", ["", "   \t\n"])
def test_empty_or_whitespace_only_required_text_is_rejected(
    field_name: str, invalid_value: str
) -> None:
    data = valid_event_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        EarningsEvent(**data)


@pytest.mark.parametrize(
    "invalid_value",
    ["before_open", "after_close", "bmo", "amc", "", "next_session"],
)
def test_invalid_announcement_timing_is_rejected(invalid_value: str) -> None:
    data = valid_event_data()
    data["announcement_timing"] = invalid_value

    with pytest.raises(ValidationError):
        EarningsEvent(**data)


@pytest.mark.parametrize(
    "invalid_value",
    [
        datetime(2026, 7, 28),
        1_774_915_200,
        1_774_915_200.0,
        "2026-07-28T00:00:00",
        "2026-07-28 00:00:00",
    ],
)
def test_invalid_earnings_date_input_is_rejected(invalid_value: object) -> None:
    data = valid_event_data()
    data["earnings_date"] = invalid_value

    with pytest.raises(ValidationError):
        EarningsEvent(**data)


@pytest.mark.parametrize(
    "field_name",
    [
        "event_id",
        "security_id",
        "symbol",
        "earnings_date",
        "announcement_timing",
        "source_name",
        "source_reference",
    ],
)
def test_each_field_is_required(field_name: str) -> None:
    data = valid_event_data()
    del data[field_name]

    with pytest.raises(ValidationError):
        EarningsEvent(**data)


def test_extra_field_is_rejected() -> None:
    data = valid_event_data()
    data["EPS_actual"] = 2.35

    with pytest.raises(ValidationError, match="extra_forbidden"):
        EarningsEvent(**data)


def test_validated_event_is_immutable() -> None:
    event = EarningsEvent(**valid_event_data())

    with pytest.raises(ValidationError, match="frozen_instance"):
        event.announcement_timing = "AMC"


def test_model_contains_exactly_the_frozen_v01_fields() -> None:
    assert set(EarningsEvent.model_fields) == {
        "event_id",
        "security_id",
        "symbol",
        "earnings_date",
        "announcement_timing",
        "source_name",
        "source_reference",
    }
    assert {
        "EPS_actual",
        "revenue_actual",
        "signal",
        "profit",
        "known_from",
    }.isdisjoint(EarningsEvent.model_fields)
