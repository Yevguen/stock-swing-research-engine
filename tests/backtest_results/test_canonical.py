from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum

import pytest
from pydantic import BaseModel, ConfigDict

from stock_swing_d1.backtest_results import (
    canonicalize_semantic_value,
    semantic_json_bytes,
    semantic_sha256,
)


@pytest.mark.parametrize(
    "value",
    [Decimal("100"), Decimal("100.0"), Decimal("100.00"), Decimal("1E+2")],
)
def test_equivalent_decimals_have_identical_canonical_bytes_and_hash(value):
    assert semantic_json_bytes(value) == b'"100"'
    assert semantic_sha256(value) == semantic_sha256(Decimal("100"))


@pytest.mark.parametrize(
    "value", [Decimal("-0"), Decimal("0"), Decimal("0.000")]
)
def test_decimal_zero_has_one_representation(value):
    assert canonicalize_semantic_value(value) == "0"


def test_decimal_plain_format_and_distinction():
    assert canonicalize_semantic_value(Decimal("100.50")) == "100.5"
    assert canonicalize_semantic_value(Decimal("0.0100")) == "0.01"
    assert canonicalize_semantic_value(Decimal("1E+3")) == "1000"
    assert semantic_sha256(Decimal("1")) != semantic_sha256(Decimal("1.0001"))


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_nonfinite_decimal_is_rejected(value):
    with pytest.raises(ValueError):
        semantic_json_bytes(value)


def test_aware_datetimes_normalize_to_fixed_six_digit_utc():
    utc = datetime(2026, 8, 24, 15, 0, 0, 123, tzinfo=timezone.utc)
    plus_two = utc.astimezone(timezone(timedelta(hours=2)))
    minus_five = utc.astimezone(timezone(timedelta(hours=-5)))
    expected = b'"2026-08-24T15:00:00.000123Z"'
    assert semantic_json_bytes(utc) == expected
    assert semantic_json_bytes(plus_two) == expected
    assert semantic_json_bytes(minus_five) == expected
    assert semantic_sha256(utc) == semantic_sha256(plus_two) == semantic_sha256(minus_five)


def test_naive_datetime_is_rejected_and_date_is_date_only():
    with pytest.raises(ValueError):
        semantic_json_bytes(datetime(2026, 8, 24, 15, 0))
    assert semantic_json_bytes(date(2026, 8, 24)) == b'"2026-08-24"'


def test_mapping_order_does_not_matter_but_sequence_order_does():
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}
    assert semantic_json_bytes(left) == semantic_json_bytes(right)
    assert semantic_sha256([1, 2]) != semantic_sha256([2, 1])
    assert semantic_json_bytes((1, 2)) == semantic_json_bytes([1, 2])


@pytest.mark.parametrize("value", [{1: "bad"}, {" bad": 1}, {"": 1}])
def test_mapping_keys_must_be_canonical_strings(value):
    with pytest.raises(TypeError):
        semantic_json_bytes(value)


@pytest.mark.parametrize("value", [{1, 2}, frozenset({1, 2})])
def test_unordered_collections_are_rejected(value):
    with pytest.raises(TypeError):
        semantic_json_bytes(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_float_is_rejected(value):
    with pytest.raises(ValueError):
        semantic_json_bytes(value)


def test_finite_float_remains_json_number():
    assert semantic_json_bytes(1.25) == b"1.25"


class ExampleEnum(StrEnum):
    VALUE = "semantic-value"


@dataclass
class ExampleDataclass:
    visible: int
    _cache: int = 2


class ExampleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def test_enum_pydantic_and_dataclass_use_declared_semantics_only():
    assert canonicalize_semantic_value(ExampleEnum.VALUE) == "semantic-value"
    assert canonicalize_semantic_value(ExampleModel(value=3)) == {"value": 3}
    assert canonicalize_semantic_value(ExampleDataclass(1)) == {"visible": 1}


def test_dataclass_private_cache_does_not_affect_semantic_bytes_or_hash():
    first = ExampleDataclass(visible=1, _cache=2)
    second = ExampleDataclass(visible=1, _cache=999)
    assert semantic_json_bytes(first) == semantic_json_bytes(second)
    assert semantic_sha256(first) == semantic_sha256(second)


def test_dataclass_public_field_remains_semantically_significant():
    first = ExampleDataclass(visible=1, _cache=2)
    second = ExampleDataclass(visible=2, _cache=2)
    assert semantic_json_bytes(first) != semantic_json_bytes(second)
    assert semantic_sha256(first) != semantic_sha256(second)
