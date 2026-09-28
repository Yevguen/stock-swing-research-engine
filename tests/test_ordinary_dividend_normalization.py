# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
"""Acceptance tests for frozen Gate-3 exact-rational D_H normalization."""

from __future__ import annotations

import decimal
import hashlib
import json
from datetime import date, datetime
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from pydantic import ValidationError

from stock_swing_d1.data.ordinary_dividend_normalization import (
    HISTORICAL_SHARE_AMOUNT_BASIS,
    NORMALIZATION_ARITHMETIC_MODE,
    NORMALIZATION_INPUTS_HASH_DOMAIN,
    NORMALIZATION_METHOD_ID,
    NORMALIZATION_ROUNDING_MODE,
    NORMALIZATION_SCALE,
    Gate3DividendNormalizationInputs,
    Gate3DividendNormalizationResult,
    compute_normalization_inputs_fingerprint,
    normalize_gate3_dividend,
)
from stock_swing_d1.models import DividendEvent


ENTITLEMENT_SESSION = date(2026, 8, 7)


def make_inputs(
    *,
    entitlement_session: date = ENTITLEMENT_SESSION,
    d_capitalspecial: Decimal = Decimal("2"),
    unadjusted_close_t: Decimal = Decimal("3"),
    close_capital_t: Decimal = Decimal("4"),
) -> Gate3DividendNormalizationInputs:
    return Gate3DividendNormalizationInputs(
        entitlement_session=entitlement_session,
        d_capitalspecial=d_capitalspecial,
        unadjusted_close_t=unadjusted_close_t,
        close_capital_t=close_capital_t,
    )


def fingerprint_payload(
    inputs: Gate3DividendNormalizationInputs,
) -> dict[str, object]:
    return {
        "entitlement_session": inputs.entitlement_session.isoformat(),
        "d_capitalspecial": canonical_decimal(inputs.d_capitalspecial),
        "unadjusted_close_t": canonical_decimal(inputs.unadjusted_close_t),
        "close_capital_t": canonical_decimal(inputs.close_capital_t),
        "normalization_method_id": NORMALIZATION_METHOD_ID,
        "normalization_scale": NORMALIZATION_SCALE,
        "normalization_rounding_mode": NORMALIZATION_ROUNDING_MODE,
        "normalization_arithmetic_mode": NORMALIZATION_ARITHMETIC_MODE,
    }


def canonical_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if value.is_zero() else rendered


def synthetic_domain_sha256(payload: dict[str, object]) -> str:
    canonical_bytes = json.dumps(
        {
            "domain": NORMALIZATION_INPUTS_HASH_DOMAIN,
            "payload": payload,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def exact_reference_result(
    inputs: Gate3DividendNormalizationInputs,
) -> Decimal:
    dividend = Fraction(*inputs.d_capitalspecial.as_integer_ratio())
    unadjusted_close = Fraction(*inputs.unadjusted_close_t.as_integer_ratio())
    capital_close = Fraction(*inputs.close_capital_t.as_integer_ratio())
    exact = dividend * unadjusted_close / capital_close
    coefficient = round(exact * (10**NORMALIZATION_SCALE))
    digits = tuple(int(character) for character in str(coefficient))
    return Decimal((0, digits, -NORMALIZATION_SCALE))


def test_exact_formula_uses_all_three_entitlement_session_inputs() -> None:
    inputs = make_inputs()

    result = normalize_gate3_dividend(inputs)

    assert inputs.entitlement_session == ENTITLEMENT_SESSION
    assert inputs.d_capitalspecial == Decimal("2")
    assert inputs.unadjusted_close_t == Decimal("3")
    assert inputs.close_capital_t == Decimal("4")
    assert result.d_h == Decimal("1.5")
    assert result.d_h != inputs.d_capitalspecial
    assert result.d_h.as_tuple().exponent == -38


def test_result_exposes_only_frozen_normalization_semantics() -> None:
    result = normalize_gate3_dividend(make_inputs())

    assert result.amount_basis == HISTORICAL_SHARE_AMOUNT_BASIS
    assert result.normalization_method_id == NORMALIZATION_METHOD_ID
    assert result.normalization_scale == NORMALIZATION_SCALE
    assert result.normalization_rounding_mode == NORMALIZATION_ROUNDING_MODE
    assert result.normalization_arithmetic_mode == NORMALIZATION_ARITHMETIC_MODE
    assert len(result.normalization_inputs_fingerprint) == 64


@pytest.mark.parametrize(
    "field_name",
    ["d_capitalspecial", "unadjusted_close_t", "close_capital_t"],
)
def test_binary_float_is_rejected_without_decimal_coercion(
    field_name: str,
) -> None:
    values = {
        "d_capitalspecial": Decimal("2"),
        "unadjusted_close_t": Decimal("3"),
        "close_capital_t": Decimal("4"),
    }
    values[field_name] = 0.25  # type: ignore[assignment]

    with pytest.raises(ValidationError, match="exact Decimal"):
        make_inputs(**values)


@pytest.mark.parametrize("invalid_value", [1, "1", True])
def test_non_decimal_authoritative_coercion_paths_are_rejected(
    invalid_value: object,
) -> None:
    with pytest.raises(ValidationError, match="exact Decimal"):
        make_inputs(d_capitalspecial=invalid_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    ["d_capitalspecial", "unadjusted_close_t", "close_capital_t"],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("0"),
        Decimal("-0.01"),
    ],
)
def test_nonfinite_and_nonpositive_authoritative_values_are_rejected(
    field_name: str,
    invalid_value: Decimal,
) -> None:
    values = {
        "d_capitalspecial": Decimal("2"),
        "unadjusted_close_t": Decimal("3"),
        "close_capital_t": Decimal("4"),
    }
    values[field_name] = invalid_value

    with pytest.raises(ValidationError):
        make_inputs(**values)


@pytest.mark.parametrize(
    "invalid_session",
    [datetime(2026, 8, 7), "2026-08-07"],
)
def test_entitlement_session_requires_exact_date(
    invalid_session: object,
) -> None:
    with pytest.raises(ValidationError, match="datetime.date"):
        make_inputs(entitlement_session=invalid_session)  # type: ignore[arg-type]


def test_normalization_is_independent_of_ambient_decimal_precision() -> None:
    inputs = make_inputs(
        d_capitalspecial=Decimal("1"),
        unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("7"),
    )

    with localcontext() as context:
        context.prec = 28
        precision_28 = normalize_gate3_dividend(inputs)
    with localcontext() as context:
        context.prec = 60
        precision_60 = normalize_gate3_dividend(inputs)

    expected = Decimal("0.14285714285714285714285714285714285714")
    assert precision_28.d_h == expected
    assert precision_28.d_h.as_tuple() == precision_60.d_h.as_tuple()
    assert precision_28.model_dump_json() == precision_60.model_dump_json()
    assert (
        precision_28.normalization_inputs_fingerprint
        == precision_60.normalization_inputs_fingerprint
    )


def test_normalizer_does_not_read_decimal_getcontext(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("ambient decimal.getcontext() was read")

    monkeypatch.setattr(decimal, "getcontext", fail_if_called)

    result = normalize_gate3_dividend(
        make_inputs(close_capital_t=Decimal("7"))
    )

    assert result.d_h.as_tuple().exponent == -38


def test_exact_rational_reference_agrees_before_the_one_rounding() -> None:
    inputs = make_inputs(
        d_capitalspecial=Decimal("0.123456789012345678901234567890123456789"),
        unadjusted_close_t=Decimal("987654321.0123456789"),
        close_capital_t=Decimal("37.0000000000000000001"),
    )

    result = normalize_gate3_dividend(inputs)

    assert result.d_h == exact_reference_result(inputs)
    assert result.d_h.as_tuple().exponent == -38


def test_terminating_quotient_is_preserved_at_fixed_scale_38() -> None:
    result = normalize_gate3_dividend(
        make_inputs(
            d_capitalspecial=Decimal("0.5"),
            unadjusted_close_t=Decimal("10"),
            close_capital_t=Decimal("4"),
        )
    )

    assert result.d_h == Decimal("1.25000000000000000000000000000000000000")
    assert result.d_h.as_tuple().exponent == -38


@pytest.mark.parametrize(
    ("dividend", "expected"),
    [
        (Decimal("5"), Decimal("0.00000000000000000000000000000000000002")),
        (Decimal("7"), Decimal("0.00000000000000000000000000000000000004")),
    ],
)
def test_exact_scale_38_half_even_ties(
    dividend: Decimal,
    expected: Decimal,
) -> None:
    result = normalize_gate3_dividend(
        make_inputs(
            d_capitalspecial=dividend,
            unadjusted_close_t=Decimal("1"),
            close_capital_t=Decimal("200000000000000000000000000000000000000"),
        )
    )

    assert result.d_h == expected
    assert result.d_h.as_tuple().exponent == -38


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("d_capitalspecial", Decimal("2.01")),
        ("unadjusted_close_t", Decimal("3.01")),
        ("close_capital_t", Decimal("4.01")),
        ("entitlement_session", date(2026, 8, 8)),
    ],
)
def test_each_semantic_input_changes_the_fingerprint(
    field_name: str,
    changed_value: object,
) -> None:
    baseline = make_inputs()
    changed_values = baseline.model_dump(mode="python")
    changed_values[field_name] = changed_value
    changed = Gate3DividendNormalizationInputs(**changed_values)

    assert compute_normalization_inputs_fingerprint(changed) != (
        compute_normalization_inputs_fingerprint(baseline)
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("normalization_method_id", "SYNTHETIC_OTHER_METHOD"),
        ("normalization_scale", 37),
        ("normalization_rounding_mode", "SYNTHETIC_OTHER_ROUNDING"),
        ("normalization_arithmetic_mode", "SYNTHETIC_OTHER_ARITHMETIC"),
    ],
)
def test_each_policy_semantic_is_bound_without_configuring_production(
    field_name: str,
    changed_value: object,
) -> None:
    inputs = make_inputs()
    canonical_payload = fingerprint_payload(inputs)
    assert compute_normalization_inputs_fingerprint(inputs) == (
        synthetic_domain_sha256(canonical_payload)
    )
    changed_payload = dict(canonical_payload)
    changed_payload[field_name] = changed_value

    assert synthetic_domain_sha256(changed_payload) != (
        compute_normalization_inputs_fingerprint(inputs)
    )


def test_equivalent_canonical_decimal_constructions_have_same_fingerprint() -> None:
    first = make_inputs(
        d_capitalspecial=Decimal("2.0"),
        unadjusted_close_t=Decimal("3.00"),
        close_capital_t=Decimal("4.000"),
    )
    second = make_inputs(
        d_capitalspecial=Decimal("2.0000"),
        unadjusted_close_t=Decimal("3"),
        close_capital_t=Decimal("4.0"),
    )

    assert compute_normalization_inputs_fingerprint(first) == (
        compute_normalization_inputs_fingerprint(second)
    )
    assert normalize_gate3_dividend(first) == normalize_gate3_dividend(second)


def test_exact_decimal_representation_survives_input_construction() -> None:
    source = Decimal("1.2300")

    inputs = make_inputs(d_capitalspecial=source)

    assert inputs.d_capitalspecial.as_tuple() == source.as_tuple()


@pytest.mark.parametrize(
    ("model_type", "values"),
    [
        (
            Gate3DividendNormalizationInputs,
            {
                "entitlement_session": ENTITLEMENT_SESSION,
                "d_capitalspecial": Decimal("2"),
                "unadjusted_close_t": Decimal("3"),
                "close_capital_t": Decimal("4"),
            },
        ),
        (
            Gate3DividendNormalizationResult,
            {
                "d_h": Decimal("1.50000000000000000000000000000000000000"),
                "normalization_inputs_fingerprint": "a" * 64,
            },
        ),
    ],
)
def test_normalization_models_are_immutable_and_forbid_extra_fields(
    model_type,
    values: dict[str, object],
) -> None:
    model = model_type(**values)

    with pytest.raises(ValidationError, match="frozen_instance"):
        model.unexpected = "mutation"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        model_type(**values, unexpected="extra")


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("amount_basis", "RAW_CAPITALSPECIAL"),
        ("normalization_method_id", "SYNTHETIC_OTHER_METHOD"),
        ("normalization_scale", 37),
        ("normalization_rounding_mode", "ROUND_DOWN"),
        ("normalization_arithmetic_mode", "DECIMAL_CONTEXT"),
    ],
)
def test_result_rejects_nonfrozen_policy_values(
    field_name: str,
    changed_value: object,
) -> None:
    values = normalize_gate3_dividend(make_inputs()).model_dump(mode="python")
    values[field_name] = changed_value

    with pytest.raises(ValidationError):
        Gate3DividendNormalizationResult(**values)


def test_legacy_dividend_event_schema_remains_separate_and_unchanged() -> None:
    legacy = DividendEvent(
        security_id="NORGATE:1900000003",
        symbol="SYNTHC",
        entitlement_date=ENTITLEMENT_SESSION,
        date_semantics="entitlement_close",
        dividend_type="ordinary_cash",
        amount_per_share=0.25,
        currency="USD",
        source_provider="Norgate Data",
        source_asset_id=1_900_000_003,
        source_adjustment_mode="CAPITALSPECIAL",
    )

    assert set(DividendEvent.model_fields) == {
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
    assert type(legacy.amount_per_share) is float
    assert not isinstance(legacy, Gate3DividendNormalizationInputs)
