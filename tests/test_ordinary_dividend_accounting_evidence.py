# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
"""Acceptance tests for ordinary-dividend accounting evidence Slice 2."""

from __future__ import annotations

import decimal
import hashlib
import json
from datetime import date, datetime
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CALENDAR_RESOLUTION_PROOF_HASH_DOMAIN,
    CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION,
    CANONICAL_DIVIDEND_CURRENCY,
    CANONICAL_NEXT_TRADING_SESSION,
    CLASSIFICATION_PROOF_HASH_DOMAIN,
    DIVIDEND_CALENDAR_RESOLUTION_PROOF_SCHEMA_VERSION,
    ORDINARY_CASH_CLASSIFICATION_PROOF_SCHEMA_VERSION,
    ORDINARY_CASH_DISTRIBUTION_TYPE,
    CanonicalDividendAccountingEvidence,
    DividendCalendarResolutionProof,
    OrdinaryCashClassificationProof,
    build_canonical_dividend_accounting_evidence,
    build_dividend_calendar_resolution_proof,
    build_ordinary_cash_classification_proof,
)
from stock_swing_d1.data.ordinary_dividend_normalization import (
    HISTORICAL_SHARE_AMOUNT_BASIS,
    NORMALIZATION_ARITHMETIC_MODE,
    NORMALIZATION_METHOD_ID,
    NORMALIZATION_ROUNDING_MODE,
    NORMALIZATION_SCALE,
    Gate3DividendNormalizationInputs,
    normalize_gate3_dividend,
)
from stock_swing_d1.models import DividendEvent


EVENT_ID = "DISTRIBUTION:NORGATE:1900000003:2026-08-10:1"
SECURITY_ID = "NORGATE:1900000003"
ENTITLEMENT_SESSION = date(2026, 8, 7)
EX_SESSION = date(2026, 8, 10)
CLASSIFICATION_CONTRACT_ID = "ordinary-cash-classification.v1"
CALENDAR_SOURCE_ID = "accepted-us-equity-calendar"
CALENDAR_POLICY_ID = "canonical-next-session"
CALENDAR_POLICY_VERSION = "1"
CLASSIFICATION_SOURCE_FINGERPRINT = "a" * 64
CALENDAR_SOURCE_FINGERPRINT = "b" * 64
SNAPSHOT_FINGERPRINT = "c" * 64


def make_normalization(
    *,
    entitlement_session: date = ENTITLEMENT_SESSION,
    d_capitalspecial: Decimal = Decimal("2"),
    unadjusted_close_t: Decimal = Decimal("3"),
    close_capital_t: Decimal = Decimal("4"),
):
    inputs = Gate3DividendNormalizationInputs(
        entitlement_session=entitlement_session,
        d_capitalspecial=d_capitalspecial,
        unadjusted_close_t=unadjusted_close_t,
        close_capital_t=close_capital_t,
    )
    return inputs, normalize_gate3_dividend(inputs)


def make_classification(
    *,
    event_id: str = EVENT_ID,
    contract_id: str = CLASSIFICATION_CONTRACT_ID,
    source_fingerprint: str = CLASSIFICATION_SOURCE_FINGERPRINT,
) -> OrdinaryCashClassificationProof:
    return build_ordinary_cash_classification_proof(
        canonical_distribution_event_id=event_id,
        classification_contract_id=contract_id,
        upstream_source_evidence_fingerprint=source_fingerprint,
    )


def make_calendar(
    *,
    entitlement_session: date = ENTITLEMENT_SESSION,
    ex_session: date = EX_SESSION,
    source_id: str = CALENDAR_SOURCE_ID,
    policy_id: str = CALENDAR_POLICY_ID,
    policy_version: str = CALENDAR_POLICY_VERSION,
    source_fingerprint: str = CALENDAR_SOURCE_FINGERPRINT,
) -> DividendCalendarResolutionProof:
    return build_dividend_calendar_resolution_proof(
        calendar_source_id=source_id,
        entitlement_session=entitlement_session,
        ex_session=ex_session,
        calendar_policy_id=policy_id,
        calendar_policy_version=policy_version,
        upstream_source_evidence_fingerprint=source_fingerprint,
    )


def make_evidence(
    *,
    event_id: str = EVENT_ID,
    security_id: str = SECURITY_ID,
    normalization_inputs=None,
    normalization_result=None,
    classification_proof=None,
    calendar_proof=None,
    currency: str = CANONICAL_DIVIDEND_CURRENCY,
    snapshot_fingerprint: str = SNAPSHOT_FINGERPRINT,
    evidence_schema_version: str = (
        CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION
    ),
) -> CanonicalDividendAccountingEvidence:
    if normalization_inputs is None or normalization_result is None:
        default_inputs, default_result = make_normalization()
        normalization_inputs = (
            default_inputs if normalization_inputs is None else normalization_inputs
        )
        normalization_result = (
            default_result if normalization_result is None else normalization_result
        )
    return build_canonical_dividend_accounting_evidence(
        canonical_distribution_event_id=event_id,
        canonical_security_id=security_id,
        normalization_inputs=normalization_inputs,
        normalization_result=normalization_result,
        classification_proof=classification_proof or make_classification(),
        calendar_resolution_proof=calendar_proof or make_calendar(),
        currency=currency,
        canonical_distribution_snapshot_fingerprint=snapshot_fingerprint,
        evidence_schema_version=evidence_schema_version,
    )


def replace_model_field(model, field_name: str, value: object):
    values = model.model_dump(mode="python")
    values[field_name] = value
    return type(model)(**values)


def canonical_domain_hash(domain: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def legacy_dividend() -> DividendEvent:
    return DividendEvent(
        security_id=SECURITY_ID,
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


def test_legacy_dividend_schema_is_unchanged_and_new_evidence_is_distinct() -> None:
    legacy = legacy_dividend()

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
    assert not isinstance(legacy, CanonicalDividendAccountingEvidence)


def test_accounting_builder_rejects_legacy_dividend_as_normalization() -> None:
    inputs, _ = make_normalization()

    with pytest.raises(TypeError, match="Gate3DividendNormalizationResult"):
        make_evidence(
            normalization_inputs=inputs,
            normalization_result=legacy_dividend(),
        )


def test_valid_ordinary_cash_classification_proof_constructs() -> None:
    proof = make_classification()

    assert proof.schema_version == (
        ORDINARY_CASH_CLASSIFICATION_PROOF_SCHEMA_VERSION
    )
    assert proof.canonical_distribution_event_id == EVENT_ID
    assert proof.classification_contract_id == CLASSIFICATION_CONTRACT_ID
    assert proof.classification_result == ORDINARY_CASH_DISTRIBUTION_TYPE
    assert proof.upstream_source_evidence_fingerprint == (
        CLASSIFICATION_SOURCE_FINGERPRINT
    )


@pytest.mark.parametrize("invalid", ["", " whitespace "])
def test_classification_contract_id_must_be_canonical_nonempty_text(
    invalid: str,
) -> None:
    with pytest.raises((ValidationError, ValueError), match="canonical non-empty"):
        make_classification(contract_id=invalid)


@pytest.mark.parametrize("invalid", ["", "A" * 64, "a" * 63, "z" * 64])
def test_classification_requires_valid_upstream_source_fingerprint(
    invalid: str,
) -> None:
    with pytest.raises((ValidationError, ValueError), match="SHA-256"):
        make_classification(source_fingerprint=invalid)


def test_nonordinary_classification_proof_is_rejected() -> None:
    values = make_classification().model_dump(mode="python")
    values["classification_result"] = "special_cash"

    with pytest.raises(ValidationError, match="ordinary_cash"):
        OrdinaryCashClassificationProof(**values)


def test_classification_proof_for_different_event_fails_evidence() -> None:
    proof = make_classification(event_id=f"{EVENT_ID}:OTHER")

    with pytest.raises(ValidationError, match="identify the accounting event"):
        make_evidence(classification_proof=proof)


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("event_id", f"{EVENT_ID}:OTHER"),
        ("contract_id", "ordinary-cash-classification.v2"),
        ("source_fingerprint", "d" * 64),
    ],
)
def test_each_classification_proof_input_changes_fingerprint(
    changed_field: str,
    changed_value: str,
) -> None:
    baseline = make_classification()
    changed = make_classification(**{changed_field: changed_value})

    assert changed.classification_proof_fingerprint != (
        baseline.classification_proof_fingerprint
    )


def test_equivalent_classification_proof_reproduces_fingerprint() -> None:
    assert make_classification() == make_classification()


def test_classification_fingerprint_uses_canonical_domain_envelope() -> None:
    proof = make_classification()
    expected = canonical_domain_hash(
        CLASSIFICATION_PROOF_HASH_DOMAIN,
        {
            "schema_version": proof.schema_version,
            "canonical_distribution_event_id": EVENT_ID,
            "classification_contract_id": CLASSIFICATION_CONTRACT_ID,
            "classification_result": ORDINARY_CASH_DISTRIBUTION_TYPE,
            "upstream_source_evidence_fingerprint": (
                CLASSIFICATION_SOURCE_FINGERPRINT
            ),
        },
    )

    assert proof.classification_proof_fingerprint == expected


def test_classification_proof_rejects_stale_fingerprint() -> None:
    proof = make_classification()

    with pytest.raises(ValidationError, match="does not match proof content"):
        replace_model_field(proof, "classification_contract_id", "changed")


def test_classification_proof_is_frozen_and_forbids_extra_fields() -> None:
    proof = make_classification()

    with pytest.raises(ValidationError, match="frozen_instance"):
        proof.classification_contract_id = "changed"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        OrdinaryCashClassificationProof(
            **proof.model_dump(mode="python"),
            unexpected="extra",
        )


def test_valid_calendar_resolution_proof_constructs_for_t_before_x() -> None:
    proof = make_calendar()

    assert proof.schema_version == (
        DIVIDEND_CALENDAR_RESOLUTION_PROOF_SCHEMA_VERSION
    )
    assert proof.entitlement_session == ENTITLEMENT_SESSION
    assert proof.ex_session == EX_SESSION
    assert proof.resolution_semantics == CANONICAL_NEXT_TRADING_SESSION


@pytest.mark.parametrize(
    ("entitlement_session", "ex_session"),
    [
        (ENTITLEMENT_SESSION, ENTITLEMENT_SESSION),
        (EX_SESSION, ENTITLEMENT_SESSION),
    ],
)
def test_calendar_resolution_requires_t_strictly_before_x(
    entitlement_session: date,
    ex_session: date,
) -> None:
    with pytest.raises((ValidationError, ValueError), match="before ex_session"):
        make_calendar(
            entitlement_session=entitlement_session,
            ex_session=ex_session,
        )


@pytest.mark.parametrize("field_name", ["source_id", "policy_id", "policy_version"])
@pytest.mark.parametrize("invalid", ["", " whitespace "])
def test_calendar_identity_and_policy_fields_are_required(
    field_name: str,
    invalid: str,
) -> None:
    with pytest.raises((ValidationError, ValueError), match="canonical non-empty"):
        make_calendar(**{field_name: invalid})


def test_wrong_calendar_resolution_semantic_is_rejected() -> None:
    values = make_calendar().model_dump(mode="python")
    values["resolution_semantics"] = "CALENDAR_DAY_PLUS_ONE"

    with pytest.raises(ValidationError, match="CANONICAL_NEXT_TRADING_SESSION"):
        DividendCalendarResolutionProof(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("entitlement_session", datetime(2026, 8, 7)),
        ("entitlement_session", "2026-08-07"),
        ("ex_session", datetime(2026, 8, 10)),
        ("ex_session", "2026-08-10"),
    ],
)
def test_calendar_sessions_require_exact_date(
    field_name: str,
    invalid: object,
) -> None:
    values = make_calendar().model_dump(mode="python")
    values[field_name] = invalid

    with pytest.raises(ValidationError, match="datetime.date"):
        DividendCalendarResolutionProof(**values)


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("entitlement_session", date(2026, 8, 6)),
        ("ex_session", date(2026, 8, 11)),
        ("source_id", "other-accepted-calendar"),
        ("policy_id", "other-next-session-policy"),
        ("policy_version", "2"),
        ("source_fingerprint", "e" * 64),
    ],
)
def test_each_calendar_proof_input_changes_fingerprint(
    changed_field: str,
    changed_value: object,
) -> None:
    baseline = make_calendar()
    changed = make_calendar(**{changed_field: changed_value})

    assert changed.calendar_resolution_fingerprint != (
        baseline.calendar_resolution_fingerprint
    )


def test_equivalent_calendar_proof_reproduces_fingerprint() -> None:
    assert make_calendar() == make_calendar()


def test_calendar_fingerprint_uses_canonical_domain_envelope() -> None:
    proof = make_calendar()
    expected = canonical_domain_hash(
        CALENDAR_RESOLUTION_PROOF_HASH_DOMAIN,
        {
            "schema_version": proof.schema_version,
            "calendar_source_id": CALENDAR_SOURCE_ID,
            "entitlement_session": ENTITLEMENT_SESSION.isoformat(),
            "ex_session": EX_SESSION.isoformat(),
            "calendar_policy_id": CALENDAR_POLICY_ID,
            "calendar_policy_version": CALENDAR_POLICY_VERSION,
            "resolution_semantics": CANONICAL_NEXT_TRADING_SESSION,
            "upstream_source_evidence_fingerprint": CALENDAR_SOURCE_FINGERPRINT,
        },
    )

    assert proof.calendar_resolution_fingerprint == expected


def test_calendar_proof_rejects_stale_fingerprint() -> None:
    proof = make_calendar()

    with pytest.raises(ValidationError, match="does not match proof content"):
        replace_model_field(proof, "calendar_policy_version", "2")


def test_calendar_proof_is_frozen_and_forbids_extra_fields() -> None:
    proof = make_calendar()

    with pytest.raises(ValidationError, match="frozen_instance"):
        proof.calendar_policy_version = "2"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DividendCalendarResolutionProof(
            **proof.model_dump(mode="python"),
            unexpected="extra",
        )


def test_valid_canonical_accounting_evidence_composes_slice1_and_proofs() -> None:
    inputs, result = make_normalization()
    classification = make_classification()
    calendar = make_calendar()

    evidence = make_evidence(
        normalization_inputs=inputs,
        normalization_result=result,
        classification_proof=classification,
        calendar_proof=calendar,
    )

    assert evidence.evidence_schema_version == (
        CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION
    )
    assert evidence.canonical_distribution_event_id == EVENT_ID
    assert evidence.canonical_security_id == SECURITY_ID
    assert evidence.entitlement_session == inputs.entitlement_session
    assert evidence.ex_session == calendar.ex_session
    assert evidence.distribution_type == ORDINARY_CASH_DISTRIBUTION_TYPE
    assert evidence.amount_per_share == result.d_h
    assert evidence.amount_per_share.as_tuple().exponent == -38
    assert evidence.amount_basis == HISTORICAL_SHARE_AMOUNT_BASIS
    assert evidence.normalization_method_id == NORMALIZATION_METHOD_ID
    assert evidence.normalization_scale == NORMALIZATION_SCALE
    assert evidence.normalization_rounding_mode == NORMALIZATION_ROUNDING_MODE
    assert evidence.normalization_arithmetic_mode == NORMALIZATION_ARITHMETIC_MODE
    assert evidence.normalization_inputs_fingerprint == (
        result.normalization_inputs_fingerprint
    )
    assert evidence.normalization_inputs == inputs
    assert evidence.normalization_result == result
    assert evidence.classification_proof == classification
    assert evidence.calendar_resolution_proof == calendar
    assert evidence.calendar_resolution_fingerprint == (
        calendar.calendar_resolution_fingerprint
    )
    assert evidence.currency == CANONICAL_DIVIDEND_CURRENCY
    assert evidence.canonical_distribution_snapshot_fingerprint == (
        SNAPSHOT_FINGERPRINT
    )


def test_builder_derives_d_h_instead_of_accepting_raw_capitalspecial() -> None:
    inputs, result = make_normalization(
        d_capitalspecial=Decimal("2"),
        unadjusted_close_t=Decimal("3"),
        close_capital_t=Decimal("4"),
    )

    evidence = make_evidence(
        normalization_inputs=inputs,
        normalization_result=result,
    )

    assert evidence.amount_per_share == Decimal(
        "1.50000000000000000000000000000000000000"
    )
    assert evidence.amount_per_share != inputs.d_capitalspecial


@pytest.mark.parametrize(
    ("field_name", "wrong_value", "message"),
    [
        (
            "amount_per_share",
            Decimal("2.00000000000000000000000000000000000000"),
            "normalized D_H",
        ),
        ("amount_basis", "RAW_CAPITALSPECIAL", "HISTORICAL_SHARE_BASIS_D_H"),
        (
            "normalization_method_id",
            "OTHER_METHOD",
            "GATE3_D_H_EXACT_RATIONAL_V0_1",
        ),
        ("normalization_scale", 37, "38"),
        ("normalization_rounding_mode", "ROUND_DOWN", "ROUND_HALF_EVEN"),
        (
            "normalization_arithmetic_mode",
            "DECIMAL_CONTEXT",
            "EXACT_RATIONAL",
        ),
        ("normalization_inputs_fingerprint", "f" * 64, "normalization"),
        ("calendar_resolution_fingerprint", "f" * 64, "calendar"),
    ],
)
def test_direct_evidence_tampering_fails_closed(
    field_name: str,
    wrong_value: object,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        replace_model_field(make_evidence(), field_name, wrong_value)


def test_evidence_rejects_different_normalization_t() -> None:
    other_inputs, other_result = make_normalization(
        entitlement_session=date(2026, 8, 6)
    )
    values = make_evidence().model_dump(mode="python")
    values["normalization_inputs"] = other_inputs
    values["normalization_result"] = other_result

    with pytest.raises(ValidationError, match="accounting T"):
        CanonicalDividendAccountingEvidence(**values)


def test_evidence_rejects_different_calendar_t() -> None:
    proof = make_calendar(entitlement_session=date(2026, 8, 6))
    values = make_evidence().model_dump(mode="python")
    values["calendar_resolution_proof"] = proof
    values["calendar_resolution_fingerprint"] = (
        proof.calendar_resolution_fingerprint
    )

    with pytest.raises(ValidationError, match="must equal T"):
        CanonicalDividendAccountingEvidence(**values)


def test_evidence_rejects_different_calendar_x() -> None:
    proof = make_calendar(ex_session=date(2026, 8, 11))
    values = make_evidence().model_dump(mode="python")
    values["calendar_resolution_proof"] = proof
    values["calendar_resolution_fingerprint"] = (
        proof.calendar_resolution_fingerprint
    )

    with pytest.raises(ValidationError, match="must equal X"):
        CanonicalDividendAccountingEvidence(**values)


def test_evidence_rejects_result_not_bound_to_retained_inputs() -> None:
    other_inputs, other_result = make_normalization(
        d_capitalspecial=Decimal("3")
    )
    values = make_evidence().model_dump(mode="python")
    values["normalization_result"] = other_result

    with pytest.raises(ValidationError, match="must bind normalization inputs"):
        CanonicalDividendAccountingEvidence(**values)
    assert other_inputs != values["normalization_inputs"]


@pytest.mark.parametrize("invalid", ["", "C" * 64, "c" * 63, "z" * 64])
def test_distribution_snapshot_fingerprint_is_required_and_strict(
    invalid: str,
) -> None:
    with pytest.raises((ValidationError, ValueError), match="SHA-256"):
        make_evidence(snapshot_fingerprint=invalid)


@pytest.mark.parametrize("invalid", [0.25, "0.25", 1, True])
def test_authoritative_evidence_amount_rejects_non_decimal_input(
    invalid: object,
) -> None:
    with pytest.raises(ValidationError, match="exact Decimal"):
        replace_model_field(make_evidence(), "amount_per_share", invalid)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("canonical_security_id", "SECURITY:1900000003"),
        ("canonical_security_id", "NORGATE:0"),
        ("currency", "EUR"),
        ("evidence_schema_version", "canonical_dividend_accounting_evidence.v9"),
    ],
)
def test_evidence_identity_currency_and_schema_are_frozen(
    field_name: str,
    invalid: str,
) -> None:
    with pytest.raises(ValidationError):
        replace_model_field(make_evidence(), field_name, invalid)


def test_accounting_evidence_is_frozen_and_forbids_extra_fields() -> None:
    evidence = make_evidence()

    with pytest.raises(ValidationError, match="frozen_instance"):
        evidence.currency = "USD"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CanonicalDividendAccountingEvidence(
            **evidence.model_dump(mode="python"),
            unexpected="extra",
        )


def test_equivalent_canonical_reconstruction_is_identical() -> None:
    evidence = make_evidence()

    reconstructed = CanonicalDividendAccountingEvidence(
        **evidence.model_dump(mode="python")
    )

    assert reconstructed == evidence
    assert (
        reconstructed.classification_proof.classification_proof_fingerprint
        == evidence.classification_proof.classification_proof_fingerprint
    )
    assert (
        reconstructed.calendar_resolution_fingerprint
        == evidence.calendar_resolution_fingerprint
    )


def test_evidence_is_identical_at_decimal_precision_28_and_60() -> None:
    inputs, result = make_normalization(
        d_capitalspecial=Decimal("1"),
        unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("7"),
    )

    with localcontext() as context:
        context.prec = 28
        precision_28 = make_evidence(
            normalization_inputs=inputs,
            normalization_result=result,
        )
    with localcontext() as context:
        context.prec = 60
        precision_60 = make_evidence(
            normalization_inputs=inputs,
            normalization_result=result,
        )

    assert precision_28 == precision_60
    assert precision_28.model_dump_json() == precision_60.model_dump_json()
    assert precision_28.amount_per_share == Decimal(
        "0.14285714285714285714285714285714285714"
    )


def test_proof_and_evidence_construction_does_not_read_decimal_context(
    monkeypatch,
) -> None:
    inputs, result = make_normalization()

    def fail_if_called():
        raise AssertionError("ambient decimal.getcontext() was read")

    monkeypatch.setattr(decimal, "getcontext", fail_if_called)

    evidence = make_evidence(
        normalization_inputs=inputs,
        normalization_result=result,
    )

    assert evidence.amount_per_share == result.d_h


def test_evidence_construction_does_not_mutate_decimal_precision() -> None:
    before = decimal.getcontext().prec

    make_evidence()

    assert decimal.getcontext().prec == before
