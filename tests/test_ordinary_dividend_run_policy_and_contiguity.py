"""Acceptance tests for ordinary-dividend Slice 3 run proof primitives."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import socket
from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import stock_swing_d1.data.ordinary_dividend_run_policy as run_policy_module
from stock_swing_d1.data.ordinary_dividend_accounting import (
    CANONICAL_NEXT_TRADING_SESSION,
)
from stock_swing_d1.data.ordinary_dividend_run_policy import (
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_HASH_DOMAIN,
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_ID,
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF,
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF_SCHEMA_VERSION,
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY,
    ORDINARY_DIVIDEND_ACCOUNTING_POLICY_VERSION,
    PROCESSED_SESSION_CONTIGUITY_HASH_DOMAIN,
    PROCESSED_SESSION_CONTIGUITY_PROOF_SCHEMA_VERSION,
    OrdinaryDividendAccountingPolicyRef,
    ProcessedSessionContiguityProof,
    ProcessedSessionLink,
    build_ordinary_dividend_accounting_policy_ref,
    build_processed_session_contiguity_proof,
    compute_ordinary_dividend_accounting_policy_fingerprint,
    ordinary_dividend_accounting_policy_manifest,
)


CALENDAR_SOURCE_ID = "accepted-us-equity-calendar"
CALENDAR_POLICY_ID = "canonical-next-session"
CALENDAR_POLICY_VERSION = "1"
UPSTREAM_CALENDAR_EVIDENCE_FINGERPRINT = "a" * 64
FRIDAY = date(2026, 8, 7)
MONDAY = date(2026, 8, 10)
TUESDAY = date(2026, 8, 11)


def canonical_domain_hash(domain: str, payload: object) -> str:
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_links(*sessions: date) -> tuple[ProcessedSessionLink, ...]:
    return tuple(
        ProcessedSessionLink(
            session=session,
            next_session=(
                None if index == len(sessions) - 1 else sessions[index + 1]
            ),
        )
        for index, session in enumerate(sessions)
    )


def make_proof(
    *,
    calendar_source_id: str = CALENDAR_SOURCE_ID,
    calendar_policy_id: str = CALENDAR_POLICY_ID,
    calendar_policy_version: str = CALENDAR_POLICY_VERSION,
    upstream_fingerprint: str = UPSTREAM_CALENDAR_EVIDENCE_FINGERPRINT,
    policy_ref: OrdinaryDividendAccountingPolicyRef = (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
    ),
    session_links: tuple[ProcessedSessionLink, ...] | None = None,
) -> ProcessedSessionContiguityProof:
    return build_processed_session_contiguity_proof(
        calendar_source_id=calendar_source_id,
        calendar_policy_id=calendar_policy_id,
        calendar_policy_version=calendar_policy_version,
        upstream_source_evidence_fingerprint=upstream_fingerprint,
        dividend_accounting_policy_ref=policy_ref,
        session_links=session_links or make_links(FRIDAY, MONDAY),
    )


def proof_values(
    proof: ProcessedSessionContiguityProof,
) -> dict[str, object]:
    return {
        field_name: getattr(proof, field_name)
        for field_name in ProcessedSessionContiguityProof.model_fields
    }


def expected_contiguity_payload(
    proof: ProcessedSessionContiguityProof,
) -> dict[str, object]:
    policy_ref = proof.dividend_accounting_policy_ref
    return {
        "schema_version": proof.schema_version,
        "calendar_source_id": proof.calendar_source_id,
        "calendar_policy_id": proof.calendar_policy_id,
        "calendar_policy_version": proof.calendar_policy_version,
        "resolution_semantics": proof.resolution_semantics,
        "upstream_source_evidence_fingerprint": (
            proof.upstream_source_evidence_fingerprint
        ),
        "dividend_accounting_policy_ref": {
            "schema_version": policy_ref.schema_version,
            "policy_id": policy_ref.policy_id,
            "policy_version": policy_ref.policy_version,
            "policy_semantic_identity": policy_ref.policy_semantic_identity,
            "policy_fingerprint": policy_ref.policy_fingerprint,
        },
        "processed_sessions": [
            link.session.isoformat() for link in proof.session_links
        ],
        "session_links": [
            {
                "session": link.session.isoformat(),
                "next_session": (
                    None
                    if link.next_session is None
                    else link.next_session.isoformat()
                ),
            }
            for link in proof.session_links
        ],
    }


def test_frozen_ordinary_dividend_run_policy_constructs() -> None:
    policy_ref = build_ordinary_dividend_accounting_policy_ref()

    assert policy_ref.schema_version == (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF_SCHEMA_VERSION
    )
    assert policy_ref.policy_id == ORDINARY_DIVIDEND_ACCOUNTING_POLICY_ID
    assert policy_ref.policy_version == (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_VERSION
    )
    assert policy_ref.policy_semantic_identity == (
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY
    )
    assert policy_ref.policy_semantic_identity == (
        "ORDINARY_DIVIDEND_ACCOUNTING_V0_1"
    )


def test_run_policy_fingerprint_uses_canonical_domain_envelope() -> None:
    expected = canonical_domain_hash(
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_HASH_DOMAIN,
        ordinary_dividend_accounting_policy_manifest(),
    )

    assert compute_ordinary_dividend_accounting_policy_fingerprint() == expected
    assert ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF.policy_fingerprint == expected


def test_identical_run_policy_reconstruction_reproduces_fingerprint() -> None:
    first = build_ordinary_dividend_accounting_policy_ref()
    second = OrdinaryDividendAccountingPolicyRef(
        **first.model_dump(mode="python")
    )

    assert first == second
    assert first.policy_fingerprint == second.policy_fingerprint


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("policy_version", "0.2"),
        ("policy_semantic_identity", "ORDINARY_DIVIDEND_ACCOUNTING_V0_2"),
        ("activation_scope", "per_session"),
        ("event_presence_inference", "allowed"),
    ],
)
def test_synthetic_policy_semantic_change_changes_fingerprint(
    field_name: str,
    changed_value: str,
) -> None:
    baseline = ordinary_dividend_accounting_policy_manifest()
    changed = dict(baseline)
    changed[field_name] = changed_value

    assert canonical_domain_hash(
        ORDINARY_DIVIDEND_ACCOUNTING_POLICY_HASH_DOMAIN, changed
    ) != compute_ordinary_dividend_accounting_policy_fingerprint()


def test_production_policy_builder_exposes_no_free_form_configuration() -> None:
    assert inspect.signature(
        build_ordinary_dividend_accounting_policy_ref
    ).parameters == {}
    changed_manifest = ordinary_dividend_accounting_policy_manifest()
    changed_manifest["activation_scope"] = "per_session"

    assert (
        ordinary_dividend_accounting_policy_manifest()["activation_scope"]
        == "run_level"
    )
    assert (
        build_ordinary_dividend_accounting_policy_ref()
        == ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
    )


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("policy_id", ""),
        ("policy_version", ""),
        ("policy_semantic_identity", ""),
        ("policy_fingerprint", "A" * 64),
    ],
)
def test_empty_or_malformed_policy_identity_fails(
    field_name: str,
    invalid: str,
) -> None:
    values = ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF.model_dump(mode="python")
    values[field_name] = invalid

    with pytest.raises(ValidationError):
        OrdinaryDividendAccountingPolicyRef(**values)


def test_policy_ref_is_frozen_and_forbids_extra_fields() -> None:
    policy_ref = build_ordinary_dividend_accounting_policy_ref()

    with pytest.raises(ValidationError, match="frozen_instance"):
        policy_ref.policy_version = "0.2"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        OrdinaryDividendAccountingPolicyRef(
            **policy_ref.model_dump(mode="python"),
            unexpected="extra",
        )


def test_policy_ref_rejects_stale_fingerprint() -> None:
    values = ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF.model_dump(mode="python")
    values["policy_fingerprint"] = "f" * 64

    with pytest.raises(ValidationError, match="does not match policy semantics"):
        OrdinaryDividendAccountingPolicyRef(**values)


@pytest.mark.parametrize(
    "sessions",
    [(FRIDAY, MONDAY), (FRIDAY, MONDAY, TUESDAY)],
)
def test_valid_two_and_three_session_proofs_construct(
    sessions: tuple[date, ...],
) -> None:
    links = make_links(*sessions)

    proof = make_proof(session_links=links)

    assert proof.session_links == links
    assert proof.processed_sessions == sessions
    assert proof.session_links[-1].next_session is None
    assert all(
        link.next_session is not None for link in proof.session_links[:-1]
    )


def test_identical_contiguity_proof_reproduces_fingerprint() -> None:
    first = make_proof(session_links=make_links(FRIDAY, MONDAY, TUESDAY))
    second = make_proof(session_links=make_links(FRIDAY, MONDAY, TUESDAY))

    assert first == second
    assert (
        first.processed_session_contiguity_fingerprint
        == second.processed_session_contiguity_fingerprint
    )


def test_contiguity_fingerprint_binds_exact_canonical_payload() -> None:
    proof = make_proof(session_links=make_links(FRIDAY, MONDAY, TUESDAY))

    assert proof.processed_session_contiguity_fingerprint == (
        canonical_domain_hash(
            PROCESSED_SESSION_CONTIGUITY_HASH_DOMAIN,
            expected_contiguity_payload(proof),
        )
    )


def test_empty_processed_session_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_processed_session_contiguity_proof(
            calendar_source_id=CALENDAR_SOURCE_ID,
            calendar_policy_id=CALENDAR_POLICY_ID,
            calendar_policy_version=CALENDAR_POLICY_VERSION,
            upstream_source_evidence_fingerprint=(
                UPSTREAM_CALENDAR_EVIDENCE_FINGERPRINT
            ),
            dividend_accounting_policy_ref=(
                ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
            ),
            session_links=(),
        )


def test_duplicate_session_is_rejected() -> None:
    links = (
        ProcessedSessionLink(session=FRIDAY, next_session=FRIDAY),
        ProcessedSessionLink(session=FRIDAY, next_session=None),
    )

    with pytest.raises(ValueError, match="unique"):
        make_proof(session_links=links)


def test_descending_or_out_of_order_sessions_are_rejected_without_sorting() -> None:
    links = (
        ProcessedSessionLink(session=MONDAY, next_session=FRIDAY),
        ProcessedSessionLink(session=FRIDAY, next_session=None),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        make_proof(session_links=links)


def test_missing_nonfinal_next_session_is_rejected() -> None:
    links = (
        ProcessedSessionLink(session=FRIDAY, next_session=None),
        ProcessedSessionLink(session=MONDAY, next_session=None),
    )

    with pytest.raises(ValueError, match="must have next_session"):
        make_proof(session_links=links)


def test_nonfinal_next_session_must_equal_following_supplied_session() -> None:
    links = (
        ProcessedSessionLink(session=FRIDAY, next_session=TUESDAY),
        ProcessedSessionLink(session=MONDAY, next_session=None),
    )

    with pytest.raises(ValueError, match="immediately following"):
        make_proof(session_links=links)


def test_final_session_must_have_none_next_session() -> None:
    links = (
        ProcessedSessionLink(session=FRIDAY, next_session=MONDAY),
        ProcessedSessionLink(session=MONDAY, next_session=TUESDAY),
    )

    with pytest.raises(ValueError, match="final processed session"):
        make_proof(session_links=links)


def test_session_links_must_be_exact_tuple_and_are_not_repaired() -> None:
    with pytest.raises((ValidationError, ValueError), match="immutable tuple"):
        build_processed_session_contiguity_proof(
            calendar_source_id=CALENDAR_SOURCE_ID,
            calendar_policy_id=CALENDAR_POLICY_ID,
            calendar_policy_version=CALENDAR_POLICY_VERSION,
            upstream_source_evidence_fingerprint=(
                UPSTREAM_CALENDAR_EVIDENCE_FINGERPRINT
            ),
            dividend_accounting_policy_ref=(
                ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
            ),
            session_links=list(make_links(FRIDAY, MONDAY)),  # type: ignore[arg-type]
        )


def test_non_link_items_are_not_coerced_into_links() -> None:
    with pytest.raises((ValidationError, ValueError), match="exact Processed"):
        make_proof(
            session_links=(
                {"session": FRIDAY, "next_session": None},  # type: ignore[arg-type]
            )
        )


def test_calendar_day_gap_is_preserved_without_inference_or_repair() -> None:
    proof = make_proof(session_links=make_links(FRIDAY, TUESDAY))

    assert proof.processed_sessions == (FRIDAY, TUESDAY)
    assert proof.session_links[0].next_session == TUESDAY


@pytest.mark.parametrize(
    "field_name",
    ["calendar_source_id", "calendar_policy_id", "calendar_policy_version"],
)
@pytest.mark.parametrize("invalid", ["", " whitespace "])
def test_calendar_source_and_policy_identity_are_required(
    field_name: str,
    invalid: str,
) -> None:
    kwargs = {field_name: invalid}

    with pytest.raises((ValidationError, ValueError), match="canonical non-empty"):
        make_proof(**kwargs)


def test_wrong_resolution_semantic_is_rejected() -> None:
    proof = make_proof()
    values = proof_values(proof)
    values["resolution_semantics"] = "CALENDAR_DAY_PLUS_ONE"

    with pytest.raises(ValidationError, match="CANONICAL_NEXT_TRADING_SESSION"):
        ProcessedSessionContiguityProof(**values)


@pytest.mark.parametrize("invalid", ["", "A" * 64, "a" * 63, "z" * 64])
def test_upstream_calendar_evidence_fingerprint_is_required_and_strict(
    invalid: str,
) -> None:
    with pytest.raises((ValidationError, ValueError), match="SHA-256"):
        make_proof(upstream_fingerprint=invalid)


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("calendar_source_id", "other-calendar-source"),
        ("calendar_policy_id", "other-next-session-policy"),
        ("calendar_policy_version", "2"),
        ("upstream_fingerprint", "b" * 64),
    ],
)
def test_each_external_calendar_binding_changes_contiguity_fingerprint(
    field_name: str,
    changed_value: str,
) -> None:
    baseline = make_proof()
    changed = make_proof(**{field_name: changed_value})

    assert changed.processed_session_contiguity_fingerprint != (
        baseline.processed_session_contiguity_fingerprint
    )


def test_changing_one_processed_session_changes_fingerprint() -> None:
    baseline = make_proof(session_links=make_links(FRIDAY, MONDAY))
    changed = make_proof(session_links=make_links(FRIDAY, TUESDAY))

    assert changed.processed_session_contiguity_fingerprint != (
        baseline.processed_session_contiguity_fingerprint
    )


def test_next_session_and_final_none_fact_are_bound_by_fingerprint() -> None:
    proof = make_proof()
    baseline_payload = expected_contiguity_payload(proof)
    changed_payload = json.loads(json.dumps(baseline_payload))
    changed_payload["session_links"][0]["next_session"] = (
        TUESDAY.isoformat()
    )

    assert canonical_domain_hash(
        PROCESSED_SESSION_CONTIGUITY_HASH_DOMAIN, changed_payload
    ) != proof.processed_session_contiguity_fingerprint
    assert baseline_payload["session_links"][-1]["next_session"] is None


def test_reversing_session_order_fails_instead_of_normalizing() -> None:
    reversed_links = make_links(MONDAY, FRIDAY)

    with pytest.raises(ValueError, match="strictly increasing"):
        make_proof(session_links=reversed_links)


def test_local_structural_equality_without_upstream_proof_is_insufficient() -> None:
    structurally_valid = make_links(FRIDAY, MONDAY)

    with pytest.raises((ValidationError, ValueError), match="SHA-256"):
        make_proof(
            session_links=structurally_valid,
            upstream_fingerprint="",
        )


def test_proof_construction_makes_no_network_call(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", fail_if_called)

    assert make_proof().processed_sessions == (FRIDAY, MONDAY)


def test_module_has_upstream_only_imports_and_no_calendar_day_resolver() -> None:
    source = Path(run_policy_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    forbidden_roots = {
        "backtester",
        "backtest_results",
        "portfolio",
        "research_metrics",
        "norgatedata",
        "requests",
        "socket",
        "urllib",
    }
    imported_roots = {
        module.removeprefix("stock_swing_d1.").split(".", 1)[0]
        for module in imported_modules
    }
    used_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert imported_roots.isdisjoint(forbidden_roots)
    assert "timedelta" not in used_names


def test_contiguity_proof_is_bound_to_frozen_dividend_policy() -> None:
    proof = make_proof(policy_ref=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF)

    assert (
        proof.dividend_accounting_policy_ref
        == ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF
    )


def test_mismatched_or_malformed_policy_ref_cannot_masquerade() -> None:
    mismatched = OrdinaryDividendAccountingPolicyRef.model_construct(
        schema_version=(
            ORDINARY_DIVIDEND_ACCOUNTING_POLICY_REF_SCHEMA_VERSION
        ),
        policy_id=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_ID,
        policy_version=ORDINARY_DIVIDEND_ACCOUNTING_POLICY_VERSION,
        policy_semantic_identity=(
            ORDINARY_DIVIDEND_ACCOUNTING_POLICY_SEMANTIC_IDENTITY
        ),
        policy_fingerprint="f" * 64,
    )

    with pytest.raises((ValidationError, ValueError), match="policy semantics"):
        make_proof(policy_ref=mismatched)


def test_policy_reference_is_semantically_bound_in_contiguity_hash() -> None:
    proof = make_proof()
    baseline_payload = expected_contiguity_payload(proof)
    changed_payload = json.loads(json.dumps(baseline_payload))
    changed_payload["dividend_accounting_policy_ref"][
        "policy_fingerprint"
    ] = "b" * 64

    assert canonical_domain_hash(
        PROCESSED_SESSION_CONTIGUITY_HASH_DOMAIN, changed_payload
    ) != proof.processed_session_contiguity_fingerprint


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("session", datetime(2026, 8, 7)),
        ("session", "2026-08-07"),
        ("next_session", datetime(2026, 8, 10)),
        ("next_session", "2026-08-10"),
    ],
)
def test_session_links_require_exact_python_dates(
    field_name: str,
    invalid: object,
) -> None:
    values = {"session": FRIDAY, "next_session": MONDAY}
    values[field_name] = invalid

    with pytest.raises(ValidationError, match="datetime.date"):
        ProcessedSessionLink(**values)


def test_link_and_proof_models_are_frozen() -> None:
    link = make_links(FRIDAY)[0]
    proof = make_proof()

    with pytest.raises(ValidationError, match="frozen_instance"):
        link.next_session = MONDAY
    with pytest.raises(ValidationError, match="frozen_instance"):
        proof.calendar_policy_version = "2"


def test_link_and_proof_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ProcessedSessionLink(
            session=FRIDAY,
            next_session=None,
            unexpected="extra",
        )

    proof = make_proof()
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ProcessedSessionContiguityProof(
            **proof_values(proof),
            unexpected="extra",
        )


def test_contiguity_proof_rejects_stale_or_malformed_fingerprint() -> None:
    proof = make_proof()
    values = proof_values(proof)
    values["processed_session_contiguity_fingerprint"] = "b" * 64

    with pytest.raises(ValidationError, match="does not match proof content"):
        ProcessedSessionContiguityProof(**values)

    values["processed_session_contiguity_fingerprint"] = "B" * 64
    with pytest.raises(ValidationError, match="SHA-256"):
        ProcessedSessionContiguityProof(**values)


def test_schema_and_resolution_semantics_are_frozen() -> None:
    proof = make_proof()

    assert proof.schema_version == (
        PROCESSED_SESSION_CONTIGUITY_PROOF_SCHEMA_VERSION
    )
    assert proof.resolution_semantics == CANONICAL_NEXT_TRADING_SESSION

    values = proof_values(proof)
    values["schema_version"] = "processed_session_contiguity_proof.v9"
    with pytest.raises(ValidationError):
        ProcessedSessionContiguityProof(**values)
