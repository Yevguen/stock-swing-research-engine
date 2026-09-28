from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from stock_swing_d1.backtest_results import (
    HistoricalBacktestValuationMark,
    HistoricalBacktestValuationPolicy,
    HistoricalBacktestValuationSnapshot,
    build_valuation_policy_ref,
    build_valuation_snapshot,
    semantic_domain_sha256,
)


def _mark(artifact_ref_factory, security_id: str, *, session=date(2026, 8, 24), close=Decimal("100")):
    return HistoricalBacktestValuationMark(
        security_id=security_id,
        session=session,
        close=close,
        source_artifact_ref=artifact_ref_factory(),
    )


def test_valuation_policy_constants_frozen_and_ref_deterministic():
    policy = HistoricalBacktestValuationPolicy()
    assert policy.model_dump() == {
        "schema_version": "historical_backtest_valuation_policy.v0.1",
        "policy_id": "completed_session_unadjusted_close_mark_v0.1",
        "currency": "USD",
        "timeframe": "D1",
        "session_type": "regular",
        "price_basis": "unadjusted",
        "mark_field": "close",
        "pending_settlement_valuation": "face_value",
        "open_position_valuation": "quantity_times_close",
    }
    assert build_valuation_policy_ref(policy) == build_valuation_policy_ref(policy)
    with pytest.raises(ValidationError):
        HistoricalBacktestValuationPolicy(currency="EUR")
    with pytest.raises(ValidationError):
        policy.currency = "EUR"


def test_valuation_policy_domain_hash_changes_for_synthetic_semantic_change():
    domain = "historical_backtest_valuation_policy.v0.1"
    baseline = {"policy_id": "a", "mark_field": "close"}
    changed = {"policy_id": "a", "mark_field": "open"}
    assert semantic_domain_sha256(domain, baseline) != semantic_domain_sha256(
        domain, changed
    )


@pytest.mark.parametrize("bad", [100, 100.0, "100", True, Decimal("NaN"), Decimal("Infinity")])
def test_valuation_mark_requires_strict_finite_decimal(artifact_ref_factory, bad):
    with pytest.raises(ValidationError):
        _mark(artifact_ref_factory, "NORGATE:1", close=bad)


@pytest.mark.parametrize("bad", [Decimal("0"), Decimal("-1")])
def test_valuation_mark_requires_positive_close(artifact_ref_factory, bad):
    with pytest.raises(ValidationError):
        _mark(artifact_ref_factory, "NORGATE:1", close=bad)


@pytest.mark.parametrize(
    ("field", "value"),
    [("currency", "EUR"), ("timeframe", "H1"), ("session_type", "extended"), ("price_basis", "adjusted")],
)
def test_valuation_mark_frozen_market_semantics(artifact_ref_factory, field, value):
    kwargs = {
        "security_id": "NORGATE:1",
        "session": date(2026, 8, 24),
        "close": Decimal("100"),
        "source_artifact_ref": artifact_ref_factory(),
        field: value,
    }
    with pytest.raises(ValidationError):
        HistoricalBacktestValuationMark(**kwargs)


def test_snapshot_public_model_rejects_unsorted_duplicates_and_wrong_session(artifact_ref_factory):
    first = _mark(artifact_ref_factory, "NORGATE:1")
    second = _mark(artifact_ref_factory, "NORGATE:2")
    valid = build_valuation_snapshot(session=first.session, marks=(first, second))
    with pytest.raises(ValidationError):
        HistoricalBacktestValuationSnapshot(
            session=first.session,
            marks=(second, first),
            snapshot_fingerprint=valid.snapshot_fingerprint,
        )
    duplicate_hash = semantic_domain_sha256(
        "historical_backtest_valuation_snapshot.v0.1",
        {"session": first.session, "marks": (first, first)},
    )
    with pytest.raises(ValidationError):
        HistoricalBacktestValuationSnapshot(
            session=first.session,
            marks=(first, first),
            snapshot_fingerprint=duplicate_hash,
        )
    wrong_session = _mark(
        artifact_ref_factory, "NORGATE:3", session=date(2026, 8, 23)
    )
    wrong_hash = semantic_domain_sha256(
        "historical_backtest_valuation_snapshot.v0.1",
        {"session": first.session, "marks": (wrong_session,)},
    )
    with pytest.raises(ValidationError):
        HistoricalBacktestValuationSnapshot(
            session=first.session,
            marks=(wrong_session,),
            snapshot_fingerprint=wrong_hash,
        )


def test_snapshot_builder_sorts_and_fingerprint_detects_corruption(artifact_ref_factory):
    first = _mark(artifact_ref_factory, "NORGATE:1")
    second = _mark(artifact_ref_factory, "NORGATE:2")
    built = build_valuation_snapshot(session=first.session, marks=(second, first))
    assert tuple(mark.security_id for mark in built.marks) == (
        "NORGATE:1",
        "NORGATE:2",
    )
    assert built == build_valuation_snapshot(
        session=first.session, marks=(first, second)
    )
    corrupted = built.model_dump(mode="python")
    corrupted["marks"] = (
        first.model_copy(update={"close": Decimal("101")}),
        second,
    )
    with pytest.raises(ValidationError):
        HistoricalBacktestValuationSnapshot.model_validate(corrupted)


def test_snapshot_requires_tuple_on_public_construction(artifact_ref_factory):
    mark = _mark(artifact_ref_factory, "NORGATE:1")
    with pytest.raises(ValidationError):
        HistoricalBacktestValuationSnapshot(
            session=mark.session,
            marks=[mark],
            snapshot_fingerprint="a" * 64,
        )
