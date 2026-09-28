"""Suite C: immutable provider-scoped idempotency acceptance."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from stock_swing_d1.earnings.persistence import (
    EarningsPersistencePaths,
    canonical_record_hash,
    publish_earnings_revisions,
    revision_identity_key,
)
from stock_swing_d1.earnings.persistence.publication import _classify_revisions


def test_persist_c01_new_revision_identity_is_insert_candidate(make_persist_revision):
    """PERSIST_C01"""
    revision = make_persist_revision()
    classified = _classify_revisions((revision,), ())
    assert classified.insert_candidates == (revision,)
    assert classified.idempotent_duplicate_count == 0
    assert classified.conflicts == ()


def test_persist_c02_same_identity_same_hash_is_idempotent_duplicate(
    make_persist_revision,
):
    """PERSIST_C02"""
    existing = make_persist_revision()
    incoming = replace(existing)
    classified = _classify_revisions((incoming,), (existing,))
    assert classified.insert_candidates == ()
    assert classified.idempotent_duplicate_count == 1
    assert classified.conflicts == ()


def test_persist_c03_same_identity_different_hash_is_immutable_conflict(
    make_persist_revision,
):
    """PERSIST_C03"""
    existing = make_persist_revision()
    incoming = replace(existing, scheduled_date=date(2026, 11, 2))
    classified = _classify_revisions((incoming,), (existing,))
    assert classified.insert_candidates == ()
    assert classified.idempotent_duplicate_count == 0
    assert len(classified.conflicts) == 1
    assert classified.conflicts[0].incoming is incoming
    assert classified.conflicts[0].prior is existing


def test_persist_c04_different_revision_ids_with_same_hash_are_retained(
    make_persist_revision,
):
    """PERSIST_C04"""
    first = make_persist_revision(event_revision_id="REV-1")
    second = replace(first, event_revision_id="REV-2")
    assert canonical_record_hash(first) == canonical_record_hash(second)
    classified = _classify_revisions((first, second), ())
    assert classified.insert_candidates == (first, second)
    assert classified.idempotent_duplicate_count == 0
    assert classified.conflicts == ()


def test_persist_c05_different_ingested_at_remains_idempotent_duplicate(
    make_persist_revision,
):
    """PERSIST_C05"""
    existing = make_persist_revision()
    incoming = replace(
        existing, ingested_at=existing.ingested_at + timedelta(days=1)
    )
    assert canonical_record_hash(existing) == canonical_record_hash(incoming)
    classified = _classify_revisions((incoming,), (existing,))
    assert classified.idempotent_duplicate_count == 1
    assert classified.insert_candidates == ()
    assert classified.conflicts == ()


def test_persist_c06_identical_duplicate_inside_same_batch_is_idempotent(
    make_persist_revision,
):
    """PERSIST_C06"""
    first = make_persist_revision()
    second = replace(first, ingested_at=first.ingested_at + timedelta(minutes=1))
    classified = _classify_revisions((first, second), ())
    assert classified.insert_candidates == (first,)
    assert classified.idempotent_duplicate_count == 1
    assert classified.conflicts == ()


def test_persist_c07_conflicting_duplicate_inside_same_batch_fails_build(
    tmp_path,
    make_persist_revision,
):
    """PERSIST_C07"""
    first = make_persist_revision()
    conflicting = replace(first, scheduled_date=date(2026, 11, 3))
    classified = _classify_revisions((first, conflicting), ())
    assert len(classified.conflicts) == 1
    assert classified.conflicts[0].incoming is conflicting

    assert revision_identity_key(first) == revision_identity_key(conflicting)
    assert canonical_record_hash(first) != canonical_record_hash(conflicting)
    paths = EarningsPersistencePaths(tmp_path / "earnings")
    result = publish_earnings_revisions(
        [first, conflicting], paths=paths, build_id="C07"
    )
    assert result.success is False
    assert result.conflict_count == 1
    assert result.inserted_count == 0
    assert result.output_path is None
    assert result.output_sha256 is None
    assert not paths.published_path.exists()


def test_persist_c08_same_revision_id_from_different_providers_has_distinct_identity_key(
    make_persist_revision,
):
    """PERSIST_C08"""
    first = make_persist_revision(provider_name="PROVIDER-A", event_revision_id="REV-X")
    second = replace(first, provider_name="PROVIDER-B")
    assert revision_identity_key(first) == ("PROVIDER-A", "REV-X")
    assert revision_identity_key(second) == ("PROVIDER-B", "REV-X")
    classified = _classify_revisions((first, second), ())
    assert classified.insert_candidates == (first, second)
    assert classified.conflicts == ()
