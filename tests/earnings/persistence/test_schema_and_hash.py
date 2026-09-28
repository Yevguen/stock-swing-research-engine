"""Suite A: exact schema, identity, ordering, and hash acceptance."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa

from stock_swing_d1.earnings.persistence import (
    canonical_earnings_arrow_schema,
    canonical_record_hash,
    canonical_sort_key,
    event_history_key,
    revision_identity_key,
)


def test_persist_a01_canonical_arrow_schema_is_exact():
    """PERSIST_A01"""
    utc_us = pa.timestamp("us", tz="UTC")
    expected = [
        ("schema_version", pa.string(), False),
        ("event_revision_id", pa.string(), False),
        ("event_instance_id", pa.string(), False),
        ("canonical_asset_id", pa.string(), False),
        ("provider_name", pa.string(), False),
        ("provider_entity_id", pa.string(), False),
        ("provider_event_id", pa.string(), True),
        ("provider_record_id", pa.string(), True),
        ("historical_symbol", pa.string(), True),
        ("historical_exchange", pa.string(), True),
        ("knowledge_date", pa.date32(), True),
        ("knowledge_available_at", utc_us, True),
        ("knowledge_precision", pa.string(), False),
        ("strategy_effective_at", utc_us, False),
        ("provider_sequence", pa.int64(), True),
        ("transition_type", pa.string(), False),
        ("lifecycle_state", pa.string(), False),
        ("scheduled_date", pa.date32(), True),
        ("timing_class", pa.string(), False),
        ("scheduled_at", utc_us, True),
        ("event_timezone", pa.string(), True),
        ("actual_event_date", pa.date32(), True),
        ("actual_event_at", utc_us, True),
        ("is_provider_correction", pa.bool_(), False),
        ("correction_of_revision_id", pa.string(), True),
        ("ingested_at", utc_us, False),
        ("canonical_record_hash", pa.string(), False),
    ]
    schema = canonical_earnings_arrow_schema()
    assert [(field.name, field.type, field.nullable) for field in schema] == expected


def test_persist_a02_schema_metadata_is_versioned():
    """PERSIST_A02"""
    assert canonical_earnings_arrow_schema().metadata == {
        b"earnings_schema_version": b"0.1",
        b"canonical_sort_key_version": b"0.1",
        b"canonical_record_hash_algorithm": b"sha256",
    }


def test_persist_a03_enums_are_persisted_as_strings():
    """PERSIST_A03"""
    schema = canonical_earnings_arrow_schema()
    for name in (
        "knowledge_precision",
        "transition_type",
        "lifecycle_state",
        "timing_class",
    ):
        assert schema.field(name).type == pa.string()


def test_persist_a04_timestamps_normalize_to_utc_microseconds(make_persist_revision):
    """PERSIST_A04"""
    schema = canonical_earnings_arrow_schema()
    for name in (
        "knowledge_available_at",
        "strategy_effective_at",
        "scheduled_at",
        "actual_event_at",
        "ingested_at",
    ):
        assert schema.field(name).type == pa.timestamp("us", tz="UTC")
    utc_value = datetime(2026, 9, 1, 14, 0, 0, 123456, tzinfo=timezone.utc)
    offset_value = utc_value.astimezone(timezone(timedelta(hours=-4)))
    left = make_persist_revision(
        knowledge_available_at=utc_value,
        strategy_effective_at=utc_value,
    )
    right = replace(
        left,
        knowledge_available_at=offset_value,
        strategy_effective_at=offset_value,
    )
    assert canonical_record_hash(left) == canonical_record_hash(right)


def test_persist_a05_canonical_record_hash_is_deterministic(make_persist_revision):
    """PERSIST_A05"""
    revision = make_persist_revision()
    first = canonical_record_hash(revision)
    assert first == canonical_record_hash(revision)
    assert len(first) == 64
    assert first == first.lower()
    int(first, 16)


def test_persist_a06_ingested_at_is_excluded_from_semantic_hash(make_persist_revision):
    """PERSIST_A06"""
    revision = make_persist_revision()
    later_ingestion = replace(
        revision,
        ingested_at=revision.ingested_at + timedelta(days=30),
    )
    assert canonical_record_hash(revision) == canonical_record_hash(later_ingestion)


def test_persist_a07_semantic_field_change_changes_hash(make_persist_revision):
    """PERSIST_A07"""
    revision = make_persist_revision()
    changed = replace(revision, scheduled_date=date(2026, 10, 30))
    assert canonical_record_hash(revision) != canonical_record_hash(changed)


def test_persist_a08_revision_identity_is_provider_plus_revision_id(
    make_persist_revision,
):
    """PERSIST_A08"""
    revision = make_persist_revision(
        provider_name="PROVIDER-A", event_revision_id="REV-X"
    )
    assert revision_identity_key(revision) == ("PROVIDER-A", "REV-X")


def test_persist_a09_event_history_key_is_asset_plus_event_instance(
    make_persist_revision,
):
    """PERSIST_A09"""
    revision = make_persist_revision(
        canonical_asset_id="ASSET-X", event_instance_id="EVENT-X"
    )
    assert event_history_key(revision) == ("ASSET-X", "EVENT-X")


def test_persist_a10_canonical_sort_key_is_deterministic_and_null_sequence_sorts_last(
    make_persist_revision,
):
    """PERSIST_A10"""
    sequenced = make_persist_revision(provider_sequence=999)
    unsequenced = replace(sequenced, provider_sequence=None, event_revision_id="REV-Z")
    assert canonical_sort_key(sequenced) == canonical_sort_key(sequenced)
    assert canonical_sort_key(sequenced) < canonical_sort_key(unsequenced)
    assert sorted([unsequenced, sequenced], key=canonical_sort_key) == [
        sequenced,
        unsequenced,
    ]
