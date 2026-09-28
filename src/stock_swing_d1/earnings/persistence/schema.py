"""Versioned, exact Arrow schemas for earnings persistence."""

from __future__ import annotations

import pyarrow as pa


CANONICAL_EARNINGS_SCHEMA_VERSION = "0.1"
CANONICAL_SORT_KEY_VERSION = "0.1"
CANONICAL_RECORD_HASH_ALGORITHM = "sha256"
OUTPUT_HASH_ALGORITHM = "sha256"

_UTC_MICROSECOND = pa.timestamp("us", tz="UTC")
_CANONICAL_METADATA = {
    b"earnings_schema_version": CANONICAL_EARNINGS_SCHEMA_VERSION.encode("ascii"),
    b"canonical_sort_key_version": CANONICAL_SORT_KEY_VERSION.encode("ascii"),
    b"canonical_record_hash_algorithm": CANONICAL_RECORD_HASH_ALGORITHM.encode(
        "ascii"
    ),
}


def canonical_earnings_arrow_schema() -> pa.Schema:
    """Return the frozen v0.1 canonical earnings physical schema."""

    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("event_revision_id", pa.string(), nullable=False),
            pa.field("event_instance_id", pa.string(), nullable=False),
            pa.field("canonical_asset_id", pa.string(), nullable=False),
            pa.field("provider_name", pa.string(), nullable=False),
            pa.field("provider_entity_id", pa.string(), nullable=False),
            pa.field("provider_event_id", pa.string()),
            pa.field("provider_record_id", pa.string()),
            pa.field("historical_symbol", pa.string()),
            pa.field("historical_exchange", pa.string()),
            pa.field("knowledge_date", pa.date32()),
            pa.field("knowledge_available_at", _UTC_MICROSECOND),
            pa.field("knowledge_precision", pa.string(), nullable=False),
            pa.field("strategy_effective_at", _UTC_MICROSECOND, nullable=False),
            pa.field("provider_sequence", pa.int64()),
            pa.field("transition_type", pa.string(), nullable=False),
            pa.field("lifecycle_state", pa.string(), nullable=False),
            pa.field("scheduled_date", pa.date32()),
            pa.field("timing_class", pa.string(), nullable=False),
            pa.field("scheduled_at", _UTC_MICROSECOND),
            pa.field("event_timezone", pa.string()),
            pa.field("actual_event_date", pa.date32()),
            pa.field("actual_event_at", _UTC_MICROSECOND),
            pa.field("is_provider_correction", pa.bool_(), nullable=False),
            pa.field("correction_of_revision_id", pa.string()),
            pa.field("ingested_at", _UTC_MICROSECOND, nullable=False),
            pa.field("canonical_record_hash", pa.string(), nullable=False),
        ],
        metadata=_CANONICAL_METADATA,
    )


def quarantine_arrow_schema() -> pa.Schema:
    """Return the strict schema for diagnostic-only quarantine rows."""

    return pa.schema(
        [
            pa.field("quarantine_id", pa.string(), nullable=False),
            pa.field("build_id", pa.string(), nullable=False),
            pa.field("quarantined_at", _UTC_MICROSECOND, nullable=False),
            pa.field("provider_name", pa.string()),
            pa.field("provider_entity_id", pa.string()),
            pa.field("canonical_asset_id", pa.string()),
            pa.field("event_instance_id", pa.string()),
            pa.field("event_revision_id", pa.string()),
            pa.field("stage", pa.string(), nullable=False),
            pa.field("error_code", pa.string(), nullable=False),
            pa.field("error_message", pa.string(), nullable=False),
            pa.field("source_record_locator", pa.string()),
            pa.field("source_record_hash", pa.string()),
            pa.field("canonical_record_hash", pa.string()),
            pa.field("details_json", pa.string()),
        ]
    )
