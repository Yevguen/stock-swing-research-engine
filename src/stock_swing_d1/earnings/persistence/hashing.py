"""Canonical identity, ordering, and semantic hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from stock_swing_d1.earnings.models import EarningsScheduleRevision


def revision_identity_key(
    revision: EarningsScheduleRevision,
) -> tuple[str, str]:
    """Return the immutable provider-scoped revision identity."""

    return (revision.provider_name, revision.event_revision_id)  # type: ignore[return-value]


def event_history_key(
    revision: EarningsScheduleRevision,
) -> tuple[str, str]:
    """Return the canonical asset/event history identity."""

    return (revision.canonical_asset_id, revision.event_instance_id)  # type: ignore[return-value]


def _utc_datetime_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("semantic timestamps must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _semantic_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _utc_datetime_text(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


def canonical_record_hash(revision: EarningsScheduleRevision) -> str:
    """Hash canonical state semantics, excluding identity/ingestion metadata."""

    payload = {
        field.name: _semantic_value(getattr(revision, field.name))
        for field in fields(EarningsScheduleRevision)
        if field.name not in {"event_revision_id", "ingested_at"}
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def canonical_sort_key(revision: EarningsScheduleRevision) -> tuple:
    """Return the frozen deterministic row order with NULL sequences last."""

    effective_at = revision.strategy_effective_at
    if effective_at is not None:
        if effective_at.tzinfo is None or effective_at.utcoffset() is None:
            raise ValueError("strategy_effective_at must be timezone-aware")
        effective_at = effective_at.astimezone(timezone.utc)
    sequence = revision.provider_sequence
    return (
        revision.canonical_asset_id,
        revision.event_instance_id,
        effective_at,
        sequence is None,
        sequence if sequence is not None else 0,
        revision.provider_name,
        revision.event_revision_id,
    )
