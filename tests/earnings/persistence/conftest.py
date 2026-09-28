"""Deterministic synthetic fixtures for persistence acceptance tests."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_swing_d1.earnings import (
    EarningsScheduleRevision,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
    TransitionType,
)


def make_persistence_revision(**overrides: object) -> EarningsScheduleRevision:
    known_at = datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc)
    values: dict[str, object] = {
        "schema_version": "0.1",
        "event_revision_id": "SYNTH-REV-001",
        "event_instance_id": "SYNTH-EVENT-2026-Q3",
        "canonical_asset_id": "SYNTH-ASSET-001",
        "provider_name": "SYNTHETIC",
        "provider_entity_id": "SYNTH-ENTITY-001",
        "provider_event_id": "SYNTH-PROVIDER-EVENT-001",
        "provider_record_id": "SYNTH-PROVIDER-RECORD-001",
        "historical_symbol": "SYN",
        "historical_exchange": "XNAS",
        "knowledge_date": known_at.date(),
        "knowledge_available_at": known_at,
        "knowledge_precision": KnowledgePrecision.TIMESTAMP,
        "strategy_effective_at": known_at,
        "provider_sequence": 1,
        "transition_type": TransitionType.CREATED,
        "lifecycle_state": LifecycleState.ESTIMATED,
        "scheduled_date": date(2026, 10, 29),
        "timing_class": TimingClass.UNKNOWN,
        "scheduled_at": None,
        "event_timezone": "America/New_York",
        "actual_event_date": None,
        "actual_event_at": None,
        "is_provider_correction": False,
        "correction_of_revision_id": None,
        "ingested_at": datetime(2026, 9, 1, 14, 5, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return EarningsScheduleRevision(**values)


@pytest.fixture
def make_persist_revision():
    return make_persistence_revision
