"""Small synthetic builders for the Phase 6C earnings test suites."""

from __future__ import annotations

from datetime import date, datetime, timezone

from stock_swing_d1.earnings import (
    EarningsScheduleRevision,
    EarningsScheduleStatus,
    EarningsStateAsOf,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
    TransitionType,
)


def make_revision(**overrides: object) -> EarningsScheduleRevision:
    known_at = datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc)
    values: dict[str, object] = {
        "schema_version": "0.1",
        "event_revision_id": "REV-001",
        "event_instance_id": "EARN-2026-Q3",
        "canonical_asset_id": "NORGATE:1001",
        "provider_name": "SYNTHETIC",
        "provider_entity_id": "TEST-ENTITY-001",
        "provider_event_id": "TEST-EVENT-001",
        "provider_record_id": "TEST-RECORD-001",
        "historical_symbol": "TEST",
        "historical_exchange": "NYSE",
        "knowledge_date": known_at.date(),
        "knowledge_available_at": known_at,
        "knowledge_precision": KnowledgePrecision.TIMESTAMP,
        "strategy_effective_at": known_at,
        "provider_sequence": None,
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


def make_revision_history(
    *revisions: EarningsScheduleRevision,
) -> tuple[EarningsScheduleRevision, ...]:
    return revisions


def make_state(**overrides: object) -> EarningsStateAsOf:
    values: dict[str, object] = {
        "as_of": datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc),
        "canonical_asset_id": "NORGATE:1001",
        "event_instance_id": "EARN-2026-Q3",
        "earnings_schedule_known": True,
        "lifecycle_state": LifecycleState.CONFIRMED,
        "scheduled_date": date(2026, 10, 14),
        "timing_class": TimingClass.UNKNOWN,
        "scheduled_at": None,
        "knowledge_effective_at": datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc),
        "transition_type": TransitionType.CONFIRMED,
        "event_revision_id": "REV-STATE",
        "historical_symbol": "TEST",
        "historical_exchange": "NYSE",
        "schedule_status": EarningsScheduleStatus.KNOWN_EVENT,
    }
    values.update(overrides)
    return EarningsStateAsOf(**values)
