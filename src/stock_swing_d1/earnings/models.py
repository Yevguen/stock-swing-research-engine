"""Canonical provider-neutral earnings schedule types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Protocol


class LifecycleState(str, Enum):
    UNKNOWN = "UNKNOWN"
    ESTIMATED = "ESTIMATED"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    OCCURRED = "OCCURRED"
    INVALID = "INVALID"


class TransitionType(str, Enum):
    CREATED = "CREATED"
    DATE_CHANGED = "DATE_CHANGED"
    TIMING_CHANGED = "TIMING_CHANGED"
    CONFIRMED = "CONFIRMED"
    UNCONFIRMED = "UNCONFIRMED"
    POSTPONED = "POSTPONED"
    ADVANCED = "ADVANCED"
    CANCELLED = "CANCELLED"
    REINSTATED = "REINSTATED"
    OCCURRED = "OCCURRED"
    PROVIDER_CORRECTION = "PROVIDER_CORRECTION"
    OTHER = "OTHER"


class TimingClass(str, Enum):
    UNKNOWN = "UNKNOWN"
    BMO = "BMO"
    DURING_MARKET = "DURING_MARKET"
    AMC = "AMC"
    EXACT_TIME = "EXACT_TIME"


class KnowledgePrecision(str, Enum):
    TIMESTAMP = "TIMESTAMP"
    DATE_ONLY = "DATE_ONLY"


class EarningsScheduleStatus(str, Enum):
    """Whether PIT data can establish an event or a safe provider horizon."""

    UNKNOWN = "EARNINGS_SCHEDULE_UNKNOWN"
    KNOWN_EVENT = "KNOWN_EARNINGS_EVENT"
    NO_EVENT_WITHIN_PROVIDER_HORIZON = "NO_EARNINGS_WITHIN_PROVIDER_HORIZON"


class EarningsValidationError(ValueError):
    """Structured contract or history validation failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class EarningsScheduleRevision:
    """One immutable, complete resulting canonical state snapshot.

    Every field describes the state effective from ``strategy_effective_at``;
    this is not a partial delta.  Provider adapters must resolve provider
    deltas before constructing a canonical revision.  ``None`` and explicit
    ``UNKNOWN`` enums are therefore meaningful values and are never inherited
    implicitly from an earlier revision during replay.
    """

    schema_version: str = "0.1"
    event_revision_id: str | None = None
    event_instance_id: str | None = None

    canonical_asset_id: str | None = None
    provider_name: str | None = None
    provider_entity_id: str | None = None
    provider_event_id: str | None = None
    provider_record_id: str | None = None

    historical_symbol: str | None = None
    historical_exchange: str | None = None

    knowledge_date: date | None = None
    knowledge_available_at: datetime | None = None
    knowledge_precision: KnowledgePrecision = KnowledgePrecision.TIMESTAMP
    strategy_effective_at: datetime | None = None
    provider_sequence: int | None = None

    transition_type: TransitionType = TransitionType.CREATED
    lifecycle_state: LifecycleState = LifecycleState.UNKNOWN

    scheduled_date: date | None = None
    timing_class: TimingClass = TimingClass.UNKNOWN
    scheduled_at: datetime | None = None
    event_timezone: str | None = None

    actual_event_date: date | None = None
    actual_event_at: datetime | None = None

    is_provider_correction: bool = False
    correction_of_revision_id: str | None = None

    ingested_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class EarningsStateAsOf:
    """Reconstructed earnings state visible at one decision timestamp."""

    as_of: datetime
    canonical_asset_id: str | None
    event_instance_id: str | None
    earnings_schedule_known: bool
    lifecycle_state: LifecycleState
    scheduled_date: date | None
    timing_class: TimingClass
    scheduled_at: datetime | None
    knowledge_effective_at: datetime | None
    transition_type: TransitionType | None = None
    event_revision_id: str | None = None
    historical_symbol: str | None = None
    historical_exchange: str | None = None
    actual_event_date: date | None = None
    actual_event_at: datetime | None = None
    is_provider_correction: bool = False
    schedule_status: EarningsScheduleStatus = EarningsScheduleStatus.UNKNOWN


@dataclass(frozen=True, slots=True)
class EarningsRiskDecision:
    """Provider-neutral strategy-facing earnings risk result."""

    entry_blackout: bool = False
    mandatory_exit: bool = False
    last_safe_exit_session: date | None = None
    pending_entry_invalidated: bool = False
    unavoidable_earnings_exposure: bool = False
    risk_reason: str | None = None
    holding_age_sessions: int | None = None


class EarningsProviderAdapter(Protocol):
    """Boundary that keeps raw provider values outside strategy code."""

    def normalize_record(self, raw_record: Any) -> EarningsScheduleRevision:
        ...
