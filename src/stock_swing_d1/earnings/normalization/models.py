"""Immutable orchestration models for provider-neutral earnings normalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from stock_swing_d1.earnings.models import EarningsScheduleRevision, KnowledgePrecision
from stock_swing_d1.earnings.persistence.models import (
    EarningsPublicationResult,
    EarningsQuarantineRecord,
    QuarantineStage,
)


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


@dataclass(frozen=True, slots=True)
class EarningsKnowledgeCutoff:
    knowledge_precision: KnowledgePrecision
    knowledge_date: date
    knowledge_available_at: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.knowledge_precision, KnowledgePrecision):
            raise TypeError("knowledge_precision must be KnowledgePrecision")
        if type(self.knowledge_date) is not date:
            raise TypeError("knowledge_date must be a genuine date")
        if self.knowledge_precision is KnowledgePrecision.TIMESTAMP:
            available_at = self.knowledge_available_at
            if not isinstance(available_at, datetime) or not _is_aware(available_at):
                raise ValueError(
                    "TIMESTAMP knowledge requires timezone-aware knowledge_available_at"
                )
            object.__setattr__(
                self,
                "knowledge_available_at",
                available_at.astimezone(timezone.utc),
            )
        elif self.knowledge_available_at is not None:
            raise ValueError("DATE_ONLY knowledge requires knowledge_available_at=None")


@dataclass(frozen=True, slots=True)
class EarningsNormalizationFailure:
    stage: QuarantineStage
    error_code: str
    error_message: str

    provider_name: str | None
    provider_entity_id: str | None
    canonical_asset_id: str | None
    event_instance_id: str | None
    event_revision_id: str | None

    source_record_locator: str | None
    source_record_hash: str | None
    details_json: str | None


@dataclass(frozen=True, slots=True)
class EarningsAdapterBatch:
    revisions: tuple[EarningsScheduleRevision, ...]
    failures: tuple[EarningsNormalizationFailure, ...]


@dataclass(frozen=True, slots=True)
class EarningsNormalizationResult:
    schema_version: str
    build_id: str
    provider_name: str
    source_delivery_id: str | None

    input_record_count: int
    normalized_record_count: int
    quarantined_count: int

    revisions: tuple[EarningsScheduleRevision, ...]
    quarantines: tuple[EarningsQuarantineRecord, ...]

    success: bool


@dataclass(frozen=True, slots=True)
class EarningsDeliveryResult:
    normalization_result: EarningsNormalizationResult
    publication_result: EarningsPublicationResult | None
    success: bool
