"""Immutable models and filesystem paths for earnings persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class QuarantineStage(str, Enum):
    """The frozen processing stages at which a record may be quarantined."""

    NORMALIZATION = "NORMALIZATION"
    IDENTITY_MAPPING = "IDENTITY_MAPPING"
    RECORD_VALIDATION = "RECORD_VALIDATION"
    HISTORY_VALIDATION = "HISTORY_VALIDATION"
    IDEMPOTENCY = "IDEMPOTENCY"
    PUBLICATION = "PUBLICATION"


@dataclass(frozen=True, slots=True)
class EarningsQuarantineRecord:
    quarantine_id: str
    build_id: str
    quarantined_at: datetime
    provider_name: str | None
    provider_entity_id: str | None
    canonical_asset_id: str | None
    event_instance_id: str | None
    event_revision_id: str | None
    stage: QuarantineStage
    error_code: str
    error_message: str
    source_record_locator: str | None
    source_record_hash: str | None
    canonical_record_hash: str | None
    details_json: str | None


@dataclass(frozen=True, slots=True)
class EarningsPublicationResult:
    schema_version: str
    build_id: str
    started_at: datetime
    completed_at: datetime
    provider_name: str | None
    input_record_count: int
    validated_record_count: int
    inserted_count: int
    idempotent_duplicate_count: int
    conflict_count: int
    quarantined_count: int
    validation_error_count: int
    published_record_count: int
    published_event_count: int
    published_asset_count: int
    minimum_strategy_effective_at: datetime | None
    maximum_strategy_effective_at: datetime | None
    output_path: str | None
    output_sha256: str | None
    success: bool


@dataclass(frozen=True, slots=True)
class EarningsPublicationManifest:
    schema_version: str
    build_id: str
    started_at: datetime
    completed_at: datetime
    provider_name: str | None
    source_delivery_id: str | None
    input_record_count: int
    published_record_count: int
    published_event_count: int
    published_asset_count: int
    minimum_strategy_effective_at: datetime | None
    maximum_strategy_effective_at: datetime | None
    canonical_sort_key_version: str
    canonical_record_hash_algorithm: str
    output_hash_algorithm: str
    output_file: str
    output_sha256: str
    success: bool


@dataclass(frozen=True, slots=True)
class EarningsPersistencePaths:
    root: Path

    @property
    def canonical_path(self) -> Path:
        return self.root / "canonical" / "earnings_schedule_revisions.parquet"

    @property
    def published_path(self) -> Path:
        return self.root / "published" / "earnings_schedule_revisions.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "published" / "earnings_schedule_revisions.manifest.json"

    @property
    def quarantine_dir(self) -> Path:
        return self.root / "quarantine"

    @property
    def temporary_dir(self) -> Path:
        return self.root / ".tmp"


@dataclass(frozen=True, slots=True)
class CanonicalParquetArtifact:
    path: str
    row_count: int
    sha256: str


class EarningsPersistenceError(RuntimeError):
    """A persistence failure carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")
