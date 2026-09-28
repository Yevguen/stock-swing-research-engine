"""Security identity and PIT-safe canonical-history boundaries."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Protocol

from stock_swing_d1.earnings.models import (
    EarningsScheduleRevision,
    KnowledgePrecision,
)
from stock_swing_d1.earnings.normalization.models import EarningsKnowledgeCutoff
from stock_swing_d1.earnings.persistence import (
    canonical_sort_key,
    read_canonical_revisions,
)


class EarningsSecurityIdentityResolver(Protocol):
    def resolve_canonical_asset_id(
        self,
        *,
        provider_name: str,
        provider_entity_id: str,
        historical_symbol: str | None,
        historical_exchange: str | None,
        mapping_evidence: tuple[tuple[str, str], ...],
    ) -> str:
        ...


class EarningsCanonicalHistoryReader(Protocol):
    def get_latest_revision(
        self,
        *,
        provider_name: str,
        canonical_asset_id: str,
        event_instance_id: str,
        knowledge_cutoff: EarningsKnowledgeCutoff,
    ) -> EarningsScheduleRevision | None:
        ...


def is_revision_admissible(
    revision: EarningsScheduleRevision,
    knowledge_cutoff: EarningsKnowledgeCutoff,
) -> bool:
    """Return whether historical knowledge is provably prior to the cutoff."""

    if knowledge_cutoff.knowledge_precision is KnowledgePrecision.DATE_ONLY:
        prior_date = revision.knowledge_date
        if type(prior_date) is not type(knowledge_cutoff.knowledge_date):
            return False
        return prior_date < knowledge_cutoff.knowledge_date

    if revision.knowledge_precision is KnowledgePrecision.DATE_ONLY:
        prior_date = revision.knowledge_date
        if type(prior_date) is not type(knowledge_cutoff.knowledge_date):
            return False
        return prior_date < knowledge_cutoff.knowledge_date

    if revision.knowledge_precision is not KnowledgePrecision.TIMESTAMP:
        return False
    prior_available_at = revision.knowledge_available_at
    incoming_available_at = knowledge_cutoff.knowledge_available_at
    if prior_available_at is None or incoming_available_at is None:
        return False
    if (
        prior_available_at.tzinfo is None
        or prior_available_at.utcoffset() is None
        or incoming_available_at.tzinfo is None
        or incoming_available_at.utcoffset() is None
    ):
        return False
    return prior_available_at.astimezone(timezone.utc) < incoming_available_at.astimezone(
        timezone.utc
    )


def select_latest_admissible_revision(
    revisions: tuple[EarningsScheduleRevision, ...],
    *,
    provider_name: str,
    canonical_asset_id: str,
    event_instance_id: str,
    knowledge_cutoff: EarningsKnowledgeCutoff,
) -> EarningsScheduleRevision | None:
    """Filter by exact scope and PIT evidence, then use canonical ordering."""

    candidates = tuple(
        revision
        for revision in revisions
        if revision.provider_name == provider_name
        and revision.canonical_asset_id == canonical_asset_id
        and revision.event_instance_id == event_instance_id
        and is_revision_admissible(revision, knowledge_cutoff)
    )
    if not candidates:
        return None
    return max(candidates, key=canonical_sort_key)


class _ParquetCanonicalHistoryReader:
    """Strict, read-only view of one published canonical Parquet artifact."""

    def __init__(self, published_path: Path) -> None:
        self._revisions = read_canonical_revisions(published_path)

    def get_latest_revision(
        self,
        *,
        provider_name: str,
        canonical_asset_id: str,
        event_instance_id: str,
        knowledge_cutoff: EarningsKnowledgeCutoff,
    ) -> EarningsScheduleRevision | None:
        return select_latest_admissible_revision(
            self._revisions,
            provider_name=provider_name,
            canonical_asset_id=canonical_asset_id,
            event_instance_id=event_instance_id,
            knowledge_cutoff=knowledge_cutoff,
        )
