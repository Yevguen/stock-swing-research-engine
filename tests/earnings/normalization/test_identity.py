from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from stock_swing_d1.earnings import EarningsValidationError
from stock_swing_d1.earnings.normalization import normalize_earnings_delivery
from stock_swing_d1.earnings.persistence import QuarantineStage
from tests.earnings.normalization.conftest import POSITIVE_EVIDENCE, PROVIDER


def test_norm_a01_successful_identity_uses_norgate_asset_id_format(
    identity_resolver,
) -> None:
    resolved = identity_resolver.resolve_canonical_asset_id(
        provider_name=PROVIDER,
        provider_entity_id="ENTITY-1",
        historical_symbol="OLDX",
        historical_exchange="XNYS",
        mapping_evidence=POSITIVE_EVIDENCE,
    )
    assert re.fullmatch(r"NORGATE:[1-9][0-9]*", resolved)


def test_norm_a02_ticker_only_mapping_is_rejected(identity_resolver) -> None:
    with pytest.raises(EarningsValidationError, match="AMBIGUOUS_SECURITY_MAPPING"):
        identity_resolver.resolve_canonical_asset_id(
            provider_name=PROVIDER,
            provider_entity_id="ENTITY-1",
            historical_symbol="ONLY",
            historical_exchange="XNYS",
            mapping_evidence=(),
        )


def test_norm_a03_single_candidate_without_mapping_evidence_is_rejected(
    identity_resolver,
) -> None:
    with pytest.raises(
        EarningsValidationError,
        match="INSUFFICIENT_SECURITY_MAPPING_EVIDENCE",
    ):
        identity_resolver.resolve_canonical_asset_id(
            provider_name=PROVIDER,
            provider_entity_id="ENTITY-1",
            historical_symbol="ONLY",
            historical_exchange="XNYS",
            mapping_evidence=(("candidate_asset_id", "NORGATE:1001"),),
        )


def test_norm_a04_mapping_evidence_resolves_one_canonical_asset(
    identity_resolver,
) -> None:
    resolved = identity_resolver.resolve_canonical_asset_id(
        provider_name=PROVIDER,
        provider_entity_id="ENTITY-1",
        historical_symbol="OLDX",
        historical_exchange="XNYS",
        mapping_evidence=POSITIVE_EVIDENCE,
    )
    assert resolved == "NORGATE:1001"


def test_norm_a05_identity_resolution_is_deterministic(identity_resolver) -> None:
    request = dict(
        provider_name=PROVIDER,
        provider_entity_id="ENTITY-1",
        historical_symbol="OLDX",
        historical_exchange="XNYS",
        mapping_evidence=POSITIVE_EVIDENCE,
    )
    assert identity_resolver.resolve_canonical_asset_id(
        **request
    ) == identity_resolver.resolve_canonical_asset_id(**request)


def test_norm_a06_ticker_change_preserves_canonical_asset_identity(
    delivery_adapter,
    identity_resolver,
    raw_record,
) -> None:
    old = raw_record()
    renamed = raw_record(
        revision_key="R2",
        provider_record_id="SYNTHETIC-RECORD-R2",
        historical_symbol="NEWX",
        knowledge_date=datetime(2026, 9, 2, tzinfo=timezone.utc).date(),
        knowledge_available_at=datetime(2026, 9, 2, 14, tzinfo=timezone.utc),
        provider_sequence=2,
    )
    batch = delivery_adapter.normalize_delivery(
        (renamed, old),
        identity_resolver=identity_resolver,
    )
    assert not batch.failures
    assert {revision.canonical_asset_id for revision in batch.revisions} == {
        "NORGATE:1001"
    }


def test_norm_a07_current_ticker_is_not_backfilled_into_old_revision(
    delivery_adapter,
    identity_resolver,
    raw_record,
) -> None:
    old = raw_record(historical_symbol="OLDX")
    renamed = raw_record(
        revision_key="R2",
        historical_symbol="NEWX",
        knowledge_date=datetime(2026, 9, 2, tzinfo=timezone.utc).date(),
        knowledge_available_at=datetime(2026, 9, 2, 14, tzinfo=timezone.utc),
        provider_sequence=2,
    )
    revisions = delivery_adapter.normalize_delivery(
        (renamed, old), identity_resolver=identity_resolver
    ).revisions
    assert [revision.historical_symbol for revision in revisions] == ["OLDX", "NEWX"]


def test_norm_a08_identity_resolution_failure_becomes_identity_mapping_quarantine(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    result = normalize_earnings_delivery(
        (raw_record(mapping_evidence=()),),
        adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        trading_calendar=normalization_calendar,
        canonical_history=None,
        build_id="norm-a08",
        ingested_at=datetime(2026, 9, 5, 12, tzinfo=timezone.utc),
    )
    assert not result.success
    assert result.revisions == ()
    assert result.quarantines[0].stage is QuarantineStage.IDENTITY_MAPPING
