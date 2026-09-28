from __future__ import annotations

from dataclasses import fields
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from stock_swing_d1.earnings import validate_revision, validate_revision_history
from stock_swing_d1.earnings.normalization import (
    normalize_earnings_delivery,
    process_earnings_delivery,
)
from stock_swing_d1.earnings.persistence import (
    EarningsPersistencePaths,
    QuarantineStage,
    read_canonical_revisions,
    read_quarantine_records,
)


def _normalize(
    records,
    *,
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    build_id="normalization-pipeline",
    ingested_at=datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
):
    return normalize_earnings_delivery(
        records,
        adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        trading_calendar=normalization_calendar,
        canonical_history=None,
        build_id=build_id,
        ingested_at=ingested_at,
    )


def _process(
    records,
    *,
    root,
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    build_id,
):
    return process_earnings_delivery(
        records,
        adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        trading_calendar=normalization_calendar,
        paths=EarningsPersistencePaths(root),
        source_delivery_id=f"delivery-{build_id}",
        build_id=build_id,
        ingested_at=datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
    )


def test_norm_d01_provider_parsing_failure_creates_normalization_quarantine(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    result = _normalize(
        (raw_record(raise_adapter_error=True),),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    )
    assert not result.success
    assert result.quarantines[0].stage is QuarantineStage.NORMALIZATION
    assert result.quarantines[0].error_code == "SYNTHETIC_PARSE_ERROR"


def test_norm_d02_quarantine_preserves_permitted_source_locator_and_hash(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    raw = raw_record(
        mapping_evidence=(),
        source_record_locator="delivery://synthetic/safe-locator",
        source_record_hash="a" * 64,
    )
    quarantine = _normalize(
        (raw,),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    ).quarantines[0]
    assert quarantine.source_record_locator == "delivery://synthetic/safe-locator"
    assert quarantine.source_record_hash == "a" * 64


def test_norm_d03_quarantine_diagnostics_never_embed_raw_provider_record(
    tmp_path,
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    secret = "FORBIDDEN-SYNTHETIC-RAW-PAYLOAD"
    raw = raw_record(mapping_evidence=(), raw_payload={"secret": secret})
    result = _process(
        (raw,),
        root=tmp_path,
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        build_id="norm-d03",
    )
    quarantine_path = (
        EarningsPersistencePaths(tmp_path).quarantine_dir
        / "earnings_quarantine_norm-d03.parquet"
    )
    stored = read_quarantine_records(quarantine_path)
    assert not result.success
    assert not any("raw" in field.name.lower() for field in fields(stored[0]))
    assert secret.encode("utf-8") not in quarantine_path.read_bytes()
    assert stored[0].details_json == '{"diagnostic":"synthetic"}'


def test_norm_d04_any_upstream_quarantine_returns_no_publishable_revisions(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    mixed = (
        raw_record(),
        raw_record(
            provider_entity_id="UNMAPPED-ENTITY",
            revision_key="BAD",
            mapping_evidence=(),
        ),
    )
    result = _normalize(
        mixed,
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    )
    assert result.normalized_record_count > 0
    assert result.quarantined_count > 0
    assert not result.success
    assert result.revisions == ()


def test_norm_d05_successful_normalization_stamps_utc_ingested_at_and_effective_time(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    available_at = datetime(2026, 9, 1, 16, tzinfo=ZoneInfo("Europe/Madrid"))
    operational_ingestion = datetime(
        2026, 9, 10, 14, tzinfo=ZoneInfo("Europe/Madrid")
    )
    raw = raw_record(knowledge_date=available_at.date(), knowledge_available_at=available_at)
    candidate = delivery_adapter.normalize_delivery(
        (raw,), identity_resolver=identity_resolver
    ).revisions[0]
    assert candidate.strategy_effective_at is None
    assert candidate.ingested_at is None
    result = _normalize(
        (raw,),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        ingested_at=operational_ingestion,
    )
    revision = result.revisions[0]
    assert revision.ingested_at == operational_ingestion.astimezone(timezone.utc)
    assert revision.ingested_at.tzinfo is timezone.utc
    assert revision.strategy_effective_at == available_at.astimezone(timezone.utc)


def test_norm_d06_successful_normalization_passes_existing_revision_and_history_validation(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    confirmed = raw_record(
        revision_key="R2",
        provider_record_id="SYNTHETIC-RECORD-R2",
        knowledge_date=date(2026, 9, 2),
        knowledge_available_at=datetime(2026, 9, 2, 14, tzinfo=timezone.utc),
        provider_sequence=2,
        lifecycle_code="CONF",
    )
    result = _normalize(
        (confirmed, raw_record()),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    )
    assert result.success
    assert all(validate_revision(revision) is revision for revision in result.revisions)
    assert validate_revision_history(result.revisions) == result.revisions


def test_norm_d07_clean_normalized_delivery_publishes_through_existing_persistence_api(
    tmp_path,
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    result = _process(
        (raw_record(),),
        root=tmp_path,
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        build_id="norm-d07",
    )
    paths = EarningsPersistencePaths(tmp_path)
    persisted = read_canonical_revisions(paths.published_path)
    assert result.success
    assert result.publication_result is not None
    assert result.publication_result.success
    assert persisted == result.normalization_result.revisions


def test_norm_d08_upstream_failure_leaves_previous_publication_unchanged(
    tmp_path,
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    first = _process(
        (raw_record(),),
        root=tmp_path,
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        build_id="norm-d08-prior",
    )
    assert first.success
    paths = EarningsPersistencePaths(tmp_path)
    parquet_before = paths.published_path.read_bytes()
    manifest_before = paths.manifest_path.read_bytes()
    failed = _process(
        (raw_record(revision_key="BAD", mapping_evidence=()),),
        root=tmp_path,
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        build_id="norm-d08-failed",
    )
    assert not failed.success
    assert failed.publication_result is None
    assert paths.published_path.read_bytes() == parquet_before
    assert paths.manifest_path.read_bytes() == manifest_before


def test_norm_d09_clean_delivery_creates_no_normalization_quarantine_artifact(
    tmp_path,
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    result = _process(
        (raw_record(),),
        root=tmp_path,
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        build_id="norm-d09",
    )
    paths = EarningsPersistencePaths(tmp_path)
    assert result.success
    assert result.normalization_result.quarantines == ()
    assert not (
        paths.quarantine_dir / "earnings_quarantine_norm-d09.parquet"
    ).exists()
