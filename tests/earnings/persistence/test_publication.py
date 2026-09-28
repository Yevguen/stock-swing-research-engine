"""Suite D: fail-closed publication acceptance."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone

import pytest

import stock_swing_d1.earnings.persistence.publication as publication_module
from stock_swing_d1.earnings import LifecycleState, TransitionType
from stock_swing_d1.earnings.persistence import (
    EarningsPersistenceError,
    EarningsPersistencePaths,
    publish_earnings_revisions,
    read_canonical_revisions,
    read_publication_manifest,
)


def _paths(tmp_path) -> EarningsPersistencePaths:
    return EarningsPersistencePaths(tmp_path / "earnings")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history_failure_revision(make_persist_revision):
    known_at = datetime(2026, 9, 2, 14, tzinfo=timezone.utc)
    return make_persist_revision(
        event_revision_id="SYNTH-REV-002",
        knowledge_date=known_at.date(),
        knowledge_available_at=known_at,
        strategy_effective_at=known_at,
        provider_sequence=2,
        transition_type=TransitionType.POSTPONED,
        lifecycle_state=LifecycleState.ESTIMATED,
        scheduled_date=date(2026, 10, 20),
        ingested_at=known_at,
    )


def test_persist_d01_first_valid_publication_succeeds(tmp_path, make_persist_revision):
    """PERSIST_D01"""
    paths = _paths(tmp_path)
    result = publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="D01"
    )
    assert result.success is True
    assert paths.published_path.exists()
    assert paths.manifest_path.exists()


def test_persist_d02_success_result_counts_are_exact(tmp_path, make_persist_revision):
    """PERSIST_D02"""
    result = publish_earnings_revisions(
        [make_persist_revision()], paths=_paths(tmp_path), build_id="D02"
    )
    assert result.input_record_count == 1
    assert result.validated_record_count == 1
    assert result.inserted_count == 1
    assert result.idempotent_duplicate_count == 0
    assert result.conflict_count == 0
    assert result.quarantined_count == 0
    assert result.validation_error_count == 0
    assert result.published_record_count == 1
    assert result.published_event_count == 1
    assert result.published_asset_count == 1


def test_persist_d03_multi_asset_multi_event_counts_are_correct(
    tmp_path, make_persist_revision
):
    """PERSIST_D03"""
    revisions = [
        make_persist_revision(event_revision_id="R1", event_instance_id="E1"),
        make_persist_revision(event_revision_id="R2", event_instance_id="E2"),
        make_persist_revision(
            event_revision_id="R3",
            event_instance_id="E3",
            canonical_asset_id="SYNTH-ASSET-002",
        ),
    ]
    result = publish_earnings_revisions(
        revisions, paths=_paths(tmp_path), build_id="D03"
    )
    assert result.success is True
    assert result.published_record_count == 3
    assert result.published_event_count == 3
    assert result.published_asset_count == 2


def test_persist_d04_coverage_bounds_use_complete_published_dataset(
    tmp_path, make_persist_revision
):
    """PERSIST_D04"""
    paths = _paths(tmp_path)
    early = datetime(2026, 8, 1, 14, tzinfo=timezone.utc)
    late = datetime(2026, 10, 1, 14, tzinfo=timezone.utc)
    first = make_persist_revision(
        event_revision_id="EARLY",
        event_instance_id="EVENT-EARLY",
        knowledge_date=early.date(),
        knowledge_available_at=early,
        strategy_effective_at=early,
        ingested_at=early,
    )
    second = make_persist_revision(
        event_revision_id="LATE",
        event_instance_id="EVENT-LATE",
        knowledge_date=late.date(),
        knowledge_available_at=late,
        strategy_effective_at=late,
        ingested_at=late,
    )
    assert publish_earnings_revisions([first], paths=paths, build_id="D04-A").success
    result = publish_earnings_revisions([second], paths=paths, build_id="D04-B")
    assert result.minimum_strategy_effective_at == early
    assert result.maximum_strategy_effective_at == late
    assert result.published_record_count == 2


def test_persist_d05_manifest_matches_published_artifact_checksum(
    tmp_path, make_persist_revision
):
    """PERSIST_D05"""
    paths = _paths(tmp_path)
    result = publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="D05"
    )
    manifest = read_publication_manifest(paths.manifest_path)
    assert manifest.output_sha256 == _sha256(paths.published_path)
    assert manifest.output_sha256 == result.output_sha256
    assert manifest.output_file == paths.published_path.name


def test_persist_d06_all_duplicate_reingestion_succeeds_without_rewriting_artifact(
    tmp_path, make_persist_revision
):
    """PERSIST_D06"""
    paths = _paths(tmp_path)
    revision = make_persist_revision()
    first = publish_earnings_revisions([revision], paths=paths, build_id="D06-A")
    parquet_before = paths.published_path.read_bytes()
    manifest_before = paths.manifest_path.read_bytes()
    mtime_before = paths.published_path.stat().st_mtime_ns
    second = publish_earnings_revisions([revision], paths=paths, build_id="D06-B")
    assert first.success and second.success
    assert second.inserted_count == 0
    assert second.idempotent_duplicate_count == 1
    assert second.output_sha256 == first.output_sha256
    assert paths.published_path.read_bytes() == parquet_before
    assert paths.manifest_path.read_bytes() == manifest_before
    assert paths.published_path.stat().st_mtime_ns == mtime_before


def test_persist_d07_record_validation_failure_blocks_publication(
    tmp_path, make_persist_revision
):
    """PERSIST_D07"""
    paths = _paths(tmp_path)
    invalid = make_persist_revision(canonical_asset_id=None)
    result = publish_earnings_revisions([invalid], paths=paths, build_id="D07")
    assert result.success is False
    assert result.validated_record_count == 0
    assert result.validation_error_count == 1
    assert result.published_record_count == 0
    assert not paths.published_path.exists()


def test_persist_d08_history_validation_failure_blocks_publication(
    tmp_path, make_persist_revision
):
    """PERSIST_D08"""
    paths = _paths(tmp_path)
    assert publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="D08-A"
    ).success
    result = publish_earnings_revisions(
        [_history_failure_revision(make_persist_revision)],
        paths=paths,
        build_id="D08-B",
    )
    assert result.success is False
    assert result.validation_error_count == 1
    assert result.inserted_count == 0


def test_persist_d09_immutable_revision_conflict_blocks_publication(
    tmp_path, make_persist_revision
):
    """PERSIST_D09"""
    paths = _paths(tmp_path)
    original = make_persist_revision()
    assert publish_earnings_revisions([original], paths=paths, build_id="D09-A").success
    conflict = make_persist_revision(scheduled_date=date(2026, 11, 5))
    result = publish_earnings_revisions([conflict], paths=paths, build_id="D09-B")
    assert result.success is False
    assert result.conflict_count == 1
    assert result.validation_error_count == 0
    assert result.inserted_count == 0


def test_persist_d10_empty_input_cannot_replace_existing_publication(
    tmp_path, make_persist_revision
):
    """PERSIST_D10"""
    paths = _paths(tmp_path)
    assert publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="D10-A"
    ).success
    parquet_before = paths.published_path.read_bytes()
    manifest_before = paths.manifest_path.read_bytes()
    result = publish_earnings_revisions([], paths=paths, build_id="D10-B")
    assert result.success is False
    assert result.input_record_count == 0
    assert result.inserted_count == 0
    assert paths.published_path.read_bytes() == parquet_before
    assert paths.manifest_path.read_bytes() == manifest_before


def test_persist_d11_failed_candidate_leaves_previous_parquet_unchanged(
    tmp_path, make_persist_revision
):
    """PERSIST_D11"""
    paths = _paths(tmp_path)
    assert publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="D11-A"
    ).success
    checksum_before = _sha256(paths.published_path)
    result = publish_earnings_revisions(
        [_history_failure_revision(make_persist_revision)],
        paths=paths,
        build_id="D11-B",
    )
    assert result.success is False
    assert _sha256(paths.published_path) == checksum_before


def test_persist_d12_failed_candidate_leaves_previous_manifest_unchanged(
    tmp_path, make_persist_revision
):
    """PERSIST_D12"""
    paths = _paths(tmp_path)
    assert publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="D12-A"
    ).success
    manifest_before = paths.manifest_path.read_bytes()
    result = publish_earnings_revisions(
        [_history_failure_revision(make_persist_revision)],
        paths=paths,
        build_id="D12-B",
    )
    assert result.success is False
    assert paths.manifest_path.read_bytes() == manifest_before


def test_persist_d13_invalid_existing_publication_is_not_silently_replaced(
    tmp_path, make_persist_revision
):
    """PERSIST_D13"""
    paths = _paths(tmp_path)
    assert publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="D13-A"
    ).success
    paths.published_path.write_bytes(b"intentionally-invalid-parquet")
    corrupted = paths.published_path.read_bytes()
    new_revision = make_persist_revision(
        event_revision_id="NEW", event_instance_id="NEW-EVENT"
    )
    result = publish_earnings_revisions(
        [new_revision], paths=paths, build_id="D13-B"
    )
    assert result.success is False
    assert paths.published_path.read_bytes() == corrupted


def test_persist_d14_failed_publication_returns_no_new_output_path_or_hash(
    tmp_path, make_persist_revision
):
    """PERSIST_D14"""
    result = publish_earnings_revisions(
        [make_persist_revision(canonical_asset_id=None)],
        paths=_paths(tmp_path),
        build_id="D14",
    )
    assert result.success is False
    assert result.output_path is None
    assert result.output_sha256 is None
    assert result.published_record_count == 0
    assert result.published_event_count == 0
    assert result.published_asset_count == 0


def test_persist_d15_successful_publication_cleans_temporary_files(
    tmp_path, make_persist_revision
):
    """PERSIST_D15"""
    paths = _paths(tmp_path)
    result = publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="D15"
    )
    assert result.success is True
    assert paths.temporary_dir.exists()
    assert list(paths.temporary_dir.iterdir()) == []


def test_persist_d16_failed_publication_cleans_temporary_files(
    tmp_path, make_persist_revision, monkeypatch
):
    """PERSIST_D16"""
    paths = _paths(tmp_path)

    def fail_manifest(*args, **kwargs):
        raise EarningsPersistenceError("MANIFEST_INVALID", "synthetic failure")

    monkeypatch.setattr(publication_module, "write_publication_manifest", fail_manifest)
    result = publish_earnings_revisions(
        [make_persist_revision()], paths=paths, build_id="D16"
    )
    assert result.success is False
    assert paths.temporary_dir.exists()
    assert list(paths.temporary_dir.iterdir()) == []


def test_persist_d17_published_output_passes_strict_reader_validation(
    tmp_path, make_persist_revision
):
    """PERSIST_D17"""
    paths = _paths(tmp_path)
    revision = make_persist_revision()
    result = publish_earnings_revisions([revision], paths=paths, build_id="D17")
    assert result.success is True
    assert read_canonical_revisions(paths.published_path) == (revision,)
