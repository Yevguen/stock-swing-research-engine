"""Manifest I/O, idempotency classification, and fail-closed publication."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stock_swing_d1.earnings.models import (
    EarningsScheduleRevision,
    EarningsValidationError,
)
from stock_swing_d1.earnings.persistence.hashing import (
    canonical_record_hash,
    canonical_sort_key,
    event_history_key,
    revision_identity_key,
)
from stock_swing_d1.earnings.persistence.models import (
    EarningsPersistenceError,
    EarningsPersistencePaths,
    EarningsPublicationManifest,
    EarningsPublicationResult,
    EarningsQuarantineRecord,
    QuarantineStage,
)
from stock_swing_d1.earnings.persistence.parquet import (
    _validate_persistence_fields,
    read_canonical_revisions,
    write_canonical_revisions,
    write_quarantine_records,
)
from stock_swing_d1.earnings.persistence.schema import (
    CANONICAL_EARNINGS_SCHEMA_VERSION,
    CANONICAL_RECORD_HASH_ALGORITHM,
    CANONICAL_SORT_KEY_VERSION,
    OUTPUT_HASH_ALGORITHM,
)
from stock_swing_d1.earnings.validation import (
    validate_revision,
    validate_revision_history,
)


def _utc_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "manifest timestamps must be timezone-aware"
        )
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc_text(value: Any, field_name: str, *, nullable: bool = False) -> datetime | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", f"{field_name} must be a UTC timestamp string"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", f"{field_name} is not a valid timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", f"{field_name} must be timezone-aware"
        )
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", f"{field_name} must be represented in UTC"
        )
    return parsed.astimezone(timezone.utc)


def _manifest_dict(manifest: EarningsPublicationManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "build_id": manifest.build_id,
        "started_at": _utc_text(manifest.started_at),
        "completed_at": _utc_text(manifest.completed_at),
        "provider_name": manifest.provider_name,
        "source_delivery_id": manifest.source_delivery_id,
        "input_record_count": manifest.input_record_count,
        "published_record_count": manifest.published_record_count,
        "published_event_count": manifest.published_event_count,
        "published_asset_count": manifest.published_asset_count,
        "minimum_strategy_effective_at": _utc_text(
            manifest.minimum_strategy_effective_at
        ),
        "maximum_strategy_effective_at": _utc_text(
            manifest.maximum_strategy_effective_at
        ),
        "canonical_sort_key_version": manifest.canonical_sort_key_version,
        "canonical_record_hash_algorithm": manifest.canonical_record_hash_algorithm,
        "output_hash_algorithm": manifest.output_hash_algorithm,
        "output_file": manifest.output_file,
        "output_sha256": manifest.output_sha256,
        "success": manifest.success,
    }


def _require_manifest_text(value: Any, field_name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", f"{field_name} must be a non-empty string"
        )
    return value


def _require_manifest_count(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", f"{field_name} must be a non-negative integer"
        )
    return value


def _manifest_from_dict(data: Any) -> EarningsPublicationManifest:
    if not isinstance(data, dict):
        raise EarningsPersistenceError("MANIFEST_INVALID", "manifest must be an object")
    expected_fields = {field.name for field in fields(EarningsPublicationManifest)}
    if set(data) != expected_fields:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "manifest fields do not match the v0.1 contract"
        )
    schema_version = _require_manifest_text(data["schema_version"], "schema_version")
    sort_version = _require_manifest_text(
        data["canonical_sort_key_version"], "canonical_sort_key_version"
    )
    record_algorithm = _require_manifest_text(
        data["canonical_record_hash_algorithm"],
        "canonical_record_hash_algorithm",
    )
    output_algorithm = _require_manifest_text(
        data["output_hash_algorithm"], "output_hash_algorithm"
    )
    if (
        schema_version != CANONICAL_EARNINGS_SCHEMA_VERSION
        or sort_version != CANONICAL_SORT_KEY_VERSION
        or record_algorithm != CANONICAL_RECORD_HASH_ALGORITHM
        or output_algorithm != OUTPUT_HASH_ALGORITHM
    ):
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "manifest version or hash algorithm is unsupported"
        )
    started_at = _parse_utc_text(data["started_at"], "started_at")
    completed_at = _parse_utc_text(data["completed_at"], "completed_at")
    minimum = _parse_utc_text(
        data["minimum_strategy_effective_at"],
        "minimum_strategy_effective_at",
        nullable=True,
    )
    maximum = _parse_utc_text(
        data["maximum_strategy_effective_at"],
        "maximum_strategy_effective_at",
        nullable=True,
    )
    assert started_at is not None and completed_at is not None
    if completed_at < started_at:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "completed_at cannot precede started_at"
        )
    if (minimum is None) != (maximum is None) or (
        minimum is not None and maximum is not None and maximum < minimum
    ):
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "manifest coverage bounds are inconsistent"
        )
    success = data["success"]
    if success is not True:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "a published manifest must represent success"
        )
    output_sha256 = _require_manifest_text(data["output_sha256"], "output_sha256")
    assert output_sha256 is not None
    if len(output_sha256) != 64 or output_sha256.lower() != output_sha256:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "output_sha256 must be lowercase SHA-256 hex"
        )
    try:
        int(output_sha256, 16)
    except ValueError as exc:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "output_sha256 must be lowercase SHA-256 hex"
        ) from exc
    input_count = _require_manifest_count(
        data["input_record_count"], "input_record_count"
    )
    record_count = _require_manifest_count(
        data["published_record_count"], "published_record_count"
    )
    event_count = _require_manifest_count(
        data["published_event_count"], "published_event_count"
    )
    asset_count = _require_manifest_count(
        data["published_asset_count"], "published_asset_count"
    )
    if (
        record_count < 1
        or event_count < 1
        or asset_count < 1
        or event_count > record_count
        or asset_count > record_count
    ):
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "published manifest counts are inconsistent"
        )
    if minimum is None:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "published manifest requires coverage bounds"
        )
    return EarningsPublicationManifest(
        schema_version=schema_version,
        build_id=_require_manifest_text(data["build_id"], "build_id"),  # type: ignore[arg-type]
        started_at=started_at,
        completed_at=completed_at,
        provider_name=_require_manifest_text(
            data["provider_name"], "provider_name", nullable=True
        ),
        source_delivery_id=_require_manifest_text(
            data["source_delivery_id"], "source_delivery_id", nullable=True
        ),
        input_record_count=input_count,
        published_record_count=record_count,
        published_event_count=event_count,
        published_asset_count=asset_count,
        minimum_strategy_effective_at=minimum,
        maximum_strategy_effective_at=maximum,
        canonical_sort_key_version=sort_version,
        canonical_record_hash_algorithm=record_algorithm,
        output_hash_algorithm=output_algorithm,
        output_file=_require_manifest_text(
            data["output_file"], "output_file"
        ),  # type: ignore[arg-type]
        output_sha256=output_sha256,
        success=True,
    )


def write_publication_manifest(
    path: Path,
    manifest: EarningsPublicationManifest,
    *,
    overwrite: bool = False,
) -> None:
    """Write deterministic compact JSON after strict manifest validation."""

    if not isinstance(manifest, EarningsPublicationManifest):
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", "expected EarningsPublicationManifest"
        )
    if path.exists() and not overwrite:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", f"refusing to overwrite existing path: {path}"
        )
    data = _manifest_dict(manifest)
    _manifest_from_dict(data)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", f"failed to write manifest {path}: {exc}"
        ) from exc


def read_publication_manifest(path: Path) -> EarningsPublicationManifest:
    """Read one strict v0.1 publication manifest without repair."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EarningsPersistenceError(
            "MANIFEST_INVALID", f"failed to read manifest {path}: {exc}"
        ) from exc
    return _manifest_from_dict(data)


@dataclass(frozen=True, slots=True)
class _IdempotencyConflict:
    incoming: EarningsScheduleRevision
    prior: EarningsScheduleRevision


@dataclass(frozen=True, slots=True)
class _IdempotencyClassification:
    insert_candidates: tuple[EarningsScheduleRevision, ...]
    idempotent_duplicate_count: int
    conflicts: tuple[_IdempotencyConflict, ...]


def _classify_revisions(
    incoming: tuple[EarningsScheduleRevision, ...],
    existing: tuple[EarningsScheduleRevision, ...],
) -> _IdempotencyClassification:
    """Classify provider-scoped immutable identities for one build."""

    existing_by_identity = {
        revision_identity_key(revision): revision for revision in existing
    }
    first_incoming: dict[tuple[str, str], EarningsScheduleRevision] = {}
    unique_incoming: list[EarningsScheduleRevision] = []
    conflicts: list[_IdempotencyConflict] = []
    duplicate_count = 0
    for revision in incoming:
        identity = revision_identity_key(revision)
        prior_in_batch = first_incoming.get(identity)
        if prior_in_batch is None:
            first_incoming[identity] = revision
            unique_incoming.append(revision)
        elif canonical_record_hash(prior_in_batch) == canonical_record_hash(revision):
            duplicate_count += 1
        else:
            conflicts.append(_IdempotencyConflict(revision, prior_in_batch))

    inserts: list[EarningsScheduleRevision] = []
    for revision in unique_incoming:
        prior = existing_by_identity.get(revision_identity_key(revision))
        if prior is None:
            inserts.append(revision)
        elif canonical_record_hash(prior) == canonical_record_hash(revision):
            duplicate_count += 1
        else:
            conflicts.append(_IdempotencyConflict(revision, prior))
    return _IdempotencyClassification(
        insert_candidates=tuple(inserts),
        idempotent_duplicate_count=duplicate_count,
        conflicts=tuple(conflicts),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _incoming_provider_name(
    incoming: tuple[EarningsScheduleRevision, ...],
) -> str | None:
    if not incoming:
        return None
    first = incoming[0].provider_name
    if isinstance(first, str) and all(
        revision.provider_name == first for revision in incoming
    ):
        return first
    return None


def _publication_metrics(
    revisions: tuple[EarningsScheduleRevision, ...],
) -> tuple[int, int, datetime, datetime]:
    event_count = len({event_history_key(revision) for revision in revisions})
    asset_count = len({revision.canonical_asset_id for revision in revisions})
    effective_times = [
        revision.strategy_effective_at.astimezone(timezone.utc)
        for revision in revisions
        if revision.strategy_effective_at is not None
    ]
    if not effective_times:
        raise EarningsPersistenceError(
            "SCHEMA_MISMATCH", "published revisions require strategy coverage"
        )
    return event_count, asset_count, min(effective_times), max(effective_times)


def _verify_existing_publication(
    paths: EarningsPersistencePaths,
) -> tuple[tuple[EarningsScheduleRevision, ...], str | None]:
    parquet_exists = paths.published_path.exists()
    manifest_exists = paths.manifest_path.exists()
    if not parquet_exists and not manifest_exists:
        return (), None
    if parquet_exists != manifest_exists:
        raise EarningsPersistenceError(
            "EXISTING_PUBLICATION_INVALID",
            "published Parquet and manifest must either both exist or both be absent",
        )
    try:
        revisions = read_canonical_revisions(paths.published_path)
        manifest = read_publication_manifest(paths.manifest_path)
        checksum = _file_sha256(paths.published_path)
        event_count, asset_count, minimum, maximum = _publication_metrics(revisions)
        if (
            manifest.output_file != paths.published_path.name
            or manifest.output_sha256 != checksum
            or manifest.published_record_count != len(revisions)
            or manifest.published_event_count != event_count
            or manifest.published_asset_count != asset_count
            or manifest.minimum_strategy_effective_at != minimum
            or manifest.maximum_strategy_effective_at != maximum
        ):
            raise EarningsPersistenceError(
                "EXISTING_PUBLICATION_INVALID",
                "published manifest does not match its canonical artifact",
            )
        return revisions, checksum
    except EarningsPersistenceError as exc:
        if exc.code == "EXISTING_PUBLICATION_INVALID":
            raise
        raise EarningsPersistenceError(
            "EXISTING_PUBLICATION_INVALID", str(exc)
        ) from exc
    except Exception as exc:
        raise EarningsPersistenceError(
            "EXISTING_PUBLICATION_INVALID", f"cannot verify publication: {exc}"
        ) from exc


def _replace_publication(
    *,
    candidate_path: Path,
    candidate_manifest_path: Path,
    paths: EarningsPersistencePaths,
    build_id: str,
) -> None:
    """Replace the artifact pair, rolling both back on any replacement failure."""

    paths.published_path.parent.mkdir(parents=True, exist_ok=True)
    paths.temporary_dir.mkdir(parents=True, exist_ok=True)
    backup_parquet = paths.temporary_dir / f"{build_id}.previous.parquet"
    backup_manifest = paths.temporary_dir / f"{build_id}.previous.manifest.json"
    had_parquet = paths.published_path.exists()
    had_manifest = paths.manifest_path.exists()
    parquet_replaced = False
    manifest_replaced = False
    try:
        if had_parquet:
            shutil.copy2(paths.published_path, backup_parquet)
        if had_manifest:
            shutil.copy2(paths.manifest_path, backup_manifest)
        os.replace(candidate_path, paths.published_path)
        parquet_replaced = True
        os.replace(candidate_manifest_path, paths.manifest_path)
        manifest_replaced = True
    except Exception as exc:
        rollback_error: Exception | None = None
        try:
            if parquet_replaced:
                if had_parquet:
                    shutil.copyfile(backup_parquet, paths.published_path)
                elif paths.published_path.exists():
                    paths.published_path.unlink()
            if manifest_replaced:
                if had_manifest:
                    shutil.copyfile(backup_manifest, paths.manifest_path)
                elif paths.manifest_path.exists():
                    paths.manifest_path.unlink()
        except Exception as restore_exc:
            rollback_error = restore_exc
        message = f"failed to replace publication atomically: {exc}"
        if rollback_error is not None:
            message += f"; rollback failed: {rollback_error}"
        raise EarningsPersistenceError("ATOMIC_PUBLICATION_FAILED", message) from exc
    finally:
        for backup in (backup_parquet, backup_manifest):
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass


def publish_earnings_revisions(
    incoming_revisions: Iterable[EarningsScheduleRevision],
    *,
    paths: EarningsPersistencePaths,
    source_delivery_id: str | None = None,
    build_id: str | None = None,
) -> EarningsPublicationResult:
    """Validate and publish one all-or-nothing canonical earnings build."""

    incoming = tuple(incoming_revisions)
    started_at = datetime.now(timezone.utc)
    actual_build_id = build_id if build_id is not None else uuid.uuid4().hex
    provider_name = _incoming_provider_name(incoming)
    quarantines: list[EarningsQuarantineRecord] = []
    validated: list[EarningsScheduleRevision] = []
    validation_error_count = 0
    duplicate_count = 0
    conflict_count = 0
    temp_parquet = paths.temporary_dir / f"{actual_build_id}.candidate.parquet"
    temp_manifest = paths.temporary_dir / f"{actual_build_id}.candidate.manifest.json"

    def add_quarantine(
        stage: QuarantineStage,
        code: str,
        message: str,
        revision: EarningsScheduleRevision | None = None,
        *,
        details_json: str | None = None,
    ) -> None:
        semantic_hash: str | None = None
        if revision is not None:
            try:
                semantic_hash = canonical_record_hash(revision)
            except Exception:
                semantic_hash = None
        quarantines.append(
            EarningsQuarantineRecord(
                quarantine_id=f"{actual_build_id}-{len(quarantines) + 1:04d}",
                build_id=actual_build_id,
                quarantined_at=datetime.now(timezone.utc),
                provider_name=(revision.provider_name if revision is not None else None),
                provider_entity_id=(
                    revision.provider_entity_id if revision is not None else None
                ),
                canonical_asset_id=(
                    revision.canonical_asset_id if revision is not None else None
                ),
                event_instance_id=(
                    revision.event_instance_id if revision is not None else None
                ),
                event_revision_id=(
                    revision.event_revision_id if revision is not None else None
                ),
                stage=stage,
                error_code=code,
                error_message=message,
                source_record_locator=None,
                source_record_hash=None,
                canonical_record_hash=semantic_hash,
                details_json=details_json,
            )
        )

    def finish_failure() -> EarningsPublicationResult:
        if quarantines:
            quarantine_path = (
                paths.quarantine_dir
                / f"earnings_quarantine_{actual_build_id}.parquet"
            )
            try:
                write_quarantine_records(
                    quarantine_path, quarantines, overwrite=quarantine_path.exists()
                )
            except EarningsPersistenceError:
                # The build remains failed even when its diagnostic artifact cannot be written.
                pass
        completed_at = datetime.now(timezone.utc)
        return EarningsPublicationResult(
            schema_version=CANONICAL_EARNINGS_SCHEMA_VERSION,
            build_id=actual_build_id,
            started_at=started_at,
            completed_at=completed_at,
            provider_name=provider_name,
            input_record_count=len(incoming),
            validated_record_count=len(validated),
            inserted_count=0,
            idempotent_duplicate_count=duplicate_count,
            conflict_count=conflict_count,
            quarantined_count=len(quarantines),
            validation_error_count=validation_error_count,
            published_record_count=0,
            published_event_count=0,
            published_asset_count=0,
            minimum_strategy_effective_at=None,
            maximum_strategy_effective_at=None,
            output_path=None,
            output_sha256=None,
            success=False,
        )

    try:
        if not incoming:
            add_quarantine(
                QuarantineStage.PUBLICATION,
                "EMPTY_INPUT_DATASET",
                "an empty input dataset cannot be published",
            )
            return finish_failure()

        for revision in incoming:
            try:
                validate_revision(revision)
                _validate_persistence_fields(revision)
                validated.append(revision)
            except (EarningsValidationError, EarningsPersistenceError) as exc:
                validation_error_count += 1
                add_quarantine(
                    QuarantineStage.RECORD_VALIDATION,
                    exc.code,
                    str(exc),
                    revision,
                )
            except Exception as exc:
                validation_error_count += 1
                add_quarantine(
                    QuarantineStage.RECORD_VALIDATION,
                    "RECORD_VALIDATION_FAILED",
                    str(exc),
                    revision,
                )

        existing: tuple[EarningsScheduleRevision, ...] = ()
        existing_sha256: str | None = None
        try:
            existing, existing_sha256 = _verify_existing_publication(paths)
        except EarningsPersistenceError as exc:
            add_quarantine(
                QuarantineStage.PUBLICATION,
                "EXISTING_PUBLICATION_INVALID",
                str(exc),
            )

        classification = _classify_revisions(tuple(validated), existing)
        duplicate_count = classification.idempotent_duplicate_count
        conflict_count = len(classification.conflicts)
        for conflict in classification.conflicts:
            details = json.dumps(
                {
                    "incoming_canonical_record_hash": canonical_record_hash(
                        conflict.incoming
                    ),
                    "prior_canonical_record_hash": canonical_record_hash(conflict.prior),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            add_quarantine(
                QuarantineStage.IDEMPOTENCY,
                "IMMUTABLE_REVISION_CONFLICT",
                "an immutable provider revision identity has different semantics",
                conflict.incoming,
                details_json=details,
            )

        candidate = existing + classification.insert_candidates
        histories: dict[
            tuple[str, str], list[EarningsScheduleRevision]
        ] = defaultdict(list)
        for revision in candidate:
            histories[event_history_key(revision)].append(revision)
        incoming_insert_ids = {
            revision_identity_key(revision)
            for revision in classification.insert_candidates
        }
        for history in histories.values():
            try:
                validate_revision_history(history)
            except EarningsValidationError as exc:
                validation_error_count += 1
                diagnostic_revision = next(
                    (
                        revision
                        for revision in reversed(history)
                        if revision_identity_key(revision) in incoming_insert_ids
                    ),
                    history[-1],
                )
                add_quarantine(
                    QuarantineStage.HISTORY_VALIDATION,
                    exc.code,
                    str(exc),
                    diagnostic_revision,
                )

        if quarantines:
            return finish_failure()

        if not classification.insert_candidates:
            if not existing or existing_sha256 is None:
                add_quarantine(
                    QuarantineStage.PUBLICATION,
                    "EMPTY_INPUT_DATASET",
                    "no canonical revisions remain to publish",
                )
                return finish_failure()
            event_count, asset_count, minimum, maximum = _publication_metrics(existing)
            return EarningsPublicationResult(
                schema_version=CANONICAL_EARNINGS_SCHEMA_VERSION,
                build_id=actual_build_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                provider_name=provider_name,
                input_record_count=len(incoming),
                validated_record_count=len(validated),
                inserted_count=0,
                idempotent_duplicate_count=duplicate_count,
                conflict_count=0,
                quarantined_count=0,
                validation_error_count=0,
                published_record_count=len(existing),
                published_event_count=event_count,
                published_asset_count=asset_count,
                minimum_strategy_effective_at=minimum,
                maximum_strategy_effective_at=maximum,
                output_path=str(paths.published_path),
                output_sha256=existing_sha256,
                success=True,
            )

        ordered_candidate = tuple(sorted(candidate, key=canonical_sort_key))
        try:
            paths.temporary_dir.mkdir(parents=True, exist_ok=True)
            artifact = write_canonical_revisions(
                temp_parquet, ordered_candidate, overwrite=temp_parquet.exists()
            )
            verified = read_canonical_revisions(temp_parquet)
            if verified != ordered_candidate or artifact.row_count != len(ordered_candidate):
                raise EarningsPersistenceError(
                    "PARQUET_WRITE_FAILED",
                    "candidate semantic contents did not verify",
                )
            event_count, asset_count, minimum, maximum = _publication_metrics(verified)
            completed_at = datetime.now(timezone.utc)
            manifest = EarningsPublicationManifest(
                schema_version=CANONICAL_EARNINGS_SCHEMA_VERSION,
                build_id=actual_build_id,
                started_at=started_at,
                completed_at=completed_at,
                provider_name=provider_name,
                source_delivery_id=source_delivery_id,
                input_record_count=len(incoming),
                published_record_count=len(verified),
                published_event_count=event_count,
                published_asset_count=asset_count,
                minimum_strategy_effective_at=minimum,
                maximum_strategy_effective_at=maximum,
                canonical_sort_key_version=CANONICAL_SORT_KEY_VERSION,
                canonical_record_hash_algorithm=CANONICAL_RECORD_HASH_ALGORITHM,
                output_hash_algorithm=OUTPUT_HASH_ALGORITHM,
                output_file=paths.published_path.name,
                output_sha256=artifact.sha256,
                success=True,
            )
            write_publication_manifest(
                temp_manifest, manifest, overwrite=temp_manifest.exists()
            )
            _replace_publication(
                candidate_path=temp_parquet,
                candidate_manifest_path=temp_manifest,
                paths=paths,
                build_id=actual_build_id,
            )
        except EarningsPersistenceError as exc:
            add_quarantine(QuarantineStage.PUBLICATION, exc.code, str(exc))
            return finish_failure()
        except Exception as exc:
            add_quarantine(
                QuarantineStage.PUBLICATION,
                "ATOMIC_PUBLICATION_FAILED",
                str(exc),
            )
            return finish_failure()

        return EarningsPublicationResult(
            schema_version=CANONICAL_EARNINGS_SCHEMA_VERSION,
            build_id=actual_build_id,
            started_at=started_at,
            completed_at=completed_at,
            provider_name=provider_name,
            input_record_count=len(incoming),
            validated_record_count=len(validated),
            inserted_count=len(classification.insert_candidates),
            idempotent_duplicate_count=duplicate_count,
            conflict_count=0,
            quarantined_count=0,
            validation_error_count=0,
            published_record_count=len(ordered_candidate),
            published_event_count=event_count,
            published_asset_count=asset_count,
            minimum_strategy_effective_at=minimum,
            maximum_strategy_effective_at=maximum,
            output_path=str(paths.published_path),
            output_sha256=artifact.sha256,
            success=True,
        )
    finally:
        for temporary_path in (temp_parquet, temp_manifest):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
