"""Provider-neutral delivery normalization and persistence handoff."""

from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Any, Protocol

from stock_swing_d1.earnings.models import (
    EarningsScheduleRevision,
    EarningsValidationError,
)
from stock_swing_d1.earnings.normalization.identity import (
    EarningsCanonicalHistoryReader,
    EarningsSecurityIdentityResolver,
    _ParquetCanonicalHistoryReader,
)
from stock_swing_d1.earnings.normalization.models import (
    EarningsAdapterBatch,
    EarningsDeliveryResult,
    EarningsKnowledgeCutoff,
    EarningsNormalizationFailure,
    EarningsNormalizationResult,
)
from stock_swing_d1.earnings.persistence import (
    EarningsPersistencePaths,
    EarningsQuarantineRecord,
    QuarantineStage,
    canonical_record_hash,
    canonical_sort_key,
    publish_earnings_revisions,
    write_quarantine_records,
)
from stock_swing_d1.earnings.pit import (
    EarningsKnowledgeCalendar,
    derive_strategy_effective_at,
)
from stock_swing_d1.earnings.validation import (
    validate_revision,
    validate_revision_history,
    validate_security_mapping,
)


_CANONICAL_ASSET_ID = re.compile(r"^NORGATE:[1-9][0-9]*$")
_PROHIBITED_DETAIL_KEYS = frozenset(
    {
        "raw",
        "raw_data",
        "raw_payload",
        "raw_provider_payload",
        "raw_record",
        "provider_payload",
        "provider_record",
        "source_payload",
        "source_record",
    }
)


class EarningsProviderDeliveryAdapter(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    def normalize_delivery(
        self,
        raw_records: tuple[Any, ...],
        *,
        identity_resolver: EarningsSecurityIdentityResolver,
        canonical_history: EarningsCanonicalHistoryReader | None = None,
    ) -> EarningsAdapterBatch:
        ...


def _require_aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    _require_aware(value, "datetime field")
    return value.astimezone(timezone.utc)


def _contains_prohibited_detail_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in _PROHIBITED_DETAIL_KEYS
            or _contains_prohibited_detail_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_prohibited_detail_key(item) for item in value)
    return False


def _safe_details_json(details_json: str | None) -> str | None:
    if details_json is None:
        return None
    try:
        details = json.loads(details_json)
    except (TypeError, json.JSONDecodeError):
        return None
    if _contains_prohibited_detail_key(details):
        return None
    return details_json


def _failure_quarantine(
    failure: EarningsNormalizationFailure,
    *,
    build_id: str,
    quarantined_at: datetime,
    ordinal: int,
) -> EarningsQuarantineRecord:
    return EarningsQuarantineRecord(
        quarantine_id=f"{build_id}-{ordinal:04d}",
        build_id=build_id,
        quarantined_at=quarantined_at,
        provider_name=failure.provider_name,
        provider_entity_id=failure.provider_entity_id,
        canonical_asset_id=failure.canonical_asset_id,
        event_instance_id=failure.event_instance_id,
        event_revision_id=failure.event_revision_id,
        stage=failure.stage,
        error_code=failure.error_code,
        error_message=failure.error_message,
        source_record_locator=failure.source_record_locator,
        source_record_hash=failure.source_record_hash,
        canonical_record_hash=None,
        details_json=_safe_details_json(failure.details_json),
    )


def _candidate_failure(
    *,
    stage: QuarantineStage,
    code: str,
    message: str,
    provider_name: str | None,
    revision: EarningsScheduleRevision | None = None,
) -> EarningsNormalizationFailure:
    return EarningsNormalizationFailure(
        stage=stage,
        error_code=code,
        error_message=message,
        provider_name=(revision.provider_name if revision is not None else provider_name),
        provider_entity_id=(
            revision.provider_entity_id if revision is not None else None
        ),
        canonical_asset_id=(
            revision.canonical_asset_id if revision is not None else None
        ),
        event_instance_id=(revision.event_instance_id if revision is not None else None),
        event_revision_id=(revision.event_revision_id if revision is not None else None),
        source_record_locator=None,
        source_record_hash=None,
        details_json=None,
    )


def _result(
    *,
    build_id: str,
    provider_name: str,
    source_delivery_id: str | None,
    input_record_count: int,
    normalized_record_count: int,
    finalized: Iterable[EarningsScheduleRevision],
    failures: Iterable[EarningsNormalizationFailure],
    quarantined_at: datetime,
) -> EarningsNormalizationResult:
    materialized_failures = tuple(failures)
    quarantines = tuple(
        _failure_quarantine(
            failure,
            build_id=build_id,
            quarantined_at=quarantined_at,
            ordinal=index,
        )
        for index, failure in enumerate(materialized_failures, start=1)
    )
    if quarantines:
        revisions: tuple[EarningsScheduleRevision, ...] = ()
        success = False
    else:
        revisions = tuple(sorted(finalized, key=canonical_sort_key))
        success = True
    return EarningsNormalizationResult(
        schema_version="0.1",
        build_id=build_id,
        provider_name=provider_name,
        source_delivery_id=source_delivery_id,
        input_record_count=input_record_count,
        normalized_record_count=normalized_record_count,
        quarantined_count=len(quarantines),
        revisions=revisions,
        quarantines=quarantines,
        success=success,
    )


def normalize_earnings_delivery(
    raw_records: Iterable[Any],
    *,
    adapter: EarningsProviderDeliveryAdapter,
    identity_resolver: EarningsSecurityIdentityResolver,
    trading_calendar: EarningsKnowledgeCalendar,
    canonical_history: EarningsCanonicalHistoryReader | None,
    build_id: str,
    ingested_at: datetime,
    source_delivery_id: str | None = None,
) -> EarningsNormalizationResult:
    """Finalize one adapter batch into canonical, validated PIT revisions."""

    materialized = tuple(raw_records)
    if not isinstance(build_id, str) or not build_id.strip():
        raise ValueError("build_id must be a non-empty string")
    provider_name = adapter.provider_name
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ValueError("adapter.provider_name must be a non-empty string")
    _require_aware(ingested_at, "ingested_at")
    normalized_ingested_at = ingested_at.astimezone(timezone.utc)

    failures: list[EarningsNormalizationFailure] = []
    finalized: list[EarningsScheduleRevision] = []
    if not materialized:
        failures.append(
            _candidate_failure(
                stage=QuarantineStage.NORMALIZATION,
                code="EMPTY_PROVIDER_DELIVERY",
                message="provider delivery contains no records",
                provider_name=provider_name,
            )
        )
        return _result(
            build_id=build_id,
            provider_name=provider_name,
            source_delivery_id=source_delivery_id,
            input_record_count=0,
            normalized_record_count=0,
            finalized=(),
            failures=failures,
            quarantined_at=normalized_ingested_at,
        )

    try:
        batch = adapter.normalize_delivery(
            materialized,
            identity_resolver=identity_resolver,
            canonical_history=canonical_history,
        )
    except Exception as exc:
        error_code = getattr(exc, "code", "PROVIDER_NORMALIZATION_FAILED")
        if not isinstance(error_code, str) or not error_code:
            error_code = "PROVIDER_NORMALIZATION_FAILED"
        failures.append(
            _candidate_failure(
                stage=QuarantineStage.NORMALIZATION,
                code=error_code,
                message="provider delivery normalization failed",
                provider_name=provider_name,
            )
        )
        return _result(
            build_id=build_id,
            provider_name=provider_name,
            source_delivery_id=source_delivery_id,
            input_record_count=len(materialized),
            normalized_record_count=0,
            finalized=(),
            failures=failures,
            quarantined_at=normalized_ingested_at,
        )

    if not isinstance(batch, EarningsAdapterBatch):
        failures.append(
            _candidate_failure(
                stage=QuarantineStage.NORMALIZATION,
                code="INVALID_ADAPTER_BATCH",
                message="adapter must return EarningsAdapterBatch",
                provider_name=provider_name,
            )
        )
        return _result(
            build_id=build_id,
            provider_name=provider_name,
            source_delivery_id=source_delivery_id,
            input_record_count=len(materialized),
            normalized_record_count=0,
            finalized=(),
            failures=failures,
            quarantined_at=normalized_ingested_at,
        )

    normalized_record_count = len(batch.revisions)
    for failure in batch.failures:
        if not isinstance(failure, EarningsNormalizationFailure):
            failures.append(
                _candidate_failure(
                    stage=QuarantineStage.NORMALIZATION,
                    code="INVALID_ADAPTER_FAILURE",
                    message="adapter failure must be EarningsNormalizationFailure",
                    provider_name=provider_name,
                )
            )
        elif failure.stage not in {
            QuarantineStage.NORMALIZATION,
            QuarantineStage.IDENTITY_MAPPING,
        }:
            failures.append(
                replace(
                    failure,
                    stage=QuarantineStage.NORMALIZATION,
                    error_code="INVALID_ADAPTER_FAILURE_STAGE",
                    error_message="adapter emitted a downstream quarantine stage",
                    details_json=None,
                )
            )
        else:
            failures.append(failure)

    for candidate in batch.revisions:
        if not isinstance(candidate, EarningsScheduleRevision):
            failures.append(
                _candidate_failure(
                    stage=QuarantineStage.NORMALIZATION,
                    code="NON_CANONICAL_REVISION",
                    message="adapter output must be EarningsScheduleRevision",
                    provider_name=provider_name,
                )
            )
            continue
        if candidate.provider_name != provider_name:
            failures.append(
                _candidate_failure(
                    stage=QuarantineStage.NORMALIZATION,
                    code="ADAPTER_PROVIDER_MISMATCH",
                    message="candidate provider_name differs from adapter provider_name",
                    provider_name=provider_name,
                    revision=candidate,
                )
            )
            continue
        if candidate.strategy_effective_at is not None or candidate.ingested_at is not None:
            failures.append(
                _candidate_failure(
                    stage=QuarantineStage.NORMALIZATION,
                    code="ADAPTER_OWNS_PIPELINE_FIELD",
                    message="adapter cannot set strategy_effective_at or ingested_at",
                    provider_name=provider_name,
                    revision=candidate,
                )
            )
            continue

        try:
            canonical_asset_id = validate_security_mapping(
                canonical_asset_id=candidate.canonical_asset_id,
                historical_symbol=candidate.historical_symbol,
                historical_exchange=candidate.historical_exchange,
            )
            if _CANONICAL_ASSET_ID.fullmatch(canonical_asset_id) is None:
                raise EarningsValidationError(
                    "INVALID_CANONICAL_ASSET_ID",
                    "canonical identity must use NORGATE:<positive AssetId>",
                )
        except EarningsValidationError as exc:
            failures.append(
                _candidate_failure(
                    stage=QuarantineStage.IDENTITY_MAPPING,
                    code=exc.code,
                    message=str(exc),
                    provider_name=provider_name,
                    revision=candidate,
                )
            )
            continue

        if type(candidate.knowledge_date) is not date:
            failures.append(
                _candidate_failure(
                    stage=QuarantineStage.NORMALIZATION,
                    code="INVALID_KNOWLEDGE_DATE",
                    message="candidate knowledge_date must be a genuine date",
                    provider_name=provider_name,
                    revision=candidate,
                )
            )
            continue

        try:
            effective_at = derive_strategy_effective_at(candidate, trading_calendar)
            completed = replace(
                candidate,
                knowledge_available_at=_utc(candidate.knowledge_available_at),
                strategy_effective_at=_utc(effective_at),
                scheduled_at=_utc(candidate.scheduled_at),
                actual_event_at=_utc(candidate.actual_event_at),
                ingested_at=normalized_ingested_at,
            )
            validate_revision(completed)
        except (EarningsValidationError, ValueError, TypeError) as exc:
            error_code = getattr(exc, "code", "CANONICAL_REVISION_INVALID")
            failures.append(
                _candidate_failure(
                    stage=QuarantineStage.NORMALIZATION,
                    code=error_code,
                    message=str(exc),
                    provider_name=provider_name,
                    revision=candidate,
                )
            )
            continue
        finalized.append(completed)

    histories: dict[tuple[str, str], list[EarningsScheduleRevision]] = defaultdict(list)
    for revision in finalized:
        assert revision.canonical_asset_id is not None
        assert revision.event_instance_id is not None
        histories[(revision.canonical_asset_id, revision.event_instance_id)].append(
            revision
        )

    for (canonical_asset_id, event_instance_id), new_revisions in histories.items():
        ordered_new = sorted(new_revisions, key=canonical_sort_key)
        extension: list[EarningsScheduleRevision] = []
        if canonical_history is not None:
            first = ordered_new[0]
            assert first.knowledge_date is not None
            try:
                antecedent = canonical_history.get_latest_revision(
                    provider_name=provider_name,
                    canonical_asset_id=canonical_asset_id,
                    event_instance_id=event_instance_id,
                    knowledge_cutoff=EarningsKnowledgeCutoff(
                        knowledge_precision=first.knowledge_precision,
                        knowledge_date=first.knowledge_date,
                        knowledge_available_at=first.knowledge_available_at,
                    ),
                )
            except Exception as exc:
                failures.append(
                    _candidate_failure(
                        stage=QuarantineStage.NORMALIZATION,
                        code=getattr(exc, "code", "CANONICAL_HISTORY_READ_FAILED"),
                        message="canonical history lookup failed",
                        provider_name=provider_name,
                        revision=first,
                    )
                )
                continue
            if antecedent is not None:
                extension.append(antecedent)
        extension.extend(ordered_new)
        try:
            validate_revision_history(extension)
        except EarningsValidationError as exc:
            failures.append(
                _candidate_failure(
                    stage=QuarantineStage.NORMALIZATION,
                    code=exc.code,
                    message=str(exc),
                    provider_name=provider_name,
                    revision=ordered_new[-1],
                )
            )

    return _result(
        build_id=build_id,
        provider_name=provider_name,
        source_delivery_id=source_delivery_id,
        input_record_count=len(materialized),
        normalized_record_count=normalized_record_count,
        finalized=finalized,
        failures=failures,
        quarantined_at=normalized_ingested_at,
    )


def process_earnings_delivery(
    raw_records: Iterable[Any],
    *,
    adapter: EarningsProviderDeliveryAdapter,
    identity_resolver: EarningsSecurityIdentityResolver,
    trading_calendar: EarningsKnowledgeCalendar,
    paths: EarningsPersistencePaths,
    canonical_history: EarningsCanonicalHistoryReader | None = None,
    source_delivery_id: str | None = None,
    build_id: str | None = None,
    ingested_at: datetime | None = None,
) -> EarningsDeliveryResult:
    """Normalize and atomically publish one provider delivery."""

    materialized = tuple(raw_records)
    actual_build_id = build_id if build_id is not None else uuid.uuid4().hex
    if not isinstance(actual_build_id, str) or not actual_build_id.strip():
        raise ValueError("build_id must be a non-empty string")
    actual_ingested_at = ingested_at or datetime.now(timezone.utc)
    _require_aware(actual_ingested_at, "ingested_at")
    actual_ingested_at = actual_ingested_at.astimezone(timezone.utc)

    resolved_history = canonical_history
    if resolved_history is None and paths.published_path.exists():
        try:
            resolved_history = _ParquetCanonicalHistoryReader(paths.published_path)
        except Exception as exc:
            failure = _candidate_failure(
                stage=QuarantineStage.NORMALIZATION,
                code=getattr(exc, "code", "CANONICAL_HISTORY_READ_FAILED"),
                message="published canonical history failed strict reading",
                provider_name=adapter.provider_name,
            )
            normalization_result = _result(
                build_id=actual_build_id,
                provider_name=adapter.provider_name,
                source_delivery_id=source_delivery_id,
                input_record_count=len(materialized),
                normalized_record_count=0,
                finalized=(),
                failures=(failure,),
                quarantined_at=actual_ingested_at,
            )
            quarantine_path = (
                paths.quarantine_dir
                / f"earnings_quarantine_{actual_build_id}.parquet"
            )
            write_quarantine_records(quarantine_path, normalization_result.quarantines)
            return EarningsDeliveryResult(
                normalization_result=normalization_result,
                publication_result=None,
                success=False,
            )

    normalization_result = normalize_earnings_delivery(
        materialized,
        adapter=adapter,
        identity_resolver=identity_resolver,
        trading_calendar=trading_calendar,
        canonical_history=resolved_history,
        build_id=actual_build_id,
        ingested_at=actual_ingested_at,
        source_delivery_id=source_delivery_id,
    )
    if not normalization_result.success:
        if normalization_result.quarantines:
            quarantine_path = (
                paths.quarantine_dir
                / f"earnings_quarantine_{actual_build_id}.parquet"
            )
            write_quarantine_records(quarantine_path, normalization_result.quarantines)
        return EarningsDeliveryResult(
            normalization_result=normalization_result,
            publication_result=None,
            success=False,
        )

    publication_result = publish_earnings_revisions(
        normalization_result.revisions,
        paths=paths,
        source_delivery_id=source_delivery_id,
        build_id=actual_build_id,
    )
    return EarningsDeliveryResult(
        normalization_result=normalization_result,
        publication_result=publication_result,
        success=(
            normalization_result.success
            and publication_result is not None
            and publication_result.success
        ),
    )
