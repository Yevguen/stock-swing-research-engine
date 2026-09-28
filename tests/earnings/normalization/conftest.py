"""Synthetic-only fixtures for the frozen normalization contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from stock_swing_d1.earnings import (
    EarningsScheduleRevision,
    EarningsValidationError,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
    TransitionType,
    validate_security_mapping,
)
from stock_swing_d1.earnings.normalization import (
    EarningsAdapterBatch,
    EarningsKnowledgeCutoff,
    EarningsNormalizationFailure,
)
from stock_swing_d1.earnings.normalization.identity import (
    is_revision_admissible,
    select_latest_admissible_revision,
)
from stock_swing_d1.earnings.persistence import QuarantineStage


PROVIDER = "SYNTHETIC_PROVIDER"
POSITIVE_EVIDENCE = (
    ("stable_entity_link", "ENTITY-1"),
    ("norgate_asset_id", "1001"),
)


class SyntheticProviderError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__("synthetic provider parsing failed")


class SyntheticIdentityResolver:
    """Evidence-gated resolver; ticker text never establishes identity."""

    _verified = {
        (PROVIDER, "ENTITY-1"): "NORGATE:1001",
        (PROVIDER, "ENTITY-2"): "NORGATE:2002",
    }

    def resolve_canonical_asset_id(
        self,
        *,
        provider_name: str,
        provider_entity_id: str,
        historical_symbol: str | None,
        historical_exchange: str | None,
        mapping_evidence: tuple[tuple[str, str], ...],
    ) -> str:
        evidence = dict(mapping_evidence)
        candidates = tuple(
            value
            for key, value in mapping_evidence
            if key == "candidate_asset_id"
        )
        verified = self._verified.get((provider_name, provider_entity_id))
        has_positive_evidence = (
            verified is not None
            and evidence.get("stable_entity_link") == provider_entity_id
            and evidence.get("norgate_asset_id") == verified.removeprefix("NORGATE:")
        )
        canonical_asset_id = verified if has_positive_evidence else None
        return validate_security_mapping(
            canonical_asset_id=canonical_asset_id,
            historical_symbol=historical_symbol,
            historical_exchange=historical_exchange,
            candidate_asset_ids=candidates,
            mapping_evidence=("synthetic-positive-evidence" if has_positive_evidence else None),
        )


class SyntheticCanonicalHistoryReader:
    def __init__(self, revisions: tuple[EarningsScheduleRevision, ...]) -> None:
        self.revisions = revisions

    def get_latest_revision(
        self,
        *,
        provider_name: str,
        canonical_asset_id: str,
        event_instance_id: str,
        knowledge_cutoff: EarningsKnowledgeCutoff,
    ) -> EarningsScheduleRevision | None:
        return select_latest_admissible_revision(
            self.revisions,
            provider_name=provider_name,
            canonical_asset_id=canonical_asset_id,
            event_instance_id=event_instance_id,
            knowledge_cutoff=knowledge_cutoff,
        )


class SyntheticNormalizationCalendar:
    sessions = (
        date(2026, 9, 4),
        date(2026, 9, 8),
        date(2026, 9, 9),
        date(2026, 10, 1),
        date(2026, 10, 2),
        date(2026, 10, 5),
        date(2026, 10, 6),
        date(2026, 10, 7),
        date(2026, 10, 8),
        date(2026, 10, 9),
        date(2026, 10, 12),
        date(2026, 10, 13),
        date(2026, 10, 14),
        date(2026, 10, 15),
        date(2026, 10, 16),
        date(2026, 10, 19),
        date(2026, 10, 20),
        date(2026, 10, 21),
        date(2026, 10, 22),
        date(2026, 10, 23),
    )

    def next_session(self, session: date) -> date:
        return next(candidate for candidate in self.sessions if candidate > session)

    def decision_time(self, session: date) -> datetime:
        if session not in self.sessions:
            raise ValueError("not a synthetic trading session")
        return datetime.combine(
            session,
            time(9, 30),
            tzinfo=ZoneInfo("America/New_York"),
        )


def _revision_id(entity: str, cycle: str, revision_key: str) -> str:
    payload = f"{PROVIDER}|{entity}|{cycle}|{revision_key}".encode("utf-8")
    return f"SYNREV-{hashlib.sha256(payload).hexdigest()[:20]}"


def _event_id(entity: str, cycle: str) -> str:
    payload = f"{PROVIDER}|{entity}|{cycle}".encode("utf-8")
    return f"SYNEVT-{hashlib.sha256(payload).hexdigest()[:20]}"


def _failure(
    raw: dict[str, Any] | None,
    *,
    stage: QuarantineStage,
    code: str,
    message: str,
    canonical_asset_id: str | None = None,
    event_instance_id: str | None = None,
    event_revision_id: str | None = None,
) -> EarningsNormalizationFailure:
    return EarningsNormalizationFailure(
        stage=stage,
        error_code=code,
        error_message=message,
        provider_name=PROVIDER,
        provider_entity_id=(raw.get("provider_entity_id") if raw else None),
        canonical_asset_id=canonical_asset_id,
        event_instance_id=event_instance_id,
        event_revision_id=event_revision_id,
        source_record_locator=(raw.get("source_record_locator") if raw else None),
        source_record_hash=(raw.get("source_record_hash") if raw else None),
        details_json='{"diagnostic":"synthetic"}',
    )


class SyntheticProviderDeliveryAdapter:
    provider_name = PROVIDER

    _lifecycle = {
        "EST": LifecycleState.ESTIMATED,
        "CONF": LifecycleState.CONFIRMED,
        "CANC": LifecycleState.CANCELLED,
        "DONE": LifecycleState.OCCURRED,
        "UNKNOWN": LifecycleState.UNKNOWN,
    }
    _timing = {
        "UNK": TimingClass.UNKNOWN,
        "PRE": TimingClass.BMO,
        "OPEN": TimingClass.DURING_MARKET,
        "POST": TimingClass.AMC,
        "CLOCK": TimingClass.EXACT_TIME,
    }
    _precision = {
        "TS": KnowledgePrecision.TIMESTAMP,
        "DATE": KnowledgePrecision.DATE_ONLY,
    }
    _transition = {
        "CREATE": TransitionType.CREATED,
        "OCCUR": TransitionType.OCCURRED,
        "CANCEL": TransitionType.CANCELLED,
        "REINSTATE": TransitionType.REINSTATED,
        "POSTPONE": TransitionType.POSTPONED,
        "ADVANCE": TransitionType.ADVANCED,
        "CONFIRM": TransitionType.CONFIRMED,
        "UNCONFIRM": TransitionType.UNCONFIRMED,
        "TIMING": TransitionType.TIMING_CHANGED,
        "CORRECT": TransitionType.PROVIDER_CORRECTION,
        "OTHER": TransitionType.OTHER,
    }

    @staticmethod
    def _sort_key(raw: dict[str, Any]) -> tuple[Any, ...]:
        available = raw.get("knowledge_available_at")
        available_key = (
            available.astimezone(timezone.utc).isoformat()
            if isinstance(available, datetime)
            and available.tzinfo is not None
            and available.utcoffset() is not None
            else ""
        )
        sequence = raw.get("provider_sequence")
        return (
            raw.get("knowledge_date", date.min),
            raw.get("knowledge_precision_code", ""),
            available_key,
            sequence is None,
            sequence if sequence is not None else 0,
            raw.get("revision_key", ""),
        )

    @staticmethod
    def _candidate_sort_key(revision: EarningsScheduleRevision) -> tuple[Any, ...]:
        available = revision.knowledge_available_at
        return (
            revision.knowledge_date,
            available.astimezone(timezone.utc) if available is not None else datetime.min.replace(tzinfo=timezone.utc),
            revision.provider_sequence is None,
            revision.provider_sequence or 0,
            revision.event_revision_id,
        )

    @staticmethod
    def _transition_for(
        prior: EarningsScheduleRevision | None,
        *,
        lifecycle: LifecycleState,
        scheduled_date: date | None,
        timing_class: TimingClass,
        scheduled_at: datetime | None,
        is_correction: bool,
    ) -> TransitionType:
        if is_correction:
            return TransitionType.PROVIDER_CORRECTION
        if lifecycle is LifecycleState.OCCURRED:
            return TransitionType.OCCURRED
        if lifecycle is LifecycleState.CANCELLED:
            return TransitionType.CANCELLED
        if prior is None:
            return TransitionType.CREATED
        if prior.lifecycle_state is LifecycleState.CANCELLED and lifecycle in {
            LifecycleState.ESTIMATED,
            LifecycleState.CONFIRMED,
        }:
            return TransitionType.REINSTATED
        if prior.scheduled_date is not None and scheduled_date is not None:
            if scheduled_date > prior.scheduled_date:
                return TransitionType.POSTPONED
            if scheduled_date < prior.scheduled_date:
                return TransitionType.ADVANCED
        if prior.lifecycle_state is not lifecycle:
            if lifecycle is LifecycleState.CONFIRMED:
                return TransitionType.CONFIRMED
            if lifecycle is LifecycleState.ESTIMATED:
                return TransitionType.UNCONFIRMED
        if prior.timing_class is not timing_class or prior.scheduled_at != scheduled_at:
            return TransitionType.TIMING_CHANGED
        return TransitionType.OTHER

    @staticmethod
    def _state_value(
        raw: dict[str, Any],
        key: str,
        prior: EarningsScheduleRevision | None,
        prior_attribute: str,
    ) -> Any:
        if key in raw:
            return raw[key]
        if prior is None:
            raise SyntheticProviderError("INCOMPLETE_SNAPSHOT")
        return getattr(prior, prior_attribute)

    def normalize_delivery(
        self,
        raw_records: tuple[Any, ...],
        *,
        identity_resolver: SyntheticIdentityResolver,
        canonical_history: SyntheticCanonicalHistoryReader | None = None,
    ) -> EarningsAdapterBatch:
        if any(isinstance(raw, dict) and raw.get("raise_adapter_error") for raw in raw_records):
            raise SyntheticProviderError("SYNTHETIC_PARSE_ERROR")

        valid_raw = [raw for raw in raw_records if isinstance(raw, dict)]
        failures: list[EarningsNormalizationFailure] = [
            _failure(
                None,
                stage=QuarantineStage.NORMALIZATION,
                code="INVALID_SYNTHETIC_RECORD",
                message="synthetic record must be a mapping",
            )
            for raw in raw_records
            if not isinstance(raw, dict)
        ]
        revisions: list[EarningsScheduleRevision] = []

        for raw in sorted(valid_raw, key=self._sort_key):
            entity = raw.get("provider_entity_id")
            cycle = raw.get("cycle_id")
            revision_key = raw.get("revision_key")
            if not all(isinstance(value, str) and value for value in (entity, cycle, revision_key)):
                failures.append(
                    _failure(
                        raw,
                        stage=QuarantineStage.NORMALIZATION,
                        code="MISSING_SYNTHETIC_IDENTITY",
                        message="synthetic provider identifiers are required",
                    )
                )
                continue
            event_instance_id = _event_id(entity, cycle)
            event_revision_id = _revision_id(entity, cycle, revision_key)
            try:
                canonical_asset_id = identity_resolver.resolve_canonical_asset_id(
                    provider_name=self.provider_name,
                    provider_entity_id=entity,
                    historical_symbol=raw.get("historical_symbol"),
                    historical_exchange=raw.get("historical_exchange"),
                    mapping_evidence=tuple(raw.get("mapping_evidence", ())),
                )
            except EarningsValidationError as exc:
                failures.append(
                    _failure(
                        raw,
                        stage=QuarantineStage.IDENTITY_MAPPING,
                        code=exc.code,
                        message=str(exc),
                        event_instance_id=event_instance_id,
                        event_revision_id=event_revision_id,
                    )
                )
                continue

            try:
                precision = self._precision[raw["knowledge_precision_code"]]
                cutoff = EarningsKnowledgeCutoff(
                    knowledge_precision=precision,
                    knowledge_date=raw["knowledge_date"],
                    knowledge_available_at=raw.get("knowledge_available_at"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(
                    _failure(
                        raw,
                        stage=QuarantineStage.NORMALIZATION,
                        code="UNMAPPED_PROVIDER_VALUE",
                        message=f"synthetic knowledge value is invalid: {type(exc).__name__}",
                        canonical_asset_id=canonical_asset_id,
                        event_instance_id=event_instance_id,
                        event_revision_id=event_revision_id,
                    )
                )
                continue

            prior: EarningsScheduleRevision | None = None
            prior_candidates = [
                revision
                for revision in revisions
                if revision.canonical_asset_id == canonical_asset_id
                and revision.event_instance_id == event_instance_id
                and is_revision_admissible(revision, cutoff)
            ]
            if prior_candidates:
                prior = max(prior_candidates, key=self._candidate_sort_key)
            elif canonical_history is not None:
                prior = canonical_history.get_latest_revision(
                    provider_name=self.provider_name,
                    canonical_asset_id=canonical_asset_id,
                    event_instance_id=event_instance_id,
                    knowledge_cutoff=cutoff,
                )

            if raw.get("is_delta") and prior is None:
                failures.append(
                    _failure(
                        raw,
                        stage=QuarantineStage.NORMALIZATION,
                        code="INCOMPLETE_DELTA_CONTEXT",
                        message="no admissible same-provider antecedent exists",
                        canonical_asset_id=canonical_asset_id,
                        event_instance_id=event_instance_id,
                        event_revision_id=event_revision_id,
                    )
                )
                continue

            try:
                lifecycle_raw = self._state_value(raw, "lifecycle_code", prior, "lifecycle_state")
                lifecycle = (
                    lifecycle_raw
                    if isinstance(lifecycle_raw, LifecycleState)
                    else self._lifecycle[lifecycle_raw]
                )
                timing_raw = self._state_value(raw, "timing_code", prior, "timing_class")
                timing_class = (
                    timing_raw if isinstance(timing_raw, TimingClass) else self._timing[timing_raw]
                )
                scheduled_date = self._state_value(raw, "scheduled_date", prior, "scheduled_date")
                scheduled_at = self._state_value(raw, "scheduled_at", prior, "scheduled_at")
                event_timezone = self._state_value(raw, "event_timezone", prior, "event_timezone")
                actual_event_date = self._state_value(raw, "actual_event_date", prior, "actual_event_date")
                actual_event_at = self._state_value(raw, "actual_event_at", prior, "actual_event_at")
                explicit_transition = raw.get("transition_code")
                is_correction = raw.get("correction_of_revision_key") is not None
                transition = (
                    self._transition[explicit_transition]
                    if explicit_transition is not None
                    else self._transition_for(
                        prior,
                        lifecycle=lifecycle,
                        scheduled_date=scheduled_date,
                        timing_class=timing_class,
                        scheduled_at=scheduled_at,
                        is_correction=is_correction,
                    )
                )
            except (KeyError, SyntheticProviderError) as exc:
                code = exc.code if isinstance(exc, SyntheticProviderError) else "UNMAPPED_PROVIDER_VALUE"
                failures.append(
                    _failure(
                        raw,
                        stage=QuarantineStage.NORMALIZATION,
                        code=code,
                        message="synthetic provider value is unmapped or incomplete",
                        canonical_asset_id=canonical_asset_id,
                        event_instance_id=event_instance_id,
                        event_revision_id=event_revision_id,
                    )
                )
                continue

            correction_key = raw.get("correction_of_revision_key")
            correction_id = (
                _revision_id(entity, cycle, correction_key)
                if correction_key is not None
                else None
            )
            revisions.append(
                EarningsScheduleRevision(
                    schema_version="0.1",
                    event_revision_id=event_revision_id,
                    event_instance_id=event_instance_id,
                    canonical_asset_id=canonical_asset_id,
                    provider_name=self.provider_name,
                    provider_entity_id=entity,
                    provider_event_id=raw.get("provider_event_id"),
                    provider_record_id=raw.get("provider_record_id"),
                    historical_symbol=(
                        raw.get("historical_symbol")
                        if "historical_symbol" in raw
                        else (prior.historical_symbol if prior else None)
                    ),
                    historical_exchange=(
                        raw.get("historical_exchange")
                        if "historical_exchange" in raw
                        else (prior.historical_exchange if prior else None)
                    ),
                    knowledge_date=cutoff.knowledge_date,
                    knowledge_available_at=cutoff.knowledge_available_at,
                    knowledge_precision=cutoff.knowledge_precision,
                    strategy_effective_at=None,
                    provider_sequence=raw.get("provider_sequence"),
                    transition_type=transition,
                    lifecycle_state=lifecycle,
                    scheduled_date=scheduled_date,
                    timing_class=timing_class,
                    scheduled_at=scheduled_at,
                    event_timezone=event_timezone,
                    actual_event_date=actual_event_date,
                    actual_event_at=actual_event_at,
                    is_provider_correction=is_correction,
                    correction_of_revision_id=correction_id,
                    ingested_at=None,
                )
            )
        return EarningsAdapterBatch(revisions=tuple(revisions), failures=tuple(failures))


def make_raw_record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "provider_entity_id": "ENTITY-1",
        "cycle_id": "2026-Q3",
        "revision_key": "R1",
        "provider_event_id": "SYNTHETIC-EVENT-Q3",
        "provider_record_id": "SYNTHETIC-RECORD-R1",
        "historical_symbol": "OLDX",
        "historical_exchange": "XNYS",
        "mapping_evidence": POSITIVE_EVIDENCE,
        "knowledge_precision_code": "TS",
        "knowledge_date": date(2026, 9, 1),
        "knowledge_available_at": datetime(2026, 9, 1, 14, tzinfo=timezone.utc),
        "provider_sequence": 1,
        "lifecycle_code": "EST",
        "scheduled_date": date(2026, 10, 14),
        "timing_code": "UNK",
        "scheduled_at": None,
        "event_timezone": None,
        "actual_event_date": None,
        "actual_event_at": None,
        "is_delta": False,
        "source_record_locator": "delivery://synthetic/R1",
        "source_record_hash": "1" * 64,
    }
    values.update(overrides)
    return values


def make_delta_record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "provider_entity_id": "ENTITY-1",
        "cycle_id": "2026-Q3",
        "revision_key": "R2",
        "provider_event_id": "SYNTHETIC-EVENT-Q3",
        "provider_record_id": "SYNTHETIC-RECORD-R2",
        "historical_symbol": "OLDX",
        "historical_exchange": "XNYS",
        "mapping_evidence": POSITIVE_EVIDENCE,
        "knowledge_precision_code": "TS",
        "knowledge_date": date(2026, 9, 2),
        "knowledge_available_at": datetime(2026, 9, 2, 14, tzinfo=timezone.utc),
        "provider_sequence": 2,
        "is_delta": True,
        "source_record_locator": "delivery://synthetic/R2",
        "source_record_hash": "2" * 64,
    }
    values.update(overrides)
    return values


def make_canonical_revision(**overrides: Any) -> EarningsScheduleRevision:
    candidate = SyntheticProviderDeliveryAdapter().normalize_delivery(
        (make_raw_record(),),
        identity_resolver=SyntheticIdentityResolver(),
    ).revisions[0]
    values = {
        field_name: getattr(candidate, field_name)
        for field_name in candidate.__dataclass_fields__
    }
    values.update(
        strategy_effective_at=candidate.knowledge_available_at,
        ingested_at=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
    )
    values.update(overrides)
    return EarningsScheduleRevision(**values)


@pytest.fixture
def identity_resolver() -> SyntheticIdentityResolver:
    return SyntheticIdentityResolver()


@pytest.fixture
def delivery_adapter() -> SyntheticProviderDeliveryAdapter:
    return SyntheticProviderDeliveryAdapter()


@pytest.fixture
def normalization_calendar() -> SyntheticNormalizationCalendar:
    return SyntheticNormalizationCalendar()


@pytest.fixture
def raw_record():
    return make_raw_record


@pytest.fixture
def delta_record():
    return make_delta_record


@pytest.fixture
def canonical_revision():
    return make_canonical_revision


@pytest.fixture
def history_reader():
    def factory(*revisions: EarningsScheduleRevision) -> SyntheticCanonicalHistoryReader:
        return SyntheticCanonicalHistoryReader(tuple(revisions))

    return factory
