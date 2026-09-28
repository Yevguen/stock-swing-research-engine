"""Validation for canonical earnings revisions and revision histories."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, datetime

from stock_swing_d1.earnings.models import (
    EarningsScheduleRevision,
    EarningsValidationError,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
    TransitionType,
)


def _fail(code: str, message: str) -> None:
    raise EarningsValidationError(code, message)


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


_TRANSITION_LIFECYCLE_COMPATIBILITY: dict[
    TransitionType, frozenset[LifecycleState]
] = {
    TransitionType.CONFIRMED: frozenset({LifecycleState.CONFIRMED}),
    TransitionType.UNCONFIRMED: frozenset({LifecycleState.ESTIMATED}),
    TransitionType.CANCELLED: frozenset({LifecycleState.CANCELLED}),
    TransitionType.OCCURRED: frozenset({LifecycleState.OCCURRED}),
    TransitionType.REINSTATED: frozenset(
        {LifecycleState.ESTIMATED, LifecycleState.CONFIRMED}
    ),
    TransitionType.POSTPONED: frozenset(
        {LifecycleState.ESTIMATED, LifecycleState.CONFIRMED}
    ),
    TransitionType.ADVANCED: frozenset(
        {LifecycleState.ESTIMATED, LifecycleState.CONFIRMED}
    ),
    TransitionType.DATE_CHANGED: frozenset(
        {LifecycleState.ESTIMATED, LifecycleState.CONFIRMED}
    ),
    TransitionType.TIMING_CHANGED: frozenset(
        {LifecycleState.ESTIMATED, LifecycleState.CONFIRMED}
    ),
}


def validate_revision(revision: EarningsScheduleRevision) -> EarningsScheduleRevision:
    """Validate one canonical revision without mutating it."""

    if not isinstance(revision, EarningsScheduleRevision):
        _fail("NON_CANONICAL_REVISION", "expected EarningsScheduleRevision")

    required_text = {
        "schema_version": revision.schema_version,
        "event_revision_id": revision.event_revision_id,
        "event_instance_id": revision.event_instance_id,
        "canonical_asset_id": revision.canonical_asset_id,
        "provider_name": revision.provider_name,
        "provider_entity_id": revision.provider_entity_id,
    }
    for field_name, value in required_text.items():
        if value is None or not isinstance(value, str) or not value.strip():
            code = (
                "MISSING_CANONICAL_ASSET_ID"
                if field_name == "canonical_asset_id"
                else "MISSING_REQUIRED_FIELD"
            )
            _fail(code, f"{field_name} is required")

    optional_text = (
        revision.provider_event_id,
        revision.provider_record_id,
        revision.historical_symbol,
        revision.historical_exchange,
        revision.event_timezone,
        revision.correction_of_revision_id,
    )
    if any(value == "" or (isinstance(value, str) and not value.strip()) for value in optional_text):
        _fail("INVALID_NULL_REPRESENTATION", "empty strings must be represented by None")

    enum_fields = (
        (revision.knowledge_precision, KnowledgePrecision),
        (revision.transition_type, TransitionType),
        (revision.lifecycle_state, LifecycleState),
        (revision.timing_class, TimingClass),
    )
    if any(not isinstance(value, enum_type) for value, enum_type in enum_fields):
        _fail("NON_CANONICAL_VALUE", "strategy-facing enum fields must be canonical")

    sentinel = date(1900, 1, 1)
    for value in (
        revision.knowledge_date,
        revision.scheduled_date,
        revision.actual_event_date,
    ):
        if value == sentinel:
            _fail("INVALID_NULL_REPRESENTATION", "sentinel dates are forbidden")

    timestamp_fields = (
        revision.knowledge_available_at,
        revision.strategy_effective_at,
        revision.scheduled_at,
        revision.actual_event_at,
        revision.ingested_at,
    )
    if any(value is not None and not _is_aware(value) for value in timestamp_fields):
        _fail("TIMEZONE_REQUIRED", "all actual timestamps must be timezone-aware")

    if revision.knowledge_precision is KnowledgePrecision.TIMESTAMP:
        if revision.knowledge_available_at is None:
            _fail("MISSING_KNOWLEDGE_TIMESTAMP", "timestamp precision requires availability time")
        if (
            revision.strategy_effective_at is not None
            and revision.strategy_effective_at < revision.knowledge_available_at
        ):
            _fail(
                "STRATEGY_EFFECTIVE_BEFORE_KNOWLEDGE",
                "strategy_effective_at cannot precede knowledge_available_at",
            )
    elif revision.knowledge_precision is KnowledgePrecision.DATE_ONLY:
        if revision.knowledge_date is None:
            _fail("MISSING_KNOWLEDGE_DATE", "date-only precision requires knowledge_date")
        if revision.knowledge_available_at is not None:
            _fail(
                "INVALID_KNOWLEDGE_PRECISION",
                "date-only knowledge cannot claim an exact provider timestamp",
            )
        if (
            revision.strategy_effective_at is not None
            and revision.strategy_effective_at.date() <= revision.knowledge_date
        ):
            _fail(
                "DATE_ONLY_EFFECTIVE_TOO_EARLY",
                "date-only knowledge must become effective after knowledge_date",
            )

    if revision.timing_class is TimingClass.EXACT_TIME:
        if revision.scheduled_at is None or not revision.event_timezone:
            _fail("TIMEZONE_REQUIRED", "exact event timing requires a zoned timestamp")

    if revision.lifecycle_state in {LifecycleState.ESTIMATED, LifecycleState.CONFIRMED}:
        if revision.scheduled_date is None:
            _fail("MISSING_SCHEDULED_DATE", "active schedules require scheduled_date")

    allowed_lifecycle_states = _TRANSITION_LIFECYCLE_COMPATIBILITY.get(
        revision.transition_type
    )
    if (
        allowed_lifecycle_states is not None
        and revision.lifecycle_state not in allowed_lifecycle_states
    ):
        _fail(
            "INVALID_TRANSITION_STATE_COMBINATION",
            f"{revision.transition_type.value} is incompatible with "
            f"{revision.lifecycle_state.value}",
        )

    correction_transition = revision.transition_type is TransitionType.PROVIDER_CORRECTION
    if correction_transition != revision.is_provider_correction:
        _fail(
            "INVALID_PROVIDER_CORRECTION",
            "provider corrections require both the correction flag and transition type",
        )
    if correction_transition and not revision.correction_of_revision_id:
        _fail(
            "INVALID_PROVIDER_CORRECTION",
            "provider correction must identify the corrected revision",
        )

    return revision


def _ordered(revisions: Sequence[EarningsScheduleRevision]) -> list[EarningsScheduleRevision]:
    for revision in revisions:
        if revision.strategy_effective_at is None:
            _fail(
                "REVISION_EFFECTIVE_TIME_REQUIRED",
                "derive strategy_effective_at before replay",
            )
    return sorted(
        revisions,
        key=lambda revision: (
            revision.strategy_effective_at,
            revision.provider_sequence if revision.provider_sequence is not None else -1,
            revision.event_revision_id or "",
        ),
    )


def validate_revision_history(
    revisions: Iterable[EarningsScheduleRevision],
) -> tuple[EarningsScheduleRevision, ...]:
    """Validate immutable history and return it in deterministic replay order."""

    history = tuple(revisions)
    for revision in history:
        validate_revision(revision)

    seen_by_id: dict[str, EarningsScheduleRevision] = {}
    for revision in history:
        assert revision.event_revision_id is not None
        prior = seen_by_id.get(revision.event_revision_id)
        if prior is not None and prior != revision:
            _fail("IMMUTABLE_REVISION", "a revision ID cannot be rewritten")
        if prior is not None:
            _fail("DUPLICATE_REVISION", "a revision may only be appended once")
        seen_by_id[revision.event_revision_id] = revision

    scopes = {
        (revision.canonical_asset_id, revision.event_instance_id) for revision in history
    }
    if len(scopes) > 1:
        _fail("HISTORY_SCOPE_MISMATCH", "replay requires one asset and event instance")

    same_time: dict[datetime, list[EarningsScheduleRevision]] = defaultdict(list)
    for revision in history:
        assert revision.strategy_effective_at is not None
        same_time[revision.strategy_effective_at].append(revision)
    for group in same_time.values():
        if len(group) <= 1:
            continue
        sequences = [revision.provider_sequence for revision in group]
        if any(sequence is None for sequence in sequences) or len(set(sequences)) != len(sequences):
            _fail(
                "AMBIGUOUS_REVISION_ORDER",
                "same-timestamp revisions require unique provider_sequence values",
            )

    ordered = _ordered(history)
    previous: EarningsScheduleRevision | None = None
    prior_ids: set[str] = set()
    for revision in ordered:
        if revision.is_provider_correction:
            if revision.correction_of_revision_id not in prior_ids:
                _fail(
                    "INVALID_PROVIDER_CORRECTION",
                    "correction target must be an earlier revision",
                )
        if previous is not None:
            if previous.lifecycle_state is LifecycleState.OCCURRED:
                allowed_occurrence_correction = (
                    revision.is_provider_correction
                    and revision.lifecycle_state is LifecycleState.OCCURRED
                )
                if not allowed_occurrence_correction:
                    _fail(
                        "INVALID_STATE_TRANSITION",
                        "history after OCCURRED requires a new event instance",
                    )
            if (
                previous.lifecycle_state is LifecycleState.CANCELLED
                and revision.lifecycle_state
                in {LifecycleState.ESTIMATED, LifecycleState.CONFIRMED}
                and revision.transition_type is not TransitionType.REINSTATED
            ):
                _fail(
                    "INVALID_STATE_TRANSITION",
                    "reactivating a cancelled event requires REINSTATED",
                )

            if revision.transition_type is TransitionType.POSTPONED:
                if (
                    previous.scheduled_date is None
                    or revision.scheduled_date is None
                    or revision.scheduled_date <= previous.scheduled_date
                ):
                    _fail("INVALID_STATE_TRANSITION", "POSTPONED must move the date later")
            if revision.transition_type is TransitionType.ADVANCED:
                if (
                    previous.scheduled_date is None
                    or revision.scheduled_date is None
                    or revision.scheduled_date >= previous.scheduled_date
                ):
                    _fail("INVALID_STATE_TRANSITION", "ADVANCED must move the date earlier")

        if revision.transition_type is TransitionType.CANCELLED and revision.lifecycle_state is not LifecycleState.CANCELLED:
            _fail("INVALID_STATE_TRANSITION", "CANCELLED transition must cancel lifecycle")
        if revision.transition_type is TransitionType.OCCURRED and revision.lifecycle_state is not LifecycleState.OCCURRED:
            _fail("INVALID_STATE_TRANSITION", "OCCURRED transition must close lifecycle")
        if revision.lifecycle_state is LifecycleState.INVALID:
            _fail("INVALID_STATE_TRANSITION", "INVALID cannot be a replayable state")

        assert revision.event_revision_id is not None
        prior_ids.add(revision.event_revision_id)
        previous = revision

    return tuple(ordered)


def validate_security_mapping(
    *,
    canonical_asset_id: str | None,
    historical_symbol: str | None,
    historical_exchange: str | None,
    candidate_asset_ids: Sequence[str] = (),
    mapping_evidence: str | None = None,
) -> str:
    """Require stable identity rather than an ambiguous ticker-only mapping."""

    if canonical_asset_id is not None and canonical_asset_id.strip():
        return canonical_asset_id.strip()
    candidates = set(candidate_asset_ids)
    if (
        historical_symbol
        and len(candidates) == 1
        and isinstance(mapping_evidence, str)
        and mapping_evidence.strip()
    ):
        return candidate_asset_ids[0]
    if len(candidates) == 1:
        _fail(
            "INSUFFICIENT_SECURITY_MAPPING_EVIDENCE",
            "one ticker candidate is not proof of stable historical identity",
        )
    _fail(
        "AMBIGUOUS_SECURITY_MAPPING",
        "ticker/exchange text alone is not a stable canonical identity",
    )
