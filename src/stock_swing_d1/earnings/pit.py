"""Point-in-time reconstruction of canonical earnings schedules."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

from stock_swing_d1.earnings.models import (
    EarningsScheduleRevision,
    EarningsScheduleStatus,
    EarningsStateAsOf,
    EarningsValidationError,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
)
from stock_swing_d1.earnings.validation import (
    validate_revision,
    validate_revision_history,
)


class EarningsKnowledgeCalendar(Protocol):
    """Calendar boundary used to convert date-only knowledge into PIT time."""

    def next_session(self, session: object) -> object:
        ...

    def decision_time(self, session: object) -> datetime:
        ...


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise EarningsValidationError("TIMEZONE_REQUIRED", f"{field_name} must be aware")


def derive_strategy_effective_at(
    revision: EarningsScheduleRevision,
    trading_calendar: EarningsKnowledgeCalendar,
) -> datetime:
    """Derive when a revision can first affect a historical strategy decision."""

    if revision.knowledge_precision is KnowledgePrecision.TIMESTAMP:
        if revision.knowledge_available_at is None:
            raise EarningsValidationError(
                "MISSING_KNOWLEDGE_TIMESTAMP",
                "timestamp precision requires knowledge_available_at",
            )
        _require_aware(revision.knowledge_available_at, "knowledge_available_at")
        return revision.knowledge_available_at

    if revision.knowledge_precision is KnowledgePrecision.DATE_ONLY:
        if revision.knowledge_date is None:
            raise EarningsValidationError(
                "MISSING_KNOWLEDGE_DATE",
                "date-only precision requires knowledge_date",
            )
        next_session = trading_calendar.next_session(revision.knowledge_date)
        effective_at = trading_calendar.decision_time(next_session)
        _require_aware(effective_at, "calendar decision_time")
        if effective_at.time().replace(tzinfo=None).isoformat() == "00:00:00":
            raise EarningsValidationError(
                "FABRICATED_MIDNIGHT",
                "date-only knowledge cannot become available at fabricated midnight",
            )
        return effective_at

    raise EarningsValidationError(
        "NON_CANONICAL_VALUE", "knowledge_precision must be canonical"
    )


def apply_revision(
    previous_state: EarningsStateAsOf | None,
    revision: EarningsScheduleRevision,
) -> EarningsStateAsOf:
    """Apply one complete snapshot without merging fields from prior state.

    ``previous_state`` is retained for state-machine API symmetry.  Canonical
    revisions already contain the complete resulting state, so its fields are
    intentionally not used as delta fallbacks.
    """

    validate_revision(revision)
    if revision.strategy_effective_at is None:
        raise EarningsValidationError(
            "REVISION_EFFECTIVE_TIME_REQUIRED",
            "derive strategy_effective_at before replay",
        )
    active = revision.lifecycle_state in {
        LifecycleState.ESTIMATED,
        LifecycleState.CONFIRMED,
    }
    return EarningsStateAsOf(
        as_of=revision.strategy_effective_at,
        canonical_asset_id=revision.canonical_asset_id,
        event_instance_id=revision.event_instance_id,
        earnings_schedule_known=active and revision.scheduled_date is not None,
        lifecycle_state=revision.lifecycle_state,
        scheduled_date=revision.scheduled_date if active else None,
        timing_class=revision.timing_class if active else TimingClass.UNKNOWN,
        scheduled_at=revision.scheduled_at if active else None,
        knowledge_effective_at=revision.strategy_effective_at,
        transition_type=revision.transition_type,
        event_revision_id=revision.event_revision_id,
        historical_symbol=revision.historical_symbol,
        historical_exchange=revision.historical_exchange,
        actual_event_date=revision.actual_event_date,
        actual_event_at=revision.actual_event_at,
        is_provider_correction=revision.is_provider_correction,
        schedule_status=(
            EarningsScheduleStatus.KNOWN_EVENT
            if revision.lifecycle_state is not LifecycleState.UNKNOWN
            else EarningsScheduleStatus.UNKNOWN
        ),
    )


def reconstruct_earnings_state(
    revisions: Iterable[EarningsScheduleRevision],
    as_of: datetime,
) -> EarningsStateAsOf:
    """Validate and replay only revisions strategy-effective by ``as_of``.

    Full-dataset validation remains the responsibility of
    :func:`validate_revision_history`.  A historical query must not depend on
    records that were not yet visible at its decision timestamp.
    """

    _require_aware(as_of, "as_of")
    history = tuple(revisions)
    qualifying: list[EarningsScheduleRevision] = []
    for revision in history:
        effective_at = revision.strategy_effective_at
        if effective_at is None:
            raise EarningsValidationError(
                "REVISION_EFFECTIVE_TIME_REQUIRED",
                "derive strategy_effective_at before replay",
            )
        _require_aware(effective_at, "strategy_effective_at")
        if effective_at <= as_of:
            qualifying.append(revision)

    ordered = validate_revision_history(qualifying)
    state: EarningsStateAsOf | None = None
    for revision in ordered:
        assert revision.strategy_effective_at is not None
        state = apply_revision(state, revision)

    if state is None:
        known_asset_ids = {
            revision.canonical_asset_id
            for revision in history
            if isinstance(revision.canonical_asset_id, str)
            and revision.canonical_asset_id.strip()
        }
        return EarningsStateAsOf(
            as_of=as_of,
            canonical_asset_id=(
                next(iter(known_asset_ids)) if len(known_asset_ids) == 1 else None
            ),
            event_instance_id=None,
            earnings_schedule_known=False,
            lifecycle_state=LifecycleState.UNKNOWN,
            scheduled_date=None,
            timing_class=TimingClass.UNKNOWN,
            scheduled_at=None,
            knowledge_effective_at=None,
            schedule_status=EarningsScheduleStatus.UNKNOWN,
        )

    return EarningsStateAsOf(
        **{
            field_name: getattr(state, field_name)
            for field_name in state.__dataclass_fields__
            if field_name != "as_of"
        },
        as_of=as_of,
    )
