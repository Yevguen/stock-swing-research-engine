from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timezone
from typing import Any

import pytest

from stock_swing_d1.earnings import (
    EarningsProviderAdapter,
    EarningsScheduleRevision,
    EarningsValidationError,
    KnowledgePrecision,
    LifecycleState,
    TimingClass,
    TransitionType,
    reconstruct_earnings_state,
    validate_revision,
    validate_revision_history,
    validate_security_mapping,
)


def test_contract_e01_canonical_asset_id_is_required(make_revision) -> None:
    revision = make_revision(canonical_asset_id=None)

    with pytest.raises(EarningsValidationError) as error:
        validate_revision(revision)

    assert error.value.code == "MISSING_CANONICAL_ASSET_ID"


def test_contract_e02_ticker_only_mapping_is_not_sufficient() -> None:
    with pytest.raises(EarningsValidationError) as error:
        validate_security_mapping(
            canonical_asset_id=None,
            historical_symbol="TEST",
            historical_exchange="NYSE",
            candidate_asset_ids=("NORGATE:1001", "NORGATE:2002"),
        )

    assert error.value.code == "AMBIGUOUS_SECURITY_MAPPING"


def test_contract_e03_ticker_change_preserves_canonical_identity(
    make_revision, utc_dt, canonical_asset_id
) -> None:
    old_ticker = make_revision(
        event_revision_id="R1",
        historical_symbol="OLD",
        canonical_asset_id=canonical_asset_id,
    )
    new_ticker = make_revision(
        event_revision_id="R2",
        historical_symbol="NEW",
        canonical_asset_id=canonical_asset_id,
        knowledge_available_at=utc_dt(2026, 9, 15),
        strategy_effective_at=utc_dt(2026, 9, 15),
        transition_type=TransitionType.OTHER,
    )

    state = reconstruct_earnings_state((old_ticker, new_ticker), utc_dt(2026, 9, 20))

    assert state.canonical_asset_id == canonical_asset_id
    assert state.historical_symbol == "NEW"


def test_contract_e04_current_ticker_is_not_backfilled_historically(
    make_revision, utc_dt
) -> None:
    old_ticker = make_revision(event_revision_id="R1", historical_symbol="OLD")
    future_ticker = make_revision(
        event_revision_id="R2",
        historical_symbol="NEW",
        knowledge_available_at=utc_dt(2026, 9, 15),
        strategy_effective_at=utc_dt(2026, 9, 15),
        transition_type=TransitionType.OTHER,
    )

    historical_state = reconstruct_earnings_state(
        (old_ticker, future_ticker), utc_dt(2026, 9, 10)
    )

    assert historical_state.historical_symbol == "OLD"


def test_contract_e05_exact_timestamp_requires_timezone(make_revision) -> None:
    revision = make_revision(
        timing_class=TimingClass.EXACT_TIME,
        scheduled_at=datetime(2026, 10, 29, 8, 0),
    )

    with pytest.raises(EarningsValidationError) as error:
        validate_revision(revision)

    assert error.value.code == "TIMEZONE_REQUIRED"


def test_contract_e06_null_date_differs_from_unknown_timing(make_revision) -> None:
    no_schedule = make_revision(
        lifecycle_state=LifecycleState.UNKNOWN,
        scheduled_date=None,
        timing_class=TimingClass.UNKNOWN,
    )
    date_known_timing_unknown = make_revision(
        event_revision_id="R2",
        scheduled_date=date(2026, 10, 29),
        timing_class=TimingClass.UNKNOWN,
    )

    validate_revision(no_schedule)
    validate_revision(date_known_timing_unknown)

    assert no_schedule.scheduled_date is None
    assert date_known_timing_unknown.scheduled_date == date(2026, 10, 29)
    assert date_known_timing_unknown.timing_class is TimingClass.UNKNOWN


def test_contract_e07_revisions_are_append_only_and_immutable(
    make_revision, utc_dt
) -> None:
    original = make_revision(
        event_revision_id="R1", scheduled_date=date(2026, 10, 27)
    )
    appended = make_revision(
        event_revision_id="R2",
        knowledge_available_at=utc_dt(2026, 9, 15),
        strategy_effective_at=utc_dt(2026, 9, 15),
        transition_type=TransitionType.POSTPONED,
        lifecycle_state=LifecycleState.CONFIRMED,
        scheduled_date=date(2026, 10, 29),
    )

    validate_revision_history((original, appended))
    assert original.scheduled_date == date(2026, 10, 27)
    assert appended.scheduled_date == date(2026, 10, 29)
    with pytest.raises(FrozenInstanceError):
        original.scheduled_date = date(2026, 10, 29)

    rewritten_original = replace(
        original,
        scheduled_date=date(2026, 10, 29),
        strategy_effective_at=utc_dt(2026, 9, 20),
    )
    with pytest.raises(EarningsValidationError) as error:
        validate_revision_history((original, rewritten_original))
    assert error.value.code == "IMMUTABLE_REVISION"


def test_contract_e08_provider_specific_values_do_not_reach_strategy(
    make_revision, utc_dt
) -> None:
    class SyntheticAdapter:
        def normalize_record(self, raw_record: Any) -> EarningsScheduleRevision:
            lifecycle = {
                "vendor-estimate": LifecycleState.ESTIMATED,
                "vendor-confirmed": LifecycleState.CONFIRMED,
            }[raw_record["vendor_status"]]
            timing = {"vendor-after-bell": TimingClass.AMC}[
                raw_record["vendor_timing"]
            ]
            return make_revision(
                lifecycle_state=lifecycle,
                timing_class=timing,
                transition_type=TransitionType.CONFIRMED,
            )

    adapter: EarningsProviderAdapter = SyntheticAdapter()
    revision = adapter.normalize_record(
        {
            "vendor_status": "vendor-confirmed",
            "vendor_timing": "vendor-after-bell",
        }
    )
    state = reconstruct_earnings_state((revision,), utc_dt(2026, 9, 2))

    assert isinstance(revision, EarningsScheduleRevision)
    assert revision.lifecycle_state is LifecycleState.CONFIRMED
    assert revision.timing_class is TimingClass.AMC
    assert state.lifecycle_state is LifecycleState.CONFIRMED
    assert state.timing_class is TimingClass.AMC


def test_contract_e09_strategy_effective_at_cannot_precede_knowledge(
    make_revision, utc_dt
) -> None:
    revision = make_revision(
        knowledge_date=date(2026, 9, 15),
        knowledge_available_at=utc_dt(2026, 9, 15),
        strategy_effective_at=utc_dt(2026, 9, 10),
    )

    with pytest.raises(EarningsValidationError) as error:
        validate_revision(revision)

    assert error.value.code == "STRATEGY_EFFECTIVE_BEFORE_KNOWLEDGE"


def test_contract_e10_strategy_effective_at_may_be_later_than_exact_knowledge(
    make_revision, utc_dt
) -> None:
    known_at = utc_dt(2026, 9, 15, 14)
    effective_at = known_at.replace(minute=5)
    revision = make_revision(
        knowledge_date=known_at.date(),
        knowledge_available_at=known_at,
        strategy_effective_at=effective_at,
    )

    assert validate_revision(revision) is revision
    assert revision.strategy_effective_at > revision.knowledge_available_at


def test_contract_e11_single_ticker_candidate_is_still_insufficient_without_mapping_evidence() -> None:
    with pytest.raises(EarningsValidationError) as error:
        validate_security_mapping(
            canonical_asset_id=None,
            historical_symbol="TEST",
            historical_exchange="NYSE",
            candidate_asset_ids=("NORGATE:1001",),
        )

    assert error.value.code == "INSUFFICIENT_SECURITY_MAPPING_EVIDENCE"


def test_contract_e12_date_only_effective_time_cannot_be_on_or_before_knowledge_date(
    make_revision,
) -> None:
    knowledge_date = date(2026, 10, 5)
    invalid_effective_times = (
        datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 4, 15, 0, tzinfo=timezone.utc),
    )

    for effective_at in invalid_effective_times:
        revision = make_revision(
            knowledge_date=knowledge_date,
            knowledge_precision=KnowledgePrecision.DATE_ONLY,
            knowledge_available_at=None,
            strategy_effective_at=effective_at,
        )
        with pytest.raises(EarningsValidationError) as error:
            validate_revision(revision)
        assert error.value.code == "DATE_ONLY_EFFECTIVE_TOO_EARLY"

    later_revision = make_revision(
        knowledge_date=knowledge_date,
        knowledge_precision=KnowledgePrecision.DATE_ONLY,
        knowledge_available_at=None,
        strategy_effective_at=datetime(
            2026, 10, 6, 9, 30, tzinfo=timezone.utc
        ),
    )
    assert validate_revision(later_revision) is later_revision
