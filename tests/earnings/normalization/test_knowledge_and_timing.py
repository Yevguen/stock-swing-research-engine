from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from stock_swing_d1.earnings import KnowledgePrecision, TimingClass
from stock_swing_d1.earnings.normalization import normalize_earnings_delivery


def _normalize(
    raw,
    *,
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    build_id="knowledge-timing",
):
    return normalize_earnings_delivery(
        (raw,),
        adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        trading_calendar=normalization_calendar,
        canonical_history=None,
        build_id=build_id,
        ingested_at=datetime(2026, 10, 10, 12, tzinfo=timezone.utc),
    )


def test_norm_c01_exact_provider_knowledge_timestamp_is_preserved(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    source_instant = datetime(2026, 9, 1, 16, tzinfo=ZoneInfo("Europe/Madrid"))
    result = _normalize(
        raw_record(
            knowledge_date=source_instant.date(),
            knowledge_available_at=source_instant,
        ),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    )
    assert result.success
    assert result.revisions[0].knowledge_available_at == source_instant.astimezone(
        timezone.utc
    )


def test_norm_c02_date_only_knowledge_does_not_fabricate_provider_timestamp(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    result = _normalize(
        raw_record(
            knowledge_precision_code="DATE",
            knowledge_date=date(2026, 10, 2),
            knowledge_available_at=None,
        ),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    )
    assert result.success
    assert result.revisions[0].knowledge_precision is KnowledgePrecision.DATE_ONLY
    assert result.revisions[0].knowledge_available_at is None


def test_norm_c03_date_only_knowledge_becomes_effective_at_next_valid_trading_session(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    weekend = _normalize(
        raw_record(
            knowledge_precision_code="DATE",
            knowledge_date=date(2026, 10, 2),
            knowledge_available_at=None,
        ),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        build_id="norm-c03-weekend",
    )
    holiday = _normalize(
        raw_record(
            revision_key="HOLIDAY",
            knowledge_precision_code="DATE",
            knowledge_date=date(2026, 9, 4),
            knowledge_available_at=None,
        ),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        build_id="norm-c03-holiday",
    )
    assert weekend.revisions[0].strategy_effective_at == normalization_calendar.decision_time(
        date(2026, 10, 5)
    ).astimezone(timezone.utc)
    assert holiday.revisions[0].strategy_effective_at == normalization_calendar.decision_time(
        date(2026, 9, 8)
    ).astimezone(timezone.utc)


def test_norm_c04_date_only_knowledge_never_creates_midnight_timestamp(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    result = _normalize(
        raw_record(
            knowledge_precision_code="DATE",
            knowledge_date=date(2026, 10, 2),
            knowledge_available_at=None,
        ),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    )
    revision = result.revisions[0]
    assert revision.knowledge_available_at is None
    assert revision.strategy_effective_at.time().replace(tzinfo=None).isoformat() != "00:00:00"


def test_norm_c05_timestamp_knowledge_uses_exact_available_instant_as_effective_time(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    available_at = datetime(2026, 9, 1, 15, 42, 13, tzinfo=timezone.utc)
    result = _normalize(
        raw_record(knowledge_available_at=available_at),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    )
    assert result.revisions[0].strategy_effective_at == available_at


def test_norm_c06_coarse_timing_does_not_fabricate_exact_event_time(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    expected = {
        "PRE": TimingClass.BMO,
        "OPEN": TimingClass.DURING_MARKET,
        "POST": TimingClass.AMC,
        "UNK": TimingClass.UNKNOWN,
    }
    for index, (provider_code, canonical_timing) in enumerate(expected.items(), start=1):
        result = _normalize(
            raw_record(
                revision_key=f"COARSE-{index}",
                timing_code=provider_code,
                scheduled_at=None,
            ),
            delivery_adapter=delivery_adapter,
            identity_resolver=identity_resolver,
            normalization_calendar=normalization_calendar,
            build_id=f"norm-c06-{index}",
        )
        assert result.success
        assert result.revisions[0].timing_class is canonical_timing
        assert result.revisions[0].scheduled_at is None


def test_norm_c07_exact_event_time_requires_timezone_aware_timestamp_and_timezone(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    naive = _normalize(
        raw_record(
            timing_code="CLOCK",
            scheduled_at=datetime(2026, 10, 14, 16),
            event_timezone="America/New_York",
        ),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        build_id="norm-c07-naive",
    )
    missing_zone = _normalize(
        raw_record(
            revision_key="NO-ZONE",
            timing_code="CLOCK",
            scheduled_at=datetime(2026, 10, 14, 20, tzinfo=timezone.utc),
            event_timezone=None,
        ),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
        build_id="norm-c07-zone",
    )
    assert not naive.success and not missing_zone.success
    assert naive.revisions == () and missing_zone.revisions == ()


def test_norm_c08_coarse_timing_and_real_exact_timestamp_may_both_be_preserved(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    exact = datetime(2026, 10, 14, 16, 5, tzinfo=ZoneInfo("America/New_York"))
    result = _normalize(
        raw_record(
            timing_code="POST",
            scheduled_at=exact,
            event_timezone="America/New_York",
        ),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    )
    assert result.success
    assert result.revisions[0].timing_class is TimingClass.AMC
    assert result.revisions[0].scheduled_at == exact.astimezone(timezone.utc)
    assert result.revisions[0].event_timezone == "America/New_York"


def test_norm_c09_unmapped_provider_values_fail_instead_of_becoming_unknown_or_other(
    delivery_adapter,
    identity_resolver,
    normalization_calendar,
    raw_record,
) -> None:
    result = _normalize(
        raw_record(timing_code="UNMAPPED-TIMING"),
        delivery_adapter=delivery_adapter,
        identity_resolver=identity_resolver,
        normalization_calendar=normalization_calendar,
    )
    assert not result.success
    assert result.revisions == ()
    assert result.quarantines[0].error_code == "UNMAPPED_PROVIDER_VALUE"
