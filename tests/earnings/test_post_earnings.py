from stock_swing_d1.earnings import (
    TimingClass,
    earliest_post_event_entry_session,
    first_clean_post_event_session,
)


def test_post_d01_bmo_event_session_is_first_clean_post_event_candle(
    synthetic_calendar,
) -> None:
    event_session = synthetic_calendar.session_date(5)

    first_clean = first_clean_post_event_session(
        event_session, TimingClass.BMO, synthetic_calendar
    )
    earliest_entry = earliest_post_event_entry_session(
        event_session, TimingClass.BMO, synthetic_calendar
    )

    assert first_clean == synthetic_calendar.session_date(5)
    assert earliest_entry == synthetic_calendar.session_date(6)


def test_post_d02_amc_requires_next_session_as_first_clean_candle(
    synthetic_calendar,
) -> None:
    event_session = synthetic_calendar.session_date(5)

    first_clean = first_clean_post_event_session(
        event_session, TimingClass.AMC, synthetic_calendar
    )
    earliest_entry = earliest_post_event_entry_session(
        event_session, TimingClass.AMC, synthetic_calendar
    )

    assert first_clean == synthetic_calendar.session_date(6)
    assert earliest_entry == synthetic_calendar.session_date(7)


def test_post_d03_during_market_requires_next_clean_session(
    synthetic_calendar,
) -> None:
    event_session = synthetic_calendar.session_date(5)

    first_clean = first_clean_post_event_session(
        event_session, TimingClass.DURING_MARKET, synthetic_calendar
    )
    earliest_entry = earliest_post_event_entry_session(
        event_session, TimingClass.DURING_MARKET, synthetic_calendar
    )

    assert first_clean == synthetic_calendar.session_date(6)
    assert earliest_entry == synthetic_calendar.session_date(7)


def test_post_d04_unknown_timing_uses_conservative_next_session(
    synthetic_calendar,
) -> None:
    event_session = synthetic_calendar.session_date(5)

    first_clean = first_clean_post_event_session(
        event_session, TimingClass.UNKNOWN, synthetic_calendar
    )
    earliest_entry = earliest_post_event_entry_session(
        event_session, TimingClass.UNKNOWN, synthetic_calendar
    )

    assert first_clean == synthetic_calendar.session_date(6)
    assert earliest_entry == synthetic_calendar.session_date(7)
