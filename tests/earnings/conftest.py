"""Deterministic fixtures for provider-neutral earnings tests."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from tests.earnings._builders import make_revision as _make_revision
from tests.earnings._builders import make_state as _make_state


class SyntheticTradingCalendar:
    sessions = (
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

    def session_date(self, index: int) -> date:
        return self.sessions[index]

    def previous_session(self, session: date) -> date:
        index = self.sessions.index(session)
        if index == 0:
            raise ValueError("no previous synthetic session")
        return self.sessions[index - 1]

    def next_session(self, session: date) -> date:
        for candidate in self.sessions:
            if candidate > session:
                return candidate
        raise ValueError("no next synthetic session")

    def session_distance(self, start: date, end: date) -> int:
        return self.sessions.index(end) - self.sessions.index(start)

    def decision_time(self, session: date) -> datetime:
        if session not in self.sessions:
            raise ValueError("not a synthetic session")
        return datetime.combine(
            session, time(9, 30), tzinfo=ZoneInfo("America/New_York")
        )


@pytest.fixture
def synthetic_calendar() -> SyntheticTradingCalendar:
    return SyntheticTradingCalendar()


@pytest.fixture
def make_revision():
    return _make_revision


@pytest.fixture
def make_state():
    return _make_state


@pytest.fixture
def canonical_asset_id() -> str:
    return "NORGATE:1001"


@pytest.fixture
def event_instance_id() -> str:
    return "EARN-2026-Q3"


@pytest.fixture
def utc_dt():
    def factory(year: int, month: int, day: int, hour: int = 14) -> datetime:
        return datetime(year, month, day, hour, tzinfo=timezone.utc)

    return factory


@pytest.fixture
def ny_dt():
    def factory(year: int, month: int, day: int, hour: int = 9, minute: int = 30) -> datetime:
        return datetime(
            year,
            month,
            day,
            hour,
            minute,
            tzinfo=ZoneInfo("America/New_York"),
        )

    return factory
