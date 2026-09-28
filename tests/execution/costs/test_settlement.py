"""Historical settlement boundaries and certified-calendar behavior."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime

import pytest

from stock_swing_d1.execution.costs import (
    ExecutionCostValidationError,
    HistoricalSettlementRegime,
    HistoricalUsEquitySettlementResolver,
)


class ExplicitSettlementCalendar:
    def __init__(self, sessions: tuple[date, ...]) -> None:
        self.sessions = sessions
        self.calls: list[date] = []

    def next_settlement_session(self, session: date) -> date:
        self.calls.append(session)
        return min(candidate for candidate in self.sessions if candidate > session)


@pytest.mark.parametrize(
    ("trade_session", "regime", "lag"),
    [
        (date(1995, 6, 7), HistoricalSettlementRegime.T_PLUS_3, 3),
        (date(2017, 9, 4), HistoricalSettlementRegime.T_PLUS_3, 3),
        (date(2017, 9, 5), HistoricalSettlementRegime.T_PLUS_2, 2),
        (date(2024, 5, 27), HistoricalSettlementRegime.T_PLUS_2, 2),
        (date(2024, 5, 28), HistoricalSettlementRegime.T_PLUS_1, 1),
    ],
)
def test_exact_historical_regime_boundaries(
    trade_session: date,
    regime: HistoricalSettlementRegime,
    lag: int,
) -> None:
    sessions = tuple(
        date.fromordinal(trade_session.toordinal() + offset)
        for offset in range(1, 8)
    )
    calendar = ExplicitSettlementCalendar(sessions)
    result = HistoricalUsEquitySettlementResolver(
        settlement_calendar=calendar
    ).resolve(trade_session=trade_session)
    assert result.regime is regime
    assert result.lag_sessions == lag
    assert result.settlement_session == sessions[lag - 1]
    assert len(calendar.calls) == lag


@pytest.mark.parametrize(
    ("trade_session", "settlement_sessions", "regime", "settlement_session"),
    [
        (
            date(2017, 9, 1),
            (date(2017, 9, 5), date(2017, 9, 6), date(2017, 9, 7)),
            HistoricalSettlementRegime.T_PLUS_3,
            date(2017, 9, 7),
        ),
        (
            date(2017, 9, 5),
            (date(2017, 9, 6), date(2017, 9, 7)),
            HistoricalSettlementRegime.T_PLUS_2,
            date(2017, 9, 7),
        ),
        (
            date(2024, 5, 24),
            (date(2024, 5, 28), date(2024, 5, 29)),
            HistoricalSettlementRegime.T_PLUS_2,
            date(2024, 5, 29),
        ),
        (
            date(2024, 5, 28),
            (date(2024, 5, 29),),
            HistoricalSettlementRegime.T_PLUS_1,
            date(2024, 5, 29),
        ),
    ],
)
def test_actual_valid_sessions_on_both_sides_of_historical_cutovers(
    trade_session: date,
    settlement_sessions: tuple[date, ...],
    regime: HistoricalSettlementRegime,
    settlement_session: date,
) -> None:
    result = HistoricalUsEquitySettlementResolver(
        settlement_calendar=ExplicitSettlementCalendar(settlement_sessions)
    ).resolve(trade_session=trade_session)

    assert result.regime is regime
    assert result.settlement_session == settlement_session


def test_day_before_supported_history_fails_closed() -> None:
    resolver = HistoricalUsEquitySettlementResolver(
        settlement_calendar=ExplicitSettlementCalendar((date(1995, 6, 7),))
    )
    with pytest.raises(ExecutionCostValidationError) as raised:
        resolver.resolve(trade_session=date(1995, 6, 6))
    assert raised.value.code == "UNSUPPORTED_TRADE_SESSION"


def test_friday_weekend_holiday_and_consecutive_unavailable_dates_are_skipped() -> None:
    trade_session = date(2024, 5, 24)
    # Weekend May 25-26 and holiday May 27 are deliberately absent.
    calendar = ExplicitSettlementCalendar(
        (date(2024, 5, 28), date(2024, 5, 29), date(2024, 5, 30))
    )
    result = HistoricalUsEquitySettlementResolver(
        settlement_calendar=calendar
    ).resolve(trade_session=trade_session)
    assert result.regime is HistoricalSettlementRegime.T_PLUS_2
    assert result.settlement_session == date(2024, 5, 29)
    assert calendar.calls == [date(2024, 5, 24), date(2024, 5, 28)]


def test_calendar_non_forward_response_fails_closed() -> None:
    class BrokenCalendar:
        @staticmethod
        def next_settlement_session(session: date) -> date:
            return session

    resolver = HistoricalUsEquitySettlementResolver(
        settlement_calendar=BrokenCalendar()
    )
    with pytest.raises(ExecutionCostValidationError) as raised:
        resolver.resolve(trade_session=date(2024, 5, 28))
    assert raised.value.code == "INVALID_SETTLEMENT_CALENDAR_RESPONSE"


def test_calendar_exception_is_converted_to_domain_error() -> None:
    class BrokenCalendar:
        @staticmethod
        def next_settlement_session(session: date) -> date:
            raise RuntimeError("provider detail")

    resolver = HistoricalUsEquitySettlementResolver(
        settlement_calendar=BrokenCalendar()
    )
    with pytest.raises(ExecutionCostValidationError) as raised:
        resolver.resolve(trade_session=date(2024, 5, 28))
    assert raised.value.code == "SETTLEMENT_CALENDAR_FAILURE"
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_trade_session_must_be_a_genuine_date() -> None:
    resolver = HistoricalUsEquitySettlementResolver(
        settlement_calendar=ExplicitSettlementCalendar((date(2024, 5, 29),))
    )
    with pytest.raises(ExecutionCostValidationError):
        resolver.resolve(trade_session=datetime(2024, 5, 28))  # type: ignore[arg-type]


def test_resolution_is_deterministic_immutable_and_self_validating() -> None:
    sessions = (date(2024, 5, 29),)
    resolver = HistoricalUsEquitySettlementResolver(
        settlement_calendar=ExplicitSettlementCalendar(sessions)
    )
    first = resolver.resolve(trade_session=date(2024, 5, 28))
    second = resolver.resolve(trade_session=date(2024, 5, 28))
    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.lag_sessions = 2  # type: ignore[misc]
    with pytest.raises(ExecutionCostValidationError) as raised:
        replace(first, lag_sessions=2)
    assert raised.value.code == "INVALID_SETTLEMENT_LAG"
