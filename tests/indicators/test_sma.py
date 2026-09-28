from datetime import date, timedelta

import pytest

from stock_swing_d1.indicators import calculate_indicators


def test_sma20_has_an_exact_full_window_and_includes_current_bar(
    make_adjusted_bars,
) -> None:
    rows = calculate_indicators(make_adjusted_bars(21))

    assert all(row.sma_20 is None for row in rows[:19])
    assert rows[19].sma_20 == pytest.approx(sum(range(1, 21)) / 20)
    assert rows[20].sma_20 == pytest.approx(sum(range(2, 22)) / 20)


@pytest.mark.parametrize(
    ("period", "field_name"),
    [(50, "sma_50"), (100, "sma_100"), (200, "sma_200")],
)
def test_long_smas_begin_only_at_their_exact_period(
    make_adjusted_bars, period: int, field_name: str
) -> None:
    rows = calculate_indicators(make_adjusted_bars(period))

    assert all(getattr(row, field_name) is None for row in rows[:-1])
    assert getattr(rows[-1], field_name) == pytest.approx((period + 1) / 2)


def test_windows_count_observed_bars_not_calendar_days(make_adjusted_bars) -> None:
    dates = [date(2020, 1, 1) + timedelta(days=index * 7) for index in range(20)]

    rows = calculate_indicators(make_adjusted_bars(20, dates=dates))

    assert rows[-1].sma_20 == pytest.approx(10.5)
