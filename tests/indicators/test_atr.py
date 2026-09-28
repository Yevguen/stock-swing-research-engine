import pytest

from stock_swing_d1.indicators import calculate_indicators


def test_first_true_range_is_high_minus_low(make_adjusted_bars) -> None:
    rows = calculate_indicators(
        make_adjusted_bars(
            14,
            closes=[10.0] * 14,
            highs=[12.0] * 14,
            lows=[8.0] * 14,
        )
    )

    assert all(row.atr_14 is None for row in rows[:13])
    assert rows[13].atr_14 == 4.0


def test_atr14_seed_handles_gap_up_and_gap_down(make_adjusted_bars) -> None:
    closes = [10.0] * 12 + [14.5, 8.5]
    highs = [11.0] + [10.5] * 11 + [15.0, 9.0]
    lows = [9.0] + [9.5] * 11 + [14.0, 8.0]

    rows = calculate_indicators(
        make_adjusted_bars(14, closes=closes, highs=highs, lows=lows)
    )

    # TRs: 2.0, eleven 1.0 values, a 5.0 gap-up, and a 6.5 gap-down.
    assert rows[13].atr_14 == pytest.approx((2.0 + 11.0 + 5.0 + 6.5) / 14)


def test_atr14_bar_15_uses_wilder_smoothing(make_adjusted_bars) -> None:
    closes = [10.0] * 12 + [14.5, 8.5, 19.5]
    highs = [11.0] + [10.5] * 11 + [15.0, 9.0, 20.0]
    lows = [9.0] + [9.5] * 11 + [14.0, 8.0, 19.0]
    rows = calculate_indicators(
        make_adjusted_bars(15, closes=closes, highs=highs, lows=lows)
    )
    seed = (2.0 + 11.0 + 5.0 + 6.5) / 14
    bar_15_true_range = 11.5

    assert rows[14].atr_14 == pytest.approx(
        (seed * 13 + bar_15_true_range) / 14
    )
