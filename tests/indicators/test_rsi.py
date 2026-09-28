import pytest

from stock_swing_d1.indicators import calculate_indicators


def test_rsi14_seed_uses_first_14_changes_and_begins_on_bar_15(
    make_adjusted_bars,
) -> None:
    changes = [1.0, -2.0, 3.0, -1.0, 2.0, -4.0, 1.0, 2.0, -3.0, 4.0,
               -2.0, 1.0, -1.0, 3.0]
    closes = [50.0]
    for change in changes:
        closes.append(closes[-1] + change)
    rows = calculate_indicators(make_adjusted_bars(15, closes=closes))
    average_gain = sum(max(change, 0.0) for change in changes) / 14
    average_loss = sum(max(-change, 0.0) for change in changes) / 14
    expected = 100.0 - 100.0 / (1.0 + average_gain / average_loss)

    assert all(row.rsi_14 is None for row in rows[:14])
    assert rows[14].rsi_14 == pytest.approx(expected)


def test_rsi14_uses_wilder_recursive_smoothing(make_adjusted_bars) -> None:
    seed_changes = [1.0, -2.0, 3.0, -1.0, 2.0, -4.0, 1.0, 2.0, -3.0, 4.0,
                    -2.0, 1.0, -1.0, 3.0]
    next_change = -5.0
    closes = [50.0]
    for change in [*seed_changes, next_change]:
        closes.append(closes[-1] + change)
    rows = calculate_indicators(make_adjusted_bars(16, closes=closes))
    seed_gain = sum(max(change, 0.0) for change in seed_changes) / 14
    seed_loss = sum(max(-change, 0.0) for change in seed_changes) / 14
    average_gain = (seed_gain * 13 + max(next_change, 0.0)) / 14
    average_loss = (seed_loss * 13 + max(-next_change, 0.0)) / 14
    expected = 100.0 - 100.0 / (1.0 + average_gain / average_loss)

    assert rows[15].rsi_14 == pytest.approx(expected)


@pytest.mark.parametrize(
    ("closes", "expected"),
    [
        ([10.0] * 16, 50.0),
        ([float(value) for value in range(1, 17)], 100.0),
        ([float(value) for value in range(16, 0, -1)], 0.0),
    ],
    ids=["flat", "rising", "falling"],
)
def test_rsi14_frozen_edge_cases(make_adjusted_bars, closes, expected) -> None:
    rows = calculate_indicators(make_adjusted_bars(16, closes=closes))

    assert rows[14].rsi_14 == expected
    assert rows[15].rsi_14 == expected
    assert 0.0 <= rows[15].rsi_14 <= 100.0
