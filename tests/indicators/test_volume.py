import pytest

from stock_swing_d1.indicators import calculate_indicators


def test_average_volume_uses_an_exact_rolling_20_bar_window(
    make_adjusted_bars,
) -> None:
    volumes = [float(value * 10) for value in range(1, 22)]
    rows = calculate_indicators(make_adjusted_bars(21, volumes=volumes))

    assert all(row.avg_volume_20 is None for row in rows[:19])
    assert rows[19].avg_volume_20 == pytest.approx(sum(volumes[:20]) / 20)
    assert rows[20].avg_volume_20 == pytest.approx(sum(volumes[1:21]) / 20)


@pytest.mark.parametrize(
    ("volumes", "relationship"),
    [
        ([100.0] * 20, "equal"),
        ([100.0] * 19 + [200.0], "above"),
        ([100.0] * 19 + [50.0], "below"),
    ],
)
def test_relative_volume_compares_current_volume_to_inclusive_average(
    make_adjusted_bars, volumes, relationship
) -> None:
    value = calculate_indicators(make_adjusted_bars(20, volumes=volumes))[-1]

    if relationship == "equal":
        assert value.relative_volume_20 == 1.0
    elif relationship == "above":
        assert value.relative_volume_20 > 1.0
    else:
        assert value.relative_volume_20 < 1.0


def test_relative_volume_is_none_when_average_is_zero(make_adjusted_bars) -> None:
    row = calculate_indicators(
        make_adjusted_bars(20, volumes=[0.0] * 20)
    )[-1]

    assert row.avg_volume_20 == 0.0
    assert row.relative_volume_20 is None


def test_zero_current_volume_produces_zero_when_average_is_positive(
    make_adjusted_bars,
) -> None:
    row = calculate_indicators(
        make_adjusted_bars(20, volumes=[100.0] * 19 + [0.0])
    )[-1]

    assert row.avg_volume_20 > 0.0
    assert row.relative_volume_20 == 0.0
