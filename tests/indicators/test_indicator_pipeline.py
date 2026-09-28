from __future__ import annotations

from datetime import date, timedelta
import inspect
import random

import pytest
from pydantic import ValidationError

from stock_swing_d1.indicators import (
    IndicatorValidationError,
    TechnicalIndicatorRow,
    calculate_indicators,
)
from stock_swing_d1.indicators import calculator


def test_interleaved_securities_have_independent_histories(
    make_adjusted_bars,
) -> None:
    first = make_adjusted_bars(
        20, security_id="NORGATE:1", closes=[float(i) for i in range(1, 21)]
    )
    second = make_adjusted_bars(
        20, security_id="NORGATE:2", closes=[100.0] * 20
    )
    interleaved = [bar for pair in zip(first, second) for bar in pair]

    rows = calculate_indicators(interleaved)
    first_last = [row for row in rows if row.security_id == "NORGATE:1"][-1]
    second_last = [row for row in rows if row.security_id == "NORGATE:2"][-1]

    assert first_last.sma_20 == pytest.approx(10.5)
    assert second_last.sma_20 == 100.0


def test_ticker_change_does_not_reset_stable_security_history(
    make_adjusted_bars,
) -> None:
    symbols = ["OLD"] * 10 + ["NEW"] * 11
    rows = calculate_indicators(make_adjusted_bars(21, symbols=symbols))

    assert rows[9].symbol == "OLD"
    assert rows[10].symbol == "NEW"
    assert rows[19].sma_20 == pytest.approx(10.5)


def test_same_ticker_different_security_ids_remain_independent(
    make_adjusted_bars,
) -> None:
    bars = make_adjusted_bars(
        20, security_id="NORGATE:1", symbol="SAME", closes=[10.0] * 20
    )
    bars += make_adjusted_bars(
        20, security_id="NORGATE:2", symbol="SAME", closes=[50.0] * 20
    )

    rows = calculate_indicators(bars)

    assert rows[19].sma_20 == 10.0
    assert rows[39].sma_20 == 50.0


def test_shuffled_input_has_identical_canonical_output(make_adjusted_bars) -> None:
    bars = make_adjusted_bars(30, security_id="NORGATE:2")
    bars += make_adjusted_bars(30, security_id="NORGATE:1", closes=[20.0] * 30)
    shuffled = list(bars)
    random.Random(703).shuffle(shuffled)

    assert calculate_indicators(shuffled) == calculate_indicators(bars)


def test_duplicate_security_date_fails_closed(make_adjusted_bars) -> None:
    bar = make_adjusted_bars(1)[0]

    with pytest.raises(IndicatorValidationError, match="duplicate") as raised:
        calculate_indicators([bar, bar.model_copy(update={"symbol": "RENAMED"})])

    assert raised.value.duplicate_count == 1
    assert raised.value.security_id == bar.security_id
    assert raised.value.trading_date == bar.trading_date


def test_wrong_price_basis_fails_at_public_boundary(make_adjusted_bars) -> None:
    bar = make_adjusted_bars(1)[0].model_copy(
        update={"price_basis": "unadjusted"}
    )

    with pytest.raises(IndicatorValidationError, match="price_basis"):
        calculate_indicators([bar])


def test_non_adjusted_bar_value_fails_at_public_boundary() -> None:
    with pytest.raises(IndicatorValidationError, match="CorporateActionAdjusted"):
        calculate_indicators([object()])


def test_warm_up_rows_are_preserved_and_all_features_exist_at_bar_200(
    make_adjusted_bars,
) -> None:
    rows = calculate_indicators(make_adjusted_bars(200))

    assert len(rows) == 200
    assert rows[198].sma_200 is None
    assert rows[199].sma_20 is not None
    assert rows[199].sma_50 is not None
    assert rows[199].sma_100 is not None
    assert rows[199].sma_200 is not None
    assert rows[199].rsi_14 is not None
    assert rows[199].atr_14 is not None
    assert rows[199].avg_volume_20 is not None
    assert rows[199].relative_volume_20 is not None


def test_future_bar_does_not_change_prior_indicator_row(make_adjusted_bars) -> None:
    bars = make_adjusted_bars(200)
    before = calculate_indicators(bars)[-1]
    future_date = bars[-1].trading_date + timedelta(days=1)
    future = make_adjusted_bars(
        1,
        closes=[1_000_000.0],
        highs=[2_000_000.0],
        lows=[1.0],
        volumes=[1_000_000_000.0],
        dates=[future_date],
    )[0]

    after = calculate_indicators([*bars, future])[-2]

    assert after == before


def test_output_model_is_immutable_and_has_exact_v01_fields(
    make_adjusted_bars,
) -> None:
    row = calculate_indicators(make_adjusted_bars(1))[0]

    assert set(TechnicalIndicatorRow.model_fields) == {
        "security_id",
        "symbol",
        "trading_date",
        "price_basis",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",
        "rsi_14",
        "atr_14",
        "avg_volume_20",
        "relative_volume_20",
    }
    with pytest.raises(ValidationError, match="frozen_instance"):
        row.sma_20 = 1.0


def test_indicator_package_has_no_adjustment_provider_or_strategy_logic() -> None:
    source = inspect.getsource(calculator).lower()

    for forbidden in (
        "norgate",
        "earnings",
        "corporate_action_pipeline",
        "split_factor",
        "fillna",
        "interpolate",
        "buy_signal",
        "sell_signal",
        "position_size",
        "macd",
        "adx",
    ):
        assert forbidden not in source


def test_empty_iterable_returns_empty_immutable_result() -> None:
    assert calculate_indicators(iter(())) == ()
