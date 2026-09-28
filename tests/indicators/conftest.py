from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

import pytest

from stock_swing_d1.models import CorporateActionAdjustedStockBar


@pytest.fixture
def make_adjusted_bars():
    def factory(
        count: int,
        *,
        security_id: str = "NORGATE:1",
        symbol: str = "TEST",
        symbols: Sequence[str] | None = None,
        closes: Sequence[float] | None = None,
        opens: Sequence[float] | None = None,
        highs: Sequence[float] | None = None,
        lows: Sequence[float] | None = None,
        volumes: Sequence[float] | None = None,
        dates: Sequence[date] | None = None,
    ) -> list[CorporateActionAdjustedStockBar]:
        close_values = list(closes or (float(index + 1) for index in range(count)))
        if len(close_values) != count:
            raise ValueError("closes must contain count values")
        open_values = list(opens or close_values)
        high_values = list(
            highs or (close_value + 1.0 for close_value in close_values)
        )
        low_values = list(
            lows or (max(0.5, close_value - 1.0) for close_value in close_values)
        )
        volume_values = list(
            volumes or (float((index + 1) * 100) for index in range(count))
        )
        date_values = list(
            dates
            or (date(2020, 1, 1) + timedelta(days=index) for index in range(count))
        )
        symbol_values = list(symbols or (symbol for _ in range(count)))
        sequences: tuple[Sequence[Any], ...] = (
            open_values,
            high_values,
            low_values,
            volume_values,
            date_values,
            symbol_values,
        )
        if any(len(values) != count for values in sequences):
            raise ValueError("all per-bar values must contain count values")

        return [
            CorporateActionAdjustedStockBar(
                security_id=security_id,
                symbol=symbol_values[index],
                trading_date=date_values[index],
                timeframe="D1",
                session_type="regular",
                currency="USD",
                price_basis="capital_special_adjusted",
                open=float(open_values[index]),
                high=float(high_values[index]),
                low=float(low_values[index]),
                close=float(close_values[index]),
                volume=float(volume_values[index]),
            )
            for index in range(count)
        ]

    return factory
