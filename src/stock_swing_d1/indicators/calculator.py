"""Deterministic Phase 7A technical-indicator calculations."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from datetime import date
from math import fsum, isfinite

from pydantic import ValidationError

from stock_swing_d1.indicators.models import TechnicalIndicatorRow
from stock_swing_d1.models import CorporateActionAdjustedStockBar


_REQUIRED_PRICE_BASIS = "capital_special_adjusted"
_SMA_PERIODS = (20, 50, 100, 200)
_RSI_PERIOD = 14
_ATR_PERIOD = 14
_VOLUME_PERIOD = 20


class IndicatorValidationError(ValueError):
    """The adjusted-bar input or a calculated value violated the contract."""

    def __init__(
        self,
        reason: str,
        *,
        security_id: str | None = None,
        trading_date: date | None = None,
        duplicate_count: int = 0,
    ) -> None:
        self.reason = reason
        self.security_id = security_id
        self.trading_date = trading_date
        self.duplicate_count = duplicate_count
        super().__init__(reason)


def _validated_groups(
    bars: Iterable[CorporateActionAdjustedStockBar],
) -> dict[str, list[CorporateActionAdjustedStockBar]]:
    normalized: list[CorporateActionAdjustedStockBar] = []

    try:
        iterator = iter(bars)
    except TypeError as error:
        raise IndicatorValidationError(
            "bars must be an iterable of CorporateActionAdjustedStockBar values"
        ) from error

    for bar in iterator:
        if not isinstance(bar, CorporateActionAdjustedStockBar):
            raise IndicatorValidationError(
                "each input must be a CorporateActionAdjustedStockBar"
            )
        record = bar.model_dump(mode="python")
        security_id = record.get("security_id")
        trading_date = record.get("trading_date")
        if record.get("price_basis") != _REQUIRED_PRICE_BASIS:
            raise IndicatorValidationError(
                "price_basis must equal 'capital_special_adjusted'",
                security_id=security_id if isinstance(security_id, str) else None,
                trading_date=trading_date if type(trading_date) is date else None,
            )
        try:
            normalized.append(
                CorporateActionAdjustedStockBar.model_validate(record)
            )
        except ValidationError as error:
            raise IndicatorValidationError(
                f"input is not a valid CorporateActionAdjustedStockBar: {error}",
                security_id=security_id if isinstance(security_id, str) else None,
                trading_date=trading_date if type(trading_date) is date else None,
            ) from error

    normalized.sort(key=lambda bar: (bar.security_id, bar.trading_date))
    duplicate_keys: list[tuple[str, date]] = []
    previous_key: tuple[str, date] | None = None
    for bar in normalized:
        key = (bar.security_id, bar.trading_date)
        if key == previous_key:
            duplicate_keys.append(key)
        previous_key = key
    if duplicate_keys:
        security_id, trading_date = duplicate_keys[0]
        raise IndicatorValidationError(
            f"duplicate security_id and trading_date: {duplicate_keys[0]}",
            security_id=security_id,
            trading_date=trading_date,
            duplicate_count=len(duplicate_keys),
        )

    grouped: dict[str, list[CorporateActionAdjustedStockBar]] = defaultdict(list)
    for bar in normalized:
        grouped[bar.security_id].append(bar)
    return grouped


def _mean(values: Sequence[float]) -> float:
    return fsum(values) / len(values)


def _rsi(average_gain: float, average_loss: float) -> float:
    if average_gain == 0.0 and average_loss == 0.0:
        return 50.0
    if average_loss == 0.0:
        return 100.0
    if average_gain == 0.0:
        return 0.0
    ratio = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + ratio))


def _finite_value(
    value: float | None,
    *,
    field_name: str,
    bar: CorporateActionAdjustedStockBar,
) -> float | None:
    if value is not None and not isfinite(value):
        raise IndicatorValidationError(
            f"calculated {field_name} must be finite",
            security_id=bar.security_id,
            trading_date=bar.trading_date,
        )
    return value


def _calculate_security(
    bars: Sequence[CorporateActionAdjustedStockBar],
) -> list[TechnicalIndicatorRow]:
    close_windows = {period: deque(maxlen=period) for period in _SMA_PERIODS}
    volume_window: deque[float] = deque(maxlen=_VOLUME_PERIOD)
    initial_gains: list[float] = []
    initial_losses: list[float] = []
    initial_true_ranges: list[float] = []
    average_gain: float | None = None
    average_loss: float | None = None
    atr: float | None = None
    previous_close: float | None = None
    rows: list[TechnicalIndicatorRow] = []

    for index, bar in enumerate(bars):
        for window in close_windows.values():
            window.append(bar.close)
        volume_window.append(bar.volume)

        sma_values = {
            period: (_mean(window) if len(window) == period else None)
            for period, window in close_windows.items()
        }
        average_volume = (
            _mean(volume_window)
            if len(volume_window) == _VOLUME_PERIOD
            else None
        )
        relative_volume = (
            None
            if average_volume is None or average_volume == 0.0
            else bar.volume / average_volume
        )

        rsi: float | None = None
        if previous_close is not None:
            change = bar.close - previous_close
            gain = max(change, 0.0)
            loss = max(-change, 0.0)
            if index <= _RSI_PERIOD:
                initial_gains.append(gain)
                initial_losses.append(loss)
                if index == _RSI_PERIOD:
                    average_gain = _mean(initial_gains)
                    average_loss = _mean(initial_losses)
                    rsi = _rsi(average_gain, average_loss)
            else:
                assert average_gain is not None and average_loss is not None
                average_gain = (
                    average_gain * (_RSI_PERIOD - 1) + gain
                ) / _RSI_PERIOD
                average_loss = (
                    average_loss * (_RSI_PERIOD - 1) + loss
                ) / _RSI_PERIOD
                rsi = _rsi(average_gain, average_loss)

        true_range = bar.high - bar.low
        if previous_close is not None:
            true_range = max(
                true_range,
                abs(bar.high - previous_close),
                abs(bar.low - previous_close),
            )
        if index < _ATR_PERIOD:
            initial_true_ranges.append(true_range)
            if index == _ATR_PERIOD - 1:
                atr = _mean(initial_true_ranges)
        else:
            assert atr is not None
            atr = (atr * (_ATR_PERIOD - 1) + true_range) / _ATR_PERIOD

        values = {
            "sma_20": sma_values[20],
            "sma_50": sma_values[50],
            "sma_100": sma_values[100],
            "sma_200": sma_values[200],
            "rsi_14": rsi,
            "atr_14": atr,
            "avg_volume_20": average_volume,
            "relative_volume_20": relative_volume,
        }
        finite_values = {
            field_name: _finite_value(
                value, field_name=field_name, bar=bar
            )
            for field_name, value in values.items()
        }
        rows.append(
            TechnicalIndicatorRow(
                security_id=bar.security_id,
                symbol=bar.symbol,
                trading_date=bar.trading_date,
                price_basis=bar.price_basis,
                **finite_values,
            )
        )
        previous_close = bar.close

    return rows


def calculate_indicators(
    bars: Iterable[CorporateActionAdjustedStockBar],
) -> tuple[TechnicalIndicatorRow, ...]:
    """Calculate frozen v0.1 indicators for adjusted D1 bars.

    Histories are independent per stable ``security_id``.  Physical input
    order is ignored; output is canonical ``security_id``, then ascending
    ``trading_date`` order.  Every valid input bar produces one row, including
    all warm-up rows.
    """

    grouped = _validated_groups(bars)
    rows: list[TechnicalIndicatorRow] = []
    for security_id in sorted(grouped):
        rows.extend(_calculate_security(grouped[security_id]))
    return tuple(rows)


__all__ = ["IndicatorValidationError", "calculate_indicators"]
