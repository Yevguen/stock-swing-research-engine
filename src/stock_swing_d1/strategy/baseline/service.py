"""Phase 8 Baseline Strategy v0.1 signal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Final

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
    PublishedEarningsRiskOverlay,
)
from stock_swing_d1.indicators import TechnicalIndicatorRow
from stock_swing_d1.models import CorporateActionAdjustedStockBar
from stock_swing_d1.strategy.baseline.models import (
    BaselineSignalCalendar,
    BaselineSignalAction,
    BaselineSignalDecision,
    BaselineStrategyValidationError,
)


_REQUIRED_PRICE_BASIS: Final[str] = "capital_special_adjusted"
_RSI_MINIMUM: Final[float] = 50.0
_ATR_MINIMUM_FRACTION: Final[float] = 0.01


def _is_aware(value: object) -> bool:
    try:
        return (
            isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        )
    except Exception:
        return False


def _require_finite_number(
    value: object,
    *,
    field_name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise BaselineStrategyValidationError(
            "INVALID_NUMERIC_INPUT", f"{field_name} must be finite"
        )
    if positive and value <= 0.0:
        raise BaselineStrategyValidationError(
            "INVALID_ADJUSTED_CLOSE", f"{field_name} must be greater than zero"
        )
    if nonnegative and value < 0.0:
        raise BaselineStrategyValidationError(
            "INVALID_INDICATOR_VALUE", f"{field_name} must be nonnegative"
        )


def _validate_inputs(
    *,
    bar: CorporateActionAdjustedStockBar,
    indicators: TechnicalIndicatorRow,
    universe_eligible: bool,
) -> None:
    if not isinstance(bar, CorporateActionAdjustedStockBar):
        raise BaselineStrategyValidationError(
            "INVALID_BAR", "bar must be a CorporateActionAdjustedStockBar"
        )
    if not isinstance(indicators, TechnicalIndicatorRow):
        raise BaselineStrategyValidationError(
            "INVALID_INDICATORS",
            "indicators must be a TechnicalIndicatorRow",
        )
    if type(universe_eligible) is not bool:
        raise BaselineStrategyValidationError(
            "INVALID_UNIVERSE_ELIGIBILITY",
            "universe_eligible must be an explicit Boolean",
        )

    mismatches = tuple(
        field_name
        for field_name in ("security_id", "symbol", "trading_date", "price_basis")
        if getattr(bar, field_name) != getattr(indicators, field_name)
    )
    if mismatches:
        raise BaselineStrategyValidationError(
            "BAR_INDICATOR_IDENTITY_MISMATCH",
            "bar and indicators disagree on " + ", ".join(mismatches),
        )
    if (
        bar.price_basis != _REQUIRED_PRICE_BASIS
        or indicators.price_basis != _REQUIRED_PRICE_BASIS
    ):
        raise BaselineStrategyValidationError(
            "INVALID_PRICE_BASIS",
            "bar and indicators must use capital_special_adjusted",
        )
    _require_finite_number(
        bar.close, field_name="bar.close", positive=True
    )
    for field_name in ("sma_20", "sma_50", "rsi_14", "atr_14"):
        value = getattr(indicators, field_name)
        if value is not None:
            _require_finite_number(
                value,
                field_name=f"indicators.{field_name}",
                nonnegative=field_name == "atr_14",
            )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class BaselineSignalEvaluator:
    """Evaluate the frozen long-only baseline after one completed D1 session."""

    _earnings_overlay: PublishedEarningsRiskOverlay
    _trading_calendar: BaselineSignalCalendar

    def __init__(
        self,
        *,
        earnings_overlay: PublishedEarningsRiskOverlay,
        trading_calendar: BaselineSignalCalendar,
    ) -> None:
        object.__setattr__(self, "_earnings_overlay", earnings_overlay)
        object.__setattr__(self, "_trading_calendar", trading_calendar)

    def evaluate_signal(
        self,
        *,
        bar: CorporateActionAdjustedStockBar,
        indicators: TechnicalIndicatorRow,
        universe_eligible: bool,
    ) -> BaselineSignalDecision:
        """Return a diagnostic T signal and prospective T+1 entry session."""

        _validate_inputs(
            bar=bar,
            indicators=indicators,
            universe_eligible=universe_eligible,
        )
        signal_session = bar.trading_date

        try:
            signal_time = self._trading_calendar.signal_decision_time(
                signal_session
            )
        except Exception as error:
            raise BaselineStrategyValidationError(
                "INVALID_SIGNAL_SESSION",
                "the signal calendar could not resolve signal_decision_time",
            ) from error
        if not _is_aware(signal_time):
            raise BaselineStrategyValidationError(
                "INVALID_SIGNAL_DECISION_TIME",
                "signal_decision_time must be an explicitly timezone-aware datetime",
            )

        try:
            planned_entry_session = self._trading_calendar.next_session(
                signal_session
            )
        except Exception as error:
            raise BaselineStrategyValidationError(
                "INVALID_PLANNED_ENTRY_SESSION",
                "the trading calendar could not derive the next session",
            ) from error
        if (
            type(planned_entry_session) is not date
            or planned_entry_session <= signal_session
        ):
            raise BaselineStrategyValidationError(
                "INVALID_PLANNED_ENTRY_SESSION",
                "planned entry must be the next trading session after the signal",
            )

        close_above_sma50 = (
            indicators.sma_50 is not None
            and bar.close > indicators.sma_50
        )
        sma20_above_sma50 = (
            indicators.sma_20 is not None
            and indicators.sma_50 is not None
            and indicators.sma_20 > indicators.sma_50
        )
        rsi_above_50 = (
            indicators.rsi_14 is not None
            and indicators.rsi_14 > _RSI_MINIMUM
        )
        atr_fraction = (
            None
            if indicators.atr_14 is None
            else indicators.atr_14 / bar.close
        )
        atr_above_minimum = (
            atr_fraction is not None
            and atr_fraction >= _ATR_MINIMUM_FRACTION
        )

        earnings_decision = self._earnings_overlay.evaluate_entry_candidate(
            canonical_asset_id=bar.security_id,
            signal_time=signal_time,
            signal_session=signal_session,
            planned_entry_session=planned_entry_session,
        )
        if not isinstance(earnings_decision, EarningsIntegrationDecision):
            raise BaselineStrategyValidationError(
                "INVALID_EARNINGS_DECISION",
                "earnings overlay must return EarningsIntegrationDecision",
            )
        earnings_entry_allowed = (
            earnings_decision.action
            is EarningsIntegrationAction.ENTRY_ALLOWED
        )

        valid_long_signal = all(
            (
                universe_eligible,
                close_above_sma50,
                sma20_above_sma50,
                rsi_above_50,
                atr_above_minimum,
                earnings_entry_allowed,
            )
        )
        action = (
            BaselineSignalAction.VALID_LONG_SIGNAL
            if valid_long_signal
            else BaselineSignalAction.NO_SIGNAL
        )

        return BaselineSignalDecision(
            security_id=bar.security_id,
            symbol=bar.symbol,
            signal_session=signal_session,
            signal_time=signal_time,
            planned_entry_session=planned_entry_session,
            adjusted_close=bar.close,
            sma_20=indicators.sma_20,
            sma_50=indicators.sma_50,
            rsi_14=indicators.rsi_14,
            atr_14=indicators.atr_14,
            atr_fraction=atr_fraction,
            universe_eligible=universe_eligible,
            close_above_sma50=close_above_sma50,
            sma20_above_sma50=sma20_above_sma50,
            rsi_above_50=rsi_above_50,
            atr_above_minimum=atr_above_minimum,
            earnings_entry_allowed=earnings_entry_allowed,
            earnings_action=earnings_decision.action,
            earnings_reason=earnings_decision.reason,
            earnings_decision=earnings_decision,
            action=action,
        )


__all__ = ["BaselineSignalEvaluator"]
