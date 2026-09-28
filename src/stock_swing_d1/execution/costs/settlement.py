"""Historical U.S. equity settlement resolution through an injected calendar."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from stock_swing_d1.execution.costs.models import (
    HistoricalSettlementRegime,
    SettlementResolution,
    SettlementSessionCalendar,
    ExecutionCostValidationError,
)


_EARLIEST_SUPPORTED_TRADE_SESSION = date(1995, 6, 7)
_T_PLUS_2_START = date(2017, 9, 5)
_T_PLUS_1_START = date(2024, 5, 28)


@dataclass(frozen=True, slots=True)
class HistoricalUsEquitySettlementResolver:
    """Resolve the historical standard cycle using certified sessions only."""

    settlement_calendar: SettlementSessionCalendar

    def __post_init__(self) -> None:
        if not callable(
            getattr(self.settlement_calendar, "next_settlement_session", None)
        ):
            raise ExecutionCostValidationError(
                "INVALID_SETTLEMENT_CALENDAR",
                "settlement_calendar must provide next_settlement_session",
            )

    def resolve(self, *, trade_session: date) -> SettlementResolution:
        """Return T+3, T+2, or T+1 by advancing certified sessions."""

        if type(trade_session) is not date:
            raise ExecutionCostValidationError(
                "INVALID_TRADE_SESSION", "trade_session must be a genuine date"
            )
        if trade_session < _EARLIEST_SUPPORTED_TRADE_SESSION:
            raise ExecutionCostValidationError(
                "UNSUPPORTED_TRADE_SESSION",
                "trade sessions before 1995-06-07 are unsupported",
            )
        if trade_session < _T_PLUS_2_START:
            regime = HistoricalSettlementRegime.T_PLUS_3
        elif trade_session < _T_PLUS_1_START:
            regime = HistoricalSettlementRegime.T_PLUS_2
        else:
            regime = HistoricalSettlementRegime.T_PLUS_1

        settlement_session = trade_session
        for _ in range(regime.lag_sessions):
            previous = settlement_session
            try:
                settlement_session = (
                    self.settlement_calendar.next_settlement_session(previous)
                )
            except Exception as error:
                raise ExecutionCostValidationError(
                    "SETTLEMENT_CALENDAR_FAILURE",
                    "the settlement calendar could not resolve the next session",
                ) from error
            if type(settlement_session) is not date or settlement_session <= previous:
                raise ExecutionCostValidationError(
                    "INVALID_SETTLEMENT_CALENDAR_RESPONSE",
                    "the settlement calendar must return a later genuine date",
                )

        return SettlementResolution(
            trade_session=trade_session,
            regime=regime,
            lag_sessions=regime.lag_sessions,
            settlement_session=settlement_session,
        )


__all__ = ["HistoricalUsEquitySettlementResolver"]
