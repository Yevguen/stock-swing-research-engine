"""Canonical raw, unadjusted daily stock-bar model."""

from datetime import date, datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PositiveFinitePrice = Annotated[
    float, Field(gt=0, allow_inf_nan=False, strict=True)
]
NonNegativeVolume = Annotated[int, Field(ge=0, strict=True)]


class StockBar(BaseModel):
    """One completed regular-session U.S. D1 OHLCV stock bar.

    Future dataset-level uniqueness is conceptually based on ``security_id``,
    ``trading_date``, ``timeframe``, ``session_type``, and ``price_basis``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str
    symbol: str
    trading_date: date
    timeframe: Literal["D1"]
    session_type: Literal["regular"]
    currency: Literal["USD"]
    price_basis: Literal["unadjusted"]
    open: PositiveFinitePrice
    high: PositiveFinitePrice
    low: PositiveFinitePrice
    close: PositiveFinitePrice
    volume: NonNegativeVolume

    @field_validator("trading_date", mode="before")
    @classmethod
    def validate_trading_date_input(cls, value: object) -> object:
        """Accept date-only inputs while rejecting datetimes and timestamps."""
        if isinstance(value, (datetime, int, float)):
            raise ValueError("must be a date or ISO date string")
        if isinstance(value, str):
            try:
                date.fromisoformat(value)
            except ValueError as error:
                raise ValueError("must be a date or ISO date string") from error
        return value

    @field_validator("security_id", "symbol", mode="before")
    @classmethod
    def normalize_required_text(cls, value: object) -> object:
        """Trim required text and reject values empty after trimming."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize ticker symbols without restricting their form."""
        return value.upper()

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> Self:
        """Reject prices that cannot form a structurally valid OHLC bar."""
        if self.low > self.high:
            raise ValueError("low must be less than or equal to high")
        if not self.low <= self.open <= self.high:
            raise ValueError("open must be between low and high")
        if not self.low <= self.close <= self.high:
            raise ValueError("close must be between low and high")
        return self
