"""Corporate-action-adjusted daily stock-bar model."""

from datetime import date
import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PositiveFinitePrice = Annotated[
    float, Field(gt=0, allow_inf_nan=False, strict=True)
]
NonNegativeFiniteAdjustedVolume = Annotated[
    float, Field(ge=0, allow_inf_nan=False, strict=True)
]


class CorporateActionAdjustedStockBar(BaseModel):
    """A derived analytical D1 bar adjusted for capital events/special distributions.

    This is not an actual historically executable price record. ``StockBar``
    remains the immutable unadjusted execution and audit record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str
    symbol: str
    trading_date: date
    timeframe: Literal["D1"]
    session_type: Literal["regular"]
    currency: Literal["USD"]
    price_basis: Literal["capital_special_adjusted"]
    open: PositiveFinitePrice
    high: PositiveFinitePrice
    low: PositiveFinitePrice
    close: PositiveFinitePrice
    volume: NonNegativeFiniteAdjustedVolume

    @field_validator("trading_date", mode="before")
    @classmethod
    def validate_trading_date_input(cls, value: object) -> object:
        """Require an actual Python date without implicit conversion."""
        if type(value) is not date:
            raise ValueError("must be a Python datetime.date")
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

    @field_validator("security_id")
    @classmethod
    def validate_security_id(cls, value: str) -> str:
        """Require a Norgate identity containing a positive AssetId."""
        if re.fullmatch(r"NORGATE:[1-9][0-9]*", value) is None:
            raise ValueError("must have the form NORGATE:<positive AssetId>")
        return value

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Match the StockBar symbol normalization policy."""
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
