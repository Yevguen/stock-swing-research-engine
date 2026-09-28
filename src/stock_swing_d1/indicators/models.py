"""Immutable output model for the Phase 7A indicator methodology."""

from datetime import date
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


FiniteIndicator = Annotated[float, Field(allow_inf_nan=False, strict=True)]
NonNegativeFiniteIndicator = Annotated[
    float, Field(ge=0.0, allow_inf_nan=False, strict=True)
]
BoundedRsi = Annotated[
    float, Field(ge=0.0, le=100.0, allow_inf_nan=False, strict=True)
]


class TechnicalIndicatorRow(BaseModel):
    """Indicators known after one completed adjusted D1 bar.

    ``None`` represents an indicator whose exact full-window warm-up has not
    completed.  The model intentionally carries only row identity, price
    basis, and the frozen Phase 7A v0.1 indicator values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str
    symbol: str
    trading_date: date
    price_basis: Literal["capital_special_adjusted"]
    sma_20: FiniteIndicator | None
    sma_50: FiniteIndicator | None
    sma_100: FiniteIndicator | None
    sma_200: FiniteIndicator | None
    rsi_14: BoundedRsi | None
    atr_14: NonNegativeFiniteIndicator | None
    avg_volume_20: NonNegativeFiniteIndicator | None
    relative_volume_20: NonNegativeFiniteIndicator | None

    @field_validator("trading_date", mode="before")
    @classmethod
    def validate_trading_date_input(cls, value: object) -> object:
        """Preserve the adjusted-bar contract's strict date semantics."""
        if type(value) is not date:
            raise ValueError("must be a Python datetime.date")
        return value

    @field_validator("security_id")
    @classmethod
    def validate_security_id(cls, value: str) -> str:
        """Preserve the adjusted-bar stable-identity contract."""
        if re.fullmatch(r"NORGATE:[1-9][0-9]*", value) is None:
            raise ValueError("must have the form NORGATE:<positive AssetId>")
        return value


__all__ = ["TechnicalIndicatorRow"]
