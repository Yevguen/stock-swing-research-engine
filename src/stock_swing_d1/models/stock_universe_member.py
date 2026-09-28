"""Stock-universe membership model."""

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class StockUniverseMember(BaseModel):
    """Membership of one security in a universe for an inclusive date interval.

    ``eligible_to`` is the last eligible calendar date and is inclusive. A value
    of ``None`` means that the membership interval is open-ended.
    """

    model_config = ConfigDict(extra="forbid")

    security_id: str
    symbol: str
    universe_kind: Literal["development", "historical_research"]
    universe_methodology_version: Literal["0.1"]
    market: Literal["United States"]
    security_type: Literal["common_stock"]
    eligible_from: date
    eligible_to: date | None
    source_name: str
    source_reference: str

    @field_validator(
        "security_id", "symbol", "source_name", "source_reference", mode="before"
    )
    @classmethod
    def normalize_required_text(cls, value: object) -> object:
        """Trim text fields and reject values that are empty after trimming."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize ticker symbols to uppercase without restricting their form."""
        return value.upper()

    @model_validator(mode="after")
    def validate_eligibility_interval(self) -> Self:
        """Ensure the inclusive eligibility interval does not end before it starts."""
        if self.eligible_to is not None and self.eligible_to < self.eligible_from:
            raise ValueError("eligible_to must be on or after eligible_from")
        return self
