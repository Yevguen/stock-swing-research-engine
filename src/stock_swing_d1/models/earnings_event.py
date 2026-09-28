"""Historical earnings-announcement event model."""

from datetime import date, datetime
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class EarningsEvent(BaseModel):
    """One historical earnings-announcement event for one security."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    security_id: str
    symbol: str
    earnings_date: date
    announcement_timing: Literal["BMO", "AMC", "during_session", "unknown"]
    source_name: str
    source_reference: str

    @field_validator("earnings_date", mode="before")
    @classmethod
    def validate_earnings_date_input(cls, value: object) -> object:
        """Accept date-only inputs while rejecting datetimes and timestamps."""
        if isinstance(value, (datetime, int, float)):
            raise ValueError("must be a date or ISO date string")
        if isinstance(value, str):
            if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
                raise ValueError("must be a date or ISO date string")
            try:
                date.fromisoformat(value)
            except ValueError as error:
                raise ValueError("must be a date or ISO date string") from error
        return value

    @field_validator(
        "event_id",
        "security_id",
        "symbol",
        "source_name",
        "source_reference",
        mode="before",
    )
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
