"""Ordinary cash-dividend event model."""

from datetime import date
import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PositiveFiniteAmount = Annotated[
    float, Field(gt=0, allow_inf_nan=False, strict=True)
]
PositiveAssetId = Annotated[int, Field(gt=0, strict=True)]


class DividendEvent(BaseModel):
    """An ordinary cash-dividend entitlement per share at entitlement close.

    The source value is associated with CAPITALSPECIAL price adjustment. This
    model records event data only: it does not credit portfolio cash, determine
    payment date or shares entitled, apply taxes, or reinvest dividends.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str
    symbol: str
    entitlement_date: date
    date_semantics: Literal["entitlement_close"]
    dividend_type: Literal["ordinary_cash"]
    amount_per_share: PositiveFiniteAmount
    currency: Literal["USD"]
    source_provider: Literal["Norgate Data"]
    source_asset_id: PositiveAssetId
    source_adjustment_mode: Literal["CAPITALSPECIAL"]

    @field_validator("entitlement_date", mode="before")
    @classmethod
    def validate_entitlement_date_input(cls, value: object) -> object:
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
        """Match the existing stock-model symbol normalization policy."""
        return value.upper()

    @model_validator(mode="after")
    def validate_asset_identity(self) -> Self:
        """Require the stable security identity to match its source AssetId."""
        expected_security_id = f"NORGATE:{self.source_asset_id}"
        if self.security_id != expected_security_id:
            raise ValueError(
                "security_id must match source_asset_id as "
                f"{expected_security_id}"
            )
        return self
