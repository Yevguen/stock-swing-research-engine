"""Corporate-action event model."""

from datetime import date
import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


PositiveFiniteRatio = Annotated[
    float, Field(gt=0, allow_inf_nan=False, strict=True)
]
PositiveAssetId = Annotated[int, Field(gt=0, strict=True)]


class CorporateActionEvent(BaseModel):
    """One immutable corporate-action event associated with a Norgate asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str
    symbol: str
    event_date: date
    date_semantics: Literal[
        "entitlement_close", "ex_date", "effective_date", "unknown"
    ]
    event_type: Literal[
        "unknown_capital_event",
        "split",
        "reverse_split",
        "stock_dividend",
        "rights_issue",
        "special_distribution",
        "spinoff",
        "merger_or_reorganization",
        "other",
    ]
    terms_verified: StrictBool
    new_shares: PositiveFiniteRatio | None
    old_shares: PositiveFiniteRatio | None
    source_provider: Literal["Norgate Data"]
    source_asset_id: PositiveAssetId

    @field_validator("event_date", mode="before")
    @classmethod
    def validate_event_date_input(cls, value: object) -> object:
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
        """Match the existing model symbol normalization policy."""
        return value.upper()

    @model_validator(mode="after")
    def validate_identity_and_terms(self) -> Self:
        """Enforce asset identity and the frozen v0.1 event-term rules."""
        expected_security_id = f"NORGATE:{self.source_asset_id}"
        if self.security_id != expected_security_id:
            raise ValueError(
                "security_id must match source_asset_id as "
                f"{expected_security_id}"
            )

        ratio_present = self.new_shares is not None
        if ratio_present != (self.old_shares is not None):
            raise ValueError("new_shares and old_shares must both be None or present")

        if self.event_type == "unknown_capital_event":
            if self.terms_verified:
                raise ValueError("unknown_capital_event terms cannot be verified")
            if ratio_present:
                raise ValueError("unknown_capital_event cannot have share-ratio terms")

        if self.event_type in {"split", "reverse_split"}:
            if ratio_present and not self.terms_verified:
                raise ValueError("supplied split-ratio terms must be verified")
            if self.terms_verified and not ratio_present:
                raise ValueError("verified split events require share-ratio terms")

        if self.event_type == "split" and self.terms_verified:
            assert self.new_shares is not None and self.old_shares is not None
            if self.new_shares <= self.old_shares:
                raise ValueError("verified split must have new_shares > old_shares")

        if self.event_type == "reverse_split" and self.terms_verified:
            assert self.new_shares is not None and self.old_shares is not None
            if self.new_shares >= self.old_shares:
                raise ValueError(
                    "verified reverse_split must have new_shares < old_shares"
                )

        return self
