"""Explicit historical decision-boundary provenance for Phase 15A."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator


def _require_exact_date(value: object) -> object:
    if type(value) is not date:
        raise ValueError("must be an exact Python datetime.date")
    return value


_ExactDate = Annotated[date, BeforeValidator(_require_exact_date)]


class HistoricalDecisionInterval(BaseModel):
    """Immutable closed calendar interval for processed decision sessions."""

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    decision_start_date: _ExactDate
    decision_end_date: _ExactDate

    @model_validator(mode="after")
    def validate_closed_interval(self) -> Self:
        if self.decision_start_date > self.decision_end_date:
            raise ValueError(
                "decision_start_date must be on or before decision_end_date"
            )
        return self


__all__ = ["HistoricalDecisionInterval"]
