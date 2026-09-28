"""The frozen shared valuation mark fact.

``PortfolioValuationMark`` is the mark model moved upstream by Task 5C-A so
that Phase 15A (runtime) and Phase 15D (audit) supply the *same* evidence type
to the *same* pure calculation.  The Phase 15D audit layer re-exports this
exact class object under its own historic name; the audit snapshot that
carries these marks, its builder and its hash domain remain Phase 15D-owned.

Fields and validators are preserved verbatim from the pre-move definition, so
every existing valuation snapshot fingerprint is unchanged.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field

from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.valuation.policy import _ImmutableValuationModel


def _require_canonical_text(value: object) -> object:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("must be canonical non-empty text")
    return value


def _require_session_date(value: object) -> object:
    if type(value) is not date:
        raise ValueError("must be a Python datetime.date")
    return value


def _require_decimal(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError(
            "must be a Decimal; binary floats, strings, and bool are forbidden"
        )
    if not value.is_finite():
        raise ValueError("must be a finite Decimal")
    return value


_CanonicalSecurityId = Annotated[
    str,
    BeforeValidator(_require_canonical_text),
    Field(pattern=r"^NORGATE:[1-9][0-9]*$"),
]
_SessionDate = Annotated[date, BeforeValidator(_require_session_date)]
_PositiveDecimal = Annotated[
    Decimal,
    BeforeValidator(_require_decimal),
    Field(gt=Decimal("0")),
]


class PortfolioValuationMark(_ImmutableValuationModel):
    security_id: _CanonicalSecurityId
    session: _SessionDate
    close: _PositiveDecimal
    currency: Literal["USD"] = "USD"
    timeframe: Literal["D1"] = "D1"
    session_type: Literal["regular"] = "regular"
    price_basis: Literal["unadjusted"] = "unadjusted"
    source_artifact_ref: ArtifactRef


__all__ = ["PortfolioValuationMark"]
