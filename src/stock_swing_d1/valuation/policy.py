"""The frozen completed-session unadjusted-close valuation policy.

Task 5C-A moved this policy *model* upstream so runtime (Phase 15A) and audit
(Phase 15D) share one frozen statement of valuation semantics.  Its ref, its
fingerprint builder and ``VALUATION_POLICY_HASH_DOMAIN`` deliberately stay
with the Phase 15D audit layer, which owns them as audit evidence.

Every field value below — including both identity strings — is preserved
verbatim from the pre-move definition, so the domain-separated policy
fingerprint is byte-identical: canonical serialization is structural over
``model_fields`` and carries no module path or class name.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


PORTFOLIO_VALUATION_POLICY_SCHEMA_VERSION = (
    "historical_backtest_valuation_policy.v0.1"
)
PORTFOLIO_VALUATION_POLICY_ID = "completed_session_unadjusted_close_mark_v0.1"


class _ImmutableValuationModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_default=True,
    )


class PortfolioValuationPolicy(_ImmutableValuationModel):
    """Frozen completed-session unadjusted-close valuation semantics."""

    schema_version: Literal[
        "historical_backtest_valuation_policy.v0.1"
    ] = PORTFOLIO_VALUATION_POLICY_SCHEMA_VERSION
    policy_id: Literal[
        "completed_session_unadjusted_close_mark_v0.1"
    ] = PORTFOLIO_VALUATION_POLICY_ID
    currency: Literal["USD"] = "USD"
    timeframe: Literal["D1"] = "D1"
    session_type: Literal["regular"] = "regular"
    price_basis: Literal["unadjusted"] = "unadjusted"
    mark_field: Literal["close"] = "close"
    pending_settlement_valuation: Literal["face_value"] = "face_value"
    open_position_valuation: Literal[
        "quantity_times_close"
    ] = "quantity_times_close"


__all__ = [
    "PORTFOLIO_VALUATION_POLICY_ID",
    "PORTFOLIO_VALUATION_POLICY_SCHEMA_VERSION",
    "PortfolioValuationPolicy",
]
