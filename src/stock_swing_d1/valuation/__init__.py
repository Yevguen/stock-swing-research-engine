"""Shared allocation-boundary portfolio valuation primitives (Task 5C-A).

This package owns only what runtime (Phase 15A) and audit (Phase 15D) must
share: the frozen valuation policy, the shared valuation mark, the single pure
equity calculation, its runtime result, and narrow typed errors.

It deliberately owns no fingerprint, no hash domain, no snapshot and no
generic ``ArtifactRef``; those remain with ``provenance`` (generic provenance)
and with the Phase 15D audit layer (audit evidence).  It must never import
``backtester``, the Phase 15D audit package, ``baseline_experiment`` or any
data/provider package.
"""

from stock_swing_d1.valuation.errors import PortfolioValuationError
from stock_swing_d1.valuation.models import PortfolioValuationMark
from stock_swing_d1.valuation.policy import (
    PORTFOLIO_VALUATION_POLICY_ID,
    PORTFOLIO_VALUATION_POLICY_SCHEMA_VERSION,
    PortfolioValuationPolicy,
)
from stock_swing_d1.valuation.results import PortfolioValuationResult
from stock_swing_d1.valuation.service import (
    value_portfolio_at_allocation_boundary,
)

__all__ = [
    "PORTFOLIO_VALUATION_POLICY_ID",
    "PORTFOLIO_VALUATION_POLICY_SCHEMA_VERSION",
    "PortfolioValuationError",
    "PortfolioValuationMark",
    "PortfolioValuationPolicy",
    "PortfolioValuationResult",
    "value_portfolio_at_allocation_boundary",
]
