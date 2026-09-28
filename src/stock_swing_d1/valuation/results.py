"""The runtime-only result of one allocation-boundary valuation.

Task 5C-A froze this as runtime state: it has no schema version, no
fingerprint, and is never persisted.  All four components are authoritative
``Decimal``; the one-way projection to ``float`` happens exactly once, in
Phase 15A, at the ``PortfolioSnapshot`` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class PortfolioValuationResult:
    """Exact Decimal components of allocation-boundary portfolio equity."""

    settled_cash: Decimal
    pending_receivable_value: Decimal
    open_position_market_value: Decimal
    portfolio_equity: Decimal


__all__ = ["PortfolioValuationResult"]
