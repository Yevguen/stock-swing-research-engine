"""Narrow typed errors for the shared portfolio valuation primitive."""

from __future__ import annotations


class PortfolioValuationError(ValueError):
    """A shared allocation-boundary valuation contract was violated."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


__all__ = ["PortfolioValuationError"]
