"""Stable public errors for historical backtest result contracts."""

from __future__ import annotations


class HistoricalBacktestResultValidationError(ValueError):
    """A Phase 15D result or semantic-hashing contract was violated."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class HistoricalBacktestPersistenceError(ValueError):
    """A future Phase 15D persistence operation failed."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


__all__ = [
    "HistoricalBacktestPersistenceError",
    "HistoricalBacktestResultValidationError",
]
