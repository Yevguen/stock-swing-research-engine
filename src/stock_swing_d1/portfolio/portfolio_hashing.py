"""Canonical serialization and SHA-256 helpers for Phase 13 portfolio data."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from stock_swing_d1.portfolio.portfolio_events import PortfolioExecutionEvent
    from stock_swing_d1.portfolio.portfolio_state_models import (
        PendingSettlement,
        PortfolioState,
    )


def _canonical_decimal(value: Decimal) -> str:
    """Return one exponent-free representation for a finite decimal value."""

    if not value.is_finite():
        raise ValueError("canonical portfolio decimals must be finite")
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _canonical_value(value: object) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    if isinstance(value, Decimal):
        return _canonical_decimal(value)
    if type(value) is date:
        return value.isoformat()
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical portfolio mappings require string keys")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        raise TypeError("binary floats are not canonical portfolio values")
    raise TypeError(f"unsupported canonical portfolio value: {type(value).__name__}")


def canonical_payload_bytes(value: object) -> bytes:
    """Serialize supported portfolio data into deterministic UTF-8 JSON bytes."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_payload_sha256(value: object) -> str:
    """Hash a canonical portfolio payload with lowercase SHA-256."""

    return hashlib.sha256(canonical_payload_bytes(value)).hexdigest()


def hash_portfolio_state(state: PortfolioState) -> str:
    """Hash the complete canonical Phase 13 portfolio state."""

    return canonical_payload_sha256(state)


def hash_execution_event(event: PortfolioExecutionEvent) -> str:
    """Hash the complete canonical external execution payload."""

    return canonical_payload_sha256(event)


def hash_settlement_event(settlement: PendingSettlement) -> str:
    """Hash the complete internally generated settlement payload."""

    return canonical_payload_sha256(settlement)


__all__ = [
    "canonical_payload_bytes",
    "canonical_payload_sha256",
    "hash_execution_event",
    "hash_portfolio_state",
    "hash_settlement_event",
]
