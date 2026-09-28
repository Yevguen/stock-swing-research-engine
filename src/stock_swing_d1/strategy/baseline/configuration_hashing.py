"""Canonical semantic hashing for the frozen baseline strategy configuration.

The ``strategy`` package sits *above* Phase 15D in the frozen
Phase 13 -> 15A -> 15D -> 16B -> 16C dependency chain, so it may not import the
Phase 15D canonical serializer; a scope guard enforces that direction. This
module therefore follows the package-local convention already established by
``ranking/hashing.py``, ``portfolio/portfolio_hashing.py`` and
``execution/costs/hashing.py``: each upstream owner canonicalizes and hashes
its own semantic content.

Binary floats are encoded with ``float.hex()``, exactly as ``ranking/hashing.py``
does. The baseline strategy's authoritative thresholds are genuine binary
floats, and hex round-trips them losslessly, so the fingerprint binds the exact
value the evaluator compares against rather than a shortened decimal rendering
that two different binary values could share.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from math import isfinite
from typing import Any

from pydantic import BaseModel

from stock_swing_d1.strategy.baseline.models import (
    BaselineStrategyValidationError,
)


def _canonical_value(value: object) -> Any:
    if isinstance(value, BaseModel):
        declared = type(value).model_fields
        return {
            name: _canonical_value(getattr(value, name)) for name in declared
        }
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if type(value) is float:
        if not isfinite(value):
            raise BaselineStrategyValidationError(
                "NON_FINITE_CONFIGURATION_VALUE",
                "canonical strategy configuration floats must be finite",
            )
        return value.hex()
    if type(value) is dict:
        if any(
            type(key) is not str or not key or key != key.strip()
            for key in value
        ):
            raise BaselineStrategyValidationError(
                "INVALID_CANONICAL_CONFIGURATION",
                "canonical strategy mappings require canonical string keys",
            )
        return {key: _canonical_value(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_canonical_value(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    raise BaselineStrategyValidationError(
        "INVALID_CANONICAL_CONFIGURATION",
        f"unsupported canonical strategy value: {type(value).__name__}",
    )


def canonical_configuration_json_bytes(value: object) -> bytes:
    """Serialize semantic content as compact, deterministic UTF-8 JSON bytes."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def baseline_strategy_domain_sha256(domain: str, payload: object) -> str:
    """Hash semantic content within an explicit, versioned domain."""

    if type(domain) is not str or not domain or domain != domain.strip():
        raise BaselineStrategyValidationError(
            "INVALID_HASH_DOMAIN", "domain must be canonical non-empty text"
        )
    return hashlib.sha256(
        canonical_configuration_json_bytes(
            {"domain": domain, "payload": payload}
        )
    ).hexdigest()


__all__ = [
    "baseline_strategy_domain_sha256",
    "canonical_configuration_json_bytes",
]
