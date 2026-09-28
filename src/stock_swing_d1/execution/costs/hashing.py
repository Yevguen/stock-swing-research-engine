"""Canonical semantic hashing for Phase 15C execution-cost policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from stock_swing_d1.execution.costs.models import (
    BacktestExecutionCostPolicy,
    ExecutionCostValidationError,
)


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ExecutionCostValidationError(
            "NON_FINITE_POLICY_DECIMAL",
            "policy fingerprint inputs must contain only finite Decimal values",
        )
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _canonical_value(value: object) -> Any:
    if isinstance(value, Decimal):
        return _canonical_decimal(value)
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ExecutionCostValidationError(
                "INVALID_CANONICAL_POLICY",
                "canonical policy mappings require string keys",
            )
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, float):
        raise ExecutionCostValidationError(
            "BINARY_FLOAT_IN_POLICY",
            "binary floats are not canonical execution-cost policy values",
        )
    raise ExecutionCostValidationError(
        "INVALID_CANONICAL_POLICY",
        f"unsupported canonical value type: {type(value).__name__}",
    )


def _canonical_policy_bytes(policy: BacktestExecutionCostPolicy) -> bytes:
    if type(policy) is not BacktestExecutionCostPolicy:
        raise ExecutionCostValidationError(
            "INVALID_POLICY", "policy must be a BacktestExecutionCostPolicy"
        )
    return json.dumps(
        _canonical_value(policy),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def compute_execution_cost_policy_fingerprint(
    policy: BacktestExecutionCostPolicy,
) -> str:
    """Return lowercase SHA-256 over the policy's semantic canonical JSON."""

    return hashlib.sha256(_canonical_policy_bytes(policy)).hexdigest()


__all__ = ["compute_execution_cost_policy_fingerprint"]
