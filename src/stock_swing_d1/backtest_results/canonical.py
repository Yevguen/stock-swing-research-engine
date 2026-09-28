"""Deterministic semantic serialization for Phase 15D audit content."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from math import isfinite
from typing import Any

from pydantic import BaseModel


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("canonical semantic Decimal values must be finite")
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _canonical_datetime(value: datetime) -> str:
    try:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "canonical semantic datetimes must be explicitly timezone-aware"
            )
    except (OverflowError, TypeError) as error:
        raise ValueError(
            "canonical semantic datetimes must be explicitly timezone-aware"
        ) from error
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical_mapping(value: Mapping[object, object]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str or not key or key != key.strip():
            raise TypeError(
                "canonical semantic mappings require canonical non-empty string keys"
            )
        result[key] = canonicalize_semantic_value(item)
    return {key: result[key] for key in sorted(result)}


def canonicalize_semantic_value(value: object) -> Any:
    """Convert supported semantic values to deterministic JSON-compatible data."""

    if isinstance(value, BaseModel):
        declared = type(value).model_fields
        return _canonical_mapping(
            {name: getattr(value, name) for name in declared}
        )
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_mapping(
            {
                field.name: getattr(value, field.name)
                for field in fields(value)
                if not field.name.startswith("_")
            }
        )
    if isinstance(value, Enum):
        return canonicalize_semantic_value(value.value)
    if isinstance(value, Decimal):
        return _canonical_decimal(value)
    if isinstance(value, datetime):
        return _canonical_datetime(value)
    if type(value) is date:
        return value.isoformat()
    if isinstance(value, Mapping):
        return _canonical_mapping(value)
    if type(value) in (tuple, list):
        return [canonicalize_semantic_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("set and frozenset are not canonical semantic values")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("canonical semantic floats must be finite")
        return value
    raise TypeError(
        f"unsupported canonical semantic value: {type(value).__name__}"
    )


def semantic_json_bytes(value: object) -> bytes:
    """Serialize semantic content as compact, deterministic UTF-8 JSON bytes."""

    return json.dumps(
        canonicalize_semantic_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


__all__ = ["canonicalize_semantic_value", "semantic_json_bytes"]
