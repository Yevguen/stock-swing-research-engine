"""Phase 4 raw/unadjusted to Phase 5 adjusted parity validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import pandas as pd

from stock_swing_d1.data.norgate_d1_adapter import security_id_for_asset_id
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    CorporateActionEvent,
    StockBar,
)


ObservationKey = tuple[str, date]


@dataclass(frozen=True, slots=True)
class CorporateActionParityResult:
    """Complete parity counts and adjustment-factor diagnostics."""

    raw_adjusted_missing_key_count: int
    raw_adjusted_extra_key_count: int
    unadjusted_close_mismatch_count: int
    minimum_adjustment_factor: float | None
    maximum_adjustment_factor: float | None
    errors: tuple[dict[str, object], ...]

    @property
    def valid(self) -> bool:
        return not self.errors


class CorporateActionParityError(ValueError):
    """One or more Phase 4/Phase 5 parity requirements failed."""

    def __init__(self, result: CorporateActionParityResult) -> None:
        self.result = result
        super().__init__(
            "; ".join(str(error["reason"]) for error in result.errors)
            or "corporate-action parity validation failed"
        )


class EventPhase4CrosscheckError(ValueError):
    """At least one capital event has no same-date Phase 4 observation."""

    def __init__(self, errors: Sequence[dict[str, object]]) -> None:
        self.errors = tuple(errors)
        super().__init__(
            "; ".join(str(error["reason"]) for error in self.errors)
        )


def _key(row: StockBar | CorporateActionAdjustedStockBar) -> ObservationKey:
    return row.security_id, row.trading_date


def _key_error(reason: str, key: ObservationKey) -> dict[str, object]:
    return {
        "security_id": key[0],
        "trading_date": key[1].isoformat(),
        "stage": "raw_adjusted_parity",
        "reason": reason,
    }


def check_corporate_action_parity(
    phase4_bars: Sequence[StockBar],
    adjusted_bars: Sequence[CorporateActionAdjustedStockBar],
    adjusted_raw_frame: pd.DataFrame,
) -> CorporateActionParityResult:
    """Collect all exact-key, metadata, close, and factor parity failures.

    Price and volume differences are intentionally not compared, and this
    function performs no corporate-event inference.
    """
    errors: list[dict[str, object]] = []
    phase4_by_key: dict[ObservationKey, StockBar] = {}
    for bar in phase4_bars:
        key = _key(bar)
        if key in phase4_by_key:
            errors.append(_key_error("duplicate Phase 4 observation key", key))
        phase4_by_key[key] = bar

    adjusted_by_key: dict[ObservationKey, CorporateActionAdjustedStockBar] = {}
    for bar in adjusted_bars:
        key = _key(bar)
        if key in adjusted_by_key:
            errors.append(_key_error("duplicate Phase 5 adjusted observation key", key))
        adjusted_by_key[key] = bar

    phase4_keys = set(phase4_by_key)
    adjusted_keys = set(adjusted_by_key)
    missing = sorted(phase4_keys - adjusted_keys)
    extra = sorted(adjusted_keys - phase4_keys)
    errors.extend(
        _key_error("Phase 4 observation is missing from adjusted data", key)
        for key in missing
    )
    errors.extend(
        _key_error("adjusted observation has no Phase 4 source observation", key)
        for key in extra
    )

    metadata_fields = (
        "security_id",
        "symbol",
        "trading_date",
        "timeframe",
        "session_type",
        "currency",
    )
    factors: list[float] = []
    for key in sorted(phase4_keys & adjusted_keys):
        source = phase4_by_key[key]
        adjusted = adjusted_by_key[key]
        for field in metadata_fields:
            if getattr(source, field) != getattr(adjusted, field):
                error = _key_error(f"metadata mismatch for {field}", key)
                error["phase4_value"] = str(getattr(source, field))
                error["adjusted_value"] = str(getattr(adjusted, field))
                errors.append(error)
        if source.price_basis != "unadjusted":
            errors.append(_key_error("Phase 4 price_basis must be unadjusted", key))
        if adjusted.price_basis != "capital_special_adjusted":
            errors.append(
                _key_error(
                    "Phase 5 price_basis must be capital_special_adjusted", key
                )
            )
        factor = adjusted.close / source.close
        if not math.isfinite(factor) or factor <= 0:
            errors.append(
                _key_error("price adjustment factor must be finite and positive", key)
            )
        else:
            factors.append(factor)

    raw_by_key: dict[ObservationKey, object] = {}
    required_raw_columns = {
        "provider_asset_id",
        "trading_date",
        "Unadjusted Close",
    }
    if not isinstance(adjusted_raw_frame, pd.DataFrame):
        errors.append(
            {
                "stage": "raw_adjusted_parity",
                "reason": "adjusted raw data must be a pandas DataFrame",
            }
        )
    elif not required_raw_columns.issubset(adjusted_raw_frame.columns):
        missing_columns = sorted(required_raw_columns - set(adjusted_raw_frame.columns))
        errors.append(
            {
                "stage": "raw_adjusted_parity",
                "reason": "adjusted raw data is missing columns: "
                + ", ".join(missing_columns),
            }
        )
    else:
        for record in adjusted_raw_frame.to_dict(orient="records"):
            try:
                key = (
                    security_id_for_asset_id(record["provider_asset_id"]),
                    record["trading_date"],
                )
            except (TypeError, ValueError) as error:
                errors.append(
                    {
                        "stage": "raw_adjusted_parity",
                        "reason": f"invalid raw adjusted observation identity: {error}",
                    }
                )
                continue
            if type(key[1]) is not date:
                errors.append(
                    {
                        "security_id": key[0],
                        "stage": "raw_adjusted_parity",
                        "reason": "raw adjusted trading_date must be a Python date",
                    }
                )
                continue
            if key in raw_by_key:
                errors.append(_key_error("duplicate raw adjusted observation key", key))
            raw_by_key[key] = record["Unadjusted Close"]

    mismatch_count = 0
    for key in sorted(phase4_keys & set(raw_by_key)):
        raw_value = raw_by_key[key]
        if isinstance(raw_value, (bool, str, bytes)) or raw_value is None:
            mismatch = True
        else:
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError, OverflowError):
                mismatch = True
            else:
                mismatch = not math.isfinite(numeric) or not math.isclose(
                    numeric,
                    phase4_by_key[key].close,
                    rel_tol=1e-9,
                    abs_tol=1e-8,
                )
        if mismatch:
            mismatch_count += 1
            errors.append(
                _key_error(
                    "provider Unadjusted Close does not match Phase 4 close", key
                )
            )

    return CorporateActionParityResult(
        raw_adjusted_missing_key_count=len(missing),
        raw_adjusted_extra_key_count=len(extra),
        unadjusted_close_mismatch_count=mismatch_count,
        minimum_adjustment_factor=min(factors) if factors else None,
        maximum_adjustment_factor=max(factors) if factors else None,
        errors=tuple(errors),
    )


def validate_corporate_action_parity(
    phase4_bars: Sequence[StockBar],
    adjusted_bars: Sequence[CorporateActionAdjustedStockBar],
    adjusted_raw_frame: pd.DataFrame,
) -> CorporateActionParityResult:
    """Return diagnostics on success and raise with all diagnostics on failure."""
    result = check_corporate_action_parity(
        phase4_bars, adjusted_bars, adjusted_raw_frame
    )
    if not result.valid:
        raise CorporateActionParityError(result)
    return result


def validate_events_against_phase4(
    events: Sequence[CorporateActionEvent], phase4_bars: Sequence[StockBar]
) -> None:
    """Require every provider flag event to land on an existing Phase 4 bar."""
    phase4_keys = {_key(bar) for bar in phase4_bars}
    errors: list[dict[str, object]] = []
    for event in events:
        key = (event.security_id, event.event_date)
        if key not in phase4_keys:
            errors.append(
                {
                    "provider_asset_id": event.source_asset_id,
                    "security_id": event.security_id,
                    "event_date": event.event_date.isoformat(),
                    "stage": "event_phase4_crosscheck",
                    "reason": "capital event has no same-date Phase 4 bar",
                }
            )
    if errors:
        raise EventPhase4CrosscheckError(errors)


__all__ = [
    "CorporateActionParityError",
    "CorporateActionParityResult",
    "EventPhase4CrosscheckError",
    "check_corporate_action_parity",
    "validate_corporate_action_parity",
    "validate_events_against_phase4",
]
