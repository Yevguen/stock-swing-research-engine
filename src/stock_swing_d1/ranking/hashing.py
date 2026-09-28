"""Canonical serialization and SHA-256 helpers for Phase 14 ranking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date, datetime, timezone
from math import isfinite
from typing import Any

from pydantic import BaseModel

from stock_swing_d1.ranking.models import (
    CANDIDATE_RANKING_POLICY_ID,
    CANDIDATE_RANKING_POLICY_VERSION,
    CandidateRankingSnapshot,
    RankedCandidate,
    RankingCandidate,
)


def _canonical_utc_timestamp(value: datetime) -> str:
    try:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical ranking timestamps must be timezone-aware")
    except (OverflowError, TypeError) as error:
        raise ValueError(
            "canonical ranking timestamps must be timezone-aware"
        ) from error
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_value(value: object) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    if isinstance(value, datetime):
        return _canonical_utc_timestamp(value)
    if type(value) is date:
        return value.isoformat()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical ranking mappings require string keys")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("canonical ranking floats must be finite")
        return value.hex()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical ranking value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize supported values as deterministic canonical UTF-8 JSON."""

    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Return lowercase SHA-256 for one canonical Phase 14 payload."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def ranking_policy_manifest() -> dict[str, object]:
    """Return the semantic manifest of frozen Candidate Ranking Policy v0.1."""

    return {
        "policy_id": CANDIDATE_RANKING_POLICY_ID,
        "policy_version": CANDIDATE_RANKING_POLICY_VERSION,
        "primary_metric": "(SMA20 - SMA50) / ATR14",
        "primary_direction": "descending",
        "secondary_metric": "RSI14",
        "secondary_direction": "descending",
        "final_tiebreak": "security_id ascending",
        "mandatory_inputs": ["SMA20", "SMA50", "ATR14", "RSI14"],
        "missing_value_policy": "reject",
        "non_finite_policy": "reject",
        "ATR14_nonpositive_policy": "reject",
        "round_before_compare": False,
        "weighted_score": False,
        "stochastic_behavior": False,
    }


def compute_policy_fingerprint() -> str:
    """Compute the semantic SHA-256 of the frozen Phase 14A policy."""

    return canonical_sha256(ranking_policy_manifest())


CANDIDATE_RANKING_POLICY_FINGERPRINT = compute_policy_fingerprint()


def compute_candidate_input_fingerprint(
    *,
    security_id: str,
    ranking_session: date,
    decision_time: datetime,
    signal_session: date,
    signal_time: datetime,
    sma20: float,
    sma50: float,
    atr14: float,
    rsi14: float,
) -> str:
    """Hash exactly the semantic inputs of one ranking candidate."""

    numeric_inputs = {
        "sma20": sma20,
        "sma50": sma50,
        "atr14": atr14,
        "rsi14": rsi14,
    }
    normalized_numbers: dict[str, float] = {}
    for field_name, value in numeric_inputs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be a binary64-compatible number")
        normalized = float(value)
        if not isfinite(normalized):
            raise ValueError(f"{field_name} must be finite")
        normalized_numbers[field_name] = normalized

    return canonical_sha256(
        {
            "security_id": security_id,
            "ranking_session": ranking_session,
            "decision_time": decision_time,
            "signal_session": signal_session,
            "signal_time": signal_time,
            **normalized_numbers,
        }
    )


def candidate_fingerprint(candidate: RankingCandidate | RankedCandidate) -> str:
    """Recompute one model's candidate input fingerprint."""

    if not isinstance(candidate, (RankingCandidate, RankedCandidate)):
        raise TypeError("candidate must be a RankingCandidate or RankedCandidate")
    return compute_candidate_input_fingerprint(
        security_id=candidate.security_id,
        ranking_session=candidate.ranking_session,
        decision_time=candidate.decision_time,
        signal_session=candidate.signal_session,
        signal_time=candidate.signal_time,
        sma20=candidate.sma20,
        sma50=candidate.sma50,
        atr14=candidate.atr14,
        rsi14=candidate.rsi14,
    )


def compute_input_set_fingerprint(
    candidates: Sequence[RankingCandidate | RankedCandidate],
) -> str:
    """Hash candidate fingerprints in canonical security-ID order."""

    supplied = tuple(candidates)
    if any(
        not isinstance(candidate, (RankingCandidate, RankedCandidate))
        for candidate in supplied
    ):
        raise TypeError(
            "candidates must contain RankingCandidate or RankedCandidate values"
        )
    security_ids = tuple(candidate.security_id for candidate in supplied)
    if len(set(security_ids)) != len(security_ids):
        raise ValueError("candidate security_ids must be unique")
    ordered_fingerprints = [
        candidate.input_fingerprint
        for candidate in sorted(supplied, key=lambda item: item.security_id)
    ]
    return canonical_sha256(
        {"candidate_input_fingerprints": ordered_fingerprints}
    )


def compute_snapshot_fingerprint(snapshot: CandidateRankingSnapshot) -> str:
    """Hash all semantic snapshot identity and ordered-ranking fields."""

    if not isinstance(snapshot, CandidateRankingSnapshot):
        raise TypeError("snapshot must be a CandidateRankingSnapshot")
    return canonical_sha256(
        {
            "schema_version": snapshot.schema_version,
            "ranking_session": snapshot.ranking_session,
            "decision_time": snapshot.decision_time,
            "policy_id": snapshot.policy.policy_id,
            "policy_version": snapshot.policy.policy_version,
            "policy_fingerprint": snapshot.policy.policy_fingerprint,
            "candidate_count": snapshot.candidate_count,
            "input_set_fingerprint": snapshot.input_set_fingerprint,
            "ranked_candidates": [
                {
                    "security_id": candidate.security_id,
                    "rank": candidate.rank,
                    "input_fingerprint": candidate.input_fingerprint,
                    "trend_separation_atr": candidate.trend_separation_atr,
                    "rsi14": candidate.rsi14,
                }
                for candidate in snapshot.ranked_candidates
            ],
        }
    )


__all__ = [
    "CANDIDATE_RANKING_POLICY_FINGERPRINT",
    "candidate_fingerprint",
    "canonical_json_bytes",
    "canonical_sha256",
    "compute_candidate_input_fingerprint",
    "compute_input_set_fingerprint",
    "compute_policy_fingerprint",
    "compute_snapshot_fingerprint",
    "ranking_policy_manifest",
]
