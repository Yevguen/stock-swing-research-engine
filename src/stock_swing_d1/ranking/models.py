"""Immutable public models for Phase 14 deterministic candidate ranking."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


CANDIDATE_RANKING_POLICY_ID = "candidate_ranking_policy_v0.1"
CANDIDATE_RANKING_POLICY_VERSION = "0.1"
CANDIDATE_RANKING_SNAPSHOT_SCHEMA_VERSION = (
    "candidate_ranking_snapshot_v0.1"
)


class CandidateRankingValidationError(ValueError):
    """A public Phase 14 ranking contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _require_session_date(value: object) -> object:
    if type(value) is not date:
        raise ValueError("must be a Python datetime.date")
    return value


def _require_aware_datetime(value: object) -> object:
    try:
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError("must be an explicitly timezone-aware datetime")
    except (OverflowError, TypeError) as error:
        raise ValueError(
            "must be an explicitly timezone-aware datetime"
        ) from error
    return value


class _ImmutableRankingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


_SessionDate = Annotated[date, BeforeValidator(_require_session_date)]
_AwareDateTime = Annotated[
    datetime,
    BeforeValidator(_require_aware_datetime),
]
_CanonicalSecurityId = Annotated[
    str,
    Field(pattern=r"^NORGATE:[1-9][0-9]*$"),
]
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_FiniteFloat = Annotated[
    float,
    Field(allow_inf_nan=False, strict=True),
]
_PositiveRank = Annotated[int, Field(gt=0, strict=True)]
_NonNegativeCount = Annotated[int, Field(ge=0, strict=True)]


def _validate_candidate_semantics(
    *,
    ranking_session: date,
    decision_time: datetime,
    signal_session: date,
    signal_time: datetime,
    sma20: float,
    sma50: float,
    atr14: float,
    rsi14: float,
) -> None:
    if signal_session != ranking_session:
        raise ValueError("signal_session must equal ranking_session")
    if signal_time > decision_time:
        raise ValueError("signal_time must not be later than decision_time")
    if atr14 <= 0.0:
        raise ValueError("atr14 must be greater than zero")
    if sma20 <= sma50:
        raise ValueError("sma20 must be greater than sma50")
    if rsi14 <= 50.0:
        raise ValueError("rsi14 must be greater than 50")


class RankingPolicyRef(_ImmutableRankingModel):
    """Identity and semantic fingerprint of the frozen Phase 14A policy."""

    policy_id: str
    policy_version: str
    policy_fingerprint: _Sha256


class RankingCandidate(_ImmutableRankingModel):
    """One already-eligible Phase 8 candidate presented for ranking."""

    security_id: _CanonicalSecurityId
    ranking_session: _SessionDate
    decision_time: _AwareDateTime
    signal_session: _SessionDate
    signal_time: _AwareDateTime
    sma20: _FiniteFloat
    sma50: _FiniteFloat
    atr14: _FiniteFloat
    rsi14: _FiniteFloat
    input_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_candidate_semantics(self) -> Self:
        _validate_candidate_semantics(
            ranking_session=self.ranking_session,
            decision_time=self.decision_time,
            signal_session=self.signal_session,
            signal_time=self.signal_time,
            sma20=self.sma20,
            sma50=self.sma50,
            atr14=self.atr14,
            rsi14=self.rsi14,
        )
        return self


class RankedCandidate(_ImmutableRankingModel):
    """One candidate with its complete-batch Phase 14 rank and metric."""

    security_id: _CanonicalSecurityId
    rank: _PositiveRank
    ranking_session: _SessionDate
    decision_time: _AwareDateTime
    signal_session: _SessionDate
    signal_time: _AwareDateTime
    sma20: _FiniteFloat
    sma50: _FiniteFloat
    atr14: _FiniteFloat
    rsi14: _FiniteFloat
    trend_separation_atr: _FiniteFloat
    input_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_candidate_semantics(self) -> Self:
        _validate_candidate_semantics(
            ranking_session=self.ranking_session,
            decision_time=self.decision_time,
            signal_session=self.signal_session,
            signal_time=self.signal_time,
            sma20=self.sma20,
            sma50=self.sma50,
            atr14=self.atr14,
            rsi14=self.rsi14,
        )
        return self


class CandidateRankingSnapshot(_ImmutableRankingModel):
    """Complete immutable result of one deterministic ranking cycle."""

    schema_version: str
    ranking_session: _SessionDate
    decision_time: _AwareDateTime
    policy: RankingPolicyRef
    candidate_count: _NonNegativeCount
    ranked_candidates: tuple[RankedCandidate, ...]
    input_set_fingerprint: _Sha256
    snapshot_fingerprint: _Sha256


__all__ = [
    "CANDIDATE_RANKING_POLICY_ID",
    "CANDIDATE_RANKING_POLICY_VERSION",
    "CANDIDATE_RANKING_SNAPSHOT_SCHEMA_VERSION",
    "CandidateRankingSnapshot",
    "CandidateRankingValidationError",
    "RankedCandidate",
    "RankingCandidate",
    "RankingPolicyRef",
]
