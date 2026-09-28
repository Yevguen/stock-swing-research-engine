"""Pure ranking computation and fail-closed Phase 14 snapshot validation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from math import isfinite

from pydantic import ValidationError

from stock_swing_d1.ranking.hashing import (
    CANDIDATE_RANKING_POLICY_FINGERPRINT,
    candidate_fingerprint,
    compute_input_set_fingerprint,
    compute_policy_fingerprint,
    compute_snapshot_fingerprint,
)
from stock_swing_d1.ranking.models import (
    CANDIDATE_RANKING_POLICY_ID,
    CANDIDATE_RANKING_POLICY_VERSION,
    CANDIDATE_RANKING_SNAPSHOT_SCHEMA_VERSION,
    CandidateRankingSnapshot,
    CandidateRankingValidationError,
    RankedCandidate,
    RankingCandidate,
    RankingPolicyRef,
)


def _is_aware(value: object) -> bool:
    try:
        return (
            isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        )
    except (OverflowError, TypeError):
        return False


def _validate_cycle_identity(
    *, ranking_session: object, decision_time: object
) -> tuple[date, datetime]:
    if type(ranking_session) is not date:
        raise CandidateRankingValidationError(
            "INVALID_RANKING_SESSION",
            "ranking_session must be a genuine date",
        )
    if not _is_aware(decision_time):
        raise CandidateRankingValidationError(
            "INVALID_DECISION_TIME",
            "decision_time must be explicitly timezone-aware",
        )
    assert isinstance(decision_time, datetime)
    return ranking_session, decision_time


def _revalidate_candidate(candidate: object) -> RankingCandidate:
    if not isinstance(candidate, RankingCandidate):
        raise CandidateRankingValidationError(
            "INVALID_CANDIDATE_BATCH",
            "candidates must contain only RankingCandidate values",
        )
    try:
        rebuilt = RankingCandidate.model_validate(
            candidate.model_dump(mode="python")
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise CandidateRankingValidationError(
            "INVALID_CANDIDATE",
            "candidate fails structural or semantic validation",
        ) from error
    if rebuilt != candidate:
        raise CandidateRankingValidationError(
            "INVALID_CANDIDATE",
            "candidate must already be canonical",
        )
    return candidate


def _validate_candidate_for_cycle(
    candidate: RankingCandidate,
    *,
    ranking_session: date,
    decision_time: datetime,
) -> None:
    if candidate.ranking_session != ranking_session:
        raise CandidateRankingValidationError(
            "CANDIDATE_SESSION_MISMATCH",
            "every candidate must belong to ranking_session",
        )
    if candidate.decision_time != decision_time:
        raise CandidateRankingValidationError(
            "CANDIDATE_DECISION_TIME_MISMATCH",
            "every candidate must preserve the ranking decision_time",
        )
    expected_fingerprint = candidate_fingerprint(candidate)
    if candidate.input_fingerprint != expected_fingerprint:
        raise CandidateRankingValidationError(
            "CANDIDATE_FINGERPRINT_MISMATCH",
            "candidate input_fingerprint does not match its semantic inputs",
        )


def _trend_separation_atr(candidate: RankingCandidate | RankedCandidate) -> float:
    metric = (candidate.sma20 - candidate.sma50) / candidate.atr14
    if not isfinite(metric):
        raise CandidateRankingValidationError(
            "INVALID_DERIVED_METRIC",
            "trend_separation_atr must be finite for finite candidate inputs",
        )
    return metric


def rank_candidates(
    *,
    ranking_session: date,
    decision_time: datetime,
    candidates: Sequence[RankingCandidate],
) -> CandidateRankingSnapshot:
    """Rank one complete eligible batch without truncation or side effects."""

    ranking_session, decision_time = _validate_cycle_identity(
        ranking_session=ranking_session,
        decision_time=decision_time,
    )
    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates, Sequence
    ):
        raise CandidateRankingValidationError(
            "INVALID_CANDIDATE_BATCH",
            "candidates must be a finite sequence",
        )
    candidate_batch = tuple(candidates)

    validated_candidates: list[RankingCandidate] = []
    for candidate in candidate_batch:
        rebuilt = _revalidate_candidate(candidate)
        _validate_candidate_for_cycle(
            rebuilt,
            ranking_session=ranking_session,
            decision_time=decision_time,
        )
        validated_candidates.append(rebuilt)

    security_ids = tuple(
        candidate.security_id for candidate in validated_candidates
    )
    if len(set(security_ids)) != len(security_ids):
        raise CandidateRankingValidationError(
            "DUPLICATE_CANDIDATE_SECURITY_ID",
            "candidate security_ids must be unique before ranking",
        )

    candidates_with_metrics = tuple(
        (
            candidate,
            _trend_separation_atr(candidate),
        )
        for candidate in validated_candidates
    )
    ordered = sorted(
        candidates_with_metrics,
        key=lambda item: (-item[1], -item[0].rsi14, item[0].security_id),
    )
    ranked_candidates = tuple(
        RankedCandidate(
            security_id=candidate.security_id,
            rank=rank,
            ranking_session=candidate.ranking_session,
            decision_time=candidate.decision_time,
            signal_session=candidate.signal_session,
            signal_time=candidate.signal_time,
            sma20=candidate.sma20,
            sma50=candidate.sma50,
            atr14=candidate.atr14,
            rsi14=candidate.rsi14,
            trend_separation_atr=trend_separation_atr,
            input_fingerprint=candidate.input_fingerprint,
        )
        for rank, (candidate, trend_separation_atr) in enumerate(
            ordered, start=1
        )
    )

    policy = RankingPolicyRef(
        policy_id=CANDIDATE_RANKING_POLICY_ID,
        policy_version=CANDIDATE_RANKING_POLICY_VERSION,
        policy_fingerprint=CANDIDATE_RANKING_POLICY_FINGERPRINT,
    )
    input_set_fingerprint = compute_input_set_fingerprint(
        tuple(validated_candidates)
    )
    draft = CandidateRankingSnapshot(
        schema_version=CANDIDATE_RANKING_SNAPSHOT_SCHEMA_VERSION,
        ranking_session=ranking_session,
        decision_time=decision_time,
        policy=policy,
        candidate_count=len(ranked_candidates),
        ranked_candidates=ranked_candidates,
        input_set_fingerprint=input_set_fingerprint,
        snapshot_fingerprint="0" * 64,
    )
    values = draft.model_dump(mode="python")
    values["snapshot_fingerprint"] = compute_snapshot_fingerprint(draft)
    snapshot = CandidateRankingSnapshot.model_validate(values)
    return validate_ranking_snapshot(snapshot)


def _revalidate_snapshot(
    snapshot: object,
) -> CandidateRankingSnapshot:
    if not isinstance(snapshot, CandidateRankingSnapshot):
        raise CandidateRankingValidationError(
            "INVALID_RANKING_SNAPSHOT",
            "snapshot must be a CandidateRankingSnapshot",
        )
    try:
        rebuilt = CandidateRankingSnapshot.model_validate(
            snapshot.model_dump(mode="python")
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise CandidateRankingValidationError(
            "INVALID_RANKING_SNAPSHOT",
            "snapshot fails structural or semantic validation",
        ) from error
    if rebuilt != snapshot:
        raise CandidateRankingValidationError(
            "INVALID_RANKING_SNAPSHOT",
            "snapshot must already be canonical",
        )
    return snapshot


def validate_ranking_snapshot(
    snapshot: CandidateRankingSnapshot,
) -> CandidateRankingSnapshot:
    """Verify every structural, ordering, derivation, and hash invariant."""

    snapshot = _revalidate_snapshot(snapshot)
    if snapshot.schema_version != CANDIDATE_RANKING_SNAPSHOT_SCHEMA_VERSION:
        raise CandidateRankingValidationError(
            "UNSUPPORTED_SCHEMA_VERSION",
            "snapshot schema_version is not supported",
        )
    if snapshot.policy.policy_id != CANDIDATE_RANKING_POLICY_ID:
        raise CandidateRankingValidationError(
            "UNSUPPORTED_POLICY_ID",
            "ranking policy_id is not supported",
        )
    if snapshot.policy.policy_version != CANDIDATE_RANKING_POLICY_VERSION:
        raise CandidateRankingValidationError(
            "UNSUPPORTED_POLICY_VERSION",
            "ranking policy_version is not supported",
        )
    expected_policy_fingerprint = compute_policy_fingerprint()
    if snapshot.policy.policy_fingerprint != expected_policy_fingerprint:
        raise CandidateRankingValidationError(
            "POLICY_FINGERPRINT_MISMATCH",
            "policy_fingerprint does not match the frozen semantic policy",
        )
    if snapshot.candidate_count != len(snapshot.ranked_candidates):
        raise CandidateRankingValidationError(
            "CANDIDATE_COUNT_MISMATCH",
            "candidate_count must equal the ranked candidate tuple length",
        )

    security_ids = tuple(
        candidate.security_id for candidate in snapshot.ranked_candidates
    )
    if len(set(security_ids)) != len(security_ids):
        raise CandidateRankingValidationError(
            "DUPLICATE_CANDIDATE_SECURITY_ID",
            "ranked candidate security_ids must be unique",
        )

    ranks = tuple(candidate.rank for candidate in snapshot.ranked_candidates)
    expected_ranks = tuple(range(1, snapshot.candidate_count + 1))
    if ranks != expected_ranks:
        raise CandidateRankingValidationError(
            "INVALID_RANK_DOMAIN_OR_ORDER",
            "tuple order must be rank ascending with ranks exactly 1..N",
        )

    for candidate in snapshot.ranked_candidates:
        if candidate.ranking_session != snapshot.ranking_session:
            raise CandidateRankingValidationError(
                "CANDIDATE_SESSION_MISMATCH",
                "ranked candidate ranking_session disagrees with snapshot",
            )
        if candidate.decision_time != snapshot.decision_time:
            raise CandidateRankingValidationError(
                "CANDIDATE_DECISION_TIME_MISMATCH",
                "ranked candidate decision_time disagrees with snapshot",
            )
        expected_input_fingerprint = candidate_fingerprint(candidate)
        if candidate.input_fingerprint != expected_input_fingerprint:
            raise CandidateRankingValidationError(
                "CANDIDATE_FINGERPRINT_MISMATCH",
                "ranked candidate input_fingerprint is invalid",
            )
        expected_metric = _trend_separation_atr(candidate)
        if candidate.trend_separation_atr != expected_metric:
            raise CandidateRankingValidationError(
                "DERIVED_METRIC_MISMATCH",
                "trend_separation_atr does not equal (sma20 - sma50) / atr14",
            )

    canonical_order = tuple(
        sorted(
            snapshot.ranked_candidates,
            key=lambda candidate: (
                -candidate.trend_separation_atr,
                -candidate.rsi14,
                candidate.security_id,
            ),
        )
    )
    if snapshot.ranked_candidates != canonical_order:
        raise CandidateRankingValidationError(
            "RANKING_POLICY_ORDER_MISMATCH",
            "ranked candidates do not follow the frozen Phase 14A order",
        )

    expected_input_set_fingerprint = compute_input_set_fingerprint(
        snapshot.ranked_candidates
    )
    if snapshot.input_set_fingerprint != expected_input_set_fingerprint:
        raise CandidateRankingValidationError(
            "INPUT_SET_FINGERPRINT_MISMATCH",
            "input_set_fingerprint does not match the complete candidate set",
        )
    expected_snapshot_fingerprint = compute_snapshot_fingerprint(snapshot)
    if snapshot.snapshot_fingerprint != expected_snapshot_fingerprint:
        raise CandidateRankingValidationError(
            "SNAPSHOT_FINGERPRINT_MISMATCH",
            "snapshot_fingerprint does not match the ranking snapshot",
        )
    return snapshot


class CandidateRankingService:
    """Stateless service facade for the pure Phase 14 operations."""

    @staticmethod
    def rank_candidates(
        *,
        ranking_session: date,
        decision_time: datetime,
        candidates: Sequence[RankingCandidate],
    ) -> CandidateRankingSnapshot:
        return rank_candidates(
            ranking_session=ranking_session,
            decision_time=decision_time,
            candidates=candidates,
        )

    @staticmethod
    def validate_snapshot(
        snapshot: CandidateRankingSnapshot,
    ) -> CandidateRankingSnapshot:
        return validate_ranking_snapshot(snapshot)


__all__ = [
    "CandidateRankingService",
    "rank_candidates",
    "validate_ranking_snapshot",
]
