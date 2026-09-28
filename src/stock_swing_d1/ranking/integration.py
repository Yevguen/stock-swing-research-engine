"""Phase 14B rank-preserving adapter into Phase 12 allocation."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ConfigDict, ValidationError, model_validator

from stock_swing_d1.portfolio.models import (
    PortfolioAllocationValidationError,
    PortfolioCandidate,
    _validate_portfolio_candidate,
)
from stock_swing_d1.ranking.hashing import (
    CANDIDATE_RANKING_POLICY_FINGERPRINT,
)
from stock_swing_d1.ranking.models import (
    CandidateRankingSnapshot,
    CandidateRankingValidationError,
    _AwareDateTime,
    _CanonicalSecurityId,
    _ImmutableRankingModel,
    _NonNegativeCount,
    _PositiveRank,
    _SessionDate,
    _Sha256,
)
from stock_swing_d1.ranking.service import validate_ranking_snapshot


RANKED_ALLOCATION_BATCH_SCHEMA_VERSION = "ranked_allocation_batch_v0.1"


class RankedAllocationValidationError(ValueError):
    """A Phase 14B ranking-to-allocation contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class _ImmutableRankedAllocationModel(_ImmutableRankingModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )


class RankedAllocationCandidate(_ImmutableRankedAllocationModel):
    """One Phase 12 candidate carrying authoritative Phase 14 provenance."""

    security_id: _CanonicalSecurityId
    rank: _PositiveRank
    ranking_snapshot_fingerprint: _Sha256
    ranking_input_fingerprint: _Sha256
    allocation_candidate: PortfolioCandidate

    @model_validator(mode="after")
    def validate_security_identity(self) -> RankedAllocationCandidate:
        try:
            _validate_portfolio_candidate(self.allocation_candidate)
        except PortfolioAllocationValidationError as error:
            raise ValueError(
                "allocation_candidate fails the Phase 12 candidate contract"
            ) from error
        if self.security_id != self.allocation_candidate.security_id:
            raise ValueError(
                "security_id must equal allocation_candidate.security_id"
            )
        return self


class RankedAllocationBatch(_ImmutableRankedAllocationModel):
    """Complete rank-ordered Phase 12 input batch with ranking provenance."""

    schema_version: str
    ranking_session: _SessionDate
    decision_time: _AwareDateTime
    ranking_snapshot_fingerprint: _Sha256
    policy_fingerprint: _Sha256
    candidate_count: _NonNegativeCount
    candidates: tuple[RankedAllocationCandidate, ...]


def _revalidate_ranked_allocation_batch(
    ranked_batch: object,
) -> RankedAllocationBatch:
    if not isinstance(ranked_batch, RankedAllocationBatch):
        raise RankedAllocationValidationError(
            "INVALID_RANKED_ALLOCATION_BATCH",
            "ranked_batch must be a RankedAllocationBatch",
        )
    try:
        rebuilt = RankedAllocationBatch.model_validate(
            ranked_batch.model_dump(mode="python")
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise RankedAllocationValidationError(
            "INVALID_RANKED_ALLOCATION_BATCH",
            "ranked batch fails structural or candidate validation",
        ) from error
    if rebuilt != ranked_batch:
        raise RankedAllocationValidationError(
            "INVALID_RANKED_ALLOCATION_BATCH",
            "ranked batch must already be canonical",
        )
    return ranked_batch


def validate_ranked_allocation_batch(
    ranked_batch: RankedAllocationBatch,
) -> RankedAllocationBatch:
    """Fail closed on every Phase 14B batch and provenance invariant."""

    ranked_batch = _revalidate_ranked_allocation_batch(ranked_batch)
    if ranked_batch.schema_version != RANKED_ALLOCATION_BATCH_SCHEMA_VERSION:
        raise RankedAllocationValidationError(
            "UNSUPPORTED_RANKED_ALLOCATION_SCHEMA",
            "ranked allocation schema_version is not supported",
        )
    if (
        ranked_batch.policy_fingerprint
        != CANDIDATE_RANKING_POLICY_FINGERPRINT
    ):
        raise RankedAllocationValidationError(
            "UNSUPPORTED_RANKING_POLICY_FINGERPRINT",
            "policy_fingerprint is not the frozen Phase 14A policy",
        )
    if ranked_batch.candidate_count != len(ranked_batch.candidates):
        raise RankedAllocationValidationError(
            "RANKED_CANDIDATE_COUNT_MISMATCH",
            "candidate_count must equal the candidate tuple length",
        )

    security_ids = tuple(
        candidate.security_id for candidate in ranked_batch.candidates
    )
    if len(set(security_ids)) != len(security_ids):
        raise RankedAllocationValidationError(
            "DUPLICATE_RANKED_SECURITY_ID",
            "ranked allocation security_ids must be unique",
        )

    ranks = tuple(candidate.rank for candidate in ranked_batch.candidates)
    if len(set(ranks)) != len(ranks):
        raise RankedAllocationValidationError(
            "DUPLICATE_RANK",
            "ranked allocation ranks must be unique",
        )
    expected_ranks = tuple(range(1, ranked_batch.candidate_count + 1))
    if set(ranks) != set(expected_ranks):
        raise RankedAllocationValidationError(
            "INVALID_RANK_DOMAIN",
            "ranked allocation ranks must have no gaps and equal 1..N",
        )
    if ranks != expected_ranks:
        raise RankedAllocationValidationError(
            "RANK_ORDER_MISMATCH",
            "candidate tuple order must be rank ascending",
        )

    for candidate in ranked_batch.candidates:
        if (
            candidate.ranking_snapshot_fingerprint
            != ranked_batch.ranking_snapshot_fingerprint
        ):
            raise RankedAllocationValidationError(
                "RANKING_SNAPSHOT_FINGERPRINT_MISMATCH",
                "every candidate must reference the batch ranking snapshot",
            )
        if candidate.security_id != candidate.allocation_candidate.security_id:
            raise RankedAllocationValidationError(
                "RANKED_ALLOCATION_IDENTITY_MISMATCH",
                "wrapper and Phase 12 candidate security_ids must agree",
            )
    return ranked_batch


def build_ranked_allocation_batch(
    *,
    ranking_snapshot: CandidateRankingSnapshot,
    allocation_candidates: Sequence[PortfolioCandidate],
) -> RankedAllocationBatch:
    """Reconcile a validated ranking one-to-one with Phase 12 candidates."""

    try:
        ranking_snapshot = validate_ranking_snapshot(ranking_snapshot)
    except CandidateRankingValidationError as error:
        raise RankedAllocationValidationError(
            "INVALID_RANKING_SNAPSHOT",
            "ranking_snapshot must satisfy the complete Phase 14 contract",
        ) from error
    if isinstance(allocation_candidates, (str, bytes)) or not isinstance(
        allocation_candidates, Sequence
    ):
        raise RankedAllocationValidationError(
            "INVALID_ALLOCATION_CANDIDATE_BATCH",
            "allocation_candidates must be a finite sequence",
        )

    supplied = tuple(allocation_candidates)
    validated_candidates: list[PortfolioCandidate] = []
    for candidate in supplied:
        try:
            validated_candidates.append(_validate_portfolio_candidate(candidate))
        except PortfolioAllocationValidationError as error:
            raise RankedAllocationValidationError(
                "INVALID_ALLOCATION_CANDIDATE",
                "allocation candidate fails the Phase 12 contract",
            ) from error

    allocation_security_ids = tuple(
        candidate.security_id for candidate in validated_candidates
    )
    if len(set(allocation_security_ids)) != len(allocation_security_ids):
        raise RankedAllocationValidationError(
            "DUPLICATE_ALLOCATION_SECURITY_ID",
            "allocation candidate security_ids must be unique",
        )

    ranking_security_ids = {
        candidate.security_id for candidate in ranking_snapshot.ranked_candidates
    }
    allocation_security_id_set = set(allocation_security_ids)
    missing = ranking_security_ids - allocation_security_id_set
    extra = allocation_security_id_set - ranking_security_ids
    if missing and extra:
        raise RankedAllocationValidationError(
            "SECURITY_IDENTITY_MISMATCH",
            "ranking and allocation candidate security sets disagree",
        )
    if missing:
        raise RankedAllocationValidationError(
            "MISSING_ALLOCATION_SECURITY",
            "one or more ranked securities lack an allocation candidate",
        )
    if extra:
        raise RankedAllocationValidationError(
            "EXTRA_ALLOCATION_SECURITY",
            "one or more allocation candidates are absent from the ranking",
        )

    candidates_by_security_id = {
        candidate.security_id: candidate for candidate in validated_candidates
    }
    ranked_allocation_candidates = tuple(
        RankedAllocationCandidate(
            security_id=ranked_candidate.security_id,
            rank=ranked_candidate.rank,
            ranking_snapshot_fingerprint=(
                ranking_snapshot.snapshot_fingerprint
            ),
            ranking_input_fingerprint=ranked_candidate.input_fingerprint,
            allocation_candidate=candidates_by_security_id[
                ranked_candidate.security_id
            ],
        )
        for ranked_candidate in ranking_snapshot.ranked_candidates
    )
    batch = RankedAllocationBatch(
        schema_version=RANKED_ALLOCATION_BATCH_SCHEMA_VERSION,
        ranking_session=ranking_snapshot.ranking_session,
        decision_time=ranking_snapshot.decision_time,
        ranking_snapshot_fingerprint=ranking_snapshot.snapshot_fingerprint,
        policy_fingerprint=ranking_snapshot.policy.policy_fingerprint,
        candidate_count=ranking_snapshot.candidate_count,
        candidates=ranked_allocation_candidates,
    )
    return validate_ranked_allocation_batch(batch)


class RankedAllocationAdapter:
    """Stateless Phase 14B adapter facade."""

    @staticmethod
    def build_batch(
        *,
        ranking_snapshot: CandidateRankingSnapshot,
        allocation_candidates: Sequence[PortfolioCandidate],
    ) -> RankedAllocationBatch:
        return build_ranked_allocation_batch(
            ranking_snapshot=ranking_snapshot,
            allocation_candidates=allocation_candidates,
        )


__all__ = [
    "RANKED_ALLOCATION_BATCH_SCHEMA_VERSION",
    "RankedAllocationAdapter",
    "RankedAllocationBatch",
    "RankedAllocationCandidate",
    "RankedAllocationValidationError",
    "build_ranked_allocation_batch",
    "validate_ranked_allocation_batch",
]
