"""Phase 14 Deterministic Candidate Ranking v0.1."""

from stock_swing_d1.ranking.hashing import (
    CANDIDATE_RANKING_POLICY_FINGERPRINT,
    candidate_fingerprint,
    canonical_json_bytes,
    canonical_sha256,
    compute_candidate_input_fingerprint,
    compute_input_set_fingerprint,
    compute_policy_fingerprint,
    compute_snapshot_fingerprint,
    ranking_policy_manifest,
)
from stock_swing_d1.ranking.integration import (
    RANKED_ALLOCATION_BATCH_SCHEMA_VERSION,
    RankedAllocationAdapter,
    RankedAllocationBatch,
    RankedAllocationCandidate,
    RankedAllocationValidationError,
    build_ranked_allocation_batch,
    validate_ranked_allocation_batch,
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
from stock_swing_d1.ranking.service import (
    CandidateRankingService,
    rank_candidates,
    validate_ranking_snapshot,
)

__all__ = [
    "CANDIDATE_RANKING_POLICY_FINGERPRINT",
    "CANDIDATE_RANKING_POLICY_ID",
    "CANDIDATE_RANKING_POLICY_VERSION",
    "CANDIDATE_RANKING_SNAPSHOT_SCHEMA_VERSION",
    "RANKED_ALLOCATION_BATCH_SCHEMA_VERSION",
    "CandidateRankingService",
    "CandidateRankingSnapshot",
    "CandidateRankingValidationError",
    "RankedCandidate",
    "RankingCandidate",
    "RankingPolicyRef",
    "RankedAllocationAdapter",
    "RankedAllocationBatch",
    "RankedAllocationCandidate",
    "RankedAllocationValidationError",
    "build_ranked_allocation_batch",
    "candidate_fingerprint",
    "canonical_json_bytes",
    "canonical_sha256",
    "compute_candidate_input_fingerprint",
    "compute_input_set_fingerprint",
    "compute_policy_fingerprint",
    "compute_snapshot_fingerprint",
    "rank_candidates",
    "ranking_policy_manifest",
    "validate_ranking_snapshot",
    "validate_ranked_allocation_batch",
]
