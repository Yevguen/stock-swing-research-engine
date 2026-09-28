"""Fail-closed Phase 14 ranking-snapshot validation tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from stock_swing_d1.ranking import (
    CandidateRankingValidationError,
    candidate_fingerprint,
    compute_input_set_fingerprint,
    compute_snapshot_fingerprint,
    rank_candidates,
    validate_ranking_snapshot,
)
from tests.ranking.conftest import (
    DECISION_TIME,
    RANKING_SESSION,
    rehash_snapshot,
)


@pytest.fixture
def ranked_snapshot(make_candidate):
    return rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=(
            make_candidate(
                security_id="NORGATE:1",
                sma20=115.0,
                sma50=100.0,
                atr14=5.0,
                rsi14=70.0,
            ),
            make_candidate(
                security_id="NORGATE:2",
                sma20=110.0,
                sma50=100.0,
                atr14=5.0,
                rsi14=60.0,
            ),
            make_candidate(
                security_id="NORGATE:3",
                sma20=105.0,
                sma50=100.0,
                atr14=5.0,
                rsi14=55.0,
            ),
        ),
    )


def with_candidates(snapshot, candidates, *, refresh_input_set: bool = False):
    values = {"ranked_candidates": tuple(candidates)}
    if refresh_input_set:
        values["input_set_fingerprint"] = compute_input_set_fingerprint(
            tuple(candidates)
        )
    changed = snapshot.model_copy(update=values)
    return rehash_snapshot(changed)


def test_valid_snapshot_is_returned_unchanged(ranked_snapshot) -> None:
    assert validate_ranking_snapshot(ranked_snapshot) is ranked_snapshot


def test_derived_metric_mismatch_is_rejected(ranked_snapshot) -> None:
    candidates = list(ranked_snapshot.ranked_candidates)
    candidates[0] = candidates[0].model_copy(
        update={
            "trend_separation_atr": (
                candidates[0].trend_separation_atr + 0.000000000000001
            )
        }
    )
    malformed = with_candidates(ranked_snapshot, candidates)

    with pytest.raises(
        CandidateRankingValidationError, match="DERIVED_METRIC_MISMATCH"
    ):
        validate_ranking_snapshot(malformed)


def test_malformed_tuple_order_is_rejected_without_sorting(ranked_snapshot) -> None:
    malformed = with_candidates(
        ranked_snapshot,
        tuple(reversed(ranked_snapshot.ranked_candidates)),
    )

    with pytest.raises(
        CandidateRankingValidationError, match="INVALID_RANK_DOMAIN_OR_ORDER"
    ):
        validate_ranking_snapshot(malformed)
    assert malformed.ranked_candidates[0].rank == 3


def test_malformed_phase14a_order_is_rejected(ranked_snapshot) -> None:
    first, second, third = ranked_snapshot.ranked_candidates
    malformed_candidates = (
        second.model_copy(update={"rank": 1}),
        first.model_copy(update={"rank": 2}),
        third,
    )
    malformed = with_candidates(ranked_snapshot, malformed_candidates)

    with pytest.raises(
        CandidateRankingValidationError, match="RANKING_POLICY_ORDER_MISMATCH"
    ):
        validate_ranking_snapshot(malformed)


def test_duplicate_rank_is_rejected(ranked_snapshot) -> None:
    first, second, third = ranked_snapshot.ranked_candidates
    malformed = with_candidates(
        ranked_snapshot,
        (first, second.model_copy(update={"rank": 1}), third),
    )

    with pytest.raises(
        CandidateRankingValidationError, match="INVALID_RANK_DOMAIN_OR_ORDER"
    ):
        validate_ranking_snapshot(malformed)


def test_rank_gap_is_rejected(ranked_snapshot) -> None:
    first, second, third = ranked_snapshot.ranked_candidates
    malformed = with_candidates(
        ranked_snapshot,
        (first, second.model_copy(update={"rank": 3}), third),
    )

    with pytest.raises(
        CandidateRankingValidationError, match="INVALID_RANK_DOMAIN_OR_ORDER"
    ):
        validate_ranking_snapshot(malformed)


def test_unsupported_schema_version_is_rejected(ranked_snapshot) -> None:
    malformed = rehash_snapshot(
        ranked_snapshot.model_copy(update={"schema_version": "unsupported.v9"})
    )

    with pytest.raises(
        CandidateRankingValidationError, match="UNSUPPORTED_SCHEMA_VERSION"
    ):
        validate_ranking_snapshot(malformed)


def test_unsupported_policy_version_is_rejected(ranked_snapshot) -> None:
    policy = ranked_snapshot.policy.model_copy(update={"policy_version": "9.0"})
    malformed = rehash_snapshot(ranked_snapshot.model_copy(update={"policy": policy}))

    with pytest.raises(
        CandidateRankingValidationError, match="UNSUPPORTED_POLICY_VERSION"
    ):
        validate_ranking_snapshot(malformed)


def test_unsupported_policy_id_is_rejected(ranked_snapshot) -> None:
    policy = ranked_snapshot.policy.model_copy(update={"policy_id": "other"})
    malformed = rehash_snapshot(ranked_snapshot.model_copy(update={"policy": policy}))

    with pytest.raises(
        CandidateRankingValidationError, match="UNSUPPORTED_POLICY_ID"
    ):
        validate_ranking_snapshot(malformed)


def test_policy_fingerprint_mismatch_is_rejected(ranked_snapshot) -> None:
    policy = ranked_snapshot.policy.model_copy(
        update={"policy_fingerprint": "f" * 64}
    )
    malformed = rehash_snapshot(ranked_snapshot.model_copy(update={"policy": policy}))

    with pytest.raises(
        CandidateRankingValidationError, match="POLICY_FINGERPRINT_MISMATCH"
    ):
        validate_ranking_snapshot(malformed)


def test_candidate_count_mismatch_is_rejected(ranked_snapshot) -> None:
    malformed = rehash_snapshot(
        ranked_snapshot.model_copy(update={"candidate_count": 2})
    )

    with pytest.raises(
        CandidateRankingValidationError, match="CANDIDATE_COUNT_MISMATCH"
    ):
        validate_ranking_snapshot(malformed)


def test_duplicate_security_is_rejected(ranked_snapshot) -> None:
    first, second, third = ranked_snapshot.ranked_candidates
    duplicate = second.model_copy(update={"security_id": first.security_id})
    malformed = with_candidates(ranked_snapshot, (first, duplicate, third))

    with pytest.raises(
        CandidateRankingValidationError, match="DUPLICATE_CANDIDATE_SECURITY_ID"
    ):
        validate_ranking_snapshot(malformed)


def test_candidate_input_fingerprint_mismatch_is_rejected(ranked_snapshot) -> None:
    candidates = list(ranked_snapshot.ranked_candidates)
    candidates[0] = candidates[0].model_copy(
        update={"input_fingerprint": "f" * 64}
    )
    malformed = with_candidates(
        ranked_snapshot, candidates, refresh_input_set=True
    )

    with pytest.raises(
        CandidateRankingValidationError, match="CANDIDATE_FINGERPRINT_MISMATCH"
    ):
        validate_ranking_snapshot(malformed)


def test_input_set_fingerprint_mismatch_is_rejected(ranked_snapshot) -> None:
    malformed = rehash_snapshot(
        ranked_snapshot.model_copy(update={"input_set_fingerprint": "f" * 64})
    )

    with pytest.raises(
        CandidateRankingValidationError, match="INPUT_SET_FINGERPRINT_MISMATCH"
    ):
        validate_ranking_snapshot(malformed)


def test_snapshot_fingerprint_mismatch_is_rejected(ranked_snapshot) -> None:
    malformed = ranked_snapshot.model_copy(
        update={"snapshot_fingerprint": "f" * 64}
    )

    with pytest.raises(
        CandidateRankingValidationError, match="SNAPSHOT_FINGERPRINT_MISMATCH"
    ):
        validate_ranking_snapshot(malformed)


def test_candidate_snapshot_session_consistency_is_verified(ranked_snapshot) -> None:
    candidate = ranked_snapshot.ranked_candidates[0]
    later_candidate = candidate.model_copy(
        update={
            "ranking_session": candidate.ranking_session + timedelta(days=1),
            "decision_time": candidate.decision_time + timedelta(days=1),
            "signal_session": candidate.signal_session + timedelta(days=1),
            "signal_time": candidate.signal_time + timedelta(days=1),
        }
    )
    later_candidate = later_candidate.model_copy(
        update={"input_fingerprint": candidate_fingerprint(later_candidate)}
    )
    candidates = (later_candidate, *ranked_snapshot.ranked_candidates[1:])
    malformed = with_candidates(
        ranked_snapshot, candidates, refresh_input_set=True
    )

    with pytest.raises(
        CandidateRankingValidationError, match="CANDIDATE_SESSION_MISMATCH"
    ):
        validate_ranking_snapshot(malformed)


def test_snapshot_validator_never_overwrites_bad_fingerprint(ranked_snapshot) -> None:
    malformed = ranked_snapshot.model_copy(
        update={"snapshot_fingerprint": "f" * 64}
    )

    with pytest.raises(CandidateRankingValidationError):
        validate_ranking_snapshot(malformed)
    assert malformed.snapshot_fingerprint == "f" * 64
    assert compute_snapshot_fingerprint(malformed) != malformed.snapshot_fingerprint
