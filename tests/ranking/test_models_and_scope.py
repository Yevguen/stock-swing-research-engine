"""Phase 14 model, public-API, and scope tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import inspect

import pytest
from pydantic import ValidationError

import stock_swing_d1.ranking as ranking_package
from stock_swing_d1.ranking import (
    CANDIDATE_RANKING_POLICY_FINGERPRINT,
    CANDIDATE_RANKING_POLICY_ID,
    CANDIDATE_RANKING_POLICY_VERSION,
    CANDIDATE_RANKING_SNAPSHOT_SCHEMA_VERSION,
    CandidateRankingService,
    CandidateRankingSnapshot,
    RankedCandidate,
    RankingCandidate,
    RankingPolicyRef,
    rank_candidates,
)
from tests.ranking.conftest import DECISION_TIME, RANKING_SESSION, SIGNAL_TIME


def test_fixed_identifiers_are_exact() -> None:
    assert CANDIDATE_RANKING_POLICY_ID == "candidate_ranking_policy_v0.1"
    assert CANDIDATE_RANKING_POLICY_VERSION == "0.1"
    assert (
        CANDIDATE_RANKING_SNAPSHOT_SCHEMA_VERSION
        == "candidate_ranking_snapshot_v0.1"
    )
    assert len(CANDIDATE_RANKING_POLICY_FINGERPRINT) == 64


def test_required_public_api_is_exported() -> None:
    assert {
        "CandidateRankingService",
        "CandidateRankingSnapshot",
        "CandidateRankingValidationError",
        "RankedCandidate",
        "RankingCandidate",
        "RankingPolicyRef",
        "compute_candidate_input_fingerprint",
        "compute_input_set_fingerprint",
        "compute_policy_fingerprint",
        "compute_snapshot_fingerprint",
        "rank_candidates",
        "validate_ranking_snapshot",
    } <= set(ranking_package.__all__)


def test_models_are_frozen_and_tuple_backed(make_candidate) -> None:
    candidate = make_candidate()
    snapshot = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=[candidate],
    )

    assert isinstance(snapshot.ranked_candidates, tuple)
    with pytest.raises(ValidationError, match="frozen"):
        candidate.sma20 = 999.0
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.policy.policy_version = "9.0"
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.candidate_count = 0
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.ranked_candidates[0].rank = 2
    with pytest.raises(TypeError):
        snapshot.ranked_candidates[0] = snapshot.ranked_candidates[0]


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (
            RankingPolicyRef,
            {
                "policy_id": CANDIDATE_RANKING_POLICY_ID,
                "policy_version": CANDIDATE_RANKING_POLICY_VERSION,
                "policy_fingerprint": CANDIDATE_RANKING_POLICY_FINGERPRINT,
            },
        ),
        (
            RankingCandidate,
            {
                "security_id": "NORGATE:1",
                "ranking_session": RANKING_SESSION,
                "decision_time": DECISION_TIME,
                "signal_session": RANKING_SESSION,
                "signal_time": SIGNAL_TIME,
                "sma20": 2.0,
                "sma50": 1.0,
                "atr14": 1.0,
                "rsi14": 51.0,
                "input_fingerprint": "a" * 64,
            },
        ),
    ],
)
def test_models_forbid_unexpected_fields(model, values) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model(**values, portfolio_cash=1_000_000.0)


@pytest.mark.parametrize("field_name", ["decision_time", "signal_time"])
def test_candidate_requires_timezone_aware_timestamps(
    make_candidate, field_name: str
) -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_candidate(
            **{
                field_name: datetime(2026, 8, 21, 20, 0),
                "input_fingerprint": "a" * 64,
            }
        )


def test_snapshot_requires_timezone_aware_decision_time(make_candidate) -> None:
    candidate = make_candidate()
    policy = RankingPolicyRef(
        policy_id=CANDIDATE_RANKING_POLICY_ID,
        policy_version=CANDIDATE_RANKING_POLICY_VERSION,
        policy_fingerprint=CANDIDATE_RANKING_POLICY_FINGERPRINT,
    )

    with pytest.raises(ValidationError, match="timezone-aware"):
        CandidateRankingSnapshot(
            schema_version=CANDIDATE_RANKING_SNAPSHOT_SCHEMA_VERSION,
            ranking_session=RANKING_SESSION,
            decision_time=datetime(2026, 8, 21, 20, 0),
            policy=policy,
            candidate_count=0,
            ranked_candidates=(),
            input_set_fingerprint="a" * 64,
            snapshot_fingerprint="b" * 64,
        )


def test_future_signal_time_is_rejected(make_candidate) -> None:
    with pytest.raises(ValidationError, match="later than decision_time"):
        make_candidate(signal_time=DECISION_TIME + timedelta(seconds=1))


def test_signal_and_ranking_session_mismatch_is_rejected(make_candidate) -> None:
    prior_session = RANKING_SESSION - timedelta(days=1)
    with pytest.raises(ValidationError, match="equal ranking_session"):
        make_candidate(
            signal_session=prior_session,
            signal_time=datetime(2026, 8, 20, 20, 0, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("atr14", 0.0, "greater than zero"),
        ("atr14", -1.0, "greater than zero"),
        ("sma20", 100.0, "greater than sma50"),
        ("sma20", 99.0, "greater than sma50"),
        ("rsi14", 50.0, "greater than 50"),
        ("rsi14", 49.0, "greater than 50"),
    ],
)
def test_phase8_consistency_invariants_are_enforced(
    make_candidate, field_name: str, value: float, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_candidate(**{field_name: value})


@pytest.mark.parametrize("field_name", ["sma20", "sma50", "atr14", "rsi14"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_mandatory_numeric_inputs_reject_missing_or_nonfinite_values(
    make_candidate, field_name: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        make_candidate(
            **{field_name: value, "input_fingerprint": "a" * 64}
        )


@pytest.mark.parametrize(
    "security_id", ["", "NORGATE:0", "NORGATE:-1", " NORGATE:1", "OTHER:1"]
)
def test_security_id_uses_the_canonical_repository_identity(
    make_candidate, security_id: str
) -> None:
    with pytest.raises(ValidationError):
        make_candidate(security_id=security_id)


def test_ranking_contract_contains_no_portfolio_or_weight_fields() -> None:
    model_fields = (
        set(RankingCandidate.model_fields)
        | set(RankedCandidate.model_fields)
        | set(CandidateRankingSnapshot.model_fields)
    )
    assert {
        "weight",
        "weighted_score",
        "portfolio_cash",
        "position_quantity",
        "available_slots",
        "sector",
        "sma100",
        "sma200",
        "relative_volume",
    }.isdisjoint(model_fields)
    assert tuple(inspect.signature(rank_candidates).parameters) == (
        "ranking_session",
        "decision_time",
        "candidates",
    )
    assert tuple(
        inspect.signature(CandidateRankingService.rank_candidates).parameters
    ) == ("ranking_session", "decision_time", "candidates")


def test_session_fields_require_real_dates_not_datetimes(make_candidate) -> None:
    with pytest.raises(ValidationError, match="datetime.date"):
        make_candidate(ranking_session=DECISION_TIME)


def test_local_timestamp_dates_do_not_define_market_session(make_candidate) -> None:
    next_date_offset = timezone(timedelta(hours=5))
    decision_time = DECISION_TIME.astimezone(next_date_offset)
    signal_time = SIGNAL_TIME.astimezone(next_date_offset)

    candidate = make_candidate(
        decision_time=decision_time,
        signal_time=signal_time,
    )
    snapshot = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=decision_time,
        candidates=(candidate,),
    )

    assert decision_time.date() == date(2026, 8, 22)
    assert signal_time.date() == date(2026, 8, 22)
    assert candidate.ranking_session == date(2026, 8, 21)
    assert candidate.signal_session == date(2026, 8, 21)
    assert snapshot.ranking_session == date(2026, 8, 21)
    assert snapshot.decision_time.date() == date(2026, 8, 22)
    assert snapshot.candidate_count == 1
    assert snapshot.ranked_candidates[0].rank == 1
