"""Deterministic ordering and canonical fingerprint tests for Phase 14."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from itertools import permutations
import json
import math

import pytest

from stock_swing_d1.ranking import (
    CANDIDATE_RANKING_POLICY_FINGERPRINT,
    candidate_fingerprint,
    canonical_json_bytes,
    compute_candidate_input_fingerprint,
    compute_input_set_fingerprint,
    compute_policy_fingerprint,
    compute_snapshot_fingerprint,
    rank_candidates,
    ranking_policy_manifest,
)
from tests.ranking.conftest import DECISION_TIME, RANKING_SESSION, SIGNAL_TIME


def ranked_ids(snapshot) -> list[str]:
    return [candidate.security_id for candidate in snapshot.ranked_candidates]


def test_primary_metric_ranks_descending_without_truncation(make_candidate) -> None:
    candidates = (
        make_candidate(
            security_id="NORGATE:1", sma20=102.0, sma50=100.0, atr14=2.0
        ),
        make_candidate(
            security_id="NORGATE:2", sma20=106.0, sma50=100.0, atr14=2.0
        ),
        make_candidate(
            security_id="NORGATE:3", sma20=104.0, sma50=100.0, atr14=2.0
        ),
    )

    snapshot = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=candidates,
    )

    assert snapshot.candidate_count == 3
    assert ranked_ids(snapshot) == ["NORGATE:2", "NORGATE:3", "NORGATE:1"]
    assert [item.trend_separation_atr for item in snapshot.ranked_candidates] == [
        3.0,
        2.0,
        1.0,
    ]
    assert [item.rank for item in snapshot.ranked_candidates] == [1, 2, 3]


def test_rsi_is_the_secondary_descending_tiebreak(make_candidate) -> None:
    candidates = (
        make_candidate(security_id="NORGATE:1", rsi14=55.0),
        make_candidate(security_id="NORGATE:2", rsi14=75.0),
        make_candidate(security_id="NORGATE:3", rsi14=65.0),
    )
    snapshot = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=candidates,
    )

    assert ranked_ids(snapshot) == ["NORGATE:2", "NORGATE:3", "NORGATE:1"]


def test_security_id_is_the_final_ascending_tiebreak(make_candidate) -> None:
    candidates = (
        make_candidate(security_id="NORGATE:20"),
        make_candidate(security_id="NORGATE:3"),
        make_candidate(security_id="NORGATE:10"),
    )
    snapshot = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=candidates,
    )

    assert ranked_ids(snapshot) == ["NORGATE:10", "NORGATE:20", "NORGATE:3"]


def test_every_caller_input_permutation_produces_the_same_snapshot(
    make_candidate,
) -> None:
    candidates = (
        make_candidate(security_id="NORGATE:1", rsi14=55.0),
        make_candidate(security_id="NORGATE:2", rsi14=65.0),
        make_candidate(security_id="NORGATE:3", rsi14=75.0),
    )
    expected = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=candidates,
    )

    for candidate_order in permutations(candidates):
        assert rank_candidates(
            ranking_session=RANKING_SESSION,
            decision_time=DECISION_TIME,
            candidates=candidate_order,
        ) == expected


def test_no_display_rounding_creates_a_false_tie(make_candidate) -> None:
    slightly_larger_sma = math.nextafter(110.0, math.inf)
    candidates = (
        make_candidate(security_id="NORGATE:1", sma20=110.0),
        make_candidate(security_id="NORGATE:2", sma20=slightly_larger_sma),
    )

    snapshot = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=candidates,
    )

    assert ranked_ids(snapshot) == ["NORGATE:2", "NORGATE:1"]
    assert (
        snapshot.ranked_candidates[0].trend_separation_atr
        > snapshot.ranked_candidates[1].trend_separation_atr
    )


def test_empty_batch_produces_one_deterministic_empty_snapshot() -> None:
    first = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=(),
    )
    second = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=[],
    )

    assert first == second
    assert first.candidate_count == 0
    assert first.ranked_candidates == ()
    assert len(first.input_set_fingerprint) == 64
    assert len(first.snapshot_fingerprint) == 64


def test_single_candidate_gets_rank_one(make_candidate) -> None:
    snapshot = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=(make_candidate(),),
    )

    assert snapshot.candidate_count == 1
    assert snapshot.ranked_candidates[0].rank == 1


def test_repeated_calls_are_exactly_equal(make_candidate) -> None:
    candidates = (make_candidate(),)
    calls = tuple(
        rank_candidates(
            ranking_session=RANKING_SESSION,
            decision_time=DECISION_TIME,
            candidates=candidates,
        )
        for _ in range(3)
    )

    assert calls[0] == calls[1] == calls[2]


def test_duplicate_security_is_rejected_before_ranking(make_candidate) -> None:
    with pytest.raises(ValueError, match="DUPLICATE_CANDIDATE_SECURITY_ID"):
        rank_candidates(
            ranking_session=RANKING_SESSION,
            decision_time=DECISION_TIME,
            candidates=(make_candidate(), make_candidate(rsi14=70.0)),
        )


def test_bad_supplied_candidate_fingerprint_is_rejected(make_candidate) -> None:
    candidate = make_candidate(input_fingerprint="f" * 64)

    with pytest.raises(ValueError, match="CANDIDATE_FINGERPRINT_MISMATCH"):
        rank_candidates(
            ranking_session=RANKING_SESSION,
            decision_time=DECISION_TIME,
            candidates=(candidate,),
        )


def test_binary64_overflow_in_derived_metric_fails_closed(make_candidate) -> None:
    candidate = make_candidate(
        sma20=2.0,
        sma50=1.0,
        atr14=float.fromhex("0x0.0000000000001p-1022"),
    )

    with pytest.raises(ValueError, match="INVALID_DERIVED_METRIC"):
        rank_candidates(
            ranking_session=RANKING_SESSION,
            decision_time=DECISION_TIME,
            candidates=(candidate,),
        )


def test_ranking_does_not_mutate_inputs(make_candidate) -> None:
    candidates = [
        make_candidate(security_id="NORGATE:2", rsi14=55.0),
        make_candidate(security_id="NORGATE:1", rsi14=75.0),
    ]
    before = tuple(candidate.model_dump(mode="python") for candidate in candidates)

    rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=candidates,
    )

    assert tuple(candidate.model_dump(mode="python") for candidate in candidates) == before
    assert [candidate.security_id for candidate in candidates] == [
        "NORGATE:2",
        "NORGATE:1",
    ]


def test_candidate_fingerprint_is_deterministic_and_matches_model(make_candidate) -> None:
    first = make_candidate()
    second = make_candidate()

    assert first.input_fingerprint == second.input_fingerprint
    assert candidate_fingerprint(first) == first.input_fingerprint


def test_integer_spellings_of_float_fields_use_binary64_fingerprints(
    make_candidate,
) -> None:
    float_candidate = make_candidate()
    integer_candidate = make_candidate(
        sma20=110,
        sma50=100,
        atr14=5,
        rsi14=60,
    )

    assert integer_candidate == float_candidate
    assert integer_candidate.input_fingerprint == float_candidate.input_fingerprint


@pytest.mark.parametrize(
    "overrides",
    [
        {"security_id": "NORGATE:101"},
        {
            "ranking_session": RANKING_SESSION + timedelta(days=1),
            "signal_session": RANKING_SESSION + timedelta(days=1),
            "decision_time": DECISION_TIME + timedelta(days=1),
            "signal_time": SIGNAL_TIME + timedelta(days=1),
        },
        {"decision_time": DECISION_TIME + timedelta(seconds=1)},
        {
            "signal_session": RANKING_SESSION + timedelta(days=1),
            "ranking_session": RANKING_SESSION + timedelta(days=1),
            "decision_time": DECISION_TIME + timedelta(days=1),
            "signal_time": SIGNAL_TIME + timedelta(days=1),
        },
        {"signal_time": SIGNAL_TIME - timedelta(seconds=1)},
        {"sma20": 111.0},
        {"sma50": 99.0},
        {"atr14": 4.0},
        {"rsi14": 61.0},
    ],
)
def test_candidate_fingerprint_changes_with_every_semantic_input(
    make_candidate, overrides: dict[str, object]
) -> None:
    baseline = make_candidate()
    changed = make_candidate(**overrides)

    assert changed.input_fingerprint != baseline.input_fingerprint


def test_input_set_fingerprint_is_permutation_invariant(make_candidate) -> None:
    candidates = (
        make_candidate(security_id="NORGATE:1"),
        make_candidate(security_id="NORGATE:2"),
    )

    assert compute_input_set_fingerprint(candidates) == (
        compute_input_set_fingerprint(tuple(reversed(candidates)))
    )


def test_policy_fingerprint_is_semantic_and_deterministic() -> None:
    manifest = ranking_policy_manifest()

    assert compute_policy_fingerprint() == compute_policy_fingerprint()
    assert compute_policy_fingerprint() == CANDIDATE_RANKING_POLICY_FINGERPRINT
    assert manifest["primary_metric"] == "(SMA20 - SMA50) / ATR14"
    assert manifest["mandatory_inputs"] == ["SMA20", "SMA50", "ATR14", "RSI14"]
    assert manifest["round_before_compare"] is False
    assert manifest["weighted_score"] is False
    assert manifest["stochastic_behavior"] is False


def test_snapshot_fingerprint_is_deterministic_and_sensitive(make_candidate) -> None:
    snapshot = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=(
            make_candidate(security_id="NORGATE:1"),
            make_candidate(security_id="NORGATE:2", rsi14=55.0),
        ),
    )
    first, second = snapshot.ranked_candidates
    rank_changed = snapshot.model_copy(
        update={
            "ranked_candidates": (
                first.model_copy(update={"rank": 2}),
                second,
            )
        }
    )
    value_changed = snapshot.model_copy(
        update={
            "ranked_candidates": (
                first.model_copy(update={"rsi14": first.rsi14 + 1.0}),
                second,
            )
        }
    )

    assert compute_snapshot_fingerprint(snapshot) == snapshot.snapshot_fingerprint
    assert compute_snapshot_fingerprint(rank_changed) != snapshot.snapshot_fingerprint
    assert compute_snapshot_fingerprint(value_changed) != snapshot.snapshot_fingerprint


def test_canonical_serialization_uses_sorted_utf8_utc_and_float_hex() -> None:
    local_time = DECISION_TIME.astimezone(timezone(timedelta(hours=2)))
    payload = canonical_json_bytes(
        {"z": 0.1, "unicode": "España", "at": local_time, "count": 12}
    )
    decoded = json.loads(payload.decode("utf-8"))

    assert payload.startswith(b'{"at":')
    assert decoded == {
        "at": "2026-08-21T20:00:00Z",
        "count": 12,
        "unicode": "España",
        "z": (0.1).hex(),
    }
    assert b" " not in payload


@dataclass
class UnsupportedCanonicalValue:
    value: str


def test_canonical_serialization_rejects_naive_nonfinite_and_unsupported_values() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        canonical_json_bytes({"at": DECISION_TIME.replace(tzinfo=None)})
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises(TypeError, match="unsupported"):
        canonical_json_bytes(UnsupportedCanonicalValue("x"))


def test_explicit_fingerprint_function_uses_utc_equivalent_instants() -> None:
    values = {
        "security_id": "NORGATE:1",
        "ranking_session": RANKING_SESSION,
        "decision_time": DECISION_TIME,
        "signal_session": RANKING_SESSION,
        "signal_time": SIGNAL_TIME,
        "sma20": 110.0,
        "sma50": 100.0,
        "atr14": 5.0,
        "rsi14": 60.0,
    }
    shifted = {
        **values,
        "decision_time": DECISION_TIME.astimezone(timezone(timedelta(hours=5))),
        "signal_time": SIGNAL_TIME.astimezone(timezone(timedelta(hours=5))),
    }

    assert shifted["decision_time"].date() != RANKING_SESSION
    assert shifted["signal_time"].date() != RANKING_SESSION
    assert compute_candidate_input_fingerprint(**values) == (
        compute_candidate_input_fingerprint(**shifted)
    )


def test_batch_and_candidate_compare_decision_times_as_instants(
    make_candidate,
) -> None:
    next_date_offset = timezone(timedelta(hours=5))
    candidate = make_candidate(
        decision_time=DECISION_TIME.astimezone(next_date_offset),
        signal_time=SIGNAL_TIME.astimezone(next_date_offset),
    )

    snapshot = rank_candidates(
        ranking_session=RANKING_SESSION,
        decision_time=DECISION_TIME,
        candidates=(candidate,),
    )

    assert candidate.decision_time.date() != RANKING_SESSION
    assert candidate.decision_time == snapshot.decision_time
    assert snapshot.ranked_candidates[0].decision_time == DECISION_TIME
    assert snapshot.ranked_candidates[0].decision_time.tzinfo == next_date_offset
    assert snapshot.decision_time.tzinfo is timezone.utc
