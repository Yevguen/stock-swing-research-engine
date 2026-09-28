"""Exact-value fixtures for Phase 14 candidate ranking."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_swing_d1.ranking import (
    RankingCandidate,
    compute_candidate_input_fingerprint,
    compute_snapshot_fingerprint,
)


RANKING_SESSION = date(2026, 8, 21)
DECISION_TIME = datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc)
SIGNAL_TIME = datetime(2026, 8, 21, 19, 59, tzinfo=timezone.utc)


@pytest.fixture
def make_candidate():
    def factory(**overrides: object) -> RankingCandidate:
        values: dict[str, object] = {
            "security_id": "NORGATE:100",
            "ranking_session": RANKING_SESSION,
            "decision_time": DECISION_TIME,
            "signal_session": RANKING_SESSION,
            "signal_time": SIGNAL_TIME,
            "sma20": 110.0,
            "sma50": 100.0,
            "atr14": 5.0,
            "rsi14": 60.0,
        }
        values.update(overrides)
        input_fingerprint = values.pop("input_fingerprint", None)
        if input_fingerprint is None:
            input_fingerprint = compute_candidate_input_fingerprint(**values)
        return RankingCandidate(
            **values,
            input_fingerprint=input_fingerprint,
        )

    return factory


def rehash_snapshot(snapshot):
    return snapshot.model_copy(
        update={"snapshot_fingerprint": compute_snapshot_fingerprint(snapshot)}
    )
