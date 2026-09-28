"""Deterministic scaffolding for the Phase 6C.8B acceptance matrix."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

import pytest

from stock_swing_d1.earnings.integration import PublishedEarningsRiskOverlay
from stock_swing_d1.earnings.models import EarningsStateAsOf


ASSET = "NORGATE:1001"
PROVIDER = "SYNTHETIC"


class RecordingPublishedQuery:
    """Small query double that records the overlay's exact boundary calls."""

    def __init__(
        self,
        states: Iterable[EarningsStateAsOf],
        *,
        error: Exception | None = None,
        build_id: str = "RECORDING-BUILD",
        output_sha256: str = "a" * 64,
    ) -> None:
        self._states = tuple(states)
        self._error = error
        self.build_id = build_id
        self.output_sha256 = output_sha256
        self.calls: list[dict[str, object]] = []

    def query(
        self,
        *,
        canonical_asset_id: str,
        as_of: datetime,
        decision_session: date,
        provider_name: str | None = None,
    ) -> EarningsStateAsOf:
        self.calls.append(
            {
                "canonical_asset_id": canonical_asset_id,
                "provider_name": provider_name,
                "as_of": as_of,
                "decision_session": decision_session,
            }
        )
        if self._error is not None:
            raise self._error
        if not self._states:
            raise AssertionError("recording query has no configured state")
        return self._states[min(len(self.calls) - 1, len(self._states) - 1)]


@pytest.fixture
def overlay_factory(synthetic_calendar):
    def factory(
        *states: EarningsStateAsOf,
        error: Exception | None = None,
    ) -> tuple[PublishedEarningsRiskOverlay, RecordingPublishedQuery]:
        query = RecordingPublishedQuery(states, error=error)
        overlay = PublishedEarningsRiskOverlay(
            query,
            provider_name=PROVIDER,
            trading_calendar=synthetic_calendar,
        )
        return overlay, query

    return factory
