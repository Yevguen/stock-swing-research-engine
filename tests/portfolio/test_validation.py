"""Fail-closed validation tests for Phase 12 inputs and dependencies."""

from datetime import date, datetime, timezone
from math import inf, nan

import pytest

from stock_swing_d1.portfolio import (
    OpenPosition,
    PortfolioAllocationService,
    PortfolioAllocationValidationError,
    PortfolioCandidate,
    PortfolioSnapshot,
)
from stock_swing_d1.strategy.baseline import BaselineSignalAction


@pytest.mark.parametrize("security_id", ["", " ", " BAD", "BAD "])
def test_open_position_requires_canonical_security_id(security_id) -> None:
    with pytest.raises(PortfolioAllocationValidationError) as raised:
        OpenPosition(
            security_id=security_id,
            symbol="GOOD",
            shares=1,
            entry_session=date(2026, 8, 1),
        )

    assert raised.value.code == "INVALID_OPEN_POSITION"


@pytest.mark.parametrize("shares", [0, -1, 1.5, True])
def test_open_position_requires_positive_whole_shares(shares) -> None:
    with pytest.raises(PortfolioAllocationValidationError) as raised:
        OpenPosition(
            security_id="SECURITY:1",
            symbol="GOOD",
            shares=shares,
            entry_session=date(2026, 8, 1),
        )

    assert raised.value.code == "INVALID_OPEN_POSITION"


@pytest.mark.parametrize("equity", [0, -1, nan, inf, -inf, True])
def test_snapshot_rejects_invalid_equity(equity) -> None:
    with pytest.raises(PortfolioAllocationValidationError) as raised:
        PortfolioSnapshot(
            allocation_session=date(2026, 8, 14),
            decision_time=datetime(2026, 8, 14, 20, tzinfo=timezone.utc),
            portfolio_equity=equity,
            cash_available=0,
        )

    assert raised.value.code == "INVALID_PORTFOLIO_EQUITY"


@pytest.mark.parametrize("cash", [-1, nan, inf, -inf, True, 10_001])
def test_snapshot_rejects_invalid_cash(cash) -> None:
    with pytest.raises(PortfolioAllocationValidationError) as raised:
        PortfolioSnapshot(
            allocation_session=date(2026, 8, 14),
            decision_time=datetime(2026, 8, 14, 20, tzinfo=timezone.utc),
            portfolio_equity=10_000,
            cash_available=cash,
        )

    assert raised.value.code == "INVALID_PORTFOLIO_CASH"


def test_snapshot_requires_aware_same_session_decision_time() -> None:
    with pytest.raises(PortfolioAllocationValidationError) as naive:
        PortfolioSnapshot(
            allocation_session=date(2026, 8, 14),
            decision_time=datetime(2026, 8, 14, 20),
            portfolio_equity=10_000,
            cash_available=10_000,
        )
    with pytest.raises(PortfolioAllocationValidationError) as wrong_day:
        PortfolioSnapshot(
            allocation_session=date(2026, 8, 14),
            decision_time=datetime(2026, 8, 15, 1, tzinfo=timezone.utc),
            portfolio_equity=10_000,
            cash_available=10_000,
        )

    assert naive.value.code == "INVALID_PORTFOLIO_SNAPSHOT"
    assert wrong_day.value.code == "INVALID_PORTFOLIO_SNAPSHOT"


def test_snapshot_rejects_future_and_duplicate_open_positions(
    make_open_position
) -> None:
    future = make_open_position(entry_session=date(2026, 8, 15))
    duplicate_a = make_open_position("OPEN:10")
    duplicate_b = make_open_position("OPEN:10", symbol="OTHER")

    with pytest.raises(PortfolioAllocationValidationError) as future_error:
        PortfolioSnapshot(
            allocation_session=date(2026, 8, 14),
            decision_time=datetime(2026, 8, 14, 20, tzinfo=timezone.utc),
            portfolio_equity=10_000,
            cash_available=10_000,
            open_positions=(future,),
        )
    with pytest.raises(PortfolioAllocationValidationError) as duplicate_error:
        PortfolioSnapshot(
            allocation_session=date(2026, 8, 14),
            decision_time=datetime(2026, 8, 14, 20, tzinfo=timezone.utc),
            portfolio_equity=10_000,
            cash_available=10_000,
            open_positions=(duplicate_a, duplicate_b),
        )

    assert future_error.value.code == "FUTURE_OPEN_POSITION"
    assert duplicate_error.value.code == "DUPLICATE_OPEN_SECURITY_ID"


def test_overfull_portfolio_fails_closed_without_candidate_decisions(
    make_portfolio, make_open_position, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service
    snapshot = make_portfolio(
        open_positions=tuple(
            make_open_position(f"OPEN:{number}") for number in range(5)
        )
    )
    object.__setattr__(
        snapshot,
        "open_positions",
        snapshot.open_positions + (make_open_position("OPEN:6"),),
    )

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        service.allocate_candidates(
            portfolio=snapshot, candidates=(make_candidate(),)
        )

    assert raised.value.code == "TOO_MANY_OPEN_POSITIONS"
    assert sizing_service.calls == []


def test_duplicate_candidate_ids_fail_before_any_sizing(
    make_portfolio, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service
    first = make_candidate("NORGATE:100", symbol="ONE")
    second = make_candidate("NORGATE:100", symbol="TWO")

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        service.allocate_candidates(
            portfolio=make_portfolio(), candidates=(first, second)
        )

    assert raised.value.code == "DUPLICATE_CANDIDATE_SECURITY_ID"
    assert sizing_service.calls == []


def test_entire_batch_is_preflighted_before_sizing_starts(
    make_portfolio, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service
    valid = make_candidate("NORGATE:100")
    invalid = make_candidate("NORGATE:200")
    object.__setattr__(
        invalid,
        "signal_bar",
        invalid.signal_bar.model_copy(update={"security_id": "OTHER"}),
    )

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        service.allocate_candidates(
            portfolio=make_portfolio(), candidates=(valid, invalid)
        )

    assert raised.value.code == "CANDIDATE_IDENTITY_MISMATCH"
    assert sizing_service.calls == []


def test_candidate_constructor_rejects_identity_mismatch(make_candidate) -> None:
    candidate = make_candidate()
    object.__setattr__(candidate.pending_entry, "symbol", "OTHER")

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        PortfolioCandidate(
            signal=candidate.signal,
            pending_entry=candidate.pending_entry,
            signal_bar=candidate.signal_bar,
        )

    assert raised.value.code == "CANDIDATE_IDENTITY_MISMATCH"


def test_candidate_session_must_equal_allocation_session(
    make_portfolio, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service
    candidate = make_candidate(
        signal_session=date(2026, 8, 13),
        planned_entry_session=date(2026, 8, 14),
    )

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        service.allocate_candidates(
            portfolio=make_portfolio(), candidates=(candidate,)
        )

    assert raised.value.code == "CANDIDATE_SESSION_MISMATCH"
    assert sizing_service.calls == []


def test_candidate_signal_time_cannot_postdate_portfolio_decision_time(
    make_portfolio, make_candidate, recording_service
) -> None:
    service, sizing_service = recording_service
    portfolio = make_portfolio()
    candidate = make_candidate()
    future_signal_time = portfolio.decision_time.replace(minute=10)
    object.__setattr__(candidate.signal, "signal_time", future_signal_time)
    object.__setattr__(
        candidate.pending_entry, "signal_time", future_signal_time
    )

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        service.allocate_candidates(
            portfolio=portfolio, candidates=(candidate,)
        )

    assert raised.value.code == "CANDIDATE_SIGNAL_AFTER_PORTFOLIO_DECISION"
    assert sizing_service.calls == []


def test_phase11_validation_failure_is_wrapped_and_preserves_cause(
    make_portfolio, make_candidate
) -> None:
    candidate = make_candidate()
    object.__setattr__(candidate.signal, "action", BaselineSignalAction.NO_SIGNAL)

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        PortfolioAllocationService().allocate_candidates(
            portfolio=make_portfolio(), candidates=(candidate,)
        )

    assert raised.value.code == "INVALID_POSITION_SIZING_INPUT"
    assert raised.value.__cause__ is not None
    assert getattr(raised.value.__cause__, "code") == "INVALID_SIGNAL"


def test_invalid_phase11_result_fails_closed(make_portfolio, make_candidate) -> None:
    class InvalidSizingService:
        def size_pending_entry(self, **kwargs):
            return object()

    service = PortfolioAllocationService(
        position_sizing_service=InvalidSizingService()
    )

    with pytest.raises(PortfolioAllocationValidationError) as raised:
        service.allocate_candidates(
            portfolio=make_portfolio(), candidates=(make_candidate(),)
        )

    assert raised.value.code == "INVALID_POSITION_SIZING_RESULT"


def test_non_sequence_candidate_batch_fails_closed(make_portfolio) -> None:
    with pytest.raises(PortfolioAllocationValidationError) as raised:
        PortfolioAllocationService().allocate_candidates(
            portfolio=make_portfolio(), candidates=iter(())
        )

    assert raised.value.code == "INVALID_CANDIDATE_BATCH"
