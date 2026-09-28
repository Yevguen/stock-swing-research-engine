"""Deterministic Phase 13B.4 portfolio orchestration tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.portfolio.backtest_orchestration import (
    PortfolioBacktestOrchestrator,
)
from stock_swing_d1.portfolio.portfolio_errors import (
    InsufficientSettledCashError,
    OutOfOrderSessionError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    PortfolioEventKind,
    PortfolioEventReference,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_persistence import (
    PortfolioArtifactReader,
    PortfolioArtifactWriter,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioSessionInput,
    PortfolioSessionSnapshot,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import buy_event, initial_state, sell_event


D1 = date(2026, 9, 1)
D2 = date(2026, 9, 2)
D3 = date(2026, 9, 3)
D4 = date(2026, 9, 4)
D5 = date(2026, 9, 5)
D6 = date(2026, 9, 6)
D7 = date(2026, 9, 7)


def deterministic_scenario() -> tuple[
    PortfolioState,
    tuple[PortfolioSessionInput, ...],
]:
    initial = initial_state(Decimal("10000"))
    buy_a = buy_event(
        "BUY-A",
        session=D1,
        asset_id="NORGATE:1",
        quantity=10,
        fill_price=Decimal("100"),
        execution_cost=Decimal("2"),
    )
    sell_a = sell_event(
        "SELL-A",
        session=D3,
        asset_id="NORGATE:1",
        quantity=10,
        fill_price=Decimal("110"),
        execution_cost=Decimal("3"),
        settlement_id="SETTLE-A",
        settlement_session=D4,
    )
    buy_b = buy_event(
        "BUY-B",
        session=D4,
        asset_id="NORGATE:2",
        quantity=10,
        fill_price=Decimal("950"),
        execution_cost=Decimal("5"),
    )
    sell_b = sell_event(
        "SELL-B",
        session=D6,
        asset_id="NORGATE:2",
        quantity=10,
        fill_price=Decimal("960"),
        execution_cost=Decimal("4"),
        settlement_id="SETTLE-B",
        settlement_session=D7,
    )
    return initial, (
        PortfolioSessionInput(session=D1, execution_events=(buy_a,)),
        PortfolioSessionInput(session=D2),
        PortfolioSessionInput(session=D3, execution_events=(sell_a,)),
        PortfolioSessionInput(session=D4, execution_events=(buy_b,)),
        PortfolioSessionInput(session=D5),
        PortfolioSessionInput(session=D6, execution_events=(sell_b,)),
        PortfolioSessionInput(session=D7),
    )


def test_multi_session_scenario_has_exact_state_progression() -> None:
    initial, sessions = deterministic_scenario()

    result = PortfolioBacktestOrchestrator.run(initial, sessions)
    states = tuple(item.resulting_state for item in result.session_results)

    assert tuple(item.as_of_session for item in states) == tuple(
        item.session for item in sessions
    )
    assert tuple(item.state_version for item in states) == tuple(range(1, 8))
    assert tuple(item.settled_cash for item in states) == (
        Decimal("8998"),
        Decimal("8998"),
        Decimal("8998"),
        Decimal("590"),
        Decimal("590"),
        Decimal("590"),
        Decimal("10186"),
    )
    assert tuple(
        tuple(position.asset_id for position in item.open_positions)
        for item in states
    ) == (
        ("NORGATE:1",),
        ("NORGATE:1",),
        (),
        ("NORGATE:2",),
        ("NORGATE:2",),
        (),
        (),
    )
    assert tuple(
        tuple(pending.settlement_id for pending in item.pending_settlements)
        for item in states
    ) == (
        (),
        (),
        ("SETTLE-A",),
        (),
        (),
        ("SETTLE-B",),
        (),
    )
    assert tuple(
        tuple(entry.event_type for entry in item.ledger_entries)
        for item in result.session_results
    ) == (
        (PortfolioLedgerEventType.BUY_APPLIED,),
        (),
        (PortfolioLedgerEventType.SELL_APPLIED,),
        (
            PortfolioLedgerEventType.SETTLEMENT_APPLIED,
            PortfolioLedgerEventType.BUY_APPLIED,
        ),
        (),
        (PortfolioLedgerEventType.SELL_APPLIED,),
        (PortfolioLedgerEventType.SETTLEMENT_APPLIED,),
    )

    previous_hash = result.initial_state_hash
    for session_input, transition_result, state, snapshot in zip(
        sessions,
        result.session_results,
        states,
        result.snapshots,
        strict=True,
    ):
        state_hash = hash_portfolio_state(state)
        assert transition_result.state_hash_before == previous_hash
        assert transition_result.state_hash_after == state_hash
        assert snapshot == PortfolioSessionSnapshot(
            session=session_input.session,
            state_version=state.state_version,
            settled_cash=state.settled_cash,
            open_position_count=len(state.open_positions),
            pending_settlement_count=len(state.pending_settlements),
            state_hash=state_hash,
        )
        previous_hash = state_hash

    assert result.session_results[3].newly_applied_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.SETTLEMENT,
            event_id="SETTLE-A",
        ),
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id="BUY-B",
        ),
    )
    assert all(not item.replayed_events for item in result.session_results)
    assert result.final_state == states[-1]
    assert result.final_state_hash == previous_hash


@pytest.mark.parametrize(
    "sessions",
    [
        (
            PortfolioSessionInput(session=D1),
            PortfolioSessionInput(session=D1),
        ),
        (
            PortfolioSessionInput(session=D2),
            PortfolioSessionInput(session=D1),
        ),
    ],
    ids=("repeated", "earlier"),
)
def test_non_strict_session_sequence_is_rejected_without_sorting(
    sessions: tuple[PortfolioSessionInput, ...],
) -> None:
    with pytest.raises(OutOfOrderSessionError, match="strict chronological"):
        PortfolioBacktestOrchestrator.run(initial_state(), sessions)


def test_noncontiguous_empty_sessions_advance_without_calendar_inference() -> None:
    friday = date(2026, 8, 21)
    monday = date(2026, 8, 24)
    sessions = (
        PortfolioSessionInput(session=friday),
        PortfolioSessionInput(session=monday),
    )

    result = PortfolioBacktestOrchestrator.run(initial_state(), sessions)

    assert tuple(item.session for item in result.session_results) == (
        friday,
        monday,
    )
    assert tuple(
        item.resulting_state.state_version for item in result.session_results
    ) == (1, 2)
    assert result.ledger_entries == ()
    assert len(result.snapshots) == 2


def test_orchestrator_results_equal_direct_transition_engine_results() -> None:
    initial, sessions = deterministic_scenario()
    expected = []
    current = initial
    for session_input in sessions:
        transition = PortfolioTransitionEngine.transition(
            current,
            session_input.session,
            session_input.execution_events,
        )
        expected.append(transition)
        current = transition.resulting_state

    result = PortfolioBacktestOrchestrator.run(initial, sessions)

    assert result.session_results == tuple(expected)


def test_failure_propagates_and_stops_before_later_sessions(monkeypatch) -> None:
    first = date(2026, 10, 1)
    failing = date(2026, 10, 2)
    later = date(2026, 10, 3)
    sessions = (
        PortfolioSessionInput(session=first),
        PortfolioSessionInput(
            session=failing,
            execution_events=(
                buy_event(
                    "TOO-EXPENSIVE",
                    session=failing,
                    fill_price=Decimal("2000"),
                    execution_cost=Decimal("0"),
                ),
            ),
        ),
        PortfolioSessionInput(session=later),
    )
    calls = []
    real_transition = PortfolioTransitionEngine.transition

    def tracking_transition(
        previous_state, session, execution_events, dividend_evidence=()
    ):
        calls.append(session)
        return real_transition(
            previous_state, session, execution_events, dividend_evidence
        )

    monkeypatch.setattr(
        PortfolioTransitionEngine,
        "transition",
        tracking_transition,
    )
    initial = initial_state()

    with pytest.raises(InsufficientSettledCashError):
        PortfolioBacktestOrchestrator.run(initial, sessions)

    assert calls == [first, failing]
    assert initial == initial_state()


def test_same_inputs_are_deterministic_and_ledger_aggregation_is_exact() -> None:
    initial, sessions = deterministic_scenario()

    first = PortfolioBacktestOrchestrator.run(initial, sessions)
    second = PortfolioBacktestOrchestrator.run(initial, sessions)

    assert first == second
    assert first.ledger_entries == tuple(
        entry
        for session_result in first.session_results
        for entry in session_result.ledger_entries
    )
    assert tuple(entry.source_event_id for entry in first.ledger_entries) == (
        "BUY-A",
        "SELL-A",
        "SETTLE-A",
        "BUY-B",
        "SELL-B",
        "SETTLE-B",
    )


def test_zero_session_run_has_frozen_empty_aggregate_contract() -> None:
    initial = initial_state(Decimal("10000"))

    result = PortfolioBacktestOrchestrator.run(initial, ())

    assert result.session_results == ()
    assert result.ledger_entries == ()
    assert result.snapshots == ()
    assert result.final_state == initial
    assert result.initial_state_hash == hash_portfolio_state(initial)
    assert result.final_state_hash == result.initial_state_hash


def test_orchestrated_result_round_trips_real_persistence(tmp_path) -> None:
    initial, sessions = deterministic_scenario()
    result = PortfolioBacktestOrchestrator.run(initial, sessions)
    artifact_directory = tmp_path / "orchestrated-artifacts"

    manifest = PortfolioArtifactWriter.write(
        artifact_directory,
        result,
        sessions,
    )
    loaded = PortfolioArtifactReader.read(artifact_directory)

    assert loaded.manifest == manifest
    assert loaded.initial_state == result.initial_state
    assert loaded.final_state == result.final_state
    assert loaded.ledger_entries == result.ledger_entries
    assert loaded.session_snapshots == result.snapshots
    assert manifest.initial_state_content_sha256 == result.initial_state_hash
    assert manifest.final_state_content_sha256 == result.final_state_hash
