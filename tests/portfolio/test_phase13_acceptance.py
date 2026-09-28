"""Frozen end-to-end golden acceptance contract for Phase 13 v0.1."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from stock_swing_d1.portfolio.backtest_orchestration import (
    PortfolioBacktestOrchestrator,
)
from stock_swing_d1.portfolio.portfolio_events import (
    PortfolioEventKind,
    PortfolioEventReference,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_persistence import (
    PortfolioArtifactReader,
    PortfolioArtifactWriter,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioSessionInput,
    PortfolioState,
)
from tests.portfolio.phase13_helpers import buy_event, sell_event


D1 = date(2026, 11, 2)
D2 = date(2026, 11, 3)
D3 = date(2026, 11, 4)
D4 = date(2026, 11, 5)
D5 = date(2026, 11, 6)
D6 = date(2026, 11, 7)
D7 = date(2026, 11, 8)
D8 = date(2026, 11, 9)

EXPECTED_INITIAL_HASH = (
    "2e71074fc09b0b985cf13e806bdda0802cc1b847dad154100e9b41e968ceb6e5"
)
EXPECTED_D1_HASH = (
    "451056c894eb41032c374118ce0fb4154aae4e7ac4e48505318e1703a406bda1"
)
EXPECTED_D3_REPLAY_HASH = (
    "44ff18f80d36d1089520eef64afbc6aeb7594877051cb46d17fa07bb719dfb26"
)
EXPECTED_D5_HASH = (
    "b41157c09d9f9ec8c206d016651995751c721967008a81cd8052798bec4e9a97"
)
EXPECTED_FINAL_HASH = (
    "63dd80b00e780160954c3d6d97f6ffd7cecd9e2249bee1048ae1bffd7c516821"
)
EXPECTED_PRESENTATIONS_HASH = (
    "c43e4acceb54559c79f2e5e095f533789d3af8865eb488ceff1c5abed6d1e85d"
)
EXPECTED_LEDGER_HASH = (
    "34ad2aeb39a900feec9bbf9fdcbd9b35a63ab5f19f3cd9f816acbbcd192f5251"
)
EXPECTED_SNAPSHOTS_HASH = (
    "ff966f29e64ea64e282f24162c64f8f34b89c68c114abb6c4c30f1db02ac5423"
)


def golden_scenario() -> tuple[
    PortfolioState,
    tuple[PortfolioSessionInput, ...],
]:
    initial = PortfolioState(settled_cash=Decimal("10000"))
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
        session=D4,
        asset_id="NORGATE:1",
        quantity=10,
        fill_price=Decimal("110"),
        execution_cost=Decimal("3"),
        settlement_id="SETTLE-A",
        settlement_session=D5,
    )
    buy_b = buy_event(
        "BUY-B",
        session=D5,
        asset_id="NORGATE:2",
        quantity=10,
        fill_price=Decimal("950"),
        execution_cost=Decimal("5"),
    )
    sell_b = sell_event(
        "SELL-B",
        session=D7,
        asset_id="NORGATE:2",
        quantity=10,
        fill_price=Decimal("960"),
        execution_cost=Decimal("4"),
        settlement_id="SETTLE-B",
        settlement_session=D8,
    )
    return initial, (
        PortfolioSessionInput(session=D1, execution_events=(buy_a,)),
        PortfolioSessionInput(session=D2),
        PortfolioSessionInput(session=D3, execution_events=(buy_a,)),
        PortfolioSessionInput(session=D4, execution_events=(sell_a,)),
        PortfolioSessionInput(session=D5, execution_events=(buy_b,)),
        PortfolioSessionInput(session=D6),
        PortfolioSessionInput(session=D7, execution_events=(sell_b,)),
        PortfolioSessionInput(session=D8),
    )


def test_phase13_v01_golden_acceptance(tmp_path: Path) -> None:
    initial, sessions = golden_scenario()
    first = PortfolioBacktestOrchestrator.run(initial, sessions)
    second = PortfolioBacktestOrchestrator.run(initial, sessions)

    assert first == second
    assert tuple(
        item.resulting_state.settled_cash for item in first.session_results
    ) == (
        Decimal("8998"),
        Decimal("8998"),
        Decimal("8998"),
        Decimal("8998"),
        Decimal("590"),
        Decimal("590"),
        Decimal("590"),
        Decimal("10186"),
    )
    assert first.final_state.settled_cash == Decimal("10186")
    assert first.final_state.state_version == 8
    assert first.final_state.as_of_session == D8
    assert first.final_state.open_positions == ()
    assert first.final_state.pending_settlements == ()
    assert tuple(item.event_type for item in first.ledger_entries) == (
        PortfolioLedgerEventType.BUY_APPLIED,
        PortfolioLedgerEventType.SELL_APPLIED,
        PortfolioLedgerEventType.SETTLEMENT_APPLIED,
        PortfolioLedgerEventType.BUY_APPLIED,
        PortfolioLedgerEventType.SELL_APPLIED,
        PortfolioLedgerEventType.SETTLEMENT_APPLIED,
    )
    assert tuple(item.source_event_id for item in first.ledger_entries) == (
        "BUY-A",
        "SELL-A",
        "SETTLE-A",
        "BUY-B",
        "SELL-B",
        "SETTLE-B",
    )
    assert first.session_results[0].newly_applied_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id="BUY-A",
        ),
    )
    assert first.session_results[2].replayed_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id="BUY-A",
        ),
    )
    assert first.session_results[4].newly_applied_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.SETTLEMENT,
            event_id="SETTLE-A",
        ),
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id="BUY-B",
        ),
    )
    assert first.session_results[7].newly_applied_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.SETTLEMENT,
            event_id="SETTLE-B",
        ),
    )

    assert first.initial_state_hash == EXPECTED_INITIAL_HASH
    assert first.session_results[0].state_hash_after == EXPECTED_D1_HASH
    assert first.session_results[2].state_hash_after == EXPECTED_D3_REPLAY_HASH
    assert first.session_results[4].state_hash_after == EXPECTED_D5_HASH
    assert first.final_state_hash == EXPECTED_FINAL_HASH

    first_manifest = PortfolioArtifactWriter.write(
        tmp_path / "golden-first",
        first,
        sessions,
    )
    second_manifest = PortfolioArtifactWriter.write(
        tmp_path / "golden-second",
        second,
        sessions,
    )
    loaded = PortfolioArtifactReader.read(tmp_path / "golden-first")

    assert first_manifest == second_manifest
    assert loaded.initial_state == first.initial_state
    assert loaded.final_state == first.final_state
    assert loaded.ledger_entries == first.ledger_entries
    assert loaded.session_snapshots == first.snapshots
    assert first_manifest.initial_state_content_sha256 == EXPECTED_INITIAL_HASH
    assert first_manifest.final_state_content_sha256 == EXPECTED_FINAL_HASH
    assert (
        first_manifest.execution_presentations_content_sha256
        == EXPECTED_PRESENTATIONS_HASH
    )
    assert first_manifest.portfolio_ledger_content_sha256 == EXPECTED_LEDGER_HASH
    assert (
        first_manifest.session_snapshots_content_sha256
        == EXPECTED_SNAPSHOTS_HASH
    )
    assert first_manifest.execution_presentation_count == 5
    assert first_manifest.portfolio_ledger_entry_count == 6
    assert first_manifest.session_snapshot_count == 8
