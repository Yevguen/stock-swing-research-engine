"""Property-based final acceptance tests for Phase 13 portfolio mechanics."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stock_swing_d1.portfolio.backtest_orchestration import (
    PortfolioBacktestOrchestrator,
)
from stock_swing_d1.portfolio.portfolio_errors import (
    DuplicateEventConflictError,
    InsufficientSettledCashError,
    OutOfOrderSessionError,
    OverdueSettlementError,
    PortfolioStateError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    ExecutionSide,
    PortfolioEventKind,
    PortfolioEventReference,
    PortfolioExecutionEvent,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import (
    canonical_payload_bytes,
    canonical_payload_sha256,
    hash_execution_event,
    hash_portfolio_state,
    hash_settlement_event,
)
from stock_swing_d1.portfolio.portfolio_invariants import PortfolioInvariantChecker
from stock_swing_d1.portfolio.portfolio_persistence import (
    PortfolioArtifactReader,
    PortfolioArtifactWriter,
    PortfolioExecutionPresentation,
    hash_execution_presentations,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PortfolioSessionInput,
    PortfolioSessionSnapshot,
    PortfolioState,
    PortfolioTransitionResult,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_property_strategies import (
    BASE_SESSION,
    CENT,
    full_sell_cases,
    funded_buy_cases,
    insufficient_buy_cases,
    nonnegative_cost,
    positive_cash,
    positive_money,
    quantities,
    valid_history_cases,
)


PROPERTY_SETTINGS = settings(max_examples=30)


@PROPERTY_SETTINGS
@given(
    coefficient=st.integers(min_value=1, max_value=100000),
    places=st.integers(min_value=0, max_value=4),
)
def test_property_canonical_hashing_decimal_spelling_and_determinism(
    coefficient: int,
    places: int,
) -> None:
    value = Decimal(coefficient).scaleb(-places)
    rendered = format(value, "f")
    expanded = Decimal(
        rendered + "000" if "." in rendered else rendered + ".000"
    )
    normalized = value.normalize()

    assert value == expanded == normalized
    payloads = {
        canonical_payload_bytes(item) for item in (value, expanded, normalized)
    }
    hashes = {
        canonical_payload_sha256(item) for item in (value, expanded, normalized)
    }
    assert len(payloads) == 1
    assert len(hashes) == 1

    state = PortfolioState(settled_cash=value)
    assert canonical_payload_bytes(state) == canonical_payload_bytes(state)
    assert canonical_payload_sha256(state) == canonical_payload_sha256(state)


@PROPERTY_SETTINGS
@given(
    position_order=st.permutations((0, 1)),
    pending_order=st.permutations((0, 1)),
    fingerprint_order=st.permutations(tuple(range(6))),
)
def test_property_state_collection_normalization_is_hash_invariant(
    position_order: tuple[int, ...],
    pending_order: tuple[int, ...],
    fingerprint_order: tuple[int, ...],
) -> None:
    entry_session = date(2026, 2, 2)
    sell_session = date(2026, 2, 3)
    buys = tuple(
        PortfolioExecutionEvent(
            execution_id=f"BUY-{asset}",
            source_order_id=f"ORDER-BUY-{asset}",
            session=entry_session,
            asset_id=f"NORGATE:{asset}",
            side=ExecutionSide.BUY,
            quantity=asset,
            fill_price=Decimal("10"),
            execution_cost=Decimal("1"),
        )
        for asset in range(1, 5)
    )
    entered = PortfolioTransitionEngine.transition(
        PortfolioState(settled_cash=Decimal("10000")),
        entry_session,
        buys,
    ).resulting_state
    sells = tuple(
        PortfolioExecutionEvent(
            execution_id=f"SELL-{asset}",
            source_order_id=f"ORDER-SELL-{asset}",
            session=sell_session,
            asset_id=f"NORGATE:{asset}",
            side=ExecutionSide.SELL,
            quantity=asset,
            fill_price=Decimal("12"),
            execution_cost=Decimal("1"),
            settlement_id=f"SETTLE-{asset}",
            settlement_session=sell_session + timedelta(days=asset - 2),
        )
        for asset in (3, 4)
    )
    state = PortfolioTransitionEngine.transition(
        entered,
        sell_session,
        sells,
    ).resulting_state
    reordered = PortfolioState(
        as_of_session=state.as_of_session,
        state_version=state.state_version,
        settled_cash=state.settled_cash,
        open_positions=tuple(state.open_positions[index] for index in position_order),
        pending_settlements=tuple(
            state.pending_settlements[index] for index in pending_order
        ),
        applied_events=tuple(
            state.applied_events[index] for index in fingerprint_order
        ),
    )

    PortfolioInvariantChecker.validate_state(reordered)
    assert reordered == state
    assert hash_portfolio_state(reordered) == hash_portfolio_state(state)


@PROPERTY_SETTINGS
@given(case=funded_buy_cases())
def test_property_canonical_payload_hash_is_sensitive_to_meaningful_fields(
    case,
) -> None:
    event = case.execution
    variants = (
        event.model_copy(update={"execution_id": event.execution_id + "-X"}),
        event.model_copy(update={"source_order_id": event.source_order_id + "-X"}),
        event.model_copy(update={"session": event.session + timedelta(days=1)}),
        event.model_copy(update={"quantity": event.quantity + 1}),
        event.model_copy(update={"fill_price": event.fill_price + CENT}),
    )
    original_hash = hash_execution_event(event)
    assert all(hash_execution_event(item) != original_hash for item in variants)

    transitioned = PortfolioTransitionEngine.transition(
        case.previous_state,
        event.session,
        (event,),
    ).resulting_state
    changed_cash = transitioned.model_copy(
        update={"settled_cash": transitioned.settled_cash + CENT}
    )
    position = transitioned.open_positions[0]
    changed_position = OpenPosition(
        asset_id=position.asset_id,
        quantity=position.quantity + 1,
        entry_session=position.entry_session,
        entry_price=position.entry_price,
        entry_execution_id=position.entry_execution_id,
        entry_execution_cost=position.entry_execution_cost,
        cost_basis=(position.quantity + 1) * position.entry_price
        + position.entry_execution_cost,
    )
    changed_quantity = transitioned.model_copy(
        update={"open_positions": (changed_position,)}
    )
    assert hash_portfolio_state(changed_cash) != hash_portfolio_state(transitioned)
    assert hash_portfolio_state(changed_quantity) != hash_portfolio_state(
        transitioned
    )


@PROPERTY_SETTINGS
@given(case=full_sell_cases())
def test_property_sell_settlement_session_changes_execution_hash(case) -> None:
    changed = case.execution.model_copy(
        update={
            "settlement_session": case.execution.settlement_session
            + timedelta(days=1)
        }
    )
    assert hash_execution_event(changed) != hash_execution_event(case.execution)


@PROPERTY_SETTINGS
@given(case=funded_buy_cases())
def test_property_buy_accounting_and_fingerprint(case) -> None:
    result = PortfolioTransitionEngine.transition(
        case.previous_state,
        case.execution.session,
        (case.execution,),
    )
    state = result.resulting_state
    expected_cash = case.previous_state.settled_cash - case.required_cash

    assert state.settled_cash == expected_cash
    assert state.settled_cash >= 0
    assert len(state.open_positions) == 1
    position = state.open_positions[0]
    assert position.asset_id == case.execution.asset_id
    assert position.quantity == case.execution.quantity
    assert position.entry_price == case.execution.fill_price
    assert position.entry_execution_id == case.execution.execution_id
    assert position.cost_basis == case.required_cash
    assert AppliedEventFingerprint(
        event_kind=PortfolioEventKind.EXECUTION,
        event_id=case.execution.execution_id,
        payload_sha256=hash_execution_event(case.execution),
    ) in state.applied_events
    assert len(result.ledger_entries) == 1
    assert result.ledger_entries[0].event_type is PortfolioLedgerEventType.BUY_APPLIED


@PROPERTY_SETTINGS
@given(case=insufficient_buy_cases())
def test_property_insufficient_cash_failure_is_atomic(case) -> None:
    original = PortfolioState.model_validate(
        case.previous_state.model_dump(mode="python")
    )
    before_bytes = canonical_payload_bytes(case.previous_state)
    before_hash = hash_portfolio_state(case.previous_state)

    with pytest.raises(InsufficientSettledCashError):
        PortfolioTransitionEngine.transition(
            case.previous_state,
            case.execution.session,
            (case.execution,),
        )

    assert case.previous_state == original
    assert canonical_payload_bytes(case.previous_state) == before_bytes
    assert hash_portfolio_state(case.previous_state) == before_hash


@PROPERTY_SETTINGS
@given(
    quantity=quantities,
    fill_price=positive_money,
    execution_cost=nonnegative_cost,
    buffer=positive_money,
)
def test_property_later_unaffordable_buy_rolls_back_whole_batch(
    quantity: int,
    fill_price: Decimal,
    execution_cost: Decimal,
    buffer: Decimal,
) -> None:
    session = date(2026, 3, 2)
    first_required = Decimal(quantity) * fill_price + execution_cost
    previous = PortfolioState(settled_cash=first_required + buffer)
    first = PortfolioExecutionEvent(
        execution_id="BUY-1",
        source_order_id="ORDER-1",
        session=session,
        asset_id="NORGATE:1",
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
    )
    second = PortfolioExecutionEvent(
        execution_id="BUY-2",
        source_order_id="ORDER-2",
        session=session,
        asset_id="NORGATE:2",
        side=ExecutionSide.BUY,
        quantity=1,
        fill_price=buffer + CENT,
        execution_cost=Decimal("0"),
    )
    before_hash = hash_portfolio_state(previous)

    with pytest.raises(InsufficientSettledCashError):
        PortfolioTransitionEngine.transition(previous, session, (first, second))

    assert previous == PortfolioState(settled_cash=first_required + buffer)
    assert hash_portfolio_state(previous) == before_hash


@PROPERTY_SETTINGS
@given(case=full_sell_cases())
def test_property_full_sell_creates_exact_pending_proceeds(case) -> None:
    result = PortfolioTransitionEngine.transition(
        case.previous_state,
        case.execution.session,
        (case.execution,),
    )
    state = result.resulting_state

    assert state.settled_cash == case.previous_state.settled_cash
    assert state.open_positions == ()
    assert len(state.pending_settlements) == 1
    pending = state.pending_settlements[0]
    assert pending.amount == case.net_proceeds
    assert pending.source_execution_id == case.execution.execution_id
    assert pending.trade_session == case.execution.session
    assert pending.settlement_session == case.execution.settlement_session
    ledger = result.ledger_entries[0]
    assert ledger.event_type is PortfolioLedgerEventType.SELL_APPLIED
    assert ledger.settled_cash_delta == 0
    assert ledger.pending_cash_delta == case.net_proceeds
    assert ledger.position_quantity_after == 0
    assert ledger.source_payload_sha256 == hash_execution_event(case.execution)


@PROPERTY_SETTINGS
@given(case=full_sell_cases())
def test_property_due_settlement_applies_exactly_once_without_calendar(case) -> None:
    sold = PortfolioTransitionEngine.transition(
        case.previous_state,
        case.execution.session,
        (case.execution,),
    ).resulting_state
    pending = sold.pending_settlements[0]
    result = PortfolioTransitionEngine.transition(
        sold,
        pending.settlement_session,
        (),
    )

    assert result.resulting_state.settled_cash == sold.settled_cash + pending.amount
    assert result.resulting_state.pending_settlements == ()
    assert AppliedEventFingerprint(
        event_kind=PortfolioEventKind.SETTLEMENT,
        event_id=pending.settlement_id,
        payload_sha256=hash_settlement_event(pending),
    ) in result.resulting_state.applied_events
    assert len(result.ledger_entries) == 1
    ledger = result.ledger_entries[0]
    assert ledger.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED
    assert ledger.settled_cash_delta == pending.amount
    assert ledger.pending_cash_delta == -pending.amount


@PROPERTY_SETTINGS
@given(case=full_sell_cases(), days_late=st.integers(min_value=1, max_value=10))
def test_property_overdue_settlement_fails_engine_and_persisted_history(
    case,
    days_late: int,
) -> None:
    sold = PortfolioTransitionEngine.transition(
        case.previous_state,
        case.execution.session,
        (case.execution,),
    ).resulting_state
    requested = case.execution.settlement_session + timedelta(days=days_late)
    before_hash = hash_portfolio_state(sold)

    with pytest.raises(OverdueSettlementError):
        PortfolioTransitionEngine.transition(sold, requested, ())
    assert hash_portfolio_state(sold) == before_hash

    impossible_final = sold.model_copy(
        update={
            "as_of_session": requested,
            "state_version": sold.state_version + 1,
        }
    )
    impossible_snapshot = PortfolioSessionSnapshot(
        session=requested,
        state_version=impossible_final.state_version,
        settled_cash=impossible_final.settled_cash,
        open_position_count=0,
        pending_settlement_count=1,
        state_hash=hash_portfolio_state(impossible_final),
    )
    with pytest.raises(PortfolioStateError, match="overdue"):
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            sold,
            impossible_final,
            (),
            (impossible_snapshot,),
        )


@PROPERTY_SETTINGS
@given(case=funded_buy_cases(), replay_gap=st.integers(min_value=1, max_value=20))
def test_property_historical_replay_is_typed_and_economically_idempotent(
    case,
    replay_gap: int,
) -> None:
    first = PortfolioTransitionEngine.transition(
        case.previous_state,
        case.execution.session,
        (case.execution,),
    )
    replay_session = case.execution.session + timedelta(days=replay_gap)
    replay = PortfolioTransitionEngine.transition(
        first.resulting_state,
        replay_session,
        (case.execution,),
    )

    assert replay.ledger_entries == ()
    assert replay.newly_applied_events == ()
    assert replay.replayed_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id=case.execution.execution_id,
        ),
    )
    assert replay.resulting_state.settled_cash == first.resulting_state.settled_cash
    assert replay.resulting_state.open_positions == first.resulting_state.open_positions
    assert replay.resulting_state.pending_settlements == ()
    assert replay.resulting_state.applied_events == first.resulting_state.applied_events
    assert replay.resulting_state.as_of_session == replay_session
    assert replay.resulting_state.state_version == (
        first.resulting_state.state_version + 1
    )


@PROPERTY_SETTINGS
@given(case=funded_buy_cases(), gap=st.integers(min_value=1, max_value=20))
def test_property_duplicate_conflict_precedes_current_session_requirement(
    case,
    gap: int,
) -> None:
    first = PortfolioTransitionEngine.transition(
        case.previous_state,
        case.execution.session,
        (case.execution,),
    )
    conflicting = case.execution.model_copy(
        update={
            "source_order_id": case.execution.source_order_id + "-CONFLICT",
            "session": case.execution.session + timedelta(days=gap + 1),
        }
    )
    requested = case.execution.session + timedelta(days=gap)

    with pytest.raises(DuplicateEventConflictError):
        PortfolioTransitionEngine.transition(
            first.resulting_state,
            requested,
            (conflicting,),
        )

    same_session_conflict = case.execution.model_copy(
        update={"fill_price": case.execution.fill_price + CENT}
    )
    with pytest.raises(DuplicateEventConflictError):
        PortfolioTransitionEngine.transition(
            case.previous_state,
            case.execution.session,
            (case.execution, same_session_conflict),
        )


@settings(max_examples=24)
@given(order=st.permutations(tuple(range(4))))
def test_property_same_session_permutations_preserve_canonical_ledger_order(
    order: tuple[int, ...],
) -> None:
    entry_session = date(2026, 4, 1)
    sell_session = date(2026, 4, 2)
    combined_session = date(2026, 4, 3)
    buys = tuple(
        PortfolioExecutionEvent(
            execution_id=f"BUY-{asset}",
            source_order_id=f"ORDER-BUY-{asset}",
            session=entry_session,
            asset_id=f"NORGATE:{asset}",
            side=ExecutionSide.BUY,
            quantity=1,
            fill_price=Decimal("10"),
            execution_cost=Decimal("0"),
        )
        for asset in range(1, 5)
    )
    entered = PortfolioTransitionEngine.transition(
        PortfolioState(settled_cash=Decimal("10000")),
        entry_session,
        buys,
    ).resulting_state
    prior_sells = (
        PortfolioExecutionEvent(
            execution_id="SELL-3",
            source_order_id="ORDER-SELL-3",
            session=sell_session,
            asset_id="NORGATE:3",
            side=ExecutionSide.SELL,
            quantity=1,
            fill_price=Decimal("20"),
            execution_cost=Decimal("0"),
            settlement_id="SETTLE-B",
            settlement_session=combined_session,
        ),
        PortfolioExecutionEvent(
            execution_id="SELL-4",
            source_order_id="ORDER-SELL-4",
            session=sell_session,
            asset_id="NORGATE:4",
            side=ExecutionSide.SELL,
            quantity=1,
            fill_price=Decimal("20"),
            execution_cost=Decimal("0"),
            settlement_id="SETTLE-A",
            settlement_session=combined_session,
        ),
    )
    pending = PortfolioTransitionEngine.transition(
        entered,
        sell_session,
        prior_sells,
    ).resulting_state
    events = (
        PortfolioExecutionEvent(
            execution_id="SELL-2",
            source_order_id="ORDER-SELL-2",
            session=combined_session,
            asset_id="NORGATE:2",
            side=ExecutionSide.SELL,
            quantity=1,
            fill_price=Decimal("20"),
            execution_cost=Decimal("0"),
            settlement_id="SETTLE-2",
            settlement_session=combined_session + timedelta(days=1),
        ),
        PortfolioExecutionEvent(
            execution_id="BUY-6",
            source_order_id="ORDER-BUY-6",
            session=combined_session,
            asset_id="NORGATE:6",
            side=ExecutionSide.BUY,
            quantity=1,
            fill_price=Decimal("10"),
            execution_cost=Decimal("0"),
        ),
        PortfolioExecutionEvent(
            execution_id="SELL-1",
            source_order_id="ORDER-SELL-1",
            session=combined_session,
            asset_id="NORGATE:1",
            side=ExecutionSide.SELL,
            quantity=1,
            fill_price=Decimal("20"),
            execution_cost=Decimal("0"),
            settlement_id="SETTLE-1",
            settlement_session=combined_session + timedelta(days=1),
        ),
        PortfolioExecutionEvent(
            execution_id="BUY-5",
            source_order_id="ORDER-BUY-5",
            session=combined_session,
            asset_id="NORGATE:5",
            side=ExecutionSide.BUY,
            quantity=1,
            fill_price=Decimal("10"),
            execution_cost=Decimal("0"),
        ),
    )
    canonical = PortfolioTransitionEngine.transition(
        pending,
        combined_session,
        events,
    )
    permuted = PortfolioTransitionEngine.transition(
        pending,
        combined_session,
        tuple(events[index] for index in order),
    )

    assert permuted == canonical
    assert tuple(item.event_type for item in canonical.ledger_entries) == (
        PortfolioLedgerEventType.SETTLEMENT_APPLIED,
        PortfolioLedgerEventType.SETTLEMENT_APPLIED,
        PortfolioLedgerEventType.SELL_APPLIED,
        PortfolioLedgerEventType.SELL_APPLIED,
        PortfolioLedgerEventType.BUY_APPLIED,
        PortfolioLedgerEventType.BUY_APPLIED,
    )
    assert tuple(item.source_event_id for item in canonical.ledger_entries) == (
        "SETTLE-A",
        "SETTLE-B",
        "SELL-1",
        "SELL-2",
        "BUY-5",
        "BUY-6",
    )


@PROPERTY_SETTINGS
@given(
    offsets=st.lists(
        st.integers(min_value=1, max_value=100),
        min_size=2,
        max_size=6,
        unique=True,
    ).map(sorted)
)
def test_property_session_order_is_semantic(offsets: list[int]) -> None:
    sessions = tuple(
        PortfolioSessionInput(session=BASE_SESSION + timedelta(days=offset))
        for offset in offsets
    )
    result = PortfolioBacktestOrchestrator.run(PortfolioState(), sessions)
    assert tuple(item.session for item in result.session_results) == tuple(
        item.session for item in sessions
    )

    repeated = (sessions[0], sessions[0])
    decreasing = (sessions[-1], sessions[0])
    with pytest.raises(OutOfOrderSessionError):
        PortfolioBacktestOrchestrator.run(PortfolioState(), repeated)
    with pytest.raises(OutOfOrderSessionError):
        PortfolioBacktestOrchestrator.run(PortfolioState(), decreasing)


@PROPERTY_SETTINGS
@given(history=valid_history_cases())
def test_property_orchestrator_equals_direct_engine(history) -> None:
    expected_results = []
    current = history.initial_state
    expected_snapshots = []
    for session_input in history.sessions:
        transition = PortfolioTransitionEngine.transition(
            current,
            session_input.session,
            session_input.execution_events,
        )
        expected_results.append(transition)
        current = transition.resulting_state
        expected_snapshots.append(
            PortfolioSessionSnapshot(
                session=transition.session,
                state_version=current.state_version,
                settled_cash=current.settled_cash,
                open_position_count=len(current.open_positions),
                pending_settlement_count=len(current.pending_settlements),
                state_hash=transition.state_hash_after,
            )
        )
    result = PortfolioBacktestOrchestrator.run(
        history.initial_state,
        history.sessions,
    )

    assert result.session_results == tuple(expected_results)
    assert result.final_state == current
    assert result.ledger_entries == tuple(
        entry for item in expected_results for entry in item.ledger_entries
    )
    assert result.snapshots == tuple(expected_snapshots)
    assert result.initial_state_hash == hash_portfolio_state(history.initial_state)
    assert result.final_state_hash == expected_results[-1].state_hash_after


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
@given(history=valid_history_cases())
def test_property_persistence_round_trip_and_logical_reproducibility(
    tmp_path: Path,
    history,
) -> None:
    # tmp_path is shared by Hypothesis, but each example gets fresh nested
    # TemporaryDirectory values that are removed before the next example.
    result = PortfolioBacktestOrchestrator.run(
        history.initial_state,
        history.sessions,
    )
    with TemporaryDirectory(dir=tmp_path) as left_root, TemporaryDirectory(
        dir=tmp_path
    ) as right_root:
        left = Path(left_root) / "artifacts"
        right = Path(right_root) / "artifacts"
        left_manifest = PortfolioArtifactWriter.write(
            left,
            result,
            history.sessions,
        )
        right_manifest = PortfolioArtifactWriter.write(
            right,
            result,
            history.sessions,
        )
        left_loaded = PortfolioArtifactReader.read(left)
        right_loaded = PortfolioArtifactReader.read(right)

    assert left_manifest == right_manifest
    assert left_loaded.initial_state == result.initial_state
    assert left_loaded.final_state == result.final_state
    assert left_loaded.ledger_entries == result.ledger_entries
    assert left_loaded.session_snapshots == result.snapshots
    assert left_loaded.execution_presentations == right_loaded.execution_presentations
    expected_presentations = Counter(
        (item.session, canonical_payload_bytes(event))
        for item in history.sessions
        for event in item.execution_events
    )
    actual_presentations = Counter(
        (item.input_session, canonical_payload_bytes(item.execution_event))
        for item in left_loaded.execution_presentations
    )
    assert actual_presentations == expected_presentations


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
@given(order=st.permutations((0, 1, 2)))
def test_property_artifact_hashes_ignore_valid_intrasession_caller_order(
    tmp_path: Path,
    order: tuple[int, ...],
) -> None:
    session = date(2026, 5, 4)
    events = tuple(
        PortfolioExecutionEvent(
            execution_id=f"BUY-{asset}",
            source_order_id=f"ORDER-{asset}",
            session=session,
            asset_id=f"NORGATE:{asset}",
            side=ExecutionSide.BUY,
            quantity=asset,
            fill_price=Decimal("10"),
            execution_cost=Decimal("1"),
        )
        for asset in (1, 2, 3)
    )
    initial = PortfolioState(settled_cash=Decimal("1000"))
    canonical_inputs = (
        PortfolioSessionInput(session=session, execution_events=events),
    )
    permuted_inputs = (
        PortfolioSessionInput(
            session=session,
            execution_events=tuple(events[index] for index in order),
        ),
    )
    canonical_result = PortfolioBacktestOrchestrator.run(initial, canonical_inputs)
    permuted_result = PortfolioBacktestOrchestrator.run(initial, permuted_inputs)
    with TemporaryDirectory(dir=tmp_path) as left_root, TemporaryDirectory(
        dir=tmp_path
    ) as right_root:
        left_manifest = PortfolioArtifactWriter.write(
            Path(left_root) / "artifacts",
            canonical_result,
            canonical_inputs,
        )
        right_manifest = PortfolioArtifactWriter.write(
            Path(right_root) / "artifacts",
            permuted_result,
            permuted_inputs,
        )

    assert canonical_result == permuted_result
    assert left_manifest == right_manifest


@PROPERTY_SETTINGS
@given(case=funded_buy_cases(), gap=st.integers(min_value=1, max_value=30))
def test_property_execution_presentation_hash_tracks_session_and_multiplicity(
    case,
    gap: int,
) -> None:
    first = PortfolioExecutionPresentation(
        input_session=case.execution.session,
        execution_event=case.execution,
    )
    later = PortfolioExecutionPresentation(
        input_session=case.execution.session + timedelta(days=gap),
        execution_event=case.execution,
    )

    assert hash_execution_presentations((first,)) != hash_execution_presentations(
        (later,)
    )
    assert hash_execution_presentations((first,)) != hash_execution_presentations(
        (first, first)
    )


@PROPERTY_SETTINGS
@given(
    event_id=st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="-_",
        ),
        min_size=1,
        max_size=24,
    )
)
def test_property_event_identity_is_typed_not_bare(event_id: str) -> None:
    execution = PortfolioEventReference(
        event_kind=PortfolioEventKind.EXECUTION,
        event_id=event_id,
    )
    settlement = PortfolioEventReference(
        event_kind=PortfolioEventKind.SETTLEMENT,
        event_id=event_id,
    )

    assert execution != settlement
    assert (execution.event_kind, execution.event_id) != (
        settlement.event_kind,
        settlement.event_id,
    )
    assert "newly_applied_event_ids" not in PortfolioTransitionResult.model_fields
    assert "replayed_event_ids" not in PortfolioTransitionResult.model_fields


@PROPERTY_SETTINGS
@given(cash=positive_cash, first_gap=st.integers(1, 10), second_gap=st.integers(1, 10))
def test_property_virgin_and_processed_state_versions_increment_once(
    cash: Decimal,
    first_gap: int,
    second_gap: int,
) -> None:
    virgin = PortfolioState(settled_cash=cash + Decimal("100"))
    PortfolioInvariantChecker.validate_state(virgin)
    assert virgin.as_of_session is None
    assert virgin.state_version == 0
    assert virgin.open_positions == ()
    assert virgin.pending_settlements == ()
    assert virgin.applied_events == ()

    buy_session = BASE_SESSION + timedelta(days=first_gap)
    empty_session = buy_session + timedelta(days=second_gap)
    replay_session = empty_session + timedelta(days=1)
    buy = PortfolioExecutionEvent(
        execution_id="VERSION-BUY",
        source_order_id="VERSION-ORDER",
        session=buy_session,
        asset_id="NORGATE:1",
        side=ExecutionSide.BUY,
        quantity=1,
        fill_price=Decimal("10"),
        execution_cost=Decimal("0"),
    )
    result = PortfolioBacktestOrchestrator.run(
        virgin,
        (
            PortfolioSessionInput(session=buy_session, execution_events=(buy,)),
            PortfolioSessionInput(session=empty_session),
            PortfolioSessionInput(session=replay_session, execution_events=(buy,)),
        ),
    )
    assert tuple(
        item.resulting_state.state_version for item in result.session_results
    ) == (1, 2, 3)
    assert result.session_results[-1].replayed_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id=buy.execution_id,
        ),
    )


@PROPERTY_SETTINGS
@given(limited_cash=positive_money)
def test_property_sale_proceeds_cannot_fund_same_session_buy(
    limited_cash: Decimal,
) -> None:
    entry_session = date(2026, 6, 1)
    trade_session = date(2026, 6, 2)
    settlement_session = date(2026, 6, 3)
    entry = PortfolioExecutionEvent(
        execution_id="ENTRY",
        source_order_id="ORDER-ENTRY",
        session=entry_session,
        asset_id="NORGATE:1",
        side=ExecutionSide.BUY,
        quantity=1,
        fill_price=Decimal("10"),
        execution_cost=Decimal("0"),
    )
    entered = PortfolioTransitionEngine.transition(
        PortfolioState(settled_cash=Decimal("10") + limited_cash),
        entry_session,
        (entry,),
    ).resulting_state
    sale = PortfolioExecutionEvent(
        execution_id="EXIT",
        source_order_id="ORDER-EXIT",
        session=trade_session,
        asset_id="NORGATE:1",
        side=ExecutionSide.SELL,
        quantity=1,
        fill_price=Decimal("100"),
        execution_cost=Decimal("0"),
        settlement_id="SETTLE-EXIT",
        settlement_session=settlement_session,
    )
    unaffordable = PortfolioExecutionEvent(
        execution_id="NEXT-ENTRY",
        source_order_id="ORDER-NEXT-ENTRY",
        session=trade_session,
        asset_id="NORGATE:2",
        side=ExecutionSide.BUY,
        quantity=1,
        fill_price=limited_cash + CENT,
        execution_cost=Decimal("0"),
    )
    before_hash = hash_portfolio_state(entered)

    with pytest.raises(InsufficientSettledCashError):
        PortfolioTransitionEngine.transition(
            entered,
            trade_session,
            (sale, unaffordable),
        )
    assert hash_portfolio_state(entered) == before_hash

    sold = PortfolioTransitionEngine.transition(
        entered,
        trade_session,
        (sale,),
    ).resulting_state
    affordable = unaffordable.model_copy(update={"session": settlement_session})
    settled_and_bought = PortfolioTransitionEngine.transition(
        sold,
        settlement_session,
        (affordable,),
    )
    assert tuple(item.event_type for item in settled_and_bought.ledger_entries) == (
        PortfolioLedgerEventType.SETTLEMENT_APPLIED,
        PortfolioLedgerEventType.BUY_APPLIED,
    )
    assert settled_and_bought.resulting_state.settled_cash == Decimal("99.99")
