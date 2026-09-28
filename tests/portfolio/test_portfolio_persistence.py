"""Real JSON/PyArrow persistence tests for Phase 13B.3."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stock_swing_d1.portfolio import portfolio_persistence as persistence
from stock_swing_d1.portfolio.portfolio_errors import (
    ArtifactHashMismatchError,
    PortfolioPersistenceError,
    PortfolioStateError,
    SchemaVersionError,
    StateHashMismatchError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    PortfolioExecutionEvent,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import (
    canonical_payload_bytes,
    hash_portfolio_state,
)
from stock_swing_d1.portfolio.portfolio_persistence import (
    DIVIDEND_EVIDENCE_FILENAME,
    DIVIDEND_LEDGER_FILENAME,
    DIVIDEND_OUTCOMES_FILENAME,
    EXECUTION_EVENTS_FILENAME,
    EXECUTION_PRESENTATION_ARROW_SCHEMA,
    FINAL_STATE_FILENAME,
    INITIAL_STATE_FILENAME,
    MANIFEST_FILENAME,
    MONEY_ARROW_TYPE,
    PORTFOLIO_LEDGER_ARROW_SCHEMA,
    PORTFOLIO_LEDGER_FILENAME,
    SESSION_SNAPSHOT_ARROW_SCHEMA,
    SESSION_SNAPSHOTS_FILENAME,
    PortfolioArtifactReader,
    PortfolioArtifactWriter,
    PortfolioExecutionPresentation,
    hash_execution_presentations,
    hash_portfolio_ledger,
    hash_session_snapshots,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioBacktestResult,
    PortfolioSessionInput,
    PortfolioSessionSnapshot,
    PortfolioState,
    PortfolioTransitionResult,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import (
    buy_event,
    initial_state,
    sell_event,
    state_with_position,
)


def snapshot(result: PortfolioTransitionResult) -> PortfolioSessionSnapshot:
    state = result.resulting_state
    return PortfolioSessionSnapshot(
        session=result.session,
        state_version=state.state_version,
        settled_cash=state.settled_cash,
        open_position_count=len(state.open_positions),
        pending_settlement_count=len(state.pending_settlements),
        state_hash=result.state_hash_after,
    )


def backtest_result(
    initial: PortfolioState,
    *results: PortfolioTransitionResult,
) -> PortfolioBacktestResult:
    final = results[-1].resulting_state if results else initial
    return PortfolioBacktestResult(
        initial_state=initial,
        final_state=final,
        session_results=results,
        ledger_entries=tuple(
            entry for result in results for entry in result.ledger_entries
        ),
        snapshots=tuple(snapshot(result) for result in results),
        initial_state_hash=hash_portfolio_state(initial),
        final_state_hash=hash_portfolio_state(final),
    )


def one_buy_history(
    *,
    event: PortfolioExecutionEvent | None = None,
) -> tuple[PortfolioBacktestResult, tuple[PortfolioSessionInput, ...]]:
    execution = event or buy_event()
    initial = initial_state()
    result = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    return backtest_result(initial, result), (
        PortfolioSessionInput(
            session=execution.session,
            execution_events=(execution,),
        ),
    )


def write_artifact(
    tmp_path: Path,
    result: PortfolioBacktestResult,
    inputs: tuple[PortfolioSessionInput, ...],
    name: str = "artifacts",
) -> Path:
    directory = tmp_path / name
    PortfolioArtifactWriter.write(directory, result, inputs)
    return directory


def manifest_payload(directory: Path) -> dict[str, object]:
    return json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def rewrite_manifest(directory: Path, **updates: object) -> None:
    payload = manifest_payload(directory)
    payload.update(updates)
    (directory / MANIFEST_FILENAME).write_bytes(canonical_payload_bytes(payload))


def rewrite_presentations(
    directory: Path,
    presentations: tuple[PortfolioExecutionPresentation, ...],
    *,
    update_manifest: bool,
) -> None:
    table = pa.Table.from_pylist(
        [persistence._presentation_row(item) for item in presentations],
        schema=EXECUTION_PRESENTATION_ARROW_SCHEMA,
    )
    pq.write_table(table, directory / EXECUTION_EVENTS_FILENAME, compression="zstd")
    if update_manifest:
        rewrite_manifest(
            directory,
            execution_presentations_content_sha256=(
                hash_execution_presentations(presentations)
            ),
            execution_presentation_count=len(presentations),
        )


def rewrite_ledger(
    directory: Path,
    ledger: tuple,
    *,
    update_manifest: bool,
) -> None:
    table = pa.Table.from_pylist(
        [persistence._ledger_row(item) for item in ledger],
        schema=PORTFOLIO_LEDGER_ARROW_SCHEMA,
    )
    pq.write_table(table, directory / PORTFOLIO_LEDGER_FILENAME, compression="zstd")
    if update_manifest:
        rewrite_manifest(
            directory,
            portfolio_ledger_content_sha256=hash_portfolio_ledger(ledger),
            portfolio_ledger_entry_count=len(ledger),
        )


def rewrite_snapshots(
    directory: Path,
    snapshots: tuple[PortfolioSessionSnapshot, ...],
    *,
    update_manifest: bool,
) -> None:
    table = pa.Table.from_pylist(
        [persistence._snapshot_row(item) for item in snapshots],
        schema=SESSION_SNAPSHOT_ARROW_SCHEMA,
    )
    pq.write_table(table, directory / SESSION_SNAPSHOTS_FILENAME, compression="zstd")
    if update_manifest:
        rewrite_manifest(
            directory,
            session_snapshots_content_sha256=hash_session_snapshots(snapshots),
            session_snapshot_count=len(snapshots),
        )


def rewrite_final_state_and_dependencies(
    directory: Path,
    result: PortfolioBacktestResult,
    final_state: PortfolioState,
) -> None:
    """Keep ordinary final/snapshot/hash checks consistent after a tamper."""

    final_hash = hash_portfolio_state(final_state)
    (directory / FINAL_STATE_FILENAME).write_bytes(
        canonical_payload_bytes(final_state)
    )
    rewrite_manifest(
        directory,
        final_state_content_sha256=final_hash,
    )
    final_snapshot = result.snapshots[-1].model_copy(
        update={
            "session": final_state.as_of_session,
            "state_version": final_state.state_version,
            "settled_cash": final_state.settled_cash,
            "open_position_count": len(final_state.open_positions),
            "pending_settlement_count": len(final_state.pending_settlements),
            "state_hash": final_hash,
        }
    )
    rewrite_snapshots(
        directory,
        (*result.snapshots[:-1], final_snapshot),
        update_manifest=True,
    )
    final_session = result.snapshots[-1].session
    ledger = tuple(
        entry.model_copy(update={"state_hash_after": final_hash})
        if entry.session == final_session
        else entry
        for entry in result.ledger_entries
    )
    rewrite_ledger(directory, ledger, update_manifest=True)


def test_state_json_round_trip_is_exact_canonical_and_validated(tmp_path: Path) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)

    loaded = PortfolioArtifactReader.read(directory)

    assert loaded.initial_state == result.initial_state
    assert loaded.final_state == result.final_state
    assert hash_portfolio_state(loaded.final_state) == result.final_state_hash
    assert (directory / INITIAL_STATE_FILENAME).read_bytes() == (
        canonical_payload_bytes(result.initial_state)
    )
    assert (directory / FINAL_STATE_FILENAME).read_bytes() == (
        canonical_payload_bytes(result.final_state)
    )
    assert b'"settled_cash":"979"' in (
        directory / FINAL_STATE_FILENAME
    ).read_bytes()


def test_funded_virgin_state_round_trip(tmp_path: Path) -> None:
    virgin = PortfolioState(settled_cash=Decimal("10000.123456789012345678"))
    result = backtest_result(virgin)
    directory = write_artifact(tmp_path, result, ())

    loaded = PortfolioArtifactReader.read(directory)

    assert loaded.initial_state == virgin
    assert loaded.final_state == virgin
    assert loaded.initial_state.settled_cash == Decimal(
        "10000.123456789012345678"
    )


def test_empty_successful_session_persists_snapshot_without_events(
    tmp_path: Path,
) -> None:
    initial = initial_state()
    session = date(2026, 8, 18)
    transition = PortfolioTransitionEngine.transition(initial, session, ())
    result = backtest_result(initial, transition)
    inputs = (PortfolioSessionInput(session=session),)

    loaded = PortfolioArtifactReader.read(write_artifact(tmp_path, result, inputs))

    assert loaded.execution_presentations == ()
    assert loaded.ledger_entries == ()
    assert len(loaded.session_snapshots) == 1
    assert loaded.final_state.state_version == 1
    assert loaded.final_state.as_of_session == session


def test_execution_presentation_round_trip_preserves_replay_session_and_payload(
    tmp_path: Path,
) -> None:
    execution = buy_event(
        fill_price=Decimal("10.123456789012345678"),
        execution_cost=Decimal("0.000000000000000001"),
    )
    initial = initial_state()
    first = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    replay_session = date(2026, 8, 20)
    replay = PortfolioTransitionEngine.transition(
        first.resulting_state, replay_session, (execution,)
    )
    result = backtest_result(initial, first, replay)
    inputs = (
        PortfolioSessionInput(
            session=execution.session, execution_events=(execution,)
        ),
        PortfolioSessionInput(
            session=replay_session, execution_events=(execution,)
        ),
    )
    directory = write_artifact(tmp_path, result, inputs)

    loaded = PortfolioArtifactReader.read(directory)

    assert len(loaded.execution_presentations) == 2
    original, historical_replay = loaded.execution_presentations
    assert original.input_session == execution.session
    assert historical_replay.input_session == replay_session
    assert historical_replay.execution_event.session == execution.session
    assert historical_replay.execution_event == execution
    assert historical_replay.execution_event.fill_price == Decimal(
        "10.123456789012345678"
    )


def test_sell_settlement_fields_round_trip_exactly(tmp_path: Path) -> None:
    initial = state_with_position()
    execution = sell_event(
        fill_price=Decimal("20.123456789012345678"),
        execution_cost=Decimal("0.000000000000000001"),
    )
    transition = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    result = backtest_result(initial, transition)
    inputs = (
        PortfolioSessionInput(
            session=execution.session, execution_events=(execution,)
        ),
    )
    directory = write_artifact(tmp_path, result, inputs)

    loaded = PortfolioArtifactReader.read(directory)
    restored = loaded.execution_presentations[0].execution_event

    assert restored == execution
    assert restored.settlement_id == execution.settlement_id
    assert restored.settlement_session == execution.settlement_session


def test_internal_settlement_ledger_and_fingerprint_round_trip(tmp_path: Path) -> None:
    initial = state_with_position()
    execution = sell_event()
    sale = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    settlement_session = sale.resulting_state.pending_settlements[0].settlement_session
    settlement = PortfolioTransitionEngine.transition(
        sale.resulting_state, settlement_session, ()
    )
    result = backtest_result(initial, sale, settlement)
    inputs = (
        PortfolioSessionInput(
            session=execution.session,
            execution_events=(execution,),
        ),
        PortfolioSessionInput(session=settlement_session),
    )

    loaded = PortfolioArtifactReader.read(write_artifact(tmp_path, result, inputs))

    assert [item.event_type for item in loaded.ledger_entries] == [
        PortfolioLedgerEventType.SELL_APPLIED,
        PortfolioLedgerEventType.SETTLEMENT_APPLIED,
    ]
    assert loaded.final_state.pending_settlements == ()


def test_parquet_accounting_columns_are_arrow_decimals(tmp_path: Path) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)

    execution_schema = pq.ParquetFile(
        directory / EXECUTION_EVENTS_FILENAME
    ).schema_arrow
    ledger_schema = pq.ParquetFile(
        directory / PORTFOLIO_LEDGER_FILENAME
    ).schema_arrow
    snapshot_schema = pq.ParquetFile(
        directory / SESSION_SNAPSHOTS_FILENAME
    ).schema_arrow

    assert execution_schema.equals(
        EXECUTION_PRESENTATION_ARROW_SCHEMA, check_metadata=False
    )
    assert ledger_schema.equals(PORTFOLIO_LEDGER_ARROW_SCHEMA, check_metadata=False)
    assert snapshot_schema.equals(
        SESSION_SNAPSHOT_ARROW_SCHEMA, check_metadata=False
    )
    for field in ("fill_price", "execution_cost"):
        assert execution_schema.field(field).type == MONEY_ARROW_TYPE
    for field in (
        "settled_cash_delta",
        "pending_cash_delta",
        "settled_cash_after",
    ):
        assert ledger_schema.field(field).type == MONEY_ARROW_TYPE
    assert snapshot_schema.field("settled_cash").type == MONEY_ARROW_TYPE
    assert not any(
        pa.types.is_floating(field.type)
        for schema in (execution_schema, ledger_schema, snapshot_schema)
        for field in schema
    )


def test_valid_new_execution_and_historical_replay_are_accepted(
    tmp_path: Path,
) -> None:
    execution = buy_event()
    initial = initial_state()
    first = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    second_session = date(2026, 8, 20)
    replay = PortfolioTransitionEngine.transition(
        first.resulting_state, second_session, (execution,)
    )
    result = backtest_result(initial, first, replay)
    inputs = (
        PortfolioSessionInput(
            session=execution.session, execution_events=(execution,)
        ),
        PortfolioSessionInput(
            session=second_session, execution_events=(execution,)
        ),
    )

    loaded = PortfolioArtifactReader.read(write_artifact(tmp_path, result, inputs))

    assert len(loaded.execution_presentations) == 2
    assert len(loaded.ledger_entries) == 1


def test_replay_predating_artifact_window_is_proven_from_initial_state(
    tmp_path: Path,
) -> None:
    execution = buy_event()
    base = PortfolioTransitionEngine.transition(
        initial_state(), execution.session, (execution,)
    )
    replay_session = date(2026, 8, 20)
    replay = PortfolioTransitionEngine.transition(
        base.resulting_state, replay_session, (execution,)
    )
    result = backtest_result(base.resulting_state, replay)
    inputs = (
        PortfolioSessionInput(
            session=replay_session, execution_events=(execution,)
        ),
    )

    loaded = PortfolioArtifactReader.read(write_artifact(tmp_path, result, inputs))

    assert loaded.ledger_entries == ()
    assert loaded.execution_presentations[0].input_session == replay_session


def test_same_batch_exact_duplicate_multiplicity_survives(tmp_path: Path) -> None:
    execution = buy_event()
    initial = initial_state()
    transition = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution, execution)
    )
    result = backtest_result(initial, transition)
    inputs = (
        PortfolioSessionInput(
            session=execution.session,
            execution_events=(execution, execution),
        ),
    )

    loaded = PortfolioArtifactReader.read(write_artifact(tmp_path, result, inputs))

    assert len(loaded.execution_presentations) == 2
    assert loaded.execution_presentations[0] == loaded.execution_presentations[1]
    assert len(loaded.ledger_entries) == 1


def test_unproven_ledgerless_execution_is_rejected(tmp_path: Path) -> None:
    virgin = initial_state()
    directory = write_artifact(tmp_path, backtest_result(virgin), ())
    presentation = PortfolioExecutionPresentation(
        input_session=date(2026, 8, 18),
        execution_event=buy_event(),
    )
    rewrite_presentations(directory, (presentation,), update_manifest=True)

    with pytest.raises(PortfolioStateError, match="prior application proof"):
        PortfolioArtifactReader.read(directory)


def test_stale_execution_cannot_be_a_new_later_application(tmp_path: Path) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    loaded = PortfolioArtifactReader.read(directory)
    stale = loaded.execution_presentations[0].model_copy(
        update={"input_session": date(2026, 8, 20)}
    )
    rewrite_presentations(directory, (stale,), update_manifest=True)

    with pytest.raises(PortfolioStateError):
        PortfolioArtifactReader.read(directory)


def test_execution_presentation_cannot_precede_execution_session(
    tmp_path: Path,
) -> None:
    virgin = initial_state()
    directory = write_artifact(tmp_path, backtest_result(virgin), ())
    execution = buy_event(session=date(2026, 8, 20))
    future = PortfolioExecutionPresentation(
        input_session=date(2026, 8, 18),
        execution_event=execution,
    )
    rewrite_presentations(directory, (future,), update_manifest=True)

    with pytest.raises(PortfolioStateError, match="cannot precede"):
        PortfolioArtifactReader.read(directory)


def test_conflicting_duplicate_execution_presentations_are_rejected(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    loaded = PortfolioArtifactReader.read(directory)
    original = loaded.execution_presentations[0]
    conflict = original.model_copy(
        update={
            "execution_event": original.execution_event.model_copy(
                update={"fill_price": original.execution_event.fill_price + 1}
            )
        }
    )
    rewrite_presentations(
        directory, (original, conflict), update_manifest=True
    )

    with pytest.raises(PortfolioStateError, match="conflicting payloads"):
        PortfolioArtifactReader.read(directory)


def test_ledger_application_without_execution_source_is_rejected(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    rewrite_presentations(directory, (), update_manifest=True)

    with pytest.raises(PortfolioStateError, match="missing its source"):
        PortfolioArtifactReader.read(directory)


def test_execution_ledger_side_mismatch_is_rejected(tmp_path: Path) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    wrong_side = result.ledger_entries[0].model_copy(
        update={"event_type": PortfolioLedgerEventType.SELL_APPLIED}
    )
    rewrite_ledger(directory, (wrong_side,), update_manifest=True)

    with pytest.raises(PortfolioStateError, match="side does not match"):
        PortfolioArtifactReader.read(directory)


def test_execution_ledger_fact_mismatch_is_rejected(tmp_path: Path) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    mismatched = result.ledger_entries[0].model_copy(
        update={"source_order_id": "OTHER-ORDER"}
    )
    rewrite_ledger(directory, (mismatched,), update_manifest=True)

    with pytest.raises(PortfolioStateError, match="ledger facts"):
        PortfolioArtifactReader.read(directory)


def test_execution_content_hash_is_order_independent_but_session_and_count_sensitive(
) -> None:
    event_a = buy_event("EXEC-A", asset_id="NORGATE:1")
    event_b = buy_event("EXEC-B", asset_id="NORGATE:2")
    first = PortfolioExecutionPresentation(
        input_session=event_a.session, execution_event=event_a
    )
    second = PortfolioExecutionPresentation(
        input_session=event_b.session, execution_event=event_b
    )
    moved = first.model_copy(update={"input_session": date(2026, 8, 20)})

    assert hash_execution_presentations((first, second)) == (
        hash_execution_presentations((second, first))
    )
    assert hash_execution_presentations((first,)) != (
        hash_execution_presentations((moved,))
    )
    assert hash_execution_presentations((first,)) != (
        hash_execution_presentations((first, first))
    )


def test_writer_manifest_is_identical_for_reordered_session_inputs(
    tmp_path: Path,
) -> None:
    execution = buy_event()
    initial = initial_state()
    first = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    replay_session = date(2026, 8, 20)
    replay = PortfolioTransitionEngine.transition(
        first.resulting_state, replay_session, (execution,)
    )
    result = backtest_result(initial, first, replay)
    first_input = PortfolioSessionInput(
        session=execution.session,
        execution_events=(execution,),
    )
    replay_input = PortfolioSessionInput(
        session=replay_session,
        execution_events=(execution,),
    )

    first_manifest = PortfolioArtifactWriter.write(
        tmp_path / "first",
        result,
        (first_input, replay_input),
    )
    second_manifest = PortfolioArtifactWriter.write(
        tmp_path / "second",
        result,
        (replay_input, first_input),
    )

    assert first_manifest == second_manifest


def test_final_execution_fingerprint_tamper_is_rejected_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    fingerprint = result.final_state.applied_events[0].model_copy(
        update={"payload_sha256": "a" * 64}
    )
    tampered = result.final_state.model_copy(
        update={"applied_events": (fingerprint,)}
    )
    (directory / FINAL_STATE_FILENAME).write_bytes(canonical_payload_bytes(tampered))
    rewrite_manifest(
        directory,
        final_state_content_sha256=hash_portfolio_state(tampered),
    )

    with pytest.raises(PortfolioStateError, match="fingerprints"):
        PortfolioArtifactReader.read(directory)


def test_final_settlement_fingerprint_tamper_is_rejected(tmp_path: Path) -> None:
    initial = state_with_position()
    execution = sell_event()
    sale = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    settlement_session = sale.resulting_state.pending_settlements[0].settlement_session
    settlement = PortfolioTransitionEngine.transition(
        sale.resulting_state, settlement_session, ()
    )
    result = backtest_result(initial, sale, settlement)
    inputs = (
        PortfolioSessionInput(
            session=execution.session,
            execution_events=(execution,),
        ),
        PortfolioSessionInput(session=settlement_session),
    )
    directory = write_artifact(tmp_path, result, inputs)
    fingerprints = list(result.final_state.applied_events)
    settlement_index = next(
        index
        for index, item in enumerate(fingerprints)
        if item.event_kind.value == "SETTLEMENT"
    )
    fingerprints[settlement_index] = fingerprints[settlement_index].model_copy(
        update={"payload_sha256": "a" * 64}
    )
    tampered = result.final_state.model_copy(
        update={"applied_events": tuple(fingerprints)}
    )
    (directory / FINAL_STATE_FILENAME).write_bytes(canonical_payload_bytes(tampered))
    rewrite_manifest(
        directory,
        final_state_content_sha256=hash_portfolio_state(tampered),
    )

    with pytest.raises(PortfolioStateError, match="fingerprints"):
        PortfolioArtifactReader.read(directory)


def test_state_with_valid_hash_but_invalid_virgin_invariants_is_rejected(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    invalid_initial = result.initial_state.model_copy(
        update={"applied_events": result.final_state.applied_events}
    )
    (directory / INITIAL_STATE_FILENAME).write_bytes(
        canonical_payload_bytes(invalid_initial)
    )
    rewrite_manifest(
        directory,
        initial_state_content_sha256=hash_portfolio_state(invalid_initial),
    )

    with pytest.raises(PortfolioStateError, match="virgin state"):
        PortfolioArtifactReader.read(directory)


def test_missing_manifest_marks_artifact_incomplete(tmp_path: Path) -> None:
    directory = tmp_path / "incomplete"
    directory.mkdir()
    (directory / INITIAL_STATE_FILENAME).write_bytes(
        canonical_payload_bytes(initial_state())
    )

    with pytest.raises(PortfolioPersistenceError, match="manifest.json is missing"):
        PortfolioArtifactReader.read(directory)


def test_unsupported_manifest_and_state_versions_fail_closed(tmp_path: Path) -> None:
    result, inputs = one_buy_history()
    manifest_directory = write_artifact(
        tmp_path, result, inputs, name="bad-manifest"
    )
    rewrite_manifest(manifest_directory, schema_version="portfolio_manifest.v9")

    with pytest.raises(SchemaVersionError):
        PortfolioArtifactReader.read(manifest_directory)

    state_directory = write_artifact(tmp_path, result, inputs, name="bad-state")
    payload = json.loads(
        (state_directory / FINAL_STATE_FILENAME).read_text(encoding="utf-8")
    )
    payload["schema_version"] = "portfolio_state.v9"
    (state_directory / FINAL_STATE_FILENAME).write_bytes(
        canonical_payload_bytes(payload)
    )

    with pytest.raises(SchemaVersionError):
        PortfolioArtifactReader.read(state_directory)


@pytest.mark.parametrize(
    ("filename", "column", "invalid_version"),
    [
        (
            EXECUTION_EVENTS_FILENAME,
            "presentation_schema_version",
            "portfolio_execution_presentation.v9",
        ),
        (
            EXECUTION_EVENTS_FILENAME,
            "execution_schema_version",
            "portfolio_execution_event.v9",
        ),
        (
            PORTFOLIO_LEDGER_FILENAME,
            "schema_version",
            "portfolio_ledger_entry.v9",
        ),
        (
            SESSION_SNAPSHOTS_FILENAME,
            "schema_version",
            "portfolio_session_snapshot.v9",
        ),
    ],
)
def test_unsupported_parquet_row_schema_versions_fail_closed(
    tmp_path: Path,
    filename: str,
    column: str,
    invalid_version: str,
) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    path = directory / filename
    table = pq.read_table(path)
    index = table.schema.get_field_index(column)
    replacement = pa.array([invalid_version] * table.num_rows, type=pa.string())
    pq.write_table(table.set_column(index, column, replacement), path)

    with pytest.raises(SchemaVersionError):
        PortfolioArtifactReader.read(directory)


def test_modified_state_and_manifest_hashes_fail_integrity(tmp_path: Path) -> None:
    result, inputs = one_buy_history()
    state_directory = write_artifact(tmp_path, result, inputs, name="state")
    payload = json.loads(
        (state_directory / FINAL_STATE_FILENAME).read_text(encoding="utf-8")
    )
    payload["settled_cash"] = "978"
    (state_directory / FINAL_STATE_FILENAME).write_bytes(
        canonical_payload_bytes(payload)
    )

    with pytest.raises(StateHashMismatchError):
        PortfolioArtifactReader.read(state_directory)

    manifest_directory = write_artifact(tmp_path, result, inputs, name="manifest")
    rewrite_manifest(
        manifest_directory,
        final_state_content_sha256="0" * 64,
    )

    with pytest.raises(StateHashMismatchError):
        PortfolioArtifactReader.read(manifest_directory)


def test_modified_execution_ledger_and_snapshot_content_fail_hashes(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    execution_directory = write_artifact(
        tmp_path, result, inputs, name="execution"
    )
    execution_loaded = PortfolioArtifactReader.read(execution_directory)
    presentation = execution_loaded.execution_presentations[0]
    changed_presentation = presentation.model_copy(
        update={
            "execution_event": presentation.execution_event.model_copy(
                update={"source_order_id": "CHANGED"}
            )
        }
    )
    rewrite_presentations(
        execution_directory, (changed_presentation,), update_manifest=False
    )
    with pytest.raises(ArtifactHashMismatchError):
        PortfolioArtifactReader.read(execution_directory)

    ledger_directory = write_artifact(tmp_path, result, inputs, name="ledger")
    changed_ledger = result.ledger_entries[0].model_copy(
        update={"settled_cash_delta": Decimal("-22")}
    )
    rewrite_ledger(ledger_directory, (changed_ledger,), update_manifest=False)
    with pytest.raises(ArtifactHashMismatchError):
        PortfolioArtifactReader.read(ledger_directory)

    snapshot_directory = write_artifact(tmp_path, result, inputs, name="snapshot")
    changed_snapshot = result.snapshots[0].model_copy(
        update={"settled_cash": Decimal("978")}
    )
    rewrite_snapshots(
        snapshot_directory, (changed_snapshot,), update_manifest=False
    )
    with pytest.raises(ArtifactHashMismatchError):
        PortfolioArtifactReader.read(snapshot_directory)


def test_writer_rejects_decimal_that_arrow_cannot_store_without_rounding(
    tmp_path: Path,
) -> None:
    execution = buy_event(
        fill_price=Decimal("0.123456789012345678901234567890123456789"),
        execution_cost=Decimal("0"),
    )
    result, inputs = one_buy_history(event=execution)

    target = tmp_path / "too-precise"
    with pytest.raises(PortfolioPersistenceError, match="represented exactly"):
        PortfolioArtifactWriter.write(target, result, inputs)

    assert not target.exists()


def test_writer_publishes_complete_fixed_artifact_set_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)

    assert {path.name for path in directory.iterdir()} == {
        MANIFEST_FILENAME,
        INITIAL_STATE_FILENAME,
        FINAL_STATE_FILENAME,
        EXECUTION_EVENTS_FILENAME,
        PORTFOLIO_LEDGER_FILENAME,
        SESSION_SNAPSHOTS_FILENAME,
        DIVIDEND_EVIDENCE_FILENAME,
        DIVIDEND_OUTCOMES_FILENAME,
        DIVIDEND_LEDGER_FILENAME,
    }
    with pytest.raises(PortfolioPersistenceError, match="already exists"):
        PortfolioArtifactWriter.write(directory, result, inputs)


def test_b3a_valid_cash_reconstruction_is_accepted(tmp_path: Path) -> None:
    result, inputs = one_buy_history()

    loaded = PortfolioArtifactReader.read(write_artifact(tmp_path, result, inputs))

    expected_cash = result.initial_state.settled_cash + sum(
        (entry.settled_cash_delta for entry in result.ledger_entries),
        start=Decimal("0"),
    )
    assert loaded.final_state.settled_cash == expected_cash


def test_b3a_cash_tamper_is_rejected_after_all_dependent_hashes_are_updated(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    tampered = result.final_state.model_copy(
        update={"settled_cash": result.final_state.settled_cash + Decimal("1")}
    )
    rewrite_final_state_and_dependencies(directory, result, tampered)

    with pytest.raises(PortfolioStateError, match="initial state plus ledger"):
        PortfolioArtifactReader.read(directory)


def test_b3a_buy_quantity_tamper_is_rejected_after_hashes_are_updated(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    directory = write_artifact(tmp_path, result, inputs)
    position = result.final_state.open_positions[0]
    tampered_position = position.model_copy(
        update={
            "quantity": 3,
            "cost_basis": Decimal("3") * position.entry_price
            + position.entry_execution_cost,
        }
    )
    tampered = result.final_state.model_copy(
        update={"open_positions": (tampered_position,)}
    )
    rewrite_final_state_and_dependencies(directory, result, tampered)

    with pytest.raises(PortfolioStateError, match="initial state plus ledger"):
        PortfolioArtifactReader.read(directory)


def test_b3a_unexplained_position_removal_is_rejected(tmp_path: Path) -> None:
    initial = state_with_position()
    empty_session = date(2026, 8, 19)
    transition = PortfolioTransitionEngine.transition(initial, empty_session, ())
    result = backtest_result(initial, transition)
    inputs = (PortfolioSessionInput(session=empty_session),)
    directory = write_artifact(tmp_path, result, inputs)
    tampered = result.final_state.model_copy(update={"open_positions": ()})
    rewrite_final_state_and_dependencies(directory, result, tampered)

    with pytest.raises(PortfolioStateError, match="initial state plus ledger"):
        PortfolioArtifactReader.read(directory)


def test_b3a_valid_sell_created_pending_settlement_is_accepted(
    tmp_path: Path,
) -> None:
    initial = state_with_position()
    execution = sell_event()
    sale = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    result = backtest_result(initial, sale)
    inputs = (
        PortfolioSessionInput(
            session=execution.session,
            execution_events=(execution,),
        ),
    )

    loaded = PortfolioArtifactReader.read(write_artifact(tmp_path, result, inputs))

    assert loaded.final_state.pending_settlements == (
        sale.resulting_state.pending_settlements[0],
    )


def test_b3a_pending_settlement_tamper_is_rejected(tmp_path: Path) -> None:
    initial = state_with_position()
    execution = sell_event()
    sale = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    result = backtest_result(initial, sale)
    inputs = (
        PortfolioSessionInput(
            session=execution.session,
            execution_events=(execution,),
        ),
    )
    directory = write_artifact(tmp_path, result, inputs)
    pending = result.final_state.pending_settlements[0]
    tampered_pending = pending.model_copy(
        update={"amount": pending.amount + Decimal("1")}
    )
    tampered = result.final_state.model_copy(
        update={"pending_settlements": (tampered_pending,)}
    )
    rewrite_final_state_and_dependencies(directory, result, tampered)

    with pytest.raises(PortfolioStateError, match="initial state plus ledger"):
        PortfolioArtifactReader.read(directory)


@pytest.mark.parametrize(
    ("advanced_session", "message"),
    [
        (date(2026, 8, 21), "due settlement lacks"),
        (date(2026, 8, 22), "pending settlement is overdue"),
    ],
)
def test_b3a_due_or_overdue_settlement_cannot_survive_processed_history(
    tmp_path: Path,
    advanced_session: date,
    message: str,
) -> None:
    initial = state_with_position()
    execution = sell_event(settlement_session=date(2026, 8, 21))
    sale = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    result = backtest_result(initial, sale)
    inputs = (
        PortfolioSessionInput(
            session=execution.session,
            execution_events=(execution,),
        ),
    )
    directory = write_artifact(tmp_path, result, inputs)
    advanced = result.final_state.model_copy(
        update={
            "as_of_session": advanced_session,
            "state_version": result.final_state.state_version + 1,
        }
    )
    advanced_hash = hash_portfolio_state(advanced)
    (directory / FINAL_STATE_FILENAME).write_bytes(canonical_payload_bytes(advanced))
    rewrite_manifest(
        directory,
        final_state_content_sha256=advanced_hash,
    )
    advanced_snapshot = PortfolioSessionSnapshot(
        session=advanced_session,
        state_version=advanced.state_version,
        settled_cash=advanced.settled_cash,
        open_position_count=len(advanced.open_positions),
        pending_settlement_count=len(advanced.pending_settlements),
        state_hash=advanced_hash,
    )
    rewrite_snapshots(
        directory,
        (*result.snapshots, advanced_snapshot),
        update_manifest=True,
    )

    with pytest.raises(PortfolioStateError, match=message):
        PortfolioArtifactReader.read(directory)


def test_b3a_settlement_applied_on_due_session_is_accepted(tmp_path: Path) -> None:
    initial = state_with_position()
    execution = sell_event(settlement_session=date(2026, 8, 21))
    sale = PortfolioTransitionEngine.transition(
        initial, execution.session, (execution,)
    )
    settlement = PortfolioTransitionEngine.transition(
        sale.resulting_state, execution.settlement_session, ()
    )
    result = backtest_result(initial, sale, settlement)
    inputs = (
        PortfolioSessionInput(
            session=execution.session,
            execution_events=(execution,),
        ),
        PortfolioSessionInput(session=execution.settlement_session),
    )

    loaded = PortfolioArtifactReader.read(write_artifact(tmp_path, result, inputs))

    assert loaded.final_state.pending_settlements == ()


def test_b3a_writer_rejects_backtest_ledger_aggregate_mismatch(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    contradictory = result.model_copy(update={"ledger_entries": ()})
    target = tmp_path / "ledger-aggregate-mismatch"

    with pytest.raises(PortfolioStateError, match="aggregate ledger"):
        PortfolioArtifactWriter.write(target, contradictory, inputs)

    assert not target.exists()


def test_b3a_writer_rejects_backtest_snapshot_aggregate_mismatch(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    changed_snapshot = result.snapshots[0].model_copy(
        update={"settled_cash": result.snapshots[0].settled_cash + Decimal("1")}
    )
    contradictory = result.model_copy(update={"snapshots": (changed_snapshot,)})
    target = tmp_path / "snapshot-aggregate-mismatch"

    with pytest.raises(PortfolioStateError, match="aggregate snapshots"):
        PortfolioArtifactWriter.write(target, contradictory, inputs)

    assert not target.exists()


def test_b3a_writer_rejects_backtest_final_state_mismatch(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    contradictory = result.model_copy(
        update={
            "final_state": result.initial_state,
            "final_state_hash": result.initial_state_hash,
        }
    )
    target = tmp_path / "final-state-aggregate-mismatch"

    with pytest.raises(PortfolioStateError, match="final state"):
        PortfolioArtifactWriter.write(target, contradictory, inputs)

    assert not target.exists()


def test_b3a_writer_rejects_backtest_session_hash_chain_mismatch(
    tmp_path: Path,
) -> None:
    result, inputs = one_buy_history()
    changed_session = result.session_results[0].model_copy(
        update={"state_hash_before": "a" * 64}
    )
    contradictory = result.model_copy(update={"session_results": (changed_session,)})
    target = tmp_path / "session-hash-chain-mismatch"

    with pytest.raises(PortfolioStateError, match="state-hash chain"):
        PortfolioArtifactWriter.write(target, contradictory, inputs)

    assert not target.exists()


def test_b3a_valid_no_session_result_round_trips(tmp_path: Path) -> None:
    initial = initial_state(Decimal("1000.25"))
    result = backtest_result(initial)

    loaded = PortfolioArtifactReader.read(write_artifact(tmp_path, result, ()))

    assert loaded.initial_state == initial
    assert loaded.final_state == initial
    assert loaded.ledger_entries == ()
    assert loaded.session_snapshots == ()


def test_b3a_writer_rejects_invalid_no_session_result(tmp_path: Path) -> None:
    initial = initial_state()
    changed_final = PortfolioState(settled_cash=initial.settled_cash + Decimal("1"))
    contradictory = backtest_result(initial).model_copy(
        update={
            "final_state": changed_final,
            "final_state_hash": hash_portfolio_state(changed_final),
        }
    )
    target = tmp_path / "invalid-no-session"

    with pytest.raises(PortfolioStateError, match="no-session"):
        PortfolioArtifactWriter.write(target, contradictory, ())

    assert not target.exists()
