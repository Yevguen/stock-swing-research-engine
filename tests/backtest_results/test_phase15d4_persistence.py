"""Phase 15D.4: persistence and deterministic read-back of a completed,
already-validated `HistoricalBacktestAuditResult`.

Every test here treats `HistoricalBacktestResultPersistence.write`/`.read` as
a pure physical-storage boundary: no upstream economic owner is invoked, and
the only thing being proven is that a bundle round-trips exactly, and that
every distinct corruption fails closed with the right coded error.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stock_swing_d1.backtest_results import (
    HistoricalBacktestPersistenceError,
    HistoricalBacktestResultPersistence,
    HistoricalBacktestResultService,
    HistoricalBacktestValuationMark,
    HistoricalClosedTradeRecord,
    HistoricalExitReason,
    HistoricalOpenTradeRecord,
    build_valuation_snapshot,
    project_entries,
)
from stock_swing_d1.backtest_results import persistence_schema
from stock_swing_d1.backtester import (
    HistoricalDecisionInterval,
    HistoricalBacktestOrchestrator,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionInput,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.backtest_results.conftest import SOURCE_DECISION_INTERVAL


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "src" / "stock_swing_d1" / "backtest_results"

E1 = date(2027, 1, 6)
E2 = date(2027, 1, 7)
E3 = date(2027, 1, 8)
PRE = date(2027, 1, 1)
PERSISTENCE_DECISION_INTERVAL = HistoricalDecisionInterval(
    decision_start_date=date(2024, 2, 29),
    decision_end_date=date(2025, 1, 9),
)


def _dt(session: date) -> datetime:
    return datetime.combine(session, time(20), tzinfo=timezone.utc)


def _session(session: date, **overrides) -> HistoricalBacktestSessionInput:
    values = {"session": session, "decision_time": _dt(session)}
    values.update(overrides)
    return HistoricalBacktestSessionInput(**values)


def _closed_trade_run() -> HistoricalBacktestRunResult:
    buy = PortfolioExecutionEvent(
        execution_id="P4-BUY-1",
        source_order_id="P4-BUY-1-ORDER",
        session=E1,
        asset_id="NORGATE:9001",
        side=ExecutionSide.BUY,
        quantity=3,
        fill_price=Decimal("50"),
        execution_cost=Decimal("2"),
    )
    sell = PortfolioExecutionEvent(
        execution_id="P4-SELL-1",
        source_order_id="P4-SELL-1-ORDER",
        session=E2,
        asset_id="NORGATE:9001",
        side=ExecutionSide.SELL,
        quantity=3,
        fill_price=Decimal("55"),
        execution_cost=Decimal("2"),
        settlement_id="P4-SETTLEMENT-1",
        settlement_session=E3,
    )
    return HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            _session(E1, scheduled_execution_events=(buy,)),
            _session(E2, scheduled_execution_events=(sell,)),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


@pytest.fixture
def rich_open_result(source_run_bundle):
    """Non-empty run with signal/ranking/allocation/rejection provenance and
    one freshly-entered, still-open trade at run end (see conftest.py's
    `source_run_bundle`)."""

    run, manifest = source_run_bundle
    entries = project_entries(run)
    snapshots = (
        build_valuation_snapshot(session=run.session_results[0].session, marks=()),
        build_valuation_snapshot(
            session=run.session_results[1].session,
            marks=(
                HistoricalBacktestValuationMark(
                    security_id=entries[0].security_id,
                    session=run.session_results[1].session,
                    close=Decimal("105"),
                    source_artifact_ref=manifest.market_data_artifact_ref,
                ),
            ),
        ),
    )
    return HistoricalBacktestResultService.build(
        run_result=run, run_manifest=manifest, valuation_snapshots=snapshots
    )


@pytest.fixture
def closed_result(run_manifest):
    """Non-empty run with one closed trade, one exit, and one settlement."""

    run = _closed_trade_run()
    mark = HistoricalBacktestValuationMark(
        security_id="NORGATE:9001",
        session=E1,
        close=Decimal("52"),
        source_artifact_ref=run_manifest.market_data_artifact_ref,
    )
    snapshots = (
        build_valuation_snapshot(session=E1, marks=(mark,)),
        build_valuation_snapshot(session=E2, marks=()),
    )
    return HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )


@pytest.fixture
def zero_cash_result(run_manifest):
    state = PortfolioState(settled_cash=Decimal("777"))
    fingerprint = hash_portfolio_state(state)
    run = HistoricalBacktestRunResult(
        decision_interval=PERSISTENCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        session_results=(),
        initial_state_fingerprint=fingerprint,
        final_state_fingerprint=fingerprint,
    )
    return HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=()
    )


@pytest.fixture
def zero_open_result(run_manifest):
    cost = Decimal(2) * Decimal("100") + Decimal("1")
    virgin = PortfolioState(settled_cash=Decimal("10000") + cost)
    event = PortfolioExecutionEvent(
        execution_id="P4-CARRY-BUY",
        source_order_id="P4-CARRY-BUY-ORDER",
        session=PRE,
        asset_id="NORGATE:6001",
        side=ExecutionSide.BUY,
        quantity=2,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    state = PortfolioTransitionEngine.transition(
        virgin, PRE, (event,)
    ).resulting_state
    fingerprint = hash_portfolio_state(state)
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state,
        final_state=state,
        session_results=(),
        initial_state_fingerprint=fingerprint,
        final_state_fingerprint=fingerprint,
    )
    mark = HistoricalBacktestValuationMark(
        security_id="NORGATE:6001",
        session=PRE,
        close=Decimal("115"),
        source_artifact_ref=run_manifest.market_data_artifact_ref,
    )
    snapshots = (build_valuation_snapshot(session=PRE, marks=(mark,)),)
    return HistoricalBacktestResultService.build(
        run_result=run, run_manifest=run_manifest, valuation_snapshots=snapshots
    )


def _corrupt_and_refresh_hash(bundle_dir: Path, filename: str, corrupt_fn) -> None:
    """Corrupt one payload file's content, then update *only* its physical
    hash in manifest.json (leaving every semantic fingerprint untouched) so
    the read path reaches the later semantic-verification layer being
    tested, rather than stopping at the earlier physical-hash guard."""

    target = bundle_dir / filename
    corrupt_fn(target)
    manifest_path = bundle_dir / persistence_schema.MANIFEST_FILENAME
    manifest = persistence_schema.decode_bundle_manifest(manifest_path.read_bytes())
    new_hashes = dict(manifest.payload_sha256)
    new_hashes[filename] = persistence_schema.sha256_file(target)
    updated = manifest.model_copy(update={"payload_sha256": new_hashes})
    manifest_path.write_bytes(persistence_schema.encode_bundle_manifest(updated))


def _rewrite_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _rewrite_ordinal_column(path: Path, mutate_values) -> None:
    table = pq.read_table(path)
    index = table.schema.get_field_index("ordinal")
    values = table.column(index).to_pylist()
    mutate_values(values)
    new_column = pa.array(values, type=pa.int64())
    table = table.set_column(index, table.schema.field(index), new_column)
    pq.write_table(table, path, compression="zstd")


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_public_api_is_exactly_write_and_read():
    assert {
        name
        for name in vars(HistoricalBacktestResultPersistence)
        if not name.startswith("_")
    } == {"write", "read"}


def test_backtest_results_all_does_not_export_persistence_schema_internals():
    import stock_swing_d1.backtest_results as package

    forbidden_substrings = ("SCHEMA", "DECIMAL_ARROW", "encode_", "decode_", "TABLE_")
    for name in package.__all__:
        for substring in forbidden_substrings:
            assert substring not in name


def test_persistence_files_never_round_quantize_or_use_tolerance():
    source = (PRODUCTION / "persistence.py").read_text(encoding="utf-8") + (
        PRODUCTION / "persistence_schema.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "round(",
        ".quantize(",
        "math.isclose",
        "pytest.approx",
        "Decimal(float",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Round trips: exact equality
# ---------------------------------------------------------------------------


def test_round_trip_zero_session_cash_only_result(zero_cash_result, tmp_path):
    destination = tmp_path / "bundle"
    published = HistoricalBacktestResultPersistence.write(
        result=zero_cash_result, destination=destination
    )
    assert published == destination
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert read_back == zero_cash_result
    assert read_back.decision_interval == PERSISTENCE_DECISION_INTERVAL
    assert read_back.decision_interval.decision_start_date == date(2024, 2, 29)
    assert read_back.decision_interval.decision_end_date == date(2025, 1, 9)
    summary_payload = json.loads(
        (destination / persistence_schema.SUMMARY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert summary_payload["decision_interval"] == {
        "decision_start_date": "2024-02-29",
        "decision_end_date": "2025-01-09",
    }
    assert "decision_start_date" not in summary_payload
    assert "decision_end_date" not in summary_payload
    for table_name in persistence_schema.TABLE_NAMES:
        assert getattr(read_back, table_name) == ()


def test_round_trip_zero_session_carried_in_open_result(zero_open_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=zero_open_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert read_back == zero_open_result
    assert len(read_back.trades) == 1
    assert isinstance(read_back.trades[0], HistoricalOpenTradeRecord)
    assert read_back.trades[0].carried_in is True


def test_round_trip_closed_trade_result(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert read_back == closed_result
    assert isinstance(read_back.trades[0], HistoricalClosedTradeRecord)


def test_round_trip_rich_open_result(rich_open_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=rich_open_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert read_back == rich_open_result


def test_repeated_read_of_same_bundle_is_logically_equal(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    first = HistoricalBacktestResultPersistence.read(source=destination)
    second = HistoricalBacktestResultPersistence.read(source=destination)
    assert first == second == closed_result


def test_write_does_not_mutate_the_source_result_object(closed_result, tmp_path):
    before = closed_result.model_copy(deep=True)
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=tmp_path / "bundle"
    )
    assert closed_result == before


def test_repeated_publish_to_different_destinations_are_logically_equal(
    closed_result, tmp_path
):
    first_destination = tmp_path / "bundle_a"
    second_destination = tmp_path / "bundle_b"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=first_destination
    )
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=second_destination
    )
    first = HistoricalBacktestResultPersistence.read(source=first_destination)
    second = HistoricalBacktestResultPersistence.read(source=second_destination)
    assert first == second == closed_result


# ---------------------------------------------------------------------------
# Bundle inventory / structure
# ---------------------------------------------------------------------------


def test_write_produces_exactly_the_20_file_bundle_inventory(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    names = {path.name for path in destination.iterdir()}
    assert names == set(persistence_schema.BUNDLE_FILENAMES)
    assert len(names) == 20


def test_all_15_parquet_tables_have_explicit_schema_when_empty(
    zero_cash_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=zero_cash_result, destination=destination
    )
    for table_name, expected_schema in persistence_schema.TABLE_SCHEMAS.items():
        table = pq.read_table(destination / f"{table_name}.parquet")
        assert table.schema.equals(expected_schema, check_metadata=False)
        assert table.num_rows == 0


def test_manifest_never_hashes_itself(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    manifest = persistence_schema.decode_bundle_manifest(
        (destination / persistence_schema.MANIFEST_FILENAME).read_bytes()
    )
    assert persistence_schema.MANIFEST_FILENAME not in manifest.payload_sha256
    assert len(manifest.payload_sha256) == 19
    assert set(manifest.payload_sha256) == set(persistence_schema.PAYLOAD_FILENAMES)


def test_manifest_row_counts_match_all_15_tables(rich_open_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=rich_open_result, destination=destination
    )
    manifest = persistence_schema.decode_bundle_manifest(
        (destination / persistence_schema.MANIFEST_FILENAME).read_bytes()
    )
    for table_name in persistence_schema.TABLE_NAMES:
        assert manifest.row_counts[table_name] == len(
            getattr(rich_open_result, table_name)
        )


# ---------------------------------------------------------------------------
# Decimal exactness
# ---------------------------------------------------------------------------


def test_decimal_fields_round_trip_exactly_including_type(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    trade = read_back.trades[0]
    assert type(trade.realized_pnl) is Decimal
    assert trade.realized_pnl == closed_result.trades[0].realized_pnl
    assert type(read_back.entries[0].cost_basis) is Decimal
    assert read_back.entries[0].cost_basis == closed_result.entries[0].cost_basis


def test_require_exact_decimal_rejects_scale_beyond_76_38():
    huge = Decimal("1." + "1" * 39)
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        persistence_schema.require_exact_decimal(huge, "probe_field")
    assert excinfo.value.code == "DECIMAL_NOT_EXACTLY_REPRESENTABLE"


def test_encode_trade_row_rejects_decimal_beyond_76_38(closed_result):
    huge = Decimal("1." + "2" * 39)
    oversized_trade = closed_result.trades[0].model_copy(
        update={"realized_pnl": huge}
    )
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        persistence_schema.encode_trade_row(oversized_trade, 0)
    assert excinfo.value.code == "DECIMAL_NOT_EXACTLY_REPRESENTABLE"


def test_write_fails_closed_on_oversized_decimal_and_leaves_no_destination(
    rich_open_result, tmp_path
):
    # `HistoricalBacktestRejectionRecord` has no cross-field `model_validator`,
    # so tampering `reserved_cash` alone survives the write path's own
    # structural revalidation and reaches the Arrow-representability
    # preflight specifically (proving that guard, not an unrelated one).
    assert len(rich_open_result.rejections) > 0
    huge = Decimal("1." + "3" * 39)
    tampered_rejection = rich_open_result.rejections[0].model_copy(
        update={"reserved_cash": huge}
    )
    tampered_result = rich_open_result.model_copy(
        update={
            "rejections": (tampered_rejection,) + rich_open_result.rejections[1:]
        }
    )
    destination = tmp_path / "bundle"

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.write(
            result=tampered_result, destination=destination
        )
    assert excinfo.value.code == "DECIMAL_NOT_EXACTLY_REPRESENTABLE"
    assert not destination.exists()


# ---------------------------------------------------------------------------
# Float, date, aware-datetime, enum, optional
# ---------------------------------------------------------------------------


def test_float_fields_persist_as_float64_and_round_trip_exactly(
    rich_open_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=rich_open_result, destination=destination
    )
    table = pq.read_table(destination / "signal_provenance.parquet")
    assert table.schema.field("adjusted_close").type == pa.float64()

    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    for original, restored in zip(
        rich_open_result.signal_provenance, read_back.signal_provenance
    ):
        assert type(restored.adjusted_close) is float
        assert restored.adjusted_close == original.adjusted_close
        assert (restored.sma20 is None) == (original.sma20 is None)
        if original.sma20 is not None:
            assert restored.sma20 == original.sma20


def test_date_fields_survive_exactly(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert type(read_back.entries[0].entry_session) is date
    assert read_back.entries[0].entry_session == closed_result.entries[0].entry_session


def test_aware_datetime_fields_survive_exactly(rich_open_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=rich_open_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    original_dt = rich_open_result.signal_provenance[0].decision_time
    restored_dt = read_back.signal_provenance[0].decision_time
    assert restored_dt.tzinfo is not None
    assert restored_dt == original_dt


def test_enum_fields_survive_exactly(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert isinstance(read_back.exits[0].exit_reason, HistoricalExitReason)
    assert read_back.exits[0].exit_reason == closed_result.exits[0].exit_reason
    assert isinstance(read_back.trades[0], HistoricalClosedTradeRecord)
    assert read_back.trades[0].status == closed_result.trades[0].status


def test_optional_fields_survive_as_none(closed_result, tmp_path):
    assert closed_result.entries[0].signal_session is None
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert read_back.entries[0].signal_session is None


def test_optional_fields_survive_when_populated(rich_open_result, tmp_path):
    original_entry = rich_open_result.entries[0]
    assert original_entry.signal_session is not None
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=rich_open_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    restored_entry = read_back.entries[0]
    assert restored_entry.signal_session == original_entry.signal_session
    assert restored_entry.signal_time == original_entry.signal_time
    assert restored_entry.ranking_snapshot_fingerprint == (
        original_entry.ranking_snapshot_fingerprint
    )


# ---------------------------------------------------------------------------
# Trade union variants
# ---------------------------------------------------------------------------


def test_closed_trade_variant_round_trips_with_only_its_own_fields(
    closed_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    trade = read_back.trades[0]
    assert isinstance(trade, HistoricalClosedTradeRecord)
    assert not isinstance(trade, HistoricalOpenTradeRecord)
    assert trade == closed_result.trades[0]


def test_open_trade_variant_round_trips_with_only_its_own_fields(
    rich_open_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=rich_open_result, destination=destination
    )
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    trade = read_back.trades[0]
    assert isinstance(trade, HistoricalOpenTradeRecord)
    assert trade == rich_open_result.trades[0]


def test_unknown_trade_status_discriminator_fails_closed(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )

    def _corrupt(path: Path) -> None:
        table = pq.read_table(path)
        index = table.schema.get_field_index("status")
        bogus = pa.array(["BOGUS_STATUS"] * table.num_rows, type=pa.string())
        table = table.set_column(index, table.schema.field(index), bogus)
        pq.write_table(table, path, compression="zstd")

    _corrupt_and_refresh_hash(destination, "trades.parquet", _corrupt)

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "PERSISTED_MODEL_INVALID"


# ---------------------------------------------------------------------------
# Failure-mode / error-taxonomy coverage
# ---------------------------------------------------------------------------


def test_write_fails_closed_when_final_publish_rename_fails(
    closed_result, tmp_path, monkeypatch
):
    # Guarded to only intercept our own temp-bundle directory's `.rename()`
    # (the last filesystem operation in `write`), so this reaches
    # BUNDLE_WRITE_FAILED specifically rather than an earlier guard.
    canary = tmp_path / "unrelated_sibling.txt"
    canary.write_text("do not touch", encoding="utf-8")
    before_result = closed_result.model_copy(deep=True)

    destination = tmp_path / "bundle"
    injected_error = OSError("simulated rename failure for BUNDLE_WRITE_FAILED coverage")
    temp_prefix = ".historical_backtest_result_bundle."
    original_rename = Path.rename

    def _raising_rename(self: Path, target):
        if self.name.startswith(temp_prefix):
            raise injected_error
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", _raising_rename)

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.write(
            result=closed_result, destination=destination
        )

    assert excinfo.value.code == "BUNDLE_WRITE_FAILED"
    assert excinfo.value.__cause__ is injected_error
    assert not destination.exists()

    remaining = {path.name for path in tmp_path.iterdir()}
    assert remaining == {"unrelated_sibling.txt"}
    assert canary.read_text(encoding="utf-8") == "do not touch"
    assert closed_result == before_result


def test_missing_payload_file_fails_closed(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    (destination / "exits.parquet").unlink()
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "BUNDLE_INVENTORY_MISMATCH"


def test_extra_unexpected_file_fails_closed(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    (destination / "unexpected.txt").write_text("surprise", encoding="utf-8")
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "BUNDLE_INVENTORY_MISMATCH"


def test_changed_payload_bytes_without_manifest_update_fails_physical_hash(
    closed_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    target = destination / "session_pnl.parquet"
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 0xFF
    target.write_bytes(bytes(raw))
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "PHYSICAL_HASH_MISMATCH"


def test_malformed_parquet_schema_fails_closed(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )

    def _drop_column(path: Path) -> None:
        table = pq.read_table(path)
        table = table.drop(["exit_reason"])
        pq.write_table(table, path, compression="zstd")

    _corrupt_and_refresh_hash(destination, "exits.parquet", _drop_column)

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "PARQUET_SCHEMA_MISMATCH"


def test_malformed_run_manifest_json_fails_closed(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )

    def _break(path: Path) -> None:
        _rewrite_json(
            path, lambda payload: payload.__setitem__(
                "software_revision_kind", "NOT_A_REAL_KIND"
            )
        )

    _corrupt_and_refresh_hash(destination, "run_manifest.json", _break)

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "PERSISTED_MODEL_INVALID"


def test_corrupted_initial_state_with_stale_manifest_fingerprint_fails_closed(
    closed_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )

    def _tamper_cash(path: Path) -> None:
        _rewrite_json(
            path, lambda payload: payload.__setitem__("settled_cash", "999999.99")
        )

    _corrupt_and_refresh_hash(destination, "initial_state.json", _tamper_cash)

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "STATE_FINGERPRINT_MISMATCH"


def test_corrupted_table_with_refreshed_hash_fails_content_fingerprint(
    closed_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )

    def _tamper_amount(path: Path) -> None:
        table = pq.read_table(path)
        index = table.schema.get_field_index("amount")
        new_column = pa.array(
            [Decimal("999999.99")], type=persistence_schema.DECIMAL_ARROW_TYPE
        )
        table = table.set_column(index, table.schema.field(index), new_column)
        pq.write_table(table, path, compression="zstd")

    _corrupt_and_refresh_hash(destination, "settlement_ledger.parquet", _tamper_amount)

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "CONTENT_FINGERPRINT_MISMATCH"


def test_corrupted_content_fingerprint_in_manifest_fails_closed(
    closed_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    manifest_path = destination / persistence_schema.MANIFEST_FILENAME
    manifest = persistence_schema.decode_bundle_manifest(manifest_path.read_bytes())
    tampered_fingerprints = manifest.content_fingerprints.model_copy(
        update={"entries": "9" * 64}
    )
    updated = manifest.model_copy(
        update={"content_fingerprints": tampered_fingerprints}
    )
    manifest_path.write_bytes(persistence_schema.encode_bundle_manifest(updated))

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "CONTENT_FINGERPRINT_MISMATCH"


def test_corrupted_result_fingerprint_in_manifest_fails_closed(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    manifest_path = destination / persistence_schema.MANIFEST_FILENAME
    manifest = persistence_schema.decode_bundle_manifest(manifest_path.read_bytes())
    updated = manifest.model_copy(update={"result_fingerprint": "8" * 64})
    manifest_path.write_bytes(persistence_schema.encode_bundle_manifest(updated))

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "RESULT_FINGERPRINT_MISMATCH"


def test_pre_amendment_bundle_without_interval_is_unsupported(
    zero_cash_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=zero_cash_result, destination=destination
    )
    _rewrite_json(
        destination / persistence_schema.SUMMARY_FILENAME,
        lambda payload: payload.pop("decision_interval"),
    )
    manifest_path = destination / persistence_schema.MANIFEST_FILENAME
    manifest = persistence_schema.decode_bundle_manifest(manifest_path.read_bytes())
    legacy = manifest.model_copy(
        update={
            "persistence_schema_version": (
                "historical_backtest_result_bundle.v0.1"
            ),
            "result_schema_version": "historical_backtest_audit_result.v0.1",
        }
    )
    manifest_path.write_bytes(persistence_schema.encode_bundle_manifest(legacy))

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "UNSUPPORTED_SCHEMA_VERSION"


def test_tampered_interval_with_stale_bound_identities_fails_closed(
    closed_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )

    def _tamper_interval(path: Path) -> None:
        _rewrite_json(
            path,
            lambda payload: payload["decision_interval"].__setitem__(
                "decision_start_date", "2020-01-02"
            ),
        )

    _corrupt_and_refresh_hash(
        destination,
        persistence_schema.SUMMARY_FILENAME,
        _tamper_interval,
    )

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "RESULT_FINGERPRINT_MISMATCH"


# ---------------------------------------------------------------------------
# Ordinal contract: exactly 0..N-1 in physical row order, never resorted
# ---------------------------------------------------------------------------


def _swap_first_two(values: list[int]) -> None:
    values[0], values[1] = values[1], values[0]


def _duplicate_second_as_first(values: list[int]) -> None:
    values[1] = values[0]


def _add_gap_to_last(values: list[int]) -> None:
    values[-1] = values[-1] + 1


def _make_first_negative(values: list[int]) -> None:
    values[0] = -1


@pytest.mark.parametrize(
    "mutate_values",
    [
        _swap_first_two,
        _duplicate_second_as_first,
        _add_gap_to_last,
        _make_first_negative,
    ],
)
def test_corrupted_ordinal_sequence_fails_closed_without_resorting(
    rich_open_result, tmp_path, mutate_values
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=rich_open_result, destination=destination
    )

    def _corrupt(path: Path) -> None:
        _rewrite_ordinal_column(path, mutate_values)

    _corrupt_and_refresh_hash(destination, "signal_provenance.parquet", _corrupt)

    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=destination)
    assert excinfo.value.code == "BUNDLE_READ_FAILED"


def test_ordinal_sequence_survives_exactly_for_a_non_trivial_table(
    rich_open_result, tmp_path
):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=rich_open_result, destination=destination
    )
    table = pq.read_table(destination / "signal_provenance.parquet")
    assert table.column("ordinal").to_pylist() == list(range(table.num_rows))
    assert table.num_rows == len(rich_open_result.signal_provenance)


# ---------------------------------------------------------------------------
# Existing-destination policy and invalid inputs
# ---------------------------------------------------------------------------


def test_existing_destination_is_refused_without_overwrite(closed_result, tmp_path):
    destination = tmp_path / "bundle"
    HistoricalBacktestResultPersistence.write(
        result=closed_result, destination=destination
    )
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.write(
            result=closed_result, destination=destination
        )
    assert excinfo.value.code == "BUNDLE_ALREADY_EXISTS"
    read_back = HistoricalBacktestResultPersistence.read(source=destination)
    assert read_back == closed_result


def test_write_rejects_non_path_destination(closed_result):
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.write(result=closed_result, destination=12345)
    assert excinfo.value.code == "INVALID_PERSISTENCE_PATH"


def test_read_rejects_non_directory_source(tmp_path):
    not_a_dir = tmp_path / "not_a_dir.txt"
    not_a_dir.write_text("nope", encoding="utf-8")
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=not_a_dir)
    assert excinfo.value.code == "INVALID_PERSISTENCE_PATH"


def test_read_rejects_non_path_source():
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=12345)
    assert excinfo.value.code == "INVALID_PERSISTENCE_PATH"


def test_read_rejects_missing_manifest(tmp_path):
    empty_dir = tmp_path / "empty_bundle"
    empty_dir.mkdir()
    with pytest.raises(HistoricalBacktestPersistenceError) as excinfo:
        HistoricalBacktestResultPersistence.read(source=empty_dir)
    assert excinfo.value.code == "BUNDLE_INVENTORY_MISMATCH"
