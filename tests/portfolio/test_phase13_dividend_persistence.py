"""Slice 6 acceptance tests: Phase 13 dividend persistence and offline
provenance/discharge revalidation (site 2 of the OD-14.7 three-site
architecture).

Covers persisting the accepted Slice-5 in-memory dividend mechanics
(`phase-13-15a-15d-ordinary-dividend-amendment-v0.7-FROZEN.md`) through
`PortfolioArtifactWriter`/`PortfolioArtifactReader`: exact round-trip of
`CanonicalDividendAccountingEvidence`, `DividendApplicationOutcome`, and
`DividendLedgerEntry`; offline re-proof of OD-14.5 row provenance and
OD-14.8/14.9 evidence/outcome discharge from the reconstructed session-start
state; context-independent settled-cash replay; and tamper/fail-closed
detection. Phase 15D projection is out of scope for this slice.
"""

from __future__ import annotations

import decimal
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CANONICAL_DIVIDEND_CURRENCY,
    CanonicalDividendAccountingEvidence,
    build_canonical_dividend_accounting_evidence,
    build_dividend_calendar_resolution_proof,
    build_ordinary_cash_classification_proof,
)
from stock_swing_d1.data.ordinary_dividend_normalization import (
    Gate3DividendNormalizationInputs,
    normalize_gate3_dividend,
)
from stock_swing_d1.portfolio import portfolio_persistence as persistence
from stock_swing_d1.portfolio.backtest_orchestration import (
    PortfolioBacktestOrchestrator,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    DividendApplicationStatus,
    compute_dividend_application_id,
    compute_dividend_application_payload_hash,
    compute_gross_dividend_cash,
)
from stock_swing_d1.portfolio.portfolio_errors import (
    ArtifactHashMismatchError,
    DividendProvenanceError,
    PortfolioPersistenceError,
    PortfolioStateError,
    SchemaVersionError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    PortfolioEventKind,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import canonical_payload_bytes
from stock_swing_d1.portfolio.portfolio_persistence import (
    DIVIDEND_EVIDENCE_ARROW_SCHEMA,
    DIVIDEND_EVIDENCE_FILENAME,
    DIVIDEND_LEDGER_ARROW_SCHEMA,
    DIVIDEND_LEDGER_FILENAME,
    DIVIDEND_OUTCOME_ARROW_SCHEMA,
    DIVIDEND_OUTCOMES_FILENAME,
    MANIFEST_FILENAME,
    PORTFOLIO_MANIFEST_SCHEMA_VERSION,
    PortfolioArtifactReader,
    PortfolioArtifactWriter,
    hash_dividend_evidence,
    hash_dividend_ledger,
    hash_dividend_outcomes,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioBacktestResult,
    PortfolioSessionInput,
    PortfolioState,
)
from tests.portfolio.phase13_helpers import buy_event, initial_state, sell_event

T_SESSION = date(2026, 8, 18)
X_SESSION = date(2026, 8, 19)
X_PLUS_1 = date(2026, 8, 20)

ASSET_A = "NORGATE:1"
ASSET_B = "NORGATE:2"


def make_dividend_evidence(
    *,
    event_id: str,
    entitlement_session: date = T_SESSION,
    ex_session: date = X_SESSION,
    security_id: str = ASSET_A,
    d_capitalspecial: Decimal = Decimal("1"),
    unadjusted_close_t: Decimal = Decimal("1"),
    close_capital_t: Decimal = Decimal("3"),
    snapshot_fingerprint: str = "c" * 64,
) -> CanonicalDividendAccountingEvidence:
    inputs = Gate3DividendNormalizationInputs(
        entitlement_session=entitlement_session,
        d_capitalspecial=d_capitalspecial,
        unadjusted_close_t=unadjusted_close_t,
        close_capital_t=close_capital_t,
    )
    result = normalize_gate3_dividend(inputs)
    classification = build_ordinary_cash_classification_proof(
        canonical_distribution_event_id=event_id,
        classification_contract_id="ordinary-cash-classification.v1",
        upstream_source_evidence_fingerprint="a" * 64,
    )
    calendar = build_dividend_calendar_resolution_proof(
        calendar_source_id="accepted-us-equity-calendar",
        entitlement_session=entitlement_session,
        ex_session=ex_session,
        calendar_policy_id="canonical-next-session",
        calendar_policy_version="1",
        upstream_source_evidence_fingerprint="b" * 64,
    )
    return build_canonical_dividend_accounting_evidence(
        canonical_distribution_event_id=event_id,
        canonical_security_id=security_id,
        normalization_inputs=inputs,
        normalization_result=result,
        classification_proof=classification,
        calendar_resolution_proof=calendar,
        currency=CANONICAL_DIVIDEND_CURRENCY,
        canonical_distribution_snapshot_fingerprint=snapshot_fingerprint,
    )


def build_applied_and_noop_run() -> (
    tuple[PortfolioBacktestResult, tuple[PortfolioSessionInput, ...]]
):
    """T: BUY 10 @ A. X: SELL old A (entitlement retained) + re-entry BUY,
    one APPLIED dividend on A, one NO_OP dividend on unheld B."""

    initial = initial_state(Decimal("100000"))
    buy = buy_event(
        "BUY-1",
        session=T_SESSION,
        asset_id=ASSET_A,
        quantity=10,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    sell = sell_event(
        "SELL-1",
        session=X_SESSION,
        asset_id=ASSET_A,
        quantity=10,
        fill_price=Decimal("120"),
        execution_cost=Decimal("1"),
        settlement_id="SETTLE-1",
        settlement_session=X_PLUS_1,
    )
    reentry = buy_event(
        "BUY-2",
        session=X_SESSION,
        asset_id=ASSET_A,
        quantity=5,
        fill_price=Decimal("120"),
        execution_cost=Decimal("1"),
    )
    applied_evidence = make_dividend_evidence(event_id="D:APPLIED", security_id=ASSET_A)
    noop_evidence = make_dividend_evidence(
        event_id="D:NOOP", security_id=ASSET_B, snapshot_fingerprint="d" * 64
    )
    sessions = (
        PortfolioSessionInput(session=T_SESSION, execution_events=(buy,)),
        PortfolioSessionInput(
            session=X_SESSION,
            execution_events=(sell, reentry),
            dividend_evidence=(applied_evidence, noop_evidence),
        ),
    )
    result = PortfolioBacktestOrchestrator.run(initial, sessions)
    return result, sessions


def build_replay_run() -> (
    tuple[PortfolioBacktestResult, tuple[PortfolioSessionInput, ...], PortfolioState]
):
    """A run whose initial state already carries a prior dividend
    application's fingerprint, so re-supplying the same evidence on X
    produces REPLAYED rather than APPLIED (no double credit)."""

    pre = PortfolioBacktestOrchestrator.run(
        initial_state(Decimal("100000")),
        (
            PortfolioSessionInput(
                session=T_SESSION,
                execution_events=(
                    buy_event(
                        "BUY-1",
                        session=T_SESSION,
                        asset_id=ASSET_A,
                        quantity=10,
                        fill_price=Decimal("100"),
                        execution_cost=Decimal("1"),
                    ),
                ),
            ),
        ),
    )
    opened = pre.final_state
    evidence = make_dividend_evidence(event_id="D:REPLAY", security_id=ASSET_A)
    application_id = compute_dividend_application_id(
        canonical_distribution_event_id=evidence.canonical_distribution_event_id,
        entitlement_session=evidence.entitlement_session,
        ex_session=evidence.ex_session,
        asset_id=ASSET_A,
        attribution_trade_id="BUY-1",
    )
    gross_cash = compute_gross_dividend_cash(10, evidence.amount_per_share)
    payload_hash = compute_dividend_application_payload_hash(
        application_id=application_id,
        evidence=evidence,
        asset_id=ASSET_A,
        attribution_trade_id="BUY-1",
        q_t=10,
        gross_cash_amount=gross_cash,
    )
    seeded_initial = PortfolioState(
        base_currency=opened.base_currency,
        as_of_session=opened.as_of_session,
        state_version=opened.state_version,
        settled_cash=opened.settled_cash,
        open_positions=opened.open_positions,
        pending_settlements=opened.pending_settlements,
        applied_events=opened.applied_events
        + (
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.DIVIDEND,
                event_id=application_id,
                payload_sha256=payload_hash,
            ),
        ),
    )
    sessions = (
        PortfolioSessionInput(session=X_SESSION, dividend_evidence=(evidence,)),
    )
    result = PortfolioBacktestOrchestrator.run(seeded_initial, sessions)
    return result, sessions, seeded_initial


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
    import json

    return json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def rewrite_manifest(directory: Path, **updates: object) -> None:
    payload = manifest_payload(directory)
    payload.update(updates)
    (directory / MANIFEST_FILENAME).write_bytes(canonical_payload_bytes(payload))


def rewrite_dividend_evidence(
    directory: Path,
    records: tuple[CanonicalDividendAccountingEvidence, ...],
    *,
    update_manifest: bool,
) -> None:
    table = pa.Table.from_pylist(
        [persistence._dividend_evidence_row(item) for item in records],
        schema=DIVIDEND_EVIDENCE_ARROW_SCHEMA,
    )
    pq.write_table(table, directory / DIVIDEND_EVIDENCE_FILENAME, compression="zstd")
    if update_manifest:
        rewrite_manifest(
            directory,
            dividend_evidence_content_sha256=hash_dividend_evidence(records),
            dividend_evidence_count=len(records),
        )


def rewrite_dividend_outcomes(
    directory: Path,
    outcomes: tuple,
    *,
    update_manifest: bool,
) -> None:
    table = pa.Table.from_pylist(
        [persistence._dividend_outcome_row(item) for item in outcomes],
        schema=DIVIDEND_OUTCOME_ARROW_SCHEMA,
    )
    pq.write_table(table, directory / DIVIDEND_OUTCOMES_FILENAME, compression="zstd")
    if update_manifest:
        rewrite_manifest(
            directory,
            dividend_outcomes_content_sha256=hash_dividend_outcomes(outcomes),
            dividend_outcome_count=len(outcomes),
        )


def rewrite_dividend_ledger(
    directory: Path,
    ledger: tuple,
    *,
    update_manifest: bool,
) -> None:
    table = pa.Table.from_pylist(
        [persistence._dividend_ledger_row(item) for item in ledger],
        schema=DIVIDEND_LEDGER_ARROW_SCHEMA,
    )
    pq.write_table(table, directory / DIVIDEND_LEDGER_FILENAME, compression="zstd")
    if update_manifest:
        rewrite_manifest(
            directory,
            dividend_ledger_content_sha256=hash_dividend_ledger(ledger),
            dividend_ledger_entry_count=len(ledger),
        )


# ---------------------------------------------------------------------------
# A. Persistence schema / round-trip
# ---------------------------------------------------------------------------


def test_full_round_trip_preserves_all_dividend_artifacts_exactly(
    tmp_path: Path,
) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    artifact_set = PortfolioArtifactReader.read(directory)

    assert set(
        item.canonical_distribution_event_id for item in artifact_set.dividend_evidence
    ) == {"D:APPLIED", "D:NOOP"}
    assert len(artifact_set.dividend_outcomes) == 2
    assert len(artifact_set.dividend_ledger_entries) == 1

    row = artifact_set.dividend_ledger_entries[0]
    assert row.event_type is PortfolioLedgerEventType.DIVIDEND_APPLIED
    assert row.canonical_distribution_event_id == "D:APPLIED"
    assert row.attribution_trade_id == "BUY-1"
    assert row.q_t == 10
    evidence = next(
        item
        for item in artifact_set.dividend_evidence
        if item.canonical_distribution_event_id == "D:APPLIED"
    )
    assert row.d_h == evidence.amount_per_share
    assert row.gross_cash_amount == compute_gross_dividend_cash(
        10, evidence.amount_per_share
    )

    outcome_by_id = {
        item.canonical_distribution_event_id: item
        for item in artifact_set.dividend_outcomes
    }
    assert outcome_by_id["D:APPLIED"].status is DividendApplicationStatus.APPLIED
    assert outcome_by_id["D:NOOP"].status is DividendApplicationStatus.NO_OP_NOT_ENTITLED
    assert outcome_by_id["D:NOOP"].q_t == 0
    assert outcome_by_id["D:NOOP"].attribution_trade_id is None


def test_replayed_outcome_round_trips_exactly(tmp_path: Path) -> None:
    result, inputs, seeded_initial = build_replay_run()
    directory = write_artifact(tmp_path, result, inputs)
    artifact_set = PortfolioArtifactReader.read(directory)

    assert len(artifact_set.dividend_outcomes) == 1
    outcome = artifact_set.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.REPLAYED
    assert outcome.q_t == 10
    assert outcome.attribution_trade_id == "BUY-1"
    assert artifact_set.dividend_ledger_entries == ()
    assert artifact_set.final_state.settled_cash == seeded_initial.settled_cash


def test_scale_38_d_h_and_sub_cent_gross_cash_round_trip_exactly(
    tmp_path: Path,
) -> None:
    initial = initial_state(Decimal("100000"))
    buy = buy_event(
        "BUY-1",
        session=T_SESSION,
        asset_id=ASSET_A,
        quantity=3,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    evidence = make_dividend_evidence(
        event_id="D:SUBCENT",
        d_capitalspecial=Decimal("1"),
        unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("3"),
    )
    sessions = (
        PortfolioSessionInput(session=T_SESSION, execution_events=(buy,)),
        PortfolioSessionInput(session=X_SESSION, dividend_evidence=(evidence,)),
    )
    result = PortfolioBacktestOrchestrator.run(initial, sessions)
    directory = write_artifact(tmp_path, result, sessions)
    artifact_set = PortfolioArtifactReader.read(directory)

    row = artifact_set.dividend_ledger_entries[0]
    assert row.d_h.as_tuple().exponent == -38
    assert row.gross_cash_amount == compute_gross_dividend_cash(3, row.d_h)
    assert row.gross_cash_amount != Decimal("1")
    reloaded_evidence = artifact_set.dividend_evidence[0]
    assert reloaded_evidence.amount_per_share == evidence.amount_per_share
    assert (
        reloaded_evidence.normalization_inputs.d_capitalspecial
        == evidence.normalization_inputs.d_capitalspecial
    )


def test_application_id_and_payload_hash_round_trip_exactly(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    artifact_set = PortfolioArtifactReader.read(directory)

    original_row = next(
        r
        for r in result.dividend_ledger_entries
        if r.canonical_distribution_event_id == "D:APPLIED"
    )
    reloaded_row = artifact_set.dividend_ledger_entries[0]
    assert reloaded_row.application_id == original_row.application_id
    assert reloaded_row.source_payload_sha256 == original_row.source_payload_sha256
    assert reloaded_row.canonical_distribution_snapshot_fingerprint == (
        original_row.canonical_distribution_snapshot_fingerprint
    )
    assert reloaded_row.normalization_inputs_fingerprint == (
        original_row.normalization_inputs_fingerprint
    )
    assert reloaded_row.calendar_resolution_fingerprint == (
        original_row.calendar_resolution_fingerprint
    )


def test_dividend_evidence_parquet_rejects_extra_column(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    table = pq.read_table(directory / DIVIDEND_EVIDENCE_FILENAME)
    table = table.append_column("unexpected_extra_field", pa.array(["x"] * table.num_rows))
    pq.write_table(table, directory / DIVIDEND_EVIDENCE_FILENAME, compression="zstd")

    with pytest.raises(SchemaVersionError):
        PortfolioArtifactReader.read(directory)


def _run_with_raw_input(
    raw_value: Decimal,
    *,
    unadjusted_close_t: Decimal = Decimal("1"),
    close_capital_t: Decimal = Decimal("3"),
) -> tuple[PortfolioBacktestResult, tuple[PortfolioSessionInput, ...], CanonicalDividendAccountingEvidence]:
    initial = initial_state(Decimal("100000"))
    buy = buy_event(
        "BUY-1", session=T_SESSION, asset_id=ASSET_A, quantity=3,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    evidence = make_dividend_evidence(
        event_id="D:RAWINPUT",
        d_capitalspecial=raw_value,
        unadjusted_close_t=unadjusted_close_t,
        close_capital_t=close_capital_t,
    )
    sessions = (
        PortfolioSessionInput(session=T_SESSION, execution_events=(buy,)),
        PortfolioSessionInput(session=X_SESSION, dividend_evidence=(evidence,)),
    )
    result = PortfolioBacktestOrchestrator.run(initial, sessions)
    return result, sessions, evidence


def test_raw_normalization_input_with_more_than_38_fractional_digits_round_trips(
    tmp_path: Path,
) -> None:
    """Risk A: Gate3DividendNormalizationInputs has no frozen scale ceiling
    of its own -- only the derived D_H is fixed to scale 38 -- so a raw
    input needing 50 fractional digits must still persist exactly rather
    than being silently rejected by a decimal256(76, 38) domain narrowing."""

    raw_value = Decimal((0, (1,) + (0,) * 49 + (1,), -50))
    assert raw_value.as_tuple().exponent == -50
    result, sessions, evidence = _run_with_raw_input(raw_value)
    directory = write_artifact(tmp_path, result, sessions)
    artifact_set = PortfolioArtifactReader.read(directory)

    reloaded = artifact_set.dividend_evidence[0].normalization_inputs.d_capitalspecial
    assert reloaded == raw_value
    assert reloaded.as_tuple() == raw_value.as_tuple()
    assert artifact_set.dividend_evidence[0].normalization_inputs_fingerprint == (
        evidence.normalization_inputs_fingerprint
    )
    assert artifact_set.dividend_evidence[0].amount_per_share == evidence.amount_per_share
    assert (
        artifact_set.dividend_ledger_entries[0].source_payload_sha256
        == result.dividend_ledger_entries[0].source_payload_sha256
    )


def test_raw_normalization_input_below_scale_38_round_trips_exactly(
    tmp_path: Path,
) -> None:
    """A very small non-zero raw input (exponent below -38) must not be
    truncated to zero or rejected by the money-column domain."""

    raw_value = Decimal((0, (1,), -50))  # 1E-50
    # 1E-50 alone would normalize to a zero D_H at frozen scale 38; scale the
    # other exact-rational factor up so the final ratio (1E-50 * 1E13 / 1)
    # rounds to a non-zero scale-38 D_H, isolating the raw-input persistence
    # question from the unrelated "D_H rounds to zero" business rule.
    result, sessions, evidence = _run_with_raw_input(
        raw_value,
        unadjusted_close_t=Decimal((0, (1,), 13)),
        close_capital_t=Decimal("1"),
    )
    directory = write_artifact(tmp_path, result, sessions)
    artifact_set = PortfolioArtifactReader.read(directory)

    reloaded = artifact_set.dividend_evidence[0].normalization_inputs.d_capitalspecial
    assert reloaded == raw_value
    assert reloaded.as_tuple() == raw_value.as_tuple()


def test_raw_normalization_input_exact_tuple_survives_trailing_zero_variants(
    tmp_path: Path,
) -> None:
    """The persisted raw input preserves the exact Decimal tuple (sign,
    digits, exponent), not merely the numeric value -- distinct tuples
    that compare equal (``1`` vs ``1.00``) must reload to distinct tuples."""

    raw_value = Decimal("1.00")
    assert raw_value.as_tuple() != Decimal("1").as_tuple()
    result, sessions, evidence = _run_with_raw_input(raw_value)
    directory = write_artifact(tmp_path, result, sessions)
    artifact_set = PortfolioArtifactReader.read(directory)

    reloaded = artifact_set.dividend_evidence[0].normalization_inputs.d_capitalspecial
    assert reloaded.as_tuple() == raw_value.as_tuple()
    assert reloaded.as_tuple() != Decimal("1").as_tuple()


# ---------------------------------------------------------------------------
# B. Authoritative ledger persistence
# ---------------------------------------------------------------------------


def test_settlement_sell_buy_dividend_rows_each_persist_exactly_once(
    tmp_path: Path,
) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    artifact_set = PortfolioArtifactReader.read(directory)

    assert len(artifact_set.ledger_entries) == len(result.ledger_entries)
    assert len(artifact_set.dividend_ledger_entries) == 1
    non_dividend_types = {entry.event_type for entry in artifact_set.ledger_entries}
    assert PortfolioLedgerEventType.DIVIDEND_APPLIED not in non_dividend_types
    dividend_types = {
        entry.event_type for entry in artifact_set.dividend_ledger_entries
    }
    assert dividend_types == {PortfolioLedgerEventType.DIVIDEND_APPLIED}


def test_persisted_dividend_order_is_independent_of_supplied_input_order(
    tmp_path: Path,
) -> None:
    initial = initial_state(Decimal("100000"))
    buy_a = buy_event(
        "BUY-A", session=T_SESSION, asset_id=ASSET_A, quantity=4,
        fill_price=Decimal("10"), execution_cost=Decimal("1"),
    )
    buy_b = buy_event(
        "BUY-B", session=T_SESSION, asset_id=ASSET_B, quantity=4,
        fill_price=Decimal("10"), execution_cost=Decimal("1"),
    )
    evidence_a = make_dividend_evidence(event_id="D:A", security_id=ASSET_A)
    evidence_b = make_dividend_evidence(
        event_id="D:B", security_id=ASSET_B, snapshot_fingerprint="e" * 64
    )

    def run_with_order(order):
        sessions = (
            PortfolioSessionInput(session=T_SESSION, execution_events=(buy_a, buy_b)),
            PortfolioSessionInput(session=X_SESSION, dividend_evidence=order),
        )
        result = PortfolioBacktestOrchestrator.run(initial, sessions)
        return result, sessions

    forward_result, forward_inputs = run_with_order((evidence_a, evidence_b))
    reversed_result, reversed_inputs = run_with_order((evidence_b, evidence_a))

    forward_dir = write_artifact(tmp_path, forward_result, forward_inputs, "forward")
    reversed_dir = write_artifact(
        tmp_path, reversed_result, reversed_inputs, "reversed"
    )
    forward_set = PortfolioArtifactReader.read(forward_dir)
    reversed_set = PortfolioArtifactReader.read(reversed_dir)

    assert forward_set.dividend_ledger_entries == reversed_set.dividend_ledger_entries
    assert [
        row.asset_id for row in forward_set.dividend_ledger_entries
    ] == sorted(row.asset_id for row in forward_set.dividend_ledger_entries)


def test_changing_dividend_ledger_content_changes_manifest_fingerprint(
    tmp_path: Path,
) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    original_manifest = manifest_payload(directory)

    tampered_row = result.dividend_ledger_entries[0].model_copy(
        update={"canonical_distribution_snapshot_fingerprint": "f" * 64}
    )
    rewrite_dividend_ledger(directory, (tampered_row,), update_manifest=True)
    tampered_manifest = manifest_payload(directory)

    assert (
        tampered_manifest["dividend_ledger_content_sha256"]
        != original_manifest["dividend_ledger_content_sha256"]
    )
    with pytest.raises((DividendProvenanceError, PortfolioStateError)):
        PortfolioArtifactReader.read(directory)


def test_persisted_dividend_row_is_explicitly_typed(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    artifact_set = PortfolioArtifactReader.read(directory)
    assert (
        artifact_set.dividend_ledger_entries[0].event_type
        is PortfolioLedgerEventType.DIVIDEND_APPLIED
    )


# ---------------------------------------------------------------------------
# C. Pre-X offline entitlement proof
# ---------------------------------------------------------------------------


def test_offline_reload_proves_q_t_and_old_trade_attribution(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    artifact_set = PortfolioArtifactReader.read(directory)

    row = artifact_set.dividend_ledger_entries[0]
    assert row.q_t == 10
    assert row.attribution_trade_id == "BUY-1"
    new_position = next(
        p for p in artifact_set.final_state.open_positions if p.asset_id == ASSET_A
    )
    assert new_position.entry_execution_id == "BUY-2"
    assert new_position.quantity == 5


def test_offline_reload_proves_zero_q_t_for_unheld_security(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    artifact_set = PortfolioArtifactReader.read(directory)

    noop = next(
        o
        for o in artifact_set.dividend_outcomes
        if o.canonical_distribution_event_id == "D:NOOP"
    )
    assert noop.q_t == 0
    assert noop.attribution_trade_id is None
    assert noop.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED


def test_tampered_persisted_q_t_fails_offline(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    original = result.dividend_ledger_entries[0]
    fake_q_t = 3
    fake_gross_cash = compute_gross_dividend_cash(fake_q_t, original.d_h)
    tampered = original.model_copy(
        update={
            "q_t": fake_q_t,
            "gross_cash_amount": fake_gross_cash,
            "settled_cash_delta": fake_gross_cash,
        }
    )
    rewrite_dividend_ledger(directory, (tampered,), update_manifest=True)

    with pytest.raises(DividendProvenanceError):
        PortfolioArtifactReader.read(directory)


def test_tampered_persisted_attribution_trade_id_fails_offline(
    tmp_path: Path,
) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    original = result.dividend_ledger_entries[0]
    tampered = original.model_copy(update={"attribution_trade_id": "BUY-2"})
    rewrite_dividend_ledger(directory, (tampered,), update_manifest=True)

    with pytest.raises(DividendProvenanceError):
        PortfolioArtifactReader.read(directory)


# ---------------------------------------------------------------------------
# D. Offline row -> evidence provenance
# ---------------------------------------------------------------------------


def test_valid_persisted_run_passes_offline_provenance(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    PortfolioArtifactReader.read(directory)  # must not raise


def test_dividend_row_with_missing_source_evidence_fails_offline(
    tmp_path: Path,
) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    rewrite_dividend_evidence(directory, (), update_manifest=True)
    # Outcomes still reference both events, but only zero evidence remains.
    with pytest.raises(DividendProvenanceError):
        PortfolioArtifactReader.read(directory)


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("amount_basis", "OTHER_BASIS"),
        ("normalization_method_id", "OTHER_METHOD"),
        ("normalization_rounding_mode", "OTHER_MODE"),
        ("normalization_arithmetic_mode", "OTHER_ARITH"),
        ("normalization_inputs_fingerprint", "1" * 64),
        ("calendar_resolution_fingerprint", "2" * 64),
        ("currency", "EUR"),
        ("canonical_distribution_snapshot_fingerprint", "3" * 64),
        ("source_payload_sha256", "4" * 64),
    ],
)
def test_dividend_row_field_mismatch_fails_offline(
    tmp_path: Path, field_name: str, new_value: object
) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    original = result.dividend_ledger_entries[0]
    dump = original.model_dump(mode="python")
    dump[field_name] = new_value
    tampered = original.model_validate(dump)
    rewrite_dividend_ledger(directory, (tampered,), update_manifest=True)

    with pytest.raises((DividendProvenanceError, PortfolioStateError)):
        PortfolioArtifactReader.read(directory)


def rewrite_dividend_ledger_raw_rows(
    directory: Path, rows: tuple[dict[str, object], ...]
) -> None:
    """Write raw row dicts directly to the dividend-ledger Parquet file,
    bypassing ``DividendLedgerEntry`` model construction entirely. Used to
    simulate on-disk corruption of fields (``d_h``, ``gross_cash_amount``,
    ``settled_cash_delta``) that ``validate_cash_identity`` makes
    impossible to construct as an internally-inconsistent *validated*
    model instance -- the corruption must still be tested at the actual
    persisted-artifact layer, not skipped."""

    table = pa.Table.from_pylist(list(rows), schema=DIVIDEND_LEDGER_ARROW_SCHEMA)
    pq.write_table(table, directory / DIVIDEND_LEDGER_FILENAME, compression="zstd")


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("d_h", Decimal((0, (9,) * 38, -38))),
        ("gross_cash_amount", Decimal("999999")),
        ("settled_cash_delta", Decimal("999999")),
    ],
)
def test_dividend_row_cash_identity_field_corruption_fails_closed(
    tmp_path: Path, field_name: str, new_value: Decimal
) -> None:
    """Slice-6 audit Risk E: d_h/gross_cash_amount/settled_cash_delta
    cannot be tampered individually via model_copy (DividendLedgerEntry's
    own validate_cash_identity rejects the inconsistency before it can
    even be constructed) -- that is defense-in-depth, not proof the
    persisted artifact fails closed. This corrupts the raw Parquet row
    directly, bypassing model construction, so the corruption is exactly
    what a genuinely damaged on-disk file could contain."""

    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    original = result.dividend_ledger_entries[0]
    raw_row = persistence._dividend_ledger_row(original)
    raw_row[field_name] = new_value
    rewrite_dividend_ledger_raw_rows(directory, (raw_row,))
    # Manifest hash deliberately left pointing at the original (valid) row
    # content -- the read must fail at the earliest legitimate layer,
    # whichever that is (hash mismatch or model reconstruction).

    with pytest.raises(
        (ArtifactHashMismatchError, PortfolioPersistenceError, DividendProvenanceError)
    ):
        PortfolioArtifactReader.read(directory)


def test_dividend_row_cannot_use_settlement_exemption_offline(tmp_path: Path) -> None:
    """A settlement-shaped row cannot be mislabeled as DIVIDEND_APPLIED and
    slip through the persisted-history dispatch."""

    initial = initial_state(Decimal("100000"))
    buy = buy_event(
        "BUY-1", session=T_SESSION, asset_id=ASSET_A, quantity=10,
        fill_price=Decimal("100"), execution_cost=Decimal("1"),
    )
    sell = sell_event(
        "SELL-1", session=X_SESSION, asset_id=ASSET_A, quantity=10,
        fill_price=Decimal("120"), execution_cost=Decimal("1"),
        settlement_id="SETTLE-1", settlement_session=X_PLUS_1,
    )
    sessions = (
        PortfolioSessionInput(session=T_SESSION, execution_events=(buy,)),
        PortfolioSessionInput(session=X_SESSION, execution_events=(sell,)),
        PortfolioSessionInput(session=X_PLUS_1),
    )
    result = PortfolioBacktestOrchestrator.run(initial, sessions)
    directory = write_artifact(tmp_path, result, sessions)
    artifact_set = PortfolioArtifactReader.read(directory)

    settlement_row = next(
        entry
        for entry in artifact_set.ledger_entries
        if entry.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED
    )
    relabeled = settlement_row.model_copy(
        update={"event_type": PortfolioLedgerEventType.DIVIDEND_APPLIED}
    )
    ledger_without_settlement = tuple(
        entry for entry in artifact_set.ledger_entries if entry is not settlement_row
    ) + (relabeled,)
    from stock_swing_d1.portfolio.portfolio_invariants import PortfolioInvariantChecker

    with pytest.raises(PortfolioStateError, match="unsupported"):
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            artifact_set.initial_state,
            artifact_set.final_state,
            ledger_without_settlement,
            artifact_set.session_snapshots,
        )


# ---------------------------------------------------------------------------
# E. Offline evidence -> outcome discharge
# ---------------------------------------------------------------------------


def test_exact_s_equals_o_passes_offline(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    artifact_set = PortfolioArtifactReader.read(directory)
    supplied = {
        item.canonical_distribution_event_id for item in artifact_set.dividend_evidence
    }
    recorded = {
        item.canonical_distribution_event_id for item in artifact_set.dividend_outcomes
    }
    assert supplied == recorded


def test_supplied_evidence_with_no_recorded_outcome_fails_offline(
    tmp_path: Path,
) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    remaining = tuple(
        o
        for o in result.dividend_outcomes
        if o.canonical_distribution_event_id != "D:NOOP"
    )
    rewrite_dividend_outcomes(directory, remaining, update_manifest=True)

    with pytest.raises(DividendProvenanceError):
        PortfolioArtifactReader.read(directory)


def test_recorded_outcome_without_supplied_evidence_fails_offline(
    tmp_path: Path,
) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    evidence_subset = tuple(
        e
        for e in inputs[1].dividend_evidence
        if e.canonical_distribution_event_id != "D:NOOP"
    )
    rewrite_dividend_evidence(directory, evidence_subset, update_manifest=True)

    with pytest.raises(DividendProvenanceError):
        PortfolioArtifactReader.read(directory)


def test_duplicate_outcome_identity_fails_offline(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    duplicated = result.dividend_outcomes + (result.dividend_outcomes[0],)
    with pytest.raises(PortfolioPersistenceError):
        rewrite_dividend_outcomes(directory, duplicated, update_manifest=True)


def test_recorded_outcome_q_t_disagreeing_with_reconstructed_state_fails_offline(
    tmp_path: Path,
) -> None:
    """OD-14.9: a recorded APPLIED outcome's q_t must match the
    independently reconstructed pre-X entitlement binding, not merely be
    internally self-consistent."""

    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    applied = next(
        o
        for o in result.dividend_outcomes
        if o.canonical_distribution_event_id == "D:APPLIED"
    )
    tampered = tuple(
        o for o in result.dividend_outcomes if o is not applied
    ) + (applied.model_copy(update={"q_t": 7}),)
    rewrite_dividend_outcomes(directory, tampered, update_manifest=True)

    with pytest.raises(DividendProvenanceError):
        PortfolioArtifactReader.read(directory)


# ---------------------------------------------------------------------------
# F. Exact Decimal / context independence
# ---------------------------------------------------------------------------


def _build_high_precision_run() -> (
    tuple[PortfolioBacktestResult, tuple[PortfolioSessionInput, ...]]
):
    initial = initial_state(Decimal("200000000"))
    buy = buy_event(
        "BUY-1",
        session=T_SESSION,
        asset_id=ASSET_A,
        quantity=123_456_789,
        fill_price=Decimal("1"),
        execution_cost=Decimal("1"),
    )
    evidence = make_dividend_evidence(
        event_id="D:PRECISION",
        d_capitalspecial=Decimal("1"),
        unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("3"),
    )
    sessions = (
        PortfolioSessionInput(session=T_SESSION, execution_events=(buy,)),
        PortfolioSessionInput(session=X_SESSION, dividend_evidence=(evidence,)),
    )
    result = PortfolioBacktestOrchestrator.run(initial, sessions)
    return result, sessions


def test_persist_reload_revalidate_is_identical_at_precision_28_and_60(
    tmp_path: Path,
) -> None:
    with decimal.localcontext() as ctx:
        ctx.prec = 28
        result_28, inputs_28 = _build_high_precision_run()
        dir_28 = write_artifact(tmp_path, result_28, inputs_28, "prec28")
        set_28 = PortfolioArtifactReader.read(dir_28)

    with decimal.localcontext() as ctx:
        ctx.prec = 60
        result_60, inputs_60 = _build_high_precision_run()
        dir_60 = write_artifact(tmp_path, result_60, inputs_60, "prec60")
        set_60 = PortfolioArtifactReader.read(dir_60)

    assert set_28.dividend_evidence[0].amount_per_share == (
        set_60.dividend_evidence[0].amount_per_share
    )
    assert set_28.dividend_ledger_entries[0].gross_cash_amount == (
        set_60.dividend_ledger_entries[0].gross_cash_amount
    )
    assert set_28.final_state.settled_cash == set_60.final_state.settled_cash
    assert set_28.dividend_ledger_entries[0].settled_cash_after == (
        set_60.dividend_ledger_entries[0].settled_cash_after
    )
    from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state

    assert hash_portfolio_state(set_28.final_state) == hash_portfolio_state(
        set_60.final_state
    )
    assert manifest_payload(dir_28)["dividend_ledger_content_sha256"] == (
        manifest_payload(dir_60)["dividend_ledger_content_sha256"]
    )


# ---------------------------------------------------------------------------
# G. Fingerprints / tamper detection
# ---------------------------------------------------------------------------


def test_tampered_dividend_ledger_content_hash_fails(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    rewrite_dividend_ledger(directory, result.dividend_ledger_entries, update_manifest=False)
    rewrite_manifest(directory, dividend_ledger_content_sha256="0" * 64)

    with pytest.raises(ArtifactHashMismatchError):
        PortfolioArtifactReader.read(directory)


def test_tampered_dividend_evidence_count_fails(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    rewrite_manifest(directory, dividend_evidence_count=999)

    with pytest.raises(ArtifactHashMismatchError):
        PortfolioArtifactReader.read(directory)


def test_tampered_dividend_outcome_status_changes_fingerprint_and_fails(
    tmp_path: Path,
) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    original_hash = manifest_payload(directory)["dividend_outcomes_content_sha256"]

    noop = next(
        o
        for o in result.dividend_outcomes
        if o.canonical_distribution_event_id == "D:NOOP"
    )
    applied = tuple(
        o for o in result.dividend_outcomes if o is not noop
    )
    tampered_noop = noop.model_copy(
        update={
            "status": DividendApplicationStatus.APPLIED,
            "q_t": 1,
            "attribution_trade_id": "BUY-1",
        }
    )
    rewrite_dividend_outcomes(
        directory, applied + (tampered_noop,), update_manifest=True
    )
    tampered_hash = manifest_payload(directory)["dividend_outcomes_content_sha256"]

    assert tampered_hash != original_hash
    with pytest.raises(DividendProvenanceError):
        PortfolioArtifactReader.read(directory)


# ---------------------------------------------------------------------------
# H. Legacy / non-dividend regression
# ---------------------------------------------------------------------------


def test_legacy_non_dividend_run_persists_and_reloads_with_empty_dividend_artifacts(
    tmp_path: Path,
) -> None:
    initial = initial_state(Decimal("10000"))
    buy = buy_event("BUY-1", session=T_SESSION, asset_id=ASSET_A, quantity=10)
    sessions = (PortfolioSessionInput(session=T_SESSION, execution_events=(buy,)),)
    result = PortfolioBacktestOrchestrator.run(initial, sessions)
    directory = write_artifact(tmp_path, result, sessions)
    artifact_set = PortfolioArtifactReader.read(directory)

    assert artifact_set.dividend_evidence == ()
    assert artifact_set.dividend_outcomes == ()
    assert artifact_set.dividend_ledger_entries == ()
    assert len(artifact_set.ledger_entries) == 1


def test_manifest_schema_version_is_v0_3(tmp_path: Path) -> None:
    """Task 5C-C bumped the manifest v0.2 -> v0.3 with the new merge-order
    identity; the written manifest binds exactly that frozen version."""

    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    assert manifest_payload(directory)["schema_version"] == (
        PORTFOLIO_MANIFEST_SCHEMA_VERSION
    )
    assert PORTFOLIO_MANIFEST_SCHEMA_VERSION == "portfolio_manifest.v0.3"


def test_pre_5cc_v0_2_manifest_fails_closed_under_the_new_reader(
    tmp_path: Path,
) -> None:
    """A v0.2 manifest carried the old `settlement_sell_buy_dividend.v1`
    merge-order literal; under the Task 5C-C reader it must fail closed
    rather than be reinterpreted under the amended chronology."""

    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    rewrite_manifest(
        directory,
        schema_version="portfolio_manifest.v0.2",
        ledger_merge_order_identity="settlement_sell_buy_dividend.v1",
    )

    with pytest.raises(SchemaVersionError):
        PortfolioArtifactReader.read(directory)


def test_old_v0_1_shaped_manifest_fails_closed(tmp_path: Path) -> None:
    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    payload = manifest_payload(directory)
    legacy_payload = {
        key: value
        for key, value in payload.items()
        if not key.startswith("dividend_")
    }
    legacy_payload["schema_version"] = "portfolio_manifest.v0.1"
    (directory / MANIFEST_FILENAME).write_bytes(
        canonical_payload_bytes(legacy_payload)
    )

    with pytest.raises((SchemaVersionError, PortfolioPersistenceError)):
        PortfolioArtifactReader.read(directory)


# ---------------------------------------------------------------------------
# I. Result-schema versioning policy (Slice-6 audit Risk B)
#
# PortfolioBacktestResult.schema_version stays "portfolio_backtest_result.v0.1"
# despite gaining dividend_ledger_entries/dividend_outcomes. That is
# conformant, not an oversight: PortfolioBacktestResult is a transient
# in-memory aggregation DTO -- it is never itself written to disk (no
# "backtest_result.json" exists) and its schema_version is never read back
# or gated on reload. The actual version-gated, round-tripped artifacts are
# PortfolioArtifactManifest (bumped v0.1 -> v0.2 in this slice) and the
# individual row/state models bound into it. This exact precedent already
# exists and was accepted in Slice 5: PortfolioTransitionResult also gained
# dividend_ledger_entries/dividend_outcomes/newly_applied_events/
# replayed_events under its unchanged "portfolio_transition_result.v0.1".
# ---------------------------------------------------------------------------


def test_backtest_result_schema_version_is_unchanged_by_dividend_fields() -> None:
    from stock_swing_d1.portfolio.portfolio_state_models import (
        PORTFOLIO_BACKTEST_RESULT_SCHEMA_VERSION,
        PORTFOLIO_TRANSITION_RESULT_SCHEMA_VERSION,
    )

    assert "dividend_ledger_entries" in PortfolioBacktestResult.model_fields
    assert "dividend_outcomes" in PortfolioBacktestResult.model_fields
    assert PORTFOLIO_BACKTEST_RESULT_SCHEMA_VERSION == "portfolio_backtest_result.v0.1"
    # Same already-accepted Slice-5 precedent for the sibling per-session
    # result contract, which carries the identical dividend fields.
    assert PORTFOLIO_TRANSITION_RESULT_SCHEMA_VERSION == (
        "portfolio_transition_result.v0.1"
    )


def test_backtest_result_is_never_an_independently_persisted_or_version_gated_artifact(
    tmp_path: Path,
) -> None:
    """Proves PortfolioBacktestResult.schema_version is not part of the
    round-tripped persistence contract: no manifest field is bound to it,
    and no file named for it exists in a written artifact directory."""

    from stock_swing_d1.portfolio.portfolio_state_models import (
        PORTFOLIO_BACKTEST_RESULT_SCHEMA_VERSION,
    )

    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    payload = manifest_payload(directory)

    assert PORTFOLIO_BACKTEST_RESULT_SCHEMA_VERSION not in payload.values()
    written_names = {path.name for path in directory.iterdir()}
    assert not any("backtest_result" in name for name in written_names)


# ---------------------------------------------------------------------------
# J. Authoritative ledger merge/order identity (Slice-6 audit Risk C)
# ---------------------------------------------------------------------------


def test_manifest_binds_explicit_ledger_merge_order_identity(tmp_path: Path) -> None:
    from stock_swing_d1.portfolio.portfolio_persistence import (
        PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY,
    )

    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    payload = manifest_payload(directory)

    assert payload["ledger_merge_order_identity"] == (
        PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY
    )
    # Task 5C-C: the literal explicitly names the amended chronology
    # (PRIOR SELL -> BUY -> ROUND_TRIP SELL) and is not the pre-5C-C v1.
    assert PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY == (
        "settlement_prior_sell_buy_round_trip_sell_dividend.v2"
    )
    assert PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY != (
        "settlement_sell_buy_dividend.v1"
    )


def test_changed_ledger_merge_order_identity_fails_closed_even_with_unchanged_rows(
    tmp_path: Path,
) -> None:
    """A manifest claiming a different (unrecognized) merge-order
    interpretation must fail closed even though the two underlying row
    files are byte-identical -- the identity is part of what a consumer
    trusts, not derivable from row content alone."""

    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)

    rewrite_manifest(
        directory, ledger_merge_order_identity="dividend_sell_buy_settlement.v1"
    )

    with pytest.raises((SchemaVersionError, PortfolioPersistenceError)):
        PortfolioArtifactReader.read(directory)


# ---------------------------------------------------------------------------
# K. Application-reference retention (Slice-6 audit Risk D)
#
# OD-14.7 literally requires retaining: the complete supplied evidence
# tuple, the distribution snapshot identity, the pre-transition/session-start
# state (or a reference sufficient to reproduce it), "the deterministic
# application/outcome identity and payload hash", and "the recorded
# dividend outcome tuple" (OD-14.8). It does not name PortfolioEventReference
# / newly_applied_events / replayed_events. Those generic, event-kind-spanning
# tuples on PortfolioTransitionResult are consumed by Phase 15D's existing
# _application_statuses (backtest_results/source_validation.py) filtered to
# PortfolioEventKind.EXECUTION only -- dividends are not read through that
# path. For dividends, the persisted DividendApplicationOutcome.status field
# IS the frozen "recorded outcome" that distinguishes APPLIED from REPLAYED;
# these tests prove that fact, combined with the reconstructed applied-state
# mutation, is independently sufficient -- no PortfolioEventReference
# persistence is required or added.
# ---------------------------------------------------------------------------


def test_dividend_outcome_status_alone_distinguishes_applied_from_replayed(
    tmp_path: Path,
) -> None:
    applied_result, applied_inputs = build_applied_and_noop_run()
    applied_dir = write_artifact(tmp_path, applied_result, applied_inputs, "applied")
    applied_set = PortfolioArtifactReader.read(applied_dir)
    applied_outcome = next(
        o
        for o in applied_set.dividend_outcomes
        if o.canonical_distribution_event_id == "D:APPLIED"
    )
    assert applied_outcome.status is DividendApplicationStatus.APPLIED
    assert any(
        row.application_id == applied_outcome.application_id
        for row in applied_set.dividend_ledger_entries
    )

    replay_result, replay_inputs, seeded_initial = build_replay_run()
    replay_dir = write_artifact(tmp_path, replay_result, replay_inputs, "replay")
    replay_set = PortfolioArtifactReader.read(replay_dir)
    replay_outcome = replay_set.dividend_outcomes[0]
    assert replay_outcome.status is DividendApplicationStatus.REPLAYED
    # REPLAYED produces zero new dividend ledger rows -- the outcome status
    # is the only persisted signal distinguishing this from APPLIED, and it
    # is sufficient: no dividend_ledger.parquet row exists for it at all.
    assert replay_set.dividend_ledger_entries == ()
    assert replay_set.final_state.settled_cash == seeded_initial.settled_cash


def test_new_application_fingerprint_provable_from_state_row_and_outcome_alone(
    tmp_path: Path,
) -> None:
    """An APPLIED event's fingerprint must appear exactly once, newly, in
    final_state.applied_events -- proven here purely from persisted
    initial_state + dividend_ledger row + outcome, with no
    PortfolioEventReference involved."""

    result, inputs = build_applied_and_noop_run()
    directory = write_artifact(tmp_path, result, inputs)
    artifact_set = PortfolioArtifactReader.read(directory)

    row = artifact_set.dividend_ledger_entries[0]
    key = (PortfolioEventKind.DIVIDEND, row.application_id)
    initial_keys = {
        (fp.event_kind, fp.event_id) for fp in artifact_set.initial_state.applied_events
    }
    final_keys = {
        (fp.event_kind, fp.event_id) for fp in artifact_set.final_state.applied_events
    }
    assert key not in initial_keys
    assert key in final_keys

    outcome = next(
        o
        for o in artifact_set.dividend_outcomes
        if o.canonical_distribution_event_id == row.canonical_distribution_event_id
    )
    assert outcome.status is DividendApplicationStatus.APPLIED
    assert outcome.application_id == row.application_id


def test_no_portfolio_event_reference_shape_is_persisted_for_dividends() -> None:
    """Confirms this slice did not (and does not need to) introduce a
    parallel PortfolioEventReference-shaped registry for dividends."""

    from stock_swing_d1.portfolio.portfolio_persistence import (
        DIVIDEND_LEDGER_ARROW_SCHEMA,
        DIVIDEND_OUTCOME_ARROW_SCHEMA,
    )

    for schema in (DIVIDEND_LEDGER_ARROW_SCHEMA, DIVIDEND_OUTCOME_ARROW_SCHEMA):
        names = set(schema.names)
        assert "event_kind" not in names
        assert "newly_applied" not in names
        assert "replayed" not in names
