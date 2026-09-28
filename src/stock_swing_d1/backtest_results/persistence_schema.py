"""Phase 15D.4: private physical persistence schema for one audit-result bundle.

Every helper here is implementation detail of `persistence.py`. Nothing in
this module reprices an execution, reruns an upstream economic owner, or
recomputes a settlement date; it only converts already-final Phase 15D.3
model instances to/from an explicit, deterministic physical representation
(Parquet for the 15 ordered tuple components, canonical JSON for the 4
singleton payloads and the root manifest) and back, exactly.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, ValidationError

from stock_swing_d1.backtest_results.canonical import semantic_json_bytes
from stock_swing_d1.backtest_results.errors import (
    HistoricalBacktestPersistenceError,
)
from stock_swing_d1.backtest_results.models import (
    ArtifactRef,
    HistoricalAllocationCandidateProvenance,
    HistoricalAllocationCycle,
    HistoricalBacktestAuditSummary,
    HistoricalBacktestContentFingerprints,
    HistoricalBacktestCostSummary,
    HistoricalBacktestEntryRecord,
    HistoricalBacktestEquityRow,
    HistoricalBacktestExecutionProvenance,
    HistoricalBacktestExitReasonRow,
    HistoricalBacktestExitReasonSummary,
    HistoricalBacktestExitRecord,
    HistoricalBacktestRejectionRecord,
    HistoricalBacktestRunManifest,
    HistoricalBacktestSessionPnl,
    HistoricalBacktestSummary,
    HistoricalBacktestTransitionAudit,
    HistoricalBacktestValuationPolicyRef,
    HistoricalCashLedgerRow,
    HistoricalClosedTradeRecord,
    HistoricalOpenTradeRecord,
    HistoricalRankingCandidateProvenance,
    HistoricalRankingCycle,
    HistoricalSettlementRecord,
    HistoricalSignalProvenance,
    HistoricalTradeStatus,
    PolicyArtifactRef,
)
from stock_swing_d1.backtester.decision_interval import (
    HistoricalDecisionInterval,
)
from stock_swing_d1.execution.costs.models import ExecutionCostPolicyRef
from stock_swing_d1.portfolio.portfolio_events import AppliedEventFingerprint
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PendingSettlement,
    PortfolioState,
)
from stock_swing_d1.ranking.models import RankingPolicyRef

# ---------------------------------------------------------------------------
# Frozen physical types
# ---------------------------------------------------------------------------

DECIMAL_ARROW_TYPE = pa.decimal256(76, 38)
_TIMESTAMP_ARROW_TYPE = pa.timestamp("us", tz="UTC")
_DATE_ARROW_TYPE = pa.date32()

# Bumped v0.2 -> v0.3 (Slice 9, OD-22.2 "Phase 15D persistence
# schemas/manifests"): CASH_LEDGER_SCHEMA gained the nullable
# `attribution_trade_id` column carried by versioned `HistoricalCashLedgerRow`
# v0.2. A bundle persisted under v0.2 never wrote that column, so it must
# fail closed here rather than be silently read as if it were v0.3 (OD-22.1:
# dividend-aware results require rerun, never backfill).
# Bumped v0.3 -> v0.4 (Slice 10, OD-22.2 "Phase 15D persistence
# schemas/manifests"): TRADES_SCHEMA gained four nullable
# HistoricalClosedTradeRecord-only columns (`ordinary_dividend_income`,
# `ordinary_dividend_event_count`, `dividend_attribution_completeness`,
# `ordinary_dividend_attribution_fingerprint`) carried by versioned
# `HistoricalClosedTradeRecord` v0.2. Same fail-closed/no-backfill rule.
# Bumped v0.4 -> v0.5 (Slice 11, OD-22.2 "Phase 15D persistence
# schemas/manifests"): TRADES_SCHEMA gained one nullable
# HistoricalClosedTradeRecord-only column (`trade_total_pnl`) carried by
# versioned `HistoricalClosedTradeRecord` v0.3. Same fail-closed/
# no-backfill rule: a bundle persisted under v0.4 never wrote that
# column and must not be read as if it silently gained one.
# Bumped v0.5 -> v0.6 (Slice 12, OD-22.2 "Phase 15D persistence
# schemas/manifests"): SESSION_PNL_SCHEMA and EQUITY_CURVE_SCHEMA each
# gained two nullable portfolio-scope ordinary-dividend columns
# (`ordinary_dividend_income_this_session`, `cumulative_ordinary_
# dividend_income`) carried by versioned `HistoricalBacktestSessionPnl`/
# `HistoricalBacktestEquityRow` v0.2, and `summary.json` gained the
# nullable `ordinary_dividend_income_total` field on
# `HistoricalBacktestSummary`. Same fail-closed/no-backfill rule: a
# bundle persisted under v0.5 never wrote those columns/field.
BUNDLE_SCHEMA_VERSION = "historical_backtest_result_bundle.v0.6"
MANIFEST_FILENAME = "manifest.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
INITIAL_STATE_FILENAME = "initial_state.json"
FINAL_STATE_FILENAME = "final_state.json"
SUMMARY_FILENAME = "summary.json"

TABLE_NAMES: tuple[str, ...] = (
    "session_transitions",
    "signal_provenance",
    "ranking_cycles",
    "ranking_candidates",
    "allocation_cycles",
    "allocation_candidates",
    "execution_provenance",
    "entries",
    "exits",
    "trades",
    "rejections",
    "cash_ledger",
    "settlement_ledger",
    "session_pnl",
    "equity_curve",
)

PARQUET_FILENAMES: tuple[str, ...] = tuple(f"{name}.parquet" for name in TABLE_NAMES)

JSON_PAYLOAD_FILENAMES: tuple[str, ...] = (
    RUN_MANIFEST_FILENAME,
    INITIAL_STATE_FILENAME,
    FINAL_STATE_FILENAME,
    SUMMARY_FILENAME,
)

PAYLOAD_FILENAMES: tuple[str, ...] = JSON_PAYLOAD_FILENAMES + PARQUET_FILENAMES

BUNDLE_FILENAMES: tuple[str, ...] = (MANIFEST_FILENAME,) + PAYLOAD_FILENAMES


class _BundleManifest(BaseModel):
    """Physical bundle-control document (`manifest.json`). Never self-hashed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    persistence_schema_version: Literal["historical_backtest_result_bundle.v0.6"] = (
        BUNDLE_SCHEMA_VERSION
    )
    result_schema_version: str
    run_configuration_fingerprint: str
    source_run_fingerprint: str
    initial_state_fingerprint: str
    final_state_fingerprint: str
    result_fingerprint: str
    content_fingerprints: HistoricalBacktestContentFingerprints
    payload_files: tuple[str, ...]
    row_counts: dict[str, int]
    payload_sha256: dict[str, str]


def encode_bundle_manifest(manifest: _BundleManifest) -> bytes:
    return semantic_json_bytes(manifest)


def decode_bundle_manifest(raw_bytes: bytes) -> _BundleManifest:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_READ_FAILED", "manifest.json is not valid UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict):
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_READ_FAILED", "manifest.json must contain a JSON object"
        )
    if payload.get("persistence_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise HistoricalBacktestPersistenceError(
            "UNSUPPORTED_SCHEMA_VERSION",
            "manifest.json uses an unsupported persistence schema; rerun the "
            "historical backtest with an explicit decision_interval",
        )
    try:
        return _BundleManifest.model_validate(payload)
    except (ValidationError, TypeError, ValueError) as error:
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_READ_FAILED", "manifest.json fails structural validation"
        ) from error


def build_bundle_manifest(
    *,
    result_schema_version: str,
    run_configuration_fingerprint: str,
    source_run_fingerprint: str,
    initial_state_fingerprint: str,
    final_state_fingerprint: str,
    result_fingerprint: str,
    content_fingerprints: HistoricalBacktestContentFingerprints,
    row_counts: dict[str, int],
    payload_sha256: dict[str, str],
) -> _BundleManifest:
    return _BundleManifest(
        result_schema_version=result_schema_version,
        run_configuration_fingerprint=run_configuration_fingerprint,
        source_run_fingerprint=source_run_fingerprint,
        initial_state_fingerprint=initial_state_fingerprint,
        final_state_fingerprint=final_state_fingerprint,
        result_fingerprint=result_fingerprint,
        content_fingerprints=content_fingerprints,
        payload_files=PAYLOAD_FILENAMES,
        row_counts=row_counts,
        payload_sha256=payload_sha256,
    )


# ---------------------------------------------------------------------------
# Physical file hashing
# ---------------------------------------------------------------------------


def sha256_bytes(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_READ_FAILED", f"failed to read {path.name} for hashing"
        ) from error


# ---------------------------------------------------------------------------
# Exact Decimal -> decimal256(76, 38) representability preflight
# ---------------------------------------------------------------------------


def require_exact_decimal(value: Decimal, field_name: str) -> Decimal:
    """Prove `value` is exactly representable by `DECIMAL_ARROW_TYPE`.

    Never rounds, quantizes, or truncates. Raises
    `DECIMAL_NOT_EXACTLY_REPRESENTABLE` instead of altering the value.
    """

    if type(value) is not Decimal:
        raise HistoricalBacktestPersistenceError(
            "DECIMAL_NOT_EXACTLY_REPRESENTABLE",
            f"{field_name} must be an exact Decimal",
        )
    if not value.is_finite():
        raise HistoricalBacktestPersistenceError(
            "DECIMAL_NOT_EXACTLY_REPRESENTABLE",
            f"{field_name} must be finite",
        )
    if value.is_zero():
        return value
    _, digits, exponent = value.as_tuple()
    trimmed = list(digits)
    while len(trimmed) > 1 and trimmed[-1] == 0 and exponent < 0:
        trimmed.pop()
        exponent += 1
    scale = max(-exponent, 0)
    integer_digits = max(value.copy_abs().adjusted() + 1, 0)
    if scale > DECIMAL_ARROW_TYPE.scale or integer_digits > (
        DECIMAL_ARROW_TYPE.precision - DECIMAL_ARROW_TYPE.scale
    ):
        raise HistoricalBacktestPersistenceError(
            "DECIMAL_NOT_EXACTLY_REPRESENTABLE",
            f"{field_name} cannot be represented exactly as {DECIMAL_ARROW_TYPE}",
        )
    return value


# ---------------------------------------------------------------------------
# Generic Pydantic-model <-> Arrow-row codec (Parquet side)
# ---------------------------------------------------------------------------


def _encode_leaf(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return require_exact_decimal(value, field_name)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


def encode_model_row(model: BaseModel, ordinal: int) -> dict[str, Any]:
    """Encode one frozen Phase 15D model instance into one physical row dict."""

    row: dict[str, Any] = {"ordinal": ordinal}
    for name in type(model).model_fields:
        row[name] = _encode_leaf(getattr(model, name), name)
    return row


def decode_model_row(model_cls: type[BaseModel], row: dict[str, Any]) -> BaseModel:
    """Decode one physical row dict (ordinal already stripped) into a model."""

    kwargs: dict[str, Any] = {}
    for name, value in row.items():
        kwargs[name] = tuple(value) if isinstance(value, list) else value
    try:
        return model_cls(**kwargs)
    except (ValidationError, TypeError, ValueError) as error:
        raise HistoricalBacktestPersistenceError(
            "PERSISTED_MODEL_INVALID",
            f"a persisted {model_cls.__name__} row fails model validation",
        ) from error


def build_arrow_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    try:
        return pa.Table.from_pylist(rows, schema=schema)
    except (
        pa.ArrowInvalid,
        pa.ArrowTypeError,
        pa.ArrowException,
        TypeError,
        ValueError,
    ) as error:
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_WRITE_FAILED",
            "values cannot be represented by the frozen Arrow schema",
        ) from error


def write_parquet_table(table: pa.Table, path: Path) -> None:
    try:
        pq.write_table(table, path, compression="zstd")
    except (OSError, pa.ArrowException) as error:
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_WRITE_FAILED", f"failed to write {path.name}"
        ) from error


def read_parquet_table(path: Path, schema: pa.Schema, table_name: str) -> pa.Table:
    if not path.is_file():
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_INVENTORY_MISMATCH", f"missing required payload file: {path.name}"
        )
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as error:
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_READ_FAILED", f"failed to read {path.name}"
        ) from error
    if not table.schema.equals(schema, check_metadata=False):
        raise HistoricalBacktestPersistenceError(
            "PARQUET_SCHEMA_MISMATCH", f"{table_name} Arrow schema mismatch"
        )
    return table


def verify_and_strip_ordinal(table: pa.Table, table_name: str) -> list[dict[str, Any]]:
    """Prove physical row order was preserved; never resorts to repair it."""

    rows = table.to_pylist()
    ordinals = [row.get("ordinal") for row in rows]
    if any(type(value) is not int for value in ordinals):
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_READ_FAILED",
            f"{table_name} ordinal column must be a genuine integer",
        )
    if ordinals != list(range(len(rows))):
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_READ_FAILED",
            f"{table_name} ordinal sequence is not exactly 0..N-1 in physical "
            "row order",
        )
    for row in rows:
        row.pop("ordinal", None)
    return rows


# ---------------------------------------------------------------------------
# The 15 explicit ordered-table Arrow schemas
# ---------------------------------------------------------------------------

_ORDINAL_FIELD = pa.field("ordinal", pa.int64(), nullable=False)

SESSION_TRANSITIONS_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("decision_time", _TIMESTAMP_ARROW_TYPE, nullable=False),
        pa.field("state_hash_before", pa.string(), nullable=False),
        pa.field("state_hash_after", pa.string(), nullable=False),
        pa.field("state_version_before", pa.int64(), nullable=False),
        pa.field("state_version_after", pa.int64(), nullable=False),
        pa.field("settled_cash_before", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("settled_cash_after", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("open_position_count_before", pa.int64(), nullable=False),
        pa.field("open_position_count_after", pa.int64(), nullable=False),
        pa.field("pending_settlement_count_before", pa.int64(), nullable=False),
        pa.field("pending_settlement_count_after", pa.int64(), nullable=False),
        pa.field("newly_applied_event_ids", pa.list_(pa.string()), nullable=False),
        pa.field("replayed_event_ids", pa.list_(pa.string()), nullable=False),
        pa.field("ledger_entry_count", pa.int64(), nullable=False),
    ]
)

SIGNAL_PROVENANCE_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("decision_time", _TIMESTAMP_ARROW_TYPE, nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("action", pa.string(), nullable=False),
        pa.field("planned_entry_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("adjusted_close", pa.float64(), nullable=False),
        pa.field("sma20", pa.float64(), nullable=True),
        pa.field("sma50", pa.float64(), nullable=True),
        pa.field("rsi14", pa.float64(), nullable=True),
        pa.field("atr14", pa.float64(), nullable=True),
        pa.field("atr_fraction", pa.float64(), nullable=True),
        pa.field("universe_eligible", pa.bool_(), nullable=False),
        pa.field("close_above_sma50", pa.bool_(), nullable=False),
        pa.field("sma20_above_sma50", pa.bool_(), nullable=False),
        pa.field("rsi_above_50", pa.bool_(), nullable=False),
        pa.field("atr_above_minimum", pa.bool_(), nullable=False),
        pa.field("earnings_entry_allowed", pa.bool_(), nullable=False),
        pa.field("earnings_action", pa.string(), nullable=False),
        pa.field("source_payload_fingerprint", pa.string(), nullable=False),
    ]
)

RANKING_CYCLES_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("ranking_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("decision_time", _TIMESTAMP_ARROW_TYPE, nullable=False),
        pa.field("policy_id", pa.string(), nullable=False),
        pa.field("policy_version", pa.string(), nullable=False),
        pa.field("policy_fingerprint", pa.string(), nullable=False),
        pa.field("candidate_count", pa.int64(), nullable=False),
        pa.field("input_set_fingerprint", pa.string(), nullable=False),
        pa.field("snapshot_fingerprint", pa.string(), nullable=False),
    ]
)

RANKING_CANDIDATES_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("ranking_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("decision_time", _TIMESTAMP_ARROW_TYPE, nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("rank", pa.int64(), nullable=False),
        pa.field("trend_separation_atr", pa.float64(), nullable=False),
        pa.field("rsi14", pa.float64(), nullable=False),
        pa.field("sma20", pa.float64(), nullable=False),
        pa.field("sma50", pa.float64(), nullable=False),
        pa.field("atr14", pa.float64(), nullable=False),
        pa.field("candidate_input_fingerprint", pa.string(), nullable=False),
        pa.field("ranking_snapshot_fingerprint", pa.string(), nullable=False),
    ]
)

ALLOCATION_CYCLES_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("allocation_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("decision_time", _TIMESTAMP_ARROW_TYPE, nullable=False),
        pa.field("ranking_snapshot_fingerprint", pa.string(), nullable=False),
        pa.field("allocation_policy_fingerprint", pa.string(), nullable=False),
        pa.field("candidate_count", pa.int64(), nullable=False),
        pa.field("admitted_count", pa.int64(), nullable=False),
        pa.field("rejected_count", pa.int64(), nullable=False),
        pa.field("settled_cash_input", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("portfolio_equity_input", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("open_position_count_input", pa.int64(), nullable=False),
    ]
)

ALLOCATION_CANDIDATES_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("allocation_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("source_rank", pa.int64(), nullable=False),
        pa.field("processing_rank", pa.int64(), nullable=False),
        pa.field("ranking_snapshot_fingerprint", pa.string(), nullable=False),
        pa.field("ranking_input_fingerprint", pa.string(), nullable=False),
        pa.field("action", pa.string(), nullable=False),
        pa.field("candidate_cash_limit", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("reserved_cash", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("fixed_shares", pa.int64(), nullable=True),
        pa.field("used_slots_before", pa.int64(), nullable=False),
        pa.field("used_slots_after", pa.int64(), nullable=False),
        pa.field("unreserved_cash_before", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("unreserved_cash_after", DECIMAL_ARROW_TYPE, nullable=False),
    ]
)

EXECUTION_PROVENANCE_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("event_order", pa.int64(), nullable=False),
        pa.field("execution_id", pa.string(), nullable=False),
        pa.field("source_order_id", pa.string(), nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        pa.field("quantity", pa.int64(), nullable=False),
        pa.field("fill_price", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("execution_cost", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("settlement_id", pa.string(), nullable=True),
        pa.field("settlement_session", _DATE_ARROW_TYPE, nullable=True),
        pa.field("application_status", pa.string(), nullable=False),
        pa.field("provenance_source", pa.string(), nullable=False),
        pa.field("source_payload_fingerprint", pa.string(), nullable=False),
        pa.field("entry_decision_fingerprint", pa.string(), nullable=True),
        pa.field("exit_decision_fingerprint", pa.string(), nullable=True),
    ]
)

ENTRIES_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("entry_execution_id", pa.string(), nullable=False),
        pa.field("source_order_id", pa.string(), nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("quantity", pa.int64(), nullable=False),
        pa.field("entry_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("fill_price", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("execution_cost", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("cost_basis", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("provenance_source", pa.string(), nullable=False),
        pa.field("signal_session", _DATE_ARROW_TYPE, nullable=True),
        pa.field("signal_time", _TIMESTAMP_ARROW_TYPE, nullable=True),
        pa.field("allocation_session", _DATE_ARROW_TYPE, nullable=True),
        pa.field("source_rank", pa.int64(), nullable=True),
        pa.field("ranking_snapshot_fingerprint", pa.string(), nullable=True),
        pa.field("ranking_input_fingerprint", pa.string(), nullable=True),
        pa.field("entry_execution_status", pa.string(), nullable=True),
        pa.field("execution_cost_policy_fingerprint", pa.string(), nullable=True),
    ]
)

EXITS_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("exit_execution_id", pa.string(), nullable=False),
        pa.field("source_order_id", pa.string(), nullable=False),
        pa.field("entry_execution_id", pa.string(), nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("quantity", pa.int64(), nullable=False),
        pa.field("exit_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("fill_price", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("execution_cost", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("gross_proceeds", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("net_proceeds", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("settlement_id", pa.string(), nullable=False),
        pa.field("settlement_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("provenance_source", pa.string(), nullable=False),
        pa.field("exit_reason", pa.string(), nullable=False),
        pa.field("reference_exit_price", DECIMAL_ARROW_TYPE, nullable=True),
    ]
)

TRADES_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("trade_id", pa.string(), nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("quantity", pa.int64(), nullable=False),
        pa.field("carried_in", pa.bool_(), nullable=False),
        pa.field("entry_execution_id", pa.string(), nullable=False),
        pa.field("entry_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("entry_fill_price", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("entry_execution_cost", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("entry_cost_basis", DECIMAL_ARROW_TYPE, nullable=False),
        # HistoricalClosedTradeRecord-only (nullable):
        pa.field("exit_execution_id", pa.string(), nullable=True),
        pa.field("exit_session", _DATE_ARROW_TYPE, nullable=True),
        pa.field("exit_fill_price", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("exit_execution_cost", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("exit_reason", pa.string(), nullable=True),
        pa.field("gross_exit_proceeds", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("net_exit_proceeds", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("realized_pnl", DECIMAL_ARROW_TYPE, nullable=True),
        # HistoricalClosedTradeRecord-only, Slice 10 (nullable):
        pa.field("ordinary_dividend_income", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("ordinary_dividend_event_count", pa.int64(), nullable=True),
        pa.field(
            "dividend_attribution_completeness", pa.string(), nullable=True
        ),
        pa.field(
            "ordinary_dividend_attribution_fingerprint",
            pa.string(),
            nullable=True,
        ),
        # HistoricalClosedTradeRecord-only, Slice 11 (nullable):
        pa.field("trade_total_pnl", DECIMAL_ARROW_TYPE, nullable=True),
        # HistoricalOpenTradeRecord-only (nullable):
        pa.field("final_mark_session", _DATE_ARROW_TYPE, nullable=True),
        pa.field("final_mark_price", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("final_market_value", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("unrealized_pnl", DECIMAL_ARROW_TYPE, nullable=True),
    ]
)

REJECTIONS_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("decision_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("requested_quantity", pa.int64(), nullable=True),
        pa.field("reserved_cash", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("released_cash", DECIMAL_ARROW_TYPE, nullable=True),
        pa.field("source_rank", pa.int64(), nullable=True),
        pa.field("source_artifact_fingerprint", pa.string(), nullable=True),
    ]
)

CASH_LEDGER_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("sequence_in_session", pa.int64(), nullable=False),
        pa.field("ledger_event_type", pa.string(), nullable=False),
        pa.field("source_event_id", pa.string(), nullable=False),
        pa.field("source_order_id", pa.string(), nullable=True),
        pa.field("security_id", pa.string(), nullable=True),
        pa.field("settled_cash_delta", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("pending_cash_delta", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("settled_cash_after", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("settlement_id", pa.string(), nullable=True),
        pa.field("settlement_session", _DATE_ARROW_TYPE, nullable=True),
        pa.field("state_hash_before", pa.string(), nullable=False),
        pa.field("state_hash_after", pa.string(), nullable=False),
        pa.field("source_payload_fingerprint", pa.string(), nullable=False),
        pa.field("attribution_trade_id", pa.string(), nullable=True),
    ]
)

SETTLEMENT_LEDGER_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("settlement_id", pa.string(), nullable=False),
        pa.field("source_sell_execution_id", pa.string(), nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("trade_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("settlement_session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("amount", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("status", pa.string(), nullable=False),
    ]
)

SESSION_PNL_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("realized_pnl_this_session", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("cumulative_realized_pnl", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("unrealized_pnl", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("execution_cost_this_session", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("cumulative_execution_cost", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("period_pnl", DECIMAL_ARROW_TYPE, nullable=False),
        # Slice 12 (nullable; present only for a dividend-aware run):
        pa.field(
            "ordinary_dividend_income_this_session",
            DECIMAL_ARROW_TYPE,
            nullable=True,
        ),
        pa.field(
            "cumulative_ordinary_dividend_income", DECIMAL_ARROW_TYPE, nullable=True
        ),
    ]
)

EQUITY_CURVE_SCHEMA = pa.schema(
    [
        _ORDINAL_FIELD,
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("session", _DATE_ARROW_TYPE, nullable=False),
        pa.field("state_hash", pa.string(), nullable=False),
        pa.field("settled_cash", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("pending_receivable_value", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("open_position_market_value", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("equity", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("realized_pnl_this_session", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("cumulative_realized_pnl", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("unrealized_pnl", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("execution_cost_this_session", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("cumulative_execution_cost", DECIMAL_ARROW_TYPE, nullable=False),
        pa.field("period_pnl", DECIMAL_ARROW_TYPE, nullable=False),
        # Slice 12 (nullable; present only for a dividend-aware run):
        pa.field(
            "ordinary_dividend_income_this_session",
            DECIMAL_ARROW_TYPE,
            nullable=True,
        ),
        pa.field(
            "cumulative_ordinary_dividend_income", DECIMAL_ARROW_TYPE, nullable=True
        ),
    ]
)

TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "session_transitions": SESSION_TRANSITIONS_SCHEMA,
    "signal_provenance": SIGNAL_PROVENANCE_SCHEMA,
    "ranking_cycles": RANKING_CYCLES_SCHEMA,
    "ranking_candidates": RANKING_CANDIDATES_SCHEMA,
    "allocation_cycles": ALLOCATION_CYCLES_SCHEMA,
    "allocation_candidates": ALLOCATION_CANDIDATES_SCHEMA,
    "execution_provenance": EXECUTION_PROVENANCE_SCHEMA,
    "entries": ENTRIES_SCHEMA,
    "exits": EXITS_SCHEMA,
    "trades": TRADES_SCHEMA,
    "rejections": REJECTIONS_SCHEMA,
    "cash_ledger": CASH_LEDGER_SCHEMA,
    "settlement_ledger": SETTLEMENT_LEDGER_SCHEMA,
    "session_pnl": SESSION_PNL_SCHEMA,
    "equity_curve": EQUITY_CURVE_SCHEMA,
}

# Non-union tables: one frozen model class per Parquet table. `trades` is
# handled separately below (discriminated union of two model classes).
TABLE_MODEL_CLASSES: dict[str, type[BaseModel]] = {
    "session_transitions": HistoricalBacktestTransitionAudit,
    "signal_provenance": HistoricalSignalProvenance,
    "ranking_cycles": HistoricalRankingCycle,
    "ranking_candidates": HistoricalRankingCandidateProvenance,
    "allocation_cycles": HistoricalAllocationCycle,
    "allocation_candidates": HistoricalAllocationCandidateProvenance,
    "execution_provenance": HistoricalBacktestExecutionProvenance,
    "entries": HistoricalBacktestEntryRecord,
    "exits": HistoricalBacktestExitRecord,
    "rejections": HistoricalBacktestRejectionRecord,
    "cash_ledger": HistoricalCashLedgerRow,
    "settlement_ledger": HistoricalSettlementRecord,
    "session_pnl": HistoricalBacktestSessionPnl,
    "equity_curve": HistoricalBacktestEquityRow,
}


# ---------------------------------------------------------------------------
# Trade union table (`trades.parquet`): status-discriminated wide schema
# ---------------------------------------------------------------------------

_TRADES_FIELD_NAMES: tuple[str, ...] = tuple(
    field.name for field in TRADES_SCHEMA if field.name != "ordinal"
)
_CLOSED_TRADE_FIELDS = frozenset(HistoricalClosedTradeRecord.model_fields)
_OPEN_TRADE_FIELDS = frozenset(HistoricalOpenTradeRecord.model_fields)


def encode_trade_row(
    trade: HistoricalClosedTradeRecord | HistoricalOpenTradeRecord, ordinal: int
) -> dict[str, Any]:
    row: dict[str, Any] = {name: None for name in _TRADES_FIELD_NAMES}
    row.update(encode_model_row(trade, ordinal))
    return row


def decode_trade_row(
    row: dict[str, Any],
) -> HistoricalClosedTradeRecord | HistoricalOpenTradeRecord:
    status = row.get("status")
    if status == HistoricalTradeStatus.CLOSED.value:
        subset = {name: row[name] for name in _CLOSED_TRADE_FIELDS if name in row}
        model = decode_model_row(HistoricalClosedTradeRecord, subset)
        assert isinstance(model, HistoricalClosedTradeRecord)
        return model
    if status == HistoricalTradeStatus.OPEN_AT_RUN_END.value:
        subset = {name: row[name] for name in _OPEN_TRADE_FIELDS if name in row}
        model = decode_model_row(HistoricalOpenTradeRecord, subset)
        assert isinstance(model, HistoricalOpenTradeRecord)
        return model
    raise HistoricalBacktestPersistenceError(
        "PERSISTED_MODEL_INVALID",
        f"unknown trades.parquet status discriminator: {status!r}",
    )


# ---------------------------------------------------------------------------
# JSON payload codecs: run_manifest.json
# ---------------------------------------------------------------------------


def _decode_artifact_ref(payload: dict[str, Any] | None) -> ArtifactRef | None:
    if payload is None:
        return None
    return ArtifactRef(**payload)


def _decode_policy_artifact_ref(payload: dict[str, Any]) -> PolicyArtifactRef:
    return PolicyArtifactRef(**payload)


def _decode_valuation_policy_ref(
    payload: dict[str, Any],
) -> HistoricalBacktestValuationPolicyRef:
    return HistoricalBacktestValuationPolicyRef(**payload)


def _decode_ranking_policy_ref(payload: dict[str, Any]) -> RankingPolicyRef:
    return RankingPolicyRef(**payload)


def _decode_execution_cost_policy_ref(
    payload: dict[str, Any],
) -> ExecutionCostPolicyRef:
    return ExecutionCostPolicyRef(**payload)


def encode_run_manifest(manifest: HistoricalBacktestRunManifest) -> bytes:
    return semantic_json_bytes(manifest)


def decode_run_manifest(raw_bytes: bytes) -> HistoricalBacktestRunManifest:
    payload = _parse_json_object(raw_bytes, RUN_MANIFEST_FILENAME)
    try:
        return HistoricalBacktestRunManifest(
            schema_version=payload["schema_version"],
            software_revision=payload["software_revision"],
            software_revision_kind=payload["software_revision_kind"],
            strategy_configuration_ref=_decode_artifact_ref(
                payload["strategy_configuration_ref"]
            ),
            allocation_policy_ref=_decode_policy_artifact_ref(
                payload["allocation_policy_ref"]
            ),
            ranking_policy_ref=_decode_ranking_policy_ref(
                payload["ranking_policy_ref"]
            ),
            execution_cost_policy_ref=_decode_execution_cost_policy_ref(
                payload["execution_cost_policy_ref"]
            ),
            valuation_policy_ref=_decode_valuation_policy_ref(
                payload["valuation_policy_ref"]
            ),
            universe_artifact_ref=_decode_artifact_ref(
                payload["universe_artifact_ref"]
            ),
            market_data_artifact_ref=_decode_artifact_ref(
                payload["market_data_artifact_ref"]
            ),
            corporate_action_artifact_ref=_decode_artifact_ref(
                payload["corporate_action_artifact_ref"]
            ),
            earnings_artifact_ref=_decode_artifact_ref(
                payload["earnings_artifact_ref"]
            ),
            earnings_provider_name=payload["earnings_provider_name"],
            run_configuration_fingerprint=payload["run_configuration_fingerprint"],
        )
    except (ValidationError, TypeError, ValueError, KeyError) as error:
        raise HistoricalBacktestPersistenceError(
            "PERSISTED_MODEL_INVALID",
            f"{RUN_MANIFEST_FILENAME} fails model validation",
        ) from error


# ---------------------------------------------------------------------------
# JSON payload codecs: initial_state.json / final_state.json
# ---------------------------------------------------------------------------


def _decode_open_position(payload: dict[str, Any]) -> OpenPosition:
    return OpenPosition(
        asset_id=payload["asset_id"],
        quantity=payload["quantity"],
        entry_session=date.fromisoformat(payload["entry_session"]),
        entry_price=Decimal(str(payload["entry_price"])),
        entry_execution_id=payload["entry_execution_id"],
        entry_execution_cost=Decimal(str(payload["entry_execution_cost"])),
        cost_basis=Decimal(str(payload["cost_basis"])),
    )


def _decode_pending_settlement(payload: dict[str, Any]) -> PendingSettlement:
    return PendingSettlement(
        settlement_id=payload["settlement_id"],
        source_execution_id=payload["source_execution_id"],
        asset_id=payload["asset_id"],
        amount=Decimal(str(payload["amount"])),
        trade_session=date.fromisoformat(payload["trade_session"]),
        settlement_session=date.fromisoformat(payload["settlement_session"]),
    )


def _decode_applied_event_fingerprint(
    payload: dict[str, Any],
) -> AppliedEventFingerprint:
    return AppliedEventFingerprint(
        event_kind=payload["event_kind"],
        event_id=payload["event_id"],
        payload_sha256=payload["payload_sha256"],
    )


def encode_portfolio_state(state: PortfolioState) -> bytes:
    return semantic_json_bytes(state)


def decode_portfolio_state(raw_bytes: bytes, filename: str) -> PortfolioState:
    payload = _parse_json_object(raw_bytes, filename)
    try:
        return PortfolioState(
            schema_version=payload["schema_version"],
            base_currency=payload["base_currency"],
            as_of_session=(
                None
                if payload["as_of_session"] is None
                else date.fromisoformat(payload["as_of_session"])
            ),
            state_version=payload["state_version"],
            settled_cash=Decimal(str(payload["settled_cash"])),
            open_positions=tuple(
                _decode_open_position(item) for item in payload["open_positions"]
            ),
            pending_settlements=tuple(
                _decode_pending_settlement(item)
                for item in payload["pending_settlements"]
            ),
            applied_events=tuple(
                _decode_applied_event_fingerprint(item)
                for item in payload["applied_events"]
            ),
        )
    except (ValidationError, TypeError, ValueError, KeyError) as error:
        raise HistoricalBacktestPersistenceError(
            "PERSISTED_MODEL_INVALID", f"{filename} fails model validation"
        ) from error


# ---------------------------------------------------------------------------
# JSON payload codec: summary.json
# ---------------------------------------------------------------------------


def _decode_cost_summary(payload: dict[str, Any]) -> HistoricalBacktestCostSummary:
    return HistoricalBacktestCostSummary(
        schema_version=payload["schema_version"],
        buy_execution_cost_total=Decimal(str(payload["buy_execution_cost_total"])),
        sell_execution_cost_total=Decimal(str(payload["sell_execution_cost_total"])),
        total_execution_cost=Decimal(str(payload["total_execution_cost"])),
        applied_buy_count=payload["applied_buy_count"],
        applied_sell_count=payload["applied_sell_count"],
    )


def _decode_exit_reason_summary(
    payload: dict[str, Any],
) -> HistoricalBacktestExitReasonSummary:
    rows = tuple(
        HistoricalBacktestExitReasonRow(
            reason=row["reason"],
            exit_count=row["exit_count"],
            realized_pnl=Decimal(str(row["realized_pnl"])),
        )
        for row in payload["rows"]
    )
    return HistoricalBacktestExitReasonSummary(
        schema_version=payload["schema_version"], rows=rows
    )


_SUMMARY_DECIMAL_FIELDS: tuple[str, ...] = (
    "gross_realized_pnl",
    "final_unrealized_pnl",
    "initial_equity",
    "final_equity",
    "period_pnl",
    "buy_execution_cost_total",
    "sell_execution_cost_total",
    "total_execution_cost",
)
# Slice 12: unlike the fields above, this one is None for a legacy
# (non-dividend-aware) run's summary and must not be blindly re-parsed
# via `Decimal(str(None))`, which raises.
_SUMMARY_OPTIONAL_DECIMAL_FIELDS: tuple[str, ...] = (
    "ordinary_dividend_income_total",
)


def _decode_summary(payload: dict[str, Any]) -> HistoricalBacktestSummary:
    kwargs: dict[str, Any] = dict(payload)
    for name in _SUMMARY_OPTIONAL_DECIMAL_FIELDS:
        value = kwargs.get(name)
        kwargs[name] = None if value is None else Decimal(str(value))
    for name in _SUMMARY_DECIMAL_FIELDS:
        kwargs[name] = Decimal(str(kwargs[name]))
    return HistoricalBacktestSummary(**kwargs)


def _decode_audit_summary(payload: dict[str, Any]) -> HistoricalBacktestAuditSummary:
    return HistoricalBacktestAuditSummary(**payload)


class _SummaryPayload:
    """Plain carrier for `summary.json`'s decoded top-level content."""

    __slots__ = (
        "decision_interval",
        "initial_equity",
        "final_equity",
        "period_pnl",
        "cost_summary",
        "exit_reason_summary",
        "summary",
        "audit_summary",
    )

    def __init__(
        self,
        *,
        decision_interval: HistoricalDecisionInterval,
        initial_equity: Decimal,
        final_equity: Decimal,
        period_pnl: Decimal,
        cost_summary: HistoricalBacktestCostSummary,
        exit_reason_summary: HistoricalBacktestExitReasonSummary,
        summary: HistoricalBacktestSummary,
        audit_summary: HistoricalBacktestAuditSummary,
    ) -> None:
        self.decision_interval = decision_interval
        self.initial_equity = initial_equity
        self.final_equity = final_equity
        self.period_pnl = period_pnl
        self.cost_summary = cost_summary
        self.exit_reason_summary = exit_reason_summary
        self.summary = summary
        self.audit_summary = audit_summary


def encode_summary_payload(
    *,
    decision_interval: HistoricalDecisionInterval,
    initial_equity: Decimal,
    final_equity: Decimal,
    period_pnl: Decimal,
    cost_summary: HistoricalBacktestCostSummary,
    exit_reason_summary: HistoricalBacktestExitReasonSummary,
    summary: HistoricalBacktestSummary,
    audit_summary: HistoricalBacktestAuditSummary,
) -> bytes:
    return semantic_json_bytes(
        {
            "decision_interval": decision_interval,
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "period_pnl": period_pnl,
            "cost_summary": cost_summary,
            "exit_reason_summary": exit_reason_summary,
            "summary": summary,
            "audit_summary": audit_summary,
        }
    )


def decode_summary_payload(raw_bytes: bytes) -> _SummaryPayload:
    payload = _parse_json_object(raw_bytes, SUMMARY_FILENAME)
    try:
        return _SummaryPayload(
            decision_interval=_decode_decision_interval(
                payload["decision_interval"]
            ),
            initial_equity=Decimal(str(payload["initial_equity"])),
            final_equity=Decimal(str(payload["final_equity"])),
            period_pnl=Decimal(str(payload["period_pnl"])),
            cost_summary=_decode_cost_summary(payload["cost_summary"]),
            exit_reason_summary=_decode_exit_reason_summary(
                payload["exit_reason_summary"]
            ),
            summary=_decode_summary(payload["summary"]),
            audit_summary=_decode_audit_summary(payload["audit_summary"]),
        )
    except (ValidationError, TypeError, ValueError, KeyError) as error:
        raise HistoricalBacktestPersistenceError(
            "PERSISTED_MODEL_INVALID", f"{SUMMARY_FILENAME} fails model validation"
        ) from error


def _decode_decision_interval(payload: object) -> HistoricalDecisionInterval:
    if type(payload) is not dict or set(payload) != {
        "decision_start_date",
        "decision_end_date",
    }:
        raise ValueError("decision_interval must contain exactly both endpoints")
    decoded: dict[str, date] = {}
    for field_name in ("decision_start_date", "decision_end_date"):
        raw_value = payload[field_name]
        if type(raw_value) is not str:
            raise ValueError("persisted decision interval endpoints must be text")
        parsed = date.fromisoformat(raw_value)
        if parsed.isoformat() != raw_value:
            raise ValueError("persisted decision interval dates must be canonical")
        decoded[field_name] = parsed
    return HistoricalDecisionInterval(**decoded)


# ---------------------------------------------------------------------------
# Shared JSON parsing helper
# ---------------------------------------------------------------------------


def _parse_json_object(raw_bytes: bytes, filename: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_READ_FAILED", f"{filename} is not valid UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict):
        raise HistoricalBacktestPersistenceError(
            "BUNDLE_READ_FAILED", f"{filename} must contain a JSON object"
        )
    return payload
