"""Deterministic local persistence for Phase 13 portfolio artifacts."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION,
    CanonicalDividendAccountingEvidence,
    DividendCalendarResolutionProof,
    OrdinaryCashClassificationProof,
)
from stock_swing_d1.data.ordinary_dividend_normalization import (
    Gate3DividendNormalizationInputs,
    Gate3DividendNormalizationResult,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    DIVIDEND_LEDGER_ENTRY_SCHEMA_VERSION,
    DividendApplicationOutcome,
    DividendApplicationStatus,
    DividendLedgerEntry,
)
from stock_swing_d1.portfolio.portfolio_errors import (
    ArtifactHashMismatchError,
    PortfolioPersistenceError,
    PortfolioStateError,
    SchemaVersionError,
    StateHashMismatchError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    PORTFOLIO_EXECUTION_EVENT_SCHEMA_VERSION,
    PORTFOLIO_LEDGER_ENTRY_SCHEMA_VERSION,
    ExecutionSide,
    PortfolioEventKind,
    PortfolioExecutionEvent,
    PortfolioLedgerEntry,
    PortfolioLedgerEventType,
    _ImmutablePortfolioModel,
    _NonNegativeInt,
    _SessionDate,
    _Sha256,
)
from stock_swing_d1.portfolio.portfolio_hashing import (
    canonical_payload_bytes,
    canonical_payload_sha256,
    hash_execution_event,
    hash_portfolio_state,
)
from stock_swing_d1.portfolio.portfolio_invariants import (
    PortfolioInvariantChecker,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PORTFOLIO_SESSION_SNAPSHOT_SCHEMA_VERSION,
    PORTFOLIO_STATE_SCHEMA_VERSION,
    PortfolioBacktestResult,
    PortfolioSessionInput,
    PortfolioSessionSnapshot,
    PortfolioState,
)


PORTFOLIO_EXECUTION_PRESENTATION_SCHEMA_VERSION = (
    "portfolio_execution_presentation.v0.1"
)
# Bumped v0.2 -> v0.3 (Task 5C-C): the per-session application chronology
# gained the ROUND_TRIP SELL class after BUY, so the ledger merge-order
# identity below changed literal; a v0.2 manifest written under the old
# chronology fails closed on read rather than being reinterpreted.
PORTFOLIO_MANIFEST_SCHEMA_VERSION = "portfolio_manifest.v0.3"
PORTFOLIO_DIVIDEND_OUTCOME_SCHEMA_VERSION = "dividend_application_outcome.v0.1"
# The frozen chronology (settlement -> PRIOR SELL -> BUY -> ROUND_TRIP SELL
# -> dividend) is the ONLY rule for interpreting portfolio_ledger.parquet and
# dividend_ledger.parquet as one authoritative per-session sequence: within a
# session, the rows in portfolio_ledger.parquet keep their existing
# chronological order, followed by the rows in dividend_ledger.parquet in
# their existing OD-20.4 order.  This constant is a purely technical
# identifier for that frozen order (not a new economic decision) so a
# consumer can check it explicitly rather than relying on an unstated code
# convention; changing what it means requires a new literal value and
# therefore a manifest schema bump (v1 -> v2 under Task 5C-C, which inserted
# the same-session ROUND_TRIP SELL class between BUY and dividend).
PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY = (
    "settlement_prior_sell_buy_round_trip_sell_dividend.v2"
)

MANIFEST_FILENAME = "manifest.json"
INITIAL_STATE_FILENAME = "initial_state.json"
FINAL_STATE_FILENAME = "final_state.json"
EXECUTION_EVENTS_FILENAME = "execution_events.parquet"
PORTFOLIO_LEDGER_FILENAME = "portfolio_ledger.parquet"
SESSION_SNAPSHOTS_FILENAME = "session_snapshots.parquet"
DIVIDEND_EVIDENCE_FILENAME = "dividend_evidence.parquet"
DIVIDEND_OUTCOMES_FILENAME = "dividend_outcomes.parquet"
DIVIDEND_LEDGER_FILENAME = "dividend_ledger.parquet"

MONEY_ARROW_TYPE = pa.decimal256(76, 38)

EXECUTION_PRESENTATION_ARROW_SCHEMA = pa.schema(
    [
        pa.field("presentation_schema_version", pa.string(), nullable=False),
        pa.field("input_session", pa.date32(), nullable=False),
        pa.field("execution_schema_version", pa.string(), nullable=False),
        pa.field("execution_id", pa.string(), nullable=False),
        pa.field("source_order_id", pa.string(), nullable=False),
        pa.field("execution_session", pa.date32(), nullable=False),
        pa.field("asset_id", pa.string(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        pa.field("quantity", pa.int64(), nullable=False),
        pa.field("fill_price", MONEY_ARROW_TYPE, nullable=False),
        pa.field("execution_cost", MONEY_ARROW_TYPE, nullable=False),
        pa.field("settlement_id", pa.string(), nullable=True),
        pa.field("settlement_session", pa.date32(), nullable=True),
    ]
)

PORTFOLIO_LEDGER_ARROW_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("session", pa.date32(), nullable=False),
        pa.field("sequence_in_session", pa.int64(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("source_event_id", pa.string(), nullable=False),
        pa.field("source_order_id", pa.string(), nullable=True),
        pa.field("asset_id", pa.string(), nullable=True),
        pa.field("quantity_delta", pa.int64(), nullable=True),
        pa.field("fill_price", MONEY_ARROW_TYPE, nullable=True),
        pa.field("execution_cost", MONEY_ARROW_TYPE, nullable=True),
        pa.field("settlement_id", pa.string(), nullable=True),
        pa.field("settlement_session", pa.date32(), nullable=True),
        pa.field("settled_cash_delta", MONEY_ARROW_TYPE, nullable=False),
        pa.field("pending_cash_delta", MONEY_ARROW_TYPE, nullable=False),
        pa.field("settled_cash_after", MONEY_ARROW_TYPE, nullable=False),
        pa.field("position_quantity_after", pa.int64(), nullable=True),
        pa.field("source_payload_sha256", pa.string(), nullable=False),
        pa.field("state_hash_before", pa.string(), nullable=False),
        pa.field("state_hash_after", pa.string(), nullable=False),
    ]
)

SESSION_SNAPSHOT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("session", pa.date32(), nullable=False),
        pa.field("state_version", pa.int64(), nullable=False),
        pa.field("settled_cash", MONEY_ARROW_TYPE, nullable=False),
        pa.field("open_position_count", pa.int64(), nullable=False),
        pa.field("pending_settlement_count", pa.int64(), nullable=False),
        pa.field("state_hash", pa.string(), nullable=False),
    ]
)

# One row per persisted CanonicalDividendAccountingEvidence record. Flattened
# (not nested JSON) so every Decimal -- including the nested Gate-3 inputs,
# which carry no fixed-scale requirement of their own -- round-trips through
# the same exact decimal256(76, 38) Arrow convention already used for money
# fields elsewhere in this module, rather than through a text-based
# canonical-JSON path that would strip the scale-38 exponent required by
# OD-6.8. Fields the model derives/verifies internally (e.g. the nested
# proofs' own T/X, which must equal the top-level T/X) are not duplicated.
DIVIDEND_EVIDENCE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("evidence_schema_version", pa.string(), nullable=False),
        pa.field("canonical_distribution_event_id", pa.string(), nullable=False),
        pa.field("canonical_security_id", pa.string(), nullable=False),
        pa.field("entitlement_session", pa.date32(), nullable=False),
        pa.field("ex_session", pa.date32(), nullable=False),
        pa.field("distribution_type", pa.string(), nullable=False),
        pa.field("amount_per_share", MONEY_ARROW_TYPE, nullable=False),
        pa.field("amount_basis", pa.string(), nullable=False),
        pa.field("normalization_method_id", pa.string(), nullable=False),
        pa.field("normalization_scale", pa.int64(), nullable=False),
        pa.field("normalization_rounding_mode", pa.string(), nullable=False),
        pa.field("normalization_arithmetic_mode", pa.string(), nullable=False),
        pa.field("normalization_inputs_fingerprint", pa.string(), nullable=False),
        # The raw Gate-3 normalization inputs (d_capitalspecial,
        # unadjusted_close_t, close_capital_t) carry no frozen scale/precision
        # ceiling of their own -- only the derived D_H output is fixed to
        # scale 38. Encoding them as decimal256(76, 38) would silently narrow
        # the accepted evidence domain (e.g. reject a legitimate input with
        # more than 38 fractional digits). Each is instead stored as its
        # exact (sign, digits, exponent) tuple -- an arbitrary-precision,
        # lossless encoding reconstructed via Decimal's exact tuple
        # constructor, never Decimal(str(...)) or a fixed-scale type.
        pa.field("ni_d_capitalspecial_sign", pa.int8(), nullable=False),
        pa.field("ni_d_capitalspecial_digits", pa.string(), nullable=False),
        pa.field("ni_d_capitalspecial_exponent", pa.int64(), nullable=False),
        pa.field("ni_unadjusted_close_t_sign", pa.int8(), nullable=False),
        pa.field("ni_unadjusted_close_t_digits", pa.string(), nullable=False),
        pa.field("ni_unadjusted_close_t_exponent", pa.int64(), nullable=False),
        pa.field("ni_close_capital_t_sign", pa.int8(), nullable=False),
        pa.field("ni_close_capital_t_digits", pa.string(), nullable=False),
        pa.field("ni_close_capital_t_exponent", pa.int64(), nullable=False),
        pa.field("classification_contract_id", pa.string(), nullable=False),
        pa.field("classification_result", pa.string(), nullable=False),
        pa.field(
            "classification_upstream_source_evidence_fingerprint",
            pa.string(),
            nullable=False,
        ),
        pa.field("classification_proof_fingerprint", pa.string(), nullable=False),
        pa.field("calendar_source_id", pa.string(), nullable=False),
        pa.field("calendar_policy_id", pa.string(), nullable=False),
        pa.field("calendar_policy_version", pa.string(), nullable=False),
        pa.field("calendar_resolution_semantics", pa.string(), nullable=False),
        pa.field(
            "calendar_upstream_source_evidence_fingerprint",
            pa.string(),
            nullable=False,
        ),
        pa.field("calendar_resolution_fingerprint", pa.string(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field(
            "canonical_distribution_snapshot_fingerprint",
            pa.string(),
            nullable=False,
        ),
    ]
)

DIVIDEND_OUTCOME_ARROW_SCHEMA = pa.schema(
    [
        pa.field("outcome_schema_version", pa.string(), nullable=False),
        pa.field("canonical_distribution_event_id", pa.string(), nullable=False),
        pa.field("application_id", pa.string(), nullable=False),
        pa.field("payload_sha256", pa.string(), nullable=False),
        pa.field("asset_id", pa.string(), nullable=False),
        pa.field("entitlement_session", pa.date32(), nullable=False),
        pa.field("ex_session", pa.date32(), nullable=False),
        pa.field("q_t", pa.int64(), nullable=False),
        pa.field("attribution_trade_id", pa.string(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
    ]
)

DIVIDEND_LEDGER_ARROW_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("session", pa.date32(), nullable=False),
        pa.field("sequence_in_session", pa.int64(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("application_id", pa.string(), nullable=False),
        pa.field("canonical_distribution_event_id", pa.string(), nullable=False),
        pa.field(
            "canonical_distribution_snapshot_fingerprint",
            pa.string(),
            nullable=False,
        ),
        pa.field("asset_id", pa.string(), nullable=False),
        pa.field("entitlement_session", pa.date32(), nullable=False),
        pa.field("attribution_trade_id", pa.string(), nullable=False),
        pa.field("q_t", pa.int64(), nullable=False),
        pa.field("d_h", MONEY_ARROW_TYPE, nullable=False),
        pa.field("amount_basis", pa.string(), nullable=False),
        pa.field("normalization_method_id", pa.string(), nullable=False),
        pa.field("normalization_scale", pa.int64(), nullable=False),
        pa.field("normalization_rounding_mode", pa.string(), nullable=False),
        pa.field("normalization_arithmetic_mode", pa.string(), nullable=False),
        pa.field("normalization_inputs_fingerprint", pa.string(), nullable=False),
        pa.field("calendar_resolution_fingerprint", pa.string(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("gross_cash_amount", MONEY_ARROW_TYPE, nullable=False),
        pa.field("settled_cash_delta", MONEY_ARROW_TYPE, nullable=False),
        pa.field("settled_cash_after", MONEY_ARROW_TYPE, nullable=False),
        pa.field("source_payload_sha256", pa.string(), nullable=False),
        pa.field("state_hash_before", pa.string(), nullable=False),
        pa.field("state_hash_after", pa.string(), nullable=False),
    ]
)


class PortfolioExecutionPresentation(_ImmutablePortfolioModel):
    """One canonical execution payload presented on one portfolio input session."""

    schema_version: Literal["portfolio_execution_presentation.v0.1"] = (
        PORTFOLIO_EXECUTION_PRESENTATION_SCHEMA_VERSION
    )
    input_session: _SessionDate
    execution_event: PortfolioExecutionEvent


class PortfolioArtifactManifest(_ImmutablePortfolioModel):
    """Canonical completion marker and logical integrity contract.

    Bumped to v0.2 for Slice 6: a v0.1 manifest (written before dividend
    persistence existed) is missing every field below
    ``session_snapshot_count`` and therefore fails closed with
    ``SchemaVersionError`` on read rather than being silently accepted
    without dividend provenance.  Bumped to v0.3 for Task 5C-C: the ledger
    merge-order identity literal changed, so a v0.2 manifest fails closed
    with ``SchemaVersionError`` under this reader.
    """

    schema_version: Literal["portfolio_manifest.v0.3"] = (
        PORTFOLIO_MANIFEST_SCHEMA_VERSION
    )
    portfolio_state_schema_version: Literal["portfolio_state.v0.1"] = (
        PORTFOLIO_STATE_SCHEMA_VERSION
    )
    portfolio_execution_event_schema_version: Literal[
        "portfolio_execution_event.v0.1"
    ] = PORTFOLIO_EXECUTION_EVENT_SCHEMA_VERSION
    execution_presentation_schema_version: Literal[
        "portfolio_execution_presentation.v0.1"
    ] = PORTFOLIO_EXECUTION_PRESENTATION_SCHEMA_VERSION
    portfolio_ledger_entry_schema_version: Literal[
        "portfolio_ledger_entry.v0.1"
    ] = PORTFOLIO_LEDGER_ENTRY_SCHEMA_VERSION
    portfolio_session_snapshot_schema_version: Literal[
        "portfolio_session_snapshot.v0.1"
    ] = PORTFOLIO_SESSION_SNAPSHOT_SCHEMA_VERSION
    dividend_evidence_schema_version: Literal[
        "canonical_dividend_accounting_evidence.v0.1"
    ] = CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION
    dividend_outcome_schema_version: Literal[
        "dividend_application_outcome.v0.1"
    ] = PORTFOLIO_DIVIDEND_OUTCOME_SCHEMA_VERSION
    dividend_ledger_entry_schema_version: Literal[
        "dividend_ledger_entry.v0.1"
    ] = DIVIDEND_LEDGER_ENTRY_SCHEMA_VERSION
    ledger_merge_order_identity: Literal[
        "settlement_prior_sell_buy_round_trip_sell_dividend.v2"
    ] = PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY
    initial_state_content_sha256: _Sha256
    final_state_content_sha256: _Sha256
    execution_presentations_content_sha256: _Sha256
    portfolio_ledger_content_sha256: _Sha256
    session_snapshots_content_sha256: _Sha256
    dividend_evidence_content_sha256: _Sha256
    dividend_outcomes_content_sha256: _Sha256
    dividend_ledger_content_sha256: _Sha256
    execution_presentation_count: _NonNegativeInt
    portfolio_ledger_entry_count: _NonNegativeInt
    session_snapshot_count: _NonNegativeInt
    dividend_evidence_count: _NonNegativeInt
    dividend_outcome_count: _NonNegativeInt
    dividend_ledger_entry_count: _NonNegativeInt


class PortfolioArtifactSet(_ImmutablePortfolioModel):
    """Fully reconstructed and verified Phase 13 persistence result."""

    manifest: PortfolioArtifactManifest
    initial_state: PortfolioState
    final_state: PortfolioState
    execution_presentations: tuple[PortfolioExecutionPresentation, ...] = ()
    ledger_entries: tuple[PortfolioLedgerEntry, ...] = ()
    session_snapshots: tuple[PortfolioSessionSnapshot, ...] = ()
    dividend_evidence: tuple[CanonicalDividendAccountingEvidence, ...] = ()
    dividend_outcomes: tuple[DividendApplicationOutcome, ...] = ()
    dividend_ledger_entries: tuple[DividendLedgerEntry, ...] = ()


def _presentation_sort_key(
    presentation: PortfolioExecutionPresentation,
) -> tuple[object, ...]:
    event = presentation.execution_event
    payload_hash = hash_execution_event(event)
    return (
        presentation.input_session,
        event.execution_id,
        payload_hash,
        canonical_payload_bytes(event),
    )


def _normalize_presentations(
    presentations: Sequence[PortfolioExecutionPresentation],
) -> tuple[PortfolioExecutionPresentation, ...]:
    validated: list[PortfolioExecutionPresentation] = []
    for presentation in presentations:
        if not isinstance(presentation, PortfolioExecutionPresentation):
            raise PortfolioPersistenceError(
                "execution presentations must use PortfolioExecutionPresentation"
            )
        try:
            rebuilt = PortfolioExecutionPresentation.model_validate(
                presentation.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioPersistenceError(
                "execution presentation fails structural validation"
            ) from error
        if rebuilt != presentation:
            raise PortfolioPersistenceError(
                "execution presentation is not canonical"
            )
        validated.append(rebuilt)
    return tuple(sorted(validated, key=_presentation_sort_key))


def _normalize_ledger(
    ledger_entries: Sequence[PortfolioLedgerEntry],
) -> tuple[PortfolioLedgerEntry, ...]:
    ledger: list[PortfolioLedgerEntry] = []
    for entry in ledger_entries:
        if not isinstance(entry, PortfolioLedgerEntry):
            raise PortfolioPersistenceError(
                "ledger entries must use PortfolioLedgerEntry"
            )
        try:
            rebuilt = PortfolioLedgerEntry.model_validate(
                entry.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioPersistenceError(
                "ledger entry fails structural validation"
            ) from error
        if rebuilt != entry:
            raise PortfolioPersistenceError("ledger entry is not canonical")
        ledger.append(rebuilt)

    normalized = tuple(
        sorted(ledger, key=lambda item: (item.session, item.sequence_in_session))
    )
    by_session: dict[date, list[int]] = {}
    for entry in normalized:
        by_session.setdefault(entry.session, []).append(entry.sequence_in_session)
    if any(sequences != list(range(len(sequences))) for sequences in by_session.values()):
        raise PortfolioPersistenceError(
            "ledger sequence_in_session must be consecutive from zero per session"
        )
    return normalized


def _normalize_snapshots(
    snapshots: Sequence[PortfolioSessionSnapshot],
) -> tuple[PortfolioSessionSnapshot, ...]:
    validated: list[PortfolioSessionSnapshot] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, PortfolioSessionSnapshot):
            raise PortfolioPersistenceError(
                "snapshots must use PortfolioSessionSnapshot"
            )
        try:
            rebuilt = PortfolioSessionSnapshot.model_validate(
                snapshot.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioPersistenceError(
                "session snapshot fails structural validation"
            ) from error
        if rebuilt != snapshot:
            raise PortfolioPersistenceError("session snapshot is not canonical")
        validated.append(rebuilt)
    normalized = tuple(sorted(validated, key=lambda item: item.session))
    if len({item.session for item in normalized}) != len(normalized):
        raise PortfolioPersistenceError("session snapshot sessions must be unique")
    return normalized


def hash_execution_presentations(
    presentations: Sequence[PortfolioExecutionPresentation],
) -> str:
    """Hash normalized logical execution-presentation content."""

    return canonical_payload_sha256(_normalize_presentations(presentations))


def hash_portfolio_ledger(
    ledger_entries: Sequence[PortfolioLedgerEntry],
) -> str:
    """Hash normalized logical portfolio-ledger content."""

    return canonical_payload_sha256(_normalize_ledger(ledger_entries))


def hash_session_snapshots(
    snapshots: Sequence[PortfolioSessionSnapshot],
) -> str:
    """Hash normalized logical session-snapshot content."""

    return canonical_payload_sha256(_normalize_snapshots(snapshots))


def _normalize_dividend_evidence(
    records: Sequence[CanonicalDividendAccountingEvidence],
) -> tuple[CanonicalDividendAccountingEvidence, ...]:
    validated: list[CanonicalDividendAccountingEvidence] = []
    for item in records:
        if not isinstance(item, CanonicalDividendAccountingEvidence):
            raise PortfolioPersistenceError(
                "dividend evidence must use CanonicalDividendAccountingEvidence"
            )
        try:
            rebuilt = CanonicalDividendAccountingEvidence.model_validate(
                item.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioPersistenceError(
                "dividend evidence fails structural validation"
            ) from error
        if rebuilt != item:
            raise PortfolioPersistenceError("dividend evidence is not canonical")
        validated.append(rebuilt)

    normalized = tuple(
        sorted(
            validated,
            key=lambda item: (
                item.ex_session,
                item.canonical_security_id,
                item.canonical_distribution_event_id,
            ),
        )
    )
    event_ids = tuple(item.canonical_distribution_event_id for item in normalized)
    if len(set(event_ids)) != len(event_ids):
        raise PortfolioPersistenceError(
            "dividend evidence canonical_distribution_event_id must be unique "
            "across one persisted run"
        )
    return normalized


def _normalize_dividend_outcomes(
    outcomes: Sequence[DividendApplicationOutcome],
) -> tuple[DividendApplicationOutcome, ...]:
    validated: list[DividendApplicationOutcome] = []
    for item in outcomes:
        if not isinstance(item, DividendApplicationOutcome):
            raise PortfolioPersistenceError(
                "dividend outcomes must use DividendApplicationOutcome"
            )
        try:
            rebuilt = DividendApplicationOutcome.model_validate(
                item.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioPersistenceError(
                "dividend outcome fails structural validation"
            ) from error
        if rebuilt != item:
            raise PortfolioPersistenceError("dividend outcome is not canonical")
        validated.append(rebuilt)

    normalized = tuple(
        sorted(
            validated,
            key=lambda item: (
                item.ex_session,
                item.asset_id,
                item.canonical_distribution_event_id,
            ),
        )
    )
    event_ids = tuple(item.canonical_distribution_event_id for item in normalized)
    if len(set(event_ids)) != len(event_ids):
        raise PortfolioPersistenceError(
            "dividend outcome canonical_distribution_event_id must be unique "
            "across one persisted run"
        )
    return normalized


def _normalize_dividend_ledger(
    ledger_entries: Sequence[DividendLedgerEntry],
) -> tuple[DividendLedgerEntry, ...]:
    ledger: list[DividendLedgerEntry] = []
    for entry in ledger_entries:
        if not isinstance(entry, DividendLedgerEntry):
            raise PortfolioPersistenceError(
                "dividend ledger entries must use DividendLedgerEntry"
            )
        try:
            rebuilt = DividendLedgerEntry.model_validate(
                entry.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioPersistenceError(
                "dividend ledger entry fails structural validation"
            ) from error
        if rebuilt != entry:
            raise PortfolioPersistenceError(
                "dividend ledger entry is not canonical"
            )
        ledger.append(rebuilt)

    normalized = tuple(
        sorted(ledger, key=lambda item: (item.session, item.sequence_in_session))
    )
    by_session: dict[date, list[DividendLedgerEntry]] = {}
    for entry in normalized:
        by_session.setdefault(entry.session, []).append(entry)
    for session_rows in by_session.values():
        sequences = [item.sequence_in_session for item in session_rows]
        if sequences != list(range(len(sequences))):
            raise PortfolioPersistenceError(
                "dividend ledger sequence_in_session must be consecutive from "
                "zero per session"
            )
        order_keys = [
            (item.asset_id, item.canonical_distribution_event_id)
            for item in session_rows
        ]
        if order_keys != sorted(order_keys):
            raise PortfolioPersistenceError(
                "dividend ledger rows must be ordered by (asset_id, "
                "canonical_distribution_event_id) per session (OD-20.4)"
            )
    application_ids = tuple(item.application_id for item in normalized)
    if len(set(application_ids)) != len(application_ids):
        raise PortfolioPersistenceError(
            "dividend ledger application_id must be unique across one "
            "persisted run"
        )
    return normalized


def hash_dividend_evidence(
    records: Sequence[CanonicalDividendAccountingEvidence],
) -> str:
    """Hash normalized logical dividend-evidence content."""

    return canonical_payload_sha256(_normalize_dividend_evidence(records))


def hash_dividend_outcomes(
    outcomes: Sequence[DividendApplicationOutcome],
) -> str:
    """Hash normalized logical dividend-outcome content."""

    return canonical_payload_sha256(_normalize_dividend_outcomes(outcomes))


def hash_dividend_ledger(
    ledger_entries: Sequence[DividendLedgerEntry],
) -> str:
    """Hash normalized logical dividend-ledger content."""

    return canonical_payload_sha256(_normalize_dividend_ledger(ledger_entries))


def _decimal_for_arrow(value: Decimal, field_name: str) -> Decimal:
    if not value.is_finite():
        raise PortfolioPersistenceError(f"{field_name} must be finite")
    if value.is_zero():
        return value
    _, raw_digits, raw_exponent = value.as_tuple()
    digits = list(raw_digits)
    exponent = raw_exponent
    while digits and digits[-1] == 0 and exponent < 0:
        digits.pop()
        exponent += 1
    scale = max(-exponent, 0)
    integer_digits = max(value.copy_abs().adjusted() + 1, 0)
    if scale > MONEY_ARROW_TYPE.scale or integer_digits > (
        MONEY_ARROW_TYPE.precision - MONEY_ARROW_TYPE.scale
    ):
        raise PortfolioPersistenceError(
            f"{field_name} cannot be represented exactly as {MONEY_ARROW_TYPE}"
        )
    return value


def _exact_decimal_tuple_columns(
    value: Decimal, field_name: str
) -> dict[str, object]:
    """Encode a Decimal as its exact ``(sign, digits, exponent)`` tuple.

    Unlike ``_decimal_for_arrow`` this imposes no scale/precision ceiling,
    so it is used for fields with no frozen fixed-scale domain of their
    own (e.g. the raw Gate-3 normalization inputs) rather than a bounded
    ``decimal256`` column, which would silently narrow what can be
    persisted. Reconstruction uses Decimal's exact tuple constructor, so
    no rounding or ambient-context arithmetic is ever involved.
    """

    if type(value) is not Decimal or not value.is_finite():
        raise PortfolioPersistenceError(f"{field_name} must be a finite Decimal")
    sign, digits, exponent = value.as_tuple()
    return {
        f"{field_name}_sign": sign,
        f"{field_name}_digits": "".join(str(digit) for digit in digits),
        f"{field_name}_exponent": exponent,
    }


def _decimal_from_exact_tuple_columns(
    row: dict[str, object], field_name: str
) -> Decimal:
    sign = row[f"{field_name}_sign"]
    digits_text = row[f"{field_name}_digits"]
    exponent = row[f"{field_name}_exponent"]
    if sign not in (0, 1):
        raise PortfolioPersistenceError(f"{field_name} sign must be 0 or 1")
    if not digits_text or not digits_text.isdigit():
        raise PortfolioPersistenceError(
            f"{field_name} digits must be a non-empty decimal digit string"
        )
    digits = tuple(int(char) for char in digits_text)
    return Decimal((sign, digits, exponent))


def _presentation_row(
    presentation: PortfolioExecutionPresentation,
) -> dict[str, object]:
    event = presentation.execution_event
    return {
        "presentation_schema_version": presentation.schema_version,
        "input_session": presentation.input_session,
        "execution_schema_version": event.schema_version,
        "execution_id": event.execution_id,
        "source_order_id": event.source_order_id,
        "execution_session": event.session,
        "asset_id": event.asset_id,
        "side": event.side.value,
        "quantity": event.quantity,
        "fill_price": _decimal_for_arrow(event.fill_price, "fill_price"),
        "execution_cost": _decimal_for_arrow(
            event.execution_cost, "execution_cost"
        ),
        "settlement_id": event.settlement_id,
        "settlement_session": event.settlement_session,
    }


def _ledger_row(entry: PortfolioLedgerEntry) -> dict[str, object]:
    return {
        "schema_version": entry.schema_version,
        "session": entry.session,
        "sequence_in_session": entry.sequence_in_session,
        "event_type": entry.event_type.value,
        "source_event_id": entry.source_event_id,
        "source_order_id": entry.source_order_id,
        "asset_id": entry.asset_id,
        "quantity_delta": entry.quantity_delta,
        "fill_price": (
            None
            if entry.fill_price is None
            else _decimal_for_arrow(entry.fill_price, "fill_price")
        ),
        "execution_cost": (
            None
            if entry.execution_cost is None
            else _decimal_for_arrow(entry.execution_cost, "execution_cost")
        ),
        "settlement_id": entry.settlement_id,
        "settlement_session": entry.settlement_session,
        "settled_cash_delta": _decimal_for_arrow(
            entry.settled_cash_delta, "settled_cash_delta"
        ),
        "pending_cash_delta": _decimal_for_arrow(
            entry.pending_cash_delta, "pending_cash_delta"
        ),
        "settled_cash_after": _decimal_for_arrow(
            entry.settled_cash_after, "settled_cash_after"
        ),
        "position_quantity_after": entry.position_quantity_after,
        "source_payload_sha256": entry.source_payload_sha256,
        "state_hash_before": entry.state_hash_before,
        "state_hash_after": entry.state_hash_after,
    }


def _snapshot_row(snapshot: PortfolioSessionSnapshot) -> dict[str, object]:
    return {
        "schema_version": snapshot.schema_version,
        "session": snapshot.session,
        "state_version": snapshot.state_version,
        "settled_cash": _decimal_for_arrow(
            snapshot.settled_cash, "settled_cash"
        ),
        "open_position_count": snapshot.open_position_count,
        "pending_settlement_count": snapshot.pending_settlement_count,
        "state_hash": snapshot.state_hash,
    }


def _dividend_evidence_row(
    item: CanonicalDividendAccountingEvidence,
) -> dict[str, object]:
    inputs = item.normalization_inputs
    classification = item.classification_proof
    calendar = item.calendar_resolution_proof
    return {
        "evidence_schema_version": item.evidence_schema_version,
        "canonical_distribution_event_id": item.canonical_distribution_event_id,
        "canonical_security_id": item.canonical_security_id,
        "entitlement_session": item.entitlement_session,
        "ex_session": item.ex_session,
        "distribution_type": item.distribution_type,
        "amount_per_share": _decimal_for_arrow(
            item.amount_per_share, "amount_per_share"
        ),
        "amount_basis": item.amount_basis,
        "normalization_method_id": item.normalization_method_id,
        "normalization_scale": item.normalization_scale,
        "normalization_rounding_mode": item.normalization_rounding_mode,
        "normalization_arithmetic_mode": item.normalization_arithmetic_mode,
        "normalization_inputs_fingerprint": item.normalization_inputs_fingerprint,
        **_exact_decimal_tuple_columns(
            inputs.d_capitalspecial, "ni_d_capitalspecial"
        ),
        **_exact_decimal_tuple_columns(
            inputs.unadjusted_close_t, "ni_unadjusted_close_t"
        ),
        **_exact_decimal_tuple_columns(
            inputs.close_capital_t, "ni_close_capital_t"
        ),
        "classification_contract_id": classification.classification_contract_id,
        "classification_result": classification.classification_result,
        "classification_upstream_source_evidence_fingerprint": (
            classification.upstream_source_evidence_fingerprint
        ),
        "classification_proof_fingerprint": (
            classification.classification_proof_fingerprint
        ),
        "calendar_source_id": calendar.calendar_source_id,
        "calendar_policy_id": calendar.calendar_policy_id,
        "calendar_policy_version": calendar.calendar_policy_version,
        "calendar_resolution_semantics": calendar.resolution_semantics,
        "calendar_upstream_source_evidence_fingerprint": (
            calendar.upstream_source_evidence_fingerprint
        ),
        "calendar_resolution_fingerprint": item.calendar_resolution_fingerprint,
        "currency": item.currency,
        "canonical_distribution_snapshot_fingerprint": (
            item.canonical_distribution_snapshot_fingerprint
        ),
    }


def _dividend_outcome_row(item: DividendApplicationOutcome) -> dict[str, object]:
    return {
        "outcome_schema_version": PORTFOLIO_DIVIDEND_OUTCOME_SCHEMA_VERSION,
        "canonical_distribution_event_id": item.canonical_distribution_event_id,
        "application_id": item.application_id,
        "payload_sha256": item.payload_sha256,
        "asset_id": item.asset_id,
        "entitlement_session": item.entitlement_session,
        "ex_session": item.ex_session,
        "q_t": item.q_t,
        "attribution_trade_id": item.attribution_trade_id,
        "status": item.status.value,
    }


def _dividend_ledger_row(entry: DividendLedgerEntry) -> dict[str, object]:
    return {
        "schema_version": entry.schema_version,
        "session": entry.session,
        "sequence_in_session": entry.sequence_in_session,
        "event_type": entry.event_type.value,
        "application_id": entry.application_id,
        "canonical_distribution_event_id": entry.canonical_distribution_event_id,
        "canonical_distribution_snapshot_fingerprint": (
            entry.canonical_distribution_snapshot_fingerprint
        ),
        "asset_id": entry.asset_id,
        "entitlement_session": entry.entitlement_session,
        "attribution_trade_id": entry.attribution_trade_id,
        "q_t": entry.q_t,
        "d_h": _decimal_for_arrow(entry.d_h, "d_h"),
        "amount_basis": entry.amount_basis,
        "normalization_method_id": entry.normalization_method_id,
        "normalization_scale": entry.normalization_scale,
        "normalization_rounding_mode": entry.normalization_rounding_mode,
        "normalization_arithmetic_mode": entry.normalization_arithmetic_mode,
        "normalization_inputs_fingerprint": entry.normalization_inputs_fingerprint,
        "calendar_resolution_fingerprint": entry.calendar_resolution_fingerprint,
        "currency": entry.currency,
        "gross_cash_amount": _decimal_for_arrow(
            entry.gross_cash_amount, "gross_cash_amount"
        ),
        "settled_cash_delta": _decimal_for_arrow(
            entry.settled_cash_delta, "settled_cash_delta"
        ),
        "settled_cash_after": _decimal_for_arrow(
            entry.settled_cash_after, "settled_cash_after"
        ),
        "source_payload_sha256": entry.source_payload_sha256,
        "state_hash_before": entry.state_hash_before,
        "state_hash_after": entry.state_hash_after,
    }


def _table_from_rows(
    rows: Sequence[dict[str, object]], schema: pa.Schema
) -> pa.Table:
    try:
        return pa.Table.from_pylist(list(rows), schema=schema)
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise PortfolioPersistenceError(
            "portfolio values cannot be represented by the Arrow schema"
        ) from error


def _write_parquet(table: pa.Table, path: Path) -> None:
    try:
        pq.write_table(table, path, compression="zstd")
    except (OSError, pa.ArrowException) as error:
        raise PortfolioPersistenceError(f"failed to write {path.name}") from error


def _require_arrow_schema(table: pa.Table, expected: pa.Schema, name: str) -> None:
    if not table.schema.equals(expected, check_metadata=False):
        raise SchemaVersionError(f"unsupported {name} Arrow schema")


def _read_parquet(path: Path, schema: pa.Schema, name: str) -> pa.Table:
    if not path.is_file():
        raise PortfolioPersistenceError(f"missing required artifact: {path.name}")
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as error:
        raise PortfolioPersistenceError(f"failed to read {path.name}") from error
    _require_arrow_schema(table, schema, name)
    return table


def _presentation_from_row(row: dict[str, object]) -> PortfolioExecutionPresentation:
    if row.get("presentation_schema_version") != (
        PORTFOLIO_EXECUTION_PRESENTATION_SCHEMA_VERSION
    ):
        raise SchemaVersionError("unsupported execution-presentation schema version")
    if row.get("execution_schema_version") != (
        PORTFOLIO_EXECUTION_EVENT_SCHEMA_VERSION
    ):
        raise SchemaVersionError("unsupported execution-event schema version")
    try:
        event = PortfolioExecutionEvent(
            schema_version=row["execution_schema_version"],
            execution_id=row["execution_id"],
            source_order_id=row["source_order_id"],
            session=row["execution_session"],
            asset_id=row["asset_id"],
            side=row["side"],
            quantity=row["quantity"],
            fill_price=row["fill_price"],
            execution_cost=row["execution_cost"],
            settlement_id=row["settlement_id"],
            settlement_session=row["settlement_session"],
        )
        return PortfolioExecutionPresentation(
            schema_version=row["presentation_schema_version"],
            input_session=row["input_session"],
            execution_event=event,
        )
    except (ValidationError, TypeError, ValueError, KeyError) as error:
        raise PortfolioPersistenceError(
            "execution presentation fails model validation"
        ) from error


def _ledger_from_row(row: dict[str, object]) -> PortfolioLedgerEntry:
    if row.get("schema_version") != PORTFOLIO_LEDGER_ENTRY_SCHEMA_VERSION:
        raise SchemaVersionError("unsupported portfolio-ledger schema version")
    try:
        return PortfolioLedgerEntry(**row)
    except (ValidationError, TypeError, ValueError) as error:
        raise PortfolioPersistenceError(
            "portfolio ledger row fails model validation"
        ) from error


def _snapshot_from_row(row: dict[str, object]) -> PortfolioSessionSnapshot:
    if row.get("schema_version") != PORTFOLIO_SESSION_SNAPSHOT_SCHEMA_VERSION:
        raise SchemaVersionError("unsupported session-snapshot schema version")
    try:
        return PortfolioSessionSnapshot(**row)
    except (ValidationError, TypeError, ValueError) as error:
        raise PortfolioPersistenceError(
            "session snapshot row fails model validation"
        ) from error


def _dividend_evidence_from_row(
    row: dict[str, object],
) -> CanonicalDividendAccountingEvidence:
    if row.get("evidence_schema_version") != (
        CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION
    ):
        raise SchemaVersionError("unsupported dividend-evidence schema version")
    try:
        inputs = Gate3DividendNormalizationInputs(
            entitlement_session=row["entitlement_session"],
            d_capitalspecial=_decimal_from_exact_tuple_columns(
                row, "ni_d_capitalspecial"
            ),
            unadjusted_close_t=_decimal_from_exact_tuple_columns(
                row, "ni_unadjusted_close_t"
            ),
            close_capital_t=_decimal_from_exact_tuple_columns(
                row, "ni_close_capital_t"
            ),
        )
        result = Gate3DividendNormalizationResult(
            d_h=row["amount_per_share"],
            amount_basis=row["amount_basis"],
            normalization_method_id=row["normalization_method_id"],
            normalization_scale=row["normalization_scale"],
            normalization_rounding_mode=row["normalization_rounding_mode"],
            normalization_arithmetic_mode=row["normalization_arithmetic_mode"],
            normalization_inputs_fingerprint=(
                row["normalization_inputs_fingerprint"]
            ),
        )
        classification = OrdinaryCashClassificationProof(
            canonical_distribution_event_id=row["canonical_distribution_event_id"],
            classification_contract_id=row["classification_contract_id"],
            classification_result=row["classification_result"],
            upstream_source_evidence_fingerprint=(
                row["classification_upstream_source_evidence_fingerprint"]
            ),
            classification_proof_fingerprint=row["classification_proof_fingerprint"],
        )
        calendar = DividendCalendarResolutionProof(
            calendar_source_id=row["calendar_source_id"],
            entitlement_session=row["entitlement_session"],
            ex_session=row["ex_session"],
            calendar_policy_id=row["calendar_policy_id"],
            calendar_policy_version=row["calendar_policy_version"],
            resolution_semantics=row["calendar_resolution_semantics"],
            upstream_source_evidence_fingerprint=(
                row["calendar_upstream_source_evidence_fingerprint"]
            ),
            calendar_resolution_fingerprint=row["calendar_resolution_fingerprint"],
        )
        return CanonicalDividendAccountingEvidence(
            evidence_schema_version=row["evidence_schema_version"],
            canonical_distribution_event_id=row["canonical_distribution_event_id"],
            canonical_security_id=row["canonical_security_id"],
            entitlement_session=row["entitlement_session"],
            ex_session=row["ex_session"],
            distribution_type=row["distribution_type"],
            amount_per_share=row["amount_per_share"],
            amount_basis=row["amount_basis"],
            normalization_method_id=row["normalization_method_id"],
            normalization_scale=row["normalization_scale"],
            normalization_rounding_mode=row["normalization_rounding_mode"],
            normalization_arithmetic_mode=row["normalization_arithmetic_mode"],
            normalization_inputs_fingerprint=(
                row["normalization_inputs_fingerprint"]
            ),
            normalization_inputs=inputs,
            normalization_result=result,
            classification_proof=classification,
            calendar_resolution_proof=calendar,
            calendar_resolution_fingerprint=row["calendar_resolution_fingerprint"],
            currency=row["currency"],
            canonical_distribution_snapshot_fingerprint=(
                row["canonical_distribution_snapshot_fingerprint"]
            ),
        )
    except (ValidationError, TypeError, ValueError, KeyError) as error:
        raise PortfolioPersistenceError(
            "dividend evidence row fails model validation"
        ) from error


def _dividend_outcome_from_row(
    row: dict[str, object],
) -> DividendApplicationOutcome:
    if row.get("outcome_schema_version") != (
        PORTFOLIO_DIVIDEND_OUTCOME_SCHEMA_VERSION
    ):
        raise SchemaVersionError("unsupported dividend-outcome schema version")
    try:
        return DividendApplicationOutcome(
            canonical_distribution_event_id=row["canonical_distribution_event_id"],
            application_id=row["application_id"],
            payload_sha256=row["payload_sha256"],
            asset_id=row["asset_id"],
            entitlement_session=row["entitlement_session"],
            ex_session=row["ex_session"],
            q_t=row["q_t"],
            attribution_trade_id=row["attribution_trade_id"],
            status=row["status"],
        )
    except (ValidationError, TypeError, ValueError, KeyError) as error:
        raise PortfolioPersistenceError(
            "dividend outcome row fails model validation"
        ) from error


def _dividend_ledger_from_row(row: dict[str, object]) -> DividendLedgerEntry:
    if row.get("schema_version") != DIVIDEND_LEDGER_ENTRY_SCHEMA_VERSION:
        raise SchemaVersionError("unsupported dividend-ledger schema version")
    if row.get("event_type") != PortfolioLedgerEventType.DIVIDEND_APPLIED.value:
        raise PortfolioPersistenceError(
            "persisted dividend ledger row has an unexpected event_type"
        )
    payload = {key: value for key, value in row.items() if key != "event_type"}
    try:
        return DividendLedgerEntry(**payload)
    except (ValidationError, TypeError, ValueError, KeyError) as error:
        raise PortfolioPersistenceError(
            "dividend ledger row fails model validation"
        ) from error


def _read_presentations(path: Path) -> tuple[PortfolioExecutionPresentation, ...]:
    table = _read_parquet(
        path,
        EXECUTION_PRESENTATION_ARROW_SCHEMA,
        "execution-presentation",
    )
    return _normalize_presentations(
        tuple(_presentation_from_row(row) for row in table.to_pylist())
    )


def _read_ledger(path: Path) -> tuple[PortfolioLedgerEntry, ...]:
    table = _read_parquet(path, PORTFOLIO_LEDGER_ARROW_SCHEMA, "portfolio-ledger")
    return _normalize_ledger(
        tuple(_ledger_from_row(row) for row in table.to_pylist())
    )


def _read_snapshots(path: Path) -> tuple[PortfolioSessionSnapshot, ...]:
    table = _read_parquet(
        path,
        SESSION_SNAPSHOT_ARROW_SCHEMA,
        "session-snapshot",
    )
    return _normalize_snapshots(
        tuple(_snapshot_from_row(row) for row in table.to_pylist())
    )


def _read_dividend_evidence(
    path: Path,
) -> tuple[CanonicalDividendAccountingEvidence, ...]:
    table = _read_parquet(path, DIVIDEND_EVIDENCE_ARROW_SCHEMA, "dividend-evidence")
    return _normalize_dividend_evidence(
        tuple(_dividend_evidence_from_row(row) for row in table.to_pylist())
    )


def _read_dividend_outcomes(path: Path) -> tuple[DividendApplicationOutcome, ...]:
    table = _read_parquet(path, DIVIDEND_OUTCOME_ARROW_SCHEMA, "dividend-outcome")
    return _normalize_dividend_outcomes(
        tuple(_dividend_outcome_from_row(row) for row in table.to_pylist())
    )


def _read_dividend_ledger(path: Path) -> tuple[DividendLedgerEntry, ...]:
    table = _read_parquet(path, DIVIDEND_LEDGER_ARROW_SCHEMA, "dividend-ledger")
    return _normalize_dividend_ledger(
        tuple(_dividend_ledger_from_row(row) for row in table.to_pylist())
    )


def _decode_state_payload(payload: dict[str, object]) -> dict[str, object]:
    decoded = dict(payload)
    if decoded.get("as_of_session") is not None:
        decoded["as_of_session"] = date.fromisoformat(str(decoded["as_of_session"]))
    decoded["settled_cash"] = Decimal(str(decoded["settled_cash"]))

    positions: list[dict[str, object]] = []
    for raw in decoded.get("open_positions", []):
        item = dict(raw)
        item["entry_session"] = date.fromisoformat(str(item["entry_session"]))
        for field in (
            "entry_price",
            "entry_execution_cost",
            "cost_basis",
        ):
            item[field] = Decimal(str(item[field]))
        positions.append(item)
    decoded["open_positions"] = positions

    settlements: list[dict[str, object]] = []
    for raw in decoded.get("pending_settlements", []):
        item = dict(raw)
        item["amount"] = Decimal(str(item["amount"]))
        item["trade_session"] = date.fromisoformat(str(item["trade_session"]))
        item["settlement_session"] = date.fromisoformat(
            str(item["settlement_session"])
        )
        settlements.append(item)
    decoded["pending_settlements"] = settlements
    return decoded


def _read_state(path: Path) -> PortfolioState:
    if not path.is_file():
        raise PortfolioPersistenceError(f"missing required artifact: {path.name}")
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PortfolioPersistenceError(f"failed to read {path.name}") from error
    if not isinstance(payload, dict):
        raise PortfolioPersistenceError(f"{path.name} must contain a JSON object")
    if set(payload) != set(PortfolioState.model_fields):
        raise PortfolioPersistenceError(
            f"{path.name} must contain every canonical PortfolioState field"
        )
    if payload.get("schema_version") != PORTFOLIO_STATE_SCHEMA_VERSION:
        raise SchemaVersionError("unsupported portfolio-state schema version")
    try:
        state = PortfolioState.model_validate(_decode_state_payload(payload))
    except (ValidationError, TypeError, ValueError, KeyError) as error:
        raise PortfolioPersistenceError(
            f"{path.name} fails portfolio-state model validation"
        ) from error
    PortfolioInvariantChecker.validate_state(state)
    if raw_bytes != canonical_payload_bytes(state):
        raise PortfolioPersistenceError(
            f"{path.name} is not canonical Phase 13 JSON"
        )
    return state


def _read_manifest(path: Path) -> PortfolioArtifactManifest:
    if not path.is_file():
        raise PortfolioPersistenceError(
            "artifact set is incomplete because manifest.json is missing"
        )
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PortfolioPersistenceError("failed to read manifest.json") from error
    if not isinstance(payload, dict):
        raise PortfolioPersistenceError("manifest.json must contain a JSON object")
    if set(payload) != set(PortfolioArtifactManifest.model_fields):
        raise PortfolioPersistenceError(
            "manifest.json must contain every canonical manifest field"
        )

    expected_versions = {
        "schema_version": PORTFOLIO_MANIFEST_SCHEMA_VERSION,
        "portfolio_state_schema_version": PORTFOLIO_STATE_SCHEMA_VERSION,
        "portfolio_execution_event_schema_version": (
            PORTFOLIO_EXECUTION_EVENT_SCHEMA_VERSION
        ),
        "execution_presentation_schema_version": (
            PORTFOLIO_EXECUTION_PRESENTATION_SCHEMA_VERSION
        ),
        "portfolio_ledger_entry_schema_version": (
            PORTFOLIO_LEDGER_ENTRY_SCHEMA_VERSION
        ),
        "portfolio_session_snapshot_schema_version": (
            PORTFOLIO_SESSION_SNAPSHOT_SCHEMA_VERSION
        ),
        "dividend_evidence_schema_version": (
            CANONICAL_DIVIDEND_ACCOUNTING_EVIDENCE_SCHEMA_VERSION
        ),
        "dividend_outcome_schema_version": (
            PORTFOLIO_DIVIDEND_OUTCOME_SCHEMA_VERSION
        ),
        "dividend_ledger_entry_schema_version": (
            DIVIDEND_LEDGER_ENTRY_SCHEMA_VERSION
        ),
        "ledger_merge_order_identity": PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY,
    }
    for field, expected in expected_versions.items():
        if payload.get(field) != expected:
            raise SchemaVersionError(f"unsupported manifest version field: {field}")
    try:
        manifest = PortfolioArtifactManifest.model_validate(payload)
    except (ValidationError, TypeError, ValueError) as error:
        raise PortfolioPersistenceError("manifest.json fails validation") from error
    if raw_bytes != canonical_payload_bytes(manifest):
        raise PortfolioPersistenceError(
            "manifest.json is not canonical Phase 13 JSON"
        )
    return manifest


def _flatten_session_inputs(
    session_inputs: Sequence[PortfolioSessionInput],
) -> tuple[
    tuple[PortfolioExecutionPresentation, ...],
    tuple[CanonicalDividendAccountingEvidence, ...],
]:
    presentations: list[PortfolioExecutionPresentation] = []
    dividend_evidence: list[CanonicalDividendAccountingEvidence] = []
    for session_input in session_inputs:
        if not isinstance(session_input, PortfolioSessionInput):
            raise PortfolioPersistenceError(
                "session_inputs must contain PortfolioSessionInput values"
            )
        try:
            rebuilt = PortfolioSessionInput.model_validate(
                session_input.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioPersistenceError(
                "portfolio session input fails validation"
            ) from error
        if rebuilt != session_input:
            raise PortfolioPersistenceError("portfolio session input is not canonical")
        presentations.extend(
            PortfolioExecutionPresentation(
                input_session=rebuilt.session,
                execution_event=event,
            )
            for event in rebuilt.execution_events
        )
        for evidence in rebuilt.dividend_evidence:
            if evidence.ex_session != rebuilt.session:
                raise PortfolioPersistenceError(
                    "dividend evidence ex_session must equal its presenting "
                    "session_input.session (OD-13.6)"
                )
        dividend_evidence.extend(rebuilt.dividend_evidence)
    return (
        _normalize_presentations(presentations),
        _normalize_dividend_evidence(dividend_evidence),
    )


def _validate_cross_artifact_fingerprints(
    initial_state: PortfolioState,
    final_state: PortfolioState,
    ledger: Sequence[PortfolioLedgerEntry],
    dividend_ledger: Sequence[DividendLedgerEntry] = (),
) -> None:
    initial = {
        (item.event_kind, item.event_id): item.payload_sha256
        for item in initial_state.applied_events
    }
    expected = dict(initial)
    for entry in ledger:
        kind = (
            PortfolioEventKind.SETTLEMENT
            if entry.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED
            else PortfolioEventKind.EXECUTION
        )
        key = (kind, entry.source_event_id)
        if key in expected:
            raise PortfolioStateError(
                "an event identity cannot be economically applied twice across "
                "initial state and ledger"
            )
        expected[key] = entry.source_payload_sha256

    for row in dividend_ledger:
        key = (PortfolioEventKind.DIVIDEND, row.application_id)
        if key in expected:
            raise PortfolioStateError(
                "a dividend application identity cannot be economically "
                "applied twice across initial state and ledger"
            )
        expected[key] = row.source_payload_sha256

    final = {
        (item.event_kind, item.event_id): item.payload_sha256
        for item in final_state.applied_events
    }
    if final != expected:
        raise PortfolioStateError(
            "final applied-event fingerprints do not match initial state plus ledger"
        )


def _validate_execution_presentation_history(
    initial_state: PortfolioState,
    presentations: Sequence[PortfolioExecutionPresentation],
    ledger: Sequence[PortfolioLedgerEntry],
) -> None:
    initial_execution_hashes = {
        item.event_id: item.payload_sha256
        for item in initial_state.applied_events
        if item.event_kind is PortfolioEventKind.EXECUTION
    }
    applications: dict[str, PortfolioLedgerEntry] = {}
    for entry in ledger:
        if entry.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED:
            continue
        if entry.source_event_id in applications:
            raise PortfolioStateError(
                "an execution identity cannot have multiple application ledger rows"
            )
        if (
            initial_state.as_of_session is not None
            and entry.session <= initial_state.as_of_session
        ):
            raise PortfolioStateError(
                "ledger application must occur after the initial state session"
            )
        applications[entry.source_event_id] = entry

    groups: dict[
        tuple[date, str, str], list[PortfolioExecutionPresentation]
    ] = {}
    for presentation in _normalize_presentations(presentations):
        event = presentation.execution_event
        payload_hash = hash_execution_event(event)
        if presentation.input_session < event.session:
            raise PortfolioStateError(
                "execution presentation cannot precede execution.session"
            )
        if (
            initial_state.as_of_session is not None
            and presentation.input_session <= initial_state.as_of_session
        ):
            raise PortfolioStateError(
                "execution presentation must follow the initial state session"
            )
        groups.setdefault(
            (presentation.input_session, event.execution_id, payload_hash), []
        ).append(presentation)

    applied: dict[str, tuple[str, date | None]] = {
        event_id: (payload_hash, None)
        for event_id, payload_hash in initial_execution_hashes.items()
    }
    consumed_applications: set[str] = set()
    for (input_session, execution_id, payload_hash), group in sorted(groups.items()):
        event = group[0].execution_event
        known = applied.get(execution_id)
        if (
            execution_id in initial_execution_hashes
            and initial_state.as_of_session is not None
            and event.session > initial_state.as_of_session
        ):
            raise PortfolioStateError(
                "initial-state execution fingerprint cannot prove an execution "
                "from after the initial state session"
            )
        application = applications.get(execution_id)
        applies_now = (
            application is not None
            and application.session == input_session
            and application.source_payload_sha256 == payload_hash
        )

        if applies_now:
            if known is not None:
                raise PortfolioStateError(
                    "an already-applied execution cannot be economically applied again"
                )
            if event.session != input_session:
                raise PortfolioStateError(
                    "new execution application requires input_session equal to "
                    "execution.session"
                )
            applied[execution_id] = (payload_hash, input_session)
            consumed_applications.add(execution_id)
            continue

        if known is None:
            raise PortfolioStateError(
                "ledger-less execution presentation lacks prior application proof"
            )
        known_hash, known_session = known
        if known_hash != payload_hash:
            raise PortfolioStateError(
                "replayed execution conflicts with its prior applied fingerprint"
            )
        if known_session is not None and known_session >= input_session:
            raise PortfolioStateError(
                "replayed execution must follow its earlier ledger application"
            )

    if consumed_applications != set(applications):
        raise PortfolioStateError(
            "BUY/SELL ledger application lacks a same-session execution presentation"
        )


def _validate_snapshot_history(
    initial_state: PortfolioState,
    final_state: PortfolioState,
    presentations: Sequence[PortfolioExecutionPresentation],
    ledger: Sequence[PortfolioLedgerEntry],
    snapshots: Sequence[PortfolioSessionSnapshot],
    dividend_evidence: Sequence[CanonicalDividendAccountingEvidence] = (),
    dividend_ledger: Sequence[DividendLedgerEntry] = (),
    dividend_outcomes: Sequence[DividendApplicationOutcome] = (),
) -> None:
    normalized = _normalize_snapshots(snapshots)
    version_delta = final_state.state_version - initial_state.state_version
    if version_delta < 0 or len(normalized) != version_delta:
        raise PortfolioStateError(
            "snapshot count must equal the persisted state-version advance"
        )
    if not normalized:
        if final_state != initial_state:
            raise PortfolioStateError(
                "state changed without persisted session snapshots"
            )
        if (
            ledger
            or presentations
            or dividend_evidence
            or dividend_ledger
            or dividend_outcomes
        ):
            raise PortfolioStateError(
                "ledger/presentations/dividend artifacts cannot exist without "
                "session snapshots"
            )
        return

    expected_versions = tuple(
        range(initial_state.state_version + 1, final_state.state_version + 1)
    )
    if tuple(item.state_version for item in normalized) != expected_versions:
        raise PortfolioStateError("snapshot state versions must be consecutive")
    if (
        initial_state.as_of_session is not None
        and normalized[0].session <= initial_state.as_of_session
    ):
        raise PortfolioStateError(
            "snapshot history must follow the initial state session"
        )
    final_snapshot = normalized[-1]
    if (
        final_snapshot.session != final_state.as_of_session
        or final_snapshot.state_hash != hash_portfolio_state(final_state)
        or final_snapshot.settled_cash != final_state.settled_cash
        or final_snapshot.open_position_count != len(final_state.open_positions)
        or final_snapshot.pending_settlement_count
        != len(final_state.pending_settlements)
    ):
        raise PortfolioStateError("final snapshot does not match final state")

    snapshot_by_session = {item.session: item for item in normalized}
    presentation_sessions = {item.input_session for item in presentations}
    ledger_sessions = {item.session for item in ledger}
    dividend_evidence_sessions = {item.ex_session for item in dividend_evidence}
    dividend_ledger_sessions = {item.session for item in dividend_ledger}
    dividend_outcome_sessions = {item.ex_session for item in dividend_outcomes}
    if not (
        presentation_sessions
        | ledger_sessions
        | dividend_evidence_sessions
        | dividend_ledger_sessions
        | dividend_outcome_sessions
    ) <= set(snapshot_by_session):
        raise PortfolioStateError(
            "every presentation, ledger, and dividend-artifact session "
            "requires a snapshot"
        )

    previous_hash = hash_portfolio_state(initial_state)
    ledger_by_session: dict[date, list[PortfolioLedgerEntry]] = {}
    for entry in ledger:
        ledger_by_session.setdefault(entry.session, []).append(entry)
    dividend_ledger_by_session: dict[date, list[DividendLedgerEntry]] = {}
    for row in dividend_ledger:
        dividend_ledger_by_session.setdefault(row.session, []).append(row)
    for current_snapshot in normalized:
        session_ledger = ledger_by_session.get(current_snapshot.session, [])
        session_dividend_ledger = dividend_ledger_by_session.get(
            current_snapshot.session, []
        )
        if session_ledger or session_dividend_ledger:
            if any(
                entry.state_hash_before != previous_hash
                or entry.state_hash_after != current_snapshot.state_hash
                for entry in (*session_ledger, *session_dividend_ledger)
            ):
                raise PortfolioStateError(
                    "ledger state hashes do not match snapshot history"
                )
        previous_hash = current_snapshot.state_hash


def _validate_artifact_semantics(
    initial_state: PortfolioState,
    final_state: PortfolioState,
    presentations: Sequence[PortfolioExecutionPresentation],
    ledger: Sequence[PortfolioLedgerEntry],
    snapshots: Sequence[PortfolioSessionSnapshot],
    dividend_evidence: Sequence[CanonicalDividendAccountingEvidence] = (),
    dividend_outcomes: Sequence[DividendApplicationOutcome] = (),
    dividend_ledger: Sequence[DividendLedgerEntry] = (),
) -> None:
    PortfolioInvariantChecker.validate_state(initial_state)
    PortfolioInvariantChecker.validate_state(final_state)
    PortfolioInvariantChecker.validate_execution_ledger_provenance(
        tuple(item.execution_event for item in presentations),
        ledger,
    )
    _validate_execution_presentation_history(
        initial_state,
        presentations,
        ledger,
    )
    _validate_cross_artifact_fingerprints(
        initial_state, final_state, ledger, dividend_ledger
    )
    _validate_snapshot_history(
        initial_state,
        final_state,
        presentations,
        ledger,
        snapshots,
        dividend_evidence,
        dividend_ledger,
        dividend_outcomes,
    )
    PortfolioInvariantChecker.validate_persisted_accounting_history(
        initial_state,
        final_state,
        ledger,
        snapshots,
        dividend_ledger_entries=dividend_ledger,
        dividend_evidence=dividend_evidence,
        dividend_outcomes=dividend_outcomes,
    )


def _validate_backtest_result_aggregates(
    result: PortfolioBacktestResult,
) -> None:
    """Require aggregate fields to exactly describe contained session results."""

    PortfolioInvariantChecker.validate_state(result.initial_state)
    PortfolioInvariantChecker.validate_state(result.final_state)

    initial_hash = hash_portfolio_state(result.initial_state)
    final_hash = hash_portfolio_state(result.final_state)
    if result.initial_state_hash != initial_hash:
        raise StateHashMismatchError("initial_state_hash is incorrect")
    if result.final_state_hash != final_hash:
        raise StateHashMismatchError("final_state_hash is incorrect")

    session_results = result.session_results
    if not session_results:
        if (
            result.final_state != result.initial_state
            or result.ledger_entries
            or result.snapshots
            or result.dividend_ledger_entries
            or result.dividend_outcomes
            or result.final_state_hash != result.initial_state_hash
        ):
            raise PortfolioStateError(
                "a no-session backtest result must preserve initial state and "
                "have empty ledger, snapshots, and dividend artifacts"
            )
        return

    sessions = tuple(item.session for item in session_results)
    if any(
        current <= previous
        for previous, current in zip(sessions, sessions[1:], strict=False)
    ):
        raise PortfolioStateError(
            "backtest session_results must be strictly chronological"
        )

    previous_state = result.initial_state
    previous_hash = initial_hash
    for session_result in session_results:
        current_hash = hash_portfolio_state(session_result.resulting_state)
        if session_result.session != session_result.resulting_state.as_of_session:
            raise PortfolioStateError(
                "session result session must match its resulting state"
            )
        if session_result.state_hash_before != previous_hash:
            raise PortfolioStateError(
                "backtest session-result state-hash chain is inconsistent"
            )
        if session_result.state_hash_after != current_hash:
            raise StateHashMismatchError(
                "session result state_hash_after is incorrect"
            )
        PortfolioInvariantChecker.validate_transition(
            previous_state,
            session_result.resulting_state,
            session_result.ledger_entries,
            dividend_ledger_entries=session_result.dividend_ledger_entries,
        )
        previous_state = session_result.resulting_state
        previous_hash = current_hash

    flattened_ledger = tuple(
        entry
        for session_result in session_results
        for entry in session_result.ledger_entries
    )
    if flattened_ledger != result.ledger_entries:
        raise PortfolioStateError(
            "backtest aggregate ledger does not exactly match session results"
        )

    flattened_dividend_ledger = tuple(
        entry
        for session_result in session_results
        for entry in session_result.dividend_ledger_entries
    )
    if flattened_dividend_ledger != result.dividend_ledger_entries:
        raise PortfolioStateError(
            "backtest aggregate dividend ledger does not exactly match "
            "session results"
        )

    flattened_dividend_outcomes = tuple(
        outcome
        for session_result in session_results
        for outcome in session_result.dividend_outcomes
    )
    if flattened_dividend_outcomes != result.dividend_outcomes:
        raise PortfolioStateError(
            "backtest aggregate dividend outcomes do not exactly match "
            "session results"
        )

    expected_snapshots = tuple(
        PortfolioSessionSnapshot(
            session=session_result.session,
            state_version=session_result.resulting_state.state_version,
            settled_cash=session_result.resulting_state.settled_cash,
            open_position_count=len(
                session_result.resulting_state.open_positions
            ),
            pending_settlement_count=len(
                session_result.resulting_state.pending_settlements
            ),
            state_hash=session_result.state_hash_after,
        )
        for session_result in session_results
    )
    if expected_snapshots != result.snapshots:
        raise PortfolioStateError(
            "backtest aggregate snapshots do not exactly match session results"
        )

    last_result = session_results[-1]
    if result.final_state != last_result.resulting_state:
        raise PortfolioStateError(
            "backtest final state does not match the last session result"
        )
    if result.final_state_hash != last_result.state_hash_after:
        raise PortfolioStateError(
            "backtest final hash does not match the last session result"
        )


class PortfolioArtifactWriter:
    """Write one complete deterministic local Phase 13 artifact directory."""

    @staticmethod
    def write(
        artifact_directory: str | Path,
        backtest_result: PortfolioBacktestResult,
        session_inputs: Sequence[PortfolioSessionInput],
    ) -> PortfolioArtifactManifest:
        target = Path(artifact_directory)
        if target.exists():
            raise PortfolioPersistenceError(
                f"artifact directory already exists: {target}"
            )
        if not isinstance(backtest_result, PortfolioBacktestResult):
            raise PortfolioPersistenceError(
                "backtest_result must be a PortfolioBacktestResult"
            )
        try:
            result = PortfolioBacktestResult.model_validate(
                backtest_result.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioPersistenceError(
                "backtest result fails structural validation"
            ) from error

        _validate_backtest_result_aggregates(result)

        initial_hash = hash_portfolio_state(result.initial_state)
        final_hash = hash_portfolio_state(result.final_state)
        if initial_hash != result.initial_state_hash:
            raise StateHashMismatchError("initial_state_hash is incorrect")
        if final_hash != result.final_state_hash:
            raise StateHashMismatchError("final_state_hash is incorrect")

        presentations, dividend_evidence = _flatten_session_inputs(session_inputs)
        ledger = _normalize_ledger(result.ledger_entries)
        snapshots = _normalize_snapshots(result.snapshots)
        dividend_outcomes = _normalize_dividend_outcomes(result.dividend_outcomes)
        dividend_ledger = _normalize_dividend_ledger(result.dividend_ledger_entries)
        input_sessions = tuple(item.session for item in session_inputs)
        if len(set(input_sessions)) != len(input_sessions):
            raise PortfolioPersistenceError(
                "PortfolioSessionInput sessions must be unique"
            )
        if set(input_sessions) != {item.session for item in snapshots}:
            raise PortfolioPersistenceError(
                "PortfolioSessionInput sessions must exactly match snapshot sessions"
            )
        _validate_artifact_semantics(
            result.initial_state,
            result.final_state,
            presentations,
            ledger,
            snapshots,
            dividend_evidence,
            dividend_outcomes,
            dividend_ledger,
        )

        manifest = PortfolioArtifactManifest(
            initial_state_content_sha256=initial_hash,
            final_state_content_sha256=final_hash,
            execution_presentations_content_sha256=(
                hash_execution_presentations(presentations)
            ),
            portfolio_ledger_content_sha256=hash_portfolio_ledger(ledger),
            session_snapshots_content_sha256=hash_session_snapshots(snapshots),
            dividend_evidence_content_sha256=hash_dividend_evidence(
                dividend_evidence
            ),
            dividend_outcomes_content_sha256=hash_dividend_outcomes(
                dividend_outcomes
            ),
            dividend_ledger_content_sha256=hash_dividend_ledger(dividend_ledger),
            execution_presentation_count=len(presentations),
            portfolio_ledger_entry_count=len(ledger),
            session_snapshot_count=len(snapshots),
            dividend_evidence_count=len(dividend_evidence),
            dividend_outcome_count=len(dividend_outcomes),
            dividend_ledger_entry_count=len(dividend_ledger),
        )

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        )
        try:
            (temporary / INITIAL_STATE_FILENAME).write_bytes(
                canonical_payload_bytes(result.initial_state)
            )
            (temporary / FINAL_STATE_FILENAME).write_bytes(
                canonical_payload_bytes(result.final_state)
            )
            _write_parquet(
                _table_from_rows(
                    tuple(_presentation_row(item) for item in presentations),
                    EXECUTION_PRESENTATION_ARROW_SCHEMA,
                ),
                temporary / EXECUTION_EVENTS_FILENAME,
            )
            _write_parquet(
                _table_from_rows(
                    tuple(_ledger_row(item) for item in ledger),
                    PORTFOLIO_LEDGER_ARROW_SCHEMA,
                ),
                temporary / PORTFOLIO_LEDGER_FILENAME,
            )
            _write_parquet(
                _table_from_rows(
                    tuple(_snapshot_row(item) for item in snapshots),
                    SESSION_SNAPSHOT_ARROW_SCHEMA,
                ),
                temporary / SESSION_SNAPSHOTS_FILENAME,
            )
            _write_parquet(
                _table_from_rows(
                    tuple(_dividend_evidence_row(item) for item in dividend_evidence),
                    DIVIDEND_EVIDENCE_ARROW_SCHEMA,
                ),
                temporary / DIVIDEND_EVIDENCE_FILENAME,
            )
            _write_parquet(
                _table_from_rows(
                    tuple(_dividend_outcome_row(item) for item in dividend_outcomes),
                    DIVIDEND_OUTCOME_ARROW_SCHEMA,
                ),
                temporary / DIVIDEND_OUTCOMES_FILENAME,
            )
            _write_parquet(
                _table_from_rows(
                    tuple(_dividend_ledger_row(item) for item in dividend_ledger),
                    DIVIDEND_LEDGER_ARROW_SCHEMA,
                ),
                temporary / DIVIDEND_LEDGER_FILENAME,
            )

            written_initial = _read_state(temporary / INITIAL_STATE_FILENAME)
            written_final = _read_state(temporary / FINAL_STATE_FILENAME)
            written_presentations = _read_presentations(
                temporary / EXECUTION_EVENTS_FILENAME
            )
            written_ledger = _read_ledger(temporary / PORTFOLIO_LEDGER_FILENAME)
            written_snapshots = _read_snapshots(
                temporary / SESSION_SNAPSHOTS_FILENAME
            )
            written_dividend_evidence = _read_dividend_evidence(
                temporary / DIVIDEND_EVIDENCE_FILENAME
            )
            written_dividend_outcomes = _read_dividend_outcomes(
                temporary / DIVIDEND_OUTCOMES_FILENAME
            )
            written_dividend_ledger = _read_dividend_ledger(
                temporary / DIVIDEND_LEDGER_FILENAME
            )
            if (
                written_initial != result.initial_state
                or written_final != result.final_state
                or written_presentations != presentations
                or written_ledger != ledger
                or written_snapshots != snapshots
                or written_dividend_evidence != dividend_evidence
                or written_dividend_outcomes != dividend_outcomes
                or written_dividend_ledger != dividend_ledger
            ):
                raise PortfolioPersistenceError(
                    "written artifact content failed exact read-back verification"
                )
            if (
                hash_execution_presentations(written_presentations)
                != manifest.execution_presentations_content_sha256
                or hash_portfolio_ledger(written_ledger)
                != manifest.portfolio_ledger_content_sha256
                or hash_session_snapshots(written_snapshots)
                != manifest.session_snapshots_content_sha256
                or hash_dividend_evidence(written_dividend_evidence)
                != manifest.dividend_evidence_content_sha256
                or hash_dividend_outcomes(written_dividend_outcomes)
                != manifest.dividend_outcomes_content_sha256
                or hash_dividend_ledger(written_dividend_ledger)
                != manifest.dividend_ledger_content_sha256
            ):
                raise ArtifactHashMismatchError(
                    "written artifact logical content hash mismatch"
                )
            _validate_artifact_semantics(
                written_initial,
                written_final,
                written_presentations,
                written_ledger,
                written_snapshots,
                written_dividend_evidence,
                written_dividend_outcomes,
                written_dividend_ledger,
            )

            (temporary / MANIFEST_FILENAME).write_bytes(
                canonical_payload_bytes(manifest)
            )
            temporary.rename(target)
        except FileExistsError as error:
            raise PortfolioPersistenceError(
                f"artifact directory already exists: {target}"
            ) from error
        except (
            PortfolioPersistenceError,
            PortfolioStateError,
            StateHashMismatchError,
            ArtifactHashMismatchError,
            SchemaVersionError,
        ):
            raise
        except Exception as error:
            raise PortfolioPersistenceError("portfolio artifact write failed") from error
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return manifest


class PortfolioArtifactReader:
    """Read and fully validate one completed local Phase 13 artifact set."""

    @staticmethod
    def read(artifact_directory: str | Path) -> PortfolioArtifactSet:
        directory = Path(artifact_directory)
        manifest = _read_manifest(directory / MANIFEST_FILENAME)
        initial_state = _read_state(directory / INITIAL_STATE_FILENAME)
        final_state = _read_state(directory / FINAL_STATE_FILENAME)
        presentations = _read_presentations(directory / EXECUTION_EVENTS_FILENAME)
        ledger = _read_ledger(directory / PORTFOLIO_LEDGER_FILENAME)
        snapshots = _read_snapshots(directory / SESSION_SNAPSHOTS_FILENAME)
        dividend_evidence = _read_dividend_evidence(
            directory / DIVIDEND_EVIDENCE_FILENAME
        )
        dividend_outcomes = _read_dividend_outcomes(
            directory / DIVIDEND_OUTCOMES_FILENAME
        )
        dividend_ledger = _read_dividend_ledger(directory / DIVIDEND_LEDGER_FILENAME)

        if hash_portfolio_state(initial_state) != manifest.initial_state_content_sha256:
            raise StateHashMismatchError("initial state content hash mismatch")
        if hash_portfolio_state(final_state) != manifest.final_state_content_sha256:
            raise StateHashMismatchError("final state content hash mismatch")
        if (
            hash_execution_presentations(presentations)
            != manifest.execution_presentations_content_sha256
        ):
            raise ArtifactHashMismatchError(
                "execution-presentation logical content hash mismatch"
            )
        if hash_portfolio_ledger(ledger) != manifest.portfolio_ledger_content_sha256:
            raise ArtifactHashMismatchError(
                "portfolio-ledger logical content hash mismatch"
            )
        if (
            hash_session_snapshots(snapshots)
            != manifest.session_snapshots_content_sha256
        ):
            raise ArtifactHashMismatchError(
                "session-snapshot logical content hash mismatch"
            )
        if (
            hash_dividend_evidence(dividend_evidence)
            != manifest.dividend_evidence_content_sha256
        ):
            raise ArtifactHashMismatchError(
                "dividend-evidence logical content hash mismatch"
            )
        if (
            hash_dividend_outcomes(dividend_outcomes)
            != manifest.dividend_outcomes_content_sha256
        ):
            raise ArtifactHashMismatchError(
                "dividend-outcome logical content hash mismatch"
            )
        if (
            hash_dividend_ledger(dividend_ledger)
            != manifest.dividend_ledger_content_sha256
        ):
            raise ArtifactHashMismatchError(
                "dividend-ledger logical content hash mismatch"
            )
        if (
            len(presentations) != manifest.execution_presentation_count
            or len(ledger) != manifest.portfolio_ledger_entry_count
            or len(snapshots) != manifest.session_snapshot_count
            or len(dividend_evidence) != manifest.dividend_evidence_count
            or len(dividend_outcomes) != manifest.dividend_outcome_count
            or len(dividend_ledger) != manifest.dividend_ledger_entry_count
        ):
            raise ArtifactHashMismatchError("manifest artifact row count mismatch")

        _validate_artifact_semantics(
            initial_state,
            final_state,
            presentations,
            ledger,
            snapshots,
            dividend_evidence,
            dividend_outcomes,
            dividend_ledger,
        )
        return PortfolioArtifactSet(
            manifest=manifest,
            initial_state=initial_state,
            final_state=final_state,
            execution_presentations=presentations,
            ledger_entries=ledger,
            session_snapshots=snapshots,
            dividend_evidence=dividend_evidence,
            dividend_outcomes=dividend_outcomes,
            dividend_ledger_entries=dividend_ledger,
        )


__all__ = [
    "DIVIDEND_EVIDENCE_ARROW_SCHEMA",
    "DIVIDEND_EVIDENCE_FILENAME",
    "DIVIDEND_LEDGER_ARROW_SCHEMA",
    "DIVIDEND_LEDGER_FILENAME",
    "DIVIDEND_OUTCOME_ARROW_SCHEMA",
    "DIVIDEND_OUTCOMES_FILENAME",
    "EXECUTION_EVENTS_FILENAME",
    "EXECUTION_PRESENTATION_ARROW_SCHEMA",
    "FINAL_STATE_FILENAME",
    "INITIAL_STATE_FILENAME",
    "MANIFEST_FILENAME",
    "MONEY_ARROW_TYPE",
    "PORTFOLIO_DIVIDEND_OUTCOME_SCHEMA_VERSION",
    "PORTFOLIO_EXECUTION_PRESENTATION_SCHEMA_VERSION",
    "PORTFOLIO_LEDGER_ARROW_SCHEMA",
    "PORTFOLIO_LEDGER_FILENAME",
    "PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY",
    "PORTFOLIO_MANIFEST_SCHEMA_VERSION",
    "PortfolioArtifactManifest",
    "PortfolioArtifactReader",
    "PortfolioArtifactSet",
    "PortfolioArtifactWriter",
    "PortfolioExecutionPresentation",
    "SESSION_SNAPSHOT_ARROW_SCHEMA",
    "SESSION_SNAPSHOTS_FILENAME",
    "hash_dividend_evidence",
    "hash_dividend_ledger",
    "hash_dividend_outcomes",
    "hash_execution_presentations",
    "hash_portfolio_ledger",
    "hash_session_snapshots",
]
