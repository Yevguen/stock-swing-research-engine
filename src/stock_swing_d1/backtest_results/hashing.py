"""Domain-separated semantic SHA-256 helpers for Phase 15D."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from stock_swing_d1.backtest_results.canonical import semantic_json_bytes

if TYPE_CHECKING:
    from stock_swing_d1.backtester.models import HistoricalBacktestRunResult


VALUATION_POLICY_HASH_DOMAIN = "historical_backtest_valuation_policy.v0.1"
VALUATION_SNAPSHOT_HASH_DOMAIN = "historical_backtest_valuation_snapshot.v0.1"
RUN_CONFIGURATION_HASH_DOMAIN = "historical_backtest_run_configuration.v0.1"
CONTENT_HASH_DOMAIN = "historical_backtest_content.v0.1"
RESULT_HASH_DOMAIN = "historical_backtest_result.v0.1"
# Bumped v0.1 -> v0.2 (OD-22.2 "Phase 15D content/result fingerprints"):
# compute_source_run_fingerprint's payload gained run_result.dividend_run_evidence.
# A digest computed under v0.1 and one computed under v0.2 must never share
# one domain identity, since they bind different facts under the same name.
# Bumped v0.2 -> v0.3 (frozen Task 5C-B baseline): the nested session results
# bound by this digest gained `entry_session_protective_decisions` (Phase 15B
# entry-session decisions, one per Phase 9 EXECUTED entry), so a v0.2 digest
# and a v0.3 digest bind different facts.  Task 5C-C requires no further
# bump: it changes which elements of that field are admissible, not the
# payload this function binds.  There is no v0.4.
SOURCE_RUN_HASH_DOMAIN = "historical_backtest_source_run.v0.3"
SOURCE_PAYLOAD_HASH_DOMAIN = "historical_backtest_source_payload.v0.1"


def _require_canonical_text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def semantic_sha256(value: object) -> str:
    """Return lowercase SHA-256 over canonical semantic JSON bytes."""

    return hashlib.sha256(semantic_json_bytes(value)).hexdigest()


def semantic_domain_sha256(domain: str, payload: object) -> str:
    """Hash semantic content within an explicit, versioned domain."""

    return semantic_sha256(
        {
            "domain": _require_canonical_text(domain, field_name="domain"),
            "payload": payload,
        }
    )


def compute_content_fingerprint(
    *, artifact_name: str, rows_or_value: object
) -> str:
    """Hash one caller-ordered logical result artifact."""

    return semantic_domain_sha256(
        CONTENT_HASH_DOMAIN,
        {
            "artifact_name": _require_canonical_text(
                artifact_name, field_name="artifact_name"
            ),
            "content": rows_or_value,
        },
    )


def compute_result_fingerprint(
    *,
    schema_version: str,
    decision_interval: object,
    run_configuration_fingerprint: str,
    source_run_fingerprint: str,
    initial_state_fingerprint: str,
    final_state_fingerprint: str,
    content_fingerprints: object,
) -> str:
    """Bind the complete set of semantic result-fingerprint components."""

    return semantic_domain_sha256(
        RESULT_HASH_DOMAIN,
        {
            "schema_version": _require_canonical_text(
                schema_version, field_name="schema_version"
            ),
            "decision_interval": decision_interval,
            "run_configuration_fingerprint": _require_sha256(
                run_configuration_fingerprint,
                field_name="run_configuration_fingerprint",
            ),
            "source_run_fingerprint": _require_sha256(
                source_run_fingerprint, field_name="source_run_fingerprint"
            ),
            "initial_state_fingerprint": _require_sha256(
                initial_state_fingerprint,
                field_name="initial_state_fingerprint",
            ),
            "final_state_fingerprint": _require_sha256(
                final_state_fingerprint, field_name="final_state_fingerprint"
            ),
            "content_fingerprints": content_fingerprints,
        },
    )


def compute_source_run_fingerprint(
    run_result: HistoricalBacktestRunResult,
) -> str:
    """Hash one genuine immutable source run without validating or deriving it."""

    from stock_swing_d1.backtester.models import HistoricalBacktestRunResult

    if type(run_result) is not HistoricalBacktestRunResult:
        raise TypeError("run_result must be a HistoricalBacktestRunResult")
    return semantic_domain_sha256(
        SOURCE_RUN_HASH_DOMAIN,
        {
            "schema_version": run_result.schema_version,
            "decision_interval": run_result.decision_interval,
            "initial_state": run_result.initial_state,
            "final_state": run_result.final_state,
            "session_results": run_result.session_results,
            "initial_state_fingerprint": run_result.initial_state_fingerprint,
            "final_state_fingerprint": run_result.final_state_fingerprint,
            # Binds the run-level retained dividend evidence bundle itself
            # (distinct from the per-session dividend ledger/outcome facts
            # already nested inside session_results): None for a
            # non-dividend-aware run, otherwise the complete supplied
            # CanonicalDividendAccountingEvidence/coverage/contiguity-proof
            # bundle, so changing it changes this fingerprint.
            "dividend_run_evidence": run_result.dividend_run_evidence,
        },
    )


def compute_source_payload_fingerprint(
    *, source_type: str, payload: object
) -> str:
    """Hash one authoritative upstream payload within its source-type boundary."""

    return semantic_domain_sha256(
        SOURCE_PAYLOAD_HASH_DOMAIN,
        {
            "source_type": _require_canonical_text(
                source_type, field_name="source_type"
            ),
            "payload": payload,
        },
    )


# Slice 10 (OD-16.7): one closed trade's dividend-attribution projection
# fingerprint. A new domain, not a reuse of SOURCE_PAYLOAD_HASH_DOMAIN,
# because its payload shape (trade/completeness/policy/snapshot plus an
# ordered grouped-row tuple) is a Phase-15D-internal projection fact, not
# a 1:1 hash of one upstream payload.
TRADE_DIVIDEND_ATTRIBUTION_HASH_DOMAIN = (
    "historical_closed_trade_dividend_attribution.v0.1"
)


def compute_dividend_attribution_fingerprint(
    *,
    trade_id: str,
    completeness: str,
    dividend_policy_semantic_identity: str,
    canonical_distribution_snapshot_fingerprint: str | None,
    grouped_rows: object,
) -> str:
    """Hash one closed trade's OD-16.7 dividend-attribution projection.

    Bound exactly to OD-16.7's frozen field list: trade ID, completeness
    discriminator, dividend-policy version, distribution snapshot, and the
    ordered grouped projected dividend application IDs/payload hashes
    (`grouped_rows`) -- never upstream Phase-13 objects (OD-16.2/16.3).
    """

    return semantic_domain_sha256(
        TRADE_DIVIDEND_ATTRIBUTION_HASH_DOMAIN,
        {
            "trade_id": _require_canonical_text(trade_id, field_name="trade_id"),
            "completeness": _require_canonical_text(
                completeness, field_name="completeness"
            ),
            "dividend_policy_semantic_identity": _require_canonical_text(
                dividend_policy_semantic_identity,
                field_name="dividend_policy_semantic_identity",
            ),
            "canonical_distribution_snapshot_fingerprint": (
                canonical_distribution_snapshot_fingerprint
            ),
            "grouped_rows": grouped_rows,
        },
    )


__all__ = [
    "compute_content_fingerprint",
    "compute_dividend_attribution_fingerprint",
    "compute_result_fingerprint",
    "compute_source_payload_fingerprint",
    "compute_source_run_fingerprint",
    "semantic_domain_sha256",
    "semantic_sha256",
]
