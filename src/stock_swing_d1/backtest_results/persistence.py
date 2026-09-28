"""Phase 15D.4: persistence and deterministic read-back of one completed,
already-validated `HistoricalBacktestAuditResult`.

`HistoricalBacktestResultPersistence.write`/`.read` are the only public
surface. This module never reprices an execution, reruns ranking/allocation/
signals, reconstructs a settlement calendar, or mutates portfolio state — its
sole economic input on write is an already-complete
`HistoricalBacktestAuditResult`, and its sole output on read is the same
object, reconstructed byte-for-byte through the frozen model's own public
constructors with no tolerance, coercion, or repair.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
from pydantic import ValidationError

from stock_swing_d1.backtest_results import persistence_schema as schema
from stock_swing_d1.backtest_results.errors import HistoricalBacktestPersistenceError
from stock_swing_d1.backtest_results.hashing import (
    compute_content_fingerprint,
    compute_result_fingerprint,
)
from stock_swing_d1.backtest_results.models import (
    HISTORICAL_BACKTEST_AUDIT_RESULT_SCHEMA_VERSION,
    HistoricalBacktestAuditResult,
    HistoricalBacktestContentFingerprints,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state


class HistoricalBacktestResultPersistence:
    """Publish and read back one `HistoricalBacktestAuditResult` bundle."""

    @staticmethod
    def write(
        *, result: HistoricalBacktestAuditResult, destination: str | Path
    ) -> Path:
        if not isinstance(destination, (str, Path)):
            raise HistoricalBacktestPersistenceError(
                "INVALID_PERSISTENCE_PATH",
                "destination must be a str or Path",
            )
        destination_path = Path(destination)
        if destination_path.exists():
            raise HistoricalBacktestPersistenceError(
                "BUNDLE_ALREADY_EXISTS",
                f"destination already exists: {destination_path}",
            )

        # Structurally revalidate through the frozen model's own constructor
        # (never `model_validate` on an existing instance, which pydantic
        # v2 accepts without revalidation by default; and never a
        # dump-to-dict round trip, which would defeat the strict
        # pre-built-instance validators on `run_manifest`/`initial_state`/
        # `final_state`). Passing the same nested object references still
        # forces every field and model validator — including the frozen
        # model's own `result_fingerprint`/`audit_passed` re-derivation — to
        # run again from scratch.
        try:
            validated = HistoricalBacktestAuditResult(
                **{
                    name: getattr(result, name)
                    for name in HistoricalBacktestAuditResult.model_fields
                }
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise HistoricalBacktestPersistenceError(
                "PERSISTED_MODEL_INVALID",
                "the supplied result fails structural revalidation",
            ) from error

        row_counts: dict[str, int] = {}
        payload_sha256: dict[str, str] = {}

        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise HistoricalBacktestPersistenceError(
                "BUNDLE_WRITE_FAILED",
                f"failed to create destination parent for {destination_path}",
            ) from error

        try:
            temp_dir = Path(
                tempfile.mkdtemp(
                    prefix=".historical_backtest_result_bundle.",
                    dir=str(destination_path.parent),
                )
            )
        except OSError as error:
            raise HistoricalBacktestPersistenceError(
                "BUNDLE_WRITE_FAILED", "failed to create a temporary bundle directory"
            ) from error

        try:
            _write_json_payloads(validated, temp_dir)
            _write_parquet_payloads(validated, temp_dir, row_counts)

            for filename in schema.PAYLOAD_FILENAMES:
                payload_sha256[filename] = schema.sha256_file(temp_dir / filename)

            manifest = schema.build_bundle_manifest(
                result_schema_version=validated.schema_version,
                run_configuration_fingerprint=(
                    validated.run_manifest.run_configuration_fingerprint
                ),
                source_run_fingerprint=validated.source_run_fingerprint,
                initial_state_fingerprint=validated.initial_state_fingerprint,
                final_state_fingerprint=validated.final_state_fingerprint,
                result_fingerprint=validated.result_fingerprint,
                content_fingerprints=validated.content_fingerprints,
                row_counts=row_counts,
                payload_sha256=payload_sha256,
            )
            manifest_bytes = schema.encode_bundle_manifest(manifest)
            try:
                (temp_dir / schema.MANIFEST_FILENAME).write_bytes(manifest_bytes)
            except OSError as error:
                raise HistoricalBacktestPersistenceError(
                    "BUNDLE_WRITE_FAILED", "failed to write manifest.json"
                ) from error

            read_back = HistoricalBacktestResultPersistence.read(source=temp_dir)
            if read_back != validated:
                raise HistoricalBacktestPersistenceError(
                    "BUNDLE_WRITE_FAILED",
                    "read-back of the freshly written bundle did not exactly "
                    "equal the source result",
                )

            try:
                temp_dir.rename(destination_path)
            except OSError as error:
                raise HistoricalBacktestPersistenceError(
                    "BUNDLE_WRITE_FAILED",
                    f"failed to publish bundle to {destination_path}",
                ) from error
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return destination_path

    @staticmethod
    def read(*, source: str | Path) -> HistoricalBacktestAuditResult:
        if not isinstance(source, (str, Path)):
            raise HistoricalBacktestPersistenceError(
                "INVALID_PERSISTENCE_PATH", "source must be a str or Path"
            )
        source_path = Path(source)
        if not source_path.is_dir():
            raise HistoricalBacktestPersistenceError(
                "INVALID_PERSISTENCE_PATH",
                f"source is not a directory: {source_path}",
            )

        manifest_path = source_path / schema.MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise HistoricalBacktestPersistenceError(
                "BUNDLE_INVENTORY_MISMATCH",
                f"missing {schema.MANIFEST_FILENAME}",
            )
        manifest = schema.decode_bundle_manifest(manifest_path.read_bytes())
        if (
            manifest.result_schema_version
            != HISTORICAL_BACKTEST_AUDIT_RESULT_SCHEMA_VERSION
        ):
            raise HistoricalBacktestPersistenceError(
                "UNSUPPORTED_SCHEMA_VERSION",
                "persisted audit result uses an unsupported result schema; "
                "rerun the historical backtest with an explicit "
                "decision_interval",
            )

        actual_names = {path.name for path in source_path.iterdir()}
        expected_names = set(schema.BUNDLE_FILENAMES)
        if actual_names != expected_names:
            raise HistoricalBacktestPersistenceError(
                "BUNDLE_INVENTORY_MISMATCH",
                "bundle directory does not contain exactly the expected "
                f"{len(expected_names)} files "
                f"(missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)})",
            )
        if manifest.payload_files != schema.PAYLOAD_FILENAMES:
            raise HistoricalBacktestPersistenceError(
                "BUNDLE_INVENTORY_MISMATCH",
                "manifest.json payload_files does not match the frozen "
                "payload inventory",
            )

        for filename in schema.PAYLOAD_FILENAMES:
            expected_hash = manifest.payload_sha256.get(filename)
            if expected_hash is None:
                raise HistoricalBacktestPersistenceError(
                    "BUNDLE_INVENTORY_MISMATCH",
                    f"manifest.json is missing a physical hash for {filename}",
                )
            actual_hash = schema.sha256_file(source_path / filename)
            if actual_hash != expected_hash:
                raise HistoricalBacktestPersistenceError(
                    "PHYSICAL_HASH_MISMATCH",
                    f"physical SHA-256 mismatch for {filename}",
                )

        run_manifest = schema.decode_run_manifest(
            (source_path / schema.RUN_MANIFEST_FILENAME).read_bytes()
        )
        initial_state = schema.decode_portfolio_state(
            (source_path / schema.INITIAL_STATE_FILENAME).read_bytes(),
            schema.INITIAL_STATE_FILENAME,
        )
        final_state = schema.decode_portfolio_state(
            (source_path / schema.FINAL_STATE_FILENAME).read_bytes(),
            schema.FINAL_STATE_FILENAME,
        )

        recomputed_initial_hash = hash_portfolio_state(initial_state)
        if recomputed_initial_hash != manifest.initial_state_fingerprint:
            raise HistoricalBacktestPersistenceError(
                "STATE_FINGERPRINT_MISMATCH",
                "recomputed initial_state hash does not match "
                "manifest.initial_state_fingerprint",
            )
        recomputed_final_hash = hash_portfolio_state(final_state)
        if recomputed_final_hash != manifest.final_state_fingerprint:
            raise HistoricalBacktestPersistenceError(
                "STATE_FINGERPRINT_MISMATCH",
                "recomputed final_state hash does not match "
                "manifest.final_state_fingerprint",
            )

        summary_payload = schema.decode_summary_payload(
            (source_path / schema.SUMMARY_FILENAME).read_bytes()
        )

        tables: dict[str, tuple[object, ...]] = {}
        for table_name in schema.TABLE_NAMES:
            table_path = source_path / f"{table_name}.parquet"
            arrow_table = schema.read_parquet_table(
                table_path, schema.TABLE_SCHEMAS[table_name], table_name
            )
            rows = schema.verify_and_strip_ordinal(arrow_table, table_name)
            if table_name == "trades":
                tables[table_name] = tuple(
                    schema.decode_trade_row(row) for row in rows
                )
            else:
                model_cls = schema.TABLE_MODEL_CLASSES[table_name]
                tables[table_name] = tuple(
                    schema.decode_model_row(model_cls, row) for row in rows
                )

        content_fingerprint_kwargs = {}
        for artifact_name in HistoricalBacktestContentFingerprints.model_fields:
            if artifact_name == "cost_summary":
                value = summary_payload.cost_summary
            elif artifact_name == "exit_reason_summary":
                value = summary_payload.exit_reason_summary
            elif artifact_name == "summary":
                value = summary_payload.summary
            elif artifact_name == "audit_summary":
                value = summary_payload.audit_summary
            else:
                value = tables[artifact_name]
            recomputed = compute_content_fingerprint(
                artifact_name=artifact_name, rows_or_value=value
            )
            expected = getattr(manifest.content_fingerprints, artifact_name)
            if recomputed != expected:
                raise HistoricalBacktestPersistenceError(
                    "CONTENT_FINGERPRINT_MISMATCH",
                    f"recomputed content fingerprint for {artifact_name!r} "
                    "does not match the persisted manifest",
                )
            content_fingerprint_kwargs[artifact_name] = recomputed

        content_fingerprints = HistoricalBacktestContentFingerprints(
            **content_fingerprint_kwargs
        )

        recomputed_result_fingerprint = compute_result_fingerprint(
            schema_version=manifest.result_schema_version,
            decision_interval=summary_payload.decision_interval,
            run_configuration_fingerprint=run_manifest.run_configuration_fingerprint,
            source_run_fingerprint=manifest.source_run_fingerprint,
            initial_state_fingerprint=manifest.initial_state_fingerprint,
            final_state_fingerprint=manifest.final_state_fingerprint,
            content_fingerprints=content_fingerprints,
        )
        if recomputed_result_fingerprint != manifest.result_fingerprint:
            raise HistoricalBacktestPersistenceError(
                "RESULT_FINGERPRINT_MISMATCH",
                "recomputed result_fingerprint does not match "
                "manifest.result_fingerprint",
            )

        try:
            return HistoricalBacktestAuditResult(
                schema_version=manifest.result_schema_version,
                run_manifest=run_manifest,
                source_run_fingerprint=manifest.source_run_fingerprint,
                decision_interval=summary_payload.decision_interval,
                initial_state=initial_state,
                final_state=final_state,
                initial_state_fingerprint=manifest.initial_state_fingerprint,
                final_state_fingerprint=manifest.final_state_fingerprint,
                initial_equity=summary_payload.initial_equity,
                final_equity=summary_payload.final_equity,
                period_pnl=summary_payload.period_pnl,
                session_transitions=tables["session_transitions"],
                signal_provenance=tables["signal_provenance"],
                ranking_cycles=tables["ranking_cycles"],
                ranking_candidates=tables["ranking_candidates"],
                allocation_cycles=tables["allocation_cycles"],
                allocation_candidates=tables["allocation_candidates"],
                execution_provenance=tables["execution_provenance"],
                entries=tables["entries"],
                exits=tables["exits"],
                trades=tables["trades"],
                rejections=tables["rejections"],
                cash_ledger=tables["cash_ledger"],
                settlement_ledger=tables["settlement_ledger"],
                session_pnl=tables["session_pnl"],
                equity_curve=tables["equity_curve"],
                cost_summary=summary_payload.cost_summary,
                exit_reason_summary=summary_payload.exit_reason_summary,
                summary=summary_payload.summary,
                audit_summary=summary_payload.audit_summary,
                content_fingerprints=content_fingerprints,
                result_fingerprint=manifest.result_fingerprint,
            )
        except HistoricalBacktestPersistenceError:
            raise
        except Exception as error:  # noqa: BLE001 - narrow, immediate re-wrap
            raise HistoricalBacktestPersistenceError(
                "PERSISTED_MODEL_INVALID",
                "reconstructed HistoricalBacktestAuditResult fails model "
                "validation",
            ) from error


def _write_json_payloads(
    result: HistoricalBacktestAuditResult, temp_dir: Path
) -> None:
    payloads = {
        schema.RUN_MANIFEST_FILENAME: schema.encode_run_manifest(
            result.run_manifest
        ),
        schema.INITIAL_STATE_FILENAME: schema.encode_portfolio_state(
            result.initial_state
        ),
        schema.FINAL_STATE_FILENAME: schema.encode_portfolio_state(
            result.final_state
        ),
        schema.SUMMARY_FILENAME: schema.encode_summary_payload(
            decision_interval=result.decision_interval,
            initial_equity=result.initial_equity,
            final_equity=result.final_equity,
            period_pnl=result.period_pnl,
            cost_summary=result.cost_summary,
            exit_reason_summary=result.exit_reason_summary,
            summary=result.summary,
            audit_summary=result.audit_summary,
        ),
    }
    for filename, raw_bytes in payloads.items():
        try:
            (temp_dir / filename).write_bytes(raw_bytes)
        except OSError as error:
            raise HistoricalBacktestPersistenceError(
                "BUNDLE_WRITE_FAILED", f"failed to write {filename}"
            ) from error


def _write_parquet_payloads(
    result: HistoricalBacktestAuditResult,
    temp_dir: Path,
    row_counts: dict[str, int],
) -> None:
    for table_name in schema.TABLE_NAMES:
        members = getattr(result, table_name)
        if table_name == "trades":
            rows = [
                schema.encode_trade_row(member, ordinal)
                for ordinal, member in enumerate(members)
            ]
        else:
            rows = [
                schema.encode_model_row(member, ordinal)
                for ordinal, member in enumerate(members)
            ]
        arrow_schema = schema.TABLE_SCHEMAS[table_name]
        table = (
            arrow_schema.empty_table()
            if not rows
            else schema.build_arrow_table(rows, arrow_schema)
        )
        row_counts[table_name] = len(rows)
        schema.write_parquet_table(table, temp_dir / f"{table_name}.parquet")


__all__ = ["HistoricalBacktestResultPersistence"]
