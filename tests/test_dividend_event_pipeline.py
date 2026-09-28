from datetime import date
import hashlib
import json
from math import inf, nan
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from stock_swing_d1.data.dividend_event_pipeline import (
    run_dividend_event_pipeline,
)
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    DividendEvent,
    StockBar,
)
from stock_swing_d1.storage.corporate_action_parquet import (
    ADJUSTED_RAW_ARROW_SCHEMA,
    write_corporate_action_clean_batch,
    write_corporate_action_raw_batch,
    write_corporate_action_validation_report,
)
from stock_swing_d1.storage.dividend_event_parquet import (
    DIVIDEND_EVENT_ARROW_SCHEMA,
    verify_dividend_event_parquet,
)
from stock_swing_d1.storage.stock_bar_parquet import (
    BatchExistsError,
    write_clean_batch,
)


def raw_row(
    dividend: object,
    *,
    asset_id: int = 1,
    symbol: str | None = None,
    trading_date: date = date(2025, 12, 31),
) -> dict[str, object]:
    return {
        "provider_asset_id": asset_id,
        "provider_symbol": symbol or f"S{asset_id}",
        "trading_date": trading_date,
        "Open": 10.0,
        "High": 11.0,
        "Low": 9.0,
        "Close": 10.5,
        "Volume": 100.0,
        "Turnover": None,
        "Unadjusted Close": 10.5,
        "Dividend": dividend,
    }


def phase4_bar(record: dict[str, object], *, symbol: str | None = None) -> StockBar:
    asset_id = int(record["provider_asset_id"])
    return StockBar(
        security_id=f"NORGATE:{asset_id}",
        symbol=symbol or str(record["provider_symbol"]),
        trading_date=record["trading_date"],
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=10.0,
        high=11.0,
        low=9.0,
        close=10.5,
        volume=100,
    )


def adjusted_bar(
    record: dict[str, object], *, symbol: str | None = None
) -> CorporateActionAdjustedStockBar:
    asset_id = int(record["provider_asset_id"])
    return CorporateActionAdjustedStockBar(
        security_id=f"NORGATE:{asset_id}",
        symbol=symbol or str(record["provider_symbol"]),
        trading_date=record["trading_date"],
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="capital_special_adjusted",
        open=10.0,
        high=11.0,
        low=9.0,
        close=10.5,
        volume=100.0,
    )


def write_source(
    data_root: Path,
    raw_rows: list[dict[str, object]],
    *,
    phase5_id: str = "phase5",
    phase4_id: str = "phase4",
    phase4_bars: list[StockBar] | None = None,
    adjusted_bars: list[CorporateActionAdjustedStockBar] | None = None,
    adjustment_mode: str = "CAPITALSPECIAL",
    provider: str = "Norgate Data",
    published: bool = True,
    adjusted_published: bool = True,
    errors: list[object] | None = None,
) -> dict[str, Path]:
    source_phase4_bars = (
        phase4_bars
        if phase4_bars is not None
        else [phase4_bar(record) for record in raw_rows]
    )
    source_adjusted_bars = (
        adjusted_bars
        if adjusted_bars is not None
        else [adjusted_bar(record) for record in raw_rows]
    )
    phase4_path = write_clean_batch(
        source_phase4_bars, data_root=data_root, batch_id=phase4_id
    )
    adjusted_frame = pd.DataFrame.from_records(raw_rows).loc[
        :, list(ADJUSTED_RAW_ARROW_SCHEMA.names)
    ]
    event_frame = pd.DataFrame(
        columns=[
            "provider_asset_id",
            "provider_symbol",
            "event_date",
            "capital_event_flag",
        ]
    )
    manifest = {
        "provider": provider,
        "package": "US Stocks Platinum",
        "phase5_batch_id": phase5_id,
        "phase4_source_batch_id": phase4_id,
        "phase4_source_path": str(phase4_path),
        "extracted_at_utc": "2026-08-12T10:00:00Z",
        "requested_asset_ids": sorted(
            {int(record["provider_asset_id"]) for record in raw_rows}
        ),
        "requested_date_ranges": [],
        "adjustment_mode": adjustment_mode,
        "padding_mode": "NONE",
        "interval": "D",
        "adjusted_raw_row_count": len(raw_rows),
        "capital_event_flag_row_count": 0,
        "norgatedata_python_package_version": "test",
        "event_source_function": "capital_event_timeseries",
        "event_date_filtering": "local_phase4_date_range",
    }
    raw_directory = write_corporate_action_raw_batch(
        adjusted_frame,
        event_frame,
        manifest,
        data_root=data_root,
        batch_id=phase5_id,
    )
    clean_paths = write_corporate_action_clean_batch(
        source_adjusted_bars,
        [],
        data_root=data_root,
        batch_id=phase5_id,
    )
    report = {
        "phase5_batch_id": phase5_id,
        "phase4_source_batch_id": phase4_id,
        "phase4_row_count": len(source_phase4_bars),
        "adjusted_raw_row_count": len(raw_rows),
        "adjusted_valid_row_count": len(source_adjusted_bars),
        "adjusted_published": adjusted_published,
        "published": published,
        "errors": errors if errors is not None else [],
    }
    report_path = write_corporate_action_validation_report(
        report, data_root=data_root, batch_id=phase5_id
    )
    return {
        "phase4": phase4_path,
        "raw": raw_directory / "adjusted_bars.parquet",
        "manifest": raw_directory / "manifest.json",
        "adjusted": clean_paths.adjusted_stock_bars,
        "report": report_path,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_raw_dividend(path: Path, rows: list[dict[str, object]]) -> None:
    table = ADJUSTED_RAW_ARROW_SCHEMA.empty_table()
    if rows:
        import pyarrow as pa

        table = pa.Table.from_pylist(rows, schema=ADJUSTED_RAW_ARROW_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def test_positive_pipeline_maps_stores_publishes_and_reports(tmp_path) -> None:
    source_paths = write_source(tmp_path, [raw_row(0.25)])
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}

    result = run_dividend_event_pipeline(
        "phase5", data_root=tmp_path, dividend_batch_id="div1"
    )

    assert result.published
    assert result.phase4_source_batch_id == "phase4"
    assert result.clean_parquet_path == (
        tmp_path / "dividends/clean/batch=div1/dividend_events.parquet"
    )
    assert result.published_paths == (
        tmp_path / "parquet/dividend_events_v0_1/year=2025/part-div1.parquet",
    )
    events = verify_dividend_event_parquet(result.clean_parquet_path)
    assert events == [
        DividendEvent(
            security_id="NORGATE:1",
            symbol="S1",
            entitlement_date=date(2025, 12, 31),
            date_semantics="entitlement_close",
            dividend_type="ordinary_cash",
            amount_per_share=0.25,
            currency="USD",
            source_provider="Norgate Data",
            source_asset_id=1,
            source_adjustment_mode="CAPITALSPECIAL",
        )
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["phase5_source_raw_path"] == str(source_paths["raw"])
    assert manifest["phase5_source_adjusted_clean_path"] == str(
        source_paths["adjusted"]
    )
    assert manifest["phase4_source_path"] == str(source_paths["phase4"])
    assert manifest["source_raw_row_count"] == 1
    assert manifest["source_dividend_nonzero_count"] == 1
    assert result.manifest_path.read_bytes().endswith(b"\n")
    assert result.validation_report == {
        "dividend_batch_id": "div1",
        "phase5_source_batch_id": "phase5",
        "phase4_source_batch_id": "phase4",
        "source_raw_row_count": 1,
        "raw_dividend_nonzero_count": 1,
        "dividend_event_count": 1,
        "dividend_invalid_count": 0,
        "dividend_missing_adjusted_bar_count": 0,
        "dividend_missing_phase4_bar_count": 0,
        "duplicate_dividend_key_count": 0,
        "number_of_securities_with_dividends": 1,
        "minimum_entitlement_date": "2025-12-31",
        "maximum_entitlement_date": "2025-12-31",
        "dividend_published": True,
        "published": True,
        "errors": [],
    }
    assert json.loads(result.validation_report_path.read_text(encoding="utf-8")) == (
        result.validation_report
    )
    assert {name: sha256(path) for name, path in source_paths.items()} == source_hashes


def test_zero_and_arrow_null_pipeline_succeeds_without_year_partition(tmp_path) -> None:
    write_source(
        tmp_path,
        [raw_row(0), raw_row(None, trading_date=date(2026, 1, 2))],
    )

    result = run_dividend_event_pipeline(
        "phase5", data_root=tmp_path, dividend_batch_id="empty"
    )

    assert result.published and result.validation_report["dividend_published"]
    assert result.published_paths == ()
    table = pq.ParquetFile(result.clean_parquet_path).read()
    assert table.num_rows == 0
    assert table.schema.equals(DIVIDEND_EVENT_ARROW_SCHEMA, check_metadata=False)
    assert not (tmp_path / "parquet/dividend_events_v0_1").exists()
    assert result.validation_report["number_of_securities_with_dividends"] == 0
    assert result.validation_report["minimum_entitlement_date"] is None
    assert result.validation_report["maximum_entitlement_date"] is None
    assert result.validation_report["errors"] == []


@pytest.mark.parametrize("value", [-0.25, inf, -inf, nan])
def test_invalid_numeric_raw_dividend_blocks_clean_and_publication(
    tmp_path, value: float
) -> None:
    rows = [raw_row(value)]
    paths = write_source(tmp_path, [raw_row(0.0)])
    replace_raw_dividend(paths["raw"], rows)

    result = run_dividend_event_pipeline(
        "phase5", data_root=tmp_path, dividend_batch_id="invalid"
    )

    assert not result.published
    assert result.clean_parquet_path is None
    assert result.validation_report["dividend_invalid_count"] == 1
    assert result.validation_report["errors"][0]["stage"] == (
        "dividend_source_validation"
    )
    assert not (tmp_path / "parquet/dividend_events_v0_1").exists()


@pytest.mark.parametrize("mode", ["NONE", "CAPITAL", "TOTALRETURN", "OTHER"])
def test_only_capitalspecial_phase5_provenance_is_accepted(tmp_path, mode: str) -> None:
    write_source(tmp_path, [raw_row(0.25)], adjustment_mode=mode)

    result = run_dividend_event_pipeline(
        "phase5", data_root=tmp_path, dividend_batch_id="wrongmode"
    )

    assert not result.published
    assert result.validation_report["errors"][0]["stage"] == (
        "phase5_source_validation"
    )
    assert "CAPITALSPECIAL" in result.validation_report["errors"][0]["reason"]


@pytest.mark.parametrize(
    ("source_options", "reason_fragment"),
    [
        ({"provider": "Other"}, "provider"),
        ({"published": False}, "published"),
        ({"adjusted_published": False}, "adjusted_published"),
        ({"errors": [{"reason": "source failed"}]}, "errors"),
    ],
)
def test_provider_and_success_report_provenance_fail_closed(
    tmp_path, source_options: dict[str, object], reason_fragment: str
) -> None:
    write_source(tmp_path, [raw_row(0.25)], **source_options)

    result = run_dividend_event_pipeline(
        "phase5", data_root=tmp_path, dividend_batch_id="badsource"
    )

    assert not result.published
    assert result.validation_report["source_raw_row_count"] == 0
    assert result.validation_report["errors"][0]["stage"] == (
        "phase5_source_validation"
    )
    assert reason_fragment in result.validation_report["errors"][0]["reason"]


def test_malformed_phase5_raw_arrow_schema_blocks_derivation(tmp_path) -> None:
    paths = write_source(tmp_path, [raw_row(0.25)])
    malformed = pq.ParquetFile(paths["raw"]).read().drop(["Dividend"])
    pq.write_table(malformed, paths["raw"], compression="zstd")

    result = run_dividend_event_pipeline(
        "phase5", data_root=tmp_path, dividend_batch_id="malformed"
    )

    assert not result.published
    assert result.validation_report["errors"][0]["stage"] == (
        "phase5_source_validation"
    )
    assert "schema mismatch" in result.validation_report["errors"][0]["reason"]


def test_pipeline_never_calls_any_norgate_fetcher(tmp_path, monkeypatch) -> None:
    write_source(tmp_path, [raw_row(0.25)])

    def forbidden(*args, **kwargs):
        raise AssertionError("provider call is forbidden")

    monkeypatch.setattr(
        "stock_swing_d1.data.norgate_adjusted_d1_adapter.fetch_norgate_adjusted_d1",
        forbidden,
    )
    monkeypatch.setattr(
        "stock_swing_d1.data.norgate_capital_event_adapter.fetch_norgate_capital_events",
        forbidden,
    )

    result = run_dividend_event_pipeline(
        "phase5", data_root=tmp_path, dividend_batch_id="offline"
    )

    assert result.published


@pytest.mark.parametrize(
    ("kind", "stage", "missing_field"),
    [
        ("missing_adjusted", "adjusted_bar_crosscheck", "dividend_missing_adjusted_bar_count"),
        ("adjusted_symbol", "adjusted_bar_crosscheck", None),
        ("missing_phase4", "phase4_bar_crosscheck", "dividend_missing_phase4_bar_count"),
        ("phase4_symbol", "phase4_bar_crosscheck", None),
    ],
)
def test_pipeline_exact_bar_crosschecks_fail_closed(
    tmp_path, kind: str, stage: str, missing_field: str | None
) -> None:
    first = raw_row(0.0, trading_date=date(2025, 12, 30))
    dividend = raw_row(0.25, trading_date=date(2025, 12, 31))
    phase4 = [phase4_bar(first), phase4_bar(dividend)]
    adjusted = [adjusted_bar(first), adjusted_bar(dividend)]
    if kind == "missing_adjusted":
        adjusted = [adjusted_bar(first)]
    elif kind == "adjusted_symbol":
        adjusted[1] = adjusted_bar(dividend, symbol="ZZZ")
    elif kind == "missing_phase4":
        phase4 = [phase4_bar(first)]
    elif kind == "phase4_symbol":
        phase4[1] = phase4_bar(dividend, symbol="ZZZ")
    write_source(
        tmp_path,
        [first, dividend],
        phase4_bars=phase4,
        adjusted_bars=adjusted,
    )

    result = run_dividend_event_pipeline(
        "phase5", data_root=tmp_path, dividend_batch_id="crosscheck"
    )

    assert not result.published
    assert result.validation_report["errors"][0]["stage"] == stage
    if missing_field:
        assert result.validation_report[missing_field] == 1


def test_duplicate_raw_canonical_key_hard_fails_without_aggregation(tmp_path) -> None:
    rows = [raw_row(0.25), raw_row(0.5)]
    write_source(
        tmp_path,
        rows,
        phase4_bars=[phase4_bar(rows[0])],
        adjusted_bars=[adjusted_bar(rows[0])],
    )

    result = run_dividend_event_pipeline(
        "phase5", data_root=tmp_path, dividend_batch_id="duplicate"
    )

    assert not result.published
    assert result.validation_report["raw_dividend_nonzero_count"] == 2
    assert result.validation_report["dividend_event_count"] == 2
    assert result.validation_report["duplicate_dividend_key_count"] == 1
    assert result.validation_report["errors"][0]["stage"] == "dividend_uniqueness"


def test_existing_publication_overlap_keeps_clean_and_old_publication(tmp_path) -> None:
    first_source = write_source(
        tmp_path, [raw_row(0.25)], phase5_id="phase5a", phase4_id="phase4a"
    )
    first = run_dividend_event_pipeline(
        "phase5a", data_root=tmp_path, dividend_batch_id="diva"
    )
    old_path = first.published_paths[0]
    old_hash = sha256(old_path)
    first_source_hash = sha256(first_source["raw"])
    write_source(
        tmp_path, [raw_row(0.5)], phase5_id="phase5b", phase4_id="phase4b"
    )

    second = run_dividend_event_pipeline(
        "phase5b", data_root=tmp_path, dividend_batch_id="divb"
    )

    assert not second.published
    assert second.clean_parquet_path.is_file()
    assert second.validation_report["duplicate_dividend_key_count"] == 1
    assert second.validation_report["errors"][0]["stage"] == "publish_validation"
    assert sha256(old_path) == old_hash
    assert sha256(first_source["raw"]) == first_source_hash


def test_existing_dividend_batch_directory_fails_before_source_access(tmp_path) -> None:
    directory = tmp_path / "dividends/clean/batch=existing"
    directory.mkdir(parents=True)

    with pytest.raises(BatchExistsError):
        run_dividend_event_pipeline(
            "missing", data_root=tmp_path, dividend_batch_id="existing"
        )


@pytest.mark.parametrize("batch_id", ["../x", "..", "a/b", "a\\b", "a.b"])
def test_unsafe_dividend_batch_ids_are_rejected(tmp_path, batch_id: str) -> None:
    with pytest.raises(ValueError):
        run_dividend_event_pipeline(
            "phase5", data_root=tmp_path, dividend_batch_id=batch_id
        )


def test_missing_source_artifact_writes_failed_derived_report(tmp_path) -> None:
    result = run_dividend_event_pipeline(
        "missing", data_root=tmp_path, dividend_batch_id="failed"
    )

    assert not result.published
    assert result.clean_parquet_path is None
    assert result.validation_report_path.is_file()
    assert result.validation_report["errors"][0]["stage"] == (
        "phase5_source_validation"
    )


def test_exact_schema_excludes_portfolio_cash_payment_tax_and_reinvestment() -> None:
    excluded = {
        "shares_held",
        "shares_entitled",
        "gross_cash",
        "net_cash",
        "payment_date",
        "record_date",
        "ex_date",
        "tax_rate",
        "withholding_tax",
        "reinvested_shares",
        "portfolio_id",
        "position_id",
    }

    assert excluded.isdisjoint(DIVIDEND_EVENT_ARROW_SCHEMA.names)
    assert len(DIVIDEND_EVENT_ARROW_SCHEMA) == 10
