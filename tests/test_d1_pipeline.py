import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

import stock_swing_d1.data.d1_pipeline as pipeline_module
from stock_swing_d1.data.d1_pipeline import generate_batch_id, run_d1_pipeline
from stock_swing_d1.validation.stock_bar_dataset import (
    DatasetValidationError,
    DatasetValidationIssue,
)


def frame_for(
    asset_id: int,
    *,
    close: float = 101.0,
    unadjusted_close: float = 101.0,
    open_price: float = 100.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [open_price],
            "High": [102.0],
            "Low": [99.0],
            "Close": [close],
            "Volume": [float(1_000_000 + asset_id)],
            "Turnover": [None],
            "Unadjusted Close": [unadjusted_close],
            "Dividend": [None],
        },
        index=pd.to_datetime(["2026-08-07"]),
    )


def symbols(asset_id: int) -> str:
    return {1: "AAA", 2: "BBB"}.get(asset_id, f"S{asset_id}")


def load_json(path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_batch_id_uses_frozen_utc_format() -> None:
    instant = datetime(2026, 8, 11, 13, 14, 20, tzinfo=timezone.utc)

    assert generate_batch_id(instant) == "20260811T131420Z"


def test_successful_one_asset_raw_clean_and_published_flow(tmp_path) -> None:
    result = run_d1_pipeline(
        [1],
        date(2026, 8, 7),
        date(2026, 8, 7),
        data_root=tmp_path,
        batch_id="batch1",
        fetcher=lambda asset_id, start, end: frame_for(asset_id),
        symbol_resolver=symbols,
        norgatedata_package_version="test-version",
    )

    assert result.published is True
    assert (result.raw_batch_directory / "bars.parquet").is_file()
    assert (result.raw_batch_directory / "manifest.json").is_file()
    assert result.clean_parquet_path.is_file()
    assert result.published_paths == (
        tmp_path / "parquet/stock_bars_v0_1/year=2026/part-batch1.parquet",
    )
    assert result.validation_report == {
        "batch_id": "batch1",
        "raw_row_count": 1,
        "valid_row_count": 1,
        "invalid_row_count": 0,
        "duplicate_count": 0,
        "number_of_securities": 1,
        "minimum_trading_date": "2026-08-07",
        "maximum_trading_date": "2026-08-07",
        "published": True,
        "errors": [],
    }
    assert load_json(result.validation_report_path) == result.validation_report


def test_multiple_asset_ids_are_sorted_deduplicated_and_only_explicit_ids_fetch(
    tmp_path,
) -> None:
    calls: list[int] = []

    def fetcher(asset_id: int, start: date, end: date) -> pd.DataFrame:
        calls.append(asset_id)
        return frame_for(asset_id)

    result = run_d1_pipeline(
        [2, 1, 2],
        date(2026, 8, 7),
        date(2026, 8, 7),
        data_root=tmp_path,
        batch_id="batch1",
        fetcher=fetcher,
        symbol_resolver=symbols,
        norgatedata_package_version="test-version",
    )

    assert result.published is True
    assert calls == [1, 2]
    raw_manifest = load_json(result.raw_batch_directory / "manifest.json")
    assert raw_manifest["requested_asset_ids"] == [1, 2]
    assert raw_manifest["raw_row_count"] == 2
    assert result.validation_report["number_of_securities"] == 2


def test_provider_validation_failure_keeps_raw_and_blocks_clean_and_publish(
    tmp_path,
) -> None:
    result = run_d1_pipeline(
        [1],
        date(2026, 8, 7),
        date(2026, 8, 7),
        data_root=tmp_path,
        batch_id="badprovider",
        fetcher=lambda asset_id, start, end: frame_for(
            asset_id, unadjusted_close=100.5
        ),
        symbol_resolver=symbols,
        norgatedata_package_version="test-version",
    )

    assert result.published is False
    assert (result.raw_batch_directory / "bars.parquet").is_file()
    assert result.clean_parquet_path is None
    assert not (
        tmp_path / "clean/d1/batch=badprovider/stock_bars.parquet"
    ).exists()
    assert not (tmp_path / "parquet/stock_bars_v0_1").exists()
    report = load_json(result.validation_report_path)
    assert report["published"] is False
    assert report["invalid_row_count"] == 1
    assert report["errors"][0]["stage"] == "provider_validation"


def test_stock_bar_validation_failure_blocks_publication(tmp_path) -> None:
    result = run_d1_pipeline(
        [1],
        date(2026, 8, 7),
        date(2026, 8, 7),
        data_root=tmp_path,
        batch_id="badbar",
        fetcher=lambda asset_id, start, end: frame_for(
            asset_id, open_price=103.0
        ),
        symbol_resolver=symbols,
        norgatedata_package_version="test-version",
    )

    assert result.published is False
    assert (result.raw_batch_directory / "bars.parquet").is_file()
    assert result.clean_parquet_path is None
    assert result.validation_report["errors"][0]["stage"] == (
        "stock_bar_validation"
    )


def test_dataset_validation_failure_blocks_clean_and_publication(
    tmp_path, monkeypatch
) -> None:
    def fail_dataset(rows):
        raise DatasetValidationError(
            [DatasetValidationIssue("duplicate canonical uniqueness key")],
            duplicate_count=1,
        )

    monkeypatch.setattr(
        pipeline_module, "validate_stock_bar_dataset", fail_dataset
    )

    result = run_d1_pipeline(
        [1, 2],
        date(2026, 8, 7),
        date(2026, 8, 7),
        data_root=tmp_path,
        batch_id="baddataset",
        fetcher=lambda asset_id, start, end: frame_for(asset_id),
        symbol_resolver=symbols,
        norgatedata_package_version="test-version",
    )

    assert result.published is False
    assert result.clean_parquet_path is None
    assert result.validation_report["valid_row_count"] == 2
    assert result.validation_report["invalid_row_count"] == 0
    assert result.validation_report["duplicate_count"] == 1
    assert result.validation_report["number_of_securities"] == 2
    assert result.validation_report["minimum_trading_date"] == "2026-08-07"
    assert result.validation_report["maximum_trading_date"] == "2026-08-07"
    assert result.validation_report["errors"][0]["stage"] == (
        "dataset_validation"
    )


def test_clean_parquet_failure_preserves_validated_row_counts(
    tmp_path, monkeypatch
) -> None:
    def fail_clean_write(*args, **kwargs):
        raise pipeline_module.ParquetValidationError("simulated write failure")

    monkeypatch.setattr(pipeline_module, "write_clean_batch", fail_clean_write)

    result = run_d1_pipeline(
        [1, 2],
        date(2026, 8, 7),
        date(2026, 8, 7),
        data_root=tmp_path,
        batch_id="badcleanwrite",
        fetcher=lambda asset_id, start, end: frame_for(asset_id),
        symbol_resolver=symbols,
        norgatedata_package_version="test-version",
    )

    report = result.validation_report
    assert result.published is False
    assert result.clean_parquet_path is None
    assert report["valid_row_count"] == 2
    assert report["invalid_row_count"] == 0
    assert report["duplicate_count"] == 0
    assert report["number_of_securities"] == 2
    assert report["minimum_trading_date"] == "2026-08-07"
    assert report["maximum_trading_date"] == "2026-08-07"
    assert report["published"] is False
    assert report["errors"] == [
        {"stage": "parquet_write", "reason": "simulated write failure"}
    ]


def test_raw_parquet_failure_does_not_label_unvalidated_rows_invalid(
    tmp_path, monkeypatch
) -> None:
    def fail_raw_write(*args, **kwargs):
        raise pipeline_module.ParquetValidationError("simulated raw write failure")

    monkeypatch.setattr(pipeline_module, "write_raw_batch", fail_raw_write)

    result = run_d1_pipeline(
        [1],
        date(2026, 8, 7),
        date(2026, 8, 7),
        data_root=tmp_path,
        batch_id="badrawwrite",
        fetcher=lambda asset_id, start, end: frame_for(asset_id),
        symbol_resolver=symbols,
        norgatedata_package_version="test-version",
    )

    report = result.validation_report
    assert result.published is False
    assert report["raw_row_count"] == 1
    assert report["valid_row_count"] == 0
    assert report["invalid_row_count"] == 0
    assert report["errors"] == [
        {"stage": "parquet_write", "reason": "simulated raw write failure"}
    ]


def test_existing_published_overlap_keeps_clean_batch_but_reports_not_published(
    tmp_path,
) -> None:
    common = {
        "asset_ids": [1],
        "start_date": date(2026, 8, 7),
        "end_date": date(2026, 8, 7),
        "data_root": tmp_path,
        "fetcher": lambda asset_id, start, end: frame_for(asset_id),
        "symbol_resolver": symbols,
        "norgatedata_package_version": "test-version",
    }
    first = run_d1_pipeline(**common, batch_id="batch1")
    second = run_d1_pipeline(**common, batch_id="batch2")

    assert first.published is True
    assert second.published is False
    assert second.clean_parquet_path.is_file()
    assert second.validation_report["valid_row_count"] == 1
    assert second.validation_report["invalid_row_count"] == 0
    assert second.validation_report["duplicate_count"] == 1
    assert second.validation_report["errors"][0]["stage"] == (
        "publish_validation"
    )
    assert not (
        tmp_path
        / "parquet/stock_bars_v0_1/year=2026/part-batch2.parquet"
    ).exists()


def test_no_observations_is_not_silently_successful(tmp_path) -> None:
    empty = frame_for(1).iloc[0:0]

    result = run_d1_pipeline(
        [1],
        date(2026, 8, 7),
        date(2026, 8, 7),
        data_root=tmp_path,
        batch_id="emptybatch",
        fetcher=lambda asset_id, start, end: empty,
        symbol_resolver=symbols,
        norgatedata_package_version="test-version",
    )

    assert result.published is False
    assert (result.raw_batch_directory / "bars.parquet").is_file()
    assert result.validation_report["raw_row_count"] == 0
    assert result.validation_report["errors"][0]["reason"] == (
        "no observations returned"
    )


def test_provider_runtime_exception_is_not_swallowed(tmp_path) -> None:
    class ProviderFailure(RuntimeError):
        pass

    def fetcher(asset_id: int, start: date, end: date) -> pd.DataFrame:
        raise ProviderFailure("NDU unavailable")

    with pytest.raises(ProviderFailure, match="NDU unavailable"):
        run_d1_pipeline(
            [1],
            date(2026, 8, 7),
            date(2026, 8, 7),
            data_root=tmp_path,
            batch_id="providerdown",
            fetcher=fetcher,
            symbol_resolver=symbols,
        )
