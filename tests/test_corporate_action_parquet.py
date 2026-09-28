# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
import json
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import stock_swing_d1.storage.corporate_action_parquet as storage_module
from stock_swing_d1.data.norgate_adjusted_d1_adapter import (
    prepare_norgate_adjusted_raw_frame,
)
from stock_swing_d1.data.norgate_capital_event_adapter import (
    prepare_norgate_capital_event_raw_frame,
)
from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    CorporateActionEvent,
)
from stock_swing_d1.storage.corporate_action_parquet import (
    ADJUSTED_CLEAN_ARROW_SCHEMA,
    ADJUSTED_RAW_ARROW_SCHEMA,
    EVENT_CLEAN_ARROW_SCHEMA,
    EVENT_RAW_ARROW_SCHEMA,
    CorporateActionDatasetValidationError,
    publish_adjusted_stock_bars,
    publish_corporate_action_datasets,
    publish_corporate_action_events,
    verify_adjusted_clean_parquet,
    verify_event_clean_parquet,
    verify_event_raw_parquet,
    write_corporate_action_clean_batch,
    write_corporate_action_raw_batch,
)
from stock_swing_d1.storage.stock_bar_parquet import (
    BatchExistsError,
    ParquetValidationError,
    PublishValidationError,
)


def provider_adjusted_frame(trading_date: str = "2026-08-07") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [50.0],
            "High": [51.0],
            "Low": [49.0],
            "Close": [50.5],
            "Volume": [2_000_000.25],
            "Turnover": [None],
            "Unadjusted Close": [101.0],
            "Dividend": [0.25],
        },
        index=pd.to_datetime([trading_date]),
    )


def adjusted_raw() -> pd.DataFrame:
    return prepare_norgate_adjusted_raw_frame(
        1_900_000_001, "SYNTHA", provider_adjusted_frame()
    )


def event_raw(flag: int = 1) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"Undocumented Name": [flag]},
        index=pd.to_datetime(["2026-08-07"]),
    )
    return prepare_norgate_capital_event_raw_frame(
        1_900_000_001,
        "SYNTHA",
        frame,
        start_date=date(2026, 8, 7),
        end_date=date(2026, 8, 7),
    )


def manifest(batch_id: str = "phase5") -> dict[str, object]:
    return {
        "provider": "Norgate Data",
        "package": "US Stocks Platinum",
        "phase5_batch_id": batch_id,
        "phase4_source_batch_id": "phase4",
        "phase4_source_path": "data/clean/d1/batch=phase4/stock_bars.parquet",
        "extracted_at_utc": "2026-08-12T12:00:00Z",
        "requested_asset_ids": [1_900_000_001],
        "requested_date_ranges": [
            {
                "provider_asset_id": 1_900_000_001,
                "start_date": "2026-08-07",
                "end_date": "2026-08-07",
            }
        ],
        "adjustment_mode": "CAPITALSPECIAL",
        "padding_mode": "NONE",
        "interval": "D",
        "adjusted_raw_row_count": 1,
        "capital_event_flag_row_count": 1,
        "norgatedata_python_package_version": "test",
        "event_source_function": "capital_event_timeseries",
        "event_date_filtering": "local_phase4_date_range",
    }


def adjusted_bar(
    *,
    trading_date: date = date(2026, 8, 7),
    security_id: str = "NORGATE:1900000001",
    symbol: str = "SYNTHA",
) -> CorporateActionAdjustedStockBar:
    return CorporateActionAdjustedStockBar(
        security_id=security_id,
        symbol=symbol,
        trading_date=trading_date,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="capital_special_adjusted",
        open=50.0,
        high=51.0,
        low=49.0,
        close=50.5,
        volume=2_000_000.25,
    )


def event(
    *,
    event_date: date = date(2026, 8, 7),
    security_id: str = "NORGATE:1900000001",
    symbol: str = "SYNTHA",
    asset_id: int = 1_900_000_001,
) -> CorporateActionEvent:
    return CorporateActionEvent(
        security_id=security_id,
        symbol=symbol,
        event_date=event_date,
        date_semantics="entitlement_close",
        event_type="unknown_capital_event",
        terms_verified=False,
        new_shares=None,
        old_shares=None,
        source_provider="Norgate Data",
        source_asset_id=asset_id,
    )


def assert_zstd(path) -> None:
    parquet_file = pq.ParquetFile(path)
    assert all(
        parquet_file.metadata.row_group(row_group).column(column).compression
        == "ZSTD"
        for row_group in range(parquet_file.metadata.num_row_groups)
        for column in range(parquet_file.metadata.row_group(row_group).num_columns)
    )


def test_raw_phase5_layout_schemas_manifest_and_compression(tmp_path) -> None:
    directory = write_corporate_action_raw_batch(
        adjusted_raw(),
        event_raw(),
        manifest(),
        data_root=tmp_path,
        batch_id="phase5",
    )

    assert directory == tmp_path / "corporate_actions/raw/norgate/batch=phase5"
    adjusted_path = directory / "adjusted_bars.parquet"
    event_path = directory / "capital_event_flags.parquet"
    manifest_path = directory / "manifest.json"
    assert adjusted_path.is_file() and event_path.is_file() and manifest_path.is_file()
    assert pq.ParquetFile(adjusted_path).schema_arrow.equals(
        ADJUSTED_RAW_ARROW_SCHEMA, check_metadata=False
    )
    assert pq.ParquetFile(event_path).schema_arrow.equals(
        EVENT_RAW_ARROW_SCHEMA, check_metadata=False
    )
    assert pq.ParquetFile(adjusted_path).schema_arrow.field("Volume").type == pa.float64()
    assert pq.ParquetFile(event_path).schema_arrow.field("capital_event_flag").type == pa.int8()
    assert pq.ParquetFile(adjusted_path).schema_arrow.field("trading_date").type == pa.date32()
    assert pq.ParquetFile(event_path).schema_arrow.field("event_date").type == pa.date32()
    assert "__index_level_0__" not in pq.ParquetFile(adjusted_path).schema_arrow.names
    assert "__index_level_0__" not in pq.ParquetFile(event_path).schema_arrow.names
    assert_zstd(adjusted_path)
    assert_zstd(event_path)
    assert manifest_path.read_bytes().endswith(b"\n")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["adjustment_mode"] == "CAPITALSPECIAL"
    assert stored["event_date_filtering"] == "local_phase4_date_range"


def test_raw_phase5_batch_overwrite_is_rejected(tmp_path) -> None:
    write_corporate_action_raw_batch(
        adjusted_raw(), event_raw(), manifest(), data_root=tmp_path, batch_id="phase5"
    )

    with pytest.raises(BatchExistsError):
        write_corporate_action_raw_batch(
            adjusted_raw(),
            event_raw(),
            manifest(),
            data_root=tmp_path,
            batch_id="phase5",
        )


def test_event_raw_readback_rejects_int8_flag_outside_binary_domain(
    tmp_path,
) -> None:
    path = tmp_path / "invalid-event-flags.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "provider_asset_id": 1_900_000_001,
                "provider_symbol": "SYNTHA",
                "event_date": date(2026, 8, 7),
                "capital_event_flag": 2,
            }
        ],
        schema=EVENT_RAW_ARROW_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")

    with pytest.raises(ParquetValidationError, match="exactly 0 or 1"):
        verify_event_raw_parquet(path, expected_row_count=1)


def test_clean_schemas_exact_round_trip_and_zstd(tmp_path) -> None:
    paths = write_corporate_action_clean_batch(
        [adjusted_bar()], [event()], data_root=tmp_path, batch_id="phase5"
    )

    assert paths.adjusted_stock_bars == (
        tmp_path / "corporate_actions/clean/batch=phase5/adjusted_stock_bars.parquet"
    )
    assert paths.corporate_action_events.is_file()
    assert pq.ParquetFile(paths.adjusted_stock_bars).schema_arrow.equals(
        ADJUSTED_CLEAN_ARROW_SCHEMA, check_metadata=False
    )
    assert pq.ParquetFile(paths.corporate_action_events).schema_arrow.equals(
        EVENT_CLEAN_ARROW_SCHEMA, check_metadata=False
    )
    assert pq.ParquetFile(paths.adjusted_stock_bars).schema_arrow.field("volume").type == pa.float64()
    assert pq.ParquetFile(paths.adjusted_stock_bars).schema_arrow.field("trading_date").type == pa.date32()
    assert pq.ParquetFile(paths.corporate_action_events).schema_arrow.field("event_date").type == pa.date32()
    assert verify_adjusted_clean_parquet(paths.adjusted_stock_bars) == [adjusted_bar()]
    assert verify_event_clean_parquet(paths.corporate_action_events) == [event()]
    assert_zstd(paths.adjusted_stock_bars)
    assert_zstd(paths.corporate_action_events)


def test_zero_events_writes_clean_zero_row_file_without_event_publication(tmp_path) -> None:
    paths = write_corporate_action_clean_batch(
        [adjusted_bar()], [], data_root=tmp_path, batch_id="phase5"
    )

    assert pq.ParquetFile(paths.corporate_action_events).metadata.num_rows == 0
    published = publish_corporate_action_datasets(
        [adjusted_bar()], [], data_root=tmp_path, batch_id="phase5"
    )
    assert len(published.adjusted_stock_bars) == 1
    assert published.corporate_action_events == ()
    assert not (
        tmp_path / "parquet/corporate_action_events_v0_1"
    ).exists()


def test_adjusted_publication_uses_year_only_partitions(tmp_path) -> None:
    paths = publish_adjusted_stock_bars(
        [
            adjusted_bar(trading_date=date(2025, 12, 31)),
            adjusted_bar(trading_date=date(2026, 1, 2)),
        ],
        data_root=tmp_path,
        batch_id="phase5",
    )

    assert paths == (
        tmp_path
        / "parquet/corporate_action_adjusted_stock_bars_v0_1/year=2025/part-phase5.parquet",
        tmp_path
        / "parquet/corporate_action_adjusted_stock_bars_v0_1/year=2026/part-phase5.parquet",
    )
    assert all(path.is_file() for path in paths)
    assert not list((tmp_path / "parquet").rglob("symbol=*"))
    assert not list((tmp_path / "parquet").rglob("security_id=*"))
    assert all("year" not in pq.ParquetFile(path).schema_arrow.names for path in paths)


def test_event_publication_uses_event_year_only(tmp_path) -> None:
    paths = publish_corporate_action_events(
        [event(event_date=date(2025, 12, 31)), event(event_date=date(2026, 1, 2))],
        data_root=tmp_path,
        batch_id="phase5",
    )

    assert paths == (
        tmp_path / "parquet/corporate_action_events_v0_1/year=2025/part-phase5.parquet",
        tmp_path / "parquet/corporate_action_events_v0_1/year=2026/part-phase5.parquet",
    )
    assert all(path.is_file() for path in paths)
    assert all("year" not in pq.ParquetFile(path).schema_arrow.names for path in paths)


def test_existing_adjusted_key_blocks_new_publication(tmp_path) -> None:
    publish_adjusted_stock_bars(
        [adjusted_bar()], data_root=tmp_path, batch_id="first"
    )

    with pytest.raises(PublishValidationError, match="overlaps"):
        publish_adjusted_stock_bars(
            [adjusted_bar()], data_root=tmp_path, batch_id="second"
        )
    assert not list(
        (tmp_path / "parquet").rglob("part-second.parquet")
    )


def test_existing_event_key_blocks_new_publication(tmp_path) -> None:
    publish_corporate_action_events([event()], data_root=tmp_path, batch_id="first")

    with pytest.raises(PublishValidationError, match="overlaps"):
        publish_corporate_action_events(
            [event()], data_root=tmp_path, batch_id="second"
        )
    assert not list((tmp_path / "parquet").rglob("part-second.parquet"))


def test_duplicate_clean_keys_are_rejected(tmp_path) -> None:
    with pytest.raises(CorporateActionDatasetValidationError, match="duplicate"):
        write_corporate_action_clean_batch(
            [adjusted_bar(), adjusted_bar()],
            [],
            data_root=tmp_path,
            batch_id="phase5",
        )


def test_multi_year_publication_failure_rolls_back_all_new_parts(
    tmp_path, monkeypatch
) -> None:
    real_link = storage_module.os.link
    calls = 0

    def fail_second_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second-year failure")
        real_link(source, target)

    monkeypatch.setattr(storage_module.os, "link", fail_second_link)

    with pytest.raises(PublishValidationError, match="simulated"):
        publish_adjusted_stock_bars(
            [
                adjusted_bar(trading_date=date(2025, 12, 31)),
                adjusted_bar(trading_date=date(2026, 1, 2)),
            ],
            data_root=tmp_path,
            batch_id="phase5",
        )

    assert not list((tmp_path / "parquet").rglob("part-phase5.parquet"))


@pytest.mark.parametrize("unsafe", ["../batch", "..", "a/b", "a\\b", "a.b"])
def test_unsafe_phase5_batch_ids_are_rejected(tmp_path, unsafe: str) -> None:
    with pytest.raises(ValueError, match="batch_id"):
        write_corporate_action_clean_batch(
            [adjusted_bar()], [], data_root=tmp_path, batch_id=unsafe
        )
