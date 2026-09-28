# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
import json
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stock_swing_d1.data.norgate_d1_adapter import prepare_norgate_raw_frame
from stock_swing_d1.models import StockBar
from stock_swing_d1.storage.stock_bar_parquet import (
    CANONICAL_ARROW_SCHEMA,
    BatchExistsError,
    PublishValidationError,
    publish_stock_bars,
    verify_canonical_parquet,
    write_clean_batch,
    write_raw_batch,
)


def provider_frame(trading_date: str = "2026-08-07") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1_000_000.0],
            "Turnover": [None],
            "Unadjusted Close": [101.0],
            "Dividend": [None],
        },
        index=pd.to_datetime([trading_date]),
    )


def raw_frame() -> pd.DataFrame:
    return prepare_norgate_raw_frame(1_900_000_001, "SYNTHA", provider_frame())


def manifest(batch_id: str = "20260811T131420Z") -> dict[str, object]:
    return {
        "provider": "Norgate Data",
        "package": "US Stocks Platinum",
        "batch_id": batch_id,
        "extracted_at_utc": "2026-08-11T13:14:20Z",
        "requested_asset_ids": [1_900_000_001],
        "requested_start_date": "2026-08-07",
        "requested_end_date": "2026-08-07",
        "adjustment_mode": "NONE",
        "padding_mode": "NONE",
        "norgatedata_python_package_version": "1.2.3",
        "raw_row_count": 1,
    }


def bar(
    *,
    trading_date: date = date(2026, 8, 7),
    security_id: str = "NORGATE:1900000001",
    symbol: str = "SYNTHA",
) -> StockBar:
    return StockBar(
        security_id=security_id,
        symbol=symbol,
        trading_date=trading_date,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=1_000_000,
    )


def test_raw_parquet_and_human_readable_manifest_are_written(tmp_path) -> None:
    batch_id = "20260811T131420Z"

    directory = write_raw_batch(
        raw_frame(), manifest(), data_root=tmp_path, batch_id=batch_id
    )

    parquet_path = directory / "bars.parquet"
    manifest_path = directory / "manifest.json"
    assert parquet_path.is_file()
    assert manifest_path.is_file()
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["requested_asset_ids"] == [1_900_000_001]
    assert stored["adjustment_mode"] == "NONE"
    assert stored["padding_mode"] == "NONE"
    assert manifest_path.read_bytes().endswith(b"\n")
    schema = pq.ParquetFile(parquet_path).schema_arrow
    assert schema.field("provider_asset_id").type == pa.int64()
    assert schema.field("provider_symbol").type == pa.string()
    assert schema.field("trading_date").type == pa.date32()


def test_raw_batch_overwrite_is_rejected(tmp_path) -> None:
    write_raw_batch(
        raw_frame(), manifest("batch1"), data_root=tmp_path, batch_id="batch1"
    )

    with pytest.raises(BatchExistsError):
        write_raw_batch(
            raw_frame(), manifest("batch1"), data_root=tmp_path, batch_id="batch1"
        )


def test_clean_arrow_schema_round_trip_and_compression(tmp_path) -> None:
    rows = [bar()]

    path = write_clean_batch(
        rows, data_root=tmp_path, batch_id="20260811T131420Z"
    )

    parquet_file = pq.ParquetFile(path)
    assert parquet_file.schema_arrow.equals(
        CANONICAL_ARROW_SCHEMA, check_metadata=False
    )
    assert parquet_file.schema_arrow.field("trading_date").type == pa.date32()
    assert parquet_file.schema_arrow.field("volume").type == pa.int64()
    assert "__index_level_0__" not in parquet_file.schema_arrow.names
    assert all(
        parquet_file.metadata.row_group(row_group).column(column).compression
        == "ZSTD"
        for row_group in range(parquet_file.metadata.num_row_groups)
        for column in range(
            parquet_file.metadata.row_group(row_group).num_columns
        )
    )
    assert verify_canonical_parquet(path, expected_row_count=1) == rows


def test_clean_round_trip_preserves_sorted_values_and_row_count(tmp_path) -> None:
    rows = [
        bar(security_id="NORGATE:2", symbol="BBB"),
        bar(security_id="NORGATE:1", symbol="AAA"),
    ]

    path = write_clean_batch(rows, data_root=tmp_path, batch_id="batch1")
    result = verify_canonical_parquet(path, expected_row_count=2)

    assert [item.security_id for item in result] == ["NORGATE:1", "NORGATE:2"]
    assert result[0].model_dump() == rows[1].model_dump()


def test_publication_uses_year_only_partitions_and_batch_part_names(tmp_path) -> None:
    rows = [
        bar(trading_date=date(2025, 12, 31)),
        bar(trading_date=date(2026, 1, 2)),
    ]

    paths = publish_stock_bars(rows, data_root=tmp_path, batch_id="batch1")

    assert paths == (
        tmp_path / "parquet/stock_bars_v0_1/year=2025/part-batch1.parquet",
        tmp_path / "parquet/stock_bars_v0_1/year=2026/part-batch1.parquet",
    )
    assert all(path.is_file() for path in paths)
    assert not list((tmp_path / "parquet").rglob("symbol=*"))
    assert not list((tmp_path / "parquet").rglob("security_id=*"))
    for path in paths:
        assert pq.ParquetFile(path).schema_arrow.names == list(
            CANONICAL_ARROW_SCHEMA.names
        )
        assert "year" not in pq.ParquetFile(path).schema_arrow.names


def test_existing_published_key_blocks_entire_new_publication(tmp_path) -> None:
    publish_stock_bars([bar()], data_root=tmp_path, batch_id="batch1")

    with pytest.raises(PublishValidationError, match="overlaps"):
        publish_stock_bars([bar()], data_root=tmp_path, batch_id="batch2")

    assert not list(
        (tmp_path / "parquet/stock_bars_v0_1").rglob(
            "part-batch2.parquet"
        )
    )


@pytest.mark.parametrize("unsafe", ["../batch", "..", "a/b", "a\\b", "a.b"])
def test_unsafe_batch_ids_are_rejected(tmp_path, unsafe: str) -> None:
    with pytest.raises(ValueError, match="batch_id"):
        write_clean_batch([bar()], data_root=tmp_path, batch_id=unsafe)
