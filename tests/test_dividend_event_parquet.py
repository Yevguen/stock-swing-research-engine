from datetime import date
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import stock_swing_d1.storage.dividend_event_parquet as storage_module
from stock_swing_d1.models import DividendEvent
from stock_swing_d1.storage.dividend_event_parquet import (
    DIVIDEND_EVENT_ARROW_SCHEMA,
    publish_dividend_events,
    verify_dividend_event_parquet,
    write_dividend_clean_batch,
)
from stock_swing_d1.storage.stock_bar_parquet import (
    BatchExistsError,
    PublishValidationError,
)


def event(
    *,
    asset_id: int = 1,
    entitlement_date: date = date(2025, 12, 31),
    amount: float = 0.25,
) -> DividendEvent:
    return DividendEvent(
        security_id=f"NORGATE:{asset_id}",
        symbol=f"S{asset_id}",
        entitlement_date=entitlement_date,
        date_semantics="entitlement_close",
        dividend_type="ordinary_cash",
        amount_per_share=amount,
        currency="USD",
        source_provider="Norgate Data",
        source_asset_id=asset_id,
        source_adjustment_mode="CAPITALSPECIAL",
    )


def manifest(batch_id: str, *, row_count: int, event_count: int) -> dict[str, object]:
    return {
        "dividend_batch_id": batch_id,
        "phase5_source_batch_id": "phase5",
        "phase5_source_raw_path": "data/corporate_actions/raw/norgate/batch=phase5/adjusted_bars.parquet",
        "phase5_source_adjusted_clean_path": "data/corporate_actions/clean/batch=phase5/adjusted_stock_bars.parquet",
        "phase4_source_batch_id": "phase4",
        "phase4_source_path": "data/clean/d1/batch=phase4/stock_bars.parquet",
        "derived_at_utc": "2026-08-12T10:00:00Z",
        "provider": "Norgate Data",
        "source_adjustment_mode": "CAPITALSPECIAL",
        "source_raw_row_count": row_count,
        "source_dividend_nonzero_count": event_count,
    }


def assert_zstd(path) -> None:
    metadata = pq.ParquetFile(path).metadata
    for group_index in range(metadata.num_row_groups):
        group = metadata.row_group(group_index)
        for column_index in range(group.num_columns):
            assert group.column(column_index).compression == "ZSTD"


def test_clean_layout_exact_schema_zstd_roundtrip_and_manifest(tmp_path) -> None:
    rows = [
        event(asset_id=2, entitlement_date=date(2026, 1, 2)),
        event(asset_id=1, entitlement_date=date(2025, 12, 31)),
    ]
    paths = write_dividend_clean_batch(
        rows,
        manifest("div1", row_count=3, event_count=2),
        data_root=tmp_path,
        batch_id="div1",
        source_raw_row_count=3,
    )

    assert paths.dividend_events == (
        tmp_path / "dividends/clean/batch=div1/dividend_events.parquet"
    )
    assert paths.manifest.name == "manifest.json"
    assert paths.manifest.read_bytes().endswith(b"\n")
    assert json.loads(paths.manifest.read_text(encoding="utf-8"))[
        "phase5_source_batch_id"
    ] == "phase5"
    table = pq.ParquetFile(paths.dividend_events).read()
    assert table.schema.equals(DIVIDEND_EVENT_ARROW_SCHEMA, check_metadata=False)
    assert table.schema == pa.schema(
        [
            pa.field("security_id", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("entitlement_date", pa.date32(), nullable=False),
            pa.field("date_semantics", pa.string(), nullable=False),
            pa.field("dividend_type", pa.string(), nullable=False),
            pa.field("amount_per_share", pa.float64(), nullable=False),
            pa.field("currency", pa.string(), nullable=False),
            pa.field("source_provider", pa.string(), nullable=False),
            pa.field("source_asset_id", pa.int64(), nullable=False),
            pa.field("source_adjustment_mode", pa.string(), nullable=False),
        ]
    )
    assert "__index_level_0__" not in table.column_names
    assert table["security_id"].to_pylist() == ["NORGATE:1", "NORGATE:2"]
    assert_zstd(paths.dividend_events)
    assert verify_dividend_event_parquet(paths.dividend_events) == [rows[1], rows[0]]


def test_zero_event_clean_batch_has_exact_schema_and_no_publication(tmp_path) -> None:
    paths = write_dividend_clean_batch(
        [],
        manifest("empty", row_count=2, event_count=0),
        data_root=tmp_path,
        batch_id="empty",
        source_raw_row_count=2,
    )

    table = pq.ParquetFile(paths.dividend_events).read()
    assert table.num_rows == 0
    assert table.schema.equals(DIVIDEND_EVENT_ARROW_SCHEMA, check_metadata=False)
    assert verify_dividend_event_parquet(paths.dividend_events) == []
    assert publish_dividend_events([], data_root=tmp_path, batch_id="empty") == ()
    assert not (tmp_path / "parquet/dividend_events_v0_1").exists()


def test_publication_is_year_only_and_uses_batch_part_names(tmp_path) -> None:
    paths = publish_dividend_events(
        [
            event(asset_id=1, entitlement_date=date(2025, 12, 31)),
            event(asset_id=2, entitlement_date=date(2026, 1, 2)),
        ],
        data_root=tmp_path,
        batch_id="div1",
    )

    assert paths == (
        tmp_path / "parquet/dividend_events_v0_1/year=2025/part-div1.parquet",
        tmp_path / "parquet/dividend_events_v0_1/year=2026/part-div1.parquet",
    )
    for path in paths:
        assert path.is_file()
        table = pq.ParquetFile(path).read()
        assert "year" not in table.column_names
        assert table.schema.equals(DIVIDEND_EVENT_ARROW_SCHEMA, check_metadata=False)
        assert_zstd(path)


def test_existing_published_key_blocks_without_modifying_old_part(tmp_path) -> None:
    original_path = publish_dividend_events(
        [event()], data_root=tmp_path, batch_id="old"
    )[0]
    original_bytes = original_path.read_bytes()

    with pytest.raises(PublishValidationError) as raised:
        publish_dividend_events(
            [event(amount=0.5)], data_root=tmp_path, batch_id="new"
        )

    assert raised.value.duplicate_count == 1
    assert original_path.read_bytes() == original_bytes
    assert not original_path.with_name("part-new.parquet").exists()


def test_existing_derived_batch_directory_is_immutable(tmp_path) -> None:
    write_dividend_clean_batch(
        [event()],
        manifest("div1", row_count=1, event_count=1),
        data_root=tmp_path,
        batch_id="div1",
        source_raw_row_count=1,
    )

    with pytest.raises(BatchExistsError):
        write_dividend_clean_batch(
            [event()],
            manifest("div1", row_count=1, event_count=1),
            data_root=tmp_path,
            batch_id="div1",
            source_raw_row_count=1,
        )


def test_multi_year_link_failure_rolls_back_every_new_part(tmp_path, monkeypatch) -> None:
    real_link = storage_module.os.link
    calls = 0

    def fail_second_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-link failure")
        return real_link(source, target)

    monkeypatch.setattr(storage_module.os, "link", fail_second_link)
    with pytest.raises(PublishValidationError, match="injected second-link failure"):
        publish_dividend_events(
            [
                event(asset_id=1, entitlement_date=date(2025, 12, 31)),
                event(asset_id=2, entitlement_date=date(2026, 1, 2)),
            ],
            data_root=tmp_path,
            batch_id="rollback",
        )

    assert list(
        (tmp_path / "parquet/dividend_events_v0_1").glob("**/*.parquet")
    ) == []
