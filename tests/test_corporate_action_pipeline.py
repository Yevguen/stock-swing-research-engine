import json
from datetime import date

import pandas as pd
import pyarrow.parquet as pq
import pytest

import stock_swing_d1.data.corporate_action_pipeline as pipeline_module
from stock_swing_d1.data.corporate_action_pipeline import (
    run_corporate_action_pipeline,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.storage.corporate_action_parquet import (
    verify_adjusted_clean_parquet,
    verify_event_clean_parquet,
)
from stock_swing_d1.storage.stock_bar_parquet import write_clean_batch


def source_bar(
    asset_id: int,
    symbol: str,
    trading_date: date,
    *,
    close: float = 100.0,
    security_id: str | None = None,
) -> StockBar:
    return StockBar(
        security_id=security_id or f"NORGATE:{asset_id}",
        symbol=symbol,
        trading_date=trading_date,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000_000,
    )


def write_source(tmp_path, bars: list[StockBar], batch_id: str = "phase4"):
    return write_clean_batch(bars, data_root=tmp_path, batch_id=batch_id)


def adjusted_frame_for_bars(
    bars: list[StockBar],
    *,
    unadjusted_offset: float = 0.0,
    invalid_open: bool = False,
    extra_date: date | None = None,
) -> pd.DataFrame:
    dates = [bar.trading_date for bar in bars]
    closes = [bar.close for bar in bars]
    if extra_date is not None:
        dates.append(extra_date)
        closes.append(100.0)
    adjusted_closes = [close / 2 for close in closes]
    opens = list(adjusted_closes)
    if invalid_open and opens:
        opens[0] = adjusted_closes[0] + 10
    return pd.DataFrame(
        {
            "Open": opens,
            "High": adjusted_closes,
            "Low": adjusted_closes,
            "Close": adjusted_closes,
            "Volume": [2_000_000.25] * len(dates),
            "Turnover": [None] * len(dates),
            "Unadjusted Close": [close + unadjusted_offset for close in closes],
            "Dividend": [0.25] * len(dates),
        },
        index=pd.to_datetime(dates),
    )


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        {"provider-specific-value": []}, index=pd.DatetimeIndex([])
    )


def event_frame(values: list[object], dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {"provider-specific-value": values}, index=pd.to_datetime(dates)
    )


def provider_functions(
    bars: list[StockBar],
    *,
    events_by_asset: dict[int, pd.DataFrame] | None = None,
    unadjusted_offset: float = 0.0,
    invalid_open: bool = False,
    extra_date: date | None = None,
):
    by_asset: dict[int, list[StockBar]] = {}
    symbols: dict[int, str] = {}
    for bar in bars:
        asset_id = int(bar.security_id.split(":", 1)[1])
        by_asset.setdefault(asset_id, []).append(bar)
        symbols[asset_id] = bar.symbol

    def adjusted_fetcher(asset_id: int, start: date, end: date) -> pd.DataFrame:
        selected = [
            bar
            for bar in by_asset[asset_id]
            if start <= bar.trading_date <= end
        ]
        return adjusted_frame_for_bars(
            selected,
            unadjusted_offset=unadjusted_offset,
            invalid_open=invalid_open,
            extra_date=extra_date,
        )

    def event_fetcher(asset_id: int) -> pd.DataFrame:
        return (events_by_asset or {}).get(asset_id, empty_events())

    def symbol_resolver(asset_id: int) -> str:
        return symbols[asset_id]

    return adjusted_fetcher, event_fetcher, symbol_resolver


def run_success_candidate(
    tmp_path,
    bars: list[StockBar],
    *,
    batch_id: str = "phase5",
    events_by_asset: dict[int, pd.DataFrame] | None = None,
):
    adjusted_fetcher, event_fetcher, symbol_resolver = provider_functions(
        bars, events_by_asset=events_by_asset
    )
    return run_corporate_action_pipeline(
        "phase4",
        data_root=tmp_path,
        phase5_batch_id=batch_id,
        adjusted_fetcher=adjusted_fetcher,
        event_fetcher=event_fetcher,
        symbol_resolver=symbol_resolver,
        norgatedata_package_version="test-version",
    )


def test_one_security_zero_event_pipeline_succeeds_end_to_end(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)

    result = run_success_candidate(tmp_path, bars)

    assert result.published is True
    assert result.raw_batch_directory == (
        tmp_path / "corporate_actions/raw/norgate/batch=phase5"
    )
    assert (result.raw_batch_directory / "adjusted_bars.parquet").is_file()
    assert (result.raw_batch_directory / "capital_event_flags.parquet").is_file()
    assert result.clean_adjusted_parquet_path.is_file()
    assert result.clean_event_parquet_path.is_file()
    assert len(result.adjusted_published_paths) == 1
    assert result.event_published_paths == ()
    assert verify_event_clean_parquet(result.clean_event_parquet_path) == []
    assert result.validation_report == {
        "phase5_batch_id": "phase5",
        "phase4_source_batch_id": "phase4",
        "phase4_row_count": 1,
        "adjusted_raw_row_count": 1,
        "adjusted_valid_row_count": 1,
        "adjusted_invalid_row_count": 0,
        "raw_event_flag_row_count": 0,
        "capital_event_count": 0,
        "event_invalid_count": 0,
        "raw_adjusted_missing_key_count": 0,
        "raw_adjusted_extra_key_count": 0,
        "unadjusted_close_mismatch_count": 0,
        "number_of_securities": 1,
        "minimum_trading_date": "2026-08-07",
        "maximum_trading_date": "2026-08-07",
        "minimum_adjustment_factor": 0.5,
        "maximum_adjustment_factor": 0.5,
        "adjusted_published": True,
        "events_validated": True,
        "published": True,
        "errors": [],
    }


def test_one_flag_event_pipeline_maps_and_publishes_event(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)

    result = run_success_candidate(
        tmp_path,
        bars,
        events_by_asset={1: event_frame([1], [date(2026, 8, 7)])},
    )

    assert result.published is True
    assert len(result.event_published_paths) == 1
    [event] = verify_event_clean_parquet(result.clean_event_parquet_path)
    assert event.event_type == "unknown_capital_event"
    assert event.date_semantics == "entitlement_close"
    assert event.terms_verified is False
    assert event.new_shares is None and event.old_shares is None
    assert result.validation_report["capital_event_count"] == 1


def test_multiple_security_scope_and_date_ranges_come_only_from_phase4(tmp_path) -> None:
    bars = [
        source_bar(2, "BBB", date(2026, 8, 6)),
        source_bar(2, "BBB", date(2026, 8, 8)),
        source_bar(1, "AAA", date(2026, 8, 7)),
    ]
    write_source(tmp_path, bars)
    adjusted_calls: list[tuple[int, date, date]] = []
    event_calls: list[int] = []
    adjusted_fetcher, base_event_fetcher, symbol_resolver = provider_functions(bars)

    def recording_adjusted(asset_id: int, start: date, end: date) -> pd.DataFrame:
        adjusted_calls.append((asset_id, start, end))
        return adjusted_fetcher(asset_id, start, end)

    def recording_event(asset_id: int) -> pd.DataFrame:
        event_calls.append(asset_id)
        return base_event_fetcher(asset_id)

    result = run_corporate_action_pipeline(
        "phase4",
        data_root=tmp_path,
        phase5_batch_id="phase5",
        adjusted_fetcher=recording_adjusted,
        event_fetcher=recording_event,
        symbol_resolver=symbol_resolver,
        norgatedata_package_version="test",
    )

    assert result.published is True
    assert adjusted_calls == [
        (1, date(2026, 8, 7), date(2026, 8, 7)),
        (2, date(2026, 8, 6), date(2026, 8, 8)),
    ]
    assert event_calls == [1, 2]
    manifest = json.loads(
        (result.raw_batch_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["requested_asset_ids"] == [1, 2]
    assert manifest["requested_date_ranges"] == [
        {
            "provider_asset_id": 1,
            "start_date": "2026-08-07",
            "end_date": "2026-08-07",
        },
        {
            "provider_asset_id": 2,
            "start_date": "2026-08-06",
            "end_date": "2026-08-08",
        },
    ]


def test_phase4_source_file_remains_byte_for_byte_unchanged(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    source_path = write_source(tmp_path, bars)
    before = source_path.read_bytes()

    result = run_success_candidate(tmp_path, bars)

    assert result.published is True
    assert source_path.read_bytes() == before


def test_adjusted_rows_have_exact_phase4_observation_keys(tmp_path) -> None:
    bars = [
        source_bar(1, "AAA", date(2026, 8, 6)),
        source_bar(1, "AAA", date(2026, 8, 7)),
    ]
    write_source(tmp_path, bars)

    result = run_success_candidate(tmp_path, bars)
    adjusted = verify_adjusted_clean_parquet(result.clean_adjusted_parquet_path)

    assert {(bar.security_id, bar.trading_date) for bar in adjusted} == {
        (bar.security_id, bar.trading_date) for bar in bars
    }


def test_missing_adjusted_provider_column_blocks_publication(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)
    adjusted_fetcher, event_fetcher, symbol_resolver = provider_functions(bars)

    result = run_corporate_action_pipeline(
        "phase4",
        data_root=tmp_path,
        phase5_batch_id="phase5",
        adjusted_fetcher=lambda asset_id, start, end: adjusted_fetcher(
            asset_id, start, end
        ).drop(columns="Dividend"),
        event_fetcher=event_fetcher,
        symbol_resolver=symbol_resolver,
    )

    assert result.published is False
    assert result.raw_batch_directory is None
    assert result.validation_report["errors"][0]["stage"] == (
        "adjusted_provider_validation"
    )
    assert not list((tmp_path / "parquet").rglob("*.parquet"))


def test_extra_adjusted_key_fails_parity_and_preserves_raw_snapshot(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)
    adjusted_fetcher, event_fetcher, symbol_resolver = provider_functions(
        bars, extra_date=date(2026, 8, 8)
    )

    result = run_corporate_action_pipeline(
        "phase4",
        data_root=tmp_path,
        phase5_batch_id="phase5",
        adjusted_fetcher=adjusted_fetcher,
        event_fetcher=event_fetcher,
        symbol_resolver=symbol_resolver,
    )

    assert result.published is False
    assert (result.raw_batch_directory / "adjusted_bars.parquet").is_file()
    assert result.validation_report["adjusted_valid_row_count"] == 2
    assert result.validation_report["adjusted_invalid_row_count"] == 0
    assert result.validation_report["raw_adjusted_extra_key_count"] == 1
    assert not list((tmp_path / "parquet").rglob("*.parquet"))


def test_unadjusted_close_mismatch_blocks_clean_and_publication(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)
    adjusted_fetcher, event_fetcher, symbol_resolver = provider_functions(
        bars, unadjusted_offset=1.0
    )

    result = run_corporate_action_pipeline(
        "phase4",
        data_root=tmp_path,
        phase5_batch_id="phase5",
        adjusted_fetcher=adjusted_fetcher,
        event_fetcher=event_fetcher,
        symbol_resolver=symbol_resolver,
    )

    assert result.published is False
    assert result.raw_batch_directory.is_dir()
    assert result.clean_adjusted_parquet_path is None
    assert result.validation_report["unadjusted_close_mismatch_count"] == 1


def test_malformed_event_response_blocks_publication(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)
    adjusted_fetcher, _, symbol_resolver = provider_functions(bars)

    result = run_corporate_action_pipeline(
        "phase4",
        data_root=tmp_path,
        phase5_batch_id="phase5",
        adjusted_fetcher=adjusted_fetcher,
        event_fetcher=lambda asset_id: pd.DataFrame(
            {"one": [0], "two": [1]}, index=pd.to_datetime(["2026-08-07"])
        ),
        symbol_resolver=symbol_resolver,
    )

    assert result.published is False
    assert result.validation_report["errors"][0]["stage"] == (
        "event_provider_validation"
    )


def test_flag_one_without_same_date_phase4_bar_fails_without_shifting(tmp_path) -> None:
    bars = [
        source_bar(1, "AAA", date(2026, 8, 7)),
        source_bar(1, "AAA", date(2026, 8, 10)),
    ]
    write_source(tmp_path, bars)

    result = run_success_candidate(
        tmp_path,
        bars,
        events_by_asset={1: event_frame([1], [date(2026, 8, 8)])},
    )

    assert result.published is False
    assert result.raw_batch_directory.is_dir()
    assert result.validation_report["event_invalid_count"] == 1
    assert result.validation_report["errors"][-1]["stage"] == (
        "event_phase4_crosscheck"
    )
    assert result.validation_report["errors"][-1]["event_date"] == "2026-08-08"


def test_adjusted_model_failure_preserves_raw_and_accurate_counts(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)
    adjusted_fetcher, event_fetcher, symbol_resolver = provider_functions(
        bars, invalid_open=True
    )

    result = run_corporate_action_pipeline(
        "phase4",
        data_root=tmp_path,
        phase5_batch_id="phase5",
        adjusted_fetcher=adjusted_fetcher,
        event_fetcher=event_fetcher,
        symbol_resolver=symbol_resolver,
    )

    report = result.validation_report
    assert result.published is False
    assert result.raw_batch_directory.is_dir()
    assert report["adjusted_raw_row_count"] == 1
    assert report["adjusted_valid_row_count"] == 0
    assert report["adjusted_invalid_row_count"] == 1
    assert report["raw_adjusted_missing_key_count"] == 1
    assert any(error["stage"] == "adjusted_model_validation" for error in report["errors"])


def test_clean_parquet_failure_retains_valid_counts_and_raw(tmp_path, monkeypatch) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)
    adjusted_fetcher, event_fetcher, symbol_resolver = provider_functions(bars)

    def fail_clean(*args, **kwargs):
        raise pipeline_module.ParquetValidationError("simulated clean failure")

    monkeypatch.setattr(pipeline_module, "write_corporate_action_clean_batch", fail_clean)
    result = run_corporate_action_pipeline(
        "phase4",
        data_root=tmp_path,
        phase5_batch_id="phase5",
        adjusted_fetcher=adjusted_fetcher,
        event_fetcher=event_fetcher,
        symbol_resolver=symbol_resolver,
    )

    assert result.published is False
    assert result.raw_batch_directory.is_dir()
    assert result.validation_report["adjusted_valid_row_count"] == 1
    assert result.validation_report["adjusted_invalid_row_count"] == 0
    assert result.validation_report["errors"] == [
        {"stage": "parquet_write", "reason": "simulated clean failure"}
    ]


def test_existing_publication_overlap_keeps_clean_but_reports_failure(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)
    first = run_success_candidate(tmp_path, bars, batch_id="first")
    second = run_success_candidate(tmp_path, bars, batch_id="second")

    assert first.published is True
    assert second.published is False
    assert second.clean_adjusted_parquet_path.is_file()
    assert second.clean_event_parquet_path.is_file()
    assert second.validation_report["adjusted_valid_row_count"] == 1
    assert second.validation_report["adjusted_invalid_row_count"] == 0
    assert second.validation_report["errors"][0]["stage"] == "publish_validation"
    assert not list((tmp_path / "parquet").rglob("part-second.parquet"))


def test_provider_runtime_exception_propagates_without_clean_result(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)

    class ProviderFailure(RuntimeError):
        pass

    with pytest.raises(ProviderFailure, match="NDU unavailable"):
        run_corporate_action_pipeline(
            "phase4",
            data_root=tmp_path,
            phase5_batch_id="phase5",
            adjusted_fetcher=lambda asset_id, start, end: (_ for _ in ()).throw(
                ProviderFailure("NDU unavailable")
            ),
            event_fetcher=lambda asset_id: empty_events(),
            symbol_resolver=lambda asset_id: "AAA",
        )

    assert not (tmp_path / "corporate_actions/raw/norgate/batch=phase5").exists()
    assert not (tmp_path / "corporate_actions/clean/batch=phase5").exists()


def test_missing_phase4_source_writes_failed_report_before_provider_calls(tmp_path) -> None:
    provider_called = False

    def adjusted_fetcher(asset_id: int, start: date, end: date) -> pd.DataFrame:
        nonlocal provider_called
        provider_called = True
        return pd.DataFrame()

    result = run_corporate_action_pipeline(
        "missing",
        data_root=tmp_path,
        phase5_batch_id="phase5",
        adjusted_fetcher=adjusted_fetcher,
        event_fetcher=lambda asset_id: empty_events(),
        symbol_resolver=lambda asset_id: "AAA",
    )

    assert result.published is False
    assert provider_called is False
    assert result.validation_report["errors"][0]["stage"] == (
        "phase4_source_validation"
    )


def test_malformed_phase4_security_id_fails_before_provider_calls(tmp_path) -> None:
    bars = [
        source_bar(
            1,
            "AAA",
            date(2026, 8, 7),
            security_id="NOT-NORGATE-1",
        )
    ]
    write_source(tmp_path, bars)
    provider_called = False

    def adjusted_fetcher(asset_id: int, start: date, end: date) -> pd.DataFrame:
        nonlocal provider_called
        provider_called = True
        return pd.DataFrame()

    result = run_corporate_action_pipeline(
        "phase4",
        data_root=tmp_path,
        phase5_batch_id="phase5",
        adjusted_fetcher=adjusted_fetcher,
        event_fetcher=lambda asset_id: empty_events(),
        symbol_resolver=lambda asset_id: "AAA",
    )

    assert result.published is False
    assert provider_called is False
    assert "malformed" in result.validation_report["errors"][0]["reason"]


def test_pipeline_does_not_emit_dividend_or_portfolio_fields(tmp_path) -> None:
    bars = [source_bar(1, "AAA", date(2026, 8, 7))]
    write_source(tmp_path, bars)

    result = run_success_candidate(tmp_path, bars)

    adjusted_schema = pq.ParquetFile(result.clean_adjusted_parquet_path).schema_arrow
    event_schema = pq.ParquetFile(result.clean_event_parquet_path).schema_arrow
    assert "Dividend" not in adjusted_schema.names
    assert "dividend" not in adjusted_schema.names
    assert "shares" not in adjusted_schema.names
    assert "cash" not in adjusted_schema.names
    assert "Dividend" not in event_schema.names
