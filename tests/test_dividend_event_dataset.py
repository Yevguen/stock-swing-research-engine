from datetime import date
from math import inf, nan

import pyarrow as pa
import pytest

from stock_swing_d1.models import (
    CorporateActionAdjustedStockBar,
    DividendEvent,
    StockBar,
)
from stock_swing_d1.storage.corporate_action_parquet import (
    ADJUSTED_RAW_ARROW_SCHEMA,
)
from stock_swing_d1.validation.dividend_event_dataset import (
    DividendEventDatasetValidationError,
    check_dividend_event_crosschecks,
    check_dividend_source_rows,
    validate_dividend_event_dataset,
)


def raw_row(
    dividend: object,
    *,
    asset_id: object = 1,
    symbol: object = "aaa",
    trading_date: object = date(2025, 12, 31),
) -> dict[str, object]:
    return {
        "provider_asset_id": asset_id,
        "provider_symbol": symbol,
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


def event(
    *,
    asset_id: int = 1,
    symbol: str = "AAA",
    entitlement_date: date = date(2025, 12, 31),
    amount: float = 0.25,
) -> DividendEvent:
    return DividendEvent(
        security_id=f"NORGATE:{asset_id}",
        symbol=symbol,
        entitlement_date=entitlement_date,
        date_semantics="entitlement_close",
        dividend_type="ordinary_cash",
        amount_per_share=amount,
        currency="USD",
        source_provider="Norgate Data",
        source_asset_id=asset_id,
        source_adjustment_mode="CAPITALSPECIAL",
    )


def adjusted_bar(
    *,
    asset_id: int = 1,
    symbol: str = "AAA",
    trading_date: date = date(2025, 12, 31),
) -> CorporateActionAdjustedStockBar:
    return CorporateActionAdjustedStockBar(
        security_id=f"NORGATE:{asset_id}",
        symbol=symbol,
        trading_date=trading_date,
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


def phase4_bar(
    *,
    asset_id: int = 1,
    symbol: str = "AAA",
    trading_date: date = date(2025, 12, 31),
) -> StockBar:
    return StockBar(
        security_id=f"NORGATE:{asset_id}",
        symbol=symbol,
        trading_date=trading_date,
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


def test_positive_raw_dividend_maps_exactly_through_model_without_rounding() -> None:
    amount = 0.123456789012345
    result = check_dividend_source_rows([raw_row(amount)])

    assert result.valid
    assert result.raw_dividend_nonzero_count == 1
    assert result.dividend_invalid_count == 0
    assert result.events == (event(amount=amount),)


def test_arrow_null_is_valid_but_nonnull_nan_is_invalid() -> None:
    table = pa.Table.from_pylist(
        [raw_row(None), raw_row(nan, trading_date=date(2026, 1, 2))],
        schema=ADJUSTED_RAW_ARROW_SCHEMA,
    )
    assert table["Dividend"].null_count == 1
    assert table["Dividend"].is_valid().to_pylist() == [False, True]

    result = check_dividend_source_rows(table)

    assert not result.valid
    assert result.source_raw_row_count == 2
    assert result.raw_dividend_nonzero_count == 0
    assert result.dividend_invalid_count == 1
    assert "finite" in result.errors[0]["reason"]


@pytest.mark.parametrize("value", [0, 0.0, None])
def test_zero_and_null_create_no_event(value: object) -> None:
    result = check_dividend_source_rows([raw_row(value)])

    assert result.valid
    assert result.events == ()
    assert result.raw_dividend_nonzero_count == 0


@pytest.mark.parametrize("value", [-0.01, inf, -inf, nan, "0.25", True, False, object()])
def test_invalid_dividend_scalars_fail_closed(value: object) -> None:
    result = check_dividend_source_rows([raw_row(value)])

    assert not result.valid
    assert result.dividend_invalid_count == 1
    assert result.events == ()
    assert result.errors[0]["stage"] == "dividend_source_validation"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_asset_id", 0),
        ("provider_asset_id", True),
        ("provider_asset_id", "1"),
        ("provider_symbol", ""),
        ("trading_date", "2025-12-31"),
    ],
)
def test_positive_dividend_requires_valid_identity_symbol_and_date(
    field: str, value: object
) -> None:
    record = raw_row(0.25)
    record[field] = value

    result = check_dividend_source_rows([record])

    assert not result.valid
    assert result.raw_dividend_nonzero_count == 1
    assert result.dividend_invalid_count == 1
    assert result.errors[0]["stage"] == "dividend_model_validation"


def test_multiple_events_are_sorted_by_security_then_entitlement_date() -> None:
    events = validate_dividend_event_dataset(
        [
            event(asset_id=2, symbol="BBB", entitlement_date=date(2025, 1, 2)),
            event(asset_id=1, entitlement_date=date(2026, 1, 2)),
            event(asset_id=1, entitlement_date=date(2025, 1, 2)),
        ]
    )

    assert [(row.security_id, row.entitlement_date) for row in events] == [
        ("NORGATE:1", date(2025, 1, 2)),
        ("NORGATE:1", date(2026, 1, 2)),
        ("NORGATE:2", date(2025, 1, 2)),
    ]


def test_duplicate_canonical_dividend_key_is_not_aggregated_or_deduplicated() -> None:
    with pytest.raises(DividendEventDatasetValidationError) as raised:
        validate_dividend_event_dataset([event(amount=0.25), event(amount=0.5)])

    assert raised.value.duplicate_count == 1
    assert raised.value.errors[0]["stage"] == "dividend_uniqueness"


def test_exact_adjusted_and_phase4_crosschecks_succeed() -> None:
    result = check_dividend_event_crosschecks(
        [event()], [adjusted_bar()], [phase4_bar()]
    )

    assert result.valid
    assert result.dividend_missing_adjusted_bar_count == 0
    assert result.dividend_missing_phase4_bar_count == 0


@pytest.mark.parametrize(
    ("adjusted", "phase4", "stage", "missing_adjusted", "missing_phase4"),
    [
        ([], [phase4_bar()], "adjusted_bar_crosscheck", 1, 0),
        ([adjusted_bar(symbol="ZZZ")], [phase4_bar()], "adjusted_bar_crosscheck", 0, 0),
        ([adjusted_bar()], [], "phase4_bar_crosscheck", 0, 1),
        ([adjusted_bar()], [phase4_bar(symbol="ZZZ")], "phase4_bar_crosscheck", 0, 0),
    ],
)
def test_missing_and_symbol_mismatch_crosschecks_fail(
    adjusted,
    phase4,
    stage: str,
    missing_adjusted: int,
    missing_phase4: int,
) -> None:
    result = check_dividend_event_crosschecks([event()], adjusted, phase4)

    assert not result.valid
    assert result.errors[0]["stage"] == stage
    assert result.dividend_missing_adjusted_bar_count == missing_adjusted
    assert result.dividend_missing_phase4_bar_count == missing_phase4
