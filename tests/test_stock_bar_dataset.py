from datetime import date

import pandas as pd
import pytest

from stock_swing_d1.models import StockBar
from stock_swing_d1.validation.stock_bar_dataset import (
    DatasetValidationError,
    validate_stock_bar_dataset,
)


def bar_data(
    *,
    security_id: str = "NORGATE:1",
    symbol: str = "AAA",
    trading_date: date = date(2026, 8, 7),
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "trading_date": trading_date,
        "timeframe": "D1",
        "session_type": "regular",
        "currency": "USD",
        "price_basis": "unadjusted",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1000,
    }


def test_valid_multi_security_dataset_is_sorted_deterministically() -> None:
    rows = [
        StockBar(**bar_data(security_id="NORGATE:2", symbol="BBB")),
        StockBar(
            **bar_data(
                security_id="NORGATE:1",
                trading_date=date(2026, 8, 8),
            )
        ),
        StockBar(**bar_data(security_id="NORGATE:1")),
    ]

    result = validate_stock_bar_dataset(rows)

    assert [(bar.security_id, bar.trading_date) for bar in result] == [
        ("NORGATE:1", date(2026, 8, 7)),
        ("NORGATE:1", date(2026, 8, 8)),
        ("NORGATE:2", date(2026, 8, 7)),
    ]


def test_exact_canonical_uniqueness_key_duplicate_is_rejected() -> None:
    row = bar_data()

    with pytest.raises(DatasetValidationError) as captured:
        validate_stock_bar_dataset([row, row.copy()])

    assert captured.value.duplicate_count == 1
    assert any(
        issue.reason == "duplicate canonical uniqueness key"
        for issue in captured.value.issues
    )


def test_duplicate_security_and_date_is_rejected_even_if_symbol_changes() -> None:
    first = bar_data(symbol="OLD")
    second = bar_data(symbol="NEW")

    with pytest.raises(DatasetValidationError) as captured:
        validate_stock_bar_dataset([first, second])

    assert any(
        issue.reason == "duplicate security_id and trading_date"
        for issue in captured.value.issues
    )


def test_null_canonical_field_is_rejected_at_dataset_stage() -> None:
    row = bar_data()
    row["symbol"] = None

    with pytest.raises(DatasetValidationError, match="must not be null"):
        validate_stock_bar_dataset(pd.DataFrame([row]))


def test_symbol_changes_for_one_security_id_are_allowed_across_dates() -> None:
    rows = [
        bar_data(symbol="OLD", trading_date=date(2026, 8, 7)),
        bar_data(symbol="NEW", trading_date=date(2026, 8, 8)),
    ]

    result = validate_stock_bar_dataset(rows)

    assert [bar.symbol for bar in result] == ["OLD", "NEW"]


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("timeframe", "H1"),
        ("session_type", "extended"),
        ("currency", "EUR"),
        ("price_basis", "adjusted"),
    ],
)
def test_frozen_dataset_values_are_enforced(field: str, invalid: str) -> None:
    row = bar_data()
    row[field] = invalid

    with pytest.raises(DatasetValidationError, match=field):
        validate_stock_bar_dataset([row])


def test_missing_canonical_field_is_not_silently_created() -> None:
    row = bar_data()
    del row["volume"]

    with pytest.raises(DatasetValidationError, match="missing canonical fields"):
        validate_stock_bar_dataset([row])


def test_extra_canonical_field_is_rejected() -> None:
    row = bar_data()
    row["adjusted_close"] = 101.0

    with pytest.raises(DatasetValidationError, match="unexpected canonical fields"):
        validate_stock_bar_dataset([row])
