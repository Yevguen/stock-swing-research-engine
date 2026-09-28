from datetime import date

import pandas as pd
import pytest

from stock_swing_d1.models import CorporateActionAdjustedStockBar, StockBar
from stock_swing_d1.validation.corporate_action_parity import (
    CorporateActionParityError,
    check_corporate_action_parity,
    validate_corporate_action_parity,
)


def source_bar(
    *,
    security_id: str = "NORGATE:1",
    symbol: str = "AAA",
    trading_date: date = date(2026, 8, 7),
    close: float = 100.0,
) -> StockBar:
    return StockBar(
        security_id=security_id,
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


def adjusted_bar(
    *,
    security_id: str = "NORGATE:1",
    symbol: str = "AAA",
    trading_date: date = date(2026, 8, 7),
    close: float = 50.0,
    volume: float = 2_000_000.5,
) -> CorporateActionAdjustedStockBar:
    return CorporateActionAdjustedStockBar(
        security_id=security_id,
        symbol=symbol,
        trading_date=trading_date,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="capital_special_adjusted",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def raw_adjusted(
    *,
    asset_id: int = 1,
    trading_date: date = date(2026, 8, 7),
    unadjusted_close: float = 100.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider_asset_id": [asset_id],
            "trading_date": [trading_date],
            "Unadjusted Close": [unadjusted_close],
        }
    )


def test_perfect_parity_succeeds_with_expected_price_basis_difference() -> None:
    result = validate_corporate_action_parity(
        [source_bar()], [adjusted_bar()], raw_adjusted()
    )

    assert result.valid is True
    assert result.raw_adjusted_missing_key_count == 0
    assert result.raw_adjusted_extra_key_count == 0
    assert result.unadjusted_close_mismatch_count == 0
    assert result.minimum_adjustment_factor == 0.5
    assert result.maximum_adjustment_factor == 0.5


def test_missing_adjusted_observation_fails_without_inner_join_loss() -> None:
    result = check_corporate_action_parity([source_bar()], [], raw_adjusted())

    assert result.raw_adjusted_missing_key_count == 1
    assert result.raw_adjusted_extra_key_count == 0
    with pytest.raises(CorporateActionParityError):
        validate_corporate_action_parity([source_bar()], [], raw_adjusted())


def test_extra_adjusted_observation_fails_without_inner_join_loss() -> None:
    extra_date = date(2026, 8, 8)
    result = check_corporate_action_parity(
        [source_bar()],
        [adjusted_bar(), adjusted_bar(trading_date=extra_date)],
        pd.concat(
            [
                raw_adjusted(),
                raw_adjusted(trading_date=extra_date),
            ],
            ignore_index=True,
        ),
    )

    assert result.raw_adjusted_missing_key_count == 0
    assert result.raw_adjusted_extra_key_count == 1
    assert any("no Phase 4" in str(error["reason"]) for error in result.errors)


def test_symbol_metadata_must_match_same_source_observation() -> None:
    result = check_corporate_action_parity(
        [source_bar(symbol="OLD")], [adjusted_bar(symbol="NEW")], raw_adjusted()
    )

    assert result.valid is False
    assert any(error["reason"] == "metadata mismatch for symbol" for error in result.errors)


def test_ohlc_and_volume_differences_are_allowed() -> None:
    source = source_bar(close=100.0)
    adjusted = CorporateActionAdjustedStockBar(
        security_id="NORGATE:1",
        symbol="AAA",
        trading_date=date(2026, 8, 7),
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="capital_special_adjusted",
        open=40.0,
        high=60.0,
        low=30.0,
        close=50.0,
        volume=123.456,
    )

    assert validate_corporate_action_parity(
        [source], [adjusted], raw_adjusted()
    ).valid


def test_exact_unadjusted_close_match_succeeds() -> None:
    result = check_corporate_action_parity(
        [source_bar(close=101.0)],
        [adjusted_bar()],
        raw_adjusted(unadjusted_close=101.0),
    )

    assert result.unadjusted_close_mismatch_count == 0


def test_unadjusted_close_within_frozen_tolerance_succeeds() -> None:
    result = validate_corporate_action_parity(
        [source_bar(close=101.0)],
        [adjusted_bar()],
        raw_adjusted(unadjusted_close=101.0 + 1e-8),
    )

    assert result.unadjusted_close_mismatch_count == 0


def test_unadjusted_close_beyond_frozen_tolerance_fails() -> None:
    result = check_corporate_action_parity(
        [source_bar(close=101.0)],
        [adjusted_bar()],
        raw_adjusted(unadjusted_close=100.5),
    )

    assert result.unadjusted_close_mismatch_count == 1
    assert any("Unadjusted Close" in str(error["reason"]) for error in result.errors)


def test_adjustment_factor_must_remain_positive_and_finite() -> None:
    result = check_corporate_action_parity(
        [source_bar(close=1e308)],
        [adjusted_bar(close=5e-324)],
        raw_adjusted(unadjusted_close=1e308),
    )

    assert result.valid is False
    assert any("factor" in str(error["reason"]) for error in result.errors)


def test_adjustment_factor_diagnostics_do_not_infer_events() -> None:
    result = validate_corporate_action_parity(
        [source_bar()], [adjusted_bar(close=25.0)], raw_adjusted()
    )

    assert result.minimum_adjustment_factor == 0.25
    assert not any("event" in str(error).lower() for error in result.errors)
