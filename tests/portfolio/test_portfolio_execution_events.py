# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
"""Focused Phase 13B.1 execution-event model tests."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    ExecutionSide,
    PortfolioEventKind,
    PortfolioEventReference,
    PortfolioExecutionEvent,
    PortfolioLedgerEventType,
)


def execution_data(side: ExecutionSide = ExecutionSide.BUY) -> dict[str, object]:
    data: dict[str, object] = {
        "execution_id": "EXEC-1",
        "source_order_id": "ORDER-1",
        "session": date(2026, 8, 18),
        "asset_id": "NORGATE:1900000001",
        "side": side,
        "quantity": 7,
        "fill_price": Decimal("101.234567"),
        "execution_cost": Decimal("0.000031"),
    }
    if side is ExecutionSide.SELL:
        data.update(
            settlement_id="SETTLE-1",
            settlement_session=date(2026, 8, 20),
        )
    return data


def test_execution_and_ledger_enums_are_frozen_to_v01_values() -> None:
    # Extended by the frozen Ordinary Dividend Amendment v0.7 (OD-13.1):
    # PortfolioEventKind.DIVIDEND and PortfolioLedgerEventType.DIVIDEND_APPLIED.
    assert tuple(ExecutionSide) == (ExecutionSide.BUY, ExecutionSide.SELL)
    assert tuple(PortfolioEventKind) == (
        PortfolioEventKind.EXECUTION,
        PortfolioEventKind.SETTLEMENT,
        PortfolioEventKind.DIVIDEND,
    )
    assert tuple(PortfolioLedgerEventType) == (
        PortfolioLedgerEventType.BUY_APPLIED,
        PortfolioLedgerEventType.SELL_APPLIED,
        PortfolioLedgerEventType.SETTLEMENT_APPLIED,
        PortfolioLedgerEventType.DIVIDEND_APPLIED,
    )


def test_valid_buy_execution() -> None:
    event = PortfolioExecutionEvent(**execution_data())

    assert event.schema_version == "portfolio_execution_event.v0.1"
    assert event.side is ExecutionSide.BUY
    assert event.settlement_id is None
    assert event.settlement_session is None


def test_valid_sell_execution() -> None:
    event = PortfolioExecutionEvent(**execution_data(ExecutionSide.SELL))

    assert event.side is ExecutionSide.SELL
    assert event.settlement_id == "SETTLE-1"
    assert event.settlement_session == date(2026, 8, 20)


@pytest.mark.parametrize(
    "settlement_fields",
    [
        {"settlement_id": "SETTLE-1"},
        {"settlement_session": date(2026, 8, 20)},
        {
            "settlement_id": "SETTLE-1",
            "settlement_session": date(2026, 8, 20),
        },
    ],
)
def test_buy_containing_settlement_fields_is_rejected(
    settlement_fields: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="BUY executions must not contain"):
        PortfolioExecutionEvent(**execution_data(), **settlement_fields)


def test_sell_missing_settlement_id_is_rejected() -> None:
    data = execution_data(ExecutionSide.SELL)
    del data["settlement_id"]

    with pytest.raises(ValidationError, match="require settlement_id"):
        PortfolioExecutionEvent(**data)


def test_sell_missing_settlement_session_is_rejected() -> None:
    data = execution_data(ExecutionSide.SELL)
    del data["settlement_session"]

    with pytest.raises(ValidationError, match="require settlement_session"):
        PortfolioExecutionEvent(**data)


@pytest.mark.parametrize(
    "settlement_session", [date(2026, 8, 18), date(2026, 8, 17)]
)
def test_sell_settlement_must_follow_trade_session(
    settlement_session: date,
) -> None:
    data = execution_data(ExecutionSide.SELL)
    data["settlement_session"] = settlement_session

    with pytest.raises(ValidationError, match="after execution session"):
        PortfolioExecutionEvent(**data)


@pytest.mark.parametrize("quantity", [0, -1])
def test_invalid_execution_quantity_is_rejected(quantity: int) -> None:
    data = execution_data()
    data["quantity"] = quantity

    with pytest.raises(ValidationError):
        PortfolioExecutionEvent(**data)


@pytest.mark.parametrize("fill_price", [Decimal("0"), Decimal("-1")])
def test_invalid_fill_price_is_rejected(fill_price: Decimal) -> None:
    data = execution_data()
    data["fill_price"] = fill_price

    with pytest.raises(ValidationError):
        PortfolioExecutionEvent(**data)


def test_negative_execution_cost_is_rejected() -> None:
    data = execution_data()
    data["execution_cost"] = Decimal("-0.01")

    with pytest.raises(ValidationError):
        PortfolioExecutionEvent(**data)


def test_unsupported_execution_side_is_rejected() -> None:
    data = execution_data()
    data["side"] = "SHORT"

    with pytest.raises(ValidationError):
        PortfolioExecutionEvent(**data)


@pytest.mark.parametrize(
    "extra_field", ["settlement_amount", "ranking_score", "signal_strength"]
)
def test_unknown_extra_fields_are_rejected(extra_field: str) -> None:
    data = execution_data()
    data[extra_field] = Decimal("1")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PortfolioExecutionEvent(**data)


def test_execution_event_is_immutable_and_preserves_decimal_values() -> None:
    event = PortfolioExecutionEvent(**execution_data())

    assert type(event.fill_price) is Decimal
    assert event.fill_price == Decimal("101.234567")
    assert event.execution_cost == Decimal("0.000031")
    with pytest.raises(ValidationError, match="frozen"):
        event.quantity = 1


@pytest.mark.parametrize("field_name", ["fill_price", "execution_cost"])
def test_execution_event_rejects_binary_float_money(field_name: str) -> None:
    data = execution_data()
    data[field_name] = 1.1

    with pytest.raises(ValidationError, match="binary float"):
        PortfolioExecutionEvent(**data)


def test_applied_event_fingerprint_requires_lowercase_sha256() -> None:
    valid = AppliedEventFingerprint(
        event_kind=PortfolioEventKind.EXECUTION,
        event_id="EXEC-1",
        payload_sha256="a" * 64,
    )
    assert valid.payload_sha256 == "a" * 64

    for invalid_hash in ("a" * 63, "A" * 64, "g" * 64):
        with pytest.raises(ValidationError):
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id="EXEC-1",
                payload_sha256=invalid_hash,
            )


def test_canonical_asset_identity_is_stable_asset_id_not_ticker() -> None:
    data = execution_data()
    data["asset_id"] = "AAPL"

    with pytest.raises(ValidationError, match="String should match pattern"):
        PortfolioExecutionEvent(**data)


@pytest.mark.parametrize("field_name", ["execution_id", "source_order_id"])
@pytest.mark.parametrize("invalid_id", ["", "   ", "\t"])
def test_execution_rejects_empty_or_whitespace_identifiers(
    field_name: str, invalid_id: str
) -> None:
    data = execution_data()
    data[field_name] = invalid_id

    with pytest.raises(ValidationError):
        PortfolioExecutionEvent(**data)


@pytest.mark.parametrize("invalid_id", ["", "   ", "\t"])
def test_applied_fingerprint_rejects_empty_or_whitespace_event_id(
    invalid_id: str,
) -> None:
    with pytest.raises(ValidationError):
        AppliedEventFingerprint(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id=invalid_id,
            payload_sha256="a" * 64,
        )


def test_typed_event_references_distinguish_event_kind_namespaces() -> None:
    execution = PortfolioEventReference(
        event_kind=PortfolioEventKind.EXECUTION,
        event_id="X",
    )
    settlement = PortfolioEventReference(
        event_kind=PortfolioEventKind.SETTLEMENT,
        event_id="X",
    )

    assert execution != settlement
    assert (execution.event_kind, execution.event_id) == (
        PortfolioEventKind.EXECUTION,
        "X",
    )
    assert (settlement.event_kind, settlement.event_id) == (
        PortfolioEventKind.SETTLEMENT,
        "X",
    )


def test_typed_event_reference_is_frozen_nonempty_and_forbids_extras() -> None:
    reference = PortfolioEventReference(
        event_kind=PortfolioEventKind.EXECUTION,
        event_id="EXEC-1",
    )

    with pytest.raises(ValidationError, match="frozen"):
        reference.event_id = "EXEC-2"
    with pytest.raises(ValidationError):
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id="   ",
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id="EXEC-1",
            side=ExecutionSide.BUY,
        )
