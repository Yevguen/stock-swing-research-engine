"""Deterministic execution and settlement identifier tests."""

from datetime import date, datetime
from inspect import signature

import pytest

from stock_swing_d1.execution.costs import (
    ExecutionCostValidationError,
    ExecutionIdentifierService,
)


def test_exact_canonical_identifier_forms() -> None:
    service = ExecutionIdentifierService()
    security_id = "NORGATE:123"
    entry_id = service.buy_execution_id(
        entry_session=date(2024, 1, 3), security_id=security_id
    )
    sell_id = service.sell_execution_id(
        entry_execution_id=entry_id,
        exit_session=date(2024, 1, 12),
        security_id=security_id,
    )
    assert service.buy_source_order_id(
        allocation_session=date(2024, 1, 2), security_id=security_id
    ) == "ALLOC:2024-01-02:NORGATE:123"
    assert entry_id == "ENTRY:2024-01-03:NORGATE:123"
    assert service.sell_source_order_id(
        entry_execution_id=entry_id,
        exit_session=date(2024, 1, 12),
        security_id=security_id,
    ) == "EXIT-ORDER:ENTRY:2024-01-03:NORGATE:123:2024-01-12:NORGATE:123"
    assert sell_id == (
        "EXIT:ENTRY:2024-01-03:NORGATE:123:2024-01-12:NORGATE:123"
    )
    assert service.settlement_id(sell_execution_id=sell_id) == (
        "SETTLE:EXIT:ENTRY:2024-01-03:NORGATE:123:2024-01-12:NORGATE:123"
    )


def test_identifiers_repeat_deterministically() -> None:
    service = ExecutionIdentifierService()
    arguments = {
        "entry_execution_id": "ENTRY:2024-01-03:NORGATE:123",
        "exit_session": date(2024, 1, 12),
        "security_id": "NORGATE:123",
    }
    assert service.sell_execution_id(**arguments) == service.sell_execution_id(
        **arguments
    )


def test_identifier_api_excludes_economic_inputs() -> None:
    forbidden = {
        "fill_price",
        "execution_cost",
        "spread",
        "commission",
        "policy_fingerprint",
    }
    for method_name in (
        "buy_source_order_id",
        "buy_execution_id",
        "sell_source_order_id",
        "sell_execution_id",
        "settlement_id",
    ):
        assert forbidden.isdisjoint(
            signature(getattr(ExecutionIdentifierService, method_name)).parameters
        )


@pytest.mark.parametrize(
    "security_id",
    ["", " NORGATE:1", "NORGATE:1 ", "NORGATE:0", "NORGATE:-1", "OTHER:1", 1],
)
def test_invalid_security_ids_fail_closed(security_id: object) -> None:
    with pytest.raises(ExecutionCostValidationError) as raised:
        ExecutionIdentifierService().buy_execution_id(
            entry_session=date(2024, 1, 3),
            security_id=security_id,  # type: ignore[arg-type]
        )
    assert raised.value.code == "INVALID_SECURITY_ID"


def test_datetime_is_not_inferred_to_a_session_date() -> None:
    with pytest.raises(ExecutionCostValidationError) as raised:
        ExecutionIdentifierService().buy_execution_id(
            entry_session=datetime(2024, 1, 3),  # type: ignore[arg-type]
            security_id="NORGATE:1",
        )
    assert raised.value.code == "INVALID_IDENTIFIER_SESSION"


def test_nested_identifier_security_mismatch_fails_closed() -> None:
    with pytest.raises(ExecutionCostValidationError) as raised:
        ExecutionIdentifierService().sell_execution_id(
            entry_execution_id="ENTRY:2024-01-03:NORGATE:1",
            exit_session=date(2024, 1, 12),
            security_id="NORGATE:2",
        )
    assert raised.value.code == "INVALID_ENTRY_EXECUTION_ID"
