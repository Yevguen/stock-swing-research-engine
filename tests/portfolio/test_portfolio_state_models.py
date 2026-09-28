"""Focused Phase 13B.1 portfolio state-model tests."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from stock_swing_d1.portfolio import OpenPosition as Phase12OpenPosition
from stock_swing_d1.portfolio.portfolio_errors import (
    ArtifactHashMismatchError,
    DuplicateEventConflictError,
    InsufficientSettledCashError,
    InvalidExitQuantityError,
    OutOfOrderSessionError,
    OverdueSettlementError,
    PortfolioError,
    PortfolioPersistenceError,
    PortfolioStateError,
    PortfolioTransitionError,
    PositionAlreadyOpenError,
    PositionNotFoundError,
    SchemaVersionError,
    SettlementIntegrityError,
    StateHashMismatchError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    PortfolioEventKind,
    PortfolioEventReference,
    PortfolioLedgerEntry,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PendingSettlement,
    PortfolioBacktestResult,
    PortfolioSessionInput,
    PortfolioSessionSnapshot,
    PortfolioState,
    PortfolioTransitionResult,
)


def position_data(asset_id: str = "NORGATE:2") -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "quantity": 3,
        "entry_session": date(2026, 8, 17),
        "entry_price": Decimal("10.12345"),
        "entry_execution_id": f"BUY-{asset_id}",
        "entry_execution_cost": Decimal("0.00007"),
        "cost_basis": Decimal("30.37042"),
    }


def settlement(
    settlement_id: str,
    *,
    asset_id: str = "NORGATE:2",
    settlement_session: date = date(2026, 8, 20),
) -> PendingSettlement:
    return PendingSettlement(
        settlement_id=settlement_id,
        source_execution_id=f"SELL-{settlement_id}",
        asset_id=asset_id,
        amount=Decimal("25.55555"),
        trade_session=date(2026, 8, 18),
        settlement_session=settlement_session,
    )


def fingerprint(kind: PortfolioEventKind, event_id: str) -> AppliedEventFingerprint:
    return AppliedEventFingerprint(
        event_kind=kind,
        event_id=event_id,
        payload_sha256=("a" if kind is PortfolioEventKind.EXECUTION else "b") * 64,
    )


def test_valid_initial_portfolio_state() -> None:
    state = PortfolioState()

    assert state.schema_version == "portfolio_state.v0.1"
    assert state.base_currency == "USD"
    assert state.as_of_session is None
    assert state.state_version == 0
    assert state.settled_cash == Decimal("0")
    assert state.open_positions == ()
    assert state.pending_settlements == ()
    assert state.applied_events == ()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("settled_cash", Decimal("-0.01")), ("state_version", -1)],
)
def test_invalid_initial_state_numbers_are_rejected(
    field_name: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        PortfolioState(**{field_name: value})


@pytest.mark.parametrize("quantity", [0, -1])
def test_zero_or_negative_position_quantity_is_rejected(quantity: int) -> None:
    data = position_data()
    data["quantity"] = quantity

    with pytest.raises(ValidationError):
        OpenPosition(**data)


@pytest.mark.parametrize("entry_price", [Decimal("0"), Decimal("-1")])
def test_invalid_entry_price_is_rejected(entry_price: Decimal) -> None:
    data = position_data()
    data["entry_price"] = entry_price

    with pytest.raises(ValidationError):
        OpenPosition(**data)


def test_negative_entry_execution_cost_is_rejected() -> None:
    data = position_data()
    data["entry_execution_cost"] = Decimal("-0.01")

    with pytest.raises(ValidationError):
        OpenPosition(**data)


def test_incorrect_cost_basis_is_rejected() -> None:
    data = position_data()
    data["cost_basis"] = Decimal("30.37043")

    with pytest.raises(ValidationError, match="cost_basis must equal"):
        OpenPosition(**data)


@pytest.mark.parametrize(
    "settlement_session", [date(2026, 8, 18), date(2026, 8, 17)]
)
def test_invalid_settlement_chronology_is_rejected(
    settlement_session: date,
) -> None:
    with pytest.raises(ValidationError, match="must be after trade_session"):
        settlement("SETTLE-1", settlement_session=settlement_session)


def test_decimal_values_are_preserved_without_cent_rounding() -> None:
    position = OpenPosition(**position_data())
    pending = settlement("SETTLE-1")
    state = PortfolioState(
        settled_cash=Decimal("100.000009"),
        open_positions=(position,),
        pending_settlements=(pending,),
    )

    assert type(position.entry_price) is Decimal
    assert position.entry_price == Decimal("10.12345")
    assert position.entry_execution_cost == Decimal("0.00007")
    assert state.settled_cash == Decimal("100.000009")
    assert pending.amount == Decimal("25.55555")


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (PortfolioState, {"settled_cash": 1.1}),
        (OpenPosition, {**position_data(), "entry_price": 1.1}),
        (PendingSettlement, {"amount": 1.1}),
    ],
)
def test_binary_float_money_is_rejected(model, kwargs: dict[str, object]) -> None:
    if model is PendingSettlement:
        kwargs = {
            "settlement_id": "SETTLE-1",
            "source_execution_id": "SELL-1",
            "asset_id": "NORGATE:1",
            "trade_session": date(2026, 8, 18),
            "settlement_session": date(2026, 8, 20),
            **kwargs,
        }

    with pytest.raises(ValidationError, match="binary float"):
        model(**kwargs)


def test_canonical_models_are_immutable_and_collections_are_tuples() -> None:
    position = OpenPosition(**position_data())
    state = PortfolioState(open_positions=[position])
    session_input = PortfolioSessionInput(
        session=date(2026, 8, 20), execution_events=[]
    )

    assert isinstance(state.open_positions, tuple)
    assert isinstance(session_input.execution_events, tuple)
    with pytest.raises(ValidationError, match="frozen"):
        state.settled_cash = Decimal("1")
    with pytest.raises(ValidationError, match="frozen"):
        position.quantity = 4


def test_state_collections_have_deterministic_canonical_ordering() -> None:
    positions = (
        OpenPosition(**position_data("NORGATE:20")),
        OpenPosition(**position_data("NORGATE:10")),
    )
    settlements = (
        settlement(
            "SETTLE-B",
            asset_id="NORGATE:20",
            settlement_session=date(2026, 8, 21),
        ),
        settlement("SETTLE-C", settlement_session=date(2026, 8, 20)),
        settlement("SETTLE-A", settlement_session=date(2026, 8, 20)),
    )
    applied = (
        fingerprint(PortfolioEventKind.SETTLEMENT, "B"),
        fingerprint(PortfolioEventKind.EXECUTION, "B"),
        fingerprint(PortfolioEventKind.EXECUTION, "A"),
    )

    state = PortfolioState(
        open_positions=positions,
        pending_settlements=settlements,
        applied_events=applied,
    )
    reversed_state = PortfolioState(
        open_positions=tuple(reversed(positions)),
        pending_settlements=tuple(reversed(settlements)),
        applied_events=tuple(reversed(applied)),
    )

    assert state == reversed_state
    assert [item.asset_id for item in state.open_positions] == [
        "NORGATE:10",
        "NORGATE:20",
    ]
    assert [item.settlement_id for item in state.pending_settlements] == [
        "SETTLE-A",
        "SETTLE-C",
        "SETTLE-B",
    ]
    assert [(item.event_kind, item.event_id) for item in state.applied_events] == [
        (PortfolioEventKind.EXECUTION, "A"),
        (PortfolioEventKind.EXECUTION, "B"),
        (PortfolioEventKind.SETTLEMENT, "B"),
    ]


@pytest.mark.parametrize(
    "extra_field",
    [
        "ranking_score",
        "signal_strength",
        "requested_risk_pct",
        "SMA50",
        "earnings_state",
        "allocation_rank",
    ],
)
def test_strategy_and_allocation_metadata_is_forbidden(extra_field: str) -> None:
    data = position_data()
    data[extra_field] = "not allowed"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OpenPosition(**data)


def test_state_rejects_duplicate_logical_collection_keys() -> None:
    position = OpenPosition(**position_data())
    applied = fingerprint(PortfolioEventKind.EXECUTION, "EXEC-1")

    with pytest.raises(ValidationError, match="asset_ids must be unique"):
        PortfolioState(open_positions=(position, position))
    with pytest.raises(ValidationError, match="kind and ID pairs must be unique"):
        PortfolioState(applied_events=(applied, applied))


def test_state_rejects_duplicate_pending_settlement_ids() -> None:
    first = settlement("SETTLE-SAME", asset_id="NORGATE:1")
    second = settlement("SETTLE-SAME", asset_id="NORGATE:2")

    with pytest.raises(ValidationError, match="settlement_ids must be unique"):
        PortfolioState(pending_settlements=(first, second))


@pytest.mark.parametrize(
    ("model", "field_name", "data"),
    [
        (OpenPosition, "entry_execution_id", position_data()),
        (
            PendingSettlement,
            "settlement_id",
            {
                "settlement_id": "SETTLE-1",
                "source_execution_id": "SELL-1",
                "asset_id": "NORGATE:1",
                "amount": Decimal("1"),
                "trade_session": date(2026, 8, 18),
                "settlement_session": date(2026, 8, 20),
            },
        ),
        (
            PendingSettlement,
            "source_execution_id",
            {
                "settlement_id": "SETTLE-1",
                "source_execution_id": "SELL-1",
                "asset_id": "NORGATE:1",
                "amount": Decimal("1"),
                "trade_session": date(2026, 8, 18),
                "settlement_session": date(2026, 8, 20),
            },
        ),
    ],
)
@pytest.mark.parametrize("invalid_id", ["", "   ", "\t"])
def test_state_models_reject_empty_or_whitespace_identifiers(
    model, field_name: str, data: dict[str, object], invalid_id: str
) -> None:
    values = {**data, field_name: invalid_id}

    with pytest.raises(ValidationError):
        model(**values)


def test_phase12_open_position_public_behavior_is_preserved() -> None:
    assert Phase12OpenPosition is not OpenPosition
    phase12_position = Phase12OpenPosition(
        security_id="NORGATE:1",
        symbol="ONE",
        shares=1,
        entry_session=date(2026, 8, 17),
    )
    assert phase12_position.shares == 1


def test_phase13_error_hierarchy() -> None:
    assert issubclass(PortfolioStateError, PortfolioError)
    for error_type in (
        OutOfOrderSessionError,
        OverdueSettlementError,
        InsufficientSettledCashError,
        PositionAlreadyOpenError,
        PositionNotFoundError,
        InvalidExitQuantityError,
        DuplicateEventConflictError,
        SettlementIntegrityError,
    ):
        assert issubclass(error_type, PortfolioTransitionError)
        assert issubclass(error_type, PortfolioError)
    for error_type in (
        StateHashMismatchError,
        ArtifactHashMismatchError,
        SchemaVersionError,
    ):
        assert issubclass(error_type, PortfolioPersistenceError)
        assert issubclass(error_type, PortfolioError)


def test_transition_and_backtest_support_models_are_immutable_tuple_contracts() -> None:
    state = PortfolioState(
        as_of_session=date(2026, 8, 18),
        state_version=1,
        settled_cash=Decimal("900.123456"),
    )
    ledger = PortfolioLedgerEntry(
        session=date(2026, 8, 18),
        sequence_in_session=0,
        event_type=PortfolioLedgerEventType.BUY_APPLIED,
        source_event_id="EXEC-1",
        source_order_id="ORDER-1",
        asset_id="NORGATE:1",
        quantity_delta=1,
        fill_price=Decimal("99.876543"),
        execution_cost=Decimal("0"),
        settled_cash_delta=Decimal("-99.876543"),
        pending_cash_delta=Decimal("0"),
        settled_cash_after=Decimal("900.123456"),
        position_quantity_after=1,
        source_payload_sha256="a" * 64,
        state_hash_before="b" * 64,
        state_hash_after="c" * 64,
    )
    transition = PortfolioTransitionResult(
        session=date(2026, 8, 18),
        state_hash_before="b" * 64,
        state_hash_after="c" * 64,
        resulting_state=state,
        ledger_entries=[ledger],
        newly_applied_events=[
            PortfolioEventReference(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id="EXEC-1",
            )
        ],
        replayed_events=[],
    )
    snapshot = PortfolioSessionSnapshot(
        session=date(2026, 8, 18),
        state_version=1,
        settled_cash=Decimal("900.123456"),
        open_position_count=1,
        pending_settlement_count=0,
        state_hash="c" * 64,
    )
    result = PortfolioBacktestResult(
        initial_state=PortfolioState(settled_cash=Decimal("1000")),
        final_state=state,
        session_results=[transition],
        ledger_entries=[ledger],
        snapshots=[snapshot],
        initial_state_hash="b" * 64,
        final_state_hash="c" * 64,
    )

    assert ledger.schema_version == "portfolio_ledger_entry.v0.1"
    assert transition.schema_version == "portfolio_transition_result.v0.1"
    assert snapshot.schema_version == "portfolio_session_snapshot.v0.1"
    assert result.schema_version == "portfolio_backtest_result.v0.1"
    assert isinstance(transition.ledger_entries, tuple)
    assert isinstance(transition.newly_applied_events, tuple)
    assert isinstance(transition.replayed_events, tuple)
    assert isinstance(result.session_results, tuple)
    assert isinstance(result.ledger_entries, tuple)
    assert isinstance(result.snapshots, tuple)
    assert "newly_applied_event_ids" not in PortfolioTransitionResult.model_fields
    assert "replayed_event_ids" not in PortfolioTransitionResult.model_fields
    assert "newly_applied_events" in PortfolioTransitionResult.model_fields
    assert "replayed_events" in PortfolioTransitionResult.model_fields
    with pytest.raises(ValidationError, match="frozen"):
        result.final_state_hash = "d" * 64
