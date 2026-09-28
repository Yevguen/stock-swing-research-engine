"""Cross-object invariant and canonical hashing tests for Phase 13B.2."""

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.portfolio.portfolio_errors import PortfolioStateError
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    PortfolioEventKind,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import (
    canonical_payload_bytes,
    hash_execution_event,
    hash_portfolio_state,
)
from stock_swing_d1.portfolio.portfolio_invariants import (
    PortfolioInvariantChecker,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PendingSettlement,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import (
    buy_event,
    initial_state,
    sell_event,
    state_with_position,
)


@pytest.mark.parametrize(
    "state",
    [
        PortfolioState(),
        PortfolioState(as_of_session=date(2026, 8, 20), state_version=1),
        PortfolioState(as_of_session=date(2026, 8, 20), state_version=7),
    ],
)
def test_valid_initial_and_processed_state_version_relationships(
    state: PortfolioState,
) -> None:
    PortfolioInvariantChecker.validate_state(state)


@pytest.mark.parametrize(
    "state",
    [
        PortfolioState(state_version=3),
        PortfolioState(as_of_session=date(2026, 8, 20), state_version=0),
    ],
)
def test_impossible_state_version_relationship_is_rejected(
    state: PortfolioState,
) -> None:
    with pytest.raises(PortfolioStateError, match="if and only if"):
        PortfolioInvariantChecker.validate_state(state)


def test_open_position_must_reference_applied_execution_fingerprint() -> None:
    position = OpenPosition(
        asset_id="NORGATE:1",
        quantity=1,
        entry_session=date(2026, 8, 20),
        entry_price=Decimal("10"),
        entry_execution_id="BUY-1",
        entry_execution_cost=Decimal("0"),
        cost_basis=Decimal("10"),
    )
    state = PortfolioState(
        as_of_session=date(2026, 8, 20),
        state_version=1,
        open_positions=(position,),
    )

    with pytest.raises(
        PortfolioStateError,
        match="reference an applied EXECUTION fingerprint",
    ):
        PortfolioInvariantChecker.validate_state(state)


def test_standalone_state_validation_is_referential_not_side_aware() -> None:
    execution_id = "EXEC-REFERENCE"
    position = OpenPosition(
        asset_id="NORGATE:1",
        quantity=1,
        entry_session=date(2026, 8, 20),
        entry_price=Decimal("10"),
        entry_execution_id=execution_id,
        entry_execution_cost=Decimal("0"),
        cost_basis=Decimal("10"),
    )
    sell_payload = sell_event(
        execution_id,
        session=date(2026, 8, 20),
        quantity=1,
        settlement_session=date(2026, 8, 22),
    )
    state = PortfolioState(
        as_of_session=date(2026, 8, 20),
        state_version=1,
        open_positions=(position,),
        applied_events=(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id=execution_id,
                payload_sha256=hash_execution_event(sell_payload),
            ),
        ),
    )

    PortfolioInvariantChecker.validate_state(state)


def test_funded_virgin_state_is_valid_and_first_transition_is_processed() -> None:
    virgin = initial_state(Decimal("10000"))
    event = buy_event()

    PortfolioInvariantChecker.validate_state(virgin)
    result = PortfolioTransitionEngine.transition(virgin, event.session, (event,))

    assert result.resulting_state.state_version == 1
    assert result.resulting_state.as_of_session == event.session
    assert result.resulting_state.settled_cash == Decimal("9979")
    assert len(result.resulting_state.open_positions) == 1


def test_virgin_state_with_open_position_is_rejected() -> None:
    processed = state_with_position()
    virgin_with_history = processed.model_copy(
        update={"as_of_session": None, "state_version": 0}
    )

    with pytest.raises(PortfolioStateError, match="virgin state"):
        PortfolioInvariantChecker.validate_state(virgin_with_history)


def test_virgin_state_with_pending_settlement_is_rejected() -> None:
    pending = PendingSettlement(
        settlement_id="SETTLE-1",
        source_execution_id="SELL-1",
        asset_id="NORGATE:1",
        amount=Decimal("10"),
        trade_session=date(2026, 8, 18),
        settlement_session=date(2026, 8, 20),
    )
    fingerprint = AppliedEventFingerprint(
        event_kind=PortfolioEventKind.EXECUTION,
        event_id="SELL-1",
        payload_sha256="a" * 64,
    )
    state = PortfolioState(
        pending_settlements=(pending,),
        applied_events=(fingerprint,),
    )

    with pytest.raises(PortfolioStateError, match="virgin state"):
        PortfolioInvariantChecker.validate_state(state)


def test_virgin_state_with_applied_fingerprint_is_rejected() -> None:
    state = PortfolioState(
        applied_events=(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id="EXEC-1",
                payload_sha256="a" * 64,
            ),
        )
    )

    with pytest.raises(PortfolioStateError, match="virgin state"):
        PortfolioInvariantChecker.validate_state(state)


def test_state_and_execution_hashes_normalize_decimal_spelling() -> None:
    state_a = PortfolioState(settled_cash=Decimal("100.000"))
    state_b = PortfolioState(settled_cash=Decimal("1E+2"))
    event_a = buy_event(fill_price=Decimal("10.00"), execution_cost=Decimal("1.0"))
    event_b = buy_event(fill_price=Decimal("1E+1"), execution_cost=Decimal("1"))

    assert hash_portfolio_state(state_a) == hash_portfolio_state(state_b)
    assert hash_execution_event(event_a) == hash_execution_event(event_b)
    assert b'"settled_cash":"100"' in canonical_payload_bytes(state_a)


def test_hashes_cover_complete_payload_and_canonical_state_order() -> None:
    event = buy_event()
    fingerprint = AppliedEventFingerprint(
        event_kind=PortfolioEventKind.EXECUTION,
        event_id=event.execution_id,
        payload_sha256=hash_execution_event(event),
    )
    result = PortfolioTransitionEngine.transition(
        initial_state(), event.session, (event,)
    )
    reconstructed = PortfolioState(
        settled_cash=result.resulting_state.settled_cash,
        as_of_session=event.session,
        state_version=1,
        applied_events=(fingerprint,),
        open_positions=tuple(reversed(result.resulting_state.open_positions)),
    )

    assert reconstructed == result.resulting_state
    assert hash_portfolio_state(reconstructed) == result.state_hash_after
    altered = event.model_copy(update={"source_order_id": "ORDER-ALTERED"})
    assert hash_execution_event(altered) != hash_execution_event(event)


def test_transition_invariant_detects_unexplained_cash_mutation() -> None:
    previous = initial_state()
    result = PortfolioTransitionEngine.transition(
        previous, date(2026, 8, 18), ()
    )
    tampered = result.resulting_state.model_copy(
        update={"settled_cash": Decimal("999")}
    )

    with pytest.raises(PortfolioStateError, match="settled-cash"):
        PortfolioInvariantChecker.validate_transition(previous, tampered, ())


def test_execution_ledger_provenance_accepts_valid_buy() -> None:
    event = buy_event()
    result = PortfolioTransitionEngine.transition(
        initial_state(), event.session, (event,)
    )

    PortfolioInvariantChecker.validate_execution_ledger_provenance(
        (event,), result.ledger_entries
    )


def test_execution_ledger_provenance_accepts_valid_sell() -> None:
    previous = state_with_position()
    event = sell_event()
    result = PortfolioTransitionEngine.transition(
        previous, event.session, (event,)
    )

    PortfolioInvariantChecker.validate_execution_ledger_provenance(
        (event,), result.ledger_entries
    )


def test_buy_payload_cannot_prove_sell_ledger_provenance() -> None:
    event = buy_event()
    result = PortfolioTransitionEngine.transition(
        initial_state(), event.session, (event,)
    )
    wrong_side_ledger = result.ledger_entries[0].model_copy(
        update={"event_type": PortfolioLedgerEventType.SELL_APPLIED}
    )

    with pytest.raises(PortfolioStateError, match="side does not match"):
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            (event,), (wrong_side_ledger,)
        )


def test_sell_payload_cannot_prove_buy_ledger_provenance() -> None:
    previous = state_with_position()
    event = sell_event()
    result = PortfolioTransitionEngine.transition(
        previous, event.session, (event,)
    )
    wrong_side_ledger = result.ledger_entries[0].model_copy(
        update={"event_type": PortfolioLedgerEventType.BUY_APPLIED}
    )

    with pytest.raises(PortfolioStateError, match="side does not match"):
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            (event,), (wrong_side_ledger,)
        )


def test_execution_ledger_provenance_rejects_missing_source() -> None:
    event = buy_event()
    result = PortfolioTransitionEngine.transition(
        initial_state(), event.session, (event,)
    )

    with pytest.raises(PortfolioStateError, match="missing its source"):
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            (), result.ledger_entries
        )


def test_execution_ledger_provenance_rejects_conflicting_identity_payloads() -> None:
    event = buy_event()
    conflict = buy_event(
        event.execution_id,
        fill_price=event.fill_price + Decimal("1"),
    )

    with pytest.raises(PortfolioStateError, match="conflicting payloads"):
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            (event, conflict), ()
        )


def test_execution_ledger_provenance_accepts_exact_duplicate_inputs() -> None:
    event = buy_event()
    result = PortfolioTransitionEngine.transition(
        initial_state(), event.session, (event,)
    )

    PortfolioInvariantChecker.validate_execution_ledger_provenance(
        (event, event), result.ledger_entries
    )


def test_execution_ledger_provenance_allows_replay_without_ledger_row() -> None:
    PortfolioInvariantChecker.validate_execution_ledger_provenance(
        (buy_event(),), ()
    )


def test_execution_ledger_provenance_rejects_mismatched_execution_facts() -> None:
    event = buy_event()
    result = PortfolioTransitionEngine.transition(
        initial_state(), event.session, (event,)
    )
    mismatched = result.ledger_entries[0].model_copy(
        update={"source_order_id": "OTHER-ORDER"}
    )

    with pytest.raises(PortfolioStateError, match="ledger facts"):
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            (event,), (mismatched,)
        )


def test_transition_provenance_rejects_side_inconsistent_payload_hash() -> None:
    previous = initial_state()
    buy = buy_event()
    result = PortfolioTransitionEngine.transition(
        previous, buy.session, (buy,)
    )
    sell_payload = sell_event(
        buy.execution_id,
        session=buy.session,
        asset_id=buy.asset_id,
        quantity=buy.quantity,
        settlement_session=date(2026, 8, 20),
    )
    mismatched = result.ledger_entries[0].model_copy(
        update={"source_payload_sha256": hash_execution_event(sell_payload)}
    )

    with pytest.raises(PortfolioStateError, match="BUY ledger source hash"):
        PortfolioInvariantChecker.validate_transition(
            previous,
            result.resulting_state,
            (mismatched,),
        )
