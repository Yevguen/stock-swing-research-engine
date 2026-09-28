"""Focused public-boundary and invariant error-branch regressions."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from stock_swing_d1.portfolio.backtest_orchestration import (
    PortfolioBacktestOrchestrator,
)
from stock_swing_d1.portfolio.portfolio_errors import (
    OutOfOrderSessionError,
    PortfolioStateError,
    PortfolioTransitionError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    PortfolioEventKind,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import (
    canonical_payload_bytes,
    hash_portfolio_state,
)
from stock_swing_d1.portfolio.portfolio_invariants import PortfolioInvariantChecker
from stock_swing_d1.portfolio.portfolio_state_models import (
    PendingSettlement,
    PortfolioSessionInput,
    PortfolioSessionSnapshot,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import (
    buy_event,
    initial_state,
    sell_event,
    state_with_position,
)


def test_canonical_serializer_rejects_noncanonical_value_types() -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_payload_bytes(Decimal("NaN"))
    with pytest.raises(TypeError, match="string keys"):
        canonical_payload_bytes({1: "not-canonical"})
    with pytest.raises(TypeError, match="binary floats"):
        canonical_payload_bytes(1.5)
    with pytest.raises(TypeError, match="unsupported"):
        canonical_payload_bytes(object())


def test_models_reject_datetime_as_a_session_date() -> None:
    from datetime import datetime

    with pytest.raises(ValidationError, match="Python datetime.date"):
        PortfolioSessionInput(session=datetime(2026, 1, 1))


@pytest.mark.parametrize(
    "sessions",
    [
        None,
        (object(),),
        (
            PortfolioSessionInput(session=date(2026, 1, 1)).model_copy(
                update={"session": "not-a-date"}
            ),
        ),
        (
            PortfolioSessionInput(session=date(2026, 1, 1)).model_copy(
                update={"execution_events": []}
            ),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
def test_orchestrator_rejects_malformed_session_collections(sessions) -> None:
    with pytest.raises(PortfolioTransitionError):
        PortfolioBacktestOrchestrator.run(initial_state(), sessions)


def test_transition_rejects_malformed_session_and_execution_collections() -> None:
    with pytest.raises(OutOfOrderSessionError, match="must be a date"):
        PortfolioTransitionEngine.transition(initial_state(), "2026-01-01", ())
    with pytest.raises(PortfolioTransitionError, match="must be a sequence"):
        PortfolioTransitionEngine.transition(
            initial_state(),
            date(2026, 1, 1),
            None,
        )
    with pytest.raises(PortfolioTransitionError, match="must contain"):
        PortfolioTransitionEngine.transition(
            initial_state(),
            date(2026, 1, 1),
            (object(),),
        )
    malformed = buy_event(session=date(2026, 1, 1)).model_copy(
        update={"quantity": 0}
    )
    with pytest.raises(PortfolioTransitionError, match="structural"):
        PortfolioTransitionEngine.transition(
            initial_state(),
            date(2026, 1, 1),
            (malformed,),
        )
    noncanonical = buy_event(session=date(2026, 1, 1)).model_copy(
        update={"execution_id": " BUY-1 "}
    )
    with pytest.raises(PortfolioTransitionError, match="canonical"):
        PortfolioTransitionEngine.transition(
            initial_state(),
            date(2026, 1, 1),
            (noncanonical,),
        )


def test_state_validation_rejects_future_and_overlapping_references() -> None:
    processed = state_with_position(session=date(2026, 1, 1))
    position = processed.open_positions[0]
    future_position = position.model_copy(
        update={"entry_session": date(2026, 1, 2)}
    )
    with pytest.raises(PortfolioStateError, match="begin after"):
        PortfolioInvariantChecker.validate_state(
            processed.model_copy(update={"open_positions": (future_position,)})
        )

    pending = PendingSettlement(
        settlement_id="SETTLE-X",
        source_execution_id=position.entry_execution_id,
        asset_id=position.asset_id,
        amount=Decimal("1"),
        trade_session=date(2026, 1, 2),
        settlement_session=date(2026, 1, 3),
    )
    with pytest.raises(PortfolioStateError, match="both"):
        PortfolioInvariantChecker.validate_state(
            processed.model_copy(update={"pending_settlements": (pending,)})
        )
    without_position = processed.model_copy(update={"open_positions": ()})
    with pytest.raises(PortfolioStateError, match="trade after"):
        PortfolioInvariantChecker.validate_state(
            without_position.model_copy(update={"pending_settlements": (pending,)})
        )


def test_execution_provenance_rejects_duplicate_application_and_source_hash() -> None:
    event = buy_event()
    transition = PortfolioTransitionEngine.transition(
        initial_state(),
        event.session,
        (event,),
    )
    ledger = transition.ledger_entries[0]
    with pytest.raises(PortfolioStateError, match="multiple application"):
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            (event,),
            (ledger, ledger),
        )
    with pytest.raises(PortfolioStateError, match="source hash"):
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            (event,),
            (ledger.model_copy(update={"source_payload_sha256": "a" * 64}),),
        )
    with pytest.raises(PortfolioStateError, match="execution event is not canonical"):
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            (event.model_copy(update={"execution_id": " BUY-1 "}),),
            (),
        )
    with pytest.raises(PortfolioStateError, match="ledger entry is not canonical"):
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            (event,),
            (ledger.model_copy(update={"source_order_id": " ORDER-BUY-1 "}),),
        )


def test_transition_invariant_rejects_metadata_and_unexplained_mutations() -> None:
    first_session = date(2026, 1, 1)
    second_session = date(2026, 1, 2)
    previous = PortfolioTransitionEngine.transition(
        initial_state(), first_session, ()
    ).resulting_state
    valid = PortfolioTransitionEngine.transition(
        previous,
        second_session,
        (buy_event(session=second_session),),
    )
    empty_valid = PortfolioTransitionEngine.transition(previous, second_session, ())

    with pytest.raises(PortfolioStateError, match="completed session"):
        PortfolioInvariantChecker.validate_transition(
            initial_state(), initial_state(), ()
        )
    same_session = empty_valid.resulting_state.model_copy(
        update={"as_of_session": previous.as_of_session}
    )
    with pytest.raises(PortfolioStateError, match="follow"):
        PortfolioInvariantChecker.validate_transition(previous, same_session, ())
    wrong_version = valid.resulting_state.model_copy(
        update={"state_version": previous.state_version + 2}
    )
    with pytest.raises(PortfolioStateError, match="increment"):
        PortfolioInvariantChecker.validate_transition(previous, wrong_version, ())

    ledger = valid.ledger_entries[0]
    with pytest.raises(PortfolioStateError, match="consecutive"):
        PortfolioInvariantChecker.validate_transition(
            previous,
            valid.resulting_state,
            (ledger.model_copy(update={"sequence_in_session": 1}),),
        )
    with pytest.raises(PortfolioStateError, match="transition session"):
        PortfolioInvariantChecker.validate_transition(
            previous,
            valid.resulting_state,
            (ledger.model_copy(update={"session": second_session + timedelta(1)}),),
        )
    with pytest.raises(PortfolioStateError, match="state hashes"):
        PortfolioInvariantChecker.validate_transition(
            previous,
            valid.resulting_state,
            (ledger.model_copy(update={"state_hash_after": "a" * 64}),),
        )
    with pytest.raises(PortfolioStateError, match="settled-cash arithmetic"):
        PortfolioInvariantChecker.validate_transition(
            previous,
            valid.resulting_state,
            (
                ledger.model_copy(
                    update={"settled_cash_after": ledger.settled_cash_after + 1}
                ),
            ),
        )

    removed_position = valid.resulting_state.model_copy(update={"open_positions": ()})
    removed_position_ledger = tuple(
        item.model_copy(
            update={"state_hash_after": hash_portfolio_state(removed_position)}
        )
        for item in valid.ledger_entries
    )
    with pytest.raises(PortfolioStateError, match="position mutation"):
        PortfolioInvariantChecker.validate_transition(
            previous,
            removed_position,
            removed_position_ledger,
        )
    added_fingerprint = valid.resulting_state.model_copy(
        update={
            "applied_events": (
                *valid.resulting_state.applied_events,
                AppliedEventFingerprint(
                    event_kind=PortfolioEventKind.EXECUTION,
                    event_id="UNEXPLAINED",
                    payload_sha256="b" * 64,
                ),
            )
        }
    )
    added_fingerprint_ledger = tuple(
        item.model_copy(
            update={"state_hash_after": hash_portfolio_state(added_fingerprint)}
        )
        for item in valid.ledger_entries
    )
    with pytest.raises(PortfolioStateError, match="applied-event mutation"):
        PortfolioInvariantChecker.validate_transition(
            previous,
            added_fingerprint,
            added_fingerprint_ledger,
        )

    sold = PortfolioTransitionEngine.transition(
        state_with_position(session=first_session),
        second_session,
        (
            sell_event(
                session=second_session,
                settlement_session=second_session + timedelta(days=2),
            ),
        ),
    ).resulting_state
    carried = PortfolioTransitionEngine.transition(
        sold,
        second_session + timedelta(days=1),
        (),
    ).resulting_state
    removed_pending = carried.model_copy(update={"pending_settlements": ()})
    with pytest.raises(PortfolioStateError, match="pending-settlement mutation"):
        PortfolioInvariantChecker.validate_transition(
            sold,
            removed_pending,
            (),
        )


def test_transition_and_persisted_audit_reject_noncanonical_ledger_order() -> None:
    session = date(2026, 2, 2)
    previous = PortfolioState(settled_cash=Decimal("100"))
    events = (
        buy_event(
            "BUY-1",
            session=session,
            asset_id="NORGATE:1",
            quantity=1,
            fill_price=Decimal("10"),
            execution_cost=Decimal("0"),
        ),
        buy_event(
            "BUY-2",
            session=session,
            asset_id="NORGATE:2",
            quantity=1,
            fill_price=Decimal("10"),
            execution_cost=Decimal("0"),
        ),
    )
    result = PortfolioTransitionEngine.transition(previous, session, events)
    first, second = result.ledger_entries
    reversed_ledger = (
        second.model_copy(
            update={"sequence_in_session": 0, "settled_cash_after": Decimal("90")}
        ),
        first.model_copy(
            update={"sequence_in_session": 1, "settled_cash_after": Decimal("80")}
        ),
    )
    snapshot = PortfolioSessionSnapshot(
        session=session,
        state_version=1,
        settled_cash=Decimal("80"),
        open_position_count=2,
        pending_settlement_count=0,
        state_hash=result.state_hash_after,
    )

    with pytest.raises(PortfolioStateError, match="canonical session order"):
        PortfolioInvariantChecker.validate_transition(
            previous,
            result.resulting_state,
            reversed_ledger,
        )
    with pytest.raises(PortfolioStateError, match="canonical session order"):
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            previous,
            result.resulting_state,
            reversed_ledger,
            (snapshot,),
        )
    bad_cash_ledger = (
        first.model_copy(update={"settled_cash_after": Decimal("91")}),
        second,
    )
    with pytest.raises(PortfolioStateError, match="settled-cash arithmetic"):
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            previous,
            result.resulting_state,
            bad_cash_ledger,
            (snapshot,),
        )


def test_persisted_history_rejects_missing_sessions_sequences_and_final_due() -> None:
    event = buy_event(session=date(2026, 1, 1))
    transition = PortfolioTransitionEngine.transition(
        initial_state(), event.session, (event,)
    )
    snapshot = PortfolioSessionSnapshot(
        session=event.session,
        state_version=transition.resulting_state.state_version,
        settled_cash=transition.resulting_state.settled_cash,
        open_position_count=1,
        pending_settlement_count=0,
        state_hash=transition.state_hash_after,
    )
    with pytest.raises(PortfolioStateError, match="requires a session snapshot"):
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            initial_state(),
            transition.resulting_state,
            transition.ledger_entries,
            (),
        )
    with pytest.raises(PortfolioStateError, match="consecutive"):
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            initial_state(),
            transition.resulting_state,
            (
                transition.ledger_entries[0].model_copy(
                    update={"sequence_in_session": 1}
                ),
            ),
            (snapshot,),
        )

    entered = state_with_position(session=date(2026, 1, 1))
    sale = PortfolioTransitionEngine.transition(
        entered,
        date(2026, 1, 2),
        (
            sell_event(
                session=date(2026, 1, 2),
                settlement_session=date(2026, 1, 3),
            ),
        ),
    ).resulting_state
    impossible = sale.model_copy(
        update={"as_of_session": date(2026, 1, 3)}
    )
    with pytest.raises(PortfolioStateError, match="due or overdue"):
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            impossible,
            impossible,
            (),
            (),
        )


def test_persisted_history_rejects_unexplained_final_state_without_snapshots() -> None:
    first = PortfolioState(settled_cash=Decimal("10"))
    changed = PortfolioState(settled_cash=Decimal("11"))
    with pytest.raises(PortfolioStateError, match="final portfolio state"):
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            first,
            changed,
            (),
            (),
        )
