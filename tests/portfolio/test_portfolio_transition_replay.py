"""Exactly-once execution and session-membership tests for Phase 13B.2."""

from datetime import date

import pytest

from stock_swing_d1.portfolio.portfolio_errors import (
    DuplicateEventConflictError,
    OutOfOrderSessionError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    PortfolioEventKind,
    PortfolioEventReference,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import buy_event, initial_state


def test_new_execution_in_current_transition_session_is_accepted() -> None:
    session = date(2026, 8, 20)
    event = buy_event(session=session)

    result = PortfolioTransitionEngine.transition(initial_state(), session, (event,))

    assert result.newly_applied_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id=event.execution_id,
        ),
    )


@pytest.mark.parametrize("event_session", [date(2026, 8, 18), date(2026, 8, 21)])
def test_new_execution_outside_current_session_is_rejected_atomically(
    event_session: date,
) -> None:
    previous = initial_state()
    event = buy_event(session=event_session)

    with pytest.raises(OutOfOrderSessionError, match="must equal"):
        PortfolioTransitionEngine.transition(
            previous, date(2026, 8, 20), (event,)
        )

    assert previous == initial_state()


def test_identical_historical_execution_is_replayed_without_economic_mutation() -> None:
    original = buy_event(session=date(2026, 8, 18))
    applied = PortfolioTransitionEngine.transition(
        initial_state(), original.session, (original,)
    ).resulting_state

    result = PortfolioTransitionEngine.transition(
        applied, date(2026, 8, 20), (original,)
    )

    assert result.replayed_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id=original.execution_id,
        ),
    )
    assert result.newly_applied_events == ()
    assert result.ledger_entries == ()
    assert result.resulting_state.settled_cash == applied.settled_cash
    assert result.resulting_state.open_positions == applied.open_positions
    assert result.resulting_state.applied_events == applied.applied_events
    assert result.resulting_state.state_version == applied.state_version + 1


def test_execution_replay_does_not_debit_cash_or_add_position_twice() -> None:
    original = buy_event()
    applied = PortfolioTransitionEngine.transition(
        initial_state(), original.session, (original,)
    ).resulting_state

    replayed = PortfolioTransitionEngine.transition(
        applied, date(2026, 8, 19), (original,)
    ).resulting_state

    assert replayed.settled_cash == applied.settled_cash
    assert replayed.open_positions == applied.open_positions
    assert len(replayed.open_positions) == 1


def test_same_execution_id_with_changed_payload_conflicts_before_session_check() -> None:
    original = buy_event(session=date(2026, 8, 18))
    applied = PortfolioTransitionEngine.transition(
        initial_state(), original.session, (original,)
    ).resulting_state
    changed_session = buy_event(
        original.execution_id, session=date(2026, 8, 20)
    )

    with pytest.raises(DuplicateEventConflictError):
        PortfolioTransitionEngine.transition(
            applied, date(2026, 8, 20), (changed_session,)
        )


def test_same_execution_id_with_altered_price_conflicts() -> None:
    original = buy_event()
    applied = PortfolioTransitionEngine.transition(
        initial_state(), original.session, (original,)
    ).resulting_state
    altered = buy_event(original.execution_id, fill_price=original.fill_price + 1)

    with pytest.raises(DuplicateEventConflictError):
        PortfolioTransitionEngine.transition(
            applied, date(2026, 8, 19), (altered,)
        )


def test_replayed_event_references_have_deterministic_canonical_order() -> None:
    original_session = date(2026, 8, 18)
    event_b = buy_event(
        "EXEC-B",
        session=original_session,
        asset_id="NORGATE:2",
    )
    event_a = buy_event(
        "EXEC-A",
        session=original_session,
        asset_id="NORGATE:1",
    )
    applied = PortfolioTransitionEngine.transition(
        initial_state(),
        original_session,
        (event_b, event_a),
    ).resulting_state

    result = PortfolioTransitionEngine.transition(
        applied,
        date(2026, 8, 20),
        (event_b, event_a),
    )

    assert result.replayed_events == (
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id="EXEC-A",
        ),
        PortfolioEventReference(
            event_kind=PortfolioEventKind.EXECUTION,
            event_id="EXEC-B",
        ),
    )
