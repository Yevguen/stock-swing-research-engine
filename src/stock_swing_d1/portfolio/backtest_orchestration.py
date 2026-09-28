"""Thin deterministic Phase 13 portfolio backtest orchestration."""

from __future__ import annotations

from typing import Sequence

from pydantic import ValidationError

from stock_swing_d1.portfolio.portfolio_errors import (
    OutOfOrderSessionError,
    PortfolioTransitionError,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_invariants import PortfolioInvariantChecker
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioBacktestResult,
    PortfolioSessionInput,
    PortfolioSessionSnapshot,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine


def _validated_sessions(
    initial_state: PortfolioState,
    sessions: Sequence[PortfolioSessionInput],
) -> tuple[PortfolioSessionInput, ...]:
    try:
        supplied = tuple(sessions)
    except TypeError as error:
        raise PortfolioTransitionError(
            "sessions must be a sequence of PortfolioSessionInput values"
        ) from error

    validated: list[PortfolioSessionInput] = []
    previous_session = initial_state.as_of_session
    for session_input in supplied:
        if not isinstance(session_input, PortfolioSessionInput):
            raise PortfolioTransitionError(
                "sessions must contain PortfolioSessionInput values"
            )
        try:
            rebuilt = PortfolioSessionInput.model_validate(
                session_input.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioTransitionError(
                "portfolio session input fails structural validation"
            ) from error
        if rebuilt != session_input:
            raise PortfolioTransitionError(
                "portfolio session input must be canonical"
            )
        if previous_session is not None and rebuilt.session <= previous_session:
            raise OutOfOrderSessionError(
                "portfolio backtest sessions must be supplied in strict "
                "chronological order"
            )
        validated.append(rebuilt)
        previous_session = rebuilt.session
    return tuple(validated)


class PortfolioBacktestOrchestrator:
    """Coordinate normalized session inputs through the transition engine."""

    @staticmethod
    def run(
        initial_state: PortfolioState,
        sessions: Sequence[PortfolioSessionInput],
    ) -> PortfolioBacktestResult:
        PortfolioInvariantChecker.validate_state(initial_state)
        session_inputs = _validated_sessions(initial_state, sessions)

        initial_state_hash = hash_portfolio_state(initial_state)
        current_state = initial_state
        session_results = []
        ledger_entries = []
        dividend_ledger_entries = []
        dividend_outcomes = []
        snapshots = []

        for session_input in session_inputs:
            transition_result = PortfolioTransitionEngine.transition(
                current_state,
                session_input.session,
                session_input.execution_events,
                session_input.dividend_evidence,
            )
            session_results.append(transition_result)
            ledger_entries.extend(transition_result.ledger_entries)
            dividend_ledger_entries.extend(transition_result.dividend_ledger_entries)
            dividend_outcomes.extend(transition_result.dividend_outcomes)
            current_state = transition_result.resulting_state
            snapshots.append(
                PortfolioSessionSnapshot(
                    session=transition_result.session,
                    state_version=current_state.state_version,
                    settled_cash=current_state.settled_cash,
                    open_position_count=len(current_state.open_positions),
                    pending_settlement_count=len(
                        current_state.pending_settlements
                    ),
                    state_hash=transition_result.state_hash_after,
                )
            )

        final_state_hash = (
            session_results[-1].state_hash_after
            if session_results
            else initial_state_hash
        )
        return PortfolioBacktestResult(
            initial_state=initial_state,
            final_state=current_state,
            session_results=tuple(session_results),
            ledger_entries=tuple(ledger_entries),
            dividend_ledger_entries=tuple(dividend_ledger_entries),
            dividend_outcomes=tuple(dividend_outcomes),
            snapshots=tuple(snapshots),
            initial_state_hash=initial_state_hash,
            final_state_hash=final_state_hash,
        )


__all__ = ["PortfolioBacktestOrchestrator"]
