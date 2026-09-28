"""The Task 5C state-aware session-input provider contract.

A fully pre-materialized ``Sequence[HistoricalBacktestSessionInput]`` cannot
express a real multi-session run: entry execution bars depend on entry intents
carried from T-1, open-position exit evaluations depend on the positions that
actually survive and on carried Phase 10 protective state, and the allocation
portfolio depends on authoritative post-transition S[T].  This module adds the
narrow, additive seam through which Phase 15A retrieves exactly those
state-dependent *external facts* -- and nothing else.

The provider is never an economic owner.  It supplies market/earnings/
corporate-action facts and valuation marks; it never supplies protective
state, entry intents, ranking or allocation candidates, sizing, allocation
decisions, dividend evidence, or a calculated portfolio equity.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from stock_swing_d1.earnings.integration import EarningsIntegrationDecision
from stock_swing_d1.execution.protective_exit.models import (
    ProtectiveCorporateAction,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.valuation import PortfolioValuationMark

from stock_swing_d1.backtester.models import (
    _AwareDateTime,
    _CanonicalSecurityId,
    _ImmutableHistoricalBacktestModel,
    _SessionDate,
)


HISTORICAL_BACKTEST_SESSION_CONTEXT_SCHEMA_VERSION = (
    "historical_backtest_session_context.v0.1"
)
HISTORICAL_BACKTEST_OPEN_POSITION_FACTS_SCHEMA_VERSION = (
    "historical_backtest_open_position_facts.v0.1"
)
HISTORICAL_BACKTEST_SESSION_STATE_INPUTS_SCHEMA_VERSION = (
    "historical_backtest_session_state_inputs.v0.1"
)
HISTORICAL_BACKTEST_ALLOCATION_MARK_CONTEXT_SCHEMA_VERSION = (
    "historical_backtest_allocation_mark_context.v0.1"
)
HISTORICAL_BACKTEST_ALLOCATION_BOUNDARY_MARKS_SCHEMA_VERSION = (
    "historical_backtest_allocation_boundary_marks.v0.1"
)


def _require_ascending_unique(
    security_ids: tuple[str, ...], *, subject: str
) -> None:
    if len(set(security_ids)) != len(security_ids):
        raise ValueError(f"{subject} must be unique by security_id")
    if security_ids != tuple(sorted(security_ids)):
        raise ValueError(f"{subject} must be ordered by security_id ascending")


class HistoricalBacktestOpenPositionRef(_ImmutableHistoricalBacktestModel):
    """Runtime-only addressing tuple for one authoritative open position.

    Unversioned by design: it is a pure nested value object with no economic
    content, exactly like Phase 13's ``OpenPosition``/``PendingSettlement``
    inside the versioned ``portfolio_state.v0.1``.  Its shape is governed by
    the schema identity of the context that carries it, so changing it is a
    change to ``historical_backtest_session_context.v0.1``.
    """

    security_id: _CanonicalSecurityId
    entry_session: _SessionDate


class HistoricalBacktestSessionContext(_ImmutableHistoricalBacktestModel):
    """The minimum state-dependent identifiers a provider may observe.

    Deliberately absent: the authoritative ``PortfolioState`` itself, settled
    cash, pending settlements, quantities, cost basis, entry prices, rankings,
    sizing, allocations, the prior session result, and every future session.
    """

    schema_version: Literal[
        "historical_backtest_session_context.v0.1"
    ] = HISTORICAL_BACKTEST_SESSION_CONTEXT_SCHEMA_VERSION
    session: _SessionDate
    decision_time: _AwareDateTime
    next_session: _SessionDate | None = None
    open_position_refs: tuple[HistoricalBacktestOpenPositionRef, ...] = ()
    scheduled_entry_security_ids: tuple[_CanonicalSecurityId, ...] = ()

    @model_validator(mode="after")
    def validate_canonical_identity(self) -> HistoricalBacktestSessionContext:
        if self.next_session is not None and self.next_session <= self.session:
            raise ValueError("next_session must be later than session")
        _require_ascending_unique(
            tuple(ref.security_id for ref in self.open_position_refs),
            subject="open position refs",
        )
        _require_ascending_unique(
            self.scheduled_entry_security_ids,
            subject="scheduled entry security IDs",
        )
        return self


class HistoricalBacktestOpenPositionFacts(_ImmutableHistoricalBacktestModel):
    """External session-T facts for one open position.

    These are exactly the non-protective inputs of the frozen Phase 15B
    evaluation boundary.  ``ProtectiveExitState`` is intentionally absent:
    Phase 10 owns it and Phase 15A carries it between sessions.
    """

    schema_version: Literal[
        "historical_backtest_open_position_facts.v0.1"
    ] = HISTORICAL_BACKTEST_OPEN_POSITION_FACTS_SCHEMA_VERSION
    security_id: _CanonicalSecurityId
    market_bar: StockBar | None = None
    corporate_actions: tuple[ProtectiveCorporateAction, ...] = ()
    earnings_decision: EarningsIntegrationDecision | None = None
    prior_boundary_earnings_decision: EarningsIntegrationDecision | None = None

    @model_validator(mode="after")
    def validate_market_bar_identity(
        self,
    ) -> HistoricalBacktestOpenPositionFacts:
        if (
            self.market_bar is not None
            and self.market_bar.security_id != self.security_id
        ):
            raise ValueError(
                "market_bar must belong to this open position's security"
            )
        return self


class HistoricalBacktestSessionStateInputs(_ImmutableHistoricalBacktestModel):
    """One provider response of state-dependent external facts for session T."""

    schema_version: Literal[
        "historical_backtest_session_state_inputs.v0.1"
    ] = HISTORICAL_BACKTEST_SESSION_STATE_INPUTS_SCHEMA_VERSION
    session: _SessionDate
    entry_execution_bars: tuple[StockBar, ...] = ()
    open_position_facts: tuple[HistoricalBacktestOpenPositionFacts, ...] = ()

    @model_validator(mode="after")
    def validate_canonical_facts(
        self,
    ) -> HistoricalBacktestSessionStateInputs:
        bar_ids = tuple(bar.security_id for bar in self.entry_execution_bars)
        _require_ascending_unique(bar_ids, subject="entry execution bars")
        for bar in self.entry_execution_bars:
            if bar.trading_date != self.session:
                raise ValueError(
                    "every entry execution bar must belong to session T"
                )
        _require_ascending_unique(
            tuple(facts.security_id for facts in self.open_position_facts),
            subject="open position facts",
        )
        for facts in self.open_position_facts:
            if (
                facts.market_bar is not None
                and facts.market_bar.trading_date != self.session
            ):
                raise ValueError(
                    "every open-position market bar must belong to session T"
                )
        return self


class HistoricalBacktestAllocationMarkContext(
    _ImmutableHistoricalBacktestModel
):
    """Which marks the allocation boundary needs, and nothing more.

    The provider learns only the completed session and the authoritative
    open-position security IDs of S[T].  Settled cash, pending settlement
    values, quantities, cost basis, portfolio equity, candidate scores,
    rankings, allocation decisions and sizing are all deliberately absent.
    ``expected_market_data_artifact_ref`` is a retrieval convenience only:
    authority over the artifact belongs to the run-level argument that Phase
    15A checks the response against.
    """

    schema_version: Literal[
        "historical_backtest_allocation_mark_context.v0.1"
    ] = HISTORICAL_BACKTEST_ALLOCATION_MARK_CONTEXT_SCHEMA_VERSION
    session: _SessionDate
    security_ids: Annotated[
        tuple[_CanonicalSecurityId, ...], Field(min_length=1)
    ]
    expected_market_data_artifact_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def validate_canonical_security_ids(
        self,
    ) -> HistoricalBacktestAllocationMarkContext:
        _require_ascending_unique(
            self.security_ids, subject="allocation mark security IDs"
        )
        return self


class HistoricalBacktestAllocationBoundaryMarks(
    _ImmutableHistoricalBacktestModel
):
    """Mark facts only: never a calculated market value or portfolio equity."""

    schema_version: Literal[
        "historical_backtest_allocation_boundary_marks.v0.1"
    ] = HISTORICAL_BACKTEST_ALLOCATION_BOUNDARY_MARKS_SCHEMA_VERSION
    session: _SessionDate
    marks: tuple[PortfolioValuationMark, ...] = ()

    @model_validator(mode="after")
    def validate_canonical_marks(
        self,
    ) -> HistoricalBacktestAllocationBoundaryMarks:
        _require_ascending_unique(
            tuple(mark.security_id for mark in self.marks),
            subject="valuation marks",
        )
        for mark in self.marks:
            if mark.session != self.session:
                raise ValueError(
                    "every valuation mark must belong to the marked session"
                )
        return self


class HistoricalBacktestSessionInputProvider(Protocol):
    """The two narrow state-aware retrieval operations of one run."""

    def session_state_inputs(
        self,
        *,
        context: HistoricalBacktestSessionContext,
    ) -> HistoricalBacktestSessionStateInputs:
        """Return session-T external facts for the supplied identifiers."""

    def allocation_boundary_marks(
        self,
        *,
        context: HistoricalBacktestAllocationMarkContext,
    ) -> HistoricalBacktestAllocationBoundaryMarks:
        """Return completed-session-T marks for the supplied securities."""


def build_session_context(
    *,
    session: date,
    decision_time: object,
    next_session: date | None,
    open_positions: tuple[object, ...],
    scheduled_entry_security_ids: tuple[str, ...],
) -> HistoricalBacktestSessionContext:
    """Project the minimum provider context from authoritative Phase 13 state."""

    return HistoricalBacktestSessionContext(
        session=session,
        decision_time=decision_time,
        next_session=next_session,
        open_position_refs=tuple(
            HistoricalBacktestOpenPositionRef(
                security_id=position.asset_id,
                entry_session=position.entry_session,
            )
            for position in sorted(
                open_positions, key=lambda item: item.asset_id
            )
        ),
        scheduled_entry_security_ids=tuple(
            sorted(scheduled_entry_security_ids)
        ),
    )


__all__ = [
    "HISTORICAL_BACKTEST_ALLOCATION_BOUNDARY_MARKS_SCHEMA_VERSION",
    "HISTORICAL_BACKTEST_ALLOCATION_MARK_CONTEXT_SCHEMA_VERSION",
    "HISTORICAL_BACKTEST_OPEN_POSITION_FACTS_SCHEMA_VERSION",
    "HISTORICAL_BACKTEST_SESSION_CONTEXT_SCHEMA_VERSION",
    "HISTORICAL_BACKTEST_SESSION_STATE_INPUTS_SCHEMA_VERSION",
    "HistoricalBacktestAllocationBoundaryMarks",
    "HistoricalBacktestAllocationMarkContext",
    "HistoricalBacktestOpenPositionFacts",
    "HistoricalBacktestOpenPositionRef",
    "HistoricalBacktestSessionContext",
    "HistoricalBacktestSessionInputProvider",
    "HistoricalBacktestSessionStateInputs",
    "build_session_context",
]
