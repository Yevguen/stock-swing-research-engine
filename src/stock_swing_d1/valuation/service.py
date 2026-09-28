"""The single repository implementation of allocation-boundary equity.

Task 5C-A froze exactly one formula and exactly one implementation of it::

    portfolio_equity[T]
        = settled_cash[T]
        + sum(pending_settlement.amount)                       (face value)
        + sum(open_position.quantity * unadjusted D1 close[T]) (market value)

Phase 15A (runtime) and Phase 15D (audit) both call this function; neither may
re-derive the arithmetic.  Pending settlements are included at face value here
and excluded from ``cash_available`` by the caller; ordinary dividends are
already inside ``settled_cash`` (Phase 13 credits them there) so there is no
separate dividend term; cost basis never enters market value.

Every accumulation uses the frozen exact-Decimal helpers rather than ``+``,
``*`` or ``sum``: ``settled_cash`` can carry Gate3 scale-38 dividend-cash
precision, and OD-6.7's "no free ambient-context precision parameter" applies
to any Decimal accumulation, not only dividend cash.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import ValidationError

from stock_swing_d1.portfolio.portfolio_dividend_events import (
    add_exact_decimal,
    exact_decimal_times_int,
    subtract_exact_decimal,
)
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.valuation.errors import PortfolioValuationError
from stock_swing_d1.valuation.models import PortfolioValuationMark
from stock_swing_d1.valuation.policy import PortfolioValuationPolicy
from stock_swing_d1.valuation.results import PortfolioValuationResult


_ZERO = Decimal("0")


def _exact_sum(values: object) -> Decimal:
    """Sum exactly via repeated ``add_exact_decimal``, never ``sum``/``+``."""

    total = _ZERO
    for value in values:
        total = add_exact_decimal(total, value)
    return total


def _require_frozen_policy(policy: object) -> PortfolioValuationPolicy:
    if type(policy) is not PortfolioValuationPolicy:
        raise PortfolioValuationError(
            "VALUATION_INPUT_INVALID",
            "policy must be exactly PortfolioValuationPolicy",
        )
    rebuilt = PortfolioValuationPolicy.model_validate(
        policy.model_dump(mode="python")
    )
    if rebuilt != PortfolioValuationPolicy():
        raise PortfolioValuationError(
            "VALUATION_INPUT_INVALID",
            "policy must be the frozen completed-session unadjusted-close policy",
        )
    return rebuilt


def _require_canonical_mark(mark: object) -> PortfolioValuationMark:
    """Fully revalidate one mark against the frozen schema and invariants.

    A nested model instance is never trusted merely because it already has
    the right Python class: ``model_copy(update=...)`` bypasses validation,
    so a forged ``price_basis``, session, price or nested source ref could
    otherwise survive outer-model rebuilding.  Rebuilding from a plain
    ``model_dump`` forces every field -- including the nested ``ArtifactRef``
    -- back through the authoritative validators, and the rebuilt value must
    equal the supplied one exactly.
    """

    if type(mark) is not PortfolioValuationMark:
        raise PortfolioValuationError(
            "VALUATION_INPUT_INVALID",
            "marks must contain only exact PortfolioValuationMark members",
        )
    try:
        rebuilt = PortfolioValuationMark.model_validate(
            mark.model_dump(mode="python")
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise PortfolioValuationError(
            "VALUATION_INPUT_INVALID",
            "a valuation mark fails authoritative revalidation",
        ) from error
    if rebuilt != mark:
        raise PortfolioValuationError(
            "VALUATION_INPUT_INVALID",
            "a valuation mark is not canonical",
        )
    return rebuilt


def _require_exact_mark_tuple(
    marks: object,
) -> tuple[PortfolioValuationMark, ...]:
    if type(marks) is not tuple:
        raise PortfolioValuationError(
            "VALUATION_INPUT_INVALID",
            "marks must be exactly a tuple of PortfolioValuationMark, not a "
            "list, generator, set, mapping, or other iterable",
        )
    return tuple(_require_canonical_mark(mark) for mark in marks)


def _require_bound_session(
    *, state: PortfolioState, session: object
) -> date | None:
    """Bind the valuation session to the authoritative state identity.

    A dated state (``as_of_session`` set) can only be valued for exactly that
    session: ``session=None`` and any other session fail closed, so no state
    can be valued against another session's marks.  Cash-only states are not
    exempt once they carry a session identity.  A virgin state
    (``as_of_session is None``) is valued only with ``session=None``.
    """

    if session is not None and type(session) is not date:
        raise PortfolioValuationError(
            "VALUATION_INPUT_INVALID",
            "session must be a Python datetime.date or None",
        )
    if session != state.as_of_session:
        raise PortfolioValuationError(
            "VALUATION_SESSION_MISMATCH",
            "the valuation session must equal the authoritative state's "
            "as_of_session",
        )
    return session


def value_portfolio_at_allocation_boundary(
    *,
    state: PortfolioState,
    session: date | None,
    marks: tuple[PortfolioValuationMark, ...],
    policy: PortfolioValuationPolicy,
) -> PortfolioValuationResult:
    """Value one authoritative post-transition state against evidenced marks.

    ``session`` must equal ``state.as_of_session``: a dated state is valued
    only for its own session, and a virgin state (``as_of_session is None``)
    only with ``session=None`` and no marks.
    """

    if type(state) is not PortfolioState:
        raise PortfolioValuationError(
            "VALUATION_INPUT_INVALID",
            "state must be an authoritative Phase 13 PortfolioState",
        )
    session = _require_bound_session(state=state, session=session)
    exact_marks = _require_exact_mark_tuple(marks)
    _require_frozen_policy(policy)

    expected_security_ids = {
        position.asset_id for position in state.open_positions
    }
    mark_security_ids = [mark.security_id for mark in exact_marks]
    if len(set(mark_security_ids)) != len(mark_security_ids):
        raise PortfolioValuationError(
            "VALUATION_MARK_COVERAGE_MISMATCH",
            "valuation marks must be unique by security_id",
        )
    if set(mark_security_ids) != expected_security_ids:
        raise PortfolioValuationError(
            "VALUATION_MARK_COVERAGE_MISMATCH",
            "valuation mark coverage must equal the exact open-position "
            "security set of the supplied state",
        )
    if session is None and exact_marks:
        raise PortfolioValuationError(
            "VALUATION_SESSION_MISMATCH",
            "marks require an explicit valuation session",
        )
    for mark in exact_marks:
        if mark.session != session:
            raise PortfolioValuationError(
                "VALUATION_SESSION_MISMATCH",
                "every valuation mark must belong to the valued session",
            )
    if exact_marks:
        source_refs = {mark.source_artifact_ref for mark in exact_marks}
        if len(source_refs) != 1:
            raise PortfolioValuationError(
                "VALUATION_ARTIFACT_MISMATCH",
                "every valuation mark must share one market-data artifact ref",
            )

    marks_by_security = {mark.security_id: mark for mark in exact_marks}
    pending_receivable_value = _exact_sum(
        settlement.amount for settlement in state.pending_settlements
    )
    open_position_market_value = _ZERO
    for position in sorted(
        state.open_positions, key=lambda item: item.asset_id
    ):
        mark = marks_by_security[position.asset_id]
        open_position_market_value = add_exact_decimal(
            open_position_market_value,
            exact_decimal_times_int(mark.close, position.quantity),
        )

    portfolio_equity = add_exact_decimal(
        add_exact_decimal(state.settled_cash, pending_receivable_value),
        open_position_market_value,
    )
    if subtract_exact_decimal(
        portfolio_equity, state.settled_cash
    ) != add_exact_decimal(
        pending_receivable_value, open_position_market_value
    ):
        raise PortfolioValuationError(
            "VALUATION_INVARIANT_VIOLATION",
            "portfolio_equity minus settled_cash must equal the pending "
            "receivable plus open-position market value",
        )

    return PortfolioValuationResult(
        settled_cash=state.settled_cash,
        pending_receivable_value=pending_receivable_value,
        open_position_market_value=open_position_market_value,
        portfolio_equity=portfolio_equity,
    )


__all__ = ["value_portfolio_at_allocation_boundary"]
