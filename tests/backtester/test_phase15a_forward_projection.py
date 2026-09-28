"""Phase 15A companion amendment: one-way Decimal -> float projection.

Authority flows forward only.  The float Phase 12 consumed must equal
``float(authoritative Decimal)`` exactly; the reverse comparison
``Decimal(str(supplied_float)) == authoritative_decimal`` is prohibited
because a legitimate high-precision balance produced by the frozen dividend
accounting can fail it while the forward projection is exactly right.
"""

from __future__ import annotations

import ast
import inspect
import math
from decimal import Decimal

import pytest

from stock_swing_d1.backtester import (
    HistoricalBacktestSessionInput,
    HistoricalBacktestValidationError,
)
from stock_swing_d1.backtester import validation as validation_module
from stock_swing_d1.backtester.validation import (
    validate_allocation_portfolio_against_state,
)
from stock_swing_d1.portfolio import PortfolioSnapshot
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState

from tests.backtester.conftest import decision_time
from tests.backtester.test_phase15a_session_input_provider import (
    INTERVAL,
    MARKET_DATA_REF,
    S1,
    S2,
    RecordingProvider,
    build_orchestrator,
    candidate_plan,
)


# Q_T * D_H credited into settled cash by the frozen Gate3 dividend path.
DIVIDEND_SCALE_CASH = Decimal(
    "10031.02658067566857563179740450795985919134"
)


def _executable_code(function) -> str:
    """Unparse a function with its docstring removed.

    Prose that quotes the prohibited form must not be able to satisfy -- or
    to break -- a check about the executable code.
    """

    tree = ast.parse(inspect.getsource(function))
    definition = tree.body[0]
    body = definition.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        definition.body = body[1:]
    return ast.unparse(tree)


def _session_input(
    *, cash: float, equity: float
) -> HistoricalBacktestSessionInput:
    return HistoricalBacktestSessionInput(
        session=S1,
        decision_time=decision_time(S1),
        next_session=S2,
        allocation_portfolio=PortfolioSnapshot(
            allocation_session=S1,
            decision_time=decision_time(S1),
            portfolio_equity=equity,
            cash_available=cash,
        ),
    )


def test_the_reverse_round_trip_fails_for_a_legitimate_balance():
    assert Decimal(str(float(DIVIDEND_SCALE_CASH))) != DIVIDEND_SCALE_CASH
    assert float(DIVIDEND_SCALE_CASH) == float(DIVIDEND_SCALE_CASH)


def test_the_prescribed_forward_projection_is_accepted():
    state = PortfolioState(settled_cash=DIVIDEND_SCALE_CASH)

    validate_allocation_portfolio_against_state(
        session_input=_session_input(
            cash=float(DIVIDEND_SCALE_CASH),
            equity=float(DIVIDEND_SCALE_CASH),
        ),
        authoritative_state=state,
        authoritative_portfolio_equity=DIVIDEND_SCALE_CASH,
    )


def test_one_ulp_cash_change_is_rejected():
    state = PortfolioState(settled_cash=DIVIDEND_SCALE_CASH)
    equity = DIVIDEND_SCALE_CASH + Decimal("1000")
    tampered = math.nextafter(float(DIVIDEND_SCALE_CASH), -math.inf)

    assert tampered != float(DIVIDEND_SCALE_CASH)
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="NONAUTHORITATIVE_ALLOCATION_CASH",
    ):
        validate_allocation_portfolio_against_state(
            session_input=_session_input(cash=tampered, equity=float(equity)),
            authoritative_state=state,
            authoritative_portfolio_equity=equity,
        )


def test_one_ulp_equity_change_is_rejected():
    state = PortfolioState(settled_cash=DIVIDEND_SCALE_CASH)
    equity = DIVIDEND_SCALE_CASH + Decimal("1000")
    tampered = math.nextafter(float(equity), math.inf)

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="NONAUTHORITATIVE_ALLOCATION_EQUITY",
    ):
        validate_allocation_portfolio_against_state(
            session_input=_session_input(
                cash=float(DIVIDEND_SCALE_CASH), equity=tampered
            ),
            authoritative_state=state,
            authoritative_portfolio_equity=equity,
        )


def test_a_non_decimal_authoritative_equity_is_rejected():
    with pytest.raises(
        HistoricalBacktestValidationError,
        match="NONAUTHORITATIVE_ALLOCATION_EQUITY",
    ):
        validate_allocation_portfolio_against_state(
            session_input=_session_input(cash=100.0, equity=100.0),
            authoritative_state=PortfolioState(settled_cash=Decimal("100")),
            authoritative_portfolio_equity=100.0,
        )


def test_the_legacy_path_still_proves_cash_without_authoritative_equity():
    """No marks exist on the legacy path, so only cash is provable there."""

    validate_allocation_portfolio_against_state(
        session_input=_session_input(cash=100.0, equity=100_000.0),
        authoritative_state=PortfolioState(settled_cash=Decimal("100")),
    )

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="NONAUTHORITATIVE_ALLOCATION_CASH",
    ):
        validate_allocation_portfolio_against_state(
            session_input=_session_input(cash=210.0, equity=100_000.0),
            authoritative_state=PortfolioState(settled_cash=Decimal("100")),
        )


def test_no_reverse_projection_appears_in_the_amended_validator():
    code = _executable_code(validate_allocation_portfolio_against_state)

    assert "Decimal(str(" not in code
    assert (
        "portfolio.cash_available != float(authoritative_state.settled_cash)"
        in code
    )
    assert (
        "portfolio.portfolio_equity != float(authoritative_portfolio_equity)"
        in code
    )
    for forbidden in ("isclose", "quantize", "abs(", "epsilon", "tolerance"):
        assert forbidden not in code


def test_the_amendment_uses_no_tolerance_anywhere_in_validation():
    source = inspect.getsource(validation_module)

    assert "math.isclose" not in source
    assert "isclose(" not in source


def test_a_dividend_scale_balance_runs_through_the_provider_path():
    """End-to-end regression: this run is impossible under reverse authority."""

    provider = RecordingProvider()

    result = build_orchestrator().run_with_session_input_provider(
        PortfolioState(settled_cash=DIVIDEND_SCALE_CASH),
        (candidate_plan(S1, next_session=S2),),
        provider=provider,
        decision_interval=INTERVAL,
        market_data_artifact_ref=MARKET_DATA_REF,
    )

    allocation = result.session_results[0].allocation_decision
    assert allocation is not None
    assert allocation.starting_cash == float(DIVIDEND_SCALE_CASH)
    assert allocation.starting_portfolio_equity == float(DIVIDEND_SCALE_CASH)
    assert result.session_results[0].authoritative_state.settled_cash == (
        DIVIDEND_SCALE_CASH
    )


def test_a_non_positive_projected_equity_fails_closed():
    provider = RecordingProvider()

    with pytest.raises(
        HistoricalBacktestValidationError,
        match="INVALID_PROJECTED_ALLOCATION_PORTFOLIO",
    ):
        build_orchestrator().run_with_session_input_provider(
            PortfolioState(settled_cash=Decimal("0")),
            (candidate_plan(S1, next_session=S2),),
            provider=provider,
            decision_interval=INTERVAL,
            market_data_artifact_ref=MARKET_DATA_REF,
        )


def test_the_projection_happens_exactly_once_in_the_orchestrator():
    from stock_swing_d1.backtester import orchestration

    code = _executable_code(orchestration._project_allocation_portfolio)

    assert code.count("float(") == 2
    assert "float(valuation.portfolio_equity)" in code
    assert "float(state.settled_cash)" in code
