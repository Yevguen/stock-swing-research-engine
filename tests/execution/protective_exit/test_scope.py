"""Static scope guards for the intentionally narrow Phase 10 package."""

from __future__ import annotations

import ast
from dataclasses import fields
import inspect

from stock_swing_d1.execution.protective_exit import (
    EXIT_SLIPPAGE_BPS,
    ProtectiveExitAction,
    ProtectiveExitCalendar,
    ProtectiveExitDecision,
    ProtectiveExitService,
    ProtectiveExitState,
    ProtectiveExitValidationError,
)
from stock_swing_d1.execution.protective_exit import models, service


def _phase10_tree() -> ast.AST:
    return ast.parse(inspect.getsource(models) + inspect.getsource(service))


def _identifiers() -> set[str]:
    tree = _phase10_tree()
    result = {
        node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    result.update(
        node.attr.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    )
    result.update(
        node.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    )
    return result


def test_public_phase10_api_is_exported() -> None:
    assert ProtectiveExitCalendar.__name__ == "ProtectiveExitCalendar"
    assert hasattr(ProtectiveExitCalendar, "next_session")
    assert ProtectiveExitService.__name__ == "ProtectiveExitService"
    assert hasattr(ProtectiveExitService, "create_state")
    assert hasattr(ProtectiveExitService, "evaluate_session")
    assert issubclass(ProtectiveExitValidationError, ValueError)
    assert EXIT_SLIPPAGE_BPS == 5.0
    assert set(ProtectiveExitAction) == {
        ProtectiveExitAction.HOLD,
        ProtectiveExitAction.STOP_LOSS_EXIT,
        ProtectiveExitAction.GAP_STOP_EXIT,
        ProtectiveExitAction.TAKE_PROFIT_EXIT,
        ProtectiveExitAction.GAP_TAKE_PROFIT_EXIT,
        ProtectiveExitAction.UNRESOLVED_CAPITAL_EVENT,
    }


def test_models_have_only_per_share_protective_exit_scope() -> None:
    state_fields = {field.name for field in fields(ProtectiveExitState)}
    decision_fields = {field.name for field in fields(ProtectiveExitDecision)}

    assert {
        "signal_atr_fraction",
        "risk_fraction",
        "stop_price",
        "take_profit_price",
        "last_evaluated_session",
    } <= state_fields
    assert {
        "effective_stop",
        "effective_take_profit",
        "reference_exit_price",
        "execution_exit_price",
        "resulting_state",
    } <= decision_fields
    assert {
        "quantity",
        "position_size",
        "capital_allocation",
        "portfolio_value",
        "realized_pnl",
        "commission",
        "tax",
    }.isdisjoint(state_fields | decision_fields)


def test_service_has_no_runtime_strategy_parameters() -> None:
    constructor = inspect.signature(ProtectiveExitService).parameters
    create = inspect.signature(ProtectiveExitService.create_state).parameters
    evaluate = inspect.signature(
        ProtectiveExitService.evaluate_session
    ).parameters

    assert set(constructor) == {"trading_calendar"}
    assert "slippage_bps" not in create | evaluate
    assert "stop_price" not in create
    assert "take_profit_price" not in create


def test_phase10_has_no_out_of_scope_methodology_identifiers() -> None:
    identifiers = _identifiers()

    assert {
        "sma_exit",
        "rsi_exit",
        "maximum_holding",
        "time_exit",
        "earnings_exit",
        "exit_arbitration",
        "trailing_stop",
        "break_even_stop",
        "atr_trailing_stop",
        "position_size",
        "capital_allocation",
        "portfolio_pnl",
        "commission",
        "tax",
        "gap_filter",
        "maximum_gap",
        "intraday_path",
        "synthetic_ohlc_path",
        "random",
        "optimization",
        "machine_learning",
        "artificial_intelligence",
    }.isdisjoint(identifiers)


def test_phase10_uses_no_adjusted_fill_or_provider_specific_logic() -> None:
    source = (inspect.getsource(models) + inspect.getsource(service)).lower()

    assert "corporateactionadjustedstockbar" not in source
    for provider_token in (
        "norgatedata",
        "wall_street_horizon",
        "factset",
        "s&p_global",
    ):
        assert provider_token not in source
