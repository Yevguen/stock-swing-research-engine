"""Static ownership and public API guards for Phase 15B."""

from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
import inspect

from stock_swing_d1.execution.open_position_exit import (
    EarningsExitStatus,
    ExitBoundary,
    ExitPrerequisiteStatus,
    IntrabarAmbiguityStatus,
    MAX_HOLDING_SESSIONS,
    OpenPositionExitCalendar,
    OpenPositionExitDecision,
    OpenPositionExitEvaluationInput,
    OpenPositionExitEvaluator,
    OpenPositionExitReason,
)
from stock_swing_d1.execution.open_position_exit import models, service


def _phase15b_source() -> str:
    return inspect.getsource(models) + inspect.getsource(service)


def test_public_phase15b_api_is_frozen_and_complete() -> None:
    assert MAX_HOLDING_SESSIONS == 10
    assert is_dataclass(OpenPositionExitEvaluationInput)
    assert is_dataclass(OpenPositionExitDecision)
    assert hasattr(OpenPositionExitCalendar, "session_distance")
    assert hasattr(OpenPositionExitEvaluator, "evaluate")
    assert tuple(ExitBoundary) == (
        ExitBoundary.OPEN,
        ExitBoundary.INTRADAY,
        ExitBoundary.CLOSE,
    )
    assert len(OpenPositionExitReason) == 6
    assert len(IntrabarAmbiguityStatus) == 2
    assert len(EarningsExitStatus) == 5
    # Task 5C-C retired the same-session persistence member; READY remains.
    assert len(ExitPrerequisiteStatus) == 1


def test_decision_preserves_required_audit_fields() -> None:
    input_names = {field.name for field in fields(OpenPositionExitEvaluationInput)}
    names = {field.name for field in fields(OpenPositionExitDecision)}

    assert {
        "earnings_decision",
        "prior_boundary_earnings_decision",
    } <= input_names
    assert {
        "security_id",
        "session",
        "entry_session",
        "holding_session_number",
        "market_bar",
        "stop_price",
        "take_profit_price",
        "triggered_reasons",
        "selected_reason",
        "exit_boundary",
        "reference_exit_price",
        "final_execution_price",
        "exit_required",
        "intrabar_ambiguity_status",
        "earnings_exit_status",
        "earnings_deadline_session",
        "protective_exit_decision",
        "earnings_decision",
        "prior_boundary_earnings_decision",
    } <= names


def test_phase15b_invokes_phase10_instead_of_duplicating_exit_economics() -> None:
    tree = ast.parse(_phase15b_source())
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "evaluate_session" in calls
    assert "create_state" not in calls
    assert "_exit_prices" not in _phase15b_source()
    assert "EXIT_SLIPPAGE_BPS" not in _phase15b_source()


def test_phase15b_has_no_out_of_scope_ownership_leakage() -> None:
    identifiers = {
        node.id.lower()
        for node in ast.walk(ast.parse(_phase15b_source()))
        if isinstance(node, ast.Name)
    }

    assert {
        "sma",
        "rsi",
        "atr",
        "ranking",
        "position_sizing",
        "allocation",
        "settlement",
        "portfolio_state",
        "commission",
        "spread",
        "slippage_bps",
        "trailing_stop",
        "partial_exit",
        "short_position",
        "random",
        "uuid",
    }.isdisjoint(identifiers)
