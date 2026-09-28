"""Static causal and scope guards for the Phase 9/11 remediation."""

from __future__ import annotations

import ast
from dataclasses import fields
import inspect

from stock_swing_d1.execution.entry import SizedPendingEntry
from stock_swing_d1.execution.protective_exit import EXIT_SLIPPAGE_BPS
from stock_swing_d1.risk import (
    TARGET_RISK_FRACTION,
    ExecutedInitialRisk,
    PortfolioSizingSnapshot,
    PositionSizingAction,
    PositionSizingConstraint,
    PositionSizingDecision,
    PositionSizingService,
    PositionSizingValidationError,
    TradeRiskRealization,
    calculate_executed_initial_risk,
    realize_risk,
)
from stock_swing_d1.risk.position_sizing import models, service


def _phase11_tree() -> ast.AST:
    return ast.parse(inspect.getsource(models) + inspect.getsource(service))


def _identifiers() -> set[str]:
    tree = _phase11_tree()
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


def test_public_phase11_api_and_constants_are_exported() -> None:
    assert TARGET_RISK_FRACTION == 0.005
    assert EXIT_SLIPPAGE_BPS == 5.0
    assert PositionSizingService.__name__ == "PositionSizingService"
    assert hasattr(PositionSizingService, "size_pending_entry")
    assert not hasattr(PositionSizingService, "size_candidate")
    assert hasattr(PositionSizingService, "calculate_executed_initial_risk")
    assert hasattr(PositionSizingService, "realize_risk")
    assert callable(calculate_executed_initial_risk)
    assert callable(realize_risk)
    assert issubclass(PositionSizingValidationError, ValueError)
    assert set(PositionSizingAction) == {
        PositionSizingAction.SIZED,
        PositionSizingAction.SKIPPED_INSUFFICIENT_CASH,
        PositionSizingAction.SKIPPED_RISK_TOO_SMALL,
    }
    assert set(PositionSizingConstraint) == {
        PositionSizingConstraint.RISK,
        PositionSizingConstraint.CASH,
        PositionSizingConstraint.BOTH_EQUAL,
    }


def test_sizing_decision_contains_only_completed_t_risk_diagnostics() -> None:
    snapshot_fields = {field.name for field in fields(PortfolioSizingSnapshot)}
    decision_fields = {field.name for field in fields(PositionSizingDecision)}

    assert snapshot_fields == {"portfolio_equity", "cash_available"}
    assert {
        "security_id",
        "symbol",
        "signal_session",
        "signal_time",
        "planned_entry_session",
        "portfolio_equity",
        "cash_available",
        "target_risk_fraction",
        "sizing_target_risk_amount",
        "signal_atr_fraction",
        "sizing_risk_fraction",
        "sizing_reference_price",
        "sizing_reference_stop_price",
        "sizing_reference_stop_exit_price",
        "sizing_loss_per_share",
        "risk_sized_shares",
        "cash_sized_shares",
        "final_shares",
        "sizing_reference_cash_required",
        "sizing_reference_cash_after_entry",
        "sizing_planned_risk_amount",
        "sizing_planned_risk_fraction",
        "unused_sizing_risk_budget",
        "binding_constraint",
        "action",
        "sized_pending_entry",
    } == decision_fields
    assert {
        "entry_execution_price",
        "entry_execution_time",
        "stop_price",
        "actual_entry_execution_price",
        "actual_stop_price",
    }.isdisjoint(decision_fields)


def test_sizing_api_accepts_only_completed_t_inputs() -> None:
    parameters = inspect.signature(
        PositionSizingService.size_pending_entry
    ).parameters

    assert set(parameters) == {
        "self",
        "signal",
        "pending_entry",
        "signal_bar",
        "portfolio",
    }
    assert "BaselineSignalDecision" in str(parameters["signal"].annotation)
    assert "PendingEntry" in str(parameters["pending_entry"].annotation)
    assert "StockBar" in str(parameters["signal_bar"].annotation)
    assert "PortfolioSizingSnapshot" in str(
        parameters["portfolio"].annotation
    )
    signature = str(inspect.signature(PositionSizingService.size_pending_entry))
    for forbidden in (
        "EntryExecutionDecision",
        "ProtectiveExitState",
        "ProtectiveExitDecision",
        "actual_execution_price",
        "actual_exit_execution_price",
    ):
        assert forbidden not in signature


def test_sizing_reads_signal_bar_close_but_no_future_ohlc() -> None:
    tree = ast.parse(
        inspect.getsource(service._validate_signal_bar)
        + inspect.getsource(service._size_pending_entry)
    )
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "close" in attributes
    assert {"open", "high", "low"}.isdisjoint(attributes)


def test_post_execution_models_keep_four_risk_concepts_distinct() -> None:
    initial_fields = {field.name for field in fields(ExecutedInitialRisk)}
    realized_fields = {field.name for field in fields(TradeRiskRealization)}

    assert {
        "sizing_target_risk_amount",
        "sizing_planned_risk_amount",
        "executed_initial_risk_amount",
        "executed_initial_risk_fraction",
    } <= initial_fields
    assert {
        "sizing_planned_risk_amount",
        "actual_entry_execution_price",
        "actual_exit_execution_price",
        "realized_risk_amount",
        "realized_risk_fraction",
        "risk_overrun_amount",
    } <= realized_fields


def test_phase11_returns_phase9_owned_dependency_safe_order() -> None:
    decision_annotation = {
        field.name: str(field.type) for field in fields(PositionSizingDecision)
    }

    assert "SizedPendingEntry" in decision_annotation["sized_pending_entry"]
    assert SizedPendingEntry.__module__.startswith(
        "stock_swing_d1.execution.entry"
    )


def test_phase11_has_no_batch_or_shared_cash_or_ordering_api() -> None:
    identifiers = _identifiers()

    assert {
        "rank_candidates",
        "size_all",
        "allocate_portfolio",
        "reserve_cash",
        "candidate_priority",
        "simultaneous_ordering",
    }.isdisjoint(identifiers)


def test_phase11_has_no_out_of_scope_capital_or_risk_methods() -> None:
    identifiers = _identifiers()

    assert {
        "margin",
        "borrowed_cash",
        "leverage_multiplier",
        "broker_buying_power",
        "kelly_criterion",
        "volatility_target",
        "risk_parity",
        "portfolio_var",
        "cvar",
        "correlation_adjustment",
        "sector_adjustment",
        "maximum_position_percentage",
        "gap_prediction_multiplier",
        "commission",
        "tax",
        "optimization",
        "machine_learning",
        "artificial_intelligence",
    }.isdisjoint(identifiers)


def test_phase11_has_no_provider_specific_logic() -> None:
    source = (inspect.getsource(models) + inspect.getsource(service)).lower()

    for provider_token in (
        "norgatedata",
        "wall_street_horizon",
        "factset",
        "s&p_global",
    ):
        assert provider_token not in source
