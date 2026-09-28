"""Static scope guards for the intentionally narrow Phase 9 package."""

from __future__ import annotations

import ast
from dataclasses import fields
import inspect
import textwrap

from stock_swing_d1.execution import (
    ENTRY_SLIPPAGE_BPS,
    EntryExecutionCalendar,
    EntryExecutionDecision,
    EntryExecutionService,
    EntryExecutionStatus,
    EntryExecutionValidationError,
    PendingEntry,
    SizedPendingEntry,
    create_pending_entry,
    create_sized_pending_entry,
)
from stock_swing_d1.execution.entry import EntryExecutionCostQuoteService
from stock_swing_d1.execution.entry import models, service
from stock_swing_d1.models import StockBar


def _phase9_tree() -> ast.AST:
    return ast.parse(inspect.getsource(models) + inspect.getsource(service))


def _identifiers() -> set[str]:
    tree = _phase9_tree()
    result = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
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


def test_public_phase9_api_is_exported() -> None:
    assert EntryExecutionCalendar.__name__ == "EntryExecutionCalendar"
    assert hasattr(EntryExecutionCalendar, "next_session")
    assert hasattr(EntryExecutionCalendar, "regular_session_open_time")
    assert hasattr(EntryExecutionCostQuoteService, "policy_ref")
    assert hasattr(EntryExecutionCostQuoteService, "quote")
    assert EntryExecutionService.__name__ == "EntryExecutionService"
    assert callable(create_pending_entry)
    assert callable(create_sized_pending_entry)
    assert issubclass(EntryExecutionValidationError, ValueError)
    assert ENTRY_SLIPPAGE_BPS == 5.0
    assert set(EntryExecutionStatus) == {
        EntryExecutionStatus.PENDING_ENTRY,
        EntryExecutionStatus.EXECUTED,
        EntryExecutionStatus.INVALIDATED_BY_EARNINGS,
        EntryExecutionStatus.NO_EXECUTABLE_BAR,
        EntryExecutionStatus.INVALID_OPEN_PRICE,
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION,
    }


def test_models_add_only_fixed_quantity_and_cost_quote_diagnostics() -> None:
    pending_fields = {field.name for field in fields(PendingEntry)}
    sized_fields = {field.name for field in fields(SizedPendingEntry)}
    decision_fields = {field.name for field in fields(EntryExecutionDecision)}

    assert pending_fields == {
        "security_id",
        "symbol",
        "signal_session",
        "signal_time",
        "planned_entry_session",
    }
    assert sized_fields == pending_fields | {"fixed_shares", "cash_available"}
    assert {
        "cash_available",
        "requested_shares",
        "executed_shares",
        "reference_open",
        "slippage_bps",
        "slippage_amount",
        "candidate_execution_price",
        "candidate_execution_cost_quote",
        "candidate_cash_required",
        "execution_price",
        "execution_cost_quote",
        "actual_cash_required",
        "status",
        "earnings_decision",
    } <= decision_fields
    assert {
        "capital",
        "rank",
        "commission",
        "fees",
        "tax",
        "portfolio_equity",
        "risk_budget",
        "risk_sized_shares",
        "cash_sized_shares",
    }.isdisjoint(pending_fields | sized_fields | decision_fields)


def test_execution_operation_accepts_existing_validated_stock_bar() -> None:
    parameters = inspect.signature(
        EntryExecutionService.execute_pending_entry
    ).parameters
    annotation = parameters["execution_bar"].annotation

    assert "StockBar" in str(annotation)
    assert "SizedPendingEntry" in str(
        parameters["sized_pending_entry"].annotation
    )


def test_fill_path_reads_open_but_never_t_plus_one_hlc_or_adjusted_open() -> None:
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(EntryExecutionService.execute_pending_entry)
        )
    )
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "open" in attributes
    assert {"high", "low", "close", "adjusted_close", "adjusted_open"}.isdisjoint(
        attributes
    )


def test_service_has_no_runtime_slippage_parameter() -> None:
    constructor_parameters = inspect.signature(EntryExecutionService).parameters
    execution_parameters = inspect.signature(
        EntryExecutionService.execute_pending_entry
    ).parameters

    assert "slippage_bps" not in constructor_parameters
    assert "slippage_bps" not in execution_parameters


def test_phase9_has_no_out_of_scope_methodology_identifiers() -> None:
    identifiers = _identifiers()

    assert {
        "exit",
        "exit_price",
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "gap_filter",
        "maximum_gap",
        "position_size",
        "portfolio",
        "portfolio_allocation",
        "cash_constraint",
        "portfolio_equity",
        "risk_budget",
        "risk_sized_shares",
        "cash_sized_shares",
        "floor",
        "partial_resize",
        "commission",
        "fees",
        "tax",
        "vwap",
        "limit_order",
        "partial_fill",
        "market_impact",
        "random",
        "t_plus_2",
        "machine_learning",
        "artificial_intelligence",
    }.isdisjoint(identifiers)


def test_phase9_consumes_but_does_not_own_cost_formulas() -> None:
    source = (inspect.getsource(models) + inspect.getsource(service)).lower()

    assert "execution_cost_service" in source
    assert "execution_cost_quote" in source
    for forbidden_formula_token in (
        "0.005",
        "minimum commission",
        "commission cap",
        "maximum_fraction_of_notional",
        "spread_bps_per_side",
        "fixed_cash_equivalent_bps_per_side",
        "commission_per_share_usd",
    ):
        assert forbidden_formula_token not in source


def test_phase9_has_no_provider_specific_logic() -> None:
    source = (inspect.getsource(models) + inspect.getsource(service)).lower()

    for provider_token in (
        "norgatedata",
        "wall_street_horizon",
        "factset",
        "s&p_global",
    ):
        assert provider_token not in source
