"""Acceptance tests for deterministic completed-T Phase 11 sizing."""

from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from math import inf, nan

import pytest
from pydantic import ValidationError

from stock_swing_d1.execution.entry import SizedPendingEntry
from stock_swing_d1.risk.position_sizing import (
    TARGET_RISK_FRACTION,
    PortfolioSizingSnapshot,
    PositionSizingAction,
    PositionSizingConstraint,
    PositionSizingDecision,
    PositionSizingValidationError,
)
from stock_swing_d1.strategy.baseline import BaselineSignalAction


def _size(
    service,
    signal,
    pending,
    signal_bar,
    *,
    equity=10_000,
    cash=10_000,
):
    return service.size_pending_entry(
        signal=signal,
        pending_entry=pending,
        signal_bar=signal_bar,
        portfolio=PortfolioSizingSnapshot(
            portfolio_equity=equity,
            cash_available=cash,
        ),
    )


def test_baseline_completed_t_sizing_example(
    service, make_completed_t
) -> None:
    signal, pending, signal_bar = make_completed_t(
        adjusted_close=75.0,
        unadjusted_close=100.0,
        atr_fraction=0.02,
    )

    decision = _size(service, signal, pending, signal_bar)

    assert TARGET_RISK_FRACTION == 0.005
    assert decision.target_risk_fraction == 0.005
    assert decision.sizing_target_risk_amount == 50.0
    assert signal.adjusted_close == 75.0
    assert decision.sizing_reference_price == 100.0
    assert decision.signal_atr_fraction == 0.02
    assert decision.sizing_risk_fraction == 0.04
    assert decision.sizing_reference_stop_price == 96.0
    assert decision.sizing_reference_stop_exit_price == pytest.approx(95.952)
    assert decision.sizing_loss_per_share == pytest.approx(4.048)
    assert decision.risk_sized_shares == 12
    assert decision.cash_sized_shares == 100
    assert decision.final_shares == 12
    assert type(decision.final_shares) is int
    assert decision.sizing_reference_cash_required == 1_200.0
    assert decision.sizing_reference_cash_after_entry == 8_800.0
    assert decision.sizing_planned_risk_amount == pytest.approx(48.576)
    assert decision.sizing_planned_risk_fraction == pytest.approx(
        48.576 / 10_000
    )
    assert decision.unused_sizing_risk_budget == pytest.approx(1.424)
    assert decision.binding_constraint is PositionSizingConstraint.RISK
    assert decision.action is PositionSizingAction.SIZED
    assert isinstance(decision.sized_pending_entry, SizedPendingEntry)
    assert decision.sized_pending_entry.fixed_shares == 12
    assert decision.sized_pending_entry.cash_available == 10_000


def test_unadjusted_signal_close_not_phase8_adjusted_close_determines_q(
    service, make_completed_t
) -> None:
    signal, pending, signal_bar = make_completed_t(
        adjusted_close=75.0,
        unadjusted_close=100.0,
    )

    decision = _size(service, signal, pending, signal_bar)

    assert decision.sizing_reference_price == signal_bar.close == 100.0
    assert decision.sizing_reference_price != signal.adjusted_close
    assert decision.cash_sized_shares == 100


def test_cash_constraint_binds(service, make_completed_t) -> None:
    signal, pending, signal_bar = make_completed_t()

    decision = _size(service, signal, pending, signal_bar, cash=850)

    assert decision.risk_sized_shares == 12
    assert decision.cash_sized_shares == 8
    assert decision.final_shares == 8
    assert decision.sizing_reference_cash_required == 800.0
    assert decision.sizing_reference_cash_after_entry == 50.0
    assert decision.sizing_planned_risk_amount == pytest.approx(32.384)
    assert decision.binding_constraint is PositionSizingConstraint.CASH


def test_equal_constraints_bind_together(service, make_completed_t) -> None:
    signal, pending, signal_bar = make_completed_t()

    decision = _size(service, signal, pending, signal_bar, cash=1_200)

    assert decision.risk_sized_shares == 12
    assert decision.cash_sized_shares == 12
    assert decision.final_shares == 12
    assert decision.binding_constraint is PositionSizingConstraint.BOTH_EQUAL


def test_risk_too_small_skips_without_execution_order(
    service, make_completed_t
) -> None:
    signal, pending, signal_bar = make_completed_t()

    decision = _size(
        service, signal, pending, signal_bar, equity=100, cash=100
    )

    assert decision.risk_sized_shares == 0
    assert decision.cash_sized_shares == 1
    assert decision.action is PositionSizingAction.SKIPPED_RISK_TOO_SMALL
    assert decision.final_shares == 0
    assert decision.sizing_reference_cash_required == 0.0
    assert decision.sizing_reference_cash_after_entry == 100.0
    assert decision.sizing_planned_risk_amount == 0.0
    assert decision.sizing_planned_risk_fraction == 0.0
    assert decision.unused_sizing_risk_budget == (
        decision.sizing_target_risk_amount
    )
    assert decision.binding_constraint is None
    assert decision.sized_pending_entry is None


def test_insufficient_cash_skip_is_pre_execution_and_has_no_order(
    service, make_completed_t
) -> None:
    signal, pending, signal_bar = make_completed_t()

    decision = _size(service, signal, pending, signal_bar, cash=99)

    assert decision.risk_sized_shares == 12
    assert decision.cash_sized_shares == 0
    assert decision.action is PositionSizingAction.SKIPPED_INSUFFICIENT_CASH
    assert decision.final_shares == 0
    assert decision.sized_pending_entry is None


def test_both_zero_preserves_insufficient_cash_precedence(
    service, make_completed_t
) -> None:
    signal, pending, signal_bar = make_completed_t()

    decision = _size(
        service, signal, pending, signal_bar, equity=100, cash=99
    )

    assert decision.risk_sized_shares == 0
    assert decision.cash_sized_shares == 0
    assert decision.action is PositionSizingAction.SKIPPED_INSUFFICIENT_CASH


def test_risk_and_cash_ratios_floor_to_whole_shares(
    service, make_completed_t
) -> None:
    intended_loss = 50.0 / 12.99
    stop = (100.0 - intended_loss) / 0.9995
    atr_fraction = (1.0 - (stop / 100.0)) / 2.0
    risk_signal, risk_pending, risk_bar = make_completed_t(
        atr_fraction=atr_fraction
    )
    cash_signal, cash_pending, cash_bar = make_completed_t()

    risk_decision = _size(
        service, risk_signal, risk_pending, risk_bar
    )
    cash_decision = _size(
        service,
        cash_signal,
        cash_pending,
        cash_bar,
        equity=100_000,
        cash=1_299,
    )

    assert (
        risk_decision.sizing_target_risk_amount
        / risk_decision.sizing_loss_per_share
    ) == pytest.approx(12.99)
    assert risk_decision.risk_sized_shares == 12
    assert (
        cash_decision.cash_available / cash_decision.sizing_reference_price
    ) == 12.99
    assert cash_decision.cash_sized_shares == 12


@pytest.mark.parametrize("cash", [10_000, 850, 1_200])
def test_sized_results_obey_reference_cash_and_risk_invariants(
    service, make_completed_t, cash
) -> None:
    signal, pending, signal_bar = make_completed_t()

    decision = _size(service, signal, pending, signal_bar, cash=cash)

    assert decision.action is PositionSizingAction.SIZED
    assert decision.sizing_reference_cash_required <= decision.cash_available
    assert decision.sizing_reference_cash_after_entry >= 0.0
    assert (
        decision.sizing_planned_risk_amount
        <= decision.sizing_target_risk_amount
    )
    assert decision.sizing_planned_risk_fraction <= TARGET_RISK_FRACTION
    assert decision.final_shares == min(
        decision.risk_sized_shares, decision.cash_sized_shares
    )


@pytest.mark.parametrize("equity", [0, -1, nan, inf, -inf, True])
def test_invalid_portfolio_equity_fails_closed(equity) -> None:
    with pytest.raises(PositionSizingValidationError) as raised:
        PortfolioSizingSnapshot(portfolio_equity=equity, cash_available=0)

    assert raised.value.code == "INVALID_PORTFOLIO_EQUITY"


@pytest.mark.parametrize("cash", [-1, nan, inf, -inf, True])
def test_invalid_cash_available_fails_closed(cash) -> None:
    with pytest.raises(PositionSizingValidationError) as raised:
        PortfolioSizingSnapshot(portfolio_equity=10_000, cash_available=cash)

    assert raised.value.code == "INVALID_CASH_AVAILABLE"


def test_cash_cannot_exceed_equity() -> None:
    with pytest.raises(PositionSizingValidationError) as raised:
        PortfolioSizingSnapshot(portfolio_equity=100, cash_available=101)

    assert raised.value.code == "CASH_EXCEEDS_PORTFOLIO_EQUITY"


@pytest.mark.parametrize("atr_fraction", [None, 0.0, -0.01, nan, inf, -inf])
def test_invalid_signal_atr_fraction_fails_closed(
    service, make_completed_t, atr_fraction
) -> None:
    signal, pending, signal_bar = make_completed_t(
        atr_fraction=0.02
    )
    invalid_signal = replace(signal, atr_fraction=atr_fraction)

    with pytest.raises(PositionSizingValidationError) as raised:
        _size(service, invalid_signal, pending, signal_bar)

    assert raised.value.code == "INVALID_SIGNAL_ATR_FRACTION"


@pytest.mark.parametrize("atr_fraction", [0.5, 0.75, 1.0])
def test_invalid_sizing_risk_fraction_fails_closed(
    service, make_completed_t, atr_fraction
) -> None:
    signal, pending, signal_bar = make_completed_t(atr_fraction=0.02)
    invalid_signal = replace(signal, atr_fraction=atr_fraction)

    with pytest.raises(PositionSizingValidationError) as raised:
        _size(service, invalid_signal, pending, signal_bar)

    assert raised.value.code == "INVALID_SIZING_RISK_FRACTION"


def test_non_valid_signal_cannot_be_sized(service, make_completed_t) -> None:
    signal, pending, signal_bar = make_completed_t()
    invalid_signal = replace(signal, action=BaselineSignalAction.NO_SIGNAL)

    with pytest.raises(PositionSizingValidationError) as raised:
        _size(service, invalid_signal, pending, signal_bar)

    assert raised.value.code == "INVALID_SIGNAL"


@pytest.mark.parametrize(
    "field_name",
    [
        "security_id",
        "symbol",
        "signal_session",
        "signal_time",
        "planned_entry_session",
    ],
)
def test_signal_and_pending_identity_must_match(
    service, make_completed_t, field_name
) -> None:
    signal, pending, signal_bar = make_completed_t()
    if field_name in {"signal_session", "planned_entry_session"}:
        value = getattr(pending, field_name) + timedelta(days=1)
    elif field_name == "signal_time":
        value = pending.signal_time + timedelta(minutes=1)
    else:
        value = "OTHER"
    object.__setattr__(pending, field_name, value)

    with pytest.raises(PositionSizingValidationError) as raised:
        _size(service, signal, pending, signal_bar)

    assert raised.value.code == "SIGNAL_PENDING_ENTRY_IDENTITY_MISMATCH"


@pytest.mark.parametrize(
    ("updates", "mismatch"),
    [
        ({"security_id": "OTHER"}, "security_id"),
        ({"symbol": "OTHER"}, "symbol"),
        ({"trading_date": None}, "trading_date"),
        ({"timeframe": "H1"}, "timeframe"),
        ({"session_type": "extended"}, "session_type"),
        ({"price_basis": "capital_special_adjusted"}, "price_basis"),
    ],
)
def test_signal_bar_identity_and_domain_must_match(
    service, make_completed_t, updates, mismatch
) -> None:
    signal, pending, signal_bar = make_completed_t()
    resolved = dict(updates)
    if resolved.get("trading_date") is None:
        resolved["trading_date"] = signal.signal_session + timedelta(days=1)
    invalid_bar = signal_bar.model_copy(update=resolved)

    with pytest.raises(PositionSizingValidationError) as raised:
        _size(service, signal, pending, invalid_bar)

    assert raised.value.code == "SIGNAL_BAR_IDENTITY_MISMATCH"
    assert mismatch in str(raised.value)


@pytest.mark.parametrize("close", [0.0, -1.0, nan, inf, -inf])
def test_invalid_unadjusted_close_fails_closed(
    service, make_completed_t, close
) -> None:
    signal, pending, signal_bar = make_completed_t()
    invalid_bar = signal_bar.model_copy(update={"close": close})

    with pytest.raises(PositionSizingValidationError) as raised:
        _size(service, signal, pending, invalid_bar)

    assert raised.value.code == "INVALID_SIZING_REFERENCE_PRICE"


def test_t_plus_one_open_paths_cannot_mutate_or_change_q(
    make_sizing, execute_sizing
) -> None:
    sizing, _, pending, _, _ = make_sizing()
    before = sizing

    gap_down = execute_sizing(sizing, pending, opening_price=50.0)
    flat = execute_sizing(sizing, pending, opening_price=100.0)
    gap_up = execute_sizing(sizing, pending, opening_price=150.0)

    assert sizing is before
    assert sizing.final_shares == 12
    assert {
        gap_down.requested_shares,
        flat.requested_shares,
        gap_up.requested_shares,
    } == {12}
    assert {
        gap_down.executed_shares,
        flat.executed_shares,
        gap_up.executed_shares,
    } == {12}


def test_sizing_is_deterministic_and_mutates_no_input(
    service, make_completed_t
) -> None:
    signal, pending, signal_bar = make_completed_t()
    portfolio = PortfolioSizingSnapshot(10_000, 10_000)
    before = (
        signal,
        pending,
        signal_bar.model_dump(mode="python"),
        portfolio,
    )

    first = service.size_pending_entry(
        signal=signal,
        pending_entry=pending,
        signal_bar=signal_bar,
        portfolio=portfolio,
    )
    second = service.size_pending_entry(
        signal=signal,
        pending_entry=pending,
        signal_bar=signal_bar,
        portfolio=portfolio,
    )

    assert first == second
    assert (
        signal,
        pending,
        signal_bar.model_dump(mode="python"),
        portfolio,
    ) == before


def test_sizing_models_and_inputs_are_immutable(
    make_sizing
) -> None:
    sizing, _, _, signal_bar, portfolio = make_sizing()
    assert sizing.sized_pending_entry is not None

    with pytest.raises(FrozenInstanceError):
        portfolio.cash_available = 0.0
    with pytest.raises(FrozenInstanceError):
        sizing.final_shares = 1
    with pytest.raises(FrozenInstanceError):
        sizing.sized_pending_entry.fixed_shares = 1
    with pytest.raises(ValidationError):
        signal_bar.close = 1.0
    with pytest.raises(TypeError):
        PositionSizingDecision(security_id="SECURITY:1001")
