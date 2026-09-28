"""Task 5B: prove the declared risk geometry is the geometry Phase 10 applies.

The stop multiple and take-profit R multiple are inline literals inside the
frozen Phase 10 owner, and ``strategy`` cannot import ``execution`` without
creating a cycle, so the canonical configuration declares them. This module is
the mechanical drift guard for that declaration: it drives the real
:class:`ProtectiveExitService` and rejects any divergence between the levels it
produces and the levels the declared multiples imply.

Changing ``2.0`` to any other value in the Phase 10 service fails these tests.
"""

from __future__ import annotations

import pytest

from stock_swing_d1.strategy.baseline import (
    build_baseline_strategy_configuration,
)


@pytest.fixture
def risk_geometry():
    return build_baseline_strategy_configuration().risk


@pytest.mark.parametrize("atr_fraction", [0.01, 0.02, 0.035, 0.125])
@pytest.mark.parametrize("execution_price", [10.0, 100.0, 237.75])
def test_declared_multiples_reproduce_phase10_protective_levels(
    service, make_signal, make_entry, risk_geometry, atr_fraction, execution_price
) -> None:
    entry = make_entry(execution_price=execution_price)
    state = service.create_state(
        signal=make_signal(atr_fraction=atr_fraction),
        entry_execution=entry,
    )

    entry_price = entry.execution_price
    expected_risk_fraction = risk_geometry.stop_risk_atr_multiple * atr_fraction
    expected_stop = entry_price * (1.0 - expected_risk_fraction)
    expected_take_profit = entry_price + risk_geometry.take_profit_r_multiple * (
        entry_price - expected_stop
    )

    assert state.signal_atr_fraction == atr_fraction
    assert state.risk_fraction == expected_risk_fraction
    assert state.stop_price == expected_stop
    assert state.take_profit_price == expected_take_profit


def test_declared_stop_multiple_is_the_only_one_that_reproduces_phase10(
    service, make_signal, make_entry, risk_geometry
) -> None:
    """A different stop multiple must not reproduce the authoritative stop."""

    atr_fraction = 0.02
    entry = make_entry(execution_price=100.0)
    state = service.create_state(
        signal=make_signal(atr_fraction=atr_fraction),
        entry_execution=entry,
    )
    entry_price = entry.execution_price

    for wrong_multiple in (1.0, 1.5, 2.5, 3.0):
        assert wrong_multiple != risk_geometry.stop_risk_atr_multiple
        assert state.stop_price != entry_price * (
            1.0 - wrong_multiple * atr_fraction
        )


def test_declared_take_profit_multiple_is_the_only_one_that_reproduces_phase10(
    service, make_signal, make_entry, risk_geometry
) -> None:
    entry = make_entry(execution_price=100.0)
    state = service.create_state(
        signal=make_signal(atr_fraction=0.02), entry_execution=entry
    )
    entry_price = entry.execution_price
    initial_risk = entry_price - state.stop_price

    for wrong_multiple in (1.0, 1.5, 2.5, 3.0):
        assert wrong_multiple != risk_geometry.take_profit_r_multiple
        assert state.take_profit_price != (
            entry_price + wrong_multiple * initial_risk
        )
