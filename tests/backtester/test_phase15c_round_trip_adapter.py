"""Task 5C-C: Phase 15C transient exposure and round-trip SELL adaptation.

The round-trip path reuses the one authoritative pricing / cost / settlement
/ identifier pipeline of the SELL adapter: the fill is exactly the Phase 10
final execution price (no second slippage), the cost quote is Phase 15C's,
settlement resolves from T, and the SELL identity derives from the BUY's
entry execution ID through the existing identifier service.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtester import (
    BacktestExecutionAdapterValidationError,
    TransientEntryExposure,
)
from stock_swing_d1.execution.costs import (
    ExecutionCostSide,
    ExecutionIdentifierService,
)
from stock_swing_d1.execution.open_position_exit import (
    ExitBoundary,
    ExitPrerequisiteStatus,
    OpenPositionExitEvaluationInput,
    OpenPositionExitEvaluator,
    OpenPositionExitReason,
)
from stock_swing_d1.execution.protective_exit import ProtectiveExitState
from stock_swing_d1.models import StockBar
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_state_models import OpenPosition

from tests.backtester.test_phase15b_exit_integration import (
    earnings_decision,
    exit_evaluation,
)
from tests.backtester.test_phase15c_execution_adapters import _sell_adapter
from tests.execution.open_position_exit.conftest import (
    ExplicitTradingCalendar,
    SESSIONS,
)


ASSET = "NORGATE:1001"
T = SESSIONS[1]
BUY_ID = f"ENTRY:{T.isoformat()}:{ASSET}"


def buy_event(*, quantity: int = 2, fill_price: Decimal = Decimal("100")):
    return PortfolioExecutionEvent(
        execution_id=BUY_ID,
        source_order_id=f"ALLOC:{SESSIONS[0].isoformat()}:{ASSET}",
        session=T,
        asset_id=ASSET,
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=Decimal("1"),
    )


def entry_session_decision(**bar_values):
    """A terminal Phase 15B decision on the entry session itself."""

    evaluation = exit_evaluation(
        ASSET, session_index=1, entry_session=T, **bar_values
    )
    return OpenPositionExitEvaluator(
        trading_calendar=ExplicitTradingCalendar()
    ).evaluate(evaluation)


# ---------------------------------------------------------------------------
# TransientEntryExposure
# ---------------------------------------------------------------------------


def test_exposure_is_derived_solely_from_the_canonical_buy_event() -> None:
    exposure = TransientEntryExposure.from_buy_event(buy_event())

    assert exposure == TransientEntryExposure(
        security_id=ASSET,
        quantity=2,
        entry_session=T,
        entry_execution_id=BUY_ID,
        entry_fill_price=Decimal("100"),
    )
    # Exactly the frozen fact set: no cost basis, P&L, or Phase 13 state.
    assert set(TransientEntryExposure.__dataclass_fields__) == {
        "security_id",
        "quantity",
        "entry_session",
        "entry_execution_id",
        "entry_fill_price",
    }
    assert not isinstance(exposure, OpenPosition)


def test_exposure_rejects_a_sell_event_and_noncanonical_values() -> None:
    sell = buy_event().model_copy(
        update={
            "side": ExecutionSide.SELL,
            "settlement_id": "S",
            "settlement_session": SESSIONS[2],
        }
    )
    with pytest.raises(BacktestExecutionAdapterValidationError) as error:
        TransientEntryExposure.from_buy_event(sell)
    assert error.value.code == "INVALID_TRANSIENT_ENTRY_EXPOSURE"

    for values in (
        {"quantity": 0},
        {"entry_fill_price": 100.0},
        {"entry_fill_price": Decimal("0")},
        {"security_id": ""},
        {"entry_execution_id": ""},
        {"entry_session": "2026-08-04"},
    ):
        with pytest.raises(BacktestExecutionAdapterValidationError):
            TransientEntryExposure(
                **{
                    "security_id": ASSET,
                    "quantity": 2,
                    "entry_session": T,
                    "entry_execution_id": BUY_ID,
                    "entry_fill_price": Decimal("100"),
                    **values,
                }
            )


# ---------------------------------------------------------------------------
# build_round_trip_sell_event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected_reason", "bar_values"),
    [
        (OpenPositionExitReason.STOP_LOSS, {"low": 95.0}),
        (OpenPositionExitReason.TAKE_PROFIT, {"high": 109.0, "low": 99.0}),
        (OpenPositionExitReason.STOP_LOSS, {"high": 109.0, "low": 95.0}),
    ],
)
def test_round_trip_sell_uses_the_phase10_final_price_once(
    expected_reason, bar_values, execution_cost_service
) -> None:
    decision = entry_session_decision(**bar_values)
    assert decision.selected_reason is expected_reason
    assert decision.exit_boundary is ExitBoundary.INTRADAY
    assert decision.holding_session_number == 1
    assert decision.exit_prerequisite_status is ExitPrerequisiteStatus.READY
    adapter = _sell_adapter(execution_cost_service)
    exposure = TransientEntryExposure.from_buy_event(buy_event())

    event = adapter.build_round_trip_sell_event(
        decision=decision, exposure=exposure
    )

    expected_fill = Decimal(str(decision.final_execution_price))
    expected_cost = execution_cost_service.quote(
        side=ExecutionCostSide.SELL, quantity=2, fill_price=expected_fill
    ).execution_cost
    assert event.side is ExecutionSide.SELL
    assert event.session == T
    assert event.asset_id == ASSET
    assert event.quantity == exposure.quantity
    # Phase 10 already applied protective slippage: no second application.
    assert event.fill_price == expected_fill
    assert event.fill_price != Decimal(str(decision.reference_exit_price))
    assert event.execution_cost == expected_cost
    assert event.settlement_session is not None and event.settlement_session > T
    identifiers = ExecutionIdentifierService()
    assert event.execution_id == identifiers.sell_execution_id(
        entry_execution_id=BUY_ID, exit_session=T, security_id=ASSET
    )
    assert event.source_order_id == identifiers.sell_source_order_id(
        entry_execution_id=BUY_ID, exit_session=T, security_id=ASSET
    )
    assert event.settlement_id == identifiers.settlement_id(
        sell_execution_id=event.execution_id
    )
    assert event.execution_id != BUY_ID


def test_round_trip_and_prior_position_paths_share_one_pipeline(
    execution_cost_service,
) -> None:
    """The same terminal decision adapted through the OpenPosition path and
    the exposure path yields the identical event: one pipeline, no
    duplicated economics."""

    decision = entry_session_decision(low=95.0)
    adapter = _sell_adapter(execution_cost_service)
    position = OpenPosition(
        asset_id=ASSET,
        quantity=2,
        entry_session=T,
        entry_price=Decimal("100"),
        entry_execution_id=BUY_ID,
        entry_execution_cost=Decimal("1"),
        cost_basis=Decimal("201"),
    )

    via_position = adapter.build_sell_event(
        decision=decision, open_position=position
    )
    via_exposure = adapter.build_round_trip_sell_event(
        decision=decision, exposure=TransientEntryExposure.from_buy_event(buy_event())
    )

    assert via_position == via_exposure


def test_round_trip_rejects_administrative_reasons(execution_cost_service) -> None:
    # An administrative close (earnings) on a later session is a valid
    # prior-position exit but never a same-session round trip.
    evaluation = exit_evaluation(
        ASSET,
        session_index=5,
        close=104.0,
        prior_boundary_earnings_decision=earnings_decision(
            asset_id=ASSET, boundary_index=4, scheduled_index=6
        ),
    )
    decision = OpenPositionExitEvaluator(
        trading_calendar=ExplicitTradingCalendar()
    ).evaluate(evaluation)
    assert decision.selected_reason is OpenPositionExitReason.EARNINGS_FORCED_EXIT
    adapter = _sell_adapter(execution_cost_service)
    exposure = TransientEntryExposure(
        security_id=ASSET,
        quantity=2,
        entry_session=SESSIONS[0],
        entry_execution_id=f"ENTRY:{SESSIONS[0].isoformat()}:{ASSET}",
        entry_fill_price=Decimal("100"),
    )

    with pytest.raises(BacktestExecutionAdapterValidationError) as error:
        adapter.build_round_trip_sell_event(decision=decision, exposure=exposure)
    assert error.value.code == "INVALID_ROUND_TRIP_SELL_INPUT"


@pytest.mark.parametrize(
    "exposure_update",
    [
        {"security_id": "NORGATE:9999"},
        {"entry_session": SESSIONS[0]},
        {"entry_fill_price": Decimal("101")},
    ],
)
def test_round_trip_rejects_exposure_that_disagrees_with_the_decision(
    exposure_update, execution_cost_service
) -> None:
    decision = entry_session_decision(low=95.0)
    adapter = _sell_adapter(execution_cost_service)
    exposure = replace(
        TransientEntryExposure.from_buy_event(buy_event()), **exposure_update
    )

    with pytest.raises(BacktestExecutionAdapterValidationError) as error:
        adapter.build_round_trip_sell_event(decision=decision, exposure=exposure)
    assert error.value.code == "INVALID_ROUND_TRIP_SELL_INPUT"


def test_round_trip_rejects_a_hold_decision_and_a_foreign_exposure_type(
    execution_cost_service,
) -> None:
    hold = entry_session_decision(low=99.0, high=101.0)
    assert hold.exit_required is False
    adapter = _sell_adapter(execution_cost_service)
    exposure = TransientEntryExposure.from_buy_event(buy_event())

    with pytest.raises(BacktestExecutionAdapterValidationError):
        adapter.build_round_trip_sell_event(decision=hold, exposure=exposure)
    with pytest.raises(BacktestExecutionAdapterValidationError):
        adapter.build_round_trip_sell_event(
            decision=entry_session_decision(low=95.0),
            exposure=OpenPosition(
                asset_id=ASSET,
                quantity=2,
                entry_session=T,
                entry_price=Decimal("100"),
                entry_execution_id=BUY_ID,
                entry_execution_cost=Decimal("1"),
                cost_basis=Decimal("201"),
            ),
        )
