"""Production Phase 15C BUY/SELL event-adapter acceptance tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.backtester import (
    BacktestBuyExecutionEventAdapter,
    BacktestExecutionAdapterValidationError,
    BacktestSellExecutionEventAdapter,
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionInput,
)
from stock_swing_d1.execution.costs import (
    AdministrativeExitPricingService,
    BacktestExecutionCostService,
    ExecutionCostPolicyRef,
    ExecutionCostSide,
    ExecutionIdentifierService,
    HistoricalUsEquitySettlementResolver,
)
from stock_swing_d1.execution.entry import EntryExecutionService
from stock_swing_d1.execution.open_position_exit import (
    OpenPositionExitEvaluator,
    OpenPositionExitReason,
)
from stock_swing_d1.execution.protective_exit import ProtectiveExitState
from stock_swing_d1.models import StockBar
from stock_swing_d1.portfolio.portfolio_events import ExecutionSide
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PortfolioState,
)
from tests.backtester.conftest import FRIDAY, MONDAY, decision_time
from tests.backtester.test_phase15a_chronology import (
    EntryCalendar,
    EntryEarningsOverlay,
    entry_bar,
)
from tests.backtester.test_phase15b_exit_integration import (
    earnings_decision,
    exit_evaluation,
)
from tests.execution.open_position_exit.conftest import (
    ExplicitTradingCalendar,
    NEW_YORK,
    SESSIONS,
)


class ExplicitSettlementCalendar:
    """Explicit certified-session sequence; never infers weekdays."""

    def __init__(self, sessions: tuple[date, ...]) -> None:
        self.sessions = sessions

    def next_settlement_session(self, session: date) -> date:
        index = self.sessions.index(session)
        return self.sessions[index + 1]


class WrongFillCostService(BacktestExecutionCostService):
    """Valid service subtype that deliberately quotes a different input."""

    def quote(self, *, side, quantity, fill_price):
        return super().quote(
            side=side,
            quantity=quantity,
            fill_price=fill_price + Decimal("1"),
        )


def _buy_artifacts(
    *,
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service: BacktestExecutionCostService,
):
    ranking, candidate = make_candidate_pair("NORGATE:1001")
    allocation = HistoricalBacktestOrchestrator().process_session(
        prior_state=PortfolioState(settled_cash=Decimal("10000")),
        session_input=HistoricalBacktestSessionInput(
            session=FRIDAY,
            decision_time=decision_time(FRIDAY),
            next_session=MONDAY,
            ranking_candidates=(ranking,),
            allocation_candidates=(candidate,),
            allocation_portfolio=make_portfolio_snapshot(),
        ),
    )
    intent = allocation.future_entry_intents[0]
    sized = intent.candidate_decision.sized_pending_entry
    assert sized is not None
    decision = EntryExecutionService(
        earnings_overlay=EntryEarningsOverlay(),
        trading_calendar=EntryCalendar(),
        execution_cost_service=execution_cost_service,
    ).execute_pending_entry(
        pending_entry=intent.allocation_candidate.pending_entry,
        sized_pending_entry=sized,
        execution_bar=entry_bar(intent.security_id),
    )
    return intent, decision


def _position(
    *,
    asset_id: str = "NORGATE:1001",
    quantity: int = 2,
    entry_session: date = SESSIONS[0],
    entry_execution_id: str | None = None,
) -> OpenPosition:
    entry_price = Decimal("100")
    entry_cost = Decimal("1")
    execution_id = entry_execution_id or ExecutionIdentifierService().buy_execution_id(
        entry_session=entry_session,
        security_id=asset_id,
    )
    return OpenPosition(
        asset_id=asset_id,
        quantity=quantity,
        entry_session=entry_session,
        entry_price=entry_price,
        entry_execution_id=execution_id,
        entry_execution_cost=entry_cost,
        cost_basis=Decimal(quantity) * entry_price + entry_cost,
    )


def _sell_adapter(
    execution_cost_service: BacktestExecutionCostService,
    *,
    settlement_sessions: tuple[date, ...] = SESSIONS,
) -> BacktestSellExecutionEventAdapter:
    return BacktestSellExecutionEventAdapter(
        execution_cost_service=execution_cost_service,
        administrative_pricing_service=AdministrativeExitPricingService(
            policy=execution_cost_service.policy
        ),
        settlement_resolver=HistoricalUsEquitySettlementResolver(
            settlement_calendar=ExplicitSettlementCalendar(settlement_sessions)
        ),
        identifier_service=ExecutionIdentifierService(),
    )


def _protective_decision_for_trade_session(trade_session: date):
    entry_session = trade_session - timedelta(days=1)
    signal_session = entry_session - timedelta(days=1)
    calendar = ExplicitTradingCalendar((entry_session, trade_session))
    state = ProtectiveExitState._validated(
        security_id="NORGATE:1001",
        symbol="S1001",
        signal_session=signal_session,
        signal_time=datetime(
            signal_session.year,
            signal_session.month,
            signal_session.day,
            16,
            tzinfo=NEW_YORK,
        ),
        entry_session=entry_session,
        entry_price=100.0,
        signal_atr_fraction=0.02,
        risk_fraction=0.04,
        stop_price=96.0,
        take_profit_price=108.0,
        last_evaluated_session=entry_session,
    )
    bar = StockBar(
        security_id=state.security_id,
        symbol=state.symbol,
        trading_date=trade_session,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=100.0,
        high=101.0,
        low=95.0,
        close=100.0,
        volume=1_000_000,
    )
    from stock_swing_d1.execution.open_position_exit import (
        OpenPositionExitEvaluationInput,
    )

    decision = OpenPositionExitEvaluator(trading_calendar=calendar).evaluate(
        OpenPositionExitEvaluationInput(
            session=trade_session,
            protective_state=state,
            market_bar=bar,
        )
    )
    return decision, _position(entry_session=entry_session)


def test_buy_adapter_copies_exact_phase9_fill_cost_and_deterministic_ids(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    intent, decision = _buy_artifacts(
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        execution_cost_service=execution_cost_service,
    )
    adapter = BacktestBuyExecutionEventAdapter(
        execution_cost_service=execution_cost_service,
        identifier_service=ExecutionIdentifierService(),
    )

    first = adapter.build_buy_event(
        session=MONDAY,
        intent=intent,
        entry_execution=decision,
    )
    second = adapter.build_buy_event(
        session=MONDAY,
        intent=intent,
        entry_execution=decision,
    )

    assert first == second
    assert adapter.policy_ref == execution_cost_service.policy_ref
    assert first.side is ExecutionSide.BUY
    assert first.fill_price == Decimal(str(decision.execution_price))
    assert first.execution_cost == decision.execution_cost_quote.execution_cost
    assert first.execution_id == "ENTRY:2026-08-24:NORGATE:1001"
    assert first.source_order_id == "ALLOC:2026-08-21:NORGATE:1001"
    assert first.settlement_id is None
    assert first.settlement_session is None


def test_buy_adapter_requotes_and_rejects_a_different_valid_quote(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    intent, decision = _buy_artifacts(
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        execution_cost_service=execution_cost_service,
    )
    adapter = BacktestBuyExecutionEventAdapter(
        execution_cost_service=WrongFillCostService(
            policy=execution_cost_service.policy
        ),
        identifier_service=ExecutionIdentifierService(),
    )

    with pytest.raises(
        BacktestExecutionAdapterValidationError,
        match="BUY_COST_QUOTE_MISMATCH",
    ):
        adapter.build_buy_event(
            session=MONDAY,
            intent=intent,
            entry_execution=decision,
        )


def test_buy_adapter_rejects_wrong_policy_and_post_construction_corruption(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    intent, decision = _buy_artifacts(
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        execution_cost_service=execution_cost_service,
    )
    adapter = BacktestBuyExecutionEventAdapter(
        execution_cost_service=execution_cost_service,
        identifier_service=ExecutionIdentifierService(),
    )
    wrong_ref = ExecutionCostPolicyRef(
        policy_id=adapter.policy_ref.policy_id,
        policy_fingerprint="b" * 64,
    )
    wrong_quote = replace(decision.execution_cost_quote, policy_ref=wrong_ref)
    wrong_decision = replace(
        decision,
        candidate_execution_cost_quote=wrong_quote,
        execution_cost_quote=wrong_quote,
    )
    with pytest.raises(
        BacktestExecutionAdapterValidationError,
        match="INVALID_BUY_EXECUTION_INPUT",
    ):
        adapter.build_buy_event(
            session=MONDAY,
            intent=intent,
            entry_execution=wrong_decision,
        )

    object.__setattr__(decision, "execution_cost_quote", None)
    with pytest.raises(
        BacktestExecutionAdapterValidationError,
        match="INVALID_BUY_EXECUTION_INPUT",
    ):
        adapter.build_buy_event(
            session=MONDAY,
            intent=intent,
            entry_execution=decision,
        )


def test_adapters_fail_closed_when_bound_policy_reference_changes(
    make_candidate_pair,
    make_portfolio_snapshot,
    execution_cost_service,
) -> None:
    intent, decision = _buy_artifacts(
        make_candidate_pair=make_candidate_pair,
        make_portfolio_snapshot=make_portfolio_snapshot,
        execution_cost_service=execution_cost_service,
    )
    buy_adapter = BacktestBuyExecutionEventAdapter(
        execution_cost_service=execution_cost_service,
        identifier_service=ExecutionIdentifierService(),
    )
    sell_adapter = _sell_adapter(execution_cost_service)
    sell_decision = OpenPositionExitEvaluator(
        trading_calendar=ExplicitTradingCalendar()
    ).evaluate(exit_evaluation(low=95.0))
    changed_ref = ExecutionCostPolicyRef(
        policy_id=buy_adapter.policy_ref.policy_id,
        policy_fingerprint="c" * 64,
    )
    object.__setattr__(execution_cost_service, "policy_ref", changed_ref)

    with pytest.raises(
        BacktestExecutionAdapterValidationError,
        match="EXECUTION_COST_POLICY_CHANGED",
    ):
        buy_adapter.build_buy_event(
            session=MONDAY,
            intent=intent,
            entry_execution=decision,
        )
    with pytest.raises(
        BacktestExecutionAdapterValidationError,
        match="EXECUTION_COST_POLICY_CHANGED",
    ):
        sell_adapter.build_sell_event(
            decision=sell_decision,
            open_position=_position(),
        )


@pytest.mark.parametrize(
    ("expected_reason", "bar_values"),
    (
        (
            OpenPositionExitReason.GAP_THROUGH_STOP,
            {"open": 95.0, "high": 100.0, "low": 94.0, "close": 97.0},
        ),
        (
            OpenPositionExitReason.GAP_THROUGH_TARGET,
            {"open": 109.0, "high": 110.0, "low": 100.0, "close": 109.0},
        ),
        (
            OpenPositionExitReason.STOP_LOSS,
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0},
        ),
        (
            OpenPositionExitReason.TAKE_PROFIT,
            {"open": 100.0, "high": 109.0, "low": 97.0, "close": 108.0},
        ),
    ),
)
def test_protective_sell_preserves_each_phase10_price_exactly(
    expected_reason,
    bar_values,
    execution_cost_service,
) -> None:
    decision = OpenPositionExitEvaluator(
        trading_calendar=ExplicitTradingCalendar()
    ).evaluate(exit_evaluation(**bar_values))
    assert decision.selected_reason is expected_reason
    adapter = _sell_adapter(execution_cost_service)
    position = _position()

    event = adapter.build_sell_event(
        decision=decision,
        open_position=position,
    )

    expected_fill = Decimal(str(decision.final_execution_price))
    expected_cost = execution_cost_service.quote(
        side=ExecutionCostSide.SELL,
        quantity=position.quantity,
        fill_price=expected_fill,
    ).execution_cost
    assert event.fill_price == expected_fill
    assert event.execution_cost == expected_cost
    assert event.quantity == position.quantity


@pytest.mark.parametrize("reason", ("EARNINGS", "MAX_HOLDING"))
def test_administrative_sell_prices_once_then_quotes_cost(
    reason,
    execution_cost_service,
) -> None:
    if reason == "EARNINGS":
        current_index = 5
        evaluation = exit_evaluation(
            session_index=current_index,
            close=104.0,
            prior_boundary_earnings_decision=earnings_decision(
                asset_id="NORGATE:1001",
                boundary_index=current_index - 1,
                scheduled_index=current_index + 1,
            ),
        )
        expected_reason = OpenPositionExitReason.EARNINGS_FORCED_EXIT
    else:
        current_index = 9
        evaluation = exit_evaluation(
            session_index=current_index,
            close=103.0,
        )
        expected_reason = OpenPositionExitReason.MAX_HOLDING
    decision = OpenPositionExitEvaluator(
        trading_calendar=ExplicitTradingCalendar()
    ).evaluate(evaluation)
    assert decision.selected_reason is expected_reason
    assert decision.final_execution_price is None
    adapter = _sell_adapter(execution_cost_service)
    position = _position()

    event = adapter.build_sell_event(
        decision=decision,
        open_position=position,
    )

    expected_fill = Decimal(str(decision.reference_exit_price)) * Decimal(
        "0.9995"
    )
    expected_cost = execution_cost_service.quote(
        side=ExecutionCostSide.SELL,
        quantity=position.quantity,
        fill_price=expected_fill,
    ).execution_cost
    assert event.fill_price == expected_fill
    assert event.execution_cost == expected_cost
    assert event.settlement_id is not None
    assert event.settlement_session is not None


def test_sell_ids_derive_from_entry_identity_and_are_deterministic(
    execution_cost_service,
) -> None:
    decision = OpenPositionExitEvaluator(
        trading_calendar=ExplicitTradingCalendar()
    ).evaluate(exit_evaluation(low=95.0))
    position = _position()
    adapter = _sell_adapter(execution_cost_service)

    first = adapter.build_sell_event(decision=decision, open_position=position)
    second = adapter.build_sell_event(decision=decision, open_position=position)

    entry_id = "ENTRY:2026-08-03:NORGATE:1001"
    sell_id = f"EXIT:{entry_id}:2026-08-04:NORGATE:1001"
    assert first == second
    assert first.source_order_id == (
        f"EXIT-ORDER:{entry_id}:2026-08-04:NORGATE:1001"
    )
    assert first.execution_id == sell_id
    assert first.settlement_id == f"SETTLE:{sell_id}"


def test_sell_adapter_rejects_hold_and_noncanonical_legacy_entry_id(
    execution_cost_service,
) -> None:
    evaluator = OpenPositionExitEvaluator(
        trading_calendar=ExplicitTradingCalendar()
    )
    hold = evaluator.evaluate(exit_evaluation())
    adapter = _sell_adapter(execution_cost_service)
    with pytest.raises(
        BacktestExecutionAdapterValidationError,
        match="INVALID_SELL_EXECUTION_INPUT",
    ):
        adapter.build_sell_event(decision=hold, open_position=_position())

    terminal = evaluator.evaluate(exit_evaluation(low=95.0))
    with pytest.raises(
        BacktestExecutionAdapterValidationError,
        match="INVALID_SELL_EXECUTION_EVENT",
    ):
        adapter.build_sell_event(
            decision=terminal,
            open_position=_position(entry_execution_id="LEGACY-BUY-1"),
        )


@pytest.mark.parametrize(
    ("trade_session", "settlement_sessions", "expected_settlement"),
    (
        (
            date(2017, 9, 4),
            (
                date(2017, 9, 4),
                date(2017, 9, 5),
                date(2017, 9, 6),
                date(2017, 9, 7),
            ),
            date(2017, 9, 7),
        ),
        (
            date(2017, 9, 5),
            (date(2017, 9, 5), date(2017, 9, 6), date(2017, 9, 7)),
            date(2017, 9, 7),
        ),
        (
            date(2024, 5, 27),
            (date(2024, 5, 27), date(2024, 5, 28), date(2024, 5, 29)),
            date(2024, 5, 29),
        ),
        (
            date(2024, 5, 28),
            (date(2024, 5, 28), date(2024, 5, 29)),
            date(2024, 5, 29),
        ),
    ),
)
def test_historical_settlement_resolution_reaches_sell_event(
    trade_session,
    settlement_sessions,
    expected_settlement,
    execution_cost_service,
) -> None:
    decision, position = _protective_decision_for_trade_session(trade_session)
    adapter = _sell_adapter(
        execution_cost_service,
        settlement_sessions=settlement_sessions,
    )

    event = adapter.build_sell_event(
        decision=decision,
        open_position=position,
    )

    assert event.settlement_session == expected_settlement


def test_settlement_calendar_can_skip_weekend_and_holiday(
    execution_cost_service,
) -> None:
    trade_session = date(2026, 8, 28)
    decision, position = _protective_decision_for_trade_session(trade_session)
    adapter = _sell_adapter(
        execution_cost_service,
        settlement_sessions=(trade_session, date(2026, 9, 1)),
    )

    event = adapter.build_sell_event(
        decision=decision,
        open_position=position,
    )

    assert event.settlement_session == date(2026, 9, 1)


def test_pre_1995_sell_fails_closed(execution_cost_service) -> None:
    trade_session = date(1995, 6, 6)
    decision, position = _protective_decision_for_trade_session(trade_session)
    adapter = _sell_adapter(
        execution_cost_service,
        settlement_sessions=(trade_session, date(1995, 6, 7)),
    )

    with pytest.raises(
        BacktestExecutionAdapterValidationError,
        match="SELL_SETTLEMENT_RESOLUTION_FAILED",
    ) as captured:
        adapter.build_sell_event(
            decision=decision,
            open_position=position,
        )
    assert captured.value.__cause__ is not None


def test_adapter_scope_has_no_strategy_or_state_mutation_ownership() -> None:
    source = Path(
        "src/stock_swing_d1/backtester/execution_adapters.py"
    ).read_text(encoding="utf-8").lower()
    for forbidden in (
        "calculate_sma",
        "calculate_rsi",
        "calculate_atr",
        "rank_candidates",
        "allocate_ranked_candidates",
        "evaluate_signal",
        "transition(",
        "portfolioinvariantchecker",
    ):
        assert forbidden not in source


def test_phase15a_has_no_phase15c_formula_constants() -> None:
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "src/stock_swing_d1/backtester/orchestration.py",
            "src/stock_swing_d1/backtester/validation.py",
        )
    )
    for forbidden in (
        'Decimal("0.005")',
        'Decimal("0.01")',
        "fixed_cash_equivalent_bps_per_side",
        "commission_per_share_usd",
        'Decimal("0.9995")',
        "1995-06-07",
        "2017-09-05",
        "2024-05-28",
    ):
        assert forbidden not in source
