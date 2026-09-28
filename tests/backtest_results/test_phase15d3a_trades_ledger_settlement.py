"""Phase 15D.3A: trades, cash ledger, and settlement-lifecycle projection."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    ExecutionApplicationStatus,
    ExecutionProvenanceSource,
    HistoricalBacktestResultValidationError,
    HistoricalExitReason,
    HistoricalSettlementStatus,
    compute_source_payload_fingerprint,
    compute_source_run_fingerprint,
    project_cash_ledger,
    project_closed_trades,
    project_entries,
    project_exits,
    project_settlement_ledger,
)
from stock_swing_d1.backtest_results.derivation import (
    _HistoricalOpenTradeLink,
    _project_open_trade_links,
)
from stock_swing_d1.backtest_results.source_validation import (
    _ExecutionLink,
    _ValidatedSourceRunContext,
)
from stock_swing_d1.backtester import (
    HistoricalBacktestOrchestrator,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionInput,
)
from stock_swing_d1.execution.open_position_exit import OpenPositionExitEvaluator
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from stock_swing_d1.ranking import rank_candidates
from tests.backtest_results.conftest import (
    SOURCE_DECISION_INTERVAL,
    SOURCE_ENTRY_SESSION,
    SOURCE_SESSION,
    source_decision_time,
)
from tests.backtest_results.test_provenance import _corrupt_dataclass
from tests.backtester.test_phase15b_exit_integration import (
    buy_event,
    exit_evaluation,
    session_input,
    state_with_positions,
)
from tests.execution.open_position_exit.conftest import ExplicitTradingCalendar


def _session(session: date, **overrides) -> HistoricalBacktestSessionInput:
    values = {"session": session, "decision_time": source_decision_time(session)}
    values.update(overrides)
    return HistoricalBacktestSessionInput(**values)


def _external_buy(
    *,
    execution_id: str,
    session: date,
    asset_id: str,
    quantity: int = 1,
    fill_price: Decimal = Decimal("100"),
    execution_cost: Decimal = Decimal("1"),
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"{execution_id}-ORDER",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.BUY,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
    )


def _external_sell(
    *,
    execution_id: str,
    session: date,
    asset_id: str,
    quantity: int,
    fill_price: Decimal,
    execution_cost: Decimal,
    settlement_id: str,
    settlement_session: date,
) -> PortfolioExecutionEvent:
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"{execution_id}-ORDER",
        session=session,
        asset_id=asset_id,
        side=ExecutionSide.SELL,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
        settlement_id=settlement_id,
        settlement_session=settlement_session,
    )


# ---------------------------------------------------------------------------
# Entries: generated and external-scheduled APPLIED BUYs
# ---------------------------------------------------------------------------


def test_generated_applied_buy_produces_exact_entry(source_run_bundle) -> None:
    run, _manifest = source_run_bundle
    session = run.session_results[1]
    event = session.ordered_execution_events[0]
    decision = next(
        value
        for value in session.entry_execution_decisions
        if value.execution_price is not None
    )
    intent = next(
        item
        for item in run.session_results[0].future_entry_intents
        if item.security_id == event.asset_id
    )

    rows = project_entries(run)

    assert len(rows) == 1
    row = rows[0]
    assert row.entry_execution_id == event.execution_id
    assert row.source_order_id == event.source_order_id
    assert row.security_id == event.asset_id
    assert row.quantity == event.quantity
    assert row.entry_session == event.session
    assert row.fill_price == event.fill_price
    assert row.execution_cost == event.execution_cost
    assert row.cost_basis == (
        Decimal(event.quantity) * event.fill_price + event.execution_cost
    )
    assert row.provenance_source is ExecutionProvenanceSource.GENERATED_ENTRY
    assert row.signal_session == decision.signal_session
    assert row.signal_time == decision.signal_time
    assert row.allocation_session == intent.allocation_session
    assert row.source_rank == intent.source_rank
    assert row.ranking_snapshot_fingerprint == intent.ranking_snapshot_fingerprint
    assert row.ranking_input_fingerprint == (
        intent.candidate_decision.ranking_input_fingerprint
    )
    assert row.entry_execution_status is decision.status
    assert row.execution_cost_policy_fingerprint == (
        decision.execution_cost_quote.policy_ref.policy_fingerprint
    )


def test_external_scheduled_applied_buy_produces_exact_entry(
    external_execution_run,
) -> None:
    event = external_execution_run.session_results[0].ordered_execution_events[0]

    rows = project_entries(external_execution_run)

    assert len(rows) == 1
    row = rows[0]
    assert row.entry_execution_id == event.execution_id
    assert row.security_id == event.asset_id
    assert row.quantity == event.quantity
    assert row.fill_price == event.fill_price
    assert row.execution_cost == event.execution_cost
    assert row.cost_basis == (
        Decimal(event.quantity) * event.fill_price + event.execution_cost
    )
    assert row.provenance_source is ExecutionProvenanceSource.EXTERNAL_SCHEDULED
    assert row.signal_session is None
    assert row.allocation_session is None
    assert row.source_rank is None
    assert row.ranking_snapshot_fingerprint is None
    assert row.entry_execution_status is None
    assert row.execution_cost_policy_fingerprint is None


def test_replayed_buy_creates_no_entry_row(replay_execution_run) -> None:
    session = replay_execution_run.session_results[0]
    assert session.ordered_execution_events[0].execution_id == "REPLAY-BUY-1"

    assert project_entries(replay_execution_run) == ()
    assert project_cash_ledger(replay_execution_run) == ()


# ---------------------------------------------------------------------------
# Entry cost_basis authority: read from Phase 13 OpenPosition, not originated
# ---------------------------------------------------------------------------


def test_entry_cost_basis_is_read_from_authoritative_open_position(
    external_execution_run,
) -> None:
    session = external_execution_run.session_results[0]
    event = session.ordered_execution_events[0]
    position = next(
        item
        for item in session.authoritative_state.open_positions
        if item.asset_id == event.asset_id
    )

    rows = project_entries(external_execution_run)

    assert len(rows) == 1
    assert rows[0].cost_basis == position.cost_basis


def test_entry_cost_basis_reconciliation_identity_holds_exactly(
    source_run_bundle, external_execution_run
) -> None:
    for run in (source_run_bundle[0], external_execution_run):
        for row in project_entries(run):
            assert row.cost_basis == (
                Decimal(row.quantity) * row.fill_price + row.execution_cost
            )


def test_derivation_module_introduces_no_float_decimal_tolerance_or_rounding() -> None:
    import inspect

    from stock_swing_d1.backtest_results import derivation

    source = inspect.getsource(derivation)
    forbidden = (
        "Decimal(float",
        "round(",
        ".quantize(",
        "math.isclose",
        "pytest.approx",
        "rel_tol",
        "abs_tol",
    )
    for token in forbidden:
        assert token not in source


def _minimal_buy_session_context(
    *,
    event_quantity: int,
    position_quantity: int,
    corrupt_position_cost_basis: bool,
) -> tuple[HistoricalBacktestRunResult, _ValidatedSourceRunContext]:
    """Build a fake validated context isolating project_entries's own guard.

    `_build_validated_source_context` is airtight against every corruption
    reachable through public construction (proven by the fail-closed tests
    below), so this directly injects a hand-built `_ExecutionLink` whose
    event disagrees with the authoritative post-BUY `OpenPosition`, to unit
    test project_entries's own reconciliation check in isolation.
    """

    day = date(2026, 7, 6)
    virgin = PortfolioState(settled_cash=Decimal("10000"))
    buy_event_ = PortfolioExecutionEvent(
        execution_id="BUY-X",
        source_order_id="ORDER-X",
        session=day,
        asset_id="NORGATE:4001",
        side=ExecutionSide.BUY,
        quantity=position_quantity,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )
    transition = PortfolioTransitionEngine.transition(virgin, day, (buy_event_,))
    snapshot = rank_candidates(
        ranking_session=day, decision_time=source_decision_time(day), candidates=()
    )
    from stock_swing_d1.backtester.models import HistoricalBacktestSessionResult

    session = HistoricalBacktestSessionResult(
        session=day,
        decision_time=source_decision_time(day),
        prior_state_fingerprint=hash_portfolio_state(virgin),
        state_transition_result=transition,
        authoritative_state=transition.resulting_state,
        ranking_snapshot=snapshot,
    )
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=virgin,
        final_state=transition.resulting_state,
        session_results=(session,),
        initial_state_fingerprint=hash_portfolio_state(virgin),
        final_state_fingerprint=hash_portfolio_state(transition.resulting_state),
    )

    if corrupt_position_cost_basis:
        object.__setattr__(
            transition.resulting_state.open_positions[0],
            "cost_basis",
            Decimal("999999"),
        )

    linked_event = buy_event_.model_copy(update={"quantity": event_quantity})
    link = _ExecutionLink(
        event=linked_event,
        application_status=ExecutionApplicationStatus.APPLIED,
        provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
    )
    context = _ValidatedSourceRunContext(
        run_result=run,
        run_manifest=None,
        states_before=(virgin,),
        execution_links=((link,),),
    )
    return run, context


def test_entry_identity_mismatch_against_open_position_fails_closed(
    monkeypatch,
) -> None:
    run, context = _minimal_buy_session_context(
        event_quantity=5,
        position_quantity=3,
        corrupt_position_cost_basis=False,
    )
    monkeypatch.setattr(
        "stock_swing_d1.backtest_results.derivation._build_validated_source_context",
        lambda **_kwargs: context,
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="ENTRY_LINKAGE_MISMATCH"
    ):
        project_entries(run)


def test_entry_cost_basis_mismatch_against_open_position_fails_closed(
    monkeypatch,
) -> None:
    run, context = _minimal_buy_session_context(
        event_quantity=3,
        position_quantity=3,
        corrupt_position_cost_basis=True,
    )
    monkeypatch.setattr(
        "stock_swing_d1.backtest_results.derivation._build_validated_source_context",
        lambda **_kwargs: context,
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="ENTRY_COST_BASIS_MISMATCH"
    ):
        project_entries(run)


# ---------------------------------------------------------------------------
# Exits: generated and external-scheduled APPLIED SELLs
# ---------------------------------------------------------------------------


def test_generated_applied_sell_produces_exact_exit(generated_exit_run) -> None:
    session = generated_exit_run.session_results[0]
    event = session.ordered_execution_events[0]
    decision = session.open_position_exit_decisions[0]

    rows = project_exits(generated_exit_run)

    assert len(rows) == 1
    row = rows[0]
    assert row.exit_execution_id == event.execution_id
    assert row.entry_execution_id == "ENTRY-NORGATE:1001"
    assert row.security_id == event.asset_id
    assert row.quantity == event.quantity
    assert row.exit_session == event.session
    assert row.fill_price == event.fill_price
    assert row.execution_cost == event.execution_cost
    assert row.gross_proceeds == Decimal(event.quantity) * event.fill_price
    assert row.net_proceeds == row.gross_proceeds - event.execution_cost
    assert row.settlement_id == event.settlement_id
    assert row.settlement_session == event.settlement_session
    assert row.provenance_source is ExecutionProvenanceSource.GENERATED_EXIT
    assert row.exit_reason is HistoricalExitReason.STOP_LOSS
    assert row.reference_exit_price == Decimal(str(decision.reference_exit_price))


def test_external_scheduled_applied_sell_produces_exact_exit(
    same_session_sell_allocation_run,
) -> None:
    event = same_session_sell_allocation_run.session_results[0].ordered_execution_events[
        0
    ]
    assert event.execution_id == "BOUNDARY-SELL"

    rows = project_exits(same_session_sell_allocation_run)

    assert len(rows) == 1
    row = rows[0]
    assert row.exit_execution_id == "BOUNDARY-SELL"
    assert row.entry_execution_id == "BOUNDARY-BUY"
    assert row.provenance_source is ExecutionProvenanceSource.EXTERNAL_SCHEDULED
    assert row.exit_reason is HistoricalExitReason.EXTERNAL_SCHEDULED
    assert row.reference_exit_price is None
    assert row.gross_proceeds == Decimal("220")
    assert row.net_proceeds == Decimal("219")


def test_replayed_sell_creates_no_exit_or_trade_row() -> None:
    d0 = date(2026, 3, 2)
    d1 = date(2026, 3, 3)
    d2 = date(2026, 3, 4)
    settlement_session = date(2026, 3, 6)

    virgin = PortfolioState(settled_cash=Decimal("10000"))
    buy = _external_buy(execution_id="RSELL-BUY", session=d0, asset_id="NORGATE:7001")
    state_after_buy = PortfolioTransitionEngine.transition(
        virgin, d0, (buy,)
    ).resulting_state
    sell = _external_sell(
        execution_id="RSELL-SELL",
        session=d1,
        asset_id="NORGATE:7001",
        quantity=1,
        fill_price=Decimal("110"),
        execution_cost=Decimal("1"),
        settlement_id="RSELL-SETTLEMENT",
        settlement_session=settlement_session,
    )
    initial_state = PortfolioTransitionEngine.transition(
        state_after_buy, d1, (sell,)
    ).resulting_state

    replay_transition = PortfolioTransitionEngine.transition(
        initial_state, d2, (sell,)
    )
    assert replay_transition.ledger_entries == ()

    from stock_swing_d1.backtester.models import HistoricalBacktestSessionResult
    from stock_swing_d1.ranking import rank_candidates

    snapshot = rank_candidates(
        ranking_session=d2, decision_time=source_decision_time(d2), candidates=()
    )
    session = HistoricalBacktestSessionResult(
        session=d2,
        decision_time=source_decision_time(d2),
        prior_state_fingerprint=hash_portfolio_state(initial_state),
        ordered_execution_events=(sell,),
        state_transition_result=replay_transition,
        authoritative_state=replay_transition.resulting_state,
        ranking_snapshot=snapshot,
    )
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=initial_state,
        final_state=replay_transition.resulting_state,
        session_results=(session,),
        initial_state_fingerprint=hash_portfolio_state(initial_state),
        final_state_fingerprint=hash_portfolio_state(replay_transition.resulting_state),
    )

    assert project_exits(run) == ()
    assert project_closed_trades(run) == ()
    assert project_cash_ledger(run) == ()
    settlements = project_settlement_ledger(run)
    assert len(settlements) == 1
    assert settlements[0].settlement_id == "RSELL-SETTLEMENT"
    assert settlements[0].status is HistoricalSettlementStatus.PENDING_AT_RUN_END


# ---------------------------------------------------------------------------
# Trades: in-run closed, carried-in closed, and open linkage
# ---------------------------------------------------------------------------


@pytest.fixture
def in_run_closed_trade_run() -> HistoricalBacktestRunResult:
    entry_session = date(2026, 4, 6)
    exit_session = date(2026, 4, 7)
    buy = _external_buy(
        execution_id="RUN-BUY-1",
        session=entry_session,
        asset_id="NORGATE:8001",
        quantity=3,
        fill_price=Decimal("50"),
        execution_cost=Decimal("2"),
    )
    sell = _external_sell(
        execution_id="RUN-SELL-1",
        session=exit_session,
        asset_id="NORGATE:8001",
        quantity=3,
        fill_price=Decimal("55"),
        execution_cost=Decimal("2"),
        settlement_id="RUN-SETTLEMENT-1",
        settlement_session=date(2026, 4, 9),
    )
    return HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=Decimal("10000")),
        (
            _session(entry_session, scheduled_execution_events=(buy,)),
            _session(exit_session, scheduled_execution_events=(sell,)),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def test_in_run_buy_then_sell_is_one_closed_trade_not_carried_in(
    in_run_closed_trade_run,
) -> None:
    entries = project_entries(in_run_closed_trade_run)
    trades = project_closed_trades(in_run_closed_trade_run)

    assert len(entries) == 1
    assert len(trades) == 1
    trade = trades[0]
    assert trade.trade_id == "RUN-BUY-1"
    assert trade.trade_id == entries[0].entry_execution_id
    assert trade.carried_in is False
    assert trade.entry_execution_id == "RUN-BUY-1"
    assert trade.exit_execution_id == "RUN-SELL-1"
    assert trade.entry_fill_price == Decimal("50")
    assert trade.entry_execution_cost == Decimal("2")
    assert trade.entry_cost_basis == Decimal("152")
    assert trade.exit_fill_price == Decimal("55")
    assert trade.exit_execution_cost == Decimal("2")
    assert trade.gross_exit_proceeds == Decimal("165")
    assert trade.net_exit_proceeds == Decimal("163")
    assert trade.realized_pnl == Decimal("11")
    assert trade.exit_reason is HistoricalExitReason.EXTERNAL_SCHEDULED
    assert _project_open_trade_links(in_run_closed_trade_run) == ()


def test_carried_in_position_then_sell_is_closed_trade_without_synthetic_entry(
    generated_exit_run,
) -> None:
    trades = project_closed_trades(generated_exit_run)

    assert project_entries(generated_exit_run) == ()
    assert len(trades) == 1
    trade = trades[0]
    assert trade.carried_in is True
    assert trade.trade_id == "ENTRY-NORGATE:1001"
    assert trade.entry_execution_id == "ENTRY-NORGATE:1001"
    assert trade.entry_fill_price == Decimal("100")
    assert trade.entry_execution_cost == Decimal("1")
    assert trade.entry_cost_basis == Decimal("201")
    assert trade.exit_reason is HistoricalExitReason.STOP_LOSS


@pytest.fixture
def mixed_open_positions_run() -> HistoricalBacktestRunResult:
    initial = state_with_positions("NORGATE:1001")
    buy_new = _external_buy(
        execution_id="NEW-BUY-2002",
        session=session_input().session,
        asset_id="NORGATE:2002",
        quantity=1,
        fill_price=Decimal("250"),
        execution_cost=Decimal("2"),
    )
    orchestrator = HistoricalBacktestOrchestrator(
        open_position_exit_evaluator=OpenPositionExitEvaluator(
            trading_calendar=ExplicitTradingCalendar()
        ),
    )
    return orchestrator.run(
        initial,
        (
            session_input(
                open_position_exit_evaluations=(exit_evaluation(low=97.0),),
                scheduled_execution_events=(buy_new,),
            ),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def test_final_open_positions_are_linked_without_forced_liquidation(
    mixed_open_positions_run,
) -> None:
    assert mixed_open_positions_run.final_state.open_positions != ()
    assert project_closed_trades(mixed_open_positions_run) == ()
    assert project_exits(mixed_open_positions_run) == ()

    links = _project_open_trade_links(mixed_open_positions_run)
    assert len(links) == 2
    by_id = {link.security_id: link for link in links}

    carried = by_id["NORGATE:1001"]
    assert isinstance(carried, _HistoricalOpenTradeLink)
    assert carried.carried_in is True
    assert carried.entry_execution_id == "ENTRY-NORGATE:1001"
    assert carried.entry_fill_price == Decimal("100")
    assert carried.entry_cost_basis == Decimal("201")

    fresh = by_id["NORGATE:2002"]
    assert fresh.carried_in is False
    assert fresh.entry_execution_id == "NEW-BUY-2002"
    assert fresh.entry_fill_price == Decimal("250")
    assert fresh.entry_cost_basis == Decimal("252")


# ---------------------------------------------------------------------------
# Cash ledger: strict 1:1 projection and fingerprint domain
# ---------------------------------------------------------------------------


def test_cash_ledger_is_strict_1to1_projection_in_order(source_run_bundle) -> None:
    run, _manifest = source_run_bundle
    authoritative_entries = tuple(
        entry
        for session in run.session_results
        for entry in session.state_transition_result.ledger_entries
    )

    rows = project_cash_ledger(run)

    assert len(rows) == len(authoritative_entries)
    for row, entry in zip(rows, authoritative_entries, strict=True):
        assert row.session == entry.session
        assert row.sequence_in_session == entry.sequence_in_session
        assert row.ledger_event_type == entry.event_type
        assert row.source_event_id == entry.source_event_id
        assert row.source_order_id == entry.source_order_id
        assert row.security_id == entry.asset_id
        assert row.settled_cash_delta == entry.settled_cash_delta
        assert row.pending_cash_delta == entry.pending_cash_delta
        assert row.settled_cash_after == entry.settled_cash_after
        assert row.settlement_id == entry.settlement_id
        assert row.settlement_session == entry.settlement_session
        assert row.state_hash_before == entry.state_hash_before
        assert row.state_hash_after == entry.state_hash_after
    assert tuple((row.session, row.sequence_in_session) for row in rows) == tuple(
        sorted((row.session, row.sequence_in_session) for row in rows)
    )


def test_cash_ledger_fingerprint_uses_phase15d_domain_not_phase13_hash(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    entry = run.session_results[1].state_transition_result.ledger_entries[0]

    rows = project_cash_ledger(run)
    row = next(item for item in rows if item.source_event_id == entry.source_event_id)

    assert row.source_payload_fingerprint == compute_source_payload_fingerprint(
        source_type="portfolio_ledger_entry", payload=entry
    )
    assert row.source_payload_fingerprint != entry.source_payload_sha256


def test_cash_ledger_has_no_synthetic_opening_balance_row(
    same_session_sell_allocation_run,
) -> None:
    run = same_session_sell_allocation_run
    rows = project_cash_ledger(run)

    assert all(
        row.source_event_id in {"BOUNDARY-SELL"} for row in rows
    )
    assert len(rows) == len(
        run.session_results[0].state_transition_result.ledger_entries
    )


# ---------------------------------------------------------------------------
# Settlement lifecycle: all three statuses, no calendar reconstruction
# ---------------------------------------------------------------------------


@pytest.fixture
def settlement_lifecycle_run() -> HistoricalBacktestRunResult:
    d_buy_x = date(2026, 5, 4)
    d_sell_x = date(2026, 5, 5)
    s1 = date(2026, 5, 6)
    s2 = date(2026, 5, 7)
    s3 = date(2026, 5, 8)
    s5 = date(2026, 5, 12)

    virgin = PortfolioState(settled_cash=Decimal("100000"))
    buy_x = _external_buy(
        execution_id="BUY-X", session=d_buy_x, asset_id="NORGATE:9001"
    )
    state_a = PortfolioTransitionEngine.transition(
        virgin, d_buy_x, (buy_x,)
    ).resulting_state
    sell_x = _external_sell(
        execution_id="SELL-X",
        session=d_sell_x,
        asset_id="NORGATE:9001",
        quantity=1,
        fill_price=Decimal("110"),
        execution_cost=Decimal("1"),
        settlement_id="SETT-X",
        settlement_session=s1,
    )
    initial_state = PortfolioTransitionEngine.transition(
        state_a, d_sell_x, (sell_x,)
    ).resulting_state

    buy_y = _external_buy(
        execution_id="BUY-Y", session=s1, asset_id="NORGATE:9002", fill_price=Decimal("200")
    )
    buy_z = _external_buy(
        execution_id="BUY-Z", session=s1, asset_id="NORGATE:9003", fill_price=Decimal("300")
    )
    sell_y = _external_sell(
        execution_id="SELL-Y",
        session=s2,
        asset_id="NORGATE:9002",
        quantity=1,
        fill_price=Decimal("210"),
        execution_cost=Decimal("1"),
        settlement_id="SETT-Y",
        settlement_session=s3,
    )
    sell_z = _external_sell(
        execution_id="SELL-Z",
        session=s2,
        asset_id="NORGATE:9003",
        quantity=1,
        fill_price=Decimal("310"),
        execution_cost=Decimal("1"),
        settlement_id="SETT-Z",
        settlement_session=s5,
    )

    return HistoricalBacktestOrchestrator().run(
        initial_state,
        (
            _session(s1, scheduled_execution_events=(buy_y, buy_z)),
            _session(s2, scheduled_execution_events=(sell_y, sell_z)),
            _session(s3),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def test_settlement_lifecycle_covers_all_three_statuses(
    settlement_lifecycle_run,
) -> None:
    rows = project_settlement_ledger(settlement_lifecycle_run)
    by_id = {row.settlement_id: row for row in rows}

    assert set(by_id) == {"SETT-X", "SETT-Y", "SETT-Z"}
    assert by_id["SETT-X"].status is HistoricalSettlementStatus.CARRIED_IN_AND_SETTLED
    assert by_id["SETT-Y"].status is HistoricalSettlementStatus.SETTLED_DURING_RUN
    assert by_id["SETT-Z"].status is HistoricalSettlementStatus.PENDING_AT_RUN_END

    assert by_id["SETT-X"].source_sell_execution_id == "SELL-X"
    assert by_id["SETT-Y"].source_sell_execution_id == "SELL-Y"
    assert by_id["SETT-Z"].source_sell_execution_id == "SELL-Z"

    # deterministic (trade_session, settlement_session, settlement_id) order
    assert tuple(row.settlement_id for row in rows) == ("SETT-X", "SETT-Y", "SETT-Z")

    closed = project_closed_trades(settlement_lifecycle_run)
    assert {trade.trade_id for trade in closed} == {"BUY-Y", "BUY-Z"}
    assert _project_open_trade_links(settlement_lifecycle_run) == ()


def test_no_settlement_calendar_reconstruction_and_no_upstream_owner_calls() -> None:
    import inspect

    from stock_swing_d1.backtest_results import derivation

    source = inspect.getsource(derivation)
    forbidden = (
        "HistoricalUsEquitySettlementResolver",
        "SettlementSessionCalendar",
        "next_settlement_session",
        "SettlementPolicyModel",
        "PortfolioTransitionEngine.transition(",
        "rank_candidates(",
        "allocate_ranked_candidates(",
        "OpenPositionExitEvaluator(",
        "BacktestBuyExecutionEventAdapter(",
        "BacktestSellExecutionEventAdapter(",
        "AdministrativeExitPricingService(",
    )
    for token in forbidden:
        assert token not in source


def test_open_trade_handoff_is_private_and_absent_from_public_api() -> None:
    import stock_swing_d1.backtest_results as backtest_results_package

    for name in (
        "HistoricalOpenTradeLink",
        "_HistoricalOpenTradeLink",
        "project_open_trade_links",
        "_project_open_trade_links",
    ):
        assert name not in backtest_results_package.__all__
        assert not hasattr(backtest_results_package, name)

    assert _HistoricalOpenTradeLink.__name__.startswith("_")
    assert _project_open_trade_links.__name__.startswith("_")

    # The private helper remains fully functional for later Phase 15D.3B use
    # even though it is not part of the public backtest_results API.
    links = _project_open_trade_links(
        HistoricalBacktestRunResult(
            decision_interval=SOURCE_DECISION_INTERVAL,
            initial_state=PortfolioState(settled_cash=Decimal("1")),
            final_state=PortfolioState(settled_cash=Decimal("1")),
            initial_state_fingerprint=hash_portfolio_state(
                PortfolioState(settled_cash=Decimal("1"))
            ),
            final_state_fingerprint=hash_portfolio_state(
                PortfolioState(settled_cash=Decimal("1"))
            ),
        )
    )
    assert links == ()


# ---------------------------------------------------------------------------
# Fail-closed on corrupted linkage
# ---------------------------------------------------------------------------


def test_corrupted_generated_buy_linkage_fails_closed_for_entries(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    first, second = run.session_results[:2]
    event = second.ordered_execution_events[0]
    changed_event = event.model_copy(
        update={"fill_price": event.fill_price + Decimal("1")}
    )
    transition = PortfolioTransitionEngine.transition(
        first.authoritative_state, second.session, (changed_event,)
    )
    malformed_session = second.model_copy(
        update={
            "ordered_execution_events": (changed_event,),
            "state_transition_result": transition,
            "authoritative_state": transition.resulting_state,
        }
    )
    malformed = run.model_copy(
        update={
            "session_results": (first, malformed_session),
            "final_state": transition.resulting_state,
            "final_state_fingerprint": hash_portfolio_state(
                transition.resulting_state
            ),
        }
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="ENTRY_LINKAGE_MISMATCH"
    ):
        project_entries(malformed)


def test_corrupted_generated_sell_linkage_fails_closed_for_exits_and_trades(
    generated_exit_run,
) -> None:
    session = generated_exit_run.session_results[0]
    event = session.ordered_execution_events[0]
    changed_event = event.model_copy(
        update={"fill_price": event.fill_price + Decimal("1")}
    )
    transition = PortfolioTransitionEngine.transition(
        generated_exit_run.initial_state, session.session, (changed_event,)
    )
    malformed_session = session.model_copy(
        update={
            "ordered_execution_events": (changed_event,),
            "state_transition_result": transition,
            "authoritative_state": transition.resulting_state,
        }
    )
    malformed = generated_exit_run.model_copy(
        update={
            "session_results": (malformed_session,),
            "final_state": transition.resulting_state,
            "final_state_fingerprint": hash_portfolio_state(
                transition.resulting_state
            ),
        }
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="EXIT_LINKAGE_MISMATCH"
    ):
        project_exits(malformed)
    with pytest.raises(
        HistoricalBacktestResultValidationError, match="EXIT_LINKAGE_MISMATCH"
    ):
        project_closed_trades(malformed)


def test_allocation_ranking_corruption_fails_closed_for_entries(
    source_run_bundle,
) -> None:
    run, _manifest = source_run_bundle
    first = run.session_results[0]
    allocation = first.allocation_decision
    assert allocation is not None
    bad_candidate = _corrupt_dataclass(
        allocation.candidate_decisions[0],
        ranking_input_fingerprint="e" * 64,
    )
    bad_allocation = _corrupt_dataclass(
        allocation,
        candidate_decisions=(bad_candidate, *allocation.candidate_decisions[1:]),
    )
    malformed_session = first.model_copy(
        update={"allocation_decision": bad_allocation}
    )
    malformed = run.model_copy(
        update={"session_results": (malformed_session, *run.session_results[1:])}
    )

    with pytest.raises(
        HistoricalBacktestResultValidationError, match="ARTIFACT_PROVENANCE_MISMATCH"
    ):
        project_entries(malformed)


def test_missing_pending_settlement_for_settlement_applied_row_fails_closed() -> None:
    d0 = date(2026, 6, 2)
    d1 = date(2026, 6, 3)
    d2 = date(2026, 6, 5)

    virgin = PortfolioState(settled_cash=Decimal("10000"))
    buy = _external_buy(execution_id="MSS-BUY", session=d0, asset_id="NORGATE:5001")
    state_after_buy = PortfolioTransitionEngine.transition(
        virgin, d0, (buy,)
    ).resulting_state
    sell = _external_sell(
        execution_id="MSS-SELL",
        session=d1,
        asset_id="NORGATE:5001",
        quantity=1,
        fill_price=Decimal("110"),
        execution_cost=Decimal("1"),
        settlement_id="MSS-SETTLEMENT",
        settlement_session=d2,
    )
    after_sell = PortfolioTransitionEngine.transition(
        state_after_buy, d1, (sell,)
    ).resulting_state
    settle_transition = PortfolioTransitionEngine.transition(after_sell, d2, ())
    assert len(settle_transition.ledger_entries) == 1

    corrupted_entry = settle_transition.ledger_entries[0].model_copy(
        update={"source_event_id": "SOME-OTHER-SETTLEMENT-ID"}
    )
    corrupted_transition = settle_transition.model_copy(
        update={"ledger_entries": (corrupted_entry,)}
    )

    from stock_swing_d1.backtester.models import HistoricalBacktestSessionResult
    from stock_swing_d1.ranking import rank_candidates

    snapshot = rank_candidates(
        ranking_session=d2, decision_time=source_decision_time(d2), candidates=()
    )
    session = HistoricalBacktestSessionResult(
        session=d2,
        decision_time=source_decision_time(d2),
        prior_state_fingerprint=hash_portfolio_state(after_sell),
        state_transition_result=corrupted_transition,
        authoritative_state=corrupted_transition.resulting_state,
        ranking_snapshot=snapshot,
    )
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=after_sell,
        final_state=corrupted_transition.resulting_state,
        session_results=(session,),
        initial_state_fingerprint=hash_portfolio_state(after_sell),
        final_state_fingerprint=hash_portfolio_state(
            corrupted_transition.resulting_state
        ),
    )

    with pytest.raises(HistoricalBacktestResultValidationError):
        project_settlement_ledger(run)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_all_new_projections_are_deterministic_and_read_only(
    source_run_bundle, generated_exit_run, settlement_lifecycle_run
) -> None:
    for run in (source_run_bundle[0], generated_exit_run, settlement_lifecycle_run):
        source_fingerprint = compute_source_run_fingerprint(run)
        for projection in (
            project_entries,
            project_exits,
            project_closed_trades,
            _project_open_trade_links,
            project_cash_ledger,
            project_settlement_ledger,
        ):
            assert projection(run) == projection(run)
        assert compute_source_run_fingerprint(run) == source_fingerprint
