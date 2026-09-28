"""Task 5C-C Phase 15D: APPLIED/REPLAYED before classification, and pins.

REPLAYED presentations retain their legacy no-op provenance: they never
enter B, never consume application cardinality, never satisfy a terminal
entry-session linkage, and create no entry, exit or closed-trade record.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    HistoricalBacktestAuditResult,
    project_closed_trades,
    project_entries,
    project_execution_provenance,
    project_exits,
)
from stock_swing_d1.backtest_results.hashing import SOURCE_RUN_HASH_DOMAIN
from stock_swing_d1.backtest_results.models import (
    ExecutionApplicationStatus,
    ExecutionProvenanceSource,
)
from stock_swing_d1.backtest_results.persistence_schema import (
    BUNDLE_SCHEMA_VERSION,
)
from stock_swing_d1.backtest_results.source_validation import (
    _build_validated_source_context,
)
from stock_swing_d1.backtester import (
    HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION,
    HISTORICAL_BACKTEST_SESSION_RESULT_SCHEMA_VERSION,
    HistoricalBacktestRunResult,
    HistoricalBacktestSessionResult,
)
from stock_swing_d1.execution.open_position_exit import (
    OPEN_POSITION_EXIT_DECISION_SCHEMA_VERSION,
    OPEN_POSITION_EXIT_INPUT_SCHEMA_VERSION,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioEventKind,
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


ASSET = "NORGATE:902"
PRIOR = SOURCE_SESSION.fromordinal(SOURCE_SESSION.toordinal() - 1)


def _buy(execution_id: str, session):
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"ORDER-{execution_id}",
        session=session,
        asset_id=ASSET,
        side=ExecutionSide.BUY,
        quantity=2,
        fill_price=Decimal("100"),
        execution_cost=Decimal("1"),
    )


def _sell(execution_id: str, session, *, settlement_session):
    return PortfolioExecutionEvent(
        execution_id=execution_id,
        source_order_id=f"ORDER-{execution_id}",
        session=session,
        asset_id=ASSET,
        side=ExecutionSide.SELL,
        quantity=2,
        fill_price=Decimal("110"),
        execution_cost=Decimal("1"),
        settlement_id=f"SETTLE-{execution_id}",
        settlement_session=settlement_session,
    )


def _session_result(prior_state, session, events, transition):
    return HistoricalBacktestSessionResult(
        session=session,
        decision_time=source_decision_time(session),
        prior_state_fingerprint=hash_portfolio_state(prior_state),
        ordered_execution_events=events,
        state_transition_result=transition,
        authoritative_state=transition.resulting_state,
        ranking_snapshot=rank_candidates(
            ranking_session=session,
            decision_time=source_decision_time(session),
            candidates=(),
        ),
    )


def replayed_sell_with_new_buy_run() -> HistoricalBacktestRunResult:
    """History: BUY(X) on PRIOR, SELL(X) on SOURCE_SESSION (applied).  Then
    on SOURCE_ENTRY_SESSION the already-applied SELL(X) is presented again
    (REPLAYED) alongside a genuinely new BUY(X)."""

    original_buy = _buy("HIST-BUY", PRIOR)
    original_sell = _sell(
        "HIST-SELL", SOURCE_SESSION, settlement_session=SOURCE_ENTRY_SESSION
    )
    state_0 = PortfolioTransitionEngine.transition(
        PortfolioState(settled_cash=Decimal("1000")), PRIOR, (original_buy,)
    ).resulting_state
    state_1 = PortfolioTransitionEngine.transition(
        state_0, SOURCE_SESSION, (original_sell,)
    ).resulting_state
    new_buy = _buy("NEW-BUY", SOURCE_ENTRY_SESSION)
    # Phase 15A presentation order: replayed SELL keeps legacy rank 0.
    presented = (original_sell, new_buy)
    transition = PortfolioTransitionEngine.transition(
        state_1, SOURCE_ENTRY_SESSION, presented
    )
    session = _session_result(state_1, SOURCE_ENTRY_SESSION, presented, transition)
    return HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state_1,
        final_state=transition.resulting_state,
        session_results=(session,),
        initial_state_fingerprint=hash_portfolio_state(state_1),
        final_state_fingerprint=hash_portfolio_state(transition.resulting_state),
    )


def test_replayed_sell_alongside_a_new_buy_is_not_a_round_trip():
    run = replayed_sell_with_new_buy_run()
    (session,) = run.session_results
    transition = session.state_transition_result

    # Phase 13: the replayed SELL applied nothing; the new BUY survives.
    assert {reference.event_id for reference in transition.replayed_events} == {
        "HIST-SELL"
    }
    assert [
        reference.event_id
        for reference in transition.newly_applied_events
        if reference.event_kind is PortfolioEventKind.EXECUTION
    ] == ["NEW-BUY"]
    assert [position.asset_id for position in run.final_state.open_positions] == [
        ASSET
    ]

    # Phase 15D: status first, then classification -- the replayed SELL is
    # not in B, is not a round-trip exit, and creates no records.
    context = _build_validated_source_context(run_result=run, run_manifest=None)
    (links,) = context.execution_links
    by_id = {link.event.execution_id: link for link in links}
    assert by_id["HIST-SELL"].application_status is ExecutionApplicationStatus.REPLAYED
    assert by_id["HIST-SELL"].round_trip is False
    assert by_id["HIST-SELL"].exit_decision is None
    assert by_id["NEW-BUY"].application_status is ExecutionApplicationStatus.APPLIED
    assert by_id["NEW-BUY"].round_trip is False

    (entry,) = project_entries(run)
    assert entry.entry_execution_id == "NEW-BUY"
    assert entry.provenance_source is ExecutionProvenanceSource.EXTERNAL_SCHEDULED
    assert project_exits(run) == ()
    assert project_closed_trades(run) == ()
    provenance = project_execution_provenance(run)
    assert [(row.execution_id, row.event_order) for row in provenance] == [
        ("HIST-SELL", 0),
        ("NEW-BUY", 1),
    ]


def test_replayed_buy_and_sell_presentations_create_no_records():
    """Test 29: exact replays of an already-applied BUY/SELL pair on a later
    session keep their legacy no-op provenance end to end."""

    buy = _buy("PAIR-BUY", PRIOR)
    sell = _sell("PAIR-SELL", PRIOR, settlement_session=SOURCE_ENTRY_SESSION)
    state_0 = PortfolioTransitionEngine.transition(
        PortfolioState(settled_cash=Decimal("1000")), PRIOR, (buy, sell)
    ).resulting_state
    presented = (sell, buy)
    transition = PortfolioTransitionEngine.transition(
        state_0, SOURCE_SESSION, presented
    )
    session = _session_result(state_0, SOURCE_SESSION, presented, transition)
    run = HistoricalBacktestRunResult(
        decision_interval=SOURCE_DECISION_INTERVAL,
        initial_state=state_0,
        final_state=transition.resulting_state,
        session_results=(session,),
        initial_state_fingerprint=hash_portfolio_state(state_0),
        final_state_fingerprint=hash_portfolio_state(transition.resulting_state),
    )

    assert transition.ledger_entries == ()
    assert len(transition.replayed_events) == 2
    assert project_entries(run) == ()
    assert project_exits(run) == ()
    assert project_closed_trades(run) == ()
    context = _build_validated_source_context(run_result=run, run_manifest=None)
    (links,) = context.execution_links
    assert all(
        link.application_status is ExecutionApplicationStatus.REPLAYED
        and link.round_trip is False
        for link in links
    )


# ---------------------------------------------------------------------------
# Versioning pins (tests 38-43)
# ---------------------------------------------------------------------------


def test_source_run_hash_domain_is_the_frozen_5cb_v0_3_and_not_v0_4():
    assert SOURCE_RUN_HASH_DOMAIN == "historical_backtest_source_run.v0.3"
    assert not SOURCE_RUN_HASH_DOMAIN.endswith("v0.4")
    assert not SOURCE_RUN_HASH_DOMAIN.endswith("v0.2")


def test_session_result_and_phase15b_decision_final_pins():
    assert HISTORICAL_BACKTEST_SESSION_RESULT_SCHEMA_VERSION == (
        "historical_backtest_session_result.v0.3"
    )
    assert OPEN_POSITION_EXIT_DECISION_SCHEMA_VERSION == (
        "open_position_exit_decision.v0.2"
    )
    assert OPEN_POSITION_EXIT_INPUT_SCHEMA_VERSION == "open_position_exit_input.v0.1"


def test_unchanged_downstream_identities():
    from stock_swing_d1.backtest_results.models import (
        HISTORICAL_BACKTEST_AUDIT_RESULT_SCHEMA_VERSION,
        HISTORICAL_BACKTEST_RUN_MANIFEST_SCHEMA_VERSION,
    )
    from stock_swing_d1.baseline_experiment.hashing import (
        CANONICAL_BASELINE_EXPERIMENT_MANIFEST_HASH_DOMAIN,
        CANONICAL_BASELINE_EXPERIMENT_RESULT_HASH_DOMAIN,
    )
    from stock_swing_d1.portfolio.portfolio_events import (
        PORTFOLIO_EXECUTION_EVENT_SCHEMA_VERSION,
        PORTFOLIO_LEDGER_ENTRY_SCHEMA_VERSION,
    )
    from stock_swing_d1.portfolio.portfolio_state_models import (
        PORTFOLIO_STATE_SCHEMA_VERSION,
        PORTFOLIO_TRANSITION_RESULT_SCHEMA_VERSION,
    )
    from stock_swing_d1.research_metrics.hashing import (
        RESEARCH_METRICS_RESULT_HASH_DOMAIN,
    )
    from stock_swing_d1.research_metrics.models import (
        RESEARCH_METRICS_RESULT_SCHEMA_VERSION,
    )

    assert HISTORICAL_BACKTEST_RUN_RESULT_SCHEMA_VERSION == (
        "historical_backtest_run_result.v0.2"
    )
    assert PORTFOLIO_EXECUTION_EVENT_SCHEMA_VERSION == "portfolio_execution_event.v0.1"
    assert PORTFOLIO_LEDGER_ENTRY_SCHEMA_VERSION == "portfolio_ledger_entry.v0.1"
    assert PORTFOLIO_STATE_SCHEMA_VERSION == "portfolio_state.v0.1"
    assert PORTFOLIO_TRANSITION_RESULT_SCHEMA_VERSION == (
        "portfolio_transition_result.v0.1"
    )
    assert HISTORICAL_BACKTEST_RUN_MANIFEST_SCHEMA_VERSION == (
        "historical_backtest_run_manifest.v0.1"
    )
    assert HISTORICAL_BACKTEST_AUDIT_RESULT_SCHEMA_VERSION == (
        "historical_backtest_audit_result.v0.2"
    )
    assert BUNDLE_SCHEMA_VERSION == "historical_backtest_result_bundle.v0.6"
    assert RESEARCH_METRICS_RESULT_SCHEMA_VERSION == "research_metrics_result.v0.3"
    assert RESEARCH_METRICS_RESULT_HASH_DOMAIN == "research_metrics_result.v0.3"
    assert CANONICAL_BASELINE_EXPERIMENT_MANIFEST_HASH_DOMAIN == (
        "canonical_baseline_experiment_manifest.v0.2"
    )
    assert CANONICAL_BASELINE_EXPERIMENT_RESULT_HASH_DOMAIN == (
        "canonical_baseline_experiment_result.v0.2"
    )
    assert "entry_session_protective_decisions" not in (
        HistoricalBacktestAuditResult.model_fields
    )
