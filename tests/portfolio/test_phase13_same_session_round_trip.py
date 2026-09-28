"""Task 5C-C: same-session BUY -> ROUND_TRIP SELL inside one Phase 13 transition.

Frozen application chronology for distinct new applications:

    SETTLEMENT -> PRIOR SELL -> BUY -> ROUND_TRIP SELL -> DIVIDEND

with the shared P/B classifier (P = immutable session-start positions, B =
distinct new APPLIED BUY assets), per-asset cardinality (at most one applied
BUY and one applied SELL), input-order invariance, and identical live /
persisted-replay semantics.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from itertools import permutations
from pathlib import Path

import pytest

from stock_swing_d1.portfolio.backtest_orchestration import (
    PortfolioBacktestOrchestrator,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    DividendApplicationStatus,
    add_exact_decimal,
    subtract_exact_decimal,
)
from stock_swing_d1.portfolio.portfolio_errors import (
    DuplicateAssetApplicationError,
    PortfolioStateError,
    PositionNotFoundError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    PortfolioEventKind,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_invariants import (
    BUY_APPLICATION_RANK,
    PRIOR_SELL_APPLICATION_RANK,
    ROUND_TRIP_SELL_APPLICATION_RANK,
    SETTLEMENT_APPLICATION_RANK,
    PortfolioInvariantChecker,
    classify_session_applications,
)
from stock_swing_d1.portfolio.portfolio_persistence import (
    PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY,
    PORTFOLIO_MANIFEST_SCHEMA_VERSION,
    PortfolioArtifactReader,
    PortfolioArtifactWriter,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    PortfolioSessionInput,
    PortfolioSessionSnapshot,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_transition import (
    PortfolioTransitionEngine,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from tests.portfolio.phase13_helpers import (
    buy_event,
    initial_state,
    sell_event,
)
from tests.portfolio.test_phase13_dividend_persistence import (
    make_dividend_evidence,
)


T0 = date(2026, 8, 17)
T1 = date(2026, 8, 18)
T2 = date(2026, 8, 19)
SETTLE = date(2026, 8, 21)
X = "NORGATE:1"
Y = "NORGATE:2"
Z = "NORGATE:3"

C = Decimal("1000")
Q = 3
PB = Decimal("10")
CB = Decimal("1")
PS = Decimal("9")
CS = Decimal("1")


def round_trip_pair(
    asset: str = X, *, session: date = T1
) -> tuple[object, object]:
    buy = buy_event(
        f"ENTRY:{session.isoformat()}:{asset}",
        session=session,
        asset_id=asset,
        quantity=Q,
        fill_price=PB,
        execution_cost=CB,
    )
    sell = sell_event(
        f"EXIT:ENTRY:{session.isoformat()}:{asset}:{session.isoformat()}:{asset}",
        session=session,
        asset_id=asset,
        quantity=Q,
        fill_price=PS,
        execution_cost=CS,
        settlement_id=f"SETTLE:EXIT:{asset}:{session.isoformat()}",
        settlement_session=SETTLE,
    )
    return buy, sell


# ---------------------------------------------------------------------------
# The shared classifier
# ---------------------------------------------------------------------------


def test_classifier_prior_round_trip_and_invalid_are_pure_set_functions() -> None:
    classification = classify_session_applications(
        session_start_asset_ids=frozenset({X}),
        buy_asset_ids=(Y, Z),
        sell_asset_ids=(X, Y),
    )
    assert classification.is_valid
    assert classification.sell_rank_by_asset == {
        X: PRIOR_SELL_APPLICATION_RANK,
        Y: ROUND_TRIP_SELL_APPLICATION_RANK,
    }
    assert (
        SETTLEMENT_APPLICATION_RANK
        < PRIOR_SELL_APPLICATION_RANK
        < BUY_APPLICATION_RANK
        < ROUND_TRIP_SELL_APPLICATION_RANK
    )


def test_classifier_reports_unclassifiable_and_duplicate_applications() -> None:
    classification = classify_session_applications(
        session_start_asset_ids=frozenset(),
        buy_asset_ids=(Y, Y),
        sell_asset_ids=(X, Z, Z),
    )
    assert not classification.is_valid
    assert classification.duplicate_buy_asset_ids == (Y,)
    assert classification.duplicate_sell_asset_ids == (Z,)
    assert classification.unclassifiable_sell_asset_ids == (X, Z)


def test_new_sell_with_empty_p_and_empty_b_fails_closed() -> None:
    sell = sell_event("SELL-X", session=T1, asset_id=X)
    with pytest.raises(PositionNotFoundError):
        PortfolioTransitionEngine.transition(initial_state(C), T1, (sell,))


# ---------------------------------------------------------------------------
# Same-session round trip economics
# ---------------------------------------------------------------------------


def test_same_session_round_trip_applies_buy_then_sell_in_one_transition() -> None:
    buy, sell = round_trip_pair()

    result = PortfolioTransitionEngine.transition(initial_state(C), T1, (buy, sell))

    state = result.resulting_state
    buy_cash = Q * PB + CB
    net_proceeds = Q * PS - CS
    # Cash: BUY consumed settled cash; SELL proceeds are pending, never netted.
    assert state.settled_cash == C - buy_cash
    assert state.open_positions == ()
    (pending,) = state.pending_settlements
    assert pending.amount == net_proceeds
    assert pending.source_execution_id == sell.execution_id
    assert pending.settlement_session == SETTLE
    # Both executions are independently applied exactly once.
    applied_ids = {
        fingerprint.event_id
        for fingerprint in state.applied_events
        if fingerprint.event_kind is PortfolioEventKind.EXECUTION
    }
    assert applied_ids == {buy.execution_id, sell.execution_id}
    # Ledger: BUY_APPLIED then ROUND_TRIP SELL_APPLIED with frozen deltas.
    buy_row, sell_row = result.ledger_entries
    assert buy_row.event_type is PortfolioLedgerEventType.BUY_APPLIED
    assert buy_row.settled_cash_delta == -buy_cash
    assert buy_row.position_quantity_after == Q
    assert sell_row.event_type is PortfolioLedgerEventType.SELL_APPLIED
    assert sell_row.settled_cash_delta == 0
    assert sell_row.pending_cash_delta == net_proceeds
    assert sell_row.position_quantity_after == 0
    assert sell_row.settled_cash_after == buy_row.settled_cash_after
    # Exact sign reversal of the ledger row is the authoritative cost basis.
    assert subtract_exact_decimal(Decimal("0"), buy_row.settled_cash_delta) == (
        buy_cash
    )


def test_round_trip_result_is_independent_of_input_order() -> None:
    prior = PortfolioTransitionEngine.transition(
        initial_state(C), T0, (buy_event("ENTRY:Y", session=T0, asset_id=Y),)
    ).resulting_state
    buy, sell = round_trip_pair()
    prior_sell = sell_event(
        "EXIT:Y",
        session=T1,
        asset_id=Y,
        settlement_id="SETTLE:Y",
        settlement_session=SETTLE,
    )
    new_buy = buy_event("ENTRY:Z", session=T1, asset_id=Z)

    results = [
        PortfolioTransitionEngine.transition(prior, T1, order)
        for order in permutations((buy, sell, prior_sell, new_buy))
    ]

    first = results[0]
    assert all(item == first for item in results)
    assert [
        (row.event_type, row.asset_id) for row in first.ledger_entries
    ] == [
        (PortfolioLedgerEventType.SELL_APPLIED, Y),
        (PortfolioLedgerEventType.BUY_APPLIED, X),
        (PortfolioLedgerEventType.BUY_APPLIED, Z),
        (PortfolioLedgerEventType.SELL_APPLIED, X),
    ]
    assert tuple(
        position.asset_id for position in first.resulting_state.open_positions
    ) == (Z,)


def test_close_and_reenter_on_the_same_session_remains_valid() -> None:
    prior = PortfolioTransitionEngine.transition(
        initial_state(C), T0, (buy_event("ENTRY:X:T0", session=T0, asset_id=X),)
    ).resulting_state
    prior_sell = sell_event(
        "EXIT:X:T1",
        session=T1,
        asset_id=X,
        settlement_id="SETTLE:X:T1",
        settlement_session=SETTLE,
    )
    reentry = buy_event("ENTRY:X:T1", session=T1, asset_id=X)

    result = PortfolioTransitionEngine.transition(prior, T1, (reentry, prior_sell))

    assert [row.event_type for row in result.ledger_entries] == [
        PortfolioLedgerEventType.SELL_APPLIED,
        PortfolioLedgerEventType.BUY_APPLIED,
    ]
    (position,) = result.resulting_state.open_positions
    assert position.entry_execution_id == "ENTRY:X:T1"


def test_prior_sell_then_buy_then_protective_sell_of_one_asset_fails_closed() -> None:
    prior = PortfolioTransitionEngine.transition(
        initial_state(C), T0, (buy_event("ENTRY:X:T0", session=T0, asset_id=X),)
    ).resulting_state
    prior_sell = sell_event(
        "EXIT:X:T1",
        session=T1,
        asset_id=X,
        settlement_id="SETTLE:X:T1",
        settlement_session=SETTLE,
    )
    reentry, second_sell = round_trip_pair(session=T1)

    with pytest.raises(DuplicateAssetApplicationError):
        PortfolioTransitionEngine.transition(
            prior, T1, (prior_sell, reentry, second_sell)
        )
    # Nothing was mutated.
    assert PortfolioTransitionEngine.transition(prior, T1, ()).resulting_state == (
        prior.model_copy(update={"as_of_session": T1, "state_version": 2})
    )


def test_two_buys_of_one_asset_in_one_transition_fail_closed() -> None:
    first = buy_event("ENTRY:X:A", session=T1, asset_id=X)
    second = buy_event("ENTRY:X:B", session=T1, asset_id=X)
    with pytest.raises(DuplicateAssetApplicationError):
        PortfolioTransitionEngine.transition(initial_state(C), T1, (first, second))


def test_round_trip_sell_quantity_must_equal_the_buy_quantity() -> None:
    from stock_swing_d1.portfolio.portfolio_errors import InvalidExitQuantityError

    buy, sell = round_trip_pair()
    partial = sell.model_copy(update={"quantity": Q - 1})
    with pytest.raises(InvalidExitQuantityError):
        PortfolioTransitionEngine.transition(initial_state(C), T1, (buy, partial))


def test_exact_duplicate_presentations_keep_existing_exactly_once_semantics() -> None:
    buy, sell = round_trip_pair()

    result = PortfolioTransitionEngine.transition(
        initial_state(C), T1, (buy, sell, buy, sell)
    )

    # Identical duplicates collapse to one application each and are reported
    # as replayed; the economics are those of a single round trip.
    assert len(result.ledger_entries) == 2
    assert {reference.event_id for reference in result.replayed_events} == {
        buy.execution_id,
        sell.execution_id,
    }
    later = PortfolioTransitionEngine.transition(
        result.resulting_state, T2, (buy, sell)
    )
    assert later.ledger_entries == ()
    assert later.resulting_state.settled_cash == result.resulting_state.settled_cash


# ---------------------------------------------------------------------------
# Live / persisted replay symmetry
# ---------------------------------------------------------------------------


def _round_trip_run():
    prior_buy = buy_event("ENTRY:Y:T0", session=T0, asset_id=Y)
    buy, sell = round_trip_pair()
    prior_sell = sell_event(
        "EXIT:Y:T1",
        session=T1,
        asset_id=Y,
        settlement_id="SETTLE:Y:T1",
        settlement_session=SETTLE,
    )
    sessions = (
        PortfolioSessionInput(session=T0, execution_events=(prior_buy,)),
        PortfolioSessionInput(
            session=T1, execution_events=(sell, buy, prior_sell)
        ),
    )
    return PortfolioBacktestOrchestrator.run(initial_state(C), sessions), sessions


def test_persisted_replay_reconstructs_the_round_trip_from_session_start_p() -> None:
    result, _sessions = _round_trip_run()

    PortfolioInvariantChecker.validate_persisted_accounting_history(
        initial_state=result.initial_state,
        final_state=result.final_state,
        ledger_entries=result.ledger_entries,
        session_snapshots=result.snapshots,
    )
    t1_rows = [row for row in result.ledger_entries if row.session == T1]
    assert [(row.event_type, row.asset_id) for row in t1_rows] == [
        (PortfolioLedgerEventType.SELL_APPLIED, Y),
        (PortfolioLedgerEventType.BUY_APPLIED, X),
        (PortfolioLedgerEventType.SELL_APPLIED, X),
    ]
    assert result.final_state.open_positions == ()
    assert result.final_state_hash == hash_portfolio_state(result.final_state)


def test_reordered_round_trip_ledger_is_rejected_by_replay() -> None:
    result, _sessions = _round_trip_run()
    t0_rows = tuple(row for row in result.ledger_entries if row.session == T0)
    prior_sell_row, buy_row, round_trip_row = tuple(
        row for row in result.ledger_entries if row.session == T1
    )
    # A SELL of an asset absent from P placed before its BUY has the right
    # rows in the wrong application order: the classifier still labels it
    # ROUND_TRIP (X not in P, X in B) and the order check fails closed.
    swapped = t0_rows + (
        prior_sell_row.model_copy(update={"sequence_in_session": 0}),
        round_trip_row.model_copy(update={"sequence_in_session": 1}),
        buy_row.model_copy(update={"sequence_in_session": 2}),
    )
    with pytest.raises(PortfolioStateError):
        PortfolioInvariantChecker.validate_persisted_accounting_history(
            initial_state=result.initial_state,
            final_state=result.final_state,
            ledger_entries=swapped,
            session_snapshots=result.snapshots,
        )


def test_a_prior_position_sell_placed_after_its_asset_buy_is_rejected() -> None:
    """[BUY(X), SELL(X)] with X in P classifies the SELL as PRIOR (rank 1)
    regardless of row order, so a ledger that lists it after a BUY(X)
    cannot pass merely because it was already ordered that way."""

    prior = PortfolioTransitionEngine.transition(
        initial_state(C), T0, (buy_event("ENTRY:X:T0", session=T0, asset_id=X),)
    ).resulting_state
    prior_sell = sell_event(
        "EXIT:X:T1",
        session=T1,
        asset_id=X,
        settlement_id="SETTLE:X:T1",
        settlement_session=SETTLE,
    )
    reentry = buy_event("ENTRY:X:T1", session=T1, asset_id=X)
    live = PortfolioTransitionEngine.transition(prior, T1, (prior_sell, reentry))
    sell_row, buy_row = live.ledger_entries
    reordered = (
        buy_row.model_copy(update={"sequence_in_session": 0}),
        sell_row.model_copy(update={"sequence_in_session": 1}),
    )
    # Rejected twice over: sequential replay cannot re-add an asset that is
    # still open (the BUY row comes first), and the set-based classifier
    # still keys the SELL as PRIOR, so the order check would fail too.
    with pytest.raises(PortfolioStateError):
        PortfolioInvariantChecker.validate_transition(
            prior, live.resulting_state, reordered
        )
    classification = classify_session_applications(
        session_start_asset_ids=frozenset({X}),
        buy_asset_ids=(X,),
        sell_asset_ids=(X,),
    )
    assert classification.sell_rank_by_asset == {X: PRIOR_SELL_APPLICATION_RANK}


def test_live_transition_and_persisted_replay_agree_on_final_state_hash() -> None:
    result, _sessions = _round_trip_run()

    settled_cash = result.initial_state.settled_cash
    for row in result.ledger_entries:
        settled_cash += row.settled_cash_delta
    assert settled_cash == result.final_state.settled_cash
    snapshot = result.snapshots[-1]
    assert snapshot == PortfolioSessionSnapshot(
        session=T1,
        state_version=result.final_state.state_version,
        settled_cash=result.final_state.settled_cash,
        open_position_count=0,
        pending_settlement_count=2,
        state_hash=hash_portfolio_state(result.final_state),
    )


# ---------------------------------------------------------------------------
# Dividend entitlement is unchanged by a same-session round trip
# ---------------------------------------------------------------------------


def test_round_trip_on_the_ex_session_earns_no_entitlement() -> None:
    buy, sell = round_trip_pair(session=T2)
    evidence = make_dividend_evidence(
        event_id="D:X", entitlement_session=T1, ex_session=T2, security_id=X
    )

    result = PortfolioTransitionEngine.transition(
        initial_state(C), T2, (buy, sell), dividend_evidence=(evidence,)
    )

    (outcome,) = result.dividend_outcomes
    assert outcome.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED
    assert outcome.q_t == 0
    assert result.dividend_ledger_entries == ()
    assert result.resulting_state.settled_cash == C - (Q * PB + CB)


def test_prior_position_entitlement_is_preserved_alongside_a_round_trip() -> None:
    prior = PortfolioTransitionEngine.transition(
        initial_state(C),
        T1,
        (buy_event("ENTRY:Y:T1", session=T1, asset_id=Y, quantity=10),),
    ).resulting_state
    buy, sell = round_trip_pair(session=T2)
    evidence = make_dividend_evidence(
        event_id="D:Y", entitlement_session=T1, ex_session=T2, security_id=Y
    )

    result = PortfolioTransitionEngine.transition(
        prior, T2, (buy, sell), dividend_evidence=(evidence,)
    )

    (outcome,) = result.dividend_outcomes
    assert outcome.status is DividendApplicationStatus.APPLIED
    assert outcome.q_t == 10
    assert outcome.attribution_trade_id == "ENTRY:Y:T1"
    (dividend_row,) = result.dividend_ledger_entries
    assert dividend_row.asset_id == Y
    # Dividend cash arrives at step D, after the round-trip SELL (exact
    # helpers: the scale-38 dividend cash must not be rounded by the
    # ambient context in this assertion either).
    assert result.resulting_state.settled_cash == add_exact_decimal(
        subtract_exact_decimal(prior.settled_cash, Q * PB + CB),
        dividend_row.gross_cash_amount,
    )


# ---------------------------------------------------------------------------
# Persistence identity
# ---------------------------------------------------------------------------


def test_manifest_v0_3_binds_the_amended_merge_order_identity(tmp_path: Path) -> None:
    result, sessions = _round_trip_run()
    directory = tmp_path / "round_trip"
    PortfolioArtifactWriter.write(directory, result, sessions)

    artifact_set = PortfolioArtifactReader.read(directory)

    assert artifact_set.manifest.schema_version == "portfolio_manifest.v0.3"
    assert artifact_set.manifest.ledger_merge_order_identity == (
        "settlement_prior_sell_buy_round_trip_sell_dividend.v2"
    )
    assert PORTFOLIO_MANIFEST_SCHEMA_VERSION == "portfolio_manifest.v0.3"
    assert PORTFOLIO_LEDGER_MERGE_ORDER_IDENTITY == (
        "settlement_prior_sell_buy_round_trip_sell_dividend.v2"
    )
    assert artifact_set.final_state == result.final_state
    assert artifact_set.ledger_entries == result.ledger_entries
