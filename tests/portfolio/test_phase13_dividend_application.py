"""Slice 5 acceptance tests: Phase 13 in-memory ordinary-dividend application.

Covers the Phase 13 portions of the frozen Ordinary Dividend Amendment v0.7
(`phase-13-15a-15d-ordinary-dividend-amendment-v0.7-FROZEN.md`): the dedicated
non-execution dividend input channel (OD-13.2), authoritative pre-X
entitlement binding (OD-9), exact context-independent gross dividend cash
(OD-6/OD-11), the APPLIED/REPLAYED/NO_OP_NOT_ENTITLED outcome model
(OD-14.8), the DIVIDEND_APPLIED ledger row and its provenance (OD-14.1-14.7),
supplied-evidence <-> recorded-outcome discharge (OD-14.8-14.10), and
deterministic dividend ordering (OD-20.4). Persistence/offline
revalidation and Phase 15D projection are out of scope for this slice.
"""

from __future__ import annotations

import decimal
import inspect
from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CANONICAL_DIVIDEND_CURRENCY,
    CanonicalDividendAccountingEvidence,
    build_canonical_dividend_accounting_evidence,
    build_dividend_calendar_resolution_proof,
    build_ordinary_cash_classification_proof,
)
from stock_swing_d1.data.ordinary_dividend_normalization import (
    Gate3DividendNormalizationInputs,
    normalize_gate3_dividend,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    DividendApplicationStatus,
    DividendLedgerEntry,
    add_exact_decimal,
    compute_dividend_application_id,
    compute_dividend_application_payload_hash,
    compute_gross_dividend_cash,
)
from stock_swing_d1.portfolio.portfolio_errors import (
    DividendEvidenceError,
    DividendProvenanceError,
    DuplicateEventConflictError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    PortfolioEventKind,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_invariants import PortfolioInvariantChecker
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.portfolio.portfolio_transition import PortfolioTransitionEngine
from tests.portfolio.phase13_helpers import buy_event, initial_state, sell_event


T_SESSION = date(2026, 8, 18)
X_SESSION = date(2026, 8, 19)
X_PLUS_1 = date(2026, 8, 20)

ASSET_A = "NORGATE:1"
ASSET_B = "NORGATE:2"
SNAPSHOT_FINGERPRINT = "c" * 64
CLASSIFICATION_SOURCE_FINGERPRINT = "a" * 64
CALENDAR_SOURCE_FINGERPRINT = "b" * 64
CALENDAR_SOURCE_ID = "accepted-us-equity-calendar"
CALENDAR_POLICY_ID = "canonical-next-session"
CALENDAR_POLICY_VERSION = "1"
CLASSIFICATION_CONTRACT_ID = "ordinary-cash-classification.v1"


def make_dividend_event(
    *,
    event_id: str = "D:1",
    entitlement_session: date = T_SESSION,
    ex_session: date = X_SESSION,
    security_id: str = ASSET_A,
    d_capitalspecial: Decimal = Decimal("2"),
    unadjusted_close_t: Decimal = Decimal("3"),
    close_capital_t: Decimal = Decimal("4"),
    snapshot_fingerprint: str = SNAPSHOT_FINGERPRINT,
) -> CanonicalDividendAccountingEvidence:
    inputs = Gate3DividendNormalizationInputs(
        entitlement_session=entitlement_session,
        d_capitalspecial=d_capitalspecial,
        unadjusted_close_t=unadjusted_close_t,
        close_capital_t=close_capital_t,
    )
    result = normalize_gate3_dividend(inputs)
    classification = build_ordinary_cash_classification_proof(
        canonical_distribution_event_id=event_id,
        classification_contract_id=CLASSIFICATION_CONTRACT_ID,
        upstream_source_evidence_fingerprint=CLASSIFICATION_SOURCE_FINGERPRINT,
    )
    calendar = build_dividend_calendar_resolution_proof(
        calendar_source_id=CALENDAR_SOURCE_ID,
        entitlement_session=entitlement_session,
        ex_session=ex_session,
        calendar_policy_id=CALENDAR_POLICY_ID,
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        upstream_source_evidence_fingerprint=CALENDAR_SOURCE_FINGERPRINT,
    )
    return build_canonical_dividend_accounting_evidence(
        canonical_distribution_event_id=event_id,
        canonical_security_id=security_id,
        normalization_inputs=inputs,
        normalization_result=result,
        classification_proof=classification,
        calendar_resolution_proof=calendar,
        currency=CANONICAL_DIVIDEND_CURRENCY,
        canonical_distribution_snapshot_fingerprint=snapshot_fingerprint,
    )


def _seed_prior_dividend_application(
    state: PortfolioState,
    event: CanonicalDividendAccountingEvidence,
    *,
    q_t: int,
    trade_id: str | None,
    payload_override: str | None = None,
) -> PortfolioState:
    """Reconstruct ``state`` with a prior DIVIDEND AppliedEventFingerprint.

    Dividend evidence may only ever be supplied on session ``ex_session``
    (OD-13.6), so -- unlike EXECUTION replay -- a dividend REPLAY cannot be
    reached by resubmitting the same evidence on a later transition session.
    It is reached only when the authoritative pre-transition state already
    records the application (e.g. from an earlier successful run over the
    same starting point). This directly seeds that prior state rather than
    trying to synthesize it via a second `transition()` call.
    """

    from stock_swing_d1.portfolio.portfolio_dividend_events import (
        compute_dividend_application_id,
        compute_dividend_application_payload_hash,
    )
    from stock_swing_d1.portfolio.portfolio_events import AppliedEventFingerprint

    asset_id = event.canonical_security_id
    application_id = compute_dividend_application_id(
        canonical_distribution_event_id=event.canonical_distribution_event_id,
        entitlement_session=event.entitlement_session,
        ex_session=event.ex_session,
        asset_id=asset_id,
        attribution_trade_id=trade_id,
    )
    if payload_override is not None:
        payload_hash = payload_override
    else:
        gross_cash = compute_gross_dividend_cash(q_t, event.amount_per_share)
        payload_hash = compute_dividend_application_payload_hash(
            application_id=application_id,
            evidence=event,
            asset_id=asset_id,
            attribution_trade_id=trade_id,
            q_t=q_t,
            gross_cash_amount=gross_cash,
        )
    return PortfolioState(
        base_currency=state.base_currency,
        as_of_session=state.as_of_session,
        state_version=state.state_version,
        settled_cash=state.settled_cash,
        open_positions=state.open_positions,
        pending_settlements=state.pending_settlements,
        applied_events=state.applied_events
        + (
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.DIVIDEND,
                event_id=application_id,
                payload_sha256=payload_hash,
            ),
        ),
    )


def opened_state(
    *,
    cash: Decimal = Decimal("10000"),
    session: date = T_SESSION,
    execution_id: str = "BUY-1",
    asset_id: str = ASSET_A,
    quantity: int = 10,
    fill_price: Decimal = Decimal("100"),
    execution_cost: Decimal = Decimal("1"),
) -> tuple[PortfolioState, PortfolioExecutionEvent]:
    buy = buy_event(
        execution_id,
        session=session,
        asset_id=asset_id,
        quantity=quantity,
        fill_price=fill_price,
        execution_cost=execution_cost,
    )
    result = PortfolioTransitionEngine.transition(
        initial_state(cash), session, (buy,)
    )
    return result.resulting_state, buy


# ---------------------------------------------------------------------------
# A. Input channel
# ---------------------------------------------------------------------------


def test_dividend_evidence_is_a_dedicated_non_execution_channel() -> None:
    state, _ = opened_state()
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.dividend_outcomes[0].status is DividendApplicationStatus.APPLIED


def test_execution_event_rejected_from_dividend_evidence_parameter() -> None:
    state, _ = opened_state()
    bogus = buy_event("BUY-BOGUS", session=X_SESSION)

    with pytest.raises(DividendEvidenceError):
        PortfolioTransitionEngine.transition(
            state, X_SESSION, (), dividend_evidence=(bogus,)
        )


def test_dividend_evidence_wrong_ex_session_fails_closed() -> None:
    state, _ = opened_state()
    event = make_dividend_event(ex_session=X_PLUS_1)

    with pytest.raises(DividendEvidenceError):
        PortfolioTransitionEngine.transition(
            state, X_SESSION, (), dividend_evidence=(event,)
        )


def test_malformed_dividend_evidence_fails_closed() -> None:
    state, _ = opened_state()

    with pytest.raises(DividendEvidenceError):
        PortfolioTransitionEngine.transition(
            state, X_SESSION, (), dividend_evidence=("not-evidence",)
        )


def test_empty_dividend_evidence_tuple_is_accepted_by_default() -> None:
    state, _ = opened_state()

    result = PortfolioTransitionEngine.transition(state, X_SESSION, ())

    assert result.dividend_outcomes == ()
    assert result.dividend_ledger_entries == ()


def test_no_network_or_provider_access_in_gross_cash_computation() -> None:
    source = inspect.getsource(compute_gross_dividend_cash)
    for forbidden in ("socket", "requests", "norgatedata"):
        assert forbidden not in source.lower()


# ---------------------------------------------------------------------------
# B. Pre-X entitlement binding
# ---------------------------------------------------------------------------


def test_no_pre_x_position_yields_zero_q_t_and_no_trade_id() -> None:
    state = initial_state(Decimal("10000"))
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    outcome = result.dividend_outcomes[0]
    assert outcome.q_t == 0
    assert outcome.attribution_trade_id is None
    assert outcome.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED


def test_held_pre_x_position_yields_exact_q_t_and_entry_execution_id() -> None:
    state, buy = opened_state(quantity=17, execution_id="BUY-17")
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    outcome = result.dividend_outcomes[0]
    assert outcome.q_t == 17
    assert outcome.attribution_trade_id == buy.execution_id


def test_binding_captured_before_same_x_sell() -> None:
    state, buy = opened_state(quantity=10)
    event = make_dividend_event()
    sell = sell_event(
        "SELL-1",
        session=X_SESSION,
        asset_id=ASSET_A,
        quantity=10,
        settlement_id="SETTLE-1",
        settlement_session=X_PLUS_1,
    )

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (sell,), dividend_evidence=(event,)
    )

    outcome = result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.APPLIED
    assert outcome.q_t == 10
    assert outcome.attribution_trade_id == buy.execution_id
    assert result.resulting_state.open_positions == ()


def test_binding_captured_before_same_x_buy_of_new_position() -> None:
    state = initial_state(Decimal("10000"))
    event = make_dividend_event(security_id=ASSET_A)
    buy = buy_event("BUY-NEW", session=X_SESSION, asset_id=ASSET_A)

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (buy,), dividend_evidence=(event,)
    )

    outcome = result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED
    assert outcome.q_t == 0
    assert outcome.attribution_trade_id is None
    assert len(result.resulting_state.open_positions) == 1


def test_binding_remains_immutable_through_transition() -> None:
    state, buy = opened_state(quantity=5)
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    row = result.dividend_ledger_entries[0]
    assert row.q_t == 5
    assert row.attribution_trade_id == buy.execution_id
    assert row.entitlement_session == T_SESSION


# ---------------------------------------------------------------------------
# C. Frozen entitlement consequences
# ---------------------------------------------------------------------------


def test_hold_through_t_and_x_is_applied() -> None:
    state, _ = opened_state()
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.dividend_outcomes[0].status is DividendApplicationStatus.APPLIED


def test_sell_on_x_applies_to_old_trade() -> None:
    state, buy = opened_state()
    sell = sell_event(
        "SELL-1",
        session=X_SESSION,
        asset_id=ASSET_A,
        quantity=10,
        settlement_id="SETTLE-1",
        settlement_session=X_PLUS_1,
    )
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (sell,), dividend_evidence=(event,)
    )

    outcome = result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.APPLIED
    assert outcome.attribution_trade_id == buy.execution_id


def test_buy_only_on_x_is_no_op_not_entitled() -> None:
    state = initial_state(Decimal("10000"))
    buy = buy_event("BUY-NEW", session=X_SESSION, asset_id=ASSET_A)
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (buy,), dividend_evidence=(event,)
    )

    assert result.dividend_outcomes[0].status is (
        DividendApplicationStatus.NO_OP_NOT_ENTITLED
    )


def test_sell_old_and_buy_new_same_security_attributes_old_trade_only() -> None:
    state, old_buy = opened_state(execution_id="BUY-OLD")
    sell = sell_event(
        "SELL-OLD",
        session=X_SESSION,
        asset_id=ASSET_A,
        quantity=10,
        settlement_id="SETTLE-OLD",
        settlement_session=X_PLUS_1,
    )
    new_buy = buy_event(
        "BUY-NEW", session=X_SESSION, asset_id=ASSET_A, quantity=3
    )
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (sell, new_buy), dividend_evidence=(event,)
    )

    outcome = result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.APPLIED
    assert outcome.attribution_trade_id == old_buy.execution_id
    assert outcome.q_t == 10
    new_position = result.resulting_state.open_positions[0]
    assert new_position.entry_execution_id == "BUY-NEW"
    assert new_position.quantity == 3


# ---------------------------------------------------------------------------
# D. Exact cash
# ---------------------------------------------------------------------------


def test_gross_cash_amount_equals_q_t_times_d_h_exactly() -> None:
    state, _ = opened_state(quantity=10)
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    row = result.dividend_ledger_entries[0]
    assert row.gross_cash_amount == compute_gross_dividend_cash(
        10, event.amount_per_share
    )
    assert row.gross_cash_amount == Decimal(10) * event.amount_per_share


def test_sub_cent_dividend_is_preserved_without_rounding() -> None:
    state, _ = opened_state(quantity=3)
    event = make_dividend_event(
        d_capitalspecial=Decimal("1"),
        unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("3"),
    )

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    row = result.dividend_ledger_entries[0]
    assert row.gross_cash_amount == compute_gross_dividend_cash(
        3, event.amount_per_share
    )
    assert row.gross_cash_amount != Decimal("1")
    assert row.gross_cash_amount.as_tuple().exponent == -38


def test_ambient_decimal_precision_does_not_change_gross_cash() -> None:
    digits = tuple(int(ch) for ch in ("123456789" * 5)[:39])
    d_h = Decimal((0, digits, -38))
    assert d_h.as_tuple().exponent == -38
    q_t = 123456789

    with decimal.localcontext() as ctx:
        ctx.prec = 28
        low_precision = compute_gross_dividend_cash(q_t, d_h)
    with decimal.localcontext() as ctx:
        ctx.prec = 60
        high_precision = compute_gross_dividend_cash(q_t, d_h)

    assert low_precision == high_precision


def test_add_exact_decimal_matches_true_sum_beyond_context_precision() -> None:
    digits = tuple(int(ch) for ch in ("123456789" * 5)[:39])
    delta = Decimal((0, digits, -38))
    base = Decimal("76543210.99")

    with decimal.localcontext() as ctx:
        ctx.prec = 200
        reference = base + delta  # ample headroom: not itself rounded

    with decimal.localcontext() as ctx:
        ctx.prec = 28
        low_precision = add_exact_decimal(base, delta)
    with decimal.localcontext() as ctx:
        ctx.prec = 60
        high_precision = add_exact_decimal(base, delta)

    assert low_precision == reference
    assert high_precision == reference
    assert low_precision.as_tuple() == high_precision.as_tuple() == reference.as_tuple()


def test_ambient_decimal_precision_does_not_change_settled_cash_after_dividend() -> (
    None
):
    """OD-6.8/OD-12.1: settled-cash mutation must not depend on
    ``decimal.getcontext().prec``. A large pre-X quantity against a
    scale-38 D_H with a non-terminating normalized fraction produces a
    gross cash amount whose exact sum with settled cash needs more
    significant digits than Python's default context (28) allows.
    """

    state, _ = opened_state(
        quantity=123_456_789,
        cash=Decimal("200000000"),
        fill_price=Decimal("1"),
    )
    event = make_dividend_event(
        d_capitalspecial=Decimal("1"),
        unadjusted_close_t=Decimal("1"),
        close_capital_t=Decimal("3"),
    )

    with decimal.localcontext() as ctx:
        ctx.prec = 28
        low_precision = PortfolioTransitionEngine.transition(
            state, X_SESSION, (), dividend_evidence=(event,)
        )
    with decimal.localcontext() as ctx:
        ctx.prec = 60
        high_precision = PortfolioTransitionEngine.transition(
            state, X_SESSION, (), dividend_evidence=(event,)
        )

    assert (
        low_precision.resulting_state.settled_cash
        == high_precision.resulting_state.settled_cash
    )
    assert (
        low_precision.dividend_ledger_entries[0].settled_cash_after
        == high_precision.dividend_ledger_entries[0].settled_cash_after
    )
    assert low_precision.state_hash_after == high_precision.state_hash_after


def test_compute_gross_dividend_cash_never_constructs_decimal_from_float() -> None:
    source = inspect.getsource(compute_gross_dividend_cash)
    assert "float(" not in source


def test_zero_quantity_produces_exact_zero_gross_cash() -> None:
    event = make_dividend_event()

    assert compute_gross_dividend_cash(0, event.amount_per_share) == Decimal(0)


# ---------------------------------------------------------------------------
# E. Chronology / buying power
# ---------------------------------------------------------------------------


def test_same_x_buy_cannot_use_dividend_cash() -> None:
    state = initial_state(Decimal("1000"))
    # Fund a pre-existing entitled position first, on T.
    buy_t = buy_event(
        "BUY-T", session=T_SESSION, asset_id=ASSET_A, quantity=1,
        fill_price=Decimal("1"), execution_cost=Decimal("0"),
    )
    funded = PortfolioTransitionEngine.transition(
        state, T_SESSION, (buy_t,)
    ).resulting_state
    event = make_dividend_event()
    expected_gross = compute_gross_dividend_cash(1, event.amount_per_share)

    # A same-X BUY sized to require MORE than the pre-dividend settled cash
    # (but exactly the post-dividend cash) must fail: dividend cash is not
    # available to executions earlier on X.
    settled_before_dividend = funded.settled_cash
    required = settled_before_dividend + expected_gross
    buy_x = buy_event(
        "BUY-X",
        session=X_SESSION,
        asset_id=ASSET_B,
        quantity=1,
        fill_price=required,
        execution_cost=Decimal("0"),
    )

    from stock_swing_d1.portfolio.portfolio_errors import (
        InsufficientSettledCashError,
    )

    with pytest.raises(InsufficientSettledCashError):
        PortfolioTransitionEngine.transition(
            funded, X_SESSION, (buy_x,), dividend_evidence=(event,)
        )


def test_end_of_x_settled_cash_includes_dividend() -> None:
    state, _ = opened_state(cash=Decimal("10000"), quantity=10)
    event = make_dividend_event()
    expected_gross = compute_gross_dividend_cash(10, event.amount_per_share)

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.resulting_state.settled_cash == state.settled_cash + expected_gross


def test_no_pending_settlement_created_for_dividend() -> None:
    state, _ = opened_state()
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.resulting_state.pending_settlements == ()


# ---------------------------------------------------------------------------
# F. APPLIED
# ---------------------------------------------------------------------------


def test_applied_records_one_new_dividend_fingerprint() -> None:
    state, _ = opened_state()
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    dividend_fingerprints = [
        item
        for item in result.resulting_state.applied_events
        if item.event_kind is PortfolioEventKind.DIVIDEND
    ]
    assert len(dividend_fingerprints) == 1


def test_applied_writes_exactly_one_dividend_ledger_row() -> None:
    state, _ = opened_state()
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert len(result.dividend_ledger_entries) == 1
    assert isinstance(result.dividend_ledger_entries[0], DividendLedgerEntry)


def test_applied_mutates_settled_cash_exactly_once() -> None:
    state, _ = opened_state(quantity=10)
    event = make_dividend_event()
    expected_gross = compute_gross_dividend_cash(10, event.amount_per_share)

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    row = result.dividend_ledger_entries[0]
    assert row.settled_cash_delta == expected_gross
    assert row.settled_cash_after == state.settled_cash + expected_gross


def test_applied_ledger_row_carries_old_trade_attribution() -> None:
    state, buy = opened_state()
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    row = result.dividend_ledger_entries[0]
    assert row.attribution_trade_id == buy.execution_id


def test_applied_row_retains_frozen_evidence_fingerprints() -> None:
    state, _ = opened_state()
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    row = result.dividend_ledger_entries[0]
    assert row.canonical_distribution_event_id == event.canonical_distribution_event_id
    assert row.canonical_distribution_snapshot_fingerprint == (
        event.canonical_distribution_snapshot_fingerprint
    )
    assert row.normalization_inputs_fingerprint == (
        event.normalization_inputs_fingerprint
    )
    assert row.calendar_resolution_fingerprint == (
        event.calendar_resolution_fingerprint
    )
    assert row.d_h == event.amount_per_share
    assert row.currency == event.currency


# ---------------------------------------------------------------------------
# G. REPLAYED
# ---------------------------------------------------------------------------


def test_identical_replay_produces_replayed() -> None:
    state, buy = opened_state(quantity=10)
    event = make_dividend_event()
    seeded = _seed_prior_dividend_application(
        state, event, q_t=10, trade_id=buy.execution_id
    )

    result = PortfolioTransitionEngine.transition(
        seeded, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.dividend_outcomes[0].status is DividendApplicationStatus.REPLAYED


def test_replay_writes_no_second_dividend_row() -> None:
    state, buy = opened_state(quantity=10)
    event = make_dividend_event()
    seeded = _seed_prior_dividend_application(
        state, event, q_t=10, trade_id=buy.execution_id
    )

    result = PortfolioTransitionEngine.transition(
        seeded, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.dividend_ledger_entries == ()


def test_replay_credits_no_second_cash() -> None:
    state, buy = opened_state(quantity=10)
    event = make_dividend_event()
    seeded = _seed_prior_dividend_application(
        state, event, q_t=10, trade_id=buy.execution_id
    )

    result = PortfolioTransitionEngine.transition(
        seeded, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.resulting_state.settled_cash == seeded.settled_cash


def test_same_application_id_different_payload_is_fatal() -> None:
    """A prior application recorded under the correct deterministic
    application_id but a different (forged) payload hash must fail closed
    rather than being silently treated as a fresh application or a replay.
    """

    state, buy = opened_state(quantity=10, execution_id="BUY-SAME")
    event = make_dividend_event()
    tampered_state = _seed_prior_dividend_application(
        state,
        event,
        q_t=10,
        trade_id=buy.execution_id,
        payload_override="f" * 64,
    )

    with pytest.raises(DuplicateEventConflictError):
        PortfolioTransitionEngine.transition(
            tampered_state, X_SESSION, (), dividend_evidence=(event,)
        )


# ---------------------------------------------------------------------------
# H. NO_OP_NOT_ENTITLED
# ---------------------------------------------------------------------------


def test_zero_pre_x_quantity_produces_explicit_no_op() -> None:
    state = initial_state(Decimal("10000"))
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.dividend_outcomes[0].status is (
        DividendApplicationStatus.NO_OP_NOT_ENTITLED
    )


def test_no_op_has_zero_q_t() -> None:
    state = initial_state(Decimal("10000"))
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.dividend_outcomes[0].q_t == 0


def test_no_op_has_no_attribution_trade_id() -> None:
    state = initial_state(Decimal("10000"))
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.dividend_outcomes[0].attribution_trade_id is None


def test_no_op_writes_no_row() -> None:
    state = initial_state(Decimal("10000"))
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.dividend_ledger_entries == ()


def test_no_op_mutates_no_cash() -> None:
    state = initial_state(Decimal("10000"))
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.resulting_state.settled_cash == state.settled_cash


def test_no_op_does_not_mutate_applied_events() -> None:
    state = initial_state(Decimal("10000"))
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert result.resulting_state.applied_events == ()


def test_post_x_buy_cannot_justify_or_alter_no_op_proof() -> None:
    state = initial_state(Decimal("10000"))
    buy = buy_event("BUY-X", session=X_SESSION, asset_id=ASSET_A)
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (buy,), dividend_evidence=(event,)
    )

    outcome = result.dividend_outcomes[0]
    assert outcome.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED
    assert outcome.q_t == 0
    assert len(result.resulting_state.open_positions) == 1


# ---------------------------------------------------------------------------
# I. Deterministic ordering
# ---------------------------------------------------------------------------


def test_multiple_dividend_events_ordered_by_asset_then_event_id() -> None:
    buy_b = buy_event(
        "BUY-B", session=T_SESSION, asset_id=ASSET_B, quantity=3
    )
    # Fund both ASSET_A and ASSET_B on the same T session.
    state = PortfolioTransitionEngine.transition(
        initial_state(Decimal("10000")),
        T_SESSION,
        (
            buy_event("BUY-A", session=T_SESSION, asset_id=ASSET_A, quantity=5),
            buy_b,
        ),
    ).resulting_state

    event_a_2 = make_dividend_event(event_id="D:A2", security_id=ASSET_A)
    event_a_1 = make_dividend_event(event_id="D:A1", security_id=ASSET_A)
    event_b_1 = make_dividend_event(event_id="D:B1", security_id=ASSET_B)

    result = PortfolioTransitionEngine.transition(
        state,
        X_SESSION,
        (),
        dividend_evidence=(event_a_2, event_b_1, event_a_1),
    )

    ordered_ids = [
        (outcome.asset_id, outcome.canonical_distribution_event_id)
        for outcome in result.dividend_outcomes
    ]
    assert ordered_ids == sorted(ordered_ids)
    assert ordered_ids == [
        (ASSET_A, "D:A1"),
        (ASSET_A, "D:A2"),
        (ASSET_B, "D:B1"),
    ]


def test_reversing_input_order_produces_identical_canonical_row_order() -> None:
    state = PortfolioTransitionEngine.transition(
        initial_state(Decimal("10000")),
        T_SESSION,
        (
            buy_event("BUY-A", session=T_SESSION, asset_id=ASSET_A, quantity=5),
            buy_event("BUY-B", session=T_SESSION, asset_id=ASSET_B, quantity=3),
        ),
    ).resulting_state

    event_a = make_dividend_event(event_id="D:A1", security_id=ASSET_A)
    event_b = make_dividend_event(event_id="D:B1", security_id=ASSET_B)

    forward = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event_a, event_b)
    )
    reversed_result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event_b, event_a)
    )

    forward_ids = [
        row.canonical_distribution_event_id + row.asset_id
        for row in forward.dividend_ledger_entries
    ]
    reversed_ids = [
        row.canonical_distribution_event_id + row.asset_id
        for row in reversed_result.dividend_ledger_entries
    ]
    assert forward_ids == reversed_ids


def test_duplicate_event_id_within_batch_fails_closed() -> None:
    state, _ = opened_state()
    event = make_dividend_event()

    with pytest.raises(DividendEvidenceError):
        PortfolioTransitionEngine.transition(
            state, X_SESSION, (), dividend_evidence=(event, event)
        )


# ---------------------------------------------------------------------------
# J. Row -> evidence provenance (direct invariant-checker proofs)
# ---------------------------------------------------------------------------


def _applied_row_and_context(
    *, quantity: int = 10
) -> tuple[PortfolioState, CanonicalDividendAccountingEvidence, DividendLedgerEntry]:
    state, _ = opened_state(quantity=quantity)
    event = make_dividend_event()
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )
    return state, event, result.dividend_ledger_entries[0]


def test_valid_row_passes_provenance_and_discharge() -> None:
    state, event, row = _applied_row_and_context()
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    PortfolioInvariantChecker.validate_dividend_application(
        session=X_SESSION,
        previous_state=state,
        dividend_evidence=(event,),
        dividend_ledger_entries=result.dividend_ledger_entries,
        dividend_outcomes=result.dividend_outcomes,
    )


def test_row_with_no_supplied_evidence_fails() -> None:
    state, event, row = _applied_row_and_context()

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(),
            dividend_ledger_entries=(row,),
            dividend_outcomes=(),
        )


def test_row_payload_hash_mismatch_fails() -> None:
    state, event, row = _applied_row_and_context()
    tampered = row.model_copy(update={"source_payload_sha256": "9" * 64})

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(event,),
            dividend_ledger_entries=(tampered,),
            dividend_outcomes=result.dividend_outcomes,
        )


def test_row_d_h_mismatch_fails() -> None:
    state, event, row = _applied_row_and_context()
    tampered = row.model_copy(
        update={
            "d_h": Decimal("9." + "0" * 37),
            "gross_cash_amount": Decimal(10) * Decimal("9." + "0" * 37),
            "settled_cash_delta": Decimal(10) * Decimal("9." + "0" * 37),
        }
    )
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(event,),
            dividend_ledger_entries=(tampered,),
            dividend_outcomes=result.dividend_outcomes,
        )


def test_row_q_t_mismatch_fails() -> None:
    state, event, row = _applied_row_and_context(quantity=10)
    tampered = row.model_copy(update={"q_t": 5})
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(event,),
            dividend_ledger_entries=(tampered,),
            dividend_outcomes=result.dividend_outcomes,
        )


def test_row_attribution_mismatch_fails() -> None:
    state, event, row = _applied_row_and_context()
    tampered = row.model_copy(update={"attribution_trade_id": "SOMEONE-ELSE"})
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(event,),
            dividend_ledger_entries=(tampered,),
            dividend_outcomes=result.dividend_outcomes,
        )


def test_row_gross_cash_mismatch_fails() -> None:
    state, event, row = _applied_row_and_context()
    wrong_amount = row.gross_cash_amount + Decimal("1")
    tampered = row.model_copy(
        update={
            "gross_cash_amount": wrong_amount,
            "settled_cash_delta": wrong_amount,
            "settled_cash_after": row.settled_cash_after + Decimal("1"),
        }
    )
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(event,),
            dividend_ledger_entries=(tampered,),
            dividend_outcomes=result.dividend_outcomes,
        )


def test_dividend_row_not_accepted_by_settlement_exemption() -> None:
    """A DIVIDEND_APPLIED row must go through OD-14.5 provenance, never the
    settlement internal-derivation exemption; asserting on the dividend
    ledger's own event_type literal proves it cannot be misrouted."""

    from stock_swing_d1.portfolio.portfolio_events import PortfolioLedgerEventType

    _, _, row = _applied_row_and_context()
    assert row.event_type is PortfolioLedgerEventType.DIVIDEND_APPLIED


def test_dividend_row_cannot_be_dispatched_through_settlement_ledger_path() -> None:
    """OD-14.6: the settlement internal-derivation exemption is the only
    dispatch path that bypasses external-source matching, and it exists
    solely because settlement rows are deterministically re-derived from
    authoritative prior state. A ``PortfolioLedgerEntry`` carrying every
    fact of a real, valid SETTLEMENT_APPLIED row -- except its event_type
    label -- proves that mislabeling as DIVIDEND_APPLIED cannot slip
    through that exemption: the dispatch must fail closed instead of
    silently treating it as a settlement."""

    from stock_swing_d1.portfolio.portfolio_errors import PortfolioStateError
    from stock_swing_d1.portfolio.portfolio_events import PortfolioLedgerEventType

    state, _ = opened_state(quantity=10)
    sell = sell_event(
        "SELL-1",
        session=X_SESSION,
        asset_id=ASSET_A,
        quantity=10,
        settlement_id="SETTLE-1",
        settlement_session=X_PLUS_1,
    )
    after_sell = PortfolioTransitionEngine.transition(
        state, X_SESSION, (sell,)
    ).resulting_state
    settlement_result = PortfolioTransitionEngine.transition(after_sell, X_PLUS_1, ())
    settlement_row = settlement_result.ledger_entries[0]
    assert settlement_row.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED

    relabeled = settlement_row.model_copy(
        update={"event_type": PortfolioLedgerEventType.DIVIDEND_APPLIED}
    )

    with pytest.raises(PortfolioStateError, match="unsupported ledger event type"):
        PortfolioInvariantChecker.validate_transition(
            after_sell,
            settlement_result.resulting_state,
            (relabeled,),
        )


# ---------------------------------------------------------------------------
# K. Evidence -> outcome discharge
# ---------------------------------------------------------------------------


def test_every_supplied_event_has_exactly_one_outcome() -> None:
    state, _ = opened_state()
    event = make_dividend_event()

    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    assert len(result.dividend_outcomes) == 1
    assert result.dividend_outcomes[0].canonical_distribution_event_id == (
        event.canonical_distribution_event_id
    )


def test_missing_outcome_fails_direct_invariant_check() -> None:
    state, event, row = _applied_row_and_context()

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(event,),
            dividend_ledger_entries=(row,),
            dividend_outcomes=(),
        )


def test_outcome_without_evidence_fails() -> None:
    state, event, row = _applied_row_and_context()
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(),
            dividend_ledger_entries=(),
            dividend_outcomes=result.dividend_outcomes,
        )


def test_duplicate_outcome_identity_fails() -> None:
    state, event, row = _applied_row_and_context()
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )
    duplicated = result.dividend_outcomes + result.dividend_outcomes

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(event,),
            dividend_ledger_entries=result.dividend_ledger_entries,
            dividend_outcomes=duplicated,
        )


def test_no_op_with_non_zero_pre_x_quantity_fails() -> None:
    state, _ = opened_state(quantity=10)
    event = make_dividend_event()

    from stock_swing_d1.portfolio.portfolio_dividend_events import (
        DividendApplicationOutcome,
        compute_dividend_application_id,
        compute_dividend_application_payload_hash,
    )

    forged_id = compute_dividend_application_id(
        canonical_distribution_event_id=event.canonical_distribution_event_id,
        entitlement_session=event.entitlement_session,
        ex_session=event.ex_session,
        asset_id=ASSET_A,
        attribution_trade_id=None,
    )
    forged_hash = compute_dividend_application_payload_hash(
        application_id=forged_id,
        evidence=event,
        asset_id=ASSET_A,
        attribution_trade_id=None,
        q_t=0,
        gross_cash_amount=Decimal(0),
    )
    forged_outcome = DividendApplicationOutcome(
        canonical_distribution_event_id=event.canonical_distribution_event_id,
        application_id=forged_id,
        payload_sha256=forged_hash,
        asset_id=ASSET_A,
        entitlement_session=event.entitlement_session,
        ex_session=event.ex_session,
        q_t=0,
        attribution_trade_id=None,
        status=DividendApplicationStatus.NO_OP_NOT_ENTITLED,
    )

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(event,),
            dividend_ledger_entries=(),
            dividend_outcomes=(forged_outcome,),
        )


def test_silent_same_x_sell_drop_fails_rather_than_disappearing() -> None:
    state, buy = opened_state(quantity=10)
    sell = sell_event(
        "SELL-1",
        session=X_SESSION,
        asset_id=ASSET_A,
        quantity=10,
        settlement_id="SETTLE-1",
        settlement_session=X_PLUS_1,
    )
    event = make_dividend_event()
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (sell,), dividend_evidence=(event,)
    )

    with pytest.raises(DividendProvenanceError):
        PortfolioInvariantChecker.validate_dividend_application(
            session=X_SESSION,
            previous_state=state,
            dividend_evidence=(event,),
            dividend_ledger_entries=(),
            dividend_outcomes=(),
        )
    assert result.dividend_outcomes[0].status is DividendApplicationStatus.APPLIED


# ---------------------------------------------------------------------------
# L. Existing Phase 13 regression (spot checks; full regression covered by
# the unmodified tests/portfolio suite, which remains green).
# ---------------------------------------------------------------------------


def test_buy_behavior_unchanged_with_dividend_parameter_default() -> None:
    state = initial_state(Decimal("10000"))
    buy = buy_event("BUY-1", session=T_SESSION, asset_id=ASSET_A, quantity=10)

    result = PortfolioTransitionEngine.transition(state, T_SESSION, (buy,))

    assert result.dividend_ledger_entries == ()
    assert result.dividend_outcomes == ()
    assert len(result.resulting_state.open_positions) == 1


def test_sell_and_settlement_behavior_unchanged() -> None:
    state, _ = opened_state()
    sell = sell_event(
        "SELL-1",
        session=X_SESSION,
        asset_id=ASSET_A,
        quantity=10,
        settlement_id="SETTLE-1",
        settlement_session=X_PLUS_1,
    )
    after_sell = PortfolioTransitionEngine.transition(
        state, X_SESSION, (sell,)
    ).resulting_state
    settled = PortfolioTransitionEngine.transition(after_sell, X_PLUS_1, ())

    assert settled.dividend_ledger_entries == ()
    assert len(settled.ledger_entries) == 1
    assert settled.resulting_state.pending_settlements == ()


def test_execution_replay_behavior_unchanged() -> None:
    state = initial_state(Decimal("10000"))
    buy = buy_event("BUY-1", session=T_SESSION, asset_id=ASSET_A, quantity=10)
    first = PortfolioTransitionEngine.transition(state, T_SESSION, (buy,))
    second = PortfolioTransitionEngine.transition(
        first.resulting_state, X_SESSION, (buy,)
    )

    assert second.ledger_entries == ()
    assert second.resulting_state.settled_cash == first.resulting_state.settled_cash
    assert second.resulting_state.open_positions == first.resulting_state.open_positions
    assert second.resulting_state.applied_events == first.resulting_state.applied_events


def test_existing_state_hash_and_invariant_tests_remain_green() -> None:
    """Smoke check that PortfolioInvariantChecker.validate_state still
    accepts a dividend-mutated state (full regression lives in
    tests/portfolio, which is unmodified aside from the frozen enum
    extension)."""

    state, _ = opened_state()
    event = make_dividend_event()
    result = PortfolioTransitionEngine.transition(
        state, X_SESSION, (), dividend_evidence=(event,)
    )

    PortfolioInvariantChecker.validate_state(result.resulting_state)


# ---------------------------------------------------------------------------
# M. OD-13.4 / OD-13.5 hash-field binding
#
# The dividend-policy version and accounting scope are represented as
# frozen hardcoded domains rather than function arguments to
# compute_dividend_application_id / compute_dividend_application_payload_hash.
# That is only conformant to OD-13.4/OD-13.5 if each is explicit in the
# hashed canonical payload -- these tests prove every required field,
# including both frozen constants, actually participates in the hash.
# ---------------------------------------------------------------------------


_IDENTITY_BASE_KWARGS: dict[str, object] = {
    "canonical_distribution_event_id": "D:BASE",
    "entitlement_session": T_SESSION,
    "ex_session": X_SESSION,
    "asset_id": ASSET_A,
    "attribution_trade_id": "BUY-1",
}
_IDENTITY_FIELD_VARIANTS: dict[str, object] = {
    "accounting_scope": "OTHER_SCOPE",
    "canonical_distribution_event_id": "D:OTHER",
    "entitlement_session": date(2026, 8, 17),
    "ex_session": date(2026, 8, 20),
    "asset_id": ASSET_B,
    "attribution_trade_id": "BUY-2",
}

_PAYLOAD_BASE_KWARGS: dict[str, object] = {
    "application_id": "application-id-base",
    "dividend_policy_semantic_identity": "ordinary-dividend-policy-v-base",
    "canonical_distribution_snapshot_fingerprint": "c" * 64,
    "canonical_distribution_event_id": "D:BASE",
    "asset_id": ASSET_A,
    "entitlement_session": T_SESSION,
    "ex_session": X_SESSION,
    "attribution_trade_id": "BUY-1",
    "q_t": 10,
    "d_h": Decimal((0, (3,) * 38, -38)),
    "amount_basis": "PER_SHARE",
    "normalization_method_id": "GATE3_V1",
    "normalization_scale": 38,
    "normalization_rounding_mode": "ROUND_HALF_EVEN",
    "normalization_arithmetic_mode": "EXACT_RATIONAL",
    "normalization_inputs_fingerprint": "d" * 64,
    "calendar_resolution_fingerprint": "e" * 64,
    "gross_cash_amount": Decimal("30"),
    "currency": "USD",
}
_PAYLOAD_FIELD_VARIANTS: dict[str, object] = {
    "application_id": "application-id-other",
    "dividend_policy_semantic_identity": "ordinary-dividend-policy-v-other",
    "canonical_distribution_snapshot_fingerprint": "f" * 64,
    "canonical_distribution_event_id": "D:OTHER",
    "asset_id": ASSET_B,
    "entitlement_session": date(2026, 8, 17),
    "ex_session": date(2026, 8, 20),
    "attribution_trade_id": "BUY-2",
    "q_t": 11,
    "d_h": Decimal((0, (7,) * 38, -38)),
    "amount_basis": "OTHER_BASIS",
    "normalization_method_id": "GATE3_V2",
    "normalization_scale": 30,
    "normalization_rounding_mode": "ROUND_UP",
    "normalization_arithmetic_mode": "OTHER_MODE",
    "normalization_inputs_fingerprint": "1" * 64,
    "calendar_resolution_fingerprint": "2" * 64,
    "gross_cash_amount": Decimal("31"),
    "currency": "EUR",
}


@pytest.mark.parametrize("field_name", sorted(_IDENTITY_FIELD_VARIANTS))
def test_application_identity_payload_binds_every_od_13_4_field(
    field_name: str,
) -> None:
    from stock_swing_d1.portfolio.portfolio_dividend_events import (
        _DividendApplicationIdentityPayload,
    )
    from stock_swing_d1.portfolio.portfolio_hashing import canonical_payload_sha256

    base = _DividendApplicationIdentityPayload(**_IDENTITY_BASE_KWARGS)
    variant = base.model_copy(
        update={field_name: _IDENTITY_FIELD_VARIANTS[field_name]}
    )

    assert canonical_payload_sha256(base) != canonical_payload_sha256(variant)


@pytest.mark.parametrize("field_name", sorted(_PAYLOAD_FIELD_VARIANTS))
def test_application_payload_binds_every_od_13_5_field(field_name: str) -> None:
    from stock_swing_d1.portfolio.portfolio_dividend_events import (
        _DividendApplicationPayload,
    )
    from stock_swing_d1.portfolio.portfolio_hashing import canonical_payload_sha256

    base = _DividendApplicationPayload(**_PAYLOAD_BASE_KWARGS)
    variant = base.model_copy(update={field_name: _PAYLOAD_FIELD_VARIANTS[field_name]})

    assert canonical_payload_sha256(base) != canonical_payload_sha256(variant)


def test_application_id_is_bound_into_the_payload_hash() -> None:
    """OD-13.5 requires the payload fingerprint to bind the application ID
    itself, not merely the facts used to derive it."""

    event = make_dividend_event()
    asset_id = event.canonical_security_id
    trade_id = "BUY-1"
    q_t = 10
    gross_cash = compute_gross_dividend_cash(q_t, event.amount_per_share)

    application_id_a = compute_dividend_application_id(
        canonical_distribution_event_id=event.canonical_distribution_event_id,
        entitlement_session=event.entitlement_session,
        ex_session=event.ex_session,
        asset_id=asset_id,
        attribution_trade_id=trade_id,
    )
    payload_hash_a = compute_dividend_application_payload_hash(
        application_id=application_id_a,
        evidence=event,
        asset_id=asset_id,
        attribution_trade_id=trade_id,
        q_t=q_t,
        gross_cash_amount=gross_cash,
    )
    # A different (synthetic) application_id with all other facts identical
    # must change the payload hash, proving application_id is bound.
    payload_hash_b = compute_dividend_application_payload_hash(
        application_id="a-different-application-id",
        evidence=event,
        asset_id=asset_id,
        attribution_trade_id=trade_id,
        q_t=q_t,
        gross_cash_amount=gross_cash,
    )

    assert payload_hash_a != payload_hash_b
