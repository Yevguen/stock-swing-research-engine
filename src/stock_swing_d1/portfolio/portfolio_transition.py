"""Stateless atomic Phase 13 portfolio transition engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Sequence

from pydantic import ValidationError

from stock_swing_d1.data.ordinary_dividend_accounting import (
    CanonicalDividendAccountingEvidence,
)
from stock_swing_d1.portfolio.portfolio_dividend_events import (
    DividendApplicationOutcome,
    DividendApplicationStatus,
    DividendLedgerEntry,
    add_exact_decimal,
    compute_dividend_application_id,
    compute_dividend_application_payload_hash,
    compute_gross_dividend_cash,
    exact_decimal_times_int,
    subtract_exact_decimal,
)
from stock_swing_d1.portfolio.portfolio_errors import (
    DividendEvidenceError,
    DuplicateAssetApplicationError,
    DuplicateEventConflictError,
    InsufficientSettledCashError,
    InvalidExitQuantityError,
    OutOfOrderSessionError,
    OverdueSettlementError,
    PortfolioTransitionError,
    PositionAlreadyOpenError,
    PositionNotFoundError,
    SettlementIntegrityError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    ExecutionSide,
    PortfolioEventKind,
    PortfolioEventReference,
    PortfolioExecutionEvent,
    PortfolioLedgerEntry,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import (
    hash_execution_event,
    hash_portfolio_state,
    hash_settlement_event,
)
from stock_swing_d1.portfolio.portfolio_invariants import (
    PRIOR_SELL_APPLICATION_RANK,
    ROUND_TRIP_SELL_APPLICATION_RANK,
    PortfolioInvariantChecker,
    classify_session_applications,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PendingSettlement,
    PortfolioState,
    PortfolioTransitionResult,
)


@dataclass(frozen=True, slots=True)
class _LedgerDraft:
    event_type: PortfolioLedgerEventType
    source_event_id: str
    source_order_id: str | None
    asset_id: str | None
    quantity_delta: int | None
    fill_price: Decimal | None
    execution_cost: Decimal | None
    settlement_id: str | None
    settlement_session: date | None
    settled_cash_delta: Decimal
    pending_cash_delta: Decimal
    settled_cash_after: Decimal
    position_quantity_after: int | None
    source_payload_sha256: str


def _validated_execution(event: object) -> PortfolioExecutionEvent:
    if not isinstance(event, PortfolioExecutionEvent):
        raise PortfolioTransitionError(
            "execution_events must contain PortfolioExecutionEvent values"
        )
    try:
        validated = PortfolioExecutionEvent.model_validate(
            event.model_dump(mode="python")
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise PortfolioTransitionError(
            "execution event fails structural validation"
        ) from error
    if validated != event:
        raise PortfolioTransitionError("execution event is not canonical")
    return validated


def _validated_dividend_evidence(
    event: object, *, session: date
) -> CanonicalDividendAccountingEvidence:
    if not isinstance(event, CanonicalDividendAccountingEvidence):
        raise DividendEvidenceError(
            "dividend_evidence must contain CanonicalDividendAccountingEvidence "
            "values, not PortfolioExecutionEvent or another type"
        )
    try:
        validated = CanonicalDividendAccountingEvidence.model_validate(
            event.model_dump(mode="python")
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise DividendEvidenceError(
            "dividend evidence fails structural validation"
        ) from error
    if validated != event:
        raise DividendEvidenceError("dividend evidence is not canonical")
    if validated.ex_session != session:
        raise DividendEvidenceError(
            "dividend evidence ex_session must equal the current processed "
            "session"
        )
    return validated


@dataclass(frozen=True, slots=True)
class _DividendDraft:
    canonical_distribution_event_id: str
    application_id: str
    payload_sha256: str
    asset_id: str
    entitlement_session: date
    attribution_trade_id: str
    q_t: int
    d_h: Decimal
    amount_basis: str
    normalization_method_id: str
    normalization_scale: int
    normalization_rounding_mode: str
    normalization_arithmetic_mode: str
    normalization_inputs_fingerprint: str
    calendar_resolution_fingerprint: str
    canonical_distribution_snapshot_fingerprint: str
    currency: str
    gross_cash_amount: Decimal
    settled_cash_after: Decimal


class PortfolioTransitionEngine:
    """Apply already-authorized executions to one immutable portfolio state."""

    @staticmethod
    def transition(
        previous_state: PortfolioState,
        session: date,
        execution_events: Sequence[PortfolioExecutionEvent],
        dividend_evidence: Sequence[CanonicalDividendAccountingEvidence] = (),
    ) -> PortfolioTransitionResult:
        PortfolioInvariantChecker.validate_state(previous_state)
        if type(session) is not date:
            raise OutOfOrderSessionError("transition session must be a date")
        if (
            previous_state.as_of_session is not None
            and session <= previous_state.as_of_session
        ):
            raise OutOfOrderSessionError(
                "transition session must be later than the previous state session"
            )

        overdue = tuple(
            settlement
            for settlement in previous_state.pending_settlements
            if settlement.settlement_session < session
        )
        if overdue:
            raise OverdueSettlementError(
                "pending settlement is overdue: "
                + ", ".join(item.settlement_id for item in overdue)
            )

        state_hash_before = hash_portfolio_state(previous_state)
        settled_cash = previous_state.settled_cash
        positions = {
            position.asset_id: position for position in previous_state.open_positions
        }
        # Authoritative pre-X entitlement binding (OD-9.2/OD-9.3): captured from
        # the immutable pre-transition state, before any SELL or BUY on this
        # session is applied. Never re-derived from the mutated `positions`
        # dict below.
        pre_x_positions = dict(positions)
        pending = {
            settlement.settlement_id: settlement
            for settlement in previous_state.pending_settlements
        }
        applied = {
            (fingerprint.event_kind, fingerprint.event_id): fingerprint
            for fingerprint in previous_state.applied_events
        }
        drafts: list[_LedgerDraft] = []

        try:
            validated_dividend_events = tuple(
                _validated_dividend_evidence(item, session=session)
                for item in dividend_evidence
            )
        except TypeError as error:
            raise DividendEvidenceError(
                "dividend_evidence must be a sequence"
            ) from error
        dividend_event_ids = tuple(
            event.canonical_distribution_event_id
            for event in validated_dividend_events
        )
        if len(set(dividend_event_ids)) != len(dividend_event_ids):
            raise DividendEvidenceError(
                "supplied dividend evidence event identities must be unique"
            )

        due_settlements = sorted(
            (
                settlement
                for settlement in previous_state.pending_settlements
                if settlement.settlement_session == session
            ),
            key=lambda item: item.settlement_id,
        )
        for settlement in due_settlements:
            payload_hash = hash_settlement_event(settlement)
            key = (PortfolioEventKind.SETTLEMENT, settlement.settlement_id)
            known = applied.get(key)
            if known is not None:
                if known.payload_sha256 != payload_hash:
                    raise DuplicateEventConflictError(
                        "settlement identity has a conflicting payload: "
                        f"{settlement.settlement_id}"
                    )
                raise SettlementIntegrityError(
                    "an already-applied settlement cannot remain pending: "
                    f"{settlement.settlement_id}"
                )

            # Exact-arithmetic hardening (Residual B): settlement.amount can
            # carry more significant digits than the ambient Decimal context
            # allows (it is itself an exact SELL net-proceeds value once
            # Residual A below is hardened), so this settled-cash mutation
            # must be exact too -- matching the invariant checker's replay
            # accumulator, which already applies add_exact_decimal to every
            # ledger entry kind, this one included.
            settled_cash = add_exact_decimal(settled_cash, settlement.amount)
            del pending[settlement.settlement_id]
            applied[key] = AppliedEventFingerprint(
                event_kind=PortfolioEventKind.SETTLEMENT,
                event_id=settlement.settlement_id,
                payload_sha256=payload_hash,
            )
            drafts.append(
                _LedgerDraft(
                    event_type=PortfolioLedgerEventType.SETTLEMENT_APPLIED,
                    source_event_id=settlement.settlement_id,
                    source_order_id=None,
                    asset_id=settlement.asset_id,
                    quantity_delta=0,
                    fill_price=None,
                    execution_cost=None,
                    settlement_id=settlement.settlement_id,
                    settlement_session=settlement.settlement_session,
                    settled_cash_delta=settlement.amount,
                    pending_cash_delta=-settlement.amount,
                    settled_cash_after=settled_cash,
                    position_quantity_after=None,
                    source_payload_sha256=payload_hash,
                )
            )

        try:
            incoming = tuple(
                _validated_execution(item) for item in execution_events
            )
        except TypeError as error:
            raise PortfolioTransitionError(
                "execution_events must be a sequence"
            ) from error

        new_by_id: dict[str, tuple[PortfolioExecutionEvent, str]] = {}
        replayed_events: set[tuple[PortfolioEventKind, str]] = set()
        for event in incoming:
            payload_hash = hash_execution_event(event)
            key = (PortfolioEventKind.EXECUTION, event.execution_id)
            known = applied.get(key)
            if known is not None:
                if known.payload_sha256 == payload_hash:
                    replayed_events.add(
                        (PortfolioEventKind.EXECUTION, event.execution_id)
                    )
                    continue
                raise DuplicateEventConflictError(
                    "execution identity has a conflicting payload: "
                    f"{event.execution_id}"
                )

            batch_known = new_by_id.get(event.execution_id)
            if batch_known is not None:
                if batch_known[1] == payload_hash:
                    replayed_events.add(
                        (PortfolioEventKind.EXECUTION, event.execution_id)
                    )
                    continue
                raise DuplicateEventConflictError(
                    "execution identity has conflicting payloads in one batch: "
                    f"{event.execution_id}"
                )
            new_by_id[event.execution_id] = (event, payload_hash)

        new_events = tuple(new_by_id.values())
        for event, _ in new_events:
            if event.session != session:
                raise OutOfOrderSessionError(
                    "new execution session must equal transition session: "
                    f"{event.execution_id}"
                )

        # Task 5C-C: classify the distinct new applications by the immutable
        # session-start position set (P, captured above as pre_x_positions)
        # and the batch's applied BUY asset set (B) through the single shared
        # Phase 13 rule.  REPLAYED presentations were filtered out above and
        # never reach this classification or its cardinality.
        classification = classify_session_applications(
            session_start_asset_ids=frozenset(pre_x_positions),
            buy_asset_ids=tuple(
                item[0].asset_id
                for item in new_events
                if item[0].side is ExecutionSide.BUY
            ),
            sell_asset_ids=tuple(
                item[0].asset_id
                for item in new_events
                if item[0].side is ExecutionSide.SELL
            ),
        )
        if classification.duplicate_buy_asset_ids:
            raise DuplicateAssetApplicationError(
                "one asset cannot receive two BUY applications in one "
                "transition: "
                + ", ".join(classification.duplicate_buy_asset_ids)
            )
        if classification.duplicate_sell_asset_ids:
            raise DuplicateAssetApplicationError(
                "one asset cannot receive two SELL applications in one "
                "transition: "
                + ", ".join(classification.duplicate_sell_asset_ids)
            )
        if classification.unclassifiable_sell_asset_ids:
            raise PositionNotFoundError(
                "no open position for SELL asset "
                + ", ".join(classification.unclassifiable_sell_asset_ids)
            )

        def _sells_of_rank(rank: int) -> list[tuple[PortfolioExecutionEvent, str]]:
            return sorted(
                (
                    item
                    for item in new_events
                    if item[0].side is ExecutionSide.SELL
                    and classification.sell_rank_by_asset[item[0].asset_id]
                    == rank
                ),
                key=lambda item: (item[0].asset_id, item[0].execution_id),
            )

        prior_sells = _sells_of_rank(PRIOR_SELL_APPLICATION_RANK)
        round_trip_sells = _sells_of_rank(ROUND_TRIP_SELL_APPLICATION_RANK)
        buys = sorted(
            (item for item in new_events if item[0].side is ExecutionSide.BUY),
            key=lambda item: (item[0].asset_id, item[0].execution_id),
        )

        def _apply_sell(event: PortfolioExecutionEvent, payload_hash: str) -> None:
            """The one authoritative full-exit SELL application.

            Shared verbatim by PRIOR and ROUND_TRIP SELLs: the position must
            exist by the time the SELL applies (a ROUND_TRIP SELL's position
            was created by this same transition's BUY), the quantity must
            equal it, proceeds are pending (never settled) and no netting
            occurs.
            """

            position = positions.get(event.asset_id)
            if position is None:
                raise PositionNotFoundError(
                    f"no open position for SELL asset {event.asset_id}"
                )
            if event.quantity != position.quantity:
                raise InvalidExitQuantityError(
                    "SELL quantity must exactly equal the open-position quantity"
                )
            # Exact-arithmetic hardening (Residual A): event.fill_price
            # carries no enforced significant-digit bound, so this
            # multiplication/subtraction -- which becomes the authoritative
            # PendingSettlement.amount and pending_cash_delta -- uses the
            # shared exact helpers, not ambient-context Decimal `*`/`-`.
            net_proceeds = subtract_exact_decimal(
                exact_decimal_times_int(event.fill_price, event.quantity),
                event.execution_cost,
            )
            if net_proceeds <= 0:
                raise SettlementIntegrityError("SELL net proceeds must be positive")

            settlement_id = event.settlement_id
            settlement_session = event.settlement_session
            if settlement_id is None or settlement_session is None:
                raise SettlementIntegrityError(
                    "SELL execution is missing settlement identity or session"
                )
            settlement_key = (PortfolioEventKind.SETTLEMENT, settlement_id)
            if settlement_id in pending or settlement_key in applied:
                raise SettlementIntegrityError(
                    f"settlement identity is not unique: {settlement_id}"
                )

            generated_settlement = PendingSettlement(
                settlement_id=settlement_id,
                source_execution_id=event.execution_id,
                asset_id=event.asset_id,
                amount=net_proceeds,
                trade_session=event.session,
                settlement_session=settlement_session,
            )
            del positions[event.asset_id]
            pending[settlement_id] = generated_settlement
            applied[(PortfolioEventKind.EXECUTION, event.execution_id)] = (
                AppliedEventFingerprint(
                    event_kind=PortfolioEventKind.EXECUTION,
                    event_id=event.execution_id,
                    payload_sha256=payload_hash,
                )
            )
            drafts.append(
                _LedgerDraft(
                    event_type=PortfolioLedgerEventType.SELL_APPLIED,
                    source_event_id=event.execution_id,
                    source_order_id=event.source_order_id,
                    asset_id=event.asset_id,
                    quantity_delta=-event.quantity,
                    fill_price=event.fill_price,
                    execution_cost=event.execution_cost,
                    settlement_id=settlement_id,
                    settlement_session=settlement_session,
                    settled_cash_delta=Decimal("0"),
                    pending_cash_delta=net_proceeds,
                    settled_cash_after=settled_cash,
                    position_quantity_after=0,
                    source_payload_sha256=payload_hash,
                )
            )

        for event, payload_hash in prior_sells:
            _apply_sell(event, payload_hash)

        for event, payload_hash in buys:
            if event.asset_id in positions:
                raise PositionAlreadyOpenError(
                    f"an open position already exists for {event.asset_id}"
                )
            # Exact-arithmetic hardening (Residual B): event.fill_price
            # carries no enforced significant-digit bound, so this
            # multiplication/addition -- which becomes the authoritative
            # cost_basis -- uses the shared exact helpers, not
            # ambient-context Decimal `*`/`+`.
            required_cash = add_exact_decimal(
                exact_decimal_times_int(event.fill_price, event.quantity),
                event.execution_cost,
            )
            if required_cash > settled_cash:
                raise InsufficientSettledCashError(
                    f"BUY {event.execution_id} requires more settled cash than available"
                )

            # required_cash can now carry more significant digits than
            # before this hardening (it is no longer prematurely rounded
            # at computation time), so the settled-cash mutation it
            # drives must also be exact -- otherwise the resulting
            # PortfolioState (and its hash) would still depend on
            # ambient Decimal context even though cost_basis itself does
            # not, exactly as OD-12.1's dividend-credit site already
            # requires via the same `add_exact_decimal` below.
            settled_cash = subtract_exact_decimal(settled_cash, required_cash)
            positions[event.asset_id] = OpenPosition(
                asset_id=event.asset_id,
                quantity=event.quantity,
                entry_session=event.session,
                entry_price=event.fill_price,
                entry_execution_id=event.execution_id,
                entry_execution_cost=event.execution_cost,
                cost_basis=required_cash,
            )
            applied[(PortfolioEventKind.EXECUTION, event.execution_id)] = (
                AppliedEventFingerprint(
                    event_kind=PortfolioEventKind.EXECUTION,
                    event_id=event.execution_id,
                    payload_sha256=payload_hash,
                )
            )
            drafts.append(
                _LedgerDraft(
                    event_type=PortfolioLedgerEventType.BUY_APPLIED,
                    source_event_id=event.execution_id,
                    source_order_id=event.source_order_id,
                    asset_id=event.asset_id,
                    quantity_delta=event.quantity,
                    fill_price=event.fill_price,
                    execution_cost=event.execution_cost,
                    settlement_id=None,
                    settlement_session=None,
                    # Exact-arithmetic hardening (Residual B): unary `-`
                    # on a Decimal also passes through `_fix()` and can
                    # round once required_cash carries enough significant
                    # digits, so this uses exact negation via subtraction
                    # from zero rather than Python's unary minus.
                    settled_cash_delta=subtract_exact_decimal(
                        Decimal("0"), required_cash
                    ),
                    pending_cash_delta=Decimal("0"),
                    settled_cash_after=settled_cash,
                    position_quantity_after=event.quantity,
                    source_payload_sha256=payload_hash,
                )
            )

        # Task 5C-C step 4b: same-session round-trip SELLs apply after every
        # BUY, with the identical authoritative SELL semantics, so the
        # position each one closes was created by this same transition.
        for event, payload_hash in round_trip_sells:
            _apply_sell(event, payload_hash)

        # Step D: ordinary-dividend applications, ordered ascending by
        # (canonical_security_id, canonical_distribution_event_id) (OD-20.4).
        # Entitlement uses only the pre-X binding captured above; the mutated
        # `positions` dict (reflecting this session's own SELL/BUY/ROUND_TRIP
        # SELL rows) is never consulted here.
        dividend_drafts: list[_DividendDraft] = []
        dividend_outcomes: list[DividendApplicationOutcome] = []
        for event in sorted(
            validated_dividend_events,
            key=lambda item: (
                item.canonical_security_id,
                item.canonical_distribution_event_id,
            ),
        ):
            asset_id = event.canonical_security_id
            pre_x_position = pre_x_positions.get(asset_id)
            q_t = pre_x_position.quantity if pre_x_position is not None else 0
            trade_id = (
                pre_x_position.entry_execution_id
                if pre_x_position is not None
                else None
            )

            application_id = compute_dividend_application_id(
                canonical_distribution_event_id=(
                    event.canonical_distribution_event_id
                ),
                entitlement_session=event.entitlement_session,
                ex_session=event.ex_session,
                asset_id=asset_id,
                attribution_trade_id=trade_id,
            )
            gross_cash_amount = compute_gross_dividend_cash(
                q_t, event.amount_per_share
            )
            payload_hash = compute_dividend_application_payload_hash(
                application_id=application_id,
                evidence=event,
                asset_id=asset_id,
                attribution_trade_id=trade_id,
                q_t=q_t,
                gross_cash_amount=gross_cash_amount,
            )

            if q_t == 0:
                dividend_outcomes.append(
                    DividendApplicationOutcome(
                        canonical_distribution_event_id=(
                            event.canonical_distribution_event_id
                        ),
                        application_id=application_id,
                        payload_sha256=payload_hash,
                        asset_id=asset_id,
                        entitlement_session=event.entitlement_session,
                        ex_session=event.ex_session,
                        q_t=0,
                        attribution_trade_id=None,
                        status=DividendApplicationStatus.NO_OP_NOT_ENTITLED,
                    )
                )
                continue

            applied_key = (PortfolioEventKind.DIVIDEND, application_id)
            known = applied.get(applied_key)
            if known is not None:
                if known.payload_sha256 != payload_hash:
                    raise DuplicateEventConflictError(
                        "dividend application identity has a conflicting "
                        f"payload: {application_id}"
                    )
                dividend_outcomes.append(
                    DividendApplicationOutcome(
                        canonical_distribution_event_id=(
                            event.canonical_distribution_event_id
                        ),
                        application_id=application_id,
                        payload_sha256=payload_hash,
                        asset_id=asset_id,
                        entitlement_session=event.entitlement_session,
                        ex_session=event.ex_session,
                        q_t=q_t,
                        attribution_trade_id=trade_id,
                        status=DividendApplicationStatus.REPLAYED,
                    )
                )
                replayed_events.add(applied_key)
                continue

            settled_cash = add_exact_decimal(settled_cash, gross_cash_amount)
            applied[applied_key] = AppliedEventFingerprint(
                event_kind=PortfolioEventKind.DIVIDEND,
                event_id=application_id,
                payload_sha256=payload_hash,
            )
            dividend_drafts.append(
                _DividendDraft(
                    canonical_distribution_event_id=(
                        event.canonical_distribution_event_id
                    ),
                    application_id=application_id,
                    payload_sha256=payload_hash,
                    asset_id=asset_id,
                    entitlement_session=event.entitlement_session,
                    attribution_trade_id=trade_id,
                    q_t=q_t,
                    d_h=event.amount_per_share,
                    amount_basis=event.amount_basis,
                    normalization_method_id=event.normalization_method_id,
                    normalization_scale=event.normalization_scale,
                    normalization_rounding_mode=(
                        event.normalization_rounding_mode
                    ),
                    normalization_arithmetic_mode=(
                        event.normalization_arithmetic_mode
                    ),
                    normalization_inputs_fingerprint=(
                        event.normalization_inputs_fingerprint
                    ),
                    calendar_resolution_fingerprint=(
                        event.calendar_resolution_fingerprint
                    ),
                    canonical_distribution_snapshot_fingerprint=(
                        event.canonical_distribution_snapshot_fingerprint
                    ),
                    currency=event.currency,
                    gross_cash_amount=gross_cash_amount,
                    settled_cash_after=settled_cash,
                )
            )
            dividend_outcomes.append(
                DividendApplicationOutcome(
                    canonical_distribution_event_id=(
                        event.canonical_distribution_event_id
                    ),
                    application_id=application_id,
                    payload_sha256=payload_hash,
                    asset_id=asset_id,
                    entitlement_session=event.entitlement_session,
                    ex_session=event.ex_session,
                    q_t=q_t,
                    attribution_trade_id=trade_id,
                    status=DividendApplicationStatus.APPLIED,
                )
            )

        resulting_state = PortfolioState(
            base_currency=previous_state.base_currency,
            as_of_session=session,
            state_version=previous_state.state_version + 1,
            settled_cash=settled_cash,
            open_positions=tuple(positions.values()),
            pending_settlements=tuple(pending.values()),
            applied_events=tuple(applied.values()),
        )
        PortfolioInvariantChecker.validate_state(resulting_state)
        state_hash_after = hash_portfolio_state(resulting_state)
        ledger_entries = tuple(
            PortfolioLedgerEntry(
                session=session,
                sequence_in_session=sequence,
                event_type=draft.event_type,
                source_event_id=draft.source_event_id,
                source_order_id=draft.source_order_id,
                asset_id=draft.asset_id,
                quantity_delta=draft.quantity_delta,
                fill_price=draft.fill_price,
                execution_cost=draft.execution_cost,
                settlement_id=draft.settlement_id,
                settlement_session=draft.settlement_session,
                settled_cash_delta=draft.settled_cash_delta,
                pending_cash_delta=draft.pending_cash_delta,
                settled_cash_after=draft.settled_cash_after,
                position_quantity_after=draft.position_quantity_after,
                source_payload_sha256=draft.source_payload_sha256,
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
            )
            for sequence, draft in enumerate(drafts)
        )
        dividend_ledger_entries = tuple(
            DividendLedgerEntry(
                session=session,
                sequence_in_session=sequence,
                application_id=draft.application_id,
                canonical_distribution_event_id=(
                    draft.canonical_distribution_event_id
                ),
                canonical_distribution_snapshot_fingerprint=(
                    draft.canonical_distribution_snapshot_fingerprint
                ),
                asset_id=draft.asset_id,
                entitlement_session=draft.entitlement_session,
                attribution_trade_id=draft.attribution_trade_id,
                q_t=draft.q_t,
                d_h=draft.d_h,
                amount_basis=draft.amount_basis,
                normalization_method_id=draft.normalization_method_id,
                normalization_scale=draft.normalization_scale,
                normalization_rounding_mode=draft.normalization_rounding_mode,
                normalization_arithmetic_mode=(
                    draft.normalization_arithmetic_mode
                ),
                normalization_inputs_fingerprint=(
                    draft.normalization_inputs_fingerprint
                ),
                calendar_resolution_fingerprint=(
                    draft.calendar_resolution_fingerprint
                ),
                currency=draft.currency,
                gross_cash_amount=draft.gross_cash_amount,
                settled_cash_delta=draft.gross_cash_amount,
                settled_cash_after=draft.settled_cash_after,
                source_payload_sha256=draft.payload_sha256,
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
            )
            for sequence, draft in enumerate(dividend_drafts)
        )

        PortfolioInvariantChecker.validate_transition(
            previous_state,
            resulting_state,
            ledger_entries,
            dividend_ledger_entries=dividend_ledger_entries,
        )
        PortfolioInvariantChecker.validate_execution_ledger_provenance(
            incoming,
            ledger_entries,
        )
        PortfolioInvariantChecker.validate_dividend_application(
            session=session,
            previous_state=previous_state,
            dividend_evidence=validated_dividend_events,
            dividend_ledger_entries=dividend_ledger_entries,
            dividend_outcomes=tuple(dividend_outcomes),
        )
        return PortfolioTransitionResult(
            session=session,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            resulting_state=resulting_state,
            ledger_entries=ledger_entries,
            newly_applied_events=tuple(
                PortfolioEventReference(
                    event_kind=(
                        PortfolioEventKind.SETTLEMENT
                        if draft.event_type
                        is PortfolioLedgerEventType.SETTLEMENT_APPLIED
                        else PortfolioEventKind.EXECUTION
                    ),
                    event_id=draft.source_event_id,
                )
                for draft in drafts
            )
            + tuple(
                PortfolioEventReference(
                    event_kind=PortfolioEventKind.DIVIDEND,
                    event_id=draft.application_id,
                )
                for draft in dividend_drafts
            ),
            replayed_events=tuple(
                PortfolioEventReference(event_kind=kind, event_id=event_id)
                for kind, event_id in sorted(
                    replayed_events,
                    key=lambda item: (item[0].value, item[1]),
                )
            ),
            dividend_ledger_entries=dividend_ledger_entries,
            dividend_outcomes=tuple(dividend_outcomes),
        )


__all__ = ["PortfolioTransitionEngine"]
