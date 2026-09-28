"""Cross-object and transition invariants for Phase 13 portfolio state."""

from __future__ import annotations

from collections import Counter
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
    DividendProvenanceError,
    PortfolioStateError,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    ExecutionSide,
    PortfolioEventKind,
    PortfolioExecutionEvent,
    PortfolioLedgerEntry,
    PortfolioLedgerEventType,
)
from stock_swing_d1.portfolio.portfolio_hashing import (
    hash_execution_event,
    hash_portfolio_state,
    hash_settlement_event,
)
from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PendingSettlement,
    PortfolioSessionSnapshot,
    PortfolioState,
)


def _fail(message: str) -> None:
    raise PortfolioStateError(message)


def _fail_dividend(message: str) -> None:
    raise DividendProvenanceError(message)


# ---------------------------------------------------------------------------
# Task 5C-C: the one authoritative application-class rule.
#
# Within one session the distinct NEW economic applications (after identical-
# duplicate elimination and historical-replay resolution) are applied in
# this order, each class ascending by (asset_id, execution_id):
#
#     SETTLEMENT (0) -> PRIOR SELL (1) -> BUY (2) -> ROUND_TRIP SELL (3)
#     -> dividend
#
# A SELL of asset X is PRIOR iff X is in the immutable session-start open
# positions (P); it is ROUND_TRIP iff X is not in P and X is bought by a
# distinct new APPLIED BUY in the same session (B); otherwise it has no class.
# Per session, per asset there is at most one applied BUY and at most one
# applied SELL.  The rule is a pure function of P and the batch's asset sets,
# never of input or row order, and it is the single implementation consumed
# by the live transition engine and by both persisted-replay validators so
# the two paths cannot drift.
# ---------------------------------------------------------------------------

SETTLEMENT_APPLICATION_RANK = 0
PRIOR_SELL_APPLICATION_RANK = 1
BUY_APPLICATION_RANK = 2
ROUND_TRIP_SELL_APPLICATION_RANK = 3


@dataclass(frozen=True, slots=True)
class SessionApplicationClassification:
    """Per-asset SELL ranks plus every cardinality/classification defect."""

    sell_rank_by_asset: dict[str, int]
    duplicate_buy_asset_ids: tuple[str, ...]
    duplicate_sell_asset_ids: tuple[str, ...]
    unclassifiable_sell_asset_ids: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not (
            self.duplicate_buy_asset_ids
            or self.duplicate_sell_asset_ids
            or self.unclassifiable_sell_asset_ids
        )


def classify_session_applications(
    *,
    session_start_asset_ids: frozenset[str],
    buy_asset_ids: Sequence[str],
    sell_asset_ids: Sequence[str],
) -> SessionApplicationClassification:
    """Classify one session's distinct new applications by P and B only."""

    buy_counts = Counter(buy_asset_ids)
    sell_counts = Counter(sell_asset_ids)
    applied_buy_assets = frozenset(buy_counts)
    sell_rank_by_asset: dict[str, int] = {}
    unclassifiable: list[str] = []
    for asset_id in sorted(sell_counts):
        if asset_id in session_start_asset_ids:
            sell_rank_by_asset[asset_id] = PRIOR_SELL_APPLICATION_RANK
        elif asset_id in applied_buy_assets:
            sell_rank_by_asset[asset_id] = ROUND_TRIP_SELL_APPLICATION_RANK
        else:
            unclassifiable.append(asset_id)
    return SessionApplicationClassification(
        sell_rank_by_asset=sell_rank_by_asset,
        duplicate_buy_asset_ids=tuple(
            sorted(asset for asset, count in buy_counts.items() if count > 1)
        ),
        duplicate_sell_asset_ids=tuple(
            sorted(asset for asset, count in sell_counts.items() if count > 1)
        ),
        unclassifiable_sell_asset_ids=tuple(unclassifiable),
    )


def _classify_session_ledger(
    *,
    session_start_state: PortfolioState,
    ledger: Sequence[PortfolioLedgerEntry],
) -> SessionApplicationClassification:
    """Apply the shared rule to one session's actual application rows.

    P is the reconstructed immutable session-start state and B the session's
    ``BUY_APPLIED`` rows; every defect fails closed with the standard
    ``PortfolioStateError`` so live and persisted validation agree.
    """

    classification = classify_session_applications(
        session_start_asset_ids=frozenset(
            position.asset_id for position in session_start_state.open_positions
        ),
        buy_asset_ids=tuple(
            entry.asset_id
            for entry in ledger
            if entry.event_type is PortfolioLedgerEventType.BUY_APPLIED
            and entry.asset_id is not None
        ),
        sell_asset_ids=tuple(
            entry.asset_id
            for entry in ledger
            if entry.event_type is PortfolioLedgerEventType.SELL_APPLIED
            and entry.asset_id is not None
        ),
    )
    if classification.duplicate_buy_asset_ids:
        _fail(
            "one asset cannot receive two BUY applications in one session: "
            + ", ".join(classification.duplicate_buy_asset_ids)
        )
    if classification.duplicate_sell_asset_ids:
        _fail(
            "one asset cannot receive two SELL applications in one session: "
            + ", ".join(classification.duplicate_sell_asset_ids)
        )
    if classification.unclassifiable_sell_asset_ids:
        _fail(
            "a SELL application is neither a prior-position SELL nor a "
            "same-session round-trip SELL: "
            + ", ".join(classification.unclassifiable_sell_asset_ids)
        )
    return classification


def _ledger_ordering_key(
    entry: PortfolioLedgerEntry,
    classification: SessionApplicationClassification,
) -> tuple[object, ...]:
    if entry.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED:
        return (SETTLEMENT_APPLICATION_RANK, entry.source_event_id)
    if entry.event_type is PortfolioLedgerEventType.SELL_APPLIED:
        return (
            classification.sell_rank_by_asset[entry.asset_id],
            entry.asset_id,
            entry.source_event_id,
        )
    if entry.event_type is PortfolioLedgerEventType.BUY_APPLIED:
        return (BUY_APPLICATION_RANK, entry.asset_id, entry.source_event_id)
    _fail("unsupported ledger event type")  # pragma: no cover
    raise AssertionError  # pragma: no cover


def _validated_state(state: object) -> PortfolioState:
    if not isinstance(state, PortfolioState):
        _fail("state must be a PortfolioState")
    try:
        validated = PortfolioState.model_validate(state.model_dump(mode="python"))
    except (ValidationError, TypeError, ValueError) as error:
        raise PortfolioStateError("state fails structural validation") from error
    if state != validated:
        _fail("state must use canonical collection ordering")
    return validated


class PortfolioInvariantChecker:
    """Fail-closed state, transition, and provenance validation.

    Standalone state validation proves references to applied EXECUTION
    fingerprints. Execution side is proven only when canonical execution
    payloads and ledger rows are available to provenance/transition validation.
    """

    @staticmethod
    def validate_state(state: PortfolioState) -> None:
        state = _validated_state(state)

        if (state.as_of_session is None) != (state.state_version == 0):
            _fail("as_of_session is None if and only if state_version is zero")
        if state.as_of_session is None and (
            state.open_positions
            or state.pending_settlements
            or state.applied_events
        ):
            _fail(
                "a virgin state cannot contain positions, pending settlements, "
                "or applied events"
            )

        execution_ids = {
            fingerprint.event_id
            for fingerprint in state.applied_events
            if fingerprint.event_kind is PortfolioEventKind.EXECUTION
        }
        settlement_ids = {
            fingerprint.event_id
            for fingerprint in state.applied_events
            if fingerprint.event_kind is PortfolioEventKind.SETTLEMENT
        }

        position_execution_ids = {
            position.entry_execution_id for position in state.open_positions
        }
        pending_execution_ids = {
            settlement.source_execution_id
            for settlement in state.pending_settlements
        }

        missing_entries = position_execution_ids - execution_ids
        if missing_entries:
            _fail(
                "every open position must reference an applied EXECUTION "
                "fingerprint"
            )
        missing_exits = pending_execution_ids - execution_ids
        if missing_exits:
            _fail(
                "every pending settlement must reference an applied EXECUTION "
                "fingerprint"
            )
        if position_execution_ids & pending_execution_ids:
            _fail(
                "one applied EXECUTION fingerprint cannot be referenced by both "
                "an open position and a pending settlement"
            )

        duplicate_settlement_ids = {
            settlement.settlement_id
            for settlement in state.pending_settlements
        } & settlement_ids
        if duplicate_settlement_ids:
            _fail("an already-applied settlement cannot remain pending")

        if state.as_of_session is not None:
            if any(
                position.entry_session > state.as_of_session
                for position in state.open_positions
            ):
                _fail("an open position cannot begin after the state session")
            if any(
                settlement.trade_session > state.as_of_session
                for settlement in state.pending_settlements
            ):
                _fail("a pending settlement cannot trade after the state session")

    @staticmethod
    def validate_execution_ledger_provenance(
        execution_events: Sequence[PortfolioExecutionEvent],
        ledger_entries: Sequence[PortfolioLedgerEntry],
    ) -> None:
        """Validate external executions against BUY/SELL ledger provenance.

        Exact duplicate execution inputs are accepted. External execution inputs
        without a ledger row are also accepted because they may be replays.
        Internally generated settlement rows have no external execution source
        and are intentionally outside this relationship check.
        """

        try:
            events = tuple(execution_events)
            ledger = tuple(ledger_entries)
        except TypeError as error:
            raise PortfolioStateError(
                "execution_events and ledger_entries must be iterable"
            ) from error

        event_by_id: dict[str, tuple[PortfolioExecutionEvent, str]] = {}
        for event in events:
            event = PortfolioInvariantChecker._validate_execution_event(event)
            payload_hash = hash_execution_event(event)
            known = event_by_id.get(event.execution_id)
            if known is not None:
                if known[1] != payload_hash:
                    _fail(
                        "persisted execution identity has conflicting payloads: "
                        f"{event.execution_id}"
                    )
                continue
            event_by_id[event.execution_id] = (event, payload_hash)

        applied_execution_ids: set[str] = set()
        for entry in ledger:
            entry = PortfolioInvariantChecker._validate_ledger_entry(entry)
            if entry.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED:
                continue

            if entry.source_event_id in applied_execution_ids:
                _fail(
                    "one execution identity cannot have multiple application "
                    f"ledger rows: {entry.source_event_id}"
                )
            applied_execution_ids.add(entry.source_event_id)

            source = event_by_id.get(entry.source_event_id)
            if source is None:
                _fail(
                    "BUY/SELL ledger row is missing its source execution: "
                    f"{entry.source_event_id}"
                )
            event, payload_hash = source
            if payload_hash != entry.source_payload_sha256:
                _fail(
                    "ledger source hash does not match the canonical execution "
                    f"payload: {entry.source_event_id}"
                )

            expected_type = (
                PortfolioLedgerEventType.BUY_APPLIED
                if event.side is ExecutionSide.BUY
                else PortfolioLedgerEventType.SELL_APPLIED
            )
            if entry.event_type is not expected_type:
                _fail(
                    "execution side does not match ledger event type: "
                    f"{entry.source_event_id}"
                )

            expected_quantity_delta = (
                event.quantity
                if event.side is ExecutionSide.BUY
                else -event.quantity
            )
            facts_match = (
                entry.source_event_id == event.execution_id
                and entry.source_order_id == event.source_order_id
                and entry.session == event.session
                and entry.asset_id == event.asset_id
                and entry.quantity_delta == expected_quantity_delta
                and entry.fill_price == event.fill_price
                and entry.execution_cost == event.execution_cost
                and entry.settlement_id == event.settlement_id
                and entry.settlement_session == event.settlement_session
            )
            if not facts_match:
                _fail(
                    "ledger facts do not match the canonical execution payload: "
                    f"{entry.source_event_id}"
                )

    @staticmethod
    def validate_transition(
        previous_state: PortfolioState,
        new_state: PortfolioState,
        ledger_entries: tuple[PortfolioLedgerEntry, ...]
        | list[PortfolioLedgerEntry],
        dividend_ledger_entries: Sequence[DividendLedgerEntry] = (),
    ) -> None:
        PortfolioInvariantChecker.validate_state(previous_state)
        PortfolioInvariantChecker.validate_state(new_state)

        if new_state.as_of_session is None:
            _fail("a successful transition must have a completed session")
        if (
            previous_state.as_of_session is not None
            and new_state.as_of_session <= previous_state.as_of_session
        ):
            _fail("a transition session must follow the previous state session")
        if new_state.state_version != previous_state.state_version + 1:
            _fail("a successful transition must increment state_version once")

        try:
            ledger = tuple(ledger_entries)
        except TypeError as error:
            raise PortfolioStateError("ledger_entries must be iterable") from error
        for entry in ledger:
            if not isinstance(entry, PortfolioLedgerEntry):
                _fail("ledger_entries must contain PortfolioLedgerEntry values")
            try:
                validated = PortfolioLedgerEntry.model_validate(
                    entry.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise PortfolioStateError(
                    "ledger entry fails structural validation"
                ) from error
            if entry != validated:
                _fail("ledger entry is not canonical")

        if tuple(entry.sequence_in_session for entry in ledger) != tuple(
            range(len(ledger))
        ):
            _fail("ledger sequence_in_session must be consecutive from zero")
        if any(entry.session != new_state.as_of_session for entry in ledger):
            _fail("every ledger entry must belong to the transition session")

        state_hash_before = hash_portfolio_state(previous_state)
        state_hash_after = hash_portfolio_state(new_state)
        if any(
            entry.state_hash_before != state_hash_before
            or entry.state_hash_after != state_hash_after
            for entry in ledger
        ):
            _fail("ledger state hashes must identify the complete transition")

        settled_cash = previous_state.settled_cash
        positions = {
            position.asset_id: position for position in previous_state.open_positions
        }
        pending = {
            settlement.settlement_id: settlement
            for settlement in previous_state.pending_settlements
        }
        applied = {
            (fingerprint.event_kind, fingerprint.event_id): fingerprint
            for fingerprint in previous_state.applied_events
        }

        previous_keys = set(applied)
        # Task 5C-C: classify this session's applications from the immutable
        # session-start state (P) and the session's BUY rows (B) before any
        # row is replayed, so the expected order never depends on row order.
        classification = _classify_session_ledger(
            session_start_state=previous_state, ledger=ledger
        )
        ordering_keys: list[tuple[object, ...]] = []

        for entry in ledger:
            # BUY entries now carry an exact settled_cash_delta (Residual
            # B hardening); this replay accumulator must add it exactly
            # too, matching the dividend-row accumulator just below,
            # rather than reintroducing ambient-context rounding here.
            settled_cash = add_exact_decimal(settled_cash, entry.settled_cash_delta)
            if settled_cash < 0 or entry.settled_cash_after != settled_cash:
                _fail("ledger settled-cash arithmetic is inconsistent")

            ordering_keys.append(_ledger_ordering_key(entry, classification))
            if entry.event_type is PortfolioLedgerEventType.SETTLEMENT_APPLIED:
                PortfolioInvariantChecker._apply_settlement_ledger(
                    entry=entry,
                    positions=positions,
                    pending=pending,
                    applied=applied,
                )
            elif entry.event_type is PortfolioLedgerEventType.SELL_APPLIED:
                PortfolioInvariantChecker._apply_sell_ledger(
                    entry=entry,
                    positions=positions,
                    pending=pending,
                    applied=applied,
                )
            elif entry.event_type is PortfolioLedgerEventType.BUY_APPLIED:
                PortfolioInvariantChecker._apply_buy_ledger(
                    entry=entry,
                    positions=positions,
                    pending=pending,
                    applied=applied,
                )
            else:  # pragma: no cover - enum/model validation already closes this
                _fail("unsupported ledger event type")

        if ordering_keys != sorted(ordering_keys):
            _fail("ledger entries are not in canonical session order")

        dividend_rows = tuple(dividend_ledger_entries)
        for row in dividend_rows:
            if not isinstance(row, DividendLedgerEntry):
                _fail(
                    "dividend_ledger_entries must contain DividendLedgerEntry "
                    "values"
                )
            try:
                rebuilt_row = DividendLedgerEntry.model_validate(
                    row.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise PortfolioStateError(
                    "dividend ledger entry fails structural validation"
                ) from error
            if row != rebuilt_row:
                _fail("dividend ledger entry is not canonical")

        if tuple(row.sequence_in_session for row in dividend_rows) != tuple(
            range(len(dividend_rows))
        ):
            _fail(
                "dividend ledger sequence_in_session must be consecutive from "
                "zero"
            )
        if any(row.session != new_state.as_of_session for row in dividend_rows):
            _fail("every dividend ledger entry must belong to the transition session")
        if any(
            row.state_hash_before != state_hash_before
            or row.state_hash_after != state_hash_after
            for row in dividend_rows
        ):
            _fail("dividend ledger state hashes must identify the complete transition")

        dividend_order_keys = [
            (row.asset_id, row.canonical_distribution_event_id)
            for row in dividend_rows
        ]
        if dividend_order_keys != sorted(dividend_order_keys):
            _fail(
                "dividend ledger entries must be ordered by (asset_id, "
                "canonical_distribution_event_id)"
            )

        for row in dividend_rows:
            settled_cash = add_exact_decimal(settled_cash, row.settled_cash_delta)
            if settled_cash < 0 or row.settled_cash_after != settled_cash:
                _fail("dividend ledger settled-cash arithmetic is inconsistent")
            key = (PortfolioEventKind.DIVIDEND, row.application_id)
            if key in applied:
                _fail(
                    "dividend ledger attempts to apply an event identity more "
                    "than once"
                )
            applied[key] = AppliedEventFingerprint(
                event_kind=PortfolioEventKind.DIVIDEND,
                event_id=row.application_id,
                payload_sha256=row.source_payload_sha256,
            )

        if settled_cash != new_state.settled_cash:
            _fail("unexplained settled-cash mutation")
        if tuple(sorted(positions.values(), key=lambda item: item.asset_id)) != (
            new_state.open_positions
        ):
            _fail("unexplained position mutation")
        if tuple(
            sorted(
                pending.values(),
                key=lambda item: (item.settlement_session, item.settlement_id),
            )
        ) != new_state.pending_settlements:
            _fail("unexplained pending-settlement mutation")

        expected_applied = tuple(
            sorted(
                applied.values(),
                key=lambda item: (item.event_kind.value, item.event_id),
            )
        )
        if expected_applied != new_state.applied_events:
            _fail("unexplained applied-event mutation")
        if not previous_keys <= set(applied):
            _fail("applied event fingerprints cannot be removed")

    @staticmethod
    def validate_dividend_application(
        *,
        session: date,
        previous_state: PortfolioState,
        dividend_evidence: Sequence[CanonicalDividendAccountingEvidence],
        dividend_ledger_entries: Sequence[DividendLedgerEntry],
        dividend_outcomes: Sequence[DividendApplicationOutcome],
    ) -> None:
        """Prove OD-14.5 row provenance and OD-14.8/14.9 evidence discharge.

        The authoritative pre-X entitlement binding is recomputed directly
        from ``previous_state.open_positions`` -- the immutable state entering
        the processed ex-session, before this session's own SELL/BUY rows are
        applied -- never from a post-execution position map. Every
        application identity, payload hash, and gross-cash amount is
        independently re-derived rather than trusted from the rows/outcomes
        under test.
        """

        PortfolioInvariantChecker.validate_state(previous_state)

        events = tuple(dividend_evidence)
        for event in events:
            if not isinstance(event, CanonicalDividendAccountingEvidence):
                _fail_dividend(
                    "dividend_evidence must contain "
                    "CanonicalDividendAccountingEvidence values"
                )
            try:
                rebuilt_event = CanonicalDividendAccountingEvidence.model_validate(
                    event.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise DividendProvenanceError(
                    "dividend evidence fails structural validation"
                ) from error
            if event != rebuilt_event:
                _fail_dividend("dividend evidence is not canonical")
            if event.ex_session != session:
                _fail_dividend(
                    "dividend evidence ex_session must equal the current "
                    "session"
                )

        event_ids = tuple(
            event.canonical_distribution_event_id for event in events
        )
        if len(set(event_ids)) != len(event_ids):
            _fail_dividend("supplied dividend evidence event identities must be unique")

        rows = tuple(dividend_ledger_entries)
        for row in rows:
            if not isinstance(row, DividendLedgerEntry):
                _fail_dividend(
                    "dividend_ledger_entries must contain DividendLedgerEntry "
                    "values"
                )
            try:
                rebuilt_row = DividendLedgerEntry.model_validate(
                    row.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise DividendProvenanceError(
                    "dividend ledger entry fails structural validation"
                ) from error
            if row != rebuilt_row:
                _fail_dividend("dividend ledger entry is not canonical")
            if row.session != session:
                _fail_dividend(
                    "dividend ledger entry must belong to the current session"
                )

        outcomes = tuple(dividend_outcomes)
        for outcome in outcomes:
            if not isinstance(outcome, DividendApplicationOutcome):
                _fail_dividend(
                    "dividend_outcomes must contain DividendApplicationOutcome "
                    "values"
                )
            try:
                rebuilt_outcome = DividendApplicationOutcome.model_validate(
                    outcome.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise DividendProvenanceError(
                    "dividend outcome fails structural validation"
                ) from error
            if outcome != rebuilt_outcome:
                _fail_dividend("dividend outcome is not canonical")

        outcome_ids = tuple(
            outcome.canonical_distribution_event_id for outcome in outcomes
        )
        if len(set(outcome_ids)) != len(outcome_ids):
            _fail_dividend("recorded dividend outcome identities must be unique")
        if set(event_ids) != set(outcome_ids):
            _fail_dividend(
                "supplied dividend evidence and recorded outcomes must be in "
                "exact bidirectional correspondence"
            )

        positions_by_asset = {
            position.asset_id: position
            for position in previous_state.open_positions
        }
        applied_lookup = {
            (fingerprint.event_kind, fingerprint.event_id): fingerprint
            for fingerprint in previous_state.applied_events
        }
        rows_by_application_id = {row.application_id: row for row in rows}
        if len(rows_by_application_id) != len(rows):
            _fail_dividend(
                "dividend ledger rows must have unique application identities"
            )
        outcomes_by_event_id = {
            outcome.canonical_distribution_event_id: outcome
            for outcome in outcomes
        }

        expected_order: list[tuple[str, str]] = []
        applied_pairs: list[tuple[str, str]] = []
        applied_row_ids: set[str] = set()

        for event in events:
            asset_id = event.canonical_security_id
            expected_order.append((asset_id, event.canonical_distribution_event_id))

            position = positions_by_asset.get(asset_id)
            q_t = position.quantity if position is not None else 0
            trade_id = (
                position.entry_execution_id if position is not None else None
            )

            expected_application_id = compute_dividend_application_id(
                canonical_distribution_event_id=(
                    event.canonical_distribution_event_id
                ),
                entitlement_session=event.entitlement_session,
                ex_session=event.ex_session,
                asset_id=asset_id,
                attribution_trade_id=trade_id,
            )
            expected_gross_cash = compute_gross_dividend_cash(
                q_t, event.amount_per_share
            )
            expected_payload_hash = compute_dividend_application_payload_hash(
                application_id=expected_application_id,
                evidence=event,
                asset_id=asset_id,
                attribution_trade_id=trade_id,
                q_t=q_t,
                gross_cash_amount=expected_gross_cash,
            )

            outcome = outcomes_by_event_id.get(
                event.canonical_distribution_event_id
            )
            if outcome is None:
                _fail_dividend(
                    "supplied dividend evidence has no recorded outcome: "
                    f"{event.canonical_distribution_event_id}"
                )
            if (
                outcome.application_id != expected_application_id
                or outcome.payload_sha256 != expected_payload_hash
                or outcome.asset_id != asset_id
                or outcome.entitlement_session != event.entitlement_session
                or outcome.ex_session != event.ex_session
                or outcome.q_t != q_t
                or outcome.attribution_trade_id != trade_id
            ):
                _fail_dividend(
                    "recorded dividend outcome disagrees with the "
                    "authoritative pre-X entitlement binding: "
                    f"{event.canonical_distribution_event_id}"
                )

            applied_key = (PortfolioEventKind.DIVIDEND, expected_application_id)
            already_applied = applied_lookup.get(applied_key)

            if q_t == 0:
                if (
                    outcome.status
                    is not DividendApplicationStatus.NO_OP_NOT_ENTITLED
                ):
                    _fail_dividend(
                        "zero pre-X quantity must produce "
                        f"NO_OP_NOT_ENTITLED: {event.canonical_distribution_event_id}"
                    )
                if expected_application_id in rows_by_application_id:
                    _fail_dividend(
                        "NO_OP_NOT_ENTITLED must not have a dividend ledger "
                        f"row: {event.canonical_distribution_event_id}"
                    )
                continue

            if outcome.status is DividendApplicationStatus.NO_OP_NOT_ENTITLED:
                _fail_dividend(
                    "an entitled dividend event cannot record "
                    f"NO_OP_NOT_ENTITLED: {event.canonical_distribution_event_id}"
                )
            elif outcome.status is DividendApplicationStatus.APPLIED:
                if already_applied is not None:
                    _fail_dividend(
                        "APPLIED requires a previously unapplied dividend "
                        f"identity: {expected_application_id}"
                    )
                row = rows_by_application_id.get(expected_application_id)
                if row is None:
                    _fail_dividend(
                        "APPLIED requires exactly one DIVIDEND_APPLIED row: "
                        f"{expected_application_id}"
                    )
                if (
                    row.canonical_distribution_event_id
                    != event.canonical_distribution_event_id
                    or row.canonical_distribution_snapshot_fingerprint
                    != event.canonical_distribution_snapshot_fingerprint
                    or row.asset_id != asset_id
                    or row.entitlement_session != event.entitlement_session
                    or row.attribution_trade_id != trade_id
                    or row.q_t != q_t
                    or row.d_h != event.amount_per_share
                    or row.amount_basis != event.amount_basis
                    or row.normalization_method_id != event.normalization_method_id
                    or row.normalization_scale != event.normalization_scale
                    or row.normalization_rounding_mode
                    != event.normalization_rounding_mode
                    or row.normalization_arithmetic_mode
                    != event.normalization_arithmetic_mode
                    or row.normalization_inputs_fingerprint
                    != event.normalization_inputs_fingerprint
                    or row.calendar_resolution_fingerprint
                    != event.calendar_resolution_fingerprint
                    or row.currency != event.currency
                    or row.gross_cash_amount != expected_gross_cash
                    or row.settled_cash_delta != expected_gross_cash
                    or row.source_payload_sha256 != expected_payload_hash
                ):
                    _fail_dividend(
                        "DIVIDEND_APPLIED row disagrees with its source "
                        f"accounting evidence: {expected_application_id}"
                    )
                applied_pairs.append((asset_id, event.canonical_distribution_event_id))
                applied_row_ids.add(expected_application_id)
            elif outcome.status is DividendApplicationStatus.REPLAYED:
                if already_applied is None:
                    _fail_dividend(
                        "REPLAYED requires an already-applied dividend "
                        f"identity: {expected_application_id}"
                    )
                if already_applied.payload_sha256 != expected_payload_hash:
                    _fail_dividend(
                        "dividend application identity has a conflicting "
                        f"payload: {expected_application_id}"
                    )
                if expected_application_id in rows_by_application_id:
                    _fail_dividend(
                        "REPLAYED must not write a second dividend ledger "
                        f"row: {expected_application_id}"
                    )
            else:  # pragma: no cover - enum validation already closes this
                _fail_dividend("unsupported dividend application outcome status")

        if set(rows_by_application_id) != applied_row_ids:
            _fail_dividend(
                "every dividend ledger row must correspond to exactly one "
                "APPLIED outcome"
            )

        actual_outcome_order = [
            (outcome.asset_id, outcome.canonical_distribution_event_id)
            for outcome in outcomes
        ]
        if actual_outcome_order != sorted(expected_order):
            _fail_dividend(
                "dividend outcomes must be ordered by (asset_id, "
                "canonical_distribution_event_id)"
            )

        actual_row_order = [
            (row.asset_id, row.canonical_distribution_event_id) for row in rows
        ]
        if actual_row_order != sorted(applied_pairs):
            _fail_dividend(
                "dividend ledger rows must be ordered by (asset_id, "
                "canonical_distribution_event_id)"
            )

    @staticmethod
    def validate_persisted_accounting_history(
        initial_state: PortfolioState,
        final_state: PortfolioState,
        ledger_entries: Sequence[PortfolioLedgerEntry],
        session_snapshots: Sequence[PortfolioSessionSnapshot],
        dividend_ledger_entries: Sequence[DividendLedgerEntry] = (),
        dividend_evidence: Sequence[CanonicalDividendAccountingEvidence] = (),
        dividend_outcomes: Sequence[DividendApplicationOutcome] = (),
    ) -> None:
        """Prove persisted accounting state from initial state plus ledger.

        This is an artifact audit, not a transition replay. It applies only the
        economic facts already present in canonical ledger rows and verifies
        every processed-session snapshot and the complete final state.

        This is site 2 of the OD-14.7 three-site provenance/discharge
        architecture: for every persisted session, ``validate_dividend_application``
        (site 1's own logic, reused verbatim) is called against the
        reconstructed *session-start* state -- captured before this
        session's own SETTLEMENT/SELL/BUY/dividend rows are replayed -- so
        dividend row provenance (OD-14.5) and evidence/outcome discharge
        (OD-14.8/14.9) are re-proven from independently retained sources,
        never from facts copied off the row/outcome under test.
        """

        PortfolioInvariantChecker.validate_state(initial_state)
        PortfolioInvariantChecker.validate_state(final_state)

        try:
            ledger = tuple(ledger_entries)
            snapshots = tuple(session_snapshots)
        except TypeError as error:
            raise PortfolioStateError(
                "ledger_entries and session_snapshots must be iterable"
            ) from error

        for entry in ledger:
            PortfolioInvariantChecker._validate_ledger_entry(entry)
        validated_snapshots: list[PortfolioSessionSnapshot] = []
        for item in snapshots:
            if not isinstance(item, PortfolioSessionSnapshot):
                _fail(
                    "session_snapshots must contain PortfolioSessionSnapshot values"
                )
            try:
                validated = PortfolioSessionSnapshot.model_validate(
                    item.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise PortfolioStateError(
                    "session snapshot fails structural validation"
                ) from error
            if item != validated:
                _fail("session snapshot is not canonical")
            validated_snapshots.append(validated)
        snapshots = tuple(validated_snapshots)

        snapshot_sessions = tuple(item.session for item in snapshots)
        if any(
            current <= previous
            for previous, current in zip(
                snapshot_sessions, snapshot_sessions[1:], strict=False
            )
        ):
            _fail("persisted session snapshots must be strictly chronological")

        ledger_keys = tuple(
            (entry.session, entry.sequence_in_session) for entry in ledger
        )
        if ledger_keys != tuple(sorted(ledger_keys)):
            _fail("persisted ledger must be in canonical chronological order")

        ledger_by_session: dict[date, list[PortfolioLedgerEntry]] = {}
        for entry in ledger:
            ledger_by_session.setdefault(entry.session, []).append(entry)
        if not set(ledger_by_session) <= set(snapshot_sessions):
            _fail("every persisted ledger session requires a session snapshot")
        for session_ledger in ledger_by_session.values():
            if tuple(item.sequence_in_session for item in session_ledger) != tuple(
                range(len(session_ledger))
            ):
                _fail(
                    "persisted ledger sequence_in_session must be consecutive "
                    "from zero"
                )

        try:
            dividend_ledger = tuple(dividend_ledger_entries)
            evidence = tuple(dividend_evidence)
            outcomes = tuple(dividend_outcomes)
        except TypeError as error:
            raise PortfolioStateError(
                "dividend_ledger_entries, dividend_evidence, and "
                "dividend_outcomes must be iterable"
            ) from error

        for row in dividend_ledger:
            if not isinstance(row, DividendLedgerEntry):
                _fail(
                    "dividend_ledger_entries must contain DividendLedgerEntry "
                    "values"
                )
            try:
                rebuilt_row = DividendLedgerEntry.model_validate(
                    row.model_dump(mode="python")
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise PortfolioStateError(
                    "persisted dividend ledger entry fails structural "
                    "validation"
                ) from error
            if row != rebuilt_row:
                _fail("persisted dividend ledger entry is not canonical")

        dividend_ledger_keys = tuple(
            (row.session, row.sequence_in_session) for row in dividend_ledger
        )
        if dividend_ledger_keys != tuple(sorted(dividend_ledger_keys)):
            _fail(
                "persisted dividend ledger must be in canonical chronological "
                "order"
            )

        dividend_ledger_by_session: dict[date, list[DividendLedgerEntry]] = {}
        for row in dividend_ledger:
            dividend_ledger_by_session.setdefault(row.session, []).append(row)
        if not set(dividend_ledger_by_session) <= set(snapshot_sessions):
            _fail(
                "every persisted dividend ledger session requires a session "
                "snapshot"
            )
        for session_rows in dividend_ledger_by_session.values():
            if tuple(item.sequence_in_session for item in session_rows) != tuple(
                range(len(session_rows))
            ):
                _fail(
                    "persisted dividend ledger sequence_in_session must be "
                    "consecutive from zero"
                )
            order_keys = [
                (item.asset_id, item.canonical_distribution_event_id)
                for item in session_rows
            ]
            if order_keys != sorted(order_keys):
                _fail(
                    "persisted dividend ledger rows must be ordered by "
                    "(asset_id, canonical_distribution_event_id)"
                )

        evidence_by_session: dict[date, list[CanonicalDividendAccountingEvidence]] = {}
        for item in evidence:
            evidence_by_session.setdefault(item.ex_session, []).append(item)
        if not set(evidence_by_session) <= set(snapshot_sessions):
            _fail(
                "every persisted dividend evidence ex_session requires a "
                "session snapshot"
            )

        outcomes_by_session: dict[date, list[DividendApplicationOutcome]] = {}
        for item in outcomes:
            outcomes_by_session.setdefault(item.ex_session, []).append(item)
        if not set(outcomes_by_session) <= set(snapshot_sessions):
            _fail(
                "every persisted dividend outcome ex_session requires a "
                "session snapshot"
            )

        settled_cash = initial_state.settled_cash
        positions = {
            position.asset_id: position
            for position in initial_state.open_positions
        }
        pending = {
            settlement.settlement_id: settlement
            for settlement in initial_state.pending_settlements
        }
        applied = {
            (fingerprint.event_kind, fingerprint.event_id): fingerprint
            for fingerprint in initial_state.applied_events
        }

        reconstructed_state = initial_state
        for offset, current_snapshot in enumerate(snapshots, start=1):
            session = current_snapshot.session
            overdue = tuple(
                sorted(
                    (
                        settlement.settlement_id
                        for settlement in pending.values()
                        if settlement.settlement_session < session
                    )
                )
            )
            if overdue:
                _fail(
                    "pending settlement is overdue before persisted session "
                    f"{session.isoformat()}: {', '.join(overdue)}"
                )
            due = {
                settlement.settlement_id
                for settlement in pending.values()
                if settlement.settlement_session == session
            }

            session_start_state = reconstructed_state
            PortfolioInvariantChecker.validate_dividend_application(
                session=session,
                previous_state=session_start_state,
                dividend_evidence=tuple(evidence_by_session.get(session, ())),
                dividend_ledger_entries=tuple(
                    dividend_ledger_by_session.get(session, ())
                ),
                dividend_outcomes=tuple(outcomes_by_session.get(session, ())),
            )

            # Task 5C-C: P is the reconstructed immutable session-start
            # state, B this session's BUY rows -- the same shared rule the
            # live transition applies, so persisted replay cannot drift.
            session_ledger = tuple(ledger_by_session.get(session, []))
            classification = _classify_session_ledger(
                session_start_state=session_start_state, ledger=session_ledger
            )
            ordering_keys: list[tuple[object, ...]] = []
            for entry in session_ledger:
                # Same exact-accumulation requirement as the live replay
                # path above (Residual B hardening: BUY settled_cash_delta
                # is now exact and can carry more significant digits).
                settled_cash = add_exact_decimal(
                    settled_cash, entry.settled_cash_delta
                )
                if settled_cash < 0 or entry.settled_cash_after != settled_cash:
                    _fail("persisted ledger settled-cash arithmetic is inconsistent")

                ordering_keys.append(
                    _ledger_ordering_key(entry, classification)
                )
                try:
                    if (
                        entry.event_type
                        is PortfolioLedgerEventType.SETTLEMENT_APPLIED
                    ):
                        PortfolioInvariantChecker._apply_settlement_ledger(
                            entry=entry,
                            positions=positions,
                            pending=pending,
                            applied=applied,
                        )
                    elif entry.event_type is PortfolioLedgerEventType.SELL_APPLIED:
                        PortfolioInvariantChecker._apply_sell_ledger(
                            entry=entry,
                            positions=positions,
                            pending=pending,
                            applied=applied,
                        )
                    elif entry.event_type is PortfolioLedgerEventType.BUY_APPLIED:
                        PortfolioInvariantChecker._apply_buy_ledger(
                            entry=entry,
                            positions=positions,
                            pending=pending,
                            applied=applied,
                        )
                    else:  # pragma: no cover - enum validation closes this
                        _fail("unsupported persisted ledger event type")
                except ValidationError as error:
                    raise PortfolioStateError(
                        "persisted ledger contains invalid accounting facts"
                    ) from error

            if ordering_keys != sorted(ordering_keys):
                _fail("persisted ledger entries are not in canonical session order")
            missing_due = tuple(sorted(due & set(pending)))
            if missing_due:
                _fail(
                    "due settlement lacks SETTLEMENT_APPLIED on persisted session "
                    f"{session.isoformat()}: {', '.join(missing_due)}"
                )

            for row in dividend_ledger_by_session.get(session, []):
                settled_cash = add_exact_decimal(settled_cash, row.settled_cash_delta)
                if settled_cash < 0 or row.settled_cash_after != settled_cash:
                    _fail(
                        "persisted dividend ledger settled-cash arithmetic is "
                        "inconsistent"
                    )
                key = (PortfolioEventKind.DIVIDEND, row.application_id)
                if key in applied:
                    _fail(
                        "persisted dividend ledger attempts to apply an event "
                        "identity more than once"
                    )
                applied[key] = AppliedEventFingerprint(
                    event_kind=PortfolioEventKind.DIVIDEND,
                    event_id=row.application_id,
                    payload_sha256=row.source_payload_sha256,
                )

            try:
                reconstructed_state = PortfolioState(
                    base_currency=initial_state.base_currency,
                    as_of_session=session,
                    state_version=initial_state.state_version + offset,
                    settled_cash=settled_cash,
                    open_positions=tuple(positions.values()),
                    pending_settlements=tuple(pending.values()),
                    applied_events=tuple(applied.values()),
                )
            except ValidationError as error:
                raise PortfolioStateError(
                    "persisted ledger reconstructs an invalid portfolio state"
                ) from error
            PortfolioInvariantChecker.validate_state(reconstructed_state)
            expected_snapshot = PortfolioSessionSnapshot(
                session=session,
                state_version=reconstructed_state.state_version,
                settled_cash=reconstructed_state.settled_cash,
                open_position_count=len(reconstructed_state.open_positions),
                pending_settlement_count=len(
                    reconstructed_state.pending_settlements
                ),
                state_hash=hash_portfolio_state(reconstructed_state),
            )
            if current_snapshot != expected_snapshot:
                _fail(
                    "persisted session snapshot is not explained by initial "
                    "state plus ledger"
                )

        if reconstructed_state != final_state:
            _fail("final portfolio state is not explained by initial state plus ledger")
        if final_state.as_of_session is not None:
            invalid_final_pending = tuple(
                settlement.settlement_id
                for settlement in final_state.pending_settlements
                if settlement.settlement_session <= final_state.as_of_session
            )
            if invalid_final_pending:
                _fail(
                    "final state retains a due or overdue pending settlement: "
                    + ", ".join(invalid_final_pending)
                )

    @staticmethod
    def _validate_execution_event(event: object) -> PortfolioExecutionEvent:
        if not isinstance(event, PortfolioExecutionEvent):
            _fail(
                "execution_events must contain PortfolioExecutionEvent values"
            )
        try:
            validated = PortfolioExecutionEvent.model_validate(
                event.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioStateError(
                "execution event fails structural validation"
            ) from error
        if event != validated:
            _fail("execution event is not canonical")
        return validated

    @staticmethod
    def _validate_ledger_entry(entry: object) -> PortfolioLedgerEntry:
        if not isinstance(entry, PortfolioLedgerEntry):
            _fail("ledger_entries must contain PortfolioLedgerEntry values")
        try:
            validated = PortfolioLedgerEntry.model_validate(
                entry.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PortfolioStateError(
                "ledger entry fails structural validation"
            ) from error
        if entry != validated:
            _fail("ledger entry is not canonical")
        return validated

    @staticmethod
    def _apply_buy_ledger(
        *,
        entry: PortfolioLedgerEntry,
        positions: dict[str, OpenPosition],
        pending: dict[str, PendingSettlement],
        applied: dict[tuple[PortfolioEventKind, str], AppliedEventFingerprint],
    ) -> None:
        del pending  # BUY ledger rows do not affect pending settlements.
        if (
            entry.asset_id is None
            or entry.source_order_id is None
            or entry.quantity_delta is None
            or entry.quantity_delta <= 0
            or entry.fill_price is None
            or entry.execution_cost is None
            or entry.settlement_id is not None
            or entry.settlement_session is not None
            or entry.pending_cash_delta != 0
        ):
            _fail("invalid BUY ledger fields")
        # Exact-arithmetic hardening (Residual B): entry.fill_price
        # carries no enforced significant-digit bound, so this
        # multiplication/addition -- which becomes the authoritative
        # cost_basis via ledger replay -- uses the shared exact helpers,
        # not ambient-context Decimal `*`/`+`/unary `-`.
        required_cash = add_exact_decimal(
            exact_decimal_times_int(entry.fill_price, entry.quantity_delta),
            entry.execution_cost,
        )
        if entry.settled_cash_delta != subtract_exact_decimal(
            Decimal("0"), required_cash
        ):
            _fail("invalid BUY ledger cash arithmetic")
        if entry.asset_id in positions:
            _fail("BUY ledger cannot add an already-open asset")
        if entry.position_quantity_after != entry.quantity_delta:
            _fail("invalid BUY ledger position quantity")

        event = PortfolioExecutionEvent(
            execution_id=entry.source_event_id,
            source_order_id=entry.source_order_id,
            session=entry.session,
            asset_id=entry.asset_id,
            side=ExecutionSide.BUY,
            quantity=entry.quantity_delta,
            fill_price=entry.fill_price,
            execution_cost=entry.execution_cost,
        )
        if hash_execution_event(event) != entry.source_payload_sha256:
            _fail("BUY ledger source hash does not match its execution payload")
        positions[entry.asset_id] = OpenPosition(
            asset_id=entry.asset_id,
            quantity=entry.quantity_delta,
            entry_session=entry.session,
            entry_price=entry.fill_price,
            entry_execution_id=entry.source_event_id,
            entry_execution_cost=entry.execution_cost,
            cost_basis=required_cash,
        )
        PortfolioInvariantChecker._add_fingerprint(
            applied,
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id=entry.source_event_id,
                payload_sha256=entry.source_payload_sha256,
            ),
        )

    @staticmethod
    def _apply_sell_ledger(
        *,
        entry: PortfolioLedgerEntry,
        positions: dict[str, OpenPosition],
        pending: dict[str, PendingSettlement],
        applied: dict[tuple[PortfolioEventKind, str], AppliedEventFingerprint],
    ) -> None:
        if (
            entry.asset_id is None
            or entry.source_order_id is None
            or entry.quantity_delta is None
            or entry.quantity_delta >= 0
            or entry.fill_price is None
            or entry.execution_cost is None
            or entry.settlement_id is None
            or entry.settlement_session is None
            or entry.settled_cash_delta != 0
        ):
            _fail("invalid SELL ledger fields")
        quantity = -entry.quantity_delta
        # Exact-arithmetic hardening (Residual A): mirrors the live SELL
        # computation in PortfolioTransitionEngine.transition so this replay
        # can never diverge from the authoritative value at any ambient
        # Decimal precision.
        net_proceeds = subtract_exact_decimal(
            exact_decimal_times_int(entry.fill_price, quantity),
            entry.execution_cost,
        )
        if net_proceeds <= 0 or entry.pending_cash_delta != net_proceeds:
            _fail("invalid SELL ledger pending-cash arithmetic")
        current = positions.get(entry.asset_id)
        if current is None or current.quantity != quantity:
            _fail("SELL ledger must fully remove one open position")
        if entry.position_quantity_after != 0:
            _fail("SELL ledger position_quantity_after must be zero")
        if entry.settlement_id in pending:
            _fail("SELL ledger settlement identity must be unique")

        event = PortfolioExecutionEvent(
            execution_id=entry.source_event_id,
            source_order_id=entry.source_order_id,
            session=entry.session,
            asset_id=entry.asset_id,
            side=ExecutionSide.SELL,
            quantity=quantity,
            fill_price=entry.fill_price,
            execution_cost=entry.execution_cost,
            settlement_id=entry.settlement_id,
            settlement_session=entry.settlement_session,
        )
        if hash_execution_event(event) != entry.source_payload_sha256:
            _fail("SELL ledger source hash does not match its execution payload")
        del positions[entry.asset_id]
        pending[entry.settlement_id] = PendingSettlement(
            settlement_id=entry.settlement_id,
            source_execution_id=entry.source_event_id,
            asset_id=entry.asset_id,
            amount=net_proceeds,
            trade_session=entry.session,
            settlement_session=entry.settlement_session,
        )
        PortfolioInvariantChecker._add_fingerprint(
            applied,
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id=entry.source_event_id,
                payload_sha256=entry.source_payload_sha256,
            ),
        )

    @staticmethod
    def _apply_settlement_ledger(
        *,
        entry: PortfolioLedgerEntry,
        positions: dict[str, OpenPosition],
        pending: dict[str, PendingSettlement],
        applied: dict[tuple[PortfolioEventKind, str], AppliedEventFingerprint],
    ) -> None:
        del positions  # Settlement ledger rows do not affect positions.
        if (
            entry.source_order_id is not None
            or entry.quantity_delta != 0
            or entry.fill_price is not None
            or entry.execution_cost is not None
            or entry.settlement_id != entry.source_event_id
            or entry.settlement_session != entry.session
            or entry.settled_cash_delta <= 0
            or entry.pending_cash_delta != -entry.settled_cash_delta
            or entry.position_quantity_after is not None
        ):
            _fail("invalid SETTLEMENT ledger fields")
        settlement = pending.get(entry.source_event_id)
        if settlement is None:
            _fail("SETTLEMENT ledger must remove an existing pending settlement")
        if (
            settlement.asset_id != entry.asset_id
            or settlement.settlement_session != entry.session
            or settlement.amount != entry.settled_cash_delta
        ):
            _fail("SETTLEMENT ledger conflicts with its pending settlement")
        if hash_settlement_event(settlement) != entry.source_payload_sha256:
            _fail("SETTLEMENT ledger source hash does not match its payload")
        del pending[entry.source_event_id]
        PortfolioInvariantChecker._add_fingerprint(
            applied,
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.SETTLEMENT,
                event_id=entry.source_event_id,
                payload_sha256=entry.source_payload_sha256,
            ),
        )

    @staticmethod
    def _add_fingerprint(
        applied: dict[tuple[PortfolioEventKind, str], AppliedEventFingerprint],
        fingerprint: AppliedEventFingerprint,
    ) -> None:
        key = (fingerprint.event_kind, fingerprint.event_id)
        if key in applied:
            _fail("ledger attempts to apply an event identity more than once")
        applied[key] = fingerprint


__all__ = [
    "BUY_APPLICATION_RANK",
    "PRIOR_SELL_APPLICATION_RANK",
    "ROUND_TRIP_SELL_APPLICATION_RANK",
    "SETTLEMENT_APPLICATION_RANK",
    "PortfolioInvariantChecker",
    "SessionApplicationClassification",
    "classify_session_applications",
]
