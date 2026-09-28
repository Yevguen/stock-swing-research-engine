"""Acceptance matrix for Phase 9 Entry Execution v0.1."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import date, datetime, time
from math import inf, nan

import pytest
from pydantic import ValidationError

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.execution.entry import (
    ENTRY_SLIPPAGE_BPS,
    EntryExecutionService,
    EntryExecutionStatus,
    EntryExecutionValidationError,
    PendingEntry,
    SizedPendingEntry,
    create_pending_entry,
    create_sized_pending_entry,
)
from stock_swing_d1.models import StockBar
from stock_swing_d1.strategy.baseline import BaselineSignalAction
from tests.execution.entry.conftest import (
    ENTRY_OPEN_TIME,
    ENTRY_SESSION,
    FOLLOWING_SESSION,
    NEW_YORK,
    SIGNAL_SESSION,
    SIGNAL_TIME,
    EntryCalendar,
)


def _pending(service, make_signal) -> PendingEntry:
    return service.create_pending_entry(signal=make_signal())


def _sized(
    pending_entry: PendingEntry,
    *,
    fixed_shares: int = 12,
    cash_available: float = 1_000_000_000.0,
) -> SizedPendingEntry:
    return create_sized_pending_entry(
        pending_entry=pending_entry,
        fixed_shares=fixed_shares,
        cash_available=cash_available,
    )


def _execute(
    service: EntryExecutionService,
    *,
    pending_entry: PendingEntry,
    execution_bar: StockBar | None,
    fixed_shares: int = 12,
    cash_available: float = 1_000_000_000.0,
):
    return EntryExecutionService.execute_pending_entry(
        service,
        pending_entry=pending_entry,
        sized_pending_entry=_sized(
            pending_entry,
            fixed_shares=fixed_shares,
            cash_available=cash_available,
        ),
        execution_bar=execution_bar,
    )


def test_valid_phase8_signal_creates_immutable_pending_entry(
    service_factory, make_signal
) -> None:
    service, _, _ = service_factory()
    signal = make_signal()

    pending = service.create_pending_entry(signal=signal)

    assert pending.security_id == signal.security_id
    assert pending.symbol == signal.symbol
    assert pending.signal_session == signal.signal_session
    assert pending.signal_time == signal.signal_time
    assert pending.planned_entry_session == signal.planned_entry_session
    assert pending.status is EntryExecutionStatus.PENDING_ENTRY
    with pytest.raises(FrozenInstanceError):
        pending.symbol = "CHANGED"
    with pytest.raises(TypeError):
        PendingEntry(
            security_id=signal.security_id,
            symbol=signal.symbol,
            signal_session=signal.signal_session,
            signal_time=signal.signal_time,
            planned_entry_session=signal.planned_entry_session,
        )


def test_module_level_pending_factory_uses_calendar(entry_calendar, make_signal) -> None:
    pending = create_pending_entry(
        make_signal(), trading_calendar=entry_calendar
    )

    assert pending.planned_entry_session == ENTRY_SESSION
    assert entry_calendar.next_session_calls == [SIGNAL_SESSION]


def test_no_signal_cannot_create_pending_entry(
    service_factory, make_signal
) -> None:
    service, _, _ = service_factory()
    signal = make_signal(action=BaselineSignalAction.NO_SIGNAL)

    with pytest.raises(EntryExecutionValidationError) as raised:
        service.create_pending_entry(signal=signal)

    assert raised.value.code == "INVALID_PENDING_ENTRY"


@pytest.mark.parametrize(
    ("sessions", "planned_session"),
    [
        ((date(2026, 8, 14), date(2026, 8, 17)), date(2026, 8, 17)),
        ((date(2026, 9, 4), date(2026, 9, 8)), date(2026, 9, 8)),
    ],
    ids=("weekend", "weekend-plus-monday-holiday"),
)
def test_t_plus_one_is_calendar_next_session(
    service_factory, make_signal, sessions, planned_session
) -> None:
    calendar = EntryCalendar(sessions)
    service, _, _ = service_factory(calendar=calendar)
    signal = make_signal(
        signal_session=sessions[0],
        signal_time=datetime.combine(
            sessions[0], time(16), tzinfo=NEW_YORK
        ),
        planned_entry_session=planned_session,
    )

    pending = service.create_pending_entry(signal=signal)

    assert pending.planned_entry_session == planned_session
    assert pending.planned_entry_session != pending.signal_session


def test_non_next_planned_session_fails_closed(
    service_factory, make_signal
) -> None:
    service, _, _ = service_factory()

    with pytest.raises(EntryExecutionValidationError) as raised:
        service.create_pending_entry(
            signal=make_signal(planned_entry_session=FOLLOWING_SESSION)
        )

    assert raised.value.code == "INVALID_EXECUTION_SESSION"


def test_exact_opening_timestamp_and_original_signal_time_revalidate_earnings(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, overlay, calendar = service_factory()
    signal = make_signal()
    pending = service.create_pending_entry(signal=signal)

    decision = _execute(
        service,
        pending_entry=pending, execution_bar=make_execution_bar()
    )

    assert calendar.open_time_calls == [ENTRY_SESSION]
    assert overlay.calls == [
        {
            "canonical_asset_id": signal.security_id,
            "signal_time": signal.signal_time,
            "execution_time": ENTRY_OPEN_TIME,
            "planned_entry_session": ENTRY_SESSION,
        }
    ]
    assert decision.execution_time == ENTRY_OPEN_TIME


def test_future_pit_knowledge_after_open_cannot_block_opening_fill(
    entry_calendar, execution_cost_service, make_signal, make_execution_bar
) -> None:
    class TimeSensitiveOverlay:
        def __init__(self) -> None:
            self.as_of_values: list[datetime] = []

        def revalidate_pending_entry(self, **kwargs):
            as_of = kwargs["execution_time"]
            self.as_of_values.append(as_of)
            action = (
                EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
                if as_of == ENTRY_OPEN_TIME
                else EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
            )
            return EarningsIntegrationDecision(action, None, None)

    overlay = TimeSensitiveOverlay()
    service = EntryExecutionService(
        earnings_overlay=overlay,
        trading_calendar=entry_calendar,
        execution_cost_service=execution_cost_service,
    )

    decision = _execute(
        service,
        pending_entry=_pending(service, make_signal),
        execution_bar=make_execution_bar(),
    )

    assert overlay.as_of_values == [ENTRY_OPEN_TIME]
    assert decision.status is EntryExecutionStatus.EXECUTED


@pytest.mark.parametrize(
    "calendar",
    [
        type(
            "NaiveOpenCalendar",
            (),
            {
                "next_session": lambda self, session: ENTRY_SESSION,
                "regular_session_open_time": lambda self, session: datetime(
                    2026, 8, 17, 9, 30
                ),
            },
        )(),
        type(
            "WrongDateOpenCalendar",
            (),
            {
                "next_session": lambda self, session: ENTRY_SESSION,
                "regular_session_open_time": lambda self, session: datetime(
                    2026, 8, 18, 9, 30, tzinfo=NEW_YORK
                ),
            },
        )(),
    ],
    ids=("naive", "not-bound-to-entry-session"),
)
def test_invalid_canonical_open_time_fails_before_earnings(
    service_factory, make_signal, make_execution_bar, calendar
) -> None:
    service, overlay, _ = service_factory(calendar=calendar)

    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(
            service,
            pending_entry=_pending(service, make_signal),
            execution_bar=make_execution_bar(),
        )

    assert raised.value.code == "INVALID_EXECUTION_TIME"
    assert overlay.calls == []


def test_unresolvable_open_time_fails_closed(
    service_factory, make_signal, make_execution_bar
) -> None:
    class BrokenOpenCalendar:
        def next_session(self, session):
            return ENTRY_SESSION

        def regular_session_open_time(self, session):
            raise ValueError("opening instant unavailable")

    service, overlay, _ = service_factory(calendar=BrokenOpenCalendar())

    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(
            service,
            pending_entry=_pending(service, make_signal),
            execution_bar=make_execution_bar(),
        )

    assert raised.value.code == "INVALID_EXECUTION_TIME"
    assert overlay.calls == []


def test_earnings_invalidation_has_precedence_and_does_not_inspect_bar(
    service_factory, make_signal
) -> None:
    class ExplosiveBar:
        def __getattribute__(self, name):
            raise AssertionError(f"execution bar was inspected: {name}")

    service, _, _ = service_factory(
        EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED,
        reason="NEW_BLOCKING_KNOWLEDGE",
    )

    decision = _execute(
        service,
        pending_entry=_pending(service, make_signal),
        execution_bar=ExplosiveBar(),
    )

    assert decision.status is EntryExecutionStatus.INVALIDATED_BY_EARNINGS
    assert decision.earnings_reason == "NEW_BLOCKING_KNOWLEDGE"
    assert decision.reference_open is None
    assert decision.slippage_amount is None
    assert decision.execution_price is None


@pytest.mark.parametrize(
    "action",
    [
        EarningsIntegrationAction.ENTRY_ALLOWED,
        EarningsIntegrationAction.ENTRY_BLOCKED,
        EarningsIntegrationAction.HOLD_POSITION,
    ],
)
def test_unexpected_earnings_action_fails_closed(
    service_factory, make_signal, make_execution_bar, action
) -> None:
    service, _, _ = service_factory(action)

    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(
            service,
            pending_entry=_pending(service, make_signal),
            execution_bar=make_execution_bar(),
        )

    assert raised.value.code == "INVALID_EARNINGS_REVALIDATION_ACTION"


def test_non_decision_earnings_result_fails_closed(
    entry_calendar, execution_cost_service, make_signal, make_execution_bar
) -> None:
    class InvalidOverlay:
        def revalidate_pending_entry(self, **kwargs):
            return EarningsIntegrationAction.PENDING_ENTRY_ALLOWED

    service = EntryExecutionService(
        earnings_overlay=InvalidOverlay(),
        trading_calendar=entry_calendar,
        execution_cost_service=execution_cost_service,
    )

    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(
            service,
            pending_entry=_pending(service, make_signal),
            execution_bar=make_execution_bar(),
        )

    assert raised.value.code == "INVALID_EARNINGS_REVALIDATION_ACTION"


@pytest.mark.parametrize(
    ("opening_price", "slippage_amount", "execution_price"),
    [(100.0, 0.05, 100.05), (250.0, 0.125, 250.125)],
)
def test_exact_five_basis_point_execution(
    service_factory,
    make_signal,
    make_execution_bar,
    opening_price,
    slippage_amount,
    execution_price,
) -> None:
    service, _, _ = service_factory()

    decision = _execute(
        service,
        pending_entry=_pending(service, make_signal),
        execution_bar=make_execution_bar(opening_price=opening_price),
    )

    assert ENTRY_SLIPPAGE_BPS == 5.0
    assert decision.status is EntryExecutionStatus.EXECUTED
    assert decision.reference_open == opening_price
    assert decision.slippage_bps == 5.0
    assert decision.slippage_amount == slippage_amount
    assert decision.execution_price == execution_price


def test_core_execution_value_is_not_rounded(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory()
    opening_price = 123.4567

    decision = _execute(
        service,
        pending_entry=_pending(service, make_signal),
        execution_bar=make_execution_bar(opening_price=opening_price),
    )

    assert decision.slippage_amount == opening_price * 0.0005
    assert decision.execution_price == opening_price + opening_price * 0.0005
    assert decision.execution_price != round(decision.execution_price, 2)


@pytest.mark.parametrize("opening_price", [0.01, 1.0, 92.0, 10_000.0])
def test_every_executed_long_entry_has_adverse_slippage(
    service_factory, make_signal, make_execution_bar, opening_price
) -> None:
    service, _, _ = service_factory()

    decision = _execute(
        service,
        pending_entry=_pending(service, make_signal),
        execution_bar=make_execution_bar(opening_price=opening_price),
    )

    assert decision.execution_price is not None
    assert decision.reference_open is not None
    assert decision.execution_price > decision.reference_open


def test_missing_t_plus_one_bar_expires_without_fallback(
    service_factory, make_signal
) -> None:
    service, _, calendar = service_factory()

    decision = _execute(
        service,
        pending_entry=_pending(service, make_signal), execution_bar=None
    )

    assert decision.status is EntryExecutionStatus.NO_EXECUTABLE_BAR
    assert decision.reference_open is None
    assert decision.execution_price is None
    assert calendar.open_time_calls == [ENTRY_SESSION]
    assert FOLLOWING_SESSION not in calendar.open_time_calls


@pytest.mark.parametrize("invalid_open", [0.0, -1.0, nan, inf, -inf])
def test_invalid_t_plus_one_open_expires_without_substitution(
    service_factory, make_signal, make_execution_bar, invalid_open
) -> None:
    service, _, calendar = service_factory()
    valid_bar = make_execution_bar()
    invalid_bar = valid_bar.model_copy(update={"open": invalid_open})

    decision = _execute(
        service,
        pending_entry=_pending(service, make_signal),
        execution_bar=invalid_bar,
    )

    assert decision.status is EntryExecutionStatus.INVALID_OPEN_PRICE
    assert decision.reference_open is None
    assert decision.slippage_amount is None
    assert decision.execution_price is None
    assert calendar.open_time_calls == [ENTRY_SESSION]


@pytest.mark.parametrize(
    "updates",
    [
        {"security_id": "NORGATE:9999"},
        {"trading_date": FOLLOWING_SESSION},
        {"timeframe": "H1"},
        {"session_type": "extended"},
        {"price_basis": "capital_special_adjusted"},
    ],
    ids=(
        "security-id",
        "trading-date",
        "timeframe",
        "session-type",
        "price-basis",
    ),
)
def test_execution_bar_identity_mismatch_fails_closed(
    service_factory, make_signal, make_execution_bar, updates
) -> None:
    service, _, _ = service_factory()
    mismatched = make_execution_bar().model_copy(update=updates)

    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(
            service,
            pending_entry=_pending(service, make_signal),
            execution_bar=mismatched,
        )

    assert raised.value.code == "EXECUTION_BAR_IDENTITY_MISMATCH"


@pytest.mark.parametrize(
    ("opening_price", "expected_price"),
    [(108.0, 108.054), (92.0, 92.046)],
    ids=("gap-up", "gap-down"),
)
def test_gap_is_preserved_by_using_t_plus_one_open(
    service_factory,
    make_signal,
    make_execution_bar,
    opening_price,
    expected_price,
) -> None:
    service, _, _ = service_factory()
    signal = make_signal(adjusted_close=100.0)

    decision = _execute(
        service,
        pending_entry=service.create_pending_entry(signal=signal),
        execution_bar=make_execution_bar(opening_price=opening_price),
    )

    assert signal.adjusted_close == 100.0
    assert decision.reference_open == opening_price
    assert decision.execution_price == expected_price


def test_identical_open_with_radically_different_hlc_has_identical_fill(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory()
    pending = _pending(service, make_signal)
    bar_a = make_execution_bar(
        opening_price=100.0, high=101.0, low=99.0, close=100.0
    )
    bar_b = make_execution_bar(
        opening_price=100.0, high=200.0, low=50.0, close=175.0
    )

    decision_a = _execute(
        service,
        pending_entry=pending, execution_bar=bar_a
    )
    decision_b = _execute(
        service,
        pending_entry=pending, execution_bar=bar_b
    )

    assert decision_a.reference_open == decision_b.reference_open == 100.0
    assert decision_a.slippage_amount == decision_b.slippage_amount == 0.05
    assert decision_a.execution_price == decision_b.execution_price == 100.05


def test_unadjusted_open_is_used_when_adjusted_open_is_also_available(
    service_factory, make_signal
) -> None:
    class BarWithAdjustedOpen(StockBar):
        adjusted_open: float

    bar = BarWithAdjustedOpen(
        security_id="NORGATE:1001",
        symbol="PH9",
        trading_date=ENTRY_SESSION,
        timeframe="D1",
        session_type="regular",
        currency="USD",
        price_basis="unadjusted",
        open=100.0,
        adjusted_open=75.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1_000_000,
    )
    service, _, _ = service_factory()

    decision = _execute(
        service,
        pending_entry=_pending(service, make_signal), execution_bar=bar
    )

    assert bar.open != bar.adjusted_open
    assert decision.reference_open == bar.open
    assert decision.execution_price == 100.05


def test_identical_inputs_are_deterministic(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory()
    pending = _pending(service, make_signal)
    bar = make_execution_bar(opening_price=250.0)

    first = _execute(
        service,
        pending_entry=pending, execution_bar=bar
    )
    second = _execute(
        service,
        pending_entry=pending, execution_bar=bar
    )

    assert first == second


def test_decision_and_inputs_are_immutable_and_not_mutated(
    execution_cost_service, make_signal, make_execution_bar
) -> None:
    @dataclass(frozen=True, slots=True)
    class ImmutableOverlay:
        decision: EarningsIntegrationDecision

        def revalidate_pending_entry(self, **kwargs):
            return self.decision

    @dataclass(frozen=True, slots=True)
    class ImmutableCalendar:
        sessions: tuple[date, ...]

        def next_session(self, session):
            return self.sessions[self.sessions.index(session) + 1]

        def regular_session_open_time(self, session):
            return datetime.combine(session, time(9, 30), tzinfo=NEW_YORK)

    overlay = ImmutableOverlay(
        EarningsIntegrationDecision(
            EarningsIntegrationAction.PENDING_ENTRY_ALLOWED, None, None
        )
    )
    calendar = ImmutableCalendar((SIGNAL_SESSION, ENTRY_SESSION))
    service = EntryExecutionService(
        earnings_overlay=overlay,
        trading_calendar=calendar,
        execution_cost_service=execution_cost_service,
    )
    signal = make_signal()
    bar = make_execution_bar()
    signal_before = replace(signal)
    bar_before = bar.model_dump(mode="python")

    decision = _execute(
        service,
        pending_entry=service.create_pending_entry(signal=signal),
        execution_bar=bar,
    )

    assert signal == signal_before
    assert bar.model_dump(mode="python") == bar_before
    assert overlay == ImmutableOverlay(overlay.decision)
    assert calendar == ImmutableCalendar(calendar.sessions)
    with pytest.raises(FrozenInstanceError):
        decision.status = EntryExecutionStatus.NO_EXECUTABLE_BAR
    with pytest.raises(ValidationError):
        bar.open = 1.0


def test_wrong_bar_type_is_rejected_only_after_earnings_allows(
    service_factory, make_signal
) -> None:
    service, overlay, _ = service_factory()

    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(
            service,
            pending_entry=_pending(service, make_signal),
            execution_bar=object(),
        )

    assert raised.value.code == "INVALID_EXECUTION_BAR"
    assert len(overlay.calls) == 1


def test_forged_pending_entry_session_is_revalidated_before_execution(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, overlay, _ = service_factory()
    pending = _pending(service, make_signal)
    object.__setattr__(pending, "planned_entry_session", FOLLOWING_SESSION)

    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(
            service,
            pending_entry=pending, execution_bar=make_execution_bar()
        )

    assert raised.value.code == "INVALID_EXECUTION_SESSION"
    assert overlay.calls == []


def test_nonfinite_calculated_fill_fails_closed(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory()
    bar = make_execution_bar().model_copy(update={"open": 1.7976931348623157e308})

    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(
            service,
            pending_entry=_pending(service, make_signal), execution_bar=bar
        )

    assert raised.value.code == "INVALID_EXECUTION_BAR"
