"""Suite A: signal-time entry and mandatory T+1 revalidation."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from stock_swing_d1.earnings.integration import EarningsIntegrationAction
from stock_swing_d1.earnings.models import (
    EarningsScheduleStatus,
    EarningsValidationError,
    LifecycleState,
)
from tests.earnings.integration.conftest import ASSET, PROVIDER


def _unknown(make_state, *, as_of=None):
    overrides = {
        "event_instance_id": None,
        "earnings_schedule_known": False,
        "lifecycle_state": LifecycleState.UNKNOWN,
        "scheduled_date": None,
        "knowledge_effective_at": None,
        "event_revision_id": None,
        "schedule_status": EarningsScheduleStatus.UNKNOWN,
    }
    if as_of is not None:
        overrides["as_of"] = as_of
    return make_state(**overrides)


def test_EINT_A01_same_session_execution_is_never_authorized(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, query = overlay_factory(make_state())

    with pytest.raises(EarningsValidationError) as error:
        overlay.evaluate_entry_candidate(
            canonical_asset_id=ASSET,
            signal_time=ny_dt(2026, 10, 5, 16),
            signal_session=synthetic_calendar.session_date(0),
            planned_entry_session=synthetic_calendar.session_date(0),
        )

    assert error.value.code == "INVALID_ENTRY_SESSION"
    assert query.calls == []


def test_EINT_A02_signal_query_uses_signal_time_and_planned_session(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    signal_time = ny_dt(2026, 10, 5, 16)
    planned = synthetic_calendar.session_date(1)
    state = make_state(scheduled_date=synthetic_calendar.session_date(11))
    overlay, query = overlay_factory(state)

    overlay.evaluate_entry_candidate(
        canonical_asset_id=ASSET,
        signal_time=signal_time,
        signal_session=synthetic_calendar.session_date(0),
        planned_entry_session=planned,
    )

    assert query.calls == [
        {
            "canonical_asset_id": ASSET,
            "provider_name": PROVIDER,
            "as_of": signal_time,
            "decision_session": planned,
        }
    ]


@pytest.mark.hard_gate_6c8b
def test_EINT_A03_unknown_signal_time_state_blocks_entry(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, _ = overlay_factory(_unknown(make_state))

    decision = overlay.evaluate_entry_candidate(
        canonical_asset_id=ASSET,
        signal_time=ny_dt(2026, 10, 5, 16),
        signal_session=synthetic_calendar.session_date(0),
        planned_entry_session=synthetic_calendar.session_date(1),
    )

    assert decision.action is EarningsIntegrationAction.ENTRY_BLOCKED
    assert decision.risk_decision is not None
    assert decision.risk_decision.entry_blackout is True
    assert decision.reason == "EARNINGS_SCHEDULE_UNKNOWN"


@pytest.mark.hard_gate_6c8b
def test_EINT_A04_s0_through_s9_block_entry_boundaries(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    planned = synthetic_calendar.session_date(1)
    for event_index in (1, 2, 5, 10):
        overlay, _ = overlay_factory(
            make_state(scheduled_date=synthetic_calendar.session_date(event_index))
        )

        decision = overlay.evaluate_entry_candidate(
            canonical_asset_id=ASSET,
            signal_time=ny_dt(2026, 10, 5, 16),
            signal_session=synthetic_calendar.session_date(0),
            planned_entry_session=planned,
        )

        assert decision.action is EarningsIntegrationAction.ENTRY_BLOCKED
        assert decision.risk_decision is not None
        assert decision.risk_decision.entry_blackout is True


def test_EINT_A05_s10_does_not_block_entry_by_holding_window(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, _ = overlay_factory(
        make_state(scheduled_date=synthetic_calendar.session_date(11))
    )

    decision = overlay.evaluate_entry_candidate(
        canonical_asset_id=ASSET,
        signal_time=ny_dt(2026, 10, 5, 16),
        signal_session=synthetic_calendar.session_date(0),
        planned_entry_session=synthetic_calendar.session_date(1),
    )

    assert decision.action is EarningsIntegrationAction.ENTRY_ALLOWED
    assert decision.risk_decision is not None
    assert decision.risk_decision.entry_blackout is False


def test_EINT_A06_estimated_and_confirmed_have_identical_entry_treatment(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    event = synthetic_calendar.session_date(6)
    estimated = make_state(
        lifecycle_state=LifecycleState.ESTIMATED,
        scheduled_date=event,
    )
    confirmed = replace(estimated, lifecycle_state=LifecycleState.CONFIRMED)
    decisions = []
    for state in (estimated, confirmed):
        overlay, _ = overlay_factory(state)
        decisions.append(
            overlay.evaluate_entry_candidate(
                canonical_asset_id=ASSET,
                signal_time=ny_dt(2026, 10, 5, 16),
                signal_session=synthetic_calendar.session_date(0),
                planned_entry_session=synthetic_calendar.session_date(1),
            )
        )

    assert [decision.action for decision in decisions] == [
        EarningsIntegrationAction.ENTRY_BLOCKED,
        EarningsIntegrationAction.ENTRY_BLOCKED,
    ]
    assert decisions[0].risk_decision == decisions[1].risk_decision


@pytest.mark.hard_gate_6c8b
def test_EINT_A07_execution_revalidation_performs_second_query(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    state = make_state(scheduled_date=synthetic_calendar.session_date(11))
    overlay, query = overlay_factory(state, state)
    signal_time = ny_dt(2026, 10, 5, 16)
    planned = synthetic_calendar.session_date(1)

    overlay.evaluate_entry_candidate(
        canonical_asset_id=ASSET,
        signal_time=signal_time,
        signal_session=synthetic_calendar.session_date(0),
        planned_entry_session=planned,
    )
    overlay.revalidate_pending_entry(
        canonical_asset_id=ASSET,
        signal_time=signal_time,
        execution_time=ny_dt(2026, 10, 6, 9, 30),
        planned_entry_session=planned,
    )

    assert len(query.calls) == 2
    assert query.calls[0]["as_of"] == signal_time
    assert query.calls[1]["as_of"] == ny_dt(2026, 10, 6, 9, 30)


@pytest.mark.hard_gate_6c8b
def test_EINT_A08_intervening_blocking_revision_invalidates_pending_entry(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    signal_time = ny_dt(2026, 10, 5, 16)
    execution_time = ny_dt(2026, 10, 6, 9, 30)
    initial = make_state(scheduled_date=synthetic_calendar.session_date(11))
    learned_at = execution_time - timedelta(minutes=30)
    blocking = make_state(
        as_of=execution_time,
        knowledge_effective_at=learned_at,
        scheduled_date=synthetic_calendar.session_date(5),
    )
    overlay, _ = overlay_factory(initial, blocking)

    at_signal = overlay.evaluate_entry_candidate(
        canonical_asset_id=ASSET,
        signal_time=signal_time,
        signal_session=synthetic_calendar.session_date(0),
        planned_entry_session=synthetic_calendar.session_date(1),
    )
    at_execution = overlay.revalidate_pending_entry(
        canonical_asset_id=ASSET,
        signal_time=signal_time,
        execution_time=execution_time,
        planned_entry_session=synthetic_calendar.session_date(1),
    )

    assert at_signal.action is EarningsIntegrationAction.ENTRY_ALLOWED
    assert (
        at_execution.action
        is EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
    )


@pytest.mark.hard_gate_6c8b
def test_EINT_A09_unknown_execution_state_invalidates_pending_entry(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    execution_time = ny_dt(2026, 10, 6, 9, 30)
    overlay, _ = overlay_factory(_unknown(make_state, as_of=execution_time))

    decision = overlay.revalidate_pending_entry(
        canonical_asset_id=ASSET,
        signal_time=ny_dt(2026, 10, 5, 16),
        execution_time=execution_time,
        planned_entry_session=synthetic_calendar.session_date(1),
    )

    assert (
        decision.action
        is EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
    )
    assert decision.risk_decision is not None
    assert decision.risk_decision.pending_entry_invalidated is True


def test_EINT_A10_no_new_blocking_knowledge_allows_pending_entry(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    state = make_state(scheduled_date=synthetic_calendar.session_date(11))
    overlay, _ = overlay_factory(state)

    decision = overlay.revalidate_pending_entry(
        canonical_asset_id=ASSET,
        signal_time=ny_dt(2026, 10, 5, 16),
        execution_time=ny_dt(2026, 10, 6, 9, 30),
        planned_entry_session=synthetic_calendar.session_date(1),
    )

    assert decision.action is EarningsIntegrationAction.PENDING_ENTRY_ALLOWED
    assert decision.risk_decision is not None
    assert decision.risk_decision.pending_entry_invalidated is False
