"""Suite B: open-position earnings deadline interpretation."""

from __future__ import annotations

import pytest

from stock_swing_d1.earnings.integration import EarningsIntegrationAction
from stock_swing_d1.earnings.models import TimingClass
from tests.earnings.integration.conftest import ASSET


def _evaluate(overlay, synthetic_calendar, ny_dt, current_index, **overrides):
    values = {
        "canonical_asset_id": ASSET,
        "as_of": ny_dt(2026, 10, synthetic_calendar.session_date(current_index).day, 16),
        "current_session": synthetic_calendar.session_date(current_index),
        "position_entry_session": synthetic_calendar.session_date(0),
    }
    values.update(overrides)
    return overlay.evaluate_open_position(**values)


def test_EINT_B01_open_position_is_freshly_queried_each_session(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    original = make_state(
        scheduled_date=synthetic_calendar.session_date(6),
        timing_class=TimingClass.BMO,
    )
    postponed = make_state(
        scheduled_date=synthetic_calendar.session_date(9),
        timing_class=TimingClass.BMO,
    )
    overlay, query = overlay_factory(original, postponed)

    first = _evaluate(overlay, synthetic_calendar, ny_dt, 2)
    second = _evaluate(overlay, synthetic_calendar, ny_dt, 3)

    assert len(query.calls) == 2
    assert first.risk_decision is not None
    assert second.risk_decision is not None
    assert first.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(5)
    assert second.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(8)


@pytest.mark.hard_gate_6c8b
def test_EINT_B02_amc_deadline_is_event_session(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    event = synthetic_calendar.session_date(7)
    overlay, _ = overlay_factory(
        make_state(scheduled_date=event, timing_class=TimingClass.AMC)
    )

    decision = _evaluate(overlay, synthetic_calendar, ny_dt, 2)

    assert decision.risk_decision is not None
    assert decision.risk_decision.last_safe_exit_session == event


@pytest.mark.hard_gate_6c8b
def test_EINT_B03_bmo_deadline_is_previous_session(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, _ = overlay_factory(
        make_state(
            scheduled_date=synthetic_calendar.session_date(7),
            timing_class=TimingClass.BMO,
        )
    )

    decision = _evaluate(overlay, synthetic_calendar, ny_dt, 2)

    assert decision.risk_decision is not None
    assert decision.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(6)


def test_EINT_B04_during_market_deadline_is_previous_session(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, _ = overlay_factory(
        make_state(
            scheduled_date=synthetic_calendar.session_date(7),
            timing_class=TimingClass.DURING_MARKET,
        )
    )

    decision = _evaluate(overlay, synthetic_calendar, ny_dt, 2)

    assert decision.risk_decision is not None
    assert decision.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(6)


@pytest.mark.hard_gate_6c8b
def test_EINT_B05_unknown_timing_deadline_is_previous_session(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    event = synthetic_calendar.session_date(7)
    overlay, _ = overlay_factory(
        make_state(scheduled_date=event, timing_class=TimingClass.UNKNOWN)
    )

    decision = _evaluate(overlay, synthetic_calendar, ny_dt, 2)

    assert decision.risk_decision is not None
    assert decision.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(6)
    assert decision.risk_decision.last_safe_exit_session != event


@pytest.mark.hard_gate_6c8b
def test_EINT_B06_mandatory_exit_before_deadline_means_hold(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, _ = overlay_factory(
        make_state(
            scheduled_date=synthetic_calendar.session_date(7),
            timing_class=TimingClass.BMO,
        )
    )

    decision = _evaluate(overlay, synthetic_calendar, ny_dt, 2)

    assert decision.risk_decision is not None
    assert decision.risk_decision.mandatory_exit is True
    assert decision.action is EarningsIntegrationAction.HOLD_POSITION


@pytest.mark.hard_gate_6c8b
def test_EINT_B07_deadline_session_requires_exit_this_session(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, _ = overlay_factory(
        make_state(
            scheduled_date=synthetic_calendar.session_date(7),
            timing_class=TimingClass.BMO,
        )
    )

    decision = _evaluate(overlay, synthetic_calendar, ny_dt, 6)

    assert decision.action is EarningsIntegrationAction.EXIT_REQUIRED_THIS_SESSION


@pytest.mark.hard_gate_6c8b
def test_EINT_B08_past_deadline_reports_missed_or_unavoidable_without_rewrite(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    event = synthetic_calendar.session_date(5)
    current = synthetic_calendar.session_date(5)
    early = synthetic_calendar.decision_time(synthetic_calendar.session_date(0))
    late = synthetic_calendar.decision_time(current)
    expected = (
        (early, EarningsIntegrationAction.MISSED_EXIT_DEADLINE),
        (late, EarningsIntegrationAction.UNAVOIDABLE_EARNINGS_EXPOSURE),
    )
    for knowledge_time, expected_action in expected:
        overlay, _ = overlay_factory(
            make_state(
                knowledge_effective_at=knowledge_time,
                scheduled_date=event,
                timing_class=TimingClass.BMO,
            )
        )

        decision = _evaluate(overlay, synthetic_calendar, ny_dt, 5)

        assert decision.action is expected_action
        assert decision.risk_decision is not None
        assert decision.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(4)


@pytest.mark.hard_gate_6c8b
def test_EINT_B09_closed_position_never_issues_another_close(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, query = overlay_factory(make_state())

    decision = _evaluate(
        overlay,
        synthetic_calendar,
        ny_dt,
        2,
        position_is_open=False,
    )

    assert decision.action is EarningsIntegrationAction.POSITION_ALREADY_CLOSED
    assert decision.earnings_state is None
    assert decision.risk_decision is None
    assert query.calls == []


def test_EINT_B10_position_entry_session_preserves_holding_age(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, _ = overlay_factory(
        make_state(
            scheduled_date=synthetic_calendar.session_date(8),
            timing_class=TimingClass.BMO,
        )
    )

    decision = _evaluate(overlay, synthetic_calendar, ny_dt, 3)

    assert decision.risk_decision is not None
    assert decision.risk_decision.holding_age_sessions == 3
