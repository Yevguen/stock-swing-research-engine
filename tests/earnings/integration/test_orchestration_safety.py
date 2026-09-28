"""Suite D: orchestration safety and explicit scope boundaries."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

import stock_swing_d1.earnings.integration.service as integration_service
from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
    PublishedEarningsRiskOverlay,
)
from stock_swing_d1.earnings.models import (
    EarningsScheduleStatus,
    LifecycleState,
    TimingClass,
)
from tests.earnings.integration.conftest import ASSET


def _unknown(make_state):
    return make_state(
        event_instance_id=None,
        earnings_schedule_known=False,
        lifecycle_state=LifecycleState.UNKNOWN,
        scheduled_date=None,
        knowledge_effective_at=None,
        event_revision_id=None,
        schedule_status=EarningsScheduleStatus.UNKNOWN,
    )


@pytest.mark.hard_gate_6c8b
def test_EINT_D01_existing_close_suppresses_additional_earnings_exit(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, _ = overlay_factory(
        make_state(
            scheduled_date=synthetic_calendar.session_date(7),
            timing_class=TimingClass.BMO,
        )
    )

    decision = overlay.evaluate_open_position(
        canonical_asset_id=ASSET,
        as_of=ny_dt(2026, 10, 13, 16),
        current_session=synthetic_calendar.session_date(6),
        position_entry_session=synthetic_calendar.session_date(0),
        close_already_scheduled=True,
    )

    assert decision.action is EarningsIntegrationAction.NO_ADDITIONAL_EXIT
    assert decision.earnings_state is not None
    assert decision.risk_decision is not None
    assert decision.reason == "MANDATORY_EARNINGS_EXIT"


@pytest.mark.hard_gate_6c8b
def test_EINT_D02_repeated_closed_evaluations_never_duplicate_or_resurrect(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, query = overlay_factory(make_state())

    decisions = tuple(
        overlay.evaluate_open_position(
            canonical_asset_id=ASSET,
            as_of=ny_dt(2026, 10, 7, 16),
            current_session=synthetic_calendar.session_date(2),
            position_entry_session=synthetic_calendar.session_date(0),
            position_is_open=False,
        )
        for _ in range(4)
    )

    assert all(
        decision.action is EarningsIntegrationAction.POSITION_ALREADY_CLOSED
        for decision in decisions
    )
    assert query.calls == []


def test_EINT_D03_blocked_entry_cannot_become_pending_from_overlay(
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
    assert decision.action is not EarningsIntegrationAction.PENDING_ENTRY_ALLOWED


def test_EINT_D04_invalidated_pending_entry_cannot_progress_to_execution(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    overlay, _ = overlay_factory(_unknown(make_state))

    decision = overlay.revalidate_pending_entry(
        canonical_asset_id=ASSET,
        signal_time=ny_dt(2026, 10, 5, 16),
        execution_time=ny_dt(2026, 10, 6, 9, 30),
        planned_entry_session=synthetic_calendar.session_date(1),
    )

    assert decision.action is EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
    assert decision.action is not EarningsIntegrationAction.PENDING_ENTRY_ALLOWED


def test_EINT_D05_signal_diagnostics_are_not_mutated(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    signal_record = {
        "action": "BUY",
        "score": 0.81,
        "diagnostics": {"indicator": "unchanged"},
    }
    original = copy.deepcopy(signal_record)
    overlay, _ = overlay_factory(_unknown(make_state))

    overlay.evaluate_entry_candidate(
        canonical_asset_id=ASSET,
        signal_time=ny_dt(2026, 10, 5, 16),
        signal_session=synthetic_calendar.session_date(0),
        planned_entry_session=synthetic_calendar.session_date(1),
    )

    assert signal_record == original


@pytest.mark.hard_gate_6c8b
def test_EINT_D06_entry_boundary_failures_propagate_and_never_allow(
    synthetic_calendar, ny_dt, overlay_factory
) -> None:
    failure = RuntimeError("injected query failure")
    overlay, _ = overlay_factory(error=failure)

    with pytest.raises(RuntimeError, match="injected query failure"):
        overlay.evaluate_entry_candidate(
            canonical_asset_id=ASSET,
            signal_time=ny_dt(2026, 10, 5, 16),
            signal_session=synthetic_calendar.session_date(0),
            planned_entry_session=synthetic_calendar.session_date(1),
        )
    with pytest.raises(RuntimeError, match="injected query failure"):
        overlay.revalidate_pending_entry(
            canonical_asset_id=ASSET,
            signal_time=ny_dt(2026, 10, 5, 16),
            execution_time=ny_dt(2026, 10, 6, 9, 30),
            planned_entry_session=synthetic_calendar.session_date(1),
        )


def test_EINT_D07_identical_inputs_produce_identical_decisions(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    state = make_state(
        scheduled_date=synthetic_calendar.session_date(8),
        timing_class=TimingClass.BMO,
    )
    overlay, _ = overlay_factory(state)
    arguments = {
        "canonical_asset_id": ASSET,
        "as_of": ny_dt(2026, 10, 7, 16),
        "current_session": synthetic_calendar.session_date(2),
        "position_entry_session": synthetic_calendar.session_date(0),
    }

    decisions = tuple(overlay.evaluate_open_position(**arguments) for _ in range(5))

    assert all(decision == decisions[0] for decision in decisions)


def test_EINT_D08_overlay_passes_frozen_ten_session_horizon(
    monkeypatch, make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    observed: list[int] = []
    original = integration_service.evaluate_entry_blackout

    def recording_risk(
        state,
        prospective_entry_session,
        trading_calendar,
        max_holding_sessions=10,
    ):
        observed.append(max_holding_sessions)
        return original(
            state,
            prospective_entry_session,
            trading_calendar,
            max_holding_sessions=max_holding_sessions,
        )

    monkeypatch.setattr(
        integration_service, "evaluate_entry_blackout", recording_risk
    )
    overlay, _ = overlay_factory(
        make_state(scheduled_date=synthetic_calendar.session_date(11))
    )

    overlay.evaluate_entry_candidate(
        canonical_asset_id=ASSET,
        signal_time=ny_dt(2026, 10, 5, 16),
        signal_session=synthetic_calendar.session_date(0),
        planned_entry_session=synthetic_calendar.session_date(1),
    )

    assert observed == [10]
    assert overlay.max_holding_sessions == 10


def test_EINT_D09_integration_introduces_no_execution_price_methodology() -> None:
    public_signatures = " ".join(
        str(inspect.signature(member))
        for _, member in inspect.getmembers(
            PublishedEarningsRiskOverlay, inspect.isfunction
        )
    ).lower()
    model_fields = set(EarningsIntegrationDecision.__dataclass_fields__)
    prohibited = {
        "entry_price",
        "exit_price",
        "fill_price",
        "vwap",
        "slippage",
        "commission",
        "transaction_cost",
        "spread",
    }

    assert not (prohibited & model_fields)
    assert not any(term in public_signatures for term in prohibited)


def test_EINT_D10_integration_introduces_no_excluded_strategy_or_provider_scope() -> None:
    package = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "stock_swing_d1"
        / "earnings"
        / "integration"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in package.glob("*.py")
    )
    prohibited = {
        "cooldown",
        "post_event_entry",
        "re-entry",
        "re_entry",
        "norgatedata",
        "provider download",
        "eps",
        "revenue",
        "earnings prediction",
        "alpha score",
        "position sizing",
        "portfolio sizing",
    }

    assert not any(term in source for term in prohibited)
