"""Suite C: PIT revisions, late knowledge, and snapshot immutability."""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    PublishedEarningsRiskOverlay,
)
from stock_swing_d1.earnings.models import (
    EarningsScheduleStatus,
    LifecycleState,
    TimingClass,
    TransitionType,
)
from stock_swing_d1.earnings.persistence import (
    EarningsPersistencePaths,
    publish_earnings_revisions,
)
from stock_swing_d1.earnings.query import PublishedEarningsPITQuery
from tests.earnings.integration.conftest import ASSET, PROVIDER


def _evaluate(overlay, synthetic_calendar, ny_dt, current_index=2):
    session = synthetic_calendar.session_date(current_index)
    return overlay.evaluate_open_position(
        canonical_asset_id=ASSET,
        as_of=ny_dt(2026, 10, session.day, 16),
        current_session=session,
        position_entry_session=synthetic_calendar.session_date(0),
    )


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


def _create_snapshot_a(tmp_path, make_revision, synthetic_calendar):
    paths = EarningsPersistencePaths(tmp_path / "snapshot")
    first = make_revision(
        event_revision_id="SNAPSHOT-A-REVISION",
        provider_record_id="SNAPSHOT-A-RECORD",
        provider_sequence=1,
        scheduled_date=synthetic_calendar.session_date(8),
        timing_class=TimingClass.BMO,
    )
    result = publish_earnings_revisions(
        [first], paths=paths, build_id="SNAPSHOT-A"
    )
    assert result.success is True
    query = PublishedEarningsPITQuery.from_persistence_paths(paths)
    overlay = PublishedEarningsRiskOverlay(
        query,
        provider_name=PROVIDER,
        trading_calendar=synthetic_calendar,
    )
    return paths, first, overlay


def _publish_snapshot_b(paths, first, synthetic_calendar, utc_dt) -> None:
    effective_at = utc_dt(2026, 10, 6, 13)
    postponed = replace(
        first,
        event_revision_id="SNAPSHOT-B-REVISION",
        provider_record_id="SNAPSHOT-B-RECORD",
        knowledge_date=effective_at.date(),
        knowledge_available_at=effective_at,
        strategy_effective_at=effective_at,
        provider_sequence=2,
        transition_type=TransitionType.POSTPONED,
        scheduled_date=synthetic_calendar.session_date(10),
        ingested_at=effective_at + timedelta(minutes=5),
    )
    result = publish_earnings_revisions(
        [postponed], paths=paths, build_id="SNAPSHOT-B"
    )
    assert result.success is True


def test_EINT_C01_confirmation_without_schedule_change_keeps_deadline(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    event = synthetic_calendar.session_date(8)
    estimated = make_state(
        lifecycle_state=LifecycleState.ESTIMATED,
        scheduled_date=event,
        timing_class=TimingClass.BMO,
    )
    confirmed = replace(estimated, lifecycle_state=LifecycleState.CONFIRMED)
    overlay, _ = overlay_factory(estimated, confirmed)

    before = _evaluate(overlay, synthetic_calendar, ny_dt)
    after = _evaluate(overlay, synthetic_calendar, ny_dt)

    assert before.risk_decision is not None
    assert after.risk_decision is not None
    assert before.risk_decision.last_safe_exit_session == after.risk_decision.last_safe_exit_session


@pytest.mark.hard_gate_6c8b
def test_EINT_C02_postponement_recomputes_a_later_deadline(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    original = make_state(
        scheduled_date=synthetic_calendar.session_date(5),
        timing_class=TimingClass.BMO,
    )
    postponed = replace(
        original,
        scheduled_date=synthetic_calendar.session_date(8),
        transition_type=TransitionType.POSTPONED,
    )
    overlay, _ = overlay_factory(original, postponed)

    before = _evaluate(overlay, synthetic_calendar, ny_dt)
    after = _evaluate(overlay, synthetic_calendar, ny_dt)

    assert before.risk_decision is not None
    assert after.risk_decision is not None
    assert before.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(4)
    assert after.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(7)


@pytest.mark.hard_gate_6c8b
def test_EINT_C03_advancement_recomputes_an_earlier_deadline_when_visible(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    original = make_state(
        scheduled_date=synthetic_calendar.session_date(8),
        timing_class=TimingClass.BMO,
    )
    advanced = replace(
        original,
        scheduled_date=synthetic_calendar.session_date(5),
        transition_type=TransitionType.ADVANCED,
    )
    overlay, query = overlay_factory(original, advanced)

    before = _evaluate(overlay, synthetic_calendar, ny_dt)
    at_visibility = _evaluate(overlay, synthetic_calendar, ny_dt)

    assert len(query.calls) == 2
    assert before.risk_decision is not None
    assert at_visibility.risk_decision is not None
    assert before.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(7)
    assert at_visibility.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(4)


@pytest.mark.hard_gate_6c8b
def test_EINT_C04_cancellation_removes_unexecuted_exit_obligation(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    active = make_state(
        scheduled_date=synthetic_calendar.session_date(5),
        timing_class=TimingClass.BMO,
    )
    overlay, _ = overlay_factory(active, _unknown(make_state))

    before = _evaluate(overlay, synthetic_calendar, ny_dt)
    after = _evaluate(overlay, synthetic_calendar, ny_dt)

    assert before.risk_decision is not None
    assert before.risk_decision.last_safe_exit_session is not None
    assert after.action is EarningsIntegrationAction.HOLD_POSITION
    assert after.risk_decision is not None
    assert after.risk_decision.last_safe_exit_session is None
    assert after.risk_decision.mandatory_exit is False


@pytest.mark.hard_gate_6c8b
def test_EINT_C05_closed_position_cannot_be_resurrected_by_revisions(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    states = (
        make_state(lifecycle_state=LifecycleState.CONFIRMED),
        make_state(scheduled_date=synthetic_calendar.session_date(10)),
        _unknown(make_state),
    )
    overlay, query = overlay_factory(*states)

    decisions = [
        overlay.evaluate_open_position(
            canonical_asset_id=ASSET,
            as_of=ny_dt(2026, 10, 7 + index, 16),
            current_session=synthetic_calendar.session_date(2 + index),
            position_entry_session=synthetic_calendar.session_date(0),
            position_is_open=False,
        )
        for index in range(3)
    ]

    assert all(
        decision.action is EarningsIntegrationAction.POSITION_ALREADY_CLOSED
        for decision in decisions
    )
    assert query.calls == []


@pytest.mark.hard_gate_6c8b
def test_EINT_C06_late_knowledge_reports_unavoidable_exposure(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    current = synthetic_calendar.session_date(5)
    learned_at = synthetic_calendar.decision_time(current)
    overlay, _ = overlay_factory(
        make_state(
            knowledge_effective_at=learned_at,
            scheduled_date=current,
            timing_class=TimingClass.BMO,
        )
    )

    decision = _evaluate(overlay, synthetic_calendar, ny_dt, 5)

    assert decision.action is EarningsIntegrationAction.UNAVOIDABLE_EARNINGS_EXPOSURE
    assert decision.risk_decision is not None
    assert decision.risk_decision.unavoidable_earnings_exposure is True
    assert decision.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(4)


def test_EINT_C07_known_deadline_missed_reports_missed_without_rewrite(
    make_state, synthetic_calendar, ny_dt, overlay_factory
) -> None:
    event = synthetic_calendar.session_date(5)
    overlay, _ = overlay_factory(
        make_state(
            knowledge_effective_at=synthetic_calendar.decision_time(
                synthetic_calendar.session_date(0)
            ),
            scheduled_date=event,
            timing_class=TimingClass.BMO,
        )
    )

    decision = _evaluate(overlay, synthetic_calendar, ny_dt, 5)

    assert decision.action is EarningsIntegrationAction.MISSED_EXIT_DEADLINE
    assert decision.risk_decision is not None
    assert decision.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(4)
    assert decision.risk_decision.unavoidable_earnings_exposure is False


@pytest.mark.hard_gate_6c8b
def test_EINT_C08_overlay_a_remains_bound_after_filesystem_snapshot_b(
    tmp_path, make_revision, synthetic_calendar, utc_dt, ny_dt
) -> None:
    paths, first, overlay_a = _create_snapshot_a(
        tmp_path, make_revision, synthetic_calendar
    )
    identity_a = (overlay_a.build_id, overlay_a.output_sha256)
    before = _evaluate(overlay_a, synthetic_calendar, ny_dt)

    _publish_snapshot_b(paths, first, synthetic_calendar, utc_dt)
    after = _evaluate(overlay_a, synthetic_calendar, ny_dt)

    assert identity_a[0] == "SNAPSHOT-A"
    assert (overlay_a.build_id, overlay_a.output_sha256) == identity_a
    assert before == after
    assert after.risk_decision is not None
    assert after.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(7)


def test_EINT_C09_new_overlay_b_sees_b_while_overlay_a_still_sees_a(
    tmp_path, make_revision, synthetic_calendar, utc_dt, ny_dt
) -> None:
    paths, first, overlay_a = _create_snapshot_a(
        tmp_path, make_revision, synthetic_calendar
    )
    _publish_snapshot_b(paths, first, synthetic_calendar, utc_dt)
    query_b = PublishedEarningsPITQuery.from_persistence_paths(paths)
    overlay_b = PublishedEarningsRiskOverlay(
        query_b,
        provider_name=PROVIDER,
        trading_calendar=synthetic_calendar,
    )

    result_a = _evaluate(overlay_a, synthetic_calendar, ny_dt)
    result_b = _evaluate(overlay_b, synthetic_calendar, ny_dt)

    assert overlay_a.build_id == "SNAPSHOT-A"
    assert overlay_b.build_id == "SNAPSHOT-B"
    assert overlay_a.output_sha256 != overlay_b.output_sha256
    assert result_a.risk_decision is not None
    assert result_b.risk_decision is not None
    assert result_a.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(7)
    assert result_b.risk_decision.last_safe_exit_session == synthetic_calendar.session_date(9)


@pytest.mark.hard_gate_6c8b
def test_EINT_C10_integration_imports_only_query_and_domain_boundaries() -> None:
    package = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "stock_swing_d1"
        / "earnings"
        / "integration"
    )
    imported_modules: set[str] = set()
    source = ""
    for path in package.glob("*.py"):
        module_source = path.read_text(encoding="utf-8")
        source += module_source.lower()
        tree = ast.parse(module_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)

    forbidden_modules = {
        "stock_swing_d1.earnings.persistence",
        "stock_swing_d1.earnings.pit",
        "stock_swing_d1.earnings.normalization",
        "norgatedata",
    }
    forbidden_calls = {
        "read_canonical_revisions",
        "write_canonical_revisions",
        "publish_earnings_revisions",
        "read_publication_manifest",
        "reconstruct_earnings_state",
    }
    assert not any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for module in imported_modules
        for forbidden in forbidden_modules
    )
    assert not (forbidden_calls & set(source.replace("(", " ").split()))
    assert "stock_swing_d1.earnings.query" in imported_modules
