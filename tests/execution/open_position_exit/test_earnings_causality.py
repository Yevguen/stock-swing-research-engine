"""Prior-boundary authorization and current-boundary PIT causality tests."""

from dataclasses import replace
from datetime import datetime, time

import pytest

from stock_swing_d1.earnings.models import (
    EarningsScheduleStatus,
    LifecycleState,
    TimingClass,
)
from stock_swing_d1.execution.open_position_exit import (
    EarningsExitStatus,
    ExitBoundary,
    OpenPositionExitReason,
    OpenPositionExitValidationError,
)
from tests.execution.open_position_exit.conftest import NEW_YORK, SESSIONS


def _inactive(make_earnings_decision, boundary_index: int):
    return make_earnings_decision(
        boundary_session=SESSIONS[boundary_index],
        scheduled_date=None,
        lifecycle_state=LifecycleState.CANCELLED,
        schedule_status=EarningsScheduleStatus.NO_EVENT_WITHIN_PROVIDER_HORIZON,
    )


def _known_event(
    make_earnings_decision,
    *,
    boundary_index: int,
    scheduled_index: int,
    timing_class: TimingClass,
    knowledge_effective_at: datetime | None = None,
):
    return make_earnings_decision(
        boundary_session=SESSIONS[boundary_index],
        scheduled_date=SESSIONS[scheduled_index],
        timing_class=timing_class,
        knowledge_effective_at=knowledge_effective_at,
    )


def test_obligation_first_visible_exactly_at_t_close_is_non_terminal(
    evaluator, make_input, make_bar, make_earnings_decision, calendar
) -> None:
    current_index = 5
    current = _known_event(
        make_earnings_decision,
        boundary_index=current_index,
        scheduled_index=6,
        timing_class=TimingClass.BMO,
        knowledge_effective_at=calendar.decision_time(SESSIONS[current_index]),
    )

    decision = evaluator.evaluate(
        make_input(
            session_index=current_index,
            bar=make_bar(session=SESSIONS[current_index], close=103.0),
            earnings_decision=current,
            prior_boundary_earnings_decision=_inactive(
                make_earnings_decision, current_index - 1
            ),
        )
    )

    assert decision.earnings_exit_status is EarningsExitStatus.DEADLINE_MISSED
    assert decision.exit_required is False
    assert decision.triggered_reasons == ()
    assert decision.selected_reason is None
    assert decision.exit_boundary is None
    assert decision.reference_exit_price is None
    assert decision.final_execution_price is None


def test_obligation_first_visible_between_prior_and_t_close_is_non_terminal(
    evaluator, make_input, make_bar, make_earnings_decision
) -> None:
    current_index = 5
    learned_intraday = datetime.combine(
        SESSIONS[current_index], time(12), tzinfo=NEW_YORK
    )
    current = _known_event(
        make_earnings_decision,
        boundary_index=current_index,
        scheduled_index=6,
        timing_class=TimingClass.DURING_MARKET,
        knowledge_effective_at=learned_intraday,
    )

    decision = evaluator.evaluate(
        make_input(
            session_index=current_index,
            bar=make_bar(session=SESSIONS[current_index]),
            earnings_decision=current,
            prior_boundary_earnings_decision=_inactive(
                make_earnings_decision, current_index - 1
            ),
        )
    )

    assert decision.earnings_exit_status is EarningsExitStatus.DEADLINE_MISSED
    assert decision.exit_required is False


def test_current_revision_cannot_create_retroactive_t_authorization(
    evaluator, make_input, make_bar, make_earnings_decision, calendar
) -> None:
    current_index = 5
    prior = _known_event(
        make_earnings_decision,
        boundary_index=4,
        scheduled_index=9,
        timing_class=TimingClass.AMC,
    )
    accelerated = _known_event(
        make_earnings_decision,
        boundary_index=current_index,
        scheduled_index=6,
        timing_class=TimingClass.BMO,
        knowledge_effective_at=calendar.decision_time(SESSIONS[current_index]),
    )

    decision = evaluator.evaluate(
        make_input(
            session_index=current_index,
            bar=make_bar(session=SESSIONS[current_index]),
            earnings_decision=accelerated,
            prior_boundary_earnings_decision=prior,
        )
    )

    assert decision.earnings_deadline_session == SESSIONS[current_index]
    assert decision.earnings_exit_status is EarningsExitStatus.DEADLINE_MISSED
    assert decision.exit_required is False


def test_current_cancellation_cannot_erase_prior_t_close_authorization(
    evaluator, make_input, make_bar, make_earnings_decision, calendar
) -> None:
    current_index = 5
    prior = _known_event(
        make_earnings_decision,
        boundary_index=4,
        scheduled_index=6,
        timing_class=TimingClass.BMO,
    )
    cancelled = make_earnings_decision(
        boundary_session=SESSIONS[current_index],
        scheduled_date=None,
        knowledge_effective_at=calendar.decision_time(SESSIONS[current_index]),
        lifecycle_state=LifecycleState.CANCELLED,
        schedule_status=EarningsScheduleStatus.NO_EVENT_WITHIN_PROVIDER_HORIZON,
    )

    decision = evaluator.evaluate(
        make_input(
            session_index=current_index,
            bar=make_bar(session=SESSIONS[current_index], close=104.0),
            earnings_decision=cancelled,
            prior_boundary_earnings_decision=prior,
        )
    )

    assert decision.selected_reason is OpenPositionExitReason.EARNINGS_FORCED_EXIT
    assert decision.exit_boundary is ExitBoundary.CLOSE
    assert decision.reference_exit_price == 104.0
    assert decision.earnings_deadline_session == SESSIONS[current_index]
    assert decision.earnings_decision is cancelled
    assert decision.prior_boundary_earnings_decision is prior


@pytest.mark.parametrize(
    ("bar_values", "expected_reason"),
    [
        ({"low": 95.0, "close": 97.0}, OpenPositionExitReason.STOP_LOSS),
        ({"high": 109.0, "close": 107.0}, OpenPositionExitReason.TAKE_PROFIT),
    ],
)
def test_late_discovery_with_protective_exit_retains_missed_status(
    evaluator,
    make_input,
    make_bar,
    make_earnings_decision,
    calendar,
    bar_values,
    expected_reason,
) -> None:
    current_index = 5
    current = _known_event(
        make_earnings_decision,
        boundary_index=current_index,
        scheduled_index=6,
        timing_class=TimingClass.BMO,
        knowledge_effective_at=calendar.decision_time(SESSIONS[current_index]),
    )

    decision = evaluator.evaluate(
        make_input(
            session_index=current_index,
            bar=make_bar(session=SESSIONS[current_index], **bar_values),
            earnings_decision=current,
            prior_boundary_earnings_decision=_inactive(
                make_earnings_decision, current_index - 1
            ),
        )
    )

    assert decision.selected_reason is expected_reason
    assert decision.earnings_exit_status is EarningsExitStatus.DEADLINE_MISSED
    assert OpenPositionExitReason.EARNINGS_FORCED_EXIT not in decision.triggered_reasons


def test_late_discovery_on_session_10_leaves_max_holding_primary(
    evaluator, make_input, make_bar, make_earnings_decision, calendar
) -> None:
    current_index = 9
    current = _known_event(
        make_earnings_decision,
        boundary_index=current_index,
        scheduled_index=current_index,
        timing_class=TimingClass.AMC,
        knowledge_effective_at=calendar.decision_time(SESSIONS[current_index]),
    )

    decision = evaluator.evaluate(
        make_input(
            session_index=current_index,
            bar=make_bar(session=SESSIONS[current_index], close=106.0, high=107.0),
            earnings_decision=current,
            prior_boundary_earnings_decision=_inactive(
                make_earnings_decision, current_index - 1
            ),
        )
    )

    assert decision.triggered_reasons == (OpenPositionExitReason.MAX_HOLDING,)
    assert decision.selected_reason is OpenPositionExitReason.MAX_HOLDING
    assert decision.earnings_exit_status is EarningsExitStatus.DEADLINE_MISSED


def test_prior_boundary_evidence_must_use_exact_previous_decision_time(
    evaluator, make_input, make_bar, make_earnings_decision
) -> None:
    current_index = 5
    wrong_boundary = _known_event(
        make_earnings_decision,
        boundary_index=current_index,
        scheduled_index=6,
        timing_class=TimingClass.BMO,
    )

    with pytest.raises(OpenPositionExitValidationError) as raised:
        evaluator.evaluate(
            make_input(
                session_index=current_index,
                bar=make_bar(session=SESSIONS[current_index]),
                prior_boundary_earnings_decision=wrong_boundary,
            )
        )

    assert raised.value.code == "INVALID_EARNINGS_DECISION_BOUNDARY"


def test_prior_boundary_risk_must_be_reproducible(
    evaluator, make_input, make_bar, make_earnings_decision
) -> None:
    current_index = 5
    prior = _known_event(
        make_earnings_decision,
        boundary_index=4,
        scheduled_index=6,
        timing_class=TimingClass.BMO,
    )
    corrupted = replace(
        prior,
        risk_decision=replace(prior.risk_decision, last_safe_exit_session=SESSIONS[4]),
    )

    with pytest.raises(OpenPositionExitValidationError) as raised:
        evaluator.evaluate(
            make_input(
                session_index=current_index,
                bar=make_bar(session=SESSIONS[current_index]),
                prior_boundary_earnings_decision=corrupted,
            )
        )

    assert raised.value.code == "EARNINGS_RISK_DECISION_MISMATCH"


def test_prior_boundary_security_identity_must_match_position(
    evaluator, make_input, make_bar, make_earnings_decision
) -> None:
    prior = _known_event(
        make_earnings_decision,
        boundary_index=4,
        scheduled_index=6,
        timing_class=TimingClass.BMO,
    )
    corrupted = replace(
        prior,
        earnings_state=replace(
            prior.earnings_state, canonical_asset_id="NORGATE:9999"
        ),
    )

    with pytest.raises(OpenPositionExitValidationError) as raised:
        evaluator.evaluate(
            make_input(
                session_index=5,
                bar=make_bar(session=SESSIONS[5]),
                prior_boundary_earnings_decision=corrupted,
            )
        )

    assert raised.value.code == "EARNINGS_IDENTITY_MISMATCH"


def test_prior_boundary_action_must_match_reproduced_risk(
    evaluator, make_input, make_bar, make_earnings_decision
) -> None:
    from stock_swing_d1.earnings.integration import EarningsIntegrationAction

    prior = _known_event(
        make_earnings_decision,
        boundary_index=4,
        scheduled_index=6,
        timing_class=TimingClass.BMO,
    )
    corrupted = replace(
        prior, action=EarningsIntegrationAction.MISSED_EXIT_DEADLINE
    )

    with pytest.raises(OpenPositionExitValidationError) as raised:
        evaluator.evaluate(
            make_input(
                session_index=5,
                bar=make_bar(session=SESSIONS[5]),
                prior_boundary_earnings_decision=corrupted,
            )
        )

    assert raised.value.code == "EARNINGS_INTEGRATION_ACTION_MISMATCH"
