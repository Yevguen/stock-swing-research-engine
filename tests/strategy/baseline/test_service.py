"""Acceptance matrix for Phase 8 Baseline Strategy v0.1."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from datetime import date, datetime, timezone
from math import inf, nan
from zoneinfo import ZoneInfo

import pytest

from stock_swing_d1.earnings.integration import (
    EarningsIntegrationAction,
    EarningsIntegrationDecision,
)
from stock_swing_d1.strategy.baseline import (
    BaselineSignalAction,
    BaselineSignalEvaluator,
    BaselineStrategyValidationError,
)


def test_all_pass_produces_valid_long_signal_and_next_session(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()

    decision = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=True,
    )

    assert decision.action is BaselineSignalAction.VALID_LONG_SIGNAL
    assert decision.valid_long_signal is True
    assert decision.signal_session == date(2026, 8, 14)
    assert decision.signal_time == signal_time
    assert decision.planned_entry_session == date(2026, 8, 17)
    assert decision.adjusted_close == 105.0
    assert decision.atr_fraction == pytest.approx(2.0 / 105.0)
    assert decision.universe_eligible is True
    assert decision.close_above_sma50 is True
    assert decision.sma20_above_sma50 is True
    assert decision.rsi_above_50 is True
    assert decision.atr_above_minimum is True
    assert decision.earnings_entry_allowed is True
    assert decision.earnings_action is EarningsIntegrationAction.ENTRY_ALLOWED


@pytest.mark.parametrize(
    ("close", "indicator_overrides", "failed_diagnostic"),
    [
        (99.0, {}, "close_above_sma50"),
        (100.0, {}, "close_above_sma50"),
        (105.0, {"sma_20": 99.0}, "sma20_above_sma50"),
        (105.0, {"sma_20": 100.0}, "sma20_above_sma50"),
        (105.0, {"rsi_14": 49.0}, "rsi_above_50"),
        (105.0, {"rsi_14": 50.0}, "rsi_above_50"),
        (105.0, {"atr_14": 1.049}, "atr_above_minimum"),
    ],
    ids=(
        "close-below-sma50",
        "close-equals-sma50",
        "sma20-below-sma50",
        "sma20-equals-sma50",
        "rsi-below-50",
        "rsi-equals-50",
        "atr-fraction-below-one-percent",
    ),
)
def test_each_individual_technical_failure_produces_no_signal(
    evaluator_factory,
    make_bar,
    make_indicators,
    signal_time,
    close,
    indicator_overrides,
    failed_diagnostic,
) -> None:
    evaluator, _ = evaluator_factory()

    decision = evaluator.evaluate_signal(
        bar=make_bar(close=close),
        indicators=make_indicators(**indicator_overrides),
        universe_eligible=True,
    )

    assert decision.action is BaselineSignalAction.NO_SIGNAL
    assert decision.valid_long_signal is False
    assert getattr(decision, failed_diagnostic) is False
    assert decision.earnings_entry_allowed is True


def test_atr_exact_one_percent_boundary_passes(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()

    decision = evaluator.evaluate_signal(
        bar=make_bar(close=100.0),
        indicators=make_indicators(sma_20=102.0, sma_50=99.0, atr_14=1.0),
        universe_eligible=True,
    )

    assert decision.atr_fraction == 0.01
    assert decision.atr_above_minimum is True
    assert decision.action is BaselineSignalAction.VALID_LONG_SIGNAL


def test_atr_just_below_one_percent_fails(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()

    decision = evaluator.evaluate_signal(
        bar=make_bar(close=100.0),
        indicators=make_indicators(sma_50=99.0, atr_14=0.999_999),
        universe_eligible=True,
    )

    assert decision.atr_fraction == pytest.approx(0.009_999_99)
    assert decision.atr_above_minimum is False
    assert decision.action is BaselineSignalAction.NO_SIGNAL


@pytest.mark.parametrize("missing_field", ["sma_20", "sma_50", "rsi_14", "atr_14"])
def test_each_missing_required_indicator_is_a_normal_no_signal(
    evaluator_factory, make_bar, make_indicators, signal_time, missing_field
) -> None:
    evaluator, _ = evaluator_factory()

    decision = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(**{missing_field: None}),
        universe_eligible=True,
    )

    assert decision.action is BaselineSignalAction.NO_SIGNAL
    assert decision.valid_long_signal is False
    assert getattr(decision, missing_field) is None
    if missing_field == "atr_14":
        assert decision.atr_fraction is None


def test_ineligible_phase3_result_produces_no_signal(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()

    decision = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=False,
    )

    assert decision.universe_eligible is False
    assert decision.action is BaselineSignalAction.NO_SIGNAL


def test_earnings_block_produces_no_signal_and_preserves_reason(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory(
        EarningsIntegrationAction.ENTRY_BLOCKED,
        reason="EARNINGS_SCHEDULE_UNKNOWN",
    )

    decision = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=True,
    )

    assert decision.earnings_entry_allowed is False
    assert decision.earnings_action is EarningsIntegrationAction.ENTRY_BLOCKED
    assert decision.earnings_reason == "EARNINGS_SCHEDULE_UNKNOWN"
    assert decision.earnings_decision.reason == "EARNINGS_SCHEDULE_UNKNOWN"
    assert decision.action is BaselineSignalAction.NO_SIGNAL


def test_only_entry_allowed_action_can_make_signal_actionable(
    baseline_calendar, make_bar, make_indicators, signal_time
) -> None:
    class NonEntryOverlay:
        def evaluate_entry_candidate(self, **kwargs) -> EarningsIntegrationDecision:
            return EarningsIntegrationDecision(
                EarningsIntegrationAction.HOLD_POSITION, None, None, "NOT_ENTRY_ALLOW"
            )

    evaluator = BaselineSignalEvaluator(
        earnings_overlay=NonEntryOverlay(), trading_calendar=baseline_calendar
    )

    decision = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=True,
    )

    assert decision.earnings_entry_allowed is False
    assert decision.action is BaselineSignalAction.NO_SIGNAL


def test_earnings_call_uses_exact_t_to_t_plus_one_arguments(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, overlay = evaluator_factory()

    evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=True,
    )

    assert overlay.calls == [
        {
            "canonical_asset_id": "NORGATE:1001",
            "signal_time": signal_time,
            "signal_session": date(2026, 8, 14),
            "planned_entry_session": date(2026, 8, 17),
        }
    ]


def test_future_pit_knowledge_cannot_enter_session_t_signal(
    baseline_calendar, make_bar, make_indicators, signal_time
) -> None:
    future_time = datetime(2026, 8, 17, 13, 30, tzinfo=timezone.utc)

    class TimeSensitiveEarningsOverlay:
        def __init__(self) -> None:
            self.calls: list[datetime] = []

        def evaluate_entry_candidate(
            self, *, signal_time: datetime, **kwargs
        ) -> EarningsIntegrationDecision:
            self.calls.append(signal_time)
            action = (
                EarningsIntegrationAction.ENTRY_ALLOWED
                if signal_time < future_time
                else EarningsIntegrationAction.ENTRY_BLOCKED
            )
            return EarningsIntegrationDecision(action, None, None)

    overlay = TimeSensitiveEarningsOverlay()
    evaluator = BaselineSignalEvaluator(
        earnings_overlay=overlay,
        trading_calendar=baseline_calendar,
    )

    decision = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=True,
    )

    assert overlay.calls == [signal_time]
    assert signal_time < future_time
    assert decision.signal_time == signal_time
    assert decision.action is BaselineSignalAction.VALID_LONG_SIGNAL


def test_phase6_style_morning_time_cannot_masquerade_as_d1_completion(
    evaluator_factory, make_bar, make_indicators
) -> None:
    market_timezone = ZoneInfo("America/New_York")
    canonical_close = datetime(2026, 8, 14, 16, tzinfo=market_timezone)
    phase6_morning = datetime(2026, 8, 14, 9, 30, tzinfo=market_timezone)

    class SemanticallyDistinctCalendar:
        def __init__(self) -> None:
            self.phase6_decision_time_calls = 0

        def signal_decision_time(self, session: date) -> datetime:
            return canonical_close

        def decision_time(self, session: date) -> datetime:
            self.phase6_decision_time_calls += 1
            return phase6_morning

        def next_session(self, session: date) -> date:
            return date(2026, 8, 17)

    _, overlay = evaluator_factory()
    calendar = SemanticallyDistinctCalendar()
    evaluator = BaselineSignalEvaluator(
        earnings_overlay=overlay,
        trading_calendar=calendar,
    )

    decision = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=True,
    )

    assert decision.signal_time == canonical_close
    assert decision.signal_time != phase6_morning
    assert overlay.calls[0]["signal_time"] == canonical_close
    assert calendar.phase6_decision_time_calls == 0


@pytest.mark.parametrize(
    ("field_name", "indicator_value"),
    [
        ("security_id", "NORGATE:2002"),
        ("symbol", "OTHER"),
        ("trading_date", date(2026, 8, 18)),
        ("price_basis", "unadjusted"),
    ],
)
def test_bar_indicator_identity_mismatch_fails_closed(
    evaluator_factory,
    make_bar,
    make_indicators,
    signal_time,
    field_name,
    indicator_value,
) -> None:
    evaluator, overlay = evaluator_factory()
    indicators = make_indicators().model_copy(update={field_name: indicator_value})

    with pytest.raises(BaselineStrategyValidationError) as raised:
        evaluator.evaluate_signal(
            bar=make_bar(),
            indicators=indicators,
            universe_eligible=True,
        )

    assert raised.value.code == "BAR_INDICATOR_IDENTITY_MISMATCH"
    assert field_name in str(raised.value)
    assert overlay.calls == []


def test_matching_nonadjusted_basis_fails_at_public_boundary(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()
    bar = make_bar().model_copy(update={"price_basis": "unadjusted"})
    indicators = make_indicators().model_copy(update={"price_basis": "unadjusted"})

    with pytest.raises(BaselineStrategyValidationError) as raised:
        evaluator.evaluate_signal(
            bar=bar,
            indicators=indicators,
            universe_eligible=True,
        )

    assert raised.value.code == "INVALID_PRICE_BASIS"


@pytest.mark.parametrize("invalid_close", [0.0, -1.0, nan, inf, -inf])
def test_invalid_adjusted_close_fails_without_division(
    evaluator_factory, make_bar, make_indicators, signal_time, invalid_close
) -> None:
    evaluator, overlay = evaluator_factory()
    invalid_bar = make_bar().model_copy(update={"close": invalid_close})

    with pytest.raises(BaselineStrategyValidationError):
        evaluator.evaluate_signal(
            bar=invalid_bar,
            indicators=make_indicators(),
            universe_eligible=True,
        )

    assert overlay.calls == []


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("sma_20", nan),
        ("sma_50", inf),
        ("rsi_14", -inf),
        ("atr_14", -0.01),
    ],
)
def test_invalid_indicator_numeric_value_fails_closed(
    evaluator_factory,
    make_bar,
    make_indicators,
    signal_time,
    field_name,
    invalid_value,
) -> None:
    evaluator, overlay = evaluator_factory()
    indicators = make_indicators().model_copy(update={field_name: invalid_value})

    with pytest.raises(BaselineStrategyValidationError):
        evaluator.evaluate_signal(
            bar=make_bar(),
            indicators=indicators,
            universe_eligible=True,
        )

    assert overlay.calls == []


def test_naive_canonical_signal_time_fails_closed(
    evaluator_factory, make_bar, make_indicators
) -> None:
    class NaiveSignalCalendar:
        def signal_decision_time(self, session: date) -> datetime:
            return datetime(2026, 8, 14, 16)

        def next_session(self, session: date) -> date:
            return date(2026, 8, 17)

    _, overlay = evaluator_factory()
    evaluator = BaselineSignalEvaluator(
        earnings_overlay=overlay,
        trading_calendar=NaiveSignalCalendar(),
    )

    with pytest.raises(BaselineStrategyValidationError) as raised:
        evaluator.evaluate_signal(
            bar=make_bar(),
            indicators=make_indicators(),
            universe_eligible=True,
        )

    assert raised.value.code == "INVALID_SIGNAL_DECISION_TIME"
    assert overlay.calls == []


def test_unresolvable_signal_decision_time_fails_closed(
    evaluator_factory, make_bar, make_indicators
) -> None:
    class UnresolvableSignalCalendar:
        def signal_decision_time(self, session: date) -> datetime:
            raise ValueError("unknown signal session")

        def next_session(self, session: date) -> date:
            raise AssertionError("next_session must not be reached")

    _, overlay = evaluator_factory()
    evaluator = BaselineSignalEvaluator(
        earnings_overlay=overlay,
        trading_calendar=UnresolvableSignalCalendar(),
    )

    with pytest.raises(BaselineStrategyValidationError) as raised:
        evaluator.evaluate_signal(
            bar=make_bar(),
            indicators=make_indicators(),
            universe_eligible=True,
        )

    assert raised.value.code == "INVALID_SIGNAL_SESSION"
    assert overlay.calls == []


def test_unresolvable_next_session_fails_closed(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    class UnresolvableNextSessionCalendar:
        def signal_decision_time(self, session: date) -> datetime:
            return signal_time

        def next_session(self, session: date) -> date:
            raise ValueError("no next session")

    _, overlay = evaluator_factory()
    evaluator = BaselineSignalEvaluator(
        earnings_overlay=overlay,
        trading_calendar=UnresolvableNextSessionCalendar(),
    )

    with pytest.raises(BaselineStrategyValidationError) as raised:
        evaluator.evaluate_signal(
            bar=make_bar(),
            indicators=make_indicators(),
            universe_eligible=True,
        )

    assert raised.value.code == "INVALID_PLANNED_ENTRY_SESSION"
    assert overlay.calls == []


def test_universe_eligibility_must_be_explicit_boolean(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, overlay = evaluator_factory()

    with pytest.raises(BaselineStrategyValidationError) as raised:
        evaluator.evaluate_signal(
            bar=make_bar(),
            indicators=make_indicators(),
            universe_eligible=1,
        )

    assert raised.value.code == "INVALID_UNIVERSE_ELIGIBILITY"
    assert overlay.calls == []


def test_adjusted_bar_close_drives_both_trend_and_atr_fraction(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()
    bar = make_bar(close=120.0)

    decision = evaluator.evaluate_signal(
        bar=bar,
        indicators=make_indicators(sma_20=115.0, sma_50=110.0, atr_14=1.2),
        universe_eligible=True,
    )

    assert decision.adjusted_close == bar.close
    assert decision.close_above_sma50 == (bar.close > 110.0)
    assert decision.atr_fraction == pytest.approx(1.2 / bar.close)
    assert decision.atr_above_minimum is True


def test_future_t_plus_one_market_data_cannot_change_t_decision(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()
    bar_t = make_bar()
    indicators_t = make_indicators()
    before = evaluator.evaluate_signal(
        bar=bar_t,
        indicators=indicators_t,
        universe_eligible=True,
    )

    make_bar(trading_date=date(2026, 8, 17), close=1_000_000.0)
    make_indicators(
        trading_date=date(2026, 8, 17),
        sma_20=1.0,
        sma_50=999_999.0,
        rsi_14=0.0,
        atr_14=999_999.0,
    )
    after = evaluator.evaluate_signal(
        bar=bar_t,
        indicators=indicators_t,
        universe_eligible=True,
    )

    assert after == before


def test_identical_inputs_are_deterministic(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()
    arguments = {
        "bar": make_bar(),
        "indicators": make_indicators(),
        "universe_eligible": True,
    }

    assert evaluator.evaluate_signal(**arguments) == evaluator.evaluate_signal(
        **arguments
    )


@pytest.mark.parametrize(
    ("indicator_overrides", "diagnostic", "expected"),
    [
        ({}, "close_above_sma50", True),
        ({"sma_50": 105.0}, "close_above_sma50", False),
        ({"sma_20": 100.0}, "sma20_above_sma50", False),
        ({"rsi_14": 50.0}, "rsi_above_50", False),
        ({"atr_14": 1.05}, "atr_above_minimum", True),
        ({"atr_14": 1.049}, "atr_above_minimum", False),
    ],
)
def test_diagnostic_booleans_match_exact_comparisons(
    evaluator_factory,
    make_bar,
    make_indicators,
    signal_time,
    indicator_overrides,
    diagnostic,
    expected,
) -> None:
    evaluator, _ = evaluator_factory()

    decision = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(**indicator_overrides),
        universe_eligible=True,
    )

    assert getattr(decision, diagnostic) is expected
    assert decision.valid_long_signal == (
        decision.action is BaselineSignalAction.VALID_LONG_SIGNAL
    )


def test_decision_is_immutable(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()
    decision = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=True,
    )

    with pytest.raises(FrozenInstanceError):
        decision.action = BaselineSignalAction.NO_SIGNAL


def test_evaluation_does_not_mutate_market_inputs(
    evaluator_factory, make_bar, make_indicators, signal_time
) -> None:
    evaluator, _ = evaluator_factory()
    bar = make_bar()
    indicators = make_indicators()
    bar_before = bar.model_dump(mode="python")
    indicators_before = indicators.model_dump(mode="python")

    evaluator.evaluate_signal(
        bar=bar,
        indicators=indicators,
        universe_eligible=True,
    )

    assert bar.model_dump(mode="python") == bar_before
    assert indicators.model_dump(mode="python") == indicators_before


def test_evaluator_does_not_write_to_immutable_overlay(
    baseline_calendar, make_bar, make_indicators, signal_time
) -> None:
    @dataclass(frozen=True, slots=True)
    class ImmutableOverlay:
        decision: EarningsIntegrationDecision

        def evaluate_entry_candidate(self, **kwargs) -> EarningsIntegrationDecision:
            return self.decision

    overlay = ImmutableOverlay(
        EarningsIntegrationDecision(
            EarningsIntegrationAction.ENTRY_ALLOWED, None, None
        )
    )
    evaluator = BaselineSignalEvaluator(
        earnings_overlay=overlay, trading_calendar=baseline_calendar
    )

    evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=True,
    )

    assert overlay == ImmutableOverlay(overlay.decision)


def test_calendar_cannot_authorize_same_session_entry(
    make_bar, make_indicators, signal_time
) -> None:
    class SameSessionCalendar:
        def signal_decision_time(self, session):
            return datetime(2026, 8, 14, 16, tzinfo=timezone.utc)

        def next_session(self, session):
            return session

    class AllowOverlay:
        def evaluate_entry_candidate(self, **kwargs):
            raise AssertionError("invalid entry must fail before earnings call")

    evaluator = BaselineSignalEvaluator(
        earnings_overlay=AllowOverlay(), trading_calendar=SameSessionCalendar()
    )

    with pytest.raises(BaselineStrategyValidationError) as raised:
        evaluator.evaluate_signal(
            bar=make_bar(),
            indicators=make_indicators(),
            universe_eligible=True,
        )

    assert raised.value.code == "INVALID_PLANNED_ENTRY_SESSION"
