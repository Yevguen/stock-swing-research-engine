"""Focused Phase 15C.2 trust-boundary and cash-awareness regressions."""

from dataclasses import replace
from decimal import Decimal

import pytest

from stock_swing_d1.earnings.integration import EarningsIntegrationAction
from stock_swing_d1.execution.costs import (
    BacktestExecutionCostService,
    ExecutionCostPolicyRef,
    ExecutionCostSide,
)
from stock_swing_d1.execution.entry import (
    EntryExecutionCostQuoteService,
    EntryExecutionService,
    EntryExecutionStatus,
    EntryExecutionValidationError,
    create_sized_pending_entry,
)


class RecordingCostService:
    def __init__(self, delegate: BacktestExecutionCostService) -> None:
        self.delegate = delegate
        self.policy_ref = delegate.policy_ref
        self.calls: list[dict[str, object]] = []
        self.return_quote = None
        self.failure: Exception | None = None

    def quote(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.failure is not None:
            raise self.failure
        if self.return_quote is not None:
            return self.return_quote(**kwargs)
        return self.delegate.quote(**kwargs)


def _execute(
    service,
    make_signal,
    execution_bar,
    *,
    quantity: int = 12,
    cash_available: float = 10_000.0,
):
    pending = service.create_pending_entry(signal=make_signal())
    sized = create_sized_pending_entry(
        pending_entry=pending,
        fixed_shares=quantity,
        cash_available=cash_available,
    )
    return service.execute_pending_entry(
        pending_entry=pending,
        sized_pending_entry=sized,
        execution_bar=execution_bar,
    )


def test_public_protocol_and_constructor_require_authoritative_cost_service(
    entry_calendar, execution_cost_service
) -> None:
    assert hasattr(EntryExecutionCostQuoteService, "policy_ref")
    assert hasattr(EntryExecutionCostQuoteService, "quote")
    with pytest.raises(TypeError):
        EntryExecutionService(
            earnings_overlay=object(), trading_calendar=entry_calendar
        )

    class MissingQuote:
        policy_ref = execution_cost_service.policy_ref

    with pytest.raises(EntryExecutionValidationError) as raised:
        EntryExecutionService(
            earnings_overlay=object(),
            trading_calendar=entry_calendar,
            execution_cost_service=MissingQuote(),
        )
    assert raised.value.code == "INVALID_EXECUTION_COST_SERVICE"


@pytest.mark.parametrize("policy_ref", [None, object(), "not-a-policy-ref"])
def test_constructor_rejects_malformed_policy_ref(
    entry_calendar, policy_ref
) -> None:
    class InvalidService:
        def quote(self, **kwargs):
            raise AssertionError("must not be called")

    service = InvalidService()
    service.policy_ref = policy_ref
    with pytest.raises(EntryExecutionValidationError) as raised:
        EntryExecutionService(
            earnings_overlay=object(),
            trading_calendar=entry_calendar,
            execution_cost_service=service,
        )
    assert raised.value.code == "INVALID_EXECUTION_COST_SERVICE"


def test_phase9_preserves_price_and_requests_exact_fixed_q_decimal_buy_quote(
    service_factory,
    execution_cost_service,
    make_signal,
    make_execution_bar,
) -> None:
    recording = RecordingCostService(execution_cost_service)
    service, _, _ = service_factory(cost_service=recording)
    decision = _execute(
        service,
        make_signal,
        make_execution_bar(opening_price=100.0),
        quantity=12,
    )

    assert decision.candidate_execution_price == 100.05
    assert decision.slippage_amount == 0.05
    assert len(recording.calls) == 1
    assert recording.calls[0] == {
        "side": ExecutionCostSide.BUY,
        "quantity": 12,
        "fill_price": Decimal(str(decision.candidate_execution_price)),
    }
    quote = decision.candidate_execution_cost_quote
    assert quote is not None
    assert decision.execution_cost_quote == quote
    assert decision.candidate_cash_required == float(
        quote.notional + quote.execution_cost
    )
    assert decision.actual_cash_required == decision.candidate_cash_required


@pytest.mark.parametrize(
    ("action", "bar_kind", "expected_status"),
    [
        (
            EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED,
            "valid",
            EntryExecutionStatus.INVALIDATED_BY_EARNINGS,
        ),
        (
            EarningsIntegrationAction.PENDING_ENTRY_ALLOWED,
            "missing",
            EntryExecutionStatus.NO_EXECUTABLE_BAR,
        ),
        (
            EarningsIntegrationAction.PENDING_ENTRY_ALLOWED,
            "invalid",
            EntryExecutionStatus.INVALID_OPEN_PRICE,
        ),
    ],
)
def test_early_terminal_paths_make_zero_cost_calls(
    service_factory,
    execution_cost_service,
    make_signal,
    make_execution_bar,
    action,
    bar_kind,
    expected_status,
) -> None:
    recording = RecordingCostService(execution_cost_service)
    service, _, _ = service_factory(action, cost_service=recording)
    bar = {
        "valid": make_execution_bar(),
        "missing": None,
        "invalid": make_execution_bar().model_copy(update={"open": 0.0}),
    }[bar_kind]
    decision = _execute(service, make_signal, bar)

    assert decision.status is expected_status
    assert recording.calls == []
    assert decision.candidate_execution_cost_quote is None
    assert decision.execution_cost_quote is None


def test_cost_can_cancel_price_affordable_fixed_q_without_resize(
    service_factory,
    execution_cost_service,
    make_signal,
    make_execution_bar,
) -> None:
    recording = RecordingCostService(execution_cost_service)
    service, _, _ = service_factory(cost_service=recording)
    decision = _execute(
        service,
        make_signal,
        make_execution_bar(opening_price=100.0),
        quantity=12,
        cash_available=1201.0,
    )

    assert Decimal("12") * Decimal(str(decision.candidate_execution_price)) < (
        Decimal("1201.0")
    )
    assert decision.status is (
        EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION
    )
    assert decision.requested_shares == 12
    assert decision.executed_shares == 0
    assert decision.candidate_execution_cost_quote is not None
    assert decision.candidate_cash_required > 1201.0
    assert decision.execution_price is None
    assert decision.execution_cost_quote is None
    assert decision.actual_cash_required is None
    assert len(recording.calls) == 1
    assert recording.calls[0]["quantity"] == 12


@pytest.mark.parametrize("cash_offset", [Decimal("0.01"), Decimal("0")])
def test_all_in_cash_below_or_equal_available_executes(
    service_factory,
    execution_cost_service,
    make_signal,
    make_execution_bar,
    cash_offset,
) -> None:
    fill = Decimal("100.05")
    quote = execution_cost_service.quote(
        side=ExecutionCostSide.BUY, quantity=12, fill_price=fill
    )
    exact_total = quote.notional + quote.execution_cost
    service, _, _ = service_factory()
    decision = _execute(
        service,
        make_signal,
        make_execution_bar(opening_price=100.0),
        cash_available=float(exact_total + cash_offset),
    )
    assert decision.status is EntryExecutionStatus.EXECUTED
    assert Decimal(str(decision.actual_cash_required)) == exact_total


def test_cost_service_failure_is_not_an_economic_cancellation(
    service_factory,
    execution_cost_service,
    make_signal,
    make_execution_bar,
) -> None:
    recording = RecordingCostService(execution_cost_service)
    recording.failure = RuntimeError("quote unavailable")
    service, _, _ = service_factory(cost_service=recording)
    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(service, make_signal, make_execution_bar())
    assert raised.value.code == "EXECUTION_COST_QUOTE_FAILED"
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_post_construction_corrupted_quote_fails_reconstruction(
    service_factory,
    execution_cost_service,
    make_signal,
    make_execution_bar,
) -> None:
    recording = RecordingCostService(execution_cost_service)

    def corrupted(**kwargs):
        quote = execution_cost_service.quote(**kwargs)
        object.__setattr__(quote, "execution_cost", Decimal("999"))
        return quote

    recording.return_quote = corrupted
    service, _, _ = service_factory(cost_service=recording)
    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(service, make_signal, make_execution_bar())
    assert raised.value.code == "INVALID_EXECUTION_COST_QUOTE"


def test_valid_quote_with_wrong_policy_fingerprint_fails_provenance(
    service_factory,
    execution_cost_service,
    make_signal,
    make_execution_bar,
) -> None:
    recording = RecordingCostService(execution_cost_service)
    wrong_ref = ExecutionCostPolicyRef(
        policy_id=execution_cost_service.policy_ref.policy_id,
        policy_fingerprint="b" * 64,
    )
    recording.return_quote = lambda **kwargs: replace(
        execution_cost_service.quote(**kwargs), policy_ref=wrong_ref
    )
    service, _, _ = service_factory(cost_service=recording)
    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(service, make_signal, make_execution_bar())
    assert raised.value.code == "INVALID_EXECUTION_COST_QUOTE"


@pytest.mark.parametrize("mismatch", ["quantity", "fill", "side"])
def test_wrong_but_mathematically_valid_quote_fails_candidate_reconciliation(
    service_factory,
    execution_cost_service,
    make_signal,
    make_execution_bar,
    mismatch,
) -> None:
    recording = RecordingCostService(execution_cost_service)

    def wrong_quote(**kwargs):
        altered = dict(kwargs)
        if mismatch == "quantity":
            altered["quantity"] += 1
        elif mismatch == "fill":
            altered["fill_price"] += Decimal("0.01")
        else:
            altered["side"] = ExecutionCostSide.SELL
        return execution_cost_service.quote(**altered)

    recording.return_quote = wrong_quote
    service, _, _ = service_factory(cost_service=recording)
    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(service, make_signal, make_execution_bar())
    assert raised.value.code == "INVALID_EXECUTION_COST_QUOTE"


def test_policy_ref_mutation_after_construction_fails_before_quote(
    service_factory,
    execution_cost_service,
    make_signal,
    make_execution_bar,
) -> None:
    recording = RecordingCostService(execution_cost_service)
    service, _, _ = service_factory(cost_service=recording)
    recording.policy_ref = ExecutionCostPolicyRef(
        policy_id=recording.policy_ref.policy_id,
        policy_fingerprint="c" * 64,
    )
    with pytest.raises(EntryExecutionValidationError) as raised:
        _execute(service, make_signal, make_execution_bar())
    assert raised.value.code == "INVALID_EXECUTION_COST_SERVICE"
    assert recording.calls == []


def test_identical_inputs_repeat_identical_cost_audit(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory()
    first = _execute(service, make_signal, make_execution_bar())
    second = _execute(service, make_signal, make_execution_bar())
    assert first == second
    assert first.candidate_execution_cost_quote == (
        second.candidate_execution_cost_quote
    )


def test_decision_reconstruction_rejects_incoherent_new_quote_fields(
    service_factory, make_signal, make_execution_bar
) -> None:
    service, _, _ = service_factory()
    executed = _execute(service, make_signal, make_execution_bar())
    with pytest.raises(EntryExecutionValidationError) as missing:
        replace(executed, execution_cost_quote=None)
    assert missing.value.code == "INVALID_EXECUTION_COST_QUOTE"

    early_service, _, _ = service_factory(
        EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
    )
    early = _execute(early_service, make_signal, make_execution_bar())
    with pytest.raises(EntryExecutionValidationError) as unexpected:
        replace(early, execution_cost_quote=executed.execution_cost_quote)
    assert unexpected.value.code == "INVALID_EXECUTION_COST_QUOTE"
