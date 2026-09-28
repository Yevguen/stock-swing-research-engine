"""Task 5C-A: the single allocation-boundary equity formula.

    portfolio_equity[T] = settled_cash[T]
                        + sum(pending_settlement.amount)
                        + sum(quantity * unadjusted D1 close[T])
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext

import pytest

from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.valuation import (
    PortfolioValuationError,
    PortfolioValuationPolicy,
    PortfolioValuationResult,
    value_portfolio_at_allocation_boundary,
)

from tests.valuation.conftest import ASSET_A, ASSET_B, SESSION


POLICY = PortfolioValuationPolicy()


def _value(state, marks=(), session=SESSION) -> PortfolioValuationResult:
    return value_portfolio_at_allocation_boundary(
        state=state, session=session, marks=marks, policy=POLICY
    )


def _assert_decomposition(result: PortfolioValuationResult) -> None:
    assert result.portfolio_equity - result.settled_cash == (
        result.pending_receivable_value + result.open_position_market_value
    )


def test_cash_only_state(make_state):
    result = _value(make_state(settled_cash=Decimal("2500.75")))

    assert result.settled_cash == Decimal("2500.75")
    assert result.pending_receivable_value == Decimal("0")
    assert result.open_position_market_value == Decimal("0")
    assert result.portfolio_equity == Decimal("2500.75")
    _assert_decomposition(result)


def test_cash_only_state_without_as_of_session(make_state):
    result = _value(
        make_state(settled_cash=Decimal("10"), as_of_session=None),
        session=None,
    )

    assert result.portfolio_equity == Decimal("10")


def test_pending_settlements_are_included_at_face_value(
    make_state, make_settlement
):
    result = _value(
        make_state(
            settled_cash=Decimal("100"),
            pending_settlements=(
                make_settlement(settlement_id="S1", amount=Decimal("500.25")),
                make_settlement(settlement_id="S2", amount=Decimal("0.75")),
            ),
        )
    )

    assert result.pending_receivable_value == Decimal("501.00")
    assert result.open_position_market_value == Decimal("0")
    assert result.portfolio_equity == Decimal("601.00")
    _assert_decomposition(result)


def test_open_position_market_value_uses_quantity_times_close(
    make_state, make_position, make_mark
):
    result = _value(
        make_state(
            settled_cash=Decimal("1000"),
            open_positions=(make_position(quantity=7),),
        ),
        marks=(make_mark(close=Decimal("101.25")),),
    )

    assert result.open_position_market_value == Decimal("708.75")
    assert result.portfolio_equity == Decimal("1708.75")
    _assert_decomposition(result)


def test_cost_basis_never_enters_market_value(
    make_state, make_position, make_mark
):
    cheap = _value(
        make_state(open_positions=(make_position(entry_price=Decimal("1")),)),
        marks=(make_mark(close=Decimal("100")),),
    )
    dear = _value(
        make_state(open_positions=(make_position(entry_price=Decimal("999")),)),
        marks=(make_mark(close=Decimal("100")),),
    )

    assert cheap.open_position_market_value == dear.open_position_market_value


def test_mixed_portfolio_with_multiple_positions(
    make_state, make_position, make_settlement, make_mark
):
    state = make_state(
        settled_cash=Decimal("2000"),
        open_positions=(
            make_position(asset_id=ASSET_A, quantity=3),
            make_position(asset_id=ASSET_B, quantity=5),
        ),
        pending_settlements=(make_settlement(amount=Decimal("250")),),
    )
    marks = (
        make_mark(security_id=ASSET_A, close=Decimal("10.50")),
        make_mark(security_id=ASSET_B, close=Decimal("20.10")),
    )

    result = _value(state, marks=marks)

    assert result.open_position_market_value == Decimal("132.00")
    assert result.pending_receivable_value == Decimal("250")
    assert result.portfolio_equity == Decimal("2382.00")
    _assert_decomposition(result)


def test_result_is_independent_of_supplied_mark_order(
    make_state, make_position, make_mark
):
    state = make_state(
        open_positions=(
            make_position(asset_id=ASSET_A, quantity=3),
            make_position(asset_id=ASSET_B, quantity=5),
        )
    )
    first = make_mark(security_id=ASSET_A, close=Decimal("10.50"))
    second = make_mark(security_id=ASSET_B, close=Decimal("20.10"))

    assert _value(state, marks=(first, second)) == _value(
        state, marks=(second, first)
    )


def test_dividend_scale_38_settled_cash_is_exact(make_state):
    settled = Decimal("10031.02658067566857563179740450795985919134")

    result = _value(make_state(settled_cash=settled))

    assert result.portfolio_equity == settled
    assert len(result.portfolio_equity.as_tuple().digits) > 28


def test_every_component_is_exact_across_ambient_precisions(
    make_state, make_position, make_settlement, make_mark
):
    state = make_state(
        settled_cash=Decimal("10031.02658067566857563179740450795985919134"),
        open_positions=(make_position(quantity=3),),
        pending_settlements=(
            make_settlement(amount=Decimal("0.0000000000000000001")),
        ),
    )
    marks = (
        make_mark(close=Decimal("33.333333333333333333333333333333333")),
    )

    results = []
    for precision in (28, 60, 200):
        with localcontext() as context:
            context.prec = precision
            results.append(_value(state, marks=marks))

    assert results[0] == results[1] == results[2]
    assert len(results[0].portfolio_equity.as_tuple().digits) > 28
    _assert_decomposition(results[0])


def test_naive_ambient_arithmetic_would_have_rounded(
    make_state, make_position, make_mark
):
    """The exactness above is load-bearing, not incidental."""

    state = make_state(
        settled_cash=Decimal("10031.02658067566857563179740450795985919134"),
        open_positions=(make_position(quantity=3),),
    )
    marks = (
        make_mark(close=Decimal("33.333333333333333333333333333333333")),
    )

    result = _value(state, marks=marks)
    with localcontext() as context:
        context.prec = 28
        naive = (
            state.settled_cash + marks[0].close * 3
        )

    assert result.portfolio_equity != naive


def test_component_totals_are_returned_separately(
    make_state, make_position, make_settlement, make_mark
):
    result = _value(
        make_state(
            settled_cash=Decimal("1"),
            open_positions=(make_position(quantity=2),),
            pending_settlements=(make_settlement(amount=Decimal("3")),),
        ),
        marks=(make_mark(close=Decimal("5")),),
    )

    assert (
        result.settled_cash,
        result.pending_receivable_value,
        result.open_position_market_value,
        result.portfolio_equity,
    ) == (Decimal("1"), Decimal("3"), Decimal("10"), Decimal("14"))


def test_result_is_runtime_only(make_state):
    result = _value(make_state())

    assert not hasattr(result, "schema_version")
    assert not hasattr(result, "fingerprint")
    with pytest.raises(AttributeError):
        result.portfolio_equity = Decimal("1")


def test_missing_mark_fails_closed(make_state, make_position):
    with pytest.raises(PortfolioValuationError) as error:
        _value(make_state(open_positions=(make_position(),)), marks=())

    assert error.value.code == "VALUATION_MARK_COVERAGE_MISMATCH"


def test_extra_mark_fails_closed(make_state, make_mark):
    with pytest.raises(PortfolioValuationError) as error:
        _value(make_state(), marks=(make_mark(),))

    assert error.value.code == "VALUATION_MARK_COVERAGE_MISMATCH"


def test_duplicate_mark_fails_closed(make_state, make_position, make_mark):
    with pytest.raises(PortfolioValuationError) as error:
        _value(
            make_state(open_positions=(make_position(),)),
            marks=(make_mark(), make_mark()),
        )

    assert error.value.code == "VALUATION_MARK_COVERAGE_MISMATCH"


def test_wrong_security_mark_fails_closed(
    make_state, make_position, make_mark
):
    with pytest.raises(PortfolioValuationError) as error:
        _value(
            make_state(open_positions=(make_position(asset_id=ASSET_A),)),
            marks=(make_mark(security_id=ASSET_B),),
        )

    assert error.value.code == "VALUATION_MARK_COVERAGE_MISMATCH"


def test_wrong_session_mark_fails_closed(
    make_state, make_position, make_mark
):
    with pytest.raises(PortfolioValuationError) as error:
        _value(
            make_state(open_positions=(make_position(),)),
            marks=(make_mark(session=date(2026, 8, 4)),),
        )

    assert error.value.code == "VALUATION_SESSION_MISMATCH"


def test_marks_without_a_session_fail_closed(make_state, make_mark):
    with pytest.raises(PortfolioValuationError) as error:
        value_portfolio_at_allocation_boundary(
            state=make_state(as_of_session=None),
            session=None,
            marks=(make_mark(),),
            policy=POLICY,
        )

    assert error.value.code in {
        "VALUATION_MARK_COVERAGE_MISMATCH",
        "VALUATION_SESSION_MISMATCH",
    }


def test_disagreeing_artifact_refs_fail_closed(
    make_state, make_position, make_mark
):
    other = ArtifactRef(
        artifact_type="market_data",
        schema_version="stock_bars_v0_1",
        content_sha256="c" * 64,
    )
    with pytest.raises(PortfolioValuationError) as error:
        _value(
            make_state(
                open_positions=(
                    make_position(asset_id=ASSET_A),
                    make_position(asset_id=ASSET_B),
                )
            ),
            marks=(
                make_mark(security_id=ASSET_A),
                make_mark(security_id=ASSET_B, source_artifact_ref=other),
            ),
        )

    assert error.value.code == "VALUATION_ARTIFACT_MISMATCH"


@pytest.mark.parametrize(
    "marks",
    [
        [],
        None,
        ("not-a-mark",),
    ],
)
def test_malformed_mark_container_fails_closed(make_state, marks):
    with pytest.raises(PortfolioValuationError) as error:
        _value(make_state(), marks=marks)

    assert error.value.code == "VALUATION_INPUT_INVALID"


def test_non_portfolio_state_fails_closed(make_mark):
    with pytest.raises(PortfolioValuationError) as error:
        _value("not-a-state")

    assert error.value.code == "VALUATION_INPUT_INVALID"


def test_a_non_frozen_policy_is_rejected(make_state):
    class Lookalike(PortfolioValuationPolicy):
        pass

    with pytest.raises(PortfolioValuationError) as error:
        value_portfolio_at_allocation_boundary(
            state=make_state(), session=SESSION, marks=(), policy=Lookalike()
        )

    assert error.value.code == "VALUATION_INPUT_INVALID"


# ---------------------------------------------------------------------------
# Task 5C defect 1: nested mark revalidation (forged model_copy instances)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
@pytest.mark.parametrize(
    "forged_update",
    [
        {"price_basis": "capital_special_adjusted"},
        {"session_type": "extended"},
        {"currency": "EUR"},
        {"timeframe": "H1"},
        {"close": Decimal("-1")},
        {"close": 100.0},
        {"session": "2026-08-03"},
        {"security_id": "AAPL"},
    ],
)
def test_forged_nested_mark_is_rejected_before_valuation(
    make_state, make_position, make_mark, forged_update
):
    """``model_copy(update=...)`` bypasses validation and keeps the right
    Python class; the boundary must revalidate every mark fully and fail
    closed before any equity is computed."""

    forged = make_mark().model_copy(update=forged_update)
    assert type(forged) is type(make_mark())

    with pytest.raises(PortfolioValuationError) as error:
        _value(make_state(open_positions=(make_position(),)), marks=(forged,))

    assert error.value.code == "VALUATION_INPUT_INVALID"


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_forged_nested_artifact_ref_inside_a_mark_is_rejected(
    make_state, make_position, make_mark, artifact_ref
):
    forged_ref = artifact_ref.model_copy(update={"content_sha256": "not-a-sha"})
    forged = make_mark().model_copy(update={"source_artifact_ref": forged_ref})

    with pytest.raises(PortfolioValuationError) as error:
        _value(make_state(open_positions=(make_position(),)), marks=(forged,))

    assert error.value.code == "VALUATION_INPUT_INVALID"


def test_a_lookalike_mark_subclass_is_rejected(make_state, make_position, make_mark):
    from stock_swing_d1.valuation import PortfolioValuationMark

    class Lookalike(PortfolioValuationMark):
        pass

    genuine = make_mark()
    lookalike = Lookalike(**genuine.model_dump(mode="python"))
    with pytest.raises(PortfolioValuationError) as error:
        _value(make_state(open_positions=(make_position(),)), marks=(lookalike,))

    assert error.value.code == "VALUATION_INPUT_INVALID"


# ---------------------------------------------------------------------------
# Task 5C defect 3: valuation session / state binding
# ---------------------------------------------------------------------------


def test_dated_state_valued_for_another_session_fails_closed(
    make_state, make_position, make_mark
):
    other = date(2026, 8, 4)
    with pytest.raises(PortfolioValuationError) as error:
        _value(
            make_state(open_positions=(make_position(),), as_of_session=SESSION),
            marks=(make_mark(session=other),),
            session=other,
        )

    assert error.value.code == "VALUATION_SESSION_MISMATCH"


def test_dated_state_with_session_none_fails_closed(make_state):
    with pytest.raises(PortfolioValuationError) as error:
        _value(make_state(as_of_session=SESSION), session=None)

    assert error.value.code == "VALUATION_SESSION_MISMATCH"


def test_cash_only_dated_state_is_not_exempt_from_session_binding(make_state):
    with pytest.raises(PortfolioValuationError) as error:
        _value(make_state(as_of_session=SESSION), session=date(2026, 8, 4))

    assert error.value.code == "VALUATION_SESSION_MISMATCH"


def test_virgin_state_valued_with_an_explicit_session_fails_closed(make_state):
    with pytest.raises(PortfolioValuationError) as error:
        _value(make_state(as_of_session=None), session=SESSION)

    assert error.value.code == "VALUATION_SESSION_MISMATCH"


def test_bound_session_equal_to_state_session_values_normally(
    make_state, make_position, make_mark
):
    result = _value(
        make_state(open_positions=(make_position(quantity=2),), as_of_session=SESSION),
        marks=(make_mark(close=Decimal("50")),),
        session=SESSION,
    )
    assert result.open_position_market_value == Decimal("100")


def test_frozen_policy_fields_are_exactly_the_approved_semantics():
    assert POLICY.model_dump(mode="python") == {
        "schema_version": "historical_backtest_valuation_policy.v0.1",
        "policy_id": "completed_session_unadjusted_close_mark_v0.1",
        "currency": "USD",
        "timeframe": "D1",
        "session_type": "regular",
        "price_basis": "unadjusted",
        "mark_field": "close",
        "pending_settlement_valuation": "face_value",
        "open_position_valuation": "quantity_times_close",
    }
