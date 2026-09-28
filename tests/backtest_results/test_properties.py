from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from hypothesis import given, strategies as st

from stock_swing_d1.backtest_results import (
    ExecutionProvenanceSource,
    HistoricalBacktestEntryRecord,
    HistoricalBacktestEquityRow,
    HistoricalBacktestExitRecord,
    HistoricalExitReason,
    semantic_json_bytes,
    semantic_sha256,
)
from stock_swing_d1.execution.entry.models import EntryExecutionStatus


positive_money = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("10000"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)
nonnegative_money = st.decimals(
    min_value=Decimal("0"),
    max_value=Decimal("1000"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)
quantities = st.integers(min_value=1, max_value=1000)


@given(
    st.decimals(
        min_value=Decimal("-100000"),
        max_value=Decimal("100000"),
        allow_nan=False,
        allow_infinity=False,
        places=4,
    )
)
def test_decimal_trailing_zero_representations_are_semantically_equal(value):
    expanded = value.quantize(Decimal("0.00000"))
    assert expanded == value
    assert semantic_json_bytes(expanded) == semantic_json_bytes(value)
    assert semantic_sha256(expanded) == semantic_sha256(value)


@given(quantity=quantities, fill=positive_money, cost=nonnegative_money)
def test_entry_identity_property(quantity, fill, cost):
    expected = Decimal(quantity) * fill + cost
    row = HistoricalBacktestEntryRecord(
        entry_execution_id="entry",
        source_order_id="order",
        security_id="NORGATE:1",
        quantity=quantity,
        entry_session=date(2026, 8, 24),
        fill_price=fill,
        execution_cost=cost,
        cost_basis=expected,
        provenance_source=ExecutionProvenanceSource.GENERATED_ENTRY,
        signal_session=date(2026, 8, 23),
        signal_time=datetime(2026, 8, 23, 20, tzinfo=timezone.utc),
        allocation_session=date(2026, 8, 23),
        source_rank=1,
        ranking_snapshot_fingerprint="b" * 64,
        ranking_input_fingerprint="c" * 64,
        entry_execution_status=EntryExecutionStatus.EXECUTED,
        execution_cost_policy_fingerprint="a" * 64,
    )
    assert row.cost_basis == expected


@given(quantity=quantities, fill=positive_money, cost=nonnegative_money)
def test_exit_identity_property(quantity, fill, cost):
    gross = Decimal(quantity) * fill
    net = gross - cost
    row = HistoricalBacktestExitRecord(
        exit_execution_id="exit",
        source_order_id="order",
        entry_execution_id="entry",
        security_id="NORGATE:1",
        quantity=quantity,
        exit_session=date(2026, 8, 24),
        fill_price=fill,
        execution_cost=cost,
        gross_proceeds=gross,
        net_proceeds=net,
        settlement_id="settlement",
        settlement_session=date(2026, 8, 25),
        provenance_source=ExecutionProvenanceSource.GENERATED_EXIT,
        exit_reason=HistoricalExitReason.TAKE_PROFIT,
        reference_exit_price=fill,
    )
    assert row.gross_proceeds == gross
    assert row.net_proceeds == net


@given(cash=nonnegative_money, pending=nonnegative_money, market=nonnegative_money)
def test_equity_identity_property(cash, pending, market):
    equity = cash + pending + market
    row = HistoricalBacktestEquityRow(
        session=date(2026, 8, 24),
        state_hash="a" * 64,
        settled_cash=cash,
        pending_receivable_value=pending,
        open_position_market_value=market,
        equity=equity,
        realized_pnl_this_session=Decimal("0"),
        cumulative_realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        execution_cost_this_session=Decimal("0"),
        cumulative_execution_cost=Decimal("0"),
        period_pnl=Decimal("0"),
    )
    assert row.equity == equity


@given(st.lists(st.integers(), max_size=30))
def test_identical_canonical_values_have_identical_hash(values):
    assert semantic_sha256(values) == semantic_sha256(list(values))
