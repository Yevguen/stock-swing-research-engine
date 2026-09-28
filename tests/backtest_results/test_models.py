from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from stock_swing_d1.backtest_results import (
    ArtifactRef,
    DividendAttributionCompleteness,
    ExecutionApplicationStatus,
    ExecutionProvenanceSource,
    HistoricalBacktestAuditResult,
    HistoricalBacktestAuditSummary,
    HistoricalBacktestContentFingerprints,
    HistoricalBacktestCostSummary,
    HistoricalBacktestEntryRecord,
    HistoricalBacktestEquityRow,
    HistoricalBacktestExitReasonRow,
    HistoricalBacktestExitReasonSummary,
    HistoricalBacktestExitRecord,
    HistoricalCashLedgerRow,
    HistoricalClosedTradeRecord,
    HistoricalExitReason,
    HistoricalOpenTradeRecord,
    HistoricalRejectionStage,
    HistoricalSettlementRecord,
    HistoricalSettlementStatus,
    HistoricalTradeStatus,
    compute_result_fingerprint,
)
from stock_swing_d1.execution.entry.models import EntryExecutionStatus
from stock_swing_d1.execution.open_position_exit.models import OpenPositionExitReason
from stock_swing_d1.portfolio.portfolio_events import PortfolioLedgerEventType
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.backtester import HistoricalDecisionInterval


def _entry(**overrides):
    values = dict(
        entry_execution_id="entry-1",
        source_order_id="order-1",
        security_id="NORGATE:1",
        quantity=10,
        entry_session=date(2026, 8, 20),
        fill_price=Decimal("100"),
        execution_cost=Decimal("2"),
        cost_basis=Decimal("1002"),
        provenance_source=ExecutionProvenanceSource.GENERATED_ENTRY,
        signal_session=date(2026, 8, 19),
        signal_time=datetime(2026, 8, 19, 20, tzinfo=timezone.utc),
        allocation_session=date(2026, 8, 19),
        source_rank=1,
        ranking_snapshot_fingerprint="b" * 64,
        ranking_input_fingerprint="c" * 64,
        entry_execution_status=EntryExecutionStatus.EXECUTED,
        execution_cost_policy_fingerprint="a" * 64,
    )
    values.update(overrides)
    return HistoricalBacktestEntryRecord(**values)


def _exit(**overrides):
    values = dict(
        exit_execution_id="exit-1",
        source_order_id="order-2",
        entry_execution_id="entry-1",
        security_id="NORGATE:1",
        quantity=10,
        exit_session=date(2026, 8, 24),
        fill_price=Decimal("110"),
        execution_cost=Decimal("2"),
        gross_proceeds=Decimal("1100"),
        net_proceeds=Decimal("1098"),
        settlement_id="settlement-1",
        settlement_session=date(2026, 8, 25),
        provenance_source=ExecutionProvenanceSource.GENERATED_EXIT,
        exit_reason=HistoricalExitReason.TAKE_PROFIT,
        reference_exit_price=Decimal("110"),
    )
    values.update(overrides)
    return HistoricalBacktestExitRecord(**values)


def _closed_trade(**overrides):
    values = dict(
        trade_id="entry-1",
        security_id="NORGATE:1",
        quantity=10,
        carried_in=False,
        entry_execution_id="entry-1",
        entry_session=date(2026, 8, 20),
        entry_fill_price=Decimal("100"),
        entry_execution_cost=Decimal("2"),
        entry_cost_basis=Decimal("1002"),
        exit_execution_id="exit-1",
        exit_session=date(2026, 8, 24),
        exit_fill_price=Decimal("110"),
        exit_execution_cost=Decimal("2"),
        exit_reason=HistoricalExitReason.TAKE_PROFIT,
        gross_exit_proceeds=Decimal("1100"),
        net_exit_proceeds=Decimal("1098"),
        realized_pnl=Decimal("96"),
        ordinary_dividend_income=Decimal("0"),
        ordinary_dividend_event_count=0,
        dividend_attribution_completeness=(
            DividendAttributionCompleteness.COMPLETE_TRADE_LIFETIME
        ),
        ordinary_dividend_attribution_fingerprint="0" * 64,
        # Slice 11 (OD-16.5a): COMPLETE_TRADE_LIFETIME requires
        # trade_total_pnl == realized_pnl + ordinary_dividend_income.
        trade_total_pnl=Decimal("96"),
    )
    values.update(overrides)
    return HistoricalClosedTradeRecord(**values)


def _open_trade(**overrides):
    values = dict(
        trade_id="entry-1",
        security_id="NORGATE:1",
        quantity=10,
        carried_in=False,
        entry_execution_id="entry-1",
        entry_session=date(2026, 8, 20),
        entry_fill_price=Decimal("100"),
        entry_execution_cost=Decimal("2"),
        entry_cost_basis=Decimal("1002"),
        final_mark_session=date(2026, 8, 24),
        final_mark_price=Decimal("105"),
        final_market_value=Decimal("1050"),
        unrealized_pnl=Decimal("48"),
    )
    values.update(overrides)
    return HistoricalOpenTradeRecord(**values)


def test_reporting_enums_have_exact_stable_values():
    assert [reason.value for reason in HistoricalExitReason] == [
        "GAP_THROUGH_STOP",
        "GAP_THROUGH_TARGET",
        "STOP_LOSS",
        "TAKE_PROFIT",
        "EARNINGS_FORCED_EXIT",
        "MAX_HOLDING",
        "EXTERNAL_SCHEDULED",
    ]
    assert [reason.value for reason in OpenPositionExitReason] == [
        reason.value for reason in HistoricalExitReason
    ][:-1]
    assert [item.value for item in ExecutionApplicationStatus] == ["APPLIED", "REPLAYED"]
    assert [item.value for item in HistoricalTradeStatus] == ["CLOSED", "OPEN_AT_RUN_END"]
    assert [item.value for item in HistoricalRejectionStage] == ["ALLOCATION", "ENTRY_EXECUTION"]


@pytest.mark.parametrize("field", ["cost_basis", "fill_price", "execution_cost"])
def test_entry_requires_strict_decimal_money(field):
    for invalid in (100, 100.0, "100", True, Decimal("NaN"), Decimal("Infinity")):
        with pytest.raises(ValidationError):
            _entry(**{field: invalid})


def test_entry_cost_basis_identity_is_exact():
    entry = _entry()
    assert entry.cost_basis == Decimal("1002")
    assert entry.entry_execution_status is EntryExecutionStatus.EXECUTED
    with pytest.raises(ValidationError):
        _entry(cost_basis=Decimal("1001.999"))


@pytest.mark.parametrize(
    "status",
    [None, EntryExecutionStatus.CANCELLED_INSUFFICIENT_CASH_AT_EXECUTION],
)
def test_generated_entry_requires_executed_status(status):
    with pytest.raises(ValidationError):
        _entry(entry_execution_status=status)


def test_generated_entry_requires_execution_cost_policy_provenance():
    with pytest.raises(ValidationError):
        _entry(execution_cost_policy_fingerprint=None)


def test_external_scheduled_entry_accepts_genuine_provenance_absence():
    entry = _entry(
        provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
        signal_session=None,
        signal_time=None,
        allocation_session=None,
        source_rank=None,
        ranking_snapshot_fingerprint=None,
        ranking_input_fingerprint=None,
        entry_execution_status=None,
        execution_cost_policy_fingerprint=None,
    )
    assert entry.cost_basis == Decimal("1002")
    with pytest.raises(ValidationError):
        _entry(
            provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
            signal_session=None,
            signal_time=None,
            allocation_session=None,
            source_rank=None,
            ranking_snapshot_fingerprint=None,
            ranking_input_fingerprint=None,
            entry_execution_status=None,
            execution_cost_policy_fingerprint=None,
            cost_basis=Decimal("1001"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entry_execution_status", EntryExecutionStatus.EXECUTED),
        ("execution_cost_policy_fingerprint", "a" * 64),
        ("signal_session", date(2026, 8, 19)),
        ("ranking_snapshot_fingerprint", "b" * 64),
    ],
)
def test_external_scheduled_entry_rejects_generated_provenance(field, value):
    external_values = dict(
        provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
        signal_session=None,
        signal_time=None,
        allocation_session=None,
        source_rank=None,
        ranking_snapshot_fingerprint=None,
        ranking_input_fingerprint=None,
        entry_execution_status=None,
        execution_cost_policy_fingerprint=None,
    )
    external_values[field] = value
    with pytest.raises(ValidationError):
        _entry(**external_values)


def test_entry_rejects_generated_exit_provenance():
    with pytest.raises(ValidationError):
        _entry(provenance_source=ExecutionProvenanceSource.GENERATED_EXIT)


def test_exit_gross_and_net_identities_are_exact():
    exit_record = _exit()
    assert exit_record.net_proceeds == Decimal("1098")
    assert exit_record.reference_exit_price == Decimal("110")
    with pytest.raises(ValidationError):
        _exit(gross_proceeds=Decimal("1099"))
    with pytest.raises(ValidationError):
        _exit(net_proceeds=Decimal("1099"))


def test_generated_exit_requires_reference_price_and_phase15b_reason():
    with pytest.raises(ValidationError):
        _exit(reference_exit_price=None)
    with pytest.raises(ValidationError):
        _exit(exit_reason=HistoricalExitReason.EXTERNAL_SCHEDULED)


def test_external_scheduled_exit_accepts_structural_provenance_and_identities():
    exit_record = _exit(
        provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
        exit_reason=HistoricalExitReason.EXTERNAL_SCHEDULED,
        reference_exit_price=None,
    )
    assert exit_record.gross_proceeds == Decimal("1100")
    assert exit_record.net_proceeds == Decimal("1098")


def test_external_scheduled_exit_rejects_phase15b_reason_and_reference_price():
    with pytest.raises(ValidationError):
        _exit(
            provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
            exit_reason=HistoricalExitReason.TAKE_PROFIT,
            reference_exit_price=None,
        )
    with pytest.raises(ValidationError):
        _exit(
            provenance_source=ExecutionProvenanceSource.EXTERNAL_SCHEDULED,
            exit_reason=HistoricalExitReason.EXTERNAL_SCHEDULED,
            reference_exit_price=Decimal("110"),
        )


def test_exit_rejects_generated_entry_provenance():
    with pytest.raises(ValidationError):
        _exit(provenance_source=ExecutionProvenanceSource.GENERATED_ENTRY)


@pytest.mark.parametrize(
    "field", ["fill_price", "execution_cost", "gross_proceeds", "net_proceeds"]
)
def test_exit_requires_strict_decimal_money(field):
    for invalid in (1, 1.0, "1", True, Decimal("NaN"), Decimal("Infinity")):
        with pytest.raises(ValidationError):
            _exit(**{field: invalid})


def test_closed_and_open_trade_identities_are_exact():
    assert _closed_trade().realized_pnl == Decimal("96")
    with pytest.raises(ValidationError):
        _closed_trade(realized_pnl=Decimal("95"))
    assert _open_trade().unrealized_pnl == Decimal("48")
    with pytest.raises(ValidationError):
        _open_trade(final_market_value=Decimal("1049"))
    with pytest.raises(ValidationError):
        _open_trade(unrealized_pnl=Decimal("49"))


@pytest.mark.parametrize(
    ("factory", "field"),
    [(_closed_trade, "realized_pnl"), (_open_trade, "unrealized_pnl")],
)
def test_trade_pnl_requires_strict_decimal(factory, field):
    for invalid in (1, 1.0, "1", True, Decimal("NaN"), Decimal("Infinity")):
        with pytest.raises(ValidationError):
            factory(**{field: invalid})


def test_equity_identity_and_strict_money():
    values = dict(
        session=date(2026, 8, 24),
        state_hash="a" * 64,
        settled_cash=Decimal("1000"),
        pending_receivable_value=Decimal("200"),
        open_position_market_value=Decimal("500"),
        equity=Decimal("1700"),
        realized_pnl_this_session=Decimal("0"),
        cumulative_realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        execution_cost_this_session=Decimal("0"),
        cumulative_execution_cost=Decimal("0"),
        period_pnl=Decimal("0"),
    )
    assert HistoricalBacktestEquityRow(**values).equity == Decimal("1700")
    with pytest.raises(ValidationError):
        HistoricalBacktestEquityRow(**{**values, "equity": Decimal("1699")})
    with pytest.raises(ValidationError):
        HistoricalBacktestEquityRow(**{**values, "settled_cash": 1000.0})
    with pytest.raises(ValidationError):
        HistoricalBacktestEquityRow(**{**values, "equity": 1700.0})


def test_cash_ledger_settlement_and_cost_summary_money_are_strict():
    ledger = dict(
        session=date(2026, 8, 24),
        sequence_in_session=0,
        ledger_event_type=PortfolioLedgerEventType.BUY_APPLIED,
        source_event_id="event-1",
        settled_cash_delta=Decimal("-10"),
        pending_cash_delta=Decimal("0"),
        settled_cash_after=Decimal("990"),
        state_hash_before="a" * 64,
        state_hash_after="b" * 64,
        source_payload_fingerprint="c" * 64,
    )
    HistoricalCashLedgerRow(**ledger)
    with pytest.raises(ValidationError):
        HistoricalCashLedgerRow(**{**ledger, "settled_cash_delta": -10.0})

    settlement = dict(
        settlement_id="settlement-1",
        source_sell_execution_id="exit-1",
        security_id="NORGATE:1",
        trade_session=date(2026, 8, 24),
        settlement_session=date(2026, 8, 25),
        amount=Decimal("100"),
        status=HistoricalSettlementStatus.SETTLED_DURING_RUN,
    )
    HistoricalSettlementRecord(**settlement)
    with pytest.raises(ValidationError):
        HistoricalSettlementRecord(**{**settlement, "amount": "100"})
    with pytest.raises(ValidationError):
        HistoricalSettlementRecord(
            **{**settlement, "settlement_session": date(2026, 8, 24)}
        )

    summary = dict(
        buy_execution_cost_total=Decimal("1"),
        sell_execution_cost_total=Decimal("2"),
        total_execution_cost=Decimal("3"),
        applied_buy_count=1,
        applied_sell_count=1,
    )
    HistoricalBacktestCostSummary(**summary)
    with pytest.raises(ValidationError):
        HistoricalBacktestCostSummary(**{**summary, "total_execution_cost": 3.0})
    with pytest.raises(ValidationError):
        HistoricalBacktestCostSummary(
            **{**summary, "total_execution_cost": Decimal("4")}
        )


def test_exit_reason_summary_requires_all_rows_in_frozen_order():
    rows = tuple(
        HistoricalBacktestExitReasonRow(
            reason=reason, exit_count=0, realized_pnl=Decimal("0")
        )
        for reason in HistoricalExitReason
    )
    assert len(HistoricalBacktestExitReasonSummary(rows=rows).rows) == 7
    for invalid in (rows[:-1], tuple(reversed(rows)), (rows[0],) + rows[:-1]):
        with pytest.raises(ValidationError):
            HistoricalBacktestExitReasonSummary(rows=invalid)


def test_audit_summary_true_requires_every_flag_true(passing_audit_summary):
    values = passing_audit_summary.model_dump(mode="python")
    values["state_chain_valid"] = False
    with pytest.raises(ValidationError):
        HistoricalBacktestAuditSummary.model_validate(values)
    values["audit_passed"] = False
    assert not HistoricalBacktestAuditSummary.model_validate(values).audit_passed


def test_artifact_ref_validation_frozen_and_no_path():
    valid = ArtifactRef(
        artifact_type="market_data",
        schema_version="bars.v0.1",
        content_sha256="a" * 64,
        build_id="build-1",
    )
    assert "path" not in type(valid).model_fields
    with pytest.raises(ValidationError):
        valid.artifact_type = "other"
    with pytest.raises(ValidationError):
        ArtifactRef(
            artifact_type="market_data",
            schema_version="bars.v0.1",
            content_sha256="a" * 64,
            path="x",
        )
    for invalid in ("a" * 63, "A" * 64, " " + "a" * 63):
        with pytest.raises(ValidationError):
            ArtifactRef(
                artifact_type="market_data",
                schema_version="bars.v0.1",
                content_sha256=invalid,
            )
    for field in ("artifact_type", "schema_version", "build_id"):
        values = dict(
            artifact_type="market_data",
            schema_version="bars.v0.1",
            content_sha256="a" * 64,
            build_id="build-1",
        )
        values[field] = ""
        with pytest.raises(ValidationError):
            ArtifactRef(**values)


def test_minimal_top_level_result_is_frozen_and_structurally_valid(
    run_manifest,
    zero_cost_summary,
    zero_exit_reason_summary,
    zero_summary,
    passing_audit_summary,
    content_fingerprints,
):
    state = PortfolioState(settled_cash=Decimal("1000"))
    interval = HistoricalDecisionInterval(
        decision_start_date=date(2024, 2, 29),
        decision_end_date=date(2025, 1, 9),
    )
    components = dict(
        schema_version="historical_backtest_audit_result.v0.2",
        decision_interval=interval,
        run_configuration_fingerprint=run_manifest.run_configuration_fingerprint,
        source_run_fingerprint="a" * 64,
        initial_state_fingerprint="b" * 64,
        final_state_fingerprint="c" * 64,
        content_fingerprints=content_fingerprints,
    )
    values = dict(
        run_manifest=run_manifest,
        source_run_fingerprint=components["source_run_fingerprint"],
        decision_interval=interval,
        initial_state=state,
        final_state=state,
        initial_state_fingerprint=components["initial_state_fingerprint"],
        final_state_fingerprint=components["final_state_fingerprint"],
        initial_equity=Decimal("1000"),
        final_equity=Decimal("1000"),
        period_pnl=Decimal("0"),
        cost_summary=zero_cost_summary,
        exit_reason_summary=zero_exit_reason_summary,
        summary=zero_summary,
        audit_summary=passing_audit_summary,
        content_fingerprints=content_fingerprints,
        result_fingerprint=compute_result_fingerprint(**components),
    )
    result = HistoricalBacktestAuditResult(**values)
    assert result.decision_interval == interval
    assert result.session_transitions == ()
    with pytest.raises(ValidationError):
        result.final_equity = Decimal("2")
    with pytest.raises(ValidationError):
        HistoricalBacktestAuditResult(**values, unexpected=True)
    failed_audit = passing_audit_summary.model_copy(update={"audit_passed": False})
    with pytest.raises(ValidationError):
        HistoricalBacktestAuditResult(**{**values, "audit_summary": failed_audit})
    with pytest.raises(ValidationError):
        HistoricalBacktestAuditResult(
            **{**values, "source_run_fingerprint": "bad"}
        )
    without_interval = dict(values)
    without_interval.pop("decision_interval")
    with pytest.raises(ValidationError, match="decision_interval"):
        HistoricalBacktestAuditResult(**without_interval)


def test_all_public_models_use_frozen_forbid_configuration():
    from stock_swing_d1.backtest_results import models

    public_models = [
        value
        for name, value in vars(models).items()
        if name.startswith(("ArtifactRef", "PolicyArtifactRef", "Historical"))
        and isinstance(value, type)
        and issubclass(value, __import__("pydantic").BaseModel)
    ]
    assert public_models
    for model in public_models:
        assert model.model_config["frozen"] is True
        assert model.model_config["extra"] == "forbid"
        assert model.model_config["arbitrary_types_allowed"] is False
