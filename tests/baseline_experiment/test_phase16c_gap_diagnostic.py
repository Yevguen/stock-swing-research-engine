"""Frozen Material Adverse Overnight Gap / Gap-Loss Frequency v0.1 semantics.

Every fixture here is synthetic. The canonical historical dataset is not
required to establish any of these properties.
"""

from __future__ import annotations

import ast
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.baseline_experiment import (
    GAP_CORPORATE_ACTION_NORMALIZATION,
    GAP_DIRECTION,
    GAP_MATERIALITY_THRESHOLD,
    GAP_MATERIALITY_THRESHOLD_TEXT,
    GAP_METRIC_VERSION,
    GAP_ORDINARY_DIVIDEND_NORMALIZATION,
    GAP_PRICE_BASIS,
    GAP_REPORT_LABEL,
    GAP_TRADE_COUNTING_RULE,
    BaselineExperimentDiagnosticError,
    CompletedTradeGapEvidence,
    GapDiagnosticUndefinedReason,
    OvernightBoundaryEvidence,
    OvernightBoundaryEvidenceStatus,
    build_material_adverse_overnight_gap_diagnostic,
    build_material_adverse_overnight_gap_policy,
)

from tests.baseline_experiment.conftest import (
    SYNTHETIC_OTHER_SECURITY_ID,
    SYNTHETIC_SECURITY_ID,
    build_synthetic_audit_result,
    make_closed_trade,
)


ENTRY = date(2025, 1, 6)
EXIT = date(2025, 1, 10)
BEFORE_ENTRY = date(2025, 1, 3)
AFTER_EXIT = date(2025, 1, 13)


def boundary(
    previous_session: date,
    current_session: date,
    *,
    close: str | None = "100",
    open_price: str | None = "100",
    new_shares: int | None = None,
    old_shares: int | None = None,
    dividend: str | None = None,
    security_id: str = SYNTHETIC_SECURITY_ID,
    status: OvernightBoundaryEvidenceStatus = (
        OvernightBoundaryEvidenceStatus.COMPLETE
    ),
    missing: str | None = None,
) -> OvernightBoundaryEvidence:
    return OvernightBoundaryEvidence(
        security_id=security_id,
        previous_session=previous_session,
        current_session=current_session,
        evidence_status=status,
        previous_regular_session_close=(
            None if close is None else Decimal(close)
        ),
        current_regular_session_open=(
            None if open_price is None else Decimal(open_price)
        ),
        share_basis_new_shares=new_shares,
        share_basis_old_shares=old_shares,
        ordinary_dividend_amount_per_share=(
            None if dividend is None else Decimal(dividend)
        ),
        missing_evidence_code=missing,
    )


def diagnose(*trade_specs):
    """Build one diagnostic over the supplied (trade, boundaries) pairs."""

    trades = []
    evidence = []
    for index, (entry_session, exit_session, boundaries) in enumerate(
        trade_specs, start=1
    ):
        trade_id = f"SYNTH-TRADE-{index}"
        trades.append(
            make_closed_trade(
                trade_id=trade_id,
                entry_session=entry_session,
                exit_session=exit_session,
            )
        )
        evidence.append(
            CompletedTradeGapEvidence(
                trade_id=trade_id,
                security_id=SYNTHETIC_SECURITY_ID,
                boundaries=tuple(boundaries),
            )
        )
    source_result = build_synthetic_audit_result(trades=tuple(trades))
    return build_material_adverse_overnight_gap_diagnostic(
        source_result=source_result,
        policy=build_material_adverse_overnight_gap_policy(),
        trade_gap_evidence=tuple(evidence),
    )


def test_frozen_policy_preserves_every_manifest_parameter(gap_policy):
    assert gap_policy.gap_metric_version == "material_adverse_overnight_gap.v0.1"
    assert gap_policy.report_label == "Gap-loss frequency"
    assert gap_policy.direction == "long_adverse"
    assert gap_policy.materiality_threshold == "-0.010000"
    assert gap_policy.price_basis == "canonical_unadjusted_tradable"
    assert gap_policy.corporate_action_normalization == "frozen_canonical"
    assert gap_policy.ordinary_dividend_normalization == "frozen_canonical"
    assert gap_policy.trade_counting_rule == "one_per_trade"
    assert GAP_METRIC_VERSION == "material_adverse_overnight_gap.v0.1"
    assert GAP_REPORT_LABEL == "Gap-loss frequency"
    assert GAP_DIRECTION == "long_adverse"
    assert GAP_MATERIALITY_THRESHOLD_TEXT == "-0.010000"
    assert GAP_MATERIALITY_THRESHOLD == Decimal("-0.010000")
    assert GAP_PRICE_BASIS == "canonical_unadjusted_tradable"
    assert GAP_CORPORATE_ACTION_NORMALIZATION == "frozen_canonical"
    assert GAP_ORDINARY_DIVIDEND_NORMALIZATION == "frozen_canonical"
    assert GAP_TRADE_COUNTING_RULE == "one_per_trade"


def test_frozen_policy_fingerprint_is_stable_and_content_bound(gap_policy):
    assert gap_policy.policy_fingerprint == (
        build_material_adverse_overnight_gap_policy().policy_fingerprint
    )
    with pytest.raises(ValueError):
        type(gap_policy)(
            **{
                **{
                    name: getattr(gap_policy, name)
                    for name in type(gap_policy).model_fields
                },
                "policy_fingerprint": "0" * 64,
            }
        )


def test_ordinary_non_gap_does_not_qualify():
    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [boundary(ENTRY, date(2025, 1, 7), close="100", open_price="100")],
        )
    )
    assert diagnostic.overnight_completed_trade_count == 1
    assert diagnostic.gap_loss_trade_count == 0
    assert diagnostic.qualifying_gap_event_count == 0
    assert diagnostic.value == Decimal("0")
    assert diagnostic.undefined_reason is None
    assert diagnostic.trade_observations[0].most_adverse_gap_return == (
        Decimal("0")
    )


def test_exactly_one_percent_adverse_gap_qualifies():
    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [boundary(ENTRY, date(2025, 1, 7), close="100", open_price="99")],
        )
    )
    observation = diagnostic.trade_observations[0]
    assert observation.most_adverse_gap_return == Decimal("-0.010000")
    assert observation.counted_in_numerator is True
    assert diagnostic.gap_loss_trade_count == 1
    assert diagnostic.value == Decimal("1")


def test_nine_thousand_nine_hundred_ninety_nine_ten_thousandths_does_not_qualify():
    """A gap of -0.9999% is immediately outside the frozen threshold."""

    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [
                boundary(
                    ENTRY,
                    date(2025, 1, 7),
                    close="10000",
                    open_price="9900.01",
                )
            ],
        )
    )
    observation = diagnostic.trade_observations[0]
    assert observation.most_adverse_gap_return == Decimal("-0.009999")
    assert observation.most_adverse_gap_return > GAP_MATERIALITY_THRESHOLD
    assert observation.counted_in_numerator is False
    assert diagnostic.gap_loss_trade_count == 0
    assert diagnostic.overnight_completed_trade_count == 1
    assert diagnostic.value == Decimal("0")


def test_multiple_adverse_gaps_in_one_trade_count_once():
    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [
                boundary(ENTRY, date(2025, 1, 7), close="100", open_price="98"),
                boundary(
                    date(2025, 1, 7),
                    date(2025, 1, 8),
                    close="100",
                    open_price="97",
                ),
                boundary(
                    date(2025, 1, 8),
                    date(2025, 1, 9),
                    close="100",
                    open_price="101",
                ),
            ],
        )
    )
    observation = diagnostic.trade_observations[0]
    assert observation.eligible_boundary_count == 3
    assert observation.qualifying_gap_event_count == 2
    assert diagnostic.gap_loss_trade_count == 1
    assert diagnostic.qualifying_gap_event_count == 2
    assert diagnostic.value == Decimal("1")


def test_adverse_gaps_across_two_trades_count_twice():
    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [boundary(ENTRY, date(2025, 1, 7), close="100", open_price="95")],
        ),
        (
            date(2025, 2, 3),
            date(2025, 2, 7),
            [
                boundary(
                    date(2025, 2, 3),
                    date(2025, 2, 4),
                    close="100",
                    open_price="96",
                )
            ],
        ),
    )
    assert diagnostic.gap_loss_trade_count == 2
    assert diagnostic.overnight_completed_trade_count == 2
    assert diagnostic.value == Decimal("1")


def test_pre_entry_gap_is_excluded():
    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [
                boundary(
                    BEFORE_ENTRY, ENTRY, close="100", open_price="50"
                )
            ],
        )
    )
    observation = diagnostic.trade_observations[0]
    assert observation.eligible_boundary_count == 0
    assert observation.ineligible_boundary_count == 1
    assert diagnostic.overnight_completed_trade_count == 0
    assert diagnostic.value is None
    assert diagnostic.undefined_reason is (
        GapDiagnosticUndefinedReason.NO_OVERNIGHT_COMPLETED_TRADES
    )


def test_post_exit_gap_is_excluded():
    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [
                boundary(ENTRY, date(2025, 1, 7), close="100", open_price="100"),
                boundary(EXIT, AFTER_EXIT, close="100", open_price="50"),
            ],
        )
    )
    observation = diagnostic.trade_observations[0]
    assert observation.eligible_boundary_count == 1
    assert observation.ineligible_boundary_count == 1
    assert observation.qualifying_gap_event_count == 0
    assert diagnostic.gap_loss_trade_count == 0


def test_exit_session_opening_gap_is_included():
    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [boundary(date(2025, 1, 9), EXIT, close="100", open_price="97")],
        )
    )
    observation = diagnostic.trade_observations[0]
    assert observation.eligible_boundary_count == 1
    assert observation.qualifying_gap_event_count == 1
    assert diagnostic.gap_loss_trade_count == 1


def test_split_normalization_prevents_a_false_gap():
    """A 4-for-1 split must not appear as a roughly -75% adverse gap."""

    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [
                boundary(
                    ENTRY,
                    date(2025, 1, 7),
                    close="400",
                    open_price="100",
                    new_shares=4,
                    old_shares=1,
                )
            ],
        )
    )
    observation = diagnostic.trade_observations[0]
    assert observation.most_adverse_gap_return == Decimal("0")
    assert diagnostic.gap_loss_trade_count == 0


def test_reverse_split_normalization_prevents_a_false_gap():
    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [
                boundary(
                    ENTRY,
                    date(2025, 1, 7),
                    close="10",
                    open_price="50",
                    new_shares=1,
                    old_shares=5,
                )
            ],
        )
    )
    observation = diagnostic.trade_observations[0]
    assert observation.most_adverse_gap_return == Decimal("0")
    assert diagnostic.gap_loss_trade_count == 0


def test_ordinary_ex_dividend_normalization_prevents_a_false_gap():
    """A -1.00% ex-dividend price move is compensated, not an adverse gap."""

    uncompensated = diagnose(
        (
            ENTRY,
            EXIT,
            [boundary(ENTRY, date(2025, 1, 7), close="100", open_price="99")],
        )
    )
    assert uncompensated.gap_loss_trade_count == 1

    compensated = diagnose(
        (
            ENTRY,
            EXIT,
            [
                boundary(
                    ENTRY,
                    date(2025, 1, 7),
                    close="100",
                    open_price="99",
                    dividend="1",
                )
            ],
        )
    )
    observation = compensated.trade_observations[0]
    assert observation.most_adverse_gap_return == Decimal("0")
    assert compensated.gap_loss_trade_count == 0


def test_no_overnight_completed_trades_is_explicitly_undefined():
    diagnostic = diagnose((ENTRY, EXIT, []))
    assert diagnostic.value is None
    assert diagnostic.undefined_reason is (
        GapDiagnosticUndefinedReason.NO_OVERNIGHT_COMPLETED_TRADES
    )
    assert diagnostic.overnight_completed_trade_count == 0
    assert diagnostic.gap_loss_trade_count == 0


def test_incomplete_eligible_evidence_is_explicitly_undefined():
    diagnostic = diagnose(
        (
            ENTRY,
            EXIT,
            [
                boundary(
                    ENTRY,
                    date(2025, 1, 7),
                    close=None,
                    open_price="100",
                    status=OvernightBoundaryEvidenceStatus.INCOMPLETE,
                    missing="PREVIOUS_REGULAR_SESSION_CLOSE_UNAVAILABLE",
                )
            ],
        )
    )
    assert diagnostic.value is None
    assert diagnostic.undefined_reason is (
        GapDiagnosticUndefinedReason.INCOMPLETE_GAP_EVIDENCE
    )
    assert diagnostic.overnight_completed_trade_count is None
    assert diagnostic.gap_loss_trade_count is None
    assert diagnostic.trade_observations == ()


def test_a_completed_trade_without_evidence_is_never_silently_omitted():
    trade = make_closed_trade(
        trade_id="SYNTH-TRADE-1", entry_session=ENTRY, exit_session=EXIT
    )
    source_result = build_synthetic_audit_result(trades=(trade,))
    diagnostic = build_material_adverse_overnight_gap_diagnostic(
        source_result=source_result,
        policy=build_material_adverse_overnight_gap_policy(),
        trade_gap_evidence=(),
    )
    assert diagnostic.completed_trade_count == 1
    assert diagnostic.undefined_reason is (
        GapDiagnosticUndefinedReason.INCOMPLETE_GAP_EVIDENCE
    )


def test_evidence_for_an_unknown_trade_fails_closed():
    trade = make_closed_trade(
        trade_id="SYNTH-TRADE-1", entry_session=ENTRY, exit_session=EXIT
    )
    source_result = build_synthetic_audit_result(trades=(trade,))
    with pytest.raises(BaselineExperimentDiagnosticError) as error:
        build_material_adverse_overnight_gap_diagnostic(
            source_result=source_result,
            policy=build_material_adverse_overnight_gap_policy(),
            trade_gap_evidence=(
                CompletedTradeGapEvidence(
                    trade_id="SYNTH-TRADE-1",
                    security_id=SYNTHETIC_SECURITY_ID,
                    boundaries=(),
                ),
                CompletedTradeGapEvidence(
                    trade_id="NOT-A-TRADE",
                    security_id=SYNTHETIC_SECURITY_ID,
                    boundaries=(),
                ),
            ),
        )
    assert error.value.code == "UNKNOWN_TRADE_GAP_EVIDENCE"


def test_evidence_naming_another_security_fails_closed():
    trade = make_closed_trade(
        trade_id="SYNTH-TRADE-1", entry_session=ENTRY, exit_session=EXIT
    )
    source_result = build_synthetic_audit_result(trades=(trade,))
    with pytest.raises(BaselineExperimentDiagnosticError) as error:
        build_material_adverse_overnight_gap_diagnostic(
            source_result=source_result,
            policy=build_material_adverse_overnight_gap_policy(),
            trade_gap_evidence=(
                CompletedTradeGapEvidence(
                    trade_id="SYNTH-TRADE-1",
                    security_id=SYNTHETIC_OTHER_SECURITY_ID,
                    boundaries=(),
                ),
            ),
        )
    assert error.value.code == "TRADE_GAP_EVIDENCE_SECURITY_MISMATCH"


def test_canonical_contract_example_six_of_forty():
    """The frozen worked example: 6 of 40 overnight trades qualify -> 0.15."""

    specs = []
    for index in range(40):
        open_price = "97" if index < 6 else "100"
        specs.append(
            (
                ENTRY,
                EXIT,
                [
                    boundary(
                        ENTRY,
                        date(2025, 1, 7),
                        close="100",
                        open_price=open_price,
                    )
                ],
            )
        )
    diagnostic = diagnose(*specs)
    assert diagnostic.gap_loss_trade_count == 6
    assert diagnostic.overnight_completed_trade_count == 40
    assert diagnostic.value == Decimal("0.15")


def test_a_profitable_trade_still_counts_when_it_gapped_adversely():
    """The diagnostic measures exposure, not final trade outcome."""

    trade = make_closed_trade(
        trade_id="SYNTH-TRADE-1",
        entry_session=ENTRY,
        exit_session=EXIT,
        entry_fill_price=Decimal("100"),
        exit_fill_price=Decimal("130"),
    )
    assert trade.trade_total_pnl > Decimal("0")
    source_result = build_synthetic_audit_result(trades=(trade,))
    diagnostic = build_material_adverse_overnight_gap_diagnostic(
        source_result=source_result,
        policy=build_material_adverse_overnight_gap_policy(),
        trade_gap_evidence=(
            CompletedTradeGapEvidence(
                trade_id="SYNTH-TRADE-1",
                security_id=SYNTHETIC_SECURITY_ID,
                boundaries=(
                    boundary(
                        ENTRY, date(2025, 1, 7), close="100", open_price="95"
                    ),
                ),
            ),
        ),
    )
    assert diagnostic.gap_loss_trade_count == 1


def test_repeated_calculation_is_deterministic():
    specs = (
        (
            ENTRY,
            EXIT,
            [
                boundary(ENTRY, date(2025, 1, 7), close="100", open_price="98"),
                boundary(
                    date(2025, 1, 8),
                    date(2025, 1, 9),
                    close="100",
                    open_price="100",
                ),
            ],
        ),
    )
    first = diagnose(*specs)
    second = diagnose(*specs)
    assert first == second
    assert first.diagnostic_fingerprint == second.diagnostic_fingerprint


def test_boundary_evidence_rejects_binary_floats_and_impossible_shapes():
    with pytest.raises(ValueError):
        OvernightBoundaryEvidence(
            security_id=SYNTHETIC_SECURITY_ID,
            previous_session=ENTRY,
            current_session=date(2025, 1, 7),
            evidence_status=OvernightBoundaryEvidenceStatus.COMPLETE,
            previous_regular_session_close=100.0,
            current_regular_session_open=Decimal("99"),
        )
    with pytest.raises(ValueError):
        boundary(date(2025, 1, 7), ENTRY)
    with pytest.raises(ValueError):
        boundary(ENTRY, date(2025, 1, 7), new_shares=4)
    with pytest.raises(ValueError):
        boundary(
            ENTRY,
            date(2025, 1, 7),
            status=OvernightBoundaryEvidenceStatus.INCOMPLETE,
        )


def test_the_gap_calculation_module_uses_no_binary_float_arithmetic():
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "stock_swing_d1"
        / "baseline_experiment"
        / "gap_diagnostic.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is float:
            offenders.append(("float literal", node.lineno))
        if isinstance(node, ast.Call):
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else getattr(function, "attr", None)
            )
            if name in {"float", "getcontext", "setcontext", "localcontext"}:
                offenders.append((name, node.lineno))
    assert offenders == []
