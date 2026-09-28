"""Frozen Phase 16C earnings-filter exclusion diagnostic v0.1 semantics."""

from __future__ import annotations

from datetime import date

import pytest

from stock_swing_d1.earnings.integration.models import (
    EarningsIntegrationAction,
)
from stock_swing_d1.strategy.baseline.models import BaselineSignalAction

from stock_swing_d1.baseline_experiment import (
    BaselineExperimentDiagnosticError,
    build_earnings_filter_exclusion_diagnostic,
)

from tests.baseline_experiment.conftest import (
    SYNTHETIC_OTHER_SECURITY_ID,
    build_synthetic_audit_result,
    make_signal_row,
)


S1 = date(2025, 3, 3)
S2 = date(2025, 3, 4)
S3 = date(2025, 3, 5)


def diagnose(rows):
    return build_earnings_filter_exclusion_diagnostic(
        source_result=build_synthetic_audit_result(
            signal_provenance=tuple(rows)
        )
    )


def test_an_earnings_prohibited_candidate_is_counted():
    diagnostic = diagnose(
        [
            make_signal_row(
                session=S1,
                earnings_action=EarningsIntegrationAction.ENTRY_BLOCKED,
            )
        ]
    )
    assert diagnostic.earnings_filter_exclusions == 1
    assert diagnostic.earnings_evaluated_entry_candidate_count == 1
    assert diagnostic.diagnostic_version == "earnings_filter_exclusions.v0.1"


def test_an_allowed_candidate_is_not_counted():
    diagnostic = diagnose(
        [
            make_signal_row(
                session=S1,
                earnings_action=EarningsIntegrationAction.ENTRY_ALLOWED,
                action=BaselineSignalAction.VALID_LONG_SIGNAL,
            )
        ]
    )
    assert diagnostic.earnings_filter_exclusions == 0
    assert diagnostic.earnings_evaluated_entry_candidate_count == 1


def test_a_candidate_rejected_for_an_unrelated_reason_is_not_counted():
    """The earnings policy allowed it; another filter rejected it."""

    diagnostic = diagnose(
        [
            make_signal_row(
                session=S1,
                earnings_action=EarningsIntegrationAction.ENTRY_ALLOWED,
                action=BaselineSignalAction.NO_SIGNAL,
                rsi_above_50=False,
                close_above_sma50=False,
            )
        ]
    )
    assert diagnostic.earnings_filter_exclusions == 0
    assert diagnostic.earnings_evaluated_entry_candidate_count == 1


def test_a_row_the_entry_policy_never_evaluated_enters_neither_count():
    diagnostic = diagnose(
        [
            make_signal_row(
                session=S1,
                earnings_action=EarningsIntegrationAction.HOLD_POSITION,
                earnings_entry_allowed=False,
            ),
            make_signal_row(
                session=S2,
                earnings_action=(
                    EarningsIntegrationAction.PENDING_ENTRY_INVALIDATED
                ),
                earnings_entry_allowed=False,
            ),
        ]
    )
    assert diagnostic.earnings_filter_exclusions == 0
    assert diagnostic.earnings_evaluated_entry_candidate_count == 0


def test_a_later_revision_cannot_alter_a_prior_exclusion_classification():
    """A later session's decision never reclassifies an earlier one."""

    early_only = [
        make_signal_row(
            session=S1,
            earnings_action=EarningsIntegrationAction.ENTRY_BLOCKED,
        )
    ]
    before = diagnose(early_only)

    # The same security is later evaluated again, and by then the revised
    # schedule permits entry. The earlier blocked decision is untouched.
    after = diagnose(
        early_only
        + [
            make_signal_row(
                session=S3,
                earnings_action=EarningsIntegrationAction.ENTRY_ALLOWED,
                action=BaselineSignalAction.VALID_LONG_SIGNAL,
            )
        ]
    )
    assert before.earnings_filter_exclusions == 1
    assert after.earnings_filter_exclusions == 1
    assert after.earnings_evaluated_entry_candidate_count == 2


def test_multiple_securities_and_sessions_are_counted_independently():
    diagnostic = diagnose(
        [
            make_signal_row(
                session=S1,
                earnings_action=EarningsIntegrationAction.ENTRY_BLOCKED,
            ),
            make_signal_row(
                session=S1,
                security_id=SYNTHETIC_OTHER_SECURITY_ID,
                earnings_action=EarningsIntegrationAction.ENTRY_ALLOWED,
                action=BaselineSignalAction.VALID_LONG_SIGNAL,
            ),
            make_signal_row(
                session=S2,
                earnings_action=EarningsIntegrationAction.ENTRY_BLOCKED,
            ),
        ]
    )
    assert diagnostic.earnings_filter_exclusions == 2
    assert diagnostic.earnings_evaluated_entry_candidate_count == 3


def test_repeated_calculation_is_deterministic():
    rows = [
        make_signal_row(
            session=S1,
            earnings_action=EarningsIntegrationAction.ENTRY_BLOCKED,
        ),
        make_signal_row(
            session=S2,
            earnings_action=EarningsIntegrationAction.ENTRY_ALLOWED,
            action=BaselineSignalAction.VALID_LONG_SIGNAL,
        ),
    ]
    first = diagnose(rows)
    second = diagnose(rows)
    assert first == second
    assert first.diagnostic_fingerprint == second.diagnostic_fingerprint


def test_a_self_contradictory_authoritative_row_fails_closed():
    with pytest.raises(BaselineExperimentDiagnosticError) as error:
        diagnose(
            [
                make_signal_row(
                    session=S1,
                    earnings_action=EarningsIntegrationAction.ENTRY_BLOCKED,
                    earnings_entry_allowed=True,
                )
            ]
        )
    assert error.value.code == "INCONSISTENT_EARNINGS_PROVENANCE"


def test_no_rate_is_published_only_the_two_counts():
    diagnostic = diagnose(
        [
            make_signal_row(
                session=S1,
                earnings_action=EarningsIntegrationAction.ENTRY_BLOCKED,
            )
        ]
    )
    fields = set(type(diagnostic).model_fields)
    assert not {
        name for name in fields if "rate" in name or "frequency" in name
    }
