from __future__ import annotations

import inspect
from decimal import Decimal
from enum import StrEnum

import pytest
from pydantic import ValidationError

from stock_swing_d1.research_metrics import (
    CANONICAL_METRIC_QUANTUM,
    PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN,
    PERFORMANCE_MEASUREMENT_POLICY_ID,
    PERFORMANCE_MEASUREMENT_POLICY_SCHEMA_VERSION,
    PERFORMANCE_MEASUREMENT_POLICY_VERSION,
    STATISTICAL_ROUNDING_MODE,
    STATISTICAL_WORKING_PRECISION,
    AnnualizationPeriodConvention,
    CagrDayCountConvention,
    PerformanceMeasurementPolicy,
    UndefinedMetricBehavior,
    UndefinedMetricReason,
    build_performance_measurement_policy,
    compute_performance_measurement_policy_fingerprint,
    semantic_domain_sha256,
)
from stock_swing_d1.research_metrics import models as models_module
from stock_swing_d1.research_metrics import policy as policy_module
from stock_swing_d1.research_metrics.hashing import (
    _compute_performance_measurement_policy_fingerprint,
)


CANONICAL_SEMANTIC_KEYS = (
    "schema_version",
    "policy_id",
    "policy_version",
    "annualization_period_convention",
    "annualization_periods_per_year",
    "annual_risk_free_rate",
    "sortino_minimum_acceptable_return",
    "cagr_day_count_convention",
    "undefined_metric_behavior",
    "statistical_working_precision",
    "statistical_rounding_mode",
    "canonical_metric_quantum",
)

# The historical nine-key v0.2 payload and the digest it has always produced.
# Phase 16B.2 v0.2.1 Clause 42 forbids editing v0.2 in place, so introducing
# the v0.3 successor must leave this exact value reachable and unchanged.
HISTORICAL_V0_2_SEMANTIC_KEYS = CANONICAL_SEMANTIC_KEYS[:9]
HISTORICAL_V0_2_PAYLOAD = {
    "schema_version": "performance_measurement_policy.v0.2",
    "policy_id": "explicit_research_performance_measurement_v0.2",
    "policy_version": "0.2",
    "annualization_period_convention": (
        AnnualizationPeriodConvention.FIXED_OBSERVATIONS_PER_YEAR
    ),
    "annualization_periods_per_year": 252,
    "annual_risk_free_rate": Decimal("0"),
    "sortino_minimum_acceptable_return": Decimal("0"),
    "cagr_day_count_convention": CagrDayCountConvention.ACTUAL_365_25,
    "undefined_metric_behavior": UndefinedMetricBehavior.RETURN_NONE,
}
HISTORICAL_V0_2_FINGERPRINT = (
    "3076aa68576f02d46228a4164ae18dd18e729ca032ec1dfe7314d3a3df79d740"
)


def _semantic_values(policy) -> dict:
    return {name: getattr(policy, name) for name in CANONICAL_SEMANTIC_KEYS}


def _construct_with_consistent_fingerprint(**changes):
    """Construct directly, so only the model invariant can reject the values.

    The fingerprint is recomputed over the mutated content, which removes the
    fingerprint check as the reason for rejection.
    """

    values = _semantic_values(build_performance_measurement_policy())
    values.update(changes)
    fingerprint = _compute_performance_measurement_policy_fingerprint(**values)
    return PerformanceMeasurementPolicy(**values, policy_fingerprint=fingerprint)


def test_canonical_policy_is_built_without_caller_supplied_methodology():
    assert inspect.signature(build_performance_measurement_policy).parameters == {}

    policy = build_performance_measurement_policy()

    assert policy.schema_version == PERFORMANCE_MEASUREMENT_POLICY_SCHEMA_VERSION
    assert policy.policy_id == PERFORMANCE_MEASUREMENT_POLICY_ID
    assert policy.policy_version == PERFORMANCE_MEASUREMENT_POLICY_VERSION
    assert PERFORMANCE_MEASUREMENT_POLICY_SCHEMA_VERSION == (
        "performance_measurement_policy.v0.3"
    )
    assert PERFORMANCE_MEASUREMENT_POLICY_ID == (
        "explicit_research_performance_measurement_v0.3"
    )
    assert PERFORMANCE_MEASUREMENT_POLICY_VERSION == "0.3"
    assert PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN == (
        "performance_measurement_policy.v0.3"
    )


def test_canonical_policy_encodes_the_six_frozen_semantics(performance_policy):
    assert performance_policy.annualization_period_convention is (
        AnnualizationPeriodConvention.FIXED_OBSERVATIONS_PER_YEAR
    )
    assert performance_policy.annualization_periods_per_year == 252
    assert performance_policy.annual_risk_free_rate == Decimal("0")
    assert performance_policy.sortino_minimum_acceptable_return == Decimal("0")
    assert performance_policy.cagr_day_count_convention is (
        CagrDayCountConvention.ACTUAL_365_25
    )
    assert performance_policy.undefined_metric_behavior is (
        UndefinedMetricBehavior.RETURN_NONE
    )


def test_v0_3_adds_exactly_the_three_frozen_arithmetic_semantics(
    performance_policy,
):
    assert performance_policy.statistical_working_precision == 50
    assert performance_policy.statistical_rounding_mode == "ROUND_HALF_EVEN"
    assert performance_policy.canonical_metric_quantum == Decimal("1E-18")
    assert type(performance_policy.canonical_metric_quantum) is Decimal
    assert (
        performance_policy.canonical_metric_quantum.as_tuple().exponent == -18
    )

    assert STATISTICAL_WORKING_PRECISION == 50
    assert STATISTICAL_ROUNDING_MODE == "ROUND_HALF_EVEN"
    assert CANONICAL_METRIC_QUANTUM == Decimal("1E-18")

    added = set(PerformanceMeasurementPolicy.model_fields) - {
        *HISTORICAL_V0_2_SEMANTIC_KEYS,
        "policy_fingerprint",
    }
    assert added == {
        "statistical_working_precision",
        "statistical_rounding_mode",
        "canonical_metric_quantum",
    }


def test_policy_is_immutable_and_forbids_unknown_fields(performance_policy):
    with pytest.raises(ValidationError):
        performance_policy.annualization_periods_per_year = 365
    with pytest.raises(ValidationError):
        PerformanceMeasurementPolicy(
            **performance_policy.model_dump(mode="python"), unexpected=True
        )


def test_every_unresolved_semantic_is_explicitly_required():
    required = {
        "annualization_period_convention",
        "annualization_periods_per_year",
        "annual_risk_free_rate",
        "sortino_minimum_acceptable_return",
        "cagr_day_count_convention",
        "undefined_metric_behavior",
        "statistical_working_precision",
        "statistical_rounding_mode",
        "canonical_metric_quantum",
        "policy_fingerprint",
    }
    assert all(
        PerformanceMeasurementPolicy.model_fields[name].is_required()
        for name in required
    )


def test_actual_365_25_exists_and_is_distinct_from_actual_365_2425():
    assert CagrDayCountConvention.ACTUAL_365_25.value == "actual_365_25"
    assert CagrDayCountConvention.ACTUAL_365_2425.value == "actual_365_2425"
    assert (
        CagrDayCountConvention.ACTUAL_365_25
        is not CagrDayCountConvention.ACTUAL_365_2425
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"annualization_periods_per_year": 365},
        {"annual_risk_free_rate": Decimal("0.03")},
        {"sortino_minimum_acceptable_return": Decimal("0.01")},
        {
            "cagr_day_count_convention": (
                CagrDayCountConvention.ACTUAL_365_2425
            )
        },
        {"cagr_day_count_convention": CagrDayCountConvention.ACTUAL_365},
        {
            "undefined_metric_behavior": (
                UndefinedMetricBehavior.RAISE_VALIDATION_ERROR
            )
        },
        {"statistical_working_precision": 28},
        {"statistical_working_precision": 76},
        {"statistical_rounding_mode": "ROUND_HALF_UP"},
        {"statistical_rounding_mode": "ROUND_DOWN"},
        {"canonical_metric_quantum": Decimal("1E-17")},
        {"canonical_metric_quantum": Decimal("1E-28")},
        {"canonical_metric_quantum": Decimal("0.0000000000000000010")},
    ),
)
def test_a_frozen_semantic_mutation_is_rejected_not_refingerprinted(changes):
    """The superseded premise was that a mutation yields another valid policy."""

    with pytest.raises(ValidationError):
        _construct_with_consistent_fingerprint(**changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"schema_version": "performance_measurement_policy.v0.1"},
        {"schema_version": "performance_measurement_policy.v0.2"},
        {"policy_id": "explicit_research_performance_measurement_v0.1"},
        {"policy_id": "explicit_research_performance_measurement_v0.2"},
        {"policy_version": "0.1"},
        {"policy_version": "0.2"},
    ),
)
def test_superseded_policy_identity_is_rejected(changes):
    with pytest.raises(ValidationError):
        _construct_with_consistent_fingerprint(**changes)


def test_a_v0_2_policy_cannot_masquerade_as_v0_3():
    """A complete v0.2 identity is rejected outright, not silently upgraded."""

    with pytest.raises(ValidationError):
        _construct_with_consistent_fingerprint(
            schema_version="performance_measurement_policy.v0.2",
            policy_id="explicit_research_performance_measurement_v0.2",
            policy_version="0.2",
        )


def test_policy_rejects_coercion_nonfinite_and_unknown_values():
    for invalid in (0.0, "0", Decimal("NaN"), Decimal("Infinity")):
        with pytest.raises((ValidationError, ValueError)):
            _construct_with_consistent_fingerprint(annual_risk_free_rate=invalid)
    for invalid in (True, 252.0, "252"):
        with pytest.raises((ValidationError, ValueError)):
            _construct_with_consistent_fingerprint(
                annualization_periods_per_year=invalid
            )
    for invalid in (True, 50.0, "50"):
        with pytest.raises((ValidationError, ValueError)):
            _construct_with_consistent_fingerprint(
                statistical_working_precision=invalid
            )
    for invalid in (1e-18, "1E-18", Decimal("NaN")):
        with pytest.raises((ValidationError, ValueError)):
            _construct_with_consistent_fingerprint(
                canonical_metric_quantum=invalid
            )
    with pytest.raises((ValidationError, ValueError)):
        _construct_with_consistent_fingerprint(
            annualization_period_convention="fixed_observations_per_year"
        )
    with pytest.raises((ValidationError, ValueError)):
        _construct_with_consistent_fingerprint(
            cagr_day_count_convention="actual_365_25"
        )


def test_policy_fingerprint_is_deterministic_and_constant(performance_policy):
    assert performance_policy.policy_fingerprint == (
        compute_performance_measurement_policy_fingerprint(performance_policy)
    )
    assert performance_policy == build_performance_measurement_policy()
    assert performance_policy.policy_fingerprint == (
        build_performance_measurement_policy().policy_fingerprint
    )


def test_policy_rejects_a_tampered_fingerprint(performance_policy):
    values = performance_policy.model_dump(mode="python")
    values["policy_fingerprint"] = "f" * 64
    with pytest.raises(ValidationError):
        PerformanceMeasurementPolicy.model_validate(values)


def test_policy_fingerprint_payload_has_exactly_the_twelve_semantic_keys(
    performance_policy,
):
    payload = {
        "schema_version": "performance_measurement_policy.v0.3",
        "policy_id": "explicit_research_performance_measurement_v0.3",
        "policy_version": "0.3",
        "annualization_period_convention": (
            AnnualizationPeriodConvention.FIXED_OBSERVATIONS_PER_YEAR
        ),
        "annualization_periods_per_year": 252,
        "annual_risk_free_rate": Decimal("0"),
        "sortino_minimum_acceptable_return": Decimal("0"),
        "cagr_day_count_convention": CagrDayCountConvention.ACTUAL_365_25,
        "undefined_metric_behavior": UndefinedMetricBehavior.RETURN_NONE,
        "statistical_working_precision": 50,
        "statistical_rounding_mode": "ROUND_HALF_EVEN",
        "canonical_metric_quantum": Decimal("1E-18"),
    }

    assert tuple(payload) == CANONICAL_SEMANTIC_KEYS
    assert len(CANONICAL_SEMANTIC_KEYS) == 12
    assert tuple(
        inspect.signature(
            _compute_performance_measurement_policy_fingerprint
        ).parameters
    ) == CANONICAL_SEMANTIC_KEYS
    assert performance_policy.policy_fingerprint == semantic_domain_sha256(
        PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN, payload
    )
    assert performance_policy.policy_fingerprint != semantic_domain_sha256(
        PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN,
        {**payload, "unexpected_thirteenth_key": None},
    )


def test_the_historical_v0_2_fingerprint_contract_is_untouched(
    performance_policy,
):
    """v0.2 is superseded, not edited: its digest is exactly what it was."""

    assert tuple(HISTORICAL_V0_2_PAYLOAD) == HISTORICAL_V0_2_SEMANTIC_KEYS
    assert len(HISTORICAL_V0_2_SEMANTIC_KEYS) == 9
    assert (
        semantic_domain_sha256(
            "performance_measurement_policy.v0.2", HISTORICAL_V0_2_PAYLOAD
        )
        == HISTORICAL_V0_2_FINGERPRINT
    )
    assert performance_policy.policy_fingerprint != HISTORICAL_V0_2_FINGERPRINT


def test_changing_an_arithmetic_value_invalidates_the_v0_3_identity(
    performance_policy,
):
    payload = {
        name: getattr(performance_policy, name)
        for name in CANONICAL_SEMANTIC_KEYS
    }
    for name, replacement in (
        ("statistical_working_precision", 76),
        ("statistical_rounding_mode", "ROUND_HALF_UP"),
        ("canonical_metric_quantum", Decimal("1E-17")),
    ):
        assert performance_policy.policy_fingerprint != semantic_domain_sha256(
            PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN,
            {**payload, name: replacement},
        )


def test_the_undefined_reason_vocabulary_is_frozen_and_lives_in_policy():
    assert {member.value for member in UndefinedMetricReason} == {
        "NO_COMPLETED_TRADES",
        "INSUFFICIENT_RETURN_OBSERVATIONS",
        "ZERO_RETURN_VARIANCE",
        "NO_DOWNSIDE_DEVIATION",
        "ZERO_GROSS_LOSS",
        # Phase 16B Amendment v0.1: a population with completed trades but no
        # winner (or no loser) is not the same fact as an empty population, so
        # each gets its own honest reason instead of borrowing another's.
        "NO_WINNING_TRADES",
        "NO_LOSING_TRADES",
    }
    assert issubclass(UndefinedMetricReason, StrEnum)
    assert UndefinedMetricReason.__module__ == policy_module.__name__

    # UF-07: the reason enum belongs to the methodology layer, and the model
    # layer keeps its frozen no-new-enum scope guard.
    declared_in_models = [
        name
        for name, member in vars(models_module).items()
        if isinstance(member, type)
        and issubclass(member, StrEnum)
        and member.__module__ == models_module.__name__
    ]
    assert declared_in_models == []
