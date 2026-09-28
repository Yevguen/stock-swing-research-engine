"""Phase 16B.3 persistence and identity for the four amendment metrics.

Phase 16B.3 serializes, deserializes, fingerprints, validates and persists the
amendment metrics. It never calculates or reconstructs one. These tests pin
both halves of that: the four values survive the artifact boundary exactly, and
no pre-amendment identity can stand for an amended payload.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from stock_swing_d1.research_metrics import (
    RESEARCH_METRICS_PERSISTENCE_HASH_DOMAIN,
    RESEARCH_METRICS_PERSISTENCE_SCHEMA_VERSION,
    RESEARCH_METRICS_REPORT_PROJECTION_SCHEMA_VERSION,
    RESEARCH_METRICS_RESULT_HASH_DOMAIN,
    RESEARCH_METRICS_RESULT_SCHEMA_VERSION,
    RESEARCH_METRIC_VALUE_SCHEMA_VERSION,
    TRADE_PERFORMANCE_METRICS_SCHEMA_VERSION,
    ResearchMetricsPersistenceError,
    UndefinedMetricReason,
    build_research_metrics_persistence_artifact,
    build_research_metrics_result,
    calculate_research_metrics,
    deserialize_research_metrics_artifact,
    project_research_metrics_report,
    serialize_research_metrics_artifact,
)

from tests.research_metrics.conftest import (
    canonical_experiment,
    make_closed_trade,
)
from tests.research_metrics.test_phase16b_amendment_trade_metrics import (
    AMENDMENT_METRICS,
    CANONICAL_ENTRY,
    CANONICAL_EXIT,
)


def _population():
    """A population in which all four amendment metrics are defined."""

    return (
        make_closed_trade(
            trade_id="T1",
            entry_session=CANONICAL_ENTRY,
            exit_session=CANONICAL_EXIT,
            entry_fill_price=Decimal("10"),
            exit_fill_price=Decimal("12"),
        ),
        make_closed_trade(
            trade_id="T2",
            entry_session=CANONICAL_ENTRY,
            exit_session=CANONICAL_ENTRY,
            entry_fill_price=Decimal("10"),
            exit_fill_price=Decimal("8"),
        ),
        make_closed_trade(
            trade_id="T3",
            entry_session=CANONICAL_ENTRY,
            exit_session=CANONICAL_EXIT,
            entry_fill_price=Decimal("10"),
            exit_fill_price=Decimal("10"),
        ),
    )


@pytest.fixture
def measured_result(performance_policy, canonical_benchmark_series):
    experiment = canonical_experiment(trades=_population())
    return calculate_research_metrics(
        source_result=experiment,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )


@pytest.fixture
def artifact(measured_result, canonical_audit_result, performance_policy):
    return build_research_metrics_persistence_artifact(
        result=measured_result,
        decision_interval=canonical_audit_result.decision_interval,
        starting_capital=canonical_audit_result.initial_equity,
        equity_observation_count=3,
        periodic_return_observation_count=2,
        policy=performance_policy,
    )


def _payload(value) -> dict:
    return json.loads(serialize_research_metrics_artifact(value))


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _rebuild_with_trade_metrics(result, **updates):
    return build_research_metrics_result(
        provenance=result.provenance,
        strategy_metrics=result.strategy_metrics,
        trade_metrics=result.trade_metrics.model_copy(update=updates),
        benchmark_metrics=result.benchmark_metrics,
        relative_metrics=result.relative_metrics,
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_the_measured_population_defines_all_four_metrics(measured_result):
    trades = measured_result.trade_metrics

    assert trades.average_winner_usd.value == Decimal("200")
    assert trades.average_loser_usd.value == Decimal("-200")
    assert trades.worst_trade_usd.value == Decimal("-200")
    # Holds of 2, 1 and 2 sessions.
    assert trades.average_holding_sessions.value == Decimal(
        "1.666666666666666667"
    )


def test_serialize_deserialize_reserialize_is_byte_identical(artifact):
    first = serialize_research_metrics_artifact(artifact)
    restored = deserialize_research_metrics_artifact(first)
    assert serialize_research_metrics_artifact(restored) == first
    assert restored.artifact_fingerprint == artifact.artifact_fingerprint


def test_all_four_values_survive_the_round_trip_exactly(artifact):
    restored = deserialize_research_metrics_artifact(
        serialize_research_metrics_artifact(artifact)
    )
    original = artifact.research_metrics_result.trade_metrics
    returned = restored.research_metrics_result.trade_metrics

    for name in AMENDMENT_METRICS:
        before = getattr(original, name)
        after = getattr(returned, name)
        assert after == before
        assert after.value.as_tuple() == before.value.as_tuple()
    assert returned == original


def test_the_four_values_are_encoded_as_lossless_decimal_strings(artifact):
    trade_payload = _payload(artifact)["research_metrics_result"][
        "trade_metrics"
    ]

    for name in AMENDMENT_METRICS:
        assert name in trade_payload
        encoded = trade_payload[name]["value"]
        assert type(encoded) is str
        assert Decimal(encoded) == getattr(
            artifact.research_metrics_result.trade_metrics, name
        ).value


@pytest.mark.parametrize(
    ("field_name", "reason"),
    (
        ("average_winner_usd", UndefinedMetricReason.NO_WINNING_TRADES),
        ("average_loser_usd", UndefinedMetricReason.NO_LOSING_TRADES),
    ),
)
def test_the_new_undefined_reasons_round_trip(
    performance_policy, canonical_benchmark_series, field_name, reason
):
    """An all-winning or all-losing population persists its honest reason."""

    losing = field_name == "average_winner_usd"
    experiment = canonical_experiment(
        trades=(
            make_closed_trade(
                trade_id="T1",
                entry_session=CANONICAL_ENTRY,
                exit_session=CANONICAL_EXIT,
                entry_fill_price=Decimal("10"),
                exit_fill_price=Decimal("8") if losing else Decimal("12"),
            ),
        )
    )
    result = calculate_research_metrics(
        source_result=experiment,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )
    assert getattr(result.trade_metrics, field_name).undefined_reason is reason

    stored = build_research_metrics_persistence_artifact(
        result=result,
        decision_interval=experiment.decision_interval,
        starting_capital=experiment.initial_equity,
        equity_observation_count=3,
        periodic_return_observation_count=2,
        policy=performance_policy,
    )
    restored = deserialize_research_metrics_artifact(
        serialize_research_metrics_artifact(stored)
    )
    metric = getattr(
        restored.research_metrics_result.trade_metrics, field_name
    )
    assert metric.value is None
    assert metric.undefined_reason is reason


# ---------------------------------------------------------------------------
# Fingerprint sensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", AMENDMENT_METRICS)
def test_changing_one_amendment_value_changes_both_fingerprints(
    measured_result, canonical_audit_result, performance_policy, field_name
):
    from stock_swing_d1.research_metrics import defined_metric

    original = getattr(measured_result.trade_metrics, field_name).value
    changed = _rebuild_with_trade_metrics(
        measured_result,
        **{field_name: defined_metric(original + Decimal("1"))},
    )
    assert changed.result_fingerprint != measured_result.result_fingerprint

    def store(result):
        return build_research_metrics_persistence_artifact(
            result=result,
            decision_interval=canonical_audit_result.decision_interval,
            starting_capital=canonical_audit_result.initial_equity,
            equity_observation_count=3,
            periodic_return_observation_count=2,
            policy=performance_policy,
        )

    assert (
        store(changed).artifact_fingerprint
        != store(measured_result).artifact_fingerprint
    )


def test_the_result_fingerprint_is_bound_to_the_v0_3_domain(measured_result):
    assert RESEARCH_METRICS_RESULT_HASH_DOMAIN == "research_metrics_result.v0.3"
    assert measured_result.schema_version == "research_metrics_result.v0.3"
    assert measured_result.trade_metrics.schema_version == (
        "trade_performance_metrics.v0.3"
    )


# ---------------------------------------------------------------------------
# No pre-amendment identity may stand for an amended payload
# ---------------------------------------------------------------------------


def test_the_advanced_identities_are_exactly_the_ones_that_changed():
    assert RESEARCH_METRICS_RESULT_SCHEMA_VERSION == (
        "research_metrics_result.v0.3"
    )
    assert TRADE_PERFORMANCE_METRICS_SCHEMA_VERSION == (
        "trade_performance_metrics.v0.3"
    )
    assert RESEARCH_METRIC_VALUE_SCHEMA_VERSION == (
        "research_metric_value.v0.2"
    )
    assert RESEARCH_METRICS_PERSISTENCE_SCHEMA_VERSION == (
        "research_metrics_persistence.v0.2"
    )
    assert RESEARCH_METRICS_PERSISTENCE_HASH_DOMAIN == (
        "research_metrics_persistence.v0.2"
    )
    # The report projection carries no trade section, so the amendment does
    # not change its payload and its identity deliberately stays put.
    assert RESEARCH_METRICS_REPORT_PROJECTION_SCHEMA_VERSION == (
        "research_metrics_report_projection.v0.1"
    )


@pytest.mark.parametrize(
    "stale",
    ("research_metrics_persistence.v0.1", "research_metrics_persistence.v0.3"),
)
def test_a_wrong_persistence_schema_version_is_refused(artifact, stale):
    payload = _payload(artifact)
    payload["schema_version"] = stale
    with pytest.raises(ResearchMetricsPersistenceError) as captured:
        deserialize_research_metrics_artifact(_canonical_bytes(payload))
    assert captured.value.code == "UNSUPPORTED_SCHEMA_VERSION"


def test_a_v0_2_result_identity_cannot_carry_the_amended_payload(artifact):
    payload = _payload(artifact)
    payload["research_metrics_result"]["schema_version"] = (
        "research_metrics_result.v0.2"
    )
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


def test_amended_serialization_uses_v0_2_metric_value_identities(artifact):
    payload = _payload(artifact)
    trade_metrics = payload["research_metrics_result"]["trade_metrics"]
    for field_name in AMENDMENT_METRICS:
        assert trade_metrics[field_name]["schema_version"] == (
            "research_metric_value.v0.2"
        )


def test_v0_1_metric_value_cannot_describe_an_amendment_reason(artifact):
    payload = _payload(artifact)
    metric = payload["research_metrics_result"]["trade_metrics"][
        "average_winner_usd"
    ]
    metric["schema_version"] = "research_metric_value.v0.1"
    metric["value"] = None
    metric["undefined_reason"] = "NO_WINNING_TRADES"
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


@pytest.mark.parametrize("field_name", AMENDMENT_METRICS)
def test_a_pre_amendment_trade_payload_is_refused(artifact, field_name):
    """Dropping a field cannot be read as a valid pre-amendment artifact."""

    payload = _payload(artifact)
    del payload["research_metrics_result"]["trade_metrics"][field_name]
    with pytest.raises(ResearchMetricsPersistenceError) as captured:
        deserialize_research_metrics_artifact(_canonical_bytes(payload))
    assert captured.value.code == "MALFORMED_ARTIFACT"


def test_an_unknown_trade_metric_field_is_refused(artifact):
    payload = _payload(artifact)
    payload["research_metrics_result"]["trade_metrics"][
        "average_winner_pct"
    ] = {
        "schema_version": "research_metric_value.v0.2",
        "value": "0.1",
        "undefined_reason": None,
    }
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


# ---------------------------------------------------------------------------
# Report boundary
# ---------------------------------------------------------------------------


def test_the_report_projection_shape_is_unchanged_by_the_amendment(artifact):
    """The Phase 16B report projection never carried trade metrics.

    Its payload therefore does not change because of these four fields, so its
    version deliberately stays at v0.1. A report consumer reads the four
    values from the authoritative ``trade_metrics`` the artifact preserves --
    which is exactly what the Phase 16C report does.
    """

    projection = project_research_metrics_report(artifact)

    assert projection.schema_version == (
        "research_metrics_report_projection.v0.1"
    )
    assert tuple(type(projection).model_fields) == (
        "schema_version",
        "context",
        "strategy",
        "benchmark",
        "comparative",
    )
    assert (
        projection.context.research_metrics_result_fingerprint
        == artifact.research_metrics_result.result_fingerprint
    )


def test_the_preserved_artifact_carries_all_four_values_for_a_report(artifact):
    trades = artifact.research_metrics_result.trade_metrics

    for name in AMENDMENT_METRICS:
        assert getattr(trades, name).value is not None
        assert type(getattr(trades, name).value) is Decimal
