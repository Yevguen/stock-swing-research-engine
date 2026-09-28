"""Phase 16B.3 persistence and report-facing acceptance tests."""

from __future__ import annotations

import ast
import json
from decimal import Decimal
from pathlib import Path

import pytest

from stock_swing_d1.research_metrics import (
    RESEARCH_METRICS_PERSISTENCE_HASH_DOMAIN,
    RESEARCH_METRICS_PERSISTENCE_SCHEMA_VERSION,
    RESEARCH_METRICS_REPORT_PROJECTION_SCHEMA_VERSION,
    ResearchMetricsPersistenceError,
    UndefinedMetricReason,
    build_research_metrics_persistence_artifact,
    build_research_metrics_result,
    calculate_research_metrics,
    compute_research_metrics_persistence_fingerprint,
    deserialize_research_metrics_artifact,
    project_research_metrics_report,
    read_research_metrics_artifact,
    semantic_sha256,
    serialize_research_metrics_artifact,
    undefined_metric,
    write_research_metrics_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "src" / "stock_swing_d1" / "research_metrics"
PHASE16B3_MODULES = (
    PRODUCTION / "persistence.py",
    PRODUCTION / "reporting.py",
)


@pytest.fixture
def authoritative_result(
    canonical_audit_result, performance_policy, canonical_benchmark_series
):
    return calculate_research_metrics(
        source_result=canonical_audit_result,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )


@pytest.fixture
def persistence_artifact(
    authoritative_result, canonical_audit_result, performance_policy
):
    return build_research_metrics_persistence_artifact(
        result=authoritative_result,
        decision_interval=canonical_audit_result.decision_interval,
        starting_capital=canonical_audit_result.initial_equity,
        equity_observation_count=3,
        periodic_return_observation_count=2,
        policy=performance_policy,
    )


def _payload(artifact) -> dict[str, object]:
    return json.loads(serialize_research_metrics_artifact(artifact))


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _replace_result(result, **updates):
    values = {
        "provenance": result.provenance,
        "strategy_metrics": result.strategy_metrics,
        "trade_metrics": result.trade_metrics,
        "benchmark_metrics": result.benchmark_metrics,
        "relative_metrics": result.relative_metrics,
    }
    values.update(updates)
    return build_research_metrics_result(**values)


def _replace_strategy_metric(result, name: str, metric):
    strategy = result.strategy_metrics.model_copy(update={name: metric})
    return _replace_result(result, strategy_metrics=strategy)


def _artifact_for_result(
    result, canonical_audit_result, performance_policy, **updates
):
    values = {
        "result": result,
        "decision_interval": canonical_audit_result.decision_interval,
        "starting_capital": canonical_audit_result.initial_equity,
        "equity_observation_count": 3,
        "periodic_return_observation_count": 2,
        "policy": performance_policy,
    }
    values.update(updates)
    return build_research_metrics_persistence_artifact(**values)


def test_schema_v02_and_deterministic_serialization(persistence_artifact):
    first = serialize_research_metrics_artifact(persistence_artifact)
    second = serialize_research_metrics_artifact(persistence_artifact)
    assert first == second
    assert first.decode("utf-8").startswith('{"artifact_fingerprint":')
    assert _payload(persistence_artifact)["schema_version"] == (
        RESEARCH_METRICS_PERSISTENCE_SCHEMA_VERSION
    )


def test_decimal_round_trip_preserves_long_precision_and_scale(
    authoritative_result, canonical_audit_result, performance_policy
):
    precise = Decimal("0.12345678901234567890123456789012345678900")
    changed = _replace_strategy_metric(
        authoritative_result,
        "total_return",
        authoritative_result.strategy_metrics.total_return.model_copy(
            update={"value": precise}
        ),
    )
    artifact = _artifact_for_result(
        changed, canonical_audit_result, performance_policy
    )
    encoded = serialize_research_metrics_artifact(artifact)
    assert str(precise).encode("ascii") in encoded
    restored = deserialize_research_metrics_artifact(encoded)
    restored_value = (
        restored.research_metrics_result.strategy_metrics.total_return.value
    )
    assert restored_value.as_tuple() == precise.as_tuple()


@pytest.mark.parametrize(
    "bad_value",
    ("not-a-decimal", "NaN", "sNaN", "Infinity", "-Infinity", "1.0E+1"),
)
def test_malformed_or_noncanonical_decimal_is_rejected(
    persistence_artifact, bad_value
):
    payload = _payload(persistence_artifact)
    payload["starting_capital"] = bad_value
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


def test_json_number_and_nonfinite_json_constant_are_rejected(
    persistence_artifact,
):
    canonical = serialize_research_metrics_artifact(persistence_artifact).decode()
    numeric = canonical.replace('"starting_capital":"100000"', '"starting_capital":100000.0')
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(numeric)
    nonfinite = canonical.replace('"starting_capital":"100000"', '"starting_capital":NaN')
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(nonfinite)


def test_defined_metric_round_trip(persistence_artifact):
    restored = deserialize_research_metrics_artifact(
        serialize_research_metrics_artifact(persistence_artifact)
    )
    metric = restored.research_metrics_result.strategy_metrics.total_return
    assert metric.value is not None
    assert metric.undefined_reason is None


@pytest.mark.parametrize("reason", tuple(UndefinedMetricReason))
def test_every_undefined_reason_round_trips(
    reason, authoritative_result, canonical_audit_result, performance_policy
):
    changed = _replace_strategy_metric(
        authoritative_result, "sortino_ratio", undefined_metric(reason)
    )
    artifact = _artifact_for_result(
        changed, canonical_audit_result, performance_policy
    )
    restored = deserialize_research_metrics_artifact(
        serialize_research_metrics_artifact(artifact)
    )
    metric = restored.research_metrics_result.strategy_metrics.sortino_ratio
    assert metric.value is None
    assert metric.undefined_reason is reason


@pytest.mark.parametrize(
    ("value", "reason"),
    ((None, None), ("1.000", "ZERO_RETURN_VARIANCE")),
)
def test_inconsistent_metric_state_is_rejected(
    persistence_artifact, value, reason
):
    payload = _payload(persistence_artifact)
    metric = payload["research_metrics_result"]["strategy_metrics"]["sharpe_ratio"]
    metric["value"] = value
    metric["undefined_reason"] = reason
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


def test_unknown_reason_code_is_rejected(persistence_artifact):
    payload = _payload(persistence_artifact)
    metric = payload["research_metrics_result"]["strategy_metrics"]["sharpe_ratio"]
    metric["value"] = None
    metric["undefined_reason"] = "UNKNOWN_REASON"
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


def test_missing_and_unknown_schema_versions_are_rejected(
    persistence_artifact,
):
    missing = _payload(persistence_artifact)
    del missing["schema_version"]
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(missing))

    unknown = _payload(persistence_artifact)
    unknown["schema_version"] = "research_metrics_persistence.v9.9"
    with pytest.raises(
        ResearchMetricsPersistenceError,
        match="UNSUPPORTED_SCHEMA_VERSION",
    ):
        deserialize_research_metrics_artifact(_canonical_bytes(unknown))


def test_missing_nested_field_and_extra_field_are_rejected(
    persistence_artifact,
):
    missing = _payload(persistence_artifact)
    del missing["research_metrics_result"]["trade_metrics"]["gross_loss"]
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(missing))

    extra = _payload(persistence_artifact)
    extra["generated_at"] = "2025-01-01T00:00:00Z"
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(extra))


def test_unknown_policy_identity_is_rejected(persistence_artifact):
    payload = _payload(persistence_artifact)
    payload["policy_ref"]["policy_version"] = "9.9"
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


def test_tampered_semantic_result_fingerprint_is_rejected(
    persistence_artifact,
):
    payload = _payload(persistence_artifact)
    payload["research_metrics_result"]["result_fingerprint"] = "f" * 64
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


def test_provenance_and_semantic_fingerprint_are_preserved(
    persistence_artifact,
):
    restored = deserialize_research_metrics_artifact(
        serialize_research_metrics_artifact(persistence_artifact)
    )
    original = persistence_artifact.research_metrics_result
    loaded = restored.research_metrics_result
    assert loaded.provenance == original.provenance
    assert loaded.result_fingerprint == original.result_fingerprint
    assert restored.policy_ref.policy_fingerprint == (
        original.provenance.performance_measurement_policy_fingerprint
    )


def test_persistence_fingerprint_is_deterministic_and_verified(
    persistence_artifact,
):
    first = compute_research_metrics_persistence_fingerprint(
        persistence_artifact
    )
    second = compute_research_metrics_persistence_fingerprint(
        persistence_artifact
    )
    assert first == second == persistence_artifact.artifact_fingerprint

    payload = _payload(persistence_artifact)
    payload["starting_capital"] = "100000.00"
    with pytest.raises(
        ResearchMetricsPersistenceError,
        match="ARTIFACT_FINGERPRINT_MISMATCH",
    ):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


def test_invalid_artifact_fingerprint_and_invalid_utf8_are_rejected(
    persistence_artifact,
):
    payload = _payload(persistence_artifact)
    payload["artifact_fingerprint"] = "f" * 64
    with pytest.raises(
        ResearchMetricsPersistenceError,
        match="ARTIFACT_FINGERPRINT_MISMATCH",
    ):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))
    with pytest.raises(ResearchMetricsPersistenceError, match="INVALID_UTF8"):
        deserialize_research_metrics_artifact(b"\xff")
    with pytest.raises(ResearchMetricsPersistenceError, match="INVALID_UTF8"):
        deserialize_research_metrics_artifact("\ud800")


def test_semantic_and_persistence_fingerprint_domains_are_separate(
    authoritative_result, canonical_audit_result, performance_policy
):
    first = _artifact_for_result(
        authoritative_result,
        canonical_audit_result,
        performance_policy,
        starting_capital=Decimal("100000"),
    )
    second = _artifact_for_result(
        authoritative_result,
        canonical_audit_result,
        performance_policy,
        starting_capital=Decimal("100000.00"),
    )
    assert first.research_metrics_result.result_fingerprint == (
        second.research_metrics_result.result_fingerprint
    )
    assert first.artifact_fingerprint != second.artifact_fingerprint
    assert first.artifact_fingerprint != first.research_metrics_result.result_fingerprint
    assert RESEARCH_METRICS_PERSISTENCE_HASH_DOMAIN != (
        first.research_metrics_result.schema_version
    )


def test_invalid_policy_provenance_relationship_is_rejected(
    persistence_artifact,
):
    payload = _payload(persistence_artifact)
    payload["policy_ref"]["policy_fingerprint"] = "f" * 64
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key != "artifact_fingerprint"
    }
    payload["artifact_fingerprint"] = semantic_sha256(
        {
            "domain": RESEARCH_METRICS_PERSISTENCE_HASH_DOMAIN,
            "payload": fingerprint_payload,
        }
    )
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(_canonical_bytes(payload))


def test_strategy_benchmark_comparative_interval_and_capital_survive(
    persistence_artifact,
):
    restored = deserialize_research_metrics_artifact(
        serialize_research_metrics_artifact(persistence_artifact)
    )
    assert restored.decision_interval == persistence_artifact.decision_interval
    assert restored.starting_capital.as_tuple() == (
        persistence_artifact.starting_capital.as_tuple()
    )
    assert restored.equity_observation_count == 3
    assert restored.periodic_return_observation_count == 2
    assert restored.research_metrics_result.strategy_metrics == (
        persistence_artifact.research_metrics_result.strategy_metrics
    )
    assert restored.research_metrics_result.trade_metrics == (
        persistence_artifact.research_metrics_result.trade_metrics
    )
    assert restored.research_metrics_result.benchmark_metrics == (
        persistence_artifact.research_metrics_result.benchmark_metrics
    )
    assert restored.research_metrics_result.relative_metrics == (
        persistence_artifact.research_metrics_result.relative_metrics
    )


def test_serialize_deserialize_serialize_is_identical(persistence_artifact):
    first = serialize_research_metrics_artifact(persistence_artifact)
    restored = deserialize_research_metrics_artifact(first)
    assert serialize_research_metrics_artifact(restored) == first


def test_noncanonical_json_is_rejected(persistence_artifact):
    payload = _payload(persistence_artifact)
    pretty = json.dumps(payload, indent=2, ensure_ascii=False)
    with pytest.raises(ResearchMetricsPersistenceError, match="NONCANONICAL_JSON"):
        deserialize_research_metrics_artifact(pretty)


def test_duplicate_json_keys_are_rejected(persistence_artifact):
    canonical = serialize_research_metrics_artifact(persistence_artifact).decode()
    duplicate = canonical.replace(
        '{"artifact_fingerprint":',
        '{"schema_version":"research_metrics_persistence.v0.2","artifact_fingerprint":',
        1,
    )
    with pytest.raises(ResearchMetricsPersistenceError):
        deserialize_research_metrics_artifact(duplicate)


def test_generic_file_helpers_round_trip(tmp_path, persistence_artifact):
    path = tmp_path / "metrics.json"
    write_research_metrics_artifact(path, persistence_artifact)
    assert path.read_bytes() == serialize_research_metrics_artifact(
        persistence_artifact
    )
    assert read_research_metrics_artifact(path) == persistence_artifact


def test_invalid_starting_capital_or_count_relationship_fails_closed(
    authoritative_result, canonical_audit_result, performance_policy
):
    with pytest.raises(ResearchMetricsPersistenceError):
        _artifact_for_result(
            authoritative_result,
            canonical_audit_result,
            performance_policy,
            starting_capital=Decimal("99999"),
        )
    with pytest.raises(ResearchMetricsPersistenceError):
        _artifact_for_result(
            authoritative_result,
            canonical_audit_result,
            performance_policy,
            periodic_return_observation_count=1,
        )


def test_report_projection_is_exact_and_stable_across_round_trip(
    persistence_artifact,
):
    before = project_research_metrics_report(persistence_artifact)
    restored = deserialize_research_metrics_artifact(
        serialize_research_metrics_artifact(persistence_artifact)
    )
    after = project_research_metrics_report(restored)
    result = persistence_artifact.research_metrics_result

    assert before == after
    assert before.schema_version == (
        RESEARCH_METRICS_REPORT_PROJECTION_SCHEMA_VERSION
    )
    assert before.context.decision_interval == persistence_artifact.decision_interval
    assert before.context.starting_capital == persistence_artifact.starting_capital
    assert before.context.policy_ref == persistence_artifact.policy_ref
    assert before.context.source_audit_result_fingerprint == (
        result.provenance.source_audit_result_fingerprint
    )
    assert before.context.benchmark_series_fingerprint == (
        result.provenance.benchmark_series_fingerprint
    )
    assert before.context.research_metrics_result_fingerprint == (
        result.result_fingerprint
    )
    assert before.strategy.ending_equity is result.strategy_metrics.ending_equity
    assert before.strategy.total_return is result.strategy_metrics.total_return
    assert before.benchmark.total_return is result.benchmark_metrics.total_return
    assert before.comparative.ending_wealth_ratio is (
        result.relative_metrics.ending_wealth_ratio
    )


def test_optional_benchmark_and_comparative_absence_survives(
    canonical_audit_result, performance_policy
):
    result = calculate_research_metrics(
        source_result=canonical_audit_result,
        policy=performance_policy,
    )
    artifact = _artifact_for_result(
        result, canonical_audit_result, performance_policy
    )
    restored = deserialize_research_metrics_artifact(
        serialize_research_metrics_artifact(artifact)
    )
    projection = project_research_metrics_report(restored)
    assert restored.research_metrics_result.benchmark_metrics is None
    assert restored.research_metrics_result.relative_metrics is None
    assert projection.context.benchmark_series_fingerprint is None
    assert projection.benchmark is None
    assert projection.comparative is None


def test_persistence_and_reporting_do_not_call_metric_calculation(
    monkeypatch, persistence_artifact
):
    from stock_swing_d1.research_metrics import calculation

    def forbidden_calculation(*args, **kwargs):
        raise AssertionError("metric calculation must not be called")

    monkeypatch.setattr(
        calculation, "calculate_research_metrics", forbidden_calculation
    )
    encoded = serialize_research_metrics_artifact(persistence_artifact)
    restored = deserialize_research_metrics_artifact(encoded)
    project_research_metrics_report(restored)


def test_phase16b3_contains_no_metric_formula_calls_or_equity_reconstruction():
    forbidden_calls = {
        "calculate_research_metrics",
        "quantize_canonical_metric",
        "square_root",
        "natural_logarithm",
        "exponential",
        "divide",
        "multiply",
        "subtract",
        "add",
    }
    forbidden_names = {
        "equity_curve",
        "benchmark_observations",
        "session_results",
        "session_transitions",
        "fills",
        "trades",
        "positions",
        "settlements",
    }
    offenders = []
    for path in PHASE16B3_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = node.func
                name = (
                    called.id
                    if isinstance(called, ast.Name)
                    else called.attr
                    if isinstance(called, ast.Attribute)
                    else None
                )
                if name in forbidden_calls:
                    offenders.append((path.name, "call", name, node.lineno))
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name in forbidden_names:
                    offenders.append((path.name, "name", name, node.lineno))
    assert offenders == []


def test_artifact_contains_no_equity_history_or_volatile_metadata(
    persistence_artifact,
):
    encoded = serialize_research_metrics_artifact(persistence_artifact)
    for token in (
        b'"equity_curve"',
        b'"observations"',
        b'"generated_at"',
        b'"timestamp"',
        b'"uuid"',
        b'"run_id"',
    ):
        assert token not in encoded


def test_no_clock_random_or_presentation_dependencies_in_phase16b3():
    source = "\n".join(path.read_text(encoding="utf-8") for path in PHASE16B3_MODULES)
    lowered = source.lower()
    for token in (
        "datetime.now",
        "datetime.utcnow",
        "date.today",
        "time.time",
        "uuid4",
        "random.",
        "markdown",
        "html",
        "matplotlib",
        "plotly",
    ):
        assert token not in lowered
