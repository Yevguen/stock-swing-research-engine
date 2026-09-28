from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

import stock_swing_d1.research_metrics as research_metrics
from stock_swing_d1.research_metrics import (
    BenchmarkPerformanceObservation,
    BenchmarkPerformanceSeries,
    BenchmarkSourceArtifactRef,
    ResearchMetricsValidationError,
    build_benchmark_performance_series,
    semantic_json_bytes,
    validate_research_metrics_inputs,
)


def test_valid_boundary_does_not_mutate_source(
    synthetic_audit_result, performance_policy, benchmark_series
):
    before = semantic_json_bytes(synthetic_audit_result)
    validate_research_metrics_inputs(
        source_result=synthetic_audit_result,
        policy=performance_policy,
        benchmark_series=benchmark_series,
    )
    assert semantic_json_bytes(synthetic_audit_result) == before


def test_malformed_source_result_fails_closed(
    synthetic_audit_result, performance_policy
):
    malformed = synthetic_audit_result.model_copy(
        update={"result_fingerprint": "f" * 64}
    )
    with pytest.raises(ResearchMetricsValidationError) as captured:
        validate_research_metrics_inputs(
            source_result=malformed, policy=performance_policy
        )
    assert captured.value.code == "INVALID_SOURCE_RESULT"


def test_unsatisfactory_source_audit_fails_closed(
    synthetic_audit_result, performance_policy
):
    failed_summary = synthetic_audit_result.audit_summary.model_copy(
        update={"audit_passed": False}
    )
    malformed = synthetic_audit_result.model_copy(
        update={"audit_summary": failed_summary}
    )
    with pytest.raises(ResearchMetricsValidationError):
        validate_research_metrics_inputs(
            source_result=malformed, policy=performance_policy
        )


def test_source_requires_the_exact_phase15d_result_type(performance_policy):
    with pytest.raises(ResearchMetricsValidationError) as captured:
        validate_research_metrics_inputs(
            source_result={}, policy=performance_policy
        )
    assert captured.value.code == "INVALID_SOURCE_RESULT"


def test_malformed_policy_and_benchmark_fingerprints_fail_closed(
    synthetic_audit_result, performance_policy, benchmark_series
):
    malformed_policy = performance_policy.model_copy(
        update={"annualization_periods_per_year": 365}
    )  # model_copy skips validation, so the boundary must catch it
    with pytest.raises(ResearchMetricsValidationError) as captured:
        validate_research_metrics_inputs(
            source_result=synthetic_audit_result,
            policy=malformed_policy,
        )
    assert captured.value.code == "INVALID_POLICY"

    malformed_benchmark = benchmark_series.model_copy(
        update={"content_fingerprint": "f" * 64}
    )
    with pytest.raises(ResearchMetricsValidationError) as captured:
        validate_research_metrics_inputs(
            source_result=synthetic_audit_result,
            policy=performance_policy,
            benchmark_series=malformed_benchmark,
        )
    assert captured.value.code == "INVALID_BENCHMARK_SERIES"


def test_benchmark_series_is_built_and_immutable_without_a_value_kind(
    benchmark_series, benchmark_source_ref
):
    assert "value_kind" not in BenchmarkPerformanceSeries.model_fields
    assert tuple(BenchmarkPerformanceSeries.model_fields) == (
        "schema_version",
        "benchmark_id",
        "source_artifact_ref",
        "observations",
        "content_fingerprint",
    )
    assert type(benchmark_series.observations) is tuple
    with pytest.raises(ValidationError):
        benchmark_series.benchmark_id = "SYNTHETIC_OTHER_BENCHMARK_V0_1"
    with pytest.raises(ValidationError):
        BenchmarkPerformanceSeries(
            **benchmark_series.model_dump(mode="python"), unexpected=True
        )


def test_v0_2_has_no_value_kind_symbol_or_construction_path(
    benchmark_source_ref, benchmark_series
):
    assert not hasattr(research_metrics, "BenchmarkValueKind")
    assert "BenchmarkValueKind" not in research_metrics.__all__
    assert "value_kind" not in inspect.signature(
        build_benchmark_performance_series
    ).parameters

    for legacy in ("price_index", "total_return_index"):
        with pytest.raises(TypeError):
            build_benchmark_performance_series(
                benchmark_id=benchmark_series.benchmark_id,
                value_kind=legacy,
                source_artifact_ref=benchmark_source_ref,
                observations=benchmark_series.observations,
            )


def test_legacy_v0_1_value_kind_payload_fails_closed(benchmark_series):
    legacy_payload = {
        **benchmark_series.model_dump(mode="python"),
        "value_kind": "total_return_index",
    }
    with pytest.raises(ValidationError):
        BenchmarkPerformanceSeries.model_validate(legacy_payload)

    superseded_schema = {
        **benchmark_series.model_dump(mode="python"),
        "schema_version": "benchmark_performance_series.v0.1",
    }
    with pytest.raises(ValidationError):
        BenchmarkPerformanceSeries.model_validate(superseded_schema)


def test_benchmark_requires_canonical_order(benchmark_source_ref):
    observations = (
        BenchmarkPerformanceObservation(
            session=date(2025, 1, 3), value=Decimal("101000")
        ),
        BenchmarkPerformanceObservation(
            session=date(2025, 1, 2), value=Decimal("100000")
        ),
    )
    with pytest.raises(ValidationError):
        build_benchmark_performance_series(
            benchmark_id="SYNTHETIC_PRIMARY_BENCHMARK_V0_1",
            source_artifact_ref=benchmark_source_ref,
            observations=observations,
        )


def test_benchmark_rejects_duplicate_sessions(benchmark_source_ref):
    observations = tuple(
        BenchmarkPerformanceObservation(
            session=date(2025, 1, 2), value=value
        )
        for value in (Decimal("100000"), Decimal("101000"))
    )
    with pytest.raises(ValidationError):
        build_benchmark_performance_series(
            benchmark_id="SYNTHETIC_PRIMARY_BENCHMARK_V0_1",
            source_artifact_ref=benchmark_source_ref,
            observations=observations,
        )


def test_benchmark_rejects_non_tuple_observations(benchmark_source_ref):
    observations = [
        BenchmarkPerformanceObservation(
            session=date(2025, 1, 2), value=Decimal("100000")
        )
    ]
    with pytest.raises(ValidationError):
        build_benchmark_performance_series(
            benchmark_id="SYNTHETIC_PRIMARY_BENCHMARK_V0_1",
            source_artifact_ref=benchmark_source_ref,
            observations=observations,
        )


@pytest.mark.parametrize("invalid", ("", "  ", " SPY", 1, None))
def test_benchmark_rejects_malformed_benchmark_identity(
    invalid, benchmark_source_ref, benchmark_series
):
    with pytest.raises(ValidationError):
        build_benchmark_performance_series(
            benchmark_id=invalid,
            source_artifact_ref=benchmark_source_ref,
            observations=benchmark_series.observations,
        )


@pytest.mark.parametrize(
    "invalid",
    (
        Decimal("0"),
        Decimal("-1"),
        Decimal("NaN"),
        Decimal("Infinity"),
        100.0,
        "100",
    ),
)
def test_benchmark_rejects_invalid_nonfinite_or_nondecimal_values(invalid):
    with pytest.raises(ValidationError):
        BenchmarkPerformanceObservation(
            session=date(2025, 1, 2), value=invalid
        )


def test_benchmark_rejects_malformed_source_provenance():
    with pytest.raises(ValidationError):
        BenchmarkSourceArtifactRef(
            artifact_id="synthetic-benchmark-evidence",
            schema_version="synthetic-index.v0.1",
            content_sha256="not-a-sha256",
        )
