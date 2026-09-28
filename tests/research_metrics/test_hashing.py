from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal

from stock_swing_d1.research_metrics import (
    BENCHMARK_PERFORMANCE_SERIES_HASH_DOMAIN,
    BENCHMARK_PERFORMANCE_SERIES_SCHEMA_VERSION,
    PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN,
    RESEARCH_METRICS_RESULT_HASH_DOMAIN,
    RESEARCH_METRICS_RESULT_SCHEMA_VERSION,
    BenchmarkPerformanceObservation,
    BenchmarkSourceArtifactRef,
    ResearchMetricsProvenance,
    build_benchmark_performance_series,
    build_research_metrics_result,
    compute_benchmark_performance_series_fingerprint,
    compute_research_metrics_result_fingerprint,
    defined_metric,
    semantic_domain_sha256,
    semantic_sha256,
)
from stock_swing_d1.research_metrics.hashing import (
    _compute_benchmark_performance_series_fingerprint,
)


BENCHMARK_SEMANTIC_KEYS = (
    "schema_version",
    "benchmark_id",
    "source_artifact_ref",
    "observations",
)


def test_hash_domains_are_explicit_and_distinct():
    assert len(
        {
            PERFORMANCE_MEASUREMENT_POLICY_HASH_DOMAIN,
            BENCHMARK_PERFORMANCE_SERIES_HASH_DOMAIN,
            RESEARCH_METRICS_RESULT_HASH_DOMAIN,
        }
    ) == 3


def test_benchmark_schema_and_hash_domain_advance_to_v0_2():
    assert BENCHMARK_PERFORMANCE_SERIES_SCHEMA_VERSION == (
        "benchmark_performance_series.v0.2"
    )
    assert BENCHMARK_PERFORMANCE_SERIES_HASH_DOMAIN == (
        "benchmark_performance_series.v0.2"
    )


def test_result_schema_and_hash_domain_advance_to_v0_3():
    """The amendment expanded the result payload, so its identity moves again."""

    assert RESEARCH_METRICS_RESULT_SCHEMA_VERSION == (
        "research_metrics_result.v0.3"
    )
    assert RESEARCH_METRICS_RESULT_HASH_DOMAIN == "research_metrics_result.v0.3"
    assert RESEARCH_METRICS_RESULT_HASH_DOMAIN != "research_metrics_result.v0.1"
    assert RESEARCH_METRICS_RESULT_HASH_DOMAIN != "research_metrics_result.v0.2"


def test_hashing_ignores_mapping_insertion_order_and_object_identity():
    left = {"a": Decimal("1.00"), "b": (1, 2)}
    right = {"b": (1, 2), "a": Decimal("1")}
    assert semantic_sha256(left) == semantic_sha256(right)


def test_benchmark_fingerprint_is_deterministic_and_content_sensitive(
    benchmark_series, benchmark_source_ref
):
    assert benchmark_series.content_fingerprint == (
        compute_benchmark_performance_series_fingerprint(benchmark_series)
    )
    rebuilt = build_benchmark_performance_series(
        benchmark_id=benchmark_series.benchmark_id,
        source_artifact_ref=benchmark_source_ref,
        observations=benchmark_series.observations,
    )
    assert rebuilt.content_fingerprint == benchmark_series.content_fingerprint

    changed = build_benchmark_performance_series(
        benchmark_id=benchmark_series.benchmark_id,
        source_artifact_ref=benchmark_source_ref,
        observations=(
            benchmark_series.observations[0],
            BenchmarkPerformanceObservation(
                session=date(2025, 1, 3), value=Decimal("101250.76")
            ),
        ),
    )
    assert changed.content_fingerprint != benchmark_series.content_fingerprint


def test_benchmark_fingerprint_binds_source_provenance(benchmark_series):
    changed_ref = BenchmarkSourceArtifactRef(
        artifact_id="synthetic-benchmark-evidence",
        schema_version="synthetic-index.v0.1",
        content_sha256="b" * 64,
        build_id="synthetic-fixture",
    )
    changed = build_benchmark_performance_series(
        benchmark_id=benchmark_series.benchmark_id,
        source_artifact_ref=changed_ref,
        observations=benchmark_series.observations,
    )
    assert changed.content_fingerprint != benchmark_series.content_fingerprint


def test_benchmark_fingerprint_binds_benchmark_identity(
    benchmark_series, benchmark_source_ref
):
    changed = build_benchmark_performance_series(
        benchmark_id="SYNTHETIC_OTHER_BENCHMARK_V0_1",
        source_artifact_ref=benchmark_source_ref,
        observations=benchmark_series.observations,
    )
    assert changed.content_fingerprint != benchmark_series.content_fingerprint


def test_benchmark_fingerprint_payload_has_exactly_the_four_semantic_keys(
    benchmark_series,
):
    payload = {
        "schema_version": "benchmark_performance_series.v0.2",
        "benchmark_id": benchmark_series.benchmark_id,
        "source_artifact_ref": benchmark_series.source_artifact_ref,
        "observations": benchmark_series.observations,
    }

    assert tuple(payload) == BENCHMARK_SEMANTIC_KEYS
    assert tuple(
        inspect.signature(
            _compute_benchmark_performance_series_fingerprint
        ).parameters
    ) == BENCHMARK_SEMANTIC_KEYS
    assert benchmark_series.content_fingerprint == semantic_domain_sha256(
        BENCHMARK_PERFORMANCE_SERIES_HASH_DOMAIN, payload
    )


def test_benchmark_fingerprint_carries_no_value_kind_or_null_residue(
    benchmark_series,
):
    payload = {
        "schema_version": "benchmark_performance_series.v0.2",
        "benchmark_id": benchmark_series.benchmark_id,
        "source_artifact_ref": benchmark_series.source_artifact_ref,
        "observations": benchmark_series.observations,
    }
    residues = (
        {"value_kind": None},
        {"value_kind": "total_return_index"},
        {"value_kind": "price_index"},
        {"benchmark_type": None},
        {"equity_kind": None},
        {"return_kind": None},
        {"total_return": None},
        {"price_index": None},
    )
    for residue in residues:
        assert benchmark_series.content_fingerprint != semantic_domain_sha256(
            BENCHMARK_PERFORMANCE_SERIES_HASH_DOMAIN, {**payload, **residue}
        )


def test_result_fingerprint_is_deterministic_and_binds_provenance(
    synthetic_audit_result,
    performance_policy,
    benchmark_series,
    strategy_metrics,
    trade_metrics,
    benchmark_metrics,
    relative_metrics,
):
    provenance = ResearchMetricsProvenance(
        source_audit_result_fingerprint=(
            synthetic_audit_result.result_fingerprint
        ),
        performance_measurement_policy_fingerprint=(
            performance_policy.policy_fingerprint
        ),
        benchmark_series_fingerprint=benchmark_series.content_fingerprint,
    )
    values = {
        "provenance": provenance,
        "strategy_metrics": strategy_metrics,
        "trade_metrics": trade_metrics,
        "benchmark_metrics": benchmark_metrics,
        "relative_metrics": relative_metrics,
    }
    first = build_research_metrics_result(**values)
    second = build_research_metrics_result(**values)
    assert first.result_fingerprint == second.result_fingerprint
    assert first.result_fingerprint == (
        compute_research_metrics_result_fingerprint(first)
    )
    changed_provenance = provenance.model_copy(
        update={"source_audit_result_fingerprint": "f" * 64}
    )
    changed = build_research_metrics_result(
        **{**values, "provenance": changed_provenance}
    )
    assert changed.result_fingerprint != first.result_fingerprint


def test_result_fingerprint_is_sensitive_to_every_metric_group(
    synthetic_audit_result,
    performance_policy,
    benchmark_series,
    strategy_metrics,
    trade_metrics,
    benchmark_metrics,
    relative_metrics,
):
    """A populated child metric is fingerprint-bearing, not decorative."""

    provenance = ResearchMetricsProvenance(
        source_audit_result_fingerprint=(
            synthetic_audit_result.result_fingerprint
        ),
        performance_measurement_policy_fingerprint=(
            performance_policy.policy_fingerprint
        ),
        benchmark_series_fingerprint=benchmark_series.content_fingerprint,
    )
    values = {
        "provenance": provenance,
        "strategy_metrics": strategy_metrics,
        "trade_metrics": trade_metrics,
        "benchmark_metrics": benchmark_metrics,
        "relative_metrics": relative_metrics,
    }
    baseline = build_research_metrics_result(**values)
    mutations = {
        "strategy_metrics": strategy_metrics.model_copy(
            update={"ending_equity": Decimal("110001")}
        ),
        "trade_metrics": trade_metrics.model_copy(update={"win_count": 2}),
        "benchmark_metrics": benchmark_metrics.model_copy(
            update={"ending_equity": Decimal("104001")}
        ),
        "relative_metrics": relative_metrics.model_copy(
            update={
                "ending_wealth_ratio": defined_metric(Decimal("1.5")),
            }
        ),
    }
    for name, mutated in mutations.items():
        changed = build_research_metrics_result(**{**values, name: mutated})
        assert changed.result_fingerprint != baseline.result_fingerprint
