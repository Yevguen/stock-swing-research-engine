"""Phase 16B.2 cross-process determinism (VER-08/VER-09).

Identical canonical inputs must produce identical canonical serialization and
identical fingerprints, including in a separate interpreter process with a
different hash seed and a different starting ambient Decimal context.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from stock_swing_d1.research_metrics import (
    calculate_research_metrics,
    semantic_json_bytes,
)


ROOT = Path(__file__).resolve().parents[2]

_CHILD_PROGRAM = """
import decimal
import sys

sys.path.insert(0, {conftest_dir!r})

from conftest import canonical_experiment, benchmark_evidence, CANONICAL_SESSIONS, CANONICAL_BENCHMARK_VALUES
from stock_swing_d1.research_metrics import (
    BenchmarkSourceArtifactRef,
    build_performance_measurement_policy,
    calculate_research_metrics,
    semantic_json_bytes,
)

decimal.setcontext(decimal.Context(prec={prec}))

series = benchmark_evidence(
    BenchmarkSourceArtifactRef(
        artifact_id="synthetic-benchmark-evidence",
        schema_version="synthetic-index.v0.1",
        content_sha256="a" * 64,
        build_id="synthetic-fixture",
    ),
    CANONICAL_SESSIONS,
    CANONICAL_BENCHMARK_VALUES,
)
result = calculate_research_metrics(
    source_result=canonical_experiment(),
    policy=build_performance_measurement_policy(),
    benchmark_series=series,
)
sys.stdout.write(result.result_fingerprint)
sys.stdout.write("\\n")
sys.stdout.write(semantic_json_bytes(result).decode("utf-8"))
"""


def _run_child(*, seed: str, prec: int) -> tuple[str, str]:
    program = _CHILD_PROGRAM.format(
        conftest_dir=str(Path(__file__).parent), prec=prec
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={
            **_child_environment(),
            "PYTHONHASHSEED": seed,
        },
        check=True,
    )
    fingerprint, _, payload = completed.stdout.partition("\n")
    return fingerprint, payload


def _child_environment() -> dict[str, str]:
    import os

    environment = dict(os.environ)
    environment.pop("PYTHONHASHSEED", None)
    return environment


def test_identical_inputs_serialize_identically_in_this_process(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """VER-08."""

    first = calculate_research_metrics(
        source_result=canonical_audit_result,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )
    second = calculate_research_metrics(
        source_result=canonical_audit_result,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )

    assert semantic_json_bytes(first) == semantic_json_bytes(second)
    assert first.result_fingerprint == second.result_fingerprint


def test_identical_inputs_produce_identical_fingerprints_across_processes(
    canonical_audit_result, canonical_benchmark_series, performance_policy
):
    """VER-09: different processes, hash seeds and ambient precisions agree."""

    in_process = calculate_research_metrics(
        source_result=canonical_audit_result,
        policy=performance_policy,
        benchmark_series=canonical_benchmark_series,
    )

    first_fingerprint, first_payload = _run_child(seed="1", prec=28)
    second_fingerprint, second_payload = _run_child(seed="12345", prec=200)

    assert first_fingerprint == second_fingerprint
    assert first_payload == second_payload
    assert first_fingerprint == in_process.result_fingerprint
    assert first_payload == semantic_json_bytes(in_process).decode("utf-8")
