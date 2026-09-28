"""Task 5B: Phase 16C Gate D closed by real repository strategy evidence.

Phase 16C production code is unchanged. The frozen ``StrategyConfigurationEvidence``
model already accepts the Task 5B artifact cleanly, because the strategy-owned
reference deliberately carries the same semantic identity fields as the
Phase 15D ``ArtifactRef``. These tests prove Gate D passes on genuine
repository evidence, still fails closed on every defective variant, and does
not accidentally unblock any other gate.

Nothing here executes the canonical baseline experiment: the preflight audit is
a pure function of supplied evidence and never invokes a producer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stock_swing_d1.backtest_results.models import ArtifactRef
from stock_swing_d1.baseline_experiment.preflight import (
    CanonicalRunPreflightInput,
    CanonicalRunPreflightVerdict,
    PreflightGateStatus,
    StrategyConfigurationEvidence,
    evaluate_canonical_run_preflight,
)
from stock_swing_d1.earnings.integration.service import _MAX_HOLDING_SESSIONS
from stock_swing_d1.execution.open_position_exit.models import (
    MAX_HOLDING_SESSIONS,
)
from stock_swing_d1.risk.position_sizing.service import TARGET_RISK_FRACTION
from stock_swing_d1.strategy.baseline import (
    BaselineStrategyConfigurationError,
    build_baseline_strategy_configuration,
    build_baseline_strategy_configuration_ref,
    load_declared_baseline_strategy_configuration,
    parse_declared_baseline_strategy_configuration,
)


STRATEGY_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "strategy.yaml"
)
AUTHORITATIVE_INJECTIONS = {
    "target_risk_fraction": TARGET_RISK_FRACTION,
    "maximum_holding_sessions": MAX_HOLDING_SESSIONS,
    "earnings_maximum_holding_sessions": _MAX_HOLDING_SESSIONS,
}


def _artifact_ref(configuration) -> ArtifactRef:
    """Map the strategy-owned reference onto the Phase 15D identity model."""

    reference = build_baseline_strategy_configuration_ref(configuration)
    return ArtifactRef(
        artifact_type=reference.artifact_type,
        schema_version=reference.schema_version,
        content_sha256=reference.content_sha256,
    )


def _gate(report, gate_id: str):
    return next(gate for gate in report.gates if gate.gate_id == gate_id)


@pytest.fixture
def declared_configuration():
    return load_declared_baseline_strategy_configuration(
        STRATEGY_CONFIG_PATH, **AUTHORITATIVE_INJECTIONS
    )


def test_gate_d_passes_on_real_repository_strategy_evidence(
    declared_configuration,
) -> None:
    evidence = StrategyConfigurationEvidence(
        artifact_ref=_artifact_ref(declared_configuration),
        configuration_frozen=True,
        tuned_parameter_count=declared_configuration.tuned_parameter_count,
    )
    report = evaluate_canonical_run_preflight(
        CanonicalRunPreflightInput(strategy_configuration_evidence=evidence)
    )
    gate_d = _gate(report, "D")
    assert gate_d.status is PreflightGateStatus.PASS
    assert gate_d.blockers == ()
    assert (
        _artifact_ref(declared_configuration).content_sha256 in gate_d.evidence
    )


def test_declared_and_built_artifacts_produce_one_gate_d_identity(
    declared_configuration,
) -> None:
    built = build_baseline_strategy_configuration()
    assert _artifact_ref(declared_configuration) == _artifact_ref(built)


def test_gate_d_is_blocked_without_strategy_evidence() -> None:
    report = evaluate_canonical_run_preflight(CanonicalRunPreflightInput())
    gate_d = _gate(report, "D")
    assert gate_d.status is PreflightGateStatus.BLOCKED
    assert gate_d.blockers == (
        "the frozen baseline strategy configuration artifact is unavailable",
    )


def test_gate_d_is_blocked_when_the_configuration_is_not_frozen(
    declared_configuration,
) -> None:
    evidence = StrategyConfigurationEvidence(
        artifact_ref=_artifact_ref(declared_configuration),
        configuration_frozen=False,
        tuned_parameter_count=0,
    )
    gate_d = _gate(
        evaluate_canonical_run_preflight(
            CanonicalRunPreflightInput(strategy_configuration_evidence=evidence)
        ),
        "D",
    )
    assert gate_d.status is PreflightGateStatus.BLOCKED
    assert "not frozen" in " ".join(gate_d.blockers)


def test_gate_d_is_blocked_when_tuned_parameters_are_claimed(
    declared_configuration,
) -> None:
    evidence = StrategyConfigurationEvidence(
        artifact_ref=_artifact_ref(declared_configuration),
        configuration_frozen=True,
        tuned_parameter_count=1,
    )
    gate_d = _gate(
        evaluate_canonical_run_preflight(
            CanonicalRunPreflightInput(strategy_configuration_evidence=evidence)
        ),
        "D",
    )
    assert gate_d.status is PreflightGateStatus.BLOCKED
    assert "tuned parameters" in " ".join(gate_d.blockers)


def test_a_malformed_fingerprint_cannot_become_gate_d_evidence() -> None:
    with pytest.raises(ValueError):
        ArtifactRef(
            artifact_type="baseline_strategy_configuration",
            schema_version="baseline_strategy_configuration.v0.1",
            content_sha256="not-a-sha256",
        )


def test_a_configuration_disagreeing_with_code_yields_no_evidence() -> None:
    """Gate-D evidence cannot be produced from a drifted declaration at all."""

    text = STRATEGY_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "rsi_minimum: 50.0", "rsi_minimum: 60.0"
    )
    with pytest.raises(BaselineStrategyConfigurationError):
        parse_declared_baseline_strategy_configuration(
            text, **AUTHORITATIVE_INJECTIONS
        )


def test_gate_d_evidence_does_not_unblock_any_other_gate(
    declared_configuration,
) -> None:
    """Task 5B closes exactly one gate and invents no other evidence."""

    evidence = StrategyConfigurationEvidence(
        artifact_ref=_artifact_ref(declared_configuration),
        configuration_frozen=True,
        tuned_parameter_count=0,
    )
    report = evaluate_canonical_run_preflight(
        CanonicalRunPreflightInput(strategy_configuration_evidence=evidence)
    )
    assert report.verdict is CanonicalRunPreflightVerdict.BLOCKED
    passing = {
        gate.gate_id
        for gate in report.gates
        if gate.status is PreflightGateStatus.PASS
    }
    assert passing == {"D"}
