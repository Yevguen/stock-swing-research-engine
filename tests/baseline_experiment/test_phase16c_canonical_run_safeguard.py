"""The exactly-once canonical baseline run cannot be reached by accident.

Every invocation below is synthetic. A synthetic invocation of the preservation
entry point does not consume the canonical baseline experiment: what consumes it
is producing the *canonical* authoritative artifacts from canonical historical
data upstream, and no test here does or can do that.
"""

from __future__ import annotations

import pytest

from stock_swing_d1.baseline_experiment import (
    CANONICAL_RUN_AUTHORIZATION_TOKEN,
    CanonicalBaselineExperimentArtifacts,
    CanonicalBaselineExperimentResult,
    CanonicalRunAuthorization,
    CanonicalRunAuthorizationError,
    CanonicalRunPreflightInput,
    CanonicalRunPreflightVerdict,
    build_earnings_filter_exclusion_diagnostic,
    build_material_adverse_overnight_gap_diagnostic,
    build_material_adverse_overnight_gap_policy,
    evaluate_canonical_run_preflight,
    execute_canonical_baseline_experiment,
)

from tests.baseline_experiment.conftest import (
    build_ready_preflight_input,
    build_synthetic_audit_result,
    build_synthetic_manifest,
    build_synthetic_metrics_artifact,
)


def synthetic_artifacts():
    source_result = build_synthetic_audit_result()
    return CanonicalBaselineExperimentArtifacts(
        source_audit_result_fingerprint=source_result.result_fingerprint,
        research_metrics_artifact=build_synthetic_metrics_artifact(source_result),
        gap_diagnostic=build_material_adverse_overnight_gap_diagnostic(
            source_result=source_result,
            policy=build_material_adverse_overnight_gap_policy(),
            trade_gap_evidence=(),
        ),
        earnings_exclusion_diagnostic=build_earnings_filter_exclusion_diagnostic(
            source_result=source_result
        ),
    )


class SyntheticProducer:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return synthetic_artifacts()


@pytest.fixture
def ready_pair():
    preflight_input = build_ready_preflight_input()
    manifest = preflight_input.manifest
    report = evaluate_canonical_run_preflight(preflight_input)
    assert report.verdict is (
        CanonicalRunPreflightVerdict.READY_FOR_CANONICAL_RUN
    )
    return report, manifest


def authorization(manifest, **overrides):
    values = {
        "token": CANONICAL_RUN_AUTHORIZATION_TOKEN,
        "acknowledged_exactly_once": True,
        "authorizing_git_commit_sha": manifest.git_commit_sha,
    }
    values.update(overrides)
    return CanonicalRunAuthorization(**values)


def test_no_authorization_refuses(ready_pair):
    report, manifest = ready_pair
    producer = SyntheticProducer()
    with pytest.raises(CanonicalRunAuthorizationError) as error:
        execute_canonical_baseline_experiment(
            authorization=None,
            preflight_report=report,
            manifest=manifest, canonical_artifact_producer=producer,
        )
    assert error.value.code == "MISSING_CANONICAL_RUN_AUTHORIZATION"
    assert producer.calls == 0


def test_a_wrong_token_never_invokes_the_producer(ready_pair):
    report, manifest = ready_pair
    producer = SyntheticProducer()
    with pytest.raises(ValueError):
        authorization(manifest, token="PLEASE_JUST_RUN_IT")
    forged = CanonicalRunAuthorization.model_construct(
        schema_version="canonical_run_authorization.v0.1",
        token="PLEASE_JUST_RUN_IT",
        acknowledged_exactly_once=True,
        authorizing_git_commit_sha=manifest.git_commit_sha,
    )
    with pytest.raises(CanonicalRunAuthorizationError) as error:
        execute_canonical_baseline_experiment(
            authorization=forged,
            preflight_report=report,
            manifest=manifest,
            canonical_artifact_producer=producer,
        )
    assert error.value.code == "INVALID_CANONICAL_RUN_AUTHORIZATION_TOKEN"
    assert producer.calls == 0


def test_an_unacknowledged_authorization_refuses(ready_pair):
    report, manifest = ready_pair
    producer = SyntheticProducer()
    with pytest.raises(CanonicalRunAuthorizationError) as error:
        execute_canonical_baseline_experiment(
            authorization=authorization(
                manifest, acknowledged_exactly_once=False
            ),
            preflight_report=report,
            manifest=manifest, canonical_artifact_producer=producer,
        )
    assert error.value.code == "EXACTLY_ONCE_NOT_ACKNOWLEDGED"
    assert producer.calls == 0


def test_a_missing_preflight_audit_refuses(ready_pair):
    _, manifest = ready_pair
    producer = SyntheticProducer()
    with pytest.raises(CanonicalRunAuthorizationError) as error:
        execute_canonical_baseline_experiment(
            authorization=authorization(manifest),
            preflight_report=None,
            manifest=manifest, canonical_artifact_producer=producer,
        )
    assert error.value.code == "MISSING_PRERUN_AUDIT"
    assert producer.calls == 0


def test_a_blocked_preflight_audit_refuses_and_names_the_blockers():
    manifest = build_synthetic_manifest()
    blocked = evaluate_canonical_run_preflight(CanonicalRunPreflightInput())
    producer = SyntheticProducer()
    with pytest.raises(CanonicalRunAuthorizationError) as error:
        execute_canonical_baseline_experiment(
            authorization=authorization(manifest),
            preflight_report=blocked,
            manifest=manifest, canonical_artifact_producer=producer,
        )
    assert error.value.code == "PRERUN_AUDIT_BLOCKED"
    assert "canonical earnings PIT artifact unavailable" in str(error.value)
    assert producer.calls == 0


def test_an_audit_of_a_different_manifest_refuses(ready_pair):
    report, _ = ready_pair
    other_manifest = build_synthetic_manifest(git_commit_sha="c" * 40)
    producer = SyntheticProducer()
    with pytest.raises(CanonicalRunAuthorizationError) as error:
        execute_canonical_baseline_experiment(
            authorization=authorization(other_manifest),
            preflight_report=report,
            manifest=other_manifest, canonical_artifact_producer=producer,
        )
    assert error.value.code == "PRERUN_AUDIT_MANIFEST_MISMATCH"
    assert producer.calls == 0


def test_an_authorization_for_another_commit_refuses(ready_pair):
    report, manifest = ready_pair
    producer = SyntheticProducer()
    with pytest.raises(CanonicalRunAuthorizationError) as error:
        execute_canonical_baseline_experiment(
            authorization=authorization(
                manifest, authorizing_git_commit_sha="d" * 40
            ),
            preflight_report=report,
            manifest=manifest, canonical_artifact_producer=producer,
        )
    assert error.value.code == "AUTHORIZED_COMMIT_MISMATCH"
    assert producer.calls == 0


def test_a_fully_authorized_synthetic_invocation_preserves_the_result(
    ready_pair,
):
    report, manifest = ready_pair
    producer = SyntheticProducer()
    result = execute_canonical_baseline_experiment(
        authorization=authorization(manifest),
        preflight_report=report,
        manifest=manifest, canonical_artifact_producer=producer,
    )
    assert type(result) is CanonicalBaselineExperimentResult
    assert result.manifest == manifest
    assert len(result.result_fingerprint) == 64
    assert producer.calls == 1


def test_a_producer_failure_is_not_retried(ready_pair):
    report, manifest = ready_pair

    class FailingSyntheticProducer:
        calls = 0

        def __call__(self):
            self.calls += 1
            raise RuntimeError("synthetic producer failure")

    producer = FailingSyntheticProducer()
    with pytest.raises(RuntimeError, match="synthetic producer failure"):
        execute_canonical_baseline_experiment(
            authorization=authorization(manifest),
            preflight_report=report,
            manifest=manifest,
            canonical_artifact_producer=producer,
        )
    assert producer.calls == 1


def test_the_entry_point_runs_no_upstream_owner():
    """The gated entry point preserves; it does not execute the pipeline."""

    from stock_swing_d1.baseline_experiment import execution as execution_module

    source = execution_module.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for token in (
        "HistoricalBacktestOrchestrator",
        "PortfolioTransitionEngine",
        "PortfolioBacktestOrchestrator",
        "BacktestExecutionCostService",
        "CandidateRankingService",
        "BaselineSignalEvaluator",
        "run_d1_pipeline",
        "run_corporate_action_pipeline",
        "run_dividend_event_pipeline",
        "norgatedata",
    ):
        assert token not in text


def test_importing_the_package_executes_nothing(ready_pair):
    """Importing Phase 16C has no side effect that could consume the run."""

    import importlib

    module = importlib.import_module("stock_swing_d1.baseline_experiment")
    assert module.CANONICAL_RUN_AUTHORIZATION_TOKEN == (
        "PHASE16C_CANONICAL_BASELINE_RUN_AUTHORIZED"
    )
    assert callable(module.execute_canonical_baseline_experiment)
