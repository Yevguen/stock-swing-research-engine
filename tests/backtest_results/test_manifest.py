from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from stock_swing_d1.backtest_results import (
    HistoricalBacktestRunManifest,
    PolicyArtifactRef,
    build_run_manifest,
)


def _manifest_values(
    artifact_ref_factory,
    valuation_policy_ref,
    ranking_policy_ref,
    execution_cost_policy_ref,
    **overrides,
):
    values = dict(
        software_revision="c" * 40,
        strategy_configuration_ref=artifact_ref_factory("strategy"),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id="allocation_v0.1",
            policy_version="0.1",
            policy_fingerprint="d" * 64,
        ),
        ranking_policy_ref=ranking_policy_ref,
        execution_cost_policy_ref=execution_cost_policy_ref,
        valuation_policy_ref=valuation_policy_ref,
        universe_artifact_ref=artifact_ref_factory("universe"),
        market_data_artifact_ref=artifact_ref_factory("market_data"),
    )
    values.update(overrides)
    return values


def test_manifest_builder_is_deterministic_and_reuses_authoritative_refs(
    artifact_ref_factory,
    valuation_policy_ref,
    ranking_policy_ref,
    execution_cost_policy_ref,
):
    values = _manifest_values(
        artifact_ref_factory,
        valuation_policy_ref,
        ranking_policy_ref,
        execution_cost_policy_ref,
    )
    first = build_run_manifest(**values)
    second = build_run_manifest(**values)
    assert first == second
    assert type(first.ranking_policy_ref) is type(ranking_policy_ref)
    assert type(first.execution_cost_policy_ref) is type(execution_cost_policy_ref)
    assert first.schema_version == "historical_backtest_run_manifest.v0.1"


def test_manifest_model_recomputes_supplied_fingerprint(run_manifest):
    values = run_manifest.model_dump(mode="python")
    values["run_configuration_fingerprint"] = "f" * 64
    with pytest.raises(ValidationError):
        HistoricalBacktestRunManifest.model_validate(values)


@pytest.mark.parametrize(
    "revision",
    ["a" * 39, "a" * 41, "A" * 40, "g" * 40, " a" * 20],
)
def test_manifest_requires_full_lowercase_git_sha(
    artifact_ref_factory,
    valuation_policy_ref,
    ranking_policy_ref,
    execution_cost_policy_ref,
    revision,
):
    with pytest.raises(ValidationError):
        build_run_manifest(
            **_manifest_values(
                artifact_ref_factory,
                valuation_policy_ref,
                ranking_policy_ref,
                execution_cost_policy_ref,
                software_revision=revision,
            )
        )


def test_earnings_artifact_and_provider_are_paired(
    artifact_ref_factory,
    valuation_policy_ref,
    ranking_policy_ref,
    execution_cost_policy_ref,
):
    base = _manifest_values(
        artifact_ref_factory,
        valuation_policy_ref,
        ranking_policy_ref,
        execution_cost_policy_ref,
    )
    assert build_run_manifest(**base).earnings_artifact_ref is None
    present = build_run_manifest(
        **base,
        earnings_artifact_ref=artifact_ref_factory("earnings"),
        earnings_provider_name="EXTERNAL_EARNINGS_PROVIDER",
    )
    assert present.earnings_provider_name == "EXTERNAL_EARNINGS_PROVIDER"
    with pytest.raises(ValidationError):
        build_run_manifest(
            **base,
            earnings_artifact_ref=artifact_ref_factory("earnings"),
        )
    with pytest.raises(ValidationError):
        build_run_manifest(**base, earnings_provider_name="EXTERNAL_EARNINGS_PROVIDER")


def test_manifest_has_no_time_or_path_identity_fields():
    fields = set(HistoricalBacktestRunManifest.model_fields)
    forbidden = {"generated_at", "created_at", "path", "url", "mtime", "machine_name"}
    assert not fields.intersection(forbidden)
    source = inspect.getsource(build_run_manifest)
    assert "git " not in source.lower()
    assert "datetime.now" not in source
