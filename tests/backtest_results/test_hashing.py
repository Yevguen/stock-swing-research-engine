from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    HistoricalBacktestContentFingerprints,
    compute_content_fingerprint,
    compute_result_fingerprint,
    compute_source_run_fingerprint,
    semantic_domain_sha256,
    semantic_sha256,
)
from stock_swing_d1.backtester import (
    HistoricalBacktestRunResult,
    HistoricalDecisionInterval,
)
from stock_swing_d1.portfolio.portfolio_hashing import hash_portfolio_state
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState


def test_semantic_and_domain_hashes_are_deterministic_lowercase_sha256():
    value = {"amount": Decimal("10.00"), "items": (1, 2)}
    digest = semantic_sha256(value)
    assert digest == semantic_sha256(value)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert semantic_domain_sha256("one.v0.1", value) != semantic_domain_sha256(
        "two.v0.1", value
    )


def test_content_hash_binds_name_content_and_order():
    rows = ({"id": "a", "amount": Decimal("1")}, {"id": "b", "amount": Decimal("2")})
    baseline = compute_content_fingerprint(artifact_name="entries", rows_or_value=rows)
    assert baseline == compute_content_fingerprint(
        artifact_name="entries", rows_or_value=rows
    )
    assert baseline != compute_content_fingerprint(
        artifact_name="exits", rows_or_value=rows
    )
    assert baseline != compute_content_fingerprint(
        artifact_name="entries", rows_or_value=tuple(reversed(rows))
    )
    assert baseline != compute_content_fingerprint(
        artifact_name="entries",
        rows_or_value=({"id": "a", "amount": Decimal("1.01")}, rows[1]),
    )


@dataclass(frozen=True)
class EqualValue:
    identifier: str


def test_hash_does_not_depend_on_object_identity():
    assert semantic_sha256(EqualValue("x")) == semantic_sha256(EqualValue("x"))


def test_result_hash_changes_with_each_semantic_component(content_fingerprints):
    interval = HistoricalDecisionInterval(
        decision_start_date=date(2024, 1, 1),
        decision_end_date=date(2024, 12, 31),
    )
    base = {
        "schema_version": "historical_backtest_audit_result.v0.2",
        "decision_interval": interval,
        "run_configuration_fingerprint": "a" * 64,
        "source_run_fingerprint": "b" * 64,
        "initial_state_fingerprint": "c" * 64,
        "final_state_fingerprint": "d" * 64,
        "content_fingerprints": content_fingerprints,
    }
    original = compute_result_fingerprint(**base)
    for name, replacement in (
        (
            "decision_interval",
            HistoricalDecisionInterval(
                decision_start_date=date(2024, 1, 2),
                decision_end_date=date(2024, 12, 31),
            ),
        ),
        ("run_configuration_fingerprint", "e" * 64),
        ("source_run_fingerprint", "e" * 64),
        ("initial_state_fingerprint", "e" * 64),
        ("final_state_fingerprint", "e" * 64),
    ):
        changed = dict(base)
        changed[name] = replacement
        assert compute_result_fingerprint(**changed) != original
    changed_content = content_fingerprints.model_copy(
        update={"entries": "f" * 64}
    )
    assert compute_result_fingerprint(
        **{**base, "content_fingerprints": changed_content}
    ) != original


def test_source_run_fingerprint_uses_genuine_run_result_without_derivation():
    state = PortfolioState(settled_cash=Decimal("1000"))
    state_hash = hash_portfolio_state(state)
    run = HistoricalBacktestRunResult(
        decision_interval=HistoricalDecisionInterval(
            decision_start_date=date(2024, 2, 29),
            decision_end_date=date(2025, 1, 9),
        ),
        initial_state=state,
        final_state=state,
        session_results=(),
        initial_state_fingerprint=state_hash,
        final_state_fingerprint=state_hash,
    )
    assert compute_source_run_fingerprint(run) == compute_source_run_fingerprint(run)


def test_result_fingerprint_accepts_equivalent_content_model_instances(content_fingerprints):
    interval = HistoricalDecisionInterval(
        decision_start_date=date(2024, 1, 1),
        decision_end_date=date(2024, 12, 31),
    )
    clone = HistoricalBacktestContentFingerprints.model_validate(
        content_fingerprints.model_dump(mode="python")
    )
    values = dict(
        schema_version="historical_backtest_audit_result.v0.2",
        decision_interval=interval,
        run_configuration_fingerprint="a" * 64,
        source_run_fingerprint="b" * 64,
        initial_state_fingerprint="c" * 64,
        final_state_fingerprint="d" * 64,
    )
    assert compute_result_fingerprint(
        **values, content_fingerprints=content_fingerprints
    ) == compute_result_fingerprint(**values, content_fingerprints=clone)


@pytest.mark.parametrize(
    "changed_interval",
    [
        HistoricalDecisionInterval(
            decision_start_date=date(2024, 1, 2),
            decision_end_date=date(2024, 12, 31),
        ),
        HistoricalDecisionInterval(
            decision_start_date=date(2024, 1, 1),
            decision_end_date=date(2024, 12, 30),
        ),
    ],
)
def test_each_decision_interval_endpoint_changes_source_and_result_fingerprints(
    changed_interval, content_fingerprints
) -> None:
    state = PortfolioState(settled_cash=Decimal("1000"))
    state_hash = hash_portfolio_state(state)
    baseline_interval = HistoricalDecisionInterval(
        decision_start_date=date(2024, 1, 1),
        decision_end_date=date(2024, 12, 31),
    )
    baseline_run = HistoricalBacktestRunResult(
        decision_interval=baseline_interval,
        initial_state=state,
        final_state=state,
        initial_state_fingerprint=state_hash,
        final_state_fingerprint=state_hash,
    )
    changed_run = baseline_run.model_copy(
        update={"decision_interval": changed_interval}
    )
    baseline_source = compute_source_run_fingerprint(baseline_run)
    changed_source = compute_source_run_fingerprint(changed_run)

    assert changed_source != baseline_source
    common = {
        "schema_version": "historical_backtest_audit_result.v0.2",
        "run_configuration_fingerprint": "a" * 64,
        "initial_state_fingerprint": state_hash,
        "final_state_fingerprint": state_hash,
        "content_fingerprints": content_fingerprints,
    }
    assert compute_result_fingerprint(
        **common,
        decision_interval=baseline_interval,
        source_run_fingerprint=baseline_source,
    ) != compute_result_fingerprint(
        **common,
        decision_interval=changed_interval,
        source_run_fingerprint=changed_source,
    )
