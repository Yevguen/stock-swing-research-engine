"""Phase 15D companion amendment, shared-primitive reuse, pinned identities.

The forward-projection amendment in ``source_validation`` is Phase 15D-owned
and separate from the Phase 15A one: the float Phase 12 consumed is compared
against ``float(authoritative settled cash)``, never the reverse.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.backtest_results import (
    HistoricalBacktestResultValidationError,
    build_valuation_policy_ref,
    build_valuation_snapshot,
)
from stock_swing_d1.backtest_results.models import (
    HistoricalBacktestValuationMark,
    HistoricalBacktestValuationPolicy,
    PolicyArtifactRef,
    build_run_manifest,
)
from stock_swing_d1.backtest_results.valuation import project_equity_curve
from stock_swing_d1.backtester import (
    HistoricalBacktestOrchestrator,
    HistoricalBacktestSessionInput,
)
from stock_swing_d1.portfolio import (
    PORTFOLIO_ALLOCATION_POLICY_REF,
    PortfolioSnapshot,
)
from stock_swing_d1.portfolio.portfolio_state_models import PortfolioState
from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.valuation import (
    PortfolioValuationPolicy,
    value_portfolio_at_allocation_boundary,
)

from tests.backtest_results.conftest import (
    SOURCE_DECISION_INTERVAL,
    SOURCE_ENTRY_SESSION,
    SOURCE_SESSION,
    _source_candidate,
    source_decision_time,
)


@pytest.fixture
def allocating_run_manifest(
    artifact_ref_factory,
    valuation_policy_ref,
    ranking_policy_ref,
    execution_cost_policy_ref,
):
    """A manifest bound to the real Phase 12 allocation policy of the run."""

    return build_run_manifest(
        software_revision="c" * 40,
        strategy_configuration_ref=artifact_ref_factory("strategy"),
        allocation_policy_ref=PolicyArtifactRef(
            policy_id=PORTFOLIO_ALLOCATION_POLICY_REF.policy_id,
            policy_version=PORTFOLIO_ALLOCATION_POLICY_REF.policy_version,
            policy_fingerprint=(
                PORTFOLIO_ALLOCATION_POLICY_REF.policy_fingerprint
            ),
        ),
        ranking_policy_ref=ranking_policy_ref,
        execution_cost_policy_ref=execution_cost_policy_ref,
        valuation_policy_ref=valuation_policy_ref,
        universe_artifact_ref=artifact_ref_factory("universe"),
        market_data_artifact_ref=artifact_ref_factory("market_data"),
    )


# A legitimate balance produced by the frozen Gate3 dividend accounting:
# Q_T * D_H credited into settled cash carries scale-38 precision.
DIVIDEND_SCALE_CASH = Decimal(
    "10031.02658067566857563179740450795985919134"
)


def _allocating_run(settled_cash: Decimal):
    _signal, ranking_candidate, allocation_candidate = _source_candidate(
        "NORGATE:101"
    )
    return HistoricalBacktestOrchestrator().run(
        PortfolioState(settled_cash=settled_cash),
        (
            HistoricalBacktestSessionInput(
                session=SOURCE_SESSION,
                decision_time=source_decision_time(SOURCE_SESSION),
                next_session=SOURCE_ENTRY_SESSION,
                ranking_candidates=(ranking_candidate,),
                allocation_candidates=(allocation_candidate,),
                allocation_portfolio=PortfolioSnapshot(
                    allocation_session=SOURCE_SESSION,
                    decision_time=source_decision_time(SOURCE_SESSION),
                    portfolio_equity=float(settled_cash),
                    cash_available=float(settled_cash),
                ),
            ),
        ),
        decision_interval=SOURCE_DECISION_INTERVAL,
    )


def _snapshots(run):
    return tuple(
        build_valuation_snapshot(session=session.session, marks=())
        for session in run.session_results
    )


def test_high_precision_settled_cash_passes_forward_projection(
    allocating_run_manifest,
):
    """The reverse round trip fails; the prescribed projection must not."""

    assert Decimal(str(float(DIVIDEND_SCALE_CASH))) != DIVIDEND_SCALE_CASH

    run = _allocating_run(DIVIDEND_SCALE_CASH)
    rows = project_equity_curve(
        run_result=run,
        run_manifest=allocating_run_manifest,
        valuation_snapshots=_snapshots(run),
    )

    assert rows[-1].settled_cash == DIVIDEND_SCALE_CASH
    assert rows[-1].equity == DIVIDEND_SCALE_CASH


def test_altered_starting_cash_still_fails_provenance(allocating_run_manifest):
    run = _allocating_run(Decimal("10000"))
    allocation = run.session_results[0].allocation_decision
    object.__setattr__(allocation, "starting_cash", 9_999.0)

    with pytest.raises(
        HistoricalBacktestResultValidationError,
        match="ARTIFACT_PROVENANCE_MISMATCH",
    ):
        project_equity_curve(
            run_result=run,
            run_manifest=allocating_run_manifest,
            valuation_snapshots=_snapshots(run),
        )


def test_source_validation_uses_no_reverse_projection_for_cash():
    """Executable code only -- comments about the old form do not count."""

    import ast
    import inspect

    from stock_swing_d1.backtest_results import source_validation

    code = ast.unparse(ast.parse(inspect.getsource(source_validation)))

    assert "Decimal(str(allocation.starting_cash))" not in code
    assert (
        "allocation.starting_cash != float(session.authoritative_state.settled_cash)"
        in code
    )


def test_phase15d_equity_equals_the_shared_primitive(allocating_run_manifest):
    run = _allocating_run(Decimal("12345.67"))
    rows = project_equity_curve(
        run_result=run,
        run_manifest=allocating_run_manifest,
        valuation_snapshots=_snapshots(run),
    )

    for row, session in zip(rows, run.session_results, strict=True):
        shared = value_portfolio_at_allocation_boundary(
            state=session.authoritative_state,
            session=session.session,
            marks=(),
            policy=PortfolioValuationPolicy(),
        )
        assert row.settled_cash == shared.settled_cash
        assert row.pending_receivable_value == shared.pending_receivable_value
        assert (
            row.open_position_market_value
            == shared.open_position_market_value
        )
        assert row.equity == shared.portfolio_equity


def test_valuation_policy_fingerprint_is_unchanged_after_relocation():
    """Pinned literal: a module move must not move a digest."""

    assert (
        build_valuation_policy_ref(
            HistoricalBacktestValuationPolicy()
        ).policy_fingerprint
        == "26db718741799e5f8300041104e26ccc8a0981b74afec5c6ee6b96d4f0a09bca"
    )


def test_valuation_snapshot_fingerprint_is_unchanged_after_relocation():
    ref = ArtifactRef(
        artifact_type="market_data",
        schema_version="stock_bars_v0_1",
        content_sha256="b" * 64,
    )
    marks = (
        HistoricalBacktestValuationMark(
            security_id="NORGATE:2002",
            session=date(2026, 8, 3),
            close=Decimal("50.5"),
            source_artifact_ref=ref,
        ),
        HistoricalBacktestValuationMark(
            security_id="NORGATE:1001",
            session=date(2026, 8, 3),
            close=Decimal("101.25"),
            source_artifact_ref=ref,
        ),
    )

    snapshot = build_valuation_snapshot(session=date(2026, 8, 3), marks=marks)

    assert snapshot.snapshot_fingerprint == (
        "a594b7a501f8731db2d711bed6fe7e1b2c048fb9f8bd2b0cc6549cadab87737a"
    )


def test_canonical_serialization_is_structural_not_class_bound():
    """Why relocation is digest-neutral: only field names and values hash."""

    from stock_swing_d1.backtest_results.hashing import (
        VALUATION_POLICY_HASH_DOMAIN,
        semantic_domain_sha256,
    )

    policy = HistoricalBacktestValuationPolicy()
    as_plain_mapping = policy.model_dump(mode="python")

    assert semantic_domain_sha256(
        VALUATION_POLICY_HASH_DOMAIN, policy
    ) == semantic_domain_sha256(
        VALUATION_POLICY_HASH_DOMAIN, as_plain_mapping
    )
