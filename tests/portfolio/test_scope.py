"""Public API, immutability, and scope guards for Phase 12."""

from dataclasses import FrozenInstanceError
import inspect
from pathlib import Path

import pytest

import stock_swing_d1.portfolio as portfolio_package
from stock_swing_d1.portfolio import (
    PORTFOLIO_ALLOCATION_POLICY,
    PORTFOLIO_ALLOCATION_POLICY_ID,
    PORTFOLIO_ALLOCATION_POLICY_REF,
    PORTFOLIO_ALLOCATION_POLICY_VERSION,
    OpenPosition,
    PortfolioAllocationDecision,
    PortfolioAllocationPolicy,
    PortfolioAllocationPolicyRef,
    PortfolioAllocationService,
    PortfolioAllocationValidationError,
    PortfolioCandidate,
    PortfolioCandidateAction,
    PortfolioCandidateDecision,
    PortfolioReservationSettlement,
    PortfolioSnapshot,
    RankedPortfolioAllocationDecision,
    RankedPortfolioCandidateDecision,
    compute_portfolio_allocation_policy_fingerprint,
)


def _production_source() -> str:
    package_path = Path(portfolio_package.__file__).parent
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package_path.glob("*.py"))
    ).lower()


def test_public_exports_are_exactly_the_narrow_phase12_contract() -> None:
    assert set(portfolio_package.__all__) == {
        "MAX_SIMULTANEOUS_POSITIONS",
        "MAX_SINGLE_POSITION_ALLOCATION",
        "PORTFOLIO_ALLOCATION_POLICY",
        "PORTFOLIO_ALLOCATION_POLICY_ID",
        "PORTFOLIO_ALLOCATION_POLICY_REF",
        "PORTFOLIO_ALLOCATION_POLICY_VERSION",
        "OpenPosition",
        "PortfolioAllocationDecision",
        "PortfolioAllocationPolicy",
        "PortfolioAllocationPolicyRef",
        "PortfolioAllocationService",
        "PortfolioAllocationValidationError",
        "PortfolioCandidate",
        "PortfolioCandidateAction",
        "PortfolioCandidateDecision",
        "PortfolioReservationSettlement",
        "PortfolioSnapshot",
        "RankedPortfolioAllocationDecision",
        "RankedPortfolioCandidateDecision",
        "compute_portfolio_allocation_policy_fingerprint",
    }


def test_allocation_api_accepts_only_completed_t_portfolio_and_candidates() -> None:
    signature = inspect.signature(
        PortfolioAllocationService.allocate_candidates
    )

    assert tuple(signature.parameters) == ("self", "portfolio", "candidates")
    assert all(
        forbidden not in signature.parameters
        for forbidden in (
            "entry_execution",
            "execution_bar",
            "execution_price",
            "protective_exit",
            "exit_price",
            "future_portfolio_equity",
            "expected_future_proceeds",
        )
    )


def test_ranked_allocation_has_a_distinct_unambiguous_entry_point() -> None:
    signature = inspect.signature(
        PortfolioAllocationService.allocate_ranked_candidates
    )

    assert tuple(signature.parameters) == (
        "self",
        "portfolio",
        "ranked_batch",
    )
    assert "ranked" not in inspect.signature(
        PortfolioAllocationService.allocate_candidates
    ).parameters


def test_settlement_api_cannot_reopen_or_reallocate_the_cycle() -> None:
    signature = inspect.signature(
        PortfolioAllocationService.settle_reservation
    )

    assert tuple(signature.parameters) == (
        "self",
        "candidate_decision",
        "entry_execution",
    )
    assert "candidates" not in signature.parameters
    public_operations = {
        name
        for name, member in inspect.getmembers(
            PortfolioAllocationService, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert public_operations == {
        "allocate_candidates",
        "allocate_ranked_candidates",
        "settle_reservation",
    }


def test_phase12_contains_no_whole_share_sizing_formula() -> None:
    source = _production_source()

    assert "floor(" not in source
    assert "risk_sized_shares" not in source
    assert "cash_sized_shares" not in source
    assert "loss_per_share" not in source


def test_phase12_does_not_import_phase15d() -> None:
    assert "stock_swing_d1.backtest_results" not in _production_source()


def test_phase12_contains_no_out_of_scope_allocation_features() -> None:
    source = _production_source()

    forbidden_fragments = (
        "max_sector_allocation",
        "sector classification",
        "sector exposure",
        "mean-variance",
        "risk parity",
        "kelly",
        "cvar",
        "correlation adjustment",
        "beta targeting",
        "volatility targeting",
        "borrowed_cash",
        "buying power",
        "leverage multiplier",
        "backfill_cancelled_entry",
        "reallocate_after_execution",
        "resize_after_gap",
        "admit_rejected_candidate_after_release",
        "expected_exit_proceeds",
        "expected_sale_cash",
        "future_cash",
        "ai score",
    )
    assert all(fragment not in source for fragment in forbidden_fragments)


def test_input_models_are_immutable(
    make_portfolio, make_open_position, make_candidate
) -> None:
    position = make_open_position()
    snapshot = make_portfolio(open_positions=[position])
    candidate = make_candidate()

    assert isinstance(snapshot.open_positions, tuple)
    with pytest.raises(FrozenInstanceError):
        position.shares = 99
    with pytest.raises(FrozenInstanceError):
        snapshot.cash_available = 0
    with pytest.raises(FrozenInstanceError):
        candidate.signal = object()


def test_public_model_types_are_the_expected_contract() -> None:
    assert issubclass(PortfolioAllocationValidationError, ValueError)
    assert inspect.isclass(OpenPosition)
    assert inspect.isclass(PortfolioSnapshot)
    assert inspect.isclass(PortfolioCandidate)
    assert inspect.isclass(PortfolioCandidateAction)
    assert inspect.isclass(PortfolioCandidateDecision)
    assert inspect.isclass(PortfolioAllocationDecision)
    assert inspect.isclass(PortfolioAllocationPolicy)
    assert inspect.isclass(PortfolioAllocationPolicyRef)
    assert PORTFOLIO_ALLOCATION_POLICY.policy_id == (
        PORTFOLIO_ALLOCATION_POLICY_ID
    )
    assert PORTFOLIO_ALLOCATION_POLICY.policy_version == (
        PORTFOLIO_ALLOCATION_POLICY_VERSION
    )
    assert PORTFOLIO_ALLOCATION_POLICY_REF.policy_fingerprint == (
        compute_portfolio_allocation_policy_fingerprint(
            PORTFOLIO_ALLOCATION_POLICY
        )
    )
    assert issubclass(
        RankedPortfolioCandidateDecision, PortfolioCandidateDecision
    )
    assert issubclass(
        RankedPortfolioAllocationDecision, PortfolioAllocationDecision
    )
    assert inspect.isclass(PortfolioReservationSettlement)
