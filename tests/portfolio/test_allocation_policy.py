"""Phase 12A.1 allocation-policy identity and provenance tests."""

from __future__ import annotations

from dataclasses import MISSING, FrozenInstanceError, asdict, fields

import pytest

from stock_swing_d1.portfolio import (
    PORTFOLIO_ALLOCATION_POLICY,
    PORTFOLIO_ALLOCATION_POLICY_ID,
    PORTFOLIO_ALLOCATION_POLICY_REF,
    PORTFOLIO_ALLOCATION_POLICY_VERSION,
    PortfolioAllocationPolicy,
    PortfolioAllocationPolicyRef,
    PortfolioAllocationDecision,
    PortfolioAllocationService,
    PortfolioCandidateAction,
    compute_portfolio_allocation_policy_fingerprint,
)
from stock_swing_d1.portfolio.allocation_policy import (
    _compute_policy_payload_fingerprint,
)
from stock_swing_d1.ranking import (
    RankingCandidate,
    build_ranked_allocation_batch,
    compute_candidate_input_fingerprint,
    rank_candidates,
)
from stock_swing_d1.ranking import integration as ranking_integration
from tests.portfolio.conftest import DECISION_TIME, SIGNAL_SESSION


def _ranked_batch(candidate):
    signal = candidate.signal
    values = {
        "security_id": candidate.security_id,
        "ranking_session": SIGNAL_SESSION,
        "decision_time": DECISION_TIME,
        "signal_session": signal.signal_session,
        "signal_time": signal.signal_time,
        "sma20": signal.sma_20,
        "sma50": signal.sma_50,
        "atr14": signal.atr_14,
        "rsi14": signal.rsi_14,
    }
    ranking_candidate = RankingCandidate(
        **values,
        input_fingerprint=compute_candidate_input_fingerprint(**values),
    )
    snapshot = rank_candidates(
        ranking_session=SIGNAL_SESSION,
        decision_time=DECISION_TIME,
        candidates=(ranking_candidate,),
    )
    return snapshot, build_ranked_allocation_batch(
        ranking_snapshot=snapshot,
        allocation_candidates=(candidate,),
    )


def test_frozen_policy_contains_exact_phase12_semantics() -> None:
    assert PORTFOLIO_ALLOCATION_POLICY == PortfolioAllocationPolicy()
    assert asdict(PORTFOLIO_ALLOCATION_POLICY) == {
        "schema_version": "portfolio_allocation_policy.v0.1",
        "policy_id": "deterministic_sequential_portfolio_allocation_v0.1",
        "policy_version": "0.1",
        "input_order": "consume_authoritative_input_order",
        "batch_preflight": "required_before_candidate_processing",
        "processing": "sequential",
        "cash_authority": "settled_cash_only",
        "slot_processing": "sequential",
        "position_quantity_owner": "phase11",
        "fixed_quantity_mutation": "forbidden",
        "candidate_cash_reservation": (
            "reserve_full_candidate_cash_limit_on_admission"
        ),
        "same_session_settlement_use": "diagnostic_only_not_spendable",
        "reopen_rejected_candidate": "forbidden",
        "backfill": "forbidden",
        "resize": "forbidden",
        "rerank": "forbidden",
        "portfolio_state_mutation": "forbidden",
        "portfolio_optimization": "forbidden",
    }
    with pytest.raises(ValueError, match="frozen Phase 12"):
        PortfolioAllocationPolicy(processing="parallel")
    with pytest.raises(FrozenInstanceError):
        PORTFOLIO_ALLOCATION_POLICY.processing = "parallel"


def test_policy_ref_is_deterministic_and_matches_frozen_policy() -> None:
    first = compute_portfolio_allocation_policy_fingerprint(
        PORTFOLIO_ALLOCATION_POLICY
    )
    second = compute_portfolio_allocation_policy_fingerprint(
        PortfolioAllocationPolicy()
    )
    rebuilt_ref = PortfolioAllocationPolicyRef(
        policy_id=PORTFOLIO_ALLOCATION_POLICY_ID,
        policy_version=PORTFOLIO_ALLOCATION_POLICY_VERSION,
        policy_fingerprint=second,
    )

    assert first == second
    assert len(first) == 64
    assert first == first.lower()
    assert rebuilt_ref == PORTFOLIO_ALLOCATION_POLICY_REF
    assert PORTFOLIO_ALLOCATION_POLICY_REF.policy_id == (
        PORTFOLIO_ALLOCATION_POLICY_ID
    )
    assert PORTFOLIO_ALLOCATION_POLICY_REF.policy_version == (
        PORTFOLIO_ALLOCATION_POLICY_VERSION
    )


def test_one_policy_semantic_change_changes_domain_separated_fingerprint() -> None:
    baseline = asdict(PORTFOLIO_ALLOCATION_POLICY)
    changed = {**baseline, "processing": "parallel"}

    assert _compute_policy_payload_fingerprint(baseline) != (
        _compute_policy_payload_fingerprint(changed)
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("policy_id", ""),
        ("policy_id", " allocation "),
        ("policy_version", ""),
        ("policy_fingerprint", "A" * 64),
        ("policy_fingerprint", "a" * 63),
        ("policy_fingerprint", "g" * 64),
    ],
)
def test_malformed_policy_refs_are_rejected(
    field_name: str, value: str
) -> None:
    values = {
        "policy_id": PORTFOLIO_ALLOCATION_POLICY_ID,
        "policy_version": PORTFOLIO_ALLOCATION_POLICY_VERSION,
        "policy_fingerprint": "a" * 64,
    }
    values[field_name] = value

    with pytest.raises(ValueError):
        PortfolioAllocationPolicyRef(**values)


def test_policy_ref_forbids_extra_fields_and_mutation() -> None:
    with pytest.raises(TypeError):
        PortfolioAllocationPolicyRef(
            policy_id=PORTFOLIO_ALLOCATION_POLICY_ID,
            policy_version=PORTFOLIO_ALLOCATION_POLICY_VERSION,
            policy_fingerprint="a" * 64,
            ranking_policy="forbidden",
        )
    with pytest.raises(FrozenInstanceError):
        PORTFOLIO_ALLOCATION_POLICY_REF.policy_id = "changed"


def test_legacy_zero_candidate_result_carries_authoritative_policy_ref(
    make_portfolio,
) -> None:
    result = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(), candidates=()
    )

    assert result.candidate_decisions == ()
    assert result.admitted_count == 0
    assert result.total_reserved_cash == 0
    assert result.remaining_unreserved_cash == result.starting_cash
    assert result.allocation_policy_ref == PORTFOLIO_ALLOCATION_POLICY_REF


def test_nonempty_result_adds_only_aggregate_policy_provenance(
    make_candidate, make_portfolio
) -> None:
    result = PortfolioAllocationService().allocate_candidates(
        portfolio=make_portfolio(), candidates=(make_candidate(),)
    )
    candidate = result.candidate_decisions[0]

    assert result.allocation_policy_ref == PORTFOLIO_ALLOCATION_POLICY_REF
    assert candidate.action is PortfolioCandidateAction.ADMITTED
    assert candidate.sized_pending_entry is not None
    assert candidate.sized_pending_entry.fixed_shares == 12
    assert candidate.candidate_cash_limit == 2_000
    assert candidate.reserved_cash == 2_000
    assert candidate.used_slots_before == 0
    assert candidate.used_slots_after == 1
    assert not hasattr(candidate, "allocation_policy_ref")


def test_aggregate_policy_ref_is_required_without_a_model_default() -> None:
    policy_field = next(
        field
        for field in fields(PortfolioAllocationDecision)
        if field.name == "allocation_policy_ref"
    )

    assert policy_field.default is MISSING
    assert policy_field.default_factory is MISSING


def test_ranking_lineage_is_independent_from_allocation_policy_ref(
    make_candidate, make_portfolio, monkeypatch
) -> None:
    _, baseline_batch = _ranked_batch(make_candidate())
    service = PortfolioAllocationService()
    baseline = service.allocate_ranked_candidates(
        portfolio=make_portfolio(), ranked_batch=baseline_batch
    )
    alternate_ranking_fingerprint = "f" * 64
    assert alternate_ranking_fingerprint != baseline_batch.policy_fingerprint
    alternate_batch = baseline_batch.model_copy(
        update={"policy_fingerprint": alternate_ranking_fingerprint}
    )
    monkeypatch.setattr(
        ranking_integration,
        "CANDIDATE_RANKING_POLICY_FINGERPRINT",
        alternate_ranking_fingerprint,
    )

    alternate = service.allocate_ranked_candidates(
        portfolio=make_portfolio(), ranked_batch=alternate_batch
    )

    assert baseline.policy_fingerprint != alternate.policy_fingerprint
    assert baseline.allocation_policy_ref == alternate.allocation_policy_ref
    assert alternate.allocation_policy_ref == PORTFOLIO_ALLOCATION_POLICY_REF
    for field in fields(type(baseline)):
        if field.name != "policy_fingerprint":
            assert getattr(baseline, field.name) == getattr(alternate, field.name)
