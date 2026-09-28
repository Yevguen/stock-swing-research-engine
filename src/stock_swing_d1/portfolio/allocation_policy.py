"""Immutable semantic identity for Phase 12 portfolio allocation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass


PORTFOLIO_ALLOCATION_POLICY_HASH_DOMAIN = "portfolio_allocation_policy.v0.1"
PORTFOLIO_ALLOCATION_POLICY_ID = (
    "deterministic_sequential_portfolio_allocation_v0.1"
)
PORTFOLIO_ALLOCATION_POLICY_VERSION = "0.1"


@dataclass(frozen=True, slots=True)
class PortfolioAllocationPolicy:
    """The frozen, non-configurable Phase 12 allocation semantics."""

    schema_version: str = "portfolio_allocation_policy.v0.1"
    policy_id: str = PORTFOLIO_ALLOCATION_POLICY_ID
    policy_version: str = PORTFOLIO_ALLOCATION_POLICY_VERSION
    input_order: str = "consume_authoritative_input_order"
    batch_preflight: str = "required_before_candidate_processing"
    processing: str = "sequential"
    cash_authority: str = "settled_cash_only"
    slot_processing: str = "sequential"
    position_quantity_owner: str = "phase11"
    fixed_quantity_mutation: str = "forbidden"
    candidate_cash_reservation: str = (
        "reserve_full_candidate_cash_limit_on_admission"
    )
    same_session_settlement_use: str = "diagnostic_only_not_spendable"
    reopen_rejected_candidate: str = "forbidden"
    backfill: str = "forbidden"
    resize: str = "forbidden"
    rerank: str = "forbidden"
    portfolio_state_mutation: str = "forbidden"
    portfolio_optimization: str = "forbidden"

    def __post_init__(self) -> None:
        expected = {
            "schema_version": "portfolio_allocation_policy.v0.1",
            "policy_id": PORTFOLIO_ALLOCATION_POLICY_ID,
            "policy_version": PORTFOLIO_ALLOCATION_POLICY_VERSION,
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
            "same_session_settlement_use": (
                "diagnostic_only_not_spendable"
            ),
            "reopen_rejected_candidate": "forbidden",
            "backfill": "forbidden",
            "resize": "forbidden",
            "rerank": "forbidden",
            "portfolio_state_mutation": "forbidden",
            "portfolio_optimization": "forbidden",
        }
        for field_name, required in expected.items():
            if getattr(self, field_name) != required:
                raise ValueError(
                    f"{field_name} must match the frozen Phase 12 v0.1 policy"
                )


@dataclass(frozen=True, slots=True)
class PortfolioAllocationPolicyRef:
    """Identity and semantic fingerprint of one Phase 12 allocation policy."""

    policy_id: str
    policy_version: str
    policy_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in ("policy_id", "policy_version"):
            value = getattr(self, field_name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    f"{field_name} must be canonical non-empty text"
                )
        fingerprint = self.policy_fingerprint
        if (
            type(fingerprint) is not str
            or len(fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in fingerprint
            )
        ):
            raise ValueError(
                "policy_fingerprint must be a lowercase SHA-256 digest"
            )


def _compute_policy_payload_fingerprint(payload: Mapping[str, object]) -> str:
    """Hash one testable semantic payload in the Phase 12 policy domain."""

    if any(type(key) is not str for key in payload):
        raise TypeError("allocation policy payload keys must be strings")
    if any(type(value) is not str for value in payload.values()):
        raise TypeError("allocation policy payload values must be strings")
    semantic = {
        "domain": PORTFOLIO_ALLOCATION_POLICY_HASH_DOMAIN,
        "payload": dict(payload),
    }
    encoded = json.dumps(
        semantic,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_portfolio_allocation_policy_fingerprint(
    policy: PortfolioAllocationPolicy,
) -> str:
    """Return the semantic SHA-256 of the frozen Phase 12 policy."""

    if type(policy) is not PortfolioAllocationPolicy:
        raise TypeError("policy must be a PortfolioAllocationPolicy")
    canonical = PortfolioAllocationPolicy(**asdict(policy))
    return _compute_policy_payload_fingerprint(asdict(canonical))


PORTFOLIO_ALLOCATION_POLICY = PortfolioAllocationPolicy()
PORTFOLIO_ALLOCATION_POLICY_REF = PortfolioAllocationPolicyRef(
    policy_id=PORTFOLIO_ALLOCATION_POLICY.policy_id,
    policy_version=PORTFOLIO_ALLOCATION_POLICY.policy_version,
    policy_fingerprint=compute_portfolio_allocation_policy_fingerprint(
        PORTFOLIO_ALLOCATION_POLICY
    ),
)


__all__ = [
    "PORTFOLIO_ALLOCATION_POLICY",
    "PORTFOLIO_ALLOCATION_POLICY_ID",
    "PORTFOLIO_ALLOCATION_POLICY_REF",
    "PORTFOLIO_ALLOCATION_POLICY_VERSION",
    "PortfolioAllocationPolicy",
    "PortfolioAllocationPolicyRef",
    "compute_portfolio_allocation_policy_fingerprint",
]
