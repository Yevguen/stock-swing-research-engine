"""The frozen Material Adverse Overnight Gap / Gap-Loss Frequency Contract v0.1.

Every semantic field below is fixed by the frozen contract rather than chosen
by a caller, so the model rejects a noncanonical value even when the builder is
bypassed. That is what makes the threshold un-tunable after the baseline result
becomes visible: a different threshold is a different policy identity with a
different fingerprint, and it can never masquerade as this one.

The materiality threshold is preserved as canonical text exactly as the frozen
contract writes it (``"-0.010000"``). Canonical semantic JSON normalizes a
Decimal by stripping trailing zeros, so storing the threshold as text is what
keeps the frozen six-decimal expression byte-stable inside the manifest. The
numeric value used by the calculation is the module constant below, and the
calculation asserts the two agree before it computes anything.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import model_validator

from stock_swing_d1.baseline_experiment.field_types import (
    ImmutableBaselineExperimentModel,
    Sha256,
)
from stock_swing_d1.baseline_experiment.hashing import (
    _compute_gap_policy_fingerprint,
    compute_gap_policy_fingerprint,
)


MATERIAL_ADVERSE_OVERNIGHT_GAP_POLICY_SCHEMA_VERSION = (
    "material_adverse_overnight_gap_policy.v0.1"
)
GAP_METRIC_VERSION = "material_adverse_overnight_gap.v0.1"
GAP_REPORT_LABEL = "Gap-loss frequency"
GAP_DIRECTION = "long_adverse"
GAP_MATERIALITY_THRESHOLD_TEXT = "-0.010000"
GAP_MATERIALITY_THRESHOLD = Decimal(GAP_MATERIALITY_THRESHOLD_TEXT)
GAP_PRICE_BASIS = "canonical_unadjusted_tradable"
GAP_CORPORATE_ACTION_NORMALIZATION = "frozen_canonical"
GAP_ORDINARY_DIVIDEND_NORMALIZATION = "frozen_canonical"
GAP_TRADE_COUNTING_RULE = "one_per_trade"


class MaterialAdverseOvernightGapPolicy(ImmutableBaselineExperimentModel):
    """The one frozen v0.1 gap-loss-frequency methodology."""

    schema_version: Literal[
        "material_adverse_overnight_gap_policy.v0.1"
    ] = MATERIAL_ADVERSE_OVERNIGHT_GAP_POLICY_SCHEMA_VERSION
    gap_metric_version: Literal[
        "material_adverse_overnight_gap.v0.1"
    ] = GAP_METRIC_VERSION
    report_label: Literal["Gap-loss frequency"] = GAP_REPORT_LABEL
    direction: Literal["long_adverse"] = GAP_DIRECTION
    materiality_threshold: Literal["-0.010000"] = GAP_MATERIALITY_THRESHOLD_TEXT
    price_basis: Literal[
        "canonical_unadjusted_tradable"
    ] = GAP_PRICE_BASIS
    corporate_action_normalization: Literal[
        "frozen_canonical"
    ] = GAP_CORPORATE_ACTION_NORMALIZATION
    ordinary_dividend_normalization: Literal[
        "frozen_canonical"
    ] = GAP_ORDINARY_DIVIDEND_NORMALIZATION
    trade_counting_rule: Literal["one_per_trade"] = GAP_TRADE_COUNTING_RULE
    policy_fingerprint: Sha256

    @model_validator(mode="after")
    def validate_fingerprint(self) -> Self:
        if self.policy_fingerprint != compute_gap_policy_fingerprint(self):
            raise ValueError("policy_fingerprint does not match policy content")
        return self


class MaterialAdverseOvernightGapPolicyRef(ImmutableBaselineExperimentModel):
    """Minimum stable identity of the gap policy that produced a diagnostic."""

    schema_version: Literal[
        "material_adverse_overnight_gap_policy.v0.1"
    ] = MATERIAL_ADVERSE_OVERNIGHT_GAP_POLICY_SCHEMA_VERSION
    gap_metric_version: Literal[
        "material_adverse_overnight_gap.v0.1"
    ] = GAP_METRIC_VERSION
    materiality_threshold: Literal["-0.010000"] = GAP_MATERIALITY_THRESHOLD_TEXT
    trade_counting_rule: Literal["one_per_trade"] = GAP_TRADE_COUNTING_RULE
    policy_fingerprint: Sha256


def build_material_adverse_overnight_gap_policy() -> (
    MaterialAdverseOvernightGapPolicy
):
    """Build the single canonical frozen v0.1 gap policy."""

    values = {
        "schema_version": MATERIAL_ADVERSE_OVERNIGHT_GAP_POLICY_SCHEMA_VERSION,
        "gap_metric_version": GAP_METRIC_VERSION,
        "report_label": GAP_REPORT_LABEL,
        "direction": GAP_DIRECTION,
        "materiality_threshold": GAP_MATERIALITY_THRESHOLD_TEXT,
        "price_basis": GAP_PRICE_BASIS,
        "corporate_action_normalization": GAP_CORPORATE_ACTION_NORMALIZATION,
        "ordinary_dividend_normalization": (
            GAP_ORDINARY_DIVIDEND_NORMALIZATION
        ),
        "trade_counting_rule": GAP_TRADE_COUNTING_RULE,
    }
    return MaterialAdverseOvernightGapPolicy(
        **values,
        policy_fingerprint=_compute_gap_policy_fingerprint(**values),
    )


def build_material_adverse_overnight_gap_policy_ref(
    policy: MaterialAdverseOvernightGapPolicy,
) -> MaterialAdverseOvernightGapPolicyRef:
    """Copy the stable identity fields of one validated gap policy."""

    if type(policy) is not MaterialAdverseOvernightGapPolicy:
        raise TypeError("policy must be a MaterialAdverseOvernightGapPolicy")
    return MaterialAdverseOvernightGapPolicyRef(
        schema_version=policy.schema_version,
        gap_metric_version=policy.gap_metric_version,
        materiality_threshold=policy.materiality_threshold,
        trade_counting_rule=policy.trade_counting_rule,
        policy_fingerprint=policy.policy_fingerprint,
    )


__all__ = [
    "GAP_CORPORATE_ACTION_NORMALIZATION",
    "GAP_DIRECTION",
    "GAP_MATERIALITY_THRESHOLD",
    "GAP_MATERIALITY_THRESHOLD_TEXT",
    "GAP_METRIC_VERSION",
    "GAP_ORDINARY_DIVIDEND_NORMALIZATION",
    "GAP_PRICE_BASIS",
    "GAP_REPORT_LABEL",
    "GAP_TRADE_COUNTING_RULE",
    "MATERIAL_ADVERSE_OVERNIGHT_GAP_POLICY_SCHEMA_VERSION",
    "MaterialAdverseOvernightGapPolicy",
    "MaterialAdverseOvernightGapPolicyRef",
    "build_material_adverse_overnight_gap_policy",
    "build_material_adverse_overnight_gap_policy_ref",
]
