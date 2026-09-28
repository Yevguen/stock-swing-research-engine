"""Typed fail-closed errors for the Phase 16B research-metrics boundary."""


class ResearchMetricsError(Exception):
    """Base of the Phase 16B fail-closed error vocabulary."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class ResearchMetricsValidationError(ResearchMetricsError, ValueError):
    """One Phase 16B model, policy, or evidence boundary was violated."""


class ResearchMetricsCalculationError(ResearchMetricsError):
    """Canonical arithmetic could not produce an otherwise defined metric.

    Phase 16B.2 v0.2.1 Clause 39.3 keeps this strictly separate from a
    mathematically undefined metric: a calculation/determinism failure must
    never be published as ``value = None`` plus a canonical undefined reason,
    because that would misclassify an implementation artifact as a
    mathematical property of the experiment.
    """


class ResearchMetricsPersistenceError(ResearchMetricsError, ValueError):
    """A Phase 16B.3 artifact could not be encoded, decoded, or verified."""


__all__ = [
    "ResearchMetricsCalculationError",
    "ResearchMetricsError",
    "ResearchMetricsPersistenceError",
    "ResearchMetricsValidationError",
]
