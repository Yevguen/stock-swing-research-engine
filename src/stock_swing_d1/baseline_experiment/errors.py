"""Typed fail-closed errors for the Phase 16C baseline-experiment boundary."""

from __future__ import annotations


class BaselineExperimentError(Exception):
    """Base of the Phase 16C fail-closed error vocabulary."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class BaselineExperimentValidationError(BaselineExperimentError, ValueError):
    """One Phase 16C manifest, evidence, or result boundary was violated."""


class BaselineExperimentDiagnosticError(BaselineExperimentError, ValueError):
    """A Phase 16C experiment diagnostic could not be established.

    This is strictly separate from an explicitly undefined diagnostic value.
    An undefined diagnostic carries ``value = None`` plus one frozen reason
    code and remains a valid published observation about the experiment; this
    error means the supplied evidence was structurally invalid, so no
    diagnostic is published at all.
    """


class CanonicalRunAuthorizationError(BaselineExperimentError, RuntimeError):
    """The exactly-once canonical baseline run was not deliberately authorized."""


__all__ = [
    "BaselineExperimentDiagnosticError",
    "BaselineExperimentError",
    "BaselineExperimentValidationError",
    "CanonicalRunAuthorizationError",
]
