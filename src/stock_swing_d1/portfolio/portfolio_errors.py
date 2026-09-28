"""Phase 13 portfolio domain exception hierarchy.

Structural model validation intentionally continues to use Pydantic's
``ValidationError``.  These exceptions are reserved for the transition and
persistence behavior implemented by later Phase 13 tasks.
"""


class PortfolioError(Exception):
    """Base exception for explicit portfolio-domain failures."""


class PortfolioStateError(PortfolioError):
    """A portfolio state violates a domain invariant."""


class PortfolioTransitionError(PortfolioError):
    """Base exception for a rejected portfolio state transition."""


class OutOfOrderSessionError(PortfolioTransitionError):
    """An event or session was supplied before the current state session."""


class OverdueSettlementError(PortfolioTransitionError):
    """A required settlement was not applied by its settlement session."""


class InsufficientSettledCashError(PortfolioTransitionError):
    """A purchase requires more settled cash than is available."""


class PositionAlreadyOpenError(PortfolioTransitionError):
    """An entry targets an asset that already has an open position."""


class PositionNotFoundError(PortfolioTransitionError):
    """An exit targets an asset without an open position."""


class InvalidExitQuantityError(PortfolioTransitionError):
    """An exit quantity does not fully close the target position."""


class DuplicateAssetApplicationError(PortfolioTransitionError):
    """One asset received two BUY or two SELL applications in one transition."""


class DuplicateEventConflictError(PortfolioTransitionError):
    """A repeated event identity has a different canonical payload hash."""


class SettlementIntegrityError(PortfolioTransitionError):
    """Settlement data conflicts with its source execution."""


class DividendEvidenceError(PortfolioTransitionError):
    """Supplied ordinary-dividend accounting evidence is malformed or misplaced."""


class DividendProvenanceError(PortfolioStateError):
    """A dividend ledger row or outcome fails provenance or discharge proof."""


class PortfolioPersistenceError(PortfolioError):
    """Base exception for persisted portfolio artifact failures."""


class StateHashMismatchError(PortfolioPersistenceError):
    """A portfolio state's content does not match its declared hash."""


class ArtifactHashMismatchError(PortfolioPersistenceError):
    """A persisted artifact does not match its declared hash."""


class SchemaVersionError(PortfolioPersistenceError):
    """A persisted artifact uses an unsupported schema version."""


__all__ = [
    "ArtifactHashMismatchError",
    "DividendEvidenceError",
    "DividendProvenanceError",
    "DuplicateAssetApplicationError",
    "DuplicateEventConflictError",
    "InsufficientSettledCashError",
    "InvalidExitQuantityError",
    "OutOfOrderSessionError",
    "OverdueSettlementError",
    "PortfolioError",
    "PortfolioPersistenceError",
    "PortfolioStateError",
    "PortfolioTransitionError",
    "PositionAlreadyOpenError",
    "PositionNotFoundError",
    "SchemaVersionError",
    "SettlementIntegrityError",
    "StateHashMismatchError",
]
