"""Minimal neutral provenance package (Task 5C-A).

Owns exactly one generic contract, ``ArtifactRef``.  Deliberately not a
framework: no registries, stores, loaders, resolvers or provider
abstractions belong here.
"""

from stock_swing_d1.provenance.models import ArtifactRef

__all__ = ["ArtifactRef"]
