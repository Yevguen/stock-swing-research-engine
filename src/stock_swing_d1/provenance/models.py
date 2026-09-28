"""Generic artifact provenance identity shared by upstream owners.

Task 5C-A froze ``ArtifactRef`` as *generic provenance*, not valuation
semantics, so it lives in this minimal neutral package rather than in the
Phase 15D audit layer or in ``valuation``.  The class was moved here
unchanged: same fields, same validators, same strictness, same canonical
representation, therefore the same semantic digest under every existing hash
domain.  The Phase 15D audit layer re-exports this exact class object, so its
public ``ArtifactRef`` name binds to this class and every existing equality
comparison, ``type(...) is ...`` check and persisted fingerprint is
preserved.

This package owns ``ArtifactRef`` and nothing else: no registry, no artifact
store, no loader, no resolver, no provider abstraction and no persistence
framework.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _require_canonical_text(value: object) -> object:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("must be canonical non-empty text")
    return value


def _require_optional_canonical_text(value: object) -> object:
    if value is None:
        return value
    return _require_canonical_text(value)


class _ImmutableProvenanceModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_default=True,
    )


_CanonicalText = Annotated[str, BeforeValidator(_require_canonical_text)]
_OptionalCanonicalText = Annotated[
    str | None, BeforeValidator(_require_optional_canonical_text)
]
_Sha256 = Annotated[
    str,
    BeforeValidator(_require_canonical_text),
    Field(pattern=r"^[0-9a-f]{64}$"),
]


class ArtifactRef(_ImmutableProvenanceModel):
    """Semantic identity of an input artifact, independent of storage."""

    artifact_type: _CanonicalText
    schema_version: _CanonicalText
    content_sha256: _Sha256
    build_id: _OptionalCanonicalText = None


__all__ = ["ArtifactRef"]
