"""Task 5C-A: ArtifactRef is generic provenance, relocated without drift."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

import stock_swing_d1.backtest_results as backtest_results_package
from stock_swing_d1.backtest_results import models as backtest_results_models
from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.provenance import models as provenance_models


PROVENANCE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "stock_swing_d1"
    / "provenance"
)


def _ref(**overrides) -> ArtifactRef:
    payload = {
        "artifact_type": "market_data",
        "schema_version": "stock_bars_v0_1",
        "content_sha256": "b" * 64,
    }
    payload.update(overrides)
    return ArtifactRef(**payload)


def test_every_public_import_path_is_the_same_class_object():
    assert ArtifactRef is provenance_models.ArtifactRef
    assert ArtifactRef is backtest_results_models.ArtifactRef
    assert ArtifactRef is backtest_results_package.ArtifactRef
    assert "ArtifactRef" in backtest_results_package.__all__


def test_refs_built_through_different_import_paths_compare_equal():
    """Pydantic equality requires identical class identity, not just fields."""

    through_provenance = ArtifactRef(
        artifact_type="market_data",
        schema_version="stock_bars_v0_1",
        content_sha256="b" * 64,
    )
    through_results = backtest_results_models.ArtifactRef(
        artifact_type="market_data",
        schema_version="stock_bars_v0_1",
        content_sha256="b" * 64,
    )
    assert through_provenance == through_results
    assert type(through_provenance) is type(through_results)


def test_field_semantics_and_strictness_are_unchanged():
    ref = _ref(build_id="BUILD-1")
    assert tuple(ArtifactRef.model_fields) == (
        "artifact_type",
        "schema_version",
        "content_sha256",
        "build_id",
    )
    assert _ref().build_id is None
    assert ref.model_config["frozen"] is True
    assert ref.model_config["extra"] == "forbid"


@pytest.mark.parametrize(
    "overrides",
    [
        {"artifact_type": ""},
        {"artifact_type": " padded "},
        {"schema_version": 1},
        {"content_sha256": "B" * 64},
        {"content_sha256": "b" * 63},
        {"build_id": ""},
        {"unexpected": "field"},
    ],
)
def test_invalid_refs_still_fail_closed(overrides):
    with pytest.raises(ValidationError):
        _ref(**overrides)


def test_provenance_owns_exactly_one_public_contract():
    assert provenance_models.__all__ == ["ArtifactRef"]
    classes = []
    for path in PROVENANCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes.extend(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        )
    assert sorted(classes) == ["ArtifactRef", "_ImmutableProvenanceModel"]


def test_provenance_is_not_a_framework():
    """Structurally minimal: two validators, no I/O, no domain coupling."""

    functions: list[str] = []
    imported: set[str] = set()
    for path in PROVENANCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

    assert sorted(functions) == [
        "_require_canonical_text",
        "_require_optional_canonical_text",
    ]
    assert imported <= {"__future__", "typing", "pydantic", "stock_swing_d1"}
    assert not any(
        name.startswith("stock_swing_d1.")
        and not name.startswith("stock_swing_d1.provenance")
        for path in PROVENANCE.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
        for name in ((node.module or ""),)
    )
