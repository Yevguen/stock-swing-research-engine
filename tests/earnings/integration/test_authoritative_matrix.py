"""Meta-validation for the Phase 6C.8B authoritative acceptance matrix."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable

from tests.earnings.integration import (
    test_entry_integration,
    test_open_position_integration,
    test_orchestration_safety,
    test_revision_integration,
)


FROZEN_EINT_CASE_IDS = tuple(
    f"EINT_{suite}{case_number:02d}"
    for suite in "ABCD"
    for case_number in range(1, 11)
)
HARD_GATE_EINT_IDS = frozenset(
    {
        "EINT_A03",
        "EINT_A04",
        "EINT_A07",
        "EINT_A08",
        "EINT_A09",
        "EINT_B02",
        "EINT_B03",
        "EINT_B05",
        "EINT_B06",
        "EINT_B07",
        "EINT_B08",
        "EINT_B09",
        "EINT_C02",
        "EINT_C03",
        "EINT_C04",
        "EINT_C05",
        "EINT_C06",
        "EINT_C08",
        "EINT_C10",
        "EINT_D01",
        "EINT_D02",
        "EINT_D06",
    }
)
_EINT_NAME = re.compile(r"^test_(EINT_[A-D][0-9]{2})(?:_|$)")
_PROHIBITED_ACCEPTANCE_MARKS = frozenset({"skip", "skipif", "xfail"})
_AUTHORITATIVE_MODULES = (
    test_entry_integration,
    test_open_position_integration,
    test_revision_integration,
    test_orchestration_safety,
)


def _authoritative_functions() -> dict[str, list[Callable[..., object]]]:
    functions: dict[str, list[Callable[..., object]]] = {}
    for module in _AUTHORITATIVE_MODULES:
        for name, function in inspect.getmembers(module, inspect.isfunction):
            match = _EINT_NAME.match(name)
            if match is not None:
                functions.setdefault(match.group(1), []).append(function)
    return functions


def _mark_names(function: Callable[..., object]) -> frozenset[str]:
    return frozenset(
        mark.name for mark in getattr(function, "pytestmark", ())
    )


def test_authoritative_eint_case_inventory_is_exact() -> None:
    functions = _authoritative_functions()

    assert len(FROZEN_EINT_CASE_IDS) == 40
    assert len(set(FROZEN_EINT_CASE_IDS)) == 40
    assert all(
        sum(case_id.startswith(f"EINT_{suite}") for case_id in functions) == 10
        for suite in "ABCD"
    )
    assert set(functions) == set(FROZEN_EINT_CASE_IDS)
    assert all(len(case_functions) == 1 for case_functions in functions.values())


def test_authoritative_eint_hard_gate_mapping_is_exact() -> None:
    functions = _authoritative_functions()
    observed_hard_gates = {
        case_id
        for case_id, (function,) in functions.items()
        if "hard_gate_6c8b" in _mark_names(function)
    }

    assert len(HARD_GATE_EINT_IDS) == 22
    assert HARD_GATE_EINT_IDS <= set(FROZEN_EINT_CASE_IDS)
    assert observed_hard_gates == HARD_GATE_EINT_IDS


def test_authoritative_eint_cases_have_no_nonexecuting_marks() -> None:
    for case_id, (function,) in _authoritative_functions().items():
        prohibited = _mark_names(function) & _PROHIBITED_ACCEPTANCE_MARKS
        assert not prohibited, f"{case_id} has prohibited marks: {sorted(prohibited)}"
