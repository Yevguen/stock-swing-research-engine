"""Meta-validation for the frozen Phase 6C.8A authoritative matrix."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable

from tests.earnings.query import test_published_pit_query as authoritative_tests


FROZEN_QRY_CASE_IDS = tuple(
    f"QRY_{suite}{case_number:02d}"
    for suite in "ABCD"
    for case_number in range(1, 11)
)
HARD_GATE_QRY_IDS = frozenset(
    {
        "QRY_A02",
        "QRY_A03",
        "QRY_A04",
        "QRY_A05",
        "QRY_A06",
        "QRY_A09",
        "QRY_A10",
        "QRY_B01",
        "QRY_B02",
        "QRY_B03",
        "QRY_B04",
        "QRY_B05",
        "QRY_B08",
        "QRY_B09",
        "QRY_C04",
        "QRY_C06",
        "QRY_C08",
        "QRY_C10",
        "QRY_D01",
        "QRY_D05",
        "QRY_D09",
    }
)
_QRY_NAME = re.compile(r"^test_(QRY_[A-D][0-9]{2})(?:_|$)")
_PROHIBITED_ACCEPTANCE_MARKS = frozenset({"skip", "skipif", "xfail"})


def _authoritative_functions() -> dict[str, list[Callable[..., object]]]:
    functions: dict[str, list[Callable[..., object]]] = {}
    for name, function in inspect.getmembers(
        authoritative_tests, inspect.isfunction
    ):
        match = _QRY_NAME.match(name)
        if match is not None:
            functions.setdefault(match.group(1), []).append(function)
    return functions


def _mark_names(function: Callable[..., object]) -> frozenset[str]:
    return frozenset(
        mark.name for mark in getattr(function, "pytestmark", ())
    )


def test_authoritative_qry_case_inventory_is_exact() -> None:
    functions = _authoritative_functions()

    assert len(FROZEN_QRY_CASE_IDS) == 40
    assert len(set(FROZEN_QRY_CASE_IDS)) == 40
    assert FROZEN_QRY_CASE_IDS[0] == "QRY_A01"
    assert FROZEN_QRY_CASE_IDS[-1] == "QRY_D10"
    assert set(functions) == set(FROZEN_QRY_CASE_IDS)
    assert all(len(case_functions) == 1 for case_functions in functions.values())


def test_authoritative_hard_gate_mapping_is_exact() -> None:
    functions = _authoritative_functions()
    observed_hard_gates = {
        case_id
        for case_id, (function,) in functions.items()
        if "hard_gate" in _mark_names(function)
    }

    assert len(HARD_GATE_QRY_IDS) == 21
    assert HARD_GATE_QRY_IDS <= set(FROZEN_QRY_CASE_IDS)
    assert observed_hard_gates == HARD_GATE_QRY_IDS


def test_authoritative_qry_cases_have_no_nonexecuting_marks() -> None:
    for case_id, (function,) in _authoritative_functions().items():
        prohibited = _mark_names(function) & _PROHIBITED_ACCEPTANCE_MARKS
        assert not prohibited, f"{case_id} has prohibited marks: {sorted(prohibited)}"
