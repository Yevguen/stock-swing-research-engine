"""Pure deterministic identifiers for future execution-event adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from stock_swing_d1.execution.costs.models import ExecutionCostValidationError


_SECURITY_ID = re.compile(r"NORGATE:(?P<asset_id>[1-9][0-9]*)")
_ENTRY_EXECUTION_ID = re.compile(
    r"ENTRY:(?P<session>[0-9]{4}-[0-9]{2}-[0-9]{2}):"
    r"(?P<security_id>NORGATE:[1-9][0-9]*)"
)
_SELL_EXECUTION_ID = re.compile(
    r"EXIT:(?P<entry_execution_id>ENTRY:[0-9]{4}-[0-9]{2}-[0-9]{2}:"
    r"NORGATE:[1-9][0-9]*):(?P<session>[0-9]{4}-[0-9]{2}-[0-9]{2}):"
    r"(?P<security_id>NORGATE:[1-9][0-9]*)"
)


def _require_session(value: object, *, field_name: str) -> date:
    if type(value) is not date:
        raise ExecutionCostValidationError(
            "INVALID_IDENTIFIER_SESSION", f"{field_name} must be a genuine date"
        )
    return value


def _require_security_id(value: object) -> str:
    if type(value) is not str or _SECURITY_ID.fullmatch(value) is None:
        raise ExecutionCostValidationError(
            "INVALID_SECURITY_ID",
            "security_id must use NORGATE:<positive AssetId>",
        )
    return value


def _require_entry_execution_id(value: object, *, security_id: str) -> str:
    if type(value) is not str or value != value.strip():
        raise ExecutionCostValidationError(
            "INVALID_ENTRY_EXECUTION_ID",
            "entry_execution_id must be canonical and non-blank",
        )
    match = _ENTRY_EXECUTION_ID.fullmatch(value)
    if match is None or match.group("security_id") != security_id:
        raise ExecutionCostValidationError(
            "INVALID_ENTRY_EXECUTION_ID",
            "entry_execution_id must be canonical and match security_id",
        )
    try:
        date.fromisoformat(match.group("session"))
    except ValueError as error:
        raise ExecutionCostValidationError(
            "INVALID_ENTRY_EXECUTION_ID",
            "entry_execution_id contains an invalid session",
        ) from error
    return value


def _require_sell_execution_id(value: object) -> str:
    if type(value) is not str or value != value.strip():
        raise ExecutionCostValidationError(
            "INVALID_SELL_EXECUTION_ID",
            "sell_execution_id must be canonical and non-blank",
        )
    match = _SELL_EXECUTION_ID.fullmatch(value)
    if match is None:
        raise ExecutionCostValidationError(
            "INVALID_SELL_EXECUTION_ID",
            "sell_execution_id must use the frozen EXIT canonical form",
        )
    try:
        date.fromisoformat(match.group("session"))
    except ValueError as error:
        raise ExecutionCostValidationError(
            "INVALID_SELL_EXECUTION_ID",
            "sell_execution_id contains an invalid session",
        ) from error
    entry = _ENTRY_EXECUTION_ID.fullmatch(match.group("entry_execution_id"))
    if entry is None or entry.group("security_id") != match.group("security_id"):
        raise ExecutionCostValidationError(
            "INVALID_SELL_EXECUTION_ID",
            "sell and entry execution IDs must identify the same security",
        )
    try:
        date.fromisoformat(entry.group("session"))
    except ValueError as error:
        raise ExecutionCostValidationError(
            "INVALID_SELL_EXECUTION_ID",
            "sell_execution_id contains an invalid entry session",
        ) from error
    return value


@dataclass(frozen=True, slots=True)
class ExecutionIdentifierService:
    """Create IDs solely from stable logical execution identity."""

    def buy_source_order_id(
        self, *, allocation_session: date, security_id: str
    ) -> str:
        session = _require_session(
            allocation_session, field_name="allocation_session"
        )
        security = _require_security_id(security_id)
        return f"ALLOC:{session.isoformat()}:{security}"

    def buy_execution_id(self, *, entry_session: date, security_id: str) -> str:
        session = _require_session(entry_session, field_name="entry_session")
        security = _require_security_id(security_id)
        return f"ENTRY:{session.isoformat()}:{security}"

    def sell_source_order_id(
        self,
        *,
        entry_execution_id: str,
        exit_session: date,
        security_id: str,
    ) -> str:
        session = _require_session(exit_session, field_name="exit_session")
        security = _require_security_id(security_id)
        entry = _require_entry_execution_id(
            entry_execution_id, security_id=security
        )
        return f"EXIT-ORDER:{entry}:{session.isoformat()}:{security}"

    def sell_execution_id(
        self,
        *,
        entry_execution_id: str,
        exit_session: date,
        security_id: str,
    ) -> str:
        session = _require_session(exit_session, field_name="exit_session")
        security = _require_security_id(security_id)
        entry = _require_entry_execution_id(
            entry_execution_id, security_id=security
        )
        return f"EXIT:{entry}:{session.isoformat()}:{security}"

    def settlement_id(self, *, sell_execution_id: str) -> str:
        execution = _require_sell_execution_id(sell_execution_id)
        return f"SETTLE:{execution}"


__all__ = ["ExecutionIdentifierService"]
