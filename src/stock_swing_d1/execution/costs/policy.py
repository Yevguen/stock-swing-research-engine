"""Local configuration loading and upstream slippage-parity validation."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from stock_swing_d1.execution.costs.hashing import (
    compute_execution_cost_policy_fingerprint,
)
from stock_swing_d1.execution.costs.models import (
    BACKTEST_EXECUTION_COST_POLICY_SCHEMA_VERSION,
    BacktestExecutionCostPolicy,
    CommissionCostModel,
    CommissionCostPolicy,
    ExecutionCostPolicyRef,
    ExecutionCostValidationError,
    SettlementPolicyModel,
    SettlementPolicyRef,
    SlippagePolicy,
    SpreadCostModel,
    SpreadCostPolicy,
)


_INTEGER = re.compile(r"[-+]?[0-9]+")
_FLOAT = re.compile(
    r"[-+]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
)


def _parse_scalar(token: str, *, line_number: int) -> object:
    if not token:
        raise ExecutionCostValidationError(
            "INVALID_POLICY_CONFIG",
            f"line {line_number} has a missing scalar value",
        )
    if token.startswith('"'):
        try:
            value = json.loads(token)
        except json.JSONDecodeError as error:
            raise ExecutionCostValidationError(
                "INVALID_POLICY_CONFIG",
                f"line {line_number} has an invalid quoted value",
            ) from error
        if type(value) is not str:
            raise ExecutionCostValidationError(
                "INVALID_POLICY_CONFIG",
                f"line {line_number} must contain a string scalar",
            )
        return value
    if token.startswith("'") and token.endswith("'") and len(token) >= 2:
        return token[1:-1].replace("''", "'")
    if _INTEGER.fullmatch(token):
        return int(token)
    if _FLOAT.fullmatch(token):
        return float(token)
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    if token.lower() in {"null", "none", "~"}:
        return None
    if any(character in token for character in "[]{}&*!|>@`"):
        raise ExecutionCostValidationError(
            "INVALID_POLICY_CONFIG",
            f"line {line_number} uses unsupported YAML syntax",
        )
    return token


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, character in enumerate(line):
        if character == "'" and not in_double:
            in_single = not in_single
        elif character == '"' and not in_single:
            in_double = not in_double
        elif character == "#" and not in_single and not in_double:
            if index == 0 or line[index - 1].isspace():
                return line[:index].rstrip()
    return line.rstrip()


def _parse_versioned_mapping(text: str) -> dict[str, object]:
    """Parse the intentionally small mapping-only YAML surface used here."""

    result: dict[str, object] = {}
    current_section: dict[str, object] | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        if "\t" in line:
            raise ExecutionCostValidationError(
                "INVALID_POLICY_CONFIG",
                f"line {line_number} must not use tab indentation",
            )
        indentation = len(line) - len(line.lstrip(" "))
        if indentation not in (0, 2):
            raise ExecutionCostValidationError(
                "INVALID_POLICY_CONFIG",
                f"line {line_number} must use zero or two spaces of indentation",
            )
        content = line[indentation:]
        if ":" not in content:
            raise ExecutionCostValidationError(
                "INVALID_POLICY_CONFIG",
                f"line {line_number} must be a mapping entry",
            )
        key, token = content.split(":", 1)
        key = key.strip()
        token = token.strip()
        if not key or key != key.strip():
            raise ExecutionCostValidationError(
                "INVALID_POLICY_CONFIG", f"line {line_number} has an invalid key"
            )
        if indentation == 0:
            if key in result:
                raise ExecutionCostValidationError(
                    "INVALID_POLICY_CONFIG",
                    f"line {line_number} duplicates key {key!r}",
                )
            if not token:
                section: dict[str, object] = {}
                result[key] = section
                current_section = section
            else:
                result[key] = _parse_scalar(token, line_number=line_number)
                current_section = None
            continue
        if current_section is None:
            raise ExecutionCostValidationError(
                "INVALID_POLICY_CONFIG",
                f"line {line_number} has no parent mapping",
            )
        if key in current_section:
            raise ExecutionCostValidationError(
                "INVALID_POLICY_CONFIG",
                f"line {line_number} duplicates key {key!r}",
            )
        current_section[key] = _parse_scalar(token, line_number=line_number)
    return result


def _require_mapping(
    value: object,
    *,
    name: str,
    expected_keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise ExecutionCostValidationError(
            "INVALID_POLICY_CONFIG", f"{name} must be a mapping"
        )
    actual_keys = frozenset(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ExecutionCostValidationError(
            "INVALID_POLICY_CONFIG",
            f"{name} keys are invalid; missing={missing}, extra={extra}",
        )
    return value


def _decimal_from_config(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)) or type(value) not in (str, int, Decimal):
        raise ExecutionCostValidationError(
            "INVALID_DECIMAL_CONFIG",
            f"{field_name} must be an integer or quoted decimal string",
        )
    try:
        converted = Decimal(str(value))
    except InvalidOperation as error:
        raise ExecutionCostValidationError(
            "INVALID_DECIMAL_CONFIG", f"{field_name} is not a valid decimal"
        ) from error
    if not converted.is_finite():
        raise ExecutionCostValidationError(
            "NON_FINITE_DECIMAL", f"{field_name} must be finite"
        )
    return converted


def _enum_from_config(enum_type: type, value: object, *, field_name: str) -> object:
    if type(value) is not str:
        raise ExecutionCostValidationError(
            "INVALID_POLICY_CONFIG", f"{field_name} must be a string"
        )
    try:
        return enum_type(value)
    except ValueError as error:
        raise ExecutionCostValidationError(
            f"UNSUPPORTED_{field_name.upper().replace('.', '_')}",
            f"{field_name} has an unsupported value",
        ) from error


def load_backtest_execution_cost_policy(
    path: str | Path,
) -> BacktestExecutionCostPolicy:
    """Load only an explicitly supplied local Phase 15C policy file."""

    if isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise ExecutionCostValidationError(
            "INVALID_POLICY_PATH", "path must be an explicit local path"
        )
    if isinstance(path, str) and (not path or path != path.strip()):
        raise ExecutionCostValidationError(
            "INVALID_POLICY_PATH", "path must be a non-blank canonical path"
        )
    policy_path = Path(path)
    try:
        text = policy_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ExecutionCostValidationError(
            "POLICY_READ_FAILED", "the local policy file could not be read"
        ) from error
    raw = _require_mapping(
        _parse_versioned_mapping(text),
        name="policy",
        expected_keys=frozenset(
            {
                "schema_version",
                "policy_id",
                "currency",
                "slippage",
                "spread",
                "commission",
                "settlement",
            }
        ),
    )
    slippage = _require_mapping(
        raw["slippage"],
        name="slippage",
        expected_keys=frozenset(
            {"entry_bps", "protective_exit_bps", "administrative_exit_bps"}
        ),
    )
    spread = _require_mapping(
        raw["spread"],
        name="spread",
        expected_keys=frozenset({"model", "bps_per_side"}),
    )
    commission = _require_mapping(
        raw["commission"],
        name="commission",
        expected_keys=frozenset(
            {
                "model",
                "per_share_usd",
                "minimum_per_order_usd",
                "maximum_fraction_of_notional",
            }
        ),
    )
    settlement = _require_mapping(
        raw["settlement"],
        name="settlement",
        expected_keys=frozenset({"model"}),
    )
    for field_name in ("schema_version", "policy_id", "currency"):
        if type(raw[field_name]) is not str:
            raise ExecutionCostValidationError(
                "INVALID_POLICY_CONFIG", f"{field_name} must be a string"
            )
    return BacktestExecutionCostPolicy(
        schema_version=raw["schema_version"],
        policy_id=raw["policy_id"],
        currency=raw["currency"],
        slippage=SlippagePolicy(
            entry_bps=_decimal_from_config(
                slippage["entry_bps"], field_name="slippage.entry_bps"
            ),
            protective_exit_bps=_decimal_from_config(
                slippage["protective_exit_bps"],
                field_name="slippage.protective_exit_bps",
            ),
            administrative_exit_bps=_decimal_from_config(
                slippage["administrative_exit_bps"],
                field_name="slippage.administrative_exit_bps",
            ),
        ),
        spread=SpreadCostPolicy(
            model=_enum_from_config(
                SpreadCostModel,
                spread["model"],
                field_name="spread_model",
            ),
            bps_per_side=_decimal_from_config(
                spread["bps_per_side"], field_name="spread.bps_per_side"
            ),
        ),
        commission=CommissionCostPolicy(
            model=_enum_from_config(
                CommissionCostModel,
                commission["model"],
                field_name="commission_model",
            ),
            per_share_usd=_decimal_from_config(
                commission["per_share_usd"],
                field_name="commission.per_share_usd",
            ),
            minimum_per_order_usd=_decimal_from_config(
                commission["minimum_per_order_usd"],
                field_name="commission.minimum_per_order_usd",
            ),
            maximum_fraction_of_notional=_decimal_from_config(
                commission["maximum_fraction_of_notional"],
                field_name="commission.maximum_fraction_of_notional",
            ),
        ),
        settlement=SettlementPolicyRef(
            model=_enum_from_config(
                SettlementPolicyModel,
                settlement["model"],
                field_name="settlement_model",
            )
        ),
    )


def build_execution_cost_policy_ref(
    policy: BacktestExecutionCostPolicy,
) -> ExecutionCostPolicyRef:
    """Create the deterministic immutable reference retained by results."""

    if type(policy) is not BacktestExecutionCostPolicy:
        raise ExecutionCostValidationError(
            "INVALID_POLICY", "policy must be a BacktestExecutionCostPolicy"
        )
    return ExecutionCostPolicyRef(
        policy_id=policy.policy_id,
        policy_fingerprint=compute_execution_cost_policy_fingerprint(policy),
    )


def _upstream_bps(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ExecutionCostValidationError(
            "INVALID_PARITY_INPUT", f"{field_name} must be a finite numeric value"
        )
    try:
        converted = Decimal(str(value))
    except InvalidOperation as error:
        raise ExecutionCostValidationError(
            "INVALID_PARITY_INPUT", f"{field_name} must be a finite numeric value"
        ) from error
    if not converted.is_finite():
        raise ExecutionCostValidationError(
            "INVALID_PARITY_INPUT", f"{field_name} must be finite"
        )
    return converted


def validate_execution_cost_policy_parity(
    policy: BacktestExecutionCostPolicy,
    *,
    entry_slippage_bps: object,
    protective_exit_slippage_bps: object,
) -> None:
    """Fail unless Phase 15C declarations equal upstream Phase 9/10 owners."""

    if type(policy) is not BacktestExecutionCostPolicy:
        raise ExecutionCostValidationError(
            "INVALID_POLICY", "policy must be a BacktestExecutionCostPolicy"
        )
    actual_entry = _upstream_bps(
        entry_slippage_bps, field_name="entry_slippage_bps"
    )
    actual_protective = _upstream_bps(
        protective_exit_slippage_bps,
        field_name="protective_exit_slippage_bps",
    )
    if policy.slippage.entry_bps != actual_entry:
        raise ExecutionCostValidationError(
            "ENTRY_SLIPPAGE_PARITY_MISMATCH",
            "Phase 15C entry slippage does not match the upstream owner",
        )
    if policy.slippage.protective_exit_bps != actual_protective:
        raise ExecutionCostValidationError(
            "PROTECTIVE_EXIT_SLIPPAGE_PARITY_MISMATCH",
            "Phase 15C protective-exit slippage does not match the upstream owner",
        )


__all__ = [
    "build_execution_cost_policy_ref",
    "load_backtest_execution_cost_policy",
    "validate_execution_cost_policy_parity",
]
