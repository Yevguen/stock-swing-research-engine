"""The immutable canonical identity of the frozen baseline strategy v0.1.

This module is *descriptive*. It does not evaluate signals, size positions,
place stops or decide exits: :mod:`stock_swing_d1.strategy.baseline.service`
and the frozen Phase 7/10/11/15B owners remain the single semantic authority
for baseline behaviour. What this module adds is an immutable, strictly
validated, deterministically fingerprinted *description* of those semantics, so
that a canonical research experiment can prove which exact strategy produced
it.

Three rules follow from that, and every design choice here serves them:

* ``config/strategy.yaml`` is a **checked declaration**, never a runtime
  authority. Nothing on the signal path reads it, so an edited YAML value can
  never change a trading decision -- it can only fail to load.
* Where an authoritative constant exists in a package ``strategy`` may import,
  the configuration is *built from* that constant, so drift is impossible.
* Where the owner sits in a package ``strategy`` may not import without
  creating a cycle (``risk`` and ``execution`` both import ``strategy``), the
  value is declared here and proven equal to its owner by
  :func:`verify_baseline_strategy_configuration_parity`, whose caller injects
  the authoritative constants. This mirrors the frozen Phase 15C precedent,
  ``execution.costs.policy.validate_execution_cost_policy_parity``.

A mismatch anywhere fails closed. YAML never overrides code.
"""

from __future__ import annotations

import json
import re
from math import isfinite
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator

from stock_swing_d1.indicators import TechnicalIndicatorRow
from stock_swing_d1.strategy.baseline.configuration_hashing import (
    baseline_strategy_domain_sha256,
)
from stock_swing_d1.strategy.baseline.models import (
    BaselineSignalAction,
    BaselineStrategyValidationError,
)
from stock_swing_d1.strategy.baseline.service import (
    _ATR_MINIMUM_FRACTION,
    _REQUIRED_PRICE_BASIS,
    _RSI_MINIMUM,
)


BASELINE_STRATEGY_CONFIGURATION_SCHEMA_VERSION = (
    "baseline_strategy_configuration.v0.1"
)
BASELINE_STRATEGY_CONFIGURATION_ID = "baseline_strategy_v0.1"
BASELINE_STRATEGY_CONFIGURATION_ARTIFACT_TYPE = (
    "baseline_strategy_configuration"
)
BASELINE_STRATEGY_CONFIGURATION_HASH_DOMAIN = (
    "baseline_strategy_configuration.v0.1"
)

# Declared here because their authoritative owners live in packages that
# import ``strategy``; binding them by import would create a cycle. Each is
# proven equal to its owner by verify_baseline_strategy_configuration_parity.
_DECLARED_SMA_FAST_PERIOD = 20
_DECLARED_SMA_SLOW_PERIOD = 50
_DECLARED_RSI_PERIOD = 14
_DECLARED_ATR_PERIOD = 14
_DECLARED_TARGET_RISK_FRACTION = 0.005
_DECLARED_STOP_RISK_ATR_MULTIPLE = 2.0
_DECLARED_TAKE_PROFIT_R_MULTIPLE = 2.0
_DECLARED_MAXIMUM_HOLDING_SESSIONS = 10


class BaselineStrategyConfigurationError(BaselineStrategyValidationError):
    """A canonical baseline strategy configuration contract was violated."""


def _require_exact_bool(value: object) -> object:
    if type(value) is not bool:
        raise ValueError("must be an explicit Boolean")
    return value


def _require_exact_int(value: object) -> object:
    if type(value) is not int:
        raise ValueError("must be an exact int; bool and float are forbidden")
    return value


def _require_positive_int(value: object) -> object:
    _require_exact_int(value)
    if value <= 0:
        raise ValueError("must be a positive whole number")
    return value


def _require_finite_float(value: object) -> object:
    if type(value) is not float:
        raise ValueError("must be an exact float; int and bool are forbidden")
    if not isfinite(value):
        raise ValueError("must be finite; NaN and infinity are forbidden")
    return value


def _require_positive_float(value: object) -> object:
    _require_finite_float(value)
    if value <= 0.0:
        raise ValueError("must be greater than zero")
    return value


def _require_fraction(value: object) -> object:
    _require_positive_float(value)
    if value >= 1.0:
        raise ValueError("must be a fraction strictly below one")
    return value


def _require_sha256(value: object) -> object:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("must be a lowercase 64-character SHA-256 digest")
    return value


_ExactTrue = Annotated[Literal[True], BeforeValidator(_require_exact_bool)]
_ExactZero = Annotated[Literal[0], BeforeValidator(_require_exact_int)]
_PositiveInt = Annotated[int, BeforeValidator(_require_positive_int)]
_FiniteFloat = Annotated[float, BeforeValidator(_require_finite_float)]
_PositiveFloat = Annotated[float, BeforeValidator(_require_positive_float)]
_Fraction = Annotated[float, BeforeValidator(_require_fraction)]
_Sha256 = Annotated[str, BeforeValidator(_require_sha256)]


class _ImmutableConfigurationModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_default=True,
    )


class BaselineDecisionInputs(_ImmutableConfigurationModel):
    """Which completed-D1 price basis and indicator windows the baseline reads.

    Phase 7 computes a wider indicator inventory (SMA 100/200, volume averages).
    Only the four windows below are consumed by the baseline decision, so only
    these four are part of the strategy's identity.
    """

    signal_price_basis: Literal["capital_special_adjusted"]
    sma_fast_period: _PositiveInt
    sma_slow_period: _PositiveInt
    rsi_period: _PositiveInt
    atr_period: _PositiveInt

    @model_validator(mode="after")
    def validate_windows(self) -> Self:
        if self.sma_fast_period >= self.sma_slow_period:
            raise ValueError(
                "the fast SMA window must be shorter than the slow SMA window"
            )
        return self


class BaselineEntryConditions(_ImmutableConfigurationModel):
    """The complete conjunction Phase 8 requires for a valid long signal.

    Every member is required, which is why each Boolean is ``Literal[True]``:
    Phase 8 combines them with ``all(...)``, so a configuration that claimed a
    condition was optional would not describe the frozen baseline.
    """

    close_above_sma_slow_required: _ExactTrue
    sma_fast_above_sma_slow_required: _ExactTrue
    rsi_minimum: _FiniteFloat
    rsi_comparison: Literal["strictly_greater_than"]
    atr_minimum_fraction: _PositiveFloat
    atr_comparison: Literal["greater_than_or_equal"]
    universe_eligibility_required: _ExactTrue
    earnings_entry_permission_required: _ExactTrue


class BaselineTradeChronology(_ImmutableConfigurationModel):
    """Direction, the frozen T -> T+1 boundary, and the holding ceiling."""

    trade_direction: Literal["LONG_ONLY"]
    entry_execution_offset_sessions: Literal[1]
    maximum_holding_sessions: _PositiveInt


class BaselineRiskGeometry(_ImmutableConfigurationModel):
    """Sizing budget and the protective stop / take-profit geometry.

    These are strategy semantics, not execution-cost semantics: slippage,
    spread, commission and settlement remain owned by the separately bound
    Phase 15C execution-cost policy.
    """

    target_risk_fraction: _Fraction
    stop_risk_atr_multiple: _PositiveFloat
    take_profit_r_multiple: _PositiveFloat


class BaselineStrategyConfiguration(_ImmutableConfigurationModel):
    """One immutable, strictly validated description of the frozen baseline."""

    schema_version: Literal["baseline_strategy_configuration.v0.1"]
    configuration_id: Literal["baseline_strategy_v0.1"]
    # ``_ExactZero`` is ``Literal[0]`` guarded by an exact-int check, so the
    # canonical baseline identity cannot be constructed with any other count,
    # and cannot be smuggled through as the bool ``False``, which would
    # otherwise compare equal to 0.
    tuned_parameter_count: _ExactZero
    decision_inputs: BaselineDecisionInputs
    entry_conditions: BaselineEntryConditions
    chronology: BaselineTradeChronology
    risk: BaselineRiskGeometry


class BaselineStrategyConfigurationRef(_ImmutableConfigurationModel):
    """Storage-independent semantic identity of one configuration artifact.

    Field names match the Phase 15D artifact-reference shape deliberately, so a
    downstream consumer can bind this identity without ``strategy`` importing
    Phase 15D, a direction the frozen dependency chain forbids.
    """

    artifact_type: Literal["baseline_strategy_configuration"]
    schema_version: Literal["baseline_strategy_configuration.v0.1"]
    content_sha256: _Sha256


def compute_baseline_strategy_configuration_fingerprint(
    configuration: BaselineStrategyConfiguration,
) -> str:
    """Return the domain-separated lowercase SHA-256 semantic fingerprint."""

    if type(configuration) is not BaselineStrategyConfiguration:
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIGURATION",
            "configuration must be a BaselineStrategyConfiguration",
        )
    return baseline_strategy_domain_sha256(
        BASELINE_STRATEGY_CONFIGURATION_HASH_DOMAIN, configuration
    )


def build_baseline_strategy_configuration() -> BaselineStrategyConfiguration:
    """Build the canonical configuration from the authoritative frozen owners.

    Phase 8's own thresholds and price basis are read directly from
    :mod:`stock_swing_d1.strategy.baseline.service`, so they cannot drift. The
    remaining values are declared and proven by
    :func:`verify_baseline_strategy_configuration_parity`.
    """

    return BaselineStrategyConfiguration(
        schema_version=BASELINE_STRATEGY_CONFIGURATION_SCHEMA_VERSION,
        configuration_id=BASELINE_STRATEGY_CONFIGURATION_ID,
        tuned_parameter_count=0,
        decision_inputs=BaselineDecisionInputs(
            signal_price_basis=_REQUIRED_PRICE_BASIS,
            sma_fast_period=_DECLARED_SMA_FAST_PERIOD,
            sma_slow_period=_DECLARED_SMA_SLOW_PERIOD,
            rsi_period=_DECLARED_RSI_PERIOD,
            atr_period=_DECLARED_ATR_PERIOD,
        ),
        entry_conditions=BaselineEntryConditions(
            close_above_sma_slow_required=True,
            sma_fast_above_sma_slow_required=True,
            rsi_minimum=_RSI_MINIMUM,
            rsi_comparison="strictly_greater_than",
            atr_minimum_fraction=_ATR_MINIMUM_FRACTION,
            atr_comparison="greater_than_or_equal",
            universe_eligibility_required=True,
            earnings_entry_permission_required=True,
        ),
        chronology=BaselineTradeChronology(
            trade_direction="LONG_ONLY",
            entry_execution_offset_sessions=1,
            maximum_holding_sessions=_DECLARED_MAXIMUM_HOLDING_SESSIONS,
        ),
        risk=BaselineRiskGeometry(
            target_risk_fraction=_DECLARED_TARGET_RISK_FRACTION,
            stop_risk_atr_multiple=_DECLARED_STOP_RISK_ATR_MULTIPLE,
            take_profit_r_multiple=_DECLARED_TAKE_PROFIT_R_MULTIPLE,
        ),
    )


def verify_baseline_strategy_configuration(
    configuration: BaselineStrategyConfiguration,
) -> BaselineStrategyConfiguration:
    """Fail closed unless the configuration equals the in-reach frozen owners.

    Checks every semantic ``strategy`` can reach without an import cycle: the
    Phase 8 thresholds and price basis, the long-only action inventory, and the
    public Phase 7 indicator fields implied by the declared windows.
    """

    if type(configuration) is not BaselineStrategyConfiguration:
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIGURATION",
            "configuration must be a BaselineStrategyConfiguration",
        )

    entry = configuration.entry_conditions
    inputs = configuration.decision_inputs
    if inputs.signal_price_basis != _REQUIRED_PRICE_BASIS:
        raise BaselineStrategyConfigurationError(
            "SIGNAL_PRICE_BASIS_MISMATCH",
            "declared signal price basis is not the frozen Phase 8 basis",
        )
    if entry.rsi_minimum != _RSI_MINIMUM:
        raise BaselineStrategyConfigurationError(
            "RSI_MINIMUM_MISMATCH",
            "declared RSI minimum is not the frozen Phase 8 threshold",
        )
    if entry.atr_minimum_fraction != _ATR_MINIMUM_FRACTION:
        raise BaselineStrategyConfigurationError(
            "ATR_MINIMUM_FRACTION_MISMATCH",
            "declared ATR minimum fraction is not the frozen Phase 8 threshold",
        )
    if (
        inputs.sma_fast_period,
        inputs.sma_slow_period,
        inputs.rsi_period,
        inputs.atr_period,
    ) != (
        _DECLARED_SMA_FAST_PERIOD,
        _DECLARED_SMA_SLOW_PERIOD,
        _DECLARED_RSI_PERIOD,
        _DECLARED_ATR_PERIOD,
    ):
        raise BaselineStrategyConfigurationError(
            "INDICATOR_WINDOW_MISMATCH",
            "declared indicator windows are not the frozen Phase 7/8 windows",
        )
    if configuration.chronology.maximum_holding_sessions != (
        _DECLARED_MAXIMUM_HOLDING_SESSIONS
    ):
        raise BaselineStrategyConfigurationError(
            "MAXIMUM_HOLDING_SESSIONS_MISMATCH",
            "declared maximum holding sessions is not the frozen ceiling",
        )
    if configuration.risk.target_risk_fraction != (
        _DECLARED_TARGET_RISK_FRACTION
    ):
        raise BaselineStrategyConfigurationError(
            "TARGET_RISK_FRACTION_MISMATCH",
            "declared target risk fraction is not the frozen Phase 11 value",
        )
    if (
        configuration.risk.stop_risk_atr_multiple,
        configuration.risk.take_profit_r_multiple,
    ) != (
        _DECLARED_STOP_RISK_ATR_MULTIPLE,
        _DECLARED_TAKE_PROFIT_R_MULTIPLE,
    ):
        raise BaselineStrategyConfigurationError(
            "RISK_GEOMETRY_MISMATCH",
            "declared stop and take-profit geometry is not the frozen Phase 10 geometry",
        )
    if set(BaselineSignalAction) != {
        BaselineSignalAction.VALID_LONG_SIGNAL,
        BaselineSignalAction.NO_SIGNAL,
    }:
        raise BaselineStrategyConfigurationError(
            "TRADE_DIRECTION_MISMATCH",
            "the Phase 8 action inventory is no longer long-only",
        )

    declared_indicator_fields = (
        f"sma_{inputs.sma_fast_period}",
        f"sma_{inputs.sma_slow_period}",
        f"rsi_{inputs.rsi_period}",
        f"atr_{inputs.atr_period}",
    )
    available = set(TechnicalIndicatorRow.model_fields)
    missing = [
        field_name
        for field_name in declared_indicator_fields
        if field_name not in available
    ]
    if missing:
        raise BaselineStrategyConfigurationError(
            "INDICATOR_WINDOW_MISMATCH",
            f"Phase 7 publishes no indicator for declared windows: {missing}",
        )
    return configuration


def verify_baseline_strategy_configuration_parity(
    configuration: BaselineStrategyConfiguration,
    *,
    target_risk_fraction: object,
    maximum_holding_sessions: object,
    earnings_maximum_holding_sessions: object,
) -> BaselineStrategyConfiguration:
    """Fail unless declared values equal the injected authoritative owners.

    ``risk`` and ``execution`` both import ``strategy``, so their constants are
    injected by the caller rather than imported here. This mirrors the frozen
    Phase 15C precedent ``validate_execution_cost_policy_parity``.
    """

    verify_baseline_strategy_configuration(configuration)
    if type(target_risk_fraction) is not float or not isfinite(
        target_risk_fraction
    ):
        raise BaselineStrategyConfigurationError(
            "INVALID_PARITY_INPUT",
            "target_risk_fraction must be a finite float",
        )
    if (
        type(maximum_holding_sessions) is not int
        or type(earnings_maximum_holding_sessions) is not int
    ):
        raise BaselineStrategyConfigurationError(
            "INVALID_PARITY_INPUT",
            "maximum holding sessions must be exact integers",
        )
    if configuration.risk.target_risk_fraction != target_risk_fraction:
        raise BaselineStrategyConfigurationError(
            "TARGET_RISK_FRACTION_MISMATCH",
            "declared target risk fraction is not the frozen Phase 11 value",
        )
    if maximum_holding_sessions != earnings_maximum_holding_sessions:
        raise BaselineStrategyConfigurationError(
            "MAXIMUM_HOLDING_SESSIONS_MISMATCH",
            "the Phase 15B and earnings-risk holding ceilings disagree",
        )
    if configuration.chronology.maximum_holding_sessions != (
        maximum_holding_sessions
    ):
        raise BaselineStrategyConfigurationError(
            "MAXIMUM_HOLDING_SESSIONS_MISMATCH",
            "declared maximum holding sessions is not the frozen ceiling",
        )
    return configuration


def build_baseline_strategy_configuration_ref(
    configuration: BaselineStrategyConfiguration,
) -> BaselineStrategyConfigurationRef:
    """Create the deterministic immutable reference retained by experiments."""

    verify_baseline_strategy_configuration(configuration)
    fingerprint = compute_baseline_strategy_configuration_fingerprint(
        configuration
    )
    return BaselineStrategyConfigurationRef(
        artifact_type=BASELINE_STRATEGY_CONFIGURATION_ARTIFACT_TYPE,
        schema_version=configuration.schema_version,
        content_sha256=fingerprint,
    )


_INTEGER = re.compile(r"[-+]?[0-9]+")
_FLOAT = re.compile(
    r"[-+]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
)

_EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "configuration_id",
        "tuned_parameter_count",
        "decision_inputs",
        "entry_conditions",
        "chronology",
        "risk",
    }
)
_EXPECTED_SECTION_KEYS: dict[str, frozenset[str]] = {
    "decision_inputs": frozenset(BaselineDecisionInputs.model_fields),
    "entry_conditions": frozenset(BaselineEntryConditions.model_fields),
    "chronology": frozenset(BaselineTradeChronology.model_fields),
    "risk": frozenset(BaselineRiskGeometry.model_fields),
}


def _parse_scalar(token: str, *, line_number: int) -> object:
    if not token:
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIG",
            f"line {line_number} has a missing scalar value",
        )
    if token.startswith('"'):
        try:
            value = json.loads(token)
        except json.JSONDecodeError as error:
            raise BaselineStrategyConfigurationError(
                "INVALID_STRATEGY_CONFIG",
                f"line {line_number} has an invalid quoted value",
            ) from error
        if type(value) is not str:
            raise BaselineStrategyConfigurationError(
                "INVALID_STRATEGY_CONFIG",
                f"line {line_number} must contain a string scalar",
            )
        return value
    if _INTEGER.fullmatch(token):
        return int(token)
    if _FLOAT.fullmatch(token):
        return float(token)
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    if any(character in token for character in "[]{}&*!|>@`'"):
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIG",
            f"line {line_number} uses unsupported YAML syntax",
        )
    return token


def _strip_comment(line: str) -> str:
    in_double = False
    for index, character in enumerate(line):
        if character == '"':
            in_double = not in_double
        elif character == "#" and not in_double:
            if index == 0 or line[index - 1].isspace():
                return line[:index].rstrip()
    return line.rstrip()


def _parse_configuration_mapping(text: str) -> dict[str, object]:
    """Parse the intentionally small mapping-only YAML surface used here."""

    result: dict[str, object] = {}
    current_section: dict[str, object] | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        if "\t" in line:
            raise BaselineStrategyConfigurationError(
                "INVALID_STRATEGY_CONFIG",
                f"line {line_number} must not use tab indentation",
            )
        indentation = len(line) - len(line.lstrip(" "))
        if indentation not in (0, 2):
            raise BaselineStrategyConfigurationError(
                "INVALID_STRATEGY_CONFIG",
                f"line {line_number} must use zero or two spaces of indentation",
            )
        content = line[indentation:]
        if ":" not in content:
            raise BaselineStrategyConfigurationError(
                "INVALID_STRATEGY_CONFIG",
                f"line {line_number} must be a mapping entry",
            )
        key, token = content.split(":", 1)
        key = key.strip()
        token = token.strip()
        if not key:
            raise BaselineStrategyConfigurationError(
                "INVALID_STRATEGY_CONFIG",
                f"line {line_number} has an invalid key",
            )
        if indentation == 0:
            if key in result:
                raise BaselineStrategyConfigurationError(
                    "INVALID_STRATEGY_CONFIG",
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
            raise BaselineStrategyConfigurationError(
                "INVALID_STRATEGY_CONFIG",
                f"line {line_number} has no parent mapping",
            )
        if key in current_section:
            raise BaselineStrategyConfigurationError(
                "INVALID_STRATEGY_CONFIG",
                f"line {line_number} duplicates key {key!r}",
            )
        current_section[key] = _parse_scalar(token, line_number=line_number)
    return result


def _require_mapping(
    value: object, *, name: str, expected_keys: frozenset[str]
) -> dict[str, object]:
    if type(value) is not dict:
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIG", f"{name} must be a mapping"
        )
    actual_keys = frozenset(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIG",
            f"{name} keys are invalid; missing={missing}, extra={extra}",
        )
    return value


def parse_declared_baseline_strategy_configuration(
    text: str,
    *,
    target_risk_fraction: object,
    maximum_holding_sessions: object,
    earnings_maximum_holding_sessions: object,
) -> BaselineStrategyConfiguration:
    """Build one configuration from declaration text without accepting drift.

    The declaration is validated structurally and then rejected outright unless
    it equals the authoritative implementation. It is never used to override a
    value, so a hand-edited declaration cannot change any trading decision.

    The authoritative constants owned by packages that import ``strategy`` are
    required keyword arguments rather than an optional follow-up check, so it
    is impossible to obtain a configuration from a declaration without proving
    full parity against every frozen owner.
    """

    if type(text) is not str:
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIG", "configuration text must be a string"
        )
    raw = _require_mapping(
        _parse_configuration_mapping(text),
        name="strategy",
        expected_keys=_EXPECTED_TOP_LEVEL_KEYS,
    )
    sections = {
        name: _require_mapping(
            raw[name], name=name, expected_keys=expected_keys
        )
        for name, expected_keys in _EXPECTED_SECTION_KEYS.items()
    }
    try:
        configuration = BaselineStrategyConfiguration(
            schema_version=raw["schema_version"],
            configuration_id=raw["configuration_id"],
            tuned_parameter_count=raw["tuned_parameter_count"],
            decision_inputs=BaselineDecisionInputs(
                **sections["decision_inputs"]
            ),
            entry_conditions=BaselineEntryConditions(
                **sections["entry_conditions"]
            ),
            chronology=BaselineTradeChronology(**sections["chronology"]),
            risk=BaselineRiskGeometry(**sections["risk"]),
        )
    except BaselineStrategyConfigurationError:
        raise
    except (TypeError, ValueError) as error:
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIG",
            f"the declared strategy configuration is not canonical: {error}",
        ) from error
    return verify_baseline_strategy_configuration_parity(
        configuration,
        target_risk_fraction=target_risk_fraction,
        maximum_holding_sessions=maximum_holding_sessions,
        earnings_maximum_holding_sessions=earnings_maximum_holding_sessions,
    )


def load_declared_baseline_strategy_configuration(
    path: str | Path,
    *,
    target_risk_fraction: object,
    maximum_holding_sessions: object,
    earnings_maximum_holding_sessions: object,
) -> BaselineStrategyConfiguration:
    """Load and validate one explicitly supplied local declaration file."""

    if isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIG_PATH",
            "path must be an explicit local path",
        )
    if isinstance(path, str) and (not path or path != path.strip()):
        raise BaselineStrategyConfigurationError(
            "INVALID_STRATEGY_CONFIG_PATH",
            "path must be a non-blank canonical path",
        )
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BaselineStrategyConfigurationError(
            "STRATEGY_CONFIG_READ_FAILED",
            "the local strategy configuration file could not be read",
        ) from error
    return parse_declared_baseline_strategy_configuration(
        text,
        target_risk_fraction=target_risk_fraction,
        maximum_holding_sessions=maximum_holding_sessions,
        earnings_maximum_holding_sessions=earnings_maximum_holding_sessions,
    )


__all__ = [
    "BASELINE_STRATEGY_CONFIGURATION_ARTIFACT_TYPE",
    "BASELINE_STRATEGY_CONFIGURATION_HASH_DOMAIN",
    "BASELINE_STRATEGY_CONFIGURATION_ID",
    "BASELINE_STRATEGY_CONFIGURATION_SCHEMA_VERSION",
    "BaselineDecisionInputs",
    "BaselineEntryConditions",
    "BaselineRiskGeometry",
    "BaselineStrategyConfiguration",
    "BaselineStrategyConfigurationError",
    "BaselineStrategyConfigurationRef",
    "BaselineTradeChronology",
    "build_baseline_strategy_configuration",
    "build_baseline_strategy_configuration_ref",
    "compute_baseline_strategy_configuration_fingerprint",
    "load_declared_baseline_strategy_configuration",
    "parse_declared_baseline_strategy_configuration",
    "verify_baseline_strategy_configuration",
    "verify_baseline_strategy_configuration_parity",
]
