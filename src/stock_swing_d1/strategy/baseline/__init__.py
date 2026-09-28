"""Phase 8 Baseline Strategy Methodology v0.1."""

from stock_swing_d1.strategy.baseline.configuration import (
    BASELINE_STRATEGY_CONFIGURATION_ARTIFACT_TYPE,
    BASELINE_STRATEGY_CONFIGURATION_HASH_DOMAIN,
    BASELINE_STRATEGY_CONFIGURATION_ID,
    BASELINE_STRATEGY_CONFIGURATION_SCHEMA_VERSION,
    BaselineStrategyConfiguration,
    BaselineStrategyConfigurationError,
    BaselineStrategyConfigurationRef,
    build_baseline_strategy_configuration,
    build_baseline_strategy_configuration_ref,
    compute_baseline_strategy_configuration_fingerprint,
    load_declared_baseline_strategy_configuration,
    parse_declared_baseline_strategy_configuration,
    verify_baseline_strategy_configuration,
    verify_baseline_strategy_configuration_parity,
)
from stock_swing_d1.strategy.baseline.models import (
    BaselineSignalCalendar,
    BaselineSignalAction,
    BaselineSignalDecision,
    BaselineStrategyValidationError,
)
from stock_swing_d1.strategy.baseline.service import BaselineSignalEvaluator

__all__ = [
    "BASELINE_STRATEGY_CONFIGURATION_ARTIFACT_TYPE",
    "BASELINE_STRATEGY_CONFIGURATION_HASH_DOMAIN",
    "BASELINE_STRATEGY_CONFIGURATION_ID",
    "BASELINE_STRATEGY_CONFIGURATION_SCHEMA_VERSION",
    "BaselineSignalCalendar",
    "BaselineSignalAction",
    "BaselineSignalDecision",
    "BaselineSignalEvaluator",
    "BaselineStrategyConfiguration",
    "BaselineStrategyConfigurationError",
    "BaselineStrategyConfigurationRef",
    "BaselineStrategyValidationError",
    "build_baseline_strategy_configuration",
    "build_baseline_strategy_configuration_ref",
    "compute_baseline_strategy_configuration_fingerprint",
    "load_declared_baseline_strategy_configuration",
    "parse_declared_baseline_strategy_configuration",
    "verify_baseline_strategy_configuration",
    "verify_baseline_strategy_configuration_parity",
]
