"""Task 5B: canonical baseline strategy configuration artifact and identity.

These tests prove three separate things:

* the artifact's semantic inventory is pinned and matches the authoritative
  frozen owners, including the owners ``strategy`` cannot import;
* its fingerprint depends on every material semantic and on nothing else;
* ``config/strategy.yaml`` is a checked declaration that can never override
  code.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from stock_swing_d1.earnings.integration.service import _MAX_HOLDING_SESSIONS
from stock_swing_d1.execution.open_position_exit.models import (
    MAX_HOLDING_SESSIONS,
)
from stock_swing_d1.indicators import calculate_indicators
from stock_swing_d1.models import CorporateActionAdjustedStockBar
from stock_swing_d1.risk.position_sizing.service import TARGET_RISK_FRACTION
from stock_swing_d1.strategy.baseline import (
    BASELINE_STRATEGY_CONFIGURATION_ARTIFACT_TYPE,
    BASELINE_STRATEGY_CONFIGURATION_HASH_DOMAIN,
    BASELINE_STRATEGY_CONFIGURATION_ID,
    BASELINE_STRATEGY_CONFIGURATION_SCHEMA_VERSION,
    BaselineSignalAction,
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
from stock_swing_d1.strategy.baseline import configuration as configuration_module
from stock_swing_d1.strategy.baseline.configuration_hashing import (
    canonical_configuration_json_bytes,
)
from stock_swing_d1.strategy.baseline.service import (
    _ATR_MINIMUM_FRACTION,
    _REQUIRED_PRICE_BASIS,
    _RSI_MINIMUM,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STRATEGY_CONFIG_PATH = REPOSITORY_ROOT / "config" / "strategy.yaml"

# The authoritative constants owned by packages that import ``strategy``.
# Injecting the real objects here is what makes these tests a drift guard.
AUTHORITATIVE_INJECTIONS = {
    "target_risk_fraction": TARGET_RISK_FRACTION,
    "maximum_holding_sessions": MAX_HOLDING_SESSIONS,
    "earnings_maximum_holding_sessions": _MAX_HOLDING_SESSIONS,
}


@pytest.fixture
def configuration() -> BaselineStrategyConfiguration:
    return build_baseline_strategy_configuration()


@pytest.fixture
def declaration_text() -> str:
    return STRATEGY_CONFIG_PATH.read_text(encoding="utf-8")


def _mutate(
    configuration: BaselineStrategyConfiguration,
    section: str,
    field_name: str,
    value: object,
) -> BaselineStrategyConfiguration:
    """Rebuild a configuration with exactly one semantic field changed."""

    payload = configuration.model_dump(mode="python")
    payload[section][field_name] = value
    return BaselineStrategyConfiguration(**payload)


# --------------------------------------------------------------------------
# A. Semantic inventory and authoritative consistency
# --------------------------------------------------------------------------


def test_configuration_inventory_is_pinned(configuration) -> None:
    """Pin the complete field inventory so a silent addition or removal fails."""

    assert set(BaselineStrategyConfiguration.model_fields) == {
        "schema_version",
        "configuration_id",
        "tuned_parameter_count",
        "decision_inputs",
        "entry_conditions",
        "chronology",
        "risk",
    }
    assert set(type(configuration.decision_inputs).model_fields) == {
        "signal_price_basis",
        "sma_fast_period",
        "sma_slow_period",
        "rsi_period",
        "atr_period",
    }
    assert set(type(configuration.entry_conditions).model_fields) == {
        "close_above_sma_slow_required",
        "sma_fast_above_sma_slow_required",
        "rsi_minimum",
        "rsi_comparison",
        "atr_minimum_fraction",
        "atr_comparison",
        "universe_eligibility_required",
        "earnings_entry_permission_required",
    }
    assert set(type(configuration.chronology).model_fields) == {
        "trade_direction",
        "entry_execution_offset_sessions",
        "maximum_holding_sessions",
    }
    assert set(type(configuration.risk).model_fields) == {
        "target_risk_fraction",
        "stop_risk_atr_multiple",
        "take_profit_r_multiple",
    }


def test_builder_reproduces_the_authoritative_baseline_values(
    configuration,
) -> None:
    assert configuration.schema_version == (
        BASELINE_STRATEGY_CONFIGURATION_SCHEMA_VERSION
    )
    assert configuration.configuration_id == BASELINE_STRATEGY_CONFIGURATION_ID
    assert configuration.tuned_parameter_count == 0

    inputs = configuration.decision_inputs
    assert inputs.signal_price_basis == _REQUIRED_PRICE_BASIS
    assert (inputs.sma_fast_period, inputs.sma_slow_period) == (20, 50)
    assert (inputs.rsi_period, inputs.atr_period) == (14, 14)

    entry = configuration.entry_conditions
    assert entry.rsi_minimum == _RSI_MINIMUM
    assert entry.atr_minimum_fraction == _ATR_MINIMUM_FRACTION
    assert entry.rsi_comparison == "strictly_greater_than"
    assert entry.atr_comparison == "greater_than_or_equal"

    chronology = configuration.chronology
    assert chronology.trade_direction == "LONG_ONLY"
    assert chronology.entry_execution_offset_sessions == 1
    assert chronology.maximum_holding_sessions == MAX_HOLDING_SESSIONS

    risk = configuration.risk
    assert risk.target_risk_fraction == TARGET_RISK_FRACTION
    assert risk.stop_risk_atr_multiple == 2.0
    assert risk.take_profit_r_multiple == 2.0


def test_built_configuration_passes_full_authoritative_parity(
    configuration,
) -> None:
    """The canonical builder must satisfy every frozen owner, not just Phase 8."""

    assert (
        verify_baseline_strategy_configuration_parity(
            configuration, **AUTHORITATIVE_INJECTIONS
        )
        is configuration
    )


@pytest.mark.parametrize(
    ("section", "field_name", "value", "code"),
    [
        ("entry_conditions", "rsi_minimum", 60.0, "RSI_MINIMUM_MISMATCH"),
        (
            "entry_conditions",
            "atr_minimum_fraction",
            0.02,
            "ATR_MINIMUM_FRACTION_MISMATCH",
        ),
        (
            "decision_inputs",
            "sma_fast_period",
            30,
            "INDICATOR_WINDOW_MISMATCH",
        ),
        (
            "decision_inputs",
            "sma_slow_period",
            100,
            "INDICATOR_WINDOW_MISMATCH",
        ),
        ("decision_inputs", "rsi_period", 21, "INDICATOR_WINDOW_MISMATCH"),
        ("decision_inputs", "atr_period", 21, "INDICATOR_WINDOW_MISMATCH"),
        (
            "chronology",
            "maximum_holding_sessions",
            12,
            "MAXIMUM_HOLDING_SESSIONS_MISMATCH",
        ),
        (
            "risk",
            "target_risk_fraction",
            0.01,
            "TARGET_RISK_FRACTION_MISMATCH",
        ),
        ("risk", "stop_risk_atr_multiple", 3.0, "RISK_GEOMETRY_MISMATCH"),
        ("risk", "take_profit_r_multiple", 3.0, "RISK_GEOMETRY_MISMATCH"),
    ],
)
def test_one_changed_semantic_fails_closed(
    configuration, section, field_name, value, code
) -> None:
    mutated = _mutate(configuration, section, field_name, value)
    with pytest.raises(BaselineStrategyConfigurationError) as error:
        verify_baseline_strategy_configuration_parity(
            mutated, **AUTHORITATIVE_INJECTIONS
        )
    assert error.value.code == code


def test_disagreeing_holding_ceilings_fail_closed(configuration) -> None:
    """Phase 15B and the earnings overlay must agree on the same ceiling."""

    with pytest.raises(BaselineStrategyConfigurationError) as error:
        verify_baseline_strategy_configuration_parity(
            configuration,
            target_risk_fraction=TARGET_RISK_FRACTION,
            maximum_holding_sessions=MAX_HOLDING_SESSIONS,
            earnings_maximum_holding_sessions=MAX_HOLDING_SESSIONS + 1,
        )
    assert error.value.code == "MAXIMUM_HOLDING_SESSIONS_MISMATCH"


@pytest.mark.parametrize("value", [1, 2, -1, False, True, 0.0, "0", None])
def test_tuned_parameter_count_other_than_zero_is_unconstructable(
    configuration, value
) -> None:
    payload = configuration.model_dump(mode="python")
    payload["tuned_parameter_count"] = value
    with pytest.raises(ValueError):
        BaselineStrategyConfiguration(**payload)


def test_required_entry_conditions_cannot_be_declared_optional(
    configuration,
) -> None:
    for field_name in (
        "close_above_sma_slow_required",
        "sma_fast_above_sma_slow_required",
        "universe_eligibility_required",
        "earnings_entry_permission_required",
    ):
        with pytest.raises(ValueError):
            _mutate(configuration, "entry_conditions", field_name, False)


def test_missing_and_unexpected_fields_are_rejected(configuration) -> None:
    payload = configuration.model_dump(mode="python")
    del payload["risk"]
    with pytest.raises(ValueError):
        BaselineStrategyConfiguration(**payload)

    payload = configuration.model_dump(mode="python")
    payload["stop_loss_atr_multiple"] = 3.0
    with pytest.raises(ValueError):
        BaselineStrategyConfiguration(**payload)


def test_numeric_types_are_not_silently_coerced(configuration) -> None:
    with pytest.raises(ValueError):
        _mutate(configuration, "entry_conditions", "rsi_minimum", 50)
    with pytest.raises(ValueError):
        _mutate(configuration, "decision_inputs", "sma_fast_period", 20.0)
    with pytest.raises(ValueError):
        _mutate(configuration, "risk", "target_risk_fraction", "0.005")


def test_verification_rejects_a_foreign_object() -> None:
    with pytest.raises(BaselineStrategyConfigurationError):
        verify_baseline_strategy_configuration(object())


# --------------------------------------------------------------------------
# A2. Behavioural parity with the frozen owners strategy cannot import
# --------------------------------------------------------------------------


def _adjusted_series(count: int) -> tuple[CorporateActionAdjustedStockBar, ...]:
    start = date(2026, 1, 5)
    return tuple(
        CorporateActionAdjustedStockBar(
            security_id="NORGATE:1001",
            symbol="CFG",
            trading_date=start + timedelta(days=index),
            timeframe="D1",
            session_type="regular",
            currency="USD",
            price_basis="capital_special_adjusted",
            open=100.0 + index,
            high=101.0 + index,
            low=99.0 + index,
            close=100.0 + index,
            volume=1_000_000.0,
        )
        for index in range(count)
    )


def test_declared_indicator_windows_match_phase7_warm_up_behaviour(
    configuration,
) -> None:
    """Drive Phase 7 and prove each declared window is its real warm-up length.

    An SMA or ATR of period N consumes N bars, so it first becomes defined at
    index N-1. RSI additionally needs a prior close to form its first change,
    so it first becomes defined one bar later, at index N. Either way the
    transition index is a function of the declared window alone, which pins the
    declaration to observed Phase 7 behaviour rather than to a private constant.
    """

    inputs = configuration.decision_inputs
    rows = calculate_indicators(_adjusted_series(80))

    for field_name, expected_first_defined_index in (
        ("sma_20", inputs.sma_fast_period - 1),
        ("sma_50", inputs.sma_slow_period - 1),
        ("atr_14", inputs.atr_period - 1),
        ("rsi_14", inputs.rsi_period),
    ):
        values = [getattr(row, field_name) for row in rows]
        defined_from = next(
            index for index, value in enumerate(values) if value is not None
        )
        assert defined_from == expected_first_defined_index, field_name


def test_declared_thresholds_match_phase8_boundary_behaviour(
    configuration, evaluator_factory, make_bar, make_indicators
) -> None:
    """Prove the declared comparison operators are the ones Phase 8 applies."""

    entry = configuration.entry_conditions
    evaluator, _ = evaluator_factory()

    # rsi_comparison == "strictly_greater_than": exactly at the minimum fails.
    at_threshold = evaluator.evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(rsi_14=entry.rsi_minimum),
        universe_eligible=True,
    )
    assert at_threshold.rsi_above_50 is False
    assert at_threshold.action is BaselineSignalAction.NO_SIGNAL

    # atr_comparison == "greater_than_or_equal": exactly at the minimum passes.
    close = 100.0
    exact = evaluator.evaluate_signal(
        bar=make_bar(close=close),
        indicators=make_indicators(
            sma_20=close - 1.0,
            sma_50=close - 2.0,
            rsi_14=55.0,
            atr_14=close * entry.atr_minimum_fraction,
        ),
        universe_eligible=True,
    )
    assert exact.atr_fraction == entry.atr_minimum_fraction
    assert exact.atr_above_minimum is True
    assert exact.action is BaselineSignalAction.VALID_LONG_SIGNAL

    # One tick below the declared minimum must fail the same comparison.
    below = evaluator.evaluate_signal(
        bar=make_bar(close=close),
        indicators=make_indicators(
            sma_20=close - 1.0,
            sma_50=close - 2.0,
            rsi_14=55.0,
            atr_14=close * entry.atr_minimum_fraction / 2.0,
        ),
        universe_eligible=True,
    )
    assert below.atr_above_minimum is False
    assert below.action is BaselineSignalAction.NO_SIGNAL


def test_declared_entry_offset_matches_phase8_next_session(
    configuration, evaluator_factory, make_bar, make_indicators, baseline_calendar
) -> None:
    decision = evaluator_factory()[0].evaluate_signal(
        bar=make_bar(),
        indicators=make_indicators(),
        universe_eligible=True,
    )
    sessions = baseline_calendar.sessions
    offset = configuration.chronology.entry_execution_offset_sessions
    assert decision.planned_entry_session == sessions[
        sessions.index(decision.signal_session) + offset
    ]


# --------------------------------------------------------------------------
# B. Fingerprint determinism
# --------------------------------------------------------------------------


def test_fingerprint_is_deterministic_and_canonical_sha256(
    configuration,
) -> None:
    first = compute_baseline_strategy_configuration_fingerprint(configuration)
    second = compute_baseline_strategy_configuration_fingerprint(
        build_baseline_strategy_configuration()
    )
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_fingerprint_is_domain_separated(configuration) -> None:
    """The bare payload must not collide with the domain-separated digest."""

    import hashlib

    undomained = hashlib.sha256(
        canonical_configuration_json_bytes(configuration)
    ).hexdigest()
    assert (
        compute_baseline_strategy_configuration_fingerprint(configuration)
        != undomained
    )
    assert BASELINE_STRATEGY_CONFIGURATION_HASH_DOMAIN == (
        "baseline_strategy_configuration.v0.1"
    )


@pytest.mark.parametrize(
    ("section", "field_name", "value"),
    [
        ("decision_inputs", "sma_fast_period", 21),
        ("decision_inputs", "sma_slow_period", 51),
        ("decision_inputs", "rsi_period", 15),
        ("decision_inputs", "atr_period", 15),
        ("entry_conditions", "rsi_minimum", 50.5),
        ("entry_conditions", "atr_minimum_fraction", 0.011),
        ("chronology", "maximum_holding_sessions", 11),
        ("risk", "target_risk_fraction", 0.006),
        ("risk", "stop_risk_atr_multiple", 2.5),
        ("risk", "take_profit_r_multiple", 3.0),
    ],
)
def test_every_material_semantic_changes_the_fingerprint(
    configuration, section, field_name, value
) -> None:
    mutated = _mutate(configuration, section, field_name, value)
    assert compute_baseline_strategy_configuration_fingerprint(
        mutated
    ) != compute_baseline_strategy_configuration_fingerprint(configuration)


def test_float_semantics_are_bound_losslessly(configuration) -> None:
    """Two floats that share a short decimal rendering must still differ.

    ``0.005`` and the next representable double both print as roughly the same
    decimal, so a decimal-rendering fingerprint could confuse them. Hex
    encoding cannot.
    """

    import math

    nudged = math.nextafter(configuration.risk.target_risk_fraction, 1.0)
    mutated = _mutate(configuration, "risk", "target_risk_fraction", nudged)
    assert mutated.risk.target_risk_fraction != (
        configuration.risk.target_risk_fraction
    )
    assert compute_baseline_strategy_configuration_fingerprint(
        mutated
    ) != compute_baseline_strategy_configuration_fingerprint(configuration)


def test_object_identity_and_key_order_do_not_change_the_fingerprint(
    configuration,
) -> None:
    rebuilt = BaselineStrategyConfiguration(
        **json.loads(
            json.dumps(configuration.model_dump(mode="python"))
        )
    )
    assert rebuilt is not configuration
    assert compute_baseline_strategy_configuration_fingerprint(rebuilt) == (
        compute_baseline_strategy_configuration_fingerprint(configuration)
    )

    payload = configuration.model_dump(mode="python")
    reordered = {key: payload[key] for key in reversed(list(payload))}
    reordered["risk"] = {
        key: payload["risk"][key] for key in reversed(list(payload["risk"]))
    }
    assert compute_baseline_strategy_configuration_fingerprint(
        BaselineStrategyConfiguration(**reordered)
    ) == compute_baseline_strategy_configuration_fingerprint(configuration)


def test_artifact_reference_binds_the_semantic_fingerprint(
    configuration,
) -> None:
    reference = build_baseline_strategy_configuration_ref(configuration)
    assert type(reference) is BaselineStrategyConfigurationRef
    assert reference.artifact_type == (
        BASELINE_STRATEGY_CONFIGURATION_ARTIFACT_TYPE
    )
    assert reference.schema_version == configuration.schema_version
    assert reference.content_sha256 == (
        compute_baseline_strategy_configuration_fingerprint(configuration)
    )
    with pytest.raises(ValueError):
        BaselineStrategyConfigurationRef(
            artifact_type=BASELINE_STRATEGY_CONFIGURATION_ARTIFACT_TYPE,
            schema_version=configuration.schema_version,
            content_sha256="NOT-A-DIGEST",
        )


def test_artifact_reference_rejects_a_noncanonical_configuration(
    configuration,
) -> None:
    mutated = _mutate(
        configuration, "risk", "stop_risk_atr_multiple", 3.0
    )
    with pytest.raises(BaselineStrategyConfigurationError) as error:
        build_baseline_strategy_configuration_ref(mutated)
    assert error.value.code == "RISK_GEOMETRY_MISMATCH"


# --------------------------------------------------------------------------
# C. The config/strategy.yaml checked-declaration boundary
# --------------------------------------------------------------------------


def test_repository_declaration_equals_the_authoritative_artifact(
    configuration,
) -> None:
    declared = load_declared_baseline_strategy_configuration(
        STRATEGY_CONFIG_PATH, **AUTHORITATIVE_INJECTIONS
    )
    assert declared == configuration
    assert compute_baseline_strategy_configuration_fingerprint(declared) == (
        compute_baseline_strategy_configuration_fingerprint(configuration)
    )


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        ("rsi_minimum: 50.0", "rsi_minimum: 60.0", "RSI_MINIMUM_MISMATCH"),
        (
            "atr_minimum_fraction: 0.01",
            "atr_minimum_fraction: 0.02",
            "ATR_MINIMUM_FRACTION_MISMATCH",
        ),
        (
            "sma_fast_period: 20",
            "sma_fast_period: 30",
            "INDICATOR_WINDOW_MISMATCH",
        ),
        (
            "maximum_holding_sessions: 10",
            "maximum_holding_sessions: 12",
            "MAXIMUM_HOLDING_SESSIONS_MISMATCH",
        ),
        (
            "target_risk_fraction: 0.005",
            "target_risk_fraction: 0.01",
            "TARGET_RISK_FRACTION_MISMATCH",
        ),
        (
            "stop_risk_atr_multiple: 2.0",
            "stop_risk_atr_multiple: 3.0",
            "RISK_GEOMETRY_MISMATCH",
        ),
        (
            "take_profit_r_multiple: 2.0",
            "take_profit_r_multiple: 3.0",
            "RISK_GEOMETRY_MISMATCH",
        ),
        (
            'signal_price_basis: "capital_special_adjusted"',
            'signal_price_basis: "unadjusted"',
            "INVALID_STRATEGY_CONFIG",
        ),
    ],
)
def test_declaration_disagreeing_with_code_is_rejected(
    declaration_text, old, new, code
) -> None:
    assert old in declaration_text
    with pytest.raises(BaselineStrategyConfigurationError) as error:
        parse_declared_baseline_strategy_configuration(
            declaration_text.replace(old, new), **AUTHORITATIVE_INJECTIONS
        )
    assert error.value.code == code


@pytest.mark.parametrize(
    "removed",
    ["  rsi_period: 14\n", "  target_risk_fraction: 0.005\n", "risk:\n"],
)
def test_declaration_missing_a_required_field_is_rejected(
    declaration_text, removed
) -> None:
    assert removed in declaration_text
    with pytest.raises(BaselineStrategyConfigurationError):
        parse_declared_baseline_strategy_configuration(
            declaration_text.replace(removed, ""), **AUTHORITATIVE_INJECTIONS
        )


def test_declaration_with_an_unknown_field_is_rejected(
    declaration_text,
) -> None:
    with pytest.raises(BaselineStrategyConfigurationError) as error:
        parse_declared_baseline_strategy_configuration(
            declaration_text + "\nunexpected_section: 1\n",
            **AUTHORITATIVE_INJECTIONS,
        )
    assert error.value.code == "INVALID_STRATEGY_CONFIG"

    with pytest.raises(BaselineStrategyConfigurationError):
        parse_declared_baseline_strategy_configuration(
            declaration_text.replace(
                "  rsi_period: 14", "  rsi_period: 14\n  extra_window: 9"
            ),
            **AUTHORITATIVE_INJECTIONS,
        )


def test_declaration_comments_and_blank_lines_do_not_change_identity(
    declaration_text, configuration
) -> None:
    stripped = "\n".join(
        line
        for line in declaration_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    reparsed = parse_declared_baseline_strategy_configuration(
        stripped, **AUTHORITATIVE_INJECTIONS
    )
    assert compute_baseline_strategy_configuration_fingerprint(reparsed) == (
        compute_baseline_strategy_configuration_fingerprint(configuration)
    )


def test_malformed_declarations_fail_closed() -> None:
    for text in ("\tschema_version: x", "not a mapping", "   deep: 1"):
        with pytest.raises(BaselineStrategyConfigurationError):
            parse_declared_baseline_strategy_configuration(
                text, **AUTHORITATIVE_INJECTIONS
            )


def test_declaration_cannot_be_loaded_without_authoritative_injection() -> None:
    """Parity injection is mandatory, so drift cannot be skipped by omission."""

    with pytest.raises(TypeError):
        load_declared_baseline_strategy_configuration(STRATEGY_CONFIG_PATH)


def test_unreadable_declaration_path_fails_closed(tmp_path) -> None:
    with pytest.raises(BaselineStrategyConfigurationError) as error:
        load_declared_baseline_strategy_configuration(
            tmp_path / "absent.yaml", **AUTHORITATIVE_INJECTIONS
        )
    assert error.value.code == "STRATEGY_CONFIG_READ_FAILED"


def test_yaml_is_not_read_on_the_signal_path() -> None:
    """No runtime decision owner may read, or depend on, the declaration.

    This is what makes ``config/strategy.yaml`` structurally incapable of
    changing a trading decision: every owner that actually decides is proven
    not to reference the file or the configuration module at all.
    """

    for module_path in (
        "strategy/baseline/service.py",
        "strategy/baseline/models.py",
        "indicators/calculator.py",
        "risk/position_sizing/service.py",
        "execution/protective_exit/service.py",
        "execution/open_position_exit/service.py",
        "execution/entry/service.py",
    ):
        source = (
            REPOSITORY_ROOT / "src" / "stock_swing_d1" / module_path
        ).read_text(encoding="utf-8")
        assert "strategy.yaml" not in source
        assert "baseline.configuration" not in source
        assert "BaselineStrategyConfiguration" not in source


# --------------------------------------------------------------------------
# F. Ownership and scope
# --------------------------------------------------------------------------


def test_configuration_module_owns_no_economics() -> None:
    """The artifact describes the strategy; it must never re-implement it."""

    source = Path(configuration_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "calculate_indicators(",
        "evaluate_signal(",
        "rank_candidates(",
        "allocate_ranked_candidates(",
        "size_pending_entry(",
        "execute_pending_entry(",
        "create_state(",
        "transition(",
        "calculate_research_metrics(",
        "execute_canonical_baseline_experiment(",
        "grid_search",
        "optimiz",
        "backtest_results",
        "norgatedata",
    ):
        assert forbidden not in source


def test_configuration_module_has_no_clock_random_or_path_identity() -> None:
    source = Path(configuration_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "datetime.now",
        "date.today",
        "time.time",
        "uuid",
        "random",
        "generated_at",
        "getenv",
        "environ",
    ):
        assert forbidden not in source


def test_strategy_package_still_declares_only_long_only_actions() -> None:
    assert set(BaselineSignalAction) == {
        BaselineSignalAction.VALID_LONG_SIGNAL,
        BaselineSignalAction.NO_SIGNAL,
    }
