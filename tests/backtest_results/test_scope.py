from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "src" / "stock_swing_d1" / "backtest_results"
DOWNSTREAM_RESEARCH_METRICS = (
    ROOT / "src" / "stock_swing_d1" / "research_metrics"
)
# Phase 16C sits below Phase 16B in the frozen Phase 13 -> 15A -> 15D -> 16B ->
# 16C dependency chain, so like Phase 16B it consumes the audited Phase 15D
# result rather than being an upstream producer of it. The directional
# invariant this guard enforces is unchanged: no package *above* Phase 15D may
# depend on it.
DOWNSTREAM_BASELINE_EXPERIMENT = (
    ROOT / "src" / "stock_swing_d1" / "baseline_experiment"
)
DOWNSTREAM_PACKAGES = (
    DOWNSTREAM_RESEARCH_METRICS,
    DOWNSTREAM_BASELINE_EXPERIMENT,
)
UPSTREAM = ROOT / "src" / "stock_swing_d1"


def _production_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PRODUCTION.glob("*.py"))
    )


def test_phase15d2_has_exactly_the_expected_production_files():
    assert {path.name for path in PRODUCTION.glob("*.py")} == {
        "__init__.py",
        "canonical.py",
        "derivation.py",
        "errors.py",
        "hashing.py",
        "models.py",
        "persistence.py",
        "persistence_schema.py",
        "reconciliation.py",
        "service.py",
        "source_validation.py",
        "valuation.py",
    }


_FORBIDDEN_UPSTREAM_OWNER_CALLS: tuple[str, ...] = (
    "rank_candidates(",
    "allocate_ranked_candidates(",
    "size_pending_entry(",
    "execute_pending_entry(",
    "evaluate_signal(",
    "OpenPositionExitEvaluator(",
    "PortfolioTransitionEngine.transition(",
    "BacktestBuyExecutionEventAdapter(",
    "BacktestSellExecutionEventAdapter(",
    "AdministrativeExitPricingService(",
    "HistoricalUsEquitySettlementResolver(",
)

# Exact, non-generic symbol names for the two forbidden-dependency categories
# the Phase 15D.5 audit found were only checked in two file-scoped Phase 15D.3
# tests (derivation.py only; service.py+reconciliation.py only), never across
# the complete 12-file production set. These are precise provider/data-pipeline
# and settlement-calendar identifiers, not generic English words, so a benign
# sentence like "Phase 15D never queries a provider" cannot trip this check.
_FORBIDDEN_PROVIDER_AND_SETTLEMENT_CALENDAR_TOKENS: tuple[str, ...] = (
    "norgatedata",
    "NorgateD1Adapter",
    "NorgateAdjustedD1Adapter",
    "NorgateCapitalEventAdapter",
    "d1_pipeline",
    "corporate_action_pipeline",
    "dividend_event_pipeline",
    "HistoricalUsEquitySettlementResolver",
    "SettlementSessionCalendar",
    "next_settlement_session",
    "SettlementPolicyModel",
)


def test_phase15d_does_not_call_upstream_owner_operations():
    source = _production_text()
    for token in _FORBIDDEN_UPSTREAM_OWNER_CALLS:
        assert token not in source


def test_phase15d_full_tree_has_no_provider_or_settlement_calendar_dependency():
    # Closes the Phase 15D.5 audit gap: proves none of the 12 production files
    # (not just derivation.py, or just service.py+reconciliation.py) reference
    # a provider/data-pipeline lookup or a settlement-calendar/resolver symbol.
    source = _production_text()
    for token in (
        _FORBIDDEN_UPSTREAM_OWNER_CALLS
        + _FORBIDDEN_PROVIDER_AND_SETTLEMENT_CALENDAR_TOKENS
    ):
        assert token not in source


def test_phase15d_has_no_execution_economics_or_performance_metrics():
    source = _production_text().lower()
    forbidden = (
        "entry_slippage_bps",
        "exit_slippage_bps",
        "0.9995",
        "commission_per_share",
        "minimum_commission",
        "spread_bps",
        "sharpe",
        "sortino",
        "cagr",
        "calmar",
        "max_drawdown",
        "annualized_volatility",
        "factor_attribution",
    )
    for token in forbidden:
        assert token not in source


def test_semantic_models_have_no_clock_random_uuid_or_paths():
    source = _production_text().lower()
    for token in (
        "datetime.now",
        "date.today",
        "time.time",
        "uuid",
        "random",
        "generated_at",
        "output_path",
        "filesystem_path",
    ):
        assert token not in source


def test_no_upstream_package_depends_on_backtest_results():
    offenders = []
    for path in UPSTREAM.rglob("*.py"):
        if PRODUCTION in path.parents or any(
            downstream in path.parents for downstream in DOWNSTREAM_PACKAGES
        ):
            continue
        if "backtest_results" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT))
    assert offenders == []


def test_no_decimal_from_binary_float_pattern():
    source = _production_text()
    assert "Decimal(float" not in source


def test_phase15d2_does_not_infer_sessions_from_datetime_date_calls():
    phase15d2_source = "\n".join(
        (PRODUCTION / filename).read_text(encoding="utf-8")
        for filename in ("source_validation.py", "derivation.py")
    )
    assert ".date()" not in phase15d2_source
