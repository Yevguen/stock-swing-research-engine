"""Static ownership boundaries for the pure Phase 15C.1 package."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "stock_swing_d1" / "execution" / "costs"


def _production_sources() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.glob("*.py")}


def test_package_imports_no_execution_owner_backtester_or_portfolio_domain() -> None:
    forbidden_prefixes = (
        "stock_swing_d1.execution.entry",
        "stock_swing_d1.execution.protective_exit",
        "stock_swing_d1.execution.open_position_exit",
        "stock_swing_d1.backtester",
        "stock_swing_d1.portfolio",
    )
    for name, source in _production_sources().items():
        tree = ast.parse(source, filename=name)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            module.startswith(forbidden_prefixes) for module in imported
        ), (name, imported)


def test_package_contains_no_forbidden_strategy_or_state_ownership() -> None:
    combined = "\n".join(_production_sources().values()).lower()
    forbidden_phrases = (
        "sma20",
        "sma50",
        "rsi14",
        "atr14",
        "rankingcandidate",
        "candidate allocation",
        "position sizing",
        "earnings reconstruction",
        "stop trigger",
        "target trigger",
        "settled_cash",
        "position removal",
        "portfoliotransition",
        "entry signal",
    )
    assert all(phrase not in combined for phrase in forbidden_phrases)


def test_settlement_uses_only_injected_session_advancement() -> None:
    source = _production_sources()["settlement.py"]
    assert "next_settlement_session" in source
    assert "timedelta" not in source
    assert "toordinal" not in source


def test_cost_assumptions_are_owned_in_the_new_package() -> None:
    model_source = _production_sources()["models.py"]
    service_source = _production_sources()["service.py"]
    pricing_source = _production_sources()["pricing.py"]
    assert 'Decimal("0.005")' in model_source
    assert 'Decimal("1.00")' in model_source
    assert 'Decimal("0.01")' in model_source
    assert 'Decimal("1")' in model_source
    assert 'Decimal("5")' in model_source
    assert "minimum_adjusted_commission" in service_source
    assert "commission_cap" in service_source
    assert "administrative_exit_bps" in pricing_source


def test_commission_and_spread_formulas_are_not_owned_by_prior_phases() -> None:
    production_root = PACKAGE_ROOT.parents[1]
    protected_paths = (
        production_root / "execution" / "entry",
        production_root / "execution" / "protective_exit",
        production_root / "execution" / "open_position_exit",
        production_root / "portfolio",
    )
    protected_source = "\n".join(
        path.read_text(encoding="utf-8")
        for root in protected_paths
        for path in root.glob("*.py")
    )
    phase15c_formula_tokens = (
        "per_share_with_minimum_and_notional_cap",
        "minimum_adjusted_commission",
        "commission_cap_fraction",
        "fixed_cash_equivalent_bps_per_side",
        "spread_bps_per_side",
    )
    assert all(token not in protected_source for token in phase15c_formula_tokens)


def test_package_contains_no_network_broker_or_dynamic_pricing_dependencies() -> None:
    combined = "\n".join(_production_sources().values()).lower()
    forbidden_imports = ("requests", "urllib", "httpx", "socket", "ibkr")
    assert all(term not in combined for term in forbidden_imports)
