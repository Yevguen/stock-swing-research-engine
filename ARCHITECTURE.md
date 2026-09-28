# Architecture

`stock_swing_d1` (the Python package in this repository) is research and backtesting software for long-only, unleveraged D1 (daily-candle) swing trading on liquid US large-cap stocks. It is not a live-trading system; it is a deterministic research and backtest pipeline.

## Framework rules

The framework rules are fixed research constraints, not optimisation parameters. They are declared in `config/research_rules.yaml`, `config/universe.yaml` and `config/costs.yaml`:

- Long-only, unleveraged, cash-only positions, with at most one open position per symbol.
- Decisions use only completed D1 candles. A signal from session T executes no earlier than session T+1, and this is enforced in code rather than left to convention.
- Holding through a scheduled earnings announcement is prohibited; positions are closed before it.
- The research universe is point-in-time Russell 1000 membership plus major-exchange status, ranked monthly by ADTV60 (60-session average daily dollar volume).
- The canonical `security_id` has the form `NORGATE:<AssetId>`, based on the identifier of the data source the universe design was built around; ticker symbols are display metadata only.
- Changes to these rules are deliberate and versioned.

## Development workflow

The codebase was built in sequential phases. Each phase follows the same contract-first process:

1. Research and design the contract.
2. For architecturally or economically significant changes, review and provisionally freeze the contract before implementation.
3. Implement against the frozen baseline.
4. Run focused, regression and full verification as appropriate.
5. Review completion separately before integrating the change.

Test files are often named after the phase they validate, for example `tests/backtester/test_phase15a_chronology.py`. A later phase consumes an earlier phase's output through a narrow, validated boundary rather than reaching into its internals: for example, `ranking/integration.py` adapts Phase 14 ranking output into Phase 12 portfolio allocation without re-deriving the rank order.

### Phase map

| Phase | Area | Main packages |
|---|---|---|
| 1–3 | Research framework and universe methodology | `config/`, `models/` |
| 4 | Historical D1 data pipeline | `data/`, `storage/`, `validation/` |
| 5 | Corporate actions and dividends | `data/`, `storage/`, `validation/` |
| 6 | Point-in-time earnings calendar and earnings-risk rules | `earnings/` |
| 7 | Technical indicators | `indicators/` |
| 8 | Baseline strategy | `strategy/baseline/` |
| 9 | Entry execution | `execution/entry/` |
| 10 | Gap-aware protective exits | `execution/protective_exit/` |
| 11 | Risk and position sizing | `risk/position_sizing/` |
| 12–13 | Portfolio allocation, state and accounting | `portfolio/` |
| 14 | Deterministic candidate ranking | `ranking/` |
| 15 | End-to-end historical backtest, execution costs, exits and audit views | `backtester/`, `execution/costs/`, `execution/open_position_exit/`, `backtest_results/` |
| 16 (experimental) | Research metrics and baseline-experiment infrastructure | `research_metrics/`, `baseline_experiment/` |

`valuation/` and `provenance/` hold small shared primitives (portfolio valuation and the generic `ArtifactRef` contract) used by the runtime and audit layers.

## Core conventions

**Determinism through frozen policy objects and SHA-256 fingerprints.** Packages such as `ranking`, `portfolio`, `execution/costs` and `backtest_results` define an immutable policy object that describes exact semantics (for example `PortfolioAllocationPolicy` in `portfolio/allocation_policy.py`), together with a domain-separated SHA-256 fingerprint over its canonical JSON. Downstream consumers store and compare these fingerprints, so a result can be shown to have been produced under one exact, unmodified policy.

**Fail-closed validation.** Services validate the identity and shape of their inputs strictly and raise typed errors instead of coercing or defaulting. See `portfolio/portfolio_errors.py` and `portfolio/portfolio_invariants.py`.

**Stateless, pure core services.** Most `*/service.py` modules are stateless or pure-functional: given inputs and a policy, they return a decision or an event without hidden state. Orchestration (`backtester/orchestration.py`, `portfolio/backtest_orchestration.py`) sequences these pure steps across sessions.

**Exact accounting, no tolerances.** Monetary and quantity paths use `Decimal` with exact equality: no `math.isclose`, no tolerance bands, no implicit rounding and never `Decimal(float)`.

## Pipeline

Data flows through these stages, each a package under `src/stock_swing_d1/`:

1. **`data/`, `storage/`, `validation/`**: raw → clean → canonical Parquet, one pipeline per data type.
   - `data/d1_pipeline.py` with `data/norgate_d1_adapter.py`: raw provider D1 bars → `models/stock_bar.py` → validation (`validation/stock_bar_dataset.py`) → canonical Parquet (`storage/stock_bar_parquet.py`).
   - `data/corporate_action_pipeline.py` with `data/norgate_adjusted_d1_adapter.py` and `data/norgate_capital_event_adapter.py`: adjusted OHLC bars and capital events, cross-validated against raw bars (`validation/corporate_action_parity.py`).
   - `data/dividend_event_pipeline.py`: dividend events derived from bar deltas and cross-checked in `validation/dividend_event_dataset.py`.
   - Provider adapters are thin, lazily imported boundaries, and every test uses synthetic fake providers. Generated and provider-derived data under `data/` is never committed.
2. **`earnings/`**: the point-in-time earnings schedule. `normalization/` (provider-neutral ingestion of date revisions) → `persistence/` (idempotent, hashed Parquet publication) → `query/` (point-in-time reconstruction from one published snapshot) → `integration/` (the query-to-risk boundary). `earnings/pit.py` reconstructs what was known at decision time; `earnings/risk.py` implements the entry blackout and forced-exit rules.
3. **`indicators/`**: technical indicators feeding the baseline strategy.
4. **`strategy/baseline/`**: the baseline signal service. `config/strategy.yaml` is a checked declaration that must equal the frozen values in code; editing it cannot change a trading decision.
5. **`ranking/`**: turns a session's signals into a deterministic, hash-verified `CandidateRankingSnapshot` and adapts it into portfolio candidates without re-ranking.
6. **`risk/position_sizing/`**: position sizing from risk parameters.
7. **`execution/`**: per-concern execution services.
   - `entry/`: T+1 entry decisions, consulting the earnings blackout.
   - `protective_exit/`: gap-aware stop-loss and protective-exit methodology.
   - `open_position_exit/`: per-session exit decisions, including earnings-forced exits and the maximum holding period.
   - `costs/`: stateless cost and settlement pricing bound to the immutable policy in `config/costs.yaml`.
8. **`portfolio/`**: the state machine at the centre of the backtester and the sole authority for cash, positions, settlements, cost basis and the ledger. It contains the allocation policy and mechanism, immutable state and event models, a pure transition function, fail-closed invariant checks, and SHA-256 identity and persistence of state.
9. **`backtester/`**: the daily run loop that ties the stages together across sessions. The exact chronology and its ordering invariants are defined in `orchestration.py`, `validation.py` and their chronology tests.
10. **`backtest_results/`**: a read-only provenance and audit layer over a completed run. It re-validates a run against its own manifest and derives audit views from it. It never re-runs decisions, reprices execution or mutates state, and missing provenance fails closed.

The Phase 16 packages (`research_metrics/`, `baseline_experiment/`) are experimental and incomplete.

## Commands

```bash
python -m pytest -q                                              # full suite
python -m pytest tests/portfolio -q                              # one package
python -m pytest tests/backtester/test_phase15a_chronology.py -q # one file
python -m compileall -q src tests                                # fast syntax check
```

Pytest markers (registered in `pyproject.toml`):

- `hard_gate`: authoritative Phase 6C.8A earnings point-in-time query acceptance gate.
- `hard_gate_6c8b`: Phase 6C.8B earnings-risk integration acceptance gate.
- `provider_acceptance`: gates that need a licensed external earnings-provider sample. They skip unless `EXTERNAL_EARNINGS_SAMPLE_PATH` is set; no sample is included.
