# stock-swing-research-engine

A deterministic research and backtesting engine, written in Python, for long-only, unleveraged, daily-bar (D1) swing-trading research on liquid US large-cap equities.

The project is an exercise in research-software engineering: point-in-time data handling, explicit anti-look-ahead rules, corporate-action and dividend semantics, exact portfolio accounting, deterministic backtest orchestration, and reproducible, hash-verified results. It is research software, not a trading system and not investment advice.

## Project status

- **Phases 1–15: complete.** Data models and pipelines, corporate actions, the point-in-time earnings model, indicators, the baseline strategy, entry and exit execution, risk sizing, the portfolio engine, candidate ranking and the end-to-end historical backtest engine.
- **Phase 16: experimental / in progress.** Research-metrics and baseline-experiment infrastructure is implemented and tested, but the phase is incomplete and no research results are claimed from it.
- Development is paused pending access to production-grade licensed market data. See [PROJECT_STATUS.md](PROJECT_STATUS.md).

This repository does **not** claim that the strategy is profitable, that any backtest demonstrates future performance, or that the system has traded real capital.

## What the code demonstrates

- **Point-in-time and anti-look-ahead design.** Decisions use only completed daily bars; a signal from session T executes no earlier than session T+1; earnings knowledge is reconstructed as it was known at decision time.
- **Deterministic, auditable state.** Frozen policy objects with domain-separated SHA-256 fingerprints, a pure portfolio state-transition engine and fail-closed invariant checks.
- **Exact accounting.** Monetary and quantity paths use `Decimal` with exact equality: no tolerances and no floating-point money.
- **Data engineering.** Raw → clean → canonical Parquet pipelines with validation and parity checks behind narrow adapter boundaries.
- **Testing.** An extensive pytest suite with Hypothesis property-based tests, authoritative acceptance gates and scope tests that enforce architectural boundaries.

Architecture details are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start

Requires Python 3.12 or later. The Python package is imported as `stock_swing_d1`.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

No market data or provider package is needed to run the tests (see [DATA_AND_LICENSING.md](DATA_AND_LICENSING.md)).

## Test evidence

Verified on this snapshot with a clean installation (Python 3.12.10 on Linux, without the optional provider package):

| Collected | Passed | Failed | Errors | Skipped |
|---:|---:|---:|---:|---:|
| 3,665 | 3,655 | 0 | 0 | 10 |

The skipped tests are acceptance gates that need a licensed external earnings-provider sample, which is not distributed.

## Research defaults

The parameters in `config/strategy.yaml` (for example SMA 20/50, RSI 14 and ATR 14) are fixed research defaults used to exercise the engine. They are not investment recommendations and are not evidence of profitability.

## Data and licensing

The MIT licence covers only this project's own source code. No market data, provider-derived data, provider software or provider documentation is distributed. See [DATA_AND_LICENSING.md](DATA_AND_LICENSING.md).

## Development note

AI coding tools assisted development and code review; requirements, design decisions and final verification were the author's.

## Licence

[MIT](LICENSE), for this project's own source code only.
