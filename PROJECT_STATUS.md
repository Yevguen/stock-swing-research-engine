# Project status

**Development is paused (portfolio freeze, September 2026).**

## Completed baseline

Phases 1–15 form the completed baseline: point-in-time data methodology, market-data processing, corporate actions, earnings-event methodology, technical indicators, portfolio construction, deterministic portfolio state transitions, candidate ranking, historical backtesting, execution integration, accounting, persistence and audit-oriented validation.

## Phase 16: experimental / in progress

Research-metrics and baseline-experiment infrastructure was implemented and tested after Phase 15. Phase 16 as a whole is **not** complete, and no research results are claimed from it.

## Explicit scope limitations

This repository does not claim that:

- all 22 planned phases are complete;
- Phase 16 is complete;
- the strategy has demonstrated profitable market alpha;
- a canonical backtest has proved future profitability;
- the system has traded live capital;
- unavailable commercial market data has been replaced with fabricated production data.

The project is a research and software-engineering system and a portfolio project.

## Why development is paused

Further empirical work, including the canonical baseline experiment and the later research phases, requires production-grade licensed historical market data. Development is paused until such data can be obtained under suitable licence terms. The architecture, contracts and tests are designed so that development can resume from this state rather than restarting.

## Test evidence

See the test evidence table in [README.md](README.md). Those figures come from a clean run of this snapshot; the skipped tests are explained there.

## Short description

Phases 1–15 completed, with additional Phase 16 research infrastructure implemented before development was intentionally paused pending access to production-grade licensed market data.
