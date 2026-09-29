# Data and licensing

## Licence scope

The [MIT licence](LICENSE) applies only to this project's own source code, including its tests, configuration and documentation. It grants no rights to any third-party data, software, documentation or trademarks.

## What this repository does not contain

- No market data of any kind: no prices, volumes, corporate actions, dividends, earnings dates, index membership or security identifiers obtained from a data provider.
- No provider-derived data, caches, extracts or reports.
- No provider software, libraries or documentation.

The `data/` and `reports/` directories contain only placeholder files. Pipeline output written there is excluded by `.gitignore` and must never be committed.

## Optional provider integration

The data pipeline was designed around Norgate Data as its intended historical US-equities source. This repository includes only the project's own adapter code (`src/stock_swing_d1/data/norgate_*_adapter.py`) and the identifier format `NORGATE:<AssetId>` used by the domain model.

The `norgatedata` Python package is not bundled; it is declared only as the optional extra `norgate`. Using the adapters requires your own installation of that package and your own valid data subscription and licence. None of this is needed to run the test suite.

This project is not affiliated with, endorsed by or sponsored by Norgate Data. Provider, index and product names are used only to identify integration targets and methodology, and remain the property of their owners.

Provider information is available from the [Norgate Data website](https://norgatedata.com/).

## Test data

Test fixtures use invented security identifiers and market values. Real ticker symbols or index names that appear are used only as format examples or benchmark names and are not paired with provider data.

The external earnings-provider acceptance tests skip unless a separately licensed sample file is supplied through `EXTERNAL_EARNINGS_SAMPLE_PATH`. No such sample is included.

## Not investment advice

This is research software. Nothing in this repository is investment advice, a recommendation to trade or evidence that any strategy is profitable. The parameters in `config/strategy.yaml` are research defaults used to exercise the engine.
