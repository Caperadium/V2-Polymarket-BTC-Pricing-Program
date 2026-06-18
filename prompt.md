# Task: Refactor Backtesting System into Unified Module

## Context

The backtesting system is currently split across three modules:
- **prob_backrunner_engine.py** — generates smoothed probabilities from BTC price data
- **The backtester** — runs auto-reco on the list of smoothed probabilities from the backrunner engine
- **signal_diagnostics.py** — computes diagnostic metrics on signals

These need to be merged into a single unified backtesting module. The module is called from the backtesting tab on the dashboard.

## Read the codebase first

Before writing any code, read and understand:
1. `prob_backrunner_engine.py` — understand its inputs, outputs, and how it generates smoothed probabilities
2. The current backtester module — understand what metrics it logs and how auto-reco works
3. `signal_diagnostics.py` — understand all metrics it computes
4. The dashboard code for the backtesting tab — understand how it currently calls the backtester and displays results

## Requirements

### 1. Historical Contract Price Storage

Maintain a local file called `historical_contract_prices` (CSV or other appropriate format) that stores daily Polymarket contract prices for all closed "bitcoin-above" style contracts.

Each row should contain at minimum: contract slug, clobTokenId, date, price, resolution (yes/no), strike price, expiry date.

### 2. Incremental Data Fetching

When the backtester is called, it should:

1. Read `historical_contract_prices` and find the latest contract close date stored.
2. Query the Polymarket Gamma API to discover all closed contracts that resolved after that date:
   ```
   GET https://gamma-api.polymarket.com/markets?closed=true&limit=100
   ```
   Paginate through results. Filter client-side for slugs matching the bitcoin-above pattern used by this repo. Use `/markets/keyset` cursor-based pagination if offset-based proves unreliable for large result sets.

3. For each newly discovered closed contract, fetch its price history:
   ```
   GET https://clob.polymarket.com/prices-history
   Params:
     market: <clobTokenId>
     interval: 1d
   ```
   This returns daily prices snapped to **midnight UTC**. If daily returns empty (can happen for resolved markets), fall back to `interval=max` with `fidelity=720` for 12-hour granularity (snaps to midnight and noon UTC) and take only the midnight data points.

4. Append new data to `historical_contract_prices`. Do not re-fetch contracts already stored.

5. Use only `requests` for API calls — no Polymarket SDK (the py-clob-client repo was archived May 2026). No API key needed; these are public read endpoints.

### 3. Backrunner Engine Integration

After the data fetch step, run the backrunner engine (currently in `prob_backrunner_engine.py`) over all stored historical contract prices.

**Critical alignment requirement:** The BTC spot prices fed into the backrunner engine must be truncated to the same timestamps as the historical contract prices (midnight UTC). If the backrunner currently uses BTC prices at different timestamps, align them. The model's probability output for a given date should be based only on BTC data available as of midnight UTC on that date — no look-ahead.

### 4. Metrics and Diagnostics

The unified module should compute and log:
- All metrics currently computed by the backtester (preserve existing functionality)
- All metrics currently computed by `signal_diagnostics.py`

Do not drop any existing metrics. Consolidate the computation into one pass where possible.

### 5. Dashboard Integration

All metrics should be displayed on the dashboard in the backtesting tab where this module is called from. Preserve the existing dashboard interface patterns — this should be a drop-in replacement, not a redesign of the dashboard.

### 6. Module Consolidation

After the refactor:
- `prob_backrunner_engine.py` functionality is absorbed into the new module
- `signal_diagnostics.py` functionality is absorbed into the new module
- The old backtester module is replaced

Remove or clearly deprecate the old modules so there's no ambiguity about which code is live.

## Constraints

- Do not change the model's pricing logic — the refactor is infrastructure only.
- Do not change how the dashboard calls the backtester unless necessary to support the new data flow.
- Handle edge cases: first run with no historical data (cold start fetches everything), API returning empty responses, contracts with no trade activity.
- The module should be idempotent — running it twice in a row should not duplicate data or produce different results.