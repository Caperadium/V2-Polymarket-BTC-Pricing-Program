# Architecture

## System Overview

The V2 BTC Contract Pricing system automates the full lifecycle of trading Bitcoin binary options on Polymarket:

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Layer │───▶│  Pricing Engine  │───▶│  Curve Fitting  │
│  (BTC CSV)  │    │  (GARCH + MC)    │    │  (Logistic)     │
└─────────────┘    └──────────────────┘    └────────┬────────┘
                                                    │
                                                    ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Execution  │◀───│  Intent Builder  │◀───│  Strategy Layer │
│  (CLOB API) │    │  (OrderIntent)   │    │  (3-Stage)      │
└─────────────┘    └──────────────────┘    └─────────────────┘
```

## Component Layers

### 1. Data Layer (`core/data/`)

| File | Purpose |
|------|---------|
| `data_fetcher.py` | Downloads BTC daily + intraday prices from CoinGecko and Binance |
| `positions.py` | CSV-based position ledger with auto-expiry and batch-sync |

BTC data flows into the pricing engine as two CSVs: daily for GARCH fitting, intraday for current spot price.

### 2. Pricing Engine (`core/pricing/`)

| File | Purpose |
|------|---------|
| `btc_pricing_engine.py` | GARCH(1,1) + Student-t + Kou Jump Diffusion Monte Carlo simulator |
| `fit_probability_curves.py` | Logistic curve fitting per expiry, logit-shift calibration |

The engine prices each strike by simulating 50k–150k BTC price paths and computing `P(S_T ≥ strike)`.

### 3. Strategy Layer (`core/strategy/`)

| File | Purpose |
|------|---------|
| `auto_reco.py` | 3-stage pipeline: Target → Delta → Action |
| `common.py` | Shared types (`TargetPosition`, `DeltaIntent`, `RebalanceConfig`) |
| `vol_gate.py` | Volatility regime detection from intraday data |
| `signal_diagnostics.py` | Spearman correlation + AUC between edge and outcomes |

The strategy layer converts model probabilities into actionable trade recommendations with Kelly sizing and risk controls.

### 4. Execution Layer (`polymarket/`)

| File | Purpose |
|------|---------|
| `intent_builder.py` | Converts recommendations to `OrderIntent` objects |
| `execution_gateway.py` | Validates + submits orders with collateral tracking |
| `provider_polymarket.py` | Polymarket CLOB API integration |
| `accounting.py` | Provider abstraction + account state management |
| `models.py` | Dataclasses: `Run`, `OrderIntent`, `Submission`, `AccountState` |
| `db.py` | SQLite persistence for runs, intents, submissions |
| `ingest.py` | Syncs trades + closed positions from Polymarket API |
| `metrics.py` | PnL, drawdown, daily loss metrics |
| `reconcile.py` | Submission status reconciliation with provider |

### 5. Scripts & Pipelines (`scripts/`)

| Directory | Purpose |
|-----------|---------|
| `pipelines/` | `run_full_pipeline.py` (end-to-end), `batch_pricing_runner.py` (live pricing) |
| `backtesting/` | `backtest_engine.py`, `prob_backrunner_engine.py`, `backtest_montecarlo_sim.py` |
| `utilities/` | `parameter_sweep.py`, `plot_batch_curves.py`, aggregation tools |

### 6. Applications (`app/`)

| File | Purpose |
|------|---------|
| `dashboard.py` | 8-tab Streamlit monitoring dashboard |
| `pages/backtesting.py` | Backtest results tab |
| `pages/polymarket_console.py` | Operator workflow: generate → approve → submit → monitor |

## Data Flow

### Live Pipeline Flow

1. **Fetch** — `data_fetcher.py` downloads latest BTC prices
2. **Query** — `batch_pricing_runner.py` fetches Polymarket contracts via Gamma API
3. **Simulate** — `btc_pricing_engine.py` runs 50k MC paths per contract
4. **Fit** — `fit_probability_curves.py` fits logistic curves, applies calibration
5. **Enrich** — Order book enrichment fetches live ask/bid from CLOB
6. **Recommend** — `auto_reco.py` generates BUY/SELL signals
7. **Execute** — `intent_builder.py` → `execution_gateway.py` → CLOB API

### Backtest Flow

1. **Time-travel** — `prob_backrunner_engine.py` iterates historical timestamps
2. **Truncate** — At each timestamp, BTC data is truncated to "available at time"
3. **Re-simulate** — Fresh MC simulations at each point using only past data
4. **Fit-per-batch** — Logistic curves fitted per historical snapshot
5. **Replay** — `backtest_engine.py` runs strategy chronologically
6. **Shuffle** — `backtest_montecarlo_sim.py` tests statistical significance

## Key Design Decisions

### Column Name Precedence (Convention over Configuration)

Columns are resolved via precedence chains, not explicit config:

```python
# Model probability: p_model_cal > p_model_fit > p_real_mc > model_probability
price_col = _pick_column(df, ["market_price", "market_pr", "Polymarket_Price"])
```

### UTC Everywhere

All timestamps are stored and normalized to UTC. ET conversions happen only for display and expiry settlement (12:00 PM ET).

### CSV-Based State

Positions and batch data use CSV files — no database for core data. Only the Polymarket execution layer uses SQLite.

### NumPy RNG v2

Uses `np.random.default_rng(seed)` consistently — never `np.random.seed()`.

### Idempotent Steps

Backtest runner skips already-processed timestamps. Intent builder uses deterministic IDs for safe upserts.
