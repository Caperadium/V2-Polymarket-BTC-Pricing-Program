# Architecture

## System Overview

The V2 BTC Contract Pricing system automates the full lifecycle of trading Bitcoin binary options on Polymarket:

```
┌──────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Data Layer     │───▶│   Pricing Engine v2  │───▶│  Curve Fitting  │
│ (BTC + Macro)    │    │ (GARCH+SVCJ+FIGARCH(1,d,1)) │    │   (Logistic)    │
└──────────────────┘    └──────────┬───────────┘    └────────┬────────┘
                                   │                         │
                    ┌──────────────┼──────────────┐          │
                    ▼              ▼              ▼          ▼
            ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐
            │  Regime  │  │    Jump      │  │  Basel   │  │  Strategy   │
            │ Detector │  │ Calibration  │  │ Backtest │  │   Layer     │
            │  (HMM)   │  │   (MAD/MLE)  │  │  (VaR)   │  │ (3-Stage)   │
            └──────────┘  └──────────────┘  └──────────┘  └──────┬──────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Execution  │◀───│  Intent Builder  │◀───│  Directional    │
│ (CLOB API)  │    │  (OrderIntent)   │    │  XGBoost        │
└─────────────┘    └──────────────────┘    └─────────────────┘
```

## Component Layers

### 1. Data Layer (`core/data/`)

| File | Purpose |
|------|---------|
| `data_fetcher.py` | Downloads BTC daily + hourly + intraday prices from Binance |
| `macro_fetcher.py` | Downloads Gold, DXY, VIX, SPX daily data from Yahoo Finance |
| `positions.py` | CSV-based position ledger with auto-expiry and batch-sync |

BTC data flows into the pricing engine. Macro data feeds the regime detector and directional XGBoost.

### 2. Pricing Engine (`core/pricing/`)

| File | Purpose | Phase |
|------|---------|-------|
| `btc_pricing_engine.py` | GARCH+SVCJ+Skewed-t+FIGARCH(1,d,1) Monte Carlo simulator (v2, hourly) | 0–2.6 |
| `fit_probability_curves.py` | Logistic curve fitting per expiry, logit-shift calibration | Post |
| `jump_calibration.py` | MAD-based Kou jump parameter estimation + SVCJ vol jump params | 0.5 |
| `regime_detector.py` | 3-state HMM regime detection (bear/sideways/bull) | 1.2 |
| `directional_xgb.py` | XGBoost classifier for P(up) modifier | 2.3 |

### 3. Validation Layer (`core/validation/`)

| File | Purpose |
|------|---------|
| `basel_backtest.py` | Kupiec POF traffic light VaR backtest + expected shortfall (Acerbi-Szekely) |
| `calibration_metrics.py` | Brier score, Expected Calibration Error (ECE), reliability diagrams |

Validates model adequacy across multiple horizons and confidence levels using regulatory-standard tests and calibration diagnostics.

### 4. Strategy Layer (`core/strategy/`)

| File | Purpose |
|------|---------|
| `auto_reco.py` | 3-stage pipeline: Target → Delta → Action |
| `common.py` | Shared types (`TargetPosition`, `DeltaIntent`, `RebalanceConfig`) |
| `vol_gate.py` | Volatility regime detection from intraday data |
| `signal_diagnostics.py` | Spearman correlation + AUC between edge and outcomes |

The strategy layer converts model probabilities into actionable trade recommendations with Kelly sizing and risk controls.

### 5. Execution Layer (`polymarket/`)

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

### 6. Scripts & Pipelines (`scripts/`)

| Directory | Purpose |
|-----------|---------|
| `pipelines/` | `run_full_pipeline.py` (end-to-end), `batch_pricing_runner.py` (live pricing) |
| `backtesting/` | `backtest_engine.py`, `prob_backrunner_engine.py`, `backtest_montecarlo_sim.py` |
| `utilities/` | `parameter_sweep.py`, `plot_batch_curves.py`, aggregation tools |

### 7. Applications (`app/`)

| File | Purpose |
|------|---------|
| `dashboard.py` | 8-tab Streamlit monitoring dashboard |
| `pages/backtesting.py` | Backtest results tab |
| `pages/polymarket_console.py` | Operator workflow: generate → approve → submit → monitor |

## Data Flow

### Live Pipeline Flow

1. **Fetch** — `data_fetcher.py` downloads latest BTC prices; `macro_fetcher.py` downloads macro data
2. **Calibrate** — `jump_calibration.py` estimates Kou + SVCJ parameters from hourly returns
3. **Detect Regime** — `regime_detector.py` classifies current market (bear/sideways/bull)
4. **Query** — `batch_pricing_runner.py` fetches Polymarket contracts via Gamma API
5. **Simulate** — `btc_pricing_engine.py` runs regime-weighted MC paths per contract
6. **Adjust** — `directional_xgb.py` provides P(up) modifier from macro + BTC features
7. **Fit** — `fit_probability_curves.py` fits logistic curves, applies calibration
8. **Enrich** — Order book enrichment fetches live ask/bid from CLOB
9. **Recommend** — `auto_reco.py` generates BUY/SELL signals
10. **Execute** — `intent_builder.py` → `execution_gateway.py` → CLOB API

### Backtest Flow

1. **Time-travel** — `prob_backrunner_engine.py` iterates historical timestamps
2. **Truncate** — At each timestamp, BTC data is truncated to "available at time"
3. **Re-calibrate** — Jump parameters and regime labels re-estimated on truncated data
4. **Re-simulate** — Fresh MC simulations at each point using only past data
5. **Fit-per-batch** — Logistic curves fitted per historical snapshot
6. **Validate** — `basel_backtest.py` runs rolling VaR backtests at multiple horizons
7. **Replay** — `backtest_engine.py` runs strategy chronologically
8. **Shuffle** — `backtest_montecarlo_sim.py` tests statistical significance

## Key Design Decisions

### Feature-Flag Backward Compatibility

All new pricing engine features default to `False` (`use_svcj`, `use_skewed_t`, `use_figarch`, `use_regime_switching`) except `use_naive_prior=True`. Existing callers work without changes. New parameters appended to function signatures.

### Phase Architecture

The pricing engine is built in phases, each adding one capability. All phases are independently toggleable:

| Phase | Feature | Flag | Status |
|-------|---------|------|--------|
| 0 | GARCH(1,1) + Student-t | (core) | Always on |
| 0.1 | Structural break filter | `training_start_date` | Default: 2019-10-01 |
| 0.5 | Jump calibration cache | `load_calibrated_jumps()` | Cache-based |
| 1.1 | Naive prior (μ=0) | `use_naive_prior=True` | Default on |
| 1.2 | HMM regime detection | `use_regime_switching` | Opt-in |
| 1.3 | SVCJ correlated jumps | `use_svcj` | Opt-in |
| 1.4 | Hansen skewed-t | `use_skewed_t` | Opt-in |
| 1.5 | Horizon gating | (automatic) | Always active |
| 2.3 | Directional XGBoost | `use_xgb_direction` | Opt-in |
| 2.4 | Regime-conditional jumps | `regime_params` | With HMM |
| 2.5 | FIGARCH(1,d,1) long memory | `use_figarch` | Opt-in |
| 2.6 | Vol gate interaction | `vol_gate_regime` | With vol gate |

Phases 0–1.5 form the base model; 2.3–2.6 add cross-signal integration.

### Post-Hoc Regime Weighting

Three independent MC simulations (bear/sideways/bull) weighted by HMM posterior, not intra-path regime switching. Avoids path-continuity issues and simplifies implementation.

### Column Name Precedence (Convention over Configuration)

Columns are resolved via precedence chains, not explicit config:

```python
# Model probability: p_model_fit > p_real_mc > model_probability
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

### Modular Risk Components

Vol gate, jump calibration, regime detector, and Basel backtest are all standalone modules — tested in isolation, plugged into the strategy pipeline as needed.
