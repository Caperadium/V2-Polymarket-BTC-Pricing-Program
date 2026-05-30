# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run full live pipeline (fetch data → price markets → fit curves)
python scripts/pipelines/run_full_pipeline.py --slug-pattern "bitcoin-above-on-december-{}" --day-range 1 31

# Run backtest engine across historical market data
python scripts/backtesting/prob_backrunner_engine.py --skip-data-fetch --limit 10

# CLI trade recommendation from latest fitted batch
python core/strategy/auto_reco.py --bankroll 1000 --min-edge 0.06 --kelly-fraction 0.15

# Parameter sweep (grid search over strategy params)
python scripts/utilities/parameter_sweep.py --batch-dir fitted_batch_results --sweep min_edge=0.04,0.06,0.08,0.10 --limited
python scripts/utilities/parameter_sweep.py --list-params  # show all 24 sweepable params

# Launch dashboard
streamlit run app/dashboard.py

# Launch Polymarket trade console
streamlit run app/pages/polymarket_console.py

# Signal diagnostics (Spearman + AUC between edge and outcomes)
python core/strategy/signal_diagnostics.py path/to/all_priced.csv

# Edge distribution plots
python diagnostics/edge_dist_plot.py

# Run pricing engine validation tests
python core/pricing/btc_pricing_engine.py

# Fetch/refresh BTC price data
python core/data/data_fetcher.py

# Vol gate (standalone risk signal)
python core/strategy/vol_gate.py --file DATA/btc_intraday_1m.csv

# Run tests
python -m pytest tests/ -v

# Build documentation
mkdocs build
mkdocs serve
```

## Directory Structure

```
.
├── app/                          # Streamlit applications
│   ├── dashboard.py              #   Main 8-tab monitoring dashboard
│   └── pages/
│       ├── backtesting.py        #   Backtesting page
│       ├── polymarket_console.py #   Trade execution operator console
│       └── polymarket_console_fixed.py
├── core/                         # Domain logic (no scripts/pipelines)
│   ├── data/
│   │   ├── data_fetcher.py       #   Binance BTC data download
│   │   └── positions.py          #   CSV-based position ledger
│   ├── pricing/
│   │   ├── btc_pricing_engine.py #   GARCH+Student-t+Jump MC simulator
│   │   └── fit_probability_curves.py # Logistic curve fitting
│   └── strategy/
│       ├── auto_reco.py          #   3-stage trade recommendation pipeline
│       ├── BH_auto_reco.py       #   Previous auto_reco (pre-refactor backup)
│       ├── common.py             #   Shared types, constants (TargetPosition, TargetRole)
│       ├── vol_gate.py           #   BTC volatility risk gate
│       └── signal_diagnostics.py #   Spearman/AUC analysis
├── scripts/                      # Executable scripts & pipelines
│   ├── backtesting/
│   │   ├── backtest_engine.py    #   Chronological simulation engine
│   │   ├── backtest_montecarlo_sim.py # Shuffle tests (expiry-only & decile-conditioned)
│   │   └── prob_backrunner_engine.py  # Time-travel backtest orchestrator
│   ├── pipelines/
│   │   ├── run_full_pipeline.py  #   Full pipeline: fetch → price → fit
│   │   └── batch_pricing_runner.py   # Live Polymarket batch pricing
│   ├── utilities/
│   │   ├── parameter_sweep.py    #   Grid search over strategy params
│   │   ├── plot_batch_curves.py  #   Probability curve plots
│   │   └── aggregate_old_batch_data.py # Legacy data aggregation
│   └── migrate_db.py
├── polymarket/                   # CLOB execution layer
│   ├── accounting.py             #   Collateral and fill tracking
│   ├── intent_builder.py         #   auto_reco → OrderIntent conversion
│   ├── execution_gateway.py      #   Validation and submission
│   ├── provider_polymarket.py    #   Polymarket CLOB/API provider
│   ├── models.py                 #   Dataclasses (OrderIntent, Submission, etc.)
│   ├── db.py                     #   SQLite persistence
│   ├── ingest.py                 #   Market data ingestion
│   ├── metrics.py                #   Performance metrics
│   ├── reconcile.py              #   Position reconciliation
│   ├── date_utils.py             #   Date/time helpers
│   └── intents.db                #   SQLite database file
├── tests/                        # Test suite
│   ├── test_auto_reco_refactor.py
│   ├── test_backtest_inversion.py
│   ├── smoke_test_console_logic.py
│   ├── verify_dashboard_refactor.py
│   ├── verify_fallback_warning.py
│   └── debug_reco.py
├── sweep_config.py               # SweepConfig dataclass (at root — shared import)
├── deprecated/                   # Old pricing/backtest engines (not used)
├── diagnostics/                  # Edge distribution analysis
├── DOCS/                         # MkDocs documentation site
├── DATA/                         # BTC price CSV files
├── mkdocs.yml                    # MkDocs configuration
└── CLAUDE.md
```

## Architecture

### Core Pricing Engine (`core/pricing/btc_pricing_engine.py`)

GARCH(1,1) + Student-t + Kou Double Exponential Jump Diffusion Monte Carlo simulator. The high-level entry point is `calculate_probabilities()` which accepts strikes and days-to-expiry, returns `{strike: probability}` dict. Key design decisions:
- **Momentum injection**: When `drift_window` is set, EMA drift replaces structural mean — with global gating (not per-path) to avoid selection bias
- **Dynamic per-path drift clamping**: Drift clamped to ±0.25 × path-specific sigma_day
- **Multi-jump aggregation**: Poisson compound jumps per day, Gamma-distributed magnitudes
- **Variance blending**: Optional RV blending via `rv_intraday` + `rv_blend_weight`
- **Jump drift correction**: Subtracted from structural mu, NOT applied to momentum mu

### Column Name Precedence Conventions

The codebase uses flexible column detection with `_pick_column()` / `get_column()` fallback chains. Critical conventions:
- **Model probability**: `p_model_cal` > `p_model_fit` > `p_real_mc` > `model_probability`
- **Market price**: `market_price` > `market_pr` > `Polymarket_Price`
- **Expiry**: `expiry_key` (derived from `expiry_date` string), `T_days` as float
- **Edge**: computed as `model_prob - market_price` (YES edge) or `market_price - model_prob` (NO edge)
- **Risk-neutral probability**: `p_rn_fit` > `risk_neutral_prob_fit` > `risk_neutral_prob`

### Data Pipeline

Two output paths produce batch CSVs in the same format (`slug, strike, market_price, p_real_mc, T_days, date, expiry_date`):
1. **Live mode**: `scripts/pipelines/batch_pricing_runner.py` → `batch_results/<timestamp_UTC>/batch_results.csv` → `fitted_batch_results/<timestamp_UTC>/batch_with_fits.csv`
2. **Backtest mode**: `scripts/backtesting/prob_backrunner_engine.py` → `backtested_probabilities/unfitted/batch_<YYYYMMDD_HHMMSS>.csv` → `backtested_probabilities/fitted/batch_<YYYYMMDD_HHMMSS>/batch_with_fits.csv`

The backtest mode does "time-travel" — at each timestamp it truncates BTC data to what was available then, runs fresh MC simulations, and saves results. `fit_probability_curves.py` runs logistic curve fitting on both paths (called internally by both runners).

### Strategy Layer (`core/strategy/auto_reco.py`)

Refactored to a **3-stage pipeline: Target → Delta → Action**:

**Stage 1 — Target**: Generate ideal position targets from edge analysis
- Filter by min/max price, model probability, DTE, moneyness
- Generate YES/NO candidates based on edge ≥ min_edge (or prob_threshold mode)
- Score candidates = edge × stability_penalty × staleness_multiplier
- Per-expiry selection: at most 1 sign change, only YES→NO structure allowed
- Also generates EXIT signals for existing positions below hysteresis threshold

**Stage 2 — Delta**: Compute required position changes
- Compare targets against existing positions
- Calculate add/reduce/exit sizing via fractional Kelly
- Kelly formula: `f* = (p-q)/(1-q)` for YES, `(q-p)/q` for NO, then × multipliers, capped at 0.30

**Stage 3 — Action**: Apply risk constraints and produce final trades
- Per-expiry cap scaling, total-cap scaling
- Net delta limit enforcement (±max_net_delta_frac)
- Correlated position penalty: `1/(1 + penalty × (n-1))` per expiry+direction group
- Exit hysteresis: edge must drop below `(entry_edge - hysteresis)` to exit, preventing churn

Key data types live in `core/strategy/common.py`:
- `TargetRole` enum: ENTRY, EXIT, HOLD_SAFETY, NEUTRAL
- `TargetPosition` dataclass: ideal state for a single contract
- Shared constants for caps, staleness windows, min trade sizes

### Vol Gate (`core/strategy/vol_gate.py`)

Standalone risk module that gates trading based on BTC realized volatility:
- Computes realized vol over 15m and 60m windows from intraday data
- Ranks current vol against trailing baseline (percentile)
- Classifies regime: `normal`, `high`, or `extreme`
- Returns `VolGateResult` with:
  - `allow_new_entries`: False in extreme regime
  - `edge_add_cents`: Additional edge required in high vol
  - `kelly_mult`: Multiplier for Kelly fraction (reduced in high vol)
  - `shock`: True if sudden spike detected

Integrated into `auto_reco.py` Stage 3 as final risk gate.

### Parameter System (`sweep_config.py`)

`SweepConfig` dataclass is the single source of truth for all 24 strategy parameters. Used by both `app/dashboard.py` sidebar and `scripts/utilities/parameter_sweep.py`. Parameters include: edge thresholds, Kelly fraction, bet limits, price/probability filters, DTE/moneyness filters, probability threshold mode, stability penalty, fixed stake, correlation penalty. Located at repo root for shared import convenience.

### Backtesting (`scripts/backtesting/backtest_engine.py`)

`BacktestEngine` class runs chronological simulation:
1. Sort batches by timestamp
2. Per batch: settle expired positions (check BTC price at 12:00 ET expiry) → execute new trades via `recommend_trades()`
3. Track all priced contracts (not just taken trades) in `_all_priced_contracts` for decile-conditioned shuffle tests
4. Output: trades_df (consolidated entries), equity_df (snapshots), all_priced_df (optional)

### Shuffle Tests (`scripts/backtesting/backtest_montecarlo_sim.py`)

Two modes:
- **Expiry-only**: Shuffles outcomes within each expiry among taken trades only
- **Decile-conditioned** (`--all_trades`): Uses all priced contracts as outcome pool, conditions on edge decile bins, cascade pool fallback (snapshot+expiry+decile → expiry+decile → expiry → global)

### Dashboard (`app/dashboard.py`)

Streamlit app with 8 tabs: Curves & Edges, Stability, Volatility & Regimes, Calibration, Recommendations, Positions, Backtest, Historical Stability. Loads batch CSVs from directory scan or upload. All strategy sliders in sidebar mirror SweepConfig defaults.

### Live Trading (`polymarket/`)

- `intent_builder.py`: Converts auto_reco DataFrame to OrderIntent objects with deterministic ID generation and share rounding (rounds DOWN to 2 decimals)
- `execution_gateway.py`: Validates intents against available collateral, submits via provider
- `provider_polymarket.py`: Polymarket CLOB API integration for order submission and fills
- `accounting.py`: Tracks collateral, fills, and settlement status
- `ingest.py`: Market data and price ingestion from Polymarket
- `metrics.py`: Trade performance and P&L metrics
- `reconcile.py`: Position reconciliation between local ledger and Polymarket
- `db.py`: SQLite persistence for runs, intents, submissions, account states
- `models.py`: Dataclasses — OrderIntent, Submission, AccountState, etc.
- `app/pages/polymarket_console.py`: Streamlit operator workflow (generate → approve → submit → monitor)

### Position Tracking (`core/data/positions.py`)

CSV-based (not SQLite) position ledger. Key functions:
- `load_positions()`: Reads `positions.csv`, auto-closes expired (past 12PM ET)
- `sync_open_positions_with_batch()`: Maintains `open_positions.csv` snapshot enriched with latest batch prices
- `ensure_position_keys()`: Creates stable `position_key = slug|side|expiry|strike` for cross-file joins

### Key Data Files

- `DATA/btc_daily.csv`, `DATA/btc_intraday_1m.csv` — fetched by `data_fetcher.py`
- `old_market_prices.csv` — historical Polymarket prices for backtesting
- `positions.csv` — live trading position ledger
- `resolved_markets.csv` — logged outcomes for calibration

### Parameter Sweep Output Structure

```
parameter_sweeps/
  temp/           # Runs during execution (deleted after completion)
  saved/          # Top 10 runs kept permanently
    0001/
      taken_trades.csv
      montecarlo_results.csv
      equity_curve.csv
      run_config.md
      logs.txt
```

## Key Design Patterns

- **3-stage pipeline (Target → Delta → Action)**: Strategy separates ideal targets from sizing and risk enforcement, enabling independent testing of each stage
- **Convention over configuration**: Column names resolved via precedence chains, not explicit config
- **UTC everywhere**: All timestamps stored/normalized to UTC; ET conversions only for display and expiry settlement (12:00 PM ET)
- **NumPy RNG v2**: Uses `np.random.default_rng(seed)` — never `np.random.seed()`
- **CSV-based state**: No database for positions or batch data; everything is CSV files on disk (except Polymarket execution state which uses SQLite)
- **Idempotent backtest steps**: Backtest runner skips already-processed timestamps
- **Vol gate as standalone module**: Volatility risk can be computed independently and tested in isolation, then plugged into the strategy pipeline
