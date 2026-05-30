# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run full live pipeline (fetch data → price markets → fit curves)
python run_full_pipeline.py --slug-pattern "bitcoin-above-on-december-{}" --day-range 1 31

# Run backtest engine across historical market data
python prob_backrunner_engine.py --skip-data-fetch --limit 10

# CLI trade recommendation from latest fitted batch
python auto_reco.py --bankroll 1000 --min-edge 0.06 --kelly-fraction 0.15

# Parameter sweep (grid search over strategy params)
python parameter_sweep.py --batch-dir fitted_batch_results --sweep min_edge=0.04,0.06,0.08,0.10 --limited
python parameter_sweep.py --list-params  # show all 24 sweepable params

# Launch dashboard
streamlit run dashboard.py

# Signal diagnostics (Spearman + AUC between edge and outcomes)
python signal_diagnostics.py path/to/all_priced.csv

# Edge distribution plots
python diagnostics/edge_dist_plot.py

# Run pricing engine validation tests
python btc_pricing_engine.py

# Fetch/refresh BTC price data
python data_fetcher.py
```

## Architecture

### Core Pricing Engine (`btc_pricing_engine.py`)

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

Two output paths produce batch CSVs:
1. **Live mode**: `batch_pricing_runner.py` → `batch_results/<timestamp_UTC>/batch_results.csv` → `fitted_batch_results/<timestamp_UTC>/batch_with_fits.csv`
2. **Backtest mode**: `prob_backrunner_engine.py` → `backtested_probabilities/unfitted/batch_<YYYYMMDD_HHMMSS>.csv` → `backtested_probabilities/fitted/batch_<YYYYMMDD_HHMMSS>/batch_with_fits.csv`

Both modes output the same format: `slug, strike, market_price, p_real_mc, T_days, date, expiry_date`. The backtest mode does "time-travel" — at each timestamp it truncates BTC data to what was available then, runs fresh MC simulations, and saves results. `fit_probability_curves.py` runs logistic curve fitting on both paths (called internally by both runners).

### Strategy Layer (`auto_reco.py`)

`recommend_trades()` is the central function used by both dashboard and backtest. Flow:
1. Filter by min/max price, model probability, DTE, moneyness
2. Generate YES/NO candidates based on edge ≥ min_edge (or prob_threshold mode)
3. Score candidates = edge × stability_penalty × staleness_multiplier
4. Per-expiry selection: `_select_expiry_candidates()` — max_bets_per_expiry, at most 1 sign change, only YES→NO structure allowed (not NO→YES)
5. Kelly sizing: `f* = (p-q)/(1-q)` for YES, `(q-p)/q` for NO, then × total_multiplier × corr_multiplier × fractional_kelly, capped at 0.30
6. Per-expiry-cap scaling, total-cap scaling, net delta limits

### Parameter System (`sweep_config.py`)

`SweepConfig` dataclass is the single source of truth for all 24 strategy parameters. Both `dashboard.py` sidebar and `parameter_sweep.py` pull defaults from it. Parameters include: edge thresholds, Kelly fraction, bet limits, price filters, DTE/moneyness filters, probability threshold mode, stability penalty, fixed stake, correlation penalty.

### Backtesting (`backtest_engine.py`)

`BacktestEngine` class runs chronological simulation:
1. Sort batches by timestamp
2. Per batch: settle expired positions (check BTC price at 12:00 ET expiry) → execute new trades via `recommend_trades()`
3. Track all priced contracts (not just taken trades) in `_all_priced_contracts` for decile-conditioned shuffle tests
4. Output: trades_df (consolidated entries), equity_df (snapshots)

### Shuffle Tests (`backtest_montecarlo_sim.py`)

Two modes:
- **Expiry-only**: Shuffles outcomes within each expiry among taken trades only
- **Decile-conditioned** (`--all_trades`): Uses all priced contracts as outcome pool, conditions on edge decile bins, cascade pool fallback (snapshot+expiry+decile → expiry+decile → expiry → global)

### Dashboard (`dashboard.py`)

Streamlit app with 8 tabs: Curves & Edges, Stability, Volatility & Regimes, Calibration, Recommendations, Positions, Backtest, Historical Stability. Loads batch CSVs from directory scan or upload. All strategy sliders in sidebar mirror SweepConfig defaults.

### Live Trading (`polymarket/`)

- `intent_builder.py`: Converts auto_reco DataFrame to OrderIntent objects with deterministic ID generation and share rounding
- `execution_gateway.py`: Validates intents against available collateral, submits via PolymarketProvider
- `accounting.py`: Tracks collateral, fills, and settlement status
- `db.py`: SQLite persistence for runs, intents, submissions, account states
- `models.py`: Dataclasses — OrderIntent, Submission, AccountState, etc.
- `pages/polymarket_console.py`: Streamlit page for operator workflow (generate → approve → submit → monitor)

### Position Tracking (`positions.py`)

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

- **Convention over configuration**: Column names resolved via precedence chains, not explicit config
- **UTC everywhere**: All timestamps stored/normalized to UTC; ET conversions only for display and expiry settlement (12:00 PM ET)
- **NumPy RNG v2**: Uses `np.random.default_rng(seed)` — never `np.random.seed()`
- **CSV-based state**: No database for positions or batch data; everything is CSV files on disk
- **Idempotent backtest steps**: Backtest runner skips already-processed timestamps
