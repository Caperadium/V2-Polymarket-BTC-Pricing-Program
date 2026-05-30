# Backtesting Methodology

`scripts/backtesting/`

The backtesting system simulates the strategy against historical data to measure performance and statistical significance.

## Architecture

Two components work together:

| Component | File | Purpose |
|-----------|------|---------|
| **Prob Backrunner** | `prob_backrunner_engine.py` | Time-travels through history, re-simulates pricing at each timestamp |
| **Backtest Engine** | `backtest_engine.py` | Chronological simulator: settle → trade → track |

## Prob Backrunner (Data Generation)

The backrunner recreates what the pricing engine *would have seen* at each point in history:

```bash
python scripts/backtesting/prob_backrunner_engine.py --skip-data-fetch --limit 10
```

### Time-Travel Algorithm

1. Load all BTC data into memory with datetime index
2. Loop through historical timestamps (from `old_market_prices.csv`)
3. At each timestamp, **truncate** BTC data to only what was available then
4. Run fresh `calculate_probabilities()` with truncated data
5. Save per-batch CSVs in `backtested_probabilities/unfitted/`
6. Run `fit_probability_curves.py` → `backtested_probabilities/fitted/`

### Key Design

- `O(log n)` DataFrame slicing per timestamp (no disk I/O in loop)
- GARCH re-fit per timestamp (expensive but accurate)
- Each batch is a self-contained "snapshot" of what the model thought at that time

## Backtest Engine (Strategy Simulation)

The engine replays trading decisions chronologically through the generated batches:

```python
from scripts.backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine(
    market_data_batches=batches,
    initial_bankroll=1000.0,
    strategy_params={'kelly_fraction': 0.15, 'min_edge': 0.06},
)
trades_df, equity_df = engine.run()
```

### Simulation Loop

For each batch (chronological):

1. **Settle expired** — Check BTC price at 12:00 ET on expiry day; mark positions as won/lost
2. **Execute trades** — Run `recommend_trades()` with current positions and bankroll
3. **Track equity** — Record bankroll snapshot after each batch

### Settlement

Outcomes are determined by comparing BTC price at 12:00 ET on the expiry day against the contract strike:

```python
outcome_yes = 1.0 if btc_price > strike else 0.0  # Strict > for YES
```

BTC prices are looked up from intraday 1-minute data with ±5 minute tolerance.

### Output DataFrames

| Output | Columns |
|--------|---------|
| `trades_df` | Consolidated entries + exits per contract |
| `equity_df` | Bankroll snapshots over time |
| `all_priced_df` | Every evaluated contract (not just trades), for shuffle tests |

## Monte Carlo Shuffle Tests

`backtest_montecarlo_sim.py` tests whether results are statistically significant.

### Expiry-Only Shuffle (Default)

Shuffles outcomes **within each expiry** among taken trades only:

```bash
python scripts/backtesting/backtest_montecarlo_sim.py
```

Tests: "Given which expiries we traded, was our contract selection better than random?"

### Decile-Conditioned Shuffle (`--all_trades`)

A stronger null model using **all priced contracts** as the outcome pool:

```bash
python scripts/backtesting/backtest_montecarlo_sim.py --all_trades
```

**Algorithm**:

1. **Edge decile binning** — Contracts grouped by `abs(model_prob − market_price)` into deciles
2. **Pool adequacy cascade** — For each trade group, selects smallest adequate pool:
    - `(snapshot_time, expiry_date, edge_decile)` — most specific
    - `(expiry_date, edge_decile)` — broader
    - `(expiry_date)` — broader still
    - Global pool — fallback
3. **Sampling** — Without replacement when pool ≥ group, with replacement otherwise

### Interpreting Results

| Metric | Meaning |
|--------|---------|
| **Z-Score** | > 2.0 suggests skill (p < 0.05) |
| **Percentile** | > 95% means your PnL beat 95% of random shuffles |
| **Significant: Yes** | Result unlikely due to luck |
| **Settled PnL** | Realized PnL from settled trades only (used in shuffle tests) |
| **Net PnL** | Final − starting bankroll (includes open positions) |

### Two PnL Metrics

- **Settled PnL** (shuffle test): Only settled trades. What the Monte Carlo uses.
- **Net PnL** (equity curve): Includes unrealized PnL from open positions.

If these differ significantly, you have open positions at the end of the backtest period.
