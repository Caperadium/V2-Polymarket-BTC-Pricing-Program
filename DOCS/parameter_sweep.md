# Parameter Sweep Tool

The parameter sweep tool (`parameter_sweep.py`) runs systematic backtests across different strategy parameter combinations to find optimal settings.

## Quick Start

```bash
# Basic sweep over min_edge
python parameter_sweep.py --batch-dir fitted_batch_results --sweep min_edge=0.04,0.06,0.08

# Multi-parameter sweep
python parameter_sweep.py --batch-dir fitted_batch_results \
    --sweep min_edge=0.04,0.06 \
    --sweep kelly_fraction=0.10,0.15,0.20

# Preview without running
python parameter_sweep.py --sweep min_edge=0.04,0.06 --dry-run

# Use stronger shuffle test with all priced contracts
python parameter_sweep.py --sweep min_edge=0.04,0.06 --all_trades --limited
```

## Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--sweep PARAM=v1,v2,v3` | Parameter to sweep (can repeat) | - |
| `--fixed PARAM=value` | Fixed parameter value (can repeat) | - |
| `--batch-dir DIR` | Batch data directory | `fitted_batch_results` |
| `--workers N` | Parallel workers | 8 |
| `--max-runs N` | Limit total runs | All |
| `--resume` | Resume from last run index | - |
| `--dry-run` | Preview runs without executing | - |
| `--fail-fast` | Stop on first error | - |
| `--limited` | Show top 10 results by Z-score | - |
| `--seed N` | Base RNG seed | 42 |
| `--mc-iterations N` | Monte Carlo iterations | 500 |
| `--all_trades` | Use decile-conditioned shuffle test | - |
| `--list-params` | List all valid parameters | - |

## Available Parameters

Run `python parameter_sweep.py --list-params` to see all 24 sweepable parameters:

**Strategy Settings:**
- `min_edge` - Minimum edge required (default: 0.06)
- `max_bets_per_expiry` - Max trades per expiry (default: 3)
- `max_capital_per_expiry_frac` - Capital limit per expiry (default: 0.15)
- `max_capital_total_frac` - Total capital limit (default: 0.40)
- `max_net_delta_frac` - Net directional limit (default: 0.20)
- `min_price` / `max_price` - Price filters (default: 0.03, 0.95)
- `use_stability_penalty` - Apply curve fit penalty (default: True)
- `correlation_penalty` - Correlated position penalty (default: 0.25)

**Sizing:**
- `kelly_fraction` - Fractional Kelly (default: 0.15)
- `use_fixed_stake` - Use fixed $ amounts (default: False)
- `fixed_stake_amount` - Fixed stake in USD (default: 10.0)
- `bankroll` - Starting capital (default: 500.0)

**Filters:**
- `use_max_dte` - Enable DTE filter (default: True)
- `max_dte` - Max days to expiry (default: 2.0)
- `use_prob_threshold` - Enable probability thresholds (default: False)
- `prob_threshold_yes` - Trade YES when model prob >= this (default: 0.70)
- `prob_threshold_no` - Trade NO when model prob <= this (default: 0.30)
- `use_max_moneyness` - Enable moneyness filter (default: False)
- `min_moneyness` / `max_moneyness` - Moneyness range (default: 0.0, 0.05)

## Output

Each run creates a folder in `parameter_sweeps/XXXX/`:

```
parameter_sweeps/
├── 0001/
│   ├── taken_trades.csv      # Trades executed
│   ├── montecarlo_results.csv # Shuffle test stats
│   ├── equity_curve.csv      # Bankroll over time
│   ├── run_config.md         # Full parameter set
│   └── logs.txt              # Execution log
├── 0002/
│   └── ...
```

### Monte Carlo Results Columns

When using `--all_trades`, `montecarlo_results.csv` includes additional fields:

| Column | Description |
|--------|-------------|
| `shuffle_mode` | `decile_conditioned_all_priced` or `expiry_only` |
| `n_all_priced_used` | Number of contracts in outcome pool |
| `n_deciles_used` | Number of edge decile bins (target: 10) |
| `n_unmatched_trades` | Trades that couldn't be assigned to a decile |

## Monte Carlo Shuffle Tests

The tool supports two shuffle test modes:

### Expiry-Only Shuffle (Default)

Shuffles outcomes **within each expiry** among the taken trades only. This tests whether trade selection within an expiry has skill.

```bash
python parameter_sweep.py --sweep min_edge=0.04,0.06 --limited
```

### Decile-Conditioned Shuffle (`--all_trades`)

A stronger null model that uses **all priced contracts** as the outcome pool:

1. **Edge decile binning**: Contracts are grouped by `abs(model_prob - market_price)` into deciles
2. **Pool adequacy cascade**: For each trade group, selects the smallest adequate pool:
   - `(snapshot_time, expiry_date, edge_decile)` if pool ≥ group size
   - `(expiry_date, edge_decile)` if above is too small
   - `(expiry_date)` if still too small
   - Global pool as final fallback
3. **Sampling**: Without replacement when pool ≥ group, with replacement otherwise

This is statistically more rigorous because:
- Outcomes are drawn from the full universe of evaluated contracts, not just taken trades
- Conditioning on edge deciles respects that different edge strengths have different base win rates
- Grouping by snapshot time avoids mixing outcomes from different market conditions

```bash
python parameter_sweep.py --sweep min_edge=0.04,0.06 --all_trades --limited
```

## Examples

### Find Optimal Edge Threshold
```bash
python parameter_sweep.py --batch-dir fitted_batch_results \
    --sweep min_edge=0.02,0.04,0.06,0.08,0.10 \
    --limited
```

### Compare Kelly vs Fixed Stake
```bash
python parameter_sweep.py --batch-dir fitted_batch_results \
    --sweep use_fixed_stake=false,true \
    --sweep kelly_fraction=0.10,0.15,0.20 \
    --fixed fixed_stake_amount=10 \
    --limited
```

### Grid Search with Stronger Null Model
```bash
python parameter_sweep.py --batch-dir fitted_batch_results \
    --sweep min_edge=0.04,0.06,0.08 \
    --sweep max_bets_per_expiry=2,3,4 \
    --all_trades \
    --limited
```

### Resume Interrupted Sweep
```bash
# Start a large sweep
python parameter_sweep.py --sweep min_edge=0.02,0.04,0.06,0.08,0.10

# If interrupted, resume:
python parameter_sweep.py --sweep min_edge=0.02,0.04,0.06,0.08,0.10 --resume
```

## Interpreting Results

The `--limited` flag shows the top 10 runs ranked by Monte Carlo Z-score:

```
======================================================================
TOP 2 RUNS BY Z-SCORE
======================================================================

--- #1: Run 0001 ---

Parameters:
  min_edge: 0.04

Shuffle Test Stats (settled trades only):
  Z-Score:       -0.710
  Settled PnL:   $-20.82
  Shuffled Mean: $13.59
  Percentile:    32.0%
  Significant:   No

Trades:
  Total:    14
  YES:      6 (42.9%)
  NO:       8 (57.1%)

Equity (includes open positions):
  Starting: $500.00
  Final:    $386.92
  Net PnL:  $-113.08
```

### Understanding the Two PnL Metrics

- **Settled PnL** (in shuffle test): Only includes realized PnL from trades that have settled. This is what the Monte Carlo simulation uses.
- **Net PnL** (in equity): Final bankroll minus starting bankroll. Includes capital locked in unsettled positions.

If these differ significantly, you have open positions at the end of the backtest. The shuffle test only evaluates settled outcomes since unsettled trades have no outcome to shuffle.

### Key Statistics

- **Z-Score** > 2.0 suggests skill (p < 0.05)
- **Percentile** > 95% means your PnL beat 95% of random shuffles
- **Significant: Yes** means the result is unlikely due to luck
- With `--all_trades`, shuffled mean is typically higher since outcomes are drawn from the broader evaluated universe
