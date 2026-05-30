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

Run `python parameter_sweep.py --list-params` to see all 24 sweepable parameters.

See [Configuration](../getting-started/configuration.md) for full parameter reference.

## Monte Carlo Shuffle Tests

Two shuffle test modes:

### Expiry-Only Shuffle (Default)

Shuffles outcomes **within each expiry** among taken trades only.

### Decile-Conditioned Shuffle (`--all_trades`)

Stronger null model using **all priced contracts** as outcome pool. See [Backtesting Methodology](../concepts/backtesting-methodology.md) for details.

## Output

Each run creates a folder in `parameter_sweeps/XXXX/`:

```
parameter_sweeps/
├── 0001/
│   ├── taken_trades.csv
│   ├── montecarlo_results.csv
│   ├── equity_curve.csv
│   ├── run_config.md
│   └── logs.txt
├── 0002/
│   └── ...
```

## Interpreting Results

`--limited` shows top 10 runs ranked by Monte Carlo Z-score:

- **Z-Score** > 2.0 suggests skill (p < 0.05)
- **Percentile** > 95% means your PnL beat 95% of random shuffles
- **Significant: Yes** means the result is unlikely due to luck
- With `--all_trades`, shuffled mean is typically higher since outcomes are drawn from the broader evaluated universe

### Two PnL Metrics

- **Settled PnL** (shuffle test): Only realized PnL from settled trades
- **Net PnL** (equity): Final bankroll − starting, includes open positions
