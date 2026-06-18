# Testing

## Running Tests

### Built-in Pricing Engine Validation

```bash
python core/pricing/btc_pricing_engine.py
```

Eight tests validate core model components:

1. **Multi-Jump Aggregation** — 99th percentile of multi-jump ≥ 1.2× single-jump
2. **Fractional dt Variance** — Variance unchanged, prices moved
3. **Dynamic Drift Clamping** — Per-path clamping produces vector output
4. **Variance Consistency** — Empirical/model variance ratio within ±15%
5. **Naive Prior** — Zero-drift paths show smaller deviation than fitted-drift
6. **SVCJ** — Volatility jumps add measurable variance vs plain SVJ
7. **FIGARCH Weights** — Binomial expansion weights decay correctly
8. **Skewed-t** — λ=-0.3 → negative skew, λ=+0.3 → positive skew, λ=0 → symmetric

### Standalone Module Tests

Each new module has a CLI that can be run for smoke-testing:

```bash
# Jump calibration (self-validates parameter ranges)
python core/pricing/jump_calibration.py --input DATA/btc_hourly.csv

# Regime detection (checks HMM convergence and state labeling)
python core/pricing/regime_detector.py --input DATA/btc_hourly.csv

# Basel backtest (runs multi-horizon VaR on historical data)
python core/validation/basel_backtest.py --input DATA/btc_hourly.csv

# Macro fetcher (downloads to DATA/macro_daily.csv)
python core/data/macro_fetcher.py --period 5y

# Directional XGBoost (trains classifier if sufficient data)
python core/pricing/directional_xgb.py --btc DATA/btc_hourly.csv
```

### Pytest Suite

```bash
python -m pytest tests/ -v
```

Test files in the `tests/` directory cover strategy refactoring, backtest inversion, dashboard verification, and console logic.

## When Tests Fail

### Pricing engine tests fail after code changes

- Check that new parameters are appended to function signatures (not inserted mid-signature)
- Verify feature flags default to `False` (except `use_naive_prior=True`)
- Ensure RNG uses `np.random.default_rng(seed)`, never `np.random.seed()`

### Jump calibration returns default values

- This is expected with < 10 detected jumps — not a failure
- Literature defaults from Teng (2025) are reasonable
- Test with ≥ 1 year of hourly data for reliable detection

### HMM reports non-convergence

- Non-critical — model still produces usable state assignments
- Expected behavior with noisy financial returns
- Verify dominant state is `sideways` and weights are sensible
