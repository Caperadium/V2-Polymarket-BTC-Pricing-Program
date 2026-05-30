# Pricing Engine

`core/pricing/btc_pricing_engine.py`

The pricing engine computes **probability that BTC ends above a given strike price** at contract expiry. It uses a GARCH(1,1) + Student-t + Kou Double Exponential Jump Diffusion Monte Carlo simulator.

## Model Components

### 1. GARCH(1,1) with Student-t Errors

Daily log returns are scaled ×100 for numerical stability, then fit with:

$$\sigma_t^2 = \omega + \alpha \cdot \epsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$$

The `arch` library provides the fitting via `arch_model(returns, vol='Garch', p=1, q=1, dist='t')`.

**Output parameters**: `omega`, `alpha`, `beta`, `nu` (degrees of freedom), `mu` (mean), `last_variance`.

### 2. Structural Mean Drift

The model uses the long-term fitted GARCH mean (`mu`) as drift, with per-path clamping to prevent extreme drifts:

```python
mu_clamped = np.clip(mu, -0.25 * sigma_day_step, 0.25 * sigma_day_step)
```

This prevents extreme drifts from dominating low-volatility paths.

### 3. Kou Double Exponential Jump Diffusion

Jumps follow a compound Poisson process with asymmetric exponential magnitudes:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LAMBDA` | 25.0 | Annual jump intensity |
| `CRASH_PROB` | 0.6 | Probability a jump is downward |
| `ETA_UP` | 50.0 | 1/mean upward jump size |
| `ETA_DOWN` | 25.0 | 1/mean downward jump size |

**Multi-jump aggregation**: Multiple jumps per day are aggregated using Gamma-distributed magnitudes with explicit masking (no reliance on Gamma(0, ...) = 0 behavior).

**Jump drift correction**: The expected jump drift is subtracted from the structural drift:

```python
expected_jump_drift = lam_daily * ((1 - p_crash)/eta_up - p_crash/eta_down)
```

### 4. Fractional Day Handling

When `days_to_expiry` is not an integer, the last step uses fractional `dt`. GARCH variance updates ONLY on full-day steps to preserve the variance recursion.

## API

### `calculate_probabilities(strikes, days_to_expiry, ...)`

High-level entry point. Returns `{strike: probability}` dict.

```python
from core.pricing.btc_pricing_engine import calculate_probabilities

probs = calculate_probabilities(
    strikes=[65000, 70000, 75000],
    days_to_expiry=3.5,
    n_sims=50000,
)
# returns: {65000: 0.723, 70000: 0.561, 75000: 0.384}
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strikes` | (required) | List of strike prices |
| `days_to_expiry` | (required) | Days until contract expiry (supports fractional) |
| `n_sims` | 50000 | Number of Monte Carlo paths |
| `seed` | None | RNG seed for reproducibility |
| `daily_df` | None | Optional DataFrame of daily prices (for backtesting) |
| `intraday_df` | None | Optional DataFrame of intraday prices (for backtesting) |
| `jump_params` | None | Optional dict overriding jump parameters |

### `simulate_paths(S0, garch_params, jump_params, days_to_expiry, ...)`

Low-level path simulator. Returns array of `(n_sims,)` terminal prices.

### `get_contract_probability(paths, strike_price)`

Calculate `P(path ≥ strike)` from simulated terminal prices.

## Validation

Run built-in tests:

```bash
python core/pricing/btc_pricing_engine.py
```

Four tests validate:

1. **Multi-Jump Aggregation** — 99th percentile of multi-jump ≥ 1.2× single-jump
2. **Fractional dt Variance** — Variance unchanged, prices moved
3. **Dynamic Drift Clamping** — Per-path clamping produces vector output
4. **Variance Consistency** — Empirical/model variance ratio within ±15%

## Dependencies

- `pandas`, `numpy` — data handling
- `arch` — GARCH fitting
- `scipy.stats.t` — Student-t distribution (for consistency checks only)
