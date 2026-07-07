# Pricing Engine (v2)

`core/pricing/btc_pricing_engine.py`

The pricing engine computes **probability that BTC ends above a given strike price** at contract expiry. It uses a FIGARCH(1,d,1) or GARCH(1,1) + SVCJ (Kou Double Exponential Jump Diffusion with correlated volatility jumps) Monte Carlo simulator running on **hourly** steps, with optional Hansen skewed-t innovations and FIGARCH(1,d,1) long-memory volatility.

## Phase Structure

The pricing engine is built incrementally across seven phases. Each phase is a toggleable feature flag:

| Phase | Feature | Flag | Status |
|-------|---------|------|--------|
| 0 | Base GARCH(1,1) + Student-t | (always on) | Core |
| 0.1 | Structural break filter | `training_start_date="2019-10-01"` | Default on |
| 0.5 | Jump calibration cache | `load_calibrated_jumps()` | Cache-based |
| 1.1 | Naive prior (μ=0) | `use_naive_prior=True` | Default on |
| 1.2 | HMM regime detection | `use_regime_switching=True` | Opt-in |
| 1.3 | SVCJ correlated jumps | `use_svcj=True` | Opt-in |
| 1.4 | Hansen skewed-t | `use_skewed_t=True` | Opt-in |
| 1.5 | Horizon gating | (automatic) | Always active |
| 2.3 | Directional XGBoost | `use_xgb_direction=True` | Opt-in |
| 2.4 | Regime-conditional jumps | `regime_params` dict | With HMM |
| 2.5 | FIGARCH(1,d,1) long memory [Baillie, Bollerslev & Mikkelsen 1996] | `use_figarch=True` | Opt-in |
| 2.6 | Regime-vol gate interaction | `vol_gate_regime` param | With vol gate |

## Model Components

### 0.1 Structural Break Filter

Per Pakstaite et al. (2025), BTC return characteristics underwent a structural break around 2019–2020 coinciding with institutional adoption. Pre-2019 data exhibits different volatility clustering and jump behavior that degrades model fit on post-2019 data.

The filter is applied via `training_start_date`:

```python
# Default: "2019-10-01" — fits GARCH on post-break data only
calculate_probabilities(strikes=[70000], hours_to_expiry=72)
```

- Filters `btc_hourly.csv` to rows with `date >= training_start_date`
- Falls back to full data if filtered data has < 500 rows
- Configurable for backtesting (set earlier to simulate pre-break conditions)

### 1. GARCH(1,1) with Student-t Errors

Hourly log returns from `DATA/btc_hourly.csv` (5yr Binance 1h klines) are scaled ×100 for numerical stability, then fit with:

$$\sigma_t^2 = \omega + \alpha \cdot \epsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$$

The `arch` library provides the fitting via `arch_model(returns, vol='Garch', p=1, q=1, dist='t')`.

**Jump-filtered fit (default)**: `fit_garch_model(..., filter_jumps=True)` first winsorizes detected jump-bar returns to +/- 3x the local bipower sigma (`filter_jump_returns`), so the GARCH/FIGARCH fit sees approximately the diffusion component only. Without this, total variance double-counts the jump contribution (the simulator adds a calibrated jump process on top). `filter_jumps=False` is kept for A/B comparison.

**Output parameters**: `omega`, `alpha`, `beta`, `nu` (degrees of freedom), `mu` (mean), `last_variance` — all in **hourly log-return units**.

### 2. Naive Prior (Default On)

Per Baquero/Shelton (2024/2026), OOS evidence shows zero-drift GARCH outperforms fitted-drift:

- **On** (`use_naive_prior=True`, default): Enforces μ=0 in GARCH fitting. Prices drift only through jump compensation.
- **Off** (`use_naive_prior=False`): Uses fitted GARCH mean with per-path clamping to ±0.25 × path-specific sigma_hourly.

The jump drift correction (expected jump drift subtracted from drift) makes the default distribution log-mean anchored (E[log S_T] = log S0); the median coincides only for a symmetric log distribution. This is the PHYSICAL measure, NOT risk-neutral. `martingale_anchor=True` corrects the JUMP compensator only -- the diffusion Jensen term (~ +sigma^2/2 per step, roughly +1% at 30 days at 50% annualized vol) is NOT subtracted, and Student-t exponential moments are finite only due to the per-step return clip.

### 3. Kou Double Exponential Jump Diffusion

Jumps follow a compound Poisson process with asymmetric exponential magnitudes:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LAMBDA` | 25.0 | Annual jump intensity |
| `CRASH_PROB` | 0.6 | Probability a jump is downward |
| `ETA_UP` | 50.0 | 1/mean upward jump size |
| `ETA_DOWN` | 25.0 | 1/mean downward jump size |

**Multi-jump aggregation per hour**: Jump intensity is scaled `lam_hourly = LAMBDA / (365 × 24)`. Multiple jumps per hour are aggregated using Gamma-distributed magnitudes with explicit masking.

**Jump drift correction**: The expected jump drift (hourly) is subtracted from the structural drift:

```python
expected_jump_drift = lam_hourly * ((1 - p_crash)/eta_up - p_crash/eta_down)
```

### 4. SVCJ — Correlated Volatility Jumps (Optional)

Per Eraker et al. (2004) and Teng et al. (2025). When enabled (`use_svcj=True`):

- **Shared Poisson driver**: Same Poisson process drives both return jumps AND volatility jumps
- **Return-vol coupling** (`rho_j_slope`): Jump sizes are coupled -- `jump_sizes += rho_j_slope * vol_jump_mag`, where `rho_j_slope` is the calibrated OLS slope of jump return on vol-jump delta (units: return per unit variance jump, sanity-capped; default 0.0 = term off). `rho_J` (the Pearson correlation) is diagnostics/reporting only -- used directly as a slope it was ~5 orders of magnitude too small to matter
- **Vol jump magnitudes**: Drawn from exponential distribution with mean `mu_v`
- **Calibration**: Parameters estimated from historical data via `core/pricing/jump_calibration.py`

SVCJ is critical for multi-day VaR — pure SVJ (no vol jumps) fails Basel backtests at h≥14.

### 5. Hansen Skewed-t Innovations (Optional)

When enabled (`use_skewed_t=True`), student-t innovations are replaced with Hansen (1994) skewed-t:

- **Asymmetry parameter** `lam` ∈ (-1, 1): Negative → left skew (crashes), positive → right skew
- **Inverse-transform sampling**: Standard-t draws mapped through the Hansen inverse CDF
- **Variance correction**: Outputs normalized to unit variance via `sqrt(1 + 3·lam²)`
- **Empirical calibration**: BTC hourly returns show negative skew (~-0.3 under normal conditions)

```python
# lam=-0.3 → negative skew (left tail heavier, captures crash asymmetry)
# lam=+0.3 → positive skew (right tail heavier)
# lam=0   → symmetric (reduces to standard Student-t)
```

### 6. FIGARCH(1,d,1) Long Memory (Optional)

When enabled (`use_figarch=True`), replaces the short-memory GARCH(1,1) variance recursion with FIGARCH(1,d,1) per Baillie, Bollerslev & Mikkelsen (1996):

$$\sigma_t^2 = \frac{\omega}{1-\beta} + \sum_{k=1}^{\infty} \lambda_k \cdot \epsilon_{t-k}^2$$

- **Jointly estimated**: phi, d, beta fitted via `arch_model(vol='FIGARCH', p=1, q=1)` on hourly BTC returns
- **ARCH(infinity) weights**: lambda_1=phi-beta+d, lambda_i=beta*lambda_{i-1}+(delta_i-phi*delta_{i-1}) where delta_i are the binomial coefficients of (1-L)^d
- **Persistence**: FIGARCH captures long-range dependence in BTC volatility (hyperbolic decay vs exponential in GARCH)
- **Positivity**: Joint estimation satisfies Bollerslev-Mikkelsen non-negativity constraints natively

### 9. Directional XGBoost Modifier (Phase 2.3)

When enabled (`use_xgb_direction=True`), an XGBoost classifier provides a directional P(up) signal that is blended with the base MC probability:

```
p_final = 0.7 × p_mc + 0.3 × p_xgb
```

The 30% blend weight is from Shelton (2024) — OOS evidence that individual directional predictors are weak.

- Uses `DirectionalXGB` instance from `core/pricing/directional_xgb.py`
- Trained on BTC daily returns + macro features (Gold, DXY, VIX, SPX)
- Feature matrix: multi-window volatility, momentum, drawdown, vol-of-vol, macro correlations
- Falls back to 0.5 (neutral, no tilt) if untrained or prediction fails
- Currently **not calibrated on this project's data** — weight and hyperparameters use literature defaults

See [Directional XGBoost](directional-xgb.md) for full model specification.

### 10. Regime-Conditional Jump Parameters (Phase 2.4)

When HMM regime detection is active, jump parameters are scaled per regime state:

```python
regime_params = {
    "bear":     {"lambda": LAMBDA * 1.5, "p_crash": CRASH_PROB * 1.3},
    "sideways": {"lambda": LAMBDA,       "p_crash": CRASH_PROB},
    "bull":     {"lambda": LAMBDA * 0.7, "p_crash": CRASH_PROB * 0.7},
}
```

- **Bear**: 50% more frequent jumps, 30% higher crash probability — captures downside acceleration
- **Bull**: 30% fewer jumps, 30% lower crash probability — captures upward drift stability
- **Sideways**: Default parameters

Applied via `build_regime_jump_params()` and passed to each regime's independent simulation.

### 11. Regime-Vol Gate Interaction (Phase 2.6)

The HMM regime detector and volatility gate are independent risk signals that interact at the simulation level:

| Vol Gate | Effect on Jump Parameters |
|----------|--------------------------|
| `normal` | No change |
| `high` | λ × 1.2, μ_v × 1.3 |
| `extreme` | λ × 1.5, μ_v × 2.0 |

Extreme vol gate **always overrides** — blocks entries regardless of HMM regime. A bull HMM regime with extreme vol gate = no entries (hard gate wins). This prevents overpaying for upside tails during volatility events.

### 7. Horizon Gating

Model complexity scales with time-to-expiry to avoid over-parameterization of long-dated contracts:

| Horizon | Model Configuration |
|---------|--------------------|
| T > 90 days | Naive prior (μ=0, GARCH+Student-t); Kou return jumps retained, SVCJ/skew/FIGARCH/regime/XGB disabled |
| 30 < T ≤ 90 days | Naive prior + simplified (GARCH+t, Kou jumps, no SVCJ/FIGARCH/skewed-t) |
| 7 < T ≤ 30 days | Intermediate (all features enabled except skewed-t) |
| T ≤ 7 days | **Full model** (SVCJ, skewed-t, FIGARCH all enabled) |

Short-dated contracts see the most complex dynamics because jump effects dominate and volatility persistence matters most.

### 8. Regime-Conditional Pricing

Uses 3-state HMM regime detection from `core/pricing/regime_detector.py`:

1. Run 3 **independent** Monte Carlo simulations (bear, sideways, bull)
2. Each simulation uses regime-scaled jump parameters via `build_regime_jump_params()`:
    - **Bear**: λ × 1.5, p_crash × 1.3 (more frequent, more directional jumps)
    - **Bull**: λ × 0.7, p_crash × 0.7 (fewer, less directional jumps)
    - **Sideways**: Default parameters unchanged
3. Weight terminal price distributions by HMM posterior:

```python
P(S_T ≥ K) = Σ_{r ∈ {bear,sideways,bull}} w_r · P_r(S_T ≥ K)
```

**Post-hoc weighting** (not intra-path switching) avoids path-continuity issues and simplifies implementation.

### 9. Fractional Hour Handling

When `hours_to_expiry` is not an integer, the last step uses fractional `dt`. GARCH variance updates ONLY on full-hour steps to preserve the variance recursion. A 30-day contract generates 720 hourly steps.

## Feature Flags

All new features are backward-compatible via flags defaulting to `False`, except `use_naive_prior=True`:

| Flag | Default | Phase | Description |
|------|---------|-------|-------------|
| `use_naive_prior` | `True` | 1.1 | Enforce μ=0 in GARCH fitting |
| `use_regime_switching` | `False` | 1.2 | Enable HMM regime detection |
| `use_svcj` | `False` | 1.3 | Enable correlated volatility jumps |
| `use_skewed_t` | `False` | 1.4 | Use Hansen skewed-t innovations |
| `use_figarch` | `False` | 2.5 | Use FIGARCH(1,d,1) long-memory variance |
| `use_xgb_direction` | `False` | 2.3 | Use XGBoost directional modifier |

Non-boolean parameters:

| Parameter | Default | Phase | Description |
|-----------|---------|-------|-------------|
| `training_start_date` | `"2019-10-01"` | 0.1 | Structural break cutoff |
| `regime_params` | `None` | 2.4 | Regime-specific jump scaling |
| `vol_gate_regime` | `"normal"` | 2.6 | Vol gate interaction level |
| `xgb_model` | `None` | 2.3 | Pre-trained DirectionalXGB instance |

New parameters are appended to function signatures — existing callers work without changes.

## API

### `calculate_probabilities(strikes, hours_to_expiry, ...)`

High-level entry point. Returns `{strike: probability}` dict with regime-weighted, horizon-gated results.

```python
from core.pricing.btc_pricing_engine import calculate_probabilities

probs = calculate_probabilities(
    strikes=[65000, 70000, 75000],
    hours_to_expiry=84,  # 3.5 days
    n_sims=15000,
)
# returns: {65000: 0.723, 70000: 0.561, 75000: 0.384}
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strikes` | (required) | List of strike prices |
| `hours_to_expiry` | (required) | Hours until contract expiry (supports fractional) |
| `n_sims` | 15000 | Number of Monte Carlo paths |
| `seed` | None | RNG seed for reproducibility |
| `hourly_df` | None | Optional DataFrame of hourly prices (for backtesting) |
| `intraday_df` | None | Optional DataFrame of intraday prices (for backtesting) |
| `hourly_csv` | `"DATA/btc_hourly.csv"` | Path to hourly BTC data |
| `jump_params` | None | Optional dict overriding jump parameters |
| `use_naive_prior` | `True` | Enforce μ=0 in GARCH fitting |
| `use_svcj` | `False` | Enable SVCJ volatility jumps |
| `use_skewed_t` | `False` | Use Hansen skewed-t innovations |
| `use_figarch` | `False` | Use FIGARCH long-memory variance |
| `training_start_date` | None | Cutoff date for training data (backtest time-travel) |

### `simulate_paths(S0, garch_params, jump_params, hours_to_expiry, ...)`

Low-level path simulator. Returns array of `(n_sims,)` terminal prices. Accepts all feature flags plus `regime_jump_params`, `regime_label`, and `vol_gate_regime`.

### `get_contract_probability(paths, strike_price)`

Calculate `P(path ≥ strike)` from simulated terminal prices.

### `build_regime_jump_params(base_params, regime_label)`

Apply regime-specific scaling factors to jump parameters. Returns dict with scaled `lam`, `p_crash`.

## Validation

Run built-in tests:

```bash
python core/pricing/btc_pricing_engine.py
```

Eight tests validate:

1. **Multi-Jump Aggregation** — 99th percentile of multi-jump ≥ 1.2× single-jump
2. **Fractional dt Variance** — Variance unchanged, prices moved
3. **Dynamic Drift Clamping** — Per-path clamping produces vector output
4. **Variance Consistency** — Empirical/model variance ratio within ±15%
5. **Naive Prior** — Zero-drift paths show smaller deviation than fitted-drift paths
6. **SVCJ** — Volatility jumps add measurable variance vs plain SVJ
7. **FIGARCH Weights** — ARCH(infinity) weights match arch library reference, weights[0]=0, lambda_1>0 (B-M positivity satisfied)
8. **Skewed-t** — λ=-0.3 → negative skew, λ=+0.3 → positive skew, λ=0 → symmetric

## Data Dependencies

- `DATA/btc_hourly.csv` — 5yr of Binance 1h klines (`date,close`) for GARCH fitting
- `DATA/btc_intraday_1m.csv` — ~3 months of 1m klines for latest price mark (S0)
- `DATA/btc_daily.csv` — Daily resampled prices for regime detection
- `pandas`, `numpy` — data handling
- `arch` — GARCH fitting
- `scipy.stats.t` — Student-t distribution (for consistency checks and skewed-t base)
- `hmmlearn` — GaussianHMM for regime detection

## Related Modules

- `core/pricing/jump_calibration.py` — MAD-based jump parameter estimation from historical data
- `core/pricing/regime_detector.py` — 3-state HMM regime classification
- `core/pricing/directional_xgb.py` — XGBoost directional classifier (P(up) modifier)
- `core/pricing/fit_probability_curves.py` — Logistic curve fitting + logit-shift calibration
- `core/validation/basel_backtest.py` — Kupiec POF VaR backtest + expected shortfall
- `core/validation/calibration_metrics.py` — Brier score, ECE, reliability diagrams
- `core/data/macro_fetcher.py` — Macro feature data (Gold, DXY, VIX, SPX)
- `core/strategy/vol_gate.py` — Volatility gate (feeds Phase 2.6 interaction)
