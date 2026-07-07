# BTC Pricing Engine (v2)

`core/pricing/btc_pricing_engine.py`

FIGARCH(1,d,1) + SVCJ (Kou Jump Diffusion with correlated volatility jumps) Monte Carlo simulator on hourly steps. Optional Hansen skewed-t innovations, FIGARCH long memory, and regime-conditional pricing via 3-state HMM.

## Functions

### `calculate_probabilities(strikes, hours_to_expiry, ...)`

High-level entry point. Horizon-gated, regime-weighted probability computation.

**Returns**: `Dict[float, float]` — `{strike: probability}`.

### `simulate_paths(S0, garch_params, jump_params, hours_to_expiry, ...)`

Monte Carlo path simulator. Returns `np.ndarray` of `(n_sims,)` terminal prices.

### `load_and_prep_data(hourly_csv, training_start_date=None)`

Load hourly data, compute log returns. `training_start_date` enables backtest-style time-travel truncation.

### `fit_garch_model(hourly_returns, use_naive_prior=True, filter_jumps=True)`

Fit GARCH(1,1) or FIGARCH(1,d,1) + Student-t via `arch` library. `filter_jumps=True` (default) first winsorizes detected jump-bar returns to +/- 3x local bipower sigma via `filter_jump_returns()`, so the fit sees approximately the diffusion component only (avoids double-counting jump variance against the separately simulated jump process). When `use_figarch=True`, fits `arch_model(vol='FIGARCH', p=1, q=1)` jointly estimating phi, d, beta. Falls back to GARCH on convergence failure. Returns dict of fitted parameters.

### `get_contract_probability(paths, strike_price)`

Calculate P(path >= strike) from simulated terminal prices. Supports strict_above for `>` vs `>=` semantics.

### `build_regime_jump_params(base_params, regime_label)`

Apply regime-specific scaling to jump parameters for regime-conditional pricing.

### `skewed_t_rvs(nu, lam, size, rng)`

Generate Hansen (1994) **standardized** skewed-t variates (mean 0, variance 1 by construction) via inverse-CDF sampling: draw U~Uniform(0,1), invert the standardized-t quantile within the left ((1-lambda)/2 mass) / right ((1+lambda)/2 mass) piece, then map with Hansen's a, b constants. lambda<0 => heavier left tail (negative skew). `skewed_t_scale_factor()` returns 1.0 (the variate is already standardized; no external rescale).

### `_compute_figarch_weights(d, phi, beta, trunc_k=1000)`

Compute FIGARCH(1,d,1) infinite-ARCH weights for the variance recursion. Recurrence matches the arch library: delta_1=d, lambda_1=phi-beta+d, delta_i=((i-1-d)/i)*delta_{i-1}, lambda_i=beta*lambda_{i-1}+(delta_i-phi*delta_{i-1}). Returns weights[0]=0 (no contemporaneous epsilon^2) and weights[k]=lambda_k for k>=1. fit_garch_model() fits arch_model(vol='FIGARCH', p=1, q=1) jointly estimating phi, d, beta -- Bollerslev-Mikkelsen positivity satisfied natively.

## Feature Flags

| Flag | Default | Phase | Description |
|------|---------|-------|-------------|
| `use_naive_prior` | `True` | 1.1 | Enforce mu=0 in GARCH |
| `martingale_anchor` | `False` | -- | Use exponential-cumulant jump compensator. Corrects the JUMP compensator only -- the diffusion Jensen term is NOT subtracted, so E[S_T]=S0 does NOT hold exactly (Student-t exponential moments are finite only due to the per-step return clip). Default False keeps the legacy log-mean compensator (log-mean anchored, E[log S_T] = log S0); flip only with calibration re-baseline. |
| `use_svcj` | `False` | 1.3 | Correlated volatility jumps (Eraker 2004: return + variance jump on the same Poisson count) |
| `use_skewed_t` | `False` | 1.4 | Hansen skewed-t innovations |
| `use_figarch` | `False` | 2.5 | FIGARCH(1,d,1) long-memory variance |
| `use_regime_switching` | `False` | 1.2 | HMM regime detection |
| `use_xgb_direction` | `False` | 2.3 | XGBoost directional modifier |

## Horizon Gating (Phase 1.5)

Model complexity scales automatically with time-to-expiry:

| Horizon | Model |
|---------|-------|
| T > 90 days | Naive prior (mu=0, GARCH+t); Kou return jumps retained, SVCJ/skew/FIGARCH/regime/XGB disabled |
| 30 < T <= 90 days | Naive prior + simplified (Kou jumps, no SVCJ/FIGARCH/skewed-t) |
| 7 < T <= 30 days | Intermediate (all features except skewed-t) |
| T <= 7 days | Full model (SVCJ, skewed-t, FIGARCH all enabled) |

Horizon gating is **automatic** -- it overrides feature flags for medium/long horizons.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strikes` | required | List of strike prices |
| `hours_to_expiry` | required | Hours until expiry |
| `n_sims` | 15000 | Monte Carlo paths per simulation |
| `seed` | None | RNG seed |
| `hourly_df` | None | Pre-loaded hourly data (backtest) |
| `intraday_df` | None | Pre-loaded intraday data (backtest) |
| `training_start_date` | `"2019-10-01"` | Data cutoff for structural break (Phase 0.1) |
| `regime_detector` | None | RegimeDetector instance (required if `use_regime_switching`) |
| `xgb_model` | None | DirectionalXGB instance (required if `use_xgb_direction`) |
| `regime_params` | None | Dict of regime-specific jump overrides (Phase 2.4) |
| `macro_df` | None | Macro DataFrame for XGBoost features (Phase 2.3) |

See [Pricing Engine Concept](../../concepts/pricing-engine.md) for full model specification, phase architecture, and validation test details.
