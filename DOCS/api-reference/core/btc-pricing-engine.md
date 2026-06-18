# BTC Pricing Engine (v2)

`core/pricing/btc_pricing_engine.py`

GARCH(1,1) + SVCJ (Kou Jump Diffusion with correlated volatility jumps) Monte Carlo simulator on hourly steps. Optional Hansen skewed-t innovations, FIGARCH long memory, and regime-conditional pricing via 3-state HMM.

## Functions

### `calculate_probabilities(strikes, hours_to_expiry, ...)`

High-level entry point. Horizon-gated, regime-weighted probability computation.

**Returns**: `Dict[float, float]` — `{strike: probability}`.

### `simulate_paths(S0, garch_params, jump_params, hours_to_expiry, ...)`

Monte Carlo path simulator. Returns `np.ndarray` of `(n_sims,)` terminal prices.

### `load_and_prep_data(hourly_csv, training_start_date=None)`

Load hourly data, compute log returns. `training_start_date` enables backtest-style time-travel truncation.

### `fit_garch_model(hourly_returns, use_naive_prior=True)`

Fit GARCH(1,1) + Student-t via `arch` library. Returns dict of fitted parameters.

### `get_contract_probability(paths, strike_price)`

Calculate P(path ≥ strike) from simulated terminal prices. Supports strict_above for `>` vs `>=` semantics.

### `build_regime_jump_params(base_params, regime_label)`

Apply regime-specific scaling to jump parameters for regime-conditional pricing.

### `skewed_t_rvs(nu, lam, size, rng)`

Generate Hansen (1994) skewed-t random variates. Inverse-transform sampling from standard-t base.

### `_compute_figarch_weights(d, trunc_k=100)`

Compute FIGARCH binomial expansion weights for long-memory variance recursion.

## Feature Flags

| Flag | Default | Phase | Description |
|------|---------|-------|-------------|
| `use_naive_prior` | `True` | 1.1 | Enforce μ=0 in GARCH |
| `use_svcj` | `False` | 1.3 | Correlated volatility jumps |
| `use_skewed_t` | `False` | 1.4 | Hansen skewed-t innovations |
| `use_figarch` | `False` | 2.5 | FIGARCH long-memory variance |
| `use_regime_switching` | `False` | 1.2 | HMM regime detection |
| `use_xgb_direction` | `False` | 2.3 | XGBoost directional modifier |

## Horizon Gating (Phase 1.5)

Model complexity scales automatically with time-to-expiry:

| Horizon | Model |
|---------|-------|
| T > 90 days | Naive prior only (μ=0, GARCH+t, no jumps) |
| 30 < T ≤ 90 days | Naive prior + simplified (Kou jumps, no SVCJ/FIGARCH/skewed-t) |
| 7 < T ≤ 30 days | Intermediate (all features except skewed-t) |
| T ≤ 7 days | Full model (SVCJ, skewed-t, FIGARCH all enabled) |

Horizon gating is **automatic** — it overrides feature flags for medium/long horizons.

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
