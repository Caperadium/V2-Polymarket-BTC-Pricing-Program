# Jump Calibration

`core/pricing/jump_calibration.py`

Historical jump calibration from threshold exceedances on BTC hourly returns. Estimates Kou double-exponential parameters and SVCJ volatility jump parameters without MCMC.

## Functions

### `calibrate_jumps(hourly_csv, returns=None, detection_method="bipower", mad_multiplier=3.0, hours_per_year=8760)`

End-to-end calibration pipeline. Detects jumps → fits Kou MLE → estimates SVCJ vol jump params (via the shared `_estimate_vol_jump_params` helper).

**Returns**: `JumpCalibrationResult`

### `detect_jumps_mad(returns, mad_multiplier=3.0)`

MAD-based jump detection. Returns `(jump_mask, threshold)`.

### `detect_jumps_bipower(returns, significance=0.01, window=78, return_sigma=False)`

Lee-Mykland (2008) local bipower jump detection. The local sigma window ends at t-1 (the contemporaneous return is excluded, so a jump cannot inflate its own detection threshold). Returns `jump_mask`, or `(jump_mask, sigma_local)` when `return_sigma=True`.

### `fit_kou_params(jump_returns, method="mle")`

MLE fit of Kou parameters. Returns `(p_crash, eta_up, eta_down, n_jumps_total)` -- the 4th element is a jump COUNT, not an annual lambda (callers annualize it themselves).

## Classes

### `JumpCalibrationResult`

| Field | Type | Description |
|-------|------|-------------|
| `lam` | float | Annual jump intensity |
| `p_crash` | float | Probability jump is downward |
| `eta_up` | float | Positive jump size decay |
| `eta_down` | float | Negative jump size decay |
| `mu_v` | float | Mean vol jump size (hourly variance); censored-at-zero mean over ALL jump events, with jump-bar squares replaced by local diffusion variance in the rolling-variance input |
| `rho_J` | float | Return-vol jump Pearson correlation (diagnostics/reporting only) |
| `rho_j_slope` | float | OLS slope of jump return on vol-jump delta (return per unit variance jump, sanity-capped; default 0.0 = term off). Used by the SVCJ return-jump adjustment in `simulate_paths` |
| `lam_v` | float | Vol jump intensity |
| `n_jumps_detected` | int | Number of jumps detected |
| `n_obs` | int | Total observations |
| `jump_threshold` | float | Detection threshold value |
| `fit_converged` | bool | Whether fit succeeded |

See [Jump Calibration Concept](../../concepts/jump-calibration.md) for full methodology.
