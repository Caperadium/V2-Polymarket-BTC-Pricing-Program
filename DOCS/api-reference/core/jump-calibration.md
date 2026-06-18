# Jump Calibration

`core/pricing/jump_calibration.py`

Historical jump calibration from threshold exceedances on BTC hourly returns. Estimates Kou double-exponential parameters and SVCJ volatility jump parameters without MCMC.

## Functions

### `calibrate_jumps(hourly_csv, returns=None, detection_method="MAD", mad_multiplier=3.0, hours_per_year=8760)`

End-to-end calibration pipeline. Detects jumps → fits Kou MLE → estimates SVCJ vol jump params.

**Returns**: `JumpCalibrationResult`

### `detect_jumps_mad(returns, mad_multiplier=3.0)`

MAD-based jump detection. Returns `(jump_mask, threshold)`.

### `detect_jumps_bipower(returns, significance=0.01)`

Bipower variation jump detection. Returns `jump_mask`.

### `fit_kou_params(jump_returns, method="mle")`

MLE fit of Kou parameters. Returns `(p_crash, eta_up, eta_down, n_jumps)`.

## Classes

### `JumpCalibrationResult`

| Field | Type | Description |
|-------|------|-------------|
| `lam` | float | Annual jump intensity |
| `p_crash` | float | Probability jump is downward |
| `eta_up` | float | Positive jump size decay |
| `eta_down` | float | Negative jump size decay |
| `mu_v` | float | Mean vol jump size (hourly variance) |
| `rho_J` | float | Return-vol jump correlation |
| `lam_v` | float | Vol jump intensity |
| `n_jumps_detected` | int | Number of jumps detected |
| `n_obs` | int | Total observations |
| `jump_threshold` | float | Detection threshold value |
| `fit_converged` | bool | Whether fit succeeded |

See [Jump Calibration Concept](../../concepts/jump-calibration.md) for full methodology.
