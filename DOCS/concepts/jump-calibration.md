# Jump Calibration

`core/pricing/jump_calibration.py`

Historical jump calibration from threshold exceedances on BTC hourly returns. Estimates Kou double-exponential parameters and SVCJ volatility jump parameters without requiring MCMC -- uses Lee-Mykland bipower jump detection (pipeline default) or MAD detection, plus MLE.

Based on: Teng et al. (2025), Qiao et al. (2025), Eraker et al. (2004).

## Why It Matters

The pricing engine's jump parameters (λ, p_crash, η_up, η_down) must be grounded in empirical BTC behavior. Literature defaults (Teng 2025) provide reasonable starting points, but calibration to actual BTC data captures regime shifts and improves multi-day VaR performance.

SVCJ vol jump parameters (μ_v, ρ_J, rho_j_slope) are essential for the correlated volatility jump model validated in Phase 1. Without them, the model is SVJ-equivalent and fails Basel backtests at horizons ≥ 14 days.

## Detection Methods

### MAD (Median Absolute Deviation)

```python
detect_jumps_mad(returns, mad_multiplier=3.0)
```

- Compute median of returns
- Compute MAD = median(|r_t - median|)
- Flag as jump when |r_t - median| > 3.0 × MAD

Fast, robust to outliers. Kept as an alternative (`detection_method="MAD"`); bipower is the pipeline default. MAD multiplier configurable.

### Bipower Variation (Barndorff-Nielsen & Shephard)

```python
detect_jumps_bipower(returns, significance=0.01)
```

- Less sensitive to volatility clustering than MAD
- Lee-Mykland (2008) local test: each return is compared against a local bipower sigma estimated over a trailing window that ends at t-1 -- the contemporaneous return is excluded, so a jump cannot inflate its own detection threshold
- `return_sigma=True` additionally returns the local sigma array (used by the SVCJ vol-jump estimator and the jump-filtered GARCH fit)

## Parameter Estimation

### Kou Double-Exponential Parameters

`fit_kou_params(jump_returns)` uses MLE:

| Parameter | Method |
|-----------|--------|
| `p_crash` | Empirical: n_down / (n_up + n_down) |
| `eta_up` | MLE for exponential: 1 / mean(positive jumps) |
| `eta_down` | MLE for exponential: 1 / mean(negative jump magnitudes) |
| `lam` (annual) | (n_jumps / n_obs) × 365 × 24 |

Parameters clamped to [5, 200] for eta, [5, 100] for lam.

### SVCJ Vol Jump Parameters

Estimated from variance dynamics around detected jump events (shared helper `_estimate_vol_jump_params`). The rolling-variance INPUT replaces each jump bar's squared return with the local diffusion variance (bipower sigma_local^2, or the median non-jump squared return as fallback), removing the mechanical J^2/window inflation of the post-jump window:

- **μ_v**: Censored-at-zero mean of variance changes over ALL jump events (zeros included; conservative, biased low -- the old mean-of-positive-deltas-only was biased high). Hourly variance units, clamped to [10⁻⁶, 10⁻³].
- **ρ_J**: Pearson correlation between return jump sizes and vol changes. Clamped to [-0.5, 0.5]. **Diagnostics/reporting only** -- no longer used in the return equation.
- **rho_j_slope**: OLS slope of jump return on vol-jump delta (units: return per unit variance jump), sanity-capped so `|rho_j_slope| * mu_v <= 0.5 * mean(|jump_returns|)`; 0.0 when var(dv)=0 or fewer than 10 events. This is the coefficient the SVCJ return-jump adjustment in `simulate_paths` actually uses (default 0.0 = term off).

### Literature Fallback

When too few jumps detected (< 10), reverts to Teng (2025) reference values:

```python
JumpCalibrationResult(
    lam=25.0, p_crash=0.6, eta_up=50.0, eta_down=25.0,
    mu_v=0.000025, rho_J=-0.08, lam_v=25.0,
)
```

## JumpCalibrationResult

```python
@dataclass
class JumpCalibrationResult:
    lam: float              # Jump intensity (jumps per year)
    p_crash: float           # Probability jump is downward
    eta_up: float            # Positive jump size decay (1/mean)
    eta_down: float          # Negative jump size decay (1/mean)
    mu_v: float              # Mean volatility jump size (hourly variance units)
    rho_J: float             # Return-vol jump Pearson correlation (diagnostics only)
    lam_v: float             # Vol jump intensity (defaults to lam)
    rho_j_slope: float = 0.0 # OLS return-per-variance-jump slope used by SVCJ
                             # return-jump adjustment (sanity-capped; 0.0 = off)
    n_jumps_detected: int    # Number of jumps detected
    n_obs: int               # Total observations
    jump_threshold: float    # Detection threshold value
    fit_converged: bool      # Whether MLE converged
```

## Usage

### CLI

```bash
python core/pricing/jump_calibration.py --input DATA/btc_hourly.csv --method MAD --mad-mult 3.0
```

### Programmatic

```python
from core.pricing.jump_calibration import calibrate_jumps

result = calibrate_jumps("DATA/btc_hourly.csv", detection_method="MAD")
print(f"Annual lambda: {result.lam:.1f}")
print(f"Crash prob: {result.p_crash:.3f}")
print(f"rho_J: {result.rho_J:.4f}")
```
