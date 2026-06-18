# Jump Calibration

`core/pricing/jump_calibration.py`

Historical jump calibration from threshold exceedances on BTC hourly returns. Estimates Kou double-exponential parameters and SVCJ volatility jump parameters without requiring MCMC — uses MAD-based jump detection + MLE.

Based on: Teng et al. (2025), Qiao et al. (2025), Eraker et al. (2004).

## Why It Matters

The pricing engine's jump parameters (λ, p_crash, η_up, η_down) must be grounded in empirical BTC behavior. Literature defaults (Teng 2025) provide reasonable starting points, but calibration to actual BTC data captures regime shifts and improves multi-day VaR performance.

SVCJ vol jump parameters (μ_v, ρ_J) are essential for the correlated volatility jump model validated in Phase 1. Without them, the model is SVJ-equivalent and fails Basel backtests at horizons ≥ 14 days.

## Detection Methods

### MAD (Median Absolute Deviation)

```python
detect_jumps_mad(returns, mad_multiplier=3.0)
```

- Compute median of returns
- Compute MAD = median(|r_t - median|)
- Flag as jump when |r_t - median| > 3.0 × MAD

Fast, robust to outliers. Default method. MAD multiplier configurable.

### Bipower Variation (Barndorff-Nielsen & Shephard)

```python
detect_jumps_bipower(returns, significance=0.01)
```

- Less sensitive to volatility clustering than MAD
- Tests realized variance vs bipower variation ratio
- Under null of no jumps, test statistic ~ N(0,1)
- If global test rejects, identifies individual jumps via rolling std threshold

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

Estimated from variance dynamics around detected jump events:

- **μ_v**: Mean of positive variance changes at jump times (hourly variance units). Clamped to [10⁻⁶, 10⁻³].
- **ρ_J**: Correlation coefficient between return jump sizes and contemporaneous vol changes. Clamped to [-0.5, 0.5].

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
    rho_J: float             # Return-vol jump correlation
    lam_v: float             # Vol jump intensity (defaults to lam)
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
