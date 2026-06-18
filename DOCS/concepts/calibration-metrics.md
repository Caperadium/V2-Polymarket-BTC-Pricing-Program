# Calibration Metrics

`core/validation/calibration_metrics.py`

Calibration diagnostics for probability forecasts: Brier score, reliability diagrams, and Expected Calibration Error (ECE). Compares model probabilities against realized binary outcomes to assess forecast quality.

## Why It Matters

A pricing model can produce probabilities that look reasonable but are systematically miscalibrated — e.g., when the model says 70%, the event only happens 55% of the time. Calibration metrics quantify this gap. Well-calibrated probabilities are essential for:

- **Kelly sizing**: Miscalibrated probabilities → wrong bet sizes → worse risk-adjusted returns
- **Edge estimation**: `model_prob - market_price` is only meaningful if model probabilities are calibrated
- **Risk management**: Overconfident forecasts lead to excessive position sizes

## Metrics

### Brier Score

Mean squared error between forecast probabilities and binary outcomes:

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2$$

- Range: [0, 1]. Lower is better.
- Baseline: 0.25 for a coin-flip (uniform 0.5) forecast
- Decomposes into reliability + resolution + uncertainty components

```python
from core.validation.calibration_metrics import brier_score
bs = brier_score(p_model, outcomes)
```

### Reliability Diagram

Bins forecasts into deciles (or configurable bins), then compares mean forecast against observed frequency in each bin:

| Bin | Mean Forecast | Observed Frequency | N |
|-----|--------------|-------------------|----|
| 0.0–0.1 | 0.07 | 0.09 | 45 |
| 0.1–0.2 | 0.15 | 0.12 | 82 |
| ... | ... | ... | ... |

A perfectly calibrated model has `mean_forecast ≈ observed_freq` in every bin (points on the diagonal).

```python
from core.validation.calibration_metrics import reliability_bins
bins_df = reliability_bins(p_model, outcomes, n_bins=10)
```

### Expected Calibration Error (ECE)

Weighted average of absolute calibration errors across bins:

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \text{mean\_forecast}_b - \text{observed\_freq}_b \right|$$

- Range: [0, 1]. Lower is better.
- ECE < 0.05: well-calibrated
- ECE 0.05–0.10: moderate miscalibration
- ECE > 0.10: poor calibration

```python
from core.validation.calibration_metrics import ece_score
ece = ece_score(p_model, outcomes, n_bins=10)
```

## Calibration Report

`run_calibration_report()` runs all metrics at once and returns a `CalibrationReport` dataclass:

```python
from core.validation.calibration_metrics import run_calibration_report

report = run_calibration_report(
    "fitted_batch_results/20250615_120000/batch_with_fits.csv",
    prob_col="p_model_fit",
    outcome_col="outcome",
)

print(f"Brier: {report.brier:.4f}")
print(f"ECE: {report.ece:.4f}")
print(f"Calibration bias: {report.calibration_bias:+.4f}")
print(report.bins)
```

| Field | Type | Description |
|-------|------|-------------|
| `brier` | float | Brier score |
| `ece` | float | Expected Calibration Error |
| `n_obs` | int | Number of observations |
| `bins` | DataFrame | Reliability diagram bins |
| `mean_forecast` | float | Average model probability |
| `mean_outcome` | float | Average observed frequency |
| `calibration_bias` | float | `mean_forecast - mean_outcome` (positive = overconfident) |

### Column Detection

The report runner auto-detects columns with flexible fallback chains:

- **Probability**: `p_model_fit` → `p_real_mc` → `model_probability`
- **Outcome**: `outcome` → `resolved` → `settled` → `result` → `actual`

## CLI

```bash
python core/validation/calibration_metrics.py fitted_batch_results/20250615_120000/batch_with_fits.csv

# With custom columns and bins
python core/validation/calibration_metrics.py batch.csv --prob-col p_real_mc --outcome-col resolved --n-bins 20
```

## Interpretation

| Signal | Meaning | Action |
|--------|---------|--------|
| Brier > 0.25 | Worse than coin flip | Investigate model; possible overfitting |
| ECE > 0.10 | Poor calibration | Recalibrate curves or adjust blend weights |
| Bias > 0.05 | Overconfident (forecast too high) | Apply Platt scaling or isotonic regression |
| Bias < -0.05 | Underconfident (forecast too low) | Check for censored training data |
| S-shaped reliability curve | Variance underestimation | Increase distribution tail weight |

See [Probability Fitting](probability-fitting.md) for calibration adjustment via logistic curves and logit-shift correction.
