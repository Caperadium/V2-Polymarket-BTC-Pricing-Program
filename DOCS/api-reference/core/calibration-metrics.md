# Calibration Metrics

`core/validation/calibration_metrics.py`

Probability forecast calibration diagnostics: Brier score, reliability diagrams, and Expected Calibration Error (ECE).

## Functions

### `brier_score(p_model, outcomes)`

Compute Brier score — mean squared error between forecast probability and binary outcome.

| Parameter | Type | Description |
|-----------|------|-------------|
| `p_model` | np.ndarray | Model probabilities in (0, 1) |
| `outcomes` | np.ndarray | Binary outcomes (0 or 1) |

**Returns**: `float` ∈ [0, 1]. 0.25 = coin-flip baseline. Lower is better.

### `reliability_bins(p_model, outcomes, n_bins=10)`

Build reliability diagram bins: mean forecast vs observed frequency per bin.

| Parameter | Type | Description |
|-----------|------|-------------|
| `p_model` | np.ndarray | Model probabilities in (0, 1) |
| `outcomes` | np.ndarray | Binary outcomes (0 or 1) |
| `n_bins` | int | Number of equal-width bins (default 10) |

**Returns**: `pd.DataFrame` with columns: `bin_center`, `bin_lower`, `bin_upper`, `n_obs`, `mean_forecast`, `observed_freq`.

### `ece_score(p_model, outcomes, n_bins=10)`

Expected Calibration Error — weighted average of |forecast − observed| per bin.

| Parameter | Type | Description |
|-----------|------|-------------|
| `p_model` | np.ndarray | Model probabilities in (0, 1) |
| `outcomes` | np.ndarray | Binary outcomes (0 or 1) |
| `n_bins` | int | Number of bins (default 10) |

**Returns**: `float` ∈ [0, 1]. < 0.05 = well-calibrated, > 0.10 = poor.

### `run_calibration_report(csv_path, prob_col='p_model_fit', outcome_col='outcome', n_bins=10)`

Run full calibration diagnostics on a priced batch CSV.

| Parameter | Type | Description |
|-----------|------|-------------|
| `csv_path` | str | Path to `batch_with_fits.csv` |
| `prob_col` | str | Model probability column name |
| `outcome_col` | str | Outcome column name |
| `n_bins` | int | Reliability diagram bins |

**Returns**: `CalibrationReport` or `None` if data insufficient.

## Dataclass

### `CalibrationReport`

| Field | Type | Description |
|-------|------|-------------|
| `brier` | float | Brier score |
| `ece` | float | Expected Calibration Error |
| `n_obs` | int | Number of observations |
| `bins` | pd.DataFrame | Reliability diagram bins |
| `mean_forecast` | float | Average model probability |
| `mean_outcome` | float | Average observed frequency |
| `calibration_bias` | float | `mean_forecast - mean_outcome` |

## CLI

```bash
python core/validation/calibration_metrics.py batch_with_fits.csv
python core/validation/calibration_metrics.py batch.csv --prob-col p_real_mc --outcome-col resolved --n-bins 20
```

See [Calibration Metrics Concept](../../concepts/calibration-metrics.md) for interpretation guidance and usage within the validation pipeline.
