# Probability Fitting

`core/pricing/fit_probability_curves.py`

After the pricing engine simulates raw probabilities per contract, logistic curves are fitted to smooth the probability-vs-strike relationship and enable calibration.

## Why Curve Fitting?

Raw Monte Carlo probabilities contain simulation noise. Fitting a logistic curve per expiry:

1. **Smooths noise** — enforces monotonicity (higher strike → lower probability)
2. **Enables calibration** — systematic bias can be corrected via logit shift
3. **Extracts structure** — curve parameters (slope `a`, midpoint `b`) reveal market shape

## Logistic Model

The fitted curve has the form:

$$p(K) = \frac{1}{1 + \exp\left(a \cdot \left(\frac{K}{1000} - b\right)\right)}$$

Where:

- $K$ = strike price (rescaled by 1000 for numerical stability)
- $a = e^{\log\_a}$ (ensures $a > 0$, strictly decreasing in $K$)
- $b$ = midpoint parameter

The reparameterization $a = e^{\log\_a}$ prevents the optimizer from exploring negative $a$ values that would produce increasing curves.

## Two Curves Per Expiry

For each expiry group, two logistic curves are fitted:

| Curve | Source Data | Purpose |
|-------|-------------|---------|
| `p_model_fit` | MC probabilities (`p_real_mc`) | Smoothed model probability |
| `p_rn_fit` | Market prices or risk-neutral probs | Smoothed market-implied probability |

The gap between these curves is the **edge**:

```python
edge_vs_market_fit = p_model_fit - market_price
```

## Logit-Shift Calibration

After fitting, a global calibration shift is applied:

$$p_{cal} = \sigma\left(\text{logit}(p_{fit}) + B\right)$$

With a fixed shift $B = -0.7$ (`PROB_LOGIT_SHIFT_B`).

This **uniformly pushes probabilities downward** without inflating low probabilities (unlike symmetric shrink-to-0.5). The shift is applied in logit space to preserve monotonicity.

## Column Output

The processed CSV adds these columns:

| Column | Description |
|--------|-------------|
| `p_model_fit` | Logistic-smoothed model probability |
| `p_rn_fit` | Logistic-smoothed risk-neutral probability |
| `p_model_cal` | Calibrated model probability (logit-shifted) |
| `edge_vs_market_fit` | Fitted model prob − market price |
| `edge_vs_rn_fit` | Fitted model prob − fitted RN prob |

## Fitting Process

```python
from core.pricing.fit_probability_curves import process_batch

process_batch(
    input_csv="batch_results/batch_summary.csv",
    output_batch_csv="fitted/batch_with_fits.csv",
    output_curve_params_csv="fitted/curve_params.csv",
)
```

1. Group contracts by expiry date (or `T_days` if no date column)
2. For each group, fit two logistic curves via `scipy.optimize.curve_fit`
3. Evaluate fitted curves at original strikes
4. Apply logit-shift calibration
5. Save augmented contract CSV + per-expiry curve params CSV

## Curve Params Output

`curve_params.csv` contains one row per expiry:

| Column | Description |
|--------|-------------|
| `T_days` | Days to expiry |
| `expiry_date` | Contract expiry date |
| `n_points` | Number of strikes in this expiry |
| `model_log_a` | Fitted log-slope for model curve |
| `model_b` | Fitted midpoint for model curve |
| `model_fit_ok` | Whether model fit succeeded |
| `rn_log_a` | Fitted log-slope for RN curve |
| `rn_b` | Fitted midpoint for RN curve |
| `rn_fit_ok` | Whether RN fit succeeded |

## Minimum Data Requirements

- At least **4 strikes** per expiry for a meaningful fit
- If fewer than 4, that expiry gets NaN for fitted columns (no crash)
- Probabilities clipped to `[1e-4, 1 − 1e-4]` to avoid logit infinities

## Usage in Pipeline

Curve fitting runs automatically as part of both:

- `run_full_pipeline.py` — after batch pricing completes
- `prob_backrunner_engine.py` — after each historical batch is simulated
