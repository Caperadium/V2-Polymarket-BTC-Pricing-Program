# Probability Fitting

`core/pricing/fit_probability_curves.py`

After the pricing engine simulates raw probabilities per contract, logistic curves are fitted to smooth the probability-vs-strike relationship and provide a clean probability surface.

## Why Curve Fitting?

Raw Monte Carlo probabilities contain simulation noise. Fitting a logistic curve per expiry:

1. **Smooths noise** — enforces monotonicity (higher strike → lower probability)
2. **Extracts structure** — curve parameters (slope `a`, midpoint `b`) reveal market shape

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
| `p_market_fit` (alias `p_rn_fit`) | Market prices or risk-neutral probs | Smoothed market-implied probability (fit to the **market price** — not a risk-neutral model prob; renamed from `p_rn_fit` per FIX 9) |

!!! note "Known model risk (FIX 10)"
    The fitted curve is a 2-parameter **symmetric** logistic with unweighted SSE,
    so it cannot represent the skewed SVCJ/skewed-t wings and over-weights the
    saturated 0/1 tails. It is a denoiser, not a full skewed risk-neutral density.

The gap between these curves is the **edge**:

```python
edge_vs_market_fit = p_model_fit - market_price
```

## Column Output

The processed CSV adds these columns:

| Column | Description |
|--------|-------------|
| `p_model_fit` | Logistic-smoothed model probability |
| `p_model_cal` | Calibrated model probability (only when `USE_CALIBRATED_PROB=True`) |
| `p_market_fit` | Logistic fit to market price (alias: `p_rn_fit`, deprecated) |
| `edge_vs_market_fit` | Fitted model prob − market price |
| `edge_vs_rn_fit` | Fitted model prob − fitted market-curve prob |

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
4. Save augmented contract CSV + per-expiry curve params CSV

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
