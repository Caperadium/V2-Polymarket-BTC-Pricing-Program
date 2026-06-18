# Probability Fitting

`core/pricing/fit_probability_curves.py`

Logistic curve fitting for model probabilities across expiry dates. Converts raw Monte Carlo probabilities into smooth probability-vs-strike curves, applies logit-shift calibration, and merges enriched market data.

## Functions

### `process_live_data(df, batch_id=None)`

Process a live batch pricing DataFrame: date parsing, market column alignment, unique slug filtering.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | pd.DataFrame | Raw batch results DataFrame |
| `batch_id` | str/None | Batch timestamp for logging |

**Returns**: Cleaned `pd.DataFrame` with standardized columns.

### `fit_logistic_curves(df, prob_col='p_real_mc', market_price_col=None, slug_col='slug', strike_col='strike', T_col='T_days', fit_floor=None)`

Fit logistic probability curves per expiry group. Models `P(above) = 1/(1 + exp(-(a + b × strike)))` per expiry.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | pd.DataFrame | Input data with probabilities |
| `prob_col` | str | Model probability column |
| `market_price_col` | str | Market price column for logit-shift |
| `slug_col` | str | Contract slug column |
| `strike_col` | str | Strike price column |
| `T_col` | str | Days-to-expiry column |
| `fit_floor` | float/None | Floor for fitted probabilities |

**Returns**: `pd.DataFrame` with added columns: `p_model_fit`, `p_rn_fit`, `logit_shift`.

### `fit_risk_neutral_curves(df, prob_col='p_real_mc', ...)`

Fit risk-neutral probability curves from market prices. Extracts implied probability from market mid-price for each strike, then fits the same logistic functional form.

### `logit_shift_calibration(fitted_probs, market_probs, floor=0.01, cap=0.99)`

Apply logit-space shift to align model probabilities with market-implied probabilities. Computes average logit difference between model and market, then shifts all model probabilities by that offset.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fitted_probs` | np.ndarray | Fitted model probabilities |
| `market_probs` | np.ndarray | Market-implied probabilities |
| `floor` | float | Floor for clipping |
| `cap` | float | Cap for clipping |

**Returns**: `np.ndarray` of calibrated probabilities, logit shift amount.

### `enrich_order_book_prices(df, order_book_df=None)`

Enrich batch results with live order book bid/ask prices from Polymarket CLOB.

## Output Columns

| Column | Source | Description |
|--------|--------|-------------|
| `p_real_mc` | MC simulation | Raw model probability |
| `p_model_fit` | Logistic fit | Smoothed model probability curve |
| `p_rn_fit` | Market fit | Risk-neutral (market-implied) probability curve |
| `market_price` | Polymarket | Mid-market price |
| `logit_shift` | Calibration | Logit-space calibration adjustment |
| `edge` | Computed | `p_model_fit - market_price` |

## CLI

```bash
python core/pricing/fit_probability_curves.py batch_results/20250615_120000/batch_results.csv
```

See [Probability Fitting Concept](../../concepts/probability-fitting.md) for methodology, calibration adjustment, and interpretation.
