# Troubleshooting

Common issues and solutions for the BTC Polymarket Pricing system.

## Column Errors

### "Missing 'market_price' or 'model_probability' column"

Batch CSV uses different column names. The system auto-detects via precedence chains:

- **Model probability**: `p_model_fit` > `p_real_mc` > `model_probability`
- **Market price**: `market_price` > `market_pr` > `Polymarket_Price`

If using old data, try re-running `fit_probability_curves.py` to generate columns with expected names.

### "No outcome column found"

Signal diagnostics expects `outcome_yes` or `outcome`. If using a different format, add a column mapping.

## Key Matching Issues

Log warning: `"Only X/N position keys matched targets"`

Positions and batch data use different key formats. Enable `condition_id` in positions for more reliable matching:

```python
from core.data.positions import ensure_position_keys
ensure_position_keys(positions_df)
```

## GARCH Convergence

If GARCH fitting fails with non-convergence:

1. Ensure at least **500 hours (~21 days)** of hourly BTC data for stable estimation
2. Run `python core/data/data_fetcher.py` to fetch/refresh `DATA/btc_hourly.csv` (5 years of 1h klines)
3. Check for data gaps — each API call fetches up to 1000 hourly candles

## Missing Data Files

If you see `"btc_hourly.csv must contain a 'Close' or 'close' column"` or file not found:

1. Run `python core/data/data_fetcher.py` to generate `DATA/btc_hourly.csv`
2. Verify the file has `date,close` columns with ≥10K rows

### Missing macro data

If macro features are unavailable:

```bash
python core/data/macro_fetcher.py --period 5y
```

This downloads Gold, DXY, VIX, and SPX data to `DATA/macro_daily.csv`. Required by the directional XGBoost module and optional for regime detection.

## Stale Data Blocks

If all trades are blocked with "batch too stale":

1. Re-run the pipeline to get fresh data
2. Set `disable_staleness=True` in config (testing only)
3. Adjust `STALE_SOFT_HOURS` / `STALE_HARD_HOURS` in `core/strategy/common.py`

## Module Import Errors

### `No module named 'core'`

Run from project root:

```bash
cd "V2 BTC Contract Pricing"
python core/strategy/auto_reco.py --bankroll 1000
```

### `No module named 'arch'`

```bash
pip install arch
```

### `No module named 'hmmlearn'`

```bash
pip install hmmlearn
```

Required for `regime_detector.py`. GaussianHMM with 3 states for bear/sideways/bull classification.

### `No module named 'xgboost'`

```bash
pip install xgboost
```

Required for `directional_xgb.py`. Used for directional probability adjustment.

### `No module named 'yfinance'`

```bash
pip install yfinance
```

Required for `macro_fetcher.py`. Downloads Gold, DXY, VIX, SPX from Yahoo Finance.

## Jump Calibration Issues

### "Too few jumps detected"

If `jump_calibration.py` reports < 10 jumps:

1. Try bipower detection: `python core/pricing/jump_calibration.py --method bipower`
2. Lower MAD threshold: `python core/pricing/jump_calibration.py --mad-mult 2.5`
3. Ensure sufficient data: 5 years of hourly data should yield ~200-500 jumps

Reverts to Teng (2025) literature defaults when insufficient jumps detected.

## HMM Convergence

If `regime_detector.py` emits "Model is not converging":

- This is usually non-critical — model still produces reasonable state assignments
- Check that dominant regime is `sideways` (expected for BTC)
- If all weight is in one state (>0.99), reduce training window or re-run

## Polymarket API Errors

### 403 / Unauthorized

Check environment variables or `.env` file has valid API credentials.

### Timeout fetching order books

CLOB API may be rate-limited. The pipeline retries automatically with backoff.

## Vol Gate Falls Back to "unknown"

If vol gate returns `unknown` regime:

1. Check BTC data file at `DATA/btc_intraday_1m.csv`
2. Ensure data is recent (within 5 minutes of query time)
3. Check for missing/failed data fetch: `python core/data/data_fetcher.py`
