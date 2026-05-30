# Troubleshooting

Common issues and solutions for the BTC Polymarket Pricing system.

## Column Errors

### "Missing 'market_price' or 'model_probability' column"

Batch CSV uses different column names. The system auto-detects via precedence chains:

- **Model probability**: `p_model_cal` > `p_model_fit` > `p_real_mc` > `model_probability`
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

1. Ensure at least **100 days** of daily BTC data
2. Check for data gaps — `data_fetcher.py` downloads 5 years

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
