# Data Fetcher

`core/data/data_fetcher.py`

Downloads BTC price data from Binance. Fetches daily, hourly (1h klines), and intraday (1m klines) data.

## Key Functions

### `fetch_daily()`

Download daily BTC prices (5yr). Saves to `DATA/btc_daily.csv`.

### `fetch_hourly()`

Download hourly BTC prices (5yr of 1h klines). Saves to `DATA/btc_hourly.csv`. Used by the pricing engine for GARCH fitting.

### `fetch_intraday_1m()`

Download 1-minute BTC prices (~3 months). Saves to `DATA/btc_intraday_1m.csv`. Used for current spot price (S0) mark and vol gate.

## Data Files

| File | Rows | Frequency | Use |
|------|------|-----------|-----|
| `DATA/btc_daily.csv` | ~1,800 | Daily | Regime detection, XGBoost |
| `DATA/btc_hourly.csv` | ~44,000 | Hourly | GARCH fitting, jump calibration |
| `DATA/btc_intraday_1m.csv` | ~150,000 | 1-minute | Spot price, vol gate |

## CLI

```bash
# Fetch/refresh all BTC data
python core/data/data_fetcher.py
```

Fetches daily, hourly, and intraday BTC prices from Binance. Incremental — only downloads new data since last fetch.
