# Macro Fetcher

`core/data/macro_fetcher.py`

Download macroeconomic data from Yahoo Finance for regime detection and directional prediction. No API key required.

## Functions

### `fetch_macro_data(days=None, period="5y", tickers=None, output_path=None)`

Fetch Gold (GC=F), DXY (DX-Y.NYB), VIX (^VIX), SPX (^GSPC) daily data from Yahoo Finance.

**Returns**: `pd.DataFrame` with columns: `gold`, `dxy`, `vix`, `spx`, `gold_ret`, `dxy_ret`, `vix_ret`, `spx_ret`, `vix_regime`, `dxy_trend`. Saved to `DATA/macro_daily.csv`.

### `load_macro_data(path=None, min_rows=60)`

Load macro data from disk. Returns `pd.DataFrame` or `None` if file missing/insufficient data.

### `merge_with_btc(btc_path="DATA/btc_hourly.csv", macro_path=None, resample="D")`

Merge macro data with daily BTC prices. Adds columns: `btc`, `btc_ret`, `btc_gold_corr_30d`, `btc_dxy_corr_30d`.

**Returns**: `pd.DataFrame` or `None` if macro data unavailable.

## Ticker Symbols

| Feature | Ticker | Source |
|---------|--------|--------|
| Gold | `GC=F` | Gold Futures |
| DXY | `DX-Y.NYB` | US Dollar Index |
| VIX | `^VIX` | CBOE Volatility Index |
| SPX | `^GSPC` | S&P 500 Index |

## Derived Features

| Column | Description |
|--------|-------------|
| `{col}_ret` | Daily pct_change for each series |
| `vix_regime` | low (< 15), medium (15–25), high (> 25) |
| `dxy_trend` | Sign of 20-day MA slope |
| `btc_gold_corr_30d` | Rolling 30-day BTC-Gold correlation |
| `btc_dxy_corr_30d` | Rolling 30-day BTC-DXY correlation |

## Evidence Basis

| Feature | Attention Weight | Source |
|---------|-----------------|--------|
| Gold | 0.85 | Köse et al. (2025) TFT |
| DXY | 0.52 | Kim et al. (2025) |
| VIX | 0.34 | Pakstaite et al. (2025) |
| SPX | — | Macro context |

See [Regime Detection Concept](../../concepts/regime-detection.md) for integration details.
