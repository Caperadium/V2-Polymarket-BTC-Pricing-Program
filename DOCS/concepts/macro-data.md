# Macro Data

`core/data/macro_fetcher.py`

Macroeconomic data feeder that downloads daily Gold, DXY, VIX, and SPX data from Yahoo Finance. Provides features for HMM regime detection (Phase 1.2) and directional XGBoost (Phase 2.3).

Based on: Köse et al. (2025) — Gold-BTC correlation via TFT, Kim et al. (2025) — macro drivers post-2019, Pakstaite et al. (2025) — structural break in BTC-macro relationships.

## Why It Matters

Post-2019, BTC price behavior has become increasingly correlated with traditional macro assets. The 2020 institutional adoption wave and 2024 ETF approvals strengthened these linkages. Macro features improve both regime detection accuracy and directional prediction:

- **Gold**: Köse et al. (2025) TFT model assigns Gold attention weight 0.85 — the strongest macro signal for BTC direction
- **DXY**: Dollar strength inversely correlated with BTC in risk-on/risk-off regimes
- **VIX**: Risk-off indicator; elevated VIX correlates with BTC drawdowns
- **SPX**: Equity market correlation, especially during bull regimes

## Data Sources

All data from Yahoo Finance (free, no API key required):

| Symbol | Yahoo Ticker | Description |
|--------|-------------|-------------|
| Gold | `GC=F` | Gold Futures (XAU/USD) |
| DXY | `DX-Y.NYB` | US Dollar Index |
| VIX | `^VIX` | CBOE Volatility Index |
| SPX | `^GSPC` | S&P 500 Index |

## Features Produced

The raw price data is enriched with derived features via `merge_with_btc()`:

| Feature | Description | Computation |
|---------|-------------|-------------|
| `gold_ret` | Daily gold return | `pct_change()` |
| `gold_level` | Gold price level | Raw close |
| `dxy_ret` | Daily DXY return | `pct_change()` |
| `dxy_level` | DXY price level | Raw close |
| `dxy_trend` | DXY trend | Rolling mean vs long-term average |
| `vix` | VIX level | Raw close |
| `spx_ret` | Daily SPX return | `pct_change()` |
| `btc_ret` | Daily BTC return | `pct_change()` |
| `btc_gold_corr_30d` | BTC-Gold rolling correlation | 30-day rolling `corr()` |
| `btc_dxy_corr_30d` | BTC-DXY rolling correlation | 30-day rolling `corr()` |

## Usage

### CLI

```bash
# Download/refresh macro data (default: 2 years)
python core/data/macro_fetcher.py

# Download 5 years
python core/data/macro_fetcher.py --days 1825
```

### Programmatic

```python
from core.data.macro_fetcher import fetch_macro_data, load_macro_data

# Download latest
fetch_macro_data()

# Load from disk
df = load_macro_data()  # Returns pd.DataFrame with date index
```

### Merged with BTC

```python
from core.data.macro_fetcher import merge_with_btc

merged = merge_with_btc(
    btc_path="DATA/btc_hourly.csv",
    macro_path="DATA/macro_daily.csv",
)
# Returns DataFrame with btc_ret + gold_ret + dxy_ret + vix + spx_ret + correlations
```

## Data Flow

```
Yahoo Finance API
      │
      ▼
macro_fetcher.py ──▶ DATA/macro_daily.csv
      │
      ├──▶ regime_detector.py (HMM features)
      │
      └──▶ directional_xgb.py (XGBoost features)
              │
              ▼
         btc_pricing_engine.py (Phase 2.3 blend)
```

## File Format

`DATA/macro_daily.csv` — CSV with date index, columns: `gold`, `gold_ret`, `dxy`, `dxy_ret`, `vix`, `spx_ret`.

Dates in UTC, resampled to daily frequency.

See [Macro Fetcher API Reference](../api-reference/core/macro-fetcher.md) for function signatures and parameters.
