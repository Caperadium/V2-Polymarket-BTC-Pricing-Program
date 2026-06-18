# Data Schema

## Key Data Files

| File | Source | Description |
|------|--------|-------------|
| `DATA/btc_hourly.csv` | `data_fetcher.py` (Binance) | 5yr Binance 1h klines: `date,close`. Primary GARCH fitting data. |
| `DATA/btc_daily.csv` | `data_fetcher.py` (Binance) | Daily resampled prices. Used for regime detection + XGBoost training. |
| `DATA/btc_intraday_1m.csv` | `data_fetcher.py` (Binance) | ~3 months of 1m klines. Current spot price (S0) mark. |
| `DATA/macro_daily.csv` | `macro_fetcher.py` (Yahoo Finance) | Gold, DXY, VIX, SPX daily prices + derived features. Used by regime detector and directional XGBoost. |

## CSV Column Conventions

### Batch Output Columns

`batch_results/<timestamp>/batch_results.csv` and `batch_with_fits.csv`:

| Column | Description |
|--------|-------------|
| `slug` | Contract slug (e.g. `bitcoin-above-90000-on-dec-31`) |
| `strike` | Strike price |
| `market_price` | Polymarket mid-price at pricing time |
| `p_real_mc` | Raw Monte Carlo probability |
| `p_model_fit` | Logistic-fitted model probability |
| `p_rn_fit` | Logistic-fitted risk-neutral probability |
| `T_days` | Days to expiry |
| `date` | Pricing date (UTC) |
| `expiry_date` | Contract expiry date (UTC) |
| `edge_vs_market_fit` | Fitted model prob − market price |
| `edge_vs_rn_fit` | Fitted model prob − fitted RN prob |
| `yes_ask_price` | Live YES ask from CLOB |
| `yes_bid_price` | Live YES bid from CLOB |
| `no_ask_price` | Live NO ask from CLOB |
| `no_bid_price` | Live NO bid from CLOB |
| `condition_id` | Polymarket condition ID |
| `clob_token_ids` | JSON array of CLOB token IDs |
| `outcomes` | JSON array of outcome labels |

### Macro Data (`DATA/macro_daily.csv`)

| Column | Description |
|--------|-------------|
| `gold` | Gold futures price (GC=F) |
| `dxy` | US Dollar Index (DX-Y.NYB) |
| `vix` | CBOE Volatility Index (^VIX) |
| `spx` | S&P 500 index (^GSPC) |
| `gold_ret` | Daily gold return |
| `dxy_ret` | Daily DXY return |
| `vix_ret` | Daily VIX return |
| `spx_ret` | Daily SPX return |
| `vix_regime` | VIX classification: low/medium/high |
| `dxy_trend` | 20-day MA slope direction |

### Merged BTC+Macro

When running `merge_with_btc()`, additional columns:

| Column | Description |
|--------|-------------|
| `btc` | BTC daily close |
| `btc_ret` | BTC daily return |
| `btc_gold_corr_30d` | Rolling 30-day BTC-Gold correlation |
| `btc_dxy_corr_30d` | Rolling 30-day BTC-DXY correlation |

### Position CSV (`positions.csv`)

| Column | Description |
|--------|-------------|
| `id` | Position UUID |
| `slug` | Contract slug |
| `expiry_key` | Expiry identifier |
| `expiry_date` | Contract expiry |
| `strike` | Strike price |
| `side` | YES or NO |
| `entry_price` | Average entry price |
| `size_shares` | Number of shares |
| `notional` | Total cost basis |
| `current_price` | Mark-to-market price |
| `mtm_value` | Market value |
| `realized_pnl` | Realized profit/loss |
| `unrealized_pnl` | Mark-to-market PnL |
| `status` | `open` or `closed` |

### Backtest Output Columns

`taken_trades.csv`:

| Column | Description |
|--------|-------------|
| `trade_id` | Unique trade identifier |
| `pricing_date` | When the trade was generated |
| `expiry_date` | Contract expiry |
| `slug` | Contract slug |
| `strike` | Strike price |
| `side` | YES or NO |
| `entry_price` | Trade execution price |
| `stake` | USD invested |
| `size_shares` | Shares bought |
| `outcome` | 1 = won, 0 = lost, NaN = unsettled |
| `pnl` | Realized PnL |
| `position_key` | slug\|expiry\|strike\|side |

## SQLite Schema

### Polymarket Console (`polymarket_console.db`)

4 core tables + 3 ledger tables. See [Database API Reference](../api-reference/polymarket/db.md) for full DDL.

### Column Name Precedence

The codebase resolves column names via precedence chains:

| Semantic Column | Fallback Order |
|----------------|----------------|
| Model probability | `p_model_fit` → `p_real_mc` → `model_probability` |
| Market price | `market_price` → `market_pr` → `Polymarket_Price` |
| Expiry | `expiry_key` (derived from `expiry_date`) → `T_days` as float |
| DTE | `dte_days` → `t_days` → `T_days` |
| Edge | Computed as `model_prob − market_price` (YES) or `market_price − model_prob` (NO) |
| Risk-neutral prob | `p_rn_fit` > `risk_neutral_prob_fit` > `risk_neutral_prob` |
