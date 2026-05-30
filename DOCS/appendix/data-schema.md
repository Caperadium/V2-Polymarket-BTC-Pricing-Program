# Data Schema

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
| `p_model_cal` | Calibrated model probability (logit-shifted) |
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
| Model probability | `p_model_cal` → `p_model_fit` → `p_real_mc` → `model_probability` |
| Market price | `market_price` → `market_pr` → `Polymarket_Price` |
| Expiry | `expiry_key` (derived from `expiry_date`) → `T_days` as float |
| DTE | `dte_days` → `t_days` → `T_days` |
| Edge | Computed as `model_prob − market_price` (YES) or `market_price − model_prob` (NO) |
