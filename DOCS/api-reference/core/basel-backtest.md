# Basel Backtest

`core/validation/basel_backtest.py`

Kupiec POF traffic light VaR backtest + expected shortfall (Acerbi-Szekely) for multi-horizon model validation.

## Functions

### `run_basel_backtest(hourly_csv)`

Run multi-horizon, multi-confidence VaR backtest on BTC hourly data.

**Returns**: `List[BaselBacktestResult]` — one result per (horizon, confidence) combination.

### `basel_traffic_light(n_violations, n_windows, confidence)`

Kupiec POF likelihood ratio test. Returns `BaselBacktestResult` with traffic light classification.

### `expected_shortfall_test(returns, var_series, confidence)`

Acerbi-Szekely ES test. Returns Z1, Z2, Z3 statistics.

## Classes

### `BaselBacktestResult`

| Field | Type | Description |
|-------|------|-------------|
| `horizon_days` | int | VaR horizon in days |
| `confidence` | float | VaR confidence level (0.95, 0.975, 0.99) |
| `n_windows` | int | Number of rolling windows |
| `n_violations` | int | Number of VaR violations |
| `violation_rate` | float | Observed violation rate |
| `expected_rate` | float | Expected rate (1 − confidence) |
| `pof_pvalue` | float | Kupiec POF p-value |
| `traffic_light` | str | Green / Yellow / Red |
| `z1` | float | ES Z1 statistic (average tail loss accuracy) |
| `z2` | float | ES Z2 statistic (severity-weighted exceedances) |
| `z3` | float | ES Z3 statistic (combined frequency + severity) |

See [Basel Validation Concept](../../concepts/basel-validation.md) for interpretation and methodology.
