# Directional XGBoost

`core/pricing/directional_xgb.py`

XGBoost directional classifier for BTC price movement prediction. Provides a P(up) modifier that blends with the base SVCJ+HMM distribution at 30% weight.

## Classes

### `DirectionalXGB`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight` | 0.3 | Blend weight of XGBoost prediction |
| `n_estimators` | 100 | Number of boosted trees |
| `max_depth` | 4 | Maximum tree depth |
| `learning_rate` | 0.05 | Boosting learning rate |
| `random_state` | 42 | RNG seed |

#### Methods

**`train(btc_returns, macro_df=None, horizon_days=30)`** — Train classifier with time-series aware split. Returns `bool`.

**`predict_direction_adjustment(S0, hours_to_expiry, btc_returns, macro_df)`** — Predict P(up) from current features. Returns `float` 0–1, or 0.5 if untrained.

**`get_feature_importance()`** — Returns `{feature_name: importance_score}` dict.

#### Properties

- `is_trained` — `bool`, whether model is trained
- `accuracy` — `Optional[float]`, test-set accuracy

### `DirectionalResult`

| Field | Type | Description |
|-------|------|-------------|
| `prob_up` | float | P(up) from XGBoost |
| `confidence` | float | Model confidence |
| `horizon_days` | int | Forecast horizon |
| `features_used` | list | Feature names |
| `trained` | bool | Whether model was trained |

## Functions

### `build_features(btc_returns, macro_df=None, horizon_days=30)`

Build feature matrix with multi-window vol, momentum, drawdown, vol-of-vol, plus macro features (Gold, DXY, VIX, SPX). Returns `pd.DataFrame` with `target` column.

## Feature Ranking

Top features by synthesis evidence:

1. Realized volatility (multi-window: 7d, 14d, 30d, 90d)
2. BTC momentum (past 1d, 3d, 5d, 10d, 21d returns)
3. Gold returns + rolling BTC-Gold correlation
4. DXY level + trend
5. VIX level
6. SPX returns

## Magic Numbers — Uncalibrated Parameters

Several parameters are hardcoded to literature-sourced or conventional defaults and have **not** been empirically calibrated on this project's data. These should be tuned before relying on XGBoost in production.

### Critical

**`weight = 0.3`** — Blend ratio of XGBoost to SVCJ+HMM (`directional_xgb.py:39`).

Most impactful single number in the module. Cites Shelton (2024) out-of-sample evidence from a different domain. A literature citation is not a substitute for calibration. Should be optimized via time-series cross-validation over a grid like `[0.05, 0.10, 0.20, 0.30, 0.40, 0.50]`. This number gates how much any XGBoost improvement matters — if tuned to 0.05 the model becomes negligible; if 0.50+ the structural MC model is half-overwritten by a weak classifier. Getting it wrong directly distorts every probability the pricing engine outputs.

### High Priority

**`n_estimators=100, max_depth=4, learning_rate=0.05`** — XGBoost hyperparams (`directional_xgb.py:241-245`).

Conservative defaults, never tuned. Should go through `GridSearchCV` or `RandomizedSearchCV` with time-series split. Candidate ranges: `n_estimators` in `[50, 100, 200, 300]`, `max_depth` in `[3, 4, 5, 6]`, `learning_rate` in `[0.01, 0.05, 0.1]`. Secondary to blend weight since XGB contribution is capped at 30%, but still leaves predictive power on the table.

**Vol windows `[7, 14, 30, 90]`** — Feature engineering (`directional_xgb.py:95`).

Common intervals (week/biweek/month/quarter) but never validated against predictive power. Should be pruned via feature importance analysis or recursive feature elimination.

**Momentum windows `[1, 3, 5, 10, 21]`** — Feature engineering (`directional_xgb.py:100`).

21 = trading days/month, rest are ad-hoc. Same issue — should be empirically validated against directional prediction accuracy.

### Medium Priority

**`MIN_TRAIN_SAMPLES = 200`** — Training guard (`directional_xgb.py:40`).

Reasonable floor, never tested against learning curve. At what sample size does accuracy plateau?

**Horizon mapping `max(7, min(30, hours_to_expiry / 24))`** — Expiry-to-horizon conversion (`directional_xgb.py:289`).

Arbitrary 7-day floor, 30-day cap. XGBoost trained on 30-day horizon but contracts have arbitrary expiry lengths. Could be parameterized per-contract.

**80/20 train/test split** — Data partitioning (`directional_xgb.py:231`).

Standard convention, not empirically justified for this data. Rolling-window cross-validation would be more rigorous for time-series data.

**`np.clip(prob, 0.01, 0.99)`** — Probability floor/cap (`btc_pricing_engine.py:910`).

Implies 99:1 odds at extremes. Reasonable but never calibrated against observed tail frequencies in BTC.

### Low Priority (reasonable defaults)

- `n < 60` minimum data guard — practical floor for 90d rolling window + 30d target horizon.
- `random_state=42` — convention, not a parameter.
- `drawdown_30d` — matches default 30d horizon.
- `vol_of_vol` (7d vol over 90d window) — single feature, less impactful than the vol window list.

See [Pricing Engine Concept](../../concepts/pricing-engine.md) for integration details.
