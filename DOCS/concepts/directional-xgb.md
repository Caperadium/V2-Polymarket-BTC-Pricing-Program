# Directional XGBoost

`core/pricing/directional_xgb.py`

XGBoost directional classifier that provides a P(up) modifier blended with the base SVCJ+HMM Monte Carlo distribution. Phase 2.3 of the pricing improvement plan.

Based on: Paskaleva & Vasenska (2025) — XGBoost 81% directional accuracy, Kim et al. (2025) — asymmetric features for up vs down, Oprea & Bâra (2026) — meta-learning architecture.

## Why It Matters

The base GARCH+SVCJ Monte Carlo distribution is directionally uninformative — it models volatility and tail behavior but does not incorporate directional signals from macro conditions or BTC momentum. The directional XGBoost provides a lightweight, interpretable P(up) estimate that tilts the distribution without replacing the structural model.

## Architecture

```
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  BTC Daily       │────▶│  Feature             │────▶│  XGBoost          │
│  Returns         │     │  Engineering         │     │  Classifier       │
├──────────────────┤     │  (vol, momentum,     │     │  (n_est=100,      │
│  Macro Data      │────▶│   drawdown, macro)   │     │   depth=4, lr=.05)│
│  (Gold, DXY,     │     └─────────────────────┘     └────────┬─────────┘
│   VIX, SPX)      │                                          │
└──────────────────┘                                          ▼
                                                    ┌──────────────────┐
                                                    │  Blend: 0.7×SVCJ │
                                                    │  + 0.3×P(up)    │
                                                    └──────────────────┘
```

## Model

### Feature Engineering

Features are built from BTC daily returns and optional macro data:

| Category | Features | Rationale |
|----------|----------|-----------|
| Volatility | `vol_7d`, `vol_14d`, `vol_30d`, `vol_90d` | Multi-window captures short/medium/long vol regimes |
| Momentum | `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_21d` | Past returns at various lookbacks |
| Drawdown | `drawdown_30d` | Drawdown magnitude signals trend exhaustion |
| Vol-of-vol | `vol_of_vol` (7d vol std over 90d) | Volatility regime changes |
| Gold | `gold_ret_30d`, `gold_level`, `btc_gold_corr` | Gold-BTC correlation (Köse 2025 TFT weight: 0.85) |
| DXY | `dxy_ret_30d`, `dxy_trend`, `btc_dxy_corr` | Dollar strength vs BTC |
| VIX | `vix_level` | Risk-off sentiment |
| SPX | `spx_ret_30d` | Equity market correlation |

**Target**: Binary `(future_Nd_return > 0)`, shifted by `-horizon_days` to prevent lookahead.

### Classifier

```python
xgb.XGBClassifier(
    n_estimators=100,      # Conservative — deep trees not needed
    max_depth=4,           # Shallow to avoid overfitting
    learning_rate=0.05,    # Slow learning
    eval_metric="logloss",
)
```

Trained on 80/20 time-series split (no shuffle — respects temporal order). Minimum 200 training samples required.

### Blend Formula

The XGBoost P(up) is blended with the base MC probability:

```
p_final = 0.7 × p_mc + 0.3 × p_xgb
```

The 30% weight (Shelton 2024) deliberately limits ML influence — OOS evidence shows individual directional predictors are weak. The weight is applied per-strike in `calculate_probabilities()`.

### Fallback Behavior

If the model is untrained, missing macro data, or prediction fails, returns 0.5 (neutral — no tilt). This ensures the pricing engine degrades gracefully to the structural model alone.

## Integration

The model sits inside `calculate_probabilities()` in `btc_pricing_engine.py` at Phase 2.3:

```python
if use_xgb_direction and xgb_model is not None:
    direction_modifier = xgb_model.predict_direction_adjustment(
        S0=S0, hours_to_expiry=hours_to_expiry, macro_df=macro_df,
    )
    prob = (1 - XGB_WEIGHT) * prob + XGB_WEIGHT * direction_modifier
    prob = np.clip(prob, 0.01, 0.99)
```

**Gating**: Disabled by default (`use_xgb_direction=False`). Auto-disabled beyond 30-day horizon (medium-horizon gate).

## Training

Training is CLI-only — not automated in any pipeline:

```bash
python core/pricing/directional_xgb.py \
  --btc DATA/btc_hourly.csv \
  --macro DATA/macro_daily.csv \
  --horizon 30
```

Prerequisites: `DATA/btc_hourly.csv` (from `data_fetcher.py`) and `DATA/macro_daily.csv` (from `macro_fetcher.py`).

The model is **never serialized to disk** — no pickle, no joblib, no checkpoint. Each session that needs XGBoost must train from scratch or receive a pre-trained instance via dependency injection.

## Uncalibrated Parameters

Several parameters remain at literature-sourced defaults and have not been empirically tuned on this project's data:

| Parameter | Value | Priority | Recommendation |
|-----------|-------|----------|---------------|
| Blend weight | 0.3 | **Critical** | Grid-search via time-series CV over [0.05, 0.10, 0.20, 0.30, 0.40, 0.50] |
| `n_estimators` | 100 | High | GridSearchCV with [50, 100, 200, 300] |
| `max_depth` | 4 | High | GridSearchCV with [3, 4, 5, 6] |
| `learning_rate` | 0.05 | High | GridSearchCV with [0.01, 0.05, 0.1] |
| Vol windows | [7, 14, 30, 90] | High | Prune via feature importance or RFE |
| Momentum windows | [1, 3, 5, 10, 21] | High | Prune via feature importance or RFE |
| Train/test split | 0.8 | Medium | Replace with rolling-window CV |
| Horizon mapping | max(7, min(30, h/24)) | Medium | Calibrate per-contract |
| Probability clip | [0.01, 0.99] | Medium | Calibrate against observed tail frequencies |

The **blend weight is the single most impactful number** — it answers "how much do we trust the ML model vs. the structural model?" A literature citation is not a substitute for empirical calibration.

See [Directional XGBoost API Reference](../api-reference/core/directional-xgb.md) for the full class and function reference.
