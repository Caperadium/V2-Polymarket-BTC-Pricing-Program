# Directional XGBoost

`core/pricing/directional_xgb.py`

XGBoost directional classifier that provides a P(up) estimate used to **drift-shift** the base SVCJ+HMM Monte Carlo terminal distribution. Phase 2.3 of the pricing improvement plan.

Based on: Paskaleva & Vasenska (2025) — XGBoost 81% directional accuracy, Kim et al. (2025) — asymmetric features for up vs down, Oprea & Bâra (2026) — meta-learning architecture.

> **FIX 3 / H2 (re-enabled).** This component was originally disabled because the
> integration was wrong, not the model. The old design blended the P(up) into each
> strike's probability additively (`0.7·p_mc + 0.3·p_xgb`), which shifted every rung
> of the ladder by the same amount and **broke monotonicity**. It is now re-enabled
> as a single, strike-agnostic **drift shift of the terminal distribution** — see
> [Drift-Shift Integration](#drift-shift-integration) below. Off by default
> (`XGB_TILT_LAMBDA = 0.0`; runners behind `--use-xgb`).

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
                                                    ┌──────────────────────┐
                                                    │ Drift-shift terminal │
                                                    │ paths: ×exp(Δ_H)     │
                                                    │ (strike-agnostic)    │
                                                    └──────────────────────┘
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

> **BTC-macro correlation features are computed in `build_features`, not read from
> the macro CSV.** `macro_fetcher.fetch_macro_data()` writes `DATA/macro_daily.csv`
> WITHOUT `btc_gold_corr_30d` / `btc_dxy_corr_30d` (those need BTC, only added by the
> unused `merge_with_btc`). `build_features` therefore computes `btc_gold_corr` /
> `btc_dxy_corr` itself as a rolling-30 correlation between the date-joined BTC
> returns and `gold_ret` / `dxy_ret` (past-only window → leak-safe), preferring a
> precomputed `btc_*_corr_30d` column if one is present. Before this fix the two
> highest-value Köse features were silently dropped (6/8 macro features active).

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

### Drift-Shift Integration

The XGBoost P(up) is **not** mixed into per-strike probabilities. It is converted into a single constant shift of the simulated terminal distribution, applied once **before** the per-strike loop, so every strike is re-derived from the shifted paths and the ladder stays monotone by construction. Lives in `apply_xgb_drift_shift()` in `btc_pricing_engine.py`.

**Math** (`temp/xgb_activation_plan.md` §2):

```
log_ret  = log(paths / S0)                      # empirical horizon log-returns
sigma_H  = std(log_ret);  p_base = mean(paths >= S0)
p_up'    = clip(p_up, 0.15, 0.85)
p_target = clip(0.5 + λ·(p_up' − 0.5), 0.02, 0.98)
Δ_H      = −quantile(log_ret, 1 − p_target)     # empirical-CDF inversion (exact)
Δ_H      = clip(Δ_H, ±MAX_SHIFT_FRAC·sigma_H)   # safety cap
paths   *= exp(Δ_H)
```

Empirical-CDF inversion (not a Gaussian probit) hits `p_target` exactly on the actual jump/regime/skew distribution. The multiplicative shift is monotone and shape-preserving.

| Constant | Default | Role |
|----------|---------|------|
| `XGB_TILT_LAMBDA` (λ) | **0.0** | tilt strength; inert by default, production value from calibration |
| `XGB_P_FLOOR / _CEIL` | 0.15 / 0.85 | clip raw p_up |
| `XGB_P_TARGET_FLOOR / _CEIL` | 0.02 / 0.98 | clip target P(up) |
| `XGB_MAX_SHIFT_FRAC` | 0.5 | cap \|Δ_H\| as fraction of σ_H |
| `XGB_P_BASE_GUARD` | 0.02 | skip shift when base P(up) ≈ 0 or 1 (deep skew) |
| `XGB_DTE_BUCKETS` | {≤7,7–14,14–30}d | per-bucket models; train horizon = bucket midpoint |

**Gating**: Disabled by default (`use_xgb_direction=False`, λ=0.0). **Skipped under `martingale_anchor=True`** (the tilt is a physical-measure directional view, incompatible with the risk-neutral martingale). Active only for DTE ≤30d (`dte_bucket_horizon` returns `None` beyond).

### Fallback Behavior

If the model is untrained, missing macro data, prediction fails, or the model is malformed, `predict_direction_adjustment` returns 0.5 → `Δ_H = 0` → paths unchanged. The engine also wraps the whole XGB block in try/except and falls back to unshifted paths on any error. Macro-absent runs fall back to BTC-only features, which carry little directional signal (expect near-neutral).

### Empirical Validation (this project's data)

A walk-forward OOS check (weekly step, train strictly on `index < as_of`, score
P(up) against the realized forward H-day return). This is **expanding-window
walk-forward (~208 OOS folds), NOT the codebase 70/30 IS/OOS** (`in_sample_oos.py`,
which is a single-cutoff partition for the *contract* backtest, not the directional
signal). **The traded book is 1–7 DTE, so only the `≤7` DTE bucket ever fires in
production** — that is the band that matters. Scripts:
`temp/xgb_macro_walkforward.py`, `temp/xgb_macro_seed_moneyness.py`,
`temp/xgb_bootstrap_dauc.py`.

**Per-day OOS AUC (traded band):**

| Horizon (DTE) | BTC-only AUC | BTC+macro AUC | Best Spearman |
|---------------|--------------|---------------|---------------|
| 1d | 0.507 | 0.524 | ns (p=0.98) |
| 2d | 0.457 | 0.491 | ns (p=0.71) |
| 3d | 0.490 | 0.499 | ns (p=0.42) |
| 4d | 0.473 | 0.494 | ns (p=0.81) |
| 5d | 0.495 | **0.534** | ns (p=0.79) |
| 6d | 0.477 | 0.499 | ns (p=0.24) |
| 7d | 0.471 | 0.492 | ns (p=0.59) |

- **No tradeable directional skill at 1–7 DTE, macro or not.** Every AUC ≈ 0.5
  (0.46–0.53); no Spearman significant (all p > 0.2). Sub-week BTC direction from
  daily features is essentially a coin flip. **5 of 7 macro AUCs are still below
  0.50** — macro mostly nudges a sub-coin-flip signal slightly less-sub-coin-flip.

**Seed robustness — model is deterministic.** Across 8 seeds the AUC std is
**0.0000 on every day**: the classifier uses `subsample=1`, `colsample=1`, so
XGBoost has no stochasticity for the seed to perturb. The "macro ≥ btc on every day"
is therefore one deterministic result, not 8 independent draws. (Those sampling
knobs are the lever if ensemble variance / regularization is ever wanted.)

**Significance — paired bootstrap of ΔAUC (macro − btc), 5000 resamples:**

| DTE | ΔAUC | 95% CI | p(Δ≤0) |
|-----|------|--------|--------|
| 1 | +0.018 | [−0.049, +0.086] | 0.30 |
| 2 | +0.034 | [−0.029, +0.099] | 0.15 |
| 3 | +0.010 | [−0.057, +0.078] | 0.39 |
| 4 | +0.021 | [−0.039, +0.079] | 0.24 |
| 5 | +0.039 | [−0.019, +0.098] | 0.10 |
| 6 | +0.022 | [−0.039, +0.083] | 0.24 |
| 7 | +0.021 | [−0.040, +0.080] | 0.25 |

The macro lift is consistently positive but **statistically insignificant** — every
95% CI straddles 0; best p ≈ 0.10 (H=5).

**Moneyness breakdown.** The XGB `P(up)` is strike-agnostic, but a "BTC above strike"
contract at moneyness `m` resolves YES iff `ret_H > ln(1+m)`, so the same predictions
were re-scored against `outcome = (ret_H > ln(1+m))` for `m ∈ {−2%,−1%,0,+1%,+2%}`
(m>0 = OTM call, m<0 = ITM). Result: **no moneyness bucket shows reliable skill** —
all cells hug 0.5 (0.40–0.58); isolated highs (H2/+2% = 0.575, H5/ATM = 0.534) are
uncorroborated by neighbours → noise; deep OTM/ITM cells are class-imbalanced and
unstable (several below 0.45).

- For context, at the **untraded 30d horizon** there IS modest momentum/vol-driven
  skill (BTC-only AUC 0.628, Spearman 0.20 p=0.004), and there macro *degrades* it
  (→0.569). Directional ML only has reach at horizons we don't price.

**Consequence**: `XGB_TILT_LAMBDA` stays **0.0** (macro wired but inert). The tilt is
useless for the 1–7 DTE book; do not enable it — or trust the macro features —
without re-validating after feature work. The λ grid-search was deliberately **not**
run (λ only scales a non-improving signal). Note only the `≤7` DTE bucket activates
in production — the `7–14` / `14–30` buckets are **dead code for the current book**.

## Integration points

- **Live** (`scripts/pipelines/batch_pricing_runner.py`): `--use-xgb` (+ `--xgb-lambda`). Trains per-DTE-bucket models once on the full live data, caches them, and applies `apply_xgb_drift_shift` to the simulated paths per expiry.
- **Backtest** (`core/backtesting/backrunner.py`): `--use-xgb` (+ `--xgb-lambda`). Per-snapshot **leak-free**: trains per `(UTC date, DTE bucket)` on the strict-`<` truncated daily returns + a `< snapshot_time` macro slice, mirroring the jump/regime discipline. Threaded into `calculate_probabilities(use_xgb_direction=…, xgb_model=…, xgb_tilt_lambda=…)`.
- **IS/OOS** (`core/backtesting/in_sample_oos.py`): XGB is walk-forward per-snapshot, **NOT frozen** at the cutoff (only M2 is). The leak verifier's BTC arm also re-applies the macro truncation rule.

## Training

`DirectionalXGB.train` / `train_from_slice(daily_returns, macro_df, horizon_days)` accept already-truncated data (preferred: a **date-indexed `pd.Series`** of daily log returns so macro is joined by **date** — leak-safe — via `build_features`; a bare ndarray triggers legacy positional alignment with a warning). The standalone CLI still works:

```bash
python core/pricing/directional_xgb.py \
  --btc DATA/btc_hourly.csv --macro DATA/macro_daily.csv --horizon 30
```

Prerequisites: `DATA/btc_hourly.csv` (from `data_fetcher.py`) and `DATA/macro_daily.csv` (from `macro_fetcher.py`).

**Serialization**: `DirectionalXGB.save(path)` / `DirectionalXGB.load(path)` persist the model + metadata (lazy `joblib`, falling back to stdlib `pickle`), so a session need not retrain from scratch.

## Uncalibrated Parameters

Several parameters remain at literature-sourced defaults and have not been empirically tuned on this project's data:

| Parameter | Value | Priority | Recommendation |
|-----------|-------|----------|---------------|
| Tilt strength `XGB_TILT_LAMBDA` (λ) | 0.0 (inert) | **Critical** | Keep at 0 until the directional signal validates OOS. Walk-forward (see [Empirical Validation](#empirical-validation-this-projects-data)) shows no skill at 7/14d and macro-degraded skill at 30d, so the λ grid-search was deferred — fix the features first, then grid-search OOS AUC/Spearman over [0, 0.05, 0.10, 0.15, 0.20, 0.30] |
| `n_estimators` | 100 | High | GridSearchCV with [50, 100, 200, 300] |
| `max_depth` | 4 | High | GridSearchCV with [3, 4, 5, 6] |
| `learning_rate` | 0.05 | High | GridSearchCV with [0.01, 0.05, 0.1] |
| Vol windows | [7, 14, 30, 90] | High | Prune via feature importance or RFE |
| Momentum windows | [1, 3, 5, 10, 21] | High | Prune via feature importance or RFE |
| Train/test split | 0.8 | Medium | Replace with rolling-window CV |
| Horizon mapping | max(7, min(30, h/24)) | Medium | Calibrate per-contract |
| Probability clip | [0.01, 0.99] | Medium | Calibrate against observed tail frequencies |

The **tilt strength λ is the single most impactful number** — it answers "how much do we trust the ML model vs. the structural model?" It defaults to 0.0 (no effect) precisely because a literature citation is not a substitute for empirical calibration. On the current dataset the directional signal has not validated OOS (see [Empirical Validation](#empirical-validation-this-projects-data)), so λ stays 0; set a production value from the backtest grid only **after** the signal earns its keep walk-forward.

See [Directional XGBoost API Reference](../api-reference/core/directional-xgb.md) for the full class and function reference.
