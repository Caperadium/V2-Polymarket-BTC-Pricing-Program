# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## Standalone favorite-longshot-bias walkforward CLI

Add `scripts/backtesting/walkforward_flb.py`: a model-free, standalone CLI that
partitions the Polymarket contract history (`DATA/historical_contract_prices.csv`)
into N contiguous calendar windows (default 4, ~2.75mo) and, per window, measures
the longshot *gap* (mean market YES − realized YES rate) with a Wilson CI, then
simulates the deployable PnL of buying NO on every OTM in-band contract
(market YES ∈ [0.05,0.20], moneyness > 0) at a modeled ask and holding to
resolution. Reports per-window gap+CI, simulated PnL/ROI, and per-trade Sharpe;
takes `--bankroll`.

Design notes:
- Nothing is fitted on outcomes — gap is a descriptive realized statistic and PnL
  is a realized mechanical rule, so windows are descriptive partitions, NOT IS/OOS
  train/test (no leak to guard, unlike the M2 shift).
- Two time axes kept distinct: `entry_date` (midnight-UTC) drives band/moneyness/
  windowing; settlement (12:00 ET) drives outcome only.
- Reuses leak-free helpers from `BacktestEngine` (`_spot_as_of`,
  `resolve_outcome_yes`, `_expiry_is_settleable`) and `ContractPriceStore.load()`.
  The store's `resolution` column is all-NaN, so outcomes come from BTC settlement.
- Entry = first in-band observation per contract; one NO trade per contract.
- No historical bid/ask in the store: NO ask modeled as (1−mid) + spread/2
  (`--spread`, default 0.02). Fees `--fee` (fraction of stake), default 0.
- Transparency: reports contracts never-in-band and dropped-unsettleable, plus an
  intraday-coverage preflight warning and per-window small-sample flag (n < 30).

Files: `scripts/backtesting/walkforward_flb.py` (new). No existing code changed.

## Show small-sample metrics on the Backtest tab (keep the warning, drop the blocker)

The Backtest tab hid summary metrics entirely below N=200 in two places. Now the
metrics always render; only the warning banner is kept (reworded to "low-confidence
exploratory view"). Files: `app/pages/backtesting.py`.

- **Signed-Edge Reliability Diagram panel** (`_render_signed_edge_panel`): removed the
  `suppress_metrics` gate so Brier / Brier(market) / BSS / ECE / calibration
  slope+intercept always show. Below N=200 the warning stays and the "📊 Panel Metrics"
  subheader gets a "⚠️ low-confidence" tag. `_panel_metrics` already no-op-returns for
  n<2 and has ECE quantile→fixed-width fallbacks, so the ungated low-N path is safe.
- **Signal Diagnostics summary**: the §8 `small_sample_state` (`isoos_suppress`) no
  longer hides Spearman ρ / AUC / Observations / mean-edge — they render with the
  small-sample info banner kept. Hardened AUC formatting: `run_full_report` returns
  `auc=None` when only one outcome class is present (`diagnostics.py:347,386`), which the
  old `f"{diag.get('auc',0.5):.4f}"` would have crashed on once shown at low N — now
  formats to "n/a". The §8 module (`in_sample_oos.py`) is unchanged; only the page's
  display policy changed.

## Dedup per-snapshot GARCH/FIGARCH refit in the backtest

The backrunner priced each expiry group at a snapshot with a separate
`calculate_probabilities` call, each re-running `fit_garch_model` (FIGARCH MLE) and
`load_and_prep_data` on the *identical* per-snapshot hourly slice. K expiry groups =
K redundant fits. Now fit/derive once per snapshot and reuse — byte-identical output,
pure speedup.

- **`core/pricing/btc_pricing_engine.py`**: `calculate_probabilities` gains two
  optional kwargs, `garch_cache: dict` and `s0_override: float` (both default `None` →
  original load-then-fit behavior). The load+fit block (which runs *after* the horizon
  gate finalizes `use_figarch`) now skips the fit on a cache hit and skips the data load
  entirely when a hit coincides with a supplied S0. The cache is keyed on the effective
  `use_figarch` flag, so the FIGARCH↔GARCH choice — and the deterministic
  FIGARCH→GARCH convergence fallback — stay correct.
- **`core/backtesting/backrunner.py`**: `_process_one` builds one
  `snapshot_garch_cache = {}` and a `snapshot_s0` (case-insensitive `close` lookup,
  matching `load_and_prep_data`) before the expiry-group loop, and threads both into
  every `calculate_probabilities` call. A snapshot with K expiry groups now does 1
  FIGARCH fit instead of K (+1 only if a >90d group coexists).
- Verified: A/B/C in-process equivalence (default vs cold-cache+override vs warm-cache)
  is exact across FIGARCH/GARCH paths and short/long horizons; live `--serial` smoke run
  shows 4 fits over 4 snapshots (two with 2 expiry groups → 6 group-calls); all 22
  `tests/test_pricing_engine_fixes.py` pass.

## Wire in macro data to XGBoost + validate (Phase A/B/C)

Make the macro directional features actually flow into the XGBoost model, fix two
silently-dropped features, and validate the signal walk-forward.

- **`DATA/macro_daily.csv`** (Phase A): fetched 5y of Gold/DXY/VIX/SPX via
  `core/data/macro_fetcher.py` (1259 rows, 2021-06-25→2026-06-25). The file was never
  on disk before, so every `--use-xgb` run had been silently falling back to BTC-only
  features. The macro plumbing in `backrunner.py` / `batch_pricing_runner.py` /
  `btc_pricing_engine.py` was already complete — this just supplies the data.
- **`core/pricing/directional_xgb.py`** (Phase B): `build_features` now **computes**
  `btc_gold_corr` / `btc_dxy_corr` as a leak-safe rolling-30 correlation between the
  date-joined BTC returns and `gold_ret` / `dxy_ret` (preferring a precomputed
  `btc_*_corr_30d` column if present). `fetch_macro_data` never wrote those columns
  (only the unused `merge_with_btc` did), so the two highest-value Köse features were
  being silently skipped — now 8/8 macro features fire (verified non-null, sensible range).
- **Validation** (Phase C): walk-forward OOS (`temp/xgb_macro_walkforward.py`). The traded
  book is **1–7 DTE**, so only the `≤7` bucket fires in production. Across 1/2/3/5/7d
  **neither BTC-only nor BTC+macro has any OOS skill** (every AUC ≈ 0.5, range 0.46–0.53;
  no significant Spearman, all p>0.3); macro is marginally-but-insignificantly higher than
  BTC-only. Only the untraded 30d horizon shows skill (momentum/vol-driven, BTC-only AUC
  0.628), where macro *hurts*. The macro lift is confirmed insignificant by a paired
  bootstrap of ΔAUC (every 95% CI crosses 0, best p≈0.10); the seed sweep was a no-op
  (XGBoost deterministic at `subsample=colsample=1`); and the moneyness breakdown
  (P(up) re-scored against `ret_H > ln(1+m)`) is flat across OTM/ATM/ITM. `XGB_TILT_LAMBDA`
  therefore **stays 0.0**, λ grid-search deferred, and the `7–14`/`14–30` DTE buckets are
  dead code for the current book.
- **Regime-detector finding**: verified `core/pricing/regime_detector.py` is **univariate
  BTC returns** — it does NOT consume macro (the `macro-data.md` data-flow diagram claiming
  a `regime_detector` arm was wrong). With the XGB tilt off, macro is currently
  **fetched but unused in production**; kept for re-validation / future feature work.
- **Docs**: `DOCS/concepts/directional-xgb.md` (corr-features-computed note + Empirical
  Validation rewritten around the 1–7 DTE band + λ guidance) and
  `DOCS/concepts/macro-data.md` (correct CSV column list, corr-columns-not-in-file note,
  fixed data-flow diagram + verified-status note, softened regime claim).

## Re-enable XGBoost directional signal as a distribution drift shift (FIX 3 / H2)

Re-activate the dormant XGBoost directional model, replacing the invalid per-strike
additive blend (which broke ladder monotonicity, hence its original disablement) with
a single strike-agnostic **drift shift** of the simulated terminal distribution. Plan,
math, freeze policy, and plan-review resolutions in `temp/xgb_activation_plan.md`.

- **`core/pricing/btc_pricing_engine.py`**: new `apply_xgb_drift_shift()` (empirical-CDF
  inversion `Δ_H = −quantile(log_ret, 1−p_target)`, `p_target = 0.5 + λ·(p_up−0.5)`,
  safety cap, `p_base`/`sigma_H` guards) and `dte_bucket_horizon()`; XGB constants
  (`XGB_TILT_LAMBDA=0.0` default-inert, floors/ceils, `XGB_MAX_SHIFT_FRAC`,
  `XGB_P_BASE_GUARD`, `XGB_DTE_BUCKETS`). `calculate_probabilities` now computes a
  strike-agnostic `p_up` once (after path assembly, before the strike loop) and shifts
  the paths; the old `NotImplementedError` block is removed; skipped under
  `martingale_anchor=True`; gated to DTE ≤30d; `_meta` gains `xgb_p_up/_delta_H/_applied`.
  New `xgb_tilt_lambda` param. Leak-free daily returns derived unconditionally (C1 fix).
- **`core/pricing/directional_xgb.py`**: `build_features` reworked for a **date-indexed
  Series** with leak-safe **macro date-join + past-only ffill** (C3 fix; legacy positional
  path kept with a warning), `include_target` flag so prediction keeps the latest row,
  and a NaN-preserving target (drops unlabelable trailing rows instead of mislabelling
  them 0). Added `to_daily_log_return_series`, `train_from_slice`, explicit `horizon_days`
  override in `predict_direction_adjustment` (per-DTE bucket, C2-a), `save`/`load`
  (lazy joblib→pickle). Deprecated `XGB_WEIGHT` / `DEFAULT_FORECAST_HORIZONS`.
- **`core/backtesting/backrunner.py`**: per-snapshot leak-free XGB — `_worker_macro`
  global + loader (parallel & serial), per-`(UTC date, DTE bucket)` model cache trained on
  the strict-`<` truncated slice + `< snapshot_time` macro, macro-required warning, and
  `--use-xgb` / `--xgb-lambda` CLI + `BackrunnerEngine(use_xgb, xgb_tilt_lambda)`.
  Default off → byte-identical to prior backtests.
- **`core/backtesting/in_sample_oos.py`**: §1.2 note updated (XGB walk-forward, not frozen);
  `_load_btc_max_index` + leak verifier BTC arm extended to assert the macro slice holds
  no row ≥ snapshot_time (C6 fix).
- **`scripts/pipelines/batch_pricing_runner.py`**: live `--use-xgb` / `--xgb-lambda`,
  per-bucket model setup, drift shift applied to simulated paths.
- **Tests**: new `tests/test_xgb_drift_shift.py` (21 cases: identity, monotonicity,
  direction, cap, ECDF accuracy/monotone-in-p_up, p_base guards, macro date-join past-only,
  DTE buckets, untrained neutral, bit-for-bit flag-off regression, martingale skip, verifier
  macro arm, n_sims stochasticity). Updated the obsolete FIX 3 test in
  `tests/test_pricing_engine_fixes.py` from "raises NotImplementedError" to graceful degrade.
- **`core/backtesting/orchestrator.py`**: `BacktestingOrchestrator(use_xgb, xgb_tilt_lambda)`
  threaded into both `BackrunnerEngine` construction sites (default off).
- **`app/pages/backtesting.py`**: Mode A (Live Fetch) sidebar gains a "Use XGBoost directional
  drift" checkbox + tilt-λ slider (with a macro-missing caption), passed into `run_full` via the
  orchestrator. Mode B (existing batches) is unaffected — it reads pre-fitted CSVs, no backrun.
- **Docs**: rewrote `DOCS/concepts/directional-xgb.md` (blend → drift-shift) and updated the
  FIX 3 note in `CLAUDE.md`.
