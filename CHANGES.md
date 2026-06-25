# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## feat: in-sample / out-of-sample evaluation window for the Backtest tab

Implements `temp/is_oos_plan.md` (plan reviewed → Conditional GO, all findings folded in). Adds a global IS/OOS window to the Backtesting dashboard tab, driven off the contract-level `all_priced_df`.

**Audit outcome:** the only active, outcome-consuming, pooled fitted component is the **M2 logit-shift B** (`fit_probability_curves.fit_calibration`). All BTC-process components (GARCH/FIGARCH `d`, jump calibration, regime HMM) and the per-expiry logistic fit are already fit per-snapshot on strictly `< snapshot_time` truncated slices by the backrunner — walk-forward leak-free, left untouched. XGBoost is excluded (dead in the hot path; when re-enabled it belongs per-snapshot in the backrunner, covered by the verifier's BTC arm).

**New module `core/backtesting/in_sample_oos.py`** (exported from `core/backtesting/__init__.py`):
- `WindowMode`/`WindowSpec`, `compute_default_cutoff` (70/30 unique-contract split, midnight-floored), `partition_contracts`, `apply_window`, `apply_window_trades`.
- `m2_training_set` — M2 training population defined by **settlement_time < cutoff** (not snapshot time), the §9 leak guard. `settlement_time` is derived from `expiry_date` via the engine's 12:00-ET rule (no such column exists in `all_priced_df`).
- `train_pipeline` (fits `m2` only, writes to a cutoff-keyed cache, **never** the global `DATA/calibration_shift.csv`; fits B on `model_prob_raw`), `load_artifacts`, `load_or_train` (OOS is load-only — a cache miss/stale fingerprint raises; never refits).
- Cache `DATA/is_oos_cache/cutoff_<date>/` with `manifest.json` (fingerprint = cutoff + n_is_train + is_label_max_ts + code_version + params_hash → invalidation) and IS-only `calibration_shift.csv`.
- `guarded_filter` (OOS hygiene — raises on outcome-conditioning queries in OOS), `small_sample_state` (N<200 suppression), `apply_oos_calibration` (overlay `p_model_cal` from cached IS B), `is_m2_inert` (banner signal), `verify_oos_leak_free` (§9: M2 + no-other-pool arms always; heavy BTC-truncation arm pytest-only).

**`app/pages/backtesting.py`** — global window control (radio + cutoff date) at the top of the results section; all panels respect it. Diagnostics (Spearman/AUC, DTE/moneyness, tail mispricing) recomputed per-window via `SignalDiagnostics(window_df)` instead of the precomputed full-frame dict (also makes panels render in Mode B for the first time). Signed-edge panel and trade-sim panels (Summary/Equity/Daily PnL/Win-Rate) all windowed. Small-sample + inert-B banners. Settlement-bucketed Daily PnL/Sharpe captioned.

**`tests/test_in_sample_oos.py`** — 15 tests incl. the §9 verifier on ≥3 random OOS contracts plus label-leak and BTC-truncation-violation injections that must raise.

## fix: M2 panel toggle — propagate p_model_cal + raw model prob for true compare

`BacktestEngine._all_priced_contracts` records were built from a fixed-key dict that omitted `p_model_cal`, so the column was dropped from `all_priced_df` even when the fitted batches carried it. The signed-edge panel's "Use M2-calibrated p" toggle gates on `"p_model_cal" in all_priced_df.columns` (backtesting.py:499), so it was permanently greyed out. Now carries `p_model_cal` through (NaN when the batch lacks it). Also added `model_prob_raw` (resolve via non-calibrated `MODEL_PROB_CANDIDATES`, independent of `USE_CALIBRATED_PROB`) so the panel is a real raw-vs-calibrated comparison: `_build_eval_df` now uses `p_model_cal` for the ON state and `model_prob_raw` for the OFF state (was `model_prob_used`, which already equals `p_model_cal` when the backtest ran with the flag on). Falls back to `model_prob_used`/resolve for older `all_priced_df` frames.

## fix: anchor calibration_shift.csv path to project root

`CALIBRATION_SHIFT_PATH` in `core/pricing/fit_probability_curves.py` was a CWD-relative `"DATA/calibration_shift.csv"`, so the writer (`fit_calibration`) and reader (`load_calibration_shift`) resolved to different files when launched from different working directories (e.g. Streamlit running from `app/`) — the Backtest tab's calibration toggle could then never find the table. Now anchored to the project root via `Path(__file__).resolve().parents[2]`, so both default to the same absolute path regardless of CWD. Callers passing an explicit `output_path`/`path` (e.g. the tmp_path test) are unaffected.

## feat: Backtest tab toggle for USE_CALIBRATED_PROB (M2)

Adds a sidebar checkbox "Use calibrated probability" to `app/pages/backtesting.py` that flips `core.strategy.common.USE_CALIBRATED_PROB` for the duration of a single run, then resets it to `False` in a `finally` block so the process-global flag never leaks to other Streamlit pages sharing the server process. Wraps both run handlers — Mode A `orch.run_full()` and Mode B `run_backtest()` — since fit + backtest both execute in the Streamlit main process (only the MC backrun is a subprocess). When the checkbox is on, surfaces a status line: a green caption with the trusted-bucket count from `load_calibration_shift()`, or a warning that the toggle is a no-op until a full pipeline run produces `DATA/calibration_shift.csv` with applied buckets. Help text documents the chicken-and-egg (calibration table is produced after a backtest) and the Mode B no-op (batches fitted with the flag off lack `p_model_cal`).

## fix: signed-edge panel — NaN count, opacity formula, Plot A market line

Three correctness fixes to `_render_signed_edge_panel` in `app/pages/backtesting.py`:

1. **`n_nan_excluded` fix** — Was computing `len(all_priced_df) - n_raw`, which lumped NaN-excluded rows with dedup removals (dominated by the latter since each contract has many daily snapshots). Now counts unresolved `outcome_yes` rows directly from source. Added a fourth header metric "Dedup removed" to surface the previously hidden number. Column layout expands from 3 to 4.
2. **Plot B opacity fix** — `_opacity` condition was wrong: used `wilson_lo − realized_yes_rate` as the comparand instead of `mean_market_p`. The CI crosses zero (bin is inconclusive) iff `wilson_lo < mean_market_p < wilson_hi`. Fixed condition now correctly fades markers whose CI brackets zero.
3. **Plot A market line sort (cosmetic)** — Market trace was plotted in edge-bin order, causing the connecting line to backtrack in x. Now sorts by `mean_market_p` into a local `_market_line` before plotting; other traces unaffected.

## feat: signed-edge reliability diagram on backtesting page

Implements `temp/signed_edge_implementation_plan.md`. Adds a calibration analysis panel to the backtesting page that plots model probability vs realized YES rate (Plot A) and mean edge vs realized−market (Plot B), with Wilson 95% CI error bars and adaptive bin merging. Includes Brier score, BSS vs market, ECE, and calibration logistic regression metrics.

- **`app/pages/backtesting.py`** — Part 1: Mode B (`run_backtest`) now passes `return_all_priced=True` and stores result in `st.session_state["bt_all_priced"]` (was silently discarded). Part 2: Added `_wilson_ci`, `_build_eval_df`, `_apply_signed_edge_filters`, `_compute_edge_bins` (adaptive merge), `_panel_metrics` (Brier/BSS/ECE/calibration slope). Part 3: Added `_SIGNED_EDGE_PRESETS` constant (4 named presets + Custom). Part 4: Added `_render_signed_edge_panel` using `plotly.graph_objects` for scatter+error-bar plots. Part 5: Wired panel into results section after Trade Log with fallback info message when `bt_all_priced` absent. Part 6: Added `import math` to top-level imports.

## Signed moneyness filter — shared filter fn, live-path injection, reusable Streamlit control

Implements `temp/signed_moneyness_plan.md`. Adds `"signed"` mode (raw `m`, OTM>0/ITM<0) alongside the existing `"abs"` (symmetric, default). Critically, also fixes the moneyness filter being **dead** in the live pipeline — `build_targets` never read `config.min/max_moneyness`; this change wires it up. Note: any saved SweepConfig with `use_max_moneyness=True` will now **actually filter** (previously a silent no-op), which changes trade outputs.

- **`core/strategy/common.py`** — New `filter_by_moneyness(df, lower, upper, mode)` (shared filter logic, no Streamlit dep) + `latest_spot_as_of(btc_df, as_of)` (leak-free intraday spot helper, strict `<` cutoff). `moneyness_mode: str = "abs"` added to `RebalanceConfig`. `from datetime import datetime` import added.
- **`core/strategy/auto_reco.py`** — Wires `filter_by_moneyness` into `build_targets` (was dead param). Injects signed `moneyness` column in `recommend_trades` for the live path when absent (backtest pre-injects its own per-snapshot column). Adds `moneyness_mode` param to `recommend_trades` signature + threads it into `RebalanceConfig`. Imports `filter_by_moneyness`, `latest_spot_as_of` from `common`.
- **`sweep_config.py`** — Adds `moneyness_mode: str = "abs"` field; `to_strategy_params` passes it through (both enabled and disabled paths).
- **`core/backtesting/backtest_engine.py`** — Forwards `moneyness_mode` from `strategy_params` (alongside existing `max/min_moneyness`).
- **`app/ui_filters.py`** — NEW: reusable `moneyness_filter_controls(container, key_prefix, ...)` widget returning `{enabled, mode, min_moneyness, max_moneyness}`. Used by all three tabs.
- **`app/dashboard.py`** — Replaces manual moneyness widgets with shared control; passes `moneyness_mode` at both call sites (`recommend_trades` and backtest `strategy_params`).
- **`app/pages/polymarket_console.py`** — Replaces manual widgets with shared control; passes `moneyness_mode` into `RebalanceConfig`.
- **`app/pages/backtesting.py`** — Adds moneyness control (was absent); wires into both `strategy_params` dicts (Live Fetch + Existing Batch Files).
- **`tests/test_moneyness_filter.py`** — NEW: 13 tests covering abs/signed modes, None bounds, missing column, NaN rows, strict `<` spot cutoff, and `build_targets` dead-filter regression guard.
- `max_dte` remains dead in `build_targets` — out of scope for this change.

## Add OTM tail-mispricing (favorite–longshot) AUC test to signal diagnostics

Implements `temp/prompt.md`: a focused diagnostic that tests whether the BTC model carries residual predictive signal in the favorite–longshot bias zone the market hasn't already encoded. Restricts to OTM contracts inside a longshot market-price band (0.05–0.20), measures AUC of `model_p` (headline: >0.54 ≈ real residual signal) and of the `model_p − market_p` edge (whether the divergence is tradeable; much lower ⇒ model just shadows the market), stratified into 0.05–0.10 / 0.10–0.15 / 0.15–0.20 sub-bands so a rising AUC toward the deep tail (the favorite–longshot prediction) is visible.

- **`core/backtesting/diagnostics.py`** — New `SignalDiagnostics.tail_mispricing_report(band, sub_bands, otm_thresholds)` + `_band_stats()` helper + `TAIL_MIN_N=10` const. Runs two OTM variants (`moneyness > 0` and `moneyness > +2%`), each intersected with the price band; per variant reports headline `auc_model`/`auc_edge` and the sub-band table. AUC is `None` on `<TAIL_MIN_N` rows or a single outcome class; counts cast to `int`, AUC to `float|None`; no AUC inversion (scores already oriented). Embedded in `run_full_report()` under key `"tail_mispricing"` (and `{"available": False}` in the empty-data path, which also covers the orchestrator empty-batches build) — no orchestrator change needed. CLI (`python core/backtesting/diagnostics.py <all_priced.csv>`) prints an "OTM TAIL MISPRICING" block.
- **`app/pages/backtesting.py`** — New "📉 OTM Tail Mispricing (favorite–longshot zone)" panel under the existing DTE/Moneyness breakdowns: per variant, AUC model_p / AUC edge / Contracts metrics + a renamed sub-band table; guarded on `tail_mispricing.available`.
- **Result (real backtest, `_figarch_run/all_priced.csv`)**: headline AUC model_p ≈ 0.67 vs AUC edge ≈ 0.57 (model partly shadows market but residual signal present); AUC rises toward the 0.05 tail (0.56 → 0.61 → 0.63), consistent with the favorite–longshot thesis.

## Clarify Backtesting Signal Diagnostics tables + drop meaningless p-value

The Signal Diagnostics DTE and Moneyness breakdown tables rendered the raw diagnostics dict keys as headers (`label`, `n`, `pos`, `neg`, `rho`, `p`, `auc`), so `pos`/`neg` were unintelligible. Separately, the Spearman p-value (both the overall metric and the breakdown column) always displayed `0` — investigated and confirmed it is statistically uninformative at this sample size (N≈20k overpowers the test; p genuinely underflows to ~0 for any non-zero ρ), so it was removed rather than reformatted.

- **`app/pages/backtesting.py`** — Added a `_label_breakdown()` helper that converts each breakdown row list to a DataFrame with self-explanatory headers (`DTE Bucket`/`Moneyness Bucket`, `Contracts`, `Resolved YES`, `Resolved NO`, `Spearman ρ`, `AUC`) and drops the raw `p` column. Added per-table captions explaining that `Resolved YES`/`Resolved NO` are counts of contracts settling in/out of the money, and that both classes must be present for Spearman ρ / AUC to compute.
- **`app/pages/backtesting.py`** — Removed the overall `p-value` metric from the Signal Diagnostics summary (reflowed 4→3 columns: Spearman ρ, AUC, Observations), with a comment noting why p is omitted (overpowered test → always ~0; use effect size ρ/AUC instead).
- Display-layer only — no change to the `SignalDiagnostics` data contract (`run_full_report` still returns `spearman_pvalue` and per-bin `p`).

## Fix daily BTC spot-fallback path resolution in BacktestEngine (Streamlit CWD bug)

Backtest run from the dashboard logged `daily BTC file not found (…/app/DATA/btc_daily.csv)` and silently disabled the daily-close spot fallback. `BacktestEngine._load_btc_prices` computed the daily-CSV directory via `Path(self.btc_price_path).resolve().parent`, which resolves a *relative* path (orchestrator default `"DATA/btc_intraday_1m.csv"`) against the CWD — `app/` under Streamlit — yielding `app/DATA`.

- **`core/backtesting/backtest_engine.py::_load_btc_prices`** — Resolve a relative `btc_price_path` against `_PROJECT_ROOT` before taking `.parent` (matches the existing intraday-load resolution). Daily file now correctly found at `<root>/DATA/btc_daily.csv` regardless of CWD.

## Fix polymarket_fetcher discovery — Gamma slug format changed (data stale since 2026-05-21)

Live Fetch reported "0 new price records added" for weeks: Polymarket changed the `bitcoin-above` daily-ladder event-slug format mid-2026 from `bitcoin-above-on-{month}-{day}` to `bitcoin-above-on-{month}-{day}-{year}`. The legacy-format-only discovery returned 0 events for all post-transition dates, so nothing new was ever fetched.

- **`core/backtesting/polymarket_fetcher.py::_generate_date_slugs`** — Now returns per-date slug *groups* `[legacy, current]` (`-{month}-{day}` + `-{month}-{day}-{year}`) instead of a flat legacy-only list. Both formats are queried in one Gamma `/events` call via repeated `slug` params (call count stays one-per-day); old dates respond to legacy, new dates to current, overlap deduped by `clobTokenId` downstream.
- **`fetch_closed_bitcoin_above_markets`** — Loop consumes slug groups, passes the group as a `slug` array to `_get_json`.
- Note: the bare `-{year}` slug is the multi-day daily product (created ~7d before noon-ET expiry, daily candles). Same-day flash variants (`…-{year}-12pm-et`, `-8pm-et`, etc.; ~80-min life) are intentionally not collected.
- **Result**: backfilled `DATA/historical_contract_prices.csv` from 2026-05-21 → 2026-06-22 (2463 new records, 0 errors, contiguous daily coverage).

## FIGARCH(1,d,1) — switch from broken FIGARCH(0,d,1) to proper joint estimation

Fixes the FIGARCH implementation which was dead code: labeled FIGARCH(1,d,1) but actually FIGARCH(0,d,1) with mismatched parameters (Siu 2025 daily d=0.578 + GARCH hourly beta~0.85), causing B-M positivity violation (d-beta=-0.272<0) that silently fell back to GARCH every time.

- **`_compute_figarch_weights(d, phi, beta, trunc_k)`** — Rewritten with canonical arch library recurrence. delta_1=d, lambda_1=phi-beta+d, then delta_i=((i-1-d)/i)*delta_{i-1}, lambda_i=beta*lambda_{i-1}+(delta_i-phi*delta_{i-1}). Returns weights[0]=0, weights[1:]=lambda_k matching simulation buffer alignment.
- **`fit_garch_model`** — When `use_figarch=True`, fits `arch_model(vol='FIGARCH', p=1, q=1, dist='t')` jointly estimating phi, d, beta, omega, nu, mu. Convergence retry with relaxed tolerance before fallback. Result dict has no 'alpha' key; consumers guarded with `.get('alpha', 0.0)`. Sets `use_figarch=True` explicitly.
- **`simulate_paths`** — `alpha = garch_params.get('alpha', 0.0)`, FIGARCH buffer init uses `last_variance` instead of unconditional formula.
- **`check_variance_consistency`** — Guarded dict access `.get('alpha', 0.0)`.
- **Test 7** — Validates against `arch.univariate.recursions_python.figarch_weights_python` reference (exact match), enforces B-M positivity, weights[0]=0, hyperbolic decay. try/except ImportError wrapper for arch version resilience.
- **Docs** — `DOCS/concepts/pricing-engine.md`, `DOCS/api-reference/core/btc-pricing-engine.md`, `DOCS/concepts/architecture.md`, `DOCS/appendix/glossary.md`, `FIGARCH_REVIEW.md` all updated for FIGARCH(1,d,1) joint estimation.

Live fit results (hourly BTC, post-2019-10-01): phi=0.306, d=0.389, beta=0.456, lambda_1=+0.239 (B-M positivity satisfied). All 9 self-tests pass, Test 7 matches arch reference to machine precision.

## Backtester correctness & reliability overhaul

Fixes the backtest producing 0 trades / empty diagnostics, plus data-leakage and robustness issues identified in `BACKTESTER_REVIEW.md` / `FIX_PLAN.md`.

- **Midnight alignment (root cause of 0 trades).** CLOB candle timestamps carried second-level jitter, so the backrunner's exact-timestamp grouping scattered each expiry's ~11 strikes across ~11 per-second snapshots (~1.4 strikes each), starving the logistic curve fit (`p_model_fit` all-NaN). Now floored to midnight UTC at ingest (`polymarket_fetcher.fetch_price_history` via new `_normalize_to_midnight`, dedup keeping the point closest to midnight) and defensively at grouping (`backrunner._preprocess_work_items`). Added one-shot migration `scripts/migrate_contract_store_midnight.py` (backs up `.bak`, floors + re-dedups the existing store: 18,239→18,159 rows, now 10.0 strikes/expiry).
- **GARCH lookahead leak.** `backrunner._process_one` truncated hourly data at end-of-day, leaking up to ~24h of future returns into the volatility fit. Both hourly and intraday cutoffs are now strict `< ts_dt` (Binance bars are open-stamped, so the bar at the snapshot closes in the future).
- **Value-level model-prob fallback.** New `core.strategy.common.resolve_model_prob` (Series-returning, per-row coalesce `p_model_fit > p_real_mc > model_probability`). Wired into `auto_reco` (live trade path) and `backtest_engine`, so an all-NaN `p_model_fit` no longer shadows a populated `p_real_mc`.
- **Settlement reliability.** `backtest_engine` no longer force-refunds un-settleable positions to PnL=0 (which silently biased every aggregate). Settlement resolves a YES/NO outcome against a strike, so it requires strike-level time precision: it uses the **1-minute intraday series only** (±5m tolerance) via `_settlement_price`, recording the tier in a `settlement_source` column. Coarser bars (hourly/daily) can land on the wrong side of a near-the-money strike if BTC oscillated within the bar, so they are deliberately NOT used for settlement. The fix is to extend 1m coverage to the contract range — `data_fetcher.fetch_intraday` backfills ~2yr (its earliest-timestamp gate forces a full re-fetch when the file starts too late). The entry gate (`_expiry_is_settleable`) only admits expiries the 1m series covers; truly-uncovered positions are excluded (stake refunded, no fake settlement). Daily closes are still loaded for the leak-free `_spot_as_of` moneyness/momentum fallback (not settlement).
- **Consistent spot source.** New `_spot_as_of` (intraday close `< ts`, else prior daily close) is the single source for moneyness/momentum, matching the backrunner's S0 logic; loads `btc_daily.csv` for the fallback.
- **Crash-safe batch writes.** `_process_one` now writes batch CSVs to a temp file + `os.replace` so an interrupted worker can't leave a truncated CSV that the idempotent skip treats as complete. Emits `batch_timestamp` so loaders key chronology off data, not the folder name.
- **Diagnostics/logging.** Calibration now reads `model_prob_used` explicitly (was always skipped); FIGARCH→GARCH fallback warning deduped to once per process; backrun progress seeded with the cached count so it reaches 100%.

Validation (recent 32-day window, end-to-end): strikes/expiry 10.2, `p_model_fit` 2079/2110, 23 trades all settled, 2110/2110 outcomes resolved, diagnostics n_obs=2110. NOTE: diagnostics now surface a possibly-inverted edge signal (AUC≈0.21 on the small window) — a model-quality item to investigate, not a plumbing regression.

## Fix backtest hang + VolGate "regime unknown" spam + BTC "file not found"

Three independent defects in the backtest→auto_reco→vol_gate chain that caused the dashboard to hang, spammed `[VolGate] Regime=unknown` on every batch, and warned about missing BTC data.

- **Fix 1a — `core/backtesting/orchestrator.py::run_full`**: If no `price_df` was supplied, loads the intraday CSV once via `auto_reco.load_btc_csv()` (with absolute-path resolution) and passes it to `run_backtest`. Previously the CSV was never loaded here, forcing each batch to hit `load_btc_csv()` inside `recommend_trades`.
- **Fix 1b — `core/backtesting/backtest_engine.py`**: Builds a deterministic vol-gate-ready DataFrame (`timestamp` column + `close`, not datetime-indexed) from the already-validated `_btc_prices` index+close column, immediately after `_btc_prices` is assigned. The `_volgate_btc_df` is passed to `recommend_trades` along with the replay `current_time` as `asof_utc`. Result: the 104 MB CSV is parsed once, not 320×.
- **Fix 2 — `core/strategy/auto_reco.py::recommend_trades`**: Added optional `asof_utc` parameter (`datetime | None`). When provided (backtest path), it replaces `datetime.now()` for vol-gate evaluation. When omitted (live path), behavior is unchanged. Vol-gate log line downgraded from `logger.info` to `logger.debug` to avoid per-batch noise.
- **Fix 3 — `core/backtesting/backtest_engine.py`**: Added module-level `_PROJECT_ROOT` / `_DEFAULT_INTRADAY` constants. Constructor default now uses absolute path. `_load_btc_prices` resolves relative paths against `_PROJECT_ROOT` before the existence check, protecting against CWD-dependent failures when called from Streamlit.

Tests: 10 new tests across `TestVolgateDataFrame`, `TestAsofUtcParameter`, `TestBtcPathResolution` in `tests/test_unified_backtester.py`. Updated `tests/test_backtest_inversion.py` to mock `_expiry_is_settleable` (pre-existing working-tree change). Full suite: 61 passed.

## BTC pricing-engine fix plan (statistical validity + backtest↔live parity)

Implements `temp/PRICING_ENGINE_FIX_PLAN.md` end-to-end. Makes the backtest and live use the same model, removes/repairs inert components, and restores validation + calibration. Engine self-test (`python core/pricing/btc_pricing_engine.py`) ALL PASSED incl. new Test 6b; new regression suite `tests/test_pricing_engine_fixes.py`.

- **FIX 1 (C1) — FIGARCH actually fits live.** `batch_pricing_runner.py` and `run_full_pipeline.py` now call `fit_garch_model(..., use_figarch=advanced_features)`; previously `simulate_paths(use_figarch=True)` silently fell back to GARCH because the params dict had no `figarch_weights`. Live now matches the backtest path.
- **FIX 2 (M1/M4) — calibrated jumps everywhere, leak-free.** Backrunner computes Kou+SVCJ jump params per snapshot from the strict-`<` truncated hourly slice (`calibrate_jumps(returns=…)`, never `hourly_csv=`) and passes them into `calculate_probabilities`. Live runners now calibrate by default (mapping `lam/p_crash` → `lambda/crash_prob`, fixing a latent key-mismatch bug). Detection switched to **Lee-Mykland (2008) local bipower** (`detect_jumps_bipower` rewritten; default `detection_method="bipower"`): the old global BNS test flagged 0 jumps on large samples; MAD over-flags ~14%. Now ~0.2% jump rate, calibration converges (lam≈17.6, p_crash≈0.49).
- **FIX 3 (H2) — XGBoost directional blend removed from the hot path.** `calculate_probabilities(use_xgb_direction=True, xgb_model=…)` now raises `NotImplementedError` (the per-strike blend of a strike-agnostic P(up) broke ladder monotonicity and was never wired). Off everywhere; runner help text updated.
- **FIX 4 (H1) — regime switching wired, leak-free.** Added keyword-only `as_of` to `calculate_probabilities`, threaded into `RegimeDetector.fit_predict(now=as_of)` so the HMM refit gate uses snapshot time, not wall-clock (deterministic + leak-free in time-travel). Backrunner injects a per-snapshot detector + per-regime calibrated jump params (`build_regime_jump_params`), avoiding `calibrate_regime_jumps`' wall-clock synthetic-timestamp path.
- **FIX 5 (H3) — SVCJ vol-jump persistence under FIGARCH.** Added a decaying `vol_jump_state` (key `svcj_persist`∈(0,1), default 0.90; cap `VOL_JUMP_STATE_CAP`) added on top of the ARCH(∞) base each step; the FIGARCH recompute no longer erases the vol jump. GARCH path keeps its inline β-persisted add (byte-identical). New self-test Test 6b asserts std lift >1.03× under FIGARCH with inflated `mu_v`.
- **FIX 6 (C2) — Basel MC validation crash + config alignment.** Fixed `paths[:, -1]` → `paths` (simulate_paths returns 1-D terminal prices). `compute_mc_var` now threads the deployed flags (FIGARCH/SVCJ/skewed-t/naive prior) and `_fit_garch_on_window(use_figarch=True)` routes through `fit_garch_model`, so MC mode validates the deployed FIGARCH variant (`--garch-only` to validate the plain GARCH variant instead).
- **FIX 7 (M2) — outcome-based recalibration, default OFF.** New `fit_calibration` (walk-forward, leak-guarded, per-DTE bucket logit shift via `calibrate_logit_shift`) persists `DATA/calibration_shift.csv`; wired into `orchestrator.run_full` (`results["calibration"]`). `process_batch` writes `p_model_cal` only when `core.strategy.common.USE_CALIBRATED_PROB` is on AND a trusted shift exists. New `MODEL_PROB_CANDIDATES_CALIBRATED` + flag gate in `resolve_model_prob` — column presence is not the switch. `p_model_fit` never mutated.
- **FIX 9 (M3) — `p_rn_fit` → `p_market_fit`.** `process_batch` emits `p_market_fit` (the logistic fit to *market price*, not a risk-neutral model prob) plus `p_rn_fit` as a deprecated identical alias. Reader fallback chains in `BH_auto_reco.py` and `dashboard.py` prefer the new name. Added a measure note to `simulate_paths` docstring (physical, median-anchored; `martingale_anchor=True` is the RN switch).
- **FIX 10 (L4) — documented** the symmetric-2-param logistic / unweighted-SSE wing-fit limitation as known model risk in `fit_logistic_to_points`.
- **FIX 11 (L6) — FIGARCH positivity guard.** `fit_garch_model` rejects FIGARCH and falls back to GARCH when the ARCH(∞) weights violate B-M positivity (any negative lag), instead of silently flooring variance.
- **FIX 8 (L1) — no-op (audit correction).** The backrunner already derives a deterministic per-snapshot seed via md5; the original "constant seed 42" claim was wrong. Within-snapshot cross-expiry offset skipped (cosmetic).
