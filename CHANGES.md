## 2026-06-16 — Advanced Features toggle + CLI flag to activate Phase 1-2 model extensions

**Intent**: All pricing engine enhancements (SVCJ, skewed-t, FIGARCH, regime switching, directional XGBoost) default to `False` in code, meaning the live pipeline runs a bare GARCH(1,1)+t baseline. Add a user-facing control to activate everything at once.

**Changed**:
- `app/dashboard.py`: Added "Generation Parameters" sidebar section with "Advanced Features" checkbox (default ON). Shows read-only sub-indicators for SVCJ, Skewed-t, FIGARCH, Regime switching, Directional XGBoost.
- `scripts/pipelines/batch_pricing_runner.py`: Added `--advanced-features`/`--no-advanced-features` flags (default: on). Wires through to `simulate_paths()` as `use_svcj`, `use_skewed_t`, `use_figarch`.
- `scripts/pipelines/run_full_pipeline.py`: Same flags added. Passes `--no-advanced-features` through to batch_pricing_runner subprocess. Programmatic entry point `run_pipeline_programmatic()` accepts `advanced_features: bool = True`.
- `scripts/backtesting/prob_backrunner_engine.py`: Same flags added. `run_backtest_loop()` accepts `advanced_features` and passes all five flags to `calculate_probabilities()`.

## 2026-06-16 — Data staleness fix: absurd edges from stale intraday BTC price

**Intent**: Fix 0.95+ edge values caused by 6-month-stale intraday data used as S0 in simulations. Add staleness guard to prevent silent recurrence.

**Root cause**: `data_fetcher.py` wrote fresh data to `core/data/DATA/` but all code read from `./DATA/`. The `./DATA/btc_intraday_1m.csv` last timestamp was Dec 24, 2025 (BTC $87.6k). Current market contracts trade around $66k.

**Fixed**:
- `core/data/data_fetcher.py`: Changed `DATA_DIR` from `Path(__file__).resolve().parent / "DATA"` to `Path(__file__).resolve().parent.parent.parent / "DATA"` — data_fetcher output now goes to repo root `./DATA/`
- Copied fresh data from `core/data/DATA/` to `./DATA/`; removed stale `core/data/DATA/` directory
- `core/pricing/btc_pricing_engine.py`: Added staleness check in `load_and_prep_data()`. WARNING if >24h, ERROR if >7 days.

## 2026-06-16 — Reviewer-identified bug fixes

**Intent**: Fix 2 bugs + 6 risks found by code review, and correct a misleading docstring about Hansen's b parameter.

**Bugs fixed**:
- `jump_calibration.py`: Missing `from datetime import datetime, timezone` import would cause NameError in calibrate_regime_jumps fallback path
- `batch_pricing_runner.py`: `--recalibrate-jumps` flag was dead — calibrated_jumps never passed to simulate_paths (always used None). Same fix applied to run_full_pipeline.py programmatic path.
- `btc_pricing_engine.py`: Corrected comments that falsely claimed Hansen's b parameter standardises variance. Hansen's b normalises distribution shape only; output variance is ~nu/(nu-2), corrected externally via scale_factor. Scale_factor is required — reviewer's "double correction" finding was incorrect.

**Risks fixed**:
- `validation/__init__.py`: Now exports basel_backtest symbols
- `basel_backtest.py`: ddof=1 for ES test std; basel_traffic_light accepts n_exceed directly (avoids rounding errors)
- `jump_calibration.py`: HMM state fallback logs warning when unrecognized states appear
- `batch_pricing_runner.py`: Stale `title` variable in result row construction → uses `c['title']`
- `fit_probability_curves.py`: PROB_LOGIT_SHIFT_B changed from unvalidated -0.7 to 0.0 (identity)

## 2026-06-16 — Model-based Basel backtest rewrite (Item 9)

**Intent**: Replace historical-percentile VaR approximations with analytical GARCH(1,1) + Student-t VaR forecasts and an optional full MC SVCJ validation mode per quant auditor Item 9.

**Scope**:
- `core/validation/basel_backtest.py`: Full rewrite. Added `_fit_garch_on_window()` (GARCH(1,1)+t via arch library), `_forecast_garch_var()` (correct cumulative total-variance recursion for multi-horizon VaR), `compute_analytical_garch_var()` (rolling refit every 500h), `compute_mc_var()` (full SVCJ Monte Carlo simulation via `simulate_paths()`). Updated `run_basel_backtest()` with `mode` parameter. Kept all existing Basel traffic light + Acerbi-Szekely ES test infrastructure.
- Rolling refit every 500h with ≥500h minimum training window. Analytical mode uses Student-t quantile for VaR; MC mode uses empirical quantile from simulated paths.

## 2026-06-16 — Per-regime jump calibration (Item 7)

**Intent**: Replace hardcoded regime jump multipliers with data-driven per-regime calibration using HMM-detected bear/sideways/bull regimes.

**Scope**:
- `core/pricing/jump_calibration.py`: Added `RegimeJumpResult` dataclass + `calibrate_regime_jumps()` (~140 lines) with ≥30-jump threshold per regime, HMM state decoding, per-regime Kou MLE + SVCJ vol jump estimation
- `core/pricing/btc_pricing_engine.py`: `build_regime_jump_params()` now accepts `regime_calibrated` dict — when provided and a regime passes threshold, uses directly calibrated params instead of hardcoded multipliers

## 2026-06-16 — Calibration accuracy metrics module + backtest integration

**Intent**: Add calibration diagnostics (Brier score, reliability diagram, ECE) per quant auditor Item 6, with automatic hook-in during backtesting.

**Scope**:
- `core/validation/calibration_metrics.py` (new): `brier_score()`, `reliability_bins()`, `ece_score()`, `run_calibration_report()` with column auto-detection, `CalibrationReport` dataclass, CLI runner
- `core/validation/__init__.py`: Export calibration symbols
- `scripts/backtesting/backtest_engine.py`: `BacktestEngine._run_calibration_if_possible()` method auto-runs after `_resolve_all_priced_contracts()` when `return_all_priced=True`

## 2026-06-15 — Fix 2 type errors in dashboard.py

**Intent**: Fix a critical KeyError / wrong-column bug in `compute_realized_pnl_total()` and a fragile `dir()` scope check in the backtest tab's residual signal metrics section.

**Scope**:
- `app/dashboard.py` — line 241: `merged["outcome"]` after a merge with `suffixes=("", "_resolved")` could read the wrong column when `closed_df` already has an `"outcome"` column (resolved's column gets renamed to `"outcome_resolved"`). Now detects the correct column name.
- `app/dashboard.py` — line 2368: `'max_dte_value' in dir()` fragile scope check replaced with direct variable access (both `max_dte_value` and `use_max_dte` are always defined at module level).

## 2026-06-15 — Consolidate Polymarket console pages

**Intent**: Remove the older `polymarket_console.py` and merge its successor `polymarket_console_fixed.py` into a single canonical console. The old file had been superseded by the _fixed version which added live price resolution, order history, reconciliation, and error recovery.

**Scope**:
- `app/pages/polymarket_console.py`: Replaced with cleaned-up version of former `_fixed` (fixed broken `prob_threshold_no` call, removed duplicate sidebar parameter section, stripped corrupted emoji encoding)
- `app/pages/polymarket_console_fixed.py`: Deleted (renamed to `polymarket_console.py`)
- `CLAUDE.md`: Removed reference to `_fixed` file in directory tree

## 2026-05-29 — Remove 4 dead features from pricing engine

**Intent**: Remove momentum drift, RV blending, momentum gating, and strict_above from `core/pricing/btc_pricing_engine.py`. None of the 4 features were engaged by any caller — all 3 call sites pass default `None`/`False`.

**Scope**:
- `core/pricing/btc_pricing_engine.py`: Removed `MOMENTUM_GATE_MULT` constant, `drift_window` from `fit_garch_model()`, `initial_variance`/`use_momentum_gating` from `simulate_paths()`, `strict_above` from `get_contract_probability()`, and `drift_window`/`rv_intraday`/`rv_blend_weight`/`strict_above`/`use_momentum_gating` from `calculate_probabilities()`. Dropped validation Test 4 (Global Momentum Gating). Net removal: ~45 lines of dead code.
- `CLAUDE.md`: Replaced momentum injection / variance blending bullet points with structural mean drift and jump drift correction summary.
- `DOCS/concepts/pricing-engine.md`: Replaced momentum injection + global gating sections with structural mean drift. Removed RV blending section. Updated API signature, parameter table, and validation test count (5→4).
- `DOCS/guides/troubleshooting.md`: Removed stale `drift_window` troubleshooting tip.

**Verification**: All 4 built-in validation tests pass. All 5 downstream imports confirmed working.

## 2026-06-17 — Fix KeyError in dashboard Calibration tab when no closed positions

**Intent**: Fix `KeyError: 'position_key'` crash in dashboard Calibration tab when `positions_df` contains no CLOSED rows.

**Changed**: `core/data/positions.py` — `ensure_position_keys()` now adds `position_key` column even when input DataFrame is empty (previously returned empty DataFrame without the column, causing downstream KeyError).
