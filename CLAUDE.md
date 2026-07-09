# CLAUDE.md

STOP! BEFORE PROCEEDING READ THIS DOCUMENT COMPLETELY!

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run full live pipeline (fetch data → price markets → fit curves)
python scripts/pipelines/run_full_pipeline.py --slug-pattern "bitcoin-above-on-december-{}" --day-range 1 31

# Fetch/refresh BTC price data (run BEFORE backtesting)
python core/data/data_fetcher.py

# Run historical backrunner (time-travel MC pricing)
python core/backtesting/backrunner.py --limit 10

# Run backrunner via old path (deprecation shim — forwards to core.backtesting)
python scripts/backtesting/prob_backrunner_engine.py --skip-data-fetch --limit 10

# Run full backtesting pipeline (fetch → backrun → fit → backtest → diagnostics)
python -c "from core.backtesting import BacktestingOrchestrator; o = BacktestingOrchestrator(); print(o.run_full())"

# CLI trade recommendation from latest fitted batch
python core/strategy/auto_reco.py --bankroll 1000 --min-edge 0.06 --kelly-fraction 0.15

# Parameter sweep (grid search over strategy params)
python scripts/utilities/parameter_sweep.py --batch-dir fitted_batch_results --sweep min_edge=0.04,0.06,0.08,0.10 --limited
python scripts/utilities/parameter_sweep.py --list-params  # show all 24 sweepable params

# Launch dashboard
streamlit run app/dashboard.py

# Launch Polymarket trade console
streamlit run app/pages/polymarket_console.py

# Signal diagnostics (Spearman + AUC between edge and outcomes)
python core/backtesting/diagnostics.py path/to/all_priced.csv
python core/strategy/signal_diagnostics.py path/to/all_priced.csv  # deprecation shim

# Favorite-longshot-bias walkforward (model-free; gap+Wilson CI + buy-NO PnL per window)
python scripts/backtesting/walkforward_flb.py --bankroll 1000 --stake 10

# Edge distribution plots
python diagnostics/edge_dist_plot.py

# Run pricing engine validation tests
python core/pricing/btc_pricing_engine.py

# Vol gate (standalone risk signal)
python core/strategy/vol_gate.py --file DATA/btc_intraday_1m.csv

# Rolling-window model evaluation (Brier + VaR)
python core/validation/rolling_evaluator.py --window-days 90 --step-days 7 --horizons 1,14,28 --max-windows 40

# Bayesian posterior estimation for pricing parameters
python core/pricing/bayesian_estimation.py --strikes 90000,100000 --hours 336 --n-posterior 50

# Market-maker Stage-B paper runner (live WS feed, simulated fills; BTC data must be fresh first)
python -m market_maker.paper_runner --event-slug bitcoin-above-on-july-10-2026 --minutes 240 --tick-s 15
python -m market_maker.paper_runner --config market_maker/paper_run_config.json  # VPS/unattended mode (event_slug "auto")

# Market-maker Stage-A shadow runner (read-only REST polling, no fills)
python -m market_maker.shadow_runner --event-slug bitcoin-above-on-july-10-2026 --minutes 40 --tick-s 30

# Market-maker engine control (same protocol the mm_monitor dashboard page and deploy/ systemd units use)
python -c "from market_maker import run_control; print(run_control.start_engine())"
python -c "from market_maker import run_control; print(run_control.engine_status())"
python -c "from market_maker import run_control; print(run_control.stop_engine())"

# Market-maker VPS deploy-kit alert check (stdlib only, cron/timer-safe, always exits 0)
python scripts/mm_alert_check.py --control-dir temp/paper_run/control

# Launch MM monitor dashboard page
streamlit run app/pages/mm_monitor.py

# Run tests
python -m pytest tests/ -v

# Build documentation
mkdocs build
mkdocs serve
```

## Directory Structure

```
.
├── app/                          # Streamlit applications
│   ├── dashboard.py              #   Main 8-tab monitoring dashboard
│   └── pages/
│       ├── backtesting.py        #   Backtesting page
|       |-- polymarket_console.py #   Trade execution operator console
|       `-- mm_monitor.py         #   Market-maker Stage-B paper-run monitor + engine control
├── core/                         # Domain logic (no scripts/pipelines)
│   ├── backtesting/              #   Unified backtesting module
│   │   ├── __init__.py           #     Public API exports
│   │   ├── contract_store.py     #     CSV store for historical Polymarket prices
│   │   ├── polymarket_fetcher.py #     Gamma + CLOB API data fetching
│   │   ├── batch_loader.py       #     Batch CSV normalization & scanning
│   │   ├── backrunner.py         #     Time-travel MC pricing engine
│   │   ├── backtest_engine.py    #     Chronological backtest simulation
│   │   ├── diagnostics.py        #     Spearman/AUC/DTE signal diagnostics
│   │   ├── in_sample_oos.py      #     IS/OOS evaluation window (cutoff, M2 train_pipeline, §9 verifier)
│   │   └── orchestrator.py       #     Full pipeline: fetch → backrun → fit → backtest
│   ├── data/
│   │   ├── data_fetcher.py       #   Binance BTC data download
│   │   └── positions.py          #   CSV-based position ledger
│   ├── pricing/
│   │   ├── btc_pricing_engine.py #   GARCH+Student-t+Jump MC simulator
│   │   ├── bayesian_estimation.py #  GARCH/jump posterior distributions + credible bands
│   │   ├── fit_probability_curves.py # Logistic curve fitting
│   │   ├── jump_calibration.py   #   Kou jump params + SVCJ vol-jump estimation
│   │   └── regime_detector.py    #   3-state HMM regime switching
│   ├── strategy/
│   │   ├── auto_reco.py          #   3-stage trade recommendation pipeline
│   │   ├── BH_auto_reco.py       #   Previous auto_reco (pre-refactor backup)
│   │   ├── common.py             #   Shared types, constants (TargetPosition, TargetRole)
│   │   ├── vol_gate.py           #   BTC volatility risk gate
│   │   └── signal_diagnostics.py #   Deprecation shim → core.backtesting.diagnostics
│   └── validation/
│       └── rolling_evaluator.py  #   Rolling-window model evaluation (Brier + VaR)
|-- market_maker/                 # Binary BTC market-making (Polymarket, paper-traded; see architecture below)
|   |-- config.py                 #   MMConfig launch defaults
|   |-- contracts.py              #   Interface dataclasses/enums + VenueAdapter ABC
|   |-- market_data_client.py     #   Live CLOB WebSocket feed adapter (BookMirror)
|   |-- pricer_adapter.py         #   Sole boundary to core/pricing/btc_pricing_engine.py
|   |-- fair_value_anchor.py      #   Beuoy bankroll-credibility consensus
|   |-- quote_engine.py           #   Dalen AS / GLFT quoting
|   |-- spread_builder.py         #   Additive spread terms (incl. belly widening) -> QuoteSet
|   |-- ladder_hedger.py          #   No-arb repair + cross-strike hedging
|   |-- robustness_sizing.py      #   Kelly -> Baker-McHale -> caps sizing
|   |-- inventory_manager.py      #   Per-contract/per-ladder inventory (q, avg_cost)
|   |-- liquidity_monitor.py      #   Depth/impact/arb-half-life regime tags
|   |-- risk_controller.py        #   Vol-gate-driven risk directives
|   |-- order_lifecycle.py        #   QuoteSet -> venue actions, restart reconciliation
|   |-- paper_fill_sim.py         #   Conservative queue-behind fill simulator
|   |-- settlement_handler.py     #   12:00 ET settlement + synthetic closing fills
|   |-- state_store.py            #   SQLite/WAL state (orders/fills/inventory/markets/mid_log/...)
|   |-- pnl_report.py             #   Settlement-aware PnL snapshots + per-region markout report
|   |-- harness.py                #   PaperTradingLoop -- one-tick orchestration + settle/restart
|   |-- run_control.py            #   Stdlib start/stop/status control-file protocol
|   |-- shadow_runner.py          #   Stage-A read-only REST runner + resolve_event/resolve_next_event
|   `-- paper_runner.py           #   Stage-B live-WS paper runner (VPS deployment target)
|-- deploy/                       # Systemd deployment kit for the Stage-B paper runner (see CLAUDE.md below)
|   |-- mm-paper.service          #   Engine unit template
|   |-- mm-datafetch.service/.timer #  BTC data refresh every 30 min
|   |-- mm-alert.service/.timer   #   Fault-check + webhook alert every 5 min
|   `-- README.md                 #   Install/runbook/72h acceptance test
├── scripts/                      # Executable scripts & pipelines
│   ├── backtesting/
│   │   ├── backtest_engine.py    #   Deprecation shim → core.backtesting.backtest_engine
│   │   ├── backtest_montecarlo_sim.py # Shuffle tests (expiry-only & decile-conditioned)
│   │   ├── prob_backrunner_engine.py  # Deprecation shim → core.backtesting.backrunner
│   │   └── walkforward_flb.py     #   Standalone favorite-longshot-bias walkforward (gap+Wilson CI, buy-NO PnL)
│   ├── pipelines/
│   │   ├── run_full_pipeline.py  #   Full pipeline: fetch → price → fit
│   │   └── batch_pricing_runner.py   # Live Polymarket batch pricing
│   ├── utilities/
│   │   ├── parameter_sweep.py    #   Grid search over strategy params
│   │   ├── plot_batch_curves.py  #   Probability curve plots
│   │   └── aggregate_old_batch_data.py # Legacy data aggregation
│   ├── migrate_db.py
|   |-- migrate_contract_store_midnight.py # One-shot: floor store dates to midnight + re-dedup
|   `-- mm_alert_check.py         # Stdlib-only market-maker fault/alert check (deploy/mm-alert.timer)
├── polymarket/                   # CLOB execution layer
│   ├── accounting.py             #   Collateral and fill tracking
│   ├── intent_builder.py         #   auto_reco → OrderIntent conversion
│   ├── execution_gateway.py      #   Validation and submission
│   ├── provider_polymarket.py    #   Polymarket CLOB/API provider
│   ├── models.py                 #   Dataclasses (OrderIntent, Submission, etc.)
│   ├── db.py                     #   SQLite persistence
│   ├── ingest.py                 #   Market data ingestion
│   ├── metrics.py                #   Performance metrics
│   ├── reconcile.py              #   Position reconciliation
│   ├── date_utils.py             #   Date/time helpers
│   └── intents.db                #   SQLite database file
├── tests/                        # Test suite
│   ├── test_auto_reco_refactor.py
│   ├── test_backtest_inversion.py
│   ├── smoke_test_console_logic.py
│   ├── verify_dashboard_refactor.py
│   ├── verify_fallback_warning.py
│   └── debug_reco.py
├── sweep_config.py               # SweepConfig dataclass (at root — shared import)
├── deprecated/                   # Old pricing/backtest engines (not used)
├── diagnostics/                  # Edge distribution analysis
├── DOCS/                         # MkDocs documentation site
├── DATA/                         # BTC price CSV files
├── mkdocs.yml                    # MkDocs configuration
└── CLAUDE.md
```

## Architecture

### Core Pricing Engine (`core/pricing/btc_pricing_engine.py`)

FIGARCH(1,d,1)/GARCH(1,1) + Skewed-t/Student-t + SVCJ (Kou Double Exponential with correlated volatility jumps) Monte Carlo simulator, regime-conditional via a 3-state HMM. The high-level entry point is `calculate_probabilities()` which accepts strikes and days-to-expiry, returns `{strike: probability}` dict. Key design decisions:
- **Measure (FIX 9/M3)**: Default is the *physical* measure, log-mean anchored (E[log S_T] = log S0); the median coincides only for a symmetric log distribution. Uses μ=0 naive prior + jump-drift compensator, no diffusion convexity correction — NOT risk-neutral. `martingale_anchor=True` corrects the jump compensator only; the diffusion Jensen term is NOT subtracted.
- **Structural mean drift**: Uses long-term fitted GARCH mean (mu) with per-path clamping to ±0.25 × path-specific sigma_day
- **Multi-jump aggregation**: Poisson compound jumps per day, Gamma-distributed magnitudes
- **Jump drift correction**: Expected jump drift subtracted from mean (legacy log-mean compensator by default)
- **SVCJ persistence (FIX 5/H3)**: Under FIGARCH the vol jump is carried by a decaying `vol_jump_state` (the ARCH(∞) base has no β to persist it); under GARCH β persists it inline. State capped by `VOL_JUMP_STATE_CAP`, decay `SVCJ_PERSIST∈(0,1)`.
- **Regime switching (FIX 4/H1)**: Wired and leak-free — pass `regime_detector` + `as_of`; the HMM refit gate uses `as_of` (snapshot time), never wall-clock, so time-travel backtests are deterministic and leak-free.
- **XGBoost directional drift shift (FIX 3/H2, RE-ENABLED)**: The old per-strike additive blend (`0.7·p_mc + 0.3·p_xgb`) corrupted the ladder (added a strike-agnostic P(up) onto strike-specific probs → broke monotonicity) and was removed. It is now re-enabled as a **distribution-level drift shift**: `apply_xgb_drift_shift()` converts the XGBoost P(up) into a single, strike-agnostic constant shift of the simulated terminal paths (`paths *= exp(Δ_H)`), applied **once before** the per-strike loop, so the ladder stays monotone by construction. `Δ_H` is solved by **empirical-CDF inversion** (`Δ_H = −quantile(log_ret, 1−p_target)`, exact on the non-Gaussian distribution), where `p_target = 0.5 + λ·(p_up−0.5)`. Controls: `XGB_TILT_LAMBDA` (λ, default **0.0** = inert; production value set by calibration), `XGB_MAX_SHIFT_FRAC` cap, `XGB_P_BASE_GUARD` (skip deep-skew snapshots), `XGB_DTE_BUCKETS` (per-DTE-bucket models {≤7,7–14,14–30}d, train horizon = bucket midpoint). **Skipped under `martingale_anchor=True`** (the tilt is a physical-measure view) and **gated to DTE ≤30d**. Off by default everywhere: engine `use_xgb_direction=False`; backrunner/live behind `--use-xgb` (+ `--xgb-lambda`). Needs `DATA/macro_daily.csv` for real directional signal (BTC-only features degrade toward neutral). Per-snapshot, leak-free in backtest (`backrunner` trains per (UTC-date, bucket) on the strict-`<` truncated daily returns + `< snapshot_time` macro slice). IS/OOS: walk-forward, NOT frozen (only M2 is frozen).
- **Jumps (FIX 2/M1)**: Data-calibrated everywhere (live + backtest) via Lee-Mykland bipower detection (`jump_calibration.calibrate_jumps`, default `detection_method="bipower"`); in backtest, calibrated per-snapshot on the leak-free truncated slice (`returns=`, never `hourly_csv=`).

### Column Name Precedence Conventions

The codebase uses flexible column detection with `_pick_column()` / `get_column()` fallback chains. Critical conventions:
- **Model probability**: `p_model_fit` > `p_real_mc` > `model_probability` (default). When outcome-based recalibration is enabled — `core.strategy.common.USE_CALIBRATED_PROB=True`, default OFF — the chain becomes `p_model_cal` > `p_model_fit` > … (FIX 7/M2). Column presence is NOT the switch; the flag is.
- **Market price**: `market_price` > `market_pr` > `Polymarket_Price`
- **Expiry**: `expiry_key` (derived from `expiry_date` string), `T_days` as float
- **Edge**: computed as `model_prob - market_price` (YES edge) or `market_price - model_prob` (NO edge)
- **Market-fit probability (FIX 9/M3)**: `p_market_fit` > `p_rn_fit` (deprecated alias) > `risk_neutral_prob_fit` > `risk_neutral_prob`. This is the logistic fit to the *market price* (or `risk_neutral_prob` when `use_rn_prob=True`), NOT a risk-neutral model probability — hence the rename from `p_rn_fit`.

### Data Pipeline

Two output paths produce batch CSVs in the same format (`slug, strike, market_price, p_real_mc, T_days, date, expiry_date`):
1. **Live mode**: `scripts/pipelines/batch_pricing_runner.py` → `batch_results/<timestamp_UTC>/batch_results.csv` → `fitted_batch_results/<timestamp_UTC>/batch_with_fits.csv`
2. **Backtest mode**: `core/backtesting/backrunner.py` → `backtested_probabilities/unfitted/batch_<YYYYMMDD_HHMMSS>.csv` → `backtested_probabilities/fitted/batch_<YYYYMMDD_HHMMSS>/batch_with_fits.csv`

The backtest mode does "time-travel" — at each timestamp it truncates BTC data to what was available then (strict `< ts` cutoff on both hourly and intraday; Binance bars are open-stamped so `<=` would leak the bar that closes after the snapshot), runs fresh MC simulations, and saves results. Market contract prices are floored to midnight UTC — enforced at ingest (`polymarket_fetcher._normalize_to_midnight`) and defensively at grouping (`backrunner._preprocess_work_items`) — so each expiry's full strike ladder lands in one snapshot for the curve fit. `fit_probability_curves.py` runs logistic curve fitting on both paths (called internally by both runners).

### Unified Backtesting Module (`core/backtesting/`)

Consolidates previously scattered backtesting logic into a single module. The orchestrator chains: fetch historical prices → time-travel MC pricing → curve fitting → backtest simulation → signal diagnostics.

**Components**:

- `ContractPriceStore` — Disk-backed CSV store for historical Polymarket contract prices (7-column schema: `slug, clobTokenId, date, price, resolution, strike, expiry_date`). Deduplication on `(clobTokenId, date)` composite key.
- `polymarket_fetcher.py` — Fetches closed `bitcoin-above` markets from Gamma API (`/markets?closed=true`) and daily price candles from CLOB API (`/prices-history`). Handles stale-data refresh (re-fetches unresolved contracts with >1 day gap) and rate limiting (200ms delay, exponential backoff on 429).
- `batch_loader.py` — Canonical batch CSV normalization (`model_probability → p_model_fit`, timestamp parsing). Extracted from duplicated dashboard copies.
- `BackrunnerEngine` — Time-travel MC pricing loop. At each historical timestamp, truncates BTC data, runs `calculate_probabilities()` for active contracts, writes batch CSVs to `unfitted_dir`. Disk-native streaming (no in-memory accumulation). Idempotent (skips existing files). Per snapshot it computes **leak-free calibrated jump params** (bipower, from the truncated slice via `returns=`) and constructs a **leak-free regime detector** (`as_of=ts` threads into the HMM refit gate); both shared across the snapshot's expiry groups. The **GARCH/FIGARCH fit and S0 are also shared across expiry groups**: `_process_one` passes a per-snapshot `garch_cache` (dict keyed on the effective post-horizon-gate `use_figarch` flag) and a precomputed `s0_override` into every `calculate_probabilities` call, so the FIGARCH MLE and data load run once per snapshot instead of once per expiry group (byte-identical output — same slice ⇒ same fit). XGBoost is off (FIX 3).
- `BacktestEngine` — Chronological backtest: sorts batches → settles expired positions → executes trades via `recommend_trades()`. Tracks all priced contracts for shuffle tests. Moved from `scripts/backtesting/`.
- `SignalDiagnostics` — Spearman/AUC between edge and outcome, with DTE and moneyness breakdowns. Returns structured dict for dashboard consumption. Absorbed from `core/strategy/signal_diagnostics.py`. Also exposes `tail_mispricing_report()` (under report key `"tail_mispricing"`): a favorite–longshot test that, within the OTM price band (default 0.05–0.20), reports AUC of `model_p` and of the `model_p − market_p` edge vs realized outcome, for two OTM filters (`moneyness > 0` and `> +2%`), stratified into 0.05–0.10 / 0.10–0.15 / 0.15–0.20 sub-bands. Rendered on the dashboard Backtest tab.
- `BacktestingOrchestrator` — Single entry point chaining all stages. `run_full()` returns dict: `{new_records, unfitted_dir, fitted_dir, trades_df, equity_df, all_priced_df, diagnostics, calibration}`. The `calibration` entry (FIX 7/M2) is the per-DTE-bucket logit-shift table fit walk-forward from backtest outcomes and persisted to `DATA/calibration_shift.csv`; it does NOT change edges unless `USE_CALIBRATED_PROB` is flipped on.

**In-sample / out-of-sample window (`core/backtesting/in_sample_oos.py`)**: A global IS/OOS evaluation window for the Backtest tab, driven off `all_priced_df`. A single midnight-UTC cutoff (default = ~70/30 unique-contract split) partitions contracts by `snapshot_time` (IS = priced before; OOS = priced on/after); every panel respects it. Key design: the **only** cutoff-frozen component is the **M2 logit-shift B** — all BTC-process components (GARCH/FIGARCH, jumps, regime) are already per-snapshot walk-forward and are NOT frozen; XGBoost is excluded. `train_pipeline(cutoff, all_priced_df)` fits `m2` only on the **settlement-based** IS training set (`settlement_time < cutoff`, the §9 leak guard — settlement derived from `expiry_date` via the engine's 12:00-ET rule) and caches to `DATA/is_oos_cache/cutoff_<date>/` (manifest + IS-only `calibration_shift.csv`, never the global file). `load_or_train` is **load-only in OOS mode** — a cache miss or stale fingerprint (cutoff + n_is_train + is_label_max_ts + code_version + params_hash) raises; OOS never refits. OOS rows get `p_model_cal` overlaid from the cached IS B. `guarded_filter` blocks outcome-conditioning queries in OOS (§7); `small_sample_state` suppresses summary stats below N=200 (§8); `verify_oos_leak_free` asserts (on ≥3 sampled OOS contracts) that the M2 training label max-timestamp precedes the contract's pricing time, plus a pytest-only BTC-truncation arm. The Backtest tab recomputes `SignalDiagnostics` per-window (instead of the precomputed full-frame dict) and windows the trade-sim panels by trade entry time.

**Deprecation shims**: Old files (`scripts/backtesting/prob_backrunner_engine.py`, `scripts/backtesting/backtest_engine.py`, `core/strategy/signal_diagnostics.py`) emit `DeprecationWarning` and forward to the new module. Parameter sweep imports updated to use `core.backtesting.*` paths.

**Historical contract prices**: Fetched from Polymarket APIs (not manually built CSV). Store at `DATA/historical_contract_prices.csv`, one row per `(clobTokenId, midnight-UTC date)`. BTC data fetch (`data_fetcher.py`) must be run manually — no longer auto-triggered by backrunner. A one-shot repair for stores written before midnight-flooring: `python scripts/migrate_contract_store_midnight.py` (backs up `.bak`, floors + re-dedups).

**Settlement reliability**: The backtest only enters contracts whose 12:00-ET expiry an intraday print can later settle (`BacktestEngine._expiry_is_settleable`); un-settleable positions are excluded (stake refunded, no fake PnL=0 settlement). Settled trades carry a `settlement_source` tag. Spot for moneyness/momentum comes from `_spot_as_of` (intraday close `< ts`, else prior daily close — the same leak-free logic as the backrunner's S0).

**Model probability resolution**: `core.strategy.common.resolve_model_prob(df)` returns a per-row coalesced Series (`p_model_fit > p_real_mc > model_probability`), so an all-NaN fitted column no longer shadows a populated raw-MC column. Used by both `auto_reco` and `backtest_engine`. When `USE_CALIBRATED_PROB=True` (FIX 7/M2, default OFF) the precedence prepends `p_model_cal`.

**Outcome-based recalibration (FIX 7/M2)**: `fit_probability_curves.fit_calibration(all_priced_df)` fits a per-DTE-bucket logit shift `B` (Platt-style single-parameter MLE, `calibrate_logit_shift`) walk-forward (train on the earliest `train_frac` of snapshots, leak-guarded) and persists `DATA/calibration_shift.csv`. `process_batch` then writes `p_model_cal = sigmoid(logit(p_model_fit) + B_bucket)` ONLY when `USE_CALIBRATED_PROB` is on and a trusted shift table exists (`applied=True`, `n_obs ≥ 200`/bucket). `p_model_fit` is never mutated. The dashboard Backtest tab (`app/pages/backtesting.py`) exposes a "Use calibrated probability" sidebar checkbox that flips `USE_CALIBRATED_PROB` for the duration of one run (Mode A `run_full` / Mode B `run_backtest`) and resets it to `False` in a `finally` block, so the process-global flag does not leak to other Streamlit pages.

### Strategy Layer (`core/strategy/auto_reco.py`)

Refactored to a **3-stage pipeline: Target → Delta → Action**:

**Stage 1 — Target**: Generate ideal position targets from edge analysis
- Filter by min/max price, model probability, DTE, moneyness
- Generate YES/NO candidates based on edge ≥ min_edge (or prob_threshold mode)
- Score candidates = edge × stability_penalty × staleness_multiplier
- Per-expiry selection: at most 1 sign change, only YES→NO structure allowed
- Also generates EXIT signals for existing positions below hysteresis threshold

**Stage 2 — Delta**: Compute required position changes
- Compare targets against existing positions
- Calculate add/reduce/exit sizing via fractional Kelly
- Kelly formula: `f* = (p-q)/(1-q)` for YES, `(q-p)/q` for NO, then × multipliers, capped at 0.30

**Stage 3 — Action**: Apply risk constraints and produce final trades
- Per-expiry cap scaling, total-cap scaling
- Net delta limit enforcement (±max_net_delta_frac)
- Correlated position penalty: `1/(1 + penalty × (n-1))` per expiry+direction group
- Exit hysteresis: edge must drop below `(entry_edge - hysteresis)` to exit, preventing churn

Key data types live in `core/strategy/common.py`:
- `TargetRole` enum: ENTRY, EXIT, HOLD_SAFETY, NEUTRAL
- `TargetPosition` dataclass: ideal state for a single contract
- Shared constants for caps, staleness windows, min trade sizes
- `filter_by_moneyness(df, lower, upper, mode)` — shared moneyness filter used by `build_targets`. `mode="abs"` applies bounds to `|m|` (symmetric, default); `mode="signed"` applies bounds to raw `m` (OTM>0, ITM<0, bounds may be negative). Live batches lack a `moneyness` column; `recommend_trades` injects it from the live BTC spot via `latest_spot_as_of` when absent (backtest pre-injects its own leak-free per-snapshot column which is preserved). `RebalanceConfig.moneyness_mode` carries the mode through the pipeline.
- **Reusable Streamlit moneyness widget**: `app/ui_filters.moneyness_filter_controls(container, key_prefix)` — used by dashboard, console, and backtesting tab.

### Vol Gate (`core/strategy/vol_gate.py`)

Standalone risk module that gates trading based on BTC realized volatility:
- Computes realized vol over 15m and 60m windows from intraday data
- Ranks current vol against trailing baseline (percentile)
- Classifies regime: `normal`, `high`, or `extreme`
- Returns `VolGateResult` with:
  - `allow_new_entries`: False in extreme regime
  - `edge_add_cents`: Additional edge required in high vol
  - `kelly_mult`: Multiplier for Kelly fraction (reduced in high vol)
  - `shock`: True if sudden spike detected

Integrated into `auto_reco.py` Stage 3 as final risk gate.

### Parameter System (`sweep_config.py`)

`SweepConfig` dataclass is the single source of truth for all 24 strategy parameters. Used by both `app/dashboard.py` sidebar and `scripts/utilities/parameter_sweep.py`. Parameters include: edge thresholds, Kelly fraction, bet limits, price/probability filters, DTE/moneyness filters, probability threshold mode, stability penalty, fixed stake, correlation penalty. Located at repo root for shared import convenience.

### Backtesting (`core/backtesting/backtest_engine.py`, was `scripts/backtesting/backtest_engine.py`)

`BacktestEngine` class runs chronological simulation:
1. Sort batches by timestamp
2. Per batch: settle expired positions (check BTC price at 12:00 ET expiry) → execute new trades via `recommend_trades()`
3. Track all priced contracts (not just taken trades) in `_all_priced_contracts` for decile-conditioned shuffle tests
4. Output: trades_df (consolidated entries), equity_df (snapshots), all_priced_df (optional)

### Shuffle Tests (`scripts/backtesting/backtest_montecarlo_sim.py`)

Two modes:
- **Expiry-only**: Shuffles outcomes within each expiry among taken trades only
- **Decile-conditioned** (`--all_trades`): Uses all priced contracts as outcome pool, conditions on edge decile bins, cascade pool fallback (snapshot+expiry+decile → expiry+decile → expiry → global)

### Dashboard (`app/dashboard.py`)

Streamlit app with 8 tabs: Curves & Edges, Stability, Volatility & Regimes, Calibration, Recommendations, Positions, Backtest, Historical Stability. Loads batch CSVs from directory scan or upload. All strategy sliders in sidebar mirror SweepConfig defaults.

### Live Trading (`polymarket/`)

- `intent_builder.py`: Converts auto_reco DataFrame to OrderIntent objects with deterministic ID generation and share rounding (rounds DOWN to 2 decimals)
- `execution_gateway.py`: Validates intents against available collateral, submits via provider
- `provider_polymarket.py`: Polymarket CLOB API integration for order submission and fills
- `accounting.py`: Tracks collateral, fills, and settlement status
- `ingest.py`: Market data and price ingestion from Polymarket
- `metrics.py`: Trade performance and P&L metrics
- `reconcile.py`: Position reconciliation between local ledger and Polymarket
- `db.py`: SQLite persistence for runs, intents, submissions, account states
- `models.py`: Dataclasses — OrderIntent, Submission, AccountState, etc.
- `app/pages/polymarket_console.py`: Streamlit operator workflow (generate → approve → submit → monitor)

### Market-Maker Stage-B Paper Runner (`market_maker/`)

Binary BTC market-making against Polymarket `bitcoin-above` ladders -- paper-traded only (no live orders sent). Components, one-liner each:

- `config.py` -- `MMConfig` launch defaults (spread terms, caps, refresh intervals).
- `contracts.py` -- interface dataclasses/enums shared by every component (`QuoteSet`, `RiskDirective`, `Side`, `VenueAdapter` ABC, ...).
- `market_data_client.py` -- `PolymarketFeedAdapter`: live CLOB WebSocket client, one connection per ladder; feed health is WS ping/pong liveness (`healthy()`), NOT message arrival (quiet books go silent 80s+).
- `pricer_adapter.py` -- sole boundary to `core/pricing/btc_pricing_engine.py`; builds a `PricerSnapshot` over a densified strike grid.
- `fair_value_anchor.py` -- Beuoy bankroll-credibility consensus probability + Bayes bankroll update.
- `quote_engine.py` / `spread_builder.py` -- Dalen AS/GLFT quoting in log-odds space, composed into a `QuoteSet` per market via six additive spread terms: (1) base arrival markup (audit-only, embedded in the proposal), (2) adverse-selection buffer (`eps_base` + directive `eps_add`), (3) inventory skew (audit-only, embedded in the proposal), (4) robust widening (`sqrt(sigma2)` + credibility widening), (5) wing/tail widening outside `belly_band`, (6) belly widening inside it -- exact complements, exactly one of wing/belly fires per quote; belly charges a flat base plus a slope past `belly_widen_free_days`, since the belly is the pricer's softest region per `temp/suitability.md`. Confidence-tier day boundaries (`MMConfig.tier_full_max_days`/`tier_degraded_max_days`, 7d/14d) gate `wing_widen_scale` only -- no other quoting/sizing effect.
- `ladder_hedger.py` -- mandatory PAV isotonic no-arb repair across the ladder before any order goes out, PLUS live vertical hedging (2026-07-08 fix batch): after fills routing each tick, `vertical_hedges` emits HedgeRecommendations for strikes past `vertical_target_frac*q_max`; `hedge_offsets_by_market(recs)` converts them to the market_id-keyed HedgeState that `inventory_manager.set_hedge_state`/`net_band_exposure` consume (the `(expiry_key, bucket)`-keyed second return is audit-only, never crosses a module boundary). NEXT tick the pending recs skew the neighbor's post-repair quote sizes under an exhaustive side-scale `max_price` rule (BUY_YES applies iff `bid_price <= max_price`; BUY_NO iff `1 - ask_price <= max_price`), never resurrecting a directive-suppressed side; applied/skipped recs land in `harness.hedge_journal`. `enable_beta_hedge` has a real (default-off, asserted-inert) call site; enabling it for real still requires threading per-market sigma_b into the hedge stage.
- `robustness_sizing.py` -- Kelly -> Baker-McHale shrinkage -> joint-ladder/bankroll caps -> fractional-Kelly. The DEPTH cap is live: the harness passes `last_liquidity` into `size_ladder`, so quote sizes are bounded by realized depth per side (`SizingCap.DEPTH` in the audit when binding).
- `inventory_manager.py` -- per-contract/per-ladder position (q, avg_cost); `mark(now)` is called every tick by the harness so `age_weighted_holding`/R3 accrue between fills; `harness.py` (`PaperTradingLoop`) is the one-tick orchestrator wiring all of the above.
- `liquidity_monitor.py` / `risk_controller.py` -- depth/impact/arb-half-life regime tags feeding vol-gate-driven risk directives (PULL/widen/cancel-all). `RiskDirective.kelly_mult` is journaled-only (decision 2026-07-08) -- deliberately not applied to sizing; sizing protection is Baker-McHale + caps + fractional-c, and the vol gate acts on quotes via eps_add + PULL instead. Fair-value staleness is a first-class trigger (`RiskTrigger.FAIR_VALUE_STALE`): consensus recomputes only when EVERY market has a mid, so the harness tracks the last recompute time and the controller widens past `MMConfig.fv_max_age_s` (300s) and PULLs past 2x -- and `_x_hist` appends are skipped on non-recompute ticks so sigma_b cannot decay to its floor on a frozen anchor. The Beuoy bankroll `frozen` flag auto-clears after `bankroll_unfreeze_clean_ticks` (20) consecutive clean BEUOY recomputes (one degenerate tick no longer kills credibility learning for the expiry).
- `order_lifecycle.py` -- `QuoteSet`+`RiskDirective` -> venue actions with minimal churn; restart reconciliation (LIVE->UNKNOWN->reconciled).
- `paper_fill_sim.py` -- conservative queue-behind fill model (no live orders; fills are simulated from the WS trade-print stream).
- `settlement_handler.py` -- 12:00 ET settlement instant; a market resolves YES only if spot is **strictly above** the strike (venue-confirmed rule, matches the backtester's `resolve_outcome_yes`); emits a synthetic closing fill through the normal fill channel so `fold(fills) == inventory` holds through resolution.
- `state_store.py` -- SQLite/WAL `MMStateStore` (orders/fills/inventory/quotes/pnl/settlements/**markets** registry/**mid_log** -- markets registry persists `{market_id: (expiry_key, strike)}` so a restarted process can find a previous run's ladder to settle; mid_log durably appends per-tick, per-market YES mids and backs `pnl_report.markout_report` via `mid_at_or_after`). Fill cost-basis price is YES-scale for both sides (never complemented for BUY_NO) -- `fold_fills_to_inventory` matches `inventory_manager`'s convention.
- `pnl_report.py` -- settlement-aware PnL snapshots (realized = cash + q*avg_cost, folded from the durable fills table every tick) plus a pure `markout_report` function: per-region (belly/wing/unknown) x per-TTE-bucket (0-1d/1-2d/2-4d/4d+) x per-horizon (60s/600s/3600s, disjoint join windows -- each horizon's window is capped at the next horizon's start, exclusive upper bound) fill markout, measuring whether the pricer's belly bias bleeds through the Beuoy anchor into realized fill quality; each cell also reports `n_attempted` (eligible fills looked up, whether or not a mid was found) alongside `n` (successful hits) so "no fills" is distinguishable from "mids missing". `paper_runner.py` writes it to `<out_dir>/markout_report.json` every `PER_MARKET_SNAPSHOT_EVERY_N_TICKS` ticks; `app/pages/mm_monitor.py` renders it read-only, adding a computed coverage column (`n`/`n_attempted`). The report is a rolling 7-day window (`MARKOUT_LOOKBACK_S`), and `paper_runner.py` prunes `mid_log` to the same window (`state_store.prune_mid_log`) right after writing the report each cycle -- snapshot the state-db before this window rolls off if full-history markout analysis is ever needed, since pruned mids cannot be recomputed. The `quotes` and `liquidity_windows` tables are pruned on the same cadence to `MMConfig.quotes_retention_s` (14d default); `ladder_state` and `liquidity_windows` are persisted at the per-market snapshot cadence (2026-07-08 fix batch -- both tables previously had no producers). Adverse-selection fill marking (`mark_fills`, fills-table `mid_p1m/p10m/p1h`) was removed; those columns are legacy-NULL, superseded by the mid_log markout report.
- `run_control.py` -- stdlib-only start/stop/status control-file protocol (used by both `app/pages/mm_monitor.py` and `deploy/`'s systemd units).
- `shadow_runner.py` -- Stage-A read-only REST-polling runner (fill-free by construction) plus `resolve_event`/`resolve_next_event` (shared with `paper_runner.py`) and `CachedEngine` (re-price cache over the real pricing engine, GARCH cache refit every `garch_refit_s`, default 6h).
- `paper_runner.py` -- Stage-B runner: same wiring as Stage-A but fed from the live WS adapter, so trade prints reach the fill simulator. This is the VPS deployment target.

**Control-file protocol** (default dir `temp/paper_run/control/`, overridable via `--control-dir`): `mm_paper.pid` (this run's PID, removed on clean exit), `mm_paper.stop` (touch to request a graceful stop; optionally PID-stamped so a stale file from a prior run in the same dir can't kill a fresh one), `mm_paper.starting` (O_CREAT|O_EXCL start lock), `current_run.json` (pid/argv/config/out_dir + `exit_reason` once ended -- never deleted, always points at the latest run). Per-run `heartbeat.json` under `<out_dir>/` is rewritten every tick and carries `tick`, `feed_healthy`, `btc_data_age_s`, `feed_restarts`, `tick_s`, `reprice_s` (the last two size `run_control.engine_status()`'s STALLED threshold, since a reprice tick blocks the loop for minutes), plus `resume_discrepancies` (count of position deltas found by the resume protocol's venue reconciliation -- when nonzero the runner holds `manual_override=True` until the first clean tick) and `bankroll_frozen` (Beuoy anchor degeneracy state). `engine_status()` derives one of `RUNNING`/`STARTING`/`STALLED`/`STOPPED`/`CRASHED` from these files.

**Resumable state + restart protocol**: `--state-db <path>` (VPS config: `market_maker/mm_paper_state.db`) makes `MMStateStore` persistent across restarts instead of per-run. A pre-existing db at that path triggers, in strict order: `store.mark_all_live_orders_unknown()` -> `loop.restart(now)` (replays the full fills table into a fresh `InventoryManager`) -> `loop.settle(now, catch_up=True)` (registry-merge catch-up: the persisted `markets` registry, which may still carry a previous event's ladder, is merged under the current run's markets, so a previous event's still-open positions get settled before quoting resumes). **Invariant**: `settle(catch_up=True)`'s post-catch-up inventory sync is unfiltered by design and MUST only run after `restart()` on a resumed store -- calling it on a fresh `InventoryManager` with a non-empty fills table would desync in-memory inventory from the store (see the code comment on `harness.PaperTradingLoop.settle()`).

**Exit-code convention** (`paper_runner.run()`): `ladder_settled` (ladder fully settled + 30min grace) or `settlement_timeout` (`--max-settlement-wait-h`, default 26h, elapsed with the ladder still not terminal) -> **42**, an expected rollover signal -- `deploy/mm-paper.service` sets `RestartForceExitStatus=42` so systemd restarts and (with `--event-slug auto`) rolls onto the next event. `feed_dead` (feed watchdog gave up after a rebuild) or `tick_errors` (`--max-consecutive-tick-errors` consecutive failures) -> **1**, a normal supervised restart. `completed`/`stop_file`/`sigterm`/`sigint` -> **0**, no restart.

**Auto event mode**: `--event-slug auto` calls `shadow_runner.resolve_next_event(now, lead_days)`, which probes the Gamma API for `bitcoin-above-on-<date>` candidates (both zero-padded and unpadded day forms -- the venue's convention is unverified) out to `lead_days + 4` days and picks the first with a real quoting window past `near_resolution_pull_hours + 12h`. Network fetches (`resolve_event`/`resolve_next_event`) retry with exponential backoff (5 attempts, 2s->30s; 404 is not retried).

**Staleness guard**: every tick, `paper_runner.py` freshly stats `DATA/btc_intraday_1m.csv`; if its age exceeds `--btc-stale-max-s` (default 7200s) or the file is missing, the tick runs with `manual_override=True`, which pulls all quotes and cancels resting orders until fresh data lands. `DATA/btc_intraday_1m.csv` is refreshed by a separate cron/timer (`deploy/mm-datafetch.timer`), never by the runner itself.

**The `deploy/` kit** (systemd unit templates + `scripts/mm_alert_check.py`): `mm-paper.service` runs the engine (`RestartForceExitStatus=42` for rollover, `TimeoutStopSec=900` since SIGTERM is only observed between ticks); `mm-datafetch.service`+`.timer` refresh BTC data every 30 min; `mm-alert.service`+`.timer` run `scripts/mm_alert_check.py` every 5 min -- a stdlib-only, always-exits-0 script that pages a generic JSON webhook (`$MM_ALERT_WEBHOOK`) on engine CRASHED/STALLED, a feed-unhealthy streak >15min, stale BTC data (>2x `--btc-stale-max-s`), low disk, a `settlement_timeout` exit while stopped, resume position discrepancies (`resume_discrepancies` > 0 in heartbeat), or a frozen Beuoy bankroll (`bankroll_frozen`), de-duped 6h per alert key via `temp/paper_run/control/alert_state.json`. It also sends one daily heartbeat message (state/tick/fills/disk one-liner) at the first check at/after 08:00 UTC (`$MM_HEARTBEAT_HOUR_UTC` overrides the hour, `$MM_HEARTBEAT_DISABLE=1` turns it off; tracked as `heartbeat_last_date` in the state file, not subject to the 6h de-dupe), so webhook silence is distinguishable from a dead alert pipeline. An optional `mm-monitor.service` (documented in `deploy/README.md` section 2, no template file) serves `app/pages/mm_monitor.py` bound to loopback for viewing through an SSH tunnel (`ssh -L 8502:127.0.0.1:8502 <vps>`). See `deploy/README.md` for the install walkthrough and the 72h VPS acceptance test procedure.

### Position Tracking (`core/data/positions.py`)

CSV-based (not SQLite) position ledger. Key functions:
- `load_positions()`: Reads `positions.csv`, auto-closes expired (past 12PM ET)
- `sync_open_positions_with_batch()`: Maintains `open_positions.csv` snapshot enriched with latest batch prices
- `ensure_position_keys()`: Creates stable `position_key = slug|side|expiry|strike` for cross-file joins

### Key Data Files

- `DATA/btc_daily.csv`, `DATA/btc_intraday_1m.csv` — fetched by `data_fetcher.py`
- `DATA/historical_contract_prices.csv` — Polymarket contract prices (fetched by `polymarket_fetcher.py`, stored by `ContractPriceStore`)
- `old_market_prices.csv` — historical Polymarket prices for backtesting (legacy, being replaced by `historical_contract_prices.csv`)
- `positions.csv` — live trading position ledger
- `resolved_markets.csv` — logged outcomes for calibration

### Parameter Sweep Output Structure

```
parameter_sweeps/
  temp/           # Runs during execution (deleted after completion)
  saved/          # Top 10 runs kept permanently
    0001/
      taken_trades.csv
      montecarlo_results.csv
      equity_curve.csv
      run_config.md
      logs.txt
```

## Key Design Patterns

- **3-stage pipeline (Target → Delta → Action)**: Strategy separates ideal targets from sizing and risk enforcement, enabling independent testing of each stage
- **Convention over configuration**: Column names resolved via precedence chains, not explicit config
- **UTC everywhere**: All timestamps stored/normalized to UTC; ET conversions only for display and expiry settlement (12:00 PM ET)
- **NumPy RNG v2**: Uses `np.random.default_rng(seed)` — never `np.random.seed()`
- **CSV-based state**: No database for positions or batch data; everything is CSV files on disk (except Polymarket execution state which uses SQLite)
- **Idempotent backtest steps**: Backtest runner skips already-processed timestamps
- **Vol gate as standalone module**: Volatility risk can be computed independently and tested in isolation, then plugged into the strategy pipeline
- **Deprecation shims**: Old module paths preserved as forwarding imports with `DeprecationWarning` to avoid breaking imports during migration

## Change Logging & Documentation
After completing a task, do the following:

1. Append a single entry to `CHANGES.md` per logical task (not per file). Describe the intent and scope of the change, including which files were affected if relevant. Use present tense.

2. Check whether any of the following need updating and update them if so:
   - `CLAUDE.md` — architecture notes, file structure, conventions, anything describing how the project works
   - Anything in the DOCS folder — comprehensive MKdocs files with ALL information about the project

   Only update sections the change actually affects. Do not rewrite accurate sections.

## Pushing to GitHub
When I say "push" or "commit":
1. Read `CHANGES.md` and draft a commit message from it
2. Show me the message for approval
3. Commit and push
4. Clear `CHANGES.md`

## Never Do Without Asking
Before taking any of the following actions, stop and explicitly ask for confirmation:

- Refactoring code that wasn't part of the requested task
- Installing new dependencies
- Deleting or renaming files
- Changing function signatures, interfaces, or APIs
- Modifying configuration files (e.g. package.json, .env, docker, CI/CD)
- Making changes outside the files/scope I specified
- Resolving ambiguity by assumption — if the task is unclear, ask first

When in doubt about whether something falls outside the requested scope, ask.

## Temporary Files

All temporary artifacts — test scripts, summaries, test results, plans, reviews, debug dumps, scratch notes — go into `temp/`. Never leave them at repo root. The directory is gitignored; no need to clean up manually.

## Your Responsibilities
1. Ask, don't assume. If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements. When running unattended, pick the most reasonable interpretation, proceed, and record the assumption rather than blocking.

2. Implement the simplest solution for simple problems, better solutions for harder problems. Do not over-engineer or add flexibility that isn't needed yet. 

3. Don't touch unrelated code but please do surface bad code or design smells you discover with me so we can address them as a separate issue.

4. Flag uncertainty explicitly. If you're unsure about something, see point 1 above. If it makes sense to do so, conduct a small, localised and low-risk experiment and bring the hypothesis and results to me to discuss. Confidence without certainty causes more damage than admitting a gap.

5. I'm always open to ideas on better ways to do things. Please don't hesitate to suggest a better way, or one that has long lasting impact over a tactical change. (as a few examples)

Do not use non ASCII characters