# Implementation Plan — Remaining Audit Items (Revised)

**Source:** PRICING_ENGINE_AUDIT.md (2026-06-16)
**Reviewed by:** Cavecrew reviewer (2026-06-16)
**Bug 1 (Critical):** DONE — removed double variance correction from `skewed_t_rvs()` and `skewed_t_scale_factor()`. Hansen's `b` handles λ-dependent variance. All 8 tests pass.

---

## Item 2: Skewed-t Variance Diagnostic (NEW — reviewer catch)

**Target file:** `core/pricing/btc_pricing_engine.py`

**Problem:** `check_variance_consistency()` (line 371) tests only standard Student-t innovations. Bug 1 regression would not be caught by any existing test.

**Proposed change:**
1. Add `check_skewed_t_variance()` diagnostic in the `if __name__ == "__main__"` block:
   - Generate 50,000 skewed-t samples at lam=-0.3, 0.0, +0.3 (nu=5.0)
   - Verify empirical variance ∈ [0.95, 1.05] for all three
   - Print PASS/FAIL with actual variance values
2. Label as Test 9

**Risk:** None. Read-only diagnostic.
**Effort:** ~15 lines.

---

## Item 3: SVCJ Stochastic Residual (Bug 3, Low) ← reordered per reviewer

**Target file:** `core/pricing/btc_pricing_engine.py`

**Current state:** SVCJ return-vol correlation is deterministic: `jump_sizes += ρ_J × vol_jump_mag`. Eraker (2004) specifies conditionally normal: `ξ_s | ξ_v ~ N(μ_s + ρ_J × ξ_v, σ_s²)`.

**Proposed change:**
1. Add `SVCJ_SIGMA_S = 0.01` constant (conditional std dev of return jump given vol jump)
2. Change jump_sizes computation from:
   ```python
   jump_sizes += rho_J * vol_jump_mag
   ```
   to:
   ```python
   jump_sizes += rho_J * vol_jump_mag + rng.normal(0, sigma_s, size=n_sims)
   ```
3. Add `sigma_s` to `jump_params` dict and `build_regime_jump_params()`

**Risk:** Very low. Audit confirms limited practical impact for binary options.
**Effort:** ~15 lines across 2 functions.

---

## Item 4: Calibrate PROB_LOGIT_SHIFT_B (Rec 2)

**Target file:** `core/pricing/fit_probability_curves.py`

**Current state:** B=-0.7 hardcoded. At p=0.5 this shifts to ~0.332 (18 cent). No empirical justification.

**Outcome data source:** `resolved_markets.csv` (logged outcomes for calibration, per CLAUDE.md). Also available from backtest pipeline (all_priced.csv with known expiries). Source: model probability + market outcome → binary realized.

**⚠️ Materiality note:** If first calibration finds B≈0, all historical edge calculations were downward-biased by up to 18 cents. First calibration is retroactively material — plan should log this prominently.

**Proposed change:**
1. Add warning log when hardcoded B is used: `logger.warning("Using hardcoded PROB_LOGIT_SHIFT_B=-0.7. Run calibrate_logit_shift() with outcome data for an empirical estimate.")`
2. Add `calibrate_logit_shift(p_model, outcomes)` function:
   - Takes array of model probabilities and binary outcomes (from `resolved_markets.csv` or backtest `all_priced.csv`)
   - Estimates B via MLE (maximum likelihood on logistic model)
   - Reports 95% CI via likelihood ratio
   - Returns B_fitted, B_ci_lower, B_ci_upper
3. When outcomes are available, use fitted B; otherwise fall back to hardcoded default with warning
4. Update docstring to explain calibration methodology and data source

**Risk:** Low. Warning is informational only. Fitted B guarded behind outcome data availability.
**Effort:** ~50 lines in one file + ~20 lines in DOCS.

---

## Item 5: Recalibrate Jump Parameters from Data (Rec 3)

**Target file:** `core/pricing/btc_pricing_engine.py` + `core/pricing/jump_calibration.py`

**Current state:** LAMBDA=25, CRASH_PROB=0.6, ETA_UP=50, ETA_DOWN=25 hardcoded. `jump_calibration.py` exists but not wired into defaults.

**Proposed change:**
1. Add `load_calibrated_jumps()` function:
   - Runs `calibrate_jumps()` on full 5-year BTC hourly data (~43,792 obs)
   - Caches results to `DATA/jump_calibration.csv`
   - On subsequent runs, loads cached if < 30 days old
2. Compare calibrated vs hardcoded, log warnings if >20% delta
3. Add `--recalibrate-jumps` flag to `run_full_pipeline.py`
4. Wire `build_regime_jump_params()` to accept calibrated base values as override

**Risk:** Low. Calibration is conservative (full 5-year window, cached).
**Effort:** ~60 lines across 3 files.

---

## Item 6: Calibration Accuracy Metrics (Rec 4)

**Target file:** New file `core/validation/calibration_metrics.py`

**Current state:** No Brier score or reliability diagram comparing model probabilities to outcomes.

**Proposed change:**
1. New module `calibration_metrics.py` with:
   - `brier_score(p_model, outcomes)` — standard Brier score
   - `reliability_diagram(p_model, outcomes, n_bins=10)` — bins probabilities, plots mean predicted vs observed frequency
   - `ece_score(p_model, outcomes, n_bins=10)` — Expected Calibration Error
   - `run_calibration_report(priced_csv_path)` — reads a batch CSV, computes all metrics, prints report
2. Integrate into backtest pipeline: after backtest completes, run calibration report on all_priced contracts
3. Add to dashboard as optional tab or metric display

**Risk:** Low. Read-only analytics. No changes to pricing engine.
**Effort:** ~80 lines in new file + ~20 lines integration into backtest runner.

---

## Item 7: Data-Driven Regime Jump Multipliers (Rec 6)

**Target file:** `core/pricing/jump_calibration.py` + `core/pricing/btc_pricing_engine.py`

**Current state:** `build_regime_jump_params()` uses heuristics: ×1.5 bear, ×0.7 bull. These already exist as a dict — not additive constants.

**Proposed change:**
1. Add `calibrate_regime_jumps(returns, regimes)` to `jump_calibration.py`:
   - Accepts daily returns and HMM regime labels
   - Computes threshold exceedances per regime
   - Estimates λ_regime, p_crash_regime, η_up_regime, η_down_regime via MLE
   - Returns dict mapping regime → calibrated parameters
2. **Minimum-sample gate:** Require ≥30 detected jumps per regime. Bear regimes are ~15-20% of observations; after MAD threshold filtering, bear jumps may be sparse. Below threshold, fall back to heuristic multipliers with a warning log.
3. Modify `build_regime_jump_params()` to accept `calibrated: dict | None` parameter:
   - If provided and all regimes pass the 30-jump threshold, use calibrated values
   - Otherwise fall back to existing heuristic multipliers
4. Cache regime calibration alongside base jump calibration

**Risk:** Medium. Sample-size gating prevents unstable MLE. Still heuristic for bear if insufficient data.
**⚠️ Quant note:** At 30 jumps, SE(η_down) ≈ 24% relative (95% CI ≈ ±50%), SE(p_crash) ≈ 0.09 (95% CI ≈ [0.42, 0.78]). MLE estimates are unbiased but imprecise at this sample size. Regime multipliers remain the primary signal; calibrated values should be interpreted with CI awareness.
**Effort:** ~100 lines across 2 files.

---

## Item 8: FIGARCH Model Labeling (Bug 2, Medium) ← documentation-only per quant review

**Quant review found:** The proposed MA(1) extension `(1-φL)(1-L)^d` is NOT standard FIGARCH(0,d,1) — the standard spec is `1 - φL(1-L)^d` which gives different ARCH weights. Additionally, with d=0.578, φ>0 produces ψ₁ = -d-φ (more negative), weakening volatility clustering — economically backwards. No φ implementation is included in this plan.

**Target file:** `core/pricing/btc_pricing_engine.py`

**Current state (correct, just mislabeled):**
The variance recursion at line 586 is:
```
σ²_t = ω/(1-β) + Σ λ_k ε²_{t-k}
```
where λ_k are the binomial coefficients of (1-L)^d. β only appears in the intercept `ω/(1-β)` — there is no AR(1) feedback on variance. The docstring (lines 4, 15, 261, 311, 577) incorrectly labels this as "FIGARCH(1,d,1)".

This is actually **Fractionally Integrated GARCH with no short-run dynamics** — equivalent to FIGARCH(0,d) or FIGARCH(d) in the Baillie-Bollerslev-Mikkelsen taxonomy with β used solely as an intercept scaling factor.

**Proposed change (documentation only, zero code changes):**
1. Fix docstring on line 4: change `FIGARCH(1,d,1)` → `Fractionally integrated variance (FIGARCH-type, β-as-intercept only)`
2. Fix line 15: change `FIGARCH(1,d,1) long memory` → `Fractionally integrated variance [Siu 2025]`
3. Fix line 261: change `FIGARCH(1,d,1) binomial expansion weights` → `Fractional differencing binomial weights for (1-L)^d`
4. Fix line 311: change `FIGARCH(1,d,1) available` → `Fractionally integrated variance available`
5. Fix line 577 inline comment: change `FIGARCH(1,d,1)` → `Fractional integration`
6. Add module-level comment explaining the simplification:
   ```
   # The fractionally integrated variance model uses (1-L)^d binomial weights
   # with β only in the intercept ω/(1-β). This is a simplified specification —
   # standard FIGARCH(1,d,1) would apply (1-βL)⁻¹ to the ARCH recursion,
   # giving AR(1) feedback on variance. For binary option pricing, the
   # long-memory parameter d dominates; the AR(1) feedback is second-order.
   ```
7. Update DOCS/concepts/pricing-engine.md accordingly

**Risk:** None. Documentation change only.
**Effort:** ~15 lines across 6 comment locations.

**Note:** Full FIGARCH(1,d,1) with (1-βL)⁻¹ AR recursion and φ short-run parameter is deferred. Per audit: "The simplification is documented and has limited practical impact for binary option pricing." Only needed if engine repurposed for VaR forecasting.

---

## Item 9: Model-Based Basel Backtest (Bug 4, Low)

**Target file:** `core/validation/basel_backtest.py`

**Current state:** Backtest uses rolling historical VaR with ad-hoc jump inflation — validates naive benchmark, not the GARCH/SVCJ model.

**Computational constraint (reviewer catch):** 42,500 rolling windows × 15,000 MC paths = 637M simulations ≈ 15 days. Not viable.

**Revised approach:**
1. **Fit once, forecast analytically.** Fit GARCH once on the full sample. Compute h-step-ahead conditional variance forecast:
   ```
   E[σ²_{t+h} | F_t] = ω/(1-α-β) + (α+β)^h × (σ²_{t+1} - ω/(1-α-β))
   ```
   Map to VaR via `VaR_α = σ_forecast × t_ν⁻¹(α) × √((ν-2)/ν)`.

2. **⚠️ Quant limitation:** The analytical approach excludes the Kou jump component (no closed-form VaR for double-exponential mixture). For BTC at λ=25/year (~7% of days contain jumps), jump contribution to 1% VaR is ~15–25% of total. Analytical VaR will systematically underestimate tail risk. Mitigations:
   - Run analytical mode as fast screening; MC mode (n_sims=1000) for final validation
   - Document the gap explicitly in output table (column: "VaR method → jump contribution excluded" vs "MC method → jump contribution included")
   - Compare analytical vs MC on a subset to quantify the gap

3. **Reduce MC for validation-only mode.** n_sims=1000 per window. At α=1%, expected exceedances per window = 10. Adequate for Basel zone classification.

4. **Rolling cadence.** Fit every 500 hours (~monthly), giving ~87 fits (not 42,500). Each GARCH fit takes ~2-5 seconds via `arch` → ~7 minutes total fitting time.

5. Keep existing historical-VaR benchmark as comparison baseline.

**Functions:**
- `compute_model_var_analytical(params, horizon, alpha)` — analytical conditional VaR (GARCH diffusion only)
- `backtest_model_var(data, window=2000, refit_every=500, alpha=0.01, use_mc=False)` — chronological backtest
- `classify_basel_zone(exceedance_rate, alpha, n_obs)` — Green/Yellow/Red per Basel traffic light

**Risk:** Medium. Major module rewrite. Analytical VaR systematically underestimates BTC tail risk (no jumps). MC mode computationally constrained. The deliverable is a comparison tool, not a regulatory backtest.
**Effort:** ~300 lines. Multi-session task.

---

## Updated Execution Order

| Step | Item | Effort | Risk | Dependencies |
|------|------|--------|------|-------------|
| 1 | Item 2: Skewed-t variance diagnostic | ~10 min | None | None |
| 2 | Item 3: SVCJ stochastic residual | ~15 min | Very Low | None |
| 3 | Item 4: Logit shift warning + calibration | ~45 min | Low | None |
| 4 | Item 5: Jump recalibration wiring | ~45 min | Low | None |
| 5 | Item 6: Calibration metrics module | ~60 min | Low | Step 4 (needs priced data) |
| 6 | Item 7: Regime jump multipliers | ~75 min | Medium | Step 4 |
| 7 | Item 8: FIGARCH labeling (docs only) | ~10 min | None | None |
| 8 | Item 9: Model-based Basel backtest | ~3 hr | Medium | Requires careful validation |

Steps 1-4 and 7 are independent and can run in parallel. Steps 5-6 depend on step 4's calibration infrastructure. Step 8 is independent.

---

## Deferred (not in this plan)

- **Full FIGARCH(1,d,1) with AR weight recursion** — only needed if engine used for VaR forecasting
- **P→Q measure change (Rec 7)** — requires liquid BTC options market for calibration; predication market context doesn't warrant it
