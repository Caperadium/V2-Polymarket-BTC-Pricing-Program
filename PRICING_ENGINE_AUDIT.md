# BTC Pricing Engine v2 — Quantitative Audit Report

**Date:** 2026-06-16  
**Auditor:** Automated quant review  
**Files reviewed:** `core/pricing/btc_pricing_engine.py`, `core/pricing/jump_calibration.py`, `core/pricing/regime_detector.py`, `core/pricing/directional_xgb.py`, `core/pricing/fit_probability_curves.py`, `core/validation/basel_backtest.py`

## Executive Summary

The engine is **mathematically sound at the structural level** — GARCH(1,1)+Kou double-exponential jump diffusion with Hansen skewed-t innovations is a defensible model class for BTC binary option pricing. All 8 built-in validation tests pass. However, **4 substantive bugs** were found, 1 of which is critical (skewed-t variance under-dispersion) that materially affects probability estimates in non-sideways regimes. The model's risk-neutrality approximation (naive prior) is appropriate for short-dated contracts given the prediction-market context.

**Overall grade: B+** — Fix the skewed-t bug, and this rises to A-.

---

## 1. Model Architecture Assessment

### 1.1 GARCH(1,1) Base (✓ Sound)

Hourly log-return GARCH(1,1) with Student-t errors, fitted via the `arch` library. Standard, well-established methodology.

- Return scaling (×100 for numerical stability, ω÷10000 to recover hourly scale) is needed because the `arch` optimizer operates better with larger numbers. This is a common and valid trick.
- Variance consistency test passes: empirical/model variance ratio = 0.988 (within ±15% threshold). This confirms the GARCH recursion and Student-t scaling are internally consistent.
- Stationarity: α+β is not enforced in code, but the `arch` library typically constrains this. Should verify in fitted output that α+β < 1.

**Verdict:** Correct implementation of standard GARCH(1,1)-t.

### 1.2 Naive Prior (μ=0) (✓ Sound, with caveat)

Setting μ=0 (per Baquero 2026, Shelton 2024) is empirically justified. Test 5 demonstrates the scale of the problem: with fitted drift, paths deviate by $12,691 from S₀ (at S₀=$100,000 over 1 week), vs. $178 with μ=0. The fitted GARCH mean is unreliable for forecasting BTC, and the zero-drift prior is a pragmatic Bayesian shrinkage choice.

**Financial interpretation:** This is an implicit risk-neutrality assumption. In equity derivatives, you'd use a proper Girsanov change of measure. For BTC prediction markets where the drift contribution to short-dated binary P(S_T ≥ K) is small relative to vol + jumps, μ=0 is a reasonable engineering approximation. For contracts beyond 30 days, the horizon gate progressively strips features anyway.

**Verdict:** Defensible given context. Not rigorously risk-neutral, but the bias is small for typical use cases.

### 1.3 Kou Double Exponential Jump Diffusion (✓ Sound)

The jump component is correctly specified:

- Compound Poisson with intensity λ = 25/year → λ_hourly = 25/8760 ≈ 0.00285
- Jump sizes: Y_up ~ Exp(η_up), Y_down ~ Exp(η_down), with P(down) = CRASH_PROB = 0.6
- Multi-jump aggregation per hour via Gamma-distributed magnitudes (sum of k independent exponentials = Gamma(k, 1/η))
- Expected jump drift correction:
  ```
  E[J] = (1-0.6)/50 - 0.6/25 = 0.008 - 0.024 = -0.016
  E[jump_drift_hourly] = 0.00285 × (-0.016) ≈ -0.0000456
  ```
  This is subtracted from μ to maintain approximate risk-neutrality. Small but correct.

**Parameter calibration question:** λ=25, η_up=50, η_down=25 are hardcoded defaults. `jump_calibration.py` exists and can calibrate from data, but the default values are not currently data-driven for this deployment. On a 5-year BTC hourly sample, `calibrate_jumps()` would estimate these from actual threshold exceedances — the hardcoded values should be treated as initial guesses.

**Verdict:** Correct implementation. Default parameters are reasonable priors but should be periodically recalibrated.

### 1.4 Hansen Skewed-t Innovations (⚠ BUG — Critical)

The Hansen (1994) skewed-t sampling function `skewed_t_rvs()` applies an internal variance correction:

```python
# Inside skewed_t_rvs (line 234):
var_correction = np.sqrt(1 + 3 * lam ** 2)
g = g / var_correction
```

But then `skewed_t_scale_factor()` applies the same correction AGAIN:

```python
# skewed_t_scale_factor (line 251):
skew_adjust = 1.0 / np.sqrt(1 + 3 * lam ** 2)  # SECOND application
```

And in `simulate_paths()` (line 547-548):
```python
scale_factor = skewed_t_scale_factor(nu, skewed_t_lam)  # includes 1/sqrt(1+3λ²)
z_t = skewed_t_rvs(nu, skewed_t_lam, n_sims, rng) * scale_factor  # rvs also divides
```

**Empirical variance at different λ values (should be ~1.0):**

| λ | Variance | Bias |
|---|----------|------|
| 0.0 | 0.999 | None |
| -0.1 | 0.935 | -6.5% |
| -0.2 | 0.814 | -18.6% |
| -0.3 | 0.599 | **-40.1%** |
| +0.3 | 0.607 | **-39.3%** |

In the bear regime (λ=-0.3), innovations have only 60% of intended variance. This **understates volatility and tail risk in bear markets by ~40%**. Since bear regimes use stronger jump parameters, the jump component partially compensates, but the GARCH diffusion component is systematically under-dispersed.

**Fix:** Remove the `var_correction` division from `skewed_t_rvs()` and keep it only in `skewed_t_scale_factor()`. OR remove it from `skewed_t_scale_factor()` and keep only in `skewed_t_rvs()`. Pick one location.

**Severity:** Critical. Impacts all probability estimates when `use_skewed_t=True` with λ ≠ 0 (bear and bull regimes in regime-switching mode). Probabilities for tail strikes will be biased.

### 1.5 SVCJ Volatility Jumps (⚠ Sound structurally, simplified from Eraker 2004)

The SVCJ implementation uses:
- Shared Poisson driver for return + vol jumps ✓
- Vol jump magnitudes from Exponential(μ_v) ✓
- Return-vol correlation via additive term: `jump_sizes += ρ_J × vol_jump_mag` (simplified)
- Vol jump added directly to variance: `variances += vol_jump_mag` ✓
- Variance floor at 1e-12 ✓

**Where it deviates from the full specification:** Eraker et al. (2004) model the return jump as conditionally normal given the vol jump:
```
ξ_s | ξ_v ~ N(μ_s + ρ_J × ξ_v, σ_s²)
```
The simplified additive approach `jump_sizes += ρ_J × vol_jump_mag` captures the correlation sign and magnitude but treats it as deterministic conditional on ξ_v rather than adding the conditional variance σ_s².

For the prediction market use case (binary options on terminal price), this simplification has limited practical impact — the terminal distribution's tail properties are primarily driven by the vol jumps themselves and the return jump magnitudes, not the conditional variance of the return jump given the vol jump.

**Verdict:** Reasonable engineering approximation. Directionally correct. Test 6 confirms SVCJ adds measurable variance vs. SVJ (12.2% log-return std dev vs. 10.2%).

### 1.6 FIGARCH Long Memory (⚠ Sound concept, simplified implementation)

FIGARCH(1,d,1) is implemented as a truncated binomial expansion of the fractional differencing operator. The weight recurrence:
```
λ₀ = 1
λ_k = λ_{k-1} × (k - 1 - d) / k
```
is the correct binomial coefficient formula for (1-L)^d.

**Where it deviates:** The variance recursion uses:
```python
σ²_t = ω/(1-β) + Σ_{k} λ_k(d) × ε²_{t-k}
```

The standard FIGARCH(1,d,1) specification is:
```
σ²_t = ω/(1-β) + [1 - (1-βL)^{-1} × (1-φL) × (1-L)^d] × ε²_t
```

The engine's version omits the φ (short-run ARCH) parameter and essentially runs a FIGARCH(1,d,0). This means the short-run volatility dynamics are captured solely by the fractional differencing, without a separate short-run ARCH component. The hyperbolic decay from d=0.578 drives both short and long memory.

d=0.578 (from Siu 2025, SE=0.271) is plausible for BTC. The high standard error (0.271) means the true d could reasonably be anywhere from 0.3 to 0.85 — wide uncertainty. The truncation at k=1000 is adequate (weights decay to ~10^-6 by this lag).

**Verdict:** Simplified but functional. The missing φ parameter means the model can't separately control short and long memory. In practice, for the current use case, this is unlikely to matter much compared to having FIGARCH at all vs. plain GARCH. The documented reference to Siu (2025) with SE=0.271 is honest about uncertainty.

### 1.7 Regime Detection via HMM (✓ Sound)

3-state Gaussian HMM on daily returns, labeled by annualized mean (bear/sideways/bull). Uses `hmmlearn.GaussianHMM` which is production-stable.

- 2-year rolling window ✓
- Weekly re-estimation cadence ✓
- Fallback to stale model on fit failure ✓
- Post-hoc weighting of regime-specific simulations (not intra-path switching) ✓

The post-hoc weighting approach is statistically correct:
```
P(S_T ≥ K) = Σ_r w_r × P_r(S_T ≥ K)
```
where w_r are the HMM posterior probabilities and each P_r is an independent MC simulation with regime-specific parameters.

**Verdict:** Correct implementation. Regime-dependent parameter multipliers are heuristic (×1.5 bear, ×0.7 bull) — these should ideally be calibrated from data, but the directional logic is correct.

### 1.8 Horizon Gating (✓ Sound)

Sensible three-tier gating:
- T > 90d: naive prior only (GARCH-t, no jumps)
- 30-90d: intermediate (GARCH-t + Kou jumps, no SVCJ/skewed-t/FIGARCH)
- 7-30d: advanced (SVCJ + FIGARCH enabled)
- <7d: full model (all features)

This aligns with the empirical finding that short-dated crypto options require the most model complexity (jump risk dominates, vol persistence matters).

**Verdict:** Appropriate. The gating prevents over-parameterization of contracts where the signal-to-noise ratio is negligible.

---

## 2. Statistical Validity

### 2.1 Monte Carlo Convergence

n_sims = 15,000 default. Standard error of a probability estimate:
- At p=0.50: SE = sqrt(0.5×0.5/15000) = 0.0041 (~0.4 percentage points)
- At p=0.10: SE = sqrt(0.1×0.9/15000) = 0.0024 (~0.24 pp)
- At p=0.01: SE = sqrt(0.01×0.99/15000) = 0.0008 (~0.08 pp)

For practical trading decisions with edge thresholds of 4-6 cents (4-6 pp), 15,000 paths provides adequate precision. The MC noise is smaller than the typical edge threshold, meaning noise-driven false positives should be rare.

### 2.2 GARCH Parameter Stability

The data filter to post-2019-10-01 (per Pakstaite 2025 structural break) uses ~43,792 hourly observations (~5 years). This is adequate for GARCH estimation. The arch library uses robust MLE optimization.

### 2.3 Basel Backtest Results

The Basel backtest module (`basel_backtest.py`) returns all-Red results across all horizons and α levels using the simple rolling historical VaR benchmark:

| Horizon | α=1% exceed | α=5% exceed | Zone |
|---------|------------|------------|------|
| 1h | 5.3% | 9.4% | Red |
| 14d | 0.0% | 0.002% | Red |
| 28d | 0.0% | 0.0% | Red |

**This is expected and actually validates the need for the complex model.** The 1h exceedances are far too high (historical VaR underestimates short-horizon risk because BTC is fat-tailed and heteroskedastic). The multi-day exceedances are zero (sqrt(h) scaling is over-conservative, ignoring mean reversion in volatility). Both failures demonstrate exactly why GARCH+Jumps+SVCJ is needed.

**However**, the backtest module uses rolling historical VaR with ad-hoc jump inflation factors, NOT the actual fitted GARCH/SVCJ model. To properly validate the engine, the backtest should use the engine's own conditional distribution to forecast VaR. As written, the backtest validates the naive historical benchmark, not the model.

### 2.4 Logistic Curve Calibration

`fit_probability_curves.py` applies a logit shift of B=-0.7 to calibrated probabilities:
```
p_cal = sigmoid(logit(p_fit) + B)
```

At p=0.5: shift → p_cal = sigmoid(0 - 0.7) = sigmoid(-0.7) ≈ 0.332
At p=0.1: shift → p_cal = sigmoid(logit(0.1) - 0.7) ≈ 0.037

This is a substantial downward calibration (18 cents at p=0.5). B=-0.7 should be empirically justified — it should come from comparing model probabilities against realized outcomes. Without seeing the calibration data, this appears to be a fudge factor. A proper calibration would estimate B via MLE on historical outcomes, report confidence intervals, and test whether B is statistically different from zero.

---

## 3. Financial Validity

### 3.1 Binary Option Pricing Framework

The fundamental quantity estimated — P(S_T ≥ K) — is correct for a cash-or-nothing binary call, which maps to Polymarket "BTC above $X on date Y" contracts. No issues with the contract mapping.

### 3.2 Risk-Neutrality and P→Q Measure Change

This is the most significant theoretical gap. The model operates in the physical (P) measure with an ad-hoc risk-neutralization:
1. Naive prior sets μ=0
2. Jump drift correction removes the expected jump component from drift
3. No explicit market price of risk or volatility risk premium

For BTC prediction markets, this is **pragmatically defensible** because:
- There is no liquid BTC options market to calibrate a proper Q-measure from
- BTC's equity risk premium is highly uncertain and time-varying
- Short-dated binary options are primarily driven by volatility and jumps, not drift
- The prediction market itself provides the "Q" reference point (market prices)

However, the engine's probability should be interpreted as **physical-world probability with drift shrinkage**, not as a true risk-neutral probability. When comparing to Polymarket prices, any systematic divergence could reflect risk premia rather than mispricing.

### 3.3 Edge Computation

Edge is computed as `p_model - market_price`. Since p_model is physical (not risk-neutral), this edge conflates two effects:
1. Genuine mispricing (model vs. market disagreement about probability)
2. Risk premium (the market's risk-neutral probability differs from physical probability)

For a directional trader, this conflation is acceptable — both sources generate expected returns if the model's physical probability estimate is accurate. But the interpretation of the edge should acknowledge this.

### 3.4 Time Decay and Volatility Term Structure

The GARCH model naturally produces a volatility term structure through the variance recursion. For short horizons, conditional variance dominates; for long horizons, variance reverts to unconditional. This is more realistic than flat Black-Scholes vol.

---

## 4. Bug Summary

| # | Severity | Location | Issue | Impact |
|---|----------|----------|-------|--------|
| 1 | **Critical** | `skewed_t_rvs()` + `skewed_t_scale_factor()` | Double variance correction (÷√(1+3λ²) applied twice) | Innovations 40% under-dispersed at λ=±0.3 |
| 2 | **Medium** | FIGARCH variance recursion | Missing short-run φ/ARCH parameter | Reduced flexibility in short-run vol dynamics |
| 3 | **Low** | SVCJ implementation | Simplified return-vol correlation (deterministic, not stochastic conditional) | Minor underestimation of conditional tail variance |
| 4 | **Low** | Basel backtest module | Uses historical VaR benchmark, not actual GARCH/SVCJ forecasts | Backtest doesn't validate the actual model |

### Bug 1 Fix (Critical)

In `skewed_t_rvs()` (line 233-234), remove:
```python
var_correction = np.sqrt(1 + 3 * lam ** 2)
g = g / var_correction
```

Keep the correction only in `skewed_t_scale_factor()` where it's composed with the Student-t scale factor `sqrt((nu-2)/nu)`.

### Bug 2 Note

The FIGARCH simplification is documented and references the literature honestly. If the engine were being used for VaR forecasting or regulatory capital, this should be upgraded to the full FIGARCH(1,d,1) specification. For binary option pricing, the impact is small because terminal distribution moments are more sensitive to d and the long-memory structure than to the short-run φ parameter.

---

## 5. Literature Support

The engine cites 17 papers. Key references checked:

| Reference | Claim | Engine Implementation | Assessment |
|-----------|-------|----------------------|------------|
| Baquero (2026) | μ=0 outperforms fitted drift | Naive prior (default on) | Direct application ✓ |
| Eraker et al. (2004) | SVCJ specification | Simplified version | Partial implementation |
| Hansen (1994) | Skewed-t distribution | Correct except variance bug | Bug needs fixing |
| Teng et al. (2025) | Basel backtest, SVCJ for BTC | Basel module + SVCJ flag | SVCJ direction is correct |
| Siu (2025) | FIGARCH d=0.578 for BTC | Implemented with SE noted | Honest about uncertainty |
| Kou (2002) | Double exponential jump diffusion | Correct multi-jump aggregation | Direct application ✓ |
| Oprea & Bâra (2026) | HMM regime detection | 3-state Gaussian HMM | Direct application ✓ |
| Shelton (2024) | OOS evidence on weak predictors | XGBoost weight = 0.3 | Direct application ✓ |
| Pakstaite (2025) | Post-2019 structural break | Training from 2019-10-01 | Direct application ✓ |

Overall, the literature grounding is solid. The engine correctly applies the key empirical findings: zero drift, structural breaks, regime dependence, and long memory in volatility.

---

## 6. Recommendations

### Immediate (Bug Fix)

1. **Fix the skewed-t double variance correction.** Remove the `var_correction` division from `skewed_t_rvs()` (line 233-234). This is a one-line fix that eliminates 40% under-dispersion in regime-specific simulations.

### Short-Term (Validation)

2. **Calibrate PROB_LOGIT_SHIFT_B** from historical outcomes rather than using a hardcoded B=-0.7. Report the calibration period and confidence interval.

3. **Recalibrate jump parameters** using `jump_calibration.py` on the full 5-year sample. Compare calibrated values to current defaults:
   - If calibrated λ is significantly different from 25, update defaults
   - If calibrated p_crash differs from 0.6, update

4. **Add a calibration accuracy metric** — compare model probabilities against realized binary outcomes across a rolling out-of-sample window. Report Brier score and reliability diagram.

### Medium-Term (Model Enhancement)

5. **Upgrade the Basel backtest** to use the actual fitted GARCH/SVCJ conditional distribution for VaR forecasting, not the simple historical benchmark.

6. **Calibrate regime jump multipliers** — the ×1.5 bear, ×0.7 bull multipliers for jump intensity should be estimated from regime-conditional jump detection rather than heuristics.

7. **Add a proper P→Q adjustment** if the engine is to be used for formal option pricing rather than prediction market edge detection. This would involve estimating the market price of BTC variance risk.

### Documentation

8. The engine's docstring and documentation accurately describe what it does. No misleading claims found.

---

## 7. Overall Verdict

The BTC Pricing Engine v2 is a **well-constructed, research-grounded model** for estimating BTC binary option probabilities. The model class (GARCH + jump diffusion + regime switching) is appropriate and defensible for the use case. The literature foundation is solid, with 17 papers cited and their findings correctly applied.

The critical skewed-t variance bug is the only issue that materially affects output probabilities. Once fixed, the engine's mathematical validity rises to A-grade. The remaining issues are simplifications that are documented and have limited practical impact for the prediction market use case.

**Engine is legitimate. Fix the skewed-t bug before going live.**
