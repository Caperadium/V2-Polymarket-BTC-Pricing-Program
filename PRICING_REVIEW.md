# Pricing Engine & Monte Carlo Review

**Date:** 2026-07-06
**Scope:** `core/pricing/btc_pricing_engine.py`, `core/pricing/jump_calibration.py`,
`core/pricing/regime_detector.py`, `core/pricing/directional_xgb.py`, plus the call
paths in `core/backtesting/backrunner.py`, `scripts/pipelines/run_full_pipeline.py`,
and `scripts/pipelines/batch_pricing_runner.py`.
**Reference documents:** `Journal Articles/synthesis.md` (17-paper meta-analysis),
`Journal Articles/improvement_plan.md`.

This review checks (1) mathematical and statistical soundness of the pricing engine
and MC simulator, and (2) whether the features specified in the synthesis and
improvement plan are implemented and implemented in a way consistent with the
literature findings.

---

## Executive Summary

The engine is structurally in good shape. The compound-Poisson Kou jump
aggregation, Hansen skewed-t sampler, FIGARCH ARCH-infinity weights, jump-drift
compensator algebra, and the leak-hygiene discipline in the backtest path are all
correct and in several places validated against reference implementations. Most of
the improvement plan (Phases 0 through 2.6, plus the Basel and calibration
validation modules) is present in code.

However, there are three high-severity issues that undermine the statistical
claims the architecture makes:

1. **Jump variance is double-counted** (H1). GARCH/FIGARCH is fitted on raw
   returns that include jumps, then a separately calibrated jump process is added
   on top in simulation. Total simulated variance exceeds historical variance by
   roughly the full jump contribution. This systematically fattens the terminal
   distribution and overprices OTM tail contracts -- the exact contracts this
   system trades.
2. **Live and backtest price with different models** (H2). The backtest path uses
   `calculate_probabilities` (regime switching, horizon gating, regime-conditional
   jumps and skew). The live pipelines call `simulate_paths` directly and get none
   of that. Backtest performance therefore does not validate the live model.
3. **The regime-switching layer barely changes the distribution** (H3). The HMM's
   fitted state means and variances are never used in simulation; regimes differ
   only via hardcoded jump multipliers and hardcoded skew lambdas, and the current
   regime posterior is held fixed over the whole simulation horizon instead of
   being propagated through the transition matrix.

Fixing H1 and H2 should come before any further calibration work, because both
bias every probability the system produces.

---

## Part 1: Findings (severity-ranked)

### H1. Jump variance double-counting (estimation/simulation mismatch)

**Where:** `fit_garch_model` + `simulate_paths` (`btc_pricing_engine.py`),
`calibrate_jumps` (`jump_calibration.py`).

**Problem.** The GARCH/FIGARCH parameters (omega, alpha, beta, nu,
`last_variance`) are estimated by MLE on the full hourly return series --
including all jump returns. The fitted model therefore already accounts for the
total historical variance, jumps included, and the fitted Student-t nu is already
fattened by the jump tail. In simulation, an independently calibrated compound
Poisson jump process (lambda up to the clip at 100/yr) is then added on top of the
GARCH diffusion. The two components are never reconciled:

- Unconditional simulated variance ~= GARCH-implied variance (which already
  matches history) + lambda * E[J^2].
- With the module defaults (lambda=25, p_crash=0.6, eta_up=50, eta_down=25),
  E[J^2] = 0.6 * 2/625 + 0.4 * 2/2500 = 0.00224 per jump, so the jump layer adds
  ~0.056 of annual variance. Against a ~0.25 annual BTC variance (50% vol) that is
  a ~20% variance overstatement, i.e. ~10% too much vol. Bipower-calibrated
  hourly lambdas can be substantially higher (clip allows 100/yr), with smaller
  eta -- the overstatement is the same order of magnitude either way.
- Effect on the product: for a strike ~1.5 sigma OTM at 14 DTE, overstating vol by
  10% moves the binary probability from ~6.7% to ~8.7% -- a 2-cent model-side bias
  on a 7-cent contract, i.e. larger than a typical `min_edge` filter. The bias is
  one-directional: the model systematically sees too much tail probability and
  will prefer buying longshots.

There is a second face of the same inconsistency: inside `simulate_paths` the
variance recursion is driven by the diffusion innovation only
(`epsilon_squared = (step_sigma * z_t)**2`, jumps excluded). That is the correct
SVCJ-style decomposition for simulation, but the alpha/beta being used were fitted
on total returns where epsilon^2 did include jumps. Estimation and simulation
disagree about what epsilon is.

**Literature context.** Teng (2025) and Eraker (2004) estimate SV and jump
parameters jointly (MCMC), precisely so the diffusion variance is the
jump-filtered variance. Qiao (2025) similarly estimates GARCH-diffusion and jump
parameters within one likelihood. The synthesis never endorses fitting the
diffusion on jump-contaminated returns and stacking calibrated jumps on top.

**Remediation (in increasing order of effort):**
1. *Jump-filtered GARCH fit (recommended first step).* The backrunner already
   computes a per-snapshot bipower jump mask. Before `fit_garch_model`, shrink
   detected jump returns to the local bipower sigma (or drop/median-replace them),
   so the GARCH sees the diffusion component only. This is a few lines and
   directly removes the double count.
2. *Variance budget check.* Add a diagnostic comparing simulated terminal
   std(log S_T/S0) at 7/14/28d against realized same-horizon return std -- the
   `core/validation/basel_backtest.py` MC mode already provides most of this;
   record the dispersion ratio, not just VaR exceedances.
3. *Joint estimation (plan Phase 3.2).* GARCH-jump joint MLE via a filtered
   likelihood (Ornthanalai-style) or Bayesian MCMC as already planned.

### H2. Live pricing bypasses the v2 model -- backtest validates a different engine

**Where:** `scripts/pipelines/run_full_pipeline.py` (~line 318),
`scripts/pipelines/batch_pricing_runner.py` (~line 327) vs
`core/backtesting/backrunner.py` (~line 347).

**Problem.** The backtest path calls `calculate_probabilities(...)` with
`use_regime_switching=True`, a per-snapshot `RegimeDetector`, per-regime
calibrated jump params, and gets horizon gating. Both live pipelines call
`simulate_paths(...)` directly with only `use_naive_prior/use_svcj/use_skewed_t/
use_figarch`. Consequences in live mode:

- No regime switching, no regime-conditional jump parameters (single global
  calibrated set).
- No horizon gating: a 45-DTE contract is priced with the full SVCJ+FIGARCH stack
  live, while the backtest would have disabled regime switching for it (and would
  disable everything at >90 DTE).
- Different skew: the regime branch uses per-regime lambdas (-0.3 / 0.0 / +0.2);
  the live path silently uses `SKEWED_T_LAMBDA_DEFAULT = -0.1` for everything.

Any backtest-derived quantity -- edge thresholds, the M2 logit-shift calibration,
sweep-selected strategy parameters -- was estimated against regime-mixture
probabilities and is being applied to single-regime live probabilities. The
CLAUDE.md/plan Section 7 promise ("backtest re-fit ... same pipeline") is not
currently true at the model-configuration level.

**Remediation.** Route both live pipelines through `calculate_probabilities`
(constructing a `RegimeDetector` and passing `regime_params` exactly as the
backrunner does; `as_of=None` already gives live wall-clock behavior). The
function was designed for this -- the live callers just predate it.

### H3. Regime layer is mostly cosmetic; posterior not propagated over the horizon

**Where:** `calculate_probabilities` regime branch, `build_regime_jump_params`,
`RegimeDetector`.

Three related gaps versus the synthesis blueprint (Findings 4, and the
architecture diagram's "regime-conditional distribution engine"):

1. **HMM emissions are unused.** The HMM estimates per-state means and variances,
   but the simulation uses the *same* GARCH params, the same mu (zeroed by the
   naive prior), and the same S0 for every regime. Regimes differ only through
   (a) jump-parameter multipliers and (b) skew lambda. Since diffusion variance
   dominates the terminal distribution at 7-30 DTE, the bear/bull/sideways
   sub-simulations produce nearly identical distributions. Oprea's 56% MAE
   reduction came from regime-conditional *models*; this implementation
   conditions only a second-order component. At minimum, scale each regime's
   initial/long-run variance by the ratio of its HMM state variance to the
   pooled variance (keeping the naive prior on the mean, per Finding 5).
2. **Hardcoded multipliers instead of estimated per-regime parameters.** The
   bear/bull multipliers in `build_regime_jump_params` (1.5x lambda, 0.7x
   eta_down, 2x mu_v, etc.) and the skew lambdas (-0.3/0.0/+0.2) are asserted, not
   estimated. `calibrate_regime_jumps` exists but is deliberately not used in the
   backtest (its wall-clock fallback would leak). Consider making it as_of-aware
   so per-regime estimation can actually run leak-free.
3. **Current posterior held fixed over the horizon.** A 14-30 DTE contract is
   priced with today's regime weights applied to the entire path. With diagonal
   persistence ~0.91-0.94 (synthesis Finding 4), the regime distribution decays
   most of the way to stationary within 2-4 weeks. `RegimeDetector.predict_weights
   (n_days_ahead)` implements exactly the needed transition-matrix propagation and
   is never called by the engine. Cheap fix: weight the mixture by
   `predict_weights(round(days_to_expiry/2))` (average occupancy approximation)
   rather than the time-0 posterior. Correct fix: mix over regime paths, but the
   plan's C1 resolution (post-hoc weighting) explicitly accepted the
   approximation -- propagating the weights keeps that design while removing its
   worst bias at longer DTE.

Also: `BEAR_THRESHOLD` / `BULL_THRESHOLD` in `regime_detector.py` are dead --
labeling is purely rank-by-mean. In a strong bull market the lowest-mean (but
still positive-drift) state is labeled "bear" and receives crash-heavy jump
multipliers. Either use the thresholds to allow degenerate labelings (e.g. two
sideways states) or document that labels are relative ranks.

### M1. SVCJ return-vol correlation channel is dimensionally inert

**Where:** `simulate_paths` SVCJ block; `calibrate_jumps` rho_J estimation.

Eraker (2004) specifies xi_s | xi_v ~ N(mu_s + rho_J * xi_v, sigma_s^2) where
rho_J is a *regression slope* (units: return per unit of variance jump). The code
adds `svcj_rho_j * vol_jump_mag` to the return jump, but:

- `vol_jump_mag` is in hourly variance units (mu_v ~ 2.5e-5, state cap 1e-3), so
  with rho_J = -0.08 the adjustment is ~ -2e-6 log-return -- five orders of
  magnitude below a typical jump size (~0.02-0.04). The term does nothing.
- The calibrated rho_J is a Pearson *correlation* (dimensionless), then used as a
  slope. Correlation and slope differ by sigma_ret / sigma_voljump, which here is
  huge -- that is exactly why the term vanishes.

Net effect: the return-vol jump dependence that makes SVCJ pass Teng's Basel
tests is delivered only through (a) shared Poisson timing (implemented correctly)
and (b) the vol jump raising subsequent diffusion variance. Those are the dominant
channels, so SVCJ is still meaningfully better than SVJ here -- but the rho_J
parameter as implemented is decorative, and reporting it as "Teng's estimate"
overstates what the model does.

**Remediation.** Either (a) implement the slope properly: estimate
b = rho * sigma_s / sigma_v from the calibration event set and use
`jump_sizes += b * vol_jump_mag`, or (b) drop the term and document that
co-timing + vol-jump persistence carry the correlation. If keeping it, the
regime multiplier `rho_J * 1.5` (bear) should be re-examined -- multiplying a
correlation-used-as-slope has no meaning either.

### M2. mu_v calibration mechanically confounded by the jump's own square

**Where:** `calibrate_jumps` / `calibrate_regime_jumps` vol-jump estimation.

The vol jump is measured as `rolling_var[idx+2] - rolling_var[idx-2]` where
`rolling_var` is the trailing 24h mean of squared returns. The post-jump window
*contains the jump return itself*, so a jump of size J mechanically raises the
post variance by ~J^2/24 (~4e-5 to 7e-5 for a 3-4% jump) even if true diffusion
volatility never moved. That is 2-3x the Teng reference mu_v of 2.5e-5, so the
calibrated mu_v is plausibly dominated by this artifact. Additionally, only
positive deltas are kept (`max(0, post - pre)` then mean of positives), which is
an upward selection bias on top.

**Remediation.** Exclude the jump bar (and ideally the +/-1 bars) from the post
window, or subtract J^2/window explicitly; estimate the mean over *all* deltas of
a truncated-at-zero exponential rather than the mean of positive deltas (MLE for
an exponential censored at 0), or at least report both. Same fix applies to the
per-regime version.

### M3. Lee-Mykland statistic includes the contemporaneous return in its own threshold

**Where:** `detect_jumps_bipower` (`jump_calibration.py`).

`bpv_local = (pi/2) * (s.shift(1) * s).rolling(window).mean()` -- at time t the
last product term is |r_{t-1}| * |r_t|, which contains the return being tested.
The comment claims the shift excludes the contemporaneous return; it does not (it
only offsets one side of each product). Lee & Mykland's sigma_hat(t) window ends
at t-1. Effect: a genuine jump inflates its own local sigma by roughly a factor
(1 + (|J|/sigma - 1)/K); with K=78 and a 10-sigma jump that is ~11% threshold
inflation -- modest under-detection, concentrated on the largest jumps.

**Fix:** one extra shift: `(s.shift(1) * s).shift(1).rolling(...)`. Also note
K=78 is the LM recommendation for 5-minute bars; their guidance for coarser bars
is larger K (order sqrt(252 * bars-per-day) scaled) -- worth a sensitivity check
on hourly data rather than adopting 78 by default.

### M4. `martingale_anchor=True` does not deliver E[S_T] = S0

**Where:** `simulate_paths` drift block.

The exponential-cumulant jump compensator is algebraically correct for the Kou
part (E[e^J] = (1-p) * eta_up/(eta_up - 1) + p * eta_down/(eta_down + 1), with the
eta_up > 1 guard). But the docstring claims the switch makes E[S_T] = S0, and it
does not, because:

- The diffusion Jensen term is never subtracted: no -sigma^2/2 * dt anywhere. At
  ~50% annualized vol that is ~ +1.0-1.5% drift in E[S_T] over 30 days.
- With Student-t innovations the exponential moment E[e^{sigma z}] is not even
  finite in theory (the +/-2 return clip makes it finite in practice, slightly
  above the Gaussian value).
- The SVCJ Gaussian residual (sigma_s) and rho_J term are excluded from the
  compensator (negligible at current magnitudes, but wrong in principle).

Since `martingale_anchor` defaults to False and nothing in the repo currently
sets it True, the practical impact today is zero -- but the docstring's claim
should be corrected, and if a risk-neutral mode is ever used for pricing, the
diffusion compensator must be added (per-step, using the pathwise
`variances * dt/2`, which also handles GARCH/FIGARCH stochasticity correctly).

Related documentation nit: the default physical-measure mode is described as
"median-anchored". With mu=0 and the log-mean compensator, E[log S_T] = log S0
exactly -- that is geometric-mean anchoring. The median coincides only for a
symmetric log distribution; with p_crash=0.6 and negative skew-t the median sits
slightly above S0. Fine in practice, but the docs should say "log-mean anchored".

### M5. Dead fallback in `load_and_prep_data`

**Where:** `btc_pricing_engine.py` ~line 189.

When the `training_start_date` filter leaves <500 rows, the code logs "Falling
back to all data" but the reload is guarded by `if hourly_df is None:` -- which is
never true at that point (it was just filtered). The fallback can never execute;
the engine proceeds with the too-small sample it just warned about. Low practical
risk today (post-2019 filter leaves years of data) but the guard is broken; in a
time-travel backtest with an early snapshot this silently fits GARCH on a sliver.
Fix: re-read (or retain a pre-filter copy) when the row count check fails.

### Low-severity / notes

- **`vol_gate_regime` is dead in the pipeline.** `simulate_paths` accepts it and
  scales lambda/mu_v (1.5x/2.0x extreme), but `calculate_probabilities` never
  passes it and no caller does either. The Phase 2.6 protocol is implemented at
  the strategy layer (hard gate), which is the documented override -- but then the
  engine-side multipliers are unreachable code and their hardcoded values are
  untested. Either wire it through or remove it.
- **Horizon gate >90d log message says "no jumps", but Kou return jumps stay
  on.** The gate disables SVCJ/skew/FIGARCH/regime/XGB and enforces mu=0, yet
  `jump_params` still applies in `simulate_paths`. The plan's BS1 table says
  "naive prior ONLY". Keeping jumps for distribution shape is defensible, but the
  code and the message should agree.
- **Skew-t parameters are asserted, not estimated.** The GARCH fit uses
  `dist='t'` (symmetric) and the simulation then bolts on a hardcoded lambda.
  The `arch` package supports `dist='skewt'` (Hansen) -- fitting nu and lambda
  jointly would replace three hardcoded lambdas with estimates and remove the
  mild nu bias from fitting a symmetric density to skewed data.
- **Hourly GARCH ignores intraday seasonality.** Hourly BTC vol has a strong
  time-of-day pattern; a constant-omega GARCH averages it. Immaterial at 7-30
  DTE, but for <2 DTE contracts the terminal distribution misallocates variance
  across the remaining hours. A diurnal multiplicative seasonal factor on omega
  would fix it if short-DTE pricing matters.
- **`weights_array` is always None** in `calculate_probabilities` (both branches)
  -- the `np.average(..., weights=...)` branch is unreachable. Leftover from the
  pre-proportional-allocation design; remove.
- **Fixed seed across snapshots.** The backrunner uses seed=42 for every
  snapshot, so the same innovation draws recur across the whole backtest. Good
  for determinism/comparability, but MC noise becomes a shared systematic rather
  than averaging out across snapshots. With n_sims=15000 the standard error at
  p=0.10 is ~0.24% (absolute), ~0.11% at p=0.02 -- acceptable, but consider
  seed=hash(snapshot) if backtest-aggregate statistics are being read at the
  sub-cent level, and/or antithetic variates for a free variance halving.
- **XGB tilt anchors at 0.5, not at the model's own p_base.** `p_target = 0.5 +
  lam * (p_up - 0.5)`: at lam=1 a neutral XGB signal would drag a legitimately
  asymmetric MC distribution back to 50/50, discarding the jump/skew asymmetry
  the engine worked to produce. Consistent with the plan's C6 formula
  (w_prior * 0.5 blend), so this is a design choice, not a bug -- but worth
  remembering when calibrating lambda: it is simultaneously "trust XGB" and
  "shrink the physical distribution's directional content toward a coin flip".
  Anchoring the tilt at p_base (p_target = p_base + lam * (p_up - p_base)) would
  make lambda purely "trust XGB".
- **XGB accuracy metric is optimistic.** Targets are overlapping rolling-horizon
  sums (h-day windows shifted by 1 day), so train/test rows are heavily
  autocorrelated and the single 80/20 split's accuracy overstates OOS skill.
  Harmless while XGB_TILT_LAMBDA=0; before activating lambda, evaluate with
  purged/embargoed splits (gap >= horizon_days between train and test).
- **`fit_kou_params` return contract.** The 4th tuple element is documented as
  `annual_lambda` but is actually the raw jump count; callers recompute lambda
  correctly, but fix the docstring/name before someone uses it.
- **Detected jump returns include diffusion.** eta_up/eta_down are fitted to the
  full return on jump bars (jump + diffusion component), slightly biasing mean
  jump sizes up / eta down. Standard practice would subtract nothing but be aware
  the etas are "jump-bar return" scales, not pure-jump scales.

---

## Part 2: Literature / improvement-plan alignment

Status of each plan item (Section 11 revised phases), judged against both the
plan text and the underlying synthesis findings.

| Item | Plan | Status | Alignment notes |
|---|---|---|---|
| 0 Basel baseline | Phase 0 | IMPLEMENTED | `core/validation/basel_backtest.py`, Teng-style traffic light, analytical + MC modes. Recommend running MC mode after any H1 fix to re-baseline. |
| 0 post-2019 training window | Phase 0 | IMPLEMENTED | `training_start_date="2019-10-01"` default; but see M5 (broken small-sample fallback). |
| 0.5 jump calibration | Phase 0.5 | IMPLEMENTED, flawed | Bipower (Lee-Mykland) detection is a genuine upgrade over MAD per the plan; but see M2 (mu_v confounding), M3 (contemporaneous return), M1 (rho_J as correlation-not-slope). |
| 1.1 naive prior mu=0 | Phase 1 | IMPLEMENTED | Correct: mu zeroed, jump compensator retained so E[log S_T]=log S0. Matches Baquero/Shelton intent. Doc wording ("median-anchored") slightly off (M4 note). |
| 1.2 3-state HMM | Phase 1 | IMPLEMENTED, shallow | hmmlearn GaussianHMM(3), weekly refit gate, leak-free as_of threading: all good. But emissions unused in simulation and labels are rank-based (H3). |
| 1.3 SVCJ vol jumps | Phase 1 | IMPLEMENTED, partial | Shared-Poisson co-timing correct per Eraker; FIGARCH persistence state (H3 fix) is a reasonable mean-reversion analog; rho_J channel inert (M1); H1 double-count means the SVCJ variance uplift stacks on an already-full GARCH variance. |
| 1.4 skewed-t | Phase 1 | IMPLEMENTED | Hansen sampler is correct (verified constants, piecewise inverse-CDF, unit variance; regression-tested). Lambdas hardcoded rather than estimated (low note). |
| 1.5 horizon gating | Phase 1 | IMPLEMENTED | Thresholds match BS1 for >90d and 30-90d (with a documented deviation: SVCJ/FIGARCH kept in 30-90d). Return jumps not disabled at >90d despite log text. Live path bypasses the gate entirely (H2). |
| 2.1 macro feed | Phase 2 | IMPLEMENTED | `core/data/macro_fetcher.py`, `DATA/macro_daily.csv`; consumed by XGB features with leak-safe date-join. |
| 2.2 macro-augmented HMM transitions | Phase 2 | NOT IMPLEMENTED | `RegimeDetector` uses returns only; no Gold/DXY/VIX in emissions or transitions. Macro reaches only the XGB classifier. Synthesis Finding 7 ranked this HIGH; currently the "macro entanglement" evidence is addressed solely by a default-off XGB tilt. |
| 2.3 directional XGB | Phase 2 | IMPLEMENTED, dormant | Drift-shift redesign is sound (strike-agnostic exp shift, empirical-CDF inversion, monotone ladder by construction, cap, deep-skew guard, physical-measure-only, DTE buckets). Default lambda=0 everywhere. Activation should wait for purged-CV evaluation (low note) and per-plan calibration grid. |
| 2.4 regime-conditional jumps | Phase 2 | IMPLEMENTED, hardcoded | Multipliers asserted, not estimated; `calibrate_regime_jumps` exists but unused in backtest for leak reasons (H3.2). |
| 2.5 FIGARCH | Phase 2 | IMPLEMENTED | Joint MLE via `arch`, ARCH-infinity weights match the library reference to 1e-12, B-M positivity checked with GARCH fallback, truncation K=1000. Sound. Note FIGARCH shares H1's double-count. |
| 2.6 vol-gate protocol | Phase 2 | IMPLEMENTED at strategy layer | Hard-gate override lives in auto_reco/vol_gate as designed; engine-side multipliers are dead code (low note). |
| 3.1 rolling evaluation | Phase 3 | PARTIAL | IS/OOS window module + walk-forward M2 calibration + Basel module collectively cover much of this; no single rolling-window Brier/VaR evaluator per Teng L=30/90 spec. |
| 3.2 Bayesian posteriors | Phase 3 | NOT IMPLEMENTED | Point estimates only, as planned for later. Becomes more attractive as the proper fix for H1 (joint estimation). |
| 3.3 on-chain features | Phase 3 | NOT IMPLEMENTED | Consistent with plan (moderate priority). |
| 3.4 coverage tracking | Phase 3 | PARTIAL | `core/validation/calibration_metrics.py` + M2 logit-shift table + dashboard calibration tab cover the intent. |
| 4.x power law / NH-HMM / meta-learner | Phase 4 | NOT IMPLEMENTED | Correctly deferred; >90d gate logs a warning instead of power-law anchoring. |

**Synthesis mandates ("INCLUDE MANDATORY") scorecard:** non-Gaussian innovations
(yes -- skew-t); double-exponential jumps (yes -- Kou, correctly aggregated);
regime switching (present but shallow, H3); naive prior (yes, cleanly);
SVCJ vol jumps (present; correlation channel inert, M1); FIGARCH (yes);
macro features (only via dormant XGB -- gap vs Finding 7).

---

## Part 3: What is sound (verified)

Worth stating explicitly, since much of the engine holds up well under scrutiny:

- **Compound Poisson aggregation.** Poisson count k, binomial thinning into
  up/down, Gamma(k, 1/eta) sums for the double-exponential magnitudes -- exactly
  right, and the multi-jump vs single-jump regression test (Test 1) guards it.
- **Jump-drift compensators.** Both the log-mean compensator
  lambda * ((1-p)/eta_up - p/eta_down) and the exponential-cumulant version are
  algebraically correct for Kou jumps (modulo M4's diffusion term, which is a
  scope issue, not an algebra error).
- **Hansen skewed-t sampler.** Constants (c, a, b), piecewise inverse-CDF, and
  the standardized-t inner quantile are all correct; mean-0/variance-1 verified
  by Tests 8-9.
- **FIGARCH weights.** Recurrence matches Chung/BBM and is validated against
  `arch`'s reference implementation at 1e-12; positivity check with clean GARCH
  fallback; sensible omega/(1-beta) intercept; eps^2 buffer warm-started at the
  fitted conditional variance.
- **SVCJ persistence under FIGARCH (FIX 5/H3).** The decaying vol-jump state with
  read-then-write ordering and a hard cap is a correct workaround for ARCH-infinity
  having no beta to carry a one-shot variance add; regression test 6b covers it.
- **Leak hygiene.** Strict `<` truncation on hourly/intraday, per-snapshot jump
  calibration from the truncated slice via `returns=`, as_of-threaded HMM refit
  gating, per-(date,bucket) XGB training on truncated data with macro sliced
  `< ts` -- this is unusually disciplined time-travel plumbing.
- **Proportional regime allocation.** Path counts proportional to posterior
  weights with independent sub-seeds preserves effective sample size and mixture
  independence (given the H3 caveat that the mixture components barely differ).
- **Numerical guards.** Per-step return clip, log-price clip, variance floors,
  eta_up>1 guard, sigma_H floor and deep-skew guard on the XGB shift, B-M weight
  check -- all reasonable and none distort the bulk of the distribution.
- **Student-t standardization.** sqrt((nu-2)/nu) scaling gives unit-variance
  innovations so sigma_t is the true conditional std; consistent between fit
  (arch standardizes internally) and simulation.

---

## Part 4: Recommended action order

1. **H1 -- jump-filtered GARCH fit** (small change, biggest statistical payoff).
   Then rerun the Basel MC backtest and the backtest pipeline to re-baseline;
   expect OTM model probabilities to drop and the favorite-longshot tail
   diagnostics to shift.
2. **H2 -- route live pipelines through `calculate_probabilities`.** Until then,
   treat backtest-derived calibrations as unvalidated for live.
3. **M2 + M3 -- fix mu_v post-window contamination and the LM contemporaneous
   term** (both are few-line fixes in `jump_calibration.py`), then recalibrate.
4. **H3 -- propagate regime weights via `predict_weights` and scale per-regime
   variance from HMM emissions.** Re-examine whether regime switching earns its
   complexity after H1/H2 (run the A/B with `--no-advanced-features`).
5. **M1 -- fix or remove the rho_J slope term**; document the co-timing channel
   as the operative SVCJ correlation either way.
6. Cleanups: M5 dead fallback, dead `vol_gate_regime` / `weights_array` /
   threshold constants, docstring corrections (martingale claim, "median-anchored"
   wording, >90d "no jumps" message, `fit_kou_params` return name).
7. Before ever activating XGB (lambda > 0): purged/embargoed CV for the accuracy
   gate and a decision on 0.5-anchored vs p_base-anchored tilt.

---

*Review conducted file-by-file against the synthesis meta-analysis and the
reviewed improvement plan; all line references are to the current working tree
(branch `main`, commit f904e4c).*
