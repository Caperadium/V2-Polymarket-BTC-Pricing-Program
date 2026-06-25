# Glossary

## Pricing Engine

**GARCH(1,1)**: Generalized AutoRegressive Conditional Heteroskedasticity model. Captures volatility clustering in asset returns. Parameters: ω (base variance), α (shock sensitivity), β (persistence).

**SVCJ**: Stochastic Volatility with Correlated Jumps. Extension of SVJ (Stochastic Volatility + Jumps) where volatility also jumps — and those jumps are correlated with return jumps. Critical for multi-day VaR calibration (Eraker et al. 2004, Teng et al. 2025).

**SVJ**: Stochastic Volatility + Jumps. The previous model — return jumps but no volatility jumps. SVCJ adds co-jumping volatility.

**Kou Jump Diffusion**: Double-exponential jump model. Jumps arrive via Poisson process; magnitudes drawn from asymmetric exponential (up vs down). Parameters: λ (intensity), p_crash (downward probability), η_up/η_down (decay rates).

**Naive Prior**: Enforcing μ=0 in GARCH fitting. OOS evidence (Baquero/Shelton 2024/2026) shows zero-drift GARCH forecasts outperform fitted-drift. Default on (`use_naive_prior=True`).

**Hansen Skewed-t**: Asymmetric student-t distribution (Hansen 1994). Parameter λ ∈ (-1,1) controls skew direction. Negative λ → left skew (crashes more likely). Optional in the pricing engine (`use_skewed_t=True`).

**FIGARCH(1,d,1)**: Fractionally Integrated GARCH per Baillie, Bollerslev & Mikkelsen (1996). Captures long-memory in volatility (hyperbolic autocorrelation decay vs exponential in standard GARCH). Parameters phi, d, beta estimated jointly via arch library from hourly BTC returns. Optional (`use_figarch=True`).

**Horizon Gating**: Scaling model complexity by time-to-expiry. T > 90d: naive prior only. T < 7d: full model (SVCJ + skewed-t + FIGARCH). Reduces over-parameterization of long-dated contracts.

**Regime-Conditional Pricing**: Running independent MC simulations for each HMM regime and weighting results by posterior probability. Bear regime uses scaled-up jump parameters; bull uses scaled-down.

## Regime Detection

**HMM (Hidden Markov Model)**: Statistical model where observed data depends on an unobserved (hidden) state sequence. Used here with 3 Gaussian states for BTC daily returns.

**Regime Weights**: Posterior probabilities of being in each regime (bear, sideways, bull) at the current time. Sum to 1.

**Transition Matrix**: 3×3 matrix T where T[i,j] = P(regime j tomorrow | regime i today). Powers of T give forward probabilities.

**Post-Hoc Weighting**: Running 3 independent simulations per regime and weighting terminal prices, rather than switching regimes mid-path. Simpler, avoids path-continuity issues.

## Jump Calibration

**MAD (Median Absolute Deviation)**: Robust measure of dispersion. MAD = median(|r_t - median|). Jump threshold = k × MAD (default k=3.0).

**Bipower Variation**: Jump-robust variance estimator. Ratio of realized variance to bipower variation tests for jump presence (Barndorff-Nielsen & Shephard).

**Kou Parameters**: λ (annual jump frequency), p_crash (fraction downward), η_up (1/mean positive jump size), η_down (1/mean negative jump magnitude).

**SVCJ Parameters**: μ_v (mean volatility jump size, hourly variance units), ρ_J (return-vol jump correlation, typically negative ~-0.08).

## Validation

**VaR (Value at Risk)**: Quantile of the loss distribution. 99% VaR = loss not exceeded with 99% probability.

**Kupiec POF Test**: Proportion of Failures test. Likelihood ratio test that observed VaR violation rate equals expected rate. Produces traffic light classification.

**Traffic Light**: Green (p > 0.05, adequate), Yellow (0.01 < p ≤ 0.05, borderline), Red (p ≤ 0.01, rejected).

**Expected Shortfall**: Average loss conditional on exceeding VaR. Acerbi-Szekely Z-statistics test whether tail severity matches expectation.

**ES (Expected Shortfall)**: Also called CVaR (Conditional Value at Risk). ES at 99% = mean loss given loss exceeds 99th percentile.

## Strategy

**Edge**: model_probability − market_price for YES contracts. Positive edge = model thinks contract is underpriced.

**Kelly Criterion**: Optimal bet sizing formula: f* = edge / odds. Fractional Kelly applies a multiplier (e.g., 0.15) for conservative sizing.

**Hysteresis**: Minimum edge decay before exiting a position. Prevents churn — edge must drop below (entry_edge − hysteresis) to trigger exit.

**Moneyness**: How far strike is from spot. ITM (in-the-money) if strike < spot for YES; OTM (out-of-the-money) if strike > spot.

**DTE (Days to Expiry)**: Time remaining until contract settlement. Also `hours_to_expiry` in the pricing engine.

## Directional XGBoost

**XGBoost**: eXtreme Gradient Boosting — tree-based ensemble classifier. Used for P(up) prediction with BTC momentum + macro features.

**Blend Weight**: 30% XGBoost + 70% SVCJ model (Shelton 2024 OOS evidence). XGBoost provides directional modifier to base distribution.

**Macro Features**: Gold price/returns, DXY (USD index) level/trend, VIX level, SPX returns. Top features by synthesis evidence weight: Gold (0.85), DXY (0.52), VIX (0.34).

## Execution

**CLOB**: Central Limit Order Book. Polymarket's order matching system.

**OrderIntent**: Dataclass representing a trade intention before submission. Contains deterministic ID for safe upserts.

**Collateral**: Funds locked as margin for open positions.

**Condition ID**: Polymarket's unique identifier for a binary market condition.

**Token ID**: CLOB-specific identifier for YES/NO outcome tokens within a condition.
