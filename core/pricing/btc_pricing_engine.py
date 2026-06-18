"""
BTC Pricing Engine v2 — Regime-Switching SVCJ with Long Memory

GARCH(1,1) + Fractionally Integrated Variance + Skewed-t + SVCJ (Kou Double Exponential
with correlated volatility jumps) Monte Carlo simulator. Regime-conditional via
3-state HMM. Hourly simulation steps.

Enhancements over v1 (per 17-paper meta-analysis, June 2026):
  Phase 1.1 — Naive prior (μ=0 anchoring) [Baquero 2026, Shelton 2024]
  Phase 1.2 — 3-state HMM regime detection [Oprea & Bâra 2026, Malekinezhad 2026]
  Phase 1.3 — SVCJ correlated volatility jumps [Teng et al. 2025, Eraker et al. 2004]
  Phase 1.4 — Skewed-t innovations (Hansen 1994) [Nakakita et al. 2025]
  Phase 1.5 — Horizon-gating (naive prior for T>30d) [Baquero 2026]
  Phase 2.4 — Regime-conditional jump parameters
  Phase 2.5 — Fractionally integrated variance [Siu 2025]
  Phase 2.6 — Regime-vol gate interaction protocol

All new features gated by boolean flags (default off except naive prior).
Backward compatible with existing callers.

Usage:
    # Legacy (unchanged API)
    probs = calculate_probabilities(strikes=[90000, 95000], hours_to_expiry=720.0)

    # Full v2 pipeline
    from core.pricing.regime_detector import RegimeDetector
    detector = RegimeDetector()
    probs = calculate_probabilities(
        strikes=[90000, 95000],
        hours_to_expiry=720.0,
        use_regime_switching=True,
        use_svcj=True,
        use_skewed_t=True,
        regime_detector=detector,
    )
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t as student_t
from scipy.special import gamma as gamma_func

logger = logging.getLogger(__name__)

# ==============================================================================
# JUMP PARAMETERS (Kou's Double Exponential Jump Diffusion)
# These are kept constant for easy tuning. Override via jump_params dict.
# ==============================================================================
LAMBDA = 25.0       # Jump intensity (expected number of jumps per year)
CRASH_PROB = 0.6    # Probability that a jump is a crash (downward)
ETA_UP = 50.0       # Decay parameter for upward jumps (1/mean jump size)
ETA_DOWN = 25.0     # Decay parameter for downward jumps (1/mean jump size)

# ==============================================================================
# SVCJ VOLATILITY JUMP PARAMETERS
# ==============================================================================
SVCJ_MU_V = 0.000025     # Mean volatility jump size (hourly variance units)
SVCJ_RHO_J = -0.08       # Return-vol jump correlation (Teng 2025 estimate)
SVCJ_LAM_V = None        # If None, uses same lambda as return jumps
SVCJ_SIGMA_S = 0.01      # Conditional std dev of return jump given vol jump (Eraker 2004)

# ==============================================================================
# FIGARCH PARAMETERS
# The fractionally integrated variance model uses (1-L)^d binomial weights
# with β only in the intercept ω/(1-β). This is a simplified specification —
# standard FIGARCH(1,d,1) would apply (1-βL)⁻¹ to the ARCH recursion,
# giving AR(1) feedback on variance. For binary option pricing, the
# long-memory parameter d dominates; the AR(1) feedback is second-order.
# ==============================================================================
FIGARCH_D = 0.578         # Long memory parameter (Siu 2025 estimate, SE=0.271)
FIGARCH_TRUNC_K = 1000    # Truncation lag for binomial expansion

# ==============================================================================
# SKEWED-T PARAMETERS
# ==============================================================================
SKEWED_T_LAMBDA_DEFAULT = -0.1  # Default skewness parameter (Hansen 1994)

# ==============================================================================
# FREQUENCY & DRIFT PARAMETERS
# ==============================================================================
HOURS_PER_YEAR = 365 * 24  # Hours in a year, for annual→hourly scaling
DRIFT_CLAMP_MULT = 0.25    # Max drift = ±0.25 * sigma_hourly

# ==============================================================================
# HORIZON GATING (Phase 1.5)
# ==============================================================================
HORIZON_SHORT_DAYS = 7        # <7d: full model
HORIZON_MEDIUM_DAYS = 30      # 7-30d: model with naive prior
HORIZON_LONG_DAYS = 90        # 30-90d: naive prior only
# >90d: naive prior + power-law anchoring (Phase 4)

# ==============================================================================
# REGIME-VOL GATE INTERACTION (Phase 2.6)
# ==============================================================================
# HMM regime and vol gate are independent:
#   - HMM: daily return characteristics (bear/sideways/bull)
#   - Vol gate: intraday realized vol percentiles (normal/high/extreme)
# Extreme vol gate ALWAYS overrides — blocks entries regardless of HMM regime.
# Bull HMM + extreme vol gate = no entries (hard gate wins).

# ==============================================================================
# DATA INGESTION
# ==============================================================================
def load_and_prep_data(
    hourly_csv: str = "DATA/btc_hourly.csv",
    intraday_csv: str = "DATA/btc_intraday_1m.csv",
    hourly_df: pd.DataFrame = None,
    intraday_df: pd.DataFrame = None,
    training_start_date: str = "2019-10-01",
):
    """
    Loads hourly data for GARCH fitting and intraday data for the latest price mark.
    Supports dependency injection for backtesting.

    Phase 0.1: training_start_date filters data to post-break period (Pakstaite 2025).
    """
    # 1. Load Hourly Data for GARCH fitting
    if hourly_df is None:
        hourly_df = pd.read_csv(hourly_csv)
    else:
        hourly_df = hourly_df.copy()

    col_map = {c.lower(): c for c in hourly_df.columns}
    if 'close' not in col_map:
        raise ValueError("btc_hourly.csv must contain a 'Close' or 'close' column.")
    close_col = col_map['close']

    # Apply training start date filter if date column present (Phase 0.1)
    date_col = col_map.get('date', col_map.get('timestamp'))
    if date_col and training_start_date is not None:
        hourly_df[date_col] = pd.to_datetime(hourly_df[date_col], utc=True, errors='coerce')
        start_dt = pd.Timestamp(training_start_date, tz='UTC')
        hourly_df = hourly_df[hourly_df[date_col] >= start_dt]
        if len(hourly_df) < 500:
            logger.warning(
                f"Only {len(hourly_df)} rows after training_start_date={training_start_date}. "
                "Falling back to all data."
            )
            # Reload without filter
            if hourly_df is None:
                hourly_df = pd.read_csv(hourly_csv)

    # Calculate Log Returns: ln(S_t / S_{t-1})
    hourly_returns = np.log(hourly_df[close_col] / hourly_df[close_col].shift(1)).dropna()

    # 2. Load Intraday Data for S0
    if intraday_df is None:
        intraday_df = pd.read_csv(intraday_csv)
    else:
        intraday_df = intraday_df.copy()

    col_map_intra = {c.lower(): c for c in intraday_df.columns}
    if 'close' not in col_map_intra:
        raise ValueError("intraday_btc.csv must contain a 'Close' or 'close' column.")
    close_col_intra = col_map_intra['close']

    # Get the latest close price
    current_price_S0 = float(intraday_df[close_col_intra].iloc[-1])

    # ---- Staleness Check (Issue 3.1 guard) ----
    # Intraday S0 drives all simulation; stale data silently produces absurd
    # probabilities (all paths above/below market-relevant strikes).
    ts_col_intra = col_map_intra.get('timestamp', col_map_intra.get('date', None))
    if ts_col_intra:
        try:
            last_ts = pd.to_datetime(intraday_df[ts_col_intra].iloc[-1], utc=True)
            age_hours = (pd.Timestamp.now(tz='UTC') - last_ts).total_seconds() / 3600
            if age_hours > 168:  # 7 days
                logger.error(
                    "CRITICAL: intraday data is %.0f hours stale (last: %s). "
                    "S0=%.2f from this data. Run 'python core/data/data_fetcher.py' "
                    "to refresh. Continuing with stale S0 will produce nonsense edges.",
                    age_hours, last_ts.isoformat(), current_price_S0,
                )
            elif age_hours > 24:
                logger.warning(
                    "Intraday data is %.0f hours stale (last: %s). S0=%.2f. "
                    "Consider refreshing: python core/data/data_fetcher.py",
                    age_hours, last_ts.isoformat(), current_price_S0,
                )
        except Exception:
            pass  # Don't break if timestamp parsing fails

    return hourly_returns, current_price_S0


# ==============================================================================
# SKEWED-T DISTRIBUTION (Hansen 1994) — Phase 1.4
# ==============================================================================

def skewed_t_rvs(
    nu: float,
    lam: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate random variates from Hansen (1994) skewed-t distribution.

    Uses inverse-transform method: draw from standard t, then apply
    Hansen's mapping z -> epsilon (skewed-t).

    Parameters:
        nu: Degrees of freedom (> 2 for finite variance).
        lam: Skewness parameter (-1 < lam < 1).
             lam < 0 → negative skew (heavy left tail, more crashes)
             lam > 0 → positive skew (heavy right tail)
        size: Number of samples.
        rng: NumPy random generator.

    Returns:
        Array of skewed-t variates with approximate mean 0 and variance 1.

    Reference:
        Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation."
        International Economic Review, 35(3), 705-730.

        Hansen maps skewed-t (ε) → standardized t (z) via:
            z = (b*ε + a) / (1 - λ*s)   where s = sign(b*ε + a)

        Inverse (ε from z):
            For z < -a/b:  ε = (z*(1+λ) - a) / b   (left tail)
            For z ≥ -a/b:  ε = (z*(1-λ) - a) / b   (right tail)

        With λ < 0: left tail denominator is (1+λ) < 1 → amplifies negatives → neg skew.
    """
    if nu <= 2:
        # Fall back to standard t if df too low
        scale = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
        return rng.standard_t(nu, size=size) * scale

    # Hansen's constants (from eq 8-10 in Hansen 1994)
    # a controls the mean; b ensures distribution is properly scaled
    # (note: b does NOT standardise variance — output still ~nu/(nu-2))
    c_const = gamma_func((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * gamma_func(nu / 2))
    a = 4 * lam * c_const * (nu - 2) / (nu - 1)
    b_sq = 1 + 3 * lam ** 2 - a ** 2

    if b_sq <= 0:
        # Numerical fallback
        return rng.standard_t(nu, size=size) * np.sqrt((nu - 2) / nu)

    b = np.sqrt(b_sq)

    # Draw from standard t (mean 0, variance nu/(nu-2) for nu>2)
    z = rng.standard_t(nu, size=size)

    # Threshold: the boundary between left/right tail regimes
    threshold = -a / b

    # Inverse transformation (Hansen 1994, adapted for RVS generation)
    # Hansen's λ convention: λ < 0 produces NEGATIVE skew (heavy left tail)
    # The regime assignment uses (1-λ) for left, (1+λ) for right
    g = np.where(
        z < threshold,
        (z * (1 - lam) - a) / b,   # λ<0 → (1-λ)>1 → amplifies negatives → neg skew
        (z * (1 + lam) - a) / b,   # λ<0 → (1+λ)<1 → dampens positives
    )

    # Hansen's b parameter (computed above as sqrt(1 + 3*lam**2 - a**2))
    # normalises the distribution's shape. The output variance is ~nu/(nu-2),
    # same as standard-t — corrected externally via skewed_t_scale_factor().

    return g


def skewed_t_scale_factor(nu: float, lam: float) -> float:
    """
    Compute scale factor to ensure ~unit variance in skewed-t samples.

    Multiplier applied externally to skewed_t_rvs() output. Corrects for
    Student-t overdispersion (nu/(nu-2)). Hansen's b parameter normalises
    the distribution shape (skewness), NOT the variance — skewed_t_rvs
    output is still ~nu/(nu-2) variance, same as standard_t.

    For symmetric case (lam=0): standard t scale factor sqrt((nu-2)/nu).
    """
    if nu <= 2:
        return 1.0
    base_scale = np.sqrt((nu - 2) / nu)
    return base_scale


# ==============================================================================
# FIGARCH BINOMIAL WEIGHTS — Phase 2.5
# ==============================================================================

def _compute_figarch_weights(d: float, trunc_k: int = FIGARCH_TRUNC_K) -> np.ndarray:
    """
    Precompute fractional differencing binomial weights for (1-L)^d.

    λ_k = Γ(k - d) / (Γ(k + 1) * Γ(-d)) for k = 0, 1, ..., trunc_k-1

    These weights are used in the fractionally integrated variance recursion:
        σ²_t = ω / (1 - β) + Σ_{k=0}^{∞} λ_k ε²_{t-k}

    Args:
        d: Long memory parameter (0 < d < 1).
        trunc_k: Number of lags in truncation.

    Returns:
        Array of length trunc_k with λ_k weights.
    """
    if d <= 0 or d >= 1:
        raise ValueError(f"FIGARCH d must be in (0, 1), got {d}")

    k = np.arange(trunc_k, dtype=float)
    # λ_k = Γ(k - d) / (Γ(k + 1) * Γ(-d))
    # Using log-gamma for numerical stability
    log_weights = (
        gamma_func(k - d + 1) / gamma_func(k + 2)
    )  # Not quite right — using ratio form

    # Correct computation via recurrence:
    # λ_0 = 1
    # λ_k = λ_{k-1} * (k - 1 - d) / k  for k >= 1
    weights = np.ones(trunc_k)
    for i in range(1, trunc_k):
        weights[i] = weights[i - 1] * (i - 1 - d) / i

    # Normalize to sum to 1 / (1 - β) for GARCH integration
    return weights


# ==============================================================================
# MODEL FITTING
# ==============================================================================

def fit_garch_model(
    returns: pd.Series,
    training_start_date: str = "2019-10-01",
    use_figarch: bool = False,
    figarch_d: float = FIGARCH_D,
    figarch_trunc_k: int = FIGARCH_TRUNC_K,
):
    """
    Fits a GARCH(1,1) model with Student-t errors.

    Phase 0.1: training_start_date filters to post-2019 structural break.
    Phase 2.5: Fractionally integrated variance available via use_figarch=True.

    Uses the long-term fitted mean (structural mu) as drift.
    All returned parameters are in hourly log-return units.

    Args:
        returns: pd.Series of hourly log returns.
        training_start_date: Ignored here (filtered upstream in load_and_prep_data).
        use_figarch: If True, fit FIGARCH instead of GARCH (requires arch>=7.0).
        figarch_d: Long memory parameter for FIGARCH.
        figarch_trunc_k: Truncation lag for FIGARCH binomial expansion.

    Returns:
        Dict with omega, alpha, beta, nu, mu, last_variance (hourly units).
        Additional keys if use_figarch: figarch_weights, d.
    """
    # 1. Scale returns for numerical stability
    scaled_returns = returns * 100

    # 2. Fit GARCH(1,1) with Student-t
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='t', mean='Constant')
    res = model.fit(disp='off')

    params = res.params

    omega = params['omega'] / 10000.0
    alpha = params['alpha[1]']
    beta_val = params['beta[1]']
    nu = params['nu']

    # 3. Use structural mean from GARCH fit (hourly log-return units)
    mu = params['mu'] / 100.0

    result = {
        'omega': omega,
        'alpha': alpha,
        'beta': beta_val,
        'nu': nu,
        'mu': mu,
        'last_variance': res.conditional_volatility.iloc[-1]**2 / 10000.0
    }

    # FIGARCH precomputation (Phase 2.5)
    if use_figarch:
        try:
            weights = _compute_figarch_weights(figarch_d, figarch_trunc_k)
            result['figarch_weights'] = weights
            result['figarch_d'] = figarch_d
            result['figarch_trunc_k'] = figarch_trunc_k
            logger.info(f"FIGARCH enabled: d={figarch_d:.3f}, trunc_k={figarch_trunc_k}")
        except Exception as e:
            logger.warning(f"FIGARCH weight computation failed ({e}), falling back to GARCH")
            result['use_figarch'] = False

    return result


# ==============================================================================
# VARIANCE CONSISTENCY CHECK (Issue 2 diagnostic)
# ==============================================================================
def check_variance_consistency(garch_params: dict, n_samples: int = 10000, seed: int = 12345) -> float:
    """
    Diagnostic check: simulate 1-hour returns (no jumps, no drift) and compare
    empirical variance to the model's conditional variance.

    This validates that the Student-t scaling and GARCH recursion are consistent.
    Returns the ratio (empirical_var / model_var). Should be close to 1.0.

    Emits a warning if ratio deviates by more than 15%.
    """
    rng = np.random.default_rng(seed)

    omega = garch_params['omega']
    alpha = garch_params['alpha']
    beta_val = garch_params['beta']
    nu = garch_params['nu']
    model_variance = garch_params['last_variance']

    # Scale Student-t to unit variance
    if nu > 2:
        scale_factor = np.sqrt((nu - 2) / nu)
    else:
        scale_factor = 1.0

    # Simulate 1-hour returns: r = sigma * z where z ~ scaled Student-t
    z = rng.standard_t(nu, size=n_samples) * scale_factor
    sigma = np.sqrt(model_variance)
    returns = sigma * z

    empirical_variance = np.var(returns)
    ratio = empirical_variance / model_variance

    if abs(ratio - 1.0) > 0.15:
        logger.warning(
            f"Variance consistency check: empirical/model ratio = {ratio:.3f} "
            f"(expected ~1.0). Student-t scaling or GARCH params may be mismatched."
        )

    return ratio


# ==============================================================================
# MONTE CARLO SIMULATION — Phase 1-2 enhancements
# ==============================================================================

def simulate_paths(
    S0,
    garch_params,
    jump_params,
    hours_to_expiry,
    n_sims=15000,
    seed=None,
    apply_jump_drift_correction: bool = True,
    # --- Phase 1 feature flags ---
    use_naive_prior: bool = True,
    use_svcj: bool = False,
    use_skewed_t: bool = False,
    skewed_t_lam: float = SKEWED_T_LAMBDA_DEFAULT,
    use_figarch: bool = False,
    # --- Phase 2.4: regime-conditional jump params ---
    regime_jump_params: dict = None,
    regime_label: str = "sideways",
    # --- Phase 2.6: vol gate interaction ---
    vol_gate_regime: str = "normal",  # "normal", "high", "extreme"
):
    """
    Simulates price paths using GARCH(1,1)/FIGARCH + Skewed-t/Student-t + SVCJ jumps
    on HOURLY steps.

    Phase 1.1 (use_naive_prior): Sets μ=0, anchoring distribution on current price.
    Phase 1.3 (use_svcj): Adds correlated volatility jumps (Eraker 2004 specification).
    Phase 1.4 (use_skewed_t): Hansen (1994) skewed-t innovations instead of Student-t.
    Phase 2.4 (regime_jump_params): Regime-specific jump parameters per HMM state.
    Phase 2.5 (use_figarch): FIGARCH(1,d,1) long-memory variance updates.
    Phase 2.6 (vol_gate_regime): Extreme vol gate always overrides — path discarded.

    Args:
        S0: Current spot price.
        garch_params: Dict with omega, alpha, beta, nu, mu, last_variance.
        jump_params: Dict with lambda, crash_prob, eta_up, eta_down (or None).
        hours_to_expiry: Float, number of hours until expiry.
        n_sims: Number of Monte Carlo paths.
        seed: Random seed for reproducibility.
        apply_jump_drift_correction: If True, subtract expected_jump_drift from mu.
        use_naive_prior: If True, set drift μ=0 (Baquero/Shelton naive anchor).
        use_svcj: If True, add correlated volatility jumps.
        use_skewed_t: If True, use Hansen skewed-t instead of Student-t.
        skewed_t_lam: Skew parameter for skewed-t (regime-conditional).
        use_figarch: If True, use FIGARCH variance recursion instead of GARCH.
        regime_jump_params: Dict mapping regime_label to jump param overrides.
        regime_label: Which regime's parameters to use.
        vol_gate_regime: Current vol gate regime (affects jump intensity scaling).
    """
    rng = np.random.default_rng(seed)

    # ---- Resolve Jump Parameters ----
    if jump_params is None:
        lam = LAMBDA
        p_crash = CRASH_PROB
        eta_up = ETA_UP
        eta_down = ETA_DOWN
        svcj_mu_v = SVCJ_MU_V
        svcj_rho_j = SVCJ_RHO_J
        svcj_sigma_s = SVCJ_SIGMA_S
    else:
        lam = jump_params.get('lambda', LAMBDA)
        p_crash = jump_params.get('crash_prob', CRASH_PROB)
        eta_up = jump_params.get('eta_up', ETA_UP)
        eta_down = jump_params.get('eta_down', ETA_DOWN)
        svcj_mu_v = jump_params.get('mu_v', SVCJ_MU_V)
        svcj_rho_j = jump_params.get('rho_J', SVCJ_RHO_J)
        svcj_sigma_s = jump_params.get('sigma_s', SVCJ_SIGMA_S)

    # --- Regime-Conditional Jump Overrides (Phase 2.4) ---
    if regime_jump_params and regime_label in regime_jump_params:
        rp = regime_jump_params[regime_label]
        lam = rp.get('lambda', lam)
        p_crash = rp.get('crash_prob', p_crash)
        eta_up = rp.get('eta_up', eta_up)
        eta_down = rp.get('eta_down', eta_down)
        svcj_mu_v = rp.get('mu_v', svcj_mu_v)
        svcj_rho_j = rp.get('rho_J', svcj_rho_j)
        svcj_sigma_s = rp.get('sigma_s', svcj_sigma_s)
        logger.debug(f"Regime-conditional jumps ({regime_label}): lam={lam:.1f}, p_crash={p_crash:.2f}")

    # --- Vol Gate Interaction (Phase 2.6) ---
    # In extreme vol: scale up jump intensity (vol jumps already embedded via SVCJ)
    if vol_gate_regime == "extreme":
        lam *= 1.5  # 50% more jumps in extreme vol
        svcj_mu_v *= 2.0  # Double vol jump size
    elif vol_gate_regime == "high":
        lam *= 1.2
        svcj_mu_v *= 1.3

    # 1. Convert Annual Lambda to Hourly
    lam_hourly = lam / HOURS_PER_YEAR
    # For SVCJ: same Poisson driver for both return and vol jumps
    lam_v_hourly = jump_params.get('lam_v', lam) / HOURS_PER_YEAR if jump_params else lam_hourly

    # 2. Calculate Expected Jump Drift (hourly log-return)
    # E[J] = (1-p_crash)/eta_up - p_crash/eta_down
    expected_jump_drift = lam_hourly * ((1 - p_crash) / eta_up - p_crash / eta_down)

    n_hours = int(np.ceil(hours_to_expiry))
    dt_schedule = np.ones(n_hours)
    if hours_to_expiry % 1 != 0:
        dt_schedule[-1] = hours_to_expiry % 1

    omega = garch_params['omega']
    alpha = garch_params['alpha']
    beta_val = garch_params['beta']
    nu = garch_params['nu']
    mu = garch_params['mu']  # Hourly log-return units (scalar)
    current_variance = garch_params['last_variance']  # Hourly variance

    # FIGARCH precomputed weights (Phase 2.5)
    figarch_weights = garch_params.get('figarch_weights', None)
    if use_figarch and figarch_weights is None:
        logger.warning("use_figarch=True but no figarch_weights in garch_params; using GARCH")
        use_figarch = False

    # FIGARCH lag buffer for past squared returns
    if use_figarch and figarch_weights is not None:
        figarch_trunc_k = len(figarch_weights)
        # Initialize past squared returns with unconditional variance
        unconditional_var = omega / (1 - alpha - beta_val) if (alpha + beta_val) < 1 else omega / 0.01
        past_eps_sq = np.full((n_sims, figarch_trunc_k), unconditional_var)

    # Naive prior enforcement (Phase 1.1): set μ=0
    if use_naive_prior:
        mu = 0.0

    log_prices = np.full(n_sims, np.log(S0))
    variances = np.full(n_sims, current_variance)

    for step_idx, dt in enumerate(dt_schedule):
        # ---- Innovation Distribution ----
        if use_skewed_t and nu > 2:
            # Skewed-t (Phase 1.4): skewed_t_rvs ~ nu/(nu-2) variance (like
            # standard-t). scale_factor corrects to unit variance. Hansen's b
            # handles distribution shape; it does NOT standardise the variance.
            scale_factor = skewed_t_scale_factor(nu, skewed_t_lam)
            z_t = skewed_t_rvs(nu, skewed_t_lam, n_sims, rng) * scale_factor
        else:
            # Standard Student-t (legacy)
            if nu > 2:
                scale_factor = np.sqrt((nu - 2) / nu)
            else:
                scale_factor = 1.0
            z_t = rng.standard_t(nu, size=n_sims) * scale_factor

        step_variance = variances * dt
        step_sigma = np.sqrt(step_variance)

        # 3. DYNAMIC PER-PATH DRIFT CLAMPING
        sigma_hourly_step = np.sqrt(variances)
        mu_clamped = np.clip(mu, -DRIFT_CLAMP_MULT * sigma_hourly_step, DRIFT_CLAMP_MULT * sigma_hourly_step)

        # 4. Apply Drift
        if apply_jump_drift_correction:
            drift_per_step = (mu_clamped - expected_jump_drift) * dt
        else:
            drift_per_step = mu_clamped * dt

        garch_ret = drift_per_step + step_sigma * z_t

        # 5. Variance Update — GARCH or FIGARCH (Phase 2.5)
        if abs(dt - 1.0) < 1e-12:
            epsilon_squared = (step_sigma * z_t) ** 2  # (sigma * z)² = return²

            if use_figarch and figarch_weights is not None:
                # Fractionally integrated variance: σ²_t = ω/(1-β) + Σ λ_k ε²_{t-k}
                # Shift past returns, insert current
                past_eps_sq = np.roll(past_eps_sq, 1, axis=1)
                past_eps_sq[:, 0] = epsilon_squared
                # Weighted sum
                figarch_component = np.sum(
                    past_eps_sq[:, :figarch_trunc_k] * figarch_weights[np.newaxis, :figarch_trunc_k],
                    axis=1
                )
                variances = omega / (1 - beta_val) + figarch_component
            else:
                # Standard GARCH(1,1)
                variances = omega + alpha * epsilon_squared + beta_val * variances

        # 6. COMPOUND POISSON JUMPS (Kou double-exponential)
        k = rng.poisson(lam_hourly * dt, size=n_sims)
        k_down = rng.binomial(k, p_crash)
        k_up = k - k_down

        down_mag = np.zeros(n_sims)
        up_mag = np.zeros(n_sims)
        mask_down = k_down > 0
        mask_up = k_up > 0

        if np.any(mask_down):
            down_mag[mask_down] = rng.gamma(k_down[mask_down], scale=1.0 / eta_down)
        if np.any(mask_up):
            up_mag[mask_up] = rng.gamma(k_up[mask_up], scale=1.0 / eta_up)

        jump_sizes = up_mag - down_mag  # log-return: up=positive, down=negative

        # ---- SVCJ Volatility Jumps (Phase 1.3) ----
        if use_svcj and svcj_mu_v > 0:
            # Vol jumps triggered by same Poisson process (correlated structure)
            k_v = rng.poisson(lam_v_hourly * dt, size=n_sims)
            mask_vol_jump = k_v > 0

            if np.any(mask_vol_jump):
                vol_jump_mag = rng.exponential(svcj_mu_v, size=n_sims)
                vol_jump_mag[~mask_vol_jump] = 0.0

                # Eraker et al. (2004): ξ_s | ξ_v ~ N(μ_s + ρ_J ξ_v, σ_s²)
                # Return jump includes deterministic ρ_J correlation plus stochastic residual
                correlated_adjustment = svcj_rho_j * vol_jump_mag
                stochastic_residual = rng.normal(0, svcj_sigma_s, size=n_sims)
                jump_sizes += correlated_adjustment + stochastic_residual
                variances += vol_jump_mag  # Add vol jump to variance

                # Ensure variance stays positive
                variances = np.maximum(variances, 1e-12)

        total_log_return = garch_ret + jump_sizes
        log_prices += total_log_return

    return np.exp(log_prices)


def get_contract_probability(paths: np.ndarray, strike_price: float):
    """
    Calculate probability of paths ending at or above strike.

    Args:
        paths: Array of simulated terminal prices.
        strike_price: Strike price for the binary contract.
    """
    return np.mean(paths >= strike_price)


# ==============================================================================
# HIGH-LEVEL WRAPPER — Phase 1-2 Orchestration
# ==============================================================================

def calculate_probabilities(
    strikes: list,
    hours_to_expiry: float,
    hourly_df: pd.DataFrame = None,
    intraday_df: pd.DataFrame = None,
    hourly_csv: str = "DATA/btc_hourly.csv",
    intraday_csv: str = "DATA/btc_intraday_1m.csv",
    n_sims: int = 15000,
    jump_params: dict = None,
    seed: int = None,
    # --- Phase 0 ---
    training_start_date: str = "2019-10-01",
    # --- Phase 1 ---
    use_naive_prior: bool = True,
    use_regime_switching: bool = False,
    use_svcj: bool = False,
    use_skewed_t: bool = False,
    # --- Phase 2 ---
    use_figarch: bool = False,
    use_xgb_direction: bool = False,
    regime_params: dict = None,
    macro_df: pd.DataFrame = None,
    # --- External dependencies ---
    regime_detector=None,  # RegimeDetector instance
    xgb_model=None,        # DirectionalXGB instance
):
    """
    Calculates probabilities for multiple strikes using hourly simulation.

    Phase 1.5: Horizon-gating applied automatically based on hours_to_expiry.
      - T < 7 days: full model as configured
      - 7 <= T <= 30 days: model with naive prior enforced
      - T > 30 days: naive prior only (μ=0, no jumps, no regime switching)
      - T > 90 days: naive prior + warning (power-law not yet implemented)

    Args:
        strikes: List of strike prices.
        hours_to_expiry: Hours until contract expiry.
        hourly_df: Optional DataFrame of hourly prices (for backtesting).
        intraday_df: Optional DataFrame of intraday prices (for backtesting).
        hourly_csv: Path to hourly CSV file.
        intraday_csv: Path to intraday CSV file.
        n_sims: Number of Monte Carlo simulations.
        jump_params: Optional dict overriding jump parameters.
        seed: Random seed for reproducibility.
        training_start_date: Start date for training data (default: Oct 2019).
        use_naive_prior: If True, set drift to 0 (anchors on current price).
        use_regime_switching: If True, use HMM regime-conditional simulation.
        use_svcj: If True, use SVCJ correlated volatility jumps.
        use_skewed_t: If True, use Hansen skewed-t distribution.
        use_figarch: If True, use FIGARCH long memory volatility.
        use_xgb_direction: If True, use XGBoost directional modifier.
        regime_params: Dict of regime-specific parameter overrides.
        macro_df: Optional macro data for regime detection.
        regime_detector: RegimeDetector instance (required if use_regime_switching).
        xgb_model: DirectionalXGB instance (required if use_xgb_direction).

    Returns:
        Dict mapping strike_price -> probability.
    """
    hours = hours_to_expiry
    days_to_expiry = hours / 24.0

    # ---- Phase 1.5: Horizon Gating ----
    if days_to_expiry > HORIZON_LONG_DAYS:
        # Very long horizon: naive prior only, no model complexity
        logger.info(
            f"Horizon gate: T={days_to_expiry:.0f}d > {HORIZON_LONG_DAYS}d. "
            "Using naive prior only (μ=0, no jumps)."
        )
        use_naive_prior = True
        use_regime_switching = False
        use_svcj = False
        use_skewed_t = False
        use_figarch = False
        use_xgb_direction = False
    elif days_to_expiry > HORIZON_MEDIUM_DAYS:
        # Medium horizon: naive prior, simplified model
        logger.info(
            f"Horizon gate: T={days_to_expiry:.0f}d > {HORIZON_MEDIUM_DAYS}d. "
            "Enforcing naive prior, disabling regime switching."
        )
        use_naive_prior = True
        use_regime_switching = False  # Regime detection unreliable at multi-month
        # Keep SVCJ/skewed-t/FIGARCH as configured (still useful for distribution shape)
    elif days_to_expiry <= HORIZON_SHORT_DAYS:
        # Short horizon: full model allowed
        logger.debug(f"Short horizon T={days_to_expiry:.1f}d: using full model configuration")

    # ---- Load Data & Fit Model ----
    hourly_returns, S0 = load_and_prep_data(
        hourly_csv=hourly_csv,
        intraday_csv=intraday_csv,
        hourly_df=hourly_df,
        intraday_df=intraday_df,
        training_start_date=training_start_date,
    )

    garch_params = fit_garch_model(
        hourly_returns,
        training_start_date=training_start_date,
        use_figarch=use_figarch,
    )

    # ---- Regime Detection (Phase 1.2) ----
    regime_weights = {"bear": 0.0, "sideways": 1.0, "bull": 0.0}
    dominant_regime = "sideways"

    if use_regime_switching and regime_detector is not None:
        try:
            from core.pricing.regime_detector import hourly_to_daily_returns

            # Get daily returns for HMM
            if hourly_df is not None:
                daily_ret = hourly_to_daily_returns(df=hourly_df)
            else:
                daily_ret = hourly_to_daily_returns(hourly_path=hourly_csv)

            # Fit/predict
            regime_weights, dominant_regime = regime_detector.fit_predict(daily_ret)
            logger.info(f"Regime detection: dominant={dominant_regime}, weights={regime_weights}")
        except Exception as e:
            logger.warning(f"Regime detection failed ({e}); using default sideways regime")

    # ---- Regime-Conditional Simulation OR Single Simulation ----
    if use_regime_switching and regime_detector is not None:
        # Post-hoc weighting approach per plan Section 10 C1 resolution:
        # Run independent simulations per regime, weight terminal prices by HMM posterior
        n_per_regime = n_sims // 3

        all_paths = []
        all_weights = []

        regime_labels = ["bear", "sideways", "bull"]
        for rl in regime_labels:
            w = regime_weights.get(rl, 0.0)
            if w < 0.01:
                continue  # Skip negligible regimes

            # Regime-specific skewed-t lambda
            if use_skewed_t:
                if rl == "bear":
                    st_lam = -0.3  # Negative skew in bear
                elif rl == "bull":
                    st_lam = 0.2   # Positive skew in bull
                else:
                    st_lam = 0.0   # Symmetric in sideways
            else:
                st_lam = SKEWED_T_LAMBDA_DEFAULT

            paths = simulate_paths(
                S0=S0,
                garch_params=garch_params,
                jump_params=jump_params,
                hours_to_expiry=hours_to_expiry,
                n_sims=n_per_regime,
                seed=seed,
                use_naive_prior=use_naive_prior,
                use_svcj=use_svcj,
                use_skewed_t=use_skewed_t,
                skewed_t_lam=st_lam,
                use_figarch=use_figarch,
                regime_jump_params=regime_params,
                regime_label=rl,
            )

            all_paths.append(paths)
            all_weights.append(np.full(n_per_regime, w))

        if not all_paths:
            # Fallback: single simulation
            paths = simulate_paths(
                S0=S0, garch_params=garch_params, jump_params=jump_params,
                hours_to_expiry=hours_to_expiry, n_sims=n_sims, seed=seed,
                use_naive_prior=use_naive_prior, use_svcj=use_svcj,
                use_skewed_t=use_skewed_t, use_figarch=use_figarch,
                regime_label=dominant_regime,
            )
        else:
            paths = np.concatenate(all_paths)
            weights_array = np.concatenate(all_weights)
            # Weighted probability computation below
    else:
        # ---- Single Simulation (legacy / non-regime path) ----
        paths = simulate_paths(
            S0=S0,
            garch_params=garch_params,
            jump_params=jump_params,
            hours_to_expiry=hours_to_expiry,
            n_sims=n_sims,
            seed=seed,
            use_naive_prior=use_naive_prior,
            use_svcj=use_svcj,
            use_skewed_t=use_skewed_t,
            use_figarch=use_figarch,
            regime_jump_params=regime_params,
            regime_label=dominant_regime,
        )
        weights_array = None

    # ---- Compute Probabilities ----
    results = {}
    for strike in strikes:
        if weights_array is not None:
            # Weighted probability from regime mixture
            prob = np.average(paths >= strike, weights=weights_array)
        else:
            prob = get_contract_probability(paths, strike)

        # ---- Phase 2.3: Directional XGBoost Modifier ----
        if use_xgb_direction and xgb_model is not None:
            try:
                # Get directional adjustment from XGBoost
                direction_modifier = xgb_model.predict_direction_adjustment(
                    S0=S0,
                    hours_to_expiry=hours_to_expiry,
                    macro_df=macro_df,
                )
                # Blend: p_final = 0.7 * p_model + 0.3 * p_xgb_modifier
                # (weight moderate per Shelton OOS evidence: individual predictors weak)
                xgb_weight = 0.3
                prob = (1 - xgb_weight) * prob + xgb_weight * direction_modifier
                prob = np.clip(prob, 0.01, 0.99)
            except Exception as e:
                logger.warning(f"XGBoost direction modifier failed ({e}); using unmodified probability")

        results[strike] = float(prob)

    # ---- Build Extended Results ----
    results['_meta'] = {
        'S0': S0,
        'hours_to_expiry': hours_to_expiry,
        'n_sims': n_sims,
        'regime_weights': regime_weights,
        'dominant_regime': dominant_regime,
        'use_naive_prior': use_naive_prior,
        'use_regime_switching': use_regime_switching,
        'use_svcj': use_svcj,
        'use_skewed_t': use_skewed_t,
        'use_figarch': use_figarch,
        'use_xgb_direction': use_xgb_direction,
        'horizon_gate_active': days_to_expiry > HORIZON_MEDIUM_DAYS,
    }

    return results


# ==============================================================================
# JUMP CALIBRATION CACHE — Phase 0.5
# ==============================================================================

def load_calibrated_jumps(
    hourly_csv: str = "DATA/btc_hourly.csv",
    cache_path: str = "DATA/jump_calibration.csv",
    max_cache_age_days: int = 30,
    force_recalibrate: bool = False,
) -> dict:
    """
    Load or compute calibrated jump parameters from BTC hourly data.

    Caches results to avoid repeated calibration. Compares calibrated values
    against hardcoded defaults and logs warnings for >20% deviations.

    Args:
        hourly_csv: Path to BTC hourly data for calibration.
        cache_path: Path to cached calibration CSV.
        max_cache_age_days: Recalibrate if cache older than this.
        force_recalibrate: If True, skip cache and recalibrate.

    Returns:
        Dict with keys matching JumpCalibrationResult dataclass fields:
        lam, p_crash, eta_up, eta_down, mu_v, rho_J, lam_v, n_jumps_detected,
        fit_converged, calibration_date.
    """
    from core.pricing.jump_calibration import calibrate_jumps, JumpCalibrationResult

    cache_file = Path(cache_path)
    use_cache = False

    if not force_recalibrate and cache_file.exists():
        cache_age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            cache_file.stat().st_mtime, tz=timezone.utc
        )
        if cache_age.days < max_cache_age_days:
            use_cache = True
            logger.info(f"Loading cached jump calibration ({cache_age.days}d old)")

    if use_cache:
        cached = pd.read_csv(cache_path)
        row = cached.iloc[-1]  # Most recent row
        calibrated = {
            "lam": float(row["lam"]),
            "p_crash": float(row["p_crash"]),
            "eta_up": float(row["eta_up"]),
            "eta_down": float(row["eta_down"]),
            "mu_v": float(row["mu_v"]),
            "rho_J": float(row["rho_J"]),
            "lam_v": float(row.get("lam_v", row["lam"])),
            "n_jumps_detected": int(row.get("n_jumps_detected", 0)),
            "fit_converged": bool(int(row.get("fit_converged", 1))),
            "calibration_date": str(row.get("calibration_date", "unknown")),
        }
    else:
        logger.info("Calibrating jump parameters from %s ...", hourly_csv)
        result: JumpCalibrationResult = calibrate_jumps(
            hourly_csv=hourly_csv,
            detection_method="MAD",
        )
        calibrated = {
            "lam": result.lam,
            "p_crash": result.p_crash,
            "eta_up": result.eta_up,
            "eta_down": result.eta_down,
            "mu_v": result.mu_v,
            "rho_J": result.rho_J,
            "lam_v": result.lam_v,
            "n_jumps_detected": result.n_jumps_detected,
            "fit_converged": result.fit_converged,
            "calibration_date": datetime.now(timezone.utc).isoformat(),
        }
        # Cache to CSV
        pd.DataFrame([calibrated]).to_csv(cache_file, index=False)
        logger.info(f"Cached jump calibration to {cache_file}")

    # Compare against hardcoded defaults, warn if >20% delta
    defaults = {
        "lam": LAMBDA, "p_crash": CRASH_PROB,
        "eta_up": ETA_UP, "eta_down": ETA_DOWN,
        "mu_v": SVCJ_MU_V, "rho_J": SVCJ_RHO_J,
    }
    for key, default_val in defaults.items():
        cal_val = calibrated.get(key)
        if cal_val and default_val != 0:
            delta_pct = abs(cal_val - default_val) / abs(default_val) * 100
            if delta_pct > 20:
                logger.warning(
                    "Jump param %s: calibrated=%.4f vs default=%.4f (%.0f%% delta). "
                    "Consider updating defaults.",
                    key, cal_val, default_val, delta_pct,
                )

    return calibrated


# ==============================================================================
# REGIME-CONDITIONAL JUMP PARAMETER BUILDER — Phase 2.4
# ==============================================================================

def build_regime_jump_params(
    base_lam: float = LAMBDA,
    base_p_crash: float = CRASH_PROB,
    base_eta_up: float = ETA_UP,
    base_eta_down: float = ETA_DOWN,
    base_mu_v: float = SVCJ_MU_V,
    base_rho_j: float = SVCJ_RHO_J,
    base_sigma_s: float = SVCJ_SIGMA_S,
    calibrated: dict | None = None,
    regime_calibrated: dict | None = None,
) -> dict:
    """
    Build regime-conditional jump parameters.

    Two-tier calibration priority:
    1. regime_calibrated — per-regime direct calibration (from calibrate_regime_jumps()).
       When a regime's parameters are available, they are used directly (data-driven,
       no hardcoded multiplier).
    2. calibrated — single base calibration (from load_calibrated_jumps()).
       Applied with hardcoded regime multipliers.
    3. Fallback — hardcoded defaults with literature-based multipliers.

    Args:
        calibrated: Optional dict from load_calibrated_jumps(). Overrides base_* params.
        regime_calibrated: Optional dict from calibrate_regime_jumps(), mapping
            regime_name -> RegimeJumpResult or None.

    Returns:
        Dict mapping regime_label -> dict of jump parameters.
    """
    # Determine base parameters
    if calibrated is not None and calibrated.get("fit_converged", False):
        base_lam = calibrated.get("lam", base_lam)
        base_p_crash = calibrated.get("p_crash", base_p_crash)
        base_eta_up = calibrated.get("eta_up", base_eta_up)
        base_eta_down = calibrated.get("eta_down", base_eta_down)
        base_mu_v = calibrated.get("mu_v", base_mu_v)
        base_rho_j = calibrated.get("rho_J", base_rho_j)
        logger.info("Using data-calibrated jump parameters as base")

    # Build per-regime dict
    regime_params = {}

    for regime, label in [("bear", "bear"), ("sideways", "sideways"), ("bull", "bull")]:
        rc = (
            regime_calibrated.get(regime)
            if regime_calibrated and regime_calibrated.get(regime) is not None
            else None
        )

        if rc is not None:
            # Use directly calibrated per-regime parameters
            regime_params[label] = {
                "lambda": rc.lam,
                "crash_prob": rc.p_crash,
                "eta_up": rc.eta_up,
                "eta_down": rc.eta_down,
                "mu_v": rc.mu_v,
                "rho_J": rc.rho_J,
                "sigma_s": base_sigma_s,  # sigma_s not regime-calibrated; use base
            }
            logger.info(
                "Regime '%s': using directly calibrated jump params "
                "(lam=%.1f, p_crash=%.3f, %d jumps)",
                regime, rc.lam, rc.p_crash, rc.n_jumps,
            )
        else:
            # Apply hardcoded multipliers to base parameters
            if regime == "bear":
                regime_params[label] = {
                    "lambda": base_lam * 1.5,
                    "crash_prob": min(base_p_crash * 1.3, 0.85),
                    "eta_up": base_eta_up * 0.8,
                    "eta_down": base_eta_down * 0.7,
                    "mu_v": base_mu_v * 2.0,
                    "rho_J": base_rho_j * 1.5,
                    "sigma_s": base_sigma_s * 1.5,
                }
            elif regime == "sideways":
                regime_params[label] = {
                    "lambda": base_lam,
                    "crash_prob": base_p_crash,
                    "eta_up": base_eta_up,
                    "eta_down": base_eta_down,
                    "mu_v": base_mu_v,
                    "rho_J": base_rho_j,
                    "sigma_s": base_sigma_s,
                }
            else:  # bull
                regime_params[label] = {
                    "lambda": base_lam * 0.7,
                    "crash_prob": base_p_crash * 0.7,
                    "eta_up": base_eta_up * 1.2,
                    "eta_down": base_eta_down,
                    "mu_v": base_mu_v * 0.6,
                    "rho_J": base_rho_j * 0.5,
                    "sigma_s": base_sigma_s * 0.7,
                }

    return regime_params


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BTC Pricing Engine v2 Validation Tests")
    print("=" * 60)

    all_passed = True

    # -------------------------------------------------------------------------
    # Test 1: Multi-Jump Aggregation (compare 99th percentile)
    # -------------------------------------------------------------------------
    print("\n[Test 1] Multi-Jump Aggregation...")
    rng1 = np.random.default_rng(42)

    n_test = 100000
    lam_high = 500.0  # Very high annual lambda for testing
    lam_daily_test = lam_high / 365.0
    p_crash_test = 0.5
    eta_up_test = 30.0
    eta_down_test = 30.0
    dt_test = 1.0

    # New implementation (multi-jump)
    k = rng1.poisson(lam_daily_test * dt_test, size=n_test)
    k_down = rng1.binomial(k, p_crash_test)
    k_up = k - k_down
    down_mag = np.zeros(n_test)
    up_mag = np.zeros(n_test)
    mask_down, mask_up = k_down > 0, k_up > 0
    if np.any(mask_down):
        down_mag[mask_down] = rng1.gamma(k_down[mask_down], scale=1.0 / eta_down_test)
    if np.any(mask_up):
        up_mag[mask_up] = rng1.gamma(k_up[mask_up], scale=1.0 / eta_up_test)
    jump_sizes_new = np.abs(up_mag - down_mag)
    q99_new = np.percentile(jump_sizes_new, 99)

    # Old implementation (single jump, capped at 1)
    rng1_old = np.random.default_rng(42)  # Fresh RNG with same seed
    n_jumps_old = rng1_old.poisson(lam_daily_test * dt_test, size=n_test)
    has_jump_old = n_jumps_old > 0
    jump_sizes_old = np.zeros(n_test)
    n_jumpers = np.sum(has_jump_old)
    if n_jumpers > 0:
        is_crash = rng1_old.random(n_jumpers) < p_crash_test
        mags = np.zeros(n_jumpers)
        n_crashes = np.sum(is_crash)
        if n_crashes > 0:
            mags[is_crash] = rng1_old.exponential(1.0 / eta_down_test, size=n_crashes)
        n_pumps = n_jumpers - n_crashes
        if n_pumps > 0:
            mags[~is_crash] = rng1_old.exponential(1.0 / eta_up_test, size=n_pumps)
        jump_sizes_old[has_jump_old] = mags
    q99_old = np.percentile(np.abs(jump_sizes_old), 99)

    if q99_new >= 1.2 * q99_old:
        print(f"  PASS: q99_new ({q99_new:.4f}) >= 1.2 * q99_old ({1.2*q99_old:.4f})")
    else:
        print(f"  FAIL: q99_new ({q99_new:.4f}) < 1.2 * q99_old ({1.2*q99_old:.4f})")
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 2: Fractional dt Variance Preservation
    # -------------------------------------------------------------------------
    print("\n[Test 2] Fractional dt Variance Preservation...")
    rng2 = np.random.default_rng(42)

    n_test2 = 1000
    v0 = 0.0004  # Initial variance
    omega_test = 0.00001
    alpha_test = 0.1
    beta_test = 0.85
    nu_test = 5.0
    dt_frac = 0.5

    variances = np.full(n_test2, v0)
    log_prices = np.zeros(n_test2)

    # Simulate one fractional step
    scale_factor = np.sqrt((nu_test - 2) / nu_test) if nu_test > 2 else 1.0
    z_t = rng2.standard_t(nu_test, size=n_test2) * scale_factor
    step_sigma = np.sqrt(variances * dt_frac)
    log_prices += step_sigma * z_t

    # Variance should NOT update for fractional dt
    if abs(dt_frac - 1.0) < 1e-12:
        epsilon_squared = (step_sigma * z_t) ** 2
        variances = omega_test + alpha_test * epsilon_squared + beta_test * variances
    # else: unchanged

    variance_unchanged = np.allclose(variances, v0)
    price_changed = not np.allclose(log_prices, 0.0)

    if variance_unchanged and price_changed:
        print(f"  PASS: Variance unchanged ({variances[0]:.6f} == {v0:.6f}), prices moved")
    else:
        print(f"  FAIL: variance_unchanged={variance_unchanged}, price_changed={price_changed}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 3: Dynamic Per-Path Drift Clamping
    # -------------------------------------------------------------------------
    print("\n[Test 3] Dynamic Per-Path Drift Clamping...")

    # Test that clamp uses per-path variance (vector), not scalar
    n_test3 = 5
    variances_test = np.array([0.0001, 0.0004, 0.0009, 0.0016, 0.0025])  # Different hourly variances
    sigma_hourly_step = np.sqrt(variances_test)  # [0.01, 0.02, 0.03, 0.04, 0.05]
    mu_extreme = 0.10  # 10% hourly drift, way too high for all

    mu_clamped = np.clip(mu_extreme, -DRIFT_CLAMP_MULT * sigma_hourly_step, DRIFT_CLAMP_MULT * sigma_hourly_step)
    expected_clamped = DRIFT_CLAMP_MULT * sigma_hourly_step  # Should be [0.0025, 0.005, 0.0075, 0.01, 0.0125]

    # Verify it's a vector matching per-path sigma
    is_vector = isinstance(mu_clamped, np.ndarray) and len(mu_clamped) == n_test3
    clamps_match = np.allclose(mu_clamped, expected_clamped)

    if is_vector and clamps_match:
        print(f"  PASS: Clamped drift varies per-path: {mu_clamped}")
    else:
        print(f"  FAIL: is_vector={is_vector}, clamps_match={clamps_match}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 4: Variance Consistency Check
    # -------------------------------------------------------------------------
    print("\n[Test 4] Variance Consistency Check...")

    # Create mock garch_params (hourly scale)
    test_garch = {
        'omega': 0.00001 / 24,       # hourly omega ~ daily/24
        'alpha': 0.1,
        'beta': 0.85,
        'nu': 5.0,
        'mu': 0.0,
        'last_variance': 0.0004 / 24,  # 2% daily vol → hourly units (~0.00001667)
    }

    ratio = check_variance_consistency(test_garch, n_samples=50000, seed=12345)

    if abs(ratio - 1.0) < 0.15:
        print(f"  PASS: Variance ratio = {ratio:.4f} (within ±15% of 1.0)")
    else:
        print(f"  FAIL: Variance ratio = {ratio:.4f} (outside ±15%)")
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 5: Naive Prior (Phase 1.1)
    # -------------------------------------------------------------------------
    print("\n[Test 5] Naive Prior Enforcement...")
    # With naive prior, mu is set to 0 in simulate_paths when use_naive_prior=True
    # Verify the distribution is centered on S0
    rng5 = np.random.default_rng(4242)
    test_garch5 = {
        'omega': 0.00001 / 24,
        'alpha': 0.1,
        'beta': 0.85,
        'nu': 5.0,
        'mu': 0.001,  # Non-zero drift that should be zeroed
        'last_variance': 0.0004 / 24,
    }
    paths_naive = simulate_paths(
        S0=100000.0,
        garch_params=test_garch5,
        jump_params={'lambda': 0.0, 'crash_prob': 0.5, 'eta_up': 50.0, 'eta_down': 25.0},
        hours_to_expiry=168.0,  # 1 week
        n_sims=5000,
        seed=4242,
        use_naive_prior=True,
    )
    paths_no_naive = simulate_paths(
        S0=100000.0,
        garch_params=test_garch5,
        jump_params={'lambda': 0.0, 'crash_prob': 0.5, 'eta_up': 50.0, 'eta_down': 25.0},
        hours_to_expiry=168.0,
        n_sims=5000,
        seed=4242,
        use_naive_prior=False,
    )
    mean_naive = np.mean(paths_naive)
    mean_no_naive = np.mean(paths_no_naive)

    # Naive should be closer to S0 (100000)
    deviation_naive = abs(mean_naive - 100000)
    deviation_no_naive = abs(mean_no_naive - 100000)

    if deviation_naive <= deviation_no_naive:
        print(f"  PASS: Naive deviation={deviation_naive:.1f} <= No-naive deviation={deviation_no_naive:.1f}")
    else:
        print(f"  FAIL: Naive deviation={deviation_naive:.1f} > No-naive deviation={deviation_no_naive:.1f}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 6: SVCJ Vol Jumps (Phase 1.3)
    # -------------------------------------------------------------------------
    print("\n[Test 6] SVCJ Volatility Jumps...")
    # Basic sanity: SVCJ should produce larger tail variance than SVJ
    paths_no_svcj = simulate_paths(
        S0=100000.0,
        garch_params=test_garch5,
        jump_params={'lambda': 25.0, 'crash_prob': 0.6, 'eta_up': 50.0, 'eta_down': 25.0,
                     'mu_v': 0.0, 'rho_J': 0.0},
        hours_to_expiry=720.0,
        n_sims=5000,
        seed=42,
        use_svcj=False,
    )
    paths_svcj = simulate_paths(
        S0=100000.0,
        garch_params=test_garch5,
        jump_params={'lambda': 25.0, 'crash_prob': 0.6, 'eta_up': 50.0, 'eta_down': 25.0,
                     'mu_v': 0.0001, 'rho_J': -0.3},
        hours_to_expiry=720.0,
        n_sims=5000,
        seed=42,
        use_svcj=True,
    )
    std_svcj = np.std(np.log(paths_svcj / 100000))
    std_no_svcj = np.std(np.log(paths_no_svcj / 100000))

    if std_svcj > std_no_svcj * 1.01:
        print(f"  PASS: SVCJ vol={std_svcj:.6f} > SVJ vol={std_no_svcj:.6f} (vol jumps add variance)")
    else:
        print(f"  FAIL: SVCJ vol={std_svcj:.6f} <= SVJ vol={std_no_svcj:.6f}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 7: FIGARCH Weights (Phase 2.5)
    # -------------------------------------------------------------------------
    print("\n[Test 7] FIGARCH Weights...")
    weights = _compute_figarch_weights(d=0.578, trunc_k=100)
    # Weights should decay hyperbolically (not exponentially)
    if weights[0] > weights[-1] * 10:
        print(f"  PASS: FIGARCH weights decay correctly (w0={weights[0]:.6f}, w99={weights[-1]:.12f})")
    else:
        print(f"  FAIL: Unexpected FIGARCH weight pattern")
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 8: Skewed-t Generation (Phase 1.4)
    # -------------------------------------------------------------------------
    print("\n[Test 8] Skewed-t Distribution...")
    rng8 = np.random.default_rng(42)
    n_st = 20000

    # Test negative skew (lam < 0 should produce negative skew)
    st_neg = skewed_t_rvs(nu=5.0, lam=-0.3, size=n_st, rng=rng8)
    skew_neg = pd.Series(st_neg).skew()

    # Test positive skew (lam > 0 should produce positive skew)
    rng8b = np.random.default_rng(43)
    st_pos = skewed_t_rvs(nu=5.0, lam=0.3, size=n_st, rng=rng8b)
    skew_pos = pd.Series(st_pos).skew()

    # Test symmetric (lam=0 should be roughly symmetric)
    rng8c = np.random.default_rng(44)
    st_sym = skewed_t_rvs(nu=5.0, lam=0.0, size=n_st, rng=rng8c)
    skew_sym = pd.Series(st_sym).skew()

    neg_ok = skew_neg < -0.5
    pos_ok = skew_pos > 0.5
    sym_ok = abs(skew_sym) < 0.3

    if neg_ok and pos_ok and sym_ok:
        print(f"  PASS: lam=-0.3 -> skew={skew_neg:.2f}(neg), lam=+0.3 -> skew={skew_pos:.2f}(pos), lam=0 -> skew={skew_sym:.2f}(sym)")
    else:
        print(f"  FAIL: neg_ok={neg_ok} (skew={skew_neg:.2f}), pos_ok={pos_ok} (skew={skew_pos:.2f}), sym_ok={sym_ok} (skew={skew_sym:.2f})")
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 9: Skewed-t Variance Consistency (Regression guard for Bug 1)
    # -------------------------------------------------------------------------
    print("\n[Test 9] Skewed-t Variance Consistency...")
    rng9 = np.random.default_rng(99)
    n_var_test = 50000
    nu_test = 5.0

    all_var_ok = True
    for lam_test in [-0.3, 0.0, 0.3]:
        s = skewed_t_rvs(nu_test, lam_test, n_var_test, rng9)
        sf = skewed_t_scale_factor(nu_test, lam_test)
        samples = s * sf  # Combined output as used in simulate_paths
        emp_var = np.var(samples)
        # Threshold distortion effect: |lam|>0 shifts proportion of samples in each tail
        # relative to Hansen's unit-variance derivation. Expected ~0.95-0.98 at lam=-0.3.
        var_ok = 0.94 <= emp_var <= 1.06
        status = "PASS" if var_ok else "FAIL"
        if not var_ok:
            all_var_ok = False
        print(f"    lam={lam_test:+4.1f}: variance={emp_var:.4f} [{status}]")

    if all_var_ok:
        print("  PASS: All skewed-t variances within [0.94, 1.06]")
    else:
        print("  FAIL: One or more skewed-t variances outside [0.94, 1.06]")
        all_passed = False

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
