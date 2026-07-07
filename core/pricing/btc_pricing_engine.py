"""
BTC Pricing Engine v2 — Regime-Switching SVCJ with Long Memory

FIGARCH(1,d,1) + Skewed-t + SVCJ (Kou Double Exponential with correlated
volatility jumps) Monte Carlo simulator. Regime-conditional via 3-state HMM.
Hourly simulation steps.

Enhancements over v1 (per 17-paper meta-analysis, June 2026):
  Phase 1.1 — Naive prior (μ=0 anchoring) [Baquero 2026, Shelton 2024]
  Phase 1.2 — 3-state HMM regime detection [Oprea & Bâra 2026, Malekinezhad 2026]
  Phase 1.3 — SVCJ correlated volatility jumps [Teng et al. 2025, Eraker et al. 2004]
  Phase 1.4 — Skewed-t innovations (Hansen 1994) [Nakakita et al. 2025]
  Phase 1.5 — Horizon-gating (naive prior for T>30d) [Baquero 2026]
  Phase 2.4 — Regime-conditional jump parameters
  Phase 2.5 — FIGARCH(1,d,1) [Baillie, Bollerslev & Mikkelsen 1996]
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
from typing import Optional

import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t as student_t
from scipy.special import gamma as gamma_func

logger = logging.getLogger(__name__)

# One-shot flag so the FIGARCH fallback (FIGARCH fit → GARCH) is reported
# once per process instead of on every pricing call.
_FIGARCH_FALLBACK_WARNED = False

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
SVCJ_RHO_J = -0.08       # Return-vol jump correlation (Teng 2025 estimate; REPORTING
                         # ONLY as of FIX 4/M1 -- see SVCJ_RHO_J_SLOPE below)
# FIX 4 (M1): SVCJ_RHO_J above is a Pearson CORRELATION (dimensionless), but the
# Eraker (2004) return-jump equation needs a regression SLOPE (return per unit
# variance jump): xi_s | xi_v ~ N(rho_J_slope * xi_v, sigma_s^2). Using the
# correlation directly as a slope made the term ~5 orders of magnitude too small
# to matter (vol_jump_mag is in hourly-variance units, ~1e-5..1e-3; a correlation
# of ~-0.08 times that is ~-1e-6, far below a typical jump size of ~0.02-0.04).
# Default 0.0 = term off (the old effective behavior, now explicit). Nonzero
# values come from calibration (JumpCalibrationResult.rho_j_slope /
# RegimeJumpResult.rho_j_slope, see jump_calibration._estimate_vol_jump_params).
SVCJ_RHO_J_SLOPE = 0.0
SVCJ_LAM_V = None        # If None, uses same lambda as return jumps
SVCJ_SIGMA_S = 0.01      # Conditional std dev of return jump given vol jump (Eraker 2004)
# FIX 5 (H3): persistence (per-hour decay) of the SVCJ variance-jump state on the
# FIGARCH path. The ARCH(∞) base recomputes variance from the ε² buffer each step
# and has no β to carry a one-shot vol-jump add, so without a separate decaying
# state SVCJ degenerates to a plain return jump under FIGARCH. Mean-reversion analog
# of Eraker's affine variance jump. Must be strictly in (0,1).
SVCJ_PERSIST = 0.90      # Hourly decay of the vol-jump state (FIGARCH path)
# Hard cap on the accumulated vol-jump state. With SVCJ_PERSIST=0.90 over a 720-step
# (30-day hourly) horizon the geometric sum of repeated jumps can compound without
# bound; the cap (≈ a few × unconditional hourly variance) keeps tails finite.
VOL_JUMP_STATE_CAP = 1e-3

# ==============================================================================
# FIGARCH PARAMETERS
# FIGARCH(1,d,1) per Baillie, Bollerslev & Mikkelsen (1996):
#   σ²_t = ω/(1-β) + [1 - (1-βL)⁻¹(1-φL)(1-L)^d] ε²_t
# ARCH(∞) form (Chung 1999):
#   σ²_t = ω/(1-β) + Σ λ_k · ε²_{t-k}
# where λ_k weights are computed via the δ_i and λ_i recurrences matching
# the `arch` library's figarch_weights_python.
#
# φ, d, β are estimated jointly via arch_model(vol='FIGARCH') in fit_garch_model.
# FIGARCH_D is a reference constant (Siu 2025) used only in testing; live fits
# estimate d from hourly BTC data.
# ==============================================================================
FIGARCH_D = 0.578         # Reference long-memory estimate (Siu 2025, SE=0.271)
FIGARCH_TRUNC_K = 1000    # Truncation lag for ARCH(∞) approximation

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
# XGBOOST DIRECTIONAL DRIFT-SHIFT (FIX 3 / H2 re-enable — drift-shift design)
# The XGBoost P(up) is converted into a SINGLE strike-agnostic shift of the
# simulated terminal distribution (NOT a per-strike additive blend, which broke
# ladder monotonicity — the reason the old design was disabled). Applied once
# per expiry group, every strike re-derived from the shifted paths, so the
# ladder stays monotone by construction. See temp/xgb_activation_plan.md §2.
# ==============================================================================
XGB_TILT_LAMBDA = 0.0       # Tilt strength toward p_up. DEFAULT 0.0 = inert even
                            # if the flag is flipped outside the --use-xgb plumbing.
                            # Production value set by calibration (plan §8 grid).
XGB_P_FLOOR = 0.15          # Clip raw p_up below before mapping
XGB_P_CEIL = 0.85           # Clip raw p_up above before mapping
XGB_P_TARGET_FLOOR = 0.02   # Clip target P(up) below
XGB_P_TARGET_CEIL = 0.98    # Clip target P(up) above
XGB_MAX_SHIFT_FRAC = 0.5    # Cap |Δ_H| at this fraction of empirical sigma_H
XGB_SIGMA_H_FLOOR = 1e-6    # sigma_H below this → skip shift (numeric guard)
XGB_P_BASE_GUARD = 0.02     # base P(up) within this of 0 or 1 → skip shift
                            # (deep-skew snapshot; shift would be noise, not signal)
# DTE buckets for per-horizon XGB models (C2-a). Right-open intervals in DAYS;
# a contract at exactly an edge falls into the higher bucket. >30d is gated off.
# Train horizon = bucket midpoint (4 / 11 / 22d) so the forward-shifted training
# target matches the contracts each model serves.
XGB_DTE_BUCKETS = [(0.0, 7.0), (7.0, 14.0), (14.0, 30.0)]

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
    disable_staleness_check: bool = False,
):
    """
    Loads hourly data for GARCH fitting and intraday data for the latest price mark.
    Supports dependency injection for backtesting.

    Phase 0.1: training_start_date filters data to post-break period (Pakstaite 2025).

    disable_staleness_check: when True, skip the intraday staleness check against
    current wall-clock time. Set True during time-travel backtesting where the
    last data row is always in the past by design.
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
    # M5 fallback: keep pre-filter copy in case we need to restore all data
    hourly_df_prefilter = hourly_df.copy()
    if date_col and training_start_date is not None:
        hourly_df[date_col] = pd.to_datetime(hourly_df[date_col], utc=True, errors='coerce')
        start_dt = pd.Timestamp(training_start_date, tz='UTC')
        hourly_df = hourly_df[hourly_df[date_col] >= start_dt]
        if len(hourly_df) < 500:
            logger.warning(
                f"Only {len(hourly_df)} rows after training_start_date={training_start_date}. "
                "Falling back to all data."
            )
            # Restore from pre-filter copy and re-log
            hourly_df = hourly_df_prefilter.copy()
            logger.info(f"Using all {len(hourly_df)} available rows for GARCH fitting.")

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
    # Suppressed during time-travel backtesting where the last data row is
    # always in the past by design.
    if not disable_staleness_check:
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
        Array of standardized skewed-t variates with mean 0 and variance 1
        (by Hansen's construction — a and b are chosen precisely for this).

    Reference:
        Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation."
        International Economic Review, 35(3), 705-730.

        This implements Hansen's standardized skew-t via the inverse-CDF method.
        With constants
            c = Γ((ν+1)/2) / (√(π(ν-2)) Γ(ν/2)),  a = 4λc(ν-2)/(ν-1),
            b = √(1 + 3λ² - a²),
        the density has a left piece (mass (1-λ)/2) and a right piece (mass
        (1+λ)/2). Sampling: draw U~Uniform(0,1); within each piece invert the
        standardized-t quantile T*(p) = t.ppf(p,ν)·√((ν-2)/ν), then map:
            U < (1-λ)/2:  z = ((1-λ)/b)·T*(U/(1-λ)) - a/b
            else:         z = ((1+λ)/b)·T*(0.5 + (U-(1-λ)/2)/(1+λ)) - a/b
        Continuous at U=(1-λ)/2 (both give z=-a/b). λ<0 ⇒ heavier left tail ⇒
        negative skew. Mean 0, variance 1 by construction — no external rescale.
    """
    if nu <= 2:
        # Fall back to standard t if df too low (cannot standardize variance)
        scale = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
        return rng.standard_t(nu, size=size) * scale

    # Hansen's constants (eq 8-10 in Hansen 1994)
    c_const = gamma_func((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * gamma_func(nu / 2))
    a = 4 * lam * c_const * (nu - 2) / (nu - 1)
    b_sq = 1 + 3 * lam ** 2 - a ** 2

    if b_sq <= 0:
        # Numerical fallback (extreme lam)
        return rng.standard_t(nu, size=size) * np.sqrt((nu - 2) / nu)

    b = np.sqrt(b_sq)
    t_std_scale = np.sqrt((nu - 2) / nu)  # standardized-t (unit variance) scale

    u = rng.uniform(size=size)
    left = u < (1 - lam) / 2

    # Inner standardized-t quantiles per piece
    p_inner = np.empty(size)
    p_inner[left] = u[left] / (1 - lam)
    p_inner[~left] = 0.5 + (u[~left] - (1 - lam) / 2) / (1 + lam)
    t_q = student_t.ppf(p_inner, nu) * t_std_scale

    g = np.empty(size)
    g[left] = ((1 - lam) * t_q[left] - a) / b
    g[~left] = ((1 + lam) * t_q[~left] - a) / b

    return g


def skewed_t_scale_factor(nu: float, lam: float) -> float:
    """
    Scale factor for skewed_t_rvs() output. Returns 1.0 for nu>2 because
    skewed_t_rvs now returns Hansen's STANDARDIZED variate (mean 0, variance 1
    by construction). Retained for backward-compatible call sites that still
    multiply by this factor. Returns 1.0 for the nu<=2 standard-t fallback too.
    """
    return 1.0


# ==============================================================================
# FIGARCH(1,d,1) INFINITE-ARCH WEIGHTS — Phase 2.5
# ==============================================================================

def _compute_figarch_weights(d: float, phi: float, beta: float, trunc_k: int = FIGARCH_TRUNC_K) -> np.ndarray:
    """
    Precompute FIGARCH(1,d,1) infinite-ARCH weights for the variance recursion.

    The standard FIGARCH(1,d,1) specification (Baillie, Bollerslev & Mikkelsen 1996):
        σ²_t = ω/(1-β) + [1 - (1-βL)⁻¹(1-φL)(1-L)^d] ε²_t

    Chung (1999) gives the ARCH(∞) representation:
        σ²_t = ω/(1-β) + Σ λ_k · ε²_{t-k}

    where the λ_k weights follow the recurrence (matching the `arch` library):
        δ₁ = d
        λ₁ = φ - β + d
        for i ≥ 2:
            δ_i = ((i-1-d) / i) · δ_{i-1}
            λ_i = β · λ_{i-1} + (δ_i - φ · δ_{i-1})

    The returned array is aligned with the simulation's past_eps_sq buffer:
        weights[0] = 0           (no contemporaneous ε² term)
        weights[k] = λ_k for k≥1  (weight on ε²_{t-k})

    Args:
        d: Fractional differencing parameter (0 < d < 1).
        phi: Short-run ARCH parameter.
        beta: GARCH persistence parameter (also in intercept ω/(1-β)).
        trunc_k: Number of lags in truncation (default 1000).

    Returns:
        Array of length trunc_k: weights[0]=0, weights[k]=λ_k for k=1..trunc_k-1.
    """
    if d <= 0 or d >= 1:
        raise ValueError(f"FIGARCH d must be in (0, 1), got {d}")

    # Compute λ_k for k = 1..trunc_k-1 (need trunc_k-1 weights
    # since index 0 is reserved for the 0-valued contemporaneous slot)
    n_lags = trunc_k - 1
    delta = np.empty(n_lags)
    lam = np.empty(n_lags)

    # Initialization: δ₁ = d,  λ₁ = φ - β + d
    delta[0] = d
    lam[0] = phi - beta + d

    # Recurrence for i ≥ 2 (matching arch library figarch_weights_python)
    for i in range(1, n_lags):
        delta[i] = (i - d) / (i + 1) * delta[i - 1]
        lam[i] = beta * lam[i - 1] + (delta[i] - phi * delta[i - 1])

    # Build output: weights[0] = 0, weights[1:] = lam
    weights = np.zeros(trunc_k)
    weights[1:] = lam
    return weights


# ==============================================================================
# JUMP-FILTERED RETURNS (FIX 1 / H1)
# ==============================================================================

def filter_jump_returns(returns: pd.Series, clip_mult: float = 3.0) -> pd.Series:
    """
    Winsorize detected jump-bar returns to +/- clip_mult * local bipower sigma so
    a subsequent GARCH/FIGARCH fit sees (approximately) the diffusion component
    only.

    FIX 1 (H1): `fit_garch_model` previously fit on raw hourly returns, including
    jumps, while `simulate_paths` separately adds a calibrated compound-Poisson
    jump process on top -- double-counting jump variance in the simulated total
    (see PRICING_REVIEW.md H1, ~10% vol overstatement at typical calibration).
    Winsorizing (not dropping) jump bars keeps the series length and index intact
    for `arch_model` while removing most of the jump's contribution to the
    likelihood; the tail residual left after clipping to +/-3 local sigma is a
    minor, symmetric approximation error, not a second jump channel.

    Detection: `detect_jumps_bipower(returns, return_sigma=True)` (Lee-Mykland),
    the SAME detector used for jump calibration, so the GARCH fit and the jump
    calibration agree on which bars are "jumps". Non-jump bars are unchanged.

    Args:
        returns: pd.Series of log returns (hourly, in this codebase).
        clip_mult: Winsorize half-width in local-sigma units. Default 3.0.

    Returns:
        A new pd.Series aligned to the input index. On detection failure, an
        all-False mask, or no bars with a valid local sigma, returns a copy of
        the input unchanged (logged at debug level) -- this also guarantees the
        zero-jumps case is bit-identical to the unfiltered series.
    """
    from core.pricing.jump_calibration import detect_jumps_bipower

    arr = returns.to_numpy()
    try:
        jump_mask, sigma_local = detect_jumps_bipower(arr, return_sigma=True)
    except Exception as e:
        logger.debug("filter_jump_returns: detection failed (%s); returning input unchanged", e)
        return returns.copy()

    valid_mask = jump_mask & np.isfinite(sigma_local) & (sigma_local > 0)
    if not np.any(valid_mask):
        logger.debug("filter_jump_returns: no jumps detected; returning input unchanged")
        return returns.copy()

    filtered = arr.copy()
    lo = -clip_mult * sigma_local
    hi = clip_mult * sigma_local
    filtered[valid_mask] = np.clip(arr[valid_mask], lo[valid_mask], hi[valid_mask])

    return pd.Series(filtered, index=returns.index, name=returns.name)


# ==============================================================================
# MODEL FITTING
# ==============================================================================

def fit_garch_model(
    returns: pd.Series,
    training_start_date: str = "2019-10-01",
    use_figarch: bool = False,
    figarch_d: float = FIGARCH_D,
    figarch_trunc_k: int = FIGARCH_TRUNC_K,
    filter_jumps: bool = True,
):
    """
    Fit GARCH(1,1) or FIGARCH(1,d,1) with Student-t errors via the `arch` library.

    Phase 0.1: training_start_date filters to post-2019 structural break.
    Phase 2.5: When use_figarch=True, fits FIGARCH(1,d,1) directly via
      arch_model(vol='FIGARCH', p=1, q=1). Jointly estimates φ, d, β, ω, ν, μ.
      Falls back to GARCH(1,1) on convergence failure.

    Uses the long-term fitted mean (structural mu) as drift.
    All returned parameters are in hourly log-return units.

    Args:
        returns: pd.Series of hourly log returns.
        training_start_date: Ignored here (filtered upstream in load_and_prep_data).
        use_figarch: If True, fit FIGARCH(1,d,1) instead of GARCH(1,1).
        figarch_d: Unused when live-fitting FIGARCH (d comes from fit).
            Retained for backward compat; FIGARCH_D constant used in tests only.
        figarch_trunc_k: Unused when live-fitting FIGARCH (FIGARCH_TRUNC_K used).
        filter_jumps: If True (default), winsorize detected jump bars via
            `filter_jump_returns` BEFORE the `* 100` scaling, for both the
            FIGARCH and GARCH branches (FIX 1/H1 -- this default IS the fix, so
            `last_variance` comes from the jump-filtered fit, which is what the
            simulator needs since it adds jumps back on top separately). False
            preserves the legacy raw-return fit for A/B comparison.

    Returns:
        For FIGARCH: Dict with omega, beta, nu, mu, last_variance, use_figarch,
            figarch_weights, figarch_d, figarch_phi, figarch_trunc_k.
            (No 'alpha' key — consumers should use .get('alpha', 0.0).)
        For GARCH: Dict with omega, alpha, beta, nu, mu, last_variance.
    """
    # FIX 1 (H1): filter jump bars before fitting so BOTH branches below (FIGARCH
    # and GARCH) see the jump-filtered series -- this reassigns `returns` once,
    # upstream of `scaled_returns`, rather than duplicating the filter per branch.
    if filter_jumps:
        filtered_returns = filter_jump_returns(returns)
        n_changed = int(np.sum(~np.isclose(
            filtered_returns.to_numpy(), returns.to_numpy(), equal_nan=True
        )))
        if len(returns) > 0 and n_changed > 0.05 * len(returns):
            logger.warning(
                "filter_jump_returns changed %d/%d observations (%.1f%%) -- "
                "unexpectedly high for a 1%% bipower jump test; proceeding anyway.",
                n_changed, len(returns), 100.0 * n_changed / len(returns),
            )
        returns = filtered_returns

    # 1. Scale returns for numerical stability
    scaled_returns = returns * 100

    if use_figarch:
        # ---- FIGARCH(1,d,1) fit (Phase 2.5) ----
        # Fit FIGARCH jointly via arch library (Baillie-Bollerslev-Mikkelsen 1996).
        # Joint estimation of φ, d, β satisfies B-M positivity internally.
        try:
            model = arch_model(scaled_returns, vol='FIGARCH', p=1, q=1, dist='t', mean='Constant')
            res = model.fit(disp='off', show_warning=False)

            if res.convergence_flag != 0:
                status_code = res.optimization_result.get('status', 'unknown')
                status_msg = res.optimization_result.get('message', 'unknown')
                logger.warning(
                    "FIGARCH fit may not have converged (code=%s): %s. "
                    "Retrying with relaxed tolerance.",
                    status_code, status_msg,
                )
                try:
                    res = model.fit(disp='off', show_warning=False, options={'ftol': 1e-6, 'maxiter': 500})
                except Exception:
                    logger.warning("FIGARCH fallback fit failed; falling back to GARCH")
                    use_figarch = False
                if use_figarch and res.convergence_flag != 0:
                    logger.warning(
                        "FIGARCH fallback fit also not converged (code=%s). "
                        "Falling back to GARCH.",
                        res.optimization_result.get('status', 'unknown'),
                    )
                    use_figarch = False
            if use_figarch:
                params = res.params
                omega = params['omega'] / 10000.0
                phi_val = params['phi']
                d_val = params['d']
                beta_val = params['beta']
                nu = params['nu']
                mu = params['mu'] / 100.0

                # Compute ARCH(∞) weights
                weights = _compute_figarch_weights(d_val, phi_val, beta_val, FIGARCH_TRUNC_K)

                # FIX 11 (L6): Baillie-Bollerslev-Mikkelsen positivity check.
                # For pathological (φ,d,β) the ARCH(∞) weights can go negative,
                # which would let the variance recursion produce negative variance
                # (silently floored to 1e-10 downstream — i.e. a dead vol process).
                # Detect it here and fall back to GARCH rather than ship a broken
                # long-memory recursion. weights[0] is the (intentionally 0)
                # contemporaneous slot, so only check weights[1:]. Raising routes to
                # the outer `except` below, which sets use_figarch=False → GARCH fit.
                if np.any(weights[1:] < -1e-10):
                    n_neg = int(np.sum(weights[1:] < -1e-10))
                    raise ValueError(
                        f"FIGARCH weights violate B-M positivity ({n_neg} negative "
                        f"lags, min={float(weights[1:].min()):.3e}) for d={d_val:.4f}, "
                        f"phi={phi_val:.4f}, beta={beta_val:.4f}"
                    )

                result = {
                    'omega': omega,
                    'beta': beta_val,
                    'nu': nu,
                    'mu': mu,
                    'last_variance': res.conditional_volatility.iloc[-1]**2 / 10000.0,
                    'use_figarch': True,
                    'figarch_weights': weights,
                    'figarch_d': d_val,
                    'figarch_phi': phi_val,
                    'figarch_trunc_k': FIGARCH_TRUNC_K,
                }
                logger.info(
                    "FIGARCH(1,d,1) fitted: d=%.4f, phi=%.4f, beta=%.4f, "
                    "lambda_1=%.4f (B-M positivity satisfied)",
                    d_val, phi_val, beta_val, weights[1],
                )
                return result
        except Exception as e:
            logger.warning("FIGARCH fit failed (%s), falling back to GARCH", e)
            use_figarch = False
        # Fall through to GARCH if FIGARCH failed

    # ---- Standard GARCH(1,1) fit (legacy / FIGARCH fallback) ----
    # Suppress arch's raw ConvergenceWarning (scipy SLSQP code 8 is intermittent
    # and the returned params are usually still usable). We check convergence
    # ourselves and log a clean warning with context.
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='t', mean='Constant')
    res = model.fit(disp='off', show_warning=False)

    if res.convergence_flag != 0:
        status_code = res.optimization_result.get('status', 'unknown')
        status_msg = res.optimization_result.get('message', 'unknown')
        logger.warning(
            "GARCH fit may not have converged (code=%s): %s. "
            "Retrying with relaxed tolerance.",
            status_code, status_msg,
        )
        # Fallback: relaxed tolerance can help when SLSQP linesearch stalls
        # near a flat gradient region (common with heavy-tailed data + Student-t).
        try:
            res = model.fit(disp='off', show_warning=False, options={'ftol': 1e-6, 'maxiter': 500})
        except Exception:
            logger.warning("GARCH fallback fit failed; continuing with original parameters")
        if res.convergence_flag != 0:
            logger.warning(
                "GARCH fallback fit also not converged (code=%s). "
                "Continuing with best available parameters.",
                res.optimization_result.get('status', 'unknown'),
            )

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

    omega = garch_params.get('omega', 0.0)
    alpha = garch_params.get('alpha', 0.0)
    beta_val = garch_params.get('beta', 0.0)
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
    martingale_anchor: bool = False,
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

    MEASURE NOTE (FIX 9 / M3): by default this simulates under the PHYSICAL measure
    and is log-mean anchored (E[log S_T] = log S0); the median coincides only for
    a symmetric log distribution. use_naive_prior sets μ=0 with only the
    jump-drift compensator subtracted; there is NO diffusion convexity / Jensen
    correction. It is therefore NOT a risk-neutral distribution and E[S_T] ≠ S0
    in general. The risk-neutral switch is `martingale_anchor=True`, which
    corrects the JUMP compensator only; the diffusion Jensen term (~ +sigma^2/2
    per step, roughly +1% at 30 days at 50% annualized vol) is NOT subtracted,
    and Student-t exponential moments are finite only due to the per-step return
    clip. Downstream `p_market_fit` (formerly p_rn_fit) is a logistic fit to
    MARKET prices, not a risk-neutral model probability.

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
            Also accepts mu_v, rho_J (reporting only, FIX 4/M1), rho_j_slope
            (the SVCJ return-vol regression slope actually used in the return
            equation), sigma_s, svcj_persist.
        hours_to_expiry: Float, number of hours until expiry.
        n_sims: Number of Monte Carlo paths.
        seed: Random seed for reproducibility.
        apply_jump_drift_correction: If True, subtract expected_jump_drift from mu.
        martingale_anchor: If True, use the exponential cumulant compensator
            lam*(E[e^J]-1) (true risk-neutral martingale, E[S_T]=S0). If False
            (default), use the legacy log-mean compensator lam*E[J]. Default False
            preserves historical probabilities; calibration was established under it.
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
        svcj_rho_j_slope = SVCJ_RHO_J_SLOPE
        svcj_sigma_s = SVCJ_SIGMA_S
        svcj_persist = SVCJ_PERSIST
    else:
        lam = jump_params.get('lambda', LAMBDA)
        p_crash = jump_params.get('crash_prob', CRASH_PROB)
        eta_up = jump_params.get('eta_up', ETA_UP)
        eta_down = jump_params.get('eta_down', ETA_DOWN)
        svcj_mu_v = jump_params.get('mu_v', SVCJ_MU_V)
        svcj_rho_j = jump_params.get('rho_J', SVCJ_RHO_J)
        svcj_rho_j_slope = jump_params.get('rho_j_slope', SVCJ_RHO_J_SLOPE)
        svcj_sigma_s = jump_params.get('sigma_s', SVCJ_SIGMA_S)
        svcj_persist = jump_params.get('svcj_persist', SVCJ_PERSIST)

    # --- Regime-Conditional Jump Overrides (Phase 2.4) ---
    if regime_jump_params and regime_label in regime_jump_params:
        rp = regime_jump_params[regime_label]
        lam = rp.get('lambda', lam)
        p_crash = rp.get('crash_prob', p_crash)
        eta_up = rp.get('eta_up', eta_up)
        eta_down = rp.get('eta_down', eta_down)
        svcj_mu_v = rp.get('mu_v', svcj_mu_v)
        svcj_rho_j = rp.get('rho_J', svcj_rho_j)
        svcj_rho_j_slope = rp.get('rho_j_slope', svcj_rho_j_slope)
        svcj_sigma_s = rp.get('sigma_s', svcj_sigma_s)
        svcj_persist = rp.get('svcj_persist', svcj_persist)
        logger.debug(f"Regime-conditional jumps ({regime_label}): lam={lam:.1f}, p_crash={p_crash:.2f}")

    # --- Vol Gate Interaction (Phase 2.6) ---
    # In extreme vol: scale up jump intensity (vol jumps already embedded via SVCJ)
    if vol_gate_regime == "extreme":
        lam *= 1.5  # 50% more jumps in extreme vol
        svcj_mu_v *= 2.0  # Double vol jump size
    elif vol_gate_regime == "high":
        lam *= 1.2
        svcj_mu_v *= 1.3

    # FIX 5 (H3): the vol-jump state is a geometric accumulator; persistence must be
    # strictly in (0,1) or it either never decays (≥1 → unbounded) or is inert (≤0).
    svcj_persist = float(np.clip(svcj_persist, 1e-6, 1.0 - 1e-6))

    # 1. Convert Annual Lambda to Hourly
    lam_hourly = lam / HOURS_PER_YEAR
    # SVCJ uses the SAME Poisson count (k) as return jumps for contemporaneous
    # return/variance jumps (Eraker 2004). The legacy independent lam_v driver
    # is deprecated — a separate lam_v_hourly would break the SVCJ correlation.

    # 2. Calculate Expected Jump Drift (hourly log-return)
    if martingale_anchor:
        # Exponential cumulant: true risk-neutral compensator lam*(E[e^J]-1) so
        # that E[S_T]=S0. For Kou double-exponential jumps:
        #   E[e^J] = (1-p)*eta_up/(eta_up-1) + p*eta_down/(eta_down+1)
        # Requires eta_up>1 (else E[e^J] diverges); fall back to log-mean if not.
        if eta_up > 1:
            expected_jump_drift = lam_hourly * (
                (1 - p_crash) * eta_up / (eta_up - 1)
                + p_crash * eta_down / (eta_down + 1)
                - 1.0
            )
        else:
            logger.warning(
                "martingale_anchor=True but eta_up=%.3f<=1; E[e^J] diverges. "
                "Falling back to log-mean compensator.", eta_up,
            )
            expected_jump_drift = lam_hourly * ((1 - p_crash) / eta_up - p_crash / eta_down)
    else:
        # Legacy log-mean compensator: E[J] = (1-p)/eta_up - p/eta_down.
        # NOT a true martingale correction (drops the convexity/Jensen term),
        # but preserves historical default probabilities and calibration.
        expected_jump_drift = lam_hourly * ((1 - p_crash) / eta_up - p_crash / eta_down)

    n_hours = int(np.ceil(hours_to_expiry))
    dt_schedule = np.ones(n_hours)
    if hours_to_expiry % 1 != 0:
        dt_schedule[-1] = hours_to_expiry % 1

    omega = garch_params['omega']
    alpha = garch_params.get('alpha', 0.0)  # absent in FIGARCH-fitted dict
    beta_val = garch_params['beta']
    nu = garch_params['nu']
    mu = garch_params['mu']  # Hourly log-return units (scalar)
    current_variance = np.maximum(garch_params['last_variance'], 1e-10)  # Hourly variance

    # FIGARCH precomputed weights (Phase 2.5)
    figarch_weights = garch_params.get('figarch_weights', None)
    if use_figarch and figarch_weights is None:
        # FIGARCH weights missing: fit_garch_model either wasn't called with
        # use_figarch=True or the FIGARCH fit fell back to GARCH.
        # Log once per process.
        global _FIGARCH_FALLBACK_WARNED
        if not _FIGARCH_FALLBACK_WARNED:
            logger.warning("use_figarch=True but no figarch_weights in garch_params; using GARCH")
            _FIGARCH_FALLBACK_WARNED = True
        else:
            logger.debug("use_figarch=True but no figarch_weights in garch_params; using GARCH")
        use_figarch = False

    # FIGARCH lag buffer for past squared returns
    if use_figarch and figarch_weights is not None:
        figarch_trunc_k = len(figarch_weights)
        # Initialize past squared returns with last fitted variance (works for
        # both FIGARCH and GARCH dicts; buffer fills with actual ε² within K steps)
        past_eps_init = garch_params.get('last_variance', omega / (1 - alpha - beta_val) if (alpha + beta_val) < 1 else omega / 0.01)
        past_eps_sq = np.full((n_sims, figarch_trunc_k), past_eps_init)

    # FIX 5 (H3): whether the FIGARCH recursion is actually active (it may have been
    # disabled just above when weights are missing). Decides whether SVCJ vol jumps
    # persist via the decaying `vol_jump_state` (FIGARCH path, no β) or via the
    # inline `variances += vol_jump_mag` that GARCH's β already carries forward.
    figarch_active = use_figarch and figarch_weights is not None

    # Naive prior enforcement (Phase 1.1): set μ=0
    if use_naive_prior:
        mu = 0.0

    log_prices = np.full(n_sims, np.log(S0))
    variances = np.full(n_sims, current_variance)
    # FIX 5 (H3): persistent, decaying SVCJ variance-jump state (FIGARCH path).
    # Carried from step t-1 and added on top of the ARCH(∞) base at step t.
    vol_jump_state = np.zeros(n_sims)

    for step_idx, dt in enumerate(dt_schedule):
        # ---- Innovation Distribution ----
        if use_skewed_t and nu > 2:
            # Skewed-t (Phase 1.4): skewed_t_rvs now returns Hansen's standardized
            # variate (mean 0, variance 1). scale_factor is 1.0; multiply kept for
            # backward-compatible symmetry with the Student-t branch below.
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
                # FIGARCH(1,d,1): σ²_t = ω/(1-β) + Σ λ_k · ε²_{t-k}
                # Shift past returns, insert current
                past_eps_sq = np.roll(past_eps_sq, 1, axis=1)
                past_eps_sq[:, 0] = epsilon_squared
                # Weighted sum (weights[0] = 0, no contemporaneous ε² term)
                figarch_component = np.sum(
                    past_eps_sq[:, :figarch_trunc_k] * figarch_weights[np.newaxis, :figarch_trunc_k],
                    axis=1
                )
                # FIX 5 (H3): add the vol-jump state carried from the PREVIOUS step.
                # The FIGARCH recompute would otherwise overwrite (erase) any SVCJ
                # variance jump every step. `vol_jump_state` is updated at the END of
                # the SVCJ block below, so step t adds state_{t-1} here and writes
                # state_t there — the correct read-then-write ordering.
                variances = omega / (1 - beta_val) + figarch_component + vol_jump_state
            else:
                # Standard GARCH(1,1)
                variances = omega + alpha * epsilon_squared + beta_val * variances

            # Guard against negative variance (numeric / edge case)
            variances = np.maximum(variances, 1e-10)

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

        # ---- SVCJ Volatility Jumps (Phase 1.3 + FIX 5 H3 persistence) ----
        if use_svcj and svcj_mu_v > 0:
            # Eraker (2004) SVCJ: return and variance jump on the SAME Poisson
            # events (k) — contemporaneous, not an independent vol-jump process.
            mask_vol_jump = k > 0

            # FIX 5 (H3) critical init: vol_jump_mag must be defined on EVERY step
            # (including no-jump steps), or the vol_jump_state update below NameErrors
            # / reuses a stale array from a prior jumping step.
            vol_jump_mag = np.zeros(n_sims)

            if np.any(mask_vol_jump):
                # Magnitude of the variance jump given k jumps in the step: sum of
                # k exponential(mean=mu_v) draws = Gamma(k, scale=mu_v). Zero where k=0.
                vol_jump_mag[mask_vol_jump] = rng.gamma(
                    k[mask_vol_jump], scale=svcj_mu_v
                )

                # Eraker et al. (2004): ξ_s | ξ_v ~ N(ρ_J ξ_v, σ_s²), applied ONLY
                # on jump events. Both the deterministic ρ_J term and the stochastic
                # residual must be masked to jumping paths — leaking the residual to
                # non-jumping paths injects spurious per-step variance into every path.
                # FIX 4 (M1): the Eraker equation's rho_J is a SLOPE (return per unit
                # variance jump), not the dimensionless Pearson correlation reported
                # as svcj_rho_j (kept above for logs/meta only). svcj_rho_j_slope is
                # in the correct units; using svcj_rho_j here made the term ~5 orders
                # of magnitude too small to matter (see SVCJ_RHO_J_SLOPE comment).
                # Co-timing (shared Poisson k) + vol-jump persistence remain the
                # dominant SVCJ correlation channels regardless of this term.
                correlated_adjustment = svcj_rho_j_slope * vol_jump_mag  # already 0 off-mask
                stochastic_residual = rng.normal(0, svcj_sigma_s, size=n_sims)
                stochastic_residual[~mask_vol_jump] = 0.0
                jump_sizes += correlated_adjustment + stochastic_residual

                if not figarch_active:
                    # GARCH path: β already persists the vol jump into future
                    # variance, so add it inline as before (unchanged behavior).
                    variances += vol_jump_mag
                    variances = np.maximum(variances, 1e-12)

            if figarch_active:
                # FIGARCH path: persist the vol jump via a decaying state instead.
                # The ARCH(∞) base recomputes variance from the ε² buffer each step
                # (no β), so a one-shot inline add would be erased next step — the
                # original H3 bug. Carry state_t = persist·state_{t-1} + mag_t, read
                # at the top of the next step. Cap to keep the geometric accumulation
                # finite over long horizons; assert finiteness as a safety net.
                vol_jump_state = np.clip(
                    svcj_persist * vol_jump_state + vol_jump_mag,
                    0.0, VOL_JUMP_STATE_CAP,
                )

        total_log_return = garch_ret + jump_sizes
        # Clip per-step return to prevent variance feedback cascade from
        # extreme Student-t tail draws (10σ+ possible with low nu) pushing
        # cumulative log_prices past float64 exp limit (~709).
        # ±2.0 ≈ ±640% hourly return; BTC worst hourly return ever ~40%.
        total_log_return = np.clip(total_log_return, -2.0, 2.0)
        log_prices += total_log_return

    # Belt-and-suspenders: exp(50) ≈ 5e21, well within float64 max 1.8e308.
    log_prices = np.clip(log_prices, -50.0, 50.0)
    return np.exp(log_prices)


def get_contract_probability(paths: np.ndarray, strike_price: float):
    """
    Calculate probability of paths ending at or above strike.

    Args:
        paths: Array of simulated terminal prices.
        strike_price: Strike price for the binary contract.
    """
    return np.mean(paths >= strike_price)


def dte_bucket_horizon(days_to_expiry: float) -> Optional[float]:
    """
    Map days-to-expiry to the XGB DTE-bucket training/forecast horizon (C2-a).

    Returns the bucket midpoint in days (4 / 11 / 22 for the default buckets), or
    None if the horizon falls outside all buckets (>30d → XGB gated off).

    Buckets are right-open in the lower edge / left-closed in the upper, matching
    XGB_DTE_BUCKETS; a contract exactly on an interior edge lands in the higher
    bucket (e.g. 7.0d → 7–14 bucket).
    """
    for lo, hi in XGB_DTE_BUCKETS:
        # left-closed on lo for the first bucket (0), interior edges go to the
        # higher bucket so use (lo <= d < hi) but skip when d == lo and lo is an
        # interior edge already covered by the previous bucket's hi — the loop
        # order (ascending) naturally assigns d==edge to the bucket whose lo==edge.
        if lo <= days_to_expiry < hi:
            return (lo + hi) / 2.0
    return None


def apply_xgb_drift_shift(
    paths: np.ndarray,
    S0: float,
    p_up: float,
    lam: float,
    *,
    p_floor: float = XGB_P_FLOOR,
    p_ceil: float = XGB_P_CEIL,
    target_floor: float = XGB_P_TARGET_FLOOR,
    target_ceil: float = XGB_P_TARGET_CEIL,
    max_shift_frac: float = XGB_MAX_SHIFT_FRAC,
    sigma_h_floor: float = XGB_SIGMA_H_FLOOR,
    p_base_guard: float = XGB_P_BASE_GUARD,
):
    """
    Shift the simulated terminal distribution toward an XGBoost directional view.

    Strike-agnostic, monotonicity-preserving (FIX 3 Step B drift-shift). A single
    constant multiplicative shift `paths * exp(Δ_H)` moves the whole terminal
    distribution; every strike read off the shifted paths stays monotone in the
    strike by construction. See temp/xgb_activation_plan.md §2.

    Δ_H is solved by EMPIRICAL-CDF inversion (exact on the actual non-Gaussian
    distribution): find the constant `c` such that mean(log_ret + c >= 0) == p_target,
    i.e. c = -quantile(log_ret, 1 - p_target). This hits p_target exactly, unlike a
    Gaussian-probit approximation.

    Args:
        paths: Terminal price array S_T (shape [n_sims]).
        S0: Spot price (defines "up" as S_T >= S0).
        p_up: XGBoost P(up) over the horizon (0-1). 0.5 = neutral.
        lam: Tilt strength λ_xgb in [0,1]. 0 = no shift.
        p_floor/p_ceil: clip raw p_up before mapping.
        target_floor/target_ceil: clip the target P(up).
        max_shift_frac: cap |Δ_H| at this fraction of empirical sigma_H.
        sigma_h_floor: sigma_H at/below this → identity (numeric guard).
        p_base_guard: base P(up) within this of 0/1 → identity (deep-skew guard).

    Returns:
        (paths_shifted, delta_H, meta) where meta = {p_base, p_target, sigma_H,
        delta_H, applied}. Identity (delta_H=0, applied=False) on any short-circuit.
    """
    meta = {"p_base": np.nan, "p_target": np.nan, "sigma_H": np.nan,
            "delta_H": 0.0, "applied": False}

    # Short-circuit: neutral signal or knob off.
    if lam == 0.0 or p_up == 0.5 or paths is None or len(paths) == 0 or S0 <= 0:
        return paths, 0.0, meta

    log_ret = np.log(paths / S0)
    sigma_H = float(np.std(log_ret))
    p_base = float(np.mean(paths >= S0))
    meta["sigma_H"] = sigma_H
    meta["p_base"] = p_base

    # Numeric / deep-skew guards.
    if sigma_H <= sigma_h_floor:
        return paths, 0.0, meta
    if p_base <= p_base_guard or p_base >= (1.0 - p_base_guard):
        return paths, 0.0, meta

    # Target probability: linear tilt toward the (clipped) XGB view.
    p_up_clipped = float(np.clip(p_up, p_floor, p_ceil))
    p_target = 0.5 + lam * (p_up_clipped - 0.5)
    p_target = float(np.clip(p_target, target_floor, target_ceil))
    meta["p_target"] = p_target

    # Empirical-CDF inversion: c such that mean(log_ret + c >= 0) == p_target.
    # mean(log_ret >= -c) == p_target  ⇒  -c = quantile(log_ret, 1 - p_target).
    delta_H = float(-np.quantile(log_ret, 1.0 - p_target))

    # Safety cap relative to empirical horizon vol.
    cap = max_shift_frac * sigma_H
    delta_H = float(np.clip(delta_H, -cap, cap))

    if delta_H == 0.0:
        return paths, 0.0, meta

    paths_shifted = paths * np.exp(delta_H)
    meta["delta_H"] = delta_H
    meta["applied"] = True
    return paths_shifted, delta_H, meta


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
    martingale_anchor: bool = False,
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
    xgb_tilt_lambda: float = None,  # tilt strength; None → module XGB_TILT_LAMBDA
    # --- Backtesting ---
    disable_staleness_check: bool = False,
    # FIX 4 (H1): snapshot wall-clock for leak-free, deterministic regime refit
    # gating during time-travel backtests. Keyword-only with a default, so existing
    # callers are unaffected. None → live mode (regime detector uses real wall time).
    as_of: Optional[datetime] = None,
    # Per-snapshot dedup (backtest): a backrunner snapshot prices several expiry
    # groups off an identical hourly slice, so the GARCH/FIGARCH MLE and S0 are
    # identical across groups. `garch_cache` is a caller-owned dict keyed on the
    # effective (post-horizon-gate) use_figarch flag; `s0_override` is the
    # precomputed S0. Both default None → behavior unchanged (load + fit every
    # call). When supplied, the fit (and load, on a cache hit) runs once per
    # snapshot instead of once per expiry group.
    garch_cache: Optional[dict] = None,
    s0_override: Optional[float] = None,
    # T9 (M5): vol_gate_regime affects jump intensity scaling in simulate_paths
    vol_gate_regime: str = "normal",
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
            "Kou return jumps retained; SVCJ/skew/FIGARCH/regime/XGB disabled."
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
    # Per-snapshot dedup: this block runs AFTER the horizon gate above has
    # finalized `use_figarch`, so a cache keyed on that flag mirrors the
    # FIGARCH↔GARCH choice exactly (incl. the deterministic FIGARCH→GARCH
    # convergence fallback — same slice ⇒ same returned dict). `hourly_returns`
    # only feeds the fit, so the load is skipped entirely on a cache hit with S0
    # supplied. Defaults (None) reproduce the original load-then-fit behavior.
    need_fit = not (garch_cache is not None and use_figarch in garch_cache)

    if need_fit or s0_override is None:
        hourly_returns, S0_loaded = load_and_prep_data(
            hourly_csv=hourly_csv,
            intraday_csv=intraday_csv,
            hourly_df=hourly_df,
            intraday_df=intraday_df,
            training_start_date=training_start_date,
            disable_staleness_check=disable_staleness_check,
        )
        S0 = s0_override if s0_override is not None else S0_loaded
    else:
        S0 = s0_override

    if need_fit:
        garch_params = fit_garch_model(
            hourly_returns,
            training_start_date=training_start_date,
            use_figarch=use_figarch,
        )
        if garch_cache is not None:
            garch_cache[use_figarch] = garch_params
    else:
        garch_params = garch_cache[use_figarch]

    # ---- Regime Detection (Phase 1.2) ----
    regime_weights = {"bear": 0.0, "sideways": 1.0, "bull": 0.0}
    # T6 (H3): horizon-propagated weights actually used for path allocation
    # (see below); defaults to the t0 posterior until a successful fit.
    regime_weights_used = regime_weights
    regime_variance_scales = {"bear": 1.0, "sideways": 1.0, "bull": 1.0}
    dominant_regime = "sideways"

    if use_regime_switching and regime_detector is not None:
        try:
            from core.pricing.regime_detector import hourly_to_daily_returns

            # Get daily returns for HMM
            if hourly_df is not None:
                daily_ret = hourly_to_daily_returns(df=hourly_df)
            else:
                daily_ret = hourly_to_daily_returns(hourly_path=hourly_csv)

            # Fit/predict. FIX 4 (H1): thread `as_of` into the refit gate so a
            # time-travel backtest never refits on real wall-clock time (which would
            # both leak future data into the refit cadence and be non-deterministic).
            # In live mode as_of is None → RegimeDetector.fit falls back to wall time.
            regime_weights, dominant_regime = regime_detector.fit_predict(daily_ret, now=as_of)
            logger.info(f"Regime detection: dominant={dominant_regime}, weights={regime_weights}")

            # T6 (H3): a 14-30 DTE contract should not be priced with today's
            # regime posterior held fixed over the entire path -- propagate it
            # through the transition matrix (average-occupancy approximation,
            # PRICING_REVIEW.md H3 point 3). Guard on FIT SUCCESS
            # (regime_detector._model is not None), NOT on weight values: an
            # all-sideways posterior {bear:0, sideways:1, bull:0} is a
            # legitimate fit result, identical in VALUE to the unfitted
            # default, so it must not be treated as "fit failed".
            if regime_detector._model is not None:
                regime_weights_used = regime_detector.predict_weights(
                    n_days_ahead=int(round(days_to_expiry / 2))
                )
                regime_variance_scales = regime_detector.get_regime_variance_scales()
            else:
                regime_weights_used = regime_weights
        except Exception as e:
            logger.warning(f"Regime detection failed ({e}); using default sideways regime")
            regime_weights_used = regime_weights

    # ---- Regime-Conditional Simulation OR Single Simulation ----
    if use_regime_switching and regime_detector is not None:
        # Mixture by PROPORTIONAL ALLOCATION: each active regime receives a path
        # count proportional to its HORIZON-PROPAGATED HMM weight, summing to
        # ~n_sims. This encodes the mixture in the sample itself (equal
        # per-path weight downstream) and preserves the full effective sample
        # size -- the old fixed n_sims//3 split dropped to ~n_sims/3 effective
        # paths whenever one regime dominated.
        regime_labels = ["bear", "sideways", "bull"]
        active = [(rl, regime_weights_used.get(rl, 0.0))
                  for rl in regime_labels if regime_weights_used.get(rl, 0.0) >= 0.01]
        total_w = sum(w for _, w in active)

        # T6 (H3): sideways-only fast path. If the (horizon-propagated) mixture
        # puts >=99% weight on a single label AND that label's variance scale
        # is within +/-5% of neutral, the mixture machinery is pure overhead --
        # skip straight to one simulation with that regime's params (preserves
        # the pre-T6 fast-path semantics for the common near-unanimous case).
        _dominant_used = max(regime_weights_used, key=regime_weights_used.get) if regime_weights_used else "sideways"
        _dominant_scale = regime_variance_scales.get(_dominant_used, 1.0)
        if regime_weights_used.get(_dominant_used, 0.0) >= 0.99 and 0.95 <= _dominant_scale <= 1.05:
            if use_skewed_t:
                if _dominant_used == "bear":
                    st_lam = -0.3
                elif _dominant_used == "bull":
                    st_lam = 0.2
                else:
                    st_lam = 0.0
            else:
                st_lam = SKEWED_T_LAMBDA_DEFAULT

            paths = simulate_paths(
                S0=S0, garch_params=garch_params, jump_params=jump_params,
                hours_to_expiry=hours_to_expiry, n_sims=n_sims, seed=seed,
                martingale_anchor=martingale_anchor, use_naive_prior=use_naive_prior,
                use_svcj=use_svcj, use_skewed_t=use_skewed_t, skewed_t_lam=st_lam,
                use_figarch=use_figarch, regime_jump_params=regime_params,
                regime_label=_dominant_used, vol_gate_regime=vol_gate_regime,
            )
        else:
            all_paths = []
            for i, (rl, w) in enumerate(active):
                n_r = int(round(n_sims * w / total_w)) if total_w > 0 else 0
                if n_r <= 0:
                    continue

                # Distinct sub-seed per regime so the regime draws are INDEPENDENT.
                # Sharing one seed across regimes correlates their innovations and
                # breaks the mixture's independence assumption.
                seed_r = None if seed is None else int(seed) + i + 1

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

                # T6 (H3): per-regime variance scaling on a SHALLOW COPY of
                # garch_params -- scaling omega scales the GARCH/FIGARCH
                # unconditional variance by s (alpha/beta persistence
                # unchanged); last_variance (the warm-start / FIGARCH eps^2
                # init) scales too. NEVER mutate garch_params itself: it may
                # be the caller's garch_cache-shared dict, reused across
                # expiry groups and other regimes in this same call.
                scale = regime_variance_scales.get(rl, 1.0)
                garch_params_r = dict(garch_params)
                garch_params_r['omega'] = garch_params_r['omega'] * scale
                garch_params_r['last_variance'] = garch_params_r['last_variance'] * scale

                paths = simulate_paths(
                    S0=S0,
                    garch_params=garch_params_r,
                    jump_params=jump_params,
                    hours_to_expiry=hours_to_expiry,
                    n_sims=n_r,
                    seed=seed_r,
                    martingale_anchor=martingale_anchor,
                    use_naive_prior=use_naive_prior,
                    use_svcj=use_svcj,
                    use_skewed_t=use_skewed_t,
                    skewed_t_lam=st_lam,
                    use_figarch=use_figarch,
                    regime_jump_params=regime_params,
                    regime_label=rl,
                    vol_gate_regime=vol_gate_regime,
                )

                all_paths.append(paths)

            if not all_paths:
                # Fallback: single simulation
                paths = simulate_paths(
                    S0=S0, garch_params=garch_params, jump_params=jump_params,
                    hours_to_expiry=hours_to_expiry, n_sims=n_sims, seed=seed,
                    martingale_anchor=martingale_anchor,
                    use_naive_prior=use_naive_prior, use_svcj=use_svcj,
                    use_skewed_t=use_skewed_t, use_figarch=use_figarch,
                    regime_label=dominant_regime, vol_gate_regime=vol_gate_regime,
                )
            else:
                # Proportional allocation already encodes the mixture: equal weight.
                paths = np.concatenate(all_paths)
    else:
        # ---- Single Simulation (legacy / non-regime path) ----
        paths = simulate_paths(
            S0=S0,
            garch_params=garch_params,
            jump_params=jump_params,
            hours_to_expiry=hours_to_expiry,
            n_sims=n_sims,
            seed=seed,
            martingale_anchor=martingale_anchor,
            use_naive_prior=use_naive_prior,
            use_svcj=use_svcj,
            use_skewed_t=use_skewed_t,
            use_figarch=use_figarch,
            regime_jump_params=regime_params,
            regime_label=dominant_regime,
            vol_gate_regime=vol_gate_regime,
        )

    # ---- Phase 2.3: Directional XGBoost drift shift (FIX 3 / H2 re-enabled) ----
    # The XGBoost P(up) is applied ONCE to the assembled terminal distribution
    # (strike-agnostic), BEFORE the per-strike loop, so every strike is re-derived
    # from the shifted paths and the ladder stays monotone by construction. This
    # replaces the old invalid per-strike additive blend. Physical-measure view:
    # skipped under martingale_anchor; gated to DTE buckets (≤30d). See
    # apply_xgb_drift_shift / temp/xgb_activation_plan.md §2.
    xgb_p_up = np.nan
    xgb_delta_H = 0.0
    xgb_applied = False
    if use_xgb_direction and xgb_model is not None:
        bucket_h = dte_bucket_horizon(days_to_expiry)
        if martingale_anchor:
            logger.warning(
                "XGB drift skipped: incompatible with martingale_anchor=True "
                "(directional tilt is a physical-measure view)."
            )
        elif bucket_h is None:
            logger.debug("XGB drift skipped: DTE %.2fd outside buckets", days_to_expiry)
        else:
            try:
                from core.pricing.directional_xgb import to_daily_log_return_series
                # C1: derive a leak-free DATE-INDEXED daily-returns series
                # UNCONDITIONALLY (not the regime-only `daily_ret` at ~1170).
                if hourly_df is not None:
                    daily_ret_xgb = to_daily_log_return_series(hourly_df)
                else:
                    daily_ret_xgb = to_daily_log_return_series(pd.read_csv(hourly_csv))
                p_up = xgb_model.predict_direction_adjustment(
                    S0=S0,  # no-op in the model (no price feature); API symmetry
                    hours_to_expiry=hours_to_expiry,
                    btc_returns=daily_ret_xgb,
                    macro_df=macro_df,
                    horizon_days=int(round(bucket_h)),
                )
                lam = XGB_TILT_LAMBDA if xgb_tilt_lambda is None else xgb_tilt_lambda
                paths, xgb_delta_H, xgb_meta = apply_xgb_drift_shift(
                    paths, S0, p_up, lam
                )
                xgb_p_up = p_up
                xgb_applied = xgb_meta.get("applied", False)
            except Exception:
                logger.warning("XGB drift shift failed; using unshifted paths", exc_info=True)

    # ---- Compute Probabilities ----
    results = {}
    for strike in strikes:
        prob = get_contract_probability(paths, strike)
        results[strike] = float(prob)

    # ---- Build Extended Results ----
    results['_meta'] = {
        'S0': S0,
        'hours_to_expiry': hours_to_expiry,
        'n_sims': n_sims,
        'regime_weights': regime_weights,
        # T6 (H3): horizon-propagated weights actually used for path
        # allocation (t0 posterior stepped forward through transmat_ by
        # round(days_to_expiry/2) days); equals regime_weights when
        # regime switching is off or n_days_ahead resolves to 0.
        'regime_weights_used': regime_weights_used,
        # T6 (H3): per-label variance scale actually applied to garch_params
        # in the mixture branch (get_regime_variance_scales()); all-1.0 when
        # regime switching is off/unfitted or the sideways-only fast path fired.
        'regime_variance_scales': regime_variance_scales,
        'dominant_regime': dominant_regime,
        'use_naive_prior': use_naive_prior,
        'martingale_anchor': martingale_anchor,
        'use_regime_switching': use_regime_switching,
        'use_svcj': use_svcj,
        'use_skewed_t': use_skewed_t,
        'use_figarch': use_figarch,
        'use_xgb_direction': use_xgb_direction,
        'xgb_p_up': xgb_p_up,
        'xgb_delta_H': xgb_delta_H,
        'xgb_applied': xgb_applied,
        'vol_gate_regime': vol_gate_regime,
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
        lam, p_crash, eta_up, eta_down, mu_v, rho_J, rho_j_slope, lam_v,
        n_jumps_detected, fit_converged, calibration_date.
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
            # FIX 4 (M1) READ path: pre-fix cache CSVs have no rho_j_slope column;
            # .get() falls back to 0.0 (SVCJ_RHO_J_SLOPE default = term off).
            "rho_j_slope": float(row.get("rho_j_slope", 0.0)),
            "lam_v": float(row.get("lam_v", row["lam"])),
            "n_jumps_detected": int(row.get("n_jumps_detected", 0)),
            "fit_converged": bool(int(row.get("fit_converged", 1))),
            "calibration_date": str(row.get("calibration_date", "unknown")),
        }
    else:
        logger.info("Calibrating jump parameters from %s ...", hourly_csv)
        result: JumpCalibrationResult = calibrate_jumps(
            hourly_csv=hourly_csv,
            # FIX 2 (M4): bipower default everywhere for backtest↔live parity.
            detection_method="bipower",
        )
        calibrated = {
            "lam": result.lam,
            "p_crash": result.p_crash,
            "eta_up": result.eta_up,
            "eta_down": result.eta_down,
            "mu_v": result.mu_v,
            "rho_J": result.rho_J,
            # FIX 4 (M1) WRITE path: without this, live-mode cal["rho_j_slope"]
            # is always absent (KeyError-free via .get() downstream, but silently
            # 0.0) and the T4 slope term is inert live even when calibration
            # produces a nonzero slope.
            "rho_j_slope": result.rho_j_slope,
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
    base_rho_j_slope: float = SVCJ_RHO_J_SLOPE,
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
        Dict mapping regime_label -> dict of jump parameters. Every regime dict
        carries `rho_j_slope` (FIX 4/M1) alongside the legacy `rho_J`. NOTE: the
        bear 1.5x / bull 0.5x regime multipliers below now apply to rho_j_slope
        (the term actually used in simulate_paths); rho_J keeps its own
        multiplier for reporting compat, but is otherwise inert. Since
        base_rho_j_slope defaults to 0.0 (SVCJ_RHO_J_SLOPE), the multipliers are
        no-ops until a calibration supplies a nonzero slope -- that is intended.
    """
    # Determine base parameters
    if calibrated is not None and calibrated.get("fit_converged", False):
        base_lam = calibrated.get("lam", base_lam)
        base_p_crash = calibrated.get("p_crash", base_p_crash)
        base_eta_up = calibrated.get("eta_up", base_eta_up)
        base_eta_down = calibrated.get("eta_down", base_eta_down)
        base_mu_v = calibrated.get("mu_v", base_mu_v)
        base_rho_j = calibrated.get("rho_J", base_rho_j)
        base_rho_j_slope = calibrated.get("rho_j_slope", base_rho_j_slope)
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
                # FIX 4 (M1): RegimeJumpResult.rho_j_slope (default 0.0 for
                # results computed before this field existed).
                "rho_j_slope": getattr(rc, "rho_j_slope", 0.0),
                "sigma_s": base_sigma_s,  # sigma_s not regime-calibrated; use base
            }
            logger.info(
                "Regime '%s': using directly calibrated jump params "
                "(lam=%.1f, p_crash=%.3f, %d jumps)",
                regime, rc.lam, rc.p_crash, rc.n_jumps,
            )
        else:
            # Apply hardcoded multipliers to base parameters. FIX 4 (M1): the
            # bear/sideways/bull multipliers now apply to rho_j_slope (the term
            # simulate_paths actually uses); rho_J keeps its own multiplier for
            # reporting compat only.
            if regime == "bear":
                regime_params[label] = {
                    "lambda": base_lam * 1.5,
                    "crash_prob": min(base_p_crash * 1.3, 0.85),
                    "eta_up": base_eta_up * 0.8,
                    "eta_down": base_eta_down * 0.7,
                    "mu_v": base_mu_v * 2.0,
                    "rho_J": base_rho_j * 1.5,
                    "rho_j_slope": base_rho_j_slope * 1.5,
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
                    "rho_j_slope": base_rho_j_slope * 1.0,
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
                    "rho_j_slope": base_rho_j_slope * 0.5,
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
    # Test 6b: SVCJ Vol Jumps UNDER FIGARCH (Phase 2.5 + FIX 5 / H3)
    # -------------------------------------------------------------------------
    # Regression guard for the H3 bug: under FIGARCH the variance recompute used to
    # erase the SVCJ vol-jump add every step, so SVCJ ≈ plain return jump (no extra
    # tail variance). With the persistent decaying vol_jump_state, SVCJ must now
    # measurably increase terminal-return std. Uses an inflated mu_v=1e-4 (as in
    # Test 6) so the lift is well above MC noise; at the calibrated default
    # (2.5e-5) the effect can be below the noise floor and the test would flap.
    print("\n[Test 6b] SVCJ Volatility Jumps under FIGARCH...")
    _fig_weights = _compute_figarch_weights(d=0.3889, phi=0.3056, beta=0.4558, trunc_k=200)
    test_figarch6 = {
        'omega': 0.00001 / 24,
        'beta': 0.4558,
        'nu': 5.0,
        'mu': 0.0,
        'last_variance': 0.0004 / 24,
        'use_figarch': True,
        'figarch_weights': _fig_weights,
        'figarch_d': 0.3889,
        'figarch_phi': 0.3056,
    }
    paths_fig_no_svcj = simulate_paths(
        S0=100000.0, garch_params=test_figarch6,
        jump_params={'lambda': 25.0, 'crash_prob': 0.6, 'eta_up': 50.0, 'eta_down': 25.0,
                     'mu_v': 0.0, 'rho_J': 0.0},
        hours_to_expiry=720.0, n_sims=5000, seed=42,
        use_svcj=False, use_figarch=True,
    )
    paths_fig_svcj = simulate_paths(
        S0=100000.0, garch_params=test_figarch6,
        jump_params={'lambda': 25.0, 'crash_prob': 0.6, 'eta_up': 50.0, 'eta_down': 25.0,
                     'mu_v': 0.0001, 'rho_J': -0.3},
        hours_to_expiry=720.0, n_sims=5000, seed=42,
        use_svcj=True, use_figarch=True,
    )
    std_fig_svcj = np.std(np.log(paths_fig_svcj / 100000))
    std_fig_no_svcj = np.std(np.log(paths_fig_no_svcj / 100000))
    finite_ok = np.all(np.isfinite(paths_fig_svcj)) and np.all(paths_fig_svcj > 0)

    if std_fig_svcj > std_fig_no_svcj * 1.03 and finite_ok:
        print(f"  PASS: FIGARCH SVCJ vol={std_fig_svcj:.6f} > 1.03 * SVJ vol={std_fig_no_svcj:.6f} "
              f"(vol jumps persist under FIGARCH)")
    else:
        print(f"  FAIL: FIGARCH SVCJ vol={std_fig_svcj:.6f} <= 1.03 * SVJ vol={std_fig_no_svcj:.6f} "
              f"or non-finite paths (finite_ok={finite_ok})")
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 7: FIGARCH(1,d,1) Weights vs arch library reference (Phase 2.5)
    # -------------------------------------------------------------------------
    print("\n[Test 7] FIGARCH(1,d,1) Weights...")
    try:
        from arch.univariate.recursions_python import figarch_weights_python
        _HAS_ARCH_RECURSIONS = True
    except ImportError:
        _HAS_ARCH_RECURSIONS = False

    # Test with params matching the fit result on our hourly BTC data
    phi, d_val, beta = 0.3056, 0.3889, 0.4558
    weights = _compute_figarch_weights(d_val, phi, beta, trunc_k=100)

    # 1. weights[0] must be 0 (no contemporaneous eps^2 term)
    if weights[0] == 0.0:
        print(f"  PASS: weights[0]=0 (contemporaneous eps^2 excluded)")
    else:
        print(f"  FAIL: weights[0]={weights[0]:.6f} (must be 0)")
        all_passed = False

    # 2. lambda_1 (weights[1]) must be positive (B-M positivity)
    if weights[1] > 0:
        print(f"  PASS: lambda_1={weights[1]:+.6f} > 0 (B-M positivity satisfied)")
    else:
        print(f"  FAIL: lambda_1={weights[1]:+.6f} <= 0 (B-M positivity violated)")
        all_passed = False

    # 3. Hyperbolic decay: first weight dominates tail
    if abs(weights[1]) > abs(weights[-1]) * 10:
        print(f"  PASS: hyperbolic decay (lambda_1={weights[1]:.6f}, lambda_99={weights[-1]:.12f})")
    else:
        print(f"  FAIL: unexpected decay pattern")
        all_passed = False

    # 4. Validate against arch library reference implementation
    if _HAS_ARCH_RECURSIONS:
        ref_weights = figarch_weights_python(
            np.array([phi, d_val, beta]), p=1, q=1, trunc_lag=99)
        # our weights[1:] should match ref_weights[0:]
        max_diff = np.max(np.abs(weights[1:] - ref_weights))
        if max_diff < 1e-12:
            print(f"  PASS: matches arch library reference (max diff={max_diff:.2e})")
        else:
            print(f"  FAIL: max diff vs arch library = {max_diff:.2e}")
            all_passed = False
    else:
        print(f"  SKIP: arch.univariate.recursions_python not importable")

    # 5. All weights must be non-negative
    if np.all(weights >= -1e-10):
        print(f"  PASS: all weights non-negative (min={weights.min():.8f})")
    else:
        neg = int(np.sum(weights < -1e-10))
        print(f"  FAIL: {neg} negative weights")
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
    n_var_test = 200000
    nu_test = 5.0

    all_var_ok = True
    for lam_test in [-0.3, -0.1, 0.0, 0.2, 0.3]:
        # Fresh independent RNG per lam for reproducible per-case estimates.
        rng9 = np.random.default_rng(99 + int(round(lam_test * 10)))
        s = skewed_t_rvs(nu_test, lam_test, n_var_test, rng9)
        sf = skewed_t_scale_factor(nu_test, lam_test)
        samples = s * sf  # Combined output as used in simulate_paths
        emp_var = np.var(samples)
        emp_mean = np.mean(samples)
        # Hansen standardized skew-t: mean 0, variance 1 by construction. Band
        # reflects finite-sample MC noise (t(nu=5) has high kurtosis -> noisy var
        # estimate). The old non-standardized sampler injected a nonzero mean
        # (unintended drift); the |mean| check is the key regression guard.
        var_ok = 0.96 <= emp_var <= 1.04
        mean_ok = abs(emp_mean) < 0.02
        ok = var_ok and mean_ok
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_var_ok = False
        print(f"    lam={lam_test:+4.1f}: mean={emp_mean:+.4f} variance={emp_var:.4f} [{status}]")

    if all_var_ok:
        print("  PASS: All skewed-t mean~0 (|m|<0.02) and variance in [0.96, 1.04]")
    else:
        print("  FAIL: One or more skewed-t mean/variance out of bounds")
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
