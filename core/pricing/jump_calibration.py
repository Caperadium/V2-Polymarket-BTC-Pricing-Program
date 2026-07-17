"""
jump_calibration.py

Historical jump calibration from threshold exceedances on BTC hourly returns.
Estimates Kou double-exponential parameters and SVCJ volatility jump parameters
without requiring MCMC — uses MAD-based jump detection + MLE.

Based on: Teng et al. (2025), Qiao et al. (2025), Eraker et al. (2004).

Usage:
    from core.pricing.jump_calibration import calibrate_jumps
    params = calibrate_jumps("DATA/btc_hourly.csv")
    print(params)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import expon, gamma

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trailing-window (era-conditioned) jump calibration constants
# (Package C, work package W1 -- see temp/mm_package_c_plan.md section
# 2.2-REV5, eta_up-only mask-slice windowing)
# ---------------------------------------------------------------------------

# Trailing window (hours) used to era-condition the Kou UP-JUMP mean size
# (eta_up) only. 8760 = 12 months of hourly bars. lam, p_crash, eta_down and
# the SVCJ vol-jump leg are NEVER windowed (always full-slice -- see
# calibrate_jumps; plan section 2.2-REV5, replacing the REV5-SUPERSEDED
# all-params blend).
JUMP_CAL_WINDOW_HOURS = 8760

# Credibility target: at this many in-window UP jumps (mask-slice count),
# the windowed eta_up leg is fully trusted (blend weight w = 1.0). Below it,
# eta_up is shrunk linearly (in mean-size space) toward the full-slice
# estimate.
JUMP_CAL_WINDOW_TARGET_UP_JUMPS = 6

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class JumpCalibrationResult:
    """Calibrated jump parameters for SVCJ + Kou double-exponential jumps."""
    # Kou return jump parameters (annualized)
    lam: float          # Jump intensity (jumps per year)
    p_crash: float      # Probability jump is downward
    eta_up: float       # Positive jump size decay (1/mean)
    eta_down: float     # Negative jump size decay (1/mean)

    # SVCJ volatility jump parameters
    mu_v: float         # Mean volatility jump size (variance units, hourly)
    rho_J: float        # Return-vol jump correlation
    lam_v: float        # Vol jump intensity (can differ from return lam; default = lam)

    # FIX 4 (M1): Eraker (2004) regression SLOPE (return per unit variance jump),
    # distinct from rho_J (a Pearson correlation, dimensionless). Default 0.0 =
    # term off (old effective behavior -- rho_J was previously misused as a slope
    # in simulate_paths, which made the term ~5 orders of magnitude too small to
    # matter). See _estimate_vol_jump_params for the OLS estimate.
    rho_j_slope: float = 0.0

    # Diagnostics
    n_jumps_detected: int = 0
    n_obs: int = 0
    jump_threshold: float = 0.0
    detection_method: str = "MAD"
    fit_converged: bool = True

    # Trailing-window calibration diagnostics (Package C / W1, plan section
    # 2.2-REV5). Additive, default-neutral so old construction sites keep
    # working unchanged. These describe the UP SIDE ONLY (eta_up-only
    # mask-slice windowing) -- lam, p_crash, eta_down and SVCJ are always
    # full-slice, regardless of window_hours.
    # None / 1.0 / 0 = "not windowed" (matches the window_hours=None path).
    calibration_window_hours: Optional[int] = None
    window_weight: float = 1.0  # up-side blend weight w
    n_window_jumps: int = 0     # mask-slice in-window UP-jump count

    # Literature reference values (Teng 2025)
    TENG_RHO_J: float = field(default=-0.08, repr=False)
    TENG_MU_V_HOURLY: float = field(default=0.000025, repr=False)


# ---------------------------------------------------------------------------
# Jump Detection
# ---------------------------------------------------------------------------

def detect_jumps_mad(
    returns: np.ndarray,
    mad_multiplier: float = 3.0,
) -> Tuple[np.ndarray, float]:
    """
    Detect jumps using Median Absolute Deviation threshold.

    Args:
        returns: Array of log returns.
        mad_multiplier: Threshold = mad_multiplier * MAD. Default 3.0.

    Returns:
        (jump_mask, threshold) where jump_mask is boolean array.
    """
    median = np.median(returns)
    mad = np.median(np.abs(returns - median))
    threshold = mad_multiplier * mad

    jump_mask = np.abs(returns - median) > threshold
    return jump_mask, threshold


def detect_jumps_bipower(
    returns: np.ndarray,
    significance: float = 0.01,
    window: int = 78,
    return_sigma: bool = False,
):
    """
    Detect jumps using the Lee & Mykland (2008) local bipower test.

    This is the rigorous form of "bipower" jump detection and replaces the old
    global Barndorff-Nielsen & Shephard aggregate test, which gated ALL detection
    behind a single sample-wide statistic that never rejects on large samples
    (e.g. 0 jumps flagged in 40k+ hourly BTC returns) — making it useless for
    calibration. Lee-Mykland tests each return against a LOCAL, jump-robust
    volatility (bipower variation over a trailing window), with a critical value
    that scales with sample size via the Gumbel-extreme-value limit. It is far
    less sensitive to volatility clustering and fat tails than a fixed MAD
    threshold (which over-flags ~14% of heavy-tailed hourly returns as "jumps").

    Statistic (Lee & Mykland 2008, eq. for L_t):
        L_t = |r_t| / σ̂_t ,   σ̂_t² = (1/(K-2)) Σ_{i=t-K+2}^{t} |r_{i-1}||r_i|
    Reject H0 (no jump) when
        (|L_t| − C_n) / S_n  >  G⁻¹(1−β),   G⁻¹(1−β) = −ln(−ln(1−β))
    with c=√(2/π), C_n=(2 ln n)^{1/2}/c − (ln π + ln ln n)/(2c(2 ln n)^{1/2}),
    S_n = 1/(c (2 ln n)^{1/2}).

    FIX 2 (M3): sigma_hat_t must use products up to |r_{t-2}||r_{t-1}| only
    (LM's window ends at t-1) -- a plain `.shift(1)` on one side of the product
    still leaves |r_{t-1}|*|r_t| as the last term, which contains the
    contemporaneous return being tested. A genuine jump at t then inflates its
    own threshold. The extra `.shift(1)` on the whole product series removes
    that term.

    Args:
        returns: Array of log returns.
        significance: Test level beta (default 0.01).
        window: Trailing window K (bars) for the local bipower vol estimate.
            K=78 is the LM guidance for 5-minute bars; the appropriate K for
            hourly bars is untested -- kept at 78 for calibration continuity
            (downstream calibrations depend on this default), not because it is
            known optimal for this sampling frequency.
        return_sigma: If True, also return the local bipower sigma array (after
            the median backfill) used for the test. Default False (old
            single-value return) for backward compatibility.

    Returns:
        Boolean jump_mask aligned to *returns* (default), or
        (jump_mask, sigma_local) when return_sigma=True.
    """
    n = len(returns)
    if n < window + 2:
        empty_mask = np.zeros(n, dtype=bool)
        if return_sigma:
            return empty_mask, np.full(n, np.nan)
        return empty_mask

    abs_ret = np.abs(returns)
    # Local jump-robust (bipower) variance: trailing mean of |r_{t-1}|*|r_t|,
    # scaled by pi/2, with the WHOLE product series shifted by one more bar so
    # sigma_hat_t at time t uses only products up to |r_{t-2}||r_{t-1}| (window
    # ends at t-1, per Lee & Mykland 2008) -- the contemporaneous return r_t
    # never enters its own threshold (FIX 2 / M3).
    s = pd.Series(abs_ret)
    bpv_local = (np.pi / 2.0) * (s.shift(1) * s).shift(1).rolling(
        window, min_periods=max(10, window // 3)
    ).mean()
    sigma_local = np.sqrt(bpv_local.to_numpy())
    # Backfill the leading NaNs with the median local vol so early bars are testable
    # against a sane scale rather than dropped.
    med = np.nanmedian(sigma_local)
    sigma_local = np.where(np.isfinite(sigma_local) & (sigma_local > 0), sigma_local, med)
    if not np.isfinite(med) or med <= 0:
        empty_mask = np.zeros(n, dtype=bool)
        if return_sigma:
            return empty_mask, sigma_local
        return empty_mask

    L = abs_ret / sigma_local

    # Gumbel-based critical value (scales with n).
    c = np.sqrt(2.0 / np.pi)
    sqrt_2logn = np.sqrt(2.0 * np.log(n))
    C_n = sqrt_2logn / c - (np.log(np.pi) + np.log(np.log(n))) / (2.0 * c * sqrt_2logn)
    S_n = 1.0 / (c * sqrt_2logn)
    gumbel_q = -np.log(-np.log(1.0 - significance))

    jump_mask = (L - C_n) / S_n > gumbel_q
    jump_mask = np.asarray(jump_mask, dtype=bool)
    if return_sigma:
        return jump_mask, sigma_local
    return jump_mask


# ---------------------------------------------------------------------------
# Kou Parameter Estimation (MLE on detected jumps)
# ---------------------------------------------------------------------------

def fit_kou_params(
    jump_returns: np.ndarray,
    method: str = "mle",
) -> Tuple[float, float, float, float]:
    """
    Fit Kou double-exponential parameters to detected jump returns.

    Args:
        jump_returns: Array of returns at detected jump times.
        method: "mle" or "moments".

    Returns:
        (p_crash, eta_up, eta_down, n_jumps_total)
    """
    if len(jump_returns) < 10:
        logger.warning("Too few detected jumps for reliable Kou fit; using literature defaults")
        return 0.6, 50.0, 25.0, 25.0

    up_jumps = jump_returns[jump_returns > 0]
    down_jumps = -jump_returns[jump_returns < 0]

    n_up = len(up_jumps)
    n_down = len(down_jumps)
    n_total = n_up + n_down

    if n_up < 3 or n_down < 3:
        return 0.6, 50.0, 25.0, 25.0

    p_crash = n_down / n_total

    # MLE for exponential: eta = 1 / mean
    eta_up = 1.0 / np.mean(up_jumps) if n_up > 0 else 50.0
    eta_down = 1.0 / np.mean(down_jumps) if n_down > 0 else 25.0

    # Lambda: jumps per year = (n_jumps / n_total_obs) * hours_per_year
    return p_crash, eta_up, eta_down, n_total


# ---------------------------------------------------------------------------
# SVCJ Volatility-Jump Estimation (shared by calibrate_jumps and
# calibrate_regime_jumps -- FIX 3 / M2)
# ---------------------------------------------------------------------------

def _estimate_vol_jump_params(
    returns: np.ndarray,
    jump_mask: np.ndarray,
    jump_returns: np.ndarray,
    window: int = 24,
    sigma_local: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, int]:
    """
    Estimate SVCJ volatility-jump parameters (mu_v, rho_J, rho_j_slope) from a
    jump mask and the returns at jump times.

    FIX 3 (M2): the trailing rolling-variance INPUT has each jump bar's squared
    return replaced with the LOCAL diffusion variance at that bar (sigma_local**2
    when available, else the median non-jump squared return) before the rolling
    mean is taken. Without this, the post-jump window (which contains the jump
    bar itself) is mechanically inflated by ~J^2/window regardless of whether
    true diffusion volatility moved at all -- the confound this fix removes.

    Args:
        returns: Full array of log returns (same array the jump_mask indexes).
        jump_mask: Boolean array aligned to returns, True at detected jump bars.
        jump_returns: returns[jump_mask] (passed separately since callers
            already compute it for the Kou fit).
        window: Rolling window (bars) for the variance estimate. Default 24 (1
            day of hourly bars) -- unchanged from the pre-fix behavior.
        sigma_local: Optional local bipower sigma array (from
            detect_jumps_bipower(..., return_sigma=True)), same length as
            returns. When None, jump bars are replaced with the median non-jump
            squared return instead (coarser, but still removes the J^2 term).

    Returns:
        (mu_v, rho_J_corr, rho_j_slope, n_events_used)

        mu_v: Censored-at-zero mean vol jump (hourly variance units), i.e.
            mean(max(delta_var, 0)) over ALL jump events (zeros included), then
            clipped to [1e-6, 1e-3]. This is the MLE mean of an exponential
            censored at zero and is conservative (biased LOW) versus the old
            selective mean of positive deltas only (biased HIGH -- an upward
            selection bias on top of the J^2 confound).
        rho_J_corr: Pearson correlation between jump return and delta-variance
            truncated at zero (same convention as the pre-fix code) -- reporting
            / diagnostics only, NOT used as a slope downstream.
        rho_j_slope: OLS slope of jump_return on delta_var (delta_var NOT
            truncated at zero here -- the slope needs the untruncated sign),
            i.e. cov(jump_ret, dv) / var(dv). Sanity-capped so
            |rho_j_slope| * mu_v <= 0.5 * mean(|jump_returns|) (prevents a noisy
            slope from dominating jump sizes downstream). 0.0 if var(dv) == 0 or
            fewer than 10 usable jump events.
        n_events_used: Number of jump events with a valid pre/post window.
    """
    returns = np.asarray(returns)
    jump_mask = np.asarray(jump_mask, dtype=bool)
    sq = returns ** 2

    if sigma_local is not None and len(sigma_local) == len(returns):
        sigma_local = np.asarray(sigma_local)
        valid_sigma = np.isfinite(sigma_local) & (sigma_local > 0)
        replace_mask = jump_mask & valid_sigma
        sq = sq.copy()
        sq[replace_mask] = sigma_local[replace_mask] ** 2
    else:
        non_jump_sq = sq[~jump_mask]
        fallback = np.median(non_jump_sq) if len(non_jump_sq) > 0 else np.median(sq)
        sq = sq.copy()
        sq[jump_mask] = fallback

    rolling_var = pd.Series(sq).rolling(window, min_periods=4).mean().to_numpy()

    # Preserve the existing +2/-2 pre/post index offsets (window geometry is not
    # part of this fix; the jump-square replacement and censored mean are).
    jump_indices = np.where(jump_mask)[0]
    deltas = []        # untruncated delta_var, for the slope
    jrets_for_slope = []
    corr_deltas = []   # truncated-at-zero delta_var, for the correlation (as before)
    corr_rets = []

    for j_idx, full_idx in enumerate(jump_indices):
        if full_idx >= 2 and full_idx < len(rolling_var) - 1 and j_idx < len(jump_returns):
            pre_var = np.nan_to_num(rolling_var[max(0, full_idx - 2)], nan=0.0)
            post_var = np.nan_to_num(rolling_var[min(len(rolling_var) - 1, full_idx + 2)], nan=0.0)
            delta_var = post_var - pre_var
            deltas.append(delta_var)
            jrets_for_slope.append(jump_returns[j_idx])
            corr_deltas.append(delta_var if delta_var > 0 else 0.0)
            corr_rets.append(jump_returns[j_idx])

    n_events_used = len(deltas)
    deltas_arr = np.array(deltas)
    jrets_arr = np.array(jrets_for_slope)

    # mu_v: censored-at-zero mean over ALL events (zeros included).
    if n_events_used > 0:
        mu_v = float(np.clip(np.mean(np.maximum(deltas_arr, 0.0)), 0.000001, 0.001))
    else:
        mu_v = 0.000025  # Teng estimate (hourly)

    # rho_J_corr: Pearson correlation, truncated-at-zero delta (unchanged convention).
    if len(corr_deltas) > 10:
        corr_returns_arr = np.array(corr_rets)
        corr_vols_arr = np.array(corr_deltas)
        if np.std(corr_returns_arr) > 0 and np.std(corr_vols_arr) > 0:
            rho_J_corr = float(np.clip(np.corrcoef(corr_returns_arr, corr_vols_arr)[0, 1], -0.5, 0.5))
        else:
            rho_J_corr = -0.08
    else:
        rho_J_corr = -0.08

    # rho_j_slope: OLS slope of jump_return on (untruncated) delta_var.
    rho_j_slope = 0.0
    if n_events_used >= 10:
        var_dv = np.var(deltas_arr)
        if var_dv > 0:
            slope = float(np.cov(jrets_arr, deltas_arr, bias=True)[0, 1] / var_dv)
            mean_abs_jret = float(np.mean(np.abs(jrets_arr)))
            if mean_abs_jret > 0 and mu_v > 0:
                cap = 0.5 * mean_abs_jret / mu_v
                slope = float(np.clip(slope, -cap, cap))
            rho_j_slope = slope

    return mu_v, rho_J_corr, rho_j_slope, n_events_used


# ---------------------------------------------------------------------------
# Trailing-window eta_up blend (Package C / W1 -- era-conditioned jump
# calibration, plan section 2.2-REV5)
# ---------------------------------------------------------------------------

def _blend_windowed_eta_up(
    returns: np.ndarray,
    jump_mask: np.ndarray,
    window_hours: int,
    eta_up_full: float,
) -> Tuple[float, int, float]:
    """
    Blend a trailing-window UP-JUMP mean size into the full-slice eta_up
    (temp/mm_package_c_plan.md section 2.2-REV5 item 2). This REPLACES the
    REV5-SUPERSEDED all-params blend (_blend_windowed_kou, which windowed
    eta_up, eta_down, lam AND p_crash and ran a SECOND, fresh detection pass
    on the windowed slice) -- that design FAILED W3 acceptance.

    MASK-SLICE, not fresh detection: the windowed up-jump sample is read off
    the FULL-SLICE jump mask restricted to the trailing window
    (`jump_mask[-window_hours:]`), not a new bipower/MAD detection call on
    the windowed slice alone. A fresh detection on an 8760-bar slice uses a
    systematically LOWER Lee-Mykland critical value C_n (C_n scales with
    sample size n), which biases windowed jump sizes small / eta_up high
    even in a stationary era (code-review angle-C finding; confirmed by
    measurement: fresh-detection windowed eta_up ~40.3-43 vs mask-slice
    ~35.6-40.9). Sharing one detection pass (and therefore one C_n) between
    the full-slice and windowed legs removes that bias -- both legs are
    reading the SAME set of flagged jump bars, just over different spans.

    Only eta_up is windowed here. lam, p_crash, eta_down (and SVCJ) stay
    full-slice-pinned in the caller -- NOT because they are hard to window
    mechanically, but because the W3 measurement showed windowing them was
    actively harmful: the lam cut is shape-blind and moved the already-cheap
    belly and down cells; eta_down windowing thinned a down tail with NO
    measured richness (the up-heavy 2024-25 windows produced windowed
    p_crash 0.375-0.50, a thin-sample artifact, not a real down-side signal).
    The up-jump FREQUENCY era signal (windowed up-intensity 5-9/yr vs full
    ~11.5/yr) is real but unusable under the belly guard: up-jump mass at
    x=2% is the same order as at x=5%, and intensity cuts are proportional
    across x, so any lam_up cut large enough to matter at 5% moves the 2%
    belly by more than the guard allows. eta_up is the only lever that
    separates tail from belly (exp(-eta_up*x) decays faster in x, so a given
    eta_up increase cuts x=5% far more than x=2%).

    Args:
        returns: FULL array of log returns (the same array jump_mask
            indexes) -- NOT pre-sliced; the trailing window is taken inside
            this function via returns[-window_hours:].
        jump_mask: FULL-SLICE boolean jump mask (from the single detection
            pass already run on `returns` in calibrate_jumps), aligned to
            `returns`.
        window_hours: Trailing window length in bars. May be >= len(returns)
            (numpy slicing then yields the whole array -- no special-casing
            needed, matches the window_hours >= len(returns) test).
        eta_up_full: Full-slice Kou eta_up (PRE-CLIP, i.e. exactly what
            fit_kou_params returned for the full slice).

    Returns:
        (eta_up, n_window_up, window_weight). eta_up is PRE-CLIP -- the
        caller applies the existing [5,200] clip AFTER this blend, at the
        same place it always has.
    """
    wmask = jump_mask[-window_hours:]
    wret = returns[-window_hours:]
    jr_win = wret[wmask]
    up_win = jr_win[jr_win > 0]
    n_window_up = int(len(up_win))

    mean_full_up = 1.0 / eta_up_full

    if n_window_up == 0:
        # Guard: never take the mean of an empty array. A genuinely empty
        # in-window up-sample degrades to the full-slice value exactly.
        window_weight = 0.0
        mean_up_blend = mean_full_up
    else:
        window_weight = min(1.0, n_window_up / JUMP_CAL_WINDOW_TARGET_UP_JUMPS)
        mean_win_up = float(np.mean(up_win))
        mean_up_blend = window_weight * mean_win_up + (1.0 - window_weight) * mean_full_up

    eta_up_blend = 1.0 / mean_up_blend

    return eta_up_blend, n_window_up, window_weight


def calibrate_jumps(
    hourly_csv: str = "DATA/btc_hourly.csv",
    returns: Optional[np.ndarray] = None,
    detection_method: str = "bipower",
    mad_multiplier: float = 3.0,
    hours_per_year: int = 365 * 24,
    window_hours: Optional[int] = JUMP_CAL_WINDOW_HOURS,
) -> JumpCalibrationResult:
    """
    Calibrate all jump parameters from BTC hourly returns.

    Args:
        hourly_csv: Path to hourly data (used if returns not provided).
        returns: Optional pre-loaded returns array (for backtesting).
        detection_method: "bipower" (default — less vol-cluster contamination per
            FIX 2/M4) or "MAD".
        mad_multiplier: Threshold multiplier for MAD detection.
        hours_per_year: Scaling factor for annualisation.
        window_hours: Trailing-window (era-conditioned) shrinkage of the Kou
            UP-JUMP mean size (eta_up) ONLY (Package C / W1, plan section
            2.2-REV5 -- REPLACES the REV5-SUPERSEDED all-params design).
            Default JUMP_CAL_WINDOW_HOURS (8760 = 12 months of hourly bars):
            the up-jump sample is read off a MASK-SLICE of the (single,
            already-run) full-slice jump detection restricted to
            returns[-window_hours:], and credibility-blended with the
            full-slice eta_up (weight w = min(1, n_window_jumps /
            JUMP_CAL_WINDOW_TARGET_UP_JUMPS)), so calm eras get a thinner
            up-tail and wild eras get a fatter one, the same way the
            diffusion (GARCH/FIGARCH) already conditions on the current vol
            state. lam, p_crash, eta_down and the SVCJ vol-jump leg (mu_v,
            rho_J, rho_j_slope) are ALWAYS full-slice, never windowed (W3
            measurement showed windowing them moved the belly and down
            cells the wrong way -- see _blend_windowed_eta_up docstring).
            window_hours=None BYPASSES windowing entirely and reproduces
            the pre-W1 full-slice-only behavior byte-identically (this is
            the regression pin -- see tests/test_jump_calibration_window.py).

    Returns:
        JumpCalibrationResult with all estimated parameters.
    """
    # F7: validate window_hours BEFORE any use. A non-positive window_hours
    # is not a valid "no windowing" sentinel (that's window_hours=None) --
    # 0 would slice `returns[-0:]`, which numpy/Python interpret as the
    # WHOLE array (silently equivalent to no windowing at all, not "empty
    # window" as the value might suggest), and a negative value slices from
    # the front of the array (silently wrong window semantics). Reject both
    # explicitly rather than let either produce a quietly-wrong result.
    if window_hours is not None and window_hours <= 0:
        raise ValueError(
            f"window_hours must be None or a positive integer, got {window_hours}"
        )

    # Load returns
    if returns is None:
        df = pd.read_csv(hourly_csv)
        col_map = {c.lower(): c for c in df.columns}
        if 'close' not in col_map:
            raise ValueError(f"Hourly CSV missing 'close' column. Found: {list(df.columns)}")
        close_col = col_map['close']
        returns = np.log(df[close_col] / df[close_col].shift(1)).dropna().values

    n_obs = len(returns)

    # Detect jumps. FIX 3 (M2) item 8: bipower detection must request sigma_local
    # so _estimate_vol_jump_params below gets the real local diffusion variance
    # instead of always falling back to the coarser median-replacement path.
    sigma_local = None
    if detection_method == "bipower":
        jump_mask, sigma_local = detect_jumps_bipower(returns, return_sigma=True)
        jump_threshold = 0.0
    else:
        jump_mask, jump_threshold = detect_jumps_mad(returns, mad_multiplier)

    n_jumps = int(np.sum(jump_mask))

    logger.info(f"Detected {n_jumps} jumps in {n_obs} observations ({100*n_jumps/n_obs:.2f}%)")

    if n_jumps < 10:
        logger.warning("Too few jumps detected; using literature defaults from Teng (2025)")
        return JumpCalibrationResult(
            lam=25.0, p_crash=0.6, eta_up=50.0, eta_down=25.0,
            mu_v=0.000025, rho_J=-0.08, lam_v=25.0,
            n_jumps_detected=n_jumps, n_obs=n_obs,
            jump_threshold=jump_threshold, fit_converged=False,
        )

    # Fit Kou parameters
    jump_returns = returns[jump_mask]
    p_crash, eta_up, eta_down, n_jumps_total = fit_kou_params(jump_returns, method="mle")

    # Annual lambda: (n_jumps / n_obs) * hours_per_year
    lam = (n_jumps / n_obs) * hours_per_year

    # --- SVCJ Vol Jump Calibration (FIX 3 / M2: shared helper) ---
    # PINNED to the full slice ALWAYS, regardless of window_hours (Package C /
    # W1, plan section 2.2-REV5 item 2): windowed SVCJ estimates at typical
    # window jump counts are unstable / sign-flipping.
    mu_v, rho_J, rho_j_slope, n_vol_events = _estimate_vol_jump_params(
        returns, jump_mask, jump_returns, window=24, sigma_local=sigma_local,
    )

    # --- Trailing-window (era-conditioned) eta_up-only blend (Package C /
    # W1, plan section 2.2-REV5) ---
    # window_hours=None SHORT-CIRCUITS to the legacy single-pass path below,
    # BEFORE any windowed blending runs -- this is the byte-identical
    # regression pin (a w=1.0 blend of identical legs is not guaranteed
    # float-identical to the unblended value). lam, p_crash, eta_down are
    # NEVER touched here -- they keep the full-slice values computed above.
    calibration_window_hours: Optional[int] = None
    window_weight = 1.0
    n_window_jumps = 0

    if window_hours is not None:
        eta_up, n_window_jumps, window_weight = _blend_windowed_eta_up(
            returns=returns,
            jump_mask=jump_mask,
            window_hours=window_hours,
            eta_up_full=eta_up,
        )
        calibration_window_hours = window_hours
        logger.info(
            "Windowed eta_up blend: window_hours=%s, n_window_up_jumps=%d, w=%.3f",
            window_hours, n_window_jumps, window_weight,
        )

    # Handle extremes for eta parameters (post-blend, same clip as legacy)
    eta_up = np.clip(eta_up, 5.0, 200.0)
    eta_down = np.clip(eta_down, 5.0, 200.0)
    lam = np.clip(lam, 5.0, 100.0)

    logger.info(
        f"Calibrated jumps: lam={lam:.1f}/yr, p_crash={p_crash:.3f}, "
        f"eta_up={eta_up:.1f}, eta_down={eta_down:.1f}, "
        f"mu_v={mu_v:.6f}, rho_J={rho_J:.3f}, rho_j_slope={rho_j_slope:.4f} "
        f"({n_vol_events} vol-jump events)"
    )

    return JumpCalibrationResult(
        lam=lam, p_crash=p_crash, eta_up=eta_up, eta_down=eta_down,
        mu_v=mu_v, rho_J=rho_J, lam_v=lam, rho_j_slope=rho_j_slope,
        n_jumps_detected=n_jumps, n_obs=n_obs,
        jump_threshold=jump_threshold, fit_converged=True,
        calibration_window_hours=calibration_window_hours,
        window_weight=window_weight,
        n_window_jumps=n_window_jumps,
    )


# ---------------------------------------------------------------------------
# Regime-Conditional Jump Calibration
# ---------------------------------------------------------------------------

@dataclass
class RegimeJumpResult:
    """Per-regime calibrated jump parameters."""
    regime: str           # "bear", "sideways", "bull"
    lam: float
    p_crash: float
    eta_up: float
    eta_down: float
    mu_v: float
    rho_J: float
    n_jumps: int
    n_obs_in_regime: int
    jump_pct: float       # % of regime observations that are jumps
    rho_j_slope: float = 0.0  # FIX 4 (M1): regression slope, see JumpCalibrationResult


def calibrate_regime_jumps(
    hourly_csv: str = "DATA/btc_hourly.csv",
    min_jumps_per_regime: int = 30,
    detection_method: str = "MAD",
    mad_multiplier: float = 3.0,
) -> Dict[str, Optional[RegimeJumpResult]]:
    """
    Calibrate Kou double-exponential jump parameters separately per market regime.

    Requires regime_detector module. Regime labels are computed from daily returns,
    broadcast to hourly observations, and jumps are detected per regime.

    Only returns results for regimes with ≥ min_jumps_per_regime detected jumps.

    Args:
        hourly_csv: Path to BTC hourly data.
        min_jumps_per_regime: Minimum jumps per regime to report results.
        detection_method: "MAD" or "bipower".
        mad_multiplier: MAD threshold multiplier.

    Returns:
        Dict mapping regime_name -> RegimeJumpResult or None if insufficient jumps.
        e.g. {"bear": RegimeJumpResult(...), "sideways": RegimeJumpResult(...), "bull": None}
    """
    from core.pricing.regime_detector import RegimeDetector

    # Load hourly returns
    df = pd.read_csv(hourly_csv)
    col_map = {c.lower(): c for c in df.columns}
    if 'close' not in col_map:
        raise ValueError(f"Hourly CSV missing 'close' column. Found: {list(df.columns)}")
    close_col = col_map['close']
    hourly_returns = np.log(df[close_col] / df[close_col].shift(1)).dropna().values
    n_hourly = len(hourly_returns)

    # Resample to daily for regime detection
    # Use close prices resampled: last available close per day
    if 'timestamp' in col_map or 'date' in col_map:
        ts_col = 'timestamp' if 'timestamp' in col_map else 'date'
        df['_ts'] = pd.to_datetime(df[ts_col])
    else:
        # Assume evenly spaced hourly from some start date
        df['_ts'] = pd.date_range(
            start=datetime.now(timezone.utc) - pd.Timedelta(hours=n_hourly),
            periods=len(df),
            freq='h',
        )

    df['_close'] = df[close_col]
    daily = df.set_index('_ts')['_close'].resample('D').last().dropna()
    daily_returns = np.log(daily / daily.shift(1)).dropna().values

    # Detect regimes
    detector = RegimeDetector()
    result = detector.fit(daily_returns, force=True)
    if result is None or detector._model is None or detector._labels is None:
        logger.warning("HMM regime detection failed — cannot calibrate regime jumps")
        return {"bear": None, "sideways": None, "bull": None}

    # Get per-observation hidden state assignments
    # decode returns (log_probability, state_sequence)
    _, hidden_states = detector._model.decode(daily_returns.reshape(-1, 1))

    # Map HMM state index -> regime name via state_order
    # state_order = [bear_idx, sideways_idx, bull_idx] → position = regime
    # NOTE (T6/H3): this is pure RANK mapping and does NOT apply the
    # threshold-aware demotion in RegimeDetector._label_states (e.g. a
    # low-but-still-positive-drift state in a strong bull market keeps being
    # called "bear" here). `detector._labels.state_labels` is the
    # authoritative, demotion-aware per-state mapping used by fit()/
    # predict_weights()/get_regime_variance_scales(); this function is not on
    # the live pipeline path today (regime_calibrated=None everywhere in
    # build_regime_jump_params callers), so it is left as rank-only rather
    # than risk changing behavior for a currently-unused code path.
    state_order = detector._labels.state_order
    hmm_state_to_regime = {}
    for regime_pos, hmm_state in enumerate(state_order):
        regime_names = ["bear", "sideways", "bull"]
        hmm_state_to_regime[hmm_state] = regime_names[regime_pos]

    daily_regime_strs = {}
    fallback_count = 0
    for i, hmm_state in enumerate(hidden_states):
        dt_idx = daily.index[1 + i]  # daily_returns[0] = return from daily[0]->daily[1]
        regime = hmm_state_to_regime.get(int(hmm_state))
        if regime is None:
            fallback_count += 1
            regime = "sideways"
        daily_regime_strs[dt_idx.date()] = regime
    if fallback_count > 0:
        logger.warning(
            "HMM state fallback: %d/%d states unrecognized, defaulted to sideways",
            fallback_count, len(hidden_states),
        )

    # Broadcast daily regime labels to hourly observations
    hourly_regimes = np.full(n_hourly, "sideways", dtype=object)
    df['_date'] = df['_ts'].dt.date
    for i in range(n_hourly):
        d = df['_date'].iloc[i]
        if d in daily_regime_strs:
            hourly_regimes[i] = daily_regime_strs[d]

    # Detect jumps per regime
    results: Dict[str, Optional[RegimeJumpResult]] = {}

    for regime in ["bear", "sideways", "bull"]:
        regime_mask = hourly_regimes == regime
        regime_returns = hourly_returns[regime_mask]
        n_regime = int(np.sum(regime_mask))

        if n_regime < 100:
            logger.info(
                "Regime '%s': only %d observations — skipping regime-specific calibration",
                regime, n_regime,
            )
            results[regime] = None
            continue

        # Detect jumps within this regime. FIX 3 (M2): request sigma_local on the
        # bipower path so _estimate_vol_jump_params gets the real local diffusion
        # variance (same treatment as calibrate_jumps).
        regime_sigma_local = None
        if detection_method == "bipower":
            jump_mask, regime_sigma_local = detect_jumps_bipower(regime_returns, return_sigma=True)
        else:
            jump_mask, threshold = detect_jumps_mad(regime_returns, mad_multiplier)

        n_jumps = int(np.sum(jump_mask))
        logger.info(
            "Regime '%s': %d jumps in %d obs (%.2f%%)",
            regime, n_jumps, n_regime, 100 * n_jumps / n_regime,
        )

        if n_jumps < min_jumps_per_regime:
            logger.warning(
                "Regime '%s': only %d jumps (need ≥%d) — "
                "using base parameters for this regime",
                regime, n_jumps, min_jumps_per_regime,
            )
            results[regime] = None
            continue

        # Fit Kou parameters on regime-specific jumps
        jump_returns = regime_returns[jump_mask]
        p_crash, eta_up, eta_down, _ = fit_kou_params(jump_returns, method="mle")

        # Annual lambda: (n_jumps_in_regime / n_regime_hours) * hours_per_year
        hours_per_year = 365 * 24
        lam = (n_jumps / n_regime) * hours_per_year

        # Estimate SVCJ vol-jump params around regime-specific jumps (FIX 3 / M2:
        # shared helper -- same jump-square replacement + censored mean as the
        # base calibration).
        mu_v, rho_J, rho_j_slope, n_vol_events = _estimate_vol_jump_params(
            regime_returns, jump_mask, jump_returns, window=24,
            sigma_local=regime_sigma_local,
        )

        # Clamp extremes
        eta_up = np.clip(eta_up, 5.0, 200.0)
        eta_down = np.clip(eta_down, 5.0, 200.0)
        lam = np.clip(lam, 5.0, 100.0)

        results[regime] = RegimeJumpResult(
            regime=regime,
            lam=lam,
            p_crash=p_crash,
            eta_up=eta_up,
            eta_down=eta_down,
            mu_v=mu_v,
            rho_J=rho_J,
            rho_j_slope=rho_j_slope,
            n_jumps=n_jumps,
            n_obs_in_regime=n_regime,
            jump_pct=100 * n_jumps / n_regime,
        )

        logger.info(
            "Regime '%s' calibrated: lam=%.1f/yr, p_crash=%.3f, "
            "eta_up=%.1f, eta_down=%.1f, mu_v=%.6f, rho_J=%.3f, "
            "rho_j_slope=%.4f (%d jumps, %d vol-jump events)",
            regime, lam, p_crash, eta_up, eta_down, mu_v, rho_J, rho_j_slope,
            n_jumps, n_vol_events,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Calibrate jump parameters from BTC hourly data")
    parser.add_argument("--input", default="DATA/btc_hourly.csv", help="Path to hourly BTC data")
    parser.add_argument("--method", default="bipower", choices=["MAD", "bipower"],
                       help="Jump detection method (default: bipower)")
    parser.add_argument("--mad-mult", type=float, default=3.0,
                       help="MAD threshold multiplier")
    args = parser.parse_args()

    result = calibrate_jumps(
        hourly_csv=args.input,
        detection_method=args.method,
        mad_multiplier=args.mad_mult,
    )

    print("\n=== Jump Calibration Results ===")
    print(f"Observations:       {result.n_obs}")
    print(f"Jumps detected:     {result.n_jumps_detected}")
    print(f"Jump threshold:     {result.jump_threshold:.6f}")
    print(f"Annual lambda:      {result.lam:.2f}")
    print(f"Crash probability:  {result.p_crash:.3f}")
    print(f"Eta up (1/mean):    {result.eta_up:.2f}")
    print(f"Eta down (1/mean):  {result.eta_down:.2f}")
    print(f"Vol jump mean μ_v:  {result.mu_v:.8f}")
    print(f"Return-vol corr ρ_J:{result.rho_J:.4f}")
    print(f"Fit converged:      {result.fit_converged}")
    print(f"Literature ρ_J:     {result.TENG_RHO_J}")
