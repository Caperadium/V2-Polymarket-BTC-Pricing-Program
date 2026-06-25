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

    # Diagnostics
    n_jumps_detected: int = 0
    n_obs: int = 0
    jump_threshold: float = 0.0
    detection_method: str = "MAD"
    fit_converged: bool = True

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
) -> np.ndarray:
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

    Args:
        returns: Array of log returns.
        significance: Test level β (default 0.01).
        window: Trailing window K (bars) for the local bipower vol estimate.

    Returns:
        Boolean jump_mask aligned to *returns*.
    """
    n = len(returns)
    if n < window + 2:
        return np.zeros(n, dtype=bool)

    abs_ret = np.abs(returns)
    # Local jump-robust (bipower) variance: trailing mean of |r_{t-1}|*|r_t|,
    # scaled by π/2. Shift(1) so σ̂_t excludes the contemporaneous return (a jump
    # at t must not inflate its own threshold).
    s = pd.Series(abs_ret)
    bpv_local = (np.pi / 2.0) * (s.shift(1) * s).rolling(
        window, min_periods=max(10, window // 3)
    ).mean()
    sigma_local = np.sqrt(bpv_local.to_numpy())
    # Backfill the leading NaNs with the median local vol so early bars are testable
    # against a sane scale rather than dropped.
    med = np.nanmedian(sigma_local)
    sigma_local = np.where(np.isfinite(sigma_local) & (sigma_local > 0), sigma_local, med)
    if not np.isfinite(med) or med <= 0:
        return np.zeros(n, dtype=bool)

    L = abs_ret / sigma_local

    # Gumbel-based critical value (scales with n).
    c = np.sqrt(2.0 / np.pi)
    sqrt_2logn = np.sqrt(2.0 * np.log(n))
    C_n = sqrt_2logn / c - (np.log(np.pi) + np.log(np.log(n))) / (2.0 * c * sqrt_2logn)
    S_n = 1.0 / (c * sqrt_2logn)
    gumbel_q = -np.log(-np.log(1.0 - significance))

    jump_mask = (L - C_n) / S_n > gumbel_q
    return np.asarray(jump_mask, dtype=bool)


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
        (p_crash, eta_up, eta_down, annual_lambda)
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


def calibrate_jumps(
    hourly_csv: str = "DATA/btc_hourly.csv",
    returns: Optional[np.ndarray] = None,
    detection_method: str = "bipower",
    mad_multiplier: float = 3.0,
    hours_per_year: int = 365 * 24,
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

    Returns:
        JumpCalibrationResult with all estimated parameters.
    """
    # Load returns
    if returns is None:
        df = pd.read_csv(hourly_csv)
        col_map = {c.lower(): c for c in df.columns}
        if 'close' not in col_map:
            raise ValueError(f"Hourly CSV missing 'close' column. Found: {list(df.columns)}")
        close_col = col_map['close']
        returns = np.log(df[close_col] / df[close_col].shift(1)).dropna().values

    n_obs = len(returns)

    # Detect jumps
    if detection_method == "bipower":
        jump_mask = detect_jumps_bipower(returns)
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
    p_crash, eta_up, eta_down, n_jumps_for_lambda = fit_kou_params(jump_returns, method="mle")

    # Annual lambda: (n_jumps / n_obs) * hours_per_year
    lam = (n_jumps / n_obs) * hours_per_year

    # --- SVCJ Vol Jump Calibration ---
    # Estimate realized variance changes around jump events
    # For each detected jump day, compute variance before/after jump
    squared_returns = returns ** 2

    # Rolling variance (1h = 1 observation)
    window = 24  # 24h = 1 day
    rolling_var = pd.Series(squared_returns).rolling(window, min_periods=4).mean().values

    # Vol jump = difference in variance at jump times vs pre-jump
    vol_changes = []
    vol_jump_corr_data = []
    jump_indices = np.where(jump_mask)[0]
    n_jumps = len(jump_indices)

    for j_idx, full_idx in enumerate(jump_indices):
        if full_idx >= 2 and full_idx < len(rolling_var) - 1 and j_idx < len(jump_returns):
            pre_var = np.nan_to_num(rolling_var[max(0, full_idx - 2)], nan=0.0)
            post_var = np.nan_to_num(rolling_var[min(len(rolling_var) - 1, full_idx + 2)], nan=0.0)
            delta_var = max(0.0, post_var - pre_var)
            vol_changes.append(delta_var)
            vol_jump_corr_data.append((jump_returns[j_idx], delta_var if delta_var > 0 else 0))

    # Estimate mu_v: mean of positive vol changes at jump times (hourly variance units)
    vol_changes_arr = np.array(vol_changes)
    positive_vol_changes = vol_changes_arr[vol_changes_arr > 0]

    if len(positive_vol_changes) > 5:
        # Fit exponential distribution to positive vol changes
        mu_v = np.mean(positive_vol_changes)
        # Cap at reasonable values
        mu_v = np.clip(mu_v, 0.000001, 0.001)
    else:
        mu_v = 0.000025  # Teng estimate (hourly)

    # Estimate rho_J: correlation between return jumps and vol jumps
    if len(vol_jump_corr_data) > 10:
        corr_returns = np.array([v[0] for v in vol_jump_corr_data])
        corr_vols = np.array([v[1] for v in vol_jump_corr_data])
        if np.std(corr_returns) > 0 and np.std(corr_vols) > 0:
            rho_J = np.corrcoef(corr_returns, corr_vols)[0, 1]
            # Clamp to reasonable range
            rho_J = np.clip(rho_J, -0.5, 0.5)
        else:
            rho_J = -0.08  # Teng estimate
    else:
        rho_J = -0.08

    # Handle extremes for eta parameters
    eta_up = np.clip(eta_up, 5.0, 200.0)
    eta_down = np.clip(eta_down, 5.0, 200.0)
    lam = np.clip(lam, 5.0, 100.0)

    logger.info(
        f"Calibrated jumps: lam={lam:.1f}/yr, p_crash={p_crash:.3f}, "
        f"eta_up={eta_up:.1f}, eta_down={eta_down:.1f}, "
        f"mu_v={mu_v:.6f}, rho_J={rho_J:.3f}"
    )

    return JumpCalibrationResult(
        lam=lam, p_crash=p_crash, eta_up=eta_up, eta_down=eta_down,
        mu_v=mu_v, rho_J=rho_J, lam_v=lam,
        n_jumps_detected=n_jumps, n_obs=n_obs,
        jump_threshold=jump_threshold, fit_converged=True,
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

        # Detect jumps within this regime
        if detection_method == "bipower":
            jump_mask = detect_jumps_bipower(regime_returns)
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

        # Estimate mu_v: mean positive vol change around regime-specific jumps
        squared_returns = regime_returns ** 2
        window = 24
        rolling_var = pd.Series(squared_returns).rolling(window, min_periods=4).mean().values
        jump_indices = np.where(jump_mask)[0]

        vol_changes = []
        vol_jump_corr_data = []
        for j_idx, full_idx in enumerate(jump_indices):
            if full_idx >= 2 and full_idx < len(rolling_var) - 1 and j_idx < len(jump_returns):
                pre_var = np.nan_to_num(rolling_var[max(0, full_idx - 2)], nan=0.0)
                post_var = np.nan_to_num(rolling_var[min(len(rolling_var) - 1, full_idx + 2)], nan=0.0)
                delta_var = max(0.0, post_var - pre_var)
                vol_changes.append(delta_var)
                vol_jump_corr_data.append((jump_returns[j_idx], delta_var if delta_var > 0 else 0))

        vol_changes_arr = np.array(vol_changes)
        positive_vol_changes = vol_changes_arr[vol_changes_arr > 0]
        if len(positive_vol_changes) > 5:
            mu_v = np.clip(np.mean(positive_vol_changes), 0.000001, 0.001)
        else:
            mu_v = 0.000025

        if len(vol_jump_corr_data) > 10:
            corr_returns = np.array([v[0] for v in vol_jump_corr_data])
            corr_vols = np.array([v[1] for v in vol_jump_corr_data])
            if np.std(corr_returns) > 0 and np.std(corr_vols) > 0:
                rho_J = np.clip(np.corrcoef(corr_returns, corr_vols)[0, 1], -0.5, 0.5)
            else:
                rho_J = -0.08
        else:
            rho_J = -0.08

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
            n_jumps=n_jumps,
            n_obs_in_regime=n_regime,
            jump_pct=100 * n_jumps / n_regime,
        )

        logger.info(
            "Regime '%s' calibrated: lam=%.1f/yr, p_crash=%.3f, "
            "eta_up=%.1f, eta_down=%.1f, mu_v=%.6f, rho_J=%.3f (%d jumps)",
            regime, lam, p_crash, eta_up, eta_down, mu_v, rho_J, n_jumps,
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
