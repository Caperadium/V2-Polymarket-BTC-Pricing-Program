#!/usr/bin/env python
"""
fit_probability_curves.py

Post-process a batch_summary.csv from the BTC pricing engine:

- For each expiry (grouped by T_days and optionally expiry_date),
  fit two logistic curves:

    p_model_fit(K) ~ logistic_model(K)
    p_rn_fit(K)    ~ logistic_rn(K)

- Append fitted probabilities & edges to each row.
- Save (per input file) to fitted_batch_results/<input_stem>/:
    - batch_with_fits.csv    (per-contract, augmented)
    - curve_params.csv       (one row per expiry with curve params)
  and invoke plot_batch_curves.py inside the same folder.

Dependencies: pandas, numpy, scipy
"""

import argparse
import logging
import subprocess
import sys
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
from scipy.special import expit, logit

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None

def calibrate_logit_shift(
    p_model: np.ndarray,
    outcomes: np.ndarray,
) -> dict:
    """
    Calibrate the logit shift parameter B via MLE on binary outcomes.

    Model: p_cal = sigmoid(logit(p_model) + B)
    Likelihood: Π p_cal_i^y_i × (1-p_cal_i)^(1-y_i)

    This is Platt scaling with a single shift parameter (no slope).
    Estimates B via MLE and reports a 95% likelihood-ratio confidence interval.

    Args:
        p_model: Array of model probabilities (values in (0,1)).
        outcomes: Array of binary outcomes (0 or 1), same length as p_model.

    Returns:
        Dict with keys: B_fitted, B_ci_lower, B_ci_upper, n_obs, converged.
        Returns None if insufficient data (need at least 10 observations with
        both positive and negative outcomes).
    """
    p_model = np.asarray(p_model, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    mask = np.isfinite(p_model) & np.isfinite(outcomes)
    p_model = p_model[mask]
    outcomes = outcomes[mask]

    n = len(p_model)
    n_pos = int(np.sum(outcomes))
    n_neg = n - n_pos

    if n < 10 or n_pos < 2 or n_neg < 2:
        logger.warning(
            "Insufficient outcome data for logit shift calibration "
            "(need ≥10 obs, ≥2 positive, ≥2 negative). Got n=%d, pos=%d.",
            n, n_pos,
        )
        return None

    eps = 1e-12
    p_clipped = np.clip(p_model, eps, 1 - eps)
    x = logit(p_clipped)  # offset (known, not fitted)

    # Negative log-likelihood: -Σ [y × log(p_cal) + (1-y) × log(1-p_cal)]
    def neg_loglik(B: float) -> float:
        p_cal = expit(x + B)
        p_cal = np.clip(p_cal, eps, 1 - eps)
        return -np.sum(outcomes * np.log(p_cal) + (1 - outcomes) * np.log(1 - p_cal))

    result = minimize_scalar(neg_loglik, bounds=(-3.0, 3.0), method="bounded")
    B_fitted = result.x
    converged = result.success
    ll_fitted = -result.fun

    # 95% CI via likelihood ratio: find B where 2×(ll_fitted - ll(B)) = 3.841 (χ²₁,0.05)
    chi2_crit = 3.841

    def _find_bound(lower: bool) -> float:
        """Find B where LR stat hits chi2_crit. Search outward from fitted B,
        then bisect between the last inside point and the first outside point."""
        step = -0.5 if lower else 0.5
        inside = B_fitted
        B = B_fitted + step
        crossed = False
        for _ in range(50):
            ll = -neg_loglik(B)
            if 2 * (ll_fitted - ll) >= chi2_crit:
                crossed = True
                break
            inside = B
            B += step
            if abs(B) > 5.0:
                break
        if not crossed:
            return B  # Bound not found within search range
        outside = B
        for _ in range(30):
            mid = 0.5 * (inside + outside)
            ll = -neg_loglik(mid)
            if 2 * (ll_fitted - ll) >= chi2_crit:
                outside = mid
            else:
                inside = mid
            if abs(outside - inside) < 1e-4:
                break
        return outside

    B_ci_lower = _find_bound(lower=True)
    B_ci_upper = _find_bound(lower=False)

    logger.info(
        "Logit shift calibrated: B=%.4f [95%% CI: %.4f, %.4f] from %d observations.",
        B_fitted, B_ci_lower, B_ci_upper, n,
    )

    return {
        "B_fitted": B_fitted,
        "B_ci_lower": B_ci_lower,
        "B_ci_upper": B_ci_upper,
        "n_obs": n,
        "converged": converged,
    }


# ---------- Outcome-based recalibration (FIX 7 / M2) ----------

# Persisted shift table: one row per DTE bucket. Anchored to the project root
# (core/pricing/ -> parents[2]) so writer (fit_calibration) and reader
# (load_calibration_shift) resolve to the SAME file regardless of CWD — Streamlit
# may launch from app/, which would otherwise split the read/write locations.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_SHIFT_PATH = str(_PROJECT_ROOT / "DATA" / "calibration_shift.csv")
# Minimum resolved observations per bucket before a fitted shift is trusted;
# below this the bucket shift falls back to 0 (no shift). Plan risk note.
CALIBRATION_MIN_OBS = 200
# DTE bucket edges (days). Buckets: [0,2], (2,7], (7,30], (30,inf).
_DTE_BUCKET_EDGES = [2.0, 7.0, 30.0]
_DTE_BUCKET_LABELS = ["0-2", "2-7", "7-30", "30+"]


def dte_bucket(dte_days: float) -> str:
    """Map days-to-expiry to a calibration bucket label. NaN/negative → '0-2'."""
    try:
        d = float(dte_days)
    except (TypeError, ValueError):
        return _DTE_BUCKET_LABELS[0]
    if not np.isfinite(d) or d <= _DTE_BUCKET_EDGES[0]:
        return _DTE_BUCKET_LABELS[0]
    for i, edge in enumerate(_DTE_BUCKET_EDGES[1:], start=1):
        if d <= edge:
            return _DTE_BUCKET_LABELS[i]
    return _DTE_BUCKET_LABELS[-1]


def fit_calibration(
    all_priced_df: pd.DataFrame,
    prob_col: str = "model_prob_used",
    outcome_col: str = "outcome_yes",
    dte_col: str = "dte_days",
    time_col: str = "snapshot_time",
    train_frac: float = 0.7,
    output_path: str = CALIBRATION_SHIFT_PATH,
    min_obs: int = CALIBRATION_MIN_OBS,
) -> dict:
    """
    Fit a per-DTE-bucket logit shift on backtest history and persist it (FIX 7 A).

    LEAK GUARD (mandatory): the shift B is fit ONLY on a *training span* of the
    backtest (the earliest ``train_frac`` of snapshots by ``time_col``). It must be
    APPLIED to the later (holdout) span — never fit and scored on the same outcomes,
    which is in-sample and inflates apparent calibration. This function emits the
    table; application happens in ``process_batch`` (Part B) and the holdout
    reliability/Brier comparison is a separate diagnostic.

    Args:
        all_priced_df: BacktestEngine all-priced contracts with resolved outcomes.
        prob_col: Model probability column (default 'model_prob_used').
        outcome_col: Binary realized outcome column (default 'outcome_yes').
        dte_col: Days-to-expiry column for bucketing.
        time_col: Snapshot timestamp column for the walk-forward split.
        train_frac: Fraction of the (time-sorted) span used to FIT B.
        output_path: Where to persist the shift table CSV.
        min_obs: Minimum resolved obs per bucket to trust a fitted B (else B=0).

    Returns:
        Dict mapping bucket label -> {B_fitted, B_ci_lower, B_ci_upper, n_obs,
        n_train, applied (bool), fit_date}. Also writes the CSV.
    """
    out: dict = {}
    if all_priced_df is None or all_priced_df.empty:
        logger.info("fit_calibration: empty all_priced_df — no shift fit.")
        return out

    df = all_priced_df.copy()
    needed = [prob_col, outcome_col, dte_col]
    if any(c not in df.columns for c in needed):
        logger.info(
            "fit_calibration: missing columns %s — skipping.",
            [c for c in needed if c not in df.columns],
        )
        return out

    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df = df[np.isfinite(df[prob_col]) & np.isfinite(df[outcome_col])]
    if df.empty:
        logger.info("fit_calibration: no finite (prob, outcome) rows.")
        return out

    # ---- Walk-forward split by time (leak guard) ----
    if time_col in df.columns:
        df["_t"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.sort_values("_t")
        cutoff = df["_t"].quantile(train_frac)
        train = df[df["_t"] <= cutoff]
    else:
        # No timestamp: fall back to row-order split (documented; weaker guard).
        logger.warning(
            "fit_calibration: no '%s' column — using row-order train split "
            "(weaker leak guard).", time_col,
        )
        n_train = int(len(df) * train_frac)
        train = df.iloc[:n_train]

    df["_bucket"] = df[dte_col].apply(dte_bucket)
    train = train.copy()
    train["_bucket"] = train[dte_col].apply(dte_bucket)

    fit_date = datetime.now(timezone.utc).isoformat()
    rows = []
    for bucket in _DTE_BUCKET_LABELS:
        g = train[train["_bucket"] == bucket]
        n_obs = len(g)
        B, lo, hi, applied = 0.0, 0.0, 0.0, False
        if n_obs >= min_obs:
            shift = calibrate_logit_shift(g[prob_col].values, g[outcome_col].values)
            if shift is not None:
                B = float(shift["B_fitted"])
                lo = float(shift["B_ci_lower"])
                hi = float(shift["B_ci_upper"])
                applied = True
        else:
            logger.info(
                "fit_calibration bucket %s: only %d train obs (<%d) — shift=0.",
                bucket, n_obs, min_obs,
            )
        entry = {
            "bucket": bucket,
            "B_fitted": B,
            "B_ci_lower": lo,
            "B_ci_upper": hi,
            "n_obs": n_obs,
            "applied": applied,
            "fit_date": fit_date,
            "train_frac": train_frac,
        }
        out[bucket] = entry
        rows.append(entry)

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info("fit_calibration: wrote shift table to %s", output_path)
    except Exception as exc:
        logger.warning("fit_calibration: could not write %s (%s)", output_path, exc)

    return out


def load_calibration_shift(path: str = CALIBRATION_SHIFT_PATH) -> dict:
    """Load the persisted per-bucket logit shift table. Returns {} if absent."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        tbl = pd.read_csv(p)
    except Exception:
        return {}
    shifts = {}
    for _, r in tbl.iterrows():
        # Only honor buckets whose fit was actually trusted (applied=True).
        if bool(r.get("applied", False)):
            shifts[str(r["bucket"])] = float(r["B_fitted"])
    return shifts


def apply_calibration_shift(p_model: np.ndarray, B: float) -> np.ndarray:
    """p_cal = sigmoid(logit(clip(p_model)) + B). Identity when B==0."""
    if not B:
        return np.asarray(p_model, dtype=float)
    eps = 1e-6
    p = np.clip(np.asarray(p_model, dtype=float), eps, 1 - eps)
    return expit(logit(p) + B)


def calibration_holdout_report(
    all_priced_df: pd.DataFrame,
    prob_col: str = "model_prob_used",
    outcome_col: str = "outcome_yes",
    dte_col: str = "dte_days",
    time_col: str = "snapshot_time",
    train_frac: float = 0.7,
    n_bins: int = 10,
    min_holdout: int = 20,
) -> Optional[dict]:
    """
    Walk-forward holdout calibration report (FIX 7 / M2 acceptance #6).

    Fits per-DTE-bucket logit shifts on the EARLIEST ``train_frac`` of snapshots and
    scores RAW vs CALIBRATED model probabilities on the LATER holdout span — the fit
    never sees the holdout outcomes it is scored against. Returns the reliability-
    diagram bins + Brier + ECE for both, so a UI can render the diagram and confirm
    recalibrated ≤ raw. Returns None if the holdout is too small.

    Returns:
        {raw_brier, cal_brier, raw_ece, cal_ece, raw_bins (DataFrame),
         cal_bins (DataFrame), n_holdout, n_train, per_bucket (list[dict])} or None.
    """
    from core.validation.calibration_metrics import (
        brier_score, ece_score, reliability_bins,
    )

    if all_priced_df is None or all_priced_df.empty:
        return None
    need = [prob_col, outcome_col, dte_col]
    if any(c not in all_priced_df.columns for c in need):
        return None

    df = all_priced_df.copy()
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df = df[np.isfinite(df[prob_col]) & np.isfinite(df[outcome_col])]
    if df.empty:
        return None

    # Walk-forward split by time (leak guard); fall back to row order if no time col.
    if time_col in df.columns:
        df["_t"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.sort_values("_t")
        cutoff = df["_t"].quantile(train_frac)
        train = df[df["_t"] <= cutoff]
        holdout = df[df["_t"] > cutoff]
    else:
        n_train = int(len(df) * train_frac)
        train = df.iloc[:n_train]
        holdout = df.iloc[n_train:]

    if len(holdout) < min_holdout:
        return None

    df["_bucket"] = df[dte_col].apply(dte_bucket)
    train = train.copy(); train["_bucket"] = train[dte_col].apply(dte_bucket)
    holdout = holdout.copy(); holdout["_bucket"] = holdout[dte_col].apply(dte_bucket)

    # Fit B per bucket on TRAIN only.
    per_bucket = []
    bucket_B: dict = {}
    for bucket in _DTE_BUCKET_LABELS:
        g = train[train["_bucket"] == bucket]
        B = 0.0
        shift = (
            calibrate_logit_shift(g[prob_col].values, g[outcome_col].values)
            if len(g) >= 10 else None
        )
        if shift is not None:
            B = float(shift["B_fitted"])
        bucket_B[bucket] = B
        per_bucket.append({"bucket": bucket, "B_fitted": B, "n_train": int(len(g))})

    # Apply to HOLDOUT.
    raw = holdout[prob_col].to_numpy(dtype=float)
    y = holdout[outcome_col].to_numpy(dtype=float)
    B_arr = holdout["_bucket"].map(bucket_B).fillna(0.0).to_numpy(dtype=float)
    eps = 1e-6
    cal = expit(logit(np.clip(raw, eps, 1 - eps)) + B_arr)

    return {
        "raw_brier": brier_score(raw, y),
        "cal_brier": brier_score(cal, y),
        "raw_ece": ece_score(raw, y, n_bins),
        "cal_ece": ece_score(cal, y, n_bins),
        "raw_bins": reliability_bins(raw, y, n_bins),
        "cal_bins": reliability_bins(cal, y, n_bins),
        "n_holdout": int(len(holdout)),
        "n_train": int(len(train)),
        "per_bucket": per_bucket,
    }


# ---------- Logistic model helpers ----------

def logistic_raw(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Basic logistic in strike space:

        p(K) = 1 / (1 + exp(a * (K - b)))

    For a > 0 this is strictly decreasing in K.
    """
    z = a * (x - b)
    return expit(-z)


def logistic_param(x: np.ndarray, log_a: float, b: float) -> np.ndarray:
    """
    Reparametrize with log_a to force a > 0:

        a = exp(log_a)

    This keeps the curve monotone decreasing in K.
    """
    a = np.exp(log_a)
    return logistic_raw(x, a, b)


@dataclass
class CurveFitResult:
    log_a: float
    b: float
    success: bool
    n_points: int


def fit_logistic_to_points(
    strikes: np.ndarray,
    probs: np.ndarray,
    K_scale: float = 1000.0,
) -> Optional[CurveFitResult]:
    """
    Fit logistic_param to (K, p) points.

    - Rescales K by K_scale to improve numerical stability.
    - Returns None if not enough points or fit fails.

    KNOWN MODEL RISK (FIX 10 / L4): this is a 2-parameter SYMMETRIC logistic and
    cannot represent the skewed wings of an SVCJ/skewed-t risk-neutral density, and
    `curve_fit` minimizes unweighted SSE on probabilities (over-weighting the
    saturated 0/1 tails relative to the informative mid-strikes). It is a denoiser,
    not a full skewed RND. If diagnostics show systematic wing mispricing, refit in
    logit space or weight points by MC variance p(1-p)/n_sims. Left as documented
    model risk until diagnostics justify the change.
    """
    mask = np.isfinite(strikes) & np.isfinite(probs)
    strikes = strikes[mask]
    probs = probs[mask]

    if len(strikes) < 4:
        # Too few points for a meaningful fit.
        return None

    # Rescale strikes
    x = strikes / K_scale
    y = probs

    # Clip y a bit away from 0/1 to avoid logit infinities
    eps = 1e-4
    y = np.clip(y, eps, 1.0 - eps)

    # Initial guesses:
    # - log_a: start with slope ~1.0 in rescaled space
    # - b: center around mid-strike
    x_mid = np.median(x)
    p0 = [0.0, x_mid]  # log_a=0 -> a=1, b=x_mid

    try:
        popt, _ = curve_fit(
            logistic_param,
            x,
            y,
            p0=p0,
            maxfev=10000,
        )
        log_a, b = popt
        return CurveFitResult(log_a=log_a, b=b, success=True, n_points=len(x))
    except Exception:
        return None


def eval_logistic_fit(
    strikes: np.ndarray,
    fit: CurveFitResult,
    K_scale: float = 1000.0,
) -> np.ndarray:
    """
    Evaluate a fitted logistic curve at given strikes.
    """
    x = strikes / K_scale
    return logistic_param(x, fit.log_a, fit.b)


def _infer_batch_timestamp(input_csv: str) -> str:
    """Return ISO timestamp from batch folder name or fallback to now."""
    match = re.search(r"batch_(\d{8}_\d{6})", input_csv)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").isoformat()
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat()


def _infer_pricing_date(input_csv: str, tz_name: str = "America/Vancouver") -> str:
    """Infer local YYYY-MM-DD pricing date from batch folder name or fallback to today."""
    match = re.search(r"batch_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", input_csv)
    if match:
        year, month, day, hh, mm, ss = map(int, match.groups())
        try:
            utc_dt = datetime(year, month, day, hh, mm, ss, tzinfo=timezone.utc)
            if ZoneInfo is not None:
                tz = ZoneInfo(tz_name)
                local_dt = utc_dt.astimezone(tz)
            else:
                local_dt = utc_dt
            return local_dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    # Fallback: use today's date in target timezone (or UTC if zoneinfo missing)
    if ZoneInfo is not None:
        tz = ZoneInfo(tz_name)
        return datetime.now(tz).strftime("%Y-%m-%d")
    return datetime.utcnow().strftime("%Y-%m-%d")


# ---------- Column normalization ----------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to match expected format.
    Maps existing batch_results format to the expected format.
    """
    # Column name mappings (existing -> expected)
    col_map = {
        "Strike": "strike",
        "Polymarket_Price": "market_price",
        "Model_Prob": "p_real_mc",
        "model_probability": "p_real_mc",  # From prob_backrunner_engine output
        "Date": "expiry_date",
        "Edge": "edge",
        "Expiry_ET": "expiry_et",
    }
    
    # Apply mappings
    for old_name, new_name in col_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Calculate T_days from expiry_date if not present
    if "T_days" not in df.columns and "t_days" not in df.columns:
        if "expiry_date" in df.columns:
            try:
                today = datetime.now(timezone.utc).date()
                df["T_days"] = pd.to_datetime(df["expiry_date"]).dt.date.apply(
                    lambda d: max(0, (d - today).days)
                )
            except Exception:
                # Fallback: assign sequential T_days based on unique expiry dates
                unique_dates = df["expiry_date"].unique()
                date_to_t = {d: i for i, d in enumerate(sorted(unique_dates))}
                df["T_days"] = df["expiry_date"].map(date_to_t)
    
    return df


# ---------- Main pipeline ----------

def process_batch(
    input_csv: str,
    output_batch_csv: str,
    output_curve_params_csv: str,
    use_rn_prob: bool = False,
) -> None:
    """Fit logistic curves per expiry bucket and augment/save batch + curve CSVs."""
    df = pd.read_csv(input_csv)
    
    # Normalize column names to expected format
    df = normalize_columns(df)

    if "T_days" in df.columns:
        t_col = "T_days"
    elif "t_days" in df.columns:
        t_col = "t_days"
    else:
        raise ValueError("Input CSV missing T_days / t_days column and could not compute from expiry_date.")

    batch_timestamp = _infer_batch_timestamp(input_csv)
    df["batch_timestamp"] = batch_timestamp
    if "pricing_date" not in df.columns:
        df["pricing_date"] = _infer_pricing_date(input_csv)

    required_cols = ["strike", t_col, "market_price", "p_real_mc"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns after normalization: {missing}")

    if use_rn_prob and "risk_neutral_prob" not in df.columns:
        raise ValueError("use_rn_prob=True but 'risk_neutral_prob' column not found.")

    # Bucket T_days to avoid float-fragmentation when no explicit expiry date exists.
    df["T_bucket"] = df[t_col].astype(float).round(3)
    if "expiry_date" in df.columns:
        group_cols = ["expiry_date"]
        use_expiry_group = True
    else:
        group_cols = ["T_bucket"]
        use_expiry_group = False

    curve_rows = []

    # Pre-allocate columns with NaN
    df["p_model_fit"] = np.nan
    # FIX 9 (M3): `p_market_fit` is the logistic fit to the MARKET price (or to
    # risk_neutral_prob when use_rn_prob=True) — it is NOT a risk-neutral MODEL
    # probability. `p_rn_fit` is kept as a deprecated alias (identical values) for
    # one release so existing readers/plots keep working.
    df["p_market_fit"] = np.nan
    df["p_rn_fit"] = np.nan
    df["edge_vs_market_fit"] = np.nan
    df["edge_vs_rn_fit"] = np.nan

    # FIX 7 (M2) Part B: outcome-based recalibration, GATED. We only write the
    # `p_model_cal` column when the master flag is on AND a fitted shift table
    # exists — column presence must NOT be the on/off switch (resolve_model_prob
    # would otherwise silently start using it). p_model_fit is left untouched.
    from core.strategy.common import USE_CALIBRATED_PROB
    _cal_shifts = load_calibration_shift() if USE_CALIBRATED_PROB else {}
    if _cal_shifts:
        df["p_model_cal"] = np.nan

    for group_key, g in df.groupby(group_cols, observed=True):
        expiry_date = g["expiry_date"].iloc[0] if use_expiry_group else None
        T_days = float(g[t_col].iloc[0])

        strikes = g["strike"].values.astype(float)

        # --- Fit model curve (p_real_mc) ---
        p_model = g["p_real_mc"].values.astype(float)
        model_fit = fit_logistic_to_points(strikes, p_model)

        # --- Fit neutral curve (market or RN) ---
        if use_rn_prob:
            y_rn_source = g["risk_neutral_prob"].values.astype(float)
        else:
            y_rn_source = g["market_price"].values.astype(float)

        rn_fit = fit_logistic_to_points(strikes, y_rn_source)

        # Evaluate fits, if they worked
        p_model_fit = np.full(len(g), np.nan)
        p_rn_fit = np.full(len(g), np.nan)

        if model_fit is not None and model_fit.success:
            p_model_fit = eval_logistic_fit(strikes, model_fit)
        if rn_fit is not None and rn_fit.success:
            p_rn_fit = eval_logistic_fit(strikes, rn_fit)

        # Write back into df at the correct indices
        df.loc[g.index, "p_model_fit"] = p_model_fit
        # FIX 9 (M3): primary name + deprecated alias (identical values).
        df.loc[g.index, "p_market_fit"] = p_rn_fit
        df.loc[g.index, "p_rn_fit"] = p_rn_fit

        # FIX 7 (M2) Part B: apply the DTE-bucket logit shift to p_model_fit.
        if _cal_shifts:
            B_bucket = _cal_shifts.get(dte_bucket(T_days), 0.0)
            df.loc[g.index, "p_model_cal"] = apply_calibration_shift(p_model_fit, B_bucket)

        # Edges based on fitted curves
        market = g["market_price"].values.astype(float)
        edge_vs_market_fit = p_model_fit - market
        edge_vs_rn_fit = p_model_fit - p_rn_fit

        df.loc[g.index, "edge_vs_market_fit"] = edge_vs_market_fit
        df.loc[g.index, "edge_vs_rn_fit"] = edge_vs_rn_fit

        # Store curve params for diagnostics
        curve_rows.append({
            "T_days": T_days,
            "expiry_date": expiry_date,
            "n_points": len(g),
            "model_log_a": getattr(model_fit, "log_a", np.nan),
            "model_b": getattr(model_fit, "b", np.nan),
            "model_fit_ok": bool(model_fit and model_fit.success),
            "rn_log_a": getattr(rn_fit, "log_a", np.nan),
            "rn_b": getattr(rn_fit, "b", np.nan),
            "rn_fit_ok": bool(rn_fit and rn_fit.success),
        })

    if "T_bucket" in df.columns:
        df = df.drop(columns=["T_bucket"])

    # Sort rows for readability (by t_days/T_days then strike)
    t_sort_col = None
    if "t_days" in df.columns:
        t_sort_col = "t_days"
    elif "T_days" in df.columns:
        t_sort_col = "T_days"
    sort_cols = []
    if t_sort_col is not None:
        sort_cols.append(t_sort_col)
    if "strike" in df.columns:
        sort_cols.append("strike")
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Save augmented per-contract file
    df.to_csv(output_batch_csv, index=False)

    # Save per-expiry curve params
    curve_df = pd.DataFrame(curve_rows)
    if "T_days" in curve_df.columns:
        curve_df = curve_df.sort_values("T_days").reset_index(drop=True)
    curve_df.to_csv(output_curve_params_csv, index=False)


def copy_metadata_files(batch_summary_path: Path, run_dir: Path) -> None:
    """
    Copy the raw batch_summary.csv and any available regime_summary.csv
    into the fitted output directory so downstream tools can load them
    alongside batch_with_fits.csv.
    """
    try:
        dest_batch = run_dir / batch_summary_path.name
        shutil.copy2(batch_summary_path, dest_batch)
        print(f"Copied batch summary to {dest_batch}")
    except Exception as exc:
        print(f"Warning: unable to copy batch_summary.csv ({exc})")

    # Regime summaries live inside individual slug directories; since each run
    # shares the same regime diagnostics we can grab the first one we find.
    batch_dir = batch_summary_path.parent
    regime_src = batch_dir / "regime_summary.csv"
    if not regime_src.exists():
        for child in batch_dir.iterdir():
            if not child.is_dir():
                continue
            candidate = child / "regime_summary.csv"
            if candidate.exists():
                regime_src = candidate
                break
        else:
            regime_src = None

    if regime_src and regime_src.exists():
        try:
            dest_regime = run_dir / "regime_summary.csv"
            shutil.copy2(regime_src, dest_regime)
            print(f"Copied regime summary to {dest_regime}")
        except Exception as exc:
            print(f"Warning: unable to copy regime_summary.csv ({exc})")
    else:
        print("Warning: no regime_summary.csv found for this batch.")


def get_latest_batch_file(directory: str = "batch_results") -> Optional[str]:
    """Find the most recent batch results CSV file (searches recursively in subdirectories)."""
    import glob
    # Look for CSVs directly in directory
    pattern_flat = Path(directory) / "*.csv"
    # Also look for CSVs in timestamped subdirectories (e.g., batch_results/2025-12-20_05-57-14_UTC/batch_results.csv)
    pattern_nested = Path(directory) / "*" / "*.csv"
    
    files = glob.glob(str(pattern_flat)) + glob.glob(str(pattern_nested))
    if not files:
        return None
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Fit logistic curves to batch_summary probabilities.")
    parser.add_argument("--input", default=None, help="Input batch_summary CSV path. If not provided, uses latest from batch_results/.")
    parser.add_argument(
        "--output-dir",
        default="fitted_batch_results",
        help="Directory where fitted outputs (CSV + plots) will be stored (per input stem).",
    )
    parser.add_argument("--output-batch", default="batch_with_fits.csv", help="Output augmented batch CSV filename.")
    parser.add_argument("--output-curves", default="curve_params.csv", help="Output curve parameters CSV filename.")
    parser.add_argument(
        "--use-rn-prob",
        action="store_true",
        help="If set, fit neutral curve to risk_neutral_prob instead of market_price.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="If set, save outputs directly in output-dir without creating a timestamped subfolder.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="If set, generate curve plots (default: no plots for faster processing).",
    )
    args = parser.parse_args()

    # Auto-detect input file if not specified
    if args.input is None:
        args.input = get_latest_batch_file()
        if args.input is None:
            print("Error: No batch results found in 'batch_results/'. Run batch_pricing_runner.py first.")
            sys.exit(1)
        print(f"Auto-detected input file: {args.input}")
    
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_dir).resolve()

    # Generate output directory (with or without timestamp)
    if args.no_timestamp:
        run_dir = output_root
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
        run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    output_batch_path = run_dir / Path(args.output_batch).name
    output_curve_path = run_dir / Path(args.output_curves).name

    process_batch(
        input_csv=str(input_path),
        output_batch_csv=str(output_batch_path),
        output_curve_params_csv=str(output_curve_path),
        use_rn_prob=args.use_rn_prob,
    )
    copy_metadata_files(input_path, run_dir)

    # Attempt to render plots using the companion script (only if requested)
    if args.generate_plots:
        try:
            subprocess.run(
                [sys.executable, str((Path(__file__).resolve().parent / "plot_batch_curves.py"))],
                check=True,
                cwd=str(run_dir),
            )
        except FileNotFoundError:
            print("Warning: plot_batch_curves.py not found; skipping plots.", file=sys.stderr)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: plot_batch_curves.py failed ({exc}).", file=sys.stderr)
        except Exception as exc:
            print(f"Warning: unable to run plot_batch_curves.py ({exc}).", file=sys.stderr)


if __name__ == "__main__":
    main()
