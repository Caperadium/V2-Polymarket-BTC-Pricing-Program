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
from scipy.optimize import curve_fit
from scipy.special import expit

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None

# ---------- Global Calibration ----------
# Logit shift calibration: p_cal = sigmoid(logit(p) + B)
# B=0.0 preserves original; B<0 shifts probabilities downward uniformly
# This avoids inflating low p's (unlike symmetric shrink-to-0.5)
PROB_LOGIT_SHIFT_B = -0.7

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
    df["p_rn_fit"] = np.nan
    df["edge_vs_market_fit"] = np.nan
    df["edge_vs_rn_fit"] = np.nan

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
        df.loc[g.index, "p_rn_fit"] = p_rn_fit

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

    # Apply logit shift calibration: p_cal = sigmoid(logit(p) + BIAS_SHIFT_B)
    # This pushes probabilities uniformly downward without inflating low p's
    eps = 1e-6
    p_fit = df["p_model_fit"].values
    p_clipped = np.clip(p_fit, eps, 1 - eps)
    logit_p = np.log(p_clipped / (1 - p_clipped))
    logit_cal = logit_p + PROB_LOGIT_SHIFT_B
    p_cal = 1 / (1 + np.exp(-logit_cal))
    df["p_model_cal"] = np.clip(p_cal, eps, 1 - eps)

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
