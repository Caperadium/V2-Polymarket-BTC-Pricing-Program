"""
calibration_metrics.py

Calibration diagnostics for probability forecasts: Brier score, reliability
diagram, and Expected Calibration Error (ECE). Compares model probabilities
against realized binary outcomes.

Usage:
    from core.validation.calibration_metrics import (
        brier_score, reliability_bins, ece_score, run_calibration_report,
    )
    report = run_calibration_report("fitted_batch_results/batch_with_fits.csv")
    print(f"Brier: {report['brier']:.4f}, ECE: {report['ece']:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def brier_score(p_model: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Compute Brier score: mean squared error between forecast and outcome.

    Args:
        p_model: Array of model probabilities in (0, 1).
        outcomes: Array of binary outcomes (0 or 1).

    Returns:
        Brier score ∈ [0, 1]. Lower is better. 0.25 = baseline (coin flip).
    """
    p = np.asarray(p_model, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[mask] - y[mask]) ** 2))


def reliability_bins(
    p_model: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute reliability diagram bins: mean forecast vs observed frequency.

    Args:
        p_model: Array of model probabilities in (0, 1).
        outcomes: Array of binary outcomes (0 or 1).
        n_bins: Number of equal-width bins for binning probabilities.

    Returns:
        DataFrame with columns: bin_center, bin_lower, bin_upper, n_obs,
        mean_forecast, observed_freq.
    """
    p = np.asarray(p_model, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        in_bin = (p >= lower) & (p < upper)
        # Include upper edge in last bin
        if i == n_bins - 1:
            in_bin = (p >= lower) & (p <= upper)
        n = int(np.sum(in_bin))
        if n > 0:
            mean_fc = float(np.mean(p[in_bin]))
            obs_freq = float(np.mean(y[in_bin]))
        else:
            mean_fc = float('nan')
            obs_freq = float('nan')
        rows.append({
            "bin_center": (lower + upper) / 2,
            "bin_lower": lower,
            "bin_upper": upper,
            "n_obs": n,
            "mean_forecast": mean_fc,
            "observed_freq": obs_freq,
        })

    return pd.DataFrame(rows)


def ece_score(
    p_model: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error: weighted average of |forecast - observed| per bin.

    Args:
        p_model: Array of model probabilities in (0, 1).
        outcomes: Array of binary outcomes (0 or 1).
        n_bins: Number of bins.

    Returns:
        ECE ∈ [0, 1]. Lower is better.
    """
    bins_df = reliability_bins(p_model, outcomes, n_bins)
    total = bins_df["n_obs"].sum()
    if total == 0:
        return float('nan')
    ece = 0.0
    for _, row in bins_df.iterrows():
        if row["n_obs"] > 0 and not np.isnan(row["mean_forecast"]):
            ece += (row["n_obs"] / total) * abs(
                row["mean_forecast"] - row["observed_freq"]
            )
    return float(ece)


# ---------------------------------------------------------------------------
# Report runner
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    brier: float
    ece: float
    n_obs: int
    bins: pd.DataFrame
    # Skew/mean diagnostics
    mean_forecast: float
    mean_outcome: float
    calibration_bias: float  # mean_forecast - mean_outcome


def run_calibration_report(
    csv_path: str,
    prob_col: str = "p_model_fit",
    outcome_col: str = "outcome",
    n_bins: int = 10,
) -> Optional[CalibrationReport]:
    """
    Run full calibration diagnostics on a priced batch CSV.

    The CSV must contain a model probability column and an outcome column.
    If no outcome column is found, attempts to infer from market resolution data.

    Args:
        csv_path: Path to batch_with_fits.csv or similar.
        prob_col: Column name for model probabilities.
        outcome_col: Column name for binary outcomes.
        n_bins: Number of bins for reliability diagram.

    Returns:
        CalibrationReport if data is sufficient, None otherwise.
    """
    df = pd.read_csv(csv_path)

    # Try to detect outcome column
    if outcome_col not in df.columns:
        # Check for common alternatives
        alt_outcomes = ["resolved", "settled", "result", "actual"]
        found = False
        for alt in alt_outcomes:
            if alt in df.columns:
                outcome_col = alt
                found = True
                break
        if not found:
            logger.warning(
                "No outcome column found in %s. Cannot compute calibration metrics. "
                "Expected column: '%s' or one of %s. "
                "Run the model over historical data with known outcomes first.",
                csv_path, outcome_col, alt_outcomes,
            )
            return None

    # Try to detect probability column
    if prob_col not in df.columns:
        alt_probs = ["p_model_fit", "p_real_mc", "model_probability"]
        found = False
        for alt in alt_probs:
            if alt in df.columns:
                prob_col = alt
                found = True
                break
        if not found:
            logger.warning(
                "No probability column found in %s. Expected one of %s.",
                csv_path, [prob_col] + alt_probs,
            )
            return None

    # Drop missing
    valid = df[[prob_col, outcome_col]].dropna()
    p_model = valid[prob_col].values
    outcomes = valid[outcome_col].values

    n = len(p_model)
    if n < 20:
        logger.warning("Only %d observations — too few for reliable calibration metrics.", n)
        return None

    brier = brier_score(p_model, outcomes)
    ece = ece_score(p_model, outcomes, n_bins)
    bins_df = reliability_bins(p_model, outcomes, n_bins)

    mean_fc = float(np.mean(p_model))
    mean_out = float(np.mean(outcomes))
    bias = mean_fc - mean_out

    report = CalibrationReport(
        brier=brier,
        ece=ece,
        n_obs=n,
        bins=bins_df,
        mean_forecast=mean_fc,
        mean_outcome=mean_out,
        calibration_bias=bias,
    )

    logger.info(
        "Calibration report: n=%d, Brier=%.4f, ECE=%.4f, "
        "mean_forecast=%.4f, mean_outcome=%.4f, bias=%.4f",
        n, brier, ece, mean_fc, mean_out, bias,
    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Compute calibration metrics for priced batch CSV"
    )
    parser.add_argument("input", help="Path to batch_with_fits.csv or similar")
    parser.add_argument("--prob-col", default="p_model_fit",
                        help="Model probability column (default: p_model_fit)")
    parser.add_argument("--outcome-col", default="outcome",
                        help="Outcome column (default: outcome)")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of reliability bins (default: 10)")
    args = parser.parse_args()

    report = run_calibration_report(
        args.input,
        prob_col=args.prob_col,
        outcome_col=args.outcome_col,
        n_bins=args.n_bins,
    )

    if report is None:
        print("No report generated — check that input CSV has outcome data.")
    else:
        print(f"\n=== Calibration Report ===")
        print(f"Observations:     {report.n_obs}")
        print(f"Brier score:      {report.brier:.4f}")
        print(f"ECE:              {report.ece:.4f}")
        print(f"Mean forecast:    {report.mean_forecast:.4f}")
        print(f"Mean outcome:     {report.mean_outcome:.4f}")
        print(f"Calibration bias: {report.calibration_bias:+.4f}")
        print(f"\nReliability Diagram:")
        print(report.bins.to_string(index=False))
