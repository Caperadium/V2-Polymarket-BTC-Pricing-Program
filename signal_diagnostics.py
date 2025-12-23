#!/usr/bin/env python3
"""
signal_diagnostics.py

Computes Spearman rank correlation and AUC between model edge and realized outcomes.
Self-contained module with no project dependencies.

Usage:
    python signal_diagnostics.py path/to/all_priced.csv
"""

import argparse
import sys
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column present in df."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def coerce_outcome(series: pd.Series) -> pd.Series:
    """
    Coerce outcome values to binary 0/1.
    
    Handles: {0,1}, {0.0,1.0}, True/False, "YES"/"NO", "yes"/"no"
    """
    # Convert to string for uniform handling, then map
    str_series = series.astype(str).str.strip().str.upper()
    
    # Map various representations to 0/1
    mapping = {
        "1": 1, "1.0": 1, "TRUE": 1, "YES": 1,
        "0": 0, "0.0": 0, "FALSE": 0, "NO": 0,
    }
    
    result = str_series.map(mapping)
    return result


def compute_metrics(
    outcome: np.ndarray, 
    score: np.ndarray
) -> Tuple[float, float, Optional[float]]:
    """
    Compute Spearman rho, p-value, and AUC.
    
    Returns:
        (rho, p_value, auc) - auc is None if no class diversity
    """
    # Spearman correlation
    rho, p_value = spearmanr(score, outcome)
    
    # AUC (only if we have class diversity)
    unique_outcomes = np.unique(outcome)
    if len(unique_outcomes) < 2:
        auc = None
    else:
        try:
            auc = roc_auc_score(outcome, score)
        except ValueError:
            auc = None
    
    return rho, p_value, auc


def interpret_auc(auc: Optional[float]) -> str:
    """Return interpretation string for AUC value."""
    if auc is None:
        return "N/A (no class diversity)"
    if auc > 0.55:
        return f"{auc:.4f}  (positive signal - model edge predicts wins)"
    elif auc < 0.45:
        return f"{auc:.4f}  (anti-signal - inverted relationship)"
    else:
        return f"{auc:.4f}  (no discrimination - ~random)"


def run_subset_analysis(
    df: pd.DataFrame,
    outcome_col: str,
    edge_col: str,
    subset_name: str,
    bins: List[Tuple[str, callable]],
):
    """Run metrics on subsets defined by filter functions."""
    print(f"\n{subset_name}:")
    
    for label, filter_fn in bins:
        try:
            subset = df[filter_fn(df)].copy()
        except Exception:
            continue
            
        if len(subset) < 10:
            print(f"  {label}: n={len(subset)} (too few samples)")
            continue
        
        outcome = subset[outcome_col].values
        score = subset[edge_col].values
        
        pos_count = int(outcome.sum())
        neg_count = len(outcome) - pos_count
        
        if pos_count == 0 or neg_count == 0:
            print(f"  {label}: n={len(subset)} (no class diversity)")
            continue
        
        rho, p_val, auc = compute_metrics(outcome, score)
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        
        print(f"  {label}: n={len(subset)}, pos={pos_count}, neg={neg_count}, "
              f"rho={rho:.4f} (p={p_val:.4f}), auc={auc_str}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_diagnostics(csv_path: str) -> None:
    """Load CSV and run full signal diagnostics."""
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}")
        sys.exit(1)
    
    print(f"Loaded: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # -------------------------------------------------------------------------
    # Column selection
    # -------------------------------------------------------------------------
    
    # Outcome column
    outcome_col = pick_column(df, ["outcome_yes", "outcome"])
    if outcome_col is None:
        print("ERROR: No outcome column found (tried: outcome_yes, outcome)")
        sys.exit(1)
    
    # Model probability column
    model_col = pick_column(df, ["model_prob_used"])
    if model_col is None:
        print("ERROR: No model probability column found (tried: model_prob_used)")
        sys.exit(1)
    
    # Market price column
    market_col = pick_column(df, ["market_yes_price"])
    if market_col is None:
        print("ERROR: No market price column found (tried: market_yes_price)")
        sys.exit(1)
    
    print(f"Using columns: outcome={outcome_col}, model_prob={model_col}, market_price={market_col}")
    
    # -------------------------------------------------------------------------
    # Data cleaning
    # -------------------------------------------------------------------------
    
    work = df.copy()
    
    # Coerce outcome to binary
    work["_outcome"] = coerce_outcome(work[outcome_col])
    
    # Convert probability columns to numeric
    work["_model_prob"] = pd.to_numeric(work[model_col], errors="coerce")
    work["_market_price"] = pd.to_numeric(work[market_col], errors="coerce")
    
    # Drop rows with missing values
    before_count = len(work)
    work = work.dropna(subset=["_outcome", "_model_prob", "_market_price"])
    after_drop = len(work)
    
    if after_drop == 0:
        print("ERROR: No valid rows after cleaning")
        sys.exit(1)
    
    # Clip probabilities and prices to valid range
    eps = 1e-6
    work["_model_prob"] = work["_model_prob"].clip(eps, 1 - eps)
    work["_market_price"] = work["_market_price"].clip(eps, 1 - eps)
    
    # Compute edge
    work["_edge"] = work["_model_prob"] - work["_market_price"]
    
    # Ensure outcome is 0/1
    work["_outcome"] = work["_outcome"].astype(int)
    
    pos_count = int(work["_outcome"].sum())
    neg_count = len(work) - pos_count
    
    print(f"After cleaning: {len(work)} rows (dropped {before_count - after_drop})")
    print(f"Class distribution: pos={pos_count}, neg={neg_count}")
    
    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------
    
    outcome = work["_outcome"].values
    edge = work["_edge"].values
    
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    
    if pos_count == 0 or neg_count == 0:
        print("WARNING: No class diversity - cannot compute meaningful metrics")
    else:
        rho, p_val, auc = compute_metrics(outcome, edge)
        
        print(f"  Spearman rho: {rho:.4f}  (p-value: {p_val:.6f})")
        print(f"  AUC:          {interpret_auc(auc)}")
        
        # Additional context
        mean_edge_winners = work[work["_outcome"] == 1]["_edge"].mean()
        mean_edge_losers = work[work["_outcome"] == 0]["_edge"].mean()
        print(f"\n  Mean edge (outcome=1): {mean_edge_winners:.4f}")
        print(f"  Mean edge (outcome=0): {mean_edge_losers:.4f}")
        print(f"  Edge difference:       {mean_edge_winners - mean_edge_losers:.4f}")
    
    # -------------------------------------------------------------------------
    # Optional: DTE breakdown
    # -------------------------------------------------------------------------
    
    dte_col = pick_column(work, ["dte_days", "t_days", "T_days"])
    if dte_col is not None:
        work["_dte"] = pd.to_numeric(work[dte_col], errors="coerce")
        
        dte_bins = [
            ("DTE 1-2", lambda d: (d["_dte"] >= 1) & (d["_dte"] <= 2)),
            ("DTE 3-4", lambda d: (d["_dte"] >= 3) & (d["_dte"] <= 4)),
            ("DTE 5-6", lambda d: (d["_dte"] >= 5) & (d["_dte"] <= 6)),
            ("DTE 7+",  lambda d: d["_dte"] >= 7),
        ]
        
        run_subset_analysis(work, "_outcome", "_edge", "By DTE", dte_bins)
    
    # -------------------------------------------------------------------------
    # Optional: Moneyness breakdown
    # -------------------------------------------------------------------------
    
    money_col = pick_column(work, ["moneyness"])
    if money_col is not None:
        work["_moneyness"] = pd.to_numeric(work[money_col], errors="coerce")
        
        money_bins = [
            ("ATM (|m| <= 2%)",     lambda d: d["_moneyness"].abs() <= 0.02),
            ("Near-ATM (|m| <= 5%)", lambda d: d["_moneyness"].abs() <= 0.05),
            ("OTM (m > 5%)",        lambda d: d["_moneyness"] > 0.05),
            ("ITM (m < -5%)",       lambda d: d["_moneyness"] < -0.05),
        ]
        
        run_subset_analysis(work, "_outcome", "_edge", "By Moneyness", money_bins)
    
    print("\n" + "="*60)


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute signal diagnostics (Spearman, AUC) for priced contracts"
    )
    parser.add_argument(
        "csv_path",
        help="Path to CSV file of all priced contracts"
    )
    
    args = parser.parse_args()
    run_diagnostics(args.csv_path)
