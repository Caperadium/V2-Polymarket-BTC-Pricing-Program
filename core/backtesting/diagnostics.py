#!/usr/bin/env python3
"""
diagnostics.py — absorbed from core/strategy/signal_diagnostics.py

Computes Spearman rank correlation and AUC between model edge and realized
outcomes, plus DTE and moneyness breakdowns.
"""

import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Helpers (preserved from signal_diagnostics.py)
# ---------------------------------------------------------------------------

def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column present in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coerce_outcome(series: pd.Series) -> pd.Series:
    """Coerce outcome values to binary 0/1.

    Handles: {0,1}, {0.0,1.0}, True/False, "YES"/"NO", "yes"/"no".
    """
    str_series = series.astype(str).str.strip().str.upper()
    mapping = {
        "1": 1, "1.0": 1, "TRUE": 1, "YES": 1,
        "0": 0, "0.0": 0, "FALSE": 0, "NO": 0,
    }
    return str_series.map(mapping)


# ---------------------------------------------------------------------------
# SignalDiagnostics
# ---------------------------------------------------------------------------

class SignalDiagnostics:
    """Absorbs ``signal_diagnostics.py`` logic.

    Computes Spearman rho, AUC, and breakdowns by DTE and moneyness.
    """

    DTE_BINS = [
        ("DTE 1-2", lambda d: (d["_dte"] >= 1) & (d["_dte"] <= 2)),
        ("DTE 3-4", lambda d: (d["_dte"] >= 3) & (d["_dte"] <= 4)),
        ("DTE 5-6", lambda d: (d["_dte"] >= 5) & (d["_dte"] <= 6)),
        ("DTE 7+",  lambda d: d["_dte"] >= 7),
    ]

    MONEYNESS_BINS = [
        ("ATM (|m| <= 2%)",     lambda d: d["_moneyness"].abs() <= 0.02),
        ("Near-ATM (|m| <= 5%)", lambda d: d["_moneyness"].abs() <= 0.05),
        ("OTM (m > 5%)",        lambda d: d["_moneyness"] > 0.05),
        ("ITM (m < -5%)",       lambda d: d["_moneyness"] < -0.05),
    ]

    def __init__(self, all_priced_df: pd.DataFrame):
        self.raw_df = all_priced_df
        self.work: Optional[pd.DataFrame] = None
        self._dte_available = False
        self._moneyness_available = False
        self._clean_data()

    # ------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------

    def _clean_data(self) -> None:
        """Coerce outcomes, compute edge, drop unusable rows."""
        df = self.raw_df.copy()

        if df.empty:
            self.work = df
            return

        # Column selection (same precedence as signal_diagnostics.py)
        outcome_col = _pick_column(df, ["outcome_yes", "outcome"])
        model_col = _pick_column(df, ["model_prob_used"])
        market_col = _pick_column(df, ["market_yes_price"])

        if outcome_col is None or model_col is None or market_col is None:
            self.work = df  # will be empty downstream
            return

        # Coerce outcome
        df["_outcome"] = _coerce_outcome(df[outcome_col])

        # Convert probabilities
        df["_model_prob"] = pd.to_numeric(df[model_col], errors="coerce")
        df["_market_price"] = pd.to_numeric(df[market_col], errors="coerce")

        # Drop rows with missing core values
        df = df.dropna(subset=["_outcome", "_model_prob", "_market_price"])

        if df.empty:
            self.work = df
            return

        # Clip to (eps, 1-eps)
        eps = 1e-6
        df["_model_prob"] = df["_model_prob"].clip(eps, 1 - eps)
        df["_market_price"] = df["_market_price"].clip(eps, 1 - eps)

        # Edge
        df["_edge"] = df["_model_prob"] - df["_market_price"]

        # Binary outcome
        df["_outcome"] = df["_outcome"].astype(int)

        # Optional columns
        dte_col = _pick_column(df, ["dte_days", "t_days", "T_days"])
        if dte_col is not None:
            df["_dte"] = pd.to_numeric(df[dte_col], errors="coerce")
            self._dte_available = True
        else:
            df["_dte"] = np.nan

        money_col = _pick_column(df, ["moneyness"])
        if money_col is not None:
            df["_moneyness"] = pd.to_numeric(df[money_col], errors="coerce")
            self._moneyness_available = True
        else:
            df["_moneyness"] = np.nan

        self.work = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(
        outcome: np.ndarray, score: np.ndarray
    ) -> Tuple[float, float, Optional[float]]:
        """Return (rho, p_value, auc). AUC is None if no class diversity."""
        rho, p_value = spearmanr(score, outcome)
        unique = np.unique(outcome)
        if len(unique) < 2:
            return rho, p_value, None
        try:
            auc = roc_auc_score(outcome, score)
        except ValueError:
            auc = None
        return rho, p_value, auc

    @staticmethod
    def interpret_auc(auc: Optional[float]) -> str:
        """Return human-readable AUC interpretation."""
        if auc is None:
            return "N/A (no class diversity)"
        if auc > 0.55:
            return f"{auc:.4f}  (positive signal - model edge predicts wins)"
        if auc < 0.45:
            return f"{auc:.4f}  (anti-signal - inverted relationship)"
        return f"{auc:.4f}  (no discrimination - ~random)"

    # ------------------------------------------------------------------
    # Breakdown helpers
    # ------------------------------------------------------------------

    def _run_breakdown(
        self,
        bins: List[Tuple[str, callable]],
        data_col: str,
    ) -> List[dict]:
        """Run metrics on subsets defined by filter functions.

        Returns a list of dicts, one per bin.
        """
        if self.work is None or self.work.empty:
            return []

        results = []
        for label, filter_fn in bins:
            try:
                subset = self.work[filter_fn(self.work)]
            except Exception:
                continue

            if len(subset) < 10:
                results.append({
                    "label": label, "n": len(subset),
                    "pos": 0, "neg": 0,
                    "rho": np.nan, "p": np.nan, "auc": None,
                })
                continue

            outcome = subset["_outcome"].values
            score = subset["_edge"].values
            pos_count = int(outcome.sum())
            neg_count = len(outcome) - pos_count

            if pos_count == 0 or neg_count == 0:
                results.append({
                    "label": label, "n": len(subset),
                    "pos": pos_count, "neg": neg_count,
                    "rho": np.nan, "p": np.nan, "auc": None,
                })
                continue

            rho, p_val, auc = self.compute_metrics(outcome, score)
            results.append({
                "label": label, "n": len(subset),
                "pos": pos_count, "neg": neg_count,
                "rho": float(rho), "p": float(p_val),
                "auc": float(auc) if auc is not None else None,
            })
        return results

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def run_full_report(self) -> dict:
        """Run full signal diagnostics and return a structured dict.

        Returns
        -------
        dict
            {
                "n_observations": int,
                "n_positive": int, "n_negative": int,
                "spearman_rho": float, "spearman_pvalue": float, "auc": float | None,
                "mean_edge_winners": float, "mean_edge_losers": float,
                "edge_difference": float,
                "dte_breakdown": [{"label": ..., "n": ..., "rho": ..., "auc": ...}, ...],
                "moneyness_breakdown": [...],
                "dte_available": bool,
                "moneyness_available": bool,
            }
        """
        if self.work is None or self.work.empty:
            return {
                "n_observations": 0, "n_positive": 0, "n_negative": 0,
                "spearman_rho": np.nan, "spearman_pvalue": np.nan, "auc": None,
                "mean_edge_winners": np.nan, "mean_edge_losers": np.nan,
                "edge_difference": np.nan,
                "dte_breakdown": [], "moneyness_breakdown": [],
                "dte_available": False, "moneyness_available": False,
            }

        outcome = self.work["_outcome"].values
        edge = self.work["_edge"].values
        pos_count = int(outcome.sum())
        neg_count = len(outcome) - pos_count

        # Overall metrics
        if pos_count == 0 or neg_count == 0:
            rho, p_val, auc = np.nan, np.nan, None
            mean_edge_winners = np.nan
            mean_edge_losers = np.nan
            edge_difference = np.nan
        else:
            rho, p_val, auc = self.compute_metrics(outcome, edge)
            mean_edge_winners = float(self.work.loc[self.work["_outcome"] == 1, "_edge"].mean())
            mean_edge_losers = float(self.work.loc[self.work["_outcome"] == 0, "_edge"].mean())
            edge_difference = mean_edge_winners - mean_edge_losers

        # Breakdowns
        dte_breakdown = self._run_breakdown(self.DTE_BINS, "_dte") if self._dte_available else []
        moneyness_breakdown = (
            self._run_breakdown(self.MONEYNESS_BINS, "_moneyness")
            if self._moneyness_available
            else []
        )

        return {
            "n_observations": len(self.work),
            "n_positive": pos_count,
            "n_negative": neg_count,
            "spearman_rho": float(rho),
            "spearman_pvalue": float(p_val),
            "auc": float(auc) if auc is not None else None,
            "mean_edge_winners": mean_edge_winners,
            "mean_edge_losers": mean_edge_losers,
            "edge_difference": edge_difference,
            "dte_breakdown": dte_breakdown,
            "moneyness_breakdown": moneyness_breakdown,
            "dte_available": self._dte_available,
            "moneyness_available": self._moneyness_available,
        }

    # ------------------------------------------------------------------
    # CLI — preserved from signal_diagnostics.py
    # ------------------------------------------------------------------

    @staticmethod
    def run_diagnostics(csv_path: str) -> None:
        """Load CSV and print full signal diagnostics to stdout.

        Preserved CLI interface from ``core/strategy/signal_diagnostics.py``.
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"ERROR: Failed to load CSV: {e}")
            sys.exit(1)

        print(f"Loaded: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")

        diag = SignalDiagnostics(df)
        report = diag.run_full_report()

        print(f"\nAfter cleaning: {report['n_observations']} rows")
        print(
            f"Class distribution: pos={report['n_positive']}, "
            f"neg={report['n_negative']}"
        )

        print("\n" + "=" * 60)
        print("OVERALL METRICS")
        print("=" * 60)

        if report["n_positive"] == 0 or report["n_negative"] == 0:
            print("WARNING: No class diversity - cannot compute meaningful metrics")
        else:
            rho = report["spearman_rho"]
            p_val = report["spearman_pvalue"]
            auc = report["auc"]
            print(f"  Spearman rho: {rho:.4f}  (p-value: {p_val:.6f})")
            print(f"  AUC:          {SignalDiagnostics.interpret_auc(auc)}")
            print(f"\n  Mean edge (outcome=1): {report['mean_edge_winners']:.4f}")
            print(f"  Mean edge (outcome=0): {report['mean_edge_losers']:.4f}")
            print(f"  Edge difference:       {report['edge_difference']:.4f}")

        # DTE breakdown
        if report["dte_available"] and report["dte_breakdown"]:
            print("\nBy DTE:")
            for row in report["dte_breakdown"]:
                n = row["n"]
                if row["auc"] is None:
                    comp = "(no class diversity)" if n >= 10 else "(too few samples)"
                    print(f"  {row['label']}: n={n} {comp}")
                else:
                    print(
                        f"  {row['label']}: n={n}, pos={row['pos']}, neg={row['neg']}, "
                        f"rho={row['rho']:.4f} (p={row['p']:.4f}), auc={row['auc']:.4f}"
                    )

        # Moneyness breakdown
        if report["moneyness_available"] and report["moneyness_breakdown"]:
            print("\nBy Moneyness:")
            for row in report["moneyness_breakdown"]:
                n = row["n"]
                if row["auc"] is None:
                    comp = "(no class diversity)" if n >= 10 else "(too few samples)"
                    print(f"  {row['label']}: n={n} {comp}")
                else:
                    print(
                        f"  {row['label']}: n={n}, pos={row['pos']}, neg={row['neg']}, "
                        f"rho={row['rho']:.4f} (p={row['p']:.4f}), auc={row['auc']:.4f}"
                    )

        print("\n" + "=" * 60)


def run_diagnostics(csv_path: str) -> None:
    """Standalone wrapper for CLI-preserved diagnostic runner.

    Usage: run_diagnostics("path/to/all_priced.csv")
    """
    SignalDiagnostics.run_diagnostics(csv_path)


def main_cli() -> int:
    """argparse entrypoint for CLI invocation.

    Usage: python core/backtesting/diagnostics.py path/to/all_priced.csv
    """
    parser = argparse.ArgumentParser(
        description="Compute signal diagnostics (Spearman, AUC) for priced contracts"
    )
    parser.add_argument("csv_path", help="Path to CSV file of all priced contracts")
    args = parser.parse_args()
    SignalDiagnostics.run_diagnostics(args.csv_path)
    return 0


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main_cli())
