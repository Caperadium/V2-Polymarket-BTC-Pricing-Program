#!/usr/bin/env python3
"""
dashboard.py

Streamlit dashboard for exploring Polymarket BTC pricing output across
multiple days, expiries, and regimes.

Features
--------
* Snapshot of current opportunities (top trades, expiry summary).
* Curves & Edges inspection with Plotly charts.
* Stability diagnostics (multi-day probability drift, cross-expiry overlays,
  logistic-fit sanity checks, RN↔Model↔Market comparisons, histograms).
* Risk & Bankroll Monte Carlo simulator (fractional Kelly).
* Regime diagnostics (variance/tail metrics, optional bias analysis).
* Calibration tab (reliability diagram + Brier scores, regime breakdown when available).
"""

from __future__ import annotations

import math
import os
import glob
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from auto_reco import LAST_RECO_DEBUG, recommend_trades, recommendations_to_dataframe
from backtest_engine import run_backtest
from positions import (
    enrich_positions_with_batch,
    load_positions,
    ensure_position_keys,
    get_open_positions,
    sync_open_positions_with_batch,
)
from sweep_config import SweepConfig

# Load default config for sidebar defaults
_DEFAULTS = SweepConfig()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_csv(path: str, cache_bust: Optional[float] = None) -> pd.DataFrame:
    """Cache CSV loading to avoid repeated disk IO.
    Return empty DataFrame if file exists but has no data."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def get_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate column present in df."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def derive_expiry_key(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has an expiry_key column (expiry_date preferred, else rounded T_days)."""
    df = df.copy()
    if "expiry_date" in df.columns:
        df["expiry_key"] = df["expiry_date"].astype(str)
    else:
        t_col = get_column(df, ["t_days", "T_days"])
        if t_col is None:
            df["expiry_key"] = "unknown"
        else:
            df["expiry_key"] = df[t_col].astype(float).round(3).astype(str)
    return df


def compute_edge_series(df: pd.DataFrame) -> pd.Series:
    """Return best-available edge series."""
    edge_col = get_column(df, ["edge_vs_market_fit", "edge_vs_market", "edge"])
    if edge_col is not None:
        return df[edge_col].astype(float)
    model_col = get_column(df, ["p_model_cal", "p_model_fit", "p_real_mc"])
    price_col = get_column(df, ["market_price", "market_pr"])
    if model_col is None or price_col is None:
        return pd.Series(np.nan, index=df.index)
    return df[model_col].astype(float) - df[price_col].astype(float)


def try_load_dataframe(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Attempt to load a CSV; return (df, error_message)."""
    if not path:
        return None, None
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        cache_bust = os.path.getmtime(path)
    except FileNotFoundError:
        return None, f"File not found: {path}"
    try:
        return load_csv(path, cache_bust), None
    except Exception as exc:  # pragma: no cover - defensive for bad files
        return None, f"Failed to read {path}: {exc}"


def build_daily_overview(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stability metrics per pricing date."""
    agg_spec = {
        "n_expiries": ("expiry_key", lambda s: s.nunique()),
    }
    if "stability_score" in summary_df.columns:
        agg_spec["avg_stability_score"] = ("stability_score", "mean")
    if "monotonic_violations" in summary_df.columns:
        agg_spec["avg_monotonic_violations"] = ("monotonic_violations", "mean")
    if "monotonic_violation_rate" in summary_df.columns:
        agg_spec["avg_monotonic_violation_rate"] = ("monotonic_violation_rate", "mean")
    if "mean_abs_resid" in summary_df.columns:
        agg_spec["avg_mean_abs_resid"] = ("mean_abs_resid", "mean")
    if "mean_abs_diff_rn_model" in summary_df.columns:
        agg_spec["avg_mean_abs_diff_rn_model"] = ("mean_abs_diff_rn_model", "mean")
    daily = summary_df.groupby("pricing_date", observed=True).agg(**agg_spec).reset_index()
    return daily.sort_values("pricing_date")


def summarize_expiry_metrics(df: pd.DataFrame, metrics: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    """Create a table with mean/max values for selected columns."""
    rows: List[Dict[str, float]] = []
    for column, label in metrics:
        if column in df.columns:
            series = df[column].dropna()
            if not series.empty:
                rows.append(
                    {
                        "metric": label,
                        "mean": series.mean(),
                        "max": series.max(),
                    }
                )
    return pd.DataFrame(rows)


def render_time_series_line(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    yaxis_label: str,
    color_col: Optional[str] = None,
) -> None:
    """Plot a time-series line chart if the column exists."""
    if y_col not in df.columns:
        return
    clean = df.dropna(subset=[y_col]).copy()
    if clean.empty:
        return
    fig = px.line(
        clean,
        x="pricing_date",
        y=y_col,
        color=color_col,
        markers=True,
        title=title,
        template="plotly_white",
    )
    fig.update_layout(xaxis_title="Pricing date", yaxis_title=yaxis_label)
    st.plotly_chart(fig, width="stretch")


def build_position_key(slug: str, side: str, expiry: str) -> str:
    return f"{slug}|{side}|{expiry}"


def append_resolved_entry(path: Path, entry: dict) -> None:
    columns = [
        "position_key",
        "slug",
        "side",
        "expiry_date",
        "strike",
        "entry_price",
        "size_shares",
        "p_model_fit_at_trade",
        "market_price_at_trade",
        "outcome",
        "recorded_at",
        "notes",
    ]
    if path.exists():
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[columns]
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(path, index=False)


def ensure_position_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["position_key"] = []
        return df
    def _key(row):
        slug = str(row.get("slug") or "")
        side = str(row.get("side") or "")
        expiry = str(row.get("expiry_key") or row.get("expiry_date") or "")
        return build_position_key(slug, side, expiry)
    df["position_key"] = df.apply(_key, axis=1)
    return df


def compute_realized_pnl_total(positions_df: pd.DataFrame, resolved_df: pd.DataFrame) -> float:
    """Aggregate realized PnL for closed positions using resolved outcomes when available."""
    if positions_df.empty or "status" not in positions_df.columns:
        return 0.0
    status_series = positions_df["status"].astype(str).str.upper()
    closed_df = positions_df[status_series == "CLOSED"].copy()
    if closed_df.empty:
        return 0.0
    closed_df["entry_price"] = pd.to_numeric(closed_df.get("entry_price"), errors="coerce")
    closed_df["size_shares"] = pd.to_numeric(closed_df.get("size_shares"), errors="coerce")
    closed_df["realized_pnl"] = pd.to_numeric(closed_df.get("realized_pnl"), errors="coerce").fillna(0.0)
    realized_series = closed_df["realized_pnl"].astype(float).copy()

    if (
        resolved_df is not None
        and not resolved_df.empty
        and "position_key" in closed_df.columns
        and "position_key" in resolved_df.columns
        and "outcome" in resolved_df.columns
    ):
        resolved_subset = resolved_df[["position_key", "outcome"]].dropna(subset=["outcome"])
        if not resolved_subset.empty:
            merged = closed_df.merge(resolved_subset, on="position_key", how="left", suffixes=("", "_resolved"))
            merged["outcome"] = pd.to_numeric(merged["outcome"], errors="coerce")
            sides = merged["side"].astype(str).str.upper()
            payouts = np.where(
                sides == "YES",
                merged["outcome"],
                np.where(sides == "NO", 1.0 - merged["outcome"], np.nan),
            )
            pnl_from_outcome = (payouts - merged["entry_price"]) * merged["size_shares"]
            realized_series = np.where(~np.isnan(pnl_from_outcome), pnl_from_outcome, realized_series)
    return float(pd.Series(realized_series).sum(skipna=True))


def normalize_resolved_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure resolved df has needed columns."""
    if df is None or df.empty:
        return df
    if "p_model_fit_at_trade" not in df.columns:
        if "model_prob_at_entry" in df.columns:
            df["p_model_fit_at_trade"] = df["model_prob_at_entry"]
        else:
            df["p_model_fit_at_trade"] = np.nan
    if "market_price_at_trade" not in df.columns:
        if "entry_price" in df.columns:
            df["market_price_at_trade"] = df["entry_price"]
        elif "market_price" in df.columns:
            df["market_price_at_trade"] = df["market_price"]
        else:
            df["market_price_at_trade"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Kelly / bankroll helpers
# ---------------------------------------------------------------------------


def kelly_fraction_yes(p: float, q: float) -> float:
    """Full Kelly fraction for a YES bet priced at q with true prob p."""
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return max((p - q) / (1.0 - q), 0.0)


def kelly_fraction_no(p: float, q: float) -> float:
    """Full Kelly fraction for buying NO (short YES) priced at q with true prob p."""
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return max((q - p) / q, 0.0)


def simulate_today_portfolio(
    reco_df: pd.DataFrame,
    bankroll: float,
    prob_col: str,
    price_col: str,
    side_col: str,
    stake_col: str,
    n_paths: int,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation for the current Auto-Reco portfolio.
    Does not re-run selection logic; simply samples outcomes using model probabilities.
    """
    if reco_df.empty:
        raise ValueError("No trades to simulate.")
    if seed and seed > 0:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    sides = reco_df[side_col].astype(str).str.lower().to_numpy()
    p_yes = reco_df[prob_col].astype(float).to_numpy()
    price_yes = reco_df[price_col].astype(float).to_numpy()
    stake = reco_df[stake_col].astype(float).to_numpy()

    price_yes = np.clip(price_yes, 1e-6, 1 - 1e-6)
    price_no = 1.0 - price_yes

    pnl_if_yes = np.zeros_like(stake, dtype=float)
    pnl_if_no = np.zeros_like(stake, dtype=float)

    mask_yes = sides == "yes"
    mask_no = sides == "no"

    pnl_if_yes[mask_yes] = stake[mask_yes] * (1.0 / price_yes[mask_yes] - 1.0)
    pnl_if_no[mask_yes] = -stake[mask_yes]

    pnl_if_yes[mask_no] = -stake[mask_no]
    pnl_if_no[mask_no] = stake[mask_no] * (1.0 / price_no[mask_no] - 1.0)

    # Bernoulli simulation for each trade
    n_trades = len(reco_df)
    outcomes_yes = rng.random((n_paths, n_trades)) < p_yes
    pnl_matrix = np.where(outcomes_yes, pnl_if_yes[np.newaxis, :], pnl_if_no[np.newaxis, :])
    total_pnl = pnl_matrix.sum(axis=1)
    final_bankroll = bankroll + total_pnl

    stats = {
        "final_bankroll_mean": float(final_bankroll.mean()),
        "final_bankroll_median": float(np.median(final_bankroll)),
        "final_bankroll_p5": float(np.percentile(final_bankroll, 5)),
        "final_bankroll_p1": float(np.percentile(final_bankroll, 1)),
        "final_bankroll_max": float(final_bankroll.max()),
        "final_bankroll_min": float(final_bankroll.min()),
        "pnl_mean": float(total_pnl.mean()),
        "pnl_p5": float(np.percentile(total_pnl, 5)),
        "pnl_p1": float(np.percentile(total_pnl, 1)),
    }
    return stats, final_bankroll, total_pnl


def simulate_bankroll_paths(
    df: pd.DataFrame,
    bankroll0: float,
    n_paths: int,
    kelly_fraction: float,
    max_bets_per_expiry: int,
    min_edge: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Monte Carlo bankroll simulator using fractional Kelly YES bets."""
    rng = np.random.default_rng(seed)

    model_col = get_column(df, ["p_model_cal", "p_model_fit", "p_real_mc"])
    price_col = get_column(df, ["market_price", "market_pr"])
    if model_col is None or price_col is None:
        raise ValueError("Batch file missing model or market price columns.")

    df = df.copy()
    df["edge_calc"] = compute_edge_series(df)
    df = derive_expiry_key(df)

    expiry_groups: List[Tuple[str, pd.DataFrame]] = []
    for expiry, group in df.groupby("expiry_key", observed=True):
        trades: List[Dict[str, float]] = []
        for _, row in group.iterrows():
            p = float(row[model_col])
            q = float(row[price_col])
            if not (0.0 < q < 1.0):
                continue
            yes_edge = p - q
            no_edge = q - p
            if yes_edge >= min_edge:
                trades.append(
                    {
                        "side": "YES",
                        "edge_value": yes_edge,
                        "model_prob": p,
                        "yes_price": q,
                        "entry_price": q,
                    }
                )
            if no_edge >= min_edge:
                trades.append(
                    {
                        "side": "NO",
                        "edge_value": no_edge,
                        "model_prob": p,
                        "yes_price": q,
                        "entry_price": 1.0 - q,
                    }
                )
        if not trades:
            continue
        g_trades = pd.DataFrame(trades)
        g_trades = g_trades.sort_values("edge_value", ascending=False).head(max_bets_per_expiry)
        expiry_groups.append((expiry, g_trades))

    def sort_key(item: Tuple[str, pd.DataFrame]) -> Tuple[int, str]:
        key = item[0]
        try:
            return (0, float(key))
        except ValueError:
            return (1, key)

    expiry_groups.sort(key=sort_key)

    if not expiry_groups:
        raise ValueError("No trades meet the edge criteria for the selected filters.")

    final_bankrolls = np.zeros(n_paths)
    min_bankrolls = np.full(n_paths, bankroll0, dtype=float)

    for path_idx in range(n_paths):
        B = bankroll0
        B_min = bankroll0

        for _, group in expiry_groups:
            for _, row in group.iterrows():
                p = float(row["model_prob"])
                q = float(row["yes_price"])
                side = row["side"]
                if side == "YES":
                    f_star = kelly_fraction_yes(p, q)
                    win_if_yes = True
                else:
                    f_star = kelly_fraction_no(p, q)
                    win_if_yes = False
                if f_star <= 0:
                    continue

                f_eff = min(kelly_fraction * f_star, 0.30)
                stake = f_eff * B
                if stake <= 0:
                    continue

                outcome = rng.random() < p
                win = outcome if win_if_yes else not outcome
                entry_price = float(row["entry_price"])
                pnl = stake * ((1.0 / entry_price) - 1.0) if win else -stake
                B += pnl
                B_min = min(B_min, B)
                if B <= 0:
                    B = 0
                    break
            if B <= 0:
                break

        final_bankrolls[path_idx] = B
        min_bankrolls[path_idx] = B_min

    return {
        "final": final_bankrolls,
        "min": min_bankrolls,
        "n_expiries": len(expiry_groups),
    }


# ---------------------------------------------------------------------------
# Batch ingestion helpers
# ---------------------------------------------------------------------------


def infer_pricing_date_from_source(source_name: str) -> str:
    """Infer pricing datetime from folder/filename (full UTC timestamp)."""
    # Try new format first: 2025-12-20_05-57-14_UTC or batch_20251113_094053
    new_match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_UTC", source_name)
    if new_match:
        try:
            date_part = new_match.group(1)
            time_part = new_match.group(2).replace("-", ":")
            return f"{date_part} {time_part}+00:00"
        except ValueError:
            pass
    
    # Try legacy format: batch_20251113_094053
    legacy_match = re.search(r"batch_(\d{8})_(\d{6})", source_name)
    if legacy_match:
        try:
            date_str = legacy_match.group(1)
            time_str = legacy_match.group(2)
            parsed = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return parsed.strftime("%Y-%m-%d %H:%M:%S+00:00")
        except ValueError:
            pass
    
    # Fallback: just date patterns
    patterns = [
        r"(20\d{2}-\d{2}-\d{2})",
        r"(20\d{2}\d{2}\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, source_name)
        if match:
            value = match.group(1)
            try:
                parsed = (
                    datetime.strptime(value, "%Y-%m-%d")
                    if "-" in value
                    else datetime.strptime(value, "%Y%m%d")
                )
                return parsed.strftime("%Y-%m-%d %H:%M:%S+00:00")
            except ValueError:
                continue
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")


def prepare_batch_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Normalize a single batch dataframe (pricing_date, types).
    
    Handles two formats:
    - New format: Has 'model_probability' column (from prob_backrunner_engine)
    - Old format: Has 'p_model_fit' or 'p_real_mc' columns (from fit_probability_curves)
    
    IMPORTANT: pricing_date is derived from batch_timestamp in the CSV,
    which is the source of truth for when the pricing was performed.
    """
    df = df.copy()
    
    # --- Detect and normalize NEW format (from prob_backrunner_engine) ---
    if "model_probability" in df.columns:
        # Map new column names to expected names
        rename_map = {
            "model_probability": "p_model_fit",  # Model probability
        }
        # Only rename columns that exist
        for old_col, new_col in rename_map.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure 'date' -> 'pricing_date' if present (but will be overwritten below)
        if "date" in df.columns and "pricing_date" not in df.columns:
            df = df.rename(columns={"date": "pricing_date"})
    
    # --- Use batch_timestamp as pricing_date (source of truth) ---
    if "batch_timestamp" in df.columns:
        # batch_timestamp is the exact time when pricing was performed
        df["pricing_date"] = pd.to_datetime(df["batch_timestamp"], errors="coerce", utc=True)
    else:
        # Fallback: use folder name timestamp
        df["pricing_date"] = infer_pricing_date_from_source(source_name)
    
    df["source_name"] = source_name
    return df


def load_batch_datasets(
    path_list: Tuple[str, ...],
    uploaded_files: Optional[List],
) -> Optional[pd.DataFrame]:
    """Load multiple batch CSVs (paths + uploaded files) into a single dataframe."""
    frames: List[pd.DataFrame] = []

    for path in path_list:
        path = path.strip()
        if not path:
            continue
        try:
            cache_bust = os.path.getmtime(path)
            df = load_csv(path, cache_bust)
            # Pass parent folder name for timestamp extraction
            folder_name = os.path.basename(os.path.dirname(path))
            frames.append(prepare_batch_df(df, folder_name))
        except FileNotFoundError:
            st.warning(f"Batch file not found: {path}")
        except Exception as exc:
            st.warning(f"Failed to load {path}: {exc}")

    if uploaded_files:
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                frames.append(prepare_batch_df(df, file.name))
            except Exception as exc:
                st.warning(f"Failed to read uploaded file {file.name}: {exc}")

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a column to datetime, returning NaT on failure."""
    return pd.to_datetime(series, errors="coerce")



def scan_batch_files(
    root_dir: str,
    start_date: datetime.date,
    end_date: datetime.date,
    filename: str = "batch_with_fits.csv",
) -> List[str]:
    """
    Scan root_dir for batch folders within the date range (LOCAL TIME) and return paths.
    
    Logic:
    1. Parse folder timestamps in various formats as UTC.
    2. Convert to local system time.
    3. Compare local date with start_date/end_date.
    
    Supports folder formats:
    - batch_YYYYMMDD_HHMMSS (legacy)
    - YYYY-MM-DD_HH-MM-SS_UTC (current)
    """
    valid_paths = []
    if not os.path.exists(root_dir):
        return []

    # Iterate over all subdirectories in root_dir
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        
        folder_date = None
        
        # Try new format first: 2025-12-20_05-57-14_UTC
        new_match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_UTC", entry.name)
        if new_match:
            try:
                date_part = new_match.group(1)
                time_part = new_match.group(2).replace("-", ":")
                dt_str = f"{date_part}T{time_part}+00:00"
                dt_utc = datetime.fromisoformat(dt_str)
                dt_local = dt_utc.astimezone(None)
                folder_date = dt_local.date()
            except ValueError:
                pass
        
        # Try legacy format: batch_20251114_073254
        if folder_date is None:
            legacy_match = re.search(r"batch_(\d{8}_\d{6})", entry.name)
            if legacy_match:
                try:
                    dt_utc = datetime.strptime(legacy_match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                    dt_local = dt_utc.astimezone(None)
                    folder_date = dt_local.date()
                except ValueError:
                    pass
        
        # Fallback: Try to extract just a date (YYYY-MM-DD or YYYYMMDD)
        if folder_date is None:
            iso_date_match = re.search(r"(\d{4}-\d{2}-\d{2})", entry.name)
            if iso_date_match:
                try:
                    folder_date = datetime.strptime(iso_date_match.group(1), "%Y-%m-%d").date()
                except ValueError:
                    pass
        
        if folder_date is None:
            simple_match = re.search(r"(\d{8})", entry.name)
            if simple_match:
                try:
                    folder_date = datetime.strptime(simple_match.group(1), "%Y%m%d").date()
                except ValueError:
                    continue

        if folder_date and start_date <= folder_date <= end_date:
            target_file = os.path.join(entry.path, filename)
            if os.path.exists(target_file):
                valid_paths.append(target_file)
    
    return sorted(valid_paths)


def ingest_csv(path: str, uploaded_file):
    """Utility for optional single-file uploads (regime/resolved)."""
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as exc:
            st.warning(f"Failed to load uploaded CSV: {exc}")
            return None
    if path:
        try:
            cache_bust = os.path.getmtime(path)
            return load_csv(path, cache_bust)
        except FileNotFoundError:
            st.warning(f"File not found: {path}")
            return None
    return None


def brier_score(pred: pd.Series, outcome: pd.Series) -> float:
    """Compute Brier score (mean squared error) for binary outcomes."""
    return float(np.mean((pred - outcome) ** 2))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def make_prob_plot(g: pd.DataFrame, expiry_label: str) -> go.Figure:
    """Probability vs strike plot (model / market / RN)."""
    strike = g["strike"].astype(float)
    fig = go.Figure()

    if "p_real_mc" in g.columns:
        fig.add_trace(
            go.Scatter(
                x=strike,
                y=g["p_real_mc"],
                mode="markers",
                name="MC p_real_mc",
                marker=dict(color="blue", opacity=0.6),
            )
        )

    model_col = get_column(g, ["p_model_cal", "p_model_fit", "p_real_mc_fit"])
    if model_col:
        fig.add_trace(
            go.Scatter(
                x=strike,
                y=g[model_col],
                mode="lines+markers",
                name="Model fit",
                line=dict(color="#00bcd4"),
            )
        )

    price_col = get_column(g, ["market_price", "market_pr"])
    if price_col:
        fig.add_trace(
            go.Scatter(
                x=strike,
                y=g[price_col],
                mode="markers",
                name="Market price",
                marker=dict(color="orange", symbol="square"),
            )
        )

    rn_col = get_column(g, ["p_rn_fit", "risk_neutral_prob_fit", "risk_neutral_prob", "rn_prob_fit"])
    if rn_col:
        fig.add_trace(
            go.Scatter(
                x=strike,
                y=g[rn_col],
                mode="lines",
                name="RN fit",
                line=dict(color="green", dash="dash"),
            )
        )

    fig.update_layout(
        title=f"Probabilities vs Strike | {expiry_label}",
        xaxis_title="Strike (K)",
        yaxis_title="Probability / Price",
        template="plotly_white",
    )
    return fig


def make_edge_plot(g: pd.DataFrame, min_edge: float) -> go.Figure:
    """Edge vs strike plot."""
    fig = go.Figure()
    strike = g["strike"].astype(float)
    edge = g["edge_calc"]
    fig.add_trace(
        go.Scatter(
            x=strike,
            y=edge,
            mode="markers",
            name="Edge vs market",
            marker=dict(color=np.where(edge >= min_edge, "crimson", "gray")),
        )
    )
    fig.update_layout(
        title="Edge vs Strike",
        xaxis_title="Strike",
        yaxis_title="Edge (model - market)",
        template="plotly_white",
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                y0=min_edge,
                y1=min_edge,
                line=dict(color="green", dash="dash"),
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="BTC Prediction Market Dashboard", layout="wide")
st.title("BTC Prediction Market Dashboard")

st.sidebar.header("Data Sources & Settings")

# Batch inputs (paths + uploads)
# Batch inputs (paths + uploads)
if "batch_paths" not in st.session_state:
    st.session_state["batch_paths"] = []

batch_paths = st.session_state["batch_paths"]

# --- Auto-Load Section ---
st.sidebar.subheader("Auto-Load Batches")
auto_load_root = st.sidebar.text_input("Root Directory", "")

col1, col2 = st.sidebar.columns(2)
with col1:
    today = datetime.now().date()
    default_start = today - timedelta(days=7)
    start_date = st.date_input("From", default_start)
with col2:
    end_date = st.date_input("To", today)

if st.sidebar.button("Load from Date Range"):
    found_paths = scan_batch_files(auto_load_root, start_date, end_date)
    if found_paths:
        st.session_state["batch_paths"] = found_paths
        # Update local ref immediately for this run
        batch_paths = found_paths 
        st.sidebar.success(f"Loaded {len(found_paths)} files.")
    else:
        st.sidebar.warning("No files found in that range.")

if batch_paths:
    with st.sidebar.expander("View Loaded Files"):
        for p in batch_paths:
            st.write(os.path.basename(os.path.dirname(p)))

# --- Manual Upload Section ---
st.sidebar.subheader("Manual Upload")
batch_uploads = st.sidebar.file_uploader(
    "Upload batch CSV(s)",
    type="csv",
    accept_multiple_files=True,
)

batch_all_df = load_batch_datasets(tuple(batch_paths), batch_uploads)
if batch_all_df is None or batch_all_df.empty:
    st.error("Please provide at least one batch CSV (path or upload).")
    st.stop()

batch_all_df = derive_expiry_key(batch_all_df)
if "strike" in batch_all_df.columns:
    batch_all_df["strike"] = pd.to_numeric(batch_all_df["strike"], errors="coerce")
else:
    batch_all_df["strike"] = np.nan
batch_all_df["edge_calc"] = compute_edge_series(batch_all_df)

initial_pricing_dates = sorted(batch_all_df["pricing_date"].astype(str).unique())
n_batches_loaded = len(initial_pricing_dates)
stability_filter_messages: List[str] = []
stability_filtered_df = batch_all_df.copy()

if n_batches_loaded > 1:
    strike_valid = stability_filtered_df.dropna(subset=["strike"])
    strike_counts = (
        strike_valid.groupby(["expiry_key", "strike"], observed=True)["pricing_date"].nunique()
    )
    common_pairs = strike_counts[strike_counts == n_batches_loaded]
    if common_pairs.empty:
        stability_filter_messages.append(
            "No strikes are shared across all batch files; stability analysis unavailable."
        )
        stability_filtered_df = stability_filtered_df.iloc[0:0]
    else:
        common_pairs_df = common_pairs.reset_index()[["expiry_key", "strike"]].drop_duplicates()
        before_rows = len(stability_filtered_df)
        stability_filtered_df = stability_filtered_df.merge(common_pairs_df, on=["expiry_key", "strike"], how="inner")
        removed = before_rows - len(stability_filtered_df)
        if removed > 0:
            stability_filter_messages.append(
                f"Filtered {removed} rows whose strikes were not present in every batch."
            )
        if not stability_filtered_df.empty:
            min_strikes_required = 5
            valid_counts = (
                stability_filtered_df.groupby(["expiry_key", "pricing_date"], observed=True)["strike"].nunique()
            )
            bad_expiries = (
                valid_counts[valid_counts < min_strikes_required]
                .index.get_level_values("expiry_key")
                .unique()
            )
            if len(bad_expiries) > 0:
                stability_filtered_df = stability_filtered_df[~stability_filtered_df["expiry_key"].isin(bad_expiries)].copy()
                stability_filter_messages.append(
                    "Dropped expiries with fewer than "
                    f"{min_strikes_required} shared strikes: "
                    + ", ".join(sorted(map(str, bad_expiries)))
                )

if stability_filtered_df.empty and n_batches_loaded > 1:
    stability_filter_messages.append(
        "No overlapping strikes remain after filtering; stability charts will be empty, but other tabs will still load."
    )

pricing_dates = sorted(batch_all_df["pricing_date"].astype(str).unique())
removed_dates = sorted(set(initial_pricing_dates) - set(pricing_dates))
if removed_dates:
    stability_filter_messages.append(
        "Removed pricing dates lacking sufficient shared strikes: " + ", ".join(removed_dates)
    )
default_idx = len(pricing_dates) - 1 if pricing_dates else 0
active_date = st.sidebar.selectbox("Active pricing date", pricing_dates, index=default_idx)

current_df = batch_all_df[batch_all_df["pricing_date"] == active_date].copy()
if current_df.empty:
    st.warning("No rows for the selected pricing date — using all rows instead.")
    current_df = batch_all_df.copy()
current_df = current_df.reset_index(drop=True)

positions_df = ensure_position_keys(load_positions())
open_positions_df = get_open_positions(positions_df)
open_positions_df = ensure_position_keys(open_positions_df)
# Sync open_positions.csv snapshot using the current batch
open_positions_enriched = sync_open_positions_with_batch(positions_df, current_df)
entry_notional_series = (
    pd.to_numeric(positions_df.get("entry_price"), errors="coerce")
    * pd.to_numeric(positions_df.get("size_shares"), errors="coerce")
)
capital_deployed_total = float(entry_notional_series.sum(skipna=True)) if not positions_df.empty else 0.0

# Optional regime/resolved files
regime_path = st.sidebar.text_input("Regime summary path (optional)", "")
regime_upload = st.sidebar.file_uploader("Upload regime summary", type="csv", key="regime")

resolved_default_path = Path(
    "C:/Users/Kieran Trythall/Documents/Trading/Prediction Market Contract Pricing/BTC Contract Pricing/resolved_markets.csv"
)
resolved_path = st.sidebar.text_input("Resolved markets path (optional)", str(resolved_default_path))
resolved_upload = None

st.sidebar.subheader("Stability history inputs")
default_stability_summary = os.path.join("stability_out", "stability_summary_per_day_expiry.csv")
stability_summary_path = st.sidebar.text_input(
    "Stability summary CSV path",
    default_stability_summary,
)
default_stability_drift = os.path.join("stability_out", "stability_cross_day_drift_summary.csv")
stability_drift_path = st.sidebar.text_input(
    "Drift summary CSV path",
    default_stability_drift,
)
diag_upload = st.sidebar.file_uploader(
    "Upload volatility_diagnostics.csv",
    type="csv",
    key="vol_diag_upload",
    help="Attach volatility_diagnostics to power sigma / IV ratios in the Stability tab.",
)

st.sidebar.subheader("Strategy Settings")
auto_reco_enabled = True  # Always enabled
auto_reco_min_edge = st.sidebar.number_input("Min edge", min_value=0.0, max_value=0.5, value=0.06, step=0.005)
auto_reco_max_bets = st.sidebar.number_input("Max bets per expiry", min_value=1, max_value=10, value=3, step=1)
auto_reco_max_expiry_frac = st.sidebar.slider("Max capital per expiry (frac of bankroll)", 0.05, 1.0, 0.15, 0.05)
auto_reco_max_total_frac = st.sidebar.slider("Max capital total (frac of bankroll)", 0.05, 1.0, 0.40, 0.05)
auto_reco_net_delta = st.sidebar.slider("Net Delta Limit (+/-)", 0.0, 1.0, 0.20, 0.05, help="Cap net directional exposure (Long - Short). 0.2 means max 20% net long or short.")
auto_reco_min_price = st.sidebar.number_input("Min price", min_value=0.01, max_value=0.99, value=0.03, step=0.01)
auto_reco_max_price = st.sidebar.number_input("Max price", min_value=0.01, max_value=0.99, value=0.95, step=0.01)
auto_reco_min_prob = st.sidebar.number_input("Min model prob", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
auto_reco_max_prob = st.sidebar.number_input("Max model prob", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
auto_reco_use_penalty = st.sidebar.checkbox("Use stability penalty", value=True)
auto_reco_allow_no = True
auto_reco_corr_penalty = st.sidebar.slider(
    "Correlation penalty (per expiry & direction)",
    0.0,
    1.0,
    0.25,
    0.05,
    help="0.0 = no shrink; 1.0 = strong shrink when many trades share the same expiry and direction.",
)
auto_reco_min_trade_pct = st.sidebar.slider("Min stake (% of bankroll)", 1, 10, 1, 1) / 100.0
reco_price_min = min(auto_reco_min_price, auto_reco_max_price)
reco_price_max = max(auto_reco_min_price, auto_reco_max_price)


def ingest_optional(path: str, upload):
    return ingest_csv(path, upload)


regime_df = ingest_optional(regime_path, regime_upload)
resolved_df = ingest_optional(resolved_path, resolved_upload)
resolved_path_editable = bool(resolved_path)
resolved_file_path = Path(resolved_path) if resolved_path_editable else None
if resolved_df is not None:
    resolved_df = normalize_resolved_df(resolved_df.copy())
    if not resolved_df.empty:
        resolved_df = ensure_position_keys(resolved_df)
    else:
        resolved_df = pd.DataFrame()
else:
    resolved_df = pd.DataFrame()
realized_pnl_total = compute_realized_pnl_total(positions_df, resolved_df)
if not open_positions_enriched.empty and "unrealized_pnl" in open_positions_enriched.columns:
    open_unrealized_total = float(
        pd.to_numeric(open_positions_enriched["unrealized_pnl"], errors="coerce").sum(skipna=True)
    )
else:
    open_unrealized_total = 0.0
st.sidebar.subheader("Bankroll")
kelly_fraction_sidebar = st.sidebar.slider("Kelly fraction", 0.05, 0.30, 0.15, 0.01)
use_fixed_stake = st.sidebar.checkbox("Use Fixed Stake (not Kelly)", value=False)
fixed_stake_amount = st.sidebar.number_input(
    "Fixed Stake Amount ($)",
    min_value=5.0,
    value=10.0,
    step=5.0,
    disabled=not use_fixed_stake,
    help="When enabled, all trades use this fixed dollar amount instead of Kelly sizing.",
)
bankroll_sidebar = st.sidebar.number_input(
    "Current liquid bankroll ($)",
    min_value=100.0,
    value=500.0,
    step=50.0,
    help="Used for portfolio sizing and existing position fractions.",
)
use_max_dte = st.sidebar.checkbox("Limit Max Days to Expiry", value=False)
max_dte_value = st.sidebar.number_input(
    "Max DTE (days)",
    min_value=1.0,
    value=2.0,
    step=1.0,
    disabled=not use_max_dte,
    help="Only recommend trades on contracts expiring within this many days.",
)
use_prob_threshold = st.sidebar.checkbox("Use Probability Thresholds", value=False)
prob_threshold_yes = st.sidebar.number_input(
    "Trade YES Above or Equal To",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    disabled=not use_prob_threshold,
    help="Trade YES when model probability >= this value.",
)
prob_threshold_no = st.sidebar.number_input(
    "Trade NO Below",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    disabled=not use_prob_threshold,
    help="Trade NO when model probability <= this value.",
)
use_max_moneyness = st.sidebar.checkbox("Limit Moneyness", value=False)
min_moneyness_value = st.sidebar.number_input(
    "Min |Moneyness|",
    min_value=0.0,
    max_value=0.5,
    value=0.0,
    step=0.01,
    format="%.2f",
    disabled=not use_max_moneyness,
    help="Only trade contracts where |moneyness| >= this value. Use to exclude ATM.",
)
max_moneyness_value = st.sidebar.number_input(
    "Max |Moneyness|",
    min_value=0.0,
    max_value=0.5,
    value=0.05,
    step=0.01,
    format="%.2f",
    disabled=not use_max_moneyness,
    help="Only trade contracts where |moneyness| <= this value. 0.05 = ±5% from spot.",
)
available_bankroll = bankroll_sidebar + realized_pnl_total + open_unrealized_total
stability_summary_df, stability_summary_error = try_load_dataframe(stability_summary_path)
if stability_summary_df is not None:
    if "pricing_date" not in stability_summary_df.columns:
        stability_summary_error = "Stability summary missing 'pricing_date' column."
        stability_summary_df = None
    elif "expiry_key" not in stability_summary_df.columns:
        stability_summary_error = "Stability summary missing 'expiry_key' column."
        stability_summary_df = None
    else:
        stability_summary_df = stability_summary_df.copy()
        stability_summary_df["pricing_date"] = pd.to_datetime(stability_summary_df["pricing_date"])
        stability_summary_df.sort_values(["pricing_date", "expiry_key"], inplace=True)
        stability_summary_df.reset_index(drop=True, inplace=True)
stability_drift_df, stability_drift_error = try_load_dataframe(stability_drift_path)
if stability_drift_df is not None:
    if "expiry_key" not in stability_drift_df.columns:
        stability_drift_error = "Drift summary missing 'expiry_key' column."
        stability_drift_df = None
    else:
        stability_drift_df = stability_drift_df.copy()
        stability_drift_df.sort_values("expiry_key", inplace=True)
        stability_drift_df.reset_index(drop=True, inplace=True)

vol_diag_df = None
if diag_upload is not None:
    try:
        vol_diag_df = pd.read_csv(diag_upload)
        vol_diag_df = derive_expiry_key(vol_diag_df)
    except Exception as exc:
        st.warning(f"Failed to read uploaded volatility_diagnostics.csv ({exc})")
        vol_diag_df = None

tabs = st.tabs(
    [
        "Curves & Edges",
        "Stability",
        "Volatility & Regimes",
        "Calibration",
        "Recommendations",
        "Positions",
        "Backtest",
        "Historical Stability",
    ]
)


# ---------------------------------------------------------------------------
# Curves & Edges tab
# ---------------------------------------------------------------------------

with tabs[0]:
    st.header("Curves & Edges")
    if current_df.empty:
        st.info("No rows available for the active pricing date.")
    else:
        expiry_choices = sorted(current_df["expiry_key"].unique())
        selected_expiry = st.selectbox(
            "Select Contract Expiry",
            expiry_choices,
            key="curve_expiry",
            help="The date the contract settles (e.g., '2025-12-26'). This is naturally in the future relative to the pricing date."
        )
        df_exp = current_df[current_df["expiry_key"] == selected_expiry].copy()
        if df_exp.empty:
            st.warning("No rows for the selected expiry.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(make_prob_plot(df_exp, selected_expiry), width="stretch")
            with col2:
                st.plotly_chart(make_edge_plot(df_exp, auto_reco_min_edge), width="stretch")

            st.subheader("Trades (filtered)")
            trades = df_exp[df_exp["edge_calc"] >= auto_reco_min_edge].copy().sort_values("edge_calc", ascending=False)
            price_col = get_column(df_exp, ["market_price", "market_pr"])
            model_col = get_column(df_exp, ["p_model_cal", "p_model_fit", "p_real_mc"])
            show_cols = ["strike"]
            if price_col:
                show_cols.append(price_col)
            if model_col:
                show_cols.append(model_col)
            show_cols.append("edge_calc")
            st.dataframe(trades[show_cols].rename(columns={"edge_calc": "edge"}), width="stretch")


# ---------------------------------------------------------------------------
# Stability tab
# ---------------------------------------------------------------------------

with tabs[1]:
    st.header("Stability & Diagnostics")
    if stability_filter_messages:
        for msg in stability_filter_messages:
            st.warning(msg)

    # 1) Multi-day probability stability
    if len(pricing_dates) > 1 and not current_df.empty:
        st.subheader("Multi-day probability stability")
        expiry_opts = sorted(batch_all_df["expiry_key"].unique())
        selected_expiry_md = st.selectbox("Expiry", expiry_opts, key="md_expiry")
        df_exp_all = batch_all_df[batch_all_df["expiry_key"] == selected_expiry_md]
        if df_exp_all.empty:
            st.info("No rows for the selected expiry across files.")
        else:
            contract_col = get_column(df_exp_all, ["slug"])
            if contract_col:
                contract_opts = sorted(df_exp_all[contract_col].astype(str).unique())
                selected_contract = st.selectbox("Contract", contract_opts, key="md_contract")
                df_contract = df_exp_all[df_exp_all[contract_col].astype(str) == selected_contract]
            else:
                strike_opts = sorted(df_exp_all["strike"].astype(float).unique())
                selected_strike = st.selectbox("Strike", strike_opts, key="md_strike")
                df_contract = df_exp_all[df_exp_all["strike"].astype(float) == float(selected_strike)]
            df_contract = df_contract.sort_values("pricing_date")
            dt_series = safe_to_datetime(df_contract["pricing_date"])
            model_col = get_column(df_contract, ["p_model_cal", "p_model_fit", "p_real_mc"])
            fig = go.Figure()
            if model_col:
                fig.add_trace(
                    go.Scatter(
                        x=dt_series,
                        y=df_contract[model_col],
                        mode="lines+markers",
                        name="Model fit",
                    )
                )
            price_col = get_column(df_contract, ["market_price", "market_pr"])
            if price_col:
                fig.add_trace(
                    go.Scatter(
                        x=dt_series,
                        y=df_contract[price_col],
                        mode="markers",
                        name="Market price",
                    )
                )
            rn_col = get_column(df_contract, ["risk_neutral_prob_fit", "risk_neutral_prob"])
            if rn_col:
                fig.add_trace(
                    go.Scatter(
                        x=dt_series,
                        y=df_contract[rn_col],
                        mode="lines+markers",
                        name="RN fit",
                        line=dict(dash="dot"),
                    )
                )
            fig.update_layout(
                xaxis_title="Pricing date",
                yaxis_title="Probability / Price",
                template="plotly_white",
                title="Probability drift over time",
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Load multiple batch files/dates to enable multi-day stability charts.")

    # 2) Cross-expiry curve overlay
    st.subheader("Cross-expiry curve overlay")
    cross_dates = st.multiselect(
        "Pricing dates",
        pricing_dates,
        default=[active_date],
        key="cross_dates",
    )
    if not cross_dates:
        st.warning("Select at least one pricing date.")
    else:
        df_cross = batch_all_df[batch_all_df["pricing_date"].astype(str).isin(cross_dates)].copy()
        if df_cross.empty:
            st.info("No rows for the selected pricing date.")
        else:
            exp_opts = sorted(df_cross["expiry_key"].unique())
            default_selection = exp_opts[: min(len(exp_opts), 3)]
            selected_expiries = st.multiselect(
                "Select expiries",
                exp_opts,
                default=default_selection,
                key="cross_expiries",
            )
            if not selected_expiries:
                st.warning("Select at least one expiry to overlay.")
            else:
                model_col = get_column(df_cross, ["p_model_cal", "p_model_fit", "p_real_mc"])
                if model_col is None:
                    st.info("Model column missing for this pricing date.")
                else:
                    fig_overlay = go.Figure()
                    grouped = (
                        df_cross[df_cross["expiry_key"].isin(selected_expiries)]
                        .groupby(["pricing_date", "expiry_key"])
                    )
                    for (date_label, expiry), df_group in grouped:
                        df_overlay = df_group.sort_values("strike")
                        if df_overlay.empty:
                            continue
                        trace_name = f"{expiry} | {date_label}"
                        fig_overlay.add_trace(
                            go.Scatter(
                                x=df_overlay["strike"],
                                y=df_overlay[model_col],
                                mode="lines+markers",
                                name=trace_name,
                            )
                        )
                    fig_overlay.update_layout(
                        xaxis_title="Strike",
                        yaxis_title="Model probability",
                        template="plotly_white",
                        title="Model probability vs strike (selected pricing dates)",
                    )
                    st.plotly_chart(fig_overlay, width="stretch")

    # 3) Logistic fit vs raw MC
    st.subheader("Logistic fit vs raw MC")
    fit_date = st.selectbox("Pricing date (fit)", pricing_dates, index=pricing_dates.index(active_date), key="fit_date")
    df_fit_root = batch_all_df[batch_all_df["pricing_date"] == fit_date]
    exp_fit_opts = sorted(df_fit_root["expiry_key"].unique())
    fit_expiry = st.selectbox("Expiry (fit)", exp_fit_opts, key="fit_expiry")
    df_fit = df_fit_root[df_fit_root["expiry_key"] == fit_expiry].sort_values("strike")
    model_col = get_column(df_fit, ["p_model_cal", "p_model_fit"])
    if df_fit.empty or model_col is None or "p_real_mc" not in df_fit.columns:
        st.info("No logistic fit column (p_model_fit) found — run curve fitting first.")
    else:
        fig_fit = go.Figure()
        fig_fit.add_trace(
            go.Scatter(
                x=df_fit["strike"],
                y=df_fit["p_real_mc"],
                mode="markers",
                name="MC p_real_mc",
            )
        )
        fig_fit.add_trace(
            go.Scatter(
                x=df_fit["strike"],
                y=df_fit[model_col],
                mode="lines",
                name="Model fit",
            )
        )
        fig_fit.update_layout(
            xaxis_title="Strike",
            yaxis_title="Probability",
            title="Raw MC vs logistic fit",
            template="plotly_white",
        )
        st.plotly_chart(fig_fit, width="stretch")

        vals = df_fit[model_col].astype(float).values
        mono_violations = int(np.sum(np.diff(vals) > 1e-6))
        large_residuals = int(np.sum(np.abs(df_fit["p_real_mc"] - df_fit[model_col]) > 0.1))
        met1, met2 = st.columns(2)
        with met1:
            st.metric("Monotonicity violations", mono_violations)
        with met2:
            st.metric("Large residuals (>|0.1|)", large_residuals)

    # 4) RN → Model → Market triangle
    st.subheader("RN → Model → Market comparison")
    rn_date = st.selectbox("Pricing date (RN)", pricing_dates, index=pricing_dates.index(active_date), key="rn_date")
    df_rn_root = batch_all_df[batch_all_df["pricing_date"] == rn_date]
    rn_exp_opts = sorted(df_rn_root["expiry_key"].unique())
    rn_expiry = st.selectbox("Expiry (RN)", rn_exp_opts, key="rn_expiry")
    df_rn = df_rn_root[df_rn_root["expiry_key"] == rn_expiry].sort_values("strike")
    model_col = get_column(df_rn, ["p_model_cal", "p_model_fit", "p_real_mc"])
    rn_col = get_column(df_rn, ["risk_neutral_prob_fit", "risk_neutral_prob"])
    price_col = get_column(df_rn, ["market_price", "market_pr"])
    if df_rn.empty or model_col is None or rn_col is None or price_col is None:
        st.info("Need model, RN, and market columns for triangle comparison.")
    else:
        fig_triangle = go.Figure()
        fig_triangle.add_trace(
            go.Scatter(x=df_rn["strike"], y=df_rn[model_col], mode="lines", name="Model", line=dict(color="#00bcd4"))
        )
        fig_triangle.add_trace(
            go.Scatter(x=df_rn["strike"], y=df_rn[rn_col], mode="lines", name="RN", line=dict(dash="dash"))
        )
        fig_triangle.add_trace(
            go.Scatter(
                x=df_rn["strike"],
                y=df_rn[price_col],
                mode="markers",
                name="Market",
            )
        )
        fig_triangle.update_layout(
            xaxis_title="Strike",
            yaxis_title="Probability",
            title="RN vs Model vs Market",
            template="plotly_white",
        )
        st.plotly_chart(fig_triangle, width="stretch")

        diff_stats = {
            "|Model - RN| mean": np.nanmean(np.abs(df_rn[model_col] - df_rn[rn_col])),
            "|Market - RN| mean": np.nanmean(np.abs(df_rn[price_col] - df_rn[rn_col])),
            "|Model - Market| mean": np.nanmean(np.abs(df_rn[model_col] - df_rn[price_col])),
            "Count |Model - RN| > 0.15": int(np.sum(np.abs(df_rn[model_col] - df_rn[rn_col]) > 0.15)),
        }
        st.table(pd.DataFrame.from_dict(diff_stats, orient="index", columns=["Value"]))
        if diff_stats["|Model - Market| mean"] >= 0.1:
            st.warning("⚠️ Mean |Model − Market| exceeds 0.1 — check volatility calibration.")

    # 5) Edge histogram
    st.subheader("Edge histogram")
    if current_df.empty:
        st.info("No edges to plot.")
    else:
        fig_hist = px.histogram(
            current_df,
            x="edge_calc",
            nbins=40,
            template="plotly_white",
            title="Edge distribution",
        )
        fig_hist.update_layout(xaxis_title="Edge", yaxis_title="Count")
        st.plotly_chart(fig_hist, width="stretch")
        edges = current_df["edge_calc"].dropna()
        if not edges.empty:
            out_of_band = float(((edges < -0.20) | (edges > 0.20)).mean())
            if out_of_band > 0.35:
                st.warning(
                    f"⚠️ {out_of_band:.0%} of edges fall outside [-0.20, 0.20]; model may be over-confident."
                )

    # 6) Monotonicity summary (current date)
    st.subheader("Model Curve Quality")
    st.caption("Checks if model probabilities decrease smoothly as strike prices increase (expected behavior)")
    model_col = get_column(current_df, ["p_model_cal", "p_model_fit", "p_real_mc"])
    if model_col is None:
        st.info("Model column missing for curve quality check.")
    else:
        violations = []
        for expiry, g in current_df.groupby("expiry_key", observed=True):
            vals = g.sort_values("strike")[model_col].values
            if np.any(np.diff(vals) > 1e-6):
                violations.append(expiry)
        if violations:
            st.warning(f"⚠️ {len(violations)} expiries have non-monotonic curves (probabilities not decreasing smoothly):")
            st.write(", ".join(violations))
        else:
            st.success("✅ All expiries have smooth, monotonically decreasing probability curves.")

    # 7) Sigma / Deribit IV ratio (current date)
    st.subheader("Sigma / Deribit IV ratio")
    if current_df.empty and (vol_diag_df is None or vol_diag_df.empty):
        st.info("Load batch data and/or upload volatility_diagnostics.csv to view sigma/IV ratios.")
    else:
        expiry_opts = sorted(current_df["expiry_key"].dropna().unique()) if not current_df.empty else []
        if not expiry_opts and vol_diag_df is not None and "expiry_key" in vol_diag_df.columns:
            expiry_opts = sorted(vol_diag_df["expiry_key"].dropna().astype(str).unique())
        if not expiry_opts and vol_diag_df is not None and "expiry_date" in vol_diag_df.columns:
            expiry_opts = sorted(vol_diag_df["expiry_date"].dropna().astype(str).unique())
        if not expiry_opts:
            st.info("No expiries available to match against volatility diagnostics.")
        else:
            selected_ratio_expiry = st.selectbox(
                "Expiry (for sigma/IV ratio)",
                expiry_opts,
                key="sigma_iv_expiry",
            )
            ratio_df = None
            # Prefer diagnostics (regime-level)
            if vol_diag_df is not None and not vol_diag_df.empty:
                diag_subset = vol_diag_df.copy()
                mask = pd.Series(False, index=diag_subset.index)
                sel_str = str(selected_ratio_expiry)
                if "expiry_key" in diag_subset.columns:
                    mask |= diag_subset["expiry_key"].astype(str) == sel_str
                if "expiry_date" in diag_subset.columns:
                    mask |= diag_subset["expiry_date"].astype(str) == sel_str
                if not mask.any() and "T_days" in diag_subset.columns:
                    try:
                        sel_t = float(selected_ratio_expiry)
                        mask |= np.isclose(
                            pd.to_numeric(diag_subset["T_days"], errors="coerce"),
                            sel_t,
                            atol=1e-3,
                        )
                    except Exception:
                        pass
                diag_subset = diag_subset[mask]
                if not diag_subset.empty and {"regime_weighted_sigma", "sigma_iv_daily"} <= set(diag_subset.columns):
                    ratio_df = diag_subset.dropna(subset=["regime_weighted_sigma", "sigma_iv_daily"]).copy()
            if ratio_df is not None and not ratio_df.empty:
                ratio_df["sigma_iv_ratio"] = ratio_df["regime_weighted_sigma"] / ratio_df["sigma_iv_daily"]
                ratio_df["sigma_iv_diff"] = ratio_df["regime_weighted_sigma"] - ratio_df["sigma_iv_daily"]
                display_cols = [
                    c
                    for c in [
                        "regime",
                        "sigma_final",
                        "sigma_iv_daily",
                        "regime_weighted_sigma",
                        "sigma_iv_ratio",
                        "sigma_iv_diff",
                    ]
                    if c in ratio_df.columns
                ]
                st.dataframe(ratio_df[display_cols], use_container_width=True)
            else:
                st.info("No sigma/IV data found for the selected expiry; upload matching volatility_diagnostics.csv.")


# ---------------------------------------------------------------------------
# Regimes & Vol tab
# ---------------------------------------------------------------------------

with tabs[2]:
    st.header("Regime & Volatility diagnostics")
    if regime_df is None or regime_df.empty:
        st.info("Load a regime_summary.csv to view regime diagnostics.")
    else:
        if "regime" not in regime_df.columns:
            st.warning("Regime CSV missing 'regime' column.")
        else:
            agg = regime_df.copy()
            if "probability" in agg.columns:
                prob_fig = px.bar(
                    agg,
                    x="regime",
                    y="probability",
                    title="Regime probabilities",
                    template="plotly_white",
                )
                st.plotly_chart(prob_fig, width="stretch")
            metric_cols = [c for c in ["var_scale", "tail_scale", "jump_lambda", "jump_sigma"] if c in agg.columns]
            if metric_cols:
                st.subheader("Per-regime metrics")
                st.dataframe(agg[["regime"] + metric_cols], width="stretch")
            else:
                st.info("No variance/tail columns found in regime CSV.")

    if resolved_df is not None and not resolved_df.empty:
        needed = {"p_model_fit_at_trade", "outcome", "regime_at_trade"}
        if needed.issubset(resolved_df.columns):
            st.subheader("Regime-specific calibration (resolved trades)")
            reg_data = resolved_df.dropna(subset=list(needed)).copy()
            if reg_data.empty:
                st.info("Resolved dataset empty after filtering for regime calibration.")
            else:
                rows = []
                for regime, sub in reg_data.groupby("regime_at_trade", observed=True):
                    mean_error = float(np.mean(sub["p_model_fit_at_trade"] - sub["outcome"]))
                    brier = brier_score(sub["p_model_fit_at_trade"], sub["outcome"])
                    rows.append({"regime": regime, "mean_error": mean_error, "brier": brier})
                reg_bias_df = pd.DataFrame(rows)
                if not reg_bias_df.empty:
                    col_bias, col_brier = st.columns(2)
                    with col_bias:
                        fig_bias = px.bar(
                            reg_bias_df,
                            x="regime",
                            y="mean_error",
                            title="Mean (model − outcome) per regime",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_bias, width="stretch")
                    with col_brier:
                        fig_brier = px.bar(
                            reg_bias_df,
                            x="regime",
                            y="brier",
                            title="Brier score per regime",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_brier, width="stretch")
                else:
                    st.info("No regime calibration rows to display.")
        else:
            st.info("Resolved dataset lacks 'regime_at_trade' for regime-specific analysis.")


# ---------------------------------------------------------------------------
# Calibration tab
# ---------------------------------------------------------------------------

with tabs[3]:
    st.header("Calibration")
    closed_positions = positions_df[
        positions_df["status"].astype(str).str.upper() == "CLOSED"
    ].copy()
    closed_positions = ensure_position_keys(closed_positions)
    resolved_keys = set(resolved_df["position_key"].dropna()) if not resolved_df.empty else set()
    pending_positions = closed_positions[
        ~closed_positions["position_key"].isin(resolved_keys)
    ]

    st.subheader("Record outcomes for closed positions")
    if pending_positions.empty:
        st.info("No closed positions are awaiting outcome logging.")
    elif not resolved_path_editable:
        st.info("Provide a resolved_markets CSV path (not upload) to record outcomes automatically.")
    else:
        options = pending_positions["position_key"].tolist()
        labels = {
            key: f"{row['slug']} | {row['side']} | {row.get('expiry_key') or row.get('expiry_date')}"
            for key, row in pending_positions.set_index("position_key").iterrows()
        }
        selected_key = st.selectbox(
            "Select a closed position",
            options,
            format_func=lambda key: labels.get(key, key),
        )
        selected_row = pending_positions[pending_positions["position_key"] == selected_key].iloc[0]
        outcome_choice = st.radio(
            "Outcome",
            ("YES", "NO"),
            horizontal=True,
            key=f"outcome_{selected_key}",
        )
        notes_input = st.text_input("Outcome notes (optional)", key=f"notes_{selected_key}")
        if st.button("Append outcome to resolved_markets.csv"):
            entry = {
                "position_key": selected_row["position_key"],
                "slug": selected_row.get("slug"),
                "side": selected_row.get("side"),
                "expiry_date": selected_row.get("expiry_date") or selected_row.get("expiry_key"),
                "strike": selected_row.get("strike"),
                "entry_price": selected_row.get("entry_price"),
                "size_shares": selected_row.get("size_shares"),
                "p_model_fit_at_trade": selected_row.get("model_prob_at_entry"),
                "market_price_at_trade": selected_row.get("entry_price"),
                "outcome": 1 if outcome_choice == "YES" else 0,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "notes": notes_input,
            }
            append_resolved_entry(resolved_file_path, entry)
            st.success("Outcome recorded.")
            st.rerun()

    if resolved_df is None or resolved_df.empty:
        st.info("Load resolved_markets.csv to view calibration metrics.")
    else:
        req_cols = ["p_model_fit_at_trade", "market_price_at_trade", "outcome"]
        missing = [col for col in req_cols if col not in resolved_df.columns]
        if missing:
            st.warning(f"Resolved CSV missing columns: {missing}")
        else:
            df_res = resolved_df.dropna(subset=req_cols).copy()
            if df_res.empty:
                st.info("Resolved dataset empty after filtering required columns.")
            else:
                bins = np.linspace(0, 1, 11)
                df_res["prob_bin"] = pd.cut(df_res["p_model_fit_at_trade"], bins, include_lowest=True)
                reliability = (
                    df_res.groupby("prob_bin", observed=False)
                    .agg(
                        predicted=("p_model_fit_at_trade", "mean"),
                        actual=("outcome", "mean"),
                        count=("outcome", "size"),
                    )
                    .dropna()
                )
                fig_rel = go.Figure()
                fig_rel.add_trace(
                    go.Bar(
                        x=[str(b) for b in reliability.index],
                        y=reliability["predicted"],
                        name="Predicted",
                    )
                )
                fig_rel.add_trace(
                    go.Bar(
                        x=[str(b) for b in reliability.index],
                        y=reliability["actual"],
                        name="Actual",
                    )
                )
                fig_rel.update_layout(
                    title="Reliability diagram",
                    xaxis_title="Probability bin",
                    yaxis_title="Frequency",
                    barmode="group",
                    template="plotly_white",
                )
                st.plotly_chart(fig_rel, width="stretch")

                scores = {
                    "Model": brier_score(df_res["p_model_fit_at_trade"], df_res["outcome"]),
                    "Market": brier_score(df_res["market_price_at_trade"], df_res["outcome"]),
                }
                rn_trade_col = get_column(df_res, ["rn_prob_at_trade"])
                if rn_trade_col:
                    scores["RN"] = brier_score(df_res[rn_trade_col], df_res["outcome"])
                st.subheader("Brier scores")
                st.table(pd.DataFrame.from_dict(scores, orient="index", columns=["Brier score"]))
                st.subheader("Logged trades")
                cols_to_show = [
                    "position_key",
                    "slug",
                    "side",
                    "expiry_date",
                    "strike",
                    "p_model_fit_at_trade",
                    "market_price_at_trade",
                    "outcome",
                    "recorded_at",
                    "notes",
                ]
                display_cols = [c for c in cols_to_show if c in df_res.columns]
                display_df = df_res[display_cols]
                if "recorded_at" in display_df.columns:
                    display_df = display_df.sort_values("recorded_at", ascending=False)
                st.dataframe(display_df, width="stretch", height=350)


# ---------------------------------------------------------------------------
# Auto reco tab
# ---------------------------------------------------------------------------

with tabs[4]:
    st.header("Auto Recommendations")
    if not auto_reco_enabled:
        st.info("Enable auto recommendations in the sidebar to generate trade suggestions.")
    elif current_df.empty:
        st.info("No rows for the selected pricing date.")
    else:
        try:
            effective_bankroll = max(available_bankroll, 0.0)
            min_trade_dollars = effective_bankroll * auto_reco_min_trade_pct
            reco_list = recommend_trades(
                current_df,
                bankroll=effective_bankroll,
                positions_df=open_positions_enriched,
                kelly_fraction=kelly_fraction_sidebar,
                min_edge=auto_reco_min_edge,
                max_bets_per_expiry=auto_reco_max_bets,
                max_capital_per_expiry_frac=auto_reco_max_expiry_frac,
                max_capital_total_frac=auto_reco_max_total_frac,
                max_net_delta_frac=auto_reco_net_delta,
                min_price=reco_price_min,
                max_price=reco_price_max,
                min_model_prob=auto_reco_min_prob,
                max_model_prob=auto_reco_max_prob,
                require_active=True,
                use_stability_penalty=auto_reco_use_penalty,
                allow_no=auto_reco_allow_no,
                correlation_penalty=auto_reco_corr_penalty,
                min_trade_usd=min_trade_dollars,
                disable_staleness=False,
                use_fixed_stake=use_fixed_stake,
                fixed_stake_amount=fixed_stake_amount,
                max_dte=max_dte_value if use_max_dte else None,
                use_prob_threshold=use_prob_threshold,
                prob_threshold_yes=prob_threshold_yes,
                prob_threshold_no=prob_threshold_no,
                max_moneyness=max_moneyness_value if use_max_moneyness else None,
                min_moneyness=min_moneyness_value if use_max_moneyness else None,
            )
        except ValueError as exc:
            st.warning(f"Auto recommendations unavailable: {exc}")
            reco_list = []

        if not reco_list:
            st.info("No trades met the current filters and risk caps.")
        else:
            reco_df = recommendations_to_dataframe(reco_list)
            total_stake = float(reco_df["suggested_stake"].sum())
            total_ev = float(reco_df["expected_value_dollars"].sum())
            avg_edge = float(reco_df["edge"].mean()) if "edge" in reco_df else float("nan")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Capital deployed", f"${total_stake:,.2f}")
            with col_b:
                st.metric("Expected value", f"${total_ev:,.2f}")
            with col_c:
                st.metric("Average edge", f"{avg_edge:.2%}" if not np.isnan(avg_edge) else "n/a")

            display_cols = [
                "slug",
                "expiry_key",
                "strike",
                "side",
                "market_price",
                "entry_price",
                "model_prob",
                "rn_prob",
                "edge",
                "kelly_fraction_applied",
                 "kelly_target",
                 "kelly_existing",
                "suggested_stake",
                "expected_value_dollars",
                "expiry_group_risk",
                "notes",
            ]
            available_cols = [c for c in display_cols if c in reco_df.columns]
            table = reco_df[available_cols].copy()
            rename_map = {
                "slug": "Contract",
                "expiry_key": "Expiry",
                "strike": "Strike",
                "side": "Side",
                "market_price": "Market Price",
                "entry_price": "Entry Price",
                "model_prob": "Model Prob",
                "rn_prob": "RN Prob",
                "edge": "Edge",
                "kelly_fraction_applied": "Kelly Used",
                "kelly_target": "Kelly Target",
                "kelly_existing": "Kelly Existing",
                "suggested_stake": "Stake ($)",
                "expected_value_dollars": "EV ($)",
                "expiry_group_risk": "Expiry Risk",
                "notes": "Notes",
            }
            st.dataframe(table.rename(columns=rename_map), width="stretch")

            chart_df = reco_df.copy()
            chart_df["label"] = chart_df["slug"].fillna(chart_df["question"]).astype(str)
            fig_reco = px.bar(
                chart_df,
                x="label",
                y="expected_value_dollars",
                color="expiry_key",
                title="Expected $ Return by Trade",
                template="plotly_white",
            )
            fig_reco.update_layout(xaxis_title="Contract", yaxis_title="Expected Value ($)")
            st.plotly_chart(fig_reco, width="stretch")

            st.markdown("### Simulate Today's Recommended Portfolio")
            prob_col = get_column(reco_df, ["model_prob", "p_model_cal", "p_model_fit", "p_real_mc"])
            price_col = get_column(reco_df, ["market_price", "market_pr"])
            side_col = get_column(reco_df, ["side", "polymarket_outcome"])
            stake_col = get_column(reco_df, ["suggested_stake", "stake_dollars", "stake_usd", "raw_stake"])

            if not all([prob_col, price_col, side_col, stake_col]):
                st.info("Simulation unavailable: missing probability, price, side, or stake data.")
            else:
                sim_col1, sim_col2 = st.columns(2)
                with sim_col1:
                    sim_paths = st.number_input(
                        "Number of Monte Carlo paths",
                        min_value=100,
                        max_value=20000,
                        value=5000,
                        step=500,
                    )
                with sim_col2:
                    sim_seed = st.number_input(
                        "Random seed (optional)",
                        min_value=0,
                        max_value=10_000_000,
                        value=0,
                        step=1,
                    )
                if st.button("Run Simulation on Today's Trades"):
                    try:
                        stats, final_bankroll_sim, total_pnl_sim = simulate_today_portfolio(
                            reco_df,
                            effective_bankroll,
                            prob_col=prob_col,
                            price_col=price_col,
                            side_col=side_col,
                            stake_col=stake_col,
                            n_paths=int(sim_paths),
                            seed=int(sim_seed) if sim_seed > 0 else None,
                        )
                        st.write(f"Initial bankroll: ${effective_bankroll:,.2f}")
                        st.write(f"Mean final bankroll: ${stats['final_bankroll_mean']:,.2f}")
                        st.write(f"Median final bankroll: ${stats['final_bankroll_median']:,.2f}")
                        st.write(f"5th percentile final bankroll: ${stats['final_bankroll_p5']:,.2f}")
                        st.write(f"1st percentile final bankroll: ${stats['final_bankroll_p1']:,.2f}")
                        st.write(f"Max / Min final bankroll: ${stats['final_bankroll_max']:,.2f} / ${stats['final_bankroll_min']:,.2f}")
                        st.write(f"Mean PnL: ${stats['pnl_mean']:,.2f}")
                        st.write(f"5th / 1st percentile PnL: ${stats['pnl_p5']:,.2f} / ${stats['pnl_p1']:,.2f}")

                        fig_hist_final = px.histogram(
                            final_bankroll_sim,
                            nbins=40,
                            title="Distribution of Final Bankroll (Today's Portfolio)",
                            labels={"value": "Final bankroll"},
                        )
                        st.plotly_chart(fig_hist_final, width="stretch")

                        fig_hist_pnl = px.histogram(
                            total_pnl_sim,
                            nbins=40,
                            title="Distribution of PnL (Today's Portfolio)",
                            labels={"value": "PnL ($)"},
                        )
                        st.plotly_chart(fig_hist_pnl, width="stretch")
                    except ValueError as sim_err:
                        st.warning(str(sim_err))

        with st.expander("Candidate diagnostics", expanded=False):
            debug_df = LAST_RECO_DEBUG
            if debug_df is not None and not debug_df.empty:
                diag_cols = [
                    "expiry_key",
                    "strike",
                    "side",
                    "score",
                    "direction",
                    "expiry_shape_label",
                    "shape_selected",
                    "stability_penalty",
                    "stale_mult",
                    "price_staleness_mult",
                    "kelly_full",
                    "kelly_full_effective",
                    "kelly_target",
                    "kelly_existing",
                    "kelly_eff",
                ]
                available = [c for c in diag_cols if c in debug_df.columns]
                st.dataframe(debug_df[available], width="stretch")
            else:
                st.write("No candidate diagnostics available.")

# ---------------------------------------------------------------------------
# Backtest tab
# ---------------------------------------------------------------------------

with tabs[6]:
    st.header("Backtest (Auto-Reco replay)")
    if batch_all_df.empty:
        st.info("Load batch CSVs to enable backtesting.")
    else:
        initial_bankroll_bt = st.number_input(
            "Initial bankroll ($)",
            min_value=0.0,
            value=bankroll_sidebar,
            step=50.0,
            help="Starting bankroll for the backtest.",
        )
        # Try to load BTC intraday prices for outcome inference
        price_df_bt = None
        default_price_path = Path("DATA") / "btc_intraday_1m.csv"
        if default_price_path.exists():
            try:
                price_df_bt = pd.read_csv(default_price_path)
            except Exception:
                price_df_bt = None
        else:
            st.info("Optional: place btc_intraday_1m.csv in DATA/ to infer outcomes for expired contracts.")
        run_button = st.button("Run backtest")
        if run_button:
            try:
                daily_batches = []
                for date_val, g in batch_all_df.groupby("pricing_date", observed=True):
                    daily_batches.append(g.copy())
                strategy_params = {
                    "kelly_fraction": kelly_fraction_sidebar,
                    "min_edge": auto_reco_min_edge,
                    "max_bets_per_expiry": auto_reco_max_bets,
                    "max_capital_per_expiry_frac": auto_reco_max_expiry_frac,
                    "max_capital_total_frac": auto_reco_max_total_frac,
                    "max_net_delta_frac": auto_reco_net_delta,
                    "min_price": reco_price_min,
                    "max_price": reco_price_max,
                    "min_model_prob": auto_reco_min_prob,
                    "max_model_prob": auto_reco_max_prob,
                    "require_active": False,  # Must be False for historical backtesting
                    "use_stability_penalty": auto_reco_use_penalty,
                    "allow_no": auto_reco_allow_no,
                    "correlation_penalty": auto_reco_corr_penalty,
                    "min_trade_usd": None,
                    "min_trade_frac": auto_reco_min_trade_pct,
                    "disable_staleness": True,
                    "use_fixed_stake": use_fixed_stake,
                    "fixed_stake_amount": fixed_stake_amount,
                    "max_dte": max_dte_value if use_max_dte else None,
                    "use_prob_threshold": use_prob_threshold,
                    "prob_threshold_yes": prob_threshold_yes,
                    "prob_threshold_no": prob_threshold_no,
                    "max_moneyness": max_moneyness_value if use_max_moneyness else None,
                    "min_moneyness": min_moneyness_value if use_max_moneyness else None,
                }
                trades_bt, equity_bt, all_priced_bt = run_backtest(
                    daily_batches, initial_bankroll_bt, strategy_params, 
                    price_df=price_df_bt, return_all_priced=True
                )
                st.session_state["bt_trades"] = trades_bt
                st.session_state["bt_equity"] = equity_bt
                st.session_state["bt_all_priced"] = all_priced_bt
                st.session_state["bt_initial"] = initial_bankroll_bt
                # Show debug info
                n_settled = len(trades_bt[trades_bt['settled']==True]) if 'settled' in trades_bt.columns and not trades_bt.empty else 0
                n_all_priced = len(all_priced_bt) if all_priced_bt is not None and not all_priced_bt.empty else 0
                st.info(f"Backtest complete: {len(daily_batches)} batches processed, {len(trades_bt)} trades, {n_settled} settled, {n_all_priced} total priced contracts")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                import traceback
                st.code(traceback.format_exc())
        # Show last backtest results (persisted in session_state) with toggleable settled filter
        trades_bt = st.session_state.get("bt_trades")
        equity_bt = st.session_state.get("bt_equity")
        initial_bt = st.session_state.get("bt_initial", initial_bankroll_bt)
        if trades_bt is not None and isinstance(trades_bt, pd.DataFrame):
            # Trade view toggle
            st.subheader("Backtest Trades")
            trades_view = trades_bt.copy()
            
            # Add Settled checkmark to trades_view manually since we might have consolidated (but dashboard logic was reverted by user)
            # Actually, the user reverted my checkmark change but the REQUEST is to not have separate rows.
            # My backtest_engine change ensures NO separate rows.
            # So here I just display everything.
            
            if "settled" in trades_view.columns:
                 trades_view["Settled?"] = trades_view["settled"].apply(lambda x: "✅" if x else "")

            
            # --- Fix Price Display (Entry Price vs YES Price) ---
            if "market_price" in trades_view.columns:
                trades_view = trades_view.rename(columns={"market_price": "yes_price"})
            
            # Create 'Market Price' (Entry Price / Price Paid)
            # Default to yes_price
            if "yes_price" in trades_view.columns:
                trades_view["Market Price"] = trades_view["yes_price"]
                if "side" in trades_view.columns:
                    mask_no = trades_view["side"].astype(str).str.upper() == "NO"
                    trades_view.loc[mask_no, "Market Price"] = 1.0 - trades_view.loc[mask_no, "yes_price"]
            
            # Reorder columns for clarity
            cols_order = ["pricing_date", "expiry_date", "Settled?", "slug", "side", "Market Price", "yes_price", "outcome_yes", "pnl", "stake"]
            cols_present = [c for c in cols_order if c in trades_view.columns]
            cols_rest = [c for c in trades_view.columns if c not in cols_present and c != "settled"] # hide raw boolean if we have checkmark
            trades_view = trades_view[cols_present + cols_rest]

            if trades_view.empty:
                st.info("No settled trades to display. Ensure expiry dates are covered by the price file.")
            else:
                st.caption("'Market Price' shows the price paid for the specific contract side (YES or NO).")
                st.metric("Settled trades", len(trades_view))
                st.dataframe(trades_view, width="stretch", height=350)
                # Equity curve from settled trades only: initial bankroll + cumulative settled PnL
                if "pricing_date" in trades_view.columns and "pnl" in trades_view.columns:
                    settled_eq = trades_view.sort_values("pricing_date")[["pricing_date", "pnl"]].copy()
                    settled_eq["bankroll"] = initial_bt + settled_eq["pnl"].cumsum()
                    eq_fig_settled = px.line(
                        settled_eq,
                        x="pricing_date",
                        y="bankroll",
                        title="Equity curve (settled trades only)",
                        template="plotly_white",
                    )
                    st.plotly_chart(eq_fig_settled, width="stretch")
                
                # Trades by Expiry
                st.subheader("Trades by Expiry")
                st.caption("Groups trades by contract expiration date, not by when the trade was placed.")
                if "expiry_date" in trades_view.columns:
                    # Prefer expiry_key if available for cleaner grouping, else expiry_date
                    group_col = "expiry_key" if "expiry_key" in trades_view.columns else "expiry_date"
                    
                    # Check for trades with missing expiry data
                    total_trades = len(trades_view)
                    trades_with_expiry = trades_view[group_col].notna().sum()
                    missing_count = total_trades - trades_with_expiry
                    if missing_count > 0:
                        st.warning(f"⚠️ {missing_count} trade(s) have missing expiry dates and are not shown in this table.")
                    
                    # Group and count
                    expiry_counts = trades_view.groupby(group_col, observed=True).size().reset_index(name="Trades Taken")
                    
                    # If we have PnL, let's sum it too because why not
                    if "pnl" in trades_view.columns:
                        expiry_pnl = trades_view.groupby(group_col, observed=True)["pnl"].sum().reset_index(name="Total PnL")
                        expiry_counts = pd.merge(expiry_counts, expiry_pnl, on=group_col)
                    
                    expiry_counts = expiry_counts.sort_values(group_col)
                    st.dataframe(expiry_counts, width="stretch")
                else:
                    st.info("No expiry date column found to aggregate trades.")

                # Moneyness histogram for taken trades
                moneyness_col = get_column(trades_view, ["moneyness"])
                if moneyness_col is not None:
                    st.subheader("Moneyness Distribution")
                    tv_moneyness = trades_view.copy()
                    tv_moneyness["abs_moneyness"] = pd.to_numeric(tv_moneyness[moneyness_col], errors="coerce").abs()
                    tv_moneyness = tv_moneyness[tv_moneyness["abs_moneyness"].notna()]
                    if not tv_moneyness.empty:
                        fig_moneyness = px.histogram(
                            tv_moneyness,
                            x="abs_moneyness",
                            nbins=20,
                            title="Distribution of |Moneyness| for Traded Contracts",
                            labels={"abs_moneyness": "|Moneyness|", "count": "Trade Count"},
                            template="plotly_white",
                        )
                        fig_moneyness.update_layout(bargap=0.1)
                        st.plotly_chart(fig_moneyness, width="stretch")

                # Spearman correlation: trade direction vs momentum_6hr
                momentum_col = get_column(trades_view, ["momentum_6hr"])
                side_col = get_column(trades_view, ["side"])
                if momentum_col is not None and side_col is not None:
                    from scipy.stats import spearmanr
                    tv_corr = trades_view.copy()
                    tv_corr["_momentum"] = pd.to_numeric(tv_corr[momentum_col], errors="coerce")
                    tv_corr["_direction"] = tv_corr[side_col].apply(lambda x: 1 if str(x).upper() == "YES" else -1)
                    tv_corr = tv_corr[tv_corr["_momentum"].notna()]
                    if len(tv_corr) >= 3:
                        rho, pval = spearmanr(tv_corr["_direction"], tv_corr["_momentum"])
                        st.metric(
                            "Direction vs Momentum Correlation",
                            f"{rho:.3f}",
                            help=f"Spearman ρ between trade direction (+1=YES, -1=NO) and 6hr momentum. p-value: {pval:.4f}"
                        )
                
                # PNL by momentum sign
                momentum_col = get_column(trades_view, ["momentum_6hr"])
                if momentum_col is not None and "pnl" in trades_view.columns:
                    tv_mom = trades_view.copy()
                    tv_mom["_momentum"] = pd.to_numeric(tv_mom[momentum_col], errors="coerce")
                    tv_mom["_pnl"] = pd.to_numeric(tv_mom["pnl"], errors="coerce")
                    tv_mom = tv_mom[tv_mom["_momentum"].notna() & tv_mom["_pnl"].notna()]
                    
                    if not tv_mom.empty:
                        pos_mom_mask = tv_mom["_momentum"] > 0
                        neg_mom_mask = tv_mom["_momentum"] < 0
                        
                        pnl_positive_momentum = tv_mom.loc[pos_mom_mask, "_pnl"].sum()
                        pnl_negative_momentum = tv_mom.loc[neg_mom_mask, "_pnl"].sum()
                        n_positive_trades = pos_mom_mask.sum()
                        n_negative_trades = neg_mom_mask.sum()
                        
                        col_pos_mom, col_neg_mom = st.columns(2)
                        with col_pos_mom:
                            st.metric(
                                "PNL (Positive Momentum)",
                                f"${pnl_positive_momentum:,.2f}",
                                help=f"Total PNL from {n_positive_trades} trades with positive 6hr momentum (BTC trending up)"
                            )
                        with col_neg_mom:
                            st.metric(
                                "PNL (Negative Momentum)",
                                f"${pnl_negative_momentum:,.2f}",
                                help=f"Total PNL from {n_negative_trades} trades with negative 6hr momentum (BTC trending down)"
                            )

                # Brier scores on settled trades
                prob_col = get_column(trades_view, ["p_model_cal", "p_model_fit", "model_prob"])
                market_col = get_column(trades_view, ["market_price"])
                outcome_col = get_column(trades_view, ["outcome_yes"])
                if prob_col and outcome_col:
                    y = pd.to_numeric(trades_view[outcome_col], errors="coerce")
                    p_model = pd.to_numeric(trades_view[prob_col], errors="coerce")
                    mask = y.notna() & p_model.notna()
                    scores = {}
                    if mask.any():
                        scores["Model"] = brier_score(p_model[mask], y[mask])
                    if market_col:
                        p_mkt = pd.to_numeric(trades_view[market_col], errors="coerce")
                        mask2 = y.notna() & p_mkt.notna()
                        if mask2.any():
                            scores["Market"] = brier_score(p_mkt[mask2], y[mask2])
                    st.subheader("Brier scores (settled trades)")
                    if scores:
                        st.table(pd.DataFrame.from_dict(scores, orient="index", columns=["Brier score"]))
                    else:
                        st.info("No settled trades with both probabilities and outcomes to compute Brier scores.")
                # PnL by probability bin
                if prob_col and "pnl" in trades_view.columns and "stake" in trades_view.columns and outcome_col:
                    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
                    tv = trades_view.copy()
                    tv["p_bin"] = pd.cut(
                        pd.to_numeric(tv[prob_col], errors="coerce"),
                        bins=bins,
                        right=True,
                        include_lowest=True,
                    )
                    tv["pnl"] = pd.to_numeric(tv["pnl"], errors="coerce")
                    tv["stake"] = pd.to_numeric(tv["stake"], errors="coerce")
                    
                    # Use 'Market Price' (Entry Price) for edge calculation
                    if "Market Price" not in tv.columns and "yes_price" in tv.columns:
                        # Reconstruct if missing (should be present from viewing logic above but careful with copies)
                        tv["Market Price"] = tv["yes_price"]
                        if "side" in tv.columns:
                            mask_no = tv["side"].astype(str).str.upper() == "NO"
                            tv.loc[mask_no, "Market Price"] = 1.0 - tv.loc[mask_no, "yes_price"]

                    tv["market_price_effective"] = pd.to_numeric(tv.get("Market Price"), errors="coerce")
                    tv["outcome_yes"] = pd.to_numeric(tv[outcome_col], errors="coerce")
                    
                    # Calculate Edge: P(Win) - EntryPrice
                    p_model = pd.to_numeric(tv[prob_col], errors="coerce")
                    
                    side_col = "side" if "side" in tv.columns else None
                    if side_col:
                        p_win = np.where(tv[side_col].astype(str).str.upper() == "YES", p_model, 1.0 - p_model)
                        tv["trade_won"] = (
                            ((tv[side_col].str.upper() == "YES") & (tv["outcome_yes"] == 1)) |
                            ((tv[side_col].str.upper() == "NO") & (tv["outcome_yes"] == 0))
                        ).astype(float)
                    else:
                        # Fallback
                        p_win = p_model
                        tv["trade_won"] = tv["outcome_yes"]
                    
                    tv["edge_at_entry"] = p_win - tv["market_price_effective"]

                    agg = (
                        tv.dropna(subset=["p_bin"])
                        .groupby("p_bin", observed=True)
                        .agg(
                            trades=(prob_col, "size"),
                            avg_model_prob=(prob_col, "mean"),
                            win_rate=("trade_won", "mean"),
                            avg_edge=("edge_at_entry", "mean"), 
                            total_staked=("stake", "sum"),
                            total_pnl=("pnl", "sum"),
                        )
                    )
                    if not agg.empty:
                        agg = agg.reset_index()
                        agg["prob_bin"] = agg["p_bin"].astype(str)
                        agg["return_on_stake"] = (agg["total_pnl"] / agg["total_staked"].replace(0, np.nan) * 100).round(1)
                        # Rename and reorder columns for display
                        display_agg = agg[["prob_bin", "trades", "avg_model_prob", "win_rate", "avg_edge", "total_staked", "total_pnl", "return_on_stake"]].copy()
                        display_agg.columns = ["Model Prob Bin", "# Trades", "Avg Model Prob", "Win Rate", "Avg Edge", "Total Staked ($)", "Total PnL ($)", "Return (%)"]
                        st.subheader("PnL by Model Probability Bin")
                        st.caption("Shows performance breakdown by what the model predicted at time of entry")
                        st.dataframe(display_agg, width="stretch", hide_index=True)
                        fig_pnl = px.bar(
                            agg,
                            x="prob_bin",
                            y="return_on_stake",
                            title="Return on Stake by Model Probability",
                            labels={"prob_bin": "Model Probability Bin", "return_on_stake": "Return (%)"},
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_pnl, width="stretch")
                    # Reliability diagram (settled trades)
                    if mask.any():
                        bins = np.linspace(0, 1, 11)
                        rel_df = pd.DataFrame({"p": p_model[mask], "y": y[mask]})
                        rel_df["prob_bin"] = pd.cut(rel_df["p"], bins, include_lowest=True)
                        reliability = (
                            rel_df.groupby("prob_bin", observed=False)
                            .agg(predicted=("p", "mean"), actual=("y", "mean"), count=("y", "size"))
                            .dropna()
                        )
                        fig_rel_bt = go.Figure()
                        fig_rel_bt.add_trace(
                            go.Bar(
                                x=[str(b) for b in reliability.index],
                                y=reliability["predicted"],
                                name="Predicted",
                            )
                        )
                        fig_rel_bt.add_trace(
                            go.Bar(
                                x=[str(b) for b in reliability.index],
                                y=reliability["actual"],
                                name="Actual",
                            )
                        )
                        fig_rel_bt.update_layout(
                            title="Reliability diagram (settled trades)",
                            xaxis_title="Probability bin",
                            yaxis_title="Frequency",
                            barmode="group",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_rel_bt, width="stretch")
                
                # --- Calibration by Model Probability Bin (All Priced Contracts) ---
                all_priced_bt = st.session_state.get("bt_all_priced")
                if all_priced_bt is not None and isinstance(all_priced_bt, pd.DataFrame) and not all_priced_bt.empty:
                    # Use exact same bin edges as PnL table for direct comparability
                    calib_bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
                    ap = all_priced_bt.copy()
                    
                    # Filter to resolved contracts only
                    ap = ap[ap["outcome_yes"].notna()].copy()
                    
                    if not ap.empty:
                        ap["model_prob_used"] = pd.to_numeric(ap["model_prob_used"], errors="coerce")
                        ap["market_yes_price"] = pd.to_numeric(ap["market_yes_price"], errors="coerce")
                        ap["outcome_yes"] = pd.to_numeric(ap["outcome_yes"], errors="coerce")
                        
                        ap["prob_bin"] = pd.cut(
                            ap["model_prob_used"],
                            bins=calib_bins,
                            right=True,
                            include_lowest=True,
                        )
                        
                        # Compute squared error for Brier score
                        ap["sq_error"] = (ap["outcome_yes"] - ap["model_prob_used"]) ** 2
                        ap["model_edge"] = ap["model_prob_used"] - ap["market_yes_price"]
                        
                        calib_agg = (
                            ap.dropna(subset=["prob_bin"])
                            .groupby("prob_bin", observed=True)
                            .agg(
                                n_contracts=("model_prob_used", "size"),
                                mean_model_prob=("model_prob_used", "mean"),
                                realized_yes_rate=("outcome_yes", "mean"),
                                mean_market_price=("market_yes_price", "mean"),
                                mean_model_edge=("model_edge", "mean"),
                                brier_score=("sq_error", "mean"),
                            )
                        )
                        
                        if not calib_agg.empty:
                            calib_agg = calib_agg.reset_index()
                            calib_agg["calibration_error"] = calib_agg["realized_yes_rate"] - calib_agg["mean_model_prob"]
                            calib_agg["prob_bin_str"] = calib_agg["prob_bin"].astype(str)
                            
                            # Rename and reorder columns for display
                            display_calib = calib_agg[[
                                "prob_bin_str", "n_contracts", "mean_model_prob", "realized_yes_rate", 
                                "calibration_error", "mean_market_price", "mean_model_edge", "brier_score"
                            ]].copy()
                            display_calib.columns = [
                                "Model Prob Bin", "# Contracts", "Avg Model Prob", "Realized YES Rate",
                                "Calibration Error", "Avg Market Price", "Avg Model Edge", "Brier Score"
                            ]
                            
                            # Round for display
                            for col in ["Avg Model Prob", "Realized YES Rate", "Calibration Error", "Avg Market Price", "Avg Model Edge", "Brier Score"]:
                                display_calib[col] = display_calib[col].round(4)
                            
                            st.subheader("Calibration by Model Probability Bin (All Priced Contracts)")
                            st.caption(f"Shows calibration metrics for ALL {len(ap)} evaluated contracts (not just traded). Calibration Error = Realized YES Rate - Avg Model Prob.")
                            st.dataframe(display_calib, width="stretch", hide_index=True)
                            
                            # Bar chart for calibration error
                            fig_calib = px.bar(
                                calib_agg,
                                x="prob_bin_str",
                                y="calibration_error",
                                title="Calibration Error by Model Probability Bin (All Priced Contracts)",
                                labels={"prob_bin_str": "Model Probability Bin", "calibration_error": "Calibration Error"},
                                template="plotly_white",
                                color="calibration_error",
                                color_continuous_scale=["red", "white", "green"],
                                color_continuous_midpoint=0,
                            )
                            fig_calib.update_layout(coloraxis_showscale=False)
                            st.plotly_chart(fig_calib, width="stretch")
                            
                # ---------------------------------------------------------------
                # RESIDUAL (MOMENTUM-NEUTRAL) SIGNAL METRICS
                # ---------------------------------------------------------------
                # Diagnose whether model's predictive signal is independent of 
                # short-term momentum by regressing out momentum from model prob,
                # then computing AUC/Spearman on residuals vs realized outcomes.
                # ---------------------------------------------------------------
                
                st.subheader("Residual (Momentum-Neutral) Signal Metrics (All Priced, DTE≤2)")
                
                # --- Helper: robust column selection ---
                def _get_col(df: pd.DataFrame, candidates: list):
                    """Return first column name that exists in df, else None."""
                    for c in candidates:
                        if c in df.columns:
                            return c
                    return None
                
                # Select columns with preference order
                prob_col_residual = _get_col(all_priced_bt, [
                    "p_model_cal", "p_model_fit", "p_real_mc", 
                    "model_probability", "model_prob_used"
                ])
                market_col_residual = _get_col(all_priced_bt, [
                    "market_yes_price", "market_price", "yes_price"
                ])
                spot_col = _get_col(all_priced_bt, ["spot_price", "btc_spot", "spot"])
                outcome_col_residual = _get_col(all_priced_bt, ["outcome_yes", "outcome", "resolved_outcome"])
                time_col = _get_col(all_priced_bt, ["snapshot_time", "pricing_date", "batch_timestamp"])
                dte_col = _get_col(all_priced_bt, ["dte_days", "T_days"])
                
                # Check required columns exist
                missing_cols = []
                if prob_col_residual is None:
                    missing_cols.append("probability (p_model_cal/p_model_fit/...)")
                if market_col_residual is None:
                    missing_cols.append("market YES price (market_yes_price/market_price/...)")
                if spot_col is None:
                    missing_cols.append("spot price (spot_price/btc_spot/spot)")
                if outcome_col_residual is None:
                    missing_cols.append("outcome (outcome_yes/outcome/...)")
                if time_col is None:
                    missing_cols.append("timestamp (snapshot_time/pricing_date/...)")
                
                if missing_cols:
                    st.warning(f"Cannot compute residual metrics: missing columns: {', '.join(missing_cols)}")
                else:
                    # Work on a copy
                    resid_df = all_priced_bt.copy()
                    
                    # --- Convert and clean data ---
                    # Ensure timestamp is datetime
                    resid_df["_time"] = pd.to_datetime(resid_df[time_col], errors="coerce", utc=True)
                    resid_df["_prob"] = pd.to_numeric(resid_df[prob_col_residual], errors="coerce")
                    resid_df["_mkt"]  = pd.to_numeric(resid_df[market_col_residual], errors="coerce")
                    resid_df["_spot"] = pd.to_numeric(resid_df[spot_col], errors="coerce")
                    resid_df["_outcome"] = pd.to_numeric(resid_df[outcome_col_residual], errors="coerce")
                    
                    # Filter to DTE <= 2 (or respect max_dte UI control)
                    dte_filter_val = max_dte_value if (use_max_dte and 'max_dte_value' in dir()) else 2
                    dte_filter_val = min(dte_filter_val, 2)  # Cap at 2 for this analysis
                    if dte_col is not None:
                        resid_df["_dte"] = pd.to_numeric(resid_df[dte_col], errors="coerce")
                        resid_df = resid_df[resid_df["_dte"] <= dte_filter_val]
                    
                    # Drop rows with missing essential data
                    n_before_time_filter = len(resid_df)
                    resid_df = resid_df.dropna(subset=["_time", "_prob", "_mkt", "_spot", "_outcome"])
                    resid_df = resid_df[resid_df["_outcome"].isin([0, 1])]  # Ensure binary
                    
                    # Define trading signal as edge (this is what you actually monetize)
                    resid_df["_edge"] = resid_df["_prob"] - resid_df["_mkt"]
                    
                    if len(resid_df) < 50:
                        st.info(f"Too few samples for residual test ({len(resid_df)} after filtering). Need at least 50.")
                    elif resid_df["_outcome"].nunique() < 2:
                        st.warning("Cannot compute AUC: outcome has only one class after filtering.")
                    else:
                        # --- Compute 6-hour momentum ---
                        # Create helper df of unique (time, spot) sorted by time
                        resid_df = resid_df.sort_values("_time")
                        spot_lookup = resid_df[["_time", "_spot"]].drop_duplicates().sort_values("_time")
                        
                        # Compute t_minus_6h for each row
                        resid_df["_t_minus_6h"] = resid_df["_time"] - pd.Timedelta(hours=6)
                        
                        # Use merge_asof to find nearest prior spot price
                        # This finds the spot_6h_ago by looking up the closest time <= t_minus_6h
                        resid_df = resid_df.sort_values("_t_minus_6h")
                        spot_lookup = spot_lookup.rename(columns={"_time": "_lookup_time", "_spot": "_spot_6h_ago"})
                        resid_df = pd.merge_asof(
                            resid_df,
                            spot_lookup,
                            left_on="_t_minus_6h",
                            right_on="_lookup_time",
                            direction="backward"
                        )
                        
                        # Compute momentum: log(spot / spot_6h_ago)
                        n_before_momentum = len(resid_df)
                        resid_df = resid_df[resid_df["_spot_6h_ago"].notna() & (resid_df["_spot_6h_ago"] > 0)]
                        resid_df["_momentum_6h"] = np.log(resid_df["_spot"] / resid_df["_spot_6h_ago"])
                        
                        # Drop any remaining NaN in momentum
                        resid_df = resid_df.dropna(subset=["_momentum_6h"])
                        n_dropped_momentum = n_before_momentum - len(resid_df)
                        
                        if n_dropped_momentum > 0:
                            st.caption(f"ℹ️ {n_dropped_momentum} rows dropped due to missing 6h price history.")
                        
                        if len(resid_df) < 50:
                            st.info(f"Too few samples after momentum computation ({len(resid_df)}). Need at least 50.")
                        else:
                            # --- Residualization: Regress EDGE on momentum ---
                            # Linear regression: edge = a + b * momentum_6h + residual
                            # Edge is what you actually monetize in trading
                            # Using numpy OLS: beta = (X'X)^(-1) X'y
                            X = np.column_stack([
                                np.ones(len(resid_df)),  # Intercept
                                resid_df["_momentum_6h"].values
                            ])
                            y_signal = resid_df["_edge"].values
                            
                            # OLS: beta = (X'X)^(-1) X'y
                            try:
                                XtX_inv = np.linalg.inv(X.T @ X)
                                beta = XtX_inv @ (X.T @ y_signal)
                                intercept_a = beta[0]
                                beta_momentum = beta[1]
                                
                                # Predicted edge and residual
                                p_hat = X @ beta
                                residual = y_signal - p_hat
                                
                                # R² of regression
                                ss_res = np.sum(residual ** 2)
                                ss_tot = np.sum((y_signal - np.mean(y_signal)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                                
                                # --- Compute metrics ---
                                from scipy.stats import spearmanr
                                from sklearn.metrics import roc_auc_score
                                
                                outcome_arr = resid_df["_outcome"].values
                                
                                # Raw EDGE metrics (edge vs outcome)
                                raw_spearman, raw_spearman_pval = spearmanr(y_signal, outcome_arr)
                                raw_auc = roc_auc_score(outcome_arr, y_signal)
                                
                                # Residual signal metrics (residual vs outcome)
                                # Residuals are ranking signals, not probabilities - compute AUC directly
                                # Do NOT apply sigmoid/logit to residual
                                resid_spearman, resid_spearman_pval = spearmanr(residual, outcome_arr)
                                resid_auc = roc_auc_score(outcome_arr, residual)
                                
                                # Optional: momentum vs outcome (sanity check)
                                mom_outcome_spearman, mom_outcome_pval = spearmanr(
                                    resid_df["_momentum_6h"].values, outcome_arr
                                )
                                
                                # --- Residual lift by decile (preferred diagnostic) ---
                                resid_df["_residual"] = residual
                                # qcut can fail when many duplicate values; handle duplicates gracefully
                                try:
                                    resid_df["_resid_decile"] = pd.qcut(
                                        resid_df["_residual"],
                                        q=10,
                                        labels=False,
                                        duplicates="drop"
                                    )
                                except Exception:
                                    resid_df["_resid_decile"] = np.nan

                                lift_table = None
                                top_bottom_lift = None
                                if resid_df["_resid_decile"].notna().sum() > 0:
                                    lift_table = (
                                        resid_df
                                        .dropna(subset=["_resid_decile"])
                                        .groupby("_resid_decile", as_index=False)
                                        .agg(
                                            n=("_outcome", "size"),
                                            mean_residual=("_residual", "mean"),
                                            mean_prob=("_prob", "mean"),
                                            realized_yes_rate=("_outcome", "mean"),
                                            mean_momentum=("_momentum_6h", "mean"),
                                        )
                                        .sort_values("_resid_decile")
                                    )
                                    lift_table["_resid_decile"] = lift_table["_resid_decile"].astype(int) + 1
                                    if len(lift_table) >= 2:
                                        top_bottom_lift = float(lift_table["realized_yes_rate"].iloc[-1] - lift_table["realized_yes_rate"].iloc[0])

                                # --- Display results ---
                                n_samples = len(resid_df)
                                
                                # Summary table
                                results_data = {
                                    "Metric": [
                                        "N (samples)", 
                                        "Raw Spearman ρ (p vs outcome)",
                                        "Raw AUC",
                                        "Residual Spearman ρ",
                                        "Residual AUC",
                                        "β (momentum coef)",
                                        "α (intercept)",
                                        "R² (regression)",
                                        "Momentum vs Outcome ρ"
                                    ],
                                    "Value": [
                                        f"{n_samples:,}",
                                        f"{raw_spearman:.4f} (p={raw_spearman_pval:.4f})",
                                        f"{raw_auc:.4f}",
                                        f"{resid_spearman:.4f} (p={resid_spearman_pval:.4f})",
                                        f"{resid_auc:.4f}",
                                        f"{beta_momentum:.6f}",
                                        f"{intercept_a:.4f}",
                                        f"{r_squared:.4f}",
                                        f"{mom_outcome_spearman:.4f} (p={mom_outcome_pval:.4f})"
                                    ]
                                }
                                results_table = pd.DataFrame(results_data)
                                st.dataframe(results_table, width="stretch", hide_index=True)
                                
                                # Residual lift table (preferred)
                                st.markdown("**Residual lift by decile (diagnostic):**")
                                if lift_table is None or lift_table.empty:
                                    st.info("Residual decile lift table unavailable (insufficient unique residual values).")
                                else:
                                    st.dataframe(
                                        lift_table.rename(columns={
                                            "_resid_decile": "Residual Decile (1=lowest)",
                                            "n": "N",
                                            "mean_residual": "Mean Residual",
                                            "mean_prob": "Mean Model Prob",
                                            "realized_yes_rate": "Realized YES Rate",
                                            "mean_momentum": "Mean Momentum (6h)",
                                        }),
                                        width="stretch",
                                        hide_index=True
                                    )
                                    if top_bottom_lift is not None:
                                        st.caption(f"Top–bottom realized YES lift (decile 10 minus decile 1): {top_bottom_lift:+.4f}")

                                # Interpretation (use lift/monotonicity, not residual AUC)
                                st.markdown("**Interpretation:**")
                                if raw_auc > 0.85:
                                    st.warning(
                                        "Raw AUC is extremely high for a real market. This often indicates leakage or label contamination. "
                                        "Verify snapshot_time/expiry_time alignment and confirm p_model_cal is not trained on the same outcomes."
                                    )
                                if top_bottom_lift is not None:
                                    if top_bottom_lift > 0.05:
                                        st.success("✅ Residual signal shows meaningful lift across deciles (non-momentum structure likely present).")
                                    elif top_bottom_lift < -0.05:
                                        st.error("⚠️ Residual looks like anti-signal (top decile underperforms bottom).")
                                    else:
                                        st.info("ℹ️ Residual lift is small; remaining signal after momentum removal may be weak or noisy.")
                                else:
                                    st.info("ℹ️ Unable to compute top–bottom lift reliably; check residual distribution and sample size.")
                                    
                            except np.linalg.LinAlgError:
                                st.warning("Could not compute regression (singular matrix).")
                            except ImportError as ie:
                                st.warning(f"Missing dependency for residual metrics: {ie}")
                            except Exception as e:
                                st.warning(f"Error computing residual metrics: {e}")
                
                # Download button for all priced contracts (filtered by max DTE if enabled)
                # Place outside the nested blocks so it always appears when all_priced_bt exists
                download_df = all_priced_bt.copy()
                if use_max_dte and "dte_days" in download_df.columns:
                    download_df["dte_days"] = pd.to_numeric(download_df["dte_days"], errors="coerce")
                    download_df = download_df[download_df["dte_days"] <= max_dte_value]
                    label_suffix = f" (DTE ≤ {max_dte_value})"
                else:
                    label_suffix = ""
                csv_all_priced = download_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download All Priced Contracts{label_suffix} (CSV)",
                    data=csv_all_priced,
                    file_name="all_priced_contracts.csv",
                    mime="text/csv",
                )
                    
        elif equity_bt is not None and isinstance(equity_bt, pd.DataFrame) and not equity_bt.empty:
            eq_fig = px.line(equity_bt, x="pricing_date", y="bankroll", title="Equity curve", template="plotly_white")
            st.plotly_chart(eq_fig, width="stretch")

# ---------------------------------------------------------------------------
# Positions tab
# ---------------------------------------------------------------------------

with tabs[5]:
    st.header("Positions")
    if positions_df.empty:
        st.info("No positions recorded. Add rows to positions.csv to see exposure and P&L here.")
    else:
        all_positions = positions_df.copy()
        all_positions["entry_price"] = pd.to_numeric(all_positions.get("entry_price"), errors="coerce")
        all_positions["size_shares"] = pd.to_numeric(all_positions.get("size_shares"), errors="coerce")
        all_positions["entry_notional"] = all_positions["entry_price"] * all_positions["size_shares"]
        all_positions["realized_pnl"] = pd.to_numeric(all_positions.get("realized_pnl"), errors="coerce")
        total_invested = float(all_positions["entry_notional"].sum(skipna=True))
        col_tot1, col_tot2 = st.columns(2)
        with col_tot1:
            st.metric("Total capital invested", f"${total_invested:,.2f}")
        with col_tot2:
            st.metric("Realized PnL (closed trades)", f"${realized_pnl_total:,.2f}")

        current_unreal_component = 0.0
        if open_positions_enriched.empty:
            st.info("No open positions currently. Add rows to positions.csv to see live exposure.")
        else:
            # Allow a manual refresh of enrichment for debugging when batch/positions change on disk.
            if st.button("Re-run enrichment (debug)"):
                pos_df = enrich_positions_with_batch(open_positions_df, current_df)
            else:
                pos_df = open_positions_enriched.copy()
            pos_df["entry_notional"] = pos_df["entry_price"].astype(float) * pos_df["size_shares"].astype(float)
            pos_df["mtm_value"] = pos_df["mtm_value"].astype(float)
            pos_df["unrealized_pnl"] = pos_df["unrealized_pnl"].astype(float)
            total_entry = float(pos_df["entry_notional"].sum(skipna=True))
            total_mtm = float(pos_df["mtm_value"].sum(skipna=True))
            total_unreal = float(pos_df["unrealized_pnl"].sum(skipna=True))
            current_unreal_component = total_unreal
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Open entry notional", f"${total_entry:,.2f}")
            with col2:
                st.metric("Open MTM value", f"${total_mtm:,.2f}")
            with col3:
                st.metric("Unrealized PnL", f"${total_unreal:,.2f}")

            exposure = (
                pos_df.groupby("expiry_key", observed=True)["mtm_value"]
                .sum()
                .reset_index()
                .sort_values("mtm_value", ascending=False)
            )
            if not exposure.empty:
                fig_pos = px.bar(
                    exposure,
                    x="expiry_key",
                    y="mtm_value",
                    title="Open Positions by Expiry ($)",
                    template="plotly_white",
                )
                fig_pos.update_layout(xaxis_title="Expiry", yaxis_title="Market Value ($)")
                st.plotly_chart(fig_pos, width="stretch")

            display_cols = [
                "slug",
                "question",
                "expiry_key",
                "side",
                "entry_date",
                "entry_price",
                "current_price",
                "size_shares",
                "entry_notional",
                "mtm_value",
                "unrealized_pnl",
                "model_prob_at_entry",
                "model_prob_latest",
                "notes",
            ]
            available_cols = [c for c in display_cols if c in pos_df.columns]
            positions_rename = {
                "slug": "Contract",
                "question": "Question",
                "expiry_key": "Expiry",
                "side": "Side",
                "entry_date": "Entry Date",
                "entry_price": "Entry Price",
                "current_price": "Current Price",
                "size_shares": "Shares",
                "entry_notional": "Cost ($)",
                "mtm_value": "Market Value ($)",
                "unrealized_pnl": "Unrealized PnL ($)",
                "model_prob_at_entry": "Model Prob (Entry)",
                "model_prob_latest": "Model Prob (Now)",
                "notes": "Notes",
            }
            st.dataframe(pos_df[available_cols].rename(columns=positions_rename), width="stretch")
            # Highlight any rows that failed to enrich so users can diagnose missing keys.
            if "current_price" in pos_df.columns and "model_prob_latest" in pos_df.columns:
                missing_mask = pos_df["current_price"].isna() | pos_df["model_prob_latest"].isna()
                if missing_mask.any():
                    st.warning(
                        f"{int(missing_mask.sum())} open position(s) missing live enrichment; "
                        "check slug/expiry/strike alignment with the active batch."
                    )
                    debug_cols = [
                        "slug",
                        "expiry_key",
                        "strike",
                        "side",
                        "current_price",
                        "model_prob_latest",
                    ]
                    st.dataframe(pos_df.loc[missing_mask, [c for c in debug_cols if c in pos_df.columns]], width="stretch")

        current_bankroll = bankroll_sidebar + realized_pnl_total + current_unreal_component
        if bankroll_sidebar > 0:
            drawdown_pct = (bankroll_sidebar - current_bankroll) / bankroll_sidebar
        else:
            drawdown_pct = 0.0
        drawdown_value = bankroll_sidebar - current_bankroll
        col_cb, col_dd, col_ddv = st.columns(3)
        with col_cb:
            st.metric("Current bankroll", f"${current_bankroll:,.2f}")
        with col_dd:
            st.metric("Drawdown %", f"{drawdown_pct * 100:.2f}%")
        with col_ddv:
            st.metric("Drawdown ($)", f"${drawdown_value:,.2f}")


# ---------------------------------------------------------------------------
# Stability history tab
# ---------------------------------------------------------------------------

with tabs[7]:
    st.header("Stability (History)")

    summary_available = stability_summary_df is not None and not stability_summary_df.empty
    drift_available = stability_drift_df is not None and not stability_drift_df.empty

    if stability_summary_error:
        st.warning(stability_summary_error)
    if stability_drift_error:
        st.warning(stability_drift_error)

    if not summary_available and not drift_available:
        st.info("Provide stability summary and drift CSVs via the sidebar to enable this tab.")
    else:
        if summary_available:
            st.subheader("Daily overview")
            daily_overview = build_daily_overview(stability_summary_df)
            if daily_overview.empty:
                st.info("No stability summary rows available after grouping.")
            else:
                st.dataframe(daily_overview, width="stretch")
                daily_metrics = [
                    ("avg_stability_score", "Average stability score over time", "Score (0-100)", "Stability score not available in summary."),
                    (
                        "avg_monotonic_violation_rate",
                        "Average monotonic violation rate",
                        "Violation rate",
                        "Monotonic violation rate not present in summary.",
                    ),
                    ("avg_mean_abs_resid", "Average |MC - fit| residual", "Abs residual", "Residual column missing from summary."),
                    (
                        "avg_mean_abs_diff_rn_model",
                        "Average |model - RN|",
                        "|model - RN|",
                        "RN comparison column missing from summary.",
                    ),
                ]
                for col, title, ylabel, msg in daily_metrics:
                    if col in daily_overview.columns and daily_overview[col].notna().any():
                        render_time_series_line(daily_overview, col, title, ylabel)
                    else:
                        st.info(msg)

            st.subheader("Expiry-level stability")
            expiry_options = sorted(stability_summary_df["expiry_key"].dropna().unique())
            if expiry_options:
                selected_expiry = st.selectbox(
                    "Select expiry for detail",
                    expiry_options,
                    key="history_expiry_select",
                )
                compare_expiries = st.multiselect(
                    "Compare expiries (stability score)",
                    expiry_options,
                    default=[selected_expiry],
                    key="history_expiry_compare",
                )
                if compare_expiries:
                    if "stability_score" in stability_summary_df.columns:
                        compare_df = stability_summary_df[
                            stability_summary_df["expiry_key"].isin(compare_expiries)
                        ].copy()
                        render_time_series_line(
                            compare_df,
                            "stability_score",
                            "Stability score comparison",
                            "Score (0-100)",
                            color_col="expiry_key",
                        )
                    else:
                        st.info("Stability score column missing; comparison chart skipped.")

                exp_df = stability_summary_df[
                    stability_summary_df["expiry_key"] == selected_expiry
                ].copy()
                if not exp_df.empty:
                    metric_defs = [
                        ("stability_score", "Stability score history", "Score (0-100)"),
                        ("monotonic_violation_rate", "Monotonic violation rate", "Rate"),
                        ("mean_abs_resid", "|MC - fit| residual", "Abs residual"),
                        ("mean_abs_diff_rn_model", "|Model - RN|", "Abs diff"),
                    ]
                    for col, title, ylabel in metric_defs:
                        render_time_series_line(
                            exp_df,
                            col,
                            f"{title} ({selected_expiry})",
                            ylabel,
                        )

                    edge_cols = [c for c in ["mean_edge", "max_abs_edge"] if c in exp_df.columns]
                    if edge_cols:
                        clean_edge = exp_df.dropna(subset=edge_cols, how="all")
                        if not clean_edge.empty:
                            fig_edge = go.Figure()
                            for col in edge_cols:
                                fig_edge.add_trace(
                                    go.Scatter(
                                        x=clean_edge["pricing_date"],
                                        y=clean_edge[col],
                                        mode="lines+markers",
                                        name=col.replace("_", " ").title(),
                                    )
                                )
                            fig_edge.update_layout(
                                title=f"Edge metrics ({selected_expiry})",
                                xaxis_title="Pricing date",
                                yaxis_title="Edge",
                                template="plotly_white",
                            )
                            st.plotly_chart(fig_edge, width="stretch")

                    stats_table = summarize_expiry_metrics(
                        exp_df,
                        [
                            ("monotonic_violation_rate", "Monotonic violation rate"),
                            ("mean_abs_resid", "Mean |MC - fit|"),
                            ("mean_abs_diff_rn_model", "Mean |model - RN|"),
                            ("mean_edge", "Mean edge"),
                            ("max_abs_edge", "Max |edge|"),
                        ],
                    )
                    if not stats_table.empty:
                        st.dataframe(stats_table, width="stretch")
                    else:
                        st.info("No numeric metrics available for the selected expiry.")
            else:
                st.info("No expiries found in stability summary.")

        st.subheader("Cross-day drift summary")
        if drift_available:
            st.dataframe(stability_drift_df, width="stretch")
            if "avg_drift_model_per_day" in stability_drift_df.columns:
                fig_drift_model = px.bar(
                    stability_drift_df,
                    x="expiry_key",
                    y="avg_drift_model_per_day",
                    title="Average model drift per day",
                    template="plotly_white",
                )
                st.plotly_chart(fig_drift_model, width="stretch")
            if "avg_drift_edge_per_day" in stability_drift_df.columns:
                fig_drift_edge = px.bar(
                    stability_drift_df,
                    x="expiry_key",
                    y="avg_drift_edge_per_day",
                    title="Average edge drift per day",
                    template="plotly_white",
                )
                st.plotly_chart(fig_drift_edge, width="stretch")
        else:
            st.info("Drift summary CSV not loaded; skipping cross-day drift plots.")

        if summary_available:
            st.subheader("Bad days & anomalies")
            stability_threshold = st.slider(
                "Stability score threshold",
                min_value=0,
                max_value=100,
                value=70,
                step=1,
                key="stability_threshold_slider",
            )
            violation_threshold = 0.05
            resid_threshold = 0.10
            rn_threshold = 0.20
            mask = pd.Series(False, index=stability_summary_df.index)
            if "stability_score" in stability_summary_df.columns:
                mask |= stability_summary_df["stability_score"] < stability_threshold
            if "monotonic_violation_rate" in stability_summary_df.columns:
                mask |= stability_summary_df["monotonic_violation_rate"] > violation_threshold
            if "mean_abs_resid" in stability_summary_df.columns:
                mask |= stability_summary_df["mean_abs_resid"] > resid_threshold
            if "mean_abs_diff_rn_model" in stability_summary_df.columns:
                mask |= stability_summary_df["mean_abs_diff_rn_model"] > rn_threshold
            bad_rows = stability_summary_df[mask].copy()
            if bad_rows.empty:
                st.success("No anomalous days detected with the current thresholds.")
            else:
                bad_rows.sort_values(["pricing_date", "expiry_key"], inplace=True)
                cols = [
                    col
                    for col in [
                        "pricing_date",
                        "expiry_key",
                        "stability_score",
                        "monotonic_violation_rate",
                        "mean_abs_resid",
                        "mean_abs_diff_rn_model",
                        "max_abs_edge",
                    ]
                    if col in bad_rows.columns
                ]
                st.dataframe(bad_rows[cols], width="stretch")
                st.caption(
                    f"Flagged rows where stability_score<{stability_threshold}, "
                    f"monotonic_violation_rate>{violation_threshold}, "
                    f"|MC - fit|>{resid_threshold}, or |model - RN|>{rn_threshold}."
                )

            heat_value_col = "stability_score" if "stability_score" in stability_summary_df.columns else None
            if heat_value_col is None and "monotonic_violation_rate" in stability_summary_df.columns:
                heat_value_col = "monotonic_violation_rate"
            if heat_value_col:
                heat_df = stability_summary_df.copy()
                heat_df["pricing_date_str"] = heat_df["pricing_date"].dt.strftime("%Y-%m-%d")
                pivot = heat_df.pivot_table(
                    index="expiry_key",
                    columns="pricing_date_str",
                    values=heat_value_col,
                    aggfunc="mean",
                )
                if not pivot.empty:
                    color_title = "Stability score" if heat_value_col == "stability_score" else "Monotonic violation rate"
                    heat_fig = go.Figure(
                        data=go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns,
                            y=pivot.index,
                            colorscale="Viridis",
                            colorbar=dict(title=color_title),
                        )
                    )
                    heat_fig.update_layout(
                        title=f"{color_title} heatmap",
                        xaxis_title="Pricing date",
                        yaxis_title="Expiry",
                        template="plotly_white",
                    )
                    st.plotly_chart(heat_fig, width="stretch")
