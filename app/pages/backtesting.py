#!/usr/bin/env python3
"""
backtesting.py

Streamlit page for running historical backtests using the BacktestEngine.
"""

import math
import os
import sys
import re
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.backtesting.backtest_engine import run_backtest
from core.backtesting.batch_loader import (
    load_batches,
    prepare_batch_df,
    scan_batch_files,
    scan_flat_batch_files,
)
from core.backtesting.orchestrator import (
    BacktestingOrchestrator,
    default_worker_count,
)
from core.backtesting.diagnostics import SignalDiagnostics
from core.backtesting import in_sample_oos as iso
from app.ui_filters import moneyness_filter_controls

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")

# Signed-edge panel filter presets (PRD §2)
_SIGNED_EDGE_PRESETS = {
    "Tail (primary)": {
        "price_min": 0.05, "price_max": 0.20,
        "moneyness_min": 0.0, "moneyness_max": None, "moneyness_mode": "signed",
    },
    "Tail (deep)": {
        "price_min": 0.05, "price_max": 0.10,
        "moneyness_min": 0.0, "moneyness_max": None, "moneyness_mode": "signed",
    },
    "Tail strict": {
        "price_min": 0.05, "price_max": 0.20,
        "moneyness_min": 0.02, "moneyness_max": None, "moneyness_mode": "signed",
    },
    "ATM control": {
        "price_min": 0.40, "price_max": 0.60,
        "moneyness_min": None, "moneyness_max": 0.02, "moneyness_mode": "abs",
    },
    "Custom": None,
}

st.title("📊 Backtesting Engine")

# -----------------------------------------------------------------------------
# Helper Functions (thin wrappers preserved for optional backward compat)
# -----------------------------------------------------------------------------
# scan_batch_files, scan_flat_batch_files, prepare_batch_df, load_batches
# are imported from core.backtesting.batch_loader above.


def compute_daily_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily PnL from settled trades."""
    if trades_df.empty or "pnl" not in trades_df.columns:
        return pd.DataFrame(columns=["date", "pnl"])
    
    settled = trades_df[trades_df["settled"] == True].copy()
    if settled.empty:
        return pd.DataFrame(columns=["date", "pnl"])
    
    settled["date"] = pd.to_datetime(settled["settlement_date"]).dt.date
    daily = settled.groupby("date")["pnl"].sum().reset_index()
    return daily


def compute_max_drawdown(equity_df: pd.DataFrame) -> float:
    """Compute max drawdown from equity curve."""
    if equity_df.empty or "bankroll" not in equity_df.columns:
        return 0.0
    
    equity = equity_df["bankroll"].values
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def compute_sharpe(trades_df: pd.DataFrame) -> float:
    """Compute daily Sharpe ratio."""
    daily_pnl = compute_daily_pnl(trades_df)
    if daily_pnl.empty or len(daily_pnl) < 2:
        return 0.0
    
    mean_pnl = daily_pnl["pnl"].mean()
    std_pnl = daily_pnl["pnl"].std()
    if std_pnl == 0:
        return 0.0
    
    return float(mean_pnl / std_pnl * np.sqrt(252))  # Annualized


def compute_edge_bucket_stats(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute win rate and avg PnL by edge bucket."""
    if trades_df.empty:
        return pd.DataFrame()
    
    settled = trades_df[trades_df["settled"] == True].copy()
    if settled.empty or "model_prob" not in settled.columns or "market_price" not in settled.columns:
        return pd.DataFrame()
    
    # Compute edge - must account for trade side
    # For YES: edge = model_prob - entry_price (market_price)
    # For NO: edge = (1 - model_prob) - (1 - market_price) = market_price - model_prob
    def calc_edge(row):
        if row.get("side", "YES").upper() == "NO":
            return row["market_price"] - row["model_prob"]  # NO edge
        return row["model_prob"] - row["market_price"]  # YES edge
    
    settled["edge"] = settled.apply(calc_edge, axis=1)
    
    # Create edge buckets
    bins = [-1, 0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 1.0]
    labels = ["<0%", "0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-40%", "40-50%", ">50%"]
    settled["edge_bucket"] = pd.cut(settled["edge"], bins=bins, labels=labels)
    
    # Compute stats
    settled["win"] = settled["pnl"] > 0
    stats = settled.groupby("edge_bucket", observed=True).agg(
        trades=("pnl", "count"),
        wins=("win", "sum"),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    
    stats["win_rate"] = (stats["wins"] / stats["trades"] * 100).round(1)
    stats["avg_pnl"] = stats["avg_pnl"].round(2)
    stats["total_pnl"] = stats["total_pnl"].round(2)
    
    return stats[["edge_bucket", "trades", "win_rate", "avg_pnl", "total_pnl"]]


# -----------------------------------------------------------------------------
# Signed-Edge Reliability Diagram helpers
# -----------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for proportion k/n. Returns (lower, upper)."""
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _build_eval_df(all_priced_df: pd.DataFrame, use_calibrated: bool = False) -> pd.DataFrame:
    """
    Prepare per-contract eval frame for the signed-edge panel.
    Deduplicates to earliest snapshot per contract_id.
    Drops rows where outcome_yes is NaN (unresolved).
    """
    from core.strategy.common import MODEL_PROB_CANDIDATES, resolve_model_prob

    df = all_priced_df.copy()

    # model_p
    #   ON  → p_model_cal (calibrated).
    #   OFF → model_prob_raw (raw, never-calibrated) so the toggle is a true
    #         raw-vs-calibrated comparison even when the backtest ran with the
    #         flag on (then model_prob_used already equals p_model_cal). Falls
    #         back to model_prob_used / resolve for older all_priced_df frames.
    if use_calibrated and "p_model_cal" in df.columns:
        df["_model_p"] = pd.to_numeric(df["p_model_cal"], errors="coerce")
    elif "model_prob_raw" in df.columns:
        df["_model_p"] = pd.to_numeric(df["model_prob_raw"], errors="coerce")
    elif "model_prob_used" in df.columns:
        df["_model_p"] = pd.to_numeric(df["model_prob_used"], errors="coerce")
    else:
        df["_model_p"] = resolve_model_prob(df, candidates=MODEL_PROB_CANDIDATES)

    # market_p
    market_col = next(
        (c for c in ["market_yes_price", "market_price", "market_pr", "Polymarket_Price"]
         if c in df.columns),
        None,
    )
    if market_col is None:
        return pd.DataFrame()
    df["_market_p"] = pd.to_numeric(df[market_col], errors="coerce")

    # outcome
    df["_outcome"] = pd.to_numeric(df.get("outcome_yes", pd.Series(dtype=float)), errors="coerce")

    # contract_id
    if "clobTokenId" in df.columns:
        df["_contract_id"] = df["clobTokenId"].astype(str)
    else:
        slugs = df["slug"].astype(str) if "slug" in df.columns else pd.Series("", index=df.index)
        strikes = df["strike"].astype(str) if "strike" in df.columns else pd.Series("", index=df.index)
        df["_contract_id"] = slugs + "|" + strikes

    df = df.dropna(subset=["_model_p", "_market_p", "_outcome"])
    if df.empty:
        return pd.DataFrame()

    # Deduplicate to earliest snapshot per contract
    ts_col = next((c for c in ["snapshot_time", "date"] if c in df.columns), None)
    if ts_col:
        df["_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.sort_values("_ts").drop_duplicates(subset=["_contract_id"], keep="first")

    if df.empty:
        return pd.DataFrame()

    eps = 1e-6
    df["_model_p"] = df["_model_p"].clip(eps, 1 - eps)
    df["_market_p"] = df["_market_p"].clip(eps, 1 - eps)
    df["_outcome"] = df["_outcome"].astype(int)

    df["_edge_yes"]    = df["_model_p"] - df["_market_p"]
    df["_market_p_no"] = 1.0 - df["_market_p"]
    df["_edge_no"]     = (1.0 - df["_model_p"]) - df["_market_p_no"]
    df["_edge_signed"] = df["_edge_yes"]

    if "moneyness" in df.columns:
        df["_moneyness"] = pd.to_numeric(df["moneyness"], errors="coerce")
    if "dte_days" in df.columns:
        df["_dte"] = pd.to_numeric(df["dte_days"], errors="coerce")

    return df.reset_index(drop=True)


def _compute_contract_counts(all_priced_df: pd.DataFrame) -> dict:
    """Unique-contract-aware counts from source, not row/snapshot counts.

    One contract priced across N daily snapshots contributes N rows to
    all_priced_df.  This function collapses each contract to a single count
    so the header metrics reflect *contracts* not *snapshots*.
    """
    if "clobTokenId" in all_priced_df.columns:
        src_ids = all_priced_df["clobTokenId"].astype(str)
    else:
        slugs = (
            all_priced_df["slug"].astype(str)
            if "slug" in all_priced_df.columns
            else pd.Series("", index=all_priced_df.index)
        )
        strikes = (
            all_priced_df["strike"].astype(str)
            if "strike" in all_priced_df.columns
            else pd.Series("", index=all_priced_df.index)
        )
        src_ids = slugs + "|" + strikes

    outcome = pd.to_numeric(
        all_priced_df.get("outcome_yes", pd.Series(index=all_priced_df.index, dtype=float)),
        errors="coerce",
    )
    resolved_mask = outcome.notna()

    unique_total = src_ids.nunique()
    unique_resolved = src_ids[resolved_mask].nunique()
    unique_unresolved = unique_total - unique_resolved
    # Snapshot rows with non-NaN outcome minus unique resolved = dedup duplication
    n_dedup_removed = int(resolved_mask.sum()) - unique_resolved

    return {
        "unique_total": unique_total,
        "unique_unresolved": unique_unresolved,
        "n_raw": unique_resolved,
        "n_dedup_removed": n_dedup_removed,
    }


def _apply_signed_edge_filters(
    df: pd.DataFrame,
    price_min: float,
    price_max: float,
    moneyness_min,
    moneyness_max,
    moneyness_mode: str = "signed",
) -> pd.DataFrame:
    """Apply market-price + moneyness filters to the eval frame."""
    mask = (df["_market_p"] >= price_min) & (df["_market_p"] <= price_max)

    if "_moneyness" in df.columns and (moneyness_min is not None or moneyness_max is not None):
        basis = df["_moneyness"].abs() if moneyness_mode == "abs" else df["_moneyness"]
        mask &= basis.notna()
        if moneyness_min is not None:
            mask &= basis >= moneyness_min
        if moneyness_max is not None:
            mask &= basis <= moneyness_max

    return df[mask].copy()


_DEFAULT_BIN_EDGES = [-0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20, np.inf]


def _compute_edge_bins(df: pd.DataFrame, bin_edges=None) -> pd.DataFrame:
    """Bin df by _edge_signed. Adaptively merges bins with n < 50."""
    if bin_edges is None:
        bin_edges = _DEFAULT_BIN_EDGES

    edges = sorted(set([-np.inf] + list(bin_edges)))

    bins = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (df["_edge_signed"] >= lo) if i == len(edges) - 2 else \
               (df["_edge_signed"] >= lo) & (df["_edge_signed"] < hi)
        sub = df[mask].copy()
        bins.append({"lo": lo, "hi": hi, "sub": sub, "n": len(sub)})

    MIN_BIN, TARGET_N = 50, 150
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(bins):
            if b["n"] < MIN_BIN and len(bins) > 1:
                options = []
                if i > 0:
                    options.append((i - 1, abs(bins[i-1]["n"] + b["n"] - TARGET_N)))
                if i < len(bins) - 1:
                    options.append((i + 1, abs(bins[i+1]["n"] + b["n"] - TARGET_N)))
                j = min(options, key=lambda x: x[1])[0]
                lo_i, hi_i = min(i, j), max(i, j)
                merged = {
                    "lo": bins[lo_i]["lo"],
                    "hi": bins[hi_i]["hi"],
                    "sub": pd.concat([bins[lo_i]["sub"], bins[hi_i]["sub"]]),
                    "n": bins[lo_i]["n"] + bins[hi_i]["n"],
                }
                bins = bins[:lo_i] + [merged] + bins[hi_i + 1:]
                changed = True
                break

    rows = []
    for b in bins:
        sub, n = b["sub"], b["n"]
        lo_s = f"{b['lo']:.2f}" if b["lo"] != -np.inf else "−∞"
        hi_s = f"{b['hi']:.2f}" if b["hi"] != np.inf else "+∞"
        label = f"[{lo_s}, {hi_s})"

        if n == 0:
            rows.append({"label": label, "lo": b["lo"], "hi": b["hi"],
                         "n": 0, "n_yes": 0, "realized_yes_rate": np.nan,
                         "mean_edge": np.nan, "mean_model_p": np.nan, "mean_market_p": np.nan,
                         "wilson_lo": np.nan, "wilson_hi": np.nan,
                         "realized_minus_market": np.nan, "low_confidence": True})
            continue

        n_yes = int(sub["_outcome"].sum())
        realized = n_yes / n
        w_lo, w_hi = _wilson_ci(n_yes, n)
        rows.append({
            "label": label, "lo": b["lo"], "hi": b["hi"],
            "n": n, "n_yes": n_yes,
            "realized_yes_rate": realized,
            "mean_edge": float(sub["_edge_signed"].mean()),
            "mean_model_p": float(sub["_model_p"].mean()),
            "mean_market_p": float(sub["_market_p"].mean()),
            "wilson_lo": w_lo, "wilson_hi": w_hi,
            "realized_minus_market": realized - float(sub["_market_p"].mean()),
            "low_confidence": n < 30,
        })

    return pd.DataFrame(rows)


def _panel_metrics(df: pd.DataFrame) -> dict:
    """Brier scores, BSS, ECE, calibration slope + intercept."""
    from scipy.special import logit, expit
    from sklearn.linear_model import LogisticRegression

    n = len(df)
    if n < 2:
        return {}

    y = df["_outcome"].values.astype(float)
    model_p = df["_model_p"].values
    market_p = df["_market_p"].values
    p_bar = float(y.mean())

    brier_model  = float(np.mean((model_p - y) ** 2))
    brier_market = float(np.mean((market_p - y) ** 2))
    brier_base   = float(np.mean((p_bar - y) ** 2))
    bss = 1.0 - brier_model / brier_market if brier_market > 0 else np.nan

    # ECE: progressive quantile reduction → fixed-width fallback
    ece = np.nan
    for q in [10, 5, 2]:
        try:
            actual_q = min(q, n // 5)
            if actual_q < 2:
                break  # too few data points for quantile bins, use fixed-width
            q_bins = pd.qcut(model_p, q=actual_q, duplicates="drop")
            ece_parts = []
            for b in q_bins.unique():
                mask = (q_bins == b).values
                if mask.sum():
                    ece_parts.append(abs(y[mask].mean() - model_p[mask].mean()) * mask.sum() / n)
            ece = float(sum(ece_parts))
            if q < 10:
                st.caption(f"ECE used {q} bins (data too clustered for 10 equal-frequency bins)")
            break
        except Exception as qe:
            if q == 2:  # last quantile attempt before fixed-width
                st.warning(f"ECE quantile binning failed — falling back to fixed-width: {qe}")
            continue

    if np.isnan(ece):
        # Last resort: 10 fixed-width bins on [0, 1]
        try:
            bin_edges = np.linspace(0, 1, 11)
            binned = np.digitize(model_p, bin_edges) - 1
            binned = np.clip(binned, 0, 9)
            ece_parts = []
            for b in range(10):
                mask = binned == b
                if mask.sum():
                    ece_parts.append(abs(y[mask].mean() - model_p[mask].mean()) * mask.sum() / n)
            ece = float(sum(ece_parts))
            st.caption("ECE computed with fixed-width bins (quantile binning not possible on this data)")
        except Exception as fwe:
            st.warning(f"ECE computation failed: {fwe}")
            ece = np.nan

    cal_slope = cal_intercept = cal_slope_se = cal_intercept_se = np.nan
    try:
        X = logit(model_p).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        lr.fit(X, y.astype(int))
        cal_slope     = float(lr.coef_[0][0])
        cal_intercept = float(lr.intercept_[0])
        p_hat = expit(X.flatten() * cal_slope + cal_intercept)
        W = p_hat * (1 - p_hat)
        Xf = X.flatten()
        H = np.array([[np.sum(W * Xf**2), np.sum(W * Xf)],
                      [np.sum(W * Xf),    np.sum(W)]])
        cov = np.linalg.inv(H)
        cal_slope_se     = float(np.sqrt(max(0, cov[0, 0])))
        cal_intercept_se = float(np.sqrt(max(0, cov[1, 1])))
    except Exception as lre:
        st.warning(f"Calibration logistic regression failed: {lre}")

    return {
        "n": n, "n_yes": int(y.sum()), "n_no": n - int(y.sum()),
        "brier_model": brier_model, "brier_market": brier_market,
        "brier_baseline": brier_base, "bss": bss, "ece": ece,
        "cal_slope": cal_slope, "cal_slope_se": cal_slope_se,
        "cal_intercept": cal_intercept, "cal_intercept_se": cal_intercept_se,
    }


def _render_signed_edge_panel(all_priced_df: pd.DataFrame) -> None:
    """Render the signed-edge reliability diagram panel."""
    import plotly.graph_objects as go

    st.header("📐 Signed-Edge Reliability Diagram")
    st.caption(
        "Reflects the **active IS/OOS window** selected above. Market price = mid "
        "(taker edge is lower by ~½ spread). Deduplicated to earliest snapshot per "
        "contract. N counts appear after dedup and NaN exclusion."
    )

    col_preset, col_m2 = st.columns([2, 1])

    with col_preset:
        preset_name = st.selectbox(
            "Filter preset",
            list(_SIGNED_EDGE_PRESETS.keys()),
            index=0,
            key="se_preset",
        )

    preset = _SIGNED_EDGE_PRESETS[preset_name]

    if preset_name == "Custom":
        c1, c2 = st.columns(2)
        with c1:
            price_min = st.number_input("Min market price", 0.01, 0.99, 0.05, 0.01, key="se_pmin")
            price_max = st.number_input("Max market price", 0.01, 0.99, 0.20, 0.01, key="se_pmax")
        with c2:
            mny_mode = st.radio("Moneyness mode", ["signed", "abs"], horizontal=True, key="se_mnymode")
            mny_min  = st.number_input("Min moneyness", -0.5, 0.5, 0.0, 0.01, key="se_mnymin")
            mny_max_val = st.number_input("Max moneyness (blank = none)", -0.5, 0.5, 0.5, 0.01, key="se_mnymax")
        moneyness_min = float(mny_min)
        moneyness_max = float(mny_max_val) if mny_max_val < 0.5 else None
        moneyness_mode = mny_mode
        _price_min, _price_max = price_min, price_max
    else:
        _price_min      = preset["price_min"]
        _price_max      = preset["price_max"]
        moneyness_min   = preset["moneyness_min"]
        moneyness_max   = preset["moneyness_max"]
        moneyness_mode  = preset["moneyness_mode"]

    has_cal = "p_model_cal" in all_priced_df.columns and \
              all_priced_df["p_model_cal"].notna().any()
    with col_m2:
        if has_cal:
            use_cal = st.toggle("Use M2-calibrated p", value=False, key="se_m2toggle")
        else:
            st.toggle("Use M2-calibrated p", value=False, disabled=True, key="se_m2toggle_off")
            st.caption("M2 column absent — run pipeline with `USE_CALIBRATED_PROB=True` to enable.")
            use_cal = False

    eval_df = _build_eval_df(all_priced_df, use_calibrated=use_cal)
    if eval_df.empty:
        st.warning("No data after column selection. Check that `outcome_yes` and model probability columns are present.")
        return

    # Unique-contract-level counts (not snapshot-row counts)
    counts = _compute_contract_counts(all_priced_df)
    n_raw = counts["n_raw"]

    filtered_df = _apply_signed_edge_filters(
        eval_df, _price_min, _price_max, moneyness_min, moneyness_max, moneyness_mode
    )

    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    hcol1.metric("Contracts (filtered)", len(filtered_df))
    hcol2.metric("Unique contracts (total)", counts["unique_total"])
    hcol3.metric("Unresolved (NaN outcome)", counts["unique_unresolved"])
    hcol4.metric("Resolved (deduped)", n_raw,
                 help=f"{counts['n_dedup_removed']} snapshot-duplicate rows collapsed via dedup")

    N_MIN = 200
    low_sample = len(filtered_df) < N_MIN
    if low_sample:
        st.warning(
            f"N = {len(filtered_df)} — below {N_MIN} minimum for reliable calibration inference. "
            "Metrics, plots, and table are shown as a LOW-CONFIDENCE exploratory view: "
            "Wilson intervals are wide and Brier/BSS/ECE/calibration are unstable at this N. "
            "Interpret accordingly."
        )

    if filtered_df.empty:
        st.info("No contracts match current filters.")
        return

    bins_df = _compute_edge_bins(filtered_df)

    # Metrics always shown (even below N_MIN — the warning above flags low confidence).
    m = _panel_metrics(filtered_df)
    st.subheader("📊 Panel Metrics" + ("  ⚠️ low-confidence (N < %d)" % N_MIN if low_sample else ""))
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Brier (model)",  f"{m.get('brier_model', np.nan):.4f}")
    mc2.metric("Brier (market)", f"{m.get('brier_market', np.nan):.4f}")
    mc3.metric("BSS vs. market", f"{m.get('bss', np.nan):.4f}",
               help="Positive = model beats market. Decision-critical number.")
    mc4.metric("ECE",            f"{m.get('ece', np.nan):.4f}")
    mc5.metric("n / n_yes / n_no",
               f"{m.get('n',0)} / {m.get('n_yes',0)} / {m.get('n_no',0)}")

    slope    = m.get("cal_slope", np.nan)
    slope_se = m.get("cal_slope_se", np.nan)
    intcpt   = m.get("cal_intercept", np.nan)
    intcpt_se = m.get("cal_intercept_se", np.nan)
    st.caption(
        f"Calibration logistic regression on logit(model_p): "
        f"slope = {slope:.3f} ± {slope_se:.3f}, "
        f"intercept = {intcpt:.3f} ± {intcpt_se:.3f}. "
        "Calibrated target: slope=1, intercept=0."
    )

    pcol1, pcol2 = st.columns(2)
    valid_bins = bins_df[bins_df["n"] > 0]

    with pcol1:
        st.subheader("Plot A — Calibration (probability space)")
        fig_a = go.Figure()

        fig_a.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Perfect calibration",
            line=dict(dash="dash", color="gray"),
        ))
        fig_a.add_trace(go.Scatter(
            x=valid_bins["mean_model_p"],
            y=valid_bins["realized_yes_rate"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(valid_bins["wilson_hi"] - valid_bins["realized_yes_rate"]).tolist(),
                arrayminus=(valid_bins["realized_yes_rate"] - valid_bins["wilson_lo"]).tolist(),
            ),
            mode="markers+text",
            name="Model",
            marker=dict(
                size=12,
                symbol=["circle-open" if r else "circle" for r in valid_bins["low_confidence"]],
                color="royalblue",
            ),
            text=valid_bins["n"].astype(str),
            textposition="top center",
        ))
        _market_line = valid_bins.sort_values("mean_market_p")
        fig_a.add_trace(go.Scatter(
            x=_market_line["mean_market_p"],
            y=_market_line["realized_yes_rate"],
            mode="lines+markers",
            name="Market",
            line=dict(color="darkorange", dash="dot"),
            marker=dict(size=7),
        ))
        fig_a.update_layout(
            xaxis_title="mean model_p per bin",
            yaxis_title="realized YES rate",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
            height=420, legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_a, use_container_width=True)

    with pcol2:
        st.subheader("Plot B — Edge calibration (deployment view)")

        def _opacity(row):
            # Y = realized − market; its Wilson CI is [wilson_lo − market, wilson_hi − market]
            # (market treated as constant per bin). CI crosses zero — i.e. the bin is
            # statistically inconclusive — iff wilson_lo < market < wilson_hi.
            if row["wilson_lo"] < row["mean_market_p"] < row["wilson_hi"]:
                return 0.35
            return 1.0

        opacities = [_opacity(r) for _, r in valid_bins.iterrows()]
        y_err_hi = (valid_bins["wilson_hi"] - valid_bins["realized_yes_rate"]).tolist()
        y_err_lo = (valid_bins["realized_yes_rate"] - valid_bins["wilson_lo"]).tolist()

        fig_b = go.Figure()
        x_range = [float(valid_bins["mean_edge"].min()) - 0.02,
                   float(valid_bins["mean_edge"].max()) + 0.02]
        fig_b.add_trace(go.Scatter(
            x=x_range, y=x_range,
            mode="lines", name="Calibrated edge target",
            line=dict(dash="dash", color="gray"),
        ))
        fig_b.add_hline(y=0, line_dash="dot", line_color="lightgray")
        fig_b.add_vline(x=0, line_dash="dot", line_color="lightgray")
        fig_b.add_annotation(x=0.08, y=0.08, text="✓ YES tradeable", showarrow=False,
                             font=dict(color="green", size=10))
        fig_b.add_annotation(x=-0.08, y=-0.08, text="✓ NO tradeable", showarrow=False,
                             font=dict(color="green", size=10))
        fig_b.add_annotation(x=-0.08, y=0.06, text="✗ Model wrong", showarrow=False,
                             font=dict(color="red", size=10))
        fig_b.add_annotation(x=0.08, y=-0.06, text="✗ Model wrong", showarrow=False,
                             font=dict(color="red", size=10))

        for i, (_, row) in enumerate(valid_bins.iterrows()):
            fig_b.add_trace(go.Scatter(
                x=[row["mean_edge"]],
                y=[row["realized_minus_market"]],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[y_err_hi[i]],
                    arrayminus=[y_err_lo[i]],
                ),
                mode="markers",
                marker=dict(
                    size=12,
                    opacity=opacities[i],
                    symbol="circle-open" if row["low_confidence"] else "circle",
                    color="royalblue",
                ),
                name=row["label"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['label']}</b><br>"
                    f"mean_edge={row['mean_edge']:.3f}<br>"
                    f"realized−market={row['realized_minus_market']:.3f}<br>"
                    f"n={row['n']}, n_yes={row['n_yes']}<br>"
                    f"Wilson [{row['wilson_lo']:.3f}, {row['wilson_hi']:.3f}]"
                    "<extra></extra>"
                ),
            ))
        fig_b.update_layout(
            xaxis_title="mean edge (model_p − market_p) per bin",
            yaxis_title="realized YES rate − mean market_p",
            height=420,
        )
        st.plotly_chart(fig_b, use_container_width=True)

    st.subheader("Bin-level table")
    st.caption(
        "Open circle bins (low_confidence=True) have n < 30 after adaptive merge. "
        "NO-side edge uses 1−market_price approximation (true NO ask would reduce edge by ~½ spread)."
    )

    display_bins = bins_df[[
        "label", "n", "n_yes",
        "mean_edge", "mean_model_p", "mean_market_p",
        "realized_yes_rate", "wilson_lo", "wilson_hi",
        "realized_minus_market",
    ]].copy()
    display_bins.columns = [
        "Edge bin", "n", "n_yes",
        "mean_edge", "mean_model_p", "mean_market_p",
        "realized_yes_rate", "Wilson_lo", "Wilson_hi",
        "realized − market",
    ]
    for col in ["mean_edge","mean_model_p","mean_market_p","realized_yes_rate",
                "Wilson_lo","Wilson_hi","realized − market"]:
        display_bins[col] = display_bins[col].round(4)

    st.dataframe(display_bins, use_container_width=True, hide_index=True)

    csv_bytes = display_bins.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export bin-level table (CSV)",
        data=csv_bytes,
        file_name=f"signed_edge_bins_{preset_name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        key="se_export",
    )


# -----------------------------------------------------------------------------
# Sidebar: Common Parameters (always visible)
# -----------------------------------------------------------------------------

st.sidebar.header("⚙️ Backtest Settings")

# Bankroll
st.sidebar.subheader("💰 Position Sizing")
initial_bankroll = st.sidebar.number_input(
    "Starting Bankroll ($)",
    min_value=100.0,
    max_value=1000000.0,
    value=1000.0,
    step=100.0
)
kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.05, 0.50, 0.15, 0.01)
min_trade_usd = st.sidebar.number_input("Min Trade ($)", 1.0, 100.0, 5.0, 1.0)
max_trade_usd = st.sidebar.number_input("Max Trade ($)", 10.0, 1000.0, 50.0, 10.0, help="Cap on single trade size")
use_fixed_stake = st.sidebar.checkbox("Use Fixed Stake Size", value=False, help="If checked, uses fixed stake instead of Kelly")

st.sidebar.divider()

# Strategy parameters
st.sidebar.subheader("📈 Strategy Parameters")
min_edge = st.sidebar.number_input("Min Edge", 0.0, 0.5, 0.06, 0.01)
max_bets_per_expiry = st.sidebar.number_input("Max Bets/Expiry", 1, 20, 3)
min_price = st.sidebar.number_input("Min Price", 0.01, 0.50, 0.03, 0.01)
max_price = st.sidebar.number_input("Max Price", 0.50, 0.99, 0.95, 0.01)
allow_no = True  # Always allow NO bets

st.sidebar.divider()

# Advanced
with st.sidebar.expander("🔧 Advanced Settings"):
    max_capital_per_expiry = st.slider("Max Capital/Expiry (%)", 5, 50, 15) / 100
    max_capital_total = st.slider("Max Capital Total (%)", 10, 80, 35) / 100
    use_stability_penalty = st.checkbox("Use Stability Penalty", value=True)
    correlation_penalty = st.slider("Correlation Penalty", 0.0, 0.5, 0.25, 0.05)
    _mny = moneyness_filter_controls(st, key_prefix="bt")

st.sidebar.divider()

# -----------------------------------------------------------------------------
# Sidebar: Outcome-based recalibration (FIX 7 / M2)
# -----------------------------------------------------------------------------
st.sidebar.subheader("🎯 Calibration (M2)")
use_calibrated_prob = st.sidebar.checkbox(
    "Use calibrated probability",
    value=False,
    help=(
        "Flip core.strategy.common.USE_CALIBRATED_PROB=True for THIS run only "
        "(reset to False afterward — no leak to other pages). When on, curve "
        "fitting writes p_model_cal and edges use it instead of p_model_fit. "
        "Requires a trusted DATA/calibration_shift.csv (produced by a prior full "
        "pipeline run; n_obs >= 200 per DTE bucket). No-op on existing batches "
        "that were fitted with the flag off."
    ),
)
if use_calibrated_prob:
    from core.pricing.fit_probability_curves import load_calibration_shift
    _trusted = load_calibration_shift()
    if _trusted:
        st.sidebar.caption(
            f"✅ {len(_trusted)} trusted DTE bucket(s) in calibration_shift.csv"
        )
    else:
        st.sidebar.warning(
            "No trusted calibration shift table found — toggle is a no-op until a "
            "full pipeline run writes DATA/calibration_shift.csv with applied buckets."
        )

st.sidebar.divider()

# -----------------------------------------------------------------------------
# Sidebar: Mode Selection
# -----------------------------------------------------------------------------

st.sidebar.subheader("🔀 Mode")
BACKTEST_MODE = st.sidebar.radio(
    "Backtest Mode",
    ["📂 Existing Batch Files", "🌐 Live Fetch from Polymarket"],
    index=0,
    help="Existing Batch Files: run backtest on already-generated batch CSVs.\n"
         "Live Fetch: fetch historical Polymarket prices, backrun pricing, then backtest.",
)

# -----------------------------------------------------------------------------
# Mode A: Live Fetch
# -----------------------------------------------------------------------------

if BACKTEST_MODE == "🌐 Live Fetch from Polymarket":
    st.sidebar.divider()
    st.sidebar.subheader("📡 Live Fetch Settings")
    n_sims = st.sidebar.number_input("MC Simulations", 100, 50000, 15000, 1000,
                                     help="Number of Monte Carlo paths per pricing run")
    fetch_limit = st.sidebar.number_input("Timestamp Limit", 0, 1000, 0, 1,
                                          help="Max number of historical timestamps to backrun (0 = all)")
    import multiprocessing as _mp
    _cpu = _mp.cpu_count()
    n_workers = st.sidebar.number_input(
        "Worker Processes", 1, _cpu, default_worker_count(), 1,
        help=f"Parallel processes for MC backrun (1 = serial). This {_cpu}-core "
             f"machine defaults to {default_worker_count()}.",
    )
    # FIX 3 re-enabled: XGBoost directional drift shift (backrun-time, Mode A only).
    use_xgb = st.sidebar.checkbox(
        "Use XGBoost directional drift",
        value=False,
        help="Per-snapshot, leak-free directional tilt of the MC distribution. "
             "Needs DATA/macro_daily.csv for real signal (BTC-only ≈ neutral).",
    )
    xgb_lambda = None
    if use_xgb:
        xgb_lambda = st.sidebar.slider(
            "XGB tilt λ", 0.0, 0.5, 0.15, 0.05,
            help="Tilt strength toward the XGBoost P(up). 0 = no effect. "
                 "Calibrate via the §8 grid before trusting.",
        )
        if not (PROJECT_ROOT / "DATA" / "macro_daily.csv").exists():
            st.sidebar.caption("⚠️ DATA/macro_daily.csv missing — running BTC-only "
                               "(weak signal). Run core/data/macro_fetcher.py.")

if BACKTEST_MODE == "🌐 Live Fetch from Polymarket":
    # Live Fetch main panel
    col1, col2 = st.columns(2)

    # ---- Purge old batch files ----
    with st.sidebar.expander("🧹 Purge Old Batch Files", expanded=False):
        st.caption(
            "Backrunner skips already-priced timestamps (idempotent). "
            "After an engine upgrade, old CSVs must be deleted or the new "
            "model is never run. This wipes `backtested_probabilities/unfitted/` "
            "and `backtested_probabilities/fitted/` so the next pipeline run "
            "re-prices every timestamp with the current engine."
        )
        if st.button("🗑️ Delete All Backtest Batch Files", type="secondary"):
            _purge_root = Path(__file__).resolve().parent.parent.parent
            _unfitted_dir = _purge_root / "backtested_probabilities" / "unfitted"
            _fitted_dir = _purge_root / "backtested_probabilities" / "fitted"

            _n_unfitted = 0
            _n_fitted = 0

            if _unfitted_dir.exists():
                _files = list(_unfitted_dir.glob("batch_*.csv"))
                _n_unfitted = len(_files)
                for _f in _files:
                    _f.unlink()

            if _fitted_dir.exists():
                _dirs = list(_fitted_dir.glob("batch_*"))
                _n_fitted = len(_dirs)
                for _d in _dirs:
                    import shutil
                    shutil.rmtree(_d, ignore_errors=True)

            if _n_unfitted == 0 and _n_fitted == 0:
                st.info("No batch files to delete.")
            else:
                st.success(
                    f"Deleted {_n_unfitted} unfitted batch CSV(s) "
                    f"and {_n_fitted} fitted batch folder(s). "
                    "Re-run the pipeline to price with the current engine."
                )

    with col1:
        if st.button("📡 Fetch Prices", help="Fetch historical Polymarket contract prices via Gamma + CLOB APIs"):
            status = st.status("Fetching historical contract prices...", expanded=True)

            def _on_progress(stage: str, n_done: int, n_total: int):
                pct = n_done / max(n_total, 1) * 100
                label = f"Scanning date-slugs…" if stage == "discovering" else f"Fetching prices…"
                status.write(f"{label} {n_done}/{n_total} ({pct:.0f}%)")

            orch = BacktestingOrchestrator(n_sims=n_sims, progress_callback=_on_progress)
            n_new, errors = orch.fetch_historical_prices()

            status.update(label="Fetch complete", state="complete", expanded=False)

            if errors:
                st.warning(f"{n_new} new records, with {len(errors)} issue(s):")
                for err in errors[:5]:
                    st.caption(f"• {err}")
                if len(errors) > 5:
                    st.caption(f"… and {len(errors) - 5} more")
            else:
                st.success(f"Fetch complete — {n_new} new price records added.")
            st.session_state["lf_fetched"] = True

    with col2:
        run_disabled = not st.session_state.get("lf_fetched", False)
        if st.button("🚀 Run Full Pipeline", type="primary", disabled=run_disabled,
                     help="Fetch → Backrun → Fit → Backtest → Diagnostics"):
            orch = BacktestingOrchestrator(
                n_sims=n_sims,
                initial_bankroll=initial_bankroll,
                use_xgb=use_xgb,
                xgb_tilt_lambda=xgb_lambda,
                strategy_params={
                    "kelly_fraction": kelly_fraction,
                    "min_edge": min_edge,
                    "max_bets_per_expiry": max_bets_per_expiry,
                    "min_price": min_price,
                    "max_price": max_price,
                    "allow_no": allow_no,
                    "min_trade_usd": min_trade_usd,
                    "max_add_per_cycle_usd": max_trade_usd,
                    "use_fixed_stake": use_fixed_stake,
                    "fixed_stake_amount": max_trade_usd if use_fixed_stake else None,
                    "max_capital_per_expiry_frac": max_capital_per_expiry,
                    "max_capital_total_frac": max_capital_total,
                    "use_stability_penalty": use_stability_penalty,
                    "correlation_penalty": correlation_penalty,
                    "min_moneyness": _mny["min_moneyness"],
                    "max_moneyness": _mny["max_moneyness"],
                    "moneyness_mode": _mny["mode"],
                },
            )
            _limit = fetch_limit if fetch_limit > 0 else None
            if _limit is None:
                st.warning(
                    "Timestamp Limit = 0 → backrunning **all** historical timestamps "
                    "with advanced features. This can take a very long time even in "
                    "parallel. Set a limit to test first.",
                    icon="⚠️",
                )
            import core.strategy.common as _common
            with st.status("Running full backtesting pipeline...") as status:
                # Scope USE_CALIBRATED_PROB to this run only (fit + backtest both
                # happen inside run_full, in this process). Reset afterward so the
                # flag never leaks to other pages sharing the Streamlit process.
                _common.USE_CALIBRATED_PROB = bool(use_calibrated_prob)
                try:
                    result = orch.run_full(
                        fetch=False, limit=_limit, workers=int(n_workers)
                    )
                finally:
                    _common.USE_CALIBRATED_PROB = False
                status.update(label="Pipeline complete!", state="complete")

            st.session_state["bt_trades"] = result["trades_df"]
            st.session_state["bt_equity"] = result["equity_df"]
            st.session_state["bt_all_priced"] = result.get("all_priced_df")
            st.session_state["bt_diagnostics"] = result.get("diagnostics")
            st.session_state["bt_initial"] = initial_bankroll

            fetch_errors = result.get("fetch_errors", [])

            if len(result.get("trades_df", pd.DataFrame())) == 0:
                st.warning(
                    "Pipeline completed but produced 0 trades. "
                    "Check that BTC data covers the contract date range. "
                    "Run `python core/data/data_fetcher.py` to refresh data."
                )
            elif fetch_errors:
                st.warning(
                    f"Pipeline complete! {len(result['trades_df'])} trades, "
                    f"{result['new_records']} new price records. "
                    f"{len(fetch_errors)} fetch issue(s) — see Fetch step."
                )
            else:
                msg = f"Pipeline complete! {len(result['trades_df'])} trades."
                if result.get("new_records", 0) > 0:
                    msg += f" {result['new_records']} new price records."
                st.success(msg)

else:
    # -----------------------------------------------------------------------------
    # Mode B: Existing Batch Files
    # -----------------------------------------------------------------------------

    # Batch folder path — selectbox with common sources
    BATCH_SOURCE_OPTIONS = {
        "Live batches (fitted_batch_results)": "fitted_batch_results",
        "Backtest batches (backtested_probabilities/fitted)": "backtested_probabilities/fitted",
        "Backtest unfitted (backtested_probabilities/unfitted)": "backtested_probabilities/unfitted",
        "Custom path...": "__custom__",
    }
    source_label = st.sidebar.selectbox(
        "Batch Source",
        list(BATCH_SOURCE_OPTIONS.keys()),
        index=0,
        help="Select where to look for batch CSV files"
    )
    if BATCH_SOURCE_OPTIONS[source_label] == "__custom__":
        batch_folder = st.sidebar.text_input(
            "Custom Batch Folder Path",
            value="fitted_batch_results",
            help="Relative or absolute path to folder containing batch CSVs"
        )
    else:
        batch_folder = BATCH_SOURCE_OPTIONS[source_label]

    # Date range
    use_date_filter = st.sidebar.checkbox("Filter by date range", value=False, help="Enable to limit batches to a date window")
    if use_date_filter:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date.today() - timedelta(days=90))
        with col2:
            end_date = st.date_input("End Date", value=date.today())
    else:
        # Use very wide range to include everything
        start_date = date(2020, 1, 1)
        end_date = date(2030, 12, 31)

    st.sidebar.divider()

    # -----------------------------------------------------------------------------
    # Main Panel — Existing Batch Files
    # -----------------------------------------------------------------------------

    # Scan for batches — use different scan logic for flat vs folder-based directories
    if batch_folder == "backtested_probabilities/unfitted":
        batch_paths = scan_flat_batch_files(batch_folder, start_date, end_date)
    else:
        batch_paths = scan_batch_files(batch_folder, start_date, end_date)

    st.info(f"Found **{len(batch_paths)}** batch files in `{batch_folder}`"
            + (f" from {start_date} to {end_date}" if use_date_filter else " (all dates)"))

    if len(batch_paths) == 0:
        st.warning(
            "No batch files found. Try:\n"
            "- Switching **Batch Source** to another directory\n"
            "- Enabling **Filter by date range** and widening the window\n"
            "- Or disabling the date filter to show all batches"
        )

    if st.button("🚀 Run Backtest", type="primary", disabled=len(batch_paths) == 0):
        with st.spinner("Running backtest..."):
            # Load batches
            batches = load_batches(batch_paths)

            if not batches:
                st.error("No valid batches loaded.")
            else:
                # Build strategy params
                strategy_params = {
                    "kelly_fraction": kelly_fraction,
                    "min_edge": min_edge,
                    "max_bets_per_expiry": max_bets_per_expiry,
                    "min_price": min_price,
                    "max_price": max_price,
                    "allow_no": allow_no,
                    "min_trade_usd": min_trade_usd,
                    "max_add_per_cycle_usd": max_trade_usd,
                    "use_fixed_stake": use_fixed_stake,
                    "fixed_stake_amount": max_trade_usd if use_fixed_stake else None,
                    "max_capital_per_expiry_frac": max_capital_per_expiry,
                    "max_capital_total_frac": max_capital_total,
                    "use_stability_penalty": use_stability_penalty,
                    "correlation_penalty": correlation_penalty,
                    "min_moneyness": _mny["min_moneyness"],
                    "max_moneyness": _mny["max_moneyness"],
                    "moneyness_mode": _mny["mode"],
                }

                # Run backtest — scope USE_CALIBRATED_PROB to this run only.
                # Note: Mode B replays already-fitted batches, so p_model_cal is
                # only present if those batches were fitted with the flag on; here
                # the flag only steers resolve_model_prob's column precedence.
                import core.strategy.common as _common
                _common.USE_CALIBRATED_PROB = bool(use_calibrated_prob)
                try:
                    trades_df, equity_df, all_priced_df = run_backtest(
                        daily_batches=batches,
                        initial_bankroll=initial_bankroll,
                        strategy_params=strategy_params,
                        return_all_priced=True,
                    )
                finally:
                    _common.USE_CALIBRATED_PROB = False

                # Store in session state
                st.session_state["bt_trades"] = trades_df
                st.session_state["bt_equity"] = equity_df
                st.session_state["bt_all_priced"] = all_priced_df
                st.success(f"Backtest complete! {len(trades_df)} trades executed.")

# -----------------------------------------------------------------------------
# Display Results (shared by both modes)
# -----------------------------------------------------------------------------
if "bt_trades" in st.session_state and "bt_equity" in st.session_state:
    trades_df = st.session_state["bt_trades"]
    equity_df = st.session_state["bt_equity"]

    # -------------------------------------------------------------------------
    # Global In-Sample / Out-of-Sample window (applies to ALL panels below)
    # -------------------------------------------------------------------------
    all_priced_full = st.session_state.get("bt_all_priced")
    window_df = all_priced_full
    isoos_spec = None
    isoos_suppress = False
    if (
        all_priced_full is not None
        and not all_priced_full.empty
        and "snapshot_time" in all_priced_full.columns
    ):
        st.subheader("🪟 In-Sample / Out-of-Sample Window")
        _default_cut = iso.compute_default_cutoff(all_priced_full, 0.7)
        wc1, wc2 = st.columns([1, 1])
        with wc1:
            _mode_label = st.radio(
                "Active window", ["In-sample", "Out-of-sample", "All"],
                index=2, horizontal=True, key="isoos_mode",
                help="Global cutoff splits contracts by pricing time. IS = priced "
                     "before cutoff; OOS = priced on/after. OOS loads the IS-cached "
                     "M2 calibration (no refit).",
            )
        with wc2:
            _cut_default = _default_cut.date() if _default_cut is not None else date.today()
            _cut_date = st.date_input("Cutoff date (UTC midnight)", value=_cut_default,
                                      key="isoos_cutoff")
        _mode = iso.WindowMode.from_label(_mode_label)
        _cutoff = iso.normalize_cutoff(pd.Timestamp(_cut_date))
        isoos_spec = iso.WindowSpec(cutoff=_cutoff, mode=_mode)

        _is_eval, _oos_eval, _straddle = iso.partition_contracts(all_priced_full, _cutoff)
        st.caption(
            f"Cutoff **{_cutoff:%Y-%m-%d}** · IS contracts: "
            f"{iso.contract_ids(_is_eval).nunique()} · OOS contracts: "
            f"{iso.contract_ids(_oos_eval).nunique()} · straddlers (priced IS, settle "
            f"OOS — excluded from M2 training): {iso.contract_ids(_straddle).nunique()}"
        )

        # Train (IS/All) or load-only (OOS) the cutoff-keyed M2 shift.
        _artifacts = None
        try:
            _artifacts = iso.load_or_train(_cutoff, all_priced_full, mode=_mode)
        except iso.OOSLeakError as _e:
            st.error(
                "OOS mode needs a cached IS artifact for this cutoff/data and found "
                "none (missing or stale). Switch to **In-sample** or **All** at this "
                "cutoff once to train it, then return to OOS.\n\n" + str(_e)
            )

        window_df = iso.apply_window(all_priced_full, isoos_spec)
        if _artifacts is not None:
            window_df = iso.apply_oos_calibration(window_df, _artifacts["shift_table"])
            if iso.is_m2_inert(_artifacts["shift_table"]):
                st.warning(
                    "M2 shift **inert** — all DTE buckets below the training-obs "
                    "threshold, so `p_model_cal == model_prob_raw` (no calibration "
                    "effect). Not a leak; the IS population is just too small per bucket."
                )

        _ss = iso.small_sample_state(len(window_df))
        isoos_suppress = _ss["suppress"]
        if _ss["banner"]:
            st.warning(_ss["banner"])

        # Window the trade-sim panels too (Decision: window everything).
        trades_df, equity_df = iso.apply_window_trades(trades_df, equity_df, isoos_spec)
        if _mode != iso.WindowMode.ALL:
            st.caption(
                "Trade-sim panels (Summary / Equity / Daily PnL / Win-Rate) are "
                "windowed by trade **entry** time. Daily PnL & Sharpe still bucket by "
                "settlement date, so their x-axis can straddle the cutoff. Equity curve "
                "is the windowed segment (not rebased)."
            )
        st.divider()

    # Summary metrics
    st.header("📊 Summary")
    
    settled_trades = trades_df[trades_df["settled"] == True] if not trades_df.empty else pd.DataFrame()
    total_pnl = settled_trades["pnl"].sum() if not settled_trades.empty else 0.0
    total_return = (total_pnl / initial_bankroll * 100) if initial_bankroll > 0 else 0.0
    max_dd = compute_max_drawdown(equity_df)
    sharpe = compute_sharpe(trades_df)
    win_rate = (settled_trades["pnl"] > 0).mean() * 100 if not settled_trades.empty else 0.0
    avg_pnl = settled_trades["pnl"].mean() if not settled_trades.empty else 0.0
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total PnL", f"${total_pnl:.2f}")
    m2.metric("Return", f"{total_return:.1f}%")
    m3.metric("Max Drawdown", f"{max_dd:.1%}")
    m4.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m5.metric("Win Rate", f"{win_rate:.1f}%")
    m6.metric("Avg PnL/Trade", f"${avg_pnl:.2f}")
    
    st.divider()
    
    # Daily PnL Chart
    st.subheader("📈 Daily PnL")
    daily_pnl = compute_daily_pnl(trades_df)
    if not daily_pnl.empty:
        daily_pnl["color"] = daily_pnl["pnl"].apply(lambda x: "green" if x >= 0 else "red")
        st.bar_chart(daily_pnl.set_index("date")["pnl"])
    else:
        st.info("No settled trades to display.")
    
    # Equity Curve
    st.subheader("💰 Equity Curve")
    if not equity_df.empty and "bankroll" in equity_df.columns:
        st.line_chart(equity_df.set_index("pricing_date")["bankroll"])
    
    st.divider()
    
    # Edge Bucket Analysis
    st.subheader("🎯 Win Rate by Edge Bucket")
    edge_stats = compute_edge_bucket_stats(trades_df)
    if not edge_stats.empty:
        st.dataframe(
            edge_stats.rename(columns={
                "edge_bucket": "Edge Bucket",
                "trades": "Trades",
                "win_rate": "Win Rate (%)",
                "avg_pnl": "Avg PnL ($)",
                "total_pnl": "Total PnL ($)",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No edge bucket data available.")
    
    # -------------------------------------------------------------------------
    # Signal Diagnostics — recomputed on the active window (not the precomputed
    # full-frame dict). [REVIEW S4] This also makes panels 1-4 render in Mode B,
    # which never set bt_diagnostics; gate on all_priced presence instead.
    # -------------------------------------------------------------------------
    _diag_source = window_df if window_df is not None else st.session_state.get("bt_all_priced")
    if _diag_source is not None and not _diag_source.empty:
        st.divider()
        st.header("🔬 Signal Diagnostics")
        diag = SignalDiagnostics(_diag_source).run_full_report()

        # p-value intentionally omitted: at N in the thousands the Spearman test is
        # overpowered (p underflows to ~0 for any non-zero ρ), so it carries no signal.
        # Effect size (ρ / AUC) is the meaningful discriminator.
        if isoos_suppress:
            st.info(
                "Small-sample window — overall ρ / AUC / mean-edge shown as a "
                "LOW-CONFIDENCE exploratory view (below the N=200 reliability "
                "threshold). Interpret with caution; breakdowns and tables below "
                "still shown."
            )
        # Metrics always shown (even small-sample — the banner above flags low confidence).
        # AUC can be None when only one outcome class is present; format defensively.
        _auc = diag.get("auc")
        _auc_str = f"{_auc:.4f}" if _auc is not None else "n/a"
        col1, col2, col3 = st.columns(3)
        col1.metric("Spearman ρ", f"{diag.get('spearman_rho', 0):.4f}")
        col2.metric("AUC", _auc_str)
        col3.metric("Observations", f"{diag.get('n_observations', 0)}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Edge (Winners)", f"{diag.get('mean_edge_winners', 0):.4f}")
        col2.metric("Mean Edge (Losers)", f"{diag.get('mean_edge_losers', 0):.4f}")
        col3.metric("Edge Difference", f"{diag.get('edge_difference', 0):.4f}")

        # Column renames so headers are self-explanatory rather than raw dict keys.
        # "pos"/"neg" are counts of contracts that resolved YES/NO within each bin;
        # both classes must be present for Spearman ρ / AUC to be computable.
        # The raw "p" (Spearman p-value) is dropped — overpowered at this N, always ~0.
        def _label_breakdown(rows: list, bucket_header: str) -> pd.DataFrame:
            df = pd.DataFrame(rows).drop(columns=["p"], errors="ignore")
            return df.rename(columns={
                "label": bucket_header,
                "n": "Contracts",
                "pos": "Resolved YES",
                "neg": "Resolved NO",
                "rho": "Spearman ρ",
                "auc": "AUC",
            })

        _breakdown_caption = (
            "Contracts = total priced in the bin · Resolved YES/NO = how many settled "
            "in/out of the money · Spearman ρ and AUC measure how well model edge "
            "predicted the outcome (need both YES and NO present to compute)."
        )

        # DTE Breakdown
        st.subheader("📅 DTE Breakdown")
        st.caption("Signal quality grouped by days-to-expiry at pricing time. " + _breakdown_caption)
        dte_breakdown = diag.get("dte_breakdown", [])
        if dte_breakdown:
            st.dataframe(
                _label_breakdown(dte_breakdown, "DTE Bucket"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No DTE breakdown available" if diag.get("dte_available") else "DTE column not available in data")

        # Moneyness Breakdown
        st.subheader("🎯 Moneyness Breakdown")
        st.caption("Signal quality grouped by moneyness (strike vs spot). " + _breakdown_caption)
        money_breakdown = diag.get("moneyness_breakdown", [])
        if money_breakdown:
            st.dataframe(
                _label_breakdown(money_breakdown, "Moneyness Bucket"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No moneyness breakdown available" if diag.get("moneyness_available") else "Moneyness column not available in data")

        # OTM Tail Mispricing (favorite–longshot)
        st.subheader("📉 OTM Tail Mispricing (favorite–longshot zone)")
        tm = diag.get("tail_mispricing")
        if not tm or not tm.get("available"):
            st.info("OTM tail-mispricing test needs moneyness data")
        else:
            lo_band, hi_band = tm["band"]
            st.caption(
                f"OTM contracts with market price in {lo_band:.2f}–{hi_band:.2f}. "
                "**AUC of model_p** = does the model rank winners better than the market alone "
                "(>0.54 ≈ real residual signal, ~0.5 ≈ nothing the market didn't encode). "
                "**AUC of edge** (model_p − market_p) = whether the divergence itself is tradeable; "
                "much lower than AUC model_p ⇒ the model is just shadowing the market. "
                "Favorite-longshot bias predicts AUC rising toward the 0.05 tail; flat/inverted ⇒ thesis fails."
            )

            def _tail_table(rows: list) -> pd.DataFrame:
                df = pd.DataFrame(rows).drop(columns=["lo", "hi"], errors="ignore")
                return df.rename(columns={
                    "label": "Band",
                    "n": "Contracts",
                    "pos": "Resolved YES",
                    "neg": "Resolved NO",
                    "auc_model": "AUC model_p",
                    "auc_edge": "AUC edge",
                })

            for variant in tm["variants"]:
                st.markdown(f"**{variant['label']}**")
                c1, c2, c3 = st.columns(3)
                am = variant.get("auc_model")
                ae = variant.get("auc_edge")
                c1.metric("AUC model_p", f"{am:.4f}" if am is not None else "n/a")
                c2.metric("AUC edge", f"{ae:.4f}" if ae is not None else "n/a")
                c3.metric("Contracts", f"{variant.get('n', 0)}")
                st.dataframe(
                    _tail_table(variant.get("sub_bands", [])),
                    use_container_width=True,
                    hide_index=True,
                )

    # Trade Log
    with st.expander("📋 Trade Log"):
        if not trades_df.empty:
            display_cols = [
                "trade_id", "slug", "side", "strike", "entry_price",
                "stake", "model_prob", "market_price", "edge", "pnl", "settled"
            ]
            available_cols = [c for c in display_cols if c in trades_df.columns]
            st.dataframe(trades_df[available_cols], use_container_width=True)
        else:
            st.info("No trades executed.")

    # -------------------------------------------------------------------------
    # Signed-Edge Reliability Diagram
    # -------------------------------------------------------------------------
    _signed_source = window_df if window_df is not None else st.session_state.get("bt_all_priced")
    if _signed_source is not None and not _signed_source.empty:
        st.divider()
        _render_signed_edge_panel(_signed_source)
    else:
        st.divider()
        st.info(
            "**Signed-Edge Reliability Diagram** requires `all_priced_df`. "
            "Run the full pipeline (Live Fetch mode) or use Existing Batch Files mode "
            "to generate it."
        )
