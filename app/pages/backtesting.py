#!/usr/bin/env python3
"""
backtesting.py

Streamlit page for running historical backtests using the BacktestEngine.
"""

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
from core.backtesting.orchestrator import BacktestingOrchestrator

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")

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

if BACKTEST_MODE == "🌐 Live Fetch from Polymarket":
    # Live Fetch main panel
    col1, col2 = st.columns(2)
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
                },
            )
            with st.status("Running full backtesting pipeline...") as status:
                result = orch.run_full(fetch=False, limit=fetch_limit if fetch_limit > 0 else None)
                status.update(label="Pipeline complete!", state="complete")

            st.session_state["bt_trades"] = result["trades_df"]
            st.session_state["bt_equity"] = result["equity_df"]
            st.session_state["bt_all_priced"] = result.get("all_priced_df")
            st.session_state["bt_diagnostics"] = result.get("diagnostics")
            st.session_state["bt_initial"] = initial_bankroll

            fetch_errors = result.get("fetch_errors", [])
            if fetch_errors:
                st.warning(
                    f"Pipeline complete! {len(result['trades_df'])} trades, "
                    f"{result['new_records']} new price records. "
                    f"{len(fetch_errors)} fetch issue(s) — see Fetch step."
                )
            else:
                st.success(
                    f"Pipeline complete! {len(result['trades_df'])} trades, "
                    f"{result['new_records']} new price records."
                )

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
                    "max_add_per_cycle_usd": max_trade_usd,  # Cap position size
                    "use_fixed_stake": use_fixed_stake,
                    "fixed_stake_amount": max_trade_usd if use_fixed_stake else None,
                    "max_capital_per_expiry_frac": max_capital_per_expiry,
                    "max_capital_total_frac": max_capital_total,
                    "use_stability_penalty": use_stability_penalty,
                    "correlation_penalty": correlation_penalty,
                }

                # Run backtest
                trades_df, equity_df = run_backtest(
                    daily_batches=batches,
                    initial_bankroll=initial_bankroll,
                    strategy_params=strategy_params,
                )

                # Store in session state
                st.session_state["bt_trades"] = trades_df
                st.session_state["bt_equity"] = equity_df
                st.success(f"Backtest complete! {len(trades_df)} trades executed.")

# -----------------------------------------------------------------------------
# Display Results (shared by both modes)
# -----------------------------------------------------------------------------
if "bt_trades" in st.session_state and "bt_equity" in st.session_state:
    trades_df = st.session_state["bt_trades"]
    equity_df = st.session_state["bt_equity"]
    
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
    # Signal Diagnostics (from orchestrator run_full)
    # -------------------------------------------------------------------------
    if "bt_diagnostics" in st.session_state and st.session_state["bt_diagnostics"]:
        st.divider()
        st.header("🔬 Signal Diagnostics")
        diag = st.session_state["bt_diagnostics"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Spearman ρ", f"{diag.get('spearman_rho', 0):.4f}")
        col2.metric("p-value", f"{diag.get('spearman_pvalue', 1):.4f}")
        col3.metric("AUC", f"{diag.get('auc', 0.5):.4f}")
        col4.metric("Observations", f"{diag.get('n_observations', 0)}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Edge (Winners)", f"{diag.get('mean_edge_winners', 0):.4f}")
        col2.metric("Mean Edge (Losers)", f"{diag.get('mean_edge_losers', 0):.4f}")
        col3.metric("Edge Difference", f"{diag.get('edge_difference', 0):.4f}")

        # DTE Breakdown
        st.subheader("📅 DTE Breakdown")
        dte_breakdown = diag.get("dte_breakdown", [])
        if dte_breakdown:
            st.dataframe(dte_breakdown, use_container_width=True, hide_index=True)
        else:
            st.info("No DTE breakdown available" if diag.get("dte_available") else "DTE column not available in data")

        # Moneyness Breakdown
        st.subheader("🎯 Moneyness Breakdown")
        money_breakdown = diag.get("moneyness_breakdown", [])
        if money_breakdown:
            st.dataframe(money_breakdown, use_container_width=True, hide_index=True)
        else:
            st.info("No moneyness breakdown available" if diag.get("moneyness_available") else "Moneyness column not available in data")

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
