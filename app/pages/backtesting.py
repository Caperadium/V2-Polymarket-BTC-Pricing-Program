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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.backtesting.backtest_engine import run_backtest

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")

st.title("📊 Backtesting Engine")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def scan_batch_files(
    root_dir: str,
    start_date: date,
    end_date: date,
    filename: str = "batch_with_fits.csv",
) -> List[str]:
    """Scan root_dir for batch folders within the date range.
    
    Supports folder formats:
    - New: 2025-12-20_05-57-14_UTC
    - Legacy: batch_20251113_220535
    """
    valid_paths = []
    if not os.path.exists(root_dir):
        return []

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        
        folder_date = None
        
        # Try new format: 2025-12-20_05-57-14_UTC
        match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_UTC", entry.name)
        if match:
            try:
                date_part = match.group(1)
                time_part = match.group(2).replace("-", ":")
                dt_str = f"{date_part}T{time_part}+00:00"
                dt_utc = datetime.fromisoformat(dt_str)
                dt_local = dt_utc.astimezone(None)
                folder_date = dt_local.date()
            except ValueError:
                pass
        
        # Try legacy format: batch_20251113_220535
        if folder_date is None:
            legacy_match = re.search(r"batch_(\d{8})_(\d{6})", entry.name)
            if legacy_match:
                try:
                    dt_str = legacy_match.group(1)  # YYYYMMDD
                    dt_utc = datetime.strptime(dt_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                    dt_local = dt_utc.astimezone(None)
                    folder_date = dt_local.date()
                except ValueError:
                    pass
        
        if folder_date is None:
            continue
        
        if start_date <= folder_date <= end_date:
            file_path = os.path.join(entry.path, filename)
            if os.path.exists(file_path):
                valid_paths.append(file_path)
    
    return sorted(valid_paths)


def prepare_batch_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize a batch dataframe."""
    df = df.copy()
    
    pricing_date = None
    
    # Try new format: 2025-12-20_05-57-14_UTC
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_UTC", source_name)
    if match:
        date_part = match.group(1)
        time_part = match.group(2).replace("-", ":")
        dt_str = f"{date_part}T{time_part}+00:00"
        pricing_date = pd.to_datetime(dt_str)
    
    # Try legacy format: batch_20251113_220535
    if pricing_date is None:
        legacy_match = re.search(r"batch_(\d{8})_(\d{6})", source_name)
        if legacy_match:
            dt_str = legacy_match.group(1) + legacy_match.group(2)  # YYYYMMDDHHMMSS
            pricing_date = pd.to_datetime(dt_str, format="%Y%m%d%H%M%S", utc=True)
    
    # Fallback
    if pricing_date is None:
        pricing_date = pd.Timestamp.now(tz="UTC")
    
    if "pricing_date" not in df.columns:
        df["pricing_date"] = pricing_date
    if "batch_timestamp" not in df.columns:
        df["batch_timestamp"] = pricing_date
    
    # Ensure expiry_key
    if "expiry_key" not in df.columns:
        if "expiry_date" in df.columns:
            df["expiry_key"] = pd.to_datetime(df["expiry_date"]).dt.strftime("%Y-%m-%d")
        else:
            df["expiry_key"] = "unknown"
    
    return df


def load_batches(paths: List[str]) -> List[pd.DataFrame]:
    """Load and prepare batch DataFrames."""
    batches = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            df = prepare_batch_df(df, path)
            batches.append(df)
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
    return batches


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
# Sidebar: Settings
# -----------------------------------------------------------------------------

st.sidebar.header("⚙️ Backtest Settings")

# Batch folder path
batch_folder = st.sidebar.text_input(
    "Batch Folder Path",
    value="fitted_batch_results",
    help="Relative or absolute path to folder containing batch CSVs"
)

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
with col2:
    end_date = st.date_input("End Date", value=date.today())

st.sidebar.divider()

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

# -----------------------------------------------------------------------------
# Main Panel
# -----------------------------------------------------------------------------

# Scan for batches
batch_paths = scan_batch_files(batch_folder, start_date, end_date)

st.info(f"Found **{len(batch_paths)}** batch files in `{batch_folder}` from {start_date} to {end_date}")

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

# Display results
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
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No edge bucket data available.")
    
    # Trade Log
    with st.expander("📋 Trade Log"):
        if not trades_df.empty:
            display_cols = [
                "trade_id", "slug", "side", "strike", "entry_price", 
                "stake", "model_prob", "market_price", "edge", "pnl", "settled"
            ]
            available_cols = [c for c in display_cols if c in trades_df.columns]
            st.dataframe(trades_df[available_cols], width="stretch")
        else:
            st.info("No trades executed.")
