"""
polymarket/metrics.py

Metrics computation for Polymarket realized PnL and drawdown warnings.

Computes:
- Daily realized PnL from closed positions
- Daily loss limit check (yesterday PnL <= -4% bankroll)
- Rolling 7-day max drawdown (peak-to-trough)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from polymarket.db import get_connection
from polymarket.ingest import get_last_sync_time, get_total_realized_pnl

logger = logging.getLogger(__name__)

# Thresholds
DAILY_LOSS_LIMIT_PCT = 0.04  # 4%
ROLLING_MDD_LIMIT_PCT = 0.10  # 10%
STALE_DATA_MINUTES = 60  # Show warning if data older than this


@dataclass
class DrawdownMetrics:
    """Container for drawdown metrics and warnings."""
    has_data: bool = False
    yesterday_pnl: float = 0.0
    yesterday_loss_pct: float = 0.0
    daily_loss_warn: bool = False
    rolling_mdd: float = 0.0
    rolling_mdd_warn: bool = False
    total_realized_pnl: float = 0.0
    equity_curve: Optional[pd.DataFrame] = None
    daily_pnl_df: Optional[pd.DataFrame] = None
    last_sync: Optional[datetime] = None
    data_stale: bool = False
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for display."""
        return {
            "has_data": self.has_data,
            "yesterday_pnl": self.yesterday_pnl,
            "yesterday_loss_pct": self.yesterday_loss_pct,
            "daily_loss_warn": self.daily_loss_warn,
            "rolling_mdd": self.rolling_mdd,
            "rolling_mdd_warn": self.rolling_mdd_warn,
            "total_realized_pnl": self.total_realized_pnl,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "data_stale": self.data_stale,
            "notes": self.notes,
        }


def get_closed_positions_by_date(
    days: int = 30,
    user_address: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get closed positions grouped by date.
    
    Args:
        days: Number of days to look back
        user_address: Filter by user (optional)
        
    Returns:
        DataFrame with columns: date, realized_pnl, count
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Calculate Unix timestamp cutoff (resolved_at is stored as Unix epoch)
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_timestamp = int(cutoff_dt.timestamp())
        
        # resolved_at is stored as Unix epoch seconds, use datetime() to convert for grouping
        query = """
            SELECT 
                DATE(datetime(resolved_at, 'unixepoch')) as date,
                SUM(realized_pnl) as realized_pnl,
                COUNT(*) as count
            FROM pm_closed_positions
            WHERE resolved_at >= ?
        """
        params = [cutoff_timestamp]
        
        if user_address:
            query += " AND user_address = ?"
            params.append(user_address)
        
        query += " GROUP BY DATE(datetime(resolved_at, 'unixepoch')) ORDER BY date"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame(columns=["date", "realized_pnl", "count"])
        
        data = [{"date": row["date"], "realized_pnl": row["realized_pnl"], "count": row["count"]} for row in rows]
        return pd.DataFrame(data)
        
    finally:
        conn.close()


def compute_daily_loss_limit(
    daily_pnl_df: pd.DataFrame,
    bankroll: float,
) -> Tuple[float, float, bool]:
    """
    Compute daily loss limit check.
    
    Args:
        daily_pnl_df: DataFrame with date and realized_pnl columns
        bankroll: Current bankroll for percentage calculation
        
    Returns:
        Tuple of (yesterday_pnl, yesterday_loss_pct, warn)
    """
    if daily_pnl_df.empty or bankroll <= 0:
        return 0.0, 0.0, False
    
    # Get yesterday's date
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Find yesterday's PnL
    yesterday_row = daily_pnl_df[daily_pnl_df["date"] == yesterday]
    
    if yesterday_row.empty:
        return 0.0, 0.0, False
    
    yesterday_pnl = float(yesterday_row["realized_pnl"].iloc[0])
    yesterday_loss_pct = -yesterday_pnl / bankroll if yesterday_pnl < 0 else 0.0
    
    # Warn if loss >= 4% of bankroll
    warn = yesterday_pnl <= -DAILY_LOSS_LIMIT_PCT * bankroll
    
    return yesterday_pnl, yesterday_loss_pct, warn


def compute_equity_curve(
    daily_pnl_df: pd.DataFrame,
    starting_bankroll: float,
) -> pd.DataFrame:
    """
    Compute cumulative equity curve from daily PnL.
    
    Args:
        daily_pnl_df: DataFrame with date and realized_pnl columns
        starting_bankroll: Initial bankroll
        
    Returns:
        DataFrame with date, daily_pnl, cumulative_pnl, equity columns
    """
    if daily_pnl_df.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "cumulative_pnl", "equity"])
    
    df = daily_pnl_df.copy()
    df = df.sort_values("date")
    df["daily_pnl"] = df["realized_pnl"]
    df["cumulative_pnl"] = df["realized_pnl"].cumsum()
    df["equity"] = starting_bankroll + df["cumulative_pnl"]
    
    return df[["date", "daily_pnl", "cumulative_pnl", "equity"]]


def compute_rolling_max_drawdown(
    equity_curve: pd.DataFrame,
    window_days: int = 7,
) -> float:
    """
    Compute rolling max drawdown (peak-to-trough).
    
    Args:
        equity_curve: DataFrame with equity column
        window_days: Rolling window size
        
    Returns:
        Maximum drawdown as a positive percentage (0.10 = 10% drawdown)
    """
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return 0.0
    
    equity = equity_curve["equity"].values
    
    if len(equity) < 2:
        return 0.0
    
    # Use last N days for rolling calculation
    equity_window = equity[-window_days:] if len(equity) >= window_days else equity
    
    # Compute drawdown: (peak - current) / peak
    running_max = np.maximum.accumulate(equity_window)
    drawdowns = (running_max - equity_window) / running_max
    
    max_drawdown = float(np.max(drawdowns))
    return max_drawdown


def get_drawdown_warnings(
    bankroll: float,
    days: int = 30,
    user_address: Optional[str] = None,
) -> DrawdownMetrics:
    """
    Main entry point for getting drawdown metrics and warnings.
    
    Args:
        bankroll: Current bankroll for percentage calculations
        days: Days of history to analyze
        user_address: Filter by user (optional)
        
    Returns:
        DrawdownMetrics with all computed values and warnings
    """
    metrics = DrawdownMetrics()
    
    # Check last sync time
    last_sync = get_last_sync_time()
    metrics.last_sync = last_sync
    
    if last_sync:
        age_minutes = (datetime.now(timezone.utc) - last_sync).total_seconds() / 60
        metrics.data_stale = age_minutes > STALE_DATA_MINUTES
    else:
        metrics.data_stale = True
        metrics.notes = "No sync data available. Click 'Sync' to fetch from Polymarket."
        return metrics
    
    # Get daily PnL data
    daily_pnl_df = get_closed_positions_by_date(days=days, user_address=user_address)
    
    if daily_pnl_df.empty:
        metrics.notes = "No closed positions found. Realized PnL unavailable."
        return metrics
    
    metrics.has_data = True
    metrics.daily_pnl_df = daily_pnl_df
    
    # Compute daily loss limit
    yesterday_pnl, yesterday_loss_pct, daily_warn = compute_daily_loss_limit(
        daily_pnl_df, bankroll
    )
    metrics.yesterday_pnl = yesterday_pnl
    metrics.yesterday_loss_pct = yesterday_loss_pct
    metrics.daily_loss_warn = daily_warn
    
    # Compute equity curve
    equity_curve = compute_equity_curve(daily_pnl_df, bankroll)
    metrics.equity_curve = equity_curve
    
    # Compute rolling max drawdown
    rolling_mdd = compute_rolling_max_drawdown(equity_curve, window_days=7)
    metrics.rolling_mdd = rolling_mdd
    metrics.rolling_mdd_warn = rolling_mdd >= ROLLING_MDD_LIMIT_PCT
    
    # Total realized PnL for reconciliation
    metrics.total_realized_pnl = get_total_realized_pnl()
    
    # Set notes based on warnings
    notes = []
    if metrics.daily_loss_warn:
        notes.append(f"⚠️ Daily loss limit exceeded: {yesterday_loss_pct:.1%} of bankroll")
    if metrics.rolling_mdd_warn:
        notes.append(f"⚠️ Rolling 7D drawdown exceeded: {rolling_mdd:.1%}")
    if metrics.data_stale:
        notes.append("⚠️ Data may be stale. Consider syncing.")
    
    metrics.notes = " | ".join(notes) if notes else "All metrics within limits."
    
    return metrics
