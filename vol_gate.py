"""
vol_gate.py

A standalone, production-ready module that computes a BTC-volatility "gate" output
for a Polymarket trading system.

Goal:
    Return additional required edge and Kelly multiplier based on current BTC volatility
    regime relative to a trailing baseline.

Usage:
    import vol_gate
    result = vol_gate.compute_vol_gate(btc_df, now_utc="2025-12-29T12:00:00Z")

CLI:
    python vol_gate.py --now 2025-12-29T12:34:00Z --file DATA/btc_intraday_1m.csv
"""

import argparse
import logging
import sys
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging to stderr to keep stdout clean for JSON output
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolGateResult:
    now_utc: str                  # ISO string
    regime: str                   # "normal" | "high" | "extreme" | "unknown"
    vol15: Optional[float]        # realized vol (stdev of log returns) over 15m
    vol60: Optional[float]        # realized vol over 60m
    vol15_pct: Optional[float]    # percentile rank of vol15 vs baseline window
    shock: bool                   # sudden-move flag
    allow_new_entries: bool       # False in extreme, True otherwise
    edge_add_cents: float         # additional edge required in cents
    kelly_mult: float             # multiplier for fractional Kelly lambda
    reason: str                   # short explanation for logs


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 1-minute log returns: r_t = ln(close_t) - ln(close_{t-1}).
    Expects dataframe sorted by timestamp with a "close" column.
    """
    # Avoid log(0) errors
    closes = df["close"].replace(0, np.nan)
    df["log_ret"] = np.log(closes / closes.shift(1))
    return df


def calculate_rolling_vol(series: pd.Series, window: int, min_periods: int = 10) -> pd.Series:
    """
    Compute rolling standard deviation of log returns.
    """
    return series.rolling(window=window, min_periods=min_periods).std()


def compute_vol_gate(
    btc_df: pd.DataFrame,
    now_utc: Union[str, datetime],
    baseline_days: int = 14,
    vol15_window_min: int = 15,
    vol60_window_min: int = 60,
    high_pct: float = 80.0,
    extreme_pct: float = 95.0,
    high_edge_add_cents: float = 2.0,
    high_kelly_mult: float = 0.5,
    normal_kelly_mult: float = 1.0,
    extreme_kelly_mult: float = 0.0,
    enable_shock_gate: bool = True,
    shock_window_min: int = 5,
    shock_pct: float = 90.0,
    shock_hold_min: int = 15,  # Included for API compatibility, though pure function is stateless
) -> VolGateResult:
    """
    Compute the volatility gate metrics and regime.

    Args:
        btc_df: DataFrame with "timestamp" (or "ts") and "close".
        now_utc: The reference time for the calculation (UTC).
        baseline_days: Lookback window for percentile ranking.
        vol15_window_min: Window for primary volatility metric.
        vol60_window_min: Window for secondary volatility metric.
        high_pct: Percentile threshold for HIGH regime.
        extreme_pct: Percentile threshold for EXTREME regime.
        high_edge_add_cents: Edge penalty for HIGH regime.
        high_kelly_mult: Kelly multiplier for HIGH regime.
        normal_kelly_mult: Kelly multiplier for NORMAL regime.
        extreme_kelly_mult: Kelly multiplier for EXTREME regime.
        enable_shock_gate: Whether to check for asking sudden price moves.
        shock_window_min: Window for shock calculation (absolute return).
        shock_pct: Percentile threshold for shock detection.
        shock_hold_min: (Not used in pure function, but part of config spec).

    Returns:
        VolGateResult object.
    """
    # 1. Parse and validate `now_utc`
    if isinstance(now_utc, str):
        try:
            now = datetime.fromisoformat(now_utc.replace('Z', '+00:00'))
        except ValueError:
            # Fallback for simple formats if needed, or raise
            now = pd.to_datetime(now_utc).to_pydatetime()
    else:
        now = now_utc
    
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    
    now_iso = now.isoformat()

    # Default/Fallback Result
    def fallback_result(reason: str, regime="unknown", edge=2.0, kelly=0.5, allow=True) -> VolGateResult:
        return VolGateResult(
            now_utc=now_iso,
            regime=regime,
            vol15=None,
            vol60=None,
            vol15_pct=None,
            shock=False,
            allow_new_entries=allow,
            edge_add_cents=edge,
            kelly_mult=kelly,
            reason=reason
        )

    # 2. Data Preparation
    if btc_df is None or btc_df.empty:
        return fallback_result("No BTC data provided")

    # Normalize columns
    df = btc_df.copy()
    
    # Map common column name variations to standard names
    col_map = {
        "Close": "close",
        "Timestamp": "timestamp",
        "Date": "timestamp",
        "Time": "timestamp",
        "ts": "timestamp"
    }
    df = df.rename(columns=col_map)
    
    # Ensure lowercase standard names
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" not in df.columns and "date" in df.columns:
         df = df.rename(columns={"date": "timestamp"})
    
    if "timestamp" not in df.columns or "close" not in df.columns:
        # Debug info for user
        cols = list(btc_df.columns)
        return fallback_result(f"Missing 'timestamp' or 'close' columns (Found: {cols})")

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df = df.set_index("timestamp")

    # Filter relevant data (Baseline Window + Buffer)
    # limit data to [now - baseline_days - buffer, now]
    # Buffer is needed for rolling window calculation at the start of baseline
    buffer_days = 1 
    start_dt = now - timedelta(days=baseline_days + buffer_days)
    subset = df[(df.index >= start_dt) & (df.index <= now)].copy()

    if subset.empty:
        return fallback_result("No data in lookback window")

    # 3. Compute Log Returns
    subset = compute_log_returns(subset)
    
    # 4. Compute Rolling Volatility
    # vol15: Rolling std dev of log returns over 15m
    subset["vol15"] = calculate_rolling_vol(
        subset["log_ret"], 
        window=vol15_window_min, 
        min_periods=max(10, vol15_window_min // 2)
    )
    
    # vol60: Rolling std dev of log returns over 60m
    subset["vol60"] = calculate_rolling_vol(
        subset["log_ret"], 
        window=vol60_window_min, 
        min_periods=max(10, vol60_window_min // 2)
    )

    # Get current metrics at `now` (or nearest previous available point within reasonable lag)
    # We look for the exact timestamp or the last valid data point within 5 mins
    tolerance = timedelta(minutes=5)
    last_valid_idx = subset.index.asof(now)
    
    if pd.isna(last_valid_idx) or (now - last_valid_idx) > tolerance:
        return fallback_result(f"Data stale or missing near {now_iso}")
        
    current_vol15 = subset.loc[last_valid_idx, "vol15"]
    current_vol60 = subset.loc[last_valid_idx, "vol60"]
    current_close = subset.loc[last_valid_idx, "close"] # for shock
    
    # 5. Baseline Distribution for Percentile
    # Baseline window: [now - baseline_days, now]
    baseline_start = now - timedelta(days=baseline_days)
    baseline_mask = (subset.index >= baseline_start) & (subset.index <= now)
    baseline_vol15 = subset.loc[baseline_mask, "vol15"].dropna()
    
    if len(baseline_vol15) < 60 * 24: # Require at least ~1 day of minute data points
        # Not enough data for reliable percentile
        return VolGateResult(
            now_utc=now_iso,
            regime="unknown",
            vol15=current_vol15 if not pd.isna(current_vol15) else None,
            vol60=current_vol60 if not pd.isna(current_vol60) else None,
            vol15_pct=None,
            shock=False,
            allow_new_entries=True,
            edge_add_cents=2.0, # Conservative fallback
            kelly_mult=0.5,
            reason=f"Insufficient baseline data ({len(baseline_vol15)} points)"
        )
    
    if pd.isna(current_vol15):
        return fallback_result("Current vol15 is NaN", allow=True)

    # Calculate Percentile
    # (Count of points <= current) / Total points * 100
    # Using scipy.stats.percentileofscore equivalent
    vol15_pct = (baseline_vol15 < current_vol15).mean() * 100.0
    
    # 7. Shock Gate
    is_shock = False
    shock_reason = ""
    
    if enable_shock_gate:
        # Absolute 5-minute log return
        # abs(ln(close_now) - ln(close_{now-5m}))
        # We need to construct a rolling 5m absolute return series for baseline comparison
        
        # Method: Resample or use shift(5) on 1-min data?
        # shift(5) is cleaner for 1-min close data
        # Note: subset is 1-min index.
        subset["price_5m_ago"] = subset["close"].shift(shock_window_min)
        subset["abs_ret_5m"] = np.abs(np.log(subset["close"] / subset["price_5m_ago"]))
        
        current_abs_ret = subset.loc[last_valid_idx, "abs_ret_5m"]
        
        if not pd.isna(current_abs_ret):
            baseline_abs_ret = subset.loc[baseline_mask, "abs_ret_5m"].dropna()
            
            if len(baseline_abs_ret) > 0:
                # Percentile of current shock metric
                shock_rank = (baseline_abs_ret < current_abs_ret).mean() * 100.0
                
                if shock_rank >= shock_pct:
                    is_shock = True
                    shock_reason = f"Shock {shock_rank:.1f}% >= {shock_pct}%"

    # 6. Regime Logic
    regime = "normal"
    edge_add = 0.0
    kelly = normal_kelly_mult
    allow = True
    reason_parts = [f"vol15_pct={vol15_pct:.1f}"]
    
    # Check shock first (immediately elevates to extreme)
    if is_shock:
        regime = "extreme"
        allow = False
        edge_add = 1e9 # Sentinel for blocked
        kelly = extreme_kelly_mult
        reason_parts.append(f"SHOCK detected ({shock_reason})")
    elif vol15_pct >= extreme_pct:
        regime = "extreme"
        allow = False
        edge_add = 1e9
        kelly = extreme_kelly_mult
        reason_parts.append(f">= {extreme_pct} (Extreme)")
    elif vol15_pct >= high_pct:
        regime = "high"
        allow = True
        edge_add = high_edge_add_cents
        kelly = high_kelly_mult
        reason_parts.append(f">= {high_pct} (High)")
    else:
        regime = "normal"
        allow = True
        edge_add = 0.0
        kelly = normal_kelly_mult
        reason_parts.append(f"< {high_pct} (Normal)")

    return VolGateResult(
        now_utc=now_iso,
        regime=regime,
        vol15=current_vol15,
        vol60=current_vol60,
        vol15_pct=vol15_pct,
        shock=is_shock,
        allow_new_entries=allow,
        edge_add_cents=edge_add,
        kelly_mult=kelly,
        reason="; ".join(reason_parts)
    )

# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

class TestVolGate(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 1-min BTC dataframe
        # 15 days of data to satisfy baseline
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=60*24*15, freq="1min")
        self.df = pd.DataFrame({
            "timestamp": dates,
            "close": [100.0] * len(dates) # Flat price
        })
        self.now = dates[-1]

    def test_constant_price(self):
        # With constant price, vol should be 0
        res = compute_vol_gate(self.df, self.now)
        self.assertEqual(res.regime, "normal")
        self.assertEqual(res.vol15, 0.0)
        self.assertEqual(res.edge_add_cents, 0.0)
        self.assertEqual(res.kelly_mult, 1.0)
    
    def test_missing_data_future(self):
        # Requesting a time far in the future
        future = self.now + timedelta(days=10)
        res = compute_vol_gate(self.df, future)
        self.assertEqual(res.regime, "unknown")
        # Should be stale
        self.assertIn("stale or missing", res.reason)

    def test_spike(self):
        # Introduce volatility
        # Make the last hour volatile
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=60*24*15, freq="1min")
        closes = np.ones(len(dates)) * 100.0
        
        # Add random noise to baseline
        np.random.seed(42)
        closes += np.random.normal(0, 0.1, len(dates))
        
        # Massive volatility in last 15 mins (alternating large moves)
        # This ensures high standard deviation of returns
        for i in range(1, 16):
            closes[-i] = 100.0 + (5.0 if i % 2 == 0 else -5.0)
        
        df = pd.DataFrame({"timestamp": dates, "close": closes})
        res = compute_vol_gate(df, dates[-1], high_pct=80, extreme_pct=99)
        
        # Vol15 should be very high relative to baseline (which had small noise)
        self.assertGreater(res.vol15_pct, 90)
        self.assertTrue(res.regime in ["high", "extreme"])
        if res.regime == "high":
            self.assertEqual(res.edge_add_cents, 2.0)
            self.assertEqual(res.kelly_mult, 0.5)
            
    def test_shock(self):
        # Immediate 5-min shock check
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=60*24*15, freq="1min")
        closes = np.ones(len(dates)) * 100.0
        # Background noise
        closes += np.random.normal(0, 0.1, len(dates))
        
        # Jump at end: at T-1 it's 100, at T it's 105
        # (Actually shock window is 5m, so change over 5 mins)
        closes[-1] = 105.0 
        
        df = pd.DataFrame({"timestamp": dates, "close": closes})
        res = compute_vol_gate(df, dates[-1], enable_shock_gate=True, shock_pct=90.0)
        
        self.assertTrue(res.shock)
        self.assertEqual(res.regime, "extreme")
        self.assertFalse(res.allow_new_entries)
        self.assertEqual(res.kelly_mult, 0.0)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute BTC Volatility Gate")
    parser.add_argument("--now", type=str, help="ISO timestamp (UTC) for analysis. Default: now")
    parser.add_argument("--file", type=str, default="DATA/btc_intraday_1m.csv", help="Path to BTC CSV")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    
    args = parser.parse_args()
    
    if args.test:
        # Run tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestVolGate)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
        
    # Get Time
    if args.now:
        now_utc = args.now
    else:
        now_utc = datetime.now(timezone.utc).isoformat()
        
    # Load Data
    path = Path(args.file)
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)
        
    try:
        logger.info(f"Loading data from {path}...")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows.")
        
        # Compute
        result = compute_vol_gate(df, now_utc)
        
        # Output JSON
        import json
        output = {
            "now_utc": result.now_utc,
            "regime": result.regime,
            "vol15": result.vol15,
            "vol60": result.vol60,
            "vol15_pct": result.vol15_pct,
            "shock": result.shock,
            "allow_new_entries": result.allow_new_entries,
            "edge_add_cents": result.edge_add_cents,
            "kelly_mult": result.kelly_mult,
            "reason": result.reason
        }
        
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        logger.exception("Error computing vol gate")
        sys.exit(1)

if __name__ == "__main__":
    main()
