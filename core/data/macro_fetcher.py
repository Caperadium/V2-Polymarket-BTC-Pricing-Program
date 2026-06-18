"""
macro_fetcher.py

Download macroeconomic data for BTC regime detection and directional prediction.
Fetches daily data from Yahoo Finance (free, no API key required):
  - Gold (XAU/USD via GC=F)
  - DXY (USD Index via DX-Y.NYB)
  - VIX (CBOE Volatility Index via ^VIX)
  - SPX (S&P 500 via ^GSPC)

Based on: Köse et al. (2025), Kim et al. (2025), Pakstaite et al. (2025).
Evidence: Gold attention weight 0.85 (Köse TFT), macro drivers dominate post-2019.

Usage:
    python core/data/macro_fetcher.py
    python core/data/macro_fetcher.py --days 730

Or programmatically:
    from core.data.macro_fetcher import fetch_macro_data, load_macro_data
    fetch_macro_data()  # Download latest
    df = load_macro_data()  # Load from disk
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "DATA"
MACRO_PATH = DATA_DIR / "macro_daily.csv"

# Yahoo Finance tickers (free, no auth required)
TICKERS = {
    "gold": "GC=F",        # Gold Futures
    "dxy": "DX-Y.NYB",     # US Dollar Index
    "vix": "^VIX",         # CBOE Volatility Index
    "spx": "^GSPC",        # S&P 500
}

# yfinance periods
DEFAULT_PERIOD = "5y"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_macro_data(
    days: int = None,
    period: str = DEFAULT_PERIOD,
    tickers: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch macro data from Yahoo Finance.

    Args:
        days: Number of days of data (alternative to period).
        period: Yahoo Finance period string (e.g., "5y", "2y", "10y").
        tickers: Dict of name -> Yahoo Finance ticker.
        output_path: Path to save CSV (default: DATA/macro_daily.csv).

    Returns:
        DataFrame with columns: date, gold, dxy, vix, spx, gold_btc_corr, ...
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Install with: pip install yfinance")
        raise

    if tickers is None:
        tickers = TICKERS

    if output_path is None:
        output_path = str(MACRO_PATH)

    logger.info(f"Fetching macro data for period={period}...")

    all_data = {}

    for name, ticker in tickers.items():
        try:
            logger.debug(f"Downloading {name} ({ticker})...")
            data = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True,
            )

            if data is None or data.empty:
                logger.warning(f"No data returned for {name} ({ticker})")
                continue

            # Get close prices
            if isinstance(data.columns, pd.MultiIndex):
                close = data['Close'].iloc[:, 0]
            else:
                close = data['Close']

            close = close.rename(name)
            all_data[name] = close
            logger.info(f"Downloaded {name}: {len(close)} rows")

        except Exception as e:
            logger.warning(f"Failed to fetch {name} ({ticker}): {e}")
            continue

    if not all_data:
        raise RuntimeError("No macro data fetched. Check ticker symbols and network connection.")

    # Merge all series on date index
    df = pd.DataFrame(all_data)
    df.index.name = "date"

    # Forward-fill missing values (markets may have different holiday schedules)
    df = df.ffill().dropna()

    # Compute additional features
    if len(df) > 1:
        # Daily returns for each series
        for col in df.columns:
            df[f"{col}_ret"] = df[col].pct_change()

        # Rolling BTC-Gold correlation (30-day window per Köse 2025)
        # Won't have BTC here; add later in pipeline when BTC data available

        # VIX level classification
        df["vix_regime"] = pd.cut(
            df["vix"],
            bins=[0, 15, 25, 100],
            labels=["low", "medium", "high"],
        )

        # DXY trend (20-day SMA slope)
        df["dxy_trend"] = np.sign(df["dxy"].rolling(20).mean().diff())

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info(f"Saved macro data to {output_path}: {len(df)} rows, {len(df.columns)} columns")

    return df


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_macro_data(
    path: Optional[str] = None,
    min_rows: int = 60,
) -> Optional[pd.DataFrame]:
    """
    Load macro data from disk.

    Args:
        path: Path to macro CSV (default: DATA/macro_daily.csv).
        min_rows: Minimum rows required (otherwise return None).

    Returns:
        DataFrame with date index, or None if file missing or insufficient data.
    """
    if path is None:
        path = str(MACRO_PATH)

    p = Path(path)
    if not p.exists():
        logger.warning(f"Macro data file not found: {path}")
        return None

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if len(df) < min_rows:
        logger.warning(f"Macro data insufficient: {len(df)} rows < {min_rows} minimum")
        return None

    logger.info(f"Loaded macro data: {len(df)} rows, columns={list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Merge with BTC Data
# ---------------------------------------------------------------------------

def merge_with_btc(
    btc_path: str = "DATA/btc_hourly.csv",
    macro_path: Optional[str] = None,
    resample: str = "D",
) -> Optional[pd.DataFrame]:
    """
    Merge macro data with daily BTC prices for regime detection.

    Args:
        btc_path: Path to BTC hourly data.
        macro_path: Path to macro data (default: DATA/macro_daily.csv).
        resample: Resampling frequency for BTC data ("D" = daily).

    Returns:
        DataFrame with BTC + macro features, date index.
    """
    macro_df = load_macro_data(path=macro_path)
    if macro_df is None:
        return None

    # Load BTC daily prices
    if not Path(btc_path).exists():
        logger.warning(f"BTC data not found: {btc_path}")
        return macro_df

    btc = pd.read_csv(btc_path)

    # Find date and close columns
    col_map = {c.lower(): c for c in btc.columns}
    date_col = col_map.get("date", col_map.get("timestamp"))
    close_col = col_map.get("close")

    if date_col and close_col:
        btc[date_col] = pd.to_datetime(btc[date_col], utc=True)
        btc = btc.set_index(date_col)
        btc_daily = btc[close_col].resample(resample).last().rename("btc").ffill()
    else:
        return macro_df

    # Merge
    merged = macro_df.join(btc_daily, how="inner")

    # Compute BTC returns
    if "btc" in merged.columns:
        merged["btc_ret"] = merged["btc"].pct_change()

    # Compute BTC-Gold rolling correlation (30-day, per Köse 2025)
    if "btc_ret" in merged.columns and "gold_ret" in merged.columns:
        merged["btc_gold_corr_30d"] = (
            merged["btc_ret"]
            .rolling(30)
            .corr(merged["gold_ret"])
        )

    # Compute BTC-DXY rolling correlation
    if "btc_ret" in merged.columns and "dxy_ret" in merged.columns:
        merged["btc_dxy_corr_30d"] = (
            merged["btc_ret"]
            .rolling(30)
            .corr(merged["dxy_ret"])
        )

    merged = merged.dropna()

    logger.info(f"Merged BTC+macro data: {len(merged)} rows")
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Fetch macroeconomic data for BTC pricing")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="Yahoo Finance period (e.g., 5y, 2y, max)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--merge", action="store_true", help="Also merge with BTC data")
    args = parser.parse_args()

    df = fetch_macro_data(period=args.period, output_path=args.output)

    print(f"\n=== Macro Data Summary ===")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLatest values:")
    for col in ["gold", "dxy", "vix", "spx"]:
        if col in df.columns:
            print(f"  {col.upper()}: {df[col].iloc[-1]:.2f}")

    if args.merge:
        merged = merge_with_btc()
        if merged is not None:
            print(f"\nMerged BTC+macro: {len(merged)} rows")
            if "btc_gold_corr_30d" in merged.columns:
                print(f"  BTC-Gold 30d corr (latest): {merged['btc_gold_corr_30d'].iloc[-1]:.4f}")
