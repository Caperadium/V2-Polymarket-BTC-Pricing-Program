#!/usr/bin/env python3
"""
backrunner.py — absorbed from scripts/backtesting/prob_backrunner_engine.py

BackrunnerEngine class: time-travels through historical market data, running
the MC pricing engine at each point in time with only the data that would have
been available then.

Key design decisions (preserved from original):
  - Loads all BTC data once into memory with datetime index
  - Uses O(log n) DataFrame slicing per timestamp
  - No disk I/O inside the main loop (only final writes)
  - All timestamps normalized to UTC
  - Disk-native streaming write (each batch → one CSV, no in-memory accumulation)

Usage:
    python core/backtesting/backrunner.py --limit 10
    python core/backtesting/backrunner.py --skip-data-fetch --limit 5 --n-sims 5000
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

# Guard: ensure repo root is on sys.path when invoked as script
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.pricing.btc_pricing_engine import calculate_probabilities

logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path("DATA")
OUTPUT_ROOT = Path("backtested_probabilities")
UNFITTED_DIR = OUTPUT_ROOT / "unfitted"
FITTED_DIR = OUTPUT_ROOT / "fitted"

# ---------------------------------------------------------------------------
# BackrunnerEngine
# ---------------------------------------------------------------------------

class BackrunnerEngine:
    """Time-travel MC pricing engine.

    Absorbs ``prob_backrunner_engine.py`` lines 73-356 logic while
    keeping the disk-native streaming-write pattern.
    """

    def __init__(
        self,
        n_sims: int = 15000,
        seed: int = 42,
        advanced_features: bool = True,
        unfitted_dir: Optional[Path] = None,
        fitted_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.n_sims = n_sims
        self.seed = seed
        self.advanced_features = advanced_features
        self.unfitted_dir = unfitted_dir or UNFITTED_DIR
        self.fitted_dir = fitted_dir or FITTED_DIR
        self._progress = progress_callback

    # ------------------------------------------------------------------
    # BTC data loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_btc_data() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Load BTC daily, intraday, and hourly data with datetime index (UTC).

        Returns
        -------
        (daily_df, intraday_df, hourly_df)
            hourly_df may be None if the file is missing or unparseable.
        """
        daily_path = DATA_DIR / "btc_daily.csv"
        intraday_path = DATA_DIR / "btc_intraday_1m.csv"
        hourly_path = DATA_DIR / "btc_hourly.csv"

        if not daily_path.exists() or not intraday_path.exists():
            raise FileNotFoundError(
                f"BTC data files not found. Run data_fetcher.py first.\n"
                f"  Expected: {daily_path}, {intraday_path}, {hourly_path}"
            )

        # --- daily ---
        daily_df = pd.read_csv(daily_path)
        daily_col_map = {c.lower(): c for c in daily_df.columns}
        date_col = daily_col_map.get("date", daily_col_map.get("timestamp"))
        if date_col is None:
            raise ValueError("Daily CSV missing 'date' or 'timestamp' column")
        daily_df["datetime"] = pd.to_datetime(daily_df[date_col], utc=True)
        daily_df = daily_df.set_index("datetime").sort_index()

        logger.info(
            "Loaded daily data: %d rows, %s to %s",
            len(daily_df), daily_df.index.min(), daily_df.index.max(),
        )

        # --- intraday ---
        intraday_df = pd.read_csv(intraday_path)
        intra_col_map = {c.lower(): c for c in intraday_df.columns}
        ts_col = intra_col_map.get(
            "timestamp",
            intra_col_map.get("date", intra_col_map.get("datetime")),
        )
        if ts_col is None:
            raise ValueError("Intraday CSV missing 'Timestamp' column")
        intraday_df["datetime"] = pd.to_datetime(intraday_df[ts_col], utc=True)
        intraday_df = intraday_df.set_index("datetime").sort_index()

        logger.info(
            "Loaded intraday data: %d rows, %s to %s",
            len(intraday_df), intraday_df.index.min(), intraday_df.index.max(),
        )

        # --- hourly ---
        hourly_df: Optional[pd.DataFrame] = None
        if hourly_path.exists():
            hourly_df = pd.read_csv(hourly_path)
            hourly_col_map = {c.lower(): c for c in hourly_df.columns}
            h_date_col = hourly_col_map.get("date", hourly_col_map.get("timestamp"))
            if h_date_col is not None:
                hourly_df["datetime"] = pd.to_datetime(
                    hourly_df[h_date_col], utc=True
                )
                hourly_df = hourly_df.set_index("datetime").sort_index()
                logger.info(
                    "Loaded hourly data: %d rows, %s to %s",
                    len(hourly_df), hourly_df.index.min(), hourly_df.index.max(),
                )
            else:
                logger.warning("Hourly CSV missing 'date' column — GARCH will fall back")
                hourly_df = None
        else:
            logger.warning("Hourly data file not found: %s", hourly_path)

        return daily_df, intraday_df, hourly_df

    # ------------------------------------------------------------------
    # Market data loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_market_df(csv_path: Optional[str] = None) -> pd.DataFrame:
        """Load historical market prices with UTC timestamp normalization.

        If *csv_path* is None, tries the new ContractPriceStore
        (``DATA/historical_contract_prices.csv``) first, then falls back to
        ``DATA/old_market_prices.csv``.

        Returns DataFrame with 'date' column as UTC datetime.
        """
        from core.backtesting.contract_store import ContractPriceStore

        if csv_path is None:
            # Try new store first
            store = ContractPriceStore()
            store.load()
            if not store.df.empty:
                logger.info(
                    "Loaded market prices from store: %d rows, %d unique timestamps",
                    len(store.df), store.df["date"].nunique(),
                )
                return store.to_market_df()
            # Fallback to legacy CSV
            csv_path = "DATA/old_market_prices.csv"

        market_df = pd.read_csv(csv_path)
        if "date" not in market_df.columns:
            raise ValueError("Market prices CSV missing 'date' column")
        market_df["date"] = pd.to_datetime(market_df["date"], utc=True)
        logger.info(
            "Loaded market prices: %d rows, %d unique timestamps",
            len(market_df), market_df["date"].nunique(),
        )
        return market_df

    # ------------------------------------------------------------------
    # Main time-travel loop
    # ------------------------------------------------------------------

    def run(
        self,
        market_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame,
        hourly_df: Optional[pd.DataFrame] = None,
        limit: Optional[int] = None,
    ) -> Path:
        """Run the time-travel backtest loop.

        Iterates each unique timestamp in *market_df*, truncates BTC data to
        that timestamp, groups contracts by expiry, runs MC pricing, and
        writes per-timestamp batch CSVs to ``self.unfitted_dir``.

        Parameters
        ----------
        market_df : pd.DataFrame
            Columns: slug, strike, market_price, date, expiry_date.
        daily_df, intraday_df : pd.DataFrame
            BTC data loaded via :meth:`load_btc_data`.
        hourly_df : pd.DataFrame or None
            Hourly BTC data for GARCH fitting.
        limit : int or None
            Cap the number of timestamps processed.

        Returns
        -------
        Path
            The output directory containing batch_*.csv files.
        """
        self.unfitted_dir.mkdir(parents=True, exist_ok=True)

        unique_timestamps = sorted(market_df["date"].unique())
        if limit:
            unique_timestamps = unique_timestamps[:limit]
            logger.info("Limited to first %d timestamps", limit)

        n_total = len(unique_timestamps)
        logger.info("Processing %d unique timestamps...", n_total)

        for i, ts in enumerate(unique_timestamps):
            ts_dt = pd.Timestamp(ts).to_pydatetime()
            ts_str = ts_dt.strftime("%Y%m%d_%H%M%S")
            output_path = self.unfitted_dir / f"batch_{ts_str}.csv"

            # Idempotent: skip already-processed timestamps
            if output_path.exists():
                logger.debug("Already exists: %s", output_path.name)
                if self._progress:
                    self._progress("pricing", i + 1, n_total)
                continue

            # --- contracts at this timestamp ---
            contracts = market_df[market_df["date"] == ts].copy()
            if contracts.empty:
                if self._progress:
                    self._progress("pricing", i + 1, n_total)
                continue

            ts_date = pd.Timestamp(ts).normalize().tz_localize(None)

            # --- truncate intraday data: include all rows <= ts ---
            intraday_slice = intraday_df[intraday_df.index <= ts]
            if intraday_slice.empty:
                logger.warning("Skipping %s: no intraday data available", ts_str)
                if self._progress:
                    self._progress("pricing", i + 1, n_total)
                continue
            intraday_for_engine = intraday_slice.reset_index(drop=True)

            # --- truncate hourly data for GARCH (time-travel) ---
            hourly_for_engine: Optional[pd.DataFrame] = None
            if hourly_df is not None and not hourly_df.empty:
                hourly_slice = hourly_df[
                    hourly_df.index.tz_localize(None).normalize() <= ts_date
                ]
                if len(hourly_slice) < 500:
                    logger.warning(
                        "Skipping %s: insufficient hourly data (%d rows, need >=500)",
                        ts_str, len(hourly_slice),
                    )
                    if self._progress:
                        self._progress("pricing", i + 1, n_total)
                    continue
                hourly_for_engine = hourly_slice.reset_index(drop=True)
            else:
                logger.warning(
                    "Skipping %s: no hourly data available for GARCH fitting", ts_str
                )
                if self._progress:
                    self._progress("pricing", i + 1, n_total)
                continue

            # --- validate expiry dates ---
            if "expiry_date" not in contracts.columns or not contracts["expiry_date"].notna().any():
                logger.warning("Skipping %s: no expiry_date column", ts_str)
                if self._progress:
                    self._progress("pricing", i + 1, n_total)
                continue

            valid_expiry_mask = contracts["expiry_date"].notna()
            contracts = contracts[valid_expiry_mask]
            if contracts.empty:
                logger.warning("Skipping %s: no contracts with valid expiry dates", ts_str)
                if self._progress:
                    self._progress("pricing", i + 1, n_total)
                continue

            # --- group by expiry_date and price each group ---
            strikes = contracts["strike"].unique().tolist()
            results = []

            for expiry, group in contracts.groupby("expiry_date"):
                try:
                    expiry_dt = pd.to_datetime(expiry, utc=True)
                    hours_to_expiry = max(
                        (expiry_dt - ts_dt).total_seconds() / 3600, 0.001
                    )
                except Exception:
                    logger.warning("Could not parse expiry date: %s", expiry)
                    continue

                if hours_to_expiry <= 0:
                    logger.debug(
                        "Skipping expired group: expiry=%s, hours=%.1f",
                        expiry, hours_to_expiry,
                    )
                    continue

                group_strikes = group["strike"].unique().tolist()

                try:
                    probs = calculate_probabilities(
                        strikes=group_strikes,
                        hours_to_expiry=hours_to_expiry,
                        hourly_df=hourly_for_engine,
                        intraday_df=intraday_for_engine,
                        n_sims=self.n_sims,
                        seed=self.seed,
                        use_svcj=self.advanced_features,
                        use_skewed_t=self.advanced_features,
                        use_figarch=self.advanced_features,
                        use_regime_switching=self.advanced_features,
                        use_xgb_direction=self.advanced_features,
                    )

                    for _, row in group.iterrows():
                        strike = row["strike"]
                        results.append({
                            "slug": row.get("slug", ""),
                            "strike": strike,
                            "market_price": row["market_price"],
                            "model_probability": probs.get(strike, np.nan),
                            "T_days": hours_to_expiry / 24.0,
                            "date": ts_dt,
                            "expiry_date": expiry,
                        })
                except Exception:
                    logger.warning(
                        "Error calculating probs for %s, expiry %s",
                        ts_str, expiry, exc_info=True,
                    )

            # --- save batch ---
            if results:
                result_df = pd.DataFrame(results)
                result_df.to_csv(output_path, index=False)

            # progress
            if self._progress:
                self._progress("pricing", i + 1, n_total)

            if (i + 1) % 10 == 0 or i == n_total - 1:
                logger.info("Progress: %d/%d timestamps processed", i + 1, n_total)

        logger.info("Backtest loop complete. Results saved to %s", self.unfitted_dir)
        return self.unfitted_dir

    # ------------------------------------------------------------------
    # Curve fitting
    # ------------------------------------------------------------------

    def run_curve_fitting(self) -> int:
        """Run curve fitting on each batch file in the unfitted results.

        Uses direct function call (not subprocess) for speed.

        Returns
        -------
        int
            Number of batch files successfully fitted.
        """
        from core.pricing.fit_probability_curves import process_batch

        logger.info("Running curve fitting on unfitted results...")
        self.fitted_dir.mkdir(parents=True, exist_ok=True)

        batch_files = sorted(self.unfitted_dir.glob("batch_*.csv"))
        if not batch_files:
            logger.warning("No batch files found in %s", self.unfitted_dir)
            return 0

        logger.info("Found %d batch files to fit", len(batch_files))
        success_count = 0

        for i, batch_file in enumerate(batch_files):
            output_dir = self.fitted_dir / batch_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            output_batch_csv = output_dir / "batch_with_fits.csv"
            output_curve_csv = output_dir / "curve_params.csv"

            try:
                process_batch(
                    input_csv=str(batch_file),
                    output_batch_csv=str(output_batch_csv),
                    output_curve_params_csv=str(output_curve_csv),
                    use_rn_prob=False,
                )
                success_count += 1
            except Exception:
                logger.debug("Curve fitting failed for %s", batch_file.name, exc_info=True)

            if self._progress:
                self._progress("fitting", i + 1, len(batch_files))

            if (i + 1) % 50 == 0 or i == len(batch_files) - 1:
                logger.info(
                    "Curve fitting progress: %d/%d files", i + 1, len(batch_files)
                )

        logger.info(
            "Curve fitting complete. %d/%d files fitted successfully.",
            success_count, len(batch_files),
        )
        return success_count


# ---------------------------------------------------------------------------
# CLI entrypoint — preserved from prob_backrunner_engine.main()
# ---------------------------------------------------------------------------

def main() -> int:
    """Preserved CLI interface from prob_backrunner_engine.py.

    Note: automatic BTC data refresh (subprocess call to data_fetcher.py)
    has been removed. Run ``python core/data/data_fetcher.py`` manually first.
    """
    parser = argparse.ArgumentParser(
        description="Backtest pricing engine across historical market data"
    )
    parser.add_argument(
        "--skip-data-fetch",
        action="store_true",
        help="Deprecated flag (kept for compat). BTC data fetch is now separate.",
    )
    parser.add_argument(
        "--skip-fitting",
        action="store_true",
        help="Skip running curve fitting after backtest",
    )
    parser.add_argument(
        "--market-prices",
        type=str,
        default=None,
        help="Path to historical market prices CSV (default: auto-detect new store or legacy CSV)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of timestamps to process",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=50000,
        help="Number of Monte Carlo simulations per pricing (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible MC pricing (default: 42)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--advanced-features",
        action="store_true",
        default=True,
        dest="advanced_features",
        help="Enable SVCJ+skewed-t+FIGARCH+regime+XGBoost (default: on)",
    )
    parser.add_argument(
        "--no-advanced-features",
        action="store_false",
        dest="advanced_features",
        help="Disable all advanced features (plain GARCH+t+Kou baseline)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create engine
    engine = BackrunnerEngine(
        n_sims=args.n_sims,
        seed=args.seed,
        advanced_features=args.advanced_features,
    )

    # Load data
    try:
        daily_df, intraday_df, hourly_df = engine.load_btc_data()
        market_df = engine.load_market_df(args.market_prices)
    except Exception:
        logger.exception("Failed to load data")
        return 1

    # Run backtest loop
    engine.run(
        market_df=market_df,
        hourly_df=hourly_df,
        intraday_df=intraday_df,
        daily_df=daily_df,
        limit=args.limit,
    )

    # Curve fitting
    if not args.skip_fitting:
        engine.run_curve_fitting()

    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")
    print(f"Unfitted results: {engine.unfitted_dir}")
    print(f"Fitted results:   {engine.fitted_dir}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
