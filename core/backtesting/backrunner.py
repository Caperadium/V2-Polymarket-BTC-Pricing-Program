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

Parallelism (v2):
  - ProcessPoolExecutor with per-worker data loading via initializer
  - BLAS thread suppression (OMP/MKL/OPENBLAS/NUMEXPR=1) per worker
  - Deterministic per-timestamp seeds via hashlib.md5
  - S0 daily-close fallback pre-computed in main process
  - Progress callback stays in main process via as_completed() loop

Usage:
    python core/backtesting/backrunner.py --limit 10
    python core/backtesting/backrunner.py --limit 10 --workers 8
    python core/backtesting/backrunner.py --serial --limit 5 --n-sims 5000
"""

import argparse
import hashlib
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Guard: ensure repo root is on sys.path when invoked as script
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# Directories — resolved relative to this file so CWD doesn't matter (Streamlit, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "DATA"
OUTPUT_ROOT = _PROJECT_ROOT / "backtested_probabilities"
UNFITTED_DIR = OUTPUT_ROOT / "unfitted"
FITTED_DIR = OUTPUT_ROOT / "fitted"

# ---------------------------------------------------------------------------
# Worker globals — populated by _init_worker() in each spawned process
# ---------------------------------------------------------------------------

_worker_daily: Optional[pd.DataFrame] = None
_worker_intraday: Optional[pd.DataFrame] = None
_worker_hourly: Optional[pd.DataFrame] = None
_worker_macro: Optional[pd.DataFrame] = None


def _init_worker(data_dir: str) -> None:
    """Worker process initializer: suppress BLAS threads, load BTC data.

    Called once per spawned process BEFORE any work items are dispatched.
    Sets OMP/MKL/OPENBLAS/NUMEXPR thread count to 1 so N workers don't
    each spawn M BLAS threads and thrash the CPU.

    Parameters
    ----------
    data_dir : str
        Path to the DATA/ directory containing BTC CSV files.
    """
    # ---- BLAS thread suppression (must be set before any numpy compute) ----
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    global _worker_daily, _worker_intraday, _worker_hourly, _worker_macro

    data_path = Path(data_dir)

    # --- daily ---
    daily_path = data_path / "btc_daily.csv"
    daily_df = pd.read_csv(daily_path)
    daily_col_map = {c.lower(): c for c in daily_df.columns}
    date_col = daily_col_map.get("date", daily_col_map.get("timestamp"))
    if date_col is None:
        raise ValueError("Daily CSV missing 'date' or 'timestamp' column")
    daily_df["datetime"] = pd.to_datetime(daily_df[date_col], utc=True)
    _worker_daily = daily_df.set_index("datetime").sort_index()

    # --- intraday ---
    intraday_path = data_path / "btc_intraday_1m.csv"
    intraday_df = pd.read_csv(intraday_path)
    intra_col_map = {c.lower(): c for c in intraday_df.columns}
    ts_col = intra_col_map.get(
        "timestamp", intra_col_map.get("date", intra_col_map.get("datetime"))
    )
    if ts_col is None:
        raise ValueError("Intraday CSV missing 'Timestamp' column")
    intraday_df["datetime"] = pd.to_datetime(intraday_df[ts_col], utc=True)
    _worker_intraday = intraday_df.set_index("datetime").sort_index()

    # --- hourly ---
    hourly_path = data_path / "btc_hourly.csv"
    if hourly_path.exists():
        hourly_df = pd.read_csv(hourly_path)
        hourly_col_map = {c.lower(): c for c in hourly_df.columns}
        h_date_col = hourly_col_map.get("date", hourly_col_map.get("timestamp"))
        if h_date_col is not None:
            hourly_df["datetime"] = pd.to_datetime(hourly_df[h_date_col], utc=True)
            _worker_hourly = hourly_df.set_index("datetime").sort_index()

    # --- macro (optional; only needed for XGB directional drift) ---
    macro_path = data_path / "macro_daily.csv"
    if macro_path.exists():
        macro_df = pd.read_csv(macro_path, index_col=0)
        macro_df.index = pd.to_datetime(macro_df.index, utc=True)
        _worker_macro = macro_df.sort_index()


def _process_one(item: Dict[str, Any]) -> Optional[str]:
    """Process a single timestamp — runs in worker process.

    Parameters
    ----------
    item : dict
        Work item dict with keys: ts_str, ts_iso, ts_date, contracts,
        s0_from_daily, output_path, seed, n_sims, advanced_features.

    Returns
    -------
    str or None
        ts_str on success (batch CSV written), None on skip/failure.
    """
    # Lazy import inside worker so it picks up BLAS-suppressed env
    from core.pricing.btc_pricing_engine import (
        calculate_probabilities, dte_bucket_horizon,
    )
    from core.pricing.engine_config import build_engine_kwargs

    ts_str: str = item["ts_str"]
    ts_iso: str = item["ts_iso"]
    ts_date_str: str = item["ts_date"]
    contracts: List[Dict[str, Any]] = item["contracts"]
    s0_from_daily: Optional[float] = item["s0_from_daily"]
    output_path: str = item["output_path"]
    seed: int = item["seed"]
    n_sims: int = item["n_sims"]
    advanced_features: bool = item["advanced_features"]
    use_xgb: bool = item.get("use_xgb", False)
    xgb_tilt_lambda = item.get("xgb_tilt_lambda", None)

    ts_dt = pd.Timestamp(ts_iso)

    # ---- truncate intraday data ----
    # Strict '<': BTC bars are open-stamped (Binance klines), so a bar timestamped
    # at ts_dt closes ~1 bar in the FUTURE relative to the snapshot instant. Using
    # '<=' would leak that bar's forward close into S0. The close of the last bar
    # strictly before ts_dt is exactly the price known AT the snapshot.
    intraday_slice = _worker_intraday[_worker_intraday.index < ts_dt]
    if intraday_slice.empty:
        if s0_from_daily is None:
            logger.warning("Skipping %s: no intraday or daily data available", ts_str)
            return None
        intraday_for_engine = pd.DataFrame({"close": [s0_from_daily]})
    else:
        intraday_for_engine = intraday_slice.reset_index(drop=True)

    # ---- truncate hourly data for GARCH ----
    if _worker_hourly is None or _worker_hourly.empty:
        logger.warning(
            "Skipping %s: no hourly data available for GARCH fitting", ts_str
        )
        return None

    # Strict '<' at the exact snapshot timestamp (NOT end-of-day). Bars are
    # open-stamped, so '<=' / a date-level cutoff would pull in the hours AFTER
    # the midnight snapshot — a lookahead leak straight into the GARCH vol
    # estimate (conditional variance is dominated by the most recent returns).
    # This now matches the intraday cutoff above.
    hourly_slice = _worker_hourly[_worker_hourly.index < ts_dt]
    if len(hourly_slice) < 500:
        logger.warning(
            "Skipping %s: insufficient hourly data (%d rows, need >=500)",
            ts_str, len(hourly_slice),
        )
        return None
    hourly_for_engine = hourly_slice.reset_index(drop=True)

    # ---- FIX 2 (M1) + FIX 4 (H1): per-snapshot, LEAK-FREE jump calibration and
    # regime detector, computed ONCE from the strict-`<` truncated hourly slice
    # (never the full file). Shared across all expiry groups in this snapshot.
    #
    # Leak-critical: build the returns array from `hourly_for_engine['close']` and
    # pass it as `returns=` to calibrate_jumps. NEVER pass `hourly_csv=` — that
    # default reads the FULL DATA/btc_hourly.csv and would leak future bars.
    jump_params: Optional[Dict[str, Any]] = None
    regime_params: Optional[Dict[str, Any]] = None
    detector = None
    if advanced_features:
        from core.pricing.jump_calibration import calibrate_jumps
        from core.pricing.btc_pricing_engine import build_regime_jump_params
        from core.pricing.regime_detector import RegimeDetector

        h_col_map = {c.lower(): c for c in hourly_for_engine.columns}
        h_close = h_col_map.get("close")
        if h_close is not None:
            returns_arr = (
                np.log(
                    hourly_for_engine[h_close]
                    / hourly_for_engine[h_close].shift(1)
                )
                .dropna()
                .to_numpy()
            )
            try:
                # FIX 2 (also M4): bipower detection — less vol-cluster contamination.
                cal = calibrate_jumps(returns=returns_arr, detection_method="bipower")
                if cal.fit_converged:
                    jump_params = {
                        "lambda": cal.lam,
                        "crash_prob": cal.p_crash,
                        "eta_up": cal.eta_up,
                        "eta_down": cal.eta_down,
                        "mu_v": cal.mu_v,
                        "rho_J": cal.rho_J,
                        # FIX 4 (M1): the SVCJ return-vol regression slope actually
                        # used in simulate_paths (rho_J above is reporting-only).
                        "rho_j_slope": cal.rho_j_slope,
                    }
                    # Per-regime jumps = calibrated base × literature multipliers.
                    # Avoids calibrate_regime_jumps (whose synthetic-timestamp path
                    # falls back to wall-clock — a leak in time-travel).
                    regime_params = build_regime_jump_params(
                        calibrated={**jump_params, "lam": cal.lam,
                                    "p_crash": cal.p_crash, "rho_J": cal.rho_J,
                                    "rho_j_slope": cal.rho_j_slope,
                                    "fit_converged": True}
                    )
            except Exception:
                logger.debug("Jump calibration failed for %s", ts_str, exc_info=True)

        # FIX 4 (H1): stateful HMM regime detector. calculate_probabilities fits it
        # on the truncated daily returns with now=as_of (leak-free, deterministic)
        # and caches across this snapshot's expiry groups. Constructed here so the
        # cache is shared; a fresh detector per snapshot prevents cross-snapshot leak.
        try:
            detector = RegimeDetector()
        except Exception:
            detector = None

    # ---- XGB directional drift (FIX 3 re-enabled, C2-a per-DTE-bucket) ----
    # Per-snapshot leak-free setup, mirroring the jump/regime discipline above.
    # The daily-return series and macro slice are derived from the strict-`<`
    # truncated data; one XGB model is trained per DTE bucket and cached for this
    # snapshot (snapshots are daily, so this is the per-(date,bucket) cache, D2).
    xgb_daily_ret = None
    xgb_macro_slice = None
    xgb_model_cache: Dict[float, Any] = {}
    if use_xgb:
        from core.pricing.directional_xgb import (
            DirectionalXGB, to_daily_log_return_series,
        )
        xgb_daily_ret = to_daily_log_return_series(hourly_for_engine)
        if _worker_macro is not None and not _worker_macro.empty:
            # Leak guard: macro rows strictly before the snapshot instant.
            xgb_macro_slice = _worker_macro[_worker_macro.index < ts_dt]
            if xgb_macro_slice.empty:
                xgb_macro_slice = None
        if xgb_macro_slice is None:
            logger.warning(
                "XGB enabled but no leak-free macro available at %s; running "
                "BTC-only (directional signal expected weak — plan §8.1).", ts_str,
            )

    def _get_xgb_model(bucket_h: float):
        """Train-or-fetch the cached per-bucket XGB model for this snapshot."""
        if bucket_h in xgb_model_cache:
            return xgb_model_cache[bucket_h]
        model = None
        try:
            m = DirectionalXGB()
            if m.train_from_slice(xgb_daily_ret, xgb_macro_slice, int(round(bucket_h))):
                model = m
        except Exception:
            logger.debug("XGB train failed (bucket %.0fd) at %s", bucket_h, ts_str,
                         exc_info=True)
        xgb_model_cache[bucket_h] = model
        return model

    # ---- validate contracts ----
    contracts_df = pd.DataFrame(contracts)
    if contracts_df.empty:
        return None

    # ---- group by expiry_date and price each group ----
    results: List[Dict[str, Any]] = []

    # Per-snapshot dedup: all expiry groups below share the SAME truncated hourly
    # slice, so the GARCH/FIGARCH MLE and S0 are identical across them. Fit/derive
    # once and reuse via these caches (passed into every calculate_probabilities
    # call). `snapshot_garch_cache` is keyed inside the engine on the effective
    # use_figarch flag (post horizon-gate), so the FIGARCH↔GARCH choice stays
    # correct. Case-insensitive 'close' lookup REQUIRED: the populated intraday
    # frame keeps raw CSV casing ("Close": Timestamp,Open,High,Low,Close,Volume),
    # while the daily-fallback frame above uses lowercase "close" — mirrors
    # load_and_prep_data's {c.lower(): c} map so S0 matches byte-for-byte.
    snapshot_garch_cache: Dict[bool, Any] = {}
    _close_col = {c.lower(): c for c in intraday_for_engine.columns}.get("close")
    try:
        snapshot_s0 = (
            float(intraday_for_engine[_close_col].iloc[-1])
            if _close_col is not None else None
        )
    except Exception:
        snapshot_s0 = None  # let calculate_probabilities derive it per group

    for expiry, group in contracts_df.groupby("expiry_date"):
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

        # FIX 3 (re-enabled): pick the per-DTE-bucket XGB model for this expiry.
        xgb_model = None
        if use_xgb:
            bucket_h = dte_bucket_horizon(hours_to_expiry / 24.0)
            if bucket_h is not None and xgb_daily_ret is not None:
                xgb_model = _get_xgb_model(bucket_h)

        try:
            # T5 (H2): single source of truth for the v2 engine flag bundle,
            # shared with the live pipelines so they cannot drift apart again.
            # See core/pricing/engine_config.py docstring for exactly which
            # keys are (by design) NOT covered here.
            engine_kwargs = build_engine_kwargs(
                advanced_features=advanced_features,
                detector=detector,
                regime_params=regime_params,
                # FIX 2 (M1): calibrated jumps (leak-free, per-snapshot, bipower).
                jump_params=jump_params,
                n_sims=n_sims,
                seed=seed,
                # FIX 4 (H1): as_of threaded for leak-free deterministic regime
                # refit gating during time-travel backtests.
                as_of=ts_dt,
                # FIX 3 (H2 re-enabled): XGBoost directional DRIFT shift (not the
                # old invalid per-strike blend). Per-DTE-bucket model + leak-free
                # macro.
                use_xgb=use_xgb,
                xgb_model=xgb_model,
                xgb_tilt_lambda=xgb_tilt_lambda,
                macro_df=xgb_macro_slice,
            )
            probs = calculate_probabilities(
                strikes=group_strikes,
                hours_to_expiry=hours_to_expiry,
                hourly_df=hourly_for_engine,
                intraday_df=intraday_for_engine,
                disable_staleness_check=True,
                # Per-snapshot dedup: fit GARCH/FIGARCH + derive S0 once, reuse
                # across this snapshot's expiry groups (byte-identical output).
                garch_cache=snapshot_garch_cache,
                s0_override=snapshot_s0,
                **engine_kwargs,
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
                    # Explicit snapshot timestamp so downstream batch loaders key
                    # chronology off the data, not the parsed folder name.
                    "batch_timestamp": ts_dt,
                    "expiry_date": expiry,
                })
        except Exception:
            logger.warning(
                "Error calculating probs for %s, expiry %s",
                ts_str, expiry, exc_info=True,
            )

    # ---- save batch (atomic: temp + os.replace) ----
    # A worker killed mid-write must NOT leave a truncated CSV that the
    # idempotent `output_path.exists()` skip would treat as complete. Write to a
    # unique temp file in the same dir, then atomically rename into place.
    if results:
        result_df = pd.DataFrame(results)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp")
        result_df.to_csv(tmp, index=False)
        os.replace(tmp, out)

    return ts_str


# ---------------------------------------------------------------------------
# BackrunnerEngine
# ---------------------------------------------------------------------------

class BackrunnerEngine:
    """Time-travel MC pricing engine.

    Absorbs ``prob_backrunner_engine.py`` lines 73-356 logic while
    keeping the disk-native streaming-write pattern.

    Supports parallel execution via ProcessPoolExecutor (--workers N)
    and serial execution for debugging (--serial).
    """

    def __init__(
        self,
        n_sims: int = 15000,
        seed: int = 42,
        advanced_features: bool = True,
        unfitted_dir: Optional[Path] = None,
        fitted_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        use_xgb: bool = False,
        xgb_tilt_lambda: Optional[float] = None,
    ):
        self.n_sims = n_sims
        self.seed = seed
        self.advanced_features = advanced_features
        self.unfitted_dir = unfitted_dir or UNFITTED_DIR
        self.fitted_dir = fitted_dir or FITTED_DIR
        self._progress = progress_callback
        # FIX 3 re-enabled: XGBoost directional drift. Default OFF → byte-identical
        # to pre-XGB backtests. xgb_tilt_lambda=None → engine module default (0.0).
        self.use_xgb = use_xgb
        self.xgb_tilt_lambda = xgb_tilt_lambda

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
            store_df = store.load()
            if not store_df.empty:
                logger.info(
                    "Loaded market prices from store: %d rows, %d unique timestamps",
                    len(store_df), store_df["date"].nunique(),
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
    # Pre-processing (builds work items for parallel dispatch)
    # ------------------------------------------------------------------

    def _preprocess_work_items(
        self,
        market_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame,
        limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Build work items for parallel processing.

        Pre-computes S0 daily-close fallbacks so workers don't need
        the full daily DataFrame. Derives deterministic per-timestamp
        seeds via hashlib.md5 (not Python ``hash()`` — broken across
        processes by PYTHONHASHSEED).

        Parameters
        ----------
        market_df : pd.DataFrame
            Columns: slug, strike, market_price, date, expiry_date.
        daily_df : pd.DataFrame
            BTC daily data, indexed by UTC datetime.
        intraday_df : pd.DataFrame
            BTC intraday data, indexed by UTC datetime.
        limit : int or None
            Cap the number of timestamps.

        Returns
        -------
        (work_items, n_total)
            n_total is the raw count of unique timestamps (used for progress
            denominator). work_items only includes processable timestamps.
        """
        # ---- floor contract timestamps to midnight UTC (defensive) ----
        # CLOB candle timestamps carry second-level jitter (00:00:1X-3X). Grouping
        # on exact equality scatters an expiry's ~11 strikes across as many
        # per-second "snapshots" (≈1.4 strikes each), starving the logistic curve
        # fit. Flooring to the day collapses them to one midnight snapshot with the
        # full strike ladder. Idempotent even if ingest already floors (item A#1).
        market_df = market_df.copy()
        market_df["date"] = pd.to_datetime(market_df["date"], utc=True).dt.floor("D")

        unique_timestamps = sorted(market_df["date"].unique())
        if limit:
            unique_timestamps = unique_timestamps[:limit]
            logger.info("Limited to first %d timestamps", limit)

        n_total = len(unique_timestamps)
        logger.info("Pre-processing %d unique timestamps...", n_total)

        # ---- precompute sorted index arrays for O(log n) lookups ----
        # Avoids O(n) boolean masking per timestamp (was O(n_ts * n_rows)).
        intraday_asi8 = intraday_df.index.asi8
        daily_asi8 = daily_df.index.asi8
        daily_col_map = {c.lower(): c for c in daily_df.columns}
        _daily_close_col = daily_col_map.get("close")
        daily_close_vals = (
            daily_df[_daily_close_col].to_numpy()
            if _daily_close_col is not None
            else None
        )

        work_items: List[Dict[str, Any]] = []
        _intraday_fallback_used = False

        for ts in unique_timestamps:
            ts_dt = pd.Timestamp(ts).to_pydatetime()
            ts_str = ts_dt.strftime("%Y%m%d_%H%M%S")
            output_path = self.unfitted_dir / f"batch_{ts_str}.csv"

            # Idempotent: skip already-processed timestamps
            if output_path.exists():
                logger.debug("Already exists: %s", output_path.name)
                continue

            # Contracts at this timestamp
            contracts = market_df[market_df["date"] == ts]
            if contracts.empty:
                continue

            # Validate expiry dates
            if "expiry_date" not in contracts.columns:
                continue
            contracts = contracts[contracts["expiry_date"].notna()]
            if contracts.empty:
                continue

            # ---- pre-compute S0 daily-close fallback (searchsorted, O(log n)) ----
            s0_from_daily: Optional[float] = None
            ts_val = pd.Timestamp(ts).value  # int64 ns, UTC
            intraday_pos = int(np.searchsorted(intraday_asi8, ts_val, side="right"))
            if intraday_pos == 0:  # no intraday rows <= ts
                if not _intraday_fallback_used:
                    logger.info(
                        "Intraday data unavailable for timestamps before %s; "
                        "using daily close as S0 fallback.",
                        intraday_df.index.min().strftime("%Y-%m-%d"),
                    )
                    _intraday_fallback_used = True

                if daily_close_vals is not None:
                    # Daily bars are indexed at day-start (D 00:00) but `close`
                    # is the END-of-day-D price (≈ D+1 00:00). The price actually
                    # KNOWN at the contract's midnight (D 00:00) is therefore the
                    # PRIOR day's close. Using row D would leak ~24h of lookahead.
                    cutoff = pd.Timestamp(ts).normalize().value  # D 00:00:00
                    daily_pos = int(np.searchsorted(daily_asi8, cutoff, side="left"))
                    if daily_pos > 0:
                        s0_from_daily = float(daily_close_vals[daily_pos - 1])

            # ---- build contract list (small — a few rows per timestamp) ----
            contract_list: List[Dict[str, Any]] = []
            for _, row in contracts.iterrows():
                contract_list.append({
                    "slug": str(row.get("slug", "")),
                    "strike": float(row["strike"]),
                    "market_price": float(row["market_price"]),
                    "expiry_date": str(row["expiry_date"]),
                })

            # ---- deterministic per-timestamp seed ----
            seed_bytes = hashlib.md5(ts_str.encode()).hexdigest()[:8]
            item_seed = self.seed + int(seed_bytes, 16)

            ts_date_str = pd.Timestamp(ts).normalize().strftime("%Y-%m-%d")

            work_items.append({
                "ts_str": ts_str,
                "ts_iso": ts_dt.isoformat(),
                "ts_date": ts_date_str,
                "contracts": contract_list,
                "s0_from_daily": s0_from_daily,
                "output_path": str(output_path),
                "seed": item_seed,
                "n_sims": self.n_sims,
                "advanced_features": self.advanced_features,
                "use_xgb": self.use_xgb,
                "xgb_tilt_lambda": self.xgb_tilt_lambda,
            })

        logger.info(
            "Pre-processed %d work items from %d timestamps (%d already cached)",
            len(work_items), n_total, n_total - len(work_items),
        )
        return work_items, n_total

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
        workers: Optional[int] = None,
    ) -> Path:
        """Run the time-travel backtest loop.

        Iterates each unique timestamp in *market_df*, truncates BTC data to
        that timestamp, groups contracts by expiry, runs MC pricing, and
        writes per-timestamp batch CSVs to ``self.unfitted_dir``.

        When *workers* > 1, uses ProcessPoolExecutor for parallel execution
        with per-worker BTC data loading via ``_init_worker()``.

        Parameters
        ----------
        market_df : pd.DataFrame
            Columns: slug, strike, market_price, date, expiry_date.
        daily_df, intraday_df : pd.DataFrame
            BTC data loaded via :meth:`load_btc_data`.
            ``daily_df`` is used as S0 fallback when intraday data is unavailable
            for a timestamp (intraday starts ~2026-03-01; daily goes back to 2021).
        hourly_df : pd.DataFrame or None
            Hourly BTC data for GARCH fitting.
        limit : int or None
            Cap the number of timestamps processed.
        workers : int or None
            Number of worker processes. None or 1 = serial mode.
            Default when called via CLI: ``min(cpu_count - 4, 12)``.

        Returns
        -------
        Path
            The output directory containing batch_*.csv files.
        """
        self.unfitted_dir.mkdir(parents=True, exist_ok=True)

        # ---- pre-process work items (S0 fallback, seeds, skips) ----
        work_items, n_total = self._preprocess_work_items(
            market_df, daily_df, intraday_df, limit
        )

        if not work_items:
            logger.info("No work items to process (all cached or empty)")
            return self.unfitted_dir

        # ---- resolve worker count ----
        # Clamp to work-item count: no point spawning more processes than tasks.
        if workers is not None and workers > 1:
            workers = min(workers, len(work_items))

        if workers is None or workers <= 1:
            self._run_serial(work_items, n_total, daily_df, intraday_df, hourly_df)
        else:
            self._run_parallel(work_items, n_total, workers)

        logger.info("Backtest loop complete. Results saved to %s", self.unfitted_dir)
        return self.unfitted_dir

    # ------------------------------------------------------------------
    # Serial execution (original loop logic, kept for debugging)
    # ------------------------------------------------------------------

    def _run_serial(
        self,
        work_items: List[Dict[str, Any]],
        n_total: int,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame,
        hourly_df: Optional[pd.DataFrame],
    ) -> None:
        """Process work items sequentially in the current process.

        Populates module-level worker globals from already-loaded DataFrames
        so ``_process_one()`` can be used unchanged.
        """
        logger.info("Processing %d timestamps (serial mode)...", len(work_items))

        # Populate worker globals from already-loaded data
        global _worker_daily, _worker_intraday, _worker_hourly, _worker_macro
        _worker_daily = daily_df
        _worker_intraday = intraday_df
        _worker_hourly = hourly_df

        # Macro (optional, XGB only) — serial path loads it directly since it is
        # not among the passed-in DataFrames.
        if self.use_xgb and _worker_macro is None:
            macro_path = DATA_DIR / "macro_daily.csv"
            if macro_path.exists():
                m = pd.read_csv(macro_path, index_col=0)
                m.index = pd.to_datetime(m.index, utc=True)
                _worker_macro = m.sort_index()

        # Seed with the already-cached count so Progress: X/n_total reaches 100%.
        completed = n_total - len(work_items)
        for i, item in enumerate(work_items):
            try:
                _process_one(item)
            except Exception:
                logger.warning(
                    "Worker failed for %s", item["ts_str"], exc_info=True,
                )
            completed += 1

            if self._progress:
                self._progress("pricing", completed, n_total)

            if (i + 1) % 10 == 0 or i == len(work_items) - 1:
                logger.info(
                    "Progress: %d/%d timestamps processed", completed, n_total
                )

    # ------------------------------------------------------------------
    # Parallel execution (ProcessPoolExecutor)
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        work_items: List[Dict[str, Any]],
        n_total: int,
        workers: int,
    ) -> None:
        """Process work items in parallel via ProcessPoolExecutor.

        Each worker loads its own BTC data via ``_init_worker()`` to avoid
        pickling DataFrames across process boundaries.
        """
        logger.info(
            "Processing %d timestamps (parallel mode, %d workers)...",
            len(work_items), workers,
        )

        # Seed with the already-cached count so Progress: X/n_total reaches 100%.
        completed = n_total - len(work_items)

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(DATA_DIR),),
        ) as executor:
            # Submit all tasks
            future_map = {
                executor.submit(_process_one, item): item["ts_str"]
                for item in work_items
            }

            # Collect results as they complete
            for future in as_completed(future_map):
                ts_str = future_map[future]
                try:
                    future.result()
                    completed += 1
                except Exception:
                    logger.warning(
                        "Worker failed for %s", ts_str, exc_info=True,
                    )
                    completed += 1

                if self._progress:
                    self._progress("pricing", completed, n_total)

                if completed % 10 == 0 or completed == len(work_items):
                    logger.info(
                        "Progress: %d/%d timestamps processed",
                        completed, n_total,
                    )

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
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    default_workers = min(max(cpu_count - 4, 1), 12)

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
        help="Enable SVCJ+skewed-t+FIGARCH+regime+calibrated-jumps (default: on)",
    )
    parser.add_argument(
        "--no-advanced-features",
        action="store_false",
        dest="advanced_features",
        help="Disable all advanced features (plain GARCH+t+Kou baseline)",
    )
    parser.add_argument(
        "--use-xgb",
        action="store_true",
        default=False,
        dest="use_xgb",
        help="Enable XGBoost directional drift shift (default: off). Needs "
             "DATA/macro_daily.csv for the directional signal (plan §8.1).",
    )
    parser.add_argument(
        "--xgb-lambda",
        type=float,
        default=None,
        dest="xgb_tilt_lambda",
        help="XGB tilt strength lambda (default: engine XGB_TILT_LAMBDA=0.0). "
             "Set during calibration sweeps.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            f"Number of worker processes for parallel execution "
            f"(default: {default_workers} on this {cpu_count}-core machine)"
        ),
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Force single-process serial execution (for debugging)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve workers: --serial overrides --workers
    if args.serial:
        workers = None  # serial mode
    elif args.workers is not None:
        workers = args.workers
    else:
        workers = default_workers  # parallel by default via CLI

    # Create engine
    engine = BackrunnerEngine(
        n_sims=args.n_sims,
        seed=args.seed,
        advanced_features=args.advanced_features,
        use_xgb=args.use_xgb,
        xgb_tilt_lambda=args.xgb_tilt_lambda,
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
        workers=workers,
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
