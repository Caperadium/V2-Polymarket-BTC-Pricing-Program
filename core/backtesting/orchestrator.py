#!/usr/bin/env python3
"""
orchestrator.py

Top-level BacktestingOrchestrator: chains the full pipeline
  fetch → backrun → curve_fit → backtest → diagnostics

Single entry point for both CLI and dashboard use.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from core.backtesting.contract_store import ContractPriceStore
from core.backtesting.batch_loader import load_batches, scan_batch_files
from core.backtesting.backrunner import BackrunnerEngine
from core.backtesting.backtest_engine import BacktestEngine, run_backtest
from core.backtesting.diagnostics import SignalDiagnostics

logger = logging.getLogger(__name__)


def default_worker_count() -> int:
    """Sane default worker count: leave 4 cores free, cap at 12.

    Mirrors the CLI heuristic in ``backrunner.main()`` so the dashboard
    path runs parallel by default too.
    """
    import multiprocessing

    return min(max(multiprocessing.cpu_count() - 4, 1), 12)


# Default directories — resolved relative to this file so CWD doesn't matter (Streamlit, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = _PROJECT_ROOT / "backtested_probabilities"
UNFITTED_DIR = OUTPUT_ROOT / "unfitted"
FITTED_DIR = OUTPUT_ROOT / "fitted"


class BacktestingOrchestrator:
    """End-to-end backtesting pipeline orchestrator."""

    def __init__(
        self,
        n_sims: int = 15000,
        seed: int = 42,
        advanced_features: bool = True,
        strategy_params: Optional[Dict] = None,
        initial_bankroll: float = 1000.0,
        btc_price_path: str = "DATA/btc_intraday_1m.csv",
        unfitted_dir: Optional[Path] = None,
        fitted_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.n_sims = n_sims
        self.seed = seed
        self.advanced_features = advanced_features
        self.strategy_params = strategy_params or {}
        self.initial_bankroll = initial_bankroll
        self.btc_price_path = btc_price_path
        self.unfitted_dir = unfitted_dir or UNFITTED_DIR
        self.fitted_dir = fitted_dir or FITTED_DIR
        self._progress = progress_callback

        # Lazily created components
        self._store: Optional[ContractPriceStore] = None
        self._backrunner: Optional[BackrunnerEngine] = None

    # ------------------------------------------------------------------
    # Step 1: Data fetching
    # ------------------------------------------------------------------

    def fetch_historical_prices(self, force: bool = False) -> Tuple[int, List[str]]:
        """Fetch incremental contract prices from Polymarket APIs.

        Returns (new_records_count, error_messages_list).
        """
        from core.backtesting.polymarket_fetcher import fetch_incremental_prices
        import requests

        if self._store is None:
            self._store = ContractPriceStore()

        session = requests.Session()
        try:
            added, errors = fetch_incremental_prices(
                store=self._store,
                session=session,
                progress_callback=self._progress,
            )
        finally:
            session.close()

        logger.info("Fetch complete: %d new records added", added)
        return added, errors

    # ------------------------------------------------------------------
    # Step 2: Backrunner (MC pricing)
    # ------------------------------------------------------------------

    def run_pricing_backrun(
        self,
        limit: Optional[int] = None,
        market_csv: Optional[str] = None,
        workers: Optional[int] = None,
    ) -> Path:
        """Run time-travel MC pricing backrunner.

        Uses the ContractPriceStore data by default, or a legacy market CSV
        if *market_csv* is provided.

        Parameters
        ----------
        workers : int or None
            Number of worker processes. None → ``default_worker_count()``
            (parallel). Pass 1 to force serial. This is what makes the
            dashboard path run parallel — previously it always ran serial.

        Returns path to ``unfitted_dir`` containing batch CSV files.
        """
        if workers is None:
            workers = default_worker_count()

        # ---- spawn-safety guard (route parallel through CLI subprocess) ----
        # An in-process ProcessPoolExecutor uses spawn on Windows; each worker
        # re-imports the parent __main__. When the orchestrator is driven from
        # Streamlit (__main__ = page script) or `python -c` (__main__ = the -c
        # string), that re-import re-runs unguarded top-level code — a warning
        # flood under Streamlit, a fork-bomb under `-c`. The backrunner CLI has
        # a proper __main__ guard, so run parallel work there instead. In-process
        # pooling is reserved for the CLI entrypoint (backrunner.main) only.
        if workers > 1:
            if self.unfitted_dir == UNFITTED_DIR and self.fitted_dir == FITTED_DIR:
                return self._run_backrun_subprocess(
                    limit=limit, market_csv=market_csv, workers=workers
                )
            logger.warning(
                "Custom backrun dirs — CLI subprocess can't target them; "
                "falling back to serial to avoid spawn re-importing __main__."
            )
            workers = 1

        if self._backrunner is None:
            self._backrunner = BackrunnerEngine(
                n_sims=self.n_sims,
                seed=self.seed,
                advanced_features=self.advanced_features,
                unfitted_dir=self.unfitted_dir,
                fitted_dir=self.fitted_dir,
                progress_callback=self._progress,
            )

        # Load BTC data
        daily_df, intraday_df, hourly_df = self._backrunner.load_btc_data()

        # Load market data
        if market_csv:
            market_df = self._backrunner.load_market_df(market_csv)
        else:
            if self._store is None:
                self._store = ContractPriceStore()
            market_df = self._store.to_market_df()
            if market_df.empty:
                logger.warning(
                    "Contract store is empty. Run fetch_historical_prices() first, "
                    "or pass market_csv for legacy market data."
                )
                self.unfitted_dir.mkdir(parents=True, exist_ok=True)
                return self.unfitted_dir

        logger.info("Backrunner workers=%s", workers)
        unfitted_path = self._backrunner.run(
            market_df=market_df,
            daily_df=daily_df,
            intraday_df=intraday_df,
            hourly_df=hourly_df,
            limit=limit,
            workers=workers,
        )
        return unfitted_path

    def _run_backrun_subprocess(
        self,
        limit: Optional[int],
        market_csv: Optional[str],
        workers: int,
    ) -> Path:
        """Run the parallel backrun via the backrunner CLI in a subprocess.

        Safe under Streamlit on Windows: the CLI has a ``__main__`` guard, so
        its spawned workers import ``core.backtesting.backrunner`` (clean), not
        the Streamlit page. Progress lines on the child's stderr drive the UI
        callback. Writes to the default ``backtested_probabilities`` dirs.
        """
        import re
        import subprocess
        import sys

        backrunner_py = _PROJECT_ROOT / "core" / "backtesting" / "backrunner.py"
        cmd = [
            sys.executable,
            str(backrunner_py),
            "--workers", str(workers),
            "--n-sims", str(self.n_sims),
            "--seed", str(self.seed),
            "--skip-fitting",  # orchestrator runs curve fitting itself afterwards
        ]
        if not self.advanced_features:
            cmd.append("--no-advanced-features")
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if market_csv:
            cmd += ["--market-prices", market_csv]

        logger.info("Launching backrun subprocess: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        progress_re = re.compile(r"Progress:\s+(\d+)/(\d+)")
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            logger.info("[backrun] %s", line)
            m = progress_re.search(line)
            if m and self._progress:
                done, total = int(m.group(1)), int(m.group(2))
                self._progress("pricing", done, total)

        proc.wait()
        if proc.returncode != 0:
            logger.error("Backrun subprocess exited with code %d", proc.returncode)
        else:
            logger.info("Backrun subprocess complete.")
        return self.unfitted_dir

    # ------------------------------------------------------------------
    # Step 3: Curve fitting
    # ------------------------------------------------------------------

    def run_curve_fitting(self) -> int:
        """Run curve fitting on unfitted batch CSVs.

        Returns count of successfully fitted files.
        """
        if self._backrunner is None:
            self._backrunner = BackrunnerEngine(
                n_sims=self.n_sims,
                seed=self.seed,
                advanced_features=self.advanced_features,
                unfitted_dir=self.unfitted_dir,
                fitted_dir=self.fitted_dir,
                progress_callback=self._progress,
            )
        return self._backrunner.run_curve_fitting()

    # ------------------------------------------------------------------
    # Step 4: Load fitted batches
    # ------------------------------------------------------------------

    def load_fitted_batches(self) -> List[pd.DataFrame]:
        """Scan fitted_dir and load all batch_with_fits.csv files.

        Returns list of normalized DataFrames ready for BacktestEngine.
        """
        from datetime import date

        # Wide date range to include everything
        paths = scan_batch_files(
            str(self.fitted_dir),
            date(2020, 1, 1),
            date(2030, 12, 31),
            filename="batch_with_fits.csv",
        )
        if not paths:
            logger.warning("No fitted batch files found in %s", self.fitted_dir)
            return []

        batches = load_batches(paths)
        logger.info("Loaded %d fitted batches from %s", len(batches), self.fitted_dir)
        return batches

    # ------------------------------------------------------------------
    # Step 5: Backtest (auto-reco replay)
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        fitted_batches: List[pd.DataFrame],
        price_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run BacktestEngine over fitted batches.

        Returns (trades_df, equity_df, all_priced_df).
        """
        return run_backtest(
            daily_batches=fitted_batches,
            initial_bankroll=self.initial_bankroll,
            strategy_params=self.strategy_params,
            btc_price_path=self.btc_price_path,
            price_df=price_df,
            return_all_priced=True,
        )

    # ------------------------------------------------------------------
    # Step 6: Diagnostics
    # ------------------------------------------------------------------

    def run_diagnostics(self, all_priced_df: pd.DataFrame) -> dict:
        """Run signal diagnostics on the all-priced contracts DataFrame.

        Returns structured diagnostics dict (see SignalDiagnostics.run_full_report).
        """
        if self._progress:
            self._progress("diagnostics", 0, 1)
        diag = SignalDiagnostics(all_priced_df)
        report = diag.run_full_report()
        if self._progress:
            self._progress("diagnostics", 1, 1)
        return report

    # ------------------------------------------------------------------
    # All-in-one
    # ------------------------------------------------------------------

    def run_full(
        self,
        fetch: bool = True,
        backrun: bool = True,
        fit_curves: bool = True,
        limit: Optional[int] = None,
        market_csv: Optional[str] = None,
        price_df: Optional[pd.DataFrame] = None,
        workers: Optional[int] = None,
    ) -> dict:
        """Run the full backtesting pipeline.

        Parameters
        ----------
        fetch : bool
            Whether to fetch incremental historical prices from Polymarket.
        backrun : bool
            Whether to run the MC pricing backrunner.
        fit_curves : bool
            Whether to run logistic curve fitting.
        limit : int or None
            Cap on backrunner timestamps.
        market_csv : str or None
            Legacy market data CSV path (bypasses contract store).
        price_df : pd.DataFrame or None
            Pre-loaded BTC price DataFrame for the backtest engine.
        workers : int or None
            Worker processes for the backrunner. None → parallel default.

        Returns
        -------
        dict
            Keys: new_records, unfitted_dir, fitted_dir, trades_df, equity_df,
                  all_priced_df, diagnostics
        """
        results: dict = {"new_records": 0}

        # ---- Fetch ----
        if fetch:
            results["new_records"], fetch_errors = self.fetch_historical_prices()
            if fetch_errors:
                results["fetch_errors"] = fetch_errors

        # ---- Backrun ----
        if backrun:
            results["unfitted_dir"] = self.run_pricing_backrun(
                limit=limit, market_csv=market_csv, workers=workers
            )
        else:
            results["unfitted_dir"] = self.unfitted_dir

        # ---- Curve fit ----
        if fit_curves:
            results["fitted_dir"] = self.fitted_dir
            self.run_curve_fitting()
        else:
            results["fitted_dir"] = self.fitted_dir

        # ---- Load fitted ----
        fitted_batches = self.load_fitted_batches()

        if not fitted_batches:
            logger.warning("No fitted batches to backtest — returning empty results")
            results["trades_df"] = pd.DataFrame()
            results["equity_df"] = pd.DataFrame()
            results["all_priced_df"] = pd.DataFrame()
            results["diagnostics"] = SignalDiagnostics(pd.DataFrame()).run_full_report()
            return results

        # ---- Backtest ----
        if self._progress:
            self._progress("backtesting", 0, 1)
        trades_df, equity_df, all_priced_df = self.run_backtest(
            fitted_batches, price_df=price_df
        )
        if self._progress:
            self._progress("backtesting", 1, 1)

        results["trades_df"] = trades_df
        results["equity_df"] = equity_df
        results["all_priced_df"] = all_priced_df

        # ---- Diagnostics ----
        results["diagnostics"] = self.run_diagnostics(all_priced_df)

        return results
