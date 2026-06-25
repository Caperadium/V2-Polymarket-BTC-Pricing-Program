"""
core.backtesting — Unified backtesting module.

Chains: fetch → backrun → curve_fit → backtest → diagnostics.

Public API
----------
ContractPriceStore      — CSV store for historical Polymarket contract prices.
fetch_incremental_prices — Polymarket API data fetcher (Gamma + CLOB).
prepare_batch_df         — Normalize batch DataFrames for backtest consumption.
load_batches             — Load & normalize batch CSV files.
scan_batch_files         — Scan folder-based batch directories.
scan_flat_batch_files    — Scan flat batch CSV files.
BackrunnerEngine         — Time-travel MC pricing engine.
BacktestEngine           — Auto-reco replay backtest engine.
run_backtest             — Convenience wrapper around BacktestEngine.
SignalDiagnostics        — Spearman/AUC/DTE/moneyness signal diagnostics.
run_diagnostics          — CLI-preserved diagnostic runner.
BacktestingOrchestrator  — Full pipeline orchestrator.
"""

from core.backtesting.contract_store import ContractPriceStore
from core.backtesting.polymarket_fetcher import fetch_incremental_prices
from core.backtesting.batch_loader import (
    load_batches,
    prepare_batch_df,
    scan_batch_files,
    scan_flat_batch_files,
)
from core.backtesting.backrunner import BackrunnerEngine
from core.backtesting.backtest_engine import BacktestEngine, run_backtest
from core.backtesting.diagnostics import SignalDiagnostics, run_diagnostics
from core.backtesting.orchestrator import BacktestingOrchestrator
from core.backtesting.in_sample_oos import (
    WindowMode,
    WindowSpec,
    OOSHygieneError,
    OOSLeakError,
    compute_default_cutoff,
    partition_contracts,
    apply_window,
    apply_window_trades,
    m2_training_set,
    guarded_filter,
    small_sample_state,
    train_pipeline,
    load_artifacts,
    load_or_train,
    apply_oos_calibration,
    is_m2_inert,
    verify_oos_leak_free,
)

__all__ = [
    "ContractPriceStore",
    "fetch_incremental_prices",
    "prepare_batch_df",
    "load_batches",
    "scan_batch_files",
    "scan_flat_batch_files",
    "BackrunnerEngine",
    "BacktestEngine",
    "run_backtest",
    "SignalDiagnostics",
    "run_diagnostics",
    "BacktestingOrchestrator",
    # In-sample / out-of-sample window
    "WindowMode",
    "WindowSpec",
    "OOSHygieneError",
    "OOSLeakError",
    "compute_default_cutoff",
    "partition_contracts",
    "apply_window",
    "apply_window_trades",
    "m2_training_set",
    "guarded_filter",
    "small_sample_state",
    "train_pipeline",
    "load_artifacts",
    "load_or_train",
    "apply_oos_calibration",
    "is_m2_inert",
    "verify_oos_leak_free",
]
