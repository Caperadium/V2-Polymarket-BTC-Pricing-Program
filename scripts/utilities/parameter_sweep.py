#!/usr/bin/env python3
"""
parameter_sweep.py

CLI tool for running systematic parameter sweeps over the backtesting pipeline.

Usage Examples:
    # 1D sweep over min_edge
    python parameter_sweep.py --batch-dir fitted_batch_results --sweep min_edge=0.04,0.06,0.08

    # 2D sweep
    python parameter_sweep.py --batch-dir fitted_batch_results \\
        --sweep min_edge=0.04,0.06 --sweep kelly_fraction=0.10,0.15,0.20

    # Fixed parameters + sweep
    python parameter_sweep.py --batch-dir fitted_batch_results \\
        --fixed bankroll=1000 --sweep min_edge=0.04,0.06

    # Dry-run (preview runs without executing)
    python parameter_sweep.py --sweep min_edge=0.04,0.06 --dry-run

    # Resume interrupted sweep
    python parameter_sweep.py --sweep min_edge=0.04,0.06 --resume

    # Limit number of runs
    python parameter_sweep.py --sweep min_edge=0.04,0.06,0.08 --max-runs 2

    # Fail fast on errors
    python parameter_sweep.py --sweep min_edge=0.04,0.06 --fail-fast
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local imports
from sweep_config import (
    SweepConfig,
    get_parameter_names,
    parse_parameter_value,
    validate_parameter_name,
)
from backtest_engine import BacktestEngine
from backtest_montecarlo_sim import run_shuffle_test, get_summary_stats, run_decile_conditioned_shuffle_test


# Configure logging - suppress INFO to keep output clean
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "parameter_sweeps"


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def parse_sweep_arg(sweep_str: str) -> Tuple[str, List[Any]]:
    """
    Parse a --sweep argument like 'min_edge=0.04,0.06,0.08'.
    
    Returns:
        Tuple of (parameter_name, list_of_values)
    """
    if "=" not in sweep_str:
        raise ValueError(f"Invalid sweep format: '{sweep_str}'. Expected 'param=v1,v2,v3'")
    
    name, values_str = sweep_str.split("=", 1)
    name = name.strip()
    
    if not validate_parameter_name(name):
        valid = ", ".join(sorted(get_parameter_names()))
        raise ValueError(f"Unknown parameter '{name}'. Valid parameters:\n  {valid}")
    
    values = []
    for v in values_str.split(","):
        v = v.strip()
        if v:
            values.append(parse_parameter_value(name, v))
    
    if not values:
        raise ValueError(f"No values provided for parameter '{name}'")
    
    return name, values


def parse_fixed_arg(fixed_str: str) -> Tuple[str, Any]:
    """
    Parse a --fixed argument like 'bankroll=1000'.
    
    Returns:
        Tuple of (parameter_name, value)
    """
    if "=" not in fixed_str:
        raise ValueError(f"Invalid fixed format: '{fixed_str}'. Expected 'param=value'")
    
    name, value_str = fixed_str.split("=", 1)
    name = name.strip()
    value_str = value_str.strip()
    
    if not validate_parameter_name(name):
        valid = ", ".join(sorted(get_parameter_names()))
        raise ValueError(f"Unknown parameter '{name}'. Valid parameters:\n  {valid}")
    
    return name, parse_parameter_value(name, value_str)


def generate_combinations(
    sweep_params: Dict[str, List[Any]],
    fixed_params: Dict[str, Any],
    max_runs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from sweep values.
    
    Args:
        sweep_params: Dict of param_name -> list of values to sweep
        fixed_params: Dict of param_name -> fixed value
        max_runs: Maximum number of combinations to return
        
    Returns:
        List of complete parameter dictionaries
    """
    if not sweep_params:
        # No sweep params, just return fixed params
        return [fixed_params.copy()]
    
    # Generate cartesian product of sweep values
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[k] for k in param_names]
    
    combinations = []
    for combo in itertools.product(*param_values):
        config = fixed_params.copy()
        for name, value in zip(param_names, combo):
            config[name] = value
        combinations.append(config)
        
        if max_runs and len(combinations) >= max_runs:
            break
    
    return combinations


def get_next_run_index(output_dir: Path) -> int:
    """Get the next available run folder index."""
    if not output_dir.exists():
        return 1
    
    existing = []
    for entry in output_dir.iterdir():
        if entry.is_dir():
            match = re.match(r"^(\d+)$", entry.name)
            if match:
                existing.append(int(match.group(1)))
    
    return max(existing, default=0) + 1


def load_batch_data(
    batch_dir: str,
    date_range: Optional[Tuple[str, str]] = None,
) -> List[pd.DataFrame]:
    """
    Load all batch CSV files from a directory.
    
    Supports both:
    - Flat structure: batch_dir/*.csv
    - Nested structure: batch_dir/*/batch_with_fits.csv
    
    Args:
        batch_dir: Path to directory containing batch CSVs
        date_range: Optional tuple of (start_date, end_date) in YYYY-MM-DD format
                   to filter rows by expiry_date or pricing_date
    """
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
    
    # Parse date range if provided
    start_date = None
    end_date = None
    if date_range:
        try:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            logger.info(f"Filtering data to date range: {date_range[0]} to {date_range[1]}")
        except Exception as e:
            raise ValueError(f"Invalid date range format: {e}")
    
    batches = []
    
    # Try nested structure first (timestamped folders)
    nested_files = list(batch_path.glob("*/batch_with_fits.csv"))
    if nested_files:
        for csv_path in sorted(nested_files):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    # Apply date filter if specified
                    if start_date is not None and end_date is not None:
                        df = _filter_by_date_range(df, start_date, end_date)
                    if not df.empty:
                        batches.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {csv_path}: {e}")
    else:
        # Try flat structure
        flat_files = list(batch_path.glob("*.csv"))
        for csv_path in sorted(flat_files):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    # Apply date filter if specified
                    if start_date is not None and end_date is not None:
                        df = _filter_by_date_range(df, start_date, end_date)
                    if not df.empty:
                        batches.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {csv_path}: {e}")
    
    if not batches:
        raise ValueError(f"No valid batch CSV files found in {batch_dir} (after date filtering)")
    
    total_rows = sum(len(b) for b in batches)
    logger.info(f"Loaded {len(batches)} batch files ({total_rows} total rows) from {batch_dir}")
    return batches


def _filter_by_date_range(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Filter DataFrame rows by date range using expiry_date or pricing_date columns.
    
    Priority: uses expiry_date if available, else pricing_date.
    """
    date_col = None
    for col in ["expiry_date", "pricing_date", "date"]:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        logger.warning("No date column found for filtering, keeping all rows")
        return df
    
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        
        # Handle timezone-aware dates by converting to UTC then stripping timezone
        if dates.dt.tz is not None:
            # Use tz_convert(None) to strip timezone (tz_localize only works for naive)
            dates = dates.dt.tz_convert(None)
        
        # Create comparison timestamps (naive)
        start_cmp = pd.Timestamp(start_date)
        end_cmp = pd.Timestamp(end_date)
        
        # Ensure comparison timestamps are also naive
        if start_cmp.tz is not None:
            start_cmp = start_cmp.tz_convert(None)
        if end_cmp.tz is not None:
            end_cmp = end_cmp.tz_convert(None)
        
        # For end date, include the entire day by adding 1 day minus 1 nanosecond
        # This ensures "2025-12-19" includes all of Dec 19th
        end_cmp = end_cmp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        
        mask = (dates >= start_cmp) & (dates <= end_cmp)
        filtered = df[mask].copy()
        
        logger.info(f"Date filter: {len(df)} -> {len(filtered)} rows using {date_col}")
        return filtered
    except Exception as e:
        logger.warning(f"Failed to filter by date: {e}")
        return df




def generate_run_config_md(
    run_index: int,
    config: SweepConfig,
    sweep_params: Dict[str, List[Any]],
    current_sweep_values: Dict[str, Any],
    git_hash: Optional[str],
) -> str:
    """Generate markdown content for run_config.md."""
    lines = [
        f"# Run {run_index:04d}",
        "",
        f"**Timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]
    
    if git_hash:
        lines.append(f"**Git Commit:** `{git_hash}`")
    
    lines.extend(["", "## Sweep Dimensions", ""])
    
    if sweep_params:
        for name, values in sweep_params.items():
            current = current_sweep_values.get(name, "N/A")
            values_str = ", ".join(str(v) for v in values)
            lines.append(f"- **{name}**: `{current}` (from: {values_str})")
    else:
        lines.append("No parameters were swept (fixed run).")
    
    lines.extend(["", "## Full Parameter Set", ""])
    
    for name, value in sorted(asdict(config).items()):
        lines.append(f"- `{name}`: {value}")
    
    return "\n".join(lines)


def run_single_sweep(
    run_index: int,
    config: SweepConfig,
    batches: List[pd.DataFrame],
    sweep_params: Dict[str, List[Any]],
    current_sweep_values: Dict[str, Any],
    output_dir: Path,
    git_hash: Optional[str],
    mc_iterations: int = 500,
    mc_seed: Optional[int] = None,
    use_all_trades: bool = False,
) -> Tuple[int, bool]:
    """
    Execute a single parameter sweep run.
    
    Args:
        mc_seed: Optional seed for Monte Carlo (for reproducibility)
    
    Returns:
        Tuple of (run_index, success_bool)
    """
    run_folder = output_dir / f"{run_index:04d}"
    run_folder.mkdir(parents=True, exist_ok=True)
    
    log_path = run_folder / "logs.txt"
    
    # Log lines for file only (not console)
    log_lines: List[str] = []
    
    def log(msg: str):
        """Log to file only (not console)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        log_lines.append(line)
    
    try:
        log(f"Starting run {run_index:04d}")
        log(f"Config: {current_sweep_values}")
        
        # Reset threshold debug accumulator for this run
        try:
            from auto_reco import reset_threshold_debug
            reset_threshold_debug()
        except Exception:
            pass
        
        # Convert config to strategy params
        strategy_params = config.to_strategy_params()
        log(f"Strategy params: {strategy_params}")
        
        # Run backtest engine
        log("Running BacktestEngine...")
        engine = BacktestEngine(
            market_data_batches=batches,
            initial_bankroll=config.bankroll,
            strategy_params=strategy_params,
        )
        
        # Only fetch all_priced_df when needed (saves runtime)
        if use_all_trades:
            trades_df, equity_df, all_priced_df = engine.run(return_all_priced=True)
            log(f"Backtest complete: {len(trades_df)} trades, {len(all_priced_df)} priced contracts")
        else:
            trades_df, equity_df = engine.run(return_all_priced=False)
            all_priced_df = None
            log(f"Backtest complete: {len(trades_df)} trades")
        
        # Log probability threshold debug info if available
        try:
            from auto_reco import LAST_RECO_THRESHOLD_DEBUG
            if LAST_RECO_THRESHOLD_DEBUG is not None:
                log(f"Prob threshold debug: YES(passed_prob_no_edge={LAST_RECO_THRESHOLD_DEBUG.get('yes_passed_prob_no_edge', 0)}, "
                    f"passed_all={LAST_RECO_THRESHOLD_DEBUG.get('yes_passed_all', 0)}), "
                    f"NO(passed_prob_no_edge={LAST_RECO_THRESHOLD_DEBUG.get('no_passed_prob_no_edge', 0)}, "
                    f"passed_all={LAST_RECO_THRESHOLD_DEBUG.get('no_passed_all', 0)})")
        except Exception:
            pass
        
        # Save taken trades
        trades_path = run_folder / "taken_trades.csv"
        if not trades_df.empty:
            trades_df.to_csv(trades_path, index=False)
            log(f"Saved {len(trades_df)} trades to taken_trades.csv")
        else:
            # Create empty file with header
            pd.DataFrame(columns=["trade_id", "side", "stake", "pnl"]).to_csv(trades_path, index=False)
            log("No trades taken, saved empty trades file")
        
        # Run Monte Carlo simulation
        log("Running Monte Carlo shuffle test...")
        mc_results = {}
        
        # Filter to settled trades only (handle empty df or missing column)
        if trades_df.empty or "settled" not in trades_df.columns:
            settled_trades = pd.DataFrame()
        else:
            settled_trades = trades_df[trades_df["settled"] == True].copy()
        
        if len(settled_trades) >= 5:
            try:
                if use_all_trades and all_priced_df is not None and not all_priced_df.empty:
                    # Use decile-conditioned shuffle test with all priced contracts
                    actual_pnl, shuffled_pnls, percentile, diagnostics = run_decile_conditioned_shuffle_test(
                        settled_trades,
                        all_priced_df,
                        n_iter=mc_iterations,
                        seed=mc_seed,
                        expiry_col="expiry_date",
                        snapshot_col="snapshot_time",
                    )
                    mc_results = get_summary_stats(actual_pnl, shuffled_pnls, percentile)
                    # Add diagnostics to results
                    mc_results["shuffle_mode"] = diagnostics.get("shuffle_mode", "decile_conditioned_all_priced")
                    mc_results["n_iter"] = mc_iterations
                    mc_results["n_all_priced_used"] = diagnostics.get("n_all_priced_used", 0)
                    mc_results["n_deciles_used"] = diagnostics.get("n_deciles_used", 0)
                    mc_results["n_unmatched_trades"] = diagnostics.get("n_unmatched_trades", 0)
                    if "error" in diagnostics:
                        mc_results["diagnostics_error"] = diagnostics["error"]
                    log(f"Decile MC complete: actual_pnl=${actual_pnl:.2f}, percentile={percentile:.2%}, mode={mc_results['shuffle_mode']}")
                else:
                    # Use existing expiry-only shuffle test
                    actual_pnl, shuffled_pnls, percentile = run_shuffle_test(
                        settled_trades,
                        n_iter=mc_iterations,
                        expiry_col="expiry_date",
                        seed=mc_seed,
                    )
                    mc_results = get_summary_stats(actual_pnl, shuffled_pnls, percentile)
                    mc_results["shuffle_mode"] = "expiry_only"
                    mc_results["n_iter"] = mc_iterations
                    log(f"Monte Carlo complete: actual_pnl=${actual_pnl:.2f}, percentile={percentile:.2%}")
            except Exception as e:
                log(f"Monte Carlo failed: {e}")
                mc_results = {"error": str(e)}
        else:
            log(f"Skipping Monte Carlo: only {len(settled_trades)} settled trades (need 5+)")
            mc_results = {"error": f"Insufficient settled trades ({len(settled_trades)})"}
        
        # Save Monte Carlo results
        mc_path = run_folder / "montecarlo_results.csv"
        mc_df = pd.DataFrame([mc_results])
        mc_df.to_csv(mc_path, index=False)
        log("Saved montecarlo_results.csv")
        
        # Save run config markdown
        config_md = generate_run_config_md(
            run_index, config, sweep_params, current_sweep_values, git_hash
        )
        config_path = run_folder / "run_config.md"
        config_path.write_text(config_md, encoding="utf-8")
        log("Saved run_config.md")
        
        # Save equity curve
        equity_path = run_folder / "equity_curve.csv"
        if not equity_df.empty:
            equity_df.to_csv(equity_path, index=False)
            log("Saved equity_curve.csv")
        
        # Save PnL grouped by expiry date
        if not settled_trades.empty and "expiry_date" in settled_trades.columns:
            expiry_col = "expiry_date"
            pnl_col = "pnl" if "pnl" in settled_trades.columns else None
            if pnl_col:
                # Group by expiry and compute summary stats
                expiry_pnl = settled_trades.groupby(expiry_col).agg({
                    pnl_col: ["sum", "count", "mean"],
                }).reset_index()
                # Flatten column names
                expiry_pnl.columns = ["expiry_date", "total_pnl", "trade_count", "avg_pnl"]
                # Sort by expiry date
                expiry_pnl = expiry_pnl.sort_values("expiry_date")
                expiry_pnl.to_csv(run_folder / "pnl_by_expiry.csv", index=False)
                log("Saved pnl_by_expiry.csv")

        
        log(f"Run {run_index:04d} completed successfully")
        
        # Write logs
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        return (run_index, True)
        
    except Exception as e:
        error_msg = f"Run {run_index:04d} failed: {e}\n{traceback.format_exc()}"
        log_lines.append(error_msg)
        logger.error(error_msg)
        
        # Save error file
        error_path = run_folder / "error.txt"
        error_path.write_text(error_msg, encoding="utf-8")
        
        # Save partial logs
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        
        return (run_index, False)


def analyze_dte_buckets(output_dir: Path) -> None:
    """
    Analyze sweep results by DTE (days-to-expiry) buckets.
    
    First identifies top 10 runs (percentile > 95%, z-score > 2, sorted by PnL),
    then computes AVERAGE metrics per DTE bucket across those runs.
    
    Buckets: 1-2 days, 3-4 days, 5-6 days
    """
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return
    
    # Define DTE buckets (0-2 includes same-day and next-day trades)
    DTE_BUCKETS = [
        (0, 2, "0-2 days"),
        (2, 4, "2-4 days"),
        (4, 6, "4-6 days"),
    ]
    
    # Step 1: Find top 10 runs by criteria
    run_stats = []
    for entry in output_dir.iterdir():
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        
        mc_path = entry / "montecarlo_results.csv"
        if not mc_path.exists():
            continue
        
        try:
            mc_df = pd.read_csv(mc_path)
            if mc_df.empty or "z_score" not in mc_df.columns:
                continue
            
            z_score = float(mc_df["z_score"].iloc[0])
            percentile = float(mc_df.get("percentile", [0]).iloc[0])
            actual_pnl = float(mc_df.get("actual_pnl", [0]).iloc[0])
            shuffled_mean = float(mc_df.get("shuffled_mean", [0]).iloc[0])
            
            # Gate: percentile > 95% and z-score > 2
            if percentile > 0.95 and z_score > 2.0:
                run_stats.append({
                    "run_path": entry,
                    "z_score": z_score,
                    "percentile": percentile,
                    "actual_pnl": actual_pnl,
                    "shuffled_mean": shuffled_mean,
                })
        except Exception:
            continue
    
    if not run_stats:
        print("\nNo runs meet criteria (percentile > 95%, z-score > 2)")
        return
    
    # Sort by PnL descending, take top 10
    run_stats.sort(key=lambda x: x["actual_pnl"], reverse=True)
    top_runs = run_stats[:10]
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS BY DTE BUCKETS (Top {len(top_runs)} runs by PnL, gated by percentile>95% & z>2)")
    print(f"{'='*80}\n")
    
    # Step 2: For each DTE bucket, collect metrics from each top run
    for min_dte, max_dte, label in DTE_BUCKETS:
        bucket_metrics = []
        
        for run in top_runs:
            run_path = run["run_path"]
            trades_path = run_path / "taken_trades.csv"
            
            if not trades_path.exists():
                continue
            
            try:
                trades_df = pd.read_csv(trades_path)
                if trades_df.empty:
                    continue
                
                # Calculate DTE
                entry_col = None
                for col in ["pricing_date", "entry_time", "entry_date"]:
                    if col in trades_df.columns:
                        entry_col = col
                        break
                
                if "expiry_date" in trades_df.columns and entry_col:
                    trades_df["expiry_dt"] = pd.to_datetime(trades_df["expiry_date"], errors="coerce")
                    trades_df["entry_dt"] = pd.to_datetime(trades_df[entry_col], errors="coerce")
                    if trades_df["expiry_dt"].dt.tz is not None:
                        trades_df["expiry_dt"] = trades_df["expiry_dt"].dt.tz_convert(None)
                    if trades_df["entry_dt"].dt.tz is not None:
                        trades_df["entry_dt"] = trades_df["entry_dt"].dt.tz_convert(None)
                    trades_df["dte"] = (trades_df["expiry_dt"] - trades_df["entry_dt"]).dt.total_seconds() / 86400
                elif "T_days" in trades_df.columns:
                    trades_df["dte"] = trades_df["T_days"]
                else:
                    continue
                
                # Filter to settled trades
                if "settled" in trades_df.columns:
                    settled_col = trades_df["settled"]
                    if settled_col.dtype == 'object':
                        settled = trades_df[settled_col.astype(str).str.lower().isin(["true", "1"])].copy()
                    else:
                        settled = trades_df[settled_col == True].copy()
                else:
                    settled = trades_df.copy()
                
                if "pnl" not in settled.columns:
                    continue
                
                # Filter to this DTE bucket
                bucket_trades = settled[(settled["dte"] >= min_dte) & (settled["dte"] <= max_dte)]
                
                if bucket_trades.empty:
                    continue
                
                # Compute edge if missing
                if "edge" not in bucket_trades.columns:
                    if "model_prob" in bucket_trades.columns and "entry_price" in bucket_trades.columns and "side" in bucket_trades.columns:
                        bucket_trades = bucket_trades.copy()
                        # Edge = P(win) - entry_price
                        # For YES: P(win) = model_prob
                        # For NO: P(win) = 1 - model_prob
                        bucket_trades["edge"] = bucket_trades.apply(
                            lambda r: r["model_prob"] - r["entry_price"] if str(r["side"]).upper() == "YES"
                                      else (1 - r["model_prob"]) - r["entry_price"],
                            axis=1
                        )
                
                # Compute metrics for this run's bucket
                pnl = bucket_trades["pnl"].sum()
                count = len(bucket_trades)
                pnl_per_trade = pnl / count if count > 0 else 0
                
                valid_edges = bucket_trades["edge"].dropna() if "edge" in bucket_trades.columns else pd.Series([])
                mean_edge = valid_edges.mean() if len(valid_edges) > 0 else np.nan
                median_edge = valid_edges.median() if len(valid_edges) > 0 else np.nan
                p90_edge = np.percentile(valid_edges, 90) if len(valid_edges) > 0 else np.nan
                
                # Worst expiry in this bucket
                worst_expiry_pnl = 0.0
                if "expiry_date" in bucket_trades.columns:
                    for exp, grp in bucket_trades.groupby("expiry_date", observed=True):
                        exp_pnl = grp["pnl"].sum()
                        if exp_pnl < worst_expiry_pnl:
                            worst_expiry_pnl = exp_pnl
                
                bucket_metrics.append({
                    "pnl": pnl,
                    "count": count,
                    "pnl_per_trade": pnl_per_trade,
                    "mean_edge": mean_edge,
                    "median_edge": median_edge,
                    "p90_edge": p90_edge,
                    "worst_expiry": worst_expiry_pnl,
                    "shuffled_mean": run["shuffled_mean"],
                    "z_score": run["z_score"],
                })
                
            except Exception:
                continue
        
        if not bucket_metrics:
            print(f"--- {label}: No data ---\n")
            continue
        
        # Average metrics across top runs
        n = len(bucket_metrics)
        avg_pnl = sum(m["pnl"] for m in bucket_metrics) / n
        avg_count = sum(m["count"] for m in bucket_metrics) / n
        avg_pnl_per_trade = sum(m["pnl_per_trade"] for m in bucket_metrics) / n
        avg_z = sum(m["z_score"] for m in bucket_metrics) / n
        avg_shuffled = sum(m["shuffled_mean"] for m in bucket_metrics) / n
        avg_worst = sum(m["worst_expiry"] for m in bucket_metrics) / n
        
        valid_mean_edges = [m["mean_edge"] for m in bucket_metrics if not np.isnan(m["mean_edge"])]
        valid_median_edges = [m["median_edge"] for m in bucket_metrics if not np.isnan(m["median_edge"])]
        valid_p90_edges = [m["p90_edge"] for m in bucket_metrics if not np.isnan(m["p90_edge"])]
        
        avg_mean_edge = sum(valid_mean_edges) / len(valid_mean_edges) if valid_mean_edges else np.nan
        avg_median_edge = sum(valid_median_edges) / len(valid_median_edges) if valid_median_edges else np.nan
        avg_p90_edge = sum(valid_p90_edges) / len(valid_p90_edges) if valid_p90_edges else np.nan
        
        print(f"--- {label} (avg of {n} runs) ---")
        print(f"  Trade Count:     {avg_count:.1f}")
        print(f"  Settled PnL:     ${avg_pnl:.2f}")
        print(f"  PnL/Trade:       ${avg_pnl_per_trade:.2f}")
        print(f"  Z-Score:         {avg_z:.3f}")
        print(f"  Shuffled Mean:   ${avg_shuffled:.2f}")
        print(f"  Mean Edge:       {avg_mean_edge:.4f}" if not np.isnan(avg_mean_edge) else "  Mean Edge:       N/A")
        print(f"  Median Edge:     {avg_median_edge:.4f}" if not np.isnan(avg_median_edge) else "  Median Edge:     N/A")
        print(f"  90p Edge:        {avg_p90_edge:.4f}" if not np.isnan(avg_p90_edge) else "  90p Edge:        N/A")
        print(f"  Worst Expiry:    ${avg_worst:.2f}")
        print()
    
    print(f"{'='*80}\n")


def analyze_top_runs(output_dir: Path, top_n: int = 3) -> List[int]:
    """
    Analyze completed runs and print summary of top N by Z-score.
    
    Args:
        output_dir: Directory containing numbered run folders
        top_n: Number of top runs to display
        
    Returns:
        List of run indices for the top N runs (to preserve during cleanup)
    """
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return []
    
    # Collect results from all run folders
    results = []
    
    for entry in output_dir.iterdir():
        if not entry.is_dir():
            continue
        if not re.match(r"^\d+$", entry.name):
            continue
        
        run_index = int(entry.name)
        mc_path = entry / "montecarlo_results.csv"
        equity_path = entry / "equity_curve.csv"
        config_path = entry / "run_config.md"
        trades_path = entry / "taken_trades.csv"
        
        if not mc_path.exists():
            continue
        
        try:
            mc_df = pd.read_csv(mc_path)
            if mc_df.empty or "z_score" not in mc_df.columns:
                continue
            
            z_score = float(mc_df["z_score"].iloc[0])
            if np.isnan(z_score):
                continue
            
            # Get Monte Carlo stats
            actual_pnl = float(mc_df.get("actual_pnl", [np.nan]).iloc[0])
            shuffled_mean = float(mc_df.get("shuffled_mean", [np.nan]).iloc[0])
            shuffled_std = float(mc_df.get("shuffled_std", [np.nan]).iloc[0]) if "shuffled_std" in mc_df.columns else np.nan
            percentile = float(mc_df.get("percentile", [np.nan]).iloc[0])
            is_significant = mc_df.get("is_significant", [False]).iloc[0]
            n_iter = int(mc_df.get("n_iter", [500]).iloc[0]) if "n_iter" in mc_df.columns else 500
            p_hat = float(mc_df.get("p_hat", [np.nan]).iloc[0]) if "p_hat" in mc_df.columns else np.nan
            tail_distance = float(mc_df.get("tail_distance", [np.nan]).iloc[0]) if "tail_distance" in mc_df.columns else np.nan
            shuffled_p95 = float(mc_df.get("shuffled_p95", [np.nan]).iloc[0]) if "shuffled_p95" in mc_df.columns else np.nan
            
            # Get trade count and YES/NO breakdown
            num_trades = 0
            num_yes_trades = 0
            num_no_trades = 0
            num_settled = 0
            open_stake = 0.0
            min_trade_edge = np.nan
            max_trade_edge = np.nan
            if trades_path.exists():
                try:
                    trades_df = pd.read_csv(trades_path)
                    num_trades = len(trades_df)
                    if "side" in trades_df.columns:
                        side_counts = trades_df["side"].str.upper().value_counts()
                        num_yes_trades = int(side_counts.get("YES", 0))
                        num_no_trades = int(side_counts.get("NO", 0))
                    # Count settled trades and compute open stake
                    if "settled" in trades_df.columns:
                        settled_mask = trades_df["settled"] == True
                        num_settled = int(settled_mask.sum())
                        if "stake" in trades_df.columns:
                            open_stake = float(trades_df.loc[~settled_mask, "stake"].sum())
                    else:
                        num_settled = num_trades  # Assume all settled if no column
                    # Extract edge range from trades
                    if "edge" in trades_df.columns:
                        edge_vals = pd.to_numeric(trades_df["edge"], errors="coerce").dropna()
                    elif "model_prob" in trades_df.columns and "market_price" in trades_df.columns:
                        # Compute edge: for YES it's p-q, for NO it's q-p
                        p = pd.to_numeric(trades_df["model_prob"], errors="coerce")
                        q = pd.to_numeric(trades_df["market_price"], errors="coerce")
                        side = trades_df["side"].str.upper() if "side" in trades_df.columns else pd.Series(["YES"]*len(trades_df))
                        edge_vals = pd.Series([
                            (p.iloc[i] - q.iloc[i]) if side.iloc[i] == "YES" else (q.iloc[i] - p.iloc[i])
                            for i in range(len(trades_df))
                        ])
                    else:
                        edge_vals = pd.Series(dtype=float)
                    edge_vals = edge_vals.dropna()
                    if len(edge_vals) > 0:
                        min_trade_edge = float(edge_vals.min())
                        max_trade_edge = float(edge_vals.max())
                except Exception:
                    pass

            # Parse probability threshold debug info from logs.txt
            prob_threshold_debug = None
            logs_path = entry / "logs.txt"
            if logs_path.exists():
                try:
                    log_content = logs_path.read_text(encoding="utf-8")
                    # Look for: Prob threshold debug: YES(passed_prob_no_edge=X, passed_all=Y), NO(...)
                    import re as logs_re
                    match = logs_re.search(
                        r"Prob threshold debug: YES\(passed_prob_no_edge=(\d+), passed_all=(\d+)\), "
                        r"NO\(passed_prob_no_edge=(\d+), passed_all=(\d+)\)",
                        log_content
                    )
                    if match:
                        prob_threshold_debug = {
                            "yes_passed_prob_no_edge": int(match.group(1)),
                            "yes_passed_all": int(match.group(2)),
                            "no_passed_prob_no_edge": int(match.group(3)),
                            "no_passed_all": int(match.group(4)),
                        }
                except Exception:
                    pass

                        
            # Get starting and final equity
            starting_equity = np.nan
            final_equity = np.nan
            if equity_path.exists():
                try:
                    eq_df = pd.read_csv(equity_path)
                    if not eq_df.empty and "bankroll" in eq_df.columns:
                        starting_equity = float(eq_df["bankroll"].iloc[0])
                        final_equity = float(eq_df["bankroll"].iloc[-1])
                except Exception:
                    pass
            
            # Extract sweep parameters and bankroll from config
            sweep_values = {}
            config_bankroll = np.nan
            if config_path.exists():
                config_text = config_path.read_text(encoding="utf-8")
                # Parse the "Sweep Dimensions" section
                in_sweep_section = False
                for line in config_text.split("\n"):
                    if "## Sweep Dimensions" in line:
                        in_sweep_section = True
                        continue
                    if in_sweep_section and line.startswith("## "):
                        in_sweep_section = False
                    if in_sweep_section and line.startswith("- **"):
                        # Parse: - **min_edge**: `0.06` (from: 0.04, 0.06, 0.08)
                        match = re.match(r"- \*\*(.+?)\*\*: `(.+?)`", line)
                        if match:
                            sweep_values[match.group(1)] = match.group(2)
                    # Parse bankroll from Full Parameter Set section
                    if "`bankroll`:" in line:
                        match = re.search(r"`bankroll`: ([\d.]+)", line)
                        if match:
                            config_bankroll = float(match.group(1))
            
            # Use config bankroll as starting equity (more accurate than first equity row)
            if not np.isnan(config_bankroll):
                starting_equity = config_bankroll
            
            results.append({
                "run_index": run_index,
                "z_score": z_score,
                "actual_pnl": actual_pnl,
                "shuffled_mean": shuffled_mean,
                "shuffled_std": shuffled_std,
                "percentile": percentile,
                "is_significant": is_significant,
                "n_iter": n_iter,
                "p_hat": p_hat,
                "tail_distance": tail_distance,
                "shuffled_p95": shuffled_p95,
                "starting_equity": starting_equity,
                "final_equity": final_equity,
                "num_trades": num_trades,
                "num_yes_trades": num_yes_trades,
                "num_no_trades": num_no_trades,
                "num_settled": num_settled,
                "open_stake": open_stake,
                "min_trade_edge": min_trade_edge,
                "max_trade_edge": max_trade_edge,
                "sweep_values": sweep_values,
                "prob_threshold_debug": prob_threshold_debug,
            })
            
        except Exception as e:
            logger.debug(f"Failed to parse run {run_index}: {e}")
            continue
    
    if not results:
        print("\nNo runs with valid Z-scores found.")
        return []
    
    # Filter to statistically significant results (Z >= 2.0 AND Percentile >= 95%)
    significant_results = [
        r for r in results 
        if r["z_score"] >= 2.0 and r["percentile"] >= 0.95
    ]
    
    if significant_results:
        # Rank significant results by actual PnL (descending)
        significant_results.sort(key=lambda x: x["actual_pnl"], reverse=True)
        display_results = significant_results[:top_n]
        header = f"TOP {min(top_n, len(significant_results))} SIGNIFICANT RUNS BY SETTLED PnL"
        subheader = f"(Z >= 2.0 AND Percentile >= 95%, {len(significant_results)} total significant)"
    else:
        # No significant results - show all sorted by Z-score as fallback
        results.sort(key=lambda x: x["z_score"], reverse=True)
        display_results = results[:top_n]
        header = f"TOP {min(top_n, len(results))} RUNS BY Z-SCORE (none significant)"
        subheader = "(No runs met Z >= 2.0 AND Percentile >= 95% threshold)"
    
    # Print results
    print(f"\n{'='*70}")
    print(header)
    print(subheader)
    print(f"{'='*70}")
    
    for i, r in enumerate(display_results, 1):
        print(f"\n--- #{i}: Run {r['run_index']:04d} ---")
        print(f"")

        
        # Sweep parameters
        if r["sweep_values"]:
            print("Parameters:")
            for k, v in sorted(r["sweep_values"].items()):
                print(f"  {k}: {v}")
        else:
            print("Parameters: (default config)")
        
        print("")
        print("Shuffle Test Stats (settled trades only):")
        print(f"  Z-Score:       {r['z_score']:.3f}")
        print(f"  Settled PnL:   ${r['actual_pnl']:.2f}")
        print(f"  Shuffled Mean: ${r['shuffled_mean']:.2f}")
        print(f"  Percentile:    {r['percentile']:.1%}")
        print(f"  Significant:   {'Yes' if r['is_significant'] else 'No'}")
        print(f"  MC Iterations: {r.get('n_iter', 500)}")
        # New stats
        if not np.isnan(r.get('p_hat', np.nan)):
            print(f"  p_hat:         {r['p_hat']:.4f}")
        if not np.isnan(r.get('tail_distance', np.nan)):
            print(f"  Tail Distance: {r['tail_distance']:.3f}")
        if not np.isnan(r.get('shuffled_p95', np.nan)):
            print(f"  Shuffled P95:  ${r['shuffled_p95']:.2f}")

        
        print("")
        print("Trades:")
        print(f"  Total:    {r['num_trades']}")
        print(f"  Settled:  {r['num_settled']}")
        if r['num_trades'] > 0:
            yes_pct = (r['num_yes_trades'] / r['num_trades']) * 100
            no_pct = (r['num_no_trades'] / r['num_trades']) * 100
            print(f"  YES:      {r['num_yes_trades']} ({yes_pct:.1f}%)")
            print(f"  NO:       {r['num_no_trades']} ({no_pct:.1f}%)")
        # Show edge range if available
        if not np.isnan(r.get('min_trade_edge', np.nan)):
            print(f"  Edge Range: {r['min_trade_edge']:.3f} - {r['max_trade_edge']:.3f}")
        # Show PnL per settled trade
        if r['num_settled'] > 0 and not np.isnan(r['actual_pnl']):
            pnl_per_trade = r['actual_pnl'] / r['num_settled']
            print(f"  PnL/Trade: ${pnl_per_trade:.2f}")
        
        # Show probability threshold debug info if available
        ptd = r.get('prob_threshold_debug')
        if ptd:
            print("")
            print("Prob Threshold Filter Stats:")
            print(f"  YES: passed_threshold={ptd.get('yes_passed_prob_no_edge', 0) + ptd.get('yes_passed_all', 0)}, "
                  f"passed_edge={ptd.get('yes_passed_all', 0)}, "
                  f"filtered_no_edge={ptd.get('yes_passed_prob_no_edge', 0)}")
            print(f"  NO:  passed_threshold={ptd.get('no_passed_prob_no_edge', 0) + ptd.get('no_passed_all', 0)}, "
                  f"passed_edge={ptd.get('no_passed_all', 0)}, "
                  f"filtered_no_edge={ptd.get('no_passed_prob_no_edge', 0)}")

        print("")
        print("Equity (includes open positions):")
        if not np.isnan(r["starting_equity"]):
            pnl_calc = r["final_equity"] - r["starting_equity"]
            print(f"  Starting:   ${r['starting_equity']:.2f}")
            print(f"  Final:      ${r['final_equity']:.2f}")
            print(f"  Net PnL:    ${pnl_calc:.2f}")
            if r['open_stake'] > 0:
                print(f"  Open Stake: ${r['open_stake']:.2f}")
        elif not np.isnan(r["final_equity"]):
            print(f"  Final: ${r['final_equity']:.2f}")
        else:
            print(f"  N/A")

    
    print(f"\n{'='*70}")
    print(f"Total runs analyzed: {len(results)}")
    print(f"{'='*70}\n")
    
    # Return list of top run indices for preservation
    return [r["run_index"] for r in display_results]


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run parameter sweeps over the backtesting pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="PARAM=V1,V2,...",
        help="Parameter to sweep with comma-separated values. Can be specified multiple times.",
    )
    
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="PARAM=VALUE",
        help="Fixed parameter value. Can be specified multiple times.",
    )
    
    parser.add_argument(
        "--batch-dir",
        default="fitted_batch_results",
        help="Directory containing batch CSV files (default: fitted_batch_results)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing.",
    )
    
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum number of runs to execute.",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from next available run index.",
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error instead of continuing.",
    )
    
    parser.add_argument(
        "--mc-iterations",
        type=int,
        default=500,
        help="Number of Monte Carlo iterations (default: 500)",
    )
    
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory for runs (default: {OUTPUT_DIR})",
    )
    
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List all valid parameter names and exit.",
    )
    
    parser.add_argument(
        "--limited",
        action="store_true",
        help="After sweep, print summary of top 3 runs by Z-score.",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for Monte Carlo (actual seed = base + run_index)",
    )
    
    parser.add_argument(
        "--all_trades",
        action="store_true",
        help="Use decile-conditioned shuffle test with all priced contracts.",
    )
    
    parser.add_argument(
        "--is",
        dest="in_sample_range",
        type=str,
        default=None,
        metavar="START-END",
        help="In-sample date range filter. Format: YYYY-MM-DD-YYYY-MM-DD. "
             "Only includes data within this date range (inclusive).",
    )
    
    parser.add_argument(
        "--dte_bucket",
        action="store_true",
        help="Analyze results by DTE buckets (1-2, 3-4, 5-6 days). "
             "Mutually exclusive with --limited.",
    )

    
    args = parser.parse_args()
    
    # Handle --list-params
    if args.list_params:
        print("Valid sweep parameters:")
        defaults = asdict(SweepConfig())
        for name in sorted(get_parameter_names()):
            print(f"  {name} (default: {defaults[name]})")
        return 0
    
    # Check mutual exclusivity of --limited and --dte_bucket
    if args.limited and args.dte_bucket:
        logger.error("--limited and --dte_bucket are mutually exclusive. Choose one.")
        return 1
    
    # Parse sweep and fixed parameters
    try:
        sweep_params: Dict[str, List[Any]] = {}
        for sweep_str in args.sweep:
            name, values = parse_sweep_arg(sweep_str)
            if name in sweep_params:
                # Merge values
                sweep_params[name].extend(values)
            else:
                sweep_params[name] = values
        
        fixed_params: Dict[str, Any] = {}
        for fixed_str in args.fixed:
            name, value = parse_fixed_arg(fixed_str)
            fixed_params[name] = value
            
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Generate combinations
    combinations = generate_combinations(sweep_params, fixed_params, args.max_runs)
    total_runs = len(combinations)
    
    if total_runs == 0:
        logger.error("No parameter combinations to run. Use --sweep or --fixed.")
        return 1
    
    # Dry run mode
    if args.dry_run:
        print(f"\n=== DRY RUN: {total_runs} planned runs ===\n")
        for i, combo in enumerate(combinations, 1):
            print(f"Run {i:04d}:")
            for name, value in sorted(combo.items()):
                print(f"  {name}: {value}")
            print()
        print(f"Total: {total_runs} runs")
        print(f"Output directory: {args.output_dir}")
        return 0
    
    # Load batch data
    try:
        # Parse date range if provided
        date_range = None
        if args.in_sample_range:
            # Expected format: YYYY-MM-DD-YYYY-MM-DD (with hyphen separating dates)
            parts = args.in_sample_range.split("-")
            if len(parts) >= 6:
                # Split into start and end dates (YYYY-MM-DD - YYYY-MM-DD)
                start_date = f"{parts[0]}-{parts[1]}-{parts[2]}"
                end_date = f"{parts[3]}-{parts[4]}-{parts[5]}"
                date_range = (start_date, end_date)
                print(f"Filtering data to date range: {start_date} to {end_date}")
            else:
                logger.error(f"Invalid date range format: '{args.in_sample_range}'. Expected: YYYY-MM-DD-YYYY-MM-DD")
                return 1
        batches = load_batch_data(args.batch_dir, date_range=date_range)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    
    # Prepare output directories
    # temp/ holds runs during execution, saved/ holds top runs after analysis
    base_output_dir = Path(args.output_dir)
    temp_dir = base_output_dir / "temp"
    saved_dir = base_output_dir / "saved"
    
    # Clear temp directory for fresh run
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    saved_dir.mkdir(parents=True, exist_ok=True)
    
    # Use temp_dir for running sweeps
    output_dir = temp_dir
    
    # Determine starting index (always 1 for temp dir since we clear it)
    start_index = 1
    
    # Get git hash
    git_hash = get_git_commit_hash()
    
    # Execute runs
    print(f"Starting {total_runs} runs with {args.workers} workers (seed: {args.seed})")
    print(f"Output: {output_dir}")
    
    successful = 0
    failed = 0
    completed = 0
    
    # Build list of run tasks
    run_tasks = []
    for i, combo in enumerate(combinations):
        run_index = start_index + i
        base_config = SweepConfig()
        config = base_config.update(combo)
        mc_seed = args.seed + run_index  # Deterministic seed per run
        run_tasks.append((run_index, config, combo, mc_seed))
    
    # Execute runs in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for run_index, config, combo, mc_seed in run_tasks:
            future = executor.submit(
                run_single_sweep,
                run_index=run_index,
                config=config,
                batches=batches,
                sweep_params=sweep_params,
                current_sweep_values=combo,
                output_dir=output_dir,
                git_hash=git_hash,
                mc_iterations=args.mc_iterations,
                mc_seed=mc_seed,
                use_all_trades=args.all_trades,
            )
            futures[future] = run_index
        
        # Collect results as they complete with progress bar
        print(f"Progress: 0/{total_runs}", end="", flush=True)
        for future in as_completed(futures):
            run_idx = futures[future]
            try:
                result_run_index, success = future.result()
                completed += 1
                if success:
                    successful += 1
                else:
                    failed += 1
                    if args.fail_fast:
                        print(f"\rProgress: {completed}/{total_runs} - FAILED run {result_run_index:04d}, stopping")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                # Update progress on same line
                print(f"\rProgress: {completed}/{total_runs}", end="", flush=True)
            except Exception as e:
                completed += 1
                failed += 1
                if args.fail_fast:
                    print(f"\rProgress: {completed}/{total_runs} - exception in run {run_idx:04d}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                print(f"\rProgress: {completed}/{total_runs}", end="", flush=True)
    
    # Final newline after progress bar
    print()
    
    # Summary
    print(f"Complete: {successful}/{total_runs} successful, {failed} failed")
    
    # Print top runs summary if --limited is set, then cleanup
    if args.limited:
        top_run_indices = analyze_top_runs(output_dir, top_n=10)
        
        # Move top runs to saved/ directory
        top_run_folders = {f"{idx:04d}" for idx in top_run_indices}
        moved_count = 0
        for folder_name in top_run_folders:
            src = temp_dir / folder_name
            if src.exists():
                # Generate unique name in saved/ to avoid overwriting
                dst_base = saved_dir / folder_name
                dst = dst_base
                counter = 1
                while dst.exists():
                    dst = saved_dir / f"{folder_name}_{counter:02d}"
                    counter += 1
                shutil.move(str(src), str(dst))
                moved_count += 1
        
        print(f"Saved {moved_count} top runs to {saved_dir}")
    
    # Analyze by DTE buckets if --dte_bucket is set
    elif args.dte_bucket:
        analyze_dte_buckets(output_dir)
    
    # Always delete temp folder contents after run (keep folder structure)
    if temp_dir.exists():
        for item in temp_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print(f"Cleared temp folder contents: {temp_dir}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
