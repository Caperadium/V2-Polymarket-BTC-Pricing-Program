#!/usr/bin/env python3
"""
batch_loader.py

Shared batch-file normalization for backtest engine consumption.
Extracted from app/dashboard.py and app/pages/backtesting.py so both
dashboard pages and the orchestrator can use the same canonical logic.
"""

import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Timestamp inference
# ---------------------------------------------------------------------------

def _infer_pricing_date_from_source(source_name: str) -> str:
    """Infer pricing datetime from folder/filename (full UTC timestamp string)."""
    # Try new format: 2025-12-20_05-57-14_UTC
    new_match = re.search(
        r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_UTC", source_name
    )
    if new_match:
        try:
            date_part = new_match.group(1)
            time_part = new_match.group(2).replace("-", ":")
            return f"{date_part} {time_part}+00:00"
        except ValueError:
            pass

    # Try legacy format: batch_20251113_094053
    legacy_match = re.search(r"batch_(\d{8})_(\d{6})", source_name)
    if legacy_match:
        try:
            date_str = legacy_match.group(1)
            time_str = legacy_match.group(2)
            parsed = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return parsed.strftime("%Y-%m-%d %H:%M:%S+00:00")
        except ValueError:
            pass

    # Fallback: just date patterns
    patterns = [
        r"(20\d{2}-\d{2}-\d{2})",
        r"(20\d{2}\d{2}\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, source_name)
        if match:
            value = match.group(1)
            try:
                parsed = (
                    datetime.strptime(value, "%Y-%m-%d")
                    if "-" in value
                    else datetime.strptime(value, "%Y%m%d")
                )
                return parsed.strftime("%Y-%m-%d %H:%M:%S+00:00")
            except ValueError:
                continue
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")


# ---------------------------------------------------------------------------
# prepare_batch_df — canonical normalization
# ---------------------------------------------------------------------------

def prepare_batch_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize a batch DataFrame for BacktestEngine consumption.

    Handles two source formats:
      - "new"  (from prob_backrunner_engine): has 'model_probability' column
      - "old"  (from fit_probability_curves):    has 'p_model_fit' / 'p_real_mc'

    Ensures these columns exist before returning:
        batch_timestamp, pricing_date, expiry_key, p_model_fit

    Parameters
    ----------
    df : pd.DataFrame
        Raw batch CSV data.
    source_name : str
        Folder name or filename, used to parse a UTC timestamp.

    Returns
    -------
    pd.DataFrame
        Normalized copy with standard column names.
    """
    df = df.copy()

    # --- map new-format column names to the canonical set ---
    if "model_probability" in df.columns and "p_model_fit" not in df.columns:
        df = df.rename(columns={"model_probability": "p_model_fit"})

    if "date" in df.columns and "pricing_date" not in df.columns:
        df = df.rename(columns={"date": "pricing_date"})

    # --- batch_timestamp (source-of-truth for when pricing was performed) ---
    if "batch_timestamp" in df.columns:
        df["pricing_date"] = pd.to_datetime(
            df["batch_timestamp"], errors="coerce", utc=True
        )
    else:
        # Fall back to parsing the folder / file name
        ts_str = _infer_pricing_date_from_source(source_name)
        df["pricing_date"] = pd.to_datetime(ts_str, utc=True)

    # Ensure batch_timestamp column exists (some consumers reference it directly)
    if "batch_timestamp" not in df.columns:
        df["batch_timestamp"] = df["pricing_date"]

    # --- expiry_key ---
    if "expiry_key" not in df.columns:
        if "expiry_date" in df.columns:
            expiry_dt = pd.to_datetime(df["expiry_date"], utc=True, errors="coerce")
            df["expiry_key"] = expiry_dt.dt.strftime("%Y-%m-%d")
        else:
            df["expiry_key"] = "unknown"

    df["source_name"] = source_name
    return df


# ---------------------------------------------------------------------------
# Batch loading
# ---------------------------------------------------------------------------

def load_batches(paths: List[str]) -> List[pd.DataFrame]:
    """Load and normalise a list of batch CSV file paths.

    Returns only successfully loaded DataFrames (warnings are logged, not raised).
    """
    import logging

    logger = logging.getLogger(__name__)
    batches: List[pd.DataFrame] = []

    for path in paths:
        try:
            df = pd.read_csv(path)
            # Use parent folder name for timestamp inference
            folder_name = os.path.basename(os.path.dirname(path))
            df = prepare_batch_df(df, folder_name)
            batches.append(df)
        except Exception:
            logger.warning("Failed to load batch file: %s", path, exc_info=True)

    return batches


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def scan_batch_files(
    root_dir: str,
    start_date: date,
    end_date: date,
    filename: str = "batch_with_fits.csv",
) -> List[str]:
    """Scan *root_dir* for batch folders within the date range.

    Supports folder name formats:
      - ``2025-12-20_05-57-14_UTC`` (current)
      - ``batch_20251113_220535``    (legacy)

    Dates are compared in **local time** (UTC → system tz).

    Parameters
    ----------
    root_dir : str
        Relative or absolute path to the root directory to scan.
    start_date, end_date : date
        Inclusive date range (local time).
    filename : str
        Name of the CSV file expected inside each batch folder.

    Returns
    -------
    List[str]
        Sorted absolute file paths.
    """
    valid_paths: List[str] = []

    root_path = Path(root_dir)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path
    if not root_path.exists():
        return []

    for entry in os.scandir(str(root_path)):
        if not entry.is_dir():
            continue

        folder_date: Optional[date] = None

        # Try new format: 2025-12-20_05-57-14_UTC
        new_match = re.search(
            r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_UTC", entry.name
        )
        if new_match:
            try:
                date_part = new_match.group(1)
                time_part = new_match.group(2).replace("-", ":")
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
                    dt_utc = datetime.strptime(
                        legacy_match.group(1) + "_" + legacy_match.group(2),
                        "%Y%m%d_%H%M%S",
                    ).replace(tzinfo=timezone.utc)
                    dt_local = dt_utc.astimezone(None)
                    folder_date = dt_local.date()
                except ValueError:
                    pass

        # Fallback: ISO date or YYYYMMDD
        if folder_date is None:
            iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", entry.name)
            if iso_match:
                try:
                    folder_date = datetime.strptime(
                        iso_match.group(1), "%Y-%m-%d"
                    ).date()
                except ValueError:
                    pass

        if folder_date is None:
            simple_match = re.search(r"(\d{8})", entry.name)
            if simple_match:
                try:
                    folder_date = datetime.strptime(
                        simple_match.group(1), "%Y%m%d"
                    ).date()
                except ValueError:
                    pass

        if folder_date is not None and start_date <= folder_date <= end_date:
            target_file = os.path.join(entry.path, filename)
            if os.path.exists(target_file):
                valid_paths.append(target_file)

    return sorted(valid_paths)


def scan_flat_batch_files(
    root_dir: str,
    start_date: date,
    end_date: date,
    glob_pattern: str = "batch_*.csv",
) -> List[str]:
    """Scan *root_dir* for flat (non-folder) batch CSV files within the date range.

    Useful for unfitted backtest output directories.

    Parameters
    ----------
    root_dir : str
        Relative or absolute path.
    start_date, end_date : date
        Inclusive date range (local time).
    glob_pattern : str
        Filename pattern. Only ``batch_YYYYMMDD_HHMMSS.csv`` is parsed.

    Returns
    -------
    List[str]
        Sorted absolute file paths.
    """
    valid_paths: List[str] = []

    root_path = Path(root_dir)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path
    if not root_path.exists():
        return []

    for entry in os.scandir(str(root_path)):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue

        match = re.search(r"batch_(\d{8})_(\d{6})\.csv$", entry.name)
        if not match:
            continue

        try:
            dt_utc = datetime.strptime(match.group(1), "%Y%m%d").replace(
                tzinfo=timezone.utc
            )
            dt_local = dt_utc.astimezone(None)
            file_date = dt_local.date()
        except ValueError:
            continue

        if start_date <= file_date <= end_date:
            valid_paths.append(entry.path)

    return sorted(valid_paths)
