#!/usr/bin/env python3
"""
aggregate_old_batch_data.py

Aggregates historical market data from old_batch_runs/ subdirectories
into a single CSV file for backtesting.

Usage:
    python aggregate_old_batch_data.py
    python aggregate_old_batch_data.py --input-dir old_batch_runs --output old_market_prices.csv
"""

import argparse
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    import pytz
    ET_TZ = pytz.timezone('America/New_York')
except ImportError:
    ET_TZ = None  # Fallback handled below

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_expiry_to_utc(expiry_value) -> Optional[str]:
    """
    Convert expiry_date from 12pm ET (noon Eastern Time) to UTC.
    
    Polymarket BTC contracts expire at 12:00 PM ET (noon).
    This function takes a date string like '2025-11-15' and returns
    the UTC datetime string like '2025-11-15 17:00:00+00:00' (during EST)
    or '2025-11-15 16:00:00+00:00' (during EDT).
    
    Args:
        expiry_value: Date string (e.g., '2025-11-15') or datetime object
        
    Returns:
        UTC datetime string, or None if parsing fails
    """
    if pd.isna(expiry_value) or expiry_value is None:
        return None
    
    try:
        # Parse the date
        if isinstance(expiry_value, str):
            # Handle various date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y']:
                try:
                    date_obj = datetime.strptime(expiry_value.strip(), fmt)
                    break
                except ValueError:
                    continue
            else:
                return str(expiry_value)  # Return as-is if can't parse
        elif isinstance(expiry_value, datetime):
            date_obj = expiry_value
        else:
            # Try pandas datetime conversion
            date_obj = pd.to_datetime(expiry_value)
            if pd.isna(date_obj):
                return None
            date_obj = date_obj.to_pydatetime()
        
        # Create datetime at 12pm (noon)
        noon_naive = datetime(date_obj.year, date_obj.month, date_obj.day, 12, 0, 0)
        
        if ET_TZ is not None:
            # Use pytz for proper timezone handling (handles EST/EDT automatically)
            noon_et = ET_TZ.localize(noon_naive)
            noon_utc = noon_et.astimezone(pytz.UTC)
        else:
            # Fallback: assume EST (UTC-5) - not DST aware
            noon_utc = noon_naive + timedelta(hours=5)
            noon_utc = noon_utc.replace(tzinfo=timezone.utc)
        
        return noon_utc.strftime('%Y-%m-%d %H:%M:%S+00:00')
        
    except Exception as e:
        logger.debug(f"Could not convert expiry_date '{expiry_value}': {e}")
        return str(expiry_value) if expiry_value else None

# Required columns to extract
REQUIRED_COLUMNS = ['slug', 'strike', 'market_price', 't_days', 'expiry_date']


def parse_timestamp_from_folder(folder_name: str) -> Optional[datetime]:
    """
    Parse timestamp from folder name like 'batch_20231201' or 'batch_1700000000'.
    
    Supports:
    - batch_YYYYMMDD (date only)
    - batch_YYYYMMDD_HHMMSS (date + time)
    - batch_<unix_timestamp> (Unix epoch seconds)
    - YYYY-MM-DD_HH-MM-SS_UTC (new format)
    
    Returns:
        datetime object in UTC, or None if parsing fails.
    """
    # Strip 'batch_' prefix if present
    if folder_name.startswith('batch_'):
        timestamp_str = folder_name[6:]
    else:
        timestamp_str = folder_name
    
    # Try new format first: YYYY-MM-DD_HH-MM-SS_UTC
    new_format_match = re.match(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_UTC', timestamp_str)
    if new_format_match:
        try:
            date_part = new_format_match.group(1)
            time_part = new_format_match.group(2).replace('-', ':')
            dt_str = f"{date_part}T{time_part}+00:00"
            return datetime.fromisoformat(dt_str)
        except ValueError:
            pass
    
    # Try YYYYMMDD_HHMMSS format
    if re.match(r'^\d{8}_\d{6}$', timestamp_str):
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    
    # Try YYYYMMDD format (date only)
    if re.match(r'^\d{8}$', timestamp_str):
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d').replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    
    # Try Unix timestamp (10+ digit number)
    if re.match(r'^\d{10,}$', timestamp_str):
        try:
            unix_ts = int(timestamp_str)
            return datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        except (ValueError, OSError):
            pass
    
    # Try to extract any date pattern from the string
    date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', timestamp_str)
    if date_match:
        try:
            year, month, day = map(int, date_match.groups())
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            pass
    
    logger.warning(f"Could not parse timestamp from folder: {folder_name}")
    return None


def load_batch_summary(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load a batch_summary.csv file and extract required columns.
    
    Returns:
        DataFrame with required columns, or None if loading fails.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"Failed to read {csv_path}: {e}")
        return None
    
    # Check for required columns (case-insensitive matching)
    col_map = {}
    df_cols_lower = {c.lower(): c for c in df.columns}
    
    for req_col in REQUIRED_COLUMNS:
        if req_col in df.columns:
            col_map[req_col] = req_col
        elif req_col.lower() in df_cols_lower:
            col_map[req_col] = df_cols_lower[req_col.lower()]
        else:
            # Try common alternatives
            alternatives = {
                'slug': ['market_slug', 'contract_slug'],
                'strike': ['strike_price', 'Strike'],
                'market_price': ['price', 'Polymarket_Price', 'poly_price'],
                't_days': ['T_days', 'T_Days', 'days_to_expiry', 'tte'],
                'expiry_date': ['Expiry_Date', 'expiry', 'expiration'],
            }
            found = False
            for alt in alternatives.get(req_col, []):
                if alt in df.columns:
                    col_map[req_col] = alt
                    found = True
                    break
                elif alt.lower() in df_cols_lower:
                    col_map[req_col] = df_cols_lower[alt.lower()]
                    found = True
                    break
            
            if not found:
                # t_days and expiry_date are optional - continue without them
                if req_col in ['t_days', 'expiry_date']:
                    logger.debug(f"Optional column '{req_col}' not found in {csv_path}")
                    col_map[req_col] = None
                else:
                    logger.warning(f"Missing required column '{req_col}' in {csv_path}")
                    return None
    
    # Extract and rename columns (only include columns that were found)
    found_cols = [c for c in REQUIRED_COLUMNS if col_map.get(c) is not None]
    result = df[[col_map[c] for c in found_cols]].copy()
    result.columns = found_cols
    
    # Add missing optional columns as NaN
    for c in REQUIRED_COLUMNS:
        if c not in result.columns:
            result[c] = None
    
    return result


def scan_batch_folders(root_dir: Path) -> List[Tuple[Path, datetime]]:
    """
    Scan root directory for batch folders and return (path, timestamp) pairs.
    
    Returns:
        List of (folder_path, parsed_timestamp) tuples, sorted by timestamp.
    """
    results = []
    
    if not root_dir.exists():
        logger.error(f"Root directory does not exist: {root_dir}")
        return results
    
    for entry in root_dir.iterdir():
        if not entry.is_dir():
            continue
        
        # Must have batch_ prefix or match timestamp pattern
        if not (entry.name.startswith('batch_') or 
                re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC', entry.name)):
            continue
        
        timestamp = parse_timestamp_from_folder(entry.name)
        if timestamp is None:
            continue
        
        # Check for batch_summary.csv
        csv_path = entry / 'batch_summary.csv'
        if not csv_path.exists():
            logger.debug(f"No batch_summary.csv in {entry.name}")
            continue
        
        results.append((entry, timestamp))
    
    # Sort by timestamp
    results.sort(key=lambda x: x[1])
    logger.info(f"Found {len(results)} valid batch folders")
    
    return results


def aggregate_batch_data(root_dir: Path) -> pd.DataFrame:
    """
    Aggregate all batch_summary.csv files into a single DataFrame.
    
    Returns:
        Combined DataFrame with date column added, sorted by date.
    """
    batch_folders = scan_batch_folders(root_dir)
    
    if not batch_folders:
        logger.warning("No valid batch folders found")
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ['date'])
    
    frames = []
    
    for folder_path, timestamp in batch_folders:
        csv_path = folder_path / 'batch_summary.csv'
        
        df = load_batch_summary(csv_path)
        if df is None or df.empty:
            continue
        
        # Add timestamp column
        df['date'] = timestamp
        
        frames.append(df)
        logger.debug(f"Loaded {len(df)} rows from {folder_path.name}")
    
    if not frames:
        logger.warning("No data extracted from batch files")
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ['date'])
    
    # Combine all frames
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined {len(combined)} total rows from {len(frames)} files")
    
    # Clean data
    initial_count = len(combined)
    
    # Drop rows with missing required values
    combined = combined.dropna(subset=REQUIRED_COLUMNS)
    
    # Convert types
    combined['strike'] = pd.to_numeric(combined['strike'], errors='coerce')
    combined['market_price'] = pd.to_numeric(combined['market_price'], errors='coerce')
    if 't_days' in combined.columns:
        combined['t_days'] = pd.to_numeric(combined['t_days'], errors='coerce')
    
    # Drop rows that failed numeric conversion on required fields
    combined = combined.dropna(subset=['strike', 'market_price'])
    
    # Remove completely empty rows
    combined = combined.dropna(how='all')
    
    cleaned_count = len(combined)
    if initial_count != cleaned_count:
        logger.info(f"Dropped {initial_count - cleaned_count} malformed rows")
    
    # Sort by date ascending
    combined = combined.sort_values('date', ascending=True).reset_index(drop=True)
    
    # Convert expiry_date from 12pm ET (noon Eastern) to UTC
    if 'expiry_date' in combined.columns:
        combined['expiry_date'] = combined['expiry_date'].apply(convert_expiry_to_utc)
        logger.info("Converted expiry_date from 12pm ET to UTC")
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate historical batch data for backtesting"
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='old_batch_runs',
        help='Root directory containing batch subfolders (default: old_batch_runs)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str, 
        default='old_market_prices.csv',
        help='Output CSV filename (default: old_market_prices.csv)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    root_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    logger.info(f"Scanning {root_dir} for batch folders...")
    
    # Aggregate data
    result_df = aggregate_batch_data(root_dir)
    
    if result_df.empty:
        logger.error("No data to save. Check that old_batch_runs/ contains valid batch folders.")
        return
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(result_df)} rows to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total rows:    {len(result_df)}")
    print(f"Unique slugs:  {result_df['slug'].nunique()}")
    print(f"Date range:    {result_df['date'].min()} to {result_df['date'].max()}")
    print(f"Output file:   {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
