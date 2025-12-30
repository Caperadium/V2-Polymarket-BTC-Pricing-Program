import argparse
import csv
import json
import logging
import requests
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
from typing import Dict, List, Tuple

from core.pricing.btc_pricing_engine import load_and_prep_data, fit_garch_model, simulate_paths, get_contract_probability

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com/events"

def parse_strike_price(title: str) -> float:
    """
    Parses strike price from market title.
    Example: "Bitcoin > $90k on Dec 17" -> 90000.0
    Example: "Bitcoin > $95,500 on..." -> 95500.0
    """
    # Regex to find $... followed by number
    # Handle "k" suffix
    match = re.search(r'\$(\d+(?:,\d+)?(?:\.\d+)?)(k)?', title, re.IGNORECASE)
    if match:
        val_str = match.group(1).replace(',', '')
        val = float(val_str)
        if match.group(2) and match.group(2).lower() == 'k':
            val *= 1000
        return val
    return None

def fetch_events(slug_pattern: str, day: int) -> List[Dict]:
    """
    Fetches events for a specific day slug.
    We construct the slug and query the API.
    """
    # Assumption provided in prompt examples like "bitcoin-above-on-december-{}"
    # We might need to handle the date suffix carefully. 
    # If placeholder is just {}, we inject the day.
    slug = slug_pattern.format(day)
    
    # Query API
    # Usually we can filter by exact slug or search
    params = {'slug': slug}
    try:
        resp = requests.get(GAMMA_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.error(f"Failed to fetch event for slug {slug}: {e}")
        return []

def get_expiry_utc(date_str: str, year: int) -> datetime:
    """
    Constructs expiry timestamp: Date + 12:00:00 PM ET -> UTC.
    date_str format example: "Dec 17" or derived from loop.
    We assume the input is just the Month/Day tuple or we construct from the iteration.
    To be precise, we rely on the runner's 'day' and 'month' context.
    
    Actually, let's use the current year logic requested (default current year).
    Prompt says: "Date + 12:00:00 in US/Eastern Time."
    """
    # Parse date part. "Dec 17"
    # For now, let's assume we are iterating days in the CURRENT month/year context based on the slug pattern hint.
    # But wait, "Dec 17" implies December. If we are in Dec, year is 2024?
    # The prompt says: "--day-range: Two integers (start, end)".
    # It assumes the slug pattern contains the Month explicitly (e.g. "...december-{}").
    # We need to extract the month from the slug pattern? Or just use the year/month of now?
    # "bitcoin-above-on-december-{}" -> Month is Dec.
    
    # Heuristic: Extract Month from slug_pattern if possible, else default to current month.
    # Simple regex for month names.
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    # Fallback to current month if not found
    now = datetime.now()
    month = now.month
    
    # Try to find month in date_str first (e.g. from API title "Dec 17")
    match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d+)', date_str, re.IGNORECASE)
    if match:
        month_str = match.group(1).lower()
        # Map abbreviated months
        abbr_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        if month_str in abbr_map:
            month = abbr_map[month_str]
        elif month_str in months: # unlikely to hit here due to regex
            month = months[month_str]
            
    day = int(match.group(2)) if match else 1
    
    # Construct naive datetime
    dt_naive = datetime(year, month, day, 12, 0, 0)
    
    # Localize to ET
    et_tz = pytz.timezone('US/Eastern')
    dt_et = et_tz.localize(dt_naive)
    
    # Convert to UTC
    dt_utc = dt_et.astimezone(pytz.utc)
    return dt_utc

def main():
    parser = argparse.ArgumentParser(description="Batch Pricing Runner for BTC Contracts")
    parser.add_argument("--slug-pattern", required=True, help="Slug pattern with placeholder, e.g., 'bitcoin-above-on-december-{}'")
    parser.add_argument("--day-range", nargs=2, type=int, required=True, help="Start and End day (inclusive)")
    parser.add_argument("--num-sims", type=int, default=10000, help="Number of Monte Carlo paths to simulate (default: 10000)")
    parser.add_argument("--min-volume", type=float, default=0.0, help="Minimum volume to process a market (default: 0)")

    args = parser.parse_args()
    
    slug_pattern = args.slug_pattern
    start_day, end_day = args.day_range
    num_sims = args.num_sims
    current_year = datetime.now().year # Default to current year as per removal of flag

    # 1. Load Data & Fit Model Once
    logger.info("Initializing Pricing Engine...")
    daily_csv = "DATA/btc_daily.csv"
    intraday_csv = "DATA/btc_intraday_1m.csv"
    
    try:
        daily_returns, S0 = load_and_prep_data(daily_csv, intraday_csv)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    logger.info(f"Data Loaded. S0: {S0}. Fitting GARCH...")
    garch_params = fit_garch_model(daily_returns)
    logger.info(f"Model Fitted: {garch_params}")

    results = []

    # 2. Iterate Days
    # We want to optimize: Fit once (Done). Simulate per Date.
    
    # Storage for contracts by date to batch simulation
    # Key: DateString, Value: List of contracts
    contracts_by_date = {} 

    logger.info(f"Fetching markets for days {start_day} to {end_day}...")
    
    for day in range(start_day, end_day + 1):
        events = fetch_events(slug_pattern, day)
        if not events:
            logger.warning(f"No events found for day {day} (slug: {slug_pattern.format(day)})")
            continue
            
        for event in events:
            # Polymarket API structure: Event contains Markets
            markets = event.get('markets', [])
            for market in markets:
                # We care about "Yes" outcome
                # Usually checks binary markets
                
                # Parse infos
                title = market.get('question', '') or event.get('title', '')
                
                strike = parse_strike_price(title)
                if strike is None:
                    continue
                
                # Volume Filter
                # Polymarket API uses 'volume' (string or float) or 'volumeNum'
                vol_str = market.get('volume', '0')
                try:
                    volume = float(vol_str)
                except (ValueError, TypeError):
                    volume = 0.0
                    
                if volume < args.min_volume:
                    # logger.debug(f"Skipping market {strike} due to low volume: {volume}")
                    continue
                
                # Extract Outcome Price (Yes)
                # market['outcomePrices'] is a JSON string often ['0.12', '0.88'] for [Yes, No] or [No, Yes]??
                # Polymarket usually: outcomes=["Yes", "No"] -> prices match order
                outcomes = json.loads(market.get('outcomes', '[]'))
                prices = json.loads(market.get('outcomePrices', '[]'))
                
                poly_price = None
                if 'Yes' in outcomes:
                    idx = outcomes.index('Yes')
                    if idx < len(prices):
                        poly_price = float(prices[idx])
                
                if poly_price is None:
                    continue
                    
                # Expiry Date Logic
                # Title often contains date "Dec 17".
                expiry_dt_utc = get_expiry_utc(title, current_year)
                
                date_key = expiry_dt_utc.date()
                if date_key not in contracts_by_date:
                    contracts_by_date[date_key] = {
                        'expiry_utc': expiry_dt_utc,
                        'contracts': []
                    }
                
                contracts_by_date[date_key]['contracts'].append({
                    'strike': strike,
                    'poly_price': poly_price,
                    'title': title
                })

    # 3. Simulate & Price
    logger.info(f"Processing {len(contracts_by_date)} unique expiry dates...")
    
    now_utc = datetime.now(timezone.utc)
    
    result_rows = []
    
    for date_key, data in contracts_by_date.items():
        expiry_utc = data['expiry_utc']
        contracts = data['contracts']
        
        # Calculate time to expiry
        delta = expiry_utc - now_utc
        days_to_expiry = delta.total_seconds() / (24 * 3600)
        
        if days_to_expiry <= 0:
            logger.warning(f"Expired contracts for {date_key}, skipping.")
            continue
            
        logger.info(f"Simulating for {date_key} (T={days_to_expiry:.4f} days)...")
        
        # Simulate Paths
        paths = simulate_paths(S0, garch_params, jump_params=None, days_to_expiry=days_to_expiry, n_sims=num_sims)
        
        # Grade each contract
        for c in contracts:
            strike = c['strike']
            poly_price = c['poly_price']
            model_prob = get_contract_probability(paths, strike)
            
            edge = model_prob - poly_price
            
            # Format expiry ET string
            # Convert back to ET for display
            et_tz = pytz.timezone('US/Eastern')
            expiry_et = expiry_utc.astimezone(et_tz)
            expiry_et_str = expiry_et.strftime("%b %d %H:%M ET")
            
            # Generate slug from title (lowercase, replace spaces with hyphens)
            slug = re.sub(r'[^a-z0-9\\-]', '', title.lower().replace(' ', '-').replace('$', '').replace(',', ''))
            
            result_rows.append({
                # Match prob_backrunner_engine.py output format for compatibility
                'slug': slug,
                'strike': strike,
                'market_price': poly_price,
                'p_real_mc': model_prob,  # Use same column name as backrunner
                'T_days': days_to_expiry,  # Float days to expiry
                'date': now_utc,  # Pricing date (when we ran the pricing)
                'expiry_date': expiry_utc,  # UTC timestamp of expiry
            })

            
    # 4. Save CSV
    # 4. Save CSV
    # 4. Save Results & Plot
    if result_rows:
        # Create timestamped directory
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
        run_dir = f"batch_results/{timestamp_str}"
        import os
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            
        # Save CSV
        output_path = f"{run_dir}/batch_results.csv"
        df_res = pd.DataFrame(result_rows)
        df_res.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        # Run Curve Fitting
        try:
            from core.pricing import fit_probability_curves
            logger.info("Fitting probability curves...")
            
            # Setup output paths for curve fitting
            fitted_output_dir = f"fitted_batch_results/{timestamp_str}"
            import os
            if not os.path.exists(fitted_output_dir):
                os.makedirs(fitted_output_dir)
            
            fit_probability_curves.process_batch(
                input_csv=output_path,
                output_batch_csv=f"{fitted_output_dir}/batch_with_fits.csv",
                output_curve_params_csv=f"{fitted_output_dir}/curve_params.csv",
            )
            logger.info(f"Saved fitted results to {fitted_output_dir}")
        except Exception as e:
            logger.error(f"Failed to fit curves: {e}")
            
    else:
        logger.warning("No results to save.")

if __name__ == "__main__":
    main()
