import argparse
import csv
import json
import logging
import sys
from pathlib import Path
import requests
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
from typing import Any, Dict, List, Tuple

# Guard: ensure repo root is on sys.path when invoked as a script (e.g.
# `python scripts/pipelines/batch_pricing_runner.py --help`), mirroring
# core/backtesting/backrunner.py. Without this, module-level `core.*`
# imports below raise ModuleNotFoundError because sys.path[0] is this
# script's own directory, not the repo root. Pre-existing gap (not
# introduced by T5) surfaced by the `--help` exit-0 acceptance check.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.pricing.btc_pricing_engine import (
    load_and_prep_data, calculate_probabilities, dte_bucket_horizon,
    load_calibrated_jumps, build_regime_jump_params,
)
from core.pricing.regime_detector import RegimeDetector
from core.pricing.engine_config import build_engine_kwargs

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
    parser.add_argument("--recalibrate-jumps", action="store_true",
                        help="Calibrate jump parameters from BTC data instead of using hardcoded defaults")
    parser.add_argument("--advanced-features", action="store_true", default=True,
                        dest="advanced_features",
                        help="Enable SVCJ+skewed-t+FIGARCH+calibrated-jumps (default: on)")
    parser.add_argument("--no-advanced-features", action="store_false",
                        dest="advanced_features",
                        help="Disable all advanced features (plain GARCH+t+Kou baseline)")
    parser.add_argument("--use-xgb", action="store_true", default=False,
                        dest="use_xgb",
                        help="Enable XGBoost directional drift shift (default: off). "
                             "Needs DATA/macro_daily.csv for the directional signal.")
    parser.add_argument("--xgb-lambda", type=float, default=None,
                        dest="xgb_tilt_lambda",
                        help="XGB tilt strength lambda (default: engine XGB_TILT_LAMBDA=0.0).")

    args = parser.parse_args()
    
    slug_pattern = args.slug_pattern
    start_day, end_day = args.day_range
    num_sims = args.num_sims
    current_year = datetime.now().year # Default to current year as per removal of flag

    # 1. Load Data Once
    logger.info("Initializing Pricing Engine...")
    hourly_csv = "DATA/btc_hourly.csv"
    intraday_csv = "DATA/btc_intraday_1m.csv"

    # T5 (H2): load hourly + intraday ONCE and route pricing through
    # calculate_probabilities (per-expiry-group), mirroring the backrunner
    # pattern, instead of a direct fit_garch_model + simulate_paths loop with
    # no regime switching / horizon gating.
    try:
        hourly_df_live = pd.read_csv(hourly_csv)
        intraday_df_live = pd.read_csv(intraday_csv)
        _, S0 = load_and_prep_data(
            hourly_csv, intraday_csv,
            hourly_df=hourly_df_live, intraday_df=intraday_df_live,
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    logger.info(f"Data Loaded. S0: {S0}.")

    # FIX 2 (M1): calibrate jump parameters by default when advanced features are on,
    # so the LIVE jump source matches the BACKTEST (calibrated everywhere via bipower).
    # `--recalibrate-jumps` forces a fresh fit (bypassing the 30-day cache).
    # NOTE: load_calibrated_jumps returns 'lam'/'p_crash' keys; simulate_paths expects
    # 'lambda'/'crash_prob'. Map them or the calibrated lambda/crash silently fall back
    # to module defaults (a latent bug in the previous --recalibrate-jumps path).
    calibrated_jumps = None
    # Raw dict from load_calibrated_jumps -- keyed 'lam'/'p_crash'/'rho_J'/
    # 'fit_converged'. build_regime_jump_params needs THIS dict (via its
    # `calibrated=` kwarg), never the remapped `calibrated_jumps` above.
    cal = None
    if args.advanced_features or args.recalibrate_jumps:
        cal = load_calibrated_jumps(
            hourly_csv=hourly_csv, force_recalibrate=args.recalibrate_jumps,
        )
        if cal.get("fit_converged"):
            calibrated_jumps = {
                "lambda": cal["lam"], "crash_prob": cal["p_crash"],
                "eta_up": cal["eta_up"], "eta_down": cal["eta_down"],
                "mu_v": cal["mu_v"], "rho_J": cal["rho_J"],
                # FIX 4 (M1): SVCJ return-vol regression slope actually used in
                # simulate_paths (rho_J above is reporting-only). Key only exists
                # in `cal` when load_calibrated_jumps has been run after the T3
                # write-path fix; .get() falls back to inert 0.0.
                "rho_j_slope": cal.get("rho_j_slope", 0.0),
            }
            logger.info("Using data-calibrated jump parameters (live)")
        else:
            cal = None  # not converged -- do not build regime params from it

    # T5 (H2): regime layer construction, mirroring the backrunner. `cal` is
    # only defined (and non-None) when fit_converged was truthy above.
    detector = None
    regime_params = None
    if args.advanced_features:
        detector = RegimeDetector()
        if calibrated_jumps is not None and cal is not None:
            regime_params = build_regime_jump_params(calibrated=cal)

    # Per-snapshot dedup: fit GARCH/FIGARCH + derive S0 once, reuse across
    # every expiry group processed in this run (byte-identical output).
    garch_cache: Dict[bool, Any] = {}

    # FIX 3 (re-enabled): XGBoost directional drift setup (live). Per-DTE-bucket
    # models trained once on the full live data (leak is not a concern live —
    # "now" is the present), cached across dates. Off by default.
    xgb_daily_ret = None
    xgb_macro = None
    xgb_model_cache = {}
    if args.use_xgb:
        from core.pricing.directional_xgb import DirectionalXGB, to_daily_log_return_series
        import os as _os
        xgb_daily_ret = to_daily_log_return_series(pd.read_csv(hourly_csv))
        macro_path = "DATA/macro_daily.csv"
        if _os.path.exists(macro_path):
            xgb_macro = pd.read_csv(macro_path, index_col=0)
            xgb_macro.index = pd.to_datetime(xgb_macro.index, utc=True)
            xgb_macro = xgb_macro.sort_index()
        else:
            logger.warning(
                "XGB enabled but DATA/macro_daily.csv missing; running BTC-only "
                "(directional signal expected weak — run core/data/macro_fetcher.py)."
            )

        def _get_xgb_model(bucket_h):
            if bucket_h in xgb_model_cache:
                return xgb_model_cache[bucket_h]
            model = None
            try:
                m = DirectionalXGB()
                if m.train_from_slice(xgb_daily_ret, xgb_macro, int(round(bucket_h))):
                    model = m
            except Exception:
                logger.warning("XGB train failed (bucket %sd)", bucket_h, exc_info=True)
            xgb_model_cache[bucket_h] = model
            return model

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
                    'title': title,
                    # Capture CLOB token IDs for order book lookup
                    'condition_id': market.get('conditionId'),
                    'clob_token_ids': market.get('clobTokenIds', '[]'),
                    'outcomes': outcomes,
                })

    # 3. Simulate & Price
    logger.info(f"Processing {len(contracts_by_date)} unique expiry dates...")
    
    now_utc = datetime.now(timezone.utc)
    
    result_rows = []
    
    for date_key, data in contracts_by_date.items():
        expiry_utc = data['expiry_utc']
        contracts = data['contracts']
        
        # Calculate time to expiry (hours for hourly GARCH simulation)
        delta = expiry_utc - now_utc
        hours_to_expiry = delta.total_seconds() / 3600

        if hours_to_expiry <= 0:
            logger.warning(f"Expired contracts for {date_key}, skipping.")
            continue

        logger.info(f"Simulating for {date_key} (T={hours_to_expiry:.1f}h)...")

        # T5 (H2): pick the per-DTE-bucket XGB model for this expiry group
        # (calculate_probabilities applies the drift shift INTERNALLY -- the
        # old post-simulation apply_xgb_drift_shift block is gone; leaving it
        # in place would double-apply the tilt).
        xgb_model = None
        if args.use_xgb and xgb_daily_ret is not None:
            bucket_h = dte_bucket_horizon(hours_to_expiry / 24.0)
            if bucket_h is not None:
                xgb_model = _get_xgb_model(bucket_h)

        # T5 (H2): per-expiry-group call to calculate_probabilities (regime
        # switching + horizon gating + calibrated jumps), same engine
        # configuration as the backtest path.
        engine_kwargs = build_engine_kwargs(
            advanced_features=args.advanced_features,
            detector=detector,
            regime_params=regime_params,
            jump_params=calibrated_jumps,
            n_sims=num_sims,
            use_xgb=args.use_xgb,
            xgb_model=xgb_model,
            xgb_tilt_lambda=args.xgb_tilt_lambda,
            macro_df=xgb_macro,
        )
        probs = calculate_probabilities(
            strikes=[c['strike'] for c in contracts],
            hours_to_expiry=hours_to_expiry,
            hourly_df=hourly_df_live,
            intraday_df=intraday_df_live,
            use_naive_prior=True,
            garch_cache=garch_cache,
            s0_override=S0,
            **engine_kwargs,
        )

        # Grade each contract
        for c in contracts:
            strike = c['strike']
            poly_price = c['poly_price']
            model_prob = probs.get(strike, float('nan'))

            edge = model_prob - poly_price
            
            # Format expiry ET string
            # Convert back to ET for display
            et_tz = pytz.timezone('US/Eastern')
            expiry_et = expiry_utc.astimezone(et_tz)
            expiry_et_str = expiry_et.strftime("%b %d %H:%M ET")
            
            # Generate slug from title (lowercase, replace spaces with hyphens)
            slug = re.sub(r'[^a-z0-9\\-]', '', c['title'].lower().replace(' ', '-').replace('$', '').replace(',', ''))
            
            result_rows.append({
                # Match prob_backrunner_engine.py output format for compatibility
                'slug': slug,
                'strike': strike,
                'market_price': poly_price,
                'p_real_mc': model_prob,  # Use same column name as backrunner
                'T_days': hours_to_expiry / 24.0,  # Float days to expiry (backward compat)
                'date': now_utc,  # Pricing date (when we ran the pricing)
                'expiry_date': expiry_utc,  # UTC timestamp of expiry
                # CLOB order book fields (for live price fetching)
                'condition_id': c.get('condition_id'),
                'clob_token_ids': c.get('clob_token_ids', '[]'),
                'outcomes': json.dumps(c.get('outcomes', [])),
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
