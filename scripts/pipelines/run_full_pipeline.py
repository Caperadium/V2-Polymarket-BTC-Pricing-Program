#!/usr/bin/env python
"""
run_full_pipeline.py

Runs the complete BTC contract pricing pipeline:
1. data_fetcher.py - Fetches latest BTC price data
2. batch_pricing_runner.py - Prices all contracts for specified dates
3. fit_probability_curves.py - Fits logistic curves and generates plots

Usage:
    python run_full_pipeline.py --slug-pattern "bitcoin-above-on-december-{}" --day-range 17 23
    python run_full_pipeline.py --slug-pattern "bitcoin-above-on-december-{}" --day-range 17 23 --num-sims 50000
"""

import argparse
import subprocess
import sys
import os
import logging
from pathlib import Path
from datetime import date, datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _enrich_with_order_book(df: "pd.DataFrame", log_fn=None) -> "pd.DataFrame":
    """
    Enrich batch DataFrame with live order book prices from CLOB API.
    
    Fetches best ask/bid prices for YES and NO outcomes based on 
    clob_token_ids stored in the batch data.
    """
    import json
    import requests
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    CLOB_API_BASE = "https://clob.polymarket.com"
    
    def log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            logger.info(msg)
    
    def fetch_book(token_id: str, timeout: float = 5.0) -> Optional[Dict]:
        """Fetch order book from CLOB API."""
        try:
            response = requests.get(
                f"{CLOB_API_BASE}/book",
                params={"token_id": token_id},
                timeout=timeout,
            )
            response.raise_for_status()
            book = response.json()
            
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            
            # Sort to get best prices
            # Best bid = highest price, Best ask = lowest price
            best_bid = max((float(b["price"]) for b in bids), default=None)
            best_ask = min((float(a["price"]) for a in asks), default=None)
            
            return {"best_bid": best_bid, "best_ask": best_ask}
        except Exception as e:
            logger.debug(f"Order book fetch failed for {token_id[:16]}...: {e}")
            return None
    
    # Initialize columns
    df["yes_ask_price"] = None
    df["yes_bid_price"] = None
    df["no_ask_price"] = None
    df["no_bid_price"] = None
    
    # Build list of unique token IDs to fetch
    token_to_rows = {}  # token_id -> list of (row_idx, outcome_type)
    
    for idx, row in df.iterrows():
        clob_ids_raw = row.get("clob_token_ids", "[]")
        outcomes_raw = row.get("outcomes", "[]")
        
        try:
            clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        except:
            continue
        
        if not clob_ids or not outcomes:
            continue
        
        # Map outcomes to token IDs
        for i, outcome in enumerate(outcomes):
            if i < len(clob_ids):
                token_id = clob_ids[i]
                outcome_upper = str(outcome).upper()
                if token_id not in token_to_rows:
                    token_to_rows[token_id] = []
                token_to_rows[token_id].append((idx, outcome_upper))
    
    log(f"  Fetching order books for {len(token_to_rows)} unique tokens...")
    
    # Fetch in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_book, tid): tid for tid in token_to_rows.keys()}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                results[tid] = future.result()
            except Exception:
                results[tid] = None
    
    # Apply results to DataFrame
    success_count = 0
    for token_id, book_data in results.items():
        if book_data is None:
            continue
        
        for row_idx, outcome_type in token_to_rows[token_id]:
            if outcome_type == "YES":
                df.at[row_idx, "yes_ask_price"] = book_data.get("best_ask")
                df.at[row_idx, "yes_bid_price"] = book_data.get("best_bid")
            elif outcome_type == "NO":
                df.at[row_idx, "no_ask_price"] = book_data.get("best_ask")
                df.at[row_idx, "no_bid_price"] = book_data.get("best_bid")
            success_count += 1
    
    log(f"  Updated {success_count} rows with order book prices")
    return df


def run_pipeline_programmatic(
    expiry_dates: List[date],
    num_sims: int = 10000,
    skip_data_fetch: bool = False,
    min_volume: float = 0.0,
) -> Dict[str, Any]:
    """
    Run the pricing pipeline programmatically (no CLI/argparse).
    
    Args:
        expiry_dates: List of expiry dates to process
        num_sims: Number of Monte Carlo simulations
        skip_data_fetch: If True, skip fetching BTC data
        min_volume: Minimum volume filter
        
    Returns:
        {
            "ok": bool - Overall success
            "processed": List[date] - Successfully processed dates
            "failed": List[Dict] - Failed dates with error details
            "output_dir": str - Output directory path (if successful)
            "logs": List[str] - Log messages
        }
    """
    from polymarket.date_utils import group_dates_by_month
    
    result = {
        "ok": False,
        "processed": [],
        "failed": [],
        "output_dir": None,
        "logs": [],
    }
    
    def log(msg: str):
        result["logs"].append(msg)
        logger.info(msg)
    
    if not expiry_dates:
        result["logs"].append("No expiry dates provided")
        return result
    
    # Step 1: Fetch data (optional)
    if not skip_data_fetch:
        log("Step 1: Fetching BTC data...")
        try:
            from core.data import data_fetcher
            # data_fetcher.main() fetches and saves data
            # We need to call it without sys.exit
            data_fetcher.main()
            log("✅ Data fetch complete")
        except Exception as e:
            log(f"⚠️ Data fetch failed: {e} (continuing with existing data)")
    else:
        log("Step 1: Skipping data fetch")
    
    # Step 2: Group dates by month for slug patterns
    date_groups = group_dates_by_month(expiry_dates)
    log(f"Processing {len(expiry_dates)} dates across {len(date_groups)} month(s)")
    
    # Step 3: Run batch pricing for each month group
    all_result_rows = []
    
    try:
        # Import batch pricing internals
        from core.pricing.btc_pricing_engine import load_and_prep_data, fit_garch_model, simulate_paths, get_contract_probability
        import scripts.pipelines.batch_pricing_runner as batch_pricing_runner
        
        # Load data once
        log("Loading BTC data and fitting GARCH model...")
        daily_csv = "DATA/btc_daily.csv"
        intraday_csv = "DATA/btc_intraday_1m.csv"
        
        daily_returns, S0 = load_and_prep_data(daily_csv, intraday_csv)
        garch_params = fit_garch_model(daily_returns)
        log(f"✅ Model fitted. S0=${S0:.2f}")
        
        now_utc = datetime.now(timezone.utc)
        
        for group in date_groups:
            slug_pattern = group["slug_pattern"]
            start_day = group["start_date"].day
            end_day = group["end_date"].day
            
            log(f"Processing {group['month'].title()} {start_day}-{end_day}...")
            
            try:
                # Fetch and process contracts
                contracts_by_date = {}
                
                for day in range(start_day, end_day + 1):
                    events = batch_pricing_runner.fetch_events(slug_pattern, day)
                    if not events:
                        log(f"  No events for day {day}")
                        continue
                    
                    for event in events:
                        markets = event.get('markets', [])
                        for market in markets:
                            title = market.get('question', '') or event.get('title', '')
                            strike = batch_pricing_runner.parse_strike_price(title)
                            if strike is None:
                                continue
                            
                            # Volume filter
                            vol_str = market.get('volume', '0')
                            try:
                                volume = float(vol_str)
                            except (ValueError, TypeError):
                                volume = 0.0
                            if volume < min_volume:
                                continue
                            
                            # Extract price
                            import json
                            outcomes = json.loads(market.get('outcomes', '[]'))
                            prices = json.loads(market.get('outcomePrices', '[]'))
                            
                            poly_price = None
                            if 'Yes' in outcomes:
                                idx = outcomes.index('Yes')
                                if idx < len(prices):
                                    poly_price = float(prices[idx])
                            if poly_price is None:
                                continue
                            
                            # Expiry
                            expiry_dt_utc = batch_pricing_runner.get_expiry_utc(title, group["year"])
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
                                # CLOB order book fields
                                'condition_id': market.get('conditionId'),
                                'clob_token_ids': market.get('clobTokenIds', '[]'),
                                'outcomes': outcomes,
                            })
                
                # Simulate and price
                for date_key, data in contracts_by_date.items():
                    expiry_utc = data['expiry_utc']
                    contracts = data['contracts']
                    
                    delta = expiry_utc - now_utc
                    days_to_expiry = delta.total_seconds() / (24 * 3600)
                    
                    if days_to_expiry <= 0:
                        result["failed"].append({"date": date_key, "error": "Already expired"})
                        continue
                    
                    paths = simulate_paths(S0, garch_params, jump_params=None, 
                                          days_to_expiry=days_to_expiry, n_sims=num_sims)
                    
                    import pytz
                    import re
                    for c in contracts:
                        model_prob = get_contract_probability(paths, c['strike'])
                        
                        et_tz = pytz.timezone('US/Eastern')
                        expiry_et = expiry_utc.astimezone(et_tz)
                        
                        slug = re.sub(r'[^a-z0-9\\-]', '', c['title'].lower().replace(' ', '-').replace('$', '').replace(',', ''))
                        
                        all_result_rows.append({
                            'slug': slug,
                            'strike': c['strike'],
                            'market_price': c['poly_price'],
                            'p_real_mc': model_prob,
                            'T_days': days_to_expiry,
                            'date': now_utc,
                            'expiry_date': expiry_utc,
                            # CLOB order book fields
                            'condition_id': c.get('condition_id'),
                            'clob_token_ids': c.get('clob_token_ids', '[]'),
                            'outcomes': json.dumps(c.get('outcomes', [])),
                        })
                    
                    result["processed"].append(date_key)
                    
            except Exception as e:
                for d in [group["start_date"] + timedelta(days=i) 
                         for i in range((group["end_date"] - group["start_date"]).days + 1)]:
                    if d not in result["processed"]:
                        result["failed"].append({"date": d, "error": str(e)})
                log(f"  ❌ Error: {e}")
        
        # Step 4: Save results
        if all_result_rows:
            import pandas as pd
            
            timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
            run_dir = f"batch_results/{timestamp_str}"
            fitted_output_dir = f"fitted_batch_results/{timestamp_str}"
            
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(fitted_output_dir, exist_ok=True)
            
            # Save raw results
            output_path = f"{run_dir}/batch_results.csv"
            df_res = pd.DataFrame(all_result_rows)
            df_res.to_csv(output_path, index=False)
            log(f"Saved raw results to {output_path}")
            
            # Fit curves
            try:
                from core.pricing import fit_probability_curves
                fit_probability_curves.process_batch(
                    input_csv=output_path,
                    output_batch_csv=f"{fitted_output_dir}/batch_with_fits.csv",
                    output_curve_params_csv=f"{fitted_output_dir}/curve_params.csv",
                )
                log(f"✅ Fitted results saved to {fitted_output_dir}")
            except Exception as e:
                log(f"⚠️ Curve fitting failed: {e}")
            
            # Step 5: Enrich with live order book prices
            try:
                batch_csv_path = f"{fitted_output_dir}/batch_with_fits.csv"
                if Path(batch_csv_path).exists():
                    log("Fetching live order book prices...")
                    enriched_df = _enrich_with_order_book(pd.read_csv(batch_csv_path), log)
                    enriched_df.to_csv(batch_csv_path, index=False)
                    log(f"✅ Enriched batch with order book prices")
            except Exception as e:
                log(f"⚠️ Order book enrichment failed: {e}")
            
            result["output_dir"] = fitted_output_dir
            result["ok"] = True
        else:
            log("No results to save")
            result["ok"] = len(result["processed"]) > 0 or len(expiry_dates) == 0
            
    except Exception as e:
        log(f"❌ Pipeline error: {e}")
        import traceback
        log(traceback.format_exc())
        
    return result


def verify_pipeline_output(expiry_dates: List[date], output_dir: str) -> Dict[str, Any]:
    """
    Verify pipeline produced outputs for all dates.
    
    Args:
        expiry_dates: Expected dates
        output_dir: Output directory from pipeline
        
    Returns:
        {"verified": [date], "missing": [date], "ok": bool}
    """
    import pandas as pd
    
    result = {"verified": [], "missing": [], "ok": False}
    
    if not output_dir or not os.path.exists(output_dir):
        result["missing"] = expiry_dates
        return result
    
    # Load fitted results
    fitted_csv = os.path.join(output_dir, "batch_with_fits.csv")
    if not os.path.exists(fitted_csv):
        result["missing"] = expiry_dates
        return result
    
    try:
        df = pd.read_csv(fitted_csv, parse_dates=['expiry_date'])
        processed_dates = set(df['expiry_date'].dt.date.unique())
        
        for d in expiry_dates:
            if d in processed_dates:
                result["verified"].append(d)
            else:
                result["missing"].append(d)
        
        result["ok"] = len(result["missing"]) == 0
    except Exception:
        result["missing"] = expiry_dates
    
    return result


# Need timedelta for the function
from datetime import timedelta


def run_step(name: str, cmd: list) -> bool:
    """Run a subprocess step and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"\n❌ {name} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {name} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the full BTC contract pricing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_full_pipeline.py --slug-pattern "bitcoin-above-on-december-{}" --day-range 17 23
    python run_full_pipeline.py --slug-pattern "bitcoin-above-on-january-{}" --day-range 1 15 --num-sims 50000
        """
    )
    
    # batch_pricing_runner arguments
    parser.add_argument("--slug-pattern", required=True, 
                        help="Slug pattern with placeholder, e.g., 'bitcoin-above-on-december-{}'")
    parser.add_argument("--day-range", nargs=2, type=int, required=True,
                        help="Start and End day (inclusive)")
    parser.add_argument("--num-sims", type=int, default=10000,
                        help="Number of Monte Carlo paths to simulate (default: 10000)")
    parser.add_argument("--min-volume", type=float, default=0.0,
                        help="Minimum volume to process a market (default: 0)")
    
    # fit_probability_curves arguments
    parser.add_argument("--use-rn-prob", action="store_true",
                        help="If set, fit neutral curve to risk_neutral_prob instead of market_price")
    
    # Pipeline control
    parser.add_argument("--skip-data-fetch", action="store_true",
                        help="Skip the data fetching step (use existing data)")
    
    args = parser.parse_args()
    
    python = sys.executable
    
    # Step 1: Fetch data
    if not args.skip_data_fetch:
        # data_fetcher is now in core/data/data_fetcher.py
        if not run_step("Fetching BTC Data", [python, "core/data/data_fetcher.py"]):
            sys.exit(1)
    else:
        print("\n⏭️  Skipping data fetch (--skip-data-fetch)")
    
    # Step 2: Run batch pricing
    batch_cmd = [
        python, "scripts/pipelines/batch_pricing_runner.py",
        "--slug-pattern", args.slug_pattern,
        "--day-range", str(args.day_range[0]), str(args.day_range[1]),
        "--num-sims", str(args.num_sims),
        "--min-volume", str(args.min_volume),
    ]
    if not run_step("Running Batch Pricing", batch_cmd):
        sys.exit(1)
    
    # Note: batch_pricing_runner.py already runs fit_probability_curves internally
    # No need to run it again here
    
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
