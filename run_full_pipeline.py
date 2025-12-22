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
from pathlib import Path


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
        if not run_step("Fetching BTC Data", [python, "data_fetcher.py"]):
            sys.exit(1)
    else:
        print("\n⏭️  Skipping data fetch (--skip-data-fetch)")
    
    # Step 2: Run batch pricing
    batch_cmd = [
        python, "batch_pricing_runner.py",
        "--slug-pattern", args.slug_pattern,
        "--day-range", str(args.day_range[0]), str(args.day_range[1]),
        "--num-sims", str(args.num_sims),
        "--min-volume", str(args.min_volume),
    ]
    if not run_step("Running Batch Pricing", batch_cmd):
        sys.exit(1)
    
    # Step 3: Fit probability curves
    fit_cmd = [python, "fit_probability_curves.py"]
    if args.use_rn_prob:
        fit_cmd.append("--use-rn-prob")
    
    if not run_step("Fitting Probability Curves", fit_cmd):
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
