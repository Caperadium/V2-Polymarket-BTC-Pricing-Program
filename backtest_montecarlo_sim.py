#!/usr/bin/env python3
"""
backtest_montecarl_sim.py

Monte Carlo simulation for backtest PnL analysis.
Shuffles outcomes within each expiry to create a null distribution,
then compares actual PnL against the shuffled distribution.

Usage:
    from backtest_montecarl_sim import run_shuffle_test, plot_shuffle_results
    
    actual_pnl, shuffled_pnls, percentile = run_shuffle_test(trades_df, n_iter=500)
    fig = plot_shuffle_results(actual_pnl, shuffled_pnls, percentile)
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


def compute_pnl_from_outcome(
    side: str, 
    stake: float, 
    entry_price: float, 
    outcome_yes: float
) -> float:
    """
    Compute PnL for a single trade given the outcome.
    
    Args:
        side: "YES" or "NO"
        stake: Dollar amount invested
        entry_price: Price paid (for YES side, else 1 - market_price for NO)
        outcome_yes: 1.0 if YES wins, 0.0 if NO wins
        
    Returns:
        PnL in dollars
    """
    if pd.isna(outcome_yes) or pd.isna(stake) or pd.isna(entry_price):
        return np.nan
    
    size_shares = stake / entry_price if entry_price > 0 else 0
    
    if side.upper() == "YES":
        payout = size_shares * outcome_yes
    else:
        payout = size_shares * (1.0 - outcome_yes)
    
    return payout - stake


def recompute_pnl(df: pd.DataFrame) -> pd.Series:
    """
    Recompute PnL for all trades in DataFrame using current outcome_yes values.
    
    Args:
        df: DataFrame with columns: side, stake, entry_price, outcome_yes
        
    Returns:
        Series of PnL values
    """
    pnl_values = []
    
    for _, row in df.iterrows():
        side = str(row.get("side", "YES"))
        stake = float(row.get("stake", 0))
        entry_price = float(row.get("entry_price", 0))
        outcome_yes = float(row.get("outcome_yes", np.nan))
        
        pnl = compute_pnl_from_outcome(side, stake, entry_price, outcome_yes)
        pnl_values.append(pnl)
    
    return pd.Series(pnl_values, index=df.index)


def run_shuffle_test(
    trades_df: pd.DataFrame,
    n_iter: int = 500,
    expiry_col: str = "expiry_date",
    seed: Optional[int] = None,
) -> Tuple[float, List[float], float]:
    """
    Run Monte Carlo shuffle test on backtest trades.
    
    For each iteration:
    1. Group trades by expiry date
    2. Shuffle outcomes within each expiry group
    3. Recompute total PnL
    
    This tests whether actual PnL is significantly better than
    random assignment of the same outcomes to the same trades.
    
    Args:
        trades_df: DataFrame of backtest trades with columns:
                   - expiry_date (or expiry_col)
                   - outcome_yes (0 or 1)
                   - side ("YES" or "NO")
                   - stake (dollars)
                   - entry_price
        n_iter: Number of shuffle iterations (default 500)
        expiry_col: Column name for expiry grouping
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (actual_pnl, shuffled_pnls, percentile)
        - actual_pnl: Your actual total PnL
        - shuffled_pnls: List of N shuffled PnL totals
        - percentile: Fraction of shuffled PnLs below your actual
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Filter to settled trades only
    df = trades_df.copy()
    if "settled" in df.columns:
        df = df[df["settled"] == True]
    
    # Require outcome_yes to be valid
    df = df[df["outcome_yes"].notna()]
    
    if df.empty:
        return 0.0, [], 0.0
    
    # Compute actual PnL
    if "pnl" in df.columns:
        actual_pnl = float(df["pnl"].sum())
    else:
        df["pnl"] = recompute_pnl(df)
        actual_pnl = float(df["pnl"].sum())
    
    # Fallback expiry column
    if expiry_col not in df.columns:
        if "expiry_key" in df.columns:
            expiry_col = "expiry_key"
        else:
            # No expiry column - shuffle all outcomes globally
            expiry_col = None
    
    shuffled_pnls: List[float] = []
    
    for _ in range(n_iter):
        shuffled = df.copy()
        
        if expiry_col is not None:
            # Shuffle within each expiry group
            for expiry, grp in df.groupby(expiry_col, observed=True):
                outcomes = grp["outcome_yes"].values.copy()
                np.random.shuffle(outcomes)
                shuffled.loc[grp.index, "outcome_yes"] = outcomes
        else:
            # Shuffle all outcomes globally
            outcomes = df["outcome_yes"].values.copy()
            np.random.shuffle(outcomes)
            shuffled["outcome_yes"] = outcomes
        
        # Recompute PnL with shuffled outcomes
        shuffled["pnl"] = recompute_pnl(shuffled)
        shuffled_pnls.append(float(shuffled["pnl"].sum()))
    
    # Compute percentile (fraction of shuffled < actual)
    percentile = float(np.mean(np.array(shuffled_pnls) < actual_pnl))
    
    return actual_pnl, shuffled_pnls, percentile


def plot_shuffle_results(
    actual_pnl: float,
    shuffled_pnls: List[float],
    percentile: float,
):
    """
    Create a Plotly histogram of shuffled PnL distribution with actual marked.
    
    Args:
        actual_pnl: Your actual total PnL
        shuffled_pnls: List of shuffled PnL totals
        percentile: Fraction of shuffled PnLs below actual
        
    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Histogram of shuffled PnLs
    fig.add_trace(go.Histogram(
        x=shuffled_pnls,
        name="Shuffled PnL",
        opacity=0.7,
        nbinsx=50,
    ))
    
    # Vertical line for actual PnL
    fig.add_vline(
        x=actual_pnl,
        line_color="red",
        line_width=3,
        annotation_text=f"Actual: ${actual_pnl:.2f} ({percentile*100:.1f}%ile)",
        annotation_position="top right",
    )
    
    # Add summary statistics
    mean_shuffled = np.mean(shuffled_pnls)
    std_shuffled = np.std(shuffled_pnls)
    
    fig.update_layout(
        title=f"Monte Carlo Shuffle Test (n={len(shuffled_pnls)})",
        xaxis_title="Total PnL ($)",
        yaxis_title="Frequency",
        showlegend=True,
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"Shuffled μ: ${mean_shuffled:.2f}<br>Shuffled σ: ${std_shuffled:.2f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
            )
        ]
    )
    
    return fig


def get_summary_stats(
    actual_pnl: float,
    shuffled_pnls: List[float],
    percentile: float,
) -> dict:
    """
    Get summary statistics from shuffle test.
    
    Returns:
        Dictionary with:
        - actual_pnl
        - shuffled_mean
        - shuffled_std
        - percentile
        - z_score (how many std devs above mean)
        - is_significant (p < 0.05, i.e. percentile > 0.95)
    """
    shuffled_arr = np.array(shuffled_pnls)
    mean_shuffled = float(np.mean(shuffled_arr))
    std_shuffled = float(np.std(shuffled_arr))
    
    z_score = (actual_pnl - mean_shuffled) / std_shuffled if std_shuffled > 0 else 0
    
    return {
        "actual_pnl": actual_pnl,
        "shuffled_mean": mean_shuffled,
        "shuffled_std": std_shuffled,
        "percentile": percentile,
        "z_score": z_score,
        "is_significant": percentile > 0.95,
    }


# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Monte Carlo shuffle test on backtest trades")
    parser.add_argument("--input", "-i", required=True, help="Path to trades CSV")
    parser.add_argument("--iterations", "-n", type=int, default=500, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    trades_df = pd.read_csv(args.input)
    
    print(f"Loaded {len(trades_df)} trades from {args.input}")
    
    actual_pnl, shuffled_pnls, percentile = run_shuffle_test(
        trades_df, 
        n_iter=args.iterations,
        seed=args.seed,
    )
    
    stats = get_summary_stats(actual_pnl, shuffled_pnls, percentile)
    
    print(f"\n{'='*50}")
    print("MONTE CARLO SHUFFLE TEST RESULTS")
    print(f"{'='*50}")
    print(f"Actual PnL:      ${stats['actual_pnl']:>10.2f}")
    print(f"Shuffled Mean:   ${stats['shuffled_mean']:>10.2f}")
    print(f"Shuffled Std:    ${stats['shuffled_std']:>10.2f}")
    print(f"Z-Score:          {stats['z_score']:>10.2f}")
    print(f"Percentile:       {stats['percentile']*100:>10.1f}%")
    print(f"{'='*50}")
    
    if stats['is_significant']:
        print("✅ Result is statistically significant (p < 0.05)")
    else:
        print("❌ Result is NOT statistically significant")
