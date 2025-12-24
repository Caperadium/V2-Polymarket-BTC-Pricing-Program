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
    rng = np.random.default_rng(seed)
    
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
                rng.shuffle(outcomes)
                shuffled.loc[grp.index, "outcome_yes"] = outcomes
        else:
            # Shuffle all outcomes globally
            outcomes = df["outcome_yes"].values.copy()
            rng.shuffle(outcomes)
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
        - p_hat: empirical p-value = (k+1)/(N+1) where k = count(shuffled >= actual)
        - tail_distance: (actual - mean) / std (same as z_score but clearer name)
        - shuffled_p95: 95th percentile of shuffled PnLs
    """
    shuffled_arr = np.array(shuffled_pnls)
    mean_shuffled = float(np.mean(shuffled_arr))
    std_shuffled = float(np.std(shuffled_arr))
    
    z_score = (actual_pnl - mean_shuffled) / std_shuffled if std_shuffled > 0 else 0
    
    # p_hat: empirical p-value using (k+1)/(N+1) formula
    # k = count of shuffled PnLs >= actual PnL (how many times random did as well or better)
    n_iter = len(shuffled_pnls)
    k = int(np.sum(shuffled_arr >= actual_pnl))
    p_hat = (k + 1) / (n_iter + 1) if n_iter > 0 else 1.0
    
    # Tail distance is the same as z_score but named for clarity
    tail_distance = z_score
    
    # 95th percentile of shuffled distribution
    shuffled_p95 = float(np.percentile(shuffled_arr, 95)) if len(shuffled_arr) > 0 else np.nan
    
    return {
        "actual_pnl": actual_pnl,
        "shuffled_mean": mean_shuffled,
        "shuffled_std": std_shuffled,
        "percentile": percentile,
        "z_score": z_score,
        "is_significant": percentile > 0.95,
        "p_hat": p_hat,
        "tail_distance": tail_distance,
        "shuffled_p95": shuffled_p95,
    }


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first available column from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _compute_edge_magnitude(
    df: pd.DataFrame,
    prob_col: Optional[str] = None,
    price_col: Optional[str] = None,
) -> pd.Series:
    """
    Compute edge magnitude = abs(model_prob - market_yes_price).
    
    Uses direction-agnostic abs(edge) so YES and NO trades are binned
    by edge strength, not direction.
    
    Args:
        df: DataFrame with probability and price columns
        prob_col: Override probability column name
        price_col: Override price column name
        
    Returns:
        Series of abs(edge) values, with inf replaced by NaN
    """
    if prob_col is None:
        prob_col = _find_column(df, ["model_prob_used", "model_prob", "p_model_cal", "p_model_fit"])
    if price_col is None:
        price_col = _find_column(df, ["market_yes_price", "market_price"])
    
    if prob_col is None or price_col is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    edge = pd.to_numeric(df[prob_col], errors="coerce") - pd.to_numeric(df[price_col], errors="coerce")
    return edge.abs().replace([np.inf, -np.inf], np.nan)


def _compute_edge_magnitude_for_trades(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Compute edge magnitude for trades = abs(model_prob - market_yes_price).
    
    IMPORTANT: For both YES and NO trades, we use the YES-side market price
    (market_price column) to compute edge, NOT entry_price. This ensures
    consistent binning across trade sides.
    
    For YES trades: entry_price == market_price (YES price)
    For NO trades: entry_price == 1 - market_price (NO price), but we still
                   want to bin by the YES-price edge for consistency.
    
    Returns:
        Series of abs(edge) values, with inf replaced by NaN
    """
    # Find probability column
    prob_col = _find_column(df, ["model_prob", "model_prob_used", "p_model_cal", "p_model_fit"])
    # ALWAYS use market_price (the YES price), not entry_price
    price_col = _find_column(df, ["market_price", "market_yes_price"])
    
    if prob_col is None or price_col is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    edge = pd.to_numeric(df[prob_col], errors="coerce") - pd.to_numeric(df[price_col], errors="coerce")
    return edge.abs().replace([np.inf, -np.inf], np.nan)



def run_decile_conditioned_shuffle_test(
    trades_df: pd.DataFrame,
    all_priced_df: pd.DataFrame,
    n_iter: int = 500,
    seed: Optional[int] = None,
    expiry_col: str = "expiry_date",
    snapshot_col: str = "snapshot_time",
    n_deciles: int = 10,
) -> Tuple[float, List[float], float, dict]:
    """
    Run Monte Carlo shuffle test using all priced contracts as the outcome pool.
    
    This is a stronger null model than expiry-only shuffling because:
    1. Outcomes are drawn from ALL evaluated contracts, not just taken trades
    2. Conditioning on edge deciles respects that different edge strengths
       correspond to different base win rates / difficulty levels
    3. Grouping by (snapshot_time, expiry_date, edge_decile) avoids mixing
       outcomes from different market moments
    
    Key implementation details:
    - Trades are grouped, and outcomes are sampled at the GROUP level
    - Sample WITHOUT replacement when pool_size >= group_size, else WITH replacement
    - Fallback cascade checks pool ADEQUACY, not just key existence
    - On any setup failure, falls back to existing run_shuffle_test()
    
    Args:
        trades_df: Settled trades with side, stake, entry_price, outcome_yes, market_price
        all_priced_df: All evaluated contracts with outcome_yes, market_yes_price,
                       model_prob_used (or equivalent columns)
        n_iter: Number of Monte Carlo iterations
        seed: Random seed for reproducibility
        expiry_col: Column name for expiry date grouping
        snapshot_col: Column name for snapshot time grouping
        n_deciles: Target number of edge deciles (may be reduced if duplicates)
        
    Returns:
        Tuple of (actual_pnl, shuffled_pnls, percentile, diagnostics_dict)
    """
    rng = np.random.default_rng(seed)
    
    diagnostics = {
        "shuffle_mode": "decile_conditioned_all_priced",
        "n_trades_used": 0,
        "n_all_priced_used": 0,
        "n_deciles_used": 0,
        "n_unmatched_trades": 0,
        "fallback_counts": {"snapshot_expiry_decile": 0, "expiry_decile": 0, "expiry_only": 0, "global": 0},
    }
    
    # --- Helper to fallback to existing shuffle test ---
    def fallback_to_expiry_shuffle(reason: str):
        """Call existing run_shuffle_test and wrap results."""
        diagnostics["shuffle_mode"] = f"expiry_only_fallback_{reason}"
        diagnostics["fallback_reason"] = reason
        actual_pnl, shuffled_pnls, percentile = run_shuffle_test(
            trades_df, n_iter=n_iter, expiry_col=expiry_col, seed=seed
        )
        return actual_pnl, shuffled_pnls, percentile, diagnostics
    
    # --- Validate inputs ---
    if trades_df is None or trades_df.empty:
        diagnostics["error"] = "No trades provided"
        return 0.0, [], 0.0, diagnostics
    
    if all_priced_df is None or all_priced_df.empty:
        return fallback_to_expiry_shuffle("missing_all_priced")
    
    # Filter trades to settled only
    df_trades = trades_df.copy()
    if "settled" in df_trades.columns:
        df_trades = df_trades[df_trades["settled"] == True]
    df_trades = df_trades[df_trades["outcome_yes"].notna()].copy()
    
    if len(df_trades) < 5:
        diagnostics["error"] = f"Insufficient settled trades ({len(df_trades)})"
        return 0.0, [], 0.0, diagnostics
    
    # Work on minimal columns for memory efficiency
    required_all_cols = ["outcome_yes"]
    if expiry_col in all_priced_df.columns:
        required_all_cols.append(expiry_col)
    if snapshot_col in all_priced_df.columns:
        required_all_cols.append(snapshot_col)
    
    # Add probability and price columns for edge computation
    prob_col_all = _find_column(all_priced_df, ["model_prob_used", "model_prob", "p_model_cal"])
    price_col_all = _find_column(all_priced_df, ["market_yes_price", "market_price"])
    
    if prob_col_all is None or price_col_all is None:
        return fallback_to_expiry_shuffle("missing_cols")
    
    required_all_cols.extend([prob_col_all, price_col_all])
    df_all = all_priced_df[list(set(required_all_cols))].copy()
    df_all = df_all[df_all["outcome_yes"].notna()]
    
    if df_all.empty:
        return fallback_to_expiry_shuffle("no_outcomes")
    
    diagnostics["n_trades_used"] = len(df_trades)
    diagnostics["n_all_priced_used"] = len(df_all)
    
    # --- Compute edge magnitude and decile bins ---
    edge_all = _compute_edge_magnitude(df_all, prob_col_all, price_col_all)
    edge_valid = edge_all.dropna()
    
    if len(edge_valid) < n_deciles * 2:
        return fallback_to_expiry_shuffle("insufficient_edge_data")
    
    # Compute quantile edges explicitly
    quantile_edges = np.quantile(edge_valid.values, np.linspace(0, 1, n_deciles + 1))
    quantile_edges = np.unique(quantile_edges)
    n_deciles_used = len(quantile_edges) - 1
    
    if n_deciles_used < 3:
        return fallback_to_expiry_shuffle("degenerate_bins")
    
    diagnostics["n_deciles_used"] = n_deciles_used
    
    # Assign decile labels using consistent bin edges
    df_all["edge_decile"] = pd.cut(
        edge_all, bins=quantile_edges, labels=False, include_lowest=True
    ).astype("Int64")
    
    # Compute trade edge using market_price (YES price), not entry_price
    edge_trades = _compute_edge_magnitude_for_trades(df_trades)
    df_trades["edge_decile"] = pd.cut(
        edge_trades, bins=quantile_edges, labels=False, include_lowest=True
    ).astype("Int64")
    
    # Count unmatched trades (edge outside bin range or NaN)
    n_unmatched = df_trades["edge_decile"].isna().sum()
    diagnostics["n_unmatched_trades"] = int(n_unmatched)
    
    # If too many unmatched, fallback
    if n_unmatched > len(df_trades) * 0.2:  # >20% unmatched
        return fallback_to_expiry_shuffle("too_many_unmatched")
    
    # --- Compute actual PnL ---
    if "pnl" in df_trades.columns:
        actual_pnl = float(df_trades["pnl"].sum())
    else:
        df_trades["pnl"] = recompute_pnl(df_trades)
        actual_pnl = float(df_trades["pnl"].sum())
    
    # --- Build outcome pools by group keys ---
    has_snapshot = snapshot_col in df_all.columns and snapshot_col in df_trades.columns
    has_expiry = expiry_col in df_all.columns and expiry_col in df_trades.columns
    
    # Pre-build pools for efficient lookup
    pools_snapshot_expiry_decile = {}
    pools_expiry_decile = {}
    pools_expiry = {}
    global_pool = df_all["outcome_yes"].dropna().values
    
    if has_snapshot and has_expiry:
        for key, grp in df_all.groupby([snapshot_col, expiry_col, "edge_decile"], observed=True, dropna=False):
            outcomes = grp["outcome_yes"].dropna().values
            if len(outcomes) > 0:
                pools_snapshot_expiry_decile[key] = outcomes
    
    if has_expiry:
        for key, grp in df_all.groupby([expiry_col, "edge_decile"], observed=True, dropna=False):
            outcomes = grp["outcome_yes"].dropna().values
            if len(outcomes) > 0:
                pools_expiry_decile[key] = outcomes
        
        for key, grp in df_all.groupby(expiry_col, observed=True, dropna=False):
            outcomes = grp["outcome_yes"].dropna().values
            if len(outcomes) > 0:
                pools_expiry[key] = outcomes
    
    # --- Build trade groups at MOST GRANULAR level first ---
    # Always group by (snapshot, expiry, decile) if available
    # Pool selection with adequacy check happens per-group, not per-trade
    granular_groups = {}  # key -> list of trade indices
    
    for idx, row in df_trades.iterrows():
        decile = row.get("edge_decile")
        expiry = row.get(expiry_col) if has_expiry else None
        snapshot = row.get(snapshot_col) if has_snapshot else None
        
        # Build the most granular key possible
        granular_key = (snapshot, expiry, decile)
        
        if granular_key not in granular_groups:
            granular_groups[granular_key] = []
        granular_groups[granular_key].append(idx)
    
    # --- Monte Carlo iterations with group-level sampling and adequacy cascade ---
    shuffled_pnls: List[float] = []
    fallback_counts = {"snapshot_expiry_decile": 0, "expiry_decile": 0, "expiry_only": 0, "global": 0}
    
    # Pre-extract required columns for fast PnL computation
    sides = df_trades["side"].values
    stakes = df_trades["stake"].values
    entry_prices = df_trades["entry_price"].values
    idx_to_pos = {idx: i for i, idx in enumerate(df_trades.index)}
    
    for _ in range(n_iter):
        shuffled_outcomes = np.full(len(df_trades), np.nan)
        
        # Process each granular group with pool adequacy cascade
        for granular_key, indices in granular_groups.items():
            snapshot, expiry, decile = granular_key
            group_size = len(indices)
            
            # Pool adequacy cascade: check each level's pool size vs group size
            pool = None
            fallback_level = None
            
            # Level 1: (snapshot, expiry, decile)
            if has_snapshot and has_expiry and pd.notna(decile):
                sed_key = (snapshot, expiry, decile)
                if sed_key in pools_snapshot_expiry_decile:
                    candidate = pools_snapshot_expiry_decile[sed_key]
                    if len(candidate) >= group_size:
                        pool = candidate
                        fallback_level = "snapshot_expiry_decile"
            
            # Level 2: (expiry, decile) - broader conditioning
            if pool is None and has_expiry and pd.notna(decile):
                ed_key = (expiry, decile)
                if ed_key in pools_expiry_decile:
                    candidate = pools_expiry_decile[ed_key]
                    if len(candidate) >= group_size:
                        pool = candidate
                        fallback_level = "expiry_decile"
            
            # Level 3: (expiry) only - even broader
            if pool is None and has_expiry and pd.notna(expiry):
                if expiry in pools_expiry:
                    candidate = pools_expiry[expiry]
                    if len(candidate) >= group_size:
                        pool = candidate
                        fallback_level = "expiry_only"
            
            # Level 4: global pool (may still need replacement if tiny)
            if pool is None:
                pool = global_pool
                fallback_level = "global"
            
            fallback_counts[fallback_level] += group_size
            
            if len(pool) == 0:
                continue
            
            # Sample: without replacement if pool >= group, else with replacement
            if len(pool) >= group_size:
                sampled = rng.choice(pool, size=group_size, replace=False)
            else:
                sampled = rng.choice(pool, size=group_size, replace=True)
            
            # Assign sampled outcomes to trade positions
            for i, idx in enumerate(indices):
                pos = idx_to_pos[idx]
                shuffled_outcomes[pos] = sampled[i]

        
        # Compute PnL for this iteration using vectorized approach
        total_pnl = 0.0
        for i in range(len(df_trades)):
            pnl = compute_pnl_from_outcome(
                sides[i], stakes[i], entry_prices[i], shuffled_outcomes[i]
            )
            if not np.isnan(pnl):
                total_pnl += pnl
        
        shuffled_pnls.append(total_pnl)
    
    # Normalize fallback counts to per-iteration average
    diagnostics["fallback_counts"] = {k: v / n_iter for k, v in fallback_counts.items()}
    
    # Compute percentile
    percentile = float(np.mean(np.array(shuffled_pnls) < actual_pnl))
    
    return actual_pnl, shuffled_pnls, percentile, diagnostics



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
