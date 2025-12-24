#!/usr/bin/env python3
"""
auto_reco.py

Generate Kelly-style trade recommendations from batch_with_fits data.

Enhancements:
- Correlation-aware Kelly sizing.
- Directional consistency per expiry (YES-only or NO-only).
- Staleness multipliers for batch age and per-market price timestamps.
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def load_latest_fitted_batch(directory: str = "fitted_batch_results") -> Optional[pd.DataFrame]:
    """
    Load the most recent batch_with_fits.csv from the fitted_batch_results directory.
    
    Returns:
        DataFrame with fitted batch data, or None if not found.
    """
    # Search for batch_with_fits.csv in timestamped subdirectories
    pattern = Path(directory) / "*" / "batch_with_fits.csv"
    files = glob.glob(str(pattern))
    
    if not files:
        # Fallback: search directly in directory
        pattern_flat = Path(directory) / "batch_with_fits.csv"
        files = glob.glob(str(pattern_flat))
    
    if not files:
        return None
    
    # Get most recent by modification time
    latest = max(files, key=lambda f: Path(f).stat().st_mtime)
    print(f"Loading fitted batch from: {latest}")
    return pd.read_csv(latest)

# Correlation + exposure defaults
CORRELATION_PENALTY_DEFAULT = 0.25
MAX_CAP_PER_EXPIRY_FRAC_DEFAULT = 0.15
MAX_CAP_TOTAL_FRAC_DEFAULT = 0.35
STALE_SOFT_HOURS = 4.0
STALE_HARD_HOURS = 12.0
PRICE_STALE_SOFT_HOURS = 1.0
PRICE_STALE_HARD_HOURS = 6.0
DEFAULT_MIN_TRADE_USD = 5.0

LAST_RECO_DEBUG: Optional[pd.DataFrame] = None


@dataclass
class TradeRecommendation:
    slug: str
    question: str
    expiry_key: str
    strike: float
    side: str
    pricing_date: Optional[pd.Timestamp]
    market_price: float
    entry_price: float
    model_prob: float
    rn_prob: Optional[float]
    edge: float
    kelly_fraction_full: float
    kelly_fraction_full_effective: float
    kelly_fraction_target: float
    kelly_fraction_existing: float
    kelly_fraction_applied: float
    suggested_stake: float
    expected_value_per_contract: float
    expected_value_dollars: float
    expiry_group_risk: float
    stability_penalty: float
    stale_mult: float
    price_staleness_mult: float
    batch_age_hours: Optional[float]
    price_age_hours: Optional[float]
    expiry_shape_label: str
    direction: str
    corr_multiplier: float
    notes: str


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _derive_expiry_key(df: pd.DataFrame) -> pd.Series:
    if "expiry_key" in df.columns:
        return df["expiry_key"].astype(str)
    if "expiry_date" in df.columns:
        return df["expiry_date"].astype(str)
    # Also check for capitalized Date column from batch_results
    if "Date" in df.columns:
        return df["Date"].astype(str)
    t_col = _pick_column(df, ["t_days", "T_days"])
    if t_col:
        return df[t_col].astype(float).round(3).astype(str)
    return pd.Series(["unknown"] * len(df), index=df.index)


def _compute_stability_penalty(df: pd.DataFrame) -> np.ndarray:
    penalty = np.ones(len(df), dtype=float)
    if "fit_residual_abs" in df.columns:
        penalty *= np.clip(1.0 - df["fit_residual_abs"].abs() / 0.3, 0.2, 1.0)
    if "monotonic_violation_flag" in df.columns:
        penalty *= np.where(df["monotonic_violation_flag"].astype(bool), 0.6, 1.0)
    if "edge_residual_zscore" in df.columns:
        penalty *= np.clip(1.0 - df["edge_residual_zscore"].abs() / 4.0, 0.2, 1.0)
    return penalty


def kelly_fraction_yes(p: float, q: float) -> float:
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return max((p - q) / (1.0 - q), 0.0)


def kelly_fraction_no(p: float, q: float) -> float:
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return max((q - p) / q, 0.0)


def _infer_direction(row: pd.Series) -> str:
    for key in ("side", "polymarket_outcome", "outcome"):
        val = row.get(key)
        if isinstance(val, str):
            val_norm = val.strip().lower()
            if val_norm == "yes":
                return "YES"
            if val_norm == "no":
                return "NO"
    return "UNKNOWN"


def _compute_corr_multiplier(n_strikes: int, penalty: float) -> float:
    if penalty <= 0.0 or n_strikes <= 1:
        return 1.0
    return 1.0 / (1.0 + penalty * max(n_strikes - 1, 0))


def recommend_trades(
    df: pd.DataFrame,
    bankroll: float,
    positions_df: Optional[pd.DataFrame] = None,
    kelly_fraction: float = 0.15,
    min_edge: float = 0.06,
    max_bets_per_expiry: int = 3,
    max_capital_per_expiry_frac: float = MAX_CAP_PER_EXPIRY_FRAC_DEFAULT,
    max_capital_total_frac: float = MAX_CAP_TOTAL_FRAC_DEFAULT,
    max_net_delta_frac: float = 0.20,
    min_price: float = 0.03,
    max_price: float = 0.95,
    min_model_prob: float = 0.0,
    max_model_prob: float = 1.0,
    require_active: bool = True,
    use_stability_penalty: bool = True,
    allow_no: bool = True,
    correlation_penalty: float = CORRELATION_PENALTY_DEFAULT,
    min_trade_usd: float = DEFAULT_MIN_TRADE_USD,
    disable_staleness: bool = False,
    check_existing_consistency: bool = False,
    current_open_positions: Optional[pd.DataFrame] = None,
    use_fixed_stake: bool = False,
    fixed_stake_amount: float = 10.0,
    max_dte: Optional[float] = None,
    use_prob_threshold: bool = False,
    prob_threshold_yes: float = 0.7,
    prob_threshold_no: float = 0.3,
    max_moneyness: Optional[float] = None,
    min_moneyness: Optional[float] = None,
) -> List[TradeRecommendation]:
    """
    Rank opportunities and size positions using fractional Kelly.
    
    Args:
        current_open_positions: Optional DataFrame of currently open positions
            from a backtest. When provided, these are counted toward max_bets_per_expiry
            to ensure total positions per expiry don't exceed the limit.
    """
    global LAST_RECO_DEBUG
    if df is None or df.empty:
        LAST_RECO_DEBUG = pd.DataFrame()
        return []
    data = df.copy()
    expiry_series = _derive_expiry_key(data)
    price_col = _pick_column(data, ["market_price", "market_pr", "Polymarket_Price"])
    # Prefer fitted logistic model probability, fallback to raw MC probability
    model_col = _pick_column(data, ["p_model_cal", "p_real_mc", "p_model_fit", "model_probability", "Model_Prob"])
    # Prefer fitted RN/market curve, fallback to raw RN probability
    rn_col = _pick_column(data, ["p_rn_fit", "risk_neutral_prob_fit", "risk_neutral_prob"])
    pricing_col = _pick_column(data, ["pricing_date", "date", "as_of_date"])

    if price_col is None or model_col is None:
        raise ValueError("Batch file missing market_price or model probability columns.")

    data["market_price"] = pd.to_numeric(data[price_col], errors="coerce")
    data["model_prob"] = pd.to_numeric(data[model_col], errors="coerce")
    if rn_col:
        data["rn_prob"] = pd.to_numeric(data[rn_col], errors="coerce")
    else:
        data["rn_prob"] = np.nan
    data["expiry_key"] = expiry_series
    if pricing_col:
        data["pricing_date"] = pd.to_datetime(data[pricing_col], errors="coerce")
    else:
        data["pricing_date"] = pd.NaT

    if require_active:
        mask = pd.Series(True, index=data.index)
        if "active" in data.columns:
            mask &= data["active"].astype(bool)
        if "closed" in data.columns:
            mask &= ~data["closed"].astype(bool)
        if "archived" in data.columns:
            mask &= ~data["archived"].astype(bool)
        data = data[mask]

    data = data.dropna(subset=["market_price", "model_prob"])
    
    # Filter by model probability
    data = data[data["model_prob"].between(min_model_prob, max_model_prob)]
    
    # Filter rows where at least one side's entry price is within range
    # YES entry price = market_price, NO entry price = 1 - market_price
    yes_entry_valid = data["market_price"].between(min_price, max_price)
    no_entry_valid = (1.0 - data["market_price"]).between(min_price, max_price)
    if allow_no:
        price_mask = yes_entry_valid | no_entry_valid
    else:
        price_mask = yes_entry_valid
    data = data[price_mask]

    if use_stability_penalty:
        data["stability_penalty"] = _compute_stability_penalty(data)
    else:
        data["stability_penalty"] = 1.0

    batch_age_hours = None
    batch_timestamp = None
    batch_ts_candidates = [
        "batch_timestamp",
        "run_timestamp",
        "pricing_timestamp",
        "live_price_as_of",
    ]
    for col in batch_ts_candidates:
        if col in data.columns:
            ts_value = data[col].dropna().iloc[0] if not data[col].dropna().empty else None
            batch_timestamp = _parse_timestamp(ts_value)
            if batch_timestamp:
                break
    stale_mult, batch_age_hours = _compute_stale_multiplier(batch_timestamp, STALE_SOFT_HOURS, STALE_HARD_HOURS)
    if disable_staleness:
        stale_mult, batch_age_hours = 1.0, None
    data["batch_age_hours"] = batch_age_hours
    data["stale_mult"] = stale_mult

    price_ts_candidates = [
        "market_price_timestamp",
        "polymarket_price_timestamp",
        "price_timestamp",
        "polymarket_price_as_of",
        "market_price_as_of",
    ]
    price_col_ts = next((c for c in price_ts_candidates if c in data.columns), None)
    if disable_staleness:
        data["price_staleness_mult"] = 1.0
        data["price_age_hours"] = np.nan
    elif price_col_ts:
        price_mult = []
        price_ages = []
        for val in data[price_col_ts]:
            ts = _parse_timestamp(val)
            mult, age = _compute_stale_multiplier(ts, PRICE_STALE_SOFT_HOURS, PRICE_STALE_HARD_HOURS)
            price_mult.append(mult)
            price_ages.append(age if age is not None else np.nan)
        data["price_staleness_mult"] = price_mult
        data["price_age_hours"] = price_ages
    else:
        data["price_staleness_mult"] = 1.0
        data["price_age_hours"] = np.nan

    total_multiplier = data["stability_penalty"].astype(float) * data["price_staleness_mult"].astype(float) * stale_mult
    data["total_multiplier"] = total_multiplier
    if data.empty:
        return []

    # Apply max DTE filter if specified
    if max_dte is not None:
        t_col = _pick_column(data, ["t_days", "T_days", "dte_days"])
        if t_col is not None:
            data["_dte"] = pd.to_numeric(data[t_col], errors="coerce")
            data = data[data["_dte"] <= max_dte].copy()
            if data.empty:
                return []

    # Apply moneyness filter if specified
    if max_moneyness is not None or min_moneyness is not None:
        m_col = _pick_column(data, ["moneyness"])
        if m_col is not None:
            data["_moneyness"] = pd.to_numeric(data[m_col], errors="coerce").abs()
            if max_moneyness is not None:
                data = data[data["_moneyness"] <= max_moneyness].copy()
            if min_moneyness is not None:
                data = data[data["_moneyness"] >= min_moneyness].copy()
            if data.empty:
                return []

    candidates: List[Dict[str, object]] = []
    for _, row in data.iterrows():
        p = float(row["model_prob"])
        q = float(row["market_price"])
        penalty = float(row["stability_penalty"])
        base_info = {
            "slug": row.get("slug", ""),
            "question": row.get("question", ""),
            "expiry_key": row["expiry_key"],
            "expiry_date": row.get("expiry_date", pd.NaT),
            "strike": float(row.get("strike", np.nan)),
            "pricing_date": row.get("pricing_date", pd.NaT),
            "market_price": q,
            "model_prob": p,
            "rn_prob": row.get("rn_prob", np.nan),
            "stability_penalty": float(row["stability_penalty"]),
            "stale_mult": stale_mult,
            "price_staleness_mult": float(row["price_staleness_mult"]),
            "batch_age_hours": batch_age_hours,
            "price_age_hours": row.get("price_age_hours", np.nan),
            "total_multiplier": float(row["total_multiplier"]),
        }
        yes_edge = p - q
        no_edge = q - p
        
        # Probability threshold mode: trade based on model_prob thresholds + edge requirement
        if use_prob_threshold:
            # Trade YES if p >= threshold_yes AND edge >= min_edge
            if p >= prob_threshold_yes and yes_edge >= min_edge and min_price <= q <= max_price:
                candidates.append({**base_info, "side": "YES", "edge": yes_edge, "entry_price": q})
            # Trade NO if p <= threshold_no AND edge >= min_edge
            if allow_no:
                no_price = 1.0 - q
                if p <= prob_threshold_no and no_edge >= min_edge and min_price <= no_price <= max_price:
                    candidates.append({**base_info, "side": "NO", "edge": no_edge, "entry_price": no_price})
        else:
            # Edge-based mode (original logic)
            if yes_edge >= min_edge and min_price <= q <= max_price:
                candidates.append({**base_info, "side": "YES", "edge": yes_edge, "entry_price": q})
            if allow_no:
                no_price = 1.0 - q
                if no_edge >= min_edge and min_price <= no_price <= max_price:
                    candidates.append({**base_info, "side": "NO", "edge": no_edge, "entry_price": no_price})

    if not candidates:
        return []

    candidates_df = pd.DataFrame(candidates)
    candidates_df["direction"] = candidates_df.apply(_infer_direction, axis=1)
    candidates_df["score"] = candidates_df["edge"] * candidates_df["total_multiplier"]

    if candidates_df.empty:
        LAST_RECO_DEBUG = pd.DataFrame()
        return []

    # Build map of existing trades per expiry from current_open_positions
    # This ensures max_bets_per_expiry is respected across existing + new trades
    existing_trades_per_expiry: Dict[str, int] = {}
    if current_open_positions is not None and not current_open_positions.empty:
        # Normalize expiry key in open positions
        pos_df = current_open_positions.copy()
        if "expiry_key" not in pos_df.columns:
            if "expiry_date" in pos_df.columns:
                pos_df["expiry_key"] = pos_df["expiry_date"].astype(str)
            else:
                pos_df["expiry_key"] = "unknown"
        for expiry, grp in pos_df.groupby("expiry_key", observed=True):
            existing_trades_per_expiry[str(expiry)] = len(grp)

    candidates_df["shape_selected"] = False
    direction_map: Dict[str, str] = {}
    selected_indices: List[int] = []
    for expiry, group in candidates_df.groupby("expiry_key", observed=True):
        # Reduce max_bets by existing positions for this expiry
        existing_count = existing_trades_per_expiry.get(str(expiry), 0)
        adjusted_max_bets = max(0, max_bets_per_expiry - existing_count)
        if adjusted_max_bets == 0:
            # Already at or above limit for this expiry
            continue
        idxs, direction = _select_expiry_candidates(group, adjusted_max_bets)
        direction_map[expiry] = direction
        if idxs:
            selected_indices.extend(idxs)
            candidates_df.loc[idxs, "shape_selected"] = True
    debug_df = candidates_df.copy()
    if not selected_indices:
        LAST_RECO_DEBUG = debug_df
        return []
    candidates_df = candidates_df.loc[selected_indices].copy()

    group_cols = ["expiry_key", "direction"]
    if "strike" in candidates_df.columns:
        n_per_group = (
            candidates_df.groupby(group_cols, observed=True)["strike"].nunique().rename("n_strikes").reset_index()
        )
    else:
        n_per_group = (
            candidates_df.groupby(group_cols, observed=True)
            .size()
            .rename("n_strikes")
            .reset_index()
        )
    n_per_group["corr_multiplier"] = n_per_group["n_strikes"].apply(
        lambda n: _compute_corr_multiplier(int(n), correlation_penalty)
    )
    candidates_df = candidates_df.merge(
        n_per_group[group_cols + ["corr_multiplier"]],
        on=group_cols,
        how="left",
    )
    candidates_df["corr_multiplier"] = candidates_df["corr_multiplier"].fillna(1.0)

    # Existing exposure map
    existing_frac: Dict[Tuple[str, str], float] = {}
    if positions_df is not None and not positions_df.empty and bankroll > 0:
        for _, pos in positions_df.iterrows():
            slug = str(pos.get("slug", ""))
            side = str(pos.get("side", "YES")).upper()
            entry_price = float(pos.get("entry_price", np.nan))
            size_shares = float(pos.get("size_shares", np.nan))
            if np.isnan(entry_price) or np.isnan(size_shares):
                continue
            key = (slug, side)
            existing_frac[key] = existing_frac.get(key, 0.0) + (entry_price * size_shares) / bankroll

    kelly_full_vals = []
    kelly_full_effective_vals = []
    kelly_target_vals = []
    kelly_existing_vals = []
    kelly_incremental_vals = []
    for _, row in candidates_df.iterrows():
        p = float(row["model_prob"])
        q = float(row["market_price"])
        if row["side"] == "YES":
            f_star = kelly_fraction_yes(p, q)
        else:
            f_star = kelly_fraction_no(p, q)
        
        # Handle zero/negative kelly by using 0.0 instead of skipping
        if f_star <= 0.0:
            kelly_full_vals.append(0.0)
            kelly_full_effective_vals.append(0.0)
            kelly_target_vals.append(0.0)
            kelly_existing_vals.append(0.0)
            kelly_incremental_vals.append(0.0)
            continue
            
        multiplier = row.get("total_multiplier", 1.0)
        corr_mult = float(row.get("corr_multiplier", 1.0))
        f_star_effective = f_star * corr_mult * multiplier
        f_target = min(kelly_fraction * f_star_effective, 0.3)
        key = (str(row.get("slug", "")), row["side"])
        f_existing = existing_frac.get(key, 0.0)
        f_incremental = max(0.0, f_target - f_existing)
        kelly_full_vals.append(f_star)
        kelly_full_effective_vals.append(f_star_effective)
        kelly_target_vals.append(f_target)
        kelly_existing_vals.append(f_existing)
        kelly_incremental_vals.append(f_incremental)

    candidates_df["kelly_full"] = kelly_full_vals
    candidates_df["kelly_full_effective"] = kelly_full_effective_vals
    candidates_df["kelly_target"] = kelly_target_vals
    candidates_df["kelly_existing"] = kelly_existing_vals
    candidates_df["kelly_eff"] = kelly_incremental_vals

    # When using fixed stake, don't filter by Kelly sizing - use edge only
    # (edge filter already applied via min_edge earlier)
    if not use_fixed_stake:
        candidates_df = candidates_df[candidates_df["kelly_eff"] > 0]
        if candidates_df.empty:
            return []

    # Sort by score and apply per-expiry logic using _select_expiry_candidates
    candidates_df = candidates_df.sort_values("score", ascending=False)
    
    # We must group by expiry_key and select the best consistent subset
    selected_indices = []
    
    for expiry, group in candidates_df.groupby("expiry_key", observed=True):
        # 1. Gather existing positions for this expiry (Optional)
        existing_for_expiry = pd.DataFrame()
        if check_existing_consistency and positions_df is not None and not positions_df.empty:
             # Match by expiry (string)
             # Ensure expiry format matches. positions_df usually has 'expiry_date'
             # We might need to construct a similar 'expiry_key' or match by date string.
             # Simplest approach: Filter positions_df where 'expiry_date' matches this group's 'expiry_date' (if available)
             # or 'expiry_key'.
             
             # Let's try matching via 'expiry_key' if present in positions, else 'expiry_date'
             # Note: positions_df comes from 'open_positions_enriched' which might have 'expiry_key' added?
             # If not, we have to rely on 'expiry_date' string matching.
             
             # Safe fallback: normalized string comparison on expiry_date
             target_expiry_date = group["expiry_date"].dropna().unique()
             if len(target_expiry_date) > 0:
                 tgt = pd.to_datetime(target_expiry_date[0])
                 # Filter positions
                 mask = pd.to_datetime(positions_df["expiry_date"], errors="coerce") == tgt
                 existing_sub = positions_df[mask].copy()
                 if not existing_sub.empty:
                     # Adapt columns to match 'group' for _select_expiry_candidates
                     # We need: strike, side, score (mocked as infinity so they are kept?)
                     # Actually, _select_expiry_candidates sorts by score.
                     # We want to KEEP existing positions and validate NEW ones against them.
                     existing_sub["score"] = 9999.0 # High score to ensure they are 'picked' locally
                     existing_sub["is_existing"] = True
                     existing_sub["expiry_key"] = expiry
                     existing_for_expiry = existing_sub

        # 2. Combine existing + candidates
        if not existing_for_expiry.empty:
             # Ensure columns match
             common_cols = ["expiry_key", "strike", "side", "score"]
             # Add missing cols to existing
             for c in common_cols:
                 if c not in existing_for_expiry.columns:
                     if c == "score": existing_for_expiry[c] = 9999.0
                     else: existing_for_expiry[c] = None # Should not happen for core keys
             
             group["is_existing"] = False
             
             # We perform selection on the UNION
             combined = pd.concat([
                 existing_for_expiry[common_cols + ["is_existing"]], 
                 group[common_cols + ["is_existing"]]
             ], ignore_index=True)
             
             # _select_expiry_candidates returns indices of the passed dataframe.
             # This is tricky because indices are reset in concat.
             # We need to map back to original 'group' indices.
             # Strategy: Use a temp ID?
             combined["_orig_idx"] = combined.index # Local index
             # Wait, clearer way:
             # We just need to know which 'group' rows survive the consistency check against 'existing'.
             # _select_expiry_candidates logic is: "Return selected indices... obeying strict monotonicity".
             
             # If existing positions violate monotonicity among themselves, we can't fix that here.
             # We assume existing are fixed. We want to find a subset of 'group' that, when added to 'existing',
             # maintains consistency.
             
             # Actually, _select logic is: "sort by score, keep adding if consistent".
             # If we give 'existing' super high scores, they are added first.
             # Then 'group' candidates are tried. If they contradict 'existing', they are skipped.
             # This is EXACTLY what we want.
             
             # We need to track which rows came from 'group'.
             group["_temp_id"] = range(len(group))
             existing_for_expiry["_temp_id"] = -1
             
             combined = pd.concat([existing_for_expiry, group], ignore_index=True)
             
             # Run selection
             valid_indices, _ = _select_expiry_candidates(combined, max_bets_per_expiry)
             
             # Extract only the ones that came from 'group'
             valid_rows = combined.loc[valid_indices]
             selected_from_group = valid_rows[valid_rows["_temp_id"] != -1]
             
             # Get the original indices from 'group' using _temp_id
             # group.iloc[k] where k is in selected_from_group["_temp_id"]
             indices = group.index[selected_from_group["_temp_id"]].tolist()
             
        else:
            # No existing positions, standard logic
            indices, direction = _select_expiry_candidates(group, max_bets_per_expiry)

        # Apply strict limit on count (already done inside _select but double check)
        if len(indices) > max_bets_per_expiry:
             indices = indices[:max_bets_per_expiry]
        selected_indices.extend(indices)
        
    if not selected_indices:
        return []
        
    # Re-build candidates from selected indices
    candidates_df = candidates_df.loc[selected_indices].copy()
    
    final_direction_map = (
        candidates_df.groupby("expiry_key", observed=True)["side"]
        .apply(lambda col: _label_direction_from_sides(col.tolist()))
        .to_dict()
    )
    candidates_df["expiry_shape_label"] = candidates_df["expiry_key"].map(final_direction_map).fillna("none")
    debug_df["expiry_shape_label"] = debug_df["expiry_key"].map(final_direction_map).fillna("none")
    
    selected_rows = [] # Kept for compatibility if used later, but we use candidates_df directly now


    # Per-expiry scaling
    for expiry, group in candidates_df.groupby("expiry_key", observed=True):
        total_frac = group["kelly_eff"].sum()
        cap = max_capital_per_expiry_frac
        if total_frac > cap and total_frac > 0:
            scale = cap / total_frac
            candidates_df.loc[group.index, "kelly_eff"] *= scale

    total_frac = candidates_df["kelly_eff"].sum()
    if total_frac > max_capital_total_frac and total_frac > 0:
        scale = max_capital_total_frac / total_frac
        candidates_df["kelly_eff"] *= scale

    # --- Max Net Delta Logic ---
    # 1. Compute existing net delta from positions
    existing_delta = 0.0
    if positions_df is not None and not positions_df.empty and bankroll > 0:
        for _, pos in positions_df.iterrows():
            side = str(pos.get("side", "YES")).upper()
            entry = float(pos.get("entry_price", 0))
            shares = float(pos.get("size_shares", 0))
            cost = entry * shares
            # Long YES = +1 directional, Long NO = -1 directional
            if side == "YES":
                existing_delta += cost / bankroll
            else:
                existing_delta -= cost / bankroll

    # 2. Iteratively approve trades, respecting delta limits
    #    We already sorted candidates by score (descending).
    #    We check if adding the trade keeps Net Delta within [-max_net_delta_frac, +max_net_delta_frac].
    
    current_delta = existing_delta
    accepted_mask = []

    # Only apply if max_net_delta_frac is set (e.g. 0.2 means +/- 20% directional skew allowed)
    limit_active = (max_net_delta_frac is not None and max_net_delta_frac > 0)

    # We modify candidates_df in place (columns "kelly_eff" or drop rows)?
    # Better to zero out "kelly_eff" if rejected.
    
    final_eff_list = []
    
    for _, row in candidates_df.iterrows():
        eff = float(row["kelly_eff"])
        if eff <= 0:
            final_eff_list.append(0.0)
            continue
            
        if not limit_active:
            final_eff_list.append(eff)
            continue

        side = str(row["side"]).upper()
        # Contribution to delta:
        # If YES: +eff
        # If NO: -eff
        trade_delta_impact = eff if side == "YES" else -eff
        
        # Scenario: fully accept
        new_delta = current_delta + trade_delta_impact
        
        # Check if inside bounds
        # Bounds: [-max_net_delta_frac, +max_net_delta_frac]
        if abs(new_delta) <= max_net_delta_frac:
            # Safe to add
            current_delta = new_delta
            final_eff_list.append(eff)
        else:
            # Try to cap it?
            # If we are exceeding the limit, can we just fill 'up to' the limit?
            # Yes, standard portfolio practice.
            
            # Case A: We are pushing positive (Long) too high
            if new_delta > max_net_delta_frac:
                # Room remaining in positive direction
                room = max_net_delta_frac - current_delta
                # If room is negative (already over limit), we can't buy any YES.
                # However, if side is NO (negative impact), that would REDUCE delta, so we should allow it!
                if trade_delta_impact > 0: # Trying to go longer
                     allowed = max(0.0, room)
                     # Cap 'eff' at 'allowed'
                     actual_fill = min(eff, allowed)
                     current_delta += actual_fill
                     final_eff_list.append(actual_fill)
                else: # Trying to go short (NO)
                    # Going short REDUCES our long delta, so it moves us towards safety. Allow full.
                    current_delta += trade_delta_impact
                    final_eff_list.append(eff)

            # Case B: We are pushing negative (Short) too deep
            elif new_delta < -max_net_delta_frac:
                 # Room remaining in negative direction
                 # e.g. Limit -0.2, Current -0.15. Room = -0.05.
                 room = -max_net_delta_frac - current_delta # e.g. -0.2 - (-0.15) = -0.05
                 
                 if trade_delta_impact < 0: # Trying to go shorter
                     # trade_impact is negative. Room is negative.
                     # We want effective fill magnitude.
                     # allowed magnitude = abs(room)
                     allowed = max(0.0, abs(room))
                     actual_fill = min(eff, allowed)
                     # Actual delta change is -actual_fill
                     current_delta -= actual_fill
                     final_eff_list.append(actual_fill)
                 else: # Trying to go Long (YES)
                     # Going long increases delta from super-negative back up. Allow full.
                     current_delta += trade_delta_impact
                     final_eff_list.append(eff)
            else:
                # Should be covered by first check, but safe fallback
                final_eff_list.append(eff)

    if limit_active:
        candidates_df["kelly_eff"] = final_eff_list

    # Calculate stake: fixed amount or Kelly-proportional
    if use_fixed_stake:
        candidates_df["raw_stake"] = fixed_stake_amount
    else:
        candidates_df["raw_stake"] = bankroll * candidates_df["kelly_eff"]
    candidates_df = candidates_df[candidates_df["raw_stake"] >= min_trade_usd]
    if candidates_df.empty:
        LAST_RECO_DEBUG = debug_df
        return []

    candidates_df["ev_per_dollar"] = candidates_df["edge"]
    expiry_capital = defaultdict(float)
    selected: List[TradeRecommendation] = []
    for _, row in candidates_df.iterrows():
        stake = float(row["raw_stake"])
        expiry = row["expiry_key"]
        expiry_capital[expiry] += stake
        expected_value_dollars = float(row["ev_per_dollar"]) * stake
        recommendation = TradeRecommendation(
            slug=str(row.get("slug", "")),
            question=str(row.get("question", "")),
            expiry_key=str(expiry),
            strike=float(row.get("strike", np.nan)),
            side=str(row["side"]),
            pricing_date=row.get("pricing_date", pd.NaT),
            market_price=float(row["market_price"]),
            entry_price=float(row["entry_price"]),
            model_prob=float(row["model_prob"]),
            rn_prob=(float(row["rn_prob"]) if not pd.isna(row["rn_prob"]) else None),
            edge=float(row["edge"]),
            kelly_fraction_full=float(row["kelly_full"]),
            kelly_fraction_full_effective=float(row["kelly_full_effective"]),
            kelly_fraction_target=float(row["kelly_target"]),
            kelly_fraction_existing=float(row["kelly_existing"]),
            kelly_fraction_applied=float(row["kelly_eff"]),
            suggested_stake=stake,
            expected_value_per_contract=float(row["ev_per_dollar"]),
            expected_value_dollars=float(expected_value_dollars),
            expiry_group_risk=0.0,
            stability_penalty=float(row.get("stability_penalty", 1.0)),
            stale_mult=float(row.get("stale_mult", 1.0)),
            price_staleness_mult=float(row.get("price_staleness_mult", 1.0)),
            batch_age_hours=float(row["batch_age_hours"]) if row.get("batch_age_hours") is not None else None,
            price_age_hours=float(row["price_age_hours"]) if not pd.isna(row.get("price_age_hours")) else None,
            expiry_shape_label=str(row.get("expiry_shape_label", "none")),
            direction=str(row.get("direction", row["side"])),
            corr_multiplier=float(row.get("corr_multiplier", 1.0)),
            notes="ok",
        )
        selected.append(recommendation)

    if not selected:
        return []

    for rec in selected:
        rec.expiry_group_risk = expiry_capital[rec.expiry_key] / bankroll
    LAST_RECO_DEBUG = debug_df
    return selected


def recommendations_to_dataframe(recs: List[TradeRecommendation]) -> pd.DataFrame:
    """Convert recommendations to a DataFrame for display."""
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(rec) for rec in recs])
    df["stake_dollars"] = df["suggested_stake"]
    df["ev_dollars"] = df["expected_value_dollars"]
    return df
def _parse_timestamp(value: object) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _compute_stale_multiplier(
    timestamp: Optional[datetime],
    soft_hours: float,
    hard_hours: float,
) -> Tuple[float, Optional[float]]:
    if timestamp is None:
        return 1.0, None
    now = datetime.now(timezone.utc)
    age_hours = (now - timestamp).total_seconds() / 3600.0
    if age_hours <= soft_hours:
        return 1.0, age_hours
    if age_hours >= hard_hours:
        return 0.0, age_hours
    t = (age_hours - soft_hours) / (hard_hours - soft_hours)
    return max(0.0, 1.0 - t), age_hours


def _side_to_sign(side: str) -> int:
    return 1 if str(side).upper() == "YES" else -1


def _sign_changes(signs: List[int]) -> int:
    changes = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            changes += 1
            if changes > 1:
                return changes
    return changes


def _label_direction_from_sides(sides: Sequence[str]) -> str:
    signs = [_side_to_sign(side) for side in sides]
    if not signs:
        return "none"
    if all(s == 1 for s in signs):
        return "YES-only"
    if all(s == -1 for s in signs):
        return "NO-only"
    if signs[0] == 1 and signs[-1] == -1:
        return "YES-then-NO"
    if signs[0] == -1 and signs[-1] == 1:
        return "NO-then-YES"
    return "mixed"


def _select_expiry_candidates(group: pd.DataFrame, max_bets: int = 3) -> Tuple[List[int], str]:
    """Return selected indices and direction label obeying ≤max_bets strikes & ≤1 sign change."""
    if group.empty:
        return [], "none"
    sorted_by_score = group.reindex(group["score"].abs().sort_values(ascending=False).index)
    kept: List[Tuple[int, pd.Series]] = []
    for idx, row in sorted_by_score.iterrows():
        candidate = kept + [(idx, row)]
        candidate_sorted = sorted(candidate, key=lambda item: float(item[1]["strike"]))
        if len(candidate_sorted) > max_bets:
            continue
        signs = [_side_to_sign(item[1]["side"]) for item in candidate_sorted]
        changes = _sign_changes(signs)
        if changes > 1:
            continue
        if changes == 1 and not (signs[0] == 1 and signs[-1] == -1):
            # Only allow YES -> NO range structures (low strike YES, high strike NO)
            # NO -> YES is illogical (betting against low, for high) for cumulative distributions
            continue
        kept = candidate_sorted

    if not kept:
        return [], "none"

    signs = [_side_to_sign(item[1]["side"]) for item in kept]
    if all(s == 1 for s in signs):
        direction = "YES-only"
    elif all(s == -1 for s in signs):
        direction = "NO-only"
    elif signs[0] == 1 and signs[-1] == -1:
        direction = "YES-then-NO"
    elif signs[0] == -1 and signs[-1] == 1:
        direction = "NO-then-YES"
    else:
        direction = "mixed"

    selected_indices = [idx for idx, _ in kept]
    return selected_indices, direction


def main():
    """CLI interface for generating trade recommendations."""
    parser = argparse.ArgumentParser(
        description="Generate Kelly-style trade recommendations from fitted batch data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python auto_reco.py --bankroll 1000
    python auto_reco.py --bankroll 5000 --min-edge 0.08 --kelly-fraction 0.10
        """
    )
    
    parser.add_argument("--bankroll", type=float, required=True, 
                        help="Total bankroll in USD")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to batch_with_fits.csv. If not provided, uses latest from fitted_batch_results/")
    parser.add_argument("--min-edge", type=float, default=0.06,
                        help="Minimum edge threshold (default: 0.06)")
    parser.add_argument("--kelly-fraction", type=float, default=0.15,
                        help="Kelly fraction multiplier (default: 0.15)")
    parser.add_argument("--max-bets-per-expiry", type=int, default=3,
                        help="Maximum bets per expiry date (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path for recommendations (optional)")
    
    args = parser.parse_args()
    
    # Load data
    if args.input:
        df = pd.read_csv(args.input)
        print(f"Loaded: {args.input}")
    else:
        df = load_latest_fitted_batch()
        if df is None:
            print("Error: No fitted batch data found. Run fit_probability_curves.py first.")
            return
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Generate recommendations
    recommendations = recommend_trades(
        df=df,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        min_edge=args.min_edge,
        max_bets_per_expiry=args.max_bets_per_expiry,
    )
    
    if not recommendations:
        print("\nNo trade recommendations found with current parameters.")
        return
    
    # Convert to DataFrame and display
    reco_df = recommendations_to_dataframe(recommendations)
    
    print(f"\n{'='*60}")
    print(f"TRADE RECOMMENDATIONS (Bankroll: ${args.bankroll:,.2f})")
    print(f"{'='*60}")
    
    # Display key columns
    display_cols = [
        "expiry_key", "strike", "side", "market_price", "model_prob", 
        "edge", "kelly_fraction_applied", "suggested_stake", "expected_value_dollars"
    ]
    available_cols = [c for c in display_cols if c in reco_df.columns]
    print(reco_df[available_cols].to_string(index=False))
    
    # Summary stats
    total_stake = reco_df["suggested_stake"].sum()
    total_ev = reco_df["expected_value_dollars"].sum()
    print(f"\n{'='*60}")
    print(f"Total Suggested Stake: ${total_stake:,.2f} ({total_stake/args.bankroll*100:.1f}% of bankroll)")
    print(f"Total Expected Value:  ${total_ev:,.2f}")
    print(f"{'='*60}")
    
    # Save if requested
    if args.output:
        reco_df.to_csv(args.output, index=False)
        print(f"\nSaved recommendations to: {args.output}")


if __name__ == "__main__":
    main()
