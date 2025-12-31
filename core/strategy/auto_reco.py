#!/usr/bin/env python3
"""
auto_reco.py

Generate Kelly-style trade recommendations from batch_with_fits data.
Refactored to a 3-stage pipeline: Target -> Delta -> Action.

Enhancements:
- Directional consistency per expiry (YES-only or NO-only).
- Batch staleness multiplier (reduces sizing for stale data).
- Exit hysteresis to prevent churn.
- Rank-and-fill portfolio allocation.
- Risk-off and vol gate enforcement.
- Probability threshold mode.
"""

from __future__ import annotations

import argparse
import glob
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from core.strategy.vol_gate import VolGateResult, compute_vol_gate

# Configure logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants & Defaults
# -----------------------------------------------------------------------------

from core.strategy.common import (
    MAX_CAP_PER_EXPIRY_FRAC_DEFAULT,
    MAX_CAP_TOTAL_FRAC_DEFAULT,
    STALE_SOFT_HOURS,
    STALE_HARD_HOURS,
    DEFAULT_MIN_TRADE_USD,
    DEFAULT_REBALANCE_MIN_ADD_USD,
    DEFAULT_REBALANCE_MIN_REDUCE_USD,
    DEFAULT_EXIT_HYSTERESIS,
    TargetRole,
    TargetPosition,
    DeltaIntent,
    TradeRecommendation,
    RebalanceConfig,
)

# Debug state
LAST_RECO_DEBUG: Optional[pd.DataFrame] = None
LAST_RECO_THRESHOLD_DEBUG: Optional[Dict[str, int]] = None


def reset_threshold_debug() -> None:
    """Reset the threshold debug accumulator."""
    global LAST_RECO_THRESHOLD_DEBUG
    LAST_RECO_THRESHOLD_DEBUG = None


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_btc_csv(path: str = "DATA/btc_intraday_1m.csv") -> Optional[pd.DataFrame]:
    """
    Load BTC price data for volatility gate computation.
    
    Parses timestamps as UTC and ensures 'close' is float.
    """
    p = Path(path)
    if not p.exists():
        # Try alternate locations
        alt_paths = [
            Path("../DATA/btc_intraday_1m.csv"),
            Path("data/btc_intraday_1m.csv"),
        ]
        for alt in alt_paths:
            if alt.exists():
                p = alt
                break
    
    if not p.exists():
        logger.warning(f"BTC data not found at {path}")
        return None
    
    df = pd.read_csv(p)
    
    # Parse timestamp column to UTC datetime
    ts_col = None
    for col in ["timestamp", "time", "datetime", "date"]:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.sort_values(ts_col)
    
    # Ensure close is float
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    
    return df


def load_latest_fitted_batch(directory: str = "fitted_batch_results") -> Optional[pd.DataFrame]:
    """Load the most recent batch_with_fits.csv."""
    pattern = Path(directory) / "*" / "batch_with_fits.csv"
    files = glob.glob(str(pattern))
    
    if not files:
        pattern_flat = Path(directory) / "batch_with_fits.csv"
        files = glob.glob(str(pattern_flat))
    
    if not files:
        return None
    
    latest = max(files, key=lambda f: Path(f).stat().st_mtime)
    logger.info(f"Loading fitted batch from: {latest}")
    return pd.read_csv(latest)


def generate_key(
    row: pd.Series,
    side: Optional[str] = None,
) -> str:
    """
    Generate a stable unique key for a position/target.
    
    Args:
        row: DataFrame row with market data
        side: Explicit side (YES/NO). If None, uses row['side'].
    
    Priority:
    1. (condition_id, side) if condition_id present
    2. (slug, expiry_key, strike, side) as fallback
    """
    # Use explicit side if provided, otherwise fall back to row
    effective_side = side if side is not None else str(row.get("side", "YES"))
    
    condition_id = row.get("condition_id")
    if condition_id and pd.notna(condition_id):
        return f"{condition_id}|{effective_side}"
    
    slug = str(row.get("slug", "unknown"))
    expiry = str(row.get("expiry_key", row.get("expiry_date", "unknown")))
    strike = float(row.get("strike", 0))
    
    return f"{slug}|{expiry}|{strike:.2f}|{effective_side}"


def compute_current_exposure_mtm(
    positions_df: Optional[pd.DataFrame],
    bankroll: float
) -> Dict[str, float]:
    """
    Compute current exposure in USD (Mark-To-Market) per position key.
    
    Returns:
        Dict mapping key -> current_mtm_value (clamped >= 0)
    """
    if positions_df is None or positions_df.empty:
        return {}
    
    exposure: Dict[str, float] = defaultdict(float)
    
    for _, pos in positions_df.iterrows():
        key = generate_key(pos)
        # Use market price for exposure (Market Value)
        # Fallback to entry_price only if market_price missing/zero
        price = float(pos.get("market_price", 0))
        if price <= 0:
             price = float(pos.get("entry_price", 0))
             
        size_shares = float(pos.get("size_shares", 0))
        
        if price <= 0 or size_shares <= 0:
            continue
        
        market_value = price * size_shares
        
        # If sells are tracked, subtract proceeds
        # For now: sell_proceeds = 0 (long-only)
        sell_proceeds = 0.0
        
        net = market_value - sell_proceeds
        exposure[key] = max(0.0, exposure[key] + net)
    
    return dict(exposure)


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


def kelly_fraction_yes(p: float, q: float) -> float:
    """Kelly fraction for YES side: f* = (p - q) / (1 - q)"""
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return max((p - q) / (1.0 - q), 0.0)


def kelly_fraction_no(p: float, q: float) -> float:
    """Kelly fraction for NO side: f* = (q - p) / q"""
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return max((q - p) / q, 0.0)

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


def _select_expiry_candidates(
    group: pd.DataFrame, 
    max_bets: int = 3
) -> Tuple[List[int], str]:
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


# -----------------------------------------------------------------------------
# Stage 1: Build Targets
# -----------------------------------------------------------------------------

def build_targets(
    batch_df: pd.DataFrame,
    current_exposure: Dict[str, float],
    vol_gate_result: VolGateResult,
    config: RebalanceConfig,
    now_utc: Optional[datetime] = None,
) -> Dict[str, TargetPosition]:
    """
    Stage 1: Compute target exposures for each position.
    
    Returns:
        Dict mapping key -> TargetPosition
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    
    targets: Dict[str, TargetPosition] = {}
    
    if batch_df is None or batch_df.empty:
        # Handle held positions with no batch data
        for key, cur_usd in current_exposure.items():
            if cur_usd > 0:
                # Parse key to get correct side (key format: condition_id|SIDE or slug|expiry|strike|SIDE)
                parts = key.split("|")
                if len(parts) >= 2:
                    parsed_side = parts[-1] if parts[-1] in ("YES", "NO") else "YES"
                    parsed_slug = parts[0]
                    parsed_condition_id = parts[0] if len(parts) == 2 else None
                else:
                    parsed_side = "YES"
                    parsed_slug = key
                    parsed_condition_id = None
                
                target_usd = cur_usd if config.missing_target_policy == "KEEP" else 0.0
                role = TargetRole.HOLD_SAFETY if target_usd > 0 else TargetRole.EXIT
                
                targets[key] = TargetPosition(
                    key=key,
                    slug=parsed_slug,
                    side=parsed_side,
                    expiry_key="unknown",
                    strike=0.0,
                    condition_id=parsed_condition_id,
                    target_fraction=target_usd / config.bankroll if config.bankroll > 0 else 0.0,
                    target_usd=target_usd,
                    model_prob=0.5,
                    market_price=0.5,
                    entry_price=0.5,
                    exit_price=0.5,
                    effective_edge=0.0,
                    allocation_score=0.0,
                    exit_score=0.0,
                    role=role,
                    is_fallback_price=False,
                )
        return targets
    
    # Prepare data
    data = batch_df.copy()
    data["expiry_key"] = _derive_expiry_key(data)
    
    price_col = _pick_column(data, ["market_price", "market_pr", "Polymarket_Price"])
    model_col = _pick_column(data, ["p_model_cal", "p_model_fit", "p_real_mc", "model_probability", "Model_Prob"])
    
    if price_col is None or model_col is None:
        logger.error("Batch file missing market_price or model probability columns")
        return targets
    
    data["market_price"] = pd.to_numeric(data[price_col], errors="coerce")
    data["model_prob"] = pd.to_numeric(data[model_col], errors="coerce")
    data = data.dropna(subset=["market_price", "model_prob"])
    
    # Compute required edge thresholds
    required_edge_entry = config.min_edge_entry + (vol_gate_result.edge_add_cents / 100.0)
    required_edge_exit = config.min_edge_exit + (vol_gate_result.edge_add_cents / 100.0)
    
    # Filters
    if config.require_active:
        mask = pd.Series(True, index=data.index)
        if "active" in data.columns:
            mask &= data["active"].astype(bool)
        if "closed" in data.columns:
            mask &= ~data["closed"].astype(bool)
        if "archived" in data.columns:
            mask &= ~data["archived"].astype(bool)
        data = data[mask]
    
    data = data[data["model_prob"].between(config.min_model_prob, config.max_model_prob)]
    
    # Stability penalty
    if config.use_stability_penalty:
        data["stability_penalty"] = _compute_stability_penalty(data)
    else:
        data["stability_penalty"] = 1.0
    
    # Staleness
    batch_ts_candidates = ["batch_timestamp", "run_timestamp", "pricing_timestamp", "live_price_as_of"]
    batch_timestamp = None
    for col in batch_ts_candidates:
        if col in data.columns:
            ts_value = data[col].dropna().iloc[0] if not data[col].dropna().empty else None
            batch_timestamp = _parse_timestamp(ts_value)
            if batch_timestamp:
                break
    
    stale_mult, batch_age_hours = _compute_stale_multiplier(batch_timestamp, STALE_SOFT_HOURS, STALE_HARD_HOURS)
    if config.disable_staleness:
        stale_mult = 1.0
    elif stale_mult == 0.0:
        # Hard-block new entries for extremely stale data; still allow exits/reductions
        if vol_gate_result.allow_new_entries:
            # Use dataclasses.replace for immutability safety
            from dataclasses import replace
            vol_gate_result = replace(
                vol_gate_result,
                allow_new_entries=False,
                reason=f"batch too stale (age_hours={batch_age_hours:.2f})"
            )
            logger.warning(f"[StaleHardBlock] Blocking new entries: batch age {batch_age_hours:.2f}h > {STALE_HARD_HOURS}h")
    
    # Process each row for candidate targets
    candidates: List[Dict[str, Any]] = []
    
    for _, row in data.iterrows():
        p = float(row["model_prob"])
        q = float(row["market_price"])
        stability = float(row["stability_penalty"])
        
        # YES side - generate key WITH side
        yes_key = generate_key(row, side="YES")
        yes_cur_usd = current_exposure.get(yes_key, 0.0)
        yes_is_held = yes_cur_usd > 0
        
        yes_price = q
        yes_is_fallback = True
        if "yes_ask_price" in row.index and pd.notna(row.get("yes_ask_price")):
            yes_price = float(row["yes_ask_price"])
            yes_is_fallback = False
        yes_effective_edge = (p - yes_price) - config.spread_cost
        
        # YES Exit Price (Bid)
        yes_exit = q # Default to mid if missing
        if "yes_bid_price" in row.index and pd.notna(row.get("yes_bid_price")):
            yes_exit = float(row["yes_bid_price"])
        
        if config.min_price <= yes_price <= config.max_price:
            candidates.append({
                "key": yes_key,
                "slug": str(row.get("slug", "")),
                "question": str(row.get("question", "")),
                "expiry_key": str(row["expiry_key"]),
                "strike": float(row.get("strike", np.nan)),
                "condition_id": row.get("condition_id"),
                "model_prob": p,
                "stability_penalty": stability,
                "stale_mult": stale_mult,
                "batch_age_hours": batch_age_hours,
                "current_usd": yes_cur_usd,
                "is_held": yes_is_held,
                "side": "YES",
                "entry_price": yes_price,
                "exit_price": yes_exit,
                "market_price": yes_price,
                "effective_edge": yes_effective_edge,
                "market_price": yes_price,
                "effective_edge": yes_effective_edge,
                "is_fallback_price": yes_is_fallback,
            })
        
        # NO side - generate key WITH side
        if config.allow_no:
            no_key = generate_key(row, side="NO")
            no_cur_usd = current_exposure.get(no_key, 0.0)
            no_is_held = no_cur_usd > 0
            
            no_price = 1.0 - q
            no_is_fallback = True
            if "no_ask_price" in row.index and pd.notna(row.get("no_ask_price")):
                no_price = float(row["no_ask_price"])
                no_is_fallback = False
            no_effective_edge = ((1.0 - p) - no_price) - config.spread_cost
            
            # NO Exit Price (Bid)
            no_exit = 1.0 - q
            if "no_bid_price" in row.index and pd.notna(row.get("no_bid_price")):
                no_exit = float(row["no_bid_price"])
            
            if config.min_price <= no_price <= config.max_price:
                candidates.append({
                    "key": no_key,
                    "slug": str(row.get("slug", "")),
                    "question": str(row.get("question", "")),
                    "expiry_key": str(row["expiry_key"]),
                    "strike": float(row.get("strike", np.nan)),
                    "condition_id": row.get("condition_id"),
                    "model_prob": p,
                    "stability_penalty": stability,
                    "stale_mult": stale_mult,
                    "batch_age_hours": batch_age_hours,
                    "current_usd": no_cur_usd,
                    "is_held": no_is_held,
                    "side": "NO",
                    "entry_price": no_price,
                    "exit_price": no_exit,
                    "market_price": no_price,
                    "effective_edge": no_effective_edge,
                    "market_price": no_price,
                    "effective_edge": no_effective_edge,
                    "is_fallback_price": no_is_fallback,
                })
    
    if not candidates:
        return targets
    
    df_cand = pd.DataFrame(candidates)
    
    # Apply probability threshold filter if enabled
    if config.use_prob_threshold:
        # YES: keep if model_prob >= threshold OR already held
        yes_mask = (
            (df_cand["side"] == "YES") &
            ((df_cand["model_prob"] >= config.prob_threshold_yes) | df_cand["is_held"])
        )
        # NO: keep if model_prob <= threshold (i.e., high NO prob) OR already held
        no_mask = (
            (df_cand["side"] == "NO") &
            ((df_cand["model_prob"] <= config.prob_threshold_no) | df_cand["is_held"])
        )
        df_cand = df_cand[yes_mask | no_mask]
        
        if df_cand.empty:
            logger.info("[ProbThreshold] No candidates pass thresholds")
            return targets
    
    # Compute Kelly and scores for each candidate
    for idx in df_cand.index:
        row = df_cand.loc[idx]
        p = row["model_prob"]
        q = row["entry_price"]
        side = row["side"]
        eff_edge = row["effective_edge"]
        is_held = row["is_held"]
        key = row["key"]
        stability = row["stability_penalty"]
        
        # Kelly calculation - use entry_price consistently for both sides
        entry_p = row["entry_price"]
        if side == "YES":
            # Buying YES: Win if YES (prob p), Cost = entry_p
            p_win = p
            cost = entry_p
        else:
            # Buying NO: Win if NO (prob 1-p), Cost = entry_p
            p_win = 1.0 - p
            cost = entry_p
        
        # Standard Kelly: f = (p_win - cost) / (1 - cost)
        # (Assuming payout is 1 unit, profit is 1-cost, loss is cost)
        if 1.0 - cost > 1e-9:
            f_star = max(0.0, (p_win - cost) / (1.0 - cost))
        else:
            f_star = 0.0
        
        # Apply vol gate kelly multiplier
        f_star_scaled = f_star * vol_gate_result.kelly_mult * stability * stale_mult
        
        # Target fraction (capped)
        f_target = min(config.kelly_fraction * f_star_scaled, 0.30)
        
        # Entry eligibility check
        # STRICT GATE: If new entries disallowed (e.g. stale), force fail
        if not vol_gate_result.allow_new_entries:
            passes_entry = False
        else:
            passes_entry = eff_edge >= required_edge_entry
        passes_exit = eff_edge >= required_edge_exit
        
        # Determine target and role
        if passes_entry:
            target_frac = f_target
            role_str = TargetRole.ENTRY.value
        elif is_held and passes_exit:
            # Exit hysteresis: hold at current
            target_frac = current_exposure.get(key, 0.0) / config.bankroll
            role_str = TargetRole.HOLD_SAFETY.value
        elif is_held and not passes_exit:
            # Exit signal - never filtered
            target_frac = 0.0
            role_str = TargetRole.EXIT.value
        else:
            # Not held, doesn't pass entry - will be filtered by consistency
            target_frac = 0.0
            role_str = TargetRole.ENTRY.value
        
        df_cand.loc[idx, "target_fraction"] = target_frac
        df_cand.loc[idx, "target_usd"] = target_frac * config.bankroll
        df_cand.loc[idx, "allocation_score"] = eff_edge * vol_gate_result.kelly_mult if target_frac > 0 else 0.0
        df_cand.loc[idx, "exit_score"] = eff_edge
        df_cand.loc[idx, "kelly_full"] = f_star
        df_cand.loc[idx, "kelly_mult_applied"] = vol_gate_result.kelly_mult
        df_cand.loc[idx, "kelly_mult_applied"] = vol_gate_result.kelly_mult
        df_cand.loc[idx, "role"] = role_str
    
    # Expiry selection (consistency constraints) - apply ONLY to entry candidates
    df_cand["score"] = df_cand["allocation_score"]
    
    selected_indices: List[int] = []
    for expiry, group in df_cand.groupby("expiry_key", observed=True):
        # Split by role
        entry_candidates = group[
            (group["role"] == TargetRole.ENTRY.value) & 
            (group["target_usd"] > 0)
        ]
        exit_candidates = group[group["role"] == TargetRole.EXIT.value]
        holds = group[group["role"] == TargetRole.HOLD_SAFETY.value]
        
        # Apply consistency constraints ONLY to entry candidates
        if not entry_candidates.empty:
            idxs, direction = _select_expiry_candidates(entry_candidates, config.max_bets_per_expiry)
            selected_indices.extend(idxs)
        
        # Always keep exit candidates (SELL signals) - don't filter them!
        selected_indices.extend(exit_candidates.index.tolist())
        
        # Always keep safety holds
        selected_indices.extend(holds.index.tolist())
    
    df_cand = df_cand.loc[selected_indices].drop_duplicates(subset=["key"])
    
    # Portfolio Allocation (Rank-and-Fill with per-expiry caps)
    cap_usd = config.bankroll * config.max_capital_total_frac
    expiry_cap_usd = config.bankroll * config.max_capital_per_expiry_frac
    
    # Compute hold budget (positions we're keeping at current or target)
    hold_keys = set()
    hold_budget = 0.0
    used_by_expiry: Dict[str, float] = defaultdict(float)
    
    for _, row in df_cand.iterrows():
        if row["role"] == TargetRole.HOLD_SAFETY.value and row["target_usd"] > 0:
            hold_budget += row["current_usd"]
            hold_keys.add(row["key"])
            used_by_expiry[row["expiry_key"]] += row["current_usd"]
    
    # Cap breach handling
    if hold_budget > cap_usd:
        logger.warning(
            f"[PortfolioCap] Hold budget (${hold_budget:.2f}) exceeds cap (${cap_usd:.2f}). "
            f"Cap breach delever={config.cap_breach_delever}"
        )
        if config.cap_breach_delever:
            # Reduce targets of lowest exit_score positions until compliant
            over_budget = hold_budget - cap_usd
            # Sort held positions by exit_score ascending (worst first)
            held_df = df_cand[df_cand["key"].isin(hold_keys)].sort_values("exit_score", ascending=True)
            delevered_keys = set()
            for idx, row in held_df.iterrows():
                if over_budget <= 0:
                    break
                # Set target to 0 for this position (force sell)
                reduction = min(row["current_usd"], over_budget)
                df_cand.loc[idx, "target_usd"] = 0.0
                df_cand.loc[idx, "target_fraction"] = 0.0
                df_cand.loc[idx, "role"] = TargetRole.EXIT.value  # Mark as sell target
                over_budget -= reduction
                delevered_keys.add(row["key"])
                logger.info(f"[CapBreachDelever] Reducing {row['key']} to target=0")
            
            # Recompute budgets after delever
            hold_keys -= delevered_keys
            hold_budget = sum(
                row["current_usd"] for _, row in df_cand.iterrows()
                if row["role"] == TargetRole.HOLD_SAFETY.value and row["target_usd"] > 0
            )
            used_by_expiry = defaultdict(float)
            for _, row in df_cand.iterrows():
                if row["role"] == TargetRole.HOLD_SAFETY.value and row["target_usd"] > 0:
                    used_by_expiry[row["expiry_key"]] += row["current_usd"]
            free_budget = max(0.0, cap_usd - hold_budget)
            logger.info(f"[CapBreachDelever] After delever: hold_budget=${hold_budget:.2f}, free_budget=${free_budget:.2f}")
        else:
            free_budget = 0.0
    else:
        free_budget = max(0.0, cap_usd - hold_budget)
    
    # Sort candidates for allocation (only entry candidates with target > current)
    alloc_candidates = df_cand[
        (df_cand["target_usd"] > df_cand["current_usd"]) &
        (df_cand["target_usd"] > 0) &
        (df_cand["role"] == TargetRole.ENTRY.value)
    ].sort_values("allocation_score", ascending=False)
    
    allocated_budget = 0.0
    final_targets: Dict[str, float] = {}
    
    for _, row in alloc_candidates.iterrows():
        key = row["key"]
        expiry = row["expiry_key"]
        desired = row["target_usd"]
        current = row["current_usd"]
        incremental = desired - current
        
        # Check portfolio-level budget
        available = free_budget - allocated_budget
        if available <= 0:
            # No more portfolio budget - hold at current
            final_targets[key] = current
            continue
        
        # Check per-expiry cap
        expiry_available = expiry_cap_usd - used_by_expiry[expiry]
        if expiry_available <= 0:
            # No more expiry budget - hold at current
            final_targets[key] = current
            continue
        
        # Take minimum of portfolio budget, expiry budget, and desired increment
        actual_increment = min(incremental, available, expiry_available)
        
        if actual_increment >= incremental:
            # Full allocation
            final_targets[key] = desired
        else:
            # Partial allocation
            final_targets[key] = current + actual_increment
        
        allocated_budget += actual_increment
        used_by_expiry[expiry] += actual_increment
    
    # Build final TargetPosition objects
    for _, row in df_cand.iterrows():
        key = row["key"]
        
        # Determine final target_usd
        # Priority: final_targets override > row's computed target_usd
        if key in final_targets:
            target_usd = final_targets[key]
        else:
            # Use computed target_usd (which may be 0 for exit signals)
            target_usd = row["target_usd"]
        
        targets[key] = TargetPosition(
            key=key,
            slug=row["slug"],
            side=row["side"],
            expiry_key=row["expiry_key"],
            strike=float(row["strike"]) if pd.notna(row["strike"]) else 0.0,
            condition_id=row.get("condition_id"),
            target_fraction=target_usd / config.bankroll if config.bankroll > 0 else 0.0,
            target_usd=target_usd,
            model_prob=row["model_prob"],
            market_price=row["market_price"],
            entry_price=row["entry_price"],
            exit_price=row.get("exit_price", row["entry_price"]),
            effective_edge=row["effective_edge"],
            allocation_score=row["allocation_score"],
            exit_score=row["exit_score"],
            role=TargetRole(row["role"]),
            kelly_full=row.get("kelly_full", 0.0),
            kelly_mult_applied=row.get("kelly_mult_applied", 1.0),
            stability_penalty=row.get("stability_penalty", 1.0),
            stale_mult=row.get("stale_mult", 1.0),
            is_fallback_price=row.get("is_fallback_price", False),
            metadata={"question": row.get("question", "")},
        )
    
    # Add safety holds for positions not in batch
    for key, cur_usd in current_exposure.items():
        if cur_usd > 0 and key not in targets:
            # Missing from batch - apply missing policy
            target_usd = cur_usd if config.missing_target_policy == "KEEP" else 0.0
            
            # Parse key to get correct side (key format: condition_id|SIDE or slug|expiry|strike|SIDE)
            parts = key.split("|")
            if len(parts) >= 2:
                # Last part is the side (YES or NO)
                parsed_side = parts[-1] if parts[-1] in ("YES", "NO") else "YES"
                # First part is condition_id or slug
                parsed_slug = parts[0]
            else:
                parsed_side = "YES"
                parsed_slug = key
            
            targets[key] = TargetPosition(
                key=key,
                slug=parsed_slug,
                side=parsed_side,
                expiry_key="unknown",
                strike=0.0,
                condition_id=parts[0] if len(parts) == 2 else None,
                target_fraction=target_usd / config.bankroll if config.bankroll > 0 else 0.0,
                target_usd=target_usd,
                model_prob=0.5,
                market_price=0.5,
                entry_price=0.5,
                exit_price=0.5,
                effective_edge=0.0,
                allocation_score=0.0,
                exit_score=-1.0,  # Low priority for exit
                role=TargetRole.HOLD_SAFETY if target_usd > 0 else TargetRole.EXIT,
                is_fallback_price=False,
            )
    
    # Debug: check key matching
    matched_keys = set(current_exposure.keys()) & set(targets.keys())
    if current_exposure and len(matched_keys) < len(current_exposure) * 0.5:
        logger.warning(
            f"[KeyMatch] Only {len(matched_keys)}/{len(current_exposure)} position keys matched targets. "
            f"Check condition_id availability in positions_df."
        )
    
    return targets


# -----------------------------------------------------------------------------
# Stage 2: Compute Deltas
# -----------------------------------------------------------------------------

def compute_deltas(
    targets: Dict[str, TargetPosition],
    current_exposure: Dict[str, float],
    vol_gate_result: VolGateResult,
    config: RebalanceConfig,
) -> List[DeltaIntent]:
    """
    Stage 2: Compute deltas between targets and current positions.
    
    Returns:
        List of DeltaIntent objects
    """
    intents: List[DeltaIntent] = []
    
    # Union of keys
    all_keys = set(targets.keys()) | set(current_exposure.keys())
    
    buy_intents: List[Tuple[float, DeltaIntent]] = []
    sell_intents: List[Tuple[float, DeltaIntent]] = []
    
    for key in all_keys:
        tgt = targets.get(key)
        cur_usd = current_exposure.get(key, 0.0)
        
        # Target resolution
        if tgt is not None:
            tgt_usd = tgt.target_usd
            model_prob = tgt.model_prob
            eff_edge = tgt.effective_edge
            slug = tgt.slug
            side = tgt.side
            expiry_key = tgt.expiry_key
            strike = tgt.strike
            condition_id = tgt.condition_id
            entry_price = tgt.entry_price
            market_price = tgt.market_price
            question = tgt.metadata.get("question", "")
            alloc_score = tgt.allocation_score
            exit_score = tgt.exit_score
            exit_price = tgt.exit_price
            is_fallback = tgt.is_fallback_price
        else:
            # Missing target (Position held but no model output)
            if config.missing_target_policy == "KEEP":
                tgt_usd = cur_usd  # Hold current exposure
            else:
                tgt_usd = 0.0  # Force exit
                
            # Sentinel values - Do NOT fabricate an executable trade
            model_prob = 0.0
            eff_edge = 0.0
            # Parse slug from key just for display, don't rely on it for execution
            slug = key.split("|")[0] if "|" in key else key
            side = "YES" 
            expiry_key = "unknown"
            strike = 0.0
            condition_id = None
            entry_price = 0.0 
            market_price = 0.0
            question = "Held position (target missing)"
            alloc_score = 0.0
            exit_score = -999.0 # Prioritize clearing if exiting
            exit_price = entry_price  # Use entry price as fallback for exit
            is_fallback = False
        
        # Risk-off override (Priority 1)
        if not vol_gate_result.allow_new_entries and config.risk_off_targets_to_zero:
            tgt_usd = 0.0
        
        raw_delta = tgt_usd - cur_usd
        
        # Vol gate entry block (Priority 2)
        if not vol_gate_result.allow_new_entries and raw_delta > 0:
            raw_delta = 0.0  # Block adds
        
        # Determine action
        if abs(raw_delta) < 0.01:  # Negligible
            action = "HOLD"
            amount = 0.0
        elif raw_delta > 0:
            # Buy
            if raw_delta < config.rebalance_min_add_usd:
                action = "HOLD"
                amount = 0.0
                raw_delta = 0.0
            else:
                action = "BUY"
                amount = raw_delta
        else:
            # Sell
            if abs(raw_delta) < config.rebalance_min_reduce_usd:
                action = "HOLD"
                amount = 0.0
                raw_delta = 0.0
            else:
                action = "SELL"
                amount = abs(raw_delta)
        
        # Determine price mode (None for HOLD)
        if action == "BUY":
            price_mode = "TAKER_ASK"
        elif action == "SELL":
            price_mode = "TAKER_BID"
        else:
            price_mode = None
        
        # Create reason
        if action == "BUY":
            reason = f"Increase exposure: ${cur_usd:.2f} -> ${tgt_usd:.2f}"
        elif action == "SELL":
            reason = f"Reduce exposure: ${cur_usd:.2f} -> ${tgt_usd:.2f}"
        else:
            reason = "No action needed"
        
        intent = DeltaIntent(
            key=key,
            intent_key=f"{key}|{action}",
            slug=slug,
            side=side,
            expiry_key=expiry_key,
            strike=strike,
            condition_id=condition_id,
            action=action,
            amount_usd=amount,
            signed_delta_usd=raw_delta,
            current_usd=cur_usd,
            target_usd=tgt_usd,
            price_mode=price_mode,
            limit_price_hint=exit_price if action == "SELL" else entry_price,
            model_prob=model_prob,
            effective_edge=eff_edge,
            reason=reason,
            question=question,
            entry_price=entry_price,
            market_price=market_price,
            suggested_stake=amount,
            expected_value_per_contract=eff_edge,
            expected_value_dollars=eff_edge * amount,
            direction=side,
            is_fallback_price=is_fallback,
        )
        
        # Store for cap processing
        if action == "BUY":
            buy_intents.append((alloc_score, intent))
        elif action == "SELL":
            sell_intents.append((exit_score, intent))
        else:
            intents.append(intent)
    
    # Apply delta caps for BUYs
    buy_intents.sort(key=lambda x: x[0], reverse=True)  # Highest alloc score first
    total_buys = sum(i[1].amount_usd for i in buy_intents)
    
    if total_buys > config.max_add_per_cycle_usd:
        remaining = config.max_add_per_cycle_usd
        for score, intent in buy_intents:
            if remaining <= 0:
                intent.action = "HOLD"
                intent.amount_usd = 0.0
                intent.signed_delta_usd = 0.0
                intent.reason = "Blocked by add cap"
            elif intent.amount_usd <= remaining:
                remaining -= intent.amount_usd
            else:
                intent.amount_usd = remaining
                intent.signed_delta_usd = remaining
                remaining = 0
    
    intents.extend(i[1] for i in buy_intents)
    
    # Apply delta caps for SELLs
    sell_intents.sort(key=lambda x: x[0])  # Lowest exit score first (worst first)
    total_sells = sum(i[1].amount_usd for i in sell_intents)
    
    if total_sells > config.max_reduce_per_cycle_usd:
        remaining = config.max_reduce_per_cycle_usd
        for score, intent in sell_intents:
            if remaining <= 0:
                intent.action = "HOLD"
                intent.amount_usd = 0.0
                intent.signed_delta_usd = 0.0
                intent.reason = "Blocked by reduce cap"
            elif intent.amount_usd <= remaining:
                remaining -= intent.amount_usd
            else:
                intent.amount_usd = remaining
                intent.signed_delta_usd = -remaining
                remaining = 0
    
    intents.extend(i[1] for i in sell_intents)
    
    return intents


# -----------------------------------------------------------------------------
# Stage 3: Public API
# -----------------------------------------------------------------------------

def recommend_trades(
    df: pd.DataFrame,
    bankroll: float,
    positions_df: Optional[pd.DataFrame] = None,
    kelly_fraction: float = 0.15,
    min_edge: float = 0.06,
    max_bets_per_expiry: int = 3,
    max_capital_per_expiry_frac: float = MAX_CAP_PER_EXPIRY_FRAC_DEFAULT,
    max_capital_total_frac: float = MAX_CAP_TOTAL_FRAC_DEFAULT,
    min_price: float = 0.03,
    max_price: float = 0.95,
    min_model_prob: float = 0.0,
    max_model_prob: float = 1.0,
    require_active: bool = True,
    use_stability_penalty: bool = True,
    allow_no: bool = True,
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
    btc_price_df: Optional[pd.DataFrame] = None,
    risk_off_targets_to_zero: bool = True,
    missing_target_policy: Literal["KEEP", "EXIT"] = "KEEP",
    return_all: bool = False,
    # Compatibility arguments to match RebalanceConfig fields
    min_edge_entry: Optional[float] = None,
    **kwargs: Any,
) -> List[DeltaIntent]:
    """
    Public API: Generate trade recommendations using the 3-stage pipeline.
    
    Maintains backwards compatibility with existing callers.
    
    Returns:
        List of DeltaIntent (compatible with TradeRecommendation)
    """
    global LAST_RECO_DEBUG
    
    if df is None or df.empty:
        LAST_RECO_DEBUG = pd.DataFrame()
        return []
    
    # Build config
    # Prefer min_edge_entry from arguments/kwargs, fallback to legacy min_edge
    final_min_edge = min_edge_entry if min_edge_entry is not None else min_edge
    
    config = RebalanceConfig(
        bankroll=bankroll,
        min_edge_entry=final_min_edge,
        min_edge_exit=final_min_edge - DEFAULT_EXIT_HYSTERESIS,
        kelly_fraction=kelly_fraction,
        use_fixed_stake=use_fixed_stake,
        fixed_stake_amount=fixed_stake_amount,
        max_capital_per_expiry_frac=max_capital_per_expiry_frac,
        max_capital_total_frac=max_capital_total_frac,
        max_bets_per_expiry=max_bets_per_expiry,
        min_price=min_price,
        max_price=max_price,
        min_model_prob=min_model_prob,
        max_model_prob=max_model_prob,
        max_dte=max_dte,
        max_moneyness=max_moneyness,
        min_moneyness=min_moneyness,
        require_active=require_active,
        use_stability_penalty=use_stability_penalty,
        disable_staleness=disable_staleness,
        allow_no=allow_no,
        min_trade_usd=min_trade_usd,
        rebalance_min_add_usd=min_trade_usd,
        rebalance_min_reduce_usd=min_trade_usd * 2,
        use_prob_threshold=use_prob_threshold,
        prob_threshold_yes=prob_threshold_yes,
        prob_threshold_no=prob_threshold_no,
        risk_off_targets_to_zero=risk_off_targets_to_zero,
        missing_target_policy=missing_target_policy,
        # Extract cycle caps from kwargs if present (defaults to 100k/200k in RebalanceConfig)
        max_add_per_cycle_usd=kwargs.get("max_add_per_cycle_usd", 100000.0),
        max_reduce_per_cycle_usd=kwargs.get("max_reduce_per_cycle_usd", 200000.0),
    )
    
    # Load BTC data for vol gate
    if btc_price_df is None:
        btc_price_df = load_btc_csv()
    
    # Compute vol gate
    now_utc = datetime.now(timezone.utc)
    if btc_price_df is not None and not btc_price_df.empty:
        vol_gate_result = compute_vol_gate(btc_price_df, now_utc)
        logger.info(f"[VolGate] Regime={vol_gate_result.regime} Reason={vol_gate_result.reason}")
    else:
        # Fallback: normal regime
        vol_gate_result = VolGateResult(
            now_utc=now_utc.isoformat(),
            regime="unknown",
            vol15=None,
            vol60=None,
            vol15_pct=None,
            shock=False,
            allow_new_entries=True,
            edge_add_cents=0.0,
            kelly_mult=1.0,
            reason="No BTC data available"
        )
        logger.warning("[VolGate] No BTC data - using fallback normal regime")
    
    # Compute current exposure
    combined_positions = positions_df
    if current_open_positions is not None and not current_open_positions.empty:
        if combined_positions is not None and not combined_positions.empty:
            combined_positions = pd.concat([combined_positions, current_open_positions], ignore_index=True)
        else:
            combined_positions = current_open_positions
    
    current_exposure = compute_current_exposure_mtm(combined_positions, bankroll)
    
    # Stage 1: Build Targets
    targets = build_targets(
        batch_df=df,
        current_exposure=current_exposure,
        vol_gate_result=vol_gate_result,
        config=config,
        now_utc=now_utc,
    )
    
    # Stage 2: Compute Deltas
    intents = compute_deltas(
        targets=targets,
        current_exposure=current_exposure,
        vol_gate_result=vol_gate_result,
        config=config,
    )
    
    # Filter to only actionable intents unless return_all is True
    if return_all:
        actionable = intents
    else:
        actionable = [i for i in intents if i.action != "HOLD" and i.amount_usd >= min_trade_usd]
    
    # Populate legacy fields for backwards compatibility
    # Populate legacy fields / Safe Mapping
    for intent in actionable:
        t = targets.get(intent.key)
        if not t:
            continue

        # Safely map attributes using metadata or defaults
        intent.question = t.metadata.get("question", "") if hasattr(t, "metadata") else ""
        
        # Ensure prices are float
        intent.entry_price = float(t.entry_price or 0.0)
        intent.market_price = float(t.market_price or 0.0)

        # Consistent hint
        intent.limit_price_hint = float(t.entry_price or t.market_price or 0.0)

        # Recalculate EV dollars based on ACTUAL capped amount
        # (DeltaIntent constructor sets this pre-cap, so we must update it)
        if intent.effective_edge:
             intent.expected_value_dollars = float(intent.effective_edge) * float(intent.amount_usd)

        # Additional logic for derived fields
        intent.suggested_stake = intent.amount_usd
        intent.direction = "Long" if intent.side == "YES" else "Short"
        
        # Map fractions safely
        current_value = current_exposure.get(intent.key, 0.0)
        intent.kelly_fraction_existing = current_value / bankroll if bankroll > 0 else 0
        intent.kelly_fraction_applied = -intent.amount_usd / bankroll if bankroll > 0 else 0
        
        intent.notes = f"{intent.action}: {intent.reason}"
    
    LAST_RECO_DEBUG = pd.DataFrame([asdict(t) for t in targets.values()]) if targets else pd.DataFrame()
    
    return actionable


# Alias for backwards compatibility
recommend_actions = recommend_trades


def recommendations_to_dataframe(recs: List[DeltaIntent]) -> pd.DataFrame:
    """Convert recommendations to a DataFrame for display."""
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(rec) for rec in recs])
    df["stake_dollars"] = df["suggested_stake"]
    df["ev_dollars"] = df["expected_value_dollars"]
    df["edge"] = df["effective_edge"]
    return df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

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
                        help="Path to batch_with_fits.csv")
    parser.add_argument("--min-edge", type=float, default=0.06,
                        help="Minimum edge threshold (default: 0.06)")
    parser.add_argument("--kelly-fraction", type=float, default=0.15,
                        help="Kelly fraction multiplier (default: 0.15)")
    parser.add_argument("--max-bets-per-expiry", type=int, default=3,
                        help="Maximum bets per expiry date (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path for recommendations")
    
    args = parser.parse_args()
    
    # Load data
    if args.input:
        df = pd.read_csv(args.input)
        print(f"Loaded: {args.input}")
    else:
        df = load_latest_fitted_batch()
        if df is None:
            print("Error: No fitted batch data found.")
            return
    
    print(f"Data shape: {df.shape}")
    
    # Generate recommendations
    recommendations = recommend_trades(
        df=df,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        min_edge=args.min_edge,
        max_bets_per_expiry=args.max_bets_per_expiry,
    )
    
    if not recommendations:
        print("\nNo trade recommendations found.")
        return
    
    # Convert to DataFrame and display
    reco_df = recommendations_to_dataframe(recommendations)
    
    print(f"\n{'='*60}")
    print(f"TRADE RECOMMENDATIONS (Bankroll: ${args.bankroll:,.2f})")
    print(f"{'='*60}")
    
    display_cols = [
        "expiry_key", "strike", "side", "action", "amount_usd",
        "effective_edge", "model_prob"
    ]
    available_cols = [c for c in display_cols if c in reco_df.columns]
    print(reco_df[available_cols].to_string(index=False))
    
    # Summary
    buys = reco_df[reco_df["action"] == "BUY"]["amount_usd"].sum()
    sells = reco_df[reco_df["action"] == "SELL"]["amount_usd"].sum()
    print(f"\nTotal BUY: ${buys:,.2f}, Total SELL: ${sells:,.2f}")
    
    if args.output:
        reco_df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
