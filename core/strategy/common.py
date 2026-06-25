"""
common.py

Shared types, constants, and data structures for the strategy layer.
Extracted from auto_reco.py to avoid circular imports and improve organization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Literal, Optional, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Model-probability resolution
# -----------------------------------------------------------------------------

# Precedence: fitted curve prob > raw MC prob > generic model prob.
MODEL_PROB_CANDIDATES: Sequence[str] = (
    "p_model_fit", "p_real_mc", "model_probability", "Model_Prob",
)

# FIX 7 (M2): when outcome-based recalibration is enabled, the calibrated column
# `p_model_cal` takes top precedence. This is a SEPARATE tuple, NOT a prepend onto
# the default — because `resolve_model_prob` coalesces by first-finite value,
# merely prepending would activate calibration the instant any batch wrote the
# column, with no off switch. The flag below selects which tuple is used.
MODEL_PROB_CANDIDATES_CALIBRATED: Sequence[str] = (
    "p_model_cal",
) + tuple(MODEL_PROB_CANDIDATES)

# Master switch for outcome-based recalibration (FIX 7). Default OFF: every edge in
# the system is derived from the model probability, so flipping this changes ALL
# edges. Keep False until a walk-forward reliability diagram shows the logit shift
# is stable (CI excludes a large move, n_obs ≥ 200 per DTE bucket) and the owner
# signs off. When True, scoring writes `p_model_cal` and resolve_model_prob prefers it.
USE_CALIBRATED_PROB: bool = False


def filter_by_moneyness(
    df: pd.DataFrame,
    *,
    lower: Optional[float],
    upper: Optional[float],
    mode: str = "abs",
    money_col_candidates: Sequence[str] = ("moneyness",),
) -> pd.DataFrame:
    """Filter rows by moneyness.

    mode="abs"    -> bounds applied to |moneyness| (legacy, symmetric).
    mode="signed" -> bounds applied to raw signed moneyness; bounds may be negative.
    A None bound is unbounded on that side. Rows with NaN/absent moneyness are
    dropped only when at least one bound is active. If no moneyness column is
    present, the frame is returned unchanged (cannot filter).
    """
    if lower is None and upper is None:
        return df
    col = next((c for c in money_col_candidates if c in df.columns), None)
    if col is None:
        return df
    m = pd.to_numeric(df[col], errors="coerce")
    basis = m.abs() if mode == "abs" else m
    mask = basis.notna()
    if lower is not None:
        mask &= basis >= lower
    if upper is not None:
        mask &= basis <= upper
    return df[mask].copy()


def latest_spot_as_of(
    btc_df: Optional[pd.DataFrame],
    as_of: datetime,
    ts_candidates: Sequence[str] = ("timestamp", "time", "datetime"),
) -> Optional[float]:
    """Last intraday close strictly before `as_of` (leak-free). None if unavailable.

    NOTE: `"date"` is intentionally NOT a candidate — in live batch CSVs `date` is a
    day-string (midnight), which would yield day-granularity spot. This helper requires
    an intraday timestamp column (which `load_btc_csv` provides).

    Mirrors the backrunner/backtest S0 convention (strict `<` cutoff: Binance bars
    are open-stamped, so `<=` would leak the bar that closes after the snapshot).
    """
    if btc_df is None or btc_df.empty or "close" not in btc_df.columns:
        return None
    ts_col = next((c for c in ts_candidates if c in btc_df.columns), None)
    if ts_col is None:
        return None
    ts = pd.to_datetime(btc_df[ts_col], utc=True, errors="coerce")
    prior = btc_df.loc[ts < pd.Timestamp(as_of, tz="UTC") if as_of.tzinfo is None else ts < pd.Timestamp(as_of)]
    if prior.empty:
        return None
    close = pd.to_numeric(prior["close"], errors="coerce").dropna()
    return float(close.iloc[-1]) if not close.empty else None


def resolve_model_prob(
    df: pd.DataFrame,
    candidates: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Per-row coalesce of model-probability columns by precedence.

    Returns a float Series aligned to ``df.index``: for each row, the first
    FINITE value across *candidates* in order. This is a value-level fallback,
    not column-level — so an all-NaN (or partly-NaN) high-precedence column,
    e.g. a ``p_model_fit`` whose logistic curve fit failed for an expiry with too
    few strikes, no longer shadows a populated lower-precedence column such as
    ``p_real_mc``. Rows with no finite candidate remain NaN (callers drop them).

    FIX 7 (M2): when *candidates* is None, the tuple is chosen by the module flag
    ``USE_CALIBRATED_PROB`` — calibrated (`p_model_cal` first) when True, otherwise
    the default precedence. Pass an explicit tuple to override.
    """
    if candidates is None:
        candidates = (
            MODEL_PROB_CANDIDATES_CALIBRATED if USE_CALIBRATED_PROB
            else MODEL_PROB_CANDIDATES
        )
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out = out.where(out.notna(), vals)
    return out


# -----------------------------------------------------------------------------
# Constants & Defaults
# -----------------------------------------------------------------------------

MAX_CAP_PER_EXPIRY_FRAC_DEFAULT = 0.15
MAX_CAP_TOTAL_FRAC_DEFAULT = 0.35
STALE_SOFT_HOURS = 4.0
STALE_HARD_HOURS = 12.0
DEFAULT_MIN_TRADE_USD = 5.0
DEFAULT_REBALANCE_MIN_ADD_USD = 5.0
DEFAULT_REBALANCE_MIN_REDUCE_USD = 10.0
DEFAULT_EXIT_HYSTERESIS = 0.02  # Edge below entry but above exit = HOLD


# -----------------------------------------------------------------------------
# Target Role Enum (replaces string-based 'source' field)
# -----------------------------------------------------------------------------

class TargetRole(str, Enum):
    """Role of a target position in the portfolio pipeline."""
    ENTRY = "entry"           # New entry or increase (subject to consistency filter)
    EXIT = "exit"             # SELL signal (never filtered)
    HOLD_SAFETY = "hold_safety"  # Held position to keep (never filtered)
    NEUTRAL = "neutral"       # No action (pass-through)


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class TargetPosition:
    """Represents the ideal state for a single contract."""
    key: str  # Unique identifier
    slug: str
    side: str  # YES or NO
    expiry_key: str
    strike: float
    condition_id: Optional[str]
    
    target_fraction: float  # Ideal Kelly fraction
    target_usd: float  # target_fraction * bankroll
    
    model_prob: float
    market_price: float
    entry_price: float  # Price to execute (ask for buys)
    exit_price: float   # Price to execute (bid for sells)
    effective_edge: float
    
    allocation_score: float  # For ranking new capital allocation
    exit_score: float  # For ranking reductions
    
    role: TargetRole  # Role in portfolio pipeline (replaces 'source' string)
    
    # Debug/metadata
    kelly_full: float = 0.0
    kelly_mult_applied: float = 1.0
    stability_penalty: float = 1.0
    stale_mult: float = 1.0
    is_fallback_price: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeltaIntent:
    """Represents the trade action from Current to Target."""
    key: str
    slug: str
    side: str
    expiry_key: str
    strike: float
    condition_id: Optional[str]
    
    action: Literal["BUY", "SELL", "HOLD"]
    amount_usd: float  # Absolute amount to trade
    signed_delta_usd: float  # + for buy, - for sell
    
    current_usd: float  # Cost basis at risk before trade
    target_usd: float  # Desired exposure after trade
    
    price_mode: Optional[str]  # TAKER_ASK, TAKER_BID, or None for HOLD
    limit_price_hint: Optional[float]  # For UI display
    
    model_prob: float
    effective_edge: float
    reason: str
    
    # Compatibility fields for TradeRecommendation consumers
    intent_key: Optional[str] = None
    question: str = ""
    entry_price: float = 0.0
    market_price: float = 0.0
    kelly_fraction_full: float = 0.0
    kelly_fraction_full_effective: float = 0.0
    kelly_fraction_target: float = 0.0
    kelly_fraction_existing: float = 0.0
    kelly_fraction_applied: float = 0.0
    suggested_stake: float = 0.0
    expected_value_per_contract: float = 0.0
    expected_value_dollars: float = 0.0
    expiry_group_risk: float = 0.0
    stability_penalty: float = 1.0
    stale_mult: float = 1.0
    batch_age_hours: Optional[float] = None
    expiry_shape_label: str = "none"
    direction: str = ""
    notes: str = ""
    rn_prob: Optional[float] = None
    pricing_date: Optional[pd.Timestamp] = None
    is_fallback_price: bool = False


# Legacy alias for backwards compatibility
TradeRecommendation = DeltaIntent


# -----------------------------------------------------------------------------
# Config Dataclass
# -----------------------------------------------------------------------------

@dataclass
class RebalanceConfig:
    """Configuration for the rebalancing pipeline."""
    bankroll: float
    
    # Edge & Entry
    min_edge_entry: float = 0.02
    min_edge_exit: float = 0.00  # Hysteresis: exit only when below this
    spread_cost: float = 0.0  # Conservative default
    
    # Kelly & Sizing
    kelly_fraction: float = 0.15
    use_fixed_stake: bool = False
    fixed_stake_amount: float = 10.0
    
    # Caps
    max_capital_per_expiry_frac: float = MAX_CAP_PER_EXPIRY_FRAC_DEFAULT
    max_capital_total_frac: float = MAX_CAP_TOTAL_FRAC_DEFAULT
    max_bets_per_expiry: int = 3
    
    # Delta Caps
    max_add_per_cycle_usd: float = float("inf")
    max_reduce_per_cycle_usd: float = float("inf")
    
    # Churn Control
    rebalance_min_add_usd: float = DEFAULT_REBALANCE_MIN_ADD_USD
    rebalance_min_reduce_usd: float = DEFAULT_REBALANCE_MIN_REDUCE_USD
    min_trade_usd: float = DEFAULT_MIN_TRADE_USD
    
    # Filters
    min_price: float = 0.03
    max_price: float = 0.95
    min_model_prob: float = 0.0
    max_model_prob: float = 1.0
    max_dte: Optional[float] = None
    max_moneyness: Optional[float] = None
    min_moneyness: Optional[float] = None
    moneyness_mode: str = "abs"
    require_active: bool = True
    allow_no: bool = True
    
    # Stability
    use_stability_penalty: bool = True
    disable_staleness: bool = False
    
    # Safety Policies
    missing_target_policy: Literal["KEEP", "EXIT"] = "KEEP"
    risk_off_targets_to_zero: bool = True
    cap_breach_delever: bool = False
    
    # Prob threshold mode
    use_prob_threshold: bool = False
    prob_threshold_yes: float = 0.7
    prob_threshold_no: float = 0.3
