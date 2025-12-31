"""
common.py

Shared types, constants, and data structures for the strategy layer.
Extracted from auto_reco.py to avoid circular imports and improve organization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional

import pandas as pd


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
