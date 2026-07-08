"""Robustness / sizing layer (plan Section 2.8, task Z1).

Converts edge into stake, never full-Kelly, through a staged pipeline whose every
stage is recorded in a per-decision audit:

  1. Per-contract Kelly f* (buying YES at price P with belief p_hat: b=(1-P)/P,
     f* = (b*p_hat-(1-p_hat))/b; buying NO is the same form with belief 1-p_hat at
     NO price 1-P). Negative f* -> 0.
  2. Baker-McHale shrink k = f*^2/(f*^2 + ((b+1)/b)^2 * sigma2) using the common
     mode sigma2_ladder (NOT per-strike independent -- Section 1.1 caveat 1).
  3. Joint ladder allocation: strikes on one expiry share one BTC path, so the
     summed per-contract fractions overbet. Conservative stand-in for the full
     joint log-wealth optimization (plan permits component-level pragmatism; the
     full optimizer is a later refinement): scale all fractions in the expiry so
     their SUM never exceeds the single largest unscaled fraction. Records
     LADDER_JOINT when binding.
  4. Downside / ruin cap: hard per-expiry at-risk cap and total bankroll
     utilization cap. Records RUIN / BANKROLL.
  5. Fractional-Kelly ceiling c<=0.5, ALWAYS applied last, records FRACTIONAL_C.

  Depth cap: quote size also bounded by LiquidityState realized depth per side
  when provided (inert when absent). Records DEPTH when binding.

Fractions convert to shares via size = f*bankroll/price_per_share, where the risk
per share is P for a YES bet at price P and 1-P for a NO bet. The phi directive
couples robustness to the inventory penalty (plan 2.5/2.8):
phi = phi_running_penalty * (1 + sigma2_scale * sqrt(sigma2_ladder)) -- a simple
monotone-in-sigma2 form.

Depends only on market_maker.contracts, market_maker.config and stdlib.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from market_maker.config import MMConfig
from market_maker.contracts import (
    LiquidityState,
    PricerSnapshot,
    SizingCap,
    SizingDecision,
)

# Module defaults for cap fractions not carried by MMConfig (plan 2.8 note).
DEFAULT_PER_EXPIRY_CAP_FRAC = 0.10
DEFAULT_BANKROLL_UTIL_CAP = 0.5  # of paper bankroll

# Canonical caps_applied ordering; FRACTIONAL_C is always last (plan 2.8 / Z1).
_CAP_ORDER = [
    SizingCap.LADDER_JOINT,
    SizingCap.RUIN,
    SizingCap.BANKROLL,
    SizingCap.DEPTH,
    SizingCap.FRACTIONAL_C,
]


@dataclass
class ContractSizingInput:
    """One ladder contract to size. bid_price is our YES bid (we buy YES there);
    ask_price is our YES ask (we sell YES there = buy NO at 1-ask_price)."""

    market_id: str
    p_hat: float
    bid_price: float
    ask_price: float


# ---------------------------------------------------------------------------
# Stage 1 / Stage 2 pure helpers
# ---------------------------------------------------------------------------


def kelly_buy(belief_p: float, price: float) -> Tuple[float, float]:
    """Kelly fraction for buying a binary at `price` with win-probability
    belief `belief_p`. Returns (f_star, b) where b=(1-price)/price is the net
    odds. Negative f_star is floored to 0 (do not bet that side)."""
    if not (0.0 < price < 1.0):
        return 0.0, 0.0
    b = (1.0 - price) / price
    f = (b * belief_p - (1.0 - belief_p)) / b
    if not math.isfinite(f) or f < 0.0:
        f = 0.0
    return f, b


def baker_mchale(f_star: float, b: float, sigma2: float) -> float:
    """Baker-McHale shrinkage factor k in [0,1], monotone decreasing in sigma2.
    sigma2=0 -> k=1. f_star=0 -> k=1 (size is 0 regardless)."""
    if f_star <= 0.0:
        return 1.0
    if b <= 0.0:
        return 1.0
    denom = f_star * f_star + ((b + 1.0) / b) ** 2 * sigma2
    if denom <= 0.0:
        return 1.0
    k = (f_star * f_star) / denom
    return max(0.0, min(1.0, k))


# ---------------------------------------------------------------------------
# Internal per-leg record
# ---------------------------------------------------------------------------


@dataclass
class _Leg:
    market_id: str
    is_yes: bool
    price_per_share: float  # risk per share: P (YES) or 1-P (NO)
    f_star: float
    b: float
    k_shrink: float
    f: float  # working fraction through the stages


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def size_ladder(
    contracts: List[ContractSizingInput],
    snapshot: PricerSnapshot,
    bankroll: float,
    ts: datetime,
    config: Optional[MMConfig] = None,
    liquidity: Optional[Dict[str, LiquidityState]] = None,
    per_expiry_cap_frac: float = DEFAULT_PER_EXPIRY_CAP_FRAC,
    bankroll_util_cap: float = DEFAULT_BANKROLL_UTIL_CAP,
    sigma2_scale: float = 1.0,
) -> Tuple[Dict[str, SizingDecision], Dict[str, Any]]:
    """Size a whole expiry ladder. Returns (decisions_by_market_id, audit)."""
    if config is None:
        config = MMConfig()
    sigma2 = float(snapshot.sigma2_ladder)  # common-mode ladder variance
    audit: Dict[str, Any] = {"stages": [], "sigma2_ladder": sigma2}

    # Stage 1 + 2: build legs (YES via bid, NO via ask).
    legs: List[_Leg] = []
    for c in contracts:
        f_yes, b_yes = kelly_buy(c.p_hat, c.bid_price)
        k_yes = baker_mchale(f_yes, b_yes, sigma2)
        legs.append(
            _Leg(c.market_id, True, c.bid_price, f_yes, b_yes, k_yes, f_yes * k_yes)
        )
        no_price = 1.0 - c.ask_price
        f_no, b_no = kelly_buy(1.0 - c.p_hat, no_price)
        k_no = baker_mchale(f_no, b_no, sigma2)
        legs.append(
            _Leg(c.market_id, False, no_price, f_no, b_no, k_no, f_no * k_no)
        )
    audit["stages"].append(
        {"stage": "kelly+baker_mchale", "f": [(lg.market_id, lg.is_yes, lg.f) for lg in legs]}
    )

    triggered = set()

    # Stage 3: joint ladder allocation -- SUM(f) <= largest single unscaled f.
    unscaled = [lg.f for lg in legs]
    sum_f = sum(unscaled)
    max_single = max(unscaled) if unscaled else 0.0
    if sum_f > max_single + 1e-15 and sum_f > 0.0:
        factor = max_single / sum_f
        for lg in legs:
            lg.f *= factor
        triggered.add(SizingCap.LADDER_JOINT)
    audit["stages"].append(
        {"stage": "ladder_joint", "sum_before": sum_f, "max_single": max_single}
    )

    # Stage 4: downside / ruin caps -- per-expiry at-risk then bankroll util.
    sum_f = sum(lg.f for lg in legs)
    if sum_f > per_expiry_cap_frac + 1e-15 and sum_f > 0.0:
        factor = per_expiry_cap_frac / sum_f
        for lg in legs:
            lg.f *= factor
        triggered.add(SizingCap.RUIN)
    sum_f = sum(lg.f for lg in legs)
    if sum_f > bankroll_util_cap + 1e-15 and sum_f > 0.0:
        factor = bankroll_util_cap / sum_f
        for lg in legs:
            lg.f *= factor
        triggered.add(SizingCap.BANKROLL)
    audit["stages"].append(
        {"stage": "ruin_bankroll", "per_expiry_cap": per_expiry_cap_frac,
         "bankroll_util_cap": bankroll_util_cap}
    )

    # Stage 5: fractional-Kelly ceiling -- ALWAYS applied last.
    c_frac = min(config.fractional_kelly_c, 0.5)
    for lg in legs:
        lg.f *= c_frac
    triggered.add(SizingCap.FRACTIONAL_C)
    audit["stages"].append({"stage": "fractional_c", "c": c_frac})

    # Convert fractions to shares and apply the depth cap per side.
    ladder_alloc: Dict[str, float] = {lg.market_id: 0.0 for lg in legs}
    bid_shares: Dict[str, float] = {}
    ask_shares: Dict[str, float] = {}
    for lg in legs:
        shares = 0.0
        if lg.f > 0.0 and lg.price_per_share > 0.0:
            shares = lg.f * bankroll / lg.price_per_share
        if lg.is_yes:
            bid_shares[lg.market_id] = shares
        else:
            ask_shares[lg.market_id] = shares
        ladder_alloc[lg.market_id] = max(ladder_alloc[lg.market_id], lg.f)

    if liquidity is not None:
        for mid, liq in liquidity.items():
            if mid in bid_shares and bid_shares[mid] > liq.realized_depth_bid + 1e-12:
                bid_shares[mid] = liq.realized_depth_bid
                triggered.add(SizingCap.DEPTH)
            if mid in ask_shares and ask_shares[mid] > liq.realized_depth_ask + 1e-12:
                ask_shares[mid] = liq.realized_depth_ask
                triggered.add(SizingCap.DEPTH)

    caps_applied = [c for c in _CAP_ORDER if c in triggered]

    phi = config.phi_running_penalty * (1.0 + sigma2_scale * math.sqrt(max(sigma2, 0.0)))

    # Assemble per-contract decisions. f_kelly/k_shrink report the dominant leg.
    by_market: Dict[str, _Leg] = {}
    for lg in legs:
        cur = by_market.get(lg.market_id)
        if cur is None or lg.f_star > cur.f_star:
            by_market[lg.market_id] = lg

    decisions: Dict[str, SizingDecision] = {}
    for c in contracts:
        dom = by_market[c.market_id]
        decisions[c.market_id] = SizingDecision(
            ts=ts,
            market_id=c.market_id,
            bid_size=bid_shares.get(c.market_id, 0.0),
            ask_size=ask_shares.get(c.market_id, 0.0),
            f_kelly=dom.f_star,
            k_shrink=dom.k_shrink,
            ladder_alloc=ladder_alloc.get(c.market_id, 0.0),
            caps_applied=list(caps_applied),
            sigma2_used=sigma2,
            phi_directive=phi,
        )
    return decisions, audit
