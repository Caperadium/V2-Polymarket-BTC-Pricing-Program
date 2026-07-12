"""Robustness / sizing layer (plan Section 2.8, task Z1; rewritten per
temp/mm_sizing_fix_plan.md C1-C5, 2026-07-12).

Converts edge into stake, never full-Kelly, through a staged pipeline whose every
stage is recorded in a per-decision audit. The pipeline runs in two spaces:
fraction space (per-bankroll fractions, comparable across legs) then share space
(actual order sizes, comparable against depth/inventory/ruin limits that are
naturally share- or notional-denominated).

FRACTION SPACE:

  1. Per-contract Kelly f* (buying YES at edge price P with belief p_hat:
     b=(1-P)/P, f* = (b*p_hat-(1-p_hat))/b; buying NO is the same form with
     belief 1-p_hat at NO price 1-P). Negative f* -> 0. The edge price P is the
     MARKET mid (ContractSizingInput.mkt_mid) when available, falling back to
     our own quote side (bid_price / 1-ask_price) when it is not (plan C1) --
     this decouples conviction sizing from our own spread calibration, which
     the old "edge vs our own quote" form did not.
  2. Baker-McHale shrink k = f*^2/(f*^2 + ((b+1)/b)^2 * sigma2), using
     snapshot.sigma2[strike] when the leg's contract carries a strike present
     in the snapshot's per-strike map, else falling back to sigma2_ladder
     (plan C4). phi (below) keeps sigma2_ladder -- it is a ladder-level
     quantity by plan design, not a per-leg one.
  3. Bankroll utilization cap: total fraction across all legs <= bankroll_util_cap
     of paper bankroll. Records BANKROLL when binding.
  4. Fractional-Kelly ceiling c<=0.5 -- the LAST fraction-space ceiling, records
     FRACTIONAL_C.

SHARE SPACE (fractions convert to shares via shares = f*bankroll/price_per_share,
where price_per_share is OUR quote side: P for a YES bet at our bid, 1-P for a
NO bet at our ask -- the capital actually at risk per share if filled):

  5. Presence floor (plan C3): leg_shares = max(kelly_shares, presence_shares),
     where presence_shares = presence_frac * bankroll / price_per_share, tapered
     toward zero as inventory on that side approaches q_max. A pure floor -- it
     only ever raises a leg's size, never lowers one respected by a firmer cap
     below. Records an audit "presence_floor" stage; does NOT add a SizingCap
     member (the floor is not a cap).
  6. Inventory headroom cap (plan C2): bid_shares <= q_max - q, ask_shares <=
     q_max + q, from InventoryState when provided. Records INVENTORY when
     binding.
  7. Depth cap: quote size also bounded by LiquidityState realized depth per
     side when provided (inert when absent). Records DEPTH when binding. Runs
     AFTER the presence floor and inventory cap so it remains a hard minimum
     over both (a floored-then-inventory-capped size that still exceeds
     realized depth is clipped to depth, exactly as before the floor existed).
  8. Bucket worst-case recheck (plan C5): the ladder's strikes partition
     terminal spot into buckets; within a bucket, every leg that loses (YES at
     K loses iff spot <= K; NO at K loses iff spot > K) contributes its
     share-space risk fraction (shares*price_per_share/bankroll) to that
     bucket's loss. If the worst bucket's loss exceeds per_expiry_cap_frac,
     ALL legs' shares are scaled down by the same factor in a single pass.
     Records RUIN when binding. This REPLACES the old two-part stand-in (a
     fraction-space "sum(f) <= max single f" joint-ladder scale, which
     ignored that YES/NO legs at different strikes cannot both lose, plus a
     separate fraction-space "sum(f) <= per_expiry_cap_frac" ruin scale) with
     the plan's true worst-case bucket decomposition, enforced once in share
     space as the final pipeline step. SizingCap.LADDER_JOINT is retired from
     new audits (member kept only for old-journal compat).

INVARIANT (restated 2026-07-12, RC-1): the long-standing "fractional-c is
always applied last" statement is true only within fraction space --
fractional-c is the LAST FRACTION-SPACE ceiling. The bucket/ruin worst-case
cap is a final SHARE-SPACE override that runs after it (a ruin control
outranks every ceiling, including a floor: a leg scaled down by the bucket
recheck may end up below its own presence floor -- caps dominate floors,
always). caps_applied display order (_CAP_ORDER) keeps FRACTIONAL_C last
for readability; that ordering is cosmetic and does not describe execution
order.

Depends only on market_maker.contracts, market_maker.config and stdlib.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from market_maker.config import MMConfig
from market_maker.contracts import (
    InventoryState,
    LiquidityState,
    PricerSnapshot,
    SizingCap,
    SizingDecision,
)

# Module defaults for cap fractions not carried by MMConfig (plan 2.8 note).
DEFAULT_PER_EXPIRY_CAP_FRAC = 0.10
DEFAULT_BANKROLL_UTIL_CAP = 0.5  # of paper bankroll

# Canonical caps_applied ordering; FRACTIONAL_C is always last (display order
# only -- see the INVARIANT paragraph above for the actual execution order).
_CAP_ORDER = [
    SizingCap.LADDER_JOINT,
    SizingCap.RUIN,
    SizingCap.BANKROLL,
    SizingCap.INVENTORY,
    SizingCap.DEPTH,
    SizingCap.FRACTIONAL_C,
]


@dataclass
class ContractSizingInput:
    """One ladder contract to size.

    bid_price/ask_price are OUR OWN quote sides (we buy YES at bid_price, we
    sell YES / buy NO at 1-ask_price): they feed price_per_share, the capital
    at risk per share if filled. They are NOT the edge price used for Kelly.

    mkt_mid is the market's own mid (YES-scale), and IS the edge price when
    present: Kelly compares belief p_hat against mkt_mid (YES leg) / 1-mkt_mid
    (NO leg). When mkt_mid is None, the edge price falls back to our own quote
    side (bid_price / 1-ask_price) -- the pre-C1 behavior.

    strike is optional; when provided and present in the PricerSnapshot's
    per-strike sigma2 map, it selects that leg's Baker-McHale sigma2 (plan C4)
    and participates in the bucket worst-case recheck (plan C5). Missing
    strikes fall back to sigma2_ladder for shrinkage and to a conservative
    sum-cap fallback for the bucket recheck (see size_ladder).
    """

    market_id: str
    p_hat: float
    bid_price: float
    ask_price: float
    mkt_mid: Optional[float] = None
    strike: Optional[float] = None


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
    strike: Optional[float]
    price_per_share: float  # risk per share: P (YES) or 1-P (NO), OUR quote side
    f_star: float
    b: float
    k_shrink: float
    f: float  # working fraction through the fraction-space stages
    shares: float = 0.0  # working shares through the share-space stages


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
    inventory: Optional[InventoryState] = None,
    per_expiry_cap_frac: float = DEFAULT_PER_EXPIRY_CAP_FRAC,
    bankroll_util_cap: float = DEFAULT_BANKROLL_UTIL_CAP,
    sigma2_scale: float = 1.0,
) -> Tuple[Dict[str, SizingDecision], Dict[str, Any]]:
    """Size a whole expiry ladder. Returns (decisions_by_market_id, audit)."""
    if config is None:
        config = MMConfig()
    sigma2_ladder = float(snapshot.sigma2_ladder)  # common-mode ladder variance
    audit: Dict[str, Any] = {"stages": [], "sigma2_ladder": sigma2_ladder}

    # Stage 1 + 2: build legs (YES via bid, NO via ask). Edge price is
    # mkt_mid when available (plan C1); price_per_share stays our own quote
    # side regardless. Baker-McHale sigma2 is per-strike when available
    # (plan C4), else falls back to the ladder common mode.
    legs: List[_Leg] = []
    for c in contracts:
        leg_sigma2 = sigma2_ladder
        if c.strike is not None and c.strike in snapshot.sigma2:
            leg_sigma2 = float(snapshot.sigma2[c.strike])

        yes_edge_price = c.mkt_mid if c.mkt_mid is not None else c.bid_price
        f_yes, b_yes = kelly_buy(c.p_hat, yes_edge_price)
        k_yes = baker_mchale(f_yes, b_yes, leg_sigma2)
        legs.append(
            _Leg(c.market_id, True, c.strike, c.bid_price, f_yes, b_yes, k_yes, f_yes * k_yes)
        )

        no_price_per_share = 1.0 - c.ask_price
        no_edge_price = (1.0 - c.mkt_mid) if c.mkt_mid is not None else no_price_per_share
        f_no, b_no = kelly_buy(1.0 - c.p_hat, no_edge_price)
        k_no = baker_mchale(f_no, b_no, leg_sigma2)
        legs.append(
            _Leg(c.market_id, False, c.strike, no_price_per_share, f_no, b_no, k_no, f_no * k_no)
        )
    audit["stages"].append(
        {"stage": "kelly+baker_mchale", "f": [(lg.market_id, lg.is_yes, lg.f) for lg in legs]}
    )

    triggered = set()

    # --- Fraction space -------------------------------------------------

    # Stage 3: bankroll utilization cap -- sum(f) across the whole ladder.
    sum_f = sum(lg.f for lg in legs)
    if sum_f > bankroll_util_cap + 1e-15 and sum_f > 0.0:
        factor = bankroll_util_cap / sum_f
        for lg in legs:
            lg.f *= factor
        triggered.add(SizingCap.BANKROLL)
    audit["stages"].append(
        {"stage": "bankroll_util", "sum_before": sum_f, "bankroll_util_cap": bankroll_util_cap}
    )

    # Stage 4: fractional-Kelly ceiling -- LAST fraction-space stage.
    c_frac = min(config.fractional_kelly_c, 0.5)
    for lg in legs:
        lg.f *= c_frac
    triggered.add(SizingCap.FRACTIONAL_C)
    audit["stages"].append({"stage": "fractional_c", "c": c_frac})

    # --- Share space ------------------------------------------------------

    # Convert fractions to shares (price_per_share = OUR quote side).
    for lg in legs:
        lg.shares = 0.0
        if lg.f > 0.0 and lg.price_per_share > 0.0:
            lg.shares = lg.f * bankroll / lg.price_per_share

    # Stage 5: presence floor (plan C3). A pure floor: only ever raises a
    # leg's shares. presence_frac<=0 disables (floor contributes 0).
    inv_by_market: Dict[str, Any] = dict(inventory.per_contract) if inventory is not None else {}
    presence_info: List[Dict[str, Any]] = []
    if config.presence_frac > 0.0:
        for lg in legs:
            if lg.price_per_share <= 0.0:
                continue
            s_presence = config.presence_frac * bankroll / lg.price_per_share
            taper = 1.0
            cinv = inv_by_market.get(lg.market_id)
            if cinv is not None:
                if cinv.q_max <= 0.0:
                    taper = 0.0
                else:
                    q_toward_side = cinv.q if lg.is_yes else -cinv.q
                    taper = max(0.0, min(1.0, 1.0 - q_toward_side / cinv.q_max))
            presence_shares = s_presence * taper
            if presence_shares > lg.shares:
                lg.shares = presence_shares
            presence_info.append(
                {"market_id": lg.market_id, "is_yes": lg.is_yes,
                 "presence_shares": presence_shares, "taper": taper}
            )
    audit["stages"].append({"stage": "presence_floor", "legs": presence_info})

    # Stage 6: inventory headroom cap (plan C2) -- hard min.
    max_add_yes: Dict[str, float] = {}
    max_add_no: Dict[str, float] = {}
    if inventory is not None:
        for lg in legs:
            cinv = inv_by_market.get(lg.market_id)
            if cinv is None:
                continue
            if lg.is_yes:
                headroom_bid = max(0.0, cinv.q_max - cinv.q)
                max_add_yes[lg.market_id] = headroom_bid
                if lg.shares > headroom_bid + 1e-12:
                    lg.shares = headroom_bid
                    triggered.add(SizingCap.INVENTORY)
            else:
                headroom_ask = max(0.0, cinv.q_max + cinv.q)
                max_add_no[lg.market_id] = headroom_ask
                if lg.shares > headroom_ask + 1e-12:
                    lg.shares = headroom_ask
                    triggered.add(SizingCap.INVENTORY)
    audit["stages"].append(
        {"stage": "inventory_headroom", "max_add_yes": dict(max_add_yes), "max_add_no": dict(max_add_no)}
    )

    # Stage 7: depth cap -- hard min, AFTER the floor and inventory cap
    # (CRITICAL: must stay a hard min here so a pre-existing depth-binding
    # size is unaffected by the floor's max()).
    if liquidity is not None:
        for lg in legs:
            liq = liquidity.get(lg.market_id)
            if liq is None:
                continue
            if lg.is_yes:
                if lg.shares > liq.realized_depth_bid + 1e-12:
                    lg.shares = liq.realized_depth_bid
                    triggered.add(SizingCap.DEPTH)
            else:
                if lg.shares > liq.realized_depth_ask + 1e-12:
                    lg.shares = liq.realized_depth_ask
                    triggered.add(SizingCap.DEPTH)

    # Stage 8: bucket worst-case recheck (plan C5) -- final share-space
    # override, single pass, dominates every floor and every earlier cap.
    all_have_strikes = all(s is not None for s in (lg.strike for lg in legs))
    if not all_have_strikes:
        # Fallback safety (no strikes to bucket): conservative fraction-space
        # sum-cap semantics from the old ruin stage, applied in share space.
        sum_risk = sum(
            (lg.shares * lg.price_per_share / bankroll) if bankroll > 0.0 else 0.0
            for lg in legs
        )
        if sum_risk > per_expiry_cap_frac + 1e-15 and sum_risk > 0.0:
            factor = per_expiry_cap_frac / sum_risk
            for lg in legs:
                lg.shares *= factor
            triggered.add(SizingCap.RUIN)
        audit["stages"].append(
            {"stage": "bucket_worst_case", "fallback": "sum_cap_no_strikes",
             "sum_risk": sum_risk, "per_expiry_cap_frac": per_expiry_cap_frac}
        )
    else:
        unique_strikes = sorted(set(lg.strike for lg in legs if lg.strike is not None))
        if unique_strikes and bankroll > 0.0:
            # n+1 buckets: (-inf, K1], (K1, K2], ..., (Kn, +inf).
            bucket_bounds = unique_strikes  # upper bound of each of the first n buckets
            n_buckets = len(bucket_bounds) + 1
            losses = [0.0] * n_buckets

            for b_idx in range(n_buckets):
                # bucket b_idx represents outcomes: spot <= bucket_bounds[0] for
                # b_idx==0; bucket_bounds[b_idx-1] < spot <= bucket_bounds[b_idx]
                # for interior buckets; spot > bucket_bounds[-1] for the last.
                if b_idx == 0:
                    lo, hi = float("-inf"), bucket_bounds[0]
                elif b_idx == n_buckets - 1:
                    lo, hi = bucket_bounds[-1], float("inf")
                else:
                    lo, hi = bucket_bounds[b_idx - 1], bucket_bounds[b_idx]
                # Representative spot strictly inside (lo, hi] for the strictly-
                # above resolution rule: pick hi when finite (a YES@K loses iff
                # spot<=K, a NO@K loses iff spot>K -- using spot=hi, finite,
                # correctly classifies every leg whose strike==hi as "at K",
                # and every other leg by its strike's relation to (lo, hi]).
                loss = 0.0
                for lg in legs:
                    if lg.strike is None or lg.shares <= 0.0:
                        continue
                    k = lg.strike
                    risk_frac = lg.shares * lg.price_per_share / bankroll
                    if lg.is_yes:
                        # YES loses iff spot <= k; true for this bucket iff
                        # bucket's representative range is entirely <= k, i.e.
                        # hi <= k (bucket upper bound at or below strike).
                        loses = hi <= k
                    else:
                        # NO loses iff spot > k; true for this bucket iff the
                        # bucket's range is entirely > k, i.e. lo >= k.
                        loses = lo >= k
                    if loses:
                        loss += risk_frac
                losses[b_idx] = loss

            max_loss = max(losses) if losses else 0.0
            bucket_losses = {
                "bucket_%d" % i: losses[i] for i in range(n_buckets)
            }
            if max_loss > per_expiry_cap_frac + 1e-15 and max_loss > 0.0:
                factor = per_expiry_cap_frac / max_loss
                for lg in legs:
                    lg.shares *= factor
                triggered.add(SizingCap.RUIN)
            audit["stages"].append(
                {"stage": "bucket_worst_case", "bucket_losses": bucket_losses,
                 "max_loss": max_loss, "per_expiry_cap_frac": per_expiry_cap_frac}
            )
        else:
            audit["stages"].append(
                {"stage": "bucket_worst_case", "fallback": "no_strikes_or_bankroll"}
            )

    caps_applied = [c for c in _CAP_ORDER if c in triggered]

    phi = config.phi_running_penalty * (1.0 + sigma2_scale * math.sqrt(max(sigma2_ladder, 0.0)))

    # Assemble per-contract decisions. f_kelly/k_shrink report the dominant leg.
    by_market: Dict[str, _Leg] = {}
    for lg in legs:
        cur = by_market.get(lg.market_id)
        if cur is None or lg.f_star > cur.f_star:
            by_market[lg.market_id] = lg

    bid_shares = {lg.market_id: lg.shares for lg in legs if lg.is_yes}
    ask_shares = {lg.market_id: lg.shares for lg in legs if not lg.is_yes}
    ladder_alloc: Dict[str, float] = {}
    for lg in legs:
        ladder_alloc[lg.market_id] = max(ladder_alloc.get(lg.market_id, 0.0), lg.f)

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
            sigma2_used=sigma2_ladder,
            phi_directive=phi,
            max_add_yes=max_add_yes.get(c.market_id, 0.0),
            max_add_no=max_add_no.get(c.market_id, 0.0),
        )
    return decisions, audit
