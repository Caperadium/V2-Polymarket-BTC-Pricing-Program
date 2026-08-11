"""Robustness / sizing layer (plan Section 2.8, task Z1; wave 1 per
temp/mm_sizing_fix_plan.md C1-C5; rewritten again per
temp/mm_sizing_wave2_plan.md W2-W5, 2026-07-12).

Converts edge into stake, never full-Kelly, through a staged pipeline whose every
stage is recorded in a per-decision audit. The pipeline runs in two spaces:
fraction space (per-bankroll fractions, comparable across legs) then share space
(actual order sizes, comparable against depth/inventory/ruin limits that are
naturally share- or notional-denominated).

FRACTION SPACE:

  1. Per-contract Kelly f* -- POSTED-QUOTE edge net of measured markout (wave
     2 W2, explicit reversal of wave 1's mkt_mid edge choice). Per leg, in the
     leg's own price scale:

       YES leg: price_leg = bid_price (our posted bid);  belief_leg = p_hat
       NO  leg: price_leg = 1 - ask_price (our posted ask); belief_leg = 1-p_hat

       structural_edge = belief_leg - price_leg
       m_prior = structural_edge - config.eps_base   # AS prior charge
       baseline = mk_avg if (mk_avg is not None and mk_n >= config.markout_min_n)
                  else m_prior                       # measured net edge (mid channel)
       m = min(baseline, mk_slow_avg) if slow channel measured else baseline
                                    # slow-horizon haircut (2026-08-08 wing-bleed
                                    # fix): strictly ONE-DIRECTIONAL -- min() can
                                    # lower m, never raise it
       m = max(m, 0.0)              # Glosten-Milgrom: negative net edge -> no size
       f_star, b = kelly_buy(price_leg + m, price_leg)  # belief_eff = price_leg + m

     The slow channel (2026-08-08 wing-bleed fix, temp/mm_wing_bleed_fix_plan.md
     2c): mk_slow_avg/mk_slow_n are the SAME market cell's markout at
     config.markout_slow_horizon_s (21600s/6h -- the 600s mid markout is
     structurally blind to slow theta bleed on low-delta wings, measured
     -2.7c at 600s vs -10 to -13c realized at settlement). It is "measured"
     only when markout_slow_horizon_s > 0 (belt-and-braces kill switch even
     for direct size_ladder callers), mk_slow_avg is not None and mk_slow_n
     >= markout_min_n. The slow channel NEVER sets sigma2_edge (Baker-McHale
     variance stays the mid channel's mk_var/mk_n or the prior).

     bid_price/ask_price are now the POSTED quote prices (the caller passes
     post-spread-builder prices, not our raw proposal) -- this is the
     Chen-Pennock utility-maker frame: Kelly at our OWN posted quote, belief
     haircut by measured adverse selection (realized-spread decomposition),
     not the market mid. The shifted-belief form of kelly_buy makes the
     resulting Kelly edge algebraically equal to exactly m at odds
     b=(1-price_leg)/price_leg -- no separate "edge" helper is needed.
  2. Baker-McHale shrink k = f*^2/(f*^2 + ((b+1)/b)^2 * sigma2_edge), where
     sigma2_edge is the markout-measurement variance of the SAME leg's net
     edge (wave 2 W2): sigma2_edge = mk_var/mk_n when measured (mk_n >=
     markout_min_n), else config.markout_prior_var (an uninformed prior).
     The per-strike/per-ladder MC-SE snapshot.sigma2 is DROPPED from leg
     shrinkage as of wave 2 (it double-charged the spread bet against a
     model-parameter uncertainty that is a different quantity from
     fill-level adverse selection); snapshot sigma2_ladder is kept for phi
     (below) and for audit only.
  3. Bankroll utilization cap: total fraction across all legs <= bankroll_util_cap
     of paper bankroll. Records BANKROLL when binding.
  4. Fractional-Kelly ceiling c<=0.5 -- the LAST fraction-space ceiling, records
     FRACTIONAL_C.

SHARE SPACE (fractions convert to shares via shares = f*bankroll/price_per_share,
where price_per_share is OUR quote side: P for a YES bet at our bid, 1-P for a
NO bet at our ask -- the capital actually at risk per share if filled):

  5. Presence floor (wave 1 C3, GATED as of wave 2 W4): leg_shares =
     max(kelly_shares, presence_shares), where presence_shares =
     presence_frac * bankroll / price_per_share, tapered toward zero as
     inventory on that side approaches q_max. Gated on measured net edge:

       m_gate = min-rule net edge from Stage 1 (baseline, haircut by a
                measured slow channel)                # NOT clamped at 0 here
       gate = (m_gate >= 0.0) or (
           mk_n_attempted < config.markout_min_n and not measured_slow_toxic)
       floor applies only when gate is True

     The exploration carve-out (mk_n_attempted < markout_min_n -> gate stays
     True regardless of m_gate's sign) is the anti-starvation clause: an
     unmeasured cell keeps the floor so fills can accumulate -- fills are the
     only source of markout/k/credibility calibration. A cold-start leg (no
     markout fields at all, mk_n_attempted=0) is always in the exploration
     carve-out, so it behaves exactly as wave 1 (unconditional floor). Once
     measured (mk_n_attempted >= min_n) with a negative net edge, the floor
     turns off on that side. 2026-08-08 wing-bleed fix: the carve-out is
     additionally SUPPRESSED while the slow channel is measured-toxic
     (measured_slow_toxic, see _leg_edge) -- without this the gate closure
     has a 28-day period (no fills -> the cell ages out of
     MARKOUT_LOOKBACK_S -> n_attempted falls below min_n -> carve-out
     re-arms -> ~20 exploration fills of tuition -> re-measure -> repeat). A pure floor otherwise -- it only ever raises a
     leg's size, never lowers one respected by a firmer cap below. Records an
     audit "presence_floor" stage; does NOT add a SizingCap member (the floor
     is not a cap).
  5b. Unmeasured-cell size multiplier (Fix 2b, 2026-07-26): for every leg
     whose market's mk_n_attempted < config.markout_min_n (the sizing cell is
     not yet trusted-measured), leg_shares *= config.unmeasured_size_mult,
     then the result is floored back up to config.depth_cap_floor_shares IF
     the pre-mult shares were already >= that floor (a leg below the floor
     pre-mult keeps shares*mult -- never resurrected). EXCEPT the reduce-side
     leg of a positioned market (q>0 -> ask/NO exempt; q<0 -> bid/YES exempt,
     same side-derivation as Stage 5c) -- shedding inventory is never
     throttled (Stage 5c's reduce floor is only presence-sized, so a
     multiplied reduce leg would unwind ~3x slower). depth_cap_floor_shares
     (1.0) is used as a proxy for VenueDescriptor.min_size (1.0), which sizing
     cannot see: without the floor-back, order_lifecycle would silently
     no-quote the throttled side, no fills would accumulate, and the cell
     could never become measured (learning deadlock); if the venue min ever
     diverges, this floor-back must follow it. Rationale: an unmeasured cell
     paid ~20 cap-sized losses of "tuition" at full size before the m-clamp
     could engage (m_prior ~ +half-spread is positive, so both Kelly and the
     floor ran full size); this cuts tuition per fill ~3x while fill COUNT
     (the learning signal) is nearly unchanged. Records an audit
     "unmeasured_mult" stage; not a cap. mult=1.0 disables (sizes
     byte-identical to pre-2b).
  5c. Reduce-side exemption (wave 2 W3): the f*>=0 floor above (and the Kelly
     edge itself) can zero the inventory-UNLOAD side exactly when skew
     exceeds the effective spread -- cash-EV Kelly cannot see the
     risk-relief utility of shedding inventory. Fix, UNGATED, applied at the
     same pipeline position as the presence floor (before headroom/depth/
     bucket caps -- caps still dominate):

       if q > 0: the ask/NO leg is this market's reduce side
       if q < 0: the bid/YES leg is the reduce side
       reduce_floor = min(abs(q), s_presence)   # s_presence: UNtapered wave-1
                                                 # floor unit for that leg
       leg_shares = max(leg_shares, reduce_floor)   # unconditional

     MAGNITUDE LIMITATION (explicit): min(|q|, s_presence) restores PRESENCE
     on the unload side (a floor-sized clip), not shedding capacity
     proportional to the excess -- a large position unwinds over multiple
     fills, not one; proportional unload sizing belongs to the Kelly path
     once measured markout data exists on that side. Directive suppression
     still wins downstream (build_quote_set mode zeroing) -- this exemption
     cannot resurrect a PULLED/one-sided market's suppressed side.
  6. Inventory headroom cap (plan C2): bid_shares <= q_max - q, ask_shares <=
     q_max + q, from InventoryState when provided. Records INVENTORY when
     binding.
  6b. Skew-aware entry cap (2026-08-10 skew-fix wave item 2, temp/
     mm_skew_fix_plan.md): item 1 (quote_engine's skew_x_cap) makes big q
     SAFE but not SENSIBLE -- once the reservation-shift clamp binds,
     additional inventory no longer moves the quote, so the ladder's main
     inventory-control channel saturates. Cap the ADD side so a leg's
     position cannot outrun that channel's authority:

       q_skew_max = skew_q_headroom_mult * config.skew_x_cap / unit_skew_x
       bid_shares <= max(0.0, q_skew_max - q)   # YES/bid = add side when q>=0
       ask_shares <= max(0.0, q_skew_max + q)   # NO/ask  = add side when q<0

     unit_skew_x (ContractSizingInput, additive) is the per-share reservation
     shift for this market this tick, resolved by the harness via
     quote_engine.per_share_skew_x(quote_variant, sigma_b, gamma, k_arrival,
     arrival_scale_A, tte) -- the SAME code path and the SAME sigma_b the
     quote engine used to build this tick's proposal, so the sizing cap and
     the quote clamp agree on the bind point under both variants (this
     module never imports quote_engine -- the field arrives pre-computed).
     0.0 (default) = unwired/disabled for that leg -> stage inert for it.
     config.skew_x_cap <= 0 is the paired kill switch (also stage inert,
     mirrors the quote-engine clamp's own disable). Gated on `inventory is
     not None` (needs the same per-market q the headroom cap above already
     resolves). Placed AFTER Stage 6 (inventory headroom) and BEFORE Stage 7
     (depth) / Stage 8 (bucket) -- caps dominate floors, same discipline as
     every other cap in this pipeline; runs after the Stage 5/5b/5c floors.
     No reduce-side exemption: for q >= 0, q_skew_max - q < q_skew_max + q
     always, so a long position's bid (its ADD side) binds strictly before
     its ask (its REDUCE side) -- the reduce side always has strictly more
     headroom and can never be the tighter bound; symmetric for q < 0.
     Verified by test, not special-cased (the plan is explicit that this is
     NOT needed, unlike Stage 5c). NO floor-back to depth_cap_floor_shares:
     this is a RISK cap, not a presence mechanism -- a side capped below the
     venue minimum becomes a no-quote via the existing order_lifecycle
     min-size rule. Records SizingCap.SKEW when it actually reduces a leg's
     shares. Journal truthfulness: max_add_yes/max_add_no report min(Stage-6
     value, q_skew_max bound) only for a leg where this stage is active;
     otherwise they are left exactly as Stage 6 set them (a naive min
     against an unset/zero q_skew_max would collapse reported headroom to 0
     on every legacy/kill-switch run).
  7. Depth cap, FLOORED (wave 2 W5): quote size bounded by
     max(realized_depth_side, config.depth_cap_floor_shares) rather than raw
     realized_depth_side -- a dead book (realized_depth=0) no longer
     permanently zeroes our size; the depth cap's purpose is impact control,
     not presence control. Records DEPTH when binding. Runs AFTER the
     presence floor, reduce-side exemption, and inventory cap so it remains a
     hard minimum over all three (a floored-then-inventory-capped size that
     still exceeds the floored depth bound is clipped to it, exactly as
     before the floor existed).
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

Depends only on market_maker.contracts, market_maker.config and stdlib
(deliberately NOT market_maker.pnl_report -- all markout lookups resolve in
the harness and arrive as plain fields on ContractSizingInput).
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
    SizingCap.SKEW,
    SizingCap.DEPTH,
    SizingCap.FRACTIONAL_C,
]


@dataclass
class ContractSizingInput:
    """One ladder contract to size.

    bid_price/ask_price are now the POSTED quote prices (wave 2 W1/W2): the
    harness computes these via spread_builder.compute_posted_prices and
    passes them here BEFORE sizing runs, then feeds the same tuple into
    build_quote_set(posted=...) so the two never disagree. They serve two
    roles: (a) price_per_share, the capital at risk per share if filled
    (bid_price for a YES buy, 1-ask_price for a NO buy); (b) as of wave 2,
    they are also the edge price itself -- Kelly compares belief against our
    OWN posted quote (Chen-Pennock utility-maker frame), not the market mid
    (mkt_mid, wave 1's edge choice, is REMOVED as of wave 2 -- see W2 in the
    module docstring).

    mk_avg/mk_var/mk_n/mk_n_attempted are the resolved markout-measurement
    cell for this market at the sizing horizon (wave 2 W2/W6, via
    pnl_report.markout_stats, resolved by the harness -- this module never
    imports pnl_report). Sign convention matches pnl_report.markout_report:
    mk = sign*(mid_h - fill_price), so mk_avg > 0 means we kept value on
    average. All default to "unmeasured" (None/0) so a caller that never
    wires markout data gets the m_prior fallback path everywhere.

    mk_slow_avg/mk_slow_n (2026-08-08 wing-bleed fix, additive) are the SAME
    market cell's markout at config.markout_slow_horizon_s (the slow
    channel). When measured (slow horizon enabled, mk_slow_avg not None,
    mk_slow_n >= markout_min_n) the Kelly net edge is haircut one-directionally
    to min(baseline, mk_slow_avg) and a NEGATIVE mk_slow_avg additionally
    suppresses the W4 exploration carve-out (see _leg_edge / the module
    docstring). Defaults (None/0) leave every existing caller byte-identical.

    strike is optional; when provided and present in the PricerSnapshot's
    per-strike sigma2 map, it participates in the bucket worst-case recheck
    (plan C5). Missing strikes fall back to a conservative sum-cap fallback
    for the bucket recheck (see size_ladder). NOTE: as of wave 2 W2 the
    per-strike snapshot.sigma2 map is no longer used for leg shrinkage
    (sigma2_edge instead); strike's only remaining sizing role is the bucket
    recheck.

    unit_skew_x (2026-08-10 skew-fix wave item 2, additive) is the per-share
    reservation shift quote_engine.per_share_skew_x(variant, sigma_b, gamma,
    k, A, tte) would produce for THIS market this tick, variant-correct
    (resolved by the harness -- this module never imports quote_engine).
    Feeds Stage 6b's skew-aware entry cap (see the module docstring).
    Default 0.0 = unwired/disabled -> that stage is inert for this contract
    (all existing callers unaffected).
    """

    market_id: str
    p_hat: float
    bid_price: float
    ask_price: float
    strike: Optional[float] = None
    mk_avg: Optional[float] = None
    mk_var: Optional[float] = None
    mk_n: int = 0
    mk_n_attempted: int = 0
    mk_slow_avg: Optional[float] = None
    mk_slow_n: int = 0
    unit_skew_x: float = 0.0


# ---------------------------------------------------------------------------
# Stage 1 / Stage 2 pure helpers
# ---------------------------------------------------------------------------


def kelly_buy(belief_p: float, price: float) -> Tuple[float, float]:
    """Kelly fraction for buying a binary at `price` with win-probability
    belief `belief_p`. Returns (f_star, b) where b=(1-price)/price is the net
    odds. belief_p <= price (no edge) returns f_star exactly 0."""
    if not (0.0 < price < 1.0):
        return 0.0, 0.0
    b = (1.0 - price) / price
    if belief_p <= price:
        # Exact early-out, not redundant with the f < 0 floor below: at
        # belief_p == price (the m-clamped no-edge case) the f formula
        # leaves +/-1-ulp rounding residue from b, and a positive residue
        # survives the whole sizing pipeline as an ~1e-45-share dust order.
        return 0.0, b
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


def _leg_edge(
    price_leg: float,
    belief_leg: float,
    mk_avg: Optional[float],
    mk_var: Optional[float],
    mk_n: int,
    config: MMConfig,
    mk_slow_avg: Optional[float] = None,
    mk_slow_n: int = 0,
) -> Tuple[float, float, float, bool]:
    """Wave 2 W2 per-leg edge/variance resolution, shared by the YES and NO
    legs (both are computed in the leg's own price scale, so this one
    function serves both -- see the module docstring's explicit per-side
    frame). Returns a 4-tuple (m_gate, m_clamped, sigma2_edge,
    measured_slow_toxic) as of the 2026-08-08 wing-bleed fix:

      structural_edge = belief_leg - price_leg
      m_prior = structural_edge - config.eps_base
      measured_mid = mk_avg is not None and mk_n >= config.markout_min_n
      baseline = mk_avg if measured_mid else m_prior
      slow_h_on = config.markout_slow_horizon_s > 0     # kill switch
      measured_slow = slow_h_on and mk_slow_avg is not None
                      and mk_slow_n >= config.markout_min_n
      m_gate = min(baseline, mk_slow_avg) if measured_slow else baseline
                                                     # UNclamped, W4 floor gate;
                                                     # min() = one-directional
      m_clamped = max(m_gate, 0.0)                   # Glosten-Milgrom floor at 0
      sigma2_edge = (mk_var / mk_n) if (measured_mid and mk_var is not None)
                    else config.markout_prior_var    # slow NEVER sets sigma2
      measured_slow_toxic = measured_slow and mk_slow_avg < 0.0

    measured_slow_toxic is the single home of the slow-toxicity predicate
    (round-5 major 2: never re-derived in size_ladder) -- it suppresses the
    W4 exploration carve-out downstream.
    """
    structural_edge = belief_leg - price_leg
    m_prior = structural_edge - config.eps_base
    measured_mid = mk_avg is not None and mk_n >= config.markout_min_n
    baseline = mk_avg if measured_mid else m_prior
    slow_h_on = float(getattr(config, "markout_slow_horizon_s", 0.0) or 0.0) > 0.0
    measured_slow = (slow_h_on and mk_slow_avg is not None
                     and mk_slow_n >= config.markout_min_n)
    m_gate = min(baseline, mk_slow_avg) if measured_slow else baseline
    m_clamped = max(m_gate, 0.0)
    # sigma2_edge UNCHANGED by the slow channel: mid channel's mk_var/mk_n
    # when measured_mid, else markout_prior_var. The slow channel NEVER sets
    # sigma2.
    if measured_mid and mk_var is not None and mk_n > 0:
        sigma2_edge = mk_var / mk_n
    else:
        sigma2_edge = config.markout_prior_var
    measured_slow_toxic = measured_slow and mk_slow_avg < 0.0
    return m_gate, m_clamped, sigma2_edge, measured_slow_toxic


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
    m_gate: float = 0.0  # wave 2 W4: measured net edge (unclamped) used for the floor gate
    sigma2_edge: float = 0.0  # wave 2 W2: markout-measurement variance used for Baker-McHale
    measured_slow_toxic: bool = False  # 2026-08-08: slow channel measured AND negative (_leg_edge)
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

    # Stage 1 + 2 (wave 2 W2): build legs (YES via bid, NO via ask). Edge is
    # now the POSTED-quote edge net of measured markout (see _leg_edge and the
    # module docstring) -- mkt_mid is gone. Baker-McHale sigma2 is the
    # markout-measurement variance of the SAME leg's net edge (sigma2_edge),
    # not the pricer's per-strike/ladder MC-SE.
    legs: List[_Leg] = []
    for c in contracts:
        yes_price = c.bid_price
        yes_belief = c.p_hat
        m_gate_yes, m_yes, sigma2_yes, slow_toxic_yes = _leg_edge(
            yes_price, yes_belief, c.mk_avg, c.mk_var, c.mk_n, config,
            mk_slow_avg=c.mk_slow_avg, mk_slow_n=c.mk_slow_n,
        )
        f_yes, b_yes = kelly_buy(yes_price + m_yes, yes_price)
        k_yes = baker_mchale(f_yes, b_yes, sigma2_yes)
        legs.append(
            _Leg(c.market_id, True, c.strike, yes_price, f_yes, b_yes, k_yes, f_yes * k_yes,
                 m_gate=m_gate_yes, sigma2_edge=sigma2_yes,
                 measured_slow_toxic=slow_toxic_yes)
        )

        no_price = 1.0 - c.ask_price
        no_belief = 1.0 - c.p_hat
        m_gate_no, m_no, sigma2_no, slow_toxic_no = _leg_edge(
            no_price, no_belief, c.mk_avg, c.mk_var, c.mk_n, config,
            mk_slow_avg=c.mk_slow_avg, mk_slow_n=c.mk_slow_n,
        )
        f_no, b_no = kelly_buy(no_price + m_no, no_price)
        k_no = baker_mchale(f_no, b_no, sigma2_no)
        legs.append(
            _Leg(c.market_id, False, c.strike, no_price, f_no, b_no, k_no, f_no * k_no,
                 m_gate=m_gate_no, sigma2_edge=sigma2_no,
                 measured_slow_toxic=slow_toxic_no)
        )
    audit["stages"].append(
        {"stage": "kelly+baker_mchale",
         "f": [(lg.market_id, lg.is_yes, lg.f) for lg in legs],
         "sigma2_edge": [(lg.market_id, lg.is_yes, lg.sigma2_edge) for lg in legs]}
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

    # Stage 5: presence floor (plan C3), GATED on measured net edge as of
    # wave 2 W4. A pure floor when gated on: only ever raises a leg's
    # shares. presence_frac<=0 disables (floor contributes 0) independent of
    # the gate.
    #
    # gate = (m_gate >= 0.0) or (n_attempted < markout_min_n
    #                            and not measured_slow_toxic)   -- W4 + 2026-08-08
    # m_gate is the leg's UNclamped measured/prior net edge (set in Stage 1);
    # the exploration carve-out (n_attempted below the trust threshold) keeps
    # the floor ON regardless of m_gate's sign so a never-measured cell can
    # accumulate fills. mk_n_attempted is per-CONTRACT (ContractSizingInput),
    # shared by both legs of that market (one measurement cell), so we look
    # it up by market_id via the original contracts, not per-leg.
    #
    # 2026-08-08 wing-bleed fix (ONE deliberate change, round-4 major 3): the
    # carve-out is suppressed while the slow channel is measured-toxic.
    # EQUIVALENCE (round-5 major 3): since measured_slow_toxic =>
    # mk_slow_avg < 0 => m_gate = min(baseline, mk_slow_avg) < 0, the new
    # gate is exactly "if slow-toxic -> floor OFF unconditionally; otherwise
    # unchanged". Rationale: without it, item 4's gate closure has a 28-DAY
    # PERIOD -- no wing fills -> the wing cells age out of MARKOUT_LOOKBACK_S
    # -> n_attempted falls below min_n -> carve-out re-arms -> ~20
    # exploration fills of tuition (~-12 at the observed -0.6/fill) ->
    # re-measure -> close -> repeat. The W4 rationale ("fills are the only
    # calibration source") does not apply when a second channel IS measured.
    # No deadlock: a fill-less slow cell also ages out at 28d, restoring the
    # carve-out. n_attempted stays the MID cell's; stage 5b is unchanged (its
    # 0.33x is moot on a floor-gated-off leg: measured_slow_toxic ->
    # m_clamped = 0 -> f* = 0 -> shares 0).
    mk_n_attempted_by_market: Dict[str, int] = {c.market_id: c.mk_n_attempted for c in contracts}
    inv_by_market: Dict[str, Any] = dict(inventory.per_contract) if inventory is not None else {}
    presence_info: List[Dict[str, Any]] = []
    # s_presence_by_leg: the UNtapered wave-1 floor unit per leg, needed by
    # both the (tapered, gated) presence floor below and the (untapered,
    # ungated) reduce-side exemption (Stage 5b) -- computed once here.
    s_presence_by_leg: Dict[Tuple[str, bool], float] = {}
    for lg in legs:
        if lg.price_per_share > 0.0:
            s_presence_by_leg[(lg.market_id, lg.is_yes)] = (
                config.presence_frac * bankroll / lg.price_per_share
            )

    if config.presence_frac > 0.0:
        for lg in legs:
            s_presence = s_presence_by_leg.get((lg.market_id, lg.is_yes))
            if s_presence is None:
                continue
            n_attempted = mk_n_attempted_by_market.get(lg.market_id, 0)
            gate = (lg.m_gate >= 0.0) or (
                n_attempted < config.markout_min_n and not lg.measured_slow_toxic)
            taper = 1.0
            cinv = inv_by_market.get(lg.market_id)
            if cinv is not None:
                if cinv.q_max <= 0.0:
                    taper = 0.0
                else:
                    q_toward_side = cinv.q if lg.is_yes else -cinv.q
                    taper = max(0.0, min(1.0, 1.0 - q_toward_side / cinv.q_max))
            presence_shares = (s_presence * taper) if gate else 0.0
            if presence_shares > lg.shares:
                lg.shares = presence_shares
            presence_info.append(
                {"market_id": lg.market_id, "is_yes": lg.is_yes,
                 "presence_shares": presence_shares, "taper": taper, "gate": gate}
            )
    audit["stages"].append({"stage": "presence_floor", "legs": presence_info})

    # Stage 5b: unmeasured-cell size multiplier (Fix 2b). A cell not yet
    # trusted-measured (mk_n_attempted < markout_min_n) quotes at
    # config.unmeasured_size_mult of full size so exploration tuition drops
    # ~3x, EXCEPT its reduce side (a multiplied reduce leg would unwind ~3x
    # slower -- Stage 5c's reduce floor is only presence-sized). Runs BEFORE
    # the reduce-side exemption and all caps (6/7/8), which still dominate.
    # depth_cap_floor_shares (1.0) is a proxy for VenueDescriptor.min_size
    # (1.0), which sizing cannot see: a leg already >= that floor pre-mult is
    # floored back up to it so the side is not silently no-quoted (learning
    # deadlock); a leg below the floor pre-mult keeps shares*mult (never
    # resurrect). mult=1.0 disables (sizes byte-identical to pre-2b).
    unmeasured_info: List[Dict[str, Any]] = []
    mult = config.unmeasured_size_mult
    if mult != 1.0:
        for lg in legs:
            n_attempted = mk_n_attempted_by_market.get(lg.market_id, 0)
            if n_attempted >= config.markout_min_n:
                continue  # trusted-measured cell -- full size
            cinv = inv_by_market.get(lg.market_id)
            if cinv is not None and cinv.q != 0.0:
                # q > 0 (net long YES): ask/NO leg is the reduce side.
                # q < 0 (net long NO / short YES): bid/YES leg is the reduce side.
                is_reduce_side = (not lg.is_yes) if cinv.q > 0.0 else lg.is_yes
                if is_reduce_side:
                    continue  # reduce side exempt (unwinds at full size; see 5c)
            pre = lg.shares
            scaled = pre * mult
            if pre >= config.depth_cap_floor_shares:
                scaled = max(scaled, config.depth_cap_floor_shares)
            lg.shares = scaled
            if scaled != pre:
                unmeasured_info.append(
                    {"market_id": lg.market_id, "is_yes": lg.is_yes, "pre": pre, "post": scaled}
                )
    audit["stages"].append({"stage": "unmeasured_mult", "legs": unmeasured_info})

    # Stage 5c: reduce-side exemption (wave 2 W3), UNGATED -- shedding
    # inventory is utility-positive via a risk-relief term cash-EV Kelly
    # ignores, so this floor applies regardless of the W4 gate above. Applied
    # in share space at the same pipeline position as the presence floor
    # (before headroom/depth/bucket caps -- caps still dominate).
    reduce_info: List[Dict[str, Any]] = []
    if inventory is not None:
        for lg in legs:
            cinv = inv_by_market.get(lg.market_id)
            if cinv is None or cinv.q == 0.0:
                continue
            # q > 0 (net long YES): the ask/NO leg is the reduce side.
            # q < 0 (net long NO / short YES): the bid/YES leg is the reduce side.
            is_reduce_side = (not lg.is_yes) if cinv.q > 0.0 else lg.is_yes
            if not is_reduce_side:
                continue
            s_presence = s_presence_by_leg.get((lg.market_id, lg.is_yes))
            if s_presence is None:
                continue
            reduce_floor = min(abs(cinv.q), s_presence)
            if reduce_floor > lg.shares:
                lg.shares = reduce_floor
                reduce_info.append(
                    {"market_id": lg.market_id, "is_yes": lg.is_yes,
                     "reduce_floor": reduce_floor, "q": cinv.q}
                )
    audit["stages"].append({"stage": "reduce_side_exemption", "legs": reduce_info})

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

    # Stage 6b: skew-aware entry cap (2026-08-10 skew-fix wave item 2) --
    # caps ONLY the side of each leg that GROWS |q| (bid when q>=0, ask when
    # q<0). Slotted with Stage 6 (same discipline: caps dominate floors),
    # after the 5b multiplier and 5c reduce floor, before Stage 7 depth /
    # Stage 8 bucket. See the module docstring's "6b" paragraph for the full
    # rationale. Guards, BOTH per-leg (unit_skew_x <= 0 -- unwired/disabled
    # for that contract) and pipeline-global (config.skew_x_cap <= 0 -- the
    # kill switch paired with the quote-engine clamp): either leaves the leg
    # exactly as Stage 6 left it (shares AND max_add_yes/no unchanged).
    skew_x_cap_cfg = float(getattr(config, "skew_x_cap", 0.0) or 0.0)
    skew_headroom_mult = float(getattr(config, "skew_q_headroom_mult", 1.0) or 1.0)
    skew_info: List[Dict[str, Any]] = []
    if inventory is not None and skew_x_cap_cfg > 0.0:
        contract_by_market: Dict[str, ContractSizingInput] = {
            c.market_id: c for c in contracts
        }
        for lg in legs:
            cinv = inv_by_market.get(lg.market_id)
            if cinv is None:
                continue
            c_in = contract_by_market.get(lg.market_id)
            unit_skew = c_in.unit_skew_x if c_in is not None else 0.0
            if unit_skew <= 0.0:
                continue  # unwired/disabled for this contract -- stage inert
            # q_skew_max: inventory level at which the per-share skew term
            # alone reaches the quote-engine's clamp, scaled by the headroom
            # multiplier (some saturation tolerated -- fills are the
            # calibration source and the venue-min floor must stay
            # reachable).
            q_skew_max = skew_headroom_mult * skew_x_cap_cfg / unit_skew
            if lg.is_yes:
                bound = max(0.0, q_skew_max - cinv.q)
                # Journal truthfulness: report min(Stage-6, this bound) only
                # here (this leg is active) -- max_add_yes was already set by
                # Stage 6 above for every leg with cinv present.
                max_add_yes[lg.market_id] = min(
                    max_add_yes.get(lg.market_id, bound), bound
                )
            else:
                bound = max(0.0, q_skew_max + cinv.q)
                max_add_no[lg.market_id] = min(
                    max_add_no.get(lg.market_id, bound), bound
                )
            if lg.shares > bound + 1e-12:
                lg.shares = bound
                triggered.add(SizingCap.SKEW)
            skew_info.append(
                {"market_id": lg.market_id, "is_yes": lg.is_yes,
                 "unit_skew_x": unit_skew, "q_skew_max": q_skew_max, "bound": bound}
            )
    audit["stages"].append(
        {"stage": "skew_entry_cap", "legs": skew_info,
         "skew_x_cap": skew_x_cap_cfg, "skew_q_headroom_mult": skew_headroom_mult}
    )

    # Stage 7: depth cap -- hard min, AFTER the floor and inventory cap
    # (CRITICAL: must stay a hard min here so a pre-existing depth-binding
    # size is unaffected by the floor's max()). FLOORED as of wave 2 W5: the
    # bound is max(realized_depth_side, config.depth_cap_floor_shares), not
    # raw realized_depth_side -- a dead book (realized_depth=0) no longer
    # permanently zeroes size; depth control is for impact, not presence.
    if liquidity is not None:
        for lg in legs:
            liq = liquidity.get(lg.market_id)
            if liq is None:
                continue
            if lg.is_yes:
                depth_bound = max(liq.realized_depth_bid, config.depth_cap_floor_shares)
                if lg.shares > depth_bound + 1e-12:
                    lg.shares = depth_bound
                    triggered.add(SizingCap.DEPTH)
            else:
                depth_bound = max(liq.realized_depth_ask, config.depth_cap_floor_shares)
                if lg.shares > depth_bound + 1e-12:
                    lg.shares = depth_bound
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
