"""Spread builder (plan Section 2.5, task S1, contract 4.5).

Composes the final half-spread additively per side, in probability units, from
six terms, then enforces floor/clamp/quantize/no-cross, in that order:

1. base arrival markup: 1/k_arrival is a delta_x (log-odds half-spread);
   converted to p-units at the quote center (proposal.r_x) via the EXACT
   two-point conversion (logodds.half_spread_p_exact), not the Jacobian
   linearization, so it stays correct near the p-clamps.
2. adverse-selection buffer: MMConfig.eps_base + RiskDirective.eps_add
   (already in p-units).
3. inventory skew: NOT re-applied here -- it is already embedded in
   proposal.x_bid/x_ask (which this builder centers on). Reported only, as the
   exact p-space displacement caused by proposal.skew_x:
   S(r_x) - S(r_x - skew_x).
4. robust widening: robust_scale*sqrt(sigma2) + (1-credibility)*credibility_widen_scale.
   MMConfig has no fields for these two scales (plan note: "add module
   defaults if MMConfig lacks fields") so they are module constants below,
   overridable per call.
5. wing/tail widening: outside MMConfig.belly_band, add a base wing width
   (module constant, same rationale as term 4) scaled by
   MMConfig.wing_widen_scale[confidence_tier].
6. belly widening: inside MMConfig.belly_band (exact complement of term 5, so
   exactly one of wing/belly fires per quote), add a flat base plus a slope
   applied to time-to-expiry beyond a free-days window
   (MMConfig.belly_widen_base_p / belly_widen_slope_p_per_day /
   belly_widen_free_days) -- the belly is the model's softest region per
   temp/suitability.md (+4.8c bias at 1-2d growing to +8.6c at 5-7d).

Terms 2, 4, 5, 6 are symmetric (applied to bid down / ask up); terms 1 and 3
are audit-only (already embedded in the proposal's x_bid/x_ask, reported in
`terms` for decomposition, never added to the widening -- see the term-1
inline comment). Composition mechanics per plan 2.5: widen -> floor half-spread to
>= 1 tick -> clamp to the venue price band -> tick-quantize (floor bid, ceil
ask, so quantization never shrinks the spread) -> resolve any crossing left by
quantization by widening the ask one tick.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from market_maker.config import MMConfig, in_belly_band
from market_maker.contracts import (
    ConfidenceTier,
    LiquidityState,
    QuoteMode,
    QuoteProposal,
    QuoteSet,
    RiskDirective,
    SizingCap,
    SizingDecision,
    VenueDescriptor,
)
from market_maker.logodds import floor_half_spread, half_spread_p_exact, sigmoid

# Module defaults for terms 4/5 (plan 2.5: "add module defaults if MMConfig
# lacks fields; keep configurable" -- overridable per call to build_quote_set).
DEFAULT_ROBUST_SCALE: float = 1.0
# 2026-07-11 zero-fill recal: credibility widening was contributing
# ~1.4c/side at credibility~0.3 (the dominant robust-term component; MC-SE
# sqrt(sigma2) is sub-cent in the belly), and the wing base another 1c/side
# -- against a 0.2-1c market half-touch. Halved both; further cuts should
# come from measured fill markouts, not guesses.
DEFAULT_CREDIBILITY_WIDEN_SCALE: float = 0.01  # prob units at credibility=0 (0.02 -> 0.01, 2026-07-11)
DEFAULT_WING_BASE_P: float = 0.005  # prob units, before confidence-tier scaling (0.01 -> 0.005, 2026-07-11)


def make_stub_directive(market_id: str, ts: datetime) -> RiskDirective:
    """Stub RiskDirective per plan/task: TWO_SIDED, zero eps_add, until R1 lands."""
    return RiskDirective(
        ts=ts,
        market_id=market_id,
        mode=QuoteMode.TWO_SIDED,
        eps_add=0.0,
        kelly_mult=1.0,
        triggers=[],
        latched_until=ts,
        cancel_all=False,
    )


def make_stub_sizing(market_id: str, ts: datetime, bid_size: float = 10.0, ask_size: float = 10.0) -> SizingDecision:
    """Stub SizingDecision per task: fixed sizes, until Z1 lands."""
    return SizingDecision(
        ts=ts,
        market_id=market_id,
        bid_size=bid_size,
        ask_size=ask_size,
        f_kelly=0.0,
        k_shrink=1.0,
        ladder_alloc=0.0,
        caps_applied=[SizingCap.FRACTIONAL_C],
        sigma2_used=0.0,
        phi_directive=0.0,
    )


def _quantize(bid_price: float, ask_price: float, tick: float, lo: float, hi: float) -> tuple[float, float]:
    if tick > 0:
        eps = 1e-9
        bid_price = math.floor(bid_price / tick + eps) * tick
        ask_price = math.ceil(ask_price / tick - eps) * tick
    bid_price = min(max(bid_price, lo), hi)
    ask_price = min(max(ask_price, lo), hi)
    if bid_price >= ask_price:
        ask_price = min(bid_price + tick, hi)
        if bid_price >= ask_price:
            bid_price = max(ask_price - tick, lo)
    return bid_price, ask_price


def build_quote_set(
    proposal: QuoteProposal,
    directive: RiskDirective,
    sizing: SizingDecision,
    venue: VenueDescriptor,
    config: MMConfig,
    sigma2: float,
    confidence_tier: ConfidenceTier,
    credibility: float,
    consensus_p: float,
    source_seq: int,
    liquidity: Optional[LiquidityState] = None,  # reserved; sizes come from `sizing`
    robust_scale: float = DEFAULT_ROBUST_SCALE,
    credibility_widen_scale: float = DEFAULT_CREDIBILITY_WIDEN_SCALE,
    wing_base_p: float = DEFAULT_WING_BASE_P,
    ts: Optional[datetime] = None,
    tte_days: Optional[float] = None,
) -> QuoteSet:
    p_lo, p_hi = config.p_clamp
    ts_final = ts if ts is not None else proposal.ts

    # term 1: base arrival markup — AUDIT-ONLY, like the skew term. The quote
    # engine's delta_x already carries the arrival component ((2/k)ln(1+g/k),
    # Dalen Eq 9), embedded in proposal.x_bid/x_ask; re-adding a raw 1/kappa
    # here double-counted arrival and dominated the half-spread (~23c at the
    # belly with launch k=1 — Stage-A shadow finding 2026-07-07). Reported in
    # `terms` for decomposition, never added to the widening.
    k = max(config.k_arrival, 1e-9)
    markup_x = 1.0 / k
    markup_p = half_spread_p_exact(proposal.r_x, markup_x, p_lo, p_hi)

    # term 2: adverse-selection buffer
    eps_p = config.eps_base + directive.eps_add

    # term 3: skew, audit-only (already embedded in proposal.x_bid/x_ask)
    skew_p = sigmoid(proposal.r_x, p_lo, p_hi) - sigmoid(proposal.r_x - proposal.skew_x, p_lo, p_hi)

    # term 4: robust widening
    robust_p = robust_scale * math.sqrt(max(sigma2, 0.0)) + (1.0 - credibility) * credibility_widen_scale

    # terms 5/6: wing/tail vs belly widening share one membership test
    # (config.in_belly_band, F7) -- wing = NOT in belly, belly = in belly, so
    # exactly one of the two fires per quote by construction.
    in_belly = in_belly_band(consensus_p, config.belly_band)

    # term 5: wing/tail widening (launch default)
    if not in_belly:
        wing_p = wing_base_p * config.wing_widen_scale.get(confidence_tier, 1.0)
    else:
        wing_p = 0.0

    # term 6: belly widening (temp/suitability.md). Flat base inside the free-
    # days window; base + slope*(tte-free) beyond it. tte_days=None (caller did
    # not pass it) falls back to base only, for back-compat.
    if in_belly:
        if tte_days is not None:
            belly_p = config.belly_widen_base_p + config.belly_widen_slope_p_per_day * max(
                0.0, tte_days - config.belly_widen_free_days
            )
        else:
            belly_p = config.belly_widen_base_p
    else:
        belly_p = 0.0

    widen = eps_p + robust_p + wing_p + belly_p  # markup + skew live in the proposal

    p_bid_center = sigmoid(proposal.x_bid, p_lo, p_hi)
    p_ask_center = sigmoid(proposal.x_ask, p_lo, p_hi)
    bid_price = p_bid_center - widen
    ask_price = p_ask_center + widen

    half_spread_pre = 0.5 * (ask_price - bid_price)
    half_spread_floored = floor_half_spread(half_spread_pre, venue.tick_size)
    floor_applied = 1.0 if half_spread_floored > half_spread_pre + 1e-12 else 0.0
    center = 0.5 * (bid_price + ask_price)
    bid_price = center - half_spread_floored
    ask_price = center + half_spread_floored

    band_lo, band_hi = venue.price_band
    bid_price, ask_price = _quantize(bid_price, ask_price, venue.tick_size, band_lo, band_hi)

    bid_size = sizing.bid_size
    ask_size = sizing.ask_size
    if directive.mode == QuoteMode.BID_ONLY:
        ask_size = 0.0
    elif directive.mode == QuoteMode.ASK_ONLY:
        bid_size = 0.0
    elif directive.mode == QuoteMode.PULLED:
        bid_size = 0.0
        ask_size = 0.0

    terms = {
        "markup": markup_p,
        "eps": eps_p,
        "skew": skew_p,
        "robust": robust_p,
        "wing": wing_p,
        "belly": belly_p,
        "floor_applied": floor_applied,
    }

    return QuoteSet(
        ts=ts_final,
        market_id=proposal.market_id,
        bid_price=bid_price,
        ask_price=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
        terms=terms,
        risk_mode=directive.mode,
        noarb_checked=False,
        source_seq=source_seq,
    )
