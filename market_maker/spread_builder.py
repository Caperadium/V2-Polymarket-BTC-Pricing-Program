"""Spread builder (plan Section 2.5, task S1, contract 4.5).

Composes the final half-spread additively per side, in probability units, from
seven terms, then enforces floor/clamp/quantize/no-cross, in that order:

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
7. markout-fed widening (package E, 2026-07-15, SIDE-ASYMMETRIC): unlike
   terms 2/4/5/6 (symmetric, applied to both sides equally), term 7 is two
   independent quantities -- `markout_widen_bid` off the BUY_YES (our bid)
   side's measured markout, `markout_widen_ask` off the BUY_NO (our ask)
   side's -- each the output of `markout_widen()` (clamp(-mk_avg, 0, cap) *
   scale) applied to `pnl_report.markout_stats_side` at
   `MMConfig.markout_widen_horizon_s` (60s, deliberately different from
   sizing's 600s `markout_horizon_s` -- see `markout_widen`'s docstring).
   Sizing already trusts measured markout (posted-edge/markout Kelly, wave
   2); this closes the reverse direction -- widen the QUOTE on the side that
   is measurably getting picked off, mechanically, per the standing doctrine
   "remaining spread floor cut only against measured fill markouts". Rebate
   is deliberately NOT netted against the widening (conservative; the
   accounting layer is display-only everywhere else too). Terms 2/4/5/6/7 are
   applied to `compute_posted_prices`'s center BEFORE floor/clamp/quantize;
   `markout_bid`/`markout_ask` land in `terms` as additive audit entries.

   No-arb PAV repair interaction (accepted, pinned by the plan): term 7 is
   piecewise-constant across the ladder within a tick (one belly value, one
   wing value per side), so widening one region's bid while a neighboring
   region's bid stays put can invert the ladder's non-increasing-bid
   invariant at a region boundary. The mandatory `LadderHedger.repair` PAV
   step then pools the violating neighbors to their (weighted) average. This
   is accepted, not a bug: PAV redistributes widening across the pooled
   neighbors but never removes net widening (a pool average preserves the
   sum), and no pooled bid can ever exceed its pre-widening baseline -- for
   pre-widening monotone bids b0 >= b1 with only b0 widened by w, the pooled
   average (b0 - w + b1) / 2 <= b0 - w/2 < b0 (symmetric argument for asks,
   which only ever move UP under widening so a pooled average cannot fall
   below its pre-widening baseline). Every belly quote still moves in the
   intended direction post-repair; it just may not carry the FULL local
   widening amount at a region boundary.

   CHARACTERIZATION (found while implementing package E's required
   PAV-interaction test, tests/test_mm_harness_ws1.py): the baseline-relative
   bound above holds unconditionally and is what every consumer of term 7
   should rely on. Full no-arb RESTORATION (LadderHedger.check().ok == True
   after one repair() pass) is a SEPARATE, stronger property that is only
   guaranteed at strike spacing wide enough that the natural adjacent-strike
   bid gap exceeds markout_widen_cap (true at realistic Polymarket BTC daily
   spacing with the launch-default cap, confirmed by test). At pathologically
   tight strike spacing, `LadderHedger.repair()`'s "pool mid via PAV, then
   reconstruct bid/ask from each market's OWN preserved half-spread" method
   can leave a residual bid-monotonicity violation even after pooling,
   because term 7's asymmetric (bid-only or ask-only) widening inflates only
   the widened market's own half-spread -- confirmed non-convergent even
   under repeated repair() calls. This is a structural property of
   `ladder_hedger.py` (out of scope for package E), not something this
   module works around.

   UPDATE (2026-08-13 bleed-2 fix, item 2): the post-only book clamp
   (`post_only_clamp`, applied by the harness AFTER this restoration, per
   quote/per market) can move a repaired ladder's prices further OUTWARD
   (bid down, ask up) to stay off the venue's opposite touch. This can
   reintroduce an ASK-ladder monotonicity violation (a lower-strike ask
   clamped up past a higher-strike ask left untouched) but can NEVER create
   the exploitable `ask_K < bid_{K+1}` crossing: per strike, bid < ask
   always holds, and both the post-clamp bid and ask move outward (bid down
   or unchanged, ask up or unchanged) from their already-repaired values, so
   `bid_{K+1}^new <= bid_{K+1}^old < ask_K^old <= ask_K^new`. "Full no-arb
   RESTORATION (LadderHedger.check().ok == True after one repair() pass)"
   above therefore describes the ladder as `_compose_quote_sets` +
   `LadderHedger.repair()` hand it off, not necessarily the final ladder
   that reaches the lifecycle -- `QuoteSet.noarb_checked` documents this
   distinction (contracts.py) and the clamp is never re-checked against
   `LadderHedger.check()` (a harmless post-clamp violation would otherwise
   pollute the very signal `noarb_checked` protects).

Terms 2, 4, 5, 6, 7 are symmetric-per-side (term 7 asymmetric between bid and
ask, but each side's amount is applied independently, same as the others);
terms 1 and 3 are audit-only (already embedded in the proposal's x_bid/x_ask,
reported in `terms` for decomposition, never added to the widening -- see the
term-1 inline comment). Composition mechanics per plan 2.5: widen -> floor
half-spread to >= 1 tick -> clamp to the venue price band -> tick-quantize
(floor bid, ceil ask, so quantization never shrinks the spread) -> resolve any
crossing left by quantization by widening the ask one tick.

Deliberate basis inconsistencies (canonical region-basis enumeration,
referenced from harness._compose_quote_sets):

- QUOTING region (the wing/belly terms 5/6, term 7's side-split lookups, and
  the region-appropriate credibility in term 4) is classified from the Beuoy
  CONSENSUS p of the strike being quoted (harness `region`). Widening is
  unconditionally protective, so it needs no alignment with any measurement
  cell.
- The Beuoy ANCHOR's own region map (fair_value_anchor, per-region bankroll
  updates + the wing pricer-weight pin) classifies from the SANITIZED MARKET
  ladder vs belly_band -- never from the consensus being built.
- The markout REPORT tags each fill's region from the fill's OWN recorded
  book mid (`mid_at_fill`, pnl_report) -- the measurement basis.
- SIZING region (item 4, 2026-08-08 wing-bleed fix) is classified from the
  market's live BOOK MID via harness._market_mid (consensus only as the
  empty-book fallback), latched with hysteresis -- NOT from consensus: the
  W4 exploration gate and the Kelly markout haircut read a
  (region, tte-bucket) cell, and the fills that feed that cell are tagged by
  the report's mid basis above, so only the mid basis makes "the cell the
  gate checks" and "the cell the fills feed" the same cell in BOTH
  directions. The consensus basis let a pricer-rich consensus (~0.21)
  classify a mid-0.13 market "belly" for the gate while its fills measured
  into the WING cell, so the exploration faucet could never close (the
  2026-08-08 wing bleed).
- The sizing markout HORIZONS (600s mid channel + 21600s slow channel) also
  deliberately differ from term 7's 60s widening horizon -- net edge for
  Kelly vs pick-off signal for widening (see MMConfig.markout_widen_horizon_s).
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from datetime import datetime
from typing import Dict, Optional, Tuple

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

logger = logging.getLogger("mm.spread_builder")

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


def markout_widen(mk_avg: Optional[float], scale: float, cap: float) -> float:
    """Term 7 widening amount for ONE side, off that side's measured markout
    (package E, 2026-07-15): ``0.0`` if ``mk_avg`` is None (no trusted
    measurement yet -- degrade to inert, matching every other markout-gated
    consumer's cold-start behavior), else ``clamp(-mk_avg, 0, cap) * scale``.

    A NEGATIVE ``mk_avg`` (we are measurably getting picked off on this side)
    widens; a positive or zero ``mk_avg`` (favorable or neutral markout)
    widens by exactly 0.0 -- this is deliberately one-directional, there is no
    symmetric "tighten on good markout" branch. Rebate is deliberately NOT
    netted against ``mk_avg`` here (conservative; consistent with the rest of
    the maker-rebate accounting layer, which is display-only everywhere else
    too -- see ``pnl_report``'s "Maker rebates" module docstring section).
    """
    if mk_avg is None:
        return 0.0
    return max(0.0, min(-mk_avg, cap)) * scale


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


def compute_posted_prices(
    proposal: QuoteProposal,
    directive: RiskDirective,
    venue: VenueDescriptor,
    config: MMConfig,
    sigma2: float,
    confidence_tier: ConfidenceTier,
    credibility: float,
    consensus_p: float,
    robust_scale: float = DEFAULT_ROBUST_SCALE,
    credibility_widen_scale: float = DEFAULT_CREDIBILITY_WIDEN_SCALE,
    wing_base_p: float = DEFAULT_WING_BASE_P,
    tte_days: Optional[float] = None,
    markout_widen_bid: float = 0.0,
    markout_widen_ask: float = 0.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Price-building half of build_quote_set (wave 2 W1 split): the seven
    additive spread terms plus floor/clamp/quantize/no-cross composition.
    Pure function of (proposal, directive, venue, config, sigma2,
    confidence_tier, credibility, consensus_p) -- no sizing, no QuoteSet.
    Returns (bid_price, ask_price, terms); `terms` is the same audit dict
    build_quote_set has always returned on its QuoteSet.

    `markout_widen_bid` / `markout_widen_ask` (package E, term 7, default
    0.0 -- byte-identical to pre-package-E behavior): the caller's already-
    resolved `spread_builder.markout_widen()` output for the BUY_YES (bid)
    and BUY_NO (ask) side respectively, applied ASYMMETRICALLY (unlike every
    other term, which widens both sides by the same amount) directly onto
    the per-side center price below, BEFORE the floor/clamp/quantize steps --
    the floor step recomputes the center from these already-asymmetric
    prices, so the center shift survives flooring; when the floor binds, the
    per-side amounts collapse to a symmetric half-spread around that shifted
    center, so the directional signal is preserved even in the degenerate
    case. See module docstring term 7 for the full rationale and the no-arb
    PAV repair interaction.
    """
    p_lo, p_hi = config.p_clamp

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
    # term 7: markout-fed widening (package E) -- SIDE-ASYMMETRIC, unlike the
    # symmetric terms above; each side's amount is independently resolved by
    # the caller via markout_widen() and passed in (default 0.0, inert).
    bid_price = p_bid_center - (widen + markout_widen_bid)
    ask_price = p_ask_center + (widen + markout_widen_ask)

    half_spread_pre = 0.5 * (ask_price - bid_price)
    half_spread_floored = floor_half_spread(half_spread_pre, venue.tick_size)
    floor_applied = 1.0 if half_spread_floored > half_spread_pre + 1e-12 else 0.0
    center = 0.5 * (bid_price + ask_price)
    bid_price = center - half_spread_floored
    ask_price = center + half_spread_floored

    band_lo, band_hi = venue.price_band
    bid_price, ask_price = _quantize(bid_price, ask_price, venue.tick_size, band_lo, band_hi)

    terms = {
        "markup": markup_p,
        "eps": eps_p,
        "skew": skew_p,
        "robust": robust_p,
        "wing": wing_p,
        "belly": belly_p,
        "markout_bid": markout_widen_bid,
        "markout_ask": markout_widen_ask,
        "floor_applied": floor_applied,
    }

    return bid_price, ask_price, terms


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
    posted: Optional[Tuple[float, float, Dict[str, float]]] = None,
) -> QuoteSet:
    ts_final = ts if ts is not None else proposal.ts

    # wave 2 W1: posted prices are computed by compute_posted_prices, either
    # here (posted=None, bit-identical to pre-wave-2 behavior) or upstream by
    # the caller (harness sizes on the posted prices, then passes them back
    # in via `posted` so they are not recomputed).
    if posted is None:
        bid_price, ask_price, terms = compute_posted_prices(
            proposal, directive, venue, config, sigma2, confidence_tier,
            credibility, consensus_p, robust_scale=robust_scale,
            credibility_widen_scale=credibility_widen_scale,
            wing_base_p=wing_base_p, tte_days=tte_days,
        )
    else:
        bid_price, ask_price, terms = posted

    bid_size = sizing.bid_size
    ask_size = sizing.ask_size
    if directive.mode == QuoteMode.BID_ONLY:
        ask_size = 0.0
    elif directive.mode == QuoteMode.ASK_ONLY:
        bid_size = 0.0
    elif directive.mode == QuoteMode.PULLED:
        bid_size = 0.0
        ask_size = 0.0

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


def _usable_ref(v: Optional[float]) -> bool:
    """True iff `v` is usable as a post-only clamp reference: a finite float
    strictly inside the open unit interval (0, 1). Mirrors the None/NaN-safe
    EXPLICIT-check style of `paper_fill_sim._is_num` (never bare `min`/`max`
    against a possibly-NaN value -- that is argument-order-dependent), but
    is intentionally stricter: a clamp reference must be a legal, usable
    price, not merely "not NaN". Deliberately NOT `harness._market_mid`,
    which has no NaN guard of its own.
    """
    if v is None or isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    fv = float(v)
    return math.isfinite(fv) and 0.0 < fv < 1.0


def _round_grid_outward(price: float, tick: float, floor: bool) -> float:
    """Snap `price` outward onto the tick grid: floor for the bid side,
    ceil for the ask side, so the crossing-repair arithmetic in
    `post_only_clamp` never leaves float residue that re-opens the crossing
    it just removed. Deliberately local and separate from
    `ladder_hedger._quantize` (rounds to the NEAREST tick, fixed [0,1] band)
    and this module's own `_quantize` (rounds toward center with its own
    implicit no-cross repair) -- neither matches the outward-only, explicit-
    band semantics this function needs, and `post_only_clamp` handles its
    own band clamp and degenerate-size check as separate, later steps.
    """
    if tick <= 0.0:
        return price
    eps = 1e-9
    if floor:
        return math.floor(price / tick + eps) * tick
    return math.ceil(price / tick - eps) * tick


def post_only_clamp(
    qs: QuoteSet,
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick: float,
    band: Tuple[float, float],
    margin_ticks: int,
) -> QuoteSet:
    """Post-only book clamp (2026-08-13 bleed-2 fix, item 2, structural
    backstop for both bleed-2 faucets -- temp/mm_bleed2_fix_plan.md). Bounds
    EACH side of `qs` to stay `margin_ticks` ticks inside the OPPOSITE venue
    touch: `bid <= best_ask - margin*tick`, `ask >= best_bid + margin*tick`.
    This emulates venue post-only semantics with the minimum behavioral
    delta: LIVE intent is post-only maker orders, and a real post-only order
    that would cross is rejected/repriced by the venue -- but
    `paper_fill_sim` fills a resting-crossed order at OUR OWN crossed price
    with `queue_ahead=0` (see that module's docstring note). This clamp
    removes the crossing before the QuoteSet ever reaches the fill sim.

    Guarantee scope: DESIRED-ladder only. With `requote_price_tol` (a
    1-tick deadband) a resting order may lag this freshly-clamped desired
    price by up to one tick and be crossed INTO by a subsequent book move --
    that is normal maker behavior, matching `order_lifecycle`'s existing
    resting-vs-desired language, not a gap in this clamp.

    Deep-wing note: a venue book whose best_ask sits at the minimum tick
    (e.g. a 1c-wide wing book) forces the post-round, post-band-clamp bid
    below `max(tick, band_lo)` for any positive margin -- the bid side is
    then zeroed (rule 5 below). This is CORRECT, not a bug: there is no
    legal maker bid below the venue's price floor.

    One-sided-book hole (known, recorded, NOT fixed here): when the
    OPPOSITE touch is absent/None/NaN/non-finite/outside (0, 1)
    (`_usable_ref` fails), that side is left UNCLAMPED, by design -- there
    is nothing to cross against. Unbounded-vs-mid exposure therefore
    persists exactly on thin, one-sided wing books; a `post_only_join`
    (join-the-touch) variant is the recorded follow-up (see
    `MMConfig.post_only_margin_ticks`) if the belly-consensus-divergence
    faucet (explicitly out of scope this wave) keeps arming that hole.

    Pure and idempotent: `post_only_clamp(post_only_clamp(qs, ...), ...) ==
    post_only_clamp(qs, ...)`. Returns the SAME `qs` object (identity) when
    neither side needs adjustment. NEVER touches size except the rule-5
    degenerate zeroing below, and NEVER resurrects a side whose size is
    already 0 (rule 6) -- both sides are evaluated independently off `qs`'s
    ORIGINAL bid_size/ask_size, and a directive-suppressed or min-size-
    zeroed side is skipped entirely.

    Per-side rules, in order, mirrored for bid/ask:
      1. reference guard (`_usable_ref`, above).
      2. `new_bid = min(bid_price, best_ask - margin*tick)` (only if
         `bid_size > 0` and `best_ask` usable); mirror for the ask side.
      3. outward grid rounding (`_round_grid_outward`, above).
      4. band clamp into `[band_lo, band_hi]`.
      5. degenerate check on the FINAL value: if the clamped bid is below
         `max(tick, band_lo)`, the side is unquotable as a maker ->
         `bid_size = 0.0`, price left at its OLD valid value (mirror: ask
         above `min(1 - tick, band_hi)` -> `ask_size = 0.0`). The "both
         sides crossed" case is unreachable by construction (both moves are
         outward from an already bid < ask pair) -- logged at debug only,
         no dedicated test budget.
      6. never resurrect (see above).
      7. journal: when a side's price actually moves, the returned
         QuoteSet's `terms` gains `"post_only_bid"` (= old_bid - new_bid,
         always > 0) and/or `"post_only_ask"` (= new_ask - old_ask, always
         > 0), present ONLY when nonzero.

    Forensic decomposition identity: this is applied AFTER `build_quote_set`
    (not inside `compute_posted_prices`), so the module docstring's
    reconstruction identity (`bid = sigmoid(x_bid) - sum(spread terms)`)
    must now ALSO subtract `post_only_bid` when present (mirror: `ask =
    sigmoid(x_ask) + sum(spread terms) + post_only_ask`), or the
    reconstructed price will not match the journaled one.

    `band` is the venue price band (`config.p_clamp`), passed explicitly by
    the caller (`harness.tick`) -- not re-derived from `qs`.
    """
    band_lo, band_hi = band
    margin = margin_ticks * tick

    bid_price, bid_size = qs.bid_price, qs.bid_size
    ask_price, ask_size = qs.ask_price, qs.ask_size
    post_only_bid = 0.0
    post_only_ask = 0.0

    if qs.bid_size > 0.0 and _usable_ref(best_ask):
        candidate = min(qs.bid_price, float(best_ask) - margin)
        candidate = _round_grid_outward(candidate, tick, floor=True)
        candidate = min(max(candidate, band_lo), band_hi)
        if candidate < max(tick, band_lo):
            bid_size = 0.0  # unquotable as a maker; price left at old valid value
        elif candidate < qs.bid_price:
            post_only_bid = qs.bid_price - candidate
            bid_price = candidate

    if qs.ask_size > 0.0 and _usable_ref(best_bid):
        candidate = max(qs.ask_price, float(best_bid) + margin)
        candidate = _round_grid_outward(candidate, tick, floor=False)
        candidate = min(max(candidate, band_lo), band_hi)
        if candidate > min(1.0 - tick, band_hi):
            ask_size = 0.0  # unquotable as a maker; price left at old valid value
        elif candidate > qs.ask_price:
            post_only_ask = candidate - qs.ask_price
            ask_price = candidate

    if bid_price >= ask_price and bid_size > 0.0 and ask_size > 0.0:
        # Unreachable given both moves are outward from an already
        # bid < ask pair (module docstring proof above) -- debug-log only.
        logger.debug(
            "post_only_clamp: unexpected crossing market_id=%s bid=%.6g ask=%.6g",
            qs.market_id, bid_price, ask_price,
        )

    if (bid_price == qs.bid_price and ask_price == qs.ask_price
            and bid_size == qs.bid_size and ask_size == qs.ask_size):
        return qs

    terms = dict(qs.terms)
    if post_only_bid > 0.0:
        terms["post_only_bid"] = post_only_bid
    if post_only_ask > 0.0:
        terms["post_only_ask"] = post_only_ask

    return replace(
        qs, bid_price=bid_price, ask_price=ask_price,
        bid_size=bid_size, ask_size=ask_size, terms=terms,
    )
