"""Tests for market_maker.spread_builder (plan 2.5, task S1, contract 4.5)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import ConfidenceTier, QuoteMode, QuoteProposal, VenueDescriptor
from market_maker.logodds import half_spread_p_exact, logit, sigmoid
from market_maker.spread_builder import (
    DEFAULT_CREDIBILITY_WIDEN_SCALE,
    DEFAULT_ROBUST_SCALE,
    DEFAULT_WING_BASE_P,
    _quantize,
    build_quote_set,
    compute_posted_prices,
    make_stub_directive,
    make_stub_sizing,
)

TS = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
VENUE = VenueDescriptor(
    tick_size=0.01, min_size=1.0, price_band=(0.001, 0.999),
    maker_fee=0.0, maker_rebate=0.0, settlement_rule="12:00 ET", supports_ladder=True,
)


def _proposal(r_x=0.0, delta_x=0.5, skew_x=0.0, market_id="m1"):
    return QuoteProposal(
        ts=TS, market_id=market_id, r_x=r_x, delta_x=delta_x, skew_x=skew_x,
        sigma_b=0.2, params_id="p1", x_bid=r_x - delta_x, x_ask=r_x + delta_x,
        p_bid_raw=sigmoid(r_x - delta_x), p_ask_raw=sigmoid(r_x + delta_x),
    )


def _build(config=None, proposal=None, directive=None, sizing=None, venue=None, **kw):
    config = config or MMConfig()
    proposal = proposal or _proposal()
    directive = directive or make_stub_directive("m1", TS)
    sizing = sizing or make_stub_sizing("m1", TS)
    venue = venue or VENUE
    return build_quote_set(
        proposal, directive, sizing, venue, config,
        sigma2=kw.pop("sigma2", 0.0001),
        confidence_tier=kw.pop("confidence_tier", ConfidenceTier.FULL),
        credibility=kw.pop("credibility", 0.8),
        consensus_p=kw.pop("consensus_p", 0.5),
        source_seq=kw.pop("source_seq", 1),
        **kw,
    )


# ---------------------------------------------------------------------------
# Decomposition: sum of terms + proposal's own half spread == pre-floor half spread
# ---------------------------------------------------------------------------

def test_terms_decomposition_matches_pre_floor_half_spread():
    config = MMConfig()
    proposal = _proposal(r_x=0.0, delta_x=0.3)
    directive = make_stub_directive("m1", TS)
    sizing = make_stub_sizing("m1", TS)
    fine_venue = VenueDescriptor(
        tick_size=1e-6, min_size=1.0, price_band=(0.001, 0.999),
        maker_fee=0.0, maker_rebate=0.0, settlement_rule="12:00 ET", supports_ladder=True,
    )
    qs = build_quote_set(
        proposal, directive, sizing, fine_venue, config,
        sigma2=0.0002, confidence_tier=ConfidenceTier.FULL, credibility=0.7,
        consensus_p=0.5, source_seq=1,
    )
    proposal_half_spread = half_spread_p_exact(proposal.r_x, proposal.delta_x, *config.p_clamp)
    # markup is AUDIT-ONLY (arrival already lives in proposal.delta_x — adding
    # it here double-counted arrival; Stage-A shadow finding 2026-07-07).
    expected_half_spread = (
        proposal_half_spread + qs.terms["eps"] + qs.terms["robust"] + qs.terms["wing"] + qs.terms["belly"]
    )
    actual_half_spread = 0.5 * (qs.ask_price - qs.bid_price)
    assert actual_half_spread == pytest.approx(expected_half_spread, abs=1e-4)
    assert qs.terms["markup"] > 0.0  # still reported for decomposition audit


def test_terms_dict_has_all_keys():
    qs = _build()
    for key in ["markup", "eps", "skew", "robust", "wing", "belly", "floor_applied"]:
        assert key in qs.terms


# ---------------------------------------------------------------------------
# eps_add widens
# ---------------------------------------------------------------------------

def test_eps_add_widens_spread():
    base = _build(directive=make_stub_directive("m1", TS))
    directive_wide = make_stub_directive("m1", TS)
    import dataclasses
    directive_wide = dataclasses.replace(directive_wide, eps_add=0.05)
    wide = _build(directive=directive_wide)
    assert wide.terms["eps"] > base.terms["eps"]
    assert (wide.ask_price - wide.bid_price) > (base.ask_price - base.bid_price)


# ---------------------------------------------------------------------------
# Wing term: zero in belly, active outside, tier-scaled
# ---------------------------------------------------------------------------

def test_wing_zero_in_belly():
    qs = _build(consensus_p=0.5)
    assert qs.terms["wing"] == pytest.approx(0.0)


def test_wing_active_outside_belly():
    qs = _build(consensus_p=0.1)
    assert qs.terms["wing"] == pytest.approx(DEFAULT_WING_BASE_P * MMConfig().wing_widen_scale[ConfidenceTier.FULL])


def test_wing_larger_under_degraded_than_full():
    qs_full = _build(consensus_p=0.1, confidence_tier=ConfidenceTier.FULL)
    qs_degraded = _build(consensus_p=0.1, confidence_tier=ConfidenceTier.DEGRADED)
    assert qs_degraded.terms["wing"] > qs_full.terms["wing"]


# ---------------------------------------------------------------------------
# Belly term: exact complement of wing (exclusivity), free-days flat base,
# slope beyond free days, tte_days=None back-compat
# ---------------------------------------------------------------------------

def test_belly_zero_outside_band_wing_zero_inside_exclusivity():
    qs_belly = _build(consensus_p=0.5, tte_days=1.0)
    assert qs_belly.terms["belly"] > 0.0
    assert qs_belly.terms["wing"] == pytest.approx(0.0)

    qs_wing = _build(consensus_p=0.1, tte_days=1.0)
    assert qs_wing.terms["wing"] > 0.0
    assert qs_wing.terms["belly"] == pytest.approx(0.0)


def test_belly_base_only_at_or_below_free_days():
    config = MMConfig()
    qs_at_free = _build(config=config, consensus_p=0.5, tte_days=config.belly_widen_free_days)
    assert qs_at_free.terms["belly"] == pytest.approx(config.belly_widen_base_p)

    qs_below_free = _build(config=config, consensus_p=0.5, tte_days=0.5)
    assert qs_below_free.terms["belly"] == pytest.approx(config.belly_widen_base_p)


def test_belly_base_plus_slope_beyond_free_days():
    config = MMConfig()
    tte_days = 5.0
    qs = _build(config=config, consensus_p=0.5, tte_days=tte_days)
    expected = config.belly_widen_base_p + config.belly_widen_slope_p_per_day * (
        tte_days - config.belly_widen_free_days
    )
    assert qs.terms["belly"] == pytest.approx(expected)


def test_belly_tte_days_none_gives_base_only():
    config = MMConfig()
    qs = _build(config=config, consensus_p=0.5, tte_days=None)
    assert qs.terms["belly"] == pytest.approx(config.belly_widen_base_p)


# ---------------------------------------------------------------------------
# F7: shared belly-band membership predicate (config.in_belly_band) --
# boundary cases the review found missing, plus a complement-invariant sweep.
# ---------------------------------------------------------------------------

def test_belly_band_boundary_exact_lo_and_hi_fire_belly_not_wing():
    config = MMConfig()
    belly_lo, belly_hi = config.belly_band

    qs_lo = _build(config=config, consensus_p=belly_lo, tte_days=1.0)
    assert qs_lo.terms["belly"] > 0.0
    assert qs_lo.terms["wing"] == pytest.approx(0.0)

    qs_hi = _build(config=config, consensus_p=belly_hi, tte_days=1.0)
    assert qs_hi.terms["belly"] > 0.0
    assert qs_hi.terms["wing"] == pytest.approx(0.0)


def test_belly_wing_exclusivity_sweep_including_edges():
    # Sweep of consensus_p values including both belly_band edges: exactly
    # one of wing/belly fires at every point (config.in_belly_band is the
    # single source of truth both terms now share, F7).
    config = MMConfig()
    belly_lo, belly_hi = config.belly_band
    p_values = [0.01, belly_lo - 0.05, belly_lo, 0.5, belly_hi, belly_hi + 0.05, 0.99]
    for p in p_values:
        qs = _build(config=config, consensus_p=p, tte_days=1.0)
        wing_zero = qs.terms["wing"] == pytest.approx(0.0)
        belly_zero = qs.terms["belly"] == pytest.approx(0.0)
        assert wing_zero != belly_zero, f"p={p}: wing={qs.terms['wing']}, belly={qs.terms['belly']}"


# ---------------------------------------------------------------------------
# Floor: tiny proposal spread -> half-spread == 1 tick exactly
# ---------------------------------------------------------------------------

def test_floor_enforces_one_tick_when_all_terms_near_zero():
    config = MMConfig(eps_base=0.0, k_arrival=1e6)  # markup ~ 0, eps ~ 0
    proposal = _proposal(r_x=0.0, delta_x=1e-9)
    directive = make_stub_directive("m1", TS)
    sizing = make_stub_sizing("m1", TS)
    qs = build_quote_set(
        proposal, directive, sizing, VENUE, config,
        sigma2=0.0, confidence_tier=ConfidenceTier.FULL, credibility=1.0,
        consensus_p=0.5, source_seq=1, credibility_widen_scale=0.0,
    )
    half_spread = 0.5 * (qs.ask_price - qs.bid_price)
    assert half_spread == pytest.approx(VENUE.tick_size, abs=1e-9)
    assert qs.terms["floor_applied"] == 1.0


# ---------------------------------------------------------------------------
# Clamp: extreme x proposals -> prices inside band and tick-quantized
# ---------------------------------------------------------------------------

def test_clamp_and_quantize_extreme_proposal():
    proposal = _proposal(r_x=logit(0.9995), delta_x=0.05)
    qs = _build(proposal=proposal, consensus_p=0.9995)
    lo, hi = VENUE.price_band
    assert lo <= qs.bid_price <= hi
    assert lo <= qs.ask_price <= hi
    assert qs.bid_price < qs.ask_price
    # bid_price is unclamped in this scenario, so it must land exactly on a tick
    n_ticks = qs.bid_price / VENUE.tick_size
    assert abs(n_ticks - round(n_ticks)) < 1e-6


def test_quantize_lands_on_tick_when_not_band_clamped():
    bid, ask = _quantize(0.401, 0.599, tick=0.01, lo=0.001, hi=0.999)
    for price in (bid, ask):
        n_ticks = price / 0.01
        assert abs(n_ticks - round(n_ticks)) < 1e-6


# ---------------------------------------------------------------------------
# Crossing after quantization resolved
# ---------------------------------------------------------------------------

def test_quantize_resolves_crossing():
    bid, ask = _quantize(0.9989, 0.9991, tick=0.01, lo=0.001, hi=0.999)
    assert bid < ask
    assert bid <= 0.999
    assert ask <= 0.999 + 1e-12


def test_quantize_never_produces_bid_ge_ask_on_random_inputs():
    import random
    rng = random.Random(0)
    for _ in range(200):
        c = rng.uniform(0.001, 0.999)
        hs = rng.uniform(0.0, 0.02)
        bid, ask = _quantize(c - hs, c + hs, tick=0.01, lo=0.001, hi=0.999)
        assert bid < ask


# ---------------------------------------------------------------------------
# Modes: BID_ONLY / ASK_ONLY / PULLED zero the right sizes
# ---------------------------------------------------------------------------

def test_mode_bid_only_zeros_ask_size():
    import dataclasses
    d = dataclasses.replace(make_stub_directive("m1", TS), mode=QuoteMode.BID_ONLY)
    qs = _build(directive=d, sizing=make_stub_sizing("m1", TS, bid_size=5.0, ask_size=7.0))
    assert qs.bid_size == pytest.approx(5.0)
    assert qs.ask_size == pytest.approx(0.0)


def test_mode_ask_only_zeros_bid_size():
    import dataclasses
    d = dataclasses.replace(make_stub_directive("m1", TS), mode=QuoteMode.ASK_ONLY)
    qs = _build(directive=d, sizing=make_stub_sizing("m1", TS, bid_size=5.0, ask_size=7.0))
    assert qs.bid_size == pytest.approx(0.0)
    assert qs.ask_size == pytest.approx(7.0)


def test_mode_pulled_zeros_both_sizes():
    import dataclasses
    d = dataclasses.replace(make_stub_directive("m1", TS), mode=QuoteMode.PULLED)
    qs = _build(directive=d, sizing=make_stub_sizing("m1", TS, bid_size=5.0, ask_size=7.0))
    assert qs.bid_size == pytest.approx(0.0)
    assert qs.ask_size == pytest.approx(0.0)
    assert qs.risk_mode == QuoteMode.PULLED


def test_mode_two_sided_keeps_both_sizes():
    qs = _build(sizing=make_stub_sizing("m1", TS, bid_size=5.0, ask_size=7.0))
    assert qs.bid_size == pytest.approx(5.0)
    assert qs.ask_size == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# noarb_checked False, and stub helpers construct valid objects
# ---------------------------------------------------------------------------

def test_noarb_checked_false_by_default():
    qs = _build()
    assert qs.noarb_checked is False


def test_stub_directive_and_sizing_shape():
    d = make_stub_directive("m1", TS)
    assert d.mode == QuoteMode.TWO_SIDED
    assert d.eps_add == pytest.approx(0.0)
    s = make_stub_sizing("m1", TS)
    assert s.bid_size > 0.0 and s.ask_size > 0.0


# ---------------------------------------------------------------------------
# wave 2 W1: compute_posted_prices split -- bit-identity with build_quote_set
# (posted=None path), and posted= short-circuit (verbatim tuple respected).
# ---------------------------------------------------------------------------


def _directive_variants():
    base = make_stub_directive("m1", TS)
    import dataclasses
    return [
        base,
        dataclasses.replace(base, eps_add=0.05),
        dataclasses.replace(base, mode=QuoteMode.BID_ONLY),
    ]


def _proposal_variants():
    return [
        _proposal(r_x=0.0, delta_x=0.3),
        _proposal(r_x=logit(0.7), delta_x=0.1, skew_x=0.02),
        _proposal(r_x=logit(0.9995), delta_x=0.05),
    ]


def test_compute_posted_prices_matches_build_quote_set_over_grid():
    config = MMConfig()
    sizing = make_stub_sizing("m1", TS)
    for proposal in _proposal_variants():
        for directive in _directive_variants():
            for consensus_p in (0.1, 0.5, 0.9):
                for tte_days in (0.5, 5.0, None):
                    bid_p, ask_p, terms_p = compute_posted_prices(
                        proposal, directive, VENUE, config,
                        sigma2=0.0002, confidence_tier=ConfidenceTier.FULL,
                        credibility=0.7, consensus_p=consensus_p, tte_days=tte_days,
                    )
                    qs = build_quote_set(
                        proposal, directive, sizing, VENUE, config,
                        sigma2=0.0002, confidence_tier=ConfidenceTier.FULL,
                        credibility=0.7, consensus_p=consensus_p, source_seq=1,
                        tte_days=tte_days,
                    )
                    assert qs.bid_price == pytest.approx(bid_p, abs=1e-12)
                    assert qs.ask_price == pytest.approx(ask_p, abs=1e-12)
                    assert qs.terms == terms_p


def test_build_quote_set_posted_short_circuit_respected_verbatim():
    # A posted= tuple with prices that DISAGREE with what compute_posted_prices
    # would have computed must be respected exactly (no recomputation) --
    # proves the harness's "compute once, pass in" ordering (wave 2 W1/W7)
    # actually short-circuits rather than silently recomputing.
    config = MMConfig()
    proposal = _proposal(r_x=0.0, delta_x=0.3)
    directive = make_stub_directive("m1", TS)
    sizing = make_stub_sizing("m1", TS)
    fake_posted = (0.111, 0.222, {"markup": 0.0, "eps": 0.0, "skew": 0.0,
                                   "robust": 0.0, "wing": 0.0, "belly": 0.0,
                                   "floor_applied": 0.0})
    qs = build_quote_set(
        proposal, directive, sizing, VENUE, config,
        sigma2=0.0002, confidence_tier=ConfidenceTier.FULL, credibility=0.7,
        consensus_p=0.5, source_seq=1, posted=fake_posted,
    )
    assert qs.bid_price == pytest.approx(0.111)
    assert qs.ask_price == pytest.approx(0.222)
    assert qs.terms == fake_posted[2]
