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
    expected_half_spread = proposal_half_spread + qs.terms["eps"] + qs.terms["robust"] + qs.terms["wing"]
    actual_half_spread = 0.5 * (qs.ask_price - qs.bid_price)
    assert actual_half_spread == pytest.approx(expected_half_spread, abs=1e-4)
    assert qs.terms["markup"] > 0.0  # still reported for decomposition audit


def test_terms_dict_has_all_keys():
    qs = _build()
    for key in ["markup", "eps", "skew", "robust", "wing", "floor_applied"]:
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
