"""Tests for market_maker.fair_value_anchor (plan task F1, Section 2.3;
per-region bankrolls package B2, 2026-07-15).

Covers the five NORMATIVE invariants (each its own test, generalized to the
per-region dict signature), a hand-computed 3-strike consensus, monotone
re-integration, the degeneracy fallback, the bankroll floor, and B2's
per-region mechanics (region assignment, tail-bucket pinning, the two-phase
update's empty/degenerate-region skip rule, boundary monotonicity, and the
ladder-space repair-then-clamp ordering).
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import (
    AnchorMethod,
    BankrollState,
    ConfidenceTier,
    PricerSnapshot,
    Sigma2Source,
)
from market_maker.fair_value_anchor import (
    BELLY_REGION,
    DEFAULT_BANKROLL_FLOOR,
    WING_REGION,
    ladder_to_buckets,
    buckets_to_ladder,
    compute_fair_value,
)
import market_maker.fair_value_anchor as fva

TS = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def _snapshot(strikes, p_list, expiry_key="2026-07-20"):
    p_hat = {float(k): float(p) for k, p in zip(strikes, p_list)}
    sigma2 = {float(k): 1e-6 for k in strikes}
    return PricerSnapshot(
        ts=TS,
        expiry_key=expiry_key,
        tte_days=14.0,
        s0=100000.0,
        n_sims=15000,
        strikes=[float(k) for k in strikes],
        grid_strikes=[float(k) for k in strikes],
        p_hat=p_hat,
        p_grid=dict(p_hat),
        sigma2=sigma2,
        sigma2_ladder=1e-6,
        sigma2_source=Sigma2Source.MC,
        confidence_tier=ConfidenceTier.FULL,
        horizon_gate_active=False,
        stale=False,
    )


def _bankrolls(pricer=0.5, market=0.5, frozen=False, update_count=0):
    return BankrollState(
        model_ids=["pricer", "market"],
        bankrolls={"pricer": pricer, "market": market},
        last_update=TS,
        update_count=update_count,
        frozen=frozen,
    )


def _states(pricer=0.5, market=0.5, frozen=False):
    """Both regions seeded with the SAME weights. A single `compute_fair_
    value` call (no threaded history) with region-uniform weights is
    provably identical to the old single-region algorithm: the ladder-space
    blend at every strike uses the same (pricer, market) weight pair
    regardless of that strike's region, so `_ladder_space_consensus`
    reduces exactly to the old bucket-space constant-weight blend (both are
    linear in the per-model sanitized ladders, and neither the cummin-repair
    nor the clamp step engages for a well-formed convex combination of two
    monotone ladders)."""
    return {
        BELLY_REGION: _bankrolls(pricer, market, frozen),
        WING_REGION: _bankrolls(pricer, market, frozen),
    }


def _mids(strikes, p_list):
    return {float(k): float(p) for k, p in zip(strikes, p_list)}


CFG = MMConfig()
# Legacy (pre-Fix-1, pre-item-3) behavior: negative pin disables the wing
# pricer weight pin, restoring the wing region's own Bayes updates; temper=1.0
# disables Bayes-factor tempering (2026-08-10 skew-fix wave item 3), restoring
# the untempered per-tick step size. Tests whose PURPOSE is the legacy Bayes
# dynamics (region attribution, floor, skip rules, ...) run with this config
# -- MMConfig's own default (0.1) would otherwise make every such test move
# ~10x slower per tick and silently invalidate any magnitude assertion
# calibrated to the untempered math; tests of the production path use CFG
# (pin = 0.5, temper = 0.1). See the "Bankroll update tempering" section below
# for tests whose PURPOSE is the temper mechanism itself.
LEGACY_CFG = MMConfig(wing_pricer_weight_pin=-1.0, bankroll_update_temper=1.0)


# ---------------------------------------------------------------------------
# Hand-computed 3-strike consensus (unequal bankrolls, region-uniform)
# ---------------------------------------------------------------------------

def test_hand_computed_consensus_unequal_bankrolls():
    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        _mids(strikes, [0.8, 0.5, 0.2]),
        _states(0.7, 0.3),
        CFG,
    )
    cp = res.fair_value.consensus_p
    # 0.7*pricer + 0.3*market per strike
    assert cp[1.0] == pytest.approx(0.87, abs=1e-9)
    assert cp[2.0] == pytest.approx(0.57, abs=1e-9)
    assert cp[3.0] == pytest.approx(0.27, abs=1e-9)
    # All three market mids (0.8, 0.5, 0.2) classify belly (in_belly_band is
    # inclusive), so the scalar credibility is the belly weight and the
    # per-strike consensus above is untouched by the wing pin. The wing
    # region's reported credibility IS the pin (Fix 1, default 0.5), not the
    # stored 0.7 -- wing owns no strikes here, only the two tail buckets.
    assert res.fair_value.credibility == pytest.approx(0.7, abs=1e-9)
    assert res.fair_value.credibility_by_region[BELLY_REGION] == pytest.approx(0.7, abs=1e-9)
    assert res.fair_value.credibility_by_region[WING_REGION] == pytest.approx(0.5, abs=1e-9)
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY


# ---------------------------------------------------------------------------
# NORMATIVE invariant 1: unanimity fixed point
# ---------------------------------------------------------------------------

def test_invariant1_unanimity_fixed_point():
    strikes = [1.0, 2.0, 3.0]
    p = [0.8, 0.5, 0.2]
    res = compute_fair_value(
        _snapshot(strikes, p), _mids(strikes, p), _states(0.3, 0.7), CFG,
    )
    cp = res.fair_value.consensus_p
    for k, pv in zip(strikes, p):
        assert cp[k] == pytest.approx(pv, abs=1e-9)


# ---------------------------------------------------------------------------
# NORMATIVE invariant 2: per-bucket boundedness within [min_i, max_i]
# ---------------------------------------------------------------------------

def test_invariant2_per_bucket_boundedness():
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.9, 0.6, 0.3]
    market = [0.8, 0.5, 0.2]
    res = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _states(0.6, 0.4), CFG,
    )
    bp = ladder_to_buckets(np.array(pricer))
    bm = ladder_to_buckets(np.array(market))
    lo = np.minimum(bp, bm)
    hi = np.maximum(bp, bm)
    cons = res.consensus_bucket
    assert np.all(cons >= lo - 1e-12)
    assert np.all(cons <= hi + 1e-12)


# ---------------------------------------------------------------------------
# NORMATIVE invariant 3: buckets nonnegative and sum to 1
# ---------------------------------------------------------------------------

def test_invariant3_buckets_nonneg_sum_to_one():
    strikes = [1.0, 2.0, 3.0, 4.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.85, 0.6, 0.35, 0.1]),
        _mids(strikes, [0.8, 0.55, 0.3, 0.15]),
        _states(0.5, 0.5),
        CFG,
    )
    cons = res.consensus_bucket
    assert np.all(cons >= 0.0)
    assert cons.sum() == pytest.approx(1.0, abs=1e-9)
    assert cons.size == len(strikes) + 1


# ---------------------------------------------------------------------------
# NORMATIVE invariant 4: credibility weights nonnegative and sum to 1, PER
# REGION (plan: "every invariant test ... must pass per region").
# ---------------------------------------------------------------------------

def test_invariant4_credibility_weights_nonneg_sum_to_one_per_region():
    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.8, 0.5, 0.2]),
        _mids(strikes, [0.7, 0.5, 0.3]),
        _states(0.4, 0.6),
        CFG,
    )
    for region in (BELLY_REGION, WING_REGION):
        vals = list(res.bankroll_states[region].bankrolls.values())
        assert all(v >= 0.0 for v in vals)
        assert sum(vals) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# NORMATIVE invariant 5: monotone credibility gain for the better model
# (region-uniform threading; qualitative property, both the weighted-average
# `credibility` and each region's own `credibility_by_region` are checked).
# ---------------------------------------------------------------------------

def test_invariant5_monotone_credibility_gain():
    # LEGACY_CFG: this invariant is about the Bayes credibility dynamics in
    # BOTH regions; the default-on wing pin (Fix 1) freezes wing at 0.5 by
    # design, so the legacy config keeps the test exercising what it names.
    strikes = [1.0, 2.0, 3.0]
    pricer = np.array([0.8, 0.5, 0.2])
    market0 = np.array([0.55, 0.5, 0.45])
    T = 20
    states = _states(0.5, 0.5)
    prev_fc = None
    prev_cons = None
    creds = []
    for t in range(T):
        frac = t / (T - 1)
        market_t = market0 + frac * (pricer - market0)
        res = compute_fair_value(
            _snapshot(strikes, pricer),
            _mids(strikes, market_t),
            states,
            LEGACY_CFG,
            prev_forecasts=prev_fc,
            prev_consensus=prev_cons,
        )
        creds.append(res.fair_value.credibility)
        states = res.bankroll_states
        prev_fc = res.forecasts
        prev_cons = res.consensus_bucket
    # pricer credibility should be monotonically non-decreasing (weighted
    # average across regions); 1e-6 slack for accumulated float summation.
    for a, b in zip(creds, creds[1:]):
        assert b >= a - 1e-6
    assert creds[-1] > creds[0]


# ---------------------------------------------------------------------------
# Monotone re-integration: consensus_p non-increasing in strike
# ---------------------------------------------------------------------------

def test_consensus_p_monotone_non_increasing():
    strikes = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.7, 0.5, 0.3, 0.1]),
        _mids(strikes, [0.85, 0.68, 0.48, 0.28, 0.12]),
        _states(0.5, 0.5),
        CFG,
    )
    vals = [res.fair_value.consensus_p[k] for k in strikes]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12


# ---------------------------------------------------------------------------
# Degeneracy fallback (risk 8.8) -- freezes BOTH regions (plan step 8)
# ---------------------------------------------------------------------------

def test_fallback_on_nan_mid():
    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        {1.0: 0.8, 2.0: float("nan"), 3.0: 0.2},
        _states(0.5, 0.5),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    assert res.bankroll_states[BELLY_REGION].frozen is True
    assert res.bankroll_states[WING_REGION].frozen is True


def test_fallback_on_all_zero_bankrolls():
    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        _mids(strikes, [0.8, 0.5, 0.2]),
        _states(0.0, 0.0),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    assert res.bankroll_states[BELLY_REGION].frozen is True
    assert res.bankroll_states[WING_REGION].frozen is True
    # w=0.5 blend of pricer and mid per strike
    cp = res.fair_value.consensus_p
    assert cp[1.0] == pytest.approx(0.85, abs=1e-9)
    assert cp[3.0] == pytest.approx(0.25, abs=1e-9)


def test_non_monotone_pricer_sanitized_not_fallback():
    # Was test_fallback_on_sanity_bound_violation. Under the OLD raw-ladder
    # sanity band this non-monotone pricer ([0.3, 0.9, 0.2]) tripped the
    # fallback. Under the fix (Change 1), the band is built from the
    # SANITIZED (bucket-round-tripped) ladders, and the consensus is
    # provably a convex combination of those same sanitized ladders, so this
    # is now the normal BEUOY path -- non-monotone venue mids get repaired
    # by the bucket transform instead of freezing the bankroll.
    strikes = [1.0, 2.0, 3.0]
    pricer = np.array([0.3, 0.9, 0.2])
    market = np.array([0.8, 0.5, 0.2])
    res = compute_fair_value(
        _snapshot(strikes, pricer),
        _mids(strikes, market),
        _states(0.5, 0.5),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res.bankroll_states[BELLY_REGION].frozen is False
    assert res.bankroll_states[WING_REGION].frozen is False

    recon_pricer = buckets_to_ladder(ladder_to_buckets(pricer))
    recon_market = buckets_to_ladder(ladder_to_buckets(market))
    lo = np.minimum(recon_pricer, recon_market) - 1e-6
    hi = np.maximum(recon_pricer, recon_market) + 1e-6

    cp = res.fair_value.consensus_p
    cons = np.array([cp[k] for k in strikes])
    assert np.all(cons >= lo)
    assert np.all(cons <= hi)
    for a, b in zip(cons, cons[1:]):
        assert b <= a + 1e-12

    # Hand-checked (w=0.5/0.5, region-uniform): recon_pricer=[0.5625,0.5625,
    # 0.125], recon_market=[0.8,0.5,0.2], consensus=[0.68125,0.53125,0.1625].
    assert recon_pricer == pytest.approx([0.5625, 0.5625, 0.125], abs=1e-9)
    assert recon_market == pytest.approx([0.8, 0.5, 0.2], abs=1e-9)
    assert cons == pytest.approx([0.68125, 0.53125, 0.1625], abs=1e-9)


def test_crossed_wing_mids_production_regression():
    # Actual VPS 2026-07-17 production ladder (crossed wing mids: K=70000
    # mid 0.0055 < K=72000 mid 0.0140). Under the OLD raw-ladder sanity band
    # this exact ladder fell back on 1949/1949 recomputes for the expiry
    # (FIXED_BLEND_FALLBACK) and froze the bankroll for 7h+ -- auto-unfreeze
    # (20 consecutive clean BEUOY ticks) could never even start. Under the
    # sanitized-band fix (Change 1) it must price BEUOY, not frozen.
    strikes = [
        54000, 56000, 58000, 60000, 62000, 64000, 66000, 68000,
        70000, 72000, 74000,
    ]
    mids = [
        0.9980, 0.9935, 0.9745, 0.9045, 0.6550, 0.2850, 0.0745,
        0.0115, 0.0055, 0.0140, 0.0045,
    ]
    pricer = [
        0.9985, 0.9930, 0.9700, 0.9000, 0.6600, 0.2900, 0.0800,
        0.0150, 0.0080, 0.0060, 0.0040,
    ]
    res = compute_fair_value(
        _snapshot(strikes, pricer),
        _mids(strikes, mids),
        _states(0.5, 0.5),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res.bankroll_states[BELLY_REGION].frozen is False
    assert res.bankroll_states[WING_REGION].frozen is False


def test_sanity_fallback_still_fires_on_band_violation(monkeypatch):
    # Safety-net test for the band check itself, re-targeted for the B2
    # ladder-space architecture (plan step 4): the consensus is now clamped
    # into band INSIDE `_ladder_space_consensus` (repair-then-clamp), so the
    # band check downstream passes "by construction" in normal operation --
    # the OLD injection point (corrupting `buckets_to_ladder`'s
    # reconstruction of the blended consensus) no longer exists, since the
    # final consensus is never round-tripped through buckets_to_ladder.
    # Instead, monkeypatch `_ladder_space_consensus` itself to skip its own
    # clamp (still returning a value derived from real inputs, just
    # deliberately out of band) so the downstream safety-net check has
    # something to catch.
    _real = fva._ladder_space_consensus

    def _unclamped(weight_dicts_by_region, region_of_strike, sanitized, model_ids):
        n = len(region_of_strike)
        raw = np.empty(n, dtype=float)
        for k in range(n):
            w = weight_dicts_by_region[region_of_strike[k]]
            raw[k] = sum(w[mid] * sanitized[mid][k] for mid in model_ids)
        for j in range(1, n):
            if raw[j] > raw[j - 1]:
                raw[j] = raw[j - 1]
        # Deliberately SKIP the clamp step -- push the last value far out of
        # band so the safety net must catch it.
        raw[-1] = raw[-1] + 0.5
        return raw

    monkeypatch.setattr(fva, "_ladder_space_consensus", _unclamped)

    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        _mids(strikes, [0.8, 0.5, 0.2]),
        _states(0.7, 0.3),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    assert res.bankroll_states[BELLY_REGION].frozen is True
    assert res.bankroll_states[WING_REGION].frozen is True


def test_frozen_state_recovers_beuoy_on_crossed_mids():
    # Contract: the anchor's top-level path prices BEUOY regardless of the
    # incoming frozen flag -- frozen only skips the Bayes mark-to-market
    # bankroll update (per region), not the consensus computation. The
    # anchor itself NEVER self-clears frozen; auto-unfreeze is owned
    # entirely by the harness, keyed on 20 consecutive clean BEUOY
    # recomputes. This test documents that contract with a crossed-mid
    # (non-monotone) ladder on top of an already-frozen bankroll state
    # (both regions frozen, matching the lockstep freeze semantics).
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.9, 0.6, 0.3]
    mids = [0.8, 0.5, 0.52]  # non-monotone: K3 mid (0.52) > K2 mid (0.5)
    res = compute_fair_value(
        _snapshot(strikes, pricer),
        _mids(strikes, mids),
        _states(0.5, 0.5, frozen=True),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res.bankroll_states[BELLY_REGION].frozen is True
    assert res.bankroll_states[WING_REGION].frozen is True


# ---------------------------------------------------------------------------
# Bankroll floor: always-wrong model does not go below floor (either region)
# ---------------------------------------------------------------------------

def test_bankroll_floor_holds():
    # LEGACY_CFG: the floor property is only exercised when wing actually
    # Bayes-updates; the pin (Fix 1) would hold wing at 0.5/0.5 vacuously.
    strikes = [1.0, 2.0, 3.0]
    pricer = np.array([0.8, 0.5, 0.2])
    market0 = np.array([0.55, 0.5, 0.45])
    T = 60
    states = _states(0.5, 0.5)
    prev_fc = None
    prev_cons = None
    for t in range(T):
        frac = min(t / 10.0, 1.0)
        market_t = market0 + frac * (pricer - market0)
        res = compute_fair_value(
            _snapshot(strikes, pricer),
            _mids(strikes, market_t),
            states,
            LEGACY_CFG,
            prev_forecasts=prev_fc,
            prev_consensus=prev_cons,
        )
        states = res.bankroll_states
        prev_fc = res.forecasts
        prev_cons = res.consensus_bucket
    for region in (BELLY_REGION, WING_REGION):
        assert states[region].bankrolls["market"] >= DEFAULT_BANKROLL_FLOOR - 1e-12
        assert states[region].bankrolls["pricer"] <= 1.0 - DEFAULT_BANKROLL_FLOOR + 1e-12


# ===========================================================================
# Package B2: per-region mechanics
# ===========================================================================


# ---------------------------------------------------------------------------
# Two-phase unanimity: both models' forecasts always equal to prev-consensus
# -> weights unchanged in BOTH regions (round-2 review item 1's reworded
# property -- NOT "a single model keeps its bankroll", which is false).
# ---------------------------------------------------------------------------

def test_two_phase_unanimity_weights_unchanged_both_regions():
    # LEGACY_CFG: the property "unanimity -> Bayes leaves weights unchanged"
    # requires the wing Bayes update to actually run; the pin (Fix 1) would
    # overwrite the seeded 0.6/0.4 wing weights with 0.5/0.5 on tick one.
    strikes = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = [0.9, 0.6, 0.5, 0.4, 0.1]  # spans both belly (0.6/0.5/0.4) and wing (0.9/0.1)
    states = {BELLY_REGION: _bankrolls(0.3, 0.7), WING_REGION: _bankrolls(0.6, 0.4)}
    prev_fc = None
    prev_cons = None
    for _ in range(4):
        before = {r: dict(states[r].bankrolls) for r in (BELLY_REGION, WING_REGION)}
        res = compute_fair_value(
            _snapshot(strikes, p), _mids(strikes, p), states, LEGACY_CFG,
            prev_forecasts=prev_fc, prev_consensus=prev_cons,
        )
        for region in (BELLY_REGION, WING_REGION):
            for mid_id, w in res.bankroll_states[region].bankrolls.items():
                assert w == pytest.approx(before[region][mid_id], abs=1e-9)
        assert res.fair_value.anchor_method == AnchorMethod.BEUOY
        states = res.bankroll_states
        prev_fc = res.forecasts
        prev_cons = res.consensus_bucket


# ---------------------------------------------------------------------------
# Region attribution: pricer wrong ONLY in the wings -> wing pricer bankroll
# falls; belly pricer bankroll stays close to its start (within floor/
# normalization tolerance, plan step 12).
# ---------------------------------------------------------------------------

def test_region_attribution_pricer_wrong_only_in_wings():
    strikes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # Fixed, CONFIDENT pricer: correct/agreeing in the belly (indices 2,3,4)
    # and at the near-wing BUFFER indices (1,5, which never move); wrong
    # only at the two EXTREME strikes (0,6).
    pricer = np.array([0.97, 0.85, 0.6, 0.5, 0.4, 0.15, 0.03])
    # Market tracks pricer exactly everywhere except the two extreme
    # strikes, which drift toward their FIXED neighbor (staying wing-
    # classified and ladder-monotone throughout) -- "the market decides
    # pricer's extreme-tail confidence was overstated". Because indices
    # 1..5 never move, ladder_to_buckets' bucket-boundary coupling confines
    # this disagreement structurally to WING's own buckets (0, 1, 6, 7):
    # belly's buckets (3, 4, 5) reference only indices 2, 3, 4, 5 -- wait,
    # bucket 5 = p[4]-p[5] DOES reference index 5, but index 5 is the FIXED
    # buffer (never drifts), so bucket 5 sees zero disagreement too. Belly
    # sees perfect, unchanging unanimity throughout -> its bankroll should
    # be exactly (not just approximately) unchanged.
    market0 = pricer.copy()
    states = _states(0.5, 0.5)
    prev_fc = None
    prev_cons = None
    T = 20
    for t in range(1, T):
        frac = t / (T - 1)
        market_t = market0.copy()
        market_t[0] = pricer[0] + frac * (0.86 - pricer[0])   # 0.97 -> 0.86 (stays > pricer[1]=0.85)
        market_t[6] = pricer[6] + frac * (0.14 - pricer[6])   # 0.03 -> 0.14 (stays < pricer[5]=0.15)
        # LEGACY_CFG: region attribution IS the wing Bayes update the pin
        # (Fix 1) deliberately disables.
        res = compute_fair_value(
            _snapshot(strikes, pricer), _mids(strikes, market_t), states, LEGACY_CFG,
            prev_forecasts=prev_fc, prev_consensus=prev_cons,
        )
        assert res.fair_value.anchor_method == AnchorMethod.BEUOY
        states = res.bankroll_states
        prev_fc = res.forecasts
        prev_cons = res.consensus_bucket

    # Wing pricer credibility clearly fell from the 0.5 start (pricer is
    # measurably wrong there); belly is untouched (perfect, unmoving
    # agreement throughout -> zero evidence either way).
    assert states[WING_REGION].bankrolls["pricer"] < 0.45
    assert states[BELLY_REGION].bankrolls["pricer"] == pytest.approx(0.5, abs=1e-6)


def test_tail_bucket_error_lands_on_wing_even_when_extreme_strikes_are_belly():
    # Construct a ladder where EVERY strike classifies belly (market values
    # all inside belly_band), yet the FIRST strike's pricer/market values
    # disagree substantially. If the tail-bucket-always-wing rule (plan step
    # 2) were broken (e.g. bucket 0 mis-assigned to the extreme strike's OWN
    # region instead of being pinned wing), this disagreement would show up
    # nowhere in the wing bankroll (since no strike is nominally "wing") and
    # wing's weights would sit frozen at their start. Asserting wing's
    # credibility actually moves falsifies that regression.
    strikes = [1.0, 2.0, 3.0, 4.0]
    market = [0.75, 0.55, 0.4, 0.25]  # all in [0.2, 0.8] -> ALL belly
    pricer = [0.95, 0.75, 0.4, 0.25]  # interior (idx 1,2,3) agree; idx0 disagrees
    states = _states(0.5, 0.5)
    prev_fc = None
    prev_cons = None
    wing_creds = []
    T = 6
    for _ in range(T):
        # LEGACY_CFG: the falsifiable signal here is wing's bankroll MOVING
        # off 0.5 -- impossible under the pin (Fix 1), which is exactly the
        # legacy Bayes evidence-routing this test exists to check.
        res = compute_fair_value(
            _snapshot(strikes, pricer), _mids(strikes, market), states, LEGACY_CFG,
            prev_forecasts=prev_fc, prev_consensus=prev_cons,
        )
        assert res.fair_value.anchor_method == AnchorMethod.BEUOY
        wing_creds.append(res.bankroll_states[WING_REGION].bankrolls["pricer"])
        states = res.bankroll_states
        prev_fc = res.forecasts
        prev_cons = res.consensus_bucket
    # Wing's bankroll must have MOVED from the 0.5 start (the tail-bucket
    # disagreement is real evidence), even though every strike nominally
    # classifies belly.
    assert wing_creds[-1] != pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Empty-region skip (both directions) + degenerate-factor skip.
# ---------------------------------------------------------------------------

def test_empty_region_skip_all_wing_ladder_belly_unaffected():
    # All strikes classify wing (market deep ITM-like, all > belly_band hi)
    # -> belly owns ZERO buckets (the tail-pinning rule only ever guarantees
    # wing is non-empty; belly can be empty on an all-wing ladder). Belly
    # must skip cleanly: no fallback, no freeze, weights/update_count
    # unchanged; wing (non-empty) updates normally.
    # LEGACY_CFG: asserts wing's update_count ADVANCES (normal Bayes update)
    # while empty belly skips -- the pin (Fix 1) would skip wing too.
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.97, 0.92, 0.83]
    market0 = [0.95, 0.90, 0.85]
    states = _states(0.5, 0.5)
    snap = _snapshot(strikes, pricer)

    res1 = compute_fair_value(snap, _mids(strikes, market0), states, LEGACY_CFG)
    assert res1.fair_value.anchor_method == AnchorMethod.BEUOY
    belly_before = dict(res1.bankroll_states[BELLY_REGION].bankrolls)
    belly_uc_before = res1.bankroll_states[BELLY_REGION].update_count
    wing_uc_before = res1.bankroll_states[WING_REGION].update_count

    market1 = [0.96, 0.94, 0.90]
    res2 = compute_fair_value(
        snap, _mids(strikes, market1), res1.bankroll_states, LEGACY_CFG,
        prev_forecasts=res1.forecasts, prev_consensus=res1.consensus_bucket,
    )
    assert res2.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res2.bankroll_states[BELLY_REGION].bankrolls == pytest.approx(belly_before)
    assert res2.bankroll_states[BELLY_REGION].update_count == belly_uc_before
    assert res2.bankroll_states[BELLY_REGION].frozen is False
    assert res2.bankroll_states[WING_REGION].frozen is False
    assert res2.bankroll_states[WING_REGION].update_count == wing_uc_before + 1


def test_empty_region_skip_all_belly_ladder_wing_evidence_from_tails_only():
    # All strikes classify belly -> wing owns ONLY the two tail buckets
    # (never empty, by the tail-pinning rule); no fallback, no freeze.
    # LEGACY_CFG: wing's "evidence from the tail buckets only" update is the
    # legacy Bayes path the pin (Fix 1) deliberately disables.
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.7, 0.5, 0.3]
    market0 = [0.65, 0.5, 0.35]
    states = _states(0.5, 0.5)
    snap = _snapshot(strikes, pricer)

    res1 = compute_fair_value(snap, _mids(strikes, market0), states, LEGACY_CFG)
    assert res1.fair_value.anchor_method == AnchorMethod.BEUOY

    market1 = [0.6, 0.5, 0.4]
    res2 = compute_fair_value(
        snap, _mids(strikes, market1), res1.bankroll_states, LEGACY_CFG,
        prev_forecasts=res1.forecasts, prev_consensus=res1.consensus_bucket,
    )
    assert res2.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res2.bankroll_states[BELLY_REGION].frozen is False
    assert res2.bankroll_states[WING_REGION].frozen is False
    # No fallback occurred and wing was never bucket-empty -- it either
    # updated (evidence from its two tail buckets) or cleanly skipped via
    # the s_R<=0 branch; either way update_count is a valid non-decreasing
    # int and the run did not fall back/freeze.
    assert res2.bankroll_states[WING_REGION].update_count >= res1.bankroll_states[WING_REGION].update_count


def test_degenerate_factor_skip_wing_belly_still_updates():
    # Engineered so wing's two tail buckets carry EXACTLY zero mass in the
    # pre-update consensus (s_R == 0 exactly), forcing wing to skip, while
    # belly (interior buckets, nonzero mass) updates normally. Belly's
    # PRE-update weight is set to 100% pricer so the belly-weighted blend at
    # the boundary strikes equals pricer's own (extreme) value exactly,
    # while the market's OWN value there (0.75 / 0.25) independently stays
    # in-band for region classification -- see the module test docstring in
    # the implementation notes for the derivation.
    strikes = [1.0, 2.0, 3.0]
    # pricer[1]=0.6 (strictly between pricer[0]=1.0 and pricer[2]=0.0) keeps
    # belly's OWN buckets (1,2) non-degenerate; a degenerate pricer like
    # [1.0, 1.0, 0.0] would ALSO zero belly's bucket 1 (since belly is
    # 100%-weighted to pricer below), spuriously skipping belly too.
    pricer = [1.0, 0.6, 0.0]
    market0 = [0.75, 0.6, 0.25]  # all in [0.2, 0.8] -> ALL belly-classified
    states = {
        BELLY_REGION: _bankrolls(1.0, 0.0),
        WING_REGION: _bankrolls(0.5, 0.5),
    }
    snap = _snapshot(strikes, pricer)

    # LEGACY_CFG: the s_R == 0 degenerate-factor skip branch for wing is
    # unreachable under the pin (Fix 1) -- wing never evaluates factors.
    res1 = compute_fair_value(snap, _mids(strikes, market0), states, LEGACY_CFG)
    assert res1.fair_value.anchor_method == AnchorMethod.BEUOY
    wing_before = dict(res1.bankroll_states[WING_REGION].bankrolls)
    wing_uc_before = res1.bankroll_states[WING_REGION].update_count
    belly_uc_before = res1.bankroll_states[BELLY_REGION].update_count

    market1 = [0.7, 0.55, 0.3]
    res2 = compute_fair_value(
        snap, _mids(strikes, market1), res1.bankroll_states, LEGACY_CFG,
        prev_forecasts=res1.forecasts, prev_consensus=res1.consensus_bucket,
    )
    assert res2.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res2.bankroll_states[WING_REGION].bankrolls == pytest.approx(wing_before)
    assert res2.bankroll_states[WING_REGION].update_count == wing_uc_before  # skipped
    assert res2.bankroll_states[WING_REGION].frozen is False  # no freeze
    assert res2.bankroll_states[BELLY_REGION].update_count == belly_uc_before + 1  # still updates
    assert res2.bankroll_states[BELLY_REGION].frozen is False


# ---------------------------------------------------------------------------
# Boundary monotonicity: sharp cross-boundary weight difference -> monotone,
# in-band, method BEUOY (no spurious FIXED_BLEND fallback).
# ---------------------------------------------------------------------------

def test_boundary_monotonicity_sharp_region_weight_difference():
    strikes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    pricer = [0.95, 0.85, 0.6, 0.5, 0.4, 0.15, 0.05]
    market = [0.9, 0.8, 0.55, 0.5, 0.45, 0.2, 0.1]
    # Sharp weight difference across the belly/wing boundary. LEGACY_CFG so
    # the designed 0.95-vs-0.05 contrast actually prices (the pin (Fix 1)
    # would soften the wing side to 0.5/0.5); the pinned-path monotonicity is
    # covered by the Fix 1 tests below.
    states = {
        BELLY_REGION: _bankrolls(0.95, 0.05),
        WING_REGION: _bankrolls(0.05, 0.95),
    }
    res = compute_fair_value(_snapshot(strikes, pricer), _mids(strikes, market), states, LEGACY_CFG)
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY
    for region in (BELLY_REGION, WING_REGION):
        assert res.bankroll_states[region].frozen is False

    cp = res.fair_value.consensus_p
    vals = [cp[k] for k in strikes]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-9  # monotone non-increasing

    recon_pricer = buckets_to_ladder(ladder_to_buckets(np.array(pricer)))
    recon_market = buckets_to_ladder(ladder_to_buckets(np.array(market)))
    lo = np.minimum(recon_pricer, recon_market) - 1e-6
    hi = np.maximum(recon_pricer, recon_market) + 1e-6
    cons = np.array(vals)
    assert np.all(cons >= lo)
    assert np.all(cons <= hi)


# ===========================================================================
# Fix 1 (2026-08-08 wing-bleed fix): wing pricer weight PIN
# ===========================================================================
#
# Default-on (MMConfig.wing_pricer_weight_pin = 0.5): the wing region's Bayes
# update is a self-confirmation loop (factors score against a consensus built
# from the pre-update weights), so wing weights are pinned and the wing update
# is skipped entirely. LEGACY_CFG (pin = -1.0) disables.


def _mixed_region_inputs():
    """5-strike ladder spanning both regions. Market values classify (via the
    sanitized-market ladder + inclusive in_belly_band) wing/belly/belly/belly/
    wing. Both ladders monotone, so sanitized == raw and the per-strike blend
    is hand-computable (neither cummin repair nor band clamp engages)."""
    strikes = [1.0, 2.0, 3.0, 4.0, 5.0]
    pricer = [0.95, 0.75, 0.55, 0.35, 0.15]
    market = [0.9, 0.7, 0.5, 0.3, 0.1]
    return strikes, pricer, market


def _pin_states(update_count=0):
    """Stored bankrolls mirroring the VPS forensics shape: belly 0.7/0.3,
    wing re-learned to pricer 0.978 / market 0.022."""
    return {
        BELLY_REGION: _bankrolls(0.7, 0.3),
        WING_REGION: _bankrolls(0.978, 0.022, update_count=update_count),
    }


def test_pin_binds_wing_consensus_belly_per_strike_unchanged():
    strikes, pricer, market = _mixed_region_inputs()
    res = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _pin_states(), CFG,
    )
    cp = res.fair_value.consensus_p
    # Wing strikes price at the 0.5/0.5 PIN blend of the sanitized ladders,
    # NOT the stored 0.978/0.022 blend.
    assert cp[1.0] == pytest.approx(0.5 * 0.95 + 0.5 * 0.9, abs=1e-9)
    assert cp[5.0] == pytest.approx(0.5 * 0.15 + 0.5 * 0.1, abs=1e-9)
    # Belly PER-STRIKE consensus unchanged vs legacy (0.7/0.3 blend). NOTE:
    # deliberately NOT asserting belly BANKROLLS unchanged -- the whole-ladder
    # consensus feeds belly factors, so belly bankroll trajectories may shift.
    legacy = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _pin_states(),
        LEGACY_CFG,
    )
    for k, pv, mv in [(2.0, 0.75, 0.7), (3.0, 0.55, 0.5), (4.0, 0.35, 0.3)]:
        assert cp[k] == pytest.approx(0.7 * pv + 0.3 * mv, abs=1e-9)
        assert cp[k] == pytest.approx(legacy.fair_value.consensus_p[k], abs=1e-12)
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY


def test_pin_persists_first_tick_and_update_path_update_count_unchanged():
    strikes, pricer, market = _mixed_region_inputs()
    res1 = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market),
        _pin_states(update_count=7), CFG,
    )
    # First tick (prev history None) -- a non-fallback SKIP path: the
    # persisted wing state is the pinned dict, update_count untouched.
    assert res1.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res1.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})
    assert res1.bankroll_states[WING_REGION].update_count == 7
    assert res1.bankroll_states[WING_REGION].frozen is False

    # Second tick WITH threaded history -- the UPDATE path: belly Bayes
    # fires, wing stays pinned and its update_count stays frozen at 7.
    market2 = [0.91, 0.68, 0.52, 0.31, 0.11]
    res2 = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market2),
        res1.bankroll_states, CFG,
        prev_forecasts=res1.forecasts, prev_consensus=res1.consensus_bucket,
    )
    assert res2.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res2.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})
    assert res2.bankroll_states[WING_REGION].update_count == 7
    assert (res2.bankroll_states[BELLY_REGION].update_count
            == res1.bankroll_states[BELLY_REGION].update_count + 1)


def test_pin_persists_on_frozen_wing_skip_path():
    strikes, pricer, market = _mixed_region_inputs()
    states = {
        BELLY_REGION: _bankrolls(0.6, 0.4),
        WING_REGION: _bankrolls(0.9, 0.1, frozen=True, update_count=3),
    }
    res1 = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), states, CFG,
    )
    # Pin applies regardless of the wing frozen flag; the flag itself is
    # carried through untouched.
    assert res1.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res1.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})
    assert res1.bankroll_states[WING_REGION].frozen is True
    assert res1.bankroll_states[WING_REGION].update_count == 3

    market2 = [0.91, 0.68, 0.52, 0.31, 0.11]
    res2 = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market2),
        res1.bankroll_states, CFG,
        prev_forecasts=res1.forecasts, prev_consensus=res1.consensus_bucket,
    )
    assert res2.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})
    assert res2.bankroll_states[WING_REGION].frozen is True
    assert res2.bankroll_states[WING_REGION].update_count == 3


def test_pin_negative_disables_byte_identical_legacy():
    # Regression pin on the legacy semantics the pin replaces: with pin =
    # -1.0 the wing strikes price at the STORED weights, the wing credibility
    # reports the stored weight, the persisted wing row is the stored dict,
    # and the wing Bayes update still fires on the next tick.
    strikes, pricer, market = _mixed_region_inputs()
    res1 = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _pin_states(),
        LEGACY_CFG,
    )
    cp = res1.fair_value.consensus_p
    assert cp[1.0] == pytest.approx(0.978 * 0.95 + 0.022 * 0.9, abs=1e-9)
    assert cp[5.0] == pytest.approx(0.978 * 0.15 + 0.022 * 0.1, abs=1e-9)
    assert res1.fair_value.credibility_by_region[WING_REGION] == pytest.approx(0.978)
    assert res1.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.978, "market": 0.022})
    assert res1.bankroll_states[WING_REGION].update_count == 0

    market2 = [0.89, 0.69, 0.51, 0.31, 0.11]
    res2 = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market2),
        res1.bankroll_states, LEGACY_CFG,
        prev_forecasts=res1.forecasts, prev_consensus=res1.consensus_bucket,
    )
    assert res2.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res2.bankroll_states[WING_REGION].update_count == 1  # Bayes ran


def test_pin_rescues_degenerate_wing_bankrolls_with_warning(caplog):
    strikes, pricer, market = _mixed_region_inputs()
    states = {
        BELLY_REGION: _bankrolls(0.6, 0.4),
        WING_REGION: _bankrolls(0.0, 0.0),  # zero-sum -> degenerate stored row
    }
    with caplog.at_level(logging.WARNING, logger="market_maker.fair_value_anchor"):
        res = compute_fair_value(
            _snapshot(strikes, pricer), _mids(strikes, market), states, CFG,
        )
    # NO fallback: the pin rescues the tick, loudly, and the returned wing
    # state is pinned (so the degenerate stored row gets overwritten).
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res.bankroll_states[WING_REGION].frozen is False
    assert res.bankroll_states[BELLY_REGION].frozen is False
    assert res.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})
    assert "pin rescued" in caplog.text

    # Contrast: legacy (pin off) still falls back on the same degenerate
    # wing row (the `weights_wing is None and not pinned` clause).
    states2 = {
        BELLY_REGION: _bankrolls(0.6, 0.4),
        WING_REGION: _bankrolls(0.0, 0.0),
    }
    legacy = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), states2, LEGACY_CFG,
    )
    assert legacy.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK


def test_pin_clamped_into_floor_band():
    # pin = 1.0 with bankroll_floor = 0.02 -> effective 0.98/0.02 (the
    # module's floor invariant holds at read time).
    cfg = MMConfig(wing_pricer_weight_pin=1.0)
    strikes, pricer, market = _mixed_region_inputs()
    res = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _pin_states(), cfg,
    )
    cp = res.fair_value.consensus_p
    assert cp[1.0] == pytest.approx(0.98 * 0.95 + 0.02 * 0.9, abs=1e-9)
    assert res.fair_value.credibility_by_region[WING_REGION] == pytest.approx(0.98)
    assert res.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.98, "market": 0.02})


def test_pin_credibility_reporting_scalar_strike_weighted():
    strikes, pricer, market = _mixed_region_inputs()  # 3 belly + 2 wing strikes
    res = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _pin_states(), CFG,
    )
    assert res.fair_value.credibility_by_region[WING_REGION] == pytest.approx(0.5)
    assert res.fair_value.credibility_by_region[BELLY_REGION] == pytest.approx(0.7)
    # Scalar = strike-count-weighted average USING the pin for wing.
    assert res.fair_value.credibility == pytest.approx(
        (3 * 0.7 + 2 * 0.5) / 5, abs=1e-9)


# ===========================================================================
# Item 3 (2026-08-10 skew-fix wave): bankroll update tempering
# ===========================================================================
#
# MMConfig.bankroll_update_temper (default 0.1) tempers each region's
# per-tick Bayes-factor vector (factors = factors ** t) after the existing
# non-finite-factor check (3.3) and before the weight update, for every
# UNPINNED region reaching that point. temper=1.0 (or any clamped-to-1.0
# garbage value) is a true no-op -- see fair_value_anchor._bankroll_update_
# temper and the module docstring "BANKROLL UPDATE TEMPERING" section.
#
# All tests below use wing_pricer_weight_pin=-1.0 (pin disabled) so tempering
# is actually exercised on BOTH regions' own Bayes math, unless a test's
# purpose is specifically the pin/temper interaction.


def _temper_ladder():
    """5-strike ladder spanning both regions (belly-classified interior,
    wing-classified extremes via the tail-bucket rule), monotone throughout
    so sanitized == raw and no cummin/band-clamp step engages."""
    strikes = [1.0, 2.0, 3.0, 4.0, 5.0]
    pricer = [0.95, 0.75, 0.55, 0.35, 0.15]
    market0 = [0.6, 0.55, 0.5, 0.45, 0.4]
    market1 = [0.75, 0.65, 0.53, 0.42, 0.32]  # drifts toward pricer on every strike
    return strikes, pricer, market0, market1


def _run_two_ticks(temper, pin=-1.0, start=(0.5, 0.5)):
    strikes, pricer, market0, market1 = _temper_ladder()
    cfg = MMConfig(wing_pricer_weight_pin=pin, bankroll_update_temper=temper)
    states = _states(*start)
    r1 = compute_fair_value(_snapshot(strikes, pricer), _mids(strikes, market0), states, cfg)
    r2 = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market1), r1.bankroll_states, cfg,
        prev_forecasts=r1.forecasts, prev_consensus=r1.consensus_bucket,
    )
    return r1, r2


def test_temper_one_is_a_true_noop_vs_temper_disabled():
    # temper=1.0 read from the config field must be indistinguishable from a
    # world where tempering does not run at all (monkeypatch the resolver to
    # unconditionally report 1.0, ignoring whatever the config says) -- this
    # proves the factors**t step is genuinely skipped for temper=1.0, not
    # merely numerically close (factors**1.0 == factors exactly, but the
    # implementation short-circuits on `if temper < 1.0` so 1.0 never even
    # reaches the ** operator).
    import market_maker.fair_value_anchor as fva_mod

    _, r2_field = _run_two_ticks(1.0)

    class _NoTemperPatch:
        def __enter__(self):
            self._orig = fva_mod._bankroll_update_temper
            fva_mod._bankroll_update_temper = lambda config: 1.0
            return self

        def __exit__(self, *a):
            fva_mod._bankroll_update_temper = self._orig

    with _NoTemperPatch():
        _, r2_patched = _run_two_ticks(0.1)  # config says 0.1; resolver forced to 1.0

    for region in (BELLY_REGION, WING_REGION):
        assert r2_field.bankroll_states[region].bankrolls == pytest.approx(
            r2_patched.bankroll_states[region].bankrolls, abs=0.0)


def test_garbage_temper_clamps_to_legacy_unit_level():
    assert fva._bankroll_update_temper(MMConfig(bankroll_update_temper=0.0)) == 1.0
    assert fva._bankroll_update_temper(MMConfig(bankroll_update_temper=-1.0)) == 1.0
    assert fva._bankroll_update_temper(MMConfig(bankroll_update_temper=float("nan"))) == 1.0
    assert fva._bankroll_update_temper(MMConfig(bankroll_update_temper=1.5)) == 1.0

    class _StrTemper:
        bankroll_update_temper = "x"

    assert fva._bankroll_update_temper(_StrTemper()) == 1.0

    class _NoTemperAttr:
        pass

    assert fva._bankroll_update_temper(_NoTemperAttr()) == 1.0
    # Valid values pass through unchanged.
    assert fva._bankroll_update_temper(MMConfig(bankroll_update_temper=0.1)) == pytest.approx(0.1)
    assert fva._bankroll_update_temper(MMConfig(bankroll_update_temper=1.0)) == 1.0


def test_garbage_temper_integration_matches_untempered():
    baseline_r1, baseline_r2 = _run_two_ticks(1.0)
    for garbage in (0.0, -5.0, float("nan"), 2.0):
        _, candidate_r2 = _run_two_ticks(garbage)
        for region in (BELLY_REGION, WING_REGION):
            assert candidate_r2.bankroll_states[region].bankrolls == pytest.approx(
                baseline_r2.bankroll_states[region].bankrolls, abs=1e-12)


def test_temper_slows_learning_same_direction_smaller_step():
    # Each region's own direction (which model gains credibility) is a
    # function of the bucket-mass dynamics, not a global "pricer always
    # wins" rule -- belly and wing can (and here do) move opposite ways.
    # The temper guarantee is PER REGION: same sign of movement, smaller
    # magnitude -- not that both regions move the same way as each other.
    _, untempered = _run_two_ticks(1.0)
    _, tempered = _run_two_ticks(0.1)
    for region in (BELLY_REGION, WING_REGION):
        d_un = untempered.bankroll_states[region].bankrolls["pricer"] - 0.5
        d_te = tempered.bankroll_states[region].bankrolls["pricer"] - 0.5
        # same direction relative to the 0.5 start ...
        assert d_un * d_te > 0
        # ...but a strictly smaller step under tempering.
        assert abs(d_te) < abs(d_un)


def test_temper_default_config_value_matches_explicit_point_one():
    # MMConfig()'s default (0.1) must actually be threaded through -- not
    # just the explicit-field path.
    strikes, pricer, market0, market1 = _temper_ladder()
    cfg_default = MMConfig(wing_pricer_weight_pin=-1.0)  # temper defaults to 0.1
    cfg_explicit = MMConfig(wing_pricer_weight_pin=-1.0, bankroll_update_temper=0.1)
    assert cfg_default.bankroll_update_temper == pytest.approx(0.1)

    states_a = _states(0.5, 0.5)
    r1a = compute_fair_value(_snapshot(strikes, pricer), _mids(strikes, market0), states_a, cfg_default)
    r2a = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market1), r1a.bankroll_states, cfg_default,
        prev_forecasts=r1a.forecasts, prev_consensus=r1a.consensus_bucket,
    )
    states_b = _states(0.5, 0.5)
    r1b = compute_fair_value(_snapshot(strikes, pricer), _mids(strikes, market0), states_b, cfg_explicit)
    r2b = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market1), r1b.bankroll_states, cfg_explicit,
        prev_forecasts=r1b.forecasts, prev_consensus=r1b.consensus_bucket,
    )
    for region in (BELLY_REGION, WING_REGION):
        assert r2a.bankroll_states[region].bankrolls == pytest.approx(
            r2b.bankroll_states[region].bankrolls, abs=1e-15)


def test_floor_enforced_under_temper():
    # Same scenario/shape as test_bankroll_floor_holds, but with tempering
    # explicitly enabled (t=0.1) and 10x the ticks (matching the documented
    # ~10x slower full-range convergence) -- the floor clip in `_apply_floor`
    # is unconditional on every update, so it must never be violated at any
    # tick along the way, AND it must still be reachable given enough ticks
    # (tempering slows the approach, it does not disable the floor).
    strikes = [1.0, 2.0, 3.0]
    pricer = np.array([0.8, 0.5, 0.2])
    market0 = np.array([0.55, 0.5, 0.45])
    T = 600
    cfg = MMConfig(wing_pricer_weight_pin=-1.0, bankroll_update_temper=0.1)
    states = _states(0.5, 0.5)
    prev_fc = None
    prev_cons = None
    for t in range(T):
        frac = min(t / 100.0, 1.0)  # informative window scaled 10x vs the untempered test
        market_t = market0 + frac * (pricer - market0)
        res = compute_fair_value(
            _snapshot(strikes, pricer), _mids(strikes, market_t), states, cfg,
            prev_forecasts=prev_fc, prev_consensus=prev_cons,
        )
        states = res.bankroll_states
        prev_fc = res.forecasts
        prev_cons = res.consensus_bucket
        for region in (BELLY_REGION, WING_REGION):
            assert states[region].bankrolls["market"] >= DEFAULT_BANKROLL_FLOOR - 1e-9
            assert states[region].bankrolls["pricer"] <= 1.0 - DEFAULT_BANKROLL_FLOOR + 1e-9
    # Which model ends up at the floor is a function of the bucket-mass
    # dynamics per region (belly and wing can converge to OPPOSITE corners --
    # see test_temper_slows_learning_same_direction_smaller_step); the
    # direction-agnostic invariant is that the LOSING model in each region
    # has been driven all the way down to the floor by the end of this long
    # a run, proving tempering slows but does not prevent reaching it.
    for region in (BELLY_REGION, WING_REGION):
        assert min(states[region].bankrolls.values()) == pytest.approx(
            DEFAULT_BANKROLL_FLOOR, abs=1e-6)


def test_skip_paths_unchanged_under_temper_empty_region():
    # Empty-region skip (3.3): all-wing ladder -> belly owns zero buckets and
    # must skip cleanly (unchanged weights/update_count), identically whether
    # tempering is on or off (the skip `continue` precedes the temper line).
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.97, 0.92, 0.83]
    market0 = [0.95, 0.90, 0.85]
    cfg = MMConfig(wing_pricer_weight_pin=-1.0, bankroll_update_temper=0.1)
    states = _states(0.5, 0.5)
    snap = _snapshot(strikes, pricer)

    res1 = compute_fair_value(snap, _mids(strikes, market0), states, cfg)
    belly_before = dict(res1.bankroll_states[BELLY_REGION].bankrolls)
    belly_uc_before = res1.bankroll_states[BELLY_REGION].update_count
    wing_uc_before = res1.bankroll_states[WING_REGION].update_count

    market1 = [0.96, 0.94, 0.90]
    res2 = compute_fair_value(
        snap, _mids(strikes, market1), res1.bankroll_states, cfg,
        prev_forecasts=res1.forecasts, prev_consensus=res1.consensus_bucket,
    )
    assert res2.bankroll_states[BELLY_REGION].bankrolls == pytest.approx(belly_before)
    assert res2.bankroll_states[BELLY_REGION].update_count == belly_uc_before
    assert res2.bankroll_states[BELLY_REGION].frozen is False
    assert res2.bankroll_states[WING_REGION].frozen is False
    # Wing (non-empty) DID update -- count advanced under tempering too.
    assert res2.bankroll_states[WING_REGION].update_count == wing_uc_before + 1


def test_skip_paths_unchanged_under_temper_degenerate_factor():
    # Degenerate-factor skip (3.3): same construction as
    # test_degenerate_factor_skip_wing_belly_still_updates, with tempering
    # explicitly enabled -- wing's s_R == 0 skip and belly's normal update
    # must both behave identically to the untempered case.
    strikes = [1.0, 2.0, 3.0]
    pricer = [1.0, 0.6, 0.0]
    market0 = [0.75, 0.6, 0.25]  # all belly-classified
    cfg = MMConfig(wing_pricer_weight_pin=-1.0, bankroll_update_temper=0.1)
    states = {
        BELLY_REGION: _bankrolls(1.0, 0.0),
        WING_REGION: _bankrolls(0.5, 0.5),
    }
    snap = _snapshot(strikes, pricer)

    res1 = compute_fair_value(snap, _mids(strikes, market0), states, cfg)
    wing_before = dict(res1.bankroll_states[WING_REGION].bankrolls)
    wing_uc_before = res1.bankroll_states[WING_REGION].update_count
    belly_uc_before = res1.bankroll_states[BELLY_REGION].update_count

    market1 = [0.7, 0.55, 0.3]
    res2 = compute_fair_value(
        snap, _mids(strikes, market1), res1.bankroll_states, cfg,
        prev_forecasts=res1.forecasts, prev_consensus=res1.consensus_bucket,
    )
    assert res2.bankroll_states[WING_REGION].bankrolls == pytest.approx(wing_before)
    assert res2.bankroll_states[WING_REGION].update_count == wing_uc_before  # still skipped
    assert res2.bankroll_states[WING_REGION].frozen is False
    assert res2.bankroll_states[BELLY_REGION].update_count == belly_uc_before + 1  # still updates
    assert res2.bankroll_states[BELLY_REGION].frozen is False


def test_pinned_wing_unaffected_by_temper():
    # Wing pin (Fix 1) + tempering both default-on together (production
    # config shape): the pinned wing `continue` precedes factor computation
    # entirely, so tempering never gets a chance to touch it.
    strikes, pricer, market = _mixed_region_inputs()
    cfg = MMConfig(wing_pricer_weight_pin=0.5, bankroll_update_temper=0.1)
    res1 = compute_fair_value(_snapshot(strikes, pricer), _mids(strikes, market), _pin_states(), cfg)
    assert res1.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})

    market2 = [0.91, 0.68, 0.52, 0.31, 0.11]
    res2 = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market2), res1.bankroll_states, cfg,
        prev_forecasts=res1.forecasts, prev_consensus=res1.consensus_bucket,
    )
    assert res2.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})
    assert res2.bankroll_states[WING_REGION].update_count == res1.bankroll_states[WING_REGION].update_count
    # Belly (unpinned) did move under tempering -- contrast to prove temper
    # is actually wired for the region that isn't pin-shielded.
    assert res2.bankroll_states[BELLY_REGION].update_count == res1.bankroll_states[BELLY_REGION].update_count + 1
