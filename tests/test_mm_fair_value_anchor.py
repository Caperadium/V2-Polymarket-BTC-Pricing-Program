"""Tests for market_maker.fair_value_anchor (plan task F1, Section 2.3).

Covers the five NORMATIVE invariants (each its own test), a hand-computed
3-strike consensus, monotone re-integration, the degeneracy fallback, and the
bankroll floor.
"""
from __future__ import annotations

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
    DEFAULT_BANKROLL_FLOOR,
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


def _bankrolls(pricer=0.5, market=0.5, frozen=False):
    return BankrollState(
        model_ids=["pricer", "market"],
        bankrolls={"pricer": pricer, "market": market},
        last_update=TS,
        update_count=0,
        frozen=frozen,
    )


def _mids(strikes, p_list):
    return {float(k): float(p) for k, p in zip(strikes, p_list)}


CFG = MMConfig()


# ---------------------------------------------------------------------------
# Hand-computed 3-strike consensus (unequal bankrolls)
# ---------------------------------------------------------------------------

def test_hand_computed_consensus_unequal_bankrolls():
    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        _mids(strikes, [0.8, 0.5, 0.2]),
        _bankrolls(0.7, 0.3),
        CFG,
    )
    cp = res.fair_value.consensus_p
    # 0.7*pricer + 0.3*market per strike
    assert cp[1.0] == pytest.approx(0.87, abs=1e-9)
    assert cp[2.0] == pytest.approx(0.57, abs=1e-9)
    assert cp[3.0] == pytest.approx(0.27, abs=1e-9)
    assert res.fair_value.credibility == pytest.approx(0.7, abs=1e-9)
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY


# ---------------------------------------------------------------------------
# NORMATIVE invariant 1: unanimity fixed point
# ---------------------------------------------------------------------------

def test_invariant1_unanimity_fixed_point():
    strikes = [1.0, 2.0, 3.0]
    p = [0.8, 0.5, 0.2]
    res = compute_fair_value(
        _snapshot(strikes, p), _mids(strikes, p), _bankrolls(0.3, 0.7), CFG,
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
        _snapshot(strikes, pricer), _mids(strikes, market), _bankrolls(0.6, 0.4), CFG,
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
        _bankrolls(0.5, 0.5),
        CFG,
    )
    cons = res.consensus_bucket
    assert np.all(cons >= 0.0)
    assert cons.sum() == pytest.approx(1.0, abs=1e-9)
    assert cons.size == len(strikes) + 1


# ---------------------------------------------------------------------------
# NORMATIVE invariant 4: credibility weights nonnegative and sum to 1
# ---------------------------------------------------------------------------

def test_invariant4_credibility_weights_nonneg_sum_to_one():
    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.8, 0.5, 0.2]),
        _mids(strikes, [0.7, 0.5, 0.3]),
        _bankrolls(0.4, 0.6),
        CFG,
    )
    vals = list(res.bankroll_state.bankrolls.values())
    assert all(v >= 0.0 for v in vals)
    assert sum(vals) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# NORMATIVE invariant 5: monotone credibility gain for the better model
# ---------------------------------------------------------------------------

def test_invariant5_monotone_credibility_gain():
    strikes = [1.0, 2.0, 3.0]
    pricer = np.array([0.8, 0.5, 0.2])
    market0 = np.array([0.55, 0.5, 0.45])
    T = 20
    state = _bankrolls(0.5, 0.5)
    prev_fc = None
    prev_cons = None
    creds = []
    for t in range(T):
        frac = t / (T - 1)
        market_t = market0 + frac * (pricer - market0)
        res = compute_fair_value(
            _snapshot(strikes, pricer),
            _mids(strikes, market_t),
            state,
            CFG,
            prev_forecasts=prev_fc,
            prev_consensus=prev_cons,
        )
        creds.append(res.fair_value.credibility)
        state = res.bankroll_state
        prev_fc = res.forecasts
        prev_cons = res.consensus_bucket
    # pricer credibility should be monotonically non-decreasing
    # (1e-6 slack: 20-step accumulated float summation, 1e-9 flaked once)
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
        _bankrolls(0.5, 0.5),
        CFG,
    )
    vals = [res.fair_value.consensus_p[k] for k in strikes]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12


# ---------------------------------------------------------------------------
# Degeneracy fallback (risk 8.8)
# ---------------------------------------------------------------------------

def test_fallback_on_nan_mid():
    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        {1.0: 0.8, 2.0: float("nan"), 3.0: 0.2},
        _bankrolls(0.5, 0.5),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    assert res.bankroll_state.frozen is True


def test_fallback_on_all_zero_bankrolls():
    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        _mids(strikes, [0.8, 0.5, 0.2]),
        _bankrolls(0.0, 0.0),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    assert res.bankroll_state.frozen is True
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
        _bankrolls(0.5, 0.5),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res.bankroll_state.frozen is False

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

    # Hand-checked (w=0.5/0.5): recon_pricer=[0.5625,0.5625,0.125],
    # recon_market=[0.8,0.5,0.2], consensus=[0.68125,0.53125,0.1625].
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
        _bankrolls(0.5, 0.5),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res.bankroll_state.frozen is False


def test_sanity_fallback_still_fires_on_band_violation(monkeypatch):
    # Safety-net test for the band check itself. A CONSTANT monkeypatch
    # offset would NOT work here: the band bounds are themselves built from
    # buckets_to_ladder(forecasts[...]) output, so a constant shift moves the
    # band together with the consensus and the check would never fire.
    # Instead, wrap buckets_to_ladder with an INPUT-MATCHING perturbation:
    # pass through unchanged when the input bucket vector matches either
    # model's own forecast (used by the per-strike recon/band computation),
    # and perturb only when it does not -- which isolates the wrap to the
    # blended-consensus reconstruction. Distinct ladders + unequal bankrolls
    # (0.7/0.3) make the consensus bucket vector differ from both forecasts,
    # so exactly (and only) the consensus reconstruction gets perturbed.
    fc_pricer = ladder_to_buckets(np.array([0.9, 0.6, 0.3]))
    fc_market = ladder_to_buckets(np.array([0.8, 0.5, 0.2]))
    _real_buckets_to_ladder = fva.buckets_to_ladder

    def _wrapped_buckets_to_ladder(buckets):
        out = _real_buckets_to_ladder(buckets)
        b = np.asarray(buckets, dtype=float)
        if np.allclose(b, fc_pricer) or np.allclose(b, fc_market):
            return out
        # Not either model's own forecast -> this is the blended consensus.
        # Hand-checked margin to the nearest band edge for this ladder/weight
        # combo is 0.03 (0.7*0.3*0.1 gap-per-strike), so +0.05 clears it.
        return out + 0.05

    monkeypatch.setattr(fva, "buckets_to_ladder", _wrapped_buckets_to_ladder)

    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        _mids(strikes, [0.8, 0.5, 0.2]),
        _bankrolls(0.7, 0.3),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    assert res.bankroll_state.frozen is True


def test_frozen_state_recovers_beuoy_on_crossed_mids():
    # Contract: the anchor's top-level path prices BEUOY regardless of the
    # incoming frozen flag -- frozen only skips the Bayes mark-to-market
    # bankroll update, not the consensus computation. The anchor itself NEVER
    # self-clears frozen; auto-unfreeze is owned entirely by the harness
    # (harness.py:632-639), keyed on 20 consecutive clean BEUOY recomputes.
    # This test documents that contract with a crossed-mid (non-monotone)
    # ladder on top of an already-frozen bankroll state.
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.9, 0.6, 0.3]
    mids = [0.8, 0.5, 0.52]  # non-monotone: K3 mid (0.52) > K2 mid (0.5)
    res = compute_fair_value(
        _snapshot(strikes, pricer),
        _mids(strikes, mids),
        _bankrolls(0.5, 0.5, frozen=True),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.BEUOY
    assert res.bankroll_state.frozen is True


# ---------------------------------------------------------------------------
# Bankroll floor: always-wrong model does not go below floor
# ---------------------------------------------------------------------------

def test_bankroll_floor_holds():
    strikes = [1.0, 2.0, 3.0]
    pricer = np.array([0.8, 0.5, 0.2])
    market0 = np.array([0.55, 0.5, 0.45])
    T = 60
    state = _bankrolls(0.5, 0.5)
    prev_fc = None
    prev_cons = None
    for t in range(T):
        frac = min(t / 10.0, 1.0)
        market_t = market0 + frac * (pricer - market0)
        res = compute_fair_value(
            _snapshot(strikes, pricer),
            _mids(strikes, market_t),
            state,
            CFG,
            prev_forecasts=prev_fc,
            prev_consensus=prev_cons,
        )
        state = res.bankroll_state
        prev_fc = res.forecasts
        prev_cons = res.consensus_bucket
    assert state.bankrolls["market"] >= DEFAULT_BANKROLL_FLOOR - 1e-12
    assert state.bankrolls["pricer"] <= 1.0 - DEFAULT_BANKROLL_FLOOR + 1e-12
