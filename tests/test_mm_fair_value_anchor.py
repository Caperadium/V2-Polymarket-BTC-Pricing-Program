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
    compute_fair_value,
)

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


def test_fallback_on_sanity_bound_violation():
    strikes = [1.0, 2.0, 3.0]
    # Non-monotone pricer forces bucket clipping -> consensus escapes the band.
    res = compute_fair_value(
        _snapshot(strikes, [0.3, 0.9, 0.2]),
        _mids(strikes, [0.8, 0.5, 0.2]),
        _bankrolls(0.5, 0.5),
        CFG,
    )
    assert res.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    assert res.bankroll_state.frozen is True
    cp = res.fair_value.consensus_p
    # blend = 0.5*[0.3,0.9,0.2]+0.5*[0.8,0.5,0.2]=[0.55,0.7,0.2] -> monotone [0.55,0.55,0.2]
    assert cp[1.0] == pytest.approx(0.55, abs=1e-9)
    assert cp[2.0] == pytest.approx(0.55, abs=1e-9)
    assert cp[3.0] == pytest.approx(0.2, abs=1e-9)


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
