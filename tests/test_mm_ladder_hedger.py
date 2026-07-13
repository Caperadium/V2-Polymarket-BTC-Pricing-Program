"""Tests for market_maker.ladder_hedger (plan tasks L1 + L2)."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from market_maker.config import MMConfig
from market_maker.contracts import (
    ContractInv,
    HedgeReason,
    HedgeRecommendation,
    InventoryState,
    QuoteMode,
    QuoteSet,
    Side,
)
from market_maker.ladder_hedger import LadderHedger, hedge_offsets_by_market

TS = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


def _qs(market_id: str, bid: float, ask: float) -> QuoteSet:
    return QuoteSet(
        ts=TS,
        market_id=market_id,
        bid_price=bid,
        ask_price=ask,
        bid_size=10.0,
        ask_size=10.0,
        terms={},
        risk_mode=QuoteMode.TWO_SIDED,
        noarb_checked=False,
        source_seq=0,
    )


def _inv(items):
    per = {
        m: ContractInv(q=q, avg_cost=0.5, q_max=qm, age_weighted_holding=0.0)
        for (m, q, qm) in items
    }
    return InventoryState(ts=TS, per_contract=per, per_ladder={})


# --- (a) no-arb check + repair -------------------------------------------


def test_clean_ladder_passes_check():
    h = LadderHedger()
    strikes = [90000.0, 100000.0, 110000.0]
    qs = [_qs("m0", 0.78, 0.82), _qs("m1", 0.48, 0.52), _qs("m2", 0.18, 0.22)]
    v = h.check(qs, strikes)
    assert v.ok
    assert v.violations == []


def test_violating_ladder_named_violations():
    h = LadderHedger()
    strikes = [90000.0, 100000.0, 110000.0]
    qs = [_qs("m0", 0.48, 0.52), _qs("m1", 0.78, 0.82), _qs("m2", 0.18, 0.22)]
    v = h.check(qs, strikes)
    assert not v.ok
    kinds = " ".join(v.violations)
    assert "bid_monotonicity" in kinds
    assert "ask_monotonicity" in kinds
    assert "negative_density" in kinds


def test_repair_produces_monotone_and_preserves_half_spread():
    h = LadderHedger()
    strikes = [90000.0, 100000.0, 110000.0]
    qs = [_qs("m0", 0.48, 0.52), _qs("m1", 0.78, 0.82), _qs("m2", 0.18, 0.22)]
    cdf = {90000.0: 0.5, 100000.0: 0.5, 110000.0: 0.2}
    out = h.repair(qs, strikes, cdf)
    assert out is not None
    mids = [0.5 * (q.bid_price + q.ask_price) for q in out]
    # monotone non-increasing
    for i in range(len(mids) - 1):
        assert mids[i] - mids[i + 1] >= -1e-9  # density >= 0
    # half-spreads preserved within a tick
    for orig, rep in zip(qs, out):
        hs0 = 0.5 * (orig.ask_price - orig.bid_price)
        hs1 = 0.5 * (rep.ask_price - rep.bid_price)
        assert abs(hs0 - hs1) <= h.tick + 1e-9
        assert rep.noarb_checked


def test_repair_idempotent():
    h = LadderHedger()
    strikes = [90000.0, 100000.0, 110000.0]
    qs = [_qs("m0", 0.48, 0.52), _qs("m1", 0.78, 0.82), _qs("m2", 0.18, 0.22)]
    cdf = {90000.0: 0.5, 100000.0: 0.5, 110000.0: 0.2}
    once = h.repair(qs, strikes, cdf)
    twice = h.repair(once, strikes, cdf)
    for a, b in zip(once, twice):
        assert a.bid_price == b.bid_price
        assert a.ask_price == b.ask_price


def test_reject_mode_returns_none_and_journals():
    h = LadderHedger(repair_or_reject="reject")
    strikes = [90000.0, 100000.0, 110000.0]
    qs = [_qs("m0", 0.48, 0.52), _qs("m1", 0.78, 0.82), _qs("m2", 0.18, 0.22)]
    out = h.repair(qs, strikes, {}, expiry_key="2026-08-01")
    assert out is None
    assert len(h.journal) == 1
    assert h.journal[0]["event"] == "reject"
    assert h.journal[0]["violations"]
    assert h.repair_count == 1


def test_repair_count_increments_on_violation_only():
    h = LadderHedger()
    strikes = [90000.0, 100000.0, 110000.0]
    cdf = {90000.0: 0.5, 100000.0: 0.5, 110000.0: 0.2}
    clean = [_qs("m0", 0.68, 0.72), _qs("m1", 0.48, 0.52), _qs("m2", 0.18, 0.22)]
    assert h.repair(clean, strikes, cdf) is not None
    assert h.repair_count == 0  # clean ladder: no increment
    bad = [_qs("m0", 0.48, 0.52), _qs("m1", 0.78, 0.82), _qs("m2", 0.18, 0.22)]
    once = h.repair(bad, strikes, cdf)
    assert once is not None
    assert h.repair_count == 1
    # repaired output is clean: re-checking it does not increment again
    h.repair(once, strikes, cdf)
    assert h.repair_count == 1


# --- (b) vertical-spread internal hedge ----------------------------------


def test_vertical_hedge_over_cap_middle_strike():
    h = LadderHedger(vertical_target_frac=0.5, hedge_ttl_seconds=300.0)
    strikes = [90000.0, 100000.0, 110000.0]
    mids = ["m0", "m1", "m2"]
    inv = _inv([("m0", 0.0, 40.0), ("m1", 100.0, 50.0), ("m2", 0.0, 40.0)])
    fair = {"m0": 0.7, "m1": 0.5, "m2": 0.3}
    recs, hedge_state = h.vertical_hedges(inv, "2026-08-01", strikes, mids, fair, TS)
    assert len(recs) == 1
    r = recs[0]
    # target = 0.5*50 = 25; excess = 100-25 = 75; neighbor cap = 40; size = 40
    assert r.size == 40.0
    assert r.side == Side.BUY_NO  # long YES -> buy NO to offset
    assert r.reason == HedgeReason.VERTICAL_OFFSET
    assert r.target_market_id == "m0"  # tie -> lower strike neighbor
    assert r.paired_market_id == "m1"
    assert r.expires == TS + timedelta(seconds=300.0)
    assert r.beta is None
    # hedge_state signed offset on bucket 0
    assert hedge_state[("2026-08-01", 0)] == -40.0


def test_vertical_hedge_depth_hint_prefers_liquidity():
    h = LadderHedger()
    strikes = [90000.0, 100000.0, 110000.0]
    mids = ["m0", "m1", "m2"]
    inv = _inv([("m0", 0.0, 40.0), ("m1", 100.0, 50.0), ("m2", 0.0, 40.0)])
    fair = {"m0": 0.7, "m1": 0.5, "m2": 0.3}
    # higher depth on m2 -> chosen despite equal strike distance
    recs, _ = h.vertical_hedges(
        inv, "e", strikes, mids, fair, TS, depth_hint={"m0": 1.0, "m2": 99.0}
    )
    assert recs[0].target_market_id == "m2"


# --- (c) cross-strike beta hedge -----------------------------------------


def test_beta_zero_outside_band():
    h = LadderHedger(enable_beta_hedge=True)
    # p_j at/above the 0.999 clamp -> beta forced to 0
    assert h.beta_ratio(0.5, 0.999, 0.5) == 0.0
    assert h.beta_ratio(0.5, 0.9995, 0.5) == 0.0
    # just inside the band -> nonzero but clamped
    inside = h.beta_ratio(0.5, 0.9989, 0.5)
    assert abs(inside) <= h.config.beta_max


def test_beta_clamped_property():
    h = LadderHedger(enable_beta_hedge=True)
    rng = random.Random(7)
    for _ in range(2000):
        p_i = rng.uniform(1e-4, 1 - 1e-4)
        p_j = rng.uniform(1e-4, 1 - 1e-4)
        sig = rng.uniform(1e-6, 5.0)
        b = h.beta_ratio(p_i, p_j, sig)
        assert abs(b) <= h.config.beta_max + 1e-9
        assert b == b  # not NaN


def test_beta_disabled_by_default():
    h = LadderHedger()  # enable_beta_hedge defaults False
    inv = _inv([("m0", 100.0, 10.0)])
    recs = h.beta_hedges(inv, "e", [90000.0], ["m0"], {"m0": 0.5}, {"m0": 0.5}, TS)
    assert recs == []


def test_beta_hedge_notional_bounded_adversarial():
    h = LadderHedger(enable_beta_hedge=True)
    strikes = [90000.0, 100000.0]
    mids = ["m0", "m1"]
    rng = random.Random(11)
    for _ in range(500):
        qj = rng.uniform(1.0, 100.0)
        inv = _inv([("m0", 0.0, 5.0), ("m1", qj, 5.0)])
        p_i = rng.uniform(1e-4, 1 - 1e-4)
        p_j = rng.uniform(1e-4, 1 - 1e-4)
        sig = rng.uniform(1e-9, 1e-6)  # tiny sigma_b -> raw beta explodes
        recs = h.beta_hedges(
            inv, "e", strikes, mids, {"m0": p_i, "m1": p_j}, {"m1": sig}, TS
        )
        for r in recs:
            assert r.size <= h.config.beta_max * qj + 1e-6
            assert 0.0 <= r.max_price <= 1.0
            assert r.reason == HedgeReason.BETA_HEDGE


# --- W2.0 hedge_offsets_by_market -----------------------------------------


def _rec(target_market_id, side, size, paired="paired", max_price=0.5):
    return HedgeRecommendation(
        ts=TS, expiry_key="e", target_market_id=target_market_id, side=side,
        size=size, max_price=max_price, reason=HedgeReason.VERTICAL_OFFSET,
        paired_market_id=paired, beta=None, expires=TS + timedelta(seconds=300.0),
    )


def test_hedge_offsets_by_market_signs():
    recs = [
        _rec("m0", Side.BUY_YES, 10.0),
        _rec("m2", Side.BUY_NO, 7.0),
    ]
    offsets = hedge_offsets_by_market(recs)
    assert offsets == {"m0": 10.0, "m2": -7.0}


def test_hedge_offsets_by_market_aggregates_same_target():
    recs = [
        _rec("m0", Side.BUY_YES, 10.0, paired="m1"),
        _rec("m0", Side.BUY_YES, 5.0, paired="m2"),
        _rec("m0", Side.BUY_NO, 3.0, paired="m1"),
    ]
    offsets = hedge_offsets_by_market(recs)
    # 10 + 5 - 3 = 12, all landing on the same target market_id
    assert offsets == {"m0": 12.0}


def test_hedge_offsets_by_market_empty_recs():
    assert hedge_offsets_by_market([]) == {}
