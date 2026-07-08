"""Tests for market_maker.inventory_manager (plan 2.6, task I1, contract 4.6)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import Fill, LiquiditySource, Side
from market_maker.inventory_manager import InventoryManager
from market_maker.logodds import logit, s_prime

T0 = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)


def _fill(ts, market_id, side, price, size, liquidity=LiquiditySource.MAKER, order_id="o1"):
    return Fill(
        ts=ts, market_id=market_id, order_id=order_id, side=side, price=price,
        size=size, liquidity=liquidity, venue_ts=ts,
    )


# ---------------------------------------------------------------------------
# Scripted fill stream -> hand-computed q per contract / ladder gross
# ---------------------------------------------------------------------------

def test_scripted_fill_stream_hand_computed_q():
    mgr = InventoryManager()
    mgr.register_market("m1", "2026-07-20", 100000.0)
    mgr.register_market("m2", "2026-07-20", 110000.0)

    mgr.apply_fill(_fill(T0, "m1", Side.BUY_YES, 0.5, 10.0))
    assert mgr._contracts["m1"].q == pytest.approx(10.0)
    assert mgr._contracts["m1"].avg_cost == pytest.approx(0.5)

    mgr.apply_fill(_fill(T0 + timedelta(hours=1), "m1", Side.BUY_YES, 0.6, 10.0))
    # same-direction add: volume-weighted avg_cost = (10*0.5 + 10*0.6)/20 = 0.55
    assert mgr._contracts["m1"].q == pytest.approx(20.0)
    assert mgr._contracts["m1"].avg_cost == pytest.approx(0.55)

    mgr.apply_fill(_fill(T0 + timedelta(hours=2), "m2", Side.BUY_NO, 0.3, 5.0))
    assert mgr._contracts["m2"].q == pytest.approx(-5.0)
    assert mgr._contracts["m2"].avg_cost == pytest.approx(0.3)

    gross = mgr._ladder_gross("2026-07-20")
    assert gross == pytest.approx(20.0 + 5.0)


def test_reducing_position_leaves_avg_cost_unchanged():
    mgr = InventoryManager()
    mgr.register_market("m1", "2026-07-20", 100000.0)
    mgr.apply_fill(_fill(T0, "m1", Side.BUY_YES, 0.4, 10.0))
    mgr.apply_fill(_fill(T0 + timedelta(hours=1), "m1", Side.BUY_NO, 0.7, 4.0))  # reduce, not flip
    assert mgr._contracts["m1"].q == pytest.approx(6.0)
    assert mgr._contracts["m1"].avg_cost == pytest.approx(0.4)  # unchanged


def test_flip_resets_avg_cost():
    mgr = InventoryManager()
    mgr.register_market("m1", "2026-07-20", 100000.0)
    mgr.apply_fill(_fill(T0, "m1", Side.BUY_YES, 0.4, 10.0))
    mgr.apply_fill(_fill(T0 + timedelta(hours=1), "m1", Side.BUY_NO, 0.9, 15.0))  # flips through 0
    assert mgr._contracts["m1"].q == pytest.approx(-5.0)
    assert mgr._contracts["m1"].avg_cost == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# SETTLEMENT fill closes q to 0 through the normal channel
# ---------------------------------------------------------------------------

def test_settlement_fill_closes_position_via_normal_channel():
    mgr = InventoryManager()
    mgr.register_market("m1", "2026-07-20", 100000.0)
    mgr.apply_fill(_fill(T0, "m1", Side.BUY_YES, 0.5, 10.0))
    assert mgr._contracts["m1"].q == pytest.approx(10.0)

    settle_ts = T0 + timedelta(days=19)
    mgr.apply_fill(_fill(settle_ts, "m1", Side.BUY_NO, 1.0, 10.0, liquidity=LiquiditySource.SETTLEMENT))
    assert mgr._contracts["m1"].q == pytest.approx(0.0)
    assert mgr._contracts["m1"].position_open_ts is None


# ---------------------------------------------------------------------------
# Cap shrinks as fair p -> extreme; cap_breached fires
# ---------------------------------------------------------------------------

def test_cap_shrinks_toward_extreme_p():
    mgr = InventoryManager()
    mgr.update_fair_x("m_mid", 0.0)  # p = 0.5
    mgr.update_fair_x("m_wing", logit(0.99))  # p = 0.99
    q_max_mid = mgr._contracts["m_mid"].q_max
    q_max_wing = mgr._contracts["m_wing"].q_max
    assert q_max_mid > q_max_wing
    cfg = MMConfig()
    assert q_max_mid == pytest.approx(cfg.q_max_scale * s_prime(0.0))
    assert q_max_wing == pytest.approx(cfg.q_max_scale * max(s_prime(logit(0.99)), cfg.s_prime_floor))


def test_cap_breached_fires_when_q_exceeds_q_max():
    mgr = InventoryManager()
    mgr.register_market("m1", "2026-07-20", 100000.0)
    mgr.update_fair_x("m1", logit(0.99))  # small cap
    q_max = mgr._contracts["m1"].q_max
    mgr.apply_fill(_fill(T0, "m1", Side.BUY_YES, 0.99, q_max + 1.0))
    assert mgr.cap_breached("m1") is True
    assert "m1" in mgr.breaches()


def test_cap_not_breached_within_limit():
    mgr = InventoryManager()
    mgr.register_market("m1", "2026-07-20", 100000.0)
    mgr.update_fair_x("m1", 0.0)
    q_max = mgr._contracts["m1"].q_max
    mgr.apply_fill(_fill(T0, "m1", Side.BUY_YES, 0.5, q_max - 1.0))
    assert mgr.cap_breached("m1") is False
    assert mgr.breaches() == []


# ---------------------------------------------------------------------------
# net_band_exposure with a two-strike ladder + vertical HedgeState offset
# ---------------------------------------------------------------------------

def test_net_band_exposure_two_strike_ladder_hand_computed():
    mgr = InventoryManager()
    mgr.register_market("m_low", "2026-07-20", 100000.0)
    mgr.register_market("m_high", "2026-07-20", 110000.0)
    mgr.apply_fill(_fill(T0, "m_low", Side.BUY_YES, 0.6, 10.0))   # q_low = 10
    mgr.apply_fill(_fill(T0, "m_high", Side.BUY_NO, 0.3, 4.0))    # q_high = -4

    # no hedge offsets: bucket0=0, bucket1=cumsum(10)=10, bucket2=cumsum(10,-4)=6
    exposure = mgr.net_band_exposure("2026-07-20")
    assert exposure == pytest.approx([0.0, 10.0, 6.0])

    # apply a vertical offset of -3 on m_low (partial internal hedge)
    mgr.set_hedge_state("2026-07-20", {"m_low": -3.0})
    exposure2 = mgr.net_band_exposure("2026-07-20")
    # q_eff_low = 10-3=7, q_eff_high=-4+0=-4 -> bucket1=7, bucket2=7-4=3
    assert exposure2 == pytest.approx([0.0, 7.0, 3.0])


def test_net_band_exposure_empty_ladder():
    mgr = InventoryManager()
    assert mgr.net_band_exposure("nonexistent") == [0.0]


# ---------------------------------------------------------------------------
# R3 histogram on a scripted timeline, hand-computed
# ---------------------------------------------------------------------------

def test_r3_histogram_hand_computed():
    mgr = InventoryManager()
    mgr.register_market("m1", "2026-07-20", 100000.0)

    # t=0: open q=5 (level 5); held for 2h
    mgr.apply_fill(_fill(T0, "m1", Side.BUY_YES, 0.5, 5.0))
    # t=2h: add 5 more -> q=10 (level 10); held for 3h
    mgr.apply_fill(_fill(T0 + timedelta(hours=2), "m1", Side.BUY_YES, 0.5, 5.0))
    # t=5h: mark only, no fill -> attributes 1h more at level 10
    mgr.mark(T0 + timedelta(hours=6))

    hist = mgr._ladders["2026-07-20"].r3_histogram
    # level 0 (before first fill) never entered since r3_last_ts starts at the first fill
    assert hist.get(5) == pytest.approx(2.0)
    assert hist.get(10) == pytest.approx(4.0)


def test_r3_histogram_accumulates_across_multiple_visits_to_same_level():
    mgr = InventoryManager()
    mgr.register_market("m1", "2026-07-20", 100000.0)
    mgr.apply_fill(_fill(T0, "m1", Side.BUY_YES, 0.5, 5.0))  # level 5
    mgr.mark(T0 + timedelta(hours=1))  # +1h at level 5
    mgr.apply_fill(_fill(T0 + timedelta(hours=1), "m1", Side.BUY_YES, 0.5, 5.0))  # level 10
    mgr.apply_fill(_fill(T0 + timedelta(hours=2), "m1", Side.BUY_NO, 0.5, 5.0))  # back to level 5
    mgr.mark(T0 + timedelta(hours=4))  # +2h at level 5

    hist = mgr._ladders["2026-07-20"].r3_histogram
    assert hist.get(5) == pytest.approx(1.0 + 2.0)
    assert hist.get(10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# to_rows / from_rows round trip
# ---------------------------------------------------------------------------

def test_to_rows_from_rows_round_trip():
    mgr = InventoryManager()
    mgr.register_market("m_low", "2026-07-20", 100000.0)
    mgr.register_market("m_high", "2026-07-20", 110000.0)
    mgr.update_fair_x("m_low", 0.2)
    mgr.set_phi("2026-07-20", 0.05)
    mgr.set_hedge_state("2026-07-20", {"m_low": -2.0})
    mgr.apply_fill(_fill(T0, "m_low", Side.BUY_YES, 0.6, 10.0))
    mgr.apply_fill(_fill(T0 + timedelta(hours=3), "m_high", Side.BUY_NO, 0.3, 4.0))
    mgr.mark(T0 + timedelta(hours=5))

    rows = mgr.to_rows()
    restored = InventoryManager.from_rows(rows)

    snap_orig = mgr.snapshot(T0 + timedelta(hours=5))
    snap_restored = restored.snapshot(T0 + timedelta(hours=5))

    assert snap_restored.per_contract.keys() == snap_orig.per_contract.keys()
    for mid in snap_orig.per_contract:
        a, b = snap_orig.per_contract[mid], snap_restored.per_contract[mid]
        assert a.q == pytest.approx(b.q)
        assert a.avg_cost == pytest.approx(b.avg_cost)
        assert a.q_max == pytest.approx(b.q_max)
        assert a.age_weighted_holding == pytest.approx(b.age_weighted_holding)

    for ek in snap_orig.per_ladder:
        a, b = snap_orig.per_ladder[ek], snap_restored.per_ladder[ek]
        assert a.net_band_exposure == pytest.approx(b.net_band_exposure)
        assert a.gross == pytest.approx(b.gross)
        assert a.phi == pytest.approx(b.phi)
        assert a.r3_histogram == pytest.approx(b.r3_histogram)


# ---------------------------------------------------------------------------
# Decision D1: q_max mode switch (shrinking default, Dalen mode dormant)
# ---------------------------------------------------------------------------

def test_q_max_dalen_mode_grows_at_wings():
    from market_maker.config import MMConfig
    from market_maker.inventory_manager import InventoryManager
    from market_maker.logodds import logit

    im_shrink = InventoryManager(MMConfig(q_max_mode="shrinking"))
    im_dalen = InventoryManager(MMConfig(q_max_mode="dalen"))
    for im in (im_shrink, im_dalen):
        im.register_market("m", "2026-07-20", 100000.0)

    x_mid, x_wing = logit(0.5), logit(0.99)

    im_shrink.update_fair_x("m", x_mid)
    mid_shrink = im_shrink.snapshot(T0).per_contract["m"].q_max
    im_shrink.update_fair_x("m", x_wing)
    wing_shrink = im_shrink.snapshot(T0).per_contract["m"].q_max
    assert wing_shrink < mid_shrink  # conservative default: shrinks at wings

    im_dalen.update_fair_x("m", x_mid)
    mid_dalen = im_dalen.snapshot(T0).per_contract["m"].q_max
    im_dalen.update_fair_x("m", x_wing)
    wing_dalen = im_dalen.snapshot(T0).per_contract["m"].q_max
    assert wing_dalen > mid_dalen  # Dalen verbatim: grows at wings
    # bounded by 1/s_prime_floor
    cfg = MMConfig(q_max_mode="dalen")
    assert wing_dalen <= cfg.q_max_scale / cfg.s_prime_floor + 1e-9
