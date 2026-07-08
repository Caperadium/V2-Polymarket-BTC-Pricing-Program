"""Tests for market_maker.liquidity_monitor (plan 2.9, task M1, contract 4.9)."""
from __future__ import annotations

import inspect
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import LiquidityRegime, LiquidityState, MarketState
import market_maker.liquidity_monitor as lm
from market_maker.liquidity_monitor import LiquidityMonitor

T0 = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)


def _state(ts, best_bid, best_ask, bid_depth, ask_depth, prints=None, healthy=True, market_id="m1"):
    return MarketState(
        ts=ts, market_id=market_id, expiry_key="2026-07-20", strike=100000.0,
        best_bid=best_bid, best_ask=best_ask, bid_depth=bid_depth, ask_depth=ask_depth,
        last_prints=prints or [], feed_healthy=healthy,
    )


# ---------------------------------------------------------------------------
# Realized depth: hand-computed
# ---------------------------------------------------------------------------

def test_realized_depth_hand_computed():
    mon = LiquidityMonitor(depth_ticks=3, tick_size=0.01)
    bid_depth = [(0.50, 10.0), (0.49, 5.0), (0.47, 20.0), (0.44, 8.0)]
    ask_depth = [(0.52, 6.0), (0.53, 3.0), (0.55, 9.0), (0.58, 2.0)]
    mon.update(_state(T0, 0.50, 0.52, bid_depth, ask_depth))
    depth_bid, depth_ask = mon.realized_depth()
    # band = 3*0.01 = 0.03; bid prices >= 0.47 -> 10+5+20=35; ask prices <= 0.55 -> 6+3+9=18
    assert depth_bid == pytest.approx(35.0)
    assert depth_ask == pytest.approx(18.0)


def test_realized_depth_rolling_average():
    mon = LiquidityMonitor(depth_ticks=3, tick_size=0.01, depth_window=2)
    mon.update(_state(T0, 0.50, 0.52, [(0.50, 10.0)], [(0.52, 10.0)]))
    mon.update(_state(T0, 0.50, 0.52, [(0.50, 20.0)], [(0.52, 20.0)]))
    depth_bid, depth_ask = mon.realized_depth()
    assert depth_bid == pytest.approx(15.0)
    assert depth_ask == pytest.approx(15.0)


def test_realized_depth_empty_side_is_zero():
    mon = LiquidityMonitor()
    mon.update(_state(T0, None, 0.52, [], [(0.52, 5.0)]))
    depth_bid, depth_ask = mon.realized_depth()
    assert depth_bid == pytest.approx(0.0)
    assert depth_ask == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Kyle's lambda: synthetic |dMid| = c*size -> recovers c; NaN before min_obs
# ---------------------------------------------------------------------------

def test_kyle_lambda_recovers_known_impact_coefficient():
    c = 0.0002
    mon = LiquidityMonitor(min_obs=30)
    mid = 0.50
    ts = T0
    # first update establishes prev_mid, no trade yet
    mon.update(_state(ts, mid - 0.01, mid + 0.01, [(mid - 0.01, 100.0)], [(mid + 0.01, 100.0)]))
    assert math.isnan(mon.kyle_lambda())

    size = 10.0
    for i in range(1, 40):
        dmid = c * size
        mid = mid + dmid  # monotone drift so best_bid/ask move consistently
        ts = ts + timedelta(seconds=1)
        bb, ba = mid - 0.01, mid + 0.01
        if i < 30:
            assert math.isnan(mon.kyle_lambda())
        mon.update(_state(ts, bb, ba, [(bb, 100.0)], [(ba, 100.0)],
                          prints=[(ts, ba, size)]))
    assert mon.kyle_lambda() == pytest.approx(c, rel=1e-6)


def test_kyle_lambda_nan_with_no_trades():
    mon = LiquidityMonitor(min_obs=5)
    ts = T0
    for i in range(10):
        ts = ts + timedelta(seconds=1)
        mon.update(_state(ts, 0.50, 0.52, [(0.50, 10.0)], [(0.52, 10.0)]))
    assert math.isnan(mon.kyle_lambda())


# ---------------------------------------------------------------------------
# Arb half-life: synthetic exponential decay -> recovered within tolerance
# ---------------------------------------------------------------------------

def test_arb_halflife_recovers_known_half_life():
    half_life_s = 3600.0  # 1 hour
    tau = half_life_s / math.log(2.0)
    dt = 60.0  # 1 minute sampling
    mon = LiquidityMonitor(min_obs=30)
    ts = T0
    d0 = 0.05
    for i in range(60):
        dev = d0 * math.exp(-(i * dt) / tau)
        mon.update_pair(ts, dev)
        ts = ts + timedelta(seconds=dt)
    recovered = mon.arb_halflife_s()
    assert recovered == pytest.approx(half_life_s, rel=1e-3)


def test_arb_halflife_immune_to_fee_wedge_baseline():
    # Polymarket fee wedge: YES+NO settles at a persistent nonzero deviation
    # (e.g. -0.02), NOT at exactly 1. Shock decay toward that baseline must be
    # measured, not decay toward zero (which would overstate the half-life).
    half_life_s = 3600.0
    tau = half_life_s / math.log(2.0)
    dt = 60.0
    wedge = -0.02
    mon = LiquidityMonitor(min_obs=30)
    ts = T0
    d0 = 0.05
    for i in range(60):
        dev = wedge + d0 * math.exp(-(i * dt) / tau)
        mon.update_pair(ts, dev)
        ts = ts + timedelta(seconds=dt)
    recovered = mon.arb_halflife_s()
    assert recovered == pytest.approx(half_life_s, rel=1e-3)


def test_arb_halflife_nan_without_paired_series():
    mon = LiquidityMonitor()
    mon.update(_state(T0, 0.50, 0.52, [(0.50, 10.0)], [(0.52, 10.0)]))
    assert math.isnan(mon.arb_halflife_s())


def test_arb_halflife_nan_before_min_obs():
    mon = LiquidityMonitor(min_obs=30)
    ts = T0
    for i in range(10):
        mon.update_pair(ts, 0.05 * math.exp(-i / 10.0))
        ts = ts + timedelta(seconds=60)
    assert math.isnan(mon.arb_halflife_s())


# ---------------------------------------------------------------------------
# Volume discount
# ---------------------------------------------------------------------------

def test_volume_discount_applied():
    mon = LiquidityMonitor(config=MMConfig(volume_discount=2.5))
    assert mon.discount_volume(1000.0) == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# Regime: thick / thin / empty-book / unhealthy-feed
# ---------------------------------------------------------------------------

def test_regime_thick():
    mon = LiquidityMonitor()
    reg = mon.regime(300.0, 300.0, True, 0.50, 0.52)
    assert reg == LiquidityRegime.THICK


def test_regime_normal():
    mon = LiquidityMonitor()
    reg = mon.regime(60.0, 60.0, True, 0.50, 0.52)
    assert reg == LiquidityRegime.NORMAL


def test_regime_thin():
    mon = LiquidityMonitor()
    reg = mon.regime(10.0, 10.0, True, 0.50, 0.52)
    assert reg == LiquidityRegime.THIN


def test_regime_degenerate_on_empty_book():
    mon = LiquidityMonitor()
    reg = mon.regime(0.0, 0.0, True, None, None)
    assert reg == LiquidityRegime.DEGENERATE


def test_regime_degenerate_on_one_sided_book():
    mon = LiquidityMonitor()
    reg = mon.regime(50.0, 0.0, True, 0.50, None)
    assert reg == LiquidityRegime.DEGENERATE


def test_regime_degenerate_on_unhealthy_feed():
    mon = LiquidityMonitor()
    reg = mon.regime(300.0, 300.0, False, 0.50, 0.52)
    assert reg == LiquidityRegime.DEGENERATE


def test_regime_degenerate_below_floor():
    mon = LiquidityMonitor()
    reg = mon.regime(1.0, 1.0, True, 0.50, 0.52)
    assert reg == LiquidityRegime.DEGENERATE


# ---------------------------------------------------------------------------
# emit() -> LiquidityState
# ---------------------------------------------------------------------------

def test_emit_produces_liquidity_state():
    mon = LiquidityMonitor()
    mon.update(_state(T0, 0.50, 0.52, [(0.50, 300.0)], [(0.52, 300.0)]))
    out = mon.emit(window="w1")
    assert isinstance(out, LiquidityState)
    assert out.market_id == "m1"
    assert out.window == "w1"
    assert out.regime == LiquidityRegime.THICK
    assert math.isnan(out.kyle_lambda)
    assert math.isnan(out.arb_halflife_s)
    assert out.vol_discount == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# No order-flow direction signal API exists anywhere in the module (Finding 10)
# ---------------------------------------------------------------------------

def test_no_direction_or_flow_sign_api():
    names = []
    for mod_name in dir(lm):
        names.append(mod_name)
        obj = getattr(lm, mod_name)
        if inspect.isclass(obj):
            names.extend(dir(obj))
    banned = [n for n in names if "direction" in n.lower() or "flow_sign" in n.lower()]
    assert banned == []
