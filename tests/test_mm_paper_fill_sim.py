"""Golden-scenario tests for market_maker.paper_fill_sim (plan X1, Section 6.3).

Every scenario is hand-computed and asserted exactly. The fill model must be
conservative with zero optimistic loopholes: mid-touch never fills, a print
smaller than queue-ahead never fills, cancel-latency windows can be hit.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import MarketState, Side
from market_maker.paper_fill_sim import (
    ASSUMPTION_QUEUEBEHIND,
    ASSUMPTION_TRADETHROUGH,
    PaperFillSimulator,
)

T0 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def _sec(n: float) -> timedelta:
    return timedelta(seconds=n)


def _ms(ts, *, bid_depth=None, ask_depth=None, prints=None,
        best_bid=0.49, best_ask=0.51, feed_healthy=True,
        market_id="m1"):
    return MarketState(
        ts=ts, market_id=market_id, expiry_key="2026-07-20", strike=100000.0,
        best_bid=best_bid, best_ask=best_ask,
        bid_depth=bid_depth or [], ask_depth=ask_depth or [],
        last_prints=prints or [], feed_healthy=feed_healthy,
    )


def _cfg():
    # Deterministic default latencies (2000/2000 ms), 5s gap threshold.
    return MMConfig()


# ---------------------------------------------------------------------------
# 1. Join existing level: queue 100 ahead, prints consume queue then fill.
# ---------------------------------------------------------------------------

def test_join_existing_level_queue_then_fill():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 20.0, T0)  # effective T0+2s

    # Activate at T0+3s against a 100-share displayed level.
    f0 = sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[(0.50, 100.0)]))
    assert f0 == []
    assert sim.open_orders()[0]["queue_ahead"] == pytest.approx(100.0)

    # First print of 60 at our price: consumes 60 of queue, no fill.
    f1 = sim.on_market_state(_ms(T0 + _sec(4), bid_depth=[(0.50, 100.0)],
                                 prints=[(T0 + _sec(4), 0.50, 60.0)]))
    assert f1 == []
    assert sim.open_orders()[0]["queue_ahead"] == pytest.approx(40.0)

    # Second print of 60: consumes remaining 40 of queue, fills 20.
    f2 = sim.on_market_state(_ms(T0 + _sec(5), bid_depth=[(0.50, 100.0)],
                                 prints=[(T0 + _sec(5), 0.50, 60.0)]))
    assert len(f2) == 1
    assert f2[0].size == pytest.approx(20.0)
    assert f2[0].price == pytest.approx(0.50)
    assert f2[0].side == Side.BUY_YES


# ---------------------------------------------------------------------------
# 2. Mid crossing our price with NO print never fills.
# ---------------------------------------------------------------------------

def test_mid_touch_never_fills():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.60, 10.0, T0)
    # New best level (empty depth at our price) -> queue 0.
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[], best_bid=0.55, best_ask=0.65))
    # Mid moves above our bid but NO prints -> still no fill.
    f = sim.on_market_state(_ms(T0 + _sec(4), bid_depth=[], best_bid=0.62,
                                best_ask=0.66, prints=[]))
    assert f == []
    assert sim.fills() == []


# ---------------------------------------------------------------------------
# 3. Print strictly through our price: queue consumed then fill.
# ---------------------------------------------------------------------------

def test_print_through_consumes_queue_then_fills():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 30.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[(0.50, 100.0)]))
    # A 150-share print at 0.48 (through): consume 100 queue, fill 30.
    f = sim.on_market_state(_ms(T0 + _sec(4), bid_depth=[(0.50, 100.0)],
                                prints=[(T0 + _sec(4), 0.48, 150.0)]))
    assert len(f) == 1
    assert f[0].size == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 4. Cancel latency: live during the window, dead after it.
# ---------------------------------------------------------------------------

def test_cancel_latency_fill_inside_window():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 10.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[]))  # queue 0
    sim.cancel("o1", T0 + _sec(5))  # cancel effective T0+7s
    # Print at T0+6s is inside the cancel window -> we own the stale quote.
    f = sim.on_market_state(_ms(T0 + _sec(6), bid_depth=[],
                                prints=[(T0 + _sec(6), 0.50, 5.0)]))
    assert len(f) == 1
    assert f[0].size == pytest.approx(5.0)


def test_cancel_latency_no_fill_after_window():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 10.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[]))
    sim.cancel("o1", T0 + _sec(5))  # effective T0+7s
    # Print exactly at/after the cancel effective time -> no fill.
    f = sim.on_market_state(_ms(T0 + _sec(7), bid_depth=[],
                                prints=[(T0 + _sec(7), 0.50, 5.0)]))
    assert f == []
    assert sim.fills() == []


# ---------------------------------------------------------------------------
# 5. Placement latency: a print before the order is effective never fills.
# ---------------------------------------------------------------------------

def test_placement_latency_pre_effective_no_fill():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 10.0, T0)  # effective T0+2s
    f = sim.on_market_state(_ms(T0 + _sec(1), bid_depth=[],
                                prints=[(T0 + _sec(1), 0.50, 50.0)]))
    assert f == []
    # After it is effective, the same kind of print fills.
    f2 = sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[],
                                 prints=[(T0 + _sec(3), 0.50, 20.0)]))
    assert len(f2) == 1
    assert f2[0].size == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 6. Unattributable level reduction (print present) -> queue not over-reduced.
# ---------------------------------------------------------------------------

def test_unattributable_reduction_only_print_logic():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 5.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[(0.50, 100.0)]))
    # Level drops 100 -> 40 AND a 30-share print is present: ambiguous, so
    # queue reduces ONLY by the print's 30 (to 70), not by the 60 drop.
    f = sim.on_market_state(_ms(T0 + _sec(4), bid_depth=[(0.50, 40.0)],
                                prints=[(T0 + _sec(4), 0.50, 30.0)]))
    assert f == []
    assert sim.open_orders()[0]["queue_ahead"] == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# 7. Attributable cancel-ahead reduction (no print) -> queue reduced.
# ---------------------------------------------------------------------------

def test_attributable_cancel_ahead_reduces_queue():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 5.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[(0.50, 100.0)]))
    # Level drops 100 -> 40 with NO print -> attribute the 60 drop to cancels.
    f = sim.on_market_state(_ms(T0 + _sec(4), bid_depth=[(0.50, 40.0)], prints=[]))
    assert f == []
    assert sim.open_orders()[0]["queue_ahead"] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# 8. Trade-through fallback: print AT our price never fills, through fills.
# ---------------------------------------------------------------------------

def test_trade_through_mode():
    sim = PaperFillSimulator(_cfg(), trade_through_only=True)
    assert sim.assumption_set == ASSUMPTION_TRADETHROUGH
    sim.place("o1", "m1", "bid", 0.50, 10.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[(0.50, 100.0)]))
    # Print AT our price -> no fill (queue infinite at our exact price).
    f_at = sim.on_market_state(_ms(T0 + _sec(4), bid_depth=[(0.50, 100.0)],
                                   prints=[(T0 + _sec(4), 0.50, 20.0)]))
    assert f_at == []
    # Print strictly through -> fill up to the observed print size.
    f_thru = sim.on_market_state(_ms(T0 + _sec(5), bid_depth=[(0.50, 100.0)],
                                     prints=[(T0 + _sec(5), 0.49, 8.0)]))
    assert len(f_thru) == 1
    assert f_thru[0].size == pytest.approx(8.0)
    assert f_thru[0].assumption_set == ASSUMPTION_TRADETHROUGH


# ---------------------------------------------------------------------------
# 9. Feed gap: quotes exposed, no fills inside the gap, incident recorded.
# ---------------------------------------------------------------------------

def test_feed_gap_exposure_no_fills():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 10.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[]))  # activate, last_ts=T0+3s

    # 7s gap (> 5s threshold): a would-be filling print is IGNORED inside gap.
    f_gap = sim.on_market_state(_ms(T0 + _sec(10), bid_depth=[],
                                    prints=[(T0 + _sec(10), 0.49, 20.0)]))
    assert f_gap == []
    assert sim.fills() == []

    # Recovery closes the incident.
    sim.on_market_state(_ms(T0 + _sec(11), bid_depth=[]))
    incs = sim.exposure_incidents()
    assert len(incs) == 1
    assert incs[0].start == T0 + _sec(3)
    assert incs[0].end == T0 + _sec(11)
    assert incs[0].duration_s == pytest.approx(8.0)
    assert incs[0].n_live_orders == 1


def test_feed_unhealthy_flag_marks_gap():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 10.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[]))
    f = sim.on_market_state(_ms(T0 + _sec(4), bid_depth=[], feed_healthy=False,
                                prints=[(T0 + _sec(4), 0.49, 50.0)]))
    assert f == []
    assert len(sim.exposure_incidents()) == 1


# ---------------------------------------------------------------------------
# 10. Adverse-selection marks + assumption_set on every fill.
# ---------------------------------------------------------------------------

def test_adverse_marks_backfill_on_horizon():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 5.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[]))
    fills = sim.on_market_state(_ms(T0 + _sec(4), bid_depth=[], best_bid=0.49,
                                    best_ask=0.51,
                                    prints=[(T0 + _sec(4), 0.50, 5.0)]))
    assert len(fills) == 1
    fill_ts = fills[0].ts
    assert fills[0].mid_at_fill == pytest.approx(0.50)
    assert fills[0].assumption_set == ASSUMPTION_QUEUEBEHIND

    # Before 1 minute: p1m still None.
    snap = sim.mark_fills(fill_ts + _sec(30), 0.52)
    assert snap[0].mid_p1m is None

    # After 1 minute: p1m backfilled; p10m/p1h still None.
    snap = sim.mark_fills(fill_ts + _sec(61), 0.60)
    assert snap[0].mid_p1m == pytest.approx(0.60)
    assert snap[0].mid_p10m is None
    assert snap[0].mid_p1h is None


# ---------------------------------------------------------------------------
# 11. Partial fills accumulate; overfill is impossible.
# ---------------------------------------------------------------------------

def test_partial_fills_accumulate_no_overfill():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 10.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), bid_depth=[]))  # queue 0

    f1 = sim.on_market_state(_ms(T0 + _sec(4), bid_depth=[],
                                 prints=[(T0 + _sec(4), 0.50, 4.0)]))
    f2 = sim.on_market_state(_ms(T0 + _sec(5), bid_depth=[],
                                 prints=[(T0 + _sec(5), 0.50, 4.0)]))
    # Oversized print: only the remaining 2 shares can fill.
    f3 = sim.on_market_state(_ms(T0 + _sec(6), bid_depth=[],
                                 prints=[(T0 + _sec(6), 0.50, 5.0)]))
    total = sum(f[0].size for f in (f1, f2, f3))
    assert total == pytest.approx(10.0)
    # Order fully filled and removed.
    assert sim.open_orders() == []


# ---------------------------------------------------------------------------
# 12. Determinism: same stream twice -> identical fills.
# ---------------------------------------------------------------------------

def _run_stream():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "bid", 0.50, 20.0, T0)
    sim.place("o2", "m1", "ask", 0.55, 15.0, T0)
    stream = [
        _ms(T0 + _sec(3), bid_depth=[(0.50, 100.0)], ask_depth=[(0.55, 50.0)]),
        _ms(T0 + _sec(4), bid_depth=[(0.50, 100.0)], ask_depth=[(0.55, 50.0)],
            prints=[(T0 + _sec(4), 0.50, 130.0)]),
        _ms(T0 + _sec(5), bid_depth=[(0.50, 100.0)], ask_depth=[(0.55, 50.0)],
            prints=[(T0 + _sec(5), 0.56, 80.0)]),
    ]
    out = []
    for ms in stream:
        for f in sim.on_market_state(ms):
            out.append((f.order_id, f.price, f.size, f.side))
    return out


def test_determinism_same_stream():
    a = _run_stream()
    b = _run_stream()
    assert a == b
    assert len(a) == 2  # one bid fill (30) + one ask fill (30)


def test_ask_side_fill_semantics():
    sim = PaperFillSimulator(_cfg())
    sim.place("o1", "m1", "ask", 0.55, 15.0, T0)
    sim.on_market_state(_ms(T0 + _sec(3), ask_depth=[(0.55, 50.0)]))
    # Print at 0.56 (>= our ask) with size 80: consume 50 queue, fill 15.
    f = sim.on_market_state(_ms(T0 + _sec(4), ask_depth=[(0.55, 50.0)],
                                prints=[(T0 + _sec(4), 0.56, 80.0)]))
    assert len(f) == 1
    assert f[0].size == pytest.approx(15.0)
    assert f[0].side == Side.BUY_NO
