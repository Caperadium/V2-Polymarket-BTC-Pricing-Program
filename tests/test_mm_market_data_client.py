"""Tests for market_maker.market_data_client (plan 2.14, task D1, contract 4.2)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import MarketState
from market_maker.market_data_client import BookMirror, FeedCapability, PolymarketFeedAdapter

T0 = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)

# Large threshold so tests not exercising staleness are unaffected by the real
# wall-clock default (T0 is a fixed fixture date, not "now").
_NO_STALE_CONFIG = MMConfig(feed_gap_threshold_s=1.0e9)


def _mirror(**kwargs):
    kwargs.setdefault("config", _NO_STALE_CONFIG)
    return BookMirror(**kwargs)


def _snap(ts, seq, bids, asks):
    return {"type": "snapshot", "bids": bids, "asks": asks, "ts": ts, "seq": seq}


def _delta(ts, seq, side, price, size):
    return {"type": "delta", "side": side, "price": price, "size": size, "ts": ts, "seq": seq}


def _trade(ts, seq, price, size):
    return {"type": "trade", "price": price, "size": size, "ts": ts, "seq": seq}


# ---------------------------------------------------------------------------
# Scripted snapshot + deltas -> hand-checked book mirror
# ---------------------------------------------------------------------------

def test_scripted_snapshot_and_deltas_hand_checked():
    mirror = _mirror()
    mirror.on_message(_snap(T0, 1, [(0.50, 10.0), (0.49, 5.0)], [(0.52, 8.0), (0.53, 4.0)]))
    assert mirror.best_bid() == pytest.approx(0.50)
    assert mirror.best_ask() == pytest.approx(0.52)
    assert mirror.bid_depth() == [(0.50, 10.0), (0.49, 5.0)]
    assert mirror.ask_depth() == [(0.52, 8.0), (0.53, 4.0)]

    mirror.on_message(_delta(T0 + timedelta(seconds=1), 2, "bid", 0.51, 3.0))
    assert mirror.best_bid() == pytest.approx(0.51)
    assert (0.51, 3.0) in mirror.bid_depth()

    # size 0 removes the level
    mirror.on_message(_delta(T0 + timedelta(seconds=2), 3, "bid", 0.51, 0.0))
    assert mirror.best_bid() == pytest.approx(0.50)
    assert all(p != 0.51 for p, _ in mirror.bid_depth())


def test_ask_side_delta_and_removal():
    mirror = _mirror()
    mirror.on_message(_snap(T0, 1, [(0.50, 10.0)], [(0.52, 8.0), (0.53, 4.0)]))
    mirror.on_message(_delta(T0 + timedelta(seconds=1), 2, "ask", 0.515, 6.0))
    assert mirror.best_ask() == pytest.approx(0.515)
    mirror.on_message(_delta(T0 + timedelta(seconds=2), 3, "ask", 0.515, 0.0))
    assert mirror.best_ask() == pytest.approx(0.52)


# ---------------------------------------------------------------------------
# Trade messages accumulate in last_prints and drain on emit
# ---------------------------------------------------------------------------

def test_trade_messages_accumulate_and_drain_on_emit():
    mirror = _mirror()
    mirror.on_message(_snap(T0, 1, [(0.5, 10.0)], [(0.52, 8.0)]))
    mirror.on_message(_trade(T0 + timedelta(seconds=1), 2, 0.51, 2.0))
    mirror.on_message(_trade(T0 + timedelta(seconds=2), 3, 0.52, 1.0))
    state = mirror.emit("m1", "2026-07-20", 100000.0)
    assert state.last_prints == [
        (T0 + timedelta(seconds=1), 0.51, 2.0),
        (T0 + timedelta(seconds=2), 0.52, 1.0),
    ]
    # drained: a second emit with no new messages sees no prints
    state2 = mirror.emit("m1", "2026-07-20", 100000.0)
    assert state2.last_prints == []


# ---------------------------------------------------------------------------
# Sequence gap -> unhealthy until next snapshot heals
# ---------------------------------------------------------------------------

def test_seq_gap_marks_unhealthy_until_next_snapshot():
    mirror = _mirror()
    mirror.on_message(_snap(T0, 1, [(0.5, 10.0)], [(0.52, 8.0)]))
    assert mirror.feed_healthy() is True
    assert mirror.gap_events == []

    mirror.on_message(_delta(T0 + timedelta(seconds=1), 5, "bid", 0.49, 3.0))  # 1 -> 5: gap
    assert mirror.feed_healthy() is False
    assert len(mirror.gap_events) == 1

    mirror.on_message(_delta(T0 + timedelta(seconds=2), 6, "bid", 0.48, 3.0))  # contiguous, still unhealthy
    assert mirror.feed_healthy() is False
    assert len(mirror.gap_events) == 1

    mirror.on_message(_snap(T0 + timedelta(seconds=3), 7, [(0.5, 10.0)], [(0.52, 8.0)]))
    assert mirror.feed_healthy() is True


# ---------------------------------------------------------------------------
# Staleness via fake clock -> feed_healthy False
# ---------------------------------------------------------------------------

def test_staleness_via_fake_clock():
    now = {"t": T0}

    def fake_clock():
        return now["t"]

    config = MMConfig(feed_gap_threshold_s=5.0)
    mirror = BookMirror(config=config, clock=fake_clock)
    mirror.on_message(_snap(T0, 1, [(0.5, 10.0)], [(0.52, 8.0)]))
    assert mirror.is_stale() is False
    assert mirror.feed_healthy() is True

    now["t"] = T0 + timedelta(seconds=10)
    assert mirror.is_stale() is True
    assert mirror.feed_healthy() is False


# ---------------------------------------------------------------------------
# TOP_OF_BOOK mode emits touch-only depth
# ---------------------------------------------------------------------------

def test_top_of_book_mode_emits_touch_only_depth():
    mirror = _mirror(capability=FeedCapability.TOP_OF_BOOK, depth_n=10)
    mirror.on_message(_snap(
        T0, 1,
        [(0.50, 10.0), (0.49, 5.0), (0.48, 2.0)],
        [(0.52, 8.0), (0.53, 4.0)],
    ))
    assert mirror.bid_depth() == [(0.50, 10.0)]
    assert mirror.ask_depth() == [(0.52, 8.0)]

    state = mirror.emit("m1", "2026-07-20", 100000.0)
    assert state.bid_depth == [(0.50, 10.0)]
    assert state.ask_depth == [(0.52, 8.0)]


def test_full_l2_mode_emits_up_to_depth_n():
    mirror = _mirror(capability=FeedCapability.FULL_L2, depth_n=2)
    mirror.on_message(_snap(
        T0, 1,
        [(0.50, 10.0), (0.49, 5.0), (0.48, 2.0)],
        [(0.52, 8.0), (0.53, 4.0), (0.54, 1.0)],
    ))
    state = mirror.emit("m1", "2026-07-20", 100000.0)
    assert state.bid_depth == [(0.50, 10.0), (0.49, 5.0)]
    assert state.ask_depth == [(0.52, 8.0), (0.53, 4.0)]


# ---------------------------------------------------------------------------
# MarketState contract fields all populated
# ---------------------------------------------------------------------------

def test_market_state_contract_fields_populated():
    mirror = _mirror()
    mirror.on_message(_snap(T0, 1, [(0.5, 10.0)], [(0.52, 8.0)]))
    mirror.on_message(_trade(T0 + timedelta(seconds=1), 2, 0.51, 2.0))
    state = mirror.emit("m1", "2026-07-20", 100000.0)

    assert isinstance(state, MarketState)
    assert state.market_id == "m1"
    assert state.expiry_key == "2026-07-20"
    assert state.strike == pytest.approx(100000.0)
    assert state.best_bid == pytest.approx(0.5)
    assert state.best_ask == pytest.approx(0.52)
    assert state.bid_depth == [(0.5, 10.0)]
    assert state.ask_depth == [(0.52, 8.0)]
    assert state.last_prints == [(T0 + timedelta(seconds=1), 0.51, 2.0)]
    assert state.feed_healthy is True
    assert state.ts == T0 + timedelta(seconds=1)


def test_empty_book_emits_none_best_bid_ask():
    mirror = _mirror()
    mirror.on_message(_snap(T0, 1, [], []))
    state = mirror.emit("m1", "2026-07-20", 100000.0)
    assert state.best_bid is None
    assert state.best_ask is None
    assert state.bid_depth == []
    assert state.ask_depth == []


# ---------------------------------------------------------------------------
# PolymarketFeedAdapter: venue payload translation + drain + health
# (no network -- _handle_raw is driven directly with observed P0b payloads)
# ---------------------------------------------------------------------------

import json

TOK_A = "111"
TOK_B = "222"
TOK_OTHER = "999"  # complement token, present in price_change but unsubscribed


def _adapter():
    return PolymarketFeedAdapter({"mkt-a": TOK_A, "mkt-b": TOK_B})


def test_adapter_rejects_empty_and_duplicate_token_maps():
    with pytest.raises(ValueError):
        PolymarketFeedAdapter({})
    with pytest.raises(ValueError):
        PolymarketFeedAdapter({"mkt-a": TOK_A, "mkt-b": TOK_A})


def test_adapter_book_event_becomes_snapshot():
    a = _adapter()
    raw = json.dumps([{
        "event_type": "book", "asset_id": TOK_A, "timestamp": "1783436736324",
        "bids": [{"price": "0.50", "size": "10"}],
        "asks": [{"price": "0.52", "size": "8"}, {"price": "0.53", "size": "4"}],
    }])
    a._handle_raw(raw, now=T0)
    out = a.drain()
    assert out["mkt-b"] == []
    (msg,) = out["mkt-a"]
    assert msg["type"] == "snapshot"
    assert msg["bids"] == [(0.50, 10.0)]
    assert msg["asks"] == [(0.52, 8.0), (0.53, 4.0)]
    assert msg["ts"] == datetime.fromtimestamp(1783436736.324, tz=timezone.utc)
    assert "seq" not in msg  # venue has no seq; harness assigns its own


def test_adapter_price_change_filters_to_subscribed_tokens():
    a = _adapter()
    raw = json.dumps({
        "event_type": "price_change", "timestamp": "1783436745830",
        "price_changes": [
            {"asset_id": TOK_A, "price": "0.30", "size": "761.5", "side": "SELL"},
            {"asset_id": TOK_OTHER, "price": "0.70", "size": "761.5", "side": "BUY"},
            {"asset_id": TOK_B, "price": "0.45", "size": "0", "side": "BUY"},
        ],
    })
    a._handle_raw(raw, now=T0)
    out = a.drain()
    (ma,) = out["mkt-a"]
    assert ma == {"type": "delta", "side": "ask", "price": 0.30, "size": 761.5, "ts": ma["ts"]}
    (mb,) = out["mkt-b"]
    assert mb["side"] == "bid" and mb["size"] == 0.0  # BUY=bid; size 0 removes


def test_adapter_last_trade_price_becomes_trade_and_feeds_book_mirror():
    a = _adapter()
    a._handle_raw(json.dumps({
        "event_type": "last_trade_price", "asset_id": TOK_A,
        "price": "0.52", "size": "25.9", "side": "BUY", "timestamp": "1783436789081",
    }), now=T0)
    (msg,) = a.drain()["mkt-a"]
    assert msg["type"] == "trade" and msg["price"] == 0.52 and msg["size"] == 25.9
    # round-trip: translated message is BookMirror-consumable
    mirror = _mirror()
    mirror.on_message(dict(msg, seq=1))
    state = mirror.emit("mkt-a", "2026-07-20", 100000.0)
    assert state.last_prints[0][1:] == (0.52, 25.9)


def test_adapter_ignores_unknown_types_unsubscribed_assets_and_garbage():
    a = _adapter()
    a._handle_raw(json.dumps({"event_type": "tick_size_change", "asset_id": TOK_A}), now=T0)
    a._handle_raw(json.dumps({"event_type": "book", "asset_id": TOK_OTHER, "bids": [], "asks": []}), now=T0)
    a._handle_raw("not json {", now=T0)
    out = a.drain()
    assert out == {"mkt-a": [], "mkt-b": []}


def test_adapter_drain_clears_and_health_defaults_false():
    a = _adapter()
    a._handle_raw(json.dumps({
        "event_type": "book", "asset_id": TOK_A, "timestamp": "1783436736324",
        "bids": [], "asks": [],
    }), now=T0)
    assert len(a.drain()["mkt-a"]) == 1
    assert a.drain() == {"mkt-a": [], "mkt-b": []}  # second drain empty
    assert a.healthy() is False  # never started -> no connection liveness
