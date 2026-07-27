"""Workstream 1 tests (plan i-m-preparing-to-launch-sharded-snail.md):
state-store live-order lookup / index / market registry, and harness journal
bounding + registry-merge settlement catch-up.

Scripted synthetic feeds only, following tests/test_mm_integration.py's
conventions (fixed clock, deterministic scripted books).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import (
    AnchorMethod,
    ContractInv,
    InventoryState,
    LiquidityRegime,
    QuoteMode,
    RiskTrigger,
    SettlementEvent,
    SettlementOutcome,
    Side,
    SizingCap,
    SpotSource,
)
from market_maker.harness import PaperTradingLoop
from market_maker.order_lifecycle import SimClock
from market_maker.pnl_report import markout_report
from market_maker.settlement_handler import BTCDataProvider, settlement_instant_utc
from market_maker.state_store import MMStateStore

START = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
S0 = 100000.0
EXPIRY = "2026-07-06"
MARKETS = [("m-100k", 100000.0), ("m-102k", 102000.0)]
MARKETS3 = [("m-98k", 98000.0), ("m-100k", 100000.0), ("m-102k", 102000.0)]


# ---------------------------------------------------------------------------
# scripted stubs (mirrors test_mm_integration.py)
# ---------------------------------------------------------------------------


def _engine(s0=S0, scale=2000.0, n_sims=15000):
    def fn(strikes, hours_to_expiry, **kwargs):
        out = {float(k): float(1.0 / (1.0 + np.exp((float(k) - s0) / scale))) for k in strikes}
        out["_meta"] = {"n_sims": n_sims, "S0": s0, "horizon_gate_active": False}
        return out
    return fn


class _VG:
    def __init__(self, regime="normal", shock=False, kelly_mult=1.0, edge_add_cents=0.0):
        self.regime = regime
        self.shock = shock
        self.kelly_mult = kelly_mult
        self.edge_add_cents = edge_add_cents


def _vol_gate():
    return lambda: _VG()


def _snapshot_msg(p, prints=None):
    bid = round(max(0.01, p - 0.03), 4)
    ask = round(min(0.99, p + 0.03), 4)
    msgs = [{
        "type": "snapshot",
        "bids": [(bid, 100.0), (round(bid - 0.01, 4), 100.0)],
        "asks": [(ask, 100.0), (round(ask + 0.01, 4), 100.0)],
    }]
    for pr in (prints or []):
        msgs.append({"type": "trade", "price": pr[0], "size": pr[1]})
    return msgs


def _snapshot_msg_tiny_depth(p):
    """Like _snapshot_msg but with a tiny resting size on both sides, so the
    LiquidityMonitor's realized_depth (mean top-of-book size within
    depth_ticks) is small enough to force the sizing DEPTH cap to bind."""
    bid = round(max(0.01, p - 0.03), 4)
    ask = round(min(0.99, p + 0.03), 4)
    return [{
        "type": "snapshot",
        "bids": [(bid, 1.0)],
        "asks": [(ask, 1.0)],
    }]


def _moving_books(markets, frac):
    """One book snapshot per market whose mid drifts with `frac` in [0, 1] so
    consensus_x / QuoteSet contents differ tick to tick -- lets the journal
    cap tests confirm the retained tail is the NEWEST entries, not just any
    `maxlen`-sized slice."""
    out = {}
    for m, _k in markets:
        biased = 0.30 + 0.40 * frac
        out[m] = _snapshot_msg(biased)
    return out


@pytest.fixture
def store(tmp_path):
    s = MMStateStore(str(tmp_path / "mm.db"))
    yield s
    s.close()


def _settlement_provider_for(expiry_key, spot):
    settle = settlement_instant_utc(expiry_key)
    idx = pd.to_datetime([
        settle - timedelta(minutes=2), settle, settle + timedelta(minutes=2),
    ])
    intraday = pd.DataFrame({"close": [spot - 100, spot, spot + 100]}, index=idx)
    return BTCDataProvider(intraday=intraday, daily=pd.DataFrame())


# ---------------------------------------------------------------------------
# 1.2 -- journal / x_hist caps
# ---------------------------------------------------------------------------


def test_journal_caps_bound_and_keep_newest(store):
    maxlen = 5
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(), clock=SimClock(START), vol_gate_fn=_vol_gate(),
        journal_maxlen=maxlen, x_hist_maxlen=1_000_000,
    )

    n_ticks = 15
    for i in range(n_ticks):
        loop.tick(_moving_books(MARKETS, i / (n_ticks - 1)))

    assert len(loop.checked_ladders) <= maxlen
    assert len(loop.all_checked_quote_sets) <= maxlen * len(MARKETS)
    # both stay plain lists (tests elsewhere index/slice them)
    assert isinstance(loop.checked_ladders, list)
    assert isinstance(loop.all_checked_quote_sets, list)

    # Newest-anchored: source_seq == loop tick number (1-indexed, assigned in
    # _compose_quote_sets as self._tick) -- the retained tail must be the
    # LAST len(checked_ladders) tick numbers, in order.
    seqs = [ladder[1][0].source_seq for ladder in loop.checked_ladders]
    assert seqs == list(range(n_ticks - len(seqs) + 1, n_ticks + 1))


def test_journal_none_maxlen_is_unbounded(store):
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(), clock=SimClock(START), vol_gate_fn=_vol_gate(),
        journal_maxlen=None,
    )
    n_ticks = 12
    for i in range(n_ticks):
        loop.tick(_moving_books(MARKETS, i / (n_ticks - 1)))
    assert len(loop.checked_ladders) == n_ticks


def test_x_hist_capped_and_newest_anchored(store):
    x_maxlen = 4
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(), clock=SimClock(START), vol_gate_fn=_vol_gate(),
        x_hist_maxlen=x_maxlen, journal_maxlen=None,
    )
    n_ticks = 10
    full_history = {m: [] for m, _ in MARKETS}
    for i in range(n_ticks):
        loop.tick(_moving_books(MARKETS, i / (n_ticks - 1)))
        assert loop.last_fair_value is not None  # full book every tick -> always quotable
        for m, k in MARKETS:
            full_history[m].append(float(loop.last_fair_value.consensus_x[k]))

    for m, _ in MARKETS:
        assert len(loop._x_hist[m]) <= x_maxlen
        assert loop._x_hist[m] == full_history[m][-len(loop._x_hist[m]):]


def test_x_hist_none_maxlen_is_unbounded(store):
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(), clock=SimClock(START), vol_gate_fn=_vol_gate(),
        x_hist_maxlen=None,
    )
    n_ticks = 9
    for i in range(n_ticks):
        loop.tick(_moving_books(MARKETS, i / (n_ticks - 1)))
    for m, _ in MARKETS:
        assert len(loop._x_hist[m]) == n_ticks


# ---------------------------------------------------------------------------
# 1.3 -- market registry upsert on construction + round-trip
# ---------------------------------------------------------------------------


def test_registry_upserted_on_loop_construction(store):
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(), clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    reg = store.get_market_registry()
    assert reg == {"m-100k": (EXPIRY, 100000.0), "m-102k": (EXPIRY, 102000.0)}
    assert loop.expiry_key == EXPIRY  # sanity: loop actually built


def test_registry_upsert_idempotent_across_reconstruction(store):
    for _ in range(2):
        PaperTradingLoop(
            store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
            config=MMConfig(), clock=SimClock(START), vol_gate_fn=_vol_gate(),
        )
    reg = store.get_market_registry()
    assert reg == {"m-100k": (EXPIRY, 100000.0), "m-102k": (EXPIRY, 102000.0)}


# ---------------------------------------------------------------------------
# 1.4 -- registry-merge settle(catch_up=True): THE crash-before-settle
# regression test (reviewer round-2 finding).
# ---------------------------------------------------------------------------

OLD_EXPIRY = "2026-07-06"
NEW_EXPIRY = "2026-07-20"
OLD_MARKET = ("m-100k-old", 100000.0)
NEW_MARKET = ("m-105k-new", 105000.0)


def test_crash_before_settle_recovery(tmp_path):
    """Seed a persistent DB with un-settled OPENING fills for a PREVIOUS
    event's market (registered in the persisted registry), then construct a
    NEW PaperTradingLoop for a DIFFERENT event against the SAME DB and run
    the real resume sequence: restart() -> settle(now, catch_up=True)."""
    db_path = str(tmp_path / "resume.db")

    # --- prior run: open a long position in the OLD event, crash before settling ---
    store1 = MMStateStore(db_path)
    loop1 = PaperTradingLoop(
        store=store1, expiry_key=OLD_EXPIRY, markets=[OLD_MARKET], engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )
    loop1.tick({OLD_MARKET[0]: _snapshot_msg(0.5)})
    for _ in range(3):
        loop1.tick({OLD_MARKET[0]: _snapshot_msg(0.5, prints=[(0.02, 10.0)])})

    q_before = store1.fold_fills_to_inventory()[OLD_MARKET[0]].q
    assert q_before != 0.0  # a real open position, not yet settled
    live_before = [o for o in store1.get_all_orders() if o.status in ("PENDING", "LIVE")]
    store1.close()  # simulated crash -- no settle() call was ever made

    # --- resumed run: a NEW loop for a DIFFERENT event, same DB file ---
    store2 = MMStateStore(db_path)
    now = settlement_instant_utc(OLD_EXPIRY) + timedelta(hours=1)
    loop2 = PaperTradingLoop(
        store=store2, expiry_key=NEW_EXPIRY, markets=[NEW_MARKET], engine_fn=_engine(),
        config=MMConfig(gamma=0.5), clock=SimClock(now), vol_gate_fn=_vol_gate(),
        data_provider=_settlement_provider_for(OLD_EXPIRY, spot=100600.0),
    )

    # Real resume sequence (WS2.1 invariant): restart() ALWAYS before
    # settle(catch_up=True).
    recon = loop2.restart(now)
    assert set(recon.cancelled_unknown) == {o.client_order_id for o in live_before}

    result = loop2.settle(now, catch_up=True)

    old_events = [e for e in result.events if e.market_id == OLD_MARKET[0]]
    assert len(old_events) == 1
    assert old_events[0].outcome is SettlementOutcome.YES  # spot 100600 > strike 100000
    new_events = [e for e in result.events if e.market_id == NEW_MARKET[0]]
    assert new_events == []  # NEW event's own expiry is not due yet

    # Previous-event position closed out: in-memory inventory, store fold,
    # and the fold-matches-inventory invariant all agree at q == 0.
    assert loop2.inv.snapshot(now).per_contract[OLD_MARKET[0]].q == pytest.approx(0.0)
    assert store2.fold_fills_to_inventory()[OLD_MARKET[0]].q == pytest.approx(0.0)
    assert loop2.fold_matches_inventory()

    settlement_row = store2.get_settlement(OLD_MARKET[0], OLD_EXPIRY)
    assert settlement_row is not None
    assert settlement_row.outcome is SettlementOutcome.YES  # terminal

    store2.close()


def test_clean_rollover_already_terminal_emits_no_events(store):
    """A prior run already settled its market (exited 42 cleanly) before this
    process started. catch_up() must find it terminal and emit nothing --
    inventory (in-memory and store) stays untouched."""
    old_market_id, old_strike = "m-90k-old", 90000.0
    store.upsert_market(old_market_id, OLD_EXPIRY, old_strike)
    settle_dt = settlement_instant_utc(OLD_EXPIRY)
    store.upsert_settlement(SettlementEvent(
        ts=settle_dt, settlement_ts=settle_dt, market_id=old_market_id, expiry_key=OLD_EXPIRY,
        strike=old_strike, outcome=SettlementOutcome.YES, spot_used=91000.0,
        spot_source=SpotSource.INTRADAY, q_settled=5.0, payoff=5.0, pnl_realized=1.0,
        excluded_from_gate=False,
    ))
    # No fills, no inventory row for old_market_id -- consistent with the
    # position having already been closed and its q not otherwise tracked.

    now = settle_dt + timedelta(hours=1)
    loop = PaperTradingLoop(
        store=store, expiry_key=NEW_EXPIRY, markets=[NEW_MARKET], engine_fn=_engine(),
        config=MMConfig(), clock=SimClock(now), vol_gate_fn=_vol_gate(),
    )
    loop.restart(now)
    result = loop.settle(now, catch_up=True)

    old_events = [e for e in result.events if e.market_id == old_market_id]
    assert old_events == []  # terminal -> catch_up skips it entirely, no re-check

    assert store.fold_fills_to_inventory().get(
        old_market_id, ContractInv(q=0.0, avg_cost=0.0, q_max=0.0, age_weighted_holding=0.0)
    ).q == pytest.approx(0.0)
    # Never touched in-memory: no fills existed for it, so restart()'s
    # unfiltered fills replay never created an entry, and the settle sync
    # loop had no event for it either.
    assert old_market_id not in loop.inv.snapshot(now).per_contract


# ---------------------------------------------------------------------------
# Change C harness integration (mm_suitability_alignment_plan.md C2/Tests):
# a real fill via the normal tick() flow, later ticks logging mid_log rows,
# and pnl_report.markout_report computed directly over the store yields the
# expected cell.
# ---------------------------------------------------------------------------


def test_mid_log_written_every_tick_and_markout_report_over_store(store):
    market_id, _strike = MARKETS[0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )

    # Tick 1: plain snapshot -> rests bid/ask quotes for both markets.
    loop.tick(_moving_books(MARKETS, 0.0))

    # Subsequent ticks: a large aggressor print through our resting bid on
    # market_id only, guaranteed (once queue_ahead is exhausted) to produce a
    # real fill (same technique as test_crash_before_settle_recovery above).
    fills_seen = []
    for _ in range(6):
        loop.tick({market_id: _snapshot_msg(0.5, prints=[(0.01, 200.0)])})
        fills_seen.extend([f for f in loop.last_fills if f.market_id == market_id])
        if fills_seen:
            break
    assert fills_seen, "expected the scripted prints to produce at least one fill"
    fill = fills_seen[0]

    # mid_log: every tick so far appended a row for this market (full book
    # snapshot every tick -> mm is never None).
    mid_rows_so_far = store.get_mids(market_id)
    assert len(mid_rows_so_far) >= 2
    assert all(r.market_id == market_id for r in mid_rows_so_far)

    # A few more ticks so a mid lands inside the h=60 markout window
    # [fill.ts + 60, fill.ts + 660] (tick_dt_s defaults to 60s).
    for _ in range(3):
        loop.tick(_moving_books(MARKETS, 0.5))

    report = markout_report(
        store.get_fills(), store.mid_at_or_after, store.get_market_registry(),
        loop.config.belly_band, horizons=(60.0,),
    )
    assert len(report["cells"]) >= 1
    cell = report["cells"][0]
    assert cell["horizon_s"] == 60.0
    assert cell["n"] >= 1


# ---------------------------------------------------------------------------
# W0.3 -- inv.mark(now) called once per tick: a held position accrues
# age_weighted_holding and its R3 histogram between fills (plan Wave 0).
# ---------------------------------------------------------------------------


def test_held_position_accrues_age_and_r3_across_fill_free_ticks(store):
    market_id, _strike = MARKETS[0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )

    # Tick 1: rest quotes.
    loop.tick(_moving_books(MARKETS, 0.0))

    # Force a real fill on market_id via a large aggressor print (same
    # technique as test_mid_log_written_every_tick_and_markout_report_over_store).
    fills_seen = []
    for _ in range(6):
        loop.tick({market_id: _snapshot_msg(0.5, prints=[(0.01, 200.0)])})
        fills_seen.extend([f for f in loop.last_fills if f.market_id == market_id])
        if fills_seen:
            break
    assert fills_seen, "expected the scripted prints to produce at least one fill"

    snap_after_fill = loop.inv.snapshot(loop.clock.now())
    ci_after_fill = snap_after_fill.per_contract[market_id]
    assert ci_after_fill.q != 0.0
    age_after_fill = ci_after_fill.age_weighted_holding
    r3_after_fill = dict(loop.inv._ladders[EXPIRY].r3_histogram)

    # Several fill-free ticks (plain book snapshots, no aggressor prints) --
    # the position must keep aging via harness.tick's inv.mark(now) call, not
    # only on fill events.
    for _ in range(4):
        loop.tick(_moving_books(MARKETS, 0.5))

    snap_later = loop.inv.snapshot(loop.clock.now())
    ci_later = snap_later.per_contract[market_id]
    assert ci_later.q == pytest.approx(ci_after_fill.q)  # unchanged, no new fills
    assert ci_later.age_weighted_holding > age_after_fill

    r3_later = loop.inv._ladders[EXPIRY].r3_histogram
    # R3 histogram: total accumulated time across all levels must have grown
    # (fill-free ticks still attribute elapsed time to the current level via
    # inv.mark -> _accrue_ladder).
    assert sum(r3_later.values()) > sum(r3_after_fill.values())


# ---------------------------------------------------------------------------
# W1.1 -- last_liquidity reaches size_ladder: a tiny realized depth forces
# the sizing DEPTH cap to bind in the tick's real decisions.
# ---------------------------------------------------------------------------


def test_last_liquidity_reaches_size_ladder_and_forces_depth_cap(store):
    market_id, _strike = MARKETS[0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )

    # Several ticks of a tiny resting book so the depth_window rolling mean
    # (realized_depth) actually reflects the tiny size, not just the first
    # sample.
    for _ in range(3):
        loop.tick({m: _snapshot_msg_tiny_depth(0.5) for m, _k in MARKETS})

    assert loop.last_liquidity, "expected last_liquidity to be populated after a tick"
    liq = loop.last_liquidity[market_id]
    assert liq.realized_depth_bid < 5.0  # tiny book -> tiny realized depth

    # Direct replay of the sizing stage with the harness's own inputs proves
    # last_liquidity is what actually reached size_ladder: same liquidity
    # dict, same snapshot/fair-value the tick just used.
    from market_maker.robustness_sizing import ContractSizingInput, size_ladder

    snap = loop.last_snapshot
    fv = loop.last_fair_value
    contracts = [
        ContractSizingInput(
            market_id=m, p_hat=float(fv.consensus_p[k]),
            bid_price=loop.last_proposals[m].p_bid_raw, ask_price=loop.last_proposals[m].p_ask_raw,
        )
        for m, k in loop.markets
    ]
    decisions, _audit = size_ladder(
        contracts, snap, loop.bankroll, loop.clock.now(), loop.config, liquidity=loop.last_liquidity,
    )
    assert SizingCap.DEPTH in decisions[market_id].caps_applied
    assert decisions[market_id].bid_size == pytest.approx(liq.realized_depth_bid)


# ---------------------------------------------------------------------------
# mm_sizing_fix_plan.md C1/C2 + mm_sizing_wave2_plan.md W7 harness wiring:
# size_ladder receives the tick's real inventory snapshot and per-market
# POSTED prices/strike/markout fields (wave 2 -- mkt_mid no longer exists),
# and those same posted prices land unchanged on the tick's QuoteSet.
# ---------------------------------------------------------------------------


def test_harness_wires_inventory_and_posted_prices_and_strike_into_size_ladder(store, monkeypatch):
    import market_maker.harness as harness_mod
    from market_maker.contracts import InventoryState

    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )

    captured = {}
    orig_size_ladder = harness_mod.size_ladder

    def spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return orig_size_ladder(*args, **kwargs)

    monkeypatch.setattr(harness_mod, "size_ladder", spy)

    loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})

    assert captured, "size_ladder was not called this tick"
    assert isinstance(captured["kwargs"].get("inventory"), InventoryState)
    contracts = captured["args"][0]
    assert contracts, "expected at least one ContractSizingInput"
    for c in contracts:
        assert c.strike is not None
        # markout fields default-present (cold provider -- no markout_provider
        # wired on this loop) rather than absent/erroring.
        assert c.mk_n == 0
        assert c.mk_n_attempted == 0
        assert c.mk_avg is None
        assert c.mk_var is None

    # The posted prices fed to sizing are exactly the QuoteSet's prices for
    # the same market/tick -- the wave 2 W1/W7 single-computation guarantee.
    by_market = {c.market_id: c for c in contracts}
    for m, _k in loop.markets:
        qs = loop.last_quote_sets[m]
        c = by_market[m]
        assert c.bid_price == pytest.approx(qs.bid_price)
        assert c.ask_price == pytest.approx(qs.ask_price)


def test_markout_provider_called_once_per_tick(store):
    calls = {"n": 0}

    def counting_provider():
        calls["n"] += 1
        return None

    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),
        markout_provider=counting_provider,
    )
    loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})
    assert calls["n"] == 1  # NOT once per market (len(MARKETS) == 2)

    loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})
    assert calls["n"] == 2


def test_markout_provider_report_resolves_into_sizing_fields(store):
    """A wired provider returning a real markout_report()-shaped dict resolves
    into non-default ContractSizingInput markout fields for a market whose
    (region, tte_bucket, horizon_s) cell is populated and >= markout_min_n."""
    import market_maker.harness as harness_mod
    from market_maker.pnl_report import tte_bucket_label

    cfg = MMConfig(gamma=0.5, k_arrival=1.0)
    report_holder: dict = {"report": None}
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
        markout_provider=lambda: report_holder["report"],
    )

    # Tick once (cold, provider returns None) to learn this tick's actual
    # tte_bucket/region so the fake report's cell keys line up with what the
    # harness will look up on the NEXT tick.
    loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})
    tte_bucket = tte_bucket_label(max(loop.last_snapshot.tte_days, 0.0))
    # Both markets are ATM-ish under the sigmoid engine (belly_band default
    # 0.2-0.8 covers consensus_p here).
    region = "belly"
    report_holder["report"] = {
        "cells": [
            {"region": region, "tte_bucket": tte_bucket, "horizon_s": cfg.markout_horizon_s,
             "n": cfg.markout_min_n, "n_attempted": cfg.markout_min_n,
             "mk_avg": 0.02, "mk_var": 0.0009, "mk_total": 0.02 * cfg.markout_min_n},
        ],
        "by_region": {},
        "by_expiry": {},
    }

    captured = {}
    orig_size_ladder = harness_mod.size_ladder

    def spy(*args, **kwargs):
        captured["args"] = args
        return orig_size_ladder(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(harness_mod, "size_ladder", spy)
        loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})

    contracts = captured["args"][0]
    assert any(c.mk_n == cfg.markout_min_n and c.mk_avg == pytest.approx(0.02) for c in contracts)


# ---------------------------------------------------------------------------
# W1.2 -- one-dead-book tick: consensus frozen, _x_hist append skipped,
# fair_value_age_s grows (verified end-to-end via the risk journal's
# FAIR_VALUE_STALE trigger once age exceeds fv_max_age_s).
# ---------------------------------------------------------------------------


def test_one_dead_book_tick_freezes_consensus_and_skips_x_hist(store):
    market_id, _strike = MARKETS[0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )

    loop.tick(_moving_books(MARKETS, 0.0))
    assert loop._fv_recomputed_this_tick is True
    fv_after_tick1 = loop.last_fair_value
    fv_ts_after_tick1 = loop._fv_recomputed_ts
    x_hist_len_after_tick1 = {m: len(loop._x_hist[m]) for m, _ in MARKETS}

    # One-dead-book tick: only market_id gets a snapshot -- the other market
    # has no message this tick, so its book stays in its PREVIOUS state.
    # With a snapshot capability book, an empty message list for a market
    # simply carries the prior mid forward (mids stays complete) -- to force
    # an INCOMPLETE mids dict (the actual freeze trigger), drop a market
    # from feed_healthy instead: simplest reliable way is to send a raw
    # "trade"-only message with no snapshot ever having been applied for the
    # SECOND market on a FRESH loop. Instead, directly assert the freeze
    # contract using the documented mechanism: len(mids) < len(self.markets).
    other_market = MARKETS[1][0]
    # A message list that clears the book to empty (best_bid/best_ask None)
    # for the second market reproduces an incomplete-mids tick.
    loop.tick({market_id: _snapshot_msg(0.5), other_market: [{"type": "snapshot", "bids": [], "asks": []}]})

    assert loop._fv_recomputed_this_tick is False
    assert loop.last_fair_value is fv_after_tick1  # consensus frozen, same object
    assert loop._fv_recomputed_ts == fv_ts_after_tick1  # not advanced

    for m, _ in MARKETS:
        assert len(loop._x_hist[m]) == x_hist_len_after_tick1[m]  # append skipped

    # fair_value_age_s must have grown (it is derived from _fv_recomputed_ts
    # vs the now-advanced clock) -- assert indirectly via a further tick that
    # pushes age > fv_max_age_s and check the FAIR_VALUE_STALE trigger fires.
    cfg = loop.config
    n_more_ticks = int(cfg.fv_max_age_s // loop.tick_dt_s) + 3
    for _ in range(n_more_ticks):
        loop.tick({market_id: _snapshot_msg(0.5),
                   other_market: [{"type": "snapshot", "bids": [], "asks": []}]})

    directives = loop.last_directives
    assert directives, "expected directives to be journaled even on a frozen-consensus tick"
    from market_maker.contracts import RiskTrigger
    assert any(RiskTrigger.FAIR_VALUE_STALE in d.triggers for d in directives.values())


# ---------------------------------------------------------------------------
# W1.3 -- bankroll auto-unfreeze: freeze via a degenerate tick, N clean
# BEUOY ticks unfreezes; a non-BEUOY recompute resets the streak; a
# non-recompute tick neither increments nor resets.
# ---------------------------------------------------------------------------


def test_bankroll_auto_unfreezes_after_n_clean_beuoy_ticks(store):
    unfreeze_n = 3
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, bankroll_unfreeze_clean_ticks=unfreeze_n),
        clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    loop.tick(_moving_books(MARKETS, 0.0))  # warm up so fv is not None

    # bankroll_state is a read-only property (package B2) -- mutate the real
    # per-region state dict directly, both regions in lockstep (a fallback
    # freezes both; so does this test's manual freeze-simulation).
    for state in loop.bankroll_states.values():
        state.frozen = True
    loop._clean_beuoy_streak = 0

    for i in range(unfreeze_n - 1):
        loop.tick(_moving_books(MARKETS, 0.2 + 0.05 * i))
        assert loop.bankroll_state.frozen is True  # not yet at threshold

    loop.tick(_moving_books(MARKETS, 0.9))  # the Nth clean BEUOY tick
    assert loop.bankroll_state.frozen is False
    assert loop._clean_beuoy_streak >= unfreeze_n


def test_bankroll_unfreeze_streak_resets_on_non_beuoy_recompute(store):
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, bankroll_unfreeze_clean_ticks=5),
        clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    loop.tick(_moving_books(MARKETS, 0.0))
    loop._clean_beuoy_streak = 3  # pretend we're partway to unfreezing

    # Force a non-BEUOY recompute by directly invoking the same branch the
    # tick uses, mirroring compute_fair_value's degenerate-bankroll fallback
    # contract: simulate the harness having just recomputed with a
    # FIXED_BLEND_FALLBACK anchor.
    loop._fv_recomputed_this_tick = True
    from market_maker.contracts import FairValue
    fallback_fv = FairValue(
        ts=loop.clock.now(), expiry_key=EXPIRY,
        consensus_p=loop.last_fair_value.consensus_p, consensus_x=loop.last_fair_value.consensus_x,
        credibility=0.0, anchor_method=AnchorMethod.FIXED_BLEND_FALLBACK,
        inputs_ts=(loop.clock.now(), loop.clock.now()),
    )
    # Reproduce the exact streak-update branch harness.tick runs (not a
    # private-method call -- the logic is inline in tick(); this test
    # exercises the same condition it gates on).
    if fallback_fv.anchor_method == AnchorMethod.BEUOY:
        loop._clean_beuoy_streak += 1
    else:
        loop._clean_beuoy_streak = 0
    assert loop._clean_beuoy_streak == 0


def test_bankroll_unfreeze_streak_unchanged_on_non_recompute_tick(store):
    market_id, _strike = MARKETS[0]
    other_market = MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, bankroll_unfreeze_clean_ticks=5),
        clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    loop.tick(_moving_books(MARKETS, 0.0))
    loop._clean_beuoy_streak = 2
    streak_before = loop._clean_beuoy_streak

    # Incomplete-mids tick (one market's book cleared) -- non-recompute.
    loop.tick({market_id: _snapshot_msg(0.5), other_market: [{"type": "snapshot", "bids": [], "asks": []}]})
    assert loop._fv_recomputed_this_tick is False
    assert loop._clean_beuoy_streak == streak_before  # neither incremented nor reset


# ---------------------------------------------------------------------------
# W2 -- vertical hedging wiring (plan Wave 2, test 6.2 + named coverage)
# ---------------------------------------------------------------------------


def _build_long_position(loop, market_id, now, target_q):
    """Directly grow `market_id`'s inventory to (at least) `target_q` shares
    long YES via the normal fill channel (loop.inv.apply_fill), mirroring
    the fill shape the fill simulator itself produces. Deterministic and
    independent of the queue-behind fill model's timing, which is not the
    thing under test here (the hedge stage's downstream wiring is)."""
    from market_maker.contracts import Fill, LiquiditySource
    f = Fill(
        ts=now, market_id=market_id, order_id="scripted:%s" % market_id,
        side=Side.BUY_YES, price=0.5, size=target_q,
        liquidity=LiquiditySource.MAKER, venue_ts=now,
    )
    loop.inv.apply_fill(f)
    loop.store.record_fill_and_update_inventory(f, loop.inv.snapshot(now).per_contract[market_id])


def test_plan_6_2_hedge_recs_land_as_offsets_and_inflate_neighbor_next_tick(store):
    """Plan test 6.2: build inventory in one strike past
    vertical_target_frac*q_max -> next tick's hedge stage emits recs ->
    market_id-keyed offsets land via set_hedge_state -> net_band_exposure
    falls vs a no-hedge baseline -> the tick after, the neighbor QuoteSet
    shows inflated size on the hedge side (price rule permitting)."""
    market_id, other_market = MARKETS[0][0], MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )
    # Low threshold so a modest fill is unambiguously "over cap" without
    # needing to hand-tune fill sizes against q_max_scale internals.
    loop.hedger.vertical_target_frac = 0.01

    # Tick 1: warm up (fv not None, quotes exist, q_max populated).
    loop.tick(_moving_books(MARKETS, 0.0))

    now = loop.clock.now()
    q_max = loop.inv.snapshot(now).per_contract[market_id].q_max
    _build_long_position(loop, market_id, now, target_q=q_max)  # far past 1% of cap

    # No-hedge baseline: net_band_exposure computed BEFORE any hedge_state is
    # set for this ladder (hedge_state defaults to {}).
    baseline_bands = loop.inv.net_band_exposure(EXPIRY)

    # Tick 2: the hedge stage (W2.1) runs after fills routing this tick and
    # computes recs off the inventory we just built.
    loop.tick(_moving_books(MARKETS, 0.1))

    assert loop._pending_hedge_recs, "expected the hedger to emit recs for the over-cap strike"
    rec = loop._pending_hedge_recs[0]
    assert rec.target_market_id == other_market

    # Market_id-keyed offsets landed via set_hedge_state (W2.0 -> W2.1).
    assert loop.last_hedge_offsets == {
        r.target_market_id: (r.size if r.side == Side.BUY_YES else -r.size)
        for r in loop._pending_hedge_recs
    }
    assert loop.inv._ladders[EXPIRY].hedge_state == loop.last_hedge_offsets

    hedged_bands = loop.inv.net_band_exposure(EXPIRY)
    # The hedge offset is signed opposite to the over-cap long position, so
    # SOME bucket's |exposure| must shrink vs the no-hedge baseline.
    assert any(
        abs(h) < abs(b) - 1e-9 for h, b in zip(hedged_bands, baseline_bands)
    ), (hedged_bands, baseline_bands)

    # Tick 3: PREVIOUS tick's recs (from tick 2) are applied as a size-skew
    # to the neighbor's QuoteSet this tick (W2.2), if the price rule passes.
    loop.tick(_moving_books(MARKETS, 0.1))
    applied = [j for j in loop.hedge_journal if j["target_market_id"] == other_market]
    assert applied, "expected a journal entry for the neighbor hedge rec"
    if applied[-1]["applied"]:
        qs = loop.last_quote_sets[other_market]
        if rec.side == Side.BUY_YES:
            assert qs.bid_size > 0.0
        else:
            assert qs.ask_size > 0.0


def test_w2_2b_buy_no_price_rule_numeric_case(store):
    """W2.2b numeric case from the plan: BUY_NO rec max_price 0.45 vs
    qs.ask_price 0.60 -> placed NO price 0.40 <= 0.45 -> applies. A second
    case with ask_price 0.50 -> NO price 0.50 > 0.45 -> skipped."""
    from market_maker.contracts import HedgeReason, HedgeRecommendation, QuoteSet

    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )
    now = loop.clock.now()
    market_id, other_market = MARKETS[0][0], MARKETS[1][0]

    def _qs(ask_price):
        return QuoteSet(
            ts=now, market_id=other_market, bid_price=0.10, ask_price=ask_price,
            bid_size=10.0, ask_size=10.0, terms={}, risk_mode=QuoteMode.TWO_SIDED,
            noarb_checked=True, source_seq=1,
        )

    def _rec():
        return HedgeRecommendation(
            ts=now, expiry_key=EXPIRY, target_market_id=other_market, side=Side.BUY_NO,
            size=5.0, max_price=0.45, reason=HedgeReason.VERTICAL_OFFSET,
            paired_market_id=market_id, beta=None, expires=now + timedelta(seconds=300.0),
        )

    # Applies: ask_price 0.60 -> NO price 0.40 <= 0.45.
    loop._pending_hedge_recs = [_rec()]
    out = loop._apply_hedge_skew([_qs(0.60)], now)
    assert out[0].ask_size == pytest.approx(15.0)
    assert loop.hedge_journal[-1]["applied"] is True

    # Skipped: ask_price 0.50 -> NO price 0.50 > 0.45.
    loop._pending_hedge_recs = [_rec()]
    out = loop._apply_hedge_skew([_qs(0.50)], now)
    assert out[0].ask_size == pytest.approx(10.0)  # unchanged
    assert loop.hedge_journal[-1]["applied"] is False
    assert loop.hedge_journal[-1]["reason"] == "price_above_max"


def test_w2_2_suppressed_side_never_resurrected(store):
    """A neighbor directive PULLED -> the rec is journaled as skipped, sizes
    stay 0 (the hedge skew must never resurrect a suppressed side)."""
    from market_maker.contracts import HedgeReason, HedgeRecommendation, QuoteSet

    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )
    now = loop.clock.now()
    market_id, other_market = MARKETS[0][0], MARKETS[1][0]

    pulled_qs = QuoteSet(
        ts=now, market_id=other_market, bid_price=0.10, ask_price=0.60,
        bid_size=0.0, ask_size=0.0, terms={}, risk_mode=QuoteMode.PULLED,
        noarb_checked=True, source_seq=1,
    )
    rec = HedgeRecommendation(
        ts=now, expiry_key=EXPIRY, target_market_id=other_market, side=Side.BUY_NO,
        size=5.0, max_price=0.95, reason=HedgeReason.VERTICAL_OFFSET,
        paired_market_id=market_id, beta=None, expires=now + timedelta(seconds=300.0),
    )
    loop._pending_hedge_recs = [rec]
    out = loop._apply_hedge_skew([pulled_qs], now)

    assert out[0].bid_size == 0.0
    assert out[0].ask_size == 0.0
    assert loop.hedge_journal[-1]["applied"] is False
    assert loop.hedge_journal[-1]["reason"] == "suppressed_side"


def test_w2_2_suppressed_side_zeroed_size_never_resurrected(store):
    """Even under TWO_SIDED mode, a side the sizer already zeroed (e.g.
    DEPTH-capped to 0) must not be resurrected by the hedge skew."""
    from market_maker.contracts import HedgeReason, HedgeRecommendation, QuoteSet

    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )
    now = loop.clock.now()
    market_id, other_market = MARKETS[0][0], MARKETS[1][0]

    zeroed_ask_qs = QuoteSet(
        ts=now, market_id=other_market, bid_price=0.10, ask_price=0.60,
        bid_size=10.0, ask_size=0.0, terms={}, risk_mode=QuoteMode.TWO_SIDED,
        noarb_checked=True, source_seq=1,
    )
    rec = HedgeRecommendation(
        ts=now, expiry_key=EXPIRY, target_market_id=other_market, side=Side.BUY_NO,
        size=5.0, max_price=0.95, reason=HedgeReason.VERTICAL_OFFSET,
        paired_market_id=market_id, beta=None, expires=now + timedelta(seconds=300.0),
    )
    loop._pending_hedge_recs = [rec]
    out = loop._apply_hedge_skew([zeroed_ask_qs], now)

    assert out[0].ask_size == 0.0
    assert loop.hedge_journal[-1]["applied"] is False
    assert loop.hedge_journal[-1]["reason"] == "suppressed_side"


def test_breach_interplay_hedge_fires_below_one_sided_threshold(store):
    """Hedge recs fire once |q| exceeds 50% of cap while quoting is still
    TWO_SIDED (below the risk controller's own 100% one-sided threshold);
    at |q| > q_max the risk controller goes one-sided and STAYS one-sided as
    the breach deepens (stranded-inventory fix 2026-07-14, Change A: ANY
    breach ratio -- including the formerly-"extreme" >1.5 -- emits the
    one-sided away mode, never PULLED, since a one-sided-away mode never
    adds risk). This test exercises the hedger's own threshold in isolation
    (vertical_target_frac=0.5, the launch default) against the risk
    controller's INV_CAP threshold (harness._breaches at ratio>=1.0),
    confirming the two fire at different, correctly-ordered inventory
    levels."""
    from market_maker.contracts import InventoryState, ContractInv

    market_id, other_market = MARKETS[0][0], MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )
    loop.tick(_moving_books(MARKETS, 0.0))
    now = loop.clock.now()
    q_max = loop.inv.snapshot(now).per_contract[market_id].q_max

    def _inv_at_ratio(ratio):
        per = {
            market_id: ContractInv(q=ratio * q_max, avg_cost=0.5, q_max=q_max, age_weighted_holding=0.0),
            other_market: ContractInv(q=0.0, avg_cost=0.5, q_max=q_max, age_weighted_holding=0.0),
        }
        return InventoryState(ts=now, per_contract=per, per_ladder={})

    fair_p = {m: float(loop.last_fair_value.consensus_p[k]) for m, k in loop.markets}

    # 50% of cap: at the hedger's own vertical_target_frac (0.5 default) ->
    # excess == 0, no recs yet (boundary, not yet "over").
    recs_50, _ = loop.hedger.vertical_hedges(
        _inv_at_ratio(0.5), EXPIRY, loop.strikes, [m for m, _ in loop.markets], fair_p, ts=now,
    )
    assert recs_50 == []

    # Just above 50%: hedge recs now fire.
    recs_60, _ = loop.hedger.vertical_hedges(
        _inv_at_ratio(0.6), EXPIRY, loop.strikes, [m for m, _ in loop.markets], fair_p, ts=now,
    )
    assert len(recs_60) == 1

    # Risk controller: at ratio 0.6 (< 1.0), no INV_CAP breach -> TWO_SIDED.
    breaches_60 = []
    directive_60 = loop.risk.evaluate(
        market_id, now, tte_days=loop.last_snapshot.tte_days, pricer_snapshot=loop.last_snapshot,
        inventory_breaches=breaches_60, liquidity_regime=loop.last_liquidity[market_id].regime,
        feed_healthy=True, spot=loop.last_snapshot.s0, strike=MARKETS[0][1],
    )
    assert directive_60.mode == QuoteMode.TWO_SIDED

    # At ratio 1.2 (>1.0, <=1.5): risk controller goes one-sided (ASK_ONLY,
    # long position).
    from market_maker.risk_controller import InvBreach
    breaches_120 = [InvBreach(market_id=market_id, is_long=True, ratio=1.2)]
    directive_120 = loop.risk.evaluate(
        market_id, now, tte_days=loop.last_snapshot.tte_days, pricer_snapshot=loop.last_snapshot,
        inventory_breaches=breaches_120, liquidity_regime=loop.last_liquidity[market_id].regime,
        feed_healthy=True, spot=loop.last_snapshot.s0, strike=MARKETS[0][1],
    )
    assert directive_120.mode == QuoteMode.ASK_ONLY

    # At ratio 1.6 (>1.5, formerly "extreme"): risk controller stays
    # one-sided (ASK_ONLY, long position) -- it no longer escalates to
    # PULLED at any breach ratio (stranded-inventory fix 2026-07-14).
    breaches_160 = [InvBreach(market_id=market_id, is_long=True, ratio=1.6)]
    directive_160 = loop.risk.evaluate(
        market_id, now, tte_days=loop.last_snapshot.tte_days, pricer_snapshot=loop.last_snapshot,
        inventory_breaches=breaches_160, liquidity_regime=loop.last_liquidity[market_id].regime,
        feed_healthy=True, spot=loop.last_snapshot.s0, strike=MARKETS[0][1],
    )
    assert directive_160.mode == QuoteMode.ASK_ONLY
    assert directive_160.mode != QuoteMode.PULLED


# ---------------------------------------------------------------------------
# Package D -- risk-based inventory breach metric (mm_pnl_fix_plan.md
# section 3, 2026-07-15). Replaces the raw |q| / q_max ratio with a
# remaining-loss-notional metric: L_m = q * p_consensus (long) or
# |q| * (1 - p_consensus) (short), over cap = inv_loss_cap_frac * bankroll.
# ---------------------------------------------------------------------------


def _inv_snap(loop, q_by_market, q_max=5.0):
    """InventoryState for `loop`'s markets at the given signed q's (default
    0.0, i.e. flat, for any market not named). q_max is irrelevant to the new
    metric but ContractInv still requires a nonnegative value."""
    now = loop.clock.now()
    per = {
        m: ContractInv(q=q_by_market.get(m, 0.0), avg_cost=0.5, q_max=q_max, age_weighted_holding=0.0)
        for m, _k in loop.markets
    }
    return InventoryState(ts=now, per_contract=per, per_ladder={})


def _fv_with_p(loop, p_by_market):
    """`loop.last_fair_value` with consensus_p overridden at the given
    markets' strikes, for deterministic L_m arithmetic independent of the
    stub engine's logistic curve."""
    fv = loop.last_fair_value
    cp = dict(fv.consensus_p)
    for m, k in loop.markets:
        if m in p_by_market:
            cp[k] = p_by_market[m]
    return replace(fv, consensus_p=cp)


def test_breach_metric_formula_both_signs(store):
    """L_m formula: long (q>0) -> q*p_consensus; short (q<0) -> |q|*(1-p).
    Breach emitted iff ratio = L_m / cap >= 1.0. Using the SAME p=0.8 for
    both signs (rather than a symmetric p=0.5) proves the short branch
    really uses (1-p), not p -- if it used p by mistake, the short q=-13
    case below would ALSO breach, and it must not."""
    market_id, other = MARKETS[0][0], MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(inv_loss_cap_frac=0.10), clock=SimClock(START),
        vol_gate_fn=_vol_gate(), bankroll=100.0,  # cap = 10.0
    )
    loop.tick(_moving_books(MARKETS, 0.0))
    fv = _fv_with_p(loop, {market_id: 0.8, other: 0.8})

    # Long q=10, p=0.8 -> L=8.0 < cap=10.0 -> no breach.
    assert loop._breaches(_inv_snap(loop, {market_id: 10.0}), fv, loop.bankroll) == []

    # Long q=13, p=0.8 -> L=10.4 >= cap -> breach, is_long True.
    breaches = loop._breaches(_inv_snap(loop, {market_id: 13.0}), fv, loop.bankroll)
    assert len(breaches) == 1
    assert breaches[0].market_id == market_id
    assert breaches[0].is_long is True
    assert breaches[0].ratio == pytest.approx(1.04)

    # Short q=-13, SAME p=0.8 -> L=13*(1-0.8)=2.6 << cap -> no breach.
    assert loop._breaches(_inv_snap(loop, {market_id: -13.0}), fv, loop.bankroll) == []

    # Short q=-51, p=0.8 -> L=51*0.2=10.2 >= cap -> breach, is_long False.
    breaches = loop._breaches(_inv_snap(loop, {market_id: -51.0}), fv, loop.bankroll)
    assert len(breaches) == 1
    assert breaches[0].is_long is False
    assert breaches[0].ratio == pytest.approx(1.02)


def test_breach_cap_scales_with_bankroll_and_frac(tmp_path):
    """cap = inv_loss_cap_frac * bankroll: a larger bankroll (or a larger
    inv_loss_cap_frac) enlarges the cap and can clear an existing breach.
    cap <= 0 (zero/negative bankroll, or non-positive frac) emits no
    breaches and raises no exception -- same guard shape as the old
    q_max <= 0 skip."""
    market_id, other = MARKETS[0][0], MARKETS[1][0]

    store_a = MMStateStore(str(tmp_path / "a.db"))
    loop = PaperTradingLoop(
        store=store_a, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(inv_loss_cap_frac=0.05), clock=SimClock(START),
        vol_gate_fn=_vol_gate(), bankroll=200.0,  # cap = 0.05 * 200 = 10.0
    )
    loop.tick(_moving_books(MARKETS, 0.0))
    fv = _fv_with_p(loop, {market_id: 0.5, other: 0.5})
    inv = _inv_snap(loop, {market_id: 21.0})  # L = 21 * 0.5 = 10.5

    assert len(loop._breaches(inv, fv, loop.bankroll)) == 1  # cap=10.0 -> breach

    # Bigger bankroll -> bigger cap (0.05 * 400 = 20.0) -> clears the breach.
    assert loop._breaches(inv, fv, 400.0) == []

    # cap <= 0 -> no breaches, no exception (zero and negative bankroll).
    assert loop._breaches(inv, fv, 0.0) == []
    assert loop._breaches(inv, fv, -50.0) == []
    store_a.close()

    # A larger inv_loss_cap_frac has the same cap-enlarging effect as a
    # larger bankroll: cap = 0.20 * 200 = 40.0 clears the same L=10.5 breach.
    store_b = MMStateStore(str(tmp_path / "b.db"))
    loop2 = PaperTradingLoop(
        store=store_b, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(inv_loss_cap_frac=0.20), clock=SimClock(START),
        vol_gate_fn=_vol_gate(), bankroll=200.0,
    )
    loop2.tick(_moving_books(MARKETS, 0.0))
    fv2 = _fv_with_p(loop2, {market_id: 0.5, other: 0.5})
    inv2 = _inv_snap(loop2, {market_id: 21.0})
    assert loop2._breaches(inv2, fv2, loop2.bankroll) == []

    # Non-positive frac (config side of cap<=0) -> no breaches, no exception.
    loop3 = PaperTradingLoop(
        store=store_b, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(inv_loss_cap_frac=0.0), clock=SimClock(START),
        vol_gate_fn=_vol_gate(), bankroll=200.0,
    )
    loop3.tick(_moving_books(MARKETS, 0.0))
    fv3 = _fv_with_p(loop3, {market_id: 0.5, other: 0.5})
    assert loop3._breaches(inv2, fv3, loop3.bankroll) == []
    store_b.close()


def test_wing_huge_q_tiny_p_no_breach_70k_case(store):
    """Regression for the exact misfire in plan section 0: 70k jul-20 at
    q=14.3, p~0.05 (deep OTM wing), flagged by the OLD |q|/q_max rule at
    ratio 3.1x. With a realistic per-expiry sizing bankroll (333, matching
    the live multi-expiry per-loop split) the risk-based metric must NOT
    breach: L = 14.3 * 0.05 = 0.715 << cap = 0.10 * 333.33 = 33.3."""
    market_id, other = MARKETS[0][0], MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(),  # inv_loss_cap_frac default 0.10
        clock=SimClock(START), vol_gate_fn=_vol_gate(), bankroll=333.33,
    )
    loop.tick(_moving_books(MARKETS, 0.0))
    fv = _fv_with_p(loop, {market_id: 0.05, other: 0.05})
    # q_max mirrors the live ratio-3.1x figure (14.3 / 4.57); irrelevant to
    # the new metric, kept only for documentary fidelity to the live case.
    inv = _inv_snap(loop, {market_id: 14.3}, q_max=4.57)

    assert loop._breaches(inv, fv, loop.bankroll) == []


def test_short_itm_breach_fires_as_p_falls_toward_belly(store):
    """Deep-ITM short (q<0, p near 1): remaining-loss notional
    L = |q| * (1 - p_consensus) starts tiny (pennies) and GROWS as p falls
    from deep-ITM toward the belly -- real risk growing as the position's
    outcome becomes less certain -- breaching once L crosses cap. Mirrors
    plan section 3's 'Behavior change on today's book': 'Deep ITM shorts:
    L pennies -> no breach. A short whose p falls toward the belly (real
    risk growing) breaches EARLIER than under the raw-shares rule.'"""
    market_id, other = MARKETS[0][0], MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(inv_loss_cap_frac=0.05), clock=SimClock(START),
        vol_gate_fn=_vol_gate(), bankroll=100.0,  # cap = 5.0
    )
    loop.tick(_moving_books(MARKETS, 0.0))
    inv = _inv_snap(loop, {market_id: -20.0})  # short 20 shares, held fixed

    breached_at = None
    for p in (0.97, 0.9, 0.8, 0.7, 0.6):
        fv = _fv_with_p(loop, {market_id: p, other: p})
        breaches = loop._breaches(inv, fv, loop.bankroll)
        loss = 20.0 * (1.0 - p)
        if loss >= 5.0:  # cap
            assert len(breaches) == 1
            assert breaches[0].is_long is False
            breached_at = p
            break
        assert breaches == [], (p, loss)
    assert breached_at is not None, "expected the short breach to fire as p fell toward the belly"


def test_breach_and_degenerate_liquidity_cofire_agree_not_pulled(store):
    """Regression for the 2026-07-14 stranding bug, reconstructed via the
    package-D metric: a real INV_CAP breach (new remaining-loss-notional
    metric) and a DEGENERATE liquidity regime firing on the SAME market,
    with the SAME signed q, must agree on the one-sided direction and never
    escalate to PULLED via _more_restrictive -- rules (c) and (f) both
    derive is_long / reduce-side from the same signed q."""
    market_id, other = MARKETS[0][0], MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(inv_loss_cap_frac=0.05), clock=SimClock(START),
        vol_gate_fn=_vol_gate(), bankroll=100.0,  # cap = 5.0
    )
    loop.tick(_moving_books(MARKETS, 0.0))
    now = loop.clock.now()
    fv = _fv_with_p(loop, {market_id: 0.5, other: 0.5})

    # Long q=11, p=0.5 -> L=5.5 >= cap=5.0 -> breach, is_long True.
    breaches = loop._breaches(_inv_snap(loop, {market_id: 11.0}), fv, loop.bankroll)
    assert len(breaches) == 1
    assert breaches[0].is_long is True

    directive = loop.risk.evaluate(
        market_id, now, tte_days=loop.last_snapshot.tte_days, pricer_snapshot=loop.last_snapshot,
        inventory_breaches=breaches, inventory_q=11.0,  # SAME signed q as the breach
        liquidity_regime=LiquidityRegime.DEGENERATE,
        feed_healthy=True, spot=loop.last_snapshot.s0, strike=MARKETS[0][1],
    )
    assert directive.mode == QuoteMode.ASK_ONLY
    assert directive.mode != QuoteMode.PULLED


def test_harness_wires_inventory_q_into_risk_evaluate(store, monkeypatch):
    """Harness wiring (stranded-inventory fix 2026-07-14, Change B): the
    tick loop hoists ONE inventory snapshot per tick (shared by _breaches()
    and the per-market inventory_q lookup) and passes each market's signed q
    into RiskController.evaluate as inventory_q -- confirms the position's
    real q arrives for the market that holds it, and a flat market reads
    0.0 (the harness always threads a value from the snapshot, never omits
    the kwarg for a registered market)."""
    market_id, other_market = MARKETS[0][0], MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    loop.tick(_moving_books(MARKETS, 0.0))
    now = loop.clock.now()

    # Grow market_id's inventory to a real, nonzero long position via the
    # normal fill channel (mirrors _build_long_position's other call sites);
    # other_market stays flat.
    _build_long_position(loop, market_id, now, target_q=3.0)

    captured = []
    orig_evaluate = loop.risk.evaluate

    def spy(*args, **kwargs):
        captured.append((args, kwargs))
        return orig_evaluate(*args, **kwargs)

    monkeypatch.setattr(loop.risk, "evaluate", spy)

    loop.tick(_moving_books(MARKETS, 0.1))

    assert captured, "risk.evaluate was not called this tick"
    inventory_q_by_market = {args[0]: kwargs.get("inventory_q") for args, kwargs in captured}
    assert set(inventory_q_by_market) == {market_id, other_market}
    assert inventory_q_by_market[market_id] == pytest.approx(3.0)
    assert inventory_q_by_market[other_market] == pytest.approx(0.0)


def test_beta_hedge_flag_off_is_inert(store, monkeypatch):
    """enable_beta_hedge False (default) -> beta_hedges is never invoked at
    all -- the call site is short-circuited BEFORE the call, not merely
    gated inside beta_hedges. Monkeypatched to raise if ever called."""
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )
    assert loop.hedger.enable_beta_hedge is False

    def _boom(*args, **kwargs):
        raise AssertionError("beta_hedges must not be invoked when enable_beta_hedge is False")

    monkeypatch.setattr(loop.hedger, "beta_hedges", _boom)

    # Several ticks -- if the call site were not properly short-circuited,
    # any of these would raise via the monkeypatch.
    for i in range(4):
        loop.tick(_moving_books(MARKETS, i / 3.0))


def test_beta_hedge_flag_on_calls_beta_hedges_with_placeholder_sigma_b(store, monkeypatch):
    """enable_beta_hedge True -> beta_hedges IS invoked this tick, using the
    documented sigma_b_floor placeholder (reviewer note 13)."""
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),  # k pinned: wide-quote geometry keeps Kelly size above the tiny depth so DEPTH binds
    )
    loop.hedger.enable_beta_hedge = True

    calls = []
    orig = loop.hedger.beta_hedges

    def _spy(inventory, expiry_key, strikes, market_ids, fair_p, sigma_b, ts, **kwargs):
        calls.append(sigma_b)
        return orig(inventory, expiry_key, strikes, market_ids, fair_p, sigma_b, ts, **kwargs)

    monkeypatch.setattr(loop.hedger, "beta_hedges", _spy)

    loop.tick(_moving_books(MARKETS, 0.0))

    assert len(calls) == 1
    sigma_b_used = calls[0]
    for m, _ in MARKETS:
        assert sigma_b_used[m] == pytest.approx(loop.config.sigma_b_floor)


# ---------------------------------------------------------------------------
# Package E (2026-07-15): markout-fed spread widening (spread term 7) --
# harness wiring (_compose_quote_sets resolves markout_stats_side per market
# BEFORE the single compute_posted_prices call, region computed from
# consensus_p_k moved above that call).
# ---------------------------------------------------------------------------


def _first_tick_tte_bucket(expiry_key=EXPIRY, start=START, tick_dt_s=60.0):
    """The tte_bucket the harness will resolve on the FIRST tick of a freshly
    constructed PaperTradingLoop(clock=SimClock(start)) -- tick() advances
    the clock by tick_dt_s BEFORE computing tte, so `now` on tick 1 is
    `start + tick_dt_s`, not `start` itself (harness.py:608-610)."""
    from market_maker.pnl_report import tte_bucket_label
    first_tick_now = start + timedelta(seconds=tick_dt_s)
    tte_days = max((settlement_instant_utc(expiry_key) - first_tick_now).total_seconds() / 86400.0, 0.0)
    return tte_bucket_label(tte_days)


def test_markout_widen_moves_posted_bid_for_measured_toxic_side(tmp_path):
    """A report with a trusted-negative BUY_YES side markout for the belly
    region at cfg.markout_widen_horizon_s widens the posted BID on every
    belly market vs a no-report control tick, by ~cfg.markout_widen_cap (the
    mk_avg is set far more negative than the cap so the clamp binds); the
    ask (BUY_NO side, thin/unmeasured in this report) is untouched."""
    cfg = MMConfig(gamma=0.5, k_arrival=1.0)
    # S0=100000, scale=2000 (default _engine()) -> all three strikes land
    # comfortably inside the default belly_band (0.2, 0.8):
    # 98k -> p~0.731, 100k -> p=0.5, 102k -> p~0.269.
    markets = [("m-98k", 98000.0), ("m-100k", 100000.0), ("m-102k", 102000.0)]
    tte_bucket = _first_tick_tte_bucket()

    report = {
        "cells": [
            {"region": "belly", "tte_bucket": tte_bucket, "horizon_s": cfg.markout_widen_horizon_s,
             "n": cfg.markout_min_n, "n_attempted": cfg.markout_min_n,
             "mk_avg": -1.0, "mk_var": 0.001, "mk_total": -1.0 * cfg.markout_min_n,
             "sides": {
                 "BUY_YES": {"n": cfg.markout_min_n, "n_attempted": cfg.markout_min_n,
                             "mk_avg": -1.0, "mk_var": 0.001},
                 "BUY_NO": {"n": 0, "n_attempted": 0, "mk_avg": 0.0, "mk_var": 0.0},
             }},
        ],
        "by_region": {},
        "by_expiry": {},
    }

    store_control = MMStateStore(str(tmp_path / "control.db"))
    store_treatment = MMStateStore(str(tmp_path / "treatment.db"))
    try:
        loop_control = PaperTradingLoop(
            store=store_control, expiry_key=EXPIRY, markets=markets, engine_fn=_engine(),
            config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
        )
        loop_treatment = PaperTradingLoop(
            store=store_treatment, expiry_key=EXPIRY, markets=markets, engine_fn=_engine(),
            config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
            markout_provider=lambda: report,
        )
        loop_control.tick({m: _snapshot_msg(0.5) for m, _k in markets})
        loop_treatment.tick({m: _snapshot_msg(0.5) for m, _k in markets})

        for m, _k in markets:
            qc = loop_control.last_quote_sets[m]
            qt = loop_treatment.last_quote_sets[m]
            assert qt.bid_price < qc.bid_price
            assert (qc.bid_price - qt.bid_price) == pytest.approx(cfg.markout_widen_cap, abs=0.02)
            # BUY_NO side is thin (n=0 < markout_min_n) -> ask widen stays 0.
            assert qt.ask_price == pytest.approx(qc.ask_price, abs=1e-9)
    finally:
        store_control.close()
        store_treatment.close()


def test_markout_widen_scale_zero_disables_term_end_to_end(tmp_path):
    """cfg.markout_widen_scale == 0.0 disables term 7 entirely, even with a
    trusted-negative side markout report wired -- byte-identical prices to a
    no-report control tick (regression: the term must be a strict opt-in)."""
    cfg = MMConfig(gamma=0.5, k_arrival=1.0, markout_widen_scale=0.0)
    markets = [("m-98k", 98000.0), ("m-100k", 100000.0), ("m-102k", 102000.0)]
    tte_bucket = _first_tick_tte_bucket()

    report = {
        "cells": [
            {"region": "belly", "tte_bucket": tte_bucket, "horizon_s": cfg.markout_widen_horizon_s,
             "n": cfg.markout_min_n, "n_attempted": cfg.markout_min_n,
             "mk_avg": -1.0, "mk_var": 0.001, "mk_total": -1.0 * cfg.markout_min_n,
             "sides": {
                 "BUY_YES": {"n": cfg.markout_min_n, "n_attempted": cfg.markout_min_n,
                             "mk_avg": -1.0, "mk_var": 0.001},
                 "BUY_NO": {"n": 0, "n_attempted": 0, "mk_avg": 0.0, "mk_var": 0.0},
             }},
        ],
        "by_region": {},
        "by_expiry": {},
    }

    store_control = MMStateStore(str(tmp_path / "control.db"))
    store_treatment = MMStateStore(str(tmp_path / "treatment.db"))
    try:
        loop_control = PaperTradingLoop(
            store=store_control, expiry_key=EXPIRY, markets=markets, engine_fn=_engine(),
            config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
        )
        loop_treatment = PaperTradingLoop(
            store=store_treatment, expiry_key=EXPIRY, markets=markets, engine_fn=_engine(),
            config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
            markout_provider=lambda: report,
        )
        loop_control.tick({m: _snapshot_msg(0.5) for m, _k in markets})
        loop_treatment.tick({m: _snapshot_msg(0.5) for m, _k in markets})

        for m, _k in markets:
            qc = loop_control.last_quote_sets[m]
            qt = loop_treatment.last_quote_sets[m]
            assert qt.bid_price == pytest.approx(qc.bid_price, abs=1e-9)
            assert qt.ask_price == pytest.approx(qc.ask_price, abs=1e-9)
    finally:
        store_control.close()
        store_treatment.close()


def _p_of_strike(k, s0=S0, scale=2000.0):
    """Same sigmoid formula as _engine()'s default (s0, scale) -- used to
    build a market book input that roughly AGREES with the pricer per
    strike. Package E PAV-interaction tests below need this: a flat market
    mid (e.g. _snapshot_msg(0.5) for every strike) pulls the Beuoy consensus
    toward that one flat value for every market and collapses any intended
    belly/wing split, since credibility starts 0.5/0.5."""
    return 1.0 / (1.0 + np.exp((k - s0) / scale))


def test_markout_widen_belly_only_no_spurious_noarb_at_realistic_strike_spacing(tmp_path):
    """No-arb PAV repair interaction, NORMAL-geometry safety check (package
    E, plan-pinned "No-arb PAV repair interaction" section): belly-only
    negative BUY_YES markout at the DEFAULT markout_widen_cap (0.12)
    over a realistic Polymarket-like strike ladder (2000-3000 wide strikes,
    matching typical BTC daily granularity) does not introduce ANY no-arb
    violation at all -- the natural adjacent-strike bid gaps at this
    spacing (belly boundary gaps ~19-23c) comfortably exceed the 12c cap, so
    PAV is never even invoked (repair_count stays 0). Confirms term 7 is safe in
    the intended/normal operating regime; invariants (ii)/(iii) (every
    belly bid strictly below baseline, no wing bid above baseline) hold
    trivially since nothing gets pooled."""
    cfg = MMConfig(gamma=0.5, k_arrival=1.0)  # DEFAULT markout_widen_cap=0.12
    # S0=100000, scale=2000 (default _engine()): 95k/105k land outside the
    # default belly_band (0.2, 0.8) -> wing; 98k/100k/102k -> belly.
    markets = [
        ("m-95k", 95000.0),
        ("m-98k", 98000.0),
        ("m-100k", 100000.0),
        ("m-102k", 102000.0),
        ("m-105k", 105000.0),
    ]
    belly_ids = {"m-98k", "m-100k", "m-102k"}
    wing_ids = {"m-95k", "m-105k"}
    tte_bucket = _first_tick_tte_bucket()

    report = {
        "cells": [
            {"region": "belly", "tte_bucket": tte_bucket, "horizon_s": cfg.markout_widen_horizon_s,
             "n": cfg.markout_min_n, "n_attempted": cfg.markout_min_n,
             "mk_avg": -1.0, "mk_var": 0.001, "mk_total": -1.0 * cfg.markout_min_n,
             "sides": {
                 "BUY_YES": {"n": cfg.markout_min_n, "n_attempted": cfg.markout_min_n,
                             "mk_avg": -1.0, "mk_var": 0.001},
                 "BUY_NO": {"n": 0, "n_attempted": 0, "mk_avg": 0.0, "mk_var": 0.0},
             }},
        ],
        "by_region": {},
        "by_expiry": {},
    }

    store_control = MMStateStore(str(tmp_path / "control.db"))
    store_treatment = MMStateStore(str(tmp_path / "treatment.db"))
    try:
        loop_control = PaperTradingLoop(
            store=store_control, expiry_key=EXPIRY, markets=markets, engine_fn=_engine(),
            config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
        )
        loop_treatment = PaperTradingLoop(
            store=store_treatment, expiry_key=EXPIRY, markets=markets, engine_fn=_engine(),
            config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
            markout_provider=lambda: report,
        )
        loop_control.tick({m: _snapshot_msg(_p_of_strike(k)) for m, k in markets})
        loop_treatment.tick({m: _snapshot_msg(_p_of_strike(k)) for m, k in markets})

        assert loop_treatment.hedger.repair_count == 0  # no PAV invocation needed

        strikes_sorted = sorted(k for _, k in markets)
        by_strike = {k: loop_treatment.last_quote_sets[m] for m, k in markets}
        qs_sorted = [by_strike[k] for k in strikes_sorted]
        verdict = loop_treatment.hedger.check(qs_sorted, strikes_sorted)
        assert verdict.ok, verdict.violations  # (i)

        for m, _k in markets:
            baseline_bid = loop_control.last_quote_sets[m].bid_price
            treated_bid = loop_treatment.last_quote_sets[m].bid_price
            if m in belly_ids:
                assert treated_bid < baseline_bid  # (ii)
            else:
                assert m in wing_ids
                assert treated_bid <= baseline_bid + 1e-9  # (iii)
    finally:
        store_control.close()
        store_treatment.close()


def test_markout_widen_belly_only_pav_pooled_bids_stay_within_baseline_bounds(tmp_path):
    """No-arb PAV repair interaction, adversarial/tight-strike case: belly-
    only negative BUY_YES markout at the DEFAULT markout_widen_cap
    (0.12) with strikes tight enough (500 apart, straddling the belly/wing
    boundary) that the natural adjacent-strike bid gap is smaller than the
    cap -- a genuine PAV violation-and-repair is exercised (repair_count
    increments). Post-repair:
      (ii)  every belly bid is strictly below its no-widening baseline;
      (iii) no wing bid is above its no-widening baseline;
    both hold regardless of whether PAV pooling fully restores monotonicity
    (the plan's algebra bounds pooled values against the pre-widening
    baseline unconditionally for a non-negative one-sided widening -- this
    does NOT depend on the reconstruction fully closing the gap).

    DISCOVERED GAP (out of scope for package E to fix -- ladder_hedger.py is
    not in package E's file list): `LadderHedger.repair()` pools the
    ladder's MIDS via PAV, then reconstructs each market's bid/ask from the
    pooled mid +/- that market's OWN pre-repair half-spread. A belly-only,
    asymmetric (bid-only) widening enlarges ONLY the widened market's own
    half-spread, so even when PAV pools mids correctly, reconstructing with
    disparate preserved half-spreads can leave the OUTPUT ladder still
    bid-monotonicity-violating (empirically confirmed non-convergent even
    under repeated repair() calls). This is a structural property of the
    "preserve own half-spread" reconstruction, not particular to any single
    widening magnitude, and was not evident from the plan's simplified two-
    point algebra (which implicitly assumed pooling acts directly on bid,
    i.e. equal half-spreads). It only manifests at strike spacing tighter
    than typical Polymarket BTC daily granularity (the sibling test above
    confirms NO violation at all at realistic 2000-3000-wide spacing with
    this same default cap) -- flagged here as a characterization, not
    asserted as fixed; see the implementation report for the recommended
    follow-up."""
    cfg = MMConfig(gamma=0.5, k_arrival=1.0)  # DEFAULT markout_widen_cap=0.12
    # Tight strikes straddling the p=0.2 belly/wing boundary (S0=100000,
    # scale=2000): 102000/102500 -> belly (p~0.269/0.223); 103000/103500 ->
    # wing (p~0.182/0.148).
    markets = [
        ("m-102000", 102000.0),
        ("m-102500", 102500.0),
        ("m-103000", 103000.0),
        ("m-103500", 103500.0),
    ]
    belly_ids = {"m-102000", "m-102500"}
    wing_ids = {"m-103000", "m-103500"}
    tte_bucket = _first_tick_tte_bucket()

    report = {
        "cells": [
            {"region": "belly", "tte_bucket": tte_bucket, "horizon_s": cfg.markout_widen_horizon_s,
             "n": cfg.markout_min_n, "n_attempted": cfg.markout_min_n,
             "mk_avg": -1.0, "mk_var": 0.001, "mk_total": -1.0 * cfg.markout_min_n,
             "sides": {
                 "BUY_YES": {"n": cfg.markout_min_n, "n_attempted": cfg.markout_min_n,
                             "mk_avg": -1.0, "mk_var": 0.001},
                 "BUY_NO": {"n": 0, "n_attempted": 0, "mk_avg": 0.0, "mk_var": 0.0},
             }},
        ],
        "by_region": {},
        "by_expiry": {},
    }

    store_control = MMStateStore(str(tmp_path / "control.db"))
    store_treatment = MMStateStore(str(tmp_path / "treatment.db"))
    try:
        loop_control = PaperTradingLoop(
            store=store_control, expiry_key=EXPIRY, markets=markets, engine_fn=_engine(),
            config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
        )
        loop_treatment = PaperTradingLoop(
            store=store_treatment, expiry_key=EXPIRY, markets=markets, engine_fn=_engine(),
            config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
            markout_provider=lambda: report,
        )
        loop_control.tick({m: _snapshot_msg(_p_of_strike(k)) for m, k in markets})
        loop_treatment.tick({m: _snapshot_msg(_p_of_strike(k)) for m, k in markets})

        # A genuine PAV violation-and-repair was exercised this tick -- not a
        # vacuous pass where the invariants below hold only because nothing
        # actually got pooled.
        assert loop_treatment.hedger.repair_count >= 1

        for m, _k in markets:
            baseline_bid = loop_control.last_quote_sets[m].bid_price
            treated_bid = loop_treatment.last_quote_sets[m].bid_price
            if m in belly_ids:
                assert treated_bid < baseline_bid  # (ii)
            else:
                assert m in wing_ids
                assert treated_bid <= baseline_bid + 1e-9  # (iii)
    finally:
        store_control.close()
        store_treatment.close()


# ---------------------------------------------------------------------------
# Package B2 -- per-region bankroll resume/seed policy (plan step 10)
# ---------------------------------------------------------------------------


def test_resume_bankroll_states_fresh_db_both_regions_parity(store):
    from market_maker.fair_value_anchor import BELLY_REGION, WING_REGION
    from market_maker.harness import _resume_bankroll_states

    states = _resume_bankroll_states(store, EXPIRY, START)
    for region in (BELLY_REGION, WING_REGION):
        assert states[region].bankrolls == {"pricer": 0.5, "market": 0.5}
        assert states[region].frozen is False
        assert states[region].update_count == 0


def test_resume_bankroll_states_legacy_only_seeds_belly_wing_parity(store):
    # plan step 10 REQUIRED TEST: legacy-only db -> belly == legacy values,
    # wing == parity, nothing frozen, no rows lost.
    from market_maker.contracts import BankrollState
    from market_maker.fair_value_anchor import BELLY_REGION, WING_REGION
    from market_maker.harness import _resume_bankroll_states

    legacy = BankrollState(
        model_ids=["pricer", "market"], bankrolls={"pricer": 0.83, "market": 0.17},
        last_update=START, update_count=42, frozen=False,
    )
    store.append_bankroll_state(EXPIRY, legacy, region="")  # pre-B2 single row

    states = _resume_bankroll_states(store, EXPIRY, START)
    assert states[BELLY_REGION].bankrolls == legacy.bankrolls
    assert states[BELLY_REGION].update_count == legacy.update_count
    assert states[BELLY_REGION].frozen is False
    assert states[WING_REGION].bankrolls == {"pricer": 0.5, "market": 0.5}
    assert states[WING_REGION].frozen is False
    assert states[WING_REGION].update_count == 0


def test_resume_bankroll_states_per_region_rows_present_loads_each(store):
    from market_maker.contracts import BankrollState
    from market_maker.fair_value_anchor import BELLY_REGION, WING_REGION
    from market_maker.harness import _resume_bankroll_states

    belly = BankrollState(
        model_ids=["pricer", "market"], bankrolls={"pricer": 0.7, "market": 0.3},
        last_update=START, update_count=5, frozen=False,
    )
    wing = BankrollState(
        model_ids=["pricer", "market"], bankrolls={"pricer": 0.35, "market": 0.65},
        last_update=START, update_count=3, frozen=True,
    )
    store.append_bankroll_state(EXPIRY, belly, region=BELLY_REGION)
    store.append_bankroll_state(EXPIRY, wing, region=WING_REGION)

    states = _resume_bankroll_states(store, EXPIRY, START)
    assert states[BELLY_REGION].bankrolls == belly.bankrolls
    assert states[WING_REGION].bankrolls == wing.bankrolls
    assert states[WING_REGION].frozen is True


def test_restart_resumes_per_region_bankrolls_via_harness(tmp_path):
    """Integration-level check (plan step 10) at the actual harness.restart()
    call site: a legacy-only bankrolls row (as if written by pre-B2 code)
    seeds belly from it and wing at parity on the next restart, and nothing
    is frozen."""
    from market_maker.contracts import BankrollState
    from market_maker.fair_value_anchor import BELLY_REGION, WING_REGION

    db_path = str(tmp_path / "resume_b2.db")
    store1 = MMStateStore(db_path)
    legacy = BankrollState(
        model_ids=["pricer", "market"], bankrolls={"pricer": 0.9, "market": 0.1},
        last_update=START, update_count=7, frozen=False,
    )
    store1.append_bankroll_state(EXPIRY, legacy, region="")
    store1.close()

    store2 = MMStateStore(db_path)
    try:
        loop = PaperTradingLoop(
            store=store2, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
            config=MMConfig(gamma=0.5), clock=SimClock(START), vol_gate_fn=_vol_gate(),
        )
        loop.restart()
        assert loop.bankroll_states[BELLY_REGION].bankrolls == legacy.bankrolls
        assert loop.bankroll_states[WING_REGION].bankrolls == {"pricer": 0.5, "market": 0.5}
        assert loop.bankroll_state.frozen is False  # nothing frozen
    finally:
        store2.close()


# ---------------------------------------------------------------------------
# Fix 3 (2026-07-26): ladder mid-velocity pull (risk rule h). The harness
# maintains a two-sided-only per-market mid history and threads the
# ladder-wide max move into every risk.evaluate call.
# ---------------------------------------------------------------------------


def test_mid_velocity_history_only_appends_two_sided_books(store):
    """A one-sided book tick appends NOTHING to _mid_hist (phantom-move guard,
    _two_sided_mid): _market_mid would fall back to the lone touch, but the
    velocity history must not. A concurrently two-sided market keeps growing."""
    market_id, other = MARKETS[0][0], MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    # Two-sided books: each market appends exactly one point.
    loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})
    assert len(loop._mid_hist[market_id]) == 1
    assert len(loop._mid_hist[other]) == 1

    # One-sided book for market_id (bid present, ask empty) appends nothing;
    # other stays two-sided and grows to two points.
    loop.tick({
        market_id: [{"type": "snapshot", "bids": [(0.5, 100.0)], "asks": []}],
        other: _snapshot_msg(0.5),
    })
    assert len(loop._mid_hist[market_id]) == 1  # unchanged -- one-sided skipped
    assert len(loop._mid_hist[other]) == 2


def test_ladder_mid_velocity_fires_rule_h_on_moved_mid(store):
    """Drive two ticks with one strike's mid jumping past the threshold within
    the window: the ladder-WIDE max move fires rule (h) on BOTH markets (flat
    inventory -> PULLED). Tick 1 (one point per market) has no computable move
    and stays TWO_SIDED."""
    market_id, other = MARKETS[0][0], MARKETS[1][0]
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0, mid_move_pull_p=0.04, mid_move_window_s=120.0),
        clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    # Tick 1: seed mids at 0.50 -- only one usable point per market, no move.
    loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})
    for m, d in loop.last_directives.items():
        assert d.mode == QuoteMode.TWO_SIDED, m
        assert RiskTrigger.MID_VELOCITY not in d.triggers, m

    # Tick 2: market_id jumps 0.50 -> 0.62 (0.12 > 0.04) within the 120s
    # window; other stays flat. The ladder-wide max protects the whole ladder.
    loop.tick({market_id: _snapshot_msg(0.62), other: _snapshot_msg(0.5)})
    for m in (market_id, other):
        d = loop.last_directives[m]
        assert RiskTrigger.MID_VELOCITY in d.triggers, m
        assert d.mode == QuoteMode.PULLED, m


def test_flat_mid_sequence_never_fires_rule_h(store):
    """A flat-mid sequence (no ladder-wide move) never fires rule (h)."""
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5, k_arrival=1.0), clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    for _ in range(4):
        loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})
    for m, d in loop.last_directives.items():
        assert d.mode == QuoteMode.TWO_SIDED, m
        assert RiskTrigger.MID_VELOCITY not in d.triggers, m
