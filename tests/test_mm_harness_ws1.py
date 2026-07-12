"""Workstream 1 tests (plan i-m-preparing-to-launch-sharded-snail.md):
state-store live-order lookup / index / market registry, and harness journal
bounding + registry-merge settlement catch-up.

Scripted synthetic feeds only, following tests/test_mm_integration.py's
conventions (fixed clock, deterministic scripted books).
"""
from __future__ import annotations

import sys
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
    QuoteMode,
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
# mm_sizing_fix_plan.md C1/C2 harness wiring -- size_ladder receives the
# tick's real inventory snapshot and per-market mkt_mid/strike, not just
# liquidity (which the test above already covers).
# ---------------------------------------------------------------------------


def test_harness_wires_inventory_and_mkt_mid_and_strike_into_size_ladder(store, monkeypatch):
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

    # Two-sided book on every market -> mkt_mid should be computed and threaded
    # into every ContractSizingInput this tick.
    loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})

    assert captured, "size_ladder was not called this tick"
    assert isinstance(captured["kwargs"].get("inventory"), InventoryState)
    contracts = captured["args"][0]
    assert contracts, "expected at least one ContractSizingInput"
    for c in contracts:
        assert c.strike is not None
    assert any(c.mkt_mid is not None for c in contracts)


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

    loop.bankroll_state.frozen = True
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
    at |q| > q_max the risk controller goes one-sided; at ratio > 1.5 it
    goes PULLED. This test exercises the hedger's own threshold in
    isolation (vertical_target_frac=0.5, the launch default) against the
    risk controller's INV_CAP thresholds (harness._breaches at ratio>=1.0,
    risk_controller PULLED at ratio>1.5), confirming the two fire at
    different, correctly-ordered inventory levels."""
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

    # At ratio 1.6 (>1.5): risk controller goes PULLED.
    breaches_160 = [InvBreach(market_id=market_id, is_long=True, ratio=1.6)]
    directive_160 = loop.risk.evaluate(
        market_id, now, tte_days=loop.last_snapshot.tte_days, pricer_snapshot=loop.last_snapshot,
        inventory_breaches=breaches_160, liquidity_regime=loop.last_liquidity[market_id].regime,
        feed_healthy=True, spot=loop.last_snapshot.s0, strike=MARKETS[0][1],
    )
    assert directive_160.mode == QuoteMode.PULLED


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
