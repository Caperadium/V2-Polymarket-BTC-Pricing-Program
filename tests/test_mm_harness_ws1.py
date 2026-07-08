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
from market_maker.contracts import ContractInv, SettlementEvent, SettlementOutcome, SpotSource
from market_maker.harness import PaperTradingLoop
from market_maker.order_lifecycle import SimClock
from market_maker.pnl_report import markout_report
from market_maker.settlement_handler import BTCDataProvider, settlement_instant_utc
from market_maker.state_store import MMStateStore

START = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
S0 = 100000.0
EXPIRY = "2026-07-06"
MARKETS = [("m-100k", 100000.0), ("m-102k", 102000.0)]


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
        config=MMConfig(gamma=0.5), clock=SimClock(START), vol_gate_fn=_vol_gate(),
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
        config=MMConfig(gamma=0.5), clock=SimClock(START), vol_gate_fn=_vol_gate(),
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
