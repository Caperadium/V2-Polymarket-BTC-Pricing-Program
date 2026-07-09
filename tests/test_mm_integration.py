"""Integration tests for the market-making paper-trading loop (plan G1 / 6.2).

Scripted synthetic feeds only -- no live data, no backtest replay. Every
scenario is deterministic (fixed clock, scripted books/prints) and the key
assertions (fills, settlement PnL) are hand-computed.
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

from dataclasses import replace

from market_maker.config import MMConfig
from market_maker.contracts import QuoteMode, Side
from market_maker.harness import PaperTradingLoop
from market_maker.ladder_hedger import LadderHedger
from market_maker.order_lifecycle import SimClock
from market_maker.settlement_handler import BTCDataProvider, settlement_instant_utc
from market_maker.state_store import MMStateStore

# 5-day-out expiry so tte keeps quoting alive (near-resolution pull is < 1 day).
EXPIRY = "2026-07-06"
START = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
S0 = 100000.0
STRIKES = [98000.0, 100000.0, 102000.0]
ATM = 100000.0
MARKETS = [("m-98k", 98000.0), ("m-100k", 100000.0), ("m-102k", 102000.0)]


# ---------------------------------------------------------------------------
# scripted stubs
# ---------------------------------------------------------------------------


def _engine(s0=S0, scale=2000.0, n_sims=15000):
    """Monotone-decreasing P(S_T >= K) logistic pricer stub. Covers any strike
    grid the adapter asks for (quoted + densified midpoints)."""
    def fn(strikes, hours_to_expiry, **kwargs):
        out = {float(k): float(1.0 / (1.0 + np.exp((float(k) - s0) / scale))) for k in strikes}
        out["_meta"] = {"n_sims": n_sims, "S0": s0, "horizon_gate_active": False}
        return out
    return fn


def _p_of(k, s0=S0, scale=2000.0):
    return 1.0 / (1.0 + np.exp((float(k) - s0) / scale))


class _VG:
    def __init__(self, regime="normal", shock=False, kelly_mult=1.0, edge_add_cents=0.0):
        self.regime = regime
        self.shock = shock
        self.kelly_mult = kelly_mult
        self.edge_add_cents = edge_add_cents


def _vol_gate(regime="normal", shock=False):
    return lambda: _VG(regime=regime, shock=shock)


def _snapshot_msg(p, prints=None, mid_equals_p=False):
    """One book snapshot bracketing the pricer probability, with optional trade
    prints. mid_equals_p makes the market mid land exactly on p (used by the
    stability test so consensus is perfectly constant)."""
    bid = round(max(0.01, p - 0.03), 4)
    ask = round(min(0.99, p + 0.03), 4)
    if mid_equals_p:
        bid = max(0.001, p - 0.03)
        ask = min(0.999, p + 0.03)
    msgs = [{
        "type": "snapshot",
        "bids": [(bid, 100.0), (round(bid - 0.01, 4), 100.0)],
        "asks": [(ask, 100.0), (round(ask + 0.01, 4), 100.0)],
    }]
    for pr in (prints or []):
        msgs.append({"type": "trade", "price": pr[0], "size": pr[1]})
    return msgs


def _static_books(prints_by_market=None, mid_equals_p=False):
    prints_by_market = prints_by_market or {}
    out = {}
    for m, k in MARKETS:
        out[m] = _snapshot_msg(_p_of(k), prints_by_market.get(m), mid_equals_p)
    return out


@pytest.fixture
def store(tmp_path):
    s = MMStateStore(str(tmp_path / "mm.db"))
    yield s
    s.close()


def _make_loop(store, *, config=None, clock=None, bankroll=1000.0,
               data_provider=None, hedger_mode="repair", quote_variant="dalen"):
    return PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=config or MMConfig(), clock=clock or SimClock(START),
        vol_gate_fn=_vol_gate(), data_provider=data_provider, bankroll=bankroll,
        hedger_mode=hedger_mode, quote_variant=quote_variant,
    )


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_full_loop(store):
    cfg = MMConfig(gamma=0.5)
    loop = _make_loop(store, config=cfg)

    # Warm up: two static ticks establish resting orders (placement latency 2s
    # means orders from tick t become fillable at tick t+1).
    for _ in range(2):
        loop.tick(_static_books())
        assert loop.fold_matches_inventory()

    # Every emitted QuoteSet ladder must pass the no-arb check.
    for strikes, ladder in loop.checked_ladders:
        assert LadderHedger(config=cfg).check(ladder, strikes).ok

    # Orders reached the venue/store.
    live = [o for o in store.get_all_orders() if o.status in ("PENDING", "LIVE")]
    assert live

    # r_x of the ATM market before the fill (q == 0).
    r_x_before = loop.last_proposals["m-100k"].r_x
    assert loop.inv.snapshot(loop.clock.now()).per_contract["m-100k"].q == pytest.approx(0.0)

    # A scripted aggressor print sweeps through our ATM bid -> a PaperFill.
    loop.tick(_static_books(prints_by_market={"m-100k": [(0.05, 500.0)]}))
    assert len(loop.last_fills) >= 1
    fill = [f for f in loop.last_fills if f.market_id == "m-100k"][0]
    assert fill.side == Side.BUY_YES
    assert fill.size > 0.0

    q_after = loop.inv.snapshot(loop.clock.now()).per_contract["m-100k"].q
    assert q_after > 0.0  # inventory updated
    assert loop.fold_matches_inventory()  # 8.2 invariant after the fill

    # Next tick: the long inventory skews the ATM quote center DOWN.
    loop.tick(_static_books())
    r_x_after = loop.last_proposals["m-100k"].r_x
    assert loop.last_proposals["m-100k"].skew_x < 0.0
    assert r_x_after < r_x_before  # reservation shifted down after the long fill

    # invariant still holds at the end
    assert loop.fold_matches_inventory()


# ---------------------------------------------------------------------------
# 2. Closed-loop stability (static book, static pricer)
# ---------------------------------------------------------------------------


def test_closed_loop_stability_no_oscillation(store):
    loop = _make_loop(store)
    order_counts = []
    price_snaps = []
    for _ in range(20):
        loop.tick(_static_books(mid_equals_p=True))
        order_counts.append(len(store.get_all_orders()))
        price_snaps.append({m: (qs.bid_price, qs.ask_price) for m, qs in loop.last_quote_sets.items()})

    # After tick 5, re-quote actions stop: order count is frozen and quoted
    # prices are constant (churn tolerance holds, consensus is stationary).
    assert order_counts[5:] == [order_counts[5]] * len(order_counts[5:])
    for snap in price_snaps[5:]:
        assert snap == price_snaps[5]

    # And no self-inflicted no-arb violation ever.
    for strikes, ladder in loop.checked_ladders:
        assert LadderHedger(config=loop.config).check(ladder, strikes).ok


# ---------------------------------------------------------------------------
# 3. Fault injection -- feed gap
# ---------------------------------------------------------------------------


def test_feed_gap_pulls_and_recovers(store):
    loop = _make_loop(store)

    # Warm up with live orders.
    for _ in range(3):
        loop.tick(_static_books())
    live_before = [o for o in store.get_all_orders() if o.status in ("PENDING", "LIVE")]
    assert live_before

    fills_before = len(store.get_fills())

    # Feed goes unhealthy mid-run.
    loop.tick(_static_books(), feed_healthy=False)
    d = loop.last_directives["m-100k"]
    assert d.mode == QuoteMode.PULLED
    assert d.cancel_all is True
    assert loop.last_fills == []  # no fills inside the gap
    # lifecycle cancelled every resting order
    live_during = [o for o in store.get_all_orders() if o.status in ("PENDING", "LIVE")]
    assert live_during == []
    assert len(store.get_fills()) == fills_before  # still no fills

    # Recover: healthy feed + advance past the hysteresis latch (2 ticks).
    loop.tick(_static_books())
    loop.tick(_static_books())
    assert loop.last_directives["m-100k"].mode == QuoteMode.TWO_SIDED
    live_after = [o for o in store.get_all_orders() if o.status in ("PENDING", "LIVE")]
    assert live_after  # quoting resumed

    # An exposure incident was recorded by the fill sim.
    incidents = loop.fill_sim.exposure_incidents()
    assert len(incidents) >= 1


# ---------------------------------------------------------------------------
# 4. Fault injection -- pricer failure -> stale -> widen-then-pull
# ---------------------------------------------------------------------------


def test_pricer_failure_reuses_snapshot_then_widens_then_pulls(store):
    cfg = MMConfig(pricer_max_age_s=70.0)  # tick_dt 60s -> widen ~2 ticks, pull ~3
    calls = {"n": 0}

    def flaky(strikes, hours, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("engine boom")
        out = {float(k): _p_of(k) for k in strikes}
        out["_meta"] = {"n_sims": 15000, "S0": S0, "horizon_gate_active": False}
        return out

    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=flaky,
        config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )

    loop.tick(_static_books())  # success (call 1)
    loop.tick(_static_books())  # success (call 2)
    prev_snap = loop.last_snapshot  # last good snapshot -> reused while flaky

    saw_widen = False
    saw_pull = False
    for _ in range(4):
        loop.tick(_static_books())  # engine raises now
        assert loop.snapshot_failed  # reused the previous snapshot
        assert loop.last_snapshot is prev_snap
        d = loop.last_directives["m-100k"]
        from market_maker.contracts import RiskTrigger
        if RiskTrigger.PRICER_STALE in d.triggers and d.eps_add > 0.0 and d.mode != QuoteMode.PULLED:
            saw_widen = True
        if d.mode == QuoteMode.PULLED and RiskTrigger.PRICER_STALE in d.triggers:
            saw_pull = True

    assert saw_widen  # widen path fired first
    assert saw_pull   # then pull on staleness beyond 2x max age


# ---------------------------------------------------------------------------
# 5. Fault injection -- forced no-arb violation
# ---------------------------------------------------------------------------


def _crossing_patch(loop):
    orig = loop._compose_quote_sets

    def patched(snap, fv, directives, liquidity=None):
        composed = orig(snap, fv, directives, liquidity=liquidity)  # keeps last_proposals populated
        composed.sort(key=lambda t: t[0])
        out = []
        for i, (k, m, qs) in enumerate(composed):
            bid = 0.20 + 0.30 * i  # ascending in strike -> monotonicity violation
            ask = bid + 0.05
            out.append((k, m, replace(qs, bid_price=bid, ask_price=ask)))
        return out

    loop._compose_quote_sets = patched


def test_forced_noarb_repair_yields_monotone_ladder(store):
    loop = _make_loop(store, hedger_mode="repair")
    _crossing_patch(loop)
    loop.tick(_static_books())

    checked = loop.last_checked_quote_sets
    assert checked is not None
    strikes, ladder = loop.checked_ladders[-1]
    # Repaired ladder is monotone before any order goes out.
    assert LadderHedger(config=loop.config).check(ladder, strikes).ok
    assert any(e["event"] == "repair" for e in loop.hedger.journal)


def test_forced_noarb_reject_blocks_orders(store):
    loop = _make_loop(store, hedger_mode="reject")
    _crossing_patch(loop)
    orders_before = len(store.get_all_orders())
    loop.tick(_static_books())

    assert loop.last_checked_quote_sets is None  # nothing reached the lifecycle
    assert len(store.get_all_orders()) == orders_before
    assert any(e["event"] == "reject" for e in loop.hedger.journal)


# ---------------------------------------------------------------------------
# 6. Inventory cap loop
# ---------------------------------------------------------------------------


def test_inventory_cap_goes_one_sided_then_pulls(store):
    # Small q_max so a few fills breach it. Large bankroll -> large resting bid;
    # SMALL prints (3 shares/tick) PARTIALLY fill it so the same bid stays live
    # every tick. The ratio then steps through the (1,1.5] one-sided band and,
    # via a cancel-window fill, into the >1.5 pull band.
    cfg = MMConfig(q_max_scale=20.0)  # q_max(ATM) = 20 * 0.25 = 5
    loop = _make_loop(store, config=cfg, bankroll=5000.0)

    modes = []
    fill_counts = []
    for _ in range(12):
        loop.tick(_static_books(prints_by_market={"m-100k": [(0.02, 3.0)]}))
        modes.append(loop.last_directives["m-100k"].mode)
        fill_counts.append(len(store.get_fills("m-100k")))
        assert loop.fold_matches_inventory()

    assert QuoteMode.ASK_ONLY in modes  # long breach -> quote asks only
    first_pull = next((i for i, m in enumerate(modes) if m == QuoteMode.PULLED), None)
    assert first_pull is not None
    assert modes.index(QuoteMode.ASK_ONLY) < first_pull  # one-sided BEFORE pull

    # After the pull fires, no further fills occur (orders cancelled).
    assert fill_counts[-1] == fill_counts[first_pull]


# ---------------------------------------------------------------------------
# 7. Settlement close-out
# ---------------------------------------------------------------------------


def _settlement_provider(spot=100500.0):
    settle = settlement_instant_utc(EXPIRY)
    idx = pd.to_datetime([
        settle - timedelta(minutes=2), settle, settle + timedelta(minutes=2),
    ])
    intraday = pd.DataFrame({"close": [spot - 100, spot, spot + 100]}, index=idx)
    return BTCDataProvider(intraday=intraday, daily=pd.DataFrame())


def test_settlement_closeout_and_pnl(store):
    loop = _make_loop(store, config=MMConfig(gamma=0.5), data_provider=_settlement_provider())

    # Build a long ATM position via bid fills.
    loop.tick(_static_books())
    for _ in range(3):
        loop.tick(_static_books(prints_by_market={"m-100k": [(0.02, 1000.0)]}))
    q0 = store.get_inventory("m-100k").q
    avg_cost0 = store.get_inventory("m-100k").avg_cost
    assert q0 > 0.0

    # Hand-computed settlement PnL from the fold's cost basis (payoff YES=1.0).
    expected_pnl = q0 * (1.0 - avg_cost0)

    loop.clock.set(settlement_instant_utc(EXPIRY) + timedelta(minutes=5))
    result = loop.settle()

    ev = [e for e in result.events if e.market_id == "m-100k"][0]
    assert ev.outcome.value == "YES"
    assert ev.q_settled == pytest.approx(q0)
    assert ev.pnl_realized == pytest.approx(expected_pnl)

    # SETTLEMENT pseudo-fill written; inventory zeroed; fold invariant holds.
    fills = store.get_fills("m-100k")
    assert fills[-1].liquidity.value == "SETTLEMENT"
    assert fills[-1].price == 1.0
    assert store.fold_fills_to_inventory()["m-100k"].q == pytest.approx(0.0)
    assert loop.fold_matches_inventory()

    # Terminal idempotency: a second settle is a no-op.
    result2 = loop.settle(now=loop.clock.now() + timedelta(minutes=1))
    assert all(e.market_id != "m-100k" or e.outcome.value == "YES" for e in result2.events)
    m100_events = [e for e in result2.events if e.market_id == "m-100k"]
    assert m100_events == []  # already terminal -> skipped
    assert len(store.get_fills("m-100k")) == len(fills)  # no duplicate pseudo-fill


# ---------------------------------------------------------------------------
# 8. Kill / restart
# ---------------------------------------------------------------------------


def test_kill_restart_reconciles(tmp_path):
    db = str(tmp_path / "restart.db")
    store1 = MMStateStore(db)
    loop1 = PaperTradingLoop(
        store=store1, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5), clock=SimClock(START), vol_gate_fn=_vol_gate(),
        data_provider=_settlement_provider(),
    )
    loop1.tick(_static_books())
    for _ in range(3):
        loop1.tick(_static_books(prints_by_market={"m-100k": [(0.02, 1000.0)]}))

    q_before = store1.fold_fills_to_inventory()["m-100k"].q
    bankrolls_before = dict(loop1.bankroll_state.bankrolls)
    assert q_before > 0.0
    live_coids = {o.client_order_id for o in store1.get_all_orders() if o.status in ("PENDING", "LIVE")}
    assert live_coids
    store1.close()

    # Restart on the same DB file with a fresh loop + store.
    store2 = MMStateStore(db)
    loop2 = PaperTradingLoop(
        store=store2, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(gamma=0.5),
        clock=SimClock(settlement_instant_utc(EXPIRY) + timedelta(hours=1)),
        vol_gate_fn=_vol_gate(), data_provider=_settlement_provider(),
    )
    # An orphan venue order the store never knew about.
    loop2.fill_sim.place("orphan-coid", "m-100k", "bid", 0.30, 5.0, loop2.clock.now())

    recon = loop2.restart()

    # Inventory + bankrolls survived the restart.
    assert store2.fold_fills_to_inventory()["m-100k"].q == pytest.approx(q_before)
    assert loop2.inv.snapshot(loop2.clock.now()).per_contract["m-100k"].q == pytest.approx(q_before)
    assert loop2.bankroll_state.bankrolls == bankrolls_before

    # Restart reconciliation cancelled the unknown resting orders and the orphan.
    assert set(recon.cancelled_unknown) == live_coids
    assert "orphan-coid" in recon.orphans_cancelled

    # Settlement catch-up settles the now-expired ladder before quoting resumes.
    result = loop2.settle(catch_up=True)
    settled = [e for e in result.events if e.market_id == "m-100k"]
    assert settled and settled[0].outcome.value == "YES"
    assert store2.fold_fills_to_inventory()["m-100k"].q == pytest.approx(0.0)
    assert loop2.fold_matches_inventory()
    store2.close()


# ---------------------------------------------------------------------------
# 9. Beuoy credibility motion (pricer consistently closer to realized mid)
# ---------------------------------------------------------------------------


def test_beuoy_credibility_rises_when_pricer_leads(store):
    loop = _make_loop(store)

    # The market mid lags: it starts biased and converges toward the pricer's
    # (fixed, correct) probabilities over the run. The pricer's prior forecast
    # therefore keeps matching the realized consensus better than the market's,
    # so its bankroll (credibility) grows.
    creds = []
    n = 12
    for i in range(n):
        frac = i / (n - 1)
        msgs = {}
        for m, k in MARKETS:
            p = _p_of(k)
            # market mid starts pulled toward 0.5, converges to p
            biased = 0.5 + (p - 0.5) * frac
            bid = max(0.001, biased - 0.02)
            ask = min(0.999, biased + 0.02)
            msgs[m] = [{
                "type": "snapshot",
                "bids": [(bid, 100.0), (round(bid - 0.01, 4), 100.0)],
                "asks": [(ask, 100.0), (round(ask + 0.01, 4), 100.0)],
            }]
        loop.tick(msgs)
        creds.append(loop.last_fair_value.credibility)

    # Credibility rises across the run (pricer out-predicts the lagging market).
    assert creds[-1] > creds[0]
    assert loop.last_fair_value.anchor_method.value == "BEUOY"
