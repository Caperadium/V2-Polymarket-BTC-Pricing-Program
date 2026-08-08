"""Multi-expiry orchestration tests (market_maker/multi_runner.py):
SharedPricingEngine token/stagger/shared-fit semantics, LadderSlot lifecycle
(warmup skip, settle-independent-of-grant, in-process rollover, settlement
timeout), the shared-db resume protocol (catch-up BEFORE filtered
resume_attach), the recurring orphan settlement catch-up, and the harness
additions (resume_attach, fold_matches_inventory(own_markets_only=True)).

Follows tests/test_mm_harness_ws1.py conventions: per-test tmp MMStateStore,
scripted engine/vol-gate, injected BTCDataProvider frames, no network.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import ContractInv, Fill, LiquiditySource, Side
from market_maker.harness import PaperTradingLoop
from market_maker.multi_runner import (
    MultiExpiryOrchestrator,
    SharedPricingEngine,
)
from market_maker.order_lifecycle import SimClock
from market_maker.settlement_handler import BTCDataProvider, settlement_instant_utc
from market_maker.state_store import MMStateStore

S0 = 100000.0
NOW0 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
TICK_S = 15.0

# Expiries relative to NOW0: A/B are live (5 and 6 days out), P is past.
EXPIRY_A = "2026-07-06"
EXPIRY_B = "2026-07-07"
EXPIRY_C = "2026-07-08"
EXPIRY_PAST = "2026-06-30"

LADDER_A = [("a-98k", 98000.0, "tok-a98"), ("a-102k", 102000.0, "tok-a102")]
LADDER_B = [("b-98k", 98000.0, "tok-b98"), ("b-102k", 102000.0, "tok-b102")]
LADDER_C = [("c-98k", 98000.0, "tok-c98"), ("c-102k", 102000.0, "tok-c102")]
LADDER_PAST = [("p-98k", 98000.0, "tok-p98"), ("p-102k", 102000.0, "tok-p102")]


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------


class _ScriptedCompute:
    """Scripted stand-in for calculate_probabilities: records every call's
    garch_cache identity and populates it (so the shared-fit bookkeeping in
    SharedPricingEngine sees a 'fitted' cache)."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fail = fail

    def __call__(self, strikes, hours_to_expiry, **kwargs):
        cache = kwargs.get("garch_cache")
        self.calls.append({
            "strikes": tuple(strikes), "hours": hours_to_expiry,
            "cache_id": id(cache) if cache is not None else None,
            "cache_was_empty": (not cache) if cache is not None else None,
            "jump_params": kwargs.get("jump_params"),
        })
        if self.fail:
            raise RuntimeError("synthetic pricing failure")
        if cache is not None and not cache:
            cache["fit"] = len(self.calls)  # "fit" the shared cache
        scale = 3000.0
        out = {float(k): float(1.0 / (1.0 + math.exp((float(k) - S0) / scale))) for k in strikes}
        out["_meta"] = {"n_sims": 1000, "S0": S0, "horizon_gate_active": False}
        return out


class _FakeAdapter:
    """Resends a full book snapshot on EVERY drain (unlike the runner tests'
    send-once fake) so a slot skipped during warmup still gets a live book on
    the first tick it actually runs."""

    instances: List["_FakeAdapter"] = []

    def __init__(self, tokens: Dict[str, str]) -> None:
        self.tokens = tokens
        self.started = False
        self.stopped = False
        self.drain_calls = 0
        _FakeAdapter.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self, join_timeout_s: float = 10.0) -> None:
        self.stopped = True

    def healthy(self) -> bool:
        return True

    def drain(self) -> Dict[str, List[Dict[str, object]]]:
        self.drain_calls += 1
        msg = [{
            "type": "snapshot",
            "bids": [(0.45, 100.0), (0.44, 100.0)],
            "asks": [(0.55, 100.0), (0.56, 100.0)],
        }]
        return {slug: list(msg) for slug in self.tokens}


class _VG:
    regime = "normal"
    shock = False
    kelly_mult = 1.0
    edge_add_cents = 0.0


def _vol_gate():
    return _VG()


def _provider_covering(*expiry_keys: str, spot: float = 101000.0) -> BTCDataProvider:
    rows = []
    for ek in expiry_keys:
        dt = settlement_instant_utc(ek)
        rows.extend([dt - timedelta(minutes=2), dt, dt + timedelta(minutes=2)])
    idx = pd.DatetimeIndex(sorted(rows), tz="UTC") if rows else pd.DatetimeIndex([], tz="UTC")
    intraday = pd.DataFrame({"close": [spot] * len(idx)}, index=idx)
    return BTCDataProvider(intraday=intraday, daily=pd.DataFrame())


def _empty_provider() -> BTCDataProvider:
    return BTCDataProvider(intraday=pd.DataFrame({"close": []}), daily=pd.DataFrame())


def _mk_orch(
    store: MMStateStore,
    compute: Optional[_ScriptedCompute] = None,
    *,
    max_expiries: int = 2,
    bankroll_total: float = 1000.0,
    auto_mode: bool = False,
    resolver=None,
    data_provider: Optional[BTCDataProvider] = None,
    max_settlement_wait_h: float = 26.0,
    reprice_s: float = 0.0,
    jump_loader=None,
    markout_provider=None,
    sizing_markout_provider=None,
) -> Tuple[MultiExpiryOrchestrator, SharedPricingEngine, _ScriptedCompute]:
    compute = compute or _ScriptedCompute()
    engine = SharedPricingEngine(reprice_s=reprice_s, compute_fn=compute, jump_loader=jump_loader)
    orch = MultiExpiryOrchestrator(
        store=store,
        engine=engine,
        config=MMConfig(),
        bankroll_total=bankroll_total,
        max_expiries=max_expiries,
        tick_s=TICK_S,
        vol_gate_fn=_vol_gate,
        data_provider=data_provider or _empty_provider(),
        markout_provider=markout_provider,
        sizing_markout_provider=sizing_markout_provider,
        adapter_factory=lambda tokens: _FakeAdapter(tokens),
        resolver=resolver,
        auto_mode=auto_mode,
        max_settlement_wait_h=max_settlement_wait_h,
    )
    return orch, engine, compute


@pytest.fixture
def store(tmp_path):
    s = MMStateStore(str(tmp_path / "mm.db"))
    yield s
    s.close()


def _ticks(orch: MultiExpiryOrchestrator, n: int, start: datetime = NOW0):
    reports = []
    for i in range(n):
        reports.append(orch.tick(start + timedelta(seconds=TICK_S * i)))
    return reports


# ---------------------------------------------------------------------------
# SharedPricingEngine
# ---------------------------------------------------------------------------


def test_shared_garch_fit_and_jump_loader_across_views():
    loads = {"n": 0}

    def _jump_loader():
        loads["n"] += 1
        return {"lambda": 0.5}

    compute = _ScriptedCompute()
    engine = SharedPricingEngine(reprice_s=0.0, compute_fn=compute, jump_loader=_jump_loader)

    va = engine.view("2026-07-06")
    vb = engine.view("2026-07-07")

    engine.begin_tick()
    va([98000.0, 102000.0], 120.0)
    engine.begin_tick()
    vb([99000.0, 101000.0], 144.0)

    assert len(compute.calls) == 2
    # SAME shared garch_cache object across views; only the FIRST compute saw
    # it empty (the fit ran once, the second call reused it).
    assert compute.calls[0]["cache_id"] == compute.calls[1]["cache_id"]
    assert compute.calls[0]["cache_was_empty"] is True
    assert compute.calls[1]["cache_was_empty"] is False
    # jump params loaded once (on the empty-cache compute) and passed to both.
    assert loads["n"] == 1
    assert compute.calls[1]["jump_params"] == {"lambda": 0.5}
    # per-view ladder caches are distinct
    assert va.has_cache() and vb.has_cache()


def test_garch_refit_clears_shared_cache_and_reloads_jumps():
    loads = {"n": 0}

    def _jump_loader():
        loads["n"] += 1
        return None

    compute = _ScriptedCompute()
    engine = SharedPricingEngine(
        reprice_s=0.0, garch_refit_s=0.0, compute_fn=compute, jump_loader=_jump_loader
    )
    v = engine.view("2026-07-06")
    engine.begin_tick()
    v([98000.0], 120.0)
    engine.begin_tick()
    v([98000.0], 119.0)
    # refit_s=0 -> the second compute saw the cache cleared again.
    assert compute.calls[1]["cache_was_empty"] is True
    assert loads["n"] == 2


def test_one_reprice_token_per_tick_stale_cache_served():
    compute = _ScriptedCompute()
    engine = SharedPricingEngine(reprice_s=0.0, compute_fn=compute)
    va = engine.view("2026-07-06")
    vb = engine.view("2026-07-07")

    # Warm both caches (one grant per tick).
    engine.begin_tick()
    va([98000.0], 120.0)
    engine.begin_tick()
    vb([98000.0], 144.0)
    assert len(compute.calls) == 2

    # Both due (reprice_s=0): only the first caller recomputes; the second
    # returns its stale cache without a compute.
    engine.begin_tick()
    va([98000.0], 119.0)
    res_b = vb([98000.0], 143.0)
    assert len(compute.calls) == 3
    assert 98000.0 in res_b  # stale-but-usable ladder served


def test_failed_compute_returns_token_to_siblings():
    good = _ScriptedCompute()
    engine = SharedPricingEngine(reprice_s=0.0, compute_fn=good)
    va = engine.view("2026-07-06")
    vb = engine.view("2026-07-07")

    engine.begin_tick()
    good.fail = True
    with pytest.raises(RuntimeError):
        va([98000.0], 120.0)
    # Token was returned: vb can still take this tick's grant.
    good.fail = False
    res = vb([98000.0], 144.0)
    assert 98000.0 in res
    assert not va.has_cache() and vb.has_cache()


def test_uncached_view_without_token_raises_defensively():
    compute = _ScriptedCompute()
    engine = SharedPricingEngine(reprice_s=0.0, compute_fn=compute)
    va = engine.view("2026-07-06")
    vb = engine.view("2026-07-07")
    engine.begin_tick()
    va([98000.0], 120.0)  # takes the grant
    with pytest.raises(RuntimeError):
        vb([98000.0], 144.0)  # uncached + no token -> defensive raise


# ---------------------------------------------------------------------------
# Orchestrator: warmup stagger + skipped-slot semantics
# ---------------------------------------------------------------------------


def test_startup_stagger_one_first_price_per_tick(store):
    _FakeAdapter.instances = []
    orch, engine, compute = _mk_orch(store)
    orch.startup(NOW0, [("ev-a", EXPIRY_A, LADDER_A), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)

    r1 = orch.tick(NOW0)
    assert len(compute.calls) == 1
    ticked = [r for r in r1.slot_reports if r.ticked]
    skipped = [r for r in r1.slot_reports if r.skipped_warmup]
    assert len(ticked) == 1 and len(skipped) == 1
    # skipped slot emitted no quotes and has no live orders
    skipped_slot = skipped[0].slot
    assert not skipped_slot.loop.last_quote_sets
    for m, _k in skipped_slot.markets:
        assert store.get_live_orders(m) == []
    # its adapter was still drained (drain-and-discard, unbounded buffer)
    assert skipped_slot.adapter.drain_calls == 1

    r2 = orch.tick(NOW0 + timedelta(seconds=TICK_S))
    assert len(compute.calls) == 2  # exactly one compute per tick
    assert all(r.ticked for r in r2.slot_reports)
    # both slots quoting now
    for slot in orch.slots.values():
        assert slot.loop.last_quote_sets
        assert slot.state == "active"


def test_markout_provider_threaded_into_every_slot_loop(store):
    # wave 2 W7: ONE shared markout_provider reaches every slot's
    # PaperTradingLoop via _build_slot, the single construction point -- both
    # startup slots and (separately asserted below) a mid-run acquisition.
    stub_provider = lambda: {"stub": True}
    orch, engine, compute = _mk_orch(store, markout_provider=stub_provider)
    orch.startup(NOW0, [("ev-a", EXPIRY_A, LADDER_A), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)
    assert len(orch.slots) == 2
    for slot in orch.slots.values():
        assert slot.loop.markout_provider is stub_provider


def test_markout_provider_none_default_keeps_existing_constructors_green(store):
    # Default (no markout_provider passed) must not break any existing
    # MultiExpiryOrchestrator caller -- every slot's loop gets None, same as
    # an unwired single-expiry PaperTradingLoop.
    orch, engine, compute = _mk_orch(store)
    orch.startup(NOW0, [("ev-a", EXPIRY_A, LADDER_A)], db_existed=False)
    for slot in orch.slots.values():
        assert slot.loop.markout_provider is None
        assert slot.loop.sizing_markout_provider is None


def test_sizing_markout_provider_threaded_into_every_slot_loop(store):
    # Fix 3 (2026-08-08 wing-bleed fix, 3d): the belly epoch-filtered sizing
    # provider is threaded through _build_slot exactly like markout_provider
    # (single construction point -- covers startup slots AND mid-run
    # acquisitions by the same code path).
    stub_full = lambda: {"stub": "full"}
    stub_sizing = lambda: {"stub": "sizing"}
    orch, engine, compute = _mk_orch(
        store, markout_provider=stub_full, sizing_markout_provider=stub_sizing)
    orch.startup(NOW0, [("ev-a", EXPIRY_A, LADDER_A), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)
    assert len(orch.slots) == 2
    for slot in orch.slots.values():
        assert slot.loop.markout_provider is stub_full
        assert slot.loop.sizing_markout_provider is stub_sizing


def test_reprice_grant_rotates_round_robin(store):
    orch, engine, compute = _mk_orch(store)
    orch.startup(NOW0, [("ev-a", EXPIRY_A, LADDER_A), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)
    _ticks(orch, 4)
    assert len(compute.calls) == 4  # one per tick with reprice_s=0
    # strikes tuples differ per ladder only by market ids, both 98k/102k --
    # use hours to distinguish expiries: A (5d) < B (6d) hours.
    hours = [c["hours"] for c in compute.calls]
    a_hours = [h for h in hours if h < 130.0]
    b_hours = [h for h in hours if h >= 130.0]
    # both expiries got grants (rotation), interleaved
    assert len(a_hours) == 2 and len(b_hours) == 2


def test_bankroll_static_split(store):
    orch, _e, _c = _mk_orch(store, max_expiries=3, bankroll_total=900.0)
    orch.startup(NOW0, [("ev-a", EXPIRY_A, LADDER_A), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)
    # Fixed share total/max_expiries regardless of active count (2 of 3).
    assert orch.bankroll_share == pytest.approx(300.0)
    for slot in orch.slots.values():
        assert slot.loop.bankroll == pytest.approx(300.0)


def test_fill_isolation_and_scoped_fold(store):
    orch, _e, _c = _mk_orch(store)
    orch.startup(NOW0, [("ev-a", EXPIRY_A, LADDER_A), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)
    _ticks(orch, 2)

    slot_a = orch.slots[EXPIRY_A]
    slot_b = orch.slots[EXPIRY_B]
    fill = Fill(ts=NOW0, market_id="a-98k", order_id="f-1", side=Side.BUY_YES,
                price=0.40, size=5.0, liquidity=LiquiditySource.MAKER, venue_ts=NOW0)
    slot_a.loop._route_fill(fill, NOW0)

    # A holds the position; B's inventory never sees it.
    assert slot_a.loop.inv.snapshot(NOW0).per_contract["a-98k"].q == pytest.approx(5.0)
    assert "a-98k" not in slot_b.loop.inv.snapshot(NOW0).per_contract
    # Scoped fold holds per loop even though the store fold is global.
    assert slot_a.loop.fold_matches_inventory(own_markets_only=True)
    assert slot_b.loop.fold_matches_inventory(own_markets_only=True)
    # The GLOBAL legacy fold check would fail for B (foreign-market fill).
    assert not slot_b.loop.fold_matches_inventory()


def test_per_expiry_beuoy_bankroll_rows(store):
    orch, _e, _c = _mk_orch(store)
    orch.startup(NOW0, [("ev-a", EXPIRY_A, LADDER_A), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)
    _ticks(orch, 3)
    # Package B2: bankroll rows are per-region ("belly"/"wing"); the legacy
    # region='' row is never written by the harness anymore.
    assert store.get_latest_bankroll_state(EXPIRY_A, region="belly") is not None
    assert store.get_latest_bankroll_state(EXPIRY_A, region="wing") is not None
    assert store.get_latest_bankroll_state(EXPIRY_B, region="belly") is not None
    assert store.get_latest_bankroll_state(EXPIRY_B, region="wing") is not None


def test_past_instant_skipped_slot_still_settles(store):
    """A slot that never got the reprice grant (uncached, skipped) must still
    settle once past its instant -- settle is not gated on being priced."""
    now = settlement_instant_utc(EXPIRY_PAST) + timedelta(hours=1)
    provider = _provider_covering(EXPIRY_PAST)
    orch, engine, compute = _mk_orch(store, data_provider=provider)
    orch.startup(now, [("ev-a", EXPIRY_A, LADDER_A), ("ev-p", EXPIRY_PAST, LADDER_PAST)],
                 db_existed=False)
    # Force rotation so the LIVE slot (A) is processed first and takes the
    # grant; the past slot P is then uncached + grant-less -> skipped.
    orch._rr_offset = 1  # sorted order [P, A] rotated -> [A, P]
    r1 = orch.tick(now)
    p_rep = next(r for r in r1.slot_reports if r.slot.expiry_key == EXPIRY_PAST)
    assert p_rep.skipped_warmup and not p_rep.ticked
    for m, _k, _t in LADDER_PAST:
        ev = store.get_settlement(m, EXPIRY_PAST)
        assert ev is not None and ev.outcome.value in ("YES", "NO")


# ---------------------------------------------------------------------------
# Rollover / teardown / acquisition
# ---------------------------------------------------------------------------


def test_in_process_rollover_settled_ladder_replaced(store):
    _FakeAdapter.instances = []
    now = settlement_instant_utc(EXPIRY_PAST) + timedelta(hours=1)  # > 30min grace
    provider = _provider_covering(EXPIRY_PAST)
    resolver_calls: List[Tuple] = []

    def _resolver(now_arg, lead, cap, exclude):
        resolver_calls.append((cap, set(exclude)))
        return [("ev-c", EXPIRY_C, LADDER_C)]

    orch, engine, compute = _mk_orch(
        store, auto_mode=True, resolver=_resolver, data_provider=provider,
    )
    orch.startup(now, [("ev-p", EXPIRY_PAST, LADDER_PAST), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)
    slot_p = orch.slots[EXPIRY_PAST]
    slot_b = orch.slots[EXPIRY_B]
    b_bankroll_state = slot_b.loop.bankroll_state

    r1 = orch.tick(now)

    # P settled + torn down in-process; C acquired; B untouched; no exit.
    assert r1.exit_request is None
    assert (EXPIRY_PAST, "ladder_settled") in r1.teardowns
    assert EXPIRY_PAST not in orch.slots
    assert EXPIRY_C in orch.slots
    assert EXPIRY_B in orch.slots
    assert slot_p.adapter.stopped
    assert orch.ladders_settled_total == 1
    assert EXPIRY_PAST in orch.completed_expiries
    # resolver saw the exclusion set (active B + completed P)
    assert resolver_calls and EXPIRY_PAST in resolver_calls[0][1]
    assert EXPIRY_B in resolver_calls[0][1]
    # B's slot survived the rollover intact (same loop object, same ladder)
    assert orch.slots[EXPIRY_B].loop is slot_b.loop
    # P's live orders were cancelled (scoped)
    for m, _k, _t in LADDER_PAST:
        assert store.get_live_orders(m) == []


def test_markout_provider_threaded_into_mid_run_acquired_slot(store):
    # wave 2 W7 (reviewer Q6 item 5): _build_slot is the SINGLE construction
    # point covering both startup slots and mid-run acquisitions -- a slot
    # acquired via in-process rollover must get the same shared provider.
    _FakeAdapter.instances = []
    now = settlement_instant_utc(EXPIRY_PAST) + timedelta(hours=1)
    provider = _provider_covering(EXPIRY_PAST)
    stub_provider = lambda: {"stub": True}

    def _resolver(now_arg, lead, cap, exclude):
        return [("ev-c", EXPIRY_C, LADDER_C)]

    orch, engine, compute = _mk_orch(
        store, auto_mode=True, resolver=_resolver, data_provider=provider,
        markout_provider=stub_provider,
    )
    orch.startup(now, [("ev-p", EXPIRY_PAST, LADDER_PAST), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)
    orch.tick(now)  # P settles + torn down; C acquired in-process

    assert EXPIRY_C in orch.slots
    assert orch.slots[EXPIRY_C].loop.markout_provider is stub_provider
    assert orch.slots[EXPIRY_B].loop.markout_provider is stub_provider


def test_settlement_timeout_teardown_and_later_catchup_closes(store):
    """UNSETTLEABLE ladder times out, is torn down with its position still
    open, and a LATER catch-up pass (fresh BTC data) closes it."""
    now = settlement_instant_utc(EXPIRY_PAST) + timedelta(hours=1)

    def _resolver(now_arg, lead, cap, exclude):
        return []

    orch, engine, compute = _mk_orch(
        store, auto_mode=True, resolver=_resolver,
        data_provider=_empty_provider(), max_settlement_wait_h=0.5,
    )
    orch.startup(now, [("ev-p", EXPIRY_PAST, LADDER_PAST), ("ev-b", EXPIRY_B, LADDER_B)],
                 db_existed=False)
    # Open position on the past ladder, seeded through the fills channel.
    seed_ts = settlement_instant_utc(EXPIRY_PAST) - timedelta(days=1)
    orch.slots[EXPIRY_PAST].loop._route_fill(
        Fill(ts=seed_ts, market_id="p-98k", order_id="seed", side=Side.BUY_YES,
             price=0.40, size=5.0, liquidity=LiquiditySource.MAKER, venue_ts=seed_ts), now)

    r1 = orch.tick(now)
    assert (EXPIRY_PAST, "settlement_timeout") in r1.teardowns
    assert orch.ladder_settlement_timeouts == 1
    assert EXPIRY_PAST not in orch.slots
    # Position still open in the global fold (UNSETTLEABLE, not fake-settled).
    assert store.fold_fills_to_inventory()["p-98k"].q == pytest.approx(5.0)

    # BTC bar covering the instant lands later -> recurring catch-up settles.
    orch._catchup_handler.data = _provider_covering(EXPIRY_PAST)
    orch._last_catchup_wall = 0.0
    orch.settlement_catchup_pass(now + timedelta(minutes=5))
    assert store.fold_fills_to_inventory()["p-98k"].q == pytest.approx(0.0)
    ev = store.get_settlement("p-98k", EXPIRY_PAST)
    assert ev is not None and ev.outcome.value == "YES"


def test_orphan_retry_cadence_settles_without_teardown(store):
    """An orphaned (no-slot) UNSETTLEABLE registry market is re-driven by the
    recurring per-tick catch-up pass once data lands -- no teardown needed."""
    # Seed a previous-event market + open position directly in the store.
    store.upsert_market("p-98k", EXPIRY_PAST, 98000.0)
    seed_ts = settlement_instant_utc(EXPIRY_PAST) - timedelta(days=1)
    store.record_fill_and_update_inventory(
        Fill(ts=seed_ts, market_id="p-98k", order_id="seed", side=Side.BUY_YES,
             price=0.40, size=5.0, liquidity=LiquiditySource.MAKER, venue_ts=seed_ts),
        ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0),
    )

    now = settlement_instant_utc(EXPIRY_PAST) + timedelta(hours=1)
    orch, engine, compute = _mk_orch(store, data_provider=_empty_provider())
    orch.startup(now, [("ev-a", EXPIRY_A, LADDER_A)], db_existed=True)
    # Startup catch-up ran but had no data -> still open.
    assert store.fold_fills_to_inventory()["p-98k"].q == pytest.approx(5.0)

    # Data lands; next tick's recurring pass (throttle reset) settles it.
    orch._catchup_handler.data = _provider_covering(EXPIRY_PAST)
    orch._last_catchup_wall = 0.0
    orch.tick(now)
    assert store.fold_fills_to_inventory()["p-98k"].q == pytest.approx(0.0)


def test_acquisition_empty_keeps_running_and_backs_off(store):
    calls = {"n": 0}

    def _resolver(now_arg, lead, cap, exclude):
        calls["n"] += 1
        return []

    orch, _e, _c = _mk_orch(store, auto_mode=True, resolver=_resolver, max_expiries=2)
    orch.startup(NOW0, [("ev-a", EXPIRY_A, LADDER_A)], db_existed=False)
    r1 = orch.tick(NOW0)
    r2 = orch.tick(NOW0 + timedelta(seconds=TICK_S))
    assert r1.exit_request is None and r2.exit_request is None
    assert EXPIRY_A in orch.slots
    # empty result set the wall-clock backoff -> resolver called only once
    assert calls["n"] == 1


def test_zero_slots_and_empty_acquisition_requests_exit_42(store):
    def _resolver(now_arg, lead, cap, exclude):
        return []

    orch, _e, _c = _mk_orch(store, auto_mode=True, resolver=_resolver)
    orch.startup(NOW0, [], db_existed=False)
    r = orch.tick(NOW0)
    assert r.exit_request == "no_quotable_events"


def test_fixed_mode_terminal_requests_legacy_exit(store):
    now = settlement_instant_utc(EXPIRY_PAST) + timedelta(hours=1)
    orch, _e, _c = _mk_orch(store, auto_mode=False,
                            data_provider=_provider_covering(EXPIRY_PAST))
    orch.startup(now, [("ev-p", EXPIRY_PAST, LADDER_PAST)], db_existed=False)
    r = orch.tick(now)
    assert r.exit_request == "ladder_settled"
    assert EXPIRY_PAST in orch.slots  # fixed mode: NOT torn down


# ---------------------------------------------------------------------------
# Resume protocol on a shared multi-expiry db
# ---------------------------------------------------------------------------


def _seed_two_expiry_db(store: MMStateStore) -> None:
    """Fills for a PAST expiry (with a leftover LIVE order) and a LIVE expiry
    A, exactly the shape a crashed multi-expiry process leaves behind."""
    seed_ts = settlement_instant_utc(EXPIRY_PAST) - timedelta(days=1)
    store.upsert_market("p-98k", EXPIRY_PAST, 98000.0)
    store.upsert_order("old-order-1", "p-98k", Side.BUY_YES, 0.40, 5.0, "LIVE",
                       ts_placed=seed_ts)
    store.record_fill_and_update_inventory(
        Fill(ts=seed_ts, market_id="p-98k", order_id="old-fill", side=Side.BUY_YES,
             price=0.40, size=5.0, liquidity=LiquiditySource.MAKER, venue_ts=seed_ts),
        ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0),
    )
    a_ts = seed_ts + timedelta(hours=1)
    store.upsert_market("a-98k", EXPIRY_A, 98000.0)
    store.record_fill_and_update_inventory(
        Fill(ts=a_ts, market_id="a-98k", order_id="a-fill", side=Side.BUY_YES,
             price=0.30, size=3.0, liquidity=LiquiditySource.MAKER, venue_ts=a_ts),
        ContractInv(q=3.0, avg_cost=0.30, q_max=100.0, age_weighted_holding=0.0),
    )


def test_resume_partitions_fills_and_settles_expired_before_replay(store):
    _seed_two_expiry_db(store)
    now = settlement_instant_utc(EXPIRY_PAST) + timedelta(hours=2)
    orch, _e, _c = _mk_orch(
        store, data_provider=_provider_covering(EXPIRY_PAST),
    )
    recon = orch.startup(
        now, [("ev-a", EXPIRY_A, LADDER_A), ("ev-b", EXPIRY_B, LADDER_B)],
        db_existed=True,
    )

    # 1. standalone catch-up settled the expired position BEFORE replay
    ev = store.get_settlement("p-98k", EXPIRY_PAST)
    assert ev is not None and ev.outcome.value == "YES"
    assert store.fold_fills_to_inventory()["p-98k"].q == pytest.approx(0.0)

    # 2. filtered replay: A's loop holds ONLY a-98k; B starts flat; neither
    # ingested the previous event's fills.
    loop_a = orch.slots[EXPIRY_A].loop
    loop_b = orch.slots[EXPIRY_B].loop
    assert loop_a.inv.snapshot(now).per_contract["a-98k"].q == pytest.approx(3.0)
    assert "p-98k" not in loop_a.inv.snapshot(now).per_contract
    assert all(ci.q == 0.0 for ci in loop_b.inv.snapshot(now).per_contract.values())

    # 3. per-loop scoped fold holds through the whole sequence
    assert loop_a.fold_matches_inventory(own_markets_only=True)
    assert loop_b.fold_matches_inventory(own_markets_only=True)

    # 4. ONE venue reconcile: the stale LIVE order was cancelled, no
    # discrepancies (fold == store inventory globally).
    order = store.get_order("old-order-1")
    assert order is not None and order.status == "CANCELLED"
    assert recon is not None and not recon.position_discrepancies


def test_resume_tampered_inventory_reports_discrepancy(store):
    _seed_two_expiry_db(store)
    # Desync inventory table from fills fold for the LIVE expiry market.
    store.upsert_inventory(
        "a-98k", ContractInv(q=9.0, avg_cost=0.30, q_max=100.0, age_weighted_holding=0.0),
    )
    now = settlement_instant_utc(EXPIRY_PAST) + timedelta(hours=2)
    orch, _e, _c = _mk_orch(store, data_provider=_provider_covering(EXPIRY_PAST))
    recon = orch.startup(now, [("ev-a", EXPIRY_A, LADDER_A)], db_existed=True)
    assert recon is not None
    assert "a-98k" in recon.position_discrepancies


# ---------------------------------------------------------------------------
# resume_attach unit-level (harness addition)
# ---------------------------------------------------------------------------


def test_resume_attach_registers_markets_and_replays_filtered(store):
    _seed_two_expiry_db(store)
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY_A,
        markets=[(m, k) for m, k, _t in LADDER_A],
        engine_fn=_ScriptedCompute(), config=MMConfig(),
        clock=SimClock(NOW0), vol_gate_fn=_vol_gate,
        data_provider=_empty_provider(), bankroll=500.0, tick_dt_s=TICK_S,
    )
    replayed = loop.resume_attach(NOW0, store.get_fills())
    assert replayed == 1  # only a-98k's fill; p-98k filtered out

    snap = loop.inv.snapshot(NOW0)
    assert snap.per_contract["a-98k"].q == pytest.approx(3.0)
    assert "p-98k" not in snap.per_contract
    # full re-registration ran: unfilled own markets exist with q_max set
    assert snap.per_contract["a-102k"].q == pytest.approx(0.0)
    assert snap.per_contract["a-102k"].q_max > 0.0
