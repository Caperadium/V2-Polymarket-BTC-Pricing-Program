"""Workstream 2 tests (plan i-m-preparing-to-launch-sharded-snail.md, section
2.1-2.8): paper_runner.py resumable state, exit-after-settlement + rollover
exit codes, feed watchdog, tick-failure circuit breaker, BTC staleness guard.

Follows tests/test_mm_paper_runner_control.py's conventions: run() is driven
directly (in-process, via a background thread when a few real ticks must
elapse first) against temp control/out dirs, with resolve_event,
PolymarketFeedAdapter, CachedEngine and compute_vol_gate stubbed at the
paper_runner module seam.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker import paper_runner
from market_maker.contracts import ContractInv, Fill, LiquiditySource, Side
from market_maker.settlement_handler import BTCDataProvider, settlement_instant_utc
from market_maker.state_store import MMStateStore

S0 = 100000.0


# ---------------------------------------------------------------------------
# stubs (mirrors test_mm_paper_runner_control.py)
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """healthy() is patched per-instance by tests that need unhealthy runs;
    default behavior mirrors test_mm_paper_runner_control's _FakeAdapter."""

    def __init__(self, tokens: Dict[str, str]) -> None:
        self.tokens = tokens
        self._sent = False
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, join_timeout_s: float = 10.0) -> None:
        self.stopped = True

    def healthy(self) -> bool:
        return True

    def drain(self) -> Dict[str, List[Dict[str, object]]]:
        if not self._sent:
            self._sent = True
            msg = [{
                "type": "snapshot",
                "bids": [(0.45, 100.0), (0.44, 100.0)],
                "asks": [(0.55, 100.0), (0.56, 100.0)],
            }]
            return {slug: list(msg) for slug in self.tokens}
        return {slug: [] for slug in self.tokens}


class _AlwaysUnhealthyAdapter(_FakeAdapter):
    """Every constructed instance is unhealthy forever -- exercises the 2.4
    feed watchdog (restart, then feed_dead on a second trip with zero
    healthy ticks since the restart)."""

    instances: List["_AlwaysUnhealthyAdapter"] = []

    def __init__(self, tokens: Dict[str, str]) -> None:
        super().__init__(tokens)
        _AlwaysUnhealthyAdapter.instances.append(self)

    def healthy(self) -> bool:
        return False


class _FakeEngine:
    def __init__(self, reprice_s: float, seed: int = 42, garch_refit_s: float = 21_600.0) -> None:
        self.reprice_s = reprice_s
        self.garch_refit_s = garch_refit_s
        self.latencies: List[float] = []

    def __call__(self, strikes, hours_to_expiry, **kwargs):
        scale = 3000.0
        out = {float(k): float(1.0 / (1.0 + math.exp((float(k) - S0) / scale))) for k in strikes}
        out["_meta"] = {"n_sims": 1000, "S0": S0, "horizon_gate_active": False}
        return out


class _AlwaysRaisingEngine:
    """Stands in for CachedEngine but raises on every call -- exercises the
    2.6 tick-failure circuit breaker."""

    def __init__(self, reprice_s: float, seed: int = 42, garch_refit_s: float = 21_600.0) -> None:
        self.reprice_s = reprice_s
        self.latencies: List[float] = []

    def __call__(self, strikes, hours_to_expiry, **kwargs):
        raise RuntimeError("synthetic pricing failure")


class _FakeVolGate:
    def __init__(self):
        self.regime = "normal"
        self.shock = False
        self.kelly_mult = 1.0
        self.edge_add_cents = 0.0


def _fake_compute_vol_gate(df, now):
    return _FakeVolGate()


def _default_ladder(event_slug: str) -> Tuple[str, List[Tuple[str, float, str]]]:
    # 5-day-out expiry so tte keeps quoting alive (no near-resolution PULLED,
    # no settlement gate firing).
    expiry_key = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d")
    ladder = [("m-98k", 98000.0, "tok-98k"), ("m-102k", 102000.0, "tok-102k")]
    return expiry_key, ladder


def _install_common_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resolve_event_fn: Optional[Callable[[str], Tuple[str, List[Tuple[str, float, str]]]]] = None,
    adapter_cls=_FakeAdapter,
    engine_cls=_FakeEngine,
    btc_csv_path: Optional[Path] = None,
) -> None:
    monkeypatch.setattr(paper_runner, "resolve_event", resolve_event_fn or _default_ladder)
    monkeypatch.setattr(paper_runner, "PolymarketFeedAdapter", adapter_cls)
    # Multi-expiry refactor: the engine seam is now a compute callable
    # injected into the SharedPricingEngine (was: the CachedEngine class).
    monkeypatch.setattr(paper_runner, "_ENGINE_COMPUTE_FN", engine_cls(reprice_s=0.0))
    monkeypatch.setattr("core.strategy.vol_gate.compute_vol_gate", _fake_compute_vol_gate)

    if btc_csv_path is None:
        btc_csv_path = tmp_path / "btc_intraday_1m.csv"
        btc_csv_path.write_text("timestamp,close\n2026-01-01T00:00:00Z,100000.0\n", encoding="ascii")
    monkeypatch.setattr(paper_runner, "_BTC_INTRADAY_PATH", btc_csv_path)


def _wait_for_heartbeat_tick(out_dir: Path, min_tick: int, timeout_s: float = 20.0) -> int:
    hb_path = out_dir / "heartbeat.json"
    deadline = time.time() + timeout_s
    tick = 0
    while time.time() < deadline:
        if hb_path.exists():
            try:
                tick = json.loads(hb_path.read_text(encoding="ascii")).get("tick", 0)
            except (OSError, ValueError):
                tick = 0
        if tick >= min_tick:
            return tick
        time.sleep(0.05)
    return tick


def _run_in_thread(argv: List[str]) -> Tuple[threading.Thread, Dict[str, int]]:
    result: Dict[str, int] = {}

    def _target():
        result["code"] = paper_runner.run(argv)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t, result


# ---------------------------------------------------------------------------
# 2.1 -- resumable state
# ---------------------------------------------------------------------------


def test_initial_heartbeat_written_before_resolve_event(tmp_path, monkeypatch):
    """The very first heartbeat.json write happens right after out_dir
    creation, before resolve_event/warmup -- verified by making resolve_event
    itself check that heartbeat.json already exists with tick=0."""
    seen = {}

    def _resolve(event_slug: str):
        hb = tmp_path / "out" / "heartbeat.json"
        seen["existed"] = hb.exists()
        if hb.exists():
            seen["tick"] = json.loads(hb.read_text(encoding="ascii")).get("tick")
        return _default_ladder(event_slug)

    _install_common_stubs(monkeypatch, tmp_path, resolve_event_fn=_resolve)
    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"
    (ctl_dir).mkdir(parents=True, exist_ok=True)
    # stop file present from before start would be removed at startup, so we
    # instead just run briefly and stop via thread.
    t, result = _run_in_thread([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 1)
        assert tick >= 1
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)

    assert seen.get("existed") is True
    assert seen.get("tick") == 0


def test_state_db_resume_appends_csvs_not_truncated(tmp_path, monkeypatch):
    _install_common_stubs(monkeypatch, tmp_path)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"
    state_db = tmp_path / "persistent.db"

    argv = [
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        "--state-db", str(state_db),
    ]

    # --- run 1: fresh state db ---
    t, result = _run_in_thread(argv)
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 3)
        assert tick >= 3
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)
    assert result.get("code") == 0
    assert state_db.exists()

    quotes_path = out_dir / "quotes.csv"
    lines_after_run1 = quotes_path.read_text(encoding="ascii").splitlines()
    assert len(lines_after_run1) > 1  # header + at least one data row

    # --- run 2: same state db AND same --out -- must append, not truncate ---
    # heartbeat.json is overwritten (not appended) every tick, so it still
    # holds run 1's last tick count at this point; delete it so
    # _wait_for_heartbeat_tick doesn't observe a stale "tick >= 3" from run 1
    # and fire the stop file before run 2 has even reached its tick loop.
    (out_dir / "heartbeat.json").unlink(missing_ok=True)
    t2, result2 = _run_in_thread(argv)
    try:
        tick2 = _wait_for_heartbeat_tick(out_dir, 3)
        assert tick2 >= 3
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t2.join(timeout=20.0)
        assert not t2.is_alive()
    finally:
        if t2.is_alive():  # pragma: no cover
            t2.join(timeout=5.0)
    assert result2.get("code") == 0

    lines_after_run2 = quotes_path.read_text(encoding="ascii").splitlines()
    # exactly one header line total (not duplicated), and strictly more data
    # rows than after run 1.
    header_count = sum(1 for ln in lines_after_run2 if ln.startswith("ts,market,strike"))
    assert header_count == 1
    assert len(lines_after_run2) > len(lines_after_run1)


def test_state_db_resume_runs_restart_and_settle_catchup(tmp_path, monkeypatch):
    """THE regression test for 2.1: a previous-event LIVE order and open
    position are seeded directly into a persistent db; a paper_runner.run()
    against a DIFFERENT (current) event but the SAME --state-db must run the
    resume protocol (mark_all_live_orders_unknown -> restart -> settle(
    catch_up=True)) and settle/cancel them out."""
    old_expiry = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
    old_market, old_strike = "m-old-98k", 98000.0

    state_db = tmp_path / "persistent.db"
    seed_store = MMStateStore(str(state_db))
    seed_store.upsert_market(old_market, old_expiry, old_strike)
    seed_store.upsert_order(
        "seed-order-1", old_market, Side.BUY_YES, 0.40, 5.0, "LIVE",
        ts_placed=datetime.now(timezone.utc) - timedelta(days=11),
    )
    seed_ts = datetime.now(timezone.utc) - timedelta(days=11)
    seed_store.record_fill_and_update_inventory(
        Fill(ts=seed_ts, market_id=old_market, order_id="seed-fill-1", side=Side.BUY_YES,
             price=0.40, size=5.0, liquidity=LiquiditySource.MAKER, venue_ts=seed_ts),
        ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0),
    )
    q_before = seed_store.fold_fills_to_inventory()[old_market].q
    assert q_before == pytest.approx(5.0)
    seed_store.close()

    old_settle_dt = settlement_instant_utc(old_expiry)
    idx = pd.to_datetime([
        old_settle_dt - timedelta(minutes=2), old_settle_dt, old_settle_dt + timedelta(minutes=2),
    ])
    intraday = pd.DataFrame({"close": [99000.0, 99500.0, 99900.0]}, index=idx)  # spot > 98000 -> YES
    data_provider = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    monkeypatch.setattr(paper_runner, "_DATA_PROVIDER", data_provider)

    _install_common_stubs(monkeypatch, tmp_path)  # current event 5 days out

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    t, result = _run_in_thread([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        "--state-db", str(state_db),
    ])
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 1)
        assert tick >= 1
        time.sleep(0.3)
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)
    assert result.get("code") == 0

    store = MMStateStore(str(state_db))
    try:
        # restart() ran: the previously-LIVE order was marked UNKNOWN then
        # reconciled (paper venue has no cross-process open-order memory, so
        # it is cancelled, matching test_mm_harness_ws1's cancelled_unknown).
        seed_order = store.get_order("seed-order-1")
        assert seed_order is not None
        assert seed_order.status == "CANCELLED"

        # settle(catch_up=True) ran: the old-event position is closed out.
        settlement_row = store.get_settlement(old_market, old_expiry)
        assert settlement_row is not None
        assert settlement_row.outcome.value == "YES"
        assert store.fold_fills_to_inventory()[old_market].q == pytest.approx(0.0)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# W0.1 -- ReconciliationResult consumption on resume
# ---------------------------------------------------------------------------


def test_resume_position_discrepancy_journals_manual_and_holds_override(tmp_path, monkeypatch):
    """A scripted store/venue position mismatch on resume must: (1) surface
    as heartbeat.resume_discrepancies > 0, (2) write a MANUAL-trigger PULLED
    risk-journal row for the discrepant market, and (3) force
    manual_override=True on ticks until the first clean tick completes, then
    release it (plan Wave 0 W0.1).

    Discrepancy is engineered by desyncing the `inventory` table (what
    PaperVenueAdapter.fetch_positions reads -- "venue truth" in paper mode)
    from the `fills` table (what fold_fills_to_inventory derives "store
    truth" from): a fill records q=5, but the inventory row is overwritten to
    q=8 afterward, so restart_reconcile's store-vs-venue compare disagrees.
    """
    old_expiry = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
    old_market, old_strike = "m-old-98k", 98000.0

    state_db = tmp_path / "persistent.db"
    seed_store = MMStateStore(str(state_db))
    seed_store.upsert_market(old_market, old_expiry, old_strike)
    seed_ts = datetime.now(timezone.utc) - timedelta(days=11)
    seed_store.record_fill_and_update_inventory(
        Fill(ts=seed_ts, market_id=old_market, order_id="seed-fill-1", side=Side.BUY_YES,
             price=0.40, size=5.0, liquidity=LiquiditySource.MAKER, venue_ts=seed_ts),
        ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0),
    )
    # Desync: inventory table now disagrees with fold(fills) for old_market.
    seed_store.upsert_inventory(
        old_market, ContractInv(q=8.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0),
        updated_ts=seed_ts,
    )
    seed_store.close()

    old_settle_dt = settlement_instant_utc(old_expiry)
    idx = pd.to_datetime([
        old_settle_dt - timedelta(minutes=2), old_settle_dt, old_settle_dt + timedelta(minutes=2),
    ])
    intraday = pd.DataFrame({"close": [99000.0, 99500.0, 99900.0]}, index=idx)
    data_provider = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    monkeypatch.setattr(paper_runner, "_DATA_PROVIDER", data_provider)

    _install_common_stubs(monkeypatch, tmp_path)  # current event 5 days out

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    t, result = _run_in_thread([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        "--state-db", str(state_db),
    ])
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 2)
        assert tick >= 2
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)
    assert result.get("code") == 0

    hb = json.loads((out_dir / "heartbeat.json").read_text(encoding="ascii"))
    assert hb["resume_discrepancies"] == 1

    store = MMStateStore(str(state_db))
    try:
        directives = store.get_risk_journal(old_market)
        manual_rows = [
            d for d in directives
            if d.mode.value == "PULLED" and any(t.value == "MANUAL" for t in d.triggers)
        ]
        assert manual_rows, "expected a MANUAL-trigger PULLED risk-journal row for the discrepant market"
        assert manual_rows[0].cancel_all is True
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 2.2 -- exit after ladder settles + rollover + exit-code mapping
# ---------------------------------------------------------------------------


def _settled_ladder_slug(event_slug: str) -> Tuple[str, List[Tuple[str, float, str]]]:
    # Expiry far enough in the past that "now >= settlement_instant + 30min"
    # is already true on tick 1.
    expiry_key = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    ladder = [("m-set-a", 98000.0, "tok-a"), ("m-set-b", 102000.0, "tok-b")]
    return expiry_key, ladder


def test_ladder_settled_exit_returns_42(tmp_path, monkeypatch):
    expiry_key, ladder = _settled_ladder_slug("fake-event")

    def _resolve(event_slug):
        return expiry_key, ladder

    settle_dt = settlement_instant_utc(expiry_key)
    idx = pd.to_datetime([settle_dt - timedelta(minutes=2), settle_dt, settle_dt + timedelta(minutes=2)])
    intraday = pd.DataFrame({"close": [101000.0, 101000.0, 101000.0]}, index=idx)
    data_provider = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    monkeypatch.setattr(paper_runner, "_DATA_PROVIDER", data_provider)

    _install_common_stubs(monkeypatch, tmp_path, resolve_event_fn=_resolve)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    code = paper_runner.run([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])

    assert code == 42
    current_run = json.loads((ctl_dir / "current_run.json").read_text(encoding="ascii"))
    assert current_run["exit_reason"] == "ladder_settled"


def test_settlement_timeout_exit_returns_42(tmp_path, monkeypatch):
    """An UNSETTLEABLE market (no intraday coverage at all) never becomes
    terminal -- the hard --max-settlement-wait-h fallback must still exit
    with code 42 / exit_reason settlement_timeout."""
    expiry_key, ladder = _settled_ladder_slug("fake-event")

    def _resolve(event_slug):
        return expiry_key, ladder

    # Empty intraday frame -> _expiry_is_settleable-style range check always
    # fails -> UNSETTLEABLE forever.
    data_provider = BTCDataProvider(intraday=pd.DataFrame({"close": []}), daily=pd.DataFrame())
    monkeypatch.setattr(paper_runner, "_DATA_PROVIDER", data_provider)

    _install_common_stubs(monkeypatch, tmp_path, resolve_event_fn=_resolve)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    code = paper_runner.run([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.05",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        "--max-settlement-wait-h", "0.0",  # already elapsed on tick 1
    ])

    assert code == 42
    current_run = json.loads((ctl_dir / "current_run.json").read_text(encoding="ascii"))
    assert current_run["exit_reason"] == "settlement_timeout"


def test_stop_file_still_exits_zero(tmp_path, monkeypatch):
    """Exit-code mapping regression: exit reasons NOT in the 42/1 sets
    (stop_file here) must still map to 0."""
    _install_common_stubs(monkeypatch, tmp_path)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    t, result = _run_in_thread([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 2)
        assert tick >= 2
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)
    assert result.get("code") == 0


# ---------------------------------------------------------------------------
# 2.4 -- feed watchdog
# ---------------------------------------------------------------------------


def test_feed_watchdog_restarts_then_exits_feed_dead(tmp_path, monkeypatch):
    _AlwaysUnhealthyAdapter.instances = []
    _install_common_stubs(monkeypatch, tmp_path, adapter_cls=_AlwaysUnhealthyAdapter)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    code = paper_runner.run([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.01",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        "--feed-dead-ticks", "3",
    ])

    assert code == 1
    current_run = json.loads((ctl_dir / "current_run.json").read_text(encoding="ascii"))
    assert current_run["exit_reason"] == "feed_dead"
    # one restart (new adapter instance constructed) before the fatal exit.
    assert len(_AlwaysUnhealthyAdapter.instances) == 2
    assert _AlwaysUnhealthyAdapter.instances[0].stopped


# ---------------------------------------------------------------------------
# 2.6 -- tick-failure circuit breaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_exits_tick_errors(tmp_path, monkeypatch):
    _install_common_stubs(monkeypatch, tmp_path, engine_cls=_AlwaysRaisingEngine)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    code = paper_runner.run([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.01",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        "--max-consecutive-tick-errors", "4",
    ])

    assert code == 1
    current_run = json.loads((ctl_dir / "current_run.json").read_text(encoding="ascii"))
    assert current_run["exit_reason"] == "tick_errors"

    ticks_csv = (out_dir / "ticks.csv").read_text(encoding="ascii")
    assert ticks_csv.count("TICK_ERROR") == 4


# ---------------------------------------------------------------------------
# 2.3 -- BTC staleness guard
# ---------------------------------------------------------------------------


def test_stale_btc_csv_triggers_manual_override_pulled(tmp_path, monkeypatch):
    btc_csv = tmp_path / "btc_intraday_1m.csv"
    btc_csv.write_text("timestamp,close\n2026-01-01T00:00:00Z,100000.0\n", encoding="ascii")
    # Backdate the mtime well past --btc-stale-max-s.
    old_time = time.time() - 100_000
    import os
    os.utime(btc_csv, (old_time, old_time))

    _install_common_stubs(monkeypatch, tmp_path, btc_csv_path=btc_csv)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    t, result = _run_in_thread([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        "--btc-stale-max-s", "10",
    ])
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 2)
        assert tick >= 2
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)
    assert result.get("code") == 0

    hb = json.loads((out_dir / "heartbeat.json").read_text(encoding="ascii"))
    assert hb["btc_data_age_s"] is not None
    assert hb["btc_data_age_s"] > 10.0

    quotes_csv = (out_dir / "quotes.csv").read_text(encoding="ascii")
    assert "PULLED" in quotes_csv


def test_missing_btc_csv_counts_as_stale(tmp_path, monkeypatch):
    missing_csv = tmp_path / "does_not_exist.csv"
    # _install_common_stubs would try to write it; pass it directly instead
    # and skip the write by monkeypatching _read_btc_intraday's target file
    # to something that exists for the initial read, then point the staleness
    # check at the missing path.
    real_csv = tmp_path / "btc_intraday_1m.csv"
    real_csv.write_text("timestamp,close\n2026-01-01T00:00:00Z,100000.0\n", encoding="ascii")

    _install_common_stubs(monkeypatch, tmp_path, btc_csv_path=real_csv)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    t, result = _run_in_thread([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 1)
        assert tick >= 1
        # Delete the csv mid-run so the FRESH per-tick stat sees it missing.
        real_csv.unlink()
        tick2 = _wait_for_heartbeat_tick(out_dir, tick + 2)
        assert tick2 >= tick + 2
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)
    assert result.get("code") == 0

    hb = json.loads((out_dir / "heartbeat.json").read_text(encoding="ascii"))
    assert hb["btc_data_age_s"] is None  # missing -> None, still counts as stale
    quotes_csv = (out_dir / "quotes.csv").read_text(encoding="ascii")
    assert "PULLED" in quotes_csv
