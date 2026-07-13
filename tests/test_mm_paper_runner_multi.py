"""Multi-expiry paper_runner tests: auto-mode acquisition of several ladders,
heartbeat aggregate + per-expiry payload, run pointer (current_run/run_meta)
events list, in-process rollover without a process exit, the
no_quotable_events exit mapping, fixed-slug legacy isolation, and the
one-reprice-per-tick budget at the runner level.

Follows tests/test_mm_paper_runner_ws2.py's conventions: run() driven
in-process against temp control/out dirs; resolve_events_multi,
PolymarketFeedAdapter, _ENGINE_COMPUTE_FN, compute_vol_gate and the BTC csv
stubbed at the paper_runner module seam.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker import paper_runner
from market_maker.settlement_handler import BTCDataProvider, settlement_instant_utc

S0 = 100000.0


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """Resends a full book snapshot on EVERY drain so a slot skipped during
    its warmup tick still gets a live book on the first tick it runs."""

    instances: List["_FakeAdapter"] = []

    def __init__(self, tokens: Dict[str, str]) -> None:
        self.tokens = tokens
        self.started = False
        self.stopped = False
        _FakeAdapter.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self, join_timeout_s: float = 10.0) -> None:
        self.stopped = True

    def healthy(self) -> bool:
        return True

    def drain(self) -> Dict[str, List[Dict[str, object]]]:
        msg = [{
            "type": "snapshot",
            "bids": [(0.45, 100.0), (0.44, 100.0)],
            "asks": [(0.55, 100.0), (0.56, 100.0)],
        }]
        return {slug: list(msg) for slug in self.tokens}


class _CountingCompute:
    def __init__(self) -> None:
        self.calls: List[Tuple] = []

    def __call__(self, strikes, hours_to_expiry, **kwargs):
        self.calls.append((tuple(strikes), hours_to_expiry))
        scale = 3000.0
        out = {float(k): float(1.0 / (1.0 + math.exp((float(k) - S0) / scale))) for k in strikes}
        out["_meta"] = {"n_sims": 1000, "S0": S0, "horizon_gate_active": False}
        return out


class _FakeVolGate:
    regime = "normal"
    shock = False
    kelly_mult = 1.0
    edge_add_cents = 0.0


def _fake_compute_vol_gate(df, now):
    return _FakeVolGate()


def _ladder(prefix: str) -> List[Tuple[str, float, str]]:
    return [(f"{prefix}-98k", 98000.0, f"tok-{prefix}-98"),
            (f"{prefix}-102k", 102000.0, f"tok-{prefix}-102")]


def _future_expiry(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


def _install(monkeypatch, tmp_path, resolver_fn, compute=None) -> _CountingCompute:
    compute = compute or _CountingCompute()
    monkeypatch.setattr(paper_runner, "resolve_events_multi", resolver_fn)
    monkeypatch.setattr(paper_runner, "PolymarketFeedAdapter", _FakeAdapter)
    monkeypatch.setattr(paper_runner, "_ENGINE_COMPUTE_FN", compute)
    monkeypatch.setattr("core.strategy.vol_gate.compute_vol_gate", _fake_compute_vol_gate)
    btc_csv = tmp_path / "btc_intraday_1m.csv"
    btc_csv.write_text("timestamp,close\n2026-01-01T00:00:00Z,100000.0\n", encoding="ascii")
    monkeypatch.setattr(paper_runner, "_BTC_INTRADAY_PATH", btc_csv)
    return compute


def _run_in_thread(argv: List[str]) -> Tuple[threading.Thread, Dict[str, int]]:
    result: Dict[str, int] = {}

    def _target():
        result["code"] = paper_runner.run(argv)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t, result


def _read_heartbeat(out_dir: Path) -> Optional[dict]:
    hb_path = out_dir / "heartbeat.json"
    if not hb_path.exists():
        return None
    try:
        return json.loads(hb_path.read_text(encoding="ascii"))
    except (OSError, ValueError):
        return None


def _wait_for_heartbeat(out_dir: Path, pred, timeout_s: float = 20.0) -> Optional[dict]:
    deadline = time.time() + timeout_s
    hb = None
    while time.time() < deadline:
        hb = _read_heartbeat(out_dir)
        if hb is not None and pred(hb):
            return hb
        time.sleep(0.05)
    return hb


def _stop_and_join(ctl_dir: Path, t: threading.Thread) -> None:
    (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
    t.join(timeout=20.0)
    assert not t.is_alive()


# ---------------------------------------------------------------------------
# auto mode: two concurrent expiries
# ---------------------------------------------------------------------------


def test_auto_two_expiries_heartbeat_and_run_pointers(tmp_path, monkeypatch):
    ek_a, ek_b = _future_expiry(5), _future_expiry(6)

    def _resolver(now, lead, cap, exclude):
        events = [("ev-a", ek_a, _ladder("a")), ("ev-b", ek_b, _ladder("b"))]
        return [e for e in events if e[1] not in exclude][:cap]

    _install(monkeypatch, tmp_path, _resolver)
    out_dir, ctl_dir = tmp_path / "out", tmp_path / "control"

    t, result = _run_in_thread([
        "--event-slug", "auto", "--max-expiries", "2", "--minutes", "0",
        "--tick-s", "0.2", "--warmup-s", "0",
        "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])
    try:
        hb = _wait_for_heartbeat(out_dir, lambda h: h.get("tick", 0) >= 4)
        assert hb is not None and hb["tick"] >= 4
        _stop_and_join(ctl_dir, t)
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)
    assert result.get("code") == 0

    hb = _read_heartbeat(out_dir)
    # aggregate semantics
    assert hb["n_expiries_active"] == 2
    assert hb["feed_healthy"] is True
    assert hb["bankroll_frozen"] in (True, False)  # OR over loops, present
    assert hb["ladders_settled_total"] == 0
    # noarb_repairs: true violating-ladder count (sum over ladders), distinct
    # from the legacy warm-up 'noarb_violations' counter
    assert hb["noarb_repairs"] >= 0
    # per-expiry payload
    assert set(hb["expiries"].keys()) == {ek_a, ek_b}
    for ek, slug in ((ek_a, "ev-a"), (ek_b, "ev-b")):
        e = hb["expiries"][ek]
        assert e["event_slug"] == slug
        assert e["state"] == "active"
        assert "mode_counts" in e and "fills" in e
        assert e["noarb_repairs"] >= 0

    # run pointers: events list + legacy singular fields = nearest expiry
    run_meta = json.loads((out_dir / "run_meta.json").read_text(encoding="ascii"))
    assert [e["expiry_key"] for e in run_meta["events"]] == sorted([ek_a, ek_b])
    assert run_meta["expiry_key"] == min(ek_a, ek_b)
    assert run_meta["max_expiries"] == 2
    current_run = json.loads((ctl_dir / "current_run.json").read_text(encoding="ascii"))
    assert [e["expiry_key"] for e in current_run["events"]] == sorted([ek_a, ek_b])
    assert current_run["expiry_key"] == min(ek_a, ek_b)

    # both ladders quoted (rows for both expiries' markets in quotes.csv)
    quotes = (out_dir / "quotes.csv").read_text(encoding="ascii")
    assert "a-98k" in quotes and "b-98k" in quotes


def test_auto_in_process_rollover_keeps_process_alive(tmp_path, monkeypatch):
    _FakeAdapter.instances = []
    ek_past = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ek_b = _future_expiry(5)
    calls = {"n": 0}

    def _resolver(now, lead, cap, exclude):
        calls["n"] += 1
        if calls["n"] == 1:
            return [("ev-p", ek_past, _ladder("p")), ("ev-b", ek_b, _ladder("b"))]
        return []

    _install(monkeypatch, tmp_path, _resolver)

    settle_dt = settlement_instant_utc(ek_past)
    idx = pd.to_datetime([settle_dt - timedelta(minutes=2), settle_dt, settle_dt + timedelta(minutes=2)])
    intraday = pd.DataFrame({"close": [101000.0, 101000.0, 101000.0]}, index=idx)
    monkeypatch.setattr(paper_runner, "_DATA_PROVIDER",
                        BTCDataProvider(intraday=intraday, daily=pd.DataFrame()))

    out_dir, ctl_dir = tmp_path / "out", tmp_path / "control"
    t, result = _run_in_thread([
        "--event-slug", "auto", "--max-expiries", "2", "--minutes", "0",
        "--tick-s", "0.2", "--warmup-s", "0",
        "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])
    try:
        hb = _wait_for_heartbeat(
            out_dir,
            lambda h: h.get("ladders_settled_total", 0) >= 1 and h.get("tick", 0) >= 3,
        )
        assert hb is not None
        assert hb["ladders_settled_total"] == 1
        assert hb["n_expiries_active"] == 1
        assert set(hb["expiries"].keys()) == {ek_b}
        assert t.is_alive(), "in-process rollover must NOT exit the process"
        _stop_and_join(ctl_dir, t)
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)
    assert result.get("code") == 0

    current_run = json.loads((ctl_dir / "current_run.json").read_text(encoding="ascii"))
    assert current_run["exit_reason"] == "stop_file"
    assert [e["expiry_key"] for e in current_run["events"]] == [ek_b]
    # the settled ladder's adapter was stopped in-place
    p_adapters = [a for a in _FakeAdapter.instances if any("p-" in s for s in a.tokens)]
    assert p_adapters and all(a.stopped for a in p_adapters)


def test_auto_no_events_exits_42_no_quotable(tmp_path, monkeypatch):
    def _resolver(now, lead, cap, exclude):
        return []

    _install(monkeypatch, tmp_path, _resolver)
    out_dir, ctl_dir = tmp_path / "out", tmp_path / "control"

    code = paper_runner.run([
        "--event-slug", "auto", "--max-expiries", "3", "--minutes", "0",
        "--tick-s", "0.2", "--warmup-s", "0",
        "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])
    assert code == 42
    current_run = json.loads((ctl_dir / "current_run.json").read_text(encoding="ascii"))
    assert current_run["exit_reason"] == "no_quotable_events"


def test_fixed_slug_never_calls_multi_resolver(tmp_path, monkeypatch):
    """Fixed --event-slug mode is bit-compatible legacy: no acquisition, exit
    42 on ladder_settled, resolve_events_multi never touched."""
    ek_past = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")

    def _resolver(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("resolve_events_multi must not be called in fixed mode")

    _install(monkeypatch, tmp_path, _resolver)
    monkeypatch.setattr(paper_runner, "resolve_event",
                        lambda slug: (ek_past, _ladder("p")))

    settle_dt = settlement_instant_utc(ek_past)
    idx = pd.to_datetime([settle_dt - timedelta(minutes=2), settle_dt, settle_dt + timedelta(minutes=2)])
    intraday = pd.DataFrame({"close": [101000.0, 101000.0, 101000.0]}, index=idx)
    monkeypatch.setattr(paper_runner, "_DATA_PROVIDER",
                        BTCDataProvider(intraday=intraday, daily=pd.DataFrame()))

    out_dir, ctl_dir = tmp_path / "out", tmp_path / "control"
    code = paper_runner.run([
        "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
        "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])
    assert code == 42
    current_run = json.loads((ctl_dir / "current_run.json").read_text(encoding="ascii"))
    assert current_run["exit_reason"] == "ladder_settled"


def test_one_reprice_per_tick_at_runner_level(tmp_path, monkeypatch):
    """With the default (long) reprice_s, two fresh ladders cost exactly TWO
    engine computes total (one first-price per tick, staggered), never two in
    one tick."""
    ek_a, ek_b = _future_expiry(5), _future_expiry(6)

    def _resolver(now, lead, cap, exclude):
        events = [("ev-a", ek_a, _ladder("a")), ("ev-b", ek_b, _ladder("b"))]
        return [e for e in events if e[1] not in exclude][:cap]

    compute = _install(monkeypatch, tmp_path, _resolver)
    out_dir, ctl_dir = tmp_path / "out", tmp_path / "control"

    t, result = _run_in_thread([
        "--event-slug", "auto", "--max-expiries", "2", "--minutes", "0",
        "--tick-s", "0.2", "--warmup-s", "0", "--reprice-s", "300",
        "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])
    try:
        hb = _wait_for_heartbeat(out_dir, lambda h: h.get("tick", 0) >= 4)
        assert hb is not None and hb["tick"] >= 4
        _stop_and_join(ctl_dir, t)
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)
    assert result.get("code") == 0
    # exactly one first-price per ladder, staggered across ticks 1 and 2
    assert len(compute.calls) == 2
    hours = sorted(h for _s, h in compute.calls)
    assert hours[0] < hours[1]  # two DIFFERENT expiries priced
