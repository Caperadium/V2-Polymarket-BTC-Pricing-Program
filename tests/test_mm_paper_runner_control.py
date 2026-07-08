"""Tests for market_maker.paper_runner's control-file / heartbeat / settlement
/ pnl-journaling plumbing (plan "MM Monitor Dashboard Page + Engine Start/Stop
Control", Step 2 / Step 6.3).

Everything network- or GARCH-adjacent is stubbed at the paper_runner module
seam (resolve_event, PolymarketFeedAdapter, CachedEngine) plus
core.strategy.vol_gate.compute_vol_gate and the BTC intraday csv path
(_BTC_INTRADAY_PATH) -- no live network calls, no real pricing engine, no
touching the real DATA/ csvs or temp/paper_run/control.

run() is invoked directly (in-process, not via subprocess) either
synchronously (when the stop file is pre-created so the very first loop
condition check exits immediately, after run_meta.json is already written)
or in a background thread (when we need a few real ticks to elapse first).
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
# stubs
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """Stands in for PolymarketFeedAdapter: one synthetic order-book snapshot
    on the first drain() (so BookMirror has a valid mid and the harness can
    compute a fair-value snapshot), silence afterward -- message silence on
    a quiet book is not feed loss (matches the real adapter's semantics)."""

    def __init__(self, tokens: Dict[str, str]) -> None:
        self.tokens = tokens
        self._sent = False

    def start(self) -> None:
        pass

    def stop(self, join_timeout_s: float = 10.0) -> None:
        pass

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


class _FakeEngine:
    """Stands in for CachedEngine: same constructor signature, no GARCH."""

    def __init__(self, reprice_s: float, seed: int = 42, garch_refit_s: float = 21_600.0) -> None:
        self.reprice_s = reprice_s
        self.garch_refit_s = garch_refit_s
        self.latencies: List[float] = []

    def __call__(self, strikes, hours_to_expiry, **kwargs):
        scale = 3000.0
        out = {float(k): float(1.0 / (1.0 + math.exp((float(k) - S0) / scale))) for k in strikes}
        out["_meta"] = {"n_sims": 1000, "S0": S0, "horizon_gate_active": False}
        return out


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
    # no settlement gate firing) -- matches the convention in
    # tests/test_mm_integration.py.
    expiry_key = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d")
    ladder = [("m-98k", 98000.0, "tok-98k"), ("m-102k", 102000.0, "tok-102k")]
    return expiry_key, ladder


def _install_common_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resolve_event_fn: Optional[Callable[[str], Tuple[str, List[Tuple[str, float, str]]]]] = None,
) -> None:
    monkeypatch.setattr(paper_runner, "resolve_event", resolve_event_fn or _default_ladder)
    monkeypatch.setattr(paper_runner, "PolymarketFeedAdapter", _FakeAdapter)
    monkeypatch.setattr(paper_runner, "CachedEngine", _FakeEngine)
    monkeypatch.setattr("core.strategy.vol_gate.compute_vol_gate", _fake_compute_vol_gate)

    btc_csv = tmp_path / "btc_intraday_1m.csv"
    btc_csv.write_text("timestamp,close\n2026-01-01T00:00:00Z,100000.0\n", encoding="ascii")
    monkeypatch.setattr(paper_runner, "_BTC_INTRADAY_PATH", btc_csv)


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


# ---------------------------------------------------------------------------
# graceful stop-file round trip
# ---------------------------------------------------------------------------


def test_stop_file_graceful_shutdown(tmp_path, monkeypatch):
    _install_common_stubs(monkeypatch, tmp_path)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    result: Dict[str, int] = {}

    def _target():
        result["code"] = paper_runner.run([
            "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
            "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        ])

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 3)
        assert tick >= 3, "runner did not reach tick 3 in time"

        pid_path = ctl_dir / "mm_paper.pid"
        assert pid_path.exists()

        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive(), "runner did not exit after stop file was created"
    finally:
        if t.is_alive():  # pragma: no cover - safety net, should not trigger
            t.join(timeout=5.0)

    assert result.get("code") == 0
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "heartbeat.json").exists()
    assert (out_dir / "run_meta.json").exists()
    assert not (ctl_dir / "mm_paper.pid").exists()
    assert not (ctl_dir / "mm_paper.stop").exists()

    current_run = json.loads((ctl_dir / "current_run.json").read_text(encoding="ascii"))
    assert current_run["exit_reason"] == "stop_file"
    assert "ended_utc" in current_run

    run_meta = json.loads((out_dir / "run_meta.json").read_text(encoding="ascii"))
    assert run_meta["event_slug"] == "fake-event"

    store = MMStateStore(str(out_dir / "paper_state.db"))
    try:
        pnl_rows = store.get_pnl_snapshots()
        assert any(r.market_id is None for r in pnl_rows), "expected at least one TOTAL pnl row"
    finally:
        store.close()


def test_stop_file_with_mismatched_pid_is_ignored(tmp_path, monkeypatch):
    _install_common_stubs(monkeypatch, tmp_path)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"

    result: Dict[str, int] = {}

    def _target():
        result["code"] = paper_runner.run([
            "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
            "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        ])

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 2)
        assert tick >= 2

        # Stamp with a PID that is definitely not ours -> must be ignored.
        bogus_pid = 999999
        (ctl_dir / "mm_paper.stop").write_text(str(bogus_pid), encoding="ascii")
        time.sleep(1.0)
        assert t.is_alive(), "runner stopped on a stop file stamped with a different PID"

        # Now the real (unstamped) stop request.
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)

    assert result.get("code") == 0


# ---------------------------------------------------------------------------
# settlement round trip (B2/R3): settle -> pnl snapshot includes payoff
# ---------------------------------------------------------------------------


def test_settlement_populates_settlements_and_pnl_includes_payoff(tmp_path, monkeypatch):
    # Expiry "yesterday" -> settlement_instant_utc(expiry_key) is already in
    # the past relative to wall-clock `now`, so the B2 settlement gate fires
    # on the very first tick.
    expiry_key = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ladder = [("m-set-lo", 98000.0, "tok-lo"), ("m-set-hi", 102000.0, "tok-hi")]

    def _resolve(event_slug: str):
        return expiry_key, ladder

    _install_common_stubs(monkeypatch, tmp_path, resolve_event_fn=_resolve)

    settle_dt = settlement_instant_utc(expiry_key)
    intraday = pd.DataFrame({"close": [101000.0]}, index=pd.DatetimeIndex([settle_dt], tz="UTC"))
    data_provider = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    monkeypatch.setattr(paper_runner, "_DATA_PROVIDER", data_provider)

    # WS2.2: paper_runner now auto-exits (exit_reason=ladder_settled, code 42)
    # once every market is terminal AND 30min have elapsed since the
    # settlement instant -- true almost immediately for a genuinely
    # backdated "yesterday" expiry_key, which would race the stop-file path
    # this test exercises below. Patch ONLY paper_runner's imported
    # settlement_instant_utc (its per-tick gate + grace-period check) to a
    # fixed instant 5min in the past -- well past the gate (so settlement
    # still runs on tick 1) but short of the 30min grace, so the runner
    # keeps ticking and the stop-file shutdown path below still applies.
    # settlement_handler's OWN internal resolution (a separate binding of
    # the same function) is untouched, so it still resolves spot against the
    # REAL "yesterday" settle_dt matching the data_provider fixture above.
    fake_gate_instant = datetime.now(timezone.utc) - timedelta(minutes=5)
    monkeypatch.setattr(paper_runner, "settlement_instant_utc", lambda expiry_key: fake_gate_instant)

    out_dir = tmp_path / "out"
    ctl_dir = tmp_path / "control"
    out_dir.mkdir(parents=True)

    # Pre-seed an open long YES position on m-set-lo through the SAME fills
    # channel settlement uses, so the fold(fills)==inventory invariant stays
    # meaningful and the settlement pseudo-fill has a real position to close.
    # strike 98000 < spot 101000 -> settles YES.
    seed_ts = settle_dt - timedelta(days=1)
    pre_store = MMStateStore(str(out_dir / "paper_state.db"))
    pre_store.record_fill_and_update_inventory(
        Fill(ts=seed_ts, market_id="m-set-lo", order_id="seed", side=Side.BUY_YES,
             price=0.40, size=5.0, liquidity=LiquiditySource.MAKER, venue_ts=seed_ts),
        ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0),
    )
    pre_store.close()

    result: Dict[str, int] = {}

    def _target():
        result["code"] = paper_runner.run([
            "--event-slug", "fake-event", "--minutes", "0", "--tick-s", "0.2",
            "--warmup-s", "0", "--out", str(out_dir), "--control-dir", str(ctl_dir),
        ])

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    try:
        tick = _wait_for_heartbeat_tick(out_dir, 1)
        assert tick >= 1
        # Give the settlement + pnl-snapshot step inside tick 1 a moment to
        # land before we stop (both run synchronously within the tick, but
        # the heartbeat write races the very end of the tick body slightly).
        time.sleep(0.3)
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)

    assert result.get("code") == 0

    store = MMStateStore(str(out_dir / "paper_state.db"))
    try:
        settlements = store.get_all_settlements()
        ev = next((e for e in settlements if e.market_id == "m-set-lo"), None)
        assert ev is not None, "expected a settlement row for m-set-lo"
        assert ev.outcome.value == "YES"

        fills = store.get_fills("m-set-lo")
        assert len(fills) == 2  # seed open + settlement pseudo-fill
        assert fills[-1].liquidity is LiquiditySource.SETTLEMENT

        inv = store.get_inventory("m-set-lo")
        assert inv is not None
        assert inv.q == pytest.approx(0.0)

        pnl_rows = store.get_pnl_snapshots()
        totals = [r for r in pnl_rows if r.market_id is None]
        assert totals, "expected at least one TOTAL pnl row"
        # cash = -0.40*5 (open) + 1.0*5 (settlement, payoff_yes=1.0) = 3.0;
        # q is back to 0 post-settlement -> realized = cash + 0*0 = 3.0.
        # m-set-hi never had a fill so it never enters get_all_inventory()
        # and contributes nothing to the TOTAL row.
        assert totals[-1].realized == pytest.approx(3.0)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# --config merge + CLI override
# ---------------------------------------------------------------------------


def test_config_file_merge_and_cli_override(tmp_path, monkeypatch):
    _install_common_stubs(monkeypatch, tmp_path)

    # Tiny `minutes` so each synchronous run ends on its own after ~1 tick.
    # (A pre-created stop file would NOT work here: the runner deletes any
    # stale stop file at startup, per the M3 startup order.)
    cfg = {
        "event_slug": "cfg-event", "minutes": 0.002, "tick_s": 0.2,
        "reprice_s": 111.0, "bankroll": 555.0, "warmup_s": 0.0,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="ascii")

    # Run 1: config values used verbatim.
    out_dir = tmp_path / "out1"
    ctl_dir = tmp_path / "ctl1"

    code = paper_runner.run([
        "--config", str(cfg_path), "--out", str(out_dir), "--control-dir", str(ctl_dir),
    ])
    assert code == 0
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="ascii"))
    assert meta["event_slug"] == "cfg-event"
    assert meta["bankroll"] == pytest.approx(555.0)
    assert meta["tick_s"] == pytest.approx(0.2)
    assert meta["reprice_s"] == pytest.approx(111.0)

    # Run 2: an explicit CLI flag overrides the same config key.
    out_dir2 = tmp_path / "out2"
    ctl_dir2 = tmp_path / "ctl2"

    code2 = paper_runner.run([
        "--config", str(cfg_path), "--tick-s", "0.3",
        "--out", str(out_dir2), "--control-dir", str(ctl_dir2),
    ])
    assert code2 == 0
    meta2 = json.loads((out_dir2 / "run_meta.json").read_text(encoding="ascii"))
    assert meta2["tick_s"] == pytest.approx(0.3)  # CLI override wins
    assert meta2["bankroll"] == pytest.approx(555.0)  # untouched config value still applied


def test_event_slug_required_without_config():
    with pytest.raises(SystemExit):
        paper_runner.run(["--minutes", "0"])
