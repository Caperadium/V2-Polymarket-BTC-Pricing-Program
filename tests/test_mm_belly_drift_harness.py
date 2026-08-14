"""Tests for C1 belly drift-anchored Bayes scoring (temp/mm_c1_belly_drift_
plan.md v3), Part B: market_maker/harness.py's lag buffer, scoring-event
gate, hypothetical shadow/control trajectories, bayes_score_log journaling,
and restart reload; plus market_maker/paper_runner.py's prune wiring.

Part A (fair_value_anchor.py's advance_weights()/belly_lag_* kwargs and
state_store.py's bayes_score_log table) is covered by
tests/test_mm_belly_drift_scoring.py -- this file does not re-derive the C1
drift-factor math, only the harness's sequencing around it.

Scripted synthetic feeds only, following tests/test_mm_harness_ws1.py's
conventions (fixed clock, deterministic scripted books, direct attribute
introspection on the loop).
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import AnchorMethod, BankrollState
from market_maker.fair_value_anchor import (
    BELLY_REGION,
    MARKET_MODEL_ID,
    PRICER_MODEL_ID,
    WING_REGION,
)
from market_maker.harness import PaperTradingLoop
from market_maker.order_lifecycle import SimClock
from market_maker.state_store import MMStateStore

START = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
EXPIRY = "2026-08-20"
# 3 strikes, scale=1000 in _engine below -> middle strike lands in the
# default belly_band (0.2, 0.8), outer two land outside it (wing) -- both
# from the PRICER curve and from the market book (constructed at a
# constant -0.05 offset from the pricer curve, so the market's OWN
# sanitized ladder classifies the same way fair_value_anchor.py requires).
MARKETS = [("m-98k", 98000.0), ("m-100k", 100000.0), ("m-102k", 102000.0)]
S0 = 100000.0
SCALE = 1000.0

# Small horizon/interval/slack so a scripted run reaches scoring events in a
# handful of ticks (plan's own suggestion).
HORIZON_S = 60.0
INTERVAL_S = 30.0
SLACK_S = 60.0
TICK_DT_S = 15.0


def _engine(s0=S0, scale=SCALE, n_sims=15000):
    def fn(strikes, hours_to_expiry, **kwargs):
        out = {float(k): float(1.0 / (1.0 + np.exp((float(k) - s0) / scale))) for k in strikes}
        out["_meta"] = {"n_sims": n_sims, "S0": s0, "horizon_gate_active": False}
        return out
    return fn


class _VG:
    regime = "normal"
    shock = False
    kelly_mult = 1.0
    edge_add_cents = 0.0


def _vol_gate():
    return lambda: _VG()


def _book_msg(mid: float):
    bid = round(max(0.01, mid - 0.01), 4)
    ask = round(min(0.99, mid + 0.01), 4)
    return [{"type": "snapshot", "bids": [(bid, 100.0)], "asks": [(ask, 100.0)]}]


def _pricer_curve(s0=S0, scale=SCALE):
    return {k: float(1.0 / (1.0 + np.exp((k - s0) / scale))) for _, k in MARKETS}


def _market_mids(offset: float = -0.05):
    """Market book mid per strike, offset from the pricer curve so the
    market is uniformly CHEAPER than the pricer (a "rich consensus"
    scenario -- consensus should sit above the market at the belly strike).
    Region classification (belly/wing) is computed from THIS ladder's own
    sanitized round-trip in fair_value_anchor.py, and at this offset it
    agrees with the pricer's own classification (same strikes land belly/
    wing either way)."""
    curve = _pricer_curve()
    return {k: float(np.clip(p + offset, 0.02, 0.98)) for k, p in curve.items()}


def _static_books():
    mids = _market_mids()
    return {m: _book_msg(mids[k]) for m, k in MARKETS}


def _cfg(mode: str, **overrides) -> MMConfig:
    kwargs = dict(
        belly_score_mode=mode,
        belly_drift_horizon_s=HORIZON_S,
        belly_drift_interval_s=INTERVAL_S,
        belly_drift_max_slack_s=SLACK_S,
        belly_drift_temper=0.3,
    )
    kwargs.update(overrides)
    return MMConfig(**kwargs)


def _make_loop(store, mode: str, **cfg_overrides) -> PaperTradingLoop:
    return PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=_cfg(mode, **cfg_overrides), clock=SimClock(START), vol_gate_fn=_vol_gate(),
        tick_dt_s=TICK_DT_S,
    )


@pytest.fixture
def store(tmp_path):
    s = MMStateStore(str(tmp_path / "mm.db"))
    yield s
    s.close()


def _run_n_ticks(loop: PaperTradingLoop, n: int) -> None:
    for _ in range(n):
        loop.tick(_static_books())


# ---------------------------------------------------------------------------
# lag buffer: append-on-BEUOY-only, fallback not appended, shape-mismatch
# drops the whole buffer
# ---------------------------------------------------------------------------


def test_lag_buffer_appends_only_on_beuoy_not_on_fallback(store):
    loop = _make_loop(store, "shadow")
    _run_n_ticks(loop, 4)
    assert loop.last_fair_value.anchor_method == AnchorMethod.BEUOY
    assert len(loop._belly_lag_buffer) == 4

    # Force a whole-anchor FIXED_BLEND_FALLBACK on tick 5 -- the tick where
    # the scoring gate fires (interval elapsed + a qualifying lag entry
    # present, derived in the cadence test below) -- degenerate belly
    # bankrolls (both weights zero) fail compute_fair_value's normalized-
    # weights check regardless of the market/pricer inputs themselves.
    loop.bankroll_states[BELLY_REGION].bankrolls = {PRICER_MODEL_ID: 0.0, MARKET_MODEL_ID: 0.0}
    loop.tick(_static_books())
    assert loop.last_fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    # Not appended -- buffer length unchanged from before this tick.
    assert len(loop._belly_lag_buffer) == 4

    # A skip row with reason "fallback" was journaled for the fallback tick.
    rows = store.get_bayes_scores()
    assert any(r.skip_reason == "fallback" and r.model_id == "" for r in rows)


def test_lag_buffer_shape_mismatch_drops_whole_buffer(store, caplog):
    loop = _make_loop(store, "shadow")
    loop.tick(_static_books())
    assert len(loop._belly_lag_buffer) == 1

    # Fabricate a result with a WRONG bucket width (no live reshape path
    # exists in the harness -- this directly unit-tests the defensive
    # check per plan S4, rather than hunting for a real reshape trigger).
    bad_result = types.SimpleNamespace(
        fair_value=types.SimpleNamespace(anchor_method=AnchorMethod.BEUOY),
        forecasts={PRICER_MODEL_ID: np.array([0.5, 0.5]), MARKET_MODEL_ID: np.array([0.5, 0.5])},
        consensus_bucket=np.array([0.5, 0.5]),  # width 2, ladder width is 4 (n=3)
    )
    loop._belly_lag_buffer_append(loop.clock.now(), bad_result)
    assert len(loop._belly_lag_buffer) == 0


# ---------------------------------------------------------------------------
# scoring-event gate: cadence + [horizon, horizon+slack] window
# ---------------------------------------------------------------------------


def test_scoring_gate_fires_only_in_window_with_expected_cadence(store):
    loop = _make_loop(store, "shadow")
    _run_n_ticks(loop, 5)  # derivation in the module docstring/plan review

    rows = store.get_bayes_scores()
    skips = [r for r in rows if r.model_id == ""]
    successes = [r for r in rows if r.model_id != ""]

    skip_reasons = [r.skip_reason for r in skips]
    assert skip_reasons == ["no_lag", "stale_lag"]  # ticks 1 and 3
    assert len(successes) == 2  # one row per model (pricer, market) at tick 5

    # Event identity: the two success rows share (ts, expiry_key).
    assert successes[0].ts == successes[1].ts
    assert successes[0].expiry_key == EXPIRY == successes[1].expiry_key
    assert {r.model_id for r in successes} == {PRICER_MODEL_ID, MARKET_MODEL_ID}
    assert successes[0].lag_s == pytest.approx(60.0)  # tick5(75s) - tick1(15s)

    # Cadence: exactly one gate attempt per belly_drift_interval_s (30s) --
    # ticks 1, 3, 5 attempted (3 distinct event ts values); ticks 2, 4 did
    # not (interval not yet elapsed).
    assert len({r.ts for r in rows}) == 3


# ---------------------------------------------------------------------------
# shadow trajectories: advance only on success, persisted under the two
# region keys, never inside self.bankroll_states, quoting-neutral
# ---------------------------------------------------------------------------


def test_shadow_trajectories_advance_only_on_success(store):
    loop = _make_loop(store, "shadow")
    for _ in range(4):  # ticks 1..4 -- no successful event yet (see gate test)
        loop.tick(_static_books())
        assert loop._belly_shadow_state is None
        assert loop._belly_control_state is None

    loop.tick(_static_books())  # tick 5 -- first successful event
    assert loop._belly_shadow_state is not None
    assert loop._belly_control_state is not None
    assert loop._belly_shadow_state.update_count == 1
    assert loop._belly_control_state.update_count == 1


def test_shadow_states_never_enter_bankroll_states_dict(store):
    loop = _make_loop(store, "shadow")
    _run_n_ticks(loop, 5)
    assert loop._belly_shadow_state is not None  # sanity: an event did fire
    assert set(loop.bankroll_states.keys()) == {BELLY_REGION, WING_REGION}
    assert "belly_drift_shadow" not in loop.bankroll_states
    assert "belly_legacy_control" not in loop.bankroll_states


def test_shadow_trajectories_persist_under_region_keys(store):
    loop = _make_loop(store, "shadow")
    _run_n_ticks(loop, 5)

    shadow_hist = store.get_bankroll_history(EXPIRY, region="belly_drift_shadow")
    control_hist = store.get_bankroll_history(EXPIRY, region="belly_legacy_control")
    assert len(shadow_hist) == 1
    assert len(control_hist) == 1
    assert shadow_hist[0].bankrolls == pytest.approx(loop._belly_shadow_state.bankrolls)
    assert control_hist[0].bankrolls == pytest.approx(loop._belly_control_state.bankrolls)

    # mm_monitor filters IN ('belly','wing') -- the legacy region rows are
    # unaffected/absent under these new region keys.
    assert store.get_bankroll_history(EXPIRY, region=BELLY_REGION)
    assert not store.get_bankroll_history(EXPIRY, region="belly_drift_shadow_typo")


def test_shadow_mode_is_quoting_neutral_vs_legacy(store, tmp_path):
    store_legacy = MMStateStore(str(tmp_path / "legacy.db"))
    try:
        loop_legacy = _make_loop(store_legacy, "legacy")
        loop_shadow = _make_loop(store, "shadow")
        for _ in range(6):
            books = _static_books()
            loop_legacy.tick(books)
            loop_shadow.tick(books)

        for region in (BELLY_REGION, WING_REGION):
            bl = loop_legacy.bankroll_states[region].bankrolls
            bs = loop_shadow.bankroll_states[region].bankrolls
            for mid in (PRICER_MODEL_ID, MARKET_MODEL_ID):
                assert bs[mid] == pytest.approx(bl[mid], rel=1e-9, abs=1e-12)
            assert (loop_legacy.bankroll_states[region].update_count
                    == loop_shadow.bankroll_states[region].update_count)

        for k in (98000.0, 100000.0, 102000.0):
            assert (loop_shadow.last_fair_value.consensus_p[k]
                    == pytest.approx(loop_legacy.last_fair_value.consensus_p[k], rel=1e-9))
    finally:
        store_legacy.close()


# ---------------------------------------------------------------------------
# journaling shape: belly_divergence sign, belly_snapshot filters to belly
# strikes only
# ---------------------------------------------------------------------------


def test_belly_divergence_sign_and_snapshot_belly_only(store):
    loop = _make_loop(store, "shadow")
    _run_n_ticks(loop, 5)

    rows = store.get_bayes_scores()
    success = [r for r in rows if r.model_id != ""]
    assert success  # sanity
    row = success[0]

    # The market book is uniformly CHEAPER than the pricer (rich-consensus
    # scenario, see _market_mids) -- consensus should sit above the market
    # at the belly strike.
    assert row.belly_divergence is not None
    assert row.belly_divergence > 0.0

    assert isinstance(row.belly_snapshot, list)
    strikes_in_snapshot = {entry[0] for entry in row.belly_snapshot}
    assert strikes_in_snapshot == {100000.0}  # only the belly strike, per _pricer_curve/SCALE
    for strike, pricer_p, market_p in row.belly_snapshot:
        assert 0.0 <= pricer_p <= 1.0
        assert 0.0 <= market_p <= 1.0


def test_gate_miss_skip_rows_carry_divergence_and_snapshot_too(store):
    loop = _make_loop(store, "shadow")
    _run_n_ticks(loop, 1)  # tick 1: gate miss, reason "no_lag"
    rows = store.get_bayes_scores()
    assert len(rows) == 1
    row = rows[0]
    assert row.skip_reason == "no_lag"
    assert row.model_id == ""
    assert row.factor_drift is None and row.factor_control is None
    assert isinstance(row.belly_snapshot, list)


# ---------------------------------------------------------------------------
# restart / resume reload
# ---------------------------------------------------------------------------


def test_restart_reloads_belly_score_states_from_store(store):
    loop = _make_loop(store, "shadow")
    _run_n_ticks(loop, 5)
    shadow_before = dict(loop._belly_shadow_state.bankrolls)
    control_before = dict(loop._belly_control_state.bankrolls)
    count_before = loop._belly_shadow_state.update_count

    new_loop = _make_loop(store, "shadow")
    new_loop.restart()
    assert new_loop._belly_shadow_state.bankrolls == pytest.approx(shadow_before)
    assert new_loop._belly_control_state.bankrolls == pytest.approx(control_before)
    assert new_loop._belly_shadow_state.update_count == count_before


def test_restart_missing_rows_defaults_to_parity(store):
    loop = _make_loop(store, "shadow")
    loop.restart()  # nothing persisted yet under either region key
    assert loop._belly_shadow_state.bankrolls == pytest.approx(
        {PRICER_MODEL_ID: 0.5, MARKET_MODEL_ID: 0.5}
    )
    assert loop._belly_control_state.bankrolls == pytest.approx(
        {PRICER_MODEL_ID: 0.5, MARKET_MODEL_ID: 0.5}
    )
    assert loop._belly_shadow_state.update_count == 0
    assert loop._belly_control_state.update_count == 0


def test_resume_attach_reloads_belly_score_states(store):
    loop = _make_loop(store, "shadow")
    _run_n_ticks(loop, 5)
    shadow_before = dict(loop._belly_shadow_state.bankrolls)

    new_loop = _make_loop(store, "shadow")
    new_loop.resume_attach(now=new_loop.clock.now(), all_fills=[])
    assert new_loop._belly_shadow_state.bankrolls == pytest.approx(shadow_before)


# ---------------------------------------------------------------------------
# live mode: applied belly update only on scoring events; wing pin
# untouched; no shadow/control trajectory rows written
# ---------------------------------------------------------------------------


def test_live_mode_belly_update_count_advances_only_on_events(store):
    loop = _make_loop(store, "live")
    for i in range(1, 5):
        loop.tick(_static_books())
        # legacy per-refresh belly branch is unconditionally skipped in
        # live mode (fair_value_anchor.py C1 section); no event has fired
        # yet through tick 4.
        assert loop.bankroll_states[BELLY_REGION].update_count == 0

    loop.tick(_static_books())  # tick 5: first successful event
    assert loop.bankroll_states[BELLY_REGION].update_count == 1


def test_live_mode_wing_pin_untouched(store):
    loop = _make_loop(store, "live")
    _run_n_ticks(loop, 5)
    pin = loop.config.wing_pricer_weight_pin
    assert loop.bankroll_states[WING_REGION].bankrolls[PRICER_MODEL_ID] == pytest.approx(pin)


def test_live_mode_writes_no_shadow_control_trajectory_rows(store):
    loop = _make_loop(store, "live")
    _run_n_ticks(loop, 5)
    assert loop._belly_shadow_state is None
    assert loop._belly_control_state is None
    assert store.get_bankroll_history(EXPIRY, region="belly_drift_shadow") == []
    assert store.get_bankroll_history(EXPIRY, region="belly_legacy_control") == []

    rows = store.get_bayes_scores()
    success = [r for r in rows if r.model_id != ""]
    assert success
    assert success[0].weight_control_after is None
    assert success[0].weight_drift_after == pytest.approx(success[0].weight_applied_after)


def test_frozen_belly_live_mode_skip_journaled_no_advance(store):
    loop = _make_loop(store, "live")
    _run_n_ticks(loop, 4)
    loop.bankroll_states[BELLY_REGION].frozen = True
    before = dict(loop.bankroll_states[BELLY_REGION].bankrolls)
    before_count = loop.bankroll_states[BELLY_REGION].update_count

    loop.tick(_static_books())  # tick 5: gate fires, anchor skips (frozen)
    assert loop.last_fair_value.anchor_method == AnchorMethod.BEUOY  # not a fallback
    assert loop.bankroll_states[BELLY_REGION].update_count == before_count
    assert loop.bankroll_states[BELLY_REGION].bankrolls == pytest.approx(before)
    # tick 5's entry still lands in the lag buffer (freeze is a per-region
    # skip, not a whole-anchor fallback).
    assert len(loop._belly_lag_buffer) == 5

    rows = store.get_bayes_scores()
    assert any(r.skip_reason == "frozen" and r.model_id == "" for r in rows)


# ---------------------------------------------------------------------------
# mode validation
# ---------------------------------------------------------------------------


def test_unknown_belly_score_mode_defaults_to_legacy_with_warning(store, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="mm.harness"):
        loop = _make_loop(store, "bogus-mode")
    assert loop._belly_score_mode == "legacy"
    assert any("belly_score_mode" in rec.getMessage() for rec in caplog.records)

    # Legacy mode never touches the buffer/gate machinery.
    _run_n_ticks(loop, 5)
    assert len(loop._belly_lag_buffer) == 0
    assert store.get_bayes_scores() == []


# ---------------------------------------------------------------------------
# paper_runner.py: prune_bayes_score_log wired into the existing prune
# cadence block (mirrors the prune_trade_prints/prune_mid_log wiring --
# there is no pre-existing runner-level prune test to literally reuse, so
# this follows the same run()-in-a-thread convention as
# tests/test_mm_paper_runner_multi.py and patches MMStateStore directly to
# assert the call happens, rather than depending on real-time retention
# windows).
# ---------------------------------------------------------------------------


def test_paper_runner_prunes_bayes_score_log(tmp_path, monkeypatch):
    import math
    import threading
    import time as time_mod

    from market_maker import paper_runner

    def _resolver(now, lead, cap, exclude):
        ek = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d")
        return [("ev-x", ek, [("x-98k", 98000.0, "tok-x-98"), ("x-102k", 102000.0, "tok-x-102")])]

    class _FakeAdapter:
        def __init__(self, tokens):
            self.tokens = tokens

        def start(self):
            pass

        def stop(self, join_timeout_s=10.0):
            pass

        def healthy(self):
            return True

        def drain(self):
            msg = [{
                "type": "snapshot",
                "bids": [(0.45, 100.0), (0.44, 100.0)],
                "asks": [(0.55, 100.0), (0.56, 100.0)],
            }]
            return {slug: list(msg) for slug in self.tokens}

    def _compute(strikes, hours_to_expiry, **kwargs):
        scale = 3000.0
        out = {float(k): float(1.0 / (1.0 + math.exp((float(k) - S0) / scale))) for k in strikes}
        out["_meta"] = {"n_sims": 1000, "S0": S0, "horizon_gate_active": False}
        return out

    class _FakeVolGate:
        regime = "normal"
        shock = False
        kelly_mult = 1.0
        edge_add_cents = 0.0

    monkeypatch.setattr(paper_runner, "resolve_events_multi", _resolver)
    monkeypatch.setattr(paper_runner, "PolymarketFeedAdapter", _FakeAdapter)
    monkeypatch.setattr(paper_runner, "_ENGINE_COMPUTE_FN", _compute)
    monkeypatch.setattr("core.strategy.vol_gate.compute_vol_gate", lambda df, now: _FakeVolGate())
    btc_csv = tmp_path / "btc_intraday_1m.csv"
    btc_csv.write_text("timestamp,close\n2026-01-01T00:00:00Z,100000.0\n", encoding="ascii")
    monkeypatch.setattr(paper_runner, "_BTC_INTRADAY_PATH", btc_csv)

    calls: List[datetime] = []
    orig_prune = MMStateStore.prune_bayes_score_log

    def _wrapped(self, cutoff_ts):
        calls.append(cutoff_ts)
        return orig_prune(self, cutoff_ts)

    monkeypatch.setattr(MMStateStore, "prune_bayes_score_log", _wrapped)

    out_dir, ctl_dir = tmp_path / "out", tmp_path / "control"
    result: Dict[str, int] = {}

    def _target():
        result["code"] = paper_runner.run([
            "--event-slug", "auto", "--max-expiries", "1", "--minutes", "0",
            "--tick-s", "0.05", "--warmup-s", "0",
            "--out", str(out_dir), "--control-dir", str(ctl_dir),
        ])

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    try:
        deadline = time_mod.time() + 30.0
        while time_mod.time() < deadline and len(calls) < 1:
            time_mod.sleep(0.05)
        assert len(calls) >= 1, "prune_bayes_score_log was never called"
        (ctl_dir / "mm_paper.stop").write_text("", encoding="ascii")
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        if t.is_alive():  # pragma: no cover
            t.join(timeout=5.0)

    # Sanity on the cutoff: roughly "now - 28 days", not e.g. "now" or an
    # unrelated value -- confirms the wiring uses BAYES_SCORE_RETENTION_S,
    # not some other constant.
    from market_maker.state_store import BAYES_SCORE_RETENTION_S
    age_s = (datetime.now(timezone.utc) - calls[0]).total_seconds()
    assert abs(age_s - BAYES_SCORE_RETENTION_S) < 3600.0
