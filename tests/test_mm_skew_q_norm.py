"""Tests for the skew q-normalization fix (2026-08-13 bleed-2 fix, item 1;
temp/mm_bleed2_fix_plan.md). The AS/GLFT skew term skew_x =
-q*gamma*sigma_b^2*tte takes q in RAW SHARES; quote_engine.py's module
docstring always specified "q is a float (caller normalizes shares by a
config unit)" but no caller ever did until this fix -- the harness now
divides both the quote-engine q and the Stage 6b unit_skew_x by
MMConfig.skew_q_norm (default 20.0) before either reaches its consumer.

Scripted synthetic feeds only, following tests/test_mm_harness_ws1.py's and
tests/test_mm_integration.py's conventions (fixed clock, deterministic
scripted books).
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import Fill, LiquiditySource, Side
from market_maker.harness import PaperTradingLoop
from market_maker.order_lifecycle import SimClock
from market_maker.quote_engine import make_quote_from_config, per_share_skew_x
from market_maker.state_store import MMStateStore

START = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
S0 = 100000.0
EXPIRY = "2026-07-06"
MARKETS = [("m-100k", 100000.0), ("m-102k", 102000.0)]


# ---------------------------------------------------------------------------
# scripted stubs (mirrors test_mm_harness_ws1.py / test_mm_integration.py)
# ---------------------------------------------------------------------------


def _engine(s0=S0, scale=2000.0, n_sims=15000):
    import numpy as np

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


@pytest.fixture
def store(tmp_path):
    s = MMStateStore(str(tmp_path / "mm.db"))
    yield s
    s.close()


def _build_long_position(loop, market_id, now, q_shares):
    """Grow `market_id`'s inventory to q_shares long YES via the normal fill
    channel (loop.inv.apply_fill), mirroring tests/test_mm_harness_ws1.py's
    _build_long_position helper."""
    f = Fill(
        ts=now, market_id=market_id, order_id="scripted:%s" % market_id,
        side=Side.BUY_YES, price=0.5, size=q_shares,
        liquidity=LiquiditySource.MAKER, venue_ts=now,
    )
    loop.inv.apply_fill(f)


# ---------------------------------------------------------------------------
# (a) harness-level: journaled skew_x matches the normalized formula, and
# the r_x == x_fair + skew_x identity holds. Spies on the harness's own call
# to make_quote_from_config (not a reimplementation) to prove the CALL SITE
# divides q by skew_q_norm before the quote engine ever sees it.
# ---------------------------------------------------------------------------


def test_harness_journaled_skew_matches_normalized_formula_default_norm(store):
    import market_maker.harness as harness_mod

    market_id, _strike = MARKETS[0]
    cfg = MMConfig(gamma=0.5, k_arrival=1.0)  # skew_q_norm default 20.0
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    now = loop.clock.now()
    q_shares = 10.0
    _build_long_position(loop, market_id, now, q_shares)

    orig = harness_mod.make_quote_from_config
    calls = {}

    def spy(config, mid, x_fair, q_eff, tte_days, sigma_b, **kwargs):
        calls[mid] = (x_fair, q_eff, tte_days, sigma_b)
        return orig(config, mid, x_fair, q_eff, tte_days, sigma_b, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(harness_mod, "make_quote_from_config", spy)
        loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})

    assert market_id in calls
    x_fair, q_eff, tte_days, sigma_b = calls[market_id]

    # Call-site proof: the harness passed q/skew_q_norm, NOT raw q.
    assert q_eff == pytest.approx(q_shares / cfg.skew_q_norm)

    prop = loop.last_proposals[market_id]
    expected_skew = -(q_shares / cfg.skew_q_norm) * cfg.gamma * (sigma_b ** 2) * tte_days
    assert abs(expected_skew) < cfg.skew_x_cap, "test setup must not pin the cap"
    assert prop.skew_x == pytest.approx(expected_skew)
    # Identity (module docstring): x_fair == r_x - skew_x, i.e. r_x == x_fair + skew_x.
    assert prop.r_x == pytest.approx(x_fair + prop.skew_x)


# ---------------------------------------------------------------------------
# (b) skew_q_norm=1.0 (legacy kill switch) -> the harness proposal is
# byte-identical to a DIRECT make_quote_from_config call using the RAW
# (un-normalized) q. This only holds at skew_q_norm=1.0 -- at the default
# (20.0) the harness's q_eff (q/20) would diverge from the raw-q direct call
# used here, so this test genuinely exercises the kill switch, not a
# tautology.
# ---------------------------------------------------------------------------


def test_skew_q_norm_one_matches_legacy_raw_q_behavior(store):
    market_id, _strike = MARKETS[0]
    cfg = MMConfig(gamma=0.5, k_arrival=1.0, skew_q_norm=1.0)
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    now = loop.clock.now()
    q_shares = 10.0
    _build_long_position(loop, market_id, now, q_shares)

    loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})
    prop = loop.last_proposals[market_id]
    tte = max(loop.last_snapshot.tte_days, 0.0)

    # Recover x_fair via the quote engine's own guaranteed identity
    # (r_x - skew_x == x_fair, holds under the cap too -- module docstring).
    x_fair = prop.r_x - prop.skew_x

    direct = make_quote_from_config(
        cfg, market_id, x_fair, q_shares, tte, prop.sigma_b,
        variant=loop.quote_variant, ts=now,
    )
    assert prop.r_x == pytest.approx(direct.r_x)
    assert prop.skew_x == pytest.approx(direct.skew_x)
    assert prop.x_bid == pytest.approx(direct.x_bid)
    assert prop.x_ask == pytest.approx(direct.x_ask)


# ---------------------------------------------------------------------------
# (c) ContractSizingInput.unit_skew_x threading: mirrors the
# tests/test_mm_harness_ws1.py threading-test pattern (spy on size_ladder to
# capture the ContractSizingInput list built this tick).
# ---------------------------------------------------------------------------


def test_contract_sizing_input_unit_skew_x_normalized_by_skew_q_norm(store):
    import market_maker.harness as harness_mod

    cfg = MMConfig(gamma=0.5, k_arrival=1.0)
    assert cfg.skew_q_norm == 20.0
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )

    captured = {}
    orig_size_ladder = harness_mod.size_ladder

    def spy(*args, **kwargs):
        captured["contracts"] = args[0]
        return orig_size_ladder(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(harness_mod, "size_ladder", spy)
        loop.tick({m: _snapshot_msg(0.5) for m, _k in MARKETS})

    contracts = captured.get("contracts")
    assert contracts, "expected at least one ContractSizingInput"
    tte = max(loop.last_snapshot.tte_days, 0.0)
    for c in contracts:
        prop = loop.last_proposals[c.market_id]
        raw = per_share_skew_x(
            loop.quote_variant, prop.sigma_b, cfg.gamma, cfg.k_arrival,
            cfg.arrival_scale_A, tte,
        )
        assert c.unit_skew_x == pytest.approx(raw / cfg.skew_q_norm)


# ---------------------------------------------------------------------------
# (d) Defaults replay (plan-stated magnitude): q=5 shares, sigma_b=1.7,
# tte=1.0d at config defaults -> |skew_x| < 0.08 and the cap does not pin.
# Pure formula check (no harness -- sigma_b/tte are not independently
# controllable through the estimator, so the plan's own numbers are
# replayed directly).
# ---------------------------------------------------------------------------


def test_defaults_replay_five_shares_stays_well_under_cap():
    cfg = MMConfig()
    assert cfg.skew_q_norm == 20.0
    q_shares, sigma_b, tte_days = 5.0, 1.7, 1.0
    per_share = per_share_skew_x(
        "dalen", sigma_b, cfg.gamma, cfg.k_arrival, cfg.arrival_scale_A, tte_days
    )
    skew_x = -(q_shares / cfg.skew_q_norm) * per_share
    assert abs(skew_x) < 0.08
    assert abs(skew_x) < cfg.skew_x_cap  # cap not pinned


# ---------------------------------------------------------------------------
# (e) Extreme-sigma residual (plan-mandated -- do NOT write a "never binds"
# test): at the config sigma_b_cap (5.0) and tte=4d, Stage 6b's q_skew_max
# still binds hard (3 shares). Pure math check, mirrors
# temp/mm_bleed2_fix_plan.md's own worked numbers.
# ---------------------------------------------------------------------------


def test_extreme_sigma_residual_stage_6b_still_binds():
    cfg = MMConfig()
    sigma_b, tte_days = 5.0, 4.0
    raw = per_share_skew_x(
        "dalen", sigma_b, cfg.gamma, cfg.k_arrival, cfg.arrival_scale_A, tte_days
    )
    unit_skew_x = raw / cfg.skew_q_norm
    assert unit_skew_x == pytest.approx(0.5)
    q_skew_max = cfg.skew_q_headroom_mult * cfg.skew_x_cap / unit_skew_x
    assert q_skew_max == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# (f) Guard: skew_q_norm <= 0 (incl. NaN via the `> 0` test) -> the harness
# resolves to 1.0 (legacy raw-share behavior) and warns exactly once at
# __init__ (precedent: sizing_region_basis validation).
# ---------------------------------------------------------------------------


def test_invalid_skew_q_norm_resolves_to_one_with_one_warning(store, caplog):
    for bad in (0.0, -5.0, float("nan")):
        with caplog.at_level(logging.WARNING, logger="mm.harness"):
            loop = PaperTradingLoop(
                store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
                config=MMConfig(skew_q_norm=bad), clock=SimClock(START), vol_gate_fn=_vol_gate(),
            )
        assert loop._skew_q_norm == 1.0
        warnings = [r for r in caplog.records if "skew_q_norm" in r.getMessage()]
        assert len(warnings) == 1
        caplog.clear()
