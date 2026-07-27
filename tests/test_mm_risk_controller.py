"""Tests for market_maker.risk_controller (plan 2.10, task R1, contract 4.10).

Trigger matrix, hysteresis latching (no flapping), one-sided inventory mapping,
staleness escalation, journal, and vol-gate passthrough.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import (
    ConfidenceTier,
    LiquidityRegime,
    PricerSnapshot,
    QuoteMode,
    RiskTrigger,
    Sigma2Source,
)
from market_maker.risk_controller import InvBreach, RiskController

T0 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


@dataclass
class StubVG:
    regime: str = "normal"
    shock: bool = False
    kelly_mult: float = 1.0
    edge_add_cents: float = 0.0
    allow_new_entries: bool = True


def _snap(now, age_s=0.0, tte=5.0):
    ts = now - timedelta(seconds=age_s)
    return PricerSnapshot(
        ts=ts, expiry_key="2026-07-20", tte_days=tte, s0=100000.0, n_sims=15000,
        strikes=[100000.0], grid_strikes=[100000.0],
        p_hat={100000.0: 0.5}, p_grid={100000.0: 0.5},
        sigma2={100000.0: 1e-5}, sigma2_ladder=1e-5,
        sigma2_source=Sigma2Source.MC, confidence_tier=ConfidenceTier.FULL,
        horizon_gate_active=False, stale=False,
    )


def _rc(latch=0.0):
    return RiskController(MMConfig(), latch_seconds=latch)


def _eval(rc, now=T0, *, tte=5.0, age=0.0, vg=None, breaches=None,
          liq=LiquidityRegime.NORMAL, feed=True, spot=None, strike=None,
          manual=False, fv_age=None, inventory_q=None, mid_move_p=None):
    return rc.evaluate(
        "m1", now, tte_days=tte, pricer_snapshot=_snap(now, age, tte),
        inventory_breaches=breaches, inventory_q=inventory_q,
        liquidity_regime=liq, feed_healthy=feed,
        spot=spot, strike=strike, manual_override=manual,
        vol_gate_result=vg if vg is not None else StubVG(),
        fair_value_age_s=fv_age, mid_move_p=mid_move_p,
    )


# ---------------------------------------------------------------------------
# Trigger matrix -- each trigger alone.
# ---------------------------------------------------------------------------

def test_a_vol_extreme_pulls():
    d = _eval(_rc(), vg=StubVG(regime="extreme", kelly_mult=0.0))
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.SPOT_JUMP in d.triggers


def test_a_vol_shock_pulls():
    d = _eval(_rc(), vg=StubVG(regime="normal", shock=True, kelly_mult=0.0))
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.SPOT_JUMP in d.triggers


def test_a_vol_high_widens_not_pull():
    d = _eval(_rc(), vg=StubVG(regime="high", edge_add_cents=2.0, kelly_mult=0.5))
    assert d.mode == QuoteMode.TWO_SIDED
    assert d.eps_add == pytest.approx(0.02)
    assert d.kelly_mult == pytest.approx(0.5)
    assert RiskTrigger.SPOT_JUMP in d.triggers


def test_b_near_resolution_pulls():
    # tte=0.2d = 4.8h, inside the 6h default window (24h -> 6h, 2026-07-11).
    d = _eval(_rc(), tte=0.2)
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.NEAR_RESOLUTION in d.triggers


def test_b_near_resolution_no_pull_outside_window():
    # tte=0.5d = 12h was PULLED under the old 24h default; must quote now.
    d = _eval(_rc(), tte=0.5)
    assert d.mode == QuoteMode.TWO_SIDED
    assert RiskTrigger.NEAR_RESOLUTION not in d.triggers


def test_b_gap_through_strike_pulls():
    d = _eval(_rc(), vg=StubVG(regime="high", edge_add_cents=2.0, kelly_mult=0.5),
              spot=100000.0, strike=100000.0)
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.SPOT_GAPPING_STRIKE in d.triggers


def test_b_gap_through_inert_when_vol_calm():
    # Spot on strike but vol normal -> no gap-through pull.
    d = _eval(_rc(), spot=100000.0, strike=100000.0)
    assert d.mode == QuoteMode.TWO_SIDED
    assert RiskTrigger.SPOT_GAPPING_STRIKE not in d.triggers


def test_c_long_breach_ask_only():
    d = _eval(_rc(), breaches=[InvBreach("m1", is_long=True, ratio=1.2)])
    assert d.mode == QuoteMode.ASK_ONLY
    assert RiskTrigger.INV_CAP in d.triggers


def test_c_short_breach_bid_only():
    d = _eval(_rc(), breaches=[InvBreach("m1", is_long=False, ratio=1.2)])
    assert d.mode == QuoteMode.BID_ONLY
    assert RiskTrigger.INV_CAP in d.triggers


def test_c_extreme_breach_one_sided():
    # Stranded-inventory fix 2026-07-14: ANY breach ratio (even > 1.5,
    # formerly "extreme") emits the one-sided away mode, never PULLED -- a
    # one-sided-away mode never adds risk, so escalating to PULLED only
    # removed the unwind path.
    d = _eval(_rc(), breaches=[InvBreach("m1", is_long=True, ratio=1.6)])
    assert d.mode == QuoteMode.ASK_ONLY
    assert d.mode != QuoteMode.PULLED
    assert d.cancel_all is False
    assert RiskTrigger.INV_CAP in d.triggers


def test_c_extreme_breach_one_sided_short():
    # Short twin of the above: extreme short breach -> BID_ONLY, not PULLED.
    d = _eval(_rc(), breaches=[InvBreach("m1", is_long=False, ratio=1.6)])
    assert d.mode == QuoteMode.BID_ONLY
    assert d.mode != QuoteMode.PULLED
    assert d.cancel_all is False
    assert RiskTrigger.INV_CAP in d.triggers


def test_c_breach_other_market_ignored():
    d = _eval(_rc(), breaches=[InvBreach("m2", is_long=True, ratio=1.6)])
    assert d.mode == QuoteMode.TWO_SIDED


def test_d_feed_loss_pulls_and_cancel_all():
    d = _eval(_rc(), feed=False)
    assert d.mode == QuoteMode.PULLED
    assert d.cancel_all is True
    assert RiskTrigger.FEED_STALE in d.triggers


def test_e_pricer_stale_widens_then_pulls():
    # Between 1x and 2x max age -> widen only.
    d1 = _eval(_rc(), age=400.0)  # max_age 300
    assert d1.mode == QuoteMode.TWO_SIDED
    assert d1.eps_add == pytest.approx(0.01)
    assert RiskTrigger.PRICER_STALE in d1.triggers
    # Beyond 2x max age -> pull.
    d2 = _eval(_rc(), age=700.0)
    assert d2.mode == QuoteMode.PULLED
    assert RiskTrigger.PRICER_STALE in d2.triggers


def test_f_liquidity_degenerate_pulls():
    d = _eval(_rc(), liq=LiquidityRegime.DEGENERATE)
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.LIQ_DEGENERATE in d.triggers


# ---------------------------------------------------------------------------
# (f) rule update -- stranded-inventory fix 2026-07-14: DEGENERATE liquidity
# with real (non-dust) inventory quotes the reduce-only side instead of
# pulling entirely. The default inventory_q=None (test above) preserves the
# pre-fix PULLED behavior byte-identically.
# ---------------------------------------------------------------------------


def test_f_degenerate_with_long_inventory_ask_only():
    d = _eval(_rc(), liq=LiquidityRegime.DEGENERATE, inventory_q=3.0)
    assert d.mode == QuoteMode.ASK_ONLY
    assert RiskTrigger.LIQ_DEGENERATE in d.triggers


def test_f_degenerate_with_short_inventory_bid_only():
    d = _eval(_rc(), liq=LiquidityRegime.DEGENERATE, inventory_q=-3.0)
    assert d.mode == QuoteMode.BID_ONLY
    assert RiskTrigger.LIQ_DEGENERATE in d.triggers


def test_f_degenerate_with_flat_inventory_pulls():
    # inventory_q=0.0 (flat, not None) is still "no real inventory" -- dust
    # threshold, not None-ness, gates the reduce-only branch.
    d = _eval(_rc(), liq=LiquidityRegime.DEGENERATE, inventory_q=0.0)
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.LIQ_DEGENERATE in d.triggers


# ---------------------------------------------------------------------------
# Co-fire: rules (c)/(f) agree on side (no accidental PULLED escalation),
# but (d)/(b) still win when they co-fire with a one-sided (c)/(f) result.
# ---------------------------------------------------------------------------


def test_cofire_degenerate_and_extreme_breach_agree_no_escalation():
    # DEGENERATE (f) + extreme short breach (c), both derived from the same
    # short inventory_q -- agreeing sides (BID_ONLY) must not escalate to
    # PULLED via _more_restrictive's opposite-sides rule.
    d = _eval(_rc(), liq=LiquidityRegime.DEGENERATE, inventory_q=-3.0,
              breaches=[InvBreach("m1", is_long=False, ratio=1.6)])
    assert d.mode == QuoteMode.BID_ONLY
    assert RiskTrigger.LIQ_DEGENERATE in d.triggers
    assert RiskTrigger.INV_CAP in d.triggers


def test_cofire_degenerate_with_inventory_feed_loss_still_pulls():
    # Rule (d) feed loss is mandatory and ranks above one-sided -- wins even
    # with real inventory present.
    d = _eval(_rc(), liq=LiquidityRegime.DEGENERATE, inventory_q=-3.0, feed=False)
    assert d.mode == QuoteMode.PULLED
    assert d.cancel_all is True
    assert RiskTrigger.FEED_STALE in d.triggers


def test_cofire_degenerate_with_inventory_near_resolution_still_pulls():
    # Rule (b) near-resolution wins over the (f) reduce-only side even with
    # real inventory present. tte=0.2d = 4.8h, inside the 6h default window.
    d = _eval(_rc(), liq=LiquidityRegime.DEGENERATE, inventory_q=-3.0, tte=0.2)
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.NEAR_RESOLUTION in d.triggers


# ---------------------------------------------------------------------------
# (g) fair-value staleness (plan Wave 1 W1.2) -- mirrors rule (e).
# ---------------------------------------------------------------------------


def test_g_fair_value_stale_inert_when_none():
    # Default fair_value_age_s=None must not affect behavior at all.
    d = _eval(_rc())
    assert d.mode == QuoteMode.TWO_SIDED
    assert RiskTrigger.FAIR_VALUE_STALE not in d.triggers
    assert d.eps_add == pytest.approx(0.0)


def test_g_fair_value_stale_widens_between_1x_and_2x_max_age():
    cfg = MMConfig()
    d = _eval(_rc(), fv_age=cfg.fv_max_age_s + 50.0)
    assert d.mode == QuoteMode.TWO_SIDED
    assert d.eps_add == pytest.approx(0.01)  # same widen constant as pricer-stale
    assert RiskTrigger.FAIR_VALUE_STALE in d.triggers


def test_g_fair_value_stale_pulls_beyond_2x_max_age():
    cfg = MMConfig()
    d = _eval(_rc(), fv_age=2.0 * cfg.fv_max_age_s + 1.0)
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.FAIR_VALUE_STALE in d.triggers


def test_g_fair_value_stale_under_threshold_inert():
    cfg = MMConfig()
    d = _eval(_rc(), fv_age=cfg.fv_max_age_s - 1.0)
    assert d.mode == QuoteMode.TWO_SIDED
    assert RiskTrigger.FAIR_VALUE_STALE not in d.triggers


# ---------------------------------------------------------------------------
# (h) ladder mid-velocity pull (Fix 3, 2026-07-26). mid_move_p > mid_move_pull_p
# pulls flat, reduce-only when positioned; None / knob<=0 / sub-threshold inert.
# ---------------------------------------------------------------------------


def test_h_mid_velocity_inert_when_none():
    # Default mid_move_p=None must not affect behavior at all.
    d = _eval(_rc())
    assert d.mode == QuoteMode.TWO_SIDED
    assert RiskTrigger.MID_VELOCITY not in d.triggers


def test_h_mid_velocity_inert_below_threshold():
    # Move under the 0.04 default knob -> inert.
    d = _eval(_rc(), mid_move_p=0.02)
    assert d.mode == QuoteMode.TWO_SIDED
    assert RiskTrigger.MID_VELOCITY not in d.triggers


def test_h_mid_velocity_inert_when_knob_disabled():
    # mid_move_pull_p <= 0 disables the rule even for a large move.
    rc = RiskController(MMConfig(mid_move_pull_p=0.0), latch_seconds=0.0)
    d = _eval(rc, mid_move_p=0.5)
    assert d.mode == QuoteMode.TWO_SIDED
    assert RiskTrigger.MID_VELOCITY not in d.triggers


def test_h_mid_velocity_pull_flat_inventory_none():
    # inventory_q default None (unknown) -> full PULLED.
    d = _eval(_rc(), mid_move_p=0.10)
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.MID_VELOCITY in d.triggers


def test_h_mid_velocity_pull_flat_inventory_zero():
    # inventory_q=0.0 (flat, not None) is still "no real inventory" -- dust
    # threshold, not None-ness, gates the reduce-only branch (mirrors rule f).
    d = _eval(_rc(), mid_move_p=0.10, inventory_q=0.0)
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.MID_VELOCITY in d.triggers


def test_h_mid_velocity_reduce_only_long():
    # Long (q>0) -> ASK_ONLY so the unwind path survives the burst.
    d = _eval(_rc(), mid_move_p=0.10, inventory_q=5.0)
    assert d.mode == QuoteMode.ASK_ONLY
    assert RiskTrigger.MID_VELOCITY in d.triggers


def test_h_mid_velocity_reduce_only_short():
    # Short (q<0) -> BID_ONLY.
    d = _eval(_rc(), mid_move_p=0.10, inventory_q=-5.0)
    assert d.mode == QuoteMode.BID_ONLY
    assert RiskTrigger.MID_VELOCITY in d.triggers


def test_h_mid_velocity_latches_through_burst():
    # The pull holds through the 60s latch after the instantaneous move drops
    # back under threshold (burst-continuation protection).
    rc = _rc(latch=60.0)
    d0 = _eval(rc, now=T0, mid_move_p=0.10)  # flat inventory -> PULLED
    assert d0.mode == QuoteMode.PULLED
    d1 = _eval(rc, now=T0 + timedelta(seconds=10), mid_move_p=0.0)  # move cleared, latched
    assert d1.mode == QuoteMode.PULLED
    d2 = _eval(rc, now=T0 + timedelta(seconds=61), mid_move_p=0.0)  # latch expired -> release
    assert d2.mode == QuoteMode.TWO_SIDED


def test_h_cofire_with_breach_same_sign_no_escalation():
    # Long breach (c) -> ASK_ONLY and mid-velocity (h) with the SAME long
    # inventory_q -> ASK_ONLY. Agreeing sides must not escalate to PULLED via
    # _more_restrictive's opposite-sides rule (same signed-q basis as c/f).
    d = _eval(_rc(), breaches=[InvBreach("m1", is_long=True, ratio=1.2)],
              inventory_q=3.0, mid_move_p=0.10)
    assert d.mode == QuoteMode.ASK_ONLY
    assert d.mode != QuoteMode.PULLED
    assert RiskTrigger.INV_CAP in d.triggers
    assert RiskTrigger.MID_VELOCITY in d.triggers


def test_manual_override_pulls():
    d = _eval(_rc(), manual=True)
    assert d.mode == QuoteMode.PULLED
    assert RiskTrigger.MANUAL in d.triggers


# ---------------------------------------------------------------------------
# Combinations -- most restrictive wins; feed loss always pulls + cancel_all.
# ---------------------------------------------------------------------------

def test_combo_most_restrictive_wins():
    # High widen (TWO_SIDED) + long breach (ASK_ONLY) -> ASK_ONLY, widen kept.
    d = _eval(_rc(), vg=StubVG(regime="high", edge_add_cents=2.0, kelly_mult=0.5),
              breaches=[InvBreach("m1", is_long=True, ratio=1.2)])
    assert d.mode == QuoteMode.ASK_ONLY
    assert d.eps_add == pytest.approx(0.02)


def test_combo_feed_loss_always_pulls():
    # Even with only a one-sided inventory reason, feed loss forces PULLED+cancel.
    d = _eval(_rc(), breaches=[InvBreach("m1", is_long=True, ratio=1.2)], feed=False)
    assert d.mode == QuoteMode.PULLED
    assert d.cancel_all is True


# ---------------------------------------------------------------------------
# Hysteresis latching -- no flapping.
# ---------------------------------------------------------------------------

def test_hysteresis_latches_until_expiry():
    rc = _rc(latch=60.0)
    d0 = rc.evaluate("m1", T0, tte_days=5.0, pricer_snapshot=_snap(T0),
                     feed_healthy=False, vol_gate_result=StubVG())
    assert d0.mode == QuoteMode.PULLED

    # Trigger cleared 10s later, but latch still holds PULLED.
    d1 = rc.evaluate("m1", T0 + timedelta(seconds=10), tte_days=5.0,
                     pricer_snapshot=_snap(T0 + timedelta(seconds=10)),
                     feed_healthy=True, vol_gate_result=StubVG())
    assert d1.mode == QuoteMode.PULLED

    # After the latch expires and the trigger is clear -> release.
    d2 = rc.evaluate("m1", T0 + timedelta(seconds=61), tte_days=5.0,
                     pricer_snapshot=_snap(T0 + timedelta(seconds=61)),
                     feed_healthy=True, vol_gate_result=StubVG())
    assert d2.mode == QuoteMode.TWO_SIDED


def test_hysteresis_no_flapping_on_oscillation():
    rc = _rc(latch=60.0)
    times_feed = [(0, False), (5, True), (10, False), (12, True), (20, True)]
    modes = []
    for secs, feed in times_feed:
        d = rc.evaluate("m1", T0 + timedelta(seconds=secs), tte_days=5.0,
                        pricer_snapshot=_snap(T0 + timedelta(seconds=secs)),
                        feed_healthy=feed, vol_gate_result=StubVG())
        modes.append(d.mode)
    # Never flaps back to TWO_SIDED inside the latch window.
    assert all(m == QuoteMode.PULLED for m in modes)
    # Exactly one journaled transition (into PULLED).
    assert len(rc.journal()) == 1


# ---------------------------------------------------------------------------
# Journal + passthrough.
# ---------------------------------------------------------------------------

def test_journal_records_transitions_with_triggers():
    rc = _rc(latch=0.0)
    _eval(rc, now=T0, feed=False)                       # -> PULLED
    _eval(rc, now=T0 + timedelta(seconds=1))            # -> TWO_SIDED
    _eval(rc, now=T0 + timedelta(seconds=2),
          breaches=[InvBreach("m1", is_long=True, ratio=1.2)])  # -> ASK_ONLY
    j = rc.journal()
    assert len(j) == 3
    assert j[0][2] == QuoteMode.TWO_SIDED and j[0][3] == QuoteMode.PULLED
    assert RiskTrigger.FEED_STALE in j[0][4]
    assert j[1][3] == QuoteMode.TWO_SIDED
    assert j[2][3] == QuoteMode.ASK_ONLY
    assert RiskTrigger.INV_CAP in j[2][4]


def test_vol_gate_passthrough_kelly_and_eps():
    d = _eval(_rc(), vg=StubVG(regime="high", edge_add_cents=3.0, kelly_mult=0.42))
    assert d.kelly_mult == pytest.approx(0.42)
    assert d.eps_add == pytest.approx(0.03)


def test_injectable_vol_gate_fn():
    # The vol gate is an injectable callable (tests stub it).
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return StubVG(regime="extreme", kelly_mult=0.0)

    rc = RiskController(MMConfig(), vol_gate_fn=fn, latch_seconds=0.0)
    d = rc.evaluate("m1", T0, tte_days=5.0, pricer_snapshot=_snap(T0),
                    feed_healthy=True)
    assert calls["n"] == 1
    assert d.mode == QuoteMode.PULLED
