"""Tests for market_maker.robustness_sizing (plan task Z1)."""
from __future__ import annotations

from datetime import datetime, timezone

from market_maker.config import MMConfig
from market_maker.contracts import (
    ConfidenceTier,
    LiquidityRegime,
    LiquidityState,
    PricerSnapshot,
    Sigma2Source,
    SizingCap,
)
from market_maker.robustness_sizing import (
    ContractSizingInput,
    baker_mchale,
    kelly_buy,
    size_ladder,
)

TS = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


def _snap(sigma2_ladder: float) -> PricerSnapshot:
    return PricerSnapshot(
        ts=TS,
        expiry_key="2026-08-01",
        tte_days=10.0,
        s0=100000.0,
        n_sims=15000,
        strikes=[100000.0],
        grid_strikes=[100000.0],
        p_hat={100000.0: 0.5},
        p_grid={100000.0: 0.5},
        sigma2={100000.0: sigma2_ladder},
        sigma2_ladder=sigma2_ladder,
        sigma2_source=Sigma2Source.MC,
        confidence_tier=ConfidenceTier.FULL,
        horizon_gate_active=False,
        stale=False,
    )


def _liq(market_id, dbid, dask):
    return LiquidityState(
        ts=TS,
        market_id=market_id,
        realized_depth_bid=dbid,
        realized_depth_ask=dask,
        kyle_lambda=None,
        arb_halflife_s=None,
        regime=LiquidityRegime.NORMAL,
        window="w",
    )


# --- Stage 1: Kelly golden values ----------------------------------------


def test_kelly_golden_yes():
    f, b = kelly_buy(0.6, 0.5)
    assert abs(b - 1.0) < 1e-12
    assert abs(f - 0.2) < 1e-12


def test_kelly_no_symmetric():
    # NO leg: ask_price=0.5, p_hat=0.4 -> NO belief 0.6 at NO price 0.5 -> f=0.2
    f, b = kelly_buy(1.0 - 0.4, 1.0 - 0.5)
    assert abs(b - 1.0) < 1e-12
    assert abs(f - 0.2) < 1e-12


def test_kelly_negative_floored():
    f, _ = kelly_buy(0.4, 0.5)
    assert f == 0.0


# --- Stage 2: Baker-McHale -----------------------------------------------


def test_baker_mchale_monotone_and_sigma0():
    assert baker_mchale(0.2, 1.0, 0.0) == 1.0
    ks = [baker_mchale(0.2, 1.0, s) for s in (0.0, 0.01, 0.1, 1.0)]
    for a, b in zip(ks, ks[1:]):
        assert b < a  # strictly decreasing in sigma2


def test_sigma2_zero_still_capped_by_fractional_c():
    # sigma2=0 -> k_shrink=1, so the c<=0.5 ceiling must be the binding control.
    c = ContractSizingInput("m0", p_hat=0.6, bid_price=0.5, ask_price=0.55)
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    d = dec["m0"]
    assert d.k_shrink == 1.0
    assert SizingCap.FRACTIONAL_C in d.caps_applied
    # f=0.2, c=0.5 -> 0.1; size = 0.1*1000/0.5 = 200 (full-Kelly would be 400)
    assert abs(d.bid_size - 200.0) < 1e-6


# --- Stage 3: joint ladder allocation ------------------------------------


def test_joint_ladder_sum_bounded():
    # 3 strikes, each YES f*=0.1 (p_hat=0.55, bid=0.5); NO legs inert (ask=0.5).
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.55, bid_price=0.5, ask_price=0.5)
        for i in range(3)
    ]
    dec, audit = size_ladder(cs, _snap(0.0), bankroll=1000.0, ts=TS)
    total_alloc = sum(d.ladder_alloc for d in dec.values())
    assert total_alloc <= 0.1 + 1e-9  # <= single largest unscaled fraction
    assert SizingCap.LADDER_JOINT in dec["m0"].caps_applied


# --- Stage 4: ruin / bankroll caps ---------------------------------------


def test_ruin_cap_binds():
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.9, bid_price=0.5, ask_price=0.5)
        for i in range(5)
    ]
    dec, _ = size_ladder(cs, _snap(0.0), bankroll=1000.0, ts=TS)
    assert SizingCap.RUIN in dec["m0"].caps_applied


def test_bankroll_cap_binds():
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.9, bid_price=0.5, ask_price=0.5)
        for i in range(5)
    ]
    dec, _ = size_ladder(
        cs, _snap(0.0), bankroll=1000.0, ts=TS,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.3,
    )
    assert SizingCap.BANKROLL in dec["m0"].caps_applied


# --- Depth cap ------------------------------------------------------------


def test_depth_cap_binds_and_inert_when_absent():
    c = ContractSizingInput("m0", p_hat=0.6, bid_price=0.5, ask_price=0.55)
    # inert when liquidity absent
    dec0, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    assert SizingCap.DEPTH not in dec0["m0"].caps_applied
    assert dec0["m0"].bid_size > 5.0
    # binds when depth small
    dec1, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS,
        liquidity={"m0": _liq("m0", 5.0, 5.0)},
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    assert SizingCap.DEPTH in dec1["m0"].caps_applied
    assert dec1["m0"].bid_size == 5.0


# --- Negative edge -> zero size ------------------------------------------


def test_negative_edge_zero_both_sides():
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.55, ask_price=0.45)
    dec, _ = size_ladder([c], _snap(0.0), bankroll=1000.0, ts=TS)
    assert dec["m0"].bid_size == 0.0
    assert dec["m0"].ask_size == 0.0


# --- caps ordering: FRACTIONAL_C last and always present -----------------


def test_fractional_c_always_last_and_present():
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.9, bid_price=0.5, ask_price=0.5)
        for i in range(3)
    ]
    dec, _ = size_ladder(
        cs, _snap(0.0), bankroll=1000.0, ts=TS,
        liquidity={"m0": _liq("m0", 1.0, 1.0)},
    )
    for d in dec.values():
        assert SizingCap.FRACTIONAL_C in d.caps_applied
        assert d.caps_applied[-1] == SizingCap.FRACTIONAL_C
