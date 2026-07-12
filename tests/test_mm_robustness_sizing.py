"""Tests for market_maker.robustness_sizing (plan task Z1; rewritten per
temp/mm_sizing_fix_plan.md C1-C5 / Section 4, 2026-07-12)."""
from __future__ import annotations

from datetime import datetime, timezone

from market_maker.config import MMConfig
from market_maker.contracts import (
    ConfidenceTier,
    ContractInv,
    InventoryState,
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


def _snap(sigma2_ladder, strikes=None, sigma2_by_strike=None, s0=100000.0):
    """Build a PricerSnapshot. Single-strike by default (back-compat with the
    original fixture); pass `strikes` (+ optional `sigma2_by_strike`) for a
    multi-strike ladder snapshot (plan Section 4 fixture note, SG-4)."""
    strikes = strikes if strikes is not None else [100000.0]
    sigma2_by_strike = sigma2_by_strike or {}
    sigma2 = {k: sigma2_by_strike.get(k, sigma2_ladder) for k in strikes}
    p_hat = {k: 0.5 for k in strikes}
    return PricerSnapshot(
        ts=TS,
        expiry_key="2026-08-01",
        tte_days=10.0,
        s0=s0,
        n_sims=15000,
        strikes=list(strikes),
        grid_strikes=list(strikes),
        p_hat=p_hat,
        p_grid=p_hat,
        sigma2=sigma2,
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


def _inv(per_contract):
    """per_contract: dict market_id -> (q, q_max)."""
    return InventoryState(
        ts=TS,
        per_contract={
            m: ContractInv(q=q, avg_cost=0.5, q_max=q_max, age_weighted_holding=0.0)
            for m, (q, q_max) in per_contract.items()
        },
        per_ladder={},
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
        config=MMConfig(presence_frac=0.0),
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    d = dec["m0"]
    assert d.k_shrink == 1.0
    assert SizingCap.FRACTIONAL_C in d.caps_applied
    # f=0.2, c=0.5 -> 0.1; size = 0.1*1000/0.5 = 200 (full-Kelly would be 400)
    assert abs(d.bid_size - 200.0) < 1e-6


# --- Stage 2/C4: per-strike Baker-McHale sigma2 ---------------------------


def test_per_strike_sigma2_wing_shrinks_more_than_atm():
    # Same directional edge (p_hat=0.7, mkt_mid=0.5) at both strikes; only the
    # per-strike sigma2 differs -- the wing's large parameter uncertainty must
    # shrink its Kelly fraction (k_shrink) more than the ATM leg's tiny one.
    strikes = [98000.0, 100000.0]
    snap = _snap(
        sigma2_ladder=0.05, strikes=strikes,
        sigma2_by_strike={98000.0: 0.05, 100000.0: 0.0001},
    )
    cs = [
        ContractSizingInput("wing", p_hat=0.7, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=98000.0),
        ContractSizingInput("atm", p_hat=0.7, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=100000.0),
    ]
    dec, _ = size_ladder(
        cs, snap, bankroll=1000.0, ts=TS,
        config=MMConfig(presence_frac=0.0, fractional_kelly_c=1.0),
        per_expiry_cap_frac=0.9, bankroll_util_cap=10.0,
    )
    assert dec["wing"].k_shrink < dec["atm"].k_shrink
    assert dec["wing"].bid_size < dec["atm"].bid_size


# --- C5: bucket worst-case joint-ladder cap (replaces the old stand-in) ---


def test_hedged_book_not_scaled_by_bucket_cap():
    # 3 strikes 98k/100k/102k. Internally-hedged book: YES@98k (p_hat=0.7,
    # f*=0.4) and NO@102k (p_hat=0.3 -> NO belief 0.7, f*=0.4) are the only
    # active legs; the 100k leg is at zero edge (p_hat==mkt_mid) so its Kelly
    # legs are inert (presence floor off here to isolate the bucket stage).
    # fractional_kelly_c is hard-ceilinged at 0.5 (size_ladder: `min(c, 0.5)`)
    # regardless of config, so post-fractional-c f=0.2 -> shares =
    # 0.2*1000/0.5 = 400, risk_frac = 400*0.5/1000 = 0.2 per leg.
    # YES@98k loses only when spot<=98k; NO@102k loses only when spot>102k --
    # these NEVER coincide in one bucket, so true worst-case bucket loss is
    # 0.2 (one leg's risk_frac), not 0.4 (their sum). The OLD stand-in
    # (sum(f) <= max single f) would have scaled sum(f)=0.4 down to a single
    # leg's 0.2 -- a 2x cut on a book that is not actually risky.
    # per_expiry_cap_frac=0.25 sits above the TRUE worst case (0.2) so
    # nothing should be scaled.
    strikes = [98000.0, 100000.0, 102000.0]
    snap = _snap(0.0, strikes=strikes)
    cs = [
        ContractSizingInput("m98", p_hat=0.7, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=98000.0),
        ContractSizingInput("m100", p_hat=0.5, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=100000.0),
        ContractSizingInput("m102", p_hat=0.3, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=102000.0),
    ]
    dec, audit = size_ladder(
        cs, snap, bankroll=1000.0, ts=TS,
        config=MMConfig(presence_frac=0.0, fractional_kelly_c=1.0),
        per_expiry_cap_frac=0.25, bankroll_util_cap=10.0,
    )
    assert SizingCap.RUIN not in dec["m98"].caps_applied
    assert SizingCap.RUIN not in dec["m102"].caps_applied
    # Unscaled: f*=0.4, c capped at 0.5 -> f=0.2 -> shares = 0.2*1000/0.5 = 400.
    assert abs(dec["m98"].bid_size - 400.0) < 1e-6
    assert abs(dec["m102"].ask_size - 400.0) < 1e-6
    bucket_stage = next(s for s in audit["stages"] if s["stage"] == "bucket_worst_case")
    assert abs(bucket_stage["max_loss"] - 0.2) < 1e-9


def test_concentrated_book_scaled_to_bucket_cap():
    # Same 3 strikes, but all three legs are YES with the SAME directional
    # edge (p_hat=0.7 at every strike) -- a genuinely concentrated book: all
    # three lose together when spot<=98k (the worst bucket). True worst-case
    # bucket loss pre-scale is 3*0.2=0.6, comfortably above
    # per_expiry_cap_frac=0.1, so the bucket recheck must scale ALL legs down
    # to hit exactly 0.1 in the worst bucket and record RUIN.
    strikes = [98000.0, 100000.0, 102000.0]
    snap = _snap(0.0, strikes=strikes)
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.7, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=k)
        for i, k in enumerate(strikes)
    ]
    dec, audit = size_ladder(
        cs, snap, bankroll=1000.0, ts=TS,
        config=MMConfig(presence_frac=0.0, fractional_kelly_c=1.0),
        per_expiry_cap_frac=0.10, bankroll_util_cap=10.0,
    )
    for d in dec.values():
        assert SizingCap.RUIN in d.caps_applied
    # Post-scale worst-case bucket loss must equal per_expiry_cap_frac exactly.
    bucket_stage = next(s for s in audit["stages"] if s["stage"] == "bucket_worst_case")
    assert abs(bucket_stage["max_loss"] - 0.6) < 1e-9  # reported pre-scale
    total_risk_frac = sum(
        d.bid_size * 0.5 / 1000.0 for d in dec.values()
    )  # all 3 legs lose together in the worst bucket
    assert abs(total_risk_frac - 0.10) < 1e-9


def test_post_floor_bucket_recheck_rescales_floored_sizes():
    # All legs at zero directional edge (p_hat==mkt_mid) -> Kelly f*=0
    # everywhere, so EVERY leg's size comes purely from the presence floor.
    # A large-enough presence_frac makes the floor itself breach the
    # worst-case bucket cap; the recheck must catch a purely-floor-driven
    # breach (not just a Kelly-driven one) and rescale every leg.
    strikes = [98000.0, 100000.0, 102000.0]
    snap = _snap(0.0, strikes=strikes)
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.5, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=k)
        for i, k in enumerate(strikes)
    ]
    dec, audit = size_ladder(
        cs, snap, bankroll=1000.0, ts=TS,
        config=MMConfig(presence_frac=0.05, fractional_kelly_c=1.0),
        per_expiry_cap_frac=0.10, bankroll_util_cap=10.0,
    )
    presence_stage = next(s for s in audit["stages"] if s["stage"] == "presence_floor")
    assert all(abs(leg["presence_shares"] - 100.0) < 1e-9 for leg in presence_stage["legs"])
    bucket_stage = next(s for s in audit["stages"] if s["stage"] == "bucket_worst_case")
    assert abs(bucket_stage["max_loss"] - 0.15) < 1e-9  # pre-scale, all 3 YES legs at bucket 0
    for d in dec.values():
        assert SizingCap.RUIN in d.caps_applied
        # scaled down from the 100-share floor: 100 * (0.10/0.15)
        assert abs(d.bid_size - (100.0 * (0.10 / 0.15))) < 1e-6


def test_ruin_cap_binds_no_strikes_fallback():
    # No strike info on any leg -> the conservative sum-cap fallback (old
    # ruin-stage semantics, applied in share space) is exercised.
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.9, bid_price=0.5, ask_price=0.5)
        for i in range(5)
    ]
    dec, _ = size_ladder(cs, _snap(0.0), bankroll=1000.0, ts=TS, config=MMConfig(presence_frac=0.0))
    assert SizingCap.RUIN in dec["m0"].caps_applied


def test_bankroll_cap_binds():
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.9, bid_price=0.5, ask_price=0.5)
        for i in range(5)
    ]
    dec, _ = size_ladder(
        cs, _snap(0.0), bankroll=1000.0, ts=TS, config=MMConfig(presence_frac=0.0),
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


# --- Negative edge -> zero size (presence_frac=0 reproduces old behavior) -


def test_negative_edge_zero_both_sides():
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.55, ask_price=0.45)
    # Old-behavior regression: explicit presence_frac=0.0 disables the floor,
    # so a negative-edge leg on both sides is exactly zero, as before C3.
    dec, _ = size_ladder([c], _snap(0.0), bankroll=1000.0, ts=TS, config=MMConfig(presence_frac=0.0))
    assert dec["m0"].bid_size == 0.0
    assert dec["m0"].ask_size == 0.0


def test_negative_edge_default_config_gets_tapered_floor():
    # With the default (presence_frac=0.005, ON), the same negative-edge
    # contract gets nonzero, floor-sized quotes on both sides instead of 0 --
    # the presence floor is a pure max(), not conditioned on edge sign.
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.55, ask_price=0.45)
    cfg = MMConfig()
    dec, _ = size_ladder([c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg)
    expected_bid = cfg.presence_frac * 1000.0 / 0.55
    expected_ask = cfg.presence_frac * 1000.0 / 0.55  # NO price_per_share = 1-ask_price = 0.55
    assert abs(dec["m0"].bid_size - expected_bid) < 1e-9
    assert abs(dec["m0"].ask_size - expected_ask) < 1e-9
    assert dec["m0"].bid_size > 0.0
    assert dec["m0"].ask_size > 0.0


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


# --- C1: mid-edge decoupling (regression test for the recal-coupling bug) -


def test_mid_edge_decoupling_from_our_own_quote():
    # Fixed mkt_mid=0.5, p_hat=0.6 -> f_kelly is fixed at 0.2 regardless of
    # our own bid/ask proposal prices; only the share conversion (which uses
    # OUR quote side as price_per_share) should move.
    snap = _snap(0.0)
    sizes = []
    for bid_price, ask_price in [(0.45, 0.55), (0.30, 0.60), (0.10, 0.90)]:
        c = ContractSizingInput(
            "m0", p_hat=0.6, bid_price=bid_price, ask_price=ask_price,
            mkt_mid=0.5, strike=100000.0,
        )
        dec, _ = size_ladder(
            [c], snap, bankroll=1000.0, ts=TS, config=MMConfig(presence_frac=0.0),
            per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
        )
        assert abs(dec["m0"].f_kelly - 0.2) < 1e-9
        sizes.append(dec["m0"].bid_size)
    # Sizes DO change with our own quote price (price_per_share), proving the
    # edge and the share conversion are genuinely decoupled, not coincidentally
    # equal.
    assert len(set(round(s, 3) for s in sizes)) == len(sizes)


# --- C1: fallback exactness (mkt_mid=None reproduces the pre-C1 edge) -----


def test_mkt_mid_none_reproduces_pre_c1_edge_quantities():
    # mkt_mid=None -> edge price falls back to our own quote side, exactly
    # the pre-C1 formula. Assert EDGE quantities (f_kelly, k_shrink), NOT
    # sizes: the default-ON presence floor changes final sizes even on the
    # fallback path (plan RC-2).
    c = ContractSizingInput("m0", p_hat=0.6, bid_price=0.5, ask_price=0.55)
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    f_expected, b_expected = kelly_buy(0.6, 0.5)
    assert abs(dec["m0"].f_kelly - f_expected) < 1e-12
    assert dec["m0"].k_shrink == 1.0  # sigma2=0 -> k=1 regardless


def test_mkt_mid_none_size_equality_when_depth_cap_binds():
    # For a genuine size-equality fallback assertion (RC-2's suggested
    # construction): a binding depth cap dominates the floor either way, so
    # the fallback path's size matches the depth cap exactly, same as before
    # the floor existed.
    c = ContractSizingInput("m0", p_hat=0.6, bid_price=0.5, ask_price=0.55)
    liq = {"m0": _liq("m0", 5.0, 5.0)}
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, liquidity=liq,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    assert dec["m0"].bid_size == 5.0
    assert SizingCap.DEPTH in dec["m0"].caps_applied


# --- C2: inventory headroom cap -------------------------------------------


def test_inventory_headroom_caps_bid_and_ask():
    c = ContractSizingInput("m0", p_hat=0.9, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=100000.0)
    inv = _inv({"m0": (3.0, 5.0)})  # q=3, q_max=5 -> headroom_bid=2, headroom_ask=8
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=100000.0, ts=TS, config=MMConfig(presence_frac=0.0),
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9, inventory=inv,
    )
    d = dec["m0"]
    assert d.bid_size == 2.0
    assert SizingCap.INVENTORY in d.caps_applied
    assert d.max_add_yes == 2.0
    assert d.max_add_no == 8.0


def test_inventory_headroom_zero_at_q_equals_q_max():
    c = ContractSizingInput("m0", p_hat=0.9, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=100000.0)
    inv = _inv({"m0": (5.0, 5.0)})  # q == q_max -> headroom_bid == 0
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=100000.0, ts=TS, config=MMConfig(presence_frac=0.0),
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9, inventory=inv,
    )
    d = dec["m0"]
    assert d.bid_size == 0.0
    assert d.max_add_yes == 0.0
    assert SizingCap.INVENTORY in d.caps_applied


def test_max_add_fields_zero_when_inventory_not_passed():
    c = ContractSizingInput("m0", p_hat=0.9, bid_price=0.5, ask_price=0.5)
    dec, _ = size_ladder([c], _snap(0.0), bankroll=1000.0, ts=TS)
    assert dec["m0"].max_add_yes == 0.0
    assert dec["m0"].max_add_no == 0.0
    assert SizingCap.INVENTORY not in dec["m0"].caps_applied


# --- C3: presence floor + taper -------------------------------------------


def test_presence_floor_nonzero_at_zero_edge():
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=100000.0)
    cfg = MMConfig(presence_frac=0.005)
    dec, audit = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    d = dec["m0"]
    expected = cfg.presence_frac * 1000.0 / 0.5
    assert abs(d.bid_size - expected) < 1e-9
    assert abs(d.ask_size - expected) < 1e-9
    assert d.bid_size > 0.0 and d.ask_size > 0.0


def test_presence_floor_taper_reaches_zero_at_q_max():
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=100000.0)
    inv = _inv({"m0": (5.0, 5.0)})  # q == q_max -> bid-side taper == 0
    cfg = MMConfig(presence_frac=0.005)
    dec, audit = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9, inventory=inv,
    )
    presence_stage = next(s for s in audit["stages"] if s["stage"] == "presence_floor")
    bid_leg = next(lg for lg in presence_stage["legs"] if lg["is_yes"])
    ask_leg = next(lg for lg in presence_stage["legs"] if not lg["is_yes"])
    assert bid_leg["taper"] == 0.0
    assert dec["m0"].bid_size == 0.0
    assert ask_leg["taper"] == 1.0  # short-side headroom is untouched
    assert dec["m0"].ask_size > 0.0


def test_presence_floor_never_overrides_depth_cap():
    # Depth cap (1 share) binds below what the floor would otherwise set.
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=100000.0)
    cfg = MMConfig(presence_frac=0.005)
    liq = {"m0": _liq("m0", 1.0, 1.0)}
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg, liquidity=liq,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    d = dec["m0"]
    assert d.bid_size == 1.0
    assert d.ask_size == 1.0
    assert SizingCap.DEPTH in d.caps_applied


def test_presence_floor_never_overrides_inventory_cap():
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=100000.0)
    cfg = MMConfig(presence_frac=0.5)  # deliberately large floor
    inv = _inv({"m0": (4.5, 5.0)})  # headroom_bid = 0.5, well below the floor
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9, inventory=inv,
    )
    d = dec["m0"]
    assert d.bid_size == 0.5
    assert SizingCap.INVENTORY in d.caps_applied


def test_presence_floor_never_overrides_bucket_cap():
    # Presence floor sets 100 shares/leg; bucket recheck cuts to 66.67 --
    # asserted by test_post_floor_bucket_recheck_rescales_floored_sizes above
    # with hand-verified numbers. Here just confirm the floor value ITSELF
    # (pre-bucket-cap) would have exceeded the final size, proving the cap
    # dominates.
    strikes = [98000.0, 100000.0, 102000.0]
    snap = _snap(0.0, strikes=strikes)
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.5, bid_price=0.5, ask_price=0.5,
                            mkt_mid=0.5, strike=k)
        for i, k in enumerate(strikes)
    ]
    cfg = MMConfig(presence_frac=0.05, fractional_kelly_c=1.0)
    dec, audit = size_ladder(
        cs, snap, bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.10, bankroll_util_cap=10.0,
    )
    presence_stage = next(s for s in audit["stages"] if s["stage"] == "presence_floor")
    floor_value = presence_stage["legs"][0]["presence_shares"]
    assert floor_value == 100.0
    assert dec["m0"].bid_size < floor_value  # bucket cap cut it below the floor
