"""Tests for market_maker.robustness_sizing (plan task Z1; wave 1 rewrite per
temp/mm_sizing_fix_plan.md C1-C5 / Section 4; wave 2 rewrite per
temp/mm_sizing_wave2_plan.md W2-W5 / Section 5, 2026-07-12).

Wave 2 replaces wave 1's mkt_mid edge (ContractSizingInput.mkt_mid is
REMOVED) with posted-quote edge net of measured markout: bid_price/ask_price
are now the POSTED prices themselves (both the price_per_share basis AND the
edge price), and mk_avg/mk_var/mk_n/mk_n_attempted resolve the per-leg net
edge m and its Baker-McHale variance sigma2_edge. See robustness_sizing.py's
module docstring and _leg_edge for the exact formulas mirrored below.
"""
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
    # Cold-start (no markout fields) -> m_prior = (p_hat - bid_price) - eps_base.
    # sigma2_edge falls back to config.markout_prior_var regardless of the
    # snapshot's sigma2_ladder (wave 2: per-strike/ladder MC-SE is dropped
    # from leg shrinkage) -- set markout_prior_var=0.0 here so k_shrink=1 and
    # the c<=0.5 ceiling is isolated as the binding control, matching the
    # original (wave 1) intent of this test.
    c = ContractSizingInput("m0", p_hat=0.6, bid_price=0.5, ask_price=0.55)
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS,
        config=MMConfig(presence_frac=0.0, markout_prior_var=0.0),
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    d = dec["m0"]
    assert d.k_shrink == 1.0
    assert SizingCap.FRACTIONAL_C in d.caps_applied
    # structural_edge = 0.6-0.5 = 0.1; m_prior = 0.1-0.0085 = 0.0915
    # f*,b = kelly_buy(0.5915, 0.5); c=0.5 -> f = f*/2*... ; size = f*bankroll/price
    f_expected, _b = kelly_buy(0.5 + 0.0915, 0.5)
    expected_size = f_expected * 0.5 * 1000.0 / 0.5
    assert abs(d.bid_size - expected_size) < 1e-6


# --- Stage 2/W2: sigma2_edge -- prior var (unmeasured) vs mk_var/mk_n (measured)


def test_sigma2_edge_prior_var_when_unmeasured():
    # No markout fields -> sigma2_edge falls back to config.markout_prior_var
    # for every leg, regardless of the snapshot's per-strike/ladder sigma2
    # (wave 2 drops the MC-SE channel from leg shrinkage entirely).
    strikes = [98000.0, 100000.0]
    snap = _snap(
        sigma2_ladder=0.05, strikes=strikes,
        sigma2_by_strike={98000.0: 0.05, 100000.0: 0.0001},
    )
    cs = [
        ContractSizingInput("wing", p_hat=0.7, bid_price=0.5, ask_price=0.5, strike=98000.0),
        ContractSizingInput("atm", p_hat=0.7, bid_price=0.5, ask_price=0.5, strike=100000.0),
    ]
    cfg = MMConfig(presence_frac=0.0, fractional_kelly_c=1.0, markout_prior_var=0.02)
    dec, _ = size_ladder(
        cs, snap, bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=10.0,
    )
    # Same p_hat/bid_price -> identical structural edge -> identical m_prior
    # -> identical sigma2_edge (the shared prior) -> identical k_shrink/size,
    # regardless of the very different per-strike snapshot.sigma2 values.
    assert abs(dec["wing"].k_shrink - dec["atm"].k_shrink) < 1e-12
    assert abs(dec["wing"].bid_size - dec["atm"].bid_size) < 1e-9


def test_sigma2_edge_switches_to_measured_at_min_n():
    # A leg with SMALL measured variance (well-calibrated cell) shrinks LESS
    # (higher k) than a leg on the uninformed prior var, once mk_n crosses
    # config.markout_min_n.
    cfg = MMConfig(presence_frac=0.0, fractional_kelly_c=1.0, markout_min_n=20,
                    markout_prior_var=0.02)
    c_prior = ContractSizingInput("prior", p_hat=0.6, bid_price=0.5, ask_price=0.5)
    c_measured = ContractSizingInput(
        "measured", p_hat=0.6, bid_price=0.5, ask_price=0.5,
        mk_avg=0.05, mk_var=0.0004, mk_n=25, mk_n_attempted=25,
    )
    dec, _ = size_ladder(
        [c_prior, c_measured], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=10.0,
    )
    assert dec["measured"].k_shrink > dec["prior"].k_shrink


# --- C5: bucket worst-case joint-ladder cap (replaces the old stand-in) ---


def test_hedged_book_not_scaled_by_bucket_cap():
    # 3 strikes 98k/100k/102k. Internally-hedged book: YES@98k (p_hat=0.7,
    # bid_price=0.5) and NO@102k (p_hat=0.3 -> NO belief 0.7 at NO price 0.5)
    # are the only active legs; the 100k leg is at zero structural edge
    # (p_hat==bid_price==ask_price=0.5, so m_prior<0 -> Kelly clamps to 0) so
    # its legs are inert (presence floor off here to isolate the bucket
    # stage). fractional_kelly_c is hard-ceilinged at 0.5 (size_ladder:
    # `min(c, 0.5)`) regardless of config.
    # YES@98k loses only when spot<=98k; NO@102k loses only when spot>102k --
    # these NEVER coincide in one bucket, so true worst-case bucket loss is
    # bounded by ONE leg's risk_frac, not their sum. The OLD stand-in
    # (sum(f) <= max single f) would have scaled the sum down to a single
    # leg's fraction -- a 2x cut on a book that is not actually risky.
    strikes = [98000.0, 100000.0, 102000.0]
    snap = _snap(0.0, strikes=strikes)
    cs = [
        ContractSizingInput("m98", p_hat=0.7, bid_price=0.5, ask_price=0.5, strike=98000.0),
        ContractSizingInput("m100", p_hat=0.5, bid_price=0.5, ask_price=0.5, strike=100000.0),
        ContractSizingInput("m102", p_hat=0.3, bid_price=0.5, ask_price=0.5, strike=102000.0),
    ]
    cfg = MMConfig(presence_frac=0.0, fractional_kelly_c=1.0, markout_prior_var=0.0)
    # per_expiry_cap_frac sits above the true worst-case single-leg risk_frac
    # so nothing should be scaled.
    dec, audit = size_ladder(
        cs, snap, bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=10.0,
    )
    assert SizingCap.RUIN not in dec["m98"].caps_applied
    assert SizingCap.RUIN not in dec["m102"].caps_applied
    assert dec["m98"].bid_size > 0.0
    assert dec["m102"].ask_size > 0.0
    bucket_stage = next(s for s in audit["stages"] if s["stage"] == "bucket_worst_case")
    # worst bucket loss is exactly one leg's risk_frac (98k and 102k never
    # co-lose), so it must be < the sum of both legs' risk_frac.
    m98_risk = dec["m98"].bid_size * 0.5 / 1000.0
    m102_risk = dec["m102"].ask_size * 0.5 / 1000.0
    assert bucket_stage["max_loss"] < m98_risk + m102_risk + 1e-9
    assert abs(bucket_stage["max_loss"] - max(m98_risk, m102_risk)) < 1e-9


def test_concentrated_book_scaled_to_bucket_cap():
    # Same 3 strikes, but all three legs are YES with the SAME directional
    # edge (p_hat=0.7 at every strike) -- a genuinely concentrated book: all
    # three lose together when spot<=98k (the worst bucket). True worst-case
    # bucket loss pre-scale exceeds per_expiry_cap_frac=0.05, so the bucket
    # recheck must scale ALL legs down to hit exactly 0.05 in the worst
    # bucket and record RUIN.
    strikes = [98000.0, 100000.0, 102000.0]
    snap = _snap(0.0, strikes=strikes)
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.9, bid_price=0.5, ask_price=0.5, strike=k)
        for i, k in enumerate(strikes)
    ]
    cfg = MMConfig(presence_frac=0.0, fractional_kelly_c=1.0, markout_prior_var=0.0)
    dec, audit = size_ladder(
        cs, snap, bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.05, bankroll_util_cap=10.0,
    )
    for d in dec.values():
        assert SizingCap.RUIN in d.caps_applied
    bucket_stage = next(s for s in audit["stages"] if s["stage"] == "bucket_worst_case")
    # Post-scale worst-case bucket loss must equal per_expiry_cap_frac exactly.
    total_risk_frac = sum(
        d.bid_size * 0.5 / 1000.0 for d in dec.values()
    )  # all 3 legs lose together in the worst bucket
    assert abs(total_risk_frac - 0.05) < 1e-9
    assert bucket_stage["max_loss"] > 0.05  # reported pre-scale


def test_post_floor_bucket_recheck_rescales_floored_sizes():
    # All legs at zero directional edge (p_hat==bid_price==ask_price) -> Kelly
    # f*=0 everywhere, so EVERY leg's size comes purely from the presence
    # floor (cold-start -> exploration carve-out -> gate stays ON). A
    # large-enough presence_frac makes the floor itself breach the
    # worst-case bucket cap; the recheck must catch a purely-floor-driven
    # breach (not just a Kelly-driven one) and rescale every leg.
    strikes = [98000.0, 100000.0, 102000.0]
    snap = _snap(0.0, strikes=strikes)
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.5, bid_price=0.5, ask_price=0.5, strike=k)
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


# --- Depth cap (wave 2 W5: floored at config.depth_cap_floor_shares) ------


def test_depth_cap_binds_and_inert_when_absent():
    c = ContractSizingInput("m0", p_hat=0.6, bid_price=0.5, ask_price=0.55)
    # inert when liquidity absent
    dec0, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    assert SizingCap.DEPTH not in dec0["m0"].caps_applied
    assert dec0["m0"].bid_size > 5.0
    # binds when depth small (5 > depth_cap_floor_shares default 1.0)
    dec1, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS,
        liquidity={"m0": _liq("m0", 5.0, 5.0)},
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    assert SizingCap.DEPTH in dec1["m0"].caps_applied
    assert dec1["m0"].bid_size == 5.0


def test_depth_cap_floored_at_min_restorable_size():
    # realized_depth=0 (dead book) no longer permanently zeroes size -- the
    # cap floors at config.depth_cap_floor_shares (default 1.0), not 0.0.
    c = ContractSizingInput("m0", p_hat=0.6, bid_price=0.5, ask_price=0.55)
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS,
        liquidity={"m0": _liq("m0", 0.0, 0.0)},
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    d = dec["m0"]
    assert SizingCap.DEPTH in d.caps_applied
    assert d.bid_size == MMConfig().depth_cap_floor_shares
    assert d.bid_size > 0.0


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
    # contract gets nonzero, floor-sized quotes on both sides instead of 0.
    # Cold-start (no markout fields at all -> mk_n_attempted=0 <
    # markout_min_n) is the exploration carve-out -- the W4 gate stays ON
    # regardless of the (negative) m_gate sign, so the floor behaves exactly
    # as wave 1's unconditional floor.
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


# --- W2: posted-quote edge with measured-markout haircut (supersedes wave 1
# mid-edge decoupling -- Kelly now edges off our OWN posted quote, haircut
# by measured/prior net edge m, not the market mid) -------------------------


def test_posted_edge_with_markout_haircut():
    # Measured path (mk_n >= markout_min_n): m = mk_avg directly, independent
    # of p_hat/structural edge.
    cfg = MMConfig(presence_frac=0.0, markout_min_n=20)
    c_measured = ContractSizingInput(
        "m0", p_hat=0.5, bid_price=0.5, ask_price=0.5,
        mk_avg=0.03, mk_var=0.0009, mk_n=25, mk_n_attempted=25,
    )
    dec, _ = size_ladder(
        [c_measured], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    f_expected, _b = kelly_buy(0.5 + 0.03, 0.5)
    assert abs(dec["m0"].f_kelly - f_expected) < 1e-9

    # Below min_n: falls back to m_prior = structural_edge - eps_base, EVEN
    # THOUGH mk_avg is present (measurement not trusted yet).
    c_thin = ContractSizingInput(
        "m0", p_hat=0.6, bid_price=0.5, ask_price=0.5,
        mk_avg=0.03, mk_var=0.0009, mk_n=5, mk_n_attempted=5,
    )
    dec2, _ = size_ladder(
        [c_thin], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    m_prior = (0.6 - 0.5) - cfg.eps_base
    f_prior_expected, _b = kelly_buy(0.5 + m_prior, 0.5)
    assert abs(dec2["m0"].f_kelly - f_prior_expected) < 1e-9

    # Negative m (measured) -> zero Kelly on that leg.
    c_neg = ContractSizingInput(
        "m0", p_hat=0.5, bid_price=0.5, ask_price=0.5,
        mk_avg=-0.02, mk_var=0.0004, mk_n=25, mk_n_attempted=25,
    )
    dec3, _ = size_ladder(
        [c_neg], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    assert dec3["m0"].f_kelly == 0.0


def test_posted_edge_prior_path_all_fields_default():
    # Cold-start contract (no markout kwargs at all -- the "mkt_mid=None"
    # equivalent of wave 1) -> m_prior = structural_edge - eps_base
    # everywhere; f_kelly matches the hand-computed kelly_buy(price+m, price).
    cfg = MMConfig(presence_frac=0.0)
    c = ContractSizingInput("m0", p_hat=0.6, bid_price=0.5, ask_price=0.55)
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    m_prior_yes = (0.6 - 0.5) - cfg.eps_base
    f_expected, b_expected = kelly_buy(0.5 + m_prior_yes, 0.5)
    k_expected = baker_mchale(f_expected, b_expected, cfg.markout_prior_var)
    assert abs(dec["m0"].f_kelly - f_expected) < 1e-12
    assert abs(dec["m0"].k_shrink - k_expected) < 1e-12  # unmeasured -> markout_prior_var
    assert abs(dec["m0"].bid_size - f_expected * k_expected * 0.5 * 1000.0 / 0.5) < 1e-6


def test_markout_haircut_monotonicity():
    # More negative mk_avg -> smaller size, down to (and staying at) 0, once
    # measured (mk_n >= min_n).
    cfg = MMConfig(presence_frac=0.0, markout_min_n=20)
    sizes = []
    for mk_avg in (0.02, 0.0, -0.01, -0.05):
        c = ContractSizingInput(
            "m0", p_hat=0.5, bid_price=0.5, ask_price=0.5,
            mk_avg=mk_avg, mk_var=0.0004, mk_n=25, mk_n_attempted=25,
        )
        dec, _ = size_ladder(
            [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
            per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
        )
        sizes.append(dec["m0"].bid_size)
    for a, b in zip(sizes, sizes[1:]):
        assert b <= a + 1e-12
    assert sizes[-1] == 0.0
    assert sizes[-2] == 0.0  # mk_avg=-0.01 also clamps to 0 (Glosten-Milgrom)
    assert sizes[0] > sizes[1]


def test_mkt_mid_field_removed():
    # wave 2 W2: mkt_mid no longer exists on ContractSizingInput; constructing
    # with it must raise, and no instance carries the attribute.
    import pytest
    with pytest.raises(TypeError):
        ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5, mkt_mid=0.5)
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5)
    assert not hasattr(c, "mkt_mid")


# --- C2: inventory headroom cap -------------------------------------------


def test_inventory_headroom_caps_bid_and_ask():
    c = ContractSizingInput("m0", p_hat=0.9, bid_price=0.5, ask_price=0.5, strike=100000.0)
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
    c = ContractSizingInput("m0", p_hat=0.9, bid_price=0.5, ask_price=0.5, strike=100000.0)
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
    # p_hat == posted bid == posted ask (1-ask=0.5): structural edge is 0 on
    # both legs; the floor is still nonzero (cold-start -> exploration
    # carve-out -> gate ON).
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5, strike=100000.0)
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
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5, strike=100000.0)
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
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5, strike=100000.0)
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
    c = ContractSizingInput("m0", p_hat=0.5, bid_price=0.5, ask_price=0.5, strike=100000.0)
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
    # Presence floor sets 100 shares/leg; bucket recheck cuts below it (see
    # test_post_floor_bucket_recheck_rescales_floored_sizes for the
    # hand-verified numbers). Here just confirm the floor value ITSELF
    # (pre-bucket-cap) exceeds the final size, proving the cap dominates.
    strikes = [98000.0, 100000.0, 102000.0]
    snap = _snap(0.0, strikes=strikes)
    cs = [
        ContractSizingInput("m%d" % i, p_hat=0.5, bid_price=0.5, ask_price=0.5, strike=k)
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


# --- W4: presence floor gated on measured net edge (+ exploration carve-out)


def test_floor_gate_off_when_measured_negative_and_trusted():
    # Measured negative net edge with BOTH mk_n and mk_n_attempted >= min_n:
    # the gate closes on that leg -> the presence floor no longer applies
    # (only the Kelly path, which is also 0 since m clamps at 0).
    cfg = MMConfig(presence_frac=0.05, markout_min_n=20)
    c = ContractSizingInput(
        "m0", p_hat=0.5, bid_price=0.5, ask_price=0.5,
        mk_avg=-0.02, mk_var=0.0004, mk_n=25, mk_n_attempted=25,
    )
    dec, audit = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    presence_stage = next(s for s in audit["stages"] if s["stage"] == "presence_floor")
    bid_leg = next(lg for lg in presence_stage["legs"] if lg["is_yes"])
    assert bid_leg["gate"] is False
    assert dec["m0"].bid_size == 0.0


def test_floor_gate_on_exploration_carve_out_when_unattempted():
    # mk_n_attempted below markout_min_n (never measured, or barely measured)
    # keeps the gate ON regardless of the (negative) m_gate sign -- the
    # anti-starvation clause.
    cfg = MMConfig(presence_frac=0.05, markout_min_n=20)
    c = ContractSizingInput(
        "m0", p_hat=0.5, bid_price=0.5, ask_price=0.5,
        mk_avg=-0.02, mk_var=0.0004, mk_n=3, mk_n_attempted=3,
    )
    dec, audit = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9,
    )
    presence_stage = next(s for s in audit["stages"] if s["stage"] == "presence_floor")
    bid_leg = next(lg for lg in presence_stage["legs"] if lg["is_yes"])
    assert bid_leg["gate"] is True
    assert dec["m0"].bid_size > 0.0


# --- W3: reduce-side exemption from the f*>=0 floor ------------------------


def test_reduce_side_exemption_floors_unload_side_ungated():
    # q=8 (net long YES) -> the ask/NO leg is the reduce side. A trusted,
    # measured negative net edge (mk_avg<0, mk_n/mk_n_attempted >=
    # markout_min_n) closes the W4 gate on BOTH legs (ordinary presence floor
    # off, Kelly also 0 since m clamps at 0) -- yet the reduce-side exemption
    # still floors the ask at min(|q|, s_presence), UNGATED. The accumulating
    # (bid/YES) side stays at 0: the exemption is not applied there.
    cfg = MMConfig(presence_frac=0.02, markout_min_n=20)
    c = ContractSizingInput(
        "m0", p_hat=0.5, bid_price=0.5, ask_price=0.5, strike=100000.0,
        mk_avg=-0.05, mk_var=0.0004, mk_n=25, mk_n_attempted=25,
    )
    inv = _inv({"m0": (8.0, 50.0)})
    dec, audit = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9, inventory=inv,
    )
    presence_stage = next(s for s in audit["stages"] if s["stage"] == "presence_floor")
    assert all(leg["gate"] is False for leg in presence_stage["legs"])  # ordinary floor off
    reduce_stage = next(s for s in audit["stages"] if s["stage"] == "reduce_side_exemption")
    ask_entry = next(e for e in reduce_stage["legs"] if not e["is_yes"])
    s_presence = cfg.presence_frac * 1000.0 / 0.5  # NO price_per_share = 1-ask=0.5
    expected_floor = min(8.0, s_presence)
    assert abs(ask_entry["reduce_floor"] - expected_floor) < 1e-9
    assert abs(dec["m0"].ask_size - expected_floor) < 1e-6
    assert dec["m0"].ask_size > 0.0
    # NOT applied to the accumulating (bid/YES) side.
    assert not any(e["is_yes"] for e in reduce_stage["legs"] if e["market_id"] == "m0")
    assert dec["m0"].bid_size == 0.0


def test_reduce_side_exemption_short_yes_floors_bid_side():
    # q=-8 (net long NO / short YES) -> the bid/YES leg is the reduce side.
    # Same trusted-negative-edge setup, gate off on both legs.
    cfg = MMConfig(presence_frac=0.02, markout_min_n=20)
    c = ContractSizingInput(
        "m0", p_hat=0.5, bid_price=0.5, ask_price=0.5, strike=100000.0,
        mk_avg=-0.05, mk_var=0.0004, mk_n=25, mk_n_attempted=25,
    )
    inv = _inv({"m0": (-8.0, 50.0)})
    dec, audit = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9, inventory=inv,
    )
    reduce_stage = next(s for s in audit["stages"] if s["stage"] == "reduce_side_exemption")
    bid_entry = next(e for e in reduce_stage["legs"] if e["is_yes"])
    s_presence = cfg.presence_frac * 1000.0 / 0.5
    expected_floor = min(8.0, s_presence)
    assert abs(bid_entry["reduce_floor"] - expected_floor) < 1e-9
    assert abs(dec["m0"].bid_size - expected_floor) < 1e-6
    assert dec["m0"].ask_size == 0.0


def test_reduce_side_exemption_caps_still_dominate():
    # Even with a large reduce-side floor, a binding depth cap still wins
    # (caps dominate floors, always).
    cfg = MMConfig(presence_frac=0.5)  # large floor unit
    c = ContractSizingInput("m0", p_hat=0.9, bid_price=0.5, ask_price=0.5, strike=100000.0)
    inv = _inv({"m0": (8.0, 50.0)})
    liq = {"m0": _liq("m0", 5.0, 2.0)}
    dec, _ = size_ladder(
        [c], _snap(0.0), bankroll=1000.0, ts=TS, config=cfg, liquidity=liq,
        per_expiry_cap_frac=0.9, bankroll_util_cap=0.9, inventory=inv,
    )
    assert dec["m0"].ask_size == 2.0
    assert SizingCap.DEPTH in dec["m0"].caps_applied
