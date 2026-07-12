"""Contract-layer tests: construction, validation, enum completeness (plan Section 4)."""
from datetime import datetime, timezone

import pytest

from market_maker import contracts as c
from market_maker.config import MMConfig


NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def _snapshot(**over):
    kw = dict(
        ts=NOW,
        expiry_key="2026-07-20",
        tte_days=14.0,
        s0=100000.0,
        n_sims=15000,
        strikes=[90000.0, 100000.0, 110000.0],
        grid_strikes=[90000.0, 95000.0, 100000.0, 105000.0, 110000.0],
        p_hat={90000.0: 0.8, 100000.0: 0.5, 110000.0: 0.2},
        p_grid={90000.0: 0.8, 95000.0: 0.65, 100000.0: 0.5, 105000.0: 0.35, 110000.0: 0.2},
        sigma2={90000.0: 0.8 * 0.2 / 15000, 100000.0: 0.25 / 15000, 110000.0: 0.2 * 0.8 / 15000},
        sigma2_ladder=0.25 / 15000,
        sigma2_source=c.Sigma2Source.MC,
        confidence_tier=c.ConfidenceTier.FULL,
        horizon_gate_active=False,
        stale=False,
        engine_meta={},
    )
    kw.update(over)
    return c.PricerSnapshot(**kw)


def test_pricer_snapshot_constructs():
    snap = _snapshot()
    assert snap.confidence_tier is c.ConfidenceTier.FULL
    assert snap.sigma2_ladder == pytest.approx(0.25 / 15000)


def test_pricer_snapshot_rejects_out_of_range_probability():
    with pytest.raises((ValueError, AssertionError)):
        _snapshot(p_hat={90000.0: 1.5, 100000.0: 0.5, 110000.0: 0.2})


def test_fair_value_constructs_and_bounds_credibility():
    fv = c.FairValue(
        ts=NOW,
        expiry_key="2026-07-20",
        consensus_p={90000.0: 0.79, 100000.0: 0.5},
        consensus_x={90000.0: 1.32, 100000.0: 0.0},
        credibility=0.6,
        anchor_method=c.AnchorMethod.BEUOY,
        inputs_ts=(NOW, NOW),
        skew_correction=None,
    )
    assert fv.anchor_method is c.AnchorMethod.BEUOY
    with pytest.raises((ValueError, AssertionError)):
        c.FairValue(
            ts=NOW,
            expiry_key="2026-07-20",
            consensus_p={100000.0: 0.5},
            consensus_x={100000.0: 0.0},
            credibility=1.7,
            anchor_method=c.AnchorMethod.BEUOY,
            inputs_ts=(NOW, NOW),
            skew_correction=None,
        )


def test_quote_set_constructs():
    qs = c.QuoteSet(
        ts=NOW,
        market_id="mkt-1",
        bid_price=0.48,
        ask_price=0.52,
        bid_size=25.0,
        ask_size=25.0,
        terms={"markup": 0.01, "eps": 0.0085, "skew": 0.0, "robust": 0.001, "wing": 0.0},
        risk_mode=c.QuoteMode.TWO_SIDED,
        noarb_checked=True,
        source_seq=1,
    )
    assert qs.bid_price < qs.ask_price


def test_fill_and_paper_fill():
    pf = c.PaperFill(
        ts=NOW,
        market_id="mkt-1",
        order_id="ord-1",
        side=c.Side.BUY_YES,
        price=0.48,
        size=10.0,
        liquidity=c.LiquiditySource.MAKER,
        venue_ts=NOW,
        queue_ahead_at_fill=120.0,
        print_size=150.0,
        latency_applied_ms=2000,
        assumption_set="fillmodel-v1",
        mid_at_fill=0.50,
        mid_p1m=None,
        mid_p10m=None,
        mid_p1h=None,
    )
    assert isinstance(pf, c.Fill)
    assert pf.liquidity is c.LiquiditySource.MAKER


def test_settlement_event_unsettleable():
    ev = c.SettlementEvent(
        ts=NOW,
        settlement_ts=NOW,
        market_id="mkt-1",
        expiry_key="2026-07-06",
        strike=100000.0,
        outcome=c.SettlementOutcome.UNSETTLEABLE,
        spot_used=None,
        spot_source=c.SpotSource.NONE,
        q_settled=10.0,
        payoff=None,
        pnl_realized=None,
        excluded_from_gate=True,
    )
    assert ev.excluded_from_gate


def test_inventory_state_composition():
    inv = c.InventoryState(
        ts=NOW,
        per_contract={"mkt-1": c.ContractInv(q=10.0, avg_cost=0.48, q_max=100.0, age_weighted_holding=2.5)},
        per_ladder={"2026-07-20": c.LadderInv(net_band_exposure=[10.0, -5.0], gross=15.0, phi=0.01, r3_histogram={0: 1.0})},
    )
    assert inv.per_contract["mkt-1"].q == 10.0


def test_enums_complete():
    assert {m.name for m in c.QuoteMode} == {"TWO_SIDED", "BID_ONLY", "ASK_ONLY", "PULLED"}
    assert {m.name for m in c.ConfidenceTier} == {"FULL", "DEGRADED", "MINIMAL", "NAIVE_GATED"}
    assert {m.name for m in c.LiquiditySource} == {"MAKER", "TAKER", "SETTLEMENT"}
    assert {m.name for m in c.SettlementOutcome} == {"YES", "NO", "UNSETTLEABLE"}
    assert {m.name for m in c.Sigma2Source} == {"MC", "PARAM_POSTERIOR"}
    assert {m.name for m in c.RiskTrigger} >= {
        "SPOT_JUMP", "NEAR_RESOLUTION", "SPOT_GAPPING_STRIKE", "INV_CAP",
        "FEED_STALE", "PRICER_STALE", "LIQ_DEGENERATE", "MANUAL",
    }
    assert {m.name for m in c.SizingCap} == {
        "LADDER_JOINT", "RUIN", "BANKROLL", "INVENTORY", "DEPTH", "FRACTIONAL_C",
    }
    assert {m.name for m in c.LiquidityRegime} == {"THICK", "NORMAL", "THIN", "DEGENERATE"}


def test_venue_adapter_is_abstract():
    with pytest.raises(TypeError):
        c.VenueAdapter()  # abstract; must not be instantiable


def test_mmconfig_wave1_promoted_field_defaults():
    # W1.4: promoted phantom-config fields must default to exactly the old
    # module defaults (fair_value_anchor.DEFAULT_BANKROLL_FLOOR=0.02,
    # risk_controller._DEFAULT_LATCH_SECONDS=60.0), plus the new W1.2/W1.3
    # fields (reviewer finding 10).
    cfg = MMConfig()
    assert cfg.bankroll_floor == 0.02
    assert cfg.risk_latch_seconds == 60.0
    assert cfg.fv_max_age_s == 300.0
    assert cfg.bankroll_unfreeze_clean_ticks == 20


def test_venue_descriptor():
    vd = c.VenueDescriptor(
        tick_size=0.01,
        min_size=5.0,
        price_band=(0.001, 0.999),
        maker_fee=0.0,
        maker_rebate=0.0,
        settlement_rule="polymarket-12et",
        supports_ladder=True,
    )
    assert vd.price_band == (0.001, 0.999)
