"""Tests for market_maker.pnl_report (plan "MM Monitor Dashboard Page + Engine
Start/Stop Control", Step 1 / Step 6.1).

Every scenario is hand-computed and asserted exactly, including the B1
fill_cash quadrants, the realized-PnL identity (open / partial-reduce /
flip / settlement-close), the B3 restart-equivalence of cash_by_market, the
mid-None policy, long+short utilization, and TOTAL == sum(per-market).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.contracts import (
    ContractInv, Fill, LiquiditySource, PaperFill, SettlementEvent, SettlementOutcome, Side, SpotSource,
)
from market_maker.inventory_manager import InventoryManager
from market_maker.pnl_report import (
    MARKOUT_LOOKBACK_S,
    MARKOUT_WINDOW_S,
    MID_LOG_RETENTION_S,
    cash_by_market,
    compute_pnl_rows,
    fill_cash,
    markout_report,
    markout_stats,
    markout_stats_side,
    rebate_for_fill,
    tte_bucket_label,
)
from market_maker.settlement_handler import settlement_instant_utc
from market_maker.state_store import MMStateStore

NOW = datetime(2026, 7, 6, 16, 5, tzinfo=timezone.utc)


def _fill(market_id, side, price, size, liquidity=LiquiditySource.MAKER, ts=NOW, order_id="o1"):
    return Fill(ts=ts, market_id=market_id, order_id=order_id, side=side, price=price,
                size=size, liquidity=liquidity, venue_ts=ts)


def _paper_fill(market_id, side, price, size, mid_at_fill, ts=NOW, order_id="o1",
                liquidity=LiquiditySource.MAKER):
    return PaperFill(
        ts=ts, market_id=market_id, order_id=order_id, side=side, price=price, size=size,
        liquidity=liquidity, venue_ts=ts, mid_at_fill=mid_at_fill,
    )


# ---------------------------------------------------------------------------
# fill_cash -- four quadrants (B1)
# ---------------------------------------------------------------------------


def test_fill_cash_regular_buy_yes():
    # price IS the YES price already; BUY_YES is a cash outflow.
    assert fill_cash(Side.BUY_YES, 0.40, 10.0) == pytest.approx(-4.0)


def test_fill_cash_regular_buy_no():
    # price is already YES-scale (C0) -> used directly, no complement;
    # BUY_NO is booked as a cash inflow in the YES-equivalent frame.
    assert fill_cash(Side.BUY_NO, 0.70, 5.0) == pytest.approx(3.5)


def test_fill_cash_settlement_buy_yes_no_complement():
    # SETTLEMENT price is ALWAYS payoff_yes, never complemented -- here the
    # market resolved NO (payoff_yes=0.0) and a short position is closed via
    # BUY_YES; 0.0 * size = 0.0, NOT (1 - 0.0) * size. (F8: fill_cash no
    # longer takes a liquidity param -- it never affected the computation.)
    assert fill_cash(Side.BUY_YES, 0.0, 5.0) == pytest.approx(0.0)


def test_fill_cash_settlement_buy_no_no_complement():
    # SETTLEMENT price is payoff_yes -- here the market resolved YES
    # (payoff_yes=1.0) and a long position is closed via BUY_NO; the closing
    # side does NOT flip the settlement price's meaning (B1's key point).
    assert fill_cash(Side.BUY_NO, 1.0, 5.0) == pytest.approx(5.0)


def test_regular_and_settlement_transforms_now_agree():
    # Post-C0: price is YES-scale for every fill regardless of liquidity, so
    # MAKER/TAKER and SETTLEMENT fills with the same side/price/size produce
    # IDENTICAL cash -- liquidity no longer changes the computation (unlike
    # pre-C0, where a regular BUY_NO fill at "price"=0.0 complemented to
    # yes_price=1.0 (cash=+5.0) while the settlement variant did not
    # (cash=0.0); that asymmetry was the bug this fix removes). fill_cash
    # dropped the liquidity param entirely (F8); both calls below are simply
    # the same call twice, which is itself the regression gate.
    regular = fill_cash(Side.BUY_NO, 0.0, 5.0)
    settlement = fill_cash(Side.BUY_NO, 0.0, 5.0)
    assert regular == pytest.approx(0.0)
    assert settlement == pytest.approx(0.0)
    assert regular == pytest.approx(settlement)


# ---------------------------------------------------------------------------
# C0 scale-consistency (mm_suitability_alignment_plan.md pre-step C0):
# paper_fill_sim stores YES-scale prices for BOTH sides (the harness bridge,
# harness.py:98-105, un-complements BUY_NO's order-placement NO-price back to
# the geometric YES-book price before any fill reaches inventory_manager /
# state_store / pnl_report). inventory_manager._apply_contract_fill already
# uses the raw stored price for BOTH sides and is the untouched reference;
# fold_fills_to_inventory and fill_cash's MAKER/TAKER BUY_NO branches used to
# complement (1 - price), which desynced them from that reference and
# produced a phantom -0.20/share PnL on every open BUY_NO fill.
# ---------------------------------------------------------------------------


def test_c0_fold_avg_cost_matches_inventory_manager_reference_on_buy_no(tmp_path):
    # One MAKER BUY_NO paper fill at YES-scale price 0.60. Before the C0 fix,
    # fold_fills_to_inventory complemented this to avg_cost=0.40 while
    # InventoryManager (the reference) correctly kept the raw 0.60 -- the two
    # sources of truth disagreed. This must fail on the fold side pre-fix.
    fill = _fill("m1", Side.BUY_NO, 0.60, 10.0)

    store = MMStateStore(str(tmp_path / "c0_diag.db"))
    store.append_fill(fill)
    folded = store.fold_fills_to_inventory()["m1"]
    store.close()

    inv_mgr = InventoryManager()
    inv_mgr.apply_fill(fill)
    reference = inv_mgr.snapshot(NOW).per_contract["m1"]

    assert reference.avg_cost == pytest.approx(0.60)  # raw, YES-scale -- never complemented
    assert folded.avg_cost == pytest.approx(reference.avg_cost)

    # Flat YES mid at the same 0.60: no PnL should have moved yet.
    rows = compute_pnl_rows(
        NOW, "2026-07-06", [fill], {"m1": reference},
        mids={"m1": 0.60}, consensus={}, initial_bankroll=1000.0,
    )
    row = next(r for r in rows if r.market_id == "m1")
    assert row.realized == pytest.approx(0.0)
    assert row.unrealized_mid == pytest.approx(0.0)


def test_c0_lifecycle_open_buy_no_settle_no_win_realized_gain():
    # Open BUY_NO @ YES-price 0.60 (size 10) -> settle NO-win (payoff_yes=0.0):
    # economically this is "sold 10 YES-equivalent shares at 0.60, they later
    # became worthless" -> realized == +0.60 * size.
    size = 10.0
    fills = [
        _fill("m1", Side.BUY_NO, 0.60, size, order_id="open"),
        _fill("m1", Side.BUY_YES, 0.0, size, liquidity=LiquiditySource.SETTLEMENT, order_id="settle"),
    ]
    inventory = {"m1": ContractInv(q=0.0, avg_cost=0.0, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    assert per_market["m1"].realized == pytest.approx(0.60 * size)


def test_c0_lifecycle_open_buy_no_settle_yes_win_realized_loss():
    # Same open, but the market settles YES (payoff_yes=1.0): closing side is
    # BUY_YES at price=1.0 -> realized == -(1.0 - 0.60) * size == -0.40 * size.
    size = 10.0
    fills = [
        _fill("m1", Side.BUY_NO, 0.60, size, order_id="open"),
        _fill("m1", Side.BUY_YES, 1.0, size, liquidity=LiquiditySource.SETTLEMENT, order_id="settle"),
    ]
    inventory = {"m1": ContractInv(q=0.0, avg_cost=0.0, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    assert per_market["m1"].realized == pytest.approx(-0.40 * size)


# ---------------------------------------------------------------------------
# realized identity: cash + q*avg_cost
# ---------------------------------------------------------------------------


def test_realized_open_then_partial_reduce():
    # Open 10 BUY_YES @ 0.4 (q=10, avg_cost=0.4). Reduce 5 via BUY_NO @ 0.7
    # (raw YES-scale, C0 -- no complement): q=5, avg_cost unchanged at 0.4
    # (fold rule: the remaining lot keeps its existing avg_cost on a partial
    # reduce).
    fills = [
        _fill("m1", Side.BUY_YES, 0.40, 10.0, order_id="open"),
        _fill("m1", Side.BUY_NO, 0.70, 5.0, order_id="reduce"),
    ]
    inventory = {"m1": ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    # cash = -0.4*10 + 0.7*5 = -4.0 + 3.5 = -0.5; realized = -0.5 + 5*0.4 = 1.5
    # (selling 5 of the 10 shares bought @0.4 at 0.7 nets +1.5 realized profit;
    # matches "sold 5 YES-equivalent shares at 0.7" economics under C0.)
    assert per_market["m1"].realized == pytest.approx(1.5)


def test_realized_flip_pins_avg_cost_reset():
    # Open 5 BUY_YES @ 0.3 (q=5, avg_cost=0.3). Flip via 8 BUY_NO @ 0.5 (raw
    # YES-scale, C0 -- price 0.5 happens to be its own complement, so this
    # scenario's numbers are unchanged by the C0 fix): closes the 5 long
    # shares and opens a new -3 short lot at avg_cost = the flipping fill's
    # cost_basis_price (0.5).
    fills = [
        _fill("m1", Side.BUY_YES, 0.30, 5.0, order_id="open"),
        _fill("m1", Side.BUY_NO, 0.50, 8.0, order_id="flip"),
    ]
    inventory = {"m1": ContractInv(q=-3.0, avg_cost=0.50, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    # cash = -0.3*5 + 0.5*8 = -1.5 + 4.0 = 2.5; realized = 2.5 + (-3)*0.5 = 1.0
    # (closing the 5 long shares bought @0.3 at 0.5 nets +1.0; the fresh -3
    # lot is unrealized only, correctly excluded here.)
    assert per_market["m1"].realized == pytest.approx(1.0)


def test_realized_settlement_closes_long_includes_payoff():
    # Open 5 BUY_YES @ 0.4. Market settles YES (payoff_yes=1.0): closing
    # side is BUY_NO per settlement_handler.py:296, price=payoff_yes=1.0.
    fills = [
        _fill("m1", Side.BUY_YES, 0.40, 5.0, order_id="open"),
        _fill("m1", Side.BUY_NO, 1.0, 5.0, liquidity=LiquiditySource.SETTLEMENT, order_id="settle"),
    ]
    inventory = {"m1": ContractInv(q=0.0, avg_cost=0.0, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory,
                             mids={"m1": 0.9}, consensus={"m1": 0.9}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    # cash = -0.4*5 + 1.0*5 = -2.0 + 5.0 = 3.0; realized = 3.0 + 0*0 = 3.0
    assert per_market["m1"].realized == pytest.approx(3.0)
    # q==0 -> unrealized collapses to 0.0 regardless of the mid/consensus passed.
    assert per_market["m1"].unrealized_mid == pytest.approx(0.0)
    assert per_market["m1"].unrealized_consensus == pytest.approx(0.0)


def test_realized_settlement_closes_short_includes_payoff():
    # Open 5 BUY_NO @ 0.7 (raw YES-scale, C0 -- cost_basis=0.7, q=-5). Market
    # settles NO (payoff_yes=0.0): closing side is BUY_YES, price=payoff_yes
    # =0.0.
    fills = [
        _fill("m1", Side.BUY_NO, 0.70, 5.0, order_id="open"),
        _fill("m1", Side.BUY_YES, 0.0, 5.0, liquidity=LiquiditySource.SETTLEMENT, order_id="settle"),
    ]
    inventory = {"m1": ContractInv(q=0.0, avg_cost=0.0, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    # cash = 0.7*5 + 0.0*5 = 3.5 + 0.0 = 3.5; realized = 3.5 + 0*0 = 3.5
    # (shorted 5 YES-equivalent shares at 0.7, market resolved NO -> profit
    # = 5*(0.7-0) = 3.5; matches the C0 lifecycle regression test above.)
    assert per_market["m1"].realized == pytest.approx(3.5)
    assert per_market["m1"].unrealized_mid == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cash_by_market restart equivalence (B3)
# ---------------------------------------------------------------------------


def test_cash_by_market_matches_incremental_sum():
    fills = [
        _fill("m1", Side.BUY_YES, 0.40, 10.0, order_id="1"),
        _fill("m2", Side.BUY_NO, 0.60, 3.0, order_id="2"),
        _fill("m1", Side.BUY_NO, 0.55, 4.0, order_id="3"),
        _fill("m2", Side.BUY_YES, 0.35, 2.0, order_id="4"),
    ]
    incremental_total = sum(fill_cash(f.side, f.price, f.size) for f in fills)
    recomputed_total = sum(cash_by_market(fills).values())
    assert recomputed_total == pytest.approx(incremental_total)


def test_cash_by_market_recompute_from_full_history_is_additive():
    # Simulates a restart: cash recomputed from the full fills table (as
    # store.get_fills() would return after reload) must equal folding the
    # fills incrementally in one pass, with no double counting and no
    # dependence on any in-process running total.
    fills = [
        _fill("m1", Side.BUY_YES, 0.40, 10.0, order_id="1"),
        _fill("m1", Side.BUY_NO, 0.55, 4.0, order_id="2"),
        _fill("m1", Side.BUY_YES, 0.20, 3.0, order_id="3"),
    ]
    before_restart = cash_by_market(fills[:2])["m1"]
    tail_cash = fill_cash(fills[2].side, fills[2].price, fills[2].size)
    after_restart_full_reload = cash_by_market(fills)["m1"]
    assert after_restart_full_reload == pytest.approx(before_restart + tail_cash)


# ---------------------------------------------------------------------------
# mid-None policy, utilization, TOTAL == sum(per-market)
# ---------------------------------------------------------------------------


def test_mid_none_policy_and_utilization_long_short_mix():
    fills = [
        _fill("m-long", Side.BUY_YES, 0.60, 10.0, order_id="1"),
        _fill("m-short", Side.BUY_NO, 0.75, 4.0, order_id="2"),  # cost_basis=0.75 (raw, C0)
    ]
    inventory = {
        "m-long": ContractInv(q=10.0, avg_cost=0.60, q_max=100.0, age_weighted_holding=0.0),
        "m-short": ContractInv(q=-4.0, avg_cost=0.75, q_max=100.0, age_weighted_holding=0.0),
    }
    # No mid for m-long (missing key), explicit None for m-short.
    mids = {"m-short": None}
    consensus = {"m-long": 0.65}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids, consensus, initial_bankroll=100.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    total = next(r for r in rows if r.market_id is None)

    # mid-None policy: both markets have no usable mid -> unrealized_mid=0.0.
    assert per_market["m-long"].unrealized_mid == pytest.approx(0.0)
    assert per_market["m-short"].unrealized_mid == pytest.approx(0.0)
    # consensus present for m-long only.
    assert per_market["m-long"].unrealized_consensus == pytest.approx(10.0 * (0.65 - 0.60))
    assert per_market["m-short"].unrealized_consensus == pytest.approx(0.0)

    # utilization: long at_risk = q*avg_cost = 10*0.6=6.0 -> 6/100=0.06
    # short at_risk = |q|*(1-avg_cost) = 4*(1-0.75)=1.0 -> 1/100=0.01
    assert per_market["m-long"].bankroll_utilization == pytest.approx(0.06)
    assert per_market["m-short"].bankroll_utilization == pytest.approx(0.01)
    assert total.bankroll_utilization == pytest.approx(0.07)

    # TOTAL row == sum of per-market rows.
    assert total.realized == pytest.approx(per_market["m-long"].realized + per_market["m-short"].realized)
    assert total.unrealized_mid == pytest.approx(
        per_market["m-long"].unrealized_mid + per_market["m-short"].unrealized_mid
    )
    assert total.unrealized_consensus == pytest.approx(
        per_market["m-long"].unrealized_consensus + per_market["m-short"].unrealized_consensus
    )
    assert total.market_id is None
    assert total.expiry_key == "2026-07-06"


def test_settlement_pnl_breakdown_is_report_only_and_summed():
    fills = [
        _fill("m1", Side.BUY_YES, 0.40, 5.0, order_id="open"),
        _fill("m1", Side.BUY_NO, 1.0, 5.0, liquidity=LiquiditySource.SETTLEMENT, order_id="settle"),
    ]
    inventory = {"m1": ContractInv(q=0.0, avg_cost=0.0, q_max=100.0, age_weighted_holding=0.0)}
    settlements = [
        SettlementEvent(
            ts=NOW, settlement_ts=NOW, market_id="m1", expiry_key="2026-07-06", strike=100000.0,
            outcome=SettlementOutcome.YES, spot_used=101000.0, spot_source=SpotSource.INTRADAY,
            q_settled=5.0, payoff=5.0, pnl_realized=3.0, excluded_from_gate=False,
        )
    ]
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={},
                             initial_bankroll=1000.0, settlements=settlements)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    total = next(r for r in rows if r.market_id is None)
    assert per_market["m1"].settlement_pnl == pytest.approx(3.0)
    assert total.settlement_pnl == pytest.approx(3.0)
    # settlement_pnl is a report-only breakdown -- realized (which already
    # contains the settlement cash) is unaffected by whether settlements
    # were supplied at all (M1 partition).
    assert per_market["m1"].realized == pytest.approx(3.0)


def test_omitting_settlements_leaves_settlement_pnl_zero():
    fills = [_fill("m1", Side.BUY_YES, 0.40, 5.0, order_id="open")]
    inventory = {"m1": ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    assert per_market["m1"].settlement_pnl == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# markout_report (mm_suitability_alignment_plan.md Change C3)
# ---------------------------------------------------------------------------

BELLY_BAND = (0.2, 0.8)
FAR_EXPIRY = "2026-07-20"  # >> NOW + any horizon in these tests, so tte stays positive


def _mid_lookup_from_rows(rows):
    """Test double for MMStateStore.mid_at_or_after: rows is
    {market_id: [(ts, mid), ...]}; returns the first ts in [ts_min, ts_max)
    (matching the store method's ORDER BY ts LIMIT 1 semantics). F1: the
    upper bound is EXCLUSIVE, in lockstep with the real mid_at_or_after's
    `ts < ts_max` -- flipping only one side of this pair would silently
    validate the wrong semantics.
    """
    def lookup(market_id, ts_min, ts_max):
        candidates = sorted(rows.get(market_id, []), key=lambda tm: tm[0])
        for ts, mid in candidates:
            if ts_min <= ts < ts_max:
                return mid
        return None
    return lookup


def test_markout_flat_mid_buy_no_regression():
    # THE reviewer's blocker gate: BUY_NO fill at YES-scale price 0.60, YES
    # mid flat at 0.60 -> mk ~= 0 at EVERY horizon (never complement the
    # stored price).
    fill = _paper_fill("m1", Side.BUY_NO, 0.60, 10.0, mid_at_fill=0.60)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [
        (NOW + timedelta(seconds=60), 0.60),
        (NOW + timedelta(seconds=600), 0.60),
        (NOW + timedelta(seconds=3600), 0.60),
    ]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0, 600.0, 3600.0),
    )
    assert len(report["cells"]) == 3
    for cell in report["cells"]:
        assert cell["n"] == 1
        assert cell["mk_avg"] == pytest.approx(0.0, abs=1e-12)
        assert cell["mk_total"] == pytest.approx(0.0, abs=1e-12)


def test_markout_buy_yes_adverse_case():
    # buy YES at 0.55, mid drops to 0.50 -> mk = +1*(0.50-0.55) = -0.05.
    fill = _paper_fill("m1", Side.BUY_YES, 0.55, 5.0, mid_at_fill=0.55)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(NOW + timedelta(seconds=60), 0.50)]}
    report = markout_report([fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,))
    cell = report["cells"][0]
    assert cell["horizon_s"] == 60.0
    assert cell["mk_avg"] == pytest.approx(-0.05)


def test_markout_buy_no_adverse_case():
    # sold-YES (BUY_NO) at 0.60, mid rises to 0.65 -> mk = -1*(0.65-0.60) = -0.05.
    fill = _paper_fill("m1", Side.BUY_NO, 0.60, 5.0, mid_at_fill=0.60)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(NOW + timedelta(seconds=60), 0.65)]}
    report = markout_report([fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,))
    cell = report["cells"][0]
    assert cell["mk_avg"] == pytest.approx(-0.05)


def test_markout_horizon_window_exclusion():
    # Only a mid row satisfying the h=60 window; nothing in [600, 1200) for
    # h=600. F2: the h=600 horizon is still an ATTEMPTED cell (every
    # non-SETTLEMENT fill within the lookback is attempted at every horizon,
    # regardless of lookup success) -- it is emitted with n=0, n_attempted=1,
    # not silently dropped; only h=60 records an actual hit.
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(NOW + timedelta(seconds=60), 0.50)]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0, 600.0),
    )
    by_horizon = {c["horizon_s"]: c for c in report["cells"]}
    assert set(by_horizon) == {60.0, 600.0}
    assert by_horizon[60.0]["n"] == 1
    assert by_horizon[60.0]["n_attempted"] == 1
    assert by_horizon[600.0]["n"] == 0
    assert by_horizon[600.0]["n_attempted"] == 1


def test_markout_window_boundary_is_exclusive_of_markout_window_s():
    # F1: the window is [fill.ts + h, fill.ts + h + MARKOUT_WINDOW_S) -- the
    # upper bound is now EXCLUSIVE (state_store.mid_at_or_after's `ts <
    # ts_max`), so a mid row landing exactly on the far edge is excluded (F2:
    # still emitted as an attempted-but-missed cell, n=0/n_attempted=1), while
    # one tick earlier is included.
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    edge_ts = fill.ts + timedelta(seconds=60.0 + MARKOUT_WINDOW_S)

    mid_rows_edge = {"m1": [(edge_ts, 0.55)]}
    report_edge = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows_edge), registry, BELLY_BAND, horizons=(60.0,),
    )
    assert len(report_edge["cells"]) == 1
    assert report_edge["cells"][0]["n"] == 0
    assert report_edge["cells"][0]["n_attempted"] == 1

    mid_rows_just_inside = {"m1": [(edge_ts - timedelta(seconds=1), 0.55)]}
    report_inside = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows_just_inside), registry, BELLY_BAND, horizons=(60.0,),
    )
    assert len(report_inside["cells"]) == 1
    assert report_inside["cells"][0]["n"] == 1
    assert report_inside["cells"][0]["n_attempted"] == 1


def test_markout_disjoint_windows_default_horizons():
    # F1: default horizons (60, 600, 3600) with MARKOUT_WINDOW_S=600 used to
    # overlap (h=60's flat window [60,660] overlapped h=600's [600,1200]);
    # capping each horizon's window at the next horizon's start makes the
    # effective windows [60,600) / [600,1200) / [3600,4200) disjoint.
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}

    # A mid at fill.ts+650 falls in h=600's window [600,1200) but NOT in
    # h=60's capped window [60,600) -> serves ONLY h=600.
    mid_rows = {"m1": [(fill.ts + timedelta(seconds=650), 0.55)]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0, 600.0, 3600.0),
    )
    by_horizon = {c["horizon_s"]: c for c in report["cells"]}
    assert by_horizon[60.0]["n"] == 0
    assert by_horizon[600.0]["n"] == 1
    assert by_horizon[3600.0]["n"] == 0

    # A mid at exactly fill.ts+600 -- h=60's window excludes the 600 boundary
    # (exclusive upper bound), so it serves h=600 only, never both.
    mid_rows_600 = {"m1": [(fill.ts + timedelta(seconds=600), 0.55)]}
    report_600 = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows_600), registry, BELLY_BAND, horizons=(60.0, 600.0, 3600.0),
    )
    by_horizon_600 = {c["horizon_s"]: c for c in report_600["cells"]}
    assert by_horizon_600[60.0]["n"] == 0
    assert by_horizon_600[600.0]["n"] == 1


def test_markout_disjoint_windows_custom_horizons_closer_than_window():
    # Custom ascending horizons closer together than MARKOUT_WINDOW_S (600s)
    # apart stay disjoint: h=60's window caps at the next horizon's start
    # (100), not at a flat 60+600.
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}

    # A mid at fill.ts+150 falls in h=100's window [100,700) but is outside
    # h=60's capped window [60,100).
    mid_rows = {"m1": [(fill.ts + timedelta(seconds=150), 0.55)]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0, 100.0),
    )
    by_horizon = {c["horizon_s"]: c for c in report["cells"]}
    assert by_horizon[60.0]["n"] == 0
    assert by_horizon[100.0]["n"] == 1


def test_markout_duplicate_horizons_deduped():
    # A duplicated horizon must NOT create a zero-width [h, h) window (which
    # would report a permanent n=0 cell): duplicates are de-duped up front.
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(fill.ts + timedelta(seconds=60), 0.55)]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND,
        horizons=(60.0, 60.0, 600.0),
    )
    by_horizon = {c["horizon_s"]: c for c in report["cells"]}
    assert set(by_horizon) == {60.0, 600.0}
    assert by_horizon[60.0]["n"] == 1
    assert by_horizon[60.0]["n_attempted"] == 1


def test_markout_bad_registry_expiry_degrades_to_unknown_bucket():
    # One malformed registry expiry_key must not abort the whole report:
    # the bad market's fill degrades to tte_bucket "unknown" and every other
    # fill still reports normally.
    fill_ok = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="ok")
    fill_bad = _paper_fill("m2", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="bad")
    registry = {"m1": (FAR_EXPIRY, 100000.0), "m2": ("not-a-date", 100000.0)}
    mid_rows = {
        "m1": [(fill_ok.ts + timedelta(seconds=60), 0.55)],
        "m2": [(fill_bad.ts + timedelta(seconds=60), 0.55)],
    }
    report = markout_report(
        [fill_ok, fill_bad], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND,
        horizons=(60.0,),
    )
    buckets = {c["tte_bucket"] for c in report["cells"]}
    assert "unknown" in buckets
    assert any(b != "unknown" for b in buckets)
    for cell in report["cells"]:
        assert cell["n"] == 1


def test_markout_n_attempted_tracks_misses_and_rolls_up():
    # F2: two fills that land in the SAME (region, tte_bucket, horizon) cell;
    # one fill's horizon lookup hits, the other's misses -> the cell reports
    # n=1 (successful lookups) but n_attempted=2 (eligible fills), not just
    # silently dropping the miss; the by_region rollup sums both counts.
    fill_a = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="a")
    fill_b = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="b",
        ts=NOW + timedelta(hours=1),
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    # Only fill_a's window has a mid; fill_b's window (an hour later) is empty.
    mid_rows = {"m1": [(fill_a.ts + timedelta(seconds=60), 0.55)]}
    report = markout_report(
        [fill_a, fill_b], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    assert len(report["cells"]) == 1
    cell = report["cells"][0]
    assert cell["n"] == 1
    assert cell["n_attempted"] == 2

    rollup = report["by_region"][cell["region"]][str(cell["horizon_s"])]
    assert rollup["n"] == 1
    assert rollup["n_attempted"] == 2


def test_markout_now_excludes_fills_older_than_lookback():
    # F3: an explicit `now` bounds the lookback -- a fill older than
    # `now - MARKOUT_LOOKBACK_S` is skipped entirely (not even counted as
    # "attempted"), while a fill within the window is processed normally.
    old_fill = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="old",
        ts=NOW - timedelta(seconds=MARKOUT_LOOKBACK_S + 3600.0),
    )
    recent_fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="recent")
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(recent_fill.ts + timedelta(seconds=60), 0.55)]}
    report = markout_report(
        [old_fill, recent_fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND,
        horizons=(60.0,), now=NOW,
    )
    assert len(report["cells"]) == 1
    assert report["cells"][0]["n_attempted"] == 1
    assert report["lookback_s"] == MARKOUT_LOOKBACK_S


# ---------------------------------------------------------------------------
# Fix 2a: persisted per-fill markouts + decoupled retention n_attempted
# ---------------------------------------------------------------------------


def _paper_fill_id(market_id, side, price, size, mid_at_fill, fid, ts=NOW,
                   liquidity=LiquiditySource.MAKER):
    return PaperFill(
        ts=ts, market_id=market_id, order_id="o-%s" % fid, side=side, price=price,
        size=size, liquidity=liquidity, venue_ts=ts, mid_at_fill=mid_at_fill, id=fid,
    )


def test_markout_persisted_short_circuit_uses_stored_mk_no_lookup():
    # Fix 2a: a (fill, horizon) whose key is in `persisted` uses the stored mk
    # and does NOT call mid_lookup, yet counts toward n and n_attempted
    # exactly like a live hit.
    fill = _paper_fill_id("m1", Side.BUY_YES, 0.50, 5.0, 0.50, fid=7)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    calls = []

    def _lookup(market_id, ts_min, ts_max):
        calls.append(market_id)
        return None

    report = markout_report(
        [fill], _lookup, registry, BELLY_BAND, horizons=(60.0,), now=NOW,
        persisted={(7, 60.0): 0.03},
    )
    assert calls == []  # short-circuited; no live lookup
    cell = report["cells"][0]
    assert cell["n"] == 1
    assert cell["n_attempted"] == 1
    assert abs(cell["mk_avg"] - 0.03) < 1e-12


def test_markout_persist_cb_receives_only_newly_resolved_live_tuples():
    # A live-mid hit on an id'd fill is handed to persist_cb as
    # (id, horizon_s, mk); a persisted hit is NOT re-persisted; an id=None
    # fill contributes nothing even when its mid hits.
    fill_live = _paper_fill_id("m1", Side.BUY_YES, 0.50, 1.0, 0.50, fid=11)
    fill_persisted = _paper_fill_id("m1", Side.BUY_YES, 0.50, 1.0, 0.50, fid=12)
    fill_noid = _paper_fill("m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50)  # id=None
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(NOW + timedelta(seconds=60), 0.55)]}
    captured = []
    report = markout_report(
        [fill_live, fill_persisted, fill_noid], _mid_lookup_from_rows(mid_rows),
        registry, BELLY_BAND, horizons=(60.0,), now=NOW,
        persisted={(12, 60.0): -0.01}, persist_cb=captured.extend,
    )
    # All three share one cell (belly / same tte / 60s): n_attempted == 3.
    assert report["cells"][0]["n_attempted"] == 3
    assert report["cells"][0]["n"] == 3
    # Only fill_live's fresh live-mid hit is newly persisted.
    assert len(captured) == 1
    assert captured[0][0] == 11
    assert captured[0][1] == 60.0
    assert captured[0][2] == pytest.approx(0.05)


def test_markout_report_persist_params_inert_for_idless_fills():
    # An id=None fill (hand-built / legacy) never consults `persisted` and
    # never feeds `persist_cb`, so the report is byte-identical (minus the
    # wall-clock generated_ts) whether the params are omitted or supplied.
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)  # id=None
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(fill.ts + timedelta(seconds=60), 0.55)]}
    lookup = _mid_lookup_from_rows(mid_rows)
    captured = []
    r_plain = markout_report([fill], lookup, registry, BELLY_BAND, horizons=(60.0,), now=NOW)
    r_wired = markout_report(
        [fill], lookup, registry, BELLY_BAND, horizons=(60.0,), now=NOW,
        persisted={(999, 60.0): 0.5}, persist_cb=captured.extend,
    )
    assert captured == []  # id=None -> nothing persisted
    r_plain.pop("generated_ts")
    r_wired.pop("generated_ts")
    assert r_plain == r_wired


def test_markout_window_roll_old_fill_measured_via_persisted():
    # A fill older than MID_LOG_RETENTION_S (its mids pruned -> live lookup
    # MISSES) but within MARKOUT_LOOKBACK_S stays MEASURED via the persisted
    # map -- the whole point of Fix 2a (28d memory on 7d mids).
    old_ts = NOW - timedelta(seconds=MID_LOG_RETENTION_S + 86400.0)  # ~8d old
    fill = _paper_fill_id("m1", Side.BUY_YES, 0.50, 1.0, 0.50, fid=5, ts=old_ts)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}

    def _lookup(market_id, ts_min, ts_max):
        return None  # mids pruned

    report = markout_report(
        [fill], _lookup, registry, BELLY_BAND, horizons=(60.0,), now=NOW,
        persisted={(5, 60.0): -0.04},
    )
    cell = report["cells"][0]
    assert cell["n"] == 1  # measured despite pruned mids
    assert cell["n_attempted"] == 1
    assert abs(cell["mk_avg"] - (-0.04)) < 1e-12


def test_markout_n_attempted_old_miss_neither_young_miss_attempted_only():
    # Fix 2a n_attempted semantics: an OLD (> MID_LOG_RETENTION_S) unresolved
    # fill with no persisted markout counts in NEITHER n nor n_attempted (its
    # mids are gone -- a phantom attempt would falsely trip the measured
    # thresholds); a YOUNG unresolved fill still counts in n_attempted only.
    # Both cells are still EMITTED (n=0) -- emission structure unchanged.
    old_ts = NOW - timedelta(seconds=MID_LOG_RETENTION_S + 86400.0)  # ~8d old
    old_miss = _paper_fill_id("m1", Side.BUY_YES, 0.50, 1.0, 0.90, fid=1, ts=old_ts)  # wing
    young_ts = NOW - timedelta(seconds=3600.0)  # 1h old, within retention
    young_miss = _paper_fill_id("m1", Side.BUY_YES, 0.50, 1.0, 0.50, fid=2, ts=young_ts)  # belly
    registry = {"m1": (FAR_EXPIRY, 100000.0)}

    def _lookup(market_id, ts_min, ts_max):
        return None  # both miss (no mids, no persisted)

    report = markout_report(
        [old_miss, young_miss], _lookup, registry, BELLY_BAND, horizons=(60.0,), now=NOW,
    )
    by_region = {c["region"]: c for c in report["cells"]}
    assert by_region["wing"]["n"] == 0
    assert by_region["wing"]["n_attempted"] == 0   # old miss -> NEITHER counter
    assert by_region["belly"]["n"] == 0
    assert by_region["belly"]["n_attempted"] == 1  # young miss -> attempted only


def test_markout_region_and_tte_bucketing():
    # belly (mid_at_fill inside [0.2, 0.8]), wing (outside), and unknown
    # (mid_at_fill None -- a plain Fill, not a PaperFill) all classify
    # correctly; tte buckets span 0-1d/1-2d/2-4d/4d+ off settlement_instant_
    # utc(expiry_key) - fill.ts.
    settle = settlement_instant_utc(FAR_EXPIRY)

    belly_fill = _paper_fill(
        "m-belly", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50,
        ts=settle - timedelta(hours=12),  # tte ~0.5d -> "0-1d"
    )
    wing_fill = _paper_fill(
        "m-wing", Side.BUY_YES, 0.90, 1.0, mid_at_fill=0.90,
        ts=settle - timedelta(hours=36),  # tte 1.5d -> "1-2d"
    )
    unknown_region_fill = _fill(  # plain Fill -> no mid_at_fill attribute at all
        "m-unknown-region", Side.BUY_YES, 0.50, 1.0, ts=settle - timedelta(days=3),  # tte 3d -> "2-4d"
    )
    far_fill = _paper_fill(
        "m-far", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50,
        ts=settle - timedelta(days=6),  # tte 6d -> "4d+"
    )
    # ts chosen within the F3 default lookback (MARKOUT_LOOKBACK_S, 28 days as
    # of Fix 2a) of the other fills above (whose ts cluster within
    # [settle-6d, settle-12h]); the actual tte value is irrelevant here since
    # the missing registry entry forces tte_bucket="unknown" regardless.
    unknown_registry_fill = _paper_fill(
        "m-not-registered", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50, ts=settle - timedelta(days=5),
    )

    registry = {
        "m-belly": (FAR_EXPIRY, 100000.0),
        "m-wing": (FAR_EXPIRY, 100000.0),
        "m-unknown-region": (FAR_EXPIRY, 100000.0),
        "m-far": (FAR_EXPIRY, 100000.0),
        # "m-not-registered" deliberately omitted from the registry.
    }
    mid_rows = {
        "m-belly": [(belly_fill.ts + timedelta(seconds=60), 0.50)],
        "m-wing": [(wing_fill.ts + timedelta(seconds=60), 0.90)],
        "m-unknown-region": [(unknown_region_fill.ts + timedelta(seconds=60), 0.50)],
        "m-far": [(far_fill.ts + timedelta(seconds=60), 0.50)],
        "m-not-registered": [(unknown_registry_fill.ts + timedelta(seconds=60), 0.50)],
    }
    fills = [belly_fill, wing_fill, unknown_region_fill, far_fill, unknown_registry_fill]

    report = markout_report(fills, _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,))

    labels = {(c["region"], c["tte_bucket"]) for c in report["cells"]}
    assert ("belly", "0-1d") in labels
    assert ("wing", "1-2d") in labels
    assert ("unknown", "2-4d") in labels  # mid_at_fill None -> region "unknown"
    assert ("belly", "4d+") in labels
    assert ("belly", "unknown") in labels  # market_id missing from registry -> tte_bucket "unknown"


def test_markout_region_tagging_belly_band_boundary_is_inclusive():
    # F7: region tagging now goes through config.in_belly_band, the same
    # inclusive-both-ends predicate spread_builder uses -- mid_at_fill
    # exactly AT either belly_band edge must classify as "belly", not "wing".
    belly_lo, belly_hi = BELLY_BAND
    fill_lo = _paper_fill("m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=belly_lo, order_id="lo")
    fill_hi = _paper_fill("m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=belly_hi, order_id="hi")
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(NOW + timedelta(seconds=60), 0.50)]}
    report = markout_report(
        [fill_lo, fill_hi], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    regions = {c["region"] for c in report["cells"]}
    assert regions == {"belly"}


def test_markout_settlement_fills_excluded():
    settlement_fill = _paper_fill(
        "m1", Side.BUY_YES, 0.0, 5.0, mid_at_fill=0.50, liquidity=LiquiditySource.SETTLEMENT,
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(NOW + timedelta(seconds=60), 0.50)]}
    report = markout_report(
        [settlement_fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    assert report["cells"] == []
    assert report["by_region"] == {}


def test_markout_by_region_rollup_collapses_tte_bucket():
    # Two fills in the SAME region but DIFFERENT tte buckets must roll up
    # together in by_region[region][str(horizon)] (collapsed across
    # tte_bucket), while remaining two separate rows in "cells".
    settle = settlement_instant_utc(FAR_EXPIRY)
    fill_a = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 10.0, mid_at_fill=0.50, ts=settle - timedelta(hours=12), order_id="a",
    )
    fill_b = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 10.0, mid_at_fill=0.50, ts=settle - timedelta(hours=36), order_id="b",
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [
        (fill_a.ts + timedelta(seconds=60), 0.60),  # mk = +0.10
        (fill_b.ts + timedelta(seconds=60), 0.40),  # mk = -0.10
    ]}
    report = markout_report(
        [fill_a, fill_b], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    belly_cells = [c for c in report["cells"] if c["region"] == "belly"]
    assert len(belly_cells) == 2  # "0-1d" and "1-2d" stay separate in cells

    rollup = report["by_region"]["belly"]["60.0"]
    assert rollup["n"] == 2
    assert rollup["mk_avg"] == pytest.approx(0.0)  # (+0.10 + -0.10) / 2


# ---------------------------------------------------------------------------
# multi-expiry -- expiry_by_market stamping + expiry_key=None TOTAL mode
# ---------------------------------------------------------------------------


def test_expiry_by_market_stamps_per_market_rows():
    fills = [
        _fill("m-a", Side.BUY_YES, 0.40, 5.0),
        _fill("m-b", Side.BUY_YES, 0.30, 2.0, order_id="o2"),
    ]
    inv = {
        "m-a": ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0),
        "m-b": ContractInv(q=2.0, avg_cost=0.30, q_max=100.0, age_weighted_holding=0.0),
    }
    rows = compute_pnl_rows(
        NOW, None, fills, inv, {"m-a": None, "m-b": None}, {"m-a": None, "m-b": None},
        1000.0, expiry_by_market={"m-a": "2026-07-06", "m-b": "2026-07-07"},
    )
    by_market = {r.market_id: r for r in rows}
    assert by_market["m-a"].expiry_key == "2026-07-06"
    assert by_market["m-b"].expiry_key == "2026-07-07"
    # TOTAL row keeps the passed expiry_key (None = all expiries)
    assert by_market[None].expiry_key is None
    # and is still exactly the sum of per-market rows
    assert by_market[None].realized == pytest.approx(
        by_market["m-a"].realized + by_market["m-b"].realized
    )


def test_expiry_key_none_settlement_breakdown_spans_all_expiries():
    fills = [_fill("m-a", Side.BUY_YES, 0.40, 5.0)]
    inv = {"m-a": ContractInv(q=0.0, avg_cost=0.0, q_max=100.0, age_weighted_holding=0.0)}
    def _sev(expiry_key, pnl):
        return SettlementEvent(
            ts=NOW, settlement_ts=NOW, market_id="m-a", expiry_key=expiry_key,
            strike=98000.0, outcome=SettlementOutcome.YES, spot_used=101000.0,
            spot_source=SpotSource.INTRADAY, q_settled=5.0, payoff=5.0,
            pnl_realized=pnl, excluded_from_gate=False,
        )

    settlements = [_sev("2026-07-06", 3.0), _sev("2026-07-05", 0.5)]
    # None -> BOTH expiries' settlement pnl included
    rows = compute_pnl_rows(NOW, None, fills, inv, {"m-a": None}, {"m-a": None},
                            1000.0, settlements=settlements)
    row_a = next(r for r in rows if r.market_id == "m-a")
    assert row_a.settlement_pnl == pytest.approx(3.5)
    # Legacy single-expiry filter unchanged: only the matching expiry counts.
    rows2 = compute_pnl_rows(NOW, "2026-07-06", fills, inv, {"m-a": None}, {"m-a": None},
                             1000.0, settlements=settlements)
    row_a2 = next(r for r in rows2 if r.market_id == "m-a")
    assert row_a2.settlement_pnl == pytest.approx(3.0)


def test_legacy_default_without_expiry_by_market_unchanged():
    fills = [_fill("m-a", Side.BUY_YES, 0.40, 5.0)]
    inv = {"m-a": ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inv, {"m-a": None}, {"m-a": None}, 1000.0)
    assert all(r.expiry_key == "2026-07-06" for r in rows)


# ---------------------------------------------------------------------------
# multi-expiry -- markout_report by_expiry rollup
# ---------------------------------------------------------------------------


def test_markout_report_by_expiry_rollup():
    ek_a, ek_b = "2026-07-06", "2026-07-07"
    registry = {"m-a": (ek_a, 98000.0), "m-b": (ek_b, 98000.0)}
    fills = [
        _paper_fill("m-a", Side.BUY_YES, 0.40, 5.0, mid_at_fill=0.50, ts=NOW),
        _paper_fill("m-b", Side.BUY_NO, 0.60, 2.0, mid_at_fill=0.60, ts=NOW, order_id="o2"),
    ]

    def _mid_lookup(market_id, ts, ts_max):
        return {"m-a": 0.55, "m-b": 0.58}[market_id]

    report = markout_report(fills, _mid_lookup, registry, (0.2, 0.8), horizons=(60.0,), now=NOW)
    assert set(report["by_expiry"].keys()) == {ek_a, ek_b}
    cell_a = report["by_expiry"][ek_a]["60.0"]
    cell_b = report["by_expiry"][ek_b]["60.0"]
    # BUY_YES @ 0.40 vs mid 0.55 -> +0.15; BUY_NO @ 0.60 vs mid 0.58 -> +0.02
    assert cell_a["n"] == 1 and cell_a["mk_avg"] == pytest.approx(0.15)
    assert cell_b["n"] == 1 and cell_b["mk_avg"] == pytest.approx(0.02)


def test_markout_report_by_expiry_unknown_bucket_for_unregistered():
    fills = [_paper_fill("m-x", Side.BUY_YES, 0.40, 1.0, mid_at_fill=0.50, ts=NOW)]

    def _mid_lookup(market_id, ts, ts_max):
        return None

    report = markout_report(fills, _mid_lookup, {}, (0.2, 0.8), horizons=(60.0,), now=NOW)
    assert set(report["by_expiry"].keys()) == {"unknown"}
    cell = report["by_expiry"]["unknown"]["60.0"]
    assert cell["n"] == 0 and cell["n_attempted"] == 1


# ---------------------------------------------------------------------------
# wave 2 W6: mk_var correctness + markout_stats resolution helper
# ---------------------------------------------------------------------------


def test_mk_var_hand_computed_population_variance():
    # Three fills in the SAME cell with markouts +0.10 / -0.10 / 0.00 (mean 0,
    # population variance = (0.01+0.01+0.0)/3 = 0.02/3). Staggered ts (each 10
    # minutes apart) so each fill's [ts+60, ts+60+window) lookup window only
    # ever matches its OWN mid row (the test double picks the first candidate
    # ts in-window, and identical fill.ts would make every fill match the
    # same row).
    fill_a = _paper_fill("m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50, order_id="a", ts=NOW)
    fill_b = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50, order_id="b",
        ts=NOW + timedelta(minutes=10),
    )
    fill_c = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50, order_id="c",
        ts=NOW + timedelta(minutes=20),
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [
        (fill_a.ts + timedelta(seconds=60), 0.60),  # mk = +0.10
        (fill_b.ts + timedelta(seconds=60), 0.40),  # mk = -0.10
        (fill_c.ts + timedelta(seconds=60), 0.50),  # mk = 0.00
    ]}
    report = markout_report(
        [fill_a, fill_b, fill_c], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND,
        horizons=(60.0,),
    )
    cell = report["cells"][0]
    assert cell["n"] == 3
    assert cell["mk_avg"] == pytest.approx(0.0, abs=1e-12)
    assert cell["mk_var"] == pytest.approx(0.02 / 3.0)

    rollup = report["by_region"][cell["region"]]["60.0"]
    assert rollup["mk_var"] == pytest.approx(0.02 / 3.0)


def test_mk_var_zero_below_two_samples():
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(fill.ts + timedelta(seconds=60), 0.60)]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    cell = report["cells"][0]
    assert cell["n"] == 1
    assert cell["mk_var"] == 0.0


def test_markout_stats_exact_cell_resolution():
    report = {
        "cells": [
            {"region": "belly", "tte_bucket": "0-1d", "horizon_s": 600.0,
             "n": 25, "n_attempted": 30, "mk_avg": 0.02, "mk_var": 0.0004},
        ],
        "by_region": {
            "belly": {"600.0": {"n": 100, "n_attempted": 120, "mk_avg": 0.05, "mk_var": 0.001}},
        },
    }
    mk_avg, mk_var, n, n_attempted = markout_stats(report, "belly", "0-1d", 600.0, min_n=20)
    # exact cell wins over the (also-eligible) region rollup
    assert (mk_avg, mk_var, n, n_attempted) == (0.02, 0.0004, 25, 30)


def test_markout_stats_falls_back_to_region_rollup_when_cell_thin():
    # Measurement comes from the rollup, but n_attempted stays CELL-scoped
    # (8, not 120) -- the W4 exploration gate is per-cell (2026-07-15 fix:
    # the rollup's n_attempted closed the gate on every cell of the region).
    report = {
        "cells": [
            {"region": "belly", "tte_bucket": "0-1d", "horizon_s": 600.0,
             "n": 5, "n_attempted": 8, "mk_avg": 0.02, "mk_var": 0.0004},
        ],
        "by_region": {
            "belly": {"600.0": {"n": 100, "n_attempted": 120, "mk_avg": 0.05, "mk_var": 0.001}},
        },
    }
    mk_avg, mk_var, n, n_attempted = markout_stats(report, "belly", "0-1d", 600.0, min_n=20)
    assert (mk_avg, mk_var, n, n_attempted) == (0.05, 0.001, 100, 8)


def test_markout_stats_null_tuple_reports_cell_n_attempted():
    # Neither the cell nor the region rollup reaches min_n -> null tuple;
    # n_attempted is the CELL's attempted count (per-cell exploration gate),
    # never the rollup's.
    report = {
        "cells": [
            {"region": "belly", "tte_bucket": "0-1d", "horizon_s": 600.0,
             "n": 3, "n_attempted": 5, "mk_avg": 0.02, "mk_var": 0.0004},
        ],
        "by_region": {
            "belly": {"600.0": {"n": 8, "n_attempted": 12, "mk_avg": 0.03, "mk_var": 0.0006}},
        },
    }
    mk_avg, mk_var, n, n_attempted = markout_stats(report, "belly", "0-1d", 600.0, min_n=20)
    assert mk_avg is None and mk_var is None and n == 0
    assert n_attempted == 5


def test_markout_stats_rollup_measurement_keeps_unmeasured_cell_exploring():
    # Live 2026-07-15 deadlock regression: trusted-NEGATIVE region rollup
    # (wing n=23 >= min_n, mk_avg -1.7c) resolved for a cell with ZERO fills
    # (wing/4d+). The measurement must still come back (Kelly leg zeroed),
    # but n_attempted must be the cell's 0 so the sizing W4 gate
    # (n_attempted < min_n) keeps the presence-floor probes flowing --
    # otherwise no orders -> no fills -> the negative verdict can never be
    # re-measured and the whole region stays dark permanently.
    report = {
        "cells": [],  # wing/4d+ never filled
        "by_region": {
            "wing": {"600.0": {"n": 23, "n_attempted": 23,
                               "mk_avg": -0.0167, "mk_var": 0.0021}},
        },
    }
    mk_avg, mk_var, n, n_attempted = markout_stats(report, "wing", "4d+", 600.0, min_n=20)
    assert (mk_avg, mk_var, n) == (-0.0167, 0.0021, 23)
    assert n_attempted == 0  # cell-scoped: exploration gate stays open


def test_markout_stats_never_attempted_returns_zero():
    report = {"cells": [], "by_region": {}}
    assert markout_stats(report, "belly", "0-1d", 600.0, min_n=20) == (None, None, 0, 0)


def test_markout_stats_region_horizon_key_is_str_not_float():
    # W6 required-change: by_region is keyed by str(horizon_s) ("600.0"), not
    # a float -- a report missing the exact cell must still resolve via the
    # str-keyed region rollup, not silently miss. n_attempted is cell-scoped
    # (cell absent -> 0), not the rollup's 60.
    report = {
        "cells": [],
        "by_region": {"wing": {"600.0": {"n": 50, "n_attempted": 60, "mk_avg": -0.01, "mk_var": 0.0002}}},
    }
    mk_avg, mk_var, n, n_attempted = markout_stats(report, "wing", "1-2d", 600.0, min_n=20)
    assert (mk_avg, mk_var, n, n_attempted) == (-0.01, 0.0002, 50, 0)


def test_markout_stats_malformed_report_never_raises():
    assert markout_stats(None, "belly", "0-1d", 600.0, min_n=20) == (None, None, 0, 0)
    assert markout_stats({}, "belly", "0-1d", 600.0, min_n=20) == (None, None, 0, 0)
    assert markout_stats({"cells": "not-a-list"}, "belly", "0-1d", 600.0, min_n=20) == (None, None, 0, 0)
    assert markout_stats({"cells": [{"region": "belly"}]}, "belly", "0-1d", 600.0, min_n=20) == (None, None, 0, 0)
    assert markout_stats(
        {"cells": [], "by_region": {"belly": "not-a-dict"}}, "belly", "0-1d", 600.0, min_n=20
    ) == (None, None, 0, 0)
    assert markout_stats(
        {"cells": [], "by_region": {"belly": {"600.0": "not-a-dict"}}}, "belly", "0-1d", 600.0, min_n=20
    ) == (None, None, 0, 0)


def test_tte_bucket_label_matches_private_alias():
    from market_maker.pnl_report import _tte_bucket
    for tte in (0.0, 0.5, 1.0, 1.5, 2.0, 3.9, 4.0, 10.0):
        assert tte_bucket_label(tte) == _tte_bucket(tte)


# ---------------------------------------------------------------------------
# Maker rebates (accounting layer, 2026-07-13)
# ---------------------------------------------------------------------------


def test_rebate_for_fill_hand_computed_values():
    # 0.20 * 0.07 * p*(1-p) * size, per temp/mm_rebate_accounting_plan.md.
    assert rebate_for_fill(0.5, 1.0) == pytest.approx(0.0035)
    assert rebate_for_fill(0.1, 1.0) == pytest.approx(0.00126)


def test_rebate_for_fill_scales_with_size():
    assert rebate_for_fill(0.5, 10.0) == pytest.approx(0.035)
    assert rebate_for_fill(0.5, 1.0) * 10.0 == pytest.approx(rebate_for_fill(0.5, 10.0))


def test_rebate_for_fill_zero_at_p_zero_and_p_one():
    assert rebate_for_fill(0.0, 5.0) == pytest.approx(0.0)
    assert rebate_for_fill(1.0, 5.0) == pytest.approx(0.0)


def test_rebate_for_fill_side_agnostic():
    # rebate_for_fill takes no side/liquidity argument at all -- a BUY_NO
    # fill stored at YES-scale price 0.6 (C0 convention, never complemented)
    # and a BUY_YES fill at the same 0.6 read the identical price, so they
    # must produce the identical rebate (price*(1-price) is symmetric).
    buy_no_at_06 = rebate_for_fill(0.6, 4.0)
    buy_yes_at_06 = rebate_for_fill(0.6, 4.0)
    assert buy_no_at_06 == pytest.approx(buy_yes_at_06)
    assert buy_no_at_06 == pytest.approx(0.20 * 0.07 * 0.6 * 0.4 * 4.0)


def test_markout_report_rebate_avg_n_matched_maker_only():
    # Two fills sharing one cell (same region/tte_bucket/horizon): fill_a is
    # MAKER (earns rebate), fill_b is TAKER (excluded from rebate, but NOT
    # excluded from the markout report itself -- only SETTLEMENT is). Both
    # mid lookups HIT, so both contribute to mk_avg/n, but rebate_avg must
    # average [reb_share_a, 0.0], not [reb_share_a] alone.
    fill_a = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="a",
        liquidity=LiquiditySource.MAKER,
    )
    fill_b = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="b",
        ts=NOW + timedelta(minutes=10), liquidity=LiquiditySource.TAKER,
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [
        (fill_a.ts + timedelta(seconds=60), 0.55),
        (fill_b.ts + timedelta(seconds=60), 0.55),
    ]}
    report = markout_report(
        [fill_a, fill_b], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    cell = report["cells"][0]
    # Regression: n/n_attempted/mk_avg/mk_var unaffected by the rebate layer.
    assert cell["n"] == 2
    assert cell["n_attempted"] == 2
    assert cell["mk_avg"] == pytest.approx(0.05)
    expected_reb_avg = (rebate_for_fill(0.50, 1.0) + 0.0) / 2.0
    assert cell["rebate_avg"] == pytest.approx(expected_reb_avg)
    assert cell["rebate_avg"] == pytest.approx(0.0035 / 2.0)


def test_markout_report_rebate_avg_excludes_missed_lookups():
    # F2-style n-matching for rebates: fill_a's lookup hits, fill_b's misses.
    # rebate_avg must be computed over ONLY the hit (fill_a's per-share
    # rebate), matching mk_avg's n=1, NOT averaged in fill_b's non-rebate.
    fill_a = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="a",
        liquidity=LiquiditySource.MAKER,
    )
    fill_b = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="b",
        ts=NOW + timedelta(hours=1), liquidity=LiquiditySource.MAKER,
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    # Only fill_a's window has a mid; fill_b's window (an hour later) is empty.
    mid_rows = {"m1": [(fill_a.ts + timedelta(seconds=60), 0.55)]}
    report = markout_report(
        [fill_a, fill_b], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    cell = report["cells"][0]
    assert cell["n"] == 1
    assert cell["n_attempted"] == 2
    assert cell["rebate_avg"] == pytest.approx(rebate_for_fill(0.50, 1.0))


def test_markout_report_rebate_avg_zero_when_n_zero():
    # A cell with n==0 (every lookup missed) must report rebate_avg=0.0, not
    # raise (empty-list mean).
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": []}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    cell = report["cells"][0]
    assert cell["n"] == 0
    assert cell["rebate_avg"] == pytest.approx(0.0)


def test_markout_report_rebate_avg_in_by_region_and_by_expiry_rollups():
    ek_a, ek_b = "2026-07-06", "2026-07-07"
    registry = {"m-a": (ek_a, 98000.0), "m-b": (ek_b, 98000.0)}
    fills = [
        _paper_fill("m-a", Side.BUY_YES, 0.40, 5.0, mid_at_fill=0.50, ts=NOW,
                    liquidity=LiquiditySource.MAKER),
        _paper_fill("m-b", Side.BUY_NO, 0.60, 2.0, mid_at_fill=0.60, ts=NOW, order_id="o2",
                    liquidity=LiquiditySource.MAKER),
    ]

    def _mid_lookup(market_id, ts, ts_max):
        return {"m-a": 0.55, "m-b": 0.58}[market_id]

    report = markout_report(fills, _mid_lookup, registry, (0.2, 0.8), horizons=(60.0,), now=NOW)

    # by_region: both fills land in "belly" (0.50 and 0.60 both inside [0.2,0.8]).
    region_rollup = report["by_region"]["belly"]["60.0"]
    assert region_rollup["rebate_avg"] == pytest.approx(
        (rebate_for_fill(0.40, 1.0) + rebate_for_fill(0.60, 1.0)) / 2.0
    )

    # by_expiry: each expiry has exactly one fill, so its rebate_avg equals
    # that single fill's per-share rebate.
    cell_a = report["by_expiry"][ek_a]["60.0"]
    cell_b = report["by_expiry"][ek_b]["60.0"]
    assert cell_a["rebate_avg"] == pytest.approx(rebate_for_fill(0.40, 1.0))
    assert cell_b["rebate_avg"] == pytest.approx(rebate_for_fill(0.60, 1.0))
    # Regression: pre-existing mk_avg values (test_markout_report_by_expiry_
    # rollup) are unaffected by the additive rebate_avg key.
    assert cell_a["mk_avg"] == pytest.approx(0.15)
    assert cell_b["mk_avg"] == pytest.approx(0.02)


def test_markout_stats_resolution_unchanged_with_rebate_avg_key_present():
    # markout_stats must resolve the SAME (mk_avg, mk_var, n, n_attempted)
    # tuple whether or not the report carries the new additive rebate_avg
    # key -- it deliberately never reads that key (quoting layer not
    # implemented).
    report = {
        "cells": [
            {"region": "belly", "tte_bucket": "0-1d", "horizon_s": 600.0,
             "n": 25, "n_attempted": 30, "mk_avg": 0.02, "mk_var": 0.0004,
             "rebate_avg": 0.0031},
        ],
        "by_region": {
            "belly": {"600.0": {"n": 100, "n_attempted": 120, "mk_avg": 0.05,
                                 "mk_var": 0.001, "rebate_avg": 0.0032}},
        },
    }
    mk_avg, mk_var, n, n_attempted = markout_stats(report, "belly", "0-1d", 600.0, min_n=20)
    assert (mk_avg, mk_var, n, n_attempted) == (0.02, 0.0004, 25, 30)


# ---------------------------------------------------------------------------
# Package E (2026-07-15): side-split markout data + markout_stats_side
# ---------------------------------------------------------------------------


def test_markout_side_split_flat_mid_buy_no_regression():
    # Extends the reviewer's blocker gate (test_markout_flat_mid_buy_no_
    # regression): a BUY_NO fill at YES-scale price 0.60 against a flat YES
    # mid of 0.60 markouts to ~0 at every horizon in the AGGREGATE (already
    # covered) AND in its own "sides"["BUY_NO"] entry; "sides"["BUY_YES"]
    # must be present with n=0 (no BUY_YES fills at all in this report).
    fill = _paper_fill("m1", Side.BUY_NO, 0.60, 10.0, mid_at_fill=0.60)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [
        (NOW + timedelta(seconds=60), 0.60),
        (NOW + timedelta(seconds=600), 0.60),
        (NOW + timedelta(seconds=3600), 0.60),
    ]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0, 600.0, 3600.0),
    )
    assert len(report["cells"]) == 3
    for cell in report["cells"]:
        assert cell["mk_avg"] == pytest.approx(0.0, abs=1e-12)
        sides = cell["sides"]
        assert set(sides.keys()) == {"BUY_YES", "BUY_NO"}
        assert sides["BUY_NO"]["n"] == 1
        assert sides["BUY_NO"]["n_attempted"] == 1
        assert sides["BUY_NO"]["mk_avg"] == pytest.approx(0.0, abs=1e-12)
        assert sides["BUY_YES"]["n"] == 0
        assert sides["BUY_YES"]["n_attempted"] == 0
        assert sides["BUY_YES"]["mk_avg"] == pytest.approx(0.0, abs=1e-12)


def test_markout_side_split_n_sums_to_aggregate_n_per_cell():
    # Two fills landing in the SAME cell, one per side -> per-cell side n
    # values sum to the aggregate cell n.
    fill_yes = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="y",
    )
    fill_no = _paper_fill(
        "m1", Side.BUY_NO, 0.55, 5.0, mid_at_fill=0.55, order_id="n",
        ts=NOW + timedelta(minutes=10),
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [
        (fill_yes.ts + timedelta(seconds=60), 0.52),
        (fill_no.ts + timedelta(seconds=60), 0.50),
    ]}
    report = markout_report(
        [fill_yes, fill_no], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    assert len(report["cells"]) == 1
    cell = report["cells"][0]
    assert cell["n"] == 2
    sides = cell["sides"]
    assert sides["BUY_YES"]["n"] + sides["BUY_NO"]["n"] == cell["n"]
    assert sides["BUY_YES"]["n"] == 1
    assert sides["BUY_NO"]["n"] == 1


def test_markout_side_split_n_attempted_sums_to_aggregate_n_attempted():
    # One side hits, the other side's lookup misses -> side n_attempted
    # values still sum to the aggregate n_attempted (F2-style: attempted
    # counts regardless of hit/miss).
    fill_yes = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="y",
    )
    fill_no = _paper_fill(
        "m1", Side.BUY_NO, 0.55, 5.0, mid_at_fill=0.55, order_id="n",
        ts=NOW + timedelta(hours=1),  # window empty -> lookup misses
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(fill_yes.ts + timedelta(seconds=60), 0.52)]}
    report = markout_report(
        [fill_yes, fill_no], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    cell = report["cells"][0]
    assert cell["n"] == 1
    assert cell["n_attempted"] == 2
    sides = cell["sides"]
    assert sides["BUY_YES"]["n"] == 1 and sides["BUY_YES"]["n_attempted"] == 1
    assert sides["BUY_NO"]["n"] == 0 and sides["BUY_NO"]["n_attempted"] == 1
    assert sides["BUY_YES"]["n_attempted"] + sides["BUY_NO"]["n_attempted"] == cell["n_attempted"]


def test_markout_side_split_present_in_by_region_rollup():
    # Two fills, same region, DIFFERENT tte_bucket, one per side -- the
    # by_region rollup collapses across tte_bucket (existing behavior) AND
    # carries a "sides" key summing across both cells.
    settle = settlement_instant_utc(FAR_EXPIRY)
    fill_yes = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 10.0, mid_at_fill=0.50, order_id="y",
        ts=settle - timedelta(hours=12),  # "0-1d"
    )
    fill_no = _paper_fill(
        "m1", Side.BUY_NO, 0.50, 10.0, mid_at_fill=0.50, order_id="n",
        ts=settle - timedelta(hours=36),  # "1-2d"
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [
        (fill_yes.ts + timedelta(seconds=60), 0.55),
        (fill_no.ts + timedelta(seconds=60), 0.45),
    ]}
    report = markout_report(
        [fill_yes, fill_no], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    rollup = report["by_region"]["belly"]["60.0"]
    assert rollup["n"] == 2
    sides = rollup["sides"]
    assert set(sides.keys()) == {"BUY_YES", "BUY_NO"}
    assert sides["BUY_YES"]["n"] == 1
    assert sides["BUY_NO"]["n"] == 1
    assert sides["BUY_YES"]["n"] + sides["BUY_NO"]["n"] == rollup["n"]
    # mk_avg per side matches the single fill's own markout (n=1 -> mean ==
    # the one value): BUY_YES @ 0.50 vs mid 0.55 -> +0.05; BUY_NO @ 0.50 vs
    # mid 0.45 -> -1*(0.45-0.50) = +0.05.
    assert sides["BUY_YES"]["mk_avg"] == pytest.approx(0.05)
    assert sides["BUY_NO"]["mk_avg"] == pytest.approx(0.05)


def test_markout_side_split_sides_shape_has_no_mk_total():
    # Plan's pinned side-entry shape is exactly {n, n_attempted, mk_avg,
    # mk_var} -- no mk_total (unlike the aggregate cell/rollup dicts, which
    # do carry mk_total).
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(fill.ts + timedelta(seconds=60), 0.55)]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND, horizons=(60.0,),
    )
    cell = report["cells"][0]
    assert "mk_total" in cell  # aggregate cell keeps it
    for side_entry in cell["sides"].values():
        assert set(side_entry.keys()) == {"n", "n_attempted", "mk_avg", "mk_var"}


def test_markout_stats_side_exact_cell_resolution():
    report = {
        "cells": [
            {"region": "belly", "tte_bucket": "0-1d", "horizon_s": 60.0,
             "n": 10, "n_attempted": 12, "mk_avg": 0.01, "mk_var": 0.0002,
             "sides": {
                 "BUY_YES": {"n": 25, "n_attempted": 30, "mk_avg": -0.02, "mk_var": 0.0004},
                 "BUY_NO": {"n": 5, "n_attempted": 8, "mk_avg": 0.01, "mk_var": 0.0001},
             }},
        ],
        "by_region": {
            "belly": {"60.0": {"n": 100, "n_attempted": 120, "mk_avg": 0.0, "mk_var": 0.0,
                                "sides": {
                                    "BUY_YES": {"n": 80, "n_attempted": 90, "mk_avg": -0.05, "mk_var": 0.001},
                                    "BUY_NO": {"n": 20, "n_attempted": 30, "mk_avg": 0.02, "mk_var": 0.0005},
                                }}},
        },
    }
    mk_avg, n = markout_stats_side(report, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20)
    # exact cell's BUY_YES side (n=25 >= min_n=20) wins over the (also-
    # eligible) region rollup.
    assert (mk_avg, n) == (-0.02, 25)


def test_markout_stats_side_falls_back_to_region_rollup_when_cell_thin():
    report = {
        "cells": [
            {"region": "belly", "tte_bucket": "0-1d", "horizon_s": 60.0,
             "n": 10, "n_attempted": 12, "mk_avg": 0.01, "mk_var": 0.0002,
             "sides": {
                 "BUY_YES": {"n": 5, "n_attempted": 8, "mk_avg": -0.02, "mk_var": 0.0004},
                 "BUY_NO": {"n": 5, "n_attempted": 8, "mk_avg": 0.01, "mk_var": 0.0001},
             }},
        ],
        "by_region": {
            "belly": {"60.0": {"n": 100, "n_attempted": 120, "mk_avg": 0.0, "mk_var": 0.0,
                                "sides": {
                                    "BUY_YES": {"n": 80, "n_attempted": 90, "mk_avg": -0.05, "mk_var": 0.001},
                                    "BUY_NO": {"n": 20, "n_attempted": 30, "mk_avg": 0.02, "mk_var": 0.0005},
                                }}},
        },
    }
    # BUY_YES cell-side n=5 < min_n=20 -> falls back to the region rollup's
    # BUY_YES side (n=80 >= 20).
    mk_avg, n = markout_stats_side(report, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20)
    assert (mk_avg, n) == (-0.05, 80)


def test_markout_stats_side_both_thin_returns_null_tuple():
    report = {
        "cells": [
            {"region": "belly", "tte_bucket": "0-1d", "horizon_s": 60.0,
             "n": 3, "n_attempted": 5, "mk_avg": 0.0, "mk_var": 0.0,
             "sides": {
                 "BUY_YES": {"n": 3, "n_attempted": 5, "mk_avg": -0.02, "mk_var": 0.0004},
                 "BUY_NO": {"n": 0, "n_attempted": 0, "mk_avg": 0.0, "mk_var": 0.0},
             }},
        ],
        "by_region": {
            "belly": {"60.0": {"n": 8, "n_attempted": 12, "mk_avg": 0.0, "mk_var": 0.0,
                                "sides": {
                                    "BUY_YES": {"n": 8, "n_attempted": 12, "mk_avg": -0.03, "mk_var": 0.0006},
                                    "BUY_NO": {"n": 0, "n_attempted": 0, "mk_avg": 0.0, "mk_var": 0.0},
                                }}},
        },
    }
    assert markout_stats_side(report, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20) == (None, 0)
    assert markout_stats_side(report, "belly", "0-1d", 60.0, Side.BUY_NO, min_n=20) == (None, 0)


def test_markout_stats_side_region_horizon_key_is_str_not_float():
    # Same CRITICAL TRAP as markout_stats: by_region is keyed by str(h).
    report = {
        "cells": [],
        "by_region": {
            "wing": {"60.0": {"n": 50, "n_attempted": 60, "mk_avg": -0.01, "mk_var": 0.0002,
                               "sides": {
                                   "BUY_YES": {"n": 50, "n_attempted": 60, "mk_avg": -0.04, "mk_var": 0.0009},
                                   "BUY_NO": {"n": 0, "n_attempted": 0, "mk_avg": 0.0, "mk_var": 0.0},
                               }}},
        },
    }
    mk_avg, n = markout_stats_side(report, "wing", "1-2d", 60.0, Side.BUY_YES, min_n=20)
    assert (mk_avg, n) == (-0.04, 50)


def test_markout_stats_side_never_attempted_returns_null_tuple():
    report = {"cells": [], "by_region": {}}
    assert markout_stats_side(report, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20) == (None, 0)


def test_markout_stats_side_malformed_report_never_raises():
    assert markout_stats_side(None, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20) == (None, 0)
    assert markout_stats_side({}, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20) == (None, 0)
    assert markout_stats_side(
        {"cells": "not-a-list"}, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20
    ) == (None, 0)
    assert markout_stats_side(
        {"cells": [{"region": "belly"}]}, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20
    ) == (None, 0)
    assert markout_stats_side(
        {"cells": [{"region": "belly", "tte_bucket": "0-1d", "horizon_s": 60.0, "sides": "not-a-dict"}]},
        "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20,
    ) == (None, 0)
    assert markout_stats_side(
        {"cells": [], "by_region": {"belly": "not-a-dict"}}, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20,
    ) == (None, 0)
    assert markout_stats_side(
        {"cells": [], "by_region": {"belly": {"60.0": "not-a-dict"}}}, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20,
    ) == (None, 0)
    assert markout_stats_side(
        {"cells": [], "by_region": {"belly": {"60.0": {"sides": {"BUY_YES": "not-a-dict"}}}}},
        "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20,
    ) == (None, 0)


def test_markout_stats_side_resolution_unaffected_by_markout_stats_calls():
    # markout_stats and markout_stats_side read from the same report but
    # different keys ("sides" vs the aggregate) -- calling one must not
    # affect the other's resolution (they are pure, independent reads).
    report = {
        "cells": [
            {"region": "belly", "tte_bucket": "0-1d", "horizon_s": 60.0,
             "n": 25, "n_attempted": 30, "mk_avg": 0.02, "mk_var": 0.0004,
             "sides": {
                 "BUY_YES": {"n": 25, "n_attempted": 30, "mk_avg": -0.03, "mk_var": 0.0004},
                 "BUY_NO": {"n": 25, "n_attempted": 30, "mk_avg": 0.01, "mk_var": 0.0004},
             }},
        ],
        "by_region": {},
    }
    agg = markout_stats(report, "belly", "0-1d", 60.0, min_n=20)
    side = markout_stats_side(report, "belly", "0-1d", 60.0, Side.BUY_YES, min_n=20)
    assert agg == (0.02, 0.0004, 25, 30)
    assert side == (-0.03, 25)


# ---------------------------------------------------------------------------
# 2026-08-08 wing-bleed fix, section 2a: extended default horizons
# (60, 600, 3600) -> (60, 600, 3600, 21600, 86400)
# ---------------------------------------------------------------------------


def test_markout_default_horizons_extended_to_slow_and_daily():
    # Section 2a: the DEFAULT horizon tuple appends 21600 (slow sizing
    # channel) and 86400 (diagnostics-only). All five horizons get cells,
    # by_region entries, and side splits.
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [
        (NOW + timedelta(seconds=60), 0.55),
        (NOW + timedelta(seconds=600), 0.55),
        (NOW + timedelta(seconds=3600), 0.55),
        (NOW + timedelta(seconds=21600), 0.55),
        (NOW + timedelta(seconds=86400), 0.55),
    ]}
    report = markout_report([fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND)
    by_horizon = {c["horizon_s"]: c for c in report["cells"]}
    assert set(by_horizon) == {60.0, 600.0, 3600.0, 21600.0, 86400.0}
    for cell in by_horizon.values():
        assert cell["n"] == 1
        assert cell["mk_avg"] == pytest.approx(0.05)
        assert set(cell["sides"].keys()) == {"BUY_YES", "BUY_NO"}
        assert cell["sides"]["BUY_YES"]["n"] == 1
    region = report["cells"][0]["region"]
    assert set(report["by_region"][region].keys()) == {
        "60.0", "600.0", "3600.0", "21600.0", "86400.0"
    }
    for entry in report["by_region"][region].values():
        assert "sides" in entry


def test_markout_new_default_tuple_legacy_horizon_cells_identical():
    # Window-validity regression (section 2a proof): with horizons APPENDED,
    # the 60/600/3600 cells produced by the NEW default tuple must be
    # IDENTICAL to those from an explicit (60.0, 600.0, 3600.0) run --
    # 21600 - 3600 = 18000 >= MARKOUT_WINDOW_S = 600, so 3600's window stays
    # [3600, 4200) and nothing about the existing horizons changes.
    f1 = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, order_id="f1", ts=NOW)
    f2 = _paper_fill(
        "m2", Side.BUY_NO, 0.90, 5.0, mid_at_fill=0.90, order_id="f2",
        ts=NOW + timedelta(minutes=10),
    )
    registry = {"m1": (FAR_EXPIRY, 100000.0), "m2": (FAR_EXPIRY, 110000.0)}
    mid_rows = {
        "m1": [
            (f1.ts + timedelta(seconds=60), 0.55),     # hits 60's [60, 600)
            (f1.ts + timedelta(seconds=650), 0.52),    # hits 600's [600, 1200)
            (f1.ts + timedelta(seconds=4100), 0.48),   # hits 3600's [3600, 4200) -- both runs
            (f1.ts + timedelta(seconds=21650), 0.60),  # hits 21600's window -- NEW run only
        ],
        "m2": [
            (f2.ts + timedelta(seconds=600), 0.88),    # hits 600 only
            (f2.ts + timedelta(seconds=4200), 0.85),   # EXCLUDED from 3600 (exclusive upper bound) in both
        ],
    }
    fills = [f1, f2]
    lookup = _mid_lookup_from_rows(mid_rows)
    report_new = markout_report(fills, lookup, registry, BELLY_BAND, now=NOW)
    report_old = markout_report(
        fills, lookup, registry, BELLY_BAND, horizons=(60.0, 600.0, 3600.0), now=NOW,
    )

    legacy = (60.0, 600.0, 3600.0)
    new_legacy_cells = [c for c in report_new["cells"] if c["horizon_s"] in legacy]
    assert new_legacy_cells == report_old["cells"]
    # 3600's window proof point: the +4100 mid resolves in BOTH runs.
    for rep in (report_new, report_old):
        cell_3600 = next(
            c for c in rep["cells"] if c["region"] == "belly" and c["horizon_s"] == 3600.0
        )
        assert cell_3600["n"] == 1
        assert cell_3600["mk_avg"] == pytest.approx(-0.02)
    # by_region / by_expiry legacy-horizon entries identical too.
    for region, horizons_entry in report_old["by_region"].items():
        for hk, entry in horizons_entry.items():
            assert report_new["by_region"][region][hk] == entry
    for ek, horizons_entry in report_old["by_expiry"].items():
        for hk, entry in horizons_entry.items():
            assert report_new["by_expiry"][ek][hk] == entry
    # The appended horizon accumulates its own fresh data without disturbing
    # the legacy cells above.
    cell_21600 = next(
        c for c in report_new["cells"] if c["region"] == "belly" and c["horizon_s"] == 21600.0
    )
    assert cell_21600["n"] == 1
    assert cell_21600["mk_avg"] == pytest.approx(0.10)


def test_markout_persisted_legacy_horizon_rows_valid_under_new_defaults():
    # Every persisted (fill_id, 60/600/3600) fill_markouts row remains valid
    # under the appended default tuple: the stored mk short-circuits the mid
    # lookup for the legacy horizons; only the two NEW horizons hit the live
    # lookup (attempted-only here, no mids supplied).
    fill = _paper_fill_id("m1", Side.BUY_YES, 0.50, 1.0, 0.50, fid=7)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    calls = []

    def _lookup(market_id, ts_min, ts_max):
        calls.append((ts_min, ts_max))
        return None

    persisted = {(7, 60.0): 0.01, (7, 600.0): -0.02, (7, 3600.0): 0.03}
    report = markout_report([fill], _lookup, registry, BELLY_BAND, now=NOW, persisted=persisted)
    by_horizon = {c["horizon_s"]: c for c in report["cells"]}
    assert by_horizon[60.0]["n"] == 1
    assert by_horizon[60.0]["mk_avg"] == pytest.approx(0.01)
    assert by_horizon[600.0]["n"] == 1
    assert by_horizon[600.0]["mk_avg"] == pytest.approx(-0.02)
    assert by_horizon[3600.0]["n"] == 1
    assert by_horizon[3600.0]["mk_avg"] == pytest.approx(0.03)
    # Only the two new horizons ever reached the live lookup.
    assert len(calls) == 2
    assert by_horizon[21600.0]["n"] == 0
    assert by_horizon[21600.0]["n_attempted"] == 1
    assert by_horizon[86400.0]["n"] == 0
    assert by_horizon[86400.0]["n_attempted"] == 1


def test_markout_86400_attempted_only_when_settlement_within_24h():
    # Plan 2a: 86400 is DIAGNOSTICS-ONLY -- a fill with TTE < 24h can never
    # resolve it (mid_log stops at settlement, so the [86400, 87000) window
    # lies past the last mid). The cell must be attempted-only (n=0,
    # n_attempted=1), never phantom-measured; every pre-settlement horizon
    # still resolves normally.
    expiry = "2026-07-08"
    settle = settlement_instant_utc(expiry)
    fill_ts = settle - timedelta(hours=12)  # TTE 0.5d ("0-1d"), < 24h
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50, ts=fill_ts)
    registry = {"m1": (expiry, 100000.0)}
    # Mids exist only up to settlement (mid_log stops there): nothing at
    # +86400, which is 12h past the settlement instant.
    mid_rows = {"m1": [
        (fill_ts + timedelta(seconds=60), 0.55),
        (fill_ts + timedelta(seconds=600), 0.55),
        (fill_ts + timedelta(seconds=3600), 0.55),
        (fill_ts + timedelta(seconds=21600), 0.55),  # 6h, still pre-settlement
    ]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND,
        now=settle + timedelta(hours=13),
    )
    by_horizon = {c["horizon_s"]: c for c in report["cells"]}
    for h in (60.0, 600.0, 3600.0, 21600.0):
        assert by_horizon[h]["n"] == 1
    assert by_horizon[86400.0]["n"] == 0
    assert by_horizon[86400.0]["n_attempted"] == 1


# ---------------------------------------------------------------------------
# 2026-08-08 wing-bleed fix, section 3a: epoch_ts keyword-only filter
# ---------------------------------------------------------------------------


def test_markout_epoch_ts_skips_pre_epoch_fills_entirely():
    # Section 3a: a fill BEFORE epoch_ts is invisible -- neither n nor
    # n_attempted, at every horizon, aggregate and side channels alike, EVEN
    # when a persisted markout exists for it (the skip precedes resolution)
    # -- and it never feeds persist_cb.
    epoch = NOW - timedelta(days=2)
    pre_wing = _paper_fill_id(
        "m1", Side.BUY_YES, 0.50, 1.0, 0.90, fid=1, ts=epoch - timedelta(hours=1),
    )
    pre_belly = _paper_fill_id(
        "m1", Side.BUY_NO, 0.50, 1.0, 0.50, fid=2, ts=epoch - timedelta(hours=2),
    )
    post = _paper_fill_id("m1", Side.BUY_YES, 0.50, 1.0, 0.50, fid=3, ts=NOW)

    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    captured = []

    def _lookup(market_id, ts_min, ts_max):
        return 0.55  # every eligible window hits -- skips must show in counts

    report = markout_report(
        [pre_wing, pre_belly, post], _lookup, registry, BELLY_BAND,
        horizons=(60.0, 600.0), now=NOW,
        persisted={(2, 60.0): -0.50},  # pre-epoch persisted row must NOT resolve
        persist_cb=captured.extend,
        epoch_ts=epoch,
    )
    # The pre-epoch wing fill emitted NO cell at all, at any horizon.
    assert {c["region"] for c in report["cells"]} == {"belly"}
    assert "wing" not in report["by_region"]
    by_horizon = {c["horizon_s"]: c for c in report["cells"]}
    assert set(by_horizon) == {60.0, 600.0}
    for cell in by_horizon.values():
        # Only the post-epoch fill counts; pre_belly (same cell) is skipped.
        assert cell["n"] == 1
        assert cell["n_attempted"] == 1
        assert cell["sides"]["BUY_YES"]["n"] == 1
        assert cell["sides"]["BUY_NO"]["n"] == 0
        assert cell["sides"]["BUY_NO"]["n_attempted"] == 0
    # Only the post-epoch fill's fresh hits were persisted.
    assert sorted(t[:2] for t in captured) == [(3, 60.0), (3, 600.0)]


def test_markout_epoch_ts_boundary_inclusive_keep():
    # Boundary: a fill with ts == epoch_ts is KEPT (skip is strict `<`).
    epoch = NOW
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50, ts=epoch)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(fill.ts + timedelta(seconds=60), 0.55)]}
    report = markout_report(
        [fill], _mid_lookup_from_rows(mid_rows), registry, BELLY_BAND,
        horizons=(60.0,), now=NOW, epoch_ts=epoch,
    )
    cell = report["cells"][0]
    assert cell["n"] == 1
    assert cell["n_attempted"] == 1
    assert report["epoch_ts"] == epoch.isoformat()


def test_markout_epoch_ts_none_inert_and_additive_key():
    # epoch_ts=None (or omitted) is byte-identical to today's report except
    # for the additive "epoch_ts": None key.
    fill = _paper_fill("m1", Side.BUY_YES, 0.50, 5.0, mid_at_fill=0.50)
    registry = {"m1": (FAR_EXPIRY, 100000.0)}
    mid_rows = {"m1": [(fill.ts + timedelta(seconds=60), 0.55)]}
    lookup = _mid_lookup_from_rows(mid_rows)
    r_omitted = markout_report([fill], lookup, registry, BELLY_BAND, horizons=(60.0,), now=NOW)
    r_explicit = markout_report(
        [fill], lookup, registry, BELLY_BAND, horizons=(60.0,), now=NOW, epoch_ts=None,
    )
    assert r_omitted["epoch_ts"] is None
    assert r_explicit["epoch_ts"] is None
    # Legacy structure untouched around the additive key.
    assert r_omitted["cells"][0]["n"] == 1
    assert r_omitted["lookback_s"] == MARKOUT_LOOKBACK_S
    r_omitted.pop("generated_ts")
    r_explicit.pop("generated_ts")
    assert r_omitted == r_explicit


def test_markout_epoch_ts_max_with_lookback_cutoff_both_directions():
    # Effective skip bound = max(lookback_cutoff, epoch_ts):
    #  - epoch OLDER than the 28d cutoff -> the lookback binds (a fill past
    #    the window stays excluded even though it is post-epoch);
    #  - epoch NEWER than the cutoff -> the epoch binds (a fill inside the
    #    window but pre-epoch is excluded).
    registry = {"m1": (FAR_EXPIRY, 100000.0)}

    def _lookup(market_id, ts_min, ts_max):
        return 0.55  # every eligible window hits

    old_fill = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50, order_id="old",
        ts=NOW - timedelta(seconds=MARKOUT_LOOKBACK_S + 3600.0),  # past 28d
    )
    mid_fill = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50, order_id="mid",
        ts=NOW - timedelta(days=6),  # inside 28d lookback AND 7d retention
    )
    new_fill = _paper_fill(
        "m1", Side.BUY_YES, 0.50, 1.0, mid_at_fill=0.50, order_id="new",
        ts=NOW - timedelta(days=1),
    )
    fills = [old_fill, mid_fill, new_fill]

    r_lookback_binds = markout_report(
        fills, _lookup, registry, BELLY_BAND, horizons=(60.0,), now=NOW,
        epoch_ts=NOW - timedelta(days=40),  # older than the cutoff
    )
    assert len(r_lookback_binds["cells"]) == 1
    assert r_lookback_binds["cells"][0]["n"] == 2           # mid + new
    assert r_lookback_binds["cells"][0]["n_attempted"] == 2

    r_epoch_binds = markout_report(
        fills, _lookup, registry, BELLY_BAND, horizons=(60.0,), now=NOW,
        epoch_ts=NOW - timedelta(days=5),  # newer than the cutoff
    )
    assert len(r_epoch_binds["cells"]) == 1
    assert r_epoch_binds["cells"][0]["n"] == 1              # new only
    assert r_epoch_binds["cells"][0]["n_attempted"] == 1
    assert r_epoch_binds["epoch_ts"] == (NOW - timedelta(days=5)).isoformat()
