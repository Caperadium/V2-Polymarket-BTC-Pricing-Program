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
    cash_by_market,
    compute_pnl_rows,
    fill_cash,
    markout_report,
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
    # ts chosen within the F3 default lookback (MARKOUT_LOOKBACK_S, 7 days) of
    # the other fills above (whose ts cluster within [settle-6d, settle-12h])
    # -- default `now` (max fill ts, since this test passes no `now=`) would
    # otherwise exclude a fill as far back as NOW (~14 days before `settle`),
    # and the actual tte value is irrelevant here since the missing registry
    # entry forces tte_bucket="unknown" regardless.
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
