"""Tests for market_maker.pnl_report (plan "MM Monitor Dashboard Page + Engine
Start/Stop Control", Step 1 / Step 6.1).

Every scenario is hand-computed and asserted exactly, including the B1
fill_cash quadrants, the realized-PnL identity (open / partial-reduce /
flip / settlement-close), the B3 restart-equivalence of cash_by_market, the
mid-None policy, long+short utilization, and TOTAL == sum(per-market).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.contracts import ContractInv, Fill, LiquiditySource, SettlementEvent, SettlementOutcome, Side, SpotSource
from market_maker.pnl_report import cash_by_market, compute_pnl_rows, fill_cash

NOW = datetime(2026, 7, 6, 16, 5, tzinfo=timezone.utc)


def _fill(market_id, side, price, size, liquidity=LiquiditySource.MAKER, ts=NOW, order_id="o1"):
    return Fill(ts=ts, market_id=market_id, order_id=order_id, side=side, price=price,
                size=size, liquidity=liquidity, venue_ts=ts)


# ---------------------------------------------------------------------------
# fill_cash -- four quadrants (B1)
# ---------------------------------------------------------------------------


def test_fill_cash_regular_buy_yes():
    # price IS the YES price already; BUY_YES is a cash outflow.
    assert fill_cash(Side.BUY_YES, 0.40, 10.0, LiquiditySource.MAKER) == pytest.approx(-4.0)


def test_fill_cash_regular_buy_no():
    # price is the NO price -> complement to YES-equivalent (1 - 0.7 = 0.3);
    # BUY_NO is booked as a cash inflow in the YES-equivalent frame.
    assert fill_cash(Side.BUY_NO, 0.70, 5.0, LiquiditySource.MAKER) == pytest.approx(1.5)


def test_fill_cash_settlement_buy_yes_no_complement():
    # SETTLEMENT price is ALWAYS payoff_yes, never complemented -- here the
    # market resolved NO (payoff_yes=0.0) and a short position is closed via
    # BUY_YES; 0.0 * size = 0.0, NOT (1 - 0.0) * size.
    assert fill_cash(Side.BUY_YES, 0.0, 5.0, LiquiditySource.SETTLEMENT) == pytest.approx(0.0)


def test_fill_cash_settlement_buy_no_no_complement():
    # SETTLEMENT price is payoff_yes -- here the market resolved YES
    # (payoff_yes=1.0) and a long position is closed via BUY_NO; the closing
    # side does NOT flip the settlement price's meaning (B1's key point).
    assert fill_cash(Side.BUY_NO, 1.0, 5.0, LiquiditySource.SETTLEMENT) == pytest.approx(5.0)


def test_settlement_bypass_differs_from_regular_transform():
    # Same numeric price/size/side, only liquidity differs: a regular BUY_NO
    # fill at "price"=0.0 would complement to yes_price=1.0 (cash=+5.0), but
    # the settlement variant must NOT complement (cash=0.0). This isolates
    # the B1 bypass condition from the BUY_NO sign convention.
    regular = fill_cash(Side.BUY_NO, 0.0, 5.0, LiquiditySource.MAKER)
    settlement = fill_cash(Side.BUY_NO, 0.0, 5.0, LiquiditySource.SETTLEMENT)
    assert regular == pytest.approx(5.0)
    assert settlement == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# realized identity: cash + q*avg_cost
# ---------------------------------------------------------------------------


def test_realized_open_then_partial_reduce():
    # Open 10 BUY_YES @ 0.4 (q=10, avg_cost=0.4). Reduce 5 via BUY_NO @ 0.7
    # (yes-equivalent 0.3): q=5, avg_cost unchanged at 0.4 (fold rule: the
    # remaining lot keeps its existing avg_cost on a partial reduce).
    fills = [
        _fill("m1", Side.BUY_YES, 0.40, 10.0, order_id="open"),
        _fill("m1", Side.BUY_NO, 0.70, 5.0, order_id="reduce"),
    ]
    inventory = {"m1": ContractInv(q=5.0, avg_cost=0.40, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    # cash = -0.4*10 + 0.3*5 = -4.0 + 1.5 = -2.5; realized = -2.5 + 5*0.4 = -0.5
    assert per_market["m1"].realized == pytest.approx(-0.5)


def test_realized_flip_pins_avg_cost_reset():
    # Open 5 BUY_YES @ 0.3 (q=5, avg_cost=0.3). Flip via 8 BUY_NO @ 0.5
    # (yes-equivalent 0.5): closes the 5 long shares and opens a new -3
    # short lot at avg_cost = the flipping fill's cost_basis_price (0.5).
    fills = [
        _fill("m1", Side.BUY_YES, 0.30, 5.0, order_id="open"),
        _fill("m1", Side.BUY_NO, 0.50, 8.0, order_id="flip"),
    ]
    inventory = {"m1": ContractInv(q=-3.0, avg_cost=0.50, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    # cash = -0.3*5 + 0.5*8 = -1.5 + 4.0 = 2.5; realized = 2.5 + (-3)*0.5 = 1.0
    # (closing the 5 long shares bought @0.3 at yes-equiv 0.5 nets +1.0;
    # the fresh -3 lot is unrealized only, correctly excluded here.)
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
    # Open 5 BUY_NO @ 0.7 (cost_basis=0.3, q=-5). Market settles NO
    # (payoff_yes=0.0): closing side is BUY_YES, price=payoff_yes=0.0.
    fills = [
        _fill("m1", Side.BUY_NO, 0.70, 5.0, order_id="open"),
        _fill("m1", Side.BUY_YES, 0.0, 5.0, liquidity=LiquiditySource.SETTLEMENT, order_id="settle"),
    ]
    inventory = {"m1": ContractInv(q=0.0, avg_cost=0.0, q_max=100.0, age_weighted_holding=0.0)}
    rows = compute_pnl_rows(NOW, "2026-07-06", fills, inventory, mids={}, consensus={}, initial_bankroll=1000.0)
    per_market = {r.market_id: r for r in rows if r.market_id is not None}
    # cash = 0.3*5 + 0.0*5 = 1.5 + 0.0 = 1.5; realized = 1.5 + 0*0 = 1.5
    assert per_market["m1"].realized == pytest.approx(1.5)
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
    incremental_total = sum(fill_cash(f.side, f.price, f.size, f.liquidity) for f in fills)
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
    tail_cash = fill_cash(fills[2].side, fills[2].price, fills[2].size, fills[2].liquidity)
    after_restart_full_reload = cash_by_market(fills)["m1"]
    assert after_restart_full_reload == pytest.approx(before_restart + tail_cash)


# ---------------------------------------------------------------------------
# mid-None policy, utilization, TOTAL == sum(per-market)
# ---------------------------------------------------------------------------


def test_mid_none_policy_and_utilization_long_short_mix():
    fills = [
        _fill("m-long", Side.BUY_YES, 0.60, 10.0, order_id="1"),
        _fill("m-short", Side.BUY_NO, 0.75, 4.0, order_id="2"),  # cost_basis=0.25
    ]
    inventory = {
        "m-long": ContractInv(q=10.0, avg_cost=0.60, q_max=100.0, age_weighted_holding=0.0),
        "m-short": ContractInv(q=-4.0, avg_cost=0.25, q_max=100.0, age_weighted_holding=0.0),
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
    # short at_risk = |q|*(1-avg_cost) = 4*0.75=3.0 -> 3/100=0.03
    assert per_market["m-long"].bankroll_utilization == pytest.approx(0.06)
    assert per_market["m-short"].bankroll_utilization == pytest.approx(0.03)
    assert total.bankroll_utilization == pytest.approx(0.09)

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
