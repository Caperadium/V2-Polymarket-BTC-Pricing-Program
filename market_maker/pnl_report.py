"""PnL reporting helpers for the paper runner (plan "MM Monitor Dashboard Page
+ Engine Start/Stop Control", Step 1). Pure functions -- no I/O, no store
access -- so they are trivially unit-testable; the runner (market_maker/
paper_runner.py) is the only caller and owns the store reads.

Cash convention (B1)
---------------------
``Side`` is BUY_YES / BUY_NO only. Regular (MAKER/TAKER) fills carry the
price of whichever side was bought: a BUY_YES fill's ``price`` is already a
YES price, a BUY_NO fill's ``price`` is a NO price. SETTLEMENT pseudo-fills
(settlement_handler.py:296-301) are the one exception: regardless of which
side they close, ``price`` is always ``payoff_yes`` (1.0 on YES, 0.0 on NO)
-- NOT a NO-price to be complemented. ``fill_cash`` normalizes every fill to
a YES-equivalent price first, then applies the BUY_YES/BUY_NO cash sign.

Recompute-from-store every snapshot (B3)
-----------------------------------------
``cash_by_market`` folds over a fills list (``store.get_fills()``) rather
than accumulating cash in-process across ticks. Restart-safe by
construction: the durable ``fills`` table (which includes SETTLEMENT
pseudo-fills, per settlement_handler.py:318) is the single source of truth.
Fills are sparse in paper trading, so O(n_fills) per tick is trivial.

Realized-PnL identity
----------------------
``realized[m] = cash[m] + q_m * avg_cost_m`` is an exact identity under the
avg_cost bookkeeping in inventory_manager.py:133-146 / state_store.py's
``fold_fills_to_inventory`` (VWAP on same-sign adds, unchanged avg_cost on
reduces, reset to the new fill's price on a flip) -- including settlement
pseudo-fills, which fully close a position (q -> 0) so realized correctly
absorbs the settlement payoff once a market settles.

Equity partition (M1)
-----------------------
``equity = initial_bankroll + realized_total + unrealized_mid_total``.
``settlement_pnl`` (when a ``settlements`` list is supplied) is a
report-only breakdown of how much of ``realized`` came from settlement
payoffs -- it is already INSIDE ``realized`` (via the SETTLEMENT pseudo-fill
cash) and must NEVER be added again when computing equity.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence

from market_maker.contracts import ContractInv, Fill, LiquiditySource, SettlementEvent, Side
from market_maker.state_store import PnlSnapshot

# Row-volume cap (W4): the runner writes the TOTAL row every tick and
# per-market rows only every Nth tick. The cadence constant lives here so
# both the runner and tests reference one source of truth.
PER_MARKET_SNAPSHOT_EVERY_N_TICKS = 20


def fill_cash(side: Side, price: float, size: float, liquidity: LiquiditySource) -> float:
    """Signed cash flow of one fill, in the YES-equivalent accounting used by
    the realized-PnL identity above.

    - SETTLEMENT fills: ``price`` IS already a YES price (payoff_yes) --
      never complemented, regardless of side (B1).
    - Regular BUY_YES fills: ``price`` is a YES price.
    - Regular BUY_NO fills: ``price`` is a NO price -> complement to the
      YES-equivalent (``1 - price``).

    BUY_YES is a cash outflow (``-yes_price * size``); BUY_NO is booked as a
    cash inflow in this YES-equivalent frame (``+yes_price * size``), the
    same sign convention ``fold_fills_to_inventory`` uses for ``delta_q``
    (BUY_NO reduces/shorts the net YES-equivalent position).
    """
    yes_price = price if (liquidity is LiquiditySource.SETTLEMENT or side is Side.BUY_YES) else 1.0 - price
    sign = -1.0 if side is Side.BUY_YES else 1.0
    return sign * yes_price * size


def cash_by_market(fills: Sequence[Fill]) -> Dict[str, float]:
    """Fold ``fill_cash`` over a fills list, grouped by ``market_id``. Callers
    pass ``store.get_fills()`` (all markets) at snapshot time -- never an
    in-process running total (B3)."""
    out: Dict[str, float] = {}
    for f in fills:
        out[f.market_id] = out.get(f.market_id, 0.0) + fill_cash(f.side, f.price, f.size, f.liquidity)
    return out


def _capital_at_risk(q: float, avg_cost: float) -> float:
    """Worst-case capital at risk for one binary position: the max possible
    loss if it settles against you. Long: what you paid (``q * avg_cost``).
    Short (net NO): what you owe if it settles YES (``|q| * (1 - avg_cost)``)."""
    if q > 0.0:
        return q * avg_cost
    if q < 0.0:
        return abs(q) * (1.0 - avg_cost)
    return 0.0


def compute_pnl_rows(
    now: datetime,
    expiry_key: Optional[str],
    fills: Sequence[Fill],
    inventory: Dict[str, ContractInv],
    mids: Dict[str, Optional[float]],
    consensus: Dict[str, Optional[float]],
    initial_bankroll: float,
    settlements: Optional[Sequence[SettlementEvent]] = None,
) -> List[PnlSnapshot]:
    """Build one ``PnlSnapshot`` per market with activity (any market present
    in ``inventory``, i.e. it has at least one fill) plus one aggregate
    ``market_id=None`` TOTAL row (the equity-curve row; dashboard/tests
    filter on ``market_id IS NULL``). Per-market rows are ordered by
    ``market_id`` for determinism; the TOTAL row is appended last and is
    exactly the sum of the per-market rows.

    ``mids`` / ``consensus`` are market_id-keyed (the runner maps the
    strike-keyed ``fv.consensus_p`` / book best_bid/best_ask through the
    ``markets`` ladder before calling this). A missing or ``None`` entry
    means "no current mark" -> that unrealized component is 0.0 for that
    market (mid-None policy).

    ``settlements`` is an *optional* extra source (not in the plan's minimal
    7-arg sketch) used only to populate the report-only ``settlement_pnl``
    breakdown column (sum of ``SettlementEvent.pnl_realized`` for this
    ``expiry_key``, per market) -- omitting it (or passing None) simply
    leaves ``settlement_pnl`` at 0.0 for every row; ``realized`` is
    unaffected either way since it is always derived from ``cash`` +
    ``avg_cost`` alone.
    """
    cash = cash_by_market(fills)

    settlement_pnl_by_market: Dict[str, float] = {}
    for ev in settlements or ():
        if ev.expiry_key != expiry_key or ev.pnl_realized is None:
            continue
        settlement_pnl_by_market[ev.market_id] = settlement_pnl_by_market.get(ev.market_id, 0.0) + ev.pnl_realized

    rows: List[PnlSnapshot] = []
    tot_realized = 0.0
    tot_unrealized_mid = 0.0
    tot_unrealized_consensus = 0.0
    tot_settlement_pnl = 0.0
    tot_at_risk = 0.0

    for market_id in sorted(inventory.keys()):
        ci = inventory[market_id]
        realized = cash.get(market_id, 0.0) + ci.q * ci.avg_cost

        mid = mids.get(market_id)
        unrealized_mid = ci.q * (mid - ci.avg_cost) if mid is not None else 0.0

        cons = consensus.get(market_id)
        unrealized_consensus = ci.q * (cons - ci.avg_cost) if cons is not None else 0.0

        settlement_pnl = settlement_pnl_by_market.get(market_id, 0.0)
        at_risk = _capital_at_risk(ci.q, ci.avg_cost)
        utilization = (at_risk / initial_bankroll) if initial_bankroll else 0.0

        rows.append(PnlSnapshot(
            ts=now, market_id=market_id, expiry_key=expiry_key,
            realized=realized, unrealized_consensus=unrealized_consensus,
            unrealized_mid=unrealized_mid, settlement_pnl=settlement_pnl,
            bankroll_utilization=utilization,
        ))

        tot_realized += realized
        tot_unrealized_mid += unrealized_mid
        tot_unrealized_consensus += unrealized_consensus
        tot_settlement_pnl += settlement_pnl
        tot_at_risk += at_risk

    tot_utilization = (tot_at_risk / initial_bankroll) if initial_bankroll else 0.0
    rows.append(PnlSnapshot(
        ts=now, market_id=None, expiry_key=expiry_key,
        realized=tot_realized, unrealized_consensus=tot_unrealized_consensus,
        unrealized_mid=tot_unrealized_mid, settlement_pnl=tot_settlement_pnl,
        bankroll_utilization=tot_utilization,
    ))
    return rows
