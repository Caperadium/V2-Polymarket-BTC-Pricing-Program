"""PnL reporting helpers for the paper runner (plan "MM Monitor Dashboard Page
+ Engine Start/Stop Control", Step 1). Pure functions -- no I/O, no store
access -- so they are trivially unit-testable; the runner (market_maker/
paper_runner.py) is the only caller and owns the store reads.

Cash convention (B1, revised C0)
----------------------------------
``Side`` is BUY_YES / BUY_NO only. ``price`` is ALWAYS on the YES scale,
for BOTH sides and BOTH liquidity kinds -- it is never a "NO price" to be
complemented. This holds because the harness bridge (harness.py:98-105)
un-complements order_lifecycle's sell-YES-via-buy-NO order-placement
convention (BUY_NO quoted @ (1 - ask_price), a NO price) back to the
geometric YES-book price before any fill reaches this module; SETTLEMENT
pseudo-fills (settlement_handler.py:296-301) independently carry
``payoff_yes`` (1.0 on YES, 0.0 on NO), also raw. ``fill_cash`` therefore
uses ``price`` directly for every fill -- the BUY_YES/BUY_NO sign is what
carries the direction, not a per-side price transform. (Prior to the C0 fix
this module complemented MAKER/TAKER BUY_NO prices, which disagreed with
inventory_manager's ``_apply_contract_fill`` -- the untouched reference --
and produced a phantom -0.20/share PnL on every open BUY_NO fill; see
mm_suitability_alignment_plan.md pre-step C0.)

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

Per-region markout report (mm_suitability_alignment_plan.md Change C)
-----------------------------------------------------------------------
``markout_report`` measures whether the pricer's belly-region bias (the
model's softest region per temp/suitability.md) bleeds through the Beuoy
anchor blend into realized fill quality. Same fill price-scale convention as
above (never complemented); see the function docstring for the full
region/tte-bucket/horizon join logic. It reads mids from the durable
``mid_log`` table (``market_maker/state_store.py``'s mid-log design) via an
injected lookup callable -- the runner owns the store read, this module
stays pure.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from market_maker.config import in_belly_band
from market_maker.contracts import ContractInv, Fill, LiquiditySource, SettlementEvent, Side
from market_maker.settlement_handler import settlement_instant_utc
from market_maker.state_store import PnlSnapshot

# Row-volume cap (W4): the runner writes the TOTAL row every tick and
# per-market rows only every Nth tick. The cadence constant lives here so
# both the runner and tests reference one source of truth.
PER_MARKET_SNAPSHOT_EVERY_N_TICKS = 20

# Markout lookup window width (mm_suitability_alignment_plan.md Change C,
# C3): a reprice tick blocks the harness loop for minutes (CLAUDE.md), which
# pauses mid_log writes for that long. 600s tolerates one such reprice gap
# while keeping the mark reasonably close to the intended horizon. F1: each
# horizon's window is additionally capped at the NEXT horizon's start (see
# markout_report), so this constant is a ceiling, not always the realized
# window width.
MARKOUT_WINDOW_S = 600.0

# Markout report lookback (F3): bounds both how far back markout_report ever
# looks for fills AND how far back mid_log rows are retained (state_store.
# prune_mid_log, called by the runner with this same constant) -- the report
# is a rolling Stage-B measurement, not a full-history archive; snapshot the
# state-db before this window rolls off if full-history analysis is needed.
# 7 days comfortably covers the 72h VPS acceptance test with margin.
MARKOUT_LOOKBACK_S = 7 * 86400.0

_TTE_BUCKETS: Tuple[Tuple[float, str], ...] = (
    (1.0, "0-1d"),
    (2.0, "1-2d"),
    (4.0, "2-4d"),
)


def _tte_bucket(tte_days: float) -> str:
    for upper, label in _TTE_BUCKETS:
        if tte_days < upper:
            return label
    return "4d+"


def fill_cash(side: Side, price: float, size: float) -> float:
    """Signed cash flow of one fill, in the YES-equivalent accounting used by
    the realized-PnL identity above.

    ``price`` is already YES-scale for every fill -- MAKER/TAKER and
    SETTLEMENT alike, BUY_YES and BUY_NO alike (C0; see module docstring) --
    so it is used directly, with no per-side/per-liquidity complement.

    BUY_YES is a cash outflow (``-price * size``); BUY_NO is booked as a cash
    inflow in this YES-equivalent frame (``+price * size``), the same sign
    convention ``fold_fills_to_inventory`` uses for ``delta_q`` (BUY_NO
    reduces/shorts the net YES-equivalent position).
    """
    sign = -1.0 if side is Side.BUY_YES else 1.0
    return sign * price * size


def cash_by_market(fills: Sequence[Fill]) -> Dict[str, float]:
    """Fold ``fill_cash`` over a fills list, grouped by ``market_id``. Callers
    pass ``store.get_fills()`` (all markets) at snapshot time -- never an
    in-process running total (B3)."""
    out: Dict[str, float] = {}
    for f in fills:
        out[f.market_id] = out.get(f.market_id, 0.0) + fill_cash(f.side, f.price, f.size)
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
    expiry_by_market: Optional[Dict[str, str]] = None,
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

    Multi-expiry mode (both knobs default to legacy behavior):
      - ``expiry_by_market`` (market_id -> expiry_key, typically the store's
        ``markets`` registry): when given, each PER-MARKET row is stamped
        with that market's OWN expiry instead of the single ``expiry_key``
        argument -- which also fixes the pre-existing single-expiry
        mislabeling where a rolled-over previous event's markets (still in
        the global inventory) were stamped with the CURRENT run's expiry.
        The TOTAL row always keeps ``expiry_key`` as passed.
      - ``expiry_key=None`` means "all expiries" (the multi-expiry
        orchestrator's global TOTAL mode): the settlements breakdown then
        includes EVERY expiry's settlement PnL instead of filtering to one.
    """
    cash = cash_by_market(fills)

    settlement_pnl_by_market: Dict[str, float] = {}
    for ev in settlements or ():
        if ev.pnl_realized is None:
            continue
        if expiry_key is not None and ev.expiry_key != expiry_key:
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
        # Multi-expiry note (documented, not fixed): initial_bankroll is the
        # TOTAL bankroll even when sizing runs on a per-expiry share, so a
        # per-market utilization understates the per-share utilization by
        # ~max_expiries x. Display-only; the TOTAL row's utilization (vs the
        # total bankroll) stays correct.
        utilization = (at_risk / initial_bankroll) if initial_bankroll else 0.0
        row_expiry = (
            expiry_by_market.get(market_id, expiry_key)
            if expiry_by_market is not None else expiry_key
        )

        rows.append(PnlSnapshot(
            ts=now, market_id=market_id, expiry_key=row_expiry,
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


# ---------------------------------------------------------------------------
# Per-region markout report (mm_suitability_alignment_plan.md Change C)
# ---------------------------------------------------------------------------


def _summarize(vals: List[float]) -> Dict[str, object]:
    """Aggregate one cell's (or one by_region rollup's) markout values (F9,
    shared by both the cells loop and the by_region rollup so the n/mk_total/
    mk_avg formula lives in exactly one place). `mk_avg` is 0.0 on an empty
    list (F2: a cell with zero hits is still emitted, never divides by zero).
    Does NOT include `n_attempted` -- callers own that count (it is tracked
    per attempted lookup, not per successful value, so it cannot be derived
    from `vals` alone) and merge it into the returned dict themselves.
    """
    n = len(vals)
    total = sum(vals)
    return {"n": n, "mk_avg": (total / n) if n else 0.0, "mk_total": total}


def markout_report(
    fills: Sequence[Fill],
    mid_lookup: Callable[[str, datetime, datetime], Optional[float]],
    markets_registry: Dict[str, Tuple[str, float]],
    belly_band: Tuple[float, float],
    horizons: Sequence[float] = (60.0, 600.0, 3600.0),
    *,
    now: Optional[datetime] = None,
) -> Dict[str, object]:
    """Per-region, per-TTE-bucket, per-horizon markout report for Stage-B
    paper fills (plan Change C: measure whether the belly's model bias --
    +4.8c at 1-2d growing to +8.6c at 5-7d, temp/suitability.md -- bleeds
    through the Beuoy anchor blend before touching spread terms).

    Pure function -- no store access. `mid_lookup(market_id, ts, ts_max)`
    mirrors `MMStateStore.mid_at_or_after`'s signature exactly (the runner
    passes the bound method directly); the runner owns all store reads, per
    this module's existing "runner owns the store" rule.

    Price-scale ground truth (binding, see plan "Price-scale ground truth"):
    the fill's stored `price` is ALREADY YES-scale for both sides (the
    harness bridge un-complements order_lifecycle's sell-YES-via-buy-NO
    order-placement convention before any fill reaches the fill sim/store),
    so it is NEVER complemented here. Markout baseline:
        mk_h = sign * (mid_h - fill.price),  sign = +1 BUY_YES / -1 BUY_NO.
    A BUY_NO fill at 0.60 against a flat YES mid of 0.60 must therefore
    markout to ~0 at every horizon -- that is the mandatory regression gate.

    Lookback (F3): fills with `ts < now - MARKOUT_LOOKBACK_S` are skipped
    entirely -- not even counted as "attempted" -- since mid_log rows that
    old may already be pruned (`state_store.prune_mid_log`, called by the
    runner with this same constant) and recomputing over unbounded history
    is the problem this bound exists to fix. `now` defaults to the max fill
    timestamp in `fills` (None if `fills` is empty) when not given; the
    runner always passes its own tick `now` explicitly so the report's window
    and the store's prune bound share one anchor (see paper_runner.py).

    For each remaining fill (SETTLEMENT-tagged pseudo-fills excluded -- they
    are not a quoting decision to markout) and each horizon `h` in `horizons`
    (iterated ascending, regardless of the order passed in):
      - the lookup window's lower bound is always `fill.ts + h`; the upper
        bound is capped at the NEXT (sorted) horizon's start rather than a
        flat `h + MARKOUT_WINDOW_S` (F1 -- with the old flat window, adjacent
        horizons closer together than MARKOUT_WINDOW_S apart could serve the
        same mid_log row, collapsing their cells): `hi_s = h + min(
        MARKOUT_WINDOW_S, next_h - h)` for every horizon but the last, `hi_s
        = h + MARKOUT_WINDOW_S` for the last. `mid_lookup`'s upper bound is
        EXCLUSIVE (state_store.mid_at_or_after's `ts < ts_max`, F1), so the
        windows `[h, hi_s)` are disjoint by construction for any ascending
        `horizons`.
      - every (region, tte_bucket, horizon) combination touched by an
        eligible fill is counted in `n_attempted`, whether or not the lookup
        found a mid (F2): a cell is emitted with `n=0` when every lookup for
        it missed, so the report distinguishes "no attempts" (no cell at all)
        from "attempted but no mid data" (n=0, n_attempted>0) from "some/all
        hits" (n>0).
      - region: "belly" if `mid_at_fill` is inside `belly_band`
        (`config.in_belly_band`, F7) else "wing"; "unknown" if
        `fill.mid_at_fill` is None (a plain `Fill`, e.g. hand-built in tests,
        has no `mid_at_fill` -- `getattr` treats that the same as an unset
        `PaperFill` field). `mid_at_fill` is the fill's OWN recorded book mid
        (paper_fill_sim's `_make_record`), already YES-scale -- not a fresh
        `mid_lookup` call.
      - tte_bucket: `(settlement_instant_utc(expiry_key) - fill.ts)` in days,
        via `markets_registry[fill.market_id] -> (expiry_key, strike)`;
        bucketed 0-1d / 1-2d / 2-4d / 4d+. "unknown" if the fill's market_id
        is missing from the registry (defensive fallback only -- every
        quoted market is registered on `PaperTradingLoop` construction).

    Output is JSON-serializable:
        {"cells": [{"region", "tte_bucket", "horizon_s", "n", "n_attempted",
                    "mk_avg", "mk_total"}, ...],
         "by_region": {region: {str(horizon_s): {"n", "n_attempted",
                                                  "mk_avg", "mk_total"}}}
         (rolled up across tte_bucket -- the region x horizon summary is the
         headline "is this region's markout biased" view; the finer
         region x tte_bucket x horizon breakdown lives in "cells"),
         "by_expiry": {expiry_key: {str(horizon_s): {"n", "n_attempted",
                                                      "mk_avg", "mk_total"}}}
         (multi-expiry rollup, keyed by the fill's OWN expiry via
         ``markets_registry``; "unknown" bucket for unregistered markets --
         additive key, existing consumers ignore it),
         "lookback_s": MARKOUT_LOOKBACK_S,
         "generated_ts": iso}
    """
    # De-dup as well as sort: a duplicated horizon would produce a zero-width
    # [h, h) window (unsatisfiable under the exclusive upper bound) and report
    # permanent n=0 for that horizon while still counting n_attempted.
    sorted_horizons = sorted(set(horizons))
    eligible = [f for f in fills if f.liquidity is not LiquiditySource.SETTLEMENT]

    if now is None:
        now = max((f.ts for f in eligible), default=None)
    cutoff = (now - timedelta(seconds=MARKOUT_LOOKBACK_S)) if now is not None else None

    cells: Dict[Tuple[str, str, float], List[float]] = {}
    attempted: Dict[Tuple[str, str, float], int] = {}
    expiry_cells: Dict[Tuple[str, float], List[float]] = {}
    expiry_attempted: Dict[Tuple[str, float], int] = {}

    for f in eligible:
        if cutoff is not None and f.ts < cutoff:
            continue

        mid_at_fill = getattr(f, "mid_at_fill", None)
        if mid_at_fill is None:
            region = "unknown"
        elif in_belly_band(mid_at_fill, belly_band):
            region = "belly"
        else:
            region = "wing"

        reg_entry = markets_registry.get(f.market_id)
        if reg_entry is None:
            tte_bucket = "unknown"
            fill_expiry = "unknown"
        else:
            expiry_key, _strike = reg_entry
            fill_expiry = expiry_key
            try:
                tte_days = (settlement_instant_utc(expiry_key) - f.ts).total_seconds() / 86400.0
                tte_bucket = _tte_bucket(tte_days)
            except Exception:
                # One malformed registry expiry_key must not abort the whole
                # report -- degrade that fill to the "unknown" bucket instead.
                tte_bucket = "unknown"

        sign = 1.0 if f.side is Side.BUY_YES else -1.0

        for idx, h in enumerate(sorted_horizons):
            if idx + 1 < len(sorted_horizons):
                hi_s = h + min(MARKOUT_WINDOW_S, sorted_horizons[idx + 1] - h)
            else:
                hi_s = h + MARKOUT_WINDOW_S

            key = (region, tte_bucket, float(h))
            attempted[key] = attempted.get(key, 0) + 1
            vals = cells.setdefault(key, [])

            ekey = (fill_expiry, float(h))
            expiry_attempted[ekey] = expiry_attempted.get(ekey, 0) + 1
            evals = expiry_cells.setdefault(ekey, [])

            mid_h = mid_lookup(f.market_id, f.ts + timedelta(seconds=h), f.ts + timedelta(seconds=hi_s))
            if mid_h is None:
                continue
            mk = sign * (mid_h - f.price)
            vals.append(mk)
            evals.append(mk)

    cell_list: List[Dict[str, object]] = []
    for key in sorted(cells.keys()):
        region, tte_bucket, h = key
        summary = _summarize(cells[key])
        summary["n_attempted"] = attempted[key]
        cell_list.append({"region": region, "tte_bucket": tte_bucket, "horizon_s": h, **summary})

    by_region: Dict[str, Dict[str, Dict[str, object]]] = {}
    grouped: Dict[Tuple[str, float], List[float]] = {}
    grouped_attempted: Dict[Tuple[str, float], int] = {}
    for key, vals in cells.items():
        region, _bucket, h = key
        grouped.setdefault((region, h), []).extend(vals)
        grouped_attempted[(region, h)] = grouped_attempted.get((region, h), 0) + attempted[key]
    for (region, h), vals in grouped.items():
        summary = _summarize(vals)
        summary["n_attempted"] = grouped_attempted[(region, h)]
        by_region.setdefault(region, {})[str(h)] = summary

    by_expiry: Dict[str, Dict[str, Dict[str, object]]] = {}
    for (ek, h), vals in expiry_cells.items():
        summary = _summarize(vals)
        summary["n_attempted"] = expiry_attempted[(ek, h)]
        by_expiry.setdefault(ek, {})[str(h)] = summary

    return {
        "cells": cell_list,
        "by_region": by_region,
        "by_expiry": by_expiry,
        "lookback_s": MARKOUT_LOOKBACK_S,
        "generated_ts": datetime.now(timezone.utc).isoformat(),
    }
