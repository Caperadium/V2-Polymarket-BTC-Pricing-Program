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
stays pure. Each cell/rollup also carries ``mk_var`` (wave 2 W6, sample
variance of that cell's markouts) so the sizing layer can shrink toward the
measured edge with confidence proportional to the evidence (Baker-McHale on
``mk_var/n``). ``markout_stats`` (wave 2 W6) is the pure resolution helper
the harness uses to pull one (mk_avg, mk_var, n, n_attempted) tuple out of a
report for a given (region, tte_bucket, horizon_s) -- see its own docstring
for the cell -> region-rollup fallback order.

Maker rebates (accounting layer, 2026-07-13)
-----------------------------------------------
Polymarket pays makers 20% of the crypto category's taker-fee pool daily in
pUSD, pro-rata by filled-volume fee-equivalent. ``rebate_for_fill`` estimates
one fill's share of that pool as
``MAKER_REBATE_SHARE_CRYPTO * TAKER_FEE_RATE_CRYPTO * price*(1-price) * size``
(``market_maker/config.py`` constants) -- an ESTIMATE, since it assumes the
pro-rata pool identity returns exactly our own fee-equivalent share (holds
when total maker fee-equivalent == total taker fee-equivalent per market;
unverified against real daily payouts). Eligibility is MAKER-liquidity fills
only (TAKER pays the fee instead of earning a rebate; SETTLEMENT pseudo-fills
are not venue fills) -- the CALLER's job, ``rebate_for_fill`` is just the
formula. ``price`` is used exactly as stored (always YES-scale for both
sides, per the cash convention above), never complemented.

``markout_report`` carries this into two ADDITIVE, n-matched ``rebate_avg``
keys (per ``cells[]`` entry and per ``by_region``/``by_expiry`` rollup entry):
the mean per-share rebate over exactly the fills whose mid lookup HIT (in
lockstep with ``mk_avg`` -- a fill whose horizon lookup missed contributes no
value to either), so ``mk_avg + rebate_avg`` reads as net-of-rebate per-share
fill quality. This is strictly an ACCOUNTING/reporting addition: existing
consumers ignore unknown keys, ``markout_stats`` and the sizing layer it
feeds (``robustness_sizing``) DELIBERATELY do not read ``rebate_avg`` (the
quoting layer -- folding rebate into sizing net edge, spread floor, etc. -- is
NOT implemented). Hard rule, everywhere in this module: rebates never enter
``realized``, ``equity``, ``PnlSnapshot``, the ``pnl`` table, bankroll, or
sizing -- this is a read-only estimate for humans and dashboards.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from market_maker.config import MAKER_REBATE_SHARE_CRYPTO, TAKER_FEE_RATE_CRYPTO, in_belly_band
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

# Markout report lookback (F3; Fix 2a 2026-07-26 DECOUPLED from mid_log
# retention): how far back markout_report looks for fills. As of Fix 2a this
# is NO LONGER the same bound as mid_log retention -- per-fill markouts are
# persisted durably (state_store.fill_markouts) as they resolve, so a fill's
# markout survives after its raw mids are pruned. The report therefore
# remembers a cell's measurement for 28d (killing the weekly re-arm cycle
# where a measured-toxic cell's rolling window rolled off, reverted to the
# optimistic structural prior, and re-bled at full size) while mid_log itself
# is still pruned to the shorter MID_LOG_RETENTION_S below (disk cost
# unchanged). A fill older than MID_LOG_RETENTION_S whose markout was never
# persisted is permanently unresolvable and is excluded from n_attempted (see
# markout_report's n_attempted semantics). Snapshot the state-db before this
# window rolls off if full-history analysis is needed.
MARKOUT_LOOKBACK_S = 28 * 86400.0

# mid_log retention (Fix 2a): how far back raw per-tick mids are kept
# (state_store.prune_mid_log, called by the runner with this constant).
# Deliberately shorter than MARKOUT_LOOKBACK_S -- persisted per-fill markouts
# (fill_markouts) carry a fill's measurement past this window, so mids that
# old are never read again. Also the boundary that gates n_attempted: a miss
# on a fill older than this can never resolve (its mids are gone), so it must
# not inflate n_attempted.
MID_LOG_RETENTION_S = 7 * 86400.0

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


# Public alias (wave 2 W6): the harness needs to classify a market into the
# same tte buckets the report uses, without reaching for the private name.
# The private name stays working (kept as the implementation).
def tte_bucket_label(tte_days: float) -> str:
    return _tte_bucket(tte_days)


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


def rebate_for_fill(price: float, size: float) -> float:
    """Estimated Polymarket maker rebate for ONE eligible fill:
    ``MAKER_REBATE_SHARE_CRYPTO * TAKER_FEE_RATE_CRYPTO * price*(1-price) *
    size`` (see module docstring "Maker rebates" section for the derivation
    and the pro-rata-pool-identity estimate caveat). ``price`` is used
    directly -- already YES-scale for both sides, never complemented (same
    convention as ``fill_cash``). Side-agnostic: ``price*(1-price)`` is
    symmetric, so a BUY_NO fill stored at YES-scale price ``p`` yields the
    identical value a BUY_YES fill at the same ``p`` would.

    Eligibility (MAKER liquidity only) is the CALLER's job -- this is just
    the formula, with no ``liquidity`` argument at all."""
    return MAKER_REBATE_SHARE_CRYPTO * TAKER_FEE_RATE_CRYPTO * price * (1.0 - price) * size


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
    `mk_var` (wave 2 W6) is the population variance of `vals` (denominator n,
    not n-1 -- these are the full observed sample, not an estimate of a
    larger population), 0.0 when n < 2 (undefined/degenerate otherwise).
    Additive key: existing consumers (mm_monitor read-only rendering) ignore
    unknown keys. Does NOT include `n_attempted` -- callers own that count
    (it is tracked per attempted lookup, not per successful value, so it
    cannot be derived from `vals` alone) and merge it into the returned dict
    themselves.
    """
    n = len(vals)
    total = sum(vals)
    mk_avg = (total / n) if n else 0.0
    if n >= 2:
        mk_var = sum((v - mk_avg) ** 2 for v in vals) / n
    else:
        mk_var = 0.0
    return {"n": n, "mk_avg": mk_avg, "mk_var": mk_var, "mk_total": total}


def _rebate_avg(rebate_vals: List[float]) -> float:
    """Mean of a cell's/rollup's parallel per-share rebate list, 0.0 when
    empty (F2-style: n==0 must not divide by zero). Callers append to
    ``rebate_vals`` in exact lockstep with the markout ``vals`` list (only on
    a mid-lookup HIT), so this is always n-matched with ``_summarize``'s
    ``mk_avg`` over the SAME fills -- ``mk_avg + rebate_avg`` is therefore
    net-of-rebate per-share fill quality. Kept separate from ``_summarize``
    (untouched) per the accounting-layer plan."""
    return (sum(rebate_vals) / len(rebate_vals)) if rebate_vals else 0.0


def _side_summaries(
    vals_by_side: Dict[Side, List[float]],
    attempted_by_side: Dict[Side, int],
) -> Dict[str, Dict[str, object]]:
    """Per-side (BUY_YES / BUY_NO) markout summary for one cell/rollup key
    (package E, spread term 7). Additive alongside the existing aggregate
    summary -- see `markout_report`'s "Side-split markout data" docstring
    section. Both sides are ALWAYS present (F2-style: a side with zero hits
    still gets an entry, n=0, never a missing key), so `markout_stats_side`
    can rely on the shape unconditionally. Deliberately narrower than
    `_summarize`'s aggregate shape -- no `mk_total` -- per the plan's pinned
    `{n, n_attempted, mk_avg, mk_var}` side-entry shape.
    """
    out: Dict[str, Dict[str, object]] = {}
    for side in (Side.BUY_YES, Side.BUY_NO):
        summary = _summarize(vals_by_side.get(side, []))
        summary.pop("mk_total", None)
        summary["n_attempted"] = attempted_by_side.get(side, 0)
        out[side.value] = summary
    return out


def markout_report(
    fills: Sequence[Fill],
    mid_lookup: Callable[[str, datetime, datetime], Optional[float]],
    markets_registry: Dict[str, Tuple[str, float]],
    belly_band: Tuple[float, float],
    horizons: Sequence[float] = (60.0, 600.0, 3600.0),
    *,
    now: Optional[datetime] = None,
    persisted: Optional[Dict[Tuple[int, float], float]] = None,
    persist_cb: Optional[Callable[[List[Tuple[int, float, float]]], None]] = None,
) -> Dict[str, object]:
    """Per-region, per-TTE-bucket, per-horizon markout report for Stage-B
    paper fills (plan Change C: measure whether the belly's model bias --
    +4.8c at 1-2d growing to +8.6c at 5-7d, temp/suitability.md -- bleeds
    through the Beuoy anchor blend before touching spread terms).

    No direct store access; all store interaction is via injected callables
    (`mid_lookup`, `persist_cb`). `mid_lookup(market_id, ts, ts_max)` mirrors
    `MMStateStore.mid_at_or_after`'s signature exactly (the runner passes the
    bound method directly); the runner owns all store reads/writes, per this
    module's existing "runner owns the store" rule.

    Persisted markouts (Fix 2a): `persisted` is a `{(fill_id, horizon_s): mk}`
    map of markouts already resolved on an earlier report (state_store.
    get_fill_markouts). A (fill, horizon) whose key is present short-circuits
    the mid lookup and uses the stored mk (counting toward n and n_attempted
    exactly like a live hit) -- this is what lets a fill stay measured for
    MARKOUT_LOOKBACK_S (28d) after its raw mids are pruned at
    MID_LOG_RETENTION_S (7d). On a fresh live-mid hit the `(fill_id,
    horizon_s, mk)` tuple is collected and handed to `persist_cb` once at the
    end (never per-row). Fills with `id=None` (hand-built or off-store
    pseudo-fills) are never persisted and never looked up in `persisted`,
    degrading to the recompute-every-call behavior. The persisted map is
    keyed `(fill_id, float(horizon_s))` and does NOT capture the disjoint-
    window upper bound `hi_s`; safe only because the production horizon set is
    fixed at (60, 600, 3600) -- a changed horizon set could reuse a stored
    markout under a different window width. Both params default to inert, so a
    caller that omits them gets a byte-identical report.

    Price-scale ground truth (binding, see plan "Price-scale ground truth"):
    the fill's stored `price` is ALREADY YES-scale for both sides (the
    harness bridge un-complements order_lifecycle's sell-YES-via-buy-NO
    order-placement convention before any fill reaches the fill sim/store),
    so it is NEVER complemented here. Markout baseline:
        mk_h = sign * (mid_h - fill.price),  sign = +1 BUY_YES / -1 BUY_NO.
    A BUY_NO fill at 0.60 against a flat YES mid of 0.60 must therefore
    markout to ~0 at every horizon -- that is the mandatory regression gate.

    Lookback (F3; Fix 2a): fills with `ts < now - MARKOUT_LOOKBACK_S` (28d)
    are skipped entirely -- not even counted as "attempted". `now` defaults to
    the max fill timestamp in `fills` (None if `fills` is empty) when not
    given; the runner always passes its own tick `now` explicitly. As of Fix
    2a the report lookback (28d) and the mid_log prune bound
    (MID_LOG_RETENTION_S, 7d) are DELIBERATELY different: persisted per-fill
    markouts carry a cell's measurement across the gap, so the report can
    remember 28d while mids are kept only 7d.

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
      - n_attempted semantics (Fix 2a): a (region, tte_bucket, horizon)
        touched by an eligible fill is counted in `n_attempted` iff the fill
        DID resolve at that horizon (persisted markout or a live mid hit --
        also counted in `n`) OR the miss can still resolve later (fill young
        enough that its mids can exist, `fill.ts >= now - MID_LOG_RETENTION_S`).
        A miss on an OLDER fill (mids already pruned, permanently
        unresolvable) is counted in NEITHER `n` NOR `n_attempted` -- otherwise
        phantom attempts would push `n_attempted >= markout_min_n` while `n`
        stays below it, switching OFF both the W4 exploration gate and the
        unmeasured-cell size multiplier on cells that are not actually
        measured (worst exactly in the post-deploy window, when every
        pre-deploy fill is 7-28d old). Invariant: `n_attempted` = "fills that
        did, or still can, yield a measurement." Cell emission is unchanged
        (F2): a cell all of whose fills are old-misses is still emitted with
        `n=0, n_attempted=0`, so the report still distinguishes "no attempts"
        (no cell) from "attempted but no mid data yet" (n=0, n_attempted>0)
        from "some/all hits" (n>0). The same gating applies to every
        n_attempted in the report (cells, sides, by_region, by_expiry).
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

    Maker rebates (2026-07-13, ADDITIVE): every cell/rollup dict below also
    carries a `"rebate_avg"` key -- the mean per-share estimated maker rebate
    (`rebate_for_fill(fill.price, 1.0)` for MAKER-liquidity fills, `0.0` for
    non-MAKER) over exactly the fills that contributed to that same `mk_avg`
    (n-matched: appended only on a mid-lookup HIT, in lockstep with `mk`), so
    `mk_avg + rebate_avg` is net-of-rebate per-share fill quality. `0.0` when
    the cell/rollup has `n == 0`. Existing/older consumers ignore the unknown
    key; `markout_stats` and the sizing layer it feeds deliberately do NOT
    read it (module docstring "Maker rebates" section).

    Side-split markout data (package E, spread term 7, 2026-07-15, ADDITIVE):
    every `cells[]` entry and every `by_region` rollup entry also carries a
    `"sides"` key -- `{"BUY_YES": {"n", "n_attempted", "mk_avg", "mk_var"},
    "BUY_NO": {...}}` -- populated in exact lockstep with the aggregate
    lists above (same eligibility, same mid-lookup-hit gating, same attempted
    counting), just additionally partitioned by the fill's own `Side`
    (BUY_YES fills = our bid side; BUY_NO fills = our ask side; fills are
    YES-scale both sides, see module docstring). Both sides are always
    present, `n=0` when a side has no hits (never a missing key). `by_expiry`
    does NOT get a side split (not needed by the spread-widening consumer).
    `markout_stats_side` is the resolver that reads this key (region rollup
    fallback, mirroring `markout_stats`).

    Output is JSON-serializable:
        {"cells": [{"region", "tte_bucket", "horizon_s", "n", "n_attempted",
                    "mk_avg", "mk_total", "rebate_avg", "sides"}, ...],
         "by_region": {region: {str(horizon_s): {"n", "n_attempted",
                                                  "mk_avg", "mk_total",
                                                  "rebate_avg", "sides"}}}
         (rolled up across tte_bucket -- the region x horizon summary is the
         headline "is this region's markout biased" view; the finer
         region x tte_bucket x horizon breakdown lives in "cells"),
         "by_expiry": {expiry_key: {str(horizon_s): {"n", "n_attempted",
                                                      "mk_avg", "mk_total",
                                                      "rebate_avg"}}}
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
    # Fix 2a: the n_attempted young/old boundary (see docstring). A miss on a
    # fill older than this can never resolve (its mids are pruned), so it must
    # not count as an attempt. None (no `now`) -> treat every fill as young,
    # reproducing the pre-Fix-2a "always count the attempt" behavior.
    mid_retention_cutoff = (
        (now - timedelta(seconds=MID_LOG_RETENTION_S)) if now is not None else None
    )
    # Newly-resolved live-mid markouts collected for one persist_cb call at the
    # end (Fix 2a); id=None fills contribute nothing here.
    newly: List[Tuple[int, float, float]] = []

    cells: Dict[Tuple[str, str, float], List[float]] = {}
    attempted: Dict[Tuple[str, str, float], int] = {}
    expiry_cells: Dict[Tuple[str, float], List[float]] = {}
    expiry_attempted: Dict[Tuple[str, float], int] = {}
    # Parallel per-share-rebate lists (maker-rebate accounting layer): kept
    # separate from `cells`/`expiry_cells` (never fed into `_summarize`,
    # which stays untouched) but appended in exact lockstep with them below.
    reb_cells: Dict[Tuple[str, str, float], List[float]] = {}
    reb_expiry_cells: Dict[Tuple[str, float], List[float]] = {}
    # Side-split markout data (package E, spread term 7): parallel per-side
    # (BUY_YES / BUY_NO) value/attempted trackers, keyed exactly like `cells`/
    # `attempted` and populated in lockstep (same eligibility, same
    # mid-lookup-hit gating, same attempted counting) -- just additionally
    # partitioned by `f.side`. Never fed into `_summarize` directly (see
    # `_side_summaries` below). by_expiry does NOT get a side split (only
    # `cells` and `by_region` do, per plan "Side-split markout data").
    side_cells: Dict[Tuple[str, str, float], Dict[Side, List[float]]] = {}
    side_attempted: Dict[Tuple[str, str, float], Dict[Side, int]] = {}

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
        # Per-share rebate estimate for this fill (MAKER-only; see module
        # docstring "Maker rebates" section) -- constant across horizons
        # since it depends only on the fill's own price, not the lookup.
        reb_share = rebate_for_fill(f.price, 1.0) if f.liquidity is LiquiditySource.MAKER else 0.0
        # Fix 2a: a miss on a fill this old is permanently unresolvable (mids
        # pruned) and must NOT count as an attempt (n_attempted semantics).
        fill_is_young = (mid_retention_cutoff is None) or (f.ts >= mid_retention_cutoff)
        # getattr: a plain Fill (hand-built in tests) has no `id` field, same
        # None-degrade path as the mid_at_fill getattr above -- never persists,
        # never resolves from `persisted`.
        fill_id = getattr(f, "id", None)

        for idx, h in enumerate(sorted_horizons):
            if idx + 1 < len(sorted_horizons):
                hi_s = h + min(MARKOUT_WINDOW_S, sorted_horizons[idx + 1] - h)
            else:
                hi_s = h + MARKOUT_WINDOW_S

            key = (region, tte_bucket, float(h))
            ekey = (fill_expiry, float(h))
            # Ensure the cell/attempted containers exist for EVERY eligible
            # (fill, horizon) so an all-miss cell is still emitted with n=0,
            # n_attempted=0 (F2 emission structure); the counter INCREMENTS
            # happen AFTER resolution below (Fix 2a n_attempted semantics).
            vals = cells.setdefault(key, [])
            attempted.setdefault(key, 0)
            rebs = reb_cells.setdefault(key, [])
            side_att_key = side_attempted.setdefault(key, {})
            evals = expiry_cells.setdefault(ekey, [])
            expiry_attempted.setdefault(ekey, 0)
            erebs = reb_expiry_cells.setdefault(ekey, [])

            # Resolution: persisted markout first (survives mid_log pruning),
            # else the live mid_log lookup. `hkey` is keyed (fill_id,
            # float(h)) only -- it does NOT capture hi_s, safe only because
            # the production horizon set is fixed (see docstring). id=None
            # fills never persist and never resolve from `persisted`.
            hkey = (fill_id, float(h))
            mk: Optional[float] = None
            if fill_id is not None and persisted is not None and hkey in persisted:
                mk = persisted[hkey]
            elif fill_is_young:
                # Live lookup only while the fill's mids can still exist: a
                # miss on an older fill is a guaranteed miss forever (mids
                # pruned), so skipping the query saves hundreds of pointless
                # mid_log probes per report cycle without changing the report
                # (mk stays None -> the old-miss neither-counter branch).
                mid_h = mid_lookup(
                    f.market_id, f.ts + timedelta(seconds=h), f.ts + timedelta(seconds=hi_s)
                )
                if mid_h is not None:
                    mk = sign * (mid_h - f.price)
                    if fill_id is not None:
                        newly.append((fill_id, float(h), mk))

            if mk is not None:
                # Hit (persisted or live): counts in n AND n_attempted. rebs/
                # erebs are appended in lockstep with vals/evals so rebate_avg
                # averages over exactly the same fills as mk_avg.
                vals.append(mk)
                evals.append(mk)
                side_cells.setdefault(key, {}).setdefault(f.side, []).append(mk)
                rebs.append(reb_share)
                erebs.append(reb_share)
                attempted[key] += 1
                expiry_attempted[ekey] += 1
                side_att_key[f.side] = side_att_key.get(f.side, 0) + 1
            elif fill_is_young:
                # Miss but mids can still exist later -> n_attempted only (n
                # unchanged). Same gating for side/expiry attempted counters.
                attempted[key] += 1
                expiry_attempted[ekey] += 1
                side_att_key[f.side] = side_att_key.get(f.side, 0) + 1
            # else: old miss -> counted in NEITHER counter (Fix 2a).

    # Fix 2a: persist every newly-resolved live-mid markout once (never
    # per-row). INSERT OR IGNORE on the store side makes a re-resolve a no-op.
    if persist_cb is not None and newly:
        persist_cb(newly)

    cell_list: List[Dict[str, object]] = []
    for key in sorted(cells.keys()):
        region, tte_bucket, h = key
        summary = _summarize(cells[key])
        summary["n_attempted"] = attempted[key]
        summary["rebate_avg"] = _rebate_avg(reb_cells.get(key, []))
        summary["sides"] = _side_summaries(side_cells.get(key, {}), side_attempted.get(key, {}))
        cell_list.append({"region": region, "tte_bucket": tte_bucket, "horizon_s": h, **summary})

    by_region: Dict[str, Dict[str, Dict[str, object]]] = {}
    grouped: Dict[Tuple[str, float], List[float]] = {}
    grouped_attempted: Dict[Tuple[str, float], int] = {}
    grouped_reb: Dict[Tuple[str, float], List[float]] = {}
    grouped_side: Dict[Tuple[str, float], Dict[Side, List[float]]] = {}
    grouped_side_attempted: Dict[Tuple[str, float], Dict[Side, int]] = {}
    for key, vals in cells.items():
        region, _bucket, h = key
        grouped.setdefault((region, h), []).extend(vals)
        grouped_attempted[(region, h)] = grouped_attempted.get((region, h), 0) + attempted[key]
        grouped_reb.setdefault((region, h), []).extend(reb_cells.get(key, []))
        gs = grouped_side.setdefault((region, h), {})
        for side, svals in side_cells.get(key, {}).items():
            gs.setdefault(side, []).extend(svals)
        gsa = grouped_side_attempted.setdefault((region, h), {})
        for side, satt in side_attempted.get(key, {}).items():
            gsa[side] = gsa.get(side, 0) + satt
    for (region, h), vals in grouped.items():
        summary = _summarize(vals)
        summary["n_attempted"] = grouped_attempted[(region, h)]
        summary["rebate_avg"] = _rebate_avg(grouped_reb.get((region, h), []))
        summary["sides"] = _side_summaries(
            grouped_side.get((region, h), {}), grouped_side_attempted.get((region, h), {})
        )
        by_region.setdefault(region, {})[str(h)] = summary

    by_expiry: Dict[str, Dict[str, Dict[str, object]]] = {}
    for (ek, h), vals in expiry_cells.items():
        summary = _summarize(vals)
        summary["n_attempted"] = expiry_attempted[(ek, h)]
        summary["rebate_avg"] = _rebate_avg(reb_expiry_cells.get((ek, h), []))
        by_expiry.setdefault(ek, {})[str(h)] = summary

    return {
        "cells": cell_list,
        "by_region": by_region,
        "by_expiry": by_expiry,
        "lookback_s": MARKOUT_LOOKBACK_S,
        "generated_ts": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Markout lookup helpers for sizing (wave 2 W6) and quoting (package E)
# ---------------------------------------------------------------------------


def _find_cell(
    report: Optional[dict], region: str, tte_bucket: str, horizon_s: float,
) -> Optional[dict]:
    """Exact `report["cells"]` entry for (region, tte_bucket, horizon_s), or
    None. Shared lookup helper (package E review item -- factor once, use by
    both `markout_stats` and `markout_stats_side`) so the malformed-input
    defensiveness (non-dict report/cells/cell entries) and the horizon match
    (`float(...) == float(horizon_s)`) live in exactly one place. Returns the
    first match; the cells list is built with unique (region, tte_bucket,
    horizon_s) keys, so there is at most one to find. Never raises on
    malformed input -- callers wrap the whole resolution in try/except, but
    this helper itself only reads with `.get`/`isinstance` guards."""
    cells = report.get("cells") if isinstance(report, dict) else None
    if not isinstance(cells, list):
        return None
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if (
            cell.get("region") == region
            and cell.get("tte_bucket") == tte_bucket
            and float(cell.get("horizon_s", float("nan"))) == float(horizon_s)
        ):
            return cell
    return None


def _region_rollup_entry(
    report: Optional[dict], region: str, horizon_s: float,
) -> Optional[dict]:
    """Exact `report["by_region"][region][str(horizon_s)]` entry, or None.
    CRITICAL TRAP (shared by both resolvers): by_region horizons are keyed by
    `str(h)`, e.g. `"600.0"` -- `markout_report` builds this with `str(h)`, so
    a float-keyed lookup here would silently miss every time; this helper is
    the ONE place that string-keys the lookup, so `markout_stats` and
    `markout_stats_side` cannot independently get it wrong."""
    by_region = report.get("by_region") if isinstance(report, dict) else None
    if not isinstance(by_region, dict):
        return None
    region_entry = by_region.get(region)
    if not isinstance(region_entry, dict):
        return None
    horizon_entry = region_entry.get(str(horizon_s))
    if not isinstance(horizon_entry, dict):
        return None
    return horizon_entry


def markout_stats(
    report: Optional[dict],
    region: str,
    tte_bucket: str,
    horizon_s: float,
    min_n: int,
) -> Tuple[Optional[float], Optional[float], int, int]:
    """Resolve (mk_avg, mk_var, n, n_attempted) for one (region, tte_bucket,
    horizon_s) sizing lookup out of a `markout_report()` dict (wave 2 W6 --
    the robustness_sizing markout haircut reads this via the harness, never
    directly, so robustness_sizing itself stays free of any pnl_report
    import).

    Resolution order:
      1. Exact cell from `report["cells"]` (region + tte_bucket + horizon_s,
         matched as float) -- the finest-grained measurement.
      2. If that cell is missing, or its `n < min_n`, fall back to the
         region-only rollup `report["by_region"][region][str(horizon_s)]`
         (NOTE: by_region horizons are keyed by `str(h)`, e.g. "600.0" --
         markout_report builds this with `str(h)`, so a float-keyed lookup
         here would silently miss every time; must match that exactly).
      3. If that is also missing or still `n < min_n`, return
         `(None, None, 0, cell_n_attempted)`.

    n_attempted is ALWAYS the exact CELL's attempted count (0 if the cell is
    absent), even when the measurement itself comes from the region rollup
    (fix 2026-07-15): the wave 2 W4 exploration gate is per-cell by design
    ("a never-measured cell can accumulate fills"), and returning the
    rollup's n_attempted globalized ~23 fills' negative verdict across every
    cell of the region -- gate closed fleet-wide, presence floor off, no
    orders, no new fills, so the measurement could never update (the
    2026-07-15 quote shutdown deadlock). mk_avg/mk_var/n keep rollup
    semantics: a trusted-negative rollup still zeroes the Kelly leg; the
    cell-scoped n_attempted only keeps the exploration probes flowing in
    cells that have not themselves been measured.

    Never raises: a malformed, empty, or None `report`, or a report missing
    expected keys/shapes, degrades to the null tuple `(None, None, 0, 0)`
    rather than propagating a KeyError/TypeError into the sizing pipeline.
    """
    try:
        if not report:
            return None, None, 0, 0

        # Per-cell attempted count -- the ONLY n_attempted this function ever
        # returns (see docstring: the W4 exploration gate is per-cell; the
        # rollup's n_attempted must never leak out of here).
        cell_n_attempted = 0

        cell = _find_cell(report, region, tte_bucket, horizon_s)
        if cell is not None:
            n = int(cell.get("n", 0) or 0)
            cell_n_attempted = int(cell.get("n_attempted", 0) or 0)
            if n >= min_n:
                return (
                    float(cell.get("mk_avg", 0.0)),
                    float(cell.get("mk_var", 0.0)),
                    n,
                    cell_n_attempted,
                )
            # exact cell found but thin; fall through to region rollup

        horizon_entry = _region_rollup_entry(report, region, horizon_s)
        if horizon_entry is not None:
            n = int(horizon_entry.get("n", 0) or 0)
            if n >= min_n:
                return (
                    float(horizon_entry.get("mk_avg", 0.0)),
                    float(horizon_entry.get("mk_var", 0.0)),
                    n,
                    cell_n_attempted,
                )

        return None, None, 0, cell_n_attempted
    except Exception:
        return None, None, 0, 0


def markout_stats_side(
    report: Optional[dict],
    region: str,
    tte_bucket: str,
    horizon_s: float,
    side: Side,
    min_n: int,
) -> Tuple[Optional[float], int]:
    """Resolve (mk_avg, n) for one (region, tte_bucket, horizon_s, side)
    quoting lookup out of a `markout_report()` dict's additive `"sides"` key
    (package E, spread term 7 -- `spread_builder.markout_widen` haircuts the
    posted bid/ask off this per side).

    Resolution order identical to `markout_stats` (same shared helpers,
    `_find_cell` / `_region_rollup_entry`, so the two resolvers cannot drift):
      1. Exact cell's `sides[side.value]` entry, if its `n >= min_n`.
      2. Else the region rollup's `sides[side.value]` entry, if its
         `n >= min_n`.
      3. Else `(None, 0)`.

    Never raises: a malformed, empty, or None `report`, a report/cell/rollup
    missing the `"sides"` key entirely (e.g. an older report predating
    package E), or any other malformed shape degrades to `(None, 0)` rather
    than propagating a KeyError/TypeError into the quoting pipeline.
    """
    try:
        if not report:
            return None, 0

        side_key = side.value

        cell = _find_cell(report, region, tte_bucket, horizon_s)
        if cell is not None:
            sides = cell.get("sides")
            if isinstance(sides, dict):
                side_entry = sides.get(side_key)
                if isinstance(side_entry, dict):
                    n = int(side_entry.get("n", 0) or 0)
                    if n >= min_n:
                        return float(side_entry.get("mk_avg", 0.0)), n

        horizon_entry = _region_rollup_entry(report, region, horizon_s)
        if horizon_entry is not None:
            sides = horizon_entry.get("sides")
            if isinstance(sides, dict):
                side_entry = sides.get(side_key)
                if isinstance(side_entry, dict):
                    n = int(side_entry.get("n", 0) or 0)
                    if n >= min_n:
                        return float(side_entry.get("mk_avg", 0.0)), n

        return None, 0
    except Exception:
        return None, 0
