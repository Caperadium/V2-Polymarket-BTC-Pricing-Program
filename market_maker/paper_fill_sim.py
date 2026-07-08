"""Paper-trading fill simulator (plan 2.12, task X1, Section 6.3).

THE load-bearing validation component (plan 8.7). Simulates maker fills for OUR
resting orders against a live MarketState stream (book snapshots + aggressor
prints), under deliberately CONSERVATIVE assumptions with zero optimistic
loopholes. Every rule below implements plan Section 6.3 assumptions 1-8 exactly.

Fill-model assumptions (6.3), implemented verbatim:
  1. Queue-behind placement. Joining an EXISTING price level puts us BEHIND the
     entire displayed size at that level at placement time (queue_ahead = that
     size). A NEW best level (price improvement) gives queue_ahead = 0, but a
     fill STILL requires an actual print at (or through) our price -- improving
     the book alone earns nothing.
  2. Fill trigger = observed aggressor prints ONLY. For our BID at Pb, a print
     at price <= Pb first consumes queue_ahead, then fills max(0, print_size -
     queue_ahead) capped at our remaining size (symmetric for asks, print >=
     Pa). Queue-ahead additionally decreases via observed level-size reductions
     ONLY when attributable to cancels ahead (level size dropped with NO
     fill-triggering print at our level in the same update). If a print is
     present (ambiguous), queue_ahead is NOT reduced beyond the print logic.
  3. No price improvement -- fills occur at OUR quoted price exactly, never
     better.
  4. Latency -- placements take effect placement_latency_ms after the decision
     timestamp; cancels take effect cancel_latency_ms after the cancel decision.
     During the cancel window the quote is LIVE and can be hit (we own our stale
     quotes). Defaults from MMConfig (2000/2000 ms).
  5. Adverse-selection marks -- every PaperFill records mid_at_fill; mark_fills()
     backfills mid_p1m/mid_p10m/mid_p1h once their horizons elapse (None until).
  6. Feed gaps -- when a market's feed_healthy flag is False on a snapshot,
     that market's live quotes are marked "exposed" (an exposure incident is
     recorded with start/end + duration); NO fills are simulated for that
     market inside a gap. Gap state is tracked PER market_id (one simulator
     instance commonly serves a whole ladder, fed per-market MarketStates in
     a loop each tick) -- see _detect_gap for why this is feed_healthy-only,
     not dt-based.
  7. Self-impact -- NONE. Our simulated orders never tighten the sim book, so the
     displayed sizes we queue behind are the real book's (this is optimistic in
     the OTHER direction, mitigated by assumptions 1 and 3; residual optimism is
     in the risk register 8.7).
  8. Every PaperFill carries assumption_set (a version constant) so gate metrics
     can never silently mix fill-model versions. The strictly-more-conservative
     TOP-OF-BOOK fallback (trade_through_only=True) fills ONLY on a print STRICTLY
     through our level, queue assumed infinite at our exact price.

Deterministic: identical input streams produce identical fills. stdlib + numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from market_maker.config import MMConfig
from market_maker.contracts import LiquiditySource, MarketState, PaperFill, Side

# Assumption-set version constants (6.3.8).
ASSUMPTION_QUEUEBEHIND = "fillmodel-v1-queuebehind"
ASSUMPTION_TRADETHROUGH = "fillmodel-v1-tradethrough"

_PRICE_TOL = 1e-9
_ADVERSE_HORIZONS = (
    ("mid_p1m", 60.0),
    ("mid_p10m", 600.0),
    ("mid_p1h", 3600.0),
)


# ---------------------------------------------------------------------------
# Public value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureIncident:
    """A stale-quote exposure window (6.3.6). end/duration_s None while open."""
    start: datetime
    end: Optional[datetime]
    duration_s: Optional[float]
    n_live_orders: int
    order_ids: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal mutable state
# ---------------------------------------------------------------------------


@dataclass
class _Order:
    order_id: str
    market_id: str
    side: str  # "bid" or "ask" (geometric book side)
    price: float
    size: float
    remaining: float
    placed_effective_ts: datetime
    queue_ahead: float = 0.0
    last_level_size: float = 0.0
    activated: bool = False
    cancel_effective_ts: Optional[datetime] = None

    def is_bid(self) -> bool:
        return self.side == "bid"

    def triggers_on(self, print_price: float, trade_through_only: bool) -> bool:
        """Does a print at print_price fill/consume-queue for this order?"""
        if trade_through_only:
            # Strictly THROUGH our level only; queue infinite at our exact price.
            if self.is_bid():
                return print_price < self.price - _PRICE_TOL
            return print_price > self.price + _PRICE_TOL
        if self.is_bid():
            return print_price <= self.price + _PRICE_TOL
        return print_price >= self.price - _PRICE_TOL


@dataclass
class _FillRecord:
    ts: datetime
    market_id: str
    order_id: str
    side: Side
    price: float
    size: float
    mid_at_fill: Optional[float]
    queue_ahead_at_fill: float
    print_size: float
    latency_applied_ms: int
    assumption_set: str
    mid_p1m: Optional[float] = None
    mid_p10m: Optional[float] = None
    mid_p1h: Optional[float] = None

    def to_paperfill(self) -> PaperFill:
        return PaperFill(
            ts=self.ts,
            market_id=self.market_id,
            order_id=self.order_id,
            side=self.side,
            price=self.price,
            size=self.size,
            liquidity=LiquiditySource.MAKER,
            venue_ts=self.ts,
            queue_ahead_at_fill=self.queue_ahead_at_fill,
            print_size=self.print_size,
            latency_applied_ms=self.latency_applied_ms,
            assumption_set=self.assumption_set,
            mid_at_fill=self.mid_at_fill,
            mid_p1m=self.mid_p1m,
            mid_p10m=self.mid_p10m,
            mid_p1h=self.mid_p1h,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_num(v: Optional[float]) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _level_size(depth: List[Tuple[float, float]], price: float) -> float:
    for lp, ls in depth:
        if abs(lp - price) <= _PRICE_TOL:
            return float(ls)
    return 0.0


def _mid(ms: MarketState) -> Optional[float]:
    b, a = ms.best_bid, ms.best_ask
    if _is_num(b) and _is_num(a):
        return (float(b) + float(a)) / 2.0
    if _is_num(b):
        return float(b)
    if _is_num(a):
        return float(a)
    return None


def _side_enum(side: str) -> Side:
    # A resting bid that gets hit acquires YES (BUY_YES); a resting ask that gets
    # lifted takes the opposite exposure (BUY_NO in the YES-signed inventory
    # convention). Fill price is always our own quoted price (assumption 3).
    return Side.BUY_YES if side == "bid" else Side.BUY_NO


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class PaperFillSimulator:
    """Conservative maker-fill simulator. Feed it MarketState snapshots via
    on_market_state(); it returns the PaperFills produced by that snapshot.

    trade_through_only=True selects the strictly-more-conservative top-of-book
    fallback variant (plan 2.14): fills only on a print strictly through our
    level.

    One instance may serve an entire ladder (the harness feeds it per-market
    MarketStates in a loop each tick). All gap-tracking state (_last_ms_ts,
    _in_gap, _open_incident) is therefore keyed PER market_id -- a single
    shared "previous timestamp" would make the first market processed each
    tick see dt = the full tick interval against the LAST market processed
    last tick, a false gap unrelated to that market's own feed. See
    _detect_gap for the gap criterion itself.
    """

    def __init__(self, config: Optional[MMConfig] = None,
                 trade_through_only: bool = False) -> None:
        self._cfg = config or MMConfig()
        self._trade_through_only = bool(trade_through_only)
        self._assumption_set = (
            ASSUMPTION_TRADETHROUGH if self._trade_through_only
            else ASSUMPTION_QUEUEBEHIND
        )
        self._orders: Dict[str, _Order] = {}
        self._fills: List[_FillRecord] = []
        self._incidents: List[ExposureIncident] = []
        # Per-market_id gap state (see class docstring).
        self._open_incident: Dict[str, Optional[dict]] = {}
        self._last_ms_ts: Dict[str, Optional[datetime]] = {}
        self._in_gap: Dict[str, bool] = {}

    # -- lifecycle -------------------------------------------------------

    def place(self, order_id: str, market_id: str, side: str, price: float,
              size: float, decision_ts: datetime) -> None:
        """Register a resting order. Takes effect placement_latency_ms after
        decision_ts (assumption 4). side is "bid" or "ask"."""
        s = str(side).lower()
        if s not in ("bid", "ask"):
            raise ValueError("side must be 'bid' or 'ask', got " + repr(side))
        if order_id in self._orders:
            raise ValueError("duplicate order_id " + repr(order_id))
        eff = decision_ts + timedelta(milliseconds=self._cfg.placement_latency_ms)
        self._orders[order_id] = _Order(
            order_id=order_id, market_id=market_id, side=s, price=float(price),
            size=float(size), remaining=float(size), placed_effective_ts=eff,
        )

    def cancel(self, order_id: str, decision_ts: datetime) -> None:
        """Request cancel. Effective cancel_latency_ms after decision_ts; the
        quote is LIVE and hittable until then (assumption 4)."""
        o = self._orders.get(order_id)
        if o is None:
            return
        eff = decision_ts + timedelta(milliseconds=self._cfg.cancel_latency_ms)
        # Never move an existing cancel later; keep the earliest decision.
        if o.cancel_effective_ts is None or eff < o.cancel_effective_ts:
            o.cancel_effective_ts = eff

    # -- market data -----------------------------------------------------

    def on_market_state(self, ms: MarketState) -> List[PaperFill]:
        """Advance the sim by one snapshot; return PaperFills produced now.

        All gap/activation/queue-maintenance bookkeeping below is scoped to
        ms.market_id -- one simulator instance may serve a whole ladder, and
        a snapshot for market A must never read or mutate market B's state.
        """
        mkt = ms.market_id
        gap = self._detect_gap(ms)

        if gap:
            self._enter_gap(ms)
            self._last_ms_ts[mkt] = ms.ts
            return []

        recovering = self._in_gap.get(mkt, False)
        if recovering:
            self._exit_gap(ms)

        self._activate_pending(ms)
        if not recovering:
            self._apply_cancel_ahead(ms)
        else:
            # Reset level baselines across the gap; do NOT attribute the jump.
            # Scoped to this market only -- other markets' orders were never
            # in a gap and their baselines must not move on this snapshot.
            for o in self._orders.values():
                if o.market_id != mkt:
                    continue
                if o.activated:
                    o.last_level_size = _level_size(
                        ms.bid_depth if o.is_bid() else ms.ask_depth, o.price)

        new_records = self._process_prints(ms)

        self._prune(ms)
        self._last_ms_ts[mkt] = ms.ts
        return [r.to_paperfill() for r in new_records]

    # -- gap handling (6.3.6) -------------------------------------------

    def _detect_gap(self, ms: MarketState) -> bool:
        """Per-market feed-loss check.

        Gap = `not ms.feed_healthy` ONLY. There is deliberately no dt-based
        arm: feed_healthy is the connection-liveness override threaded in by
        the runner (P0b design) -- message silence on a quiet book is NOT
        feed loss, and BookMirror already owns message staleness. This sim
        only ever sees tick-cadence snapshots, so the elapsed time between
        two calls carries no gap information of its own; a shared simulator
        fed multiple markets per tick would otherwise see the FIRST market
        of a tick appear dt = tick_interval after the LAST market of the
        PREVIOUS tick and be wrongly declared gapped every single tick.
        `feed_gap_threshold_s` stays in MMConfig for BookMirror.is_stale
        only -- it is not consulted here.
        """
        return not ms.feed_healthy

    def _enter_gap(self, ms: MarketState) -> None:
        mkt = ms.market_id
        self._in_gap[mkt] = True
        if self._open_incident.get(mkt) is None:
            live = self._live_order_ids(ms.ts, mkt)
            prev = self._last_ms_ts.get(mkt)
            start = prev if prev is not None else ms.ts
            self._open_incident[mkt] = {
                "start": start,
                "order_ids": tuple(live),
            }

    def _exit_gap(self, ms: MarketState) -> None:
        mkt = ms.market_id
        self._in_gap[mkt] = False
        inc = self._open_incident.get(mkt)
        if inc is not None:
            start = inc["start"]
            oids = inc["order_ids"]
            self._incidents.append(ExposureIncident(
                start=start, end=ms.ts,
                duration_s=(ms.ts - start).total_seconds(),
                n_live_orders=len(oids), order_ids=oids,
            ))
            self._open_incident[mkt] = None

    def _live_order_ids(self, now: datetime, market_id: str) -> List[str]:
        out = []
        for o in self._orders.values():
            if o.market_id != market_id:
                continue
            if now < o.placed_effective_ts:
                continue
            if o.cancel_effective_ts is not None and now >= o.cancel_effective_ts:
                continue
            if o.remaining <= _PRICE_TOL:
                continue
            out.append(o.order_id)
        return out

    # -- activation / queue maintenance ---------------------------------

    def _activate_pending(self, ms: MarketState) -> None:
        for o in self._orders.values():
            if o.market_id != ms.market_id:
                continue
            if o.activated:
                continue
            if ms.ts >= o.placed_effective_ts:
                lvl = _level_size(ms.bid_depth if o.is_bid() else ms.ask_depth,
                                  o.price)
                o.queue_ahead = lvl  # full displayed size = queue behind (6.3.1)
                o.last_level_size = lvl
                o.activated = True

    def _apply_cancel_ahead(self, ms: MarketState) -> None:
        """Reduce queue_ahead by level-size drops attributable to cancels ahead
        (6.3.2): only when NO fill-triggering print at our level this update."""
        prints = ms.last_prints
        for o in self._orders.values():
            if o.market_id != ms.market_id:
                continue
            if not o.activated:
                continue
            depth = ms.bid_depth if o.is_bid() else ms.ask_depth
            cur = _level_size(depth, o.price)
            drop = o.last_level_size - cur
            if drop > 0:
                has_print = any(
                    o.triggers_on(pp, self._trade_through_only)
                    for (_pts, pp, _psz) in prints
                )
                if not has_print:
                    o.queue_ahead = max(0.0, o.queue_ahead - drop)
            o.last_level_size = cur

    # -- fills (6.3.2 / 6.3.3) ------------------------------------------

    def _process_prints(self, ms: MarketState) -> List[_FillRecord]:
        records: List[_FillRecord] = []
        mid = _mid(ms)
        prints = sorted(ms.last_prints, key=lambda p: p[0])
        for p_ts, p_price, p_size in prints:
            avail = float(p_size)
            if avail <= 0:
                continue
            # Cascade one print's volume across our orders in placement order so
            # a single print can never double-count fills (conservative).
            for o in self._orders.values():
                if avail <= _PRICE_TOL:
                    break
                if o.market_id != ms.market_id:
                    continue
                if not o.activated or o.remaining <= _PRICE_TOL:
                    continue
                if p_ts < o.placed_effective_ts:
                    continue  # placement latency not yet elapsed (6.3.4)
                if o.cancel_effective_ts is not None and p_ts >= o.cancel_effective_ts:
                    continue  # cancel has taken effect (still live before it)
                if not o.triggers_on(p_price, self._trade_through_only):
                    continue

                if self._trade_through_only:
                    # Queue infinite at our price; a strictly-through print proves
                    # the level was consumed. Fill up to the observed print size
                    # (conservative under-fill vs the full remaining).
                    fill = min(avail, o.remaining)
                    if fill <= _PRICE_TOL:
                        continue
                    o.remaining -= fill
                    avail -= fill
                    records.append(self._make_record(
                        o, p_ts, fill, mid, o.queue_ahead, float(p_size)))
                    continue

                consumed = min(o.queue_ahead, avail)
                o.queue_ahead -= consumed
                avail -= consumed
                if avail <= _PRICE_TOL or o.queue_ahead > _PRICE_TOL:
                    continue  # queue not yet exhausted -> no fill this print
                fill = min(avail, o.remaining)
                if fill <= _PRICE_TOL:
                    continue
                o.remaining -= fill
                avail -= fill
                records.append(self._make_record(
                    o, p_ts, fill, mid, o.queue_ahead, float(p_size)))
        self._fills.extend(records)
        return records

    def _make_record(self, o: _Order, ts: datetime, fill: float,
                     mid: Optional[float], queue_at_fill: float,
                     print_size: float) -> _FillRecord:
        return _FillRecord(
            ts=ts, market_id=o.market_id, order_id=o.order_id,
            side=_side_enum(o.side), price=o.price, size=fill,
            mid_at_fill=mid, queue_ahead_at_fill=max(0.0, queue_at_fill),
            print_size=print_size,
            latency_applied_ms=self._cfg.placement_latency_ms,
            assumption_set=self._assumption_set,
        )

    def _prune(self, ms: MarketState) -> None:
        dead = []
        for oid, o in self._orders.items():
            if o.market_id != ms.market_id:
                continue
            if o.remaining <= _PRICE_TOL:
                dead.append(oid)
            elif o.cancel_effective_ts is not None and ms.ts >= o.cancel_effective_ts:
                dead.append(oid)
        for oid in dead:
            del self._orders[oid]

    # -- adverse-selection marks (6.3.5) --------------------------------

    def mark_fills(self, now: datetime, mid: float) -> List[PaperFill]:
        """Backfill mid_p1m/mid_p10m/mid_p1h for fills whose horizons have
        elapsed at `now` (None until then). Returns the current fill snapshots."""
        for r in self._fills:
            elapsed = (now - r.ts).total_seconds()
            for attr, horizon in _ADVERSE_HORIZONS:
                if elapsed >= horizon and getattr(r, attr) is None:
                    setattr(r, attr, float(mid))
        return [r.to_paperfill() for r in self._fills]

    # -- accessors -------------------------------------------------------

    def open_orders(self) -> List[dict]:
        """Snapshot of resting/pending orders (audit)."""
        out = []
        for o in self._orders.values():
            out.append({
                "order_id": o.order_id, "market_id": o.market_id,
                "side": o.side, "price": o.price, "size": o.size,
                "remaining": o.remaining, "queue_ahead": o.queue_ahead,
                "activated": o.activated,
                "placed_effective_ts": o.placed_effective_ts,
                "cancel_effective_ts": o.cancel_effective_ts,
            })
        return out

    def fills(self) -> List[PaperFill]:
        return [r.to_paperfill() for r in self._fills]

    def exposure_incidents(self) -> List[ExposureIncident]:
        out = list(self._incidents)
        for mkt, inc in self._open_incident.items():
            if inc is None:
                continue
            start = inc["start"]
            oids = inc["order_ids"]
            end = self._last_ms_ts.get(mkt)
            dur = (end - start).total_seconds() if end is not None else None
            out.append(ExposureIncident(
                start=start, end=None, duration_s=dur,
                n_live_orders=len(oids), order_ids=oids))
        return out

    def total_exposure_seconds(self) -> float:
        return sum(i.duration_s or 0.0 for i in self.exposure_incidents())

    @property
    def assumption_set(self) -> str:
        return self._assumption_set
