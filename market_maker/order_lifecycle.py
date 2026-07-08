"""Order lifecycle manager (plan Section 2.11, task O1, contracts 4.11/4.12).

Converts (QuoteSet, RiskDirective) per market into venue actions through a
VenueAdapter (contract 4.12), with minimal churn, deterministic idempotent
client order IDs, and restart reconciliation (plan Section 5 steps 2-3).

Side convention (inferred, not spelled out verbatim by contracts.py): the
Side enum has only BUY_YES / BUY_NO -- no SELL_* value -- and the rest of the
module (inventory_manager.apply_fill, state_store.fold_fills_to_inventory)
already treats BUY_YES as +1 share and BUY_NO as -1 share, with cost basis
tracked uniformly in YES-price terms (price for BUY_YES, 1-price for
BUY_NO). Selling YES is economically identical to buying the complementary
NO token, so: the bid (buy YES) is quoted as Side.BUY_YES @ bid_price; the
ask (offer to sell YES) is quoted as Side.BUY_NO @ (1 - ask_price), same
size. This keeps every order/fill on the single BUY_YES/BUY_NO axis the rest
of the module already uses -- no new enum value needed.

Re-quote policy (plan 2.11): re-quote only when |price change| >
MMConfig.requote_price_tol or |relative size change| > requote_size_tol;
otherwise the resting order is left untouched (minimal churn). Re-quoting is
implemented as cancel-then-place rather than amend: the deterministic client
order ID is a hash of (market_id, side, price, size, source_seq), so a
changed price/size necessarily produces a new ID -- "amend in place" would
require keeping the OLD id under new terms, breaking the
same-params-always-same-id idempotency invariant relied on elsewhere.
VenueAdapter.replace_order() is still implemented by adapters for venues
that prefer amend semantics, but this manager does not call it.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from market_maker.config import MMConfig
from market_maker.contracts import (
    QuoteMode,
    QuoteSet,
    RiskDirective,
    Side,
    VenueAdapter,
    VenueDescriptor,
)
from market_maker.state_store import MMStateStore, OrderRecord

logger = logging.getLogger(__name__)

_LIVE_STATUSES = ("PENDING", "LIVE")


# ---------------------------------------------------------------------------
# SimClock -- injectable clock, no wall-clock reads inside the manager
# ---------------------------------------------------------------------------


class SimClock:
    """Injectable clock. Every timestamp the manager needs (ts_placed,
    ts_final on cancel) comes from here, never datetime.now(), so tests and
    backtests/paper-runs can drive time explicitly.
    """

    def __init__(self, now: Optional[datetime] = None) -> None:
        self._now = now if now is not None else datetime.now(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def set(self, now: datetime) -> None:
        self._now = now


# ---------------------------------------------------------------------------
# Deterministic client order IDs
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    # Fixed precision so float repr jitter never changes the hash for
    # numerically-identical prices/sizes.
    return f"{float(v):.6f}"


def client_order_id(market_id: str, side: Side, price: float, size: float, source_seq: int) -> str:
    """Deterministic client order id: sha256 of a stable pipe-joined payload,
    truncated -- mirrors the pattern in polymarket/intent_builder.py
    (compute_intent_id: hashlib.sha256(payload).hexdigest()[:N]). Same
    (market_id, side, price, size, source_seq) -> same id, so idempotent
    replays (same QuoteSet applied twice) never create duplicate orders.
    """
    payload = f"{market_id}|{side.value}|{_fmt(price)}|{_fmt(size)}|{source_seq}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Restart reconciliation result
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationResult:
    restored: List[str] = field(default_factory=list)
    cancelled_unknown: List[str] = field(default_factory=list)
    orphans_cancelled: List[str] = field(default_factory=list)
    position_discrepancies: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    manual_trigger: bool = False


def _get(obj: Any, key: str) -> Any:
    """Read `key` off either a dict or an attribute-bearing object -- venue
    payload shapes are adapter-specific (contract 4.12 leaves fetch_open_orders
    / fetch_positions return types as `Any`)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# ---------------------------------------------------------------------------
# Order lifecycle manager
# ---------------------------------------------------------------------------


class OrderLifecycleManager:
    """Converts (QuoteSet, RiskDirective) per market into venue actions.

    Desired state per side: an order at QuoteSet's price/size, or none if
    that side's size is zero, or if the effective mode (intersection of
    QuoteSet.risk_mode and RiskDirective.mode) suppresses that side.
    RiskDirective.cancel_all or either mode being PULLED cancels everything
    for that market.
    """

    def __init__(self, venue: VenueAdapter, store: MMStateStore, config: MMConfig, clock: SimClock):
        self.venue = venue
        self.store = store
        self.config = config
        self.clock = clock

    # -- public entry point ------------------------------------------------

    def apply(self, market_id: str, quote_set: QuoteSet, risk: RiskDirective) -> None:
        """Reconcile one market's resting orders against the desired
        (QuoteSet, RiskDirective) state."""
        if risk.cancel_all or risk.mode == QuoteMode.PULLED or quote_set.risk_mode == QuoteMode.PULLED:
            self.cancel_all(market_id)
            return

        bid_allowed = (
            quote_set.risk_mode in (QuoteMode.TWO_SIDED, QuoteMode.BID_ONLY)
            and risk.mode in (QuoteMode.TWO_SIDED, QuoteMode.BID_ONLY)
        )
        ask_allowed = (
            quote_set.risk_mode in (QuoteMode.TWO_SIDED, QuoteMode.ASK_ONLY)
            and risk.mode in (QuoteMode.TWO_SIDED, QuoteMode.ASK_ONLY)
        )

        desired_bid: Optional[Tuple[float, float]] = None
        if bid_allowed and quote_set.bid_size > 0.0:
            desired_bid = (quote_set.bid_price, quote_set.bid_size)

        desired_ask: Optional[Tuple[float, float]] = None
        if ask_allowed and quote_set.ask_size > 0.0:
            # Sell-YES-via-buy-NO convention (see module docstring).
            desired_ask = (1.0 - quote_set.ask_price, quote_set.ask_size)

        self._reconcile_side(market_id, Side.BUY_YES, desired_bid, quote_set.source_seq)
        self._reconcile_side(market_id, Side.BUY_NO, desired_ask, quote_set.source_seq)

    # -- churn / placement ---------------------------------------------------

    def _live_order_for(self, market_id: str, side: Side) -> Optional[OrderRecord]:
        recs = self.store.get_live_orders(market_id, side)
        return recs[0] if recs else None

    def _reconcile_side(
        self, market_id: str, side: Side, desired: Optional[Tuple[float, float]], source_seq: int
    ) -> None:
        current = self._live_order_for(market_id, side)

        if desired is None:
            if current is not None:
                self._cancel_order(current)
            return

        price, size = desired
        if current is not None:
            price_ok = abs(price - current.price) <= self.config.requote_price_tol
            size_ref = max(abs(current.size), 1e-9)
            size_ok = abs(size - current.size) / size_ref <= self.config.requote_size_tol
            if price_ok and size_ok:
                return  # inside tolerance -- minimal churn, leave resting order alone
            self._cancel_order(current)

        self._place_order(market_id, side, price, size, source_seq)

    def _place_order(self, market_id: str, side: Side, price: float, size: float, source_seq: int) -> None:
        coid = client_order_id(market_id, side, price, size, source_seq)
        existing = self.store.get_order(coid)
        if existing is not None and existing.status in _LIVE_STATUSES:
            return  # idempotent replay: identical desired order already resting

        now = self.clock.now()
        self.store.upsert_order(coid, market_id, side, price, size, "PENDING", ts_placed=now)
        ack = self.venue.submit_order(coid, market_id, side, price, size)
        venue_order_id = _get(ack, "venue_order_id") if ack is not None else None
        self.store.upsert_order(
            coid, market_id, side, price, size, "LIVE", venue_order_id=venue_order_id, ts_placed=now
        )

    def _cancel_order(self, rec: OrderRecord) -> None:
        self.venue.cancel_order(rec.client_order_id)
        self.store.upsert_order(
            rec.client_order_id, rec.market_id, rec.side, rec.price, rec.size,
            "CANCELLED", venue_order_id=rec.venue_order_id, ts_placed=rec.ts_placed,
            ts_final=self.clock.now(),
        )

    def cancel_all(self, market_id: Optional[str] = None) -> None:
        """Cancel every PENDING/LIVE order (optionally scoped to one
        market). Called on PULLED directives and on process shutdown."""
        for rec in self.store.get_live_orders(market_id):
            self._cancel_order(rec)

    # -- restart reconciliation (plan Section 5, steps 2-3) -----------------

    def restart_reconcile(self) -> ReconciliationResult:
        """Step 2: mark every LIVE order UNKNOWN. Step 3: fetch venue open
        orders + positions; orders we recognize are restored to LIVE, orders
        we don't are cancelled; venue orders with an unrecognized client
        order id (orphans) are cancelled; position deltas (fold(fills) vs
        venue truth) are returned as discrepancies for the caller to raise a
        MANUAL trigger on (plan risk 8.2).
        """
        self.store.mark_all_live_orders_unknown()

        open_orders = self.venue.fetch_open_orders() or []
        venue_coids = {_get(o, "client_order_id") for o in open_orders if _get(o, "client_order_id")}

        restored: List[str] = []
        cancelled_unknown: List[str] = []
        for rec in self.store.get_all_orders():
            if rec.status != "UNKNOWN":
                continue
            if rec.client_order_id in venue_coids:
                self.store.upsert_order(
                    rec.client_order_id, rec.market_id, rec.side, rec.price, rec.size,
                    "LIVE", venue_order_id=rec.venue_order_id, ts_placed=rec.ts_placed,
                )
                restored.append(rec.client_order_id)
            else:
                self.venue.cancel_order(rec.client_order_id)
                self.store.upsert_order(
                    rec.client_order_id, rec.market_id, rec.side, rec.price, rec.size,
                    "CANCELLED", venue_order_id=rec.venue_order_id, ts_placed=rec.ts_placed,
                    ts_final=self.clock.now(),
                )
                cancelled_unknown.append(rec.client_order_id)

        known_coids = {r.client_order_id for r in self.store.get_all_orders()}
        orphans_cancelled: List[str] = []
        for o in open_orders:
            coid = _get(o, "client_order_id")
            if coid and coid not in known_coids:
                self.venue.cancel_order(coid)
                orphans_cancelled.append(coid)

        venue_positions = self.venue.fetch_positions() or {}
        store_inv = self.store.fold_fills_to_inventory()
        market_ids = set(store_inv.keys()) | set(venue_positions.keys())
        discrepancies: Dict[str, Tuple[float, float]] = {}
        for mid in market_ids:
            store_q = store_inv[mid].q if mid in store_inv else 0.0
            raw = venue_positions.get(mid, 0.0)
            venue_q = _get(raw, "q") if not isinstance(raw, (int, float)) else raw
            venue_q = 0.0 if venue_q is None else float(venue_q)
            if abs(store_q - venue_q) > 1e-9:
                discrepancies[mid] = (store_q, venue_q)

        return ReconciliationResult(
            restored=restored,
            cancelled_unknown=cancelled_unknown,
            orphans_cancelled=orphans_cancelled,
            position_discrepancies=discrepancies,
            manual_trigger=bool(discrepancies),
        )


# ---------------------------------------------------------------------------
# Paper mode adapter (plan 2.11: "in paper mode, routes actions to the fill
# simulator instead of the venue")
# ---------------------------------------------------------------------------


class PaperVenueAdapter(VenueAdapter):
    """Thin VenueAdapter wrapping a paper fill-sim-like object. Place/cancel
    pass straight through to the sim (if it exposes them) and ack
    immediately -> the caller (OrderLifecycleManager) marks LIVE right after,
    since paper mode has no separate venue-ack round trip. Reconciliation
    reads open orders from the sim (falling back to the state store's own
    LIVE/PENDING rows if the sim doesn't track that) and positions from the
    state store, since in paper mode the simulator/state store IS the venue
    truth (plan Section 5 step 3 note).
    """

    def __init__(self, fill_sim: Any, store: MMStateStore, descriptor: VenueDescriptor):
        self._sim = fill_sim
        self._store = store
        self._descriptor = descriptor

    def submit_order(self, client_order_id: str, market_id: str, side: Side, price: float, size: float) -> Any:
        if hasattr(self._sim, "place_order"):
            self._sim.place_order(client_order_id, market_id, side, price, size)
        return {"client_order_id": client_order_id, "venue_order_id": client_order_id}

    def replace_order(self, client_order_id: str, price: float, size: float) -> Any:
        if hasattr(self._sim, "replace_order"):
            self._sim.replace_order(client_order_id, price, size)
        return {"client_order_id": client_order_id, "venue_order_id": client_order_id}

    def cancel_order(self, client_order_id: str) -> Any:
        if hasattr(self._sim, "cancel_order"):
            self._sim.cancel_order(client_order_id)
        return {"client_order_id": client_order_id}

    def fetch_open_orders(self) -> Any:
        if hasattr(self._sim, "open_orders"):
            return self._sim.open_orders()
        return [
            {"client_order_id": r.client_order_id, "market_id": r.market_id}
            for r in self._store.get_all_orders()
            if r.status in _LIVE_STATUSES
        ]

    def fetch_positions(self) -> Any:
        return {mid: inv.q for mid, inv in self._store.get_all_inventory().items()}

    def stream_market_data(self) -> Any:
        if hasattr(self._sim, "stream_market_data"):
            return self._sim.stream_market_data()
        raise NotImplementedError("PaperVenueAdapter.stream_market_data requires a fill-sim exposing stream_market_data()")

    def descriptor(self) -> VenueDescriptor:
        return self._descriptor
