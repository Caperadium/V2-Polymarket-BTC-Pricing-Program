"""Market-data client (plan Section 2.14, task D1, contract 4.2).

Sole producer of MarketState. BookMirror is a feed-agnostic book mirror driven
by an injectable stream of dict messages: {"type":"snapshot",
"bids":[(price,size)...], "asks":[...], "ts", "seq"}, {"type":"delta",
"side":"bid"/"ask", "price", "size", "ts", "seq"} (size 0 removes the level),
{"type":"trade", "price", "size", "ts", "seq"}.

PolymarketFeedAdapter is the live wiring (P0b resolved 2026-07-07: the CLOB
WebSocket market channel exposes FULL_L2 depth + trade prints): it subscribes
one connection per ladder and translates venue payloads into the BookMirror
schema above. See its docstring for schemas and the ping/pong-based health
rule.

capability records what the live feed provides (FULL_L2 per P0b; both modes
must work here) -- a constructor argument, not one of the numbered interface
contracts.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from market_maker.config import MMConfig
from market_maker.contracts import MarketState


class FeedCapability(Enum):
    FULL_L2 = "FULL_L2"
    TOP_OF_BOOK = "TOP_OF_BOOK"


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


class BookMirror:
    """Local order-book mirror for one instrument, fed via on_message().

    Identity (market_id/expiry_key/strike) is supplied at emit() time, not
    held by the mirror -- the mirror is purely a book/health state machine.
    """

    def __init__(
        self,
        capability: FeedCapability = FeedCapability.FULL_L2,
        depth_n: int = 10,
        config: Optional[MMConfig] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.capability = capability
        self.depth_n = depth_n
        self.config = config if config is not None else MMConfig()
        self._clock = clock if clock is not None else _default_clock

        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self._last_seq: Optional[int] = None
        self._last_msg_ts: Optional[datetime] = None
        self._gap_ok: bool = True  # False (unhealthy) once a seq gap is seen, until next snapshot
        self._gap_events: List[datetime] = []
        self._last_prints: List[Tuple[datetime, float, float]] = []

    # -- message ingestion ----------------------------------------------------

    def on_message(self, msg: Dict[str, Any]) -> None:
        mtype = msg["type"]
        seq = msg.get("seq")
        ts = msg["ts"]
        if mtype == "snapshot":
            self._apply_seq(seq, ts, is_snapshot=True)
            self._bids = {float(p): float(s) for p, s in msg.get("bids", []) if s != 0}
            self._asks = {float(p): float(s) for p, s in msg.get("asks", []) if s != 0}
        elif mtype == "delta":
            self._apply_seq(seq, ts, is_snapshot=False)
            book = self._bids if msg["side"] == "bid" else self._asks
            price = float(msg["price"])
            size = float(msg["size"])
            if size == 0.0:
                book.pop(price, None)
            else:
                book[price] = size
        elif mtype == "trade":
            self._apply_seq(seq, ts, is_snapshot=False)
            self._last_prints.append((ts, float(msg["price"]), float(msg["size"])))
        else:
            raise ValueError("unknown message type: " + repr(mtype))
        self._last_msg_ts = ts

    def _apply_seq(self, seq: Optional[int], ts: datetime, is_snapshot: bool) -> None:
        if is_snapshot:
            self._gap_ok = True  # snapshot always resyncs and heals
        elif seq is not None and self._last_seq is not None and seq != self._last_seq + 1:
            self._gap_ok = False
            self._gap_events.append(ts)
        if seq is not None:
            self._last_seq = seq

    # -- book views -------------------------------------------------------------

    def _depth_n(self) -> int:
        return 1 if self.capability is FeedCapability.TOP_OF_BOOK else self.depth_n

    def bid_depth(self) -> List[Tuple[float, float]]:
        n = self._depth_n()
        return [(p, self._bids[p]) for p in sorted(self._bids, reverse=True)[:n]]

    def ask_depth(self) -> List[Tuple[float, float]]:
        n = self._depth_n()
        return [(p, self._asks[p]) for p in sorted(self._asks)[:n]]

    def best_bid(self) -> Optional[float]:
        return max(self._bids) if self._bids else None

    def best_ask(self) -> Optional[float]:
        return min(self._asks) if self._asks else None

    def is_stale(self) -> bool:
        if self._last_msg_ts is None:
            return False
        now = self._clock()
        return (now - self._last_msg_ts).total_seconds() > self.config.feed_gap_threshold_s

    def feed_healthy(self) -> bool:
        return self._gap_ok and not self.is_stale()

    @property
    def gap_events(self) -> List[datetime]:
        return list(self._gap_events)

    # -- emission -----------------------------------------------------------------

    def emit(self, market_id: str, expiry_key: str, strike: float) -> MarketState:
        """Build MarketState (contract 4.2) and drain accumulated last_prints."""
        ts = self._last_msg_ts if self._last_msg_ts is not None else self._clock()
        prints = self._last_prints
        self._last_prints = []
        return MarketState(
            ts=ts,
            market_id=market_id,
            expiry_key=expiry_key,
            strike=strike,
            best_bid=self.best_bid(),
            best_ask=self.best_ask(),
            bid_depth=self.bid_depth(),
            ask_depth=self.ask_depth(),
            last_prints=prints,
            feed_healthy=self.feed_healthy(),
        )


CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def _venue_ts(ms_str: Any, fallback: datetime) -> datetime:
    """Polymarket 'timestamp' is a millisecond-epoch string."""
    try:
        return datetime.fromtimestamp(int(ms_str) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError):
        return fallback


class PolymarketFeedAdapter:
    """Live Polymarket CLOB WebSocket feed -> BookMirror message dicts
    (plan 2.14; boundary resolved by P0b 2026-07-07: FULL_L2 + trade prints).

    One WebSocket connection covers a whole ladder: `token_by_market` maps
    market_id (slug) -> YES clobTokenId; all tokens are subscribed on the
    "market" channel in a single subscription. A background thread runs the
    asyncio receive loop; the trading loop calls `drain()` once per tick to
    collect translated messages and `healthy()` for feed health.

    Venue payload translation (observed schemas in
    Market Maker/verification/p0b_feed_boundary_note.md and
    temp/feed_probe_*.jsonl):
      - `book` (full snapshot, sent on subscribe and on venue resync)
          -> {"type": "snapshot", "bids": [(p, s)...], "asks": [...], "ts"}
      - `price_change` (level delta; carries entries for BOTH tokens of the
        condition -- entries are filtered to subscribed tokens; side BUY=bid,
        SELL=ask; size is the new absolute level size, 0 removes)
          -> {"type": "delta", "side", "price", "size", "ts"}
      - `last_trade_price` (aggressor print with price+size)
          -> {"type": "trade", "price", "size", "ts"}
      - `tick_size_change` / anything else -> ignored.
    The venue provides no sequence numbers, so messages carry no "seq";
    BookMirror skips gap detection on seq=None and the harness assigns its
    own contiguous seq. Book integrity is restored by the venue's full `book`
    snapshot on every (re)connect.

    Feed health (P0b consequence 3): quiet books go silent 80s+, so health
    keys off CONNECTION liveness, not message arrival. The websockets library
    pings every `ping_interval_s`; a missed pong closes the socket, recv()
    raises, and `healthy()` flips False until the reconnect (exponential
    backoff, venue re-sends `book` on subscribe) succeeds. Pass
    `healthy()` as the tick's feed_healthy override so BookMirror's
    message-staleness heuristic never false-alarms a quiet book.
    """

    def __init__(
        self,
        token_by_market: Dict[str, str],
        capability: FeedCapability = FeedCapability.FULL_L2,
        ws_url: str = CLOB_WS_URL,
        ping_interval_s: float = 10.0,
        ping_timeout_s: float = 10.0,
        max_backoff_s: float = 30.0,
    ) -> None:
        if not token_by_market:
            raise ValueError("token_by_market must not be empty")
        self.token_by_market = dict(token_by_market)
        self.market_by_token: Dict[str, str] = {
            tok: m for m, tok in self.token_by_market.items()
        }
        if len(self.market_by_token) != len(self.token_by_market):
            raise ValueError("duplicate token ids in token_by_market")
        self.capability = capability
        self.ws_url = ws_url
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self.max_backoff_s = max_backoff_s

        self._lock = threading.Lock()
        self._buffers: Dict[str, List[Dict[str, Any]]] = {
            m: [] for m in self.token_by_market
        }
        self._connected = False
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._logger = logging.getLogger("mm.feed")

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("adapter already started")
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="polymarket-feed", daemon=True
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 10.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
        self._connected = False

    def healthy(self) -> bool:
        """Connection liveness (ping/pong-backed), NOT message recency."""
        return self._connected

    def drain(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return and clear all messages accumulated since the last drain,
        keyed by market_id (every market present, possibly [])."""
        with self._lock:
            out = self._buffers
            self._buffers = {m: [] for m in self.token_by_market}
        return out

    # -- receive loop (background thread) --------------------------------------

    def _thread_main(self) -> None:
        import asyncio

        try:
            asyncio.run(self._run())
        except Exception:
            self._logger.error("feed thread died", exc_info=True)
        finally:
            self._connected = False

    async def _run(self) -> None:
        import asyncio

        import websockets

        sub = json.dumps(
            {"type": "market", "assets_ids": list(self.market_by_token)}
        )
        backoff = 1.0
        while not self._stop_evt.is_set():
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=self.ping_interval_s,
                    ping_timeout=self.ping_timeout_s,
                ) as ws:
                    await ws.send(sub)
                    self._connected = True
                    backoff = 1.0
                    self._logger.info(
                        "subscribed %d tokens on %s", len(self.market_by_token), self.ws_url
                    )
                    while not self._stop_evt.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue  # quiet book; liveness is ping/pong's job
                        self._handle_raw(raw if isinstance(raw, str) else raw.decode())
            except Exception:
                if self._stop_evt.is_set():
                    break
                self._connected = False
                self._logger.warning(
                    "feed connection lost; reconnecting in %.0fs", backoff, exc_info=True
                )
                self._stop_evt.wait(backoff)
                backoff = min(backoff * 2.0, self.max_backoff_s)
            finally:
                self._connected = False

    # -- translation (pure; unit-testable without a socket) ---------------------

    def _handle_raw(self, raw: str, now: Optional[datetime] = None) -> None:
        now = now if now is not None else _default_clock()
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            self._logger.warning("unparseable feed message dropped")
            return
        events = payload if isinstance(payload, list) else [payload]
        for ev in events:
            if not isinstance(ev, dict):
                continue
            for market_id, msg in self._translate_event(ev, now):
                with self._lock:
                    self._buffers[market_id].append(msg)

    def _translate_event(
        self, ev: Dict[str, Any], now: datetime
    ) -> List[Tuple[str, Dict[str, Any]]]:
        et = ev.get("event_type", ev.get("type"))
        ts = _venue_ts(ev.get("timestamp"), now)
        out: List[Tuple[str, Dict[str, Any]]] = []
        if et == "book" or (et is None and ("bids" in ev or "asks" in ev)):
            market_id = self.market_by_token.get(str(ev.get("asset_id")))
            if market_id is not None:
                out.append((market_id, {
                    "type": "snapshot",
                    "bids": [(float(l["price"]), float(l["size"])) for l in ev.get("bids") or []],
                    "asks": [(float(l["price"]), float(l["size"])) for l in ev.get("asks") or []],
                    "ts": ts,
                }))
        elif et == "price_change":
            # list form observed live; flat form kept as documented fallback
            changes = ev.get("price_changes")
            if changes is None and "price" in ev:
                changes = [ev]
            for ch in changes or []:
                market_id = self.market_by_token.get(str(ch.get("asset_id")))
                if market_id is None:
                    continue
                out.append((market_id, {
                    "type": "delta",
                    "side": "bid" if ch.get("side") == "BUY" else "ask",
                    "price": float(ch["price"]),
                    "size": float(ch["size"]),
                    "ts": ts,
                }))
        elif et == "last_trade_price":
            market_id = self.market_by_token.get(str(ev.get("asset_id")))
            if market_id is not None:
                out.append((market_id, {
                    "type": "trade",
                    "price": float(ev["price"]),
                    "size": float(ev["size"]),
                    "ts": ts,
                }))
        # tick_size_change and unknown types are ignored safely (P0b note)
        return out
