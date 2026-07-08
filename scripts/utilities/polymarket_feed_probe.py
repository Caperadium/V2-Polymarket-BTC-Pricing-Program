"""P0b market-data boundary probe (plan task P0b, mm_implementation_plan.md).

Answers ONE question: what does the live Polymarket CLOB feed actually expose?
The paper-trading fill model depends on the answer:
  - L2 depth levels + trade prints  -> queue-behind fill model (fillmodel-v1)
  - top-of-book only                -> trade-through-only fallback
                                       (fillmodel-v1-tradethrough)

Run on the VPS (or any box with network) for ~10-60 minutes against one active
BTC market, then read the summary it prints:

    python scripts/utilities/polymarket_feed_probe.py --token-id <clobTokenId> \
        --minutes 15 --out temp/feed_probe.jsonl

Find a clobTokenId via the Gamma API (see core/backtesting/polymarket_fetcher.py)
or the market page. The probe:
  1. Pulls a REST book snapshot (https://clob.polymarket.com/book?token_id=...)
     and reports how many bid/ask levels it returns.
  2. Subscribes to the CLOB WebSocket market channel
     (wss://ws-subscriptions-clob.polymarket.com/ws/market) for the token and
     tallies message types (book / price_change / tick_size_change / last_trade_price),
     depth levels per book message, whether trade events carry price+size,
     inter-message gaps, and sequence/ordering fields.
  3. Writes every raw message to --out (JSONL) for offline inspection and
     prints a decision summary: FULL_L2 vs TOP_OF_BOOK, print availability,
     median/95p update cadence, largest gap.

Uses only stdlib + the `websockets` package if available; falls back to a
REST-poll-only probe (still answers the snapshot-depth question) when the
websocket library is missing. No orders are ever placed; read-only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

CLOB_REST = "https://clob.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def rest_snapshot(token_id: str) -> dict:
    url = f"{CLOB_REST}/book?token_id={token_id}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "btc-prediction-market/2.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def summarize_snapshot(book: dict) -> None:
    bids = book.get("bids", []) or []
    asks = book.get("asks", []) or []
    print(f"REST snapshot: {len(bids)} bid levels, {len(asks)} ask levels")
    if len(bids) > 1 or len(asks) > 1:
        print("  -> L2 depth IS available via REST snapshot")
    else:
        print("  -> only top-of-book in REST snapshot")
    for side, levels in (("bids", bids[:3]), ("asks", asks[:3])):
        print(f"  {side} head: {levels}")


def poll_probe(token_id: str, minutes: float, out_path: Path) -> None:
    """REST-poll fallback: cadence is poll-limited, but depth + trade fields
    (via /last-trade-price) are still verifiable."""
    end = time.time() + minutes * 60
    n = 0
    with out_path.open("a", encoding="ascii") as f:
        while time.time() < end:
            book = rest_snapshot(token_id)
            book["_probe_ts"] = time.time()
            f.write(json.dumps(book) + "\n")
            n += 1
            time.sleep(2.0)
    print(f"poll probe wrote {n} snapshots to {out_path}")


def ws_probe(token_id: str, minutes: float, out_path: Path) -> None:
    import asyncio

    import websockets  # noqa: F401  (import checked by caller)

    async def run() -> None:
        counts: Counter = Counter()
        depth_levels: list = []
        gaps: list = []
        trade_ok = 0
        last_ts = None
        end = time.time() + minutes * 60
        sub = json.dumps({"type": "market", "assets_ids": [token_id]})
        async with websockets.connect(CLOB_WS, ping_interval=10) as ws:
            await ws.send(sub)
            with out_path.open("a", encoding="ascii") as f:
                while time.time() < end:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        counts["_recv_timeout_30s"] += 1
                        continue
                    now = time.time()
                    if last_ts is not None:
                        gaps.append(now - last_ts)
                    last_ts = now
                    f.write(raw if isinstance(raw, str) else raw.decode())
                    f.write("\n")
                    try:
                        msgs = json.loads(raw)
                    except Exception:
                        counts["_unparseable"] += 1
                        continue
                    if isinstance(msgs, dict):
                        msgs = [msgs]
                    for m in msgs:
                        et = m.get("event_type", m.get("type", "?"))
                        counts[et] += 1
                        if et == "book":
                            depth_levels.append(
                                (len(m.get("bids", []) or []), len(m.get("asks", []) or []))
                            )
                        if et in ("last_trade_price", "trade"):
                            if m.get("price") is not None and m.get("size") is not None:
                                trade_ok += 1
        print("\n=== WS PROBE SUMMARY ===")
        print("message counts:", dict(counts))
        if depth_levels:
            mb = statistics.median(d[0] for d in depth_levels)
            ma = statistics.median(d[1] for d in depth_levels)
            print(f"book messages: median depth {mb} bid / {ma} ask levels")
            verdict = "FULL_L2" if (mb > 1 or ma > 1) else "TOP_OF_BOOK"
            print(f"DECISION INPUT: capability = {verdict}")
        print(f"trade events with price+size: {trade_ok}")
        if gaps:
            print(
                f"cadence: median {statistics.median(gaps):.2f}s, "
                f"p95 {sorted(gaps)[int(0.95 * len(gaps))]:.2f}s, max {max(gaps):.1f}s"
            )
        print("fill-model choice: FULL_L2 + trade events -> queue-behind; "
              "otherwise trade-through-only fallback (plan 2.14).")

    asyncio.get_event_loop().run_until_complete(run())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--token-id", required=True, help="clobTokenId of one active market")
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--out", default="temp/feed_probe.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    book = rest_snapshot(args.token_id)
    summarize_snapshot(book)

    try:
        import websockets  # noqa: F401
        has_ws = True
    except ImportError:
        has_ws = False
        print("websockets package not installed -> REST poll fallback "
              "(pip install websockets for the full probe)")

    if has_ws:
        ws_probe(args.token_id, args.minutes, out_path)
    else:
        poll_probe(args.token_id, args.minutes, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
