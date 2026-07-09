"""SQLite state store (plan Section 5, task T2).

Durable persistence and restart survival for the market-making module: per-
contract/per-ladder inventory, open orders, quote history, PnL snapshots,
Beuoy bankrolls, the risk-mode journal, the append-only fill log, and rolling
liquidity windows. One SQLite DB file (WAL mode), single file path passed to
the constructor (default `market_maker/mm_state.db`; tests use tmp paths).

Design notes (plan Section 5 + Section 8.2/8.9):
- `fills` is append-only: no update/delete method is exposed for it anywhere
  in this module.
- `fold_fills_to_inventory()` recomputes per-market signed inventory purely
  from the `fills` table (SETTLEMENT-tagged pseudo-fills included, no special
  casing) so callers can assert `fold(fills) == inventory` as a standing
  invariant (risk 8.2).
- `settlements` is keyed on `(market_id, expiry_key)`; the idempotency guard
  trips ONLY when an existing row already has a TERMINAL outcome (YES/NO).
  An existing UNSETTLEABLE row never blocks a later successful settlement and
  is overwritten by it (plan Section 2.13).
- `orders.status` lifecycle: PENDING -> LIVE -> {CANCELLED, FILLED, UNKNOWN}.
  `mark_all_live_orders_unknown()` is the restart-protocol step 2 helper
  (plan Section 5): on boot, every LIVE order becomes UNKNOWN pending venue
  reconciliation.
- Crash-consistency: `record_fill_and_update_inventory()` commits the fill
  insert and the inventory upsert in ONE transaction, so a fill is never
  observable without its resulting inventory state (plan Section 5
  "write-ahead of action" rule) -- any dependent quote/hedge action must be
  emitted only after this call returns.
- `mid_log` (mm_suitability_alignment_plan.md Change C, mid-log design): a
  durable per-tick, per-market mid history the harness appends to every tick
  (`append_mids`), independent of the `fills`/`quotes` tables. It backs the
  paper runner's markout report (`pnl_report.markout_report`), which joins a
  fill's `ts + horizon` against this table via `mid_at_or_after` rather than
  requiring any fill-mutation step -- restart-robust by construction since it
  is computed from durable state at report time.
- All datetimes are serialized as UTC ISO-8601 strings; dict/list-valued
  fields are serialized as JSON text columns.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from market_maker.contracts import (
    BankrollState,
    ContractInv,
    Fill,
    LadderInv,
    LiquidityRegime,
    LiquidityState,
    PaperFill,
    QuoteMode,
    QuoteSet,
    RiskDirective,
    RiskTrigger,
    SettlementEvent,
    SettlementOutcome,
    Side,
    SpotSource,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join("market_maker", "mm_state.db")

# Order lifecycle statuses (plan Section 5). Not in contracts.py (no
# `Order`/`OrderState` dataclass is defined there -- 4.12 only defines the
# VenueAdapter boundary); kept local to the store.
ORDER_STATUSES = ("PENDING", "LIVE", "CANCELLED", "FILLED", "UNKNOWN")


# ---------------------------------------------------------------------------
# Small local record types for tables with no Section-4 contract dataclass.
# ---------------------------------------------------------------------------


@dataclass
class OrderRecord:
    client_order_id: str
    market_id: str
    side: Side
    price: float
    size: float
    status: str
    venue_order_id: Optional[str]
    ts_placed: datetime
    ts_final: Optional[datetime]


@dataclass
class QuoteRecord:
    """QuoteSet plus the QuoteProposal fields carried alongside it in the
    `quotes` history table (plan Section 5: "QuoteSet + term decomposition +
    the QuoteProposal fields (r_x, delta_x, sigma_b)").
    """
    quote_set: QuoteSet
    r_x: float
    delta_x: float
    skew_x: float
    sigma_b: float
    params_id: str
    x_bid: float
    x_ask: float
    p_bid_raw: float
    p_ask_raw: float


@dataclass
class MidLogRow:
    ts: datetime
    market_id: str
    mid: float


@dataclass
class PnlSnapshot:
    ts: datetime
    market_id: Optional[str]
    expiry_key: Optional[str]
    realized: float
    unrealized_consensus: float
    unrealized_mid: float
    settlement_pnl: float
    bankroll_utilization: float


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_json(obj: Any) -> str:
    return json.dumps(obj)


def _from_json(s: Optional[str], default: Any = None) -> Any:
    if s is None:
        return default
    return json.loads(s)


class MMStateStore:
    """SQLite-backed durable state store for the market-making module."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MMStateStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inventory (
                    market_id TEXT PRIMARY KEY,
                    q REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    q_max REAL NOT NULL,
                    age_weighted_holding REAL NOT NULL,
                    updated_ts TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ladder_state (
                    expiry_key TEXT PRIMARY KEY,
                    net_band_exposure TEXT NOT NULL,
                    gross REAL NOT NULL,
                    phi REAL NOT NULL,
                    r3_histogram TEXT NOT NULL,
                    vertical_offsets TEXT NOT NULL,
                    updated_ts TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    client_order_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    status TEXT NOT NULL,
                    venue_order_id TEXT,
                    ts_placed TEXT NOT NULL,
                    ts_final TEXT
                )
                """
            )
            self._conn.execute(
                # mid_p1m/mid_p10m/mid_p1h: legacy adverse-selection backfill
                # columns for the now-deleted PaperFillSimulator.mark_fills()
                # channel (plan Wave 0 W0.4) -- kept as legacy-NULL, no schema
                # migration. Adverse-selection marking is superseded by the
                # mid_log markout report (pnl_report.markout_report).
                """
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    liquidity TEXT NOT NULL,
                    venue_ts TEXT NOT NULL,
                    queue_ahead_at_fill REAL NOT NULL,
                    print_size REAL NOT NULL,
                    latency_applied_ms INTEGER NOT NULL,
                    assumption_set TEXT NOT NULL,
                    mid_at_fill REAL,
                    mid_p1m REAL,
                    mid_p10m REAL,
                    mid_p1h REAL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fills_market_id ON fills(market_id)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    bid_price REAL NOT NULL,
                    ask_price REAL NOT NULL,
                    bid_size REAL NOT NULL,
                    ask_size REAL NOT NULL,
                    terms TEXT NOT NULL,
                    risk_mode TEXT NOT NULL,
                    noarb_checked INTEGER NOT NULL,
                    source_seq INTEGER NOT NULL,
                    r_x REAL NOT NULL,
                    delta_x REAL NOT NULL,
                    skew_x REAL NOT NULL,
                    sigma_b REAL NOT NULL,
                    params_id TEXT NOT NULL,
                    x_bid REAL NOT NULL,
                    x_ask REAL NOT NULL,
                    p_bid_raw REAL NOT NULL,
                    p_ask_raw REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_quotes_market_id ON quotes(market_id)"
            )
            # prune_quotes filters on ts alone; the market_id index above is
            # unusable there, so give the DELETE its own index (mirrors
            # idx_mid_log_ts).
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_quotes_ts ON quotes(ts)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pnl (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    market_id TEXT,
                    expiry_key TEXT,
                    realized REAL NOT NULL,
                    unrealized_consensus REAL NOT NULL,
                    unrealized_mid REAL NOT NULL,
                    settlement_pnl REAL NOT NULL,
                    bankroll_utilization REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settlements (
                    market_id TEXT NOT NULL,
                    expiry_key TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    settlement_ts TEXT NOT NULL,
                    strike REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    spot_used REAL,
                    spot_source TEXT NOT NULL,
                    q_settled REAL NOT NULL,
                    payoff REAL,
                    pnl_realized REAL,
                    excluded_from_gate INTEGER NOT NULL,
                    PRIMARY KEY (market_id, expiry_key)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bankrolls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expiry_key TEXT NOT NULL,
                    model_ids TEXT NOT NULL,
                    bankrolls TEXT NOT NULL,
                    last_update TEXT NOT NULL,
                    update_count INTEGER NOT NULL,
                    frozen INTEGER NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bankrolls_expiry ON bankrolls(expiry_key)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    eps_add REAL NOT NULL,
                    kelly_mult REAL NOT NULL,
                    triggers TEXT NOT NULL,
                    latched_until TEXT NOT NULL,
                    cancel_all INTEGER NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_risk_journal_market_id ON risk_journal(market_id)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS liquidity_windows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    realized_depth_bid REAL NOT NULL,
                    realized_depth_ask REAL NOT NULL,
                    kyle_lambda REAL,
                    arb_halflife_s REAL,
                    regime TEXT NOT NULL,
                    window TEXT NOT NULL,
                    vol_discount REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_liquidity_windows_market_id ON liquidity_windows(market_id)"
            )
            # prune_liquidity_windows filters on ts alone; the market_id
            # index above is unusable there, so give the DELETE its own
            # index (mirrors idx_quotes_ts / idx_mid_log_ts).
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_liquidity_windows_ts ON liquidity_windows(ts)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status, market_id)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS markets (
                    market_id TEXT PRIMARY KEY,
                    expiry_key TEXT NOT NULL,
                    strike REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mid_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    mid REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mid_log_market_ts ON mid_log(market_id, ts)"
            )
            # prune_mid_log filters on ts alone; the composite index above is
            # unusable there (market_id leads), so give the DELETE its own index.
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mid_log_ts ON mid_log(ts)"
            )

    # ------------------------------------------------------------------
    # inventory
    # ------------------------------------------------------------------

    def upsert_inventory(self, market_id: str, inv: ContractInv, updated_ts: Optional[datetime] = None) -> None:
        ts = updated_ts or datetime.now(timezone.utc)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO inventory (market_id, q, avg_cost, q_max, age_weighted_holding, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    q=excluded.q, avg_cost=excluded.avg_cost, q_max=excluded.q_max,
                    age_weighted_holding=excluded.age_weighted_holding, updated_ts=excluded.updated_ts
                """,
                (market_id, inv.q, inv.avg_cost, inv.q_max, inv.age_weighted_holding, _dt_to_iso(ts)),
            )

    def get_inventory(self, market_id: str) -> Optional[ContractInv]:
        row = self._conn.execute(
            "SELECT * FROM inventory WHERE market_id = ?", (market_id,)
        ).fetchone()
        if row is None:
            return None
        return ContractInv(
            q=row["q"], avg_cost=row["avg_cost"], q_max=row["q_max"],
            age_weighted_holding=row["age_weighted_holding"],
        )

    def get_all_inventory(self) -> Dict[str, ContractInv]:
        rows = self._conn.execute("SELECT * FROM inventory").fetchall()
        return {
            row["market_id"]: ContractInv(
                q=row["q"], avg_cost=row["avg_cost"], q_max=row["q_max"],
                age_weighted_holding=row["age_weighted_holding"],
            )
            for row in rows
        }

    # ------------------------------------------------------------------
    # ladder_state
    # ------------------------------------------------------------------

    def upsert_ladder_state(
        self,
        expiry_key: str,
        ladder: LadderInv,
        vertical_offsets: Optional[dict] = None,
        updated_ts: Optional[datetime] = None,
    ) -> None:
        ts = updated_ts or datetime.now(timezone.utc)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO ladder_state
                    (expiry_key, net_band_exposure, gross, phi, r3_histogram, vertical_offsets, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expiry_key) DO UPDATE SET
                    net_band_exposure=excluded.net_band_exposure, gross=excluded.gross,
                    phi=excluded.phi, r3_histogram=excluded.r3_histogram,
                    vertical_offsets=excluded.vertical_offsets, updated_ts=excluded.updated_ts
                """,
                (
                    expiry_key,
                    _to_json(ladder.net_band_exposure),
                    ladder.gross,
                    ladder.phi,
                    _to_json({str(k): v for k, v in ladder.r3_histogram.items()}),
                    _to_json(vertical_offsets or {}),
                    _dt_to_iso(ts),
                ),
            )

    def get_ladder_state(self, expiry_key: str) -> Optional[Tuple[LadderInv, dict]]:
        row = self._conn.execute(
            "SELECT * FROM ladder_state WHERE expiry_key = ?", (expiry_key,)
        ).fetchone()
        if row is None:
            return None
        r3_raw = _from_json(row["r3_histogram"], {})
        ladder = LadderInv(
            net_band_exposure=_from_json(row["net_band_exposure"], []),
            gross=row["gross"],
            phi=row["phi"],
            r3_histogram={int(k): v for k, v in r3_raw.items()},
        )
        vertical_offsets = _from_json(row["vertical_offsets"], {})
        return ladder, vertical_offsets

    # ------------------------------------------------------------------
    # orders
    # ------------------------------------------------------------------

    def upsert_order(
        self,
        client_order_id: str,
        market_id: str,
        side: Side,
        price: float,
        size: float,
        status: str,
        venue_order_id: Optional[str] = None,
        ts_placed: Optional[datetime] = None,
        ts_final: Optional[datetime] = None,
    ) -> None:
        if status not in ORDER_STATUSES:
            raise ValueError(f"invalid order status {status!r}; must be one of {ORDER_STATUSES}")
        placed = ts_placed or datetime.now(timezone.utc)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO orders
                    (client_order_id, market_id, side, price, size, status, venue_order_id, ts_placed, ts_final)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    market_id=excluded.market_id, side=excluded.side, price=excluded.price,
                    size=excluded.size, status=excluded.status, venue_order_id=excluded.venue_order_id,
                    ts_placed=excluded.ts_placed, ts_final=excluded.ts_final
                """,
                (
                    client_order_id, market_id, side.value, price, size, status,
                    venue_order_id, _dt_to_iso(placed), _dt_to_iso(ts_final),
                ),
            )

    def _order_from_row(self, row: sqlite3.Row) -> OrderRecord:
        return OrderRecord(
            client_order_id=row["client_order_id"],
            market_id=row["market_id"],
            side=Side(row["side"]),
            price=row["price"],
            size=row["size"],
            status=row["status"],
            venue_order_id=row["venue_order_id"],
            ts_placed=_iso_to_dt(row["ts_placed"]),
            ts_final=_iso_to_dt(row["ts_final"]),
        )

    def get_order(self, client_order_id: str) -> Optional[OrderRecord]:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return None if row is None else self._order_from_row(row)

    def get_all_orders(self) -> List[OrderRecord]:
        rows = self._conn.execute("SELECT * FROM orders").fetchall()
        return [self._order_from_row(r) for r in rows]

    def get_live_orders(
        self, market_id: Optional[str] = None, side: Optional[Side] = None
    ) -> List[OrderRecord]:
        """Orders currently PENDING/LIVE, optionally scoped to one market
        and/or side (plan B4-CPU: avoids the full-table deserialize
        `get_all_orders()` does -- `order_lifecycle.py`'s per-tick hot paths
        use this instead). `ORDER BY rowid` reproduces the row order
        `get_all_orders()`'s unordered `SELECT *` happens to yield (insertion
        order), so callers that used to scan-and-filter see byte-identical
        first-hit results.
        """
        query = "SELECT * FROM orders WHERE status IN ('PENDING', 'LIVE')"
        params: List[Any] = []
        if market_id is not None:
            query += " AND market_id = ?"
            params.append(market_id)
        if side is not None:
            query += " AND side = ?"
            params.append(side.value)
        query += " ORDER BY rowid"
        rows = self._conn.execute(query, params).fetchall()
        return [self._order_from_row(r) for r in rows]

    def mark_all_live_orders_unknown(self) -> int:
        """Restart protocol step 2 (plan Section 5): mark every LIVE order
        UNKNOWN pending venue reconciliation. Returns the number of rows
        updated.
        """
        with self._conn:
            cur = self._conn.execute(
                "UPDATE orders SET status = 'UNKNOWN' WHERE status = 'LIVE'"
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # fills (append-only; no update/delete API)
    # ------------------------------------------------------------------

    def append_fill(self, fill: Fill) -> int:
        pf = fill if isinstance(fill, PaperFill) else PaperFill(
            ts=fill.ts, market_id=fill.market_id, order_id=fill.order_id, side=fill.side,
            price=fill.price, size=fill.size, liquidity=fill.liquidity, venue_ts=fill.venue_ts,
        )
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO fills
                    (ts, market_id, order_id, side, price, size, liquidity, venue_ts,
                     queue_ahead_at_fill, print_size, latency_applied_ms, assumption_set,
                     mid_at_fill, mid_p1m, mid_p10m, mid_p1h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _dt_to_iso(pf.ts), pf.market_id, pf.order_id, pf.side.value, pf.price, pf.size,
                    pf.liquidity.value, _dt_to_iso(pf.venue_ts), pf.queue_ahead_at_fill, pf.print_size,
                    pf.latency_applied_ms, pf.assumption_set, pf.mid_at_fill, pf.mid_p1m,
                    pf.mid_p10m, pf.mid_p1h,
                ),
            )
            return int(cur.lastrowid)

    def _fill_from_row(self, row: sqlite3.Row) -> PaperFill:
        from market_maker.contracts import LiquiditySource
        return PaperFill(
            ts=_iso_to_dt(row["ts"]),
            market_id=row["market_id"],
            order_id=row["order_id"],
            side=Side(row["side"]),
            price=row["price"],
            size=row["size"],
            liquidity=LiquiditySource(row["liquidity"]),
            venue_ts=_iso_to_dt(row["venue_ts"]),
            queue_ahead_at_fill=row["queue_ahead_at_fill"],
            print_size=row["print_size"],
            latency_applied_ms=row["latency_applied_ms"],
            assumption_set=row["assumption_set"],
            mid_at_fill=row["mid_at_fill"],
            mid_p1m=row["mid_p1m"],
            mid_p10m=row["mid_p10m"],
            mid_p1h=row["mid_p1h"],
        )

    def get_fills(self, market_id: Optional[str] = None) -> List[PaperFill]:
        if market_id is None:
            rows = self._conn.execute("SELECT * FROM fills ORDER BY id ASC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM fills WHERE market_id = ? ORDER BY id ASC", (market_id,)
            ).fetchall()
        return [self._fill_from_row(r) for r in rows]

    def fold_fills_to_inventory(self) -> Dict[str, ContractInv]:
        """Recompute per-market signed inventory purely from the `fills`
        table (SETTLEMENT-tagged pseudo-fills included, no special casing),
        for the standing invariant `fold(fills) == inventory` (plan risk
        8.2). `q_max` and `age_weighted_holding` are not fill-derived (they
        come from S'(x) and holding-time tracking respectively) so they are
        returned as 0.0 placeholders here -- callers comparing against
        `get_all_inventory()` should compare `q` (and `avg_cost` if desired),
        not those two fields.

        Cost-basis price (C0): `price` is already YES-scale for every fill --
        MAKER/TAKER and SETTLEMENT alike, BUY_YES and BUY_NO alike (paper_
        fill_sim stores the YES-book price for both sides; the harness bridge
        un-complements order_lifecycle's sell-YES-via-buy-NO order-placement
        convention before any fill reaches this table) -- so it is used
        directly as `cost_basis_price`, with no per-side complement. This
        matches inventory_manager._apply_contract_fill (the untouched
        reference); prior to the C0 fix this used `1 - price` for BUY_NO,
        which disagreed with that reference and produced a phantom -0.20/
        share PnL on every open BUY_NO fill (see
        mm_suitability_alignment_plan.md pre-step C0).
        """
        rows = self._conn.execute("SELECT * FROM fills ORDER BY id ASC").fetchall()
        state: Dict[str, Dict[str, float]] = {}
        for row in rows:
            market_id = row["market_id"]
            side = Side(row["side"])
            price = row["price"]
            size = row["size"]
            sign = 1.0 if side is Side.BUY_YES else -1.0
            cost_basis_price = price
            delta_q = sign * size

            s = state.setdefault(market_id, {"q": 0.0, "avg_cost": 0.0})
            q_old = s["q"]
            avg_cost_old = s["avg_cost"]

            if q_old == 0.0 or (q_old * delta_q) >= 0.0:
                # Opening or adding to an existing position (same sign).
                new_q = q_old + delta_q
                if new_q != 0.0:
                    s["avg_cost"] = (
                        avg_cost_old * abs(q_old) + cost_basis_price * abs(delta_q)
                    ) / abs(new_q)
                s["q"] = new_q
            else:
                # Reducing or flipping an existing position (opposite sign).
                if abs(delta_q) <= abs(q_old):
                    s["q"] = q_old + delta_q
                    if s["q"] == 0.0:
                        s["avg_cost"] = 0.0
                    # else: remaining lot keeps its existing avg_cost.
                else:
                    remainder = delta_q + q_old
                    s["q"] = remainder
                    s["avg_cost"] = cost_basis_price

        return {
            market_id: ContractInv(
                q=v["q"], avg_cost=v["avg_cost"], q_max=0.0, age_weighted_holding=0.0,
            )
            for market_id, v in state.items()
        }

    # ------------------------------------------------------------------
    # Crash-consistent transactional write (fill + inventory, atomic)
    # ------------------------------------------------------------------

    def record_fill_and_update_inventory(self, fill: Fill, resulting_inventory: ContractInv) -> int:
        """Insert the fill and upsert the resulting per-market inventory in
        ONE transaction (plan Section 5 write-ahead rule): either both writes
        land or neither does. Callers MUST NOT emit any dependent quote/hedge
        action until this call returns.
        """
        pf = fill if isinstance(fill, PaperFill) else PaperFill(
            ts=fill.ts, market_id=fill.market_id, order_id=fill.order_id, side=fill.side,
            price=fill.price, size=fill.size, liquidity=fill.liquidity, venue_ts=fill.venue_ts,
        )
        updated_ts = fill.ts
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO fills
                    (ts, market_id, order_id, side, price, size, liquidity, venue_ts,
                     queue_ahead_at_fill, print_size, latency_applied_ms, assumption_set,
                     mid_at_fill, mid_p1m, mid_p10m, mid_p1h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _dt_to_iso(pf.ts), pf.market_id, pf.order_id, pf.side.value, pf.price, pf.size,
                    pf.liquidity.value, _dt_to_iso(pf.venue_ts), pf.queue_ahead_at_fill, pf.print_size,
                    pf.latency_applied_ms, pf.assumption_set, pf.mid_at_fill, pf.mid_p1m,
                    pf.mid_p10m, pf.mid_p1h,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO inventory (market_id, q, avg_cost, q_max, age_weighted_holding, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    q=excluded.q, avg_cost=excluded.avg_cost, q_max=excluded.q_max,
                    age_weighted_holding=excluded.age_weighted_holding, updated_ts=excluded.updated_ts
                """,
                (
                    fill.market_id, resulting_inventory.q, resulting_inventory.avg_cost,
                    resulting_inventory.q_max, resulting_inventory.age_weighted_holding,
                    _dt_to_iso(updated_ts),
                ),
            )
            return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # quotes (append-only history)
    # ------------------------------------------------------------------

    def append_quote(
        self,
        quote_set: QuoteSet,
        r_x: float,
        delta_x: float,
        skew_x: float,
        sigma_b: float,
        params_id: str,
        x_bid: float,
        x_ask: float,
        p_bid_raw: float,
        p_ask_raw: float,
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO quotes
                    (ts, market_id, bid_price, ask_price, bid_size, ask_size, terms, risk_mode,
                     noarb_checked, source_seq, r_x, delta_x, skew_x, sigma_b, params_id,
                     x_bid, x_ask, p_bid_raw, p_ask_raw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _dt_to_iso(quote_set.ts), quote_set.market_id, quote_set.bid_price,
                    quote_set.ask_price, quote_set.bid_size, quote_set.ask_size,
                    _to_json(quote_set.terms), quote_set.risk_mode.value,
                    int(quote_set.noarb_checked), quote_set.source_seq,
                    r_x, delta_x, skew_x, sigma_b, params_id, x_bid, x_ask, p_bid_raw, p_ask_raw,
                ),
            )
            return int(cur.lastrowid)

    def _quote_from_row(self, row: sqlite3.Row) -> QuoteRecord:
        qs = QuoteSet(
            ts=_iso_to_dt(row["ts"]),
            market_id=row["market_id"],
            bid_price=row["bid_price"],
            ask_price=row["ask_price"],
            bid_size=row["bid_size"],
            ask_size=row["ask_size"],
            terms=_from_json(row["terms"], {}),
            risk_mode=QuoteMode(row["risk_mode"]),
            noarb_checked=bool(row["noarb_checked"]),
            source_seq=row["source_seq"],
        )
        return QuoteRecord(
            quote_set=qs, r_x=row["r_x"], delta_x=row["delta_x"], skew_x=row["skew_x"],
            sigma_b=row["sigma_b"], params_id=row["params_id"], x_bid=row["x_bid"],
            x_ask=row["x_ask"], p_bid_raw=row["p_bid_raw"], p_ask_raw=row["p_ask_raw"],
        )

    def get_quotes(self, market_id: Optional[str] = None) -> List[QuoteRecord]:
        if market_id is None:
            rows = self._conn.execute("SELECT * FROM quotes ORDER BY id ASC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM quotes WHERE market_id = ? ORDER BY id ASC", (market_id,)
            ).fetchall()
        return [self._quote_from_row(r) for r in rows]

    def prune_quotes(self, older_than: datetime) -> int:
        """Delete `quotes` rows strictly older than `older_than` (plan Wave 0
        W0.2 -- the quotes table is otherwise unbounded on a persistent
        --state-db). Mirrors `prune_mid_log` exactly: same `_dt_to_iso`
        serialization, same `ts < ?` bound. Returns the number of rows
        deleted."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM quotes WHERE ts < ?", (_dt_to_iso(older_than),)
            )
        return cur.rowcount

    # ------------------------------------------------------------------
    # pnl (periodic snapshots)
    # ------------------------------------------------------------------

    def append_pnl_snapshot(self, snapshot: PnlSnapshot) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO pnl
                    (ts, market_id, expiry_key, realized, unrealized_consensus, unrealized_mid,
                     settlement_pnl, bankroll_utilization)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _dt_to_iso(snapshot.ts), snapshot.market_id, snapshot.expiry_key,
                    snapshot.realized, snapshot.unrealized_consensus, snapshot.unrealized_mid,
                    snapshot.settlement_pnl, snapshot.bankroll_utilization,
                ),
            )
            return int(cur.lastrowid)

    def get_pnl_snapshots(self, market_id: Optional[str] = None) -> List[PnlSnapshot]:
        if market_id is None:
            rows = self._conn.execute("SELECT * FROM pnl ORDER BY id ASC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pnl WHERE market_id = ? ORDER BY id ASC", (market_id,)
            ).fetchall()
        return [
            PnlSnapshot(
                ts=_iso_to_dt(row["ts"]), market_id=row["market_id"], expiry_key=row["expiry_key"],
                realized=row["realized"], unrealized_consensus=row["unrealized_consensus"],
                unrealized_mid=row["unrealized_mid"], settlement_pnl=row["settlement_pnl"],
                bankroll_utilization=row["bankroll_utilization"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # settlements (idempotent on terminal outcomes only)
    # ------------------------------------------------------------------

    def upsert_settlement(self, event: SettlementEvent) -> bool:
        """Insert/overwrite the settlement row for `(market_id, expiry_key)`.

        Returns False (no write performed) if an existing row already has a
        TERMINAL outcome (YES/NO) -- the idempotency guard (plan Section
        2.13/5). An existing UNSETTLEABLE row never blocks and is overwritten.
        Returns True when the write happens.
        """
        existing = self._conn.execute(
            "SELECT outcome FROM settlements WHERE market_id = ? AND expiry_key = ?",
            (event.market_id, event.expiry_key),
        ).fetchone()
        if existing is not None and existing["outcome"] in (
            SettlementOutcome.YES.value, SettlementOutcome.NO.value,
        ):
            logger.info(
                "settlements idempotency guard: (%s, %s) already terminal (%s); skipping",
                event.market_id, event.expiry_key, existing["outcome"],
            )
            return False

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO settlements
                    (market_id, expiry_key, ts, settlement_ts, strike, outcome, spot_used,
                     spot_source, q_settled, payoff, pnl_realized, excluded_from_gate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id, expiry_key) DO UPDATE SET
                    ts=excluded.ts, settlement_ts=excluded.settlement_ts, strike=excluded.strike,
                    outcome=excluded.outcome, spot_used=excluded.spot_used,
                    spot_source=excluded.spot_source, q_settled=excluded.q_settled,
                    payoff=excluded.payoff, pnl_realized=excluded.pnl_realized,
                    excluded_from_gate=excluded.excluded_from_gate
                """,
                (
                    event.market_id, event.expiry_key, _dt_to_iso(event.ts),
                    _dt_to_iso(event.settlement_ts), event.strike, event.outcome.value,
                    event.spot_used, event.spot_source.value, event.q_settled, event.payoff,
                    event.pnl_realized, int(event.excluded_from_gate),
                ),
            )
        return True

    def _settlement_from_row(self, row: sqlite3.Row) -> SettlementEvent:
        return SettlementEvent(
            ts=_iso_to_dt(row["ts"]),
            settlement_ts=_iso_to_dt(row["settlement_ts"]),
            market_id=row["market_id"],
            expiry_key=row["expiry_key"],
            strike=row["strike"],
            outcome=SettlementOutcome(row["outcome"]),
            spot_used=row["spot_used"],
            spot_source=SpotSource(row["spot_source"]),
            q_settled=row["q_settled"],
            payoff=row["payoff"],
            pnl_realized=row["pnl_realized"],
            excluded_from_gate=bool(row["excluded_from_gate"]),
        )

    def get_settlement(self, market_id: str, expiry_key: str) -> Optional[SettlementEvent]:
        row = self._conn.execute(
            "SELECT * FROM settlements WHERE market_id = ? AND expiry_key = ?",
            (market_id, expiry_key),
        ).fetchone()
        return None if row is None else self._settlement_from_row(row)

    def get_all_settlements(self) -> List[SettlementEvent]:
        rows = self._conn.execute("SELECT * FROM settlements").fetchall()
        return [self._settlement_from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # bankrolls (append-only versions per expiry)
    # ------------------------------------------------------------------

    def append_bankroll_state(self, expiry_key: str, state: BankrollState) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO bankrolls (expiry_key, model_ids, bankrolls, last_update, update_count, frozen)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    expiry_key, _to_json(state.model_ids), _to_json(state.bankrolls),
                    _dt_to_iso(state.last_update), state.update_count, int(state.frozen),
                ),
            )
            return int(cur.lastrowid)

    def _bankroll_from_row(self, row: sqlite3.Row) -> BankrollState:
        return BankrollState(
            model_ids=_from_json(row["model_ids"], []),
            bankrolls=_from_json(row["bankrolls"], {}),
            last_update=_iso_to_dt(row["last_update"]),
            update_count=row["update_count"],
            frozen=bool(row["frozen"]),
        )

    def get_latest_bankroll_state(self, expiry_key: str) -> Optional[BankrollState]:
        row = self._conn.execute(
            "SELECT * FROM bankrolls WHERE expiry_key = ? ORDER BY id DESC LIMIT 1",
            (expiry_key,),
        ).fetchone()
        return None if row is None else self._bankroll_from_row(row)

    def get_bankroll_history(self, expiry_key: str) -> List[BankrollState]:
        rows = self._conn.execute(
            "SELECT * FROM bankrolls WHERE expiry_key = ? ORDER BY id ASC", (expiry_key,)
        ).fetchall()
        return [self._bankroll_from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # risk_journal (append-only transitions)
    # ------------------------------------------------------------------

    def append_risk_directive(self, directive: RiskDirective) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO risk_journal
                    (ts, market_id, mode, eps_add, kelly_mult, triggers, latched_until, cancel_all)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _dt_to_iso(directive.ts), directive.market_id, directive.mode.value,
                    directive.eps_add, directive.kelly_mult,
                    _to_json([t.value for t in directive.triggers]),
                    _dt_to_iso(directive.latched_until), int(directive.cancel_all),
                ),
            )
            return int(cur.lastrowid)

    def _risk_directive_from_row(self, row: sqlite3.Row) -> RiskDirective:
        triggers_raw = _from_json(row["triggers"], [])
        return RiskDirective(
            ts=_iso_to_dt(row["ts"]),
            market_id=row["market_id"],
            mode=QuoteMode(row["mode"]),
            eps_add=row["eps_add"],
            kelly_mult=row["kelly_mult"],
            triggers=[RiskTrigger(t) for t in triggers_raw],
            latched_until=_iso_to_dt(row["latched_until"]),
            cancel_all=bool(row["cancel_all"]),
        )

    def get_risk_journal(self, market_id: Optional[str] = None) -> List[RiskDirective]:
        if market_id is None:
            rows = self._conn.execute("SELECT * FROM risk_journal ORDER BY id ASC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM risk_journal WHERE market_id = ? ORDER BY id ASC", (market_id,)
            ).fetchall()
        return [self._risk_directive_from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # liquidity_windows (rolling raw aggregates for liquidity_monitor)
    # ------------------------------------------------------------------

    def append_liquidity_window(self, state: LiquidityState) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO liquidity_windows
                    (ts, market_id, realized_depth_bid, realized_depth_ask, kyle_lambda,
                     arb_halflife_s, regime, window, vol_discount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _dt_to_iso(state.ts), state.market_id, state.realized_depth_bid,
                    state.realized_depth_ask, state.kyle_lambda, state.arb_halflife_s,
                    state.regime.value, state.window, state.vol_discount,
                ),
            )
            return int(cur.lastrowid)

    def _liquidity_window_from_row(self, row: sqlite3.Row) -> LiquidityState:
        return LiquidityState(
            ts=_iso_to_dt(row["ts"]),
            market_id=row["market_id"],
            realized_depth_bid=row["realized_depth_bid"],
            realized_depth_ask=row["realized_depth_ask"],
            kyle_lambda=row["kyle_lambda"],
            arb_halflife_s=row["arb_halflife_s"],
            regime=LiquidityRegime(row["regime"]),
            window=row["window"],
            vol_discount=row["vol_discount"],
        )

    def get_liquidity_windows(self, market_id: Optional[str] = None) -> List[LiquidityState]:
        if market_id is None:
            rows = self._conn.execute("SELECT * FROM liquidity_windows ORDER BY id ASC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM liquidity_windows WHERE market_id = ? ORDER BY id ASC", (market_id,)
            ).fetchall()
        return [self._liquidity_window_from_row(r) for r in rows]

    def prune_liquidity_windows(self, older_than: datetime) -> int:
        """Delete `liquidity_windows` rows strictly older than `older_than`
        (plan Wave 1 W1.1 -- the table is otherwise unbounded on a
        persistent --state-db). Mirrors `prune_quotes` exactly: same
        `_dt_to_iso` serialization, same `ts < ?` bound. Returns the number
        of rows deleted."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM liquidity_windows WHERE ts < ?", (_dt_to_iso(older_than),)
            )
        return cur.rowcount

    # ------------------------------------------------------------------
    # markets (persisted market_id -> (expiry_key, strike) registry;
    # plan B3-schema: enables settlement catch_up() to find a PREVIOUS
    # run's markets after a restart, since `inventory` persists q/avg_cost
    # per market_id but not its ladder membership).
    # ------------------------------------------------------------------

    def upsert_market(self, market_id: str, expiry_key: str, strike: float) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO markets (market_id, expiry_key, strike)
                VALUES (?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    expiry_key=excluded.expiry_key, strike=excluded.strike
                """,
                (market_id, expiry_key, strike),
            )

    def get_market_registry(self) -> Dict[str, Tuple[str, float]]:
        rows = self._conn.execute("SELECT market_id, expiry_key, strike FROM markets").fetchall()
        return {row["market_id"]: (row["expiry_key"], row["strike"]) for row in rows}

    # ------------------------------------------------------------------
    # mid_log (per-tick per-market mid history; mm_suitability_alignment_
    # plan.md Change C mid-log design -- backs pnl_report.markout_report)
    # ------------------------------------------------------------------

    def append_mids(self, ts: datetime, mids: Dict[str, float]) -> None:
        """Durably log this tick's per-market mids, one INSERT executemany
        (plan C1/C2). No-op on an empty `mids` dict (harness only calls this
        when at least one market had a mid this tick)."""
        if not mids:
            return
        ts_str = _dt_to_iso(ts)
        with self._conn:
            self._conn.executemany(
                "INSERT INTO mid_log (ts, market_id, mid) VALUES (?, ?, ?)",
                [(ts_str, market_id, mid) for market_id, mid in mids.items()],
            )

    def get_mids(self, market_id: Optional[str] = None) -> List[MidLogRow]:
        if market_id is None:
            rows = self._conn.execute("SELECT * FROM mid_log ORDER BY id ASC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM mid_log WHERE market_id = ? ORDER BY id ASC", (market_id,)
            ).fetchall()
        return [
            MidLogRow(ts=_iso_to_dt(row["ts"]), market_id=row["market_id"], mid=row["mid"])
            for row in rows
        ]

    def mid_at_or_after(self, market_id: str, ts: datetime, ts_max: datetime) -> Optional[float]:
        """First `mid_log` mid for `market_id` with `ts_log in [ts, ts_max)`
        (the markout report's horizon-window join, plan C1/C3). The lower
        bound is inclusive (`>=`), the upper bound is EXCLUSIVE (`<`, F1 fix)
        -- adjacent horizon windows are constructed to abut at `ts_max`
        (pnl_report.markout_report caps each horizon's window at the next
        horizon's start), so an exclusive upper bound is what makes a mid
        landing exactly on that boundary serve only the later horizon, never
        both. BOTH bounds are serialized via `_dt_to_iso` -- never raw
        `.isoformat()` -- since Python's `isoformat()` omits the microseconds
        field entirely when `microsecond == 0`, which would otherwise produce
        variable-width TEXT that a naive mixed serializer could compare
        inconsistently; routing both bounds (and every stored row) through
        the same `_dt_to_iso` call keeps the TEXT range compare below
        correct. Backed by `idx_mid_log_market_ts (market_id, ts)`. Returns
        None if no row falls in the window (NULL semantics -- that horizon is
        excluded by the caller)."""
        row = self._conn.execute(
            """
            SELECT mid FROM mid_log
            WHERE market_id = ? AND ts >= ? AND ts < ?
            ORDER BY ts LIMIT 1
            """,
            (market_id, _dt_to_iso(ts), _dt_to_iso(ts_max)),
        ).fetchone()
        return None if row is None else row["mid"]

    def prune_mid_log(self, older_than: datetime) -> int:
        """Delete `mid_log` rows strictly older than `older_than` (F3 -- the
        markout report's own lookback bounds how far back mids are ever
        needed, so anything older can be dropped to keep the table's growth
        bounded on a persistent --state-db). Bound via `_dt_to_iso`, same
        serialization as every other mid_log timestamp compare. Returns the
        number of rows deleted."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM mid_log WHERE ts < ?", (_dt_to_iso(older_than),)
            )
        return cur.rowcount
