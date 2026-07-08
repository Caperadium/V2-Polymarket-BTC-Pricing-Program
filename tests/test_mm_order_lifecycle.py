"""Order lifecycle manager tests (plan Section 2.11, task O1).

Uses a mock VenueAdapter recording every call, plus a tmp-path MMStateStore.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from market_maker.config import MMConfig
from market_maker.contracts import (
    QuoteMode,
    QuoteSet,
    RiskDirective,
    RiskTrigger,
    Side,
    VenueAdapter,
    VenueDescriptor,
)
from market_maker.order_lifecycle import (
    OrderLifecycleManager,
    PaperVenueAdapter,
    SimClock,
    client_order_id,
)
from market_maker.state_store import MMStateStore

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


class MockVenueAdapter(VenueAdapter):
    """Records every call; open_orders/positions are simple dict-backed
    fakes the test can pre-seed."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self._open_orders: List[Dict[str, Any]] = []
        self._positions: Dict[str, float] = {}

    def submit_order(self, client_order_id, market_id, side, price, size):
        self.calls.append(("submit", client_order_id, market_id, side, price, size))
        self._open_orders.append({"client_order_id": client_order_id, "market_id": market_id})
        return {"client_order_id": client_order_id, "venue_order_id": "v-" + client_order_id}

    def replace_order(self, client_order_id, price, size):
        self.calls.append(("replace", client_order_id, price, size))
        return {"client_order_id": client_order_id}

    def cancel_order(self, client_order_id):
        self.calls.append(("cancel", client_order_id))
        self._open_orders = [o for o in self._open_orders if o["client_order_id"] != client_order_id]
        return {"client_order_id": client_order_id}

    def fetch_open_orders(self):
        return list(self._open_orders)

    def fetch_positions(self):
        return dict(self._positions)

    def stream_market_data(self):
        return iter([])

    def descriptor(self):
        return VenueDescriptor(
            tick_size=0.01, min_size=1.0, price_band=(0.001, 0.999),
            maker_fee=0.0, maker_rebate=0.0, settlement_rule="12:00 ET", supports_ladder=True,
        )


@pytest.fixture
def store(tmp_path):
    s = MMStateStore(str(tmp_path / "mm_state.db"))
    yield s
    s.close()


@pytest.fixture
def venue():
    return MockVenueAdapter()


@pytest.fixture
def config():
    return MMConfig()


@pytest.fixture
def clock():
    return SimClock(NOW)


@pytest.fixture
def mgr(venue, store, config, clock):
    return OrderLifecycleManager(venue, store, config, clock)


def _quote(seq=1, bid=0.40, ask=0.45, bid_size=10.0, ask_size=10.0, risk_mode=QuoteMode.TWO_SIDED):
    return QuoteSet(
        ts=NOW, market_id="mkt-1", bid_price=bid, ask_price=ask, bid_size=bid_size, ask_size=ask_size,
        terms={}, risk_mode=risk_mode, noarb_checked=True, source_seq=seq,
    )


def _risk(mode=QuoteMode.TWO_SIDED, cancel_all=False):
    return RiskDirective(
        ts=NOW, market_id="mkt-1", mode=mode, eps_add=0.0, kelly_mult=1.0,
        triggers=[], latched_until=NOW, cancel_all=cancel_all,
    )


# ---------------------------------------------------------------------------
# churn
# ---------------------------------------------------------------------------


def test_no_requote_inside_tolerance(mgr, venue, store):
    mgr.apply("mkt-1", _quote(), _risk())
    submit_calls = [c for c in venue.calls if c[0] == "submit"]
    assert len(submit_calls) == 2  # bid + ask

    # Tiny nudge inside tolerance (requote_price_tol default 0.005) -> no venue calls.
    q2 = _quote(seq=2, bid=0.401, ask=0.451)
    mgr.apply("mkt-1", q2, _risk())
    assert len(venue.calls) == 2  # unchanged -- no new submit/cancel


def test_requote_outside_tolerance_replaces(mgr, venue, store):
    mgr.apply("mkt-1", _quote(), _risk())
    assert len([c for c in venue.calls if c[0] == "submit"]) == 2

    q2 = _quote(seq=2, bid=0.50, ask=0.55)  # well outside tolerance
    mgr.apply("mkt-1", q2, _risk())
    cancels = [c for c in venue.calls if c[0] == "cancel"]
    submits = [c for c in venue.calls if c[0] == "submit"]
    assert len(cancels) == 2  # old bid + old ask cancelled
    assert len(submits) == 4  # 2 original + 2 replacements


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


def test_same_quoteset_twice_no_duplicates(mgr, venue, store):
    q = _quote()
    mgr.apply("mkt-1", q, _risk())
    n_after_first = len(venue.calls)
    mgr.apply("mkt-1", q, _risk())  # exact replay
    assert len(venue.calls) == n_after_first  # no new submit/cancel calls

    orders = [o for o in store.get_all_orders() if o.market_id == "mkt-1"]
    assert len(orders) == 2  # still exactly one bid + one ask order


def test_deterministic_ids_stable(config):
    id1 = client_order_id("mkt-1", Side.BUY_YES, 0.40, 10.0, 1)
    id2 = client_order_id("mkt-1", Side.BUY_YES, 0.40, 10.0, 1)
    assert id1 == id2
    id3 = client_order_id("mkt-1", Side.BUY_YES, 0.41, 10.0, 1)
    assert id3 != id1


# ---------------------------------------------------------------------------
# PULLED / cancel_all
# ---------------------------------------------------------------------------


def test_cancel_all_on_risk_cancel_all(mgr, venue, store):
    mgr.apply("mkt-1", _quote(), _risk())
    assert len([c for c in venue.calls if c[0] == "submit"]) == 2

    mgr.apply("mkt-1", _quote(seq=2), _risk(cancel_all=True))
    cancels = [c for c in venue.calls if c[0] == "cancel"]
    assert len(cancels) == 2
    live = [o for o in store.get_all_orders() if o.status in ("PENDING", "LIVE")]
    assert live == []


def test_cancel_all_on_pulled_mode(mgr, venue, store):
    mgr.apply("mkt-1", _quote(), _risk())
    mgr.apply("mkt-1", _quote(seq=2, risk_mode=QuoteMode.PULLED), _risk())
    cancels = [c for c in venue.calls if c[0] == "cancel"]
    assert len(cancels) == 2


def test_cancel_all_direct_call(mgr, venue, store):
    mgr.apply("mkt-1", _quote(), _risk())
    mgr.cancel_all()
    cancels = [c for c in venue.calls if c[0] == "cancel"]
    assert len(cancels) == 2


# ---------------------------------------------------------------------------
# zero-size sides
# ---------------------------------------------------------------------------


def test_zero_size_side_produces_no_order(mgr, venue, store):
    q = _quote(bid_size=10.0, ask_size=0.0)
    mgr.apply("mkt-1", q, _risk())
    submits = [c for c in venue.calls if c[0] == "submit"]
    assert len(submits) == 1
    assert submits[0][3] == Side.BUY_YES

    orders = [o for o in store.get_all_orders() if o.market_id == "mkt-1"]
    assert len(orders) == 1


# ---------------------------------------------------------------------------
# restart reconciliation
# ---------------------------------------------------------------------------


def test_restart_reconciliation(mgr, venue, store):
    mgr.apply("mkt-1", _quote(), _risk())
    live_orders = [o for o in store.get_all_orders() if o.status == "LIVE"]
    assert len(live_orders) == 2
    recognized_coid = live_orders[0].client_order_id

    # Venue only recognizes one of our two orders, plus an orphan it thinks is open.
    venue._open_orders = [
        {"client_order_id": recognized_coid, "market_id": "mkt-1"},
        {"client_order_id": "orphan-coid", "market_id": "mkt-1"},
    ]
    # Position mismatch: store has flat inventory (no fills yet), venue reports a position.
    venue._positions = {"mkt-1": 5.0}

    result = mgr.restart_reconcile()

    assert recognized_coid in result.restored
    other_coid = [o.client_order_id for o in live_orders if o.client_order_id != recognized_coid][0]
    assert other_coid in result.cancelled_unknown
    assert "orphan-coid" in result.orphans_cancelled
    assert "mkt-1" in result.position_discrepancies
    assert result.position_discrepancies["mkt-1"] == (0.0, 5.0)
    assert result.manual_trigger is True

    restored_rec = store.get_order(recognized_coid)
    assert restored_rec.status == "LIVE"
    cancelled_rec = store.get_order(other_coid)
    assert cancelled_rec.status == "CANCELLED"

    cancel_calls = [c[1] for c in venue.calls if c[0] == "cancel"]
    assert other_coid in cancel_calls
    assert "orphan-coid" in cancel_calls


def test_restart_reconciliation_no_discrepancy_when_matched(mgr, venue, store):
    mgr.apply("mkt-1", _quote(), _risk())
    live_orders = [o for o in store.get_all_orders() if o.status == "LIVE"]
    venue._open_orders = [{"client_order_id": o.client_order_id, "market_id": "mkt-1"} for o in live_orders]
    venue._positions = {"mkt-1": 0.0}

    result = mgr.restart_reconcile()
    assert set(result.restored) == {o.client_order_id for o in live_orders}
    assert result.cancelled_unknown == []
    assert result.orphans_cancelled == []
    assert result.position_discrepancies == {}
    assert result.manual_trigger is False


# ---------------------------------------------------------------------------
# order states persisted through the store
# ---------------------------------------------------------------------------


def test_order_states_persisted(mgr, venue, store):
    mgr.apply("mkt-1", _quote(), _risk())
    orders = sorted(store.get_all_orders(), key=lambda o: o.side.value)
    assert all(o.status == "LIVE" for o in orders)
    assert all(o.venue_order_id is not None for o in orders)

    mgr.cancel_all()
    orders = store.get_all_orders()
    assert all(o.status == "CANCELLED" for o in orders)
    assert all(o.ts_final == NOW for o in orders)


# ---------------------------------------------------------------------------
# PaperVenueAdapter (thin wrapper)
# ---------------------------------------------------------------------------


class _StubFillSim:
    def __init__(self) -> None:
        self.placed = []
        self.cancelled = []

    def place_order(self, coid, market_id, side, price, size):
        self.placed.append((coid, market_id, side, price, size))

    def cancel_order(self, coid):
        self.cancelled.append(coid)


def test_paper_venue_adapter_passthrough(store):
    sim = _StubFillSim()
    descriptor = VenueDescriptor(
        tick_size=0.01, min_size=1.0, price_band=(0.001, 0.999),
        maker_fee=0.0, maker_rebate=0.0, settlement_rule="12:00 ET", supports_ladder=True,
    )
    adapter = PaperVenueAdapter(sim, store, descriptor)
    clock = SimClock(NOW)
    mgr = OrderLifecycleManager(adapter, store, MMConfig(), clock)

    mgr.apply("mkt-1", _quote(), _risk())
    assert len(sim.placed) == 2

    live = [o for o in store.get_all_orders() if o.status == "LIVE"]
    assert len(live) == 2

    mgr.cancel_all()
    assert len(sim.cancelled) == 2
