"""Settlement handler tests (plan Section 2.13, task E1).

Uses fixture BTCDataProvider frames (never touches DATA/ CSVs) and a
tmp-path MMStateStore.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_maker.config import MMConfig
from market_maker.contracts import ContractInv, Fill, LiquiditySource, Side, SettlementOutcome, SpotSource
from market_maker.settlement_handler import (
    BTCDataProvider,
    MarketPosition,
    SettlementHandler,
    settlement_instant_utc,
)
from market_maker.state_store import MMStateStore

EXPIRY = "2026-07-06"  # summer date -> EDT (UTC-4) -> 12:00 ET == 16:00 UTC
SETTLE_DT = settlement_instant_utc(EXPIRY)


def _intraday_df(rows):
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame({"close": [r[1] for r in rows]}, index=idx)


def _daily_df(rows):
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame({"close": [r[1] for r in rows]}, index=idx)


@pytest.fixture
def store(tmp_path):
    s = MMStateStore(str(tmp_path / "mm_state.db"))
    yield s
    s.close()


def _open_via_fill(store, market_id, side, price, size, avg_cost, ts):
    """Seed a position through the SAME fills channel settlement uses, so
    the fold(fills) == inventory invariant (risk 8.2) is meaningful to check
    after settlement (fold_fills_to_inventory only ever looks at the `fills`
    table, never at a directly-upserted `inventory` row)."""
    q = size if side is Side.BUY_YES else -size
    fill = Fill(
        ts=ts, market_id=market_id, order_id=f"open:{market_id}", side=side,
        price=price, size=size, liquidity=LiquiditySource.MAKER, venue_ts=ts,
    )
    store.record_fill_and_update_inventory(
        fill, ContractInv(q=q, avg_cost=avg_cost, q_max=1000.0, age_weighted_holding=0.0)
    )


def test_settlement_instant_utc_summer_edt():
    # noon ET under EDT (UTC-4) == 16:00 UTC.
    dt = settlement_instant_utc("2026-07-06")
    assert dt == datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# YES / NO outcomes with pseudo-fills
# ---------------------------------------------------------------------------


def test_yes_outcome_settles_long_position(store):
    intraday = _intraday_df([
        ("2026-07-06 15:58:00", 100400.0),
        ("2026-07-06 16:00:00", 100500.0),
        ("2026-07-06 16:02:00", 100600.0),
    ])
    data = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    handler = SettlementHandler(store, MMConfig(), data)

    # Long YES: q=100 @ avg_cost 0.40, opened through the fills channel so the
    # fold(fills) == inventory invariant is checkable after settlement.
    _open_via_fill(store, "mkt-yes", Side.BUY_YES, 0.40, 100.0, 0.40, SETTLE_DT - timedelta(days=1))
    # Strike 100000 < spot 100500 -> YES.
    m = MarketPosition(market_id="mkt-yes", strike=100000.0, q=100.0, avg_cost=0.40)
    result = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=5))

    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.outcome is SettlementOutcome.YES
    assert ev.spot_used == 100500.0
    assert ev.spot_source is SpotSource.INTRADAY
    assert ev.q_settled == 100.0
    assert ev.payoff == pytest.approx(100.0)  # q * 1.0
    assert ev.pnl_realized == pytest.approx(100.0 * (1.0 - 0.40))  # 60.0
    assert ev.excluded_from_gate is False

    inv = store.fold_fills_to_inventory().get("mkt-yes")
    assert inv is not None
    assert inv.q == pytest.approx(0.0)

    fills = store.get_fills("mkt-yes")
    assert len(fills) == 2  # opening fill + settlement pseudo-fill
    closing = fills[-1]
    assert closing.price == 1.0
    assert closing.size == 100.0


def test_exact_tie_resolves_no(store):
    # Venue rule: YES only if spot STRICTLY > strike. Exact tie -> NO.
    intraday = _intraday_df([
        ("2026-07-06 16:00:00", 100000.0),
    ])
    data = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    handler = SettlementHandler(store, MMConfig(), data)
    _open_via_fill(store, "mkt-tie", Side.BUY_YES, 0.50, 10.0, 0.50, SETTLE_DT - timedelta(days=1))
    m = MarketPosition(market_id="mkt-tie", strike=100000.0, q=10.0, avg_cost=0.50)
    result = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=5))
    assert result.events[0].outcome is SettlementOutcome.NO


def test_no_outcome_settles_short_position(store):
    intraday = _intraday_df([
        ("2026-07-06 15:58:00", 99400.0),
        ("2026-07-06 16:00:00", 99500.0),
        ("2026-07-06 16:02:00", 99600.0),
    ])
    data = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    handler = SettlementHandler(store, MMConfig(), data)

    # Net short YES (holding NO): q=-50, avg_cost 0.70 (raw YES-scale price of
    # the opening BUY_NO fill, C0 -- no complement), opened through the fills
    # channel.
    _open_via_fill(store, "mkt-no", Side.BUY_NO, 0.70, 50.0, 0.70, SETTLE_DT - timedelta(days=1))
    # Strike 100000 > spot 99500 -> NO.
    m = MarketPosition(market_id="mkt-no", strike=100000.0, q=-50.0, avg_cost=0.70)
    result = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=5))

    ev = result.events[0]
    assert ev.outcome is SettlementOutcome.NO
    assert ev.q_settled == -50.0
    assert ev.payoff == pytest.approx(50.0 * (1.0 - 0.0))  # 50 NO shares pay 1 each
    assert ev.pnl_realized == pytest.approx(-50.0 * (0.0 - 0.70))  # 35.0

    inv = store.fold_fills_to_inventory().get("mkt-no")
    assert inv.q == pytest.approx(0.0)


def test_flat_position_settles_without_fill(store):
    intraday = _intraday_df([("2026-07-06 16:00:00", 100500.0)])
    data = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    handler = SettlementHandler(store, MMConfig(), data)

    m = MarketPosition(market_id="mkt-flat", strike=100000.0, q=0.0, avg_cost=0.0)
    result = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT)

    assert result.events[0].q_settled == 0.0
    assert store.get_fills("mkt-flat") == []


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


def test_second_settle_call_is_noop(store):
    intraday = _intraday_df([("2026-07-06 16:00:00", 100500.0)])
    data = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    handler = SettlementHandler(store, MMConfig(), data)

    m = MarketPosition(market_id="mkt-idem", strike=100000.0, q=10.0, avg_cost=0.5)
    r1 = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT)
    assert len(r1.events) == 1
    assert len(store.get_fills("mkt-idem")) == 1

    r2 = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=1))
    assert r2.events == []  # no-op: already terminal
    assert len(store.get_fills("mkt-idem")) == 1  # no duplicate fill


# ---------------------------------------------------------------------------
# UNSETTLEABLE + retry + escalation
# ---------------------------------------------------------------------------


def test_unsettleable_no_fill_position_stays_open(store):
    # Intraday data exists but does not cover the settlement instant at all.
    intraday = _intraday_df([("2026-01-01 00:00:00", 50000.0)])
    data = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    handler = SettlementHandler(store, MMConfig(), data)

    m = MarketPosition(market_id="mkt-unset", strike=100000.0, q=20.0, avg_cost=0.5)
    result = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=1))

    ev = result.events[0]
    assert ev.outcome is SettlementOutcome.UNSETTLEABLE
    assert ev.excluded_from_gate is True
    assert ev.payoff is None
    assert ev.pnl_realized is None
    assert store.get_fills("mkt-unset") == []
    assert result.escalated_market_ids == []


def test_retry_settles_when_provider_replaced(store):
    intraday_empty = _intraday_df([("2026-01-01 00:00:00", 50000.0)])
    data = BTCDataProvider(intraday=intraday_empty, daily=pd.DataFrame())
    handler = SettlementHandler(store, MMConfig(), data)

    m = MarketPosition(market_id="mkt-retry", strike=100000.0, q=20.0, avg_cost=0.5)
    r1 = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=1))
    assert r1.events[0].outcome is SettlementOutcome.UNSETTLEABLE

    # Data now present -- mutate the provider's frame in place (same handler
    # instance, so the pre-existing UNSETTLEABLE row is retried/overwritten).
    handler.data = BTCDataProvider(
        intraday=_intraday_df([("2026-07-06 16:00:00", 100500.0)]), daily=pd.DataFrame()
    )
    r2 = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(hours=2))
    assert r2.events[0].outcome is SettlementOutcome.YES
    assert len(store.get_fills("mkt-retry")) == 1

    got = store.get_settlement("mkt-retry", EXPIRY)
    assert got.outcome is SettlementOutcome.YES


# ---------------------------------------------------------------------------
# path-backed cache invalidation (refresh() -- plan Task 1)
# ---------------------------------------------------------------------------


def _write_intraday_csv(path, rows):
    """Write a timestamp,close CSV in the format `_load_close_csv` reads."""
    lines = ["timestamp,close"] + [f"{ts},{close}" for ts, close in rows]
    path.write_text("\n".join(lines) + "\n")


def test_path_backed_retry_settles_when_csv_rewritten(store, tmp_path):
    # 1. CSV whose range does NOT cover the settlement instant.
    intraday_path = tmp_path / "intraday.csv"
    daily_path = tmp_path / "daily.csv"
    _write_intraday_csv(intraday_path, [("2026-01-01 00:00:00", 50000.0)])
    daily_path.write_text("timestamp,close\n")  # empty (header-only)

    data = BTCDataProvider(intraday_path=intraday_path, daily_path=daily_path)
    handler = SettlementHandler(store, MMConfig(), data)

    m = MarketPosition(market_id="mkt-path-retry", strike=100000.0, q=20.0, avg_cost=0.5)

    # 2. First settle attempt -> UNSETTLEABLE (range gate fails).
    r1 = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=1))
    assert r1.events[0].outcome is SettlementOutcome.UNSETTLEABLE

    # 3. Rewrite the CSV with a print covering the instant; force a later
    # mtime (coarse filesystem mtime resolution would otherwise make this
    # flaky in a fast test).
    stat0 = intraday_path.stat()
    _write_intraday_csv(intraday_path, [("2026-07-06 16:00:00", 100500.0)])
    os.utime(intraday_path, (stat0.st_mtime + 10, stat0.st_mtime + 10))

    # 4. Same handler instance (the regression this test covers: settle_expiry
    # picks up the rewritten file without swapping handler.data).
    r2 = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(hours=2))
    assert r2.events[0].outcome is SettlementOutcome.YES
    assert r2.events[0].spot_used == 100500.0


def test_torn_read_guard_keeps_serving_good_cache(store, tmp_path):
    intraday_path = tmp_path / "intraday.csv"
    daily_path = tmp_path / "daily.csv"
    _write_intraday_csv(intraday_path, [("2026-07-06 16:00:00", 100500.0)])
    daily_path.write_text("timestamp,close\n")

    data = BTCDataProvider(intraday_path=intraday_path, daily_path=daily_path)
    handler = SettlementHandler(store, MMConfig(), data)
    m = MarketPosition(market_id="mkt-torn", strike=100000.0, q=1.0, avg_cost=0.5)

    # Good CSV loaded via a settle (refresh() runs at the top of
    # _resolve_settlement_spot).
    r1 = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=1))
    assert r1.events[0].outcome is SettlementOutcome.YES

    good_mtime = intraday_path.stat().st_mtime

    # Overwrite with garbage/empty content, bump mtime -> reload must be
    # rejected (torn-read guard) so the provider still serves the previous
    # good frame.
    intraday_path.write_text("")
    os.utime(intraday_path, (good_mtime + 10, good_mtime + 10))
    data.refresh()

    m2 = MarketPosition(market_id="mkt-torn-2", strike=100000.0, q=1.0, avg_cost=0.5)
    r2 = handler.settle_expiry(EXPIRY, [m2], now=SETTLE_DT + timedelta(minutes=2))
    assert r2.events[0].outcome is SettlementOutcome.YES  # still the good cached frame
    assert r2.events[0].spot_used == 100500.0

    # A subsequent good rewrite + later mtime IS picked up -- proves the
    # rejected reload above did not advance the stored mtime.
    _write_intraday_csv(intraday_path, [("2026-07-06 16:00:00", 99000.0)])
    os.utime(intraday_path, (good_mtime + 20, good_mtime + 20))

    m3 = MarketPosition(market_id="mkt-torn-3", strike=100000.0, q=1.0, avg_cost=0.5)
    r3 = handler.settle_expiry(EXPIRY, [m3], now=SETTLE_DT + timedelta(minutes=3))
    assert r3.events[0].outcome is SettlementOutcome.NO  # 99000 < strike 100000
    assert r3.events[0].spot_used == 99000.0


def test_empty_then_populated_file_is_picked_up(store, tmp_path):
    # File missing at first use.
    intraday_path = tmp_path / "intraday.csv"
    daily_path = tmp_path / "daily.csv"
    daily_path.write_text("timestamp,close\n")

    data = BTCDataProvider(intraday_path=intraday_path, daily_path=daily_path)
    handler = SettlementHandler(store, MMConfig(), data)
    m = MarketPosition(market_id="mkt-empty-first", strike=100000.0, q=1.0, avg_cost=0.5)

    r1 = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=1))
    assert r1.events[0].outcome is SettlementOutcome.UNSETTLEABLE

    # Now populate the file + bump mtime -> next refresh loads it.
    _write_intraday_csv(intraday_path, [("2026-07-06 16:00:00", 100500.0)])
    os.utime(intraday_path, (intraday_path.stat().st_mtime + 10, intraday_path.stat().st_mtime + 10))

    r2 = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=2))
    assert r2.events[0].outcome is SettlementOutcome.YES
    assert r2.events[0].spot_used == 100500.0


def test_injected_frames_never_stat_or_reload(store, tmp_path):
    # Nonexistent paths -- if refresh() ever stat'd/reloaded these, it would
    # either raise or silently drop the injected frames to empty.
    nonexistent_intraday = tmp_path / "does_not_exist_intraday.csv"
    nonexistent_daily = tmp_path / "does_not_exist_daily.csv"
    assert not nonexistent_intraday.exists()
    assert not nonexistent_daily.exists()

    intraday = _intraday_df([("2026-07-06 16:00:00", 100500.0)])
    data = BTCDataProvider(
        intraday=intraday, daily=pd.DataFrame(),
        intraday_path=nonexistent_intraday, daily_path=nonexistent_daily,
    )
    handler = SettlementHandler(store, MMConfig(), data)
    m = MarketPosition(market_id="mkt-injected-static", strike=100000.0, q=1.0, avg_cost=0.5)

    # Settlement behavior is driven purely by the injected frame -- YES,
    # not UNSETTLEABLE -- proving refresh() didn't try (and fail) to load
    # from the nonexistent paths.
    result = handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT + timedelta(minutes=1))
    assert result.events[0].outcome is SettlementOutcome.YES
    assert result.events[0].spot_used == 100500.0


def test_unsettleable_escalates_after_retry_window(store):
    intraday_empty = _intraday_df([("2026-01-01 00:00:00", 50000.0)])
    data = BTCDataProvider(intraday=intraday_empty, daily=pd.DataFrame())
    config = MMConfig(settlement_retry_window_hours=24.0)
    handler = SettlementHandler(store, config, data)

    m = MarketPosition(market_id="mkt-escalate", strike=100000.0, q=5.0, avg_cost=0.5)
    t0 = SETTLE_DT + timedelta(minutes=1)
    r1 = handler.settle_expiry(EXPIRY, [m], now=t0)
    assert r1.escalated_market_ids == []

    t1 = t0 + timedelta(hours=25)
    r2 = handler.settle_expiry(EXPIRY, [m], now=t1)
    assert r2.events[0].outcome is SettlementOutcome.UNSETTLEABLE
    assert "mkt-escalate" in r2.escalated_market_ids


# ---------------------------------------------------------------------------
# catch_up (restart protocol step 4)
# ---------------------------------------------------------------------------


def test_catch_up_settles_missed_expiry(store):
    intraday = _intraday_df([("2026-07-06 16:00:00", 100500.0)])
    data = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    handler = SettlementHandler(store, MMConfig(), data)

    store.upsert_inventory(
        "mkt-catchup",
        ContractInv(q=15.0, avg_cost=0.45, q_max=100.0, age_weighted_holding=10.0),
        updated_ts=SETTLE_DT - timedelta(hours=1),
    )
    registry = {"mkt-catchup": (EXPIRY, 100000.0)}

    result = handler.catch_up(now=SETTLE_DT + timedelta(hours=1), registry=registry)
    assert len(result.events) == 1
    assert result.events[0].outcome is SettlementOutcome.YES
    assert store.get_inventory("mkt-catchup").q == pytest.approx(0.0)


def test_catch_up_skips_not_yet_due(store):
    handler = SettlementHandler(store, MMConfig(), BTCDataProvider(intraday=pd.DataFrame(), daily=pd.DataFrame()))
    store.upsert_inventory(
        "mkt-notdue", ContractInv(q=1.0, avg_cost=0.5, q_max=10.0, age_weighted_holding=0.0),
    )
    registry = {"mkt-notdue": (EXPIRY, 100000.0)}
    result = handler.catch_up(now=SETTLE_DT - timedelta(hours=1), registry=registry)
    assert result.events == []


def test_catch_up_skips_already_terminal(store):
    intraday = _intraday_df([("2026-07-06 16:00:00", 100500.0)])
    data = BTCDataProvider(intraday=intraday, daily=pd.DataFrame())
    handler = SettlementHandler(store, MMConfig(), data)

    m = MarketPosition(market_id="mkt-done", strike=100000.0, q=1.0, avg_cost=0.5)
    handler.settle_expiry(EXPIRY, [m], now=SETTLE_DT)
    store.upsert_inventory(
        "mkt-done", ContractInv(q=0.0, avg_cost=0.0, q_max=10.0, age_weighted_holding=0.0),
    )

    registry = {"mkt-done": (EXPIRY, 100000.0)}
    result = handler.catch_up(now=SETTLE_DT + timedelta(hours=1), registry=registry)
    assert result.events == []  # already terminal -- nothing pending
