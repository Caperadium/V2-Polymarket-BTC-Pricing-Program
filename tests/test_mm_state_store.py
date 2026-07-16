"""State store tests (plan Section 5, task T2).

Round-trips every table, the kill/restart protocol, settlement idempotency,
and the fills-fold-to-inventory invariant (plan risk 8.2).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from market_maker.contracts import (
    BankrollState,
    ContractInv,
    Fill,
    LadderInv,
    LiquidityRegime,
    LiquiditySource,
    LiquidityState,
    QuoteMode,
    QuoteSet,
    RiskDirective,
    RiskTrigger,
    SettlementEvent,
    SettlementOutcome,
    Side,
    SpotSource,
)
from market_maker.state_store import MMStateStore, PnlSnapshot

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "mm_state.db")
    s = MMStateStore(db_path)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_inventory_round_trip(store):
    inv = ContractInv(q=12.5, avg_cost=0.42, q_max=100.0, age_weighted_holding=3.5)
    store.upsert_inventory("mkt-1", inv, updated_ts=NOW)
    got = store.get_inventory("mkt-1")
    assert got == inv

    all_inv = store.get_all_inventory()
    assert all_inv == {"mkt-1": inv}


def test_inventory_upsert_overwrites(store):
    inv1 = ContractInv(q=1.0, avg_cost=0.5, q_max=10.0, age_weighted_holding=0.0)
    inv2 = ContractInv(q=2.0, avg_cost=0.6, q_max=20.0, age_weighted_holding=1.0)
    store.upsert_inventory("mkt-1", inv1, updated_ts=NOW)
    store.upsert_inventory("mkt-1", inv2, updated_ts=NOW)
    assert store.get_inventory("mkt-1") == inv2


def test_inventory_missing_returns_none(store):
    assert store.get_inventory("nope") is None


# ---------------------------------------------------------------------------
# ladder_state
# ---------------------------------------------------------------------------


def test_ladder_state_round_trip(store):
    ladder = LadderInv(
        net_band_exposure=[1.0, -2.0, 0.5],
        gross=10.0,
        phi=0.02,
        r3_histogram={1: 0.5, 24: 0.3, 168: 0.2},
    )
    vertical_offsets = {"mkt-2": 5.0, "mkt-3": -5.0}
    store.upsert_ladder_state("2026-07-20", ladder, vertical_offsets, updated_ts=NOW)
    got_ladder, got_offsets = store.get_ladder_state("2026-07-20")
    assert got_ladder == ladder
    assert got_offsets == vertical_offsets


def test_ladder_state_missing_returns_none(store):
    assert store.get_ladder_state("nope") is None


# ---------------------------------------------------------------------------
# orders
# ---------------------------------------------------------------------------


def test_orders_round_trip(store):
    store.upsert_order(
        "coid-1", "mkt-1", Side.BUY_YES, 0.45, 25.0, "LIVE",
        venue_order_id="v-1", ts_placed=NOW, ts_final=None,
    )
    rec = store.get_order("coid-1")
    assert rec.client_order_id == "coid-1"
    assert rec.market_id == "mkt-1"
    assert rec.side is Side.BUY_YES
    assert rec.price == 0.45
    assert rec.size == 25.0
    assert rec.status == "LIVE"
    assert rec.venue_order_id == "v-1"
    assert rec.ts_placed == NOW
    assert rec.ts_final is None


def test_orders_invalid_status_rejected(store):
    with pytest.raises(ValueError):
        store.upsert_order("coid-x", "mkt-1", Side.BUY_YES, 0.5, 1.0, "BOGUS")


def test_mark_all_live_orders_unknown(store):
    store.upsert_order("coid-1", "mkt-1", Side.BUY_YES, 0.45, 25.0, "LIVE", ts_placed=NOW)
    store.upsert_order("coid-2", "mkt-2", Side.BUY_NO, 0.30, 10.0, "LIVE", ts_placed=NOW)
    store.upsert_order("coid-3", "mkt-3", Side.BUY_YES, 0.55, 5.0, "FILLED", ts_placed=NOW)

    n = store.mark_all_live_orders_unknown()
    assert n == 2
    assert store.get_order("coid-1").status == "UNKNOWN"
    assert store.get_order("coid-2").status == "UNKNOWN"
    assert store.get_order("coid-3").status == "FILLED"


# ---------------------------------------------------------------------------
# fills (append-only)
# ---------------------------------------------------------------------------


def test_fills_round_trip(store):
    fill = Fill(
        ts=NOW, market_id="mkt-1", order_id="coid-1", side=Side.BUY_YES,
        price=0.45, size=10.0, liquidity=LiquiditySource.MAKER, venue_ts=NOW,
    )
    row_id = store.append_fill(fill)
    assert row_id > 0
    got = store.get_fills("mkt-1")
    assert len(got) == 1
    assert got[0].ts == NOW
    assert got[0].market_id == "mkt-1"
    assert got[0].order_id == "coid-1"
    assert got[0].side is Side.BUY_YES
    assert got[0].price == 0.45
    assert got[0].size == 10.0
    assert got[0].liquidity is LiquiditySource.MAKER
    assert got[0].venue_ts == NOW


def test_fills_no_update_or_delete_api_exposed(store):
    assert not hasattr(store, "update_fill")
    assert not hasattr(store, "delete_fill")
    assert not hasattr(store, "remove_fill")


def test_fills_append_only_multiple_rows_preserved(store):
    for i in range(3):
        fill = Fill(
            ts=NOW + timedelta(minutes=i), market_id="mkt-1", order_id=f"coid-{i}",
            side=Side.BUY_YES, price=0.4 + 0.01 * i, size=1.0,
            liquidity=LiquiditySource.MAKER, venue_ts=NOW,
        )
        store.append_fill(fill)
    got = store.get_fills("mkt-1")
    assert len(got) == 3
    assert [f.order_id for f in got] == ["coid-0", "coid-1", "coid-2"]


# ---------------------------------------------------------------------------
# quotes (append-only history)
# ---------------------------------------------------------------------------


def test_quotes_round_trip(store):
    qs = QuoteSet(
        ts=NOW, market_id="mkt-1", bid_price=0.44, ask_price=0.46,
        bid_size=10.0, ask_size=10.0,
        terms={"markup": 0.01, "eps": 0.0085, "skew": 0.0, "robust": 0.001},
        risk_mode=QuoteMode.TWO_SIDED, noarb_checked=True, source_seq=1,
    )
    row_id = store.append_quote(
        qs, r_x=0.1, delta_x=0.05, skew_x=0.0, sigma_b=0.2,
        params_id="cfg-v1", x_bid=0.05, x_ask=0.15, p_bid_raw=0.44, p_ask_raw=0.46,
    )
    assert row_id > 0
    got = store.get_quotes("mkt-1")
    assert len(got) == 1
    rec = got[0]
    assert rec.quote_set == qs
    assert rec.r_x == 0.1
    assert rec.delta_x == 0.05
    assert rec.sigma_b == 0.2
    assert rec.params_id == "cfg-v1"


def _quote_at(ts) -> QuoteSet:
    return QuoteSet(
        ts=ts, market_id="mkt-1", bid_price=0.44, ask_price=0.46,
        bid_size=10.0, ask_size=10.0,
        terms={"markup": 0.01, "eps": 0.0085, "skew": 0.0, "robust": 0.001},
        risk_mode=QuoteMode.TWO_SIDED, noarb_checked=True, source_seq=1,
    )


def _append_quote_at(store, ts) -> None:
    store.append_quote(
        _quote_at(ts), r_x=0.1, delta_x=0.05, skew_x=0.0, sigma_b=0.2,
        params_id="cfg-v1", x_bid=0.05, x_ask=0.15, p_bid_raw=0.44, p_ask_raw=0.46,
    )


def test_prune_quotes_deletes_only_rows_strictly_older_than_bound(store):
    # W0.2: prune_quotes mirrors prune_mid_log exactly -- insert at two
    # timestamps, prune, assert only the newer row(s) remain, and the
    # returned count matches the number of deleted rows.
    old_ts = NOW - timedelta(days=20)
    new_ts = NOW - timedelta(days=1)
    _append_quote_at(store, old_ts)
    _append_quote_at(store, new_ts)

    bound = NOW - timedelta(days=14)
    deleted = store.prune_quotes(bound)
    assert deleted == 1

    remaining = store.get_quotes("mkt-1")
    assert len(remaining) == 1
    assert remaining[0].quote_set.ts == new_ts


def test_prune_quotes_no_op_when_nothing_older(store):
    _append_quote_at(store, NOW)
    deleted = store.prune_quotes(NOW - timedelta(days=14))
    assert deleted == 0
    assert len(store.get_quotes("mkt-1")) == 1


# ---------------------------------------------------------------------------
# pnl
# ---------------------------------------------------------------------------


def test_pnl_round_trip(store):
    snap = PnlSnapshot(
        ts=NOW, market_id="mkt-1", expiry_key="2026-07-20",
        realized=1.5, unrealized_consensus=0.5, unrealized_mid=0.4,
        settlement_pnl=0.0, bankroll_utilization=0.3,
    )
    row_id = store.append_pnl_snapshot(snap)
    assert row_id > 0
    got = store.get_pnl_snapshots("mkt-1")
    assert len(got) == 1
    assert got[0] == snap


# ---------------------------------------------------------------------------
# settlements (idempotency)
# ---------------------------------------------------------------------------


def _settlement(outcome: SettlementOutcome, ts=NOW) -> SettlementEvent:
    return SettlementEvent(
        ts=ts, settlement_ts=ts, market_id="mkt-1", expiry_key="2026-07-20",
        strike=100000.0, outcome=outcome,
        spot_used=None if outcome is SettlementOutcome.UNSETTLEABLE else 101000.0,
        spot_source=SpotSource.NONE if outcome is SettlementOutcome.UNSETTLEABLE else SpotSource.INTRADAY,
        q_settled=10.0,
        payoff=None if outcome is SettlementOutcome.UNSETTLEABLE else 10.0,
        pnl_realized=None if outcome is SettlementOutcome.UNSETTLEABLE else 5.0,
        excluded_from_gate=outcome is SettlementOutcome.UNSETTLEABLE,
    )


def test_settlements_round_trip(store):
    ev = _settlement(SettlementOutcome.YES)
    assert store.upsert_settlement(ev) is True
    got = store.get_settlement("mkt-1", "2026-07-20")
    assert got == ev


def test_settlements_terminal_blocks_resettlement(store):
    ev1 = _settlement(SettlementOutcome.YES)
    assert store.upsert_settlement(ev1) is True

    ev2 = _settlement(SettlementOutcome.NO, ts=NOW + timedelta(hours=1))
    assert store.upsert_settlement(ev2) is False

    # Original terminal row is unchanged.
    got = store.get_settlement("mkt-1", "2026-07-20")
    assert got.outcome is SettlementOutcome.YES


def test_settlements_unsettleable_does_not_block_and_is_overwritten(store):
    ev1 = _settlement(SettlementOutcome.UNSETTLEABLE)
    assert store.upsert_settlement(ev1) is True

    ev2 = _settlement(SettlementOutcome.YES, ts=NOW + timedelta(hours=1))
    assert store.upsert_settlement(ev2) is True

    got = store.get_settlement("mkt-1", "2026-07-20")
    assert got.outcome is SettlementOutcome.YES
    assert got == ev2


def test_settlements_unsettleable_can_be_overwritten_by_another_unsettleable(store):
    ev1 = _settlement(SettlementOutcome.UNSETTLEABLE)
    assert store.upsert_settlement(ev1) is True
    ev2 = _settlement(SettlementOutcome.UNSETTLEABLE, ts=NOW + timedelta(hours=1))
    assert store.upsert_settlement(ev2) is True
    got = store.get_settlement("mkt-1", "2026-07-20")
    assert got == ev2


# ---------------------------------------------------------------------------
# bankrolls
# ---------------------------------------------------------------------------


def test_bankrolls_round_trip_and_append_only_versions(store):
    b1 = BankrollState(
        model_ids=["pricer", "market"], bankrolls={"pricer": 0.5, "market": 0.5},
        last_update=NOW, update_count=1, frozen=False,
    )
    b2 = BankrollState(
        model_ids=["pricer", "market"], bankrolls={"pricer": 0.6, "market": 0.4},
        last_update=NOW + timedelta(hours=1), update_count=2, frozen=False,
    )
    store.append_bankroll_state("2026-07-20", b1)
    store.append_bankroll_state("2026-07-20", b2)

    latest = store.get_latest_bankroll_state("2026-07-20")
    assert latest == b2

    history = store.get_bankroll_history("2026-07-20")
    assert history == [b1, b2]


def test_bankrolls_per_region_round_trip(store):
    # Package B2: region is an additive arg, default '' preserves the legacy
    # single-scalar shape (test above); "belly"/"wing" rows are independent
    # append-only streams keyed on (expiry_key, region).
    belly = BankrollState(
        model_ids=["pricer", "market"], bankrolls={"pricer": 0.7, "market": 0.3},
        last_update=NOW, update_count=1, frozen=False,
    )
    wing = BankrollState(
        model_ids=["pricer", "market"], bankrolls={"pricer": 0.4, "market": 0.6},
        last_update=NOW, update_count=1, frozen=False,
    )
    store.append_bankroll_state("2026-07-20", belly, region="belly")
    store.append_bankroll_state("2026-07-20", wing, region="wing")

    assert store.get_latest_bankroll_state("2026-07-20", region="belly") == belly
    assert store.get_latest_bankroll_state("2026-07-20", region="wing") == wing
    # The legacy region='' stream is untouched by per-region writes.
    assert store.get_latest_bankroll_state("2026-07-20") is None

    assert store.get_bankroll_history("2026-07-20", region="belly") == [belly]
    assert store.get_bankroll_history("2026-07-20", region="wing") == [wing]


def test_bankrolls_migration_adds_region_column_to_pre_migration_db(tmp_path):
    # Guarded migration (plan B2 step 9, review item 2 BLOCKER resolution):
    # a pre-existing DB written before the region column existed must gain
    # it on the next MMStateStore open, WITHOUT losing the legacy row (which
    # must remain readable at region='').
    import sqlite3

    db_path = str(tmp_path / "pre_migration.db")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE bankrolls (
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
        conn.execute(
            "INSERT INTO bankrolls (expiry_key, model_ids, bankrolls, last_update, update_count, frozen) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "2026-07-20", '["pricer", "market"]', '{"pricer": 0.5, "market": 0.5}',
                NOW.isoformat(), 3, 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    store = MMStateStore(db_path)
    try:
        cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(bankrolls)")}
        assert "region" in cols

        legacy = store.get_latest_bankroll_state("2026-07-20")  # region='' default
        assert legacy is not None
        assert legacy.bankrolls == {"pricer": 0.5, "market": 0.5}
        assert legacy.update_count == 3

        # Per-region round-trip still works after the migration.
        belly = BankrollState(
            model_ids=["pricer", "market"], bankrolls={"pricer": 0.65, "market": 0.35},
            last_update=NOW, update_count=1, frozen=False,
        )
        store.append_bankroll_state("2026-07-20", belly, region="belly")
        assert store.get_latest_bankroll_state("2026-07-20", region="belly") == belly
        # The legacy row is unaffected by the new per-region write.
        assert store.get_latest_bankroll_state("2026-07-20") == legacy
    finally:
        store.close()


# ---------------------------------------------------------------------------
# risk_journal
# ---------------------------------------------------------------------------


def test_risk_journal_round_trip(store):
    directive = RiskDirective(
        ts=NOW, market_id="mkt-1", mode=QuoteMode.PULLED, eps_add=0.02,
        kelly_mult=0.5, triggers=[RiskTrigger.SPOT_JUMP, RiskTrigger.FEED_STALE],
        latched_until=NOW + timedelta(minutes=5), cancel_all=True,
    )
    store.append_risk_directive(directive)
    got = store.get_risk_journal("mkt-1")
    assert len(got) == 1
    assert got[0] == directive


# ---------------------------------------------------------------------------
# liquidity_windows
# ---------------------------------------------------------------------------


def test_liquidity_windows_round_trip(store):
    ls = LiquidityState(
        ts=NOW, market_id="mkt-1", realized_depth_bid=100.0, realized_depth_ask=90.0,
        kyle_lambda=0.001, arb_halflife_s=30.0, regime=LiquidityRegime.NORMAL,
        window="5m", vol_discount=2.5,
    )
    store.append_liquidity_window(ls)
    got = store.get_liquidity_windows("mkt-1")
    assert len(got) == 1
    assert got[0] == ls


def _liq_at(ts, market_id="mkt-1") -> LiquidityState:
    return LiquidityState(
        ts=ts, market_id=market_id, realized_depth_bid=50.0, realized_depth_ask=40.0,
        kyle_lambda=None, arb_halflife_s=None, regime=LiquidityRegime.NORMAL,
        window="5m", vol_discount=2.5,
    )


def test_prune_liquidity_windows_deletes_only_rows_strictly_older_than_bound(store):
    # W1.1: prune_liquidity_windows mirrors prune_quotes exactly -- insert at
    # two timestamps, prune, assert only the newer row(s) remain, and the
    # returned count matches the number of deleted rows.
    old_ts = NOW - timedelta(days=20)
    new_ts = NOW - timedelta(days=1)
    store.append_liquidity_window(_liq_at(old_ts))
    store.append_liquidity_window(_liq_at(new_ts))

    bound = NOW - timedelta(days=14)
    deleted = store.prune_liquidity_windows(bound)
    assert deleted == 1

    remaining = store.get_liquidity_windows("mkt-1")
    assert len(remaining) == 1
    assert remaining[0].ts == new_ts


def test_prune_liquidity_windows_no_op_when_nothing_older(store):
    store.append_liquidity_window(_liq_at(NOW))
    deleted = store.prune_liquidity_windows(NOW - timedelta(days=14))
    assert deleted == 0
    assert len(store.get_liquidity_windows("mkt-1")) == 1


# ---------------------------------------------------------------------------
# Crash-consistency: transactional fill + inventory write
# ---------------------------------------------------------------------------


def test_record_fill_and_update_inventory_atomic(store):
    fill = Fill(
        ts=NOW, market_id="mkt-1", order_id="coid-1", side=Side.BUY_YES,
        price=0.5, size=10.0, liquidity=LiquiditySource.MAKER, venue_ts=NOW,
    )
    resulting_inv = ContractInv(q=10.0, avg_cost=0.5, q_max=100.0, age_weighted_holding=0.0)
    store.record_fill_and_update_inventory(fill, resulting_inv)

    assert len(store.get_fills("mkt-1")) == 1
    assert store.get_inventory("mkt-1") == resulting_inv


# ---------------------------------------------------------------------------
# fills fold invariant
# ---------------------------------------------------------------------------


def test_fold_fills_to_inventory_matches_scripted_scenario(store):
    # BUY_YES 10 @ 0.40 -> q=10
    store.append_fill(Fill(
        ts=NOW, market_id="mkt-1", order_id="c1", side=Side.BUY_YES,
        price=0.40, size=10.0, liquidity=LiquiditySource.MAKER, venue_ts=NOW,
    ))
    # BUY_YES 5 @ 0.50 -> q=15
    store.append_fill(Fill(
        ts=NOW + timedelta(minutes=1), market_id="mkt-1", order_id="c2", side=Side.BUY_YES,
        price=0.50, size=5.0, liquidity=LiquiditySource.TAKER, venue_ts=NOW,
    ))
    # BUY_NO 15 @ 0.30 -> closes the position exactly (q -> 0)
    store.append_fill(Fill(
        ts=NOW + timedelta(minutes=2), market_id="mkt-1", order_id="c3", side=Side.BUY_NO,
        price=0.30, size=15.0, liquidity=LiquiditySource.MAKER, venue_ts=NOW,
    ))
    # A second market with an open position that gets closed by SETTLEMENT.
    store.append_fill(Fill(
        ts=NOW, market_id="mkt-2", order_id="c4", side=Side.BUY_YES,
        price=0.60, size=20.0, liquidity=LiquiditySource.MAKER, venue_ts=NOW,
    ))
    # SETTLEMENT pseudo-fill closes mkt-2 fully (YES won, payoff=1 per share).
    store.append_fill(Fill(
        ts=NOW + timedelta(days=1), market_id="mkt-2", order_id="settle-mkt-2", side=Side.BUY_NO,
        price=0.0, size=20.0, liquidity=LiquiditySource.SETTLEMENT, venue_ts=NOW + timedelta(days=1),
    ))

    folded = store.fold_fills_to_inventory()
    assert folded["mkt-1"].q == pytest.approx(0.0)
    assert folded["mkt-2"].q == pytest.approx(0.0)


def test_fold_fills_to_inventory_partial_position(store):
    store.append_fill(Fill(
        ts=NOW, market_id="mkt-3", order_id="c1", side=Side.BUY_YES,
        price=0.40, size=10.0, liquidity=LiquiditySource.MAKER, venue_ts=NOW,
    ))
    store.append_fill(Fill(
        ts=NOW + timedelta(minutes=1), market_id="mkt-3", order_id="c2", side=Side.BUY_NO,
        price=0.60, size=4.0, liquidity=LiquiditySource.MAKER, venue_ts=NOW,
    ))
    folded = store.fold_fills_to_inventory()
    assert folded["mkt-3"].q == pytest.approx(6.0)


def test_fold_fills_matches_persisted_inventory_after_transactional_writes(store):
    # Two sequential atomic fill+inventory writes; fold(fills) should match
    # the persisted inventory table exactly (plan risk 8.2 invariant).
    f1 = Fill(
        ts=NOW, market_id="mkt-4", order_id="c1", side=Side.BUY_YES,
        price=0.4, size=10.0, liquidity=LiquiditySource.MAKER, venue_ts=NOW,
    )
    store.record_fill_and_update_inventory(
        f1, ContractInv(q=10.0, avg_cost=0.4, q_max=100.0, age_weighted_holding=0.0)
    )
    f2 = Fill(
        ts=NOW + timedelta(minutes=5), market_id="mkt-4", order_id="c2", side=Side.BUY_YES,
        price=0.5, size=5.0, liquidity=LiquiditySource.TAKER, venue_ts=NOW,
    )
    store.record_fill_and_update_inventory(
        f2, ContractInv(q=15.0, avg_cost=(0.4 * 10 + 0.5 * 5) / 15, q_max=100.0, age_weighted_holding=0.0)
    )

    folded = store.fold_fills_to_inventory()
    persisted = store.get_all_inventory()
    assert folded["mkt-4"].q == pytest.approx(persisted["mkt-4"].q)


# ---------------------------------------------------------------------------
# get_live_orders (plan B4-CPU / 1.1)
# ---------------------------------------------------------------------------


def _seed_mixed_status_orders(store):
    store.upsert_order("c-pending", "mkt-1", Side.BUY_YES, 0.40, 10.0, "PENDING", ts_placed=NOW)
    store.upsert_order("c-live-yes", "mkt-1", Side.BUY_YES, 0.42, 5.0, "LIVE", ts_placed=NOW)
    store.upsert_order("c-live-no", "mkt-1", Side.BUY_NO, 0.55, 5.0, "LIVE", ts_placed=NOW)
    store.upsert_order("c-cancelled", "mkt-1", Side.BUY_YES, 0.30, 1.0, "CANCELLED", ts_placed=NOW)
    store.upsert_order("c-filled", "mkt-1", Side.BUY_YES, 0.35, 2.0, "FILLED", ts_placed=NOW)
    store.upsert_order("c-unknown", "mkt-1", Side.BUY_YES, 0.36, 2.0, "UNKNOWN", ts_placed=NOW)
    store.upsert_order("c-other-mkt", "mkt-2", Side.BUY_YES, 0.60, 8.0, "LIVE", ts_placed=NOW)


def test_get_live_orders_status_filter(store):
    _seed_mixed_status_orders(store)
    live = store.get_live_orders()
    ids = {o.client_order_id for o in live}
    assert ids == {"c-pending", "c-live-yes", "c-live-no", "c-other-mkt"}


def test_get_live_orders_market_and_side_filters(store):
    _seed_mixed_status_orders(store)
    by_market = store.get_live_orders(market_id="mkt-1")
    assert {o.client_order_id for o in by_market} == {"c-pending", "c-live-yes", "c-live-no"}

    by_side = store.get_live_orders(market_id="mkt-1", side=Side.BUY_NO)
    assert [o.client_order_id for o in by_side] == ["c-live-no"]

    by_market_2 = store.get_live_orders(market_id="mkt-2")
    assert [o.client_order_id for o in by_market_2] == ["c-other-mkt"]

    none_match = store.get_live_orders(market_id="mkt-nope")
    assert none_match == []


def test_get_live_orders_matches_old_scan_and_filter(store):
    """Regression guard: get_live_orders() must return exactly what the
    prior `[r for r in get_all_orders() if ...]` scan pattern produced,
    in the same order (plan 1.1 -- guards tests/test_mm_integration.py's
    exact-stability assertions)."""
    _seed_mixed_status_orders(store)
    for market_id in (None, "mkt-1", "mkt-2", "mkt-nope"):
        for side in (None, Side.BUY_YES, Side.BUY_NO):
            expected = [
                r for r in store.get_all_orders()
                if r.status in ("PENDING", "LIVE")
                and (market_id is None or r.market_id == market_id)
                and (side is None or r.side == side)
            ]
            got = store.get_live_orders(market_id, side)
            assert [o.client_order_id for o in got] == [o.client_order_id for o in expected]


def test_idx_orders_status_index_exists(store):
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'orders'"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert "idx_orders_status" in names


# ---------------------------------------------------------------------------
# markets registry (plan B3-schema / 1.3)
# ---------------------------------------------------------------------------


def test_markets_registry_round_trip(store):
    store.upsert_market("m-98k", "2026-07-20", 98000.0)
    store.upsert_market("m-100k", "2026-07-20", 100000.0)
    reg = store.get_market_registry()
    assert reg == {"m-98k": ("2026-07-20", 98000.0), "m-100k": ("2026-07-20", 100000.0)}


def test_markets_registry_upsert_overwrites(store):
    store.upsert_market("m-98k", "2026-07-20", 98000.0)
    store.upsert_market("m-98k", "2026-08-03", 99000.0)  # market_id reused for a new event
    reg = store.get_market_registry()
    assert reg == {"m-98k": ("2026-08-03", 99000.0)}


def test_markets_registry_empty_by_default(store):
    assert store.get_market_registry() == {}


# ---------------------------------------------------------------------------
# mid_log (mm_suitability_alignment_plan.md Change C1)
# ---------------------------------------------------------------------------


def test_append_mids_and_get_mids_round_trip(store):
    store.append_mids(NOW, {"m1": 0.40, "m2": 0.60})
    store.append_mids(NOW + timedelta(seconds=30), {"m1": 0.42})

    all_rows = store.get_mids()
    assert len(all_rows) == 3
    assert {r.market_id for r in all_rows} == {"m1", "m2"}

    m1_rows = store.get_mids("m1")
    assert [r.mid for r in m1_rows] == [0.40, 0.42]
    assert m1_rows[0].ts == NOW
    assert m1_rows[1].ts == NOW + timedelta(seconds=30)


def test_append_mids_empty_dict_is_noop(store):
    store.append_mids(NOW, {})
    assert store.get_mids() == []


def test_idx_mid_log_market_ts_index_exists(store):
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'mid_log'"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert "idx_mid_log_market_ts" in names


# ---------------------------------------------------------------------------
# trade_prints (2026-07-11, arrival-decay calibration input)
# ---------------------------------------------------------------------------


def test_append_trade_prints_round_trip(store):
    n = store.append_trade_prints({
        "m1": [(NOW, 0.61, 5.0), (NOW + timedelta(seconds=2), 0.62, 1.0)],
        "m2": [(NOW, 0.11, 3.0)],
    })
    assert n == 3

    all_rows = store.get_trade_prints()
    assert len(all_rows) == 3
    assert {r.market_id for r in all_rows} == {"m1", "m2"}

    m1_rows = store.get_trade_prints("m1")
    assert [(r.price, r.size) for r in m1_rows] == [(0.61, 5.0), (0.62, 1.0)]
    assert m1_rows[0].ts == NOW


def test_append_trade_prints_empty_is_noop(store):
    assert store.append_trade_prints({}) == 0
    assert store.append_trade_prints({"m1": []}) == 0
    assert store.get_trade_prints() == []


def test_prune_trade_prints_deletes_strictly_older(store):
    store.append_trade_prints({"m1": [(NOW, 0.5, 1.0)]})
    store.append_trade_prints({"m1": [(NOW + timedelta(seconds=60), 0.51, 1.0)]})
    deleted = store.prune_trade_prints(NOW + timedelta(seconds=60))
    assert deleted == 1
    rows = store.get_trade_prints("m1")
    assert len(rows) == 1
    assert rows[0].price == 0.51


def test_mid_at_or_after_picks_first_row_at_or_after_bound(store):
    store.append_mids(NOW, {"m1": 0.10})
    store.append_mids(NOW + timedelta(seconds=60), {"m1": 0.20})
    store.append_mids(NOW + timedelta(seconds=120), {"m1": 0.30})

    # Bound lands exactly on the middle row -> that row (inclusive >=).
    got = store.mid_at_or_after("m1", NOW + timedelta(seconds=60), NOW + timedelta(seconds=600))
    assert got == pytest.approx(0.20)

    # Bound lands strictly between rows -> the NEXT row at/after the bound,
    # not the nearest.
    got2 = store.mid_at_or_after("m1", NOW + timedelta(seconds=90), NOW + timedelta(seconds=600))
    assert got2 == pytest.approx(0.30)


def test_mid_at_or_after_outside_window_returns_none(store):
    store.append_mids(NOW, {"m1": 0.10})
    # Only row is before the window entirely.
    got = store.mid_at_or_after("m1", NOW + timedelta(seconds=60), NOW + timedelta(seconds=120))
    assert got is None


def test_mid_at_or_after_no_rows_at_all_returns_none(store):
    assert store.mid_at_or_after("m1", NOW, NOW + timedelta(seconds=600)) is None


def test_mid_at_or_after_scoped_to_market_id(store):
    store.append_mids(NOW, {"m1": 0.10, "m2": 0.90})
    got = store.mid_at_or_after("m1", NOW, NOW + timedelta(seconds=1))
    assert got == pytest.approx(0.10)
    got_other = store.mid_at_or_after("m2", NOW, NOW + timedelta(seconds=1))
    assert got_other == pytest.approx(0.90)
    got_missing = store.mid_at_or_after("m3", NOW, NOW + timedelta(seconds=1))
    assert got_missing is None


def test_mid_at_or_after_microsecond_width_consistency(store):
    # _dt_to_iso ground truth: Python's dt.isoformat() omits the microseconds
    # field entirely when microsecond == 0, producing variable-width TEXT
    # across rows. Mixing microsecond=0 and microsecond!=0 timestamps in the
    # same market must still order correctly through the (market_id, ts)
    # TEXT index -- this is the exact scenario the module docstring on
    # mid_at_or_after calls out.
    ts0 = NOW  # microsecond == 0 -> short ISO string
    ts1 = NOW + timedelta(seconds=1, microseconds=500000)  # nonzero -> long ISO string
    store.append_mids(ts0, {"m1": 0.10})
    store.append_mids(ts1, {"m1": 0.20})

    bound = NOW + timedelta(seconds=1)  # strictly between ts0 and ts1
    got = store.mid_at_or_after("m1", bound, bound + timedelta(seconds=5))
    assert got == pytest.approx(0.20)


def test_mid_at_or_after_upper_bound_is_exclusive(store):
    # F1: mid_at_or_after's upper bound flipped from `ts <= ts_max` to
    # `ts < ts_max` -- a row landing exactly ON ts_max must now be excluded
    # (this is what makes pnl_report.markout_report's abutting horizon
    # windows disjoint rather than double-counting the boundary row).
    ts_max = NOW + timedelta(seconds=600)
    store.append_mids(ts_max, {"m1": 0.50})

    got = store.mid_at_or_after("m1", NOW, ts_max)
    assert got is None

    # One microsecond earlier is inside the window.
    store.append_mids(ts_max - timedelta(microseconds=1), {"m1": 0.40})
    got_inside = store.mid_at_or_after("m1", NOW, ts_max)
    assert got_inside == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# mid_log pruning (F3)
# ---------------------------------------------------------------------------


def test_prune_mid_log_deletes_only_rows_strictly_older_than_bound(store):
    store.append_mids(NOW - timedelta(days=10), {"m1": 0.10})
    store.append_mids(NOW - timedelta(days=8), {"m1": 0.20})
    bound = NOW - timedelta(days=7)
    store.append_mids(bound, {"m1": 0.30})  # exactly on the bound -> survives (not "< bound")
    store.append_mids(NOW - timedelta(days=1), {"m1": 0.40})

    deleted = store.prune_mid_log(bound)
    assert deleted == 2  # the two rows strictly older than `bound`

    remaining = sorted(r.mid for r in store.get_mids("m1"))
    assert remaining == [pytest.approx(0.30), pytest.approx(0.40)]


def test_prune_mid_log_no_op_when_nothing_older(store):
    store.append_mids(NOW, {"m1": 0.10})
    deleted = store.prune_mid_log(NOW - timedelta(days=7))
    assert deleted == 0
    assert len(store.get_mids("m1")) == 1


def test_prune_mid_log_after_report_leaves_rows_within_retention_intact(store):
    # F3: paper_runner writes the markout report BEFORE pruning; this asserts
    # the ordering invariant at the store level -- rows within the retention
    # window used by a just-generated report survive a subsequent prune call,
    # so a report generated immediately after pruning would see the same
    # rows as one generated immediately before.
    retention_s = 7 * 86400.0
    now = NOW
    within_row_ts = now - timedelta(seconds=retention_s - 3600.0)  # just inside retention
    outside_row_ts = now - timedelta(seconds=retention_s + 3600.0)  # just outside retention
    store.append_mids(within_row_ts, {"m1": 0.55})
    store.append_mids(outside_row_ts, {"m1": 0.45})

    # Simulate "read for the report" happening before the prune call.
    pre_prune_rows = store.get_mids("m1")
    assert len(pre_prune_rows) == 2

    store.prune_mid_log(now - timedelta(seconds=retention_s))

    post_prune_rows = store.get_mids("m1")
    assert len(post_prune_rows) == 1
    assert post_prune_rows[0].ts == within_row_ts


# ---------------------------------------------------------------------------
# Kill/restart round-trip
# ---------------------------------------------------------------------------


def test_kill_restart_round_trip(tmp_path):
    db_path = str(tmp_path / "mm_state.db")
    s1 = MMStateStore(db_path)
    try:
        inv = ContractInv(q=5.0, avg_cost=0.45, q_max=50.0, age_weighted_holding=2.0)
        s1.upsert_inventory("mkt-1", inv, updated_ts=NOW)
        s1.upsert_order("coid-1", "mkt-1", Side.BUY_YES, 0.45, 5.0, "LIVE", ts_placed=NOW)
        b1 = BankrollState(
            model_ids=["pricer", "market"], bankrolls={"pricer": 0.5, "market": 0.5},
            last_update=NOW, update_count=1, frozen=False,
        )
        s1.append_bankroll_state("2026-07-20", b1)
    finally:
        s1.close()

    # Simulate a process restart: brand-new store instance on the same file.
    s2 = MMStateStore(db_path)
    try:
        assert s2.get_inventory("mkt-1") == inv
        assert s2.get_order("coid-1").status == "LIVE"
        assert s2.get_latest_bankroll_state("2026-07-20") == b1

        n = s2.mark_all_live_orders_unknown()
        assert n == 1
        assert s2.get_order("coid-1").status == "UNKNOWN"

        # Inventory/bankroll state survived the restart unchanged.
        assert s2.get_inventory("mkt-1") == inv
        assert s2.get_latest_bankroll_state("2026-07-20") == b1
    finally:
        s2.close()
