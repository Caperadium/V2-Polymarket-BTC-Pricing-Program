"""Paper-trading integration harness (plan task G1, scripted-feed subset).

A THIN orchestrator wiring one tick of the full market-making loop for ONE
expiry ladder against SCRIPTED synthetic feeds (no live data, no backtest
replay). Wiring and sequencing only -- no business logic lives here; every
computation is delegated to the frozen market_maker components.

Per-tick sequence (Option: decide-then-observe, so a one-tick risk lag exists
between a fill and the controller's reaction, matching a real venue):
  1. advance the injected clock by tick_dt
  2. apply scripted book messages to each market's BookMirror -> MarketState;
     feed the liquidity monitor
  3. build the pricer snapshot from the injected engine_fn (reuse the previous
     snapshot on engine failure, keeping its stale ts so the risk controller's
     staleness path can fire)
  4. Beuoy fair-value anchor, threading BankrollState / forecasts / consensus
  5. risk directives per market (inventory breaches, liquidity regime, feed
     health, pricer staleness)
  6. quote proposals -> joint ladder sizing -> spread builder -> MANDATORY
     ladder-hedger no-arb check/repair (a rejected/failed ladder never reaches
     the lifecycle) -> order lifecycle over the PaperVenueAdapter
  7. feed the same MarketState to the fill simulator; route fills atomically to
     the InventoryManager AND the state store
     (record_fill_and_update_inventory); reconcile fully-consumed store orders

`settle()` delegates to the SettlementHandler (with catch_up support) and syncs
the settlement pseudo-fills back into the in-memory InventoryManager so the
fold(fills) == inventory invariant survives resolution.

This file doubles as the Stage-A shadow-loop skeleton, so every stage journals
its output on the instance (last_snapshot, last_fair_value, last_directives,
last_quote_sets, last_proposals, last_fills, checked_ladders, ...).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from market_maker.config import MMConfig
from market_maker.contracts import (
    BankrollState,
    ContractInv,
    Fill,
    LiquidityRegime,
    LiquiditySource,
    QuoteSet,
    Side,
    VenueDescriptor,
)
from market_maker.fair_value_anchor import (
    MARKET_MODEL_ID,
    PRICER_MODEL_ID,
    compute_fair_value,
)
from market_maker.inventory_manager import InventoryManager
from market_maker.ladder_hedger import LadderHedger
from market_maker.liquidity_monitor import LiquidityMonitor
from market_maker.market_data_client import BookMirror, FeedCapability
from market_maker.order_lifecycle import OrderLifecycleManager, PaperVenueAdapter, SimClock
from market_maker.paper_fill_sim import PaperFillSimulator
from market_maker.pricer_adapter import build_snapshot
from market_maker.quote_engine import estimate_sigma_b, make_quote_from_config
from market_maker.risk_controller import InvBreach, RiskController
from market_maker.robustness_sizing import ContractSizingInput, size_ladder
from market_maker.settlement_handler import (
    BTCDataProvider,
    MarketPosition,
    SettlementHandler,
    settlement_instant_utc,
)
from market_maker.spread_builder import build_quote_set
from market_maker.state_store import MMStateStore

_LIVE_STATUSES = ("PENDING", "LIVE")


class _PaperFillSimBridge:
    """Adapts PaperFillSimulator to the place_order/cancel_order/open_orders
    shape the PaperVenueAdapter and OrderLifecycleManager expect.

    Interface reconciliation (reported, not a bug): PaperVenueAdapter probes its
    fill_sim for `place_order(coid, market_id, side, price, size)` /
    `cancel_order(coid)` / `open_orders()`, but PaperFillSimulator exposes
    `place(order_id, market_id, side_str, price, size, decision_ts)` /
    `cancel(order_id, decision_ts)` / `open_orders()` keyed on `order_id`. This
    bridge translates the Side enum + lifecycle's sell-YES-via-buy-NO price
    convention back to the geometric book (bid @ bid_price, ask @ ask_price) and
    supplies decision_ts from the SimClock.
    """

    def __init__(self, sim: PaperFillSimulator, clock: SimClock) -> None:
        self._sim = sim
        self._clock = clock

    def place_order(self, coid: str, market_id: str, side: Side, price: float, size: float) -> None:
        # Lifecycle quotes the ask as Side.BUY_NO @ (1 - ask_price); undo that to
        # rest the ask on the geometric YES book at ask_price.
        if side == Side.BUY_YES:
            book_side, book_price = "bid", price
        else:
            book_side, book_price = "ask", 1.0 - price
        self._sim.place(coid, market_id, book_side, book_price, size, self._clock.now())

    def cancel_order(self, coid: str) -> None:
        self._sim.cancel(coid, self._clock.now())

    def open_orders(self) -> List[Dict[str, Any]]:
        return [
            {"client_order_id": o["order_id"], "market_id": o["market_id"]}
            for o in self._sim.open_orders()
        ]


def _initial_bankroll_state(now: datetime) -> BankrollState:
    return BankrollState(
        model_ids=[PRICER_MODEL_ID, MARKET_MODEL_ID],
        bankrolls={PRICER_MODEL_ID: 0.5, MARKET_MODEL_ID: 0.5},
        last_update=now,
        update_count=0,
        frozen=False,
    )


def _market_mid(ms) -> Optional[float]:
    b, a = ms.best_bid, ms.best_ask
    if b is not None and a is not None:
        return 0.5 * (float(b) + float(a))
    if b is not None:
        return float(b)
    if a is not None:
        return float(a)
    return None


class PaperTradingLoop:
    """One-expiry-ladder paper-trading loop over scripted feeds."""

    def __init__(
        self,
        *,
        store: MMStateStore,
        expiry_key: str,
        markets: List[Tuple[str, float]],
        engine_fn: Callable[..., Dict[Any, Any]],
        config: Optional[MMConfig] = None,
        clock: Optional[SimClock] = None,
        vol_gate_fn: Optional[Callable[[], object]] = None,
        data_provider: Optional[BTCDataProvider] = None,
        bankroll: float = 1000.0,
        tick_dt_s: float = 60.0,
        quote_variant: str = "dalen",
        hedger_mode: str = "repair",
        trade_through_only: bool = False,
        feed_capability: FeedCapability = FeedCapability.FULL_L2,
        tick: float = 0.01,
        journal_maxlen: Optional[int] = 20_000,
        x_hist_maxlen: Optional[int] = 20_000,
    ) -> None:
        self.store = store
        self.expiry_key = expiry_key
        self.config = config or MMConfig()
        self.clock = clock or SimClock(datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc))
        self.engine_fn = engine_fn
        self.bankroll = bankroll
        self.tick_dt_s = tick_dt_s
        self.quote_variant = quote_variant
        self.tick_size = tick
        # B4-memory: bound the in-memory journals (unbounded => ~1GB+ of
        # QuoteSets over a month-long unattended run, harness.py:399-401 /
        # :208,274). None means unbounded (legacy behavior); everything
        # trimmed here is already durably persisted via store.append_quote,
        # so trimming loses no data (only in-memory journal convenience).
        self.journal_maxlen = journal_maxlen
        self.x_hist_maxlen = x_hist_maxlen

        # markets sorted ascending by strike (ladder order)
        self.markets: List[Tuple[str, float]] = sorted(markets, key=lambda t: t[1])
        self.strikes: List[float] = [k for _, k in self.markets]
        self.mid_by_strike: Dict[float, str] = {k: m for m, k in self.markets}
        self.strike_by_mid: Dict[str, float] = {m: k for m, k in self.markets}

        # Persist the market registry (plan B3-schema 1.3): idempotent,
        # one-time per construction -- enables a restarted process's
        # settlement catch_up() to find THIS run's markets even before any
        # quoting activity, and (merged with the store's prior contents in
        # settle()) a PREVIOUS run's markets too.
        for m, k in self.markets:
            self.store.upsert_market(m, expiry_key, k)

        self.venue_descriptor = VenueDescriptor(
            tick_size=tick, min_size=1.0, price_band=self.config.p_clamp,
            maker_fee=0.0, maker_rebate=0.0,
            settlement_rule="12:00 ET on expiry date", supports_ladder=True,
        )

        # components
        self.inv = InventoryManager(self.config)
        for m, k in self.markets:
            self.inv.register_market(m, expiry_key, k)
            self.inv.update_fair_x(m, 0.0)

        self.hedger = LadderHedger(config=self.config, repair_or_reject=hedger_mode, tick=tick)
        self.fill_sim = PaperFillSimulator(self.config, trade_through_only=trade_through_only)
        self.bridge = _PaperFillSimBridge(self.fill_sim, self.clock)
        self.venue = PaperVenueAdapter(self.bridge, store, self.venue_descriptor)
        self.lifecycle = OrderLifecycleManager(self.venue, store, self.config, self.clock)
        self.risk = RiskController(self.config, vol_gate_fn=vol_gate_fn)
        self.settlement = SettlementHandler(store, self.config, data_provider)

        self.books: Dict[str, BookMirror] = {
            m: BookMirror(capability=feed_capability, config=self.config, clock=self.clock.now)
            for m, _ in self.markets
        }
        self.monitors: Dict[str, LiquidityMonitor] = {
            m: LiquidityMonitor(self.config, tick_size=tick) for m, _ in self.markets
        }

        # threaded fair-value state
        self.bankroll_state = _initial_bankroll_state(self.clock.now())
        self.prev_forecasts: Optional[Dict[str, np.ndarray]] = None
        self.prev_consensus: Optional[np.ndarray] = None
        self._x_hist: Dict[str, List[float]] = {m: [] for m, _ in self.markets}
        self._seq: Dict[str, int] = {m: 0 for m, _ in self.markets}

        # journal
        self._tick = 0
        self.last_snapshot = None
        self.last_fair_value = None
        self.last_directives: Dict[str, Any] = {}
        self.last_proposals: Dict[str, Any] = {}
        self.last_quote_sets: Dict[str, QuoteSet] = {}
        self.last_checked_quote_sets: Optional[List[QuoteSet]] = None
        self.last_fills: List = []
        self.last_liquidity: Dict[str, Any] = {}
        self.checked_ladders: List[Tuple[List[float], List[QuoteSet]]] = []
        self.all_checked_quote_sets: List[QuoteSet] = []
        self.snapshot_failed: bool = False

    # -- helpers ----------------------------------------------------------

    def _hours_to_expiry(self, now: datetime) -> float:
        return (settlement_instant_utc(self.expiry_key) - now).total_seconds() / 3600.0

    def _apply_messages(self, market_id: str, msgs: List[Dict[str, Any]], now: datetime) -> None:
        book = self.books[market_id]
        for msg in msgs:
            m = dict(msg)
            m["ts"] = now
            if "seq" not in m:
                self._seq[market_id] += 1
                m["seq"] = self._seq[market_id]
            if m.get("type") == "trade":
                # stamp the print ts to the tick time so latency math is exact
                pass
            book.on_message(m)

    def _breaches(self) -> List[InvBreach]:
        breaches: List[InvBreach] = []
        snap = self.inv.snapshot(self.clock.now())
        for m, _ in self.markets:
            ci = snap.per_contract.get(m)
            if ci is None or ci.q_max <= 0.0:
                continue
            ratio = abs(ci.q) / ci.q_max
            if ratio >= 1.0:
                breaches.append(InvBreach(market_id=m, is_long=ci.q > 0.0, ratio=ratio))
        return breaches

    def _compose_quote_sets(self, snap, fv, directives) -> List[Tuple[float, str, QuoteSet]]:
        """Quote engine -> sizing -> spread builder for the whole ladder. A
        seam the forced-no-arb test monkeypatches to inject a crossing ladder."""
        cfg = self.config
        now = self.clock.now()
        tte = max(snap.tte_days, 0.0)

        proposals = {}
        contracts: List[ContractSizingInput] = []
        # sigma_b sampling stride: estimate on consensus-x subsampled to
        # >= cfg.sigma_b_sample_s so tick-frequency microstructure noise
        # (mid jitter, spread bounce) is not annualized into belief vol
        # (Stage-A shadow finding 2026-07-07; plan 10.2).
        sample_s = float(getattr(cfg, "sigma_b_sample_s", 0.0) or 0.0)
        stride = max(1, int(round(sample_s / self.tick_dt_s))) if sample_s > 0 else 1
        sample_dt_days = (self.tick_dt_s * stride) / 86400.0

        for m, k in self.markets:
            self._x_hist[m].append(float(fv.consensus_x[k]))
            if self.x_hist_maxlen is not None and len(self._x_hist[m]) > self.x_hist_maxlen:
                # Trim from the front -- keep the newest entries (the sigma_b
                # estimator below is itself newest-anchored, plan B4-memory).
                del self._x_hist[m][: len(self._x_hist[m]) - self.x_hist_maxlen]
            hist = self._x_hist[m][::-1][::stride][::-1]  # newest-anchored subsample
            sigma_b = estimate_sigma_b(
                hist, sample_dt_days, cfg.sigma_b_floor, cfg.sigma_b_cap
            )
            q = self.inv.snapshot(now).per_contract[m].q
            prop = make_quote_from_config(
                cfg, m, float(fv.consensus_x[k]), q, tte, sigma_b, variant=self.quote_variant, ts=now
            )
            proposals[m] = prop
            contracts.append(
                ContractSizingInput(market_id=m, p_hat=float(fv.consensus_p[k]),
                                    bid_price=prop.p_bid_raw, ask_price=prop.p_ask_raw)
            )
        self.last_proposals = proposals

        decisions, _audit = size_ladder(contracts, snap, self.bankroll, now, cfg, liquidity=None)

        out: List[Tuple[float, str, QuoteSet]] = []
        for m, k in self.markets:
            qs = build_quote_set(
                proposals[m], directives[m], decisions[m], self.venue_descriptor, cfg,
                sigma2=float(snap.sigma2[k]), confidence_tier=snap.confidence_tier,
                credibility=fv.credibility, consensus_p=float(fv.consensus_p[k]),
                source_seq=self._tick, ts=now,
            )
            out.append((k, m, qs))
        return out

    def _route_fill(self, fill: Fill, now: datetime) -> None:
        self.inv.apply_fill(fill)
        ci = self.inv.snapshot(now).per_contract[fill.market_id]
        self.store.record_fill_and_update_inventory(fill, ci)

    # -- one tick ---------------------------------------------------------

    def tick(
        self,
        messages_by_market: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        *,
        feed_healthy: Optional[bool] = None,
        vol_gate_result: Optional[object] = None,
        manual_override: bool = False,
    ) -> None:
        messages_by_market = messages_by_market or {}
        self._tick += 1
        self.clock.set(self.clock.now() + timedelta(seconds=self.tick_dt_s))
        now = self.clock.now()

        # 1. book mirrors -> market states, liquidity update
        market_states = {}
        mids: Dict[float, float] = {}
        for m, k in self.markets:
            self._apply_messages(m, messages_by_market.get(m, []), now)
            ms = self.books[m].emit(m, self.expiry_key, k)
            if feed_healthy is not None:
                ms = replace(ms, feed_healthy=feed_healthy)
            market_states[m] = ms
            self.monitors[m].update(ms)
            mm = _market_mid(ms)
            if mm is not None:
                mids[k] = mm

        # 2. pricer snapshot (reuse previous on engine failure)
        hours = self._hours_to_expiry(now)
        try:
            snap = build_snapshot(
                self.strikes, self.expiry_key, hours,
                engine_fn=self.engine_fn, config=self.config, ts=now, now=now,
            )
            self.snapshot_failed = False
        except Exception:
            snap = self.last_snapshot
            self.snapshot_failed = True
            if snap is None:
                raise
        self.last_snapshot = snap

        # 3. fair value (thread bankroll state; reuse last consensus if the
        # market is momentarily incomplete, e.g. a feed gap)
        if len(mids) == len(self.markets):
            result = compute_fair_value(
                snap, mids, self.bankroll_state, self.config, market_ts=now,
                prev_forecasts=self.prev_forecasts, prev_consensus=self.prev_consensus, ts=now,
            )
            self.bankroll_state = result.bankroll_state
            self.prev_forecasts = result.forecasts
            self.prev_consensus = result.consensus_bucket
            self.last_fair_value = result.fair_value
            self.store.append_bankroll_state(self.expiry_key, self.bankroll_state)
        fv = self.last_fair_value
        if fv is None:
            return  # nothing quotable yet

        # refresh q_max off the fresh consensus x
        for m, k in self.markets:
            self.inv.update_fair_x(m, float(fv.consensus_x[k]))

        # 4. risk directives
        breaches = self._breaches()
        directives = {}
        for m, k in self.markets:
            liq = self.monitors[m].emit()
            self.last_liquidity[m] = liq
            directive = self.risk.evaluate(
                m, now, tte_days=snap.tte_days, pricer_snapshot=snap,
                inventory_breaches=breaches, liquidity_regime=liq.regime,
                feed_healthy=market_states[m].feed_healthy,
                spot=snap.s0, strike=k, manual_override=manual_override,
                vol_gate_result=vol_gate_result,
            )
            directives[m] = directive
            self.store.append_risk_directive(directive)
        self.last_directives = directives

        # 5. quotes -> sizing -> spread -> no-arb -> lifecycle
        composed = self._compose_quote_sets(snap, fv, directives)
        composed.sort(key=lambda t: t[0])
        strikes_sorted = [k for k, _, _ in composed]
        qs_list = [qs for _, _, qs in composed]
        self.last_quote_sets = {m: qs for _, m, qs in composed}
        model_cdf = {k: float(fv.consensus_p[k]) for k in strikes_sorted}

        checked = self.hedger.repair(qs_list, strikes_sorted, model_cdf, expiry_key=self.expiry_key)
        self.last_checked_quote_sets = checked
        if checked is not None:
            self.checked_ladders.append((strikes_sorted, checked))
            self.all_checked_quote_sets.extend(checked)
            if self.journal_maxlen is not None:
                # Trim from the front -- keep the newest entries. Both stay
                # plain lists (not deques) so callers can still index/slice
                # them (plan B4-memory 1.2); everything trimmed here is
                # already durably persisted via store.append_quote below.
                if len(self.checked_ladders) > self.journal_maxlen:
                    del self.checked_ladders[: len(self.checked_ladders) - self.journal_maxlen]
                if len(self.all_checked_quote_sets) > self.journal_maxlen:
                    del self.all_checked_quote_sets[: len(self.all_checked_quote_sets) - self.journal_maxlen]
            for (k, m, _), qs in zip(composed, checked):
                p = self.last_proposals[m]
                self.store.append_quote(
                    qs, p.r_x, p.delta_x, p.skew_x, p.sigma_b, p.params_id,
                    p.x_bid, p.x_ask, p.p_bid_raw, p.p_ask_raw,
                )
                self.lifecycle.apply(m, qs, directives[m])

        # 6. fill simulator -> route fills -> reconcile filled orders
        fills_this_tick: List = []
        for m, _ in self.markets:
            for f in self.fill_sim.on_market_state(market_states[m]):
                self._route_fill(f, now)
                fills_this_tick.append(f)
        self.last_fills = fills_this_tick

        sim_open_ids = {o["order_id"] for o in self.fill_sim.open_orders()}
        for f in fills_this_tick:
            if f.order_id in sim_open_ids:
                continue
            rec = self.store.get_order(f.order_id)
            if rec is not None and rec.status in _LIVE_STATUSES:
                self.store.upsert_order(
                    rec.client_order_id, rec.market_id, rec.side, rec.price, rec.size,
                    "FILLED", venue_order_id=rec.venue_order_id,
                    ts_placed=rec.ts_placed, ts_final=now,
                )

    # -- invariant helper -------------------------------------------------

    def fold_matches_inventory(self) -> bool:
        """The 8.2 standing invariant: fold(fills).q == InventoryManager.q per
        market (q_max/age are not fill-derived, so only q is compared)."""
        folded = self.store.fold_fills_to_inventory()
        snap = self.inv.snapshot(self.clock.now())
        keys = set(folded) | set(snap.per_contract)
        for k in keys:
            fq = folded[k].q if k in folded else 0.0
            iq = snap.per_contract[k].q if k in snap.per_contract else 0.0
            if abs(fq - iq) > 1e-9:
                return False
        return True

    # -- settlement -------------------------------------------------------

    def settle(self, now: Optional[datetime] = None, *, catch_up: bool = False):
        """Settle this expiry (or run the startup catch-up scan). Syncs the
        settlement pseudo-fills back into the in-memory InventoryManager so the
        fold invariant survives resolution."""
        now = now or self.clock.now()
        if catch_up:
            # Registry-merge (plan B3 / 1.4): the persisted registry (which
            # may still carry a PREVIOUS run's markets, e.g. a rolled-over
            # event) is merged UNDER this run's current-ladder markets, so a
            # stale persisted (expiry_key, strike) for a market_id this run
            # also owns can never shadow the current, authoritative values.
            registry = {
                **self.store.get_market_registry(),
                **{m: (self.expiry_key, k) for m, k in self.markets},
            }
            result = self.settlement.catch_up(now, registry)
        else:
            positions: List[MarketPosition] = []
            snap = self.inv.snapshot(now)
            for m, k in self.markets:
                stored = self.store.get_inventory(m)
                if stored is not None:
                    q, avg_cost = stored.q, stored.avg_cost
                else:
                    ci = snap.per_contract.get(m)
                    q, avg_cost = (ci.q, ci.avg_cost) if ci is not None else (0.0, 0.0)
                positions.append(MarketPosition(market_id=m, strike=k, q=q, avg_cost=avg_cost))
            result = self.settlement.settle_expiry(self.expiry_key, positions, now)

        # Sync in-memory inventory with the store's SETTLEMENT pseudo-fills.
        # UNFILTERED by design (plan 1.4, reviewer round-2 finding: an
        # earlier proposed current-ladder filter here was inverted on the
        # real resume path). Correctness argument:
        #   - The resume path (WS2.1) ALWAYS runs `loop.restart()` BEFORE
        #     `settle(catch_up=True)`. `restart()` replays the ENTIRE fills
        #     table (`store.get_fills()` is unfiltered), so a PREVIOUS-event
        #     position enters in-memory inventory at its true q (auto-created
        #     by `InventoryManager.apply_fill`). This unfiltered sync then
        #     applies the closing SETTLEMENT pseudo-fill and drives it to 0,
        #     matching `fold_fills_to_inventory` (which folds SETTLEMENT
        #     fills too, state_store.py's `fold_fills_to_inventory`) -- so
        #     `fold_matches_inventory()` holds for the previous event too.
        #   - Clean-rollover path (prior run exited 42 after settling): the
        #     prior market is already terminal, so `catch_up` finds it
        #     terminal and emits no event for it -- this sync is inert for
        #     that market.
        #   - Fresh-DB path: no prior fills exist, the registry contains
        #     only this run's current markets -- this sync is inert.
        # INVARIANT: `settle(catch_up=True)` must only be called after
        # `restart()` on a RESUMED store (the WS2.1 sequence guarantees
        # this). Calling it on a fresh `InventoryManager` (no prior
        # `restart()`) with a non-empty `fills` table would desync
        # in-memory inventory from the store, since the previous event's
        # opening fills would never have been replayed in.
        for ev in result.events:
            if ev.outcome.value in ("YES", "NO") and ev.q_settled != 0.0:
                closing_side = Side.BUY_NO if ev.q_settled > 0.0 else Side.BUY_YES
                payoff_yes = 1.0 if ev.outcome.value == "YES" else 0.0
                f = Fill(
                    ts=now, market_id=ev.market_id,
                    order_id="settlement:%s:%s" % (ev.market_id, self.expiry_key),
                    side=closing_side, price=payoff_yes, size=abs(ev.q_settled),
                    liquidity=LiquiditySource.SETTLEMENT, venue_ts=now,
                )
                self.inv.apply_fill(f)
        return result

    # -- restart protocol -------------------------------------------------

    def restart(self, now: Optional[datetime] = None):
        """Restart protocol (plan Section 5): mark LIVE orders UNKNOWN, rebuild
        the in-memory inventory by replaying the persisted fills, reconcile
        against the venue (cancelling unknowns + orphans), and reload the Beuoy
        bankroll state. Returns the ReconciliationResult."""
        now = now or self.clock.now()
        self.store.mark_all_live_orders_unknown()

        self.inv = InventoryManager(self.config)
        for m, k in self.markets:
            self.inv.register_market(m, self.expiry_key, k)
            self.inv.update_fair_x(m, 0.0)
        for f in self.store.get_fills():
            self.inv.apply_fill(f)

        recon = self.lifecycle.restart_reconcile()

        loaded = self.store.get_latest_bankroll_state(self.expiry_key)
        if loaded is not None:
            self.bankroll_state = loaded
        return recon
