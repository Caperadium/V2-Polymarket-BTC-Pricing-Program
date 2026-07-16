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
     (plan Wave 1 W1.2/W1.3: only on a real recompute -- tracked via
     `_fv_recomputed_this_tick` -- does the sigma_b x-history get a new
     sample, the consecutive-clean-BEUOY streak advance, and
     `_fv_recomputed_ts` update; a frozen/reused consensus tick skips all
     three)
  5. risk directives per market (inventory breaches, liquidity regime, feed
     health, pricer staleness, fair-value staleness)
  6. quote proposals -> joint ladder sizing (DEPTH-capped by this tick's
     `last_liquidity`, plan Wave 1 W1.1) -> spread builder -> MANDATORY
     ladder-hedger no-arb check/repair -> W2.2/W2.2b size-skew (inflates the
     hedge side of a neighbor strike per the PREVIOUS tick's
     `_pending_hedge_recs`, price-capped by `max_price`, never resurrecting a
     suppressed side; a rejected/failed ladder never reaches the lifecycle,
     so the skew never runs on one) -> order lifecycle over the
     PaperVenueAdapter
  7. feed the same MarketState to the fill simulator; route fills atomically to
     the InventoryManager AND the state store
     (record_fill_and_update_inventory); reconcile fully-consumed store orders
  8. W2.1 ladder-hedger stage: compute NEXT tick's vertical (and, behind
     `enable_beta_hedge`, beta) hedge recommendations off this tick's
     post-fill inventory; rebuild the market_id-keyed pending-hedge-demand
     offsets (`hedge_offsets_by_market`, plan W2.0) from scratch and push them
     into `inv.set_hedge_state`; wire this tick's joint-ladder phi into
     `inv.set_phi`.

`settle()` delegates to the SettlementHandler (with catch_up support) and syncs
the settlement pseudo-fills back into the in-memory InventoryManager so the
fold(fills) == inventory invariant survives resolution.

This file doubles as the Stage-A shadow-loop skeleton, so every stage journals
its output on the instance (last_snapshot, last_fair_value, last_directives,
last_quote_sets, last_proposals, last_fills, checked_ladders, ...).
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from market_maker.config import MMConfig, in_belly_band
from market_maker.contracts import (
    AnchorMethod,
    BankrollState,
    ContractInv,
    Fill,
    HedgeRecommendation,
    LiquidityRegime,
    LiquiditySource,
    LiquidityState,
    QuoteMode,
    QuoteSet,
    Side,
    VenueDescriptor,
)
from market_maker.fair_value_anchor import (
    BELLY_REGION,
    MARKET_MODEL_ID,
    PRICER_MODEL_ID,
    REGIONS,
    WING_REGION,
    compute_fair_value,
)
from market_maker.inventory_manager import InventoryManager
from market_maker.ladder_hedger import LadderHedger, hedge_offsets_by_market
from market_maker.liquidity_monitor import LiquidityMonitor
from market_maker.market_data_client import BookMirror, FeedCapability
from market_maker.order_lifecycle import OrderLifecycleManager, PaperVenueAdapter, SimClock
from market_maker.paper_fill_sim import PaperFillSimulator
from market_maker.pnl_report import markout_stats, markout_stats_side, tte_bucket_label
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
from market_maker.spread_builder import build_quote_set, compute_posted_prices, markout_widen
from market_maker.state_store import MMStateStore

logger = logging.getLogger("mm.harness")

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


def _resume_bankroll_states(
    store: MMStateStore, expiry_key: str, now: datetime,
) -> Dict[str, BankrollState]:
    """Resume/seed policy (package B2 plan step 10): per-region rows present
    -> load each; ONLY a legacy (region='') row present -- this deploy's
    first restart -- belly INHERITS the legacy bankrolls (continuity: belly
    credibility was legitimately earned), wing RESETS to 0.5/0.5 (the
    measured wing bias says wing authority was NOT legitimately earned;
    parity is the honest prior) -- USER-CONFIRMED DECISION at review.
    Neither present -> fresh 0.5/0.5 both regions (matches
    _initial_bankroll_state, the __init__ default)."""
    belly = store.get_latest_bankroll_state(expiry_key, region=BELLY_REGION)
    wing = store.get_latest_bankroll_state(expiry_key, region=WING_REGION)
    if belly is not None or wing is not None:
        return {
            BELLY_REGION: belly if belly is not None else _initial_bankroll_state(now),
            WING_REGION: wing if wing is not None else _initial_bankroll_state(now),
        }
    legacy = store.get_latest_bankroll_state(expiry_key, region="")
    if legacy is not None:
        return {
            BELLY_REGION: legacy,
            WING_REGION: _initial_bankroll_state(now),
        }
    return {
        BELLY_REGION: _initial_bankroll_state(now),
        WING_REGION: _initial_bankroll_state(now),
    }


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
        markout_provider: Optional[Callable[[], Optional[dict]]] = None,
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
        # wave 2 W7: optional callable returning the latest markout_report()
        # dict (or None -- cold start / never wired). Called ONCE per tick by
        # _compose_quote_sets, never per-market.
        self.markout_provider = markout_provider
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
        self.lifecycle = OrderLifecycleManager(
            self.venue, store, self.config, self.clock,
            min_order_size=self.venue_descriptor.min_size,
        )
        self.risk = RiskController(self.config, vol_gate_fn=vol_gate_fn)
        self.settlement = SettlementHandler(store, self.config, data_provider)

        self.books: Dict[str, BookMirror] = {
            m: BookMirror(capability=feed_capability, config=self.config, clock=self.clock.now)
            for m, _ in self.markets
        }
        self.monitors: Dict[str, LiquidityMonitor] = {
            m: LiquidityMonitor(self.config, tick_size=tick) for m, _ in self.markets
        }

        # threaded fair-value state (package B2, 2026-07-15): TWO independent
        # Beuoy bankroll states, keyed BELLY_REGION/WING_REGION -- this is
        # the real state; `self.bankroll_state` (singular) below is a
        # read-only synthesized legacy view.
        self.bankroll_states: Dict[str, BankrollState] = {
            BELLY_REGION: _initial_bankroll_state(self.clock.now()),
            WING_REGION: _initial_bankroll_state(self.clock.now()),
        }
        self.prev_forecasts: Optional[Dict[str, np.ndarray]] = None
        self.prev_consensus: Optional[np.ndarray] = None
        self._x_hist: Dict[str, List[float]] = {m: [] for m, _ in self.markets}
        self._seq: Dict[str, int] = {m: 0 for m, _ in self.markets}

        # W1.2: last successful consensus recompute time + per-tick recompute
        # flag (True only on the `len(mids) == len(self.markets)` branch).
        self._fv_recomputed_ts: Optional[datetime] = None
        self._fv_recomputed_this_tick: bool = False

        # W1.3: consecutive-clean-BEUOY-tick streak for bankroll auto-unfreeze.
        self._clean_beuoy_streak: int = 0

        # W2.1: this tick's joint-ladder phi directive (from SizingDecision),
        # cached in _compose_quote_sets and consumed by the hedge stage's
        # inv.set_phi call. 0.0 until the first tick sizes anything.
        self._last_phi_directive: float = 0.0

        # W2.1 (plan reviewer note 14): defined empty so tick 1 and the
        # `fv is None` early-return have a value.
        self._pending_hedge_recs: List[HedgeRecommendation] = []

        # W2.1/W2.4: market_id-keyed pending-hedge-demand offsets (W2.0
        # builder output), exposed on the loop so paper_runner.py can persist
        # them via store.upsert_ladder_state at the snapshot cadence.
        self.last_hedge_offsets: Dict[str, float] = {}

        # W2.2: bounded journal of hedge-skew apply/skip decisions -- same
        # trim pattern as checked_ladders (journal_maxlen), everything
        # trimmed here already reached its terminal state (applied or
        # skipped) and is not separately persisted.
        self.hedge_journal: List[dict] = []

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

    # -- legacy read-only view (package B2) --------------------------------

    @property
    def bankroll_state(self) -> BankrollState:
        """Read-only legacy single-scalar view synthesized from the two
        region states (package B2, 2026-07-15): frozen = OR over regions
        (either region degenerate freezes the legacy view); bankrolls =
        unweighted mean per model across belly/wing; model_ids/last_update/
        update_count taken from belly (an arbitrary but consistent single-
        region pick -- belly and wing update in lockstep on freeze/unfreeze
        events, so these rarely diverge in practice). External readers
        (paper_runner.py's heartbeat, historical tests) keep working
        unchanged. Read-only BY DESIGN: assigning `loop.bankroll_state = ...`
        or mutating an attribute on the returned (freshly-built) object is a
        no-op on the real state -- internal code must operate on
        `self.bankroll_states` instead."""
        belly = self.bankroll_states[BELLY_REGION]
        wing = self.bankroll_states[WING_REGION]
        model_ids = list(belly.model_ids)
        bankrolls = {
            mid: 0.5 * (float(belly.bankrolls.get(mid, 0.0)) + float(wing.bankrolls.get(mid, 0.0)))
            for mid in model_ids
        }
        return BankrollState(
            model_ids=model_ids,
            bankrolls=bankrolls,
            last_update=belly.last_update,
            update_count=belly.update_count,
            frozen=bool(belly.frozen or wing.frozen),
        )

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

    def _breaches(self, inv_snap, fv, bankroll: float) -> List[InvBreach]:
        """Inventory-cap breaches for the current tick, risk-based (package D,
        2026-07-15). `inv_snap` is the single per-tick inventory snapshot
        (hoisted by the caller so it is taken once, not once per call --
        stranded-inventory fix 2026-07-14, Change B). `fv` is this tick's
        FairValue (consensus_p, keyed by strike) and `bankroll` is the loop's
        sizing bankroll.

        Metric = remaining-loss notional, not raw share ratio: the old
        |q| / q_max rule punished wings hardest exactly where remaining
        per-share risk is smallest (S'(x)-shrinking q_max), per the
        2026-07-14 stranded-inventory decision (deferred "option C").
        L_m = q * p_consensus (long YES: marks to 0 on a NO outcome) or
        |q| * (1 - p_consensus) (short YES: marks to 1 on a YES outcome).
        NOTE: consensus p is measurably rich in the OTM (section-0 FV-vs-mkt
        audit), so wing L_m is slightly OVER-stated here -- conservative,
        accepted.

        `is_long` is derived from the RAW signed q (no hedge adjustment --
        mandatory descope, phase 1): rules (c) and (f) in RiskController
        must keep deriving their one-sided direction from the same signed
        quantity, or `_more_restrictive` escalates opposite one-sided modes
        to PULLED and re-introduces the 2026-07-14 stranding bug. Hedge-aware
        q_eff is deferred to a follow-up (package D phase 2)."""
        cap = float(self.config.inv_loss_cap_frac) * float(bankroll)
        if cap <= 0.0:
            return []
        breaches: List[InvBreach] = []
        for m, k in self.markets:
            ci = inv_snap.per_contract.get(m)
            if ci is None:
                continue
            p_consensus = float(fv.consensus_p[k])
            if ci.q > 0.0:
                loss = ci.q * p_consensus
            else:
                loss = abs(ci.q) * (1.0 - p_consensus)
            ratio = loss / cap
            if ratio >= 1.0:
                breaches.append(InvBreach(market_id=m, is_long=ci.q > 0.0, ratio=ratio))
        return breaches

    def _compose_quote_sets(
        self, snap, fv, directives,
        liquidity: Optional[Dict[str, LiquidityState]] = None,
        market_states: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[float, str, QuoteSet]]:
        """Quote engine -> posted prices -> sizing -> QuoteSet for the whole
        ladder. A seam the forced-no-arb test monkeypatches to inject a
        crossing ladder.

        `liquidity` (W1.1): dict market_id -> LiquidityState, forwarded to
        `size_ladder(liquidity=...)` so the DEPTH cap actually binds; the
        tick passes `self.last_liquidity` (populated earlier in the same
        tick's risk-directive loop, so it is fresh). None (default) keeps
        the DEPTH cap inert, matching prior behavior.

        `market_states` (plan C1): dict market_id -> MarketState, the same
        dict `tick()` builds in step 1. No longer used to compute a sizing
        mid (wave 2 W2 removed ContractSizingInput.mkt_mid -- sizing now
        edges off our OWN posted quote, not the market mid); kept as a
        parameter for signature stability in case a future caller needs it.

        Wave 2 W1/W7 ordering (package E extends this, 2026-07-15): for each
        market, build the proposal, resolve `region` from `consensus_p_k`,
        then resolve term 7's per-side markout widening (`markout_stats_side`
        at `cfg.markout_widen_horizon_s` -> `spread_builder.markout_widen`),
        then `compute_posted_prices` (the seven-term spread builder, computed
        ONCE, now including that widening), then feed the posted bid/ask into
        `ContractSizingInput` (both as the capital-at-risk price basis and as
        the Kelly edge price) plus this tick's resolved AGGREGATE markout
        stats (W6, via `markout_stats` at `cfg.markout_horizon_s` -- a
        deliberately different horizon and a deliberately different
        aggregate-vs-side-split basis than term 7, see spread_builder module
        docstring "Deliberate basis inconsistencies"), THEN size the ladder,
        THEN `build_quote_set(..., posted=...)` so the QuoteSet's prices are
        exactly the ones sizing used -- never recomputed."""
        cfg = self.config
        now = self.clock.now()
        tte = max(snap.tte_days, 0.0)

        # W7: pull the latest markout report ONCE per tick (not per market).
        # A cold/unwired provider (None) or an empty report (None return)
        # both degrade every market to the m_prior path in robustness_sizing.
        report = self.markout_provider() if self.markout_provider is not None else None
        tte_bucket = tte_bucket_label(tte)

        proposals = {}
        posted_by_market: Dict[str, Tuple[float, float, Dict[str, float]]] = {}
        contracts: List[ContractSizingInput] = []
        # sigma_b sampling stride: estimate on consensus-x subsampled to
        # >= cfg.sigma_b_sample_s so tick-frequency microstructure noise
        # (mid jitter, spread bounce) is not annualized into belief vol
        # (Stage-A shadow finding 2026-07-07; plan 10.2).
        sample_s = float(getattr(cfg, "sigma_b_sample_s", 0.0) or 0.0)
        stride = max(1, int(round(sample_s / self.tick_dt_s))) if sample_s > 0 else 1
        sample_dt_days = (self.tick_dt_s * stride) / 86400.0

        inventory = self.inv.snapshot(now)
        for m, k in self.markets:
            # W1.2: only feed the sigma_b estimator on ticks where consensus
            # was actually recomputed -- a frozen anchor (incomplete-mids
            # reuse) must not decay sigma_b toward the floor via a
            # repeated-value append.
            if self._fv_recomputed_this_tick:
                self._x_hist[m].append(float(fv.consensus_x[k]))
            if self.x_hist_maxlen is not None and len(self._x_hist[m]) > self.x_hist_maxlen:
                # Trim from the front -- keep the newest entries (the sigma_b
                # estimator below is itself newest-anchored, plan B4-memory).
                del self._x_hist[m][: len(self._x_hist[m]) - self.x_hist_maxlen]
            hist = self._x_hist[m][::-1][::stride][::-1]  # newest-anchored subsample
            sigma_b = estimate_sigma_b(
                hist, sample_dt_days, cfg.sigma_b_floor, cfg.sigma_b_cap
            )
            q = inventory.per_contract[m].q
            prop = make_quote_from_config(
                cfg, m, float(fv.consensus_x[k]), q, tte, sigma_b, variant=self.quote_variant, ts=now
            )
            proposals[m] = prop

            consensus_p_k = float(fv.consensus_p[k])
            # region moved above compute_posted_prices (package E, round-2
            # review item 4): consensus_p_k is already available here, and
            # term 7's markout_stats_side lookups (below) need it BEFORE the
            # single compute_posted_prices call, not after.
            region = "belly" if in_belly_band(consensus_p_k, cfg.belly_band) else "wing"

            # term 7 (package E): markout-fed widening, resolved per side off
            # the SIDE-SPLIT markout report (BUY_YES -> bid, BUY_NO -> ask) at
            # cfg.markout_widen_horizon_s (60s, deliberately different from
            # sizing's cfg.markout_horizon_s below). A cold/unwired provider
            # (report is None) degrades both sides to 0.0 widening, same as
            # the sizing markout fields below.
            if report is not None:
                mk_avg_bid_side, _mk_n_bid_side = markout_stats_side(
                    report, region, tte_bucket, cfg.markout_widen_horizon_s,
                    Side.BUY_YES, cfg.markout_min_n,
                )
                mk_avg_ask_side, _mk_n_ask_side = markout_stats_side(
                    report, region, tte_bucket, cfg.markout_widen_horizon_s,
                    Side.BUY_NO, cfg.markout_min_n,
                )
            else:
                mk_avg_bid_side, mk_avg_ask_side = None, None
            markout_widen_bid = markout_widen(mk_avg_bid_side, cfg.markout_widen_scale, cfg.markout_widen_cap)
            markout_widen_ask = markout_widen(mk_avg_ask_side, cfg.markout_widen_scale, cfg.markout_widen_cap)

            # Package B2 (2026-07-15): region-appropriate credibility into
            # compute_posted_prices ONLY (build_quote_set's credibility arg
            # is moot when posted= is threaded, plan review item 13). Region
            # basis is the SAME `region` variable computed just above from
            # consensus_p_k (round-2 review item 2) -- NOT the anchor's
            # market-mid region map (a deliberate, documented basis
            # inconsistency, see fair_value_anchor / spread_builder docs).
            region_credibility = (
                fv.credibility_by_region[region]
                if fv.credibility_by_region is not None else fv.credibility
            )

            posted = compute_posted_prices(
                prop, directives[m], self.venue_descriptor, cfg,
                sigma2=float(snap.sigma2[k]), confidence_tier=snap.confidence_tier,
                credibility=region_credibility, consensus_p=consensus_p_k, tte_days=tte,
                markout_widen_bid=markout_widen_bid, markout_widen_ask=markout_widen_ask,
            )
            posted_by_market[m] = posted
            posted_bid, posted_ask, _terms = posted

            if report is not None:
                mk_avg, mk_var, mk_n, mk_n_attempted = markout_stats(
                    report, region, tte_bucket, cfg.markout_horizon_s, cfg.markout_min_n
                )
            else:
                mk_avg, mk_var, mk_n, mk_n_attempted = None, None, 0, 0

            contracts.append(
                ContractSizingInput(market_id=m, p_hat=float(fv.consensus_p[k]),
                                    bid_price=posted_bid, ask_price=posted_ask,
                                    strike=float(k), mk_avg=mk_avg, mk_var=mk_var,
                                    mk_n=mk_n, mk_n_attempted=mk_n_attempted)
            )
        self.last_proposals = proposals

        decisions, _audit = size_ladder(
            contracts, snap, self.bankroll, now, cfg, liquidity=liquidity, inventory=inventory
        )

        # W2.1: phi_directive is the same joint-ladder value on every
        # SizingDecision this tick (plan: "any decision carries it") -- cache
        # it on the instance so the tick's hedge stage can wire it into
        # inv.set_phi without re-deriving it.
        if decisions:
            self._last_phi_directive = next(iter(decisions.values())).phi_directive

        out: List[Tuple[float, str, QuoteSet]] = []
        for m, k in self.markets:
            qs = build_quote_set(
                proposals[m], directives[m], decisions[m], self.venue_descriptor, cfg,
                sigma2=float(snap.sigma2[k]), confidence_tier=snap.confidence_tier,
                credibility=fv.credibility, consensus_p=float(fv.consensus_p[k]),
                source_seq=self._tick, ts=now, tte_days=tte,
                posted=posted_by_market[m],
            )
            out.append((k, m, qs))
        return out

    def _apply_hedge_skew(self, checked: List[QuoteSet], now: datetime) -> List[QuoteSet]:
        """W2.2/W2.2b: inflate this tick's just-repaired ladder's passive
        quote size on the hedge side of each PREVIOUS-tick hedge
        recommendation's target market ("size-skew" hedge execution,
        plan Wave 2 decision table). Pure w.r.t. `checked` (returns a new
        list; QuoteSet is frozen) but appends to the bounded
        `self.hedge_journal`.

        Suppressed-side precedence (plan W2.2, NEVER resurrect a suppressed
        side): a rec is skipped if the target QuoteSet's directive mode
        forecloses the hedge side (BUY_YES needs bid allowed --
        TWO_SIDED/BID_ONLY; BUY_NO needs ask allowed -- TWO_SIDED/ASK_ONLY;
        PULLED always skips) OR the checked ladder already sized that side to
        0 (a suppressed/zeroed side is never resurrected by the hedge skew).

        W2.2b price rule (side-scale, exhaustive):
          BUY_YES: apply iff qs.bid_price <= rec.max_price (both YES-scale).
          BUY_NO:  the placed NO price is (1 - qs.ask_price)
                   (order_lifecycle.py's sell-YES-via-buy-NO convention);
                   apply iff (1 - qs.ask_price) <= rec.max_price (both
                   NO-scale).
        """
        by_market = {qs.market_id: qs for qs in checked}
        out = list(checked)
        for rec in self._pending_hedge_recs:
            qs = by_market.get(rec.target_market_id)
            if qs is None:
                self.hedge_journal.append({
                    "ts": now, "target_market_id": rec.target_market_id,
                    "side": rec.side.value, "size": rec.size,
                    "applied": False, "reason": "target_not_in_ladder",
                })
                continue

            if rec.side == Side.BUY_YES:
                side_allowed = qs.risk_mode in (QuoteMode.TWO_SIDED, QuoteMode.BID_ONLY)
                side_zeroed = qs.bid_size <= 0.0
            else:
                side_allowed = qs.risk_mode in (QuoteMode.TWO_SIDED, QuoteMode.ASK_ONLY)
                side_zeroed = qs.ask_size <= 0.0

            if qs.risk_mode == QuoteMode.PULLED or not side_allowed or side_zeroed:
                self.hedge_journal.append({
                    "ts": now, "target_market_id": rec.target_market_id,
                    "side": rec.side.value, "size": rec.size,
                    "applied": False, "reason": "suppressed_side",
                })
                continue

            if rec.side == Side.BUY_YES:
                applies = qs.bid_price <= rec.max_price
            else:
                applies = (1.0 - qs.ask_price) <= rec.max_price

            if not applies:
                self.hedge_journal.append({
                    "ts": now, "target_market_id": rec.target_market_id,
                    "side": rec.side.value, "size": rec.size,
                    "applied": False, "reason": "price_above_max",
                })
                continue

            if rec.side == Side.BUY_YES:
                new_qs = replace(qs, bid_size=qs.bid_size + rec.size)
            else:
                new_qs = replace(qs, ask_size=qs.ask_size + rec.size)
            by_market[rec.target_market_id] = new_qs
            self.hedge_journal.append({
                "ts": now, "target_market_id": rec.target_market_id,
                "side": rec.side.value, "size": rec.size,
                "applied": True, "reason": "ok",
            })

        if self.journal_maxlen is not None and len(self.hedge_journal) > self.journal_maxlen:
            del self.hedge_journal[: len(self.hedge_journal) - self.journal_maxlen]

        return [by_market[qs.market_id] for qs in out]

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

        # W0.3: accrue age_weighted_holding / R3 between fills -- a held
        # position must age even on a tick with no fill activity, so gate
        # metrics that read it live stay current (plan Wave 0).
        self.inv.mark(now)

        # 1. book mirrors -> market states, liquidity update
        market_states = {}
        mids: Dict[float, float] = {}
        mids_by_market: Dict[str, float] = {}
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
                mids_by_market[m] = mm

        # C2 (mm_suitability_alignment_plan.md Change C): durably log this
        # tick's per-market mids for the markout report BEFORE the
        # `if fv is None: return` early-out below, so warmup ticks (no full
        # book yet) still get their partial mids logged. `mids` above is
        # STRIKE-keyed (consumed by fair-value/quote code below);
        # `mids_by_market` is the separate market_id-keyed view the store
        # needs.
        if mids_by_market:
            self.store.append_mids(now, mids_by_market)

        # 2026-07-11: durably record this tick's drained trade prints so
        # scripts/mm_calibrate_k.py can fit the arrival decay k from print
        # distance-to-mid (no fill data needed). last_prints was already
        # drained into the MarketState by the feed snapshot; the fill sim
        # below reads the same list, this only copies it to the store.
        prints_by_market = {
            m: ms.last_prints for m, ms in market_states.items() if ms.last_prints
        }
        if prints_by_market:
            self.store.append_trade_prints(prints_by_market)

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
        # W1.2/W1.3: recomputed-this-tick flag -- True only on this branch
        # (a real consensus recompute), never on the reuse/incomplete-mids
        # path below. Downstream (sigma_b append, unfreeze streak) key off
        # this flag, not off fv.anchor_method alone (reviewer note 12):
        # last_fair_value retains a stale BEUOY method on non-recompute ticks.
        self._fv_recomputed_this_tick = len(mids) == len(self.markets)
        if self._fv_recomputed_this_tick:
            result = compute_fair_value(
                snap, mids, self.bankroll_states, self.config, market_ts=now,
                prev_forecasts=self.prev_forecasts, prev_consensus=self.prev_consensus, ts=now,
            )
            self.bankroll_states = result.bankroll_states
            self.prev_forecasts = result.forecasts
            self.prev_consensus = result.consensus_bucket
            self.last_fair_value = result.fair_value
            self._fv_recomputed_ts = now
            for region in REGIONS:
                self.store.append_bankroll_state(
                    self.expiry_key, self.bankroll_states[region], region=region,
                )

            # W1.3: bankroll auto-unfreeze streak. Resets to 0 on any
            # recomputed tick whose anchor is non-BEUOY (fallback fired);
            # non-recompute ticks (this branch not taken) neither increment
            # nor reset -- see below, after the early return.
            if result.fair_value.anchor_method == AnchorMethod.BEUOY:
                self._clean_beuoy_streak += 1
            else:
                self._clean_beuoy_streak = 0

            # bankroll_state (property, OR over regions): unfreeze streak
            # clears BOTH region states in lockstep (plan step 8 -- a
            # fallback is a whole-ladder event, so its recovery is too).
            if (self.bankroll_state.frozen
                    and self._clean_beuoy_streak >= self.config.bankroll_unfreeze_clean_ticks):
                for region in REGIONS:
                    self.bankroll_states[region].frozen = False
                logger.warning(
                    "bankroll auto-unfrozen after %d consecutive clean BEUOY ticks (expiry %s)",
                    self._clean_beuoy_streak, self.expiry_key,
                )
                for region in REGIONS:
                    self.store.append_bankroll_state(
                        self.expiry_key, self.bankroll_states[region], region=region,
                    )
        fv = self.last_fair_value
        if fv is None:
            return  # nothing quotable yet

        # refresh q_max off the fresh consensus x
        for m, k in self.markets:
            self.inv.update_fair_x(m, float(fv.consensus_x[k]))

        # 4. risk directives -- one inventory snapshot per tick, reused by
        # _breaches() and the per-market inventory_q lookup below
        # (stranded-inventory fix 2026-07-14, Change B).
        inv_snap = self.inv.snapshot(now)
        breaches = self._breaches(inv_snap, fv, self.bankroll)
        q_by_market = {m: ci.q for m, ci in inv_snap.per_contract.items()}
        # W1.2: age of the last successful consensus recompute, fed to the
        # risk controller's fair-value staleness rule; None until the first
        # recompute ever happens (inert -- see RiskController.evaluate).
        fair_value_age_s = (
            (now - self._fv_recomputed_ts).total_seconds()
            if self._fv_recomputed_ts is not None else None
        )
        directives = {}
        for m, k in self.markets:
            liq = self.monitors[m].emit()
            self.last_liquidity[m] = liq
            directive = self.risk.evaluate(
                m, now, tte_days=snap.tte_days, pricer_snapshot=snap,
                inventory_breaches=breaches, inventory_q=q_by_market.get(m),
                liquidity_regime=liq.regime,
                feed_healthy=market_states[m].feed_healthy,
                spot=snap.s0, strike=k, manual_override=manual_override,
                vol_gate_result=vol_gate_result,
                fair_value_age_s=fair_value_age_s,
            )
            directives[m] = directive
            self.store.append_risk_directive(directive)
        self.last_directives = directives

        # 5. quotes -> sizing -> spread -> no-arb -> lifecycle
        # W1.1: self.last_liquidity is populated fresh THIS tick by the
        # risk-directive loop above (step 4 runs before this), so no re-emit
        # is needed here.
        composed = self._compose_quote_sets(
            snap, fv, directives, liquidity=self.last_liquidity, market_states=market_states
        )
        composed.sort(key=lambda t: t[0])
        strikes_sorted = [k for k, _, _ in composed]
        qs_list = [qs for _, _, qs in composed]
        model_cdf = {k: float(fv.consensus_p[k]) for k in strikes_sorted}

        checked = self.hedger.repair(qs_list, strikes_sorted, model_cdf, expiry_key=self.expiry_key)
        if checked is not None:
            # W2.2/W2.2b: apply the PREVIOUS tick's hedge recommendations as a
            # size-skew on this tick's just-repaired ladder, BEFORE it is
            # journaled/persisted/sent to the lifecycle -- the replaced
            # QuoteSet must be the single object every downstream consumer
            # (append_quote, lifecycle.apply, last_quote_sets, checked_ladders,
            # all_checked_quote_sets) sees. repair() only ever touches prices,
            # never sizes, so no-arb stays valid after this skew too.
            checked = self._apply_hedge_skew(checked, now)
        self.last_checked_quote_sets = checked
        # last_quote_sets journals the FINAL (post-repair, post-skew) ladder
        # when repair succeeded; falls back to the pre-repair/pre-skew ladder
        # on a rejected ladder (checked is None) so callers still see
        # something rather than a stale prior-tick value.
        self.last_quote_sets = (
            {m: qs for _, m, qs in composed} if checked is None
            else {m: qs for (_, m, _), qs in zip(composed, checked)}
        )
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

        # 7. W2.1: ladder hedger -- compute NEXT tick's hedge inputs. Runs
        # AFTER fills routing (step 6) so inv_snapshot reflects this tick's
        # fills. market_id-keyed offsets (W2.0) are rebuilt from scratch every
        # tick from this tick's fresh recs -- see hedge_offsets_by_market's
        # docstring for the PENDING-hedge-demand / expires semantics.
        market_ids = [m for m, _ in self.markets]
        inv_snapshot = self.inv.snapshot(now)
        fair_p = {m: float(fv.consensus_p[k]) for m, k in self.markets}
        # depth_hint: bid+ask realized depth from this tick's last_liquidity
        # (step 4's fresh emit(), same source W1.1 wires into size_ladder);
        # guard missing entries rather than assume every market reported.
        depth_hint: Dict[str, float] = {}
        for m in market_ids:
            liq = self.last_liquidity.get(m)
            if liq is not None:
                depth_hint[m] = float(liq.realized_depth_bid) + float(liq.realized_depth_ask)

        recs, _audit_state = self.hedger.vertical_hedges(
            inv_snapshot, self.expiry_key, self.strikes, market_ids, fair_p,
            ts=now, depth_hint=depth_hint,
        )

        # W2.5: beta-hedge call site, behind enable_beta_hedge (default
        # False). Flag-off short-circuits BEFORE beta_hedges() is invoked at
        # all -- no offsets, no journal entries (asserted inert by test).
        # sigma_b plumbing deferred (reviewer note 13): enabling this flag for
        # real requires threading per-market sigma_b out of
        # _compose_quote_sets; the flag-on path here uses a placeholder
        # constant (sigma_b_floor) purely so the flag is functional rather
        # than dead code, per the plan's decision table.
        if self.hedger.enable_beta_hedge:
            sigma_b_placeholder = {m: self.config.sigma_b_floor for m in market_ids}
            beta_recs = self.hedger.beta_hedges(
                inv_snapshot, self.expiry_key, self.strikes, market_ids, fair_p,
                sigma_b_placeholder, now, depth_hint=depth_hint,
            )
            recs = recs + beta_recs

        self._pending_hedge_recs = recs
        self.last_hedge_offsets = hedge_offsets_by_market(recs)
        self.inv.set_hedge_state(self.expiry_key, self.last_hedge_offsets)
        self.inv.set_phi(self.expiry_key, self._last_phi_directive)

    # -- invariant helper -------------------------------------------------

    def fold_matches_inventory(self, *, own_markets_only: bool = False) -> bool:
        """The 8.2 standing invariant: fold(fills).q == InventoryManager.q AND
        fold(fills).avg_cost == InventoryManager.avg_cost per market, each to
        1e-9 (q_max/age are not fill-derived, so only these two are
        compared). The avg_cost leg was added by the C0 fix (mm_suitability_
        alignment_plan.md pre-step C0): before it, state_store.
        fold_fills_to_inventory complemented BUY_NO cost-basis prices while
        InventoryManager did not, so the two could silently disagree on
        avg_cost while still agreeing on q -- this invariant would not have
        caught that class of bug.

        ``own_markets_only=True`` (multi-expiry): restrict the comparison to
        THIS loop's markets. Under the multi-expiry orchestrator the fills
        table (and its global fold) spans every expiry's loops, while this
        loop's InventoryManager holds only its own ladder (``resume_attach``
        replays filtered) -- the global-key comparison would spuriously fail
        on every foreign-expiry market. Default False preserves the legacy
        single-expiry check exactly."""
        folded = self.store.fold_fills_to_inventory()
        snap = self.inv.snapshot(self.clock.now())
        keys = set(folded) | set(snap.per_contract)
        if own_markets_only:
            keys = {k for k in keys if k in self.strike_by_mid}
        for k in keys:
            fq = folded[k].q if k in folded else 0.0
            iq = snap.per_contract[k].q if k in snap.per_contract else 0.0
            if abs(fq - iq) > 1e-9:
                return False
            f_avg = folded[k].avg_cost if k in folded else 0.0
            i_avg = snap.per_contract[k].avg_cost if k in snap.per_contract else 0.0
            if abs(f_avg - i_avg) > 1e-9:
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

    def resume_attach(self, now: Optional[datetime] = None,
                      all_fills: Optional[List[Fill]] = None) -> int:
        """Multi-expiry attach: rebuild the in-memory inventory from THIS
        loop's own fills only, and reload this expiry's Beuoy bankroll state.
        Returns the number of fills replayed.

        The multi-expiry replacement for the ladder-local part of
        ``restart()``: on a shared store, ``restart()``'s unfiltered
        ``get_fills()`` replay would pull every OTHER loop's fills into this
        loop's InventoryManager (``apply_fill`` auto-creates unregistered
        markets at expiry_key=None), and its ``mark_all_live_orders_unknown``
        + ``restart_reconcile`` are store-GLOBAL -- a mid-run-attached loop
        calling ``restart()`` would mark the other live loops' orders UNKNOWN
        and cancel them. So this method does NEITHER: no mark-unknown, no
        venue reconcile, no catch-up sync.

        Orchestrator ordering contract (cross-reference the invariant comment
        on ``settle(catch_up=True)`` below): the orchestrator runs its
        standalone store-wide settlement catch-up pass BEFORE calling this,
        so any previous-event position is already closed by SETTLEMENT
        pseudo-fills inside the fills table -- the filtered replay here then
        reproduces post-settlement inventory exactly, and per-loop
        ``fold_matches_inventory(own_markets_only=True)`` holds without ever
        running the unfiltered catch-up sync on this loop.

        ``all_fills`` lets the orchestrator fetch the (global) fills table
        once and share it across N loops; None fetches from the store.
        """
        now = now or self.clock.now()
        fills = self.store.get_fills() if all_fills is None else all_fills

        # Full re-registration, exactly as restart(): without register_market
        # + update_fair_x, q_max/fair_x/ladder membership are unset and the
        # first tick's sizing/breach logic is wrong.
        self.inv = InventoryManager(self.config)
        for m, k in self.markets:
            self.inv.register_market(m, self.expiry_key, k)
            self.inv.update_fair_x(m, 0.0)

        replayed = 0
        for f in fills:
            if f.market_id in self.strike_by_mid:
                self.inv.apply_fill(f)
                replayed += 1

        self.bankroll_states = _resume_bankroll_states(self.store, self.expiry_key, now)
        return replayed

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

        self.bankroll_states = _resume_bankroll_states(self.store, self.expiry_key, now)
        return recon
