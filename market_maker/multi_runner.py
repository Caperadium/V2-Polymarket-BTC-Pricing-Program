"""Multi-expiry orchestration for the Stage-B paper runner.

Architecture ("orchestrator of loops"): `PaperTradingLoop` (harness.py) stays
single-expiry and untouched; `MultiExpiryOrchestrator` owns one `LadderSlot`
per concurrent expiry -- each slot bundles its own PaperTradingLoop, its own
PolymarketFeedAdapter (one WS connection per ladder), its own SimClock (a
loop self-advances its clock inside tick(), so a shared clock would drift
N*tick_s per orchestrator tick), and an `ExpiryEngineView` over the shared
pricing engine. Shared across slots: ONE MMStateStore, ONE
`SharedPricingEngine` (one GARCH fit + one set of calibrated jump params,
per-expiry ladder caches), ONE vol-gate closure, ONE BTCDataProvider.

Reprice budget: `calculate_probabilities` blocks for minutes (GARCH fit +
MC), and `run_control._heartbeat_threshold` sizes the STALLED alarm as
`max(3*tick_s, reprice_s + 60)` -- i.e. it assumes AT MOST ONE engine call
per tick. The `SharedPricingEngine` therefore hands out a single reprice
token per orchestrator tick: the first DUE view (in the orchestrator's
round-robin-rotated slot order) recomputes; every other due view returns its
stale cached ladder and tries again next tick. A view with NO cache yet
(fresh slot) also needs the token; the orchestrator SKIPS such a slot's
tick entirely (drain-and-discard its feed messages) until its first-price
grant lands, so startup/acquisition of K ladders costs K warmup ticks of one
compute each, never K computes in one tick. A failed compute RETURNS the
token so a chronically-failing expiry cannot starve its siblings.

Rollover (in-process continuous): when a ladder is fully terminal (+30min
grace) or its per-ladder settlement-timeout clock expires, the slot is torn
down in place (final settle attempt, scoped per-market order cancels, ladder
state flush, adapter stop) and acquisition immediately probes for the next
event (`shadow_runner.resolve_events_multi`), excluding active and completed
expiries. The process itself exits only on: zero active ladders with empty
acquisition (`no_quotable_events` -> 42, systemd retries), feed death after
a per-slot adapter restart (1), the tick-error circuit breaker (1), or an
intentional stop (0). In fixed --event-slug mode there is no acquisition and
a terminal ladder requests the legacy `ladder_settled`/`settlement_timeout`
process exit (42) instead of a teardown.

Resume protocol on a shared multi-expiry db (see PaperTradingLoop.settle's
invariant comment): the orchestrator runs ONE standalone store-wide
settlement catch-up pass BEFORE any fills replay -- catch_up writes
SETTLEMENT pseudo-fills through the normal fills channel, so previous-event
positions net to zero inside the fills table itself, and each loop's
filtered `resume_attach` replay then reproduces post-settlement inventory
without ever running the unfiltered catch-up sync. One venue reconcile
(first slot's lifecycle) runs at process start only, when every fill sim is
empty; mid-run-acquired slots NEVER run restart()/reconcile (both are
store-global and would cancel the other live loops' orders).

The recurring `settlement_catchup_pass` (throttled, at most once per 60s,
and only while some non-slot registry market is past its settlement instant
and non-terminal) re-drives orphaned UNSETTLEABLE previous-event positions
mid-run -- the BTC bar covering the 12:00 ET instant lands from the 30-min
datafetch timer, and `settlement_retry_window_hours` is short enough that
waiting for the next teardown would routinely miss the window.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from market_maker.config import MMConfig
from market_maker.harness import PaperTradingLoop
from market_maker.market_data_client import FeedCapability
from market_maker.order_lifecycle import SimClock
from market_maker.settlement_handler import (
    BTCDataProvider,
    SettlementHandler,
    TERMINAL_OUTCOMES,
    settlement_instant_utc,
)
from market_maker.shadow_runner import load_jump_params_for_engine
from market_maker.state_store import MMStateStore

logger = logging.getLogger("mm.multi")

# Recurring settlement catch-up throttle (reviewer RISK-1 cadence bound).
CATCHUP_MIN_INTERVAL_S = 60.0

# Grace after the settlement instant before a fully-terminal ladder is
# declared done (same 30min the single-expiry runner used).
SETTLED_GRACE_MIN = 30.0


# ---------------------------------------------------------------------------
# Shared pricing engine with per-expiry views + one reprice token per tick
# ---------------------------------------------------------------------------


class SharedPricingEngine:
    """One GARCH fit + one set of calibrated jump params shared across all
    expiry views; each `ExpiryEngineView` keeps its own cached ladder.

    `calculate_probabilities(..., garch_cache=...)` is the engine's supported
    caller-owned fit-reuse hook (same pattern the backrunner uses to share
    one MLE fit across a snapshot's expiry groups), so strikes/hours can
    differ per view while the expensive fit runs once per refit window.

    `compute_fn` is an injectable seam for tests (defaults to the real
    `calculate_probabilities`, imported lazily like CachedEngine does). When
    a `compute_fn` is injected and no `jump_loader` is given, jump loading is
    disabled (tests must not touch DATA/ csvs by default); inject
    `jump_loader` explicitly to exercise the reload-on-refit path.

    All timing is WALL-CLOCK (`time.time()`), consistent with CachedEngine.
    """

    def __init__(
        self,
        reprice_s: float,
        seed: int = 42,
        garch_refit_s: float = 21_600.0,
        compute_fn: Optional[Callable[..., Dict[Any, Any]]] = None,
        jump_loader: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    ) -> None:
        self.reprice_s = reprice_s
        self.seed = seed
        self.garch_refit_s = garch_refit_s
        self._compute_fn = compute_fn
        if jump_loader is not None:
            self._jump_loader: Callable[[], Optional[Dict[str, Any]]] = jump_loader
        elif compute_fn is None:
            self._jump_loader = load_jump_params_for_engine
        else:
            self._jump_loader = lambda: None
        self._garch_cache: Dict[Any, Any] = {}
        self._garch_fitted_at: Optional[float] = None
        self._jump_params: Optional[Dict[str, Any]] = None
        self._views: Dict[str, "ExpiryEngineView"] = {}
        self._token_taken: bool = False
        self.latencies: List[float] = []

    # -- reprice token (one grant per orchestrator tick) -------------------

    def begin_tick(self) -> None:
        self._token_taken = False

    def token_available(self) -> bool:
        return not self._token_taken

    def try_take_reprice_token(self) -> bool:
        if self._token_taken:
            return False
        self._token_taken = True
        return True

    def return_reprice_token(self) -> None:
        """A failed compute must not spend the tick's grant (reviewer
        RISK-3): the next due view this tick still gets its reprice."""
        self._token_taken = False

    # -- views --------------------------------------------------------------

    def view(self, expiry_key: str) -> "ExpiryEngineView":
        v = self._views.get(expiry_key)
        if v is None:
            v = ExpiryEngineView(self, expiry_key)
            self._views[expiry_key] = v
        return v

    def drop_view(self, expiry_key: str) -> None:
        self._views.pop(expiry_key, None)

    # -- the shared compute (GARCH refit + jump reload inside the grant) ----

    def _compute(self, strikes, hours_to_expiry) -> Dict[Any, Any]:
        if self._compute_fn is None:
            from core.pricing.btc_pricing_engine import calculate_probabilities
            compute = calculate_probabilities
        else:
            compute = self._compute_fn

        now = time.time()
        if (self._garch_fitted_at is not None
                and (now - self._garch_fitted_at) >= self.garch_refit_s):
            self._garch_cache.clear()
            self._garch_fitted_at = None
            logger.info("shared GARCH cache age >= %.0fs; cleared for refit", self.garch_refit_s)

        cache_was_empty = not self._garch_cache
        if cache_was_empty:
            # Same cadence as the GARCH refit (mirrors CachedEngine).
            self._jump_params = self._jump_loader()

        t0 = time.time()
        res = compute(
            list(strikes),
            hours_to_expiry,
            n_sims=15000,
            seed=self.seed,
            use_svcj=True,
            use_skewed_t=True,
            use_figarch=True,
            jump_params=self._jump_params,
            garch_cache=self._garch_cache,
        )
        self.latencies.append(time.time() - t0)
        if cache_was_empty and self._garch_cache:
            self._garch_fitted_at = time.time()
        logger.info("re-priced ladder in %.1fs", self.latencies[-1])
        return res


class ExpiryEngineView:
    """Per-expiry cached ladder over the shared engine. Callable with the
    `engine_fn(strikes, hours_to_expiry, **kw)` contract `build_snapshot`
    expects (extra kwargs ignored, like CachedEngine)."""

    def __init__(self, parent: SharedPricingEngine, expiry_key: str) -> None:
        self._parent = parent
        self.expiry_key = expiry_key
        self._cache: Optional[Dict[Any, Any]] = None
        self._cached_at: float = 0.0
        self.latencies: List[float] = []

    def has_cache(self) -> bool:
        return self._cache is not None

    def __call__(self, strikes, hours_to_expiry, **kwargs) -> Dict[Any, Any]:
        now = time.time()
        parent = self._parent
        if self._cache is not None and (now - self._cached_at) < parent.reprice_s:
            return dict(self._cache)
        # Due (or uncached): needs this tick's single reprice grant.
        if not parent.try_take_reprice_token():
            if self._cache is not None:
                return dict(self._cache)  # stale but usable; retry next tick
            # Defensive only: the orchestrator skips uncached slots without a
            # grant, so a granted-less first call should never happen.
            raise RuntimeError(
                "ExpiryEngineView(%s): first price needs the reprice token" % self.expiry_key
            )
        n_before = len(parent.latencies)
        try:
            res = parent._compute(strikes, hours_to_expiry)
        except BaseException:
            parent.return_reprice_token()
            raise
        self.latencies.extend(parent.latencies[n_before:])
        self._cache = dict(res)
        self._cached_at = time.time()
        return dict(res)


# ---------------------------------------------------------------------------
# LadderSlot + tick report structures
# ---------------------------------------------------------------------------


@dataclass
class LadderSlot:
    """Everything one concurrent expiry ladder owns."""
    event_slug: str
    expiry_key: str
    markets: List[Tuple[str, float]]
    tokens: Dict[str, str]
    loop: PaperTradingLoop
    adapter: Any
    clock: SimClock
    view: ExpiryEngineView
    acquired_at: datetime
    state: str = "warming"          # warming | active | past_instant
    consec_unhealthy: int = 0
    feed_restarts: int = 0
    healthy_since_restart: bool = True
    consec_tick_errors: int = 0
    fills_total: int = 0
    all_terminal: bool = False      # cached once true (terminal is sticky)


@dataclass
class SlotTickReport:
    slot: LadderSlot
    ticked: bool = False
    skipped_warmup: bool = False
    error: bool = False
    feed_healthy: bool = False
    n_msgs: int = 0
    repriced: List[float] = field(default_factory=list)
    fills: List[Any] = field(default_factory=list)


@dataclass
class TickReport:
    slot_reports: List[SlotTickReport] = field(default_factory=list)
    teardowns: List[Tuple[str, str]] = field(default_factory=list)   # (expiry_key, reason)
    acquired: List[Tuple[str, str]] = field(default_factory=list)    # (event_slug, expiry_key)
    exit_request: Optional[str] = None
    any_slot_error: bool = False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class MultiExpiryOrchestrator:
    """Owns Dict[expiry_key, LadderSlot]; one tick() drives every slot.

    `adapter_factory(tokens) -> adapter` and
    `resolver(now, lead_days, max_n, exclude) -> [(slug, expiry_key, ladder)]`
    are late-binding callables supplied by paper_runner so its existing
    monkeypatch seams (paper_runner.PolymarketFeedAdapter,
    paper_runner.resolve_events_multi) keep working.
    """

    def __init__(
        self,
        *,
        store: MMStateStore,
        engine: SharedPricingEngine,
        config: Optional[MMConfig] = None,
        bankroll_total: float = 1000.0,
        max_expiries: int = 1,
        tick_s: float = 15.0,
        vol_gate_fn: Optional[Callable[[], object]] = None,
        data_provider: Optional[BTCDataProvider] = None,
        markout_provider: Optional[Callable[[], Optional[dict]]] = None,
        adapter_factory: Callable[[Dict[str, str]], Any] = None,
        resolver: Optional[Callable[..., List[Tuple[str, str, list]]]] = None,
        auto_mode: bool = False,
        lead_days: int = 3,
        feed_dead_ticks: int = 40,
        max_settlement_wait_h: float = 26.0,
        acquire_retry_s: float = 600.0,
        heartbeat_cb: Optional[Callable[[], None]] = None,
        settlement_instant_fn: Callable[[str], datetime] = settlement_instant_utc,
    ) -> None:
        self.store = store
        self.engine = engine
        self.config = config or MMConfig()
        self.bankroll_total = bankroll_total
        self.max_expiries = max(1, int(max_expiries))
        # Static split (user decision): fixed total/max_expiries share,
        # independent of the active-slot count -- order-independent and the
        # global utilization cap holds by construction.
        self.bankroll_share = bankroll_total / float(self.max_expiries)
        self.tick_s = tick_s
        self.vol_gate_fn = vol_gate_fn
        # ONE shared provider for every loop + the catch-up handler: the
        # DATA/ csvs are large and each provider caches full frames, so
        # per-loop providers would hold N copies. refresh() is mtime-gated,
        # so sharing costs nothing.
        self.data_provider = data_provider if data_provider is not None else BTCDataProvider()
        # wave 2 W7: ONE shared markout-report provider threaded into every
        # slot's PaperTradingLoop (via _build_slot, the single construction
        # point) -- markout is a venue/regime property, not per-ladder, so
        # sharing one callable across slots is correct (matches data_provider
        # sharing above). None (default) keeps every loop's sizing on the
        # m_prior path, same as an unwired single-expiry loop.
        self.markout_provider = markout_provider
        self.adapter_factory = adapter_factory
        self.resolver = resolver
        self.auto_mode = auto_mode
        self.lead_days = int(lead_days)
        self.feed_dead_ticks = int(feed_dead_ticks)
        self.max_settlement_wait_h = float(max_settlement_wait_h)
        self.acquire_retry_s = float(acquire_retry_s)
        self.heartbeat_cb = heartbeat_cb
        # Patchable settlement-gate seam (paper_runner passes a late-binding
        # lambda over ITS module attribute so the existing test that patches
        # paper_runner.settlement_instant_utc keeps steering the gate without
        # touching SettlementHandler's own internal binding).
        self.settlement_instant_fn = settlement_instant_fn

        self.slots: Dict[str, LadderSlot] = {}
        self.completed_expiries: Set[str] = set()
        self.ladders_settled_total = 0
        self.ladder_settlement_timeouts = 0
        # Process-lifetime adapter-restart count (per-slot counters die with
        # their slot at teardown; the heartbeat aggregate must be monotonic).
        self.feed_restarts_total = 0

        self._rr_offset = 0
        self._next_acquire_wall = 0.0
        self._last_catchup_wall = 0.0
        # (market_id, expiry_key) pairs known terminal -- terminal is sticky,
        # so cache to keep the recurring catch-up pending check cheap.
        self._terminal_markets: Set[Tuple[str, str]] = set()
        # ONE persistent handler so its provider cache persists across passes
        # (refresh() inside settle_expiry is mtime-gated -- repeated calls in
        # one pass are stats, not reloads).
        self._catchup_handler = SettlementHandler(self.store, self.config, self.data_provider)

    # -- slot construction ---------------------------------------------------

    def _build_slot(
        self, event_slug: str, expiry_key: str, ladder: List[Tuple[str, float, str]],
        now: datetime, *, resume_fills: bool = False, order_hygiene: bool = False,
    ) -> LadderSlot:
        markets = [(slug, strike) for slug, strike, _tok in ladder]
        tokens = {slug: tok for slug, _strike, tok in ladder}
        clock = SimClock(now - timedelta(seconds=self.tick_s))
        view = self.engine.view(expiry_key)
        loop = PaperTradingLoop(
            store=self.store,
            expiry_key=expiry_key,
            markets=markets,
            engine_fn=view,
            config=self.config,
            clock=clock,
            vol_gate_fn=self.vol_gate_fn,
            data_provider=self.data_provider,
            markout_provider=self.markout_provider,
            bankroll=self.bankroll_share,
            tick_dt_s=self.tick_s,
            feed_capability=FeedCapability.FULL_L2,
        )
        if resume_fills:
            # Defensive position restore for a mid-run-acquired expiry that
            # already has fills in the shared db (e.g. re-acquired after a
            # crash). No-op for a truly fresh expiry. NEVER restart() here --
            # it is store-global (see module docstring).
            try:
                replayed = loop.resume_attach(now, self.store.get_fills())
                if replayed:
                    logger.info("slot %s: resume_attach replayed %d fill(s)", expiry_key, replayed)
            except Exception:
                logger.warning("slot %s: resume_attach failed", expiry_key, exc_info=True)
        if order_hygiene:
            # Scoped stale-order hygiene: cancel any leftover PENDING/LIVE
            # store orders for THESE markets only (never the no-arg
            # cancel_all, which is store-wide and reserved for shutdown).
            for m, _k in markets:
                try:
                    loop.lifecycle.cancel_all(m)
                except Exception:
                    logger.warning("slot %s: order hygiene failed for %s",
                                   expiry_key, m, exc_info=True)
        adapter = self.adapter_factory(tokens)
        adapter.start()
        slot = LadderSlot(
            event_slug=event_slug, expiry_key=expiry_key, markets=markets,
            tokens=tokens, loop=loop, adapter=adapter, clock=clock, view=view,
            acquired_at=now,
        )
        self.slots[expiry_key] = slot
        logger.info("slot built: %s (expiry %s, %d strikes, bankroll share %.2f)",
                    event_slug, expiry_key, len(markets), self.bankroll_share)
        return slot

    def _sorted_slots(self) -> List[LadderSlot]:
        return [self.slots[k] for k in sorted(self.slots)]

    # -- startup / resume ------------------------------------------------------

    def startup(self, now: datetime, initial_events: List[Tuple[str, str, list]],
                db_existed: bool):
        """Build the initial slots and (on a pre-existing db) run the resume
        protocol. Returns the ReconciliationResult (or None).

        Resume order (see module docstring):
          1. build slots (constructors register markets in the registry)
          2. standalone settlement catch-up pass (writes SETTLEMENT
             pseudo-fills BEFORE any replay)
          3. per-slot filtered resume_attach over ONE shared fills fetch
          4. ONE venue reconcile via the first slot's lifecycle (all fill
             sims are empty at process start, so every UNKNOWN row is
             cancelled and the position check is the global fold vs the
             global store inventory -- W0.1 semantics preserved)
        """
        for event_slug, expiry_key, ladder in initial_events:
            self._build_slot(event_slug, expiry_key, ladder, now)

        recon = None
        if db_existed:
            logger.info("pre-existing state db: running multi-expiry resume protocol")
            self.settlement_catchup_pass(now, force=True)
            try:
                all_fills = self.store.get_fills()
            except Exception:
                all_fills = []
                logger.warning("resume: get_fills failed; loops start flat", exc_info=True)
            for slot in self._sorted_slots():
                slot.loop.resume_attach(now, all_fills)
            first = next(iter(self._sorted_slots()), None)
            if first is not None:
                recon = first.loop.lifecycle.restart_reconcile()
        return recon

    # -- settlement catch-up (standalone, store-wide) -------------------------

    def settlement_catchup_pass(self, now: datetime, *, force: bool = False):
        """Store-wide settlement catch-up over the merged registry (persisted
        registry merged UNDER the current slots' markets). Throttled to at
        most once per CATCHUP_MIN_INTERVAL_S and skipped entirely unless some
        non-slot registry market is past its settlement instant and
        non-terminal (the orphan-retry cadence, reviewer RISK-1)."""
        wall = time.time()
        if not force and (wall - self._last_catchup_wall) < CATCHUP_MIN_INTERVAL_S:
            return None

        registry = {
            **self.store.get_market_registry(),
            **{m: (s.expiry_key, k) for s in self.slots.values() for m, k in s.markets},
        }

        if not force:
            active_eks = set(self.slots)
            pending = False
            for market_id, (ek, _strike) in registry.items():
                if ek in active_eks or (market_id, ek) in self._terminal_markets:
                    continue
                try:
                    if now < self.settlement_instant_fn(ek):
                        continue
                except Exception:
                    continue
                try:
                    ev = self.store.get_settlement(market_id, ek)
                except Exception:
                    continue
                if ev is not None and ev.outcome in TERMINAL_OUTCOMES:
                    self._terminal_markets.add((market_id, ek))
                    continue
                pending = True
                break
            self._last_catchup_wall = wall
            if not pending:
                return None
        else:
            self._last_catchup_wall = wall

        try:
            result = self._catchup_handler.catch_up(now, registry)
            if result.events:
                logger.info("settlement catch-up pass settled %d event(s)", len(result.events))
            return result
        except Exception:
            logger.warning("settlement catch-up pass failed", exc_info=True)
            return None

    # -- teardown / acquisition ------------------------------------------------

    def teardown(self, expiry_key: str, reason: str, now: datetime) -> None:
        slot = self.slots.pop(expiry_key, None)
        if slot is None:
            return
        logger.info("tearing down ladder %s (%s): %s", expiry_key, slot.event_slug, reason)
        try:
            slot.loop.settle(now)  # final idempotent attempt
        except Exception:
            logger.warning("teardown %s: final settle failed", expiry_key, exc_info=True)
        for m, _k in slot.markets:
            try:
                slot.loop.lifecycle.cancel_all(m)
            except Exception:
                logger.warning("teardown %s: cancel_all(%s) failed", expiry_key, m, exc_info=True)
        try:
            per_ladder = slot.loop.inv.snapshot(now).per_ladder.get(expiry_key)
            if per_ladder is not None:
                self.store.upsert_ladder_state(
                    expiry_key, per_ladder, vertical_offsets=slot.loop.last_hedge_offsets
                )
        except Exception:
            logger.warning("teardown %s: ladder_state flush failed", expiry_key, exc_info=True)
        try:
            slot.adapter.stop()
        except Exception:
            logger.warning("teardown %s: adapter.stop failed", expiry_key, exc_info=True)
        self.engine.drop_view(expiry_key)
        self.completed_expiries.add(expiry_key)
        if reason == "settlement_timeout":
            self.ladder_settlement_timeouts += 1
        else:
            self.ladders_settled_total += 1
        # Acquire a replacement as soon as this tick's acquisition stage runs.
        self._next_acquire_wall = 0.0

    def acquire(self, now: datetime) -> List[LadderSlot]:
        """Auto-mode acquisition: fill spare capacity from the resolver.
        Never raises; never first-prices synchronously (new slots warm up via
        the reprice-token stagger like startup slots)."""
        if not self.auto_mode or self.resolver is None:
            return []
        capacity = self.max_expiries - len(self.slots)
        if capacity <= 0:
            return []
        wall = time.time()
        if wall < self._next_acquire_wall:
            return []
        # Heartbeat FIRST: probing can block minutes on venue retry backoff,
        # and the STALLED alarm must stay honest through it.
        if self.heartbeat_cb is not None:
            try:
                self.heartbeat_cb()
            except Exception:
                pass
        exclude = set(self.slots) | set(self.completed_expiries)
        try:
            events = self.resolver(now, self.lead_days, capacity, exclude)
        except SystemExit as exc:
            logger.warning("acquisition resolver SystemExit (treated as empty): %s", exc)
            events = []
        except Exception:
            logger.warning("acquisition resolver failed (treated as empty)", exc_info=True)
            events = []
        if not events:
            self._next_acquire_wall = wall + self.acquire_retry_s
            return []
        built: List[LadderSlot] = []
        for event_slug, expiry_key, ladder in events:
            if expiry_key in self.slots or expiry_key in self.completed_expiries:
                continue
            try:
                built.append(self._build_slot(
                    event_slug, expiry_key, ladder, now,
                    resume_fills=True, order_hygiene=True,
                ))
            except Exception:
                logger.warning("failed to build slot for %s", event_slug, exc_info=True)
        if len(self.slots) < self.max_expiries:
            self._next_acquire_wall = wall + self.acquire_retry_s
        return built

    # -- helpers ---------------------------------------------------------------

    def _all_settled_terminal(self, slot: LadderSlot) -> bool:
        if slot.all_terminal:
            return True
        for m, _k in slot.markets:
            ev = self.store.get_settlement(m, slot.expiry_key)
            if ev is None or ev.outcome not in TERMINAL_OUTCOMES:
                return False
        slot.all_terminal = True
        return True

    # -- one orchestrator tick ---------------------------------------------------

    def tick(self, now: datetime, *, btc_stale: bool = False,
             awaiting_clean_resume: bool = False) -> TickReport:
        report = TickReport()
        self.engine.begin_tick()

        ordered = self._sorted_slots()
        if ordered:
            off = self._rr_offset % len(ordered)
            ordered = ordered[off:] + ordered[:off]
        self._rr_offset += 1

        teardowns: List[Tuple[str, str]] = []
        for slot in ordered:
            st = SlotTickReport(slot=slot)

            # 1. drain ALWAYS (the adapter buffer is unbounded) -- discard if
            # this slot ends up skipped (drain-and-discard warmup policy).
            msgs: Dict[str, List[Dict[str, Any]]] = {}
            try:
                msgs = slot.adapter.drain()
            except Exception:
                logger.warning("slot %s: drain failed", slot.expiry_key, exc_info=True)
            st.n_msgs = sum(len(v) for v in msgs.values())
            try:
                st.feed_healthy = bool(slot.adapter.healthy())
            except Exception:
                st.feed_healthy = False

            # 2. per-slot feed watchdog (same two-trip rule as the
            # single-expiry runner; escalation is PROCESS-level by design).
            if not st.feed_healthy:
                slot.consec_unhealthy += 1
            else:
                slot.consec_unhealthy = 0
                slot.healthy_since_restart = True
            if slot.consec_unhealthy >= self.feed_dead_ticks:
                if not slot.healthy_since_restart:
                    logger.error(
                        "slot %s: feed still dead %d ticks after restart #%d; requesting feed_dead exit",
                        slot.expiry_key, slot.consec_unhealthy, slot.feed_restarts,
                    )
                    report.exit_request = "feed_dead"
                else:
                    logger.warning(
                        "slot %s: feed unhealthy %d consecutive ticks; restarting adapter (#%d)",
                        slot.expiry_key, slot.consec_unhealthy, slot.feed_restarts + 1,
                    )
                    try:
                        slot.adapter.stop()
                    except Exception:
                        logger.warning("slot %s: adapter.stop failed during watchdog restart",
                                       slot.expiry_key, exc_info=True)
                    slot.adapter = self.adapter_factory(slot.tokens)
                    slot.adapter.start()
                    slot.feed_restarts += 1
                    self.feed_restarts_total += 1
                    slot.consec_unhealthy = 0
                    slot.healthy_since_restart = False

            # 3. warmup skip: an uncached view needs this tick's single
            # reprice grant; without it the slot is NOT ticked at all
            # (tick() has no pre-snapshot early-return).
            warming = not slot.view.has_cache()
            if warming and not self.engine.token_available():
                st.skipped_warmup = True
                slot.state = "warming"
            else:
                slot.clock.set(now - timedelta(seconds=self.tick_s))
                n_lat_before = len(self.engine.latencies)
                try:
                    slot.loop.tick(
                        msgs, feed_healthy=st.feed_healthy,
                        manual_override=btc_stale or awaiting_clean_resume,
                    )
                    st.ticked = True
                    slot.consec_tick_errors = 0
                except Exception:
                    slot.consec_tick_errors += 1
                    st.error = True
                    report.any_slot_error = True
                    logger.error("slot %s: tick failed (consecutive=%d)",
                                 slot.expiry_key, slot.consec_tick_errors, exc_info=True)
                st.repriced = list(self.engine.latencies[n_lat_before:])
                if st.ticked:
                    st.fills = list(slot.loop.last_fills)
                    slot.fills_total += len(st.fills)
                    if slot.state == "warming":
                        slot.state = "active"

            # 4. settle -- UNCONDITIONAL of tick/grant (a skipped or
            # grant-less slot near settlement must still settle).
            try:
                instant = self.settlement_instant_fn(slot.expiry_key)
            except Exception:
                instant = None
            if instant is not None and now >= instant:
                slot.state = "past_instant"
                if not self._all_settled_terminal(slot):
                    try:
                        slot.loop.settle(now)
                    except Exception:
                        logger.error("slot %s: settlement step failed",
                                     slot.expiry_key, exc_info=True)
                # 5. terminal / timeout transitions
                if (now >= instant + timedelta(minutes=SETTLED_GRACE_MIN)
                        and self._all_settled_terminal(slot)):
                    teardowns.append((slot.expiry_key, "ladder_settled"))
                elif now >= instant + timedelta(hours=self.max_settlement_wait_h):
                    teardowns.append((slot.expiry_key, "settlement_timeout"))

            report.slot_reports.append(st)

        # 6. teardown (auto mode) or legacy process exit (fixed mode)
        for expiry_key, reason in teardowns:
            if self.auto_mode:
                self.teardown(expiry_key, reason, now)
                report.teardowns.append((expiry_key, reason))
            else:
                report.exit_request = reason

        # 7. acquisition (auto mode; also fires right after a teardown)
        if self.auto_mode and report.exit_request is None:
            for slot in self.acquire(now):
                report.acquired.append((slot.event_slug, slot.expiry_key))
            if not self.slots:
                report.exit_request = "no_quotable_events"

        # 8. recurring orphan settlement catch-up (throttled + gated inside)
        self.settlement_catchup_pass(now)

        return report
