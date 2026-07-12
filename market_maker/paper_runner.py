"""Stage-B PAPER runner (plan Section 7 Stage B) -- live WebSocket data, no
orders sent to the venue; fills are simulated in-process.

Same skeleton as the Stage-A shadow runner but the feed is the live Polymarket
CLOB WebSocket (PolymarketFeedAdapter) instead of REST polling, so the stream
carries L2 deltas AND trade prints -- the queue-behind PaperFillSimulator can
now fill our resting quotes. Everything downstream of the feed (quote engine,
sizing, spread builder, no-arb hedger, order lifecycle, fill routing,
inventory, state store) is the frozen PaperTradingLoop wiring.

MULTI-EXPIRY (market_maker/multi_runner.py): the runner drives a
MultiExpiryOrchestrator owning up to --max-expiries concurrent expiry ladders
(one PaperTradingLoop + one WS adapter + one SimClock per ladder; one shared
state store / pricing engine / vol gate / BTC data provider). In auto mode
(--event-slug auto) a settled ladder is torn down IN-PROCESS and replaced by
the next event (shadow_runner.resolve_events_multi) without a restart; the
process itself exits 42 only when there are zero quotable ladders left
(`no_quotable_events`). A fixed --event-slug <slug> runs exactly one ladder
with no acquisition and exits 42 on ladder_settled/settlement_timeout, as
before. At most ONE engine reprice runs per tick (the shared engine's reprice
token), so the STALLED heartbeat threshold math is unchanged.

Feed health is per-adapter connection liveness (WS ping/pong), passed as each
slot tick's feed_healthy override -- message silence on a quiet book is NOT
feed loss (P0b boundary note, consequence 3). The heartbeat's top-level
`feed_healthy` is the AND over active adapters; per-expiry detail lives in
the additive `expiries` dict.

Usage (from repo root; BTC data must be fresh -- run core/data/data_fetcher.py
first):

    python -m market_maker.paper_runner --event-slug bitcoin-above-on-july-10-2026 \
        --minutes 240 --tick-s 15 --reprice-s 300

    # multi-expiry auto mode (in-process rollover, up to 3 ladders):
    python -m market_maker.paper_runner --event-slug auto --max-expiries 3

    # or from a fixed config file (VPS deployment; see paper_run_config.json):
    python -m market_maker.paper_runner --config market_maker/paper_run_config.json

--minutes 0 runs indefinitely (until a stop file / SIGTERM / Ctrl-C / zero
quotable ladders / feed-death / tick-error-circuit-breaker).

--state-db points MMStateStore at a persistent path (VPS config:
market_maker/mm_paper_state.db) instead of the default per-run
out_dir/paper_state.db; a pre-existing db at that path triggers the
multi-expiry resume protocol (standalone settlement catch-up pass ->
per-slot filtered resume_attach -> ONE venue reconcile) so a crash/restart
rebuilds in-memory state from the fills table instead of starting flat.
Exit code 42 (ladder_settled / settlement_timeout in fixed mode;
no_quotable_events in auto mode) tells the systemd unit
(deploy/mm-paper.service, RestartForceExitStatus=42) to restart and retry;
1 (feed_dead / tick_errors / early failure) is a normal supervised restart;
everything else (completed / stop_file / sigterm / sigint) is 0, no restart.

Outputs under --out (default temp/paper_run/<UTC ts>/):
    quotes.csv   one row per (tick, market): market touch vs our quote, mode,
                 credibility, no-arb status (all expiries interleaved; join
                 market -> expiry via the state db's markets registry)
    fills.csv    one row per simulated fill (queue/print/latency detail)
    ticks.csv    one row per tick: wall latency, reprice latency, feed health
                 (quotes/fills/ticks csvs are appended to, header written only
                 once -- safe to reuse --out or --state-db across restarts)
    summary.md   end-of-run Stage-B report (fills, ending inventory, fold check)
    markout_report.json  per-region/tte-bucket/horizon markout report (plan
                     Change C; additive by_expiry rollup), rewritten every
                     PER_MARKET_SNAPSHOT_EVERY_N_TICKS ticks
    paper_state.db  MMStateStore (orders/fills/inventory/quotes/mid_log
                     journal); relocated by --state-db
    run_meta.json    self-describing run config; the additive `events` list is
                     rewritten on every in-process acquisition/teardown, the
                     legacy singular event_slug/expiry_key/strikes fields
                     always point at the NEAREST active expiry
    heartbeat.json   liveness file, rewritten every tick (atomic); legacy
                     top-level fields keep aggregate semantics (feed_healthy
                     = AND, bankroll_frozen = OR, counters = sums); additive:
                     n_expiries_active, ladders_settled_total,
                     ladder_settlement_timeouts, and a per-expiry `expiries`
                     dict. An initial heartbeat is written immediately after
                     out_dir creation, before event resolution/warmup, and
                     again before every acquisition probe, so slow venue
                     probing does not trip a false STALLED alert.

Control-file protocol (--control-dir, default temp/paper_run/control/):
    mm_paper.pid      this process's PID; removed in the finally block
    mm_paper.stop     touch (optionally with a target PID as the first line)
                      to request a graceful stop; polled once per tick
    current_run.json  pointer to the latest run (pid/argv/config_path first,
                      then event/expiry/out_dir/events, then
                      ended_utc/exit_reason)
See the plan doc's control-file protocol table for the full lifecycle;
market_maker/run_control.py (separate scope) is the launcher/status side of
this protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from market_maker.config import MMConfig
from market_maker.contracts import QuoteMode, RiskDirective, RiskTrigger
from market_maker.market_data_client import PolymarketFeedAdapter
from market_maker.multi_runner import (
    MultiExpiryOrchestrator,
    SharedPricingEngine,
)
from market_maker.pnl_report import (
    MARKOUT_LOOKBACK_S,
    PER_MARKET_SNAPSHOT_EVERY_N_TICKS,
    compute_pnl_rows,
    markout_report,
)
from market_maker.settlement_handler import TERMINAL_OUTCOMES, settlement_instant_utc
from market_maker.shadow_runner import (
    resolve_event,
    resolve_events_multi,
    resolve_next_event,
)
from market_maker.state_store import MMStateStore

logger = logging.getLogger("mm.paper")

DEFAULT_CONTROL_DIR = Path("temp/paper_run/control")
_BTC_INTRADAY_PATH = Path("DATA/btc_intraday_1m.csv")

# Settlement data-provider injection seam (unavoidable minimal addition, not
# in the plan's Step 2 item list): the orchestrator threads this into every
# PaperTradingLoop -> SettlementHandler (data_provider=None builds ONE real
# shared provider reading DATA/ csvs). Production always leaves this None;
# tests monkeypatch the module attribute to a fixture-backed BTCDataProvider
# so a settlement round trip can be exercised without touching DATA/ csvs.
_DATA_PROVIDER = None

# Pricing-compute injection seam (replaces the old paper_runner.CachedEngine
# patch point): None means the SharedPricingEngine lazily imports the real
# calculate_probabilities; tests patch this to a scripted
# callable(strikes, hours_to_expiry, **kw) -> {strike: p, "_meta": ...} so no
# GARCH fit ever runs in-process.
_ENGINE_COMPUTE_FN = None


# ---------------------------------------------------------------------------
# small control-file / JSON helpers
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="ascii")
    os.replace(str(tmp), str(path))


def _control_paths(control_dir: Path) -> Tuple[Path, Path, Path]:
    return control_dir / "mm_paper.pid", control_dir / "mm_paper.stop", control_dir / "current_run.json"


def _check_stop_file(stop_path: Path, own_pid: int) -> bool:
    """True if a stop file exists and is either unstamped (any PID) or
    stamped with THIS process's PID (W5). A stop file stamped with a
    different PID (a stale leftover from a prior run in the same control
    dir) is ignored -- it is still deleted in the finally block."""
    if not stop_path.exists():
        return False
    try:
        content = stop_path.read_text(encoding="ascii").strip()
    except OSError:
        return False
    if content == "":
        return True
    first_line = content.splitlines()[0].strip()
    try:
        target = int(first_line)
    except ValueError:
        return True  # malformed stamp -> treat as "any"
    return target == own_pid


def _read_btc_intraday(path: Path):
    import pandas as pd
    return pd.read_csv(path).tail(100_000)


def _safe_mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _all_settled_terminal(store: MMStateStore, markets: List[Tuple[str, float]], expiry_key: str) -> bool:
    """Kept for tests/back-compat; the orchestrator has its own cached copy."""
    for m, _k in markets:
        ev = store.get_settlement(m, expiry_key)
        if ev is None or ev.outcome not in TERMINAL_OUTCOMES:
            return False
    return True


def _book_mid(book) -> Optional[float]:
    b, a = book.best_bid(), book.best_ask()
    if b is not None and a is not None:
        return 0.5 * (float(b) + float(a))
    if b is not None:
        return float(b)
    if a is not None:
        return float(a)
    return None


def _write_heartbeat(
    out_dir: Path, ts: datetime, tick: int, feed_healthy: bool, n_msgs: int,
    fills_total: int, noarb_violations: int, unhealthy_ticks: int, pulled_ticks: int,
    tick_s: float, reprice_s: float,
    btc_data_age_s: Optional[float] = None, feed_restarts: int = 0,
    resume_discrepancies: int = 0, bankroll_frozen: bool = False,
    n_expiries_active: Optional[int] = None,
    ladders_settled_total: int = 0, ladder_settlement_timeouts: int = 0,
    expiries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    # tick_s/reprice_s feed run_control._heartbeat_threshold: a reprice tick
    # blocks in calculate_probabilities for minutes, so the STALLED threshold
    # must exceed the reprice duration, not just 3x tick_s. The multi-expiry
    # reprice token guarantees at most ONE engine call per tick, so that
    # threshold math is unchanged.
    payload: Dict[str, Any] = {
        "ts_utc": ts.isoformat(), "tick": tick, "feed_healthy": bool(feed_healthy),
        "n_msgs": n_msgs, "fills_total": fills_total, "noarb_violations": noarb_violations,
        "unhealthy_ticks": unhealthy_ticks, "pulled_ticks": pulled_ticks,
        "tick_s": tick_s, "reprice_s": reprice_s,
        "btc_data_age_s": btc_data_age_s, "feed_restarts": feed_restarts,
        "resume_discrepancies": resume_discrepancies,
        "bankroll_frozen": bool(bankroll_frozen),
    }
    if n_expiries_active is not None:
        payload["n_expiries_active"] = int(n_expiries_active)
        payload["ladders_settled_total"] = int(ladders_settled_total)
        payload["ladder_settlement_timeouts"] = int(ladder_settlement_timeouts)
        payload["expiries"] = expiries or {}
    _write_json_atomic(out_dir / "heartbeat.json", payload)


def _load_config_json(config_path: str) -> dict:
    with open(config_path, "r", encoding="ascii") as f:
        return json.load(f)


def run(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)

    ap = argparse.ArgumentParser(description="Stage-B paper runner (live WS feed, simulated fills)")
    ap.add_argument("--event-slug", default=None,
                     help="required, either here or via --config (post-parse validated)")
    ap.add_argument("--minutes", type=float, default=240.0, help="0 = run indefinitely")
    ap.add_argument("--tick-s", type=float, default=15.0)
    ap.add_argument("--reprice-s", type=float, default=300.0)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--warmup-s", type=float, default=15.0,
                    help="wait for the WS connection(s) + initial book snapshots before the first tick")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None, help="JSON file; keys match arg dests (underscore names)")
    ap.add_argument("--control-dir", default=str(DEFAULT_CONTROL_DIR))
    ap.add_argument("--btc-refresh-s", type=float, default=900.0,
                    help="re-read DATA/btc_intraday_1m.csv when its mtime changes, checked at most "
                         "this often (R1)")
    ap.add_argument("--state-db", default="",
                    help="path to a persistent MMStateStore db; \"\" (default) keeps the current "
                         "per-run out_dir/paper_state.db behavior. A pre-existing db triggers the "
                         "multi-expiry resume protocol (catch-up pass -> filtered resume_attach -> "
                         "one venue reconcile)")
    ap.add_argument("--max-settlement-wait-h", type=float, default=26.0,
                    help="per-ladder fallback: a ladder still not fully terminal this long after its "
                         "settlement instant is torn down (auto mode) or exits settlement_timeout "
                         "(fixed mode)")
    ap.add_argument("--auto-event-lead-days", type=int, default=3,
                    help="used only when --event-slug auto: probe this many days ahead (+4 margin) "
                         "for the next bitcoin-above events")
    ap.add_argument("--max-expiries", type=int, default=1,
                    help="auto mode only: quote up to this many concurrent expiry ladders; the "
                         "sizing bankroll is statically split as --bankroll / --max-expiries per "
                         "ladder. Fixed --event-slug runs always use exactly 1")
    ap.add_argument("--acquire-retry-s", type=float, default=600.0,
                    help="auto mode: how long to wait before re-probing the venue after an "
                         "acquisition attempt found no (or not enough) events")
    ap.add_argument("--btc-stale-max-s", type=float, default=7200.0,
                    help="if DATA/btc_intraday_1m.csv's mtime is older than this (or missing), pull "
                         "quotes via manual_override until fresh data lands (plan 2.3)")
    ap.add_argument("--feed-dead-ticks", type=int, default=40,
                    help="consecutive unhealthy-feed ticks before a ladder's adapter is restarted; a "
                         "second trip with zero healthy ticks since the restart exits feed_dead")
    ap.add_argument("--garch-refit-s", type=float, default=21_600.0,
                    help="shared engine GARCH cache max age before it is cleared and refit (plan 2.5)")
    ap.add_argument("--max-consecutive-tick-errors", type=int, default=20,
                    help="consecutive TICK_ERRORs before exiting tick_errors (plan 2.6)")

    # --config pre-scan (parse_known_args so unrelated flags don't error out
    # here): load the JSON, apply as argparse defaults BEFORE the real parse
    # so any flag actually present in argv still overrides it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_ns, _ = pre.parse_known_args(raw_argv)
    config_dict: Optional[dict] = None
    if pre_ns.config:
        config_dict = _load_config_json(pre_ns.config)
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in config_dict.items()})

    args = ap.parse_args(raw_argv)

    if not args.event_slug:
        ap.error("--event-slug is required (via CLI or --config)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    start = datetime.now(timezone.utc)
    own_pid = os.getpid()
    auto_mode = args.event_slug == "auto"
    max_expiries = max(1, int(args.max_expiries)) if auto_mode else 1

    # -- M3 startup order: control-dir plumbing FIRST, before any heavy or
    # network work (event resolution, BTC csv read). --
    control_dir = Path(args.control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    pid_path, stop_path, run_json_path = _control_paths(control_dir)

    if stop_path.exists():
        try:
            stop_path.unlink()
        except OSError:
            logger.warning("could not remove stale stop file %s", stop_path)

    try:
        pid_path.write_text(str(own_pid), encoding="ascii")
    except OSError:
        logger.warning("could not write PID file %s", pid_path)

    current_run: Dict[str, Any] = {
        "pid": own_pid,
        "started_utc": start.isoformat(),
        "argv": raw_argv,
        "config_path": args.config,
    }
    _write_json_atomic(run_json_path, current_run)

    stop_state: Dict[str, Optional[str]] = {"reason": None}

    def _sigterm_handler(signum, frame):  # pragma: no cover - exercised via os.kill in manual testing
        stop_state["reason"] = "sigterm"

    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError, AttributeError):
        # ValueError: not the main thread (e.g. under a test harness);
        # Windows/limited platforms may also raise -- best effort only,
        # stop-file polling still works either way.
        logger.debug("could not install SIGTERM handler", exc_info=True)

    exit_reason = "completed"
    out_dir: Optional[Path] = None
    orch: Optional[MultiExpiryOrchestrator] = None
    quotes_csv = fills_csv = ticks_csv = None
    store: Optional[MMStateStore] = None
    engine = None
    state_db_path: Optional[Path] = None
    fixed_expiry_key: Optional[str] = None
    tick_n = 0
    n_fills_total = 0
    noarb_violations = 0
    pulled_ticks = 0
    unhealthy_ticks = 0
    # 2.6 tick-failure circuit breaker.
    consec_tick_errors = 0
    # 2.3 BTC staleness guard.
    btc_data_age_s: Optional[float] = None
    last_btc_stale_warn_wall = 0.0
    # W0.1: resume protocol position-discrepancy count (0 unless db_existed
    # AND the startup reconcile found a venue/store position mismatch); once
    # set, ticks are forced manual_override=True until the first clean tick
    # completes.
    resume_discrepancies = 0
    awaiting_clean_resume_tick = False
    # Last-known per-tick aggregates so the mid-tick acquisition heartbeat
    # can re-emit a truthful payload without new adapter calls.
    hb_agg: Dict[str, Any] = {"feed_healthy": False, "n_msgs": 0}

    def _expiries_heartbeat_dict() -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if orch is None:
            return out
        for s in orch._sorted_slots():
            mode_counts: Dict[str, int] = {}
            for d in (s.loop.last_directives or {}).values():
                mode_counts[d.mode.name] = mode_counts.get(d.mode.name, 0) + 1
            try:
                fh = bool(s.adapter.healthy())
            except Exception:
                fh = False
            out[s.expiry_key] = {
                "event_slug": s.event_slug, "state": s.state,
                "feed_healthy": fh, "feed_restarts": s.feed_restarts,
                "bankroll_frozen": bool(s.loop.bankroll_state.frozen),
                "fills": s.fills_total, "mode_counts": mode_counts,
            }
        return out

    def _emit_heartbeat(ts: datetime) -> None:
        expiries = _expiries_heartbeat_dict()
        _write_heartbeat(
            out_dir, ts, tick_n, hb_agg["feed_healthy"], hb_agg["n_msgs"],
            n_fills_total, noarb_violations, unhealthy_ticks, pulled_ticks,
            args.tick_s, args.reprice_s,
            btc_data_age_s=btc_data_age_s,
            feed_restarts=orch.feed_restarts_total if orch is not None else 0,
            resume_discrepancies=resume_discrepancies,
            bankroll_frozen=any(e["bankroll_frozen"] for e in expiries.values()),
            n_expiries_active=len(orch.slots) if orch is not None else 0,
            ladders_settled_total=orch.ladders_settled_total if orch is not None else 0,
            ladder_settlement_timeouts=orch.ladder_settlement_timeouts if orch is not None else 0,
            expiries=expiries,
        )

    def _events_meta() -> List[Dict[str, Any]]:
        if orch is None:
            return []
        return [
            {"event_slug": s.event_slug, "expiry_key": s.expiry_key,
             "strikes": [k for _, k in s.markets],
             "acquired_utc": s.acquired_at.isoformat()}
            for s in orch._sorted_slots()
        ]

    def _update_run_pointers() -> None:
        """Rewrite current_run.json + run_meta.json event fields: legacy
        singular fields point at the NEAREST active expiry, the additive
        `events` list carries every active ladder."""
        events_meta = _events_meta()
        if events_meta:
            nearest = events_meta[0]
            current_run.update({
                "event_slug": nearest["event_slug"] if not auto_mode else args.event_slug,
                "expiry_key": nearest["expiry_key"],
                "strikes": nearest["strikes"],
            })
        current_run.update({"out_dir": str(out_dir), "events": events_meta})
        _write_json_atomic(run_json_path, current_run)
        if out_dir is not None:
            run_meta = {
                "bankroll": args.bankroll, "event_slug": current_run.get("event_slug", args.event_slug),
                "expiry_key": current_run.get("expiry_key"),
                "strikes": current_run.get("strikes", []),
                "tick_s": args.tick_s, "reprice_s": args.reprice_s,
                "argv": raw_argv, "started_utc": start.isoformat(), "config": config_dict,
                "state_db": str(state_db_path) if state_db_path is not None else "",
                "max_expiries": max_expiries, "events": events_meta,
            }
            _write_json_atomic(out_dir / "run_meta.json", run_meta)

    try:
        out_dir = Path(args.out) if args.out else Path("temp/paper_run") / start.strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2.2: write an initial heartbeat BEFORE event resolution/warmup --
        # auto event resolution (retries, up to ~minutes) plus warmup can
        # exceed run_control's _STARTING_GRACE_S and trip a false STALLED
        # alert on every restart otherwise. tick=0, feed_healthy=False is
        # fine here.
        try:
            _write_heartbeat(out_dir, start, 0, False, 0, 0, 0, 0, 0,
                              args.tick_s, args.reprice_s,
                              btc_data_age_s=None, feed_restarts=0)
        except Exception:
            logger.warning("initial heartbeat write failed", exc_info=True)

        # -- event resolution ------------------------------------------------
        if auto_mode:
            events = resolve_events_multi(
                datetime.now(timezone.utc), int(args.auto_event_lead_days),
                max_expiries, set(),
            )
        else:
            fixed_expiry_key, ladder = resolve_event(args.event_slug)
            events = [(args.event_slug, fixed_expiry_key, ladder)]
        for slug, ek, mkts in events:
            logger.info("event %s expiry %s: %d strikes %s", slug, ek,
                        len(mkts), [k for _, k, _t in mkts])

        state_db_path = Path(args.state_db) if args.state_db else (out_dir / "paper_state.db")

        engine = SharedPricingEngine(
            reprice_s=args.reprice_s, garch_refit_s=args.garch_refit_s,
            compute_fn=_ENGINE_COMPUTE_FN,
        )

        # 2.1: resumable state. The existence check MUST happen before
        # MMStateStore is constructed -- its __init__ creates the file, so
        # checking afterwards would never see db_existed=True.
        db_existed = state_db_path.exists()
        store = MMStateStore(str(state_db_path))

        # wave 2 W7: seed the shared markout-report holder from the
        # persistent store's own fills BEFORE the orchestrator/loops are
        # built, so a restart does not lose a week of measurement (the
        # holder stays {"report": None} on a fresh/empty store -- guarded,
        # never fatal). The SAME holder is updated at the periodic
        # markout_report() write further below so every slot's loop (via
        # markout_provider=holder.get lambda) always sees the latest report.
        _markout_holder: Dict[str, Optional[dict]] = {"report": None}
        try:
            _seed_fills = store.get_fills()
            if _seed_fills:
                _markout_holder["report"] = markout_report(
                    _seed_fills, store.mid_at_or_after,
                    store.get_market_registry(), MMConfig().belly_band,
                    now=datetime.now(timezone.utc),
                )
        except Exception:
            logger.warning("markout report seed failed; starting on the m_prior sizing path", exc_info=True)

        from core.strategy.vol_gate import compute_vol_gate

        btc_df = _read_btc_intraday(_BTC_INTRADAY_PATH)
        btc_mtime = _safe_mtime(_BTC_INTRADAY_PATH)
        btc_last_check_wall = time.time()

        def live_vol_gate():
            return compute_vol_gate(btc_df, datetime.now(timezone.utc))

        orch = MultiExpiryOrchestrator(
            store=store,
            engine=engine,
            config=MMConfig(),
            bankroll_total=args.bankroll,
            max_expiries=max_expiries,
            tick_s=args.tick_s,
            vol_gate_fn=live_vol_gate,
            data_provider=_DATA_PROVIDER,
            # wave 2 W7: one shared provider over the holder seeded above; the
            # periodic markout_report() write further below (existing C4
            # cadence) updates the SAME holder in place, so every slot's loop
            # picks up the latest report without any additional wiring.
            markout_provider=lambda: _markout_holder["report"],
            # Late-binding lambdas so the existing monkeypatch seams
            # (paper_runner.PolymarketFeedAdapter / .resolve_events_multi)
            # keep working -- the names resolve against THIS module's globals
            # at call time.
            adapter_factory=lambda tokens: PolymarketFeedAdapter(tokens),
            resolver=lambda now, lead, cap, excl: resolve_events_multi(now, lead, cap, excl),
            auto_mode=auto_mode,
            lead_days=int(args.auto_event_lead_days),
            feed_dead_ticks=args.feed_dead_ticks,
            max_settlement_wait_h=args.max_settlement_wait_h,
            acquire_retry_s=args.acquire_retry_s,
            heartbeat_cb=lambda: _emit_heartbeat(datetime.now(timezone.utc)),
            # Late-binding over THIS module's attribute: tests patch
            # paper_runner.settlement_instant_utc to steer the settle gate
            # without touching SettlementHandler's internal binding.
            settlement_instant_fn=lambda ek: settlement_instant_utc(ek),
        )

        startup_now = datetime.now(timezone.utc)
        if db_existed:
            logger.info("state db %s already existed; running resume protocol", state_db_path)
        recon = orch.startup(startup_now, events, db_existed)

        # W0.1: consume the ReconciliationResult -- log + journal any
        # position discrepancy (store fold(fills) vs venue truth) as a
        # MANUAL-trigger risk directive, and hold quoting at
        # manual_override=True until the first post-resume tick completes
        # cleanly (plan Section 5 "leave PULLED only when all health checks
        # pass", scoped to this discrepancy case).
        if recon is not None and recon.position_discrepancies:
            resume_discrepancies = len(recon.position_discrepancies)
            awaiting_clean_resume_tick = True
            logger.warning(
                "resume reconciliation found %d position discrepancy(ies): %s",
                resume_discrepancies, recon.position_discrepancies,
            )
            for market_id, (store_q, venue_q) in recon.position_discrepancies.items():
                directive = RiskDirective(
                    ts=startup_now, market_id=market_id, mode=QuoteMode.PULLED,
                    eps_add=0.0, kelly_mult=1.0, triggers=[RiskTrigger.MANUAL],
                    latched_until=startup_now, cancel_all=True,
                )
                try:
                    store.append_risk_directive(directive)
                except Exception:
                    logger.warning(
                        "failed to journal MANUAL resume-discrepancy directive for %s",
                        market_id, exc_info=True,
                    )

        _update_run_pointers()

        if not orch.slots:
            # Auto mode with nothing quotable at startup: the resume
            # catch-up already ran above; exit 42 so systemd retries later.
            exit_reason = "no_quotable_events"
            logger.warning("no quotable events at startup; exiting %s", exit_reason)
        else:
            warm_end = time.time() + args.warmup_s
            while time.time() < warm_end and not all(
                s.adapter.healthy() for s in orch.slots.values()
            ):
                time.sleep(0.5)
            if not all(s.adapter.healthy() for s in orch.slots.values()):
                logger.warning("feed(s) not healthy after %.0fs warmup; starting anyway", args.warmup_s)

            # 2.1: append, not truncate -- a fixed --out reused across resumed
            # runs must not lose prior rows. Header written only for a
            # new/empty file.
            quotes_path, fills_path, ticks_path = out_dir / "quotes.csv", out_dir / "fills.csv", out_dir / "ticks.csv"
            quotes_new = (not quotes_path.exists()) or quotes_path.stat().st_size == 0
            fills_new = (not fills_path.exists()) or fills_path.stat().st_size == 0
            ticks_new = (not ticks_path.exists()) or ticks_path.stat().st_size == 0
            quotes_csv = quotes_path.open("a", newline="", encoding="ascii")
            fills_csv = fills_path.open("a", newline="", encoding="ascii")
            ticks_csv = ticks_path.open("a", newline="", encoding="ascii")
            qw = csv.writer(quotes_csv)
            fw = csv.writer(fills_csv)
            tw = csv.writer(ticks_csv)
            if quotes_new:
                qw.writerow([
                    "ts", "market", "strike", "mkt_bid", "mkt_ask", "mkt_spread",
                    "our_bid", "our_ask", "our_spread", "bid_size", "ask_size",
                    "mode", "credibility", "p_hat", "consensus_p", "noarb_ok",
                ])
            if fills_new:
                fw.writerow([
                    "ts", "market", "order_id", "side", "price", "size", "liquidity",
                    "queue_ahead_at_fill", "print_size", "latency_applied_ms",
                    "mid_at_fill", "assumption_set",
                ])
            if ticks_new:
                tw.writerow(["ts", "tick", "wall_s", "reprice_s", "feed_healthy", "n_msgs", "n_fills",
                             "snapshot_failed"])

            end_time = None if args.minutes == 0 else (time.time() + args.minutes * 60.0)

            while end_time is None or time.time() < end_time:
                if stop_state["reason"] is not None:
                    exit_reason = stop_state["reason"]
                    break
                if _check_stop_file(stop_path, own_pid):
                    exit_reason = "stop_file"
                    break

                loop_start = time.time()
                now = datetime.now(timezone.utc)
                tick_n += 1

                # R1: re-read the BTC intraday csv when its mtime changes,
                # checked at most once per --btc-refresh-s. The `btc_df`
                # reassignment below is picked up by live_vol_gate() on its
                # next call since it is a closure over this same run()-local
                # name.
                if (loop_start - btc_last_check_wall) >= args.btc_refresh_s:
                    btc_last_check_wall = loop_start
                    new_mtime = _safe_mtime(_BTC_INTRADAY_PATH)
                    if new_mtime is not None and new_mtime != btc_mtime:
                        try:
                            btc_df = _read_btc_intraday(_BTC_INTRADAY_PATH)
                            btc_mtime = new_mtime
                            logger.info("re-read BTC intraday csv (mtime changed)")
                        except Exception:
                            logger.warning("failed to re-read BTC intraday csv", exc_info=True)

                # 2.3 BTC staleness guard: stat the csv FRESH every tick (not
                # the R1-cached btc_mtime, which only refreshes every
                # btc_refresh_s and would keep quoting up to ~15min after
                # fresh data lands).
                fresh_btc_mtime = _safe_mtime(_BTC_INTRADAY_PATH)
                if fresh_btc_mtime is None:
                    btc_data_age_s = None
                    btc_stale = True
                else:
                    btc_data_age_s = loop_start - fresh_btc_mtime
                    btc_stale = btc_data_age_s > args.btc_stale_max_s
                if btc_stale and (loop_start - last_btc_stale_warn_wall) >= 600.0:
                    logger.warning(
                        "BTC intraday csv stale (age=%s, max=%.0fs); pulling quotes via manual_override",
                        "missing" if fresh_btc_mtime is None else f"{btc_data_age_s:.0f}s",
                        args.btc_stale_max_s,
                    )
                    last_btc_stale_warn_wall = loop_start

                n_lat_before = len(engine.latencies)
                try:
                    report = orch.tick(
                        now, btc_stale=btc_stale,
                        awaiting_clean_resume=awaiting_clean_resume_tick,
                    )
                except Exception:
                    # Whole-orchestrator failure (should be rare -- per-slot
                    # tick errors are caught inside): same circuit breaker as
                    # a legacy tick error.
                    consec_tick_errors += 1
                    logger.error("orchestrator tick %d failed (consecutive tick errors=%d)",
                                 tick_n, consec_tick_errors, exc_info=True)
                    tw.writerow([now.isoformat(), tick_n, f"{time.time() - loop_start:.2f}",
                                 "", 0, 0, "TICK_ERROR", ""])
                    ticks_csv.flush()
                    try:
                        _emit_heartbeat(now)
                    except Exception:
                        logger.warning("heartbeat write failed", exc_info=True)
                    if consec_tick_errors >= args.max_consecutive_tick_errors:
                        logger.error("%d consecutive tick errors; exiting tick_errors", consec_tick_errors)
                        exit_reason = "tick_errors"
                        break
                    time.sleep(max(0.0, args.tick_s - (time.time() - loop_start)))
                    continue

                slot_reps = report.slot_reports
                feed_healthy_agg = all(r.feed_healthy for r in slot_reps) if slot_reps else True
                n_msgs = sum(r.n_msgs for r in slot_reps)
                hb_agg["feed_healthy"] = feed_healthy_agg
                hb_agg["n_msgs"] = n_msgs
                if not feed_healthy_agg:
                    unhealthy_ticks += 1

                # 2.6: >=1 failing slot this tick counts one consecutive tick
                # error and writes the legacy TICK_ERROR ticks.csv row
                # (preserves the circuit-breaker semantics + row format); a
                # fully clean tick resets the counter and releases the
                # post-resume manual_override hold (W0.1).
                if report.any_slot_error:
                    consec_tick_errors += 1
                    tw.writerow([now.isoformat(), tick_n, f"{time.time() - loop_start:.2f}",
                                 "", int(feed_healthy_agg), n_msgs, "TICK_ERROR", ""])
                    ticks_csv.flush()
                    try:
                        _emit_heartbeat(now)
                    except Exception:
                        logger.warning("heartbeat write failed", exc_info=True)
                    if consec_tick_errors >= args.max_consecutive_tick_errors:
                        logger.error("%d consecutive tick errors; exiting tick_errors", consec_tick_errors)
                        exit_reason = "tick_errors"
                        break
                    if report.exit_request is not None:
                        exit_reason = report.exit_request
                        break
                    time.sleep(max(0.0, args.tick_s - (time.time() - loop_start)))
                    continue

                consec_tick_errors = 0
                if awaiting_clean_resume_tick:
                    awaiting_clean_resume_tick = False

                n_fills_this_tick = 0
                snapshot_failed_any = False
                for r in slot_reps:
                    if not r.ticked:
                        continue
                    slot = r.slot
                    lp = slot.loop
                    snap = lp.last_snapshot
                    fv = lp.last_fair_value
                    directives = lp.last_directives or {}
                    quote_sets = lp.last_quote_sets or {}
                    noarb_ok = lp.last_checked_quote_sets is not None
                    if not noarb_ok:
                        noarb_violations += 1
                    if lp.snapshot_failed:
                        snapshot_failed_any = True

                    for slug, strike in slot.markets:
                        ms = lp.books[slug]
                        qs = quote_sets.get(slug)
                        d = directives.get(slug)
                        mode = d.mode.name if d is not None else ""
                        if mode == "PULLED":
                            pulled_ticks += 1
                        mkt_bid = ms.best_bid()
                        mkt_ask = ms.best_ask()
                        cred = fv.credibility if fv is not None else float("nan")
                        p_hat = snap.p_hat.get(strike, float("nan")) if snap is not None else float("nan")
                        cons = fv.consensus_p.get(strike, float("nan")) if fv is not None else float("nan")
                        qw.writerow([
                            now.isoformat(), slug, strike,
                            f"{mkt_bid:.4f}" if mkt_bid is not None else "",
                            f"{mkt_ask:.4f}" if mkt_ask is not None else "",
                            f"{(mkt_ask - mkt_bid):.4f}" if (mkt_bid is not None and mkt_ask is not None) else "",
                            f"{qs.bid_price:.4f}" if qs else "", f"{qs.ask_price:.4f}" if qs else "",
                            f"{(qs.ask_price - qs.bid_price):.4f}" if qs else "",
                            f"{qs.bid_size:.2f}" if qs else "", f"{qs.ask_size:.2f}" if qs else "",
                            mode, f"{cred:.4f}", f"{p_hat:.4f}", f"{cons:.4f}", int(noarb_ok),
                        ])

                    for f in r.fills:
                        n_fills_total += 1
                        n_fills_this_tick += 1
                        fw.writerow([
                            f.ts.isoformat(), f.market_id, f.order_id, f.side.name,
                            f"{f.price:.4f}", f"{f.size:.2f}", f.liquidity.name,
                            f"{getattr(f, 'queue_ahead_at_fill', float('nan')):.2f}",
                            f"{getattr(f, 'print_size', float('nan')):.2f}",
                            getattr(f, "latency_applied_ms", ""),
                            getattr(f, "mid_at_fill", ""), getattr(f, "assumption_set", ""),
                        ])
                        logger.info("FILL %s %s %.2f @ %.4f", f.market_id, f.side.name, f.size, f.price)

                repriced = engine.latencies[n_lat_before:] if len(engine.latencies) > n_lat_before else []
                tw.writerow([
                    now.isoformat(), tick_n, f"{time.time() - loop_start:.2f}",
                    f"{repriced[0]:.1f}" if repriced else "",
                    int(feed_healthy_agg), n_msgs, n_fills_this_tick,
                    int(snapshot_failed_any),
                ])
                quotes_csv.flush()
                fills_csv.flush()
                ticks_csv.flush()

                # In-process rollover bookkeeping: rewrite the run pointers
                # whenever the active event set changed this tick.
                if report.teardowns or report.acquired:
                    for ek, reason in report.teardowns:
                        logger.info("ladder %s torn down (%s)", ek, reason)
                    for slug, ek in report.acquired:
                        logger.info("ladder acquired: %s (expiry %s)", slug, ek)
                    try:
                        _update_run_pointers()
                    except Exception:
                        logger.warning("run pointer update failed", exc_info=True)

                # F5: ONE guarded fills fetch, hoisted above both the pnl and
                # markout blocks below.
                try:
                    fills_all = store.get_fills()
                except Exception:
                    fills_all = None
                    logger.warning("get_fills failed at tick %d", tick_n, exc_info=True)

                # R3 invariant: PnL snapshot MUST run AFTER settle (which the
                # orchestrator tick already did) so realized (from
                # cash+avg_cost, B3) and inventory stay in sync -- a
                # settlement pseudo-fill's cash and its q->0 inventory update
                # must land in the SAME snapshot, never split across ticks.
                # ONE global TOTAL row per tick (single-writer -- the
                # dashboard equity curve stays one curve); per-market rows
                # are stamped with their OWN expiry via the markets registry.
                try:
                    if fills_all is not None:
                        inv_all = store.get_all_inventory()
                        registry = store.get_market_registry()
                        expiry_by_market = {m: ek for m, (ek, _k) in registry.items()}
                        mids_by_market: Dict[str, Optional[float]] = {}
                        consensus_by_market: Dict[str, Optional[float]] = {}
                        for s in orch.slots.values():
                            fv_s = s.loop.last_fair_value
                            for m, k in s.markets:
                                mids_by_market[m] = _book_mid(s.loop.books[m])
                                consensus_by_market[m] = fv_s.consensus_p.get(k) if fv_s is not None else None
                        settlements = store.get_all_settlements()
                        pnl_rows = compute_pnl_rows(
                            now, None if auto_mode else fixed_expiry_key,
                            fills_all, inv_all, mids_by_market, consensus_by_market,
                            args.bankroll, settlements=settlements,
                            expiry_by_market=expiry_by_market,
                        )
                        # W4 row-volume cap: TOTAL row every tick, per-market
                        # rows only every Nth tick.
                        for row in pnl_rows:
                            if row.market_id is None or tick_n % PER_MARKET_SNAPSHOT_EVERY_N_TICKS == 0:
                                store.append_pnl_snapshot(row)
                except Exception:
                    logger.warning("pnl snapshot failed at tick %d", tick_n, exc_info=True)

                # C4: per-region markout report, same cadence as the
                # per-market PnL rows above (row-volume cap, W4).
                if tick_n % PER_MARKET_SNAPSHOT_EVERY_N_TICKS == 0:
                    try:
                        if fills_all is not None:
                            report_json = markout_report(
                                fills_all, store.mid_at_or_after,
                                store.get_market_registry(), orch.config.belly_band,
                                now=now,
                            )
                            _write_json_atomic(out_dir / "markout_report.json", report_json)
                            # wave 2 W7: keep the shared sizing-provider holder
                            # current -- every slot's loop reads this same
                            # dict via markout_provider=lambda: holder["report"].
                            _markout_holder["report"] = report_json
                            store.prune_mid_log(now - timedelta(seconds=MARKOUT_LOOKBACK_S))
                    except Exception:
                        logger.warning("markout report failed at tick %d", tick_n, exc_info=True)

                    # W0.2: prune the quotes table at the same cadence.
                    try:
                        store.prune_quotes(now - timedelta(seconds=orch.config.quotes_retention_s))
                    except Exception:
                        logger.warning("prune_quotes failed at tick %d", tick_n, exc_info=True)

                    # 2026-07-11: prune the trade_prints table on the same
                    # cadence/retention as quotes.
                    try:
                        store.prune_trade_prints(now - timedelta(seconds=orch.config.quotes_retention_s))
                    except Exception:
                        logger.warning("prune_trade_prints failed at tick %d", tick_n, exc_info=True)

                    # W1.1: persist per-market LiquidityState per slot.
                    for s in orch.slots.values():
                        for m, _k in s.markets:
                            liq = s.loop.last_liquidity.get(m)
                            if liq is None:
                                continue
                            try:
                                store.append_liquidity_window(liq)
                            except Exception:
                                logger.warning(
                                    "append_liquidity_window failed for %s at tick %d", m, tick_n,
                                    exc_info=True,
                                )
                    try:
                        store.prune_liquidity_windows(now - timedelta(seconds=orch.config.quotes_retention_s))
                    except Exception:
                        logger.warning("prune_liquidity_windows failed at tick %d", tick_n, exc_info=True)

                    # W2.4: persist each active expiry's ladder state.
                    for s in orch.slots.values():
                        try:
                            per_ladder = s.loop.inv.snapshot(now).per_ladder.get(s.expiry_key)
                            if per_ladder is not None:
                                store.upsert_ladder_state(
                                    s.expiry_key, per_ladder,
                                    vertical_offsets=s.loop.last_hedge_offsets,
                                )
                        except Exception:
                            logger.warning("upsert_ladder_state failed for %s at tick %d",
                                           s.expiry_key, tick_n, exc_info=True)

                try:
                    _emit_heartbeat(now)
                except Exception:
                    logger.warning("heartbeat write failed", exc_info=True)

                # 2.2: process-level exit requests from the orchestrator --
                # fixed-mode ladder_settled/settlement_timeout (42), auto-mode
                # no_quotable_events (42), feed_dead (1).
                if report.exit_request is not None:
                    exit_reason = report.exit_request
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0.0, args.tick_s - elapsed))
    except KeyboardInterrupt:
        exit_reason = "sigint"
    finally:
        # L4: best-effort cancel of any still-LIVE paper orders so no DB rows
        # are left LIVE after shutdown (crash-before-cancel would otherwise
        # confuse the next resume's reconcile). Scoped per market through
        # each slot's OWN lifecycle/fill-sim -- a cross-slot store-wide
        # cancel would route other sims' orders through the wrong bridge.
        if orch is not None:
            for slot in list(orch.slots.values()):
                for m, _k in slot.markets:
                    try:
                        slot.loop.lifecycle.cancel_all(m)
                    except Exception:
                        logger.warning("cancel_all(%s) failed during shutdown", m, exc_info=True)
                try:
                    slot.adapter.stop()
                except Exception:
                    logger.warning("adapter.stop failed during shutdown", exc_info=True)
        for fh in (quotes_csv, fills_csv, ticks_csv):
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        try:
            if stop_path.exists():
                stop_path.unlink()
        except OSError:
            pass
        try:
            if pid_path.exists():
                pid_path.unlink()
        except OSError:
            pass

        exc_type, exc_val, _exc_tb = sys.exc_info()
        final_reason = exit_reason
        if exc_type is not None:
            final_reason = "error: %s: %s" % (exc_type.__name__, exc_val)
        current_run["ended_utc"] = datetime.now(timezone.utc).isoformat()
        current_run["exit_reason"] = final_reason
        try:
            _write_json_atomic(run_json_path, current_run)
        except OSError:
            logger.warning("could not write %s", run_json_path)

    if orch is None:
        # Failed before the orchestrator was constructed (e.g. resolve_event
        # raised, or KeyboardInterrupt landed very early) -- nothing to
        # summarize.
        return 1

    inv_lines = []
    fold_lines = []
    summary_now = datetime.now(timezone.utc)
    for slot in orch._sorted_slots():
        fold_ok = slot.loop.fold_matches_inventory(own_markets_only=True)
        fold_lines.append(f"    - {slot.expiry_key}: {fold_ok}")
        inv_snap = slot.loop.inv.snapshot(summary_now)
        for m, _k in slot.markets:
            ci = inv_snap.per_contract.get(m)
            if ci is not None and ci.q != 0.0:
                inv_lines.append(f"    - {m}: q={ci.q:.2f} avg_cost={ci.avg_cost:.4f}")
    summary = out_dir / "summary.md"
    summary.write_text(
        "# Stage-B paper run summary\n\n"
        f"- event(s): {args.event_slug} (max_expiries {max_expiries}); active at end: "
        f"{[s.expiry_key for s in orch._sorted_slots()] or 'none'}\n"
        f"- start: {start.isoformat()}  duration: {args.minutes} min "
        f"({'indefinite' if args.minutes == 0 else 'fixed'}), "
        f"tick {args.tick_s}s, reprice {args.reprice_s}s, bankroll {args.bankroll} "
        f"(share/ladder {orch.bankroll_share:.2f})\n"
        f"- exit_reason: {exit_reason}\n"
        f"- ticks: {tick_n} (feed-unhealthy ticks: {unhealthy_ticks})\n"
        f"- simulated fills: {n_fills_total}\n"
        f"- ladders settled in-process: {orch.ladders_settled_total} "
        f"(settlement timeouts: {orch.ladder_settlement_timeouts})\n"
        f"- ending open inventory:\n" + ("\n".join(inv_lines) or "    - flat") + "\n"
        f"- fold(own fills) == inventory per ladder:\n" + ("\n".join(fold_lines) or "    - n/a") + "\n"
        f"- self-inflicted no-arb violations (post-repair): {noarb_violations}\n"
        f"- PULLED (market,tick) rows: {pulled_ticks}\n"
        f"- re-price latencies (s): {['%.1f' % v for v in engine.latencies]}\n"
        f"- feed restarts: {orch.feed_restarts_total}\n"
        f"- data: quotes.csv / fills.csv / ticks.csv in this directory; state db: {state_db_path}\n",
        encoding="ascii",
    )
    logger.info("paper run complete: %s (exit_reason=%s)", out_dir, exit_reason)
    print(str(out_dir))
    # 2.2 exit-code mapping: ladder_settled/settlement_timeout (fixed mode) and
    # no_quotable_events (auto mode) -> 42 (systemd RestartForceExitStatus
    # restarts and retries); feed_dead/tick_errors -> 1 (supervised restart,
    # clean re-init); everything else -- completed, stop_file, sigterm,
    # sigint -- -> 0 (no restart on an intentional stop).
    if exit_reason in ("ladder_settled", "settlement_timeout", "no_quotable_events"):
        return 42
    if exit_reason in ("feed_dead", "tick_errors"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
