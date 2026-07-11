"""Stage-B PAPER runner (plan Section 7 Stage B) -- live WebSocket data, no
orders sent to the venue; fills are simulated in-process.

Same skeleton as the Stage-A shadow runner but the feed is the live Polymarket
CLOB WebSocket (PolymarketFeedAdapter) instead of REST polling, so the stream
carries L2 deltas AND trade prints -- the queue-behind PaperFillSimulator can
now fill our resting quotes. Everything downstream of the feed (quote engine,
sizing, spread builder, no-arb hedger, order lifecycle, fill routing,
inventory, state store) is the frozen PaperTradingLoop wiring.

Feed health is the adapter's connection liveness (WS ping/pong), passed as the
tick's feed_healthy override -- message silence on a quiet book is NOT feed
loss (P0b boundary note, consequence 3).

Usage (from repo root; BTC data must be fresh -- run core/data/data_fetcher.py
first):

    python -m market_maker.paper_runner --event-slug bitcoin-above-on-july-10-2026 \
        --minutes 240 --tick-s 15 --reprice-s 300

    # or from a fixed config file (VPS deployment; see paper_run_config.json):
    python -m market_maker.paper_runner --config market_maker/paper_run_config.json

--minutes 0 runs indefinitely (until a stop file / SIGTERM / Ctrl-C / ladder
settlement / feed-death / tick-error-circuit-breaker).

--event-slug auto resolves the next bitcoin-above event automatically
(market_maker.shadow_runner.resolve_next_event, --auto-event-lead-days).

--state-db points MMStateStore at a persistent path (VPS config:
market_maker/mm_paper_state.db) instead of the default per-run
out_dir/paper_state.db; a pre-existing db at that path triggers the resume
protocol (mark_all_live_orders_unknown -> loop.restart -> loop.settle(
catch_up=True)) so a crash/restart rebuilds in-memory state from the fills
table instead of starting flat. Exit code 42 (ladder_settled /
settlement_timeout) tells the systemd unit (deploy/mm-paper.service,
RestartForceExitStatus=42) to roll over to the next event; 1 (feed_dead /
tick_errors / early failure) is a normal supervised restart; everything
else (completed / stop_file / sigterm / sigint) is 0, no restart.

Outputs under --out (default temp/paper_run/<UTC ts>/):
    quotes.csv   one row per (tick, market): market touch vs our quote, mode,
                 credibility, no-arb status
    fills.csv    one row per simulated fill (queue/print/latency detail)
    ticks.csv    one row per tick: wall latency, reprice latency, feed health
                 (quotes/fills/ticks csvs are appended to, header written only
                 once -- safe to reuse --out or --state-db across restarts)
    summary.md   end-of-run Stage-B report (fills, ending inventory, fold check)
    markout_report.json  per-region/tte-bucket/horizon markout report (plan
                     Change C), rewritten every PER_MARKET_SNAPSHOT_EVERY_N_
                     TICKS ticks (pnl_report.markout_report over store fills
                     + mid_log via store.mid_at_or_after)
    paper_state.db  MMStateStore (orders/fills/inventory/quotes/mid_log
                     journal); relocated by --state-db
    run_meta.json    self-describing run config, written once after resolve_event
    heartbeat.json   liveness file, rewritten every tick (atomic); includes
                     btc_data_age_s, feed_restarts, resume_discrepancies, and
                     bankroll_frozen (plan Wave 1 W1.3 -- true while the Beuoy
                     bankroll is degenerate and not yet auto-unfrozen). An
                     initial heartbeat is written immediately after out_dir
                     creation, before resolve_event/warmup, so slow
                     auto-resolve/warmup does not trip a false STALLED alert.

Control-file protocol (--control-dir, default temp/paper_run/control/):
    mm_paper.pid      this process's PID; removed in the finally block
    mm_paper.stop     touch (optionally with a target PID as the first line)
                      to request a graceful stop; polled once per tick
    current_run.json  pointer to the latest run (pid/argv/config_path first,
                      then event/expiry/out_dir, then ended_utc/exit_reason)
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
from market_maker.harness import PaperTradingLoop
from market_maker.market_data_client import FeedCapability, PolymarketFeedAdapter
from market_maker.order_lifecycle import SimClock
from market_maker.pnl_report import (
    MARKOUT_LOOKBACK_S,
    PER_MARKET_SNAPSHOT_EVERY_N_TICKS,
    compute_pnl_rows,
    markout_report,
)
from market_maker.settlement_handler import TERMINAL_OUTCOMES, settlement_instant_utc
from market_maker.shadow_runner import CachedEngine, resolve_event, resolve_next_event
from market_maker.state_store import MMStateStore

logger = logging.getLogger("mm.paper")

DEFAULT_CONTROL_DIR = Path("temp/paper_run/control")
_BTC_INTRADAY_PATH = Path("DATA/btc_intraday_1m.csv")

# Settlement data-provider injection seam (unavoidable minimal addition, not
# in the plan's Step 2 item list): PaperTradingLoop -> SettlementHandler
# already accepts an injectable BTCDataProvider (data_provider=None builds
# the real one reading DATA/ csvs). Production always leaves this None;
# tests monkeypatch the module attribute to a fixture-backed BTCDataProvider
# so a settlement round trip can be exercised without touching DATA/ csvs.
_DATA_PROVIDER = None


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
) -> None:
    # tick_s/reprice_s feed run_control._heartbeat_threshold: a reprice tick
    # blocks in calculate_probabilities for minutes, so the STALLED threshold
    # must exceed the reprice duration, not just 3x tick_s.
    _write_json_atomic(out_dir / "heartbeat.json", {
        "ts_utc": ts.isoformat(), "tick": tick, "feed_healthy": bool(feed_healthy),
        "n_msgs": n_msgs, "fills_total": fills_total, "noarb_violations": noarb_violations,
        "unhealthy_ticks": unhealthy_ticks, "pulled_ticks": pulled_ticks,
        "tick_s": tick_s, "reprice_s": reprice_s,
        "btc_data_age_s": btc_data_age_s, "feed_restarts": feed_restarts,
        "resume_discrepancies": resume_discrepancies,
        "bankroll_frozen": bool(bankroll_frozen),
    })


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
                    help="wait for the WS connection + initial book snapshots before the first tick")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None, help="JSON file; keys match arg dests (underscore names)")
    ap.add_argument("--control-dir", default=str(DEFAULT_CONTROL_DIR))
    ap.add_argument("--btc-refresh-s", type=float, default=900.0,
                    help="re-read DATA/btc_intraday_1m.csv when its mtime changes, checked at most "
                         "this often (R1)")
    ap.add_argument("--state-db", default="",
                    help="path to a persistent MMStateStore db; \"\" (default) keeps the current "
                         "per-run out_dir/paper_state.db behavior. A pre-existing db triggers the "
                         "restart()->settle(catch_up=True) resume protocol (plan 2.1)")
    ap.add_argument("--max-settlement-wait-h", type=float, default=26.0,
                    help="hard fallback: exit settlement_timeout if the ladder is still not fully "
                         "terminal this long after settlement_instant_utc (plan 2.2)")
    ap.add_argument("--auto-event-lead-days", type=int, default=3,
                    help="used only when --event-slug auto: probe this many days ahead (+4 margin) "
                         "for the next bitcoin-above event (plan 2.2)")
    ap.add_argument("--btc-stale-max-s", type=float, default=7200.0,
                    help="if DATA/btc_intraday_1m.csv's mtime is older than this (or missing), pull "
                         "quotes via manual_override until fresh data lands (plan 2.3)")
    ap.add_argument("--feed-dead-ticks", type=int, default=40,
                    help="consecutive unhealthy-feed ticks before the adapter is restarted; a second "
                         "trip with zero healthy ticks since the restart exits feed_dead (plan 2.4)")
    ap.add_argument("--garch-refit-s", type=float, default=21_600.0,
                    help="CachedEngine GARCH cache max age before it is cleared and refit (plan 2.5)")
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

    # -- M3 startup order: control-dir plumbing FIRST, before any heavy or
    # network work (resolve_event, BTC csv read). --
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
    loop: Optional[PaperTradingLoop] = None
    adapter: Optional[PolymarketFeedAdapter] = None
    quotes_csv = fills_csv = ticks_csv = None
    store: Optional[MMStateStore] = None
    markets: List[Tuple[str, float]] = []
    engine = None
    expiry_key = None
    tick_n = 0
    n_fills_total = 0
    noarb_violations = 0
    pulled_ticks = 0
    unhealthy_ticks = 0
    # 2.4 feed watchdog: dedicated counter (NOT the lifetime unhealthy_ticks
    # accumulator above, which never resets) plus a restart-generation flag.
    consec_unhealthy_feed = 0
    feed_restarts = 0
    healthy_since_restart = True
    # 2.6 tick-failure circuit breaker.
    consec_tick_errors = 0
    # 2.3 BTC staleness guard.
    btc_data_age_s: Optional[float] = None
    last_btc_stale_warn_wall = 0.0
    # W0.1: resume protocol position-discrepancy count (0 unless db_existed
    # AND restart() found a venue/store position mismatch); once set, ticks
    # are forced manual_override=True until the first clean tick completes.
    resume_discrepancies = 0
    awaiting_clean_resume_tick = False

    try:
        out_dir = Path(args.out) if args.out else Path("temp/paper_run") / start.strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2.2: write an initial heartbeat BEFORE resolve_event/warmup -- auto
        # event resolution (retries, up to ~150s) plus warmup can exceed
        # run_control's _STARTING_GRACE_S and trip a false STALLED alert on
        # every rollover otherwise. tick=0, feed_healthy=False is fine here.
        try:
            _write_heartbeat(out_dir, start, 0, False, 0, 0, 0, 0, 0,
                              args.tick_s, args.reprice_s,
                              btc_data_age_s=None, feed_restarts=0)
        except Exception:
            logger.warning("initial heartbeat write failed", exc_info=True)

        if args.event_slug == "auto":
            expiry_key, ladder = resolve_next_event(
                datetime.now(timezone.utc), int(args.auto_event_lead_days)
            )
        else:
            expiry_key, ladder = resolve_event(args.event_slug)
        markets = [(slug, strike) for slug, strike, _tok in ladder]
        tokens = {slug: tok for slug, _strike, tok in ladder}
        logger.info("event %s expiry %s: %d strikes %s", args.event_slug, expiry_key,
                    len(markets), [k for _, k in markets])

        current_run.update({
            "event_slug": args.event_slug, "expiry_key": expiry_key,
            "out_dir": str(out_dir), "strikes": [k for _, k in markets],
        })
        _write_json_atomic(run_json_path, current_run)

        state_db_path = Path(args.state_db) if args.state_db else (out_dir / "paper_state.db")

        run_meta = {
            "bankroll": args.bankroll, "event_slug": args.event_slug, "expiry_key": expiry_key,
            "strikes": [k for _, k in markets], "tick_s": args.tick_s, "reprice_s": args.reprice_s,
            "argv": raw_argv, "started_utc": start.isoformat(), "config": config_dict,
            "state_db": str(state_db_path),
        }
        _write_json_atomic(out_dir / "run_meta.json", run_meta)

        engine = CachedEngine(reprice_s=args.reprice_s, garch_refit_s=args.garch_refit_s)

        # 2.1: resumable state. The existence check MUST happen before
        # MMStateStore is constructed -- its __init__ creates the file, so
        # checking afterwards would never see db_existed=True.
        db_existed = state_db_path.exists()
        store = MMStateStore(str(state_db_path))
        clock = SimClock(start - timedelta(seconds=args.tick_s))

        from core.strategy.vol_gate import compute_vol_gate

        btc_df = _read_btc_intraday(_BTC_INTRADAY_PATH)
        btc_mtime = _safe_mtime(_BTC_INTRADAY_PATH)
        btc_last_check_wall = time.time()

        def live_vol_gate():
            return compute_vol_gate(btc_df, datetime.now(timezone.utc))

        loop = PaperTradingLoop(
            store=store,
            expiry_key=expiry_key,
            markets=markets,
            engine_fn=engine,
            config=MMConfig(),
            clock=clock,
            vol_gate_fn=live_vol_gate,
            data_provider=_DATA_PROVIDER,
            bankroll=args.bankroll,
            tick_dt_s=args.tick_s,
            feed_capability=FeedCapability.FULL_L2,
        )

        if db_existed:
            # 2.1 resume protocol. ORDER MATTERS: restart() strictly before
            # settle(catch_up=True) -- restart() replays the full fills
            # table into a fresh InventoryManager; settle's registry-merge
            # catch-up then closes out any previous-event position it finds
            # still open (harness.py settle() docstring/invariant comment).
            # W0.1: the redundant store.mark_all_live_orders_unknown() call
            # that used to precede loop.restart() here is dropped --
            # restart() already calls it internally (harness.py:574).
            resume_now = datetime.now(timezone.utc)
            logger.info("state db %s already existed; running resume protocol", state_db_path)
            recon = loop.restart(resume_now)
            loop.settle(resume_now, catch_up=True)

            # W0.1: consume the ReconciliationResult -- log + journal any
            # position discrepancy (store fold(fills) vs venue truth) as a
            # MANUAL-trigger risk directive, and hold quoting at
            # manual_override=True until the first post-resume tick
            # completes cleanly (plan Section 5 "leave PULLED only when all
            # health checks pass", scoped to this discrepancy case).
            if recon.position_discrepancies:
                resume_discrepancies = len(recon.position_discrepancies)
                awaiting_clean_resume_tick = True
                logger.warning(
                    "resume reconciliation found %d position discrepancy(ies): %s",
                    resume_discrepancies, recon.position_discrepancies,
                )
                for market_id, (store_q, venue_q) in recon.position_discrepancies.items():
                    directive = RiskDirective(
                        ts=resume_now, market_id=market_id, mode=QuoteMode.PULLED,
                        eps_add=0.0, kelly_mult=1.0, triggers=[RiskTrigger.MANUAL],
                        latched_until=resume_now, cancel_all=True,
                    )
                    try:
                        store.append_risk_directive(directive)
                    except Exception:
                        logger.warning(
                            "failed to journal MANUAL resume-discrepancy directive for %s",
                            market_id, exc_info=True,
                        )

        adapter = PolymarketFeedAdapter(tokens)
        adapter.start()
        warm_end = time.time() + args.warmup_s
        while time.time() < warm_end and not adapter.healthy():
            time.sleep(0.5)
        if not adapter.healthy():
            logger.warning("feed not healthy after %.0fs warmup; starting anyway", args.warmup_s)

        # 2.1: append, not truncate -- a fixed --out reused across resumed
        # runs must not lose prior rows. Header written only for a new/empty
        # file.
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
            # reassignment below is picked up by live_vol_gate() on its next
            # call since it is a closure over this same run()-local name (no
            # nonlocal needed -- the rebind happens in this same frame).
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

            messages: Dict[str, List[Dict[str, Any]]] = adapter.drain()
            feed_healthy = adapter.healthy()
            if not feed_healthy:
                unhealthy_ticks += 1
                consec_unhealthy_feed += 1
            else:
                consec_unhealthy_feed = 0
                healthy_since_restart = True
            n_msgs = sum(len(v) for v in messages.values())

            # 2.4 feed watchdog: an exception escaping the adapter's asyncio
            # loop kills the feed thread once and forever, so healthy()
            # stays False silently for the rest of the month unless
            # something restarts it.
            if consec_unhealthy_feed >= args.feed_dead_ticks:
                if not healthy_since_restart:
                    logger.error(
                        "feed still dead %d ticks after restart #%d; exiting feed_dead",
                        consec_unhealthy_feed, feed_restarts,
                    )
                    exit_reason = "feed_dead"
                    break
                logger.warning(
                    "feed unhealthy for %d consecutive ticks; restarting adapter (restart #%d)",
                    consec_unhealthy_feed, feed_restarts + 1,
                )
                try:
                    adapter.stop()
                except Exception:
                    logger.warning("adapter.stop() failed during watchdog restart", exc_info=True)
                adapter = PolymarketFeedAdapter(tokens)
                adapter.start()
                feed_restarts += 1
                consec_unhealthy_feed = 0
                healthy_since_restart = False

            # 2.3 BTC staleness guard: stat the csv FRESH every tick (not the
            # R1-cached btc_mtime, which only refreshes every btc_refresh_s
            # and would keep quoting up to ~15min after fresh data lands).
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

            clock.set(now - timedelta(seconds=args.tick_s))
            n_lat_before = len(engine.latencies)
            # W0.1: hold quoting at manual_override=True on every post-resume
            # tick until one tick completes cleanly (no exception) -- cleared
            # right after that tick below, never inside the except branch.
            tick_manual_override = btc_stale or awaiting_clean_resume_tick
            try:
                loop.tick(messages, feed_healthy=feed_healthy, manual_override=tick_manual_override)
            except Exception:
                consec_tick_errors += 1
                logger.error("tick %d failed (consecutive tick errors=%d)", tick_n,
                             consec_tick_errors, exc_info=True)
                tw.writerow([now.isoformat(), tick_n, f"{time.time() - loop_start:.2f}",
                             "", int(feed_healthy), n_msgs, "TICK_ERROR", ""])
                ticks_csv.flush()
                try:
                    _write_heartbeat(out_dir, now, tick_n, feed_healthy, n_msgs, n_fills_total,
                                      noarb_violations, unhealthy_ticks, pulled_ticks,
                                      args.tick_s, args.reprice_s,
                                      btc_data_age_s=btc_data_age_s, feed_restarts=feed_restarts,
                                      resume_discrepancies=resume_discrepancies,
                                      bankroll_frozen=loop.bankroll_state.frozen)
                except Exception:
                    logger.warning("heartbeat write failed", exc_info=True)
                if consec_tick_errors >= args.max_consecutive_tick_errors:
                    logger.error("%d consecutive tick errors; exiting tick_errors", consec_tick_errors)
                    exit_reason = "tick_errors"
                    break
                time.sleep(max(0.0, args.tick_s - (time.time() - loop_start)))
                continue

            consec_tick_errors = 0
            if awaiting_clean_resume_tick:
                # First post-resume tick completed without raising -- release
                # the forced manual_override held since the resume protocol
                # found position discrepancies (W0.1). resume_discrepancies
                # itself is left intact for the heartbeat/alert record.
                awaiting_clean_resume_tick = False
            repriced = engine.latencies[n_lat_before:] if len(engine.latencies) > n_lat_before else []
            snap = loop.last_snapshot
            fv = loop.last_fair_value
            directives = loop.last_directives or {}
            quote_sets = loop.last_quote_sets or {}
            noarb_ok = loop.last_checked_quote_sets is not None
            if not noarb_ok:
                noarb_violations += 1

            for slug, strike in markets:
                ms = loop.books[slug]
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

            for f in loop.last_fills:
                n_fills_total += 1
                fw.writerow([
                    f.ts.isoformat(), f.market_id, f.order_id, f.side.name,
                    f"{f.price:.4f}", f"{f.size:.2f}", f.liquidity.name,
                    f"{getattr(f, 'queue_ahead_at_fill', float('nan')):.2f}",
                    f"{getattr(f, 'print_size', float('nan')):.2f}",
                    getattr(f, "latency_applied_ms", ""),
                    getattr(f, "mid_at_fill", ""), getattr(f, "assumption_set", ""),
                ])
                logger.info("FILL %s %s %.2f @ %.4f", f.market_id, f.side.name, f.size, f.price)

            tw.writerow([
                now.isoformat(), tick_n, f"{time.time() - loop_start:.2f}",
                f"{repriced[0]:.1f}" if repriced else "",
                int(feed_healthy), n_msgs, len(loop.last_fills),
                int(loop.snapshot_failed),
            ])
            quotes_csv.flush()
            fills_csv.flush()
            ticks_csv.flush()

            # B2: settlement, gated on the 12:00 ET expiry instant and on not
            # every market already being terminally settled (loop.settle is
            # idempotent regardless, this just avoids the store round trip
            # every tick once the ladder is fully resolved).
            settlement_instant = settlement_instant_utc(expiry_key)
            try:
                if now >= settlement_instant and not _all_settled_terminal(store, markets, expiry_key):
                    loop.settle(now)
            except Exception:
                logger.error("settlement step failed at tick %d", tick_n, exc_info=True)

            # F5: ONE guarded fills fetch, hoisted above both the pnl and
            # markout blocks below -- they used to each call store.get_fills()
            # independently (redundant on every report tick); a bare reuse
            # would have left fills_all unbound in the markout block whenever
            # the pnl block's own fetch raised, so this fetch has its own
            # try/except and both consumers explicitly skip on None rather
            # than depending on each other's internals.
            try:
                fills_all = store.get_fills()
            except Exception:
                fills_all = None
                logger.warning("get_fills failed at tick %d", tick_n, exc_info=True)

            # R3 invariant: PnL snapshot MUST run AFTER settle so realized
            # (from cash+avg_cost, B3) and inventory stay in sync -- a
            # settlement pseudo-fill's cash and its q->0 inventory update
            # must land in the SAME snapshot, never split across ticks.
            try:
                if fills_all is not None:
                    inv_all = store.get_all_inventory()
                    mids_by_market: Dict[str, Optional[float]] = {}
                    consensus_by_market: Dict[str, Optional[float]] = {}
                    for m, k in markets:
                        mids_by_market[m] = _book_mid(loop.books[m])
                        consensus_by_market[m] = fv.consensus_p.get(k) if fv is not None else None
                    settlements = store.get_all_settlements()
                    pnl_rows = compute_pnl_rows(
                        now, expiry_key, fills_all, inv_all, mids_by_market, consensus_by_market,
                        args.bankroll, settlements=settlements,
                    )
                    # W4 row-volume cap: TOTAL row every tick, per-market rows
                    # only every Nth tick.
                    for row in pnl_rows:
                        if row.market_id is None or tick_n % PER_MARKET_SNAPSHOT_EVERY_N_TICKS == 0:
                            store.append_pnl_snapshot(row)
            except Exception:
                logger.warning("pnl snapshot failed at tick %d", tick_n, exc_info=True)

            # C4 (mm_suitability_alignment_plan.md Change C): per-region
            # markout report, same PER_MARKET_SNAPSHOT_EVERY_N_TICKS cadence
            # as the per-market PnL rows above (row-volume cap, W4) -- a
            # fresh independent try/except so a markout failure never blocks
            # the pnl snapshot (or vice versa). F3: `now` is passed through so
            # the report's own lookback window shares the SAME anchor as the
            # prune bound below (a defaulted now=max-fill-ts here would drift
            # from the prune's wall-clock anchor on a stale-fill run); pruning
            # happens only after the report using the still-unpruned rows has
            # been written, and at this cadence (not every tick) so the hot
            # path is unaffected.
            if tick_n % PER_MARKET_SNAPSHOT_EVERY_N_TICKS == 0:
                try:
                    if fills_all is not None:
                        report = markout_report(
                            fills_all, store.mid_at_or_after,
                            store.get_market_registry(), loop.config.belly_band,
                            now=now,
                        )
                        _write_json_atomic(out_dir / "markout_report.json", report)
                        store.prune_mid_log(now - timedelta(seconds=MARKOUT_LOOKBACK_S))
                except Exception:
                    logger.warning("markout report failed at tick %d", tick_n, exc_info=True)

                # W0.2: prune the quotes table at the same per-market snapshot
                # cadence, own try/except so a failure here never blocks the
                # markout report above (or vice versa).
                try:
                    store.prune_quotes(now - timedelta(seconds=loop.config.quotes_retention_s))
                except Exception:
                    logger.warning("prune_quotes failed at tick %d", tick_n, exc_info=True)

                # 2026-07-11: prune the trade_prints table (arrival-decay
                # calibration input, see scripts/mm_calibrate_k.py) on the
                # same cadence and retention as quotes, own try/except so a
                # failure here never blocks the blocks above.
                try:
                    store.prune_trade_prints(now - timedelta(seconds=loop.config.quotes_retention_s))
                except Exception:
                    logger.warning("prune_trade_prints failed at tick %d", tick_n, exc_info=True)

                # W1.1: persist this tick's per-market LiquidityState (the
                # dead liquidity_windows table, activated) at the same
                # per-market snapshot cadence, own try/except per market so
                # one market's failure never blocks its neighbors or the
                # blocks above. Pruned with the same retention as quotes.
                for m, _k in markets:
                    liq = loop.last_liquidity.get(m)
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
                    store.prune_liquidity_windows(now - timedelta(seconds=loop.config.quotes_retention_s))
                except Exception:
                    logger.warning("prune_liquidity_windows failed at tick %d", tick_n, exc_info=True)

                # W2.4: persist this expiry's ladder state (net_band_exposure,
                # gross, phi, r3_histogram, plus the market_id-keyed pending
                # hedge-demand offsets from W2.1's hedge stage) at the same
                # per-market snapshot cadence, own try/except so a failure
                # here never blocks the blocks above.
                try:
                    store.upsert_ladder_state(
                        expiry_key, loop.inv.snapshot(now).per_ladder[expiry_key],
                        vertical_offsets=loop.last_hedge_offsets,
                    )
                except Exception:
                    logger.warning("upsert_ladder_state failed at tick %d", tick_n, exc_info=True)

            try:
                _write_heartbeat(out_dir, now, tick_n, feed_healthy, n_msgs, n_fills_total,
                                  noarb_violations, unhealthy_ticks, pulled_ticks,
                                  args.tick_s, args.reprice_s,
                                  btc_data_age_s=btc_data_age_s, feed_restarts=feed_restarts,
                                  resume_discrepancies=resume_discrepancies,
                                  bankroll_frozen=loop.bankroll_state.frozen)
            except Exception:
                logger.warning("heartbeat write failed", exc_info=True)

            # 2.2: exit-after-settlement + rollover. A 30min grace after the
            # settlement instant lets the settlement step above (and any
            # retry) actually land before declaring the ladder done; the
            # hard --max-settlement-wait-h fallback prevents a stuck
            # UNSETTLEABLE market from blocking rollover for the rest of the
            # month (unsettled positions stay open in the persisted DB and
            # are retried by the next run's resume catch-up, WS1.4).
            if (now >= settlement_instant + timedelta(minutes=30)
                    and _all_settled_terminal(store, markets, expiry_key)):
                exit_reason = "ladder_settled"
                break
            if now >= settlement_instant + timedelta(hours=args.max_settlement_wait_h):
                exit_reason = "settlement_timeout"
                break

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, args.tick_s - elapsed))
    except KeyboardInterrupt:
        exit_reason = "sigint"
    finally:
        # L4: best-effort cancel of any still-LIVE paper orders so no DB rows
        # are left LIVE after shutdown (crash-before-cancel would otherwise
        # confuse the next resume's restart_reconcile).
        if loop is not None:
            try:
                loop.lifecycle.cancel_all()
            except Exception:
                logger.warning("cancel_all() failed during shutdown", exc_info=True)
        if adapter is not None:
            adapter.stop()
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

    if loop is None:
        # Failed before the loop was constructed (e.g. resolve_event raised
        # and was somehow swallowed above, or KeyboardInterrupt landed very
        # early) -- nothing to summarize.
        return 1

    fold_ok = loop.fold_matches_inventory()
    inv_lines = []
    inv_snap = loop.inv.snapshot(loop.clock.now())
    for m, _k in markets:
        ci = inv_snap.per_contract.get(m)
        if ci is not None and ci.q != 0.0:
            inv_lines.append(f"    - {m}: q={ci.q:.2f} avg_cost={ci.avg_cost:.4f}")
    summary = out_dir / "summary.md"
    summary.write_text(
        "# Stage-B paper run summary\n\n"
        f"- event: {args.event_slug} (expiry {expiry_key}, {len(markets)} strikes)\n"
        f"- start: {start.isoformat()}  duration: {args.minutes} min "
        f"({'indefinite' if args.minutes == 0 else 'fixed'}), "
        f"tick {args.tick_s}s, reprice {args.reprice_s}s, bankroll {args.bankroll}\n"
        f"- exit_reason: {exit_reason}\n"
        f"- ticks: {tick_n} (feed-unhealthy ticks: {unhealthy_ticks})\n"
        f"- simulated fills: {n_fills_total}\n"
        f"- ending open inventory:\n" + ("\n".join(inv_lines) or "    - flat") + "\n"
        f"- fold(fills) == inventory: {fold_ok}\n"
        f"- self-inflicted no-arb violations (post-repair): {noarb_violations}\n"
        f"- PULLED (market,tick) rows: {pulled_ticks}\n"
        f"- re-price latencies (s): {['%.1f' % v for v in engine.latencies]}\n"
        f"- feed restarts: {feed_restarts}\n"
        f"- data: quotes.csv / fills.csv / ticks.csv in this directory; state db: {state_db_path}\n",
        encoding="ascii",
    )
    logger.info("paper run complete: %s (exit_reason=%s)", out_dir, exit_reason)
    print(str(out_dir))
    # 2.2 exit-code mapping: ladder_settled/settlement_timeout -> 42 (systemd
    # RestartForceExitStatus rolls to the next event); feed_dead/tick_errors
    # -> 1 (supervised restart, clean re-init); everything else -- completed,
    # stop_file, sigterm, sigint -- -> 0 (no restart on an intentional stop).
    if exit_reason in ("ladder_settled", "settlement_timeout"):
        return 42
    if exit_reason in ("feed_dead", "tick_errors"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
