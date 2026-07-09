#!/usr/bin/env python3
"""Stage-B paper-runner alert check (plan Workstream 3 / H3).

Stdlib-only, cron/systemd-timer-safe: this script ALWAYS exits 0. Every
failure mode (missing control dir, torn JSON, unreachable webhook, ...) is
caught and printed rather than raised, so a bug here can never itself look
like a fault (and can never break the timer unit).

What it does, once per invocation:
  1. `market_maker.run_control.engine_status()` -> RUNNING / STARTING /
     STALLED / STOPPED / CRASHED, plus the current run's heartbeat.json
     (if any) and current_run.json (`run_info`).
  2. Evaluate a fixed set of fault conditions against that snapshot (see
     `_collect_alerts`):
       - engine state is CRASHED or STALLED
       - heartbeat.feed_healthy has been False for more than 15 minutes
         (the heartbeat itself has no streak field, so the "since when" is
         tracked across invocations in the alert state file below)
       - heartbeat.btc_data_age_s exceeds 2x the runner's --btc-stale-max-s
         (default 7200s, so alert threshold is 14400s)
       - heartbeat.resume_discrepancies > 0 (the resume protocol found a
         store/venue position mismatch on restart -- plan Wave 0 W0.1)
       - heartbeat.bankroll_frozen is true (the Beuoy bankroll degenerated
         and has not yet auto-unfrozen -- plan Wave 1 W1.3)
       - free disk on the repo-root filesystem is below 1 GB
       - the engine is STOPPED and the last run's current_run.json recorded
         exit_reason == "settlement_timeout" (an UNSETTLEABLE ladder that
         needs operator attention -- see deploy/README.md)
  3. For each triggered condition, POST `{"text": "<message>"}` to
     `$MM_ALERT_WEBHOOK` (a generic JSON webhook body -- works with Discord/
     Slack/ntfy-style relays). If the env var is unset, the message is
     printed to stdout instead (still visible via `journalctl -u mm-alert`
     under systemd) -- this counts as "delivered" for de-dupe purposes.
  4. De-dupe: each condition has a stable key; a key that fired within the
     last 6h is suppressed, so a persistent fault pages once, not every 5
     minutes. State (last-sent timestamp per key, plus the feed-health
     streak baseline) is persisted to `<control-dir>/alert_state.json`.
  5. Daily heartbeat: once per UTC day, at the first invocation at/after
     08:00 UTC (override via $MM_HEARTBEAT_HOUR_UTC; disable via
     $MM_HEARTBEAT_DISABLE=1), a one-line status summary is sent through the
     same webhook regardless of engine state -- so webhook silence means
     "alert pipeline dead", never "nothing to say". Not subject to the 6h
     de-dupe; tracked separately as `heartbeat_last_date` in the state file.

Usage (see deploy/mm-alert.service / deploy/mm-alert.timer):
    python scripts/mm_alert_check.py [--control-dir temp/paper_run/control]
                                      [--btc-stale-max-s 7200]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Guard: ensure repo root is on sys.path regardless of invocation cwd (systemd
# runs this with WorkingDirectory=<repo>, but `python scripts/mm_alert_check.py`
# alone only puts this script's own directory on sys.path[0] -- mirrors
# scripts/pipelines/batch_pricing_runner.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BTC_STALE_MAX_S_DEFAULT = 7200.0
BTC_STALE_ALERT_MULT = 2.0
FEED_UNHEALTHY_ALERT_S = 15 * 60.0
DISK_FREE_MIN_BYTES = 1 * 1024 ** 3
DEDUPE_WINDOW_S = 6 * 3600.0
ALERT_STATE_FILENAME = "alert_state.json"
# Daily "still alive" heartbeat: sent once per UTC day at the first timer
# tick at/after this hour, regardless of engine state, so webhook silence is
# distinguishable from a dead alert pipeline. Override the hour with
# $MM_HEARTBEAT_HOUR_UTC; set $MM_HEARTBEAT_DISABLE=1 to turn it off.
HEARTBEAT_HOUR_UTC_DEFAULT = 8


# ---------------------------------------------------------------------------
# alert state file (dedupe timestamps + feed-health streak baseline)
# ---------------------------------------------------------------------------


def _load_alert_state(control_dir: Path) -> Dict[str, Any]:
    path = control_dir / ALERT_STATE_FILENAME
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def _save_alert_state(control_dir: Path, state: Dict[str, Any]) -> None:
    path = control_dir / ALERT_STATE_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="ascii")
    os.replace(str(tmp), str(path))


def _should_send(key: str, state: Dict[str, Any], now: float,
                  window_s: float = DEDUPE_WINDOW_S) -> bool:
    """True if `key` was not sent within the last `window_s` seconds."""
    sent = state.get("alerts_sent")
    if not isinstance(sent, dict):
        return True
    last = sent.get(key)
    if last is None:
        return True
    try:
        return (now - float(last)) >= window_s
    except (TypeError, ValueError):
        return True


def _mark_sent(key: str, state: Dict[str, Any], now: float) -> None:
    sent = state.setdefault("alerts_sent", {})
    if not isinstance(sent, dict):
        sent = {}
        state["alerts_sent"] = sent
    sent[key] = now


# ---------------------------------------------------------------------------
# individual fault checks (pure functions, no I/O -- unit-testable)
# ---------------------------------------------------------------------------


def _check_engine_state(status: Any) -> Optional[Tuple[str, str]]:
    if status.state in ("CRASHED", "STALLED"):
        return ("state_%s" % status.state.lower(),
                "mm-paper engine state=%s: %s" % (status.state, status.detail))
    return None


def _check_feed_unhealthy(
    heartbeat: Optional[Dict[str, Any]], state: Dict[str, Any], now: float,
    threshold_s: float = FEED_UNHEALTHY_ALERT_S,
) -> Optional[Tuple[str, str]]:
    """feed_healthy=False for more than `threshold_s`. The heartbeat carries
    only the current flag, not a streak, so the "last seen healthy" instant
    is tracked in `state` across invocations (mutates `state` in place)."""
    if not isinstance(heartbeat, dict):
        return None
    if heartbeat.get("feed_healthy"):
        state["last_feed_healthy_ts"] = now
        return None
    last_healthy = state.get("last_feed_healthy_ts")
    if last_healthy is None:
        # First time we've ever observed feed_healthy=False -- start the
        # clock now rather than assuming it has already been down forever.
        state["last_feed_healthy_ts"] = now
        return None
    try:
        age = now - float(last_healthy)
    except (TypeError, ValueError):
        state["last_feed_healthy_ts"] = now
        return None
    if age > threshold_s:
        return ("feed_unhealthy",
                "feed_healthy has been False for %.0fs (> %.0fs threshold)" % (age, threshold_s))
    return None


def _check_btc_stale(
    heartbeat: Optional[Dict[str, Any]], btc_stale_max_s: float = BTC_STALE_MAX_S_DEFAULT,
) -> Optional[Tuple[str, str]]:
    if not isinstance(heartbeat, dict):
        return None
    age = heartbeat.get("btc_data_age_s")
    if not isinstance(age, (int, float)):
        return None
    threshold = btc_stale_max_s * BTC_STALE_ALERT_MULT
    if age > threshold:
        return ("btc_stale",
                "BTC intraday data is %.0fs stale (> %.0fs = 2x --btc-stale-max-s)" % (age, threshold))
    return None


def _check_resume_discrepancies(heartbeat: Optional[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """heartbeat.resume_discrepancies > 0 -- the resume protocol
    (loop.restart()) found a store/venue position mismatch (plan Wave 0
    W0.1). De-duped like every other check (6h window); the count itself is
    NOT a streak (it is fixed for the life of the run, unlike feed health),
    so no cross-invocation state is needed here."""
    if not isinstance(heartbeat, dict):
        return None
    n = heartbeat.get("resume_discrepancies")
    if not isinstance(n, (int, float)) or n <= 0:
        return None
    return ("resume_discrepancies",
            "mm-paper resume found %d position discrepancy(ies) on restart -- "
            "check the risk journal for MANUAL-trigger PULLED entries" % int(n))


def _check_bankroll_frozen(heartbeat: Optional[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """heartbeat.bankroll_frozen is true -- the Beuoy bankroll credibility
    consensus has degenerated and quoting has fallen back to
    FIXED_BLEND_FALLBACK until enough consecutive clean BEUOY ticks
    auto-unfreeze it (plan Wave 1 W1.3). De-duped like every other check (6h
    window) -- the flag is a level, not a streak, so no cross-invocation
    state is needed here."""
    if not isinstance(heartbeat, dict):
        return None
    if not heartbeat.get("bankroll_frozen"):
        return None
    return ("bankroll_frozen",
            "mm-paper bankroll is FROZEN -- Beuoy consensus degenerated; quoting is on "
            "FIXED_BLEND_FALLBACK until enough consecutive clean BEUOY ticks auto-unfreeze it")


def _check_disk_free(repo_root: Path) -> Optional[Tuple[str, str]]:
    try:
        free = shutil.disk_usage(str(repo_root)).free
    except OSError:
        return None
    if free < DISK_FREE_MIN_BYTES:
        return ("disk_low",
                "free disk on %s is %.2f GB (< 1 GB threshold)" % (repo_root, free / (1024 ** 3)))
    return None


def _check_settlement_timeout(status: Any) -> Optional[Tuple[str, str]]:
    if status.state != "STOPPED":
        return None
    run_info = status.run_info
    if not isinstance(run_info, dict):
        return None
    if run_info.get("exit_reason") == "settlement_timeout":
        return ("settlement_timeout",
                "mm-paper stopped with exit_reason=settlement_timeout -- a market did not "
                "settle within --max-settlement-wait-h; unsettled positions stay open in the "
                "state db and are retried on the next start's resume catch-up, but this needs "
                "operator attention (see deploy/README.md)")
    return None


def _heartbeat_due(state: Dict[str, Any], now_utc: "datetime",
                   hour_utc: int = HEARTBEAT_HOUR_UTC_DEFAULT) -> bool:
    """True if today's (UTC) heartbeat has not been sent yet and the current
    UTC hour is at/after the send hour. Tracked as an ISO date string in the
    alert state file (`heartbeat_last_date`), so exactly one heartbeat goes
    out per UTC day no matter how often the timer fires."""
    if now_utc.hour < hour_utc:
        return False
    return state.get("heartbeat_last_date") != now_utc.date().isoformat()


def _mark_heartbeat_sent(state: Dict[str, Any], now_utc: "datetime") -> None:
    state["heartbeat_last_date"] = now_utc.date().isoformat()


def _heartbeat_message(status: Any, repo_root: Path) -> str:
    """One-line daily status summary. Pure formatting -- tolerates a missing
    heartbeat dict (engine stopped/crashed) and missing fields."""
    parts = ["daily heartbeat: state=%s" % status.state]
    run_info = status.run_info if isinstance(status.run_info, dict) else {}
    if status.state == "STOPPED" and run_info.get("exit_reason"):
        parts.append("exit_reason=%s" % run_info.get("exit_reason"))
    hb = status.heartbeat if isinstance(status.heartbeat, dict) else {}
    if hb:
        parts.append("tick=%s" % hb.get("tick"))
        parts.append("feed_healthy=%s" % hb.get("feed_healthy"))
        parts.append("fills=%s" % hb.get("fills_total"))
        age = hb.get("btc_data_age_s")
        if isinstance(age, (int, float)):
            parts.append("btc_age=%.0fs" % age)
        parts.append("feed_restarts=%s" % hb.get("feed_restarts"))
    try:
        free_gb = shutil.disk_usage(str(repo_root)).free / (1024 ** 3)
        parts.append("disk_free=%.1fGB" % free_gb)
    except OSError:
        pass
    return " ".join(parts)


def _collect_alerts(
    status: Any, state: Dict[str, Any], now: float, repo_root: Path,
    btc_stale_max_s: float = BTC_STALE_MAX_S_DEFAULT,
) -> List[Tuple[str, str]]:
    """Run every check; `state` is mutated in place by the feed-health check
    (streak baseline) regardless of whether an alert fires."""
    checks = [
        _check_engine_state(status),
        _check_feed_unhealthy(status.heartbeat, state, now),
        _check_btc_stale(status.heartbeat, btc_stale_max_s),
        _check_resume_discrepancies(status.heartbeat),
        _check_bankroll_frozen(status.heartbeat),
        _check_disk_free(repo_root),
        _check_settlement_timeout(status),
    ]
    return [c for c in checks if c is not None]


# ---------------------------------------------------------------------------
# delivery
# ---------------------------------------------------------------------------


def _send_webhook(message: str) -> bool:
    """POST {"text": message} to $MM_ALERT_WEBHOOK. Prints to stdout and
    returns True (delivered) if the env var is unset. Returns False only on
    an actual send failure (network/HTTP error), so the caller can leave the
    dedupe timestamp unset and retry on the next timer tick."""
    webhook = os.environ.get("MM_ALERT_WEBHOOK")
    if not webhook:
        print("MM_ALERT (no MM_ALERT_WEBHOOK set): %s" % message)
        return True
    try:
        payload = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            webhook, data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as exc:  # noqa: BLE001 - deliberately broad, never raise
        print("MM_ALERT: failed to POST webhook (%s); message: %s" % (exc, message))
        return False


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _run(argv: Optional[List[str]]) -> None:
    from market_maker import run_control

    ap = argparse.ArgumentParser(description="Stage-B paper-runner alert check (timer-safe)")
    ap.add_argument("--control-dir", default=str(run_control.CONTROL_DIR),
                     help="matches paper_runner --control-dir; default is run_control.CONTROL_DIR")
    ap.add_argument("--btc-stale-max-s", type=float, default=BTC_STALE_MAX_S_DEFAULT,
                     help="matches paper_runner --btc-stale-max-s; alert fires at 2x this value")
    args = ap.parse_args(argv)

    control_dir = Path(args.control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)

    status = run_control.engine_status(control_dir=control_dir)
    state = _load_alert_state(control_dir)
    now = time.time()

    alerts = _collect_alerts(status, state, now, _REPO_ROOT, args.btc_stale_max_s)

    for key, message in alerts:
        if not _should_send(key, state, now):
            continue
        if _send_webhook("[mm-paper] %s" % message):
            _mark_sent(key, state, now)

    if os.environ.get("MM_HEARTBEAT_DISABLE") != "1":
        try:
            hb_hour = int(os.environ.get("MM_HEARTBEAT_HOUR_UTC", HEARTBEAT_HOUR_UTC_DEFAULT))
        except ValueError:
            hb_hour = HEARTBEAT_HOUR_UTC_DEFAULT
        now_utc = datetime.now(timezone.utc)
        if _heartbeat_due(state, now_utc, hb_hour):
            if _send_webhook("[mm-paper] %s" % _heartbeat_message(status, _REPO_ROOT)):
                _mark_heartbeat_sent(state, now_utc)

    _save_alert_state(control_dir, state)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        _run(argv)
    except Exception as exc:  # noqa: BLE001 - this script must never raise
        print("mm_alert_check: unhandled error: %s" % exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
