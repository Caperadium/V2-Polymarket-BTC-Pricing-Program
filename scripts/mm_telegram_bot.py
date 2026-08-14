#!/usr/bin/env python3
"""Telegram slash-command bot for the Stage-B paper runner.

Stdlib-only, long-polling daemon (Telegram getUpdates; no inbound HTTP
endpoint needed, so it runs fine behind the VPS NAT). Read-only counterpart
to scripts/mm_alert_check.py: the alert check PUSHES faults to the webhook,
this bot ANSWERS operator queries about current engine metrics.

Commands
--------
  /status     engine state (run_control), tick, feed health, per-expiry lines
              + compact C1 shadow belly pricer weights when shadow rows exist
  /bankroll   initial bankroll + current equity (latest pnl TOTAL row)
              + rebates accrued (est, maker-rebate accounting layer)
              + C1 belly pricer weight per expiry (applied/shadow/control,
                bankrolls regions belly / belly_drift_shadow /
                belly_legacy_control -- shadow rows exist only while
                MMConfig.belly_score_mode == "shadow")
  /pnl        realized / unrealized / settlement breakdown of the TOTAL row
  /fills      fill counts (maker/taker/settlement, last 24h) + last fill
  /inventory  open positions (q != 0) with strike/expiry
  /quotes     latest resting quote per market (bid/ask/spread, staleness)
  /markout    by_region rollup of <out_dir>/markout_report.json
  /help       command list

Data sources (all read-only; this bot NEVER writes engine state):
  - market_maker.run_control.engine_status()  (control-dir files + heartbeat)
  - <out_dir>/run_meta.json                   (initial bankroll, state_db path)
  - the MMStateStore sqlite db, opened with sqlite3 URI mode=ro (WAL allows
    concurrent readers; falls back to a default connect if the ro open fails,
    e.g. no -shm yet)
  - <out_dir>/markout_report.json

Credentials
-----------
  $MM_TELEGRAM_TOKEN / $MM_TELEGRAM_CHAT_ID take precedence; when unset, both
  are parsed out of $MM_ALERT_WEBHOOK (the deploy kit's existing
  https://api.telegram.org/bot<token>/sendMessage?chat_id=<id> URL), so the
  VPS needs no new secret plumbing. The chat id is a hard allowlist: messages
  from any other chat are silently ignored (never answered), since anyone can
  discover and message a Telegram bot.

Resilience: the outer loop never raises -- every failure mode (venue/network
error, torn JSON, missing db) is caught, printed (journalctl-visible) and
retried with a backoff. The getUpdates offset is persisted to
<control-dir>/telegram_bot_state.json so a restart does not replay old
commands.

Usage (see deploy/mm-telegram.service):
    python scripts/mm_telegram_bot.py [--control-dir temp/paper_run/control]
                                      [--state-db path/to/db] [--once]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Guard: ensure repo root is on sys.path regardless of invocation cwd
# (mirrors scripts/mm_alert_check.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BOT_STATE_FILENAME = "telegram_bot_state.json"
POLL_TIMEOUT_S_DEFAULT = 50
ERROR_BACKOFF_S = 10.0
REPLY_MAX_CHARS = 3900  # Telegram hard limit is 4096; leave headroom
QUOTE_FRESH_WINDOW_S = 15 * 60.0
DEFAULT_CONFIG_JSON = _REPO_ROOT / "market_maker" / "paper_run_config.json"


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------


def parse_webhook_creds(webhook_url: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (bot_token, chat_id) from a Telegram sendMessage webhook URL.
    Returns (None, None) for non-Telegram URLs."""
    token_m = re.search(r"/bot([0-9]+:[A-Za-z0-9_-]+)/", webhook_url)
    chat_m = re.search(r"[?&]chat_id=(-?[0-9]+)", webhook_url)
    return (token_m.group(1) if token_m else None,
            chat_m.group(1) if chat_m else None)


def resolve_creds() -> Tuple[Optional[str], Optional[str]]:
    token = os.environ.get("MM_TELEGRAM_TOKEN") or None
    chat_id = os.environ.get("MM_TELEGRAM_CHAT_ID") or None
    if token and chat_id:
        return token, chat_id
    hook = os.environ.get("MM_ALERT_WEBHOOK") or ""
    hook_token, hook_chat = parse_webhook_creds(hook)
    return token or hook_token, chat_id or hook_chat


# ---------------------------------------------------------------------------
# metrics source (all reads; unit-testable seam)
# ---------------------------------------------------------------------------


class MetricsSource:
    """Read-only access to the engine's observable state. Every method
    tolerates missing/torn files and a missing db by returning None/empty --
    the command handlers turn that into a human 'no data' line."""

    def __init__(self, control_dir: Path, repo_root: Path = _REPO_ROOT,
                 state_db_override: Optional[Path] = None):
        self.control_dir = Path(control_dir)
        self.repo_root = Path(repo_root)
        self.state_db_override = state_db_override

    # -- control-dir / run files --------------------------------------------

    def status(self) -> Any:
        from market_maker import run_control
        return run_control.engine_status(control_dir=self.control_dir)

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.repo_root / path

    def out_dir(self, status: Any = None) -> Optional[Path]:
        status = status or self.status()
        run_info = status.run_info if isinstance(status.run_info, dict) else {}
        raw = run_info.get("out_dir")
        return self._resolve(str(raw)) if raw else None

    def run_meta(self, status: Any = None) -> Dict[str, Any]:
        out = self.out_dir(status)
        if out is None:
            return {}
        return self._read_json(out / "run_meta.json") or {}

    def markout_json(self, status: Any = None) -> Optional[Dict[str, Any]]:
        out = self.out_dir(status)
        if out is None:
            return None
        return self._read_json(out / "markout_report.json")

    # -- state db -------------------------------------------------------------

    def db_path(self, status: Any = None) -> Optional[Path]:
        if self.state_db_override is not None:
            return self.state_db_override
        status = status or self.status()
        meta = self.run_meta(status)
        raw = meta.get("state_db")
        if raw:
            return self._resolve(str(raw))
        out = self.out_dir(status)
        if out is not None and (out / "paper_state.db").exists():
            return out / "paper_state.db"
        cfg = self._read_json(DEFAULT_CONFIG_JSON) or {}
        if cfg.get("state_db"):
            return self._resolve(str(cfg["state_db"]))
        return None

    def query(self, sql: str, params: Sequence[Any] = ()) -> Optional[List[Tuple]]:
        """Run one read-only query against the state db. None = no db found
        or unreadable (distinct from an empty result set)."""
        db = self.db_path()
        if db is None or not Path(db).exists():
            return None
        uri = "file:%s?mode=ro" % urllib.parse.quote(str(db).replace("\\", "/"))
        conn = None
        try:
            try:
                conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            except sqlite3.OperationalError:
                # ro open can fail on a WAL db with no -shm yet; fall back to
                # a normal (still non-writing) connection.
                conn = sqlite3.connect(str(db), timeout=5.0)
            return conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:
            print("mm_telegram_bot: db query failed: %s" % exc)
            return None
        finally:
            if conn is not None:
                conn.close()


# ---------------------------------------------------------------------------
# small formatting helpers
# ---------------------------------------------------------------------------


def _money(x: Any) -> str:
    try:
        return "%.2f" % float(x)
    except (TypeError, ValueError):
        return "?"


def _smoney(x: Any) -> str:
    try:
        return "%+.2f" % float(x)
    except (TypeError, ValueError):
        return "?"


def _short_ts(iso: Any) -> str:
    try:
        dt = datetime.fromisoformat(str(iso))
        return dt.strftime("%m-%d %H:%M:%SZ")
    except (TypeError, ValueError):
        return str(iso)


def _hb(status: Any) -> Dict[str, Any]:
    return status.heartbeat if isinstance(status.heartbeat, dict) else {}


# ---------------------------------------------------------------------------
# command handlers (each: MetricsSource -> str)
# ---------------------------------------------------------------------------


_C1_REGIONS = ("belly", "belly_drift_shadow", "belly_legacy_control")


def _c1_belly_weights(src: MetricsSource) -> Dict[str, Dict[str, Tuple[float, int]]]:
    """Latest belly pricer weight per (expiry, region) from the bankrolls
    table: {expiry_key: {region: (pricer_weight, update_count)}}. Empty dict
    when no C1 shadow rows exist (mode legacy/live, or no db) -- callers skip
    their section entirely then. Weight = pricer bankroll / sum (rows are
    stored normalized; the division is a defensive no-op)."""
    rows = src.query(
        "SELECT b.expiry_key, b.region, b.bankrolls, b.update_count "
        "FROM bankrolls b JOIN (SELECT expiry_key, region, MAX(id) AS mid "
        "  FROM bankrolls WHERE region IN (?, ?, ?) GROUP BY expiry_key, region) m "
        "ON b.id = m.mid", _C1_REGIONS)
    out: Dict[str, Dict[str, Tuple[float, int]]] = {}
    for ek, region, bk_json, n_upd in rows or []:
        try:
            bk = json.loads(bk_json)
            total = sum(float(v) for v in bk.values())
            w = float(bk.get("pricer", 0.0)) / total if total > 0 else float("nan")
        except (ValueError, TypeError, AttributeError):
            continue
        out.setdefault(str(ek), {})[str(region)] = (w, int(n_upd or 0))
    # Only report when at least one SHADOW row exists -- the plain 'belly'
    # region always has rows and alone is not C1 information.
    if not any("belly_drift_shadow" in regions for regions in out.values()):
        return {}
    return out


def cmd_status(src: MetricsSource) -> str:
    status = src.status()
    hb = _hb(status)
    lines = ["engine: %s (%s)" % (status.state, status.detail)]
    if hb:
        lines.append("tick=%s feed_healthy=%s fills_total=%s" % (
            hb.get("tick"), hb.get("feed_healthy"), hb.get("fills_total")))
        age = hb.get("btc_data_age_s")
        lines.append("btc_age=%s pulled_ticks=%s noarb=%s feed_restarts=%s" % (
            ("%.0fs" % age) if isinstance(age, (int, float)) else "?",
            hb.get("pulled_ticks"), hb.get("noarb_violations"), hb.get("feed_restarts")))
        if hb.get("bankroll_frozen"):
            lines.append("WARNING: bankroll FROZEN")
        expiries = hb.get("expiries")
        if isinstance(expiries, dict) and expiries:
            lines.append("expiries active: %s" % hb.get("n_expiries_active"))
            for ek in sorted(expiries):
                e = expiries[ek]
                lines.append("  %s %s feed=%s fills=%s%s" % (
                    ek, e.get("state"),
                    "ok" if e.get("feed_healthy") else "DOWN",
                    e.get("fills"),
                    " FROZEN" if e.get("bankroll_frozen") else ""))
    else:
        run_info = status.run_info if isinstance(status.run_info, dict) else {}
        if run_info.get("exit_reason"):
            lines.append("last exit_reason: %s" % run_info.get("exit_reason"))
    c1 = _c1_belly_weights(src)
    if c1:
        parts = []
        for ek in sorted(c1):
            sh = c1[ek].get("belly_drift_shadow")
            if sh is not None:
                parts.append("%s %.2f(n%d)" % (ek[5:], sh[0], sh[1]))
        if parts:
            lines.append("C1 shadow belly w: %s" % ", ".join(parts))
    return "\n".join(lines)


def cmd_bankroll(src: MetricsSource) -> str:
    status = src.status()
    meta = src.run_meta(status)
    initial = meta.get("bankroll")
    rows = src.query(
        "SELECT ts, realized, unrealized_mid, bankroll_utilization "
        "FROM pnl WHERE market_id IS NULL ORDER BY id DESC LIMIT 1")
    lines = ["bankroll (initial: %s)" % _money(initial)]
    if rows is None:
        lines.append("no state db found -- cannot compute equity")
    elif not rows:
        lines.append("no pnl snapshots yet; equity = initial")
    else:
        ts, realized, unreal_mid, util = rows[0]
        if isinstance(initial, (int, float)):
            equity = float(initial) + float(realized) + float(unreal_mid)
            lines.append("equity: %s" % _money(equity))
        lines.append("realized: %s  unrealized(mid): %s" % (
            _smoney(realized), _smoney(unreal_mid)))
        try:
            lines.append("at-risk utilization: %.1f%%" % (float(util) * 100.0))
        except (TypeError, ValueError):
            pass
        lines.append("as of %s" % _short_ts(ts))
    reb = src.query(
        # 0.014 = MAKER_REBATE_SHARE_CRYPTO * TAKER_FEE_RATE_CRYPTO
        # (market_maker/config.py) -- duplicated here because this script is
        # stdlib-only by design; keep in sync.
        "SELECT COALESCE(SUM(0.014 * price * (1.0 - price) * size), 0.0) "
        "FROM fills WHERE liquidity = 'MAKER'")
    if reb:
        lines.append("rebates accrued (est, not in equity): %s" % _money(reb[0][0]))
    if _hb(status).get("bankroll_frozen"):
        lines.append("WARNING: Beuoy bankroll FROZEN (fixed-blend fallback)")
    c1 = _c1_belly_weights(src)
    if c1:
        lines.append("C1 belly pricer w (applied/shadow/control):")
        for ek in sorted(c1):
            regions = c1[ek]

            def _fmt(region: str) -> str:
                v = regions.get(region)
                return "%.2f" % v[0] if v is not None else "-"

            sh = regions.get("belly_drift_shadow")
            lines.append("  %s: %s / %s / %s (events %d)" % (
                ek, _fmt("belly"), _fmt("belly_drift_shadow"),
                _fmt("belly_legacy_control"), sh[1] if sh else 0))
    return "\n".join(lines)


def cmd_pnl(src: MetricsSource) -> str:
    rows = src.query(
        "SELECT ts, realized, unrealized_mid, unrealized_consensus, "
        "settlement_pnl, bankroll_utilization "
        "FROM pnl WHERE market_id IS NULL ORDER BY id DESC LIMIT 1")
    if rows is None:
        return "no state db found"
    if not rows:
        return "no pnl snapshots yet"
    ts, realized, u_mid, u_cons, settle, util = rows[0]
    lines = [
        "pnl (TOTAL row, as of %s)" % _short_ts(ts),
        "realized: %s (of which settlement: %s)" % (_smoney(realized), _smoney(settle)),
        "unrealized mid: %s  consensus: %s" % (_smoney(u_mid), _smoney(u_cons)),
    ]
    try:
        lines.append("at-risk utilization: %.1f%%" % (float(util) * 100.0))
    except (TypeError, ValueError):
        pass
    srow = src.query(
        "SELECT COUNT(*), COALESCE(SUM(pnl_realized), 0.0) FROM settlements")
    if srow:
        n_settle, settle_total = srow[0]
        lines.append("settlements: %d markets, pnl %s" % (int(n_settle), _smoney(settle_total)))
    return "\n".join(lines)


def cmd_fills(src: MetricsSource) -> str:
    by_liq = src.query(
        "SELECT liquidity, COUNT(*), COALESCE(SUM(size), 0.0) "
        "FROM fills GROUP BY liquidity")
    if by_liq is None:
        return "no state db found"
    counts = {liq: (int(n), float(sz)) for liq, n, sz in by_liq}
    total = sum(n for n, _ in counts.values())
    lines = ["fills: %d total" % total]
    for liq in ("MAKER", "TAKER", "SETTLEMENT"):
        if liq in counts:
            n, sz = counts[liq]
            lines.append("  %s: %d (%.1f shares)" % (liq.lower(), n, sz))
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    day = src.query(
        "SELECT COUNT(*) FROM fills WHERE ts >= ? AND liquidity != 'SETTLEMENT'",
        (cutoff,))
    if day:
        lines.append("last 24h (ex-settlement): %d" % int(day[0][0]))
    last = src.query(
        "SELECT f.ts, f.side, f.price, f.size, f.liquidity, m.expiry_key, m.strike "
        "FROM fills f LEFT JOIN markets m ON m.market_id = f.market_id "
        "ORDER BY f.id DESC LIMIT 1")
    if last:
        ts, side, price, size, liq, ek, strike = last[0]
        where = ("%s %s" % (ek, ("%g" % strike) if strike is not None else "?")
                 if ek else "unknown market")
        lines.append("last: %s %s %.1f @ %.3f (%s, %s)" % (
            _short_ts(ts), side, float(size), float(price), liq.lower(), where))
    return "\n".join(lines)


def cmd_inventory(src: MetricsSource) -> str:
    rows = src.query(
        "SELECT i.market_id, i.q, i.avg_cost, m.expiry_key, m.strike "
        "FROM inventory i LEFT JOIN markets m ON m.market_id = i.market_id "
        "WHERE ABS(i.q) > 1e-9 "
        "ORDER BY m.expiry_key, m.strike")
    if rows is None:
        return "no state db found"
    if not rows:
        return "inventory: flat (no open positions)"
    lines = ["inventory: %d open position(s)" % len(rows)]
    for market_id, q, avg_cost, ek, strike in rows:
        where = ("%s %s" % (ek, "%g" % strike) if ek is not None
                 else market_id[:16])
        lines.append("  %s: q=%+.2f @ %.3f" % (where, float(q), float(avg_cost)))
    return "\n".join(lines)


def cmd_quotes(src: MetricsSource) -> str:
    cutoff = (datetime.now(timezone.utc)
              - timedelta(seconds=QUOTE_FRESH_WINDOW_S)).isoformat()
    rows = src.query(
        "SELECT q.ts, q.bid_price, q.ask_price, q.bid_size, q.ask_size, "
        "       m.expiry_key, m.strike "
        "FROM quotes q "
        "JOIN (SELECT market_id, MAX(id) AS max_id FROM quotes "
        "      WHERE ts >= ? GROUP BY market_id) latest "
        "  ON q.id = latest.max_id "
        "LEFT JOIN markets m ON m.market_id = q.market_id "
        "ORDER BY m.expiry_key, m.strike", (cutoff,))
    if rows is None:
        return "no state db found"
    if not rows:
        return "no quotes in the last %.0f min" % (QUOTE_FRESH_WINDOW_S / 60.0)
    now = datetime.now(timezone.utc)
    lines = ["quotes (latest per market, last %.0f min):" % (QUOTE_FRESH_WINDOW_S / 60.0)]
    for ts, bid, ask, bid_sz, ask_sz, ek, strike in rows:
        try:
            age_s = (now - datetime.fromisoformat(str(ts))).total_seconds()
            age = "%.0fs" % age_s
        except (TypeError, ValueError):
            age = "?"
        where = ("%s %s" % (ek, "%g" % strike) if ek is not None else "?")
        spread_c = (float(ask) - float(bid)) * 100.0
        lines.append("  %s: %.3f/%.3f (%.1fc) sz %.0fx%.0f age %s" % (
            where, float(bid), float(ask), spread_c,
            float(bid_sz), float(ask_sz), age))
    return "\n".join(lines)


def cmd_markout(src: MetricsSource) -> str:
    report = src.markout_json()
    if not report:
        return "no markout report found (needs a running/recent run with fills)"
    by_region = report.get("by_region")
    if not isinstance(by_region, dict) or not by_region:
        return "markout report present but empty (no eligible fills yet)"
    lines = ["markout by region (avg, YES-scale; + = fills aged well):"]
    for region in sorted(by_region):
        horizons = by_region[region]
        if not isinstance(horizons, dict):
            continue
        parts = []
        for h in sorted(horizons, key=lambda s: float(s)):
            cell = horizons[h] or {}
            parts.append("%ss: %+.4f (n=%s/%s)" % (
                ("%g" % float(h)), float(cell.get("mk_avg", 0.0)),
                cell.get("n"), cell.get("n_attempted")))
        lines.append("  %s: %s" % (region, "  ".join(parts)))
    if report.get("generated_ts"):
        lines.append("generated %s" % _short_ts(report["generated_ts"]))
    return "\n".join(lines)


def cmd_help(_src: MetricsSource) -> str:
    return "\n".join([
        "mm-paper metrics bot -- commands:",
        "/status    engine state, tick, feed, per-expiry",
        "/bankroll  initial bankroll + current equity + rebates (est)",
        "/pnl       realized/unrealized/settlement breakdown",
        "/fills     fill counts + last fill",
        "/inventory open positions",
        "/quotes    latest quote per market",
        "/markout   fill markout by region",
    ])


COMMANDS: Dict[str, Callable[[MetricsSource], str]] = {
    "/status": cmd_status,
    "/bankroll": cmd_bankroll,
    "/pnl": cmd_pnl,
    "/fills": cmd_fills,
    "/inventory": cmd_inventory,
    "/quotes": cmd_quotes,
    "/markout": cmd_markout,
    "/help": cmd_help,
    "/start": cmd_help,
}


def handle_command(text: Optional[str], src: MetricsSource) -> Optional[str]:
    """Dispatch one message text. None = not a command (ignore silently);
    a str is always safe to send (handler errors are caught and reported)."""
    if not isinstance(text, str) or not text.strip().startswith("/"):
        return None
    word = text.strip().split()[0].lower()
    if "@" in word:  # /status@MyBot form in group chats
        word = word.split("@", 1)[0]
    handler = COMMANDS.get(word)
    if handler is None:
        return "unknown command %s -- try /help" % word
    try:
        reply = handler(src)
    except Exception as exc:  # noqa: BLE001 - a handler bug must not kill the bot
        reply = "error handling %s: %s" % (word, exc)
    if len(reply) > REPLY_MAX_CHARS:
        reply = reply[:REPLY_MAX_CHARS] + "\n[truncated]"
    return "[mm-paper] " + reply


# ---------------------------------------------------------------------------
# telegram transport
# ---------------------------------------------------------------------------


def _api_call(token: str, method: str, params: Dict[str, Any],
              timeout_s: float) -> Dict[str, Any]:
    url = "https://api.telegram.org/bot%s/%s" % (token, method)
    payload = json.dumps(params).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_offset(control_dir: Path) -> int:
    try:
        with open(control_dir / BOT_STATE_FILENAME, "r", encoding="ascii") as f:
            data = json.load(f)
        return int(data.get("offset", 0))
    except (OSError, ValueError, TypeError):
        return 0


def _save_offset(control_dir: Path, offset: int) -> None:
    path = control_dir / BOT_STATE_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps({"offset": offset}), encoding="ascii")
        os.replace(str(tmp), str(path))
    except OSError as exc:
        print("mm_telegram_bot: could not persist offset: %s" % exc)


def poll_once(token: str, allowed_chat_id: str, src: MetricsSource,
              offset: int, poll_timeout_s: int) -> int:
    """One getUpdates pass. Returns the new offset. Raises on transport
    errors (caller backs off and retries)."""
    resp = _api_call(
        token, "getUpdates",
        {"offset": offset + 1, "timeout": poll_timeout_s,
         "allowed_updates": ["message"]},
        timeout_s=poll_timeout_s + 15,
    )
    for update in resp.get("result", []):
        offset = max(offset, int(update.get("update_id", offset)))
        msg = update.get("message")
        if not isinstance(msg, dict):
            continue
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if chat_id != str(allowed_chat_id):
            continue  # allowlist: never answer unknown chats
        reply = handle_command(msg.get("text"), src)
        if reply is None:
            continue
        try:
            _api_call(token, "sendMessage",
                      {"chat_id": chat_id, "text": reply}, timeout_s=15)
        except Exception as exc:  # noqa: BLE001
            print("mm_telegram_bot: sendMessage failed: %s" % exc)
    return offset


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    from market_maker import run_control

    ap = argparse.ArgumentParser(
        description="Telegram metrics bot for the Stage-B paper runner (read-only)")
    ap.add_argument("--control-dir", default=str(run_control.CONTROL_DIR),
                    help="matches paper_runner --control-dir")
    ap.add_argument("--state-db", default="",
                    help="override state-db path (default: discovered via run_meta.json)")
    ap.add_argument("--poll-timeout-s", type=int, default=POLL_TIMEOUT_S_DEFAULT,
                    help="Telegram getUpdates long-poll timeout")
    ap.add_argument("--once", action="store_true",
                    help="single getUpdates pass then exit (testing)")
    args = ap.parse_args(argv)

    token, chat_id = resolve_creds()
    if not token or not chat_id:
        print("mm_telegram_bot: no credentials -- set MM_TELEGRAM_TOKEN + "
              "MM_TELEGRAM_CHAT_ID, or MM_ALERT_WEBHOOK with a Telegram "
              "sendMessage URL (token and chat_id are parsed from it)")
        return 2

    control_dir = Path(args.control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    src = MetricsSource(
        control_dir,
        state_db_override=Path(args.state_db) if args.state_db else None,
    )
    offset = _load_offset(control_dir)
    print("mm_telegram_bot: polling as chat %s (control-dir %s)" % (chat_id, control_dir))

    while True:
        try:
            new_offset = poll_once(token, chat_id, src, offset, args.poll_timeout_s)
            if new_offset != offset:
                offset = new_offset
                _save_offset(control_dir, offset)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:  # noqa: BLE001 - daemon must never die on transport errors
            print("mm_telegram_bot: poll error: %s" % exc)
            time.sleep(ERROR_BACKOFF_S)
        if args.once:
            return 0


if __name__ == "__main__":
    sys.exit(main())
