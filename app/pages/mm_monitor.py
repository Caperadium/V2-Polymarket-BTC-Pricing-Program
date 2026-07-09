"""
app/pages/mm_monitor.py

Read-only Streamlit monitoring page for the Stage-B paper-trading engine
(market_maker/paper_runner.py), plus START/STOP/FORCE KILL controls wired
to market_maker/run_control.py. This page never instantiates MMStateStore
and never writes into a run's output directory -- it only reads control
files, run_meta.json/heartbeat.json, the run's CSVs, and paper_state.db
(opened read-only).

VPS runbook (short)
--------------------
- The launcher (market_maker/run_control.start_engine) spawns a detached
  subprocess (posix: start_new_session=True). It survives a Streamlit
  restart; a restarted dashboard reattaches via the PID file.
- Run Streamlit under systemd bound to 127.0.0.1 and access it via an SSH
  tunnel ONLY. This page has process-control buttons (START/STOP/FORCE
  KILL) -- never expose it publicly without authentication in front of it.
- Use EITHER systemd (e.g. an `mm-paper.service` with `Restart=on-failure`)
  OR this page's buttons to control the runner -- not both. A systemd
  `Restart=` policy will fight a graceful STOP requested from here.
- Cron: refresh BTC data periodically so the vol gate and settlement stay
  accurate on long/indefinite runs, e.g.:
      */20 * * * * cd <repo> && python core/data/data_fetcher.py
  Without this, the runner's own periodic re-read (--btc-refresh-s) just
  re-reads a stale file and expiries can land UNSETTLEABLE.

Data access rules (see market_maker/run_control.py and market_maker/
state_store.py docstrings for the full control-file / schema contracts):
- NEVER instantiate MMStateStore here (its __init__ writes a WAL pragma +
  the schema). Read paper_state.db via a short-lived read-only sqlite3
  connection (`mode=ro` URI + `PRAGMA query_only=ON`), close it promptly.
- CSVs (quotes.csv/fills.csv/ticks.csv) go through an @st.cache_data loader
  keyed on (path, mtime) so a growing file is re-read only when it changes.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Bootstrap: allow `market_maker`/`core` imports when Streamlit runs this
# page directly (polymarket_console.py pattern).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from market_maker import run_control  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_RUN_DIR = PROJECT_ROOT / "temp" / "paper_run"
BTC_INTRADAY_PATH = PROJECT_ROOT / "DATA" / "btc_intraday_1m.csv"

DEFAULT_TICK_S = 15.0
BTC_STALE_WARN_S = 2 * 3600.0
AUTO_REFRESH_INTERVALS = (5, 10, 30, 60)

PNL_TOTAL_SQL = "SELECT * FROM pnl WHERE market_id IS NULL ORDER BY id ASC"
INVENTORY_SQL = "SELECT * FROM inventory"
RISK_LATEST_SQL = """
SELECT r.* FROM risk_journal r
JOIN (SELECT market_id, MAX(id) AS mid FROM risk_journal GROUP BY market_id) x
ON r.id = x.mid
"""
LIQUIDITY_LATEST_SQL = """
SELECT l.* FROM liquidity_windows l
JOIN (SELECT market_id, MAX(id) AS mid FROM liquidity_windows GROUP BY market_id) x
ON l.id = x.mid
"""
SETTLEMENTS_SQL = "SELECT * FROM settlements"

st.set_page_config(page_title="MM Monitor", layout="wide")


# ---------------------------------------------------------------------------
# Small pure / data-access helpers (top-level and testable without a running
# Streamlit runtime).
# ---------------------------------------------------------------------------


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _file_age_s(path: Path) -> Optional[float]:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _num_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip()
    except Exception:
        return None
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def tail_file_bytes(path: Path, n: int = 16384) -> str:
    """Return the last `n` bytes of `path`, decoded leniently. Empty string
    if the file is missing or unreadable."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > n:
                f.seek(size - n)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    """Cached CSV load keyed on (path, mtime) so a growing file is only
    re-read when it actually changes. `on_bad_lines='skip'` tolerates a
    mid-line flush race with the runner's own writer (W6)."""
    try:
        return pd.read_csv(path_str, on_bad_lines="skip")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def load_csv(path: Path) -> pd.DataFrame:
    """mtime-bust wrapper around load_csv_cached; empty DataFrame if the
    file does not exist yet (fresh/STARTING run)."""
    try:
        m = path.stat().st_mtime
    except OSError:
        return pd.DataFrame()
    return load_csv_cached(str(path), m)


def resolve_state_db(out_dir: Optional[Path], run_meta: Optional[dict]) -> Optional[Path]:
    """Effective state-db path for a run. A run launched with --state-db
    (e.g. the VPS config's persistent market_maker/mm_paper_state.db) does
    NOT have its db under out_dir, so prefer, in order: the resolved
    `state_db` the runner records in run_meta (newer runs), the `state_db`
    in the run's config dict, a `--state-db` in the recorded argv (older
    run_meta without the top-level key), then the per-run default
    out_dir/paper_state.db. Relative paths are anchored at PROJECT_ROOT
    (the runner's working directory). Note a shared persistent db shows
    CURRENT inventory/PnL state even when viewing a historical run."""
    candidates: List[Any] = []
    if isinstance(run_meta, dict):
        candidates.append(run_meta.get("state_db"))
        cfg = run_meta.get("config")
        if isinstance(cfg, dict):
            candidates.append(cfg.get("state_db"))
        argv = run_meta.get("argv")
        if isinstance(argv, list) and "--state-db" in argv:
            i = argv.index("--state-db")
            if i + 1 < len(argv):
                candidates.append(argv[i + 1])
    for c in candidates:
        if isinstance(c, str) and c:
            p = Path(c)
            return p if p.is_absolute() else (PROJECT_ROOT / p)
    return (out_dir / "paper_state.db") if out_dir else None


def load_db_table(
    db_path: Optional[Path], sql: str, params: Tuple[Any, ...] = ()
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Read-only query against paper_state.db. Returns (df, None) on
    success, or (None, message) when there is nothing to show yet, or the
    DB can't be opened read-only (e.g. a leftover -wal from an unclean
    shutdown -- the reader can't checkpoint, W1)."""
    if db_path is None:
        return None, "no run selected yet"
    if not db_path.exists():
        return None, "paper_state.db not created yet"
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=2.0)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=2000")
        df = pd.read_sql_query(sql, conn, params=params)
        return df, None
    except sqlite3.OperationalError:
        return None, (
            "run DB not readable (unclean WAL or locked); "
            "open after the runner exits cleanly"
        )
    except sqlite3.DatabaseError:
        return None, "run DB not readable (corrupt or mid-write)"
    finally:
        if conn is not None:
            conn.close()


def get_live_out_dir(control_dir: Path) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Resolve the out_dir for the 'Live / current' selection straight from
    current_run.json. During STARTING the minimal current_run.json written
    first thing in run() has no out_dir key yet -- returns (None, run_info)
    in that case; callers must render placeholders, not crash (R4)."""
    run_info = _load_json(control_dir / run_control.RUN_JSON)
    if not run_info:
        return None, None
    out_dir_raw = run_info.get("out_dir")
    return (Path(out_dir_raw) if out_dir_raw else None), run_info


def list_historical_runs(base_dir: Path) -> List[Dict[str, Any]]:
    """Sorted (newest first) list of temp/paper_run/<ts>/ dirs, excluding
    the 'control' dir. Each entry carries its run_meta.json (or None)."""
    if not base_dir.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for p in sorted(base_dir.iterdir(), reverse=True):
        if not p.is_dir() or p.name == "control":
            continue
        run_meta = _load_json(p / "run_meta.json")
        if run_meta:
            label = "%s  %s  %s" % (
                p.name, run_meta.get("event_slug", "?"), run_meta.get("expiry_key", "?"),
            )
        else:
            label = "%s  (no run_meta.json)" % p.name
        entries.append({"dir": p, "label": label, "run_meta": run_meta})
    return entries


def build_positions_table(
    inv_df: Optional[pd.DataFrame], quotes_df: Optional[pd.DataFrame], expiry_key: Optional[str]
) -> pd.DataFrame:
    """Join `inventory` rows with the latest quotes.csv row per market to
    recover strike + a mark price (mid of mkt_bid/mkt_ask). run_meta.json
    does not carry a market-slug -> strike map (only the strike ladder), so
    strike is sourced from quotes.csv instead, which already carries it
    per row -- see report for this deviation from the plan sketch."""
    if inv_df is None or inv_df.empty:
        return pd.DataFrame()

    latest_quotes: Dict[str, pd.Series] = {}
    if quotes_df is not None and not quotes_df.empty and "market" in quotes_df.columns:
        tail1 = quotes_df.groupby("market", sort=False).tail(1)
        latest_quotes = {row["market"]: row for _, row in tail1.iterrows()}

    rows = []
    for _, r in inv_df.iterrows():
        market = r["market_id"]
        q = float(r["q"])
        avg_cost = float(r["avg_cost"])
        q_max = float(r["q_max"])
        strike = None
        mark = None
        qr = latest_quotes.get(market)
        if qr is not None:
            strike = _num_or_none(qr.get("strike"))
            bid = _num_or_none(qr.get("mkt_bid"))
            ask = _num_or_none(qr.get("mkt_ask"))
            if bid is not None and ask is not None:
                mark = 0.5 * (bid + ask)
            elif bid is not None:
                mark = bid
            elif ask is not None:
                mark = ask
        util = (abs(q) / q_max) if q_max else 0.0
        unrealized = (q * (mark - avg_cost)) if mark is not None else None
        rows.append({
            "market": market, "strike": strike, "expiry": expiry_key,
            "q": q, "avg_cost": avg_cost, "q_max": q_max, "util": util,
            "mark": mark, "unrealized": unrealized,
        })
    return pd.DataFrame(rows)


def _run_tick_s(run_meta: Optional[Dict[str, Any]]) -> float:
    if run_meta and run_meta.get("tick_s") is not None:
        try:
            return float(run_meta["tick_s"])
        except (TypeError, ValueError):
            pass
    return DEFAULT_TICK_S


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_status_row(status: "run_control.EngineStatus") -> None:
    cols = st.columns(6)

    text = "%s -- %s" % (status.state, status.detail)
    if status.state == "RUNNING":
        cols[0].success(text)
    elif status.state == "STARTING":
        cols[0].info(text)
    elif status.state in ("STALLED", "CRASHED"):
        cols[0].error(text)
    else:  # STOPPED
        cols[0].write(text)

    hb = status.heartbeat or {}
    cols[1].metric("Heartbeat age (s)", "%.1f" % status.heartbeat_age_s if status.heartbeat_age_s is not None else "n/a")

    started = (status.run_info or {}).get("started_utc")
    uptime_txt = "n/a"
    if started:
        try:
            started_dt = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
            if started_dt.tzinfo is None:
                started_dt = started_dt.replace(tzinfo=timezone.utc)
            uptime_s = (datetime.now(timezone.utc) - started_dt).total_seconds()
            uptime_txt = "%.0f min" % (uptime_s / 60.0)
        except ValueError:
            uptime_txt = "n/a"
    tick_txt = str(hb.get("tick", "n/a"))
    cols[2].metric("Tick / uptime", "%s / %s" % (tick_txt, uptime_txt))

    feed_healthy = hb.get("feed_healthy")
    cols[3].metric("Feed healthy", "yes" if feed_healthy else ("no" if feed_healthy is not None else "n/a"))
    cols[4].metric("Fills total", str(hb.get("fills_total", "n/a")))
    cols[5].metric("No-arb violations", str(hb.get("noarb_violations", "n/a")))


def render_controls(status: "run_control.EngineStatus", control_dir: Path) -> None:
    col_start, col_stop, col_kill = st.columns(3)

    start_disabled = status.state not in ("STOPPED", "CRASHED")
    if col_start.button("START", disabled=start_disabled, key="mm_monitor_start"):
        ok, msg = run_control.start_engine(control_dir=control_dir)
        (st.success if ok else st.error)(msg)
        st.rerun()

    btc_age = _file_age_s(BTC_INTRADAY_PATH)
    if btc_age is None:
        col_start.caption("DATA/btc_intraday_1m.csv: not found")
    else:
        col_start.caption("BTC data age: %.0f min" % (btc_age / 60.0))
        if btc_age > BTC_STALE_WARN_S:
            col_start.warning("BTC data stale (> 2h) -- run core/data/data_fetcher.py / check cron")

    stop_path = control_dir / run_control.STOP_FILE
    stop_exists = stop_path.exists()
    stop_disabled = status.state in ("STOPPED", "CRASHED")
    if col_stop.button("STOP", disabled=stop_disabled, key="mm_monitor_stop"):
        ok, msg = run_control.request_stop(control_dir=control_dir)
        (st.info if ok else st.error)(msg)
        st.rerun()
    if stop_exists:
        col_stop.caption("stop requested - takes up to one reprice cycle")

    show_kill = False
    if stop_exists and status.pid is not None and run_control.pid_alive(status.pid):
        live_run_meta = _load_json(status.out_dir / "run_meta.json") if status.out_dir else None
        tick_s = _run_tick_s(live_run_meta)
        try:
            stop_age = time.time() - stop_path.stat().st_mtime
        except OSError:
            stop_age = None
        if stop_age is not None and stop_age > 3.0 * tick_s:
            show_kill = True
    if show_kill:
        if col_kill.button("FORCE KILL", key="mm_monitor_kill"):
            ok, msg = run_control.kill_engine(control_dir=control_dir)
            (st.warning if ok else st.error)(msg)
            st.rerun()

    with st.expander("paper_run_config.json"):
        cfg = _load_json(run_control.DEFAULT_CONFIG)
        if cfg is None:
            st.info("config file not found: %s" % run_control.DEFAULT_CONFIG)
        else:
            st.json(cfg)

    with st.expander("runner.log (tail)"):
        if status.out_dir is None:
            st.info("no run directory yet")
        else:
            tail = tail_file_bytes(status.out_dir / "runner.log")
            if not tail:
                st.info("runner.log not found or empty")
            else:
                st.text(tail)


def render_pnl(db_path: Optional[Path], run_meta: Optional[Dict[str, Any]]) -> None:
    df, err = load_db_table(db_path, PNL_TOTAL_SQL)
    if err:
        st.info(err)
        return
    if df is None or df.empty:
        st.info("no PnL snapshots yet")
        return

    bankroll = float(run_meta.get("bankroll", 0.0)) if run_meta else 0.0
    df["equity"] = bankroll + df["realized"] + df["unrealized_mid"]
    latest = df.iloc[-1]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity", "%.2f" % latest["equity"])
    c2.metric("Realized", "%.2f" % latest["realized"])
    c3.metric("Unrealized (mid)", "%.2f" % latest["unrealized_mid"])
    c4.metric("Settlement PnL (info)", "%.2f" % latest["settlement_pnl"])
    c5.metric("Bankroll utilization", "%.1f%%" % (100.0 * latest["bankroll_utilization"]))
    st.caption(
        "Equity = bankroll + realized + unrealized_mid. Settlement PnL is a "
        "report-only breakdown of realized -- it is already inside realized "
        "and is NOT added to equity."
    )

    df["ts_dt"] = pd.to_datetime(df["ts"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts_dt"], y=df["equity"], mode="lines", name="equity"))
    fig.update_layout(title="Equity curve", height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_positions(db_path: Optional[Path], out_dir: Optional[Path], run_meta: Optional[Dict[str, Any]]) -> None:
    if run_meta:
        strikes = run_meta.get("strikes") or []
        st.caption(
            "%s | expiry %s | strikes: %s"
            % (run_meta.get("event_slug", "?"), run_meta.get("expiry_key", "?"), strikes)
        )

    inv_df, err = load_db_table(db_path, INVENTORY_SQL)
    if err:
        st.info(err)
        return
    if inv_df is None or inv_df.empty:
        st.info("no inventory rows yet")
        return

    quotes_df = load_csv(out_dir / "quotes.csv") if out_dir else pd.DataFrame()
    expiry_key = run_meta.get("expiry_key") if run_meta else None
    table = build_positions_table(inv_df, quotes_df, expiry_key)
    if table.empty:
        st.info("no positions to show")
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_fills(out_dir: Optional[Path]) -> None:
    if out_dir is None:
        st.info("no run directory yet")
        return
    fills_df = load_csv(out_dir / "fills.csv")
    if fills_df.empty:
        st.info("no fills yet")
        return
    st.dataframe(fills_df.tail(50).iloc[::-1], use_container_width=True, hide_index=True)


def render_risk_panel(db_path: Optional[Path]) -> None:
    risk_df, risk_err = load_db_table(db_path, RISK_LATEST_SQL)
    if risk_err:
        st.info(risk_err)
    elif risk_df is None or risk_df.empty:
        st.info("no risk_journal rows yet")
    else:
        risk_df = risk_df.copy()
        risk_df["triggers"] = risk_df["triggers"].apply(
            lambda s: ", ".join(json.loads(s)) if isinstance(s, str) else s
        )
        risk_df["cancel_all"] = risk_df["cancel_all"].astype(bool)
        st.dataframe(
            risk_df[["market_id", "mode", "triggers", "latched_until", "cancel_all"]],
            use_container_width=True, hide_index=True,
        )

    liq_df, liq_err = load_db_table(db_path, LIQUIDITY_LATEST_SQL)
    if liq_err:
        st.info(liq_err)
    elif liq_df is None or liq_df.empty:
        st.info("no liquidity_windows rows yet")
    else:
        st.dataframe(
            liq_df[["market_id", "regime", "kyle_lambda"]],
            use_container_width=True, hide_index=True,
        )

    settle_df, settle_err = load_db_table(db_path, SETTLEMENTS_SQL)
    if settle_err:
        st.info(settle_err)
    elif settle_df is not None and not settle_df.empty:
        st.markdown("**Settlements**")
        st.dataframe(settle_df, use_container_width=True, hide_index=True)


def render_markout(out_dir: Optional[Path]) -> None:
    """Read-only render of <out_dir>/markout_report.json (mm_suitability_
    alignment_plan.md Change C5) -- per-region/tte-bucket/horizon markout
    cells, written by paper_runner.py every PER_MARKET_SNAPSHOT_EVERY_N_TICKS
    ticks. No new controls; a plain table like the other panels on this page.
    """
    if out_dir is None:
        st.info("no run directory yet")
        return
    report = _load_json(out_dir / "markout_report.json")
    if not report:
        st.info("no markout_report.json yet")
        return
    cells = report.get("cells") or []
    if not cells:
        st.info("markout report has no cells yet")
        return
    cells_df = pd.DataFrame(cells)
    # F2: explicit computed coverage column (n / n_attempted) -- a cell can
    # have n_attempted > 0 with n == 0 (every lookup for it missed) so this is
    # distinct from just showing the raw n/n_attempted columns.
    if "n_attempted" in cells_df.columns:
        n_attempted = pd.to_numeric(cells_df["n_attempted"], errors="coerce")
        n = pd.to_numeric(cells_df["n"], errors="coerce")
        cells_df["coverage"] = (n / n_attempted).where(n_attempted > 0, 0.0)
    st.dataframe(cells_df, use_container_width=True, hide_index=True)
    st.caption(
        "mk_h = sign*(mid_h - fill price), sign=+1 BUY_YES / -1 BUY_NO (never "
        "complemented -- stored fill price is already YES-scale for both "
        "sides). coverage = n / n_attempted (share of eligible fills whose "
        "horizon lookup found a mid). Generated %s." % report.get("generated_ts", "?")
    )


def render_quotes_latency(out_dir: Optional[Path]) -> None:
    if out_dir is None:
        st.info("no run directory yet")
        return

    quotes_df = load_csv(out_dir / "quotes.csv")
    if not quotes_df.empty and "market" in quotes_df.columns:
        markets = sorted(quotes_df["market"].dropna().unique().tolist())
        sel = st.selectbox("Market", markets, key="mm_monitor_quote_market")
        mdf = quotes_df[quotes_df["market"] == sel].copy()
        mdf["ts_dt"] = pd.to_datetime(mdf["ts"], errors="coerce")
        mdf["our_spread"] = pd.to_numeric(mdf["our_spread"], errors="coerce")
        mdf["mkt_spread"] = pd.to_numeric(mdf["mkt_spread"], errors="coerce")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mdf["ts_dt"], y=mdf["our_spread"], mode="lines", name="our_spread"))
        fig.add_trace(go.Scatter(x=mdf["ts_dt"], y=mdf["mkt_spread"], mode="lines", name="mkt_spread"))
        fig.update_layout(title="Spread: %s" % sel, height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("no quotes.csv data yet")

    ticks_df = load_csv(out_dir / "ticks.csv")
    if not ticks_df.empty and "wall_s" in ticks_df.columns:
        tdf = ticks_df.copy()
        tdf["ts_dt"] = pd.to_datetime(tdf["ts"], errors="coerce")
        tdf["wall_s"] = pd.to_numeric(tdf["wall_s"], errors="coerce")
        tdf["feed_healthy"] = pd.to_numeric(tdf["feed_healthy"], errors="coerce")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tdf["ts_dt"], y=tdf["wall_s"], mode="lines", name="wall_s"))
        unhealthy = tdf[tdf["feed_healthy"] == 0]
        if not unhealthy.empty:
            fig2.add_trace(go.Scatter(
                x=unhealthy["ts_dt"], y=unhealthy["wall_s"], mode="markers",
                marker=dict(color="red", size=8), name="feed unhealthy",
            ))
        fig2.update_layout(title="Tick wall latency (s)", height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("no ticks.csv data yet")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("MM Monitor")

    control_dir = run_control.CONTROL_DIR
    status = run_control.engine_status(control_dir=control_dir)

    st.header("Status")
    render_status_row(status)

    st.header("Controls")
    render_controls(status, control_dir)

    # -- Sidebar: run selector + auto-refresh --
    st.sidebar.header("Run")
    historical = list_historical_runs(PAPER_RUN_DIR)
    options = ["Live / current"] + [h["label"] for h in historical]
    choice = st.sidebar.selectbox("View run", options, key="mm_monitor_run_choice")

    if choice == "Live / current":
        out_dir, _run_info = get_live_out_dir(control_dir)
        run_meta = _load_json(out_dir / "run_meta.json") if out_dir else None
    else:
        entry = next((h for h in historical if h["label"] == choice), None)
        out_dir = entry["dir"] if entry else None
        run_meta = entry["run_meta"] if entry else None

    db_path = resolve_state_db(out_dir, run_meta)

    if out_dir is None:
        st.info("no out_dir yet for this run (engine likely still STARTING)")

    st.header("PnL")
    render_pnl(db_path, run_meta)

    st.header("Positions")
    render_positions(db_path, out_dir, run_meta)

    st.header("Fills (latest 50)")
    render_fills(out_dir)

    st.header("Risk")
    render_risk_panel(db_path)

    st.header("Quotes / Latency")
    render_quotes_latency(out_dir)

    st.header("Markout")
    render_markout(out_dir)

    st.sidebar.divider()
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True, key="mm_monitor_auto_refresh")
    interval = st.sidebar.selectbox(
        "Refresh interval (s)", AUTO_REFRESH_INTERVALS, index=1, key="mm_monitor_refresh_interval",
    )

    if auto_refresh:
        # Streamlit reruns synchronously; any button click that lands while
        # this sleep is in flight is queued and only processed once this
        # rerun completes. Keep the interval modest (single-operator tool).
        time.sleep(interval)
        st.rerun()


main()
