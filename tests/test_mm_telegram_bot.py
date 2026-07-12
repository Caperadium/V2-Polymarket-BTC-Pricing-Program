"""Tests for scripts/mm_telegram_bot.py -- command handlers, dispatch, and
credential parsing. All handler tests run against a real MMStateStore db in a
tmp dir (the same schema the runner writes) with a stubbed EngineStatus, so no
network/Telegram access is ever needed.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# scripts/ is not a package (no __init__.py) -- load the module by path
# (mirrors tests/test_mm_alert_check.py).
_SPEC = importlib.util.spec_from_file_location(
    "mm_telegram_bot", PROJECT_ROOT / "scripts" / "mm_telegram_bot.py"
)
bot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bot)  # type: ignore[union-attr]

from market_maker.contracts import ContractInv, Fill, LiquiditySource, Side  # noqa: E402
from market_maker.run_control import EngineStatus  # noqa: E402
from market_maker.state_store import MMStateStore, PnlSnapshot  # noqa: E402


NOW = datetime.now(timezone.utc)


def _status(heartbeat=None, run_info=None, state="RUNNING") -> EngineStatus:
    return EngineStatus(
        state=state, pid=123, run_info=run_info, heartbeat=heartbeat,
        heartbeat_age_s=5.0, out_dir=None, detail="test",
    )


class StubSource(bot.MetricsSource):
    """MetricsSource with a canned EngineStatus (no control-dir reads)."""

    def __init__(self, control_dir, status, **kwargs):
        super().__init__(control_dir, **kwargs)
        self._status = status

    def status(self):
        return self._status


@pytest.fixture()
def env(tmp_path):
    """(source, store) against a populated tmp state db + run_meta.json."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    db_path = tmp_path / "state.db"
    store = MMStateStore(str(db_path))

    import json
    (out_dir / "run_meta.json").write_text(
        json.dumps({"bankroll": 1000.0, "state_db": str(db_path)}),
        encoding="ascii")

    status = _status(
        heartbeat={"tick": 42, "feed_healthy": True, "fills_total": 3,
                   "bankroll_frozen": False, "n_expiries_active": 1,
                   "btc_data_age_s": 120.0, "pulled_ticks": 0,
                   "noarb_violations": 0, "feed_restarts": 0,
                   "expiries": {"2026-07-13": {
                       "event_slug": "s", "state": "ACTIVE",
                       "feed_healthy": True, "feed_restarts": 0,
                       "bankroll_frozen": False, "fills": 3,
                       "mode_counts": {}}}},
        run_info={"out_dir": str(out_dir)},
    )
    src = StubSource(tmp_path / "control", status)
    yield src, store
    store.close()


def _fill(market_id="m1", side=Side.BUY_YES, price=0.45, size=3.0,
          liquidity=LiquiditySource.MAKER, ts=None) -> Fill:
    ts = ts or NOW
    return Fill(ts=ts, market_id=market_id, order_id="o1", side=side,
                price=price, size=size, liquidity=liquidity, venue_ts=ts)


# ---------------------------------------------------------------------------
# credential parsing
# ---------------------------------------------------------------------------


def test_parse_webhook_creds_telegram_url():
    url = "https://api.telegram.org/bot123456:AAGabc_DEF-ghi/sendMessage?chat_id=6855476087"
    token, chat_id = bot.parse_webhook_creds(url)
    assert token == "123456:AAGabc_DEF-ghi"
    assert chat_id == "6855476087"


def test_parse_webhook_creds_negative_chat_id():
    url = "https://api.telegram.org/bot1:AAA/sendMessage?chat_id=-100123"
    assert bot.parse_webhook_creds(url) == ("1:AAA", "-100123")


def test_parse_webhook_creds_non_telegram():
    assert bot.parse_webhook_creds("https://hooks.slack.com/services/X/Y/Z") == (None, None)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_handle_command_ignores_non_commands(env):
    src, _ = env
    assert bot.handle_command("hello", src) is None
    assert bot.handle_command(None, src) is None
    assert bot.handle_command("", src) is None


def test_handle_command_unknown(env):
    src, _ = env
    reply = bot.handle_command("/bogus", src)
    assert "unknown command" in reply and "/help" in reply


def test_handle_command_strips_bot_suffix(env):
    src, _ = env
    reply = bot.handle_command("/help@SomeBot", src)
    assert "/bankroll" in reply


def test_handler_exception_is_reported_not_raised(env, monkeypatch):
    src, _ = env
    monkeypatch.setitem(bot.COMMANDS, "/status", lambda s: 1 / 0)
    reply = bot.handle_command("/status", src)
    assert reply.startswith("[mm-paper] error handling /status")


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


def test_cmd_status(env):
    src, _ = env
    reply = bot.cmd_status(src)
    assert "RUNNING" in reply
    assert "tick=42" in reply
    assert "2026-07-13" in reply and "fills=3" in reply


def test_cmd_bankroll_equity_from_total_row(env):
    src, store = env
    store.append_pnl_snapshot(PnlSnapshot(
        ts=NOW, market_id=None, expiry_key=None, realized=5.0,
        unrealized_consensus=1.0, unrealized_mid=3.0, settlement_pnl=0.0,
        bankroll_utilization=0.10))
    reply = bot.cmd_bankroll(src)
    assert "initial: 1000.00" in reply
    assert "equity: 1008.00" in reply
    assert "+5.00" in reply and "+3.00" in reply
    assert "10.0%" in reply
    assert "FROZEN" not in reply


def test_cmd_bankroll_no_snapshots(env):
    src, _ = env
    reply = bot.cmd_bankroll(src)
    assert "no pnl snapshots yet" in reply


def test_cmd_bankroll_frozen_warning(env):
    src, store = env
    src._status.heartbeat["bankroll_frozen"] = True
    reply = bot.cmd_bankroll(src)
    assert "FROZEN" in reply


def test_cmd_pnl_breakdown(env):
    src, store = env
    store.append_pnl_snapshot(PnlSnapshot(
        ts=NOW, market_id=None, expiry_key=None, realized=-2.5,
        unrealized_consensus=0.5, unrealized_mid=1.5, settlement_pnl=-1.0,
        bankroll_utilization=0.05))
    reply = bot.cmd_pnl(src)
    assert "-2.50" in reply and "-1.00" in reply
    assert "+1.50" in reply and "+0.50" in reply


def test_cmd_fills_counts_and_last(env):
    src, store = env
    store.upsert_market("m1", "2026-07-13", 118000.0)
    old = NOW - timedelta(hours=30)
    store.append_fill(_fill(ts=old))
    store.append_fill(_fill(side=Side.BUY_NO, price=0.60, size=2.0))
    store.append_fill(_fill(liquidity=LiquiditySource.SETTLEMENT, price=1.0))
    reply = bot.cmd_fills(src)
    assert "3 total" in reply
    assert "maker: 2" in reply
    assert "settlement: 1" in reply
    assert "last 24h (ex-settlement): 1" in reply
    assert "2026-07-13 118000" in reply  # last fill labeled via markets registry


def test_cmd_fills_empty_db(env):
    src, _ = env
    assert "0 total" in bot.cmd_fills(src)


def test_cmd_inventory(env):
    src, store = env
    store.upsert_market("m1", "2026-07-13", 118000.0)
    store.upsert_inventory("m1", ContractInv(
        q=3.0, avg_cost=0.45, q_max=10.0, age_weighted_holding=0.0))
    store.upsert_inventory("m2", ContractInv(  # flat -> excluded
        q=0.0, avg_cost=0.50, q_max=10.0, age_weighted_holding=0.0))
    reply = bot.cmd_inventory(src)
    assert "1 open position" in reply
    assert "2026-07-13 118000: q=+3.00 @ 0.450" in reply


def test_cmd_inventory_flat(env):
    src, _ = env
    assert "flat" in bot.cmd_inventory(src)


def test_cmd_quotes_latest_per_market(env):
    src, store = env
    store.upsert_market("m1", "2026-07-13", 118000.0)
    iso_now = NOW.isoformat()
    iso_older = (NOW - timedelta(seconds=60)).isoformat()
    with store._conn:
        for ts, bid, ask in ((iso_older, 0.40, 0.50), (iso_now, 0.43, 0.47)):
            store._conn.execute(
                "INSERT INTO quotes (ts, market_id, bid_price, ask_price, bid_size,"
                " ask_size, terms, risk_mode, noarb_checked, source_seq, r_x,"
                " delta_x, skew_x, sigma_b, params_id, x_bid, x_ask, p_bid_raw,"
                " p_ask_raw) VALUES (?, 'm1', ?, ?, 12, 12, '{}', 'NORMAL', 1, 0,"
                " 0, 0, 0, 0, 'p', 0, 0, 0, 0)",
                (ts, bid, ask))
    reply = bot.cmd_quotes(src)
    assert "0.430/0.470" in reply  # newest row wins
    assert "0.400" not in reply
    assert "4.0c" in reply


def test_cmd_quotes_stale_excluded(env):
    src, store = env
    stale = (NOW - timedelta(hours=2)).isoformat()
    with store._conn:
        store._conn.execute(
            "INSERT INTO quotes (ts, market_id, bid_price, ask_price, bid_size,"
            " ask_size, terms, risk_mode, noarb_checked, source_seq, r_x, delta_x,"
            " skew_x, sigma_b, params_id, x_bid, x_ask, p_bid_raw, p_ask_raw)"
            " VALUES (?, 'm1', 0.4, 0.5, 1, 1, '{}', 'NORMAL', 1, 0, 0, 0, 0, 0,"
            " 'p', 0, 0, 0, 0)", (stale,))
    assert "no quotes" in bot.cmd_quotes(src)


def test_cmd_markout_missing_report(env):
    src, _ = env
    assert "no markout report" in bot.cmd_markout(src)


def test_cmd_markout_renders_regions(env, tmp_path):
    import json
    src, _ = env
    out_dir = Path(src._status.run_info["out_dir"])
    (out_dir / "markout_report.json").write_text(json.dumps({
        "by_region": {"belly": {"60": {"n": 2, "n_attempted": 3,
                                        "mk_avg": -0.004, "mk_total": -0.008}}},
        "generated_ts": NOW.isoformat(),
    }), encoding="ascii")
    reply = bot.cmd_markout(src)
    assert "belly" in reply
    assert "-0.0040" in reply
    assert "n=2/3" in reply


def test_no_state_db_degrades(tmp_path):
    """Handlers report 'no state db' instead of raising when discovery fails."""
    status = _status(run_info=None, heartbeat=None, state="STOPPED")
    src = StubSource(tmp_path / "control", status,
                     state_db_override=tmp_path / "missing.db")
    assert "no state db" in bot.cmd_bankroll(src)
    assert "no state db" in bot.cmd_fills(src)
    assert "no state db" in bot.cmd_inventory(src)
    assert "no state db" in bot.cmd_pnl(src)
    assert "no state db" in bot.cmd_quotes(src)


def test_reply_truncation(env, monkeypatch):
    src, _ = env
    monkeypatch.setitem(bot.COMMANDS, "/status", lambda s: "x" * 10_000)
    reply = bot.handle_command("/status", src)
    assert len(reply) < 4096
    assert reply.endswith("[truncated]")
