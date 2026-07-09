"""Tests for scripts/mm_alert_check.py decision logic (plan Workstream 3 / H3).

Pure-function tests only -- no network (webhook sends are monkeypatched) and
no dependency on a real paper-runner process. `market_maker.run_control.
EngineStatus` is constructed directly as a fixture rather than driving a real
engine_status() call, mirroring the isolation style of test_mm_run_control.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.run_control import EngineStatus  # noqa: E402

# scripts/ is not a package (no __init__.py) -- load the module by path so
# `import scripts.mm_alert_check` isn't required.
_SPEC = importlib.util.spec_from_file_location(
    "mm_alert_check", PROJECT_ROOT / "scripts" / "mm_alert_check.py"
)
mm_alert_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mm_alert_check)  # type: ignore[union-attr]


def _status(
    state: str = "RUNNING",
    heartbeat: Optional[dict] = None,
    run_info: Optional[dict] = None,
) -> EngineStatus:
    return EngineStatus(
        state=state, pid=1234, run_info=run_info, heartbeat=heartbeat,
        heartbeat_age_s=1.0, out_dir=None, detail="test",
    )


# ---------------------------------------------------------------------------
# _check_engine_state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["CRASHED", "STALLED"])
def test_check_engine_state_alerts_on_crashed_or_stalled(state):
    result = mm_alert_check._check_engine_state(_status(state=state))
    assert result is not None
    key, msg = result
    assert key == "state_%s" % state.lower()
    assert state in msg


@pytest.mark.parametrize("state", ["RUNNING", "STARTING", "STOPPED"])
def test_check_engine_state_silent_otherwise(state):
    assert mm_alert_check._check_engine_state(_status(state=state)) is None


# ---------------------------------------------------------------------------
# _check_feed_unhealthy (streak tracked across calls via `state`)
# ---------------------------------------------------------------------------


def test_feed_unhealthy_no_heartbeat_is_silent():
    state = {}
    assert mm_alert_check._check_feed_unhealthy(None, state, now=1000.0) is None
    assert state == {}


def test_feed_unhealthy_true_resets_baseline_no_alert():
    state = {"last_feed_healthy_ts": 0.0}
    result = mm_alert_check._check_feed_unhealthy({"feed_healthy": True}, state, now=1000.0)
    assert result is None
    assert state["last_feed_healthy_ts"] == 1000.0


def test_feed_unhealthy_first_observation_starts_clock_no_alert():
    state = {}
    result = mm_alert_check._check_feed_unhealthy({"feed_healthy": False}, state, now=1000.0)
    assert result is None
    assert state["last_feed_healthy_ts"] == 1000.0


def test_feed_unhealthy_under_threshold_no_alert():
    state = {"last_feed_healthy_ts": 1000.0}
    now = 1000.0 + mm_alert_check.FEED_UNHEALTHY_ALERT_S - 1.0
    result = mm_alert_check._check_feed_unhealthy({"feed_healthy": False}, state, now=now)
    assert result is None


def test_feed_unhealthy_over_threshold_alerts():
    state = {"last_feed_healthy_ts": 1000.0}
    now = 1000.0 + mm_alert_check.FEED_UNHEALTHY_ALERT_S + 1.0
    result = mm_alert_check._check_feed_unhealthy({"feed_healthy": False}, state, now=now)
    assert result is not None
    key, msg = result
    assert key == "feed_unhealthy"
    assert "900s" in msg or "%.0f" % (mm_alert_check.FEED_UNHEALTHY_ALERT_S + 1.0) in msg


# ---------------------------------------------------------------------------
# _check_btc_stale
# ---------------------------------------------------------------------------


def test_btc_stale_no_heartbeat_is_silent():
    assert mm_alert_check._check_btc_stale(None) is None


def test_btc_stale_missing_field_is_silent():
    assert mm_alert_check._check_btc_stale({"tick": 1}) is None


def test_btc_stale_under_threshold_no_alert():
    threshold = mm_alert_check.BTC_STALE_MAX_S_DEFAULT * mm_alert_check.BTC_STALE_ALERT_MULT
    result = mm_alert_check._check_btc_stale({"btc_data_age_s": threshold - 1.0})
    assert result is None


def test_btc_stale_over_threshold_alerts():
    threshold = mm_alert_check.BTC_STALE_MAX_S_DEFAULT * mm_alert_check.BTC_STALE_ALERT_MULT
    result = mm_alert_check._check_btc_stale({"btc_data_age_s": threshold + 1.0})
    assert result is not None
    key, _msg = result
    assert key == "btc_stale"


def test_btc_stale_respects_custom_threshold_arg():
    # A custom --btc-stale-max-s should shift the 2x alert threshold too.
    custom_max = 100.0
    result = mm_alert_check._check_btc_stale({"btc_data_age_s": 250.0}, btc_stale_max_s=custom_max)
    assert result is not None  # 250 > 2*100


# ---------------------------------------------------------------------------
# _check_resume_discrepancies (W0.1)
# ---------------------------------------------------------------------------


def test_resume_discrepancies_no_heartbeat_is_silent():
    assert mm_alert_check._check_resume_discrepancies(None) is None


def test_resume_discrepancies_missing_field_is_silent():
    assert mm_alert_check._check_resume_discrepancies({"tick": 1}) is None


def test_resume_discrepancies_zero_is_silent():
    assert mm_alert_check._check_resume_discrepancies({"resume_discrepancies": 0}) is None


def test_resume_discrepancies_positive_alerts():
    result = mm_alert_check._check_resume_discrepancies({"resume_discrepancies": 2})
    assert result is not None
    key, msg = result
    assert key == "resume_discrepancies"
    assert "2" in msg


def test_resume_discrepancies_alert_dedupes_within_window():
    # Mirrors the de-dupe pattern used by every other alert key: a key sent
    # within DEDUPE_WINDOW_S is suppressed on the next check.
    state = {}
    assert mm_alert_check._should_send("resume_discrepancies", state, now=1000.0) is True
    mm_alert_check._mark_sent("resume_discrepancies", state, now=1000.0)
    assert mm_alert_check._should_send("resume_discrepancies", state, now=1000.0 + 100.0) is False
    later = 1000.0 + mm_alert_check.DEDUPE_WINDOW_S + 1.0
    assert mm_alert_check._should_send("resume_discrepancies", state, now=later) is True


# ---------------------------------------------------------------------------
# _check_bankroll_frozen (W1.3)
# ---------------------------------------------------------------------------


def test_bankroll_frozen_no_heartbeat_is_silent():
    assert mm_alert_check._check_bankroll_frozen(None) is None


def test_bankroll_frozen_missing_field_is_silent():
    assert mm_alert_check._check_bankroll_frozen({"tick": 1}) is None


def test_bankroll_frozen_false_is_silent():
    assert mm_alert_check._check_bankroll_frozen({"bankroll_frozen": False}) is None


def test_bankroll_frozen_true_alerts():
    result = mm_alert_check._check_bankroll_frozen({"bankroll_frozen": True})
    assert result is not None
    key, msg = result
    assert key == "bankroll_frozen"
    assert "FROZEN" in msg


def test_bankroll_frozen_alert_dedupes_within_window():
    # Mirrors the de-dupe pattern used by every other alert key.
    state = {}
    assert mm_alert_check._should_send("bankroll_frozen", state, now=1000.0) is True
    mm_alert_check._mark_sent("bankroll_frozen", state, now=1000.0)
    assert mm_alert_check._should_send("bankroll_frozen", state, now=1000.0 + 100.0) is False
    later = 1000.0 + mm_alert_check.DEDUPE_WINDOW_S + 1.0
    assert mm_alert_check._should_send("bankroll_frozen", state, now=later) is True


# ---------------------------------------------------------------------------
# _check_disk_free
# ---------------------------------------------------------------------------


def test_disk_free_alerts_below_threshold(monkeypatch):
    class _Usage:
        free = 500 * 1024 * 1024  # 500 MB < 1 GB

    monkeypatch.setattr(mm_alert_check.shutil, "disk_usage", lambda path: _Usage())
    result = mm_alert_check._check_disk_free(Path("."))
    assert result is not None
    key, _msg = result
    assert key == "disk_low"


def test_disk_free_silent_above_threshold(monkeypatch):
    class _Usage:
        free = 10 * 1024 ** 3  # 10 GB

    monkeypatch.setattr(mm_alert_check.shutil, "disk_usage", lambda path: _Usage())
    assert mm_alert_check._check_disk_free(Path(".")) is None


def test_disk_free_silent_on_oserror(monkeypatch):
    def _raise(path):
        raise OSError("no such path")

    monkeypatch.setattr(mm_alert_check.shutil, "disk_usage", _raise)
    assert mm_alert_check._check_disk_free(Path("/nonexistent")) is None


# ---------------------------------------------------------------------------
# _check_settlement_timeout
# ---------------------------------------------------------------------------


def test_settlement_timeout_alerts_when_stopped_with_reason():
    status = _status(state="STOPPED", run_info={"exit_reason": "settlement_timeout"})
    result = mm_alert_check._check_settlement_timeout(status)
    assert result is not None
    key, _msg = result
    assert key == "settlement_timeout"


def test_settlement_timeout_silent_when_not_stopped():
    status = _status(state="RUNNING", run_info={"exit_reason": "settlement_timeout"})
    assert mm_alert_check._check_settlement_timeout(status) is None


def test_settlement_timeout_silent_on_other_reason():
    status = _status(state="STOPPED", run_info={"exit_reason": "completed"})
    assert mm_alert_check._check_settlement_timeout(status) is None


def test_settlement_timeout_silent_on_missing_run_info():
    status = _status(state="STOPPED", run_info=None)
    assert mm_alert_check._check_settlement_timeout(status) is None


# ---------------------------------------------------------------------------
# _collect_alerts (integration of the individual checks)
# ---------------------------------------------------------------------------


def test_collect_alerts_empty_for_healthy_running_engine():
    status = _status(state="RUNNING", heartbeat={"feed_healthy": True, "btc_data_age_s": 10.0})
    state = {}
    alerts = mm_alert_check._collect_alerts(status, state, now=1000.0, repo_root=Path("."))
    assert alerts == []


def test_collect_alerts_multiple_conditions_all_fire(monkeypatch):
    class _Usage:
        free = 1  # essentially 0 bytes free

    monkeypatch.setattr(mm_alert_check.shutil, "disk_usage", lambda path: _Usage())
    threshold = mm_alert_check.BTC_STALE_MAX_S_DEFAULT * mm_alert_check.BTC_STALE_ALERT_MULT
    status = _status(state="STALLED", heartbeat={
        "feed_healthy": False, "btc_data_age_s": threshold + 1, "resume_discrepancies": 1,
    })
    state = {"last_feed_healthy_ts": 0.0}
    alerts = mm_alert_check._collect_alerts(status, state, now=threshold + 1000.0, repo_root=Path("."))
    keys = {k for k, _m in alerts}
    assert keys == {"state_stalled", "feed_unhealthy", "btc_stale", "resume_discrepancies", "disk_low"}


def test_collect_alerts_includes_bankroll_frozen(monkeypatch):
    class _Usage:
        free = 10 * 1024 ** 3  # healthy disk, isolates the bankroll_frozen key

    monkeypatch.setattr(mm_alert_check.shutil, "disk_usage", lambda path: _Usage())
    status = _status(state="RUNNING", heartbeat={
        "feed_healthy": True, "btc_data_age_s": 10.0, "bankroll_frozen": True,
    })
    state = {}
    alerts = mm_alert_check._collect_alerts(status, state, now=1000.0, repo_root=Path("."))
    keys = {k for k, _m in alerts}
    assert keys == {"bankroll_frozen"}


# ---------------------------------------------------------------------------
# dedupe (_should_send / _mark_sent)
# ---------------------------------------------------------------------------


def test_should_send_true_when_never_sent():
    assert mm_alert_check._should_send("k", {}, now=1000.0) is True


def test_should_send_false_within_window():
    state = {}
    mm_alert_check._mark_sent("k", state, now=1000.0)
    assert mm_alert_check._should_send("k", state, now=1000.0 + 100.0) is False


def test_should_send_true_after_window_elapses():
    state = {}
    mm_alert_check._mark_sent("k", state, now=1000.0)
    later = 1000.0 + mm_alert_check.DEDUPE_WINDOW_S + 1.0
    assert mm_alert_check._should_send("k", state, now=later) is True


def test_mark_sent_does_not_clobber_other_keys():
    state = {}
    mm_alert_check._mark_sent("a", state, now=1.0)
    mm_alert_check._mark_sent("b", state, now=2.0)
    assert state["alerts_sent"] == {"a": 1.0, "b": 2.0}


# ---------------------------------------------------------------------------
# _send_webhook
# ---------------------------------------------------------------------------


def test_send_webhook_prints_and_returns_true_when_env_unset(monkeypatch, capsys):
    monkeypatch.delenv("MM_ALERT_WEBHOOK", raising=False)
    assert mm_alert_check._send_webhook("hello") is True
    out = capsys.readouterr().out
    assert "hello" in out


def test_send_webhook_posts_and_returns_true_on_success(monkeypatch):
    monkeypatch.setenv("MM_ALERT_WEBHOOK", "http://example.invalid/webhook")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    captured = {}

    def _fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        captured["data"] = req.data
        return _Resp()

    monkeypatch.setattr(mm_alert_check.urllib.request, "urlopen", _fake_urlopen)
    assert mm_alert_check._send_webhook("hello world") is True
    assert captured["url"] == "http://example.invalid/webhook"
    assert b"hello world" in captured["data"]


def test_send_webhook_returns_false_on_network_error(monkeypatch, capsys):
    monkeypatch.setenv("MM_ALERT_WEBHOOK", "http://example.invalid/webhook")

    def _raise(req, timeout=10):
        raise OSError("connection refused")

    monkeypatch.setattr(mm_alert_check.urllib.request, "urlopen", _raise)
    assert mm_alert_check._send_webhook("hello") is False
    assert "failed to POST" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# daily heartbeat
# ---------------------------------------------------------------------------


def _utc(hour: int, day: int = 15):
    from datetime import datetime, timezone
    return datetime(2026, 7, day, hour, 30, tzinfo=timezone.utc)


def test_heartbeat_not_due_before_send_hour():
    assert not mm_alert_check._heartbeat_due({}, _utc(7), hour_utc=8)


def test_heartbeat_due_at_or_after_send_hour_when_never_sent():
    assert mm_alert_check._heartbeat_due({}, _utc(8), hour_utc=8)
    assert mm_alert_check._heartbeat_due({}, _utc(23), hour_utc=8)


def test_heartbeat_not_due_twice_same_day():
    state = {}
    now = _utc(9)
    assert mm_alert_check._heartbeat_due(state, now, hour_utc=8)
    mm_alert_check._mark_heartbeat_sent(state, now)
    assert not mm_alert_check._heartbeat_due(state, _utc(10), hour_utc=8)


def test_heartbeat_due_again_next_day():
    state = {}
    mm_alert_check._mark_heartbeat_sent(state, _utc(9, day=15))
    assert mm_alert_check._heartbeat_due(state, _utc(9, day=16), hour_utc=8)


def test_heartbeat_message_running_engine():
    hb = {"tick": 42, "feed_healthy": True, "fills_total": 3,
          "btc_data_age_s": 305.7, "feed_restarts": 0}
    msg = mm_alert_check._heartbeat_message(_status(heartbeat=hb), PROJECT_ROOT)
    assert "state=RUNNING" in msg
    assert "tick=42" in msg
    assert "fills=3" in msg
    assert "btc_age=306s" in msg
    assert "disk_free=" in msg


def test_heartbeat_message_stopped_with_exit_reason_and_no_heartbeat():
    status = _status(state="STOPPED", run_info={"exit_reason": "sigterm"})
    msg = mm_alert_check._heartbeat_message(status, PROJECT_ROOT)
    assert "state=STOPPED" in msg
    assert "exit_reason=sigterm" in msg
    assert "tick=" not in msg


def test_run_sends_heartbeat_once_per_day(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MM_ALERT_WEBHOOK", raising=False)
    monkeypatch.setenv("MM_HEARTBEAT_HOUR_UTC", "0")
    control_dir = tmp_path / "control"
    assert mm_alert_check.main(["--control-dir", str(control_dir)]) == 0
    first = capsys.readouterr().out
    assert "daily heartbeat" in first
    assert mm_alert_check.main(["--control-dir", str(control_dir)]) == 0
    second = capsys.readouterr().out
    assert "daily heartbeat" not in second


def test_run_heartbeat_disabled_by_env(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MM_ALERT_WEBHOOK", raising=False)
    monkeypatch.setenv("MM_HEARTBEAT_HOUR_UTC", "0")
    monkeypatch.setenv("MM_HEARTBEAT_DISABLE", "1")
    control_dir = tmp_path / "control"
    assert mm_alert_check.main(["--control-dir", str(control_dir)]) == 0
    assert "daily heartbeat" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main() never raises / always exits 0
# ---------------------------------------------------------------------------


def test_main_exits_zero_even_on_internal_error(monkeypatch):
    def _raise(argv):
        raise RuntimeError("boom")

    monkeypatch.setattr(mm_alert_check, "_run", _raise)
    assert mm_alert_check.main([]) == 0


def test_main_runs_end_to_end_against_tmp_control_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MM_ALERT_WEBHOOK", raising=False)
    control_dir = tmp_path / "control"
    rc = mm_alert_check.main(["--control-dir", str(control_dir)])
    assert rc == 0
    assert (control_dir / mm_alert_check.ALERT_STATE_FILENAME).exists()
