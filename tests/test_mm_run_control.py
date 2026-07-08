"""Tests for market_maker.run_control (plan Step 4/6.2).

Drives start_engine() against a dummy "protocol" subprocess (via the `cmd`
override) that implements the same control-file contract the real
market_maker/paper_runner.py is expected to follow: rewrite the PID file at
startup, write current_run.json + a heartbeat file, honor a PID-stamped stop
file, and clean up PID/stop files on exit. This lets the whole
start/status/stop/kill lifecycle be exercised without importing or running
the real runner (which is being modified concurrently -- this suite must
never import market_maker.paper_runner).

ALWAYS pass control_dir=tmp_path (or a subdirectory of it) -- never touch
the real repo control dir. `start_engine()` itself always creates its
per-run log/out directory under the real `temp/paper_run/<ts>/` (that path
is not parameterized, matching plan Step 4); the `cleanup_real_run_dirs`
fixture removes anything created there during a test so the repo temp dir
does not accumulate test artifacts (it is gitignored regardless).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import market_maker.run_control as run_control

POLL_DEADLINE_S = 10.0
POLL_INTERVAL_S = 0.05

REAL_PAPER_RUN_DIR = run_control.PROJECT_ROOT / "temp" / "paper_run"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cleanup_real_run_dirs():
    """start_engine() always writes its per-run log dir under the real repo
    temp/paper_run/<ts>/ (not parameterized by control_dir). Snapshot before
    and remove anything new after, so tests don't litter the real repo tree.
    """
    before = set()
    if REAL_PAPER_RUN_DIR.exists():
        before = {p.name for p in REAL_PAPER_RUN_DIR.iterdir() if p.is_dir()}
    yield
    if REAL_PAPER_RUN_DIR.exists():
        after = {p.name for p in REAL_PAPER_RUN_DIR.iterdir() if p.is_dir()}
        for name in after - before:
            if name == "control":
                continue
            shutil.rmtree(REAL_PAPER_RUN_DIR / name, ignore_errors=True)


_DUMMY_SCRIPT = '''
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

control_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
tick_s = float(sys.argv[3])
hb_delay_s = float(sys.argv[4])

pid = os.getpid()
control_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

pid_path = control_dir / "mm_paper.pid"
stop_path = control_dir / "mm_paper.stop"
run_json_path = control_dir / "current_run.json"

# Runner-side protocol: rewrite own pid (matches the value the launcher
# already wrote right after Popen).
pid_path.write_text(str(pid), encoding="ascii")

run_info = {
    "pid": pid,
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "out_dir": str(out_dir),
    "tick_s": tick_s,
}
run_json_path.write_text(json.dumps(run_info), encoding="utf-8")

time.sleep(hb_delay_s)

hb_tmp = out_dir / "heartbeat.json.tmp"
hb_path = out_dir / "heartbeat.json"


def write_heartbeat(tick):
    # Rewritten every iteration, like the real runner writes per tick.
    # A one-shot heartbeat leaves RUNNING observable for only 3*tick_s
    # before engine_status flips to STALLED -- a flaky 3s window under
    # load (observed ~1-in-8 full-suite failures before this).
    hb_tmp.write_text(
        json.dumps({"ts_utc": datetime.now(timezone.utc).isoformat(),
                    "tick": tick, "tick_s": tick_s}),
        encoding="utf-8",
    )
    os.replace(str(hb_tmp), str(hb_path))


tick = 0
deadline = time.time() + 30.0
while time.time() < deadline:
    tick += 1
    write_heartbeat(tick)
    if stop_path.exists():
        try:
            lines = stop_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        first = lines[0].strip() if lines else ""
        should_stop = True
        if first:
            try:
                target_pid = int(first)
                should_stop = (target_pid == pid)
            except ValueError:
                should_stop = True
        if should_stop:
            break
    time.sleep(0.05)

for p in (pid_path, stop_path):
    try:
        p.unlink()
    except OSError:
        pass
'''


def _write_dummy_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "dummy_runner.py"
    script_path.write_text(_DUMMY_SCRIPT, encoding="utf-8")
    return script_path


def _dummy_cmd(script_path: Path, control_dir: Path, out_dir: Path, tick_s=1.0, hb_delay_s=0.2):
    return [
        sys.executable, str(script_path),
        str(control_dir), str(out_dir), str(tick_s), str(hb_delay_s),
    ]


def _poll_until(predicate, deadline_s=POLL_DEADLINE_S, interval_s=POLL_INTERVAL_S):
    deadline = time.time() + deadline_s
    result = predicate()
    while time.time() < deadline and not result:
        time.sleep(interval_s)
        result = predicate()
    return result


def _wait_state(control_dir: Path, state: str, deadline_s=POLL_DEADLINE_S):
    status_holder = {}

    def _check():
        status_holder["status"] = run_control.engine_status(control_dir=control_dir)
        return status_holder["status"].state == state

    _poll_until(_check, deadline_s=deadline_s)
    return status_holder["status"]


def _stop_and_wait(control_dir: Path, deadline_s=POLL_DEADLINE_S):
    run_control.request_stop(control_dir=control_dir)
    _poll_until(lambda: not (control_dir / run_control.PID_FILE).exists(), deadline_s=deadline_s)


# ---------------------------------------------------------------------------
# pid_alive
# ---------------------------------------------------------------------------


def test_pid_alive_self_is_true():
    assert run_control.pid_alive(os.getpid()) is True


def test_pid_alive_spawned_then_killed():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        alive = _poll_until(lambda: run_control.pid_alive(proc.pid))
        assert alive is True
    finally:
        proc.kill()
        proc.wait(timeout=10)
    dead = _poll_until(lambda: not run_control.pid_alive(proc.pid))
    assert dead is True


# ---------------------------------------------------------------------------
# start -> STARTING/RUNNING -> stop -> STOPPED, files cleaned
# ---------------------------------------------------------------------------


def test_start_running_then_stop(tmp_path, cleanup_real_run_dirs):
    control_dir = tmp_path / "control"
    out_dir = tmp_path / "run_out"
    script = _write_dummy_script(tmp_path)
    cmd = _dummy_cmd(script, control_dir, out_dir, tick_s=1.0, hb_delay_s=0.2)

    ok, msg = run_control.start_engine(control_dir=control_dir, cmd=cmd)
    assert ok, msg

    status = run_control.engine_status(control_dir=control_dir)
    assert status.state in ("STARTING", "RUNNING")
    assert status.pid is not None

    status = _wait_state(control_dir, "RUNNING")
    assert status.state == "RUNNING", status.detail
    assert status.heartbeat_age_s is not None

    ok, msg = run_control.request_stop(control_dir=control_dir)
    assert ok, msg

    status = _wait_state(control_dir, "STOPPED")
    assert status.state == "STOPPED", status.detail
    assert not (control_dir / run_control.PID_FILE).exists()
    assert not (control_dir / run_control.STOP_FILE).exists()


# ---------------------------------------------------------------------------
# stale PID -> CRASHED + cleanup, restart allowed afterwards
# ---------------------------------------------------------------------------


def test_stale_pid_is_crashed_and_cleaned_then_restart_allowed(tmp_path, cleanup_real_run_dirs):
    control_dir = tmp_path / "control"
    control_dir.mkdir(parents=True)

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    dead_pid = proc.pid
    proc.kill()
    proc.wait(timeout=10)
    _poll_until(lambda: not run_control.pid_alive(dead_pid))
    assert run_control.pid_alive(dead_pid) is False

    pid_path = control_dir / run_control.PID_FILE
    pid_path.write_text(str(dead_pid), encoding="ascii")

    status = run_control.engine_status(control_dir=control_dir)
    assert status.state == "CRASHED"
    assert not pid_path.exists()

    script = _write_dummy_script(tmp_path)
    out_dir = tmp_path / "run_out2"
    cmd = _dummy_cmd(script, control_dir, out_dir, tick_s=1.0, hb_delay_s=0.2)
    ok, msg = run_control.start_engine(control_dir=control_dir, cmd=cmd)
    assert ok, msg

    _stop_and_wait(control_dir)


# ---------------------------------------------------------------------------
# double-start refused: status guard, and O_EXCL lock
# ---------------------------------------------------------------------------


def test_double_start_refused_while_running(tmp_path, cleanup_real_run_dirs):
    control_dir = tmp_path / "control"
    script = _write_dummy_script(tmp_path)
    out_dir = tmp_path / "run_out3"
    cmd = _dummy_cmd(script, control_dir, out_dir, tick_s=1.0, hb_delay_s=5.0)

    ok, msg = run_control.start_engine(control_dir=control_dir, cmd=cmd)
    assert ok, msg

    status = run_control.engine_status(control_dir=control_dir)
    assert status.state in ("STARTING", "RUNNING", "STALLED")

    ok2, msg2 = run_control.start_engine(control_dir=control_dir, cmd=cmd)
    assert ok2 is False
    assert "already" in msg2.lower()

    _stop_and_wait(control_dir)


def test_double_start_refused_while_lock_held(tmp_path, cleanup_real_run_dirs):
    control_dir = tmp_path / "control"
    control_dir.mkdir(parents=True)
    lock_path = control_dir / run_control.START_LOCK
    lock_path.write_text("", encoding="ascii")

    ok, msg = run_control.start_engine(
        control_dir=control_dir, cmd=[sys.executable, "-c", "pass"]
    )
    assert ok is False
    assert "lock" in msg.lower()
    # start_engine must not have removed a fresh lock it didn't create.
    assert lock_path.exists()


# ---------------------------------------------------------------------------
# stop file with mismatched pid stamp is ignored
# ---------------------------------------------------------------------------


def test_stop_file_mismatched_pid_is_ignored(tmp_path, cleanup_real_run_dirs):
    control_dir = tmp_path / "control"
    script = _write_dummy_script(tmp_path)
    out_dir = tmp_path / "run_out4"
    cmd = _dummy_cmd(script, control_dir, out_dir, tick_s=1.0, hb_delay_s=0.2)

    ok, msg = run_control.start_engine(control_dir=control_dir, cmd=cmd)
    assert ok, msg

    status = _wait_state(control_dir, "RUNNING")
    assert status.state == "RUNNING", status.detail
    real_pid = status.pid
    assert real_pid is not None

    stop_path = control_dir / run_control.STOP_FILE
    stop_path.write_text(str(real_pid + 1) + "\n2026-01-01T00:00:00+00:00\n", encoding="ascii")

    time.sleep(1.0)
    assert run_control.pid_alive(real_pid) is True
    assert (control_dir / run_control.PID_FILE).exists()

    stop_path.unlink()
    _stop_and_wait(control_dir)
    status = run_control.engine_status(control_dir=control_dir)
    assert status.state == "STOPPED"


# ---------------------------------------------------------------------------
# request_stop / stop_engine / kill_engine when nothing is running
# ---------------------------------------------------------------------------


def test_request_stop_when_not_running(tmp_path):
    control_dir = tmp_path / "control"
    ok, msg = run_control.request_stop(control_dir=control_dir)
    assert ok is False
    assert msg == "not running"


def test_kill_engine_when_not_running(tmp_path):
    control_dir = tmp_path / "control"
    ok, msg = run_control.kill_engine(control_dir=control_dir)
    assert ok is False
    assert msg == "not running"


def test_engine_status_stopped_when_empty(tmp_path):
    control_dir = tmp_path / "control"
    status = run_control.engine_status(control_dir=control_dir)
    assert status.state == "STOPPED"
    assert status.pid is None
