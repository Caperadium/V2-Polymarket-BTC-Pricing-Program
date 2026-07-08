"""Engine start/stop/status control for the Stage-B paper runner (plan Step 4).

Stdlib only, cross-platform (Linux VPS is the deployment target, Windows is the
dev box). This module is the ONLY thing that spawns/monitors/stops the paper
runner subprocess; `market_maker/paper_runner.py` (the child) and
`app/pages/mm_monitor.py` (the caller) are implemented separately and are out
of scope here.

Control-file protocol (see plan "Control-file protocol" table). All files live
in a single control directory (default `temp/paper_run/control/`):

  mm_paper.pid       ASCII pid. Written by the launcher (this module)
                      immediately after Popen; the child rewrites the same
                      value at its own startup. Deleted by the child in its
                      `finally` block. Present + dead pid => CRASHED.
  mm_paper.stop      Written by `request_stop()` (or an operator `touch`).
                      First line = target pid (empty/unparseable = universal
                      stop, matches ANY pid); second line = ISO timestamp.
                      The runner polls this each tick and ignores a stop file
                      stamped with a pid that is not its own -- this lets a
                      stale stop file from a previous run not kill a fresh
                      one that reused the same control dir.
  mm_paper.starting  O_CREAT|O_EXCL lock held by `start_engine()` around the
                      status-check + spawn; removed in a `finally` so a
                      failed start never leaves a permanent lock. A lock file
                      older than 60s is treated as orphaned (crash) and is
                      removed + retried once.
  current_run.json   Written by the runner: pid, started_utc, out_dir,
                      argv/config, and (in `finally`) ended_utc/exit_reason.
                      Never deleted -- always points at the latest run, even
                      after the process has exited. `exit_reason` values:
                      completed / stop_file / sigterm / sigint /
                      "error: <ExcType>: <msg>".

Per-run artifacts (heartbeat.json, runner.log, run_meta.json) live under
`<out_dir>/`, a per-run timestamped directory this module creates for the
launcher's log redirection; the runner's own `current_run.json["out_dir"]`
is the source of truth `engine_status()` uses to find the heartbeat file
(it may be absent while the child is still starting up -- handled
gracefully, see `engine_status()`).
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = PROJECT_ROOT / "temp" / "paper_run" / "control"

PID_FILE = "mm_paper.pid"
STOP_FILE = "mm_paper.stop"
START_LOCK = "mm_paper.starting"
RUN_JSON = "current_run.json"

DEFAULT_CONFIG = PROJECT_ROOT / "market_maker" / "paper_run_config.json"

_START_LOCK_FRESH_S = 60.0
_STARTING_GRACE_S = 120.0
_DEFAULT_STALLED_AFTER_S = 60.0


@dataclass
class EngineStatus:
    """Snapshot of the paper-runner engine's current state.

    state: one of "RUNNING", "STARTING", "STALLED", "STOPPED", "CRASHED".
    """

    state: str
    pid: Optional[int]
    run_info: Optional[Dict[str, Any]]
    heartbeat: Optional[Dict[str, Any]]
    heartbeat_age_s: Optional[float]
    out_dir: Optional[Path]
    detail: str


# ---------------------------------------------------------------------------
# pid liveness (no psutil)
# ---------------------------------------------------------------------------


def pid_alive(pid: Optional[int]) -> bool:
    """Return True if `pid` refers to a live process. No psutil dependency."""
    if pid is None:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we don't own it -- still alive.
            return True
        except OSError:
            return False
        else:
            return True
    return _pid_alive_win32(pid)


def _pid_alive_win32(pid: int) -> bool:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong(0)
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


# ---------------------------------------------------------------------------
# small file helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        # Tolerate a torn/partially-written file (mid-write race) -- caller
        # falls back to file-mtime-only liveness where applicable.
        return None


def _read_pid(path: Path) -> Optional[int]:
    try:
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            content = f.read().strip()
        return int(content)
    except (OSError, ValueError):
        return None


def _write_pid(path: Path, pid: int) -> None:
    with open(path, "w", encoding="ascii") as f:
        f.write(str(pid))


def _parse_iso(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _read_heartbeat(out_dir: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Return (heartbeat_dict_or_None, age_seconds_or_None).

    Liveness is derived from file mtime even if the JSON body is torn by a
    concurrent write (parse errors are tolerated -- age is still returned).
    """
    if not out_dir:
        return None, None
    hb_path = Path(out_dir) / "heartbeat.json"
    try:
        mtime = hb_path.stat().st_mtime
    except OSError:
        return None, None
    age = max(0.0, time.time() - mtime)
    heartbeat = _read_json(hb_path)
    return heartbeat, age


def _heartbeat_threshold(
    run_info: Optional[Dict[str, Any]],
    heartbeat: Optional[Dict[str, Any]],
    stalled_after_s: Optional[float],
) -> float:
    tick_s = None
    if isinstance(heartbeat, dict):
        tick_s = heartbeat.get("tick_s")
    if tick_s is None and run_info:
        tick_s = run_info.get("tick_s")
        if tick_s is None and isinstance(run_info.get("config"), dict):
            tick_s = run_info["config"].get("tick_s")
    reprice_s = heartbeat.get("reprice_s") if isinstance(heartbeat, dict) else None
    if tick_s is not None:
        try:
            base = 3.0 * float(tick_s)
        except (TypeError, ValueError):
            base = None
        if base is not None:
            # A reprice tick blocks the loop (and thus the heartbeat) for the
            # full calculate_probabilities call -- minutes. Allow one reprice
            # duration (bounded by reprice_s cadence + margin) before STALLED.
            try:
                if reprice_s is not None:
                    return max(base, float(reprice_s) + 60.0)
            except (TypeError, ValueError):
                pass
            return base
    return float(stalled_after_s) if stalled_after_s is not None else _DEFAULT_STALLED_AFTER_S


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def engine_status(
    control_dir: Path = CONTROL_DIR,
    stalled_after_s: Optional[float] = None,
) -> EngineStatus:
    """Derive the current engine state from the control-dir files.

    See module docstring for the file protocol. Never raises on missing or
    torn files -- every field degrades to None/best-effort.
    """
    control_dir = Path(control_dir)
    pid_path = control_dir / PID_FILE
    run_json_path = control_dir / RUN_JSON

    run_info = _read_json(run_json_path)
    pid = _read_pid(pid_path)

    if pid is None:
        detail = "not running"
        if run_info:
            exit_reason = run_info.get("exit_reason")
            if exit_reason:
                if isinstance(exit_reason, str) and exit_reason.startswith("error:"):
                    detail = "CRASHED (last run): " + exit_reason
                else:
                    detail = "stopped; last exit_reason=" + str(exit_reason)
        return EngineStatus(
            state="STOPPED", pid=None, run_info=run_info, heartbeat=None,
            heartbeat_age_s=None, out_dir=None, detail=detail,
        )

    if not pid_alive(pid):
        try:
            pid_path.unlink()
        except OSError:
            pass
        return EngineStatus(
            state="CRASHED", pid=pid, run_info=run_info, heartbeat=None,
            heartbeat_age_s=None, out_dir=None,
            detail="stale PID file removed (pid %d not alive)" % pid,
        )

    out_dir_raw = run_info.get("out_dir") if run_info else None
    out_dir = Path(out_dir_raw) if out_dir_raw else None
    heartbeat, heartbeat_age_s = _read_heartbeat(out_dir_raw)

    if heartbeat_age_s is not None:
        threshold = _heartbeat_threshold(run_info, heartbeat, stalled_after_s)
        if heartbeat_age_s < threshold:
            state = "RUNNING"
            detail = "heartbeat age %.1fs (threshold %.1fs)" % (heartbeat_age_s, threshold)
        else:
            state = "STALLED"
            detail = "heartbeat stale: age %.1fs >= threshold %.1fs" % (heartbeat_age_s, threshold)
    else:
        started_utc = _parse_iso(run_info.get("started_utc")) if run_info else None
        age = (datetime.now(timezone.utc) - started_utc).total_seconds() if started_utc else None
        if age is None or age < _STARTING_GRACE_S:
            state = "STARTING"
            detail = "no heartbeat yet"
        else:
            state = "STALLED"
            detail = "no heartbeat after startup grace (%.0fs)" % _STARTING_GRACE_S

    return EngineStatus(
        state=state, pid=pid, run_info=run_info, heartbeat=heartbeat,
        heartbeat_age_s=heartbeat_age_s, out_dir=out_dir, detail=detail,
    )


# ---------------------------------------------------------------------------
# start lock
# ---------------------------------------------------------------------------


def _acquire_start_lock(lock_path: Path, retried: bool = False) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            # Lock disappeared between the failed open and the stat -- someone
            # else released it; the caller's own open attempt lost the race
            # so treat as still-held rather than recursing forever.
            return False
        if age < _START_LOCK_FRESH_S:
            return False
        if retried:
            return False
        try:
            lock_path.unlink()
        except OSError:
            pass
        return _acquire_start_lock(lock_path, retried=True)


def _release_start_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# start / stop / kill
# ---------------------------------------------------------------------------


def start_engine(
    config_path: Path = DEFAULT_CONFIG,
    control_dir: Path = CONTROL_DIR,
    cmd: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Spawn the paper-runner subprocess (detached, non-blocking).

    Returns (ok, message). Refuses if the engine is already
    RUNNING/STARTING/STALLED, or if another start is in progress (fresh
    START_LOCK held). `cmd` overrides the default argv (test hook).
    """
    control_dir = Path(control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    lock_path = control_dir / START_LOCK

    if not _acquire_start_lock(lock_path):
        return False, "start already in progress (lock held): %s" % lock_path

    try:
        status = engine_status(control_dir=control_dir)
        if status.state in ("RUNNING", "STARTING", "STALLED"):
            return False, "engine already %s (pid=%s)" % (status.state, status.pid)

        stop_path = control_dir / STOP_FILE
        if stop_path.exists():
            try:
                stop_path.unlink()
            except OSError:
                pass

        out_dir = PROJECT_ROOT / "temp" / "paper_run" / datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "runner.log"

        if cmd is None:
            cmd = [
                sys.executable, "-m", "market_maker.paper_runner",
                "--config", str(config_path),
                "--out", str(out_dir),
                "--control-dir", str(control_dir),
            ]

        log_fh = open(log_path, "ab")
        try:
            popen_kwargs: Dict[str, Any] = dict(
                cwd=str(PROJECT_ROOT),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            else:
                popen_kwargs["creationflags"] = (
                    subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                )
            proc = subprocess.Popen(cmd, **popen_kwargs)
        finally:
            log_fh.close()

        pid_path = control_dir / PID_FILE
        _write_pid(pid_path, proc.pid)

        return True, "engine started (pid=%d, out_dir=%s)" % (proc.pid, out_dir)
    finally:
        _release_start_lock(lock_path)


def request_stop(control_dir: Path = CONTROL_DIR) -> Tuple[bool, str]:
    """Write the PID-stamped stop file. Non-blocking."""
    control_dir = Path(control_dir)
    pid_path = control_dir / PID_FILE
    pid = _read_pid(pid_path)
    if pid is None:
        return False, "not running"
    stop_path = control_dir / STOP_FILE
    ts = datetime.now(timezone.utc).isoformat()
    with open(stop_path, "w", encoding="ascii") as f:
        f.write(str(pid) + "\n" + ts + "\n")
    return True, "stop requested for pid %d" % pid


def stop_engine(timeout_s: float = 120.0, control_dir: Path = CONTROL_DIR) -> Tuple[bool, str]:
    """Request a graceful stop and poll until the pid dies or timeout. CLI/test convenience."""
    control_dir = Path(control_dir)
    pid_path = control_dir / PID_FILE
    pid = _read_pid(pid_path)
    if pid is None:
        return True, "not running"

    ok, msg = request_stop(control_dir=control_dir)
    if not ok:
        return False, msg

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not pid_alive(pid):
            return True, "engine stopped (pid=%d)" % pid
        time.sleep(0.5)
    return False, "timeout waiting for engine (pid=%d) to stop" % pid


def kill_engine(control_dir: Path = CONTROL_DIR) -> Tuple[bool, str]:
    """Escalation: SIGKILL (posix) / TerminateProcess (win32). Cleans the PID file."""
    control_dir = Path(control_dir)
    pid_path = control_dir / PID_FILE
    pid = _read_pid(pid_path)
    if pid is None:
        return False, "not running"

    if os.name == "posix":
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            return False, "failed to kill pid %d: %s" % (pid, exc)
    else:
        PROCESS_TERMINATE = 0x0001
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if handle:
            try:
                kernel32.TerminateProcess(handle, 1)
            finally:
                kernel32.CloseHandle(handle)

    try:
        pid_path.unlink()
    except OSError:
        pass
    return True, "engine killed (pid=%d)" % pid
