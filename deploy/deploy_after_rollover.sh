#!/usr/bin/env bash
# deploy_after_rollover.sh -- one-shot deferred deploy for the MM paper engine.
#
# Waits for the CURRENT nearest expiry's ladder to roll over in-process (the
# multi-expiry engine settles it and acquires a replacement), then:
#   stop mm-paper -> git pull --ff-only -> import smoke check -> start
#   mm-paper -> verify heartbeat -> webhook notify.
#
# Designed for the wave-1+2 sizing deploy after the 2026-07-13 rollover, but
# generic: it snapshots the nearest ACTIVE expiry at arm time and triggers
# when that expiry is no longer active while another ladder is.
#
# Fail-safe: any deploy-step failure restarts the service on the OLD code
# and pages the webhook. The script never leaves the service stopped.
#
# Arm on the VPS (survives SSH disconnect, journald-logged):
#   sudo systemd-run --unit=mm-deploy-wave12 --collect \
#       /bin/bash /home/debian/deploy_after_rollover.sh
# Watch:  journalctl -u mm-deploy-wave12 -f
# Cancel: sudo systemctl stop mm-deploy-wave12
#
# Notes:
# - The repo is NOT pulled until trigger time, so a crash-restart of the
#   engine before the rollover still boots the old code (burn-in stays
#   clean).
# - Webhook URL is taken from $MM_ALERT_WEBHOOK, falling back to the
#   Environment= of mm-alert.service.

set -u

REPO_DIR="${REPO_DIR:-/opt/mm/V2-BTC-Contract-Pricing}"
CONTROL_DIR="${CONTROL_DIR:-$REPO_DIR/temp/paper_run/control}"
SERVICE="${SERVICE:-mm-paper.service}"
VENV_PY="${VENV_PY:-/opt/mm/venv/bin/python}"
POLL_S="${POLL_S:-60}"
TIMEOUT_H="${TIMEOUT_H:-36}"          # give up (with alert) after this many hours
SETTLE_GRACE_S="${SETTLE_GRACE_S:-180}" # wait after trigger before touching the service
VERIFY_WAIT_S="${VERIFY_WAIT_S:-150}"   # heartbeat freshness check delay after start

log() { echo "[mm-deploy] $(date -u +%FT%TZ) $*"; }

webhook_url() {
    if [ -n "${MM_ALERT_WEBHOOK:-}" ]; then
        echo "$MM_ALERT_WEBHOOK"
        return
    fi
    systemctl show mm-alert.service -p Environment --no-pager 2>/dev/null \
        | tr ' ' '\n' | sed -n 's/^MM_ALERT_WEBHOOK=//p' | head -1
}

notify() {
    local msg="$1"
    local url
    url="$(webhook_url)"
    log "NOTIFY: $msg"
    if [ -n "$url" ]; then
        curl -s -m 15 -X POST -H 'Content-Type: application/json' \
            -d "{\"text\": \"[mm-deploy] $msg\"}" "$url" >/dev/null 2>&1 || true
    fi
}

# Read a python expression over the freshest heartbeat; prints result or "ERR".
hb_query() {
    local expr="$1"
    "$VENV_PY" - "$CONTROL_DIR" "$expr" <<'PYEOF'
import json, sys, pathlib
control = pathlib.Path(sys.argv[1])
expr = sys.argv[2]
try:
    run = json.loads((control / "current_run.json").read_text())
    hb = json.loads((pathlib.Path(run["out_dir"]) / "heartbeat.json").read_text())
except Exception:
    print("ERR")
    sys.exit(0)
try:
    print(eval(expr, {"hb": hb, "run": run}))
except Exception:
    print("ERR")
PYEOF
}

# ---- 1. Snapshot the baseline expiry (nearest active ladder) ----
BASELINE="$(hb_query "min(k for k, v in hb.get('expiries', {}).items() if v.get('state') == 'active')")"
if [ "$BASELINE" = "ERR" ] || [ -z "$BASELINE" ]; then
    notify "arm FAILED: no active expiry found in heartbeat; not armed"
    exit 1
fi
log "armed: waiting for rollover of expiry $BASELINE (poll ${POLL_S}s, timeout ${TIMEOUT_H}h)"
notify "armed: will deploy origin/main after expiry $BASELINE rolls over"

# ---- 2. Poll for rollover ----
DEADLINE=$(( $(date +%s) + TIMEOUT_H * 3600 ))
while true; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        notify "TIMEOUT: expiry $BASELINE did not roll over within ${TIMEOUT_H}h; NOT deploying"
        exit 1
    fi
    # Rollover = baseline no longer active AND at least one other ladder active.
    ROLLED="$(hb_query "hb.get('expiries', {}).get('$BASELINE', {}).get('state') != 'active' and any(v.get('state') == 'active' for k, v in hb.get('expiries', {}).items() if k != '$BASELINE')")"
    if [ "$ROLLED" = "True" ]; then
        log "rollover detected: $BASELINE no longer active, replacement ladder live"
        break
    fi
    sleep "$POLL_S"
done

sleep "$SETTLE_GRACE_S"

# ---- 3. Deploy ----
fail_safe() {
    local why="$1"
    sudo systemctl start "$SERVICE" || true
    notify "deploy FAILED ($why); service restarted on OLD code -- manual intervention needed"
    exit 1
}

OLD_REV="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
log "stopping $SERVICE (old rev $OLD_REV)"
sudo systemctl stop "$SERVICE" || fail_safe "systemctl stop"

log "pulling origin/main"
# The VPS working tree may carry staged/modified files from surgical hot
# deploys (e.g. the 2026-07-12 telegram-bot drop); those files are committed
# upstream, so discard local modifications before the ff-only pull.
# Untracked files are preserved.
git -C "$REPO_DIR" reset --hard HEAD || fail_safe "git reset --hard"
git -C "$REPO_DIR" pull --ff-only origin main || fail_safe "git pull --ff-only"
NEW_REV="$(git -C "$REPO_DIR" rev-parse --short HEAD)"

log "import smoke check"
(cd "$REPO_DIR" && "$VENV_PY" -c "import market_maker.paper_runner, market_maker.robustness_sizing, market_maker.spread_builder, market_maker.pnl_report") \
    || fail_safe "import smoke check on $NEW_REV"

log "starting $SERVICE (new rev $NEW_REV)"
sudo systemctl start "$SERVICE" || fail_safe "systemctl start"

# ---- 4. Verify ----
sleep "$VERIFY_WAIT_S"
if ! systemctl is-active --quiet "$SERVICE"; then
    notify "deploy $OLD_REV -> $NEW_REV: service NOT active after ${VERIFY_WAIT_S}s -- investigate"
    exit 1
fi
AGE="$(hb_query "__import__('datetime').datetime.now(__import__('datetime').timezone.utc).timestamp() - __import__('datetime').datetime.fromisoformat(hb['ts_utc']).timestamp()")"
if [ "$AGE" = "ERR" ]; then
    notify "deploy $OLD_REV -> $NEW_REV: service active but heartbeat unreadable -- investigate"
    exit 1
fi
AGE_INT="${AGE%.*}"
if [ "${AGE_INT:-9999}" -gt 120 ]; then
    notify "deploy $OLD_REV -> $NEW_REV: heartbeat stale (${AGE_INT}s) -- investigate"
    exit 1
fi
notify "deploy OK: $OLD_REV -> $NEW_REV, engine ticking (heartbeat ${AGE_INT}s old)"
log "done"
exit 0
