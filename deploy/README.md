# Stage-B paper runner: VPS deployment kit

This directory holds systemd unit templates and the operator runbook for
running `market_maker.paper_runner` unattended on a headless Debian 13 VPS
for a month at a time. Everything here is a template the operator copies
and edits -- nothing in `deploy/` is loaded automatically by the app.

Files:

- `mm-paper.service` -- the paper-trading engine itself.
- `mm-datafetch.service` / `mm-datafetch.timer` -- refreshes `DATA/btc_daily.csv`
  and `DATA/btc_intraday_1m.csv` every 30 minutes (nothing else on the VPS
  does this).
- `mm-alert.service` / `mm-alert.timer` -- runs `scripts/mm_alert_check.py`
  every 5 minutes and pages a webhook on a handful of fault conditions.
  Also sends one daily "still alive" heartbeat (state/tick/fills/disk
  one-liner) at the first check at/after 08:00 UTC -- `$MM_HEARTBEAT_HOUR_UTC`
  changes the hour, `$MM_HEARTBEAT_DISABLE=1` turns it off -- so webhook
  silence always means the alert pipeline itself is broken, never "nothing
  to report".

## 1. Install

```bash
# 1. Clone the repo to a stable path (matches WorkingDirectory= in the units
#    below -- pick one and use it consistently).
sudo mkdir -p /opt/mm
sudo git clone <your-remote-url> /opt/mm/V2-BTC-Contract-Pricing
cd /opt/mm/V2-BTC-Contract-Pricing

# 2. Create a venv and install dependencies. There is no repo-root
#    requirements.txt (yet) -- the practical minimal set for the paper
#    runner + its dependency chain (pricing engine, market data feed,
#    control/monitor pages) is:
python3 -m venv /opt/mm/venv
/opt/mm/venv/bin/pip install --upgrade pip
/opt/mm/venv/bin/pip install pandas numpy scipy arch hmmlearn xgboost \
    statsmodels websockets requests pytz streamlit
# `arch`/`hmmlearn`/`xgboost`/`statsmodels` back the GARCH/regime/directional
# components of core/pricing (regime + XGB are off by default in the paper
# runner's CachedEngine, but the modules must still import cleanly).
# `streamlit` is only needed if you also run the mm_monitor dashboard page
# from this box; the paper runner itself does not need it.

# 3. Fetch BTC data once before the first run (the runner does NOT do this
#    itself -- see mm-datafetch.timer below for the recurring refresh):
/opt/mm/venv/bin/python core/data/data_fetcher.py

# 4. Edit the unit templates: every file in this directory has <EDIT>
#    placeholders for WorkingDirectory (your clone path) and the venv
#    python path. Then install them:
sudo cp deploy/mm-paper.service deploy/mm-datafetch.service \
        deploy/mm-datafetch.timer deploy/mm-alert.service \
        deploy/mm-alert.timer /etc/systemd/system/
sudo systemctl daemon-reload

# 5. Set the alert webhook. Prefer a drop-in over editing mm-alert.service
#    in place (keeps your secret out of a file that might get overwritten
#    by re-copying an updated template):
sudo mkdir -p /etc/systemd/system/mm-alert.service.d
sudo tee /etc/systemd/system/mm-alert.service.d/webhook.conf <<'EOF'
[Service]
Environment=MM_ALERT_WEBHOOK=https://your-real-webhook-url
EOF
sudo systemctl daemon-reload

# 6. Enable + start everything. Timers first (so data is fresh before the
#    engine's first tick), then the engine:
sudo systemctl enable --now mm-datafetch.timer
sudo systemctl enable --now mm-alert.timer
sudo systemctl enable --now mm-paper.service
```

`market_maker/paper_run_config.json` (checked in) is the config the unit
points at: `event_slug: "auto"` (auto-rolls to the next `bitcoin-above`
event via `resolve_next_event`), `state_db: "market_maker/mm_paper_state.db"`
(persistent, survives restarts -- see "Resumable state" below), `minutes: 0`
(run indefinitely). Edit it in place if you want different tick/reprice
cadence or bankroll; no code changes needed.

**Start the 72h acceptance run (section 5) on a fresh `--state-db`.**
Resuming a pre-fix state-db that has an open BUY_NO position will show a
one-time step in the mm_monitor equity series at the deploy boundary (old
snapshots keep the phantom -0.20/share from the pre-C0 accounting bug, new
ones do not) -- harmless once understood, but avoid the confusion by
starting the acceptance test on a database created after this fix.

## 2. Checking status

```bash
# systemd's own view (active/inactive, recent log tail, restart count):
systemctl status mm-paper.service
systemctl status mm-datafetch.timer mm-alert.timer

# full log (journald captures stdout/stderr; the runner also writes its own
# temp/paper_run/<ts>/runner.log via the launcher's log redirect if started
# through run_control.start_engine() instead of systemd directly -- under
# systemd, journalctl is the source of truth):
journalctl -u mm-paper -f            # follow live
journalctl -u mm-paper --since "1 hour ago"
journalctl -u mm-alert --since today  # alert-check's own stdout/stderr

# programmatic status (same thing the mm_monitor dashboard page uses):
cd /opt/mm/V2-BTC-Contract-Pricing
/opt/mm/venv/bin/python -c "
from market_maker import run_control
s = run_control.engine_status()
print(s.state, s.detail)
print(s.heartbeat)
"
```

`engine_status()` returns one of `RUNNING` / `STARTING` / `STALLED` /
`STOPPED` / `CRASHED`:

- `RUNNING` -- heartbeat is fresh (age below `max(3*tick_s, reprice_s+60)`).
- `STARTING` -- process alive, no heartbeat yet, within 120s of launch
  (covers auto-event-resolution retries + WS warmup).
- `STALLED` -- process alive but heartbeat has gone stale past the above
  window (a genuinely wedged tick loop -- rare, since most failure modes
  now exit the process instead of hanging, see H1/M1 below).
- `STOPPED` -- no PID file; check `current_run.json`'s `exit_reason` for
  why the last run ended.
- `CRASHED` -- a PID file exists but that PID is not alive (process died
  without cleaning up its own PID file, e.g. SIGKILL).

`heartbeat.json` (under the run's `out_dir`, path is `EngineStatus.out_dir`)
fields worth knowing: `ts_utc`, `tick`, `feed_healthy`, `n_msgs`,
`fills_total`, `noarb_violations`, `unhealthy_ticks`, `pulled_ticks`,
`tick_s`, `reprice_s`, `btc_data_age_s` (staleness of
`DATA/btc_intraday_1m.csv` at the last tick), `feed_restarts` (how many
times the WS feed thread has been rebuilt this run).

`markout_report.json` (same `out_dir`, rewritten every 20 ticks) is the
fill-quality report: signed markout at 1m/10m/1h horizons per region
(belly/wing) and TTE bucket, over a rolling 7-day lookback. Each cell
carries `n` (markouts computed) and `n_attempted` (fills eligible) -- a
low `n / n_attempted` ratio means mid data was missing around those
fills' horizons (feed outage or long reprice ticks), not that fills were
absent. The mm_monitor page renders this with a coverage column.
Persistently negative belly markouts are the signal that model bias is
bleeding through the fair-value anchor -- the thing Stage B exists to
measure. The backing `mid_log` table in the state db is pruned to the
same 7-day window automatically at report cadence, so the db stays
size-bounded on a month-long run; snapshot the db file first if you want
full-history markout analysis later.

### Optional: mm_monitor dashboard over an SSH tunnel

The `app/pages/mm_monitor.py` Streamlit page must run on the VPS (it reads
the state db and control files locally), but should never be exposed
publicly -- bind it to loopback and reach it through an SSH tunnel instead.
There is no template file for this unit (it is optional); create it
directly:

```bash
/opt/mm/venv/bin/pip install streamlit   # not installed by the base steps

sudo tee /etc/systemd/system/mm-monitor.service <<'EOF'
[Unit]
Description=MM monitor dashboard (Streamlit, loopback only)
After=network-online.target

[Service]
User=debian
WorkingDirectory=/opt/mm/V2-BTC-Contract-Pricing
ExecStart=/opt/mm/venv/bin/streamlit run app/pages/mm_monitor.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now mm-monitor.service
```

Pick a free port for `--server.port` (8501 is Streamlit's default and may be
taken by another app on the box). Then, from the local machine:

```bash
ssh -L 8502:127.0.0.1:8502 <vps-host>
# keep that session open and browse http://localhost:8502
```

`--server.address 127.0.0.1` is the security boundary: the dashboard is
reachable only through the tunnel, no firewall rule needed.

## 3. Stopping / starting cleanly

```bash
# Graceful stop (SIGTERM -> the runner finishes its current tick, cancels
# all live paper orders, flushes CSVs, writes exit_reason=sigterm, exits 0
# -- systemd does NOT restart a clean exit 0):
sudo systemctl stop mm-paper.service

# Start (or restart) it:
sudo systemctl start mm-paper.service
sudo systemctl restart mm-paper.service
```

Do not `kill -9` / `systemctl kill` a running engine as your normal stop
method -- that is reserved for the acceptance test's fault-injection step
(section 5) and for genuine emergencies. `systemctl stop` is always
graceful and always the right tool for routine stops.

### What exit code 42 means

The runner intentionally exits **42** when a ladder finishes settling
(`ladder_settled`) or when it gives up waiting for an UNSETTLEABLE market
(`settlement_timeout`, after `--max-settlement-wait-h`, default 26h). This
is a *rollover signal*, not a fault: `RestartForceExitStatus=42` in
`mm-paper.service` makes systemd treat it the same as `Restart=on-failure`,
so the process restarts and (with `event_slug: "auto"`) picks up the next
`bitcoin-above` event automatically. `settlement_timeout` specifically also
triggers a page from `mm-alert` (an UNSETTLEABLE market needs a human to
check why BTC data coverage was missing at the settlement instant) --
unsettled positions themselves are not lost: they stay open in the
persisted state db and are retried by the next start's resume/catch-up.

Exit **1** (`feed_dead` / `tick_errors` / an early unhandled exception) is
an ordinary supervised restart. Exit **0** (`completed` / `stop_file` /
`sigterm` / `sigint`) means an intentional stop -- no restart.

### Known benign crash-loop: no future event published yet

After a rollover exit (42), `--event-slug auto` calls `resolve_next_event`,
which probes the Gamma API for the next `bitcoin-above-on-<date>` event up
to `auto_event_lead_days + 4` days out. If the venue has not published a
market for any of those dates yet (this happens -- new daily markets are
not always listed multiple days ahead), `resolve_next_event` raises
`SystemExit`, the runner exits with an early-failure reason (`exit_reason`
prefixed `error:`), and returns exit code 1. Systemd retries at
`RestartSec=60` until a suitable event appears on the venue, then the run
proceeds normally.

This is **expected behavior**, not a fault, and does not need operator
intervention -- it self-resolves once Polymarket lists the next market. The
`mm-alert` de-dupe window (6h per alert key) keeps this to at most one page
even if the retry loop runs for hours; the alert fires under the `CRASHED`
state check (PID file present-then-gone between retries can also surface
transiently as `STOPPED` with an `error:` exit_reason, which is not one of
the alert conditions by itself -- only sustained `CRASHED`/`STALLED` pages).

## 4. Log rotation

The runner's own `runner.log` (written when launched via
`run_control.start_engine()`'s log redirect; under systemd, prefer
`journalctl` and its own retention/vacuum settings instead) and the per-tick
CSVs under `temp/paper_run/<ts>/` are not rotated by the application (log
rotation beyond the M2-spam fix -- fewer duplicate tracebacks on reconnect
storms -- was explicitly deferred, plan Workstream 2 scope note). Example
`logrotate` config if you run the launcher path instead of pure systemd, or
just want size-bounded log files regardless:

```
# /etc/logrotate.d/mm-paper
/opt/mm/V2-BTC-Contract-Pricing/temp/paper_run/*/runner.log {
    weekly
    rotate 8
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

`copytruncate` is required here (not `create`) because the runner holds the
log file handle open for the life of the process and has no SIGHUP-triggered
reopen logic.

## 5. 72h acceptance test

Before trusting this for an unattended month, run the following once,
end-to-end, on the actual VPS (not the dev machine):

1. Start `mm-paper.service` against `event_slug: "auto"` so it picks up
   whatever event is currently closest to settling (pick a moment where
   settlement is within the 72h window so you actually observe a rollover).
2. Let it run undisturbed long enough to observe: at least one full ladder
   reach `settlement_instant_utc`, settle (`_all_settled_terminal` true),
   exit 42, and restart onto the next auto-resolved event.
3. **Forced kill -9**: pick a moment mid-run, `sudo systemctl kill -s SIGKILL
   mm-paper.service` (or `kill -9 <pid>` directly). Confirm:
   - systemd restarts it (`Restart=on-failure` covers SIGKILL same as any
     other abnormal exit).
   - The resume protocol fires (`mark_all_live_orders_unknown` ->
     `loop.restart()` -> `loop.settle(catch_up=True)`) -- check the log for
     "state db ... already existed; running resume protocol".
   - Post-restart inventory matches `fold(fills)` (the runner's own
     `summary.md` on the eventual clean exit reports `fold(fills) ==
     inventory`; the check compares both `q` AND `avg_cost` per market;
     you can also query the state db directly).
   - Exactly one alert fires for this event (mm-alert should catch the
     `CRASHED` state in the gap between the kill and the restart landing,
     assuming the 5-min timer polls during that window; a very fast restart
     may be missed entirely, which is fine -- it means nothing needed
     paging).
4. **Forced network cut**: block outbound network (e.g. `sudo iptables -A
   OUTPUT -p tcp --dport 443 -j DROP` for a few minutes, or physically pull
   the VPS's network if you have console access) for long enough to trip
   the feed watchdog (`--feed-dead-ticks`, default 40 ticks ~= 10 min at
   15s) and/or a `resolve_next_event`/data-fetch retry storm. Confirm:
   - The feed adapter is rebuilt once (`feed_restarts` increments in
     heartbeat.json), and if the outage persists past a second dead-tick
     trip, the runner exits `feed_dead` (code 1) and systemd restarts it
     cleanly once the network returns.
   - `mm-alert` pages once for the fault (state STALLED/CRASHED, or the
     feed-unhealthy-for-15min condition, depending on timing), not once per
     5-minute timer tick during the outage (dedupe window is 6h).
5. Confirm resource stability over the 72h: `ps -o rss= -p $(systemctl show
   -p MainPID --value mm-paper.service)` sampled a few times a day should
   stay roughly flat (WS1's journal caps target this -- unbounded growth
   would indicate a cap regression). Tick wall-clock time
   (`ticks.csv`'s `wall_s` column) should also stay flat, not trend upward.
6. Confirm `python core/data/data_fetcher.py` ran on schedule
   (`journalctl -u mm-datafetch`) and that `DATA/btc_intraday_1m.csv`'s
   mtime never falls behind `--btc-stale-max-s` (7200s) for long during
   normal operation (a brief staleness window right after a fetch failure
   is fine -- the runner's own staleness guard pulls quotes until fresh
   data lands; this should self-heal within one or two 30-min fetch
   cycles).
7. Confirm `markout_report.json` is regenerating (mtime advances every
   ~20 ticks) and, once fills have accumulated, that coverage
   (`n / n_attempted`) is high (>0.9) outside of known feed-outage
   windows -- persistently low coverage means the mid_log is gapping and
   the markout numbers should not be trusted for the belly-bias readout.

Record the outcome (settlement observed, kill-9 recovery, network-cut
recovery, RSS/tick-time trend, alert count per fault) before promoting this
to a real month-long unattended run.
