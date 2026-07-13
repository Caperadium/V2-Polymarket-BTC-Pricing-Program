# Market-Maker Stage-B Deployment

Operator runbook for running `market_maker.paper_runner` unattended on a
headless VPS. The deployment kit itself (systemd unit templates + the alert
script) lives in `deploy/` and `scripts/mm_alert_check.py`; this page mirrors
`deploy/README.md` with links back into the wider docs.

For the module's internal architecture (components, control-file protocol,
resumable state, exit-code convention) see
[Architecture](../concepts/architecture.md) and the "Market-Maker Stage-B
Paper Runner" section of the repo's `CLAUDE.md`.

## What ships in `deploy/`

| File | Purpose |
|------|---------|
| `mm-paper.service` | The paper-trading engine unit. `Restart=on-failure` + `RestartForceExitStatus=42` (rollover), `RestartSec=60`, `TimeoutStopSec=900`, `KillMode=mixed`. |
| `mm-datafetch.service` / `.timer` | Runs `core/data/data_fetcher.py` every 30 minutes (`Persistent=true`). Nothing else refreshes BTC data on the VPS. |
| `mm-alert.service` / `.timer` | Runs `scripts/mm_alert_check.py` every 5 minutes. Fault alerts (CRASHED/STALLED, feed unhealthy >15min, stale BTC data, low disk, `settlement_timeout`) are de-duped 6h per condition; additionally one daily heartbeat message ("still alive" one-liner with state/tick/fills/disk) is sent at the first check at/after 08:00 UTC (`$MM_HEARTBEAT_HOUR_UTC` to change, `$MM_HEARTBEAT_DISABLE=1` to turn off), so webhook silence always means the alert pipeline itself is broken. |
| `mm-telegram.service` | Optional. Runs `scripts/mm_telegram_bot.py`, a stdlib-only read-only Telegram bot (long-polling `getUpdates`, no inbound endpoint) that answers operator slash commands with current engine metrics. Reuses the mm-alert webhook URL for credentials. |
| `README.md` | Full install steps + the 72h acceptance test (reproduced below). |

All unit files are templates: every `<EDIT>` placeholder (repo path, venv
python path, webhook URL) must be filled in before copying to
`/etc/systemd/system/`.

## Install

```bash
# 1. Clone + venv
sudo mkdir -p /opt/mm
sudo git clone <your-remote-url> /opt/mm/V2-BTC-Contract-Pricing
cd /opt/mm/V2-BTC-Contract-Pricing
python3 -m venv /opt/mm/venv
/opt/mm/venv/bin/pip install --upgrade pip
/opt/mm/venv/bin/pip install pandas numpy scipy arch hmmlearn xgboost \
    statsmodels websockets requests pytz streamlit

# 2. Prime BTC data once (the runner never fetches data itself)
/opt/mm/venv/bin/python core/data/data_fetcher.py

# 3. Edit + install the units (see deploy/README.md for the full listing)
sudo cp deploy/mm-paper.service deploy/mm-datafetch.service \
        deploy/mm-datafetch.timer deploy/mm-alert.service \
        deploy/mm-alert.timer /etc/systemd/system/
sudo systemctl daemon-reload

# 4. Alert webhook, via a drop-in (keeps the secret out of the tracked unit file)
sudo mkdir -p /etc/systemd/system/mm-alert.service.d
sudo tee /etc/systemd/system/mm-alert.service.d/webhook.conf <<'EOF'
[Service]
Environment=MM_ALERT_WEBHOOK=https://your-real-webhook-url
EOF
sudo systemctl daemon-reload

# 5. Enable + start: timers first, then the engine
sudo systemctl enable --now mm-datafetch.timer
sudo systemctl enable --now mm-alert.timer
sudo systemctl enable --now mm-paper.service
```

`market_maker/paper_run_config.json` (checked in) drives the engine:
`event_slug: "auto"` acquires the next `bitcoin-above` event(s) and rolls
over IN-PROCESS when one settles (no restart), `state_db` points at a
persistent SQLite file so a restart resumes instead of starting flat,
`minutes: 0` runs indefinitely. Add `"max_expiries": N` (default 1) to
quote up to N concurrent expiry ladders -- each gets its own WS connection
and quoting loop over one shared state db and one shared pricing engine
(one GARCH fit; at most one engine reprice per tick, staggered
round-robin), and the sizing bankroll is statically split as
`bankroll / max_expiries` per ladder. The existing state db is forward-
compatible as-is (no schema change).

## Checking status

```bash
systemctl status mm-paper.service
journalctl -u mm-paper -f
journalctl -u mm-alert --since today

cd /opt/mm/V2-BTC-Contract-Pricing
/opt/mm/venv/bin/python -c "
from market_maker import run_control
s = run_control.engine_status()
print(s.state, s.detail)
print(s.heartbeat)
"
```

`engine_status()` states: `RUNNING`, `STARTING` (within 120s of launch, no
heartbeat yet -- covers auto-event-resolution retries + WS warmup),
`STALLED` (heartbeat gone stale), `STOPPED` (no PID file -- check
`current_run.json`'s `exit_reason`), `CRASHED` (PID file present but that
PID is dead).

`heartbeat.json` fields worth watching: `feed_healthy` (AND over active
ladders' adapters), `btc_data_age_s` (staleness of
`DATA/btc_intraday_1m.csv`), `feed_restarts`, `noarb_repairs` (ladders
that arrived at the LadderHedger violating no-arb and were PAV-repaired
-- the true violation count, summed over all ladders this run including
torn-down ones; a climbing rate means skew/anchor/spread terms are
generating inconsistent ladders), `pulled_ticks`, and the multi-expiry
additions `n_expiries_active`, `ladders_settled_total`,
`ladder_settlement_timeouts` and the per-expiry `expiries` dict
(per-ladder state/feed/fills/frozen breakdown, incl. per-ladder
`noarb_repairs`). Note: the legacy `noarb_violations` field does NOT
count arb violations -- it counts warm-up ticks before a slot's first
checked ladder (kept under its old name for consumer compat).

### mm_monitor dashboard over an SSH tunnel (optional)

The `app/pages/mm_monitor.py` Streamlit page has to run on the VPS (it
reads the state db and control files locally) but should stay off the
public internet: run it bound to loopback under a small systemd unit
(`streamlit run app/pages/mm_monitor.py --server.address 127.0.0.1
--server.port 8502 --server.headless true`; `pip install streamlit plotly` into
the venv first -- the base install skips both) and reach it with a local
port forward:

```bash
ssh -L 8502:127.0.0.1:8502 <vps-host>   # keep open, browse http://localhost:8502
```

The loopback bind is the security boundary -- no firewall rule needed.
Pick a port that is free on the box (8501 is Streamlit's default and may
be taken). Full unit file in `deploy/README.md` section 2.

### Telegram metrics bot (optional)

`deploy/mm-telegram.service` runs `scripts/mm_telegram_bot.py`: message the
alert bot on Telegram and it answers with current metrics instead of you
waiting for the daily heartbeat. Commands:

| Command | Answer |
|---------|--------|
| `/status` | Engine state (`engine_status()`), tick, feed health, per-expiry lines |
| `/bankroll` | Initial bankroll (run_meta.json) + current equity from the latest pnl TOTAL row |
| `/pnl` | Realized / unrealized (mid + consensus) / settlement breakdown |
| `/fills` | Fill counts by liquidity (maker/taker/settlement), last-24h count, last fill |
| `/inventory` | Open positions (q != 0) with expiry/strike |
| `/quotes` | Latest resting quote per market (bid/ask/spread/sizes, age) |
| `/markout` | `markout_report.json` by-region rollup |
| `/help` | Command list |

Credentials come from the same `MM_ALERT_WEBHOOK` Telegram URL mm-alert uses
(token + chat_id parsed from it; `MM_TELEGRAM_TOKEN`/`MM_TELEGRAM_CHAT_ID`
override). The chat_id is a hard allowlist -- the bot never answers any
other chat. It is read-only by construction (state db opened `mode=ro`) and
persists its getUpdates offset to `<control-dir>/telegram_bot_state.json` so
a restart does not replay old commands. Run only one instance per bot token
(Telegram rejects a second concurrent `getUpdates` consumer with a 409).

## Stopping / starting cleanly

```bash
sudo systemctl stop mm-paper.service      # graceful: SIGTERM, exit 0, no restart
sudo systemctl start mm-paper.service
sudo systemctl restart mm-paper.service
```

### Exit code 42

In auto mode, a settled (or settlement-timed-out) ladder is torn down
IN-PROCESS and replaced by the next acquired event -- the process does not
exit for a rollover anymore. Exit **42** now means `no_quotable_events`
(zero active ladders and the acquisition probe found nothing);
`RestartForceExitStatus=42` makes systemd retry every `RestartSec=60`
until the venue lists a suitable event -- expected, not a fault. In
fixed-slug mode the legacy behavior is unchanged: `ladder_settled` /
`settlement_timeout` -> 42. Exit **1** (`feed_dead` / `tick_errors` /
early failure) is an ordinary supervised restart. Exit **0** (`completed` /
`stop_file` / `sigterm` / `sigint`) means an intentional stop -- no restart.

### Known benign retry loop

If the venue has not yet published `bitcoin-above` markets far enough out,
the runner exits 42 (`no_quotable_events`) and systemd retries every
`RestartSec=60` until one appears; while at least one ladder is alive, an
empty acquisition probe just backs off (`acquire_retry_s`, default 600s)
and the remaining ladders keep quoting. Expected and self-resolving; the
alert de-dupe window (6h per condition) keeps it to at most one page.

## Log rotation

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

`copytruncate` is required -- the runner holds the log file handle open for
the life of the process.

## 72h acceptance test

Run once on the actual VPS before trusting an unattended month-long run:

1. Start against `event_slug: "auto"` close enough to a settlement that a
   rollover happens within the 72h window.
2. Observe a full ladder settle and roll over IN-PROCESS onto the next
   event (log shows "tearing down ladder ..." then "ladder acquired: ...";
   heartbeat `ladders_settled_total` increments; the PID does not change;
   with `max_expiries` > 1 the other ladders keep quoting throughout).
3. **Forced `kill -9`** mid-run: confirm systemd restarts it, the resume
   protocol fires (standalone settlement catch-up pass -> per-slot filtered
   `resume_attach` -> one venue reconcile), per-ladder inventory matches
   `fold(own fills)` afterward, and exactly one alert fires for the fault.
4. **Forced network cut** long enough to trip the feed watchdog
   (`--feed-dead-ticks`, default ~10 min): confirm the adapter rebuilds
   once, `feed_dead` exits and restarts cleanly if the outage persists, and
   `mm-alert` pages once (not once per 5-minute timer tick).
5. Confirm RSS and per-tick wall-clock time (`ticks.csv`'s `wall_s`) stay
   flat over the 72h window (validates the journal-cap bounding in
   `market_maker/harness.py`).
6. Confirm `mm-datafetch.timer` ran on schedule and `DATA/btc_intraday_1m.csv`
   never falls far behind `--btc-stale-max-s` (7200s) during normal
   operation.

See `deploy/README.md` for the fully detailed version of this procedure.
