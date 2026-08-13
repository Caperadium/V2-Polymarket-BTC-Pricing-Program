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

As of the per-region Beuoy bankroll credibility change (package B2,
2026-07-15), the state db's `bankrolls` table gains a `region` column. A
pre-existing db without it is migrated automatically and safely on the next
start via a guarded `ALTER TABLE ... ADD COLUMN` -- no manual step. On that
first restart, the `belly` region inherits the ladder's existing (legacy)
bankroll weights and `wing` resets to 50/50 parity; both regions persist
independently from then on. See [Market Making](../concepts/market-making.md#45-per-region-credibility-belly-vs-wing)
for the reasoning.

### 2026-08-08 wing-bleed fix wave: knobs and kill switches

The wave ships five new `MMConfig` knobs, all **default-ON** after a plain
pull + restart: `paper_run_config.json` keys map to argparse dests only and
the runner constructs `MMConfig` bare, so new config fields activate at
their defaults without any config-file change. Each has a kill switch
(config-code edit, plus a CLI flag for the epoch):

| Knob (default) | What it does | Kill switch |
|------|------|------|
| `wing_pricer_weight_pin` (0.5) | Pins the wing region's pricer weight in the fair-value consensus and skips the wing Bayes update (self-confirmation loop; see Market Making 4.6) | `-1.0` restores legacy wing Bayes |
| `markout_slow_horizon_s` (21600.0) | Second sizing markout lookup at 6 h, a one-directional `min()` haircut on the Kelly net edge; slow-measured-toxic also suppresses the exploration carve-out | `0.0` = mid channel only |
| `markout_epoch_utc` ("2026-08-11T03:21:00+00:00" since 2026-08-13; was the 07-27 restart) | Fills before this UTC instant are hidden from the EPOCH markout view, which feeds belly SIZING and (since 2026-08-13) spread term 7's widening on all markets; wing sizing and the monitor keep the full 28d window. Bumped to the skew-fix deploy because the 08-10 incident's own fire-sale fills cap-bound term 7 at 0.12/side and stalled the book (0 fills, 2 days) | `""` disables; CLI `--markout-epoch` overrides the field (an explicit `--markout-epoch ""` also disables). Bump SPARINGLY: each bump resets the belly slow-channel backstop (6h + 20 fills to re-arm) |
| `sizing_region_basis` ("mid") | Classifies the sizing/exploration-gate region from the live book mid (matching the markout report's fill tagging) instead of the consensus p | see the exact-legacy note below |
| `sizing_region_hysteresis_p` (0.02) | Latches the sizing region per market; it flips only when the classifying probability clears the belly-band edge by this margin | `0.0` = raw region every tick (also a valid standalone switch if a stale held region misbehaves) |

EXACT legacy sizing-region behavior requires BOTH
`sizing_region_basis = "consensus"` AND `sizing_region_hysteresis_p = 0.0`
-- the basis change alone yields consensus-with-latch, a third behavior
that has never run in production. Each switch is independent, so item-level
reverts do not require unbundling the deploy.

Operator rule for the epoch: **bump `markout_epoch_utc` (or pass
`--markout-epoch`) at any deploy that materially changes quoting
behavior**, so belly sizing draws its evidence only from the current
quoting regime; the runner logs the active epoch at startup and notes when
it has aged past the 28d lookback (inert until bumped).

Before rolling this wave -- or any quoting-behavior change -- to the VPS,
take a **pre-deploy baseline snapshot** for post-deploy attribution:
(a) `markout_report.json`, (b) each market's current QuoteMode and bid/ask
sizes from the `quotes` table, (c) the `bankrolls` table's wing rows.
Capture the same three after the deploy.

### 2026-08-10 skew-fix wave: knobs and kill switches

Fixes the 2026-08-10 incident (-7.4 realized, 64k Aug-11 ladder): an
unbounded inventory-skew term pinned the reservation price at the p-clamp
floor and fire-sold a winning position ~55c under fair. See
[Market Making 6.3](../concepts/market-making.md#63-the-skew-displacement-cap-2026-08-10)
for the full mechanism and forensics. Three new `MMConfig` knobs, same
default-ON deployment story as the wave above: `paper_run_config.json`
maps to argparse dests only and `paper_runner` constructs `MMConfig` bare,
so these fields activate at their production defaults on a plain
pull + restart, no config-file change needed.

| Knob (default) | What it does | Kill switch |
|------|------|------|
| `skew_x_cap` (1.0) | Caps the AS/GLFT reservation displacement `skew_x` at this many x-units in `quote_engine.make_quote` (Market Making 6.3) | `<= 0` restores legacy unbounded `skew_x` exactly |
| `skew_q_headroom_mult` (1.5) | Sizing-side entry cap (`robustness_sizing` Stage 6b, Market Making 8.2): add-side shares capped at `skew_q_headroom_mult * skew_x_cap / unit_skew_x` | raise to loosen; the stage is already inert whenever `skew_x_cap <= 0` |
| `bankroll_update_temper` (0.1) | Tempers each region's per-tick Bayes factor (`factors ** t`) before the bankroll weight update in `fair_value_anchor.compute_fair_value` (Market Making 4.7) | `1.0` restores legacy (untempered) Bayes speed |

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

Since 2026-08-08 the per-run output directory carries a second markout
artifact next to `markout_report.json`: `markout_report_sizing.json`, the
belly epoch-filtered SIZING view, written on the same cadence. The file
holds what belly-region sizing actually reads -- it reflects the provider's
fallback chain (a failed sizing build degrades to the previous view, else
the protective full report), not the raw per-cadence build. The mm_monitor
page captions the markout section with the active epoch, renders the
sizing view as its own "belly sizing view" expander, and flags the wing
`bankrolls` rows as PINNED: a flat 0.5 pricer weight with a frozen
`update_count` on the wing rows is the wing pricer weight pin working, not
a fault.

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
| `/bankroll` | Initial bankroll (run_meta.json) + current equity from the latest pnl TOTAL row + rebates accrued (est, maker-rebate accounting layer -- off-equity estimate, not included in the equity figure) |
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

## Wing-bleed fix wave: post-deploy watch (2026-08-08)

Watch 3-5 days after deploying the wave, against the pre-deploy baseline
snapshot from the install section:

- **Wing exploration-fill detector.** Flag ANY new maker fill at market mid
  < 0.30 on a market whose SIZING cell was not trusted-measured at quote
  time (`n_attempted < markout_min_n` in `markout_report_sizing.json` for
  the fill's region/tte cell), regardless of size -- the open exploration
  gate produced floor-sized, Kelly-sized and tapered-floor fills alike, so
  a size-match trigger misses cases. Report the fill size as a multiple of
  `0.55/price` (the untapered floor unit) as a diagnostic column, not as
  the trigger. Check-time caveat: the sizing report is overwritten each
  cadence, so evaluate against the nearest retained snapshot if one is
  kept, else the current file -- `n_attempted` only grows within the
  window, so "flagged now" implies "under-attempted then" (no false
  positives; a cell that crossed `markout_min_n` between fill and check
  hides its earlier fills, the accepted miss direction). Expect ~0 wing YES
  maker fills/day at mid < 0.2 from the exploration path.
- **Settled PnL, two bands.** Fills at mid < 0.20 should settle to a mean
  better than -1c/share (they ran -10 to -13c pre-fix); track mid 0.20-0.30
  as its own line -- that band is deliberately REOPENED for bounded (0.33x)
  exploration by the epoch + mid-basis changes, protected day-one by term-7
  width alone until the slow haircut arms (fills aging past 6 h, roughly
  1-3 weeks).
- **Belly alive.** Sustained TWO_SIDED resting orders on near-money
  markets; no fleet-wide zero-quote state.
- **Boundary churn.** Orders + cancels per 10 minutes on markets with book
  mid in [0.17, 0.23] (consensus band as a secondary cut): no square wave
  vs baseline -- the hysteresis latch's job.
- **Pin persistence.** After a restart, the state db's `bankrolls` wing
  rows show pricer == 0.5, flat -- the pin overwrites the stale stored
  weights on the first clean tick.
- `stranded_markets` stays 0; no STALLED / CRASHED alerts.

## Skew-fix wave: post-deploy watch (2026-08-10)

Watch 2-4 days after deploying the wave, against a pre-deploy baseline
snapshot of the `quotes` and `bankrolls` tables:

- **Displacement invariant.** Journaled `|skew_x| <= skew_x_cap` on every
  `quotes` row -- with the cap on, this must hold exactly (Market Making
  6.3's re-derivation step makes it so). A violation means the cap is not
  actually wired for that call site.
- **Consensus-vs-skew divergence.** Reconstruct `x_fair = r_x - skew_x` per
  quoted row and compare it against `logit(mid)` from `mid_log`. Divergence
  beyond roughly 1.5 x-units flags a **consensus** fault (fair-value
  anchor, staleness, a bad snapshot) -- not a skew fault; the cap only
  bounds displacement, it says nothing about whether `x_fair` itself is
  reasonable.
- **No fire-sale fills.** No fill should land more than about 2
  half-spreads through the prevailing `mid_log` mid. This is the direct
  regression check for the incident pattern (asks 0.14-0.23 into a 0.71
  market).
- **Belly bankroll flip rate.** No belly region 0.02 <-> 0.98 full flip
  inside less than 6 hours. This is a **rate** criterion, not a ceiling --
  the 0.98/0.02 corner is still the attractor (Market Making 4.7); the
  temper only slows the approach.
- **Position bound.** Per-market `|q|` stays under roughly 1.5x the
  clamp-bind quantity (`skew_q_headroom_mult`'s own bound) -- a
  sustained excess means Stage 6b (Market Making 8.2) is not binding where
  expected, e.g. `unit_skew_x` not threaded for that market.
