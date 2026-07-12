# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## Multi-expiry market making (concurrent ladders + in-process rollover)

Adds multi-expiry market making to the Stage-B paper runner: up to
`--max-expiries` (default 1) concurrent `bitcoin-above` expiry ladders quote
simultaneously, and a settled ladder rolls over IN-PROCESS onto the next
venue event instead of exiting the process -- eliminating the idle gaps and
single-ladder PnL ceiling of the serial design.

Architecture is "orchestrator of loops": the single-expiry
`PaperTradingLoop` (harness.py) is unchanged except two additive methods
(`resume_attach` -- market re-registration + fills replay FILTERED to the
loop's own markets + per-expiry Beuoy reload; and
`fold_matches_inventory(own_markets_only=True)`); the new
`market_maker/multi_runner.py` owns one LadderSlot per expiry (own loop, own
WS adapter, own SimClock) over one shared MMStateStore, vol gate, BTC data
provider and `SharedPricingEngine`. The engine shares one GARCH fit + one
set of calibrated jump params across per-expiry ladder caches (engine
`garch_cache` hook, backrunner pattern) and grants ONE reprice token per
tick, round-robin among due expiries, returned on a failed compute; fresh
slots are skipped (drain-and-discard) until their first-price grant lands,
so warmup of K ladders costs K ticks of one engine call each and
run_control's STALLED threshold math is unchanged. Sizing bankroll is
statically split (`bankroll / max_expiries` per ladder).

Rollover state machine: past-instant slots settle every tick (unconditional
of pricing), terminal + 30min grace (or per-ladder settlement timeout,
default 26h) tears the slot down (final settle, scoped per-market cancels,
ladder-state flush, adapter stop) and acquisition immediately probes
`shadow_runner.resolve_events_multi` (new: capped, exclusion set, intra-call
expiry dedup for padded/unpadded slugs, per-candidate SystemExit swallowed,
possibly-empty result, partial result on venue outage). Exit 42 now means
`no_quotable_events` in auto mode (fixed-slug mode keeps legacy
ladder_settled/settlement_timeout exits). Resume protocol on a shared db:
standalone store-wide settlement catch-up pass BEFORE replay (sidesteps the
settle(catch_up=True)-after-restart invariant), then per-slot filtered
resume_attach, then ONE venue reconcile; a recurring throttled catch-up pass
(1/60s while an orphaned non-slot market is past-instant non-terminal)
re-drives UNSETTLEABLE stragglers mid-run, with the BTCDataProvider shared
across all handlers.

Back-compat surfaces: heartbeat keeps every legacy top-level field with
aggregate semantics (feed_healthy=AND, bankroll_frozen=OR, counters=sums)
plus additive `n_expiries_active`/`ladders_settled_total`/
`ladder_settlement_timeouts`/`expiries`; current_run.json/run_meta.json keep
legacy singular fields pointing at the nearest expiry plus an additive
`events` list; pnl TOTAL row stays a single writer (global equity curve
unchanged); NO state-db schema or CSV changes (expiry attribution joins the
existing markets registry). pnl_report gains `expiry_by_market` per-market
stamping (also fixes the pre-existing cross-rollover mislabeling) and an
`expiry_key=None` all-expiries settlement breakdown; markout_report gains an
additive `by_expiry` rollup. The engine test seam moves from
paper_runner.CachedEngine to paper_runner._ENGINE_COMPUTE_FN (injected
compute callable); CachedEngine's jump loader is extracted to the shared
module-level `load_jump_params_for_engine`.

Dashboard (app/pages/mm_monitor.py + new app/mm_monitor_helpers.py):
Positions / new Open orders / Fills / Markout-by-expiry sections render one
tab per expiry (registry join is the primary strike/expiry source,
quotes.csv demoted to mark enrichment; per-tab event captions; single-expiry
runs render one tab); the global equity graph is untouched; the status row
shows per-expiry feed/state/fills badges. scripts/mm_alert_check.py gains
two absence-tolerant checks: `ladder_settlement_timeouts > 0` and sustained
`n_expiries_active == 0` while running.

Tests: 670 passing (was 619) -- new tests/test_mm_multi_runner.py (engine
sharing/stagger/token-return, skipped-slot semantics, fill isolation +
scoped fold, in-process rollover, settlement-timeout teardown + later
catch-up, orphan retry cadence, resume partition, tampered-inventory
discrepancy, resume_attach unit), tests/test_mm_paper_runner_multi.py
(2-expiry heartbeat/pointers, in-process rollover keeps process alive,
no_quotable_events=42, fixed-slug never touches the multi resolver,
one-reprice-per-tick at runner level), tests/test_mm_monitor_helpers.py,
plus appends to shadow_runner/pnl_report/alert_check test files; the two
legacy runner test files updated only for the intended engine-seam diff.

Files: market_maker/multi_runner.py (new), market_maker/paper_runner.py
(refactor), market_maker/harness.py, market_maker/shadow_runner.py,
market_maker/pnl_report.py, app/pages/mm_monitor.py,
app/mm_monitor_helpers.py (new), scripts/mm_alert_check.py, CLAUDE.md,
deploy/README.md, DOCS/guides/market-maker-deployment.md,
DOCS/concepts/market-making.md. market_maker/paper_run_config.json is
deliberately NOT changed -- flip `"max_expiries": 3` on the VPS only after
a burn-in through one in-process rollover at max_expiries=1.
