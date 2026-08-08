# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## fix(mm): wing-bleed fix wave (4 items) -- 2026-08-08

Closes the post-2026-07-27 slow bleed (-4.91 realized over 11.5 days, 15/15
settled positions negative). VPS forensics traced every wing YES bleed fill
to exploration-floor/Kelly bids sustained by a region-basis mismatch: the
sizing/exploration-gate lookup classified markets by CONSENSUS p (pricer-rich
-> "belly") while the markout report tags fills by book MID ("wing"), so the
gate checked a cell the fills never fed and could never close. Plan iterated
through 7 adversarial review rounds (temp/mm_wing_bleed_fix_plan.md, rev 7 +
final nits); final diff reviewed ALL CLEAR; 909 tests pass (2 known
pre-existing test_mm_run_control flakes, verified on clean HEAD).

1. Item 4 (the day-one brake) -- sizing-region basis alignment + hysteresis
   (harness.py): sizing lookups now classify region from the live book mid
   via _market_mid over the threaded market_states (policy-identical to
   paper_fill_sim._mid modulo a NaN guard), consensus only as the empty-book
   fallback, with a per-market hysteresis latch
   (MMConfig.sizing_region_hysteresis_p=0.02) to prevent boundary flapping.
   Kill switch: sizing_region_basis="consensus" (+ hysteresis 0.0 for exact
   legacy). Term-7 widening and credibility keep the consensus basis
   (enumerated in spread_builder's new "Deliberate basis inconsistencies"
   docstring section).
2. Fix 1 -- wing pricer weight PIN (fair_value_anchor.py, config.py):
   wing-region Bayes had re-awarded the pricer 0.978 weight via a
   self-confirmation loop while wing YES fills settled worthless daily. Wing
   pricer weight is now pinned (wing_pricer_weight_pin=0.5, clamped into
   [floor, 1-floor]; -1 disables), wing Bayes updates skipped, pinned state
   persisted on every non-fallback return path. Belly untouched.
3. Fix 2 -- slow mid-horizon sizing haircut (pnl_report.py,
   robustness_sizing.py, harness.py): markout report horizons extended to
   (60, 600, 3600, 21600, 86400) -- existing windows byte-identical,
   persisted markouts stay valid; sizing does a second markout_stats lookup
   at markout_slow_horizon_s (21600; 86400 is diagnostics-only) used as a
   strictly one-directional min() haircut on the Kelly net edge (never
   raises m, never sets sigma2). One deliberate gate change: the W4
   exploration carve-out is suppressed when the slow channel is
   measured-toxic (kills the 28-day relapse cycle; cannot deadlock).
4. Fix 3 -- belly-scoped sizing markout epoch (pnl_report.py,
   paper_runner.py, multi_runner.py, harness.py): markout_report gains
   keyword-only epoch_ts (max(lookback, epoch) cutoff); the runner builds
   TWO reports per cadence -- full (term 7/monitor/telegram, protective 28d
   window) and epoch-filtered sizing view (markout_epoch_utc =
   2026-07-27 restart; --markout-epoch CLI override) -- served via a new
   sizing_markout_provider; the harness routes belly-region sizing to the
   epoch view and wing-region sizing to the full view (wing 600s cells are
   currently measured-toxic and protective). Reopens the belly, which was
   Kelly-clamped until ~Aug 23 by 289 pre-restart burst fills dominating
   the 28d window. New artifact markout_report_sizing.json; mm_monitor
   shows the epoch, a belly-sizing-view table, and a wing-PINNED caption.

Files: market_maker/{config,fair_value_anchor,robustness_sizing,harness,
multi_runner,paper_runner,pnl_report,spread_builder}.py,
app/pages/mm_monitor.py, tests (6 files, ~40 new tests incl. a forensics
regression: consensus 0.21/mid 0.13 + measured-toxic wing cell -> no bid),
CLAUDE.md, DOCS/concepts/market-making.md,
DOCS/guides/market-maker-deployment.md.
Deploy notes: take the pre-deploy baseline snapshot (markout_report.json,
quotes modes/sizes, bankrolls wing rows) before rolling to the VPS;
acceptance metrics + kill switches in temp/mm_wing_bleed_fix_plan.md.
