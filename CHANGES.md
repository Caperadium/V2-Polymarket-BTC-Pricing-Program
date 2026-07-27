# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## fix(mm): 3-part bleed fix -- widen cap, markout memory, mid-velocity pull

Diagnosis (VPS state-db, 2026-07-26): -39.78 realized over 283 maker fills,
only -4.70 at settlement -- the bleed is stale-quote pick-off during BTC
bursts (post-07-17 ask fills landed 10.7c below / bid fills 6.7c above the
prevailing mid; measured markout -8 to -16c flat across horizons; BUY_NO
-17.2c/share vs BUY_YES +5.5c). Three defenses existed but were
under-calibrated; fixed per the 3-round-reviewed plan temp/mm_bleed_fix_plan.md:

1. Term-7 widening cap: `markout_widen_cap` 0.05 -> 0.12 (the 5c cap bound
   everywhere against 9-16c measured toxicity). Both PAV characterization
   tests re-verified at 0.12, no assertion relaxing needed. The proposed
   separate widen-min-n was REJECTED in review (would up-size the toxic
   side in the n=[10,20) window).
2. Markout memory: new `fill_markouts` state-store table persists each
   fill's resolved markout once (auto-migrating, INSERT OR IGNORE, pruned
   at the lookback); `MARKOUT_LOOKBACK_S` 7d -> 28d with `mid_log`
   retention decoupled at a new `MID_LOG_RETENTION_S` (7d, disk unchanged)
   -- kills the weekly re-arm cycle where a measured-toxic verdict expired
   with the window and full-size quoting resumed. `PaperFill` gains an
   optional `id`; `markout_report` gains inert-default `persisted`/
   `persist_cb` params; n_attempted now counts only fills that did or
   still can yield a measurement (old unresolved fills count in neither n
   nor n_attempted). Plus sizing stage 5b `unmeasured_size_mult` (0.33):
   legs of unmeasured cells (mk_n_attempted < markout_min_n) are throttled
   to 1/3 size -- fill count (the learning signal) accrues at nearly the
   same rate but per-cell tuition drops ~3x; reduce-side legs exempt,
   venue-min floor-back prevents a learning deadlock.
3. Ladder mid-velocity pull (risk rule h, `RiskTrigger.MID_VELOCITY`): the
   harness tracks two-sided mids per market over `mid_move_window_s`
   (120s); a ladder-wide max move > `mid_move_pull_p` (0.04) pulls quotes
   (reduce-only when positioned, same signed-q basis as rules (c)/(f)).
   Covers the live-burst blind spot: the vol gate's CSV refreshes every 30
   min and cannot see a burst; the 60s latch holds the pull through the
   trend continuation where the repeated cap-sized -1.67 losses occurred.

Files: market_maker/{config,contracts,state_store,pnl_report,paper_runner,
robustness_sizing,risk_controller,harness}.py; tests updated/added in
test_mm_{harness_ws1,pnl_report,state_store,robustness_sizing,
risk_controller}.py (4 named + 3 grep-found existing tests revised per the
plan's authorization); docs updated in CLAUDE.md and
DOCS/concepts/market-making.md.
