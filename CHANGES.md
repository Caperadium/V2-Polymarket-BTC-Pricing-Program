# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

- Add a belly-widening spread term (term 6) to `market_maker/spread_builder.py`
  and tighten the confidence-tier day boundaries in `market_maker/config.py`,
  aligning the market-maker with the pricer's suitability envelope
  (`temp/suitability.md`): the belly (softest region, +4.8c bias at 1-2d
  growing to +8.6c at 5-7d) now gets a flat + slope-past-2d additive
  half-spread (`belly_widen_base_p`/`belly_widen_slope_p_per_day`/
  `belly_widen_free_days`, harness passes `tte_days`), and
  `tier_full_max_days`/`tier_degraded_max_days` move 14/28 -> 7/14 (no
  backtest coverage past 7 DTE). The tier change's only behavioral effect is
  `wing_widen_scale`: wing spreads now widen 1.0x -> 1.5x in the 7-14d band
  and 1.5x -> 2.0x in the 14-28d band; sizing and quote-engine logic are
  unaffected.
- Fix a pre-existing live bug where `pnl_report.fill_cash` and
  `state_store.fold_fills_to_inventory` wrongly complemented MAKER/TAKER
  BUY_NO fill prices to `1 - price`; stored paper-fill prices are YES-scale
  for both sides (matching `inventory_manager`, which was already correct),
  so the complement produced a phantom -0.20/share on every open BUY_NO
  position. Both now use the raw stored price, and
  `harness.fold_matches_inventory` now also compares `avg_cost` (not just
  `q`) so this class of bug cannot regress silently. Resuming a pre-fix
  `--state-db` that has an open BUY_NO position will show a one-time step
  in the mm_monitor equity series at the deploy boundary (old snapshots
  keep the phantom -0.20/share, new ones do not) -- start the VPS
  acceptance run on a fresh state-db.
- Add a durable `mid_log` table (`market_maker/state_store.py`) that the
  harness appends per-market YES mids to every tick, and a pure
  `pnl_report.markout_report` function that measures fill markout (60s/600s/
  3600s horizons, 600s join window) split by region (belly/wing/unknown),
  TTE bucket (0-1d/1-2d/2-4d/4d+), settling whether the pricer's belly bias
  bleeds through the Beuoy fair-value anchor into realized fill quality.
  `paper_runner.py` writes `<out_dir>/markout_report.json` every 20 ticks
  and `app/pages/mm_monitor.py` renders it read-only.
- Fix a round of review findings on the markout report and its supporting
  plumbing: `pnl_report.markout_report`'s per-horizon lookup windows are
  now disjoint by construction (each horizon capped at the next horizon's
  start; `state_store.mid_at_or_after`'s upper bound flipped to exclusive),
  fixing collapsed cells when horizons sit closer together than the join
  window; cells now report `n_attempted` (eligible fills looked up)
  alongside `n` (successful hits) so `mm_monitor` can show real coverage,
  not just silently-dropped misses; the report is bounded to a rolling 7-day
  lookback (`MARKOUT_LOOKBACK_S`) and `paper_runner.py` prunes `mid_log` to
  the same window (`state_store.prune_mid_log`) right after each report
  write, keeping the table's growth bounded on a persistent `--state-db`;
  `paper_runner.py` now does a single guarded `get_fills()` shared by the
  pnl-snapshot and markout blocks instead of two independent fetches;
  `config.in_belly_band` is a shared predicate for belly/wing membership,
  used by both `spread_builder` and `pnl_report` region tagging; and
  `pnl_report.fill_cash` drops its unused `liquidity` parameter, with
  aggregation logic (`n`/`mk_total`/`mk_avg`) deduplicated into one
  `_summarize` helper shared by the cells loop and the by-region rollup.
- Harden the markout report against four residual review findings:
  `markout_report` de-dups the horizons sequence (a duplicated horizon
  produced a zero-width lookup window and a permanent 0-coverage cell) and
  isolates per-fill TTE derivation (a malformed registry `expiry_key` now
  degrades that fill to the "unknown" bucket instead of aborting the whole
  report); `state_store` adds a plain `mid_log(ts)` index so
  `prune_mid_log`'s ts-only DELETE stops full-scanning the table; the
  `spread_builder` module docstring now correctly lists terms 1 AND 3 as
  audit-only (term 1, arrival markup, was never added to the widening).
