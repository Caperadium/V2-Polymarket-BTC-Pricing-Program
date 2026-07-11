# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## MM zero-fill spread recalibration + trade-print recording for arrival-decay calibration

Root-caused the market maker's zero fills (VPS quote journal: our spread 5-10x market touch, 2 fills in 2 days, half-spread ~5.7c/side at ATM vs 0.5c market half-touch) and applies the first recalibration wave plus the instrumentation needed to make the next wave data-driven:

- `market_maker/config.py`: `k_arrival` 1.0 -> 10.0 (the k=1 launch placeholder's arrival term was ~2.2c/side at ATM, the single largest spread component; 10.0 is an interim judgment value pending trade-print calibration). `near_resolution_pull_hours` 24 -> 6 (24h pulled quotes for the entire final day of a daily event -- 0-1 DTE, the highest-volume regime and the model's sweet spot, was never quoted).
- `market_maker/spread_builder.py`: `DEFAULT_CREDIBILITY_WIDEN_SCALE` 0.02 -> 0.01, `DEFAULT_WING_BASE_P` 0.01 -> 0.005. Post-recal ATM half-spread ~2.9c at 1.7 DTE (was ~5.7c).
- `market_maker/state_store.py`: new `trade_prints` table (+ indexes, `TradePrintRow`, `append_trade_prints`/`get_trade_prints`/`prune_trade_prints`) durably recording the drained per-tick WS aggressor prints.
- `market_maker/harness.py`: tick() persists each tick's `MarketState.last_prints` right after `append_mids` (read-only copy; fill sim unaffected).
- `market_maker/paper_runner.py`: prunes `trade_prints` on the quotes cadence/retention (`quotes_retention_s`).
- `scripts/mm_calibrate_k.py` (new): fits the Dalen arrival decay k from recorded prints -- lambda(delta_x) = A*exp(-k*delta_x) via backward asof-join of prints to mid_log mids, binned WLS on log rates. Breaks the "quote wide -> no fills -> nothing to calibrate from" loop since the print stream needs none of our fills. Validated on a synthetic db (recovers k=11.5 from true k=12).
- Tests: `test_b_near_resolution_pulls` re-anchored to the 6h window plus a new no-pull-at-12h companion; `test_resolve_next_event_skips_events_too_close_to_settlement` pins 24h explicitly (tests the mechanism, not the default); two geometry-tuned tests (`test_inventory_cap_goes_one_sided_then_pulls`, `test_last_liquidity_reaches_size_ladder_and_forces_depth_cap`) pin `k_arrival=1.0`; new trade_prints round-trip/prune tests. Full suite 619 passed.
- CLAUDE.md: commands, scripts tree, state_store table list, spread-recal note.
